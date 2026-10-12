// splice_count: the SPLICE-H offline count and the pruned grid (docs/SPLICE_DESIGN.md section 4).
// Counts only, never timings of lookups: E[D] is in milli-D from exact plan counts priced by the shared-L2
// simulation. Writes results/splice_h/<ds>.json and the per-dataset args files GRE's build replays.
//   splice_count --dataset fb --data DIR [--out results/splice_h] [--threads 16] [--grid pruned|cell]
//                [--args "k=v ..."] [--verify] [--parity] [--hash] [--perturb]
//   splice_count --samples DIR --hash-samples [--args "k=v ..."]     (one "hash <sample> <layout_hash>" line each)
#include "splice/build.hpp"
#include <cinttypes>
#include <cmath>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <sys/stat.h>

using namespace splice;
namespace fs = std::filesystem;

namespace {

double secs(long long ns) { return double(ns) / 1e9; }
std::string hex(std::uint64_t h) { char b[24]; std::snprintf(b, sizeof b, "0x%016" PRIx64, h); return b; }
std::string q(const std::string& s) { return "\"" + s + "\""; }
std::string fmt_s(double s) { char b[32]; std::snprintf(b, sizeof b, "%.3f", s); return b; }

// Read-only mapping of a SOSD file (u64 count, then keys).
struct Mapped {
    void* base = nullptr; std::size_t bytes = 0; const std::uint64_t* keys = nullptr; std::size_t n = 0;
    std::vector<std::uint64_t> own;        // perturbed copy, when used
    Mapped() = default; Mapped(const Mapped&) = delete; Mapped& operator=(const Mapped&) = delete;
    ~Mapped() { if (base) ::munmap(base, bytes); }
};
bool needs_sorted(const std::string& ds) {
    for (const char* s : {"covid", "genome", "history", "libio", "planet", "stack", "wise"}) if (ds == s) return true;
    return false;
}
// First index i with keys[i] <= keys[i-1], or 0 when strictly increasing.
std::size_t first_descent(const std::uint64_t* k, std::size_t n, unsigned threads) {
    const std::size_t ch = 1u << 22, nc = (n + ch - 1) / ch;
    std::vector<std::size_t> bad(nc, 0);
    par_for(nc, threads, [&](std::size_t c) {
        for (std::size_t i = std::max<std::size_t>(1, c * ch); i < std::min(n, (c + 1) * ch); ++i) if (k[i] <= k[i - 1]) { bad[c] = i; return; }
    });
    for (auto b : bad) if (b) return b;
    return 0;
}
void map_file(Mapped& m, const std::string& path, const std::string& ds, unsigned threads) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st{}; ::fstat(fd, &st);
    m.bytes = std::size_t(st.st_size);
    if (m.bytes < 8) { ::close(fd); throw std::runtime_error(path + ": too short"); }
    m.base = ::mmap(nullptr, m.bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m.base == MAP_FAILED) { m.base = nullptr; throw std::runtime_error("mmap failed: " + path); }
    const auto* w = static_cast<const std::uint64_t*>(m.base);
    m.n = std::size_t(w[0]); m.keys = w + 1;
    if (8 + 8 * m.n != m.bytes) throw std::runtime_error(path + ": size does not match the SOSD header");
    if (const std::size_t d = first_descent(m.keys, m.n, threads))
        throw std::runtime_error(path + " is not strictly increasing at " + std::to_string(d) + (ds.empty() ? "" : "; use " + ds + ".sorted"));
}

struct CellKey {
    std::uint32_t k1; unsigned el; std::uint32_t cbar; unsigned compact;
    bool operator<(const CellKey& o) const { return std::tie(k1, el, cbar, compact) < std::tie(o.k1, o.el, o.cbar, o.compact); }
    bool operator==(const CellKey& o) const { return k1 == o.k1 && el == o.el && cbar == o.cbar && compact == o.compact; }
};
// Everything evaluated for one fit: candidate caches by alpha and by (entry, cbar).
struct FitBundle {
    Fit fit; long long wall_ns = 0;
    std::map<unsigned, DirectCache> direct;
    std::map<std::pair<unsigned, std::uint32_t>, FencedCache> fenced;
};
SpliceParams cell_params(const SpliceParams& base, const Fit& f, const CellKey& c, std::uint32_t cap) {
    SpliceParams p = base;
    p.k1 = c.k1; p.eps = f.eps; p.tier1 = f.tier1; p.entry = std::uint8_t(c.el); p.cbar = c.cbar; p.compact = std::uint8_t(c.compact);
    p.cap_tenths = cap; p.exc = std::uint32_t(base.exc);
    return p;
}
Cell bundle_cell(const FitBundle& b, const SpliceParams& p) {
    std::vector<const DirectCache*> dc;
    for (auto& [qa, d] : b.direct) dc.push_back(&d);
    const FencedCache* fc = nullptr;
    auto it = b.fenced.find({p.entry, p.cbar});
    if (it != b.fenced.end()) fc = &it->second;
    return make_cell(b.fit, p, dc, fc);
}
void ensure_caches(FitBundle& b, const SpliceParams& p, unsigned threads) {
    if (p.mode != MODE_FENCED) for (unsigned qa : p.alphas()) if (!b.direct.count(qa)) b.direct.emplace(qa, eval_direct(b.fit, qa, threads));
    if (p.mode != MODE_DIRECT) {
        auto it = b.fenced.find({p.entry, p.cbar});
        if (it == b.fenced.end() || !(it->second.variants >> p.compact & 1)) b.fenced[{p.entry, p.cbar}] = eval_fenced(b.fit, p.entry, p.cbar, 3, threads);
    }
}

// Selection summary over the chosen candidates (stage 1: segment-local pages, default residency).
std::string selection_json(const Cell& c, const Selection& s) {
    const Fit& f = *c.fit; const std::uint64_t n = f.n;
    std::uint64_t dk = 0, fk = 0, depth[4] = {}, ties = 0, ck = 0, dep = 0, par = 0, pg = 0, scan = 0;
    for (std::size_t j = 0; j < f.K(); ++j) {
        const Sum& u = c.sum(j, s.choice[j]); const std::uint64_t nj = f.nseg_keys(j);
        if (c.desc[s.choice[j]].fenced) { fk += nj; for (int d = 0; d < 4; ++d) depth[d] += u.depth[d]; ties += u.ties; ck += u.compact_keys; }
        else { dk += nj; scan += u.scan; }
        dep += u.dep; par += u.par; pg += u.pgA + u.pgB;
    }
    std::ostringstream o;
    o << "\"lambda\":" << s.lambda << ",\"cap_violated\":" << (s.cap_violated ? 1 : 0) << ",\"bytes\":" << s.bytes
      << ",\"bytes_per_key_milli\":" << s.bytes * 1000 / n << ",\"direct_key_ppm\":" << dk * 1000000 / n << ",\"fenced_key_ppm\":" << fk * 1000000 / n
      << ",\"overcap_key_ppm\":" << (depth[1] + depth[2] + depth[3]) * 1000000 / n
      << ",\"depth_hist\":[" << depth[0] << ',' << depth[1] << ',' << depth[2] << ',' << depth[3] << "],\"tie_lines\":" << ties
      << ",\"scan_lines\":" << scan << ",\"compact_coverage_ppm\":" << (fk ? ck * 1000000 / fk : 0)
      << ",\"dep_lines_milli\":" << dep * 1000 / n << ",\"par_lines_milli\":" << par * 1000 / n << ",\"pages4k_milli\":" << pg * 1000 / n;
    return o.str();
}
std::string counts_json(const CountTotals& t, std::uint64_t n) {
    std::ostringstream o;
    o << "{\"dep_lines_milli\":" << t.dep * 1000 / n << ",\"par_lines_milli\":" << t.par * 1000 / n
      << ",\"pages4k_milli\":[" << t.pages4k[0] * 1000 / n << ',' << t.pages4k[1] * 1000 / n << ',' << t.pages4k[2] * 1000 / n << ']'
      << ",\"pages2m_milli\":[" << t.pages2m[0] * 1000 / n << ',' << t.pages2m[1] * 1000 / n << ',' << t.pages2m[2] * 1000 / n << ']'
      << ",\"router_child_milli\":" << t.rchild * 1000 / n << ",\"record_lines_milli\":" << t.rlines * 1000 / n
      << ",\"direct_key_ppm\":" << t.direct_keys * 1000000 / n << ",\"fenced_key_ppm\":" << t.fenced_keys * 1000000 / n
      << ",\"overcap_key_ppm\":" << t.overcap() * 1000000 / n << ",\"depth_hist\":[" << t.depth[0] << ',' << t.depth[1] << ',' << t.depth[2] << ',' << t.depth[3] << ']'
      << ",\"tie_lines\":" << t.ties() << ",\"scan_lines\":" << t.tagdep[T_SCAN] << ",\"exceptions\":" << t.exc << '}';
    return o.str();
}
std::string sim_json(const SimOut& s) {
    std::ostringstream o;
    o << "{\"thp\":" << (s.thp ? 1 : 0) << ",\"l2_kb\":" << s.l2 << ",\"P\":[" << s.P[0] << ',' << s.P[1] << ',' << s.P[2] << ']'
      << ",\"walks\":[" << s.walks[0] << ',' << s.walks[1] << ',' << s.walks[2] << "],\"router_miss_milli\":" << s.router_milli
      << ",\"record_miss_milli\":" << s.record_milli << ",\"exc_miss_milli\":" << s.exc_milli << ",\"lookups\":" << s.lookups << '}';
    return o.str();
}
struct Stage2 { CellKey key; int arm; SpliceParams p; FixedPoint fx; std::string json; const FitBundle* b = nullptr; };
std::string fixed_json(const Stage2& s, const Fit& f) {
    const FixedPoint& fx = s.fx; const Cell c = bundle_cell(*s.b, s.p);
    std::ostringstream o;
    o << "{\"splice_args\":" << q(format_params(s.p, true)) << ",\"arm\":" << q(s.arm ? "compact" : "fast") << ',' << selection_json(c, fx.sel)
      << ",\"outer_iters\":" << fx.iters << ",\"trace_keys\":" << fx.trace_keys << ",\"trace_truncated\":" << fx.trace_truncated << ",\"ed_hist\":[";
    for (std::size_t i = 0; i < fx.ed_hist.size(); ++i) o << (i ? "," : "") << fx.ed_hist[i];
    o << "],\"stage1_ed_milli\":" << fx.stage1 << ",\"ed_sel_milli\":" << fx.ed_sel << ",\"ed_milli\":{\"4k_2048\":" << fx.ed[0] << ",\"4k_1024\":" << fx.ed[1]
      << ",\"thp_2048\":" << fx.ed[2] << ",\"thp_1024\":" << fx.ed[3] << "},\"P_selected_with\":[" << fx.P.ad << ',' << fx.P.ae << ',' << fx.P.b << ']'
      << ",\"sim_sel\":" << sim_json(fx.sim) << ",\"sims\":[" << sim_json(fx.rep[0]) << ',' << sim_json(fx.rep[1]) << ',' << sim_json(fx.rep[2]) << ',' << sim_json(fx.rep[3]) << ']'
      << ",\"counts\":" << counts_json(fx.counts, f.n) << ",\"total_bytes\":" << fx.plan.total_bytes
      << ",\"arena_bytes\":" << fx.plan.arena_bytes << ",\"walk_s\":" << fmt_s(secs(fx.walk_ns)) << ",\"sim_s\":" << fmt_s(secs(fx.sim_ns)) << '}';
    return o.str();
}

struct Opts {
    std::string dataset, data, out = "results/splice_h", grid = "pruned", args, samples;
    unsigned threads = 16; bool verify = false, parity = false, hash = false, perturb = false, hash_samples = false;
};

struct GridResult {
    std::string json;                      // the body (without the outer braces' dataset header)
    SpliceParams chosen[2]; std::int64_t chosen_ed[2] = {}; CellKey chosen_key[2]{};
    bool have[2] = {false, false};
    bool feasible[2] = {false, false};   // the chosen cell meets the arm's bytes cap
};

// The pruned grid of the spec: stage 1 counts 36 cells x 2 arms with default residency; stage 2 runs the
// fixed point on the baselines and the 2 best cells per arm (shared between the arms); stage 3 ablates the
// chosen fast cell and counts the synthesis' own cell.
GridResult run_grid(const KeySrc& ks, const SpliceParams& base, unsigned threads, bool ablations) {
    GridResult g;
    std::ostringstream js;
    const std::uint32_t K1s[3] = {4096, 8192, 16384};
    const unsigned ELs[2] = {1, 2}; const std::uint32_t CBs[3] = {16, 24, 32};
    const std::uint32_t caps[2] = {220, 165};
    std::map<std::uint32_t, FitBundle> fits;
    // Stage 1a: three eps bisections at once.
    const long long t_fit = now_ns();
    {
        std::vector<FitBundle> tmp(3);
        const unsigned per = std::max(1u, threads / 3);
        par_for(3, 3, [&](std::size_t i) {
            SpliceParams p = base; p.k1 = K1s[i]; p.eps = 0; p.tier1 = TIER1_PLA;
            const long long t0 = now_ns(); tmp[i].fit = make_fit(ks, p, per); tmp[i].wall_ns = now_ns() - t0;
        });
        for (std::size_t i = 0; i < 3; ++i) fits.emplace(K1s[i], std::move(tmp[i]));
    }
    const long long fit_ns = now_ns() - t_fit;
    // Stage 1b: candidate passes.
    const long long t_cand = now_ns();
    for (auto& [k1, b] : fits) {
        for (unsigned qa : {1u, 2u}) b.direct.emplace(qa, eval_direct(b.fit, qa, threads));
        for (unsigned el : ELs) for (std::uint32_t cb : CBs) b.fenced[{el, cb}] = eval_fenced(b.fit, el, cb, 3, threads);
    }
    const long long cand_ns = now_ns() - t_cand;
    const std::uint64_t n = ks.n;
    js << "\"exceptions\":{\"bb\":" << fits.at(16384).fit.bb << ",\"bt\":" << fits.at(16384).fit.bt << ",\"keys\":[";
    {
        const Fit& f = fits.at(16384).fit;
        for (std::size_t i = 0; i < f.exc.size(); ++i) js << (i ? "," : "") << f.exc[i].key;
    }
    js << "]},\"fits\":[";
    bool first = true;
    for (auto& [k1, b] : fits) {
        const Fit& f = b.fit;
        js << (first ? "" : ",") << "{\"k1\":" << k1 << ",\"eps\":" << f.eps << ",\"segments\":" << f.K() << ",\"passes\":" << f.passes
           << ",\"router_nodes\":" << f.rnodes.size() << ",\"router_bytes\":" << f.router_bytes()
           << ",\"router_child_milli\":" << f.rchild_total * 1000 / n << ",\"router_depth_hist\":[" << f.router_depth_hist[0] << ',' << f.router_depth_hist[1]
           << ',' << f.router_depth_hist[2] << ',' << f.router_depth_hist[3] << ',' << f.router_depth_hist[4] << "],\"wall_s\":" << fmt_s(secs(b.wall_ns)) << '}';
        first = false;
    }
    js << "],\"cells\":[";
    // Stage 1c: score every cell for both arms.
    struct Scored { CellKey key; int arm; Selection sel; std::int64_t ed; std::uint64_t bpk; };
    std::vector<Scored> sc;
    const long long t_s1 = now_ns();
    first = true;
    for (std::uint32_t k1 : K1s) for (unsigned el : ELs) for (std::uint32_t cb : CBs) for (unsigned cp : {0u, 1u}) for (int arm : {0, 1}) {
        const CellKey key{k1, el, cb, cp};
        const FitBundle& b = fits.at(k1);
        const SpliceParams p = cell_params(base, b.fit, key, caps[arm]);
        const Cell c = bundle_cell(b, p);
        Selection s = select_cell(c, default_p(p), threads);
        const std::int64_t ed = stage1_ed(c, s, default_p(p));
        js << (first ? "" : ",") << "{\"k1\":" << k1 << ",\"entry\":" << el << ",\"cbar\":" << cb << ",\"compact\":" << cp << ",\"arm\":" << q(arm ? "compact" : "fast")
           << ",\"cap\":" << caps[arm] / 10 << '.' << caps[arm] % 10 << ",\"ed_milli\":" << ed << ',' << selection_json(c, s) << '}';
        first = false;
        sc.push_back(Scored{key, arm, std::move(s), ed, 0});
        sc.back().bpk = sc.back().sel.bytes * 1000 / n;
    }
    const long long s1_ns = now_ns() - t_s1;
    // Pareto front over all stage-1 scorings: bytes/key vs E[D], non-dominated.
    js << "],\"pareto\":[";
    {
        std::vector<std::size_t> idx(sc.size());
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        // Ties (one layout scored for both arms) resolve to the lower arm, then the grid order: deterministic.
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return std::tie(sc[a].bpk, sc[a].ed, sc[a].arm, a) < std::tie(sc[b].bpk, sc[b].ed, sc[b].arm, b); });
        std::int64_t best = INT64_MAX; first = true;
        for (auto i : idx) if (sc[i].ed < best) {
            best = sc[i].ed;
            js << (first ? "" : ",") << "{\"k1\":" << sc[i].key.k1 << ",\"entry\":" << sc[i].key.el << ",\"cbar\":" << sc[i].key.cbar << ",\"compact\":" << sc[i].key.compact
               << ",\"cap\":" << caps[sc[i].arm] / 10 << '.' << caps[sc[i].arm] % 10 << ",\"cap_violated\":" << sc[i].sel.cap_violated << ",\"bytes_per_key_milli\":" << sc[i].bpk << ",\"ed_milli\":" << sc[i].ed << '}';
            first = false;
        }
    }
    // Stage 2: fixed point on the baselines and the two best feasible cells per arm. The arms share their
    // finalists: each finalist also runs under the other arm's cap when stage 1 found it feasible there,
    // because stage-1 E[D] (default residency) can misrank cells by ~0.4 D and a cell that fits 16.5 B/key
    // is also a fast-arm candidate (genome, planet: the compact pick beat the fast pick on both axes).
    const CellKey baseline[2] = {{16384, 2, 24, 0}, {16384, 2, 24, 1}};
    std::vector<CellKey> finalists;
    auto add_key = [&](const CellKey& k) { if (std::find(finalists.begin(), finalists.end(), k) == finalists.end()) finalists.push_back(k); };
    for (int arm : {0, 1}) {
        add_key(baseline[arm]);
        std::vector<const Scored*> v;
        for (auto& s : sc) if (s.arm == arm && !s.sel.cap_violated && !(s.key == baseline[arm])) v.push_back(&s);
        std::stable_sort(v.begin(), v.end(), [](const Scored* a, const Scored* b) { return a->ed < b->ed; });
        for (std::size_t i = 0; i < v.size() && i < 2; ++i) add_key(v[i]->key);
    }
    std::vector<std::pair<CellKey, int>> pick;
    for (int arm : {0, 1})
        for (const CellKey& k : finalists) {
            bool ok = k == baseline[arm];
            for (auto& s : sc) if (s.arm == arm && s.key == k && !s.sel.cap_violated) ok = true;
            if (ok) pick.push_back({k, arm});
        }
    const Draws draws = make_draws(n, base.warm, base.sim, base.seed);
    // The cells run concurrently (at most 6 at a time, for memory): their LRU replays are single-threaded,
    // and results never depend on threads.
    std::vector<Stage2> s2(pick.size());
    const long long t_s2 = now_ns();
    const unsigned conc = unsigned(std::min<std::size_t>(pick.size(), 6));
    par_for(pick.size(), conc, [&](std::size_t i) {
        Stage2& s = s2[i]; s.key = pick[i].first; s.arm = pick[i].second; s.b = &fits.at(s.key.k1);
        s.p = cell_params(base, s.b->fit, s.key, caps[s.arm]);
        const Cell c = bundle_cell(*s.b, s.p);
        s.fx = fixed_point(c, base.sim ? &draws : nullptr, std::max(2u, threads / conc));
        s.fx.traces = Traces{};
        s.json = fixed_json(s, s.b->fit);
    });
    const long long s2_ns = now_ns() - t_s2;
    js << "],\"stage2\":[";
    for (std::size_t i = 0; i < s2.size(); ++i) js << (i ? "," : "") << s2[i].json;
    js << ']';
    // Choice per arm: the baseline stays unless a feasible cell is at least 0.3 D better.
    js << ",\"chosen\":{";
    for (int arm : {0, 1}) {
        const Stage2* bl = nullptr; const Stage2* best = nullptr;
        for (auto& s : s2) if (s.arm == arm) {
            if (s.key == baseline[arm]) bl = &s;
            if (!s.fx.sel.cap_violated && (!best || s.fx.ed_sel < best->fx.ed_sel)) best = &s;
        }
        const Stage2* ch = bl;
        std::string why = "baseline";
        if (!bl || bl->fx.sel.cap_violated) { ch = best ? best : bl; why = best ? "baseline infeasible; best feasible cell" : "no feasible cell; baseline"; }
        else if (best && best->fx.ed_sel + 300 <= bl->fx.ed_sel) { ch = best; why = "beats the baseline by >= 0.3 D"; }
        const bool ok = !ch->fx.sel.cap_violated;
        js << (arm ? "," : "") << q(arm ? "compact" : "fast") << ":{\"why\":" << q(why) << ",\"feasible\":" << (ok ? "true" : "false")
           << ",\"cell\":" << ch->json << '}';
        g.chosen[arm] = ch->p; g.chosen_ed[arm] = ch->fx.ed_sel; g.chosen_key[arm] = ch->key; g.have[arm] = true; g.feasible[arm] = ok;
    }
    js << '}';
    // Stage 3: ablations at the chosen fast cell.
    long long s3_ns = 0;
    js << ",\"ablations\":[";
    if (ablations) {
        const long long t3 = now_ns();
        const SpliceParams cp = g.chosen[0];
        FitBundle& cb = fits.at(g.chosen_key[0].k1);
        // Ablation fits (Hist-Tree knots; no exception list) are built side by side, then the four fixed
        // points run concurrently.
        FitBundle hb, eb;
        SpliceParams ph = cp, pe = cp;
        ph.tier1 = TIER1_HISTTREE; ph.eps = 0; pe.exc = 0; pe.eps = 0;
        par_for(2, 2, [&](std::size_t i) {
            FitBundle& b = i ? eb : hb; SpliceParams& p = i ? pe : ph;
            const long long t0 = now_ns(); b.fit = make_fit(ks, p, std::max(1u, threads / 2)); b.wall_ns = now_ns() - t0;
            p.eps = b.fit.eps;
            ensure_caches(b, p, std::max(1u, threads / 2));
        });
        struct Ab { std::string name; FitBundle* b; SpliceParams p; Stage2 s; };
        std::vector<Ab> ab;
        { SpliceParams p = cp; p.mode = MODE_DIRECT; ab.push_back({"mode=direct", &cb, p, {}}); }
        { SpliceParams p = cp; p.mode = MODE_FENCED; ab.push_back({"mode=fenced", &cb, p, {}}); }
        ab.push_back({"tier1=histtree", &hb, ph, {}});
        ab.push_back({"exc=0", &eb, pe, {}});
        // The synthesis' own cell, like for like with its predictions: FENCED only, K1 16384, 2-line entries,
        // cbar 24, fast lines (C = 228), no bytes cap (40 B/key never binds).
        {
            FitBundle& b16 = fits.at(16384);
            SpliceParams p = cell_params(base, b16.fit, CellKey{16384, 2, 24, 0}, 400); p.mode = MODE_FENCED;
            ab.push_back({"synthesis_cell", &b16, p, {}});
        }
        for (auto& a : ab) ensure_caches(*a.b, a.p, threads);
        par_for(ab.size(), unsigned(ab.size()), [&](std::size_t i) {
            Ab& a = ab[i];
            const Cell c = bundle_cell(*a.b, a.p);
            a.s.key = a.name == "synthesis_cell" ? CellKey{16384, 2, 24, 0} : g.chosen_key[0]; a.s.arm = 0; a.s.p = a.p; a.s.b = a.b;
            a.s.fx = fixed_point(c, base.sim ? &draws : nullptr, std::max(2u, threads / unsigned(ab.size()))); a.s.fx.traces = Traces{};
        });
        first = true;
        for (auto& a : ab) {
            js << (first ? "" : ",") << "{\"ablation\":" << q(a.name) << ",\"eps\":" << a.b->fit.eps << ",\"segments\":" << a.b->fit.K() << ",\"exceptions\":" << a.b->fit.exc.size()
               << ",\"fit_wall_s\":" << fmt_s(secs(a.b->wall_ns)) << ",\"ed_delta_milli\":" << (a.s.fx.ed_sel - g.chosen_ed[0]) << ",\"cell\":" << fixed_json(a.s, a.b->fit) << '}';
            first = false;
        }
        s3_ns = now_ns() - t3;
    }
    js << ']';
    js << ",\"stage_wall_s\":{\"fit\":" << fmt_s(secs(fit_ns)) << ",\"candidates\":" << fmt_s(secs(cand_ns)) << ",\"stage1\":" << fmt_s(secs(s1_ns))
       << ",\"stage2\":" << fmt_s(secs(s2_ns)) << ",\"stage3\":" << fmt_s(secs(s3_ns)) << ",\"candidate_passes\":{";
    first = true;
    for (auto& [k1, b] : fits) {
        for (auto& [qa, d] : b.direct) { js << (first ? "" : ",") << "\"k" << k1 << "_a" << qa << "\":" << fmt_s(secs(d.ns)); first = false; }
        for (auto& [k, d] : b.fenced) { js << ",\"k" << k1 << "_e" << k.first << "_c" << k.second << "\":" << fmt_s(secs(d.ns)); }
    }
    js << "}}";
    g.json = js.str();
    return g;
}

// Offline-only perturbation: t' = (t + 1e-6 t^2) / (1 + 1e-6) on the normalised key, ends fixed,
// then +1 bumps keep the keys strictly increasing.
std::vector<std::uint64_t> perturb_keys(const std::uint64_t* k, std::size_t n) {
    std::vector<std::uint64_t> out(n);
    if (!n) return out;
    const std::uint64_t lo = k[0], hi = k[n - 1];
    const double span = double(hi - lo);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = span > 0 ? double(k[i] - lo) / span : 0.0;
        const double tp = (t + 1e-6 * t * t) / (1.0 + 1e-6);
        const double d = std::nearbyint(tp * span);
        std::uint64_t x = d >= 18446744073709549568.0 ? hi : lo + std::uint64_t(d);
        if (i && x <= out[i - 1]) {
            if (out[i - 1] == ~0ull) throw std::runtime_error("perturbation ran out of key space");
            x = out[i - 1] + 1;
        }
        out[i] = x;
    }
    return out;
}

// Every stored key returns its payload (payload = key here) and 10M absent keys (splitmix64 stream,
// filtered by binary search) return false.
std::string verify_json(const Index& ix, const Mapped& m, unsigned threads) {
    const long long tv = now_ns();
    std::atomic<std::uint64_t> found{0}, absent_false{0}, absent_true{0};
    const std::size_t ch = 1u << 20, nc = (m.n + ch - 1) / ch;
    par_for(nc, threads, [&](std::size_t c) {
        std::uint64_t f = 0;
        for (std::size_t i = c * ch; i < std::min(m.n, (c + 1) * ch); ++i) { std::uint64_t v; f += ix.get(m.keys[i], v) && v == m.keys[i]; }
        found += f;
    });
    const std::size_t na = 10000000, ac = 64;
    par_for(ac, threads, [&](std::size_t c) {
        std::uint64_t ok = 0, bad = 0, s = 0x5eed0000ull + c;
        for (std::size_t got = 0; got < na / ac;) {
            const std::uint64_t x = splitmix64(s);
            const std::size_t u = std::size_t(std::upper_bound(m.keys, m.keys + m.n, x) - m.keys);
            if (u && m.keys[u - 1] == x) continue;
            std::uint64_t v; (ix.get(x, v) ? bad : ok) += 1; ++got;
        }
        absent_false += ok; absent_true += bad;
    });
    if (found != m.n || absent_true) std::cerr << "splice_count: verify FAILED\n";
    std::ostringstream js;
    js << "{\"found\":" << found.load() << ",\"n\":" << m.n << ",\"absent_false\":" << absent_false.load() << ",\"absent_true\":" << absent_true.load()
       << ",\"pass\":" << ((found == m.n && absent_true == 0) ? "true" : "false") << ",\"wall_s\":" << fmt_s(secs(now_ns() - tv)) << '}';
    return js.str();
}

int run_dataset(const Opts& o) {
    const long long t0 = now_ns();
    std::string path = o.data + "/" + o.dataset;
    if (needs_sorted(o.dataset)) path += ".sorted";
    Mapped m; map_file(m, path, o.dataset, o.threads);
    const KeySrc ks{m.keys, nullptr, 1, m.n};
    SpliceParams base = parse_params(o.args, false);
    base.threads = o.threads;
    fs::create_directories(o.out);
    std::ostringstream js;
    js << "{\"dataset\":" << q(o.dataset) << ",\"file\":" << q(path) << ",\"n\":" << m.n << ",\"threads\":" << o.threads
       << ",\"base_args\":" << q(format_params(base, true)) << ',';
    SpliceParams fast, compact; std::int64_t fast_ed = 0, compact_ed = 0; bool compact_ok = false;
    if (o.grid == "pruned") {
        const GridResult g = run_grid(ks, base, o.threads, true);
        js << g.json;
        fast = g.chosen[0]; compact = g.chosen[1]; fast_ed = g.chosen_ed[0]; compact_ed = g.chosen_ed[1]; compact_ok = g.feasible[1];
        fs::create_directories(o.out + "/args");
        // An arm whose chosen cell violates its bytes cap gets no args file (a stale one is removed), so
        // gre_run.sh --splice-plan refuses that dataset and arm instead of running a cap-violating layout.
        for (int arm : {0, 1}) {
            const std::string f = o.out + "/args/" + o.dataset + (arm ? ".compact.args" : ".args");
            if (!g.feasible[arm]) {
                fs::remove(f);
                std::cerr << "splice_count: " << o.dataset << ' ' << (arm ? "compact" : "fast") << " arm has no feasible cell; no " << f << '\n';
                continue;
            }
            std::ofstream a(f);
            a << format_params(arm ? compact : fast, true) << '\n';
        }
        if (o.perturb) {
            const long long tp = now_ns();
            const auto pk = perturb_keys(m.keys, m.n);
            const GridResult gp = run_grid(KeySrc{pk.data(), nullptr, 1, pk.size()}, base, o.threads, false);
            const bool same = gp.chosen_key[0] == g.chosen_key[0];
            const std::int64_t d = gp.chosen_ed[0] - g.chosen_ed[0];
            js << ",\"perturb\":{\"same_cell\":" << (same ? "true" : "false") << ",\"ed_milli\":" << gp.chosen_ed[0] << ",\"ed_delta_milli\":" << d
               << ",\"pass\":" << ((same && d <= 10 && d >= -10) ? "true" : "false") << ",\"chosen_args\":" << q(format_params(gp.chosen[0], true))
               << ",\"wall_s\":" << fmt_s(secs(now_ns() - tp)) << '}';
        }
    } else if (o.grid == "cell") {
        FitBundle b; b.fit = make_fit(ks, base, o.threads);
        SpliceParams p = base; p.eps = b.fit.eps;
        ensure_caches(b, p, o.threads);
        const Cell c = bundle_cell(b, p);
        Draws d; if (p.sim) d = make_draws(m.n, p.warm, p.sim, p.seed);
        Stage2 s; s.p = p; s.b = &b; s.arm = p.compact ? 1 : 0; s.key = CellKey{p.k1, p.entry, p.cbar, p.compact};
        s.fx = fixed_point(c, p.sim ? &d : nullptr, o.threads); s.fx.traces = Traces{};
        js << "\"cell\":" << fixed_json(s, b.fit);
        fast = p; fast_ed = s.fx.ed_sel;
    } else throw std::invalid_argument("--grid must be pruned or cell");
    // The chosen fast cell, built as GRE builds it (eps explicit), for its layout hash and the checks.
    if (o.hash || o.verify || o.parity || o.grid == "pruned") {
        SpliceParams p = fast; p.threads = o.threads; p.hash = 1; p.verify = 0;
        Index ix;
        ix.build(Input{m.keys, nullptr, 1, m.n}, p);
        const Stats& st = ix.stats();
        js << ",\"layout_hash\":" << q(hex(ix.layout_hash())) << ",\"full_hash\":" << q(hex(ix.full_hash()))
           << ",\"index\":" << stats_json(st) << ",\"reproduced\":" << (st.ed_sel_milli == fast_ed ? "true" : "false");
        if (o.verify) js << ",\"verify\":" << verify_json(ix, m, o.threads);
        if (o.parity) {
            const long long tq = now_ns();
            const std::size_t ch = 1u << 20, nc = (m.n + ch - 1) / ch;
            std::vector<CountTotals> part(nc);
            par_for(nc, o.threads, [&](std::size_t c) {
                Count cnt(ix.view());
                for (std::size_t i = c * ch; i < std::min(m.n, (c + 1) * ch); ++i) { std::uint64_t v; get_impl(ix.view(), m.keys[i], v, cnt); }
                part[c] = cnt.core.t;
            });
            CountTotals t; for (auto& x : part) t += x;
            const CountTotals& w = ix.plan_counts();
            const bool eq = t == w;
            js << ",\"parity\":{\"equal\":" << (eq ? "true" : "false") << ",\"get\":" << counts_json(t, m.n) << ",\"walker\":" << counts_json(w, m.n)
               << ",\"dep\":[" << t.dep << ',' << w.dep << "],\"par\":[" << t.par << ',' << w.par << "],\"pages4k\":[" << (t.pages4k[0] + t.pages4k[1] + t.pages4k[2])
               << ',' << (w.pages4k[0] + w.pages4k[1] + w.pages4k[2]) << "],\"pages2m\":[" << (t.pages2m[0] + t.pages2m[1] + t.pages2m[2]) << ',' << (w.pages2m[0] + w.pages2m[1] + w.pages2m[2])
               << "],\"wall_s\":" << fmt_s(secs(now_ns() - tq)) << '}';
            if (!eq) std::cerr << "splice_count: parity FAILED\n";
        }
        std::cout << "layout_hash " << o.dataset << ' ' << hex(ix.layout_hash()) << '\n';
    }
    // The feasible compact arm too, built from its args (gre_report --splice-json checks GRE's hash
    // against compact_layout_hash under --splice-arm compact).
    if (o.grid == "pruned" && compact_ok) {
        SpliceParams p = compact; p.threads = o.threads; p.hash = 1; p.verify = 0;
        Index ix;
        ix.build(Input{m.keys, nullptr, 1, m.n}, p);
        js << ",\"compact_layout_hash\":" << q(hex(ix.layout_hash())) << ",\"compact_index\":" << stats_json(ix.stats())
           << ",\"compact_reproduced\":" << (ix.stats().ed_sel_milli == compact_ed ? "true" : "false");
        if (o.verify) js << ",\"compact_verify\":" << verify_json(ix, m, o.threads);
        std::cout << "compact_layout_hash " << o.dataset << ' ' << hex(ix.layout_hash()) << '\n';
    }
    const double wall = secs(now_ns() - t0);
    js << ",\"timings\":{\"wall_s\":" << fmt_s(wall) << ",\"target_s\":600,\"over_target\":" << (wall > 600 ? "true" : "false") << "}}";
    const std::string file = o.out + "/" + o.dataset + (o.grid == "cell" ? ".cell.json" : ".json");
    std::ofstream(file) << js.str() << '\n';
    std::cout << "wrote " << file << " (" << fmt_s(wall) << " s)\n";
    return 0;
}

int run_samples(const Opts& o) {
    std::vector<std::string> names;
    for (auto& e : fs::directory_iterator(o.samples)) {
        const auto s = e.path().filename().string();
        if (s.find("_2M_") != std::string::npos && s.find('.') == std::string::npos) names.push_back(s);
    }
    std::sort(names.begin(), names.end());
    SpliceParams p = parse_params(o.args.empty() ? "k1=1024 sim=200000 warm=50000" : o.args, false);
    p.threads = o.threads; p.hash = 1;
    for (auto& s : names) {
        Mapped m; map_file(m, o.samples + "/" + s, "", o.threads);
        Index ix; ix.build(Input{m.keys, nullptr, 1, m.n}, p);
        std::cout << "hash " << s << ' ' << hex(ix.layout_hash()) << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Opts o;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&]() -> std::string { if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value"); return argv[++i]; };
            if (a == "--dataset") o.dataset = val();
            else if (a == "--data") o.data = val();
            else if (a == "--out") o.out = val();
            else if (a == "--threads") o.threads = unsigned(std::stoul(val()));
            else if (a == "--grid") o.grid = val();
            else if (a == "--args") o.args = val();
            else if (a == "--samples") o.samples = val();
            else if (a == "--verify") o.verify = true;
            else if (a == "--parity") o.parity = true;
            else if (a == "--hash") o.hash = true;
            else if (a == "--perturb") o.perturb = true;
            else if (a == "--hash-samples") o.hash_samples = true;
            else if (a == "--help" || a == "-h") {
                std::cout << "splice_count --dataset NAME --data DIR [--out results/splice_h] [--threads 16] [--grid pruned|cell]\n"
                             "             [--args \"k=v ...\"] [--verify] [--parity] [--hash] [--perturb]\n"
                             "splice_count --samples DIR --hash-samples [--args \"k=v ...\"]\n";
                return 0;
            } else throw std::invalid_argument("unknown option " + a);
        }
        if (o.threads == 0 || o.threads > 1024) throw std::invalid_argument("--threads must be in [1,1024]");
        if (o.hash_samples) { if (o.samples.empty()) throw std::invalid_argument("--hash-samples needs --samples DIR"); return run_samples(o); }
        if (o.dataset.empty() || o.data.empty()) throw std::invalid_argument("--dataset and --data are required");
        return run_dataset(o);
    } catch (const std::exception& e) {
        std::cerr << "splice_count: " << e.what() << '\n';
        return 2;
    }
}
