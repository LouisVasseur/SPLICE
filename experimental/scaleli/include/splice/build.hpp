// SPLICE-H build: fit, select, plan, fill one arena (C++20). GRE reaches it through
// integrations/gre_splice/splice_gre.cpp; tools/splice_count.cpp uses cost.hpp for the grid and this
// class for the chosen cell, so the offline layout hash is the one GRE builds from the same args.
#pragma once
#include "splice/cost.hpp"
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <unistd.h>

namespace splice {

struct Input { const std::uint64_t* key; const std::uint64_t* val; std::size_t stride_words; std::size_t n; };

struct Stats {
    std::uint64_t n = 0, exceptions = 0, segments = 0, eps = 0, k1 = 0, tier1 = 0, passes = 0;
    std::uint64_t router_nodes = 0, router_bytes = 0, records_bytes = 0, arena_bytes = 0, arena_used_bytes = 0, total_bytes = 0;
    std::uint64_t direct_key_ppm = 0, fenced_key_ppm = 0, overcap_key_ppm = 0, depth_hist[4] = {}, tie_lines = 0;
    std::uint64_t dep_lines_milli = 0, par_lines_milli = 0, pages4k_milli = 0, pages2m_milli = 0, router_child_milli = 0, record_lines_milli = 0;
    std::int64_t ed_milli[4] = {}, ed_sel_milli = 0, stage1_ed_milli = 0, p_ad = 0, p_ae = 0, p_b = 0, router_miss_milli = 0, record_miss_milli = 0;
    std::uint64_t outer_iters = 0, lambda = 0, cap_violated = 0, thp_applied = 0, threads = 0;
    std::uint64_t direct_segments = 0, fenced_segments = 0, bytes_per_key_milli = 0, compact_keys = 0;
    long long fit_ns = 0, candidates_ns = 0, walk_ns = 0, sim_ns = 0, fill_ns = 0, verify_ns = 0, build_ns = 0, hash_ns = 0;
};
inline std::string stats_json(const Stats& s) {
    std::ostringstream o;
    auto kv = [&](const char* k, auto v, bool first = false) { o << (first ? "" : ",") << '"' << k << "\":" << v; };
    o << '{';
    kv("n", s.n, true); kv("exceptions", s.exceptions); kv("segments", s.segments); kv("eps", s.eps); kv("k1", s.k1);
    kv("tier1", s.tier1 == TIER1_PLA ? "\"pla\"" : "\"histtree\""); kv("passes", s.passes);
    kv("router_nodes", s.router_nodes); kv("router_bytes", s.router_bytes); kv("records_bytes", s.records_bytes);
    kv("arena_bytes", s.arena_bytes); kv("arena_used_bytes", s.arena_used_bytes); kv("total_bytes", s.total_bytes);
    kv("bytes_per_key_milli", s.bytes_per_key_milli); kv("direct_segments", s.direct_segments); kv("compact_keys", s.compact_keys); kv("fenced_segments", s.fenced_segments);
    kv("direct_key_ppm", s.direct_key_ppm); kv("fenced_key_ppm", s.fenced_key_ppm); kv("overcap_key_ppm", s.overcap_key_ppm);
    o << ",\"depth_hist\":[" << s.depth_hist[0] << ',' << s.depth_hist[1] << ',' << s.depth_hist[2] << ',' << s.depth_hist[3] << ']';
    kv("tie_lines", s.tie_lines); kv("dep_lines_milli", s.dep_lines_milli); kv("par_lines_milli", s.par_lines_milli);
    kv("pages4k_milli", s.pages4k_milli); kv("pages2m_milli", s.pages2m_milli); kv("router_child_milli", s.router_child_milli);
    kv("record_lines_milli", s.record_lines_milli);
    o << ",\"ed_milli\":[" << s.ed_milli[0] << ',' << s.ed_milli[1] << ',' << s.ed_milli[2] << ',' << s.ed_milli[3] << ']';
    kv("ed_sel_milli", s.ed_sel_milli); kv("stage1_ed_milli", s.stage1_ed_milli); kv("p_ad", s.p_ad); kv("p_ae", s.p_ae); kv("p_b", s.p_b);
    kv("router_miss_milli", s.router_miss_milli); kv("record_miss_milli", s.record_miss_milli);
    kv("outer_iters", s.outer_iters); kv("lambda", s.lambda); kv("cap_violated", s.cap_violated); kv("thp_applied", s.thp_applied);
    kv("threads", s.threads); kv("fit_ns", s.fit_ns); kv("candidates_ns", s.candidates_ns); kv("walk_ns", s.walk_ns); kv("sim_ns", s.sim_ns);
    kv("fill_ns", s.fill_ns); kv("verify_ns", s.verify_ns); kv("build_ns", s.build_ns); kv("hash_ns", s.hash_ns);
    o << '}';
    return o.str();
}

// mix64 chain over little-endian 64-bit words; a ragged tail is zero-padded into one word.
inline std::uint64_t hash_bytes(std::uint64_t h, const void* p, std::size_t bytes) noexcept {
    const auto* b = static_cast<const unsigned char*>(p);
    std::size_t i = 0;
    for (; i + 8 <= bytes; i += 8) { std::uint64_t w; std::memcpy(&w, b + i, 8); h = mix64(h ^ w); }
    if (i < bytes) { std::uint64_t w = 0; std::memcpy(&w, b + i, bytes - i); h = mix64(h ^ w); }
    return h;
}

// One anonymous mapping, 2 MiB aligned (THP-eligible), unmapped in the destructor.
class Arena {
    unsigned char* p_ = nullptr; std::size_t bytes_ = 0; bool thp_ = false;
public:
    Arena() = default; Arena(const Arena&) = delete; Arena& operator=(const Arena&) = delete;
    ~Arena() { if (p_) ::munmap(p_, bytes_); }
    void map(std::size_t bytes, bool want_thp) {
        constexpr std::size_t A = 2u << 20;
        void* raw = ::mmap(nullptr, bytes + A, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) throw std::runtime_error("arena mmap of " + std::to_string(bytes) + " bytes failed");
        const auto r = reinterpret_cast<std::uintptr_t>(raw), a = (r + A - 1) / A * A;
        if (a > r) ::munmap(raw, a - r);
        if (r + bytes + A > a + bytes) ::munmap(reinterpret_cast<void*>(a + bytes), r + bytes + A - (a + bytes));
        p_ = reinterpret_cast<unsigned char*>(a); bytes_ = bytes;
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (want_thp) thp_ = ::madvise(p_, bytes_, MADV_HUGEPAGE) == 0;   // before the first write, so faults can take huge pages
#else
        (void)want_thp;
#endif
    }
    unsigned char* data() const noexcept { return p_; }
    std::size_t size() const noexcept { return bytes_; }
    bool thp() const noexcept { return thp_; }
    // AnonHugePages of the mapping from /proc/self/smaps; -1 where unavailable.
    long long anon_huge_bytes() const {
#if defined(__linux__)
        std::ifstream in("/proc/self/smaps");
        if (!in || !p_) return -1;
        const auto lo = reinterpret_cast<std::uintptr_t>(p_), hi = lo + bytes_;
        std::string line; bool inside = false; long long kb = 0; bool seen = false;
        while (std::getline(in, line)) {
            const std::string tok = line.substr(0, line.find(' '));
            const auto dash = tok.find('-');
            if (dash != std::string::npos && tok.find(':') == std::string::npos) {   // "start-end perms ..." header
                const std::uintptr_t s = std::uintptr_t(std::stoull(tok.substr(0, dash), nullptr, 16));
                inside = s >= lo && s < hi;
            } else if (inside && line.rfind("AnonHugePages:", 0) == 0) { kb += std::stoll(line.substr(14)); seen = true; }
        }
        return seen ? kb * 1024 : -1;
#else
        return -1;
#endif
    }
};

class Index {
public:
    Index() = default;
    Index(const Index&) = delete; Index& operator=(const Index&) = delete;
    Index(Index&&) = delete; Index& operator=(Index&&) = delete;

    void build(const Input& in, const SpliceParams& p) {
        const long long t0 = now_ns();
        if (in.n && (!in.key || in.stride_words == 0)) throw std::invalid_argument("splice::Input needs keys and a stride");
        const KeySrc ks{in.key, in.val, in.stride_words, in.n};
        for (std::size_t i = 1; i < in.n; ++i)
            if (ks.key(i) <= ks.key(i - 1)) throw std::invalid_argument("keys must be sorted and unique (duplicate or descent at " + std::to_string(i) + ")");
        params_ = p;
        const unsigned threads = resolve_threads(p.threads);
        stats_ = Stats{}; stats_.threads = threads; stats_.n = in.n; stats_.k1 = p.k1; stats_.tier1 = p.tier1;
        if (!in.n) build_empty(p);
        else build_keys(ks, p, threads);
        stats_.build_ns = build_ns_ = now_ns() - t0;
        if (p.verify) verify(ks);
        if (p.hash) { const long long h0 = now_ns(); compute_hashes(threads); stats_.hash_ns = hash_ns_ = now_ns() - h0; }
        lk_.clear(); lk_.shrink_to_fit();
    }
    const View& view() const noexcept { return view_; }
    bool get(std::uint64_t k, std::uint64_t& v) const noexcept { return splice::get(view_, k, v); }
    const Stats& stats() const noexcept { return stats_; }
    std::uint64_t layout_hash() const { return lhash_; }
    std::uint64_t full_hash() const { return fhash_; }
    std::size_t total_bytes() const noexcept { return std::size_t(stats_.total_bytes); }
    long long arena_anon_huge_bytes() const { return arena_.anon_huge_bytes(); }
    long long build_ns() const noexcept { return build_ns_; }
    long long hash_ns() const noexcept { return hash_ns_; }
    const CountTotals& plan_counts() const noexcept { return plan_counts_; }   // the plan walker's totals (parity target)
    const SpliceParams& params() const noexcept { return params_; }
    std::uint64_t touch_sum() const noexcept { return touch_; }

private:
    SpliceParams params_;
    Arena arena_;
    ABuf<Seg16> seg_;
    ABuf<std::uint32_t> rtop_, rsub_;
    ABuf<RNode> rnodes_;
    ABuf<Exc> exc_;
    View view_{};
    Stats stats_;
    CountTotals plan_counts_;
    std::vector<std::uint8_t> lk_;         // per arena line: which words are payloads (hash only)
    std::uint64_t lhash_ = 0, fhash_ = 0, touch_ = 0;
    long long build_ns_ = 0, hash_ns_ = 0;

    void set_view(std::uint64_t lo, std::uint64_t span, std::size_t K, unsigned rshift, unsigned entry) {
        view_ = View{};
        view_.lo = lo; view_.span = span; view_.rtop = rtop_.data(); view_.rnodes = rnodes_.data(); view_.rsub = rsub_.data();
        view_.seg = seg_.data(); view_.nseg = std::uint32_t(K); view_.rshift = std::uint8_t(rshift);
        view_.entry_lines = std::uint8_t(entry); view_.nfence = std::uint8_t(entry == 2 ? 56 : 24);
        view_.arena = arena_.data(); view_.exc = exc_.data(); view_.nexc = std::uint32_t(exc_.size());
    }
    // n == 0: one DIRECT segment whose 8 lines hold key 1, which the range check sends to the (empty)
    // exception list, so every lookup fails without a special case.
    void build_empty(const SpliceParams& p) {
        arena_.map(2u << 20, p.thp);
        Line one{}; for (int t = 0; t < 4; ++t) { one.k[t] = 1; one.v[t] = 0; }
        for (int l = 0; l < 16; ++l) if (l == 0 || l >= 8) std::memcpy(arena_.data() + l * 64, &one, 64);
        lk_.assign(arena_.size() / 64, LK_NONE);
        lk_[0] = LK_LINE; for (int l = 8; l < 16; ++l) lk_[std::size_t(l)] = LK_LINE;
        seg_.reset(10);
        seg_[0] = Seg16{0, 0, seg_meta(1, 0, 1)}; seg_[1] = Seg16{0, 0, seg_meta(2, 0, 0)};
        rtop_.reset(std::size_t(1) << p.router_bits);
        set_view(0, 0, 1, 0, p.entry);
        stats_.segments = 1; stats_.arena_bytes = arena_.size(); stats_.arena_used_bytes = 16 * 64;
        stats_.records_bytes = 160; stats_.router_bytes = rtop_.size() * 4;
        stats_.total_bytes = stats_.arena_bytes + stats_.records_bytes + stats_.router_bytes;
        stats_.thp_applied = arena_.thp();
    }
    void build_keys(const KeySrc& ks, const SpliceParams& p, unsigned threads) {
        Fit fit = make_fit(ks, p, threads);
        const long long tc = now_ns();
        std::vector<DirectCache> dcs; std::vector<const DirectCache*> dptr;
        FencedCache fc; const FencedCache* fp_ = nullptr;
        if (p.mode != MODE_FENCED) for (unsigned q : p.alphas()) dcs.push_back(eval_direct(fit, q, threads));
        for (auto& d : dcs) dptr.push_back(&d);
        if (p.mode != MODE_DIRECT) { fc = eval_fenced(fit, p.entry, p.cbar, 1u << p.compact, threads); fp_ = &fc; }
        const Cell cell = make_cell(fit, p, dptr, fp_);
        stats_.candidates_ns = now_ns() - tc;
        Draws draws;
        if (p.sim) draws = make_draws(ks.n, p.warm, p.sim, p.seed);
        FixedPoint fx = fixed_point(cell, p.sim ? &draws : nullptr, threads);
        draws = Draws{};
        const long long tf = now_ns();
        const Plan& pl = fx.plan; const std::size_t K = fit.K();
        arena_.map(pl.arena_bytes, p.thp);
        if (p.hash) lk_.assign(pl.arena_bytes / 64, LK_NONE);
        // Line 0, the empty line: EMPTY entries point here; it holds the bulk's first key and payload,
        // so an absent key never matches it and a stored one matches with its own payload.
        Line z{}; for (int t = 0; t < 4; ++t) { z.k[t] = fit.lo; z.v[t] = ks.val(fit.bb); }
        std::memcpy(arena_.data(), &z, 64);
        if (p.hash) lk_[0] = LK_LINE;
        fill_arena(cell, pl, arena_.data(), p.hash ? lk_.data() : nullptr, threads);
        seg_.reset(K + 9); fill_records(cell, pl, seg_.data());
        rtop_.reset(fit.rtop.size()); std::memcpy(rtop_.data(), fit.rtop.data(), 4 * fit.rtop.size());
        rnodes_.reset(fit.rnodes.size()); if (!fit.rnodes.empty()) std::memcpy(static_cast<void*>(rnodes_.data()), fit.rnodes.data(), 16 * fit.rnodes.size());
        rsub_.reset(fit.rsub.size()); if (!fit.rsub.empty()) std::memcpy(rsub_.data(), fit.rsub.data(), 4 * fit.rsub.size());
        exc_.reset(fit.exc.size()); if (fit.exc.size()) std::memcpy(static_cast<void*>(exc_.data()), fit.exc.data(), 16 * fit.exc.size());
        set_view(fit.lo, fit.span, K, fit.rshift, p.entry);
        // Read the hot tables once so the first timed lookups do not fault them in. Read-only: the
        // lookup keeps no state, so this is identical with and without GRE's warm-up.
        std::uint64_t t = 0;
        for (std::size_t i = 0; i < rtop_.size(); i += 16) t += rtop_[i];
        for (std::size_t i = 0; i < rsub_.size(); i += 16) t += rsub_[i];
        for (std::size_t i = 0; i < rnodes_.size(); i += 4) t += rnodes_[i].lo;
        for (std::size_t i = 0; i < seg_.size(); i += 4) t += seg_[i].key;
        touch_ = t;
        stats_.fill_ns = now_ns() - tf;
        plan_counts_ = fx.counts;
        fill_stats(fit, cell, fx);
    }
    void fill_stats(const Fit& f, const Cell& c, const FixedPoint& fx) {
        Stats& s = stats_; const CountTotals& t = fx.counts; const std::uint64_t n = f.n;
        s.exceptions = f.exc.size(); s.segments = f.K(); s.eps = f.eps; s.passes = f.passes;
        s.router_nodes = f.rnodes.size(); s.router_bytes = f.router_bytes(); s.records_bytes = (f.K() + 9) * 16;
        s.arena_bytes = fx.plan.arena_bytes; s.arena_used_bytes = fx.plan.b_end; s.total_bytes = fx.plan.total_bytes;
        s.bytes_per_key_milli = s.total_bytes * 1000 / n;
        for (std::size_t j = 0; j < f.K(); ++j) {
            const bool fen = c.desc[fx.plan.choice[j]].fenced;
            (fen ? s.fenced_segments : s.direct_segments) += 1;
            if (fen) s.compact_keys += c.sum(j, fx.plan.choice[j]).compact_keys;
        }
        s.direct_key_ppm = t.direct_keys * 1000000 / n; s.fenced_key_ppm = t.fenced_keys * 1000000 / n; s.overcap_key_ppm = t.overcap() * 1000000 / n;
        for (int i = 0; i < 4; ++i) s.depth_hist[i] = t.depth[i];
        s.tie_lines = t.ties();
        s.dep_lines_milli = t.dep * 1000 / n; s.par_lines_milli = t.par * 1000 / n;
        s.pages4k_milli = (t.pages4k[0] + t.pages4k[1] + t.pages4k[2]) * 1000 / n;
        s.pages2m_milli = (t.pages2m[0] + t.pages2m[1] + t.pages2m[2]) * 1000 / n;
        s.router_child_milli = t.rchild * 1000 / n; s.record_lines_milli = t.rlines * 1000 / n;
        for (int i = 0; i < 4; ++i) s.ed_milli[i] = fx.ed[i];
        s.stage1_ed_milli = fx.stage1; s.ed_sel_milli = fx.ed_sel;
        s.p_ad = fx.sim.P[REG_AD]; s.p_ae = fx.sim.P[REG_AE]; s.p_b = fx.sim.P[REG_B];
        s.router_miss_milli = fx.sim.router_milli; s.record_miss_milli = fx.sim.record_milli;
        s.outer_iters = fx.iters; s.lambda = fx.sel.lambda; s.cap_violated = fx.sel.cap_violated; s.thp_applied = arena_.thp();
        s.fit_ns = f.fit_ns + f.router_ns; s.walk_ns = fx.walk_ns; s.sim_ns = fx.sim_ns;
    }
    void verify(const KeySrc& ks) {
        const long long t0 = now_ns();
        for (std::size_t i = 0; i < ks.n; ++i) {
            std::uint64_t v = 0;
            if (!get(ks.key(i), v) || v != ks.val(i)) throw std::runtime_error("verify: key " + std::to_string(ks.key(i)) + " at " + std::to_string(i) + " not found with its payload");
        }
        std::uint64_t probes = 0;
        for (std::uint64_t i = 0; probes < 1000000 && i < 4000000; ++i) {
            const std::uint64_t x = mix64(i ^ 0x5eed);
            if (ks.n && ks.upper_bound(0, ks.n, x) > 0 && ks.key(ks.upper_bound(0, ks.n, x) - 1) == x) continue;
            std::uint64_t v = 0;
            if (get(x, v)) throw std::runtime_error("verify: absent key " + std::to_string(x) + " reported present");
            ++probes;
        }
        stats_.verify_ns = now_ns() - t0;
    }
    // layout_hash skips every payload word (comparable across payload sets); full_hash covers all.
    void compute_hashes(unsigned threads) {
        std::uint64_t lh = mix64(stats_.n), fh = lh;
        auto both = [&](const void* p, std::size_t b) { lh = hash_bytes(lh, p, b); fh = hash_bytes(fh, p, b); };
        both(&view_.lo, 8); both(&view_.span, 8);
        both(seg_.data(), seg_.size() * 16);
        both(rtop_.data(), rtop_.size() * 4); both(rnodes_.data(), rnodes_.size() * 16); both(rsub_.data(), rsub_.size() * 4);
        for (std::size_t i = 0; i < exc_.size(); ++i) { lh = hash_bytes(lh, &exc_[i].key, 8); fh = hash_bytes(fh, &exc_[i], 16); }
        const std::size_t lines = arena_.size() / 64, CH = 32768;   // 2 MiB chunks: independent of the thread count
        const std::size_t nch = (lines + CH - 1) / CH;
        std::vector<std::uint64_t> cl(nch), cf(nch);
        const unsigned char* a = arena_.data();
        par_for(nch, threads, [&](std::size_t c) {
            std::uint64_t l = mix64(c), f = l;
            for (std::size_t i = c * CH; i < std::min(lines, (c + 1) * CH); ++i) {
                const unsigned char* L = a + i * 64;
                const unsigned skip = lk_.empty() ? 8 : lk_[i] == LK_LINE ? 4 : lk_[i] == LK_CLINE ? 3 : 8;
                for (unsigned w = 0; w < 8; ++w) {
                    std::uint64_t x; std::memcpy(&x, L + 8 * w, 8);
                    if (w < skip) l = mix64(l ^ x);
                    f = mix64(f ^ x);
                }
            }
            cl[c] = l; cf[c] = f;
        });
        for (std::size_t c = 0; c < nch; ++c) { lh = mix64(lh ^ cl[c]); fh = mix64(fh ^ cf[c]); }
        lhash_ = lh; fhash_ = fh;
    }
};

} // namespace splice
