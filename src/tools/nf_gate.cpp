// nf_gate: does the NFL-style flow pay on this key set, and can a few-ms test on a key sample tell in advance?
// For each case (the whole SOSD file and/or contiguous windows of it) it builds SCALE-LI cells B and N (= B + the
// flow, --flow-bypass 1 --flow-cost 0, exactly the GRE facade's cell flags), replays the same uniformly sampled
// existing keys through Index::find with exact work counters, and prints one JSON line with the counters, memory,
// build time, the ground-truth flow benefit and candidate gate scores on m-key samples (evenly spaced ranks).
// Counts only: build and score times are reported where the tool ran (efficiency cores), never as GRE timings.
//
// Ground truth (per lookup, from the counters; the lookup path is index.hpp Index::find):
//   D = 2 root_probes + coordinate_probes + fence_probes + key_at_calls + delta_probes
//   root probe:   ok(i) reads regions_[i] (the pointer slot) and then regions_[i]->low_fence through it: two loads,
//                 the second addressed by the first, and the next probe's index depends on the comparison -> 2.
//   coordinate / fence probe: one BlockDescriptor (64 B) picked by the previous comparison -> 1 each.
//   key_at:       one arena read at a position picked by the previous comparison -> 1.
//   delta probe:  1 (0 here: read-only, the delta is empty).
//   transform_calls: compute on the dependent path (2x2 tanh net, weights in one or two cached lines), no new
//   memory -> 0 in D, as N's own --flow-cost 0; D_net charges 1 probe-equivalent per call as the sensitivity.
//   Distinct 64-B lines per lookup (QueryStats::track_lines) are reported next to D as the memory-footprint view.
// benefit = (D_B - D_N) / D_B; benefit_net uses D_net; mem_rel = (bytes_N + flow bytes - bytes_B) / bytes_B.
// use_flow = benefit_net > 0 and mem_rel < 0.01.
//
//   nf_gate --data SOSD --flow FLOW --name NAME [--windows 8] [--window-keys 12500000] [--full 0|1]
//           [--lookups 1000000] [--samples 1024,4096,16384,65536] [--threads 4] [--seed 42] [--region-keys 4096]
#include "scaleli/cli.hpp"
#include "scaleli/hardness.hpp"
#include "scaleli/workload.hpp"  // mix64
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>

using namespace scaleli;
namespace {
using Clock = std::chrono::steady_clock;
double ns_since(Clock::time_point t) { return std::chrono::duration<double, std::nano>(Clock::now() - t).count(); }
std::string num(double v) { if (!std::isfinite(v)) return "null"; std::ostringstream o; o << std::setprecision(10) << v; return o.str(); }

std::map<std::string, std::string> parse(int argc, char** argv) {
    const std::set<std::string> allowed = {"data", "flow", "name", "windows", "window-keys", "full", "lookups", "samples", "threads", "seed", "region-keys", "only-window"};
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; ++i) {
        std::string x = argv[i]; if (x.rfind("--", 0) != 0 || i + 1 >= argc) throw std::invalid_argument("usage: --key value");
        x = x.substr(2); if (!allowed.count(x)) throw std::invalid_argument("unknown option: " + x); a[x] = argv[++i];
    }
    for (const char* k : {"data", "flow", "name"}) if (!a.count(k)) throw std::invalid_argument(std::string("--") + k + " is required");
    return a;
}
std::size_t get_n(const std::map<std::string, std::string>& a, const std::string& k, std::size_t d) { auto it = a.find(k); return it == a.end() ? d : std::stoull(it->second); }

// SOSD uint64: 8-byte little-endian count, then the keys (the Mac copies are sorted; checked, duplicates dropped).
std::vector<Key> read_sosd(const std::string& path) {
    std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot open " + path);
    std::uint64_t n = 0; in.read(reinterpret_cast<char*>(&n), 8);
    std::vector<Key> k(n); in.read(reinterpret_cast<char*>(k.data()), std::streamsize(n * 8));
    if (!in) throw std::runtime_error("short read: " + path);
    if (!std::is_sorted(k.begin(), k.end())) throw std::runtime_error("keys not sorted (use the .sorted copy): " + path);
    k.erase(std::unique(k.begin(), k.end()), k.end());
    return k;
}

// The GRE facade's cell flags (scaleli_gre.cpp cell_flags) for B and N, through scaleli_bench's own parser.
Config cell_config(bool flow, const std::string& flow_path, unsigned threads, std::size_t region_keys, std::optional<FlowTransform>& holder) {
    std::vector<std::string> v = {"x", "--policy", "min_bytes", "--routing", "rank", "--root", "model", "--build-threads", std::to_string(threads),
                                  "--region-keys", std::to_string(region_keys)};
    if (flow) for (const char* s : {"--flow", flow_path.c_str(), "--flow-bypass", "1", "--flow-cost", "0"}) v.push_back(s);
    std::vector<char*> argv; for (auto& s : v) argv.push_back(s.data());
    return config_from_args(Args(int(argv.size()), argv.data()), holder);
}

struct CellResult {
    double build_ns = 0, replay_ns = 0; std::uint64_t bytes = 0, found = 0, digest = 0; QueryStats s; Index::Learnability l; std::size_t regions = 0;
    double per(std::uint64_t c, std::size_t n) const { return double(c) / double(n); }
};
CellResult run_cell(std::span<const Key> keys, const Config& cfg, std::span<const Key> lookups) {
    CellResult r;
    {
        std::vector<Record> rows(keys.size()); for (std::size_t i = 0; i < keys.size(); ++i) rows[i] = {keys[i], mix64(i)};
        Index ix(cfg); const auto t = Clock::now(); ix.bulk_load(std::move(rows)); r.build_ns = ns_since(t);
        r.bytes = ix.memory().accounted_bytes(); r.l = ix.learnability(); r.regions = ix.region_count();
        r.s.track_lines = true; const auto t2 = Clock::now();
        for (auto k : lookups) { const auto v = ix.find(k, &r.s); r.s.new_operation(); r.found += v.has_value();
            r.digest = mix64(r.digest ^ (v ? mix64(*v) ^ mix64(k) : 0)); }  // the facade's replay_reads digest
        r.replay_ns = ns_since(t2);
        r.l.joint = nullptr;  // points into ix, which dies here
    }
    return r;
}
double dep_cost(const QueryStats& s, std::size_t n) {
    return double(2 * s.root_probes + s.coordinate_probes + s.fence_probes + s.key_at_calls + s.delta_probes) / double(n);
}
std::string cell_json(const CellResult& r, std::size_t n, const Config& cfg) {
    const auto& s = r.s; std::ostringstream o; o << std::setprecision(10);
    o << "{\"build_ns\":" << r.build_ns << ",\"replay_ns_ecore\":" << r.replay_ns << ",\"accounted_bytes\":" << r.bytes
      << ",\"flow_bytes\":" << (cfg.flow ? cfg.flow->bytes() : 0) << ",\"regions\":" << r.regions << ",\"flow_regions\":" << r.l.flow_regions
      << ",\"root\":\"" << (!r.l.root_model ? "binary" : r.l.root_flow ? "flow" : "raw") << "\""
      << ",\"root_probes_fit\":{\"binary\":" << r.l.root_probes_binary << ",\"raw\":" << r.l.root_probes_raw << ",\"flow\":" << r.l.root_probes_flow << "}"
      << ",\"tail_conflicts_raw_mean\":" << r.l.tail_conflicts_raw_mean << ",\"tail_conflicts_flow_mean\":" << r.l.tail_conflicts_flow_mean
      << ",\"found\":" << r.found << ",\"digest\":" << r.digest << ",\"per_lookup\":{"
      << "\"root_probes\":" << r.per(s.root_probes, n) << ",\"coordinate_probes\":" << r.per(s.coordinate_probes, n)
      << ",\"fence_probes\":" << r.per(s.fence_probes, n) << ",\"delta_probes\":" << r.per(s.delta_probes, n)
      << ",\"key_at_calls\":" << r.per(s.key_at_calls, n) << ",\"decoded_keys\":" << r.per(s.decoded_keys, n)
      << ",\"codec_bytes_examined\":" << r.per(s.codec_bytes_examined, n) << ",\"block_routes\":" << r.per(s.block_routes, n)
      << ",\"correction_distance\":" << r.per(s.correction_distance, n) << ",\"transform_calls\":" << r.per(s.transform_calls, n)
      << ",\"cache_lines\":" << r.per(s.cache_lines, n) << ",\"D\":" << dep_cost(s, n) << "},\"max_correction_distance\":" << s.max_correction_distance
      << ",\"lines_overflow\":" << s.lines_overflow << "}";
    return o.str();
}

// ---------------------------------------------------------------- gate scores on an m-key sample
// KL(p || uniform) of the feature histogram over [min, max] with `bins` equal-width bins (max log bins).
double kl_uniform(std::span<const double> f, std::size_t bins) {
    const auto [lo, hi] = std::minmax_element(f.begin(), f.end()); const double a = *lo, w = *hi - *lo; if (!(w > 0)) return std::log(double(bins));
    std::vector<std::uint32_t> c(bins, 0);
    for (double x : f) ++c[std::min(bins - 1, std::size_t((x - a) / w * double(bins)))];
    double kl = 0; const double m = double(f.size());
    for (auto x : c) if (x) { const double p = x / m; kl += p * std::log(p * double(bins)); }
    return kl;
}
// Sorted feature -> W1 (mean |quantile - uniform quantile|) and KS (max) against uniform on [min, max].
std::pair<double, double> w1_ks(std::span<const double> z) {
    const double a = z.front(), w = z.back() - z.front(); if (!(w > 0)) return {0.5, 1};
    const std::size_t m = z.size(); double s = 0, k = 0;
    for (std::size_t i = 0; i < m; ++i) { const double d = std::abs((z[i] - a) / w - double(i) / double(m - 1)); s += d; k = std::max(k, d); }
    return {s / double(m), k};
}
// Share of log-gap variance in the slow trend (centered moving mean over w gaps; top 0.1% clipped as spikes),
// the repo's F05 decomposition (main@ca8d80e docs/figures/slides F05) at sample scale.
double trend_share(std::span<const double> z) {
    const std::size_t g = z.size() - 1; if (g < 16) return 0;
    std::vector<double> lg(g); double minpos = INFINITY;
    for (std::size_t i = 0; i < g; ++i) { const double d = z[i + 1] - z[i]; if (d > 0) minpos = std::min(minpos, d); }
    if (!std::isfinite(minpos)) return 0;
    for (std::size_t i = 0; i < g; ++i) lg[i] = std::log(std::max(z[i + 1] - z[i], minpos));
    std::vector<double> srt = lg; const auto q = std::size_t(0.999 * double(g - 1)); std::nth_element(srt.begin(), srt.begin() + std::ptrdiff_t(q), srt.end());
    const double cap = srt[q]; for (auto& x : lg) x = std::min(x, cap);
    const std::size_t w = std::max<std::size_t>(8, g / 64), h = w / 2;
    std::vector<double> pre(g + 1, 0); for (std::size_t i = 0; i < g; ++i) pre[i + 1] = pre[i] + lg[i];
    double mean = pre[g] / double(g), vt = 0, va = 0;
    for (std::size_t i = 0; i < g; ++i) { const std::size_t lo = i > h ? i - h : 0, hi = std::min(g, i + h + 1);
        const double mm = (pre[hi] - pre[lo]) / double(hi - lo); vt += (mm - mean) * (mm - mean); va += (lg[i] - mean) * (lg[i] - mean); }
    return va > 0 ? vt / va : 0;
}
// Index::locate_from_prediction over n fences with ok(i) = (i <= t), table-free (B and N have no virtual fences).
std::size_t root_count(double y, std::size_t n, std::size_t t) {
    std::size_t c = 0; auto ok = [&](std::size_t i) { ++c; return i <= t; };
    auto last_true = [&](std::size_t lo, std::size_t hi) { while (hi - lo > 1) { const auto m = lo + (hi - lo) / 2; if (ok(m)) lo = m; else hi = m; } return lo; };
    const std::size_t p = y <= 0 ? 0 : (y >= double(n - 1) ? n - 1 : std::size_t(y));
    if (ok(p)) { std::size_t lo = p, step = 1, hi = p + 1; while (hi < n && ok(hi)) { lo = hi; step *= 2; hi = std::min(n, p + step); } (void)last_true(lo, hi); }
    else { std::size_t hi = p, step = 1, lo = p > step ? p - step : 0; while (lo > 0 && !ok(lo)) { hi = lo; step *= 2; lo = p > step ? p - step : 0; } (void)last_true(lo, hi); }
    return c;
}
// Root proxy: fit_root's least-squares line from feature to region index, fitted on q evenly spaced region fences
// (first key of region j, target j) as the build does, then the real exponential + binary correction counted for each
// sample key. Fences, not sample keys, carry the fit: fb's 21 top outliers sit inside its last region, and a fit
// through them (the sample's last key) more than doubles the predicted root probes. Returns probes per lookup.
double root_proxy(std::span<const double> ff, std::span<const std::size_t> fj, std::span<const double> f, std::span<const std::size_t> rank,
                  std::size_t region_keys, std::size_t regions) {
    long double mx = 0, my = 0, xx = 0, xy = 0;
    for (std::size_t i = 0; i < ff.size(); ++i) { const long double x = ff[i], y = (long double)fj[i], k = (long double)(i + 1);
        const long double dx = x - mx, dy = y - my; mx += dx / k; my += dy / k; xx += dx * (x - mx); xy += dx * (y - my); }
    const long double sl = xx > 0 ? xy / xx : 0, ic = my - sl * mx; std::size_t c = 0;
    for (std::size_t i = 0; i < f.size(); ++i) c += root_count(double(sl * f[i] + ic), regions, rank[i] / region_keys);
    return double(c) / double(f.size());
}
double binary_root(std::size_t n) { return std::ceil(std::log2(double(std::max<std::size_t>(n, 2)))); }

// Best of 3. The asm barriers keep the pure score computation between the two clock reads (without them the
// compiler may sink it past the second read or hoist it above the first: 0-ns scores).
struct Timed { double v = 0, ns = 0; };
template<class F> Timed timed(F f) {
    Timed t; t.ns = INFINITY;
    for (int r = 0; r < 3; ++r) { asm volatile("" ::: "memory"); const auto c = Clock::now(); double v = f(); asm volatile("" : "+r,m"(v) : : "memory");
        t.ns = std::min(t.ns, ns_since(c)); t.v = v; }
    return t;
}

std::string gate_json(std::span<const Key> keys, std::size_t m, const Config& cb, const Config& cn, std::size_t region_keys) {
    const std::size_t n = keys.size(); m = std::min(m, n);
    std::ostringstream o; o << std::setprecision(8);
    std::vector<std::size_t> rank(m); std::vector<Key> s(m); std::vector<double> x(m), z(m), zs;
    const auto t0 = Clock::now();
    for (std::size_t i = 0; i < m; ++i) { rank[i] = m > 1 ? std::size_t((unsigned __int128)i * (n - 1) / (m - 1)) : 0; s[i] = keys[rank[i]]; }
    const double sample_ns = ns_since(t0);
    for (std::size_t i = 0; i < m; ++i) x[i] = double(s[i] - s[0]);  // exact u64 difference before the cast
    const auto t1 = Clock::now(); for (std::size_t i = 0; i < m; ++i) z[i] = (*cn.flow)(s[i]); const double flow_ns = ns_since(t1);
    const auto t2 = Clock::now(); zs = z; std::sort(zs.begin(), zs.end()); const double sort_ns = ns_since(t2);
    const bool mono = std::is_sorted(z.begin(), z.end());
    o << "{\"m\":" << m << ",\"sample_ns\":" << sample_ns << ",\"flow_ns\":" << flow_ns << ",\"flow_sort_ns\":" << sort_ns << ",\"flow_monotone\":" << (mono ? "true" : "false");
    // (a) KL to uniform at several bin scales
    o << ",\"kl\":{";
    bool first = true;
    for (std::size_t b : {16, 64, 256, 1024, 4096}) {
        if (b * 4 > m) continue;
        const auto kr = timed([&] { return kl_uniform(x, b); }), kf = timed([&] { return kl_uniform(z, b); });
        o << (first ? "" : ",") << '"' << b << "\":{\"raw\":" << kr.v << ",\"flow\":" << kf.v << ",\"diff\":" << kr.v - kf.v << ",\"ns\":" << kr.ns + kf.ns << "}"; first = false;
    }
    o << "}";
    // (b) trend share of log-gap variance
    { const auto tr = timed([&] { return trend_share(x); }), tf = timed([&] { return trend_share(zs); });
      o << ",\"trend_share\":{\"raw\":" << tr.v << ",\"flow\":" << tf.v << ",\"diff\":" << tr.v - tf.v << ",\"ns\":" << tr.ns + tf.ns << "}"; }
    // (c) NFL's rule: tail conflict degree under one linear model, raw vs flow (keep the flow at >= 10% fewer)
    { std::vector<double> xn(m); const double w = std::max(1.0, x.back()); for (std::size_t i = 0; i < m; ++i) xn[i] = x[i] / w;
      const auto tr = timed([&] { return double(tail_conflict_degree(xn)); }), tf = timed([&] { return double(tail_conflict_degree(zs)); });
      const bool keep = tf.v < tr.v && tr.v - tf.v >= cn.flow_min_gain * tr.v;
      o << ",\"tail_conflict\":{\"raw\":" << tr.v << ",\"flow\":" << tf.v << ",\"gain\":" << (tr.v > 0 ? (tr.v - tf.v) / tr.v : 0) << ",\"keep\":" << (keep ? "true" : "false") << ",\"ns\":" << tr.ns + tf.ns << "}"; }
    // (d) eps-PLA segments (eps in sample ranks), raw vs flow
    o << ",\"pla\":{";
    first = true;
    for (std::size_t e : {16, 64}) {
        const auto pr = timed([&] { return double(hardness::pla_segments<Key>(s, e)); }), pf = timed([&] { return double(hardness::pla_segments<double>(zs, e)); });
        o << (first ? "" : ",") << '"' << e << "\":{\"raw\":" << pr.v << ",\"flow\":" << pf.v << ",\"ratio\":" << pf.v / pr.v << ",\"ns\":" << pr.ns + pf.ns << "}"; first = false;
    }
    o << "}";
    // (e1) W1 and KS to uniform, raw vs flow (sorted features)
    { double w1r = 0, ksr = 0, w1f = 0, ksf = 0;
      const auto tr = timed([&] { auto p = w1_ks(x); w1r = p.first; ksr = p.second; return p.first + p.second; }), tf = timed([&] { auto p = w1_ks(zs); w1f = p.first; ksf = p.second; return p.first + p.second; });
      o << ",\"w1\":{\"raw\":" << w1r << ",\"flow\":" << w1f << ",\"diff\":" << w1r - w1f << "},\"ks\":{\"raw\":" << ksr << ",\"flow\":" << ksf << ",\"diff\":" << ksr - ksf << "},\"w1ks_ns\":" << tr.ns + tf.ns; }
    // (e2) root proxy: expected root probes raw vs flow vs binary at this case's region count; the build keeps the cheapest
    const std::size_t regions = (n + region_keys - 1) / region_keys;
    const std::size_t q = std::max<std::size_t>(2, std::min(m, regions));
    std::vector<std::size_t> fj(q); std::vector<Key> fk(q); std::vector<double> fx(q), fz(q);
    const auto rr = timed([&] {
        for (std::size_t i = 0; i < q; ++i) { fj[i] = std::size_t((unsigned __int128)i * (regions - 1) / (q - 1)); fk[i] = keys[fj[i] * region_keys]; fx[i] = double(fk[i] - s[0]); }
        return root_proxy(fx, fj, x, rank, region_keys, regions); });
    const auto rf = timed([&] { for (std::size_t i = 0; i < q; ++i) fz[i] = (*cn.flow)(fk[i]); return root_proxy(fz, fj, z, rank, region_keys, regions); });
    const double bin = binary_root(regions), root_b = std::min(rr.v, bin), root_n = std::min(root_b, rf.v);
    o << ",\"root_proxy\":{\"raw\":" << rr.v << ",\"flow\":" << rf.v << ",\"binary\":" << bin << ",\"B\":" << root_b << ",\"N\":" << root_n << ",\"ns\":" << rr.ns + rf.ns << "}";
    // (e3) mini build: contiguous region-size chunks at evenly spaced offsets (m keys in all), each rebuilt as a real
    // Region under B and N (N's tail-conflict bypass included) and probed with its own keys (uniform lookups hit every
    // key of a region equally). Gives the region part of D exactly for those regions.
    {
        const std::size_t chunk = std::min(region_keys, m), k = std::max<std::size_t>(1, m / chunk);
        // Policy raw: under rank routing slot_begin, fences, the flow decision and the key_at count do not depend on
        // the codec, so the counts equal min_bytes' and the codec trials (most of a rebuild) are skipped.
        Config rb_cfg = cb, rn_cfg = cn; rb_cfg.policy = rn_cfg.policy = Policy::Raw;
        double regB = 0, regN = 0; std::size_t flow_chunks = 0, probes = 0; const auto t = Clock::now();
        std::vector<Record> rows(chunk);
        for (std::size_t c = 0; c < k; ++c) {
            const std::size_t off = k > 1 ? std::size_t((unsigned __int128)c * (n - chunk) / (k - 1)) : (n - chunk) / 2;
            for (std::size_t i = 0; i < chunk; ++i) rows[i] = {keys[off + i], i};
            Region rb, rn; rb.rebuild(rows, rb_cfg); rn.rebuild(rows, rn_cfg); flow_chunks += rn.flow_used;
            QueryStats sb, sn;
            for (std::size_t i = 0; i < chunk; ++i) { (void)rb.find(rows[i].first, rb_cfg, &sb); (void)rn.find(rows[i].first, rn_cfg, &sn); }
            regB += double(sb.coordinate_probes + sb.fence_probes + sb.key_at_calls); regN += double(sn.coordinate_probes + sn.fence_probes + sn.key_at_calls);
            probes += chunk;
        }
        const double ns = ns_since(t); regB /= double(probes); regN /= double(probes);
        const double DB = 2 * root_b + regB, DN = 2 * root_n + regN;
        o << ",\"mini\":{\"chunks\":" << k << ",\"chunk_keys\":" << chunk << ",\"flow_chunks\":" << flow_chunks << ",\"region_B\":" << regB << ",\"region_N\":" << regN
          << ",\"ns\":" << ns << "},\"pred\":{\"D_B\":" << DB << ",\"D_N\":" << DN << ",\"benefit\":" << (DB - DN) / DB
          << ",\"ns\":" << rr.ns + rf.ns + ns + flow_ns + sample_ns << "}";
    }
    o << "}";
    return o.str();
}

void run_case(const std::string& name, const std::string& kind, int index, std::size_t offset, std::span<const Key> keys, const Config& cb, const Config& cn,
              const std::string& flow_path, std::size_t lookups_n, std::uint64_t seed, const std::vector<std::size_t>& samples, std::size_t region_keys, unsigned threads) {
    const auto t = Clock::now();
    std::mt19937_64 rng(seed ^ mix64(std::uint64_t(index + 1))); std::uniform_int_distribution<std::size_t> u(0, keys.size() - 1);
    std::vector<Key> look(lookups_n); for (auto& k : look) k = keys[u(rng)];
    const auto B = run_cell(keys, cb, look);
    std::fprintf(stderr, "[nf_gate] %s %s %d B build %.1f s\n", name.c_str(), kind.c_str(), index, B.build_ns / 1e9);
    const auto N = run_cell(keys, cn, look);
    std::fprintf(stderr, "[nf_gate] %s %s %d N build %.1f s\n", name.c_str(), kind.c_str(), index, N.build_ns / 1e9);
    const std::size_t n = look.size();
    const double DB = dep_cost(B.s, n), DN = dep_cost(N.s, n), DNn = DN + double(N.s.transform_calls) / double(n), DBn = DB + double(B.s.transform_calls) / double(n);
    const double LB = double(B.s.cache_lines) / double(n), LN = double(N.s.cache_lines) / double(n);
    const double mem = (double(N.bytes) + double(cn.flow->bytes()) - double(B.bytes)) / double(B.bytes);
    const double bnet = (DBn - DNn) / DBn;
    std::ostringstream o; o << std::setprecision(10);
    o << "{\"dataset\":\"" << name << "\",\"case\":\"" << kind << "\",\"window\":" << index << ",\"offset\":" << offset << ",\"keys\":" << keys.size()
      << ",\"key_min\":" << keys.front() << ",\"key_max\":" << keys.back() << ",\"flow_file\":\"" << flow_path << "\",\"region_keys\":" << region_keys
      << ",\"build_threads\":" << threads << ",\"lookups\":" << n << ",\"lookup_seed\":" << seed
      << ",\"digest_match\":" << (B.digest == N.digest ? "true" : "false") << ",\"B\":" << cell_json(B, n, cb) << ",\"N\":" << cell_json(N, n, cn)
      << ",\"gt\":{\"D_B\":" << DB << ",\"D_N\":" << DN << ",\"benefit\":" << (DB - DN) / DB << ",\"benefit_net\":" << bnet
      << ",\"lines_B\":" << LB << ",\"lines_N\":" << LN << ",\"benefit_lines\":" << (LB - LN) / LB
      << ",\"d_root\":" << 2 * (double(B.s.root_probes) - double(N.s.root_probes)) / double(n)
      << ",\"d_fence\":" << (double(B.s.fence_probes) - double(N.s.fence_probes)) / double(n)
      << ",\"d_coordinate\":" << (double(B.s.coordinate_probes) - double(N.s.coordinate_probes)) / double(n)
      << ",\"mem_rel\":" << mem << ",\"use_flow\":" << (bnet > 0 && mem < 0.01 ? "true" : "false") << "},\"gate\":{";
    for (std::size_t i = 0; i < samples.size(); ++i) o << (i ? "," : "") << '"' << samples[i] << "\":" << gate_json(keys, samples[i], cb, cn, region_keys);
    o << "},\"case_ns\":" << num(ns_since(t)) << "}";
    std::cout << o.str() << std::endl;
    if (B.found != n || N.found != n) throw std::runtime_error("lookups not all found");
}
}  // namespace

int main(int argc, char** argv) { try {
    const auto a = parse(argc, argv);
    const std::string name = a.at("name"), flow_path = a.at("flow");
    const unsigned threads = unsigned(get_n(a, "threads", 4));
    const std::size_t region_keys = get_n(a, "region-keys", 4096), windows = get_n(a, "windows", 8), wkeys = get_n(a, "window-keys", 12500000),
                      lookups = get_n(a, "lookups", 1000000), seed = get_n(a, "seed", 42);
    const long only = a.count("only-window") ? long(get_n(a, "only-window", 0)) : -1;
    std::vector<std::size_t> samples; { std::stringstream ss(a.count("samples") ? a.at("samples") : "1024,4096,16384,65536"); for (std::string t; std::getline(ss, t, ',');) if (!t.empty()) samples.push_back(std::stoull(t)); }
    std::optional<FlowTransform> fb, fn; const Config cb = cell_config(false, flow_path, threads, region_keys, fb), cn = cell_config(true, flow_path, threads, region_keys, fn);
    const auto t = Clock::now(); const auto keys = read_sosd(a.at("data"));
    std::fprintf(stderr, "[nf_gate] %s: %zu unique keys, read %.1f s\n", name.c_str(), keys.size(), ns_since(t) / 1e9);
    if (get_n(a, "full", 0)) run_case(name, "full", -1, 0, keys, cb, cn, flow_path, lookups, seed, samples, region_keys, threads);
    if (windows && wkeys <= keys.size())
        for (std::size_t w = 0; w < windows; ++w) {
            if (only >= 0 && long(w) != only) continue;
            const std::size_t off = windows > 1 ? std::size_t((unsigned __int128)w * (keys.size() - wkeys) / (windows - 1)) : 0;
            run_case(name, "window", int(w), off, std::span<const Key>(keys).subspan(off, wkeys), cb, cn, flow_path, lookups, seed, samples, region_keys, threads);
        }
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "nf_gate: %s\n", e.what()); return 2; } }
