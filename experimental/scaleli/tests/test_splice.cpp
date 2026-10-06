// SPLICE-H tests (C++20): exact lookups against a sorted reference, exact Count parity between get() and
// the build's plan walker, edge sizes around every entry capacity, extreme keys, exceptions, ties,
// determinism and perturbation stability. Counts only; nothing here depends on timing.
//   splice_tests DATA_SAMPLES_DIR          (prints "hash <sample> <layout_hash>" lines for tools/splice_xarch.sh)
#include "splice/build.hpp"
#include "scaleli/workload.hpp"
#include <cinttypes>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

using namespace splice;
static std::uint64_t assertions = 0;
#define CHECK(x) do{++assertions;if(!(x))throw std::runtime_error(std::string("CHECK failed: ")+ #x + " line " + std::to_string(__LINE__));}while(false)
template<class F> void must_throw(F f) { bool threw = false; try { f(); } catch (const std::invalid_argument&) { threw = true; } CHECK(threw); }

static std::string hex(std::uint64_t h) { char b[24]; std::snprintf(b, sizeof b, "0x%016" PRIx64, h); return b; }
static std::vector<std::uint64_t> load(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p);
    std::uint64_t n = 0; f.read(reinterpret_cast<char*>(&n), 8);
    std::vector<std::uint64_t> k(n); f.read(reinterpret_cast<char*>(k.data()), std::streamsize(8 * n));
    if (!f) throw std::runtime_error("short read " + p);
    return k;
}
static std::vector<std::uint64_t> payloads(const std::vector<std::uint64_t>& k) {
    std::vector<std::uint64_t> v(k.size());
    for (std::size_t i = 0; i < k.size(); ++i) v[i] = scaleli::mix64(k[i]);
    return v;
}
static bool contains(const std::vector<std::uint64_t>& k, std::uint64_t x) { return std::binary_search(k.begin(), k.end(), x); }

// Builds, then checks: every key returns its payload, `absent` absent keys miss, and get_impl<Count>
// over all keys equals the plan walker's totals exactly.
static void check_index(const Index& ix, const std::vector<std::uint64_t>& k, const std::vector<std::uint64_t>& v, std::size_t absent) {
    for (std::size_t i = 0; i < k.size(); ++i) { std::uint64_t o = 0; CHECK(ix.get(k[i], o)); CHECK(o == v[i]); }
    std::size_t probes = 0;
    for (std::uint64_t i = 0; probes < absent; ++i) {
        const std::uint64_t x = scaleli::mix64(i);
        if (contains(k, x)) continue;
        std::uint64_t o = 0; CHECK(!ix.get(x, o)); ++probes;
    }
    // Neighbours of stored keys exercise the clamps and the tie fallbacks with absent keys.
    for (std::size_t i = 0; i < k.size(); i += std::max<std::size_t>(1, k.size() / 4096)) {
        for (std::uint64_t x : {k[i] - 1, k[i] + 1}) { if (contains(k, x)) continue; std::uint64_t o = 0; CHECK(!ix.get(x, o)); }
    }
    if (!k.empty()) {
        Count c(ix.view());
        for (auto x : k) { std::uint64_t o; get_impl(ix.view(), x, o, c); }
        const CountTotals& w = ix.plan_counts();
        CHECK(c.core.t.lookups == k.size() && c.core.t.found == k.size());
        CHECK(c.core.t == w);
    }
}
static void build_check(Index& ix, const std::vector<std::uint64_t>& k, const std::string& args, std::size_t absent = 1000000) {
    const auto v = payloads(k);
    ix.build(Input{k.data(), v.data(), 1, k.size()}, parse_params(args, false));
    check_index(ix, k, v, absent);
}

static void samples(const std::string& dir) {
    static const char* ds[10] = {"books", "covid", "fb", "genome", "history", "libio", "osm", "planet", "stack", "wise"};
    for (const char* d : ds) for (const char* mode : {"uniform", "window"}) {
        const std::string name = std::string(d) + "_2M_" + mode + "_s42";
        const auto k = load(dir + "/" + name);
        for (const char* extra : {"", " mode=fenced cbar=24", " mode=direct"}) {
            Index ix;
            build_check(ix, k, std::string("k1=1024 sim=200000 warm=50000") + extra);
            if (!*extra) std::cout << "hash " << name << ' ' << hex(ix.layout_hash()) << '\n';
            if (std::string(extra).find("fenced") != std::string::npos) CHECK(ix.stats().direct_key_ppm == 0);
            if (std::string(extra).find("direct") != std::string::npos) CHECK(ix.stats().fenced_key_ppm == 0);
        }
    }
}

// One FENCED entry holding all n keys (k1=1, cbar >= n): depth follows the entry capacities exactly.
static unsigned expected_depth(std::size_t n, unsigned F, unsigned kpl) {
    const std::size_t cap = std::size_t(kpl) * (F + 1);
    unsigned d = 1;
    for (std::size_t c = n; c > cap; ++d) { const std::size_t k = std::min<std::size_t>(F + 1, (c + cap - 1) / cap); c = (c + k - 1) / k; }
    return d;
}
static void edge_sizes() {
    for (std::size_t n : {0u, 1u, 2u, 3u, 4u, 5u, 56u, 57u, 100u, 101u, 228u, 229u, 232u, 233u, 2500u, 2501u, 12996u, 12997u, 13224u, 13225u}) {
        std::vector<std::uint64_t> k(n);
        for (std::size_t i = 0; i < n; ++i) k[i] = 1000 + i * 1000 + scaleli::mix64(i) % 900;
        for (unsigned entry : {1u, 2u}) for (unsigned compact : {0u, 1u}) {
            Index ix;
            build_check(ix, k, "mode=fenced k1=1 exc=0 cbar=1048576 sim=20000 warm=5000 entry=" + std::to_string(entry) + " compact=" + std::to_string(compact), 20000);
            if (n) {
                const Stats& s = ix.stats();
                CHECK(s.segments == 1 && s.fenced_key_ppm == 1000000);
                unsigned deepest = 0;
                for (unsigned d = 0; d < 4; ++d) if (s.depth_hist[d]) deepest = d + 1;
                CHECK(deepest == expected_depth(n, entry == 2 ? 56 : 24, compact ? 5 : 4));
                if (compact) CHECK(s.compact_keys == n);
            }
        }
        Index ix;
        build_check(ix, k, "mode=direct k1=1 exc=0 sim=20000 warm=5000", 20000);
        if (n) CHECK(ix.stats().direct_key_ppm == 1000000);
    }
}

static void extreme_keys() {
    const std::uint64_t M = ~0ull;
    // Bulk path (n < 2E+2: no exceptions): 0 and 2^64-1 stored, spans of 2^64-1.
    for (auto k : {std::vector<std::uint64_t>{0, M}, std::vector<std::uint64_t>{0, 1, 5, 1ull << 63, M - 1, M}, std::vector<std::uint64_t>{M}, std::vector<std::uint64_t>{0}}) {
        for (const char* a : {"k1=4 sim=1000 warm=100", "mode=fenced k1=1 cbar=64 sim=1000 warm=100", "mode=direct k1=1 sim=1000 warm=100", "mode=fenced compact=1 entry=1 sim=1000 warm=100"}) {
            Index ix; build_check(ix, k, a, 10000);
            CHECK(ix.stats().exceptions == 0);
        }
    }
    // Exception path: dense bulk with 0 and 2^64-1 as outliers.
    std::vector<std::uint64_t> k{0};
    for (std::uint64_t i = 0; i < 10000; ++i) k.push_back((1ull << 40) + i * 7);
    k.push_back(M);
    Index ix; build_check(ix, k, "k1=64 sim=20000 warm=5000", 100000);
    CHECK(ix.stats().exceptions == 2);
    Index ex0; build_check(ex0, k, "k1=64 exc=0 sim=20000 warm=5000", 100000);
    CHECK(ex0.stats().exceptions == 0);
}

static void fb_outliers(const std::string& dir) {
    auto k = load(dir + "/fb_2M_window_s42");
    const std::uint64_t top = k.back();
    CHECK(top < 140000000000000ull);
    // 12 keys in [1.4e14, 1.13e15] and 9 keys 2.05e18 apart ending at 2^64-1.
    for (int i = 0; i < 12; ++i) k.push_back(140000000000000ull + std::uint64_t(i) * 89999999999999ull);
    for (int i = 8; i >= 0; --i) k.push_back(~0ull - std::uint64_t(i) * 2050000000000000000ull);
    std::sort(k.begin(), k.end());
    CHECK(std::adjacent_find(k.begin(), k.end()) == k.end());
    Index ix; build_check(ix, k, "k1=1024 sim=200000 warm=50000");
    CHECK(ix.stats().exceptions == 21);
    const std::string full = dir + "/../external/gre/fb";
    if (std::filesystem::exists(full)) {
        std::ifstream f(full, std::ios::binary);
        std::uint64_t n = 0; f.read(reinterpret_cast<char*>(&n), 8);
        std::vector<std::uint64_t> t(21);
        f.seekg(std::streamoff(8 + (n - 21) * 8)); f.read(reinterpret_cast<char*>(t.data()), 21 * 8);
        auto r = load(dir + "/fb_2M_window_s42");
        for (auto x : t) if (x > r.back()) r.push_back(x);
        std::sort(r.begin(), r.end()); r.erase(std::unique(r.begin(), r.end()), r.end());
        Index iy; build_check(iy, r, "k1=1024 sim=200000 warm=50000");
        std::cout << "fb real top-21 spliced into the window: exceptions " << iy.stats().exceptions << '\n';
        CHECK(iy.stats().exceptions == 21);
    } else std::cout << "SKIP fb real top-21 (data/external/gre/fb absent)\n";
}

static void holes_and_runs(const std::string& dir) {
    {   // osm: holes; the count-split router must go below the top table somewhere
        const auto k = load(dir + "/osm_2M_uniform_s42");
        Index ix; build_check(ix, k, "k1=4096 sim=200000 warm=50000");
        CHECK(ix.stats().router_nodes > 0);
    }
    {   // a run of 10^6 consecutive keys inside stack: the slope cap (one unit per key unit) is hit
        auto k = load(dir + "/stack_2M_uniform_s42");
        const std::uint64_t s0 = k[k.size() / 2] + 1;
        for (std::uint64_t i = 0; i < 1000000; ++i) k.push_back(s0 + i);
        std::sort(k.begin(), k.end()); k.erase(std::unique(k.begin(), k.end()), k.end());
        for (const char* a : {"k1=1024 sim=200000 warm=50000", "k1=1024 mode=fenced sim=200000 warm=50000"}) { Index ix; build_check(ix, k, a, 200000); }
    }
}

static void wide_and_ties() {
    {   // keys 2^30 apart: every leaf spans >= 2^32, so fences are shifted and compact is never eligible
        std::vector<std::uint64_t> k;
        for (std::uint64_t i = 0; i < 600; ++i) k.push_back(12345 + (i << 30));
        for (unsigned entry : {1u, 2u}) {
            Index ix; build_check(ix, k, "mode=fenced k1=1 cbar=100000 compact=1 sim=10000 warm=1000 entry=" + std::to_string(entry), 50000);
            CHECK(ix.stats().compact_keys == 0);
        }
    }
    // Dense keys plus one at 2^60 in one entry: shifted fences tie and the fallback walks back.
    for (std::size_t dense : {200u, 400u, 1000u}) {
        std::vector<std::uint64_t> k;
        for (std::uint64_t i = 0; i < dense; ++i) k.push_back(5000 + i * 3);
        k.push_back(1ull << 60);
        for (unsigned entry : {1u, 2u}) for (unsigned compact : {0u, 1u}) {
            Index ix; build_check(ix, k, "mode=fenced k1=1 exc=0 cbar=100000 sim=10000 warm=1000 entry=" + std::to_string(entry) + " compact=" + std::to_string(compact), 50000);
            CHECK(ix.stats().tie_lines > 0);
            CHECK(ix.plan_counts().tagdep[T_TIE_LINE] > 0);
        }
    }
}

static void errors() {
    std::vector<std::uint64_t> dup{1, 2, 2, 3}, desc{1, 3, 2}, ok{1, 2, 3};
    must_throw([&] { Index ix; ix.build(Input{dup.data(), nullptr, 1, dup.size()}, SpliceParams{}); });
    must_throw([&] { Index ix; ix.build(Input{desc.data(), nullptr, 1, desc.size()}, SpliceParams{}); });
    for (const char* bad : {"k1=0", "k1=70000", "nope=1", "mode=fast", "alpha=0.3", "w=3", "entry=3", "cap=22.05", "cap=x", "l2=1536",
                            "router_leaf=6", "exc=65", "outer=0", "k1", "=4", "k1=1O", "mode=AUTO", "tier1=pgm", "alpha=", "thp=1", "cbar=0"})
        must_throw([&] { parse_params(bad, false); });
    must_throw([&] { parse_params("thp=0", true); });
    CHECK(parse_params("thp=1", true).thp == 1 && parse_params("", true).thp == 1 && parse_params("thp=0", false).thp == 0);
    // Later tokens win; the canonical form round-trips.
    const auto p = parse_params("k1=4096 cbar=16 k1=8192 alpha=0.5,0.25 w=4,1 cap=16.5 compact=1 tier1=histtree mode=fenced", false);
    CHECK(p.k1 == 8192 && p.cbar == 16 && p.alpha_mask == 6 && p.w_mask == 5 && p.cap_tenths == 165 && p.tier1 == TIER1_HISTTREE);
    CHECK(format_params(parse_params(format_params(p), false)) == format_params(p));
    CHECK(format_params(parse_params(format_params(p, true), false), true) == format_params(p, true));
    Index ix; ix.build(Input{ok.data(), nullptr, 1, ok.size()}, parse_params("sim=1000 warm=10", false));
    for (auto x : ok) { std::uint64_t o = 0; CHECK(ix.get(x, o) && o == x); }   // val == nullptr: payload = key
}

static void determinism(const std::string& dir) {
    const auto k = load(dir + "/osm_2M_window_s42");
    const auto v = payloads(k);
    std::uint64_t lh = 0, fh = 0;
    for (const char* t : {"threads=1", "threads=16", "threads=16", "threads=3"}) {
        Index ix; ix.build(Input{k.data(), v.data(), 1, k.size()}, parse_params(std::string("k1=1024 sim=200000 warm=50000 ") + t, false));
        if (!lh) { lh = ix.layout_hash(); fh = ix.full_hash(); }
        CHECK(ix.layout_hash() == lh && ix.full_hash() == fh);
    }
    // Strided input (GRE's pair array) and a different payload set: same layout hash, different full hash.
    std::vector<std::uint64_t> pairs(2 * k.size());
    for (std::size_t i = 0; i < k.size(); ++i) { pairs[2 * i] = k[i]; pairs[2 * i + 1] = v[i] ^ 0xabcdef; }
    Index ip; ip.build(Input{pairs.data(), pairs.data() + 1, 2, k.size()}, parse_params("k1=1024 sim=200000 warm=50000", false));
    CHECK(ip.layout_hash() == lh && ip.full_hash() != fh);
    for (std::size_t i = 0; i < k.size(); i += 97) { std::uint64_t o = 0; CHECK(ip.get(k[i], o) && o == (v[i] ^ 0xabcdef)); }
}

// The simulator's input: the walker's per-key access traces must equal get_impl<Trace> on the built arena.
static void trace_parity(const std::string& dir) {
    for (const char* name : {"fb_2M_window_s42", "books_2M_uniform_s42"}) {
        const auto k = load(dir + "/" + name);
        const auto v = payloads(k);
        SpliceParams p = parse_params("k1=1024 sim=100000 warm=10000 cap=30", false);
        const KeySrc ks{k.data(), v.data(), 1, k.size()};
        Fit f = make_fit(ks, p, 16);
        std::vector<DirectCache> dc; std::vector<const DirectCache*> dp;
        for (unsigned q : p.alphas()) dc.push_back(eval_direct(f, q, 16));
        for (auto& d : dc) dp.push_back(&d);
        const FencedCache fc = eval_fenced(f, p.entry, p.cbar, 1, 16);
        const Cell c = make_cell(f, p, dp, &fc);
        const Draws dr = make_draws(k.size(), p.warm, p.sim, p.seed);
        const FixedPoint fx = fixed_point(c, &dr, 16);
        Arena ar; ar.map(fx.plan.arena_bytes, false);
        Line z{}; for (int t = 0; t < 4; ++t) { z.k[t] = f.lo; z.v[t] = v[f.bb]; }
        std::memcpy(ar.data(), &z, 64);
        fill_arena(c, fx.plan, ar.data(), nullptr, 16);
        ABuf<Seg16> seg(f.K() + 9); fill_records(c, fx.plan, seg.data());
        View view = f.rview(); view.seg = seg.data(); view.arena = ar.data(); view.entry_lines = p.entry; view.nfence = p.entry == 2 ? 56 : 24;
        CHECK(fx.traces.idx.size() > 50000);
        bool mixed[2] = {false, false};
        for (std::size_t s = 0; s < fx.traces.idx.size(); ++s) {
            Trace t(view); std::uint64_t o = 0;
            CHECK(get_impl(view, k[fx.traces.idx[s]], o, t) && o == v[fx.traces.idx[s]]);
            CHECK(t.core.n == fx.traces.off[s + 1] - fx.traces.off[s]);
            for (unsigned e = 0; e < t.core.n; ++e) {
                CHECK(t.core.ids[e] == fx.traces.ids[fx.traces.off[s] + e]);
                if ((t.core.ids[e] >> 56) == 4) mixed[((t.core.ids[e] >> 52) & 3) == REG_AD ? 0 : 1] = true;
            }
        }
        std::cout << name << ": " << fx.traces.idx.size() << " traces equal (DIRECT " << mixed[0] << ", FENCED " << mixed[1] << ")\n";
    }
}

// x -> lo + round((t + 1e-6 t^2)/(1 + 1e-6) * span), then +1 bumps (offline only; doubles never reach the index).
static std::vector<std::uint64_t> perturb(const std::vector<std::uint64_t>& k) {
    std::vector<std::uint64_t> o(k.size());
    const std::uint64_t lo = k.front(); const double span = double(k.back() - lo);
    for (std::size_t i = 0; i < k.size(); ++i) {
        const double t = double(k[i] - lo) / span, tp = (t + 1e-6 * t * t) / (1.0 + 1e-6);
        std::uint64_t x = lo + std::uint64_t(std::nearbyint(tp * span));
        if (i && x <= o[i - 1]) x = o[i - 1] + 1;
        o[i] = x;
    }
    return o;
}
static void perturbation(const std::string& dir) {
    for (const char* name : {"fb_2M_window_s42", "osm_2M_uniform_s42"}) {
        const auto k = load(dir + "/" + name), kp = perturb(k);
        Index a, b;
        build_check(a, k, "k1=1024 sim=400000 warm=100000", 10000);
        build_check(b, kp, "k1=1024 sim=400000 warm=100000", 10000);
        std::size_t moved = 0;
        for (std::size_t i = 0; i < k.size(); ++i) moved += k[i] != kp[i];
        CHECK(moved > k.size() / 2 && a.layout_hash() != b.layout_hash());
        const std::int64_t d = b.stats().ed_sel_milli - a.stats().ed_sel_milli;
        const std::int64_t m = std::int64_t(b.stats().direct_key_ppm) - std::int64_t(a.stats().direct_key_ppm);
        std::cout << name << " perturbed (" << moved << " keys moved): E[D] " << a.stats().ed_sel_milli << " -> " << b.stats().ed_sel_milli << " milli, DIRECT share "
                  << a.stats().direct_key_ppm << " -> " << b.stats().direct_key_ppm << " ppm\n";
        CHECK(d <= 10 && d >= -10);
        CHECK(m <= 20000 && m >= -20000);
    }
}

int main(int argc, char** argv) {
    try {
        const std::string dir = argc > 1 ? argv[1] : "data/samples";
        errors(); std::cout << "duplicates / descents / bad SPLICE_ARGS throw, params round-trip: PASS\n";
        edge_sizes(); std::cout << "edge sizes 0..13225 (leaf, inner, 2-level inner; entry 1/2, compact 0/1, DIRECT): PASS\n";
        extreme_keys(); std::cout << "keys 0 and 2^64-1 (bulk and exceptions): PASS\n";
        wide_and_ties(); std::cout << "line span >= 2^32 (shifted fences, no compact) / fence ties and fallbacks: PASS\n";
        fb_outliers(dir); std::cout << "fb outliers: 21 exceptions: PASS\n";
        holes_and_runs(dir); std::cout << "osm holes / 10^6-key run at gap 1: PASS\n";
        determinism(dir); std::cout << "thread-count and payload-independent determinism: PASS\n";
        trace_parity(dir); std::cout << "walker traces == get_impl<Trace>: PASS\n";
        perturbation(dir); std::cout << "perturbation stability (|dE[D]| <= 10 milli, DIRECT share within 0.02): PASS\n";
        samples(dir); std::cout << "20 samples x {auto, fenced, direct}: every key, 1M absent keys, exact Count parity: PASS\n";
        std::cout << "ASSERTIONS=" << assertions << " ALL TESTS PASSED (" << (SPLICE_SSE ? "SSE4.2" : "scalar") << " path)\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
