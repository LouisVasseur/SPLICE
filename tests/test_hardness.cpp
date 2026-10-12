// Tests for include/scaleli/hardness.hpp and the scaleli_hardness CLI.
// Usage: scaleli_hardness_tests [hardness_binary fixture_sosd flow_weights]
// Without the three paths the CLI check is skipped (unit checks still run).
#include "scaleli/hardness.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <random>
#include <string>

using namespace scaleli;
using namespace scaleli::hardness;
static std::uint64_t assertions = 0;
#define CHECK(x) do { ++assertions; if (!(x)) throw std::runtime_error(std::string("CHECK failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)
static const std::size_t EPS[] = {32, 4096};

// Independent PLA oracle: minimum segment count by dynamic programming over the same
// point sequence PGM uses ((x_i, i) plus the sentinel (x_last + 1, n)) with PGM's strips
// [max(0, y - eps), y + eps]. A segment is feasible iff the (slope, intercept) polygon
// cut out by its strips is nonempty; with two or more distinct x it is bounded, so it
// is nonempty iff a pairwise boundary intersection satisfies every constraint.
struct Pt { long double x, lo, hi; };
static bool feasible(const std::vector<Pt>& p, std::size_t s, std::size_t e) {
    if (e - s <= 2) return true;
    const long double tol = 1e-9L;
    for (std::size_t i = s; i < e; ++i) for (int bi = 0; bi < 2; ++bi) for (std::size_t j = i + 1; j < e; ++j) for (int bj = 0; bj < 2; ++bj) {
        const long double ci = bi ? p[i].hi : p[i].lo, cj = bj ? p[j].hi : p[j].lo;
        const long double a = (ci - cj) / (p[i].x - p[j].x), b = ci - a * p[i].x; bool ok = true;
        for (std::size_t k = s; k < e && ok; ++k) { const long double v = a * p[k].x + b; ok = v >= p[k].lo - tol && v <= p[k].hi + tol; }
        if (ok) return true;
    }
    return false;
}
static std::size_t oracle_segments(const std::vector<std::uint64_t>& keys, std::size_t eps) {
    std::vector<Pt> p; const std::size_t n = keys.size();
    auto strip = [&](long double x, std::size_t y) { p.push_back({x, y <= eps ? 0.0L : static_cast<long double>(y - eps), static_cast<long double>(y + eps)}); };
    for (std::size_t i = 0; i < n; ++i) strip(static_cast<long double>(keys[i]), i);
    strip(static_cast<long double>(keys[n - 1]) + 1, n);
    const std::size_t m = p.size(); std::vector<std::size_t> dp(m + 1, m + 1); dp[0] = 0;
    for (std::size_t i = 0; i < m; ++i) for (std::size_t j = i + 1; j <= m && feasible(p, i, j); ++j) dp[j] = std::min(dp[j], dp[i] + 1); // feasibility is monotone in j
    return dp[m];
}

void exactness_near_2_64() {
    // 1000 consecutive keys just below 2^64: every one of them rounds to the same double.
    std::vector<std::uint64_t> keys(1000); for (std::size_t i = 0; i < keys.size(); ++i) keys[i] = std::numeric_limits<std::uint64_t>::max() - 999 + i;
    const auto ls = least_squares<std::uint64_t>(keys); CHECK(ls.rmse < 1e-9); CHECK(ls.max_error < 1e-9); CHECK(std::fabs(ls.slope - 1) < 1e-12);
    CHECK(pla_segments<std::uint64_t>(keys, 0) == 1); CHECK(pla_segments<std::uint64_t>(keys, 32) == 1);
    const auto m = fmcd_fit<std::uint64_t>(keys); CHECK(!m.fallback); CHECK(m.capacity == 6000); CHECK(m.gap == 5); CHECK(m.d == 1);
    CHECK(conflict_degree<std::uint64_t>(keys, m) == 1); // slots are ~6 apart, no collision
    CHECK(m.predict(keys[1]) == 1); CHECK(m.predict(keys[998]) == 5998); // LIPP maps keys[D] to 1 + 0.018 and keys[size-1-D] to L - 1 - 0.018 (the +1e-6 on U_T), floored
    CHECK(m.predict(keys.front()) < m.predict(keys.back()));
    std::vector<double> rounded(keys.begin(), keys.end()); // the same keys as doubles collapse completely
    std::size_t distinct = 1; for (std::size_t i = 1; i < rounded.size(); ++i) distinct += rounded[i] != rounded[i - 1]; CHECK(distinct == 1);
    CHECK(conflict_degree<double>(rounded, fmcd_fit<double>(rounded, default_ut_epsilon<double>(rounded))) == 1000);
    // A tiny dense window at the top of the key space, 2^64 - 1 included: sentinel x_last + 1 must not overflow.
    std::vector<std::uint64_t> top(50); for (std::size_t i = 0; i < top.size(); ++i) top[i] = std::numeric_limits<std::uint64_t>::max() - 49 + i;
    CHECK(pla_segments<std::uint64_t>(top, 1) == 1); CHECK(top.back() == std::numeric_limits<std::uint64_t>::max());
    // Full-range keys: 0, 2^63, 2^64 - 1 and neighbours; must not throw and must stay consistent.
    std::vector<std::uint64_t> wide = {0, 1, 2, std::uint64_t(1) << 63, (std::uint64_t(1) << 63) + 1, std::numeric_limits<std::uint64_t>::max() - 1, std::numeric_limits<std::uint64_t>::max()};
    CHECK(pla_segments<std::uint64_t>(wide, 4096) == 1); CHECK(pla_segments<std::uint64_t>(wide, 0) >= 2);
    const auto lw = least_squares<std::uint64_t>(wide); CHECK(lw.max_error > 0 && lw.max_error < 7);
}
void pla_cases() {
    // Perfect line: one segment for every eps >= 1 (eps = 0 also, because the sentinel lies on the line).
    std::vector<std::uint64_t> line(10000); for (std::size_t i = 0; i < line.size(); ++i) line[i] = 7 * i + 3;
    for (auto e : EPS) CHECK(pla_segments<std::uint64_t>(line, e) == 1);
    CHECK(pla_segments<std::uint64_t>(line, 1) == 1); CHECK(pla_segments<std::uint64_t>(line, 0) == 2); // sentinel (x+1, n) is 6/7 off the line
    // Staircase: 10 runs of 100 consecutive keys separated by jumps of 10^6. Each run needs its own
    // eps = 32 segment (a run at slope ~0 has error 50), while one line covers all of it at eps = 4096.
    std::vector<std::uint64_t> stairs; for (std::uint64_t r = 0; r < 10; ++r) for (std::uint64_t i = 0; i < 100; ++i) stairs.push_back(r * 1000000 + i);
    CHECK(pla_segments<std::uint64_t>(stairs, 32) == 10); CHECK(pla_segments<std::uint64_t>(stairs, 4096) == 1);
    CHECK(pla_segments<std::uint64_t>(stairs, 32) == oracle_segments(stairs, 32));
    // Double features give the same counts as their integral twins for small exact values.
    std::vector<double> stairs_d(stairs.begin(), stairs.end()); CHECK(pla_segments<double>(stairs_d, 32) == 10); CHECK(pla_segments<double>(stairs_d, 4096) == 1);
    // Duplicates follow PGM's rule: a run of equal keys is one point; (x + 1, rank) is added if there is room.
    std::vector<std::uint64_t> dup = {5, 5, 5, 9, 9, 20}; CHECK(pla_segments<std::uint64_t>(dup, 0) >= 1); CHECK(pla_segments<std::uint64_t>(dup, 4096) == 1);
    std::vector<double> dup_d = {5, 5, 5, 9, 9, 20}; CHECK(pla_segments<double>(dup_d, 4096) == 1);
    CHECK(pla_segments<std::uint64_t>(std::vector<std::uint64_t>{42}, 0) == 1); CHECK(pla_segments<std::uint64_t>(std::vector<std::uint64_t>{}, 0) == 0);
    // Differential against the DP oracle on random small instances.
    std::mt19937_64 rng(2026); std::size_t trials = 0;
    for (unsigned t = 0; t < 80; ++t) {
        const std::size_t n = 3 + rng() % 14; std::vector<std::uint64_t> k; std::uint64_t v = rng() % 20;
        for (std::size_t i = 0; i < n; ++i) { k.push_back(v); v += 1 + (rng() % 4 == 0 ? rng() % 200 : rng() % 6); }
        for (std::size_t e : {0, 1, 2, 5}) { CHECK(pla_segments<std::uint64_t>(k, e) == oracle_segments(k, e)); ++trials; }
    }
    CHECK(trials == 320);
    // Monotone in eps on a random uint64 set.
    std::vector<std::uint64_t> rnd(20000); for (auto& x : rnd) x = rng(); std::sort(rnd.begin(), rnd.end()); rnd.erase(std::unique(rnd.begin(), rnd.end()), rnd.end());
    const auto s32 = pla_segments<std::uint64_t>(rnd, 32), s4096 = pla_segments<std::uint64_t>(rnd, 4096); CHECK(s32 >= s4096); CHECK(s32 > 1);
}
void cd_cases() {
    // Obvious cluster: 50 consecutive keys, then 950 keys 100 apart starting at 10^6. FMCD (hand-traced):
    // D grows to 50, U_T = 89900/5998 + 1e-6, the cluster lands below slot 0 and is clamped there.
    std::vector<std::uint64_t> cluster; for (std::uint64_t i = 0; i < 50; ++i) cluster.push_back(i); for (std::uint64_t j = 0; j < 950; ++j) cluster.push_back(1000000 + 100 * j);
    const auto m = fmcd_fit<std::uint64_t>(cluster); CHECK(!m.fallback); CHECK(m.d == 50); CHECK(m.capacity == 6000);
    CHECK(std::fabs(m.ut - (89900.0 / 5998 + 1e-6)) < 1e-9); CHECK(std::fabs(m.a - 1 / m.ut) < 1e-9);
    CHECK(m.predict(cluster[49]) == 0); CHECK(m.predict(cluster[50]) == 1); CHECK(m.predict(cluster.back()) < 6000);
    CHECK(conflict_degree<std::uint64_t>(cluster, m) == 50);
    // Fallback: 600 sparse keys 10^4 apart, then 400 consecutive keys. For every D <= size/3 the window
    // gets stuck at the first dense gap (1 < U_T, since U_T >= (6e6 - 3.33e6) / 5998 ~ 445), so LIPP fits the
    // tertile midpoints: mid1 = (keys[333] + keys[334]) / 2 -> 2001 and mid2 = (keys[666] + keys[667]) / 2 -> 3999.
    // The dense cluster straddles mid2 by index: keys[600..666] land just below 3999 (slot 3998, 67 keys),
    // keys[667..999] just above (slot 3999, 333 keys); the 67 lowest sparse keys clamp to slot 0.
    std::vector<std::uint64_t> fb; for (std::uint64_t j = 0; j < 600; ++j) fb.push_back(j * 10000); for (std::uint64_t i = 0; i < 400; ++i) fb.push_back(6000000 + i);
    const auto f = fmcd_fit<std::uint64_t>(fb); CHECK(f.fallback); CHECK(f.d == 334); CHECK(f.capacity == 6000);
    const long double mid1 = 3335000.0L, mid2 = 6000066.5L, a_expect = 1998.0L / (mid2 - mid1);
    CHECK(std::fabs(f.a - a_expect) < 1e-15); CHECK(std::fabs(f.intercept() - (2001 - a_expect * mid1)) < 1e-6);
    CHECK(f.predict(fb[666]) == 3998); CHECK(f.predict(fb[667]) == 3999); CHECK(f.predict(fb[999]) == 3999); CHECK(f.predict(fb[600]) == 3998);
    CHECK(f.predict(fb[66]) == 0); CHECK(f.predict(fb[67]) == 3);
    CHECK(conflict_degree<std::uint64_t>(fb, f) == 333);
    // Uniform keys: no conflicts at all; capacity follows LIPP's gap table.
    std::vector<std::uint64_t> u(200000); for (std::size_t i = 0; i < u.size(); ++i) u[i] = 10 * i;
    const auto mu = fmcd_fit<std::uint64_t>(u); CHECK(mu.gap == 2); CHECK(mu.capacity == 600000); CHECK(conflict_degree<std::uint64_t>(u, mu) == 1);
    std::vector<std::uint64_t> big(1000000); for (std::size_t i = 0; i < big.size(); ++i) big[i] = 3 * i; CHECK(fmcd_fit<std::uint64_t>(big).gap == 1); CHECK(fmcd_fit<std::uint64_t>(big).capacity == 2000000);
    CHECK(conflict_degree<std::uint64_t>(std::vector<std::uint64_t>{1, 2}, fmcd_fit<std::uint64_t>(std::vector<std::uint64_t>{1, 2})) == 1);
    CHECK(conflict_degree<std::uint64_t>(std::vector<std::uint64_t>{}, FmcdModel{}) == 0);
    // Double features: same cluster scaled to a unit range, mean-gap-scaled epsilon keeps the result.
    std::vector<double> cd(cluster.size()); for (std::size_t i = 0; i < cd.size(); ++i) cd[i] = double(cluster[i]) / double(cluster.back());
    const auto md = fmcd_fit<double>(cd, default_ut_epsilon<double>(cd)); CHECK(!md.fallback); CHECK(conflict_degree<double>(cd, md) == 50);
}
void least_squares_cases() {
    std::vector<std::uint64_t> line(5000); for (std::size_t i = 0; i < line.size(); ++i) line[i] = 1000 + 13 * i;
    const auto l = least_squares<std::uint64_t>(line); CHECK(l.rmse < 1e-9); CHECK(l.max_error < 1e-9); CHECK(std::fabs(l.slope - 1.0L / 13) < 1e-12); CHECK(std::fabs(l.intercept) < 1e-9);
    // Hand-computed: x = {0, 1, 3}, y = {0, 1, 2}: slope 9/14, intercept 1/7, residuals -1/7, 3/14, -1/14.
    std::vector<std::uint64_t> three = {0, 1, 3}; const auto t = least_squares<std::uint64_t>(three);
    CHECK(std::fabs(t.slope - 9.0L / 14) < 1e-12); CHECK(std::fabs(t.intercept - 1.0L / 7) < 1e-12);
    CHECK(std::fabs(t.max_error - 3.0 / 14) < 1e-12); CHECK(std::fabs(t.rmse - std::sqrt(1.0 / 42)) < 1e-12);
    std::vector<double> three_d = {0, 1, 3}; const auto td = least_squares<double>(three_d); CHECK(std::fabs(td.rmse - t.rmse) < 1e-12);
    CHECK(least_squares<std::uint64_t>(std::vector<std::uint64_t>{}).n == 0); CHECK(least_squares<std::uint64_t>(std::vector<std::uint64_t>{7}).rmse == 0);
    // Metric block runs the four passes in parallel and agrees with the serial calls.
    std::vector<std::uint64_t> rnd(50000); std::mt19937_64 rng(5); for (auto& x : rnd) x = rng() % 1000000000; std::sort(rnd.begin(), rnd.end()); rnd.erase(std::unique(rnd.begin(), rnd.end()), rnd.end());
    const auto par = compute_metrics<std::uint64_t>(rnd, EPS, 1e-6, true), ser = compute_metrics<std::uint64_t>(rnd, EPS, 1e-6, false);
    CHECK(par.ls.rmse == ser.ls.rmse); CHECK(par.conflict_degree == ser.conflict_degree); CHECK(par.pla == ser.pla); CHECK(par.pla.size() == 2 && par.pla[0].first == 32);
    CHECK(par.conflict_degree == conflict_degree<std::uint64_t>(rnd, fmcd_fit<std::uint64_t>(rnd))); CHECK(par.pla[1].second == pla_segments<std::uint64_t>(rnd, 4096));
}
void smoothing_cases() {
    std::vector<double> f; std::mt19937_64 rng(11); double v = 0; for (unsigned i = 0; i < 1000; ++i) { v += i % 100 < 80 ? 1 : 500; f.push_back(v); }
    const auto s = smooth_regions(f, 128, 0.2, 4); CHECK(s.regions == 8); CHECK(s.virtual_points > 0); CHECK(s.features.size() == f.size() + s.virtual_points);
    for (std::size_t i = 1; i < s.features.size(); ++i) CHECK(s.features[i] >= s.features[i - 1]);
    CHECK(s.sse_after <= s.sse_before);
    std::size_t real = 0; for (auto x : s.features) real += std::binary_search(f.begin(), f.end(), x); CHECK(real == f.size()); // every real feature survives
    const auto serial = smooth_regions(f, 128, 0.2, 1); CHECK(serial.features == s.features); // thread count does not change the result
    const auto block = compute_metrics<double>(s.features, EPS, default_ut_epsilon<double>(s.features), true); CHECK(block.n == s.features.size()); CHECK(block.pla[0].second >= 1);
    CHECK(smooth_regions(f, 128, 0.0, 2).virtual_points == 0);
    bool threw = false; try { std::vector<double> bad = {3, 2, 1}; smooth_regions(bad, 2, 0.1, 1); } catch (const std::exception&) { threw = true; } CHECK(threw);
}
// Minimal JSON validator (objects, arrays, strings, numbers, literals) sufficient for the CLI output.
struct Json { const std::string& s; std::size_t i = 0; std::size_t objects = 0, nulls = 0;
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
    void expect(char c) { ws(); if (i >= s.size() || s[i] != c) throw std::runtime_error(std::string("json: expected ") + c + " at " + std::to_string(i)); ++i; }
    void string() { expect('"'); while (i < s.size() && s[i] != '"') { if (s[i] == '\\') ++i; ++i; } expect('"'); }
    void value() { ws(); if (i >= s.size()) throw std::runtime_error("json: truncated");
        if (s[i] == '{') { ++objects; ++i; ws(); if (s[i] == '}') { ++i; return; } for (;;) { string(); expect(':'); value(); ws(); if (s[i] == ',') { ++i; continue; } expect('}'); return; } }
        if (s[i] == '[') { ++i; ws(); if (s[i] == ']') { ++i; return; } for (;;) { value(); ws(); if (s[i] == ',') { ++i; continue; } expect(']'); return; } }
        if (s[i] == '"') { string(); return; }
        if (s.compare(i, 4, "null") == 0) { ++nulls; i += 4; return; } if (s.compare(i, 4, "true") == 0) { i += 4; return; } if (s.compare(i, 5, "false") == 0) { i += 5; return; }
        const auto b = i; if (s[i] == '-') ++i; while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) ++i;
        if (i == b) throw std::runtime_error("json: bad token at " + std::to_string(i)); }
};
void parallel_sort_cases() {
    // Identical to std::sort for every size class (below and above the sequential cut-off), thread count
    // (including non-powers of two, which round down to 1, 2, 4, 8) and value type; duplicates are preserved.
    std::mt19937_64 rng(7);
    for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(1000), std::size_t(70000), std::size_t(300001)}) {
        std::vector<std::uint64_t> keys(n); for (auto& k : keys) k = rng() % (n ? 2 * n : 1) + (rng() % 3 == 0 ? std::numeric_limits<std::uint64_t>::max() - n : 0);
        auto expect = keys; std::sort(expect.begin(), expect.end());
        for (unsigned threads : {0u, 1u, 2u, 3u, 5u, 8u, 16u, 64u}) { auto v = keys; parallel_sort(v, threads); CHECK(v == expect); }
        std::vector<double> d(keys.begin(), keys.end()); auto ed = d; std::sort(ed.begin(), ed.end());
        parallel_sort(d, 8); CHECK(d == ed);
    }
    std::vector<std::uint64_t> desc(200000); for (std::size_t i = 0; i < desc.size(); ++i) desc[i] = desc.size() - i; // fully reversed, many merges
    parallel_sort(desc, 16); CHECK(std::is_sorted(desc.begin(), desc.end())); CHECK(desc.front() == 1 && desc.back() == desc.size());
}
static std::string run(const std::string& cmd) { std::string out; FILE* p = popen(cmd.c_str(), "r"); if (!p) throw std::runtime_error("popen failed"); char buf[4096]; std::size_t n; while ((n = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n); if (pclose(p) != 0) throw std::runtime_error("command failed: " + cmd); return out; }
static std::string field(const std::string& json, const std::string& key) { const auto p = json.find("\"" + key + "\":"); if (p == std::string::npos) throw std::runtime_error("missing field " + key); const auto b = p + key.size() + 3; auto e = b; while (e < json.size() && json[e] != ',' && json[e] != '}' && json[e] != '\n') ++e; return json.substr(b, e - b); }
void cli_case(const std::string& bin, const std::string& fixture, const std::string& flow) {
    const auto out = run("'" + bin + "' --data '" + fixture + "' --dtype uint64 --flow '" + flow + "' --virtual-alpha 0.1 --region-keys 1024 --threads 4");
    Json j{out}; j.value(); j.ws(); CHECK(j.i == out.size()); CHECK(j.objects >= 8); CHECK(j.nulls == 0); // original, transformed and smoothed all present
    CHECK(field(out, "keys") == "16384"); CHECK(field(out, "sorted") == "true"); CHECK(field(out, "duplicates") == "0");
    CHECK(out.find("\"transformed\":{") != std::string::npos); CHECK(out.find("\"smoothed\":{") != std::string::npos);
    CHECK(std::stoull(field(out, "virtual_points")) > 0); CHECK(std::stoull(field(out, "regions")) == 16); CHECK(out.find("\"32\":") != std::string::npos && out.find("\"4096\":") != std::string::npos);
    const auto plain = run("'" + bin + "' --data '" + fixture + "'"); Json pj{plain}; pj.value(); CHECK(pj.nulls == 3); // flow_weights, transformed and smoothed are null
    CHECK(field(plain, "keys") == "16384"); CHECK(std::stoull(field(plain, "conflict_degree")) >= 1);
    const auto limited = run("'" + bin + "' --data '" + fixture + "' --limit 1000 --pla-eps 8"); CHECK(field(limited, "keys") == "1000"); CHECK(limited.find("\"8\":") != std::string::npos);
    bool failed = false; try { run("'" + bin + "' --data '" + fixture + "' --bogus 1 2>/dev/null"); } catch (const std::exception&) { failed = true; } CHECK(failed); // unknown option rejected
    // --sort-only on an unsorted file with duplicates; the path holds a double quote so the JSON must escape it.
    const auto dir = std::filesystem::temp_directory_path() / ("scaleli_hardness_q\"uote_" + std::to_string(::getpid())); std::filesystem::create_directories(dir);
    const auto unsorted = dir / "shuffled.sosd", sorted_copy = dir / "shuffled.sorted";
    { std::mt19937_64 rng(11); std::vector<std::uint64_t> keys(5000); for (auto& k : keys) k = rng() % 4000; // ~1,100 duplicates expected
      std::ofstream f(unsorted, std::ios::binary); auto put = [&](std::uint64_t v) { char b[8]; for (unsigned i = 0; i < 8; ++i) b[i] = char(v >> (8 * i)); f.write(b, 8); }; put(keys.size()); for (auto k : keys) put(k); }
    const auto audit = run("'" + bin + "' --data '" + unsorted.string() + "' --check-sorted 0 --sort-only 1 --write-sorted '" + sorted_copy.string() + "'");
    Json aj{audit}; aj.value(); aj.ws(); CHECK(aj.i == audit.size()); CHECK(field(audit, "sorted") == "false"); CHECK(audit.find("q\\\"uote_") != std::string::npos);
    const auto dups = std::stoull(field(audit, "duplicates")); CHECK(dups > 500); CHECK(std::stoull(field(audit, "written_keys")) == 5000 - dups); CHECK(std::stod(field(audit, "sort_ns")) > 0);
    { std::ifstream f(sorted_copy, std::ios::binary); std::vector<char> raw((std::istreambuf_iterator<char>(f)), {}); CHECK(raw.size() == 8 + 8 * (5000 - dups));
      std::vector<std::uint64_t> k(5000 - dups); for (std::size_t i = 0; i < k.size(); ++i) { std::uint64_t v = 0; for (unsigned j = 0; j < 8; ++j) v |= std::uint64_t(static_cast<unsigned char>(raw[8 + 8 * i + j])) << (8 * j); k[i] = v; }
      CHECK(std::is_sorted(k.begin(), k.end())); CHECK(std::adjacent_find(k.begin(), k.end()) == k.end()); }
    failed = false; try { run("'" + bin + "' --data '" + unsorted.string() + "' 2>/dev/null"); } catch (const std::exception&) { failed = true; } CHECK(failed); // --check-sorted 1 (default) refuses it
    const auto metrics = run("'" + bin + "' --data '" + sorted_copy.string() + "' --flow '" + flow + "'"); CHECK(field(metrics, "sorted") == "true"); CHECK(metrics.find("\"conflict_degree_lipp_epsilon\":") != std::string::npos);
    std::filesystem::remove_all(dir);
}
int main(int argc, char** argv) { try {
    exactness_near_2_64(); std::cout << "exact uint64 arithmetic near 2^64: PASS\n";
    pla_cases(); std::cout << "optimal PLA segment counts (hand cases + 320 DP-oracle differentials): PASS\n";
    cd_cases(); std::cout << "LIPP FMCD conflict degree (cluster, fallback, uniform): PASS\n";
    least_squares_cases(); std::cout << "least-squares RMSE / ME and parallel metric block: PASS\n";
    smoothing_cases(); std::cout << "CSV-style region smoothing sequence: PASS\n";
    parallel_sort_cases(); std::cout << "parallel chunk sort + merge equals std::sort: PASS\n";
    if (argc >= 4) { cli_case(argv[1], argv[2], argv[3]); std::cout << "scaleli_hardness CLI on the fixture with --flow and --virtual-alpha 0.1, and --sort-only on an unsorted file: PASS\n"; }
    else std::cout << "scaleli_hardness CLI check skipped (pass binary, fixture and flow paths)\n";
    std::cout << "ASSERTIONS=" << assertions << " ALL TESTS PASSED\n"; return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
