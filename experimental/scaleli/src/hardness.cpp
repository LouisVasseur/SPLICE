// scaleli_hardness: dataset hardness metrics (RMSE, ME, CD, PLA-eps) on a SOSD key
// file, optionally after the NFL-style flow transform and after CSV-style virtual
// point smoothing. One JSON object on stdout. See include/scaleli/hardness.hpp for
// the metric definitions and the LIPP / PGM-index provenance of the ports.
#include "scaleli/hardness.hpp"
#include "scaleli/transform.hpp"
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <thread>

using namespace scaleli;
using Clock = std::chrono::steady_clock;
static double nanos(Clock::duration d) { return std::chrono::duration<double, std::nano>(d).count(); }
static std::string quote(const std::string& s) { std::string out = "\""; for (char c : s) { if (c == '"' || c == '\\') out += '\\'; if (c == '\n') out += "\\n"; else if (c == '\r') out += "\\r"; else out += c; } return out + '"'; }
static std::string num(double v) { if (!std::isfinite(v)) return "null"; std::ostringstream o; o << std::setprecision(17) << v; return o.str(); }
static std::string num(long double v) { return num(static_cast<double>(v)); }

struct Args { // same style as scaleli_bench: --key value or --key=value, allow-listed
    std::map<std::string, std::string> a;
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) { std::string x = argv[i]; if (x.rfind("--", 0) != 0) throw std::invalid_argument("expected --option"); x = x.substr(2); const auto eq = x.find('=');
            if (eq != std::string::npos) a[x.substr(0, eq)] = x.substr(eq + 1); else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a[x] = argv[++i]; else a[x] = "1"; }
        const std::set<std::string> allowed = {"help", "data", "dtype", "limit", "flow", "virtual-alpha", "region-keys", "pla-eps", "check-sorted", "verbose", "threads", "write-sorted", "sort-only"};
        for (auto& [k, v] : a) if (!allowed.count(k)) throw std::invalid_argument("unknown option: " + k);
    }
    std::string get(std::string k, std::string d) const { auto it = a.find(k); return it == a.end() ? d : it->second; }
    std::size_t number(std::string k, std::size_t d) const { auto s = get(k, std::to_string(d)); if (s.empty() || s[0] == '-') throw std::invalid_argument("nonnegative integer required: " + k); std::size_t pos; const auto v = std::stoull(s, &pos); if (pos != s.size()) throw std::invalid_argument("invalid integer: " + k); return v; }
    double real(std::string k, double d) const { auto s = get(k, std::to_string(d)); std::size_t pos; auto v = std::stod(s, &pos); if (pos != s.size() || !std::isfinite(v)) throw std::invalid_argument("finite number required: " + k); return v; }
    bool flag(std::string k, bool d) const { auto s = get(k, d ? "1" : "0"); if (s != "1" && s != "0") throw std::invalid_argument("flag requires 0 or 1: " + k); return s == "1"; }
};

// SOSD: 8-byte little-endian count, then count little-endian keys of `width` bytes.
static std::vector<Key> read_sosd(const std::string& path, unsigned width, std::size_t limit) {
    std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot open dataset: " + path);
    in.seekg(0, std::ios::end); const auto bytes = static_cast<std::uint64_t>(in.tellg()); in.seekg(0);
    if (bytes < 8 || (bytes - 8) % width) throw std::invalid_argument("invalid SOSD byte length");
    unsigned char h[8]; in.read(reinterpret_cast<char*>(h), 8); std::uint64_t count = 0; for (unsigned i = 0; i < 8; ++i) count |= std::uint64_t(h[i]) << (8 * i);
    if (count != (bytes - 8) / width) throw std::invalid_argument("SOSD header count does not match file length");
    if (limit) count = std::min<std::uint64_t>(count, limit);
    std::vector<Key> keys(count);
    if (width == 8) {
        in.read(reinterpret_cast<char*>(keys.data()), std::streamsize(count * 8));
        if constexpr (std::endian::native != std::endian::little) for (auto& k : keys) { unsigned char b[8]; std::memcpy(b, &k, 8); k = 0; for (unsigned i = 0; i < 8; ++i) k |= std::uint64_t(b[i]) << (8 * i); }
    } else {
        std::vector<unsigned char> raw(count * 4); in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size()));
        for (std::uint64_t i = 0; i < count; ++i) { std::uint64_t k = 0; for (unsigned j = 0; j < 4; ++j) k |= std::uint64_t(raw[i * 4 + j]) << (8 * j); keys[i] = k; }
    }
    if (!in) throw std::runtime_error("short read on dataset");
    return keys;
}

template<class F> static void parallel_for(std::size_t n, unsigned threads, F f) { // f(begin, end)
    threads = std::max(1u, threads); const std::size_t chunk = std::max<std::size_t>(1, (n + threads - 1) / threads);
    std::vector<std::thread> pool;
    for (std::size_t b = chunk; b < n; b += chunk) pool.emplace_back([=] { f(b, std::min(n, b + chunk)); });
    f(0, std::min(n, chunk)); for (auto& t : pool) t.join();
}

template<class T> static std::string metric_json(const hardness::MetricBlock& b, std::span<const T> x, std::optional<std::size_t> cd_literal = std::nullopt) {
    std::ostringstream o; o << std::setprecision(17);
    o << "{\"keys\":" << b.n << ",\"rmse\":" << num(b.ls.rmse) << ",\"max_error\":" << num(b.ls.max_error) << ",\"conflict_degree\":" << b.conflict_degree << ",\"pla\":{";
    for (std::size_t i = 0; i < b.pla.size(); ++i) o << (i ? "," : "") << '"' << b.pla[i].first << "\":" << b.pla[i].second;
    o << "},\"least_squares\":{\"slope\":" << num(b.ls.slope) << ",\"intercept\":" << num(b.ls.intercept) << ",\"origin\":";
    if constexpr (std::is_integral_v<T>) o << (x.empty() ? 0 : *std::min_element(x.begin(), x.end())); else o << num(x.empty() ? 0.0 : *std::min_element(x.begin(), x.end()));
    o << "},\"fmcd\":{\"slope\":" << num(b.fmcd.a) << ",\"intercept\":" << num(b.fmcd.intercept()) << ",\"capacity\":" << b.fmcd.capacity
      << ",\"gap\":" << b.fmcd.gap << ",\"d\":" << b.fmcd.d << ",\"ut\":" << num(b.fmcd.ut) << ",\"ut_epsilon\":" << num(b.fmcd.ut_epsilon)
      << ",\"fallback\":" << (b.fmcd.fallback ? "true" : "false") << ",\"degenerate\":" << (b.fmcd.degenerate ? "true" : "false");
    if (cd_literal) o << ",\"conflict_degree_lipp_epsilon\":" << *cd_literal; // literal LIPP +1e-6 on U_T, for comparison with the scaled epsilon above
    o << "}";
    if (!x.empty()) { o << ",\"min\":"; if constexpr (std::is_integral_v<T>) o << x.front() << ",\"max\":" << x.back(); else o << num(x.front()) << ",\"max\":" << num(x.back()); }
    return o.str();
}

int main(int argc, char** argv) { try {
    Args a(argc, argv);
    if (a.flag("help", false)) { std::cout << R"(SCALE-LI dataset hardness metrics (AIDB 2026 protocol; clean-room ports, not the paper's code)
  --data PATH --dtype uint64|uint32 [--limit N]   SOSD file (8-byte count header); --limit loads a PREFIX
  --flow weights.txt        also compute the block on flow(key) values (NFL-format weights), sorted ascending
  --virtual-alpha A         also compute the block on the CSV-style augmented sequence (alpha * region virtual points
                            per region; features are the transformed keys with --flow, else double(key - min)).
                            Costs O(alpha * keys * region) loss evaluations: meant for samples of a few million keys.
  --region-keys R           smoothing region size (default 4096)
  --pla-eps 32,4096         PLA error bounds (PGM-index make_segmentation semantics)
  --check-sorted 1          1 = fail on an unsorted file; 0 = sort in memory (parallel chunk sort + merge over --threads)
                            and report "sorted": false. Seven GRE files are served unsorted: covid genome history libio planet stack wise.
  --write-sorted PATH       write a sorted, de-duplicated SOSD copy of the keys (only when the input is unsorted or has duplicates)
  --sort-only 1             stop after the sortedness/duplicate audit (and --write-sorted); no metrics
  --threads T               worker threads (default: hardware concurrency); metrics are deterministic either way
  --verbose 1               progress on stderr
Output: one JSON object with "original", "transformed" (null without --flow) and "smoothed" (null without --virtual-alpha)
metric blocks: rmse, max_error, conflict_degree (LIPP FMCD), pla {eps: segments}, fmcd {slope, intercept, capacity}.
Raw uint64 keys use exact __int128 arithmetic; transformed and smoothed blocks use double features.
CD on transformed/smoothed features adds 1e-6 * mean gap to U_T instead of LIPP's absolute 1e-6 (which would dominate
O(1)-range flow outputs); fmcd.conflict_degree_lipp_epsilon gives the literal-constant value for comparison. CD is
floor() of a double product, as in LIPP, so it carries +-1 rounding uncertainty near slot boundaries.
Cost: ~3 s and 1.6 GB RSS for a sorted 200M-key file on 16 threads (the sequential PLA-32 pass bounds the wall clock),
plus the parallel sort when the file is unsorted; --flow adds an 8-byte double per key.
)"; return 0; }
    if (!a.a.count("data")) throw std::invalid_argument("--data is required");
    const auto path = a.get("data", ""); const auto dtype = a.get("dtype", "uint64");
    if (dtype != "uint64" && dtype != "uint32") throw std::invalid_argument("invalid dtype");
    const auto region_keys = a.number("region-keys", 4096); if (!region_keys) throw std::invalid_argument("region-keys must be positive");
    const double alpha = a.real("virtual-alpha", 0); if (alpha < 0 || alpha >= 1) throw std::invalid_argument("virtual-alpha must be in [0, 1)");
    const bool check_sorted = a.flag("check-sorted", true), verbose = a.flag("verbose", false);
    const unsigned threads = unsigned(a.number("threads", std::max(1u, std::thread::hardware_concurrency())));
    std::vector<std::size_t> eps; { std::stringstream ss(a.get("pla-eps", "32,4096")); std::string tok; while (std::getline(ss, tok, ',')) { if (tok.empty()) continue; std::size_t pos; const auto v = std::stoull(tok, &pos); if (pos != tok.size()) throw std::invalid_argument("invalid --pla-eps entry: " + tok); eps.push_back(v); } }
    if (eps.empty()) throw std::invalid_argument("--pla-eps needs at least one value");
    auto log = [&](const std::string& s) { if (verbose) std::cerr << "[hardness] " << s << '\n'; };

    const auto t_start = Clock::now();
    auto keys = read_sosd(path, dtype == "uint64" ? 8 : 4, a.number("limit", 0));
    const double read_ns = nanos(Clock::now() - t_start); log("read " + std::to_string(keys.size()) + " keys");
    bool sorted = true; std::size_t duplicates = 0;
    for (std::size_t i = 1; i < keys.size(); ++i) { if (keys[i] < keys[i - 1]) { sorted = false; break; } }
    double sort_ns = 0;
    if (!sorted) { if (check_sorted) throw std::runtime_error("keys are not sorted (use --check-sorted 0 to sort in memory)"); const auto ts = Clock::now(); hardness::parallel_sort(keys, threads); sort_ns = nanos(Clock::now() - ts); }
    for (std::size_t i = 1; i < keys.size(); ++i) duplicates += keys[i] == keys[i - 1];
    log("sorted=" + std::string(sorted ? "true" : "false") + " duplicates=" + std::to_string(duplicates) + (sorted ? "" : " sort_ns=" + std::to_string(std::uint64_t(sort_ns))));
    std::string written = "null"; std::size_t written_keys = 0;
    if (a.a.count("write-sorted") && (!sorted || duplicates)) {
        std::vector<Key> uniq; uniq.reserve(keys.size() - duplicates);
        for (std::size_t i = 0; i < keys.size(); ++i) if (!i || keys[i] != keys[i - 1]) uniq.push_back(keys[i]);
        const auto out_path = a.get("write-sorted", ""); std::ofstream out(out_path, std::ios::binary); if (!out) throw std::runtime_error("cannot write " + out_path);
        auto put = [&](std::uint64_t v) { char b[8]; for (unsigned i = 0; i < 8; ++i) b[i] = char(v >> (8 * i)); out.write(b, 8); };
        put(uniq.size()); for (auto k : uniq) put(k); out.close(); if (!out) throw std::runtime_error("short write to " + out_path);
        written = quote(out_path); written_keys = uniq.size(); log("wrote sorted copy " + out_path + " with " + std::to_string(written_keys) + " keys");
    }
    if (a.flag("sort-only", false)) {
        std::cout << std::setprecision(17) << "{\"data\":" << quote(path) << ",\"keys\":" << keys.size() << ",\"sorted\":" << (sorted ? "true" : "false")
                  << ",\"duplicates\":" << duplicates << ",\"written\":" << written << ",\"written_keys\":" << written_keys << ",\"threads\":" << threads
                  << ",\"read_ns\":" << num(read_ns) << ",\"sort_ns\":" << num(sort_ns) << ",\"elapsed_ns\":" << num(nanos(Clock::now() - t_start)) << "}\n";
        return 0;
    }

    auto t = Clock::now();
    const auto original = hardness::compute_metrics<Key>(keys, eps, 1e-6, threads > 1);
    const double original_ns = nanos(Clock::now() - t); log("original metrics done");

    std::optional<FlowTransform> flow; std::vector<double> transformed; std::string transformed_json = "null";
    std::size_t unordered_pairs = 0; double transform_ns = 0;
    if (a.a.count("flow")) {
        flow = FlowTransform::load(a.get("flow", "")); t = Clock::now();
        transformed.resize(keys.size());
        parallel_for(keys.size(), threads, [&](std::size_t b, std::size_t e) { for (std::size_t i = b; i < e; ++i) transformed[i] = (*flow)(keys[i]); });
        transform_ns = nanos(Clock::now() - t);
        for (std::size_t i = 1; i < transformed.size(); ++i) unordered_pairs += transformed[i] < transformed[i - 1];
        if (unordered_pairs) hardness::parallel_sort(transformed, threads);
        std::size_t tdup = 0; for (std::size_t i = 1; i < transformed.size(); ++i) tdup += transformed[i] == transformed[i - 1];
        log("transformed: unordered_pairs=" + std::to_string(unordered_pairs));
        t = Clock::now();
        const auto block = hardness::compute_metrics<double>(transformed, eps, hardness::default_ut_epsilon<double>(transformed), threads > 1);
        const auto cd_lit = hardness::conflict_degree<double>(transformed, hardness::fmcd_fit<double>(transformed, 1e-6)); // literal LIPP epsilon, for transparency
        std::ostringstream o; o << metric_json<double>(block, transformed, cd_lit) << ",\"unordered_pairs\":" << unordered_pairs << ",\"duplicates\":" << tdup
          << ",\"transform_ns\":" << num(transform_ns) << ",\"metrics_ns\":" << num(nanos(Clock::now() - t)) << ",\"flow_bytes\":" << flow->bytes() << "}";
        transformed_json = o.str(); log("transformed metrics done");
    }

    std::string smoothed_json = "null";
    if (alpha > 0) {
        std::vector<double> features;
        if (flow) features = transformed; // sorted transformed keys; equals key order for a monotone flow
        else { features.resize(keys.size()); const Key origin = keys.empty() ? 0 : keys.front(); for (std::size_t i = 0; i < keys.size(); ++i) features[i] = double(keys[i] - origin); }
        t = Clock::now();
        const auto sm = hardness::smooth_regions(features, region_keys, alpha, threads);
        const double smoothing_ns = nanos(Clock::now() - t); log("smoothing done: virtual_points=" + std::to_string(sm.virtual_points));
        t = Clock::now();
        const auto block = hardness::compute_metrics<double>(sm.features, eps, hardness::default_ut_epsilon<double>(sm.features), threads > 1);
        const auto cd_lit = hardness::conflict_degree<double>(sm.features, hardness::fmcd_fit<double>(sm.features, 1e-6));
        std::ostringstream o; o << metric_json<double>(block, sm.features, cd_lit) << ",\"virtual_points\":" << sm.virtual_points << ",\"regions\":" << sm.regions
          << ",\"region_keys\":" << region_keys << ",\"alpha\":" << num(alpha) << ",\"features\":" << quote(flow ? "flow" : "key_minus_min") << ",\"rounds\":" << sm.rounds
          << ",\"rank_sse_before\":" << num(sm.sse_before) << ",\"rank_sse_after\":" << num(sm.sse_after)
          << ",\"smoothing_ns\":" << num(smoothing_ns) << ",\"metrics_ns\":" << num(nanos(Clock::now() - t)) << "}";
        smoothed_json = o.str(); log("smoothed metrics done");
    }

    std::cout << std::setprecision(17) << "{\n\"schema\":\"scaleli_hardness/1\",\"data\":" << quote(path) << ",\"dtype\":" << quote(dtype) << ",\"limit\":" << a.number("limit", 0)
      << ",\"keys\":" << keys.size() << ",\"sorted\":" << (sorted ? "true" : "false") << ",\"duplicates\":" << duplicates
      << ",\"min\":" << (keys.empty() ? 0 : keys.front()) << ",\"max\":" << (keys.empty() ? 0 : keys.back())
      << ",\"pla_eps\":["; for (std::size_t i = 0; i < eps.size(); ++i) std::cout << (i ? "," : "") << eps[i];
    std::cout << "],\"flow_weights\":" << (flow ? quote(a.get("flow", "")) : "null") << ",\"virtual_alpha\":" << num(alpha) << ",\"region_keys\":" << region_keys
      << ",\"threads\":" << threads << ",\"read_ns\":" << num(read_ns) << ",\"sort_ns\":" << num(sort_ns) << ",\"original_metrics_ns\":" << num(original_ns) << ",\"elapsed_ns\":" << num(nanos(Clock::now() - t_start))
      << ",\n\"original\":" << metric_json<Key>(original, keys) << "},\n\"transformed\":" << transformed_json << ",\n\"smoothed\":" << smoothed_json
      << ",\n\"note\":\"Reduced-scale, single-host, clean-room controls: FMCD ported from LIPP (MIT), PLA from PGM-index (Apache-2.0); not the paper's measurements and not NFL/CSV.\"\n}\n";
    return 0;
} catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; } }
