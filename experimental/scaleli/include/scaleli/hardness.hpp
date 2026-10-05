#pragma once
// Dataset hardness metrics for learned indexes, after Zhang, Tang and Ailamaki,
// "How Hard Can Indexing Be? Principled Dataset Hardness Measurement for Learned
// Indexes" (AIDB @ VLDB 2026), Section 4.1. Every metric is O(n) over a sorted
// feature sequence x_0 <= ... <= x_{n-1} whose rank target is y_i = i:
//   RMSE / ME  root-mean-square and maximum absolute error of the least-squares line
//   CD         conflict degree: the largest number of keys that LIPP's FMCD bulk-load
//              model maps to one slot
//   PLA-eps    number of segments of the optimal eps-bounded piecewise-linear
//              approximation (O'Rourke 1981), counted exactly as PGM-index does
//
// Third-party algorithms ported here. Nothing is copied verbatim beyond the
// arithmetic that defines the result; both are re-typed so that raw uint64 keys
// are never silently rounded to double (Apple arm64 has long double == double).
//  * FMCD: LIPP, Wu et al., PVLDB 2021. github.com/Jiacheng-WU/lipp, file
//    src/core/lipp.h: LIPP::build_tree_bulk_fmcd() (the "// FMCD method" block,
//    including its +1e-6 on U_T and the D*3 > size fallback that fits a line
//    through the two tertile midpoints), compute_gap_count() (BUILD_GAP_CNT = 1, 2
//    or 5 keyed on size; capacity L = size * (BUILD_GAP_CNT + 1)) and
//    PREDICT_POS() (floor + clamp to [0, L-1]). Commit
//    fe6ca4954f00875482f9e4dd63b34dae2384d23b (2022-02-07). BUILD_LR_REMAIN keeps
//    LIPP's default 0. License: MIT, Copyright (c) 2021 Jiacheng-WU.
//  * PLA: PGM-index, Ferragina and Vinciguerra, PVLDB 2020.
//    github.com/gvinciguerra/PGM-index, include/pgm/piecewise_linear_model.hpp:
//    OptimalPiecewiseLinearModel::add_point() and make_segmentation() (duplicate
//    handling and the trailing (x_last + 1, n) sentinel included). Commit
//    c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407. License: Apache-2.0,
//    Copyright (c) 2018 Giorgio Vinciguerra.
//
// Numeric contract. Integral inputs use exact differences (unsigned/signed
// __int128) and __int128 cross products; only the final slope/residual products
// are long double. Floating inputs (flow-transformed keys, CSV-smoothed slots)
// use the same code with long double slopes, as PGM does for floating keys.
#include "smoothing.hpp"
#include "types.hpp"
#include <atomic>
#include <cmath>
#include <exception>
#include <future>
#include <mutex>
#include <thread>
#include <limits>
#include <span>
#include <type_traits>

namespace scaleli::hardness {
using I128 = __int128;
using U128 = unsigned __int128;

// ---------------------------------------------------------------- parallel sort
// Seven of the ten GRE files are served unsorted (covid, genome, history, libio,
// planet, stack, wise; GRE's own loader sorts at load time), so the CLI has to sort
// 200M keys before any metric. std::sort alone needs ~17 s for that; this splits the
// range into a power-of-two number of chunks (<= threads), sorts them concurrently
// and merges pairwise with std::inplace_merge (temporary buffer <= n/2 elements).
// The result is identical to std::sort; duplicates are kept (the caller de-duplicates).
template<class T> void parallel_sort(std::vector<T>& v, unsigned threads) {
    const std::size_t n = v.size();
    unsigned parts = 1; while (parts * 2 <= std::max(1u, threads)) parts *= 2;
    if (parts == 1 || n < (std::size_t(1) << 16)) { std::sort(v.begin(), v.end()); return; }
    const std::size_t chunk = (n + parts - 1) / parts;
    std::vector<std::size_t> bounds(parts + 1); for (unsigned p = 0; p <= parts; ++p) bounds[p] = std::min(n, std::size_t(p) * chunk);
    {
        std::vector<std::thread> pool;
        for (unsigned p = 1; p < parts; ++p) pool.emplace_back([&v, &bounds, p] { std::sort(v.begin() + bounds[p], v.begin() + bounds[p + 1]); });
        std::sort(v.begin(), v.begin() + bounds[1]); for (auto& t : pool) t.join();
    }
    for (std::size_t width = 1; width < parts; width *= 2) { // merge disjoint pairs of runs, one thread per merge
        std::vector<std::thread> pool;
        for (std::size_t p = 0; p + width < parts; p += 2 * width) {
            auto job = [&v, &bounds, p, width, parts] { std::inplace_merge(v.begin() + bounds[p], v.begin() + bounds[p + width], v.begin() + bounds[std::min<std::size_t>(parts, p + 2 * width)]); };
            if (p + 2 * width < parts) pool.emplace_back(job); else job();
        }
        for (auto& t : pool) t.join();
    }
}

// ---------------------------------------------------------------- least squares
struct LeastSquares {
    long double slope = 0, intercept = 0; // rank ~ slope * (x - x_min) + intercept
    double rmse = 0, max_error = 0;
    std::size_t n = 0;
};
template<class T> LeastSquares least_squares(std::span<const T> x) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    LeastSquares r; r.n = x.size(); if (!r.n) return r;
    const T x0 = *std::min_element(x.begin(), x.end()); // x[0] for sorted input
    auto feature = [&](std::size_t i) -> long double {
        if constexpr (std::is_integral_v<T>) return static_cast<long double>(static_cast<U128>(x[i]) - static_cast<U128>(x0));
        else return static_cast<long double>(x[i]) - static_cast<long double>(x0);
    };
    long double mx = 0, my = 0, sxx = 0, sxy = 0; // centred (Welford) accumulators
    for (std::size_t i = 0; i < r.n; ++i) {
        const long double xi = feature(i), yi = static_cast<long double>(i), k = static_cast<long double>(i + 1);
        const long double dx = xi - mx, dy = yi - my;
        mx += dx / k; my += dy / k; sxx += dx * (xi - mx); sxy += dx * (yi - my);
    }
    r.slope = sxx > 0 ? sxy / sxx : 0; r.intercept = my - r.slope * mx;
    long double ss = 0, worst = 0;
    for (std::size_t i = 0; i < r.n; ++i) {
        const long double e = static_cast<long double>(i) - (r.slope * feature(i) + r.intercept);
        ss += e * e; worst = std::max(worst, std::fabs(e));
    }
    r.rmse = static_cast<double>(std::sqrt(ss / static_cast<long double>(r.n))); r.max_error = static_cast<double>(worst);
    return r;
}

// ---------------------------------------------------------------- FMCD (LIPP)
inline int lipp_gap_count(std::size_t size) { // LIPP::compute_gap_count
    if (size >= 1000000) return 1;
    if (size >= 100000) return 2;
    return 5;
}
// LIPP's node model in anchored form: position(key) = a * (key - anchor) + base.
// This is algebraically LIPP's a * key + b with b = base - a * anchor; the anchor
// form keeps 64-bit key differences exact (LIPP itself evaluates a * (long double)key,
// which is exact only on 80-bit long double platforms).
//
// Rounding caveat on CD. The slot is floor() of a floating-point product, exactly
// as LIPP's PREDICT_POS does it, so a key whose exact position lies within one
// double ulp of an integer slot boundary is assigned by rounding. Near L ~ 4e6 the
// position ulp is ~5e-10 while one key unit moves the position by a ~ 4e-13, so keys
// within ~1000 units of a boundary are rounding-decided; on real samples this shifts
// CD by +-1 (books 2M sample: 11 here and with LIPP's own double evaluation, 10 with
// exact rational arithmetic; an adversarial dense cluster moved 1.7 %). CD is therefore
// LIPP-faithful but not bit-reproducible across arithmetic; a ranking of datasets by
// CD alone should treat differences of one as ties.
struct FmcdModel {
    long double a = 0, base = 0, anchor = 0; // anchor is used for floating features
    I128 anchor2 = 0;                        // 2 * anchor, exact, used for integral keys
    std::size_t capacity = 0;                // LIPP num_items (= L; BUILD_LR_REMAIN = 0)
    int gap = 0;                             // LIPP BUILD_GAP_CNT
    std::size_t d = 0;                       // final D of the FMCD search
    double ut = 0, ut_epsilon = 0;           // U_T actually used and the additive epsilon
    bool fallback = false;                   // D*3 > size: tertile two-point fit
    bool degenerate = false;                 // size < 3 or non-finite slope
    long double intercept() const { return base - a * anchor; }
    template<class T> std::size_t predict(T key) const { // LIPP::PREDICT_POS
        long double v;
        if constexpr (std::is_integral_v<T>) v = a * (static_cast<long double>(2 * static_cast<I128>(key) - anchor2) / 2) + base;
        else v = a * (static_cast<long double>(key) - anchor) + base;
        const double p = static_cast<double>(v); // LIPP: predict_double returns double
        if (std::isnan(p)) return 0;
        if (p > double(std::numeric_limits<int>::max() / 2)) return capacity ? capacity - 1 : 0;
        if (p < 0) return 0;
        return std::min<std::size_t>(capacity ? capacity - 1 : 0, static_cast<std::size_t>(p));
    }
};
// The additive U_T epsilon: LIPP's absolute 1e-6 for integer keys (U_T is then at
// least ~0.5); for real-valued features it is scaled by the mean gap so that it stays
// negligible whatever the transform's output scale is.
//
// This is a documented DEVIATION from a literal LIPP fit on transformed or smoothed
// features: a flow output has range O(1), so LIPP's absolute 1e-6 would be a large
// fraction of U_T (147 % on the books 2M sample's flow features) and would dominate
// the fit. The CD reported for the transformed / smoothed blocks is therefore not the
// value LIPP's own code would compute on the same doubles; scaleli_hardness also
// emits fmcd.conflict_degree_lipp_epsilon (the literal +1e-6 fit) next to it so both
// are visible. Raw-key CD uses LIPP's constant verbatim.
template<class T> double default_ut_epsilon(std::span<const T> x) {
    if constexpr (std::is_integral_v<T>) { (void)x; return 1e-6; }
    else {
        if (x.size() < 2) return 0;
        const long double range = static_cast<long double>(x.back()) - static_cast<long double>(x.front());
        return static_cast<double>(1e-6L * range / static_cast<long double>(x.size() - 1));
    }
}
template<class T> FmcdModel fmcd_fit(std::span<const T> keys, double ut_epsilon = 1e-6) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    const std::size_t size = keys.size();
    FmcdModel m; m.ut_epsilon = ut_epsilon; m.gap = lipp_gap_count(size);
    if (size > std::size_t(std::numeric_limits<int>::max()) / std::size_t(m.gap + 1)) throw std::invalid_argument("FMCD capacity exceeds LIPP's int range");
    m.capacity = size * std::size_t(m.gap + 1);
    if (size < 3) { m.degenerate = true; return m; } // LIPP: build_tree_two, no model
    auto diff = [&](std::size_t hi, std::size_t lo) -> long double { // keys[hi] - keys[lo]
        if constexpr (std::is_integral_v<T>) return static_cast<long double>(static_cast<I128>(keys[hi]) - static_cast<I128>(keys[lo]));
        else return static_cast<long double>(keys[hi]) - static_cast<long double>(keys[lo]);
    };
    const long double L = static_cast<long double>(m.capacity);
    auto ut_for = [&](std::size_t D) { return static_cast<double>(diff(size - 1 - D, D) / (L - 2) + static_cast<long double>(ut_epsilon)); };
    auto gap_ge = [&](std::size_t i, std::size_t D, double Ut) { // LIPP: keys[i + D] - keys[i] >= Ut
        if constexpr (std::is_integral_v<T>) return static_cast<double>(static_cast<U128>(keys[i + D]) - static_cast<U128>(keys[i])) >= Ut;
        else return keys[i + D] - keys[i] >= Ut;
    };
    std::size_t i = 0, D = 1; double Ut = ut_for(D);
    while (i < size - 1 - D) {
        while (i + D < size && gap_ge(i, D, Ut)) ++i;
        if (i + D >= size) break;
        ++D;
        if (D * 3 > size) break;
        Ut = ut_for(D);
    }
    m.d = D; m.ut = Ut;
    if (D * 3 <= size) {
        m.a = 1.0L / static_cast<long double>(Ut);
        m.base = L / 2;
        if constexpr (std::is_integral_v<T>) { m.anchor2 = static_cast<I128>(keys[size - 1 - D]) + static_cast<I128>(keys[D]); m.anchor = static_cast<long double>(m.anchor2) / 2; }
        else { m.anchor = (static_cast<long double>(keys[size - 1 - D]) + static_cast<long double>(keys[D])) / 2; }
    } else {
        m.fallback = true;
        const std::size_t mid1 = (size - 1) / 3, mid2 = (size - 1) * 2 / 3, g = std::size_t(m.gap + 1);
        const long double t1 = static_cast<long double>(mid1 * g + g / 2), t2 = static_cast<long double>(mid2 * g + g / 2);
        long double dk;
        if constexpr (std::is_integral_v<T>) {
            m.anchor2 = static_cast<I128>(keys[mid1]) + static_cast<I128>(keys[mid1 + 1]); m.anchor = static_cast<long double>(m.anchor2) / 2;
            dk = static_cast<long double>(static_cast<I128>(keys[mid2]) + static_cast<I128>(keys[mid2 + 1]) - m.anchor2) / 2;
        } else {
            m.anchor = (static_cast<long double>(keys[mid1]) + static_cast<long double>(keys[mid1 + 1])) / 2;
            dk = (static_cast<long double>(keys[mid2]) + static_cast<long double>(keys[mid2 + 1])) / 2 - m.anchor;
        }
        m.a = (t2 - t1) / dk; m.base = t1;
    }
    if (!std::isfinite(m.a)) { m.a = 0; m.degenerate = true; } // LIPP asserts here; we report instead
    return m;
}
// Maximum keys per predicted slot. Slope >= 0 on sorted keys gives non-decreasing
// predictions, so equal slots are contiguous runs.
template<class T> std::size_t conflict_degree(std::span<const T> keys, const FmcdModel& m) {
    if (keys.empty()) return 0;
    if (keys.size() < 3) return 1; // LIPP's two-key node has no conflicts
    std::size_t best = 1, run = 1, last = m.predict(keys[0]);
    for (std::size_t i = 1; i < keys.size(); ++i) {
        const auto p = m.predict(keys[i]);
        if (p == last) ++run; else { best = std::max(best, run); run = 1; last = p; }
    }
    return std::max(best, run);
}

// ---------------------------------------------------------------- optimal PLA (PGM)
template<class X> class OptimalPLA {
    static_assert(std::is_floating_point_v<X> || std::is_same_v<X, U128>);
    using Y = std::size_t;
    using SX = std::conditional_t<std::is_floating_point_v<X>, long double, I128>;
    using SY = I128;
    struct Slope {
        SX dx{}; SY dy{};
        bool operator<(const Slope& p) const { return dy * p.dx < dx * p.dy; }
        bool operator>(const Slope& p) const { return dy * p.dx > dx * p.dy; }
    };
    struct Point {
        X x{}; Y y{};
        Slope operator-(const Point& p) const { return {SX(x) - SX(p.x), SY(y) - SY(p.y)}; }
    };
    const Y epsilon;
    std::vector<Point> lower, upper;
    X first_x = 0, last_x = 0;
    std::size_t lower_start = 0, upper_start = 0, points_in_hull = 0;
    Point rectangle[4];
    auto cross(const Point& O, const Point& A, const Point& B) const { const auto OA = A - O, OB = B - O; return OA.dx * OB.dy - OA.dy * OB.dx; }
public:
    explicit OptimalPLA(Y eps) : epsilon(eps) { upper.reserve(1u << 16); lower.reserve(1u << 16); }
    bool add_point(const X& x, const Y& y) {
        if (points_in_hull > 0 && x <= last_x) throw std::logic_error("PLA points must be increasing by x");
        last_x = x;
        const Y max_y = std::numeric_limits<Y>::max(), min_y = std::numeric_limits<Y>::lowest();
        const Point p1{x, y >= max_y - epsilon ? max_y : y + epsilon};
        const Point p2{x, y <= min_y + epsilon ? min_y : y - epsilon};
        if (points_in_hull == 0) {
            first_x = x; rectangle[0] = p1; rectangle[1] = p2;
            upper.clear(); lower.clear(); upper.push_back(p1); lower.push_back(p2);
            upper_start = lower_start = 0; ++points_in_hull; return true;
        }
        if (points_in_hull == 1) { rectangle[2] = p2; rectangle[3] = p1; upper.push_back(p1); lower.push_back(p2); ++points_in_hull; return true; }
        const auto slope1 = rectangle[2] - rectangle[0], slope2 = rectangle[3] - rectangle[1];
        const bool outside_line1 = p1 - rectangle[2] < slope1, outside_line2 = p2 - rectangle[3] > slope2;
        if (outside_line1 || outside_line2) { points_in_hull = 0; return false; }
        if (p1 - rectangle[1] < slope2) {
            auto min = lower[lower_start] - p1; auto min_i = lower_start;
            for (auto i = lower_start + 1; i < lower.size(); ++i) { const auto val = lower[i] - p1; if (val > min) break; min = val; min_i = i; }
            rectangle[1] = lower[min_i]; rectangle[3] = p1; lower_start = min_i;
            auto end = upper.size();
            for (; end >= upper_start + 2 && cross(upper[end - 2], upper[end - 1], p1) <= 0; --end) continue;
            upper.resize(end); upper.push_back(p1);
        }
        if (p2 - rectangle[0] > slope1) {
            auto max = upper[upper_start] - p2; auto max_i = upper_start;
            for (auto i = upper_start + 1; i < upper.size(); ++i) { const auto val = upper[i] - p2; if (val < max) break; max = val; max_i = i; }
            rectangle[0] = upper[max_i]; rectangle[2] = p2; upper_start = max_i;
            auto end = lower.size();
            for (; end >= lower_start + 2 && cross(lower[end - 2], lower[end - 1], p2) >= 0; --end) continue;
            lower.resize(end); lower.push_back(p2);
        }
        ++points_in_hull; return true;
    }
    void reset() { points_in_hull = 0; lower.clear(); upper.clear(); }
};
// Segment count identical to pgm::internal::make_segmentation(n, epsilon, in, out)
// on the same sequence: duplicate keys map the value x+1 (next representable for
// floating x) to the rank of the last duplicate, and the sentinel (x_last + 1, n)
// is appended. Integral keys are widened to unsigned __int128 so the sentinel
// exists even for x_last == 2^64 - 1, where PGM's make_segmentation<uint64_t>
// itself throws "Points must be increasing by x." (its sentinel wraps to 0). PLA is
// translation-invariant in x, so the count here equals PGM's count on the same keys
// shifted down by any constant (verified on lognormal keys offset to end at 2^64-1).
template<class T> std::size_t pla_segments(std::span<const T> in, std::size_t epsilon) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    using X = std::conditional_t<std::is_floating_point_v<T>, T, U128>;
    const std::size_t n = in.size(); if (!n) return 0;
    OptimalPLA<X> opt(epsilon); std::size_t c = 0;
    auto add = [&](X x, std::size_t y) { if (!opt.add_point(x, y)) { ++c; opt.add_point(x, y); } };
    auto at = [&](std::size_t i) { return static_cast<X>(in[i]); };
    auto next = [](X x) { if constexpr (std::is_floating_point_v<X>) return std::nextafter(x, std::numeric_limits<X>::infinity()); else return x + 1; };
    add(at(0), 0);
    for (std::size_t i = 1; i + 1 < n; ++i) {
        if (at(i) == at(i - 1)) { if (next(at(i)) < at(i + 1)) add(next(at(i)), i); }
        else add(at(i), i);
    }
    if (n >= 2 && at(n - 1) != at(n - 2)) add(at(n - 1), n - 1);
    add(next(at(n - 1)), n);
    return c + 1;
}

// ---------------------------------------------------------------- one metric block
struct MetricBlock {
    std::size_t n = 0;
    LeastSquares ls;
    FmcdModel fmcd;
    std::size_t conflict_degree = 0;
    std::vector<std::pair<std::size_t, std::size_t>> pla; // (epsilon, segments)
};
// The four passes are independent read-only scans; with parallel == true they run
// on separate threads (least squares, FMCD + CD, one thread per PLA epsilon).
template<class T> MetricBlock compute_metrics(std::span<const T> x, std::span<const std::size_t> epsilons, double ut_epsilon, bool parallel = true) {
    MetricBlock b; b.n = x.size();
    const auto policy = parallel ? std::launch::async : std::launch::deferred;
    auto ls = std::async(policy, [x] { return least_squares<T>(x); });
    auto cd = std::async(policy, [x, ut_epsilon] { const auto m = fmcd_fit<T>(x, ut_epsilon); return std::make_pair(m, conflict_degree<T>(x, m)); });
    std::vector<std::future<std::size_t>> segs;
    for (auto e : epsilons) segs.push_back(std::async(policy, [x, e] { return pla_segments<T>(x, e); }));
    b.ls = ls.get(); auto [m, degree] = cd.get(); b.fmcd = m; b.conflict_degree = degree;
    for (std::size_t i = 0; i < epsilons.size(); ++i) b.pla.emplace_back(epsilons[i], segs[i].get());
    return b;
}

// ---------------------------------------------------------------- CSV-style smoothing
// Applies smooth_cdf (smoothing.hpp, Algorithm 1 of CSV) to consecutive regions of
// `region_keys` sorted features and returns the augmented sequence: real and virtual
// features in feature order, so that position j in `features` is slot j of the
// concatenated per-region slot spaces. Cost is O(alpha * n * region_keys) loss
// evaluations, so this is meant for samples (a few million keys), not 200M keys.
struct SmoothedSequence {
    std::vector<double> features;
    std::size_t virtual_points = 0, regions = 0, rounds = 0;
    double sse_before = 0, sse_after = 0; // summed over regions
};
inline SmoothedSequence smooth_regions(std::span<const double> f, std::size_t region_keys, double alpha, unsigned threads = 1) {
    if (!region_keys) throw std::invalid_argument("region_keys must be positive");
    for (std::size_t i = 1; i < f.size(); ++i) if (f[i] < f[i - 1]) throw std::invalid_argument("smoothing features must be sorted ascending");
    const std::size_t n = f.size(), regions = n ? (n + region_keys - 1) / region_keys : 0;
    struct Part { std::vector<double> aug; std::size_t virt = 0, rounds = 0; double before = 0, after = 0; };
    std::vector<Part> parts(regions);
    std::atomic<std::size_t> next{0}; std::exception_ptr failure; std::mutex failure_lock;
    auto worker = [&] {
        try {
            for (std::size_t r; (r = next.fetch_add(1)) < regions;) {
                const std::size_t b = r * region_keys, len = std::min(region_keys, n - b);
                const auto s = smooth_cdf(f.subspan(b, len), alpha);
                Part& p = parts[r]; p.aug.assign(len + s.virtual_features.size(), 0);
                std::vector<unsigned char> used(p.aug.size(), 0);
                for (std::size_t i = 0; i < len; ++i) { p.aug[s.slot[i]] = f[b + i]; used[s.slot[i]] = 1; }
                auto v = s.virtual_features; std::sort(v.begin(), v.end()); std::size_t k = 0;
                for (std::size_t j = 0; j < p.aug.size(); ++j) if (!used[j]) p.aug[j] = v.at(k++);
                if (k != v.size()) throw std::logic_error("virtual point slots do not match");
                p.virt = v.size(); p.rounds = s.rounds; p.before = s.sse_before; p.after = s.sse_after;
            }
        } catch (...) { std::lock_guard<std::mutex> g(failure_lock); if (!failure) failure = std::current_exception(); }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < std::max(1u, threads) && t < regions; ++t) pool.emplace_back(worker);
    worker(); for (auto& t : pool) t.join();
    if (failure) std::rethrow_exception(failure);
    SmoothedSequence out; out.regions = regions; std::size_t total = 0;
    for (const auto& p : parts) { total += p.aug.size(); out.virtual_points += p.virt; out.rounds += p.rounds; out.sse_before += p.before; out.sse_after += p.after; }
    out.features.reserve(total);
    for (const auto& p : parts) out.features.insert(out.features.end(), p.aug.begin(), p.aug.end());
    return out;
}
} // namespace scaleli::hardness
