#pragma once
// CDF smoothing by virtual points, after CSV (Amarasinghe et al., EDBT 2025),
// Algorithm 1: greedily insert up to lambda = alpha * n virtual points into the
// gaps between consecutive keys, refitting the linear model after each insertion,
// and stop early when no candidate lowers the augmented-set SSE.
//
// No public implementation of CSV exists (checked September 2026), so this is
// written from the paper's description. Choices that are ours:
//  * The loss is the OLS residual SSE over real AND virtual points, matching
//    equation 4 of the paper, computed in closed form from running sums. A
//    candidate at gap i shifts every later target by one; suffix sums make each
//    candidate evaluation O(1), so one round costs O(gaps) and the whole run
//    O(lambda * n). The paper's O(n + lambda) claim is not reproduced.
//  * Within a gap, the loss is sampled just inside both ends; if it is
//    decreasing on the left and increasing on the right, a ternary search finds
//    the interior minimum (the paper uses the sign of the derivative at the
//    endpoints for the same purpose). Otherwise the better end is used.
//  * Features are doubles, so this can run on flow-transformed keys. Gaps whose
//    right feature is not greater than the left are skipped.
//  * Virtual points are returned as per-key slot offsets only. They are never
//    stored as records and never affect logical rank.
#include "types.hpp"
#include <cmath>
#include <span>

namespace scaleli {
struct SmoothingResult {
    std::vector<std::size_t> slot;        // slot rank of each original key (rank + virtual points before it)
    std::vector<double> virtual_features; // feature values of inserted virtual points, insertion order
    double sse_before = 0, sse_after = 0; // augmented-set OLS SSE
    std::size_t rounds = 0;
};

namespace detail {
struct Sums {
    long double n = 0, sx = 0, sxx = 0, sy = 0, syy = 0, sxy = 0;
    void add(long double x, long double y) { ++n; sx += x; sxx += x * x; sy += y; syy += y * y; sxy += x * y; }
    long double sse() const {
        if (n < 2) return 0;
        const long double cxx = sxx - sx * sx / n, cxy = sxy - sx * sy / n, cyy = syy - sy * sy / n;
        const long double s = cxx > 0 ? cyy - cxy * cxy / cxx : cyy;
        return s > 0 ? s : 0;
    }
};
} // namespace detail

inline SmoothingResult smooth_cdf(std::span<const double> x, double alpha, std::size_t max_ternary_steps = 40) {
    if (!(alpha >= 0) || alpha >= 64 || !std::isfinite(alpha)) throw std::invalid_argument("smoothing alpha must be in [0, 64)"); // regions cap alpha < 1 in Config; the root may use more
    const std::size_t n = x.size();
    SmoothingResult out; out.slot.resize(n); for (std::size_t i = 0; i < n; ++i) out.slot[i] = i;
    // Working sequence: (feature, is_virtual). Targets are sequence positions.
    struct P { double x; bool virt; };
    std::vector<P> seq; seq.reserve(n + std::size_t(alpha * double(n)) + 1);
    for (auto v : x) seq.push_back({v, false});
    auto total = [&] { detail::Sums s; for (std::size_t i = 0; i < seq.size(); ++i) s.add(seq[i].x, static_cast<long double>(i)); return s; };
    detail::Sums base = total(); out.sse_before = double(base.sse()); out.sse_after = out.sse_before;
    const std::size_t budget = std::size_t(alpha * double(n));
    if (n < 3 || !budget) return out;
    std::vector<long double> suf_x(seq.size() + 1), suf_y(seq.size() + 1);
    auto rebuild_suffix = [&] {
        suf_x.assign(seq.size() + 1, 0); suf_y.assign(seq.size() + 1, 0);
        for (std::size_t i = seq.size(); i-- > 0;) { suf_x[i] = suf_x[i + 1] + seq[i].x; suf_y[i] = suf_y[i + 1] + static_cast<long double>(i); }
    };
    // Loss if a virtual point with feature xv is inserted after position i (i.e. at target i+1).
    auto loss_at = [&](std::size_t i, long double xv) {
        detail::Sums s = base; const long double cnt = static_cast<long double>(seq.size() - (i + 1));
        s.sy += cnt; s.syy += 2 * suf_y[i + 1] + cnt; s.sxy += suf_x[i + 1];
        s.add(xv, static_cast<long double>(i + 1));
        return s.sse();
    };
    long double current = base.sse();
    for (std::size_t round = 0; round < budget; ++round) {
        rebuild_suffix();
        long double best = current; std::size_t best_i = 0; long double best_x = 0; bool found = false;
        for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
            const long double lo = seq[i].x, hi = seq[i + 1].x; if (!(hi > lo)) continue;
            const long double w = hi - lo, e = w * 1e-3L;
            const long double l0 = loss_at(i, lo + e), l1 = loss_at(i, lo + 2 * e), r1 = loss_at(i, hi - 2 * e), r0 = loss_at(i, hi - e);
            long double cx, cl;
            if (l1 < l0 && r1 < r0) { // interior minimum: ternary search
                long double a = lo + e, b = hi - e;
                for (std::size_t it = 0; it < max_ternary_steps && b - a > e; ++it) {
                    const long double m1 = a + (b - a) / 3, m2 = b - (b - a) / 3;
                    if (loss_at(i, m1) < loss_at(i, m2)) b = m2; else a = m1;
                }
                cx = (a + b) / 2; cl = loss_at(i, cx);
            } else if (l0 <= r0) { cx = lo + e; cl = l0; } else { cx = hi - e; cl = r0; }
            if (cl < best - 1e-12L * std::max<long double>(1, best)) { best = cl; best_i = i; best_x = cx; found = true; }
        }
        if (!found) break; // Algorithm 1 line 27: no candidate lowers the loss
        seq.insert(seq.begin() + std::ptrdiff_t(best_i + 1), P{double(best_x), true});
        out.virtual_features.push_back(double(best_x)); base = total(); current = base.sse(); ++out.rounds;
    }
    std::size_t k = 0; for (std::size_t i = 0; i < seq.size(); ++i) if (!seq[i].virt) out.slot[k++] = i;
    out.sse_after = double(current);
    return out;
}
} // namespace scaleli
