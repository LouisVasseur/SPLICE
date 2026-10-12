#pragma once
#include "types.hpp"
#include <cmath>
#include <limits>
#include <span>

namespace scaleli {
// A local affine model, NOT the PGM algorithm and NOT an epsilon certificate.
// Subtract integers before converting to floating point, preserving local detail
// even when keys are close to UINT64_MAX. Exact fences establish correctness.
struct LinearModel {
    Key origin = 0;
    std::uint64_t span = 1;
    double slope = 0, intercept = 0;
    double normalized(Key k) const {
        return k >= origin ? double(k - origin) / double(span)
                           : -double(origin - k) / double(span);
    }
    double predict_x(double x) const {
        const auto y = std::fma(slope, x, intercept);
        return std::isfinite(y) ? y : 0.0;
    }
    double predict(Key k) const { return predict_x(normalized(k)); }
    // Fit directly in feature space (features may be flow-transformed keys).
    // origin/span are still recorded from `keys` so normalized() stays defined.
    void fit_xy(std::span<const Key> keys, std::span<const double> x, std::span<const double> targets) {
        if (keys.size() != x.size() || x.size() != targets.size()) throw std::invalid_argument("model feature size mismatch");
        *this = {};
        if (keys.empty()) return;
        origin = keys.front();
        span = std::max<std::uint64_t>(1, keys.back() - origin);
        long double mx = 0, my = 0, xx = 0, xy = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const long double xi = x[i], y = targets[i];
            const auto n = static_cast<long double>(i + 1);
            const auto dx = xi - mx, dy = y - my;
            mx += dx / n; my += dy / n;
            xx += dx * (xi - mx); xy += dx * (y - my);
        }
        slope = xx > 0 ? double(xy / xx) : 0;
        intercept = double(my - static_cast<long double>(slope) * mx);
    }
    void fit(std::span<const Key> keys, std::span<const double> targets) {
        if (keys.size() != targets.size()) throw std::invalid_argument("model target size mismatch");
        *this = {};
        if (keys.empty()) return;
        origin = keys.front();
        span = std::max<std::uint64_t>(1, keys.back() - origin);
        long double mx = 0, my = 0, xx = 0, xy = 0;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            const long double x = static_cast<long double>(keys[i] - origin) / span;
            const long double y = targets[i];
            const auto n = static_cast<long double>(i + 1);
            const auto dx = x - mx, dy = y - my;
            mx += dx / n; my += dy / n;
            xx += dx * (x - mx); xy += dx * (y - my);
        }
        slope = xx > 0 ? double(xy / xx) : 0;
        intercept = double(my - static_cast<long double>(slope) * mx);
    }
};
} // namespace scaleli
