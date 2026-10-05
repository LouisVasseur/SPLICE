#pragma once
// Key-space transformation in the style of NFL (Wu et al., PVLDB 2022).
//
// This is a clean-room re-implementation of the INFERENCE path only. The
// official repository (github.com/luffy06/NFL, GPL-3) trains a block neural
// autoregressive flow in PyTorch and exports a plain text weight file; its
// C++ side evaluates that file as a small tanh network on Intel MKL. The
// shipped author models are all "2D2H2L": input dimension 2, hidden 2, two
// weight matrices, no bias. We read the same text format so that either the
// author-trained weights or weights from tools/train_flow.py can be used.
//
// Differences from the original, stated explicitly:
//  * No MKL, no batching: matrices are 2x2, plain loops are used.
//  * The transformed value is used ONLY as the model feature. Records stay in
//    original key order, so exact search, fences and scans are unaffected even
//    when the network is not monotone. NFL instead sorts by transformed key.
//  * Keys are converted to double before transformation, as in the original
//    (which works on float64 keys). Precision above 2^53 is lost in the FEATURE
//    only; stored keys are exact.
#include "types.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>

namespace scaleli {
struct FlowTransform {
    unsigned in_dim = 0, hidden = 0, layers = 0;
    double mean = 0, var = 1;
    // weights[l] is row-major with rows x cols as recorded in the file.
    std::vector<std::vector<double>> weights;
    std::vector<std::pair<unsigned, unsigned>> shapes;

    static FlowTransform load(const std::filesystem::path& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open flow weights: " + path.string());
        FlowTransform f;
        if (!(in >> f.in_dim >> f.hidden >> f.layers >> f.mean >> f.var)) throw std::runtime_error("malformed flow header");
        if (f.in_dim != 1 && f.in_dim != 2) throw std::runtime_error("flow input dimension must be 1 or 2 (original in_dim=4 path is unsupported)");
        if (!f.layers || f.layers > 16 || !f.hidden || f.hidden > 64) throw std::runtime_error("unsupported flow shape");
        if (!(f.var > 0) || !std::isfinite(f.var) || !std::isfinite(f.mean)) throw std::runtime_error("invalid flow normalization");
        for (unsigned l = 0; l < f.layers; ++l) {
            unsigned rows, cols;
            if (!(in >> rows >> cols)) throw std::runtime_error("malformed flow matrix header");
            const unsigned expect_rows = l == 0 ? f.in_dim : f.hidden, expect_cols = l + 1 == f.layers ? f.in_dim : f.hidden;
            if (rows != expect_rows || cols != expect_cols) throw std::runtime_error("flow matrix shape does not match header");
            std::vector<double> w(std::size_t(rows) * cols);
            for (auto& x : w) { if (!(in >> x) || !std::isfinite(x)) throw std::runtime_error("malformed flow weight"); }
            f.weights.push_back(std::move(w)); f.shapes.emplace_back(rows, cols);
        }
        return f;
    }
    void save(const std::filesystem::path& path) const {
        std::ofstream out(path); if (!out) throw std::runtime_error("cannot write flow weights");
        out.precision(17); out << in_dim << '\t' << hidden << '\t' << layers << '\n' << mean << '\t' << var << '\n';
        for (unsigned l = 0; l < layers; ++l) {
            out << shapes[l].first << '\t' << shapes[l].second << '\n';
            for (unsigned r = 0; r < shapes[l].first; ++r) { for (unsigned c = 0; c < shapes[l].second; ++c) out << weights[l][std::size_t(r) * shapes[l].second + c] << '\t'; out << '\n'; }
        }
    }
    std::size_t bytes() const { std::size_t n = sizeof(*this); for (const auto& w : weights) n += w.size() * sizeof(double); return n; }

    double transform(double key, QueryStats* s = nullptr) const {
        if (s) { ++s->transform_calls;
            // Note every line the matvec below actually reads. The earlier version noted a
            // matrix by its first and last element only, which is equivalent to the true
            // footprint only while a matrix fits in at most two 64-byte lines (<= 8 doubles
            // when aligned). The shipped 2x2 author weights do fit, so the counts they
            // produced were right by accident; any wider flow (hidden > 2, or in_dim 4)
            // silently under-counted every line strictly between the two endpoints.
            s->note_range(this, sizeof(*this));                                  // in_dim, hidden, layers, mean, var, vector headers
            s->note_range(shapes.data(), shapes.size() * sizeof(shapes[0]));     // rows/cols read once per layer
            s->note_range(weights.data(), weights.size() * sizeof(weights[0]));  // the per-layer vector headers
            for (const auto& w : weights) s->note_range(w.data(), w.size() * sizeof(double)); }
        const double x = (key - mean) / var;
        double a[64], b[64];
        // Input encoding identical to the original "partition" encoder: [x, x - floor(x)].
        a[0] = x; if (in_dim == 2) a[1] = x - std::floor(x);
        unsigned width = in_dim;
        for (unsigned l = 0; l < layers; ++l) {
            const auto [rows, cols] = shapes[l];
            for (unsigned c = 0; c < cols; ++c) { double acc = 0; for (unsigned r = 0; r < rows; ++r) acc += a[r] * weights[l][std::size_t(r) * cols + c]; b[c] = acc; }
            const bool last = l + 1 == layers;
            for (unsigned c = 0; c < cols; ++c) a[c] = last ? b[c] : std::tanh(b[c]);
            width = cols;
        }
        double z = 0; for (unsigned c = 0; c < width; ++c) z += a[c]; // "sum" decoder
        return std::isfinite(z) ? z : 0.0;
    }
    double operator()(Key k, QueryStats* s = nullptr) const { return transform(static_cast<double>(k), s); }
};

// Tail conflict degree (NFL Definitions 3.1 and 3.2), on arbitrary sorted-order
// features. A single linear model is fitted from feature to rank, predicted
// positions are floored into [0, amplification * n), and the 99th percentile of
// per-position collision counts (minus one) is the tail conflict degree.
// Re-implemented from the paper's definition; the original also rescales the
// intercept so that the first key maps near zero, which we replicate.
inline unsigned tail_conflict_degree(std::span<const double> x, double amplification = 1.5, double tail = 0.99) {
    const std::size_t n = x.size(); if (n < 2) return 0;
    if (!(x.back() > x.front())) return unsigned(n - 1);
    long double mx = 0, my = 0, xx = 0, xy = 0;
    for (std::size_t i = 0; i < n; ++i) { const auto k = static_cast<long double>(i + 1); const long double dx = x[i] - mx, dy = static_cast<long double>(i) - my; mx += dx / k; my += dy / k; xx += dx * (x[i] - mx); xy += dx * (static_cast<long double>(i) - my); }
    double slope = xx > 0 ? double(xy / xx) : 0; if (!(slope > 0)) slope = double(n) / double(x.back() - x.front());
    const double intercept = -slope * x.front() + 0.5;
    const auto max_size = std::max<std::size_t>(1, std::min<std::size_t>(std::size_t(double(n) * amplification), std::size_t(std::max(1.0, std::floor(slope * x.back() + intercept) + 1))));
    std::vector<unsigned> counts; unsigned run = 1; std::size_t last = std::size_t(std::clamp(std::floor(slope * x[0] + intercept), 0.0, double(max_size - 1)));
    for (std::size_t i = 1; i < n; ++i) {
        const auto p = std::size_t(std::clamp(std::floor(slope * x[i] + intercept), 0.0, double(max_size - 1)));
        if (p == last) ++run; else { counts.push_back(run); last = p; run = 1; }
    }
    counts.push_back(run); std::sort(counts.begin(), counts.end());
    const auto idx = std::size_t(std::max(0.0, std::ceil(tail * double(counts.size())) - 1));
    return counts[std::min(idx, counts.size() - 1)] - 1;
}
} // namespace scaleli
