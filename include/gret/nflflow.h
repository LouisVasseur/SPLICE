// SPDX-License-Identifier: MIT
// NflFlow: NFL's own "numerical flow" (Wu et al., PVLDB 2022), inference only, evaluated on the weight files that NFL's
// trainer writes (github.com/luffy06/NFL fe71172, train/numerical_flow.py save_weights). Written from the published
// model and file format; NFL's GPL-3 code is not copied. What NFL's C++ computes (src/models/numerical_flow.h and
// bnaf.h, there with MKL dgemm/vdTanh), per key k converted to double as NFL does:
//
//   x = (double(k) - mean) / var                       min-max scaling to [0, shifts] on NFL's training keys
//   a = [x, x - floor(x)]   (in_dim 2; [x] for in_dim 1)  "partition" encoder
//   h = tanh(a W_0), then h = tanh(h W_l) for l = 1 .. L-2, o = h W_{L-1}   (no bias; the last layer has no tanh)
//   z = o_0 + o_1           (in_dim 2; o_0 for in_dim 1)  "sum" decoder
//
// The flow is not monotone in general (the sum decoder can fold the key range), so z is mapped to the wrapper's
// [0, 2^52) by the z range of the build keys, fitted in bulk_load: T(k) = clamp(floor((z - zlo) * scale), 0, 2^52-1),
// scale = (2^52 - 1) / (zhi - zlo). That is a linear rescale, so the index sees NFL's transformed keys up to scale;
// equal T (folding or rounding) go to the wrapper's overflow, which keeps every lookup exact. Non-finite z map to 0.
//
// File (text, NFL's format):  in_dim hidden layers / mean var / then per layer: rows cols and rows*cols weights,
// row-major: layer 0 is in_dim x hidden, the middle ones hidden x hidden, the last hidden x in_dim.
// Reproducibility: bulk_load and get() must compute the same T bit for bit, so the evaluation is out of line, with
// a * b + c contraction off (clang pragma / GCC optimize attribute), as in flow.h.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gret/common.h"

#if defined(__clang__)
#define GRET_NFL_STRICT _Pragma("clang fp contract(off)")
#define GRET_NFL_FN
#define GRET_NFL_OUTLINE __attribute__((noinline))
#elif defined(__GNUC__)
#define GRET_NFL_STRICT
#define GRET_NFL_FN __attribute__((optimize("fp-contract=off")))
#define GRET_NFL_OUTLINE __attribute__((noipa))
#else
#define GRET_NFL_STRICT
#define GRET_NFL_FN
#define GRET_NFL_OUTLINE
#endif

namespace gret {

class NflFlow {
 public:
  static constexpr unsigned kMaxDim = 64, kMaxLayers = 16;

  static NflFlow load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("NflFlow: cannot open " + path);
    NflFlow f;
    if (!(in >> f.in_ >> f.hidden_ >> f.layers_ >> f.mean_ >> f.var_)) throw std::runtime_error("NflFlow: bad header in " + path);
    if ((f.in_ != 1 && f.in_ != 2) || f.hidden_ < 1 || f.hidden_ > kMaxDim || f.layers_ < 2 || f.layers_ > kMaxLayers)
      throw std::runtime_error("NflFlow: unsupported shape in " + path);
    if (!std::isfinite(f.mean_) || !std::isfinite(f.var_) || !(f.var_ > 0)) throw std::runtime_error("NflFlow: bad mean/var");
    for (unsigned l = 0; l < f.layers_; ++l) {
      unsigned rows, cols;
      if (!(in >> rows >> cols)) throw std::runtime_error("NflFlow: missing matrix header");
      const unsigned er = l == 0 ? f.in_ : f.hidden_, ec = l + 1 == f.layers_ ? f.in_ : f.hidden_;
      if (rows != er || cols != ec) throw std::runtime_error("NflFlow: matrix shape does not match the header");
      std::vector<double> w(size_t(rows) * cols);
      for (double& v : w)
        if (!(in >> v) || !std::isfinite(v)) throw std::runtime_error("NflFlow: bad weight");
      f.w_.push_back(std::move(w));
    }
    std::string extra;
    if (in >> extra) throw std::runtime_error("NflFlow: trailing data in " + path);
    return f;
  }

  // z, NFL's transformed key.
  GRET_NFL_OUTLINE GRET_NFL_FN double z(uint64_t k) const {
    GRET_NFL_STRICT
    const double x = (double(k) - mean_) / var_;
    double a[kMaxDim], b[kMaxDim];
    a[0] = x;
    if (in_ == 2) a[1] = x - std::floor(x);
    unsigned width = in_;
    for (unsigned l = 0; l < layers_; ++l) {
      const unsigned rows = width, cols = l + 1 == layers_ ? in_ : hidden_;
      const double* W = w_[l].data();
      for (unsigned c = 0; c < cols; ++c) {
        double acc = 0;
        for (unsigned r = 0; r < rows; ++r) acc += a[r] * W[size_t(r) * cols + c];
        b[c] = l + 1 == layers_ ? acc : std::tanh(acc);
      }
      for (unsigned c = 0; c < cols; ++c) a[c] = b[c];
      width = cols;
    }
    return in_ == 2 ? a[0] + a[1] : a[0];
  }

  // The z range of the build keys (bulk_load), for the rescale to [0, 2^52).
  template <class KeyAt>
  void fit(size_t n, KeyAt key_at) {
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (size_t i = 0; i < n; ++i) {
      const double v = z(key_at(i));
      if (std::isfinite(v)) { lo = v < lo ? v : lo; hi = v > hi ? v : hi; }
    }
    zlo_ = std::isfinite(lo) ? lo : 0;
    zhi_ = std::isfinite(hi) ? hi : 0;
    scale_ = zhi_ > zlo_ ? double(kOutMax) / (zhi_ - zlo_) : 0;
  }

  GRET_NFL_OUTLINE GRET_NFL_FN uint64_t operator()(uint64_t k) const {
    GRET_NFL_STRICT
    const double v = z(k);
    if (!std::isfinite(v)) return 0;
    const double t = (v - zlo_) * scale_;
    return t <= 0 ? 0 : t >= double(kOutMax) ? kOutMax : uint64_t(t);
  }

  double zlo() const { return zlo_; }
  double zhi() const { return zhi_; }
  unsigned in_dim() const { return in_; }
  unsigned hidden() const { return hidden_; }
  unsigned layers() const { return layers_; }
  size_t bytes() const {
    size_t s = sizeof(*this);
    for (const auto& w : w_) s += w.size() * sizeof(double);
    return s;
  }

 private:
  unsigned in_ = 0, hidden_ = 0, layers_ = 0;
  double mean_ = 0, var_ = 1, zlo_ = 0, zhi_ = 0, scale_ = 0;
  std::vector<std::vector<double>> w_;
};

}  // namespace gret

#undef GRET_NFL_STRICT
#undef GRET_NFL_FN
#undef GRET_NFL_OUTLINE
