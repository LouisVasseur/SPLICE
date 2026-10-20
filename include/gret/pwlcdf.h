// SPDX-License-Identifier: MIT
// PwlCdf: the direct transform. A piecewise-linear interpolation of the empirical CDF of the build keys, through K+1
// knots at equi-rank positions, evaluated in exact integer arithmetic into [0, 2^52) (gret::kOutBits).
//
//   PwlCdf t = PwlCdf::build(sorted, n, 512);   uint64_t y = t(key);
//
// Knot j sits at x_j = sorted[round(j (n-1) / K)] with y_j = round(j / K * (2^52 - 1)). Equal x (only when K > n-1)
// keep the first y of their run, except the last run, which keeps y_K so that the outputs still reach 2^52 - 1.
// Inside segment j (x_j <= k < x_{j+1}):  T(k) = y_j + (((k - x_j) * mul_j) >> 64),  mul_j = floor((dy << 64) / dx),
// dx = x_{j+1} - x_j >= 1, dy = y_{j+1} - y_j < 2^52, all in unsigned __int128. No overflow:
//   dy << 64 < 2^116, and (k - x_j) * mul_j <= (dx - 1) * dy * 2^64 / dx < dy * 2^64 < 2^116 < 2^128,
// so the shifted product is at most dy - 1 and T(k) < y_{j+1}: T is non-decreasing in k, T(x_j) = y_j, and the result
// is the same on every machine (no floating point). Keys below x_0 give y_0 = 0, keys at or above x_last give y_K.
// The floor in mul_j makes T at most 1 below y_j + floor((k - x_j) dy / dx) (the error of the product is below dx).
//
// Knot search (on every lookup): a radix table on the top bits of k - x_0 gives the first candidate knot, then a
// branch-free binary lifting with a fixed number of steps (search_steps(), the same for every key) over the dense
// knot array. The table size is chosen at build to minimize the steps (below 4 entries per knot); with skewed keys
// it degenerates to one bucket, i.e. a plain branch-free binary search. Segment records {x, y, mul} are 32-byte
// aligned, so the final load touches one cache line.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "gret/common.h"

namespace gret {

class PwlCdf {
 public:
  __extension__ typedef unsigned __int128 u128;  // __extension__: no -Wpedantic warning

  // sorted: non-decreasing keys (distinct in GRE's use; equal knots are dropped anyway), n >= 1, K >= 1.
  static PwlCdf build(const uint64_t* sorted, size_t n, unsigned K) {
    return build_by(n, [sorted](size_t i) { return sorted[i]; }, K);
  }

  // The same on key_at(i), the i-th key; it is read only at the K + 1 knot ranks (TransformIndex: GRE's pairs).
  template <class KeyAt>
  static PwlCdf build_by(size_t n, KeyAt key_at, unsigned K) {
    if (n == 0) throw std::invalid_argument("PwlCdf::build: no keys");
    if (K == 0) throw std::invalid_argument("PwlCdf::build: K must be >= 1");
    std::vector<uint64_t> x, y;
    x.reserve(size_t(K) + 1);
    y.reserve(size_t(K) + 1);
    const u128 twoK = 2 * u128(K);
    for (uint64_t j = 0; j <= K; ++j) {
      // round half up of j (n-1) / K and j (2^52 - 1) / K: products below 2^97, exact in 128 bits
      const size_t i = size_t((2 * u128(j) * (n - 1) + K) / twoK);
      const uint64_t yj = uint64_t((2 * u128(j) * kOutMax + K) / twoK);
      const uint64_t xj = key_at(i);
      if (!x.empty() && xj < x.back()) throw std::invalid_argument("PwlCdf::build: keys are not sorted");
      if (x.empty() || xj != x.back()) {
        x.push_back(xj);
        y.push_back(yj);
      } else if (j == K && x.size() > 1) {
        y.back() = yj;  // the last run keeps y_K
      }
    }
    return from_knots(std::move(x), std::move(y), K);
  }

  // Explicit knots: x strictly increasing, y non-decreasing with consecutive values distinct, y < 2^52 (the knots
  // of an eps-PLA, SpliceIndex). The same evaluation and search as the equi-rank build.
  static PwlCdf from_knots(std::vector<uint64_t> x, std::vector<uint64_t> y, unsigned K = 0) {
    if (x.empty() || x.size() != y.size()) throw std::invalid_argument("PwlCdf::from_knots: need one y per x");
    for (size_t j = 1; j < x.size(); ++j)
      if (!(x[j] > x[j - 1]) || !(y[j] > y[j - 1]) || y[j] > kOutMax)
        throw std::invalid_argument("PwlCdf::from_knots: x and y must increase strictly, y < 2^52");
    PwlCdf t;
    t.K_ = K ? K : unsigned(x.size() - 1);
    t.m_ = x.size();
    t.x0_ = x.front();
    t.y0_ = y.front();
    t.xl_ = x.back();
    t.yl_ = y.back();
    if (t.m_ == 1) return t;  // every key maps to y_0: operator() never reaches the search
    t.seg_.resize(t.m_ - 1);
    for (size_t j = 0; j + 1 < t.m_; ++j) {
      // y is strictly increasing over kept knots (consecutive j differ by >= 2^52 / K >= 1 in y), dx >= 1
      const uint64_t dx = x[j + 1] - x[j], dy = y[j + 1] - y[j];
      t.seg_[j] = Seg{x[j], y[j], (u128(dy) << 64) / dx};
    }
    t.build_search(x);
    return t;
  }

  // Locate-only (SpliceIndex's top): the knots' search, without the interpolation; only locate(), segments(),
  // knots() and bytes() may be called on it.
  static PwlCdf locator(std::vector<uint64_t> x) {
    if (x.empty()) throw std::invalid_argument("PwlCdf::locator: no knots");
    for (size_t j = 1; j < x.size(); ++j)
      if (!(x[j] > x[j - 1])) throw std::invalid_argument("PwlCdf::locator: x must increase strictly");
    PwlCdf t;
    t.K_ = unsigned(x.size() - 1);
    t.m_ = x.size();
    t.x0_ = x.front();
    t.xl_ = x.back();
    if (t.m_ > 1) t.build_search(x);
    return t;
  }

  uint64_t operator()(uint64_t k) const {
    size_t seg;
    return eval(k, seg);
  }

  // T(k) and the segment it falls in: 0 below x_0, the last segment (m - 2) at or above x_last, 0 when m = 1.
  uint64_t eval(uint64_t k, size_t& seg) const {
    if (k < x0_) { seg = 0; return y0_; }
    if (k >= xl_) { seg = m_ > 1 ? m_ - 2 : 0; return yl_; }
    // invariant: xs_[j] <= k; the predicate xs_[j + d] <= k is monotone in d and true at most up to span <= 2^steps - 1,
    // and xs_ is padded with UINT64_MAX > k, so the lifting stops at the last knot <= k, which is below m-1.
    size_t j = start_[(k - x0_) >> shift_];
    for (size_t h = half_; h != 0; h >>= 1) j += xs_[j + h] <= k ? h : 0;
    seg = j;
    const Seg& s = seg_[j];
    return s.y + uint64_t((u128(k - s.x) * s.mul) >> 64);
  }
  // the knot interval of k only (eval without the interpolation): 0 below x0, m - 2 at or above the last knot
  size_t locate(uint64_t k) const {
    if (k < x0_) return 0;
    if (k >= xl_) return m_ > 1 ? m_ - 2 : 0;
    size_t j = start_[(k - x0_) >> shift_];
    for (size_t h = half_; h != 0; h >>= 1) j += xs_[j + h] <= k ? h : 0;
    return j;
  }
  size_t segments() const { return m_ > 1 ? m_ - 1 : 1; }

  unsigned K() const { return K_; }
  size_t knots() const { return m_; }  // after dropping equal x
  uint64_t knot_x(size_t i) const { return i + 1 < m_ ? seg_[i].x : xl_; }
  uint64_t knot_y(size_t i) const { return i + 1 < m_ ? seg_[i].y : yl_; }
  unsigned search_steps() const { return steps_; }
  size_t radix_buckets() const { return start_.size(); }
  size_t bytes() const {
    return sizeof(*this) + xs_.capacity() * sizeof(uint64_t) + seg_.capacity() * sizeof(Seg) +
           start_.capacity() * sizeof(uint32_t);
  }

 private:
  struct alignas(32) Seg {
    uint64_t x, y;  // knot j
    u128 mul;       // floor((y_{j+1} - y_j) 2^64 / (x_{j+1} - x_j)) < 2^116
  };

  static unsigned bit_width(uint64_t v) {
    unsigned w = 0;
    while (v) ++w, v >>= 1;
    return w;
  }

  // For bucket b of width 2^shift starting at x_0: the candidates for k in it run from the last knot <= its first key
  // to the last knot <= its last key. Tries 2^0 .. ~2m buckets and keeps the fewest steps (then the smallest table).
  void build_search(const std::vector<uint64_t>& x) {
    if (m_ - 2 > UINT32_MAX) throw std::invalid_argument("PwlCdf::build: too many knots");
    const uint64_t d1 = xl_ - x0_ - 1;  // keys reaching the search have k - x_0 in [0, d1]
    const unsigned w = bit_width(d1), rmax = bit_width(m_) + 1;
    unsigned best_steps = ~0u, best_shift = 0;
    std::vector<uint32_t> start, best_start;
    for (unsigned r = 0; r <= rmax; ++r) {
      unsigned shift = w > r ? w - r : 0;
      if (shift > 63) shift = 63;  // w = 64, r = 0: a shift by 64 is undefined, so 2 buckets instead of 1
      const uint64_t nb = (d1 >> shift) + 1;
      start.assign(size_t(nb), 0);
      size_t lo = 0, hi = 0, span = 0;
      for (uint64_t b = 0; b < nb; ++b) {
        const uint64_t first = x0_ + (b << shift);  // b << shift <= d1: no overflow
        const uint64_t last = b + 1 == nb ? xl_ - 1 : x0_ + ((b + 1) << shift) - 1;
        while (x[lo + 1] <= first) ++lo;  // x[m-1] = xl_ > last >= first stops both scans
        if (hi < lo) hi = lo;
        while (x[hi + 1] <= last) ++hi;
        start[size_t(b)] = uint32_t(lo);
        if (hi - lo > span) span = hi - lo;
      }
      const unsigned steps = bit_width(span);
      if (steps < best_steps) best_steps = steps, best_shift = shift, best_start.swap(start);
      if (best_steps == 0 || shift == 0 || nb > 2 * m_) break;
    }
    steps_ = best_steps;
    shift_ = best_shift;
    half_ = steps_ ? size_t(1) << (steps_ - 1) : 0;
    start_.swap(best_start);
    // reads reach start + 2^steps - 1 <= (m - 2) + 2^steps - 1
    xs_.assign(x.begin(), x.end());
    xs_.resize(m_ - 1 + (size_t(1) << steps_), UINT64_MAX);
  }

  unsigned K_ = 0, steps_ = 0, shift_ = 0;
  size_t m_ = 1, half_ = 0;
  uint64_t x0_ = 0, y0_ = 0, xl_ = 0, yl_ = 0;  // a default PwlCdf maps every key to 0
  std::vector<uint64_t> xs_;                    // knot x, padded with UINT64_MAX for the fixed-step search
  std::vector<Seg> seg_;                        // m - 1 segments
  std::vector<uint32_t> start_;                 // radix table: first candidate knot per bucket
};

}  // namespace gret
