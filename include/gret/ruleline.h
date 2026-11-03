// SPDX-License-Identifier: MIT
// PeriodRule: G as one rule computed at lookup rather than a table of gaps, the coordinate of AdaptiveLineIndex in gr4. Keys that
// fall into rows of a fixed width, with most of each row empty, are hard for a line and for equal-width cells; the rule closes the
// empty part of every row. The blocks are read from a file, by the detector or by hand.
//
//   blocks c = {first key, period P}, from $RULE_FILE ("first period" per line, first keys increasing, after an optional "S value"
//   line). With x in block c, q = (x - first) / P and r = (x - first) mod P, the coordinate is
//     y = base(c) + q (S + P) + floor(S F(r)) + r
//   F is the keys' empirical distribution of r, tabulated once per octave of r and linear between knots; $RULE_S overrides S (default
//   2^26). It keeps stored keys strictly increasing, which AdaptiveLineIndex's build checks. Without $RULE_FILE the coordinate is
//   the key.
//
// A lookup compares raw keys, so a key that was never loaded returns false wherever the rule sends it: exactness needs the rule
// ordered on stored keys only. scan gallops from the prediction on the sorted slots, and needs no more than that either.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "gret/adaptiveline.h"

namespace gret {

class PeriodRule {
 public:
  static constexpr bool kIdentity = false;
  template <class Pair>
  void load(const Pair* d, size_t m) {
    first_.clear(), period_.clear(), base_.clear();
    const char* path = std::getenv("RULE_FILE");
    if (!path || !*path) return;
    FILE* f = std::fopen(path, "r");
    if (!f) throw std::runtime_error("PeriodRule: cannot read RULE_FILE");
    unsigned long long a, p;
    S_ = uint64_t(67108864);
    if (std::fscanf(f, "S %llu", &a) == 1) S_ = a;
    while (std::fscanf(f, "%llu %llu", &a, &p) == 2) first_.push_back(a), period_.push_back(p);
    std::fclose(f);
    if (first_.empty() || first_[0] > uint64_t(d[0].first)) throw std::runtime_error("PeriodRule: the rule does not cover the first key");
    const char* e = std::getenv("RULE_S");
    if (e && *e) S_ = uint64_t(std::strtod(e, nullptr));
    base_.assign(first_.size(), 0);
    for (size_t c = 0; c + 1 < first_.size(); ++c) {
      const uint64_t rows = (first_[c + 1] - first_[c] + period_[c] - 1) / period_[c];
      base_[c + 1] = base_[c] + rows * (S_ + period_[c]);
    }
    inv_.resize(first_.size());
    for (size_t c = 0; c < first_.size(); ++c) inv_[c] = ~uint64_t(0) / period_[c];
    hb_ = 1;
    while (hb_ * 2 < first_.size()) hb_ *= 2;
    std::vector<uint64_t> h(66, 0);
    for (size_t r = 0; r < m; ++r) {
      const uint64_t x = uint64_t(d[r].first);
      const size_t c = block_of(x);
      ++h[size_t(bits_of((x - first_[c]) % period_[c]))];
    }
    uint64_t cum = 0;
    u_[0] = 0;
    for (int k = 0; k <= 64; ++k) {
      cum += h[size_t(k)];
      u_[k + 1] = double(cum) / double(m);
    }
    u_[0] = u_[1];
    for (int j = 0; j <= 64; ++j) {  // the slope of F between knots, so a lookup does not divide
      const double den = j > 1 ? std::ldexp(1.0, j - 1) : 1.0;
      slope_[j] = (u_[j + 1] - u_[j]) / den;
    }
  }
  uint64_t operator()(uint64_t x) const {
    if (first_.empty()) return x;
    const size_t c = block_of(x);
    __extension__ typedef unsigned __int128 u128;
    const uint64_t xp = x - first_[c], pd = period_[c];
    uint64_t q = uint64_t((u128(xp) * inv_[c]) >> 64), r = xp - q * pd;  // the reciprocal is short by a little, so q may be one or two low
    while (__builtin_expect(r >= pd, 0)) ++q, r -= pd;
    return base_[c] + q * (S_ + pd) + uint64_t(double(S_) * phase(r)) + r;
  }
  size_t bytes() const { return first_.size() * (sizeof(uint64_t) * 4) + sizeof(u_) + sizeof(slope_); }

 private:
  size_t block_of(uint64_t x) const {
    size_t c = 0;
    for (size_t h = hb_; h; h >>= 1) c += (c + h < first_.size() && x >= first_[c + h]) ? h : 0;
    return c;
  }
  static int bits_of(uint64_t r) { return r ? 64 - __builtin_clzll(r) : 0; }
  // the distribution of r: u_[k + 1] holds it at r = 2^k - 1, linear in r between knots
  double phase(uint64_t r) const {
    const int k = bits_of(r);
    const uint64_t half = k ? uint64_t(1) << (k - 1) : 0;
    return u_[k] + slope_[k] * double(r - (half - 1));
  }

  uint64_t S_ = 67108864;
  size_t hb_ = 0;
  double u_[66] = {}, slope_[65] = {};
  std::vector<uint64_t> first_, period_, base_, inv_;
};

template <class K, class P, unsigned B>
using RuleLineIndex = AdaptiveLineIndex<K, P, B, PeriodRule>;

}  // namespace gret
