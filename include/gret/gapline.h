// SPDX-License-Identifier: MIT
// FullG: the whole of G as the coordinate of AdaptiveLineIndex in gf4. After the far keys are peeled (as in ab4), the period rule
// of ruleline.h maps the keys when $RULE_FILE is set, and then the GAP_K widest gaps between consecutive stored keys, measured in
// that coordinate, shrink to the typical gap W of the others. Equal-width cells then fall on keys rather than on empty space, so
// the directory needs fewer cells and levels for the same lines. A lookup finds its gap with a branch-free binary search over the
// gap starts and subtracts what was removed before it.
//
//   gap j: from stored key s_j to the next stored key, width g_j, removed r_j = g_j - W, C_j = r summed over the gaps before j.
//   y in (s_j, s_{j+1}]: coordinate = y - C_j - min(max(y - s_j - W, 0), r_j)
// Non-decreasing in y, and strictly increasing on stored keys (none lies inside a gap), which AdaptiveLineIndex's build checks;
// a lookup compares raw keys, as with the rule alone. $GAP_K = 0 (the default) leaves the rule's coordinate as it is.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

#include "gret/ruleline.h"

namespace gret {

class FullG {
 public:
  static constexpr bool kIdentity = false;
  template <class Pair>
  void load(const Pair* d, size_t m) {
    rule_.load(d, m);
    s_.clear(), c_.clear(), r_.clear();
    const char* e = std::getenv("GAP_K");
    const size_t k = e && *e ? size_t(std::strtoull(e, nullptr, 10)) : 0;
    if (k == 0 || m < 3) return;
    // the k widest gaps, in one pass: a min-heap of (width, start)
    using G = std::pair<uint64_t, uint64_t>;
    std::priority_queue<G, std::vector<G>, std::greater<G>> h;
    uint64_t prev = rule_(uint64_t(d[0].first)), sum = 0;
    const uint64_t first = prev;
    for (size_t r = 1; r < m; ++r) {
      const uint64_t y = rule_(uint64_t(d[r].first)), g = y - prev;
      if (h.size() < k) h.push({g, prev}), sum += g;
      else if (g > h.top().first) sum -= h.top().first, h.pop(), h.push({g, prev}), sum += g;
      prev = y;
    }
    const size_t rest = m - 1 - h.size();
    w_ = rest ? std::max<uint64_t>(1, (prev - first - sum) / rest) : 1;  // the mean gap of the others
    std::vector<G> gaps;
    for (; !h.empty(); h.pop())
      if (h.top().first > w_) gaps.push_back({h.top().second, h.top().first - w_});
    std::sort(gaps.begin(), gaps.end());
    uint64_t cum = 0;
    for (const auto& [start, removed] : gaps) s_.push_back(start), c_.push_back(cum), r_.push_back(removed), cum += removed;
    removed_ = cum;
  }
  uint64_t operator()(uint64_t x) const {
    const uint64_t y = rule_(x);
    size_t n = s_.size();
    if (n == 0) return y;
    const uint64_t* s = s_.data();
    size_t base = 0;  // the number of gap starts below y, branch-free
    while (n > 1) {
      const size_t half = n / 2;
      base = s[base + half] < y ? base + half : base;
      n -= half;
    }
    const size_t i = base + (s[base] < y);
    if (i == 0) return y;
    const size_t j = i - 1;
    const uint64_t into = y - s[j], extra = into > w_ ? std::min(into - w_, r_[j]) : 0;
    return y - c_[j] - extra;
  }
  size_t bytes() const { return rule_.bytes() + s_.size() * 3 * sizeof(uint64_t); }
  size_t gaps() const { return s_.size(); }
  uint64_t removed() const { return removed_; }
  uint64_t kept_width() const { return w_; }

 private:
  PeriodRule rule_;
  std::vector<uint64_t> s_, c_, r_;
  uint64_t w_ = 1, removed_ = 0;
};

template <class K, class P, unsigned B>
using GapLineIndex = AdaptiveLineIndex<K, P, B, FullG>;

}  // namespace gret
