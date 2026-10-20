// SPDX-License-Identifier: MIT
// ExceptionIndex<K, P>: the G of SPLICE's G/T/V, as SPLICE-H's exception list: outlier keys at the two ends of the
// key set are peeled into a small sorted array, so that the inner index (any registered index: raw, T, V or T+V) is
// built on the bulk only. A lookup is one predictable range check, k - lo > span (unsigned, so keys below lo wrap and
// fail it too), then either a binary search of the exceptions or the inner index's own get().
//
// Peeling rule: E = TRANSFORM_G_E (default 64) keys at most at each end. With T = the span of the keys left after
// trimming E keys at each end (the bulk without its tails), keys are peeled one at a time from the end whose last gap
// is wider, while the remaining span exceeds 2 T and that end's budget lasts, i.e. until the remaining span is within
// 2x of the trimmed span (SPLICE-H, docs/DESIGN.md section 2.1). Sets without outliers peel nothing.
// Exact for every loaded key; read-only (put/update/remove return false, scan returns 0).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <string>
#include <utility>
#include <vector>

namespace gret {

// The peeling rule above, shared with SpliceIndex: the bulk of the sorted kv[0, n) is kv[lo, hi).
template <class K, class P>
void peel_outliers(const std::pair<K, P>* kv, size_t n, size_t E, size_t& lo, size_t& hi) {
  lo = 0, hi = n;
  if (n > 2 * E + 2 && E > 0) {
    const uint64_t trimmed = uint64_t(kv[n - 1 - E].first) - uint64_t(kv[E].first);
    size_t pl = 0, ph = 0;
    // span > 2 * trimmed, written as span - trimmed > trimmed: 2 * trimmed wraps when the keys span more than 2^63 (osm)
    while (hi - lo > 2 && uint64_t(kv[hi - 1].first) - uint64_t(kv[lo].first) - trimmed > trimmed && (pl < E || ph < E)) {
      const uint64_t gl = uint64_t(kv[lo + 1].first) - uint64_t(kv[lo].first);
      const uint64_t gh = uint64_t(kv[hi - 1].first) - uint64_t(kv[hi - 2].first);
      if (ph < E && (gh >= gl || pl >= E)) { --hi; ++ph; } else { ++lo; ++pl; }
    }
  }
}

template <class K, class P, class Inner = indexInterface<K, P>>
class ExceptionIndex final : public indexInterface<K, P> {
  static constexpr bool kByValue = !std::is_same<Inner, indexInterface<K, P>>::value;

 public:
  // Inner = indexInterface<K, P>: owns a built index of any composition, called through its virtual interface.
  // Inner = a concrete GRE interface (g_<base>): held by value and called directly, so G adds only the range check.
  ExceptionIndex(std::string name, indexInterface<K, P>* inner) : name_(std::move(name)), owned_(inner) {}
  explicit ExceptionIndex(std::string name) : name_(std::move(name)) {}
  ExceptionIndex(const ExceptionIndex&) = delete;
  ExceptionIndex& operator=(const ExceptionIndex&) = delete;

  void init(Param* param = nullptr) override { inner().init(param); }

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* param = nullptr) override {
    const char* e = std::getenv("TRANSFORM_G_E");
    const size_t E = e && *e ? size_t(std::strtoull(e, nullptr, 10)) : 64;
    size_t lo, hi;  // the bulk is kv[lo, hi)
    peel_outliers(kv, n, E, lo, hi);
    exc_.assign(kv, kv + lo);
    exc_.insert(exc_.end(), kv + hi, kv + n);
    lo_ = n ? uint64_t(kv[lo].first) : 0;
    span_ = hi > lo ? uint64_t(kv[hi - 1].first) - lo_ : 0;
    std::fprintf(stderr, "gret: %s n=%zu exceptions=%zu (low %zu, high %zu) bulk=[%llu, %llu]\n", name_.c_str(), n,
                 exc_.size(), lo, n - hi, (unsigned long long)lo_, (unsigned long long)(lo_ + span_));
    inner().bulk_load(kv + lo, hi - lo, param);
  }

  // flatten: inline the base's own lookup here, so the wrapper adds a range check, not a second call frame (LIPP's
  // ~20 ns lookups lose 13-22% to the extra call otherwise)
  __attribute__((flatten)) bool get(K key, P& val, Param* param = nullptr) override {
    if (__builtin_expect(uint64_t(key) - lo_ > span_, 0)) {  // outside the bulk: the exceptions (rare, predictable)
      const auto it = std::lower_bound(exc_.begin(), exc_.end(), key,
                                       [](const std::pair<K, P>& a, K k) { return a.first < k; });
      if (it == exc_.end() || it->first != key) return false;
      val = it->second;
      return true;
    }
    return inner().get(key, val, param);
  }
  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K, size_t, std::pair<K, P>*, Param* = nullptr) override { return 0; }
  long long memory_consumption() override {
    const long long m = inner().memory_consumption();
    return m ? m + (long long)(exc_.size() * sizeof(std::pair<K, P>)) : 0;
  }

 private:
  Inner& inner() {
    if constexpr (kByValue) return held_;
    else return *owned_;
  }
  struct Empty {};
  std::string name_;
  std::unique_ptr<indexInterface<K, P>> owned_;
  typename std::conditional<kByValue, Inner, Empty>::type held_;
  std::vector<std::pair<K, P>> exc_;
  uint64_t lo_ = 0, span_ = 0;
};

}  // namespace gret
