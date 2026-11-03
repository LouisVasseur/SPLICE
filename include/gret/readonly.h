// SPDX-License-Identifier: MIT
// SortedBoundIndex<K, P, Model>: read-only learned indexes in GRE's harness, so they can be compared with the rest on the same footing: one sorted array
// of (key, payload) pairs, and a model that bounds where a key can be, searched with a branch-free binary search. Exact for hits and misses; scan copies
// from the lower bound. spgm is the static PGM-index from GRE's copy, at error bounds 16, 64 and 256.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gret {

struct KeyBound {
  size_t lo, hi;  // the lower bound of a stored key is in [lo, hi)
};

template <class K, size_t Eps>
struct PgmBound {
  // The largest value of K is PGM's sentinel, and searching at it reads past its segments, as GRE's own runs on fb avoid by dropping that key. Here the
  // model is built without it, and anything above the keys it was built on goes straight to the end of the array.
  pgm::PGMIndex<K, Eps, 4> m;
  K top = 0;
  size_t built = 0, n = 0;
  void build(std::vector<K>& keys) {
    n = keys.size();
    if (keys.back() == std::numeric_limits<K>::max()) keys.pop_back();
    built = keys.size();
    top = keys.back();
    m = pgm::PGMIndex<K, Eps, 4>(keys);
  }
  KeyBound get(K x) const {
    if (__builtin_expect(x > top, 0)) return {built, n};
    const auto r = m.search(x);
    return {size_t(r.lo), size_t(r.hi)};
  }
  size_t bytes() const { return m.size_in_bytes(); }
};

template <class K, class P, class Model>
class SortedBoundIndex final : public indexInterface<K, P> {
  using Pair = std::pair<K, P>;

 public:
  explicit SortedBoundIndex(std::string name) : name_(std::move(name)) {}
  void init(Param* = nullptr) override {}
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (n == 0) throw std::invalid_argument("SortedBoundIndex: bulk_load of 0 keys");
    a_.assign(kv, kv + n);
    small_ = n < 64;  // too few for a model's builder to mean anything
    if (!small_) {
      std::vector<K> keys(n);
      for (size_t i = 0; i < n; ++i) keys[i] = kv[i].first;
      m_.build(keys);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu dir_bytes=%zu bytes_per_key=%.2f build_ms=%.1f\n", name_.c_str(), n, small_ ? size_t(0) : m_.bytes(),
                 double(memory_consumption()) / double(n), ms);
  }
  bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    if (__builtin_expect(x < uint64_t(a_.front().first) || x > uint64_t(a_.back().first), 0)) return false;
    const size_t i = lower(x, bound(key));
    if (i >= a_.size() || uint64_t(a_[i].first) != x) return false;
    val = a_[i].second;
    return true;
  }
  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    size_t i = 0;
    if (x > uint64_t(a_.back().first)) return 0;
    if (x > uint64_t(a_.front().first)) {
      const KeyBound b = bound(key);
      i = lower(x, b);
      // a model promises its bound for stored keys; for a key that was never stored, what the array says decides
      if (i == b.lo && i > 0 && uint64_t(a_[i - 1].first) >= x) i = size_t(std::lower_bound(a_.begin(), a_.begin() + std::ptrdiff_t(i), x, [](const Pair& p, uint64_t k) { return uint64_t(p.first) < k; }) - a_.begin());
      else if (i == b.hi && i < a_.size() && uint64_t(a_[i].first) < x) i = size_t(std::lower_bound(a_.begin() + std::ptrdiff_t(i), a_.end(), x, [](const Pair& p, uint64_t k) { return uint64_t(p.first) < k; }) - a_.begin());
    }
    const size_t got = std::min(n, a_.size() - i);
    std::copy_n(a_.data() + i, got, out);
    return got;
  }
  long long memory_consumption() override { return (long long)(a_.size() * sizeof(Pair) + (small_ ? size_t(0) : m_.bytes())); }

 private:
  KeyBound bound(K key) const { return small_ ? KeyBound{0, a_.size()} : m_.get(key); }
  size_t lower(uint64_t x, const KeyBound& b) const {  // the first position in [lo, hi] whose key is >= x, assuming the bound holds
    if (b.hi <= b.lo) return b.lo;
    const Pair* base = a_.data() + b.lo;
    size_t len = b.hi - b.lo;
    while (len > 1) {
      const size_t half = len / 2;
      base += uint64_t(base[half - 1].first) < x ? half : 0;
      len -= half;
    }
    return size_t(base - a_.data()) + size_t(uint64_t(base->first) < x);
  }

  std::string name_;
  bool small_ = false;
  std::vector<Pair> a_;
  Model m_;
};

}  // namespace gret
