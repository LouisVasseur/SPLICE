// SPDX-License-Identifier: MIT
// RadixLineIndex<K, P, B>: HashLineIndex's buckets with a model in place of the hash, so key order is kept and the keys
// spread more evenly than a hash spreads them, on a path nearly as short: one directory load and one integer multiply.
// Read-only, exact for hits and misses. SPLICE's parts, with T reduced to what a lookup can afford:
//
//   G  peel_outliers (exceptions.h): end outliers go to a small sorted array behind one range check.
//   T  a directory of equal-width cells over the bulk's key range (2^RADIX_BITS at most, default 2^16, and at most about
//      n / 8), each cell {first bucket, buckets}: within a cell the key's bucket is a straight line. The cell's bounds
//      are the exact key counts, so this is the keys' CDF through equal-width knots, in integer arithmetic.
//   V  buckets are given in proportion to keys, HASHLINE_SPK slots per key (default 1.25): cell j starts at bucket
//      floor(keys before j * spk / B). Keys go in sorted order into the first free slot of their bucket or later, and an
//      empty slot holds a copy of the next pair, as in SpliceIndex.
//
// get: the range check, the cell, the bucket, then HashLineIndex's read: count the keys below x without branching, and
// the first key >= x decides; all B below x moves to the next bucket, at most d (exact, from the build) further.
// Keys must be sorted and unique, as GRE's driver guarantees. put, update and remove return false. scan reads the sorted slots
// from the start key's predicted bucket on.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gret/exceptions.h"
#include "gret/hashline.h"

namespace gret {

template <class K, class P, unsigned B>
class RadixLineIndex final : public indexInterface<K, P> {
  using Slot = std::pair<uint64_t, P>;
  struct Cell {
    uint32_t base;   // first bucket
    uint32_t count;  // buckets
  };

 public:
  explicit RadixLineIndex(std::string name) : name_(std::move(name)) {}

  void init(Param* = nullptr) override {}

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (n == 0) throw std::invalid_argument("RadixLineIndex: bulk_load of 0 keys");
    const char* e = std::getenv("HASHLINE_SPK");
    const char* rb = std::getenv("RADIX_BITS");
    const char* ge = std::getenv("TRANSFORM_G_E");
    const double spk = e && *e ? std::strtod(e, nullptr) : 1.25;
    const unsigned want = rb && *rb ? unsigned(std::strtoul(rb, nullptr, 10)) : 16;
    if (!(spk > 0)) throw std::invalid_argument("RadixLineIndex: HASHLINE_SPK > 0");
    size_t lo, hi;
    peel_outliers(kv, n, ge && *ge ? size_t(std::strtoull(ge, nullptr, 10)) : size_t(64), lo, hi);
    exc_.assign(kv, kv + lo);
    exc_.insert(exc_.end(), kv + hi, kv + n);
    const std::pair<K, P>* d = kv + lo;
    const size_t m = hi - lo;
    lo_ = uint64_t(d[0].first);
    span_ = uint64_t(d[m - 1].first) - lo_;
    unsigned sb = 0, nb = 0;  // bits of the span, and of the cells the keys can pay for
    for (uint64_t v = span_; v; v >>= 1) ++sb;
    for (size_t v = m / 8; v; v >>= 1) ++nb;
    const unsigned bits = std::max(std::min({want, nb, sb}), std::min(1u, sb));  // a span of 64 bits needs a shift below 64
    shift_ = sb - bits;
    mask_ = shift_ ? (uint64_t(1) << shift_) - 1 : 0;
    rs_ = shift_ > 32 ? shift_ - 32 : 0;
    ls_ = shift_ < 32 ? 32 - shift_ : 0;
    const size_t cells = size_t(span_ >> shift_) + 1;
    std::vector<size_t> upto(cells + 1, 0);  // keys before cell j
    for (size_t r = 0; r < m; ++r) ++upto[size_t((uint64_t(d[r].first) - lo_) >> shift_) + 1];
    for (size_t j = 0; j < cells; ++j) upto[j + 1] += upto[j];
    const double per = spk / B;
    if (double(m) * per >= 4294967295.0) throw std::length_error("RadixLineIndex: 2^32 buckets");
    cells_.resize(cells);
    for (size_t j = 0; j < cells; ++j) {
      const size_t b0 = size_t(double(upto[j]) * per), b1 = j + 1 < cells ? size_t(double(upto[j + 1]) * per) : size_t(std::ceil(double(m) * per));
      cells_[j] = Cell{uint32_t(b0), uint32_t(b1 - b0)};
    }
    size_t prev = SIZE_MAX;
    LineCount lc;  // over all bulk keys, each looked up once
    for (size_t r = 0; r < m; ++r) {
      const size_t p = bucket(uint64_t(d[r].first) - lo_), pos = place(p, prev);
      lc.add(pos / B - p + 1);
      prev = pos;
    }
    d_ = lc.worst() - 1;
    total_ = std::max(size_t(std::ceil(double(m) * per)), prev / B + 1);
    slots_.resize(total_ * B);
    size_t i = 0;
    prev = SIZE_MAX;
    for (size_t r = 0; r < m; ++r) {
      const size_t pos = place(bucket(uint64_t(d[r].first) - lo_), prev);
      const Slot s(uint64_t(d[r].first), d[r].second);
      while (i <= pos) slots_[i++] = s;
      prev = pos;
    }
    const Slot last(uint64_t(d[m - 1].first), d[m - 1].second);
    while (i < slots_.size()) slots_[i++] = last;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu B=%u exceptions=%zu cells=%zu slots/key=%.3f buckets=%zu d=%zu E[buckets]=%.4f lines_p99=%zu lines_p999=%zu lines_max=%zu dir_bytes=%zu bytes_per_key=%.2f build_ms=%.1f\n",
                 name_.c_str(), n, B, exc_.size(), cells, double(slots_.size()) / double(n), total_, d_, lc.mean(), lc.tail(0.99), lc.tail(0.999), lc.worst(),
                 cells * sizeof(Cell), double(memory_consumption()) / double(n), ms);
  }

  bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t x = uint64_t(key), v = x - lo_;
    if (__builtin_expect(v > span_, 0)) return find_exception(key, val);
    size_t b = bucket(v);
    const size_t b0 = b;
    for (const Slot* l = &slots_[b * B];; l += B) {
      size_t c = 0;
      for (unsigned j = 0; j < B; ++j) c += size_t(l[j].first < x);
      if (__builtin_expect(c < B, 1)) {  // the first key >= x decides
        if (l[c].first != x) return false;
        val = l[c].second;
        return true;
      }
      if (++b > b0 + d_ || b >= total_) return false;
    }
  }

  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    return scan_sorted_slots(slots_, exc_, key, n, out, [this](uint64_t x) { return lower_slot(x); });
  }
  long long memory_consumption() override {
    return (long long)(slots_.size() * sizeof(Slot) + cells_.size() * sizeof(Cell) + exc_.size() * sizeof(std::pair<K, P>));
  }

 private:
  // v = key - lo_ <= span_: the cell's first bucket, plus the key's position in the cell (32 bits of fraction) scaled to its buckets
  size_t bucket(uint64_t v) const {
    const Cell c = cells_[v >> shift_];
    return size_t(c.base) + size_t((((v & mask_) >> rs_ << ls_) * uint64_t(c.count)) >> 32);
  }
  // the first slot whose key is >= x, for x within the bulk: bucket() is non-decreasing in the key, so every key >= x sits at or
  // after x's predicted bucket, and the slots are sorted
  size_t lower_slot(uint64_t x) const {
    size_t i = bucket(x - lo_) * B;
    while (i < slots_.size() && slots_[i].first < x) ++i;
    return i;
  }
  static size_t place(size_t p, size_t prev) {  // the first free slot of bucket p or later
    const size_t s = p * B;
    return prev == SIZE_MAX || s > prev ? s : prev + 1;
  }
  bool find_exception(K key, P& val) const {
    const auto it = std::lower_bound(exc_.begin(), exc_.end(), key,
                                     [](const std::pair<K, P>& a, K k) { return a.first < k; });
    if (it == exc_.end() || it->first != key) return false;
    val = it->second;
    return true;
  }

  std::string name_;
  uint64_t lo_ = 1, span_ = 0, mask_ = 0;
  unsigned shift_ = 0, rs_ = 0, ls_ = 32;
  size_t total_ = 1, d_ = 0;
  std::vector<Cell> cells_;
  std::vector<Slot, BucketAlloc<Slot>> slots_;
  std::vector<std::pair<K, P>> exc_;
};

}  // namespace gret
