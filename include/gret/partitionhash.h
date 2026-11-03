// SPDX-License-Identifier: MIT
// PartitionHashIndex<K, P, B>: T, then G, then V. A directory routes on the raw key, as in AdaptiveLineIndex, and inside each
// leaf the keys are hashed, so crowding inside a leaf cannot cost more lines than a hash table's, however bumpy the keys.
// Read-only, exact for hits and misses. Key order is kept between leaves and lost inside them, so a range scan reads whole
// leaves of at most about AR_LEAF keys and sorts what it returns.
//
//   G  peel_outliers, for the directory's range. Then, inside a leaf, mul64 (hashline.h), a bijection on 64 bits.
//   T  a root of equal-width cells (2^RADIX_BITS at most, default 2^18, and at most about n / 8), then at most AR_DEPTH levels
//      of child blocks (default 3, root excluded). A node splits only while it holds more than AR_LEAF keys (default 4096);
//      k aims at AR_LEAF keys per child, at most AR_KMAX bits (default 8). Leaf: {first bucket, buckets}.
//   V  HASHLINE_SPK slots per key (default 1.25), buckets in proportion to the exact key counts. Inside a leaf, keys sorted
//      by their hash go into the first free slot of bucket first + floor(z * buckets / 2^64) or later; an empty slot holds a
//      copy of the next pair.
//
// get walks the levels, then compares each slot of the bucket for equality (a bucket can
// also hold spill from the leaf before it, so its slots are not sorted by z), at most d buckets further.
// Keys must be sorted and unique, as GRE's driver guarantees. put, update and remove return false.
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
class PartitionHashIndex final : public indexInterface<K, P> {
  __extension__ typedef unsigned __int128 u128;  // __extension__: no -Wpedantic warning
  using Slot = std::pair<uint64_t, P>;
  struct Node {
    uint32_t a;  // leaf: first bucket. Inner (top bit set): first child, in nodes_
    uint32_t b;  // leaf: buckets. Inner: k, 2^k children
  };
  static constexpr uint32_t kInner = 0x80000000u;

  static double env_d(const char* k, double def) {
    const char* e = std::getenv(k);
    return e && *e ? std::strtod(e, nullptr) : def;
  }

 public:
  explicit PartitionHashIndex(std::string name) : name_(std::move(name)) {}

  void init(Param* = nullptr) override {}

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (n == 0) throw std::invalid_argument("PartitionHashIndex: bulk_load of 0 keys");
    const double spk = env_d("HASHLINE_SPK", 1.25);
    const unsigned want = unsigned(env_d("RADIX_BITS", 18));
    const double leaf = env_d("AR_LEAF", 4096), kmax = env_d("AR_KMAX", 8), depth = env_d("AR_DEPTH", 3);
    if (!(spk > 0) || !(leaf >= 1) || !(kmax >= 1 && kmax <= 30) || !(depth >= 0 && depth <= 64))
      throw std::invalid_argument("PartitionHashIndex: HASHLINE_SPK > 0, AR_LEAF >= 1, AR_KMAX in 1..30, AR_DEPTH in 0..64");
    leaf_ = size_t(leaf);
    kmax_ = unsigned(kmax);
    depth_ = unsigned(depth);
    sorted_scan_ = env_d("PH_SCAN_SORTED", 1) != 0;
    size_t lo, hi;
    peel_outliers(kv, n, size_t(env_d("TRANSFORM_G_E", 64)), lo, hi);
    exc_.assign(kv, kv + lo);
    exc_.insert(exc_.end(), kv + hi, kv + n);
    d_keys_ = kv + lo;
    m_ = hi - lo;
    per_ = spk / B;
    lo_ = uint64_t(d_keys_[0].first);
    span_ = uint64_t(d_keys_[m_ - 1].first) - lo_;
    unsigned sb = 0, nb = 0;  // bits of the span, and of the cells the keys can pay for
    for (uint64_t v = span_; v; v >>= 1) ++sb;
    for (size_t v = m_ / 8; v; v >>= 1) ++nb;
    const unsigned bits = std::max(std::min({want, nb, sb}), std::min(1u, sb));  // a span of 64 bits needs a shift below 64
    shift_ = sb - bits;
    const size_t cells = size_t(span_ >> shift_) + 1;
    build_all(cells);
    if (prefer_mix64(lines_, mix64_lines<B>(d_keys_, m_, spk))) {  // one multiply clumps on these keys: see hashline.h
      fallback_ = true;
      build_all(cells);
    }
    nodes_.shrink_to_fit();  // memory_consumption counts size(), and resize leaves spare capacity
    d_ = lines_.worst() - 1;
    total_ = std::max(rank_bucket(m_), prev_ / B + 1);
    slots_.resize(total_ * B);
    const Slot last = slots_[prev_];  // trailing slots: any pair of the bulk keeps every slot a real key
    while (w_ < slots_.size()) slots_[w_++] = last;
    d_keys_ = nullptr;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu B=%u exceptions=%zu cells=%zu nodes=%zu leaves=%zu biggest_leaf=%zu slots/key=%.3f buckets=%zu d=%zu E[buckets]=%.4f lines_p99=%zu lines_p999=%zu lines_max=%zu fallback=%d E[levels]=%.3f dir_bytes=%zu bytes_per_key=%.2f build_ms=%.1f\n",
                 name_.c_str(), n, B, exc_.size(), cells, nodes_.size(), leaves_, biggest_, double(slots_.size()) / double(n), total_, d_,
                 lines_.mean(), lines_.tail(0.99), lines_.tail(0.999), lines_.worst(), int(fallback_), double(levels_) / double(m_), nodes_.size() * sizeof(Node),
                 double(memory_consumption()) / double(n), ms);
  }

  bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t x = uint64_t(key), v = x - lo_;
    if (__builtin_expect(v > span_, 0)) return find_exception(key, val);
    const Node* const n = nodes_.data();
    Node e = n[v >> shift_];
    unsigned s = shift_;
    while (e.a >> 31) {  // one branch per level: a fixed branchless walk measured about 1.8x slower on easy data
      const unsigned s2 = s - e.b;
      e = n[size_t(e.a & ~kInner) + size_t((v >> s2) & ((uint64_t(1) << e.b) - 1))];
      s = s2;
    }
    size_t b = size_t(e.a) + size_t((u128(hash(x)) * e.b) >> 64);
    const size_t b0 = b;
    for (const Slot* l = &slots_[b * B];; l += B) {
      unsigned hit = 0;
      for (unsigned j = 0; j < B; ++j) hit |= unsigned(l[j].first == x) << j;
      if (__builtin_expect(hit != 0, 1)) {
        val = l[__builtin_ctz(hit)].second;
        return true;
      }
      if (++b > b0 + d_ || b >= total_) return false;
    }
  }

  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  // Leaves in key order; inside a leaf, read all its slots (spill from the leaf before is filtered out), keep the keys >= the
  // start, and sort as many as the scan still needs: the price of giving up order inside the leaves. PH_SCAN_SORTED=0 returns the
  // same keys in no particular order, for a caller that aggregates and has no use for the sort.
  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    const uint64_t x = uint64_t(key), last = lo_ + span_;
    auto ex = std::lower_bound(exc_.begin(), exc_.end(), key, [](const std::pair<K, P>& a, K k) { return a.first < k; });
    size_t got = 0;
    while (got < n && ex != exc_.end() && uint64_t(ex->first) < lo_) out[got++] = *ex++;
    if (got < n && x <= last) {
      thread_local std::vector<Slot> tmp;
      const Node* const nd = nodes_.data();
      for (uint64_t v = x > lo_ ? x - lo_ : 0;;) {
        Node e = nd[v >> shift_];
        unsigned s = shift_;
        while (e.a >> 31) {
          const unsigned s2 = s - e.b;
          e = nd[size_t(e.a & ~kInner) + size_t((v >> s2) & ((uint64_t(1) << e.b) - 1))];
          s = s2;
        }
        const uint64_t cell_end = v | ((uint64_t(1) << s) - 1);
        const size_t from = size_t(e.a) * B, to = std::min(slots_.size(), (size_t(e.a) + std::max<size_t>(e.b, 1) + d_) * B);
        tmp.clear();
        for (size_t i = from; i < to; ++i) {
          const uint64_t k = slots_[i].first;
          if (k - lo_ < v || k - lo_ > cell_end || (i > from && slots_[i - 1].first == k)) continue;  // another leaf's, or a copy
          tmp.push_back(slots_[i]);
        }
        const size_t take = std::min(n - got, tmp.size());
        const auto by_key = [](const Slot& a, const Slot& b) { return a.first < b.first; };
        if (sorted_scan_) std::partial_sort(tmp.begin(), tmp.begin() + std::ptrdiff_t(take), tmp.end(), by_key);
        else if (take < tmp.size()) std::nth_element(tmp.begin(), tmp.begin() + std::ptrdiff_t(take), tmp.end(), by_key);
        for (size_t i = 0; i < take; ++i) out[got++] = std::pair<K, P>(K(tmp[i].first), tmp[i].second);
        if (got >= n || cell_end >= span_) break;
        v = cell_end + 1;
      }
    }
    while (got < n && ex != exc_.end()) out[got++] = *ex++;
    return got;
  }
  long long memory_consumption() override {
    return (long long)(slots_.size() * sizeof(Slot) + nodes_.size() * sizeof(Node) + exc_.size() * sizeof(std::pair<K, P>));
  }
  double expected_lines() const { return lines_.mean(); }  // lines a lookup of a stored key reads, from the build's exact count
  size_t worst_lines() const { return d_ + 1; }
  bool uses_fallback() const { return fallback_; }

 private:
  uint64_t hash(uint64_t x) const { return fallback_ ? mix64(x) : mul64(x); }
  void build_all(size_t cells) {
    nodes_.assign(cells, Node{0, 0});
    levels_ = leaves_ = 0;
    biggest_ = 0;
    slots_.resize(rank_bucket(m_) * B);
    prev_ = SIZE_MAX, w_ = 0;
    lines_ = LineCount();
    size_t i = 0;
    for (size_t c = 0; c < cells; ++c) {
      size_t j = i;
      while (j < m_ && ((key(j) - lo_) >> shift_) == c) ++j;
      make_node(c, i, j, shift_, 0);
      i = j;
    }
  }
  uint64_t key(size_t r) const { return uint64_t(d_keys_[r].first); }
  size_t rank_bucket(size_t r) const {  // the first bucket of the keys from rank r on
    return r < m_ ? size_t(double(r) * per_) : size_t(std::ceil(double(m_) * per_));
  }
  void make_node(size_t at, size_t i, size_t j, unsigned s, unsigned depth) {
    const size_t b0 = rank_bucket(i), b1 = rank_bucket(j), nk = j - i;
    if (b1 >= kInner) throw std::length_error("PartitionHashIndex: 2^31 buckets");
    if (nk > leaf_ && s > 0 && depth < depth_) {
      const unsigned want = unsigned(std::ceil(std::log2(double(nk) / double(leaf_))));
      const unsigned k = std::min({std::max(want, 1u), kmax_, s});
      const size_t first = nodes_.size();
      if (first + (size_t(1) << k) >= kInner) throw std::length_error("PartitionHashIndex: 2^31 nodes");
      nodes_.resize(first + (size_t(1) << k), Node{0, 0});
      nodes_[at] = Node{kInner | uint32_t(first), k};
      const unsigned s2 = s - k;
      const uint64_t mask = (uint64_t(1) << k) - 1;
      size_t r = i;
      for (size_t c = 0; c < (size_t(1) << k); ++c) {
        size_t e = r;
        while (e < j && (((key(e) - lo_) >> s2) & mask) == c) ++e;
        make_node(first + c, r, e, s2, depth + 1);
        r = e;
      }
      return;
    }
    nodes_[at] = Node{uint32_t(b0), uint32_t(b1 - b0)};
    if (!nk) return;
    ++leaves_;
    levels_ += nk * (depth + 1);
    biggest_ = std::max(biggest_, nk);
    tmp_.resize(nk);
    for (size_t r = i; r < j; ++r) tmp_[r - i] = std::pair<uint64_t, size_t>(hash(key(r)), r);
    std::sort(tmp_.begin(), tmp_.end());
    const size_t cnt = b1 - b0;
    for (const auto& t : tmp_) {  // hash order inside the leaf, forward across leaves
      const size_t p = b0 + size_t((u128(t.first) * cnt) >> 64), st = p * B, pos = prev_ == SIZE_MAX || st > prev_ ? st : prev_ + 1;
      lines_.add(pos / B - p + 1);
      if (pos >= slots_.size()) slots_.resize(std::max(slots_.size() * 2, (pos / B + 1) * B));
      const Slot sl(key(t.second), d_keys_[t.second].second);
      while (w_ <= pos) slots_[w_++] = sl;
      prev_ = pos;
    }
  }
  bool find_exception(K key, P& val) const {
    const auto it = std::lower_bound(exc_.begin(), exc_.end(), key,
                                     [](const std::pair<K, P>& a, K k) { return a.first < k; });
    if (it == exc_.end() || it->first != key) return false;
    val = it->second;
    return true;
  }

  std::string name_;
  uint64_t lo_ = 1, span_ = 0;
  unsigned shift_ = 0, depth_ = 3, kmax_ = 8;
  double per_ = 1;
  size_t leaf_ = 4096, m_ = 0, total_ = 1, d_ = 0, prev_ = SIZE_MAX, w_ = 0, levels_ = 0, leaves_ = 0, biggest_ = 0;
  bool sorted_scan_ = true, fallback_ = false;
  LineCount lines_;
  const std::pair<K, P>* d_keys_ = nullptr;  // the bulk, during bulk_load only
  std::vector<std::pair<uint64_t, size_t>> tmp_;
  std::vector<Node> nodes_;
  std::vector<Slot, BucketAlloc<Slot>> slots_;
  std::vector<std::pair<K, P>> exc_;
};

}  // namespace gret
