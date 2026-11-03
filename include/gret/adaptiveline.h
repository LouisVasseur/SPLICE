// SPDX-License-Identifier: MIT
// AdaptiveLineIndex<K, P, B>: RadixLineIndex with T refined only where lookups would read too many lines, so the directory
// follows how hard each region is (an adaptive radix directory, as in PLEX; the split rule here is the line count, not a
// rank error). Read-only, exact for hits and misses, keeps key order.
//
//   G  peel_outliers, as RadixLineIndex.
//   T  a root of equal-width cells (2^RADIX_BITS at most, default 2^16, and at most about n / 8), then at most AR_DEPTH
//      levels of child blocks (default 3, root excluded). A node first skips the key bits all its keys share, so a tight cluster
//      in a wide cell costs one level, not a chain of them. It is then a leaf {first bucket, buckets}, a straight line over
//      what is left of its range, or it splits into 2^k equal-width children. Top-down: a node splits when its keys, placed
//      alone in its buckets, read more than AR_TAU lines per lookup (default 1.5) and it holds more than AR_LEAF keys (default
//      64); k aims at AR_LEAF keys per child, at most AR_KMAX bits (default 8).
//   V  as RadixLineIndex: HASHLINE_SPK slots per key (default 1.25), slots in proportion to the exact key counts. With AB_V=1 each
//      leaf's line is then chosen among shifted and rescaled ones, in key order, by the lines its keys and the next leaf's really
//      read given the backlog so far. No metadata: a leaf's first slot and slope already hold it.
//
// AB_BUDGET=<bytes per key> replaces HASHLINE_SPK: everything counts against it, slots, directory and exceptions. Finer splits
// leave fewer slots, so the directory is rebuilt until the two agree; below one slot per key it throws.
//
// get walks the levels with one branch per level (a fixed branchless walk measured 1.3 to 1.8x slower), then reads as
// RadixLineIndex does. A key that was never stored may differ from a node's skipped bits and so reach any of its children; the raw
// key compare still answers exactly. Keys must be sorted and unique, as GRE's driver guarantees.
// put, update and remove return false. scan reads the sorted slots from the start key on.
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

struct IdentityCoord {  // no G beyond the peeled ends: the directory is on the key itself
  static constexpr bool kIdentity = true;
  template <class Pair> void load(const Pair*, size_t) {}
  uint64_t operator()(uint64_t x) const { return x; }
  size_t bytes() const { return 0; }
};

template <class K, class P, unsigned B, class Coord = IdentityCoord>
class AdaptiveLineIndex final : public indexInterface<K, P> {
  __extension__ typedef unsigned __int128 u128;  // __extension__: no -Wpedantic warning
  using Slot = std::pair<uint64_t, P>;
  struct Node {
    uint32_t a;  // leaf: first slot. Inner (top bit set): first child, in nodes_
    uint32_t b;  // leaf: 64 - bits of its range, << 24 | slots. Inner: 64 - k, << 8 | 64 - bits of its range
  };
  struct LeafAt {  // a leaf with keys, for AB_V's pass
    uint64_t i;
    uint32_t nk, at;
  };
  static constexpr uint32_t kInner = 0x80000000u, kSlots = (uint32_t(1) << 24) - 1;

  static double env_d(const char* k, double def) {
    const char* e = std::getenv(k);
    return e && *e ? std::strtod(e, nullptr) : def;
  }

 public:
  explicit AdaptiveLineIndex(std::string name) : name_(std::move(name)) {}

  void init(Param* = nullptr) override {}

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (n == 0) throw std::invalid_argument("AdaptiveLineIndex: bulk_load of 0 keys");
    double spk = env_d("HASHLINE_SPK", 1.25);
    const double budget = env_d("AB_BUDGET", 0);
    fit_ = env_d("AB_V", 0) != 0;
    skip_ = env_d("AB_NOSKIP", 0) == 0;  // for ablation: the same lookup path, with nodes spanning their whole cell
    const unsigned want = unsigned(env_d("RADIX_BITS", 16));
    tau_ = env_d("AR_TAU", 1.5);
    const double leaf = env_d("AR_LEAF", 64), kmax = env_d("AR_KMAX", 8), depth = env_d("AR_DEPTH", 3);
    if (!(spk > 0) || !(budget >= 0) || !(leaf >= 1) || !(kmax >= 1 && kmax <= 30) || !(depth >= 0 && depth <= 64))
      throw std::invalid_argument("AdaptiveLineIndex: HASHLINE_SPK > 0, AB_BUDGET >= 0, AR_LEAF >= 1, AR_KMAX in 1..30, AR_DEPTH in 0..64");
    leaf_ = size_t(leaf);
    kmax_ = unsigned(kmax);
    depth_ = unsigned(depth);
    size_t lo, hi;
    peel_outliers(kv, n, size_t(env_d("TRANSFORM_G_E", 64)), lo, hi);
    exc_.assign(kv, kv + lo);
    exc_.insert(exc_.end(), kv + hi, kv + n);
    d_keys_ = kv + lo;
    m_ = hi - lo;
    coord_.load(d_keys_, m_);
    if constexpr (!Coord::kIdentity) {
      y_.resize(m_);
      for (size_t r = 0; r < m_; ++r) {
        y_[r] = coord_(uint64_t(d_keys_[r].first));
        if (r && y_[r] <= y_[r - 1]) throw std::runtime_error("AdaptiveLineIndex: the coordinate does not keep stored keys strictly increasing");
      }
    }
    lo_ = key(0);
    span_ = key(m_ - 1) - lo_;
    unsigned sb = 0, nb = 0;  // bits of the span, and of the cells the keys can pay for
    for (uint64_t v = span_; v; v >>= 1) ++sb;
    for (size_t v = m_ / 8; v; v >>= 1) ++nb;
    const unsigned bits = std::max(std::min({want, nb, sb}), std::min(1u, sb));  // a span of 64 bits needs a shift below 64
    shift_ = sb - bits;
    const size_t cells = size_t(span_ >> shift_) + 1;
    unsigned rounds = 0;
    if (budget > 0) {
      const auto slots_left = [&](size_t nodes) {  // slots per bulk key once the rest is paid for
        return (budget * double(n) - double(exc_.size() * sizeof(std::pair<K, P>) + coord_.bytes()) - double(nodes * sizeof(Node))) / double(sizeof(Slot)) / double(m_);
      };
      spk = slots_left(cells);
      for (;;) {
        if (spk < 1) throw std::invalid_argument("AdaptiveLineIndex: AB_BUDGET leaves under one slot per key");
        build_tree(cells, spk);
        ++rounds;
        const double next = slots_left(nodes_.size());
        const bool close = std::abs(next - spk) < 0.001;  // 0.016 bytes per key
        if (close && next >= 1) break;
        if (rounds >= 6 && next >= spk) break;
        spk = next;
      }
    } else {
      build_tree(cells, spk);
    }
    if (fit_) fit_lines();
    std::vector<LeafAt>().swap(leaves_);
    nodes_.shrink_to_fit();  // memory_consumption counts size(), and resize leaves spare capacity
    size_t prev = SIZE_MAX;
    LineCount lc;  // over all bulk keys, each looked up once
    for (size_t r = 0; r < m_; ++r) {
      const size_t p = bucket(key(r) - lo_), pos = place(p, prev);
      lc.add(pos / B - p + 1);
      prev = pos;
    }
    d_ = lc.worst() - 1;
    e_lines_ = lc.mean();
    total_ = std::max(rank_slot(m_) / B, prev / B + 1);
    slots_.resize(total_ * B);
    size_t w = 0;
    prev = SIZE_MAX;
    for (size_t r = 0; r < m_; ++r) {
      const size_t pos = place(bucket(key(r) - lo_), prev);
      const Slot s(uint64_t(d_keys_[r].first), d_keys_[r].second);
      while (w <= pos) slots_[w++] = s;
      prev = pos;
    }
    const Slot last(uint64_t(d_keys_[m_ - 1].first), d_keys_[m_ - 1].second);
    while (w < slots_.size()) slots_[w++] = last;
    d_keys_ = nullptr;
    std::vector<uint64_t>().swap(y_);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu B=%u exceptions=%zu cells=%zu nodes=%zu slots/key=%.3f buckets=%zu d=%zu E[buckets]=%.4f lines_p99=%zu lines_p999=%zu lines_max=%zu E[levels]=%.3f dir_bytes=%zu fit=%d budget_rounds=%u rule_bytes=%zu bytes_per_key=%.2f build_ms=%.1f\n",
                 name_.c_str(), n, B, exc_.size(), cells, nodes_.size(), double(slots_.size()) / double(n), total_, d_,
                 lc.mean(), lc.tail(0.99), lc.tail(0.999), lc.worst(), double(levels_) / double(m_), nodes_.size() * sizeof(Node), int(fit_), rounds, coord_.bytes(),
                 double(memory_consumption()) / double(n), ms);
  }

  bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t x = uint64_t(key), v = coord_(x) - lo_;
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
    return (long long)(slots_.size() * sizeof(Slot) + nodes_.size() * sizeof(Node) + exc_.size() * sizeof(std::pair<K, P>) + coord_.bytes());
  }
  double expected_lines() const { return e_lines_; }  // lines a lookup of a stored key reads, from the build's exact count
  size_t worst_lines() const { return d_ + 1; }
  double expected_levels() const { return double(levels_) / double(m_); }
  size_t directory_bytes() const { return nodes_.size() * sizeof(Node); }

 private:
  uint64_t key(size_t r) const {  // what the directory orders: the key's coordinate, during bulk_load
    if constexpr (Coord::kIdentity) return uint64_t(d_keys_[r].first);
    else return y_[r];
  }
  size_t rank_slot(size_t r) const {  // the first slot of the keys from rank r on
    return r < m_ ? size_t(double(r) * per_ * double(B)) : size_t(std::ceil(double(m_) * per_)) * B;
  }
  void build_tree(size_t cells, double spk) {
    per_ = spk / B;
    if (rank_slot(m_) >= kInner) throw std::length_error("AdaptiveLineIndex: 2^31 slots");
    nodes_.assign(cells, Node{0, 0});
    leaves_.clear();
    levels_ = 0;
    size_t i = 0;
    for (size_t c = 0; c < cells; ++c) {
      size_t j = i;
      while (j < m_ && ((key(j) - lo_) >> shift_) == c) ++j;
      make_node(c, i, j, shift_, 0);
      i = j;
    }
  }
  // Shifts rather than masks: every instruction on the path counts, since fewer of them let more lookups' misses overlap. A leaf's
  // position in its range: v's low bits moved to the top are a 64-bit fraction, and with c slots the slot is its high word times c.
  static size_t slot_in(uint64_t v, unsigned to_top, size_t c) { return size_t((u128(v << to_top) * c) >> 64); }
  // the lines keys [i, i + nk) read on the line {first slot a, slope c} over a range of 2^(64 - to_top) values, after the slots up to prev
  size_t lines_on(size_t i, size_t nk, unsigned to_top, size_t a, size_t c, size_t prev, size_t& last, size_t& worst) const {
    size_t read = 0;
    worst = 0;
    for (size_t r = i; r < i + nk; ++r) {
      const size_t p = (a + slot_in(key(r) - lo_, to_top, c)) / B, pos = place(p, prev), lines = pos / B - p + 1;
      read += lines;
      worst = std::max(worst, lines);
      prev = pos;
    }
    last = prev;
    return read;
  }
  void make_node(size_t at, size_t i, size_t j, unsigned s, unsigned depth) {
    size_t a = rank_slot(i);
    const size_t a1 = rank_slot(j), nk = j - i;
    if (nk) {  // only the bits that differ between its keys are left to tell them apart
      const uint64_t differ = (key(i) - lo_) ^ (key(j - 1) - lo_);
      if (skip_) s = differ ? std::min(s, 64u - unsigned(__builtin_clzll(differ))) : 0u;
    } else {
      a = std::min(a, rank_slot(m_) - 1);  // past the last key: a never-stored key can get here and must read inside the slots
    }
    s = std::max(s, 1u);  // a shift by 64 is undefined, and one bit is harmless to a node that has no two keys to tell apart
    const size_t c = a1 > a ? a1 - a : 0;
    size_t unused, unused2;
    if (nk > 1 && (c > kSlots || (nk > leaf_ && depth < depth_ && double(lines_on(i, nk, 64 - s, a, c, SIZE_MAX, unused, unused2)) > tau_ * double(nk)))) {
      const double ratio = double(nk) / double(leaf_);
      const unsigned want = ratio > 1 ? unsigned(std::ceil(std::log2(ratio))) : 1u;
      const unsigned k = std::min({std::max(want, 1u), kmax_, s});
      const size_t first = nodes_.size();
      if (first + (size_t(1) << k) >= kInner) throw std::length_error("AdaptiveLineIndex: 2^31 nodes");
      nodes_.resize(first + (size_t(1) << k), Node{0, 0});
      const unsigned s2 = s - k;
      nodes_[at] = Node{kInner | uint32_t(first), (64 - k) << 8 | (64 - s)};
      const uint64_t mask = (uint64_t(1) << k) - 1;
      size_t r = i;
      for (size_t ch = 0; ch < (size_t(1) << k); ++ch) {
        size_t e = r;
        while (e < j && (((key(e) - lo_) >> s2) & mask) == ch) ++e;
        make_node(first + ch, r, e, s2, depth + 1);
        r = e;
      }
      return;
    }
    nodes_[at] = Node{uint32_t(a), (64 - s) << 24 | uint32_t(c)};
    levels_ += nk * (depth + 1);
    if (fit_ && nk) leaves_.push_back(LeafAt{uint64_t(i), uint32_t(nk), uint32_t(at)});
  }
  // AB_V's search, leaf by leaf in key order, behind the real backlog. A line replaces the leaf's rank line only if its keys read fewer
  // lines (counting the next leaf's behind whatever backlog it leaves), none of them reads more than the rank line's worst, and it ends
  // no later than the rank line would or the leaf's own last slot. Positions only move forward when the slots before fill later, so
  // what follows cannot read more because of it: the mean, the worst key and the bytes can only stay or improve.
  void fit_lines() {
    const size_t total = rank_slot(m_);
    size_t prev = SIZE_MAX;
    for (size_t l = 0; l < leaves_.size(); ++l) {
      const LeafAt& f = leaves_[l];
      Node& e = nodes_[f.at];
      const unsigned to_top = e.b >> 24;
      const size_t a0 = e.a, c0 = e.b & kSlots, nk = f.nk;
      const LeafAt* const nx = l + 1 < leaves_.size() ? &leaves_[l + 1] : nullptr;
      size_t best_a = a0, best_c = c0, best_last, rank_worst, unused, unused2;
      auto next_lines = [&](size_t last) {
        if (!nx) return size_t(0);
        const Node& ne = nodes_[nx->at];
        return lines_on(nx->i, nx->nk, ne.b >> 24, ne.a, ne.b & kSlots, last, unused, unused2);
      };
      size_t best = lines_on(f.i, nk, to_top, a0, c0, prev, best_last, rank_worst);
      best += next_lines(best_last);
      const size_t may_end = std::max(best_last, a0 + c0 - 1), least = nk + (nx ? nx->nk : 0);
      if (best > least) {
        const double top = std::ldexp(1.0, 64);
        const double f0 = double((key(f.i) - lo_) << to_top) / top, f1 = double((key(f.i + nk - 1) - lo_) << to_top) / top;
        const double fit = f1 > f0 ? double(c0) * (1 - 1 / double(nk)) / (f1 - f0) : double(c0);  // first key to last, in the leaf's own slots
        const double slopes[2] = {double(c0), fit}, scales[3] = {1, 0.9, 1.1};
        for (int line = 0; line < 2 && best > least; ++line)
          for (double scale : scales)
            for (size_t shift = 0; shift < B && best > least; ++shift) {
              if (line == 0 && scale == 1 && shift == 0) continue;
              const double c = slopes[line] * scale, a = double(a0 + shift) - (line ? f0 * c : 0);
              if (a < 0 || c < 1 || c > double(kSlots) || a + c > double(total)) continue;
              size_t last, worst;
              size_t lines = lines_on(f.i, nk, to_top, size_t(a), size_t(c), prev, last, worst);
              if (last > may_end || worst > rank_worst) continue;
              lines += next_lines(last);
              if (lines < best) best = lines, best_a = size_t(a), best_c = size_t(c), best_last = last;
            }
      }
      e = Node{uint32_t(best_a), to_top << 24 | uint32_t(best_c)};
      prev = best_last;
    }
  }
  size_t bucket(uint64_t v) const {
    const Node* const n = nodes_.data();
    Node e = n[v >> shift_];
    while (e.a >> 31) e = n[size_t(e.a & ~kInner) + size_t((v << (e.b & 255)) >> (e.b >> 8))];
    return (size_t(e.a) + slot_in(v, e.b >> 24, e.b & kSlots)) / B;
  }
  // the first slot whose key is >= x, for x within the bulk: the slots are sorted, so gallop from x's prediction in the direction its
  // key says. A stored key's prediction is a few slots behind it, but a key that was never stored may be predicted anywhere: it can
  // differ from a node's skipped bits, and AB_V's lines need not meet.
  size_t lower_slot(uint64_t x) const {
    const size_t n = slots_.size(), i0 = bucket(std::min(coord_(x) - lo_, span_)) * B;
    size_t lo = i0, hi = i0, step = 1;
    if (slots_[i0].first < x) {
      lo = hi = i0 + 1;
      for (; hi < n && slots_[hi].first < x; hi += step, step *= 2) lo = hi + 1;
      hi = std::min(hi, n);
    } else {
      for (; lo > 0 && slots_[lo - 1].first >= x; lo = lo > step ? lo - step : 0, step *= 2) hi = lo - 1;
    }
    return size_t(std::lower_bound(slots_.begin() + std::ptrdiff_t(lo), slots_.begin() + std::ptrdiff_t(hi), x, [](const Slot& a, uint64_t k) { return a.first < k; }) - slots_.begin());
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
  uint64_t lo_ = 1, span_ = 0;
  unsigned shift_ = 0, depth_ = 3, kmax_ = 8;
  double tau_ = 1.5, per_ = 1, e_lines_ = 0;
  size_t leaf_ = 64, m_ = 0, total_ = 1, d_ = 0, levels_ = 0;
  bool fit_ = false, skip_ = true;
  std::vector<LeafAt> leaves_;  // during bulk_load only
  std::vector<uint64_t> y_;     // during bulk_load only, and only with a Coord
  Coord coord_;
  const std::pair<K, P>* d_keys_ = nullptr;  // the bulk, during bulk_load only
  std::vector<Node> nodes_;
  std::vector<Slot, BucketAlloc<Slot>> slots_;
  std::vector<std::pair<K, P>> exc_;
};

}  // namespace gret
