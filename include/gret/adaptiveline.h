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
// AB_SLACK_Z=<z> (default 0: by key, as above) shares the slots beyond one a key by leaf instead, k + z sqrt(k) for k keys, for
// layouts that will take writes (tools/corrupt/sim.cpp measures what that buys).
//
// AB_BUDGET=<bytes per key> replaces HASHLINE_SPK: everything counts against it, slots, directory and exceptions. Finer splits
// leave fewer slots, so the directory is rebuilt until the two agree; below one slot per key it throws.
//
// get walks the levels with one branch per level (a fixed branchless walk measured 1.3 to 1.8x slower), then reads as
// RadixLineIndex does. A key that was never stored may differ from a node's skipped bits and so reach any of its children; the raw
// key compare still answers exactly. Keys must be sorted and unique, as GRE's driver guarantees.
// update and remove work in place on stored keys (see update); put returns false: no inserts. scan reads the sorted slots from the
// start key on.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gret/exceptions.h"
#include "gret/hashline.h"
#include "gret/overflow.h"

namespace gret {

struct IdentityCoord {  // no G beyond the peeled ends: the directory is on the key itself
  static constexpr bool kIdentity = true;
  template <class Pair> void load(const Pair*, size_t) {}
  uint64_t operator()(uint64_t x) const { return x; }
  size_t bytes() const { return 0; }
};

template <class K, class P, unsigned B, class Coord> struct AdaptiveLineProbe;  // tools/corrupt/sim.cpp: reads and edits the layout

// W (default false): the writable variant (aw4). put takes a free slot within the key's reach, or moves at most kShift keys by one
// slot, and otherwise keeps the key in an OverflowTable, which get then asks only when the slots do not have the key. With W false
// none of that is compiled in, and get is the code it was.
template <class K, class P, unsigned B, class Coord = IdentityCoord, bool W = false>
class AdaptiveLineIndex final : public indexInterface<K, P> {
  friend struct AdaptiveLineProbe<K, P, B, Coord>;
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
  struct SlackAt {  // AB_SLACK_Z: the keys from rank i on weigh w each, after a total weight of cum
    uint64_t i;
    double cum, w;
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
    const double slack_z = env_d("AB_SLACK_Z", 0);
    if (!(slack_z >= 0)) throw std::invalid_argument("AdaptiveLineIndex: AB_SLACK_Z >= 0");
    collect_ = fit_ || slack_z > 0;
    slack_.clear();
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
    rs_base_ = 0, rs_cnt_ = m_;
    raw_lo_ = uint64_t(d_keys_[0].first);
    raw_hi_ = uint64_t(d_keys_[m_ - 1].first);
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
    if (slack_z > 0) {
      // Spare slots by leaf rather than by key: every key keeps one slot, and the slots beyond that are shared by weight, a leaf
      // of k keys weighing k + z sqrt(k), as the keys later written into it vary by about sqrt(k). So small leaves, which fill
      // first, get more room and large ones give some up. The directory is then built again by the same rule on these shares.
      // The total is unchanged.
      double cum = 0;
      slack_.reserve(leaves_.size());
      for (const LeafAt& f : leaves_) {  // in key order, and every bulk key is in one
        const double w = 1 + slack_z / std::sqrt(double(f.nk));
        slack_.push_back(SlackAt{f.i, cum, w});
        cum += w * double(f.nk);
      }
      slack_total_ = cum;
      build_tree(cells, spk);
    }
    if (fit_) fit_lines();
    std::vector<SlackAt>().swap(slack_);
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
    total_ = std::max(rank_slot(rs_cnt_) / B, prev / B + 1);
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
    live_ = m_;
    last_key_ = last.first;
    removed_ = false;
    // AB_DEL: what remove fetches along with the lookup. 0 = the slot before the bucket and the line after it (both neighbours
    // a removal may need), 1 = the slot before only, 2 = nothing: a neighbour's line is then read only when the key's run of
    // copies reaches its line's edge. (A removal that stays inside the key's own line is not possible: once a key was removed a
    // run can go on into the next line, so that line must be read whenever the run reaches the end of its line; tried, refuted
    // by tests/test_write.) AB_UPD_DRY=1: update finds the key and stores nothing (a measurement of what the store costs).
    del_mode_ = int(env_d("AB_DEL", 0)), upd_dry_ = env_d("AB_UPD_DRY", 0) != 0;
    if constexpr (W) {
      back_ = env_d("AW_BACK", 0) != 0;
      const char* pw = std::getenv("AW_PUT");
      put_wide_ = pw && std::string(pw) == "wide";
      put_next_ = pw && std::string(pw) == "line2";
      ov_bits_.assign(total_ / 64 + 1, 0), ov_out_ = 0;
      ov_ = OverflowTable<P>();
      {  // AW_OVTABLE=sparse (default) | dense, AW_OVBIT=0 turns the per-line overflow bit off: for measuring what each is worth
        const char* ot = std::getenv("AW_OVTABLE");
        ov_dense_ = ot && std::string(ot) == "dense";
        ov_.set_dense(ov_dense_);
        ov_bit_on_ = env_d("AW_OVBIT", 1) != 0;
      }
      // AW_INPLACE=1: a cell is laid out again inside its own slots when its keys and the overflow's fit there (no new slots).
      // AW_ROT=1 | 3: at the index's break-even the overflow is merged cell by cell, a bounded share on each put, instead of one
      // rebuild of everything; 3 = three tables in rotation (the full one is frozen and read second, an empty one takes the puts),
      // 1 = the same merge on the one table. AW_DENSE: the fewest slots a key an in-place layout may have.
      frz_ = OverflowTable<P>(), frz_live_ = false;
      inplace_ = env_d("AW_INPLACE", 0) != 0, rot_ = unsigned(env_d("AW_ROT", 0)), dense_min_ = env_d("AW_DENSE", 1.0);
      drain_cells_.clear(), drain_pos_ = 0, drain_last_t_ = t_put_, drain_keys_ = 0, drain_back_ = 0, drain_credit_ = 0, drain_rate_ = 0;
      draining_ = false, drain_fail_ = false, rs_all_ = 0;
      ov_sorted_.clear(), ov_dirty_ = false;
      regions_.clear(), cell_ov_.clear(), cell_t_.clear(), cell_skip_.clear();
      main_n_ = 0, debt_ = 0;
      // rebuilds (AW_RW) lay cells out by the raw key: with G's coordinate they stay off, and keys the slots cannot take stay in
      // the overflow until a rebuild of everything
      rw_ = Coord::kIdentity ? env_d("AW_RW", 0) : 0, ov_ns_ = env_d("AW_OV_NS", 60), hot_ = env_d("AW_HOT", 3);
      if (rw_ > 0) {  // by root cell, dense: one small entry a cell, no hashing on the way of a put
        const size_t cells_n = size_t(span_ >> shift_) + 1;
        cell_ov_.assign(cells_n, std::vector<uint64_t>()), cell_t_.assign(cells_n, 0.0), cell_skip_.assign(cells_n, 0);
      }
      build_ns_ = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(n);
      // AW_ARENA: how far the slots may grow for rebuilt cells, as a share of the slots: the bound on what cell rebuilds add to
      // the footprint. When it is used up cells are refused, until the whole index is rebuilt by its own break-even.
      arena_left_ = rw_ > 0 ? size_t(env_d("AW_ARENA", 0.1) * double(slots_.size())) : 0;
      // rebuilt cells go to slots after the main ones: reserved now, so the array never moves (untouched pages cost nothing,
      // and memory_consumption counts the slots in use)
      if (arena_left_) slots_.reserve(slots_.size() + arena_left_ + (d_ + 2) * B);
    }
    std::vector<uint64_t>().swap(y_);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu B=%u exceptions=%zu cells=%zu nodes=%zu slots/key=%.3f buckets=%zu d=%zu E[buckets]=%.4f lines_p99=%zu lines_p999=%zu lines_max=%zu E[levels]=%.3f dir_bytes=%zu fit=%d budget_rounds=%u rule_bytes=%zu bytes_per_key=%.2f build_ms=%.1f\n",
                 name_.c_str(), n, B, exc_.size(), cells, nodes_.size(), double(slots_.size()) / double(n), total_, d_,
                 lc.mean(), lc.tail(0.99), lc.tail(0.999), lc.worst(), double(levels_) / double(m_), nodes_.size() * sizeof(Node), int(fit_), rounds, coord_.bytes(),
                 double(memory_consumption_ro()) / double(n), ms);
  }

  bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    if constexpr (!Coord::kIdentity) {  // a far key's coordinate can wrap into the span, so the raw key decides whether it is in the bulk
      if (__builtin_expect(x < raw_lo_ || x > raw_hi_, 0)) {
        if constexpr (W) return find_exception(key, val) || ov_find(x, val);  // (W) a put outside it went to the overflow
        else return find_exception(key, val);
      }
    }
    const uint64_t v = coord_(x) - lo_;
    if (__builtin_expect(v > span_, 0)) {
      if constexpr (W) return find_exception(key, val) || ov_find(x, val);
      else return find_exception(key, val);
    }
    size_t b = bucket(v);
    const size_t b0 = b;
    for (const Slot* l = &slots_[b * B];; l += B) {
      size_t c = 0;
      for (unsigned j = 0; j < B; ++j) c += size_t(l[j].first < x);
      if (__builtin_expect(c < B, 1)) {  // the first key >= x decides
        if (l[c].first != x) {
          if constexpr (W) return ov_line(b0) && ov_find(x, val);  // the overflow is asked only where a key of this line went to it
          else return false;
        }
        val = l[c].second;
        return true;
      }
      if (++b > b0 + d_ || b >= total_) {
        if constexpr (W) return ov_line(b0) && ov_find(x, val);
        else return false;
      }
    }
  }

  // put (W only): the key must be new. In order of preference: a free slot within its reach (the copies before the next key's own
  // slot, at or after its bucket's start); the nearest free slot within kShift slots to the right, the keys between moving up by
  // one; the same to the left when the key's place is past its bucket's start; else the overflow table. A move is made only if
  // every key it touches stays within d_ lines of its bucket, so get needs no new bound. tools/corrupt/sim.cpp measured this
  // policy: longer shifts absorb more keys but cost more lines than the overflow lookups they save.
  // The rule put follows, from the loss. Over every later lookup, taking keys as equally likely, an insert costs
  //   in the overflow:   c_ov lines, for the new key alone (the overflow's own lookup, about 1.0 to 1.5)
  //   by a shift:        one line for each key the shift carries across a cache-line boundary, from then on
  // A shift that stays inside one line carries no key across: it adds nothing to any lookup, needs no directory walk to check a
  // moved key (a key that stays in its line stays where get looks), and writes a line get has just read. So AW_PUT=line (the
  // default) shifts only within the line holding the key's place, takes the line's last slot when it is a free copy, and sends
  // everything else to the overflow. AW_PUT=wide is the earlier rule (up to kShift slots across lines, each moved key checked).
  bool put(K key, P val, Param* = nullptr) override {
    if constexpr (!W) {
      return false;
    } else {
      const uint64_t x = uint64_t(key);
      if constexpr (!Coord::kIdentity) {
        // G's coordinate (gw4): defined only over the bulk's raw range, so a key outside it, or any key once the bulk is empty,
        // goes to the overflow, which get asks for such keys. Inside it the coordinate is monotone, so a new key's bucket is at
        // or before its successor's and the put below is the plain one, every stored key's bucket taken through the coordinate.
        if (__builtin_expect(x < raw_lo_ || x > raw_hi_ || live_ == 0, 0)) {
          if (exception_at(key) != exc_.end()) return false;
          return ++puts_ov_, ov_insert(x, val, SIZE_MAX);
        }
      }
      const uint64_t v = coord_(x) - lo_;
      if (__builtin_expect(rw_ > 0, 0)) {
        // Rent or buy, for the whole index. Each write stands for rw_ lookups, and a lookup falls on an overflow key with the
        // overflow's share of the keys and then costs ov_ns_ more. Once that has added up to what a rebuild costs (the build's
        // measured time per key, times the keys), rebuild: at most twice what the best choice in hindsight would have paid.
        ++t_put_;
        if (__builtin_expect(draining_, 0)) drain_step();
        const size_t ovn = ov_n();
        const double all = double(live_ + ovn);
        debt_ += rw_ * ov_ns_ * double(ovn) / all;
        if (debt_ >= build_ns_ * all && ovn) {
          // AW_ROT: merge cell by cell from here on, unless the last merge could not place most of its keys (the slots are
          // full: only new slots help) or most of the overflow lies outside the directory's span (only a new span helps)
          if (rot_ && !drain_fail_ && live_ && ov_out_ * 2 <= ovn) {
            if (!draining_) start_drain();
          } else {
            rebuild_all(false, 0);
            return put(key, val);
          }
        }
      }
      if (__builtin_expect(v > span_ || live_ == 0, 0)) {  // outside the directory's span: G takes it
        if (exception_at(key) != exc_.end()) return false;
        return ++puts_ov_, ov_insert(x, val, v > span_ ? SIZE_MAX : bucket(v));  // inside the span only when no bulk key is left
      }
      if (__builtin_expect(put_wide_, 0)) return put_general(key, val);
      size_t b = bucket(v);
      const size_t b0 = b, n = slots_.size();
      Slot* l = &slots_[b * B];
      size_t c;
      for (;; l += B) {  // get's walk: the first slot at or after the bucket's start whose key is >= x
        c = 0;
        for (unsigned j = 0; j < B; ++j) c += size_t(l[j].first < x);
        if (__builtin_expect(c < B, 1)) break;
        if (++b > b0 + d_ || b >= total_) return ++puts_ov_, ov_insert(x, val, b0);  // no slot within reach
      }
      if (l[c].first == x) return false;
      if (__builtin_expect(ov_line(b0), 0) && ov_has(x)) return false;  // it can be in the overflow only if this line sent a key there
      // x's place in order is here only if the slot before holds a smaller key. In a line reached by walking it does (the line
      // before was all smaller); at the bucket's own first slot it has to be read, and if it is not smaller (copies of the next
      // key reach back before the bucket, or the prediction is out of order) the general routine places or refuses the key.
      if (__builtin_expect(c == 0 && b == b0 && b > 0 && l[-1].first >= x, 0)) {
        // AW_BACK: the next key's free copies reach back before the bucket, and the bucket's first slot is one of them (the next
        // key has another copy after it). x takes that slot and the copies before it become x's: a free slot is a copy of the
        // next pair, and that is x now. The next key keeps its own slot; its first copy moves to this line, which must be within
        // d_ lines of its bucket. The general routine does the same after a search for the run's start; here the run is walked
        // back from the bucket, at most two lines.
        if (back_ && !main_n_ && l[0].first == l[1].first && l[-1].first == l[0].first) {
          const uint64_t k = l[0].first;
          const size_t j = b * B, lim = j > 2 * B ? j - 2 * B : 0;
          size_t st = j;
          while (st > lim && slots_[st - 1].first == k) --st;
          // not a longer run, no key between x and the next key before the bucket, and the next key, whose first copy is now
          // in this line, still within its reach (a key that was never stored can be predicted far from its place in order)
          if ((st == 0 || slots_[st - 1].first < x) && b <= key_bucket(k) + d_) {
            const Slot me(x, val);
            for (size_t t = st; t <= j; ++t) slots_[t] = me;
            removed_ = true;
            ++puts_back_;
            return ++live_, true;
          }
        }
        return put_general(key, val);
      }
      size_t e = c;  // the first free slot of the line at or after c: a copy that is not the last of its run
      while (e + 1 < B && l[e].first != l[e + 1].first) ++e;
      if (e + 1 == B) {  // none inside the line: its last slot is free if the next line starts with the same key
        const size_t nx = b * B + B;
        if (nx >= n) return ++puts_ov_, ov_insert(x, val, b0);
        if (l[B].first != l[B - 1].first) {
          // The line is full. AW_PUT=line2: take a free slot of the NEXT line instead. One key, this line's last, is carried
          // across the boundary and reads one more line from then on, against c_ov (about 1.0 to 1.5) for x in the overflow: about
          // even in lines, and it keeps the key in a 16-byte slot rather than the overflow's 21 to 32 bytes, and off the path of
          // later misses. Only that one key needs the directory (is the next line within its reach); the next line's own keys stay
          // in their line. Not once cells were rebuilt: their guard lines must not be written into.
          if (!put_next_ || main_n_ || nx + B > n) return ++puts_ov_, ov_insert(x, val, b0);
          size_t e2 = B;
          while (e2 + 1 < 2 * B && l[e2].first != l[e2 + 1].first) ++e2;
          if (e2 + 1 == 2 * B) return ++puts_ov_, ov_insert(x, val, b0);  // no free slot inside the next line either
          if (b + 1 > key_bucket(l[B - 1].first) + d_) return ++puts_ov_, ov_insert(x, val, b0);
          removed_ = true;
          for (size_t j = e2; j > c; --j) l[j] = l[j - 1];
          l[c] = Slot(x, val);
          ++puts_next_;
          return ++live_, true;
        }
        // that key then loses a copy it may have been found by: allowed if it is still within reach from the next line
        const size_t bk = key_bucket(l[B - 1].first);
        if (nx / B > bk + d_) return ++puts_ov_, ov_insert(x, val, b0);
      }
      removed_ = true;  // a key may now have copies after the one get reads
      for (size_t j = e; j > c; --j) l[j] = l[j - 1];
      l[c] = Slot(x, val);
      ++puts_line_;
      return ++live_, true;
    }
  }
  // put, the general way: any free slot within reach, or a shift of at most kShift slots across lines, each moved key checked
  // against the directory. Used for the keys whose place lies before their bucket, and for everything with AW_PUT=wide.
  bool put_general(K key, P val) {
    if constexpr (!W) {
      return (void)key, (void)val, false;
    } else {
      const uint64_t x = uint64_t(key);
      const uint64_t v = coord_(x) - lo_;
      ++puts_general_;
      if (main_n_) return ++puts_ov_, ov_insert(x, val, bucket(v));  // with rebuilt cells the slots are no longer one sorted run
      const size_t n = slots_.size(), b = bucket(v), p = lower_slot(x);
      if (p < n && slots_[p].first == x) return false;
      if (__builtin_expect(ov_n() != 0, 0) && ov_has(x)) return false;
      removed_ = true;  // from now on a key may have copies after the one get reads
      const Slot me(x, val);
      if (__builtin_expect(p == n, 0)) {  // above every key in the slots: the copies after the last key's own slot are free
        const uint64_t lastk = slots_[n - 1].first;
        size_t f = n - 1;
        while (f > 0 && slots_[f - 1].first == lastk) --f;
        const size_t q = std::max(std::max(f, key_bucket(lastk) * B) + 1, b * B);
        if (q >= n || q / B > b + d_) return ++puts_ov_, ov_insert(x, val, b);
        std::fill(slots_.begin() + std::ptrdiff_t(q), slots_.end(), me);
        last_key_ = x;
        return ++live_, true;
      }
      const uint64_t sk = slots_[p].first;  // the next key, whose copies fill [p, r]
      size_t r = p;
      while (r + 1 < n && slots_[r + 1].first == sk) ++r;
      const size_t q = std::max(p, b * B);  // the first slot x may take
      if (q > r || q / B > b + d_) return ++puts_ov_, ov_insert(x, val, b);
      if (q < r) {
        if (!reachable(sk, q + 1, r)) return ++puts_ov_, ov_insert(x, val, b);
        std::fill(slots_.begin() + std::ptrdiff_t(p), slots_.begin() + std::ptrdiff_t(q + 1), me);
        return ++live_, true;
      }
      // q == r: the slot x needs holds the next key itself
      size_t er = SIZE_MAX, el = SIZE_MAX;
      for (size_t i = r + 1; i + 1 < n && i <= r + kShift; ++i)
        if (slots_[i].first == slots_[i + 1].first) { er = i; break; }
      if (er != SIZE_MAX) {  // every key in [r, er) moves one slot right, and the key at er loses that copy
        for (size_t i = r; i <= er && er != SIZE_MAX;) {
          const uint64_t k = slots_[i].first;
          size_t j = i;
          while (j + 1 < n && slots_[j + 1].first == k) ++j;
          if (!(i < er ? reachable(k, i + 1, j + 1) : reachable(k, er + 1, j))) er = SIZE_MAX;
          i = j + 1;
        }
      }
      if (er == SIZE_MAX && p == r && r >= 1 && r - 1 >= b * B) {
        for (size_t i = r - 1; i-- > 0 && r - 1 - i <= kShift;)
          if (slots_[i].first == slots_[i + 1].first) { el = i; break; }
        if (el != SIZE_MAX) {  // every key in (el, r - 1] moves one slot left
          for (size_t i = el + 1; i <= r - 1 && el != SIZE_MAX;) {
            const uint64_t k = slots_[i].first;
            size_t j = i, f = i;
            while (j + 1 <= r - 1 && slots_[j + 1].first == k) ++j;
            while (f > 0 && slots_[f - 1].first == k) --f;
            if (!reachable(k, f == i ? i - 1 : f, j - 1)) el = SIZE_MAX;
            i = j + 1;
          }
        }
      }
      if (er != SIZE_MAX) {
        std::copy_backward(slots_.begin() + std::ptrdiff_t(r), slots_.begin() + std::ptrdiff_t(er), slots_.begin() + std::ptrdiff_t(er + 1));
        std::fill(slots_.begin() + std::ptrdiff_t(p), slots_.begin() + std::ptrdiff_t(r + 1), me);  // the next key's copies before r are x's now
        return ++live_, true;
      }
      if (el != SIZE_MAX) {
        std::copy(slots_.begin() + std::ptrdiff_t(el + 1), slots_.begin() + std::ptrdiff_t(r), slots_.begin() + std::ptrdiff_t(el));
        slots_[r - 1] = me;
        return ++live_, true;
      }
      return ++puts_ov_, ov_insert(x, val, b);
    }
  }
  size_t overflow_keys() const { return ov_n(); }
  // how the puts so far were taken: within the line, by the general routine (some of which then overflowed), by the overflow
  void put_counts(size_t& line, size_t& general, size_t& overflow) const { line = puts_line_ + puts_next_, general = puts_general_, overflow = puts_ov_; }
  // Writes on stored keys, in place; get is not touched. An empty slot holds a copy of the next pair, so a key's pairs are one run
  // of equal slots. As built, the run ends at the slot get reads (the key's own; the copies are in the lines before its bucket), so
  // only the last key, whose copies also fill the slots after it, has copies behind that slot. Removing a key turns its run into
  // copies of the next pair (of the previous pair when the key is the last), which is how the build writes empty slots, and from
  // then on a run may go on past the slot get reads.
  //   update  writes the slot get reads, and the copies after it only once a key was removed or for the last key. The copies
  //           before it keep the old payload: no lookup reads them, and scan takes a key's payload from its last copy. So an
  //           update is get's lines and one store, with no second line to wait for.
  //   remove  needs the copies on both sides: those before the slot exist only when it is the first of its line, and that
  //           neighbour's address is known before the line arrives, so both neighbours are fetched along with it.
  // No other key moves, so the directory and d_ stay valid, d_ as an upper bound once keys are gone.
  bool update(K key, P val, Param* = nullptr) override {
    // get's own loop, so that an update costs a lookup and one store to the line the lookup has just read
    const uint64_t x = uint64_t(key);
    if constexpr (!Coord::kIdentity) {
      if (__builtin_expect(x < raw_lo_ || x > raw_hi_, 0)) return update_outside(key, val);
    }
    const uint64_t v = coord_(x) - lo_;
    if (__builtin_expect(v > span_, 0)) return update_outside(key, val);
    size_t b = bucket(v);
    const size_t b0 = b;
    for (Slot* l = &slots_[b * B];; l += B) {
      size_t c = 0;
      for (unsigned j = 0; j < B; ++j) c += size_t(l[j].first < x);
      if (__builtin_expect(c < B, 1)) {
        if (l[c].first != x) return update_absent(x, val);
        if (__builtin_expect(upd_dry_, 0)) return true;
        l[c].second = val;
        if (__builtin_expect(removed_ | (x == last_key_), 0))  // only then can copies follow the slot get reads
          for (size_t j = b * B + c + 1, n = slots_.size(); j < n && slots_[j].first == x; ++j) slots_[j].second = val;
        return true;
      }
      if (++b > b0 + d_ || b >= total_) return update_absent(x, val);
    }
  }
  bool remove(K key, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    size_t i = 0;
    const At at = find_slot(x, i, del_mode_ == 0 ? 3 : del_mode_ == 1 ? 1 : 0);
    if (__builtin_expect(at != At::kSlot, 0)) {
      if (at == At::kNone) {
        if constexpr (W) return ov_dirty_ = true, ov_erase(x);
        else return false;
      }
      const auto it = exception_at(key);
      if (it == exc_.end()) {
        if constexpr (W) return ov_dirty_ = true, ov_erase(x);
        else return false;
      }
      exc_.erase(it);
      return true;
    }
    size_t f = i, e = i + 1, lo_b, n;
    home(x, lo_b, n);  // the slots x lives among: all of them, or (W) the main ones, or its rebuilt cell's
    if (i % B == 0)  // the slots before i in its line hold smaller keys, so copies before i exist only when i starts a line
      while (f > lo_b && slots_[f - 1].first == x) --f;
    while (e < n && slots_[e].first == x) ++e;
    --live_;
    removed_ = true;
    if (e < n) {
      std::fill(slots_.begin() + std::ptrdiff_t(f), slots_.begin() + std::ptrdiff_t(e), slots_[e]);
    } else if (f > lo_b) {
      std::fill(slots_.begin() + std::ptrdiff_t(f), slots_.begin() + std::ptrdiff_t(n), slots_[f - 1]);
      last_key_ = slots_[f - 1].first;
    } else if (lo_b != 0 || n != slots_.size()) {
      // (W) the last key of the main slots or of a rebuilt cell: no pair is left there to copy, so everything is laid out again
      rebuild_all(true, x);
    } else {
      // the last bulk key: from here only x passes get's range check, and no slot answers it
      for (Slot& s : slots_) s.first = x + 1;
      if constexpr (Coord::kIdentity) lo_ = x;
      span_ = 0;
      raw_lo_ = 1, raw_hi_ = 0;
    }
    return true;
  }
  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    if constexpr (W) {
      if (__builtin_expect(ov_n() != 0 || main_n_, 0)) return scan_with_overflow(key, n, out);
    }
    if (__builtin_expect(live_ == 0, 0)) {  // every bulk key removed: the exceptions are all there is
      size_t got = 0;
      for (auto e = std::lower_bound(exc_.begin(), exc_.end(), key, [](const std::pair<K, P>& a, K k) { return a.first < k; }); got < n && e != exc_.end(); ++e) out[got++] = *e;
      return got;
    }
    return scan_sorted_slots_last(slots_, exc_, key, n, out, [this](uint64_t x) { return lower_slot(x); });
  }
  long long memory_consumption() override {
    if constexpr (W) {
      if (puts_line_ + puts_general_ + puts_ov_)
        std::fprintf(stderr, "gret: %s puts: in_line=%zu next_line=%zu general=%zu to_overflow=%zu overflow_now=%zu live_in_slots=%zu rule=%s rebuilds=%zu cells_rebuilt=%zu keys_taken_back=%zu cells_refused=%zu cells_apart=%zu cells_in_place=%zu merges=%zu moved_between_tables=%zu rot=%u back=%zu\n",
                     name_.c_str(), puts_line_, puts_next_, puts_general_, puts_ov_, ov_n(), live_, put_wide_ ? "wide" : put_next_ ? "line2" : "line", rebuilds_, cell_rebuilds_, cell_taken_, cell_refused_, regions_.size(), cell_inplace_, rotations_, moved_back_, rot_, puts_back_);
      return (long long)(slots_.size() * sizeof(Slot) + nodes_.size() * sizeof(Node) + exc_.size() * sizeof(std::pair<K, P>) + coord_.bytes() + ov_.bytes() + frz_.bytes() + drain_cells_.capacity() * 4 + ov_sorted_.capacity() * sizeof(std::pair<K, P>) + ov_bits_.size() * 8 + cell_bytes());
    } else {
      return memory_consumption_ro();
    }
  }
  long long memory_consumption_ro() {
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
    // rs_cnt_ keys from slot rs_base_ on: all the bulk from 0 in bulk_load, one cell's keys in slots of their own in rebuild_cell
    if constexpr (W) {  // an in-place layout: the keys spread evenly over the rs_all_ slots the cell already has
      if (__builtin_expect(rs_all_ != 0, 0)) return rs_base_ + (r >= rs_cnt_ ? rs_all_ : size_t(double(r) * double(rs_all_) / double(rs_cnt_)));
    }
    const size_t all = size_t(std::ceil(double(rs_cnt_) * per_)) * B;
    if (r >= rs_cnt_) return rs_base_ + all;
    if (!slack_.empty() && all > rs_cnt_) {  // one slot a key, and the spare ones in proportion to the weight before rank r
      const auto it = std::upper_bound(slack_.begin(), slack_.end(), r, [](size_t x, const SlackAt& a) { return x < a.i; }) - 1;
      return std::min(r + size_t((it->cum + double(r - it->i) * it->w) / slack_total_ * double(all - rs_cnt_)), all - 1);
    }
    return rs_base_ + size_t(double(r) * per_ * double(B));
  }
  void build_tree(size_t cells, double spk) {
    per_ = spk / B;
    if (rank_slot(rs_cnt_) >= kInner) throw std::length_error("AdaptiveLineIndex: 2^31 slots");
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
      a = std::min(a, rank_slot(rs_cnt_) - 1);  // past the last key: a never-stored key can get here and must read inside the slots
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
    if (collect_ && nk) leaves_.push_back(LeafAt{uint64_t(i), uint32_t(nk), uint32_t(at)});
  }
  // AB_V's search, leaf by leaf in key order, behind the real backlog. A line replaces the leaf's rank line only if its keys read fewer
  // lines (counting the next leaf's behind whatever backlog it leaves), none of them reads more than the rank line's worst, and it ends
  // no later than the rank line would or the leaf's own last slot. Positions only move forward when the slots before fill later, so
  // what follows cannot read more because of it: the mean, the worst key and the bytes can only stay or improve.
  void fit_lines() {
    const size_t total = rank_slot(rs_cnt_);
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
              if (a < double(rs_base_) || c < 1 || c > double(kSlots) || a + c > double(total)) continue;
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
  size_t key_bucket(uint64_t k) const { return bucket(coord_(k) - lo_); }  // a stored key's bucket, through the coordinate
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
    size_t lo_b, n;
    home(x, lo_b, n);
    const size_t i0 = bucket(std::min(coord_(x) - lo_, span_)) * B;
    size_t lo = i0, hi = i0, step = 1;
    if (slots_[i0].first < x) {
      lo = hi = i0 + 1;
      for (; hi < n && slots_[hi].first < x; hi += step, step *= 2) lo = hi + 1;
      hi = std::min(hi, n);
    } else {
      for (; lo > lo_b && slots_[lo - 1].first >= x; lo = lo - lo_b > step ? lo - step : lo_b, step *= 2) hi = lo - 1;
    }
    return size_t(std::lower_bound(slots_.begin() + std::ptrdiff_t(lo), slots_.begin() + std::ptrdiff_t(hi), x, [](const Slot& a, uint64_t k) { return a.first < k; }) - slots_.begin());
  }
  static size_t place(size_t p, size_t prev) {  // the first free slot of bucket p or later
    const size_t s = p * B;
    return prev == SIZE_MAX || s > prev ? s : prev + 1;
  }
  // get's walk, for the writes: the first slot holding x (kSlot), x not stored (kNone), or x outside the bulk (kException)
  enum class At { kSlot, kNone, kException };
  // around: also fetch the slot before the bucket and the line after it, for remove
  At find_slot(uint64_t x, size_t& i, int around) const {  // around: 1 = fetch the slot before the bucket, 2 = the line after it
    if constexpr (!Coord::kIdentity) {
      if (__builtin_expect(x < raw_lo_ || x > raw_hi_, 0)) return At::kException;
    }
    const uint64_t v = coord_(x) - lo_;
    if (__builtin_expect(v > span_, 0)) return At::kException;
    if (__builtin_expect(live_ == 0, 0)) return At::kNone;
    size_t b = bucket(v);
    const size_t b0 = b;
    // a prefetch never faults, so the slots before the first and after the last need no check
    if (around & 1) __builtin_prefetch(slots_.data() + b * B - 1);
    if (around & 2) __builtin_prefetch(slots_.data() + b * B + B);
    for (;;) {
      const Slot* l = &slots_[b * B];
      size_t c = 0;
      for (unsigned j = 0; j < B; ++j) c += size_t(l[j].first < x);
      if (__builtin_expect(c < B, 1)) {
        if (l[c].first != x) return At::kNone;
        i = b * B + c;
        return At::kSlot;
      }
      if (++b > b0 + d_ || b >= total_) return At::kNone;
    }
  }
  typename std::vector<std::pair<K, P>>::iterator exception_at(K key) {
    const auto it = std::lower_bound(exc_.begin(), exc_.end(), key, [](const std::pair<K, P>& a, K k) { return a.first < k; });
    return it != exc_.end() && it->first == key ? it : exc_.end();
  }
  size_t cell_bytes() const {  // what the rebuild bookkeeping holds
    size_t b = cell_ov_.capacity() * sizeof(std::vector<uint64_t>) + cell_t_.capacity() * 8 + cell_skip_.capacity() * 4;
    for (const auto& l : cell_ov_) b += l.capacity() * 8;
    return b;
  }
  // whether a key whose bucket is this line may be in the overflow: one bit a line, set when such a key is sent there and never
  // cleared (a rebuilt cell starts on fresh lines; a rebuild of everything starts over)
  bool ov_line(size_t line) const {
    if constexpr (W) {
      if (__builtin_expect(!ov_bit_on_, 0)) return ov_n() != 0;
      return line < ov_bits_.size() * 64 && ((ov_bits_[line >> 6] >> (line & 63)) & 1);
    } else {
      return (void)line, false;
    }
  }
  bool update_absent(uint64_t x, P val) {  // not in the slots
    if constexpr (W) return ov_dirty_ = true, ov_assign(x, val);
    else return (void)x, (void)val, false;
  }
  bool update_outside(K key, P val) {  // outside the span: an exception, or (W) a key the overflow took
    const auto it = exception_at(key);
    if (it == exc_.end()) return update_absent(uint64_t(key), val);
    it->second = val;
    return true;
  }
  // for put: key k, whose copies would start at slot f and end at slot r, is where get finds it
  bool reachable(uint64_t k, size_t f, size_t r) const {
    const size_t b = key_bucket(k), first = std::max(f, b * B);
    return first <= r && first / B <= b + d_;
  }
  // line: the key's bucket, or SIZE_MAX for a key outside the directory's span
  bool ov_insert(uint64_t x, P val, size_t line) {
    if (__builtin_expect(frz_live_, 0) && frz_.contains(x)) return false;
    if (!ov_.insert(x, val)) return false;
    ov_dirty_ = true;
    if (line == SIZE_MAX) ++ov_out_;
    else if (ov_bit_on_) {
      if (ov_bits_.size() * 64 <= line) ov_bits_.resize(std::max(total_, line + 1) / 64 + 1, 0);
      ov_bits_[line >> 6] |= uint64_t(1) << (line & 63);
    }
    if constexpr (W) {
      if (rw_ > 0) note_overflow(x);
    }
    return true;
  }
  // the slots a key lives among: [lo_b, hi_b)
  void home(uint64_t x, size_t& lo_b, size_t& hi_b) const {
    lo_b = 0, hi_b = slots_.size();
    if constexpr (W) {
      if (main_n_) {
        hi_b = main_n_;
        const uint64_t v = x - lo_;
        if (v <= span_) {
          const auto it = regions_.find(uint32_t(v >> shift_));
          if (it != regions_.end()) lo_b = it->second.s0, hi_b = it->second.s1;
        }
      }
    }
  }
  struct SlotView {  // a run of the slots, for the scan
    const Slot* p;
    size_t n;
    const Slot& front() const { return p[0]; }
    const Slot& back() const { return p[n - 1]; }
    size_t size() const { return n; }
    const Slot& operator[](size_t i) const { return p[i]; }
  };
  size_t main_slots() const { return main_n_ ? main_n_ : slots_.size(); }
  size_t lower_main(uint64_t x) const {  // the first main slot whose key is >= x, by bisection: the main slots stay one sorted run
    const auto b = slots_.begin(), e = b + std::ptrdiff_t(main_slots());
    return size_t(std::lower_bound(b, e, x, [](const Slot& a, uint64_t k) { return a.first < k; }) - b);
  }
  // A key went to the overflow. Per root cell of the directory: rent or buy again, on the cell's own keys. The cell's overflow
  // keys have cost (rw_ lookups a write) x (their share of the keys) x ov_ns_ since each arrived; laying the cell out again costs
  // the build's time per key times its keys. Once the first reaches the second the cell is rebuilt, long before the whole index
  // would be: a hot range pays for itself alone.
  void note_overflow(uint64_t x) {
    const uint64_t v = x - lo_;
    if (v > span_ || live_ == 0) return;
    const uint32_t c = uint32_t(v >> shift_);
    auto& list = cell_ov_[c];
    list.push_back(x);
    double& ts = cell_t_[c];
    ts += double(t_put_);
    if (list.size() < cell_skip_[c]) return;
    const double all = double(live_ + ov_n()), cnt = double(list.size());
    const double paid = rw_ * ov_ns_ / all * (cnt * double(t_put_) - ts);
    const uint64_t v1 = std::min((uint64_t(c) << shift_) + ((uint64_t(1) << shift_) - 1), span_);
    const double held = double((bucket(v1) - bucket(uint64_t(c) << shift_) + 1) * B) / (per_ * double(B));
    if (paid < build_ns_ * (held + cnt)) return;
    // Only a cell that is hot: its overflow share at least AW_HOT times the index's. When the keys arrive everywhere alike every
    // cell reaches its break-even at about the same time, and one rebuild of everything costs less per key than all of them
    // apart (no guard lines, no slots left behind), so those are left to it.
    if (cnt * all < hot_ * double(ov_n()) * (held + cnt)) return;
    if (!rebuild_cell(c)) cell_skip_[c] = uint32_t(std::min<size_t>(2 * cell_ov_[c].size() + 1, 0xffffffffu)), ++cell_refused_;  // not again until it has doubled
  }
  // T and V of one root cell redone, with G's keys of that cell taken back: the cell's keys (in its slots and in the overflow)
  // are laid out by the build's own rules in slots of their own after the main ones, and the cell's root node is pointed there.
  // A lookup in the cell then reads as in a fresh index, with no step added: only the node it starts from has changed. The old
  // slots become copies of the next key (free, for the neighbours), each rebuilt cell ends in d_ + 2 lines of copies of its last
  // key so that no walk leaves it, and d_ is never raised (a cell that would need more is refused).
  bool rebuild_cell(uint32_t c) {
    if constexpr (!W) {
      return (void)c, false;
    } else {
      if (live_ == 0) return false;
      if (inplace_ && relay_in_place(c)) return true;
      const uint64_t x0 = lo_ + (uint64_t(c) << shift_);
      const bool last_cell = (span_ >> shift_) == c;
      const auto rit = regions_.find(c);
      const bool was = rit != regions_.end();
      if (!main_n_) {  // the first rebuilt cell: the main slots end in guard lines of their own from now on
        if (slots_.size() + (d_ + 2) * B > slots_.capacity()) return false;
        slots_.resize(slots_.size() + (d_ + 2) * B, slots_.back());
        main_n_ = slots_.size(), total_ = slots_.size() / B;
      }
      size_t g0, g1;
      if (was) g0 = rit->second.s0, g1 = rit->second.s1;
      else g0 = lower_main(x0), g1 = last_cell ? main_slots() : lower_main(lo_ + (uint64_t(c + 1) << shift_));
      if (!was && g0 == 0 && g1 == main_slots()) return false;  // the main slots would be left with no key: the whole rebuild's job
      std::vector<std::pair<K, P>> kv;
      for (size_t i = g0; i < g1; ++i)
        if (i + 1 == g1 || slots_[i + 1].first != slots_[i].first) kv.emplace_back(K(slots_[i].first), slots_[i].second);
      const size_t in_slots = kv.size();
      std::vector<uint64_t> taken;
      if (c < cell_ov_.size()) {
        auto& lst = cell_ov_[c];
        std::sort(lst.begin(), lst.end());
        lst.erase(std::unique(lst.begin(), lst.end()), lst.end());
        for (uint64_t k : lst) {
          P v;
          if (ov_find(k, v)) kv.emplace_back(K(k), v), taken.push_back(k);
        }
        if (taken.empty()) lst.clear();  // every one of them was removed since
      }
      if (taken.empty() || kv.empty()) return false;
      std::inplace_merge(kv.begin(), kv.begin() + std::ptrdiff_t(in_slots), kv.end(), [](const std::pair<K, P>& a, const std::pair<K, P>& b) { return a.first < b.first; });
      const size_t nk = kv.size(), pad = (d_ + 2) * B, data = size_t(std::ceil(double(nk) * per_)) * B;
      if (slots_.size() + 2 * data + pad > slots_.capacity() || data + pad > arena_left_) return false;
      const size_t s0 = slots_.size(), nodes_before = nodes_.size(), levels_before = levels_;
      const Node root_before = nodes_[c];
      d_keys_ = kv.data(), rs_base_ = s0, rs_cnt_ = nk;
      leaves_.clear(), collect_ = fit_;
      bool ok = true;
      size_t prev = SIZE_MAX;
      try {
        make_node(c, 0, nk, shift_, 0);
        if (fit_) fit_lines();
        for (size_t r = 0; r < nk && ok; ++r) {
          const size_t p = bucket(key(r) - lo_), pos = place(p, prev);
          ok = pos >= s0 && pos / B - p <= d_;
          prev = pos;
        }
      } catch (const std::exception&) {
        ok = false;
      }
      const size_t data_end = ok ? ((std::max(rank_slot(nk), prev + 1) + B - 1) / B) * B : 0;
      ok = ok && data_end + pad <= slots_.capacity();
      std::vector<LeafAt>().swap(leaves_);
      if (!ok) {  // as it was
        nodes_.resize(nodes_before), nodes_[c] = root_before, levels_ = levels_before;
        d_keys_ = nullptr, rs_base_ = 0, rs_cnt_ = m_;
        return false;
      }
      slots_.resize(data_end + pad);
      size_t w = s0;
      prev = SIZE_MAX;
      for (size_t r = 0; r < nk; ++r) {
        const size_t pos = place(bucket(key(r) - lo_), prev);
        const Slot sl(uint64_t(kv[r].first), kv[r].second);
        while (w <= pos) slots_[w++] = sl;
        prev = pos;
      }
      const Slot lastp(uint64_t(kv[nk - 1].first), kv[nk - 1].second);
      while (w < slots_.size()) slots_[w++] = lastp;
      total_ = slots_.size() / B;
      d_keys_ = nullptr, rs_base_ = 0, rs_cnt_ = m_;
      if (!was) {  // the cell's old slots: copies of the next key, or of the one before when the cell was the last
        const Slot fill = g1 < main_n_ ? slots_[g1] : slots_[g0 - 1];
        std::fill(slots_.begin() + std::ptrdiff_t(g0), slots_.begin() + std::ptrdiff_t(g1), fill);
      }
      regions_[c] = Region{s0, slots_.size()};
      for (uint64_t k : taken) ov_erase(k);
      std::vector<uint64_t>().swap(cell_ov_[c]), cell_t_[c] = 0, cell_skip_[c] = 0;
      live_ += taken.size();
      arena_left_ -= std::min(arena_left_, slots_.size() - s0);
      ov_dirty_ = true, removed_ = true;
      ++cell_rebuilds_, cell_taken_ += taken.size();
      return true;
    }
  }
  // The same, without new slots: the cell's keys and the overflow's keys of that cell are laid out by the build's rules over the
  // slots the cell already has (its run of the main slots, or its own slots if it was rebuilt before), so V's spare slots take
  // G's keys back and T's nodes for the cell are made again for the new keys. Refused, leaving everything as it was, when the
  // keys do not fit, when a key would be more than d_ lines from its bucket, or below AW_DENSE slots a key. The slots after the
  // cell's last key become copies of it (as after a remove), the ones before its first key copies of that key.
  bool relay_in_place(uint32_t c) {
    if constexpr (!W) {
      return (void)c, false;
    } else {
      if (c >= cell_ov_.size() || cell_ov_[c].empty()) return false;
      const bool last_cell = (span_ >> shift_) == c;
      const auto rit = regions_.find(c);
      const bool was = rit != regions_.end();
      const size_t pad = (d_ + 2) * B;
      size_t g0, g1, end;  // the cell's slots are [g0, g1); keys may be placed up to end
      if (was) {
        g0 = rit->second.s0, g1 = rit->second.s1, end = g1 - pad;
      } else {
        g0 = lower_main(lo_ + (uint64_t(c) << shift_)), g1 = last_cell ? main_slots() : lower_main(lo_ + (uint64_t(c + 1) << shift_));
        end = main_n_ && g1 == main_n_ ? g1 - std::min(pad, g1) : g1;  // not into the main slots' guard lines
      }
      const size_t base = (g0 + B - 1) / B * B;  // whole lines, as the build lays keys out
      if (base >= end) return false;
      const size_t room = end - base;
      std::vector<std::pair<K, P>> kv;
      for (size_t i = g0; i < g1; ++i)
        if (i + 1 == g1 || slots_[i + 1].first != slots_[i].first) kv.emplace_back(K(slots_[i].first), slots_[i].second);
      const size_t in_slots = kv.size();
      if (in_slots == 0) return false;
      auto& lst = cell_ov_[c];
      std::sort(lst.begin(), lst.end());
      lst.erase(std::unique(lst.begin(), lst.end()), lst.end());
      std::vector<uint64_t> taken;
      for (uint64_t k : lst) {
        P v;
        if (ov_find(k, v)) kv.emplace_back(K(k), v), taken.push_back(k);
      }
      if (taken.empty()) return lst.clear(), false;
      const size_t nk = kv.size();
      if (double(room) < dense_min_ * double(nk) || room < nk) return false;
      std::inplace_merge(kv.begin(), kv.begin() + std::ptrdiff_t(in_slots), kv.end(), [](const std::pair<K, P>& a, const std::pair<K, P>& b) { return a.first < b.first; });
      const size_t nodes_before = nodes_.size(), levels_before = levels_;
      const Node root_before = nodes_[c];
      d_keys_ = kv.data(), rs_base_ = base, rs_cnt_ = nk, rs_all_ = room;
      leaves_.clear(), collect_ = fit_;
      bool ok = true;
      size_t prev = SIZE_MAX;
      try {
        make_node(c, 0, nk, shift_, 0);
        if (fit_) fit_lines();
        for (size_t r = 0; r < nk && ok; ++r) {
          const size_t p = bucket(key(r) - lo_), pos = place(p, prev);
          ok = pos >= base && pos < end && pos / B - p <= d_;
          prev = pos;
        }
      } catch (const std::exception&) {
        ok = false;
      }
      std::vector<LeafAt>().swap(leaves_);
      if (!ok) {  // as it was
        nodes_.resize(nodes_before), nodes_[c] = root_before, levels_ = levels_before;
        d_keys_ = nullptr, rs_base_ = 0, rs_cnt_ = m_, rs_all_ = 0;
        return false;
      }
      size_t w = g0;
      prev = SIZE_MAX;
      for (size_t r = 0; r < nk; ++r) {
        const size_t pos = place(bucket(key(r) - lo_), prev);
        const Slot sl(uint64_t(kv[r].first), kv[r].second);
        while (w <= pos) slots_[w++] = sl;
        prev = pos;
      }
      const Slot lastp(uint64_t(kv[nk - 1].first), kv[nk - 1].second);
      while (w < g1) slots_[w++] = lastp;
      d_keys_ = nullptr, rs_base_ = 0, rs_cnt_ = m_, rs_all_ = 0;
      if (!was && g1 == slots_.size()) last_key_ = lastp.first;
      for (uint64_t k : taken) ov_erase(k);
      std::vector<uint64_t>().swap(lst), cell_t_[c] = 0, cell_skip_[c] = 0;
      live_ += taken.size();
      ov_dirty_ = true, removed_ = true;
      ++cell_inplace_, cell_taken_ += taken.size();
      return true;
    }
  }
  // AW_ROT. The merge of the overflow into the slots, a share at a time. start_drain lists the cells that hold overflow keys (and,
  // with three tables, freezes the table: it is moved aside whole, in constant time, and an empty one takes the puts). Each put
  // then pays for drain_rate_ keys of layout, and a cell is laid out again (in place if it fits, else in new slots) once its keys
  // are paid for: no put waits for more than one cell. The rate finishes the merge in half the puts the overflow took to fill.
  double cell_held(uint32_t c) const {  // about how many keys the cell's slots hold
    const uint64_t v0 = uint64_t(c) << shift_, v1 = std::min(v0 + ((uint64_t(1) << shift_) - 1), span_);
    const size_t b0 = bucket(v0), b1 = bucket(v1);
    return double(((b1 > b0 ? b1 - b0 : 0) + 1) * B) / (per_ * double(B));
  }
  void start_drain() {
    debt_ = 0;
    if (rot_ == 3) {
      frz_ = std::move(ov_);
      ov_ = OverflowTable<P>();
      ov_.set_dense(ov_dense_);
      frz_live_ = true;
    }
    drain_cells_.clear();
    double keys = 0;
    for (size_t c = 0; c < cell_ov_.size(); ++c)
      if (!cell_ov_[c].empty()) drain_cells_.push_back(uint32_t(c)), keys += double(cell_ov_[c].size()) + cell_held(uint32_t(c));
    drain_pos_ = 0, drain_keys_ = 0, drain_back_ = 0, drain_credit_ = 0;
    drain_rate_ = std::max(2 * keys / double(std::max<size_t>(t_put_ - drain_last_t_, 1)), 1.0);
    drain_last_t_ = t_put_;
    draining_ = true;
    ++rotations_;
  }
  void drain_step() {
    drain_credit_ += drain_rate_;
    while (live_ && drain_pos_ < drain_cells_.size()) {
      const uint32_t c = drain_cells_[drain_pos_];
      const size_t cnt = cell_ov_[c].size();
      if (cnt == 0) {  // laid out since, by its own break-even
        ++drain_pos_;
        continue;
      }
      const double cost = double(cnt) + cell_held(c);
      if (drain_credit_ < cost) return;
      drain_credit_ -= cost, ++drain_pos_;
      if (rebuild_cell(c)) {
        drain_keys_ += cnt;
        continue;
      }
      const size_t left = cell_ov_[c].size();
      drain_keys_ += left, drain_back_ += left;
      if (left) cell_skip_[c] = uint32_t(std::min<size_t>(2 * left + 1, 0xffffffffu)), ++cell_refused_;
      if (rot_ == 3)  // its keys stay in the overflow: out of the frozen table, which must end empty
        for (uint64_t k : cell_ov_[c]) {
          P v;
          if (frz_.find(k, v)) frz_.erase(k), ov_.insert(k, v), ++moved_back_;
        }
    }
    if (rot_ == 3) {  // what no cell lists (keys from outside the span)
      frz_.for_each([this](uint64_t k, P v) { ov_.insert(k, v), ++moved_back_; });
      frz_ = OverflowTable<P>(), frz_live_ = false;
    }
    draining_ = false, drain_credit_ = 0;
    drain_fail_ = live_ == 0 || drain_back_ * 2 > drain_keys_;  // most keys found no room: the next break-even rebuilds everything
  }
  size_t ov_n() const { return ov_.size() + frz_.size(); }
  bool ov_find(uint64_t x, P& val) const { return ov_.find(x, val) || (__builtin_expect(frz_live_, 0) && frz_.find(x, val)); }
  bool ov_has(uint64_t x) const {
    P v;
    return ov_find(x, v);
  }
  bool ov_erase(uint64_t x) { return ov_.erase(x) || (frz_live_ && frz_.erase(x)); }
  bool ov_assign(uint64_t x, P val) { return ov_.assign(x, val) || (frz_live_ && frz_.assign(x, val)); }
  // Every key laid out again by bulk_load (G merged back, T and V fresh), at the byte budget the index was built with. skip:
  // without this key (remove calls it when a run of slots would be left with no key to copy).
  void rebuild_all(bool skip, uint64_t skip_key) {
    if constexpr (W) {
      std::vector<std::pair<K, P>> kv;
      const auto less = [](const std::pair<K, P>& a, const std::pair<K, P>& b) { return a.first < b.first; };
      auto run = [&](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i)
          if ((i + 1 == b || slots_[i + 1].first != slots_[i].first) && !(skip && slots_[i].first == skip_key)) kv.emplace_back(K(slots_[i].first), slots_[i].second);
      };
      // the main slots hold real keys unless everything was removed (then live_ is 0 and they hold a key no lookup reaches)
      bool main_real = live_ > 0;
      if (skip && !main_n_) main_real = live_ > 0;
      if (main_real || main_n_) run(0, main_slots());
      size_t mid = kv.size();
      for (const auto& r : regions_) run(r.second.s0, r.second.s1);
      std::inplace_merge(kv.begin(), kv.begin() + std::ptrdiff_t(mid), kv.end(), less);
      mid = kv.size();
      kv.insert(kv.end(), exc_.begin(), exc_.end());
      std::inplace_merge(kv.begin(), kv.begin() + std::ptrdiff_t(mid), kv.end(), less);
      mid = kv.size();
      ov_.for_each([&](uint64_t k, P v) { kv.emplace_back(K(k), v); });
      frz_.for_each([&](uint64_t k, P v) { kv.emplace_back(K(k), v); });
      std::sort(kv.begin() + std::ptrdiff_t(mid), kv.end(), less);
      std::inplace_merge(kv.begin(), kv.begin() + std::ptrdiff_t(mid), kv.end(), less);
      ++rebuilds_;
      if (kv.empty()) {  // nothing is left anywhere: one run of slots again, holding a key no lookup reaches
        regions_.clear(), cell_ov_.clear(), cell_t_.clear(), cell_skip_.clear();
        main_n_ = 0, live_ = 0;
        ov_ = OverflowTable<P>(), ov_sorted_.clear(), ov_dirty_ = false;
        ov_.set_dense(ov_dense_);
        frz_ = OverflowTable<P>(), frz_live_ = false, draining_ = false, drain_cells_.clear();
        for (Slot& sl : slots_) sl.first = skip_key + 1;
        lo_ = skip_key, span_ = 0, raw_lo_ = 1, raw_hi_ = 0;
        return;
      }
      const size_t nxt = puts_next_;
      const size_t lines = puts_line_, general = puts_general_, to_ov = puts_ov_, rb = rebuilds_, cr = cell_rebuilds_, ct = cell_taken_, cf = cell_refused_, tp = t_put_;
      const size_t ci = cell_inplace_, ro = rotations_, mb = moved_back_, pb = puts_back_;
      std::vector<Slot, BucketAlloc<Slot>>().swap(slots_);
      bulk_load(kv.data(), kv.size());
      puts_next_ = nxt;
      puts_line_ = lines, puts_general_ = general, puts_ov_ = to_ov, rebuilds_ = rb, cell_rebuilds_ = cr, cell_taken_ = ct, cell_refused_ = cf, t_put_ = tp;
      cell_inplace_ = ci, rotations_ = ro, moved_back_ = mb, drain_last_t_ = tp, puts_back_ = pb;
      removed_ = true;
    } else {
      (void)skip, (void)skip_key;
    }
  }
  // scan when keys sit in the overflow: the slots' keys and the overflow's, merged. The overflow has no order, so a sorted copy
  // is kept and redone after a write to it.
  size_t scan_with_overflow(K key, size_t n, std::pair<K, P>* out) {
    if (ov_dirty_) {
      ov_sorted_.clear();
      ov_.for_each([this](uint64_t k, P v) { ov_sorted_.emplace_back(K(k), v); });
      frz_.for_each([this](uint64_t k, P v) { ov_sorted_.emplace_back(K(k), v); });
      std::sort(ov_sorted_.begin(), ov_sorted_.end(), [](const std::pair<K, P>& a, const std::pair<K, P>& b) { return a.first < b.first; });
      ov_dirty_ = false;
    }
    std::vector<std::pair<K, P>> main(n);
    size_t got = 0;
    if (live_ == 0) {
      for (auto e = std::lower_bound(exc_.begin(), exc_.end(), key, [](const std::pair<K, P>& a, K k) { return a.first < k; }); got < n && e != exc_.end(); ++e) main[got++] = *e;
    } else if (!main_n_) {
      got = scan_sorted_slots_last(slots_, exc_, key, n, main.data(), [this](uint64_t x) { return lower_slot(x); });
    } else {
      // rebuilt cells: the main slots are one sorted run (a rebuilt cell's old slots are copies of the key after it), and the
      // cells are sorted runs of their own, in key order by cell. Each gives its n smallest keys >= key; then they are merged.
      got = scan_sorted_slots_last(SlotView{slots_.data(), main_n_}, exc_, key, n, main.data(), [this](uint64_t x) { return lower_main(x); });
      std::vector<std::pair<K, P>> cells(n);
      size_t gc = 0;
      const uint64_t kx = uint64_t(key);
      auto r = kx <= lo_ ? regions_.begin() : (kx - lo_ > span_ ? regions_.end() : regions_.lower_bound(uint32_t((kx - lo_) >> shift_)));
      const std::vector<std::pair<K, P>> none;
      for (; r != regions_.end() && gc < n; ++r) {
        const size_t s0 = r->second.s0;
        gc += scan_sorted_slots_last(SlotView{slots_.data() + s0, r->second.s1 - s0}, none, key, n - gc, cells.data() + gc, [this, s0](uint64_t x) { return lower_slot(x) - s0; });
      }
      std::vector<std::pair<K, P>> both(got + gc);
      std::merge(main.begin(), main.begin() + std::ptrdiff_t(got), cells.begin(), cells.begin() + std::ptrdiff_t(gc), both.begin(), [](const std::pair<K, P>& a, const std::pair<K, P>& b) { return a.first < b.first; });
      got = std::min(n, both.size());
      std::copy(both.begin(), both.begin() + std::ptrdiff_t(got), main.begin());
    }
    auto o = std::lower_bound(ov_sorted_.begin(), ov_sorted_.end(), key, [](const std::pair<K, P>& a, K k) { return a.first < k; });
    size_t i = 0, w = 0;
    while (w < n && (i < got || o != ov_sorted_.end())) out[w++] = o == ov_sorted_.end() || (i < got && main[i].first < o->first) ? main[i++] : *o++;
    return w;
  }
  bool find_exception(K key, P& val) const {
    const auto it = std::lower_bound(exc_.begin(), exc_.end(), key,
                                     [](const std::pair<K, P>& a, K k) { return a.first < k; });
    if (it == exc_.end() || it->first != key) return false;
    val = it->second;
    return true;
  }

  std::string name_;
  uint64_t lo_ = 1, span_ = 0, raw_lo_ = 1, raw_hi_ = 0;
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
  // for the writes. Last, so that no member get reads moves: its code stays byte for byte what it was
  size_t live_ = 0;        // bulk keys not removed
  uint64_t last_key_ = 0;  // the last key of the slots, whose copies fill the slots after it
  bool removed_ = false;   // a bulk key was removed: a run may go on past the slot get reads
  // during bulk_load only
  std::vector<SlackAt> slack_;
  double slack_total_ = 0;
  bool collect_ = false;
  size_t rs_base_ = 0, rs_cnt_ = 0;  // rank_slot's base and count: the bulk, or one cell in rebuild_cell
  // W only
  static constexpr size_t kShift = 8;  // slots one put may move: two cache lines
  OverflowTable<P> ov_;
  std::vector<std::pair<K, P>> ov_sorted_;
  bool ov_dirty_ = false;
  bool put_wide_ = false;  // AW_PUT=wide
  bool put_next_ = false;  // AW_PUT=line2
  size_t puts_line_ = 0, puts_next_ = 0, puts_general_ = 0, puts_ov_ = 0;
  std::vector<uint64_t> ov_bits_;  // by line: a key with this bucket was sent to the overflow
  size_t ov_out_ = 0;              // keys sent to the overflow from outside the span
  bool ov_bit_on_ = true;
  // rebuilds (AW_RW > 0)
  struct Region {
    size_t s0, s1;  // a rebuilt cell's slots, guard lines included
  };
  std::map<uint32_t, Region> regions_;  // by root cell, so in key order
  size_t main_n_ = 0;                   // the main slots, guard lines included, once a cell was rebuilt (0: all the slots are main)
  std::vector<std::vector<uint64_t>> cell_ov_;  // the overflow's keys by root cell (some may have been removed since)
  std::vector<double> cell_t_;                  // the sum of the times they arrived
  std::vector<uint32_t> cell_skip_;             // a refused cell: not tried again below this many keys
  double rw_ = 0, ov_ns_ = 60, build_ns_ = 150, debt_ = 0, hot_ = 3;
  size_t t_put_ = 0, arena_left_ = 0, rebuilds_ = 0, cell_rebuilds_ = 0, cell_taken_ = 0, cell_refused_ = 0;
  // in-place layouts (AW_INPLACE) and the merge a share at a time (AW_ROT)
  OverflowTable<P> frz_;  // AW_ROT=3: the frozen table, read after ov_ and emptied cell by cell
  bool frz_live_ = false, ov_dense_ = false, inplace_ = false, draining_ = false, drain_fail_ = false;
  unsigned rot_ = 0;
  double dense_min_ = 1, drain_credit_ = 0, drain_rate_ = 0;
  size_t rs_all_ = 0;  // rank_slot: the slots an in-place layout spreads the cell's keys over (0: per_ slots a key)
  std::vector<uint32_t> drain_cells_;
  size_t drain_pos_ = 0, drain_last_t_ = 0, drain_keys_ = 0, drain_back_ = 0;
  size_t rotations_ = 0, cell_inplace_ = 0, moved_back_ = 0;
  // the trials of 2026-11-23 (AB_DEL, AB_UPD_DRY, AW_BACK)
  int del_mode_ = 0;
  bool upd_dry_ = false, back_ = false;
  size_t puts_back_ = 0;
};

}  // namespace gret
