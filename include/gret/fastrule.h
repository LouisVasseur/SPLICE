// SPDX-License-Identifier: MIT
// FastPeriodRule: the period rule of ruleline.h in fewer instructions, the coordinate of AdaptiveLineIndex in gp4. Its coordinate is
// gr4's up to a constant (rows counted from the first stored key's row, and y(first stored key) = 0, so AdaptiveLineIndex's raw
// order check sees y - y(lo)), up to the knots raised below, and, where S is not a power of two, up to one unit at a few keys (S is
// multiplied into the tables here, in ruleline.h after the multiply-add). What a G costs inside GRE is mostly how many instructions it adds before the directory's first read: they fill the
// out-of-order window, so fewer lookups overlap their misses (the 22-block rule: 9.95 ns in a tight loop, 43-81 ns per lookup in
// GRE). Measured on a layout the rule leaves unchanged, a branch-free rewrite with fixed-point arithmetic doubled the cost, so this
// keeps gr4's arithmetic, y = base(c) + q W(c) + floor(S F(r)) + r with W(c) = S + P(c) (ruleline.h), and drops the instructions
// that do not change it:
//   rule    $RULE_FILE's (ruleline.h's format; $RULE_S overrides its S), or one given from memory (FastPeriodRule::Given: the gate's
//           discovery). load() checks it: at least one block, every period >= 1, first keys strictly increasing, the first block
//           at or below the first stored key, S < 2^63 and S + P < 2^64. Without a rule, one identity block, S = 0 and P = 2^63, so
//           y = x - lo
//   block   only the blocks over the stored keys are kept, each with its last key (2^64 - 1 for the last one), so there is no
//           end-of-blocks test. With more than one, a table on the key's top bits names the block of each bucket's first key, as
//           an 8-byte offset, so the block's address is one base + index operand; the table's base is biased by the first bucket,
//           so the key needs no subtraction either. The buckets are made fine enough (2^12..2^16 of them) that none holds two
//           boundaries: a key past its block's last is one rarely taken branch, to general(), which moves on (one block, or more
//           where boundaries are closer than 2^16 buckets can separate). load() counts those steps over the stored keys, refines
//           the table while they pass one per 32 keys (plus 64), and refuses a rule still past that at 2^16 buckets: a cost the
//           gate's prediction (lines and levels) would not see. Absent keys are not counted: one between packed boundaries walks
//           past all of them (exact, but slow)
//   q, r    one multiply by 2^64 / P, rounded up where load() proves it exact for every key of the block's range (xp e < 2^64,
//           where e = ceil(2^64 / P) P - 2^64 < P: xp P < 2^64 suffices, and a power of 2 always is): no correction. Where it
//           cannot (P = 1, or P and the block both wide), the reciprocal is rounded down, q is at most one low, and one correction
//           step in general() puts it right, a branch it rarely takes; every lookup then goes through general() (each block's
//           fast_to = 0), so the branch to it is taken every time rather than by block, and stays predictable
//   direct  one block with an exact reciprocal (the identity block among them): no table and no test at all. Counted on gp4's get()
//           (clang x86-64 -O3 -march=icelake-server, a stored key at one level): 99 instructions direct, 101 through the table
//           with no such shortcut, 103 through the table with it, so it saves one-block rules 2 instructions and the table's
//           dependent load, and costs the others one compare and branch
//   phase   S F(r) from tables already multiplied by S and indexed by r's octave, lzcnt(r) ^ 63 (r's bit length less one, 127 for
//           r = 0), with r's offset into its octave from bzhi: one multiply and one add. A knot is raised only where rounding
//           would make floor(S F) drop across it, so the coordinate is strictly increasing within every row; with no knot raised,
//           it is the previous gp4's up to the constant above. Across a row's end, rounding can lift floor(S F) past S where
//           S >= 2^53, and the stored keys' coordinates can span 2^64 or more: AdaptiveLineIndex's build checks the stored keys
//           stay strictly increasing and refuses the rule if not
// The coordinate is computed only for keys from the first stored key to the last: AdaptiveLineIndex's get() sends keys outside the
// raw range of the bulk to its exceptions first, and its scan asks lower_slot only for keys above the first slot's key and at most
// the last's (hashline.h's scan_sorted_slots), which are the first and the last stored keys of the bulk.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#if defined(__x86_64__) && defined(__LZCNT__) && defined(__BMI2__)
#include <immintrin.h>
#endif

#include "gret/adaptiveline.h"

namespace gret {

class FastPeriodRule {
  __extension__ typedef unsigned __int128 u128;

 public:
  static constexpr bool kIdentity = false;
  using Blocks = std::vector<std::pair<uint64_t, uint64_t>>;  // {first key, period}, one block each, as a rule file lists them

  // A rule from memory: while a Given lives, load() on its thread takes this S and these blocks in place of $RULE_FILE and $RULE_S.
  // AdaptiveLineIndex makes its coordinate itself, so this is how a caller hands it a rule it found (gatedg.h's discovery).
  class Given {
   public:
    Given(uint64_t S, const Blocks& blocks) : S_(S), blocks_(&blocks), prev_(top()) { top() = this; }
    ~Given() { top() = prev_; }
    Given(const Given&) = delete;
    Given& operator=(const Given&) = delete;

   private:
    friend class FastPeriodRule;
    static const Given*& top() {
      static thread_local const Given* g = nullptr;
      return g;
    }
    uint64_t S_;
    const Blocks* blocks_;
    const Given* prev_;
  };

  FastPeriodRule() = default;
  FastPeriodRule(const FastPeriodRule& o) { *this = o; }
  FastPeriodRule(FastPeriodRule&&) = default;  // the vectors keep their buffers, so tb_ stays valid
  FastPeriodRule& operator=(FastPeriodRule&&) = default;
  FastPeriodRule& operator=(const FastPeriodRule& o) {  // a copy has its own table: tb_ is rebased onto it
    if (this == &o) return *this;
    blk_ = o.blk_, tab_ = o.tab_, nb_ = o.nb_, shr_ = o.shr_, b0_ = o.b0_, direct_ = o.direct_;
    std::copy(o.su_, o.su_ + kOct, su_), std::copy(o.sl_, o.sl_ + kOct, sl_);
    tb_ = tab_.empty() ? 0 : uintptr_t(tab_.data()) - b0_ * sizeof(uint16_t);
    return *this;
  }

  template <class Pair>
  void load(const Pair* d, size_t m) {
    if (m == 0) throw std::invalid_argument("FastPeriodRule: no keys");
    blk_.clear(), tab_.clear(), nb_ = 0, shr_ = 0, b0_ = 0, tb_ = 0, direct_ = false;
    std::fill(su_, su_ + kOct, 0.0), std::fill(sl_, sl_ + kOct, 0.0);
    std::vector<uint64_t> first, period;
    uint64_t S = 0;
    if (const Given* g = Given::top()) {
      S = g->S_;
      for (const auto& b : *g->blocks_) first.push_back(b.first), period.push_back(b.second);
    } else if (const char* path = std::getenv("RULE_FILE"); path && *path) {
      S = read_rule_file(path, first, period);
      if (const char* e = std::getenv("RULE_S"); e && *e) {
        const double s = std::strtod(e, nullptr);
        if (!(s >= 0 && s < 18446744073709551616.0)) throw std::runtime_error("FastPeriodRule: RULE_S must be in [0, 2^64)");
        S = uint64_t(s);
      }
    } else {
      first.assign(1, 0), period.assign(1, uint64_t(1) << 63);  // the identity block: y = x - lo once rows start at lo's
    }
    const uint64_t lo = uint64_t(d[0].first), top = uint64_t(d[m - 1].first);
    if (first.empty()) throw std::runtime_error("FastPeriodRule: the rule has no blocks");
    for (size_t c = 0; c < first.size(); ++c) {
      if (period[c] == 0) throw std::runtime_error("FastPeriodRule: block " + std::to_string(c) + " has period 0");
      if (c && first[c] <= first[c - 1]) throw std::runtime_error("FastPeriodRule: first keys not strictly increasing at block " + std::to_string(c));
    }
    if (first[0] > lo) throw std::runtime_error("FastPeriodRule: the rule does not cover the first key");
    // S F and the width S + P stay below 2^64: no float-to-integer overflow, no wrapped width
    if (S >> 63) throw std::runtime_error("FastPeriodRule: S must be below 2^63");
    for (size_t c = 0; c < first.size(); ++c)
      if (period[c] > ~uint64_t(0) - S) throw std::runtime_error("FastPeriodRule: S + period overflows at block " + std::to_string(c));

    // the blocks over [lo, top], with ruleline.h's last keys and reciprocals. Rows are counted from the row of the first stored
    // key, not from the rule's first block, so the rows below the stored keys neither widen the first block's xp nor carry y
    // past 2^64 (and load() ends by making y(lo) = 0)
    const size_t c0 = slow_block(first, lo), c1 = slow_block(first, top);
    nb_ = c1 - c0 + 1;
    if (8 * (nb_ - 1) > 0xFFFF) throw std::runtime_error("FastPeriodRule: more than 8192 blocks over the stored keys");
    uint64_t base = 0;
    bool exact = true;
    for (size_t c = c0; c <= c1; ++c) {
      const uint64_t P = period[c], last = c < c1 ? first[c + 1] - 1 : ~uint64_t(0);
      const uint64_t f = c == c0 ? first[c] + (lo - first[c]) / P * P : first[c];  // block c0 from the row of lo
      uint64_t inv = ~uint64_t(0) / P;  // rounded down: q at most one low
      bool up_ok = false;
      if (P > 1) {
        const uint64_t up = inv + 1, e = up * P;  // ceil(2^64 / P), and (mod 2^64) ceil(2^64 / P) P - 2^64
        const uint64_t xmax = std::min(last, top) - f;  // the widest xp a lookup brings to this block
        up_ok = !((u128(xmax) * e) >> 64);  // then floor(xp up / 2^64) = floor(xp / P) for every xp <= xmax
        if (up_ok) inv = up;
      }
      exact = exact && up_ok;
      blk_.push_back(Blk{f, last, P, inv, base, S + P, last, 0});
      if (c < c1) base += ((first[c + 1] - f - 1) / P + 1) * (S + P);  // rows of block c (the row count does not wrap)
    }
    // with a reciprocal rounded down anywhere, every lookup takes general(): one predictable branch, never a data-dependent one
    if (!exact)
      for (Blk& b : blk_) b.fast_to = 0;  // x = 0 can still pass, in a block that starts at 0, where xp = 0 is exact anyway
    direct_ = nb_ == 1 && exact;

    // F at the octave knots, from the same histogram as ruleline.h, multiplied by S once here
    if (S) {
      std::vector<uint64_t> h(65, 0);
      for (size_t r = 0; r < m; ++r) {
        const uint64_t x = uint64_t(d[r].first);
        const size_t c = slow_block(first, x);
        const uint64_t rr = (x - first[c]) % period[c];
        ++h[size_t(rr ? 64 - __builtin_clzll(rr) : 0)];
      }
      double u[66] = {}, su[65], sl[65];
      uint64_t cum = 0;
      for (int k = 0; k <= 64; ++k) cum += h[size_t(k)], u[k + 1] = double(cum) / double(m);
      u[0] = u[1];
      for (int j = 0; j <= 64; ++j) {
        const double den = j > 1 ? std::ldexp(1.0, j - 1) : 1.0;
        su[j] = double(S) * u[j];
        sl[j] = double(S) * (u[j + 1] - u[j]) / den;
      }
      // S F, rounded, can drop by one across a knot where the octave below ends (fastrule's tables are multiplied by S before the
      // multiply-add, ruleline.h's after). Where floor(S F) would drop, raise the knot to where the octave below ends, as a lookup
      // computes it with or without a fused multiply-add (at the knot, dd = 1, both agree): then y(2^j) > y(2^j - 1) in every row,
      // for any S. Nowhere else: with no knot raised, the coordinate is the previous gp4's up to a constant
      for (int j = 0; j < 64; ++j) {
        const double den = j > 1 ? std::ldexp(1.0, j - 1) : 1.0;  // r - 2^(j - 1) + 1 at octave j's last r
        volatile double prod = sl[j] * den;                       // not fused
        const double end = std::max(std::fma(sl[j], den, su[j]), su[j] + prod);
        if (std::floor(end) > std::floor(su[j + 1] + sl[j + 1])) su[j + 1] = end;
      }
      for (int k = 0; k <= 64; ++k) su_[k ? k - 1 : 127] = su[k], sl_[k ? k - 1 : 127] = sl[k];
    }

    // the block table over [lo, top], buckets aligned on multiples of 2^shr_: 2^12 of them at first, finer while a bucket would
    // hold two block boundaries or the stored keys would walk more than one step per 32 keys (and 64 for any set) past their
    // bucket's block in general(), up to 2^16 buckets; one or two buckets for a single block that is not direct
    if (!direct_) {
      unsigned sb = 0;
      for (uint64_t v = top - lo; v; v >>= 1) ++sb;
      const size_t limit = m / 32 + 64;
      size_t steps = 0;
      for (unsigned bits = 12;; ++bits) {
        shr_ = nb_ == 1 ? 63 : sb > bits ? sb - bits : 0;
        b0_ = lo >> shr_;
        const uint64_t buckets = (top >> shr_) - b0_ + 1;  // at most 2^bits + 1
        tab_.assign(size_t(buckets), 0);
        bool ok = true;
        for (uint64_t j = 0; j < buckets; ++j) {
          const uint64_t b = b0_ + j, at = std::max(b << shr_, lo), end = std::min((b << shr_) | ((uint64_t(1) << shr_) - 1), top);
          const size_t ca = slow_block(first, at) - c0;
          tab_[size_t(j)] = uint16_t(8 * ca);  // the block's offset in 8-byte words
          ok = ok && slow_block(first, end) - c0 <= ca + 1;
        }
        steps = nb_ > 1 && (ok || bits >= 16) ? walk_steps(d, m, limit) : 0;  // counted only where they decide
        if ((ok && steps <= limit) || bits >= 16) break;
      }
      tb_ = uintptr_t(tab_.data()) - b0_ * sizeof(uint16_t);
      if (steps > limit)  // a cost the gate's prediction (lines and levels) would not see
        throw std::runtime_error("FastPeriodRule: blocks closer than its table separates: more than " + std::to_string(limit) +
                                 " walk steps over " + std::to_string(m) + " keys");
    }
    // y(lo) = 0: AdaptiveLineIndex's raw order check on y then sees exactly y - y(lo), which is all the rest of it uses
    const uint64_t y0 = (*this)(lo);
    for (Blk& b : blk_) b.base -= y0;
  }

  // always inlined: the compiler's own heuristic left it a call inside get() once general() was inlined into it
  [[gnu::always_inline]] uint64_t operator()(uint64_t x) const {
    size_t e = 0;  // the block, in 8-byte words from the first
    if (!direct_) {
      e = *reinterpret_cast<const uint16_t*>(tb_ + (x >> shr_) * sizeof(uint16_t));
      // Unlikely, but not so unlikely (1 in 20) that clang outlines general() into a call, which costs every get() 3 registers.
      // blk_.data() + e / 8 is at(e), written so that no value is shared with the test, which keeps its base + index operand
      if (__builtin_expect_with_probability(x > at(e)->fast_to, 0, 0.95)) return general(x, blk_.data() + e / 8);
    }
    const Blk* k = at(e);
    const uint64_t xp = x - k->first, q = uint64_t((u128(xp) * k->inv) >> 64);  // exact: the reciprocal is rounded up
    return finish(k, q, xp - q * k->period);
  }
  size_t bytes() const { return blk_.size() * sizeof(Blk) + tab_.size() * sizeof(uint16_t) + sizeof(su_) + sizeof(sl_); }
  size_t blocks() const { return nb_; }                                  // over the stored keys
  size_t table_entries() const { return tab_.size(); }                   // 0 for one block
  size_t exact_blocks() const {                                          // blocks whose reciprocal is rounded up
    size_t e = 0;
    for (const Blk& b : blk_) e += b.period > 1 && b.inv != ~uint64_t(0) / b.period;
    return e;
  }
  bool direct() const { return direct_; }                                // one block, rounded up: no table, no test

 private:
  static constexpr int kOct = 128;  // indexed by lzcnt(r) ^ 63: 0..63 for r >= 1, 127 for r = 0
  struct alignas(64) Blk {          // one cache line each
    uint64_t first, last, period, inv, base, width;  // width = S + period; last: 2^64 - 1 for the last block
    uint64_t fast_to;  // the last key operator() computes here without general(): last, or 0 when the rule has a rounded-down reciprocal
    uint64_t pad;
  };
  const Blk* at(size_t e) const { return reinterpret_cast<const Blk*>(reinterpret_cast<const char*>(blk_.data()) + e * 8); }
  // the rest of the coordinate, from q and r
  [[gnu::always_inline]] uint64_t finish(const Blk* k, uint64_t q, uint64_t r) const {
#if defined(__x86_64__) && defined(__LZCNT__) && defined(__BMI2__)
    const unsigned i = unsigned(_lzcnt_u64(r)) ^ 63u;
    const uint64_t dd = _bzhi_u64(r, i) + 1;  // r - 2^i + 1: r's offset into its octave, plus one (1 for r = 0)
#else
    const unsigned i = (r ? unsigned(__builtin_clzll(r)) : 64u) ^ 63u;
    const uint64_t dd = (i < 64 ? r & ((uint64_t(1) << i) - 1) : r) + 1;
#endif
    return k->base + q * k->width + uint64_t(su_[i] + sl_[i] * double(int64_t(dd))) + r;  // dd < 2^63, as r < P < 2^64
  }
  // a key past its bucket's block, or any key under a rule with a rounded-down reciprocal: the rare path, or the only one. Inlined:
  // a call here would make every get() save and restore three more registers
  [[gnu::always_inline]] uint64_t general(uint64_t x, const Blk* k) const {
    if (__builtin_expect(x > k->last, 0)) do ++k; while (x > k->last);  // one step, but where 2^16 buckets still hold two boundaries
    const uint64_t xp = x - k->first, pd = k->period;
    uint64_t q = uint64_t((u128(xp) * k->inv) >> 64), r = xp - q * pd;
    if (__builtin_expect(r >= pd, 0)) ++q, r -= pd;  // a reciprocal rounded down was one short
    return finish(k, q, r);
  }
  static uint64_t read_rule_file(const char* path, std::vector<uint64_t>& first, std::vector<uint64_t>& period) {
    FILE* f = std::fopen(path, "r");
    if (!f) throw std::runtime_error(std::string("FastPeriodRule: cannot read RULE_FILE ") + path);
    unsigned long long a, p;
    uint64_t S = uint64_t(67108864);
    if (std::fscanf(f, "S %llu", &a) == 1) S = a;
    int got;
    while ((got = std::fscanf(f, "%llu %llu", &a, &p)) == 2) first.push_back(a), period.push_back(p);
    std::fclose(f);
    if (got != EOF) throw std::runtime_error(std::string("FastPeriodRule: RULE_FILE ") + path + " has a line that is not '<first key> <period>'");
    return S;
  }
  // the steps general() walks past the stored keys' bucket blocks, counted until they pass limit
  template <class Pair>
  size_t walk_steps(const Pair* d, size_t m, size_t limit) const {
    size_t steps = 0;
    for (size_t r = 0; r < m && steps <= limit; ++r) {
      const uint64_t x = uint64_t(d[r].first);
      for (size_t e = tab_[size_t((x >> shr_) - b0_)] / 8; x > blk_[e].last; ++e) ++steps;
    }
    return steps;
  }
  static size_t slow_block(const std::vector<uint64_t>& first, uint64_t x) {  // the last block whose first key is <= x, at build time
    const auto it = std::upper_bound(first.begin(), first.end(), x);
    return it == first.begin() ? 0 : size_t(it - first.begin()) - 1;
  }

  std::vector<Blk> blk_;
  uintptr_t tb_ = 0;  // the table's address less b0_ entries: the entry of bucket x >> shr_ is at tb_ + 2 (x >> shr_)
  unsigned shr_ = 0;
  bool direct_ = false;  // one block, reciprocal rounded up: no table, no test
  size_t nb_ = 0;
  uint64_t b0_ = 0;
  std::vector<uint16_t> tab_;
  double su_[kOct] = {}, sl_[kOct] = {};
};

template <class K, class P, unsigned B>
using FastRuleLineIndex = AdaptiveLineIndex<K, P, B, FastPeriodRule>;

}  // namespace gret
