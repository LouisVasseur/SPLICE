// SPDX-License-Identifier: MIT
// The plain searches of a SOSD-style table, in GRE's harness: what any index over a sorted array has to beat. One sorted array of
// (key, payload) pairs, as readonly.h; no model, or a table that is not learned.
//
//   name            what                                                        whose code
//   bsearch         binary search over the whole array (the floor)              ours: a Model that bounds nothing, searched by
//                                                                               SortedBoundIndex's branch-free binary search
//   bsearchstd      the same with std::lower_bound, as SOSD's "BS" runs         ours (std::lower_bound)
//   rbs<bits>       radix binary search: a table over the top <bits> bits of    SOSD's own class (the benchmark that introduced RBS),
//                   key - min gives the range, binary search inside             competitors/radix_binary_search.h, unmodified, by path
//   rbsstd<bits>    the same class and bound, searched with std::lower_bound:   SOSD's own class; std::lower_bound is what SOSD's default
//                   RBS as SOSD's own benchmark runs it, end to end             last-mile search calls (searches/branching_binary_search.h)
//   rbst<bits>      the same table, the bucket itself as the bound (SOSD        a reimplementation (ours), says so in its name line
//                   widens it by one position on each side)
//   interp[<k>][e]  interpolation search: at most k steps (default 8), then     a reimplementation (ours) of the textbook algorithm
//                   a binary search of what is left of the bracket; 'e': an     (Algorithm 1 of the TIP paper), says so in its name line
//                   exponential search from the last probe, then binary search
//   tip[<g>][a]     three-point interpolation search (Van Sandt, Chronis,       a reimplementation (ours) from the paper, says so in
//                   Patel, SIGMOD 2019), guard g (default 64); 'a': the first   its name line
//                   estimate as the authors' code computes it
//
// bits: 8 to 28 for all three rbs names (SOSD's sweep is the even ones). k: 1 to 64. g: 1 to 4096. docs/sota/searches.md says how each was built, why TIP and
// interpolation search are not the borrowed code, and every deviation.
//
// bsearch, rbs and rbst are Models for SortedBoundIndex (readonly.h). interp and tip are searches, not bounds: a Model only sees a copy
// of the keys, so running them there would cost a second array and a second cache miss per lookup. They run in SortedSearchIndex below,
// which is SortedBoundIndex with the search handed the pair array itself: the same layout, range check, scan and memory count.
// rbsstd runs there too, because SortedBoundIndex's search inside a bound is fixed (branch-free) and SOSD's is std::lower_bound: with
// SOSD's widened bound the two differ in speed (docs/sota/searches.md has the measurement), and the null hypothesis must not lose on it.
//
// What each index is, it says when it is built: every name but rbs<bits> and rbsstd<bits> is reported as "<name>(reimplementation)",
// and those two print a line of their own that names SOSD's class.
//
// The build needs SOSD (github.com/learnedsystems/SOSD, GPL-3.0; nothing of it is copied here): either the directory that holds the
// clone on the include path (-I <dir>, then "SOSD/competitors/..." is used; the registry's build already has it, for PLEX), or the
// clone itself, after everything else (-isystem <SOSD>). SOSD's headers define KeyValue, SearchBound, Competitor, util:: and more at
// global scope, and DILI's global.h defines the same SearchBound; they are therefore included inside namespace gret::sota::sosd,
// after every standard header they pull in. A unit that also includes SOSD's util.h or base.h at global scope must do so BEFORE this
// header (they are #pragma once).
//
// Include after indexInterface.h and GRE's pgm header, like readonly.h (which this includes).
#pragma once
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gret/readonly.h"

namespace gret {
namespace sota {
namespace sosd {
// SOSD's files, as they are. Every standard header they include is already included above, so only SOSD's own names land here.
#if defined(__has_include) && __has_include("SOSD/competitors/radix_binary_search.h")
#include "SOSD/competitors/radix_binary_search.h"  // -I <the directory that holds SOSD> (the one PLEX is already found through)
#else
#include "competitors/radix_binary_search.h"  // -isystem <SOSD>
#endif

// SOSD's RadixBinarySearch as a Model for SortedBoundIndex. Read from the class and pinned by tests/sota/test_searches.cpp:
//   - the table has 2^Bits + 1 uint32_t entries whatever the keys are; entry j is the first position whose prefix
//     (key - min) >> shift is >= j, and shift keeps the top Bits bits of max - min (0 when max - min has fewer bits);
//   - EqualityLookup returns [table[p] - 1, table[p + 1] + 1), cut at 0 and n: the bucket, one position wider on each side;
//     below min it returns [0, 1), above max [n - 1, n);
//   - size() is 4 bytes per entry and nothing else;
//   - __builtin_clzl(max - min) is undefined for max == min, and positions are stored as uint32_t: build() throws for both.
// Build takes a vector of SOSD's 16-byte KeyValue, which is made here from the keys and dropped after (the key copy is freed first).
template <class K, uint32_t Bits>
struct RbsBound {
  static_assert(std::is_same<K, uint64_t>::value || std::is_same<K, uint32_t>::value, "SOSD's RadixBinarySearch takes uint32_t or uint64_t keys");
  RadixBinarySearch<K, Bits> m;

  void build(std::vector<K>& keys) {
    if (keys.size() < 2 || keys.front() == keys.back()) throw std::invalid_argument("RbsBound: needs at least two distinct keys");
    if (keys.size() > size_t(std::numeric_limits<uint32_t>::max())) throw std::length_error("RbsBound: 2^32 keys or more");
    std::vector<KeyValue<K>> data(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) data[i].key = keys[i];
    std::vector<K>().swap(keys);  // SortedBoundIndex's scratch copy, not read again
    m.Build(data);
    std::fprintf(stderr, "gret: rbs%u is SOSD's own RadixBinarySearch (learnedsystems/SOSD competitors/radix_binary_search.h, unmodified, by path): table_entries=%zu table_bytes=%zu (SOSD's size())\n",
                 unsigned(Bits), m.size() / sizeof(uint32_t), size_t(m.size()));
  }
  KeyBound get(K x) const {
    const auto b = m.EqualityLookup(x);
    return {b.start, b.stop};
  }
  size_t bytes() const { return sizeof(*this) + m.size(); }  // this object and SOSD's count of its table
  static constexpr uint32_t bits() { return Bits; }
};

// SOSD's RadixBinarySearch as SOSD's own benchmark runs it: the class's bound, then std::lower_bound inside [start, stop), which is
// what SOSD's default last-mile search (searches/branching_binary_search.h, "binary") calls on its row array. A Search for
// SortedSearchIndex (below), over the index's pair array. Exact for any key in [min, max]: every position before start holds a
// smaller prefix, and the lower bound is at most table[p + 1] <= stop. The same throws as RbsBound; the 16-byte records Build takes
// are made from the pairs and dropped after (no 8-byte key copy here, unlike under SortedBoundIndex).
template <class K, uint32_t Bits>
struct RbsStdSearch {
  static_assert(std::is_same<K, uint64_t>::value || std::is_same<K, uint32_t>::value, "SOSD's RadixBinarySearch takes uint32_t or uint64_t keys");
  RadixBinarySearch<K, Bits> m{};  // value-initialized: the index copies the search before it is built

  explicit RbsStdSearch(unsigned = 0) {}
  template <class Pair>
  void build(const Pair* v, size_t n) {
    if (n < 2 || v[0].first == v[n - 1].first) throw std::invalid_argument("RbsStdSearch: needs at least two distinct keys");
    if (n > size_t(std::numeric_limits<uint32_t>::max())) throw std::length_error("RbsStdSearch: 2^32 keys or more");
    std::vector<KeyValue<K>> data(n);
    for (size_t i = 0; i < n; ++i) data[i].key = K(v[i].first);
    m.Build(data);
    std::fprintf(stderr, "gret: rbsstd%u is SOSD's own RadixBinarySearch (learnedsystems/SOSD competitors/radix_binary_search.h, unmodified, by path), searched with std::lower_bound inside its bound as SOSD's benchmark does: table_entries=%zu table_bytes=%zu (SOSD's size())\n",
                 unsigned(Bits), m.size() / sizeof(uint32_t), size_t(m.size()));
  }
  template <class Pair>
  size_t lower(const Pair* v, size_t, uint64_t x) const {
    const auto b = m.EqualityLookup(K(x));
    return size_t(std::lower_bound(v + b.start, v + b.stop, x, [](const Pair& p, uint64_t k) { return uint64_t(p.first) < k; }) - v);
  }
  size_t bytes() const { return sizeof(*this) + m.size(); }  // this object and SOSD's count of its table
  template <class Pair>
  void report(const char*, const Pair*, size_t) const {}
  static constexpr uint32_t bits() { return Bits; }
};
}  // namespace sosd

using sosd::RbsBound;
using sosd::RbsStdSearch;

inline constexpr unsigned kRbsMinBits = 8, kRbsMaxBits = 28;            // SOSD's benchmark_rbs.cc runs 8, 10, ..., 28
inline constexpr unsigned kInterpDefaultSteps = 8, kInterpMaxSteps = 64;
inline constexpr unsigned kTipDefaultGuard = 64, kTipMaxGuard = 4096;   // 64: tip<record_bytes, 64> in the authors' benchmark.h, GUARD in SOSD
inline constexpr unsigned kTipMaxIterations = 16;                       // three-point interpolations before the bracket is binary searched
inline constexpr int kTipUnroll = 8;                                    // the sequential search compares in blocks of 8, as the authors' LinearUnroll

namespace search_detail {

// the first position in [lo, hi] whose key is >= x, when there is one in [lo, hi]. Branch-free, as SortedBoundIndex::lower.
template <class Pair>
inline size_t lower_in(const Pair* v, uint64_t x, size_t lo, size_t hi) {
  const Pair* base = v + lo;
  size_t len = hi - lo;
  if (len == 0) return lo;
  while (len > 1) {
    const size_t half = len / 2;
    base += uint64_t(base[half - 1].first) < x ? half : 0;
    len -= half;
  }
  return size_t(base - v) + size_t(uint64_t(base->first) < x);
}

struct NoTrace {
  void step() {}
  void scan(size_t) {}
  void fallback() {}
  void finish(size_t, size_t) {}
};
struct CountTrace {
  uint64_t steps = 0, scanned = 0, fallbacks = 0, finishes = 0, finish_probes = 0, finish_range = 0;
  void step() { ++steps; }
  void scan(size_t records) { scanned += records; }
  void fallback() { ++fallbacks; }
  void finish(size_t probes, size_t range) { ++finishes, finish_probes += probes, finish_range += range; }
};

// up to 1000 stored keys, evenly spaced: each must be found where it is (throws otherwise), and what the search did is counted
template <class Search, class Pair>
CountTrace sample(const Search& s, const Pair* v, size_t n, size_t& taken) {
  CountTrace t;
  taken = std::min<size_t>(n, 1000);
  for (size_t j = 0; j < taken; ++j) {
    const size_t i = taken == 1 ? 0 : size_t((unsigned __int128)(j) * (n - 1) / (taken - 1));
    if (s.lower_t(v, n, uint64_t(v[i].first), t) != i) throw std::logic_error("gret::sota searches: a stored key was not found at its position");
  }
  return t;
}

}  // namespace search_detail

// ---- bsearch: no model. SOSD's BinarySearch returns the same {0, n} (its Build only records the size).
template <class K>
struct WholeArrayBound {
  size_t n = 0;
  void build(std::vector<K>& keys) { n = keys.size(); }
  KeyBound get(K) const { return {0, n}; }
  size_t bytes() const { return sizeof(*this); }
};

// ---- rbst<bits>: SOSD's table (same entries, same shift, same 4 bytes per entry), built from the keys in one pass with no copy, and
// the bucket [table[p], table[p + 1]) itself as the bound. A reimplementation. For a key that was never stored the lower bound is in
// [lo, hi], which SortedBoundIndex's search can return. One compare sends anything outside [min, max] to its exact lower bound.
template <class K, uint32_t Bits>
struct RbstBound {
  static_assert(std::is_same<K, uint64_t>::value || std::is_same<K, uint32_t>::value, "RbstBound takes uint32_t or uint64_t keys");
  K min_ = 0;
  uint64_t span_ = 0;
  unsigned shift_ = 0;
  size_t n_ = 0;
  std::vector<uint32_t> table_;

  void build(std::vector<K>& keys) {
    if (keys.size() < 2 || keys.front() == keys.back()) throw std::invalid_argument("RbstBound: needs at least two distinct keys");
    if (keys.size() > size_t(std::numeric_limits<uint32_t>::max())) throw std::length_error("RbstBound: 2^32 keys or more");
    n_ = keys.size(), min_ = keys.front(), span_ = uint64_t(keys.back()) - uint64_t(keys.front());
    const unsigned len = 64 - unsigned(__builtin_clzll(span_));  // bit length of max - min
    shift_ = len < Bits ? 0 : len - Bits;                        // SOSD's shift_bits
    const size_t buckets = size_t(1) << Bits;
    table_.assign(buckets + 1, 0);
    size_t prev = 0;
    for (size_t i = 0; i < n_; ++i) {
      const size_t cur = size_t((uint64_t(keys[i]) - uint64_t(min_)) >> shift_);
      while (prev < cur) table_[++prev] = uint32_t(i);
    }
    while (prev < buckets) table_[++prev] = uint32_t(n_);
  }
  KeyBound get(K x) const {
    const uint64_t off = uint64_t(x) - uint64_t(min_);
    if (__builtin_expect(off > span_, 0)) return x < min_ ? KeyBound{0, 0} : KeyBound{n_, n_};
    const size_t p = size_t(off >> shift_);
    return {table_[p], table_[p + 1]};
  }
  size_t bytes() const { return sizeof(*this) + table_.capacity() * sizeof(uint32_t); }
  static constexpr uint32_t bits() { return Bits; }
};

// ---- the searches that are not bounds. Each has build(pairs, n) (n >= 64, keys strictly ascending), lower(pairs, n, x) for
// pairs[0] <= x <= pairs[n - 1] (the first position whose key is >= x), bytes(), and report(name, pairs, n).

// bsearchstd: std::lower_bound over the whole array, which is what SOSD's BinarySearch runs with its default search ("binary",
// searches/branching_binary_search.h). Compilers turn it into a branching or a conditional-move loop as they see fit.
template <class K>
struct StdLowerSearch {
  explicit StdLowerSearch(unsigned = 0) {}
  template <class Pair>
  void build(const Pair*, size_t) {}
  template <class Pair>
  size_t lower(const Pair* v, size_t n, uint64_t x) const {
    return size_t(std::lower_bound(v, v + n, x, [](const Pair& p, uint64_t k) { return uint64_t(p.first) < k; }) - v);
  }
  size_t bytes() const { return 0; }
  template <class Pair>
  void report(const char*, const Pair*, size_t) const {}
};

// interp<k>: interpolation search. A reimplementation of the textbook algorithm as Van Sandt et al. print it (Algorithm 1): the
// estimate is left + (x - V[left]) * (right - left) / (V[right] - V[left]), the side of the bracket it falls on moves next to it, until
// the key is hit or the bracket shows it is absent. That loop is O(n) on skewed keys, so it stops after max_steps interpolations and
// what is left of the bracket is binary searched (O(log n) in all), as SOSD's last-mile searches/interpolation_search.h ends. With
// Gallop the rest is instead an exponential search from the end of the bracket that moved last, then a binary search: fewer probes
// when the interpolations were closing in on the key from one side (a smooth, bent CDF), up to twice as many when they were not.
// Keys must be unique (V[right] > V[left] whenever right > left).
template <class K, bool Gallop = false>
struct InterpSearch {
  unsigned max_steps = kInterpDefaultSteps;
  explicit InterpSearch(unsigned steps = kInterpDefaultSteps) : max_steps(steps) {
    if (steps < 1 || steps > kInterpMaxSteps) throw std::invalid_argument("InterpSearch: steps must be in [1, 64]");
  }
  template <class Pair>
  void build(const Pair*, size_t) {}

  template <class Pair, class Trace>
  size_t lower_t(const Pair* v, size_t n, uint64_t x, Trace& t) const {
    size_t lo = 0, hi = n - 1;  // V[lo] <= x <= V[hi] throughout
    bool up = true;
    for (unsigned s = 0; s < max_steps; ++s) {
      if (lo == hi) return lo;
      const uint64_t kl = uint64_t(v[lo].first), kh = uint64_t(v[hi].first);
      // the quotient is at most 1 and hi - lo is exact in a double, so the estimate is in [lo, hi]
      const size_t e = lo + size_t(double(x - kl) / double(kh - kl) * double(hi - lo));
      const uint64_t ke = uint64_t(v[e].first);
      t.step();
      if (ke < x) {
        lo = e + 1, up = true;  // e < hi, because V[e] < x <= V[hi]
        if (uint64_t(v[lo].first) > x) return lo;  // absent: between e and e + 1
      } else if (ke > x) {
        hi = e - 1, up = false;  // e > lo
        if (uint64_t(v[hi].first) < x) return e;  // absent: between e - 1 and e
      } else {
        return e;
      }
    }
    if (!Gallop) {
      t.finish(0, hi - lo);
      return search_detail::lower_in(v, x, lo, hi);
    }
    size_t probes = 0, w = 1;
    if (up) {  // the last estimate was below: double the step upwards from lo until a key >= x, or hi
      size_t a = lo;
      while (hi - a > w && uint64_t(v[a + w].first) < x) a += w, w <<= 1, ++probes;
      const size_t b = hi - a > w ? a + w : hi;
      t.finish(probes + 1, b - a);
      return search_detail::lower_in(v, x, a, b);
    }
    size_t b = hi;  // the last estimate was above: the same downwards from hi until a key <= x, or lo
    while (b - lo > w && uint64_t(v[b - w].first) > x) b -= w, w <<= 1, ++probes;
    const size_t a = b - lo > w ? b - w : lo;
    t.finish(probes + 1, b - a);
    return search_detail::lower_in(v, x, a, b);
  }
  template <class Pair>
  size_t lower(const Pair* v, size_t n, uint64_t x) const {
    search_detail::NoTrace t;
    return lower_t(v, n, x, t);
  }
  size_t bytes() const { return sizeof(*this); }
  template <class Pair>
  void report(const char* name, const Pair* v, size_t n) const {
    size_t taken = 0;
    const search_detail::CountTrace t = search_detail::sample(*this, v, n, taken);
    std::fprintf(stderr, "gret: %s sample=%zu stored keys: interpolations avg=%.2f, hit the bound of %u steps=%.1f%%, then exponential probes avg=%.1f and a binary search over avg=%.0f positions\n",
                 name, taken, double(t.steps) / double(taken), max_steps, 100.0 * double(t.finishes) / double(taken),
                 t.finishes ? double(t.finish_probes) / double(t.finishes) : 0.0, t.finishes ? double(t.finish_range) / double(t.finishes) : 0.0);
  }
};

// tip<g>: Three point Interpolation search (Van Sandt, Chronis, Patel, SIGMOD 2019, Algorithm 3). A reimplementation from the paper,
// read next to the authors' code (UWHustle/Efficiently-Searching-In-Memory-Sorted-Arrays, src/algorithms/tip.h) and SOSD's
// (competitors/rm_tip.h), neither of which can answer GRE's interface (docs/sota/searches.md says why).
//   line 1-4   left = 0, right = n - 1, mid = n / 2, expected = the three-point estimate through V[0], V[n / 2], V[n - 1] (Equation 8)
//   line 6-7   expected within guard of mid: sequential search from expected
//   line 9-12  the bracket closes on mid
//   line 13-16 expected within guard of an end of the bracket: sequential search from that end
//   line 17-18 mid = expected, expected = Jarratt and Nudds' three-point estimate through left, mid, right (Equation 2)
// What is ours, none of it changing a probe when the method works as the paper describes:
//   - keys are uint64_t over the whole domain; differences are taken as integers first and every estimate is cut to the bracket
//     before it becomes an index (the borrowed codes convert an unchecked double);
//   - the bracket closes on mid by comparing V[mid] with the key (the paper's text; its pseudocode and the authors' code compare
//     the positions mid and expected, which is the same decision whenever the estimate is on the right side of mid);
//   - the sequential search is the authors' unrolled one, in blocks of 8, but after 8 * max(guard, 8) records, or after
//     kTipMaxIterations estimates, the rest of the bracket is binary searched: the paper's search is O(n) there;
//   - no sentinel records around the array: the index checks the key against the first and last key first, as it does for every
//     read-only index, and then the sequential search cannot leave the bracket.
// AuthorsFirst: the first estimate as the authors' code and SOSD's compute it, d + d_a * (y_1 - x) / (diff_scale - x * (a_0 + 1)).
// The three-point formula (Equation 2, the one both codes use in their loop) through V[0], V[d], V[2d] reduces to that expression
// with 1 - a_0 where the code has a_0 + 1 (docs/sota/searches.md has the algebra; the test pins it), and so does the paper's Equation 8
// as printed, y_1 d (1 + a_0) / (y_0 - a_0 y_2): as coded the estimate passes through V[d] only, and for keys far from 0 it is about
// d whatever the key. The guard tests are <=, as in both codes (the paper's line 6 has <).
template <class K, bool AuthorsFirst = false>
struct TipSearch {
  unsigned guard = kTipDefaultGuard;
  unsigned scan_blocks = kTipDefaultGuard;  // the sequential search's budget, in blocks of kTipUnroll records
  int64_t d = 0;                            // n / 2
  uint64_t v0 = 0, vd = 0, vlast = 0;       // V[0], V[d], V[n - 1]
  double a = 0, A = 0;                      // Equation 8: (V[d] - V[0]) / (V[n - 1] - V[d]) and d * (1 + a)
  double au_y1 = 0, au_a0 = 0, au_scale = 0, au_da = 0;  // the authors' constants (y_1, a_0, diff_scale, d_a)
  double qlo = 0, qhi = 0;                  // what the first estimate may move from d

  explicit TipSearch(unsigned g = kTipDefaultGuard) : guard(g), scan_blocks(std::max<unsigned>(g, 8)) {
    if (g < 1 || g > kTipMaxGuard) throw std::invalid_argument("TipSearch: guard must be in [1, 4096]");
  }
  template <class Pair>
  void build(const Pair* v, size_t n) {
    d = int64_t(n / 2);
    v0 = uint64_t(v[0].first), vd = uint64_t(v[d].first), vlast = uint64_t(v[n - 1].first);
    a = double(vd - v0) / double(vlast - vd);
    A = double(d) * (1.0 + a);
    qlo = -double(int64_t(n) - 1 - d), qhi = double(d);
    // as the authors' constructor and SOSD's Build, in double because the keys are unsigned
    const double first = double(v0), last = double(vlast), diff_y_01 = first - double(vd);
    au_y1 = double(vd);
    au_a0 = diff_y_01 == au_y1 - last ? 0.99999999999999 : diff_y_01 / (au_y1 - last);
    au_scale = first - au_a0 * last;
    au_da = (1.0 + au_a0) * double(d);
  }

  // the first estimate, in [0, n - 1]
  int64_t first(uint64_t x) const {
    double q;
    if (AuthorsFirst) {
      const double dx = double(x);
      q = -(au_da * (au_y1 - dx) / (au_scale - dx * (au_a0 + 1.0)));
    } else {
      // Equation 8 with y_i = V[i] - x: d + y_1 * d * (1 + a) / (y_0 - a * y_2). The three differences are taken as integers, and
      // both terms of the denominator have one sign.
      const double t0 = double(x - v0), t2 = double(vlast - x), y1 = x < vd ? double(vd - x) : -double(x - vd);
      q = A * y1 / (a * t2 + t0);
    }
    q = q > qlo ? q : qlo;  // also takes a NaN
    q = q < qhi ? q : qhi;
    return d - int64_t(q);
  }
  // Equation 2, for left < mid < right and V[left] <= x <= V[right]: the denominator is a sum of two terms that are not negative
  // and not both zero. The estimate is cut to [left, right], where it is in exact arithmetic.
  template <class Pair>
  static int64_t three_point(const Pair* v, uint64_t x, int64_t x0, int64_t x1, int64_t x2) {
    const uint64_t km = uint64_t(v[x1].first);
    const double y0 = -double(x - uint64_t(v[x0].first)), y2 = double(uint64_t(v[x2].first) - x), y1 = km >= x ? double(km - x) : -double(x - km);
    const double dl = double(x1 - x0), dr = double(x1 - x2);
    double e = y1 * dr * dl * (y2 - y0) / (y2 * dr * (y0 - y1) + y0 * dl * (y1 - y2));
    const double elo = double(x0 - x1), ehi = double(x2 - x1);
    e = e > elo ? e : elo;
    e = e < ehi ? e : ehi;
    return x1 + int64_t(e);
  }
  // the first position >= s whose key is >= x; there is one at or before hi
  template <class Pair, class Trace>
  size_t forward(const Pair* v, uint64_t x, int64_t s, int64_t hi, Trace& t) const {
    for (unsigned b = 0; b < scan_blocks; ++b, s += kTipUnroll)
      for (int i = 0; i < kTipUnroll; ++i)
        if (uint64_t(v[s + i].first) >= x) {
          t.scan(size_t(b) * kTipUnroll + size_t(i) + 1);
          return size_t(s + i);
        }
    t.scan(size_t(scan_blocks) * kTipUnroll), t.fallback();
    return search_detail::lower_in(v, x, size_t(s), size_t(hi));
  }
  // V[s] >= x and there is a key <= x at or after lo: the last position whose key is <= x, turned into the lower bound of x
  template <class Pair, class Trace>
  size_t reverse(const Pair* v, uint64_t x, int64_t s, int64_t lo, Trace& t) const {
    for (unsigned b = 0; b < scan_blocks; ++b, s -= kTipUnroll)
      for (int i = 0; i < kTipUnroll; ++i)
        if (uint64_t(v[s - i].first) <= x) {
          t.scan(size_t(b) * kTipUnroll + size_t(i) + 1);
          return size_t(s - i) + size_t(uint64_t(v[s - i].first) < x);
        }
    t.scan(size_t(scan_blocks) * kTipUnroll), t.fallback();
    return search_detail::lower_in(v, x, size_t(lo), size_t(s) + 1);  // every key after s is > x
  }

  template <class Pair, class Trace>
  size_t lower_t(const Pair* v, size_t n, uint64_t x, Trace& t) const {
    int64_t left = 0, right = int64_t(n) - 1, mid = d, expected = first(x);
    const int64_t g = int64_t(guard);
    for (unsigned it = 0;; ++it) {
      // left <= expected <= right, left <= mid <= right, V[left] <= x <= V[right]
      if (expected - mid <= g && expected - mid >= -g) {
        if (uint64_t(v[expected].first) >= x) return reverse(v, x, expected, left, t);
        return forward(v, x, expected + 1, right, t);
      }
      if (uint64_t(v[mid].first) < x) left = mid;
      else right = mid;
      if (expected + g >= right) return reverse(v, x, right, left, t);
      if (expected - g <= left) return forward(v, x, left, right, t);
      if (it == kTipMaxIterations) {
        t.fallback();
        return search_detail::lower_in(v, x, size_t(left), size_t(right));
      }
      mid = expected;  // left + g < mid < right - g
      expected = three_point(v, x, left, mid, right);
      t.step();
    }
  }
  template <class Pair>
  size_t lower(const Pair* v, size_t n, uint64_t x) const {
    search_detail::NoTrace t;
    return lower_t(v, n, x, t);
  }
  size_t bytes() const { return sizeof(*this); }
  template <class Pair>
  void report(const char* name, const Pair* v, size_t n) const {
    size_t taken = 0;
    const search_detail::CountTrace t = search_detail::sample(*this, v, n, taken);
    std::fprintf(stderr, "gret: %s sample=%zu stored keys: three-point steps after the first estimate avg=%.2f, records scanned avg=%.1f, finished by binary search=%.1f%%\n",
                 name, taken, double(t.steps) / double(taken), double(t.scanned) / double(taken), 100.0 * double(t.fallbacks) / double(taken));
  }
};

// SortedBoundIndex for a search that is not a bound: one sorted array of pairs, and the search runs over it. Fewer than 64 keys are
// binary searched, the key is checked against the first and last key first, scan copies from the lower bound, writes return false:
// all as SortedBoundIndex. bulk_load throws unless the keys are strictly ascending (the interpolations divide by key differences).
template <class K, class P, class Search>
class SortedSearchIndex final : public indexInterface<K, P> {
  using Pair = std::pair<K, P>;

 public:
  SortedSearchIndex(std::string name, const Search& s) : name_(std::move(name)), s_(s) {}
  void init(Param* = nullptr) override {}
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (n == 0) throw std::invalid_argument("SortedSearchIndex: bulk_load of 0 keys");
    for (size_t i = 1; i < n; ++i)
      if (!(uint64_t(kv[i - 1].first) < uint64_t(kv[i].first))) throw std::invalid_argument("SortedSearchIndex: keys must be strictly ascending");
    a_.assign(kv, kv + n);
    small_ = n < 64;
    if (!small_) s_.build(a_.data(), n);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu dir_bytes=%zu bytes_per_key=%.2f build_ms=%.1f\n", name_.c_str(), n, small_ ? size_t(0) : s_.bytes(),
                 double(memory_consumption()) / double(n), ms);
    if (!small_) s_.report(name_.c_str(), a_.data(), n);
  }
  bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    if (__builtin_expect(x < uint64_t(a_.front().first) || x > uint64_t(a_.back().first), 0)) return false;
    const size_t i = lower(x);  // < n, because x is not above the last key
    if (uint64_t(a_[i].first) != x) return false;
    val = a_[i].second;
    return true;
  }
  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    if (x > uint64_t(a_.back().first)) return 0;
    const size_t i = x > uint64_t(a_.front().first) ? lower(x) : 0;
    const size_t got = std::min(n, a_.size() - i);
    std::copy_n(a_.data() + i, got, out);
    return got;
  }
  long long memory_consumption() override { return (long long)(a_.size() * sizeof(Pair) + (small_ ? size_t(0) : s_.bytes())); }
  const Search& search() const { return s_; }

 private:
  size_t lower(uint64_t x) const {  // for a_.front() <= x <= a_.back()
    return small_ ? search_detail::lower_in(a_.data(), x, 0, a_.size() - 1) : s_.lower(a_.data(), a_.size(), x);
  }

  std::string name_;
  bool small_ = false;
  std::vector<Pair> a_;
  Search s_;
};

// ---- names

struct SearchName {
  enum Kind { none, whole, stdlower, rbs, rbsstd, rbst, interp, tip };
  Kind kind = none;
  unsigned knob = 0;           // rbs, rbsstd, rbst: bits; interp: steps; tip: guard
  bool authors_first = false;  // tip<g>a
  bool gallop = false;         // interp<k>e
  std::string shown;           // the name the index reports: the knob spelled out, and "(reimplementation)" where the code is ours
};

// bsearch | bsearchstd | rbs<bits> | rbsstd<bits> | rbst<bits> | interp[<k>][e] | tip[<g>][a], numbers decimal without leading zeros and in range;
// anything else is not one of these names (false, out untouched).
inline bool parse_search_name(const std::string& name, SearchName& out) {
  auto number = [](const std::string& dgt, unsigned lo, unsigned hi, unsigned& v) {
    if (dgt.empty() || dgt.size() > 4 || dgt[0] == '0' || dgt.find_first_not_of("0123456789") != std::string::npos) return false;
    unsigned x = 0;
    for (const char ch : dgt) x = x * 10 + unsigned(ch - '0');
    if (x < lo || x > hi) return false;
    v = x;
    return true;
  };
  SearchName s;
  if (name == "bsearch") {
    s.kind = SearchName::whole, s.shown = "bsearch(reimplementation)";
  } else if (name == "bsearchstd") {
    s.kind = SearchName::stdlower, s.shown = "bsearchstd(reimplementation)";
  } else if (name.compare(0, 6, "rbsstd") == 0) {
    if (!number(name.substr(6), kRbsMinBits, kRbsMaxBits, s.knob)) return false;
    s.kind = SearchName::rbsstd, s.shown = name;  // SOSD's own class, and the search SOSD runs inside its bound
  } else if (name.compare(0, 4, "rbst") == 0) {
    if (!number(name.substr(4), kRbsMinBits, kRbsMaxBits, s.knob)) return false;
    s.kind = SearchName::rbst, s.shown = name + "(reimplementation)";
  } else if (name.compare(0, 3, "rbs") == 0) {
    if (!number(name.substr(3), kRbsMinBits, kRbsMaxBits, s.knob)) return false;
    s.kind = SearchName::rbs, s.shown = name;  // SOSD's own class
  } else if (name.compare(0, 6, "interp") == 0) {
    std::string dgt = name.substr(6);
    s.gallop = !dgt.empty() && dgt.back() == 'e';
    if (s.gallop) dgt.pop_back();
    s.knob = kInterpDefaultSteps;
    if (!dgt.empty() && !number(dgt, 1, kInterpMaxSteps, s.knob)) return false;
    s.kind = SearchName::interp, s.shown = "interp" + std::to_string(s.knob) + (s.gallop ? "e" : "") + "(reimplementation)";
  } else if (name.compare(0, 3, "tip") == 0) {
    std::string dgt = name.substr(3);
    s.authors_first = !dgt.empty() && dgt.back() == 'a';
    if (s.authors_first) dgt.pop_back();
    s.knob = kTipDefaultGuard;
    if (!dgt.empty() && !number(dgt, 1, kTipMaxGuard, s.knob)) return false;
    s.kind = SearchName::tip, s.shown = "tip" + std::to_string(s.knob) + (s.authors_first ? "a" : "") + "(reimplementation)";
  } else {
    return false;
  }
  out = s;
  return true;
}

namespace search_detail {

template <class T>
struct Tag {
  using type = T;
};

template <class K, class P, template <class, uint32_t> class M, class F, size_t... I>
bool for_bits(unsigned bits, F&& f, std::index_sequence<I...>) {
  return ((bits == kRbsMinBits + I && (f(Tag<SortedBoundIndex<K, P, M<K, uint32_t(kRbsMinBits + I)>>>{}), true)) || ...);
}

template <class K, class P, template <class, uint32_t> class S, class F, size_t... I>
bool for_bits_search(unsigned bits, F&& f, std::index_sequence<I...>) {
  return ((bits == kRbsMinBits + I && (f(Tag<SortedSearchIndex<K, P, S<K, uint32_t(kRbsMinBits + I)>>>{}), true)) || ...);
}

// calls f with the type of the index that s names
template <class K, class P, class F>
bool with_type(const SearchName& s, F&& f) {
  constexpr auto all_bits = std::make_index_sequence<kRbsMaxBits - kRbsMinBits + 1>{};
  switch (s.kind) {
    case SearchName::whole: return f(Tag<SortedBoundIndex<K, P, WholeArrayBound<K>>>{}), true;
    case SearchName::stdlower: return f(Tag<SortedSearchIndex<K, P, StdLowerSearch<K>>>{}), true;
    case SearchName::rbs: return for_bits<K, P, RbsBound>(s.knob, f, all_bits);
    case SearchName::rbsstd: return for_bits_search<K, P, RbsStdSearch>(s.knob, f, all_bits);
    case SearchName::rbst: return for_bits<K, P, RbstBound>(s.knob, f, all_bits);
    case SearchName::interp:
      if (s.gallop) return f(Tag<SortedSearchIndex<K, P, InterpSearch<K, true>>>{}), true;
      return f(Tag<SortedSearchIndex<K, P, InterpSearch<K, false>>>{}), true;
    case SearchName::tip:
      if (s.authors_first) return f(Tag<SortedSearchIndex<K, P, TipSearch<K, true>>>{}), true;
      return f(Tag<SortedSearchIndex<K, P, TipSearch<K, false>>>{}), true;
    case SearchName::none: break;
  }
  return false;
}

template <class K, class P, class M>
indexInterface<K, P>* construct(Tag<SortedBoundIndex<K, P, M>>, const SearchName& s) {
  return new SortedBoundIndex<K, P, M>(s.shown);
}
template <class K, class P, class S>
indexInterface<K, P>* construct(Tag<SortedSearchIndex<K, P, S>>, const SearchName& s) {
  return new SortedSearchIndex<K, P, S>(s.shown, S(s.knob));
}

}  // namespace search_detail

// the index by name; nullptr for every other name
template <class K, class P>
indexInterface<K, P>* make_searches(const std::string& name) {
  SearchName s;
  if (!parse_search_name(name, s)) return nullptr;
  indexInterface<K, P>* r = nullptr;
  search_detail::with_type<K, P>(s, [&](auto tag) { r = search_detail::construct<K, P>(tag, s); });
  return r;
}

// indexInterface has no virtual destructor: deletes what make_searches(name) returned as the type it is. false (and nothing deleted)
// if name is not one of these names.
template <class K, class P>
bool destroy_searches(indexInterface<K, P>* p, const std::string& name) {
  SearchName s;
  if (!parse_search_name(name, s)) return false;
  return search_detail::with_type<K, P>(s, [&](auto tag) { delete static_cast<typename decltype(tag)::type*>(p); });
}

}  // namespace sota
}  // namespace gret
