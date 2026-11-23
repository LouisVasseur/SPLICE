// SPDX-License-Identifier: MIT
// PGM++ (Liu, Han, Qi, Peng, Li, Lin, Chen: "Why Are Learned Indexes So Effective but Sometimes Ineffective?", arXiv 2410.00846,
// Section 6) in GRE's harness: a PGM-index with two tuned error bounds (last level eps_l, internal levels eps_i), whose sparse
// upper levels are skipped and whose levels are searched with a hybrid of linear scan and branch-free binary search.
//
// THIS IS A HYBRID, NOT THE AUTHORS' CODE AS IT STANDS, and every index built from this file says so in its build line:
// "<name>(authors' builder, reimplementation of the lookup)".
//   - The STRUCTURE is the authors': the segments and levels are built by the static pgm::PGMIndex::build of their repository
//     (github.com/qyliu-hkust/bench_search, exp_pgm/pgm_index.h and piecewise_linear_model.h, their copy of Vinciguerra's PGM-index
//     with float slopes and 16-byte segments), unmodified and included by path:  -I <bench_search>  (the directory above exp_pgm).
//     Nothing of it is copied here.
//   - The LOOKUP is a reimplementation, written here line for line after their segment_for_key and search, because theirs loses
//     keys. Their search::upper_bound_branchless (exp_pgm/search_algo.h) stops one comparison short: it returns the last candidate
//     without comparing it, so std::prev of it is one segment too early whenever the right segment is the last one of the
//     searched range, which is the case for every key under the last segment of a level. With an internal error bound above
//     their linear-scan threshold (their branch-free path) stored keys then fall outside the returned range: in about a third of
//     the reference builds of tests/sota/test_pgmpp.cpp, which counts them on every run, for a quarter to a third of the keys of
//     one 2M-key set and for most keys of some small ones. Their benchmark times lookups and never checks an answer. Here the
//     search makes that last comparison. Where their code is right (std::upper_bound, and the linear scan) this lookup returns
//     exactly what theirs returns, position for position: the test compares the two on every key of every set. One more line
//     differs: where their Segment::operator() converts a double of 2^64 or more to size_t (undefined; see position() below).
//   - LAYER SKIPPING is the paper's (6.1, step 1); in the authors' file it is a commented-out block. Here it is on by default
//     (suffix n turns it off, which is their published code path).
//   - The TUNER is not the paper's cost model (no code and no constants for it are published): see "names" below.
//   - The LAST-MILE search over the key array is SortedBoundIndex's branch-free binary search, at every eps_l. That is what the
//     authors' search_data does (search::lower_bound_branchless, completed by the final comparison). The paper's step 3 applies
//     the hybrid there too; by its rule (below) that is a linear scan only for eps_l <= 3 at delta 8 and eps_l <= 7 at delta 16,
//     so for none of its error bounds {4, ..., 1024} on its Intel and ARM machines. Not done here.
//
//   PgmppBound<K>    a Model for SortedBoundIndex (readonly.h): one sorted array of (key, payload) pairs, the model bounds the
//                    position, the same branch-free binary search inside the bound as spgm and the other read-only baselines
//   make_pgmpp       the index by name, nullptr for every other name
//
// Names: pgmpp<eps_l>[i<eps_i>][d<delta>][n], decimal numbers without leading zeros
//   <eps_l>      error bound of the last level, in positions of the key array (PGM's Epsilon), 1..65536. This is the size knob:
//                the paper derives it from the space budget (Eq. 22) and searches {4, 8, ..., 1024}.
//   i<eps_i>     error bound of the internal levels (PGM's EpsilonRecursive), 1..65536. Without it, eps_i is chosen at build among
//                the paper's set {4, 8, ..., 1024}: the internal levels are built for each by the authors' builder (the last
//                level does not depend on eps_i and is built once), whole lookups (the model, then the branch-free search of
//                the key array inside the bound) are timed on random stored keys, 2^16 to warm up and three blocks of 2^16 fresh
//                ones, in two passes over the candidates; the best block of each counts and the smallest eps_i within 2% of
//                the fastest is kept. The paper picks the eps_i with the lowest cost under its model (Eq. 21, 23), whose
//                constants it measures per machine and does not publish; timing the candidates themselves is the 'T_opt' its
//                Figure 14 compares that tuner against. The choice and the nine timings are printed; the choice can differ
//                between runs where candidates are within timing noise.
//   d<delta>     the hybrid threshold, 0..1024, default 8 (the authors' main.cpp; the paper: 8 on its Intel and ARM machines, 16
//                on its AMD one). AS IN THE AUTHORS' CODE, a level is scanned linearly when eps_i <= delta and searched
//                branch-free otherwise. The paper states the rule on the size of the range instead (Eq. 21d: linear when
//                2 * eps_i + 1 <= delta), which with its own numbers would never scan at delta 8; the two disagree, and this
//                follows the code. d3 (d7 for its delta 16) gives the paper's rule inside the levels; skipping then starts at 3 (7) segments.
//                Layer skipping starts at the lowest level with at most delta segments (the paper's step 1).
//   n            no layer skipping: start at the root, as the authors' published code does.
//   pgmpp64  pgmpp64i16  pgmpp16i4  pgmpp64d16  pgmpp64i16d16n  pgmpp1024
//
// Guards (docs/sota/pgmpp.md has the detail):
//   - 2^64 - 1 is PGM's sentinel and the authors' builder throws on it: as readonly.h's PgmBound does, the model is built without
//     it and every key above the largest built key goes straight to the end of the array.
//   - After the build every stored key is looked up once with the very code get() runs. A key found outside the nominal range
//     [pos - eps_l, pos + eps_l + 2) is counted (outside_bound in the build line) and the range is widened for all lookups by the
//     largest miss on each side, so a stored key is found whatever the float slopes did. The count is 0 on every set tested
//     but one kind: a segment over tens of millions of positions, where the rounding of the slope to a float adds up (about
//     3e-8 positions per position: 72M keys on one line lose 4.9M keys at eps_l 1 in the authors' structure, 200M lose 32M at
//     eps_l 4). There the bound is widened by 1 to 4 positions and every key is found (tests/sota/test_pgmpp.cpp, check_drift).
//   - Fewer than two keys below 2^64 - 1, or 2^32 keys or more (segment intercepts are uint32_t): build throws.
//
// Undefined behaviour in the authors' builder, not hidden: exp_pgm/pgm_index.h:327 (Segment::operator(), size_t of a double of
// 2^64 or more), reached from their build() at :112, which evaluates each level's last segment at 2^64 - 2 to decide whether to
// append one more segment. It happens whenever that segment's slope is about 1 or more (dense keys); UBSan stops there. Either
// answer is a valid index (the extra segment only serves keys above the largest one, which get() never sends to the model), and
// the check of every stored key above does not depend on it. So that a sanitized build can test the rest, this file switches
// that one UBSan check off for the functions of that one file of theirs (below, at the include; -DPGMPP_STRICT_UBSAN keeps it on).
//
// Include after indexInterface.h and GRE's pgm header, like readonly.h (which this includes). The authors' header defines
// namespace pgm, class PGMIndex and namespace search, the names GRE's own PGM copy uses; it is therefore included INSIDE
// namespace gret::sota::pgmpp_authors, after every standard header it includes itself (so those are no-ops inside the namespace).
// Their search_algo.h uses std::cout without including <iostream>; it is included first here.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "gret/readonly.h"

namespace gret {
namespace sota {
namespace pgmpp_authors {
// the PGM++ authors' PGM: -I <bench_search>. The two headers pgm_index.h includes come first, so that what follows applies to
// pgm_index.h alone.
#include "exp_pgm/piecewise_linear_model.h"
#include "exp_pgm/search_algo.h"
// In a clang build with UBSan, the float-cast-overflow check is off for the functions of the authors' pgm_index.h, and for
// nothing else: their Segment::operator() (line 327) converts doubles of 2^64 or more to size_t, from their build() (line 112)
// on any dense key set, and the sanitized test would stop there before testing anything (see the top of this file and
// docs/sota/pgmpp.md). -DPGMPP_STRICT_UBSAN leaves the check on and shows the report. No other build is affected.
#if defined(__clang__) && defined(__has_feature) && !defined(PGMPP_STRICT_UBSAN)
#if __has_feature(undefined_behavior_sanitizer)
#define GRET_PGMPP_UBSAN_RELAXED 1
#pragma clang attribute push(__attribute__((no_sanitize("float-cast-overflow"))), apply_to = function)
#endif
#endif
#include "exp_pgm/pgm_index.h"
#ifdef GRET_PGMPP_UBSAN_RELAXED
#pragma clang attribute pop
#endif
}  // namespace pgmpp_authors

struct PgmppConfig {
  size_t eps_l = 64;     // last-level error bound (PGM's default Epsilon)
  size_t eps_i = 0;      // internal error bound; 0: chosen at build among {4, ..., 1024}
  size_t delta = 8;      // hybrid threshold
  bool skip = true;      // layer skipping
  bool apart = false;    // tests only: with eps_i given, build the internal levels apart from the last one, as the tuner does
  bool quiet = false;    // tests only: no build line
};

inline constexpr size_t kPgmppMaxEps = 65536;
inline constexpr size_t kPgmppMaxDelta = 1024;
inline constexpr size_t kPgmppCandidates[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024};  // the paper's set of error bounds

inline bool pgmpp_valid(const PgmppConfig& c) {
  return c.eps_l >= 1 && c.eps_l <= kPgmppMaxEps && c.eps_i <= kPgmppMaxEps && c.delta <= kPgmppMaxDelta;
}

// "pgmpp64i16d16n" -> {64, 16, 16, false}. The whole name must be pgmpp<eps_l>[i<eps_i>][d<delta>][n]; anything else is not a
// PGM++ name (false, and out is left as it was).
inline bool parse_pgmpp_name(const std::string& name, PgmppConfig& out) {
  if (name.compare(0, 5, "pgmpp") != 0) return false;
  size_t at = 5;
  auto number = [&](size_t max, size_t min, size_t& v) {
    const size_t from = at;
    while (at < name.size() && name[at] >= '0' && name[at] <= '9') ++at;
    const size_t len = at - from;
    if (len == 0 || len > 5 || (name[from] == '0' && len > 1)) return false;
    v = 0;
    for (size_t j = from; j < at; ++j) v = v * 10 + size_t(name[j] - '0');
    return v >= min && v <= max;
  };
  PgmppConfig c;
  if (!number(kPgmppMaxEps, 1, c.eps_l)) return false;
  if (at < name.size() && name[at] == 'i') {
    ++at;
    if (!number(kPgmppMaxEps, 1, c.eps_i)) return false;
  }
  if (at < name.size() && name[at] == 'd') {
    ++at;
    if (!number(kPgmppMaxDelta, 0, c.delta)) return false;
  }
  if (at < name.size() && name[at] == 'n') {
    ++at;
    c.skip = false;
  }
  if (at != name.size()) return false;
  out = c;
  return true;
}

// SortedBoundIndex default-constructs its Model and takes only a name, so the parameters of the next PgmppBound to be constructed
// on this thread are left here by make_pgmpp (and put back to the defaults by it).
inline PgmppConfig& pgmpp_pending() {
  thread_local PgmppConfig c;
  return c;
}

// What the last PgmppBound::build on this thread made (an index made by make_pgmpp holds its model out of reach: this is how a
// harness or a test reads which eps_i the tuner kept and whether the bound had to be widened).
struct PgmppReport {
  size_t eps_l = 0, eps_i = 0, delta = 0;
  bool skip = false, tuned = false;
  size_t height = 0, first_level = 0, leaf_segments = 0, bytes = 0;
  size_t outside = 0, widened_lo = 0, widened_hi = 0;  // stored keys outside the nominal bound, and by how much it was widened
  double build_ms = 0, verify_ms = 0;
};
inline PgmppReport& pgmpp_last_build() {
  thread_local PgmppReport r;
  return r;
}

// The base class is one instantiation of the authors' PGMIndex, used for what does not depend on its template arguments: the
// Segment type (key, float slope, uint32_t intercept: 16 bytes), the vectors that hold the levels, and the static build(), which
// takes both error bounds as run-time arguments (their constructor passes it the template arguments and nothing else). None of
// its search functions is instantiated.
template <class K>
class PgmppBound : private pgmpp_authors::pgm::PGMIndex<K, 64, 4, true, 8, float> {
  static_assert(std::is_same<K, uint64_t>::value || std::is_same<K, uint32_t>::value, "PgmppBound takes uint32_t or uint64_t keys");
  using Base = pgmpp_authors::pgm::PGMIndex<K, 64, 4, true, 8, float>;
  using Seg = typename Base::Segment;

 public:
  struct SegmentView {
    K key;
    float slope;
    uint32_t intercept;
  };

  PgmppConfig cfg;

  PgmppBound() : PgmppBound(pgmpp_pending()) {}
  explicit PgmppBound(const PgmppConfig& c) : cfg(c) {
    this->n = 0, this->first_key = 0, this->start_level = 0;  // the authors' default constructor leaves these three unset
  }

  // keys: sorted ascending and unique (as SortedBoundIndex passes them); a trailing 2^64 - 1 is taken off, as PgmBound does
  void build(std::vector<K>& keys) {
    if (!pgmpp_valid(cfg)) throw std::invalid_argument("PgmppBound: eps_l must be in [1, 65536], eps_i in [0, 65536], delta in [0, 1024]");
    n_ = keys.size();
    if (!keys.empty() && keys.back() == std::numeric_limits<K>::max()) keys.pop_back();
    built_ = keys.size();
    if (built_ < 2) throw std::invalid_argument("PgmppBound: needs at least two keys below the largest value of the key type");
    if (n_ > size_t(std::numeric_limits<uint32_t>::max())) throw std::length_error("PgmppBound: 2^32 keys or more (segment intercepts are uint32_t)");
    first_ = keys.front();
    top_ = keys.back();
    this->n = built_;
    this->first_key = first_;
    e_lo_ = cfg.eps_l;
    e_hi_ = cfg.eps_l + 2;
    tuned_ = false;
    tune_count_ = 0;

    const auto t0 = std::chrono::steady_clock::now();
    apart_ = false;
    if (cfg.eps_i == 0) {
      full_build(keys, kPgmppCandidates[0]);
      tune(keys);
    } else if (cfg.apart) {
      full_build(keys, cfg.eps_i == 4 ? 8 : 4);
      internal_levels(keys, indexed_leaf_keys(), cfg.eps_i);
    } else {
      full_build(keys, cfg.eps_i);
    }
    this->segments.shrink_to_fit();
    this->levels_offsets.shrink_to_fit();
    this->levels_segment_count.shrink_to_fit();
    const auto t1 = std::chrono::steady_clock::now();
    verify(keys);
    const auto t2 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    PgmppReport& r = pgmpp_last_build();
    r.eps_l = cfg.eps_l, r.eps_i = eps_i_, r.delta = cfg.delta, r.skip = cfg.skip, r.tuned = tuned_;
    r.height = height(), r.first_level = start_, r.leaf_segments = level_segments(0), r.bytes = bytes();
    r.outside = outside_, r.widened_lo = e_lo_ - cfg.eps_l, r.widened_hi = e_hi_ - cfg.eps_l - 2;
    r.build_ms = ms(t0, t1), r.verify_ms = ms(t1, t2);
    if (!cfg.quiet) {
      std::string timed;
      for (size_t j = 0; j < tune_count_; ++j) timed += (j ? " " : " candidates_ns=") + std::to_string(kPgmppCandidates[j]) + ":" + std::to_string(int(tune_ns_[j] + 0.5));
      std::fprintf(stderr,
                   "gret: pgmpp(authors' builder, reimplementation of the lookup) eps_l=%zu eps_i=%zu%s delta=%zu internal=%s skip=%d height=%zu start_level=%zu "
                   "leaf_segments=%zu model_bytes=%zu outside_bound=%zu widened=-%zu/+%zu build_ms=%.1f verify_ms=%.1f%s\n",
                   cfg.eps_l, eps_i_, tuned_ ? "(chosen by timing)" : "", cfg.delta, linear_ ? "linear" : "branchfree", int(cfg.skip), height(), start_,
                   level_segments(0), bytes(), outside_, e_lo_ - cfg.eps_l, e_hi_ - cfg.eps_l - 2, ms(t0, t1), ms(t1, t2), timed.c_str());
    }
  }

  // [lo, hi): where the lower bound of x is, if x is stored (and, by PGM's guarantee, also if it is not)
  KeyBound get(K x) const {
    if (__builtin_expect(x > top_, 0)) return {built_, n_};
    const K k = x < first_ ? first_ : x;
    const Seg* const segs = this->segments.data();
    const size_t* const off = this->levels_offsets.data();
    // layer skipping: the start level has at most delta segments (or is the root); scan it. Its last entry is the sentinel
    // segment (key 2^64 - 1 > k), which ends the scan.
    const Seg* it = segs + off[start_];
    while (it[1].key <= k) ++it;
    if (linear_) {  // eps_i <= delta: scan forward from the lowest candidate, as the authors' segment_for_key does
      for (size_t l = start_; l-- > 0;) {
        const size_t pos = position(it, k);
        it = segs + off[l] + (pos <= ei1_ ? 0 : pos - ei1_);
        while (it[1].key <= k) ++it;
      }
    } else {  // branch-free binary search over [pos - eps_i - 1, pos + eps_i + 2), completed by the comparison the authors' lacks
      for (size_t l = start_; l-- > 0;) {
        const size_t pos = position(it, k);
        const size_t level_size = off[l + 1] - off[l] - 1;
        const size_t lo = pos <= ei1_ ? 0 : pos - ei1_;
        const size_t hi = pos + ei1_ + 1 >= level_size ? level_size : pos + ei1_ + 1;
        const Seg* base = segs + off[l] + lo;
        size_t len = hi - lo;
        while (len > 1) {
          const size_t half = len / 2;
          base += base[half - 1].key <= k ? half : 0;
          len -= half;
        }
        it = base - size_t(base->key > k);  // std::prev(upper_bound): the rightmost segment whose key is <= k
      }
    }
    const size_t pos = position(it, k);
    return {pos <= e_lo_ ? 0 : pos - e_lo_, pos + e_hi_ >= built_ ? built_ : pos + e_hi_};
  }

  // the model's bytes: this object plus what the vectors hold (the authors' size_in_bytes() counts segments and level offsets)
  size_t bytes() const {
    return sizeof(*this) + this->segments.capacity() * sizeof(Seg) + this->levels_offsets.capacity() * sizeof(size_t) +
           this->levels_segment_count.capacity() * sizeof(size_t);
  }

  // what was built
  size_t eps_i() const { return eps_i_; }                // the internal error bound in use
  bool tuned() const { return tuned_; }                  // it was chosen by timing
  bool built_apart() const { return apart_; }            // the internal levels in use were built apart from the last level
  bool linear_internal() const { return linear_; }       // internal levels are scanned, not binary searched
  size_t height() const { return this->levels_offsets.size() - 1; }
  size_t first_level() const { return start_; }          // the level a lookup starts at (0 is the last level)
  size_t level_segments(size_t l) const { return this->levels_offsets[l + 1] - this->levels_offsets[l] - 1; }  // without the sentinel
  size_t outside() const { return outside_; }            // stored keys outside the nominal bound (the range was widened for them)
  size_t widened_lo() const { return e_lo_ - cfg.eps_l; }
  size_t widened_hi() const { return e_hi_ - cfg.eps_l - 2; }
  size_t built_keys() const { return built_; }           // keys the model was built on (n, or n - 1 if 2^64 - 1 is stored)
  std::vector<double> candidate_ns() const { return std::vector<double>(tune_ns_, tune_ns_ + tune_count_); }  // ns per lookup of each candidate
  size_t segment_entries() const { return this->segments.size(); }  // all levels, sentinels included
  SegmentView segment(size_t i) const { return {this->segments[i].key, this->segments[i].slope, this->segments[i].intercept}; }
  const std::vector<size_t>& offsets() const { return this->levels_offsets; }

 private:
  // What the authors' lookup computes from a segment: std::min<size_t>((*it)(k), std::next(it)->intercept), with their
  // Segment::operator() written out, size_t(slope * double(k - key)) + intercept. Theirs converts the double whatever its size,
  // which is undefined from 2^64 up; that is reached by a key that was never stored and lies in a gap of some 2^63 behind a steep
  // segment (and what the conversion then yields differs between SSE2, AVX-512 and arm64). Here a product of 2^63 or more goes
  // to the next segment's intercept, which is what their min() makes of any product that large.
  static size_t position(const Seg* it, K k) {
    const double d = double(it->slope) * double(k - it->key);
    const size_t next = it[1].intercept;
    if (__builtin_expect(!(d < 9223372036854775808.0), 0)) return next;
    return std::min<size_t>(size_t(d) + it->intercept, next);
  }

  // the authors' build, as their constructor calls it
  void full_build(const std::vector<K>& keys, size_t eps_i) {
    this->segments.clear();
    this->levels_offsets.clear();
    this->levels_segment_count.clear();
    Base::start_level = 0;
    Base::build(keys.begin(), keys.end(), cfg.eps_l, eps_i, this->segments, this->levels_offsets, this->levels_segment_count, Base::start_level);
    apart_ = false;
    shape(eps_i);
  }

  // The last level does not depend on eps_i, and the levels above it are what the authors' build() makes, with both error bounds
  // eps_i, of the keys of the last level's segments: the same build_level calls on the same input. One entry differs. Where a
  // level's last segment does not map everything above the keys to the end, build() appends a segment (last key + 1, slope 0, n);
  // 'last key' is the last key of what build() was given, so the data's largest key in the whole index and the last segment's
  // key here. internal_levels() puts the data's back. The result is, entry for entry, what one build() call over the data makes
  // (the test compares it with the authors' constructor for every set and eps_i).
  //
  // The keys of the last level's segments that the next level indexes. How many they are is not stored as such: it is the
  // intercept of the next level's sentinel segment, (2^64 - 1, 0, count). Empty where that cannot be relied on: one level only,
  // or a largest key from 2^64 - 3 up (a segment key can then be 2^64 - 2 and the sentinel one the segmentation made itself).
  std::vector<K> indexed_leaf_keys() const {
    std::vector<K> lk;
    if (height() < 2 || top_ >= std::numeric_limits<K>::max() - 2) return lk;
    const size_t count = this->segments[this->levels_offsets[2] - 1].intercept;
    if (count < 2 || count > level_segments(0)) return lk;
    lk.resize(count);
    for (size_t j = 0; j < count; ++j) lk[j] = this->segments[j].key;
    return lk;
  }

  // the internal levels for another eps_i; by a whole build() where they cannot be built apart
  void internal_levels(const std::vector<K>& keys, const std::vector<K>& lk, size_t eps_i) {
    if (!build_apart(lk, eps_i)) full_build(keys, eps_i);
  }

  bool build_apart(const std::vector<K>& lk, size_t eps_i) {
    if (lk.empty()) return false;
    std::vector<Seg> up;
    std::vector<size_t> up_off, up_count;
    int up_start = 0;
    Base::build(lk.begin(), lk.end(), eps_i, eps_i, up, up_off, up_count, up_start);
    const size_t h = up_off.size() - 1;
    for (size_t j = 0; j < h; ++j) {
      const size_t entries = up_off[j + 1] - up_off[j] - 1;                           // without the sentinel
      const size_t real = j + 1 < h ? size_t(up[up_off[j + 2] - 1].intercept) : 1;    // the segments the next level indexes
      if (entries == real) continue;
      if (entries != real + 1) return false;
      Seg& appended = up[up_off[j + 1] - 2];
      if (appended.key != lk.back() + 1 || appended.slope != 0) return false;
      appended.key = top_ + 1;
    }
    const size_t leaf_entries = this->levels_offsets[1];
    this->segments.resize(leaf_entries);
    this->segments.insert(this->segments.end(), up.begin(), up.end());
    this->levels_offsets.resize(2);
    for (size_t j = 1; j < up_off.size(); ++j) this->levels_offsets.push_back(leaf_entries + up_off[j]);
    this->levels_segment_count.clear();
    for (size_t j = 1; j < this->levels_offsets.size(); ++j) this->levels_segment_count.push_back(this->levels_offsets[j] - this->levels_offsets[j - 1]);
    apart_ = true;
    shape(eps_i);
    return true;
  }

  void shape(size_t eps_i) {
    eps_i_ = eps_i;
    ei1_ = eps_i + 1;
    linear_ = eps_i <= cfg.delta;
    start_ = height() - 1;
    if (cfg.skip)
      for (size_t l = 0; l < height(); ++l)
        if (level_segments(l) <= std::max<size_t>(cfg.delta, 1)) {
          start_ = l;
          break;
        }
  }

  // one block of whole lookups on random stored keys: the bound, then the branch-free search of the key array inside it (the
  // last-mile search SortedBoundIndex runs over the pairs). ns per lookup.
  double time_block(const std::vector<K>& keys, std::mt19937_64& g, std::vector<K>& block) {
    for (K& k : block) k = keys[g() % built_];
    uint64_t sum = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const K k : block) {
      const KeyBound b = get(k);
      const K* base = keys.data() + b.lo;
      size_t len = b.hi - b.lo;
      while (len > 1) {
        const size_t half = len / 2;
        base += base[half - 1] < k ? half : 0;
        len -= half;
      }
      sum += *base;
    }
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    sink_ += sum;
    return ns / double(block.size());
  }

  // eps_i: each candidate's internal levels built by the authors' builder and timed, twice over (the candidates in turn, so that
  // a disturbance does not fall on one of them alone), the best block of each counts; the smallest within 2% of the fastest is kept
  void tune(const std::vector<K>& keys) {
    if (start_ == 0) return;  // lookups start at the last level: no internal level is ever searched, eps_i does not matter
    tuned_ = true;
    std::mt19937_64 g(0x9e3779b97f4a7c15ull);
    std::vector<K> block(std::min<size_t>(built_, size_t(1) << 16));
    const std::vector<K> lk = indexed_leaf_keys();
    const size_t count = sizeof(kPgmppCandidates) / sizeof(kPgmppCandidates[0]);
    const int passes = lk.empty() ? 1 : 2;  // a second pass over the candidates where their internal levels are cheap to build
    for (int pass = 0; pass < passes; ++pass)
      for (size_t j = 0; j < count; ++j) {
        const size_t e = kPgmppCandidates[j];
        if (e != eps_i_) internal_levels(keys, lk, e);
        time_block(keys, g, block);  // warm: the model's levels, the predictors
        double ns = time_block(keys, g, block);
        for (int rep = 0; rep < 2; ++rep) ns = std::min(ns, time_block(keys, g, block));
        tune_ns_[j] = pass ? std::min(tune_ns_[j], ns) : ns;
      }
    tune_count_ = count;
    const double fastest = *std::min_element(tune_ns_, tune_ns_ + count);
    size_t best = 0;
    while (tune_ns_[best] > fastest * 1.02) ++best;
    if (kPgmppCandidates[best] != eps_i_) internal_levels(keys, lk, kPgmppCandidates[best]);
  }

  // every stored key once, through get(): count the keys outside the nominal bound and widen the bound to hold them (on up to
  // 20 threads from 2^20 keys, where OpenMP is on: the authors' segmentation uses as many)
  void verify(const std::vector<K>& keys) {
    size_t under = 0, over = 0, outside = 0;
    const size_t m = built_;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(max : under, over) reduction(+ : outside) num_threads(std::min(omp_get_max_threads(), 20)) if (m >= (size_t(1) << 20))
#endif
    for (size_t i = 0; i < m; ++i) {
      const KeyBound b = get(keys[i]);
      if (i < b.lo) {
        under = std::max(under, b.lo - i);
        ++outside;
      } else if (i >= b.hi) {
        over = std::max(over, i + 1 - b.hi);
        ++outside;
      }
    }
    outside_ = outside;
    e_lo_ += under;
    e_hi_ += over;
  }

  K first_ = 0, top_ = 0;        // the smallest and the largest key the model was built on
  size_t n_ = 0, built_ = 0;     // keys in the array; keys the model was built on
  size_t e_lo_ = 0, e_hi_ = 0;   // the bound is [pos - e_lo_, pos + e_hi_): eps_l and eps_l + 2, plus what verify() added
  size_t eps_i_ = 0, ei1_ = 0;   // the internal error bound in use, and that plus one
  size_t start_ = 0;             // the level lookups start at
  size_t outside_ = 0;
  bool linear_ = false, tuned_ = false, apart_ = false;
  uint64_t sink_ = 0;
  size_t tune_count_ = 0;
  double tune_ns_[sizeof(kPgmppCandidates) / sizeof(kPgmppCandidates[0])] = {};  // ns per lookup of each candidate, when eps_i was chosen
};

// pgmpp<eps_l>[i<eps_i>][d<delta>][n] -> the index; nullptr for every other name
template <class K, class P>
indexInterface<K, P>* make_pgmpp(const std::string& name) {
  PgmppConfig c;
  if (!parse_pgmpp_name(name, c)) return nullptr;
  struct Restore {
    ~Restore() { pgmpp_pending() = PgmppConfig{}; }
  } restore;
  pgmpp_pending() = c;
  return new SortedBoundIndex<K, P, PgmppBound<K>>(name + "(authors' builder, reimplementation of the lookup)");
}

}  // namespace sota
}  // namespace gret
