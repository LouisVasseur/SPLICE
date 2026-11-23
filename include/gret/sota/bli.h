// SPDX-License-Identifier: MIT
// BliIndex<K, P, S, D>: BLI (Dong, Wang, Liu, Du: "BLI: A High-performance Bucket-based Learned Index with Concurrency Support",
// arXiv 2502.10597, 2025) in GRE's harness. This is a WRAPPER around the authors' own single-threaded code
// (https://github.com/gdymind/Buck-Learned-Index, commit b229631, used unmodified and included by path: -I <BLI>/include
// -I <BLI>/src), not a reimplementation. The bulk load runs the authors' BuckIndex::bulk_load, lookups BuckIndex::lookup, scans
// BuckIndex::scan, inserts Bucket::insert on the D-Bucket the authors' Segment::lb_lookup leads to and BuckIndex::insert when
// that bucket is full (or BuckIndex::insert alone: names ending in u). docs/sota/bli.md has what follows with file and line.
//
// The structure: unsorted D-Buckets of D slots hold the pairs, at the first free slot at or after hint = murmur64(key) % D;
// above them, levels of Segments (a linear model over S-Buckets of S entries, each entry the smallest key of a child and its
// pointer). As released, a D-Bucket lookup is a scalar scan from the hint: the AVX2 code in bucket.h (SIMD_lookup) is written
// for a key-array layout the index does not use, and BUCKINDEX_USE_SIMD is off in the authors' benchmark. This header refuses
// to compile with it on: with the released D-Bucket layout it compares against a zero vector.
//
// What the wrapper adds is what the authors' code lacks or gets wrong outside the runs it was written for:
//   - key 0. The authors' index needs a smallest key that routes everything (their GRE wrapper inserts the pair (0, 0) when
//     the data does not start at 0, and their insert turns a put of key 0 into an update). Here that pair is always there, and
//     a flag says whether key 0 is stored: without it get(0) would answer "found, 0" on every data set.
//   - key 2^64 - 1 is kept outside (a flag and its payload): it is the authors' "no pivot" value, and their lookup of it walks
//     into empty S-Buckets and returns an uninitialised pointer.
//   - put looks for the key first (the authors' insert does not, and adds a second copy of a stored key): a scan from the hint
//     up to the first free slot, which is exact because the authors' buckets fill the first free slot from the hint and never
//     free one. The check reads what a lookup reads; near a full bucket that is long (the bucket splits only when full). Names
//     ending in u leave the check out and call the authors' insert alone: only for callers that never put a stored key (GRE's
//     insert workloads do not). Then a put of a stored key adds a second copy, as it does in the authors' code.
//   - update, in place (the authors' code has none; Bucket::update, which they use for key 0, scans the bucket from slot 0).
//   - remove is not supported and returns false: the structure has no deletion, and freeing a slot would break the rule the
//     lookups above rest on.
//   - memory_consumption: every D-Bucket, every Segment with its S-Buckets, the index object (what the authors' mem_size()
//     adds up, without its quadratic walk), and this object.
//   - guards, each counted and reported, for three defects of the authors' code (tests/sota/test_bli.cpp provokes each):
//     1. Keys the models cannot place. The Segment constructor indexes its S-Buckets with a double model and no bound
//        (segment.h:91-109); from a certain key magnitude on, rounding can carry it past the array. Keys above safe_key() are
//        refused with std::invalid_argument (bulk_load and put). From D = 2^16 at the default fill every key below 2^64 - 1 is
//        safe; smaller D-Buckets have a limit (2^56.3 for D = 256).
//     2. A truncated S-Bucket number. predict_buck (segment.h:300) converts the predicted S-Bucket number to 32 bits. For a
//        key far above the keys a Segment was fitted to (an outlier behind a dense region) the number passes 2^32, the search
//        for the key's S-Bucket starts from an arbitrary one below the right one, and it stops at the first empty S-Bucket
//        on its way up: the walk can end in a wrong D-Bucket. Then the authors' lookup misses a key that is stored where it
//        belongs, and the authors' insert stores a key where scans return it out of order and a later split loses it. Here:
//        a miss of the authors' lookup is followed by a check of the walk, and a key that is in its own D-Bucket is served
//        from there (get, update, the duplicate check of put); a checked put whose walk would misplace the key throws
//        std::runtime_error and stores nothing; scans are not affected (a misrouted start is an earlier bucket, and the scan
//        filters). With the unchecked put (u) the authors' insert is not watched: memory_consumption finds misplaced keys
//        afterwards and throws.
//        The same truncation can misplace a PIVOT. When a D-Bucket is split, its new pivots are put into the Segment by
//        batch_update, which looks for the S-Bucket with the same truncating search (segment.h:255); and when a Segment is
//        cut again its pieces go to the parent the same way. A pivot that truncates there lands in an earlier S-Bucket (or,
//        if that one is full, is dropped: segment.h:260 only asserts). Lookups of its keys still arrive (their search is
//        the same one), so the audit of the keys sees nothing; but the Segment is no longer in key order, and the authors'
//        scan, which walks the S-Buckets in sequence, returns keys out of order. It needs a D-Bucket whose upper half are
//        keys the walk misroutes (they came in with a bulk load, or before a Segment above was cut again: the checked put
//        refuses a key that is misrouted when it is put). Here: after a checked put that splits a D-Bucket below a
//        Segment at risk, every Segment is checked for key order and the pairs are counted, and std::runtime_error is
//        thrown if either is off (the structure is then out of order: the run must stop); with the unchecked put (u) and
//        at any other time, memory_consumption makes the same check and throws.
//     3. A new root built past its array. When the root is split, the new root is a Segment over the pieces, with a
//        least-squares model (buck_index.h:446-459), and the same unbounded constructor. A least-squares line overshoots the
//        last pivot when the pivots lie dense first and sparse later; from about nine pieces on that can be a whole S-Bucket
//        (AddressSanitizer: heap-buffer-overflow, segment.h:98). It was provoked here only with a bulk-load error bound of
//        1024 (keys with exponentially growing gaps), never with the default 4. A pivot written there is unreachable with
//        its subtree: after every new root the pairs that can be reached are counted, and a shortfall throws
//        std::runtime_error (memory is corrupted by then; the run must stop).
//     Besides: bulk_load looks every key up afterwards and counts the pairs held, and throws std::runtime_error if the
//     structure does not answer exactly. memory_consumption counts the pairs again and looks up every key a truncation can
//     touch; $BLI_AUDIT=1 makes it check every key. It throws if anything is wrong.
//   - the authors' progress lines on std::cout are silenced ($BLI_VERBOSE=1 keeps them), and the destructor frees the nodes.
//
// Names: bli[<L>][s<S>][f<PP>][e<E>][u]
//   <L>   log2 of the D-Bucket size in slots: 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 22, 24. Default 17 (131072 slots), the
//         size of the authors' overall comparison (script/run_overall.sh in their GRE fork)
//   s<S>  the S-Bucket size: 4, 8 or 16. Default 4 (same script)
//   f<PP> the initial fill ratio in percent, 30 to 90. Default 60 (same script; the paper's section 6.4 finds it best)
//   e<E>  the error bound of the bulk load's segmentation, 1 to 1024. Default 4 (same script). Large values make defect 3 likely
//   u     put without the look-up first (see above)
// bli = bli17s4f60e4. The hint is murmur64 and the root model a regression, as the authors' build sets them
// (HINT_MURMUR_HASH, BUCKINDEX_USE_LINEAR_REGRESSION); both are compile-time choices of their code and are fixed here.
//
// Needs x86-64 with AVX2 (bucket.h includes <immintrin.h> and names AVX types), or GRET_BLI_SHIM with a stand-in immintrin.h
// on the include path (tests/sota/test_bli.cpp says how; the scalar code is all the index runs). Without either, and without
// the authors' sources on the include path, this header declares only make_bli, which then throws for a bli name.
// One translation unit of a program should include it (the authors' murmur64 is kept internal to each unit that does).
// Not thread-safe. The authors' constructor starts 11 worker threads for a parallel scan this wrapper does not call; they sleep.
#pragma once

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <time.h>

#define GRET_SOTA_HAVE_BLI 0
#if defined(__has_include)
#if __has_include("buck_index.h") && __has_include("../src/util.h")
#if (defined(__x86_64__) && defined(__AVX2__)) || defined(GRET_BLI_SHIM)
#undef GRET_SOTA_HAVE_BLI
#define GRET_SOTA_HAVE_BLI 1
#endif
#endif
#endif

#if GRET_SOTA_HAVE_BLI

#if defined(BUCKINDEX_USE_SIMD)
#error "gret/sota/bli.h: BUCKINDEX_USE_SIMD must be off: BuckIndex's D-Buckets are KeyValueList, for which bucket.h's SIMD_load_keys returns a zero vector (bucket.h:484)"
#endif
#if defined(HINT_MOD_HASH) || defined(HINT_CL_HASH) || defined(HINT_MODEL_PREDICT) || defined(NO_HINT) || defined(BUCKINDEX_DEBUG)
#error "gret/sota/bli.h sets BLI's compile-time options itself (HINT_MURMUR_HASH, BUCKINDEX_USE_LINEAR_REGRESSION): do not define others"
#endif

#include <immintrin.h>  // x86-64: the compiler's. With GRET_BLI_SHIM: a stand-in on the include path (types and declarations)

// ---- the authors' sources, unmodified, in a namespace of their own. Every standard header they include is included above, so
// inside the namespace those includes are no-ops; <cassert> is read again and only redefines assert. util.h goes into an
// unnamed namespace because it defines murmur64 without inline. 'private' is defined away for these files only: the wrapper
// reads the root, the level count, a Segment's model and a bucket's slots, which have no accessors.
#pragma push_macro("HINT_MURMUR_HASH")
#pragma push_macro("BUCKINDEX_USE_LINEAR_REGRESSION")
#pragma push_macro("DEFAULT_FILLED_RATIO")
#undef HINT_MURMUR_HASH
#undef BUCKINDEX_USE_LINEAR_REGRESSION
#define HINT_MURMUR_HASH
#define BUCKINDEX_USE_LINEAR_REGRESSION
namespace gret {
namespace bli_authors {
namespace {
#include "../src/util.h"  // <BLI>/include/../src/util.h
}  // namespace
#define private public
#include "buck_index.h"
#undef private
}  // namespace bli_authors
}  // namespace gret
#pragma pop_macro("DEFAULT_FILLED_RATIO")
#pragma pop_macro("BUCKINDEX_USE_LINEAR_REGRESSION")
#pragma pop_macro("HINT_MURMUR_HASH")

namespace gret {

struct BliConfig {
  double fill = 0.6;    // the authors' initial_filled_ratio
  int error_bound = 4;  // the authors' error_bound (bulk-load segmentation of the model layers)
  bool checked = true;  // put looks for the key first
};

template <class K, class P, size_t S, size_t D>
class BliIndex final : public indexInterface<K, P> {
  static_assert(std::is_integral<K>::value && std::is_unsigned<K>::value && sizeof(K) == 8, "BliIndex: keys are uint64");
  static_assert(std::is_integral<P>::value && sizeof(P) == 8, "BliIndex: payloads are 64-bit integers");
  static_assert(D >= 64 && (D & (D - 1)) == 0 && D <= (size_t(1) << 24), "BliIndex: D is a power of two in [64, 2^24]");
  static_assert(S >= 2 && S <= 64, "BliIndex: S in [2, 64]");

  using Bli = bli_authors::buckindex::BuckIndex<K, P, S, D>;
  using DBucket = typename Bli::DataBucketType;
  using Seg = typename Bli::SegmentType;
  using KV = typename Bli::KeyValueType;
  using KVPtr = typename Bli::KeyValuePtrType;

 public:
  static constexpr K kMaxKey = std::numeric_limits<K>::max();  // the authors' "no pivot" (bucket.h:52): never given to their code
  // A model prediction from which predict_buck (segment.h:300) truncates: 2^32 S-Buckets. Two slots are taken off because the
  // compiler may contract the authors' slope * key + offset into one fused operation and this header's into two, or the
  // reverse; for keys up to safe_key() the two results differ by less than one slot (see the constructor).
  static constexpr double kTruncates = 4294967296.0 * double(S) - 2.0;

  // what an audit found (audit(), memory_consumption)
  struct Audit {
    size_t keys = 0;           // pairs in the D-Buckets, the pair of key 0 included
    size_t buckets = 0, segments = 0;
    size_t bytes = 0;          // D-Buckets and Segments with their S-Buckets
    size_t risk_segments = 0;  // Segments whose model can pass 2^32 S-Buckets for a key routed to them (segment.h:300 truncates)
    size_t looked_up = 0;      // keys checked with the authors' lookup
    size_t rerouted = 0;       // of those, in their own D-Bucket but not found by the authors' lookup (its walk ends elsewhere): get serves them
    size_t misplaced = 0;      // keys found in a D-Bucket that is not reached for them (outside its key range)
    size_t disordered = 0;     // Segments whose S-Buckets are not in key order (a pivot placed by a truncated search, segment.h:255): scans would be out of order
    size_t wrong = 0;          // keys looked up that neither the authors' lookup nor the wrapper finds with their payload, plus the misplaced ones and the Segments out of order
  };

  explicit BliIndex(std::string name, const BliConfig& cfg = BliConfig()) : name_(std::move(name)), cfg_(cfg) {
    if (!(cfg.fill >= 0.3 && cfg.fill <= 0.9)) throw std::invalid_argument(name_ + ": the fill ratio must be in [0.3, 0.9]");
    if (cfg.error_bound < 1 || cfg.error_bound > 1024) throw std::invalid_argument(name_ + ": the error bound must be in [1, 1024]");
    // Keys the authors' Segment constructor can place: it computes bucket = uint64(slope * key + offset) / S in double and
    // indexes the S-Bucket array with it, unbounded (segment.h:91-109). In exact arithmetic the last pivot lands 1 / fill
    // slots before the end of the array. The rounding error is below 6 * 2^-53 * key * slope, and slope <= 1 / (gap * fill)
    // with gap the distance of neighbouring pivots, which is at least the number of keys a D-Bucket holds when it is made:
    // D * fill at the bulk load, D / 2 at a split. So keys up to 2^50 * min(D * fill, D / 2) * fill keep the error under a slot.
    const double m = std::min(std::floor(double(D) * cfg.fill), double(D) / 2);
    const double lim = std::ldexp(m * cfg.fill, 50);
    safe_ = lim >= 18446744073709551615.0 ? kMaxKey - 1 : K(lim);
    Quiet quiet;
    idx_.reset(new Bli(cfg.fill, cfg.error_bound));  // starts the authors' 11 scan workers; they wait on a condition variable
  }
  BliIndex(const BliIndex&) = delete;
  BliIndex& operator=(const BliIndex&) = delete;
  ~BliIndex() { reset(); }  // indexInterface has no virtual destructor: deleted through it, as GRE does, the nodes stay allocated

  void init(Param* = nullptr) override {}

  // kv: sorted by key, keys unique (GRE's contract; checked)
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 1; i < n; ++i)
      if (kv[i].first <= kv[i - 1].first) throw std::invalid_argument(name_ + ": bulk_load needs sorted, unique keys");
    reset();
    if (n == 0) return;
    size_t hi = n;
    if (kv[n - 1].first == kMaxKey) has_max_ = true, max_val_ = kv[n - 1].second, --hi;
    if (hi > 0 && kv[hi - 1].first > safe_) {
      const size_t over = size_t(kv + hi - std::upper_bound(kv, kv + hi, safe_, [](K s, const std::pair<K, P>& p) { return s < p.first; }));
      reset();
      throw std::invalid_argument(name_ + ": " + std::to_string(over) + " keys are above " + std::to_string(safe_) +
                                  ", the largest key D-Buckets of " + std::to_string(D) + " slots hold exactly (the authors' models are doubles); use 2^16 slots or more");
    }
    double load_ms = 0;
    {
      std::vector<KV> kvs;
      kvs.reserve(hi + 1);
      live0_ = hi > 0 && kv[0].first == 0;
      if (!live0_) kvs.emplace_back(K(0), P(0));  // the pair the authors' structure routes everything below the data to
      for (size_t i = 0; i < hi; ++i) kvs.emplace_back(kv[i].first, kv[i].second);
      const auto t1 = std::chrono::steady_clock::now();
      {
        Quiet quiet;
        idx_->bulk_load(kvs);  // the authors' bulk load
      }
      load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    }
    size_ = n;
    pairs_ = hi + size_t(!live0_);
    max_key_ = hi ? kv[hi - 1].first : K(0);
    // every key against the authors' lookup, and the number of pairs held
    const auto t2 = std::chrono::steady_clock::now();
    size_t wrong = 0, served = 0;  // served: not found by the authors' lookup (a truncated walk), found by get
    for (size_t i = 0; i < hi; ++i) {
      P v = P();
      if (__builtin_expect(idx_->lookup(kv[i].first, v) && v == kv[i].second, 1)) continue;
      const P* const at = rerouted(kv[i].first);
      ++(at && *at == kv[i].second ? served : wrong);
    }
    const Audit a = audit(false);
    const double verify_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
    if (wrong || a.wrong || a.keys != pairs_) {
      const std::string what = name_ + ": after the authors' bulk load " + std::to_string(wrong) + " of " + std::to_string(hi) + " keys are not found with their payload, " +
                               std::to_string(a.wrong) + " fail the audit, and the buckets hold " + std::to_string(a.keys) + " pairs for " + std::to_string(pairs_);
      reset();
      throw std::runtime_error(what);
    }
    std::fprintf(stderr,
                 "gret: %s (BLI: the authors' code, unmodified, behind guards) n=%zu dbucket=%zu sbucket=%zu fill=%.2f error_bound=%d put=%s levels=%llu buckets=%zu segments=%zu "
                 "risk_segments=%zu risk_keys_checked=%zu keys_the_authors_lookup_misses=%zu key0=%s keymax=%s bytes_per_key=%.2f load_ms=%.1f verify_ms=%.1f build_ms=%.1f\n",
                 name_.c_str(), n, D, S, cfg_.fill, cfg_.error_bound, cfg_.checked ? "checked" : "unchecked", (unsigned long long)idx_->num_levels_, a.buckets, a.segments,
                 a.risk_segments, a.looked_up, served, live0_ ? "stored" : "dummy", has_max_ ? "outside" : "absent", double(bytes_of(a)) / double(n), load_ms, verify_ms,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }

  bool get(K key, P& val, Param* = nullptr) override {
    if (__builtin_expect(K(key - 1) >= K(kMaxKey - 1), 0)) {  // 0 and 2^64 - 1
      if (key == kMaxKey) {
        if (!has_max_) return false;
        val = max_val_;
        return true;
      }
      if (!live0_) return false;
    }
    P v;
    if (__builtin_expect(!idx_->lookup(key, v), 0)) {  // the authors' lookup
      const P* const at = rerouted(key);  // a miss: was the walk misrouted, and is the key where it belongs?
      if (!at) return false;
      ++rescued_gets_;
      v = *at;
    }
    val = v;
    return true;
  }

  // true if the key was not there and is now; false, and nothing changes, if it was (names ending in u: not looked for, see the
  // head of this file). Throws std::invalid_argument for a key above safe_key().
  bool put(K key, P value, Param* = nullptr) override {
    if (__builtin_expect(K(key - 1) >= safe_, 0)) {  // 0, 2^64 - 1, and what the authors' models cannot place
      if (key == kMaxKey) {
        if (has_max_) return false;
        has_max_ = true, max_val_ = value, ++size_;
        return true;
      }
      if (key != 0) throw std::invalid_argument(name_ + ": put of key " + std::to_string(key) + ", above " + std::to_string(safe_) + ", the largest key D-Buckets of " +
                                                std::to_string(D) + " slots hold exactly");
      if (live0_) return false;
      if (!idx_->root_) seed();
      P* at = find(leaf(K(0)), K(0));
      if (!at) throw std::logic_error(name_ + ": the pair of key 0 is gone from the authors' structure");
      *at = value;
      live0_ = true, ++size_;
      return true;
    }
    if (__builtin_expect(!idx_->root_, 0)) seed();
    KV e(key, value);
    if (cfg_.checked) {
      bool risky = false;
      DBucket* const d = leaf(key, risky);
      if (__builtin_expect(risky, 0) && check_route(key, d)) return false;  // throws if the authors' walk would misplace the key
      const size_t hint = hint_of(key);
      if (find(d, key, hint)) return false;
      if (__builtin_expect(!d->insert(e, true, hint), 0)) grow(e, path_at_risk(key));  // the authors' Bucket::insert; full: their BuckIndex::insert splits
      else ++pairs_;
    } else {
      grow(e);
    }
    ++size_;
    if (key > max_key_) max_key_ = key;
    return true;
  }

  bool update(K key, P value, Param* = nullptr) override {
    if (__builtin_expect(K(key - 1) >= K(kMaxKey - 1), 0)) {
      if (key == kMaxKey) {
        if (!has_max_) return false;
        max_val_ = value;
        return true;
      }
      if (!live0_) return false;
    }
    if (!idx_->root_) return false;
    P* at = find(leaf(key), key);
    if (!at) at = rerouted(key);
    if (!at) return false;
    *at = value;
    return true;
  }

  bool remove(K, Param* = nullptr) override { return false; }  // BLI has no deletion

  // the authors' scan (a top-k selection in each D-Bucket it passes), then the key 2^64 - 1 if it is stored
  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    size_t got = 0;
    if (n == 0) return 0;
    if (idx_->root_ && key != kMaxKey) {
      if (n > size_t(INT_MAX)) n = size_t(INT_MAX);  // the authors count in int
      got = idx_->scan(key == 0 && !live0_ ? K(1) : key, n, out);
    }
    if (has_max_ && got < n) out[got++] = std::pair<K, P>(kMaxKey, max_val_);
    return got;
  }

  // Every byte held: the D-Buckets (all their slots), the Segments with their S-Buckets, the authors' index object, this
  // object. Not the allocator's overhead. GRE asks once, after the run, and gives no other hook, so this is also where the
  // structure is checked (audit(), below) and a line says what the run did. Throws std::runtime_error if the D-Buckets do not
  // hold the pairs that were stored, a key is misplaced, a Segment is out of key order, or the authors' lookup does not find a
  // key it is checked for.
  // $BLI_AUDIT=1: every key is checked.
  long long memory_consumption() override {
    const char* env = std::getenv("BLI_AUDIT");
    const bool full = env && *env && *env != '0';
    const Audit a = audit(full);
    const bool bad = a.wrong != 0 || a.keys != pairs_;
    if (bad || full || puts_grown_ != reported_grown_ || a.keys != reported_keys_) {
      reported_grown_ = puts_grown_, reported_keys_ = a.keys;
      std::fprintf(stderr,
                   "gret: %s now keys=%zu pairs_in_buckets=%zu (stored %zu%s) levels=%llu buckets=%zu segments=%zu %s=%zu new_roots=%zu risk_segments=%zu risk_puts=%zu refused_puts=%zu "
                   "risk_splits=%zu keys_looked_up=%zu keys_the_authors_lookup_misses=%zu gets_served_past_it=%zu misplaced=%zu segments_out_of_order=%zu wrong=%zu%s\n",
                   name_.c_str(), size_, a.keys, pairs_, cfg_.checked ? "" : "; unchecked puts: more pairs than keys means stored keys were put again", (unsigned long long)levels(), a.buckets,
                   a.segments, cfg_.checked ? "bucket_splits" : "authors_inserts", puts_grown_, new_roots_, a.risk_segments, risk_puts_, refused_puts_, risk_splits_, a.looked_up, a.rerouted, rescued_gets_,
                   a.misplaced, a.disordered, a.wrong, bad ? "  ERROR: the authors' structure lost or misplaced keys, or a Segment is out of key order" : "");
    }
    if (bad)
      throw std::runtime_error(name_ + ": the authors' structure holds " + std::to_string(a.keys) + " pairs for " + std::to_string(pairs_) + " stored, " + std::to_string(a.misplaced) +
                               " keys are misplaced, " + std::to_string(a.disordered) + " Segments are out of key order (scans are wrong) and " +
                               std::to_string(a.wrong - a.misplaced - a.disordered) + " of " + std::to_string(a.looked_up) + " lookups fail");
    return (long long)bytes_of(a);
  }

  // One pass over the authors' nodes. Counts them and their pairs, and checks the keys a truncation can touch. A Segment is at
  // risk when its model gives some key routed to it an S-Bucket number of 2^32 or more: predict_buck (segment.h:300) keeps the
  // low 32 bits, and the search that starts from the wrong S-Bucket stops at the first empty one. Such a key, if it is stored
  // where it belongs, may not be found: every stored key from the Segment's first such key on is looked up with the authors'
  // lookup. And a put of such a key by the authors' insert may have gone to an earlier D-Bucket below the same Segment, where
  // lookups find it, scans return it out of order, and a later split loses it: with the unchecked put (names ending in u; the
  // checked put refuses such a key) every D-Bucket below a Segment at risk is read for keys outside its range.
  // Every Segment is also checked for key order (in_order, below): a pivot placed by a truncated search breaks it.
  // all = true: every stored key is looked up and must lie in the key range its D-Bucket is reached for.
  Audit audit(bool all) const {
    Audit a;
    if (!idx_->root_) return a;
    const bool stray = all || !cfg_.checked;  // look for misplaced keys
    struct Item {
      void* p;
      uint64_t level;
      K lo, hi;  // the keys routed here
      K exam;    // stored keys from this one up are checked; kMaxKey (never stored here): none
    };
    std::vector<Item> todo;
    todo.push_back(Item{idx_->root_, 0, K(0), kMaxKey - 1, all ? K(0) : kMaxKey});
    std::vector<KVPtr> ch;
    const uint64_t levels = idx_->num_levels_;
    while (!todo.empty()) {
      const Item it = todo.back();
      todo.pop_back();
      if (it.level + 1 == levels) {
        DBucket* const d = static_cast<DBucket*>(it.p);
        ++a.buckets, a.bytes += sizeof(DBucket), a.keys += d->num_keys();
        if (it.exam == kMaxKey || (!stray && it.hi < it.exam)) continue;
        for (size_t i = 0; i < D; ++i) {
          if (!d->valid(int(i))) continue;
          const KV& e = d->list_.kvs_[i];
          if (e.key_ < it.exam) continue;
          if (e.key_ < it.lo || e.key_ > it.hi) {
            ++a.misplaced, ++a.wrong;
            continue;
          }
          P v = P();
          ++a.looked_up;
          if (idx_->lookup(e.key_, v) && v == e.value_) continue;
          const P* const at = rerouted(e.key_);
          ++(at && *at == e.value_ ? a.rerouted : a.wrong);
        }
        continue;
      }
      Seg* const s = static_cast<Seg*>(it.p);
      ++a.segments, a.bytes += s->mem_size();
      if (!in_order(s)) ++a.disordered, ++a.wrong;
      ch.clear();
      for (int b = 0; b < s->num_bucket_; ++b)
        for (size_t i = 0; i < S; ++i)
          if (s->sbucket_list_[b].valid(int(i))) ch.push_back(s->sbucket_list_[b].at(int(i)));
      std::sort(ch.begin(), ch.end());
      K exam = it.exam;
      // the smallest key whose predicted S-Bucket number reaches 2^32, if a key routed here can be that large
      const double slope = s->model_.get_slope(), offset = s->model_.get_offset();
      const K reach = std::min(it.hi, max_key_);
      if (slope > 0 && slope * double(reach) + offset >= kTruncates) {
        ++a.risk_segments;
        const double t = (kTruncates - offset) / slope * (1.0 - 1e-9) - 1.0;
        const K from = t <= 0 ? K(0) : t >= 18446744073709549568.0 ? kMaxKey - 1 : K(t);
        exam = std::min(exam, std::max(from, it.lo));
      }
      for (size_t j = 0; j < ch.size(); ++j)
        todo.push_back(Item{reinterpret_cast<void*>(ch[j].value_), it.level + 1, j ? ch[j].key_ : it.lo, j + 1 < ch.size() ? K(ch[j + 1].key_ - 1) : it.hi, exam});
    }
    return a;
  }

  // 0: the authors' walk for this key truncates nowhere; 1: it truncates an S-Bucket number and still ends in the right
  // D-Bucket; 2: it ends in a wrong one (a checked put of the key is refused). For tests and reports.
  int route_state(K key) const {
    if (!idx_->root_ || key == kMaxKey) return 0;
    bool risky = false;
    DBucket* const d = leaf(key, risky);
    return !risky ? 0 : true_leaf(key) == d ? 1 : 2;
  }

  // for tests and reports
  size_t size() const { return size_; }
  K safe_key() const { return safe_; }               // the largest key the authors' structure is given (2^64 - 2: no limit)
  size_t bucket_splits() const { return puts_grown_; }  // puts that went through the authors' BuckIndex::insert (all of them with u)
  size_t new_roots() const { return new_roots_; }
  size_t risk_puts() const { return risk_puts_; }        // checked puts whose walk truncated; each was checked against the pivots
  size_t refused_puts() const { return refused_puts_; }  // of those, refused: the authors' walk would have misplaced the key
  size_t risk_splits() const { return risk_splits_; }    // checked puts that split a D-Bucket below a Segment at risk; the structure was checked after each
  size_t pairs() const { return pairs_; }                // what the D-Buckets must hold (the pair of key 0 included)
  size_t rescued_gets() const { return rescued_gets_; }  // gets the authors' lookup missed and the wrapper answered from the key's own D-Bucket
  uint64_t levels() const { return idx_->root_ ? idx_->num_levels_ : 0; }
  bool key0_is_dummy() const { return idx_->root_ && !live0_; }
  size_t authors_mem_size() const {  // the authors' own count (it prints, and walks each Segment quadratically)
    if (!idx_->root_) return sizeof(Bli);
    Quiet quiet;
    return idx_->mem_size();
  }

 private:
  struct Quiet {  // the authors' constructor, bulk load and mem_size report on std::cout
    struct Null : std::streambuf {
      int overflow(int c) override { return c == EOF ? 0 : c; }
      std::streamsize xsputn(const char*, std::streamsize n) override { return n; }
    } sink;
    std::streambuf* old = nullptr;
    Quiet() {
      const char* v = std::getenv("BLI_VERBOSE");
      if (!v || !*v || *v == '0') old = std::cout.rdbuf(&sink);
    }
    ~Quiet() {
      if (old) std::cout.rdbuf(old);
    }
  };

  static size_t hint_of(K key) { return size_t(bli_authors::murmur64(key) % D); }  // buck_index.h:191, 370

  // the D-Bucket a key belongs to: the loop of the authors' lookup (buck_index.h:164-175)
  DBucket* leaf(K key) const {
    uintptr_t p = reinterpret_cast<uintptr_t>(idx_->root_);
    KVPtr at, next;
    for (uint64_t level = idx_->num_levels_ - 1; level > 0; --level) {
      static_cast<Seg*>(reinterpret_cast<void*>(p))->lb_lookup(key, at, next);
      p = at.value_;
    }
    return static_cast<DBucket*>(reinterpret_cast<void*>(p));
  }

  // The same walk, and risky = some Segment on it computes an S-Bucket number of 2^32 or more for the key. predict_buck
  // (segment.h:300) keeps the low 32 bits of that number, so the search for the key's S-Bucket can start anywhere below the
  // right one, and it stops at the first empty S-Bucket on its way up (segment.h:315): the walk can end in a wrong D-Bucket.
  DBucket* leaf(K key, bool& risky) const {
    uintptr_t p = reinterpret_cast<uintptr_t>(idx_->root_);
    KVPtr at, next;
    const double k = double(key);  // as LinearModel::predict converts it (linear_model.h:69)
    for (uint64_t level = idx_->num_levels_ - 1; level > 0; --level) {
      const Seg* const s = static_cast<Seg*>(reinterpret_cast<void*>(p));
      risky |= s->model_.get_slope() * k + s->model_.get_offset() >= kTruncates;
      s->lb_lookup(key, at, next);
      p = at.value_;
    }
    return static_cast<DBucket*>(reinterpret_cast<void*>(p));
  }

  // the D-Bucket a key belongs to by the pivots alone, no model: in each Segment the entry with the largest key <= key
  DBucket* true_leaf(K key) const {
    void* p = idx_->root_;
    for (uint64_t level = idx_->num_levels_ - 1; level > 0; --level) {
      const Seg* const s = static_cast<Seg*>(p);
      bool any = false;
      K best = 0;
      uintptr_t child = 0;
      for (int b = 0; b < s->num_bucket_; ++b)
        for (size_t i = 0; i < S; ++i) {
          if (!s->sbucket_list_[b].valid(int(i))) continue;
          const KVPtr e = s->sbucket_list_[b].at(int(i));
          if (e.key_ <= key && (!any || e.key_ > best)) any = true, best = e.key_, child = e.value_;
        }
      if (!any) throw std::logic_error(name_ + ": a Segment of the authors' structure has no entry at or below key " + std::to_string(key));
      p = reinterpret_cast<void*>(child);
    }
    return static_cast<DBucket*>(p);
  }

  // Does a Segment on the key's path compute an S-Bucket number of 2^32 or more for a key that can be routed to it (the
  // test of audit()'s risk_segments)? Then a pivot made by a split below it can truncate in batch_update (segment.h:231,
  // 255) and be placed in a wrong S-Bucket. Asked only when a D-Bucket is full, before the authors' insert splits it; the
  // path is followed by the pivots alone (the put has made sure that it is the authors' path).
  bool path_at_risk(K key) const {
    void* p = idx_->root_;
    const K top = std::max(key, max_key_);
    K hi = kMaxKey - 1;  // the largest key routed to the Segment at hand
    for (uint64_t level = idx_->num_levels_ - 1; level > 0; --level) {
      const Seg* const s = static_cast<Seg*>(p);
      if (s->model_.get_slope() * double(std::min(hi, top)) + s->model_.get_offset() >= kTruncates) return true;
      bool any = false;
      K best = 0;
      uintptr_t child = 0;
      for (int b = 0; b < s->num_bucket_; ++b)
        for (size_t i = 0; i < S; ++i) {
          if (!s->sbucket_list_[b].valid(int(i))) continue;
          const KVPtr e = s->sbucket_list_[b].at(int(i));
          if (e.key_ > key) hi = std::min(hi, K(e.key_ - 1));
          else if (!any || e.key_ > best) any = true, best = e.key_, child = e.value_;
        }
      if (!any) throw std::logic_error(name_ + ": a Segment of the authors' structure has no entry at or below key " + std::to_string(key));
      p = reinterpret_cast<void*>(child);
    }
    return false;
  }

  // Is the Segment as the authors' searches and scans need it: every entry of an S-Bucket above every entry of the
  // S-Buckets before it, and an S-Bucket's pivot its smallest key (2^64 - 1 when it is empty)?
  static bool in_order(const Seg* s) {
    bool any = false;
    K last = 0;  // the largest key so far
    for (int b = 0; b < s->num_bucket_; ++b) {
      const auto& sb = s->sbucket_list_[b];
      bool some = false;
      K lo = 0, hi = 0;
      for (size_t i = 0; i < S; ++i) {
        if (!sb.valid(int(i))) continue;
        const K x = sb.at(int(i)).key_;
        if (!some) lo = hi = x, some = true;
        else lo = std::min(lo, x), hi = std::max(hi, x);
      }
      if (!some) {
        if (sb.get_pivot() != kMaxKey) return false;
        continue;
      }
      if (sb.get_pivot() != lo || (any && lo <= last)) return false;
      any = true, last = hi;
    }
    return true;
  }

  // A miss in the D-Bucket the authors' walk ends in. If the walk truncated and that is not the key's D-Bucket, the key may
  // still be stored where it belongs (it was loaded there, or put there before a Segment was rebuilt): its payload, or nullptr.
  P* rerouted(K key) const {
    if (!idx_->root_) return nullptr;
    bool risky = false;
    DBucket* const d = leaf(key, risky);
    if (!risky) return nullptr;
    DBucket* const t = true_leaf(key);
    return t == d ? nullptr : find(t, key);
  }

  // A put whose walk truncates: the key goes in only if the authors' walk still ends in the right D-Bucket. True if the key is
  // stored already (in its own D-Bucket, where the authors' walk does not arrive).
  bool check_route(K key, DBucket* d) {
    ++risk_puts_;
    DBucket* const t = true_leaf(key);
    if (t == d) return false;
    if (find(t, key)) return true;
    ++refused_puts_;
    throw std::runtime_error(name_ + ": put of key " + std::to_string(key) + " refused: a Segment's model gives it an S-Bucket number of 2^32 or more, the authors' "
                             "code keeps the low 32 bits (segment.h:300) and its walk ends in a wrong D-Bucket; the key would be misplaced");
    return false;
  }

  // Where the key's payload is in its D-Bucket, or nullptr. The slots are read in the order of the authors' lookup (from the
  // hint, wrapping), and the first free slot ends the search: an insert takes the first free slot at or after its hint
  // (find_empty_slot, bucket.h:311) and no slot of a D-Bucket is ever freed, so a key is never behind a free slot.
  static P* find(DBucket* d, K key, size_t hint) {
    for (size_t i = 0, l = hint; i < D; ++i, l = (l + 1) & (D - 1)) {
      if (!((d->bitmap_[l >> 6] >> (l & 63)) & 1)) return nullptr;
      if (d->list_.kvs_[l].key_ == key) return &d->list_.kvs_[l].value_;
    }
    return nullptr;
  }
  static P* find(DBucket* d, K key) { return find(d, key, hint_of(key)); }

  // an index never loaded: the authors' structure starts from the pair of key 0 alone
  void seed() {
    std::vector<KV> one;
    one.emplace_back(K(0), P(0));
    Quiet quiet;
    idx_->bulk_load(one);
    live0_ = false, pairs_ = 1;
  }

  // the authors' insert (buck_index.h:342): finds the D-Bucket again, and splits it when it is full.
  // watch: the D-Bucket is full and a Segment above it is at risk (path_at_risk): the split's pivots may be misplaced.
  void grow(KV& e, bool watch = false) {
    const uint64_t levels = idx_->num_levels_;
    if (!idx_->insert(e)) throw std::logic_error(name_ + ": the authors' insert refused key " + std::to_string(e.key_));
    ++puts_grown_, ++pairs_;
    const bool new_root = idx_->num_levels_ != levels;
    if (__builtin_expect(new_root || watch, 0)) {
      // A new root (buck_index.h:439-464): a Segment built from a least-squares line through the pivots of the pieces the old
      // root was cut into. Its constructor indexes the S-Bucket array with the model's prediction and no bound
      // (segment.h:91-109), and a least-squares line can pass the last slot (from about nine pivots on, when they lie dense
      // first and sparse later). A pivot placed there was written behind the array, and nothing leads to its subtree any more:
      // the pairs that can be reached are counted, here, after every new root.
      // A split below a Segment at risk: batch_update finds the S-Bucket for a new pivot with the truncating search
      // (segment.h:231, 255-260). A pivot that truncates is put into an earlier S-Bucket, or dropped if that one is full:
      // the Segments are checked for key order and the pairs are counted.
      new_root ? ++new_roots_ : ++risk_splits_;
      size_t disordered = 0;
      const size_t have = count_pairs(&disordered);
      if (have != pairs_ && new_root)
        throw std::runtime_error(name_ + ": the authors' new root (buck_index.h:446-459) lost a subtree: " + std::to_string(have) + " pairs can be reached, " + std::to_string(pairs_) +
                                 " were stored. Its Segment was built past its S-Bucket array (segment.h:91-109): memory is corrupted");
      if (have != pairs_)
        throw std::runtime_error(name_ + ": the authors' split lost a pivot: " + std::to_string(have) + " pairs can be reached, " + std::to_string(pairs_) + " were stored. batch_update "
                                 "looked for its S-Bucket with a truncated S-Bucket number (segment.h:300) and found a full one (segment.h:255-260)");
      if (disordered)
        throw std::runtime_error(name_ + ": after the split made by the put of key " + std::to_string(e.key_) + ", " + std::to_string(disordered) + " Segment(s) are out of key order: the authors' "
                                 "batch_update placed a pivot with a truncated S-Bucket number (segment.h:300, 255). Scans are out of order from here on");
    }
  }

  // the pairs in the D-Buckets that can be reached from the root: one pass over the nodes, no slot of a D-Bucket is read.
  // disordered: if given, the number of Segments that are not in key order (in_order)
  size_t count_pairs(size_t* disordered = nullptr) const {
    if (!idx_->root_) return 0;
    size_t pairs = 0;
    std::vector<std::pair<void*, uint64_t>> todo(1, std::make_pair(idx_->root_, uint64_t(0)));
    const uint64_t levels = idx_->num_levels_;
    while (!todo.empty()) {
      const std::pair<void*, uint64_t> it = todo.back();
      todo.pop_back();
      if (it.second + 1 == levels) {
        pairs += static_cast<DBucket*>(it.first)->num_keys();
        continue;
      }
      Seg* const s = static_cast<Seg*>(it.first);
      if (disordered && !in_order(s)) ++*disordered;
      for (int b = 0; b < s->num_bucket_; ++b)
        for (size_t i = 0; i < S; ++i)
          if (s->sbucket_list_[b].valid(int(i))) todo.emplace_back(reinterpret_cast<void*>(s->sbucket_list_[b].at(int(i)).value_), it.second + 1);
    }
    return pairs;
  }

  size_t bytes_of(const Audit& a) const { return a.bytes + sizeof(Bli) + sizeof(*this); }

  void reset() {
    if (idx_ && idx_->root_) {  // the authors' destructor frees no node
      std::vector<std::pair<void*, uint64_t>> todo(1, std::make_pair(idx_->root_, uint64_t(0)));
      const uint64_t levels = idx_->num_levels_;
      while (!todo.empty()) {
        const std::pair<void*, uint64_t> it = todo.back();
        todo.pop_back();
        if (it.second + 1 == levels) {
          delete static_cast<DBucket*>(it.first);
          continue;
        }
        Seg* const s = static_cast<Seg*>(it.first);
        for (int b = 0; b < s->num_bucket_; ++b)
          for (size_t i = 0; i < S; ++i)
            if (s->sbucket_list_[b].valid(int(i))) todo.emplace_back(reinterpret_cast<void*>(s->sbucket_list_[b].at(int(i)).value_), it.second + 1);
        delete s;
      }
      idx_->root_ = nullptr;
      idx_->num_levels_ = 0;
    }
    live0_ = false, has_max_ = false, max_val_ = P();
    size_ = 0, pairs_ = 0, max_key_ = 0, puts_grown_ = 0, new_roots_ = 0, risk_puts_ = 0, refused_puts_ = 0, risk_splits_ = 0, rescued_gets_ = 0, reported_grown_ = 0, reported_keys_ = 0;
  }

  std::string name_;
  BliConfig cfg_;
  K safe_ = 0;                // see the constructor
  std::unique_ptr<Bli> idx_;  // the authors' index
  bool live0_ = false;        // key 0 is stored (its pair is in the authors' structure either way, once there is a root)
  bool has_max_ = false;      // key 2^64 - 1 is stored, here:
  P max_val_ = P();
  size_t size_ = 0;           // keys stored
  size_t pairs_ = 0;          // pairs given to the authors' structure and not refused: what its D-Buckets must hold
  K max_key_ = 0;             // the largest key given to the authors' structure
  size_t puts_grown_ = 0;     // puts through the authors' BuckIndex::insert
  size_t new_roots_ = 0;      // of those, how many added a level
  size_t risk_puts_ = 0;      // checked puts whose walk truncated an S-Bucket number (each was checked against the pivots)
  size_t refused_puts_ = 0;   // of those, how many the authors' walk would have misplaced: refused with std::runtime_error
  size_t risk_splits_ = 0;    // checked puts that split a D-Bucket below a Segment at risk: Segment order and pair count were checked after each
  size_t rescued_gets_ = 0;   // gets the authors' lookup missed (a truncated walk) and the key's own D-Bucket answered
  size_t reported_grown_ = 0, reported_keys_ = 0;  // what memory_consumption last printed
};

}  // namespace gret

#endif  // GRET_SOTA_HAVE_BLI

namespace gret {

// what a name of the form bli[<L>][s<S>][f<PP>][e<E>][u] says; false for any other name
struct BliName {
  int log2_d = 17, s = 4, fill_pct = 60, error_bound = 4;
  bool checked = true;
};
inline bool parse_bli_name(const std::string& name, BliName& out) {
  if (name.compare(0, 3, "bli") != 0) return false;
  BliName r;
  size_t p = 3;
  auto number = [&](int lo, int hi, int& v) {  // decimal, no leading zero, in [lo, hi]
    const size_t b = p;
    while (p < name.size() && name[p] >= '0' && name[p] <= '9') ++p;
    if (p == b || p - b > 4 || name[b] == '0') return false;
    v = std::stoi(name.substr(b, p - b));
    return v >= lo && v <= hi;
  };
  if (p < name.size() && name[p] >= '0' && name[p] <= '9' && !number(8, 24, r.log2_d)) return false;
  if (p < name.size() && name[p] == 's' && (++p, !number(4, 16, r.s))) return false;
  if (p < name.size() && name[p] == 'f' && (++p, !number(30, 90, r.fill_pct))) return false;
  if (p < name.size() && name[p] == 'e' && (++p, !number(1, 1024, r.error_bound))) return false;
  if (p < name.size() && name[p] == 'u') r.checked = false, ++p;
  if (p != name.size()) return false;
  switch (r.log2_d) {
    case 8: case 10: case 12: case 14: case 15: case 16: case 17: case 18: case 19: case 20: case 22: case 24: break;
    default: return false;
  }
  if (r.s != 4 && r.s != 8 && r.s != 16) return false;
  out = r;
  return true;
}

#if GRET_SOTA_HAVE_BLI
namespace bli_detail {
template <class K, class P, size_t S>
indexInterface<K, P>* make(const std::string& name, int log2_d, const BliConfig& c) {
  switch (log2_d) {
#define GRET_BLI_CASE(L) case L: return new BliIndex<K, P, S, size_t(1) << L>(name, c);
    GRET_BLI_CASE(8) GRET_BLI_CASE(10) GRET_BLI_CASE(12) GRET_BLI_CASE(14) GRET_BLI_CASE(15) GRET_BLI_CASE(16)
    GRET_BLI_CASE(17) GRET_BLI_CASE(18) GRET_BLI_CASE(19) GRET_BLI_CASE(20) GRET_BLI_CASE(22) GRET_BLI_CASE(24)
#undef GRET_BLI_CASE
  }
  return nullptr;
}
}  // namespace bli_detail
#endif

// "bli[<L>][s<S>][f<PP>][e<E>][u]" -> the index; nullptr for any other name, and for key or payload types other than 64-bit
// integers (unsigned keys). Throws std::invalid_argument for a bli name in a build without the authors' sources or without AVX2.
template <class K, class P>
indexInterface<K, P>* make_bli(const std::string& name) {
  BliName n;
  if (!parse_bli_name(name, n)) return nullptr;
#if GRET_SOTA_HAVE_BLI
  if constexpr (std::is_integral<K>::value && std::is_unsigned<K>::value && sizeof(K) == 8 && std::is_integral<P>::value && sizeof(P) == 8) {
    BliConfig c;
    c.fill = n.fill_pct / 100.0, c.error_bound = n.error_bound, c.checked = n.checked;
    return n.s == 4 ? bli_detail::make<K, P, 4>(name, n.log2_d, c) : n.s == 8 ? bli_detail::make<K, P, 8>(name, n.log2_d, c) : bli_detail::make<K, P, 16>(name, n.log2_d, c);
  } else {
    return nullptr;
  }
#else
  throw std::invalid_argument("gret::make_bli: '" + name + "': built without BLI (needs x86-64 with AVX2 and -I <Buck-Learned-Index>/include -I <Buck-Learned-Index>/src)");
#endif
}

}  // namespace gret
