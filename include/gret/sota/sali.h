// SPDX-License-Identifier: MIT
// SaliIndex<K, P>: SALI (Ge, Zhang, Shi, Luo, Guo, Chai, Chen, Pan: "SALI: A Scalable Adaptive Learned Index Framework based on
// Probability Models", SIGMOD 2024), the concurrent successor of LIPP, in GRE's harness. This is a WRAPPER around the authors' own
// code (https://github.com/cds-ruc/SALI, MIT, used unmodified and included by path: -I <SALI>/src/core), not a reimplementation.
// Lookups run the authors' SALI::at, inserts SALI::insert, removals SALI::remove, updates SALI::update, scans
// SALI::range_query_len, the bulk load SALI::bulk_load: the same calls the authors' own GRE wrapper makes
// (github.com/YunWorkshop/GRE_SALI, src/competitor/sali/sali.h). What this wrapper adds is what their code lacks outside the
// workloads it was written for (docs/sota/sali.md has the list with file and line):
//   - a failed runtime check of theirs (RT_ASSERT, sali.h:44: it prints and calls exit(0), so a broken run ends with status 0)
//     throws instead. Their header is included with exit and fprintf defined as macros, so that the check records where it failed
//     and throws gret::sali_detail::RtAssert. Nothing else in their sources uses either name.
//   - get of a key that is not stored. Their at() ends in such a check when the key's slot holds another key (sali.h:610): the
//     wrapper catches exactly that one and answers false. A hit runs their at() and nothing else; a miss that lands on another
//     key costs a C++ throw (microseconds). GRE's lookups are hits.
//   - put of a stored key (their insert ends in a check with an item left locked, sali.h:1618), update and remove of a key that
//     is not stored (theirs overwrite or erase whatever key sits in the slot, sali.h:796, 864): the wrapper asks their exists()
//     first. "saliu" leaves that walk out of put, as their GRE wrapper does; a stored key then throws std::runtime_error.
//   - a side store (std::map) for keys their models cannot tell from a neighbour. The models are long double; a key that is
//     closer to the key below it than (key >> kShift) + 1 stays out of SALI (kShift = 60 where long double has 64 bits of
//     mantissa, x86-64: only keys from 2^60 up can be affected, and only by a neighbour less than 2 to 16 away; 49 where it is
//     a double). Without that, their two-key node can put two keys 1 apart above 2^63.58 in one slot (their check at
//     sali.h:1677 then ends the process with status 0) and a bulk load or a rebuild of keys 1 apart above 2^63 can run until
//     memory is gone. The line bulk_load prints says how many keys are there: a run in which that number is not small is not
//     a measurement of SALI.
//   - a check after the bulk load that their at() returns every key's payload (std::runtime_error if not), a limit of 2^29 - 1
//     keys per bulk load (their slot prediction clamps above INT_MAX / 2, sali.h:121, and a root has two slots a key),
//     std::cout silenced while their constructor and destructor run ($GRET_SALI_VERBOSE=1 keeps it), and
//     memory_consumption as their total_size() plus what that leaves out: the pool of recycled two-key nodes, the nodes waiting
//     in the epoch lists, the SALI object, the side store.
//   - a rebuild when the authors' own rebuild fails. With removes and inserts mixed, their rebuild inside an insert can divide
//     0 by 0 and ask for NaN slots (sali.h:3197, 2361); in their build that ends the process. The wrapper reads every key back
//     with their scan, sets their object aside and bulk loads a new one (salvage(), below, has the account), and counts how
//     often. Inserts alone or removes alone cannot cause it, and GRE never has both in one run. Any other exception out of
//     their insert or bulk load (out of memory) makes the index refuse every call until the next bulk_load.
// Names: sali (their default build: READ_EVOLVE 0, COMPRESS 0, as released and as their GRE fork pins it) and saliu (the same,
// put unchecked). SALI has no run-time knob. Its two compile-time modes are fixed by "#define READ_EVOLVE 0" and "#define
// COMPRESS 0" inside sali.h (lines 35, 36), without #ifndef: they cannot be turned on without editing the file, so there is no
// read-evolving variant here.
//
// SALI is a concurrent index and runs here on one thread. Every operation enters and leaves an epoch (a thread-specific lookup
// in tbb::enumerable_thread_specific, two atomic stores), reads each item's version word before and after using it, and keeps a
// version word in every slot (32 bytes a slot where LIPP has 16 and a bitmap). Nodes replaced by a rebuild wait in per-thread
// lists until the epoch has advanced three times. None of that is taken out.
//
// Needs: TBB's headers (combinable, enumerable_thread_specific, spin_mutex) and omp.h. Where there is no TBB (the Mac this was
// written on), put empty files tbb/combinable.h, tbb/enumerable_thread_specific.h and tbb/spin_mutex.h on the include path, and
// off x86 an empty immintrin.h as well: the wrapper then supplies single-purpose stand-ins inside its own namespace (tests only;
// the line bulk_load prints says tbb=stand-in). Their concurrency.h defines a function that is not inline (sali::yield), and
// this header instantiates two access templates explicitly: include it in one translation unit of a program.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <new>
#include <random>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <unistd.h>

#if !__has_include(<sali.h>) || !__has_include(<concurrency.h>)
#error "gret/sota/sali.h: put the SALI authors' src/core on the include path (-I <SALI>/src/core, https://github.com/cds-ruc/SALI)"
#endif
#if !__has_include("tbb/enumerable_thread_specific.h") || !__has_include("tbb/combinable.h") || !__has_include("tbb/spin_mutex.h")
#error "gret/sota/sali.h: SALI includes three TBB headers; without TBB, put empty files tbb/combinable.h, tbb/enumerable_thread_specific.h and tbb/spin_mutex.h on the include path (docs/sota/sali.md)"
#endif
// everything the authors' headers include is included here first, outside the namespace they are read in
#include <immintrin.h>
#pragma push_macro("omp_get_num_procs")  // PGM's header defines these two as macros in a build without OpenMP: an omp.h read
#pragma push_macro("omp_get_max_threads")  // after it (the Mac's stand-in) must still be able to declare them
#undef omp_get_num_procs
#undef omp_get_max_threads
#include "omp.h"
#pragma pop_macro("omp_get_max_threads")
#pragma pop_macro("omp_get_num_procs")
#include "tbb/combinable.h"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/spin_mutex.h"

namespace gret {
namespace sali_detail {

// where a runtime check of the authors' failed (their RT_ASSERT, redirected below)
struct RtAssert {
  const char* file;
  int line;
  const char* expr;
};
inline RtAssert& rt_last() noexcept {
  static thread_local RtAssert r{"", 0, ""};
  return r;
}
inline void rt_note(const char* file, int line, const char* expr) noexcept { rt_last() = RtAssert{file, line, expr}; }
[[noreturn]] inline void rt_fail() { throw rt_last(); }

}  // namespace sali_detail

namespace sali_authors {

#ifndef TBB_INTERFACE_VERSION
#define GRET_SALI_TBB_STANDIN 1
// No TBB on this machine (the three headers above were empty files): the names SALI uses, for tests. One object per thread,
// found by a search under a lock. The authors' code, inside this namespace, sees these before it looks for ::tbb.
namespace tbb {
enum ets_key_usage_type { ets_key_per_instance, ets_no_key };
template <class T>
using cache_aligned_allocator = std::allocator<T>;
class spin_mutex {
 public:
  void lock() noexcept {
    while (flag_.test_and_set(std::memory_order_acquire)) {}
  }
  void unlock() noexcept { flag_.clear(std::memory_order_release); }

 private:
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};
template <class T, class Alloc = std::allocator<T>, ets_key_usage_type = ets_no_key>
class enumerable_thread_specific {
  struct Slot {
    std::thread::id id;
    T value;
    template <class A>
    Slot(std::thread::id i, A&& a) : id(i), value(std::forward<A>(a)) {}
  };
  using List = std::list<Slot>;

 public:
  class const_iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;
    explicit const_iterator(typename List::const_iterator it) : it_(it) {}
    reference operator*() const { return it_->value; }
    pointer operator->() const { return &it_->value; }
    const_iterator& operator++() {
      ++it_;
      return *this;
    }
    bool operator==(const const_iterator& o) const { return it_ == o.it_; }
    bool operator!=(const const_iterator& o) const { return it_ != o.it_; }

   private:
    typename List::const_iterator it_;
  };
  template <class A>
  explicit enumerable_thread_specific(A arg) : make_([arg](List& l, std::thread::id id) { l.emplace_back(id, arg); }) {}
  T& local() {
    const std::thread::id me = std::this_thread::get_id();
    std::lock_guard<std::mutex> hold(mu_);
    for (Slot& s : slots_)
      if (s.id == me) return s.value;
    make_(slots_, me);
    return slots_.back().value;
  }
  const_iterator begin() const { return const_iterator(slots_.begin()); }
  const_iterator end() const { return const_iterator(slots_.end()); }

 private:
  std::function<void(List&, std::thread::id)> make_;
  List slots_;
  std::mutex mu_;
};
}  // namespace tbb
#endif

#if !defined(__x86_64__) && !defined(__i386__)
inline void _mm_pause() noexcept { sched_yield(); }  // their spin loops; never reached on one thread
#endif

// ---- the authors' sources, unmodified, in a namespace of their own. exit and fprintf are used by them in RT_ASSERT only
// (sali.h:44-51): with these two macros a failed check throws gret::sali_detail::RtAssert instead of ending the process with
// status 0. The macros their header leaves behind (likely, unlikely, RT_ASSERT and others) are removed again.
#pragma push_macro("exit")
#pragma push_macro("fprintf")
#pragma push_macro("likely")
#pragma push_macro("unlikely")
#pragma push_macro("READ_EVOLVE")
#pragma push_macro("COMPRESS")
#pragma push_macro("RT_ASSERT")
#pragma push_macro("PRINT_DEBUG")
#pragma push_macro("RT_DEBUG")
#pragma push_macro("COLLECT_TIME")
#undef exit
#undef fprintf
#undef likely
#undef unlikely
#undef READ_EVOLVE
#undef COMPRESS
#undef RT_ASSERT
#undef PRINT_DEBUG
#undef RT_DEBUG
#undef COLLECT_TIME
#define exit(code) ::gret::sali_detail::rt_fail()
#define fprintf(stream, format, thread, file, line, expr) ::gret::sali_detail::rt_note(file, line, expr)
#include <sali.h>  // <>, not "": the file being read is called sali.h too
#ifndef __SALI_H__
#error "gret/sota/sali.h: <sali.h> is not the SALI authors' src/core/sali.h"
#endif
#if READ_EVOLVE || COMPRESS
#error "gret/sota/sali.h: written for SALI as released (READ_EVOLVE 0, COMPRESS 0)"
#endif
#pragma pop_macro("COLLECT_TIME")
#pragma pop_macro("RT_DEBUG")
#pragma pop_macro("PRINT_DEBUG")
#pragma pop_macro("RT_ASSERT")
#pragma pop_macro("COMPRESS")
#pragma pop_macro("READ_EVOLVE")
#pragma pop_macro("unlikely")
#pragma pop_macro("likely")
#pragma pop_macro("fprintf")
#pragma pop_macro("exit")

}  // namespace sali_authors

namespace sali_detail {

using Sali64 = sali_authors::sali::SALI<uint64_t, uint64_t>;

// Two private members of the authors' classes, read (never written) by memory_consumption: the pool of recycled two-key nodes
// and a thread's lists of nodes waiting for their epoch. An explicit instantiation may name private members.
struct PoolTag {
  friend auto sali_member(PoolTag);
};
struct WaitTag {
  friend auto sali_member(WaitTag);
};
template <class Tag, auto Member>
struct Grant {
  friend auto sali_member(Tag) { return Member; }
};
template struct Grant<PoolTag, &Sali64::pending_two>;
template struct Grant<WaitTag, &Sali64::ThreadSpecificEpochBasedReclamationInformation::mFreeLists>;

}  // namespace sali_detail

template <class K, class P>
class SaliIndex final : public indexInterface<K, P> {
  static_assert(std::is_integral<K>::value && std::is_unsigned<K>::value && sizeof(K) == 8, "SaliIndex: keys are uint64");
  static_assert(std::is_integral<P>::value && std::is_unsigned<P>::value && sizeof(P) == 8, "SaliIndex: payloads are uint64");
  using Sali = sali_authors::sali::SALI<K, P>;

 public:
  // A key stays out of SALI when it is closer to the key below it than (key >> kShift) + 1. With p bits of mantissa (64 on
  // x86-64) the authors' two-key node (8 slots, the keys aimed at 8/3 and 16/3) can put its keys in one slot only if their
  // distance is below key / 2^(p - 0.42), and each of their FMCD nodes of three keys or more spreads its keys over two slots
  // at least when neighbours are key / 2^(p - 2.3) apart or more (docs/sota/sali.md has the arithmetic, and what the authors'
  // code did on 33,600 runs of close keys without this rule: it failed on keys 1 apart from 2^63 up and nowhere else).
  // p - 4 asks for 3.2 times the second distance.
  // The rule costs every put of a key it can concern (from 2^kShift up) one range query of the authors' (crowded(), below):
  // about half as much again as their insert on 2M keys (docs/sota/sali.md). -DGRET_SALI_SHIFT=n builds with another shift
  // (61 is still 1.6 times the second distance and leaves keys below 2^61 alone; 63 is the narrowest the probe supports;
  // 64 is no rule), to measure what the rule costs on a data set: the line bulk_load prints names the shift in use.
#ifndef GRET_SALI_SHIFT
#define GRET_SALI_SHIFT (LDBL_MANT_DIG >= 70 ? 64 : LDBL_MANT_DIG - 4)
#endif
  static constexpr int kShift = GRET_SALI_SHIFT;
  static_assert(kShift >= 1 && kShift <= 64, "GRET_SALI_SHIFT must be in [1, 64]");
  // A root has two slots a key (sali.h:94, 2361) and a predicted slot above INT_MAX / 2 is clamped to the last one (sali.h:121)
  static constexpr size_t kMaxBulk = size_t(INT_MAX) / 4;

  // shift: kShift, or another value for tests (64 takes the guard away)
  explicit SaliIndex(std::string name, bool checked_put = true, int shift = kShift) : name_(std::move(name)), checked_(checked_put), shift_(shift) {
    if (shift < 1 || shift > 64) throw std::invalid_argument(name_ + ": shift must be in [1, 64]");
    make();
  }
  SaliIndex(const SaliIndex&) = delete;
  SaliIndex& operator=(const SaliIndex&) = delete;
  ~SaliIndex() { destroy(); }  // indexInterface has no virtual destructor: deleted through it, as GRE does, the authors' nodes stay allocated

  void init(Param* = nullptr) override {}

  // kv: sorted by key, keys unique (GRE's contract; checked)
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 1; i < n; ++i)
      if (kv[i].first <= kv[i - 1].first) throw std::invalid_argument(name_ + ": bulk_load needs sorted, unique keys");
    // the keys too close to the key kept below them go to the side store; the rest, in place if that is all of them
    std::vector<std::pair<K, P>> kept;
    std::map<uint64_t, uint64_t> side;
    size_t first_close = n;
    {
      uint64_t below = 0;
      for (size_t i = 0; i < n; ++i) {
        const uint64_t k = kv[i].first;
        if (i && k - below < margin(k)) {
          first_close = i;
          break;
        }
        below = k;
      }
    }
    if (first_close < n) {
      kept.reserve(n - 1);  // one allocation: at 200M keys a vector that grows by doubling would hold two copies for a while
      kept.assign(kv, kv + first_close);
      uint64_t below = kv[first_close - 1].first;
      for (size_t i = first_close; i < n; ++i) {
        const uint64_t k = kv[i].first;
        if (k - below < margin(k)) {
          side.emplace_hint(side.end(), k, uint64_t(kv[i].second));
        } else {
          kept.push_back(kv[i]);
          below = k;
        }
      }
    }
    const std::pair<K, P>* in = first_close < n ? kept.data() : kv;
    const size_t m = first_close < n ? kept.size() : n;
    if (m > kMaxBulk)  // refused before anything is touched: the index is what it was
      throw std::invalid_argument(name_ + ": SALI's slot prediction is an int clamped at INT_MAX / 2: at most " + std::to_string(kMaxBulk) + " keys in a bulk load");
    destroy();
    make();
    side_.swap(side);
    n_sali_ = 0, put_side_ = 0, writes_ = 0, writes_reported_ = 0, rebuilds_ = 0;
    dead_ = false;
    try {
      sali()->bulk_load(in, int(m));  // the authors' bulk load
    } catch (const sali_detail::RtAssert& e) {
      give_up("bulk_load", e);  // their root is gone and not replaced: the object cannot even be destroyed
    } catch (...) {  // out of memory inside their build: the same state
      abandon();
      throw;
    }
    n_sali_ = m;
    const auto t1 = std::chrono::steady_clock::now();
    // every key through the authors' at(): what GRE will ask, checked once
    size_t wrong = 0;
    uint64_t first_wrong = 0;
    for (size_t i = 0; i < m; ++i) {
      P v = P();
      bool found = false;
      try {
        found = sali()->at(in[i].first, v);
      } catch (const sali_detail::RtAssert& e) {
        if (!is_miss(e)) give_up("bulk_load (check)", e);
      }
      if (!found || v != in[i].second)
        if (!wrong++) first_wrong = in[i].first;
    }
    if (wrong) {
      abandon();
      throw std::runtime_error(name_ + ": after the authors' bulk load their at() does not return " + std::to_string(wrong) + " of " + std::to_string(m) +
                               " keys (the first: " + std::to_string(first_wrong) + "); the index holds nothing now");
    }
    const auto t2 = std::chrono::steady_clock::now();
    const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(), check_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const size_t authors = sali()->total_size();
    std::fprintf(stderr,
                 "gret: %s (the SALI authors' code, READ_EVOLVE 0, COMPRESS 0, one thread; put %s; tbb=%s; long double mantissa %d bits, side store for gaps below (key >> %d) + 1) "
                 "n=%zu in_sali=%zu side=%zu (%.4f%%) "
                 "bytes_per_key=%.2f authors_total_size_per_key=%.2f build_ms=%.1f check_ms=%.1f\n",
                 name_.c_str(), checked_ ? "checked" : "unchecked", tbb_name(), int(LDBL_MANT_DIG), shift_, n, n_sali_, side_.size(),
                 n ? 100.0 * double(side_.size()) / double(n) : 0.0,
                 n ? double(memory_consumption()) / double(n) : 0.0, n ? double(authors) / double(n) : 0.0, build_ms, check_ms);
  }

  bool get(K key, P& val, Param* = nullptr) override {
    try {
      if (__builtin_expect(sali()->at(key, val), 1)) return true;  // the authors' lookup
    } catch (const sali_detail::RtAssert& e) {  // their check that the slot holds the key asked for (sali.h:610): another key is there
      if (!is_miss(e)) give_up("get", e);
    }
    return side_get(key, val);
  }

  // true if the key was not there and is now; false, and nothing changes, if it was (saliu: a key stored in SALI throws)
  bool put(K key, P value, Param* = nullptr) override {
    alive();
    if (!side_.empty() && side_.count(key)) return false;
    if (checked_ && sali()->exists(key)) return false;  // the authors' walk without locks: their insert does not survive a stored key
    if (__builtin_expect(margin(key) > 1, 0)) {
      const int c = crowded(key);
      if (c == 2) return false;  // stored (saliu sees it here)
      if (c == 1) {
        side_.emplace(uint64_t(key), uint64_t(value));
        ++put_side_, ++writes_;
        return true;
      }
    }
    try {
      sali()->insert(key, value);  // the authors' insert
    } catch (const sali_detail::RtAssert& e) {
      if (!(is_failed_rebuild(e) && salvage())) give_up(checked_ ? "put" : "put (unchecked: was the key stored already?)", e);
    } catch (const std::bad_array_new_length&) {  // their rebuild asked for a negative number of slots (salvage(), below)
      if (!salvage()) {
        abandon();
        throw std::runtime_error(name_ + ": put: the SALI authors' rebuild inside an insert failed (0 / 0 after removes, sali.h:3197 and :2361) and the keys could not be read "
                                         "back. The index is unusable from here on");
      }
    } catch (...) {  // anything else out of their insert (out of memory): a slot may be left locked, or marked as a child that was never set
      abandon();
      throw;
    }
    ++n_sali_, ++writes_;
    return true;
  }

  bool update(K key, P value, Param* = nullptr) override {
    alive();
    if (!side_.empty()) {
      const auto it = side_.find(key);
      if (it != side_.end()) {
        it->second = uint64_t(value);
        return true;
      }
    }
    if (!sali()->exists(key)) return false;  // theirs would overwrite the payload of whatever key is in the slot (sali.h:864)
    try {
      return sali()->update(key, value);  // the authors' update
    } catch (const sali_detail::RtAssert& e) {
      give_up("update", e);
    }
  }

  bool remove(K key, Param* = nullptr) override {
    alive();
    if (!side_.empty()) {
      const auto it = side_.find(key);
      if (it != side_.end()) {
        side_.erase(it);
        ++writes_;
        return true;
      }
    }
    if (!sali()->exists(key)) return false;  // theirs would erase whatever key is in the slot (sali.h:796)
    bool gone = false;
    try {
      gone = sali()->remove(key);  // the authors' remove
    } catch (const sali_detail::RtAssert& e) {
      give_up("remove", e);
    }
    if (!gone) throw std::logic_error(name_ + ": the authors' remove missed a key their exists() finds");
    --n_sali_, ++writes_;
    return true;
  }

  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    alive();
    if (n == 0) return 0;  // theirs writes one result before it looks at the length
    size_t got = 0;
    try {
      got = size_t(sali()->range_query_len(out, key, int(std::min<size_t>(n, size_t(INT_MAX)))));  // the authors' scan
    } catch (const sali_detail::RtAssert& e) {
      give_up("scan", e);
    }
    if (side_.empty()) return got;
    const auto first = side_.lower_bound(uint64_t(key));
    if (first == side_.end() || (got == n && first->first > uint64_t(out[got - 1].first))) return got;
    // how many of SALI's keys (i) and of the side store's (s) make the first n of the two together, then merged from the back
    size_t i = 0, s = 0;
    auto e = first;
    while (i + s < n && (i < got || e != side_.end()))
      if (e == side_.end() || (i < got && uint64_t(out[i].first) < e->first)) ++i;
      else ++e, ++s;
    const size_t total = i + s;
    for (size_t w = total; s > 0;) {
      const auto last = std::prev(e);
      if (i > 0 && uint64_t(out[i - 1].first) > last->first) out[--w] = out[--i];
      else out[--w] = std::pair<K, P>(K(last->first), P(last->second)), e = last, --s;
    }
    return total;
  }

  // Every byte the index holds, as far as it can be counted from outside: the authors' total_size() (every node reachable from
  // the root with all its slots), the nodes it leaves out (the pool of recycled two-key nodes, and nodes replaced by a rebuild
  // that wait in the epoch lists), the SALI object with its 1024 pool stacks, the side store, this object. Not the
  // allocator's overhead, and not the authors' epoch manager (one small object).
  long long memory_consumption() override {
    alive();
    const size_t authors = sali()->total_size(), pool = pool_bytes(), waiting = waiting_bytes(), side = side_.size() * kMapNode;
    // GRE asks for the size once, after the run, and there is no other hook: after writes, say here what the number is made
    // of (the authors' own GRE wrapper reports total_size alone) and how many keys the side store holds now
    if (writes_ != writes_reported_) {
      writes_reported_ = writes_;
      const size_t all = n_sali_ + side_.size();
      std::fprintf(stderr,
                   "gret: %s now in_sali=%zu side=%zu (%.4f%% of the keys held; %zu puts went to the side store since the bulk load) bytes: authors_total_size=%zu "
                   "waiting_for_an_epoch=%zu two_key_pool=%zu side_store=%zu object=%zu; rebuilt %zu times after a failed rebuild of the authors' since the bulk load, "
                   "set_aside=%zu bytes\n",
                   name_.c_str(), n_sali_, side_.size(), all ? 100.0 * double(side_.size()) / double(all) : 0.0, put_side_, authors, waiting, pool, side, sizeof(*this), rebuilds_,
                   set_aside_);
    }
    return (long long)(sizeof(*this) + authors + pool + waiting + side + set_aside_);
  }

  // for tests and reports
  size_t in_sali() const { return n_sali_; }
  size_t in_side() const { return side_.size(); }
  size_t puts_to_side() const { return put_side_; }
  size_t rebuilds() const { return rebuilds_; }            // since the bulk load: times salvage() replaced the authors' object
  size_t set_aside_bytes() const { return set_aside_; }    // bytes of the objects it set aside, which stay allocated
  size_t authors_total_size() { return sali()->total_size(); }       // what their GRE wrapper reports as the memory
  size_t pool_bytes() { return two_key_nodes(*sali(), nullptr); }    // recycled two-key nodes, and the stacks that hold them
  size_t waiting_bytes() { return waiting_nodes(*sali()); }          // nodes replaced by a rebuild, not yet freed
  Sali& authors() { return *sali(); }                                // their object, for timing their calls directly
  static const char* tbb_name() {
#ifdef GRET_SALI_TBB_STANDIN
    return "stand-in";
#else
    return "real";
#endif
  }

 private:
  static constexpr size_t kMapNode = 4 * sizeof(void*) + 2 * sizeof(uint64_t);  // a red-black tree node holding two uint64

  Sali* sali() noexcept { return std::launder(reinterpret_cast<Sali*>(&store_)); }
  uint64_t margin(uint64_t k) const noexcept { return shift_ >= 64 ? 1 : (k >> shift_) + 1; }

  static bool is_miss(const sali_detail::RtAssert& e) noexcept { return std::strcmp(e.expr, "node->items[pos].comp.data.key == key") == 0; }

  struct Quiet {  // the authors' constructor and destructor report on std::cout
    std::ostringstream sink;
    std::streambuf* old = nullptr;
    Quiet() {
      const char* v = std::getenv("GRET_SALI_VERBOSE");
      if (!v || !*v || *v == '0') old = std::cout.rdbuf(sink.rdbuf());
    }
    ~Quiet() {
      if (old) std::cout.rdbuf(old);
    }
  };

  void make() {
    Quiet quiet;
    ::new (static_cast<void*>(&store_)) Sali(0, true, 0);  // BUILD_LR_REMAIN 0 as their wrapper has it; QUIET spares two printf
    built_ = true;
  }
  void destroy() {
    if (!built_) return;
    built_ = false;
    Quiet quiet;
    try {
      sali()->~Sali();
    } catch (...) {  // a runtime check of theirs failed while it took itself apart: nothing left to do with it
    }
  }
  // The authors' object is in a state their own destructor cannot be run on: it is left where it is (its nodes stay allocated),
  // an empty one takes its place, and every later call throws.
  void abandon() {
    make();
    side_.clear();
    n_sali_ = 0;
    dead_ = true;
  }
  [[noreturn]] void give_up(const char* op, const sali_detail::RtAssert& e) {
    abandon();
    throw std::runtime_error(name_ + ": " + op + ": a runtime check of the SALI authors' failed (" + e.file + ":" + std::to_string(e.line) + ", `" + e.expr +
                             "`), where their code prints that and exits with status 0. The index is unusable from here on");
  }
  void alive() const {
    if (__builtin_expect(dead_, 0)) throw std::logic_error(name_ + ": unusable after a failed runtime check of the SALI authors' (see the first error)");
  }

  // The one failure of the authors' insert the wrapper repairs. When an insert meets a stored key, their code may rebuild a
  // subtree on the path, and the new nodes' sizes use speed / prev_speed, where speed is (keys in the subtree now - keys when
  // it was built) / time (sali.h:3192, 3197). With removes in between, a subtree can hold exactly as many keys as when it was
  // built: speed is 0, the rebuilt nodes inherit 0, and the next such rebuild below divides 0 by 0. The NaN reaches the slot
  // count through a cast to int (sali.h:2361, undefined): INT_MIN on x86-64, where new_items(INT_MIN) throws
  // std::bad_array_new_length out of their insert; 0 on arm64, where the check `node->model.a1 >= 0` (sali.h:2435) fails for
  // a subtree of four keys or more. In the authors' own build either ends the process. Inserts alone cannot cause it (a
  // subtree that only takes keys has grown at every rebuild), nor can removes alone (only an insert rebuilds); GRE never has
  // both in a run (benchmark.h: INVARIANT(insert_delete == insert_ratio || insert_delete == delete_ratio)).
  // At that point the key is in (the insert comes before the rebuild), and the subtree that was being rebuilt is still linked
  // from its parent and complete, but every slot of it is locked for good and its nodes are on a free list: a lookup there
  // spins for ever, and their destructor would free it twice. Their scan takes no lock and still reads everything. So: every
  // key is read out with their scan, their object is set aside as it is (its bytes stay allocated and are counted in
  // memory_consumption from here on), and a new one is bulk loaded from the keys and checked. False, and the caller gives the
  // index up, unless exactly the keys expected came back.
  static bool is_failed_rebuild(const sali_detail::RtAssert& e) noexcept { return std::strcmp(e.expr, "node->model.a1 >= 0") == 0; }
  bool salvage() noexcept {
    const size_t expect = n_sali_ + 1;  // the caller has not counted the new key yet
    if (expect > kMaxBulk) return false;
    try {
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<std::pair<K, P>> all(expect + 1);
      if (size_t(sali()->range_query_len(all.data(), K(0), int(expect + 1))) != expect) return false;
      for (size_t i = 1; i < expect; ++i)
        if (all[i].first <= all[i - 1].first) return false;
      const size_t held = sali()->total_size() + pool_bytes();  // not the nodes of earlier rebuilds that were still waiting for an epoch
      make();  // in the place of the old object, which is not destroyed
      set_aside_ += held;
      sali()->bulk_load(all.data(), int(expect));
      for (size_t i = 0; i < expect; ++i) {
        P v = P();
        if (!sali()->at(all[i].first, v) || v != all[i].second) return false;
      }
      ++rebuilds_;
      std::fprintf(stderr,
                   "gret: %s: the authors' rebuild inside an insert failed (0 / 0 after removes: docs/sota/sali.md); their object was set aside (%zu bytes stay allocated) "
                   "and a new one bulk loaded from their scan: %zu keys, %.1f ms. This has happened %zu times since the bulk load\n",
                   name_.c_str(), held, expect, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), rebuilds_);
      return true;
    } catch (...) {
      return false;
    }
  }

  __attribute__((noinline)) bool side_get(K key, P& val) {
    alive();
    if (side_.empty()) return false;
    const auto it = side_.find(uint64_t(key));
    if (it == side_.end()) return false;
    val = P(it->second);
    return true;
  }

  // For a key with a margin above 1 that is not in the side store: 0 if it keeps its margin to the keys SALI holds next to it,
  // 1 if not, 2 if SALI holds the key itself. One call of the authors' scan: the first key at or above key - margin + 1.
  int crowded(uint64_t key) {
    const uint64_t m = margin(key), lo = key >= m - 1 ? key - (m - 1) : 0;
    std::pair<K, P> one;
    int got = 0;
    try {
      got = sali()->range_query_len(&one, K(lo), 1);
    } catch (const sali_detail::RtAssert& e) {
      give_up("put (neighbours)", e);
    }
    if (got == 0) return 0;
    const uint64_t f = one.first;
    if (f == key) return 2;
    if (f < key) return 1;
    return f - key < margin(f) ? 1 : 0;
  }

  // ---- what total_size() leaves out. Sali64 is Sali (the static_asserts above).
  template <class S>
  static size_t two_key_nodes(S& s, size_t* count) {
    auto& pool = s.*sali_member(sali_detail::PoolTag{});  // std::stack<Node*>[1024], one per OpenMP thread number
    using NodePtr = typename std::remove_reference<decltype(pool[0].top())>::type;
    using Node = typename std::remove_pointer<NodePtr>::type;
    using Item = typename std::remove_reference<decltype(*std::declval<Node&>().items)>::type;
    size_t nodes = 0, b = 0;
    for (auto& st : pool) {
      nodes += st.size();
      b += deque_bytes(st.size());
    }
    if (count) *count = nodes;
    return b + nodes * (sizeof(Node) + 8 * sizeof(Item));
  }
  // the heap a std::deque of pointers holds for n elements (the standard library's layout, not measured)
  static size_t deque_bytes(size_t n) {
#if defined(__GLIBCXX__)
    return 8 * sizeof(void*) + (n / 64 + 1) * 512;  // a map of 8 pointers and 512-byte blocks, one of them from the start
#else
    return n ? ((n + 511) / 512) * 4096 + 2 * sizeof(void*) : 0;  // libc++: 4096-byte blocks, none while empty
#endif
  }
  template <class S>
  static size_t waiting_nodes(S& s) {
    size_t b = 0;
    auto add = [&b](const auto& lists) {
      for (const auto& list : lists) {
        b += list.capacity() * sizeof(void*);
        for (const auto* node : list) b += sizeof(*node) + size_t(node->is_two ? 8 : node->num_items) * sizeof(*node->items);
      }
    };
    if (!s.ebr) return 0;
    add(s.ebr->gFreeLists);  // filled by the compression thread only (COMPRESS 0: empty)
    for (const auto& info : s.ebr->mThreadSpecificInformations) add(info.*sali_member(sali_detail::WaitTag{}));
    return b;
  }

  std::string name_;
  bool checked_;                       // put asks the authors' exists() first
  int shift_;
  bool built_ = false, dead_ = false;
  size_t n_sali_ = 0;                  // keys in the authors' index
  size_t put_side_ = 0;                // puts since the bulk load that went to the side store
  size_t writes_ = 0;                  // puts and removes done since the bulk load
  size_t writes_reported_ = 0;         // its value when memory_consumption last printed its line
  size_t rebuilds_ = 0;                // since the bulk load: times salvage() replaced the authors' object
  size_t set_aside_ = 0;               // bytes of the authors' objects salvage() set aside (never freed)
  std::map<uint64_t, uint64_t> side_;  // the keys the authors' long double models may not tell from a neighbour
  typename std::aligned_storage<sizeof(Sali), alignof(Sali)>::type store_;  // the authors' index, in place: no pointer to follow
};

// "sali" (put checked) and "saliu" (put unchecked, as the authors' GRE wrapper); nullptr for any other name, and for key or
// payload types other than uint64
template <class K, class P>
indexInterface<K, P>* make_sali(const std::string& name) {
  if constexpr (!(std::is_integral<K>::value && std::is_unsigned<K>::value && sizeof(K) == 8 && std::is_integral<P>::value && std::is_unsigned<P>::value && sizeof(P) == 8)) {
    (void)name;
    return nullptr;
  } else {
    if (name == "sali") return new SaliIndex<K, P>(name, true);
    if (name == "saliu") return new SaliIndex<K, P>(name, false);
    return nullptr;
  }
}

}  // namespace gret
