// SPDX-License-Identifier: MIT
// DiliIndex<K, P>: DILI (Li, Lu, Zhu, Ding, Yang, Pan: "DILI: A Distribution-Driven Learned Index", PVLDB 16(9), 2023) in GRE's
// harness. This is a WRAPPER around the authors' own code (https://github.com/pfl-cs/DILI, used unmodified and included by path:
// -I <DILI>/src), not a reimplementation. Lookups run the authors' DILI::search, inserts DILI::insert, removals DILI::erase, the
// bulk load DILI::bulk_load (BU-tree, then the top-down build). What the wrapper adds is what the authors' code lacks or gets wrong
// outside the data it was written for (docs/sota/dili.md has the list with file and line):
//   - uint64 keys. The authors' key is a signed long >= 0 (negative keys are their slot markers). Keys below 2^63 go to one DILI,
//     keys from 2^63 to a second one (minus 2^63); the top bit picks the instance.
//   - a side store (std::map) for what the authors' code cannot hold exactly: the payload 2^64 - 1 (their "not found" value), the
//     keys 2^63 - 1 and 2^64 - 1 (they compute max + 1), keys closer to a neighbour than their double models separate
//     (gap < (key >> 50) + 1: only keys above 2^50 can be), keys whose predicted slot leaves the int range at some node (the cast
//     is undefined: arm64 saturates, x86 gives slot 0 and breaks the key order), keys the bulk load lost, and a half of the key
//     space with fewer keys than the authors' bulk loader needs. The line bulk_load prints says how many keys are there: a run in
//     which that number is not small is not a measurement of DILI.
//   - scan, as an in-order walk over the authors' nodes (their range_query returns record pointers only, into a buffer of unknown
//     size, and has two defects); update, in place; memory_consumption, as the bytes the nodes hold (their total_size() leaves
//     out the empty slots of the leaves).
//   - a scratch directory for the bulk load (the authors' loader writes two files to a "mirror" directory and would read stale
//     ones from it), created under $DILI_MIRROR_DIR or the system's temporary directory and removed afterwards; std::cout
//     silenced while the loader runs ($DILI_VERBOSE=1 keeps it).
//   - bulk_load throws std::invalid_argument when neither half of the key space has kMinBulk (2,000,000) keys: below that the
//     authors' loader reads out of bounds. It verifies every key afterwards, and the key order, and throws std::runtime_error if
//     the authors' structure cannot be made to answer exactly.
// Names: dili (the authors' defaults as released) and dili<B>, B in [2, 200]: the authors' buMinFan, the fewest keys the
// BU-tree gives a leaf (default 16), which bounds the number of leaves at n / B. An optional r<PP> after that sets the cost
// model's decay rate rho to PP / 100 (PP in [1, 99]): the released code has RHO = 0.1, the paper's section 7.1 says 0.2, so
// dilir20 (= dili16r20) is the paper's stated setting and dili the released one. Their other constants are compiled in.
//
// The authors' sources are not header-only and share global state (retrain buffers, the BU-tree's static data pointer, the
// knobs): one bulk load at a time, single-threaded. They are compiled here, inside namespace gret::dili_authors, as part of the
// translation unit that includes this header, so GRE's build needs the include path and nothing else. If two translation units
// of one program include this header, define GRET_DILI_DECL_ONLY in all but one. Two more things are done to them, neither by
// editing: butree/interval.h and interval.cpp are included with 'const' defined away, because interval::data is declared
// const long* and passed to functions taking long* (the authors build with g++ -fpermissive, which only warns); and the macros
// the sources leave behind are removed (KEY_TYPE and PAYLOAD_TYPE among them, GRE's template parameter names).
// DILI's headers include <linux/perf_event.h> and <asm/unistd.h>: on macOS, put two empty files with those names in a
// directory on the include path.
#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if !__has_include(<linux/perf_event.h>) || !__has_include(<asm/unistd.h>)
#error "gret/sota/dili.h: DILI's sources include <linux/perf_event.h> and <asm/unistd.h>; off Linux, add a directory holding two empty files with those names to the include path (docs/sota/dili.md)"
#endif
#include <linux/perf_event.h>
#include <asm/unistd.h>

// ---- the authors' sources, unmodified, in a namespace of their own. Every standard header they include is included above,
// so inside the namespace those includes are no-ops; <cassert> is the one that is read again, and it only redefines assert.
#pragma push_macro("KEY_TYPE")
#pragma push_macro("PAYLOAD_TYPE")
namespace gret {
namespace dili_authors {
#include "global/global.h"
#include "butree/linearRegressor.h"
#define const  // see the head of this file: interval::data and interval::probs lose their const, as -fpermissive lets them
#include "butree/interval.h"
#undef const
#include "dili/DILI.h"
#ifndef GRET_DILI_DECL_ONLY
#include "global/global.cpp"
#include "global/linearReg.cpp"
#include "utils/data_utils.cpp"
#include "utils/file_utils.cpp"
#include "butree/linearRegressor.cpp"
#define const
#include "butree/interval.cpp"
#undef const
#include "butree/buInterval.cpp"
#include "butree/interval_utils.cpp"
#include "dili/diliNode.cpp"
#include "dili/DILI.cpp"
// utils/linux_sys_utils.cpp (perf_event_open, timing) is left out: nothing above calls it, and it is Linux-only
#endif
enum : int { gret_leaf_max = LEAF_MAX_CAPACIY };
}  // namespace dili_authors
}  // namespace gret
#undef LR_PRED
#undef MIN_INT
#undef MAX_INT
#undef MIN_LONG
#undef MAX_LONG
#undef MIN_DOUBLE
#undef MAX_DOUBLE
#undef LEAF_MAX_CAPACIY
#undef minFan
#undef MIN_KEY
#pragma pop_macro("PAYLOAD_TYPE")
#pragma pop_macro("KEY_TYPE")

namespace gret {

template <class K, class P>
class DiliIndex final : public indexInterface<K, P> {
  using D = dili_authors::DILI;
  using Node = dili_authors::diliNode;
  using Entry = dili_authors::pairEntry;
  using Fan2 = dili_authors::fan2Leaf;
  static_assert(std::is_integral<K>::value && std::is_unsigned<K>::value && sizeof(K) == 8, "DiliIndex: keys are uint64");
  static_assert(std::is_integral<P>::value && sizeof(P) == 8, "DiliIndex: payloads are 64-bit integers");
  static_assert(sizeof(long) == 8, "DILI's key and record pointer are long: 64 bits here");
  // DILI::root is private and is its first member: a standard-layout class is pointer-interconvertible with it
  static_assert(std::is_standard_layout<D>::value, "DILI is no longer standard-layout: root_of() needs another way in");

  struct Frame {
    Node* n;
    int i;
  };

 public:
  static constexpr size_t kMinBulk = 2000000;  // below this the authors' loader reads out of bounds (interval_utils.cpp:1024, 1119)
  static constexpr long kTop = LONG_MAX;       // never a DILI key: the authors compute max + 1
  static constexpr long kNone = -1;            // DILI::search's "not found", so never a DILI record pointer

  explicit DiliIndex(std::string name, int bu_min_fan = 16, double rho = 0.1) : name_(std::move(name)), bu_min_fan_(bu_min_fan), rho_(rho) {
    if (bu_min_fan < 2 || bu_min_fan > 200) throw std::invalid_argument(name_ + ": buMinFan must be in [2, 200]");
    if (!(rho > 0.0 && rho < 1.0)) throw std::invalid_argument(name_ + ": rho must be in (0, 1)");
  }
  DiliIndex(const DiliIndex&) = delete;
  DiliIndex& operator=(const DiliIndex&) = delete;
  ~DiliIndex() { reset(); }  // indexInterface has no virtual destructor: deleted through it, as GRE does, the authors' nodes stay allocated

  void init(Param* = nullptr) override {}

  // kv: sorted by key, keys unique (GRE's contract; checked)
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    reset();
    for (size_t i = 1; i < n; ++i)
      if (uint64_t(kv[i].first) <= uint64_t(kv[i - 1].first)) throw std::invalid_argument(name_ + ": bulk_load needs sorted, unique keys");
    const size_t cut = size_t(std::partition_point(kv, kv + n, [](const std::pair<K, P>& p) { return (uint64_t(p.first) >> 63) == 0; }) - kv);  // the first key >= 2^63
    try {
      load_half(0, kv, cut);
      load_half(1, kv + cut, n - cut);
    } catch (...) {
      reset();
      throw;
    }
    if (!built_[0] && !built_[1]) {
      const std::string counts = std::to_string(cut) + " and " + std::to_string(n - cut) + " keys given, of which " + std::to_string(why_[0]) + " unrepresentable, " +
                                 std::to_string(why_[1]) + " too close, " + std::to_string(why_[2]) + " past the int range, " + std::to_string(why_[3]) + " lost by the loader";
      reset();
      throw std::invalid_argument(name_ + ": the authors' bulk loader needs at least " + std::to_string(kMinBulk) + " keys it can hold below 2^63 or from 2^63 up; got " + counts);
    }
    side_reported_ = side_.size();  // the line below says it; memory_consumption speaks up when it has changed
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const size_t authors = (built_[0] ? dili_[0].total_size() : 0) + (built_[1] ? dili_[1].total_size() : 0);
    std::fprintf(stderr,
                 "gret: %s n=%zu in_dili=%zu+%zu side=%zu (unrepresentable=%zu close=%zu int_range=%zu lost=%zu small_half=%zu) buMinFan=%d rho=%.2f rebuilds=%zu "
                 "bytes_per_key=%.2f authors_total_size_per_key=%.2f build_ms=%.1f\n",
                 name_.c_str(), n, n_dili_[0], n_dili_[1], side_.size(), why_[0], why_[1], why_[2], why_[3], why_[4], bu_min_fan_, rho_, rebuilds_,
                 n ? double(memory_consumption()) / double(n) : 0.0, n ? double(authors) / double(n) : 0.0, ms);
  }

  bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    const unsigned h = unsigned(x >> 63);
    if (__builtin_expect(built_[h], 1)) {
      const long r = dili_[h].search(long(x & kLow));  // the authors' lookup
      if (__builtin_expect(r != kNone, 1)) {
        val = P(uint64_t(r));
        return true;
      }
    }
    if (side_.empty()) return false;
    const auto it = side_.find(x);
    if (it == side_.end()) return false;
    val = P(it->second);
    return true;
  }

  // true if the key was not there and is now; false, and nothing changes, if it was
  bool put(K key, P value, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    const unsigned h = unsigned(x >> 63);
    const long k = long(x & kLow);
    if (!side_.empty() && side_.count(x)) return false;
    if (built_[h]) {
      const int w = walk(h, k);
      if (w == 1 || w == -2) return false;
      if (w < 0 && dili_[h].search(k) != kNone) return false;  // the authors' search is the judge of what is there
      if (w == 0 && uint64_t(value) != ~uint64_t(0) && k != kTop && spaced(k) && insert_checked(h, k, long(uint64_t(value)))) {
        ++n_dili_[h];
        return true;
      }
    }
    ++put_side_;  // a key the authors' insert was not given
    side_.emplace(x, uint64_t(value));
    return true;
  }

  bool update(K key, P value, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    const unsigned h = unsigned(x >> 63);
    const long k = long(x & kLow);
    if (!side_.empty()) {
      const auto it = side_.find(x);
      if (it != side_.end()) {
        it->second = uint64_t(value);
        return true;
      }
    }
    if (!built_[h]) return false;
    long* where = nullptr;
    const int w = walk(h, k, &where);
    if (w != 1 && w != -2) return false;
    if (uint64_t(value) != ~uint64_t(0)) {
      *where = long(uint64_t(value));
      return true;
    }
    if (!dili_[h].erase(k)) throw std::logic_error(name_ + ": the authors' erase missed a key that is there");
    --n_dili_[h];
    side_.emplace(x, uint64_t(value));  // their "not found" value as a payload: only the side store can hold it
    return true;
  }

  bool remove(K key, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    const unsigned h = unsigned(x >> 63);
    if (!side_.empty()) {
      const auto it = side_.find(x);
      if (it != side_.end()) {
        side_.erase(it);
        return true;
      }
    }
    if (!built_[h] || !dili_[h].erase(long(x & kLow))) return false;  // the authors' erase
    --n_dili_[h];
    return true;
  }

  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    size_t got = 0;
    auto sit = side_.empty() ? side_.end() : side_.lower_bound(x);
    for (unsigned h = unsigned(x >> 63); h < 2 && got < n; ++h) {
      if (!built_[h]) continue;
      const uint64_t base = uint64_t(h) << 63;
      int sub = 0;
      bool ok = x > base ? seek(h, long(x - base), scan_, sub) : first(h, scan_, sub);
      while (ok && got < n) {
        const Entry& e = scan_.back().n->pe_data[scan_.back().i];
        const long k = e.key >= 0 ? e.key : sub ? e.fan2child->k2 : e.fan2child->k1;
        const long p = e.key >= 0 ? e.ptr : sub ? e.fan2child->p2 : e.fan2child->p1;
        const uint64_t uk = base | uint64_t(k);
        for (; sit != side_.end() && sit->first < uk && got < n; ++sit) out[got++] = std::pair<K, P>(K(sit->first), P(sit->second));
        if (got == n) break;
        out[got++] = std::pair<K, P>(K(uk), P(uint64_t(p)));
        ok = next(scan_, sub);
      }
    }
    for (; sit != side_.end() && got < n; ++sit) out[got++] = std::pair<K, P>(K(sit->first), P(sit->second));
    return got;
  }

  // every byte reachable from the index: the authors' nodes with all their slots, their two-key leaves, their retrain buffers,
  // the side store's tree nodes, this object. Not the allocator's overhead, and not what the authors' erase leaks.
  long long memory_consumption() override {
    size_t b = sizeof(*this) + side_.size() * kMapNode + (path_.capacity() + tmp_.capacity() + scan_.capacity()) * sizeof(Frame);
    if (dili_authors::dili_auxiliary::retrain_keys) b += size_t(4) * dili_authors::gret_leaf_max * sizeof(long);
    for (int h = 0; h < 2; ++h)
      if (built_[h]) b += tree_bytes(root_of(h), false);
    // GRE asks for the size once, after the run, and there is no other hook (it deletes through indexInterface, which has no
    // virtual destructor): say here how many keys the side store holds now, whenever that changed since the last line
    if ((built_[0] || built_[1]) && side_.size() != side_reported_) {
      side_reported_ = side_.size();
      const size_t all = n_dili_[0] + n_dili_[1] + side_.size();
      std::fprintf(stderr, "gret: %s now in_dili=%zu side=%zu (%.3f%% of the keys held; %zu puts went to the side store since the bulk load)\n", name_.c_str(),
                   n_dili_[0] + n_dili_[1], side_.size(), all ? 100.0 * double(side_.size()) / double(all) : 0.0, put_side_);
    }
    return (long long)b;
  }

  // for tests and reports
  size_t in_dili() const { return n_dili_[0] + n_dili_[1]; }
  size_t in_side() const { return side_.size(); }
  size_t puts_to_side() const { return put_side_; }  // puts since the bulk load that the authors' insert was not given
  bool half_built(int h) const { return built_[h]; }

 private:
  static constexpr uint64_t kLow = ~uint64_t(0) >> 1;
  static constexpr size_t kMapNode = 4 * sizeof(void*) + 2 * sizeof(uint64_t);  // a red-black tree node holding two uint64

  // the gap a key must keep to the key below it for the authors' double models to tell them apart in every node they may build
  // (three keys in one slot get slope 2 / gap, and the product slope * key must stay exact to a quarter of a slot)
  static long margin(long q) { return (q >> 50) + 1; }

  Node* root_of(int h) { return *reinterpret_cast<Node**>(&dili_[h]); }

  // The authors' LR_PRED(a, b, key, fanout) = min(max(int(a + b * key), 0), fanout - 1), with the cast kept inside the int
  // range. Below the range and for NaN (a root over a single child has b = 1 / 0) both arm64 and x86 end in slot 0. Above it
  // arm64 saturates to the last slot and x86 wraps to slot 0, which puts a large key before the small ones: far is set, the
  // last slot returned, and no such key is ever given to the authors' insert.
  static int slot_of(const Node* n, long key, bool& far) {
    const double p = n->a + n->b * key;
    if (p >= 2147483648.0) {
      far = true;
      return n->fanout - 1;
    }
    if (p > -2147483648.0) return std::min<int>(std::max<int>(static_cast<int>(p), 0), n->fanout - 1);
    return 0;
  }

  // k's path through the authors' nodes into path_ (the last frame is the slot where it is or would go). 1: there, *where its
  // record pointer; 0: not there. Negative when a prediction on the path left the int range (see slot_of): -2 there (never, for
  // a key this wrapper let in), -1 not there.
  int walk(int h, long k, long** where = nullptr) {
    path_.clear();
    local_ = true;
    Node* n = root_of(h);
    bool far = false, far2 = false;
    // a key that would break k's margin lies in [lo, hi] (see spaced): whether both ends stay in k's slot all the way down
    const bool large = k >= (long(1) << 50) - 1;
    const long m = margin(k), lo = k - m + 1, hi = k > kTop - m - 1 ? kTop : k + m + 1;
    for (;;) {
      const int i = slot_of(n, k, far);
      if (large) local_ &= (slot_of(n, lo, far2) == i) & (slot_of(n, hi, far2) == i);
      path_.push_back(Frame{n, i});
      Entry& e = n->pe_data[i];
      if (e.key == k) {
        if (where) *where = &e.ptr;
        return far ? -2 : 1;
      }
      if (e.key == -1) {
        n = e.child;
        continue;
      }
      if (e.key == -2) {
        Fan2* f = e.fan2child;
        if (f->k1 == k || f->k2 == k) {
          if (where) *where = f->k1 == k ? &f->p1 : &f->p2;
          return far ? -2 : 1;
        }
      }
      return far ? -1 : 0;
    }
  }

  // whether k (absent; path_ ends at its slot) keeps the margin to the keys stored next to it
  bool spaced(long k) {
    if (k < (long(1) << 50) - 1) return true;  // every gap is at least 1, and a key from 2^50 up is 2 away or more
    const Entry& e = path_.back().n->pe_data[path_.back().i];
    long below = -1, above = -1;
    if (e.key >= 0) (e.key < k ? below : above) = e.key;
    else if (e.key == -2) {
      const Fan2* f = e.fan2child;
      if (k < f->k1) above = f->k1;
      else if (k < f->k2) below = f->k1, above = f->k2;
      else below = f->k2;
    }
    // A key that breaks the margin lies in [k - margin(k) + 1, k + margin(k) + 1]. If both ends of that range fell in k's own
    // slot at every node of the path (walk noted it in local_; slots are in key order), such a key can only be in the last
    // slot, which is e: no need to look for the neighbours. Otherwise (k next to a slot boundary) they are looked up.
    if (local_) return (below < 0 || k - below >= margin(k)) && (above < 0 || above - k >= margin(above));
    if (below < 0) {
      tmp_ = path_;
      --tmp_.back().i;
      if (settle_back(tmp_)) {
        const Entry& b = tmp_.back().n->pe_data[tmp_.back().i];
        below = b.key >= 0 ? b.key : b.fan2child->k2;
      }
    }
    if (above < 0) {
      tmp_ = path_;
      ++tmp_.back().i;
      if (settle(tmp_)) {
        const Entry& a = tmp_.back().n->pe_data[tmp_.back().i];
        above = a.key >= 0 ? a.key : a.fan2child->k1;
      }
    }
    return (below < 0 || k - below >= margin(k)) && (above < 0 || above - k >= margin(above));
  }

  // three keys a < b < c in one slot become a node whose slope is 2 / (the smaller gap) (put_three_keys, diliNode.h:408-415):
  // c's slot is 2.5 + 2 (c - b) / (b - a), past the int range from a gap ratio of 2^30. x86 then puts c in slot 0, over a.
  static bool wraps(long a, long b, long c) { return ((c - b) >> 29) >= b - a; }

  // The authors' insert for k (absent; path_ ends at its slot), kept away from that. false: not inserted, nothing changed.
  bool insert_checked(int h, long k, long v) {
    const Entry& e = path_.back().n->pe_data[path_.back().i];
    if (e.key == -2) {  // the two-key leaf there becomes a three-key node
      const long k1 = e.fan2child->k1, k2 = e.fan2child->k2;
      if (k < k1 ? wraps(k, k1, k2) : k < k2 ? wraps(k1, k, k2) : wraps(k1, k2, k)) return false;
    }
    // A node above the last one that is not flagged internal retrains as this insert returns through it if its traversal count
    // per key has doubled (diliNode.h:406, 822): its keys are spread again and regrouped. Which nodes may do so is decided here
    // from their counters, with the count this insert adds bounded from above. If three neighbours among the keys of the
    // topmost such node could wrap as above, the nodes are flagged internal instead (as the authors' insert does to them right
    // after) and do not retrain; otherwise the keys are remembered and checked after a retrain. The same is done when such a
    // node holds more keys than the authors' retrain buffers (2 * LEAF_MAX_CAPACIY, diliNode.cpp:15; collect_and_clear,
    // diliNode.h:823, writes every key of the node into them): the bulk load makes leaves far larger than that on skewed data
    // (they are equal-width slices of their parent's range), and nothing in the authors' insert keeps such a leaf from
    // retraining. Flagged internal, it takes the key through the authors' insert like any other node.
    snap_.clear();
    size_t top = path_.size();
    bool big = false;
    long delta = 3;  // the most the last node adds to its total_n_travs (diliNode.h:808, 837, 878)
    for (size_t j = path_.size() - 1; j-- > 0;) {
      const Node* x = path_[j].n;
      delta += 1;
      if (x->is_internal()) continue;
      const long m = long(x->num_nonempty) + 1;
      if ((x->total_n_travs + delta) * x->last_nn >= ((x->last_total_n_travs * m) << 1)) {
        top = j;
        if (m + 2 > 2 * long(dili_authors::gret_leaf_max)) big = true;  // it will not retrain: delta stays what the insert adds
        else delta = m * m + (x->total_n_travs < 0 ? -x->total_n_travs : 0);  // rebuilt, a node of m keys counts at most m * depth
      }
    }
    if (top < path_.size()) {
      bool hazard = big, placed = false;
      long p2 = -1, p1 = -1;
      auto feed = [&](long x) {
        if (p2 >= 0 && wraps(p2, p1, x)) hazard = true;
        p2 = p1, p1 = x;
      };
      tmp_.clear();
      if (!big) tmp_.push_back(Frame{path_[top].n, 0});
      int sub = 0;
      for (bool ok = settle(tmp_); ok; ok = next(tmp_, sub)) {
        const Entry& t = tmp_.back().n->pe_data[tmp_.back().i];
        const long key = t.key >= 0 ? t.key : sub ? t.fan2child->k2 : t.fan2child->k1;
        if (!placed && key > k) feed(k), placed = true;
        feed(key);
        snap_.emplace_back(key, t.key >= 0 ? t.ptr : sub ? t.fan2child->p2 : t.fan2child->p1);
      }
      if (!placed) feed(k);
      if (hazard) {
        for (size_t j = top; j + 1 < path_.size(); ++j) path_[j].n->set_int_flag();
        snap_.clear();
        top = path_.size();
      }
    }
    if (!dili_authors::dili_auxiliary::retrain_keys) dili_authors::dili_auxiliary::init_insert_aux_vars();  // freed by any DILI's destructor
    const long retrains = dili_authors::num_adjust_stats;
    if (!dili_[h].insert(k, v)) throw std::logic_error(name_ + ": the authors' insert refused a key that is not there");
    if (dili_authors::num_adjust_stats != retrains) {
      if (top == path_.size()) throw std::logic_error(name_ + ": the authors' insert retrained a node the wrapper did not expect to");
      snap_.emplace_back(k, v);
      for (const std::pair<long, long>& kp : snap_) {
        long* w = nullptr;
        if (walk(h, kp.first, &w) != 1 || *w != kp.second) throw std::runtime_error(name_ + ": the authors' retrain lost or misplaced a key (diliNode.h:822-826)");
      }
    }
    return true;
  }

  // ---- the keys of one half in ascending order: a stack of (node, slot), resting on a slot that holds one key or a two-key leaf
  static bool settle(std::vector<Frame>& st) {  // forward from the top frame's slot to the first slot with a key
    while (!st.empty()) {
      Frame& f = st.back();
      if (!f.n->pe_data || f.i >= f.n->fanout) {
        st.pop_back();
        if (!st.empty()) ++st.back().i;
        continue;
      }
      const Entry& e = f.n->pe_data[f.i];
      if (e.key >= 0 || e.key == -2) return true;
      if (e.key == -1) {
        Node* c = e.child;
        st.push_back(Frame{c, 0});
        continue;
      }
      ++f.i;
    }
    return false;
  }
  static bool settle_back(std::vector<Frame>& st) {  // backward, to the last slot with a key
    while (!st.empty()) {
      Frame& f = st.back();
      if (!f.n->pe_data || f.i < 0) {
        st.pop_back();
        if (!st.empty()) --st.back().i;
        continue;
      }
      const Entry& e = f.n->pe_data[f.i];
      if (e.key >= 0 || e.key == -2) return true;
      if (e.key == -1) {
        Node* c = e.child;
        st.push_back(Frame{c, c->fanout - 1});
        continue;
      }
      --f.i;
    }
    return false;
  }
  bool first(int h, std::vector<Frame>& st, int& sub) {
    st.clear();
    st.push_back(Frame{root_of(h), 0});
    sub = 0;
    return settle(st);
  }
  bool seek(int h, long k, std::vector<Frame>& st, int& sub) {  // the first key >= k
    st.clear();
    sub = 0;
    Node* n = root_of(h);
    bool far = false;
    for (;;) {
      const int i = slot_of(n, k, far);
      st.push_back(Frame{n, i});
      const Entry& e = n->pe_data[i];
      if (e.key == -1) {
        n = e.child;
        continue;
      }
      if (e.key >= k) return true;
      if (e.key == -2) {
        if (e.fan2child->k1 >= k) return true;
        if (e.fan2child->k2 >= k) {
          sub = 1;
          return true;
        }
      }
      break;
    }
    ++st.back().i;  // slots are in key order: everything from the next slot on is larger
    return settle(st);
  }
  static bool next(std::vector<Frame>& st, int& sub) {
    if (st.back().n->pe_data[st.back().i].key == -2 && sub == 0) {
      sub = 1;
      return true;
    }
    sub = 0;
    ++st.back().i;
    return settle(st);
  }

  // bytes of a subtree; with free_fan2, also frees the two-key leaves, which the authors' destructor leaves behind
  static size_t tree_bytes(Node* root, bool free_fan2) {
    size_t b = 0;
    std::vector<Node*> todo(1, root);
    while (!todo.empty()) {
      Node* n = todo.back();
      todo.pop_back();
      b += sizeof(Node);
      if (!n->pe_data) continue;
      b += size_t(n->fanout) * sizeof(Entry);
      for (int i = 0; i < n->fanout; ++i) {
        Entry& e = n->pe_data[i];
        if (e.key == -1) todo.push_back(e.child);
        else if (e.key == -2) {
          b += sizeof(Fan2);
          if (free_fan2) {
            delete e.fan2child;
            e.setNull();
          }
        }
      }
    }
    return b;
  }

  void reset() {
    for (int h = 0; h < 2; ++h) {
      if (built_[h]) tree_bytes(root_of(h), true);
      dili_[h].clear();  // deletes the nodes
      built_[h] = false;
      n_dili_[h] = 0;
    }
    side_.clear();
    for (size_t& w : why_) w = 0;
    rebuilds_ = 0;
    put_side_ = 0;
    side_reported_ = 0;
  }

  struct Quiet {  // the authors' loader reports its steps on std::cout
    std::ostringstream sink;
    std::streambuf* old = nullptr;
    Quiet() {
      const char* v = std::getenv("DILI_VERBOSE");
      if (!v || !*v || *v == '0') old = std::cout.rdbuf(sink.rdbuf());
    }
    ~Quiet() {
      if (old) std::cout.rdbuf(old);
    }
  };
  struct ScratchDir {  // the authors' "mirror" directory: new and empty, or their loader restores another data set's layout
    std::string path;
    ScratchDir() {
      namespace fs = std::filesystem;
      const char* e = std::getenv("DILI_MIRROR_DIR");
      std::error_code ec;
      const fs::path base = e && *e ? fs::path(e) : fs::temp_directory_path(ec);
      if (ec || base.empty()) throw std::runtime_error("dili: no temporary directory for the authors' bulk load (set DILI_MIRROR_DIR)");
      static unsigned long seq = 0;
      for (int t = 0; t < 64 && path.empty(); ++t) {
        const fs::path p = base / ("gret_dili_" + std::to_string((long long)getpid()) + "_" +
                                   std::to_string((long long)std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(seq++));
        if (fs::create_directories(p, ec) && !ec) path = p.string();  // true only for a directory that was not there
      }
      if (path.empty()) throw std::runtime_error("dili: cannot create a scratch directory under '" + base.string() + "' (set DILI_MIRROR_DIR)");
    }
    ~ScratchDir() {
      std::error_code ec;
      std::filesystem::remove_all(path, ec);
    }
  };

  // one half of the key space: kv[0..m) all below 2^63 (h = 0) or all from 2^63 up (h = 1)
  void load_half(int h, const std::pair<K, P>* kv, size_t m) {
    if (m == 0) return;
    const uint64_t base = uint64_t(h) << 63;
    dili_authors::keyArray keys = std::make_unique<long[]>(m + 1);
    dili_authors::recordPtrArray ptrs = std::make_unique<long[]>(m + 1);
    size_t c = 0;
    for (size_t i = 0; i < m; ++i) {
      const long k = long(uint64_t(kv[i].first) - base);
      const uint64_t v = uint64_t(kv[i].second);
      if (v == ~uint64_t(0) || k == kTop) {
        side_.emplace(uint64_t(kv[i].first), v), ++why_[0];
      } else if (c && k - keys[c - 1] < margin(k)) {
        side_.emplace(uint64_t(kv[i].first), v), ++why_[1];
      } else {
        keys[c] = k, ptrs[c] = long(v), ++c;
      }
    }
    for (int attempt = 0;; ++attempt) {
      if (c < kMinBulk) {  // too few for the authors' loader: this half stays in the side store
        for (size_t j = 0; j < c; ++j) side_.emplace(base | uint64_t(keys[j]), uint64_t(ptrs[j]));
        why_[4] += c;
        return;
      }
      if (attempt == 5) throw std::runtime_error(name_ + ": the authors' bulk load still loses keys after four rebuilds");
      keys[c] = keys[c - 1] + 1, ptrs[c] = -1;  // the authors' sentinel (DILI.cpp:160)
      D& d = dili_[h];
      d.clear();
      d.init_insert_aux_vars();
      {
        ScratchDir dir;
        Quiet quiet;
        d.set_mirror_dir(dir.path);
        dili_authors::buMinFan = bu_min_fan_;
        dili_authors::RHO = rho_;
        d.bulk_load(keys, ptrs, long(c));  // the authors' bulk load
      }
      if (!root_of(h) || !root_of(h)->pe_data) throw std::runtime_error(name_ + ": the authors' bulk load left no root");
      built_[h] = true;
      // Every key against the authors' search and the wrapper's walk. Past the int range somewhere on its path (see slot_of): it
      // leaves DILI, and if the authors' search does not find it where x86's cast sends it, it sits in some slot out of order.
      // Lost: on a path inside the int range, and not found. The keys past the range are the cause when both occur (they are
      // counted into the wrong leaf by find_leaf, DILI.h:153, and the leaves after it are loaded with shifted keys), so then
      // only they are taken out and the rest is built again.
      std::vector<unsigned char> state(c, 0);
      size_t far = 0, stray = 0, lost = 0;
      for (size_t j = 0; j < c; ++j) {
        long* at = nullptr;
        const bool found = d.search(keys[j]) == ptrs[j];
        const int w = walk(h, keys[j], &at);
        if (w == 1 && found && *at == ptrs[j]) continue;
        if (w < 0) state[j] = 1, ++far, stray += !found;
        else state[j] = 2, ++lost;
      }
      const bool again = lost || stray;
      const unsigned char out = !again || far ? 1 : 2;  // what goes to the side store this round
      if (far || lost) {
        size_t o = 0;
        for (size_t j = 0; j < c; ++j) {
          if (state[j] != out) {
            keys[o] = keys[j], ptrs[o] = ptrs[j], ++o;
            continue;
          }
          if (!again && !d.erase(keys[j])) throw std::runtime_error(name_ + ": the authors' erase missed a key their search finds");
          side_.emplace(base | uint64_t(keys[j]), uint64_t(ptrs[j]));
          ++why_[out == 1 ? 2 : 3];
        }
        c = o;
      }
      if (again) {
        tree_bytes(root_of(h), true);
        d.clear();
        built_[h] = false;
        ++rebuilds_;
        continue;
      }
      // the walk scan relies on: exactly these keys, ascending
      int sub = 0;
      size_t j = 0;
      for (bool ok = first(h, tmp_, sub); ok; ok = next(tmp_, sub), ++j) {
        const Entry& e = tmp_.back().n->pe_data[tmp_.back().i];
        const long k = e.key >= 0 ? e.key : sub ? e.fan2child->k2 : e.fan2child->k1;
        if (j >= c || k != keys[j]) throw std::runtime_error(name_ + ": the authors' structure does not hold the keys in slot order after the bulk load");
      }
      if (j != c) throw std::runtime_error(name_ + ": the authors' structure holds fewer keys than were loaded");
      n_dili_[h] = c;
      return;
    }
  }

  std::string name_;
  int bu_min_fan_;
  double rho_;                    // the authors' RHO (global.cpp:46), the decay rate of their cost model
  D dili_[2];                     // the authors' index: keys below 2^63, and keys from 2^63 up (stored minus 2^63)
  bool built_[2] = {false, false};
  size_t n_dili_[2] = {0, 0};
  std::map<uint64_t, uint64_t> side_;  // what the authors' code cannot hold (see the head of this file)
  size_t why_[5] = {0, 0, 0, 0, 0};    // at bulk load: unrepresentable, too close, int range, lost by the loader, small half
  size_t rebuilds_ = 0;
  size_t put_side_ = 0;                // puts since the bulk load that went to the side store
  size_t side_reported_ = 0;           // the side store's size when memory_consumption last printed it
  std::vector<Frame> path_, tmp_, scan_;
  std::vector<std::pair<long, long>> snap_;
  bool local_ = true;
};

// "dili", "dili<B>" (B = buMinFan in [2, 200], no leading zero), each optionally followed by "r<PP>" (rho = PP / 100, PP in
// [1, 99], no leading zero; without it rho is the released 0.1); nullptr for any other name, and for key or payload types other
// than 64-bit integers (unsigned keys)
template <class K, class P>
indexInterface<K, P>* make_dili(const std::string& name) {
  if constexpr (!(std::is_integral<K>::value && std::is_unsigned<K>::value && sizeof(K) == 8 && std::is_integral<P>::value && sizeof(P) == 8)) {
    (void)name;
    return nullptr;
  } else {
    if (name.compare(0, 4, "dili") != 0) return nullptr;
    const std::string d = name.substr(4);
    const size_t rp = d.find('r');
    const std::string bs = d.substr(0, rp), rs = rp == std::string::npos ? std::string() : d.substr(rp + 1);
    auto number = [](const std::string& t, size_t digits, int& out) {  // 1 to `digits` decimal digits, no leading zero
      if (t.empty() || t.size() > digits || t[0] == '0' || t.find_first_not_of("0123456789") != std::string::npos) return false;
      out = std::stoi(t);
      return true;
    };
    int b = 16, r = 10;
    if (!bs.empty() && !number(bs, 3, b)) return nullptr;
    if (rp != std::string::npos && !number(rs, 2, r)) return nullptr;
    return b >= 2 && b <= 200 ? new DiliIndex<K, P>(name, b, r / 100.0) : nullptr;
  }
}

}  // namespace gret
