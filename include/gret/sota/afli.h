// SPDX-License-Identifier: MIT
// AfliIndex<K, P>: AFLI, the index of NFL (Wu, Cui, Yu, Sun, Kuo, Xue: "NFL: Robust Learned Index via Distribution
// Transformation", PVLDB 15(10), 2022), in GRE's harness, under two families of names:
//
//   afli, afli<B>                    the NFL AUTHORS' OWN AFLI (https://github.com/luffy06/NFL, GPL-3.0, used unmodified and
//                                    included by path: -I <NFL>/src), behind a wrapper. Not a reimplementation.
//   afli_cut, afli<B>_cut            the same code, keyed by the key's last 15 decimal digits (key mod 10^15) instead of the key:
//                                    what the authors' own data preparation does to every uint64 data set before any index of
//                                    their benchmark sees it (src/util/format_data.cc:60-66). The tree is then NOT in key order
//                                    (scan returns 0): a point-lookup index, to be reported as such. It is there for data sets
//                                    whose large keys share doubles, where plain afli sends them to the side store.
//   nfl_afli[<B>][_cut][_on]         a HYBRID, and not the authors' NFL: the authors' AFLI as above, behind THIS REPOSITORY'S
//                                    REIMPLEMENTATION of NFL's numerical flow (gret/nflflow.h, inference only, scalar, one key
//                                    at a time). The authors' own flow inference (src/models/bnaf.h, numerical_flow.h) needs
//                                    Intel MKL and is not used. The weights are read from the file $TRANSFORM_NFL names, in the
//                                    format NFL's trainer writes (the authors ship such files in <NFL>/flow_weights).
//
// <B> in [1, 6] forces AFLI's maximum bucket size; without it the size is what the authors' code derives from the keys.
// _cut feeds the flow (nfl_afli_cut) or the tree itself (afli_cut) the key's last 15 decimal digits (key mod 10^15), which is
// what the authors' data preparation (src/util/format_data.cc:60-66) does to every uint64 data set before training and before
// the benchmark: the weights they ship for books and fb were trained on such keys and mean nothing on the raw ones. (If no
// bulk-loaded key reaches 10^15 the cut changes nothing and is left out.) The key itself is kept in the slot and compared on
// every hit; two keys with the same last 15 digits cannot both be in the tree, the second goes to the side store.
// _on keeps the flow even when NFL's own switch (nfl.h:71-91) would turn it off.
// docs/sota/afli.md has all of it, with file and line.
//
// What runs in the authors' code: the bulk load (AFLI::bulk_load), every lookup (TNode::find from the root, which is all
// AFLI::find does), every insert (TNode::insert, on the node the key lands in). get() adds to their find a test of the mode,
// a test of the key against the range of the keys the tree has held and, with 24-byte slots, the comparison of the stored
// key. What is the wrapper's, because the authors' code lacks it or gets it wrong (docs/sota/afli.md):
//   - uint64 keys. The authors' benchmark has one key type, float64 (benchmark.cc:20), and AFLI's models and its integer
//     fallback (conflicts.h:82) assume it: the tree is AFLI<double, .>, keyed by the key as a double. If every bulk-loaded key
//     is below 2^53 the value is the payload (16-byte slots, the authors' layout); otherwise the value is the (key, payload)
//     pair (24-byte slots, the layout of NFL's own transformed index, nfl.h:23) and every hit is checked against the key.
//   - a side store (std::map) for the keys the tree cannot hold exactly: a key whose double (or whose flow output) is that of
//     another key, and a key whose predicted slot leaves (-2^62, 2^62) at some node (the authors cast it, linear_model.h:17).
//     The line bulk_load prints says how many keys are there: a run in which that number is not small is not a measurement of
//     AFLI, or of NFL.
//   - a check of every key after the bulk load and of every node the authors' insert rebuilds: build_linear_model keeps the
//     first key's slot from a model it then replaces (conflicts.h:75-86), so the first key of a node can sit where find does
//     not look. Such a key is taken out and put back through the authors' insert ("repaired" in the log line).
//   - put's duplicate check (the authors' insert stores a second copy), update (through the authors' find; their update and
//     remove clamp the slot with size_ where find uses capacity_, afli_nodes.h:135, 164), remove (theirs also drops the last
//     key of a dense node, afli_nodes.h:188-197), scan (they have none; an in-order walk, only where the tree is in key order:
//     afli, not nfl_afli), memory_consumption (their index_size() counts a shared child once per slot, afli.h:130).
//   - AFLI's bucket size computed with the null check the authors' compute_tail_conflicts lacks (conflicts.h:114: a null
//     pointer on one key, and on any key set too sparse for its model).
//   - a guard on the authors' cast of a predicted slot to int64 (linear_model.h:17, no range check) for keys that are NOT
//     stored: the wrapper shows, at the bulk load, after every rebuild and whenever the key range grows, that no key get()
//     can hand to the authors' find is cast out of range anywhere on its way; in a tree where that cannot be shown, lookups
//     walk the authors' nodes in the wrapper instead (a line on stderr says so, and "lookups=" in every log line). This does
//     not happen on ordinary data: the test demands the authors' find on all its ordinary sets.
//
// Build: -I <NFL>/src, and <boost/optional.hpp> must resolve: NFL's util/common.h includes it and uses nothing from it. Without
// Boost, put a directory holding a boost/optional.hpp of one line (#include <optional>) on the include path. The authors'
// headers are included inside namespace gret::afli_authors; their macros and include guards are removed again. On macOS their
// code does not compile as it is (std::max(int64_t, 0L) with int64_t = long long): there, and only there, int64_t is defined to
// long while their headers are read. util/common.h defines three non-inline functions: include this header in one translation
// unit of a program.
#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pthread.h>

#if !__has_include(<boost/optional.hpp>)
#error "gret/sota/afli.h: NFL's src/util/common.h includes <boost/optional.hpp> (and uses nothing from it). Without Boost, add a directory holding a file boost/optional.hpp with the one line '#include <optional>' to the include path (docs/sota/afli.md)"
#endif
#include <boost/optional.hpp>

#include "gret/nflflow.h"

// ---- the authors' sources, unmodified, in a namespace of their own. Every standard header they include is included above, so
// inside the namespace those includes are no-ops (<cassert> is read again and only redefines assert).
#pragma push_macro("likely")
#pragma push_macro("unlikely")
#pragma push_macro("COMMON_H")
#pragma push_macro("ITERATOR_H")
#pragma push_macro("BUCKET_H")
#pragma push_macro("LINEAR_MODEL_H")
#pragma push_macro("CONFLICTS_H")
#pragma push_macro("AFLI_NODES_H")
#pragma push_macro("AFLI_H")
#undef likely
#undef unlikely
#undef COMMON_H
#undef ITERATOR_H
#undef BUCKET_H
#undef LINEAR_MODEL_H
#undef CONFLICTS_H
#undef AFLI_NODES_H
#undef AFLI_H
#if defined(__APPLE__) && defined(__LP64__)  // int64_t is long long there, and the authors write std::max(int64_t, 0L)
#pragma push_macro("int64_t")
#undef int64_t
#define int64_t long
#define GRET_AFLI_INT64_AS_LONG 1
#endif
namespace gret {
namespace afli_authors {
#include "afli/afli.h"
static_assert(sizeof(long) == 8 && std::is_same<decltype(nfl::LinearModel<double>().predict(0.0)), long>::value,
              "AFLI's slot arithmetic is written for a 64-bit long that is int64_t");
enum : unsigned { gret_none = nfl::kNone, gret_data = nfl::kData, gret_bucket = nfl::kBucket, gret_node = nfl::kNode };
}  // namespace afli_authors
}  // namespace gret
#ifdef GRET_AFLI_INT64_AS_LONG
#undef int64_t
#pragma pop_macro("int64_t")
#undef GRET_AFLI_INT64_AS_LONG
#endif
#undef BIT_TYPE
#undef BIT_SIZE
#undef BIT_LEN
#undef BIT_IDX
#undef BIT_POS
#undef SET_BIT_ONE
#undef SET_BIT_ZERO
#undef REV_BIT
#undef GET_BIT
#undef likely
#undef unlikely
#undef COMMON_H
#undef ITERATOR_H
#undef BUCKET_H
#undef LINEAR_MODEL_H
#undef CONFLICTS_H
#undef AFLI_NODES_H
#undef AFLI_H
#pragma pop_macro("AFLI_H")
#pragma pop_macro("AFLI_NODES_H")
#pragma pop_macro("CONFLICTS_H")
#pragma pop_macro("LINEAR_MODEL_H")
#pragma pop_macro("BUCKET_H")
#pragma pop_macro("ITERATOR_H")
#pragma pop_macro("COMMON_H")
#pragma pop_macro("unlikely")
#pragma pop_macro("likely")

namespace gret {

struct AfliOptions {
  bool nfl = false;         // nfl_afli: the flow of gret/nflflow.h in front (a hybrid)
  bool cut = false;         // _cut: the tree (afli_cut) or the flow and NFL's switch (nfl_afli_cut) see key mod 10^15, as the authors' format_data.cc prepares uint64 keys
  bool force_flow = false;  // _on: the flow stays on whatever NFL's switch says
  int bucket = 0;           // <B>: AFLI's maximum bucket size, 1..6; 0: derived from the keys by the authors' rule
};

template <class K, class P>
class AfliIndex final : public indexInterface<K, P> {
  static_assert(std::is_integral<K>::value && std::is_unsigned<K>::value && sizeof(K) == 8, "AfliIndex: keys are uint64");
  static_assert(std::is_integral<P>::value && sizeof(P) == 8, "AfliIndex: payloads are 64-bit integers");

  using KP = std::pair<K, P>;
  template <class V> using ZV = std::pair<double, V>;  // what the authors' tree stores: (coordinate, value)
  template <class V> using Tree = afli_authors::nfl::AFLI<double, V>;
  template <class V> using Node = afli_authors::nfl::TNode<double, V>;
  template <class V> using Bucket = afli_authors::nfl::Bucket<double, V>;
  template <class V> using Slot = afli_authors::nfl::Entry<double, V>;
  using Model = afli_authors::nfl::LinearModel<double>;
  using Hyper = afli_authors::nfl::HyperParameter;
  enum : unsigned { kNone = afli_authors::gret_none, kData = afli_authors::gret_data, kBucket = afli_authors::gret_bucket, kNode = afli_authors::gret_node };
  enum Coord : uint8_t { kPlain, kCut, kFlow, kCutFlow };  // what the tree is keyed by: the key, key mod 10^15, or the flow of either
  enum Where { kAbsent, kFound, kOther, kFar };
  enum Mode : uint8_t { kSlow, kFast16, kFast24 };  // get's fast paths: the tree is keyed by the key itself and the authors' find is safe for it

  template <class V>
  struct Step {
    Node<V>* n;
    uint32_t i;
  };

 public:
  static constexpr uint64_t kExact = uint64_t(1) << 53;         // below this every key is its own double
  static constexpr uint64_t kCut15 = 1000000000000000ull;       // 10^15: the authors keep std::numeric_limits<double>::digits10 = 15 digits
  static constexpr double kFarSlot = 4611686018427387904.0;     // 2^62: a predicted slot beyond it is not cast (the authors' cast is to int64)
  static constexpr unsigned kMaxBucket = 6, kMinBucket = 1;     // HyperParameter::kMaxBucketSize, kMinBucketSize (afli_nodes.h:29-30)
  static constexpr size_t kMaxKeys = (size_t(1) << 31) - 64;    // the authors count keys and slots (2 per key) in uint32

  explicit AfliIndex(std::string name, AfliOptions opt = AfliOptions()) : name_(std::move(name)), opt_(opt) {
    if (opt_.bucket != 0 && (opt_.bucket < int(kMinBucket) || opt_.bucket > int(kMaxBucket))) throw std::invalid_argument(name_ + ": the bucket size must be in [1, 6]");
    if (!opt_.nfl && opt_.force_flow) throw std::invalid_argument(name_ + ": _on belongs to nfl_afli");
  }
  AfliIndex(const AfliIndex&) = delete;
  AfliIndex& operator=(const AfliIndex&) = delete;

  void init(Param* = nullptr) override {}

  // kv: sorted by key, keys unique (GRE's contract; checked)
  void bulk_load(KP* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    reset();
    for (size_t i = 1; i < n; ++i)
      if (uint64_t(kv[i].first) <= uint64_t(kv[i - 1].first)) throw std::invalid_argument(name_ + ": bulk_load needs sorted, unique keys");
    if (n > kMaxKeys) throw std::length_error(name_ + ": the authors' AFLI counts keys and slots in uint32: at most 2^31 - 64 keys");
    try {
      load(kv, n);
    } catch (...) {
      reset();
      throw;
    }
    refresh_fast();
    side_reported_ = side_.size();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const size_t held = c16_.held + c24_.held;
    static const char* const coord_name[] = {"key", "key_mod_1e15", "flow(key)", "flow(key_mod_1e15)"};
    std::fprintf(stderr,
                 "gret: %s [%s] n=%zu in_afli=%zu side=%zu (same_coordinate=%zu far_slot=%zu) repaired=%zu coordinate=%s slot_bytes=%d bucket=%u(%s) "
                 "bytes_per_key=%.2f build_ms=%.1f of_which_check_ms=%.1f lookups=%s",
                 name_.c_str(),
                 opt_.nfl ? "HYBRID, not the authors' NFL: gret's reimplementation of NFL's flow (gret/nflflow.h) in front of the NFL authors' AFLI, wrapped"
                          : cut_ ? "the NFL authors' AFLI, wrapped, keyed by key mod 10^15 as the authors' data preparation cuts uint64 keys: no key order, no scan" : "the NFL authors' AFLI, wrapped",
                 n, held, side_.size(), why_same_, why_far_, c16_.repaired + c24_.repaired, coord_name[coord_], wide_ ? 24 : 16, bucket_, bucket_rule_,
                 n ? double(memory_consumption()) / double(n) : 0.0, ms, check_ms_, checked_ ? "the wrapper's walk of the authors' nodes (a model could cast a key out of int64)" : "the authors' find");
    if (opt_.nfl)
      std::fprintf(stderr, " flow=%s (NFL's switch: tail conflicts %s%u without, %s%u with; %s) weights=%s", flow_on_ ? "on" : "off", tail_ok_[0] ? "" : "undefined:", tail_[0],
                   tail_ok_[1] ? "" : "undefined:", tail_[1], opt_.force_flow ? "forced on by _on" : "its decision", weights_.c_str());
    if (opt_.cut && !cut_) std::fprintf(stderr, " _cut: no key reaches 10^15, nothing to cut");
    std::fprintf(stderr, "\n");
  }

  // The two cases a GRE run is in, afli with the tree keyed by the key itself, come first and add to the authors' find one
  // test of the mode, one of the key against the range of the keys the tree has held, and (24-byte slots) the comparison of
  // the stored key: a lookup that misses several caches is as slow as the instructions around it are many.
  bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    const unsigned m = mode_;
    if (m == kFast16) {  // every coordinate in this tree is a key below 2^53, and so is every x in the range
      if (__builtin_expect(x - xlo_ <= xspan_, 1)) {
        auto it = root16_->find(double(x));  // the authors' lookup: AFLI::find is root_->find (afli.h:39-41)
        if (__builtin_expect(!it.is_end(), 1)) {
          val = it.value();
          return true;
        }
      }
      return side_get(x, val);
    }
    if (m == kFast24) {
      if (__builtin_expect(x - xlo_ <= xspan_, 1)) {
        auto it = root24_->find(double(x));
        if (__builtin_expect(!it.is_end(), 1)) {
          const KP* v = it.value_addr();
          if (__builtin_expect(uint64_t(v->first) == x, 1)) {
            val = v->second;
            return true;
          }
        }
      }
      return side_get(x, val);
    }
    return get_other(x, val);
  }

  // true if the key was not there and is now; false, and nothing changes, if it was
  bool put(K key, P value, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    if (!side_.empty() && side_.count(x)) return false;
    const double z = coord_of(x);
    if (wide_) return put_in(c24_, z, x, KP(key, value), value);
    if (x >= kExact) {  // a tree of 16-byte slots holds keys below 2^53 only
      ++put_side_;
      side_.emplace(x, uint64_t(value));
      return true;
    }
    return put_in(c16_, z, x, value, value);
  }

  bool update(K key, P value, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    if (!side_.empty()) {
      const auto it = side_.find(x);
      if (it != side_.end()) {
        it->second = uint64_t(value);
        return true;
      }
    }
    const double z = coord_of(x);  // through the authors' lookup; their update() is not used (see the head of this file)
    if (wide_) {
      KP* v = in_tree(c24_, z, x);
      if (!v) return false;
      v->second = value;
      return true;
    }
    P* v = in_tree(c16_, z, x);
    if (!v) return false;
    *v = value;
    return true;
  }

  bool remove(K key, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    if (!side_.empty()) {
      const auto it = side_.find(x);
      if (it != side_.end()) {
        side_.erase(it);
        return true;
      }
    }
    const double z = coord_of(x);
    if (!(z >= zmin_ && z <= zmax_)) return false;
    return wide_ ? remove_in(c24_, z, x) : remove_in(c16_, z, x);
  }

  // the keys >= key, ascending, with their payloads, where the tree is in key order (afli, and nfl_afli when NFL's switch turned
  // the flow off; neither with _cut once a bulk-loaded key reaches 10^15); 0 otherwise: NFL has no range scan, and behind the
  // flow or the cut the tree is not in key order
  size_t scan(K key, size_t n, KP* out, Param* = nullptr) override {
    if (coord_ != kPlain) return 0;
    return wide_ ? scan_in(c24_, uint64_t(key), n, out) : scan_in(c16_, uint64_t(key), n, out);
  }

  // every byte reachable from the index: the authors' nodes with all their slots and both bitmaps, their models, their buckets
  // at the capacity they are allocated with, the flow's weights, the side store's tree nodes, this object. Not the allocator's
  // overhead.
  long long memory_consumption() override {
    size_t b = sizeof(*this) + side_.size() * kMapNode + c16_.scratch_bytes() + c24_.scratch_bytes();
    if (c16_.root) b += tree_bytes(c16_.root, bucket_);
    if (c24_.root) b += tree_bytes(c24_.root, bucket_);
    if (flow_) b += flow_->bytes() - sizeof(NflFlow);
    // GRE asks for the size once, after the run, and gives no other hook: say here how many keys the side store holds now
    if (side_.size() != side_reported_ || put_side_ != put_side_reported_) {
      side_reported_ = side_.size();
      put_side_reported_ = put_side_;
      const size_t held = c16_.held + c24_.held, all = held + side_.size();
      std::fprintf(stderr, "gret: %s now in_afli=%zu side=%zu (%.3f%% of the keys held; %zu puts went to the side store since the bulk load; %zu keys repaired after a rebuild) lookups=%s\n",
                   name_.c_str(), held, side_.size(), all ? 100.0 * double(side_.size()) / double(all) : 0.0, put_side_, c16_.repaired + c24_.repaired,
                   checked_ ? "the wrapper's walk" : "the authors' find");
    }
    return (long long)b;
  }

  // for tests and reports
  size_t in_afli() const { return c16_.held + c24_.held; }
  size_t in_side() const { return side_.size(); }
  size_t puts_to_side() const { return put_side_; }            // puts since the bulk load that the authors' insert was not given
  size_t repaired() const { return c16_.repaired + c24_.repaired; }  // keys found where the authors' find does not look, and put back
  size_t rebuilds() const { return c16_.rebuilds + c24_.rebuilds; }  // nodes the authors' insert rebuilt (each one checked)
  size_t side_same() const { return why_same_; }
  size_t side_far() const { return why_far_; }
  bool flow_on() const { return flow_on_; }
  bool checked_lookups() const { return checked_; }  // lookups go through the wrapper's walk: a model was found that the authors' find could cast out of range
  size_t unsafe_models() const { return unsafe_; }   // how often such a model was found (at the bulk load, a rebuild, a new largest or smallest key)
  bool keeps_order() const { return coord_ == kPlain; }
  int slot_bytes() const { return wide_ ? 24 : 16; }
  unsigned bucket() const { return bucket_; }
  unsigned tail(int with_flow) const { return tail_[with_flow]; }
  // The authors' AFLI::index_size(). Not called by the wrapper: it walks a child that several slots share once per slot
  // (afli.h:130 tests j < size_ where the slots run to capacity_), at every level, so on a tree with such children it counts
  // them several times over and its running time grows with the product of the run lengths down the tree.
  uint64_t authors_index_size() { return c16_.tree ? c16_.tree->index_size() : c24_.tree ? c24_.tree->index_size() : 0; }

 private:
  static constexpr size_t kMapNode = 4 * sizeof(void*) + 2 * sizeof(uint64_t);  // a red-black tree node holding two uint64

  // ---- one tree of the authors', with what the wrapper keeps next to it
  template <class V>
  struct Cursor;
  template <class V>
  struct Core {
    std::optional<Tree<V>> tree;  // the authors' index
    Node<V>* root = nullptr;      // its root_ (private there: read once through the first member, see root_of)
    Hyper hp;                     // the authors' parameters as the tree has them: handed to TNode::insert
    size_t held = 0, repaired = 0, rebuilds = 0;
    std::vector<Step<V>> path;    // the last walk: the nodes from the root to the slot
    Cursor<V> cursor;             // scan's position
    Cursor<V> walk;               // the checks' position (after the bulk load, after a rebuild)
    std::vector<Node<V>*> above;  // the nodes above a rebuilt one
    size_t scratch_bytes() const {
      return (path.capacity() + cursor.st.capacity() + walk.st.capacity()) * sizeof(Step<V>) + above.capacity() * sizeof(Node<V>*);
    }
    void clear() {
      tree.reset();  // deletes the authors' nodes
      root = nullptr;
      held = repaired = rebuilds = 0;
    }
  };

  // AFLI::root_ is private and is its first member: a standard-layout class is pointer-interconvertible with it
  template <class V>
  static Node<V>* root_of(Tree<V>& t) {
    static_assert(std::is_standard_layout<Tree<V>>::value, "AFLI is no longer standard-layout: root_of() needs another way in");
    return *reinterpret_cast<Node<V>**>(&t);
  }

  // what is compared with the key on a hit, and where the payload is
  static bool same_key(const P&, uint64_t) { return true; }  // 16-byte slots: the coordinate is the key
  static bool same_key(const KP& v, uint64_t x) { return uint64_t(v.first) == x; }
  static uint64_t key_of(const ZV<P>& e) { return uint64_t(e.first); }
  static uint64_t key_of(const ZV<KP>& e) { return uint64_t(e.second.first); }
  static P payload_of(const P& v) { return v; }
  static P payload_of(const KP& v) { return v.second; }
  static P make_value(const KP& kv, P*) { return kv.second; }
  static KP make_value(const KP& kv, KP*) { return kv; }

  double coord_of(uint64_t x) const {
    switch (coord_) {
      case kPlain: return double(x);
      case kCut: return double(x % kCut15);
      case kFlow: return double((*flow_)(x));
      default: return double((*flow_)(x % kCut15));
    }
  }
  static double coord_of(Coord c, const NflFlow* f, uint64_t x) {
    switch (c) {
      case kPlain: return double(x);
      case kCut: return double(x % kCut15);
      case kFlow: return double((*f)(x));
      default: return double((*f)(x % kCut15));
    }
  }

  // ---- lookups outside get's fast paths
  __attribute__((noinline)) bool side_get(uint64_t x, P& val) const {
    if (side_.empty()) return false;
    const auto it = side_.find(x);
    if (it == side_.end()) return false;
    val = P(it->second);
    return true;
  }
  // where the tree holds key x, whose coordinate is z, or nullptr. The authors' find; the wrapper's walk instead once a model
  // was found that their find could cast a key out of int64 on (checked_; see casts_in_range)
  template <class V>
  V* in_tree(Core<V>& c, double z, uint64_t x) {
    if (!(z >= zmin_ && z <= zmax_)) return nullptr;  // also: nothing is in the tree yet
    if (__builtin_expect(checked_, 0)) {
      ZV<V>* at = nullptr;
      return locate(c, c.root, z, x, at) == kFound ? &at->second : nullptr;
    }
    auto it = c.root->find(z);
    return !it.is_end() && same_key(*it.value_addr(), x) ? it.value_addr() : nullptr;
  }
  __attribute__((noinline)) bool get_other(uint64_t x, P& val) {
    const double z = coord_of(x);
    if (wide_) {
      if (const KP* v = in_tree(c24_, z, x)) {
        val = v->second;
        return true;
      }
    } else if (const P* v = in_tree(c16_, z, x)) {  // 16-byte slots: z <= zmax_ < 2^53 says x is its own double
      val = *v;
      return true;
    }
    return side_get(x, val);
  }
  void refresh_fast() {
    root16_ = c16_.root;
    root24_ = c24_.root;
    xlo_ = kmin_;
    xspan_ = kmax_ - kmin_;
    mode_ = kSlow;
    if (coord_ != kPlain || checked_ || kmin_ > kmax_) return;
    if (!wide_ && c16_.root) mode_ = kFast16;
    else if (wide_ && c24_.root) mode_ = kFast24;
  }

  // ---- the authors' cast (linear_model.h:17: static_cast<int64_t>(std::floor(slope_ * key + intercept_)), no range check)
  // For a key in the tree the wrapper knows the cast is in range: locate() checked its path. A lookup of a key that is NOT in
  // the tree walks the same nodes, and a node can be steep where the key is far from the node's own keys (a few dense keys in a
  // slot of the parent that is 2^60 wide). So the wrapper keeps this true: for every z in [zmin_, zmax_], the range of the
  // coordinates the tree has ever held, which is all get() hands to the authors' find, every model on z's way predicts inside
  // (-2^62, 2^62). It is established for the whole tree after the bulk load (casts_in_range), for what a rebuild built
  // (path_range, casts_in_range) and for the nodes a new largest or smallest key widens the reach of (end_path_in_range).
  // Where it cannot be shown, lookups leave the authors' find for the wrapper's walk, for good (mark_unsafe), and the log
  // lines say so. The models' slopes are positive (linear_model.h:81, conflicts.h:64), so the ends of an interval decide.

  // the z that slots i..j of model node n take, within [lo, hi]: two slots more on each side than the arithmetic says, for its
  // rounding (the intercept is -slope * (the node's first key) + 0.5, as large as 2^53 where keys are one double apart)
  template <class V>
  static void slot_range(Node<V>* n, uint32_t i, uint32_t j, double lo, double hi, double& clo, double& chi) {
    const double s = n->model_->slope_, c = n->model_->intercept_;
    clo = i == 0 ? lo : std::max(lo, (double(i) - 2 - c) / s);                 // slot 0 takes everything below it
    chi = j + 1 >= n->capacity_ ? hi : std::min(hi, (double(j) + 3 - c) / s);  // the last slot everything above
  }
  // the slots around slot i of n that hold the same child
  template <class V>
  static void child_run(Node<V>* n, uint32_t i, uint32_t& a, uint32_t& b) {
    Node<V>* child = n->entries_[i].child_;
    for (a = i; a > 0 && n->entry_type(a - 1) == kNode && n->entries_[a - 1].child_ == child;) --a;
    for (b = i; b + 1 < n->capacity_ && n->entry_type(b + 1) == kNode && n->entries_[b + 1].child_ == child;) ++b;
  }
  // true if no z in [lo, hi], the z that arrive at n, is cast out of range at n or under it. Every child has held a key of
  // the range, so an empty interval means the arithmetic above failed: not shown, hence false
  template <class V>
  static bool casts_in_range(Node<V>* n, double lo, double hi) {
    if (n->model_ == nullptr) return true;  // a sorted array: no model, no cast
    const double s = n->model_->slope_, c = n->model_->intercept_;
    if (!(s > 0) || !(lo <= hi) || !(s * lo + c > -kFarSlot) || !(s * hi + c < kFarSlot)) return false;
    for (uint32_t i = 0; i < n->capacity_; ++i) {
      if (n->entry_type(i) != kNode) continue;
      uint32_t a, b;
      child_run(n, i, a, b);
      double clo, chi;
      slot_range(n, i, b, lo, hi, clo, chi);
      if (!casts_in_range(n->entries_[i].child_, clo, chi)) return false;
      i = b;
    }
    return true;
  }
  // the z that arrive at the last node of c.path, the walk of a key
  template <class V>
  void path_range(const Core<V>& c, double& lo, double& hi) const {
    lo = zmin_, hi = zmax_;
    for (size_t k = 0; k + 1 < c.path.size(); ++k) {
      uint32_t a, b;
      child_run(c.path[k].n, c.path[k].i, a, b);
      slot_range(c.path[k].n, a, b, lo, hi, lo, hi);
    }
  }
  // [zmin_, zmax_] is about to grow from its end zold to znew. No slot past zold's has ever held a key, so none has a child:
  // the z in between arrive at the nodes on zold's path and at no other model.
  template <class V>
  static bool end_path_in_range(Node<V>* n, double zold, double znew) {
    double bound = znew;  // the new z that arrive at n lie between zold and bound
    for (;;) {
      if (n->model_ == nullptr) return true;
      const double s = n->model_->slope_, c = n->model_->intercept_, p = s * bound + c;
      if (!(s > 0) || !(p > -kFarSlot && p < kFarSlot)) return false;
      bool far = false;
      const uint32_t i = slot_of(n, zold, far);
      if (far) return false;
      if (n->entry_type(i) != kNode) return true;
      uint32_t a, b;
      child_run(n, i, a, b);
      double lo = std::min(zold, bound), hi = std::max(zold, bound);
      slot_range(n, a, b, lo, hi, lo, hi);
      bound = znew > zold ? hi : lo;
      n = n->entries_[i].child_;
    }
  }
  void mark_unsafe() {
    ++unsafe_;
    if (!checked_)
      std::fprintf(stderr, "gret: %s: a model in the authors' tree could cast a key that is not stored out of int64 (NFL/src/models/linear_model.h:17): "
                           "from here on lookups walk the authors' nodes in the wrapper, which checks the range, instead of calling the authors' find\n", name_.c_str());
    checked_ = true;
    refresh_fast();
  }

  // The slot the authors' find, insert and build give z at a model node: min(max(model_->predict(z), 0L), capacity_ - 1)
  // (afli_nodes.h:107, 208). predict casts floor(slope * z + intercept) to int64, which is undefined outside its range: there
  // far is set and the end slot the clamp would give is returned, without the cast.
  template <class V>
  static uint32_t slot_of(Node<V>* n, double z, bool& far) {
    const double pd = n->model_->predict_double(z);
    if (!(pd > -kFarSlot && pd < kFarSlot)) {
      far = true;
      return pd >= kFarSlot ? n->capacity_ - 1 : 0;
    }
    const long long p = n->model_->predict(z);  // the authors' arithmetic, in range
    return uint32_t(std::min<long long>(std::max<long long>(p, 0), (long long)(n->capacity_) - 1));
  }
  template <class V>
  static uint32_t dense_lower_bound(const Node<V>* n, double z) {  // a node without a model is a sorted array (afli_nodes.h:120)
    uint32_t lo = 0, hi = n->size_;
    while (lo < hi) {
      const uint32_t mid = lo + (hi - lo) / 2;
      if (n->entries_[mid].kv_.first < z) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  // z's path through the authors' nodes from `from` into c.path (the last step is the slot where z is or would go).
  // kFound: there, with key x (at: its entry). kOther: there, under another key. kAbsent. kFar: a predicted slot on the way is
  // outside what the authors' cast is defined for; the path ends at that node.
  template <class V>
  static Where locate(Core<V>& c, Node<V>* from, double z, uint64_t x, ZV<V>*& at) {
    c.path.clear();
    Node<V>* n = from;
    for (;;) {
      if (n->model_ == nullptr) {
        const uint32_t i = dense_lower_bound(n, z);
        c.path.push_back(Step<V>{n, i});
        if (i < n->size_ && n->entries_[i].kv_.first == z) {
          at = &n->entries_[i].kv_;
          return same_key(at->second, x) ? kFound : kOther;
        }
        return kAbsent;
      }
      bool far = false;
      const uint32_t i = slot_of(n, z, far);
      c.path.push_back(Step<V>{n, i});
      if (far) return kFar;
      const unsigned t = n->entry_type(i);
      if (t == kNode) {
        n = n->entries_[i].child_;
        continue;
      }
      if (t == kData) {
        if (n->entries_[i].kv_.first != z) return kAbsent;
        at = &n->entries_[i].kv_;
        return same_key(at->second, x) ? kFound : kOther;
      }
      if (t == kBucket) {
        Bucket<V>* b = n->entries_[i].bucket_;
        for (unsigned j = 0; j < b->size_; ++j)
          if (b->data_[j].first == z) {
            at = &b->data_[j];
            return same_key(at->second, x) ? kFound : kOther;
          }
      }
      return kAbsent;
    }
  }

  // ---- the entries under a node in ascending order of their coordinate: a stack of (node, slot)
  template <class V>
  struct Cursor {
    std::vector<Step<V>> st;
    ZV<V>* cur = nullptr;
    ZV<V>* bkt[8];  // the bucket the cursor is in, sorted (a bucket is unsorted and holds at most 6 entries)
    unsigned bn = 0, bi = 0;
    bool in_bucket = false;

    bool begin(Node<V>* top) {
      st.clear();
      st.push_back(Step<V>{top, 0});
      return settle();
    }
    bool seek(Node<V>* top, double z) {  // the first entry >= z
      st.clear();
      Node<V>* n = top;
      for (;;) {
        if (n->model_ == nullptr) {
          st.push_back(Step<V>{n, dense_lower_bound(n, z)});
          break;
        }
        bool far = false;
        const uint32_t i = slot_of(n, z, far);
        st.push_back(Step<V>{n, i});
        if (n->entry_type(i) != kNode) break;
        n = n->entries_[i].child_;
      }
      // slots are in coordinate order: what is left of the slot is smaller, what is right of it larger; inside it, a few entries
      if (!settle()) return false;
      while (cur->first < z)
        if (!next()) return false;
      return true;
    }
    bool next() {
      if (in_bucket && ++bi < bn) {
        cur = bkt[bi];
        return true;
      }
      ++st.back().i;
      return settle();
    }

   private:
    bool settle() {  // forward from the top step's slot to the first entry
      in_bucket = false;
      while (!st.empty()) {
        Node<V>* n = st.back().n;
        uint32_t& i = st.back().i;
        if (n->model_ == nullptr) {
          if (i < n->size_) {
            cur = &n->entries_[i].kv_;
            return true;
          }
        } else if (i < n->capacity_) {
          const unsigned t = n->entry_type(i);
          if (t == kData) {
            cur = &n->entries_[i].kv_;
            return true;
          }
          if (t == kNode) {
            Node<V>* child = n->entries_[i].child_;
            st.push_back(Step<V>{child, 0});
            continue;
          }
          if (t == kBucket && enter(n->entries_[i].bucket_)) return true;
          ++i;
          continue;
        }
        st.pop_back();  // this node is done: on in its parent, past every slot that points to it (afli_nodes.h:386-390)
        if (st.empty()) return false;
        Node<V>* p = st.back().n;
        uint32_t& pi = st.back().i;
        do ++pi;
        while (pi < p->capacity_ && p->entry_type(pi) == kNode && p->entries_[pi].child_ == n);
      }
      return false;
    }
    bool enter(Bucket<V>* b) {
      bn = b->size_;
      if (bn > 8) throw std::logic_error("gret afli: a bucket of the authors' AFLI holds more than 8 entries");
      for (unsigned j = 0; j < bn; ++j) {
        unsigned k = j;
        for (; k > 0 && b->data_[j].first < bkt[k - 1]->first; --k) bkt[k] = bkt[k - 1];
        bkt[k] = &b->data_[j];
      }
      if (bn == 0) return false;
      bi = 0;
      in_bucket = true;
      cur = bkt[0];
      return true;
    }
  };

  // ---- removal, as the authors' remove means it (afli_nodes.h:161-203), at the slot the wrapper's walk found
  template <class V>
  static void erase_entry(Node<V>* n, uint32_t i, double z) {
    if (n->model_ == nullptr) {
      const uint32_t at = dense_lower_bound(n, z);
      if (at >= n->size_ || n->entries_[at].kv_.first != z) throw std::logic_error("gret afli: an entry to remove is not in its dense node");
      for (uint32_t j = at; j + 1 < n->size_; ++j) n->entries_[j].kv_ = n->entries_[j + 1].kv_;
      --n->size_;
      return;
    }
    const unsigned t = n->entry_type(i);
    if (t == kData) {  // the authors' set_entry_type(idx, kNone), which is private: both type bits to 0
      n->bitmap0_[i / 8] = uint8_t(n->bitmap0_[i / 8] & ~(1u << (i % 8)));
      n->bitmap1_[i / 8] = uint8_t(n->bitmap1_[i / 8] & ~(1u << (i % 8)));
      --n->size_;
      return;
    }
    if (t != kBucket) throw std::logic_error("gret afli: an entry to remove is in a slot that holds none");
    Bucket<V>* b = n->entries_[i].bucket_;
    unsigned at = 0;
    while (at < b->size_ && b->data_[at].first != z) ++at;
    if (at == b->size_) throw std::logic_error("gret afli: an entry to remove is not in its bucket");
    for (unsigned j = at; j + 1 < b->size_; ++j) b->data_[j] = b->data_[j + 1];
    --b->size_;  // an emptied bucket stays, as with the authors' remove
  }

  template <class V>
  bool remove_in(Core<V>& c, double z, uint64_t x) {
    ZV<V>* at = nullptr;
    if (locate(c, c.root, z, x, at) != kFound) return false;
    erase_entry(c.path.back().n, c.path.back().i, z);
    for (const Step<V>& s : c.path) --s.n->size_sub_tree_;
    --c.held;
    return true;
  }

  // ---- insertion
  template <class V>
  bool put_in(Core<V>& c, double z, uint64_t x, const V& v, P value) {
    if (c.root == nullptr) {  // never bulk-loaded, or bulk-loaded with no key: the first key is the authors' bulk load of one pair
      const unsigned b = opt_.bucket ? unsigned(opt_.bucket) : kMaxBucket;
      ZV<V> one(z, v);
      c.tree.emplace();
      c.hp.max_bucket_size_ = b;
      c.hp.aggregate_size_ = 0;
      c.tree->bulk_load(&one, 1, int32_t(b), 0);
      c.root = root_of(*c.tree);
      c.held = 1;
      bucket_ = b;
      bucket_rule_ = opt_.bucket ? "name" : "no bulk load: the authors' initial value";
      zmin_ = zmax_ = z;
      kmin_ = kmax_ = x;
      refresh_fast();
      return true;
    }
    ZV<V>* at = nullptr;
    const Where w = locate(c, c.root, z, x, at);
    if (w == kFound) return false;
    if (w == kAbsent) {
      if (__builtin_expect(z < zmin_ || z > zmax_ || x < kmin_ || x > kmax_, 0)) {  // a new end: before the insert, whose rebuild check reads the range
        if (z < zmin_ || z > zmax_) {
          if (!checked_ && !end_path_in_range(c.root, z < zmin_ ? zmin_ : zmax_, z)) mark_unsafe();
          if (z < zmin_) zmin_ = z;
          else zmax_ = z;
        }
        if (x < kmin_) kmin_ = x;
        if (x > kmax_) kmax_ = x;
        refresh_fast();
      }
      insert_here(c, z, v, 0);
      return true;
    }
    ++put_side_;  // another key has this coordinate, or a slot on the way is outside the cast's range: not for the authors' insert
    side_.emplace(x, uint64_t(value));
    return true;
  }

  // z is absent and c.path is its walk from the root. The authors' TNode::insert on the last node of the walk: that is where
  // their AFLI::insert arrives after adding 1 to size_sub_tree_ of each node above, which is done here. If the insert will
  // rebuild (a full bucket becomes a child node, a full dense node is built again), what it built is checked.
  template <class V>
  void insert_here(Core<V>& c, double z, const V& v, int depth) {
    Node<V>* n = c.path.back().n;
    const uint32_t i = c.path.back().i;
    const bool dense = n->model_ == nullptr;
    bool rebuild;
    if (dense) {
      rebuild = n->size_ >= n->capacity_;
    } else {
      const unsigned t = n->entry_type(i);
      rebuild = (t == kBucket && n->entries_[i].bucket_->size_ >= c.hp.max_bucket_size_) || (t == kData && c.hp.max_bucket_size_ <= 1);
    }
    for (size_t j = 0; j + 1 < c.path.size(); ++j) ++c.path[j].n->size_sub_tree_;
    n->insert(ZV<V>(z, v), uint32_t(c.path.size()), c.hp);  // the authors' insert
    ++c.held;
    if (!rebuild) return;
    ++c.rebuilds;
    if (!dense && n->entry_type(i) != kNode) throw std::logic_error(name_ + ": the authors' insert did not rebuild where the wrapper expected it to");
    Node<V>* built = dense ? n : n->entries_[i].child_;
    if (!checked_) {  // the keys that can arrive at what was built, against the authors' cast (before the path is walked over)
      double lo, hi;
      path_range(c, lo, hi);
      if (!dense) slot_range(n, i, i, lo, hi, lo, hi);
      if (!casts_in_range(built, lo, hi)) mark_unsafe();
    }
    std::vector<Node<V>*> nested;  // a rebuild inside a repair: rare, and the outer one still reads c.above
    std::vector<Node<V>*>& above = depth == 0 ? c.above : nested;
    above.clear();
    for (size_t j = 0; j + (dense ? 1 : 0) < c.path.size(); ++j) above.push_back(c.path[j].n);
    // build() sets size_sub_tree_ of what it builds to the number of pairs it was given: the walk must find that many. (If the
    // authors' builder ever emitted a slot twice, the later run of keys would overwrite the earlier one and a key would be gone
    // without a trace. Not seen in 30M fuzzed nodes on arm64 and x86-64; the count makes it a loud error and not a lost key.)
    check_subtree(c, built, above, depth + 1, built->size_sub_tree_);
  }

  // Every entry under top must be where the authors' find looks for it (top->find, and the wrapper's walk must agree), on a
  // path whose slots are all inside the cast's range. One that is not is taken out of the slot it sits in and put again
  // through the authors' insert, or into the side store if its path leaves the range. The entries must be ascending and, with
  // expect != 0 (after a bulk load: the keys given; after a rebuild: the size the authors' build recorded), exactly that many.
  template <class V>
  void check_subtree(Core<V>& c, Node<V>* top, const std::vector<Node<V>*>& above, int depth, size_t expect) {
    if (depth > 12) throw std::runtime_error(name_ + ": the authors' AFLI keeps misplacing keys after 12 rounds of repair");
    struct Lost {
      ZV<V> e;
      std::vector<Node<V>*> chain;
      Node<V>* n;
      uint32_t i;
      bool far;
    };
    std::vector<Lost> lost;
    Cursor<V>& cur = c.walk;  // done with before anything below walks again
    size_t seen = 0;
    double prev = -std::numeric_limits<double>::infinity();
    for (bool ok = cur.begin(top); ok; ok = cur.next()) {
      ++seen;
      ZV<V>* e = cur.cur;
      ZV<V>* at = nullptr;
      const Where w = locate(c, top, e->first, key_of(*e), at);
      bool good = w == kFound && at == e;
      if (good) {
        auto it = top->find(e->first);  // the authors' lookup, from this node
        good = !it.is_end() && it.kv() == e;
      }
      if (good) {
        if (!(e->first > prev)) throw std::runtime_error(name_ + ": the authors' AFLI does not hold its keys in slot order");
        prev = e->first;
        continue;
      }
      Lost l{*e, {}, cur.st.back().n, cur.st.back().i, w == kFar};
      for (const Step<V>& s : cur.st) l.chain.push_back(s.n);
      lost.push_back(std::move(l));
    }
    if (expect && seen != expect)
      throw std::runtime_error(name_ + ": a node the authors' AFLI built (" + (depth == 0 ? "at the bulk load" : "in an insert") + ") holds " + std::to_string(seen) + " of the " + std::to_string(expect) + " keys it was given");
    for (const Lost& l : lost) {  // out of the slots they sit in; nothing else moves
      erase_entry(l.n, l.i, l.e.first);
      for (Node<V>* a : above) --a->size_sub_tree_;
      for (Node<V>* a : l.chain) --a->size_sub_tree_;
      --c.held;
    }
    for (const Lost& l : lost) {  // and in again
      ZV<V>* at = nullptr;
      const Where w = l.far ? kFar : locate(c, c.root, l.e.first, key_of(l.e), at);
      if (w == kAbsent) {
        insert_here(c, l.e.first, l.e.second, depth);
        ++c.repaired;
      } else if (w == kFar) {
        side_.emplace(key_of(l.e), uint64_t(payload_of(l.e.second)));
        ++why_far_;
      } else {
        throw std::runtime_error(name_ + ": a key the authors' AFLI misplaced is also where it belongs");
      }
    }
  }

  // ---- scan
  template <class V>
  size_t scan_in(Core<V>& c, uint64_t x, size_t n, KP* out) {
    size_t got = 0;
    auto sit = side_.empty() ? side_.end() : side_.lower_bound(x);
    if (c.root != nullptr && n != 0) {
      Cursor<V>& cur = c.cursor;
      bool ok = cur.seek(c.root, double(x));
      while (ok && key_of(*cur.cur) < x) ok = cur.next();  // at most the one entry whose double is x's double
      while (ok && got < n) {
        const uint64_t k = key_of(*cur.cur);
        for (; sit != side_.end() && sit->first < k && got < n; ++sit) out[got++] = KP(K(sit->first), P(sit->second));
        if (got == n) break;
        out[got++] = KP(K(k), payload_of(cur.cur->second));
        ok = cur.next();
      }
    }
    for (; sit != side_.end() && got < n; ++sit) out[got++] = KP(K(sit->first), P(sit->second));
    return got;
  }

  // bytes of the authors' nodes under root: each node once (a child shared by a run of slots is one node), every slot whether
  // used or not, both bitmaps, the model, each bucket at its allocated capacity
  template <class V>
  static size_t tree_bytes(Node<V>* root, unsigned bucket_capacity) {
    size_t b = 0;
    std::vector<Node<V>*> todo(1, root);
    while (!todo.empty()) {
      Node<V>* n = todo.back();
      todo.pop_back();
      b += sizeof(Node<V>) + size_t(n->capacity_) * sizeof(Slot<V>);
      if (n->model_ == nullptr) continue;
      b += sizeof(Model) + 2 * ((size_t(n->capacity_) + 7) / 8);
      for (uint32_t i = 0; i < n->capacity_; ++i) {
        const unsigned t = n->entry_type(i);
        if (t == kBucket) {
          b += sizeof(Bucket<V>) + size_t(bucket_capacity) * sizeof(ZV<V>);
        } else if (t == kNode) {
          Node<V>* child = n->entries_[i].child_;
          todo.push_back(child);
          while (i + 1 < n->capacity_ && n->entry_type(i + 1) == kNode && n->entries_[i + 1].child_ == child) ++i;
        }
      }
    }
    return b;
  }

  // ---- bulk load
  // The authors' compute_tail_conflicts (conflicts.h:106-123): the 99th percentile of the number of keys per used slot under one
  // linear model over all keys, minus one. Theirs dereferences a null pointer when no model can be fitted (one key; a slope
  // below 2.2e-16, i.e. fewer keys than 2.2e-16 times their range): this copy returns false there. The model is theirs.
  template <class V>
  static bool tail_conflicts(const ZV<V>* a, uint32_t m, double amplification, float percent, uint32_t& tail) {
    namespace au = afli_authors::nfl;
    Model* model = new Model();
    au::ConflictsInfo* ci = au::build_linear_model<double, V>(a, m, model, amplification);
    delete model;  // null when the fit failed
    if (ci == nullptr) return false;
    if (ci->num_conflicts_ == 0) {
      delete ci;
      tail = 0;
      return true;
    }
    std::sort(ci->conflicts_, ci->conflicts_ + ci->num_conflicts_);
    const uint32_t t = ci->conflicts_[std::max(0, int(ci->num_conflicts_ * percent) - 1)];
    delete ci;
    tail = t - 1;
    return true;
  }

  // (coordinate, value) for every pair, ascending in the coordinate; pairs whose coordinate another pair already has go to spill
  template <class V>
  static std::vector<ZV<V>> make_input(const KP* kv, size_t n, Coord coord, const NflFlow* flow, std::vector<KP>& spill) {
    std::vector<ZV<V>> a;
    a.reserve(n);
    for (size_t i = 0; i < n; ++i) a.emplace_back(coord_of(coord, flow, uint64_t(kv[i].first)), make_value(kv[i], static_cast<V*>(nullptr)));
    if (coord != kPlain)  // the plain coordinate is monotone in the key, and the keys came sorted
      std::sort(a.begin(), a.end(), [](const ZV<V>& l, const ZV<V>& r) { return l.first < r.first || (l.first == r.first && key_of(l) < key_of(r)); });
    size_t o = 0;
    for (size_t i = 0; i < n; ++i) {
      if (o != 0 && a[i].first == a[o - 1].first) spill.emplace_back(K(key_of(a[i])), payload_of(a[i].second));
      else a[o++] = a[i];
    }
    a.resize(o);
    return a;
  }

  // the authors' AFLI::bulk_load on `in`, then the check of every key
  template <class V>
  void build(Core<V>& c, std::vector<ZV<V>>& in, bool tail_ok, uint32_t tail, const char* rule) {
    if (opt_.bucket) {
      bucket_ = unsigned(opt_.bucket);
      bucket_rule_ = "name";
    } else if (tail_ok) {  // the clamp of AFLI::compute_bucket_size and AFLI::bulk_load (afli.h:27-32, 80-81)
      bucket_ = std::min<uint32_t>(std::max<uint32_t>(tail, kMinBucket), kMaxBucket);
      bucket_rule_ = rule;
    } else {
      bucket_ = kMaxBucket;
      bucket_rule_ = "no model fits the keys: the authors' initial value";
    }
    c.tree.emplace();
    c.hp.max_bucket_size_ = bucket_;
    c.hp.aggregate_size_ = 0;  // the authors' default; none of their configurations sets it
    c.tree->bulk_load(in.data(), uint32_t(in.size()), int32_t(bucket_), 0);
    c.root = root_of(*c.tree);
    c.held = in.size();
    zmin_ = in.front().first;
    zmax_ = in.back().first;
    kmin_ = key_of(in.front());  // read by get only where the coordinate is the key
    kmax_ = key_of(in.back());
    const size_t m = in.size(), far0 = why_far_;
    std::vector<ZV<V>>().swap(in);
    const auto t0 = std::chrono::steady_clock::now();
    check_subtree(c, c.root, std::vector<Node<V>*>(), 0, m);
    if (why_far_ != far0) {  // keys left the tree for the side store: the range is that of the keys that stayed
      zmin_ = std::numeric_limits<double>::infinity(), zmax_ = -zmin_, kmin_ = ~uint64_t(0), kmax_ = 0;
      for (bool ok = c.walk.begin(c.root); ok; ok = c.walk.next()) {
        const double z = c.walk.cur->first;
        const uint64_t k = key_of(*c.walk.cur);
        zmin_ = std::min(zmin_, z), zmax_ = std::max(zmax_, z), kmin_ = std::min(kmin_, k), kmax_ = std::max(kmax_, k);
      }
    }
    if (zmin_ <= zmax_ && !casts_in_range(c.root, zmin_, zmax_)) mark_unsafe();
    check_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }

  void to_side(const std::vector<KP>& spill) {
    for (const KP& s : spill) side_.emplace(uint64_t(s.first), uint64_t(s.second));
    why_same_ += spill.size();
  }

  void load(const KP* kv, size_t n) {
    namespace au = afli_authors::nfl;
    const bool small = n != 0 && uint64_t(kv[n - 1].first) < kExact;  // every key is its own double
    const Hyper defaults;
    if (opt_.nfl) {  // the flow is loaded even if it ends up unused: a missing file is an error of the run
      const char* e = std::getenv("TRANSFORM_NFL");
      if (!e || !*e) throw std::runtime_error(name_ + ": set TRANSFORM_NFL to NFL's weight file for this data set (docs/sota/afli.md)");
      weights_ = e;
      flow_.emplace(NflFlow::load(weights_));
    }
    if (!opt_.nfl || n == 0) {
      // afli_cut: the tree is keyed by key mod 10^15, the authors' own preparation of uint64 keys (format_data.cc:60-66), with
      // the key kept in the slot and compared on every hit. With every key below 10^15 the cut changes nothing and is left out.
      cut_ = !opt_.nfl && opt_.cut && n != 0 && uint64_t(kv[n - 1].first) >= kCut15;
      coord_ = cut_ ? kCut : kPlain;
      wide_ = cut_ || !small;
      if (n == 0) return;  // the first put builds the tree (on the keys themselves, whatever the name)
      std::vector<KP> spill;
      uint32_t tail = 0;
      // AFLI alone, as run_afli calls it (benchmark.h:457): bucket size from compute_bucket_size
      if (wide_) {
        std::vector<ZV<KP>> in = make_input<KP>(kv, n, coord_, nullptr, spill);
        const bool ok = tail_conflicts(in.data(), uint32_t(in.size()), defaults.kSizeAmplification, float(defaults.kTailPercent), tail);
        build(c24_, in, ok, tail, "AFLI::compute_bucket_size");
      } else {
        std::vector<ZV<P>> in = make_input<P>(kv, n, kPlain, nullptr, spill);
        const bool ok = tail_conflicts(in.data(), uint32_t(in.size()), defaults.kSizeAmplification, float(defaults.kTailPercent), tail);
        build(c16_, in, ok, tail, "AFLI::compute_bucket_size");
      }
      to_side(spill);
      return;
    }
    // NFL::auto_switch (nfl.h:71-91): the tail conflicts of the keys as they are and behind the flow, both with NFL's
    // amplification 1.5; the flow is kept if it lowers them by a tenth or more. Then AFLI on what was chosen, its bucket
    // size the tail conflicts of the choice (nfl.h:93-105, benchmark.h:519-522).
    const float amplification = 1.5f, percent = 0.99f, decay = 0.1f;  // NFL::kSizeAmplification, kTailPercent, kConflictsDecay
    cut_ = opt_.cut && uint64_t(kv[n - 1].first) >= kCut15;  // with every key below 10^15 the authors' cut changes nothing
    const Coord without = cut_ ? kCut : kPlain, with = cut_ ? kCutFlow : kFlow;
    const bool wide_without = cut_ || !small;
    std::vector<KP> spill_without, spill_with;
    std::vector<ZV<P>> in16;
    std::vector<ZV<KP>> in24;
    if (wide_without) {
      in24 = make_input<KP>(kv, n, without, nullptr, spill_without);
      tail_ok_[0] = tail_conflicts(in24.data(), uint32_t(in24.size()), amplification, percent, tail_[0]);
    } else {
      in16 = make_input<P>(kv, n, without, nullptr, spill_without);
      tail_ok_[0] = tail_conflicts(in16.data(), uint32_t(in16.size()), amplification, percent, tail_[0]);
    }
    flow_->fit(n, [&](size_t i) { return cut_ ? uint64_t(kv[i].first) % kCut15 : uint64_t(kv[i].first); });
    std::vector<ZV<KP>> flowed = make_input<KP>(kv, n, with, &*flow_, spill_with);
    tail_ok_[1] = tail_conflicts(flowed.data(), uint32_t(flowed.size()), amplification, percent, tail_[1]);
    // where a tail is undefined the authors' code stops on a null pointer: here the flow is then off, unless forced
    const bool nfl_says_on = tail_ok_[0] && tail_ok_[1] && !(tail_[0] <= tail_[1] || tail_[0] - tail_[1] < uint32_t(float(tail_[0]) * decay));
    flow_on_ = opt_.force_flow || nfl_says_on;
    if (flow_on_) {
      std::vector<ZV<P>>().swap(in16);
      std::vector<ZV<KP>>().swap(in24);
      coord_ = with;
      wide_ = true;
      build(c24_, flowed, tail_ok_[1], tail_[1], "NFL::auto_switch");
      to_side(spill_with);
    } else {
      std::vector<ZV<KP>>().swap(flowed);
      coord_ = without;
      wide_ = wide_without;
      if (wide_) build(c24_, in24, tail_ok_[0], tail_[0], "NFL::auto_switch");
      else build(c16_, in16, tail_ok_[0], tail_[0], "NFL::auto_switch");
      to_side(spill_without);
    }
  }

  void reset() {
    c16_.clear();
    c24_.clear();
    side_.clear();
    flow_.reset();
    weights_.clear();
    coord_ = kPlain;
    wide_ = true;  // an index never loaded takes any uint64 key
    flow_on_ = false;
    zmin_ = std::numeric_limits<double>::infinity();
    zmax_ = -std::numeric_limits<double>::infinity();
    kmin_ = ~uint64_t(0);
    kmax_ = 0;
    checked_ = false;
    unsafe_ = 0;
    cut_ = false;
    refresh_fast();
    bucket_ = kMaxBucket;
    bucket_rule_ = "not built";
    tail_[0] = tail_[1] = 0;
    tail_ok_[0] = tail_ok_[1] = false;
    why_same_ = why_far_ = put_side_ = side_reported_ = put_side_reported_ = 0;
    check_ms_ = 0;
  }

  // what get reads first
  Mode mode_ = kSlow;
  Coord coord_ = kPlain;
  bool wide_ = true;      // 24-byte slots (coordinate, key, payload) in c24_; otherwise 16-byte slots (key as double, payload) in c16_
  bool checked_ = false;  // lookups through the wrapper's walk (mark_unsafe)
  bool flow_on_ = false;
  bool cut_ = false;      // _cut, and a bulk-loaded key reaches 10^15
  uint64_t xlo_ = ~uint64_t(0), xspan_ = 0;  // kFast16, kFast24: kmin_ and kmax_ - kmin_
  Node<P>* root16_ = nullptr;                // c16_.root
  Node<KP>* root24_ = nullptr;               // c24_.root
  double zmin_ = std::numeric_limits<double>::infinity(), zmax_ = -std::numeric_limits<double>::infinity();  // the coordinates the tree has ever held
  uint64_t kmin_ = ~uint64_t(0), kmax_ = 0;  // and the keys: they never shrink (a removed key stays inside)
  std::string name_;
  AfliOptions opt_;
  size_t unsafe_ = 0;
  Core<P> c16_;
  Core<KP> c24_;
  std::optional<NflFlow> flow_;
  std::map<uint64_t, uint64_t> side_;  // what the authors' tree cannot hold (see the head of this file)
  std::string weights_;
  unsigned bucket_ = kMaxBucket;
  const char* bucket_rule_ = "not built";
  uint32_t tail_[2] = {0, 0};          // NFL's switch: tail conflicts without and with the flow
  bool tail_ok_[2] = {false, false};
  size_t why_same_ = 0, why_far_ = 0;  // into the side store at the bulk load and at repairs: coordinate taken, slot out of range
  size_t put_side_ = 0;                // puts since the bulk load that went to the side store
  size_t side_reported_ = 0, put_side_reported_ = 0;
  double check_ms_ = 0;
};

// "afli", "afli<B>", "nfl_afli", "nfl_afli<B>", each optionally followed by "_cut", the nfl_ ones then optionally by "_on"
// (B one digit in [1, 6]); nullptr for any other name, and for key or payload types other than 64-bit integers (unsigned keys)
template <class K, class P>
indexInterface<K, P>* make_afli(const std::string& name) {
  if constexpr (!(std::is_integral<K>::value && std::is_unsigned<K>::value && sizeof(K) == 8 && std::is_integral<P>::value && sizeof(P) == 8)) {
    (void)name;
    return nullptr;
  } else {
    AfliOptions o;
    size_t p = 0;
    if (name.compare(0, 8, "nfl_afli") == 0) o.nfl = true, p = 8;
    else if (name.compare(0, 4, "afli") == 0) p = 4;
    else return nullptr;
    if (p < name.size() && name[p] >= '1' && name[p] <= '6') o.bucket = name[p++] - '0';
    if (name.compare(p, 4, "_cut") == 0) o.cut = true, p += 4;
    if (o.nfl && name.compare(p, 3, "_on") == 0) o.force_flow = true, p += 3;
    if (p != name.size()) return nullptr;
    return new AfliIndex<K, P>(name, o);
  }
}

}  // namespace gret
