// SPDX-License-Identifier: MIT
// DytisIndex<K, P>: DyTIS (Jin Yang, Heejin Yoon, Gyeongchan Yun, Sam H. Noh, Young-ri Choi: "DyTIS: A Dynamic Dataset Targeted
// Index Structure Simultaneously Efficient for Search, Insert, and Scan", EuroSys 2023) in GRE's harness. This is a WRAPPER
// around the authors' own code (https://github.com/unist-ssl/DyTIS, commit 5cd954d, Apache-2.0, used unmodified and included by
// path: -I <DyTIS>), not a reimplementation. Lookups run the authors' DyTIS::Find; an insert runs the authors' Directory::Insert
// on the bucket (their DyTIS::Insert, whole, when the bucket is full or the first-level table is new); a removal runs their
// Directory::Delete; an update is their Update (Find and an assignment). DyTIS has no bulk load: bulk_load is one insert per key.
// What the wrapper adds is what the authors' code lacks or gets wrong outside the runs it was written for (docs/sota/dytis.md
// has the list with file and line):
//   - Boost.Pool. The authors' Directory.h includes <boost/pool/pool_alloc.hpp>, and Boost is not installed where this runs. The
//     two classes it uses (boost::pool<>, boost::pool_allocator<T>) are restated below with Boost's policy (a free list per chunk
//     size, blocks of 32, 64, 128, ... chunks), and the include is satisfied by a file of that name without code on the include
//     path (readonly_competitors/DyTIS_shim). The restated pool puts 16 bytes of 0xff in front of every chunk: the authors'
//     search reads the key slot before the first one of a segment (Directory_impl.h:807-809), and with the pad that read sees
//     "empty".
//   - a side store (std::map) for the keys the authors' code cannot hold: the last 64 keys of every aligned block of 2^24 keys
//     ("edge keys": 2^64 - 1, their empty slot, is one of them, and so is 2^64 - 2, which util/pair.h names SENTINEL and no
//     code uses; at the top of a segment's key range the authors' model, computed in doubles, can send a key one bucket past
//     the segment's last, and their code then writes outside the segment), and a key whose insert would push a segment past
//     local depth 31 (the depth lives in 5 bits of a tagged pointer; nothing in their code stops at 31). 4 keys in a million
//     are edge keys on spread data. bulk_load's line says how many keys are in the side store.
//   - get through DyTIS::Find (the search of DyTIS::Get, returning the slot): Get returns 0 for "not found", so a payload of 0
//     could not be told from a miss. bulk_load ends by looking every key up through Find and throws if one is not found
//     where it lies.
//   - put returns false and changes nothing for a stored key (the authors' Insert overwrites): one search of the key's bucket
//     before the insert. An insert into a full bucket makes the authors' code split, remap or expand the segment; the wrapper
//     counts and sums the segment's pairs first and checks afterwards that the same pairs and the new one are there (count
//     and a 64-bit sum of hashes), the keys in order, each in the bucket the authors' model sends it to, and that the model
//     is monotone and keeps every key of the segment inside its buckets. It throws std::runtime_error if not: never a silent
//     loss.
//   - remove: the authors' DyTIS::Delete returns true whether or not the key was there, reads one slot past a full bucket and
//     leaves a copy of the next bucket's first key in it (Directory_impl.h:412-415), and runs off the end of a bool function
//     when key 0 heads the bucket (ExtendibleHash_impl.h:164-171). The wrapper looks the key up first, calls the authors'
//     Directory::Delete when neither case applies, and closes the gap in the bucket itself when one does.
//   - scan: the authors' Scan returns payloads without keys and treats a payload of 2^64 - 1 as an empty slot. The wrapper walks
//     the authors' segments itself (buckets in order, then the sibling pointer), merged with the side store.
//   - memory_consumption: every byte the authors' code holds (pool blocks with their unused chunks, segments, directories,
//     models), the side store, this object.
// Names: dytis, dytiss, dytisu, each optionally followed by a.
//   - dytis is the released code and the paper's default: the code decides once, when a first-level table reaches global depth
//     8, whether the data is "uniform" (a segment may then grow to 2^7 times the buckets its local depth gives it) or "skewed"
//     (2^1 times). dytiss forces skewed from the start, dytisu forces uniform from the start: the knob to sweep.
//   - the insertion order of bulk_load. DyTIS is built by inserts, so its structure depends on their order. Without a suffix
//     bulk_load inserts the keys in a random order (seed fixed), which is the order GRE's own insert workload uses and the
//     order of the paper's shuffled data sets. With "a" (dytisa, dytissa, dytisua) it inserts them in ascending order, as
//     RoBin's GRE wrapper of DyTIS does. Ascending was 9 times faster to build on uniform keys and 18 to 53 times slower, with
//     up to 9 times the memory, on skewed and on dense keys (2.7M keys; docs/sota/dytis.md), and the cost grows with the
//     number of keys: not feasible at 200M on such data.
//
// The authors' headers define non-inline globals (the pools, the mode switch, y_mask, max_bucket_num): ONE translation unit of
// a program may include this header, and ONE DytisIndex may be alive at a time (the constructor throws std::logic_error for a
// second one). GRE deletes its index through indexInterface*, which has no virtual destructor, so under GRE the first index
// never counts as gone: ONE dytis name per GRE process (the repository's drivers run one index per process). Single-threaded,
// as the released code is.
// Memory: on keys in tight clusters far apart the authors' code takes kilobytes a key (docs/sota/dytis.md has measurements);
// set $DYTIS_MAX_GB in a benchmark (dytis@DYTIS_MAX_GB=100 through the registry) so that this ends in an exception, not in
// the kernel's out-of-memory killer.
// Layout: the authors build with -DSEP by default (keys and payloads of a segment in two arrays), and so does this header under
// g++. Their SEP code does arithmetic on void* (Directory.h:104), which clang rejects in C++: under clang, or with
// -DGRET_DYTIS_NOSEP, the header builds the authors' other target (DTS_noSEP: an array of pairs), and says so in bulk_load's
// line. Include path: -I <DyTIS> -I <DyTIS_shim, a directory holding boost/pool/pool_alloc.hpp without code>.
#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

#if !__has_include(<boost/pool/pool_alloc.hpp>)
#error "gret/sota/dytis.h: DyTIS's Directory.h includes <boost/pool/pool_alloc.hpp>; this header restates the two Boost.Pool classes it uses, so add a directory holding a file boost/pool/pool_alloc.hpp without code to the include path (readonly_competitors/DyTIS_shim; docs/sota/dytis.md)"
#endif
#if !__has_include("src/DyTIS.h") || !__has_include("util/pair.h")
#error "gret/sota/dytis.h: put the clone of https://github.com/unist-ssl/DyTIS on the include path (-I <DyTIS>)"
#endif

#if !defined(GRET_DYTIS_NOSEP) && !defined(__clang__)
#define GRET_DYTIS_SEP 1  // the authors' default target (make DTS)
#else
#define GRET_DYTIS_SEP 0  // their DTS_noSEP target; clang cannot compile the SEP code (arithmetic on void*)
#endif

#if defined(__SANITIZE_ADDRESS__)
#define GRET_DYTIS_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define GRET_DYTIS_ASAN 1
#endif
#endif
#ifndef GRET_DYTIS_ASAN
#define GRET_DYTIS_ASAN 0
#endif

namespace gret {
namespace dytis_authors {

// ---- memory the authors' code holds through the two allocators below
struct MemCount {
  size_t pool_held = 0;  // bytes of the pools' blocks (chunks in use and free ones)
  size_t pool_live = 0;  // bytes of the chunks in use
  size_t big_live = 0;   // bytes of the segments of more than 20 buckets (SEP layout: the authors' malloc)
  size_t big_blocks = 0;
  size_t peak = 0;       // the most pool_held + big_live has been since the index was reset
  void note_peak() {
    if (pool_held + big_live > peak) peak = pool_held + big_live;
  }
};
inline MemCount g_mem;
inline size_t g_byte_cap = ~size_t(0);  // the most the pools and the large segments may hold together ($DYTIS_MAX_GB; none by default)

// Thrown by pool_allocator::allocate when the authors' Insert asks for a segment header it may not have: see
// DytisIndex::insert_slow. g_split_allocs is the number of headers it may still take (two per split).
struct DepthLimit {};
inline long g_split_allocs = 0x7fffffffL;

enum : size_t { kChunkPad = 16, kBigPad = 32 };

// What the authors' code calls malloc and free (Directory.h:109, :204 and Directory_impl.h: a segment of more than 20
// buckets, SEP layout). Unqualified calls inside this namespace find these two. The block is counted, and its payload has
// 24 bytes of 0xff in front (the size sits before them): see the pad of the pool below.
inline void* malloc(size_t n) {
  const uint64_t total = uint64_t(n) + kBigPad;
  if (total > g_byte_cap || g_mem.big_live + g_mem.pool_held > g_byte_cap - total) throw std::bad_alloc();
  char* raw = static_cast<char*>(std::malloc(total));
  if (!raw) throw std::bad_alloc();
  std::memcpy(raw, &total, sizeof(total));
  std::memset(raw + sizeof(total), 0xff, kBigPad - sizeof(total));
  g_mem.big_live += total;
  ++g_mem.big_blocks;
  g_mem.note_peak();
  return raw + kBigPad;
}
inline void free(void* p) {
  if (!p) return;
  char* raw = static_cast<char*>(p) - kBigPad;
  uint64_t total;
  std::memcpy(&total, raw, sizeof(total));
  g_mem.big_live -= total;
  --g_mem.big_blocks;
  std::free(raw);
}

// ---- Boost.Pool, the part DyTIS uses, restated (Boost is not installed here; nothing of Boost's source is copied).
// boost::pool<>(size): malloc() pops a chunk of `size` bytes off a free list, free() pushes it back; an empty list takes a
// new block from the system, of 32 chunks the first time and twice as many each time after (Boost's next_size, start 32, no
// upper limit by default). Differences from Boost: every chunk has `pad` bytes of 0xff in front of it (the pools DyTIS
// declares: 16; the pool behind pool_allocator: none), the blocks are counted in g_mem, and under AddressSanitizer every
// chunk is its own malloc block, so that the sanitizer sees its bounds.
namespace boost {

struct default_user_allocator_new_delete {};

template <class UserAllocator = default_user_allocator_new_delete>
class pool {
 public:
  explicit pool(size_t requested_size, size_t next_size = 32, size_t pad = kChunkPad)
      : pad_(pad), stride_(pad + ((requested_size + 15) & ~size_t(15))), start_(next_size), next_(next_size) {}
  pool(const pool&) = delete;
  pool& operator=(const pool&) = delete;
  ~pool() { purge_memory(); }

  void* malloc() {
#if GRET_DYTIS_ASAN
    if (stride_ > g_byte_cap || g_mem.big_live + g_mem.pool_held > g_byte_cap - stride_) throw std::bad_alloc();
    char* raw = static_cast<char*>(std::malloc(stride_));
    if (!raw) throw std::bad_alloc();
    std::memset(raw, 0xff, pad_);
    chunks_.insert(raw);
    g_mem.pool_held += stride_;
    g_mem.pool_live += stride_;
    g_mem.note_peak();
    return raw + pad_;
#else
    if (!free_) grow();
    char* p = free_;
    std::memcpy(&free_, p, sizeof(free_));
    g_mem.pool_live += stride_;
    ++live_;
    return p;
#endif
  }

  void free(void* p) {
    if (!p) return;
    g_mem.pool_live -= stride_;
#if GRET_DYTIS_ASAN
    char* raw = static_cast<char*>(p) - pad_;
    chunks_.erase(raw);
    g_mem.pool_held -= stride_;
    std::free(raw);
#else
    char* c = static_cast<char*>(p);
    std::memcpy(c, &free_, sizeof(free_));
    free_ = c;
    --live_;
#endif
  }

  // every block back to the system, whether its chunks were freed or not (Boost's purge_memory)
  bool purge_memory() {
#if GRET_DYTIS_ASAN
    const bool any = !chunks_.empty();
    for (char* raw : chunks_) {
      g_mem.pool_held -= stride_;
      g_mem.pool_live -= stride_;
      std::free(raw);
    }
    chunks_.clear();
    return any;
#else
    const bool any = blocks_ != nullptr;
    while (blocks_) {
      Block* b = blocks_;
      blocks_ = b->next;
      g_mem.pool_held -= b->bytes;
      std::free(b);
    }
    g_mem.pool_live -= live_ * stride_;
    live_ = 0;
    free_ = nullptr;
    next_ = start_;
    return any;
#endif
  }

  size_t chunk_bytes() const { return stride_; }

 private:
  struct Block {
    Block* next;
    size_t bytes, chunks;
  };
  enum : size_t { kHeader = 32 };  // sizeof(Block), rounded to keep chunks 16-aligned

#if !GRET_DYTIS_ASAN
  void grow() {
    const size_t n = next_;
    const size_t bytes = kHeader + n * stride_;
    if (bytes > g_byte_cap || g_mem.big_live + g_mem.pool_held > g_byte_cap - bytes) throw std::bad_alloc();
    char* raw = static_cast<char*>(std::malloc(bytes));
    if (!raw) throw std::bad_alloc();
    Block* b = reinterpret_cast<Block*>(raw);
    b->next = blocks_;
    b->bytes = bytes;
    b->chunks = n;
    blocks_ = b;
    g_mem.pool_held += bytes;
    g_mem.note_peak();
    char* base = raw + kHeader;
    for (size_t i = n; i-- > 0;) {  // the free list in address order, as Boost's segregate leaves it
      char* c = base + i * stride_;
      std::memset(c, 0xff, pad_);
      char* payload = c + pad_;
      std::memcpy(payload, &free_, sizeof(free_));
      free_ = payload;
    }
    next_ = n << 1;
  }
#endif

  size_t pad_, stride_, start_, next_, live_ = 0;
  char* free_ = nullptr;
  Block* blocks_ = nullptr;
#if GRET_DYTIS_ASAN
  std::unordered_set<char*> chunks_;
#endif
};

// boost::pool_allocator<T>, as DyTIS uses it for its segment headers: allocate(1) from one pool per type, destroy(p) runs the
// destructor and returns no memory (the authors never call deallocate: a segment header replaced by a split stays allocated).
template <class T, class UserAllocator = default_user_allocator_new_delete>
class pool_allocator {
 public:
  typedef T value_type;
  static pool<UserAllocator>& the_pool() {
    static pool<UserAllocator> p(sizeof(T), 32, 0);
    return p;
  }
  T* allocate(size_t n) {
    if (n != 1) throw std::bad_alloc();
    if (g_split_allocs <= 0) throw DepthLimit();
    --g_split_allocs;
    return static_cast<T*>(the_pool().malloc());
  }
  void deallocate(T* p, size_t) { the_pool().free(p); }
  void destroy(T* p) { p->~T(); }
};

}  // namespace boost
}  // namespace dytis_authors
}  // namespace gret

// ---- the authors' sources, unmodified, in a namespace of their own. Every standard header they include is included above,
// so inside the namespace those includes are no-ops; their <boost/pool/pool_alloc.hpp> finds the empty file (or, should Boost
// be installed after all, Boost's own, switched off through its include guard: the classes above are the ones used).
// Their assert()s are compiled out, as in the authors' own build (Makefile: -DNDEBUG): none was seen to fail, but one of them
// casts a negative double to uint64_t inside its condition (Directory_impl.h:1055), which is undefined by itself.
#pragma push_macro("SEP")
#pragma push_macro("BOOST_POOL_ALLOC_HPP")
#pragma push_macro("NDEBUG")
#ifndef NDEBUG
#define NDEBUG
#endif
#include <cassert>
#undef SEP
#if GRET_DYTIS_SEP
#define SEP
#endif
#ifndef BOOST_POOL_ALLOC_HPP
#define BOOST_POOL_ALLOC_HPP
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wpointer-arith"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wparentheses"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wnull-conversion"
#pragma GCC diagnostic ignored "-Wsometimes-uninitialized"
#pragma GCC diagnostic ignored "-Wunused-private-field"
#else
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wconversion-null"
#endif
#endif
namespace gret {
namespace dytis_authors {
#include "util/pair.h"
#include "src/DyTIS.h"
#include "src/DyTIS_impl.h"
enum : int {
  gret_addr_bits = ADDR_BITS,
  gret_local_depth_bits = LOCAL_DEPTH_BITS,
  gret_skewed_max_bits = SKEWED_MAX_BITS,
  gret_uniform_max_bits = UNIFORM_MAX_BITS,
  gret_remap_thre = REMAP_THRE,
  gret_range_bits_limit = RANGE_BITS_LIMIT
};
}  // namespace dytis_authors
}  // namespace gret
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#undef DO_NOTHING
#undef INITIAL_RANGE_BITS
#undef RANGE_BITS_LIMIT
#undef REMAP_THRE
#undef BUC_THRE
#undef RECLAIM_THRE
#undef SKEWED_MAX_BITS
#undef UNIFORM_MAX_BITS
#undef ADDR_BITS
#undef DIRECTORY_BITS
#undef LOCAL_DEPTH_BITS
#pragma pop_macro("NDEBUG")
#pragma pop_macro("BOOST_POOL_ALLOC_HPP")
#pragma pop_macro("SEP")
#include <cassert>  // assert() again as the including file has it

namespace gret {

enum class DytisMode { released, skewed, uniform };

template <class K, class P>
class DytisIndex final : public indexInterface<K, P> {
  using D = dytis_authors::DyTIS;
  using Dir = dytis_authors::Directory;
  using Eh = dytis_authors::ExtendibleHash;
  static_assert(std::is_integral<K>::value && std::is_unsigned<K>::value && sizeof(K) == 8, "DytisIndex: keys are uint64");
  static_assert(std::is_integral<P>::value && sizeof(P) == 8, "DytisIndex: payloads are 64-bit integers");
  // DyTIS::EH (its table of 512 first-level tables) is private and is its only member: a standard-layout class is
  // pointer-interconvertible with its first member
  static_assert(std::is_standard_layout<D>::value && sizeof(D) == sizeof(void*), "DyTIS is no longer one pointer: table() needs another way in");
  // the authors' tagged pointers put the local depth above bit 59 by adding (depth << 53) to a Directory*, in units of its size
  static_assert(sizeof(Dir) == 64, "the authors' pointer tags need sizeof(Directory) == 64 (Directory.h, DIRECTORY_BITS)");

 public:
  static constexpr uint64_t kInvalid = ~uint64_t(0);  // the authors' INVALID: an empty key slot
  // Keys that never enter DyTIS: the last kEdge keys of every aligned block of 2^24 keys (their low 24 bits are 2^24 - kEdge
  // or more). A segment covers an aligned range of 2^(55 - depth) keys, depth <= 31, so the top of every segment's range is
  // the top of such a block, and there the authors' model can send a key one bucket past the segment's last (see
  // check_model). 2^64 - 1 (INVALID) and 2^64 - 2 (SENTINEL in util/pair.h, unused) are two of these keys. kEdge / 2^24 of
  // uniformly spread keys are of this kind: 4 in a million.
  static constexpr uint64_t kEdge = 64;
  static constexpr unsigned kEdgeBits = 24;
  static bool edge_key(uint64_t k) { return ((k + kEdge) & ((uint64_t(1) << kEdgeBits) - 1)) < kEdge; }
  static constexpr unsigned kMaxLd = (1u << dytis_authors::gret_local_depth_bits) - 1;  // 31: the largest local depth a tag holds
  static constexpr size_t kB = dytis_authors::kNumSlot;    // slots of a bucket (128)
  static_assert((kB & (kB - 1)) == 0, "the cursor rounds slot indexes to buckets with a mask");

  explicit DytisIndex(std::string name, DytisMode mode = DytisMode::released, bool random_order = true)
      : name_(std::move(name)), mode_(mode), random_order_(random_order) {
    if (poisoned())
      throw std::logic_error(name_ + ": an earlier DytisIndex of this process ran out of memory inside the authors' code, which keeps scratch state in function statics "
                                     "(Directory_impl.h:31, 50) and leaves it undefined then: no further DyTIS in this process");
    if (alive()) throw std::logic_error(name_ + ": one DytisIndex at a time (the authors' pools and mode switch are globals)");
    if (const char* e = std::getenv("DYTIS_MAX_DEPTH")) {
      const long v = std::atol(e);
      if (v < 1 || v > long(kMaxLd)) throw std::invalid_argument(name_ + ": DYTIS_MAX_DEPTH must be in [1, 31]");
      max_depth_ = unsigned(v);
    }
    dytis_authors::g_byte_cap = ~size_t(0);
    if (const char* e = std::getenv("DYTIS_MAX_GB")) {
      const double v = std::atof(e);
      if (!(v > 0)) throw std::invalid_argument(name_ + ": DYTIS_MAX_GB must be a positive number");
      if (v < 1e9) dytis_authors::g_byte_cap = size_t(v * 1073741824.0);
    }
    alive() = true;
    reset();
  }
  DytisIndex(const DytisIndex&) = delete;
  DytisIndex& operator=(const DytisIndex&) = delete;
  ~DytisIndex() {  // indexInterface has no virtual destructor: deleted through it, as GRE does, nothing is freed
    teardown();
    alive() = false;
  }

  void init(Param* = nullptr) override {}

  // kv: sorted by key, keys unique (GRE's contract; checked). One authors' Insert per key.
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (poisoned()) throw std::logic_error(name_ + ": the authors' code ran out of memory in this process and cannot be used again (see the constructor)");
    reset();
    for (size_t i = 1; i < n; ++i)
      if (uint64_t(kv[i].first) <= uint64_t(kv[i - 1].first)) throw std::invalid_argument(name_ + ": bulk_load needs sorted, unique keys");
    auto one = [&](size_t i) {
      const uint64_t k = uint64_t(kv[i].first), v = uint64_t(kv[i].second);
      if (edge_key(k)) {
        side_.emplace(k, v), ++why_[0];
      } else if (insert(k, v, false) == 2) {
        side_.emplace(k, v), ++why_[1], ++side_plain_;
      }
    };
    if (!random_order_) {
      for (size_t i = 0; i < n; ++i) one(i);
    } else {
      if (n > size_t(UINT32_MAX)) throw std::invalid_argument(name_ + ": the random insert order is for fewer than 2^32 keys");
      std::vector<uint32_t> order(n);
      for (size_t i = 0; i < n; ++i) order[i] = uint32_t(i);
      std::mt19937_64 g(0x9e3779b97f4a7c15ull);
      for (size_t i = n; i > 1; --i) std::swap(order[i - 1], order[g() % i]);
      for (size_t i = 0; i < n; ++i) one(order[i]);
    }
    const double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    // the walk scan relies on: exactly the keys given (but those in the side store), ascending, with their payloads; and the
    // lookup get() runs (the authors' Find) must lead to the very slot the walk is on. The second check is not implied by
    // the first: Find evaluates the segment's model in its own inlined copy of lcdf, and a compiler that fused the multiply
    // and the add in one copy and not in another would send a key on a bucket boundary to two different buckets. In key
    // order this costs 7 ns a key on the Xeon (2M keys), against 250 to 500 for the insert.
    {
      Cursor c;
      bool ok = first(c);
      size_t held = 0;
      for (size_t i = 0; i < n; ++i) {
        const uint64_t k = uint64_t(kv[i].first);
        if (may_be_in_side(k) && side_.count(k)) continue;
        if (!ok || key_at(c.t, c.i) != k || val_at(c.t, c.i) != uint64_t(kv[i].second)) broken("after the bulk load the authors' structure does not hold the keys given, in order; first difference at key " + std::to_string(k));
        if (find(k) != &val_at(const_cast<Dir*>(c.t), c.i)) broken("after the bulk load the authors' Find does not lead to the slot that holds key " + std::to_string(k));
        ++held;
        ok = next(c);
      }
      if (ok || held != n_) broken("after the bulk load the authors' structure holds keys that were not given");
    }
    side_reported_ = side_.size();
    const Shape s = shape();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr,
                 "gret: %s (the DyTIS authors' code, unist-ssl/DyTIS; layout=%s; Boost.Pool restated) n=%zu in_dytis=%zu side=%zu (edge_keys=%zu depth_limit=%zu) "
                 "mode=%s%s order=%s tables=%zu segments=%zu buckets=%zu fill=%.3f max_global_depth=%u max_local_depth=%u modelled_segments=%zu "
                 "restructuring_inserts=%zu models_past_last_bucket=%zu (at most %zu keys each) bytes_per_key=%.2f (pool blocks %.2f, of which in use %.2f; large segments %.2f; directories %.2f) "
                 "peak_bytes_per_key=%.2f (%s) insert_ns_per_key=%.1f build_ms=%.1f total_ms=%.1f\n",
                 name_.c_str(), GRET_DYTIS_SEP ? "SEP" : "noSEP", n, n_, side_.size(), why_[0], why_[1], mode_now(), mode_ == DytisMode::released ? " (chosen by the released code)" : " (forced)",
                 random_order_ ? "random" : "ascending", s.tables, s.segments, s.buckets, s.buckets ? double(n_) / double(s.buckets * kB) : 0.0, s.max_gd, s.max_ld, s.modelled,
                 slow_, edge_models_, edge_worst_, n ? double(memory_consumption()) / double(n) : 0.0, n ? double(dytis_authors::g_mem.pool_held) / double(n) : 0.0,
                 n ? double(dytis_authors::g_mem.pool_live) / double(n) : 0.0, n ? double(s.big_bytes) / double(n) : 0.0, n ? double(s.dir_bytes) / double(n) : 0.0,
                 n ? double(dytis_authors::g_mem.peak) / double(n) : 0.0, GRET_DYTIS_SEP ? "pools and large segments at their most during the load" : "pools only: this layout's large segments are not tracked",
                 n ? build_ms * 1e6 / double(n) : 0.0, build_ms, ms);
  }

  bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t k = uint64_t(key);
    if (__builtin_expect(!edge_key(k), 1)) {
      const dytis_authors::Value_t* p = find(k);  // the authors' lookup
      if (p) {
        val = P(*p);
        return true;
      }
    }
    if (!may_be_in_side(k)) return false;
    const auto it = side_.find(k);
    if (it == side_.end()) return false;
    val = P(it->second);
    return true;
  }

  // true if the key was not there and is now; false, and nothing changes, if it was
  bool put(K key, P value, Param* = nullptr) override {
    const uint64_t k = uint64_t(key);
    if (may_be_in_side(k) && side_.count(k)) return false;
    if (!edge_key(k)) {
      const int r = insert(k, uint64_t(value), true);
      if (r != 2) return r == 1;
      ++side_plain_;
    }
    ++put_side_;  // a key the authors' Insert was not given
    side_.emplace(k, uint64_t(value));
    return true;
  }

  bool update(K key, P value, Param* = nullptr) override {
    const uint64_t k = uint64_t(key);
    if (may_be_in_side(k)) {
      const auto it = side_.find(k);
      if (it != side_.end()) {
        it->second = uint64_t(value);
        return true;
      }
    }
    if (edge_key(k)) return false;
    dytis_authors::Value_t* p = find(k);  // the authors' Update is Find and an assignment (DyTIS_impl.h:158)
    if (!p) return false;
    *p = uint64_t(value);
    return true;
  }

  bool remove(K key, Param* = nullptr) override {
    const uint64_t k = uint64_t(key);
    if (may_be_in_side(k)) {
      const auto it = side_.find(k);
      if (it != side_.end()) {
        side_.erase(it);
        if (!edge_key(k)) --side_plain_;
        return true;
      }
    }
    if (edge_key(k)) return false;
    Loc l;
    if (!locate(k, l)) return false;
    if (l.z >= l.t->seg_num) broken(outside(k));
    const size_t base = l.z * kB, pos = lower(l.t, base, k);
    if (pos == kB || key_at(l.t, base + pos) != k) return false;
    if (key_at(l.t, base + kB - 1) != kInvalid || key_at(l.t, base) == 0) {
      // the two cases the authors' Delete gets wrong (see the head of this file): the gap is closed here
      size_t j = pos;
      for (; j + 1 < kB && key_at(l.t, base + j + 1) != kInvalid; ++j) {
        key_at(l.t, base + j) = key_at(l.t, base + j + 1);
        val_at(l.t, base + j) = val_at(l.t, base + j + 1);
      }
      key_at(l.t, base + j) = kInvalid;
      val_at(l.t, base + j) = kInvalid;
      --l.t->num_key;
      ++own_removes_;
    } else {
      dytis_authors::Key_t kk = k;
      l.t->Delete(kk, size_t(k & dytis_authors::y_mask), l.z, false, int(l.ld));  // the authors' Directory::Delete, on the bucket found above
    }
    --n_;
    return true;
  }

  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    const uint64_t k = uint64_t(key);
    size_t got = 0;
    auto sit = side_.empty() ? side_.end() : side_.lower_bound(k);
    Cursor c;
    bool ok = seek(k, c);
    if (sit == side_.end()) {  // the usual case: nothing to merge
      while (ok && got < n) {
        out[got++] = std::pair<K, P>(K(key_at(c.t, c.i)), P(val_at(c.t, c.i)));
        ok = next(c);
      }
      return got;
    }
    while (ok && got < n) {
      const uint64_t dk = key_at(c.t, c.i);
      for (; sit != side_.end() && sit->first < dk && got < n; ++sit) out[got++] = std::pair<K, P>(K(sit->first), P(sit->second));
      if (got == n) break;
      out[got++] = std::pair<K, P>(K(dk), P(val_at(c.t, c.i)));
      ok = next(c);
    }
    for (; sit != side_.end() && got < n; ++sit) out[got++] = std::pair<K, P>(K(sit->first), P(sit->second));
    return got;
  }

  // Every byte held: the blocks of the authors' pools (chunks in use and free ones, with the pads), their large segments,
  // the directories of the first-level tables, the models too large for a pool, the side store's tree nodes, this object.
  // Not the allocator's own overhead per block.
  long long memory_consumption() override {
    const Shape s = shape();
    const size_t b = sizeof(*this) + side_.size() * kMapNode + sizeof(D) + dytis_authors::kCapacity * sizeof(void*) +
                     dytis_authors::g_mem.pool_held + s.big_bytes + s.dir_bytes + s.line_bytes;
    // GRE asks for the size once, after the run, and gives no other hook: say here how many keys the side store holds now,
    // whenever that changed since the last line
    if (side_.size() != side_reported_) {
      side_reported_ = side_.size();
      const size_t all = n_ + side_.size();
      std::fprintf(stderr, "gret: %s now in_dytis=%zu side=%zu (%.3f%% of the keys held; %zu puts went to the side store since the bulk load)\n", name_.c_str(), n_,
                   side_.size(), all ? 100.0 * double(side_.size()) / double(all) : 0.0, put_side_);
    }
    return (long long)b;
  }

  // for tests and reports
  size_t in_dytis() const { return n_; }
  size_t in_side() const { return side_.size(); }
  size_t puts_to_side() const { return put_side_; }           // puts since the bulk load that the authors' Insert was not given
  size_t restructuring_inserts() const { return slow_; }      // inserts into a full bucket: the authors' code split, remapped or expanded
  size_t own_removes() const { return own_removes_; }         // removes the wrapper closed itself
  size_t depth_refusals() const { return depth_refused_; }    // inserts stopped at the depth limit (the key is in the side store)
  unsigned max_depth() const { return max_depth_; }
  size_t edge_models() const { return edge_models_; }         // segment models checked that send the top keys of their range past the last bucket
  size_t edge_worst() const { return edge_worst_; }           // the most keys one of them sent there (kEdge are tolerated)
  const char* mode_now() const { return dytis_authors::DEFAULT_MAX_BITS == dytis_authors::gret_uniform_max_bits ? "uniform" : "skewed"; }
  static const char* layout() { return GRET_DYTIS_SEP ? "SEP" : "noSEP"; }
  struct Shape {
    size_t tables = 0, segments = 0, buckets = 0, modelled = 0, dir_bytes = 0, big_bytes = 0, line_bytes = 0;
    unsigned max_gd = 0, max_ld = 0;
  };
  Shape shape() const {
    Shape s;
    Eh** eh = table();
    for (size_t x = 0; x < dytis_authors::kCapacity; ++x) {
      const uint64_t e = uint64_t(eh[x]);
      if (!e) continue;
      const Eh* t = untag_eh(e);
      const unsigned gd = unsigned(e >> dytis_authors::gret_addr_bits);
      ++s.tables;
      s.max_gd = std::max(s.max_gd, gd);
      s.dir_bytes += sizeof(Eh) + (size_t(1) << gd) * sizeof(Dir*);
      for (size_t y = 0; y < (size_t(1) << gd);) {
        const uint64_t tag = uint64_t(t->seg[y]);
        const Dir* d = dir_of(tag);
        const unsigned ld = ld_of(tag);
        ++s.segments;
        s.buckets += d->seg_num;
        s.max_ld = std::max(s.max_ld, ld);
        if (d->line) {
          ++s.modelled;
          if (d->range_bits > 10) s.line_bytes += (size_t(1) << d->range_bits) * sizeof(dytis_authors::LineFriends);  // new[], not a pool
        }
#if !GRET_DYTIS_SEP
        if (d->seg_num > size_t(dytis_authors::pool_num)) s.big_bytes += d->seg_num * kB * sizeof(dytis_authors::Pair);  // new Pair[], not a pool
#endif
        y += size_t(1) << (gd - std::min(gd, ld));
      }
    }
#if GRET_DYTIS_SEP
    s.big_bytes = dytis_authors::g_mem.big_live;
#endif
    return s;
  }

 private:
  static constexpr size_t kMapNode = 4 * sizeof(void*) + 2 * sizeof(uint64_t);  // a red-black tree node holding two uint64
  static constexpr long kNoBudget = 0x7fffffffL;

  // the side store holds edge keys, and other keys only after an insert was stopped at the depth limit (side_plain_ of them)
  bool may_be_in_side(uint64_t k) const { return edge_key(k) || side_plain_ != 0; }

  static bool& poisoned() {  // see out_of_memory
    static bool p = false;
    return p;
  }
  static bool& alive() {
    static bool a = false;
    return a;
  }

  // ---- the authors' structure, read by the wrapper
  static uint64_t& key_at(Dir* d, size_t i) {
#if GRET_DYTIS_SEP
    return d->key_slot[i].item;
#else
    return d->slot[i].key;
#endif
  }
  static uint64_t& val_at(Dir* d, size_t i) {
#if GRET_DYTIS_SEP
    return d->val_slot[i].item;
#else
    return d->slot[i].value;
#endif
  }
  static uint64_t key_at(const Dir* d, size_t i) { return key_at(const_cast<Dir*>(d), i); }
  static uint64_t val_at(const Dir* d, size_t i) { return val_at(const_cast<Dir*>(d), i); }

  Eh** table() const { return *reinterpret_cast<Eh***>(d_); }
  static Eh* untag_eh(uint64_t e) { return reinterpret_cast<Eh*>(e & dytis_authors::ADDR_MASK); }
  static Dir* dir_of(uint64_t tag) { return reinterpret_cast<Dir*>(tag & dytis_authors::ADDR_MASK); }
  static unsigned ld_of(uint64_t tag) { return unsigned(tag >> (64 - dytis_authors::gret_local_depth_bits)); }
  static unsigned hash_bits() { return unsigned(64 - dytis_authors::kDepth); }  // 55: the key bits below the first-level table's
  // the bucket the authors' model of segment d (local depth ld) sends key k to (ExtendibleHash_impl.h:176-180)
  static size_t bucket_of(Dir* d, unsigned ld, uint64_t k) {
    const unsigned sh = hash_bits() - ld;
    const uint64_t masked = k & ((uint64_t(1) << sh) - 1);
    return size_t(d->lcdf(int(ld), masked) >> sh);
  }

  struct Loc {
    Eh* eh;
    Dir* t;
    size_t x, y, z;
    unsigned gd, ld;
  };
  // where the authors' code looks for k (DyTIS_impl.h:149-155, ExtendibleHash_impl.h:170-181); false: no first-level table
  bool locate(uint64_t k, Loc& l) const {
    l.x = size_t(k >> hash_bits());
    const uint64_t e = uint64_t(table()[l.x]);
    if (!e) return false;
    l.eh = untag_eh(e);
    l.gd = unsigned(e >> dytis_authors::gret_addr_bits);
    const uint64_t h = k & dytis_authors::y_mask;
    l.y = size_t(h >> (hash_bits() - l.gd));
    const uint64_t tag = uint64_t(l.eh->seg[l.y]);
    l.t = dir_of(tag);
    l.ld = ld_of(tag);
    l.z = bucket_of(l.t, l.ld, k);  // may be seg_num for an edge key (see kEdge): the caller looks
    return true;
  }
  static std::string outside(uint64_t k) { return "the authors' model sends key " + std::to_string(k) + ", which is not an edge key, outside its segment"; }
  // the first slot of the bucket at `base` whose key is not below k (empty slots hold 2^64 - 1 and sort last); kB if there
  // is none. Seven steps without a data-dependent branch.
  static size_t lower(const Dir* d, size_t base, uint64_t k) {
    size_t lo = 0;
    for (size_t half = kB / 2; half; half /= 2) lo += key_at(d, base + lo + half - 1) < k ? half : 0;
    return lo + (key_at(d, base + lo) < k);
  }

  dytis_authors::Value_t* find(uint64_t k) const {
    dytis_authors::Key_t kk = k;
#if GRET_DYTIS_SEP
    return d_->Find(kk);  // the authors' lookup, whole
#else
    // noSEP: a segment of more than 20 buckets comes from new Pair[] and has no pad in front of it, so the read of the slot
    // before the first one (Directory_impl.h:814) is kept from happening: a key below every key of bucket 0 is not there
    Loc l;
    if (!locate(k, l)) return nullptr;
    if (l.z >= l.t->seg_num) const_cast<DytisIndex*>(this)->broken(outside(k));
    if (l.z == 0 && k < key_at(l.t, 0)) return nullptr;
    return l.t->Find(kk, l.z);  // the authors' search of the bucket
#endif
  }

  // 0: the key is stored (only looked for if may_exist), nothing changed; 1: inserted by the authors' code; 2: refused.
  // The bucket is found once, here, the way the authors' ExtendibleHash::Insert finds it (locate). With room in it the key goes
  // in through the authors' Directory::Insert, which is what their DyTIS::Insert would call after finding the same bucket a
  // second time; a full bucket and a missing first-level table go through their DyTIS::Insert, whole.
  int insert(uint64_t k, uint64_t v, bool may_exist) {
    Loc l;
    dytis_authors::Key_t kk = k;
    if (locate(k, l)) {
      if (l.z >= l.t->seg_num) broken(outside(k));
      const size_t base = l.z * kB;
      if (may_exist) {
        const size_t pos = lower(l.t, base, k);
        if (pos < kB && key_at(l.t, base + pos) == k) return 0;
      }
      if (key_at(l.t, base + kB - 1) != kInvalid) return insert_slow(k, v, l);  // a full bucket: the authors' code will restructure
      if (l.t->Insert(kk, v, size_t(k & dytis_authors::y_mask), l.z) < 0) broken("the authors' Directory::Insert refused key " + std::to_string(k) + " in a bucket with room");
      ++n_;
      return 1;
    }
    try {
      d_->Insert(kk, v);  // the authors' Insert: it creates the first-level table
    } catch (const std::bad_alloc&) {
      out_of_memory();
    }
    ++n_;
    return 1;
  }

  // std::bad_alloc inside the authors' code (no memory, or $DYTIS_MAX_GB reached) cannot be recovered from: their Expand and
  // LocalRemap change the segment before they allocate, and LocalRemap keeps its scratch in two function statics that only
  // its return paths clear. The index throws and stays unusable, and so does every DytisIndex made later in this process.
  [[noreturn]] void out_of_memory() {
    dytis_authors::g_split_allocs = kNoBudget;
    poisoned() = true;
    broken("out of memory inside the authors' Insert" + std::string(dytis_authors::g_byte_cap != ~size_t(0) ? " (the limit set by DYTIS_MAX_GB)" : ""));
  }

  // An insert into a full bucket: the authors' Insert remaps, expands or splits the segment, as often as it takes. Three things
  // are done around it.
  //  - The segment's pairs are counted and summed first (pair_hash), and check_range compares afterwards: one pass over the
  //    segment before and one after, where the authors' restructuring makes three or more.
  //  - Every split takes the key's segment one local depth down, and the depth lives in 5 bits of a tagged pointer: a split of
  //    a segment at depth 31 would write depth 32 as 0 (ExtendibleHash_impl.h:96, 104, 125) and the table would be garbage.
  //    Nothing in the authors' code stops there. A split starts by taking two segment headers from seg_alloc
  //    (Directory_impl.h:931, 960) before it changes anything, so the wrapper gives the allocator a budget of two headers
  //    per split still allowed; the allocator throws DepthLimit at the first header beyond it, the authors' Insert unwinds
  //    with the structure as the last completed step left it, and the key goes to the side store. The one thing the unwinding
  //    skips is the global depth, which the authors keep in a tag of the table pointer and write after the call
  //    (DyTIS_impl.h:63-68): it is written here.
  //  - std::bad_alloc inside cannot be recovered from: see out_of_memory.
  int insert_slow(uint64_t k, uint64_t v, const Loc& l) {
    ++slow_;
    const Dir* t = l.t;
    size_t count = 0;
    uint64_t sum = 0;
    for (size_t b = 0; b < t->seg_num; ++b)
      for (size_t i = b * kB, e = i + kB; i < e; ++i) {
        const uint64_t q = key_at(t, i);
        if (q == kInvalid) break;
        sum += pair_hash(q, val_at(t, i));
        ++count;
      }
    const uint64_t prefix = (k & dytis_authors::y_mask) >> (hash_bits() - l.ld);
    const long budget = l.ld < max_depth_ ? 2 * long(max_depth_ - l.ld) : 0;
    bool refused = false;
    dytis_authors::g_split_allocs = budget;
    try {
      dytis_authors::Key_t kk = k;
      d_->Insert(kk, v);  // the authors' Insert
    } catch (const dytis_authors::DepthLimit&) {
      refused = true;
    } catch (const std::bad_alloc&) {
      out_of_memory();
    }
    const unsigned splits = unsigned((budget - dytis_authors::g_split_allocs) / 2);
    dytis_authors::g_split_allocs = kNoBudget;
    if (refused) {
      const unsigned gd = std::max(l.gd, l.ld + splits);
      table()[l.x] = reinterpret_cast<Eh*>(uint64_t(l.eh) | (uint64_t(gd) << dytis_authors::gret_addr_bits));
      check_range(l.x, l.eh, prefix, l.ld, count, sum);
      ++depth_refused_;
      return 2;
    }
    ++n_;
    check_range(l.x, l.eh, prefix, l.ld, count + 1, sum + pair_hash(k, v));
    return 1;
  }

  // what a segment's pairs are summed with, before and after a restructuring insert: a lost, doubled or changed pair changes
  // the sum (but for one case in 2^64)
  static uint64_t pair_hash(uint64_t k, uint64_t v) {
    uint64_t m = (k ^ (v * 0x9e3779b97f4a7c15ull)) * 0xd6e8feb86659fd93ull;
    m ^= m >> 32;
    return m * 0xff51afd7ed558ccdull;
  }

  // After a restructuring insert: the segments that now cover what one segment (first-level table x, hash prefix `prefix` of
  // ld0 bits) covered must hold `count` pairs whose pair_hash sums to `sum`, the keys ascending, every key in a segment the
  // directory leads to and in the bucket the authors' model gives it (the model is monotone, check_model, so the first and
  // the last key of a bucket speak for those between them), with models that keep every key of their segment inside its
  // buckets, and chained by sibling.
  void check_range(size_t x, const Eh* eh, uint64_t prefix, unsigned ld0, size_t count, uint64_t sum) {
    const uint64_t e = uint64_t(table()[x]);
    const unsigned gd = unsigned(e >> dytis_authors::gret_addr_bits);
    if (untag_eh(e) != eh || gd < ld0 || gd > max_depth_) broken("the first-level table changed under a restructuring insert (global depth " + std::to_string(gd) + ")");
    Dir* const* seg = eh->seg;
    size_t y = size_t(prefix) << (gd - ld0);
    const size_t yend = size_t(prefix + 1) << (gd - ld0), cap = size_t(1) << gd;
    const Dir* prevd = y ? dir_of(uint64_t(seg[y - 1])) : nullptr;
    size_t idx = 0;
    uint64_t prev = 0, found = 0;
    while (y < yend) {
      const uint64_t tag = uint64_t(seg[y]);
      Dir* d = dir_of(tag);
      const unsigned ld = ld_of(tag);
      if (ld < ld0 || ld > gd) broken("a segment's local depth " + std::to_string(ld) + " does not fit its table (global depth " + std::to_string(gd) + ")");
      const size_t chunk = size_t(1) << (gd - ld);
      if (y & (chunk - 1)) broken("a segment's directory entries are not aligned");
      for (size_t j = 1; j < chunk; ++j)
        if (uint64_t(seg[y + j]) != tag) broken("a segment's directory entries differ");
      if (prevd && prevd->sibling != d) broken("the sibling chain skips a segment");
      check_model(d, ld);
      // the keys the directory sends to this segment
      const uint64_t seg_lo = (uint64_t(x) << hash_bits()) | (uint64_t(y >> (gd - ld)) << (hash_bits() - ld)), seg_hi = seg_lo + ((uint64_t(1) << (hash_bits() - ld)) - 1);
      for (size_t b = 0; b < d->seg_num; ++b) {
        const size_t base = b * kB;
        const uint64_t first = key_at(d, base);
        if (first == kInvalid) continue;
        for (size_t i = base, end = base + kB; i < end; ++i) {
          const uint64_t q = key_at(d, i);
          if (q == kInvalid) break;
          if (idx && q <= prev) broken("a restructuring insert left keys out of order near key " + std::to_string(q));
          found += pair_hash(q, val_at(d, i));
          prev = q;
          ++idx;
        }
        if (first < seg_lo || prev > seg_hi) broken("key " + std::to_string(first < seg_lo ? first : prev) + " is in a segment the directory does not lead to");
        if (bucket_of(d, ld, first) != b || bucket_of(d, ld, prev) != b)
          broken("key " + std::to_string(bucket_of(d, ld, first) != b ? first : prev) + " is not in the bucket the authors' model sends it to");
      }
      prevd = d;
      y += chunk;
    }
    if (idx != count || found != sum)
      broken("a restructuring insert lost, duplicated or changed pairs (" + std::to_string(idx) + " pairs where " + std::to_string(count) + " were due" + (idx == count ? ", and not the same ones" : "") + ")");
    const Dir* after = yend < cap ? dir_of(uint64_t(seg[yend])) : nullptr;
    if (!prevd || prevd->sibling != after) broken("the sibling chain is broken after a restructured segment");
  }

  // The authors' model of a segment is piecewise linear in doubles (Directory_impl.h:1156-1162): bucket = (gradient * key +
  // intercept) >> shift, per piece ("range"). A piece is monotone in the key, so looking at both ends of every piece, and that
  // the pieces follow one another, shows that buckets follow the key order over the whole segment. The top is where it goes
  // wrong: the pieces are fitted so that the range ends exactly on a bucket boundary, the product is rounded to a double, and
  // the last few keys of the range come out one bucket past the segment's last (keys 2^37 - 1 and 2^37 - 2 of a segment at
  // depth 18 with intercept 3.6e16, for one). The authors' code then reads and writes past the segment's arrays. Those keys
  // are among the edge keys, which the wrapper never gives to DyTIS; here it is checked that no other key is affected.
  void check_model(Dir* d, unsigned ld) {
    if (d->seg_num == 0) broken("a segment without buckets");
    if (!d->line) return;  // bucket = seg_num * key >> shift: exact
    const int rb = d->range_bits;
    if (rb < 1 || rb > dytis_authors::gret_range_bits_limit || unsigned(rb) > hash_bits() - ld) broken("a segment's model has " + std::to_string(rb) + " range bits");
    const unsigned sh = hash_bits() - ld;
    const uint64_t limit = uint64_t(1) << sh, one_range = limit >> rb;
    auto z = [&](uint64_t key) { return size_t(d->lcdf(int(ld), key) >> sh); };
    size_t last = 0;
    for (uint64_t r = 0; r < (uint64_t(1) << rb); ++r) {
      const size_t zs = z(r * one_range), ze = z((r + 1) * one_range - 1);
      if (zs < last || ze < zs || zs >= d->seg_num)
        broken("a segment's model runs backwards or leaves its buckets (local depth " + std::to_string(ld) + ", range " + std::to_string(r) + " of " + std::to_string(uint64_t(1) << rb) +
               ": buckets " + std::to_string(zs) + " to " + std::to_string(ze) + " of " + std::to_string(d->seg_num) + "; gradient " + std::to_string(d->line[r].gradient) + ", intercept " +
               std::to_string(d->line[r].y_intercept) + ", " + std::to_string(d->num_key) + " keys)");
      last = ze;
    }
    if (last < d->seg_num) return;
    ++edge_models_;
    // the keys past the last bucket are the top of the range (the model is monotone): all of them must be edge keys
    const uint64_t first_safe_from_top = limit - kEdge - 1;  // the largest key of the range that is not an edge key
    if (z(first_safe_from_top) >= d->seg_num)
      broken("a segment's model sends more than the last " + std::to_string(kEdge) + " keys of its range past its last bucket (local depth " + std::to_string(ld) + ", " + std::to_string(d->seg_num) +
             " buckets): keys the wrapper let in may have been written outside the segment");
    uint64_t lo = first_safe_from_top, hi = limit - 1;  // z(lo) inside, z(hi) outside
    while (hi - lo > 1) {
      const uint64_t mid = lo + (hi - lo) / 2;
      if (z(mid) >= d->seg_num) hi = mid;
      else lo = mid;
    }
    edge_worst_ = std::max<size_t>(edge_worst_, size_t(limit - hi));
  }

  [[noreturn]] void broken(const std::string& what) {
    broken_ = true;  // the structure is not walked again, not even to free it
    throw std::runtime_error(name_ + ": " + what + " (the authors' code; docs/sota/dytis.md)");
  }

  // ---- the keys in ascending order: a segment and a slot index in it, resting on a slot that holds a key
  struct Cursor {
    size_t x = 0;
    const Dir* t = nullptr;
    size_t i = 0;
  };
  // from (c.t, c.i), which may be an empty slot or one past a bucket, forward to the next slot with a key
  bool settle(Cursor& c) const {
    for (;;) {
      const size_t slots = c.t->seg_num * kB;
      while (c.i < slots) {
        if (key_at(c.t, c.i) != kInvalid) return true;
        c.i = (c.i / kB + 1) * kB;  // the rest of this bucket is empty
      }
      c.i = 0;
      c.t = c.t->sibling;  // the next segment of this first-level table
      while (!c.t) {
        Eh** eh = table();
        do {
          if (++c.x >= dytis_authors::kCapacity) return false;
        } while (!eh[c.x]);
        c.t = dir_of(uint64_t(untag_eh(uint64_t(eh[c.x]))->seg[0]));
      }
    }
  }
  bool first(Cursor& c) const {
    Eh** eh = table();
    for (c.x = 0; c.x < dytis_authors::kCapacity && !eh[c.x]; ++c.x) {}
    if (c.x >= dytis_authors::kCapacity) return false;
    c.t = dir_of(uint64_t(untag_eh(uint64_t(eh[c.x]))->seg[0]));
    c.i = 0;
    return settle(c);
  }
  bool seek(uint64_t k, Cursor& c) const {  // the first key >= k
    Loc l;
    if (locate(k, l)) {
      c.x = l.x;
      c.t = l.t;
      if (l.z >= l.t->seg_num) {  // an edge key at the top of the segment's range: every key of the segment is below it
        c.i = l.t->seg_num * kB;
        return settle(c);
      }
      c.i = l.z * kB + lower(l.t, l.z * kB, k);
      if (c.i < (l.z + 1) * kB && key_at(c.t, c.i) != kInvalid) return true;
      c.i = (l.z + 1) * kB;
      return settle(c);
    }
    Eh** eh = table();
    for (c.x = size_t(k >> hash_bits()) + 1; c.x < dytis_authors::kCapacity && !eh[c.x]; ++c.x) {}
    if (c.x >= dytis_authors::kCapacity) return false;
    c.t = dir_of(uint64_t(untag_eh(uint64_t(eh[c.x]))->seg[0]));
    c.i = 0;
    return settle(c);
  }
  bool next(Cursor& c) const {
    ++c.i;
    if ((c.i & (kB - 1)) != 0 && key_at(c.t, c.i) != kInvalid) return true;  // the next slot of the same bucket
    c.i = (c.i + kB - 1) / kB * kB;
    return settle(c);
  }

  // every segment and first-level table back to the authors' pools and the heap (their destructor frees only the table of
  // tables), then the pools' blocks back to the system
  void teardown() {
    if (d_ && !broken_) {
      Eh** eh = table();
      for (size_t x = 0; x < dytis_authors::kCapacity; ++x) {
        const uint64_t e = uint64_t(eh[x]);
        if (!e) continue;
        Eh* t = untag_eh(e);
        const unsigned gd = unsigned(e >> dytis_authors::gret_addr_bits);
        for (size_t y = 0; y < (size_t(1) << gd);) {
          const uint64_t tag = uint64_t(t->seg[y]);
          y += size_t(1) << (gd - std::min(gd, ld_of(tag)));
          dytis_authors::seg_alloc.destroy(dir_of(tag));
        }
        delete t;
        eh[x] = nullptr;
      }
    }
    if (d_ && !broken_) delete d_;
    if (broken_) dytis_authors::g_mem.big_live = dytis_authors::g_mem.big_blocks = 0;  // a broken structure is not walked: its large segments stay allocated, and are no longer counted
    d_ = nullptr;
    for (auto& p : dytis_authors::chunk_alloc) p.purge_memory();
    for (auto& p : dytis_authors::line_alloc) p.purge_memory();
    dytis_authors::boost::pool_allocator<Dir>::the_pool().purge_memory();
  }

  void reset() {
    teardown();
    broken_ = false;
    // the authors' mode switch (util/util.h:20-21): a one-time test in DyTIS::Insert sets DEFAULT_MAX_BITS to UNIFORM_MAX_BITS
    dytis_authors::UNIFORM_TEST = mode_ != DytisMode::released;
    dytis_authors::DEFAULT_MAX_BITS = mode_ == DytisMode::uniform ? int(dytis_authors::gret_uniform_max_bits) : int(dytis_authors::gret_skewed_max_bits);
    d_ = new D();
    side_.clear();
    side_plain_ = 0;
    n_ = 0;
    why_[0] = why_[1] = 0;
    slow_ = put_side_ = own_removes_ = side_reported_ = depth_refused_ = edge_models_ = edge_worst_ = 0;
    dytis_authors::g_split_allocs = kNoBudget;
    dytis_authors::g_mem.peak = 0;
  }

  std::string name_;
  DytisMode mode_;
  bool random_order_;
  D* d_ = nullptr;                        // the authors' index
  bool broken_ = false;
  size_t n_ = 0;                          // keys in the authors' index
  std::map<uint64_t, uint64_t> side_;     // what the authors' code cannot hold (see the head of this file)
  size_t side_plain_ = 0;                 // keys in it that are not edge keys (stopped at the depth limit)
  size_t why_[2] = {0, 0};                // at bulk load: reserved keys, keys refused for the depth limit
  size_t slow_ = 0, put_side_ = 0, own_removes_ = 0, side_reported_ = 0, depth_refused_ = 0;
  size_t edge_models_ = 0, edge_worst_ = 0;  // models seen that send the top keys of their range out, and the most keys one did
  unsigned max_depth_ = kMaxLd;           // the deepest segment the authors' Insert may make ($DYTIS_MAX_DEPTH, at most 31)
};

// "dytis" (the released code's own choice of mode), "dytiss" (skewed), "dytisu" (uniform), each optionally followed by "a"
// (bulk_load inserts the keys in ascending order instead of a random one); nullptr for any other name, and for key or payload
// types other than 64-bit integers (unsigned keys)
template <class K, class P>
indexInterface<K, P>* make_dytis(const std::string& name) {
  if constexpr (!(std::is_integral<K>::value && std::is_unsigned<K>::value && sizeof(K) == 8 && std::is_integral<P>::value && sizeof(P) == 8)) {
    (void)name;
    return nullptr;
  } else {
    if (name.compare(0, 5, "dytis") != 0) return nullptr;
    std::string rest = name.substr(5);
    bool random_order = true;
    if (!rest.empty() && rest.back() == 'a') random_order = false, rest.pop_back();
    if (rest.empty()) return new DytisIndex<K, P>(name, DytisMode::released, random_order);
    if (rest == "s") return new DytisIndex<K, P>(name, DytisMode::skewed, random_order);
    if (rest == "u") return new DytisIndex<K, P>(name, DytisMode::uniform, random_order);
    return nullptr;
  }
}

}  // namespace gret
