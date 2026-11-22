// SPDX-License-Identifier: MIT
// ht_<table>[<L>]: established open-source hash tables as GRE indexes, the comparison points for hf4 (hashline.h).
// Each one is its authors' own code, unmodified, included by path; this file only maps GRE's interface onto it.
// No key order: scan returns 0. get, put, update and remove are all supported.
//
//   ht_std      std::unordered_map (the floor): chained nodes, std::hash<uint64_t> (the identity in libstdc++ and
//               libc++), max load factor 1.0
//   ht_ankerl   ankerl::unordered_dense::map (martinus/unordered_dense): the pairs in one dense std::vector, an index of
//               16-slot groups of one-byte fingerprints over it; its own hash (one 64x64->128 multiply by
//               0x9E3779B97F4A7C15, the halves xored), max load factor 0.8
//   ht_phmap    phmap::flat_hash_map (greg7mdp/parallel-hashmap): the header-only port of Abseil's Swiss table
//               (absl::flat_hash_map itself needs Abseil's compiled library); its own hash (one multiply by
//               0xde5fb9d2630458e9, the halves added), load factor fixed at 7/8, so it takes no <L>
//   ht_robin    tsl::robin_map (Tessil/robin-map): robin hood, the pair inside the bucket, power-of-two bucket counts,
//               max load factor 0.5. Its default hash is std::hash, the identity, which its README warns against with
//               power-of-two counts ("use a better hash function and/or tsl::robin_pg_map"): this name takes the
//               better hash, MumHash below (ankerl's integer hash, restated here), and
//   ht_robinpg  tsl::robin_pg_map takes the other advice: std::hash with prime bucket counts. Max load factor 0.5
//
//   <L>         the max load factor in percent, written without leading zeros (ht_robin90, ht_ankerl95, ht_std200),
//               set before the one reserve() of bulk_load. Without it the table keeps its author's default. Accepted:
//               ht_std 10..1000, ht_ankerl 10..100, ht_robin and ht_robinpg 20..95 (the library clamps to that);
//               anything else throws std::invalid_argument. ankerl, phmap and robin round the bucket count up to a
//               power of two, so <L> moves bytes per key in steps of 2x of the index (docs/sota/hashtable.md).
//
// bulk_load: clear, max_load_factor(L), reserve(n) once, then one insert per pair in the order given (ascending keys).
// put inserts (try_emplace) and returns true, or returns false and changes nothing if the key is stored (as GRE's ALEX and B+tree
// wrappers do); update overwrites a stored key's payload and returns false for an absent key; remove erases.
// memory_consumption: every byte the table has asked its allocator for and not yet returned, plus the table object.
// The allocator is CountingAlloc, std::allocator with a byte counter: it adds two additions to each allocate and
// deallocate, nothing to a lookup (measured: tests/sota/test_hashtable.cpp, docs/sota/hashtable.md). What malloc adds
// around each block (std::unordered_map: one block per key) is not counted.
// One thread writes at a time, as in GRE's other single-threaded indexes; concurrent get without writers is safe.
//
// A table whose header is not on the include path is left out (make_hashtable then throws for its name). Include paths:
//   -I<unordered_dense>/include -I<robin-map>/include -I<parallel-hashmap>
#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#if defined(__has_include)
#if __has_include("ankerl/unordered_dense.h")
#include "ankerl/unordered_dense.h"
#define GRET_SOTA_HAVE_ANKERL 1
#endif
#if __has_include("tsl/robin_map.h")
#include "tsl/robin_map.h"
#define GRET_SOTA_HAVE_ROBIN 1
#endif
#if __has_include("parallel_hashmap/phmap.h")
#include "parallel_hashmap/phmap.h"
#define GRET_SOTA_HAVE_PHMAP 1
#endif
#endif
#ifndef GRET_SOTA_HAVE_ANKERL
#define GRET_SOTA_HAVE_ANKERL 0
#endif
#ifndef GRET_SOTA_HAVE_ROBIN
#define GRET_SOTA_HAVE_ROBIN 0
#endif
#ifndef GRET_SOTA_HAVE_PHMAP
#define GRET_SOTA_HAVE_PHMAP 0
#endif

namespace gret {
namespace sota {

struct ByteCount {
  size_t live = 0, peak = 0;  // bytes asked for and not yet returned; the most that ever was
};

// std::allocator that adds what it hands out to a ByteCount. Stateful (one pointer), equal when the counter is shared.
template <class T>
class CountingAlloc {
 public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;
  using is_always_equal = std::false_type;

  CountingAlloc() noexcept = default;  // counts nothing: only for containers that must default-construct one
  explicit CountingAlloc(ByteCount* count) noexcept : count_(count) {}
  template <class U>
  CountingAlloc(const CountingAlloc<U>& o) noexcept : count_(o.counter()) {}

  T* allocate(size_t n) {
    T* p = std::allocator<T>().allocate(n);
    if (count_) {
      count_->live += n * sizeof(T);
      if (count_->live > count_->peak) count_->peak = count_->live;
    }
    return p;
  }
  void deallocate(T* p, size_t n) noexcept {
    if (count_) count_->live -= n * sizeof(T);
    std::allocator<T>().deallocate(p, n);
  }
  ByteCount* counter() const noexcept { return count_; }
  template <class U> bool operator==(const CountingAlloc<U>& o) const noexcept { return count_ == o.counter(); }
  template <class U> bool operator!=(const CountingAlloc<U>& o) const noexcept { return count_ != o.counter(); }

 private:
  ByteCount* count_ = nullptr;
};

// The integer hash of ankerl::unordered_dense (hash_int: one 64x64->128 multiply by the golden-ratio constant, low half
// xor high half), restated so that ht_robin does not depend on that header. Good in the low bits, which a power-of-two
// bucket count reads. A bijection it is not, and need not be: the tables compare whole keys.
struct MumHash {
  size_t operator()(uint64_t x) const noexcept {
    __extension__ typedef unsigned __int128 u128;
    const u128 r = u128(x) * 0x9E3779B97F4A7C15ull;
    return size_t(uint64_t(r) ^ uint64_t(r >> 64));
  }
};

template <class K, class P>
using StdTable = std::unordered_map<K, P, std::hash<K>, std::equal_to<K>, CountingAlloc<std::pair<const K, P>>>;
#if GRET_SOTA_HAVE_ANKERL
template <class K, class P>
using AnkerlTable = ankerl::unordered_dense::map<K, P, ankerl::unordered_dense::hash<K>, std::equal_to<K>, CountingAlloc<std::pair<K, P>>>;
#endif
#if GRET_SOTA_HAVE_ROBIN
template <class K, class P>
using RobinTable = tsl::robin_map<K, P, MumHash, std::equal_to<K>, CountingAlloc<std::pair<K, P>>>;
template <class K, class P>
using RobinPgTable = tsl::robin_pg_map<K, P, std::hash<K>, std::equal_to<K>, CountingAlloc<std::pair<K, P>>>;
#endif
#if GRET_SOTA_HAVE_PHMAP
template <class K, class P>
using PhmapTable = phmap::flat_hash_map<K, P, phmap::Hash<K>, phmap::EqualTo<K>, CountingAlloc<std::pair<const K, P>>>;
#endif

// The max load factor a table runs at, for the log line of bulk_load: what it reports, except phmap, whose
// max_load_factor() is a stub that returns 1 while the table grows at 7/8 (CapacityToGrowth: capacity - capacity / 8).
template <class Table>
double real_max_load_factor(const Table& t) { return double(t.max_load_factor()); }
#if GRET_SOTA_HAVE_PHMAP
template <class K, class P>
double real_max_load_factor(const PhmapTable<K, P>&) { return 0.875; }
#endif

// max_lf: the table's max load factor, or 0 to keep its default (and for tables that have none to set: phmap)
template <class K, class P, class Table>
class HashTableIndex final : public indexInterface<K, P> {
 public:
  explicit HashTableIndex(std::string name, float max_lf = 0)
      : name_(std::move(name)), max_lf_(max_lf), table_(typename Table::allocator_type(&bytes_)) {}
  HashTableIndex(const HashTableIndex&) = delete;  // the table's allocator points at bytes_
  HashTableIndex& operator=(const HashTableIndex&) = delete;

  void init(Param* = nullptr) override {}

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    table_.clear();
    if (max_lf_ > 0) table_.max_load_factor(max_lf_);
    table_.reserve(n);
    for (size_t i = 0; i < n; ++i) table_.emplace(kv[i].first, kv[i].second);
    if (table_.size() != n) throw std::invalid_argument("HashTableIndex: bulk_load keys are not unique");
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu buckets=%zu load=%.4f max_load=%.3f bytes_per_key=%.2f peak_bytes_per_key=%.2f build_ms=%.1f\n",
                 name_.c_str(), n, size_t(table_.bucket_count()), n ? double(table_.load_factor()) : 0.0, real_max_load_factor(table_),
                 n ? double(memory_consumption()) / double(n) : 0.0, n ? double(bytes_.peak + sizeof(table_)) / double(n) : 0.0, ms);
  }

  bool get(K key, P& val, Param* = nullptr) override {
    const auto it = table_.find(key);
    if (it == table_.end()) return false;
    val = it->second;
    return true;
  }

  // try_emplace, not emplace: the same insert in the flat tables, but libstdc++'s std::unordered_map::emplace (up to
  // GCC 11 at least) allocates a node before it looks the key up and frees it again when the key is stored
  bool put(K key, P value, Param* = nullptr) override { return table_.try_emplace(key, value).second; }

  bool update(K key, P value, Param* = nullptr) override {
    auto it = table_.find(key);
    if (it == table_.end()) return false;
    assign(it, value, 0);
    return true;
  }

  bool remove(K key, Param* = nullptr) override { return table_.erase(key) > 0; }

  size_t scan(K, size_t, std::pair<K, P>*, Param* = nullptr) override { return 0; }

  long long memory_consumption() override { return (long long)(bytes_.live + sizeof(table_)); }

  const Table& table() const { return table_; }
  size_t heap_bytes() const { return bytes_.live; }
  size_t peak_heap_bytes() const { return bytes_.peak; }

 private:
  // tsl::robin_map's iterators hand out the payload through value(); the others through ->second
  template <class It>
  static auto assign(It it, const P& v, int) -> decltype(it.value() = v, void()) { it.value() = v; }
  template <class It>
  static void assign(It it, const P& v, long) { it->second = v; }

  std::string name_;
  float max_lf_;
  ByteCount bytes_;  // before table_: the table's allocator counts into it, to the table's last deallocation
  Table table_;
};

// "ht_<table>[<L>]" -> the index, nullptr for names that do not start with "ht_". Throws std::invalid_argument for an
// unknown table, an <L> outside the table's range, and a table whose header was not on the include path.
template <class K, class P>
indexInterface<K, P>* make_hashtable(const std::string& name) {
  if (name.compare(0, 3, "ht_") != 0) return nullptr;
  auto bad = [&](const std::string& why) -> indexInterface<K, P>* { throw std::invalid_argument("gret::sota::make_hashtable: '" + name + "': " + why); };
  size_t e = name.size();
  while (e > 3 && name[e - 1] >= '0' && name[e - 1] <= '9') --e;
  const std::string table = name.substr(3, e - 3), digits = name.substr(e);
  int pct = 0;  // 0: the table's default
  if (!digits.empty()) {
    if (digits[0] == '0' || digits.size() > 4) return bad("<L> is a percentage without leading zeros");
    pct = std::stoi(digits);
  }
  auto lf = [&](int lo, int hi) -> float {
    if (pct != 0 && (pct < lo || pct > hi)) bad("max load factor " + digits + "% is outside " + std::to_string(lo) + ".." + std::to_string(hi));
    return float(pct) / 100.0f;
  };
  if (table == "std") return new HashTableIndex<K, P, StdTable<K, P>>(name, lf(10, 1000));
  if (table == "ankerl") {
#if GRET_SOTA_HAVE_ANKERL
    return new HashTableIndex<K, P, AnkerlTable<K, P>>(name, lf(10, 100));
#else
    return bad("built without ankerl/unordered_dense.h (-I<unordered_dense>/include)");
#endif
  }
  if (table == "robin" || table == "robinpg") {
#if GRET_SOTA_HAVE_ROBIN
    if (table == "robin") return new HashTableIndex<K, P, RobinTable<K, P>>(name, lf(20, 95));
    return new HashTableIndex<K, P, RobinPgTable<K, P>>(name, lf(20, 95));
#else
    return bad("built without tsl/robin_map.h (-I<robin-map>/include)");
#endif
  }
  if (table == "phmap") {
#if GRET_SOTA_HAVE_PHMAP
    if (pct != 0) return bad("phmap's load factor is fixed at 7/8");
    return new HashTableIndex<K, P, PhmapTable<K, P>>(name);
#else
    return bad("built without parallel_hashmap/phmap.h (-I<parallel-hashmap>)");
#endif
  }
  return bad("unknown table (std, ankerl, phmap, robin, robinpg)");
}

}  // namespace sota
}  // namespace gret
