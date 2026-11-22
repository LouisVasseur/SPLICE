// SPDX-License-Identifier: MIT
// The ht_<table>[<L>] indexes (gret/sota/hashtable.h) against a sorted array and a std::map.
//   static: after bulk_load, get on every stored key, on both neighbours of every stored key, on 0, 1, 2^63, 2^64 - 2, 2^64 - 1, just
//           below the smallest and above the largest key, and on random keys; scan returns 0 and writes nothing;
//           memory_consumption is the allocator's count plus the table object, and the count equals what each library's own layout
//           gives (ankerl: index_bytes() and the value vector; robin: 24-byte buckets; phmap: control bytes and slots; std: nodes
//           and bucket pointers).
//   writes: rounds of put, update and remove on stored keys, neighbours, random keys and keys already removed, with get checked on
//           every key the set ever held after each round; then every key removed in one of three orders, then all put back.
//   growth: an empty bulk_load and a small one, then puts far past what was reserved (the tables rehash); bulk_load twice.
//   names:  what make_hashtable must refuse.
//   adversarial: every key in one bucket (robinpg grows during bulk_load, std chains), and tables exactly full.
// Key sets as in tests/test_scan.cpp (tiny sets, uniform, wide, a dense run, clusters, far outliers, 0 and 2^64 - 1, a regular
// stride, a huge gap) plus 2M random keys.
//   test_hashtable [KEYS...]   GRE key files (uint64 count, then the keys), optional: the static check on each (a 2M-key slice of
//                              its middle if it is larger), and, when built with NDEBUG, a small timing on them as well.
// When built with NDEBUG (cxx-O3) it ends with a sanity timing on 2M random keys: each index through GRE's interface, and the
// same tables called directly with std::allocator, which is what the wrapper and its counting allocator must not be slower than.
// -DSOTA_WITH_HF4 adds hf4 (gret/hashline.h) to that timing.
#include "indexInterface.h"
#include "gret/sota/hashtable.h"
#ifdef SOTA_WITH_HF4
#include "gret/hashline.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

using Index = indexInterface<uint64_t, uint64_t>;
using KV = std::pair<uint64_t, uint64_t>;
namespace sota = gret::sota;
static const uint64_t M = ~uint64_t(0);

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

static std::vector<KV> pairs(const std::vector<uint64_t>& keys) {
  std::vector<KV> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  return kv;
}

// What the table's own layout says it holds on the heap, or -1 where this test knows no formula.
static long long layout_bytes(const sota::StdTable<uint64_t, uint64_t>& t, size_t heap) {
  // one node per key (next pointer and the pair: 24 bytes in libstdc++, which does not cache the hash of an integer; 32 in
  // libc++, which does) and one pointer per bucket
  const size_t buckets = t.bucket_count() * sizeof(void*);
  if (t.empty()) return (long long)(heap == 0 ? 0 : buckets);
  for (size_t node : {size_t(24), size_t(32)})
    if (heap == t.size() * node + buckets) return (long long)heap;
  return (long long)(t.size() * 24 + buckets);
}
static long long layout_bytes(const sota::AnkerlTable<uint64_t, uint64_t>& t, size_t) {
  return (long long)(t.index_bytes() + t.values().capacity() * sizeof(KV));
}
template <class T>
static long long robin_bytes(const T& t) {
  // the pair, a 16-bit distance, a flag and (in the padding, free) 32 bits of the hash
  static_assert(sizeof(tsl::detail_robin_hash::bucket_entry<KV, true>) == 24 && sizeof(tsl::detail_robin_hash::bucket_entry<KV, false>) == 24, "robin bucket");
  return (long long)(t.bucket_count() * 24);
}
static long long layout_bytes(const sota::RobinTable<uint64_t, uint64_t>& t, size_t) { return robin_bytes(t); }
static long long layout_bytes(const sota::RobinPgTable<uint64_t, uint64_t>& t, size_t) { return robin_bytes(t); }
static long long layout_bytes(const sota::PhmapTable<uint64_t, uint64_t>& t, size_t) {
  // capacity + 1 + (Group::kWidth - 1) cloned control bytes... the library lays out capacity + Group::kWidth + 1 of them,
  // padded to the slots' alignment, then capacity slots of 16 bytes
  const size_t cap = t.capacity();
  if (cap == 0) return 0;
  const size_t ctrl = cap + size_t(phmap::priv::Group::kWidth) + 1;
  return (long long)(((ctrl + 7) & ~size_t(7)) + cap * sizeof(KV));
}

// memory_consumption is the allocator's count plus the table object, and the count is what the table's layout gives
static void check_memory(Index* idx, const std::string& what, size_t& bad) {
  auto one = [&](auto tag) {
    using T = typename decltype(tag)::type;
    auto* h = dynamic_cast<sota::HashTableIndex<uint64_t, uint64_t, T>*>(idx);
    if (!h) return false;
    const long long heap = (long long)h->heap_bytes(), want = layout_bytes(h->table(), h->heap_bytes()), mem = idx->memory_consumption();
    if (heap != want || mem != heap + (long long)sizeof(T) || h->peak_heap_bytes() < h->heap_bytes() || heap < (long long)(h->table().size() * sizeof(KV)))
      if (bad++ < 5)
        std::printf("FAIL %s: memory: counted %lld heap bytes, the layout gives %lld; memory_consumption %lld, table object %zu, %zu keys\n",
                    what.c_str(), heap, want, mem, sizeof(T), size_t(h->table().size()));
    return true;
  };
  if (!(one(std::common_type<sota::StdTable<uint64_t, uint64_t>>{}) || one(std::common_type<sota::AnkerlTable<uint64_t, uint64_t>>{}) ||
        one(std::common_type<sota::RobinTable<uint64_t, uint64_t>>{}) || one(std::common_type<sota::RobinPgTable<uint64_t, uint64_t>>{}) ||
        one(std::common_type<sota::PhmapTable<uint64_t, uint64_t>>{})))
    ++bad, std::printf("FAIL %s: not a HashTableIndex this test knows\n", what.c_str());
}

// GRE's indexInterface has no virtual destructor: an index is destroyed through its own type
struct Destroy {
  void operator()(Index* idx) const {
    auto one = [&](auto tag) {
      auto* h = dynamic_cast<sota::HashTableIndex<uint64_t, uint64_t, typename decltype(tag)::type>*>(idx);
      delete h;
      return h != nullptr;
    };
    if (!idx) return;
    if (!(one(std::common_type<sota::StdTable<uint64_t, uint64_t>>{}) || one(std::common_type<sota::AnkerlTable<uint64_t, uint64_t>>{}) ||
          one(std::common_type<sota::RobinTable<uint64_t, uint64_t>>{}) || one(std::common_type<sota::RobinPgTable<uint64_t, uint64_t>>{}) ||
          one(std::common_type<sota::PhmapTable<uint64_t, uint64_t>>{})))
      std::printf("FAIL: an index of an unknown type was not destroyed\n"), std::abort();
  }
};
using Owned = std::unique_ptr<Index, Destroy>;

static bool stored(const std::vector<uint64_t>& keys, uint64_t k) { return std::binary_search(keys.begin(), keys.end(), k); }

// keys: sorted, unique
static size_t check_static(const std::vector<uint64_t>& keys, const std::string& label, const char* name) {
  std::vector<KV> kv = pairs(keys);
  const std::string what = std::string(name) + " on " + label;
  Owned idx;
  Param p(1, 0);
  try {
    idx.reset(sota::make_hashtable<uint64_t, uint64_t>(name));
    if (!idx) { std::printf("FAIL %s: not a hash table name\n", name); return 1; }
    idx->init(&p);
    idx->bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::exception& e) {
    std::printf("FAIL %s: threw %s\n", what.c_str(), e.what());
    return 1;
  }
  size_t bad = 0;
  auto fail = [&](const char* op, uint64_t k) { if (bad++ < 5) std::printf("FAIL %s: %s wrong at key %llu\n", what.c_str(), op, (unsigned long long)k); };
  const uint64_t canary = 0xDEADBEEFCAFEF00Dull;
  for (uint64_t k : keys) {  // every stored key
    uint64_t v = canary;
    if (!idx->get(k, v, &p) || v != payload(k)) fail("get of a stored key", k);
  }
  auto miss = [&](uint64_t k) {  // a key that may or may not be stored
    uint64_t v = canary;
    const bool got = idx->get(k, v, &p), is = stored(keys, k);
    if (got != is || (got && v != payload(k)) || (!got && v != canary)) fail("get", k);
  };
  for (uint64_t k : keys) miss(k + 1), miss(k - 1);  // wraps at the ends: 0 - 1 is 2^64 - 1, also a probe
  for (uint64_t k : {uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(1) << 63, (uint64_t(1) << 63) - 1, M - 2, M - 1, M}) miss(k);
  if (!keys.empty()) {
    for (uint64_t d : {uint64_t(1), uint64_t(2), uint64_t(1000), uint64_t(1) << 32}) {
      if (keys.front() >= d) miss(keys.front() - d);
      if (M - keys.back() >= d) miss(keys.back() + d);
    }
    miss(keys.front() / 2), miss(keys.back() + (M - keys.back()) / 2);
  }
  std::mt19937_64 g(23);
  for (size_t i = 0; i < std::max<size_t>(20000, keys.size() / 4); ++i) {
    miss(g());
    if (!keys.empty()) {  // inside the range, between two stored keys
      const size_t r = g() % keys.size();
      if (r + 1 < keys.size() && keys[r + 1] - keys[r] > 2) miss(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));
    }
  }
  std::vector<KV> out(64, KV(canary, canary));
  for (uint64_t s : {uint64_t(0), keys.empty() ? uint64_t(5) : keys[keys.size() / 2], M})
    for (size_t len : {size_t(1), size_t(64)}) {
      if (idx->scan(s, len, out.data(), &p) != 0) fail("scan (must return 0)", s);
      for (const KV& o : out)
        if (o.first != canary || o.second != canary) { fail("scan (wrote to the result)", s); break; }
    }
  check_memory(idx.get(), what, bad);
  return bad;
}

static size_t verify(Index& idx, const std::map<uint64_t, uint64_t>& ref, const std::vector<uint64_t>& all, std::mt19937_64& g, const std::string& what) {
  size_t bad = 0;
  auto probe = [&](uint64_t k) {
    uint64_t v = 0;
    const bool got = idx.get(k, v);
    const auto it = ref.find(k);
    if (got != (it != ref.end()) || (got && v != it->second))
      if (bad++ < 5) std::printf("FAIL %s: get wrong at key %llu\n", what.c_str(), (unsigned long long)k);
  };
  for (uint64_t k : all) probe(k), probe(k + 1), probe(k - 1);
  probe(0), probe(M);
  for (int i = 0; i < 1000; ++i) probe(g());
  KV out[4];
  if (idx.scan(0, 4, out) != 0) ++bad, std::printf("FAIL %s: scan returned keys\n", what.c_str());
  check_memory(&idx, what, bad);
  return bad;
}

// order: how the keys left after the rounds are removed (0 random, 1 ascending, 2 descending)
static size_t check_writes(const std::vector<uint64_t>& keys, const std::string& label, const char* name, int order, std::mt19937_64& g) {
  const std::string what = std::string(name) + " on " + label + " order " + std::to_string(order);
  std::vector<KV> kv = pairs(keys);
  Owned idx(sota::make_hashtable<uint64_t, uint64_t>(name));
  idx->init();
  idx->bulk_load(kv.data(), kv.size());
  std::map<uint64_t, uint64_t> ref(kv.begin(), kv.end());
  std::vector<uint64_t> all = keys;  // every key the index ever held
  size_t bad = verify(*idx, ref, all, g, what + " after load");
  auto complain = [&](const char* op, uint64_t k, bool ok, bool there) {
    if (bad++ < 5) std::printf("FAIL %s: %s of key %llu returned %d, stored %d\n", what.c_str(), op, (unsigned long long)k, int(ok), int(there));
  };
  auto write = [&](uint64_t k, int op) {  // 0 update, 1 remove, 2 put
    const auto it = ref.find(k);
    const bool there = it != ref.end();
    if (op == 0) {
      const uint64_t v = k ^ g();
      const bool ok = idx->update(k, v);
      if (ok != there) return complain("update", k, ok, there);
      if (there) it->second = v;
    } else if (op == 1) {
      const bool ok = idx->remove(k);
      if (ok != there) return complain("remove", k, ok, there);
      if (there) ref.erase(it);
    } else {
      const uint64_t v = k + g();
      const bool ok = idx->put(k, v);  // true only for a key that was absent; a stored key keeps its payload
      if (ok == there) return complain("put", k, ok, there);
      if (!there) ref.emplace(k, v), all.push_back(k);
    }
  };
  const size_t n0 = keys.size();
  for (int round = 0; round < 3; ++round) {
    const size_t ops = n0 / 2 + 16;
    for (size_t i = 0; i < ops; ++i) {
      const uint64_t k = all.empty() ? g() : all[g() % all.size()];
      const unsigned r = unsigned(g() % 12);
      if (r < 3) write(k, 0);
      else if (r < 6) write(k, 1);
      else if (r < 8) write(k, 2);                // stored: refused; removed earlier: back in
      else if (r == 8) write(k + 1, int(g() % 3));  // a neighbour, mostly absent
      else if (r == 9) write(g(), int(g() % 3));    // a random key, absent
      else if (r == 10) write(g(), 2);              // a new key
      else write(k, 1), write(k, 1), write(k, 0);   // gone, then nothing to remove or update
    }
    bad += verify(*idx, ref, all, g, what + " round " + std::to_string(round));
  }
  std::vector<uint64_t> rest;
  for (const auto& kvp : ref) rest.push_back(kvp.first);
  if (order == 0) std::shuffle(rest.begin(), rest.end(), g);
  if (order == 2) std::reverse(rest.begin(), rest.end());
  for (size_t i = 0; i < rest.size(); ++i) {
    write(rest[i], 1);
    if (i + 1 == rest.size() / 2) bad += verify(*idx, ref, all, g, what + " emptying");
  }
  bad += verify(*idx, ref, all, g, what + " empty");
  for (size_t i = 0; i < std::min<size_t>(rest.size(), 50); ++i) write(rest[i], int(g() & 1));  // nothing is left to write on
  bad += verify(*idx, ref, all, g, what + " empty, after writes");
  for (uint64_t k : rest) write(k, 2);  // and all back in
  bad += verify(*idx, ref, all, g, what + " refilled");
  return bad;
}

// far past what bulk_load reserved, and a second bulk_load over a used table
static size_t check_growth(const char* name, std::mt19937_64& g) {
  size_t bad = 0;
  for (size_t n0 : {size_t(0), size_t(1), size_t(100)}) {
    const std::string what = std::string(name) + " growing from " + std::to_string(n0);
    Owned idx(sota::make_hashtable<uint64_t, uint64_t>(name));
    idx->init();
    std::vector<uint64_t> keys;
    for (size_t i = 0; i < n0; ++i) keys.push_back(g() | 1);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<KV> kv = pairs(keys);
    idx->bulk_load(kv.data(), kv.size());
    std::map<uint64_t, uint64_t> ref(kv.begin(), kv.end());
    std::vector<uint64_t> all = keys;
    bad += verify(*idx, ref, all, g, what + " after load");
    for (size_t i = 0; i < 40000; ++i) {
      const uint64_t k = i % 3 == 0 ? i * 4096 : g();  // some regular, some random
      const bool ok = idx->put(k, payload(k)), fresh = ref.emplace(k, payload(k)).second;
      if (ok != fresh && bad++ < 5) std::printf("FAIL %s: put of key %llu returned %d\n", what.c_str(), (unsigned long long)k, int(ok));
      if (fresh) all.push_back(k);
    }
    bad += verify(*idx, ref, all, g, what + " after puts");
    // a second bulk_load replaces everything
    std::vector<uint64_t> again;
    for (size_t i = 0; i < 3000; ++i) again.push_back(g());
    std::sort(again.begin(), again.end());
    again.erase(std::unique(again.begin(), again.end()), again.end());
    kv = pairs(again);
    idx->bulk_load(kv.data(), kv.size());
    ref = std::map<uint64_t, uint64_t>(kv.begin(), kv.end());
    all.insert(all.end(), again.begin(), again.end());
    bad += verify(*idx, ref, all, g, what + " after a second bulk_load");
  }
  return bad;
}

// Keys built against a table's own layout (added by the verification pass): every key in one bucket of robinpg's prime, so
// the probe sequence passes tsl's limit of 8192 and the table has to grow in the middle of bulk_load, which is what an
// identity hash can do to it on real keys; every key in one bucket of std's; and 2^k keys at max load 1.0, a table with no
// empty slot left, where a miss must still end.
static size_t check_adversarial(std::mt19937_64& g) {
  size_t bad = 0;
  {
    const size_t n = 20000;
    tsl::robin_pg_map<uint64_t, uint64_t> t;
    t.reserve(n);
    const uint64_t p = t.bucket_count();  // what ht_robinpg reserves for n keys
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < n; ++i) keys.push_back(i * p + 7);
    for (const char* nm : {"ht_robinpg", "ht_robinpg95", "ht_robin", "ht_phmap", "ht_ankerl", "ht_std"}) bad += check_static(keys, "one bucket of robinpg's prime", nm);
    Owned idx(sota::make_hashtable<uint64_t, uint64_t>("ht_robinpg"));
    std::vector<KV> kv = pairs(keys);
    idx->bulk_load(kv.data(), kv.size());
    auto* h = dynamic_cast<sota::HashTableIndex<uint64_t, uint64_t, sota::RobinPgTable<uint64_t, uint64_t>>*>(idx.get());
    if (!h || h->table().bucket_count() <= p) ++bad, std::printf("FAIL adversarial: ht_robinpg kept %llu buckets for %zu keys of one bucket\n", (unsigned long long)p, n);
  }
  {
    std::unordered_map<uint64_t, uint64_t> u;
    u.reserve(3000);
    const uint64_t b = u.bucket_count();
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < 3000; ++i) keys.push_back(i * b);
    for (const char* nm : {"ht_std", "ht_std1000"}) bad += check_static(keys, "one bucket of std's", nm);
  }
  for (size_t n : {size_t(16), size_t(256), size_t(4096), size_t(65536)}) {
    std::vector<uint64_t> keys;
    while (keys.size() < n) {  // exactly n distinct keys
      while (keys.size() < n) keys.push_back(g());
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    }
    for (const char* nm : {"ht_ankerl100", "ht_robin95", "ht_robinpg95", "ht_phmap", "ht_std100"}) bad += check_static(keys, "exactly " + std::to_string(n) + " keys", nm);
  }
  return bad;
}

static size_t check_names() {
  size_t bad = 0;
  for (const char* nm : {"hl4", "hf4", "", "ht", "h", "alex", "xht_std"})
    try {
      if (Owned i{sota::make_hashtable<uint64_t, uint64_t>(nm)}) { ++bad; std::printf("FAIL names: '%s' built\n", nm); }
    } catch (const std::exception& e) { ++bad; std::printf("FAIL names: '%s' threw %s\n", nm, e.what()); }
  for (const char* nm : {"ht_", "ht_nope", "ht_80", "ht_phmap80", "ht_robin19", "ht_robin96", "ht_robinpg100", "ht_ankerl101", "ht_ankerl9",
                         "ht_std9", "ht_std1001", "ht_std080", "ht_ankerl0", "ht_robin00050", "ht_std_80", "ht_ankerl80x"})
    try {
      Owned i(sota::make_hashtable<uint64_t, uint64_t>(nm));
      ++bad, std::printf("FAIL names: '%s' %s, but should have thrown\n", nm, i ? "built" : "returned nullptr");
    } catch (const std::invalid_argument&) {
    } catch (const std::exception& e) { ++bad; std::printf("FAIL names: '%s' threw something else: %s\n", nm, e.what()); }
  for (const char* nm : {"ht_std", "ht_std10", "ht_std1000", "ht_ankerl10", "ht_ankerl100", "ht_robin20", "ht_robin95", "ht_robinpg20", "ht_robinpg95", "ht_phmap"})
    try {
      Owned i(sota::make_hashtable<uint64_t, uint64_t>(nm));
      if (!i) ++bad, std::printf("FAIL names: '%s' returned nullptr\n", nm);
    } catch (const std::exception& e) { ++bad; std::printf("FAIL names: '%s' threw %s\n", nm, e.what()); }
  return bad;
}

static bool read_keys(const char* path, std::vector<uint64_t>& k) {
  FILE* f = std::fopen(path, "rb");
  uint64_t n = 0;
  if (!f || std::fread(&n, 8, 1, f) != 1) { if (f) std::fclose(f); return false; }
  k.resize(n);
  const bool ok = std::fread(k.data(), 8, n, f) == n;
  std::fclose(f);
  if (!ok) return false;
  std::sort(k.begin(), k.end());
  k.erase(std::unique(k.begin(), k.end()), k.end());
  const size_t cap = 2000000;
  if (k.size() > cap) {  // a slice of the middle, neighbours kept as they are
    const size_t lo = (k.size() - cap) / 2;
    k = std::vector<uint64_t>(k.begin() + lo, k.begin() + lo + cap);
  }
  return true;
}

#ifdef NDEBUG
static volatile uint64_t g_sink;

template <class Get>
static double ns_per(const std::vector<uint64_t>& q, Get get) {
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t sum = 0;
    for (uint64_t k : q) sum += get(k);
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(q.size());
    g_sink = sum;
    best = std::min(best, ns);
  }
  return best;
}

static void timing(const std::vector<uint64_t>& keys, const std::string& label) {
  std::vector<KV> kv = pairs(keys);
  std::mt19937_64 g(5);
  std::vector<uint64_t> hits(keys), misses;
  std::shuffle(hits.begin(), hits.end(), g);
  // misses inside the key range: between two stored keys where there is room, else a random key
  for (size_t i = 0; i < keys.size(); ++i) {
    const size_t r = g() % keys.size();
    misses.push_back(r + 1 < keys.size() && keys[r + 1] - keys[r] > 1 ? keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1) : g());
  }
  std::printf("timing on %s, %zu keys, lookups in random order, best of 3 (ns per lookup; a sanity check, not a benchmark)\n", label.c_str(), keys.size());
  std::printf("  %-14s %10s %10s %10s %12s\n", "index", "hit", "miss", "build ms", "bytes/key");
  auto row = [&](const char* name, Index* raw) {
    Owned idx(raw);
    idx->init();
    const auto t0 = std::chrono::steady_clock::now();
    idx->bulk_load(kv.data(), kv.size());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    auto get = [&](uint64_t k) { uint64_t v = 0; idx->get(k, v); return v; };
    const double h = ns_per(hits, get), m = ns_per(misses, get);
    std::printf("  %-14s %10.1f %10.1f %10.0f %12.2f\n", name, h, m, ms, double(idx->memory_consumption()) / double(keys.size()));
  };
  for (const char* nm : {"ht_std", "ht_ankerl", "ht_ankerl95", "ht_phmap", "ht_robin", "ht_robin90", "ht_robinpg", "ht_robinpg90"})
    row(nm, sota::make_hashtable<uint64_t, uint64_t>(nm));
#ifdef SOTA_WITH_HF4
  {
    gret::HashLineIndex<uint64_t, uint64_t, 4, gret::mul64, true> hf("hf4");
    const auto t0 = std::chrono::steady_clock::now();
    hf.bulk_load(kv.data(), kv.size());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    Index* idx = &hf;  // through the interface, as the others are
    auto get = [&](uint64_t k) { uint64_t v = 0; idx->get(k, v); return v; };
    const double h = ns_per(hits, get), m = ns_per(misses, get);
    std::printf("  %-14s %10.1f %10.1f %10.0f %12.2f\n", "hf4", h, m, ms, double(hf.memory_consumption()) / double(keys.size()));
  }
#endif
  // the same tables with nothing around them: the library's own defaults and std::allocator, found through find()
  auto direct = [&](const char* name, auto& t) {
    t.reserve(kv.size());
    for (const KV& e : kv) t.emplace(e.first, e.second);
    auto get = [&](uint64_t k) { const auto it = t.find(k); return it == t.end() ? uint64_t(0) : uint64_t(it->second); };
    const double h = ns_per(hits, get), m = ns_per(misses, get);
    std::printf("  %-14s %10.1f %10.1f %10s %12s\n", name, h, m, "-", "-");
  };
  { std::unordered_map<uint64_t, uint64_t> t; direct("std direct", t); }
  { ankerl::unordered_dense::map<uint64_t, uint64_t> t; direct("ankerl direct", t); }
  { phmap::flat_hash_map<uint64_t, uint64_t> t; direct("phmap direct", t); }
  { tsl::robin_map<uint64_t, uint64_t, sota::MumHash> t; direct("robin direct", t); }
  { tsl::robin_pg_map<uint64_t, uint64_t> t; direct("robinpg direct", t); }
}
#endif

int main(int argc, char** argv) {
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  // sz: the size of the large sets; the write test uses smaller ones (every write is followed against a std::map)
  auto make_sets = [&](size_t sz) {
    std::vector<std::pair<std::string, std::vector<uint64_t>>> sets = {
        {"n0", {}}, {"n1", {5}}, {"n1max", {M}}, {"n1zero", {0}}, {"n2", {0, M}}, {"n3", {1, 2, M}}, {"n5", {0, 1, 2, 3, M}}, {"n17", rnd(0, uint64_t(1) << 40, 17)},
        {"uniform", rnd(1, uint64_t(1) << 50, sz)}, {"wide", rnd(0, M, sz)}};
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < sz * 5 / 2; ++i) v.push_back(1000 + i); sets.push_back({"run", v}); }
    { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), sz); for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003); v.push_back(3); v.push_back(M); sets.push_back({"outliers", v}); }
    { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + 100000, sz * 3 / 2); for (uint64_t j = 0; j < 100; ++j) v.push_back(M - j * 1000); sets.push_back({"over64outliers", v}); }
    { std::vector<uint64_t> v; for (size_t c = 0; c < sz / 100 + 1; ++c) { const uint64_t b = g() >> 1; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } sets.push_back({"clusters", v}); }
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < sz; ++i) v.push_back(i * 65536 + 12345); sets.push_back({"stride", v}); }
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < sz; ++i) v.push_back(i << 32); sets.push_back({"highbits", v}); }  // nothing in the low 32 bits
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < sz / 2; ++i) { v.push_back(i); v.push_back(M - i); } sets.push_back({"hugegap", v}); }
    for (auto& s : sets) {
      std::sort(s.second.begin(), s.second.end());
      s.second.erase(std::unique(s.second.begin(), s.second.end()), s.second.end());
    }
    return sets;
  };
  // every table at its default, and at the ends and the middle of what <L> accepts
  const char* names[] = {"ht_std", "ht_ankerl", "ht_phmap", "ht_robin", "ht_robinpg",
                         "ht_std50", "ht_std400", "ht_ankerl50", "ht_ankerl95", "ht_ankerl100", "ht_robin25", "ht_robin90", "ht_robin95", "ht_robinpg90"};
  const size_t n_default = 5;
  size_t bad = 0, cases = 0;
  auto ran = [&](size_t e) { ++cases, bad += e != 0; };

  ran(check_names());
  {
    const auto sets = make_sets(20000);
    for (const auto& [label, keys] : sets)
      for (const char* nm : names) ran(check_static(keys, label, nm));
  }
  {
    std::vector<uint64_t> big = rnd(0, M, 2000000);
    std::sort(big.begin(), big.end());
    big.erase(std::unique(big.begin(), big.end()), big.end());
    for (const char* nm : names) ran(check_static(big, "random2M", nm));
  }
  {
    const auto sets = make_sets(2500);
    std::mt19937_64 gw(11);
    for (const auto& [label, keys] : sets)
      for (size_t i = 0; i < sizeof names / sizeof *names; ++i)
        for (int order = 0; order < (i < n_default ? 3 : 1); ++order) ran(check_writes(keys, label, names[i], order, gw));
    for (const char* nm : names) ran(check_growth(nm, gw));
    ran(check_adversarial(gw));
  }
  for (int a = 1; a < argc; ++a) {
    std::vector<uint64_t> k;
    if (!read_keys(argv[a], k)) { std::printf("cannot read %s\n", argv[a]); return 2; }
    for (size_t i = 0; i < n_default; ++i) ran(check_static(k, argv[a], names[i]));
  }
  std::printf("test_hashtable: %s (%zu cases, %zu failed)\n", bad ? "FAILED" : "ok", cases, bad);
#ifdef NDEBUG
  if (!bad) {
    std::vector<uint64_t> big = rnd(0, M, 2000000);
    std::sort(big.begin(), big.end());
    big.erase(std::unique(big.begin(), big.end()), big.end());
    timing(big, "random2M");
    for (int a = 1; a < argc; ++a) {
      std::vector<uint64_t> k;
      if (read_keys(argv[a], k)) timing(k, argv[a]);
    }
  }
#endif
  return bad ? 1 : 0;
}
