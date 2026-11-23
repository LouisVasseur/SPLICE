// SPDX-License-Identifier: MIT
// dytis (include/gret/sota/dytis.h: the DyTIS authors' code behind GRE's interface) against a sorted array and a std::map: get on
// every stored key and on keys never stored (neighbours, 0, 2^63, 2^64 - 2, 2^64 - 1, random), scan from stored, neighbouring,
// in-between and extreme start keys, then put (neighbours of stored keys, short runs, keys far outside, duplicates, the
// payloads 0 and 2^64 - 1, which are the authors' "not found" and "empty slot"), update, remove (from full buckets too, and with
// key 0 stored: the two cases the authors' Delete gets wrong), the same checks after each, puts after the removes, and finally
// every key removed and the index filled again. Key sets: test_scan's patterns (tiny sets, uniform, the whole key space with
// its ends, a dense run, clusters, far outliers, a stride, two runs at the ends, large keys, fb-like, lognormal) at 100,000 to
// 350,000 keys in the modes dytis, dytiss, dytisu, and dytisa on sets where ascending inserts are affordable, a 2M-key
// random set and a 2M-key dense one, 96,000 edge keys, a set run with DYTIS_MAX_DEPTH=12 so that the depth limit sends keys to
// the side store, the rule that only one index may be alive, and (last, because no DyTIS can follow it in the process) out of
// memory inside the authors' code. memory_consumption is held against a count of every byte the index makes the program
// allocate.
//
// Build (docs/sota/dytis.md; C = the directory of the clones, readonly_competitors; B = build/tests/sota/dytis):
//   C/DyTIS       the clone of https://github.com/unist-ssl/DyTIS
//   C/DyTIS_shim  boost/pool/pool_alloc.hpp, a file without code (Boost is not installed: see the header)
//   printf '[pointer-overflow]\nsrc:*/DyTIS/src/*\n' > $B/ubsan_dytis.ignore
//   build/tests/cxx    -fsanitize-ignorelist=$B/ubsan_dytis.ignore -I $C/DyTIS_shim -I $C/DyTIS tests/sota/test_dytis.cpp -o $B/t
//   build/tests/cxx-O3 -I $C/DyTIS_shim -I $C/DyTIS tests/sota/test_dytis.cpp -o $B/t-O3
// The ignore list takes one UBSan check off the authors' sources only (ASan and the other checks stay on them, and the wrapper
// stays fully checked): they keep a segment's local depth in the top bits of its pointer by pointer arithmetic that overflows
// (DyTIS_impl.h:53, ExtendibleHash_impl.h:103-128). Without the list the test stops there on the first insert.
// Apple clang cannot compile the authors' default layout (SEP: arithmetic on void*), so the two lines above test their other
// layout (DTS_noSEP); g++ builds SEP:  g++ -std=gnu++17 -O3 -I $C/DyTIS_shim -I $C/DyTIS ... (and wf_cxx on the cluster).
//
// -DTEST_DYTIS_ALONE leaves GRE's own indexes out (they are here only to show the header lives next to them, and for --time).
//
//   t [SET...] [--time]     only the named sets; --time: build, lookup and insert timing on one key set (use t-O3), against
//                           the authors' Insert and Find called directly; the set is chosen by environment variables (see
//                           the timing block in main)
#include "indexInterface.h"
#ifndef TEST_DYTIS_ALONE  // GRE's own indexes first, as GRE's competitor.h has them: the new header must live next to them
#include "alex/alex.h"
#include "btree/btree.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#endif
#include "gret/sota/dytis.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// GRE's get_index names its template parameters like this; the header must not leak macros of these names
template <class KEY_TYPE, class PAYLOAD_TYPE>
static size_t names_are_free() { return sizeof(KEY_TYPE) + sizeof(PAYLOAD_TYPE); }
#if defined(SEP) || defined(REMAP_THRE) || defined(ADDR_BITS) || defined(DO_NOTHING) || defined(BOOST_POOL_ALLOC_HPP)
#error "gret/sota/dytis.h leaks a macro of the authors' headers"
#endif

// What the index makes the process allocate, to hold memory_consumption() against: every operator new of the program is
// counted here, by the size asked for (kept in front of the block; the allocator's own size for a block is no use, macOS
// hands out cached blocks far larger than asked), and the header counts what it passes to malloc for the authors' pools and
// large segments. Not under ASan, which has its own operator new.
#if !GRET_DYTIS_ASAN
#define TEST_DYTIS_COUNTS_HEAP 1
static size_t g_new_live = 0;
static void* counted_new(size_t n, size_t align) {
  const size_t front = align > 16 ? align : 16;
  void* raw = nullptr;
  if (align <= 16) raw = std::malloc(n + front);
  else if (posix_memalign(&raw, align, n + front) != 0) raw = nullptr;
  if (!raw) throw std::bad_alloc();
  std::memcpy(raw, &n, sizeof(n));
  g_new_live += n;
  return static_cast<char*>(raw) + front;
}
static void counted_delete(void* p, size_t align) noexcept {
  if (!p) return;
  void* raw = static_cast<char*>(p) - (align > 16 ? align : 16);
  size_t n;
  std::memcpy(&n, raw, sizeof(n));
  g_new_live -= n;
  std::free(raw);
}
void* operator new(size_t n) { return counted_new(n, 1); }
void* operator new[](size_t n) { return counted_new(n, 1); }
void* operator new(size_t n, std::align_val_t a) { return counted_new(n, size_t(a)); }
void* operator new[](size_t n, std::align_val_t a) { return counted_new(n, size_t(a)); }
void operator delete(void* p) noexcept { counted_delete(p, 1); }
void operator delete[](void* p) noexcept { counted_delete(p, 1); }
void operator delete(void* p, size_t) noexcept { counted_delete(p, 1); }
void operator delete[](void* p, size_t) noexcept { counted_delete(p, 1); }
void operator delete(void* p, std::align_val_t a) noexcept { counted_delete(p, size_t(a)); }
void operator delete[](void* p, std::align_val_t a) noexcept { counted_delete(p, size_t(a)); }
void operator delete(void* p, size_t, std::align_val_t a) noexcept { counted_delete(p, size_t(a)); }
void operator delete[](void* p, size_t, std::align_val_t a) noexcept { counted_delete(p, size_t(a)); }
static size_t heap_in_use() { return g_new_live + gret::dytis_authors::g_mem.pool_held + gret::dytis_authors::g_mem.big_live; }
#else
#define TEST_DYTIS_COUNTS_HEAP 0
static size_t heap_in_use() { return 0; }
#endif

using Index = indexInterface<uint64_t, uint64_t>;
using Dytis = gret::DytisIndex<uint64_t, uint64_t>;
using Map = std::map<uint64_t, uint64_t>;
static const uint64_t M = ~uint64_t(0);
static const uint64_t H = uint64_t(1) << 63;

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

static size_t g_bad = 0;
static void fail(const std::string& what, const char* op, uint64_t k) {
  if (g_bad++ < 40) std::printf("FAIL %s: %s wrong at key %llu\n", what.c_str(), op, (unsigned long long)k);
}

// every answer against the reference: get on every key it holds and on probes, scans from starts
static void verify(Index& idx, const Map& ref, const std::vector<uint64_t>& probes, std::mt19937_64& g, const std::string& what) {
  std::vector<std::pair<uint64_t, uint64_t>> all(ref.begin(), ref.end());
  uint64_t v = 0;
  for (const auto& kv : all)
    if (!idx.get(kv.first, v) || v != kv.second) fail(what, "get", kv.first);
  auto probe = [&](uint64_t k) {
    uint64_t val = 0;
    const bool got = idx.get(k, val);
    const auto it = ref.find(k);
    if (got != (it != ref.end()) || (got && val != it->second)) fail(what, "get (probe)", k);
  };
  for (uint64_t k : probes) probe(k), probe(k + 1), probe(k - 1);
  for (uint64_t k : {uint64_t(0), uint64_t(1), M, M - 1, M - 2, M - 3, H, H - 1, H + 1, H - 2}) probe(k);
  for (uint64_t x = 0; x < 512; ++x) probe(x << 55), probe((x << 55) - 1), probe((x << 55) + 1);  // the ends of the authors' 512 first-level tables
  for (int i = 0; i < 3000 && !all.empty(); ++i) {  // the top of the 2^24 block of a stored key: the wrapper's edge keys, and the last key that is not one
    const uint64_t top = all[g() % all.size()].first | 0xffffff;
    probe(top), probe(top - 1), probe(top - (Dytis::kEdge - 1)), probe(top - Dytis::kEdge), probe(top + 1);
  }
  for (int i = 0; i < 20000; ++i) probe(g());
  for (int i = 0; i < 20000 && !all.empty(); ++i) {  // in the gaps and next to stored keys
    const size_t r = g() % all.size();
    probe(all[r].first + 1), probe(all[r].first - 1);
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) probe(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  std::vector<uint64_t> starts = {0, 1, M, M - 1, M - 2, H, H - 1, H + 1};
  auto around = [&](uint64_t k) { starts.push_back(k), starts.push_back(k + 1), starts.push_back(k - 1); };
  for (size_t i = 0; i < std::min<size_t>(40, all.size()); ++i) around(all[i].first), around(all[all.size() - 1 - i].first);
  for (int i = 0; i < 1500 && !all.empty(); ++i) {
    const size_t r = g() % all.size();
    around(all[r].first);
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) starts.push_back(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  for (size_t i = 0; i < probes.size() && i < 600; ++i) around(probes[i]);
  for (int i = 0; i < 100; ++i) starts.push_back(g());
  for (uint64_t x = 0; x < 512; x += 37) around(x << 55);
  for (int i = 0; i < 60 && !all.empty(); ++i) {
    const uint64_t top = all[g() % all.size()].first | 0xffffff;
    around(top), around(top - Dytis::kEdge);
  }
  const size_t lens[] = {1, 2, 3, 16, 256, 5000, 70000};
  std::vector<std::pair<uint64_t, uint64_t>> out;
  auto less = [](const std::pair<uint64_t, uint64_t>& a, uint64_t b) { return a.first < b; };
  for (uint64_t s : starts)
    for (size_t len : lens) {
      if (len > 256 && (g() & 31)) continue;  // long scans on a sample of the starts
      out.assign(len + 1, {0, 0});
      const size_t got = idx.scan(s, len, out.data());
      const size_t at = size_t(std::lower_bound(all.begin(), all.end(), s, less) - all.begin());
      bool ok = got == std::min(len, all.size() - at) && out[len] == std::pair<uint64_t, uint64_t>(0, 0);
      for (size_t i = 0; ok && i < got; ++i) ok = out[i] == all[at + i];
      if (!ok) fail(what, "scan", s);
    }
  {  // one scan over everything
    out.assign(all.size() + 7, {0, 0});
    const size_t got = idx.scan(0, all.size() + 7, out.data());
    bool ok = got == all.size();
    for (size_t i = 0; ok && i < got; ++i) ok = out[i] == all[i];
    if (!ok) fail(what, "full scan", 0);
  }
}

struct Expect {
  bool depth_limited = false;  // the set is run with DYTIS_MAX_DEPTH low: keys must reach the side store
  bool runs = true;            // put short runs of consecutive keys (dense keys cost the authors' code memory: docs/sota/dytis.md)
  bool empty_it = true;        // remove every key at the end
};

static void check(const std::string& label, std::vector<uint64_t> keys, const char* name, const Expect& ex = Expect()) {
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  const bool empty_set = keys.empty();
  const std::string what = std::string(name) + " on " + label;
  const size_t bad0 = g_bad;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::pair<uint64_t, uint64_t>> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  if (empty_set) keys.push_back(uint64_t(1) << 40);  // nothing is loaded; the writes below still need a place to aim at
  // the authors' "not found" (0) and "empty slot" (2^64 - 1) as payloads
  for (size_t i = 0; i < kv.size(); i += std::max<size_t>(1, kv.size() / 7)) kv[i].second = (i & 1) ? M : 0;
  const size_t heap0 = heap_in_use();
  Index* idx = gret::make_dytis<uint64_t, uint64_t>(name);
  if (!idx) { std::printf("FAIL %s: no such index\n", name); ++g_bad; return; }
  std::unique_ptr<Dytis> owner(dynamic_cast<Dytis*>(idx));  // deleted as what it is: indexInterface has no virtual destructor
  Dytis* d = owner.get();
  Param p(1, 0);
  idx->init(&p);
  try {
    idx->bulk_load(kv.data(), kv.size(), &p);
    const size_t side0 = d->in_side(), in0 = d->in_dytis();
    size_t reserved = 0;  // the keys the wrapper keeps out of DyTIS by rule (2^64 - 1 and 2^64 - 2 among them)
    for (const auto& e : kv) reserved += Dytis::edge_key(e.first);
    if (in0 + side0 != kv.size()) fail(what, "bulk_load holds another number of keys than", kv.size());
    if (!ex.depth_limited && side0 != reserved) fail(what, "the side store does not hold exactly the edge keys after the bulk load; side keys:", side0);
    if (ex.depth_limited && (side0 <= reserved || d->depth_refusals() == 0)) fail(what, "the depth limit sent no key to the side store; side keys:", side0);
    const size_t heap1 = heap_in_use();
    const long long bytes = idx->memory_consumption();
    if (bytes < (long long)(16 * in0)) fail(what, "memory_consumption below 16 bytes a key:", uint64_t(bytes));
    // against the count of what the index made the process allocate. Not in memory_consumption: two scratch vectors the
    // authors' LocalRemap keeps (function statics, at most 2.6 MB together, kept from one index to the next) and what the
    // name string and the side store's nodes take beyond their nominal size
    const double heap = TEST_DYTIS_COUNTS_HEAP ? double(heap1 - heap0) : 0.0;
    if (TEST_DYTIS_COUNTS_HEAP && (double(bytes) > heap * 1.001 + 256 || double(bytes) < heap * 0.999 - 2.7e6)) fail(what, "memory_consumption is not what the index made the process allocate, which is", uint64_t(heap));
    const Dytis::Shape sh = d->shape();

    std::mt19937_64 g(std::hash<std::string>()(label) ^ 0x9e3779b97f4a7c15ull);
    Map ref(kv.begin(), kv.end());
    std::vector<uint64_t> probes;
    verify(*idx, ref, probes, g, what + " (loaded)");

    auto pick = [&]() { return keys[g() % keys.size()]; };
    // put: neighbours of stored keys, short runs behind a stored key, keys anywhere, keys far outside, special keys,
    // duplicates, and the payloads 0 and 2^64 - 1
    {
      std::vector<std::pair<uint64_t, uint64_t>> add;
      for (int i = 0; i < 20000; ++i) { const uint64_t k = pick(); add.push_back({k + 1, payload(k + 1)}); if (i % 3 == 0) add.push_back({k + 2, payload(k + 2)}); if (i % 7 == 0) add.push_back({k - 1, payload(k - 1)}); }
      if (ex.runs) for (int r = 0; r < 4; ++r) { const uint64_t k = pick(); for (uint64_t j = 1; j <= 100; ++j) add.push_back({k + j * (r + 1), payload(k + j)}); }
      for (int i = 0; i < 20000; ++i) { const uint64_t span = keys.back() - keys.front(), k = keys.front() + (span == M ? g() : g() % (span + 1)); add.push_back({k, payload(k)}); }
      for (int i = 0; i < 3000; ++i) { const uint64_t k = g(); add.push_back({k, payload(k)}); }
      for (uint64_t j = 0; j < 40; ++j) {
        add.push_back({keys.back() + 1 + j, j}), add.push_back({keys.back() + (uint64_t(1) << (j + 4)), j}), add.push_back({keys.front() - 1 - j, j});
        add.push_back({keys.front() >> (j % 20 + 1), j}), add.push_back({keys.back() + (keys.back() - keys.front()) * (j + 1), j});
      }
      for (uint64_t k : {uint64_t(0), uint64_t(1), M, M - 1, M - 2, M - 3, H, H - 1, H + 1, H - 2}) add.push_back({k, payload(k)});
      for (int i = 0; i < 300; ++i) add.push_back({pick() + 3, i & 1 ? M : 0});  // the authors' special values
      for (int i = 0; i < 400; ++i) {  // edge keys (side store) and the keys just below them (DyTIS)
        const uint64_t top = pick() | 0xffffff;
        add.push_back({top - i % 3, payload(top)}), add.push_back({top - (Dytis::kEdge - 1), 1}), add.push_back({top - Dytis::kEdge, 2}), add.push_back({top - Dytis::kEdge - 1 - i % 5, 3});
      }
      for (int i = 0; i < 2000; ++i) add.push_back({pick(), 99});               // mostly there already
      std::shuffle(add.begin(), add.end(), g);
      for (const auto& a : add) {
        const bool fresh = ref.emplace(a.first, a.second).second;
        if (idx->put(a.first, a.second) != fresh) fail(what, "put", a.first);
        probes.push_back(a.first);
      }
      if (d->in_dytis() + d->in_side() != ref.size()) fail(what, "after put the index counts another number of keys than", ref.size());
      verify(*idx, ref, probes, g, what + " (put)");
    }
    // update: present keys (some to 0 and 2^64 - 1 and back), absent keys return false
    {
      for (int i = 0; i < 30000; ++i) {
        uint64_t k = i % 4 ? pick() : probes[g() % probes.size()];
        if (i % 9 == 0) k += 1;
        const uint64_t v = i % 50 == 0 ? M : i % 50 == 1 ? 0 : g();
        const auto it = ref.find(k);
        if (idx->update(k, v) != (it != ref.end())) fail(what, "update", k);
        if (it != ref.end()) it->second = v;
        if (i % 100 == 0 && it != ref.end()) {  // and back to an ordinary payload
          if (!idx->update(k, v ^ 5)) fail(what, "second update", k);
          it->second = v ^ 5;
        }
      }
      for (uint64_t k : {M, M - 1, M - 2, uint64_t(0)}) {
        const auto it = ref.find(k);
        if (idx->update(k, 4242) != (it != ref.end())) fail(what, "update (special key)", k);
        if (it != ref.end()) it->second = 4242;
      }
      verify(*idx, ref, probes, g, what + " (updated)");
    }
    // remove: random keys (loaded ones and put ones) and both ends; absent keys and a second remove return false
    std::vector<uint64_t> gone;
    {
      for (int i = 0; i < 30000; ++i) gone.push_back(i % 3 ? pick() : probes[g() % probes.size()]);
      for (size_t i = 0; i < 60 && i < keys.size(); ++i) gone.push_back(keys[i]), gone.push_back(keys[keys.size() - 1 - i]);
      for (uint64_t k : gone) {
        const bool there = ref.erase(k) != 0;
        if (idx->remove(k) != there) fail(what, "remove", k);
        if (idx->remove(k)) fail(what, "second remove", k);
        probes.push_back(k);
      }
      for (int i = 0; i < 2000; ++i) {
        const uint64_t k = pick() + 1;
        if (idx->remove(k) != (ref.erase(k) != 0)) fail(what, "remove (neighbour)", k);
      }
      if (d->in_dytis() + d->in_side() != ref.size()) fail(what, "after remove the index counts another number of keys than", ref.size());
      verify(*idx, ref, probes, g, what + " (removed)");
      for (size_t i = 0; i < gone.size(); i += 2) {
        const uint64_t k = gone[i] + (i % 4 == 0);
        if (idx->put(k, payload(k) ^ 1) != ref.emplace(k, payload(k) ^ 1).second) fail(what, "put (after remove)", k);
        probes.push_back(k);
      }
      verify(*idx, ref, probes, g, what + " (put after remove)");
    }
    const size_t side1 = d->in_side(), in1 = d->in_dytis(), slow1 = d->restructuring_inserts(), own1 = d->own_removes();
    if (ex.empty_it) {  // every key out, in a random order; then an index that holds nothing, and takes keys again
      std::vector<uint64_t> order;
      for (const auto& kvp : ref) order.push_back(kvp.first);
      std::shuffle(order.begin(), order.end(), g);
      for (uint64_t k : order)
        if (!idx->remove(k)) fail(what, "remove (emptying)", k);
      ref.clear();
      if (d->in_dytis() || d->in_side()) fail(what, "emptied, but the index still counts keys:", d->in_dytis() + d->in_side());
      probes.resize(std::min<size_t>(probes.size(), 20000));
      for (size_t i = 0; i < 20000 && !order.empty(); ++i) probes.push_back(order[i % order.size()]);
      verify(*idx, ref, probes, g, what + " (emptied)");
      for (size_t i = 0; i < 30000 && i < order.size(); ++i) {
        const uint64_t k = order[i] + (i % 2);
        if (idx->put(k, k ^ 7) != ref.emplace(k, k ^ 7).second) fail(what, "put (into the emptied index)", k);
      }
      verify(*idx, ref, probes, g, what + " (refilled)");
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%-7s %-13s n=%-8zu loaded: dytis=%zu side=%zu mode=%s tables=%zu segments=%zu depth=%u fill=%.2f %.1f B/key (%.1f%% of the bytes allocated); after writes: dytis=%zu side=%zu restructured=%zu own_removes=%zu depth_refusals=%zu edge_models=%zu(%zu)  %s  %.1fs\n",
                name, label.c_str(), kv.size(), in0, side0, d->mode_now(), sh.tables, sh.segments, sh.max_ld, sh.buckets ? double(in0) / double(sh.buckets * Dytis::kB) : 0.0,
                kv.empty() ? 0.0 : double(bytes) / double(kv.size()), heap > 0 ? 100.0 * double(bytes) / heap : 0.0, in1, side1, slow1, own1, d->depth_refusals(), d->edge_models(), d->edge_worst(), g_bad == bad0 ? "ok" : "FAILED", s);
  } catch (const std::exception& e) {
    std::printf("FAIL %s: threw %s (the index then held %zu + %zu keys)\n", what.c_str(), e.what(), d->in_dytis(), d->in_side());
    ++g_bad;
  }
  std::fflush(stdout);
}

// the authors' code with nothing around it: what a GRE wrapper of five lines would run
struct Thin {
  gret::dytis_authors::DyTIS d;
  void put(uint64_t k, uint64_t v) {
    if (!Dytis::edge_key(k)) d.Insert(k, v);  // an edge key can make the authors' code write outside a segment: left out here too
  }
  bool get(uint64_t k, uint64_t& v) {
    const uint64_t* p = d.Find(k);
    if (!p) return false;
    v = *p;
    return true;
  }
};

template <class Put, class Get>
static void time_one(const char* name, const std::vector<std::pair<uint64_t, uint64_t>>& kv, const std::vector<std::pair<uint64_t, uint64_t>>& extra, Put put, Get get, double build_s, double bytes_per_key) {
  std::mt19937_64 g(3);
  std::vector<uint64_t> q(2000000);
  for (uint64_t& x : q) x = kv[g() % kv.size()].first;
  uint64_t sum = 0, v = 0;
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t x : q) sum += get(x, v) ? v : 1;
    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(q.size()));
  }
  const auto t0 = std::chrono::steady_clock::now();
  size_t in = 0;
  for (const auto& e : extra) in += put(e.first, e.second);
  const double put_ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(extra.size());
  std::printf("time %-12s n=%zu build %.2fs (%.0f ns a key)  get %.1f ns  put %.1f ns (%zu new)  %.1f B/key  [checksum %llu]\n", name, kv.size(), build_s, build_s * 1e9 / double(kv.size()), best, put_ns, in,
              bytes_per_key, (unsigned long long)sum);
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  std::vector<std::string> only;
  bool timing = false;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--time")) timing = true;
    else only.push_back(argv[i]);
  auto want = [&](const std::string& s) { return only.empty() ? !timing : std::find(only.begin(), only.end(), s) != only.end(); };
  if (names_are_free<uint64_t, uint64_t>() != 16) return 2;
  if (!std::getenv("DYTIS_MAX_GB")) setenv("DYTIS_MAX_GB", "10", 1);  // a test must not take the machine down
  std::printf("test_dytis: layout %s\n", Dytis::layout());

  for (const char* bad : {"dyti", "dytisx", "dytis1", "dytis_", "dytissu", "dytisaa", "dytisas", "dytisr", "dytis "})
    if (gret::make_dytis<uint64_t, uint64_t>(bad)) {
      std::printf("FAIL make_dytis accepts the malformed name '%s'\n", bad);
      ++g_bad;
    }
  for (const char* good : {"dytis", "dytiss", "dytisu", "dytisa", "dytissa", "dytisua"}) {
    std::unique_ptr<Dytis> made(dynamic_cast<Dytis*>(gret::make_dytis<uint64_t, uint64_t>(good)));
    if (!made) {
      std::printf("FAIL make_dytis refuses the name '%s'\n", good);
      ++g_bad;
    }
  }
  if (gret::make_dytis<uint32_t, uint64_t>("dytis") || gret::make_dytis<int64_t, uint64_t>("dytis")) {
    std::printf("FAIL make_dytis accepts key types the authors' code does not have\n");
    ++g_bad;
  }
  {  // one index at a time: the second constructor throws, and the first is not disturbed
    std::unique_ptr<Dytis> a(new Dytis("dytis"));
    std::pair<uint64_t, uint64_t> two[2] = {{7, 70}, {9, 90}};
    a->bulk_load(two, 2);
    bool threw = false;
    try {
      std::unique_ptr<Dytis> b(new Dytis("dytisu"));
    } catch (const std::logic_error&) {
      threw = true;
    }
    uint64_t v = 0;
    if (!threw || !a->get(7, v) || v != 70 || !a->get(9, v) || v != 90 || a->get(8, v)) {
      std::printf("FAIL a second DytisIndex was allowed next to a live one, or disturbed it\n");
      ++g_bad;
    }
    bool unsorted = false;
    std::pair<uint64_t, uint64_t> bad[3] = {{7, 70}, {9, 90}, {9, 91}};
    try {
      a->bulk_load(bad, 3);
    } catch (const std::invalid_argument&) {
      unsorted = true;
    }
    if (!unsorted) {
      std::printf("FAIL bulk_load took duplicate keys\n");
      ++g_bad;
    }
    a->bulk_load(two, 2);  // and can be loaded again
    if (!a->get(7, v) || v != 70 || a->in_dytis() != 2) {
      std::printf("FAIL an index reloaded after a refused bulk_load does not answer\n");
      ++g_bad;
    }
  }

  const size_t N = 300000;
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; v.reserve(m); std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  Expect noruns;
  noruns.runs = false;

  if (want("tiny")) {
    check("n0", {}, "dytis");
    check("n1", {5}, "dytis");
    check("n1max", {M}, "dytis");
    check("n1zero", {0}, "dytis");
    check("n2", {0, M}, "dytis");
    check("n3", {1, 2, M}, "dytis");
    check("n5", {0, 1, 2, 3, M}, "dytis");
    check("reserved", {M - 1, M}, "dytis");
    check("n17", rnd(0, uint64_t(1) << 40, 17), "dytis");
    check("n17", rnd(0, uint64_t(1) << 40, 17), "dytisa");
    check("n128", rnd(0, M, 128), "dytiss");
    check("n129", rnd(0, M, 129), "dytisu");
    check("uniform20k", rnd(1, uint64_t(1) << 50, 20000), "dytis");
  }
  if (want("uniform")) check("uniform", rnd(1, uint64_t(1) << 50, N), "dytis");
  if (want("uniforms")) check("uniform", rnd(1, uint64_t(1) << 50, N), "dytiss");  // the knob: the authors' mode switch
  if (want("uniformu")) check("uniform", rnd(1, uint64_t(1) << 50, N), "dytisu");
  if (want("wide")) {  // the whole key space, with its ends: all 512 first-level tables
    std::vector<uint64_t> v = rnd(0, M, N);
    for (uint64_t k : {uint64_t(0), M, M - 1, M - 2, H, H - 1}) v.push_back(k);
    check("wide", v, "dytis");
    check("wide", v, "dytisa");  // ascending inserts: affordable on keys this sparse
    check("wide", v, "dytisua");
  }
  if (want("run")) {  // consecutive keys; in uniform mode a third of them, because the authors' code takes 1,800 bytes a key there
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < N; ++i) v.push_back(1000 + i);
    check("run", v, "dytis");
    v.resize(N / 3);
    check("run100k", v, "dytisu");
  }
  if (want("run0")) { std::vector<uint64_t> v; for (uint64_t i = 0; i < 150000; ++i) v.push_back(i); check("run0", v, "dytiss"); }  // key 0 heads a bucket
  if (want("clusters")) {
    std::vector<uint64_t> v;
    for (int c = 0; c < 3000; ++c) { const uint64_t b = g() >> 16; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); }
    check("clusters", v, "dytis", noruns);
    check("clusters", v, "dytisu", noruns);
  }
  if (want("outliers")) {  // a narrow bulk and keys far from it
    std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), N);
    for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003);
    v.push_back(3), v.push_back(M);
    check("outliers", v, "dytis", noruns);
  }
  if (want("stride")) { std::vector<uint64_t> v; for (uint64_t i = 0; i < N; ++i) v.push_back(i * 65536 + 12345); check("stride", v, "dytis"); }
  if (want("hugegap")) {  // a run from 0 and a run down from 2^64 - 1
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < N / 2; ++i) v.push_back(i), v.push_back(M - i);
    check("hugegap", v, "dytis");
  }
  if (want("highmag")) {  // large keys, with pairs, triples and a short run
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < N; ++i) v.push_back((uint64_t(1) << 61) + (i << 20) + (g() & 0x7ffff));
    for (int i = 0; i < 30000; ++i) { const uint64_t k = v[g() % N]; v.push_back(k + 1); if (i & 1) v.push_back(k + 2); if (i % 5 == 0) v.push_back(k + 700); }
    for (uint64_t i = 0; i < 100; ++i) v.push_back((uint64_t(1) << 62) + (uint64_t(1) << 50) + i);
    check("highmag", v, "dytis", noruns);
  }
  if (want("fblike")) {  // small keys, a few very large ones on both sides of 2^63
    std::vector<uint64_t> v = rnd(0, uint64_t(1) << 36, N);
    for (int j = 0; j < 21; ++j) v.push_back((uint64_t(1) << 59) + (g() >> 2));
    for (int j = 0; j < 10; ++j) v.push_back(H + (g() >> 1));
    check("fblike", v, "dytis");
    check("fblike", v, "dytiss");
  }
  if (want("lognormal")) {
    std::vector<uint64_t> v;
    std::lognormal_distribution<double> ln(0, 2);
    for (size_t i = 0; i < N; ++i) v.push_back(uint64_t(ln(g) * 1e12));
    check("lognormal", v, "dytis");
    check("lognormal", v, "dytisu");
  }
  if (want("edges")) {  // the last keys of 2^24 blocks, 70 of each: 64 are edge keys and stay out of DyTIS, 6 are not
    std::vector<uint64_t> v = rnd(0, uint64_t(1) << 44, 100000);
    for (uint64_t b = 1; b <= 1500; ++b)
      for (uint64_t j = 1; j <= 70; ++j) v.push_back((b << 24) * 977 - j);
    check("edges", v, "dytis", noruns);
    check("edges", v, "dytisu", noruns);
  }
  if (want("depthlimit")) {  // the depth limit, set low: spread keys fit, a dense run in them needs deeper segments and goes to the side store
    setenv("DYTIS_MAX_DEPTH", "12", 1);
    Expect e;
    e.depth_limited = true;
    e.runs = false;
    std::vector<uint64_t> v = rnd(1, uint64_t(1) << 50, 150000);
    for (uint64_t i = 0; i < 3000; ++i) v.push_back((uint64_t(1) << 49) + 12345 + i);
    check("depth12", v, "dytis", e);
    check("depth12", v, "dytisu", e);
    check("depth12", v, "dytissa", e);
    unsetenv("DYTIS_MAX_DEPTH");
  }
  if (want("big")) {  // about 2M keys: random over the key space, and dense
    Expect e;
    e.empty_it = false;
    check("random2M", rnd(0, M, 2000000), "dytis", e);
    check("dense2M", rnd(0, uint64_t(1) << 36, 2000000), "dytis", e);
  }
  if (timing) {  // not a benchmark: what the wrapper's guards cost against the authors' code called directly, and what a build costs
    // TEST_DYTIS_TIME_N keys (default 3M), TEST_DYTIS_TIME_DIST: uniform below 2^TEST_DYTIS_TIME_BITS (default: 64 bits),
    // lognormal, dense (consecutive keys in runs of 1 to 200,000 with gaps: ids), clusters (runs of 100 at random places);
    // TEST_DYTIS_TIME_NAMES: the index names to time, comma-separated; TEST_DYTIS_TIME_RAW=0 and TEST_DYTIS_TIME_GRE=0 leave out
    // the authors' code alone and GRE's indexes, TEST_DYTIS_TIME_RAW=2 gives the authors' code alone the keys in ascending order
    const char* bits = std::getenv("TEST_DYTIS_TIME_BITS");
    const int b = bits ? std::min(64, std::max(24, std::atoi(bits))) : 64;
    const uint64_t top = b == 64 ? M - 2 : uint64_t(1) << b;
    const char* ne = std::getenv("TEST_DYTIS_TIME_N");
    const size_t tn = ne ? size_t(std::atof(ne)) : 3000000;
    const std::string dist = std::getenv("TEST_DYTIS_TIME_DIST") ? std::getenv("TEST_DYTIS_TIME_DIST") : "uniform";
    std::vector<uint64_t> keys;
    if (dist == "uniform") {
      keys = rnd(1, top, tn);
    } else if (dist == "lognormal") {
      std::lognormal_distribution<double> ln(0, 2);
      for (size_t i = 0; i < tn; ++i) keys.push_back(uint64_t(ln(g) * 1e12));
    } else if (dist == "dense") {
      uint64_t k = 1000;
      while (keys.size() < tn) {
        const uint64_t run = 1 + g() % 200000;
        for (uint64_t j = 0; j < run && keys.size() < tn; ++j) keys.push_back(k++);
        k += 1 + g() % (g() % 16 ? 64 : 4000000);
      }
    } else if (dist == "clusters") {
      for (size_t c = 0; c < tn / 100; ++c) { const uint64_t base = g() >> (64 - b); for (uint64_t i = 0; i < 100; ++i) keys.push_back(base + i); }
    } else {
      std::printf("FAIL unknown TEST_DYTIS_TIME_DIST '%s'\n", dist.c_str());
      return 1;
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    // a tenth of the keys is held back and put afterwards, as GRE's insert workload does (keys of the data set, random order)
    std::vector<std::pair<uint64_t, uint64_t>> kv, extra;
    for (size_t i = 0; i < keys.size(); ++i) (g() % 10 ? kv : extra).push_back({keys[i], 123456789});
    std::shuffle(extra.begin(), extra.end(), g);
    std::printf("timing: %s, %zu keys loaded, %zu put afterwards, keys %llu to %llu\n", dist.c_str(), kv.size(), extra.size(), (unsigned long long)keys.front(), (unsigned long long)keys.back());
    Param p(1, 0);
    std::vector<std::string> names;
    {
      std::string list = std::getenv("TEST_DYTIS_TIME_NAMES") ? std::getenv("TEST_DYTIS_TIME_NAMES") : "dytis,dytisa,dytiss,dytisu";
      for (size_t at = 0; at <= list.size();) {
        const size_t c = std::min(list.find(',', at), list.size());
        if (c > at) names.push_back(list.substr(at, c - at));
        at = c + 1;
      }
    }
    for (const std::string& name : names) {
      std::unique_ptr<Dytis> d(dynamic_cast<Dytis*>(gret::make_dytis<uint64_t, uint64_t>(name)));
      if (!d) { std::printf("FAIL no index '%s'\n", name.c_str()); ++g_bad; continue; }
      std::vector<std::pair<uint64_t, uint64_t>> copy = kv;
      const auto t0 = std::chrono::steady_clock::now();
      try {
        d->bulk_load(copy.data(), copy.size(), &p);
      } catch (const std::exception& e) {
        std::printf("time %-12s bulk_load threw after %.1fs: %s\n", name.c_str(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), e.what());
        std::printf("timing stops here: after an exception out of the authors' code no further DyTIS can be made in this process\n");
        std::fflush(stdout);
        return g_bad ? 1 : 0;
      }
      const double build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const double bpk = double(d->memory_consumption()) / double(kv.size());
      Dytis* raw = d.get();
      time_one(name.c_str(), kv, extra, [&](uint64_t k, uint64_t v) { return raw->put(k, v, &p); }, [&](uint64_t k, uint64_t& v) { return raw->get(k, v, &p); }, build, bpk);
    }
    const char* raw_env = std::getenv("TEST_DYTIS_TIME_RAW");
    const int raw_mode = raw_env ? std::atoi(raw_env) : 1;
    if (raw_mode) {  // the authors' code alone: the insert order of dytis (random, same seed), or with TEST_DYTIS_TIME_RAW=2 ascending; its memory is left to the pools
      using namespace gret::dytis_authors;
      UNIFORM_TEST = false, DEFAULT_MAX_BITS = gret_skewed_max_bits;
      std::unique_ptr<Thin> t(new Thin);
      std::vector<uint32_t> order(kv.size());
      for (size_t i = 0; i < order.size(); ++i) order[i] = uint32_t(i);
      std::mt19937_64 g2(0x9e3779b97f4a7c15ull);
      if (raw_mode != 2)
        for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[g2() % i]);
      const auto t0 = std::chrono::steady_clock::now();
      for (uint32_t i : order) t->put(kv[i].first, kv[i].second);
      const double build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      Thin* raw = t.get();
      time_one(raw_mode == 2 ? "authors-raw-a" : "authors-raw", kv, extra, [&](uint64_t k, uint64_t v) { raw->put(k, v); return true; }, [&](uint64_t k, uint64_t& v) { return raw->get(k, v); }, build, 0.0);
      t.release();  // the authors' destructor frees one array of the structure; nothing else is torn down here
    }
    const char* gre_env = std::getenv("TEST_DYTIS_TIME_GRE");
    if (!gre_env || std::atoi(gre_env)) {
#ifndef TEST_DYTIS_ALONE
    auto gre = [&](const char* name, Index* idx) {
      std::vector<std::pair<uint64_t, uint64_t>> copy = kv;
      idx->init(&p);
      const auto t0 = std::chrono::steady_clock::now();
      idx->bulk_load(copy.data(), copy.size(), &p);
      const double build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      time_one(name, kv, extra, [&](uint64_t k, uint64_t v) { return idx->put(k, v, &p); }, [&](uint64_t k, uint64_t& v) { return idx->get(k, v, &p); }, build,
               double(idx->memory_consumption()) / double(kv.size()));
    };
    gre("alex", new alexInterface<uint64_t, uint64_t>);
    gre("lipp", new LIPPInterface<uint64_t, uint64_t>);
    gre("btree", new BTreeInterface<uint64_t, uint64_t>);
#endif
    }
  }
  if (want("oom")) {  // last, because nothing can follow it: out of memory inside the authors' code (a limit of 1 MB) is a loud failure, for this index and for every later one of the process
    setenv("DYTIS_MAX_GB", "0.001", 1);
    std::vector<uint64_t> v = rnd(0, M, 100000);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    std::vector<std::pair<uint64_t, uint64_t>> kv;
    for (uint64_t k : v) kv.push_back({k, payload(k)});
    bool threw = false, refused = false, refused_again = false;
    {
      std::unique_ptr<Dytis> d(new Dytis("dytis"));
      try {
        d->bulk_load(kv.data(), kv.size());
      } catch (const std::runtime_error& e) {
        threw = std::strstr(e.what(), "out of memory") != nullptr;
      }
      try {
        d->bulk_load(kv.data(), 10);
      } catch (const std::logic_error&) {
        refused_again = true;
      }
    }
    unsetenv("DYTIS_MAX_GB");
    try {
      std::unique_ptr<Dytis> d(new Dytis("dytis"));
    } catch (const std::logic_error&) {
      refused = true;
    }
    if (!threw || !refused || !refused_again) {
      std::printf("FAIL oom: a bulk load past the memory limit %s, the same index %s, a later index %s\n", threw ? "threw" : "did not throw", refused_again ? "refused another load" : "took another load",
                  refused ? "was refused" : "was allowed");
      ++g_bad;
    } else {
      std::printf("dytis   oom           the memory limit stops the bulk load with an exception, and no index is made after it  ok\n");
    }
  }
  std::printf(g_bad ? "test_dytis: %zu FAILURES\n" : "test_dytis: all passed\n", g_bad);
  return g_bad ? 1 : 0;
}
