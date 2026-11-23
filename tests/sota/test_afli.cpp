// SPDX-License-Identifier: MIT
// afli and nfl_afli (include/gret/sota/afli.h: the NFL authors' AFLI behind GRE's interface, alone and behind this repository's
// reimplementation of NFL's flow) against a std::map: get on every stored key and on keys never stored (neighbours, 0, 2^53 and
// its neighbours, 2^63, 2^64 - 1, random), scan from stored, neighbouring, in-between and extreme start keys (afli keeps key
// order; nfl_afli behind the flow does not and must return 0), then put (neighbours of stored keys, runs, keys far outside,
// duplicates), update and remove, the same checks after each, every key removed, and keys put into the emptied index. The key
// sets are test_scan's patterns (tiny sets, uniform, wide, dense runs, clusters, far outliers, keys 0 and 2^64 - 1, a regular
// stride) at sizes from 1 to 2.1M keys, and sets made for what the authors' code cannot do: keys that share a double, keys a
// double apart at 2^63 (where their bulk load puts the first key of a node in a slot their find does not look in), key sets too
// sparse for any model, an index that is never bulk-loaded, and trees in which the authors' find would cast the slot of a key
// that is not stored out of int64 (cast_guard: the wrapper must notice, leave their find for its own walk, and answer right).
// On the ordinary sets the test also demands that lookups stayed with the authors' find.
// afli_cut (the authors' AFLI keyed by key mod 10^15, as their data preparation cuts uint64 keys) runs on the sets where plain
// afli fills the side store, on keys that share their last 15 digits, and on keys below 10^15, where it must be afli.
//
// nfl_afli runs on weight files the test writes itself (a monotone flow, a folding one) and, when the clone of the authors'
// repository is given, on the weights they ship for books and fb, with keys in the domain those were trained on (15 decimal
// digits) and with raw 64-bit keys, with and without _cut and _on.
//
// Build (docs/sota/afli.md; NFL = the clone of https://github.com/luffy06/NFL, SHIM = a directory with boost/optional.hpp,
// B = build/tests/sota/afli):
//   build/tests/cxx    -I $SHIM -I $NFL/src tests/sota/test_afli.cpp -o $B/t
//   build/tests/cxx-O3 -I $SHIM -I $NFL/src tests/sota/test_afli.cpp -o $B/t-O3
//   t [SET...] [--nfl DIR] [--time]
// --nfl DIR (or $NFL_DIR): the clone, for the shipped weights; without it those sets are skipped and the test says so.
// --time: a small lookup and insert timing on 3M uniform keys, against the authors' AFLI called directly (use t-O3).
// -DTEST_AFLI_ALONE leaves GRE's own indexes out (they are here to show the header lives next to them).
// The -O3 build replaces operator new to compare memory_consumption() with the bytes the index really holds.
#include "indexInterface.h"
#ifndef TEST_AFLI_ALONE  // GRE's own indexes first, as GRE's competitor.h has them: the new header must live next to them
#include "alex/alex.h"
#include "btree/btree.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#endif
#include "gret/sota/afli.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

// the authors' headers define these as macros; the wrapper's header must not leak them
#if defined(BIT_TYPE) || defined(BIT_LEN) || defined(GET_BIT) || defined(COMMON_H) || defined(AFLI_H) || defined(BUCKET_H)
#error "gret/sota/afli.h leaks the authors' macros"
#endif
template <class likely, class unlikely>
static size_t names_are_free() { return sizeof(likely) + sizeof(unlikely); }
static int64_t int64_is_itself = 0;  // and int64_t is a type again

// ---- the bytes the process holds through operator new (not under a sanitizer, which has its own operator new)
#if defined(__SANITIZE_ADDRESS__)
#define TEST_AFLI_NO_HEAP 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define TEST_AFLI_NO_HEAP 1
#endif
#endif
static size_t g_live = 0;
#ifndef TEST_AFLI_NO_HEAP
static void* counted_new(size_t n) {
  void* p = std::malloc(n + 16);
  if (!p) throw std::bad_alloc();
  *static_cast<size_t*>(p) = n;
  g_live += n;
  return static_cast<char*>(p) + 16;
}
static void counted_delete(void* q) noexcept {
  if (!q) return;
  void* p = static_cast<char*>(q) - 16;
  g_live -= *static_cast<size_t*>(p);
  std::free(p);
}
void* operator new(size_t n) { return counted_new(n); }
void* operator new[](size_t n) { return counted_new(n); }
void operator delete(void* p) noexcept { counted_delete(p); }
void operator delete[](void* p) noexcept { counted_delete(p); }
void operator delete(void* p, size_t) noexcept { counted_delete(p); }
void operator delete[](void* p, size_t) noexcept { counted_delete(p); }
static const bool g_heap = true;
#else
static const bool g_heap = false;
#endif

using Index = indexInterface<uint64_t, uint64_t>;
using Afli = gret::AfliIndex<uint64_t, uint64_t>;
using Map = std::map<uint64_t, uint64_t>;
static const uint64_t M = ~uint64_t(0);
static const uint64_t H = uint64_t(1) << 63;
static const uint64_t E = uint64_t(1) << 53;

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

static size_t g_bad = 0;
static void fail(const std::string& what, const char* op, uint64_t k) {
  if (g_bad++ < 40) std::printf("FAIL %s: %s wrong at key %llu\n", what.c_str(), op, (unsigned long long)k);
}

// every answer against the reference: get on every key it holds and on probes, scans from starts
// (sparse: the index holds few keys in a tree built for many, where every scan walks the emptied slots: fewer scans)
static void verify(Index& idx, bool ordered, const Map& ref, const std::vector<uint64_t>& probes, std::mt19937_64& g, const std::string& what, bool sparse = false) {
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
  for (uint64_t k : {uint64_t(0), uint64_t(1), M, M - 1, H, H - 1, H + 1, H - 2, E, E - 1, E + 1, E + 2, gret::AfliIndex<uint64_t, uint64_t>::kCut15, 2 * gret::AfliIndex<uint64_t, uint64_t>::kCut15 + 1}) probe(k);
  for (int i = 0; i < 20000; ++i) probe(g());
  for (int i = 0; i < 20000 && !all.empty(); ++i) {  // in the gaps and next to stored keys, and what shares a stored key's double or its last 15 digits
    const size_t r = g() % all.size();
    const uint64_t k = all[r].first;
    probe(k + 1), probe(k - 1), probe(k ^ 1), probe(k ^ 512), probe(k + 1000000000000000ull), probe(k - 1000000000000000ull);
    if (r + 1 < all.size() && all[r + 1].first - k > 2) probe(k + 1 + g() % (all[r + 1].first - k - 1));
  }
  std::vector<uint64_t> starts = {0, 1, M, M - 1, H, H - 1, H + 1, E, E - 1, E + 1};
  auto around = [&](uint64_t k) { starts.push_back(k), starts.push_back(k + 1), starts.push_back(k - 1); };
  for (size_t i = 0; i < std::min<size_t>(sparse ? 3 : 40, all.size()); ++i) around(all[i].first), around(all[all.size() - 1 - i].first);
  for (int i = 0; i < (sparse ? 10 : 1500) && !all.empty(); ++i) {
    const size_t r = g() % all.size();
    around(all[r].first);
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) starts.push_back(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  for (size_t i = 0; i < probes.size() && i < (sparse ? 10 : 600); ++i) around(probes[i]);
  for (int i = 0; i < (sparse ? 5 : 100); ++i) starts.push_back(g());
  const size_t lens[] = {1, 2, 3, 16, 256, 5000, 70000};
  std::vector<std::pair<uint64_t, uint64_t>> out;
  auto less = [](const std::pair<uint64_t, uint64_t>& a, uint64_t b) { return a.first < b; };
  for (uint64_t s : starts)
    for (size_t len : lens) {
      if (len > 256 && (g() & 31)) continue;  // long scans on a sample of the starts
      out.assign(len + 1, {0, 0});
      const size_t got = idx.scan(s, len, out.data());
      if (!ordered) {  // no key order behind the flow: scan is unsupported, returns 0 and writes nothing
        if (got != 0 || out[0] != std::pair<uint64_t, uint64_t>(0, 0)) fail(what, "scan (must return 0)", s);
        continue;
      }
      const size_t at = size_t(std::lower_bound(all.begin(), all.end(), s, less) - all.begin());
      bool ok = got == std::min(len, all.size() - at) && out[len] == std::pair<uint64_t, uint64_t>(0, 0);
      for (size_t i = 0; ok && i < got; ++i) ok = out[i] == all[at + i];
      if (!ok) fail(what, "scan", s);
    }
  if (ordered) {  // one scan over everything
    out.assign(all.size() + 7, {0, 0});
    const size_t got = idx.scan(0, all.size() + 7, out.data());
    bool ok = got == all.size();
    for (size_t i = 0; ok && i < got; ++i) ok = out[i] == all[i];
    if (!ok) fail(what, "full scan", 0);
  }
}

struct Expect {
  double side_max = 0;   // the largest share of the keys the side store may take at the bulk load
  double side_min = 0;   // and the smallest (sets made to hit the guards)
  int flow = -1;         // nfl_afli: 1 the flow must be on, 0 off, -1 either
  int slot = 0;          // 16 or 24: the slot size the wrapper must choose; 0 either
  int order = -1;        // 1: scan must work, 0: must return 0
  bool writes = true;    // the put / update / remove phases
  bool no_load = false;  // never bulk-loaded: everything arrives through put
};

static void check(const std::string& label, std::vector<uint64_t> keys, const std::string& name, const Expect& ex) {
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  const std::string what = name + " on " + label;
  const size_t bad0 = g_bad;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::pair<uint64_t, uint64_t>> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  if (kv.size() > 10) kv[kv.size() / 3].second = M, kv[2 * kv.size() / 3].second = 0;  // payloads that are sentinels elsewhere
  const size_t live0 = g_live;
  Index* idx = gret::make_afli<uint64_t, uint64_t>(name);
  if (!idx) { std::printf("FAIL %s: no such index\n", name.c_str()); ++g_bad; return; }
  std::unique_ptr<Afli> owner(dynamic_cast<Afli*>(idx));  // deleted as what it is: indexInterface has no virtual destructor
  Afli* d = owner.get();
  Param p(1, 0);
  idx->init(&p);
  std::mt19937_64 g(std::hash<std::string>()(label) ^ 0x9e3779b97f4a7c15ull);
  Map ref;
  std::vector<uint64_t> probes;
  try {
    if (ex.no_load) {
      uint64_t v = 0;
      if (idx->get(5, v) || idx->remove(5) || idx->update(5, 1) || idx->scan(0, 3, kv.data()) != 0) fail(what, "an index never loaded answers", 5);
      std::vector<size_t> order(kv.size());
      for (size_t i = 0; i < order.size(); ++i) order[i] = i;
      std::shuffle(order.begin(), order.end(), g);
      for (size_t i : order)
        if (!idx->put(kv[i].first, kv[i].second)) fail(what, "put (first)", kv[i].first);
      for (size_t i = 0; i < order.size(); i += 3)
        if (idx->put(kv[order[i]].first, 1)) fail(what, "put (again)", kv[order[i]].first);
    } else {
      idx->bulk_load(kv.data(), kv.size(), &p);
    }
  } catch (const std::exception& e) {
    std::printf("FAIL %s: loading threw %s\n", what.c_str(), e.what());
    ++g_bad;
    return;
  }
  const size_t held = g_live - live0;  // what the index holds through operator new, before the test allocates anything else
  ref.insert(kv.begin(), kv.end());
  const size_t side0 = d->in_side(), in0 = d->in_afli();
  if (in0 + side0 != keys.size()) fail(what, "the index holds another number of keys than", keys.size());
  if (double(side0) > ex.side_max * double(keys.size()) + (ex.no_load ? double(keys.size()) : 0) || double(side0) < ex.side_min * double(keys.size()))
    fail(what, "the side store's share is outside what this set should give; side keys:", side0);
  if (ex.flow >= 0 && int(d->flow_on()) != ex.flow) fail(what, "NFL's switch decided otherwise than this set should make it; flow_on:", d->flow_on());
  if (ex.slot && d->slot_bytes() != ex.slot) fail(what, "the slot size is not the expected one, but", uint64_t(d->slot_bytes()));
  if (ex.order >= 0 && int(d->keeps_order()) != ex.order) fail(what, "key order kept:", d->keeps_order());
  if (keys.size() >= 1000 && d->checked_lookups()) fail(what, "lookups left the authors' find after the load; models found unsafe:", d->unsafe_models());
  const long long bytes = idx->memory_consumption();
  if (bytes < (long long)(16 * in0)) fail(what, "memory_consumption below 16 bytes a key in the tree:", uint64_t(bytes));
  double heap_ratio = 0;
  if (g_heap && !keys.empty()) {  // against what it reports (the name and the flow's vector headers are not counted: 256 bytes of slack)
    heap_ratio = double(held) / double(bytes);
    const double diff = double(held) - double(bytes);
    if (diff > 256 + 0.002 * double(bytes) || diff < -256 - 0.002 * double(bytes)) fail(what, "memory_consumption is not the bytes held; held:", uint64_t(held));
  }
  verify(*idx, d->keeps_order(), ref, probes, g, what + " (loaded)");

  auto pick = [&]() { return keys[g() % keys.size()]; };
  if (ex.writes && !keys.empty()) {
    const size_t scale = std::max<size_t>(1, std::min<size_t>(20, keys.size() / 100000 + 1));
    // put: neighbours of stored keys, runs behind a stored key, keys anywhere, keys far outside, special keys, keys that share a
    // stored key's double or its last 15 digits, duplicates
    {
      std::vector<std::pair<uint64_t, uint64_t>> add;
      for (size_t i = 0; i < 1000 * scale; ++i) { const uint64_t k = pick(); add.push_back({k + 1, payload(k + 1)}); if (i % 3 == 0) add.push_back({k + 2, payload(k + 2)}); if (i % 7 == 0) add.push_back({k - 1, payload(k - 1)}); }
      for (int r = 0; r < 4; ++r) { const uint64_t k = pick(); for (uint64_t j = 1; j <= 150 * scale; ++j) add.push_back({k + j * (r + 1), payload(k + j)}); }
      for (size_t i = 0; i < 1000 * scale; ++i) { const uint64_t span = keys.back() - keys.front(), k = keys.front() + (span == M ? g() : g() % (span + 1)); add.push_back({k, payload(k)}); }
      for (size_t i = 0; i < 150 * scale; ++i) { const uint64_t k = g(); add.push_back({k, payload(k)}); }
      for (size_t i = 0; i < 100 * scale; ++i) { const uint64_t k = pick(); add.push_back({k ^ 3, i}), add.push_back({k + 1024, i}), add.push_back({k + 1000000000000000ull, i}), add.push_back({k % 1000000000000000ull, i}); }
      for (uint64_t j = 0; j < 40; ++j) {
        add.push_back({keys.back() + 1 + j, j}), add.push_back({keys.back() + (uint64_t(1) << (j + 4)), j}), add.push_back({keys.front() - 1 - j, j});
        add.push_back({keys.front() >> (j % 20 + 1), j}), add.push_back({keys.back() + (keys.back() - keys.front()) * (j + 1), j});
      }
      for (uint64_t k : {uint64_t(0), uint64_t(1), M, M - 1, H, H - 1, H + 1, H - 2, E, E - 1, E + 1}) add.push_back({k, payload(k)});
      for (size_t i = 0; i < 100 * scale; ++i) add.push_back({pick(), 99});  // there already
      std::shuffle(add.begin(), add.end(), g);
      for (const auto& a : add) {
        const bool fresh = ref.emplace(a.first, a.second).second;
        if (idx->put(a.first, a.second) != fresh) fail(what, "put", a.first);
        probes.push_back(a.first);
      }
      if (d->in_afli() + d->in_side() != ref.size()) fail(what, "after put the index counts another number of keys than", ref.size());
      verify(*idx, d->keeps_order(), ref, probes, g, what + " (put)");
    }
    // update: present keys, absent keys return false
    {
      for (size_t i = 0; i < 1500 * scale; ++i) {
        uint64_t k = i % 4 ? pick() : probes[g() % probes.size()];
        if (i % 9 == 0) k += 1;
        const uint64_t v = i % 50 == 0 ? M : g();
        const auto it = ref.find(k);
        if (idx->update(k, v) != (it != ref.end())) fail(what, "update", k);
        if (it != ref.end()) it->second = v;
      }
      verify(*idx, d->keeps_order(), ref, probes, g, what + " (updated)");
    }
    // remove: random keys (loaded ones and put ones) and both ends; absent keys and a second remove return false; then puts again
    {
      std::vector<uint64_t> gone;
      for (size_t i = 0; i < 1500 * scale; ++i) gone.push_back(i % 3 ? pick() : probes[g() % probes.size()]);
      for (size_t i = 0; i < 60 && i < keys.size(); ++i) gone.push_back(keys[i]), gone.push_back(keys[keys.size() - 1 - i]);
      for (uint64_t k : gone) {
        const bool there = ref.erase(k) != 0;
        if (idx->remove(k) != there) fail(what, "remove", k);
        if (idx->remove(k)) fail(what, "second remove", k);
        probes.push_back(k);
      }
      for (size_t i = 0; i < 100 * scale; ++i) {
        const uint64_t k = pick() + 1;
        if (idx->remove(k) != (ref.erase(k) != 0)) fail(what, "remove (neighbour)", k);
      }
      verify(*idx, d->keeps_order(), ref, probes, g, what + " (removed)");
      for (size_t i = 0; i < gone.size(); i += 2) {
        const uint64_t k = gone[i] + (i % 4 == 0);
        if (idx->put(k, payload(k) ^ 1) != ref.emplace(k, payload(k) ^ 1).second) fail(what, "put (after remove)", k);
        probes.push_back(k);
      }
      if (d->in_afli() + d->in_side() != ref.size()) fail(what, "after the removes the index counts another number of keys than", ref.size());
      verify(*idx, d->keeps_order(), ref, probes, g, what + " (put after remove)");
    }
  }
  const size_t side1 = d->in_side(), in1 = d->in_afli(), fixed = d->repaired(), rebuilt = d->rebuilds();
  if (keys.size() >= 1000 && d->checked_lookups()) fail(what, "lookups left the authors' find during the writes; models found unsafe:", d->unsafe_models());
  if (ex.writes && !keys.empty()) {  // every key out, in a random order; then an index that holds nothing, and takes keys again
    std::vector<uint64_t> order;
    for (const auto& kvp : ref) order.push_back(kvp.first);
    std::shuffle(order.begin(), order.end(), g);
    for (uint64_t k : order)
      if (!idx->remove(k)) fail(what, "remove (emptying)", k);
    ref.clear();
    if (d->in_afli() || d->in_side()) fail(what, "emptied, but the index still counts keys:", d->in_afli() + d->in_side());
    probes.resize(std::min<size_t>(probes.size(), 5000));
    for (size_t i = 0; i < 5000; ++i) probes.push_back(order[i % order.size()]);
    verify(*idx, d->keeps_order(), ref, probes, g, what + " (emptied)", true);
    for (size_t i = 0; i < 20000 && i < order.size(); ++i) {
      const uint64_t k = order[i] + (i % 2);
      if (idx->put(k, k ^ 7) != ref.emplace(k, k ^ 7).second) fail(what, "put (into the emptied index)", k);
    }
    verify(*idx, d->keeps_order(), ref, probes, g, what + " (refilled)", true);
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%-17s %-15s n=%-8zu afli=%zu side=%zu (%.3f%%) slot=%dB bucket=%u%s %.1f B/key", name.c_str(), label.c_str(), keys.size(), in0, side0,
              keys.empty() ? 0.0 : 100.0 * double(side0) / double(keys.size()), d->slot_bytes(), d->bucket(),
              name.compare(0, 3, "nfl") ? "" : d->flow_on() ? " flow=on" : " flow=off", keys.empty() ? 0.0 : double(bytes) / double(keys.size()));
  if (g_heap && !keys.empty()) std::printf(" (heap %.4f of it)", heap_ratio);
  std::printf("; after writes: afli=%zu side=%zu rebuilds=%zu repaired=%zu%s  %s  %.1fs\n", in1, side1, rebuilt, fixed, d->checked_lookups() ? " LOOKUPS=wrapper's walk" : "",
              g_bad == bad0 ? "ok" : "FAILED", s);
  std::fflush(stdout);
}

// The authors' find casts a predicted slot to int64 with no range check (NFL/src/models/linear_model.h:17; UBSan: "1.84467e+19
// is outside the range of representable values of type 'long'" when the wrapper does not step in). Trees made to get there
// with a key that is not stored: a few dense keys (a steep model) under a slot that a far key, removed since, stretched. The
// wrapper must notice (checked_lookups), answer every lookup right, and the sanitized build must not see the cast. A third
// tree keeps the far key: nothing is unsafe there and lookups must stay with the authors' find.
static void cast_guard() {
  const uint64_t A = uint64_t(1) << 20;
  for (int variant = 0; variant < 3; ++variant) {
    const size_t bad0 = g_bad;
    const char* const label[] = {"far key removed before the root got its model", "far key removed from the last slot's bucket, which then became a node", "the same with the far key kept"};
    const std::string name = variant == 0 ? "afli2" : "afli3", what = name + " cast guard, " + label[variant];
    std::mt19937_64 g(99 + variant);
    Map ref;
    std::vector<uint64_t> probes;
    std::unique_ptr<Afli> d(dynamic_cast<Afli*>(gret::make_afli<uint64_t, uint64_t>(name)));
    Index* idx = d.get();
    auto put = [&](uint64_t k) {
      if (idx->put(k, payload(k)) != ref.emplace(k, payload(k)).second) fail(what, "put", k);
      probes.push_back(k);
    };
    auto del = [&](uint64_t k) {
      if (idx->remove(k) != (ref.erase(k) != 0)) fail(what, "remove", k);
      probes.push_back(k);
    };
    if (variant == 0) {  // never loaded: 2^64 - 1 sits in the root while that is a sorted array and is gone when the root is built
      put(A), put(M), del(M), put(A + 1), put(A + 2), put(0);
    } else {  // a root of five slots over 0 .. 4000; 2^64 - 1 joins 4000 in the last slot's bucket
      put(0), put(1000), put(2000), put(3000), put(4000), put(M);
      if (variant == 1) del(M);
      put(4001), put(4002), put(4003);  // the bucket overflows: with 2^64 - 1 gone, a node over 4000 .. 4003, slope 1, that every key above 4000 arrives at
    }
    if (d->checked_lookups() != (variant != 2)) fail(what, variant != 2 ? "the wrapper did not notice the unsafe model; unsafe models:" : "the wrapper left the authors' find without need; unsafe models:", d->unsafe_models());
    for (uint64_t k : {M - 100000, M, H, H + 4096, H - 12345, uint64_t(1) << 62, (uint64_t(1) << 62) + 7, uint64_t(4100), A + 3, uint64_t(3) << 61}) probes.push_back(k);
    verify(*idx, true, ref, probes, g, what);
    for (int i = 0; i < 6000; ++i) {  // and it keeps working: writes near the dense keys, far away, anywhere
      const uint64_t r = g(), k = i % 4 == 0 ? r : i % 4 == 1 ? (variant ? 4000 : A) + r % 3000 : i % 4 == 2 ? M - r % (uint64_t(1) << 40) : r % (uint64_t(1) << 23);
      if (i % 5 == 4 && !probes.empty()) del(probes[r % probes.size()]);
      else put(k);
    }
    if (d->in_afli() + d->in_side() != ref.size()) fail(what, "the index counts another number of keys than", ref.size());
    verify(*idx, true, ref, probes, g, what + " (after writes)");
    std::printf("%-17s cast guard: %-70s unsafe models found=%zu lookups=%s afli=%zu side=%zu  %s\n", name.c_str(), label[variant], d->unsafe_models(),
                d->checked_lookups() ? "wrapper's walk" : "authors' find", d->in_afli(), d->in_side(), g_bad == bad0 ? "ok" : "FAILED");
    std::fflush(stdout);
  }
}

// a weight file in the format NFL's trainer writes (train/numerical_flow.py save_weights), for flows the test makes up
static std::string write_weights(const std::string& dir, const char* file, const char* text) {
  const std::string path = dir + "/" + file;
  std::ofstream(path) << text;
  return path;
}

// the authors' AFLI called directly, for --time: the thinnest thing GRE could call
struct ThinAfli {
  gret::afli_authors::nfl::AFLI<double, uint64_t> a;
  void bulk_load(std::pair<uint64_t, uint64_t>* kv, size_t n) {
    std::vector<std::pair<double, uint64_t>> in(n);
    for (size_t i = 0; i < n; ++i) in[i] = {double(kv[i].first), kv[i].second};
    a.bulk_load(in.data(), uint32_t(n));
  }
  bool get(uint64_t k, uint64_t& v) {
    auto it = a.find(double(k));
    if (it.is_end()) return false;
    v = it.value();
    return true;
  }
  bool put(uint64_t k, uint64_t v) {
    a.insert({double(k), v});
    return true;
  }
  long long memory_consumption() { return (long long)a.index_size(); }
};

// and the same behind GRE's interface, with nothing added: what a wrapper without any check would cost
struct ThinVirtual final : Index {
  ThinAfli t;
  void init(Param* = nullptr) override {}
  void bulk_load(std::pair<uint64_t, uint64_t>* kv, size_t n, Param* = nullptr) override { t.bulk_load(kv, n); }
  bool get(uint64_t k, uint64_t& v, Param* = nullptr) override { return t.get(k, v); }
  bool put(uint64_t k, uint64_t v, Param* = nullptr) override { return t.put(k, v); }
  bool update(uint64_t, uint64_t, Param* = nullptr) override { return false; }
  bool remove(uint64_t, Param* = nullptr) override { return false; }
  size_t scan(uint64_t, size_t, std::pair<uint64_t, uint64_t>*, Param* = nullptr) override { return 0; }
  long long memory_consumption() override { return t.memory_consumption(); }
};

template <class I, class Get, class Put>
static void time_one(const char* name, I* idx, const std::vector<std::pair<uint64_t, uint64_t>>& kv, const std::vector<std::pair<uint64_t, uint64_t>>& extra, Get get, Put put) {
  std::vector<std::pair<uint64_t, uint64_t>> copy = kv;
  auto t0 = std::chrono::steady_clock::now();
  idx->bulk_load(copy.data(), copy.size());
  const double build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::mt19937_64 g(3);
  std::vector<uint64_t> q(2000000);
  for (uint64_t& x : q) x = kv[g() % kv.size()].first;
  uint64_t sum = 0, v = 0;
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    t0 = std::chrono::steady_clock::now();
    for (uint64_t x : q) sum += get(idx, x, v) ? v : 1;
    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(q.size()));
  }
  t0 = std::chrono::steady_clock::now();
  size_t in = 0;
  for (const auto& e : extra) in += put(idx, e.first, e.second);
  const double ins = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(extra.size());
  std::printf("time %-22s n=%zu build %.2fs  get %.1f ns  put %.1f ns (%zu new)  %.1f B/key  [checksum %llu]\n", name, kv.size(), build, best, ins, in,
              double(idx->memory_consumption()) / double(kv.size() + in), (unsigned long long)sum);
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  std::vector<std::string> only;
  bool timing = false;
  std::string nfl_dir = std::getenv("NFL_DIR") ? std::getenv("NFL_DIR") : "";
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--time")) timing = true;
    else if (!std::strcmp(argv[i], "--nfl") && i + 1 < argc) nfl_dir = argv[++i];
    else only.push_back(argv[i]);
  auto want = [&](const std::string& s) { return only.empty() || std::find(only.begin(), only.end(), s) != only.end(); };
  if (names_are_free<uint64_t, uint64_t>() != 16 || int64_is_itself != 0) return 2;

  for (const char* bad : {"afl", "afli0", "afli7", "afli12", "afli_on", "afli_cut_on", "afli_cut_cut", "afli3_cu", "afli_cut3", "nfl_afli_", "nfl_afli_on_cut", "nfl_afli_cut_cut", "nfl_afli7", "nfl_afli3x", "nfl_aflicut", "afli ", "Afli", "nfl_afl"})
    if (gret::make_afli<uint64_t, uint64_t>(bad)) {
      std::printf("FAIL make_afli accepts the malformed name '%s'\n", bad);
      ++g_bad;
    }
  for (const char* good : {"afli", "afli1", "afli6", "afli_cut", "afli3_cut", "nfl_afli", "nfl_afli4", "nfl_afli_cut", "nfl_afli_on", "nfl_afli_cut_on", "nfl_afli2_cut_on"}) {
    std::unique_ptr<Afli> made(dynamic_cast<Afli*>(gret::make_afli<uint64_t, uint64_t>(good)));
    if (!made) {
      std::printf("FAIL make_afli refuses the name '%s'\n", good);
      ++g_bad;
    }
  }
  if (gret::make_afli<uint32_t, uint64_t>("afli") || gret::make_afli<uint64_t, uint32_t>("afli")) {
    std::printf("FAIL make_afli takes key or payload types it cannot hold\n");
    ++g_bad;
  }

  const size_t N = 2100000;
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; v.reserve(m); std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  Expect all;                       // everything in the authors' tree
  all.side_max = 0, all.order = 1;
  Expect small = all;  small.slot = 16;   // keys below 2^53: the authors' 16-byte slots
  Expect big = all;    big.slot = 24;     // keys from 2^53: the key is stored next to its double

  // ---- afli: the authors' AFLI alone
  if (want("tiny")) {
    check("empty", {}, "afli", all);
    check("n1", {5}, "afli", small);
    check("n1max", {M}, "afli", big);
    check("n1zero", {0}, "afli", small);
    check("n2", {0, M}, "afli", big);
    check("n3", {1, 2, M}, "afli", big);
    check("n5", {0, 1, 2, 3, M}, "afli", big);
    check("n17", rnd(0, uint64_t(1) << 40, 17), "afli", small);
    check("n17", rnd(0, uint64_t(1) << 40, 17), "afli1", small);
    check("uniform20k", rnd(1, uint64_t(1) << 50, 20000), "afli", small);
    check("sparse3k", rnd(0, M, 3000), "afli", big);  // fewer keys than 2.2e-16 times their range: no model fits, the root is a sorted array
    { Expect e = all; e.no_load = true; check("never_loaded", rnd(0, uint64_t(1) << 40, 30000), "afli", e); check("never_loaded64", rnd(0, M, 30000), "afli3", e); }
  }
  if (want("uniform")) check("uniform", rnd(1, uint64_t(1) << 50, N), "afli", small);
  if (want("buckets")) {  // the knob: the maximum bucket size
    check("uniform", rnd(1, uint64_t(1) << 50, 300000), "afli1", small);
    check("uniform", rnd(1, uint64_t(1) << 50, 300000), "afli3", small);
    check("uniform", rnd(1, uint64_t(1) << 50, 300000), "afli6", small);
    check("wide", rnd(0, M, 300000), "afli1", big);
    check("wide", rnd(0, M, 300000), "afli6", big);
  }
  if (want("wide")) {  // the whole key space, with its ends
    std::vector<uint64_t> v = rnd(0, M, N);
    for (uint64_t k : {uint64_t(0), M, H, H - 1}) v.push_back(k);
    Expect e = big; e.side_max = 1e-5;  // H - 1 has H's double
    check("wide", v, "afli", e);
  }
  if (want("run")) { std::vector<uint64_t> v; for (uint64_t i = 0; i < 500000; ++i) v.push_back(1000 + i); check("run", v, "afli", small); }
  if (want("clusters")) {
    std::vector<uint64_t> v;
    for (int c = 0; c < 5000; ++c) { const uint64_t b = g() >> 16; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); }
    check("clusters", v, "afli", small);
  }
  if (want("clusters64")) {  // runs of consecutive keys up to 2^63: all keys of a run above 2^53 share a few doubles
    std::vector<uint64_t> v;
    for (int c = 0; c < 3000; ++c) { const uint64_t b = g() >> 1; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); }
    Expect e = big; e.side_max = 1.0, e.side_min = 0.5;
    check("clusters64", v, "afli", e);
  }
  if (want("outliers")) {  // a narrow bulk and keys far from it
    std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), 300000);
    for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003);
    v.push_back(3), v.push_back(M);
    Expect e = big; e.side_max = 1e-3;
    check("outliers", v, "afli", e);
  }
  if (want("over64outliers")) {
    std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + 100000, 30000);
    for (uint64_t j = 0; j < 100; ++j) v.push_back(M - j * 1000);
    Expect e = big; e.side_max = 0.01;
    check("over64outliers", v, "afli", e);
  }
  if (want("stride")) { std::vector<uint64_t> v; for (uint64_t i = 0; i < 400000; ++i) v.push_back(i * 65536 + 12345); check("stride", v, "afli", small); }
  if (want("hugegap")) {  // a run from 0 and a run down from 2^64 - 1: up there 2048 consecutive keys share one double
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < 200000; ++i) v.push_back(i), v.push_back(M - i);
    Expect e = big; e.side_min = 0.49, e.side_max = 0.51;
    check("hugegap", v, "afli", e);
  }
  if (want("highmag")) {  // large keys far enough apart, with pairs and triples that share a double
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < 400000; ++i) v.push_back((uint64_t(1) << 61) + (i << 20) + (g() & 0x7ffff));
    for (int i = 0; i < 30000; ++i) { const uint64_t k = v[g() % 400000]; v.push_back(k + 1); if (i & 1) v.push_back(k + 2); if (i % 5 == 0) v.push_back(k + 700); }
    Expect e = big; e.side_min = 0.02, e.side_max = 0.15;
    check("highmag", v, "afli", e);
  }
  if (want("highrun")) {  // keys one double apart at 2^63: no regression fits, and the authors' fallback misplaces the first key of a node
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < 2000000; ++i) v.push_back(H + i * 2048);
    check("highrun", v, "afli", big);
    v.clear();
    for (uint64_t i = 0; i < 300000; ++i) v.push_back((uint64_t(1) << 62) + (uint64_t(1) << 60) + i * 1024);
    check("highrun62", v, "afli2", big);
    v.clear();
    for (int c = 0; c < 3000; ++c) { const uint64_t b = (g() >> 2) | (uint64_t(1) << 61); for (uint64_t i = 0; i < 100; ++i) v.push_back((b & ~uint64_t(1023)) + i * 1024); }
    check("highclusters", v, "afli", big);
    v.clear();  // and the same kind of keys arriving one at a time: every node is one the authors' insert built
    for (uint64_t i = 0; i < 200000; ++i) v.push_back(H + (uint64_t(1) << 62) + i * 2048);
    Expect e = big; e.no_load = true;
    check("highrun_puts", v, "afli", e);
    check("highrun_puts", v, "afli1", e);
  }
  if (want("fblike")) {  // small keys, a few very large ones
    std::vector<uint64_t> v = rnd(0, uint64_t(1) << 36, 500000);
    for (int j = 0; j < 21; ++j) v.push_back((uint64_t(1) << 59) + (g() >> 2));
    for (int j = 0; j < 10; ++j) v.push_back(H + (g() >> 1));
    Expect e = big; e.side_max = 1e-3;
    check("fblike", v, "afli", e);
  }
  if (want("cut")) {  // afli_cut: the authors' AFLI keyed by key mod 10^15, their own preparation of uint64 keys; no key order, scan returns 0
    Expect c;  // 24-byte slots, no scan; the side store only for keys that share their last 15 digits
    c.side_max = 1e-3, c.slot = 24, c.order = 0;
    check("wide", rnd(0, M, 300000), "afli_cut", c);
    check("wide", rnd(0, M, 300000), "afli1_cut", c);
    check("wide", rnd(0, M, N), "afli_cut", c);
    { std::vector<uint64_t> v; for (int cl = 0; cl < 3000; ++cl) { const uint64_t b = g() >> 1; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } check("clusters64", v, "afli_cut", c); }  // 98% in the side store under plain afli
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < 200000; ++i) v.push_back(i), v.push_back(M - i); check("hugegap", v, "afli_cut", c); }                                              // 50% under plain afli
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < 300000; ++i) v.push_back(H + i * 2048); check("highrun", v, "afli3_cut", c); }
    { std::vector<uint64_t> v = rnd(0, uint64_t(1) << 36, 300000); for (int j = 0; j < 31; ++j) v.push_back((uint64_t(1) << 59) + (g() >> 2)); check("fblike", v, "afli_cut", c); }
    {  // pairs that share their last 15 digits: one of each pair is in the side store
      std::vector<uint64_t> v = rnd(0, 999999999999999ull, 100000);
      const size_t m = v.size();
      for (size_t i = 0; i < m; ++i) v.push_back(v[i] + 1000000000000000ull * (1 + g() % 18000));
      Expect e = c; e.side_min = 0.45, e.side_max = 0.55;
      check("same15", v, "afli_cut", e);
    }
    check("uniform", rnd(1, uint64_t(1) << 49, 300000), "afli_cut", small);  // no key reaches 10^15: nothing to cut, the tree is afli's, scan works
    check("n2", {0, M}, "afli_cut", c);
    check("n1max", {M}, "afli_cut", c);
    check("n1", {5}, "afli_cut", small);
    check("empty", {}, "afli_cut", all);
    { Expect e = all; e.no_load = true; check("never_loaded64", rnd(0, M, 30000), "afli_cut", e); }  // no bulk load: nothing was cut, the tree is on the keys
  }
  if (want("castguard")) cast_guard();
  if (want("lognormal")) {
    std::lognormal_distribution<double> ln(0, 2);
    std::vector<uint64_t> v;
    for (int i = 0; i < 500000; ++i) v.push_back(uint64_t(ln(g) * 1e9));
    check("lognormal", v, "afli", small);
  }

  // ---- nfl_afli: the hybrid
  namespace fs = std::filesystem;
  const std::string tmp = (fs::temp_directory_path() / ("gret_test_afli_" + std::to_string((long long)getpid()))).string();
  fs::create_directories(tmp);
  // one input, one hidden unit, two layers: z = 3 tanh(x), x = key / 2^62: monotone, and flat enough at the top to fold keys together
  const std::string w_mono = write_weights(tmp, "mono.txt", "1\t1\t2\n0.0\t4611686018427387904.0\n1\t1\n1.0\n1\t1\n3.0\n");
  // the shape the authors ship (2D2H2L): z = 0.9 tanh(1e-6 x) + 1.3 tanh(frac(x)), x = key / 1e9: the low nine digits lead
  const std::string w_low9 = write_weights(tmp, "low9.txt", "2\t2\t2\n0.0\t1000000000.0\n2\t2\n0.000001\t0.0\n0.0\t1.0\n2\t2\n0.9\t0.0\n0.0\t1.3\n");
  auto with_weights = [&](const std::string& path, const std::string& label, const std::vector<uint64_t>& keys, const std::string& name, const Expect& e) {
    setenv("TRANSFORM_NFL", path.c_str(), 1);
    check(label, keys, name, e);
    unsetenv("TRANSFORM_NFL");
  };
  Expect hy;  // either decision of the switch; scan as the wrapper says
  hy.side_max = 1e-3;
  // keys whose high digits are skewed and whose low nine digits are uniform, below 10^15: where a low-digit flow helps
  auto skewed = [&](size_t m) {
    std::lognormal_distribution<double> ln(0, 2);
    std::vector<uint64_t> v;
    for (size_t i = 0; i < m; ++i) v.push_back((uint64_t(std::min(ln(g) * 2000.0, 999999.0)) * 1000000000ull + g() % 1000000000ull));
    return v;
  };
  if (want("nfl")) {
    {  // without the weights the index must refuse to load
      unsetenv("TRANSFORM_NFL");
      std::unique_ptr<Afli> d(dynamic_cast<Afli*>(gret::make_afli<uint64_t, uint64_t>("nfl_afli")));
      std::vector<std::pair<uint64_t, uint64_t>> kv = {{1, 1}, {2, 2}};
      bool threw = false;
      try { d->bulk_load(kv.data(), kv.size()); } catch (const std::runtime_error&) { threw = true; }
      if (!threw || d->in_afli() || d->in_side()) { std::printf("FAIL nfl_afli without TRANSFORM_NFL did not throw\n"); ++g_bad; }
      setenv("TRANSFORM_NFL", (tmp + "/missing.txt").c_str(), 1);
      threw = false;
      try { d->bulk_load(kv.data(), kv.size()); } catch (const std::runtime_error&) { threw = true; }
      if (!threw) { std::printf("FAIL nfl_afli with a missing weight file did not throw\n"); ++g_bad; }
      unsetenv("TRANSFORM_NFL");
    }
    with_weights(w_mono, "empty", {}, "nfl_afli", hy);
    with_weights(w_mono, "n1", {5}, "nfl_afli", hy);
    { Expect e = hy; e.side_max = 1.0; with_weights(w_mono, "n3", {1, 2, M}, "nfl_afli_on", e); with_weights(w_mono, "n17", rnd(0, uint64_t(1) << 40, 17), "nfl_afli_on", e); }  // 1 and 2 are one value behind this flow
    { Expect e = hy; e.no_load = true; with_weights(w_low9, "never_loaded", skewed(20000), "nfl_afli_cut_on", e); }  // no bulk load, no flow: the tree is on the keys
    { Expect e = hy; e.flow = 0, e.order = 1, e.slot = 16; with_weights(w_mono, "uniform", rnd(1, uint64_t(1) << 50, 300000), "nfl_afli", e); }  // uniform keys: nothing to gain, NFL's switch says off
    { Expect e = hy; e.flow = 1, e.order = 0, e.slot = 24, e.side_max = 0.02; with_weights(w_mono, "uniform", rnd(1, uint64_t(1) << 50, 300000), "nfl_afli_on", e); }
    { Expect e = hy; e.flow = 1, e.order = 0, e.side_max = 0.6; with_weights(w_mono, "wide", rnd(0, M, 300000), "nfl_afli3_on", e); }  // tanh is flat above 2^62 x 18: keys fold
    { Expect e = hy; e.flow = 1, e.order = 0, e.slot = 24; with_weights(w_low9, "skewed", skewed(N), "nfl_afli", e); }
    { Expect e = hy; e.flow = 1, e.order = 0; with_weights(w_low9, "skewed", skewed(300000), "nfl_afli2_cut", e); }
    { Expect e = hy; e.order = 0, e.flow = 1, e.side_max = 1.0; with_weights(w_low9, "wide_raw", rnd(0, M, 300000), "nfl_afli_on", e); }       // raw 64-bit keys: frac() has no digits left, keys fold
    { Expect e = hy; e.order = 0, e.flow = 1, e.side_max = 1e-3; with_weights(w_low9, "wide_cut", rnd(0, M, 300000), "nfl_afli_cut_on", e); }  // the same keys, cut as the authors cut them
    { Expect e = hy; e.order = 0, e.side_max = 1.0; with_weights(w_low9, "wide_cut_off", rnd(0, M, 300000), "nfl_afli_cut", e); }             // whatever the switch says, key order is gone with _cut
    { Expect e = hy; e.order = 0, e.flow = 1, e.side_max = 1.0; with_weights(w_low9, "run", [&] { std::vector<uint64_t> v; for (uint64_t i = 0; i < 200000; ++i) v.push_back(1000 + i); return v; }(), "nfl_afli_on", e); }
    { Expect e = hy; e.order = 0, e.flow = 1, e.side_max = 1.0; e.no_load = false; with_weights(w_low9, "hugegap", [&] { std::vector<uint64_t> v; for (uint64_t i = 0; i < 50000; ++i) v.push_back(i), v.push_back(M - i); return v; }(), "nfl_afli_cut_on", e); }
  }
  if (want("shipped")) {  // the weights the authors ship, on keys of the domain they were trained on and on raw 64-bit keys
    if (nfl_dir.empty()) {
      std::printf("SKIPPED the sets on the authors' shipped weights: pass --nfl <clone of github.com/luffy06/NFL> or set NFL_DIR\n");
    } else {
      const std::string books = nfl_dir + "/flow_weights/books-200M-100R-zipf_2D2H2L_weights.txt", fb = nfl_dir + "/flow_weights/fb-200M-100R-zipf_2D2H2L_weights.txt";
      Expect e = hy;
      e.order = 0, e.flow = 1;
      with_weights(books, "skewed15", skewed(N), "nfl_afli", e);
      with_weights(books, "skewed15", skewed(300000), "nfl_afli_cut", e);
      with_weights(fb, "skewed15", skewed(300000), "nfl_afli", e);
      Expect u = hy;
      u.side_max = 0.01;
      with_weights(books, "uniform15", rnd(0, 999999999999999ull, 300000), "nfl_afli", u);
      with_weights(fb, "uniform15", rnd(1, 900000000000000ull, 300000), "nfl_afli_on", u);
      Expect raw = hy;  // raw 64-bit keys: outside what the weights were trained on unless cut
      raw.side_max = 1.0, raw.order = 0, raw.flow = 1;
      with_weights(books, "wide_raw", rnd(0, M, 300000), "nfl_afli_on", raw);
      raw.side_max = 1e-3;
      with_weights(books, "wide_cut", rnd(0, M, 300000), "nfl_afli_cut_on", raw);
      with_weights(fb, "wide_cut", rnd(0, M, 300000), "nfl_afli4_cut_on", raw);
      { std::vector<uint64_t> v = rnd(0, uint64_t(1) << 36, 300000); for (int j = 0; j < 21; ++j) v.push_back((uint64_t(1) << 59) + (g() >> 2)); Expect f = hy; f.side_max = 0.05; with_weights(fb, "fblike", v, "nfl_afli", f); with_weights(fb, "fblike", v, "nfl_afli_cut_on", f); }
    }
  }
  std::error_code ec;
  fs::remove_all(tmp, ec);

  if (timing) {  // not a benchmark: one small set, to see what the wrapper costs over the authors' calls
    std::vector<uint64_t> keys = rnd(1, uint64_t(1) << 50, 3000000);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<std::pair<uint64_t, uint64_t>> kv, extra;
    for (uint64_t k : keys) kv.push_back({k, 123456789});
    for (uint64_t k : rnd(1, uint64_t(1) << 50, 500000)) extra.push_back({k, 123456789});
    auto iget = [](Index* i, uint64_t k, uint64_t& v) { return i->get(k, v); };
    auto iput = [](Index* i, uint64_t k, uint64_t v) { return i->put(k, v); };
    {
      ThinAfli thin;
      time_one("authors' AFLI, direct", &thin, kv, extra, [](ThinAfli* t, uint64_t k, uint64_t& v) { return t->get(k, v); }, [](ThinAfli* t, uint64_t k, uint64_t v) { return t->put(k, v); });
    }
    {
      ThinVirtual thin;
      time_one("authors' AFLI, virtual", static_cast<Index*>(&thin), kv, extra, iget, iput);
    }
    for (const char* nm : {"afli", "afli1", "afli6"}) {
      std::unique_ptr<Afli> d(dynamic_cast<Afli*>(gret::make_afli<uint64_t, uint64_t>(nm)));
      time_one(nm, static_cast<Index*>(d.get()), kv, extra, iget, iput);
    }
    {  // the same keys moved above 2^53: 24-byte slots and a key comparison on every hit
      std::vector<std::pair<uint64_t, uint64_t>> kv2 = kv, extra2 = extra;
      for (auto& e : kv2) e.first = (e.first << 13) | 1;
      for (auto& e : extra2) e.first = (e.first << 13) | 1;
      std::unique_ptr<Afli> d(dynamic_cast<Afli*>(gret::make_afli<uint64_t, uint64_t>("afli")));
      time_one("afli, keys to 2^63", static_cast<Index*>(d.get()), kv2, extra2, iget, iput);
    }
    if (!nfl_dir.empty()) {  // the flow, one key at a time
      setenv("TRANSFORM_NFL", (nfl_dir + "/flow_weights/books-200M-100R-zipf_2D2H2L_weights.txt").c_str(), 1);
      std::vector<uint64_t> k15 = skewed(3000000);
      std::sort(k15.begin(), k15.end());
      k15.erase(std::unique(k15.begin(), k15.end()), k15.end());
      std::vector<std::pair<uint64_t, uint64_t>> kv3, extra3;
      for (uint64_t k : k15) kv3.push_back({k, 123456789});
      for (uint64_t k : skewed(500000)) extra3.push_back({k, 123456789});
      for (const char* nm : {"afli", "nfl_afli", "nfl_afli_cut"}) {
        std::unique_ptr<Afli> d(dynamic_cast<Afli*>(gret::make_afli<uint64_t, uint64_t>(nm)));
        time_one((std::string(nm) + ", skewed15").c_str(), static_cast<Index*>(d.get()), kv3, extra3, iget, iput);
      }
      const gret::NflFlow f = [&] { gret::NflFlow x = gret::NflFlow::load(std::getenv("TRANSFORM_NFL")); x.fit(k15.size(), [&](size_t i) { return k15[i]; }); return x; }();
      uint64_t sum = 0;
      const auto t0 = std::chrono::steady_clock::now();
      for (uint64_t k : k15) sum += f(k);
      std::printf("time the flow alone (gret/nflflow.h, 2D2H2L, one key at a time): %.1f ns a key  [checksum %llu]\n",
                  std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(k15.size()), (unsigned long long)sum);
      unsetenv("TRANSFORM_NFL");
    }
#ifndef TEST_AFLI_ALONE
    {
      Param p(1, 0);
      auto gget = [&](Index* i, uint64_t k, uint64_t& v) { return i->get(k, v, &p); };
      auto gput = [&](Index* i, uint64_t k, uint64_t v) { return i->put(k, v, &p); };
      Index* a = new alexInterface<uint64_t, uint64_t>;
      a->init(&p);
      time_one("alex (GRE)", a, kv, extra, gget, gput);
      Index* l = new LIPPInterface<uint64_t, uint64_t>;
      l->init(&p);
      time_one("lipp (GRE)", l, kv, extra, gget, gput);
    }
#endif
  }
  std::printf(g_bad ? "test_afli: %zu FAILURES\n" : "test_afli: all passed\n", g_bad);
  return g_bad ? 1 : 0;
}
