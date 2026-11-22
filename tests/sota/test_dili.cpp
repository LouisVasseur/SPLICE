// SPDX-License-Identifier: MIT
// dili (include/gret/sota/dili.h: the DILI authors' code behind GRE's interface) against a sorted array and a std::map: get on
// every stored key and on keys never stored (neighbours, 0, 2^63 - 1, 2^63, 2^64 - 1, random), scan from stored, neighbouring,
// in-between and extreme start keys, then put (neighbours of stored keys, runs, keys far outside, duplicates, the payload
// 2^64 - 1), update and remove, the same checks after each, and finally every key removed. With NDEBUG (the -O3 build, or
// -DNDEBUG on the sanitized one) keys are also put after the removes and into the emptied index: with assert() on, such a put
// can stop at the authors' assert(num_nonempty > 1) (diliNode.h:877), which does not hold after deletions. The key sets are
// test_scan's patterns at 2.1M keys, because the authors' bulk loader reads out of bounds below about 2M keys: for the tiny sets
// (and one just under the limit) bulk_load must throw std::invalid_argument and leave an index that can be loaded again. One
// more set (denseleaf) has a leaf of 300,000 keys, far more than the authors' retrain buffers hold, and puts 300,000 keys into
// it: they must all go through the authors' insert, not to the side store.
//
// Build (docs/sota/dili.md; DILI = the clone of https://github.com/pfl-cs/DILI, B = build/tests/sota/dili):
//   macOS only: mkdir -p $B/macshim/linux $B/macshim/asm && : > $B/macshim/linux/perf_event.h && : > $B/macshim/asm/unistd.h
//   printf '[signed-integer-overflow]\nsrc:*/DILI/src/*\n[float-cast-overflow]\nsrc:*/DILI/src/*\n' > $B/ubsan_dili.ignore
//   build/tests/cxx    -fsanitize-ignorelist=$B/ubsan_dili.ignore -I $B/macshim -I $DILI/src tests/sota/test_dili.cpp -o $B/t
//   build/tests/cxx-O3 -I $B/macshim -I $DILI/src tests/sota/test_dili.cpp -o $B/t-O3
// The ignore list takes two UBSan checks off the authors' sources only (ASan and the other checks stay on them, and the wrapper
// stays fully checked): their BU-tree overflows signed longs on every key set here (butree/linearRegressor.cpp:175, 177) and
// their LR_PRED casts out-of-range doubles to int (global/global.h:8, from dili/DILI.h:254 and dili/diliNode.h:401, 413, 432,
// 434, 800, 890). Without the list the test stops at the first of these, linearRegressor.cpp:177:85. See docs/sota/dili.md.
//
// -DTEST_DILI_ALONE leaves GRE's own indexes out (they are here only to show the header lives next to them, and for --time).
//
//   t [SET...] [--time]     only the named sets; --time: a small lookup and insert timing on a 3M-key uniform set (use t-O3)
#include "indexInterface.h"
#ifndef TEST_DILI_ALONE  // GRE's own indexes first, as GRE's competitor.h has them: the new header must live next to them
#include "alex/alex.h"
#include "btree/btree.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#endif
#include "gret/sota/dili.h"

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

// GRE's get_index names its template parameters like this; DILI's sources define both as macros, and the header must not leak them
template <class KEY_TYPE, class PAYLOAD_TYPE>
static size_t names_are_free() { return sizeof(KEY_TYPE) + sizeof(PAYLOAD_TYPE); }

using Index = indexInterface<uint64_t, uint64_t>;
using Dili = gret::DiliIndex<uint64_t, uint64_t>;
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
  for (uint64_t k : {uint64_t(0), uint64_t(1), M, M - 1, H, H - 1, H + 1, H - 2}) probe(k);
  for (int i = 0; i < 20000; ++i) probe(g());
  for (int i = 0; i < 20000 && !all.empty(); ++i) {  // in the gaps and next to stored keys
    const size_t r = g() % all.size();
    probe(all[r].first + 1), probe(all[r].first - 1);
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) probe(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  std::vector<uint64_t> starts = {0, 1, M, M - 1, H, H - 1, H + 1};
  auto around = [&](uint64_t k) { starts.push_back(k), starts.push_back(k + 1), starts.push_back(k - 1); };
  for (size_t i = 0; i < std::min<size_t>(40, all.size()); ++i) around(all[i].first), around(all[all.size() - 1 - i].first);
  for (int i = 0; i < 1500 && !all.empty(); ++i) {
    const size_t r = g() % all.size();
    around(all[r].first);
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) starts.push_back(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  for (size_t i = 0; i < probes.size() && i < 600; ++i) around(probes[i]);
  for (int i = 0; i < 100; ++i) starts.push_back(g());
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
  bool builds = true;
  double side_max = 0;      // the largest share of the keys the side store may take at the bulk load
  double side_min = 0;      // and the smallest (sets made to hit the guards)
  bool empty_it = true;     // remove every key at the end
};

static void check(const std::string& label, std::vector<uint64_t> keys, const char* name, const Expect& ex) {
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  const std::string what = std::string(name) + " on " + label;
  const size_t bad0 = g_bad;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::pair<uint64_t, uint64_t>> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  if (label == "wide" || label == "fblike") kv[kv.size() / 3].second = M, kv[2 * kv.size() / 3].second = M;  // the authors' "not found" as a payload
  Index* idx = gret::make_dili<uint64_t, uint64_t>(name);
  if (!idx) { std::printf("FAIL %s: no such index\n", name); ++g_bad; return; }
  std::unique_ptr<Dili> owner(dynamic_cast<Dili*>(idx));  // deleted as what it is: indexInterface has no virtual destructor
  Dili* d = owner.get();
  Param p(1, 0);
  idx->init(&p);
  if (!ex.builds) {
    for (int twice = 0; twice < 2; ++twice) {  // throws, and can be asked again
      bool threw = false;
      try {
        idx->bulk_load(kv.data(), kv.size(), &p);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (!threw) fail(what, "bulk_load did not throw std::invalid_argument, keys", keys.size());
      if (d->in_dili() || d->in_side() || d->half_built(0) || d->half_built(1)) fail(what, "bulk_load threw but kept keys:", d->in_dili() + d->in_side());
    }
    std::printf("%-9s %-14s n=%-8zu throws, as it must\n", name, label.c_str(), keys.size());
    return;
  }
  try {
    idx->bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::exception& e) {
    std::printf("FAIL %s: bulk_load threw %s\n", what.c_str(), e.what());
    ++g_bad;
    return;
  }
  const size_t side0 = d->in_side(), in0 = d->in_dili();
  if (in0 + side0 != keys.size()) fail(what, "bulk_load holds another number of keys than", keys.size());
  if (double(side0) > ex.side_max * double(keys.size()) + 8 || double(side0) < ex.side_min * double(keys.size()))
    fail(what, "the side store's share at the bulk load is outside what this set should give; side keys:", side0);
  const long long bytes = idx->memory_consumption();
  if (bytes < (long long)(16 * keys.size())) fail(what, "memory_consumption below 16 bytes a key:", uint64_t(bytes));

  std::mt19937_64 g(std::hash<std::string>()(label) ^ 0x9e3779b97f4a7c15ull);
  Map ref(kv.begin(), kv.end());
  std::vector<uint64_t> probes;
  verify(*idx, ref, probes, g, what + " (loaded)");

  auto pick = [&]() { return keys[g() % keys.size()]; };
  // put: neighbours of stored keys, runs behind a stored key, keys anywhere, keys far outside, special keys, duplicates, and
  // the payload 2^64 - 1
  {
    std::vector<std::pair<uint64_t, uint64_t>> add;
    for (int i = 0; i < 20000; ++i) { const uint64_t k = pick(); add.push_back({k + 1, payload(k + 1)}); if (i % 3 == 0) add.push_back({k + 2, payload(k + 2)}); if (i % 7 == 0) add.push_back({k - 1, payload(k - 1)}); }
    for (int r = 0; r < 4; ++r) { const uint64_t k = pick(); for (uint64_t j = 1; j <= 3000; ++j) add.push_back({k + j * (r + 1), payload(k + j)}); }
    for (int i = 0; i < 20000; ++i) { const uint64_t span = keys.back() - keys.front(), k = keys.front() + (span == M ? g() : g() % (span + 1)); add.push_back({k, payload(k)}); }
    for (int i = 0; i < 3000; ++i) { const uint64_t k = g(); add.push_back({k, payload(k)}); }
    for (uint64_t j = 0; j < 40; ++j) {
      add.push_back({keys.back() + 1 + j, j}), add.push_back({keys.back() + (uint64_t(1) << (j + 4)), j}), add.push_back({keys.front() - 1 - j, j});
      add.push_back({keys.front() >> (j % 20 + 1), j}), add.push_back({keys.back() + (keys.back() - keys.front()) * (j + 1), j});
    }
    for (uint64_t k : {uint64_t(0), uint64_t(1), M, M - 1, H, H - 1, H + 1, H - 2}) add.push_back({k, payload(k)});
    for (int i = 0; i < 300; ++i) add.push_back({pick() + 3, M});  // the authors' "not found" value
    for (int i = 0; i < 2000; ++i) add.push_back({pick(), 99});   // mostly there already
    std::shuffle(add.begin(), add.end(), g);
    for (const auto& a : add) {
      const bool fresh = ref.emplace(a.first, a.second).second;
      if (idx->put(a.first, a.second) != fresh) fail(what, "put", a.first);
      probes.push_back(a.first);
    }
    if (d->in_dili() + d->in_side() != ref.size()) fail(what, "after put the index counts another number of keys than", ref.size());
    verify(*idx, ref, probes, g, what + " (put)");
  }
  // update: present keys (some to 2^64 - 1 and back), absent keys return false
  {
    for (int i = 0; i < 30000; ++i) {
      uint64_t k = i % 4 ? pick() : probes[g() % probes.size()];
      if (i % 9 == 0) k += 1;
      const uint64_t v = i % 50 == 0 ? M : g();
      const auto it = ref.find(k);
      if (idx->update(k, v) != (it != ref.end())) fail(what, "update", k);
      if (it != ref.end()) it->second = v;
      if (i % 100 == 0 && it != ref.end()) {  // and back to an ordinary payload
        if (!idx->update(k, v ^ 5)) fail(what, "second update", k);
        it->second = v ^ 5;
      }
    }
    verify(*idx, ref, probes, g, what + " (updated)");
  }
  // remove: random keys (loaded ones and put ones) and both ends; absent keys and a second remove return false
  {
    std::vector<uint64_t> gone;
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
    verify(*idx, ref, probes, g, what + " (removed)");
#ifdef NDEBUG  // a put after a remove can stop at the authors' assert(num_nonempty > 1), diliNode.h:877, which no longer holds then
    for (size_t i = 0; i < gone.size(); i += 2) {
      const uint64_t k = gone[i] + (i % 4 == 0);
      if (idx->put(k, payload(k) ^ 1) != ref.emplace(k, payload(k) ^ 1).second) fail(what, "put (after remove)", k);
      probes.push_back(k);
    }
    verify(*idx, ref, probes, g, what + " (put after remove)");
#endif
  }
  const size_t side1 = d->in_side(), in1 = d->in_dili();
  if (ex.empty_it) {  // every key out, in a random order; then an index that holds nothing, and takes keys again
    std::vector<uint64_t> order;
    for (const auto& kvp : ref) order.push_back(kvp.first);
    std::shuffle(order.begin(), order.end(), g);
    for (uint64_t k : order)
      if (!idx->remove(k)) fail(what, "remove (emptying)", k);
    ref.clear();
    if (d->in_dili() || d->in_side()) fail(what, "emptied, but the index still counts keys:", d->in_dili() + d->in_side());
    probes.resize(std::min<size_t>(probes.size(), 20000));
    for (size_t i = 0; i < 20000; ++i) probes.push_back(order[i % order.size()]);
    verify(*idx, ref, probes, g, what + " (emptied)");
#ifdef NDEBUG  // as above
    for (size_t i = 0; i < 30000 && i < order.size(); ++i) {
      const uint64_t k = order[i] + (i % 2);
      if (idx->put(k, k ^ 7) != ref.emplace(k, k ^ 7).second) fail(what, "put (into the emptied index)", k);
    }
    verify(*idx, ref, probes, g, what + " (refilled)");
#endif
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%-9s %-14s n=%-8zu loaded: dili=%zu side=%zu (%.3f%%) %.1f B/key; after writes: dili=%zu side=%zu  %s  %.1fs\n", name, label.c_str(), keys.size(), in0, side0,
              100.0 * double(side0) / double(keys.size()), double(bytes) / double(keys.size()), in1, side1, g_bad == bad0 ? "ok" : "FAILED", s);
  std::fflush(stdout);
}

template <class I>
static void time_one(const char* name, I* idx, const std::vector<std::pair<uint64_t, uint64_t>>& kv, const std::vector<std::pair<uint64_t, uint64_t>>& extra) {
  Param p(1, 0);
  idx->init(&p);
  std::vector<std::pair<uint64_t, uint64_t>> copy = kv;
  auto t0 = std::chrono::steady_clock::now();
  idx->bulk_load(copy.data(), copy.size(), &p);
  const double build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::mt19937_64 g(3);
  std::vector<uint64_t> q(2000000);
  for (uint64_t& x : q) x = kv[g() % kv.size()].first;
  uint64_t sum = 0, v = 0;
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    t0 = std::chrono::steady_clock::now();
    for (uint64_t x : q) sum += idx->get(x, v, &p) ? v : 1;
    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(q.size()));
  }
  t0 = std::chrono::steady_clock::now();
  size_t in = 0;
  for (const auto& e : extra) in += idx->put(e.first, e.second, &p);
  const double put = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(extra.size());
  std::printf("time %-8s n=%zu build %.2fs  get %.1f ns  put %.1f ns (%zu new)  %.1f B/key  [checksum %llu]\n", name, kv.size(), build, best, put, in,
              double(idx->memory_consumption()) / double(kv.size() + in), (unsigned long long)sum);
  delete idx;
}

int main(int argc, char** argv) {
  std::vector<std::string> only;
  bool timing = false;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--time")) timing = true;
    else only.push_back(argv[i]);
  auto want = [&](const std::string& s) { return only.empty() || std::find(only.begin(), only.end(), s) != only.end(); };
  if (names_are_free<uint64_t, uint64_t>() != 16) return 2;
  if (gret::make_dili<uint64_t, uint64_t>("dil") || gret::make_dili<uint64_t, uint64_t>("dili1") || gret::make_dili<uint64_t, uint64_t>("dili016") ||
      gret::make_dili<uint64_t, uint64_t>("dili201") || gret::make_dili<uint64_t, uint64_t>("dili_x")) {
    std::printf("FAIL make_dili accepts a malformed name\n");
    ++g_bad;
  }
  for (const char* bad : {"dilir", "dili16r", "dilir0", "dilir100", "dili16r05", "dilirr2", "dili16r2r3", "dili16x20", "dili1r20", "dilir2 "})
    if (gret::make_dili<uint64_t, uint64_t>(bad)) {
      std::printf("FAIL make_dili accepts the malformed name '%s'\n", bad);
      ++g_bad;
    }
  for (const char* good : {"dili", "dili2", "dili200", "dilir20", "dili32r5", "dili16r99"}) {
    std::unique_ptr<Dili> made(dynamic_cast<Dili*>(gret::make_dili<uint64_t, uint64_t>(good)));
    if (!made) {
      std::printf("FAIL make_dili refuses the name '%s'\n", good);
      ++g_bad;
    }
  }
  const size_t N = 2100000;
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; v.reserve(m); std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  Expect none; none.builds = false;
  Expect all;                       // everything in the authors' structure but for a handful
  all.side_max = 1e-5;

  // too few keys for the authors' bulk loader: it must refuse them
  if (want("tiny")) {
    check("n1", {5}, "dili", none);
    check("n1max", {M}, "dili", none);
    check("n1zero", {0}, "dili", none);
    check("n2", {0, M}, "dili", none);
    check("n3", {1, 2, M}, "dili", none);
    check("n5", {0, 1, 2, 3, M}, "dili", none);
    check("n17", rnd(0, uint64_t(1) << 40, 17), "dili", none);
    check("uniform20k", rnd(1, uint64_t(1) << 50, 20000), "dili", none);
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < 1999999; ++i) v.push_back(i * 3); check("under_limit", v, "dili", none); }
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < 1500000; ++i) v.push_back(i * 3), v.push_back(H + i * 5); check("two_small_halves", v, "dili", none); }
  }
  if (want("uniform")) check("uniform", rnd(1, uint64_t(1) << 50, N), "dili", all);
  if (want("uniform8")) check("uniform", rnd(1, uint64_t(1) << 50, N), "dili8", all);     // the knob: buMinFan
  if (want("uniform64")) check("uniform", rnd(1, uint64_t(1) << 50, N), "dili64", all);
  if (want("uniformr20")) check("uniform", rnd(1, uint64_t(1) << 50, N), "dilir20", all);  // rho as the paper's section 7.1 states it (the released code has 0.1)
  if (want("wide")) {  // both halves of the key space, with their ends
    std::vector<uint64_t> v = rnd(0, M, 2 * N + 60000);
    for (uint64_t k : {uint64_t(0), M, H, H - 1}) v.push_back(k);
    check("wide", v, "dili", all);
  }
  if (want("run")) { std::vector<uint64_t> v; for (uint64_t i = 0; i < N; ++i) v.push_back(1000 + i); check("run", v, "dili", all); }
  if (want("clusters")) {
    std::vector<uint64_t> v;
    for (int c = 0; c < 21000; ++c) { const uint64_t b = g() >> 16; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); }
    check("clusters", v, "dili", all);
  }
  if (want("outliers")) {  // a narrow bulk and keys far from it: the int casts of the authors' models
    std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), N);
    for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003);
    v.push_back(3), v.push_back(M);
    Expect e = all; e.side_max = 1e-3;
    check("outliers", v, "dili", e);
  }
  if (want("stride")) { std::vector<uint64_t> v; for (uint64_t i = 0; i < N; ++i) v.push_back(i * 65536 + 12345); check("stride", v, "dili", all); }
  if (want("hugegap")) {  // a run from 0 and a run down from 2^64 - 1: the upper run is closer than doubles separate up there
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < N; ++i) v.push_back(i), v.push_back(M - i);
    Expect e; e.side_min = 0.49, e.side_max = 0.51;
    check("hugegap", v, "dili", e);
  }
  if (want("highmag")) {  // large keys far enough apart, with pairs, triples and a run too close for the authors' models
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < N; ++i) v.push_back((uint64_t(1) << 61) + (i << 20) + (g() & 0x7ffff));
    for (int i = 0; i < 30000; ++i) { const uint64_t k = v[g() % N]; v.push_back(k + 1); if (i & 1) v.push_back(k + 2); if (i % 5 == 0) v.push_back(k + 700); }
    for (uint64_t i = 0; i < 500; ++i) v.push_back((uint64_t(1) << 62) + (uint64_t(1) << 50) + i);
    Expect e; e.side_min = 0.02, e.side_max = 0.04;
    check("highmag", v, "dili", e);
  }
  if (want("fblike")) {  // small keys, a few very large ones on both sides of 2^63
    std::vector<uint64_t> v = rnd(0, uint64_t(1) << 36, N);
    for (int j = 0; j < 21; ++j) v.push_back((uint64_t(1) << 59) + (g() >> 2));
    for (int j = 0; j < 10; ++j) v.push_back(H + (g() >> 1));
    Expect e = all; e.side_max = 1e-3;
    check("fblike", v, "dili", e);
  }
  if (want("denseleaf")) {  // a leaf far larger than the authors' retrain buffers (2 x 8192 keys): puts into it must reach the authors' insert
    std::vector<uint64_t> v = rnd(0, uint64_t(1) << 48, N);
    const uint64_t b = uint64_t(1) << 47;
    for (uint64_t i = 0; i < 300000; ++i) v.push_back(b + 2 * i);  // 300,000 keys in one equal-width slice of their parent's range
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    std::vector<std::pair<uint64_t, uint64_t>> kv;
    for (uint64_t k : v) kv.push_back({k, payload(k)});
    const size_t bad0 = g_bad;
    std::unique_ptr<Dili> d(new Dili("dili"));
    Index* idx = d.get();
    idx->bulk_load(kv.data(), kv.size());
    Map ref(kv.begin(), kv.end());
    std::vector<uint64_t> odd;
    for (uint64_t i = 0; i < 300000; ++i) odd.push_back(b + 2 * i + 1);
    std::shuffle(odd.begin(), odd.end(), g);
    for (uint64_t k : odd)
      if (idx->put(k, payload(k)) != ref.emplace(k, payload(k)).second) fail("dili on denseleaf", "put", k);
    const size_t to_side = d->puts_to_side();
    if (to_side > 4) fail("dili on denseleaf", "puts into a large leaf went to the side store instead of the authors' insert:", to_side);
    verify(*idx, ref, odd, g, "dili on denseleaf (put)");
    for (size_t i = 0; i < odd.size(); i += 2)
      if (!idx->remove(odd[i]) || !ref.erase(odd[i])) fail("dili on denseleaf", "remove", odd[i]);
    for (uint64_t i = 0; i < 300000; i += 3)
      if (idx->remove(b + 2 * i) != (ref.erase(b + 2 * i) != 0)) fail("dili on denseleaf", "remove (loaded key)", b + 2 * i);
    verify(*idx, ref, odd, g, "dili on denseleaf (removed)");
    std::printf("%-9s %-14s n=%-8zu 300000 puts into one large leaf, %zu to the side store; dili=%zu side=%zu  %s\n", "dili", "denseleaf", v.size(), to_side, d->in_dili(), d->in_side(),
                g_bad == bad0 ? "ok" : "FAILED");
    std::fflush(stdout);
  }
  if (timing) {  // not a benchmark: one small set, to see the wrapper costs nothing odd
    const char* bits = std::getenv("TEST_DILI_TIME_BITS");  // keys uniform below 2^bits (default 50; from 51 the insert guard looks at neighbours)
    const uint64_t top = uint64_t(1) << (bits ? std::min(62, std::max(30, std::atoi(bits))) : 50);
    std::vector<uint64_t> keys = rnd(1, top, 3000000);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<std::pair<uint64_t, uint64_t>> kv, extra;
    for (uint64_t k : keys) kv.push_back({k, 123456789});
    for (uint64_t k : rnd(1, top, 500000)) extra.push_back({k, 123456789});
    time_one("dili", new Dili("dili"), kv, extra);
#ifndef TEST_DILI_ALONE
    time_one("alex", new alexInterface<uint64_t, uint64_t>, kv, extra);
    time_one("lipp", new LIPPInterface<uint64_t, uint64_t>, kv, extra);
    time_one("btree", new BTreeInterface<uint64_t, uint64_t>, kv, extra);
#endif
  }
  std::printf(g_bad ? "test_dili: %zu FAILURES\n" : "test_dili: all passed\n", g_bad);
  return g_bad ? 1 : 0;
}
