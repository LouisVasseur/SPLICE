// SPDX-License-Identifier: MIT
// sali and saliu (include/gret/sota/sali.h: the SALI authors' code behind GRE's interface) against a sorted array and a std::map.
//   loaded:  get on every stored key and on keys never stored (both neighbours of stored keys, 0, 1, 2^63 - 1, 2^63, 2^64 - 2,
//            2^64 - 1, below and above the range, in the gaps, random), scan from stored, neighbouring, in-between and extreme
//            start keys at lengths 0 to 70000 and once over everything; the side store holds exactly the keys the header's rule
//            sends there; memory_consumption is the sum of its parts.
//   writes:  put (neighbours of stored keys, runs, keys anywhere, keys far outside, special keys, keys already there), update,
//            remove (absent keys and a second remove return false), put again, every key removed, the emptied index refilled,
//            with the same checks after each step. saliu is never given a key it holds (that is its contract).
//   empty:   an index that was never loaded, and one loaded with no keys, take puts and answer.
//   guards:  bulk_load of unsorted or repeated keys throws std::invalid_argument and leaves the index as it was; saliu given a
//            stored key throws std::runtime_error (the authors' check at sali.h:1618), every later call throws std::logic_error,
//            and a new bulk_load makes it usable again; make_sali refuses malformed names.
//   memory:  (glibc only) memory_consumption against what malloc itself reports in use, after the bulk load and after 200,000
//            puts, and for an index grown from nothing by 200,000 puts: it must be between 85% and 101% of it.
//   cycles:  a remove and a put in turn, at a constant number of keys. This is the workload in which the authors' rebuild
//            inside an insert divides 0 by 0 (sali.h:3197) and asks for NaN slots (sali.h:2361): their own build ends there.
//            The wrapper reads the keys back with their scan and bulk loads a new object (salvage() in the header); every
//            answer must stay right, and the line printed says how often that happened (it depends on the clock). UBSan's
//            check of that one cast is off in this test (below, at the include); -DTEST_SALI_STRICT_UBSAN leaves it on.
// Key sets as in tests/test_scan.cpp (tiny sets, uniform, wide, a dense run, clusters, far outliers, 0 and 2^64 - 1, a regular
// stride, a huge gap) plus two of 2M keys (uniform below 2^50, and over all 64 bits) and five made for the side store's rule
// (pairs, triples and runs 1 apart next to 2^63 and 2^64, and "tight": runs at the smallest gaps the rule lets into SALI).
//
//   t [SET...] [--full] [--time] [--probe]
//     SET      only the named sets
//     --full   four times the probes, scans and writes per set, and saliu on every set (without it: on six of them)
//     --time   (use t-O3) a small timing on 3M uniform keys: get through GRE's interface, the authors' at() and exists() called
//              directly, a get that misses, put on sali and saliu, and GRE's LIPP, which SALI is built on, for scale
//     --probe  (Linux, t-O3) what the authors' code does WITHOUT the side store's rule on runs of close keys from 2^60 to 2^64
//              (three shapes, bulk loaded or half put), each case in a child process under a memory and a CPU limit, and
//              which rule would have been wide enough. This is the evidence for the rule and for its width.
//
// Build (docs/sota/sali.md; SALI = the clone of https://github.com/cds-ruc/SALI, B = build/tests/sota/sali):
//   macOS only (no TBB, and arm64 has no <immintrin.h>): four empty files, the header then supplies stand-ins
//     mkdir -p $B/macshim/tbb && : > $B/macshim/tbb/combinable.h && : > $B/macshim/tbb/enumerable_thread_specific.h &&
//       : > $B/macshim/tbb/spin_mutex.h && : > $B/macshim/immintrin.h
//   build/tests/cxx    -I $B/macshim -I $SALI/src/core tests/sota/test_sali.cpp -o $B/t
//   build/tests/cxx-O3 -I $B/macshim -I $SALI/src/core tests/sota/test_sali.cpp -o $B/t-O3
//   x86-64 on the Mac (Rosetta; long double then has the benchmark machine's 64 bits of mantissa): the same two commands
//     spelled out with -arch x86_64 -mavx2 in place of -mcpu=native, -DTEST_SALI_ALONE, and a shim directory that holds the
//     three empty tbb/ files but no immintrin.h
//   Linux (the cluster; real TBB, no shim): /scratch/lvasseur/wf/wf_cxx sali -isystem $SALI/src/core test_sali.cpp -o t
// Under AddressSanitizer with libc++ this test turns the container-overflow check off (the function at the end of this file):
// the authors' bulk_load writes its copy of the keys through operator[] of two vectors it has reserved but not resized
// (sali.h:734-741) and builds from there (sali.h:2332). Every key of every bulk load goes through that memory. The memory is
// allocated and large enough, so nothing is read or written out of bounds, but libc++ marks the part of a vector beyond size()
// as off limits and the check stops at sali.h:739. -DTEST_SALI_STRICT_ASAN leaves the check on, to see it.
//
// -DTEST_SALI_ALONE leaves GRE's own indexes out (they are here to show the header lives next to them, and for --time).
#include "indexInterface.h"
#ifndef TEST_SALI_ALONE  // GRE's own indexes first, as GRE's competitor.h has them: the new header must live next to them
#include "alex/alex.h"
#include "btree/btree.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#endif
// One UBSan check is off in the authors' functions (and in the wrapper's, which convert no floating point number to an
// integer): the conversion of NaN to int at the authors' sali.h:2361, which is their bug in a rebuild after removes (the
// "cycles" part below, and docs/sota/sali.md). With the check on, the sanitized run stops there, before the wrapper's repair
// can be tested under the sanitizers; -DTEST_SALI_STRICT_UBSAN leaves it on, to see the report. On x86-64 that conversion
// gives INT_MIN and the next statement of theirs to use it computes INT_MIN - 2 (sali.h:2367): there, and only there, the
// check of signed overflow is off as well (the wrapper's own arithmetic is unsigned).
#if defined(__clang__) && !defined(TEST_SALI_STRICT_UBSAN)
#if defined(__x86_64__)
#pragma clang attribute push(__attribute__((no_sanitize("float-cast-overflow", "signed-integer-overflow"))), apply_to = function)
#else
#pragma clang attribute push(__attribute__((no_sanitize("float-cast-overflow"))), apply_to = function)
#endif
#endif
#include "gret/sota/sali.h"
#if defined(__clang__) && !defined(TEST_SALI_STRICT_UBSAN)
#pragma clang attribute pop
#endif

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

#if defined(__linux__)
#include <malloc.h>
#include <sys/resource.h>
#include <sys/wait.h>
#endif

// the authors' header defines these as macros (and RT_ASSERT and COLLECT_TIME, which GRE's LIPP defines too); the wrapper
// must not leak them into a build that has its own
#if defined(READ_EVOLVE) || defined(COMPRESS) || defined(PRINT_DEBUG) || defined(RT_DEBUG) || defined(likely) || defined(unlikely) || defined(exit) || defined(fprintf)
#error "gret/sota/sali.h leaks a macro of the SALI sources"
#endif

using Index = indexInterface<uint64_t, uint64_t>;
using Sali = gret::SaliIndex<uint64_t, uint64_t>;
using Map = std::map<uint64_t, uint64_t>;
using KV = std::pair<uint64_t, uint64_t>;
static const uint64_t M = ~uint64_t(0);
static const uint64_t H = uint64_t(1) << 63;

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

static size_t g_bad = 0;
static void fail(const std::string& what, const char* op, uint64_t k) {
  if (g_bad++ < 40) std::printf("FAIL %s: %s wrong at key %llu\n", what.c_str(), op, (unsigned long long)k);
}

// the header's rule, restated: a key closer to the key kept below it than (key >> shift) + 1 goes to the side store
static uint64_t margin(uint64_t k) { return Sali::kShift >= 64 ? 1 : (k >> Sali::kShift) + 1; }
static size_t side_share(const std::vector<uint64_t>& sorted) {
  size_t side = 0;
  uint64_t below = 0;
  for (size_t i = 0; i < sorted.size(); ++i)
    if (i && sorted[i] - below < margin(sorted[i])) ++side;
    else below = sorted[i];
  return side;
}

// How much is asked: 1 by default (the whole run is then about a minute of one core at -O3; the cluster's login node gives
// all of a user's processes one core between them), 4 with --full. A get that misses and lands on another key costs a C++
// throw (microseconds; far more under a sanitizer), so each kind of probe is made kProbes times a pass at most.
static size_t g_scale = 1;
#define kProbes (size_t(1000) * g_scale)

// every answer against the reference: get on every key it holds and on probes, scans from starts
static void verify(Index& idx, const Map& ref, const std::vector<uint64_t>& probes, std::mt19937_64& g, const std::string& what) {
  std::vector<KV> all(ref.begin(), ref.end());
  const uint64_t canary = 0xDEADBEEFCAFEF00Dull;
  for (const auto& kv : all) {
    uint64_t v = canary;
    if (!idx.get(kv.first, v) || v != kv.second) fail(what, "get", kv.first);
  }
  auto probe = [&](uint64_t k) {
    uint64_t val = canary;
    const bool got = idx.get(k, val);
    const auto it = ref.find(k);
    if (got != (it != ref.end()) || (got && val != it->second) || (!got && val != canary)) fail(what, "get (probe)", k);
  };
  // every stored key's neighbours on small sets, a sample on large ones (a miss that lands on another key costs a C++ throw)
  const size_t step = std::max<size_t>(1, all.size() / kProbes);
  for (size_t i = 0; i < all.size(); i += step) probe(all[i].first + 1), probe(all[i].first - 1);
  for (size_t i = 0, n = std::min<size_t>(probes.size(), kProbes); i < n; ++i) {
    const uint64_t k = probes[probes.size() <= kProbes ? i : g() % probes.size()];
    probe(k), probe(k + 1), probe(k - 1);
  }
  for (uint64_t k : {uint64_t(0), uint64_t(1), uint64_t(2), M, M - 1, M - 2, H, H - 1, H + 1, H - 2}) probe(k);
  if (!all.empty()) {
    const uint64_t lo = all.front().first, hi = all.back().first;
    for (uint64_t d : {uint64_t(1), uint64_t(2), uint64_t(1000), uint64_t(1) << 32}) {
      if (lo >= d) probe(lo - d);
      if (M - hi >= d) probe(hi + d);
    }
    probe(lo / 2), probe(hi + (M - hi) / 2);
  }
  for (size_t i = 0; i < kProbes; ++i) probe(g());
  for (size_t i = 0; i < kProbes && !all.empty(); ++i) {  // in the gaps
    const size_t r = g() % all.size();
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) probe(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  std::vector<uint64_t> starts = {0, 1, M, M - 1, H, H - 1, H + 1};
  auto around = [&](uint64_t k) { starts.push_back(k), starts.push_back(k + 1), starts.push_back(k - 1); };
  for (size_t i = 0; i < std::min<size_t>(40, all.size()); ++i) around(all[i].first), around(all[all.size() - 1 - i].first);
  for (size_t i = 0; i < 300 * g_scale && !all.empty(); ++i) {
    const size_t r = g() % all.size();
    around(all[r].first);
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) starts.push_back(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  for (size_t i = 0; i < probes.size() && i < 100 * g_scale; ++i) around(probes[probes.size() - 1 - i]);
  for (int i = 0; i < 100; ++i) starts.push_back(g());
  const size_t lens[] = {0, 1, 2, 3, 16, 256, 5000, 70000};
  std::vector<KV> out;
  auto less = [](const KV& a, uint64_t b) { return a.first < b; };
  for (uint64_t s : starts)
    for (size_t len : lens) {
      if (len > 256 && (g() & 15)) continue;  // long scans on a sample of the starts
      out.assign(len + 1, KV(canary, canary));
      const size_t got = idx.scan(s, len, out.data());
      const size_t at = size_t(std::lower_bound(all.begin(), all.end(), s, less) - all.begin());
      bool ok = got == std::min(len, all.size() - at);
      for (size_t i = 0; ok && i < got; ++i) ok = out[i] == all[at + i];
      for (size_t i = got; ok && i <= len; ++i) ok = out[i] == KV(canary, canary);  // nothing written past what it returned
      if (!ok) fail(what, "scan", s);
    }
  {  // one scan over everything
    out.assign(all.size() + 7, KV(canary, canary));
    const size_t got = idx.scan(0, all.size() + 7, out.data());
    bool ok = got == all.size();
    for (size_t i = 0; ok && i < got; ++i) ok = out[i] == all[i];
    if (!ok) fail(what, "full scan", 0);
  }
}

static void check_memory(Sali& d, const std::string& what) {
  const long long mem = static_cast<Index&>(d).memory_consumption();
  const size_t parts = sizeof(Sali) + d.authors_total_size() + d.pool_bytes() + d.waiting_bytes() + d.in_side() * 48 + d.set_aside_bytes();
  if (mem != (long long)parts) fail(what, "memory_consumption is not the sum of its parts:", uint64_t(mem));
  if (size_t(mem) < d.in_sali() * 2 * 16 + d.in_side() * 16) fail(what, "memory_consumption below one 32-byte slot a key:", uint64_t(mem));
}

static void check(const std::string& label, std::vector<uint64_t> keys, const char* name, bool empty_it = true) {
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  const std::string what = std::string(name) + " on " + label;
  const bool unchecked = std::string(name) == "saliu";
  const size_t bad0 = g_bad;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<KV> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  if (kv.size() > 2) kv[kv.size() / 3].second = M, kv[2 * kv.size() / 3].second = 0;  // no payload is special
  Index* idx = gret::make_sali<uint64_t, uint64_t>(name);
  if (!idx) { std::printf("FAIL %s: no such index\n", name); ++g_bad; return; }
  std::unique_ptr<Sali> owner(dynamic_cast<Sali*>(idx));  // deleted as what it is: indexInterface has no virtual destructor
  Sali* d = owner.get();
  if (!d) { std::printf("FAIL %s: not a SaliIndex\n", name); ++g_bad; return; }
  Param p(1, 0);
  idx->init(&p);
  try {
    idx->bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::exception& e) {
    std::printf("FAIL %s: bulk_load threw %s\n", what.c_str(), e.what());
    ++g_bad;
    return;
  }
  const size_t side0 = d->in_side(), in0 = d->in_sali();
  if (in0 + side0 != keys.size()) fail(what, "bulk_load holds another number of keys than", keys.size());
  if (side0 != side_share(keys)) fail(what, "the side store does not hold the keys the rule sends there; it holds", side0);
  const long long bytes = idx->memory_consumption();
  check_memory(*d, what + " (loaded)");

  std::string phases;
  auto mark = [&, last = t0](const char* ph) mutable {
    const auto now = std::chrono::steady_clock::now();
    char buf[64];
    std::snprintf(buf, sizeof buf, "%s %.1f  ", ph, std::chrono::duration<double>(now - last).count());
    phases += buf, last = now;
  };
  mark("load");
  std::mt19937_64 g(std::hash<std::string>()(label) ^ 0x9e3779b97f4a7c15ull);
  Map ref(kv.begin(), kv.end());
  std::vector<uint64_t> probes;
  verify(*idx, ref, probes, g, what + " (loaded)"); mark("verify");

  auto pick = [&]() { return keys[g() % keys.size()]; };
  auto put = [&](uint64_t k, uint64_t v, const char* op) {
    const bool fresh = ref.find(k) == ref.end();
    if (unchecked && !fresh) return;  // saliu's contract: the caller does not put a key that is stored
    if (fresh) ref.emplace(k, v);
    if (idx->put(k, v) != fresh) fail(what, op, k);
    probes.push_back(k);
  };
  if (!keys.empty()) {
    // put: neighbours of stored keys, runs behind a stored key, keys anywhere, keys far outside, special keys, duplicates
    {
      std::vector<KV> add;
      const size_t W = 5000 * g_scale;
      for (size_t i = 0; i < W; ++i) { const uint64_t k = pick(); add.push_back({k + 1, payload(k + 1)}); if (i % 3 == 0) add.push_back({k + 2, payload(k + 2)}); if (i % 7 == 0) add.push_back({k - 1, payload(k - 1)}); }
      for (int r = 0; r < 4; ++r) { const uint64_t k = pick(); for (uint64_t j = 1; j <= W / 5; ++j) add.push_back({k + j * (r + 1), payload(k + j)}); }
      for (size_t i = 0; i < W; ++i) { const uint64_t span = keys.back() - keys.front(), k = keys.front() + (span == M ? g() : g() % (span + 1)); add.push_back({k, payload(k)}); }
      for (size_t i = 0; i < W / 5; ++i) { const uint64_t k = g(); add.push_back({k, payload(k)}); }
      for (uint64_t j = 0; j < 40; ++j) {
        add.push_back({keys.back() + 1 + j, j}), add.push_back({keys.back() + (uint64_t(1) << (j + 4)), j}), add.push_back({keys.front() - 1 - j, j});
        add.push_back({keys.front() >> (j % 20 + 1), j}), add.push_back({keys.back() + (keys.back() - keys.front()) * (j + 1), j});
      }
      for (uint64_t k : {uint64_t(0), uint64_t(1), M, M - 1, H, H - 1, H + 1, H - 2}) add.push_back({k, payload(k)});
      for (int i = 0; i < 300; ++i) add.push_back({pick() + 3, i & 1 ? M : 0});
      for (size_t i = 0; i < W / 5; ++i) add.push_back({pick(), 99});  // mostly there already
      std::shuffle(add.begin(), add.end(), g);
      for (const auto& a : add) put(a.first, a.second, "put");
      if (d->in_sali() + d->in_side() != ref.size()) fail(what, "after put the index counts another number of keys than", ref.size());
      verify(*idx, ref, probes, g, what + " (put)"); mark("put");
      check_memory(*d, what + " (put)");
    }
    // update: present keys, absent keys return false and change nothing
    {
      for (size_t i = 0; i < 8000 * g_scale; ++i) {
        uint64_t k = i % 4 ? pick() : probes[g() % probes.size()];
        if (i % 9 == 0) k += 1;
        const uint64_t v = i % 50 == 0 ? M : g();
        const auto it = ref.find(k);
        if (idx->update(k, v) != (it != ref.end())) fail(what, "update", k);
        if (it != ref.end()) it->second = v;
      }
      verify(*idx, ref, probes, g, what + " (updated)"); mark("update");
    }
    // remove: random keys (loaded ones and put ones) and both ends; absent keys and a second remove return false
    {
      std::vector<uint64_t> gone;
      for (size_t i = 0; i < 8000 * g_scale; ++i) gone.push_back(i % 3 ? pick() : probes[g() % probes.size()]);
      for (size_t i = 0; i < 60 && i < keys.size(); ++i) gone.push_back(keys[i]), gone.push_back(keys[keys.size() - 1 - i]);
      for (uint64_t k : gone) {
        const bool there = ref.erase(k) != 0;
        if (idx->remove(k) != there) fail(what, "remove", k);
        if (idx->remove(k)) fail(what, "second remove", k);
        probes.push_back(k);
      }
      for (size_t i = 0; i < 1000 * g_scale; ++i) {
        const uint64_t k = pick() + 1;
        if (idx->remove(k) != (ref.erase(k) != 0)) fail(what, "remove (neighbour)", k);
      }
      verify(*idx, ref, probes, g, what + " (removed)"); mark("remove");
      for (size_t i = 0; i < gone.size(); i += 2) put(gone[i] + (i % 4 == 0), payload(gone[i]) ^ 1, "put (after remove)");
      verify(*idx, ref, probes, g, what + " (put after remove)"); mark("put2");
    }
  }
  const size_t side1 = d->in_side(), in1 = d->in_sali();
  if (in1 + side1 != ref.size()) fail(what, "after the writes the index counts another number of keys than", ref.size());
  if (empty_it) {  // every key out, in a random order; then an index that holds nothing, and takes keys again
    std::vector<uint64_t> order;
    for (const auto& kvp : ref) order.push_back(kvp.first);
    std::shuffle(order.begin(), order.end(), g);
    for (uint64_t k : order)
      if (!idx->remove(k)) fail(what, "remove (emptying)", k);
    ref.clear();
    if (d->in_sali() || d->in_side()) fail(what, "emptied, but the index still counts keys:", d->in_sali() + d->in_side());
    probes.resize(std::min<size_t>(probes.size(), 5000));
    for (size_t i = 0; i < 5000 && !order.empty(); ++i) probes.push_back(order[i % order.size()]);
    verify(*idx, ref, probes, g, what + " (emptied)"); mark("empty");
    for (size_t i = 0; i < 8000 * g_scale && i < order.size(); ++i) put(order[i] + (i % 2), order[i] ^ 7, "put (into the emptied index)");
    for (uint64_t j = 0; j < 500; ++j) put(j * 3, j, "put (into the emptied index)"), put(M - j * 5, j, "put (into the emptied index)");
    verify(*idx, ref, probes, g, what + " (refilled)"); mark("refill");
    check_memory(*d, what + " (refilled)");
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (std::getenv("TEST_SALI_PHASES")) std::printf("  phases of the next line, seconds: %s\n", phases.c_str());
  std::printf("%-6s %-15s n=%-8zu loaded: sali=%zu side=%zu (%.3f%%) %.1f B/key; after writes: sali=%zu side=%zu  %s  %.1fs\n", name, label.c_str(), keys.size(), in0, side0,
              keys.empty() ? 0.0 : 100.0 * double(side0) / double(keys.size()), keys.empty() ? 0.0 : double(bytes) / double(keys.size()), in1, side1, g_bad == bad0 ? "ok" : "FAILED", s);
  std::fflush(stdout);
}

// an index that was never loaded
static void check_never_loaded(const char* name) {
  const std::string what = std::string(name) + " never loaded";
  const size_t bad0 = g_bad;
  std::unique_ptr<Sali> d(dynamic_cast<Sali*>(gret::make_sali<uint64_t, uint64_t>(name)));
  Index* idx = d.get();
  std::mt19937_64 g(5);
  Map ref;
  std::vector<uint64_t> probes;
  verify(*idx, ref, probes, g, what);
  uint64_t v = 0;
  if (idx->update(7, 1) || idx->remove(7) || idx->get(7, v)) fail(what, "an empty index holds a key:", 7);
  for (size_t i = 0; i < 15000 * g_scale; ++i) {
    const uint64_t k = i % 3 == 0 ? g() : i % 3 == 1 ? g() >> 30 : uint64_t(i) * 11;
    if (ref.count(k)) continue;
    ref.emplace(k, payload(k));
    if (!idx->put(k, payload(k))) fail(what, "put", k);
    probes.push_back(k);
  }
  for (uint64_t k : {uint64_t(0), M, H, H - 1})
    if (!ref.count(k)) { ref.emplace(k, 1); if (!idx->put(k, 1)) fail(what, "put", k); }
  verify(*idx, ref, probes, g, what + " (put)");
  check_memory(*d, what);
  std::printf("%-6s %-15s %zu puts into an index that was never loaded: sali=%zu side=%zu  %s\n", name, "never_loaded", ref.size(), d->in_sali(), d->in_side(), g_bad == bad0 ? "ok" : "FAILED");
}

static void check_guards() {
  const size_t bad0 = g_bad;
  for (const char* bad : {"", "sal", "sali ", " sali", "salix", "sali1", "saliu2", "SALI", "salir", "sali_u", "saliuu", "sali\n"})
    if (Index* made = gret::make_sali<uint64_t, uint64_t>(bad)) {
      std::printf("FAIL make_sali accepts the malformed name '%s'\n", bad);
      ++g_bad;
      delete dynamic_cast<Sali*>(made);
    }
  if (gret::make_sali<uint32_t, uint64_t>("sali") || gret::make_sali<uint64_t, double>("sali") || gret::make_sali<int64_t, uint64_t>("sali")) {
    std::printf("FAIL make_sali makes an index for key or payload types other than uint64\n");
    ++g_bad;
  }
  std::vector<KV> kv;
  for (uint64_t i = 0; i < 5000; ++i) kv.push_back({i * 9 + 4, payload(i)});
  auto holds = [&](Index* idx, const std::string& what) {
    uint64_t v = 0;
    for (const KV& e : kv)
      if (!idx->get(e.first, v) || v != e.second) { fail(what, "get", e.first); break; }
    if (idx->get(5, v)) fail(what, "get of an absent key", 5);
  };
  {  // input that is not sorted, or has a key twice: std::invalid_argument, and the index is what it was
    std::unique_ptr<Sali> d(new Sali("sali"));
    Index* idx = d.get();
    idx->bulk_load(kv.data(), kv.size());
    for (int which = 0; which < 2; ++which) {
      std::vector<KV> wrong = kv;
      if (which) wrong[700].first = wrong[699].first;
      else std::swap(wrong[10], wrong[4000]);
      bool threw = false;
      try {
        idx->bulk_load(wrong.data(), wrong.size());
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (!threw) fail("guards", "bulk_load took unsorted or repeated keys, case", uint64_t(which));
      holds(idx, "guards (after a refused bulk_load)");
    }
    idx->bulk_load(kv.data(), 100);  // a second load replaces the first
    uint64_t v = 0;
    if (idx->get(kv[100].first, v) || !idx->get(kv[99].first, v) || d->in_sali() != 100) fail("guards", "a second bulk_load did not replace the first; keys held:", d->in_sali());
    idx->bulk_load(nullptr, 0);
    if (idx->get(kv[0].first, v) || d->in_sali() != 0 || idx->scan(0, 10, kv.data() + 4000) != 0) fail("guards", "an empty bulk_load left keys:", d->in_sali());
  }
  {  // saliu given a key it holds: the authors' check fires (sali.h:1618), their index is left with a locked item
    std::unique_ptr<Sali> d(new Sali("saliu", false));
    Index* idx = d.get();
    idx->bulk_load(kv.data(), kv.size());
    holds(idx, "guards (saliu)");
    bool threw = false;
    std::string said;
    try {
      idx->put(kv[1234].first, 1);
    } catch (const std::runtime_error& e) {
      threw = true;
      said = e.what();
    }
    if (!threw || said.find("sali.h:1618") == std::string::npos || said.find("key1 < key2") == std::string::npos)
      std::printf("FAIL guards: saliu given a stored key did not throw the authors' check at sali.h:1618 (said: %s)\n", said.c_str()), ++g_bad;
    int dead = 0;
    uint64_t v = 0;
    std::vector<KV> out(4);
    try { idx->get(kv[1].first, v); } catch (const std::logic_error&) { ++dead; }
    try { idx->get(5, v); } catch (const std::logic_error&) { ++dead; }
    try { idx->put(5, v); } catch (const std::logic_error&) { ++dead; }
    try { idx->update(kv[1].first, v); } catch (const std::logic_error&) { ++dead; }
    try { idx->remove(kv[1].first); } catch (const std::logic_error&) { ++dead; }
    try { idx->scan(0, 4, out.data()); } catch (const std::logic_error&) { ++dead; }
    try { idx->memory_consumption(); } catch (const std::logic_error&) { ++dead; }
    if (dead != 7) fail("guards", "calls on an index whose authors' check failed that did not throw, of 7:", uint64_t(7 - dead));
    idx->bulk_load(kv.data(), kv.size());  // loaded again, it works again
    holds(idx, "guards (saliu, loaded again)");
    if (!idx->put(5, 55) || !idx->get(5, v) || v != 55) fail("guards", "put after a new bulk_load", 5);
  }
  std::printf("guards: names, refused input, a second and an empty bulk_load, saliu given a stored key  %s\n", g_bad == bad0 ? "ok" : "FAILED");
}

// A remove and a put in turn, so that the index holds as many keys at every rebuild of the authors' as when the subtree was
// built: their rebuild then computes speed = 0, and the next one below it speed / prev_speed = 0 / 0 (sali.h:3192, 3197). The
// wrapper must answer exactly all the way through, whether or not it had to replace their object (Sali::rebuilds()).
static void check_cycles(const char* name) {
  const size_t bad0 = g_bad;
  size_t rebuilds = 0, steps = 0, set_aside = 0;
  std::mt19937_64 g(31);
  struct Shape { size_t n; uint64_t base, stride; size_t steps; int trials; };
  const Shape shapes[] = {{200, 1000, 1000, 12000, 4}, {3000, uint64_t(1) << 40, uint64_t(1) << 20, 30000, 2}, {500, (uint64_t(1) << 62) + 12345, uint64_t(1) << 30, 12000, 2}};
  for (const Shape& sh : shapes)
    for (int t = 0; t < sh.trials; ++t) {
      const std::string what = std::string(name) + " on remove/put cycles, n=" + std::to_string(sh.n);
      std::unique_ptr<Sali> d(new Sali(name, std::string(name) != "saliu"));
      Index* idx = d.get();
      std::vector<KV> kv;
      for (size_t i = 0; i < sh.n; ++i) kv.push_back({sh.base + i * sh.stride, payload(i)});
      Map ref(kv.begin(), kv.end());
      std::vector<uint64_t> held, probes;
      for (const KV& e : kv) held.push_back(e.first);
      try {
        idx->bulk_load(kv.data(), kv.size());
        for (size_t i = 0; i < sh.steps * g_scale; ++i) {
          const size_t r = g() % held.size();
          const uint64_t out = held[r];
          if (!idx->remove(out)) fail(what, "remove", out);
          ref.erase(out), held[r] = held.back(), held.pop_back();
          uint64_t k;
          do k = held[g() % held.size()] + 1 + g() % 40; while (ref.count(k));  // next to a stored key: it meets it in a slot
          if (!idx->put(k, payload(k) + i)) fail(what, "put", k);
          ref.emplace(k, payload(k) + i), held.push_back(k);
          uint64_t v = 0;
          if (!idx->get(k, v) || v != payload(k) + i) fail(what, "get of the key just put", k);
          if (i % 16 == 0 && idx->get(out, v) != (ref.count(out) != 0)) fail(what, "get of the key just removed", out);
          if (i % 1024 == 0) {
            probes.push_back(k), probes.push_back(out);
            if (d->in_sali() + d->in_side() != ref.size()) fail(what, "the index counts another number of keys than", ref.size());
          }
        }
        verify(*idx, ref, probes, g, what);
        check_memory(*d, what);
      } catch (const std::exception& e) {
        std::printf("FAIL %s: threw %s\n", what.c_str(), e.what());
        ++g_bad;
      }
      rebuilds += d->rebuilds(), set_aside += d->set_aside_bytes(), steps += sh.steps * g_scale;
    }
  std::printf("%-6s cycles          %zu remove+put steps in 8 runs at a constant 200, 3000 and 500 keys: the authors' rebuild failed and the wrapper replaced their object %zu times "
              "(%zu bytes set aside)  %s\n",
              name, steps, rebuilds, set_aside, g_bad == bad0 ? "ok" : "FAILED");
  std::fflush(stdout);
}

#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 33)
#define TEST_SALI_MALLINFO 1
static size_t in_use() {
  const struct mallinfo2 mi = mallinfo2();
  return mi.uordblks + mi.hblkhd;
}
// memory_consumption against the allocator's own count of what is in use
static void check_memory_against_malloc(const std::vector<uint64_t>& keys, const std::string& label, bool grow = false) {
  std::vector<KV> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  std::mt19937_64 g(99);
  std::vector<KV> add(200000);
  for (KV& a : add) a = {keys[g() % keys.size()] + 1 + g() % 3, 7};
  if (grow)  // nothing loaded: every key comes through the authors' insert, which rebuilds subtrees as they deepen
    for (KV& a : add) a = {g() >> 14, 7};
  const size_t before = in_use();
  Sali* d = new Sali("sali");
  Index* idx = d;
  idx->bulk_load(kv.data(), grow ? 0 : kv.size());
  const size_t loaded = in_use() - before;
  const long long mem0 = idx->memory_consumption();
  const size_t total0 = d->authors_total_size();
  size_t in = 0;
  for (const KV& a : add) in += idx->put(a.first, a.second);
  const size_t later = in_use() - before;
  const long long mem1 = idx->memory_consumption();
  const double r0 = double(mem0) / double(loaded), r1 = double(mem1) / double(later);
  const bool ok = r0 >= 0.85 && r0 <= 1.01 && r1 >= 0.85 && r1 <= 1.01;
  if (!ok) ++g_bad;
  std::printf("memory %-10s n=%zu loaded: memory_consumption %lld, malloc in use %zu (%.1f%%; authors' total_size alone %.1f%%); after %zu puts: %lld of %zu (%.1f%%; total_size alone %.1f%%, "
              "waiting for an epoch %zu bytes, pool %zu bytes)  %s\n",
              label.c_str(), grow ? size_t(0) : keys.size(), mem0, loaded, 100 * r0, 100.0 * double(total0) / double(loaded), in,
              mem1, later, 100 * r1, 100.0 * double(d->authors_total_size()) / double(later), d->waiting_bytes(), d->pool_bytes(), ok ? "ok" : "FAILED");
  delete d;
}
#endif
#endif

template <class I>
static void time_one(const char* name, I* idx, const std::vector<KV>& kv, const std::vector<KV>& extra) {
  Param p(1, 0);
  idx->init(&p);
  std::vector<KV> copy = kv;
  auto t0 = std::chrono::steady_clock::now();
  idx->bulk_load(copy.data(), copy.size(), &p);
  const double build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::mt19937_64 g(3);
  std::vector<uint64_t> q(2000000);
  for (uint64_t& x : q) x = kv[g() % kv.size()].first;
  uint64_t sum = 0, v = 0;
  auto best_of = [&](auto&& one) {
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
      const auto s = std::chrono::steady_clock::now();
      for (uint64_t x : q) sum += one(x);
      best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - s).count() / double(q.size()));
    }
    return best;
  };
  const double get = best_of([&](uint64_t x) { return idx->get(x, v, &p) ? v : 1; });
  std::printf("time %-6s n=%zu build %.2fs  get %.1f ns", name, kv.size(), build, get);
  if (Sali* d = dynamic_cast<Sali*>(static_cast<Index*>(idx))) {
    auto& a = d->authors();
    const double at = best_of([&](uint64_t x) { return a.at(x, v) ? v : 1; });
    const double ex = best_of([&](uint64_t x) { return uint64_t(a.exists(x)); });
    // a get that misses: the key after a stored key (not stored, in this set, bar a few)
    double miss = 1e30;
    size_t thrown = 0;
    {
      const size_t m = 200000;
      const auto s = std::chrono::steady_clock::now();
      for (size_t i = 0; i < m; ++i) thrown += !idx->get(q[i] + 1, v, &p);
      miss = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - s).count() / double(m);
    }
    std::printf("  [the authors' at() called directly %.1f ns, their exists() (no epoch, no version checks, no payload) %.1f ns; a get that misses %.0f ns (%zu of 200000 missed)]", at, ex, miss,
                thrown);
  }
  t0 = std::chrono::steady_clock::now();
  size_t in = 0;
  for (const auto& e : extra) in += idx->put(e.first, e.second, &p);
  const double put = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(extra.size());
  const double get2 = best_of([&](uint64_t x) { return idx->get(x, v, &p) ? v : 1; });
  std::printf("  put %.1f ns (%zu new)  get after the puts %.1f ns  %.1f B/key  [checksum %llu]\n", put, in, get2, double(idx->memory_consumption()) / double(kv.size() + in),
              (unsigned long long)sum);
  std::fflush(stdout);
}

#if defined(__linux__) && defined(NDEBUG)
// The authors' code WITHOUT the side store's rule (shift 64), in a child under a memory and a CPU limit: a bulk load of `load`,
// then a put of each key of `add`, then every key asked for. 0: every key answered; 3: a runtime check of the authors' failed
// (their code would have exited with status 0 there); 4: a key is not found; 5: out of memory; else a signal (no end).
static int probe_one(const std::vector<uint64_t>& load, const std::vector<uint64_t>& add) {
  std::fflush(stdout);
  const pid_t pid = fork();
  if (pid == 0) {
    struct rlimit as = {rlim_t(1) << 30, rlim_t(1) << 30}, cpu = {2, 2};
    setrlimit(RLIMIT_AS, &as), setrlimit(RLIMIT_CPU, &cpu);
    if (!std::freopen("/dev/null", "w", stderr)) _exit(9);
    int code = 0;
    try {
      std::vector<KV> kv;
      for (uint64_t k : load) kv.push_back({k, payload(k)});
      Sali* d = new Sali("sali", true, 64);
      Index* idx = d;
      idx->bulk_load(kv.data(), kv.size());
      for (uint64_t k : add)
        if (!idx->put(k, payload(k))) code = 4;
      uint64_t v = 0;
      for (uint64_t k : load)
        if (!idx->get(k, v) || v != payload(k)) code = 4;
      for (uint64_t k : add)
        if (!idx->get(k, v) || v != payload(k)) code = 4;
    } catch (const std::bad_alloc&) {
      code = 5;
    } catch (const std::exception&) {
      code = 3;
    }
    _exit(code);
  }
  int st = 0;
  waitpid(pid, &st, 0);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 100 + WTERMSIG(st);
}
// Where the authors' code needs the rule, and whether the rule's margin is wide enough. Runs of `count` keys at five
// magnitudes, in three shapes (`gap` apart; gaps drawn from [gap, 3 gap); three such runs 2^10 to 2^30 apart), alone or with
// one far key on each side, bulk loaded at once or every other key loaded and the rest put. Each case is classified by the
// tightest shift whose rule would let all of its keys into SALI: the cases the header's shift lets through, and those a rule
// half as wide would, must all be answered.
static void probe() {
  const int kCases = 8, kShifts = 8;  // shifts 64 (no rule) down to 57
  std::mt19937_64 g(41);
  struct Band { const char* name; uint64_t lo, hi; };
  const Band bands[] = {{"[2^60, 2^61)", uint64_t(1) << 60, uint64_t(1) << 61}, {"[2^61, 2^62)", uint64_t(1) << 61, uint64_t(1) << 62}, {"[2^62, 2^63)", uint64_t(1) << 62, H},
                        {"[2^63, 2^63.58)", H, H + (H >> 1)}, {"[2^63.58, 2^64)", H + (H >> 1), M}};
  const uint64_t gaps[] = {1, 2, 3, 4, 6, 8, 12, 16, 33, 65};
  const int counts[] = {2, 3, 4, 5, 7, 100, 3000};
  size_t let[kShifts] = {0}, failed[kShifts] = {0}, total = 0;
  std::printf("probe: the authors' code without the side store's rule on runs of close keys (%zu run lengths x 3 shapes x alone/with far keys x loaded/half put, %d cases each; a child under 1 GB "
              "and 2 s of CPU)\n",
              sizeof counts / sizeof *counts, kCases);
  std::printf("  %-17s %-8s %-6s %s\n", "keys in", "min gap", "cases", "not answered: check failed / key lost / out of memory / no end / other   (run lengths that failed)");
  for (const Band& band : bands)
    for (uint64_t gap : gaps) {
      int out[6] = {0}, cases = 0;
      std::string lens;
      for (int count : counts) {
        bool bad_len = false;
        for (int shape = 0; shape < 3; ++shape)
          for (int variant = 0; variant < 4; ++variant)
            for (int t = 0; t < kCases; ++t) {
              const uint64_t room = uint64_t(count) * gap * 3 + (uint64_t(1) << 42), base = band.lo + (uint64_t(1) << 41) + g() % (band.hi - band.lo - room - (uint64_t(1) << 41));
              std::vector<uint64_t> run, load, add;
              uint64_t k = base, min_gap = ~uint64_t(0);
              for (int i = 0; i < count; ++i) {
                run.push_back(k);
                uint64_t step = shape == 0 ? gap : gap + g() % (2 * gap);
                if (shape == 2 && (i == count / 3 || i == 2 * count / 3)) step = (uint64_t(1) << 10) + g() % (uint64_t(1) << 30);
                if (i + 1 < count) min_gap = std::min(min_gap, step);
                k += step;
              }
              if (variant & 2) {  // every other key loaded, the others put in a random order
                for (int i = 0; i < count; ++i) (i & 1 ? add : load).push_back(run[i]);
                std::shuffle(add.begin(), add.end(), g);
              } else load = run;
              if (variant & 1) load.insert(load.begin(), base - (uint64_t(1) << 40) + g() % 1000), load.push_back(run.back() + (uint64_t(1) << 40) + g() % 1000);
              const int r = probe_one(load, add);
              const int kind = r == 0 ? 0 : r == 3 ? 1 : r == 4 ? 2 : r == 5 ? 3 : r == 100 + SIGXCPU || r == 100 + SIGKILL ? 4 : 5;
              ++out[kind], ++cases, ++total;
              bad_len |= kind != 0;
              // the tightest rule that lets this run through: every gap >= (key >> shift) + 1, taken at its largest key
              for (int s = 0; s < kShifts; ++s) {
                const int shift = 64 - s;
                const uint64_t m = shift >= 64 ? 1 : (run.back() >> shift) + 1;
                if (min_gap >= m) ++let[s], failed[s] += kind != 0;
              }
            }
        if (bad_len) lens += (lens.empty() ? "" : ",") + std::to_string(count);
      }
      std::printf("  %-17s %-8llu %-6d %d / %d / %d / %d / %d%s%s%s\n", band.name, (unsigned long long)gap, cases, out[1], out[2], out[3], out[4], out[5], lens.empty() ? "" : "   (", lens.c_str(),
                  lens.empty() ? "" : ")");
      std::fflush(stdout);
    }
  std::printf("  %zu cases in all\n", total);
  std::printf("  by rule: cases a rule gap >= (key >> shift) + 1 would let into SALI, and how many of them the authors' code did not answer\n");
  for (int s = 0; s < kShifts; ++s) {
    const int shift = 64 - s;
    std::printf("    shift %2d%s: %zu let through, %zu not answered%s\n", shift, shift == 64 ? " (no rule)" : "", let[s], failed[s], shift == Sali::kShift ? "   <- the header's rule" : "");
    if (shift >= Sali::kShift && shift <= Sali::kShift + 1 && failed[s]) ++g_bad;  // the header's rule, and the rule half as wide
  }
  if (Sali::kShift <= 64 - kShifts || Sali::kShift >= 64) std::printf("  (the header's shift %d is outside the range classified here)\n", Sali::kShift);
}
#endif

int main(int argc, char** argv) {
  std::vector<std::string> only;
  bool timing = false, probing = false;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--time")) timing = true;
    else if (!std::strcmp(argv[i], "--probe")) probing = true;
    else if (!std::strcmp(argv[i], "--full")) g_scale = 4;
    else only.push_back(argv[i]);
  auto want = [&](const std::string& s) { return only.empty() ? !(timing || probing) : std::find(only.begin(), only.end(), s) != only.end(); };
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; v.reserve(m); std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  std::printf("test_sali: long double has %d bits of mantissa here, the side store's rule is gap >= (key >> %d) + 1, TBB is %s\n", int(LDBL_MANT_DIG), Sali::kShift, Sali::tbb_name());

  std::vector<std::pair<std::string, std::vector<uint64_t>>> sets = {
      {"n0", {}}, {"n1", {5}}, {"n1max", {M}}, {"n1zero", {0}}, {"n2", {0, M}}, {"n3", {1, 2, M}}, {"n5", {0, 1, 2, 3, M}}, {"n17", rnd(0, uint64_t(1) << 40, 17)},
      {"uniform", rnd(1, uint64_t(1) << 50, 20000)}, {"wide", rnd(0, M, 20000)}};
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 50000; ++i) v.push_back(1000 + i); sets.push_back({"run", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), 20000); for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003); v.push_back(3); v.push_back(M); sets.push_back({"outliers", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + 100000, 30000); for (uint64_t j = 0; j < 100; ++j) v.push_back(M - j * 1000); sets.push_back({"over64outliers", v}); }
  { std::vector<uint64_t> v; for (int c = 0; c < 300; ++c) { const uint64_t b = g() >> 1; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } sets.push_back({"clusters", v}); }
  { std::vector<uint64_t> v; for (int c = 0; c < 300; ++c) { const uint64_t b = g() >> 20; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } sets.push_back({"clusters44", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 20000; ++i) v.push_back(i * 65536 + 12345); sets.push_back({"stride", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 10000; ++i) { v.push_back(i); v.push_back(M - i); } sets.push_back({"hugegap", v}); }
  // for the side store's rule: large keys 1 apart, in pairs, triples and runs, among keys far enough apart
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 20000; ++i) { const uint64_t k = H + (H >> 1) + (i << 40) + (g() & 0xffff); v.push_back(k); if (i % 3 == 0) v.push_back(k + 1); } sets.push_back({"pairs_top", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 20000; ++i) { const uint64_t k = H + (i << 41) + (g() & 0xffff); v.push_back(k); if (i % 2 == 0) v.push_back(k + 1), v.push_back(k + 2); } sets.push_back({"triples63", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 50000; ++i) v.push_back(M - i); sets.push_back({"run_top", v}); }
  {
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < 30000; ++i) v.push_back((uint64_t(1) << 61) + (i << 20) + (g() & 0x7ffff));
    for (int i = 0; i < 3000; ++i) { const uint64_t k = v[g() % 30000]; v.push_back(k + 1); if (i & 1) v.push_back(k + 2); if (i % 5 == 0) v.push_back(k + 700); }
    for (uint64_t i = 0; i < 500; ++i) v.push_back((uint64_t(1) << 62) + (uint64_t(1) << 50) + i);
    sets.push_back({"highmag", v});
  }
  // the closest keys the rule lets into SALI: runs whose every gap is the smallest allowed one (or up to three times that),
  // at every magnitude from where the rule starts to 2^64
  {
    std::vector<uint64_t> v;
    const int s = Sali::kShift >= 64 ? 60 : Sali::kShift;
    const int lens[] = {2, 3, 4, 5, 7, 30, 200};
    for (int r = 0; r < 1500; ++r) {
      const int bits = s + int(g() % uint64_t(64 - s));  // the run starts in [2^bits, 2^(bits+1))
      uint64_t k = (uint64_t(1) << bits) + g() % ((uint64_t(1) << bits) - (uint64_t(1) << 40));
      const int len = lens[g() % 7];
      for (int i = 0; i < len; ++i) {
        v.push_back(k);
        uint64_t m = margin(k);
        while (m < margin(k + m)) ++m;  // the smallest step that keeps the next key out of the side store
        k += r % 3 == 2 ? m + g() % (2 * m) : m;
      }
    }
    sets.push_back({"tight", v});
  }
  // the larger random sets
  sets.push_back({"uniform2m", rnd(1, uint64_t(1) << 50, 2000000)});
  { std::vector<uint64_t> v = rnd(0, M, 2000000); for (uint64_t k : {uint64_t(0), M, H, H - 1}) v.push_back(k); sets.push_back({"wide2m", v}); }

  if (want("guards")) check_guards();
  if (want("never_loaded")) check_never_loaded("sali"), check_never_loaded("saliu");
  if (want("cycles")) check_cycles("sali"), check_cycles("saliu");
  for (auto& [label, keys] : sets)
    if (want(label)) {
      // The large sets stop before the emptying: the authors' remove leaves the emptied slots and nodes in place, so each scan of
      // an emptied index walks all of them (4M slots here), and the pass after the emptying makes hundreds of scans.
      check(label, keys, "sali", keys.size() < 1000000);
      // saliu differs in put only: on every set with --full, on six of them without
      if (g_scale > 1 || label == "n0" || label == "n5" || label == "uniform" || label == "clusters44" || label == "highmag" || label == "tight" || label == "wide2m")
        check(label, keys, "saliu", keys.size() < 1000000);
    }
#ifdef TEST_SALI_MALLINFO
  if (want("memory")) {
    std::vector<uint64_t> a = rnd(1, uint64_t(1) << 50, 1000000), b = rnd(0, M, 300000);
    for (auto* v : {&a, &b}) std::sort(v->begin(), v->end()), v->erase(std::unique(v->begin(), v->end()), v->end());
    check_memory_against_malloc(a, "uniform1m");
    check_memory_against_malloc(b, "wide300k");
    check_memory_against_malloc(b, "grown", true);
  }
#else
  if (want("memory")) std::printf("memory: not checked against the allocator here (needs glibc 2.33's mallinfo2)\n");
#endif
  if (probing) {
#if defined(__linux__) && defined(NDEBUG)
    probe();
#else
    std::printf("probe: Linux and an NDEBUG build only (it needs a memory limit a child process obeys)\n");
#endif
  }
  if (timing) {  // not a benchmark: one small set, to see what the wrapper and the authors' concurrency control cost
    const char* bits = std::getenv("TEST_SALI_TIME_BITS");  // keys uniform below 2^bits (default 50; from 59 on x86 the put guard looks at neighbours)
    const uint64_t top = (uint64_t(1) << (bits ? std::min(63, std::max(30, std::atoi(bits))) : 50)) - 1;
    std::vector<uint64_t> keys = rnd(1, top, 3000000);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<KV> kv, extra;
    for (uint64_t k : keys) kv.push_back({k, 123456789});
    for (uint64_t k : rnd(1, top, 500000))
      if (!std::binary_search(keys.begin(), keys.end(), k)) extra.push_back({k, 123456789});  // new keys only: saliu's contract
    std::sort(extra.begin(), extra.end());
    extra.erase(std::unique(extra.begin(), extra.end()), extra.end());
    std::shuffle(extra.begin(), extra.end(), g);
    { std::unique_ptr<Sali> d(new Sali("sali")); time_one("sali", d.get(), kv, extra); }
    { std::unique_ptr<Sali> d(new Sali("saliu", false)); time_one("saliu", d.get(), kv, extra); }
#ifndef TEST_SALI_ALONE
    { auto* l = new LIPPInterface<uint64_t, uint64_t>; time_one("lipp", l, kv, extra); delete l; }
    { auto* l = new alexInterface<uint64_t, uint64_t>; time_one("alex", l, kv, extra); delete l; }
    { auto* l = new BTreeInterface<uint64_t, uint64_t>; time_one("btree", l, kv, extra); delete l; }
#endif
  }
  std::printf(g_bad ? "test_sali: %zu FAILURES\n" : "test_sali: all passed\n", g_bad);
  return g_bad ? 1 : 0;
}

#if defined(__has_feature)
#if __has_feature(address_sanitizer) && defined(_LIBCPP_VERSION) && !defined(TEST_SALI_STRICT_ASAN)
// see the head of this file: the authors' bulk_load fills two reserved, empty vectors through operator[] (sali.h:739, 740)
extern "C" const char* __asan_default_options() { return "detect_container_overflow=0"; }
#endif
#endif
