// SPDX-License-Identifier: MIT
// plex<maxerror> (gret/sota/plex.h: the authors' PLEX behind SortedBoundIndex), against a sorted array:
//   - the model alone: the bound of every stored key holds its position, ends inside the array and is no wider than 2 e + 2;
//     what the wrapper builds is what the authors' ts::Builder::Finalize() builds on the same keys (same size, same bounds);
//   - the index: get on every stored key and on keys never stored (next to stored keys, between keys, 0, 2^64 - 1, below and
//     above everything), scan from each of those starts at several lengths, put/update/remove refused and without effect,
//     memory_consumption = 16 bytes a key + what the model reports;
//   - key sets as in tests/test_scan.cpp (tiny sets, uniform, wide, runs, clusters, far outliers, 0 and 2^64 - 1, a stride, a
//     bit field) plus ranges that are exactly a power of two (PLEX cannot index those: the wrapper leaves the top key out),
//     sets made only of such ranges, dense sets, and one mixed set of 2M keys;
//   - what PLEX itself cannot build (its hist-tree builder would shift by a negative width): the wrapper must refuse it with
//     an exception that says so, on two sets where the tuner makes that choice and in unbuildable() on hand-made cases. A refusal
//     anywhere else is printed and counted, not hidden; the test fails on a wrong answer, never on a stated refusal.
//   test_plex [KEYS...]   (GRE key files, optional: the same checks on each, sampled; and the timing below on each)
// Built with -DNDEBUG (the -O3 build) it also prints a small timing on the 2M set: build time, bytes, ns a lookup.
#include "indexInterface.h"
#include "pgm/pgm.h"
#include "gret/sota/plex.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

using Keys = std::vector<uint64_t>;
using Pairs = std::vector<std::pair<uint64_t, uint64_t>>;
static const uint64_t M = ~uint64_t(0);

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

static Pairs pairs(const Keys& keys) {
  Pairs kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  return kv;
}

// indexInterface has no virtual destructor: an index is deleted as what it is, or its arrays stay
template <size_t... E>
static void drop_any(indexInterface<uint64_t, uint64_t>* x) {
  bool done = false;
  (void)((done = [&] { auto* c = dynamic_cast<gret::SortedBoundIndex<uint64_t, uint64_t, gret::PlexBound<uint64_t, E>>*>(x); delete c; return c != nullptr; }()) || ...);
  if (x && !done) std::printf("FAIL: an index of make_plex that is no SortedBoundIndex over PlexBound\n"), std::exit(1);
}
struct Drop {
  void operator()(indexInterface<uint64_t, uint64_t>* x) const {
    drop_any<1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 25, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500>(x);
  }
};
using Index = std::unique_ptr<indexInterface<uint64_t, uint64_t>, Drop>;

static size_t refusals = 0;  // builds the wrapper refused because PLEX cannot build what its tuner chose

static bool is_refusal(const std::exception& e) {
  return std::strstr(e.what(), "PlexBound: plex") && std::strstr(e.what(), "PLEX's tuner chose") && std::strstr(e.what(), "ts_cht/builder.h");
}

// The model alone, at maximum error E. Returns the failures; sets *bytes to what the model reports and *refused if it was refused.
template <size_t E>
static size_t model_check(const Keys& keys, const std::string& label, size_t* bytes, bool* refused) {
  *bytes = 0, *refused = false;
  if (keys.size() < 2) return 0;
  gret::PlexBound<uint64_t, E> pb;
  Keys copy = keys;
  try {
    pb.build(copy);
  } catch (const std::exception& e) {
    if (is_refusal(e)) {
      *refused = true;
      return 0;
    }
    std::printf("FAIL plex%zu on %s: build threw %s\n", size_t(E), label.c_str(), e.what());
    return 1;
  }
  *bytes = pb.bytes();
  const size_t n = keys.size();
  size_t bad = 0;
  if (pb.n != n || pb.built > n) ++bad, std::printf("FAIL plex%zu on %s: n %zu built %zu of %zu\n", size_t(E), label.c_str(), pb.n, pb.built, n);
  // the top key is left out only where the range is a power of two
  for (size_t k = n; k > 1; --k) {
    const uint64_t d = keys[k - 1] - keys[0];
    if ((d & (d - 1)) != 0) {
      if (pb.built != k) ++bad, std::printf("FAIL plex%zu on %s: built on %zu keys, not %zu\n", size_t(E), label.c_str(), pb.built, k);
      break;
    }
    if (k == 2 && pb.built != 0) ++bad, std::printf("FAIL plex%zu on %s: a model on power-of-two ranges only\n", size_t(E), label.c_str());
  }
  for (size_t i = 0; i < n; ++i) {
    const gret::KeyBound r = pb.get(keys[i]);
    const bool ok = r.lo <= i && i < r.hi && r.hi <= n && (i >= pb.built || r.hi - r.lo <= 2 * E + 2);
    if (!ok && ++bad <= 3) std::printf("FAIL plex%zu on %s: key %zu of %zu bounded by [%zu, %zu)\n", size_t(E), label.c_str(), i, n, r.lo, r.hi);
  }
  if (pb.built >= 2) {  // the authors' Finalize() on the same keys: the same model
    ts::Builder<uint64_t> b(keys[0], keys[pb.built - 1], E);
    for (size_t i = 0; i < pb.built; ++i) b.AddKey(keys[i]);
    const ts::TrieSpline<uint64_t> theirs = b.Finalize();
    if (theirs.GetSize() != pb.bytes()) ++bad, std::printf("FAIL plex%zu on %s: %zu bytes, the authors' Finalize() gives %zu\n", size_t(E), label.c_str(), pb.bytes(), theirs.GetSize());
    size_t differ = 0;
    auto same = [&](uint64_t x) {
      if (x < keys[0] || x > keys[pb.built - 1]) return;
      const ts::SearchBound a = theirs.GetSearchBound(x);
      const gret::KeyBound c = pb.get(x);
      differ += !(a.begin == c.lo && a.end == c.hi);
    };
    const size_t step = std::max<size_t>(1, pb.built / 300000);
    for (size_t i = 0; i < pb.built; i += step) {
      same(keys[i]), same(keys[i] + 1), same(keys[i] - 1);
      if (i + 1 < pb.built) same(keys[i] + (keys[i + 1] - keys[i]) / 2);
    }
    if (differ) ++bad, std::printf("FAIL plex%zu on %s: %zu bounds differ from the authors' Finalize()\n", size_t(E), label.c_str(), differ);
  }
  return bad;
}

template <size_t... E>
static size_t model_check_any(size_t e, const Keys& keys, const std::string& label, size_t* bytes, bool* refused) {
  size_t bad = 0;
  bool found = false;
  (void)(((e == E) && (found = true, bad = model_check<E>(keys, label, bytes, refused), true)) || ...);
  if (!found) std::printf("FAIL: no model_check for plex%zu\n", e), bad = 1;
  return bad;
}
static size_t model_check_e(size_t e, const Keys& keys, const std::string& label, size_t* bytes, bool* refused) {
  return model_check_any<1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 25, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500>(
      e, keys, label, bytes, refused);
}

// The index, by name. want_refusal: this set at this error is one PLEX cannot build. sample: get on a sample of the stored keys
// (the big key files), not on all of them.
static size_t check(const Keys& keys, const std::string& label, size_t e, size_t max_starts, bool want_refusal = false, bool sample = false) {
  const std::string name = "plex" + std::to_string(e);
  size_t model_bytes = 0;
  bool model_refused = false;
  size_t bad = keys.size() >= 64 ? model_check_e(e, keys, label, &model_bytes, &model_refused) : 0;
  Pairs kv = pairs(keys);
  Index idx(gret::make_plex<uint64_t, uint64_t>(name));
  if (!idx) {
    std::printf("FAIL %s: not a name\n", name.c_str());
    return 1;
  }
  Param p(1, 0);
  idx->init(&p);
  bool refused = false;
  try {
    idx->bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::exception& ex) {
    if (!is_refusal(ex)) {
      std::printf("FAIL %s on %s: threw %s\n", name.c_str(), label.c_str(), ex.what());
      return bad + 1;
    }
    refused = true;
    if (!want_refusal) std::printf("REFUSED %s on %s (n=%zu): %s\n", name.c_str(), label.c_str(), keys.size(), ex.what());
  }
  if (refused != model_refused) ++bad, std::printf("FAIL %s on %s: the index %s, the model alone %s\n", name.c_str(), label.c_str(), refused ? "refused" : "built", model_refused ? "refused" : "built");
  if (want_refusal && !refused) ++bad, std::printf("FAIL %s on %s: built, but PLEX's tuner was expected to choose what PLEX cannot build\n", name.c_str(), label.c_str());
  if (refused) {
    ++refusals;
    return bad;
  }
  const size_t n = keys.size();
  {  // every byte: the pairs and the model
    const long long want = (long long)(n * 16 + (n >= 64 ? model_bytes : 0));
    if (idx->memory_consumption() != want) ++bad, std::printf("FAIL %s on %s: memory_consumption %lld, not %lld\n", name.c_str(), label.c_str(), idx->memory_consumption(), want);
  }
  {  // stored keys
    const size_t step = sample ? std::max<size_t>(1, n / 2000000) : 1;
    size_t lost = 0;
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i)
      if (i < 300 || i + 300 >= n || i % step == 0) lost += !(idx->get(keys[i], v, &p) && v == payload(keys[i]));
    if (lost) ++bad, std::printf("FAIL %s on %s: %zu stored keys not found\n", name.c_str(), label.c_str(), lost);
  }
  Keys starts = {0, 1, M, M - 1, uint64_t(1) << 63};
  std::mt19937_64 g(11);
  auto around = [&](uint64_t k) { starts.push_back(k); if (k) starts.push_back(k - 1); if (k != M) starts.push_back(k + 1); };
  for (size_t i = 0; i < std::min<size_t>(80, n); ++i) { around(keys[i]); around(keys[n - 1 - i]); }
  for (size_t i = 0; i < max_starts; ++i) {
    const size_t r = g() % n;
    around(keys[r]);
    if (r + 1 < n && keys[r + 1] - keys[r] > 2) starts.push_back(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));
    starts.push_back(g());  // anywhere, mostly outside the keys
  }
  if (keys.front() > 0) starts.push_back(keys.front() / 2);
  if (keys.back() < M) starts.push_back(keys.back() + (M - keys.back()) / 2);
  {  // get on every start: found exactly when stored, with its payload, and untouched otherwise
    size_t wrong = 0;
    for (uint64_t x : starts) {
      uint64_t v = 12345;
      const bool got = idx->get(x, v, &p), is = std::binary_search(keys.begin(), keys.end(), x);
      wrong += got != is || (is ? v != payload(x) : v != 12345);
    }
    if (wrong) ++bad, std::printf("FAIL %s on %s: %zu wrong answers to get on starts\n", name.c_str(), label.c_str(), wrong);
  }
  {  // read-only: writes are refused and change nothing
    const uint64_t in = keys[n / 2], out = keys[n / 2] + 1 != 0 && !std::binary_search(keys.begin(), keys.end(), keys[n / 2] + 1) ? keys[n / 2] + 1 : in;
    uint64_t v = 0;
    bool ok = !idx->put(out, 7, &p) && !idx->update(in, 7, &p) && !idx->remove(in, &p) && !idx->put(in, 7, &p);
    ok = ok && idx->get(in, v, &p) && v == payload(in) && (out == in || !idx->get(out, v, &p));
    if (!ok) ++bad, std::printf("FAIL %s on %s: a write was accepted or had an effect\n", name.c_str(), label.c_str());
  }
  const size_t lens[] = {1, 2, 3, 16, 256, 4096, 65536, n + 5};
  Pairs out;
  size_t scan_bad = 0;
  for (uint64_t x : starts)
    for (size_t len : lens) {
      if (len > 4096 && (g() & 63)) continue;  // long scans on a sample of the starts
      if (len > 65536 && n > 1000000) continue;
      out.assign(len, {0, 0});
      const size_t got = idx->scan(x, len, out.data(), &p);
      const size_t at = size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin());
      const size_t want_n = std::min(len, n - at);
      bool ok = got == want_n;
      for (size_t i = 0; ok && i < got; ++i) ok = out[i].first == keys[at + i] && out[i].second == payload(keys[at + i]);
      if (!ok && ++scan_bad <= 3) std::printf("FAIL %s on %s: scan from %llu, %zu keys: got %zu, want %zu\n", name.c_str(), label.c_str(), (unsigned long long)x, len, got, want_n);
    }
  return bad + scan_bad;
}

// unbuildable() on hand-made tunings: range 0..7 is 3 bits; 2-bit nodes leave last bins 1 bit (2 keys) wide
static size_t check_guard() {
  using gret::plex_detail::unbuildable;
  const unsigned radix = std::numeric_limits<unsigned>::max();
  const std::vector<ts::Coord<uint64_t>> close = {{0, 0.0}, {1, 1.0}, {7, 2.0}}, apart = {{0, 0.0}, {2, 1.0}, {7, 2.0}};
  struct Case { const char* what; ts::Statistics t; const std::vector<ts::Coord<uint64_t>>* spline; uint64_t lo, hi; const char* says; };
  const Case cases[] = {
      {"two points in a last bin, error 1", ts::Statistics(4, 1, 0, 0), &close, 0, 7, "negative bin width"},
      {"the same: how many", ts::Statistics(4, 1, 0, 0), &close, 0, 7, "a bin 2^1 keys wide holds 2 spline points"},
      {"two points in a last bin, error 2", ts::Statistics(4, 2, 0, 0), &close, 0, 7, nullptr},
      {"one point a last bin", ts::Statistics(4, 1, 0, 0), &apart, 0, 7, nullptr},
      {"bins divide the range", ts::Statistics(8, 1, 0, 0), &close, 0, 7, nullptr},
      {"one-bit nodes", ts::Statistics(2, 1, 0, 0), &close, 0, 7, nullptr},
      {"a radix table", ts::Statistics(4, radix, 0, 0), &close, 0, 7, nullptr},
      {"more bits than the range", ts::Statistics(16, radix, 0, 0), &close, 0, 7, "more bits than the range has"},
      {"a 64-bit range, 10-bit nodes, 4 bits left", ts::Statistics(1024, 1, 0, 0), &close, 0, M, "a bin 2^4 keys wide holds 3"},
      {"a 64-bit range, 10-bit nodes, error 3", ts::Statistics(1024, 3, 0, 0), &close, 0, M, nullptr},
  };
  size_t bad = 0;
  for (const Case& c : cases) {
    const std::string why = unbuildable(c.t, *c.spline, c.lo, c.hi);
    const bool ok = c.says ? why.find(c.says) != std::string::npos : why.empty();
    if (!ok) ++bad, std::printf("FAIL unbuildable(%s): '%s'\n", c.what, why.c_str());
  }
  // offset keys: the bins are counted from the smallest key
  const std::vector<ts::Coord<uint64_t>> off = {{1001, 0.0}, {1002, 1.0}, {1007, 2.0}};  // from 1001: 0, 1, 6: bins 0, 0, 3
  if (unbuildable(ts::Statistics(4, 1, 0, 0), off, 1001, 1008).empty()) ++bad, std::printf("FAIL unbuildable(offset keys)\n");
  return bad;
}

// ns a get on 1M stored keys in random order, the best of three passes; build time is on stderr (SortedBoundIndex prints it)
static void time_one(indexInterface<uint64_t, uint64_t>& idx, const std::string& name, const Keys& keys, const std::string& label) {
  Pairs kv = pairs(keys);
  std::mt19937_64 g(3);
  Keys probes(1000000);
  for (uint64_t& x : probes) x = keys[g() % keys.size()];
  Param p(1, 0);
  try {
    idx.bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::exception& ex) {
    std::printf("timing %s %s: %s\n", label.c_str(), name.c_str(), ex.what());
    return;
  }
  uint64_t sum = 0, v = 0;
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t x : probes) idx.get(x, v, &p), sum += v;
    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(probes.size()));
  }
  std::printf("timing %s %s: n=%zu model_bytes=%lld %.1f ns/get (sum %llu)\n", label.c_str(), name.c_str(), keys.size(),
              idx.memory_consumption() - (long long)(keys.size() * 16), best, (unsigned long long)sum);
}

// plex next to the static PGM on the same footing (spgm, readonly.h), for scale only: a few million keys on a laptop
static void timing(const Keys& keys, const std::string& label) {
  for (size_t e : {size_t(4), size_t(16), size_t(32), size_t(64), size_t(256)}) {
    const std::string name = "plex" + std::to_string(e);
    Index idx(gret::make_plex<uint64_t, uint64_t>(name));
    time_one(*idx, name, keys, label);
  }
  gret::SortedBoundIndex<uint64_t, uint64_t, gret::PgmBound<uint64_t, 16>> a("spgm16");
  gret::SortedBoundIndex<uint64_t, uint64_t, gret::PgmBound<uint64_t, 64>> b("spgm64");
  gret::SortedBoundIndex<uint64_t, uint64_t, gret::PgmBound<uint64_t, 256>> c("spgm256");
  time_one(a, "spgm16", keys, label), time_one(b, "spgm64", keys, label), time_one(c, "spgm256", keys, label);
}

int main(int argc, char** argv) {
  unsetenv("PLEX_GUARD"), unsetenv("PLEX_VERIFY");
  std::mt19937_64 g(7);
  // below(m): 0 .. m - 1 (m > 0), from the raw generator, so the sets are the same with every standard library
  auto rnd = [&](uint64_t lo, uint64_t span, size_t m) { Keys v; for (size_t i = 0; i < m; ++i) v.push_back(lo + (span == M ? g() : g() % (span + 1))); return v; };
  std::vector<std::pair<std::string, Keys>> sets = {
      {"n1", {5}}, {"n1max", {M}}, {"n1zero", {0}}, {"n2", {0, M}}, {"n3", {1, 2, M}}, {"n5", {0, 1, 2, 3, M}}, {"n17", rnd(0, uint64_t(1) << 40, 17)},
      {"n63", rnd(0, uint64_t(1) << 40, 63)}, {"n64", rnd(0, M, 64)}, {"n65", rnd(5, uint64_t(1) << 20, 65)},
      {"uniform", rnd(1, uint64_t(1) << 50, 20000)}, {"wide", rnd(0, M, 20000)}};
  { Keys v; for (uint64_t i = 0; i < 50000; ++i) v.push_back(1000 + i); sets.push_back({"run", v}); }
  { Keys v; for (uint64_t i = 0; i < 70; ++i) v.push_back(i); sets.push_back({"run_from_0", v}); }
  { Keys v; for (uint64_t i = 0; i < 3000; ++i) v.push_back(M - i); sets.push_back({"run_to_max", v}); }
  { Keys v = rnd(uint64_t(1) << 40, uint64_t(1) << 34, 20000); for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003); v.push_back(3); v.push_back(M); sets.push_back({"outliers", v}); }
  { Keys v = rnd(uint64_t(1) << 40, 100000, 30000); for (uint64_t j = 0; j < 100; ++j) v.push_back(M - j * 1000); sets.push_back({"over64outliers", v}); }
  { Keys v; for (int c = 0; c < 300; ++c) { const uint64_t b = g() >> 1; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } sets.push_back({"clusters", v}); }
  { Keys v; for (uint64_t i = 0; i < 20000; ++i) v.push_back(i * 65536 + 12345); sets.push_back({"stride", v}); }
  { Keys v; for (uint64_t i = 0; i < 10000; ++i) { v.push_back(i); v.push_back(M - i); } sets.push_back({"hugegap", v}); }
  { Keys v; for (uint64_t a = 0; a < 300; ++a) for (uint64_t b = 0; b < 200; b += 1 + (a & 3)) v.push_back((a << 32) | (b << 8) | 7); sets.push_back({"bitfield", v}); v.push_back(M); sets.push_back({"bitfield_max", v}); }
  // ranges that are exactly a power of two, which PLEX cannot index: 2^40 above a key, 2^63 below 2^64 - 1, 0 .. 2^20 dense, 0 .. 2^63
  { Keys v = rnd(777, (uint64_t(1) << 40) - 1, 5000); v.push_back(777); v.push_back(777 + (uint64_t(1) << 40)); sets.push_back({"pow2_range", v}); }
  { Keys v = rnd(M - (uint64_t(1) << 63), uint64_t(1) << 63, 5000); v.push_back(M - (uint64_t(1) << 63)); v.push_back(M); sets.push_back({"pow2_range_to_max", v}); }
  { Keys v; for (uint64_t i = 0; i <= (uint64_t(1) << 20); i += 1 + (i % 7 == 0)) v.push_back(i); v.push_back(uint64_t(1) << 20); sets.push_back({"pow2_dense", v}); }
  { Keys v = rnd(0, (uint64_t(1) << 63) - 1, 300); v.push_back(0); v.push_back(uint64_t(1) << 63); v.push_back(uint64_t(1) << 62); sets.push_back({"pow2_from_0", v}); }
  // the range stays a power of two as top keys are left out: twice, down to two keys, and down to none (no model)
  { Keys v = rnd(100, 1000, 200); v.push_back(100); v.push_back(100 + 1024); v.push_back(100 + 2048); v.push_back(100 + (uint64_t(1) << 50)); sets.push_back({"pow2_thrice", v}); }
  { Keys v = {0, 3}; for (int i = 2; i < 64; ++i) v.push_back(uint64_t(1) << i); sets.push_back({"pow2_to_two_keys", v}); }
  { Keys v = {9}; for (int i = 0; i < 64; ++i) v.push_back(9 + (uint64_t(1) << i)); v.pop_back(); sets.push_back({"pow2_only_64", v}); }
  { Keys v = {0}; for (int i = 0; i < 64; ++i) v.push_back(uint64_t(1) << i); sets.push_back({"pow2_only_65", v}); }
  // dense sets, a small range; ids with jumps; an exponential spread; few dense clusters far apart
  sets.push_back({"dense", rnd(1000, 800000, 200000)});
  { Keys v; uint64_t x = g() >> 4; for (size_t i = 0; i < 200000; ++i) { x += (g() % 100 < 97) ? 1 + g() % 3 : g() % (uint64_t(1) << (g() % 40)); v.push_back(x); } sets.push_back({"ids_with_jumps", v}); }
  { Keys v; uint64_t x = 1; for (size_t i = 0; i < 6000; ++i) { x += 1 + x / 150; v.push_back(x); } sets.push_back({"exponential", v}); }
  { Keys v; for (int c = 0; c < 50; ++c) { const uint64_t b = g() >> 2; for (size_t i = 0; i < 4000; ++i) v.push_back(b + g() % 20000); } sets.push_back({"dense_clusters", v}); }
  const size_t all_e[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 25, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500};
  const size_t some_e[] = {1, 2, 4, 8, 16, 32, 64, 256, 1024, 100};
  size_t bad = 0, cases = 0;
  bad += check_guard(), ++cases;
  for (const char* nm : {"plex", "plex0", "plex3", "plex032", "plex32x", "plex2048", "plex-1", "plex99999", "spgm16", "", "plex 32", "PLEX32"})  // not names
    if (Index x{gret::make_plex<uint64_t, uint64_t>(nm)}) ++bad, std::printf("FAIL make_plex(\"%s\") is an index\n", nm);
  if (gret::make_plex<uint32_t, uint64_t>("plex32")) ++bad, std::printf("FAIL make_plex<uint32_t> is an index\n");
  ++cases;
  for (auto& [label, keys] : sets) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const bool every = keys.size() <= 20000;
    for (size_t e : all_e) {  // every error on the small sets; on the others, and for the many starts, a spread of them
      const bool some = std::find(std::begin(some_e), std::end(some_e), e) != std::end(some_e);
      if (!every && !some) continue;
      bad += check(keys, label, e, some ? 800 : 100) != 0, ++cases;
    }
  }
  {  // what PLEX's own builder cannot build, and the wrapper must refuse: a dense set with two far keys at error 2 (the tuner takes
     // 2^11-bin nodes with error 16 over 64 bits, 9 bits left, 17 spline points in one last bin), and 50 dense clusters at error 2
    std::mt19937_64 h(1);
    Keys v;
    for (size_t i = 0; i < 50000; ++i) v.push_back(1000 + h() % 200000);
    v.push_back(M), v.push_back(3);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    bad += check(v, "refused_dense_far_keys", 2, 500, true) != 0, ++cases;
    for (size_t e : {size_t(1), size_t(4), size_t(32)}) bad += check(v, "dense_far_keys", e, 500) != 0, ++cases;
    std::mt19937_64 h2(7);
    v.clear();
    for (int c = 0; c < 50; ++c) { const uint64_t b = h2() >> 2; for (size_t i = 0; i < 8000; ++i) v.push_back(b + h2() % 40000); }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    bad += check(v, "refused_dense_clusters", 2, 500, true) != 0, ++cases;
    for (size_t e : {size_t(1), size_t(4), size_t(32)}) bad += check(v, "dense_clusters_400k", e, 500) != 0, ++cases;
  }
  Keys big;
  {  // 2M keys: half uniform over 64 bits, half in 1000 clusters of mixed density
    std::mt19937_64 h(21);
    for (size_t i = 0; i < 1000000; ++i) big.push_back(h());
    for (int c = 0; c < 1000; ++c) { const uint64_t b = h(), spread = uint64_t(1) << (12 + h() % 30); for (size_t i = 0; i < 1000; ++i) big.push_back(b + h() % spread); }
    std::sort(big.begin(), big.end());
    big.erase(std::unique(big.begin(), big.end()), big.end());
    for (size_t e : some_e) bad += check(big, "mixed2M", e, 3000) != 0, ++cases;
  }
  {  // $PLEX_VERIFY=1 builds and answers as without it; $PLEX_GUARD=0 (the authors' Finalize() itself) likewise, where PLEX can build
    setenv("PLEX_VERIFY", "1", 1);
    for (size_t e : {size_t(8), size_t(64)}) bad += check(sets[10].second, "uniform_verified", e, 500) != 0, ++cases;
    unsetenv("PLEX_VERIFY");
    setenv("PLEX_GUARD", "0", 1);
    for (size_t e : {size_t(8), size_t(64)}) bad += check(sets[11].second, "wide_unguarded", e, 500) != 0, ++cases;
    unsetenv("PLEX_GUARD");
  }
  std::vector<std::pair<std::string, Keys>> files;
  for (int a = 1; a < argc; ++a) {
    FILE* f = std::fopen(argv[a], "rb");
    uint64_t n = 0;
    if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
    Keys k(n);
    if (std::fread(k.data(), 8, n, f) != n) return 2;
    std::fclose(f);
    std::sort(k.begin(), k.end());
    k.erase(std::unique(k.begin(), k.end()), k.end());
    for (size_t e : some_e) bad += check(k, argv[a], e, 3000, false, true) != 0, ++cases;
    files.push_back({argv[a], std::move(k)});
  }
#ifdef NDEBUG
  timing(big, "mixed2M");
  for (auto& [label, k] : files) timing(k, label);
#endif
  std::printf("test_plex: %s (%zu cases, %zu failed, %zu builds refused: PLEX cannot build its tuner's choice)\n", bad ? "FAILED" : "ok", cases, bad, refusals);
  return bad ? 1 : 0;
}
