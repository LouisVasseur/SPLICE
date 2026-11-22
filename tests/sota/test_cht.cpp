// SPDX-License-Identifier: MIT
// CHT (Stoian's compact Hist-Tree, wrapped by gret/sota/cht.h) against a sorted array, on key sets that force a single node, deep trees,
// empty bins, a partial last level, a power-of-two key range (the builder's last-key overflow) and a range narrower than one node:
//   - the index by name: get for every stored key and for keys never stored (neighbours, 0, 2^64 - 1, below and above the range, inside
//     gaps), scan from every kind of start at several lengths, put/update/remove false, memory = pairs + model;
//   - the model alone: the bound holds for stored keys and for misses, is the authors' GetSearchBound less its last position, and the
//     node count is what the split rule gives (a bin over max_error keys gets a child), counted here by a separate recursion;
//   - what must throw does, and only that: a setting whose builder would split below a node's width on this key set (std::invalid_argument
//     naming the max error), duplicates; max_error >= bins / 2 never throws; names that are not cht<bins>e<max_error> give nullptr;
//   - the names ending in "std" (ChtStdIndex: the authors' std::lower_bound over their own bound) get the same index checks.
// Ends with a small timing of stored-key lookups on the 2M-key set (cht at a few settings, spgm64 beside it), for sanity only.
//   test_cht [KEYS...]   (GRE key files, optional: the same checks on each)
#include "indexInterface.h"
#include "pgm/pgm.h"
#include "gret/sota/cht.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using gret::sota::ChtBound;
using gret::sota::ChtConfig;
using Pair = std::pair<uint64_t, uint64_t>;
using Index = indexInterface<uint64_t, uint64_t>;
// indexInterface has no virtual destructor: the tests own the index by its own type, and call it through the interface as GRE does
using ChtIndex = gret::SortedBoundIndex<uint64_t, uint64_t, ChtBound<uint64_t>>;
using ChtStd = gret::sota::ChtStdIndex<uint64_t, uint64_t>;  // the names ending in "std"
struct Owned {
  std::unique_ptr<ChtIndex> a;
  std::unique_ptr<ChtStd> b;
  Index* get() const { return a ? static_cast<Index*>(a.get()) : static_cast<Index*>(b.get()); }
  explicit operator bool() const { return a || b; }
};
static Owned make(const std::string& name) {
  Owned o;
  ChtConfig c;
  Index* const p = gret::sota::make_cht<uint64_t, uint64_t>(name);
  if (!p) return o;
  if (gret::sota::parse_cht_name(name, c) && c.std_search) o.b.reset(static_cast<ChtStd*>(p));
  else o.a.reset(static_cast<ChtIndex*>(p));
  return o;
}
static const uint64_t M = ~uint64_t(0);

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

static std::vector<Pair> pairs(const std::vector<uint64_t>& keys) {
  std::vector<Pair> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  return kv;
}

// What the tree must look like, from the split rule alone: which keys reach the builder, whether a tree is built, whether the builder can
// build it (no bin over max_error at a width below one node's), and its node count.
struct Shape {
  size_t built = 0, nodes = 0;
  bool tree = false, feasible = true;
};
static Shape oracle(const std::vector<uint64_t>& k, const ChtConfig& c) {
  Shape s;
  s.built = k.size();
  auto pow2 = [](uint64_t r) { return (r & (r - 1)) == 0; };
  while (s.built > 1 && pow2(k[s.built - 1] - k[0])) --s.built;
  if (s.built <= 1) return s;
  const uint64_t range = k[s.built - 1] - k[0];
  int lg = 0, log = 0;
  while (lg < 64 && (uint64_t(1) << lg) < range) ++lg;
  while ((size_t(1) << log) < c.bins) ++log;
  if (lg < log) return s;
  s.tree = true;
  std::function<void(size_t, size_t, int)> node = [&](size_t a, size_t b, int width) {
    ++s.nodes;
    for (size_t i = a; i < b;) {
      const uint64_t bin = (k[i] - k[0]) >> width;
      size_t j = i;
      while (j < b && ((k[j] - k[0]) >> width) == bin) ++j;
      if (j - i > c.max_error) {
        if (width < log) s.feasible = false;
        else node(i, j, width - log);
      }
      i = j;
    }
  };
  node(0, s.built, lg - log);
  return s;
}

static std::vector<uint64_t> probes(const std::vector<uint64_t>& keys, size_t samples, uint64_t seed) {  // stored keys, their neighbours, gaps, the ends
  std::vector<uint64_t> x = {0, 1, M, M - 1, uint64_t(1) << 63, (uint64_t(1) << 63) - 1};
  std::mt19937_64 g(seed);
  auto around = [&](uint64_t k) { x.push_back(k); if (k) x.push_back(k - 1); if (k != M) x.push_back(k + 1); };
  for (size_t i = 0; i < std::min<size_t>(80, keys.size()); ++i) { around(keys[i]); around(keys[keys.size() - 1 - i]); }
  for (size_t i = 0; i < samples; ++i) {
    const size_t r = g() % keys.size();
    around(keys[r]);
    if (r + 1 < keys.size() && keys[r + 1] - keys[r] > 2) x.push_back(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));
    x.push_back(g());
  }
  if (keys.front() > 1000) x.push_back(keys.front() - 1000), x.push_back(keys.front() / 2);
  if (keys.back() < M - 1000) x.push_back(keys.back() + 1000), x.push_back(keys.back() + (M - keys.back()) / 2);
  return x;
}

// the model alone
static size_t check_model(const std::vector<uint64_t>& keys, const std::string& label, const ChtConfig& c) {
  if (keys.size() < 64) return 0;  // SortedBoundIndex builds no model below 64 keys
  const std::string who = c.name() + " (model) on " + label;
  const Shape want = oracle(keys, c);
  gret::sota::cht_pending() = c;
  ChtBound<uint64_t> m;
  gret::sota::cht_pending() = ChtConfig();
  std::vector<uint64_t> scratch = keys;
  try {
    m.build(scratch);
  } catch (const std::invalid_argument& e) {
    if (want.feasible || !std::strstr(e.what(), "over max error")) { std::printf("FAIL %s: threw '%s'\n", who.c_str(), e.what()); return 1; }
    return 0;
  }
  if (!want.feasible) { std::printf("FAIL %s: built, but a bin below one node's width is over max error\n", who.c_str()); return 1; }
  size_t bad = 0;
  if (m.tree != want.tree || m.built != want.built || m.n != keys.size() || m.top != keys[want.built - 1])
    std::printf("FAIL %s: tree %d built %zu, want %d %zu\n", who.c_str(), int(m.tree), m.built, int(want.tree), want.built), ++bad;
  const size_t want_bytes = sizeof(m) + want.nodes * c.bins * sizeof(unsigned);
  if (m.bytes() != want_bytes) std::printf("FAIL %s: %zu bytes, want %zu (%zu nodes)\n", who.c_str(), m.bytes(), want_bytes, want.nodes), ++bad;
  if (bad) return bad;
  auto bound_ok = [&](uint64_t x) {
    const size_t lb = size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin());
    const gret::KeyBound b = m.get(x);
    const bool stored = lb < keys.size() && keys[lb] == x;
    bool ok = b.lo <= lb && lb <= b.hi && b.hi <= keys.size() && (!stored || lb < b.hi);
    if (m.tree && x <= m.top) {  // the authors' bound, less its last position
      const cht::SearchBound a = m.m.GetSearchBound(x);
      ok = ok && a.begin == b.lo && a.end == std::min(a.begin + c.max_error + 1, m.built) && b.hi == std::min(a.begin + c.max_error, m.built) &&
           b.hi - b.lo <= c.max_error;
    }
    return ok;
  };
  const size_t step = std::max<size_t>(1, keys.size() / 300000);
  for (size_t i = 0; i < keys.size(); i += (i < 2000 || i + 2000 >= keys.size()) ? 1 : step) bad += !bound_ok(keys[i]);
  for (uint64_t x : probes(keys, 3000, 5))
    if (x >= keys.front() && x <= keys.back()) bad += !bound_ok(x);  // SortedBoundIndex asks the model inside the key range only
  if (bad) std::printf("FAIL %s: %zu bounds wrong\n", who.c_str(), bad);
  return bad;
}

// the index by name
static size_t check_index(const std::vector<uint64_t>& keys, const std::string& label, const ChtConfig& c, size_t samples) {
  const std::string name = c.name(), who = name + " on " + label;
  const Shape want = keys.size() < 64 ? Shape() : oracle(keys, c);
  std::vector<Pair> kv = pairs(keys);
  Param p(1, 0);
  const Owned owner = make(name);
  if (!owner) { std::printf("FAIL %s: make_cht gave nullptr\n", who.c_str()); return 1; }
  Index* const idx = owner.get();
  try {
    idx->init(&p);
    idx->bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::invalid_argument& e) {
    if (want.feasible || !std::strstr(e.what(), "over max error")) { std::printf("FAIL %s: threw '%s'\n", who.c_str(), e.what()); return 1; }
    return 0;
  } catch (const std::exception& e) {
    std::printf("FAIL %s: threw %s\n", who.c_str(), e.what());
    return 1;
  }
  if (!want.feasible) { std::printf("FAIL %s: built, but should have thrown\n", who.c_str()); return 1; }
  size_t bad = 0;
  uint64_t v = 0;
  {  // every stored key
    size_t lost = 0;
    for (uint64_t k : keys) lost += !(idx->get(k, v, &p) && v == payload(k));
    if (lost) std::printf("FAIL %s: %zu stored keys not found\n", who.c_str(), lost), bad += lost;
  }
  const std::vector<uint64_t> xs = probes(keys, samples, 11);
  {  // keys never stored
    size_t wrong = 0;
    for (uint64_t x : xs) {
      const bool stored = std::binary_search(keys.begin(), keys.end(), x);
      v = 12345;
      const bool got = idx->get(x, v, &p);
      wrong += got != stored || (stored && v != payload(x)) || (!stored && v != 12345);
    }
    if (wrong) std::printf("FAIL %s: %zu lookups wrong among stored keys' neighbours and misses\n", who.c_str(), wrong), bad += wrong;
  }
  if (idx->put(keys[0], 1, &p) || idx->put(keys[0] + 1, 1, &p) || idx->update(keys[0], 1, &p) || idx->remove(keys[0], &p) || !idx->get(keys[0], v, &p) ||
      v != payload(keys[0]))
    std::printf("FAIL %s: put, update or remove did something\n", who.c_str()), ++bad;
  const size_t model = keys.size() < 64 ? 0 : sizeof(ChtBound<uint64_t>) + want.nodes * c.bins * sizeof(unsigned);
  if (size_t(idx->memory_consumption()) != keys.size() * sizeof(Pair) + model)
    std::printf("FAIL %s: memory %lld, want %zu\n", who.c_str(), idx->memory_consumption(), keys.size() * sizeof(Pair) + model), ++bad;
  const size_t lens[] = {1, 2, 3, 16, 256, 4096, 65536, keys.size() + 5};
  std::mt19937_64 g(3);
  std::vector<Pair> out;
  size_t scans_bad = 0;
  for (uint64_t x : xs)
    for (size_t len : lens) {
      if (len > 4096 && (g() & 255)) continue;  // long scans on a sample of the starts
      if (len > 65536 && keys.size() > 1000000) continue;
      out.assign(len, {0, 0});
      const size_t got = idx->scan(x, len, out.data(), &p);
      const size_t at = size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin());
      const size_t want_n = std::min(len, keys.size() - at);
      bool ok = got == want_n;
      for (size_t i = 0; ok && i < got; ++i) ok = out[i].first == keys[at + i] && out[i].second == payload(keys[at + i]);
      if (!ok && ++scans_bad <= 3) std::printf("FAIL %s: scan from %llu, %zu long: got %zu, want %zu\n", who.c_str(), (unsigned long long)x, len, got, want_n);
    }
  return bad + scans_bad;
}

static double time_gets(Index& idx, const std::vector<uint64_t>& q) {  // ns per stored-key lookup, best of 3
  Param p(1, 0);
  double best = 1e30;
  uint64_t sum = 0, v = 0;
  for (int rep = 0; rep < 3; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t x : q) sum += idx.get(x, v, &p) ? v : 1;
    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(q.size()));
  }
  if (sum == 42) std::printf(" ");  // keeps the loop
  return best;
}

int main(int argc, char** argv) {
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  std::vector<std::pair<std::string, std::vector<uint64_t>>> sets = {
      {"n1", {5}}, {"n1max", {M}}, {"n1zero", {0}}, {"n2", {0, M}}, {"n3", {1, 2, M}}, {"n5", {0, 1, 2, 3, M}}, {"n17", rnd(0, uint64_t(1) << 40, 17)},
      {"n64", rnd(0, M, 64)}, {"uniform", rnd(1, uint64_t(1) << 50, 20000)}, {"wide", rnd(0, M, 20000)}};
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 50000; ++i) v.push_back(1000 + i); sets.push_back({"run", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), 20000); for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003); v.push_back(3); v.push_back(M); sets.push_back({"outliers", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + 100000, 30000); for (uint64_t j = 0; j < 100; ++j) v.push_back(M - j * 1000); sets.push_back({"over64outliers", v}); }
  { std::vector<uint64_t> v; for (int c = 0; c < 300; ++c) { const uint64_t b = g() >> 1; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } sets.push_back({"clusters", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 20000; ++i) v.push_back(i * 65536 + 12345); sets.push_back({"stride", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 10000; ++i) { v.push_back(i); v.push_back(M - i); } sets.push_back({"hugegap", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 40000; ++i) v.push_back(77 + i * 3 + (i % 7 == 0)); sets.push_back({"stride3", v}); }
  // max - min an exact power of two: the builder would bin the largest key one past the root (dense, sparse, at 2^63, and min 0 with max 2^64 - 1 beside it)
  { std::vector<uint64_t> v; for (uint64_t i = 0; i <= 64; ++i) v.push_back(i); sets.push_back({"pow2_dense65", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i <= 4096; ++i) v.push_back(900 + i); sets.push_back({"pow2_dense4097", v}); }
  { std::vector<uint64_t> v = rnd(1000, 1000 + (uint64_t(1) << 40), 20000); v.push_back(1000); v.push_back(1000 + (uint64_t(1) << 40)); sets.push_back({"pow2_sparse", v}); }
  { std::vector<uint64_t> v = rnd(0, uint64_t(1) << 63, 20000); v.push_back(0); v.push_back(uint64_t(1) << 63); sets.push_back({"pow2_63", v}); }
  { std::vector<uint64_t> v = rnd(5, 5 + (uint64_t(1) << 63), 20000); v.push_back(5); v.push_back(5 + (uint64_t(1) << 63)); v.push_back(M); sets.push_back({"pow2_63_then_max", v}); }
  // a chain of them: after each top key is left out the range is a power of two again, 44 times
  { std::vector<uint64_t> v = rnd(0, (uint64_t(1) << 20) - 1, 5000); v.push_back(0); for (int b = 20; b < 64; ++b) v.push_back(uint64_t(1) << b); sets.push_back({"pow2_chain", v}); }
  // all the way down: every key but the first is left out, no tree
  { std::vector<uint64_t> v = {0}; for (int b = 0; b < 64; ++b) v.push_back(uint64_t(1) << b); sets.push_back({"pow2_only", v}); }
  // a key range narrower than one node of 256 or 1024 bins
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 100; ++i) v.push_back((uint64_t(1) << 50) + i); sets.push_back({"narrow100", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 70; ++i) v.push_back(M - 3 * i); sets.push_back({"narrow_top", v}); }
  { std::vector<uint64_t> v = rnd(0, M, 2000000); sets.push_back({"big2M", v}); }
  {  // 1M keys: half uniform, half in runs of stride 1 to 4 (a partial last level decides which settings build)
    std::vector<uint64_t> v = rnd(0, M >> 1, 500000);
    for (int c = 0; c < 500; ++c) { const uint64_t b = g() >> 1, s = 1 + c % 4; for (uint64_t i = 0; i < 1000; ++i) v.push_back(b + i * s); }
    sets.push_back({"mix1M", v});
  }
  const char* names[] = {"cht64e32",  "cht64e4",    "cht64e8",    "cht64e16",   "cht64e33",    "cht64e64",  "cht64e1024", "cht32e32", "cht16e8", "cht16e2",
                         "cht4e2",    "cht4e1",     "cht2e1",     "cht256e128", "cht256e16",   "cht1024e512", "cht1024e32", "ccht64e32", "ccht64e4", "ccht16e8",
                         "ccht2e1",   "cht64e32std", "cht64e8std", "cht16e8std", "cht4e1std", "cht256e16std", "cht64e1024std", "ccht64e4std"};
  const char* big_names[] = {"cht64e32", "cht64e8", "cht16e8", "cht256e128", "cht1024e64", "cht2e1", "ccht64e32", "ccht16e8", "cht64e1024",
                             "cht64e32std", "cht16e8std", "cht64e8std", "cht64e1024std", "ccht16e8std"};
  size_t bad = 0, cases = 0, refused = 0;
  auto run = [&](const std::vector<uint64_t>& keys, const std::string& label, const char* nm, size_t samples) {
    ChtConfig c;
    if (!gret::sota::parse_cht_name(nm, c) || c.name() != nm) { std::printf("FAIL %s: name not parsed\n", nm); ++bad; return; }
    if (keys.size() >= 64) {
      const Shape s = oracle(keys, c);
      refused += !s.feasible;
      if (!s.feasible && c.max_error >= c.bins / 2) std::printf("FAIL %s on %s: max_error >= bins / 2, yet not buildable\n", nm, label.c_str()), ++bad;
    }
    const size_t e = (c.std_search ? 0 : check_model(keys, label, c)) + check_index(keys, label, c, samples);  // the model is the same with "std"
    ++cases;
    bad += e != 0;
  };
  for (auto& [label, keys] : sets) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const bool big = keys.size() > 200000;
    if (big) for (const char* nm : big_names) run(keys, label, nm, 1000);
    else for (const char* nm : names) run(keys, label, nm, 400);
  }
  {  // 2^20 bins: one node of 4 MB on a sparse set; and what the settings above do not reach
    std::vector<uint64_t>& u = sets[8].second;
    run(u, "uniform", "cht1048576e32", 500);
    run(u, "uniform", "cht64e1048576", 500);
    run(u, "uniform", "ccht1048576e4", 500);
  }
  {  // must throw, and must not: the dense run at 64 bins has 16 keys per bin of the last level (range 49999: shift 10, 10 mod 6 = 4)
    const std::vector<uint64_t>& r = sets[10].second;
    ChtConfig c;
    c.max_error = 8;
    if (oracle(r, c).feasible) std::printf("FAIL: cht64e8 on run should not be buildable\n"), ++bad;
    c.max_error = 15;
    if (oracle(r, c).feasible) std::printf("FAIL: cht64e15 on run should not be buildable\n"), ++bad;
    c.max_error = 16;
    if (!oracle(r, c).feasible) std::printf("FAIL: cht64e16 on run should be buildable\n"), ++bad;
    run(r, "run", "cht64e15", 500);
    ++cases;
  }
  {  // duplicates and unsorted keys: the model refuses them
    for (int t = 0; t < 2; ++t) {
      std::vector<uint64_t> d = sets[8].second;
      if (t == 0) d[100] = d[99];
      else std::swap(d[100], d[5000]);
      ChtBound<uint64_t> m;
      bool threw = false;
      try { m.build(d); } catch (const std::invalid_argument&) { threw = true; }
      if (!threw) std::printf("FAIL: %s keys accepted\n", t ? "unsorted" : "duplicate"), ++bad;
      ++cases;
    }
  }
  {  // names that are not CHT's; and a model built outside make_cht has the defaults, whatever was built before
    const char* not_names[] = {"cht", "cht64", "cht64e", "cht64e0", "chte32", "cht63e32", "cht1e32", "cht064e32", "cht64e032", "cht64e32x", "cht64x32", "cht2097152e32",
                               "cht64e2097152", "ccht", "ccht64", "cchtt64e32", "ht64e32", "cht64e32_alex", "cht-64e32", "cht64e-32", "cht64e99999999999", "",
                               "cht64e32st", "cht64e32stdx", "cht64e32stdstd", "cht64estd", "chtstd", "cht64e32STD", "cht64std32", "stdcht64e32"};
    for (const char* nm : not_names) {
      if (make(nm)) std::printf("FAIL: '%s' is not a CHT name, yet it built\n", nm), ++bad;
      ++cases;
    }
    make("ccht16e8");
    ChtBound<uint64_t> d;
    if (d.cfg.bins != 64 || d.cfg.max_error != 32 || d.cfg.cache) std::printf("FAIL: the defaults are not 64 bins, max error 32\n"), ++bad;
    if (gret::sota::make_cht<uint32_t, uint64_t>("cht64e32")) std::printf("FAIL: built for uint32_t keys\n"), ++bad;
    cases += 2;
  }
  for (int a = 1; a < argc; ++a) {  // GRE key files
    FILE* f = std::fopen(argv[a], "rb");
    uint64_t n = 0;
    if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
    std::vector<uint64_t> k(n);
    if (std::fread(k.data(), 8, n, f) != n) return 2;
    std::fclose(f);
    std::sort(k.begin(), k.end());
    k.erase(std::unique(k.begin(), k.end()), k.end());
    for (const char* nm : big_names) run(k, argv[a], nm, 3000);
  }
  {  // sanity timing: stored keys of big2M in random order
    const std::vector<uint64_t>& keys = sets[sets.size() - 2].second;
    std::vector<uint64_t> q(1000000);
    for (uint64_t& x : q) x = keys[g() % keys.size()];
    std::vector<Pair> kv = pairs(keys);
    Param p(1, 0);
    std::printf("timing, %zu uniform keys, %zu stored-key lookups, ns per lookup (best of 3):\n", keys.size(), q.size());
    for (const char* nm : {"cht64e32", "cht64e32std", "cht64e16", "cht64e16std", "cht64e8", "cht16e8", "cht16e8std", "cht256e128", "cht256e128std", "cht1024e64",
                           "cht1024e512", "cht1024e512std", "ccht64e8", "spgm64"}) {
      const Owned c = std::string(nm) == "spgm64" ? Owned() : make(nm);
      const std::unique_ptr<gret::SortedBoundIndex<uint64_t, uint64_t, gret::PgmBound<uint64_t, 64>>> s(
          c ? nullptr : new gret::SortedBoundIndex<uint64_t, uint64_t, gret::PgmBound<uint64_t, 64>>(nm));
      Index* const idx = c ? c.get() : static_cast<Index*>(s.get());
      idx->init(&p);
      const auto t0 = std::chrono::steady_clock::now();
      idx->bulk_load(kv.data(), kv.size(), &p);
      const double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("  %-15s %7.1f ns  model %9.3f MB  build %7.1f ms\n", nm, time_gets(*idx, q),
                  double(size_t(idx->memory_consumption()) - keys.size() * sizeof(Pair)) / 1e6, build_ms);
    }
  }
  std::printf("test_cht: %s (%zu cases, %zu failed; %zu settings refused as not buildable, as they must be)\n", bad ? "FAILED" : "ok", cases, bad, refused);
  return bad ? 1 : 0;
}
