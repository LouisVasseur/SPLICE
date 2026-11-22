// SPDX-License-Identifier: MIT
// RadixSpline (the authors' code, wrapped in gret/sota/radixspline.h) against a sorted array.
//   model:  for every stored key the bound [lo, hi) holds its position, is inside the array and no wider than 2 * max_error + 2; for keys
//           never stored it is still a valid range; the ends behave as the authors' code says (min -> position 0, max -> n - 1, the
//           same below and above them); the bytes reported are the bytes of the spline and the radix table; duplicates keep the bound
//           on the first position; what cannot be built throws.
//   index:  get on every stored key, on its neighbours, on 0, 1, 2^64 - 1, below and above the range and on random keys; scan from
//           stored, neighbouring, in-gap and extreme starts at several lengths; put, update and remove return false and change nothing;
//           memory_consumption is the pairs plus the model.
//   names:  rs<max_error>r<radix_bits> and nothing else.
// Key sets as in tests/test_scan.cpp (tiny, uniform, wide, dense run, outliers, clusters, stride, huge gap, bit field, keys 0 and
// 2^64 - 1), sizes either side of SortedBoundIndex's 64-key threshold, and three sets of 2M keys. Ends with a small timing, for sanity only.
//   build:  cxx -I <RadixSpline>/include tests/sota/test_radixspline.cpp
//   run:    t [KEYS...]   GRE key files (a uint64 count, then the keys), optional: the same model and index checks on each, at rs32r18,
//           rs1r18 and rs8r24. On a 200M-key file that is every key's bound and get, and about 10 GB of memory.
#include "indexInterface.h"
#include "pgm/pgm.h"  // readonly.h's PgmBound names pgm::PGMIndex
#include "gret/sota/radixspline.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using K = uint64_t;
using Pair = std::pair<K, K>;
using Index = indexInterface<K, K>;
using gret::sota::RsConfig;
using Model = gret::sota::RadixSplineBound<K>;
// indexInterface has no virtual destructor: what make_radixspline returns is deleted as what it is
struct DeleteRs {
  void operator()(Index* p) const { delete static_cast<gret::SortedBoundIndex<K, K, Model>*>(p); }
};
using IndexPtr = std::unique_ptr<Index, DeleteRs>;

static const K M = ~K(0);
static size_t g_fails = 0, g_checks = 0;

static K payload(K k) { return k * 2654435761u + 17; }

#define FAIL(...)                    \
  do {                               \
    if (g_fails++ < 40) {            \
      std::printf("FAIL ");          \
      std::printf(__VA_ARGS__);      \
      std::printf("\n");             \
    }                                \
  } while (0)

static std::string cfg_name(const RsConfig& c) { return "rs" + std::to_string(c.max_error) + "r" + std::to_string(c.radix_bits); }

static std::vector<K> probes(const std::vector<K>& keys, std::mt19937_64& g, size_t random_n) {
  std::vector<K> p = {0, 1, 2, M, M - 1, K(1) << 63, (K(1) << 63) - 1, keys.front() - 1, keys.front() + 1, keys.back() + 1, keys.back() - 1};
  for (size_t i = 0; i < random_n; ++i) {
    p.push_back(g());
    const size_t r = g() % keys.size();
    if (r + 1 < keys.size() && keys[r + 1] - keys[r] > 2) p.push_back(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));  // inside a gap
  }
  return p;
}

// the model alone. Returns its bytes (0 if it cannot be built on these keys)
static size_t check_model(const std::vector<K>& keys, const std::string& label, const RsConfig& c) {
  const std::string what = cfg_name(c) + " model on " + label;
  const size_t n = keys.size(), E = c.max_error;
  ++g_checks;
  Model m(c);
  std::vector<K> copy = keys;
  try {
    m.build(copy);
  } catch (const std::exception& e) {
    FAIL("%s: build threw %s", what.c_str(), e.what());
    return 0;
  }
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    const gret::KeyBound b = m.get(keys[i]);
    if (!(b.lo <= i && i < b.hi && b.hi <= n && b.hi - b.lo <= 2 * E + 2) && bad++ < 3)
      FAIL("%s: key %llu at %zu, bound [%zu, %zu), n %zu", what.c_str(), (unsigned long long)keys[i], i, b.lo, b.hi, n);
  }
  std::mt19937_64 g(3);
  for (const K x : probes(keys, g, std::min<size_t>(n, 20000))) {
    const gret::KeyBound b = m.get(x);
    if (!(b.lo < b.hi && b.hi <= n && b.hi - b.lo <= 2 * E + 2) && bad++ < 3)
      FAIL("%s: probe %llu, bound [%zu, %zu), n %zu", what.c_str(), (unsigned long long)x, b.lo, b.hi, n);
  }
  // the ends, as the authors' GetEstimatedPosition truncates them: min and below -> 0, max and above -> n - 1
  const gret::KeyBound first = m.get(keys.front()), last = m.get(keys.back()), below = m.get(0), above = m.get(M);
  const size_t first_hi = std::min(n, E + 2), last_lo = n - 1 < E ? 0 : n - 1 - E;
  if (first.lo != 0 || first.hi != first_hi || below.lo != 0 || below.hi != first_hi) FAIL("%s: bound at the smallest key [%zu, %zu), at 0 [%zu, %zu)", what.c_str(), first.lo, first.hi, below.lo, below.hi);
  if (last.lo != last_lo || last.hi != n || above.lo != last_lo || above.hi != n) FAIL("%s: bound at the largest key [%zu, %zu), at 2^64-1 [%zu, %zu)", what.c_str(), last.lo, last.hi, above.lo, above.hi);
  // bytes: this object, 4 per radix entry, 16 per spline point; never less than the authors' own count
  const size_t want = sizeof(Model) + m.radix_entries() * sizeof(uint32_t) + m.spline_points() * sizeof(rs::Coord<K>);
  if (m.bytes() != want || m.bytes() < m.m.GetSize()) FAIL("%s: bytes %zu, want %zu, GetSize %zu", what.c_str(), m.bytes(), want, m.m.GetSize());
  if (m.spline_points() < 2 || m.spline_points() > n) FAIL("%s: %zu spline points for %zu keys", what.c_str(), m.spline_points(), n);
  // the table is one entry per prefix of (max - min) plus two (shift as rs::Builder::GetNumShiftBits takes it)
  const K diff = keys.back() - keys.front();
  size_t bits = 0;
  while (bits < 64 && (diff >> bits) > 1) ++bits;
  ++bits;  // bit length of diff
  const size_t shift = bits < c.radix_bits ? 0 : bits - c.radix_bits;
  if (m.radix_entries() != size_t(diff >> shift) + 2) FAIL("%s: %zu radix entries, want %zu", what.c_str(), m.radix_entries(), size_t(diff >> shift) + 2);
  return m.bytes();
}

static size_t check_index(const std::vector<K>& keys, const std::string& label, const RsConfig& c, size_t model_bytes, size_t max_starts) {
  const std::string name = cfg_name(c), what = name + " on " + label;
  const size_t n = keys.size();
  const size_t before = g_fails;
  ++g_checks;
  std::vector<Pair> kv(n);
  for (size_t i = 0; i < n; ++i) kv[i] = {keys[i], payload(keys[i])};
  IndexPtr idx;
  Param p(1, 0);
  try {
    idx.reset(gret::sota::make_radixspline<K, K>(name));
    if (!idx) {
      FAIL("%s: not a name", what.c_str());
      return 1;
    }
    idx->init(&p);
    idx->bulk_load(kv.data(), n, &p);
  } catch (const std::exception& e) {
    FAIL("%s: threw %s", what.c_str(), e.what());
    return 1;
  }
  if (gret::sota::rs_pending().max_error != 32 || gret::sota::rs_pending().radix_bits != 18) FAIL("%s: pending parameters not put back", what.c_str());
  auto stored = [&](K x) { return std::binary_search(keys.begin(), keys.end(), x); };
  size_t bad = 0;
  auto probe = [&](K x) {
    K v = 0x5a5a5a5a;
    const bool got = idx->get(x, v, &p), want = stored(x);
    if ((got != want || (got && v != payload(x)) || (!got && v != 0x5a5a5a5a)) && bad++ < 3) FAIL("%s: get(%llu) = %d, want %d", what.c_str(), (unsigned long long)x, int(got), int(want));
  };
  auto all_gets = [&] {
    for (size_t i = 0; i < n; ++i) {
      K v = 0;
      if (!(idx->get(keys[i], v, &p) && v == payload(keys[i])) && bad++ < 3) FAIL("%s: stored key %llu at %zu not found", what.c_str(), (unsigned long long)keys[i], i);
    }
  };
  all_gets();
  for (size_t i = 0; i < n; ++i) {  // both neighbours of every stored key (stored themselves inside dense runs)
    if (keys[i] != 0) probe(keys[i] - 1);
    if (keys[i] != M) probe(keys[i] + 1);
  }
  std::mt19937_64 g(5);
  for (const K x : probes(keys, g, std::min<size_t>(n, 20000))) probe(x);

  // read-only: every write says no, and nothing changes
  for (size_t i = 0; i < std::min<size_t>(n, 200); ++i) {
    const K k = keys[g() % n], absent = g();
    if (idx->put(k, 1, &p) || idx->put(absent, 1, &p) || idx->update(k, 2, &p) || idx->update(absent, 2, &p) || idx->remove(k, &p) || idx->remove(absent, &p)) {
      if (bad++ < 3) FAIL("%s: a write returned true", what.c_str());
    }
    probe(k), probe(absent);
  }
  if (n <= 200000) all_gets();

  const long long mem = idx->memory_consumption(), want_mem = (long long)(n * sizeof(Pair) + (n < 64 ? 0 : model_bytes));
  if (mem != want_mem) FAIL("%s: memory_consumption %lld, want %lld", what.c_str(), mem, want_mem);

  std::vector<K> starts = {0, 1, M, M - 1, K(1) << 63};
  auto around = [&](K k) {
    starts.push_back(k);
    if (k) starts.push_back(k - 1);
    if (k != M) starts.push_back(k + 1);
  };
  for (size_t i = 0; i < std::min<size_t>(80, n); ++i) around(keys[i]), around(keys[n - 1 - i]);
  for (size_t i = 0; i < max_starts; ++i) {
    const size_t r = g() % n;
    around(keys[r]);
    if (r + 1 < n && keys[r + 1] - keys[r] > 2) starts.push_back(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));
    starts.push_back(g());
  }
  const size_t lens[] = {1, 2, 3, 16, 256, 4096, 65536, n + 5};
  std::vector<Pair> out;
  for (const K x : starts)
    for (const size_t len : lens) {
      if (len > 256 && (g() & 31)) continue;  // long scans on a sample of the starts
      if (len > 65536 && n > 300000) continue;
      out.assign(len + 1, {0xdead, 0xbeef});  // one guard slot past the end
      const size_t got = idx->scan(x, len, out.data(), &p);
      const size_t at = size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin());
      const size_t want_n = std::min(len, n - at);
      bool ok = got == want_n && out[len] == Pair{0xdead, 0xbeef};
      for (size_t i = 0; ok && i < got; ++i) ok = out[i].first == keys[at + i] && out[i].second == payload(keys[at + i]);
      if (!ok && bad++ < 3) FAIL("%s: scan(%llu, %zu) returned %zu, want %zu", what.c_str(), (unsigned long long)x, len, got, want_n);
    }
  return g_fails - before;
}

static void check_names() {
  ++g_checks;
  struct Good { const char* name; size_t e, r; };
  const Good good[] = {{"rs32r18", 32, 18}, {"rs1r1", 1, 1}, {"rs380r12", 380, 12}, {"rs2r28", 2, 28}, {"rs1048576r30", 1048576, 30}, {"rs7r26", 7, 26}};
  for (const Good& gd : good) {
    RsConfig c;
    if (!gret::sota::parse_rs_name(gd.name, c) || c.max_error != gd.e || c.radix_bits != gd.r) FAIL("name %s not parsed", gd.name);
  }
  const char* bad[] = {"", "rs", "rsr", "rs32", "rs32r", "rsr18", "rs0r18", "rs32r0", "rs32r31", "rs032r18", "rs32r018", "rs32r18x", "rs32r18 ", "rs-1r3", "rs1048577r18",
                       "rs99999999r18", "rs32r180", "rs32R18", "RS32r18", "xrs32r18", "rs32r18r4", "rs3.2r18", "rl4", "spgm16", "rs32_18", "rs32r18_alex", "rs+32r18"};
  for (const char* nm : bad) {
    RsConfig c{7, 7};
    if (gret::sota::parse_rs_name(nm, c) || c.max_error != 7 || c.radix_bits != 7) FAIL("name '%s' parsed", nm);
    IndexPtr idx(gret::sota::make_radixspline<K, K>(nm));
    if (idx) FAIL("name '%s' made an index", nm);
  }
  // two indexes made one after the other keep their own parameters; a Model made directly has the defaults
  IndexPtr a(gret::sota::make_radixspline<K, K>("rs1r4")), b(gret::sota::make_radixspline<K, K>("rs64r20"));
  const Model d;
  if (!a || !b || d.cfg.max_error != 32 || d.cfg.radix_bits != 18) FAIL("defaults after make_radixspline: %zu, %zu", d.cfg.max_error, d.cfg.radix_bits);
}

static void check_throws() {
  ++g_checks;
  auto throws = [](const RsConfig& c, std::vector<K> keys) {
    Model m(c);
    try {
      m.build(keys);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  std::vector<K> some;
  for (K i = 0; i < 100; ++i) some.push_back(i * 7 + 3);
  if (!throws(RsConfig{}, {})) FAIL("no keys: built");
  if (!throws(RsConfig{}, {5})) FAIL("one key: built");
  if (!throws(RsConfig{}, {9, 9, 9})) FAIL("one distinct key: built");
  if (!throws(RsConfig{0, 18}, some)) FAIL("max_error 0: built");
  if (!throws(RsConfig{32, 0}, some)) FAIL("radix_bits 0: built");
  if (!throws(RsConfig{32, 31}, some)) FAIL("radix_bits 31: built");
  if (throws(RsConfig{}, {0, M})) FAIL("{0, 2^64-1}: threw");
  if (throws(RsConfig{1, 1}, {0, 1})) FAIL("{0, 1}: threw");
}

// rs::Builder with repeated keys (GRE has none; this pins what the authors' code does with them): the bound holds the first position
static void check_duplicates() {
  ++g_checks;
  std::mt19937_64 g(9);
  std::vector<K> keys;
  for (size_t i = 0; i < 30000; ++i) {
    const K k = g() >> 20;
    for (size_t j = 0, m = 1 + g() % 5; j < m; ++j) keys.push_back(k);
  }
  for (size_t i = 0; i < 500; ++i) keys.push_back(12345);  // one long repeat, far wider than the error
  keys.push_back(0), keys.push_back(0), keys.push_back(M), keys.push_back(M);
  std::sort(keys.begin(), keys.end());
  for (const RsConfig c : {RsConfig{32, 18}, RsConfig{1, 10}, RsConfig{4, 24}}) {
    Model m(c);
    std::vector<K> copy = keys;
    m.build(copy);
    size_t bad = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i && keys[i] == keys[i - 1]) continue;
      const gret::KeyBound b = m.get(keys[i]);
      // the largest key is the one exception: at or above max the authors' code answers n - 1, the last of its repeats
      const size_t pos = keys[i] == keys.back() ? keys.size() - 1 : i;
      if (!(b.lo <= pos && pos < b.hi && b.hi <= keys.size()) && bad++ < 3) FAIL("%s duplicates: key %llu first at %zu, bound [%zu, %zu)", cfg_name(c).c_str(), (unsigned long long)keys[i], i, b.lo, b.hi);
    }
  }
}

static void time_gets(const char* name, Index& idx, const std::vector<Pair>& kv, const std::vector<K>& order) {  // through the interface, as GRE calls it
  std::vector<Pair> copy = kv;
  Param p(1, 0);
  idx.init(&p);
  idx.bulk_load(copy.data(), copy.size(), &p);
  K sum = 0, v = 0;
  size_t lost = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (const K k : order) {
    lost += !idx.get(k, v, &p);
    sum += v;
  }
  const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(order.size());
  if (lost) FAIL("timing %s: %zu keys lost", name, lost);
  std::printf("timing (sanity only) %-9s n=%zu  %.1f ns/get  %.2f bytes/key  checksum %llu\n", name, kv.size(), ns, double(idx.memory_consumption()) / double(kv.size()), (unsigned long long)sum);
}

int main(int argc, char** argv) {
  std::mt19937_64 g(7);
  auto rnd = [&](K lo, K hi, size_t m) {
    std::vector<K> v;
    std::uniform_int_distribution<K> u(lo, hi);
    for (size_t i = 0; i < m; ++i) v.push_back(u(g));
    return v;
  };
  std::vector<std::pair<std::string, std::vector<K>>> sets = {
      {"n1", {5}}, {"n1max", {M}}, {"n1zero", {0}}, {"n2", {0, M}}, {"n3", {1, 2, M}}, {"n5", {0, 1, 2, 3, M}}, {"n17", rnd(0, K(1) << 40, 17)},
      {"n63", rnd(0, M, 63)}, {"n64", rnd(0, M, 64)}, {"n65", rnd(0, K(1) << 20, 65)}, {"n64run", {}}, {"n200ends", rnd(K(1) << 30, K(1) << 31, 198)},
      {"uniform", rnd(1, K(1) << 50, 20000)}, {"wide", rnd(0, M, 20000)}};
  for (auto& s : sets) {
    if (s.first == "n64run") for (K i = 0; i < 64; ++i) s.second.push_back(M - 63 + i);  // the top 64 keys of the domain
    if (s.first == "n200ends") s.second.push_back(0), s.second.push_back(M);
  }
  { std::vector<K> v; for (K i = 0; i < 50000; ++i) v.push_back(1000 + i); sets.push_back({"run", v}); }
  { std::vector<K> v; for (K i = 0; i < 70000; ++i) v.push_back(i); sets.push_back({"run_from_0", v}); }
  { std::vector<K> v = rnd(K(1) << 40, (K(1) << 40) + (K(1) << 34), 20000); for (K j = 0; j < 5; ++j) v.push_back((K(1) << 62) + j * 1000003); v.push_back(3); v.push_back(M); sets.push_back({"outliers", v}); }
  { std::vector<K> v = rnd(K(1) << 40, (K(1) << 40) + 100000, 30000); for (K j = 0; j < 100; ++j) v.push_back(M - j * 1000); sets.push_back({"over64outliers", v}); }
  { std::vector<K> v; for (int c = 0; c < 300; ++c) { const K b = g() >> 1; for (K i = 0; i < 100; ++i) v.push_back(b + i); } sets.push_back({"clusters", v}); }
  { std::vector<K> v; for (K i = 0; i < 20000; ++i) v.push_back(i * 65536 + 12345); sets.push_back({"stride", v}); }
  { std::vector<K> v; for (K i = 0; i < 10000; ++i) { v.push_back(i); v.push_back(M - i); } sets.push_back({"hugegap", v}); }
  { std::vector<K> v; for (K a = 0; a < 300; ++a) for (K b = 0; b < 200; b += 1 + (a & 3)) v.push_back((a << 32) | (b << 8) | 7); v.push_back(M); sets.push_back({"bitfield_max", v}); }
  { std::vector<K> v; K k = 0; for (int i = 0; i < 40000; ++i) { k += 1 + (g() % 16) * (g() % 16) * (g() % 4096); v.push_back(k); } sets.push_back({"steps", v}); }  // a CDF that bends everywhere
  const size_t small_sets = sets.size();
  sets.push_back({"big_wide", rnd(0, M, 2000000)});
  { std::vector<K> v; for (size_t i = 0; i < 2000000; ++i) v.push_back(g() >> (g() % 44)); v.push_back(0); v.push_back(M); sets.push_back({"big_skew", v}); }  // every scale at once
  { std::vector<K> v; K k = g() >> 8; while (v.size() < 2000000) { const size_t run = 1 + g() % 1000; for (size_t i = 0; i < run; ++i) v.push_back(k++); k += g() >> 26; } sets.push_back({"big_runs", v}); }

  check_names();
  check_throws();
  check_duplicates();

  const RsConfig everywhere[] = {{32, 18}, {1, 18}, {1, 1}, {380, 12}, {8, 24}, {100000, 8}, {2, 10}};
  const RsConfig big_only[] = {{32, 18}, {1, 18}, {8, 24}, {380, 12}, {3, 26}};
  for (size_t s = 0; s < sets.size(); ++s) {
    const std::string& label = sets[s].first;
    std::vector<K>& keys = sets[s].second;
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const bool big = s >= small_sets;
    size_t bad = 0, cases = 0;
    auto run = [&](const RsConfig& c) {
      size_t model_bytes = 0;
      if (keys.size() >= 2) model_bytes = check_model(keys, label, c);  // also below SortedBoundIndex's 64 keys: the model itself must hold
      bad += check_index(keys, label, c, model_bytes, big ? 3000 : 1500) != 0;
      ++cases;
    };
    if (big) for (const RsConfig& c : big_only) run(c);
    else for (const RsConfig& c : everywhere) run(c);
    std::printf("%-15s n=%-8zu %zu configurations, %zu failed\n", label.c_str(), keys.size(), cases, bad);
  }

  {  // sanity timing: stored keys in random order on 2M wide keys, beside the static PGM of readonly.h
    const std::vector<K>& keys = sets[small_sets].second;
    std::vector<Pair> kv(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
    std::vector<K> order = keys;
    std::shuffle(order.begin(), order.end(), g);
    for (const char* nm : {"rs32r18", "rs8r24", "rs2r26"}) {
      IndexPtr idx(gret::sota::make_radixspline<K, K>(nm));
      time_gets(nm, *idx, kv, order);
    }
    gret::SortedBoundIndex<K, K, gret::PgmBound<K, 64>> pgm64("spgm64");
    time_gets("spgm64", pgm64, kv, order);
  }

  for (int a = 1; a < argc; ++a) {  // key files
    FILE* f = std::fopen(argv[a], "rb");
    uint64_t n = 0;
    if (!f || std::fread(&n, 8, 1, f) != 1) { std::printf("cannot read %s\n", argv[a]); return 2; }
    std::vector<K> keys(n);
    if (std::fread(keys.data(), 8, n, f) != n) { std::printf("%s: short file\n", argv[a]); return 2; }
    std::fclose(f);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    size_t bad = 0, cases = 0;
    for (const RsConfig& c : {RsConfig{32, 18}, RsConfig{1, 18}, RsConfig{8, 24}}) {
      const size_t model_bytes = keys.size() >= 2 ? check_model(keys, argv[a], c) : 0;
      bad += check_index(keys, argv[a], c, model_bytes, 3000) != 0;
      ++cases;
    }
    std::printf("%s n=%zu %zu configurations, %zu failed\n", argv[a], keys.size(), cases, bad);
  }

  if (g_fails) {
    std::printf("FAILED test_radixspline: %zu failures in %zu checks\n", g_fails, g_checks);
    return 1;
  }
  std::printf("PASS test_radixspline: %zu checks\n", g_checks);
  return 0;
}
