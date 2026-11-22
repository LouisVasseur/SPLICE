// SPDX-License-Identifier: MIT
// FAST (a third-party implementation: Leis, via TLI; wrapped in gret/sota/fast.h) against a sorted array.
//   model:  lower_bound(x) is std::lower_bound for every stored key, both neighbours of every stored key, 0, 1, 2^64 - 1, keys below
//           and above the range, keys inside gaps and random keys; the bytes reported are this object plus the whole mapping, and the
//           mapping is the size the layout needs; what is resident is no more than what is mapped; repeated keys give the first of
//           them; what cannot be built throws (no keys, unsorted keys, 2^30 keys or more, a second build).
//   depths: key counts 2^k - 1, 2^k and 2^k + 1 for k = 1 .. 21, so every tree depth from 1 to 22 and both sides of the step from one
//           page level to two (depth 15 to 16). "deep<k>" on the command line adds 2^k + 12345 keys (depth k + 1), model only:
//           deep27 is depth 28, the depth of 200M keys (about 5.5 GB).
//   index:  get on every stored key, on its neighbours, on the extremes and on random keys; scan from stored, neighbouring, in-gap and
//           extreme starts at several lengths; put, update and remove return false and change nothing; memory_consumption is the pairs
//           plus the model.
//   names:  "fast" and "fast_asis", nothing else.
//   both ways of calling TLI's lower_bound are checked everywhere: on a FAST object local to the call, which lets the compiler fold
//           TLI's per-object layout constants ("fast", FastBound<K, true>, what the model checks call lower_bound), and on the built
//           object itself as TLI's benchmark does ("fast_asis", FastBound<K, false>, m.t.lower_bound in the model checks). Every
//           model check compares the two with the sorted array on the same tree; every index check runs once under each name.
// Key sets as in tests/test_scan.cpp (tiny, uniform, wide, dense run, outliers, clusters, stride, huge gap, bit field, keys 0 and
// 2^64 - 1), sizes either side of SortedBoundIndex's 64-key threshold, and three sets of 2M keys. Ends with a small timing, for sanity only.
// In a build without AVX2 (arm64, or x86-64 without -mavx2) there is no FAST: the test then checks that make_fast("fast") throws and
// that every other name gives nullptr, and says that it checked nothing else.
//   build:  cxx -I <the directory that holds TLI> tests/sota/test_fast.cpp     (on the Mac: add -arch x86_64 -mavx2, runs under Rosetta)
//           -DGRET_FAST_TLI_UTIL takes the two unsigned compares from TLI's util.h instead of fast.h's own
//   run:    t [deep<k>...] [KEYS...]   GRE key files (a uint64 count, then the keys), optional: the same model and index checks on
//           each. On a 200M-key file that is about 14 GB of memory.
#include "indexInterface.h"
#include "pgm/pgm.h"  // readonly.h's PgmBound names pgm::PGMIndex
#include "gret/sota/fast.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using K = uint64_t;
using Pair = std::pair<K, K>;
using Index = indexInterface<K, K>;

static const K M = ~K(0);
static size_t g_fails = 0, g_checks = 0;

#define FAIL(...)                    \
  do {                               \
    if (g_fails++ < 40) {            \
      std::printf("FAIL ");          \
      std::printf(__VA_ARGS__);      \
      std::printf("\n");             \
    }                                \
  } while (0)

static void check_names_without_fast() {
  ++g_checks;
  for (const char* nm : {"", "fas", "fast1", "fast64", "FAST", "fast ", " fast", "fastt", "ht_fast", "fast_alex", "rs32r18", "spgm64", "fast_asi", "fast_asis2", "fast_asis ", "fastasis", "fast_as_is", "fast_fold"}) {
    Index* i = nullptr;
    try {
      i = gret::sota::make_fast<K, K>(nm);
    } catch (const std::exception& e) {
      FAIL("name '%s' threw %s", nm, e.what());
    }
    if (i) FAIL("name '%s' made an index", nm);
  }
}

#if !GRET_FAST_AVAILABLE

int main() {
  check_names_without_fast();
  ++g_checks;
  bool threw = false;
  try {
    (void)gret::sota::make_fast<K, K>("fast");
  } catch (const std::runtime_error& e) {
    threw = true;
    std::printf("make_fast(\"fast\") throws: %s\n", e.what());
  }
  if (!threw) FAIL("make_fast(\"fast\") did not throw in a build without AVX2");
  threw = false;
  try {
    (void)gret::sota::make_fast<K, K>("fast_asis");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  if (!threw) FAIL("make_fast(\"fast_asis\") did not throw in a build without AVX2");
  if (gret::sota::kFastAvailable) FAIL("kFastAvailable is true");
  if (g_fails) {
    std::printf("FAILED test_fast: %zu failures in %zu checks\n", g_fails, g_checks);
    return 1;
  }
  std::printf("PASS test_fast: %zu checks; NO AVX2 IN THIS BUILD, so FAST itself was not built and not tested\n", g_checks);
  return 0;
}

#else

template <bool Fold>
using ModelT = gret::sota::FastBound<K, Fold>;
using Model = ModelT<true>;  // lower_bound() is the folded call; t.lower_bound() on the same tree is the as-is call
static_assert(std::is_same<Model, gret::sota::FastBound<K>>::value, "the default is the folded lookup");
static_assert(sizeof(ModelT<true>) == sizeof(ModelT<false>), "both models report the same bytes");
// indexInterface has no virtual destructor: what make_fast returns is deleted as what it is
template <bool Fold>
struct DeleteFast {
  void operator()(Index* p) const { delete static_cast<gret::SortedBoundIndex<K, K, ModelT<Fold>>*>(p); }
};
template <bool Fold>
using IndexPtr = std::unique_ptr<Index, DeleteFast<Fold>>;
template <bool Fold>
static const char* name_of() { return Fold ? "fast" : "fast_asis"; }

static K payload(K k) { return k * 2654435761u + 17; }

static std::vector<K> probes(const std::vector<K>& keys, std::mt19937_64& g, size_t random_n) {
  std::vector<K> p = {0, 1, 2, M, M - 1, K(1) << 63, (K(1) << 63) - 1, (K(1) << 63) + 1, K(1) << 62, K(1) << 32, (K(1) << 32) - 1,
                      keys.front() - 1, keys.front() + 1, keys.back() + 1, keys.back() - 1};
  for (size_t i = 0; i < random_n; ++i) {
    p.push_back(g());
    p.push_back(g() >> (g() % 64));
    const size_t r = g() % keys.size();
    if (r + 1 < keys.size() && keys[r + 1] - keys[r] > 2) p.push_back(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));  // inside a gap
  }
  return p;
}

// the layout's size, written out again here on purpose (fast.h has its own copy)
static size_t want_mapped(size_t n) {
  unsigned d = 0;
  while ((n >> d) != 0) ++d;  // floor(log2 n) + 1
  size_t slots = size_t(1) << (d + 1);
  if (d > 15) slots += size_t(1) << 18;
  return slots * 8;
}

// the model alone: every stored key (or one in `step`), its neighbours, the probes. Returns its bytes (0 if it cannot be built)
static size_t check_model(const std::vector<K>& keys, const std::string& label, size_t step = 1) {
  const std::string what = "fast model on " + label;
  const size_t n = keys.size();
  ++g_checks;
  Model m;
  std::vector<K> copy = keys;
  try {
    m.build(copy);
  } catch (const std::exception& e) {
    FAIL("%s: build threw %s", what.c_str(), e.what());
    return 0;
  }
  if (copy != keys) FAIL("%s: the build changed the keys", what.c_str());
  size_t bad = 0;
  auto at = [&](K x, size_t want) {
    const size_t got = m.lower_bound(x), asis = m.t.lower_bound(x);  // folded, and TLI's own call on the built object
    const gret::KeyBound b = m.get(x);
    if ((got != want || asis != want || b.lo != want || b.hi != want) && bad++ < 3)
      FAIL("%s: lower_bound(%llu) = %zu folded, %zu as is, bound [%zu, %zu), want %zu of %zu", what.c_str(), (unsigned long long)x, got, asis, b.lo, b.hi, want, n);
  };
  for (size_t i = 0; i < n; i += step) {
    at(keys[i], i);
    if (keys[i] != 0) at(keys[i] - 1, (i > 0 && keys[i - 1] == keys[i] - 1) ? i - 1 : i);
    if (keys[i] != M) at(keys[i] + 1, i + 1);
  }
  at(keys.front(), 0), at(keys.back(), n - 1);
  std::mt19937_64 g(3);
  for (const K x : probes(keys, g, std::min<size_t>(n, 20000))) at(x, size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin()));
  // bytes: this object and the whole mapping, which is what TLI's size_in_byte() says and what the layout needs
  const size_t mapped = want_mapped(n);
  if (m.t.size_in_byte() != mapped || m.bytes() != sizeof(Model) + mapped || gret::sota::fast_mapped_bytes(n) != mapped)
    FAIL("%s: size_in_byte %llu, bytes %zu, want mapping %zu", what.c_str(), (unsigned long long)m.t.size_in_byte(), m.bytes(), mapped);
  if (m.resident > mapped + 65536) FAIL("%s: %zu bytes resident of %zu mapped", what.c_str(), m.resident, mapped);
#ifdef __linux__
  if (m.resident < 8 * n / 2) FAIL("%s: only %zu bytes resident for %zu keys", what.c_str(), m.resident, n);
  if (m.huge < 0 || size_t(m.huge) > mapped + (size_t(4) << 20)) FAIL("%s: %lld bytes on huge pages of %zu mapped", what.c_str(), m.huge, mapped);
#endif
  if (gret::sota::fast_depth(n) * 1u != unsigned(std::floor(std::log2(double(n))) + 1) || m.depth != gret::sota::fast_depth(n) || m.n != n || m.slots == nullptr)
    FAIL("%s: depth %u (the tree's: %u)", what.c_str(), gret::sota::fast_depth(n), m.depth);
  return m.bytes();
}

template <bool Fold>
static size_t check_index(const std::vector<K>& keys, const std::string& label, size_t model_bytes, size_t max_starts) {
  const std::string what = std::string(name_of<Fold>()) + " on " + label;
  const size_t n = keys.size();
  const size_t before = g_fails;
  ++g_checks;
  std::vector<Pair> kv(n);
  for (size_t i = 0; i < n; ++i) kv[i] = {keys[i], payload(keys[i])};
  IndexPtr<Fold> idx;
  Param p(1, 0);
  try {
    idx.reset(gret::sota::make_fast<K, K>(name_of<Fold>()));
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
  check_names_without_fast();
  ++g_checks;
  IndexPtr<true> a(gret::sota::make_fast<K, K>("fast"));
  if (!a) FAIL("name 'fast' made no index");
  IndexPtr<false> b(gret::sota::make_fast<K, K>("fast_asis"));
  if (!b) FAIL("name 'fast_asis' made no index");
  if (a && b && (dynamic_cast<gret::SortedBoundIndex<K, K, ModelT<true>>*>(a.get()) == nullptr || dynamic_cast<gret::SortedBoundIndex<K, K, ModelT<false>>*>(b.get()) == nullptr))
    FAIL("the two names made the wrong models");
  if (!gret::sota::kFastAvailable) FAIL("kFastAvailable is false");
}

static void check_throws() {
  ++g_checks;
  auto throws = [](std::vector<K> keys) {
    Model m;
    try {
      m.build(keys);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  if (!throws({})) FAIL("no keys: built");
  if (!throws({5, 4})) FAIL("unsorted keys: built");
  if (!throws({1, 2, 3, 9, 8, 10})) FAIL("unsorted keys: built");
  if (throws({5})) FAIL("one key: threw");
  if (throws({0, M})) FAIL("{0, 2^64-1}: threw");
  {  // a second build
    Model m;
    std::vector<K> keys = {1, 2, 3};
    m.build(keys);
    bool threw = false;
    try {
      m.build(keys);
    } catch (const std::logic_error&) {
      threw = true;
    }
    if (!threw) FAIL("a second build did not throw");
    if (m.lower_bound(2) != 1 || m.lower_bound(4) != 3 || m.t.lower_bound(2) != 1 || m.t.lower_bound(4) != 3) FAIL("the model after a refused second build");
  }
  // the limit, without building anything that large
  auto len_throws = [](size_t n) {
    try {
      gret::sota::fast_check_len(n);
    } catch (const std::length_error&) {
      return true;
    }
    return false;
  };
  if (len_throws((size_t(1) << 30) - 1) || !len_throws(size_t(1) << 30) || !len_throws(size_t(1) << 31)) FAIL("the 2^30 limit");
  if (gret::sota::fast_mapped_bytes(200000000) != 8 * ((size_t(1) << 29) + (size_t(1) << 18))) FAIL("mapping for 200M keys");
}

// repeated keys (GRE has none; this pins what the third-party code does with them): the first position of the key
static void check_duplicates() {
  ++g_checks;
  std::mt19937_64 g(9);
  std::vector<K> keys;
  for (size_t i = 0; i < 30000; ++i) {
    const K k = g() >> (g() % 3 ? 20 : 0);
    for (size_t j = 0, m = 1 + g() % 5; j < m; ++j) keys.push_back(k);
  }
  for (size_t i = 0; i < 500; ++i) keys.push_back(12345);
  keys.push_back(0), keys.push_back(0), keys.push_back(M), keys.push_back(M), keys.push_back(M);
  std::sort(keys.begin(), keys.end());
  Model m;
  std::vector<K> copy = keys;
  m.build(copy);
  size_t bad = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    const size_t want = size_t(std::lower_bound(keys.begin(), keys.end(), keys[i]) - keys.begin());
    if ((m.lower_bound(keys[i]) != want || m.t.lower_bound(keys[i]) != want) && bad++ < 3) FAIL("duplicates: key %llu at %zu, lower_bound %zu, want %zu", (unsigned long long)keys[i], i, m.lower_bound(keys[i]), want);
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

// n ascending keys without a sort: a walk from 2^62 with steps that take it across 2^63, the bit the unsigned compare is about
static std::vector<K> walk(size_t n, std::mt19937_64& g) {
  std::vector<K> v(n);
  const K span = (K(1) << 63) / n * 2;
  K k = K(1) << 62;
  for (size_t i = 0; i < n; ++i) {
    k += 1 + g() % span;
    v[i] = k;
  }
  return v;
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
  { std::vector<K> v; for (K i = 0; i < 40000; ++i) v.push_back((K(1) << 63) - 20000 + i); sets.push_back({"run_over_2^63", v}); }
  { std::vector<K> v; for (K i = 0; i < 33000; ++i) v.push_back(M - i); sets.push_back({"run_to_max", v}); }
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

  for (size_t s = 0; s < sets.size(); ++s) {
    const std::string& label = sets[s].first;
    std::vector<K>& keys = sets[s].second;
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const bool big = s >= small_sets;
    const size_t before = g_fails;
    const size_t model_bytes = check_model(keys, label);  // also below SortedBoundIndex's 64 keys: the tree itself must hold
    check_index<true>(keys, label, model_bytes, big ? 3000 : 1500);
    check_index<false>(keys, label, model_bytes, big ? 3000 : 1500);
    std::printf("%-15s n=%-8zu depth %-2u %s\n", label.c_str(), keys.size(), gret::sota::fast_depth(keys.size()), g_fails == before ? "ok" : "FAILED");
  }

  {  // every depth from 1 to 22: 2^k - 1, 2^k, 2^k + 1 keys, of three kinds in turn (wide random, a walk across 2^63, dense up to 2^64 - 1)
    size_t cases = 0, bad = 0;
    for (unsigned k = 1; k <= 21; ++k)
      for (const size_t n : {(size_t(1) << k) - 1, size_t(1) << k, (size_t(1) << k) + 1}) {
        std::vector<K> keys;
        const unsigned kind = unsigned(cases % 3);
        if (kind == 0) {
          keys = rnd(0, M, n + n / 8 + 8);
          std::sort(keys.begin(), keys.end());
          keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
          keys.resize(n);
        } else if (kind == 1) {
          keys = walk(n, g);
        } else {
          keys.resize(n);
          for (size_t i = 0; i < n; ++i) keys[i] = M - (n - 1 - i);
        }
        const size_t before = g_fails;
        const std::string label = "n=" + std::to_string(n) + (kind == 0 ? " wide" : kind == 1 ? " walk" : " dense_to_max");
        const size_t model_bytes = check_model(keys, label);
        if (k <= 17 || n == (size_t(1) << k)) check_index<true>(keys, label, model_bytes, 300), check_index<false>(keys, label, model_bytes, 300);
        bad += g_fails != before;
        ++cases;
      }
    std::printf("depths 1 to 22: %zu key counts around powers of two, %zu failed\n", cases, bad);
  }

  {  // sanity timing: stored keys in random order on 2M wide keys, beside the static PGM of readonly.h
    const std::vector<K>& keys = sets[small_sets].second;
    std::vector<Pair> kv(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
    std::vector<K> order = keys;
    std::shuffle(order.begin(), order.end(), g);
    {
      IndexPtr<true> idx(gret::sota::make_fast<K, K>("fast"));
      time_gets("fast", *idx, kv, order);
    }
    {
      IndexPtr<false> idx(gret::sota::make_fast<K, K>("fast_asis"));
      time_gets("fast_asis", *idx, kv, order);
    }
    gret::SortedBoundIndex<K, K, gret::PgmBound<K, 64>> pgm64("spgm64");
    time_gets("spgm64", pgm64, kv, order);
  }

  for (int a = 1; a < argc; ++a) {
    const std::string arg = argv[a];
    if (arg.compare(0, 4, "deep") == 0) {  // 2^k + 12345 keys, depth k + 1, the model only: every key and both its neighbours
      const unsigned k = unsigned(std::atoi(arg.c_str() + 4));
      if (k < 10 || k > 29) { std::printf("%s: deep<k> takes k in [10, 29]\n", arg.c_str()); return 2; }
      const size_t n = (size_t(1) << k) + 12345;
      const std::vector<K> keys = walk(n, g);
      const size_t before = g_fails;
      const auto t0 = std::chrono::steady_clock::now();
      const size_t bytes = check_model(keys, arg);
      std::printf("%-15s n=%-10zu depth %-2u %.2f tree bytes/key, %.0f s  %s\n", arg.c_str(), n, gret::sota::fast_depth(n), double(bytes) / double(n),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), g_fails == before ? "ok" : "FAILED");
      {  // sanity timing at this depth: the tree's lower bound beside std::lower_bound on the same keys, 2M stored keys in random order
        Model m;
        std::vector<K> copy = keys;
        m.build(copy);
        std::vector<K> order(2000000);
        for (K& x : order) x = keys[g() % n];
        size_t s0 = 0, s1 = 0, s2 = 0;
        const auto a0 = std::chrono::steady_clock::now();
        for (const K x : order) s1 += m.lower_bound(x);
        const auto a1 = std::chrono::steady_clock::now();
        for (const K x : order) s2 += size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin());
        const auto a2 = std::chrono::steady_clock::now();
        for (const K x : order) s0 += m.t.lower_bound(x);
        const auto a3 = std::chrono::steady_clock::now();
        if (s1 != s2 || s0 != s2) FAIL("%s: timing checksums differ", arg.c_str());
        std::printf("timing (sanity only) %s: tree lower bound %.1f ns folded (fast), %.1f ns as is (fast_asis), std::lower_bound on the keys %.1f ns\n", arg.c_str(),
                    std::chrono::duration<double, std::nano>(a1 - a0).count() / double(order.size()), std::chrono::duration<double, std::nano>(a3 - a2).count() / double(order.size()),
                    std::chrono::duration<double, std::nano>(a2 - a1).count() / double(order.size()));
      }
      continue;
    }
    FILE* f = std::fopen(argv[a], "rb");  // a key file
    uint64_t n = 0;
    if (!f || std::fread(&n, 8, 1, f) != 1) { std::printf("cannot read %s\n", argv[a]); return 2; }
    std::vector<K> keys(n);
    if (std::fread(keys.data(), 8, n, f) != n) { std::printf("%s: short file\n", argv[a]); return 2; }
    std::fclose(f);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const size_t before = g_fails;
    const size_t model_bytes = check_model(keys, argv[a]);
    check_index<true>(keys, argv[a], model_bytes, 3000);
    check_index<false>(keys, argv[a], model_bytes, 3000);
    std::printf("%s n=%zu depth %u %s\n", argv[a], keys.size(), gret::sota::fast_depth(keys.size()), g_fails == before ? "ok" : "FAILED");
  }

  if (g_fails) {
    std::printf("FAILED test_fast: %zu failures in %zu checks\n", g_fails, g_checks);
    return 1;
  }
  std::printf("PASS test_fast: %zu checks, compares from %s\n", g_checks,
#ifdef GRET_FAST_TLI_UTIL
              "TLI's util.h"
#else
              "fast.h's shim"
#endif
  );
  return 0;
}

#endif  // GRET_FAST_AVAILABLE
