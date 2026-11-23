// SPDX-License-Identifier: MIT
// bli (include/gret/sota/bli.h: the BLI authors' code, unmodified, behind GRE's interface and behind guards) against a std::map:
// get on every stored key and on keys never stored (neighbours, 0, 1, 2^63, 2^64 - 1, in the gaps, random), scan from stored,
// neighbouring, in-between and extreme start keys, then put (neighbours of stored keys, runs, keys below and above everything,
// the special keys, duplicates), update (stored and absent keys), remove (must return false and change nothing), then two
// things that make the authors' structure grow: keys appended in ascending order above the largest key and a dense run put
// into one gap. After each step every answer is checked again and the wrapper's audit looks every stored key up, checks that
// it lies in the key range of its D-Bucket and counts the pairs. Key sets: test_scan's patterns (tiny sets, uniform, wide,
// dense runs, clusters, far outliers, keys 0 and 2^64 - 1, a regular stride), a set with a run down from 2^64 - 1, lognormal
// gaps, small keys with a few very large ones, and random sets of 2M keys; on D-Buckets of 2^8 to 2^20 slots, S-Buckets of 4,
// 8 and 16, four fill ratios, and with the unchecked put (names ending in u, which get no duplicate puts). Also:
//   names         the name parser
//   tiny          an index never loaded, an empty bulk load
//   refuse        D-Bucket sizes too small for large keys must refuse them (bulk_load and put throw std::invalid_argument)
//                 and stay usable
//   truncation    the guard against the authors' 32-bit truncation of an S-Bucket number (segment.h:300): a key their walk
//                 misroutes is constructed; the checked put must refuse it, the unchecked put misplaces it and the audit must
//                 say so, and loaded with the data it must be served although the authors' lookup misses it
//   disorder      the guard against a pivot the authors' split misplaces (segment.h:255, the same truncation): a D-Bucket
//                 whose upper half are keys the authors' walk misroutes is loaded and filled until it splits; the checked
//                 put must throw std::runtime_error (the authors' scan would return keys out of order from then on), and
//                 with the unchecked put the audit and memory_consumption must say so
//   duplicate     the unchecked put given a stored key
//   rootoverflow  NOT in the default run: makes the authors' new root write behind its array (segment.h:98) and expects the
//                 wrapper's std::runtime_error. See check_root_overflow.
// memory_consumption must equal the authors' mem_size() plus the wrapper object.
//
// Build (docs/sota/bli.md; BLI = the clone of https://github.com/gdymind/Buck-Learned-Index, B = build/tests/sota/bli):
//   Linux x86-64 (the benchmark machine):
//     g++ -std=gnu++17 -O3 -march=native ... -isystem $BLI/include -isystem $BLI/src tests/sota/test_bli.cpp -o t
//   macOS arm64, where <immintrin.h> does not exist: a stand-in with the types and declarations bucket.h names (none of them
//   is called: the released index does not use its SIMD lookup), then the test compilers as generated:
//     mkdir -p $B/armshim && cat > $B/armshim/immintrin.h <<'EOF'
//     #pragma once
//     #include <cstdlib>
//     typedef long long __m256i __attribute__((__vector_size__(32), __aligned__(32)));
//     typedef float __m256 __attribute__((__vector_size__(32), __aligned__(32)));
//     typedef double __m256d __attribute__((__vector_size__(32), __aligned__(32)));
//     static inline int _mm256_extract_epi32(__m256i, int) { std::abort(); }
//     static inline __m256i _mm256_cmpeq_epi32(__m256i, __m256i) { std::abort(); }
//     static inline __m256i _mm256_cmpeq_epi64(__m256i, __m256i) { std::abort(); }
//     static inline int _mm256_movemask_ps(__m256) { std::abort(); }
//     static inline int _mm256_movemask_pd(__m256d) { std::abort(); }
//     static inline __m256i _mm256_loadu_si256(const __m256i*) { std::abort(); }
//     static inline __m256i _mm256_set1_epi32(int) { std::abort(); }
//     static inline __m256i _mm256_set1_epi64x(long long) { std::abort(); }
//     #define __builtin_ia32_rdtsc() 0LL  /* tscns.h, a timer the release build does not start */
//     EOF
//     build/tests/cxx    -DGRET_BLI_SHIM -I $B/armshim -I $BLI/include -I $BLI/src tests/sota/test_bli.cpp -o $B/t
//     build/tests/cxx-O3 -DGRET_BLI_SHIM -I $B/armshim -I $BLI/include -I $BLI/src tests/sota/test_bli.cpp -o $B/t-O3
//   macOS, x86-64 with AVX2 under Rosetta (the real <immintrin.h>). The generated compilers pass -mcpu=native, which this
//   target refuses, and -I<GRE>/build/mac/include, whose x86intrin.h is a stand-in for arm64 that clashes with the real one:
//   their other flags are given by hand.
//     clang++ -arch x86_64 -mavx2 -std=gnu++17 -g -O3 -DNDEBUG -faligned-new -include cstdint \
//       -D_LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE "-D__float128=long double" -Iinclude \
//       -I<GRE>/src/competitor -I<GRE>/src -I $BLI/include -I $BLI/src tests/sota/test_bli.cpp -o $B/t-x86-O3
//     (and with -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all in place of -O3 -DNDEBUG)
//
// -DTEST_BLI_ALONE leaves GRE's own indexes out (they are here to show the header lives next to them, and for --time).
//
//   t [SET...] [--time]     only the named sets (names tiny uniform run run0 clusters stride skew wide outliers hugegap fblike
//                           refuse random2m truncation disorder duplicate; rootoverflow only by name); --time: a small lookup and
//                           insert timing on 3M uniform keys (use an -O3 build)
//   TEST_BLI_SEED=<n>       another draw of every random key set and of the operations
#include "indexInterface.h"
#ifndef TEST_BLI_ALONE  // GRE's own indexes first, as GRE's competitor.h has them: the new header must live next to them
#include "alex/alex.h"
#include "btree/btree.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#endif
#include "gret/sota/bli.h"

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

#if !GRET_SOTA_HAVE_BLI
// a build without AVX2 or without the authors' sources: the header must be harmless, and say so when asked for the index
int main() {
  bool threw = false;
  try {
    gret::make_bli<uint64_t, uint64_t>("bli");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  const bool other = gret::make_bli<uint64_t, uint64_t>("alex") == nullptr;
  std::printf(threw && other ? "test_bli: built without BLI; make_bli says so: all passed\n" : "test_bli: FAILURE: make_bli in a build without BLI\n");
  return threw && other ? 0 : 1;
}
#else

// the authors' util.h says "using namespace std" and their headers define common names: none of it may reach this file
template <class KEY_TYPE, class PAYLOAD_TYPE>
static size_t names_are_free() { return sizeof(KEY_TYPE) + sizeof(PAYLOAD_TYPE); }
struct vector {};  // would be ambiguous with std::vector had the directive leaked
struct Bucket {};
struct Segment {};
#ifdef HINT_MURMUR_HASH
#error "bli.h leaked HINT_MURMUR_HASH"
#endif
#ifdef private
#error "bli.h leaked its definition of private"
#endif

using Index = indexInterface<uint64_t, uint64_t>;
using Map = std::map<uint64_t, uint64_t>;
static const uint64_t M = ~uint64_t(0);
static const uint64_t H = uint64_t(1) << 63;

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

static size_t g_bad = 0;
static double g_phase[6];  // seconds in: bulk_load, verify, audit, put, update and remove, the growing puts
struct Phase {
  int i;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  explicit Phase(int which) : i(which) {}
  ~Phase() { g_phase[i] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
static void fail(const std::string& what, const char* op, uint64_t k) {
  if (g_bad++ < 40) std::printf("FAIL %s: %s wrong at key %llu\n", what.c_str(), op, (unsigned long long)k);
}

#ifdef NDEBUG
static const double kWork = 1.0;
#else
static const double kWork = 0.3;  // the sanitized build: fewer of the probes that cost a whole D-Bucket each
#endif

// every answer against the reference: get on every key it holds and on absent keys, scans. slots: the D-Bucket size, which is
// what a lookup of an absent key and a scan cost in the authors' code; the number of those is scaled to it.
static void verify(Index& idx, const Map& ref, const std::vector<uint64_t>& probes, std::mt19937_64& g, const std::string& what, size_t slots) {
  std::vector<std::pair<uint64_t, uint64_t>> all(ref.begin(), ref.end());
  Phase phase(1);
  const size_t negs = size_t(kWork * std::min(1500.0, std::max(16.0, 3e6 / double(slots))));
  const size_t nscan = size_t(kWork * std::min(100.0, std::max(6.0, 4e5 / double(slots))));
  uint64_t v = 0;
  for (const auto& kv : all)
    if (!idx.get(kv.first, v) || v != kv.second) fail(what, "get", kv.first);
  auto probe = [&](uint64_t k) {
    uint64_t val = 0;
    const bool got = idx.get(k, val);
    const auto it = ref.find(k);
    if (got != (it != ref.end()) || (got && val != it->second)) fail(what, "get (probe)", k);
  };
  for (size_t i = 0; i < probes.size() && i < negs; ++i) {
    const uint64_t k = probes[probes.size() > negs ? g() % probes.size() : i];
    probe(k), probe(k + 1), probe(k - 1);
  }
  for (uint64_t k : {uint64_t(0), uint64_t(1), uint64_t(2), M, M - 1, H, H - 1, H + 1}) probe(k);
  for (size_t i = 0; i < negs / 4; ++i) probe(g());
  for (size_t i = 0; i < negs && !all.empty(); ++i) {  // in the gaps and next to stored keys
    const size_t r = g() % all.size();
    probe(all[r].first + 1), probe(all[r].first - 1);
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) probe(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  std::vector<uint64_t> starts = {0, 1, M, M - 1, H, H - 1, H + 1};
  auto around = [&](uint64_t k) { starts.push_back(k), starts.push_back(k + 1), starts.push_back(k - 1); };
  for (size_t i = 0; i < std::min<size_t>(6, all.size()); ++i) around(all[i].first), around(all[all.size() - 1 - i].first);
  for (size_t i = 0; i < nscan && !all.empty(); ++i) {
    const size_t r = g() % all.size();
    around(all[r].first);
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) starts.push_back(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  for (size_t i = 0; i < probes.size() && i < nscan / 4; ++i) around(probes[g() % probes.size()]);
  for (size_t i = 0; i < nscan / 8; ++i) starts.push_back(g());
  const size_t lens[] = {1, 2, 3, 16, 100, 5000, 70000};
  std::vector<std::pair<uint64_t, uint64_t>> out;
  auto less = [](const std::pair<uint64_t, uint64_t>& a, uint64_t b) { return a.first < b; };
  for (uint64_t s : starts)
    for (size_t len : lens) {
      if (len > 3 && (g() & 3)) continue;
      if (len > 100 && (g() & 7)) continue;  // long scans on a sample of the starts
      out.assign(len + 1, {0, 0});
      const size_t got = idx.scan(s, len, out.data());
      const size_t at = size_t(std::lower_bound(all.begin(), all.end(), s, less) - all.begin());
      bool ok = got == std::min(len, all.size() - at) && out[len] == std::pair<uint64_t, uint64_t>(0, 0);
      for (size_t i = 0; ok && i < got; ++i) ok = out[i] == all[at + i];
      if (!ok) fail(what, "scan", s);
    }
  {  // one scan over everything, and a scan of nothing
    out.assign(all.size() + 7, {0, 0});
    const size_t got = idx.scan(0, all.size() + 7, out.data());
    bool ok = got == all.size();
    for (size_t i = 0; ok && i < got; ++i) ok = out[i] == all[i];
    if (!ok) fail(what, "full scan", 0);
    if (idx.scan(0, 0, out.data()) != 0) fail(what, "scan of zero keys", 0);
  }
}

struct Expect {
  bool refuses = false;  // bulk_load must throw std::invalid_argument (keys too large for this D-Bucket size)
  bool grow = true;      // the two insert runs that split buckets
};

// S, L: what the name must give (S-Bucket size, log2 of the D-Bucket size)
template <size_t S, int L>
static void check(const std::string& label, std::vector<uint64_t> keys, const char* name, const Expect& ex = Expect()) {
  using I = gret::BliIndex<uint64_t, uint64_t, S, size_t(1) << L>;
  const size_t slots = size_t(1) << L;
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  const std::string what = std::string(name) + " on " + label;
  const size_t bad0 = g_bad;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::pair<uint64_t, uint64_t>> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  Index* idx = gret::make_bli<uint64_t, uint64_t>(name);
  if (!idx) { std::printf("FAIL %s: no such index\n", name); ++g_bad; return; }
  I* d = dynamic_cast<I*>(idx);
  if (!d) { std::printf("FAIL %s: the name does not give S-Buckets of %zu and D-Buckets of 2^%d\n", name, S, L); ++g_bad; return; }
  std::unique_ptr<I> owner(d);  // deleted as what it is: indexInterface has no virtual destructor
  const bool checked = name[std::strlen(name) - 1] != 'u';
  const uint64_t safe = d->safe_key();
  auto holds = [&](uint64_t k) { return k <= safe || k == M; };  // what this index takes
  Param p(1, 0);
  idx->init(&p);
  if (ex.refuses) {
    for (int twice = 0; twice < 2; ++twice) {  // throws, and can be asked again
      bool threw = false;
      try {
        idx->bulk_load(kv.data(), kv.size(), &p);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (!threw) fail(what, "bulk_load did not throw std::invalid_argument, keys", keys.size());
      if (d->size() || d->levels()) fail(what, "bulk_load threw but kept keys:", d->size());
    }
    // and still an index: small keys go in, a large one is refused again
    Map ref;
    std::mt19937_64 g(5);
    for (int i = 0; i < 3000; ++i) { const uint64_t k = g() % (safe / 2) + 1; if (idx->put(k, k ^ 3) != ref.emplace(k, k ^ 3).second) fail(what, "put (after the refusal)", k); }
    bool threw = false;
    try {
      idx->put(safe + 1, 1);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    if (!threw) fail(what, "put of a key above safe_key() did not throw", safe + 1);
    std::vector<uint64_t> none;
    verify(*idx, ref, none, g, what + " (after the refusal)", slots);
    std::printf("%-16s %-12s n=%-8zu refuses keys above %llu, as it must, and stays usable  %s\n", name, label.c_str(), keys.size(), (unsigned long long)safe, g_bad == bad0 ? "ok" : "FAILED");
    return;
  }
  for (double& x : g_phase) x = 0;
  try {
    Phase phase(0);
    idx->bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::exception& e) {
    std::printf("FAIL %s: bulk_load threw %s\n", what.c_str(), e.what());
    ++g_bad;
    return;
  }
  Map ref(kv.begin(), kv.end());
  auto audit = [&](const char* when) {
    Phase phase(2);
    const auto a = d->audit(true);
    const size_t expect = d->levels() ? ref.size() - ref.count(M) + size_t(d->key0_is_dummy()) : 0;
    if (a.wrong) fail(what + " (" + when + ")", "audit: lookups or key ranges wrong:", a.wrong);
    if (a.disordered) fail(what + " (" + when + ")", "audit: Segments out of key order:", a.disordered);
    if (a.keys != expect) fail(what + " (" + when + ")", "audit: the buckets hold another number of pairs than", expect);
    if (d->pairs() != expect) fail(what + " (" + when + ")", "pairs() is not", expect);
    if (d->refused_puts()) fail(what + " (" + when + ")", "puts were refused:", d->refused_puts());
    if (d->size() != ref.size()) fail(what + " (" + when + ")", "size() is not", ref.size());
    if (d->key0_is_dummy() != (d->levels() && !ref.count(0))) fail(what + " (" + when + ")", "the key 0 flag is wrong; keys:", ref.size());
    return a;
  };
  const auto a0 = audit("loaded");
  const unsigned long long levels0 = d->levels();
  const long long bytes = idx->memory_consumption();
  if (!keys.empty() && bytes < (long long)(16 * (keys.size() - ref.count(M)))) fail(what, "memory_consumption below 16 bytes a key:", uint64_t(bytes));
  if (size_t(bytes) != d->authors_mem_size() + sizeof(I)) fail(what, "memory_consumption is not the authors' mem_size() plus the wrapper:", uint64_t(bytes));

  const char* seed = std::getenv("TEST_BLI_SEED");
  std::mt19937_64 g(std::hash<std::string>()(label) ^ 0x9e3779b97f4a7c15ull ^ (seed && *seed ? std::strtoull(seed, nullptr, 10) * 0x100000001b3ull : 0));
  std::vector<uint64_t> probes;
  verify(*idx, ref, probes, g, what + " (loaded)", slots);

  const uint64_t lo = keys.empty() ? 1000 : keys.front(), hi = keys.empty() ? 1000000 : keys.back();
  auto pick = [&]() { return keys.empty() ? g() % 1000000 : keys[g() % keys.size()]; };
  auto put = [&](uint64_t k, uint64_t val, const char* op) {
    if (!holds(k)) return;
    if (!checked && ref.count(k)) return;  // the unchecked put is for keys that are not stored
    const bool fresh = ref.emplace(k, val).second;
    if (idx->put(k, val) != fresh) fail(what, op, k);
    probes.push_back(k);
  };
  // put: neighbours of stored keys, runs behind a stored key, keys anywhere in the range, keys below and above everything,
  // the special keys, duplicates
  {
    std::vector<std::pair<uint64_t, uint64_t>> add;
    const int scale = keys.size() < 1000 ? 300 : 12000;
    for (int i = 0; i < scale; ++i) { const uint64_t k = pick(); add.push_back({k + 1, payload(k + 1)}); if (i % 3 == 0) add.push_back({k + 2, payload(k + 2)}); if (i % 7 == 0) add.push_back({k - 1, payload(k - 1)}); }
    for (int r = 0; r < 4; ++r) { const uint64_t k = pick(); for (uint64_t j = 1; j <= uint64_t(scale / 7); ++j) add.push_back({k + j * (r + 1), payload(k + j)}); }
    for (int i = 0; i < scale; ++i) { const uint64_t span = hi - lo, k = lo + (span == M ? g() : g() % (span + 1)); add.push_back({k, payload(k)}); }
    for (int i = 0; i < scale / 7; ++i) { const uint64_t k = g(); add.push_back({k, payload(k)}); }
    for (uint64_t j = 0; j < 40; ++j) {
      add.push_back({hi + 1 + j, j}), add.push_back({hi + (uint64_t(1) << (j + 4)), j}), add.push_back({lo - 1 - j, j});
      add.push_back({lo >> (j % 20 + 1), j}), add.push_back({hi + (hi - lo) * (j + 1), j});
    }
    for (uint64_t k : {uint64_t(0), uint64_t(1), uint64_t(2), M, M - 1, H, H - 1, H + 1}) add.push_back({k, payload(k)});
    for (int i = 0; i < scale / 10; ++i) add.push_back({pick(), 99});  // there already
    std::shuffle(add.begin(), add.end(), g);
    {
      Phase phase(3);
      for (const auto& a : add) put(a.first, a.second, "put");
    }
    audit("put");
    verify(*idx, ref, probes, g, what + " (put)", slots);
  }
  // update: stored keys take the new payload, absent keys return false and stay absent
  {
    Phase phase(4);
    for (int i = 0; i < 12000; ++i) {
      uint64_t k = i % 4 ? pick() : probes[g() % probes.size()];
      if (i % 9 == 0) k += 1;
      if (i % 1000 == 0) k = i % 3000 == 0 ? 0 : i % 3000 == 1000 ? M : 1;
      const uint64_t v = g();
      const auto it = ref.find(k);
      if (idx->update(k, v) != (it != ref.end())) fail(what, "update", k);
      if (it != ref.end()) it->second = v;
      probes.push_back(k);
    }
  }
  audit("updated");
  // remove: not supported; false for stored and absent keys, and nothing changes
  {
    Phase phase(4);
    for (int i = 0; i < 2000; ++i) {
      const uint64_t k = i % 2 ? pick() : g();
      if (idx->remove(k)) fail(what, "remove returned true", k);
    }
    for (uint64_t k : {uint64_t(0), M}) if (idx->remove(k)) fail(what, "remove returned true", k);
  }
  audit("after remove");
  verify(*idx, ref, probes, g, what + " (updated)", slots);
  size_t splits = d->bucket_splits(), roots = d->new_roots();
  if (ex.grow) {
    // keys appended in ascending order above the largest key (the largest below a cap that leaves room for them: the special
    // keys next to 2^64 are stored by now): the last D-Bucket of the data splits again and again
    const size_t m1 = std::min<size_t>(size_t(kWork * 160000), std::max<size_t>(slots + slots / 4, 20000));
    {
      Phase phase(5);
      const uint64_t cap = std::min(safe, M - 1) - uint64_t(m1) * 977 - 2;
      const auto above = ref.upper_bound(cap);
      const uint64_t top = above == ref.begin() ? 7 : std::prev(above)->first;
      for (size_t i = 1; i <= m1; ++i) put(top + i * 977, i, "put (appended)");
    }
    // a dense run put into one gap, in random order
    const uint64_t base = ref.size() > 4 ? std::next(ref.begin(), long(ref.size() / 2))->first : 5;
    std::vector<uint64_t> run;
    for (size_t i = 1; i <= m1; ++i) run.push_back(base + i);
    std::shuffle(run.begin(), run.end(), g);
    {
      Phase phase(5);
      for (uint64_t k : run) put(k, payload(k) ^ 1, "put (dense run)");
    }
    audit("grown");
    verify(*idx, ref, probes, g, what + " (grown)", slots);
    if (checked && slots <= 4096 && d->bucket_splits() == splits) fail(what, "the insert runs split no bucket; puts:", 2 * m1);
  }
  splits = d->bucket_splits(), roots = d->new_roots();
  long long bytes1 = 0;
  try {
    bytes1 = idx->memory_consumption();  // counts the pairs and checks what it must; throws if the structure is wrong
  } catch (const std::exception& e) {
    std::printf("FAIL %s: memory_consumption threw %s\n", what.c_str(), e.what());
    ++g_bad;
  }
  if (size_t(bytes1) != d->authors_mem_size() + sizeof(I)) fail(what, "memory_consumption is not the authors' mem_size() plus the wrapper:", uint64_t(bytes1));
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%-16s %-12s n=%-8zu loaded: %2llu levels %6zu buckets %6zu segments (%zu at risk) %6.1f B/key; after writes: n=%-8zu %2llu levels, %zu %s, %zu new roots, %zu risky puts, "
              "%6.1f B/key  %s  %.1fs (load %.1f verify %.1f audit %.1f put %.1f update %.1f grow %.1f)\n",
              name, label.c_str(), keys.size(), levels0, a0.buckets, a0.segments, a0.risk_segments, keys.empty() ? 0.0 : double(bytes) / double(keys.size()), ref.size(),
              (unsigned long long)d->levels(), splits, checked ? "bucket splits" : "authors' inserts", roots, d->risk_puts(), ref.empty() ? 0.0 : double(bytes1) / double(ref.size()),
              g_bad == bad0 ? "ok" : "FAILED", s, g_phase[0], g_phase[1], g_phase[2], g_phase[3], g_phase[4], g_phase[5]);
  std::fflush(stdout);
}

// The guard against the authors' 32-bit truncation (segment.h:300; bli.h has the mechanism). On D-Buckets of 256 slots, keys
// at a regular stride with one hole in the middle, a few pivot distances wide: the Segment over the pivots then has an empty
// S-Bucket. The smallest key above the data whose walk truncates is found by bisection (route_state); from it on, the
// authors' search for the S-Bucket starts at S-Bucket 0 and stops at the empty one, in a wrong D-Bucket. Then: the checked put refuses that key with std::runtime_error
// and nothing changes; a key that truncates and is still routed right goes in; the unchecked put takes the key, the
// authors' insert misplaces it, and the audit and memory_consumption say so; a bulk load that contains the key keeps it
// where it belongs, where the authors' lookup does not find it and the wrapper's get, update, put and scan do.
static void check_truncation() {
  using I = gret::BliIndex<uint64_t, uint64_t, 4, 256>;
  const size_t bad0 = g_bad;
  const std::string what = "bli8 truncation";
  bool found = false;
  for (double hole : {3.5, 3.0, 2.5, 4.0, 2.0, 5.0}) {  // in pivot distances; the first that leaves the pivots in one Segment with an empty S-Bucket
    if (found) break;
    std::mt19937_64 g(uint64_t(hole * 16));
    std::vector<std::pair<uint64_t, uint64_t>> kv;
    const int n = 153 * 64;  // 153 keys a D-Bucket at 60%
    for (int i = 0; i < n; ++i) {
      const uint64_t k = 1000 + 64 * uint64_t(i) + (i >= n / 2 ? uint64_t(hole * 153 * 64) : 0);
      kv.push_back({k, payload(k)});
    }
    auto make = [&](const char* name) {
      std::unique_ptr<I> d(dynamic_cast<I*>(gret::make_bli<uint64_t, uint64_t>(name)));
      d->bulk_load(kv.data(), kv.size());
      return d;
    };
    std::unique_ptr<I> d = make("bli8");
    const uint64_t safe = d->safe_key();
    uint64_t lo = kv.back().first + 1, hi = safe;
    if (d->route_state(lo) != 0 || d->route_state(hi) == 0) continue;  // no key above the data truncates, or all do
    while (lo + 1 < hi) {  // the walks of the keys above the data pass the same Segments: truncation is monotone in the key
      const uint64_t mid = lo + (hi - lo) / 2;
      (d->route_state(mid) ? hi : lo) = mid;
    }
    uint64_t right = 0, wrong = 0;  // truncating keys: one still routed right, one misrouted
    for (uint64_t j = 0; j < 400000 && !wrong && hi + j <= safe; ++j)
      if (d->route_state(hi + j) == 2) wrong = hi + j;
    for (uint64_t j = 1; j <= 4096 && !right; ++j) {
      const uint64_t c = hi + (safe - hi) / 4096 * j;
      if (d->route_state(c) == 1) right = c;
    }
    if (!wrong || !right) continue;
    found = true;
    Index* idx = d.get();
    Map ref(kv.begin(), kv.end());
    uint64_t v = 0;
    // checked: the misrouted key is refused, loudly, twice, and nothing changes
    for (int twice = 0; twice < 2; ++twice) {
      bool threw = false;
      try {
        idx->put(wrong, 7);
      } catch (const std::runtime_error&) {
        threw = true;
      }
      if (!threw) fail(what, "the checked put of a misrouted key did not throw std::runtime_error:", wrong);
    }
    if (d->refused_puts() != 2 || d->risk_puts() != 2 || d->size() != ref.size() || idx->get(wrong, v) || idx->update(wrong, 1)) fail(what, "a refused put left traces; key", wrong);
    if (!idx->put(right, 9) || idx->put(right, 10) || !idx->get(right, v) || v != 9 || d->risk_puts() != 4 || d->refused_puts() != 2) fail(what, "put of a truncating key that is routed right:", right);
    ref[right] = 9;
    {
      const auto a = d->audit(true);
      if (a.wrong || a.misplaced || a.keys != d->pairs() || a.risk_segments == 0) fail(what, "audit after the refused put: wrong", a.wrong);
      const auto b = d->audit(false);  // the keys at risk only: the one just put is among them
      if (b.wrong || b.looked_up == 0 || b.looked_up > a.looked_up) fail(what, "the audit of the keys at risk looked up", b.looked_up);
      std::vector<uint64_t> probes = {wrong, right, hi, hi - 1, safe};
      verify(*idx, ref, probes, g, what + " (checked)", 256);
      try {
        idx->memory_consumption();
      } catch (const std::exception& e) {
        std::printf("FAIL %s: memory_consumption threw %s\n", what.c_str(), e.what());
        ++g_bad;
      }
    }
    // unchecked: the authors' insert takes the key and misplaces it; the audit finds it and memory_consumption throws
    {
      std::unique_ptr<I> u = make("bli8u");
      if (u->route_state(wrong) != 2) fail(what, "the unchecked index routes the key differently:", wrong);
      if (!u->put(wrong, 7)) fail(what, "the unchecked put returned false for", wrong);
      const auto a = u->audit(false);
      if (a.misplaced != 1 || a.wrong != 1 || a.keys != u->pairs()) fail(what, "the audit did not find the key the authors' insert misplaced; misplaced:", a.misplaced);
      bool threw = false;
      try {
        u->memory_consumption();
      } catch (const std::runtime_error&) {
        threw = true;
      }
      if (!threw) fail(what, "memory_consumption did not throw with a misplaced key:", wrong);
    }
    // loaded with the key: it is in its D-Bucket, the authors' lookup misses it, the wrapper serves it
    {
      kv.push_back({wrong, payload(wrong)});
      ref.clear();
      ref.insert(kv.begin(), kv.end());
      std::unique_ptr<I> l(dynamic_cast<I*>(gret::make_bli<uint64_t, uint64_t>("bli8")));
      size_t missed = 0;
      try {
        l->bulk_load(kv.data(), kv.size());
        const auto a = l->audit(true);
        missed = a.rerouted;
        if (a.wrong || a.misplaced || a.keys != l->pairs()) fail(what, "audit of the index loaded with the misrouted key: wrong", a.wrong);
        if (!l->get(wrong, v) || v != payload(wrong) || l->rescued_gets() != 1) fail(what, "get of the misrouted key that was loaded:", wrong);
        if (l->put(wrong, 5) || !l->update(wrong, 6) || !l->get(wrong, v) || v != 6 || l->refused_puts()) fail(what, "put and update of the misrouted key that was loaded:", wrong);
        ref[wrong] = 6;
        std::vector<uint64_t> probes = {wrong, wrong + 1, wrong - 1, right, hi};
        verify(*l, ref, probes, g, what + " (loaded with the key)", 256);
        l->memory_consumption();
      } catch (const std::exception& e) {
        std::printf("FAIL %s: with the misrouted key loaded: %s\n", what.c_str(), e.what());
        ++g_bad;
      }
      std::printf("%-16s %-12s a hole of %.1f pivot distances: keys from %llu truncate; %llu is misrouted: the checked put throws, the unchecked put misplaces it and the audit finds it; "
                  "loaded with the data, the authors' lookup misses %zu key and the wrapper serves it; %llu truncates and is routed right: stored  %s\n",
                  "bli8, bli8u", "truncation", hole, (unsigned long long)hi, (unsigned long long)wrong, missed, (unsigned long long)right, g_bad == bad0 ? "ok" : "FAILED");
    }
  }
  if (!found) {
    std::printf("FAIL %s: no key set gave a misrouted key: the guard was not exercised\n", what.c_str());
    ++g_bad;
  }
}

// The guard against a misplaced PIVOT (bli.h, defect 2, last paragraph). The key set of check_truncation without its last
// 129 keys, so that with the pair of key 0 and 128 more keys the bulk load makes 64 full cuts of 153 keys; the 128 keys are
// keys above the data that the authors' walk misroutes (route_state 2, found on an index without them). They are loaded
// into the last D-Bucket with the last 25 keys of the data. Then keys are put between those 25 until the bucket splits: its
// median is the largest of the small keys, the pivot of the upper half is a misrouted key, and the authors' batch_update
// puts it into the S-Bucket its truncated search finds (segment.h:255), before the pivots of smaller keys. Every get still
// answers; the authors' scan returns the 128 large keys in the middle of the others. The checked put must throw
// std::runtime_error at that split (the wrapper checks every Segment for key order after a split below a Segment at risk);
// with the unchecked put nothing watches the split, and the audit and memory_consumption must report the Segment.
static void check_pivot_disorder() {
  using I = gret::BliIndex<uint64_t, uint64_t, 4, 256>;
  const size_t bad0 = g_bad;
  const std::string what = "bli8 disorder";
  bool found = false;
  for (double hole : {3.5, 3.0, 2.5, 4.0, 2.0, 5.0}) {
    if (found) break;
    std::vector<std::pair<uint64_t, uint64_t>> kv;
    const int n = 153 * 64 - 129;
    for (int i = 0; i < n; ++i) {
      const uint64_t k = 1000 + 64 * uint64_t(i) + (i >= n / 2 ? uint64_t(hole * 153 * 64) : 0);
      kv.push_back({k, payload(k)});
    }
    std::vector<uint64_t> far;  // 128 keys above the data that the authors' walk misroutes
    {
      std::unique_ptr<I> a(dynamic_cast<I*>(gret::make_bli<uint64_t, uint64_t>("bli8")));
      a->bulk_load(kv.data(), kv.size());
      uint64_t lo = kv.back().first + 1, hi = a->safe_key();
      if (a->route_state(lo) != 0 || a->route_state(hi) == 0) continue;
      while (lo + 1 < hi) {
        const uint64_t mid = lo + (hi - lo) / 2;
        (a->route_state(mid) ? hi : lo) = mid;
      }
      for (uint64_t j = 0; far.size() < 128 && j < 100000; ++j)
        if (a->route_state(hi + 1 + j * 7) == 2) far.push_back(hi + 1 + j * 7);
    }
    if (far.size() < 128) continue;
    found = true;
    const uint64_t last_small = kv.back().first;
    for (uint64_t k : far) kv.push_back({k, payload(k)});
    // the puts that fill the last D-Bucket: between its 25 small keys
    std::vector<uint64_t> fill;
    for (uint64_t j = 1; j < 63; ++j)
      for (uint64_t i = 0; i < 24; ++i) fill.push_back(last_small - 64 * i - j);
    for (const char* name : {"bli8", "bli8u"}) {
      const bool checked = name[4] == 0;
      std::unique_ptr<I> d(dynamic_cast<I*>(gret::make_bli<uint64_t, uint64_t>(name)));
      Index* idx = d.get();
      Map ref(kv.begin(), kv.end());
      std::mt19937_64 g(11);
      size_t puts = 0;
      bool threw = false;
      std::string said;
      try {
        d->bulk_load(kv.data(), kv.size());
        std::vector<uint64_t> probes(far.begin(), far.begin() + 8);
        verify(*idx, ref, probes, g, what + " (loaded)", 256);  // the wrapper serves the 128 keys the authors' lookup misses, and scans are in order
        const auto a0 = d->audit(true);
        if (a0.wrong || a0.disordered || a0.rerouted != far.size()) fail(what, "the audit of the loaded index: keys served past the authors' lookup:", a0.rerouted);
        for (uint64_t k : fill) {
          if (puts == 104) break;  // 25 + 103 small keys and 128 large ones fill the 256 slots; the 104th put splits
          if (!idx->put(k, payload(k))) fail(what, "put", k);
          ref[k] = payload(k), ++puts;
        }
      } catch (const std::runtime_error& e) {
        threw = true, said = e.what();
      }
      if (checked) {
        if (!threw || puts != 103 || d->risk_splits() != 1 || said.find("out of key order") == std::string::npos) fail(what, "the checked put that splits the D-Bucket did not throw as it must; puts:", puts);
      } else {
        if (threw || puts != 104) fail(what, "the unchecked puts; done:", puts);
        const auto a = d->audit(false);
        if (a.disordered != 1 || a.wrong == 0) fail(what, "the audit did not find the Segment that is out of key order; found:", a.disordered);
        // what the guard is for: the authors' scan is out of order now
        std::vector<std::pair<uint64_t, uint64_t>> out(ref.size() + 1);
        const size_t got = idx->scan(1, ref.size() + 1, out.data());
        bool sorted = got == ref.size();
        for (size_t i = 1; sorted && i < got; ++i) sorted = out[i - 1].first < out[i].first;
        if (sorted) fail(what, "the authors' scan is in order after the misplaced pivot: the set does not provoke the defect any more; keys:", got);
        bool mthrew = false;
        try {
          idx->memory_consumption();
        } catch (const std::runtime_error&) {
          mthrew = true;
        }
        if (!mthrew) fail(what, "memory_consumption did not throw with a Segment out of key order:", a.disordered);
      }
    }
    std::printf("%-16s %-12s a hole of %.1f pivot distances; a D-Bucket whose upper half are 128 keys the authors' walk misroutes is split: the authors' batch_update misplaces its pivot. "
                "The checked put throws at that split; with the unchecked put the scan is out of order and the audit and memory_consumption say so  %s\n",
                "bli8, bli8u", "disorder", hole, g_bad == bad0 ? "ok" : "FAILED");
  }
  if (!found) {
    std::printf("FAIL %s: no key set gave 128 misrouted keys: the guard was not exercised\n", what.c_str());
    ++g_bad;
  }
}

// The unchecked put given a stored key: the authors' insert adds a second pair (documented; names ending in u are for
// callers that never do this). Lookups still answer, the pair count says what happened, nothing throws.
static void check_unchecked_duplicate() {
  using I = gret::BliIndex<uint64_t, uint64_t, 4, 256>;
  const size_t bad0 = g_bad;
  std::vector<std::pair<uint64_t, uint64_t>> kv;
  for (uint64_t i = 1; i <= 5000; ++i) kv.push_back({i * 1000, i});
  std::unique_ptr<I> u(dynamic_cast<I*>(gret::make_bli<uint64_t, uint64_t>("bli8u")));
  u->bulk_load(kv.data(), kv.size());
  uint64_t v = 0;
  if (!u->put(3000, 77) || !u->get(3000, v) || (v != 3 && v != 77)) fail("bli8u duplicate", "put of a stored key, then get:", 3000);
  const auto a = u->audit(true);
  if (a.keys != kv.size() + 2 || u->pairs() != a.keys || u->size() != kv.size() + 1) fail("bli8u duplicate", "the pair count after a second put of a stored key:", a.keys);
  bool threw = false;
  try {
    u->memory_consumption();
  } catch (const std::exception&) {
    threw = true;
  }
  if (threw) fail("bli8u duplicate", "memory_consumption threw after a duplicate put:", 3000);
  std::printf("%-16s %-12s a stored key put again without the check: a second pair, as documented; counted, lookups answer  %s\n", "bli8u", "duplicate", g_bad == bad0 ? "ok" : "FAILED");
}

// NOT in the default run (ask for it by name: t rootoverflow; an -O3 build only). It makes the authors' code write behind
// an array: keys with exponentially growing gaps, loaded with error bound 1024 so that the root is one Segment over all
// pivots, then puts until the root is cut again (error bound S / 2) and a new root is built over the many pieces
// (buck_index.h:446-459). Its regression line passes the last S-Bucket and the Segment constructor follows it
// (segment.h:91-109; AddressSanitizer: heap-buffer-overflow at segment.h:98). The wrapper must notice (it counts the pairs
// that can still be reached after every new root) and throw std::runtime_error. Memory is corrupted by then, so this run
// may also crash.
static void check_root_overflow() {
  std::vector<std::pair<uint64_t, uint64_t>> kv;
  double x = 1000;
  uint64_t last = 0;
  for (int i = 0; i < 60000; ++i) {
    uint64_t k = uint64_t(x);
    if (k <= last) k = last + 1;
    kv.push_back({k, k});
    last = k;
    x = x * 1.0004 + 1;
  }
  Index* idx = gret::make_bli<uint64_t, uint64_t>("bli8e1024");  // never deleted: its memory is not to be trusted afterwards
  idx->bulk_load(kv.data(), kv.size());
  std::mt19937_64 g(1);
  size_t puts = 0;
  bool threw = false;
  try {
    for (int round = 0; round < 3 && !threw; ++round)
      for (size_t i = 0; i + 1 < kv.size(); ++i) {
        const uint64_t gap = kv[i + 1].first - kv[i].first;
        if (gap >= 2) puts += idx->put(kv[i].first + 1 + g() % (gap - 1), 1);
      }
  } catch (const std::runtime_error& e) {
    threw = true;
    std::printf("%-16s %-12s after %zu puts the wrapper threw: %s  ok\n", "bli8e1024", "rootoverflow", puts, e.what());
  }
  if (!threw) {
    std::printf("FAIL bli8e1024 rootoverflow: %zu puts and the authors' new root was never built past its array, or the wrapper did not notice\n", puts);
    ++g_bad;
  }
  std::fflush(stdout);
}

// the thinnest GRE index over the authors' code: what the wrapper's get and put are measured against in --time
template <size_t S, size_t D>
struct Thin {
  gret::bli_authors::buckindex::BuckIndex<uint64_t, uint64_t, S, D> idx{0.6, 4};
  void bulk_load(std::pair<uint64_t, uint64_t>* kv, size_t n) {
    std::vector<gret::bli_authors::buckindex::KeyValue<uint64_t, uint64_t>> kvs;
    if (kv[0].first != 0) kvs.emplace_back(0, 0);
    for (size_t i = 0; i < n; ++i) kvs.emplace_back(kv[i].first, kv[i].second);
    idx.bulk_load(kvs);
  }
  bool get(uint64_t k, uint64_t& v) { return idx.lookup(k, v); }
  bool put(uint64_t k, uint64_t v) { gret::bli_authors::buckindex::KeyValue<uint64_t, uint64_t> e(k, v); return idx.insert(e); }
  long long memory_consumption() { return 0; }
};

template <class I>
static void time_one(const char* name, I* idx, const std::vector<std::pair<uint64_t, uint64_t>>& kv, const std::vector<std::pair<uint64_t, uint64_t>>& extra) {
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
    for (uint64_t x : q) sum += idx->get(x, v) ? v : 1;
    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(q.size()));
  }
  t0 = std::chrono::steady_clock::now();
  size_t in = 0;
  for (const auto& e : extra) in += idx->put(e.first, e.second);
  const double put = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(extra.size());
  double after = 1e30;
  for (int rep = 0; rep < 2; ++rep) {
    t0 = std::chrono::steady_clock::now();
    for (uint64_t x : q) sum += idx->get(x, v) ? v : 1;
    after = std::min(after, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(q.size()));
  }
  std::printf("time %-14s n=%zu build %.2fs  get %6.1f ns  put %7.1f ns (%zu new)  get after the puts %6.1f ns  %.1f B/key  [checksum %llu]\n", name, kv.size(), build, best, put, in,
              after, double(idx->memory_consumption()) / double(kv.size() + in), (unsigned long long)sum);
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  std::vector<std::string> only;
  bool timing = false;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--time")) timing = true;
    else only.push_back(argv[i]);
  auto want = [&](const std::string& s) { return only.empty() ? !timing : std::find(only.begin(), only.end(), s) != only.end(); };
  if (names_are_free<uint64_t, uint64_t>() != 16 || sizeof(vector) + sizeof(Bucket) + sizeof(Segment) != 3) return 2;

  if (want("names")) {
    for (const char* bad : {"bl", "blix", "bli0", "bli017", "bli9", "bli25", "bli17s", "bli17s5", "bli17s04", "bli17f29", "bli17f91", "bli17f060", "bli17e0", "bli17e1025", "bli17f60s4",
                            "bli17e4f60", "bli17uu", "bli17u4", "bli 17", "bli17 ", "bli-17", "blis4s8", "bli17s4f60e4x", "bli170"})
      if (gret::make_bli<uint64_t, uint64_t>(bad)) {
        std::printf("FAIL make_bli accepts the malformed name '%s'\n", bad);
        ++g_bad;
      }
    struct Good { const char* name; int l, s, f, e; bool checked; };
    for (const Good& w : {Good{"bli", 17, 4, 60, 4, true}, Good{"bliu", 17, 4, 60, 4, false}, Good{"bli8", 8, 4, 60, 4, true}, Good{"bli24s16f90e1024u", 24, 16, 90, 1024, false},
                          Good{"blis8", 17, 8, 60, 4, true}, Good{"blif30", 17, 4, 30, 4, true}, Good{"blie8", 17, 4, 60, 8, true}, Good{"bli20s4f60e4", 20, 4, 60, 4, true},
                          Good{"bli15f50u", 15, 4, 50, 4, false}}) {
      gret::BliName n;
      if (!gret::parse_bli_name(w.name, n) || n.log2_d != w.l || n.s != w.s || n.fill_pct != w.f || n.error_bound != w.e || n.checked != w.checked) {
        std::printf("FAIL parse_bli_name on '%s'\n", w.name);
        ++g_bad;
      }
    }
    if (gret::make_bli<uint64_t, uint32_t>("bli") || gret::make_bli<uint32_t, uint64_t>("bli")) {
      std::printf("FAIL make_bli gives an index for key or payload types BLI is not built for here\n");
      ++g_bad;
    }
    std::printf("names            the parser takes what it should and nothing else  %s\n", g_bad ? "FAILED" : "ok");
  }

  const size_t N = 120000;
  const char* seed = std::getenv("TEST_BLI_SEED");  // another draw of every random set
  std::mt19937_64 g(seed && *seed ? std::strtoull(seed, nullptr, 10) : 7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; v.reserve(m); std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  Expect refuses; refuses.refuses = true;
  Expect still; still.grow = false;

  if (want("tiny")) {  // an index never loaded, an empty bulk load, and key sets of a few keys
    check<4, 8>("n0", {}, "bli8");
    check<4, 17>("n0", {}, "bli", still);
    check<4, 8>("n1", {5}, "bli8");
    check<4, 8>("n1max", {M}, "bli8");
    check<4, 8>("n1zero", {0}, "bli8");
    check<4, 8>("n2", {0, M}, "bli8");
    check<4, 17>("n2", {0, M}, "bli");
    check<4, 8>("n3", {1, 2, M}, "bli8");
    check<4, 8>("n5", {0, 1, 2, 3, M}, "bli8");
    check<4, 8>("n17", rnd(0, uint64_t(1) << 40, 17), "bli8");
    check<4, 17>("n17", rnd(0, M, 17), "bli", still);
    check<4, 8>("n17", rnd(0, uint64_t(1) << 40, 17), "bli8u");
    check<8, 12>("n300", rnd(0, uint64_t(1) << 40, 300), "bli12s8");
    {  // never loaded at all: puts into an empty index, key 0 first
      std::unique_ptr<gret::BliIndex<uint64_t, uint64_t, 4, 256>> d(dynamic_cast<gret::BliIndex<uint64_t, uint64_t, 4, 256>*>(gret::make_bli<uint64_t, uint64_t>("bli8")));
      Index* idx = d.get();
      uint64_t v = 7;
      std::pair<uint64_t, uint64_t> out[4];
      const size_t bad0 = g_bad;
      if (idx->get(0, v) || idx->get(5, v) || idx->get(M, v) || idx->update(5, 1) || idx->remove(5) || idx->scan(0, 4, out) != 0 || idx->memory_consumption() <= 0) fail("bli8 never loaded", "an empty index answers", 0);
      if (!idx->put(0, 11) || idx->put(0, 12) || !idx->get(0, v) || v != 11 || !idx->put(M, 13) || !idx->put(9, 14) || idx->scan(0, 4, out) != 3 || out[0] != std::make_pair(uint64_t(0), uint64_t(11)) ||
          out[1] != std::make_pair(uint64_t(9), uint64_t(14)) || out[2] != std::make_pair(M, uint64_t(13)) || d->size() != 3)
        fail("bli8 never loaded", "puts into an empty index", 0);
      std::printf("%-16s %-12s puts into an index never loaded  %s\n", "bli8", "unloaded", g_bad == bad0 ? "ok" : "FAILED");
    }
  }
  // D-Buckets of 256 slots (many buckets, many Segments, many splits): the keys must stay below 2^56
  if (want("uniform")) check<4, 8>("uniform", rnd(1, uint64_t(1) << 50, N), "bli8");
  if (want("uniform")) check<8, 8>("uniform", rnd(1, uint64_t(1) << 50, N), "bli8s8f50");
  if (want("uniform")) check<16, 10>("uniform", rnd(1, uint64_t(1) << 50, N), "bli10s16f90e16");
  if (want("uniform")) check<4, 12>("uniform", rnd(1, uint64_t(1) << 50, N), "bli12f30e1");
  if (want("run")) { std::vector<uint64_t> v; for (uint64_t i = 0; i < N; ++i) v.push_back(1000 + i); check<4, 8>("run", v, "bli8u"); check<8, 14>("run", v, "bli14s8"); }
  if (want("run0")) { std::vector<uint64_t> v; for (uint64_t i = 0; i < N; ++i) v.push_back(i); check<4, 8>("run0", v, "bli8"); }
  if (want("clusters")) {
    std::vector<uint64_t> v;
    for (int c = 0; c < 1200; ++c) { const uint64_t b = g() >> 9; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); }
    check<4, 8>("clusters", v, "bli8");
    check<8, 10>("clusters", v, "bli10s8u");
  }
  if (want("stride")) { std::vector<uint64_t> v; for (uint64_t i = 0; i < N; ++i) v.push_back(i * 65536 + 12345); check<4, 8>("stride", v, "bli8"); check<16, 8>("stride", v, "bli8s16"); }
  if (want("skew")) {  // lognormal gaps: dense and sparse regions next to each other
    std::vector<uint64_t> v;
    std::lognormal_distribution<double> ln(0.0, 4.0);
    uint64_t k = 1;
    for (size_t i = 0; i < N; ++i) { k += 1 + uint64_t(std::min(ln(g), 1e9)); v.push_back(k); }
    check<4, 8>("skew", v, "bli8");
    check<4, 8>("skew", v, "bli8f90");
  }
  // keys over the whole 64-bit range: only D-Buckets of 2^16 slots and more hold them; smaller ones must refuse
  std::vector<uint64_t> wide, outliers, hugegap, fblike;
  if (want("wide") || want("refuse")) {
    wide = rnd(0, M, N);
    for (uint64_t k : {uint64_t(0), M, H, H - 1}) wide.push_back(k);
  }
  if (want("outliers") || want("refuse")) {  // a narrow bulk and keys far from it
    outliers = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), N);
    for (uint64_t j = 0; j < 5; ++j) outliers.push_back((uint64_t(1) << 62) + j * 1000003);
    outliers.push_back(3), outliers.push_back(M), outliers.push_back(M - 1), outliers.push_back(M - 3);
  }
  if (want("hugegap")) for (uint64_t i = 0; i < N; ++i) hugegap.push_back(i), hugegap.push_back(M - i);  // a run from 0 and a run down from 2^64 - 1
  if (want("fblike")) {  // small keys, a few very large ones
    fblike = rnd(0, uint64_t(1) << 36, 2 * N);
    for (int j = 0; j < 21; ++j) fblike.push_back((uint64_t(1) << 59) + (g() >> 2));
    for (int j = 0; j < 10; ++j) fblike.push_back(H + (g() >> 1));
  }
  if (want("wide")) check<4, 16>("wide", wide, "bli16"), check<4, 17>("wide", wide, "bli"), check<8, 16>("wide", wide, "bli16s8u"), check<4, 20>("wide", wide, "bli20", still);
  if (want("outliers")) check<4, 16>("outliers", outliers, "bli16"), check<4, 17>("outliers", outliers, "bliu");
  if (want("hugegap")) check<4, 16>("hugegap", hugegap, "bli16f80"), check<16, 18>("hugegap", hugegap, "bli18s16");
  if (want("fblike")) check<4, 16>("fblike", fblike, "bli16"), check<4, 17>("fblike", fblike, "bli");
  if (want("refuse")) {
    check<4, 8>("wide", wide, "bli8", refuses);
    check<4, 14>("outliers", outliers, "bli14", refuses);
    check<4, 15>("wide", wide, "bli15u", refuses);
  }
  // the larger sets: 2M random keys on the authors' benchmark setting (131072 slots, S-Buckets of 4, 60% full) and around it
  if (want("random2m")) check<4, 17>("random2m", rnd(0, M, 2000000), "bli");
  if (want("random2m")) check<4, 12>("random2m", rnd(1, uint64_t(1) << 56, 2000000), "bli12u", still);

  if (want("truncation")) check_truncation();
  if (want("disorder")) check_pivot_disorder();
  if (want("duplicate")) check_unchecked_duplicate();
  if (std::find(only.begin(), only.end(), "rootoverflow") != only.end()) check_root_overflow();  // only when asked for: see there

  if (timing) {  // not a benchmark: one small set, to see what the wrapper costs over the authors' calls
    std::vector<uint64_t> keys = rnd(1, uint64_t(1) << 50, 3000000);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<std::pair<uint64_t, uint64_t>> kv, extra;
    for (uint64_t k : keys) kv.push_back({k, 123456789});
    for (uint64_t k : rnd(1, uint64_t(1) << 50, 1500000)) if (!std::binary_search(keys.begin(), keys.end(), k)) extra.push_back({k, 123456789});
    std::sort(extra.begin(), extra.end());
    extra.erase(std::unique(extra.begin(), extra.end()), extra.end());
    std::shuffle(extra.begin(), extra.end(), g);
    for (const char* name : {"bli", "bliu", "bli20", "bli20u", "bli14", "bli14u", "bli10u"}) {
      std::unique_ptr<Index> idx(gret::make_bli<uint64_t, uint64_t>(name));
      time_one(name, idx.get(), kv, extra);
      idx.release();  // GRE deletes through indexInterface, which has no virtual destructor; here it is left to the exit
    }
    { std::cout.setstate(std::ios_base::failbit); time_one("authors 2^17", new Thin<4, 131072>, kv, extra); time_one("authors 2^14", new Thin<4, 16384>, kv, extra); std::cout.clear(); }
#ifndef TEST_BLI_ALONE
    time_one("alex", new alexInterface<uint64_t, uint64_t>, kv, extra);
    time_one("lipp", new LIPPInterface<uint64_t, uint64_t>, kv, extra);
    time_one("btree", new BTreeInterface<uint64_t, uint64_t>, kv, extra);
#endif
  }
  std::printf(g_bad ? "test_bli: %zu FAILURES\n" : "test_bli: all passed\n", g_bad);
  return g_bad ? 1 : 0;
}
#endif
