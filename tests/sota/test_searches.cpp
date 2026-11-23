// SPDX-License-Identifier: MIT
// The plain searches of gret/sota/searches.h (bsearch, bsearchstd, rbs<bits>, rbsstd<bits>, rbst<bits>, interp<k>[e], tip<g>[a]) against a sorted array.
//   models:   rbs (SOSD's class): every stored key's position inside its bound, the bound inside the array, the bytes SOSD counts;
//             rbst (ours): the bucket holds exactly the keys of its prefix, the lower bound of any key is in [lo, hi], and SOSD's bound
//             is this one widened by one position on each side, for every probe: the two tables are the same table;
//             rbsstd (SOSD's class, std::lower_bound inside its bound): the lower bound of every probe in [min, max], the same bytes;
//             bsearch: {0, n}, which is what SOSD's BinarySearch returns; what cannot be built throws.
//   searches: interp and tip return the lower bound of every stored key, of both neighbours of every key, and of random and in-gap
//             keys, at several step bounds and guards, on arrays of 3 keys and up, within their bounds on steps and records scanned
//             (in the 2M sets: every stored key, the neighbours of every fourth);
//             the first estimate of tip passes through the first, middle and last key (Equation 8), and the 'a' variant's first
//             estimate is the one SOSD's rm_tip.h computes, which does not.
//   indexes:  for every name: get on every stored key, on its neighbours, on 0, 1, 2^64 - 1, below and above the range and on random
//             keys; scan from stored, neighbouring, in-gap and extreme starts at several lengths; put, update and remove return false
//             and change nothing; memory_consumption is the pairs plus the model.
//   names:    accepted and rejected.
// Key sets as in tests/test_scan.cpp (tiny, uniform, wide, dense run, outliers, clusters, stride, huge gap, bit field, keys 0 and
// 2^64 - 1), sizes either side of the 64-key threshold, and three sets of 2M keys. Ends with a small timing, for sanity only.
//   build:  cxx -I <SOSD> tests/sota/test_searches.cpp          (or -I <the directory that holds SOSD>; -isystem on the cluster)
//   run:    t [--small | --big] [KEYS...]
//           --small: the names, the throws, the first estimate of tip and the small key sets; --big: the three 2M sets and the timing;
//           neither: everything. The two halves are for a machine where one run must stay short (a shared login node): each is about
//           half the CPU time, and together they are the whole test.
//           KEYS: GRE key files (a uint64 count, then the keys), optional: the model and index checks on each for a few names.
#include "indexInterface.h"
#include "pgm/pgm.h"  // readonly.h's PgmBound names pgm::PGMIndex
#include "gret/sota/searches.h"

namespace gret {
namespace sota {
namespace sosd {
// the test alone also reads SOSD's BinarySearch and SOSD's third-party TIP, to pin what they compute; neither answers a lookup here
#if defined(__has_include) && __has_include("SOSD/competitors/rm_tip.h")
#include "SOSD/competitors/binary_search.h"
#include "SOSD/competitors/rm_tip.h"
#else
#include "competitors/binary_search.h"
#include "competitors/rm_tip.h"
#endif
}  // namespace sosd
}  // namespace sota
}  // namespace gret

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
using gret::KeyBound;
using gret::sota::SearchName;
namespace sd = gret::sota::search_detail;

struct DeleteSearch {
  std::string name;
  void operator()(Index* p) const { gret::sota::destroy_searches<K, K>(p, name); }
};
using IndexPtr = std::unique_ptr<Index, DeleteSearch>;
static IndexPtr make(const std::string& name) { return IndexPtr(gret::sota::make_searches<K, K>(name), DeleteSearch{name}); }

static const K M = ~K(0);
static size_t g_fails = 0, g_checks = 0;
static sd::CountTrace g_interp, g_interp_e, g_tip;  // what the searches did over the whole test: every path must have been taken
static size_t g_interp_lookups = 0, g_interp_e_lookups = 0, g_tip_lookups = 0;

static K payload(K k) { return k * 2654435761u + 17; }

#define FAIL(...)                    \
  do {                               \
    if (g_fails++ < 40) {            \
      std::printf("FAIL ");          \
      std::printf(__VA_ARGS__);      \
      std::printf("\n");             \
    }                                \
  } while (0)

static std::vector<K> probes(const std::vector<K>& keys, std::mt19937_64& g, size_t random_n) {
  std::vector<K> p = {0, 1, 2, M, M - 1, K(1) << 63, (K(1) << 63) - 1, keys.front() - 1, keys.front() + 1, keys.back() + 1, keys.back() - 1};
  for (size_t i = 0; i < random_n; ++i) {
    p.push_back(g());
    const size_t r = g() % keys.size();
    if (r + 1 < keys.size() && keys[r + 1] - keys[r] > 2) p.push_back(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));  // inside a gap
  }
  return p;
}

static size_t lower_of(const std::vector<K>& keys, K x) { return size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin()); }

// ---- the radix tables: SOSD's class (rbs) and ours (rbst) on the same keys
template <uint32_t Bits>
static void check_rbs_models(const std::vector<K>& keys, const std::string& label) {
  const std::string what = "rbs" + std::to_string(Bits) + " and rbst" + std::to_string(Bits) + " models on " + label;
  const size_t n = keys.size(), entries = (size_t(1) << Bits) + 1;
  ++g_checks;
  gret::sota::RbsBound<K, Bits> s;
  gret::sota::RbstBound<K, Bits> t;
  // SOSD's class again, searched as SOSD searches it. Not at 28 bits: a third 1 GB table at once, and rbsstd28 is checked as an index
  const bool with_u = Bits < 28;
  gret::sota::RbsStdSearch<K, Bits> u;
  std::vector<Pair> pv(with_u ? n : 0);
  for (size_t i = 0; i < pv.size(); ++i) pv[i] = {keys[i], 0};
  try {
    std::vector<K> c1 = keys, c2 = keys;
    s.build(c1);
    t.build(c2);
    if (with_u) u.build(pv.data(), n);
    if (c2 != keys) FAIL("%s: our build changed the keys", what.c_str());
  } catch (const std::exception& e) {
    FAIL("%s: build threw %s", what.c_str(), e.what());
    return;
  }
  // the prefix, computed apart from both: the top Bits bits of max - min
  const K span = keys.back() - keys.front();
  unsigned len = 0;
  while (len < 64 && (span >> len) != 0) ++len;
  const unsigned shift = len < Bits ? 0 : len - Bits;
  auto prefix = [&](K k) { return size_t((k - keys.front()) >> shift); };
  if (prefix(keys.back()) >= entries - 1) FAIL("%s: the largest prefix %zu does not fit %zu buckets", what.c_str(), prefix(keys.back()), entries - 1);

  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    const KeyBound a = s.get(keys[i]), b = t.get(keys[i]);
    if (!(a.lo <= i && i < a.hi && a.hi <= n) && bad++ < 3) FAIL("%s: key %llu at %zu, SOSD's bound [%zu, %zu), n %zu", what.c_str(), (unsigned long long)keys[i], i, a.lo, a.hi, n);
    // our bucket is exactly the keys with this key's prefix
    const size_t p = prefix(keys[i]);
    const bool exact = b.lo <= i && i < b.hi && b.hi <= n && prefix(keys[b.lo]) == p && prefix(keys[b.hi - 1]) == p && (b.lo == 0 || prefix(keys[b.lo - 1]) < p) &&
                       (b.hi == n || prefix(keys[b.hi]) > p);
    if (!exact && bad++ < 3) FAIL("%s: key %llu at %zu, our bucket [%zu, %zu)", what.c_str(), (unsigned long long)keys[i], i, b.lo, b.hi);
  }
  auto in_range = [&](K x, size_t lb) {  // lb: the lower bound of x in keys
    const KeyBound a = s.get(x), b = t.get(x);
    if (!(b.lo <= lb && lb <= b.hi && b.hi <= n) && bad++ < 3) FAIL("%s: key %llu, lower bound %zu, our bucket [%zu, %zu)", what.c_str(), (unsigned long long)x, lb, b.lo, b.hi);
    // SOSD's bound is the bucket, one position wider on each side, cut at 0 and n
    if (!(a.lo == (b.lo ? b.lo - 1 : 0) && a.hi == (b.hi != n ? b.hi + 1 : n)) && bad++ < 3)
      FAIL("%s: key %llu, SOSD [%zu, %zu), our bucket [%zu, %zu)", what.c_str(), (unsigned long long)x, a.lo, a.hi, b.lo, b.hi);
    const size_t got = with_u ? u.lower(pv.data(), n, x) : lb;  // std::lower_bound inside SOSD's bound finds the lower bound itself, stored or not
    if (got != lb && bad++ < 3) FAIL("%s: key %llu, rbsstd lower %zu, want %zu", what.c_str(), (unsigned long long)x, got, lb);
  };
  std::mt19937_64 g(3);
  for (size_t i = 0; i < n; ++i) {  // the key, and both neighbours: keys[i] - 1 is stored only if it is keys[i - 1]; keys[i] + 1 is at i + 1 or absent
    in_range(keys[i], i);
    if (n > 300000 && (i & 3) != 0) continue;  // the 2M sets: every stored key, and the neighbours of every fourth
    if (i != 0) in_range(keys[i] - 1, keys[i - 1] == keys[i] - 1 ? i - 1 : i);
    if (i != n - 1) in_range(keys[i] + 1, i + 1);
  }
  for (const K x : probes(keys, g, std::min<size_t>(n, 20000))) {
    const KeyBound a = s.get(x), b = t.get(x);
    if (x >= keys.front() && x <= keys.back()) {
      in_range(x, lower_of(keys, x));
    } else if (x < keys.front()) {  // SOSD: [0, 1); ours: the lower bound itself
      if (!(a.lo == 0 && a.hi == 1 && b.lo == 0 && b.hi == 0) && bad++ < 3) FAIL("%s: below the range, SOSD [%zu, %zu), ours [%zu, %zu)", what.c_str(), a.lo, a.hi, b.lo, b.hi);
    } else {
      if (!(a.lo == n - 1 && a.hi == n && b.lo == n && b.hi == n) && bad++ < 3) FAIL("%s: above the range, SOSD [%zu, %zu), ours [%zu, %zu)", what.c_str(), a.lo, a.hi, b.lo, b.hi);
    }
  }
  // bytes: 4 per entry, 2^Bits + 1 entries, as SOSD's size() counts, plus the object
  if (s.m.size() != 4 * entries || s.bytes() != sizeof(s) + 4 * entries) FAIL("%s: SOSD size() %zu, bytes %zu, want %zu entries", what.c_str(), s.m.size(), s.bytes(), entries);
  if (with_u && (u.m.size() != 4 * entries || u.bytes() != sizeof(u) + 4 * entries)) FAIL("%s: rbsstd size() %zu, bytes %zu, want %zu entries", what.c_str(), u.m.size(), u.bytes(), entries);
  if (t.table_.size() != entries || t.bytes() != sizeof(t) + 4 * entries) FAIL("%s: our table %zu entries, bytes %zu, want %zu entries", what.c_str(), t.table_.size(), t.bytes(), entries);
  if (t.table_.front() != 0 || t.table_.back() != n || !std::is_sorted(t.table_.begin(), t.table_.end())) FAIL("%s: our table is not 0 ... n ascending", what.c_str());
}

static void check_rbs_models_at(unsigned bits, const std::vector<K>& keys, const std::string& label) {
  switch (bits) {
    case 8: return check_rbs_models<8>(keys, label);
    case 12: return check_rbs_models<12>(keys, label);
    case 16: return check_rbs_models<16>(keys, label);
    case 20: return check_rbs_models<20>(keys, label);
    case 24: return check_rbs_models<24>(keys, label);
    case 28: return check_rbs_models<28>(keys, label);
    default: FAIL("no model check at %u bits", bits);
  }
}

// ---- interp and tip, as searches over a pair array of 3 keys or more
template <class Search>
static void check_search(Search s, const std::vector<K>& keys, const std::string& what, sd::CountTrace& total, size_t& lookups, unsigned max_steps, size_t max_scan) {
  const size_t n = keys.size();
  ++g_checks;
  std::vector<Pair> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = {keys[i], payload(keys[i])};
  s.build(v.data(), n);
  size_t bad = 0;
  const bool sparse = n > 300000;  // the 2M sets: every stored key, and the neighbours of every fourth
  auto one = [&](K x, size_t want) {  // keys.front() <= x <= keys.back(); want: the lower bound of x in keys
    sd::CountTrace t;
    const size_t got = s.lower_t(v.data(), n, x, t);
    if (got != want && bad++ < 3) FAIL("%s: lower(%llu) = %zu, want %zu", what.c_str(), (unsigned long long)x, got, want);
    if ((!sparse || (want & 3) == 0) && s.lower(v.data(), n, x) != got && bad++ < 3) FAIL("%s: lower(%llu) differs with and without the trace", what.c_str(), (unsigned long long)x);
    if ((t.steps > max_steps || t.scanned > max_scan || t.fallbacks > 1 || t.finishes > 1) && bad++ < 3)
      FAIL("%s: lower(%llu) took %llu steps, scanned %llu records", what.c_str(), (unsigned long long)x, (unsigned long long)t.steps, (unsigned long long)t.scanned);
    total.steps += t.steps, total.scanned += t.scanned, total.fallbacks += t.fallbacks, total.finishes += t.finishes, total.finish_probes += t.finish_probes, total.finish_range += t.finish_range;
    ++lookups;
  };
  for (size_t i = 0; i < n; ++i) {  // as in check_rbs_models
    one(keys[i], i);
    if (sparse && (i & 3) != 0) continue;
    if (i != 0) one(keys[i] - 1, keys[i - 1] == keys[i] - 1 ? i - 1 : i);
    if (i != n - 1) one(keys[i] + 1, i + 1);
  }
  std::mt19937_64 g(11);
  for (const K x : probes(keys, g, std::min<size_t>(n, 20000)))
    if (x >= keys.front() && x <= keys.back()) one(x, lower_of(keys, x));
}

template <bool AuthorsFirst>
static void check_tip(unsigned guard, const std::vector<K>& keys, const std::string& label) {
  const std::string what = "tip" + std::to_string(guard) + (AuthorsFirst ? "a" : "") + " search on " + label;
  const size_t n = keys.size();
  gret::sota::TipSearch<K, AuthorsFirst> s(guard);
  check_search(s, keys, what, g_tip, g_tip_lookups, gret::sota::kTipMaxIterations, 8 * size_t(std::max(guard, 8u)));
  std::vector<Pair> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = {keys[i], 0};
  s.build(v.data(), n);
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    const int64_t e = s.first(keys[i]);
    if ((e < 0 || e >= int64_t(n)) && bad++ < 3) FAIL("%s: first estimate %lld for the key at %zu of %zu", what.c_str(), (long long)e, i, n);
  }
  if (!AuthorsFirst) {  // Equation 8 passes through the three keys it is built on (a position of rounding at the end)
    const int64_t e0 = s.first(keys.front()), ed = s.first(keys[n / 2]), el = s.first(keys.back());
    if (e0 > 1 || ed != int64_t(n / 2) || el < int64_t(n) - 3) FAIL("%s: first estimates %lld, %lld, %lld at the first, middle and last of %zu keys", what.c_str(), (long long)e0, (long long)ed, (long long)el, n);
  }
}

template <bool Gallop>
static void check_interp(unsigned steps, const std::vector<K>& keys, const std::string& label) {
  check_search(gret::sota::InterpSearch<K, Gallop>(steps), keys, "interp" + std::to_string(steps) + (Gallop ? "e" : "") + " search on " + label, Gallop ? g_interp_e : g_interp,
               Gallop ? g_interp_e_lookups : g_interp_lookups, steps, 0);
}

// What the borrowed TIP codes compute first, pinned on SOSD's copy (whose interpolate_r is public; the authors' tip.h has the same
// expression, private): not Equation 8. And our 'a' variant computes the same number.
static void check_tip_first_estimate() {
  ++g_checks;
  using SosdTip = gret::sota::sosd::RMThreePointInterpolationSearch<K>;
  using SosdKV = gret::sota::sosd::KeyValue<K>;
  std::mt19937_64 g(21);
  for (int mode = 0; mode < 5; ++mode) {
    std::vector<K> keys;
    for (K i = 0; i < 2001; ++i) keys.push_back(mode == 0 ? i * 10 : mode == 1 ? 1000000 + i * 10 : mode == 2 ? i * i : mode == 3 ? 1 + (g() >> 2) : (K(1) << 40) + i * i * i);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const size_t n = keys.size();
    std::vector<SosdKV> data(n);
    std::vector<Pair> v(n);
    for (size_t i = 0; i < n; ++i) data[i].key = keys[i], data[i].value = 0, v[i] = {keys[i], 0};
    SosdTip theirs;
    theirs.Build(data);
    gret::sota::TipSearch<K, true> as_authors;
    gret::sota::TipSearch<K, false> eq8;
    as_authors.build(v.data(), n), eq8.build(v.data(), n);
    size_t differ = 0, off_theirs = 0, off_eq8 = 0;
    for (size_t i = 0; i < n; ++i) {
      const int64_t t = std::min<int64_t>(std::max<int64_t>(theirs.interpolate_r(keys[i]), 0), int64_t(n) - 1), a = as_authors.first(keys[i]), e = eq8.first(keys[i]);
      differ += std::llabs(t - a) > 1;  // one position of slack for a compiler that contracts the two expressions differently
      off_theirs = std::max<size_t>(off_theirs, size_t(std::llabs(t - int64_t(i))));
      off_eq8 = std::max<size_t>(off_eq8, size_t(std::llabs(e - int64_t(i))));
    }
    if (differ) FAIL("tip first estimate, key set %d: our 'a' variant differs from SOSD's interpolate_r on %zu of %zu keys", mode, differ, n);
    if (n & 1) {  // n - 1 = 2d: our first estimate is the loop's three-point formula (Equation 2) through positions 0, d and 2d
      size_t off = 0;
      for (size_t i = 0; i < n; ++i) off += std::llabs(eq8.first(keys[i]) - gret::sota::TipSearch<K, false>::three_point(v.data(), keys[i], 0, int64_t(n / 2), int64_t(n) - 1)) > 1;
      if (off) FAIL("tip first estimate, key set %d: differs from the three-point formula through the first, middle and last key on %zu of %zu keys", mode, off, n);
    } else {
      FAIL("tip first estimate, key set %d: %zu keys, an odd count was expected", mode, n);
    }
    std::printf("tip first estimate, key set %d (n=%zu): furthest from the true position, SOSD's rm_tip.h %zu, Equation 8 %zu\n", mode, n, off_theirs, off_eq8);
    if (mode == 0) {  // keys 0, 10, ..., 20000: a straight line. Equation 8 is exact on it; the borrowed expression says 1333 for the last key
      if (theirs.interpolate_r(20000) != 1333 || theirs.interpolate_r(5000) != 667) FAIL("SOSD's interpolate_r on the line: %lld, %lld", (long long)theirs.interpolate_r(20000), (long long)theirs.interpolate_r(5000));
      if (off_eq8 > 1) FAIL("Equation 8 on the line is off by %zu", off_eq8);
    }
    if (mode == 1 && off_theirs < 900) FAIL("SOSD's interpolate_r on the shifted line is only %zu off", off_theirs);  // it answers about the middle for every key
  }
  // bsearch: SOSD's BinarySearch bounds nothing either
  std::vector<SosdKV> data(1000);
  std::vector<K> keys(1000);
  for (size_t i = 0; i < 1000; ++i) data[i].key = keys[i] = i * 7 + 3, data[i].value = 0;
  gret::sota::sosd::BinarySearch<K> bs;
  bs.Build(data);
  gret::sota::WholeArrayBound<K> w;
  w.build(keys);
  for (const K x : {K(0), K(3), K(500), K(6996), M}) {
    const auto a = bs.EqualityLookup(x);
    const KeyBound b = w.get(x);
    if (a.start != b.lo || a.stop != b.hi || b.lo != 0 || b.hi != 1000) FAIL("bsearch: SOSD [%zu, %zu), ours [%zu, %zu)", a.start, a.stop, b.lo, b.hi);
  }
  if (bs.size() != 0 || w.bytes() != sizeof(w)) FAIL("bsearch: bytes %zu and %zu", bs.size(), w.bytes());
}

static size_t model_bytes(const SearchName& s) {
  const size_t table = s.kind == SearchName::rbs || s.kind == SearchName::rbsstd || s.kind == SearchName::rbst ? 4 * ((size_t(1) << s.knob) + 1) : 0;  // 4 bytes per entry, 2^bits + 1 entries
  switch (s.kind) {
    case SearchName::whole: return sizeof(gret::sota::WholeArrayBound<K>);
    case SearchName::stdlower: return 0;
    case SearchName::rbs: return sizeof(gret::sota::RbsBound<K, 12>) + table;
    case SearchName::rbsstd: return sizeof(gret::sota::RbsStdSearch<K, 12>) + table;
    case SearchName::rbst: return sizeof(gret::sota::RbstBound<K, 12>) + table;
    case SearchName::interp: return s.gallop ? sizeof(gret::sota::InterpSearch<K, true>) : sizeof(gret::sota::InterpSearch<K, false>);
    case SearchName::tip: return sizeof(gret::sota::TipSearch<K, false>);
    case SearchName::none: break;
  }
  return 0;
}

static size_t check_index(const std::string& name, const std::vector<K>& keys, const std::string& label, size_t max_starts) {
  const std::string what = name + " on " + label;
  const size_t n = keys.size();
  const size_t before = g_fails;
  ++g_checks;
  SearchName sn;
  if (!gret::sota::parse_search_name(name, sn)) {
    FAIL("%s: not a name", what.c_str());
    return 1;
  }
  std::vector<Pair> kv(n);
  for (size_t i = 0; i < n; ++i) kv[i] = {keys[i], payload(keys[i])};
  IndexPtr idx = make(name);
  Param p(1, 0);
  try {
    if (!idx) {
      FAIL("%s: no index", what.c_str());
      return 1;
    }
    idx->init(&p);
    idx->bulk_load(kv.data(), n, &p);
  } catch (const std::exception& e) {
    FAIL("%s: threw %s", what.c_str(), e.what());
    return 1;
  }
  for (Pair& e : kv) e = {0x1111111111111111ull, 0x2222222222222222ull};  // the caller's array is the caller's: the index must not read it again
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
  auto probe_known = [&](K x, bool want) {
    K v = 0x5a5a5a5a;
    const bool got = idx->get(x, v, &p);
    if ((got != want || (got && v != payload(x)) || (!got && v != 0x5a5a5a5a)) && bad++ < 3) FAIL("%s: get(%llu) = %d, want %d", what.c_str(), (unsigned long long)x, int(got), int(want));
  };
  for (size_t i = 0; i < n; ++i) {  // both neighbours of every stored key (stored themselves inside dense runs); of every fourth in the 2M sets
    if (n > 300000 && (i & 3) != 0) continue;
    if (keys[i] != 0) probe_known(keys[i] - 1, i != 0 && keys[i - 1] == keys[i] - 1);
    if (keys[i] != M) probe_known(keys[i] + 1, i != n - 1 && keys[i + 1] == keys[i] + 1);
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

  const long long mem = idx->memory_consumption(), want_mem = (long long)(n * sizeof(Pair) + (n < 64 ? 0 : model_bytes(sn)));
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
  const size_t lens[] = {0, 1, 2, 3, 16, 256, 4096, 65536, n + 5};
  std::vector<Pair> out;
  for (const K x : starts)
    for (const size_t len : lens) {
      if (len > 256 && (g() & 31)) continue;  // long scans on a sample of the starts
      if (len > 65536 && n > 300000) continue;
      out.assign(len + 1, {0xdead, 0xbeef});  // one guard slot past the end
      const size_t got = idx->scan(x, len, out.data(), &p);
      const size_t at = lower_of(keys, x);
      const size_t want_n = std::min(len, n - at);
      bool ok = got == want_n && out[len] == Pair{0xdead, 0xbeef};
      for (size_t i = 0; ok && i < got; ++i) ok = out[i].first == keys[at + i] && out[i].second == payload(keys[at + i]);
      for (size_t i = got; ok && i < len; ++i) ok = out[i] == Pair{0xdead, 0xbeef};  // nothing written past what was returned
      if (!ok && bad++ < 3) FAIL("%s: scan(%llu, %zu) returned %zu, want %zu", what.c_str(), (unsigned long long)x, len, got, want_n);
    }
  return g_fails - before;
}

static void check_names() {
  ++g_checks;
  struct Good { const char* name; SearchName::Kind kind; unsigned knob; bool a; const char* shown; bool e = false; };
  const Good good[] = {{"bsearch", SearchName::whole, 0, false, "bsearch(reimplementation)"},
                       {"bsearchstd", SearchName::stdlower, 0, false, "bsearchstd(reimplementation)"},
                       {"rbs8", SearchName::rbs, 8, false, "rbs8"},
                       {"rbs20", SearchName::rbs, 20, false, "rbs20"},
                       {"rbs28", SearchName::rbs, 28, false, "rbs28"},
                       {"rbsstd8", SearchName::rbsstd, 8, false, "rbsstd8"},
                       {"rbsstd24", SearchName::rbsstd, 24, false, "rbsstd24"},
                       {"rbsstd28", SearchName::rbsstd, 28, false, "rbsstd28"},
                       {"rbst12", SearchName::rbst, 12, false, "rbst12(reimplementation)"},
                       {"rbst28", SearchName::rbst, 28, false, "rbst28(reimplementation)"},
                       {"interp", SearchName::interp, 8, false, "interp8(reimplementation)"},
                       {"interp1", SearchName::interp, 1, false, "interp1(reimplementation)"},
                       {"interp64", SearchName::interp, 64, false, "interp64(reimplementation)"},
                       {"interpe", SearchName::interp, 8, false, "interp8e(reimplementation)", true},
                       {"interp3e", SearchName::interp, 3, false, "interp3e(reimplementation)", true},
                       {"interp64e", SearchName::interp, 64, false, "interp64e(reimplementation)", true},
                       {"tip", SearchName::tip, 64, false, "tip64(reimplementation)"},
                       {"tip1", SearchName::tip, 1, false, "tip1(reimplementation)"},
                       {"tip4096", SearchName::tip, 4096, false, "tip4096(reimplementation)"},
                       {"tipa", SearchName::tip, 64, true, "tip64a(reimplementation)"},
                       {"tip16a", SearchName::tip, 16, true, "tip16a(reimplementation)"}};
  for (const Good& gd : good) {
    SearchName s;
    if (!gret::sota::parse_search_name(gd.name, s) || s.kind != gd.kind || s.knob != gd.knob || s.authors_first != gd.a || s.gallop != gd.e || s.shown != gd.shown) FAIL("name %s not parsed (shown '%s')", gd.name, s.shown.c_str());
    IndexPtr idx = make(gd.name);
    if (!idx) FAIL("name %s made no index", gd.name);
  }
  const char* bad[] = {"", "bs", "bsearch ", "bsearchs", "bsearchstd1", "Bsearch", "rbs", "rbs7", "rbs29", "rbs020", "rbs2o", "rbs12345", "rbs-12", "rbs 12", "rbs20t", "rbs20a", "rbsstd", "rbsstd7", "rbsstd29", "rbsstd020", "rbsstd20x", "rbsst20", "rbsstdt20", "rbs20std",
                       "rbst", "rbst7", "rbst29", "rbst012", "rbst12x", "interp0", "interp65", "interp08", "interpx", "interp8a", "interp_8", "interp99999", "interpee", "interp0e", "interp65e", "interpe8", "interp8ea", "interp e", "tip0", "tip4097",
                       "tip064", "tipb", "tip64b", "tip64aa", "tipaa", "tip99999", "tip-1", "TIP", "xtip", "tip 64", "rs32r18", "spgm16", "rl4", "ab4", "btree", "ht_std"};
  for (const char* nm : bad) {
    SearchName s;
    s.knob = 7;
    if (gret::sota::parse_search_name(nm, s) || s.knob != 7 || s.kind != SearchName::none) FAIL("name '%s' parsed", nm);
    if (Index* made = gret::sota::make_searches<K, K>(nm)) {
      FAIL("name '%s' made an index", nm);
      (void)made;
    }
    if (gret::sota::destroy_searches<K, K>(nullptr, nm)) FAIL("name '%s' destroyed something", nm);
  }
}

static void check_throws() {
  ++g_checks;
  auto model_throws = [](auto m, std::vector<K> keys) {
    try {
      m.build(keys);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  using S = gret::sota::RbsBound<K, 12>;
  using T = gret::sota::RbstBound<K, 12>;
  if (!model_throws(S{}, {}) || !model_throws(T{}, {})) FAIL("no keys: built");
  if (!model_throws(S{}, {5}) || !model_throws(T{}, {5})) FAIL("one key: built");
  if (!model_throws(S{}, {9, 9, 9}) || !model_throws(T{}, {9, 9, 9})) FAIL("one distinct key: built");
  if (model_throws(S{}, {0, M}) || model_throws(T{}, {0, M})) FAIL("{0, 2^64-1}: threw");
  if (model_throws(S{}, {0, 1}) || model_throws(T{}, {0, 1})) FAIL("{0, 1}: threw");
  auto search_throws = [](std::vector<Pair> v) {
    gret::sota::RbsStdSearch<K, 12> u;
    try {
      u.build(v.data(), v.size());
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  if (!search_throws({}) || !search_throws({{5, 0}}) || !search_throws({{9, 0}, {9, 1}}) || search_throws({{0, 0}, {M, 0}}) || search_throws({{0, 0}, {1, 0}})) FAIL("rbsstd: what cannot be built was, or the reverse");
  auto ctor_throws = [](auto make_it) {
    try {
      make_it();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  if (!ctor_throws([] { gret::sota::InterpSearch<K> s(0); }) || !ctor_throws([] { gret::sota::InterpSearch<K> s(65); })) FAIL("interp with 0 or 65 steps: made");
  if (!ctor_throws([] { gret::sota::TipSearch<K> s(0); }) || !ctor_throws([] { gret::sota::TipSearch<K> s(4097); })) FAIL("tip with guard 0 or 4097: made");
  // the searches divide by key differences: keys that are not strictly ascending are refused, loudly
  for (const char* nm : {"bsearchstd", "rbsstd12", "interp", "interpe", "tip", "tipa"}) {
    for (int what = 0; what < 3; ++what) {
      std::vector<Pair> kv;
      for (K i = 0; i < 100; ++i) kv.push_back({i * 3, i});
      if (what == 0) kv[50].first = kv[49].first;  // a repeat
      if (what == 1) std::swap(kv[10], kv[11]);    // out of order
      if (what == 2) kv.clear();                   // nothing
      IndexPtr idx = make(nm);
      Param p(1, 0);
      bool threw = false;
      try {
        idx->bulk_load(kv.data(), kv.size(), &p);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (!threw) FAIL("%s: bulk_load of bad input %d did not throw", nm, what);
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
  std::printf("timing (sanity only) %-11s n=%zu  %.1f ns/get  %.2f bytes/key  checksum %llu\n", name, kv.size(), ns, double(idx.memory_consumption()) / double(kv.size()), (unsigned long long)sum);
}

int main(int argc, char** argv) {
  bool do_small = true, do_big = true;
  int first_file = 1;
  if (argc > 1 && std::string(argv[1]) == "--small") do_big = false, first_file = 2;
  else if (argc > 1 && std::string(argv[1]) == "--big") do_small = false, first_file = 2;
  else if (argc > 1 && argv[1][0] == '-') { std::printf("usage: %s [--small | --big] [KEYS...]\n", argv[0]); return 2; }
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
  { std::vector<K> v; for (K i = 1; i <= 30000; ++i) v.push_back(i * i * i * 7); sets.push_back({"cubic", v}); }  // smooth and far from a line
  { std::vector<K> v; K k = 1; for (int i = 0; i < 6000; ++i) { k += 1 + k / 200; v.push_back(k); } sets.push_back({"exponential", v}); }  // the first keys in a corner
  { std::vector<K> v = {0}; for (K i = 0; i < 30000; ++i) v.push_back(M - 30000 + i); sets.push_back({"one_low", v}); }  // one key, then everything at the top
  { std::vector<K> v; for (K i = 0; i < 12000; ++i) v.push_back(((g() >> 30) << 32) | 0xdeadbeefu); sets.push_back({"equal_mod_2^32", v}); }  // the low 32 bits say nothing
  { std::vector<K> v; for (K i = 0; i < 12000; ++i) v.push_back((K(1) << 63) | (g() >> (1 + g() % 40))); v.push_back(K(1) << 63); v.push_back(M); sets.push_back({"above_2^63", v}); }  // a signed key would sort these first
  const size_t small_sets = sets.size();
  sets.push_back({"big_wide", rnd(0, M, 2000000)});
  { std::vector<K> v; for (size_t i = 0; i < 2000000; ++i) v.push_back(g() >> (g() % 44)); v.push_back(0); v.push_back(M); sets.push_back({"big_skew", v}); }  // every scale at once
  { std::vector<K> v; K k = g() >> 8; while (v.size() < 2000000) { const size_t run = 1 + g() % 1000; for (size_t i = 0; i < run; ++i) v.push_back(k++); k += g() >> 26; } sets.push_back({"big_runs", v}); }

  if (do_small) {
    check_names();
    check_throws();
    check_tip_first_estimate();
  }

  const char* everywhere[] = {"bsearch", "bsearchstd", "rbs8", "rbs12", "rbs16", "rbs20", "rbsstd8", "rbsstd13", "rbsstd20", "rbst8", "rbst12", "rbst20", "interp", "interp1", "interp3", "interp64", "interpe", "interp2e",
                              "tip", "tip1", "tip4", "tip16", "tip1024", "tipa", "tip2a"};
  const char* big_only[] = {"bsearch", "bsearchstd", "rbs12", "rbs20", "rbs24", "rbsstd16", "rbsstd24", "rbst20", "rbst24", "interp", "interp4e", "tip", "tip8", "tipa"};
  for (size_t s = 0; s < sets.size(); ++s) {
    const std::string& label = sets[s].first;
    std::vector<K>& keys = sets[s].second;
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const bool big = s >= small_sets;
    if (big ? !do_big : !do_small) continue;
    size_t bad = 0, cases = 0;
    const size_t fails_before = g_fails;
    const auto set_t0 = std::chrono::steady_clock::now();
    if (keys.size() >= 2)  // the models themselves, also below SortedBoundIndex's 64 keys
      for (const unsigned bits : {8u, 12u, 16u, 20u, 24u})
        if (!big || bits >= 12) check_rbs_models_at(bits, keys, label);
    if (label == "big_wide") check_rbs_models_at(28, keys, label);  // a 1 GB table, once
    if (keys.size() >= 3) {  // the searches themselves, also below the 64 keys the index hands them
      for (const unsigned steps : {1u, 2u, 3u, 8u, 64u}) {
        if (!big || steps == 8) check_interp<false>(steps, keys, label);
        if (!big || steps == 3) check_interp<true>(steps, keys, label);
      }
      for (const unsigned guard : {1u, 2u, 8u, 64u, 1024u})
        if (!big || guard == 8 || guard == 64) check_tip<false>(guard, keys, label);
      for (const unsigned guard : {1u, 64u})
        if (!big || guard == 64) check_tip<true>(guard, keys, label);
    }
    auto run = [&](const char* name) {
      bad += check_index(name, keys, label, big ? 3000 : 1500) != 0;
      ++cases;
    };
    if (big) for (const char* nm : big_only) run(nm);
    else for (const char* nm : everywhere) run(nm);
    if (label == "big_wide") run("rbs28"), run("rbsstd28"), run("rbst28");
    std::printf("%-15s n=%-8zu %zu names, %zu failed; models and searches: %zu failures  (%.1f s)\n", label.c_str(), keys.size(), cases, bad, g_fails - fails_before,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - set_t0).count());
    std::fflush(stdout);  // a run that is cut short still shows how far it got
  }

  // every path of the two searches must have been taken somewhere above
  for (const bool e : {false, true}) {
    const sd::CountTrace& c = e ? g_interp_e : g_interp;
    const size_t lookups = e ? g_interp_e_lookups : g_interp_lookups;
    std::printf("%s over the test: %zu lookups, %.2f interpolations each, %.1f%% hit the step bound (then %.1f exponential probes and a binary search over %.0f positions)\n", e ? "interp<k>e" : "interp<k>",
                lookups, double(c.steps) / double(lookups), 100.0 * double(c.finishes) / double(lookups), c.finishes ? double(c.finish_probes) / double(c.finishes) : 0.0,
                c.finishes ? double(c.finish_range) / double(c.finishes) : 0.0);
  }
  std::printf("tip over the test: %zu lookups, %.2f three-point steps each, %.1f records scanned each, %.2f%% finished by binary search\n", g_tip_lookups,
              double(g_tip.steps) / double(g_tip_lookups), double(g_tip.scanned) / double(g_tip_lookups), 100.0 * double(g_tip.fallbacks) / double(g_tip_lookups));
  if (!g_interp.steps || !g_interp.finishes || g_interp.finish_probes || !g_interp_e.steps || !g_interp_e.finishes || g_interp_e.finish_probes <= g_interp_e.finishes || !g_tip.steps ||
      !g_tip.scanned || !g_tip.fallbacks)
    FAIL("a path of interp or tip was never taken");

  if (do_big) {  // sanity timing: 1M stored keys in random order on 2M keys, beside the static PGM of readonly.h
    for (const size_t which : {small_sets, small_sets + 1}) {
      const std::vector<K>& keys = sets[which].second;
      std::printf("-- %s\n", sets[which].first.c_str());
      std::vector<Pair> kv(keys.size());
      for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
      std::vector<K> order = keys;
      std::shuffle(order.begin(), order.end(), g);
      order.resize(std::min<size_t>(order.size(), 1000000));
      for (const char* nm : {"bsearch", "bsearchstd", "rbs12", "rbs20", "rbsstd20", "rbst20", "rbs24", "rbsstd24", "rbst24", "interp", "interpe", "tip", "tipa"}) {
        IndexPtr idx = make(nm);
        time_gets(nm, *idx, kv, order);
      }
      gret::SortedBoundIndex<K, K, gret::PgmBound<K, 64>> pgm64("spgm64");
      time_gets("spgm64", pgm64, kv, order);
    }
  }

  for (int a = first_file; a < argc; ++a) {  // key files
    FILE* f = std::fopen(argv[a], "rb");
    uint64_t n = 0;
    if (!f || std::fread(&n, 8, 1, f) != 1) { std::printf("cannot read %s\n", argv[a]); return 2; }
    std::vector<K> keys(n);
    if (std::fread(keys.data(), 8, n, f) != n) { std::printf("%s: short file\n", argv[a]); return 2; }
    std::fclose(f);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    size_t bad = 0, cases = 0;
    check_rbs_models_at(24, keys, argv[a]);
    for (const char* nm : {"bsearch", "bsearchstd", "rbs24", "rbsstd24", "rbst24", "rbs28", "interp", "tip"}) {
      bad += check_index(nm, keys, argv[a], 3000) != 0;
      ++cases;
    }
    std::printf("%s n=%zu %zu names, %zu failed\n", argv[a], keys.size(), cases, bad);
  }

  if (g_fails) {
    std::printf("FAILED test_searches: %zu failures in %zu checks\n", g_fails, g_checks);
    return 1;
  }
  std::printf("PASS test_searches%s: %zu checks\n", do_small && do_big ? "" : do_small ? " --small" : " --big", g_checks);
  return 0;
}
