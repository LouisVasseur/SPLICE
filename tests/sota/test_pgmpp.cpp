// SPDX-License-Identifier: MIT
// PGM++ (gret/sota/pgmpp.h: the authors' builder and segments, the lookup written again) against a sorted array and against the
// authors' own classes.
//   model:      for every stored key the bound [lo, hi) holds its position, is inside the array and no wider than 2 * eps_l + 2 (plus
//               what the build's check added, which is expected to be nothing); for keys never stored it is a valid range; what
//               cannot be built throws; bytes are the bytes held.
//   reference:  for a list of (eps_l, eps_i) the authors' pgm::PGMIndex is built by its own constructor. Its segments and level
//               offsets must equal ours, entry for entry, both when ours come from one build() call and when the internal levels
//               are built apart from the last one (what the eps_i tuner does). Its search() in the std::upper_bound
//               configuration (BranchLessSearch = false), which is correct, must return exactly our bound for every stored key and
//               every probe, with and without layer skipping and for delta 0, 8 and 16. Its search() as published for PGM++
//               (BranchLessSearch = true, threshold 8) is run too: the stored keys it puts outside its own range are counted and
//               printed (the reason the lookup is not theirs), and where it takes the linear path (eps_i <= 8) it must equal ours.
//   index:      get on every stored key, on its neighbours, on 0, 1, 2^64 - 1, below and above the range and on random keys; scan
//               from stored, neighbouring, in-gap and extreme starts at several lengths; put, update and remove return false and
//               change nothing; memory_consumption is the pairs plus the model.
//   names:      pgmpp<eps_l>[i<eps_i>][d<delta>][n] and nothing else.
//   drift:      one line of 72M keys, a single segment long enough for its float slope to push stored keys out of their range at
//               eps_l 1: the build's check must count exactly the keys the authors' own search() loses, widen the bound, and
//               leave every key inside it (the only built-in set on which the bound is widened; about 0.6 GB).
// Key sets as in tests/test_scan.cpp (tiny, uniform, wide, dense run, outliers, clusters, stride, huge gap, bit field, keys 0,
// 2^64 - 3, 2^64 - 2 and 2^64 - 1), sizes either side of SortedBoundIndex's 64-key threshold, and three sets of 2M keys. Ends with
// a small timing, for sanity only.
// Build (docs/sota/pgmpp.md; BS = the clone of https://github.com/qyliu-hkust/bench_search, B = build/tests/sota/pgmpp):
//   build/tests/cxx    -I $BS tests/sota/test_pgmpp.cpp -o $B/t
//   build/tests/cxx-O3 -I $BS tests/sota/test_pgmpp.cpp -o $B/t-O3
// In the sanitized build one UBSan check (float-cast-overflow) is off for the functions of one file of the authors',
// exp_pgm/pgm_index.h, switched off by pgmpp.h where it includes that file; ASan and every other check stay on it, and everything
// stays on pgmpp.h and on this file. Their Segment::operator() (pgm_index.h:327) converts a double of 2^64 or more to size_t:
// their build() reaches it at :112 for every level whose last segment has a slope of about 1 or more (any dense set here), and
// their search() reaches it for keys never stored, deep in a gap behind a steep segment (this test calls their search() as the
// reference; the lookup of pgmpp.h does not run that line). With -DPGMPP_STRICT_UBSAN the check stays on and the sanitized
// test stops there, at the first dense set. See docs/sota/pgmpp.md.
//   run:    t [--only | --part I/N] [--cfg NAME]... [--gen gaps|line|runs N] [KEYS...]
//           --only          skip the built-in sets (first argument)
//           --part I/N      a share of the built-in sets, I = 0 .. N-1 (first argument): N short runs that together are the whole
//                           test, for a machine where one run may not take long
//           --cfg NAME      the configurations for the sets that follow, in place of pgmpp64 and pgmpp16i4 (may be repeated)
//           --gen KIND N    a generated set of N keys (random gaps; one line, i.e. a single segment over every position; dense
//                           runs and gaps), through the index at pgmpp64 and pgmpp16i4: the build looks up every key, then every
//                           key is looked up again, with probes, writes and scans. N = 200000000 needs about 10 GB.
//           KEYS            GRE key files (a uint64 count, then the keys): the same on each.
#include "indexInterface.h"
#include "pgm/pgm.h"  // readonly.h's PgmBound names pgm::PGMIndex (GRE's copy; the authors' one lives in gret::sota::pgmpp_authors)
#include "gret/sota/pgmpp.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
using gret::sota::PgmppConfig;
using Model = gret::sota::PgmppBound<K>;
namespace authors = gret::sota::pgmpp_authors;
// indexInterface has no virtual destructor: what make_pgmpp returns is deleted as what it is
struct DeletePgmpp {
  void operator()(Index* p) const { delete static_cast<gret::SortedBoundIndex<K, K, Model>*>(p); }
};
using IndexPtr = std::unique_ptr<Index, DeletePgmpp>;

static const K M = ~K(0);
static size_t g_fails = 0, g_checks = 0;
static size_t g_widened = 0;                          // model builds whose bound had to be widened (expected: none)
static size_t g_published_lost = 0, g_published_keys = 0, g_published_runs = 0, g_published_bad_runs = 0;  // the authors' search() as published
static double g_published_worst = 0, g_published_worst_big = 0;  // the largest share of one set's stored keys that it loses; among the sets of 1M keys and more
static std::string g_published_worst_what, g_published_worst_big_what;
static size_t g_ref_gap_differs = 0;                  // never-stored probes where the authors' position evaluation overflows and ours does not follow it
static size_t g_apart_asked = 0, g_apart_done = 0;    // reference builds with the internal levels built apart, and how many really were
static size_t g_absent_outside = 0, g_absent_probes = 0;  // never-stored keys whose lower bound was outside [lo, hi]

static K payload(K k) { return k * 2654435761u + 17; }

#define FAIL(...)                    \
  do {                               \
    if (g_fails++ < 40) {            \
      std::printf("FAIL ");          \
      std::printf(__VA_ARGS__);      \
      std::printf("\n");             \
    }                                \
  } while (0)

static std::string cfg_name(const PgmppConfig& c) {
  return "pgmpp" + std::to_string(c.eps_l) + (c.eps_i ? "i" + std::to_string(c.eps_i) : "") + (c.delta != 8 ? "d" + std::to_string(c.delta) : "") + (c.skip ? "" : "n");
}
static PgmppConfig quiet(PgmppConfig c) {
  c.quiet = true;
  return c;
}
static PgmppConfig cfg(size_t eps_l, size_t eps_i = 0, size_t delta = 8, bool skip = true, bool apart = false) {
  PgmppConfig c;
  c.eps_l = eps_l, c.eps_i = eps_i, c.delta = delta, c.skip = skip, c.apart = apart;
  return c;
}

static std::vector<K> probes(const std::vector<K>& keys, std::mt19937_64& g, size_t random_n) {
  std::vector<K> p = {0, 1, 2, M, M - 1, M - 2, M - 3, K(1) << 63, (K(1) << 63) - 1, keys.front() - 1, keys.front() + 1, keys.back() + 1, keys.back() - 1};
  for (size_t i = 0; i < random_n; ++i) {
    p.push_back(g());
    const size_t r = g() % keys.size();
    if (r + 1 < keys.size() && keys[r + 1] - keys[r] > 2) p.push_back(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));  // inside a gap
  }
  return p;
}

// the model alone. Returns its bytes (0 if it cannot be built on these keys)
static size_t check_model(const std::vector<K>& keys, const std::string& label, const PgmppConfig& c) {
  const std::string what = cfg_name(c) + " model on " + label;
  const size_t n = keys.size();
  ++g_checks;
  Model m(quiet(c));
  std::vector<K> copy = keys;
  try {
    m.build(copy);
  } catch (const std::exception& e) {
    FAIL("%s: build threw %s", what.c_str(), e.what());
    return 0;
  }
  if (m.outside() || m.widened_lo() || m.widened_hi()) {
    ++g_widened;
    std::printf("NOTE %s: %zu stored keys outside the nominal bound, widened by -%zu/+%zu\n", what.c_str(), m.outside(), m.widened_lo(), m.widened_hi());
  }
  const size_t width = 2 * c.eps_l + 2 + m.widened_lo() + m.widened_hi();
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    const gret::KeyBound b = m.get(keys[i]);
    if (!(b.lo <= i && i < b.hi && b.hi <= n && b.hi - b.lo <= width) && bad++ < 3)
      FAIL("%s: key %llu at %zu, bound [%zu, %zu), n %zu", what.c_str(), (unsigned long long)keys[i], i, b.lo, b.hi, n);
  }
  std::mt19937_64 g(3);
  for (const K x : probes(keys, g, std::min<size_t>(n, 20000))) {
    const gret::KeyBound b = m.get(x);
    if (!(b.lo <= b.hi && b.hi <= n && b.hi - b.lo <= width) && bad++ < 3) FAIL("%s: probe %llu, bound [%zu, %zu), n %zu", what.c_str(), (unsigned long long)x, b.lo, b.hi, n);
    if (x >= keys.front() && x <= keys.back()) {  // PGM promises the lower bound of a key never stored too; counted, not required
      const size_t at = size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin());
      ++g_absent_probes;
      g_absent_outside += !(b.lo <= at && at <= b.hi);
    }
  }
  // what was built is what was asked for
  if (c.eps_i && m.eps_i() != c.eps_i) FAIL("%s: eps_i %zu", what.c_str(), m.eps_i());
  if (!c.eps_i && std::find(std::begin(gret::sota::kPgmppCandidates), std::end(gret::sota::kPgmppCandidates), m.eps_i()) == std::end(gret::sota::kPgmppCandidates))
    FAIL("%s: chosen eps_i %zu is not a candidate", what.c_str(), m.eps_i());
  if (m.tuned() != (m.candidate_ns().size() == 9) || (m.tuned() && c.eps_i) || (!m.tuned() && !m.candidate_ns().empty())) FAIL("%s: tuned %d with %zu timings", what.c_str(), int(m.tuned()), m.candidate_ns().size());
  if (!c.eps_i) {  // the tuner's result is the index that eps_i builds when it is named
    PgmppConfig named = c;
    named.eps_i = m.eps_i();
    Model e(quiet(named));
    std::vector<K> again = keys;
    e.build(again);
    bool same = e.offsets() == m.offsets() && e.segment_entries() == m.segment_entries() && e.bytes() == m.bytes() && e.first_level() == m.first_level();
    for (size_t i = 0; same && i < m.segment_entries(); ++i) same = e.segment(i).key == m.segment(i).key && e.segment(i).intercept == m.segment(i).intercept && e.segment(i).slope == m.segment(i).slope;
    for (size_t i = 0; same && i < n; i += 1 + n / 5000) same = e.get(keys[i]).lo == m.get(keys[i]).lo && e.get(keys[i]).hi == m.get(keys[i]).hi;
    if (!same) FAIL("%s: the tuned index is not the one %s builds", what.c_str(), cfg_name(named).c_str());
  }
  if (m.linear_internal() != (m.eps_i() <= c.delta)) FAIL("%s: internal search kind", what.c_str());
  if (m.height() < 1 || m.first_level() >= m.height()) FAIL("%s: height %zu, start level %zu", what.c_str(), m.height(), m.first_level());
  if (!c.skip && m.first_level() != m.height() - 1) FAIL("%s: no skipping, yet lookups start at level %zu of %zu", what.c_str(), m.first_level(), m.height());
  if (c.skip && m.first_level() + 1 < m.height() && m.level_segments(m.first_level()) > std::max<size_t>(c.delta, 1)) FAIL("%s: start level has %zu segments", what.c_str(), m.level_segments(m.first_level()));
  if (c.skip && m.first_level() > 0 && m.level_segments(m.first_level() - 1) <= std::max<size_t>(c.delta, 1)) FAIL("%s: a lower level could have been the start", what.c_str());
  if (m.built_keys() != n - (keys.back() == M)) FAIL("%s: built on %zu of %zu keys", what.c_str(), m.built_keys(), n);
  // bytes: this object, 16 per segment entry, 8 per level offset and per level count
  const size_t want = sizeof(Model) + m.segment_entries() * 16 + m.offsets().size() * 8 + (m.offsets().size() - 1) * 8;
  if (m.bytes() != want) FAIL("%s: bytes %zu, want %zu", what.c_str(), m.bytes(), want);
  if (m.level_segments(0) < 1 || m.level_segments(0) > n + 1) FAIL("%s: %zu segments for %zu keys", what.c_str(), m.level_segments(0), n);
  return m.bytes();
}

// The authors' classes for one (eps_l, eps_i), built by their own constructor on the same keys, against ours.
template <size_t El, size_t Ei>
static void reference(const std::vector<K>& keys, const std::string& label) {
  std::vector<K> below = keys;
  if (below.back() == M) below.pop_back();  // their builder throws on the sentinel; ours is built without it as well
  if (below.size() < 2) return;
  ++g_checks;
  const std::string what = "reference eps_l=" + std::to_string(El) + " eps_i=" + std::to_string(Ei) + " on " + label;
  const size_t m_keys = below.size();
  std::mt19937_64 g(11);
  std::vector<K> xs = below;
  for (const K x : probes(keys, g, std::min<size_t>(keys.size(), 5000)))
    if (x <= below.back()) xs.push_back(x);  // above the largest built key their search is not ours to call (at 2^64 - 1 it runs past the sentinel)
  for (size_t i = 0; i < std::min<size_t>(m_keys, 20000); ++i) {
    const K k = below[g() % m_keys];
    if (k) xs.push_back(k - 1);
    if (k + 1 <= below.back()) xs.push_back(k + 1);
  }

  const authors::pgm::PGMIndex<K, El, Ei, false, 0, float> ref(below.begin(), below.end());  // std::upper_bound inside: correct
  const auto ref_segs = ref.get_segments();
  const auto ref_off = ref.get_levels_offsets();
  size_t ref_outside = 0;
  for (size_t i = 0; i < m_keys; ++i) {
    const auto r = ref.search(below[i]);
    ref_outside += !(r.lo <= i && i < r.hi);
  }
  if (ref_outside) std::printf("NOTE %s: the authors' std::upper_bound configuration puts %zu stored keys outside its range\n", what.c_str(), ref_outside);

  size_t bad = 0;
  for (const PgmppConfig& c : {cfg(El, Ei), cfg(El, Ei, 8, false), cfg(El, Ei, 16), cfg(El, Ei, 0), cfg(El, Ei, 0, false), cfg(El, Ei, 1024), cfg(El, Ei, 8, true, true), cfg(El, Ei, 16, false, true)}) {
    Model m(quiet(c));
    std::vector<K> copy = keys;
    m.build(copy);
    const std::string w = what + " vs " + cfg_name(c) + (c.apart ? "(internal levels apart)" : "");
    if (c.apart) ++g_apart_asked, g_apart_done += m.built_apart();
    else if (m.built_apart()) FAIL("%s: built apart without being asked", w.c_str());
    if (m.offsets() != ref_off || m.segment_entries() != ref_segs.size()) {
      FAIL("%s: %zu levels and %zu segment entries, theirs %zu and %zu", w.c_str(), m.height(), m.segment_entries(), ref_off.size() - 1, ref_segs.size());
      continue;
    }
    for (size_t i = 0; i < ref_segs.size(); ++i) {
      const auto s = m.segment(i);
      if ((s.key != ref_segs[i].key || std::memcmp(&s.slope, &ref_segs[i].slope, sizeof(float)) != 0 || s.intercept != ref_segs[i].intercept) && bad++ < 3)
        FAIL("%s: segment %zu is (%llu, %g, %u), theirs (%llu, %g, %u)", w.c_str(), i, (unsigned long long)s.key, double(s.slope), s.intercept, (unsigned long long)ref_segs[i].key,
             double(ref_segs[i].slope), unsigned(ref_segs[i].intercept));
    }
    if (m.outside() != ref_outside) FAIL("%s: %zu keys outside the nominal bound, theirs %zu", w.c_str(), m.outside(), ref_outside);
    if (m.outside()) continue;  // the bound was widened: no longer theirs
    for (size_t j = 0; j < xs.size(); ++j) {
      const K x = xs[j];
      const auto r = ref.search(x);
      const gret::KeyBound b = m.get(x);
      if (b.lo == r.lo && b.hi == r.hi) continue;
      // A key never stored, in a gap so wide that their Segment::operator() converts a double beyond size_t (undefined, and
      // different on SSE2, AVX-512 and arm64): ours goes to the next segment's start, and must then hold the lower bound.
      const size_t at = size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin());
      if (j >= m_keys && !std::binary_search(below.begin(), below.end(), x) && b.lo <= at && at <= b.hi) {
        ++g_ref_gap_differs;
        continue;
      }
      if (bad++ < 3) FAIL("%s: key %llu bound [%zu, %zu), theirs [%zu, %zu)", w.c_str(), (unsigned long long)x, b.lo, b.hi, size_t(r.lo), size_t(r.hi));
    }
  }

  // as published for PGM++ in exp_pgm/main.cpp: BranchLessSearch = true, linear scan up to eps_i = 8
  const authors::pgm::PGMIndex<K, El, Ei, true, 8, float> pub(below.begin(), below.end());
  size_t lost = 0;
  for (size_t i = 0; i < m_keys; ++i) {
    const auto r = pub.search(below[i]);
    lost += !(r.lo <= i && i < r.hi);
  }
  g_published_lost += lost, g_published_keys += m_keys, ++g_published_runs, g_published_bad_runs += lost != 0;
  const double share = double(lost) / double(m_keys);
  if (share > g_published_worst) g_published_worst = share, g_published_worst_what = what;
  if (m_keys >= 1000000 && share > g_published_worst_big) g_published_worst_big = share, g_published_worst_big_what = what;
  if (Ei <= 8) {  // their linear path is right, and is what ours does
    if (lost != ref_outside) FAIL("%s: the authors' linear path loses %zu keys", what.c_str(), lost);
    for (const K x : xs) {  // both are theirs, overflow included: equal on any one machine
      const auto r = ref.search(x), p = pub.search(x);
      if ((p.lo != r.lo || p.hi != r.hi) && bad++ < 3) FAIL("%s: key %llu, their two configurations differ", what.c_str(), (unsigned long long)x);
    }
  }
}

static void references(const std::vector<K>& keys, const std::string& label, bool big) {
  if (big) {
    reference<64, 4>(keys, label), reference<64, 16>(keys, label), reference<8, 32>(keys, label), reference<16, 8>(keys, label);
    return;
  }
  reference<64, 4>(keys, label), reference<64, 8>(keys, label), reference<64, 16>(keys, label), reference<64, 32>(keys, label), reference<64, 64>(keys, label);
  reference<64, 128>(keys, label), reference<64, 256>(keys, label), reference<64, 512>(keys, label), reference<64, 1024>(keys, label);
  reference<4, 4>(keys, label), reference<4, 16>(keys, label), reference<8, 32>(keys, label), reference<16, 8>(keys, label), reference<1, 1>(keys, label), reference<2, 9>(keys, label);
  reference<1024, 1024>(keys, label);
}

// model_bytes: what a model built apart with the same parameters holds (0: not known, eps_i is the tuner's to choose)
static size_t check_index(const std::vector<K>& keys, const std::string& label, const PgmppConfig& c, size_t model_bytes, size_t max_starts, bool light = false) {
  const std::string name = cfg_name(c), what = name + " on " + label;
  const size_t n = keys.size();
  const size_t before = g_fails;
  ++g_checks;
  std::vector<Pair> kv(n);
  for (size_t i = 0; i < n; ++i) kv[i] = {keys[i], payload(keys[i])};
  IndexPtr idx;
  Param p(1, 0);
  gret::sota::pgmpp_last_build() = gret::sota::PgmppReport{};
  try {
    idx.reset(gret::sota::make_pgmpp<K, K>(name));
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
  const gret::sota::PgmppReport built = gret::sota::pgmpp_last_build();  // of this index's model (none below 64 keys)
  if (n >= 64) {
    if (built.eps_l != c.eps_l || built.delta != c.delta || built.skip != c.skip || (c.eps_i && built.eps_i != c.eps_i) || built.bytes == 0)
      FAIL("%s: the index's model was built with eps_l %zu, eps_i %zu, delta %zu", what.c_str(), built.eps_l, built.eps_i, built.delta);
    if (built.outside || built.widened_lo || built.widened_hi) {
      ++g_widened;
      std::printf("NOTE %s: %zu stored keys outside the nominal bound, widened by -%zu/+%zu\n", what.c_str(), built.outside, built.widened_lo, built.widened_hi);
    }
    if (light)
      std::printf("  %s: eps_i %zu%s, %zu levels, %zu last-level segments, model %zu bytes, model build %.0f ms, check of every key %.0f ms\n", what.c_str(), built.eps_i, built.tuned ? " (chosen by timing)" : "",
                  built.height, built.leaf_segments, built.bytes, built.build_ms, built.verify_ms);
  } else if (built.bytes != 0) {
    FAIL("%s: a model was built for %zu keys", what.c_str(), n);
  }
  const PgmppConfig& pend = gret::sota::pgmpp_pending();
  if (pend.eps_l != 64 || pend.eps_i != 0 || pend.delta != 8 || !pend.skip || pend.apart || pend.quiet) FAIL("%s: pending parameters not put back", what.c_str());
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
  if (!light)
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

  // the pairs plus the model the index built; and that model is the size of one built apart with the same parameters
  const long long mem = idx->memory_consumption();
  if (mem != (long long)(n * sizeof(Pair) + built.bytes)) FAIL("%s: memory_consumption %lld, want %zu pairs + %zu", what.c_str(), mem, n * sizeof(Pair), built.bytes);
  if (n >= 64 && model_bytes && built.bytes != model_bytes) FAIL("%s: the index's model holds %zu bytes, one built apart %zu", what.c_str(), built.bytes, model_bytes);

  std::vector<K> starts = {0, 1, M, M - 1, M - 2, K(1) << 63};
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
  struct Good { const char* name; size_t el, ei, d; bool skip; };
  const Good good[] = {{"pgmpp64", 64, 0, 8, true},      {"pgmpp1", 1, 0, 8, true},          {"pgmpp64i16", 64, 16, 8, true},   {"pgmpp16i4", 16, 4, 8, true},
                       {"pgmpp64d16", 64, 0, 16, true},  {"pgmpp64i16d16n", 64, 16, 16, false}, {"pgmpp1024n", 1024, 0, 8, false}, {"pgmpp65536i65536d1024n", 65536, 65536, 1024, false},
                       {"pgmpp8i1d0", 8, 1, 0, true},    {"pgmpp4i1024", 4, 1024, 8, true},  {"pgmpp32d0n", 32, 0, 0, false}};
  for (const Good& gd : good) {
    PgmppConfig c;
    if (!gret::sota::parse_pgmpp_name(gd.name, c) || c.eps_l != gd.el || c.eps_i != gd.ei || c.delta != gd.d || c.skip != gd.skip || c.apart || c.quiet) FAIL("name %s not parsed", gd.name);
    if (cfg_name(c) != gd.name) FAIL("name %s comes back as %s", gd.name, cfg_name(c).c_str());
  }
  const char* bad[] = {"", "pgmpp", "pgmppi4", "pgmpp0", "pgmpp064", "pgmpp64i", "pgmpp64i0", "pgmpp64i04", "pgmpp64d", "pgmpp64d00", "pgmpp64d1025", "pgmpp65537", "pgmpp64i65537",
                       "pgmpp64x", "pgmpp64 ", "pgmpp64nn", "pgmpp64ni4", "pgmpp64d8i4", "pgmpp64i4i8", "pgmpp-64", "pgmpp+64", "pgmpp6.4", "PGMPP64", "pgmp64", "pgm64", "spgm64", "xpgmpp64",
                       "pgmpp64N", "pgmpp64I4", "pgmpp999999", "pgmpp64i4d8n_", "pgmpp64_alex", "rs32r18", "pgmpp64n8"};
  for (const char* nm : bad) {
    PgmppConfig c = cfg(7, 7, 7, false);
    if (gret::sota::parse_pgmpp_name(nm, c) || c.eps_l != 7 || c.eps_i != 7 || c.delta != 7 || c.skip) FAIL("name '%s' parsed", nm);
    IndexPtr idx(gret::sota::make_pgmpp<K, K>(nm));
    if (idx) FAIL("name '%s' made an index", nm);
  }
  // two indexes made one after the other keep their own parameters; a Model made directly has the defaults
  IndexPtr a(gret::sota::make_pgmpp<K, K>("pgmpp4i4n")), b(gret::sota::make_pgmpp<K, K>("pgmpp256d16"));
  const Model d;
  if (!a || !b || d.cfg.eps_l != 64 || d.cfg.eps_i != 0 || d.cfg.delta != 8 || !d.cfg.skip) FAIL("defaults after make_pgmpp");
}

static void check_throws() {
  ++g_checks;
  auto throws = [](const PgmppConfig& c, std::vector<K> keys) {
    Model m(quiet(c));
    try {
      m.build(keys);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  std::vector<K> some;
  for (K i = 0; i < 100; ++i) some.push_back(i * 7 + 3);
  if (!throws(cfg(64), {})) FAIL("no keys: built");
  if (!throws(cfg(64), {5})) FAIL("one key: built");
  if (!throws(cfg(64), {M})) FAIL("only 2^64-1: built");
  if (!throws(cfg(64), {5, M})) FAIL("one key below 2^64-1: built");
  if (!throws(cfg(0), some)) FAIL("eps_l 0: built");
  if (!throws(cfg(65537), some)) FAIL("eps_l 65537: built");
  if (!throws(cfg(64, 65537), some)) FAIL("eps_i 65537: built");
  if (!throws(cfg(64, 4, 1025), some)) FAIL("delta 1025: built");
  if (throws(cfg(64), {0, M - 1})) FAIL("{0, 2^64-2}: threw");
  if (throws(cfg(1, 1), {0, 1})) FAIL("{0, 1}: threw");
  if (throws(cfg(4), {0, 1, M})) FAIL("{0, 1, 2^64-1}: threw");
  {  // keys that are not strictly increasing: the authors' segmentation says so (std::logic_error), no silent model
    Model m(quiet(cfg(4, 4)));
    std::vector<K> keys = {1, 5, 3, 9, 20, 21, 22};
    bool threw = false;
    try {
      m.build(keys);
    } catch (const std::logic_error&) {
      threw = true;
    }
    if (!threw) FAIL("unsorted keys: built");
  }
}

// The guard that keeps lookups exact whatever the float slopes do, on the one kind of data found to need it: a segment over tens
// of millions of positions. 72M keys on one line (stride 3) are a single segment of slope 1/3, which is not a float; the nearest
// float is 3e-8 too steep, so the predicted position runs ahead of the key by one every 33.5M positions, and from position 67.1M
// on a key lies below its own range at eps_l 1. The authors' structure as it stands loses those keys (their own class is built
// on the same keys here, in its correct std::upper_bound configuration, and counted). The build's check of every stored key must
// count the same keys and widen the bound, after which every key must be inside it. One thread for the two builds: the authors'
// parallel segmentation would cut the line into chunks too short to drift. About 0.6 GB.
static void check_drift() {
  ++g_checks;
  const size_t n = 72000000;
  std::vector<K> keys(n);
  for (size_t i = 0; i < n; ++i) keys[i] = 1000 + K(i) * 3;
#ifdef _OPENMP
  const int threads = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
  Model m(quiet(cfg(1, 4)));
  m.build(keys);  // the largest key is not 2^64 - 1: keys is left as it is
  size_t ref_outside = 0, ref_segments = 0;
  {
    const authors::pgm::PGMIndex<K, 1, 4, false, 0, float> ref(keys.begin(), keys.end());
    ref_segments = ref.segments_count();
    for (size_t i = 0; i < n; ++i) {
      const auto r = ref.search(keys[i]);
      ref_outside += !(r.lo <= i && i < r.hi);
    }
  }
#ifdef _OPENMP
  omp_set_num_threads(threads);
#endif
  size_t bad = 0, widest = 0;
  for (size_t i = 0; i < n; ++i) {
    const gret::KeyBound b = m.get(keys[i]);
    bad += !(b.lo <= i && i < b.hi && b.hi <= n);
    widest = std::max(widest, b.hi - b.lo);
  }
  std::printf("drift           n=%-8zu one line, eps_l 1: %zu segment(s); the authors' own search() puts %zu stored keys outside its range; the build's check counted %zu and widened the bound by -%zu/+%zu; "
              "keys outside the widened bound: %zu\n", n, ref_segments, ref_outside, m.outside(), m.widened_lo(), m.widened_hi(), bad);
  if (ref_outside == 0) FAIL("drift: the authors' structure loses no key on the line (the set no longer exercises the guard)");
  if (m.outside() != ref_outside) FAIL("drift: the build's check counted %zu keys outside the nominal bound, the authors' search() loses %zu", m.outside(), ref_outside);
  if (ref_outside && m.widened_lo() + m.widened_hi() == 0) FAIL("drift: keys outside the bound and nothing widened");
  if (bad) FAIL("drift: %zu stored keys outside the bound after the build's check", bad);
  if (widest > 2 * 1 + 2 + m.widened_lo() + m.widened_hi()) FAIL("drift: a range of %zu positions", widest);
}

static void time_gets(const char* name, Index& idx, const std::vector<Pair>& kv, const std::vector<K>& order) {  // through the interface, as GRE calls it
  std::vector<Pair> copy = kv;
  Param p(1, 0);
  idx.init(&p);
  idx.bulk_load(copy.data(), copy.size(), &p);
  K sum = 0, v = 0;
  size_t lost = 0;
  double best = 1e300;
  for (int rep = 0; rep < 3; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (const K k : order) {
      lost += !idx.get(k, v, &p);
      sum += v;
    }
    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(order.size()));
  }
  if (lost) FAIL("timing %s: %zu keys lost", name, lost);
  std::printf("timing (sanity only) %-14s n=%zu  %.1f ns/get  %.3f bytes/key  checksum %llu\n", name, kv.size(), best, double(idx.memory_consumption()) / double(kv.size()), (unsigned long long)sum);
}

// a set at the benchmark's scale: through the index only (its build looks up every key, then every key is looked up again here)
static std::vector<PgmppConfig> g_large = {cfg(64), cfg(16, 4)};  // --cfg NAME replaces them

static void run_large(const std::vector<K>& keys, const std::string& label) {
  size_t bad = 0, cases = 0;
  for (const PgmppConfig& c : g_large) {
    bad += check_index(keys, label, c, 0, 3000, true) != 0;
    ++cases;
    std::fflush(stdout);
  }
  std::printf("%s n=%zu %zu configurations, %zu failed\n", label.c_str(), keys.size(), cases, bad);
}

// part of parts: the built-in sets whose number is part modulo parts (the three 2M-key sets fall into different parts), the name
// and build-refusal checks and the 72M-key line (check_drift) in part 0, the timing in the last part. 0 of 1 is everything.
static void builtin_sets(size_t part, size_t parts) {
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
  // the top of the domain: the largest stored key is 2^64 - 2 (one below PGM's sentinel), 2^64 - 3 and 2^64 - 4
  { std::vector<K> v = rnd(0, M, 30000); v.push_back(M - 1); sets.push_back({"top_m1", v}); }
  { std::vector<K> v = rnd(0, M >> 3, 30000); v.push_back(M - 2); v.push_back(M); sets.push_back({"top_m2_max", v}); }
  { std::vector<K> v = rnd(0, M, 30000); for (K j = 3; j < 300; ++j) v.push_back(M - j); sets.push_back({"top_m3_run", v}); }
  { std::vector<K> v; for (K i = 0; i < 5000; ++i) v.push_back(M - 1 - i * 3); sets.push_back({"stride_to_m1", v}); }
  // a steep segment (a dense run: slope above 1 by the error's slack) and then a gap of nearly 2^64: for a key never stored, far
  // into the gap, slope * (key - first key) is 2^64 or more, the product the authors' Segment::operator() converts undefined
  // (undefined; with a segment that is not the first, so that its intercept is not 0, the sum then wraps on arm64 and AVX-512)
  { std::vector<K> v; for (K j = 0; j < 300; ++j) v.push_back(j * 1000003); for (K i = 0; i < 200; ++i) v.push_back((K(1) << 40) + i); for (K j = 0; j < 100; ++j) v.push_back(M - 5000 + j * 37); sets.push_back({"steep_then_gap", v}); }
  { std::vector<K> v; for (K j = 0; j < 5000; ++j) v.push_back(j * 977); for (K i = 0; i < 3000; ++i) v.push_back((K(1) << 33) + i); for (K j = 0; j < 3000; ++j) v.push_back(M - 900000 + j * 211); sets.push_back({"steep_then_gap2", v}); }
  { std::vector<K> v = rnd(0, K(1) << 62, 300000); sets.push_back({"mid300k", v}); }  // enough segments at small eps_l for three and four levels
  const size_t small_sets = sets.size();
  sets.push_back({"big_wide", rnd(0, M, 2000000)});
  { std::vector<K> v; for (size_t i = 0; i < 2000000; ++i) v.push_back(g() >> (g() % 44)); v.push_back(0); v.push_back(M); sets.push_back({"big_skew", v}); }  // every scale at once
  { std::vector<K> v; K k = g() >> 8; while (v.size() < 2000000) { const size_t run = 1 + g() % 1000; for (size_t i = 0; i < run; ++i) v.push_back(k++); k += g() >> 26; } sets.push_back({"big_runs", v}); }

  if (part == 0) {
    check_names();
    check_throws();
    check_drift();
    std::fflush(stdout);
  }

  const PgmppConfig everywhere[] = {cfg(64), cfg(64, 4), cfg(64, 16), cfg(4), cfg(4, 4), cfg(8, 32, 8, false), cfg(1, 1), cfg(16, 16, 16), cfg(1024, 1024), cfg(2, 0, 0, false), cfg(32, 0, 16), cfg(3, 5, 2), cfg(64, 0, 8, false)};
  const PgmppConfig big_only[] = {cfg(64), cfg(8), cfg(16, 4), cfg(4, 64, 8, false), cfg(256, 16, 16)};
  for (size_t s = 0; s < sets.size(); ++s) {
    if (s % parts != part) continue;
    const std::string& label = sets[s].first;
    std::vector<K>& keys = sets[s].second;
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const bool big = s >= small_sets;
    size_t bad = 0, cases = 0;
    const size_t fails_before = g_fails;
    auto run = [&](const PgmppConfig& c) {
      const bool model = keys.size() - (keys.back() == M) >= 2;  // also below SortedBoundIndex's 64 keys: the model itself must hold
      const size_t mb = model ? check_model(keys, label, c) : 0;
      bad += check_index(keys, label, c, c.eps_i ? mb : 0, big ? 3000 : 1500) != 0;
      ++cases;
    };
    if (big) for (const PgmppConfig& c : big_only) run(c);
    else for (const PgmppConfig& c : everywhere) run(c);
    references(keys, label, big);
    std::printf("%-15s n=%-8zu %zu configurations, %zu failed; references %s\n", label.c_str(), keys.size(), cases, bad, g_fails == fails_before ? "equal" : "DIFFER");
    std::fflush(stdout);
  }

  if (part + 1 == parts) {  // sanity timing: stored keys in random order on 2M wide keys, beside the static PGM of readonly.h
    std::vector<K>& keys = sets[small_sets].second;
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<Pair> kv(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
    std::vector<K> order = keys;
    std::shuffle(order.begin(), order.end(), g);
    for (const char* nm : {"pgmpp64", "pgmpp64i4", "pgmpp64i16", "pgmpp16", "pgmpp16i4n", "pgmpp8"}) {
      IndexPtr idx(gret::sota::make_pgmpp<K, K>(nm));
      time_gets(nm, *idx, kv, order);
    }
    gret::SortedBoundIndex<K, K, gret::PgmBound<K, 64>> pgm64("spgm64");
    time_gets("spgm64", pgm64, kv, order);
    gret::SortedBoundIndex<K, K, gret::PgmBound<K, 16>> pgm16("spgm16");
    time_gets("spgm16", pgm16, kv, order);
  }

}

int main(int argc, char** argv) {
  const bool only = argc > 1 && std::string(argv[1]) == "--only";
  size_t part = 0, parts = 1;
  int first = only ? 2 : 1;
  if (!only && argc > 2 && std::string(argv[1]) == "--part") {
    if (std::sscanf(argv[2], "%zu/%zu", &part, &parts) != 2 || parts == 0 || part >= parts) { std::printf("--part I/N with 0 <= I < N\n"); return 2; }
    first = 3;
  }
  if (!only) builtin_sets(part, parts);
  bool cfg_given = false;
  for (int a = first; a < argc; ++a) {
    if (std::string(argv[a]) == "--cfg" && a + 1 < argc) {  // the configurations for the sets that follow
      PgmppConfig c;
      if (!gret::sota::parse_pgmpp_name(argv[++a], c)) { std::printf("--cfg %s: not a pgmpp name\n", argv[a]); return 2; }
      if (!cfg_given) g_large.clear();
      cfg_given = true;
      g_large.push_back(c);
      continue;
    }
    if (std::string(argv[a]) == "--gen" && a + 2 < argc) {  // a generated set at the benchmark's scale
      const std::string kind = argv[a + 1];
      const size_t n = std::strtoull(argv[a + 2], nullptr, 10);
      a += 2;
      std::vector<K> keys;
      keys.reserve(n);
      std::mt19937_64 h(100);
      K k = 1000;
      if (kind == "gaps") for (size_t i = 0; i < n; ++i) keys.push_back(k += 1 + (h() >> 30));  // random gaps over a wide domain
      else if (kind == "line") for (size_t i = 0; i < n; ++i) keys.push_back(k + i * 3);       // one line: a single segment over every position
      else if (kind == "runs") while (keys.size() < n) { const size_t run = 1 + h() % 100000; for (size_t i = 0; i < run && keys.size() < n; ++i) keys.push_back(k++); k += h() >> 36; }
      else { std::printf("--gen gaps|line|runs N\n"); return 2; }
      run_large(keys, "generated_" + kind);
      continue;
    }
    FILE* f = std::fopen(argv[a], "rb");
    uint64_t n = 0;
    if (!f || std::fread(&n, 8, 1, f) != 1) { std::printf("cannot read %s\n", argv[a]); return 2; }
    std::vector<K> keys(n);
    if (std::fread(keys.data(), 8, n, f) != n) { std::printf("%s: short file\n", argv[a]); return 2; }
    std::fclose(f);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    run_large(keys, argv[a]);
  }

  std::printf("model builds whose bound had to be widened: %zu\n", g_widened);
  std::printf("never-stored probes whose lower bound was outside the model's range (scan falls back to the array for them): %zu of %zu\n", g_absent_outside, g_absent_probes);
  std::printf("reference builds with the internal levels built apart from the last one: %zu of %zu asked (the others have one level, or a largest key from 2^64-3 up)\n", g_apart_done, g_apart_asked);
  if (!only && parts == 1 && g_apart_done * 4 < g_apart_asked) FAIL("only %zu of %zu builds were built apart", g_apart_done, g_apart_asked);
  std::printf("never-stored probes where the authors' search() overflows its position and differs from ours (ours holds the lower bound): %zu\n", g_ref_gap_differs);
  std::printf("the authors' search() as published (branch-free, threshold 8): %zu of %zu stored keys outside its own range, in %zu of %zu builds\n", g_published_lost, g_published_keys,
              g_published_bad_runs, g_published_runs);
  if (g_published_lost) std::printf("  at worst %.1f%% of the stored keys (%s)\n", 100.0 * g_published_worst, g_published_worst_what.c_str());
  if (g_published_worst_big > 0) std::printf("  among the sets of 1M keys and more, at worst %.1f%% (%s)\n", 100.0 * g_published_worst_big, g_published_worst_big_what.c_str());
  if (g_fails) {
    std::printf("FAILED test_pgmpp: %zu failures in %zu checks\n", g_fails, g_checks);
    return 1;
  }
  if (parts > 1) std::printf("PASS test_pgmpp, part %zu of %zu: %zu checks\n", part, parts, g_checks);
  else std::printf("PASS test_pgmpp: %zu checks\n", g_checks);
  return 0;
}
