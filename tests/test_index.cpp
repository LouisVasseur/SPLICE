// SPDX-License-Identifier: MIT
// Exactness of the gret indexes (transform_index.h, gate.h, registry.h) on GRE's own bases.
//
// Every key set is bulk-loaded with distinct payloads 3k + 1 (GRE's 123456789 for every key cannot catch a lookup
// that returns another key's payload), then every key is looked up and its payload checked. Coverage: every base
// (btree pgm alex lipp, plus artunsync on Linux) x pw512, pw4096 (through make_index), nf (synthetic flow files),
// a non-monotone test transform (the sort path), the gate and NFL's switch (each gate variant through bulk_load_as).
// The gate's decisions: every gate_/nflsw_ build must choose what gate_model.h's tree (or NFL's switch) gives, with
// gate.h's fallbacks written out again here; a hand-written depth-2 tree must route the synthetic sets to all four
// leaves; the veto runs on the placeholder tree (always raw), whatever the trained model chooses.
// Forced collisions: a dense run below a 2^64 - 1 outlier and near-duplicate clusters (PwlCdf's flat segments), a
// saturating flow, and a flow that maps every key to 0 (the sentinel base); each must reach the overflow.
// Keys that were never loaded must not crash (not on LIPP: its get() of such a key can hit RT_ASSERT) and may only
// return the payload of a loaded key with the same t.
//
//   test_index [GRE_KEY_FILE...]   each file also runs: its sorted unique keys, stride-sampled to at most 2M, on
//                                  every base x {pw512, pw4096, gate, nflsw}, plus nf if FILE.flow or
//                                  $FLOW_DIR/<basename>.flow exists
//   GRET_TEST_MAXN=N               the stride sample's size limit instead of 2M (e.g. 10000000: all of a 10M sample)
//   GRET_TEST_FULL=1               LIPP runs every synthetic case too (by default a subset: see Suite::lean)
// Build (tests/run_tests.sh): the flags and shims of scripts/build_gre.sh plus -O1 -g -fsanitize=address,undefined,
//   -I GRE/src/competitor -I include. Prints "test_index: OK" and exits 0 only if every check passed (LIPP's
//   RT_ASSERT calls exit(0): an early exit is turned into a failure).
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#if defined(__linux__)
#include "artsync/artunsync.h"
#endif
#include "gret/registry.h"

#include <unistd.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace gret;
using Keys = std::vector<uint64_t>;
using KV = std::pair<uint64_t, uint64_t>;
using Index = indexInterface<uint64_t, uint64_t>;

static int g_fail = 0, g_checks = 0;
static bool g_done = false;
#define CHECK(cond, ...)                                              \
  do {                                                                \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
      ++g_fail;                                                       \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);       \
      std::fprintf(stderr, __VA_ARGS__);                              \
      std::fputc('\n', stderr);                                       \
    }                                                                 \
  } while (0)

static uint64_t payload(uint64_t k) { return 3 * k + 1; }                          // a bijection mod 2^64
static uint64_t key_of(uint64_t v) { return (v - 1) * 0xAAAAAAAAAAAAAAABull; }  // 3^-1 mod 2^64

// ---- key sets (sorted, distinct)

static Keys sorted_unique(Keys v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}
static Keys dense(size_t n, uint64_t start) {
  Keys v(n);
  for (size_t i = 0; i < n; ++i) v[i] = start + i;
  return v;
}
static Keys uniform(size_t n, uint64_t lo, uint64_t hi, uint64_t seed) {  // in [lo, hi]
  std::mt19937_64 g(seed);
  std::uniform_int_distribution<uint64_t> d(lo, hi);
  Keys v(n);
  for (auto& k : v) k = d(g);
  return sorted_unique(v);
}
static Keys concat(Keys a, const Keys& b) {
  a.insert(a.end(), b.begin(), b.end());
  return sorted_unique(a);
}

struct Set {
  std::string name;
  Keys keys;
  bool pw_collides;  // PwlCdf(512) and PwlCdf(4096) must both send keys to the overflow
};

static const Set& by_name(const std::vector<Set>& sets, const char* name) {  // const char*: no temporary string
  for (const Set& s : sets)
    if (s.name == name) return s;
  std::fprintf(stderr, "no set %s\n", name);
  std::exit(1);
}

static std::vector<Set> synthetic_sets() {
  const size_t n = 100000;
  const uint64_t top = UINT64_MAX;
  std::vector<Set> s;
  s.push_back({"dense", dense(n, 1000), false});
  s.push_back({"random64", uniform(n, 0, top, 1), false});
  s.push_back({"above2^53", concat(uniform(n / 2, uint64_t(1) << 63, top, 2), dense(n / 2, uint64_t(1) << 62)), false});
  // fb-like: a bulk in [1e9, 1e12) and 9 outliers spread up to 2^64 - 1
  Keys fb = uniform(n - 9, 1000000000ull, 999999999999ull, 3);
  for (int i = 0; i < 8; ++i) fb.push_back((uint64_t(1) << 63) + uint64_t(i) * (uint64_t(1) << 59));
  fb.push_back(top);
  s.push_back({"fb-like", sorted_unique(fb), false});
  s.push_back({"random2^50", uniform(n, 0, uint64_t(1) << 50, 6), false});  // raw LIPP works on every platform
  // a dense run below one 2^64 - 1 outlier: the last PwlCdf segment spans ~2^64 with slope ~2^-21 (512) / 2^-24
  s.push_back({"dense+max", concat(dense(n - 1, 0), {top}), true});
  s.push_back({"two-clusters", concat(dense(n / 2, 0), dense(n / 2, uint64_t(1) << 63)), false});
  // near-duplicate clusters: 1,000 runs of 100 consecutive keys, 2^54 apart; knots fall inside runs, so segments that
  // cross a gap have slope << 1 and flatten the rest of their run
  Keys nd;
  for (uint64_t c = 0; c < 1000; ++c)
    for (uint64_t j = 0; j < 100; ++j) nd.push_back(c * (uint64_t(1) << 54) + j);
  s.push_back({"near-dup", sorted_unique(nd), true});
  s.push_back({"n=100 (K>n)", uniform(100, 0, top, 4), false});
  s.push_back({"n=1", {5}, false});
  s.push_back({"n=2", {5, top}, false});
  s.push_back({"n=3", {0, 7, uint64_t(1) << 63}, false});
  return s;
}

// ---- flow files

static std::string g_tmp;
static std::string write_flow(const std::string& tag, const Keys& keys, const std::string& body, int M) {
  const uint64_t lo = keys.front(), hi = keys.back() > lo ? keys.back() : lo + 1;
  const std::string path = g_tmp + "/gret_test_" + std::to_string(getpid()) + "_" + tag + ".flow";
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(1);
  }
  std::fprintf(f, "molflow 1 %d %llu %llu\n%s", M, (unsigned long long)lo, (unsigned long long)hi, body.c_str());
  std::fclose(f);
  return path;
}
// smooth: four logistics; saturating: one very narrow logistic (keys away from the middle give 0 or 2^52 - 1);
// zero: one logistic far to the right of [lo, hi] (u = 0, so every key maps to 0 and the base gets the sentinel)
static const char* kSmooth = "0.25 0.125 0.1\n0.25 0.375 0.1\n0.25 0.625 0.1\n0.25 0.875 0.1\n";
static const char* kSaturating = "1 0.5 1e-06\n";
static const char* kZero = "1 1000000 0.01\n";

// ---- gate trees for the tests (gate.h's GateTree format)

// the placeholder: always raw, pw512 where raw is vetoed (the veto tests must not depend on the trained model)
static constexpr GateTree kAlwaysRaw = {"*", {nullptr, nullptr, nullptr}, {0, 0, 0},
                                        {{Variant::raw, Variant::pw512}, {Variant::raw, Variant::pw512},
                                         {Variant::raw, Variant::pw512}, {Variant::raw, Variant::pw512}}};
// a depth-2 tree with four distinct leaves: log2_range <= 55 ? (gap_cv <= 0.5 ? L0 : L1) : (conflict_raw <= 50 ? L2
// : L3); dense -> L0, random2^50 -> L1, random64 -> L2, fb-like -> L3 (check_gate_decide)
static constexpr GateTree kTestTree = {"test", {"log2_range", "gap_cv", "conflict_raw"}, {55, 0.5, 50},
                                       {{Variant::pw4096, Variant::pw512}, {Variant::nf, Variant::pw4096},
                                        {Variant::raw, Variant::pw512}, {Variant::pw512, Variant::pw4096}}};
static int leaf_of(const GateTree& t, const GateLeaf& l) {  // the leaves of kTestTree are distinct
  for (int i = 0; i < 4; ++i)
    if (t.leaf[i].pick == l.pick && t.leaf[i].fallback == l.fallback) return i;
  return -1;
}

// ---- a non-monotone, colliding test transform (exercises the sort and the overflow); found by ADL

namespace testtr {
struct Hash12 {  // the top 12 bits of a multiplicative hash: 4,096 outputs, in no order
  uint64_t operator()(uint64_t k) const { return (k * 0x9E3779B97F4A7C15ull) >> 52; }
  size_t bytes() const { return sizeof(*this); }
};
template <class P>
Hash12 make_transform(Hash12*, const TransformSpec&, const std::pair<uint64_t, P>*, size_t) { return {}; }
}  // namespace testtr

// ---- checks

// Loads keys with payloads 3k + 1, then looks every key up. Returns the number of wrong lookups.
static size_t load_and_check(Index* idx, const Keys& keys, const std::string& what, bool zero_mem_ok = false) {
  std::vector<KV> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  Param param(1, 0);
  idx->init(&param);
  idx->bulk_load(kv.data(), kv.size(), &param);
  size_t bad = 0;
  for (uint64_t k : keys) {
    uint64_t v = 0;
    const bool found = idx->get(k, v, &param);
    if (!found || v != payload(k)) {
      if (bad < 3)
        std::fprintf(stderr, "  %s: get(%llu) = %d, %llu; want %llu\n", what.c_str(), (unsigned long long)k, found,
                     (unsigned long long)v, (unsigned long long)payload(k));
      ++bad;
    }
  }
  CHECK(bad == 0, "%s: %zu of %zu lookups wrong", what.c_str(), bad, keys.size());
  CHECK(zero_mem_ok || idx->memory_consumption() > 0, "%s: memory_consumption() = %lld", what.c_str(),
        idx->memory_consumption());
  uint64_t v = 0;
  CHECK(!idx->put(keys[0], 1) && !idx->update(keys[0], 1) && !idx->remove(keys[0]) && idx->scan(keys[0], 10, nullptr) == 0,
        "%s: put/update/remove/scan are not disabled", what.c_str());
  CHECK(idx->get(keys[0], v) && v == payload(keys[0]), "%s: lookup after the disabled writes", what.c_str());
  return bad;
}

// Keys that were never loaded: a hit must be the payload of a loaded key with the same t.
template <class TI>
static void probe_unloaded(TI* ti, const Keys& keys, const std::string& what) {
  std::mt19937_64 g(99);
  size_t bad = 0, hits = 0;
  for (size_t i = 0; i < 20000; ++i) {
    uint64_t q = i % 2 ? g() : keys[g() % keys.size()] + 1;
    if (std::binary_search(keys.begin(), keys.end(), q)) continue;
    uint64_t v = 0;
    if (!ti->get(q, v)) continue;
    ++hits;
    const uint64_t k = key_of(v);
    if (!std::binary_search(keys.begin(), keys.end(), k) || ti->transform()(k) != ti->transform()(q)) ++bad;
  }
  CHECK(bad == 0, "%s: %zu of %zu false positives are not a same-t key's payload", what.c_str(), bad, hits);
}

template <class Base>
struct Suite {
  using Pw = TransformIndex<uint64_t, uint64_t, Base, PwlCdf>;
  using Nf = TransformIndex<uint64_t, uint64_t, Base, MolFlow>;
  using Hx = TransformIndex<uint64_t, uint64_t, Base, testtr::Hash12>;
  using Gate = GateIndex<uint64_t, uint64_t, Base>;
  std::string base;
  bool probe;  // false for LIPP (RT_ASSERT on some unloaded keys)
  bool lean;   // LIPP: every instance first builds a pool of 10^7 nodes (~12 s under ASan), so by default it runs
               // only the sets and variants that reach its own code paths; GRET_TEST_FULL=1 runs everything

  // registry.h's raw limits for this base
  RawLimits limits() const { return RawLimits{base == "lipp" && LDBL_MANT_DIG < 64, base == "alex" || base == "pgm"}; }

  // What GateIndex must build for leaf l (gate.h's rule, written out again): pick, else fallback, else pw512; nf needs
  // $TRANSFORM_FLOW to name a readable file, raw is vetoed by the limits
  static Variant expected(const GateLeaf& l, const GateFeatures& f, RawLimits raw) {
    auto ok = [&](Variant v) {
      if (v == Variant::nf) {
        const char* p = std::getenv("TRANSFORM_FLOW");
        std::FILE* fp = p && *p ? std::fopen(p, "r") : nullptr;
        if (fp) std::fclose(fp);
        return fp != nullptr;
      }
      if (v != Variant::raw) return true;
      return !(raw.needs_53_bits && f.above_2_53 * 1000 > f.n) && !(raw.max_is_sentinel && f.n && f.max == UINT64_MAX);
    };
    return ok(l.pick) ? l.pick : ok(l.fallback) ? l.fallback : Variant::pw512;
  }

  template <class TI>
  TI* make(const std::string& name) {
    Index* idx = make_index<uint64_t, uint64_t>(name);
    TI* ti = dynamic_cast<TI*>(idx);
    CHECK(ti != nullptr, "make_index(%s) did not return the expected type", name.c_str());
    return ti;  // a wrong type leaks: indexInterface has no virtual destructor
  }

  // Returns the collided keys, or SIZE_MAX if the index could not be made.
  size_t pw(unsigned K, const Set& s) {
    const std::string name = "pw" + std::to_string(K) + "_" + base, what = s.name + " " + name;
    Pw* ti = make<Pw>(name);
    if (!ti) return SIZE_MAX;
    load_and_check(ti, s.keys, what);
    CHECK(ti->unique() + ti->collided_keys() == s.keys.size() && ti->size() == s.keys.size(), "%s: counts",
          what.c_str());
    CHECK(ti->transform().K() == K, "%s: K = %u", what.c_str(), ti->transform().K());
    if (s.pw_collides) CHECK(ti->collided_keys() > 0, "%s: the overflow path was not reached", what.c_str());
    if (probe) probe_unloaded(ti, s.keys, what);
    const size_t c = ti->collided_keys();
    delete ti;
    return c;
  }

  void nf(const Set& s, const char* tag, const char* body, int M) {
    const std::string name = "nf_" + base, what = s.name + " " + name + " (" + tag + " flow)";
    const std::string path = write_flow(std::string(tag) + "_" + base, s.keys, body, M);
    setenv("TRANSFORM_FLOW", path.c_str(), 1);
    Nf* ti = make<Nf>(name);
    if (ti) {
      load_and_check(ti, s.keys, what);
      CHECK(ti->unique() + ti->collided_keys() == s.keys.size(), "%s: counts", what.c_str());
      if (std::string(tag) == "saturating" && s.keys.size() > 1000)
        CHECK(ti->collided_keys() > s.keys.size() / 2, "%s: only %zu collided", what.c_str(), ti->collided_keys());
      if (std::string(tag) == "zero" && s.keys.size() > 1)
        CHECK(ti->unique() == 0 && ti->groups() == 1, "%s: expected one group of every key", what.c_str());
      if (probe) probe_unloaded(ti, s.keys, what);
      delete ti;
    }
    unsetenv("TRANSFORM_FLOW");
    std::remove(path.c_str());
  }

  void hash12(const Set& s) {
    const std::string what = s.name + " hash12_" + base;
    Hx* ti = new Hx("hash12_" + base);
    load_and_check(ti, s.keys, what);
    if (s.keys.size() > 10000) CHECK(ti->collided_keys() > 0 && ti->unique() > 0, "%s: expected a mix", what.c_str());
    if (probe) probe_unloaded(ti, s.keys, what);
    delete ti;
  }

  // gate_ and nflsw_ through the registry; returns the gate's choice
  Variant gate(const Set& s, bool learned) {
    const std::string name = std::string(learned ? "gate_" : "nflsw_") + base;
    Gate* gi = make<Gate>(name);
    if (!gi) return Variant::raw;
    load_and_check(gi, s.keys, s.name + " " + name, true);
    CHECK(gi->memory_consumption() > 0 || (gi->choice() == Variant::raw && base == "btree"), "%s %s: memory",
          s.name.c_str(), name.c_str());  // GRE's B+tree reports 0
    const GateFeatures f = compute_features(s.keys.data(), s.keys.size());
    bool same = true;
    for (int i = 0; i < GateFeatures::kCount; ++i) same &= f.v[i] == gi->features().v[i];
    CHECK(same, "%s %s: bulk_load's features differ from compute_features", s.name.c_str(), name.c_str());
    const Variant v = gi->choice();
    const Variant want = expected(learned ? gate_decide(gate_tree(base), f) : GateLeaf{nfl_switch(f), Variant::raw},
                                  f, limits());
    CHECK(v == want, "%s %s: chose %s, the model gives %s", s.name.c_str(), name.c_str(), variant_name(v),
          variant_name(want));
    delete gi;
    return v;
  }

  // The hand-written depth-2 tree through GateIndex: the choice follows the tree, and the chosen variant is exact
  void gate_tree2(const Set& s) {
    const std::string what = s.name + " gate_" + base + " (test tree)";
    Gate* gi = new Gate(base, GateMode::learned, limits(), &kTestTree);
    load_and_check(gi, s.keys, what, true);
    const Variant want = expected(gate_decide(kTestTree, gi->features()), gi->features(), limits());
    CHECK(gi->choice() == want, "%s: chose %s, the tree gives %s", what.c_str(), variant_name(gi->choice()),
          variant_name(want));
    delete gi;
  }

  void gate_variants(const Set& s, bool lean) {  // lean: raw and nf (pw is pw() above, through the same switch)
    const std::string path = write_flow("gate_" + base, s.keys, kSmooth, 4);
    setenv("TRANSFORM_FLOW", path.c_str(), 1);
    for (Variant v : {Variant::raw, Variant::pw512, Variant::pw4096, Variant::nf}) {
      if (lean && (v == Variant::pw512 || v == Variant::pw4096)) continue;
      const std::string what = s.name + " gate_" + base + " as " + variant_name(v);
      Gate* gi = new Gate(base, GateMode::learned);
      std::vector<KV> kv(s.keys.size());
      for (size_t i = 0; i < kv.size(); ++i) kv[i] = {s.keys[i], payload(s.keys[i])};
      Param param(1, 0);
      gi->init(&param);
      gi->bulk_load_as(v, kv.data(), kv.size(), &param);
      size_t bad = 0;
      for (uint64_t k : s.keys) {
        uint64_t val = 0;
        bad += !gi->get(k, val, &param) || val != payload(k);
      }
      CHECK(bad == 0 && gi->choice() == v, "%s: %zu wrong", what.c_str(), bad);
      CHECK(gi->memory_consumption() > 0 || (v == Variant::raw && base == "btree"), "%s: memory", what.c_str());
      delete gi;
    }
    unsetenv("TRANSFORM_FLOW");
    std::remove(path.c_str());
  }

  void synthetic(const std::vector<Set>& sets) {
    std::fprintf(stderr, "== %s\n", base.c_str());
    auto in = [](const std::string& x, std::initializer_list<const char*> l) {
      for (const char* y : l)
        if (x == y) return true;
      return false;
    };
    for (const Set& s : sets) {
      if (lean && !in(s.name, {"dense+max", "near-dup", "n=1", "n=2", "n=3"})) continue;
      pw(512, s);
      pw(4096, s);
    }
    for (const Set& s : sets) {
      if (!in(s.name, {"random64", "dense+max", "n=1", "n=2"}) || (lean && !in(s.name, {"dense+max", "n=2"}))) continue;
      if (!lean || s.name == "dense+max") {
        nf(s, "smooth", kSmooth, 4);
        nf(s, "saturating", kSaturating, 1);
      }
      nf(s, "zero", kZero, 1);
    }
    hash12({"random5000", uniform(5000, 0, UINT64_MAX, 5), false});
    const Set &dense = by_name(sets, "dense"), &above = by_name(sets, "above2^53"), &fb = by_name(sets, "fb-like");
    CHECK(gate(fb, false) == Variant::pw512, "nflsw_%s on fb-like must choose pw512", base.c_str());
    if (!lean) gate(dense, false);
    // gate_<base> with the trained model (gate() checks the choice against gate_model.h's tree and the vetoes)
    gate(dense, true);
    gate(above, true);
    const bool sentinel = base == "alex" || base == "pgm";
    if (!lean) gate(by_name(sets, "dense+max"), true);  // ALEX and PGM: 2^64 - 1 is their sentinel, raw is vetoed
    // a real depth-2 tree: four leaves (LIPP: two of them)
    for (const char* n : {"dense", "fb-like", "random2^50", "random64"})
      if (!lean || std::string(n) == "dense" || std::string(n) == "fb-like") gate_tree2(by_name(sets, n));
    gate_variants(by_name(sets, "random2^50"), lean);
    // the veto itself, on every platform, on the placeholder tree (always raw): raw is replaced by the leaf's fallback
    // (pw512) when > 0.1% of keys > 2^53. ALEX and PGM keep their sentinel veto: fb-like holds 2^64 - 1, and their raw
    // lookup of it is not reliable (GRE's own bugs, see gate.h), so this test must not depend on it
    for (const Set* s : {&above, &fb}) {
      Gate* gi = new Gate(base, GateMode::learned, RawLimits{true, sentinel}, &kAlwaysRaw);
      load_and_check(gi, s->keys, s->name + " gate_" + base + " (raw needs 53 bits)", true);
      CHECK(gi->choice() == (s == &above || sentinel ? Variant::pw512 : Variant::raw), "veto on %s", s->name.c_str());
      delete gi;
    }
  }

  void real(const std::string& file, const Keys& keys) {
    const std::string tag = file.substr(file.find_last_of('/') + 1);
    std::fprintf(stderr, "== %s %s (%zu keys)\n", tag.c_str(), base.c_str(), keys.size());
    const Set s{tag, keys, false};
    pw(512, s);
    pw(4096, s);
    std::string flow = file + ".flow";
    if (const char* d = std::getenv("FLOW_DIR")) flow = std::string(d) + "/" + tag + ".flow";
    std::FILE* ff = std::fopen(flow.c_str(), "r");
    if (ff) std::fclose(ff), setenv("TRANSFORM_FLOW", flow.c_str(), 1);  // as run_baselines.sh: the gate may pick nf
    gate(s, true);
    gate(s, false);
    if (ff) {
      Nf* ti = make<Nf>("nf_" + base);
      if (ti) {
        load_and_check(ti, keys, tag + " nf_" + base);
        delete ti;
      }
      unsetenv("TRANSFORM_FLOW");
    }
  }
};

template <class T>
static void expect(const char* name) {
  Index* idx = make_index<uint64_t, uint64_t>(name);
  T* t = dynamic_cast<T*>(idx);
  CHECK(t != nullptr, "make_index(%s) did not return the expected type", name);
  delete t;
}

static void check_registry() {
  using BT = BTreeInterface<uint64_t, uint64_t>;
  using PG = pgmInterface<uint64_t, uint64_t>;
  using AL = alexInterface<uint64_t, uint64_t>;
  for (const char* bad : {"pw1_btree", "pw65537_btree", "pw0512_btree", "pwx_btree", "pw512", "pw512_", "pw512_hot",
                          "nf_", "gate_xindex", "raw_btree", "lipp", "pw512-btree", "nflsw_btree_x"})
    CHECK((make_index<uint64_t, uint64_t>(bad) == nullptr), "make_index(%s) should be nullptr", bad);
  CHECK((make_index<uint32_t, uint64_t>("pw512_btree") == nullptr), "non-uint64 keys should give nullptr");
  if (!GRET_HAS_ARTUNSYNC) CHECK((make_index<uint64_t, uint64_t>("pw512_artunsync") == nullptr), "artunsync");
  setenv("TRANSFORM_K", "64", 1);
  auto* ti = dynamic_cast<TransformIndex<uint64_t, uint64_t, BTreeInterface<uint64_t, uint64_t>, PwlCdf>*>(
      make_index<uint64_t, uint64_t>("pw_btree"));
  CHECK(ti && ti->name() == "pw64_btree", "pw_btree with TRANSFORM_K=64");
  delete ti;
  unsetenv("TRANSFORM_K");
  expect<TransformIndex<uint64_t, uint64_t, BT, PwlCdf>>("pw2_btree");
  expect<TransformIndex<uint64_t, uint64_t, PG, PwlCdf>>("pw65536_pgm");
  expect<TransformIndex<uint64_t, uint64_t, AL, PwlCdf>>("pw512_alex");
  expect<TransformIndex<uint64_t, uint64_t, BT, MolFlow>>("nf_btree");
  expect<GateIndex<uint64_t, uint64_t, PG>>("gate_pgm");
  expect<GateIndex<uint64_t, uint64_t, AL>>("nflsw_alex");
}

static void check_features(const std::vector<Set>& sets) {
  for (const Set& s : sets) {
    const GateFeatures f = compute_features(s.keys.data(), s.keys.size());
    bool finite = true;
    for (double x : f.v) finite &= std::isfinite(x);
    CHECK(finite, "features of %s are not finite: %s", s.name.c_str(), f.list().c_str());
    CHECK(f.n == s.keys.size() && f.min == s.keys.front() && f.max == s.keys.back() && f.sample_n <= kGateSample + 1,
          "features of %s: n/min/max/sample_n", s.name.c_str());
  }
  const Set &dense = by_name(sets, "dense"), &fbs = by_name(sets, "fb-like");
  const GateFeatures d = compute_features(dense.keys.data(), dense.keys.size());
  CHECK(d.v[GateFeatures::ks_uniform] < 1e-3 && d.v[GateFeatures::central_share] > 0.99 &&
            d.v[GateFeatures::pla4_per_1k] * double(d.sample_n) / 1000 < 1.5 && d.v[GateFeatures::gap_cv] < 0.05 &&
            d.v[GateFeatures::conflict_raw] <= 2 && d.above_2_53 == 0,
        "dense features: %s", d.list().c_str());
  const GateFeatures fb = compute_features(fbs.keys.data(), fbs.keys.size());
  CHECK(fb.v[GateFeatures::central_share] < 1e-3 && fb.v[GateFeatures::conflict_raw] > 100 &&
            fb.v[GateFeatures::conflict_pw512] < fb.v[GateFeatures::conflict_raw] && fb.above_2_53 == 9,
        "fb-like features: %s", fb.list().c_str());
}

// gate_model.h is well formed, and gate_decide routes features as documented (kTestTree)
static void check_gate_decide(const std::vector<Set>& sets) {
  const size_t nt = sizeof(kGateTrees) / sizeof(kGateTrees[0]);
  CHECK(nt > 0 && std::string(kGateTrees[nt - 1].base) == "*", "gate_model.h: the last tree must be '*'");
  for (const GateTree& t : kGateTrees) {
    for (int i = 0; i < 3; ++i)
      CHECK(!t.feature[i] || (GateFeatures::id(t.feature[i]) >= 0 && std::isfinite(t.threshold[i])),
            "gate_model.h %s: node %d", t.base, i);
    for (const GateLeaf& l : t.leaf) CHECK(l.fallback != Variant::nf, "gate_model.h %s: nf as a fallback", t.base);
  }
  CHECK(std::string(gate_tree("btree").base) == "btree" || std::string(gate_tree("btree").base) == "*", "gate_tree");
  int want = 0;
  for (const char* n : {"dense", "random2^50", "random64", "fb-like"}) {
    const GateFeatures f = compute_features(by_name(sets, n).keys.data(), by_name(sets, n).keys.size());
    CHECK(leaf_of(kTestTree, gate_decide(kTestTree, f)) == want, "kTestTree: %s reaches leaf %d, not %d (%s)", n,
          leaf_of(kTestTree, gate_decide(kTestTree, f)), want, f.list().c_str());
    ++want;
  }
  GateFeatures f = compute_features(by_name(sets, "dense").keys.data(), by_name(sets, "dense").keys.size());
  f.v[GateFeatures::log2_range] = 55;  // a test holds at equality
  CHECK(leaf_of(kTestTree, gate_decide(kTestTree, f)) == 0, "kTestTree: log2_range == 55 must hold");
  f.v[GateFeatures::log2_range] = NAN;  // a NaN feature fails its test
  CHECK(leaf_of(kTestTree, gate_decide(kTestTree, f)) >= 2, "kTestTree: NaN must fail");
  GateTree one = kTestTree;  // a nullptr child always holds
  one.feature[2] = nullptr;
  f.v[GateFeatures::conflict_raw] = 1e9;
  CHECK(leaf_of(kTestTree, gate_decide(one, f)) == 2, "nullptr child must hold");
}

// GRE key file -> sorted unique keys, stride-sampled to at most maxn (plus the max)
static Keys read_gre(const char* path, size_t maxn) {
  std::FILE* f = std::fopen(path, "rb");
  uint64_t n = 0;
  Keys k;
  if (f && std::fread(&n, 8, 1, f) == 1) {
    k.resize(size_t(n));
    if (std::fread(k.data(), 8, k.size(), f) != k.size()) k.clear();
  }
  if (f) std::fclose(f);
  k = sorted_unique(k);
  const size_t stride = (k.size() + maxn - 1) / maxn;
  if (stride > 1) {  // keeps the max too: fb's few outliers would otherwise vanish from its sample
    const uint64_t mx = k.back();
    size_t j = 0;
    for (size_t i = 0; i < k.size(); i += stride) k[j++] = k[i];
    k.resize(j);
    if (k.back() != mx) k.push_back(mx);
  }
  return k;
}

static void on_exit_check() {
  if (!g_done) {
    std::fprintf(stderr, "test_index: FAILED (exited early; LIPP's RT_ASSERT calls exit(0))\n");
    _exit(1);
  }
}

int main(int argc, char** argv) {
  std::atexit(on_exit_check);
  const char* t = std::getenv("TMPDIR");
  g_tmp = t && *t ? t : "/tmp";
  if (g_tmp.size() > 1 && g_tmp.back() == '/') g_tmp.pop_back();
  const std::vector<Set> sets = synthetic_sets();
  const char* fe = std::getenv("GRET_TEST_FULL");
  const bool full = fe && std::string(fe) == "1";
  const char* mn = std::getenv("GRET_TEST_MAXN");
  const size_t maxn = mn && *mn ? size_t(std::strtoull(mn, nullptr, 10)) : 2000000;
  if (maxn < 2) {
    std::fprintf(stderr, "GRET_TEST_MAXN must be at least 2\n");
    return 1;
  }
  check_registry();
  check_features(sets);
  check_gate_decide(sets);
  Suite<BTreeInterface<uint64_t, uint64_t>>{"btree", true, false}.synthetic(sets);
  Suite<pgmInterface<uint64_t, uint64_t>>{"pgm", true, false}.synthetic(sets);
  Suite<alexInterface<uint64_t, uint64_t>>{"alex", true, false}.synthetic(sets);
  Suite<LIPPInterface<uint64_t, uint64_t>>{"lipp", false, !full}.synthetic(sets);
#if GRET_HAS_ARTUNSYNC
  Suite<ARTUnsynchronizedInterface<uint64_t, uint64_t>>{"artunsync", true, false}.synthetic(sets);
#endif
  for (int a = 1; a < argc; ++a) {
    const Keys keys = read_gre(argv[a], maxn);
    CHECK(!keys.empty(), "cannot read %s", argv[a]);
    if (keys.empty()) continue;
    Suite<BTreeInterface<uint64_t, uint64_t>>{"btree", true, false}.real(argv[a], keys);
    Suite<pgmInterface<uint64_t, uint64_t>>{"pgm", true, false}.real(argv[a], keys);
    Suite<alexInterface<uint64_t, uint64_t>>{"alex", true, false}.real(argv[a], keys);
    Suite<LIPPInterface<uint64_t, uint64_t>>{"lipp", false, false}.real(argv[a], keys);
#if GRET_HAS_ARTUNSYNC
    Suite<ARTUnsynchronizedInterface<uint64_t, uint64_t>>{"artunsync", true, false}.real(argv[a], keys);
#endif
  }
  g_done = true;
  std::fprintf(stderr, "test_index: %s (%d checks, %d failed)\n", g_fail ? "FAILED" : "OK", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
