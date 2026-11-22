// SPDX-License-Identifier: MIT
// rmi<L> (include/gret/sota/rmi.h, a REIMPLEMENTATION of the two-stage RMI), against a sorted array:
//   - the index, through GRE's interface: get on every stored key (its own payload), on keys never stored (both neighbours of the
//     stored keys, 0, 1, 2^64 - 1, below and above the range, random); scan from stored keys, their neighbours, gaps and the
//     extremes, at several lengths: exactly the keys >= the start, ascending, with their payloads; put, update and remove return
//     false and change nothing; memory_consumption is the pairs plus the model (leaves, radix table, the object);
//   - the model alone: the bound of every stored key holds its position; for the roots whose bound is proved for keys never
//     stored (all but a cubic that stayed a cubic) the bound of every probed absent key in the keys' range holds its lower bound
//     (a miss there is a failure); for the cubic the misses are counted and printed, they are not failures of the index;
//     rmi<L> has the lowest average log2 error of the five small roots; a dense run is predicted within one position;
//   - the k names (leaves and errors over the stored keys alone): the same checks of the index, which must stay exact; of the
//     model, the stored keys only (absent keys outside their bound are counted and printed: none is promised);
//   - the names: what make_rmi takes and what it refuses; a model built without a number of leaves throws.
// Key sets as in tests/test_scan.cpp (tiny, uniform, wide, dense run, outliers, clusters, stride, huge gap, bit field), plus sets
// for the arithmetic (keys whose doubles collide, keys at the top of the range, skewed gaps) and one of about 2M random keys.
// Every root (auto, linear, linear_spline, cubic, robust_linear, radix, radix18, radix22) at leaf counts from 2 to 2^22, so
// with far more keys than leaves and far more leaves than keys (leaves that hold no key).
// Ends with a small timing (2M keys, not a benchmark): ns per lookup next to the static PGM on the same footing.
//   test_rmi [KEYS...]   (GRE key files, optional: checked as well)
#include "indexInterface.h"
#include "pgm/pgm.h"
#include "gret/sota/rmi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

using Index = indexInterface<uint64_t, uint64_t>;
using Pairs = std::vector<std::pair<uint64_t, uint64_t>>;

// GRE's indexInterface has no virtual destructor: an index is deleted as the type it is, so that it frees what it holds
struct Destroy {
  template <class Model>
  static bool as(Index* i) {
    auto* p = dynamic_cast<gret::SortedBoundIndex<uint64_t, uint64_t, Model>*>(i);
    delete p;
    return p != nullptr;
  }
  void operator()(Index* i) const {
    using namespace gret::rmi_detail;
    using K = uint64_t;
    if (!i) return;
    if (as<gret::RmiBound<K, kAuto>>(i) || as<gret::RmiBound<K, kLinear>>(i) || as<gret::RmiBound<K, kLinearSpline>>(i) || as<gret::RmiBound<K, kCubic>>(i) ||
        as<gret::RmiBound<K, kRobustLinear>>(i) || as<gret::RmiBound<K, kRadix>>(i) || as<gret::RmiBound<K, kRadix18>>(i) || as<gret::RmiBound<K, kRadix22>>(i) ||
        as<gret::PgmBound<K, 16>>(i) || as<gret::PgmBound<K, 64>>(i) || as<gret::PgmBound<K, 256>>(i))
      return;
    std::printf("test_rmi: an index of a type the test does not know\n");
    std::abort();
  }
};
using Owner = std::unique_ptr<Index, Destroy>;
static const uint64_t M = ~uint64_t(0);

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

static Pairs pairs(const std::vector<uint64_t>& keys) {
  Pairs kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  return kv;
}

static size_t lower(const std::vector<uint64_t>& keys, uint64_t x) { return size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin()); }

// "rmi12rt18" -> 12, "rt18" (and "rmi12rt18k": the same)
static void split_name(const std::string& name, unsigned& log2n, std::string& root) {
  size_t p = 3;
  log2n = 0;
  while (p < name.size() && name[p] >= '0' && name[p] <= '9') log2n = log2n * 10 + unsigned(name[p++] - '0');
  root = name.substr(p);
  if (!root.empty() && root.back() == 'k') root.pop_back();
}

// the index behind GRE's interface
static size_t check_index(const std::vector<uint64_t>& keys, const std::string& label, const std::string& name, size_t max_starts) {
  Pairs kv = pairs(keys);
  const size_t n = keys.size();
  Owner idx;
  Param p(1, 0);
  try {
    idx.reset(gret::make_rmi<uint64_t, uint64_t>(name));
    if (!idx) { std::printf("FAIL %s: not a name make_rmi takes\n", name.c_str()); return 1; }
    idx->init(&p);
    idx->bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::exception& e) {
    std::printf("FAIL %s on %s: threw %s\n", name.c_str(), label.c_str(), e.what());
    return 1;
  }
  size_t bad = 0;
  auto fail = [&](const char* what, uint64_t k) {
    if (bad++ < 4) std::printf("FAIL %s on %s: %s at key %llu\n", name.c_str(), label.c_str(), what, (unsigned long long)k);
  };
  std::mt19937_64 g(11);
  const size_t step = std::max<size_t>(1, n / 400000);
  auto stored = [&](const char* when) {
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i)
      if (i < 300 || i + 300 >= n || i % step == 0)
        if (!(idx->get(keys[i], v, &p) && v == payload(keys[i]))) fail(when, keys[i]);
  };
  stored("stored key not found");
  auto absent = [&](uint64_t x) {  // a key that is not stored must not be found, and must leave the value alone
    if (std::binary_search(keys.begin(), keys.end(), x)) return;
    uint64_t v = 0x5a5a5a5a5a5a5a5aull;
    if (idx->get(x, v, &p) || v != 0x5a5a5a5a5a5a5a5aull) fail("absent key found", x);
  };
  for (size_t i = 0; i < n; ++i)
    if (i < 300 || i + 300 >= n || i % step == 0) {
      if (keys[i] != 0) absent(keys[i] - 1);
      if (keys[i] != M) absent(keys[i] + 1);
      if (i + 1 < n && keys[i + 1] - keys[i] > 2) absent(keys[i] + 1 + g() % (keys[i + 1] - keys[i] - 1));
    }
  for (uint64_t x : {uint64_t(0), uint64_t(1), M, M - 1, uint64_t(1) << 63, keys.front() / 2, keys.back() + (M - keys.back()) / 2}) absent(x);
  for (int i = 0; i < 3000; ++i) absent(g());
  for (int i = 0; i < 3000 && keys.back() > keys.front(); ++i) absent(keys.front() + g() % (keys.back() - keys.front()));

  std::vector<uint64_t> starts = {0, 1, M, M - 1, uint64_t(1) << 63};
  auto around = [&](uint64_t k) { starts.push_back(k); if (k) starts.push_back(k - 1); if (k != M) starts.push_back(k + 1); };
  for (size_t i = 0; i < std::min<size_t>(40, n); ++i) { around(keys[i]); around(keys[n - 1 - i]); }
  for (size_t i = 0; i < max_starts; ++i) {
    const size_t r = g() % n;
    around(keys[r]);
    if (r + 1 < n && keys[r + 1] - keys[r] > 2) starts.push_back(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));
    starts.push_back(g());
  }
  const size_t lens[] = {1, 2, 3, 16, 256, 4096, n + 5};
  Pairs out;
  for (uint64_t x : starts)
    for (size_t len : lens) {
      if (len >= 4096 && (g() & 15)) continue;  // long scans on a sample of the starts
      if (len > 4096 && n > 300000 && (g() & 15)) continue;
      out.assign(len, {0, 0});
      const size_t got = idx->scan(x, len, out.data(), &p);
      const size_t at = lower(keys, x), want = std::min(len, n - at);
      bool ok = got == want;
      for (size_t i = 0; ok && i < got; ++i) ok = out[i].first == keys[at + i] && out[i].second == payload(keys[at + i]);
      if (!ok) fail("scan wrong from", x);
    }

  // read-only: the writes are refused, and nothing changes
  const uint64_t some = keys[n / 2], none = keys.back() == M ? keys.front() / 2 : keys.back() + 1;
  if (idx->put(none, 1, &p) || idx->put(some, 1, &p)) fail("put returned true", some);
  if (idx->update(some, 1, &p) || idx->update(none, 1, &p)) fail("update returned true", some);
  if (idx->remove(some, &p) || idx->remove(none, &p)) fail("remove returned true", some);
  stored("stored key changed by a refused write");
  absent(none);

  // every byte: the pairs, and for 64 keys or more the leaves, the radix table and the model object
  unsigned log2n;
  std::string root;
  split_name(name, log2n, root);
  const size_t model = n < 64 ? 0 : 24 * (size_t(1) << log2n) + (root == "rt18" ? size_t(4) << 18 : root == "rt22" ? size_t(4) << 22 : 0);
  const size_t mem = size_t(idx->memory_consumption()), least = n * 16 + model, most = least + (n < 64 ? 0 : 512);
  if (mem < least || mem > most) {
    ++bad;
    std::printf("FAIL %s on %s: memory_consumption %zu, not in [%zu, %zu]\n", name.c_str(), label.c_str(), mem, least, most);
  }
  return bad;
}

struct ModelStats {
  size_t models = 0, stored = 0, absent = 0, cubic_absent = 0, cubic_missed = 0, cubic_kept = 0, cubic_as_line = 0, k_absent = 0, k_missed = 0;
};

// the model alone: Root at 2^log2n leaves. Returns failures; avg, if given, gets the average log2 error
template <int Root>
static size_t check_model(const std::vector<uint64_t>& keys, const std::string& label, unsigned log2n, ModelStats& st, double* avg = nullptr, bool keys_only = false) {
  const size_t n = keys.size();
  if (n < 2) return 0;
  const std::string what = std::string(gret::rmi_detail::root_name(Root)) + (keys_only ? " k" : "") + " 2^" + std::to_string(log2n) + " on " + label;
  gret::RmiBound<uint64_t, Root> m(log2n, keys_only);
  std::vector<uint64_t> k = keys;
  try {
    m.build(k);
  } catch (const std::exception& e) {
    std::printf("FAIL model %s: threw %s\n", what.c_str(), e.what());
    return 1;
  }
  size_t bad = 0;
  if (k != keys) ++bad, std::printf("FAIL model %s: build changed the keys\n", what.c_str());
  if (Root != gret::rmi_detail::kAuto && m.root() != Root && !(Root == gret::rmi_detail::kLinear || Root == gret::rmi_detail::kRobustLinear))
    ++bad, std::printf("FAIL model %s: root in use is %s\n", what.c_str(), gret::rmi_detail::root_name(m.root()));
  ++st.models;
  if (avg) *avg = m.avg_log2_error();
  if (m.root() == gret::rmi_detail::kCubic && !keys_only) ++(m.cubic_is_line() ? st.cubic_as_line : st.cubic_kept);
  if (m.keys_only() != keys_only || (keys_only && m.proved())) ++bad, std::printf("FAIL model %s: keys_only %d, proved %d\n", what.c_str(), int(m.keys_only()), int(m.proved()));
  uint64_t widest = 0;
  for (size_t i = 0; i < n; ++i) {
    const gret::KeyBound b = m.get(keys[i]);
    ++st.stored;
    if (!(b.lo <= i && i < b.hi && b.hi <= n)) {
      if (bad++ < 4) std::printf("FAIL model %s: stored key %llu at %zu, bound [%zu, %zu)\n", what.c_str(), (unsigned long long)keys[i], i, b.lo, b.hi);
    }
    widest = std::max<uint64_t>(widest, b.hi - b.lo);
  }
  if (widest > 2 * m.max_error() + 1) ++bad, std::printf("FAIL model %s: a bound of %llu positions, max error %llu\n", what.c_str(), (unsigned long long)widest, (unsigned long long)m.max_error());
  size_t asked = 0, missed = 0;
  uint64_t worst = 0;
  auto probe = [&](uint64_t x) {  // absent keys inside the keys' range: where SortedBoundIndex asks the model
    if (x < keys.front() || x > keys.back()) return;
    const size_t at = lower(keys, x);
    if (at < n && keys[at] == x) return;
    const gret::KeyBound b = m.get(x);
    ++asked;
    if (b.lo <= at && at <= b.hi && b.hi <= n) return;
    if (missed++ == 0) worst = x;
  };
  std::mt19937_64 g(5);
  const size_t step = std::max<size_t>(1, n / 200000);
  for (size_t i = 0; i + 1 < n; i += step) {
    const uint64_t gap = keys[i + 1] - keys[i];
    if (gap < 2) continue;
    probe(keys[i] + 1), probe(keys[i + 1] - 1), probe(keys[i] + gap / 2), probe(keys[i] + 1 + g() % (gap - 1));
    if (gap > 4096) probe(keys[i] + 2047), probe(keys[i] + 2048), probe(keys[i] + 2049), probe(keys[i + 1] - 2048);
  }
  for (int i = 0; i < 20000; ++i) probe(keys.front() + g() % (keys.back() - keys.front()));
  if (keys_only) {  // nothing is promised: counted, to show what the k names give up
    st.k_absent += asked, st.k_missed += missed;
  } else if (m.proved()) {
    st.absent += asked;
    if (missed) ++bad, std::printf("FAIL model %s: %zu of %zu absent keys outside their bound, e.g. %llu\n", what.c_str(), missed, asked, (unsigned long long)worst);
  } else {
    st.cubic_absent += asked, st.cubic_missed += missed;
    if (missed) std::printf("note  model %s: %zu of %zu absent keys outside the cubic root's bound, e.g. %llu (not proved; get and scan stay exact)\n", what.c_str(), missed, asked, (unsigned long long)worst);
  }
  return bad;
}

static size_t check_models(const std::vector<uint64_t>& keys, const std::string& label, unsigned log2n, ModelStats& st) {
  using namespace gret::rmi_detail;
  if (keys.size() < 2) return 0;
  double avg[8] = {0};
  size_t bad = 0;
  bad += check_model<kAuto>(keys, label, log2n, st, &avg[7]);
  bad += check_model<kLinear>(keys, label, log2n, st, &avg[0]);
  bad += check_model<kLinearSpline>(keys, label, log2n, st, &avg[1]);
  bad += check_model<kCubic>(keys, label, log2n, st, &avg[2]);
  bad += check_model<kRobustLinear>(keys, label, log2n, st, &avg[3]);
  bad += check_model<kRadix>(keys, label, log2n, st, &avg[4]);
  bad += check_model<kRadix18>(keys, label, log2n, st);
  bad += check_model<kRadix22>(keys, label, log2n, st);
  const double best = *std::min_element(avg, avg + 5);
  if (avg[7] != best) ++bad, std::printf("FAIL model auto 2^%u on %s: average log2 error %.6f, the best small root has %.6f\n", log2n, label.c_str(), avg[7], best);
  // the k leaves: every root again; rmi<L>k again the best of the five small ones
  bad += check_model<kAuto>(keys, label, log2n, st, &avg[7], true);
  bad += check_model<kLinear>(keys, label, log2n, st, &avg[0], true);
  bad += check_model<kLinearSpline>(keys, label, log2n, st, &avg[1], true);
  bad += check_model<kCubic>(keys, label, log2n, st, &avg[2], true);
  bad += check_model<kRobustLinear>(keys, label, log2n, st, &avg[3], true);
  bad += check_model<kRadix>(keys, label, log2n, st, &avg[4], true);
  bad += check_model<kRadix18>(keys, label, log2n, st, nullptr, true);
  const double best_k = *std::min_element(avg, avg + 5);
  if (avg[7] != best_k) ++bad, std::printf("FAIL model auto k 2^%u on %s: average log2 error %.6f, the best small root has %.6f\n", log2n, label.c_str(), avg[7], best_k);
  return bad;
}

template <class I>
static double ns_per_lookup(I& idx, const std::vector<uint64_t>& probes, uint64_t& sum) {
  const auto t0 = std::chrono::steady_clock::now();
  uint64_t v = 0;
  for (uint64_t k : probes) sum += idx.get(k, v) ? v : 1;
  return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(probes.size());
}

int main(int argc, char** argv) {
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  std::vector<std::pair<std::string, std::vector<uint64_t>>> sets = {
      {"n1", {5}}, {"n1max", {M}}, {"n1zero", {0}}, {"n2", {0, M}}, {"n3", {1, 2, M}}, {"n5", {0, 1, 2, 3, M}}, {"n17", rnd(0, uint64_t(1) << 40, 17)},
      {"n64", rnd(0, M, 64)}, {"n65", rnd(1000, 100000, 65)}, {"uniform", rnd(1, uint64_t(1) << 50, 20000)}, {"wide", rnd(0, M, 20000)}};
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 50000; ++i) v.push_back(1000 + i); sets.push_back({"run", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), 20000); for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003); v.push_back(3); v.push_back(M); sets.push_back({"outliers", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + 100000, 30000); for (uint64_t j = 0; j < 100; ++j) v.push_back(M - j * 1000); sets.push_back({"over64outliers", v}); }
  { std::vector<uint64_t> v; for (int c = 0; c < 300; ++c) { const uint64_t b = g() >> 1; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } sets.push_back({"clusters", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 20000; ++i) v.push_back(i * 65536 + 12345); sets.push_back({"stride", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 10000; ++i) { v.push_back(i); v.push_back(M - i); } sets.push_back({"hugegap", v}); }
  { std::vector<uint64_t> v; for (uint64_t a = 0; a < 300; ++a) for (uint64_t b = 0; b < 200; b += 1 + (a & 3)) v.push_back((a << 32) | (b << 8) | 7); v.push_back(M); sets.push_back({"bitfield_max", v}); }
  // for the arithmetic: keys that are one double, a few doubles, at the very top; gaps over many orders of magnitude; a CDF that bends
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 500; ++i) v.push_back((uint64_t(1) << 62) + i); sets.push_back({"one_double", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 6000; ++i) v.push_back((uint64_t(1) << 63) + i * 3); sets.push_back({"few_doubles", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 9000; ++i) v.push_back(M - i * 5); sets.push_back({"top", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 9000; ++i) v.push_back(i * 5); sets.push_back({"bottom", v}); }
  { std::vector<uint64_t> v; uint64_t k = 0; for (int i = 0; i < 30000; ++i) { k += uint64_t(1) << (g() % 44); v.push_back(k); } sets.push_back({"skewed_gaps", v}); }
  { std::vector<uint64_t> v; std::lognormal_distribution<double> d(0.0, 2.0); for (int i = 0; i < 40000; ++i) v.push_back(uint64_t(std::min(d(g) * 1e12, 1.8e19))); sets.push_back({"lognormal", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 1; i <= 30000; ++i) v.push_back(i * i * i); sets.push_back({"cubes", v}); }
  { std::vector<uint64_t> v = rnd(0, 1000000, 30000); for (uint64_t x : rnd(uint64_t(1) << 60, M, 300)) v.push_back(x); sets.push_back({"two_scales", v}); }
  for (auto& s : sets) {
    std::sort(s.second.begin(), s.second.end());
    s.second.erase(std::unique(s.second.begin(), s.second.end()), s.second.end());
  }

  size_t bad = 0, cases = 0;
  {  // the names
    const char* good[] = {"rmi1", "rmi9", "rmi10", "rmi28", "rmi20lin", "rmi20spl", "rmi20cub", "rmi20rob", "rmi20rdx", "rmi20rt18", "rmi20rt22",
                          "rmi1k", "rmi28k", "rmi20link", "rmi20splk", "rmi20cubk", "rmi20robk", "rmi20rdxk", "rmi20rt18k", "rmi20rt22k"};
    const char* refused[] = {"", "rmi", "rmi0", "rmi00", "rmi020", "rmi29", "rmi100", "rmi1000", "rmi20x", "rmi20linx", "rmi20rt", "rmi20rt19", "rmi-1", "rmi 20", "rmi20 ", "Rmi20",
                             "rmilin", "spgm16", "rmi20(reimplementation)", "pgm", "rm", "xrmi20", "rmik", "rmik20", "rmi20kk", "rmi20klin", "rmi0k", "rmi29k", "rmi20K"};
    for (const char* nm : good) {
      Owner i(gret::make_rmi<uint64_t, uint64_t>(nm));
      if (!i) ++bad, std::printf("FAIL name %s: refused\n", nm);
      ++cases;
    }
    for (const char* nm : refused) {
      Owner i(gret::make_rmi<uint64_t, uint64_t>(nm));
      if (i) ++bad, std::printf("FAIL name '%s': taken\n", nm);
      ++cases;
    }
    if (gret::make_rmi<uint32_t, uint64_t>("rmi20") != nullptr) ++bad, std::printf("FAIL rmi20 with uint32_t keys: taken\n");
    // a model that was given no number of leaves refuses to build, and one index's number does not leak into the next model
    Owner first(gret::make_rmi<uint64_t, uint64_t>("rmi12"));
    gret::RmiBound<uint64_t> orphan;
    std::vector<uint64_t> k = sets[9].second;
    bool threw = false;
    try { orphan.build(k); } catch (const std::logic_error&) { threw = true; }
    if (!threw || orphan.log2_leaves() != 0) ++bad, std::printf("FAIL a default-constructed RmiBound built with 2^%u leaves\n", orphan.log2_leaves());
    Owner second(gret::make_rmi<uint64_t, uint64_t>("rmi12k"));
    gret::RmiBound<uint64_t> after;
    if (after.log2_leaves() != 0 || after.keys_only()) ++bad, std::printf("FAIL the settings of rmi12k leaked into the next model\n");
    cases += 3;
  }

  const char* names[] = {"rmi1", "rmi3", "rmi8", "rmi14", "rmi20", "rmi6lin", "rmi6spl", "rmi6cub", "rmi6rob", "rmi6rdx", "rmi6rt18", "rmi6rt22",
                         "rmi13lin", "rmi13spl", "rmi13cub", "rmi13rob", "rmi13rdx", "rmi13rt18", "rmi13rt22", "rmi19cub", "rmi19rdx", "rmi19rt18",
                         "rmi1k", "rmi5k", "rmi11k", "rmi17k", "rmi9link", "rmi9splk", "rmi9cubk", "rmi9robk", "rmi9rdxk", "rmi9rt18k", "rmi9rt22k"};
  for (const auto& [label, keys] : sets)
    for (const char* nm : names) bad += check_index(keys, label, nm, 300) != 0, ++cases;

  ModelStats st;
  for (const auto& [label, keys] : sets)
    for (unsigned log2n : {1u, 2u, 5u, 10u, 16u}) bad += check_models(keys, label, log2n, st) != 0, ++cases;
  {  // a dense run is a line: every root predicts it within one position
    using namespace gret::rmi_detail;
    const std::vector<uint64_t>& run = sets[11].second;
    std::vector<uint64_t> k = run;
    for (unsigned log2n : {1u, 4u, 8u, 12u}) {
      gret::RmiBound<uint64_t, kLinear> a(log2n);
      gret::RmiBound<uint64_t, kLinearSpline> b(log2n);
      gret::RmiBound<uint64_t, kCubic> c(log2n);
      gret::RmiBound<uint64_t, kRadix> d(log2n);
      gret::RmiBound<uint64_t, kAuto> e(log2n);
      a.build(k), b.build(k), c.build(k), d.build(k), e.build(k);
      const uint64_t worst = std::max({a.max_error(), b.max_error(), c.max_error(), d.max_error(), e.max_error()});
      if (worst > 1) ++bad, std::printf("FAIL run at 2^%u leaves: max error %llu on a dense run\n", log2n, (unsigned long long)worst);
      ++cases;
    }
  }

  // about 2M random keys: uniform, and clustered (a mixture of dense and sparse stretches)
  std::vector<uint64_t> big = rnd(0, M >> 4, 2000000), lumpy;
  {
    uint64_t k = 0;
    for (int c = 0; c < 2000; ++c) {
      const unsigned shift = unsigned(g() % 30);
      k += g() >> 12;
      for (int i = 0; i < 1000; ++i) lumpy.push_back(k += 1 + (g() >> 34 >> shift));
    }
  }
  for (std::vector<uint64_t>* v : {&big, &lumpy}) {
    std::sort(v->begin(), v->end());
    v->erase(std::unique(v->begin(), v->end()), v->end());
  }
  for (const char* nm : {"rmi8", "rmi16", "rmi22", "rmi16cub", "rmi16rob", "rmi16rdx", "rmi18rt18", "rmi20rt22", "rmi8k", "rmi16k", "rmi22k", "rmi16cubk", "rmi18rt18k"}) {
    bad += check_index(big, "uniform2M", nm, 1500) != 0;
    bad += check_index(lumpy, "lumpy2M", nm, 1500) != 0;
    cases += 2;
  }
  for (unsigned log2n : {12u, 20u}) {
    bad += check_models(big, "uniform2M", log2n, st) != 0;
    bad += check_models(lumpy, "lumpy2M", log2n, st) != 0;
    cases += 2;
  }

  for (int a = 1; a < argc; ++a) {  // GRE key files: a count, then the keys
    FILE* f = std::fopen(argv[a], "rb");
    uint64_t n = 0;
    if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
    std::vector<uint64_t> k(n);
    if (std::fread(k.data(), 8, n, f) != n) return 2;
    std::fclose(f);
    std::sort(k.begin(), k.end());
    k.erase(std::unique(k.begin(), k.end()), k.end());
    for (const char* nm : {"rmi10", "rmi16", "rmi20", "rmi24", "rmi20cub", "rmi20rob", "rmi20rt22", "rmi16k", "rmi20k", "rmi24k"}) bad += check_index(k, argv[a], nm, 3000) != 0, ++cases;
    bad += check_models(k, argv[a], 20, st) != 0, ++cases;
  }

  std::printf("models: %zu built; %zu stored keys inside their bound; %zu absent keys inside a proved bound; cubic roots: %zu kept, %zu replaced by the line, "
              "%zu of %zu absent keys outside a cubic's (unproved) bound; k names: %zu of %zu absent keys outside their bound (none promised)\n",
              st.models, st.stored, st.absent, st.cubic_kept, st.cubic_as_line, st.cubic_missed, st.cubic_absent, st.k_missed, st.k_absent);

  {  // a small timing, for sanity only: 2M keys, 2M lookups of stored keys in random order, one thread
    std::printf("timing (2M keys, stored keys in random order, one thread; not a benchmark):\n");
    for (const auto* v : {&big, &lumpy}) {
      Pairs kv = pairs(*v);
#ifdef NDEBUG
      std::vector<uint64_t> probes(2000000);
#else
      std::vector<uint64_t> probes(200000);  // the sanitized build: a tenth
#endif
      for (uint64_t& q : probes) q = (*v)[g() % v->size()];
      uint64_t sum = 0;
      const char* set = v == &big ? "uniform2M" : "lumpy2M";
      auto run = [&](const std::string& nm, Index* raw) {
        Owner idx(raw);
        try {  // the comparison lines are not what is tested here: one that cannot be built says so and the rest goes on
          idx->init();
          idx->bulk_load(kv.data(), kv.size());
        } catch (const std::exception& e) {
          std::printf("  %-10s %-11s not built: %s\n", set, nm.c_str(), e.what());
          return;
        }
        ns_per_lookup(*idx, probes, sum);
        const double ns = ns_per_lookup(*idx, probes, sum);
        std::printf("  %-10s %-11s %7.1f ns/lookup  %10.3f MiB of model\n", set, nm.c_str(), ns, double(size_t(idx->memory_consumption()) - kv.size() * 16) / 1048576.0);
      };
      for (const char* nm : {"rmi10", "rmi14", "rmi18", "rmi18cub", "rmi18rob", "rmi18rt18", "rmi22", "rmi10k", "rmi14k", "rmi18k", "rmi22k"}) run(nm, gret::make_rmi<uint64_t, uint64_t>(nm));
      run("spgm16", new gret::SortedBoundIndex<uint64_t, uint64_t, gret::PgmBound<uint64_t, 16>>("spgm16"));
      run("spgm64", new gret::SortedBoundIndex<uint64_t, uint64_t, gret::PgmBound<uint64_t, 64>>("spgm64"));
      run("spgm256", new gret::SortedBoundIndex<uint64_t, uint64_t, gret::PgmBound<uint64_t, 256>>("spgm256"));
      if (sum == 0) std::printf("  (no payload read)\n");
    }
  }

  std::printf("test_rmi: %s (%zu cases, %zu failed)\n", bad ? "FAILED" : "ok", cases, bad);
  return bad ? 1 : 0;
}
