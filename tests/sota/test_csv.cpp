// SPDX-License-Identifier: MIT
// csv_lipp and csv_alex (include/gret/sota/csv.h: our reimplementation of CSV, arXiv 2408.06134, on GRE's LIPP and ALEX)
// against a std::map: get on every stored key and on keys never stored (neighbours, 0, 2^63 - 1, 2^63, 2^64 - 1, random,
// below and above the range, and every virtual key the pass accepted), scan from stored, neighbouring, in-between and extreme
// start keys, then put (neighbours of stored keys, runs, keys far outside, duplicates), update and remove, the same checks
// after each, and finally every key removed and the index filled again. The key sets are test_scan's patterns (tiny sets,
// uniform, wide, dense runs, clusters, far outliers, 0 and 2^64 - 1, a regular stride), three sets made so that the pass has
// something to accept (nested clusters, a staircase of densities, a lognormal), and two of 2M keys.
// Before those: Algorithm 1 alone on the paper's own example (the ten keys of its Fig. 2, alpha = 0.5), which must give the
// paper's five virtual points in the paper's order and the losses of its Table 2, and properties of it on random key sets.
// After those: the other settings (CSV_LAMBDA=floor, CSV_LIPP_RULE=loss, a CSV_BUDGET_S too small for the pass, CSV_COST_C,
// CSV_SEARCH_W = 0, malformed values). First in the alex group: the put of a stored key that splits ALEX's root data node
// (test_alex_root_split, a defect the verifier found).
//
// LIPP's RT_ASSERT calls exit(0): an exit before the end of main is turned into a failure (the atexit handler below).
// Every LIPP object allocates a pool of 10^7 two-key nodes in its constructor (lipp.h:83; about 8 s and 4.7 GB under ASan):
// the csv_lipp cases share one object per alpha, loaded again for each set.
//
// Build (docs/sota/csv.md; B = build/tests/sota/csv):
//   printf '[float-cast-overflow]\nsrc:*/competitor/alex/*\n[signed-integer-overflow]\nsrc:*/competitor/alex/*\n[vptr]\nsrc:*/competitor/alex/*\n[address]\nfun:*AlexDataNode*Iterator*initialize*\n' > $B/san_alex.ignore
//   build/tests/cxx    -fsanitize-ignorelist=$B/san_alex.ignore tests/sota/test_csv.cpp -o $B/t
//   build/tests/cxx-O3 tests/sota/test_csv.cpp -o $B/t-O3
// Without the ignore list the sanitized test stops inside ALEX's own code, on the first of four things ALEX does on ordinary
// keys (docs/sota/csv.md, "What the sanitizers report in the bases", has file and line for each):
//   float-cast-overflow      alex_base.h:79: a predicted slot outside int cast to int (then clamped)
//   signed-integer-overflow  alex_base.h:164: -x on a bitmap word that is 2^63
//   vptr                     alex_fanout_tree.h:353 and on: ALEX casts its data nodes to AlexDataNode<T, P> with the default
//                            template arguments, another type than the allow_duplicates = false nodes GRE instantiates
//   address                  alex_nodes.h:622: Iterator::initialize reads the bitmap word after the last one
// The list takes these four checks off the files under GRE's src/competitor/alex (the fourth off one function); every other
// check stays on them, and the wrapper, LIPP and this test stay fully checked. The LIPP groups pass without the list.
//
//   t [GROUP...] [set:NAME] [--time]   only the named groups (smooth, alex, alex2m, lipp, lipp2m, modes), only the key set
//                                      NAME; --time: a small lookup timing against GRE's LIPP and ALEX (use t-O3)
#include "indexInterface.h"
#include "alex/alex.h"  // GRE's own interface classes first, as GRE's competitor.h has them: the header must live next to them
#include "lipp/lipp.h"
#include "gret/sota/csv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

using Index = indexInterface<uint64_t, uint64_t>;
using Map = std::map<uint64_t, uint64_t>;
using KV = std::pair<uint64_t, uint64_t>;
static const uint64_t M = ~uint64_t(0);
static const uint64_t H = uint64_t(1) << 63;

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

static size_t g_bad = 0;
static bool g_done = false;
static void early_exit() {
  if (g_done) return;
  std::fprintf(stderr, "test_csv: FAILED (exited early; LIPP's RT_ASSERT calls exit(0))\n");
  _exit(1);
}
static void fail(const std::string& what, const char* op, uint64_t k) {
  if (g_bad++ < 40) std::printf("FAIL %s: %s wrong at key %llu\n", what.c_str(), op, (unsigned long long)k);
}

// either wrapper behind one face, for the counters the checks read
struct Handle {
  gret::CsvLipp* lipp = nullptr;
  gret::CsvAlex* alex = nullptr;
  double alpha = 0;
  Index* idx() const { return lipp ? static_cast<Index*>(lipp) : alex; }
  size_t in_base() const { return lipp ? lipp->in_base() : alex->in_base(); }
  size_t in_side() const { return lipp ? lipp->in_side() : alex->in_side(); }
  size_t n_virtual() const { return lipp ? lipp->n_virtual() : alex->n_virtual(); }
  const gret::csv_detail::PassStats& pass() const { return lipp ? lipp->pass() : alex->pass(); }
  const std::vector<uint64_t>& virtual_keys() const { return lipp ? lipp->virtual_keys() : alex->virtual_keys(); }
  bool consistent() const {
    if (alex) return alex->consistent();
    lipp->verify_base();  // exits through RT_ASSERT (caught by early_exit) if LIPP's sizes are off
    return true;
  }
};
static Handle make(const char* name) {
  Handle h;
  Index* i = gret::make_csv<uint64_t, uint64_t>(name);
  h.lipp = dynamic_cast<gret::CsvLipp*>(i), h.alex = dynamic_cast<gret::CsvAlex*>(i);
  if (!h.lipp && !h.alex) std::printf("FAIL %s: no such index\n", name), ++g_bad;
  if (h.lipp) h.lipp->keep_virtual(true);
  if (h.alex) h.alex->keep_virtual(true);
  const std::string digits = std::string(name).substr(8);
  h.alpha = digits.empty() ? 0.1 : std::atof(digits.c_str()) / 100.0;
  return h;
}
static void destroy(Handle& h) {  // deleted as what it is: indexInterface has no virtual destructor
  delete h.lipp;
  delete h.alex;
  h = Handle();
}

// every answer against the reference: get on every key it holds and on probes, scans from starts
static void verify(Index& idx, const Map& ref, const std::vector<uint64_t>& probes, std::mt19937_64& g, const std::string& what, bool full) {
  std::vector<KV> all(ref.begin(), ref.end());
  uint64_t v = 0;
  for (const auto& kv : all)
    if (!idx.get(kv.first, v) || v != kv.second) fail(what, "get", kv.first);
  auto probe = [&](uint64_t k) {
    uint64_t val = 0;
    const bool got = idx.get(k, val);
    const auto it = ref.find(k);
    if (got != (it != ref.end()) || (got && val != it->second)) fail(what, "get (probe)", k);
  };
  for (size_t i = 0; i < probes.size(); i += full ? 1 : 1 + probes.size() / 20000) probe(probes[i]), probe(probes[i] + 1), probe(probes[i] - 1);
  for (uint64_t k : {uint64_t(0), uint64_t(1), M, M - 1, H, H - 1, H + 1, H - 2}) probe(k);
  if (!all.empty()) {  // below and above the range
    const uint64_t lo = all.front().first, hi = all.back().first;
    for (uint64_t d : {uint64_t(1), uint64_t(2), uint64_t(1000), uint64_t(1) << 20, uint64_t(1) << 40, uint64_t(1) << 62}) {
      if (lo >= d) probe(lo - d);
      if (hi <= M - d) probe(hi + d);
    }
    probe(lo / 2), probe(hi + (M - hi) / 2);
  }
  for (int i = 0; i < 20000; ++i) probe(g());
  const int around_n = full ? 20000 : 4000;
  for (int i = 0; i < around_n && !all.empty(); ++i) {  // in the gaps and next to stored keys
    const size_t r = g() % all.size();
    probe(all[r].first + 1), probe(all[r].first - 1);
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) probe(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  std::vector<uint64_t> starts = {0, 1, M, M - 1, H, H - 1, H + 1};
  auto around = [&](uint64_t k) { starts.push_back(k), starts.push_back(k + 1), starts.push_back(k - 1); };
  for (size_t i = 0; i < std::min<size_t>(40, all.size()); ++i) around(all[i].first), around(all[all.size() - 1 - i].first);
  for (int i = 0; i < (full ? 1500 : 400) && !all.empty(); ++i) {
    const size_t r = g() % all.size();
    around(all[r].first);
    if (r + 1 < all.size() && all[r + 1].first - all[r].first > 2) starts.push_back(all[r].first + 1 + g() % (all[r + 1].first - all[r].first - 1));
  }
  for (size_t i = 0; i < probes.size() && i < 300; ++i) around(probes[i]);
  for (int i = 0; i < 100; ++i) starts.push_back(g());
  const size_t lens[] = {1, 2, 3, 16, 256, 5000, 70000};
  std::vector<KV> out;
  auto less = [](const KV& a, uint64_t b) { return a.first < b; };
  for (uint64_t s : starts)
    for (size_t len : lens) {
      if (len > 256 && (g() & 31)) continue;  // long scans on a sample of the starts
      out.assign(len + 1, {0, 0});
      const size_t got = idx.scan(s, len, out.data());
      const size_t at = size_t(std::lower_bound(all.begin(), all.end(), s, less) - all.begin());
      bool ok = got == std::min(len, all.size() - at) && out[len] == KV(0, 0);
      for (size_t i = 0; ok && i < got; ++i) ok = out[i] == all[at + i];
      if (!ok) fail(what, "scan", s);
    }
  {  // one scan over everything
    out.assign(all.size() + 7, {0, 0});
    const size_t got = idx.scan(0, all.size() + 7, out.data());
    bool ok = got == all.size();
    for (size_t i = 0; ok && i < got; ++i) ok = out[i] == all[i];
    if (!ok) fail(what, "full scan", 0);
  }
}

struct Expect {
  double side_max = 0;      // the largest share of the keys the side store may take at the bulk load
  double side_min = 0;      // and the smallest (sets made to hit the guards)
  bool accepts = false;     // the pass must keep at least one rebuilt subtree on this set
  bool writes = true;       // put, update, remove, empty, refill
  size_t far_puts = 3000;   // puts of keys anywhere in the 64-bit range
};

static size_t g_accepted = 0, g_virtual = 0;

static void check(const std::string& label, std::vector<uint64_t> keys, Handle& h, const char* name, const Expect& ex) {
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  const std::string what = std::string(name) + " on " + label;
  const size_t bad0 = g_bad;
  const bool big = keys.size() > 500000;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<KV> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  if (kv.size() > 10) kv[kv.size() / 3].second = M, kv[2 * kv.size() / 3].second = M - 1, kv[kv.size() / 2].second = 0;  // no payload is reserved
  Index* idx = h.idx();
  Param p(1, 0);
  idx->init(&p);
  try {
    idx->bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::exception& e) {
    std::printf("FAIL %s: bulk_load threw %s\n", what.c_str(), e.what());
    ++g_bad;
    return;
  }
  const size_t side0 = h.in_side(), in0 = h.in_base(), virt0 = h.n_virtual(), acc0 = h.pass().accepted;
  g_accepted += acc0, g_virtual += virt0;
  if (in0 + side0 != keys.size()) fail(what, "bulk_load holds another number of keys than", keys.size());
  if (double(side0) > ex.side_max * double(keys.size()) + 2 || double(side0) < ex.side_min * double(keys.size()))
    fail(what, "the side store's share at the bulk load is outside what this set should give; side keys:", side0);
  if (ex.accepts && (acc0 == 0 || virt0 == 0)) fail(what, "the pass kept no rebuilt subtree on a set made for it; subtrees smoothed:", h.pass().smoothed);
  if (!h.consistent()) fail(what, "the base's structure is inconsistent after the bulk load, keys:", in0);
  const long long bytes = idx->memory_consumption();
  if (bytes < (long long)(16 * keys.size())) fail(what, "memory_consumption below 16 bytes a key:", uint64_t(bytes));

  std::mt19937_64 g(std::hash<std::string>()(label) ^ 0x9e3779b97f4a7c15ull);
  Map ref(kv.begin(), kv.end());
  // the virtual keys: at most alpha * n, strictly between stored keys, never one of them, never returned by get or scan
  std::vector<uint64_t> probes = h.virtual_keys();
  {
    if (probes.size() != virt0) fail(what, "the virtual key list has another length than the count:", probes.size());
    // lambda = alpha * |K| per subtree: alpha * n in all with the integer part, one more per smoothed subtree when rounded up
    const char* lam = std::getenv("CSV_LAMBDA");
    const double most = h.alpha * double(keys.size()) + (lam && !std::strcmp(lam, "floor") ? 0.0 : double(h.pass().smoothed)) + 1e-6;
    if (double(virt0) > most) fail(what, "more virtual keys than the budget allows:", virt0);
    if (h.alpha == 0 && (virt0 || h.pass().candidates)) fail(what, "alpha = 0 must not run the pass; candidates:", h.pass().candidates);
    KV one[2];
    for (size_t i = 0; i < probes.size(); ++i) {
      const uint64_t v = probes[i];
      uint64_t val = 0;
      if (i && probes[i - 1] >= v) fail(what, "virtual keys not ascending at", v);
      if (ref.count(v)) fail(what, "a virtual key equals a stored key:", v);
      if (keys.empty() || v <= keys.front() || v >= keys.back()) fail(what, "a virtual key outside the keys:", v);
      if (idx->get(v, val)) fail(what, "get returns a virtual key:", v);
      if (idx->scan(v, 1, one) == 1 && one[0].first == v) fail(what, "scan returns a virtual key:", v);
    }
  }
  verify(*idx, ref, probes, g, what + " (loaded)", !big);
  if (!ex.writes || keys.empty()) {
    std::printf("%-11s %-12s n=%-8zu base=%zu side=%zu virtual=%zu accepted=%zu/%zu promoted=%zu %.1f B/key  %s  %.1fs\n", name, label.c_str(), keys.size(), in0, side0, virt0, acc0,
                h.pass().smoothed, h.pass().promoted, keys.empty() ? 0.0 : double(bytes) / double(keys.size()), g_bad == bad0 ? "ok" : "FAILED",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::fflush(stdout);
    return;
  }

  auto pick = [&]() { return keys[g() % keys.size()]; };
  const int scale = big ? 4 : 1;
  // put: neighbours of stored keys, runs behind a stored key, keys anywhere in the range, keys far outside, special keys,
  // the virtual keys themselves (a virtual point is an empty slot: it must take a real key), duplicates
  {
    std::vector<KV> add;
    for (int i = 0; i < 6000 * scale; ++i) { const uint64_t k = pick(); add.push_back({k + 1, payload(k + 1)}); if (i % 3 == 0) add.push_back({k + 2, payload(k + 2)}); if (i % 7 == 0) add.push_back({k - 1, payload(k - 1)}); }
    for (int r = 0; r < 4; ++r) { const uint64_t k = pick(); for (uint64_t j = 1; j <= 1500; ++j) add.push_back({k + j * (r + 1), payload(k + j)}); }
    for (int i = 0; i < 6000 * scale; ++i) { const uint64_t span = keys.back() - keys.front(), k = keys.front() + (span == M ? g() : g() % (span + 1)); add.push_back({k, payload(k)}); }
    for (size_t i = 0; i < ex.far_puts; ++i) { const uint64_t k = g(); add.push_back({k, payload(k)}); }
    for (uint64_t j = 0; j < 40; ++j) {
      add.push_back({keys.back() + 1 + j, j}), add.push_back({keys.back() + (uint64_t(1) << (j + 4)), j}), add.push_back({keys.front() - 1 - j, j});
      add.push_back({keys.front() >> (j % 20 + 1), j}), add.push_back({keys.back() + (keys.back() - keys.front()) * (j + 1), j});
    }
    for (uint64_t k : {uint64_t(0), uint64_t(1), M, M - 1, H, H - 1, H + 1, H - 2}) add.push_back({k, payload(k)});
    for (size_t i = 0; i < probes.size(); i += 2) add.push_back({probes[i], payload(probes[i]) ^ 3});
    for (int i = 0; i < 2000; ++i) add.push_back({pick(), 99});  // mostly there already
    std::shuffle(add.begin(), add.end(), g);
    for (const auto& a : add) {
      const bool fresh = ref.emplace(a.first, a.second).second;
      if (idx->put(a.first, a.second) != fresh) fail(what, "put", a.first);
      probes.push_back(a.first);
    }
    if (h.in_base() + h.in_side() != ref.size()) fail(what, "after put the index counts another number of keys than", ref.size());
    if (!h.consistent()) fail(what, "the base's structure is inconsistent after the puts, keys:", h.in_base());
    verify(*idx, ref, probes, g, what + " (put)", !big);
  }
  // update: present keys, absent keys return false and change nothing
  {
    for (int i = 0; i < 20000; ++i) {
      uint64_t k = i % 4 ? pick() : probes[g() % probes.size()];
      if (i % 9 == 0) k += 1;
      const uint64_t v = i % 50 == 0 ? M : g();
      const auto it = ref.find(k);
      if (idx->update(k, v) != (it != ref.end())) fail(what, "update", k);
      if (it != ref.end()) it->second = v;
    }
    verify(*idx, ref, probes, g, what + " (updated)", false);
  }
  // remove: random keys (loaded ones and put ones) and both ends; absent keys and a second remove return false
  {
    std::vector<uint64_t> gone;
    for (int i = 0; i < 20000; ++i) gone.push_back(i % 3 ? pick() : probes[g() % probes.size()]);
    for (size_t i = 0; i < 60 && i < keys.size(); ++i) gone.push_back(keys[i]), gone.push_back(keys[keys.size() - 1 - i]);
    for (uint64_t k : gone) {
      const bool there = ref.erase(k) != 0;
      if (idx->remove(k) != there) fail(what, "remove", k);
      if (idx->remove(k)) fail(what, "second remove", k);
      probes.push_back(k);
    }
    for (int i = 0; i < 2000; ++i) {
      const uint64_t k = pick() + 1;
      if (idx->remove(k) != (ref.erase(k) != 0)) fail(what, "remove (neighbour)", k);
    }
    if (!h.consistent()) fail(what, "the base's structure is inconsistent after the removes, keys:", h.in_base());
    verify(*idx, ref, probes, g, what + " (removed)", false);
    for (size_t i = 0; i < gone.size(); i += 2) {
      const uint64_t k = gone[i] + (i % 4 == 0);
      if (idx->put(k, payload(k) ^ 1) != ref.emplace(k, payload(k) ^ 1).second) fail(what, "put (after remove)", k);
      probes.push_back(k);
    }
    verify(*idx, ref, probes, g, what + " (put after remove)", false);
  }
  const size_t side1 = h.in_side(), in1 = h.in_base();
  if (!big) {  // every key out, in a random order; then an index that holds nothing, and takes keys again
    std::vector<uint64_t> order;
    for (const auto& kvp : ref) order.push_back(kvp.first);
    std::shuffle(order.begin(), order.end(), g);
    for (uint64_t k : order)
      if (!idx->remove(k)) fail(what, "remove (emptying)", k);
    ref.clear();
    if (h.in_base() || h.in_side()) fail(what, "emptied, but the index still counts keys:", h.in_base() + h.in_side());
    probes.resize(std::min<size_t>(probes.size(), 20000));
    for (size_t i = 0; i < 20000; ++i) probes.push_back(order[i % order.size()]);
    verify(*idx, ref, probes, g, what + " (emptied)", false);
    for (size_t i = 0; i < 30000 && i < order.size(); ++i) {
      const uint64_t k = order[i] + (i % 2);
      if (idx->put(k, k ^ 7) != ref.emplace(k, k ^ 7).second) fail(what, "put (into the emptied index)", k);
    }
    verify(*idx, ref, probes, g, what + " (refilled)", false);
  }
  std::printf("%-11s %-12s n=%-8zu base=%zu side=%zu (%.3f%%) virtual=%zu accepted=%zu/%zu promoted=%zu %.1f B/key; after writes: base=%zu side=%zu  %s  %.1fs\n", name, label.c_str(),
              keys.size(), in0, side0, 100.0 * double(side0) / double(keys.size()), virt0, acc0, h.pass().smoothed, h.pass().promoted, double(bytes) / double(keys.size()), in1, side1,
              g_bad == bad0 ? "ok" : "FAILED", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  std::fflush(stdout);
}

// the squared error of the least-squares line rank ~ key over a sorted key set, in long double: the paper's loss L
static long double sse(const std::vector<uint64_t>& k) {
  const long double n = (long double)k.size();
  long double sx = 0, sy = 0;
  for (size_t i = 0; i < k.size(); ++i) sx += (long double)(k[i] - k[0]), sy += (long double)i;
  const long double mx = sx / n, my = sy / n;
  long double cxx = 0, cxy = 0, cyy = 0;
  for (size_t i = 0; i < k.size(); ++i) {
    const long double dx = (long double)(k[i] - k[0]) - mx, dy = (long double)i - my;
    cxx += dx * dx, cxy += dx * dy, cyy += dy * dy;
  }
  return cxx > 0 ? cyy - cxy * cxy / cxx : cyy;
}

// Algorithm 1 (gret::csv_detail::smooth) by itself
static void test_smooth() {
  const size_t bad0 = g_bad;
  auto any = [](uint64_t, uint64_t, uint64_t) { return true; };
  gret::csv_detail::Scratch w;
  {  // the paper's example: Fig. 2 (the ten keys, read off the figure's coordinates) and Table 2 (alpha = 0.5: five points)
    const std::vector<uint64_t> paper_keys = {2, 4, 6, 7, 8, 10, 14, 20, 26, 32}, paper_points = {23, 28, 18, 16, 24};
    std::vector<uint64_t> reg = paper_keys;
    std::vector<uint8_t> isv(reg.size(), 0);
    if (std::fabs(double(sse(reg)) - 8.327) > 5e-4) std::printf("FAIL smooth: the loss of the paper's Fig. 2 keys is %.4f, the paper has 8.327\n", double(sse(reg))), ++g_bad;
    for (size_t r = 0; r < paper_points.size(); ++r) {  // one point a call: the order in which they come
      const std::vector<uint64_t> before = reg;
      if (gret::csv_detail::smooth(reg, isv, 1, any, w) != 1) { std::printf("FAIL smooth: no point %zu on the paper's example\n", r + 1), ++g_bad; break; }
      std::vector<uint64_t> added;
      std::set_difference(reg.begin(), reg.end(), before.begin(), before.end(), std::back_inserter(added));
      if (added.size() != 1 || added[0] != paper_points[r])
        std::printf("FAIL smooth: point %zu on the paper's example is %llu, the paper's Fig. 2b has %llu\n", r + 1, (unsigned long long)(added.empty() ? 0 : added[0]), (unsigned long long)paper_points[r]), ++g_bad;
    }
    if (std::fabs(double(sse(reg)) - 2.293) > 5e-4) std::printf("FAIL smooth: the loss after five points is %.4f, the paper's Table 2 has 2.293\n", double(sse(reg))), ++g_bad;
    std::vector<uint64_t> all = paper_keys;  // and the five in one call
    isv.assign(all.size(), 0);
    if (gret::csv_detail::smooth(all, isv, gret::csv_detail::lambda_of(0.5, all.size(), true), any, w) != 5 || all != reg) std::printf("FAIL smooth: one call with lambda = 5 differs from five calls\n"), ++g_bad;
    for (size_t i = 0; i < all.size(); ++i)
      if (bool(isv[i]) != !std::binary_search(paper_keys.begin(), paper_keys.end(), all[i])) std::printf("FAIL smooth: the virtual flag of %llu is wrong\n", (unsigned long long)all[i]), ++g_bad;
  }
  // lambda as a count
  using gret::csv_detail::lambda_of;
  if (lambda_of(0.1, 30, true) != 3 || lambda_of(0.1, 30, false) != 3 || lambda_of(0.1, 5, true) != 1 || lambda_of(0.1, 5, false) != 0 || lambda_of(0.5, 10, true) != 5 ||
      lambda_of(0.05, 21, true) != 2 || lambda_of(0.05, 21, false) != 1 || lambda_of(0, 1000, true) != 0 || lambda_of(0.8, 2, false) != 1)
    std::printf("FAIL lambda_of\n"), ++g_bad;
  // on random key sets: the result is sorted and unique, holds every key, flags exactly the others, never has more than
  // lambda of them, every one strictly inside the keys and allowed by the filter, the loss never rises, and each single
  // point is as good as the best integer of an exhaustive search over every free position (small sets)
  std::mt19937_64 g(11);
  size_t points = 0, exact = 0;
  for (int t = 0; t < 400; ++t) {
    const size_t m = 3 + g() % (t < 300 ? 40 : 3000);
    const unsigned bits = 4 + unsigned(g() % (t % 5 == 4 ? 60 : 16));
    std::vector<uint64_t> keys;
    uint64_t k = t % 7 == 0 ? (~uint64_t(0) >> 1) : g() % 1000;
    for (size_t i = 0; i < m; ++i) k += 1 + ((g() & ((uint64_t(1) << bits) - 1)) >> (g() % bits)) * (g() % 4 == 0), keys.push_back(k);
    if (!std::is_sorted(keys.begin(), keys.end())) continue;  // wrapped past 2^64
    const size_t lambda = 1 + g() % (m / 2 + 1);
    const bool filtered = t % 3 == 0;
    auto even = [&](uint64_t v, uint64_t l, uint64_t r) { return !filtered || (v % 2 == 0 && v > l && v < r); };
    std::vector<uint64_t> reg = keys;
    std::vector<uint8_t> isv(m, 0);
    const size_t added = gret::csv_detail::smooth(reg, isv, lambda, even, w);
    points += added;
    bool ok = added <= lambda && reg.size() == m + added && isv.size() == reg.size() && std::is_sorted(reg.begin(), reg.end()) && std::adjacent_find(reg.begin(), reg.end()) == reg.end();
    size_t real = 0;
    for (size_t i = 0; ok && i < reg.size(); ++i) {
      const bool is_key = std::binary_search(keys.begin(), keys.end(), reg[i]);
      ok = bool(isv[i]) == !is_key && (is_key || (reg[i] > keys.front() && reg[i] < keys.back() && (!filtered || reg[i] % 2 == 0)));
      real += is_key;
    }
    ok = ok && real == m && (added == 0 || sse(reg) < sse(keys));
    if (!ok) std::printf("FAIL smooth: random set %d (m=%zu lambda=%zu added=%zu)\n", t, m, lambda, added), ++g_bad;
    if (m <= 40 && bits <= 10 && !filtered) {  // one point against every free integer
      std::vector<uint64_t> one = keys;
      std::vector<uint8_t> f(m, 0);
      const size_t a1 = gret::csv_detail::smooth(one, f, 1, any, w);
      long double best = sse(keys);
      for (uint64_t c = keys.front() + 1; c < keys.back(); ++c) {
        if (std::binary_search(keys.begin(), keys.end(), c)) continue;
        std::vector<uint64_t> with = keys;
        with.insert(std::upper_bound(with.begin(), with.end(), c), c);
        best = std::min(best, sse(with));
      }
      const long double got = a1 ? sse(one) : sse(keys);
      if (got > best + 1e-9L * (1 + best)) std::printf("FAIL smooth: random set %d: one point gives loss %.9Lf, the best free integer %.9Lf\n", t, got, best), ++g_bad;
      ++exact;
    }
  }
  std::printf("smooth      the paper's Fig. 2 / Table 2 example, %zu points on 400 random sets, %zu single points against exhaustive search  %s\n", points, exact, g_bad == bad0 ? "ok" : "FAILED");
  std::fflush(stdout);
}

// A put of a key that is already stored can be the call that splits ALEX's root data node: AlexDataNode::insert looks at
// whether the node is full or too costly before it looks for the key, and Alex::insert then splits the node and fixes ALEX's
// key domain to the keys the node holds. The wrapper has to take the new domain from that call too, although the call
// returns "not inserted". When it did not (found by the verifier's interleaved runs: a dense run with one far key), later
// puts of keys outside the domain went to ALEX, whose root expansion left a corrupted domain and threw
// std::bad_array_new_length. Here: a root data node with one far key, its first keys removed, then far-apart new keys (they
// make ALEX's inserts shift a lot, which is what makes it split) and stored keys put in turn; the split comes on a stored
// key's put. Then keys on both sides of the new domain. CsvAlex::consistent() checks ALEX's domain against the wrapper's.
static void test_alex_root_split() {
  const size_t bad0 = g_bad;
  size_t on_stored = 0;
  for (int variant = 0; variant < 4; ++variant) {
    const std::string what = "csv_alex0 on rootsplit:" + std::to_string(variant);
    Handle h = make("csv_alex0");
    if (!h.alex) return;
    Index* idx = h.idx();
    const uint64_t step = variant & 2 ? 1 : 4, far = variant & 2 ? M - 1 : uint64_t(1) << 62;
    std::vector<KV> kv;
    for (uint64_t i = 0; i < 50000; ++i) kv.push_back({1000 + step * i, payload(i)});
    kv.push_back({far, 7});
    Map ref(kv.begin(), kv.end());
    std::mt19937_64 g(100 + variant);
    try {
      idx->bulk_load(kv.data(), kv.size());
      if (h.alex->root_is_model_node()) fail(what, "ALEX's root is not a data node after the bulk load; keys:", kv.size());
      for (uint64_t i = 0; i < 3000; ++i)  // the root data node no longer starts at the bulk load's first key
        if (idx->remove(1000 + step * i) != (ref.erase(1000 + step * i) != 0)) fail(what, "remove", 1000 + step * i);
      bool split = false;
      for (uint64_t i = 0; i < 20000 && !split; ++i) {  // a new key and a stored key in turn (or the other way round)
        const uint64_t fresh = far / 2 + g() % (far / 4), stored = 1000 + step * (3000 + g() % 47000);
        for (int j = 0; j < 2 && !split; ++j) {
          const bool is_fresh = j == (variant & 1);
          const uint64_t k = is_fresh ? fresh : stored;
          if (idx->put(k, k ^ 5) != ref.emplace(k, k ^ 5).second) fail(what, "put", k);
          if (h.alex->root_is_model_node()) split = true, on_stored += !is_fresh;
        }
      }
      if (!split) fail(what, "ALEX's root data node never split; keys in ALEX:", h.in_base());
      if (!h.consistent()) fail(what, "inconsistent (ALEX's key domain against the wrapper's) after the put that split the root; keys in ALEX:", h.in_base());
      const size_t in_base1 = h.in_base();
      std::vector<uint64_t> probes;
      for (uint64_t i = 0; i < 3000; ++i) {  // below the first key the node held when it split, and above the last
        for (uint64_t k : {1000 + step * i, far + 1 - (uint64_t(i) << 40)}) {
          if (idx->put(k, k + 1) != ref.emplace(k, k + 1).second) fail(what, "put (outside)", k);
          probes.push_back(k);
        }
      }
      if (h.in_base() > in_base1 + 3000) fail(what, "keys below ALEX's key domain were handed to ALEX; keys in ALEX:", h.in_base());
      if (!h.consistent()) fail(what, "inconsistent (ALEX's key domain against the wrapper's) after the puts outside the domain; keys in ALEX:", h.in_base());
      for (int i = 0; i < 20000; ++i) {  // and ordinary writes go on
        const uint64_t k = g() % 3 ? 1000 + step * (g() % 50000) + g() % 3 : g() % 2 ? g() : far / 2 + g() % (far / 4);
        if (g() % 3) {
          if (idx->put(k, i) != ref.emplace(k, i).second) fail(what, "put (after)", k);
        } else if (idx->remove(k) != (ref.erase(k) != 0)) {
          fail(what, "remove (after)", k);
        }
        if (i % 16 == 0) probes.push_back(k);
      }
      if (!h.consistent()) fail(what, "inconsistent after the writes that followed; keys in ALEX:", h.in_base());
      verify(*idx, ref, probes, g, what, true);
    } catch (const std::exception& e) {
      std::printf("FAIL %s: threw %s\n", what.c_str(), e.what()), ++g_bad;
    }
    destroy(h);
  }
  if (on_stored == 0) std::printf("FAIL rootsplit: in no variant did a stored key's put split ALEX's root data node, the case tests nothing\n"), ++g_bad;
  std::printf("csv_alex0   rootsplit    ALEX's root data node split by a put, in %zu of 4 variants by the put of a stored key  %s\n", on_stored, g_bad == bad0 ? "ok" : "FAILED");
  std::fflush(stdout);
}

template <class I>
static void time_one(const char* name, I* idx, const std::vector<KV>& kv) {
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
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    t0 = std::chrono::steady_clock::now();
    for (uint64_t x : q) sum += idx->get(x, v, &p) ? v : 1;
    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(q.size()));
  }
  std::printf("time %-11s n=%zu build %.2fs  get %.1f ns  %.1f B/key  [checksum %llu]\n", name, kv.size(), build, best, double(idx->memory_consumption()) / double(kv.size()),
              (unsigned long long)sum);
  std::fflush(stdout);
  // deleted as what it is (indexInterface has no virtual destructor): a LIPP object holds gigabytes of spare nodes
  if (auto* l = dynamic_cast<gret::CsvLipp*>(idx)) {
    delete l;
  } else if (auto* a = dynamic_cast<gret::CsvAlex*>(idx)) {
    delete a;
  } else {
    if constexpr (!std::is_same<I, Index>::value) delete idx;
  }
}

int main(int argc, char** argv) {
  std::atexit(early_exit);
  std::vector<std::string> only;
  bool timing = false;
  std::string one_set;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--time")) timing = true;
    else if (!std::strncmp(argv[i], "set:", 4)) one_set = argv[i] + 4;
    else only.push_back(argv[i]);
  auto want = [&](const std::string& s) { return only.empty() || std::find(only.begin(), only.end(), s) != only.end(); };
  for (const char* bad : {"csv", "csv_", "csv_lip", "csv_lipp_", "csv_lipp05", "csv_lipp100", "csv_lipp1x", "csv_alex-1", "csv_alex 5", "csv_btree", "csv_lipp5 ", "xcsv_lipp"})
    if (gret::make_csv<uint64_t, uint64_t>(bad)) std::printf("FAIL make_csv accepts the malformed name '%s'\n", bad), ++g_bad;
  if (gret::make_csv<uint64_t, uint32_t>("csv_alex") || gret::make_csv<uint32_t, uint64_t>("csv_alex")) std::printf("FAIL make_csv accepts other key or payload types\n"), ++g_bad;
  for (const char* good : {"csv_alex", "csv_alex0", "csv_alex5", "csv_alex80", "csv_alex99"}) {
    Handle h = make(good);
    destroy(h);
  }
  if (want("smooth")) test_smooth();

  const size_t N = 20000;
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; v.reserve(m); std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  // nested clusters: 40 clusters, each of sub-clusters whose sizes and spreads vary: subtrees with subtrees
  auto nested = [&](size_t per) {
    std::vector<uint64_t> v;
    for (int c = 0; c < 40; ++c) {
      const uint64_t b = (g() >> 14) & ~((uint64_t(1) << 34) - 1);  // below 2^50: LIPP holds such keys where long double is double, too
      for (size_t s = 0; s < per / 20; ++s) {
        const uint64_t sb = b + (g() & ((uint64_t(1) << 33) - 1));
        const unsigned spread = 4 + unsigned(g() % 14);
        for (int i = 0; i < 20; ++i) v.push_back(sb + (g() & ((uint64_t(1) << spread) - 1)));
      }
    }
    return v;
  };
  // a staircase: runs whose key density doubles from one to the next, with a few holes
  auto stairs = [&](size_t m) {
    std::vector<uint64_t> v;
    uint64_t k = uint64_t(1) << 30;
    for (size_t i = 0; i < m; ++i) {
      const unsigned step = unsigned((i / 97) % 12);
      k += (uint64_t(1) << step) + (i % 53 == 0 ? (uint64_t(3) << (step + 6)) : 0);
      v.push_back(k);
    }
    return v;
  };
  auto lognormal = [&](size_t m, double sigma) {
    std::vector<uint64_t> v;
    std::lognormal_distribution<double> d(0.0, sigma);
    while (v.size() < m) {
      const double x = d(g) * 1e9;
      if (x < 1.8e19) v.push_back(uint64_t(x));
    }
    return v;
  };

  struct Case {
    const char* set;
    std::vector<uint64_t> keys;
    Expect ex;
  };
  auto sets = [&]() {
    std::vector<Case> c;
    Expect plain;
    Expect ro;
    ro.writes = false;
    Expect nofar = plain;
    nofar.far_puts = 0;
    c.push_back({"n0", {}, ro});
    c.push_back({"n1", {5}, plain});
    c.push_back({"n1max", {M}, [&] { Expect e = plain; e.side_max = 1; return e; }()});  // csv_alex: its one key is ALEX's sentinel
    c.push_back({"n1zero", {0}, plain});
    c.push_back({"n2", {0, M}, [&] { Expect e = plain; e.side_max = 0.5; return e; }()});
    c.push_back({"n3", {1, 2, M}, [&] { Expect e = plain; e.side_max = 0.34; return e; }()});
    c.push_back({"n5", {0, 1, 2, 3, M}, [&] { Expect e = plain; e.side_max = 0.2; return e; }()});
    c.push_back({"n17", rnd(0, uint64_t(1) << 40, 17), plain});
    c.push_back({"uniform", rnd(1, uint64_t(1) << 50, N), plain});
    { std::vector<uint64_t> v = rnd(0, M, N); for (uint64_t k : {uint64_t(0), M, H, H - 1}) v.push_back(k); Expect e = plain; e.side_max = 1e-3; c.push_back({"wide", v, e}); }
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < N; ++i) v.push_back(1000 + i); c.push_back({"run", v, nofar}); }
    { std::vector<uint64_t> v; for (int k = 0; k < 200; ++k) { const uint64_t b = g() >> 16; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } c.push_back({"clusters", v, plain}); }
    { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), N); for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003); v.push_back(3), v.push_back(M); Expect e = plain; e.side_max = 1e-3; c.push_back({"outliers", v, e}); }
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < N; ++i) v.push_back(i * 65536 + 12345); c.push_back({"stride", v, plain}); }
    { std::vector<uint64_t> v; for (uint64_t i = 0; i < N / 2; ++i) v.push_back(i), v.push_back(M - i); Expect e = plain; e.side_max = 0.51; c.push_back({"hugegap", v, e}); }  // a run down from 2^64 - 1
    c.push_back({"nested", nested(2000), plain});
    c.push_back({"stairs", stairs(60000), plain});
    c.push_back({"lognormal", lognormal(60000, 2.0), plain});
    return c;
  };

  if (want("alex")) {
    if (one_set.empty()) test_alex_root_split();
    for (const char* name : {"csv_alex", "csv_alex40", "csv_alex0"}) {
      for (Case& c : sets()) {
        if (!one_set.empty() && one_set != c.set) continue;
        if (std::string(name) != "csv_alex" && std::string(c.set) != "nested" && std::string(c.set) != "stairs" && std::string(c.set) != "lognormal" && std::string(c.set) != "wide") continue;
        Handle h = make(name);
        Expect e = c.ex;
        if (std::string(c.set) == "lognormal" && std::string(name) != "csv_alex0") e.accepts = true;
        if (h.idx()) check(c.set, c.keys, h, name, e);
        destroy(h);
      }
    }
  }
  if (want("alex2m")) {  // ALEX has model nodes only from about 100,000 keys on: the pass needs sets of this size to act
    // Algorithm 1 costs alpha * m^2 on a subtree of m keys, and ALEX's subtrees here reach 130,000 keys (15 s each): the cap
    // keeps this test short, and checks that a subtree above it is left alone and counted
    setenv("CSV_MAX_SUBTREE", "30000", 1);
    Handle h = make("csv_alex");
    check("uniform2m", rnd(1, uint64_t(1) << 62, 2000000), h, "csv_alex", Expect());
    destroy(h);
    h = make("csv_alex");
    check("lognormal2m", lognormal(2000000, 2.0), h, "csv_alex", Expect());
    if (h.pass().too_big == 0) std::printf("FAIL csv_alex on lognormal2m: no subtree above CSV_MAX_SUBTREE was counted\n"), ++g_bad;
    destroy(h);
    for (const char* name : {"csv_alex", "csv_alex40"}) {
      h = make(name);
      check("stairs2m", stairs(2000000), h, name, Expect());
      destroy(h);
    }
    unsetenv("CSV_MAX_SUBTREE");
  }
  if (want("lipp") || (!only.empty() && want("lipp2m"))) {
    for (const char* name : {"csv_lipp", "csv_lipp40"}) {
      Handle h = make(name);  // one LIPP object for all the sets: its constructor is the expensive part
      if (!h.idx()) continue;
      for (Case& c : sets()) {
        if (!want("lipp")) break;
        const std::string s = c.set;
        if (!one_set.empty() && one_set != s) continue;
        if (std::string(name) != "csv_lipp" && s != "nested" && s != "stairs" && s != "lognormal" && s != "wide" && s != "n5") continue;
        Expect e = c.ex;
        if (s == "n1max" || s == "n2" || s == "n3" || s == "n5") e.side_max = 0;  // LIPP holds 2^64 - 1
        if (s == "hugegap") e.side_min = 0.24;  // the run down from 2^64 - 1: every second key is too close for LIPP
        if (s == "nested" || s == "lognormal") e.accepts = true;
        check(c.set, c.keys, h, name, e);
      }
      if (one_set.empty() && want("lipp2m")) {
        if (std::string(name) == "csv_lipp") check("uniform2m", rnd(1, uint64_t(1) << 62, 2000000), h, name, Expect());
        Expect e;
        e.accepts = true;
        check("lognormal2m", lognormal(2000000, 2.0), h, name, e);
      }
      destroy(h);
    }
  }
  if (want("modes") && one_set.empty()) {  // the settings that are not the default, on sets where the pass acts
    auto run = [&](Handle& h, const char* name, const char* var, const char* value, const char* label, std::vector<uint64_t> keys, bool accepts) {
      setenv(var, value, 1);
      Expect e;
      e.accepts = accepts;
      check(std::string(label) + ":" + value, std::move(keys), h, name, e);
      unsetenv(var);
    };
    Handle l = make("csv_lipp"), a = make("csv_alex");
    if (l.idx() && a.idx()) {
      // the integer part of alpha * |K| (the paper's Definition 1): subtrees of fewer than ten keys get no point
      run(l, "csv_lipp", "CSV_LAMBDA", "floor", "lognormal", lognormal(60000, 2.0), true);
      const size_t floor_virtual = l.n_virtual();
      run(l, "csv_lipp", "CSV_LAMBDA", "ceil", "lognormal", lognormal(60000, 2.0), true);
      if (l.n_virtual() == 0 || floor_virtual == 0) std::printf("FAIL modes: CSV_LAMBDA floor / ceil gave %zu / %zu virtual keys\n", floor_virtual, l.n_virtual()), ++g_bad;
      run(a, "csv_alex", "CSV_LAMBDA", "floor", "lognormal", lognormal(60000, 2.0), true);
      // the paper's cost condition for LIPP read to the letter: every smoothed subtree is rebuilt
      run(l, "csv_lipp", "CSV_LIPP_RULE", "loss", "lognormal", lognormal(60000, 2.0), true);
      if (l.pass().accepted != l.pass().smoothed) std::printf("FAIL modes: CSV_LIPP_RULE=loss kept %zu of %zu smoothed subtrees\n", l.pass().accepted, l.pass().smoothed), ++g_bad;
      run(l, "csv_lipp", "CSV_LIPP_RULE", "loss", "nested", nested(2000), true);
      // a budget the pass cannot fit: the largest subtrees are left alone and counted, the rest is still exact
      run(l, "csv_lipp", "CSV_BUDGET_S", "1e-6", "nested", nested(2000), false);
      if (l.pass().over_budget + l.pass().late == 0) std::printf("FAIL modes: csv_lipp under CSV_BUDGET_S=1e-6 left no subtree out\n"), ++g_bad;
      run(a, "csv_alex", "CSV_BUDGET_S", "1e-6", "stairs", stairs(60000), false);
      if (a.pass().over_budget + a.pass().late == 0) std::printf("FAIL modes: csv_alex under CSV_BUDGET_S=1e-6 left no subtree out\n"), ++g_bad;
      run(a, "csv_alex", "CSV_BUDGET_S", "0", "lognormal", lognormal(60000, 2.0), false);
      if (a.pass().accepted || a.n_virtual()) std::printf("FAIL modes: csv_alex under CSV_BUDGET_S=0 still smoothed\n"), ++g_bad;
      // a stricter cost threshold for ALEX keeps no more subtrees than c = 0
      run(a, "csv_alex", "CSV_COST_C", "0", "lognormal", lognormal(60000, 2.0), true);
      const size_t kept0 = a.pass().accepted;
      run(a, "csv_alex", "CSV_COST_C", "-8", "lognormal", lognormal(60000, 2.0), false);
      if (a.pass().accepted > kept0) std::printf("FAIL modes: CSV_COST_C=-8 kept %zu subtrees, c = 0 kept %zu\n", a.pass().accepted, kept0), ++g_bad;
      // the paper's two cost constants: with the search constant at 0 only the level counts, and every subtree below the
      // root becomes one data node (the widest merges the wrapper can make: exactness under them)
      run(a, "csv_alex", "CSV_SEARCH_W", "0", "stairs", stairs(60000), true);
      if (a.pass().accepted != a.pass().candidates || a.pass().accepted == 0) std::printf("FAIL modes: CSV_SEARCH_W=0 merged %zu of %zu subtrees\n", a.pass().accepted, a.pass().candidates), ++g_bad;
      run(a, "csv_alex", "CSV_SEARCH_W", "0", "nested", nested(2000), false);
      // malformed settings are refused, loudly
      for (const auto& bad : {std::pair<const char*, const char*>{"CSV_LAMBDA", "round"}, {"CSV_LIPP_RULE", "cost"}, {"CSV_BUDGET_S", "-1"}, {"CSV_MAX_SUBTREE", "x"}, {"CSV_COST_C", "5"}, {"CSV_SEARCH_W", "-1"}, {"CSV_LEVEL_W", "many"}}) {
        setenv(bad.first, bad.second, 1);
        std::vector<KV> kv;
        for (uint64_t k : lognormal(60000, 2.0)) kv.push_back({k, 1});
        std::sort(kv.begin(), kv.end());
        kv.erase(std::unique(kv.begin(), kv.end()), kv.end());
        for (Index* i : {l.idx(), a.idx()}) {
          if (!std::strcmp(bad.first, "CSV_LIPP_RULE") && i == a.idx()) continue;
          if ((!std::strcmp(bad.first, "CSV_COST_C") || !std::strcmp(bad.first, "CSV_SEARCH_W") || !std::strcmp(bad.first, "CSV_LEVEL_W")) && i == l.idx()) continue;
          bool threw = false;
          try { i->bulk_load(kv.data(), kv.size()); } catch (const std::invalid_argument&) { threw = true; }
          if (!threw) std::printf("FAIL modes: %s=%s was accepted\n", bad.first, bad.second), ++g_bad;
        }
        unsetenv(bad.first);
      }
    }
    destroy(l);
    destroy(a);
  }
  if (timing) {  // not a benchmark: to see the wrapper costs nothing odd per lookup, and what the pass does on two small sets
    for (int which = 0; which < 3; ++which) {
      std::vector<uint64_t> keys = which == 0 ? rnd(1, uint64_t(1) << 62, 3000000) : which == 1 ? lognormal(3000000, 2.0) : nested(150000);
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      std::vector<KV> kv;
      for (uint64_t k : keys) kv.push_back({k, 123456789});
      std::printf("-- %s\n", which == 0 ? "uniform" : which == 1 ? "lognormal" : "nested");
      time_one("lipp", new LIPPInterface<uint64_t, uint64_t>, kv);
      for (const char* name : {"csv_lipp0", "csv_lipp", "csv_lipp40"}) time_one(name, gret::make_csv<uint64_t, uint64_t>(name), kv);
      time_one("alex", new alexInterface<uint64_t, uint64_t>, kv);
      for (const char* name : {"csv_alex0", "csv_alex", "csv_alex40"}) time_one(name, gret::make_csv<uint64_t, uint64_t>(name), kv);
    }
  }
  std::printf(g_bad ? "test_csv: %zu FAILURES\n" : "test_csv: all passed (%zu rebuilt subtrees kept, %zu virtual keys)\n", g_bad ? g_bad : g_accepted, g_virtual);
  g_done = true;
  return g_bad ? 1 : 0;
}
