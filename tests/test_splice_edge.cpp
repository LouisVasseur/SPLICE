// SPDX-License-Identifier: MIT
// SpliceIndex exactness on adversarial key sets, generated here (no data files): every key returns its own payload, and
// every key - 1, key + 1, 0, 2^63 - 1, 2^63 and 2^64 - 1 that is not a key returns false, for every name of the
// attribution ladder. Sets: n = 1, 2, 3, 5, 17; keys at 0 and at 2^64 - 1; a huge gap in the middle; dense runs of
// consecutive integers next to sparse keys; a perfectly linear set; a set straddling 2^63; one line of keys; far
// outliers at both ends; 200,000 random keys; and a run whose last key forms PGM's last segment alone (far, mild).
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

// also bounds the index's size (at most 64 B/key), and on the lone-last-key sets its line count and size
static size_t check(const std::vector<uint64_t>& keys, const char* name, bool tight = false) {
  std::vector<std::pair<uint64_t, uint64_t>> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  auto* idx = gret::make_index<uint64_t, uint64_t>(name);
  if (!idx) return 1;
  idx->bulk_load(kv.data(), kv.size());
  std::vector<uint64_t> q = {0, (1ull << 63) - 1, 1ull << 63, ~0ull};
  for (uint64_t x : keys) q.insert(q.end(), {x, x - 1, x + 1});
  size_t err = 0;
  for (uint64_t x : q) {
    uint64_t v = 0;
    const bool got = idx->get(x, v), want = std::binary_search(keys.begin(), keys.end(), x);
    err += got != want || (got && v != payload(x));
  }
  if (keys.size() > 1000 && idx->memory_consumption() > (long long)(64 * keys.size())) ++err;
  if (tight) {  // a run plus a lone last key: about one line per lookup and close to 16 B/key (G-off names included)
    auto* sp = dynamic_cast<gret::SpliceIndex<uint64_t, uint64_t>*>(idx);
    if (sp && (sp->expected_lines() > 1.5 || idx->memory_consumption() > (long long)(20 * keys.size()))) ++err;
  }
  delete idx;
  return err;
}

int main() {
  const uint64_t M = ~0ull;
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) {
    std::uniform_int_distribution<uint64_t> u(lo, hi);
    std::vector<uint64_t> v(m);
    for (auto& x : v) x = u(g);
    return v;
  };
  std::vector<std::pair<std::string, std::vector<uint64_t>>> sets = {
      {"n1", {5}}, {"n1max", {M}}, {"n1zero", {0}}, {"n2", {0, M}}, {"n3", {1, 2, M}}, {"n5", {0, 1, 2, 3, M}},
      {"n17", rnd(0, 1ull << 40, 17)}, {"oneline", {100, 101, 102, 103}}};
  {
    std::vector<uint64_t> v = {0, 1, 2, M - 2, M - 1, M}, r = rnd(1000000, 1100000, 994);
    v.insert(v.end(), r.begin(), r.end());
    sets.push_back({"n1000_far", v});
  }
  {
    std::vector<uint64_t> v = rnd(1000000000000ull, 1000000000000000ull, 5000);
    for (uint64_t i = 1000; i < 6000; ++i) v.push_back(i);
    sets.push_back({"dense_sparse", v});
  }
  {
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < 100000; ++i) v.push_back(7 * i + 3);
    sets.push_back({"linear", v});
  }
  sets.push_back({"straddle", rnd((1ull << 63) - 1000000000, (1ull << 63) + 1000000000, 50000)});
  {
    std::vector<uint64_t> v;
    for (uint64_t i = 0; i < 50000; ++i) v.push_back(i), v.push_back(M - i);
    sets.push_back({"hugegap", v});
  }
  {
    std::vector<uint64_t> v = rnd(1000000000, 2000000000, 99996);
    v.insert(v.end(), {0, 1, M - 1, M});
    sets.push_back({"outliers", v});
  }
  sets.push_back({"random", rnd(0, M, 200000)});
  {  // PGM's last segment is the last key alone (review, 2026-10-20): it must keep its own interval
    std::vector<uint64_t> v, w;
    for (uint64_t i = 0; i < 33334; ++i) v.push_back(3 * i), w.push_back(3 * i);
    v.push_back(M);
    w.push_back(190000);
    sets.push_back({"lastalone_far", v});
    sets.push_back({"lastalone_mild", w});
  }
  const char* names[] = {"hl4", "hl8", "hf4", "rl4", "spgm64", "ab4", "ph4", "gr4", "splice", "spl_t", "spl_gt", "spl_tv", "splice_eq", "spl_fixk", "splice_gi", "splice2", "splice2_gi", "s2_t", "s2_gt", "s2_tv", "s2_eq", "s2_fixk", "splice3", "s3_t", "s3_gt", "s3_tv", "s3_eq", "s3_fixk"};
  size_t bad = 0, cases = 0;
  for (auto& [label, keys] : sets) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    for (const char* nm : names) {
      const size_t e = check(keys, nm, label.compare(0, 9, "lastalone") == 0 && std::string(nm).find("eq") == std::string::npos && std::string(nm).compare(0, 2, "hl") != 0 && std::string(nm).compare(0, 2, "rl") != 0);
      ++cases;
      if (e) { ++bad; std::printf("FAIL %s %s: %zu wrong answers or bound violations\n", label.c_str(), nm, e); }
    }
  }
  std::printf("test_splice_edge: %s (%zu cases, %zu failed)\n", bad ? "FAILED" : "ok", cases, bad);
  return bad ? 1 : 0;
}
