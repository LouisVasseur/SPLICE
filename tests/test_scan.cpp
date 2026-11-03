// SPDX-License-Identifier: MIT
// scan on the order-keeping indexes, against a sorted array: every start key (stored, next to a stored key, between keys, below and
// above everything, either side of the peeled exceptions, around key 0 and 2^64 - 1) and length returns exactly the keys >= the start
// with their own payloads, ascending, on key sets that force single leaves, deep directories, spills and empty cells.
//   test_scan [KEYS...]   (GRE key files, optional: sampled starts on each as well)
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
#include <utility>
#include <vector>

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

static size_t check(const std::vector<uint64_t>& keys, const std::string& label, const char* name, size_t max_starts) {
  std::vector<std::pair<uint64_t, uint64_t>> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  auto* idx = gret::make_index<uint64_t, uint64_t>(name);
  if (!idx) { std::printf("%s: not registered\n", name); return 1; }
  Param p(1, 0);
  idx->init(&p);
  idx->bulk_load(kv.data(), kv.size(), &p);
  const uint64_t M = ~uint64_t(0);
  std::vector<uint64_t> starts = {0, 1, M, M - 1, uint64_t(1) << 63};
  std::mt19937_64 g(11);
  auto around = [&](uint64_t k) { starts.push_back(k); if (k) starts.push_back(k - 1); if (k != M) starts.push_back(k + 1); };
  // both ends (where the exceptions are), the middle, and random stored keys and gaps
  for (size_t i = 0; i < std::min<size_t>(80, keys.size()); ++i) { around(keys[i]); around(keys[keys.size() - 1 - i]); }
  for (size_t i = 0; i < max_starts && !keys.empty(); ++i) {
    const size_t r = g() % keys.size();
    around(keys[r]);
    if (r + 1 < keys.size() && keys[r + 1] - keys[r] > 2) starts.push_back(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));
  }
  {  // point lookups too: the ends, where the peeled keys are, and a sample through the middle
    const size_t step = std::max<size_t>(1, keys.size() / 20000);
    size_t lost = 0;
    uint64_t v = 0;
    for (size_t i = 0; i < keys.size(); ++i)
      if (i < 300 || i + 300 >= keys.size() || i % step == 0) lost += !(idx->get(keys[i], v, &p) && v == payload(keys[i]));
    if (lost) std::printf("FAIL %s on %s: %zu stored keys not found\n", name, label.c_str(), lost);
    if (lost) { delete idx; return lost; }
  }
  const size_t lens[] = {1, 2, 3, 16, 256, 4096, 65536, keys.size() + 5};
  std::vector<std::pair<uint64_t, uint64_t>> out;
  size_t bad = 0;
  for (uint64_t x : starts)
    for (size_t len : lens) {
      if (len > 4096 && (g() & 63)) continue;  // long scans on a sample of the starts
      if (len > 65536 && keys.size() > 1000000) continue;
      out.assign(len, {0, 0});
      const size_t got = idx->scan(x, len, out.data(), &p);
      const size_t at = size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin());
      const size_t want = std::min(len, keys.size() - at);
      bool ok = got == want;
      for (size_t i = 0; ok && i < got; ++i) ok = out[i].first == keys[at + i] && out[i].second == payload(keys[at + i]);
      if (!ok) {
        if (++bad <= 3) std::printf("FAIL %s on %s: start %llu len %zu: got %zu, want %zu\n", name, label.c_str(), (unsigned long long)x, len, got, want);
      }
    }
  delete idx;
  return bad;
}

int main(int argc, char** argv) {
  const uint64_t M = ~uint64_t(0);
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  std::vector<std::pair<std::string, std::vector<uint64_t>>> sets = {
      {"n1", {5}}, {"n1max", {M}}, {"n1zero", {0}}, {"n2", {0, M}}, {"n3", {1, 2, M}}, {"n5", {0, 1, 2, 3, M}}, {"n17", rnd(0, uint64_t(1) << 40, 17)},
      {"uniform", rnd(1, uint64_t(1) << 50, 20000)}, {"wide", rnd(0, M, 20000)}};
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 50000; ++i) v.push_back(1000 + i); sets.push_back({"run", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), 20000); for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003); v.push_back(3); v.push_back(M); sets.push_back({"outliers", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + 100000, 30000); for (uint64_t j = 0; j < 100; ++j) v.push_back(M - j * 1000); sets.push_back({"over64outliers", v}); }
  { std::vector<uint64_t> v; for (int c = 0; c < 300; ++c) { const uint64_t b = g() >> 1; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } sets.push_back({"clusters", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 20000; ++i) v.push_back(i * 65536 + 12345); sets.push_back({"stride", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 10000; ++i) { v.push_back(i); v.push_back(M - i); } sets.push_back({"hugegap", v}); }
  const char* names[] = {"rl4", "ab4", "ph4", "gr4", "spgm16", "spgm256"};
  const char* leaves[] = {"4", "64", "1024"};
  size_t bad = 0, cases = 0;
  for (auto& [label, keys] : sets) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    for (const char* nm : names)
      for (const char* lf : leaves)
        for (int v = 0; v < (std::string(nm) == "ab4" ? 3 : 1); ++v) {  // ab4 also with its lines fitted, and on a byte budget
          setenv("AR_LEAF", lf, 1);
          setenv("AR_TAU", "1.01", 1);
          setenv("AB_V", v ? "1" : "0", 1);
          setenv("AB_BUDGET", v == 2 ? "40" : "0", 1);
          const size_t e = check(keys, label, nm, 2000);
          ++cases;
          bad += e != 0;
        }
    setenv("AB_V", "0", 1);
    setenv("AB_BUDGET", "0", 1);
  }
  {  // gr4 with a period rule that fits the keys, one that does not, and periods at the edges of what the reciprocal handles
    struct Rule { const char* label; std::string text; uint64_t period, row; };
    const Rule rules[] = {{"fits", "S 67108864\n1000 100000\n", 100000, 10000},
                          {"wrong_period", "S 67108864\n1000 77777\n", 100000, 10000},
                          {"period_1", "0 1\n", 100000, 10000},
                          {"period_2_40", "S 1048576\n0 1099511627776\n", uint64_t(1) << 40, 1000},
                          {"two_blocks", "S 67108864\n1000 100000\n2000000000 77777\n", 100000, 10000},
                          {"wraps", "S 2199023255552\n1000 100000\n", 100000, 10000}};
    const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/test_scan_rule.txt";
    for (const Rule& r : rules) {
      std::vector<uint64_t> keys;
      for (uint64_t q = 0; q < 2000; ++q)
        for (uint64_t j = 0; j < 20; ++j) keys.push_back(1000 + q * r.period + j * j * 3);
      if (std::string(r.label) == "wraps")  // far keys beyond what peeling takes, whose coordinates overflow into the bulk's range
        for (uint64_t j = 0; j < 60; ++j) keys.push_back((uint64_t(1) << 63) + j * 0x0100000000000000ull / 2 + 12345);
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      FILE* o = std::fopen(path.c_str(), "w");
      std::fputs(r.text.c_str(), o);
      std::fclose(o);
      setenv("RULE_FILE", path.c_str(), 1);
      setenv("AR_LEAF", "16", 1);
      setenv("AR_TAU", "1.5", 1);
      bad += check(keys, std::string("rule_") + r.label, "gr4", 1000) != 0;
      ++cases;
      setenv("RULE_FILE", "", 1);
    }
    std::remove(path.c_str());
  }
  for (int a = 1; a < argc; ++a) {
    FILE* f = std::fopen(argv[a], "rb");
    uint64_t n = 0;
    if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
    std::vector<uint64_t> k(n);
    if (std::fread(k.data(), 8, n, f) != n) return 2;
    std::fclose(f);
    std::sort(k.begin(), k.end());
    k.erase(std::unique(k.begin(), k.end()), k.end());
    for (const char* nm : names) {
      setenv("AR_LEAF", std::string(nm) == "ph4" ? "1024" : "16", 1);
      setenv("AR_TAU", "2", 1);
      const size_t e = check(k, argv[a], nm, 3000);
      ++cases;
      bad += e != 0;
    }
  }
  std::printf("test_scan: %s (%zu cases, %zu failed)\n", bad ? "FAILED" : "ok", cases, bad);
  return bad ? 1 : 0;
}
