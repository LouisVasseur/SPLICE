// SPDX-License-Identifier: MIT
// update and remove on ab4, against a std::map: after every batch of writes, get on every key ever stored, on its neighbours and on
// random keys, and scans from stored, removed, never-stored and extreme start keys, return exactly what the map holds. Writes on
// keys that are absent or already removed return false and change nothing. Each set then loses all its keys in one of three orders
// (random, ascending, descending: the first and the last key of the slots go at different times), and must answer as an empty
// index does. Key sets as in test_scan (single leaves, deep directories, spills, empty cells, peeled exceptions, keys 0 and
// 2^64 - 1), with AB_V in {0, 1} x AB_BUDGET in {0, 40} and three leaf sizes. put still returns false.
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

using Index = indexInterface<uint64_t, uint64_t>;
using Map = std::map<uint64_t, uint64_t>;

// every answer the index gives, against the map. all: every key the set ever held, so removed ones are probed too
static size_t verify(Index& idx, const Map& ref, const std::vector<uint64_t>& all, std::mt19937_64& g, const std::string& what) {
  size_t bad = 0;
  auto fail = [&](const char* op, uint64_t k) {
    if (bad++ < 5) std::fprintf(stderr, "  %s: %s wrong at key %llu\n", what.c_str(), op, (unsigned long long)k);
  };
  auto probe = [&](uint64_t k) {
    uint64_t v = 0;
    const bool got = idx.get(k, v);
    const auto it = ref.find(k);
    if (got != (it != ref.end()) || (got && v != it->second)) fail("get", k);
  };
  for (uint64_t k : all) probe(k), probe(k + 1), probe(k - 1);
  probe(0), probe(~uint64_t(0));
  for (int i = 0; i < 2000; ++i) probe(g());
  std::vector<uint64_t> starts = {0, 1, ~uint64_t(0), ~uint64_t(0) - 1};
  for (int i = 0; i < 60 && !all.empty(); ++i) {
    const uint64_t k = all[g() % all.size()];
    starts.push_back(k), starts.push_back(k + 1), starts.push_back(k - 1);
  }
  for (int i = 0; i < 20; ++i) starts.push_back(g());
  std::vector<std::pair<uint64_t, uint64_t>> out(600);
  for (uint64_t s : starts)
    for (size_t n : {size_t(1), size_t(7), size_t(600)}) {
      const size_t got = idx.scan(s, n, out.data());
      auto it = ref.lower_bound(s);
      size_t want = 0;
      bool same = true;
      for (; want < n && it != ref.end(); ++want, ++it)
        if (want < got && (out[want].first != it->first || out[want].second != it->second)) same = false;
      if (got != want || !same) fail("scan", s);
    }
  return bad;
}

static size_t run(const std::vector<uint64_t>& keys, const std::string& label, int order, std::mt19937_64& g) {
  std::vector<std::pair<uint64_t, uint64_t>> kv;
  for (uint64_t k : keys) kv.emplace_back(k, payload(k));
  std::unique_ptr<Index> idx(gret::make_index<uint64_t, uint64_t>("ab4"));
  idx->init();
  idx->bulk_load(kv.data(), kv.size());
  Map ref(kv.begin(), kv.end());
  size_t bad = verify(*idx, ref, keys, g, label + " after load");
  if (idx->put(123, 456)) ++bad, std::fprintf(stderr, "  %s: put returned true\n", label.c_str());
  auto write = [&](uint64_t k, bool rm) {
    const auto it = ref.find(k);
    const bool there = it != ref.end(), ok = rm ? idx->remove(k) : idx->update(k, k ^ g());
    if (ok != there) {
      if (bad++ < 5) std::fprintf(stderr, "  %s: %s of key %llu returned %d, stored %d\n", label.c_str(), rm ? "remove" : "update", (unsigned long long)k, int(ok), int(there));
      return;
    }
    if (!there) return;
    if (rm) ref.erase(it);
    else {  // the payload the index was given: read it back rather than draw it twice
      uint64_t v = 0;
      if (!idx->get(k, v)) ++bad;
      it->second = v;
    }
  };
  // rounds of mixed writes: stored keys, their neighbours (mostly absent), random keys, keys already removed
  for (int round = 0; round < 4; ++round) {
    const size_t ops = keys.size() / 3 + 8;
    for (size_t i = 0; i < ops; ++i) {
      const uint64_t k = keys[g() % keys.size()];
      const unsigned r = unsigned(g() % 10);
      if (r < 4) write(k, false);
      else if (r < 7) write(k, true);
      else if (r == 7) write(k + 1, g() & 1);
      else if (r == 8) write(g(), g() & 1);
      else write(k, false), write(k, false);
    }
    bad += verify(*idx, ref, keys, g, label + " round " + std::to_string(round));
  }
  // then every key that is left, in this order
  std::vector<uint64_t> rest;
  for (const auto& p : ref) rest.push_back(p.first);
  if (order == 0) std::shuffle(rest.begin(), rest.end(), g);
  if (order == 2) std::reverse(rest.begin(), rest.end());
  for (size_t i = 0; i < rest.size(); ++i) {
    write(rest[i], true);
    if (i + 1 == rest.size() / 2 || i + 2 == rest.size()) bad += verify(*idx, ref, keys, g, label + " emptying");
  }
  bad += verify(*idx, ref, keys, g, label + " empty");
  for (size_t i = 0; i < std::min<size_t>(rest.size(), 50); ++i) write(rest[i], g() & 1);  // nothing is left to write on
  bad += verify(*idx, ref, keys, g, label + " empty, after writes");
  return bad;
}

int main() {
  const uint64_t M = ~uint64_t(0);
  unsetenv("RULE_FILE"), unsetenv("RULE_S"), unsetenv("GATE_DISCOVER"), unsetenv("TRANSFORM_G_E");
  std::mt19937_64 g(11);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  std::vector<std::pair<std::string, std::vector<uint64_t>>> sets = {
      {"n1", {5}}, {"n1max", {M}}, {"n1zero", {0}}, {"n2", {0, M}}, {"n3", {1, 2, M}}, {"n5", {0, 1, 2, 3, M}}, {"n17", rnd(0, uint64_t(1) << 40, 17)},
      {"uniform", rnd(1, uint64_t(1) << 50, 6000)}, {"wide", rnd(0, M, 6000)}};
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 8000; ++i) v.push_back(1000 + i); sets.push_back({"run", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), 6000); for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003); v.push_back(3); sets.push_back({"outliers", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + 100000, 8000); for (uint64_t j = 0; j < 100; ++j) v.push_back(M - j * 1000); sets.push_back({"over64outliers", v}); }
  { std::vector<uint64_t> v; for (int c = 0; c < 80; ++c) { const uint64_t b = g() >> 1; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } sets.push_back({"clusters", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 6000; ++i) v.push_back(i * 65536 + 12345); sets.push_back({"stride", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 3000; ++i) { v.push_back(i); v.push_back(M - i); } sets.push_back({"hugegap", v}); }
  const char* leaves[] = {"4", "64", "1024"};
  size_t bad = 0, cases = 0;
  for (auto& [label, keys] : sets) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    for (const char* lf : leaves)
      for (int v = 0; v < 4; ++v)
        for (int order = 0; order < 3; ++order) {
          setenv("AR_LEAF", lf, 1);
          setenv("AR_TAU", "1.01", 1);
          setenv("AB_V", v & 1 ? "1" : "0", 1);
          setenv("AB_BUDGET", v & 2 ? "40" : "0", 1);
          const std::string what = label + " leaf " + lf + " v" + std::to_string(v) + " order " + std::to_string(order);
          const size_t e = run(keys, what, order, g);
          if (e) std::fprintf(stderr, "FAIL %s: %zu wrong answers\n", what.c_str(), e);
          ++cases;
          bad += e != 0;
        }
  }
  std::printf("test_write: %s (%zu cases, %zu failed)\n", bad ? "FAILED" : "ok", cases, bad);
  return bad ? 1 : 0;
}
