// SPDX-License-Identifier: MIT
// Lookups that miss, and lookups that lean on a few keys: mean ns per lookup at 0, 10, 50 and 100% misses (keys between two stored keys) and
// on skewed hits (Zipf s = 0.99 by the continuous approximation, so the head is a little lighter than discrete Zipf; ranks shuffled so the popular
// keys are not neighbours). The same keys for every index on a dataset; answers are counted against what they must be.
//   misses KEYS NAME LABEL
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
using u64 = uint64_t;

// LIPP's at() asserts that the slot it lands on holds the key, and its assert calls exit(0), so GRE's get cannot look up a key that is not
// stored. exists() walks the same path without the assert. The value is not read here for any index.
class LippExists final : public indexInterface<u64, u64> {
  LIPP<u64, u64> t;
 public:
  void init(Param* = nullptr) override {}
  void bulk_load(std::pair<u64, u64>* kv, size_t n, Param* = nullptr) override { t.bulk_load(kv, int(n)); }
  bool get(u64 k, u64&, Param* = nullptr) override { return t.exists(k); }
  bool put(u64, u64, Param* = nullptr) override { return false; }
  bool update(u64, u64, Param* = nullptr) override { return false; }
  bool remove(u64, Param* = nullptr) override { return false; }
  size_t scan(u64, size_t, std::pair<u64, u64>*, Param* = nullptr) override { return 0; }
  long long memory_consumption() override { return t.total_size(); }
};

int main(int, char** v) {
  FILE* f = std::fopen(v[1], "rb");
  u64 n = 0;
  if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
  std::vector<u64> k(n);
  if (std::fread(k.data(), 8, n, f) != n) return 2;
  std::fclose(f);
  if (!std::is_sorted(k.begin(), k.end())) std::sort(k.begin(), k.end());
  k.erase(std::unique(k.begin(), k.end()), k.end());
  n = k.size();
  const std::string name = v[2];
  const bool sentinel = k.back() == ~u64(0) && (name == "alex" || name == "pgm");  // their sentinel, as in GRE's own runs on fb
  const u64 stored = sentinel ? n - 1 : n;
  std::vector<std::pair<u64, u64>> kv(stored);
  for (u64 i = 0; i < stored; ++i) kv[i] = {k[i], k[i] * 2654435761u + 17};
  indexInterface<u64, u64>* idx = name == "lipp" ? static_cast<indexInterface<u64, u64>*>(new LippExists) : gret::make_index<u64, u64>(name);
  if (!idx) return 3;
  Param p(1, 0);
  idx->init(&p);
  idx->bulk_load(kv.data(), stored, &p);
  std::vector<std::pair<u64, u64>>().swap(kv);
  const size_t S = size_t(1) << 24;
  std::mt19937_64 g(21);
  std::vector<u64> hit(S), miss(S), zipf(S), zipf_near(S), hot(S);
  for (auto& x : hit) x = k[g() % n];
  size_t got_miss = 0;
  for (auto& x : miss) {
    x = 0;
    for (int t = 0; t < 256 && !x; ++t) {
      const size_t r = g() % (n - 1);
      if (k[r + 1] - k[r] >= 2) x = k[r] + 1 + g() % (k[r + 1] - k[r] - 1);
    }
    if (x) ++got_miss;
    else x = k[n - 1] + 1;  // no gaps at all: above the last key
  }
  {  // Zipf ranks by inversion of the continuous approximation, each rank mapped to a key by a fixed multiplicative shuffle
    const double s = 0.99, a = 1 - s, H = (std::pow(double(n), a) - 1) / a;
    std::uniform_real_distribution<double> U(0, 1);
    for (auto& x : zipf) {
      const u64 rank = std::min<u64>(n - 1, u64(std::pow(U(g) * H * a + 1, 1 / a)) - 1);
      x = k[(rank * 0x9E3779B97F4A7C15ull) % n];
    }
    // the same skew with no shuffle, so popular keys are neighbours; and uniform lookups inside one random 1% of the keys
    for (auto& x : zipf_near) x = k[std::min<u64>(n - 1, u64(std::pow(U(g) * H * a + 1, 1 / a)) - 1)];
    const u64 width = std::max<u64>(1, n / 100), from = g() % (n - width);
    for (auto& x : hot) x = k[from + g() % width];
  }
  if (sentinel) {  // the key the index cannot hold is looked up as the one before it, in every list, so all indexes draw the same sample
    for (auto* l : {&hit, &zipf, &zipf_near, &hot})
      for (auto& x : *l)
        if (x == ~u64(0)) x = k[n - 2];
  }
  const size_t Q = 30000000;
  auto bench = [&](const char* label, const std::vector<u64>* a, const std::vector<u64>* b, unsigned pct_b, size_t want_hits_num) {
    std::vector<uint8_t> which(1 << 20);
    std::mt19937 h(5);
    for (auto& w : which) w = (h() % 100) < pct_b;
    size_t hits = 0;
    size_t at[2] = {0, 0};
    const u64* const lists[2] = {a->data(), b->data()};
    u64 val = 0;
    for (size_t i = 0; i < 2000000; ++i) {  // warm-up, not counted
      const unsigned w = which[i & ((1 << 20) - 1)];
      hits += idx->get(lists[w][at[w]++ & (S - 1)], val, &p);
    }
    hits = 0;
    at[0] = at[1] = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < Q; ++i) {
      const unsigned w = which[i & ((1 << 20) - 1)];  // no branch of its own, so the mix does not charge the loop's mispredictions to the index
      hits += idx->get(lists[w][at[w]++ & (S - 1)], val, &p);
    }
    const size_t ia = at[0];
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(Q);
    std::printf("%s\t%s\t%s\tns=%.1f\thits=%zu\texpected_hits=%zu\tMops=%.2f\n", v[3], v[2], label, ns, hits, want_hits_num == SIZE_MAX ? ia : want_hits_num, 1e3 / ns);
    std::fflush(stdout);
  };
  bench("miss0", &hit, &miss, 0, SIZE_MAX);
  bench("miss10", &hit, &miss, 10, SIZE_MAX);
  bench("miss50", &hit, &miss, 50, SIZE_MAX);
  bench("miss100", &hit, &miss, 100, 0);
  bench("zipf099", &zipf, &miss, 0, SIZE_MAX);
  bench("zipf099_near", &zipf_near, &miss, 0, SIZE_MAX);
  bench("hot1pct", &hot, &miss, 0, SIZE_MAX);
  std::printf("%s\t%s\tgaps_found=%zu of %zu\tBkey=%.2f\n", v[3], v[2], got_miss, S, double(idx->memory_consumption()) / double(stored));
  return 0;
}
