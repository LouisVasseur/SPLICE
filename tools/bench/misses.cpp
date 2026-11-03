// SPDX-License-Identifier: MIT
// Lookups that miss, and lookups that lean on a few keys: mean ns per lookup at 0, 10, 50 and 100% misses (keys between two stored keys) and
// on Zipf-distributed hits (s = 0.99, ranks shuffled so the popular keys are not neighbours). The same keys for every index on a dataset;
// answers are counted against what they must be.
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
  if (k.back() == ~u64(0) && (name == "alex" || name == "pgm")) { k.pop_back(); --n; }  // their sentinel, as in GRE's own runs on fb
  std::vector<std::pair<u64, u64>> kv(n);
  for (u64 i = 0; i < n; ++i) kv[i] = {k[i], k[i] * 2654435761u + 17};
  auto* idx = gret::make_index<u64, u64>(name);
  if (!idx) return 3;
  Param p(1, 0);
  idx->init(&p);
  idx->bulk_load(kv.data(), n, &p);
  std::vector<std::pair<u64, u64>>().swap(kv);
  const size_t S = size_t(1) << 24;
  std::mt19937_64 g(21);
  std::vector<u64> hit(S), miss(S), zipf(S);
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
  }
  const size_t Q = 30000000;
  auto bench = [&](const char* label, const std::vector<u64>* a, const std::vector<u64>* b, unsigned pct_b, size_t want_hits_num) {
    std::vector<uint8_t> which(1 << 20);
    std::mt19937 h(5);
    for (auto& w : which) w = (h() % 100) < pct_b;
    size_t hits = 0, ia = 0, ib = 0;
    u64 val = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < Q; ++i) {
      const u64 x = which[i & ((1 << 20) - 1)] ? (*b)[ib++ & (S - 1)] : (*a)[ia++ & (S - 1)];
      hits += idx->get(x, val, &p);
    }
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(Q);
    std::printf("%s\t%s\t%s\tns=%.1f\thits=%zu\texpected_hits=%zu\tMops=%.2f\n", v[3], v[2], label, ns, hits, want_hits_num == SIZE_MAX ? ia : want_hits_num, 1e3 / ns);
    std::fflush(stdout);
  };
  bench("miss0", &hit, &miss, 0, SIZE_MAX);
  bench("miss10", &hit, &miss, 10, SIZE_MAX);
  bench("miss50", &hit, &miss, 50, SIZE_MAX);
  bench("miss100", &hit, &miss, 100, 0);
  bench("zipf099", &zipf, &miss, 0, SIZE_MAX);
  std::printf("%s\t%s\tgaps_found=%zu of %zu\tBkey=%.2f\n", v[3], v[2], got_miss, S, double(idx->memory_consumption()) / double(n));
  return 0;
}
