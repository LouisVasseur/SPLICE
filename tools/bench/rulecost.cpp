// The transform's own cost: ns per coordinate on a list of stored keys in random order, read from a list that stays out of the way.
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
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
  std::vector<std::pair<u64, u64>> kv(n);
  for (u64 i = 0; i < n; ++i) kv[i] = {k[i], k[i]};
  gret::PeriodRule rule;
  rule.load(kv.data(), n);
  std::mt19937_64 g(3);
  std::vector<u64> q(size_t(1) << 22);
  for (auto& x : q) x = k[g() % n];
  u64 sink = 0;
  const size_t Q = 200000000;
  const auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < Q; ++i) sink += rule(q[i & (q.size() - 1)]);
  const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(Q);
  std::printf("%s\trule_ns=%.2f\tbytes=%zu\tcheck=%llu\n", v[2], ns, rule.bytes(), (unsigned long long)sink);
  return 0;
}
