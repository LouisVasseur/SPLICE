// SPDX-License-Identifier: MIT
// Worst cases for the line-read indexes: key sets chosen to break one multiply (regular strides), and equal-width splits (end
// outliers past what G peels, huge gaps, tight clusters in a wide range). Every stored key is found and no other key is; the build's
// exact line counts stay bounded, in the mean and in the worst key, where they used to reach 15 lines (hf4) and 115680 (ab4).
//   test_worstcase [KEYS_PER_SET]
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

using u64 = uint64_t;

template <class Index>
static bool exact(Index& idx, std::vector<std::pair<u64, u64>>& kv, std::mt19937_64& g) {
  Param p(1, 0);
  idx.init(&p);
  idx.bulk_load(kv.data(), kv.size(), &p);
  u64 v = 0;
  const size_t n = kv.size(), step = std::max<size_t>(1, n / 200000);
  for (size_t i = 0; i < n; i += step)
    if (!idx.get(kv[i].first, v, &p) || v != kv[i].second) return false;
  for (size_t i = 0; i < 200000; ++i) {
    const u64 x = i & 1 ? g() : kv[g() % n].first + 1 + (g() & 3);
    const bool stored = std::binary_search(kv.begin(), kv.end(), std::pair<u64, u64>(x, 0), [](const std::pair<u64, u64>& a, const std::pair<u64, u64>& b) { return a.first < b.first; });
    if (idx.get(x, v, &p) != stored) return false;
  }
  return true;
}

int main(int argc, char** argv) {
  const u64 N = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000;
  const u64 M = ~u64(0);
  std::mt19937_64 g(5);
  std::vector<std::pair<std::string, std::vector<u64>>> sets;
  auto stride = [&](u64 base, u64 d) { std::vector<u64> v; for (u64 i = 0; i < N; ++i) v.push_back(base + i * d); return v; };
  sets.push_back({"sequential", stride(0, 1)});
  { std::vector<u64> v(N); for (auto& x : v) x = g(); sets.push_back({"random64", v}); }
  { std::vector<u64> v(N); for (auto& x : v) x = (g() >> 40) << 40; sets.push_back({"high_bits", v}); }
  for (u64 d : {u64(1) << 13, u64(1) << 16, u64(1) << 19, u64(100000), u64(3524578), u64(14930352), u64(1134903170)}) sets.push_back({"stride_" + std::to_string(d), stride(12345, d)});
  { std::vector<u64> v; for (u64 i = 0; i < N / 2; ++i) { v.push_back(i); v.push_back(M - i); } sets.push_back({"both_ends", v}); }
  { std::vector<u64> v = stride(u64(1) << 40, 7); for (u64 j = 0; j < 65; ++j) v.push_back(M - j * 1000003); sets.push_back({"outliers_65", v}); }
  { std::vector<u64> v = stride(u64(1) << 40, 7); for (u64 j = 0; j < 1000; ++j) v.push_back(M - j * 1000003); sets.push_back({"outliers_1000", v}); }
  { std::vector<u64> v; for (u64 c = 0; c < 1000; ++c) { const u64 b = g() >> 1; for (u64 i = 0; i < N / 1000; ++i) v.push_back(b + i * 3); } sets.push_back({"clusters_1000", v}); }
  { std::vector<u64> v; for (u64 c = 0; c < 10; ++c) { const u64 b = g() >> 1; for (u64 i = 0; i < N / 20; ++i) v.push_back(b + i); } for (u64 i = 0; i < N / 2; ++i) v.push_back(g()); sets.push_back({"clusters_and_noise", v}); }
  { std::vector<u64> v = stride(1000, 1); v.resize(N / 2); for (u64 i = 0; i < N / 2; ++i) v.push_back(g()); sets.push_back({"dense_and_sparse", v}); }

  setenv("HASHLINE_SPK", "1.25", 1);
  setenv("AR_TAU", "2", 1);
  std::printf("%-20s | %-22s | %-22s | %-22s\n", "set", "hf4 lines max fb", "ab4 T8 lines max", "ph4 T1024 lines max fb");
  size_t bad = 0;
  for (auto& [label, keys] : sets) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<std::pair<u64, u64>> kv(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], keys[i] * 31 + 7};
    gret::HashLineIndex<u64, u64, 4, gret::mul64, true> hf("hf4");
    setenv("AR_LEAF", "8", 1);
    gret::AdaptiveLineIndex<u64, u64, 4> ab("ab4"), abv("ab4v");
    const bool ok_hf = exact(hf, kv, g), ok_ab = exact(ab, kv, g);
    setenv("AB_V", "1", 1);
    const bool ok_abv = exact(abv, kv, g);
    setenv("AB_V", "0", 1);
    setenv("AR_LEAF", "1024", 1);
    gret::PartitionHashIndex<u64, u64, 4> ph("ph4");
    const bool ok_ph = exact(ph, kv, g);
    std::printf("%-20s | %6.3f %4zu %d | %6.3f %4zu | %6.3f %4zu %d\n", label.c_str(), hf.expected_lines(), hf.worst_lines(), int(hf.uses_fallback()), ab.expected_lines(), ab.worst_lines(), ph.expected_lines(), ph.worst_lines(), int(ph.uses_fallback()));
    const bool within = hf.expected_lines() <= 1.6 && hf.worst_lines() <= 40 && ph.expected_lines() <= 1.6 && ph.worst_lines() <= 40 && ab.expected_lines() <= 3 && ab.worst_lines() <= 200 &&
                        abv.expected_lines() <= ab.expected_lines() + 1e-9 && abv.worst_lines() <= ab.worst_lines() && abv.memory_consumption() <= ab.memory_consumption();
    if (!ok_hf || !ok_ab || !ok_abv || !ok_ph || !within) {
      ++bad;
      std::printf("  FAIL %s: exact %d %d %d, line bounds %d\n", label.c_str(), int(ok_hf), int(ok_ab), int(ok_ph), int(within));
    }
  }
  std::printf("test_worstcase: %s (%zu sets, %zu failed)\n", bad ? "FAILED" : "ok", sets.size(), bad);
  return bad ? 1 : 0;
}
