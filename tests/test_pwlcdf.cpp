// SPDX-License-Identifier: MIT
// Unit test of gret::PwlCdf (include/gret/pwlcdf.h) on synthetic key sets, against an independent reference: knots by
// a different integer rounding, the segment by std::upper_bound, and the exact rational y_j + floor((k - x_j) dy / dx)
// in __int128 (PwlCdf may be at most 1 below it, never above).
//
//   c++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude
//       tests/test_pwlcdf.cpp -o test_pwlcdf && ./test_pwlcdf      (one command line; tests/run_tests.sh does this)
//
// Checks on sorted queries (build keys, knots +- 2, random 64-bit keys, keys inside the range, 0 and 2^64-1):
// non-decreasing, T(x_j) = y_j, every output below 2^52, and an exact match with the reference.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "gret/pwlcdf.h"

using gret::PwlCdf;
__extension__ typedef unsigned __int128 u128;
using Keys = std::vector<uint64_t>;

static int failures = 0;
#define CHECK(cond, ...)                                    \
  do {                                                      \
    if (!(cond)) {                                          \
      if (++failures <= 20) {                               \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__);                  \
        std::fprintf(stderr, "\n");                         \
      }                                                     \
    }                                                       \
  } while (0)

struct Ref {
  Keys x, y;
  // knot positions and outputs written as quotient + remainder rounding (half up), not PwlCdf's (2a + b) / 2b
  Ref(const Keys& keys, unsigned K) {
    const u128 n1 = keys.size() - 1, M = gret::kOutMax;
    for (u128 j = 0; j <= K; ++j) {
      const u128 a = j * n1, b = j * M;
      const size_t i = size_t(a / K + (2 * (a % K) >= K ? 1 : 0));
      const uint64_t yj = uint64_t(b / K + (2 * (b % K) >= K ? 1 : 0));
      if (x.empty() || keys[i] != x.back()) x.push_back(keys[i]), y.push_back(yj);
      else if (j == K && x.size() > 1) y.back() = yj;
    }
  }
  // returns the PwlCdf formula; *exact gets y_j + floor((k - x_j) dy / dx)
  uint64_t eval(uint64_t k, uint64_t* exact) const {
    if (k < x.front()) return *exact = y.front();
    if (k >= x.back()) return *exact = y.back();
    const size_t j = size_t(std::upper_bound(x.begin(), x.end(), k) - x.begin()) - 1;
    const u128 dx = x[j + 1] - x[j], dy = y[j + 1] - y[j], d = k - x[j];
    *exact = y[j] + uint64_t(d * dy / dx);  // d * dy < 2^64 2^52
    return y[j] + uint64_t((d * ((dy << 64) / dx)) >> 64);
  }
};

static Keys sorted_unique(Keys v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

static void run_case(const std::string& name, const Keys& keys, unsigned K, std::mt19937_64& rng) {
  const PwlCdf t = PwlCdf::build(keys.data(), keys.size(), K);
  const Ref ref(keys, K);
  const int before = failures;
  // knots
  CHECK(t.knots() == ref.x.size(), "%s K=%u: %zu knots, reference %zu", name.c_str(), K, t.knots(), ref.x.size());
  for (size_t i = 0; i < std::min(t.knots(), ref.x.size()); ++i) {
    CHECK(t.knot_x(i) == ref.x[i] && t.knot_y(i) == ref.y[i], "%s K=%u: knot %zu differs", name.c_str(), K, i);
    CHECK(t(ref.x[i]) == ref.y[i], "%s K=%u: T(x_%zu) != y_%zu", name.c_str(), K, i, i);
  }
  CHECK(ref.y.front() == 0, "%s: y_0 = %llu", name.c_str(), (unsigned long long)ref.y.front());
  if (ref.x.size() > 1)
    CHECK(ref.y.back() == gret::kOutMax, "%s: y_last = %llu", name.c_str(), (unsigned long long)ref.y.back());
  // queries
  Keys q(keys);
  for (uint64_t x : ref.x)
    for (int d = -2; d <= 2; ++d) q.push_back(x + uint64_t(int64_t(d)));  // wraps at 0 and 2^64-1: still keys
  const uint64_t lo = keys.front(), hi = keys.back();
  std::uniform_int_distribution<uint64_t> inside(lo, hi);
  for (int i = 0; i < 20000; ++i) q.push_back(rng()), q.push_back(inside(rng));
  for (uint64_t v : {uint64_t(0), uint64_t(1), UINT64_MAX - 1, UINT64_MAX, lo - 1, lo + 1, hi - 1, hi + 1}) q.push_back(v);
  std::sort(q.begin(), q.end());
  uint64_t prev = 0;
  for (size_t i = 0; i < q.size(); ++i) {
    const uint64_t k = q[i], v = t(k);
    uint64_t exact;
    const uint64_t r = ref.eval(k, &exact);
    CHECK(v <= gret::kOutMax, "%s K=%u: T(%llu) = %llu >= 2^52", name.c_str(), K, (unsigned long long)k,
          (unsigned long long)v);
    CHECK(v == r, "%s K=%u: T(%llu) = %llu, reference %llu", name.c_str(), K, (unsigned long long)k,
          (unsigned long long)v, (unsigned long long)r);
    CHECK(v == exact || v + 1 == exact, "%s K=%u: T(%llu) = %llu, exact %llu", name.c_str(), K,
          (unsigned long long)k, (unsigned long long)v, (unsigned long long)exact);
    CHECK(i == 0 || v >= prev, "%s K=%u: decreasing at %llu", name.c_str(), K, (unsigned long long)k);
    prev = v;
  }
  CHECK(t.bytes() > 0, "%s: bytes", name.c_str());
  std::printf("%-14s n=%-8zu K=%-6u knots=%-6zu buckets=%-6zu steps=%-2u bytes=%-8zu queries=%-7zu %s\n",
              name.c_str(), keys.size(), K, t.knots(), t.radix_buckets(), t.search_steps(), t.bytes(), q.size(),
              failures == before ? "ok" : "FAIL");
}

int main() {
  std::mt19937_64 rng(20261019);
  std::vector<std::pair<std::string, Keys>> sets;
  const size_t n = 100000;
  {
    Keys v(n);
    for (size_t i = 0; i < n; ++i) v[i] = i;
    sets.emplace_back("dense", v);
    for (size_t i = 0; i < n; ++i) v[i] = 1000000007 + 3 * i;
    sets.emplace_back("dense_stride3", v);
  }
  {
    Keys v(n);
    for (auto& k : v) k = rng();
    sets.emplace_back("random64", sorted_unique(v));
  }
  {
    Keys v(n);
    for (auto& k : v) k = (uint64_t(1) << 53) + (rng() >> 4);
    sets.emplace_back("above2^53", sorted_unique(v));
  }
  {
    // fb-like: the bulk below 2^36, a handful of outliers up to 2^64-1
    Keys v(n);
    for (auto& k : v) k = rng() >> 28;
    for (int i = 0; i < 20; ++i) v.push_back(UINT64_MAX - (rng() >> 8));
    v.push_back(UINT64_MAX);
    v.push_back(0);
    sets.emplace_back("outliers", sorted_unique(v));
  }
  {
    Keys v;
    for (uint64_t i = 0; i < n / 2; ++i) v.push_back(1000 + i), v.push_back((uint64_t(1) << 63) + 7 * i);
    sets.emplace_back("two_clusters", sorted_unique(v));
  }
  {
    std::lognormal_distribution<double> ln(0.0, 2.0);
    Keys v(n);
    for (auto& k : v) k = uint64_t(ln(rng) * 1e12);
    sets.emplace_back("lognormal", sorted_unique(v));
  }
  for (auto& s : sets)
    for (unsigned K : {1u, 2u, 7u, 512u, 4096u, 65536u}) run_case(s.first, s.second, K, rng);
  // tiny sets and K > n
  const std::vector<Keys> tiny = {{42}, {0}, {UINT64_MAX}, {5, 9}, {0, UINT64_MAX}, {1, 2, 3},
                                  {0, uint64_t(1) << 63, UINT64_MAX}, {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}};
  for (const Keys& v : tiny)
    for (unsigned K : {1u, 2u, 3u, 4u, 512u, 4096u}) run_case("tiny" + std::to_string(v.size()), v, K, rng);
  {
    Keys v(1000);
    for (auto& k : v) k = rng();
    v = sorted_unique(v);
    for (unsigned K : {999u, 1000u, 1001u, 4096u, 65536u}) run_case("K>n", v, K, rng);
  }
  // a default-constructed PwlCdf maps everything to 0 without reading memory
  const PwlCdf d;
  CHECK(d(0) == 0 && d(UINT64_MAX) == 0, "default PwlCdf");
  if (failures) {
    std::printf("test_pwlcdf: %d failures\n", failures);
    return 1;
  }
  std::printf("test_pwlcdf: all ok\n");
  return 0;
}
