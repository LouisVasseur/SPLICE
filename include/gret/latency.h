// SPDX-License-Identifier: MIT
// The machine's price of a dependent access, by footprint: measured once per process, at the first bulk_load that asks.
//
//   gret::access_cost(bytes)  ->  the cost of one dependent cache-line access into a structure of `bytes`, in units of
//                                 one access into DRAM (a structure far larger than any cache): about 0.05 for an L1/L2-
//                                 resident structure, up to 1.0 for one that lives in DRAM.
//
// How: a dependent pointer chase over 64-byte lines in a random cycle (Sattolo's algorithm, fixed seed), for buffers of
// 256 KiB, 1, 4, 16, 64 and 256 MiB; 2^17 steps each after a warm pass; the median of 3 runs. Costs between two sizes
// are interpolated in log(bytes); the curve is flat beyond 256 MiB. Each cost is rounded to 1/64, so the decisions that
// use it are stable from run to run on one machine. SPLICE_ACCESS_COSTS="c1,c2,c3,c4,c5,c6" (in DRAM units, for the six
// sizes) skips the probe: for reproducing a decision on another machine. About 0.2 s and 256 MiB, once.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <vector>

namespace gret {

struct AccessCurve {
  static constexpr int kN = 6;
  double bytes[kN] = {256.0 * 1024, 1048576.0, 4194304.0, 16777216.0, 67108864.0, 268435456.0};
  double cost[kN] = {0, 0, 0, 0, 0, 1};  // in DRAM-access units (cost[kN - 1] = 1)
  double ns_dram = 0;                    // ns per dependent DRAM access (diagnostic)
  bool measured = false;
};

inline double chase_ns(size_t lines, uint64_t seed) {
  // one 64-byte line per node; next index in the first 8 bytes
  std::vector<uint64_t> buf(lines * 8);
  std::vector<uint32_t> perm(lines);
  for (size_t i = 0; i < lines; ++i) perm[i] = uint32_t(i);
  std::mt19937_64 g(seed);
  for (size_t i = lines - 1; i > 0; --i) {  // Sattolo: one cycle through every line
    std::uniform_int_distribution<size_t> u(0, i - 1);
    std::swap(perm[i], perm[u(g)]);
  }
  for (size_t i = 0; i < lines; ++i) buf[size_t(perm[i]) * 8] = perm[(i + 1) % lines];
  const size_t steps = size_t(1) << 17;
  uint64_t p = perm[0];
  for (size_t s = 0; s < std::min(lines, steps); ++s) p = buf[p * 8];  // warm: touch the cycle once (or 2^17 steps)
  const auto t0 = std::chrono::steady_clock::now();
  for (size_t s = 0; s < steps; ++s) p = buf[p * 8];
  const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(steps);
  if (p == ~uint64_t(0)) std::printf("!");
  return ns;
}

inline const AccessCurve& access_curve() {
  static AccessCurve c;
  static std::once_flag once;
  std::call_once(once, [] {
    if (const char* e = std::getenv("SPLICE_ACCESS_COSTS"); e && *e) {
      char* q = const_cast<char*>(e);
      for (int i = 0; i < AccessCurve::kN; ++i) { c.cost[i] = std::strtod(q, &q); if (*q == ',') ++q; }
      c.measured = false;
      return;
    }
    double ns[AccessCurve::kN];
    for (int i = 0; i < AccessCurve::kN; ++i) {
      double r[3];
      for (int k = 0; k < 3; ++k) r[k] = chase_ns(size_t(c.bytes[i] / 64), 1866 + uint64_t(k) + 7 * uint64_t(i));
      std::sort(r, r + 3);
      ns[i] = r[1];
    }
    c.ns_dram = ns[AccessCurve::kN - 1];
    for (int i = 0; i < AccessCurve::kN; ++i)
      c.cost[i] = std::min(1.0, std::round(64.0 * ns[i] / c.ns_dram) / 64.0);
    c.measured = true;
    std::fprintf(stderr, "gret: access costs (DRAM units) 256K %.3f 1M %.3f 4M %.3f 16M %.3f 64M %.3f 256M %.3f; "
                 "DRAM %.1f ns\n", c.cost[0], c.cost[1], c.cost[2], c.cost[3], c.cost[4], c.cost[5], c.ns_dram);
  });
  return c;
}

inline double access_cost(double bytes) {
  const AccessCurve& c = access_curve();
  if (bytes <= c.bytes[0]) return c.cost[0];
  for (int i = 1; i < AccessCurve::kN; ++i)
    if (bytes <= c.bytes[i]) {
      const double t = std::log(bytes / c.bytes[i - 1]) / std::log(c.bytes[i] / c.bytes[i - 1]);
      return c.cost[i - 1] + t * (c.cost[i] - c.cost[i - 1]);
    }
  return c.cost[AccessCurve::kN - 1];
}

}  // namespace gret
