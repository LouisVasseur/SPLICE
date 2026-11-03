// SPDX-License-Identifier: MIT
// The deterministic selector for the order-keeping index, as a build would run it: given keys and a byte budget, build every leaf size and
// split threshold on the grid, count exact lines, levels and directory bytes, take the two lowest predicted costs, fit their leaf lines,
// and keep the lower. A period rule, when discovery found one, is tried the same way and pays its transform cost. No lookup is timed.
//   gate_select KEYS BUDGET "A B C D" RULE_NS [RULE_FILE]     ns = A + B (lines - 1) + C (levels - 1) + D log2(directory MB)
// Prints the choice and the seconds it took.
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
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
using u64 = uint64_t;
struct Cand { bool rule; int leaf; double tau; double lines = 0, levels = 0, dirmb = 0, cost = 1e300, bkey = 0; bool ok = false; };
static double A, B, C, D, RULE_NS;
template <class Index>
static void count(Cand& c, std::vector<std::pair<u64, u64>>& kv, const char* budget, int fit) {
  setenv("AB_BUDGET", budget, 1);
  setenv("AB_V", fit ? "1" : "0", 1);
  setenv("AR_LEAF", std::to_string(c.leaf).c_str(), 1);
  setenv("AR_TAU", std::to_string(c.tau).c_str(), 1);
  Index idx(c.rule ? "gr4" : "ab4");
  Param p(1, 0);
  idx.init(&p);
  try {
    idx.bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::exception&) {
    c.ok = false;
    return;
  }
  c.ok = true;
  c.lines = idx.expected_lines(), c.levels = idx.expected_levels(), c.dirmb = double(idx.directory_bytes()) / 1048576;
  c.bkey = double(idx.memory_consumption()) / double(kv.size());
  c.cost = A + B * (c.lines - 1) + C * (c.levels - 1) + D * std::log2(std::max(c.dirmb, 1.0)) + (c.rule ? RULE_NS : 0);
}
int main(int argc, char** v) {
  FILE* f = std::fopen(v[1], "rb");
  u64 n = 0;
  if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
  std::vector<u64> k(n);
  if (std::fread(k.data(), 8, n, f) != n) return 2;
  std::fclose(f);
  if (!std::is_sorted(k.begin(), k.end())) std::sort(k.begin(), k.end());
  k.erase(std::unique(k.begin(), k.end()), k.end());
  std::vector<std::pair<u64, u64>> kv(k.size());
  for (size_t i = 0; i < k.size(); ++i) kv[i] = {k[i], k[i]};
  std::vector<u64>().swap(k);
  const char* budget = v[2];
  std::sscanf(v[3], "%lf %lf %lf %lf", &A, &B, &C, &D);
  RULE_NS = std::strtod(v[4], nullptr);
  const bool have_rule = argc > 5 && v[5][0];
  setenv("RULE_FILE", have_rule ? v[5] : "", 1);
  const auto t0 = std::chrono::steady_clock::now();
  const std::pair<int, double> grid[] = {{4, 2}, {8, 2}, {8, 3}, {16, 2}, {16, 3}, {64, 3}};
  std::vector<Cand> cands;
  for (const auto& g : grid) {
    cands.push_back(Cand{false, g.first, g.second});
    if (have_rule) cands.push_back(Cand{true, g.first, g.second});
  }
  for (auto& c : cands) {
    if (c.rule) count<gret::RuleLineIndex<u64, u64, 4>>(c, kv, budget, 0);
    else count<gret::AdaptiveLineIndex<u64, u64, 4>>(c, kv, budget, 0);
  }
  std::vector<Cand> live;
  for (const auto& c : cands)
    if (c.ok) live.push_back(c);
  if (live.empty()) { std::printf("infeasible\t%s\n", budget); return 5; }
  std::sort(live.begin(), live.end(), [](const Cand& a, const Cand& b) { return a.cost < b.cost; });
  live.resize(std::min<size_t>(2, live.size()));
  for (auto& c : live) {
    if (c.rule) count<gret::RuleLineIndex<u64, u64, 4>>(c, kv, budget, 1);
    else count<gret::AdaptiveLineIndex<u64, u64, 4>>(c, kv, budget, 1);
  }
  const Cand& w = live[0].cost <= (live.size() > 1 ? live[1].cost : 1e300) ? live[0] : live[1];
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s\t%d\t%g\t1\tlines=%.4f\tlevels=%.3f\tdirMB=%.1f\tBkey=%.2f\tpredicted_ns=%.1f\tgate_s=%.1f\n", w.rule ? "gr4" : "ab4", w.leaf, w.tau, w.lines, w.levels, w.dirmb, w.bkey, w.cost, secs);
  return 0;
}
