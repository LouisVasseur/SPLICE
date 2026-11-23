// SPDX-License-Identifier: MIT
// regret: what the simulator counts, timed. It bulk-loads aw4 on a base key set, times a stream of puts, then times lookups
// spread evenly over EVERY key now stored (GRE's own insert workload only ever reads the keys that were bulk-loaded, so it never
// pays for a key that went to the overflow), and does the same on a fresh ab4 built from all the keys at the same total bytes.
//   regret = ns per lookup in place / ns per lookup fresh
// It also times the fresh build, which gives the break-even of a rebuild: with dt = in place - fresh (ns per lookup) and tb the
// build's ns per key, a rebuild is paid back after tb / dt lookups per key.
//
//   regret KEYS N STREAM GROWTH      STREAM: holdout | hot | append    GROWTH: inserts as a fraction of the base, e.g. 0.2
//   environment: the index's (AB_BUDGET default 20, AR_LEAF, AR_TAU, AB_V), AW_PUT (line | wide), LOOKUPS (default 20000000),
//                SEED (1). Run it pinned to a core and with huge pages off, as the benchmark is.
//                PUT_LAT=1 times every put on its own (a clock read around each, so ns_put is some 40 ns higher) and adds the
//                99th and 99.9th percentile and the longest put to the line: what a rebuild makes one put wait.
// BASELINE=<name> (alex, lipp, ...) runs that index instead, through the same loops: bulk load on the base, the puts, lookups over
// every key. Its line has the put and read times and its bytes a key, so the three can be set side by side.
// Output: one tab-separated line (and a header line), and the index's own lines on stderr.
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
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
using KV = std::pair<uint64_t, uint64_t>;
using Clock = std::chrono::steady_clock;
uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }
double env_d(const char* k, double def) {
  const char* e = std::getenv(k);
  return e && *e ? std::strtod(e, nullptr) : def;
}
double ns_since(Clock::time_point t0) { return std::chrono::duration<double, std::nano>(Clock::now() - t0).count(); }

// ns per lookup over these keys, through the interface as GRE calls it; every key must be found with its payload
// noinline: one compiled loop for every index, calling get through the interface, so no index gets its lookup inlined into it
__attribute__((noinline)) double time_lookups(indexInterface<uint64_t, uint64_t>* idx, const std::vector<uint64_t>& q, size_t& wrong) {
  uint64_t val = 0, sum = 0;
  size_t bad = 0;
  const auto t0 = Clock::now();
  for (uint64_t k : q) {
    const bool ok = idx->get(k, val);
    bad += size_t(!ok);
    sum += val;
  }
  const double ns = ns_since(t0) / double(q.size());
  uint64_t want = 0;
  for (uint64_t k : q) want += payload(k);
  wrong += bad + size_t(sum != want);
  return ns;
}
// one put through the interface, never inlined into the loops below (GRE calls it this way, and the whole write path inlined
// into two loops takes the compiler tens of minutes)
__attribute__((noinline)) bool put_one(indexInterface<uint64_t, uint64_t>* idx, uint64_t k) { return idx->put(k, payload(k)); }
}  // namespace

// the in-place index and its fresh counterpart: aw4 / ab4, or with AW_G=1 gw4 / gp4 (G's period rule from $RULE_FILE in both)
template <class AW, class FR>
int run_aw(const char* awn, const char* frn, std::vector<KV>& kv, const std::vector<uint64_t>& base, const std::vector<uint64_t>& ops,
           const std::vector<uint64_t>& all, size_t n, size_t nq, std::mt19937_64& g, double budget, const std::string& stream, double growth) {
  char buf[32];
  auto* aw = new AW(awn);
  aw->init();
  aw->bulk_load(kv.data(), kv.size());
  std::vector<KV>().swap(kv);
  const double lines0 = aw->expected_lines();

  // lookups: over the base keys before the stream, and over all keys after it (the same positions of `all`, so the fresh index
  // answers the same questions)
  std::vector<uint64_t> q_base(nq), q_all(nq), q_new(std::min(nq, ops.size() * 4));
  for (uint64_t& k : q_base) k = base[g() % base.size()];
  for (uint64_t& k : q_all) k = all[g() % n];
  for (uint64_t& k : q_new) k = ops[g() % ops.size()];
  size_t wrong = 0;
  const double ns_before = time_lookups(aw, q_base, wrong);

  size_t put_fail = 0;
  const bool lat = env_d("PUT_LAT", 0) != 0;
  std::vector<float> each;
  if (lat) each.reserve(ops.size());
  const auto tp = Clock::now();
  if (lat) {
    for (uint64_t k : ops) {
      const auto t1 = Clock::now();
      put_fail += size_t(!put_one(aw, k));
      each.push_back(float(ns_since(t1)));
    }
  } else {
    for (uint64_t k : ops) put_fail += size_t(!put_one(aw, k));
  }
  const double ns_put = ns_since(tp) / double(ops.size());
  double p99 = NAN, p999 = NAN, pmax = NAN;
  if (lat && !each.empty()) {
    std::sort(each.begin(), each.end());
    p99 = each[size_t(0.99 * double(each.size() - 1))], p999 = each[size_t(0.999 * double(each.size() - 1))], pmax = each.back();
    std::vector<float>().swap(each);
  }
  size_t in_line = 0, general = 0, to_ov = 0;
  aw->put_counts(in_line, general, to_ov);
  const double bytes = double(aw->memory_consumption());

  const double ns_all = time_lookups(aw, q_all, wrong), ns_base_after = time_lookups(aw, q_base, wrong), ns_new = time_lookups(aw, q_new, wrong);
  const size_t ov_now = aw->overflow_keys();
  delete aw;

  // the same keys in a fresh read-only index holding the same bytes
  kv.reserve(n);
  for (uint64_t k : all) kv.emplace_back(k, payload(k));
  const double same = bytes / double(n);
  std::snprintf(buf, sizeof buf, "%.4f", same);
  setenv("AB_BUDGET", buf, 1);
  double ns_fresh = NAN, build_ns = NAN, lines_fresh = NAN;
  try {
    auto* fr = new FR(frn);
    fr->init();
    const auto tb = Clock::now();
    fr->bulk_load(kv.data(), kv.size());
    build_ns = ns_since(tb) / double(n);
    lines_fresh = fr->expected_lines();
    ns_fresh = time_lookups(fr, q_all, wrong);
    delete fr;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "regret: fresh build at %.2f bytes a key: %s\n", same, e.what());
  }
  const double dt = ns_all - ns_fresh;
  std::printf("stream\tgrowth\tbase\tinserts\tbudget\tput_rule\tleaf\tlines0\tns_read_before\tns_put\tin_line_pct\tgeneral_pct\toverflow_pct\tbytes_per_key\tns_read_all\tns_read_base\tns_read_inserted\tns_read_fresh\tregret_time\tlines_fresh\tbuild_ns_per_key\tlookups_per_key_to_repay\tput_refused\twrong\tput_p99_ns\tput_p999_ns\tput_max_ns\n");
  std::printf("%s\t%.3f\t%zu\t%zu\t%.2f\t%s\t%s\t%.4f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.2f\t%.1f\t%.1f\t%.1f\t%.1f\t%.3f\t%.4f\t%.1f\t%.1f\t%zu\t%zu\t%.0f\t%.0f\t%.0f\n", stream.c_str(), growth, base.size(), ops.size(), budget,
              std::getenv("AW_PUT") ? std::getenv("AW_PUT") : "line", std::getenv("AR_LEAF") ? std::getenv("AR_LEAF") : "64", lines0, ns_before, ns_put,
              100.0 * double(in_line) / double(ops.size()), 100.0 * double(general) / double(ops.size()), 100.0 * double(ov_now) / double(ops.size()), same, ns_all, ns_base_after, ns_new, ns_fresh,
              ns_all / ns_fresh, lines_fresh, build_ns, dt > 0 ? build_ns / dt : INFINITY, put_fail, wrong, p99, p999, pmax);
  return wrong || put_fail ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 5) return std::fprintf(stderr, "usage: regret KEYS N STREAM GROWTH\n"), 2;
  const std::string path = argv[1], stream = argv[3];
  const size_t want = size_t(std::strtoull(argv[2], nullptr, 10));
  const double growth = std::strtod(argv[4], nullptr), budget = env_d("AB_BUDGET", 20);
  const size_t nq = size_t(env_d("LOOKUPS", 20000000));
  std::mt19937_64 g(uint64_t(env_d("SEED", 1)));
  std::vector<uint64_t> all;
  {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return std::perror(path.c_str()), 2;
    uint64_t have = 0;
    if (std::fread(&have, 8, 1, f) != 1) return 2;
    all.resize(std::min<uint64_t>(have, want));
    if (std::fread(all.data(), 8, all.size(), f) != all.size()) return 2;
    std::fclose(f);
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
  }
  const size_t n = all.size(), s = size_t(growth * double(n) / (1 + growth));
  std::vector<char> mark(n, 0);
  if (stream == "holdout") {
    for (size_t c = 0; c < s;) { const size_t i = g() % n; if (!mark[i]) mark[i] = 1, ++c; }
  } else if (stream == "hot") {
    const size_t w = std::min(n / 2, std::max(2 * s, n / 50)), r0 = size_t(0.6 * double(n - w));
    for (size_t c = 0; c < s;) { const size_t i = r0 + g() % w; if (!mark[i]) mark[i] = 1, ++c; }
  } else if (stream == "append") {
    for (size_t i = n - s; i < n; ++i) mark[i] = 1;
  } else {
    return std::fprintf(stderr, "unknown stream %s\n", stream.c_str()), 2;
  }
  std::vector<uint64_t> base, ops;
  for (size_t i = 0; i < n; ++i) (mark[i] ? ops : base).push_back(all[i]);
  std::vector<char>().swap(mark);
  if (stream != "append") std::shuffle(ops.begin(), ops.end(), g);
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.4f", budget);
  setenv("AB_BUDGET", buf, 1);

  std::vector<KV> kv;
  kv.reserve(base.size());
  for (uint64_t k : base) kv.emplace_back(k, payload(k));
  if (const char* bn = std::getenv("BASELINE")) {
    std::unique_ptr<indexInterface<uint64_t, uint64_t>> bi(gret::make_index<uint64_t, uint64_t>(bn));
    if (!bi) return std::fprintf(stderr, "regret: no index named %s\n", bn), 2;
    bi->init();
    bi->bulk_load(kv.data(), kv.size());
    std::vector<KV>().swap(kv);
    std::vector<uint64_t> qb(nq), qa(nq);
    for (uint64_t& k : qb) k = base[g() % base.size()];
    for (uint64_t& k : qa) k = all[g() % n];
    size_t wrong = 0, put_fail = 0;
    const double before = time_lookups(bi.get(), qb, wrong);
    const auto tp = Clock::now();
    for (uint64_t k : ops) put_fail += size_t(!bi->put(k, payload(k)));
    const double ns_put = ns_since(tp) / double(ops.size());
    const double after = time_lookups(bi.get(), qa, wrong), bytes = double(bi->memory_consumption());
    std::printf("stream\tgrowth\tbase\tinserts\tindex\tns_read_before\tns_put\tns_read_all\tbytes_per_key\tput_refused\twrong\n");
    std::printf("%s\t%.3f\t%zu\t%zu\t%s\t%.1f\t%.1f\t%.1f\t%.2f\t%zu\t%zu\n", stream.c_str(), growth, base.size(), ops.size(), bn, before, ns_put, after, bytes / double(n), put_fail, wrong);
    std::fflush(stdout);
    std::_Exit(wrong || put_fail ? 1 : 0);  // no destructor: some of these indexes take minutes to free
  }
  if (env_d("AW_G", 0) != 0)
    return run_aw<gret::AdaptiveLineIndex<uint64_t, uint64_t, 4, gret::FastPeriodRule, true>, gret::FastRuleLineIndex<uint64_t, uint64_t, 4>>(
        "gw4", "gp4", kv, base, ops, all, n, nq, g, budget, stream, growth);
  return run_aw<gret::AdaptiveLineIndex<uint64_t, uint64_t, 4, gret::IdentityCoord, true>, gret::AdaptiveLineIndex<uint64_t, uint64_t, 4>>(
      "aw4", "ab4", kv, base, ops, all, n, nq, g, budget, stream, growth);
}
