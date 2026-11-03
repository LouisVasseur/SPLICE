// Payload-chained latency: each key's payload is the next key of one random cycle over all keys, so k = get(k) is a serial
// dependent lookup with no separate key-array read. Reports the mean, sampled percentiles and hardware counters per lookup,
// next to the mean with independent lookups (keys read from an array of 2^20 prepared keys, which stays in cache).
//   chain KEYS NAME LABEL
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <linux/perf_event.h>
#include <random>
#include <string>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
#include <x86intrin.h>
using u64 = uint64_t;

static int open_counter(u64 type, u64 config) {
  perf_event_attr a;
  std::memset(&a, 0, sizeof a);
  a.size = sizeof a;
  a.type = uint32_t(type);
  a.config = config;
  a.disabled = 1;
  a.exclude_kernel = 1;
  a.exclude_hv = 1;
  return int(syscall(SYS_perf_event_open, &a, 0, -1, -1, 0));
}
struct Counters {
  int fd[5];
  long long v[5];
  Counters() {
    const u64 dtlb = PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    fd[0] = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    fd[1] = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    fd[2] = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    fd[3] = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
    fd[4] = open_counter(PERF_TYPE_HW_CACHE, dtlb);
  }
  void start() { for (int f : fd) if (f >= 0) { ioctl(f, PERF_EVENT_IOC_RESET, 0); ioctl(f, PERF_EVENT_IOC_ENABLE, 0); } }
  void stop() { for (int i = 0; i < 5; ++i) { v[i] = -1; if (fd[i] >= 0) { ioctl(fd[i], PERF_EVENT_IOC_DISABLE, 0); if (read(fd[i], &v[i], sizeof v[i]) != sizeof v[i]) v[i] = -1; } } }
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
  if (k.back() == ~u64(0) && (name == "alex" || name == "pgm")) { k.pop_back(); --n; }  // their sentinel, as in GRE's own runs on fb
  std::vector<uint32_t> cyc(n);
  for (u64 i = 0; i < n; ++i) cyc[i] = uint32_t(i);
  std::mt19937_64 g(5);
  for (u64 i = n - 1; i > 0; --i) std::swap(cyc[i], cyc[g() % i]);  // Sattolo: one cycle through all keys
  std::vector<std::pair<u64, u64>> kv(n);
  for (u64 i = 0; i < n; ++i) kv[i] = {k[i], 0};
  for (u64 i = 0; i < n; ++i) kv[cyc[i]].second = k[cyc[(i + 1) % n]];
  std::vector<uint32_t>().swap(cyc);
  auto* idx = gret::make_index<u64, u64>(name);
  if (!idx) { std::printf("%s not registered\n", v[2]); return 3; }
  Param p(1, 0);
  idx->init(&p);
  const auto b0 = std::chrono::steady_clock::now();
  idx->bulk_load(kv.data(), n, &p);
  const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - b0).count();
  std::vector<std::pair<u64, u64>>().swap(kv);
  std::vector<u64> q(size_t(1) << 20);
  for (auto& x : q) x = k[g() % n];
  const u64 start = k[0];
  std::vector<u64>().swap(k);

  const size_t Qi = 100000000, Qc = 20000000;
  size_t hit = 0;
  u64 val = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < Qi; ++i) hit += idx->get(q[i & ((size_t(1) << 20) - 1)], val, &p);
  const double ind = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(Qi);

  Counters c;
  u64 x = start;
  size_t chits = 0;
  c.start();
  t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < Qc; ++i) { u64 w = 0; chits += idx->get(x, w, &p); x = w; }
  const double chain = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(Qc);
  c.stop();

  // sampled latencies, every 64th lookup of the chain; the timer itself costs a few ns
  std::vector<uint32_t> lat;
  lat.reserve(Qc / 64 + 1);
  const auto c0 = std::chrono::steady_clock::now();
  const u64 r0 = __rdtsc();
  unsigned aux;
  for (size_t i = 0; i < Qc; ++i) {
    u64 w = 0;
    if ((i & 63) == 0) {
      const u64 a = __rdtscp(&aux);
      chits += idx->get(x, w, &p);
      const u64 b = __rdtscp(&aux);
      lat.push_back(uint32_t(b - a));
    } else {
      chits += idx->get(x, w, &p);
    }
    x = w;
  }
  const double tsc_ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - c0).count() / double(__rdtsc() - r0);
  std::sort(lat.begin(), lat.end());
  auto pct = [&](double q2) { return double(lat[std::min(lat.size() - 1, size_t(q2 * double(lat.size())))]) * tsc_ns; };
  std::printf("%s\t%s\tspk=%s\tleaf=%s\tindependent_ns=%.1f\tchain_ns=%.1f\tp50=%.0f\tp95=%.0f\tp99=%.0f\tp99.9=%.0f\tinstr=%.1f\tllc_ref=%.2f\tllc_miss=%.2f\tbr_miss=%.2f\tdtlb_miss=%.2f\thits=%zu,%zu\tMB=%lld\tbuild_s=%.1f\tcheck=%llu\n",
              v[3], v[2], getenv("HASHLINE_SPK") ? getenv("HASHLINE_SPK") : "", getenv("AR_LEAF") ? getenv("AR_LEAF") : "", ind, chain,
              pct(0.5), pct(0.95), pct(0.99), pct(0.999), double(c.v[0]) / double(Qc), double(c.v[1]) / double(Qc), double(c.v[2]) / double(Qc),
              double(c.v[3]) / double(Qc), double(c.v[4]) / double(Qc), hit, chits, (long long)(idx->memory_consumption() / 1048576), build_s,
              (unsigned long long)(x ^ val));
  std::fflush(stdout);
}
