// SPLICE GRE facade check (C++17; includes only splice_gre.hpp and, through it, splice/layout.hpp, like GRE's TU).
// Loads a SOSD key file the way GRE does (sort, unique), builds through the facade with payload mix64(key), then
// checks every stored key and 1M absent keys with splice::get() on a copy of the View, exactly as SpliceInterface
// does. The facade prints its splice_* lines (layout hash included); this adds the check result and wall times.
//   splice_gre_check SOSD_FILE [LIMIT] [--thp]
// SPLICE_ARGS and SPLICE_BUILD_THREADS are read as in GRE. Exit status 1 on any wrong answer.
#include "splice_gre.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {
std::uint64_t mix64(std::uint64_t x) {  // scaleli/workload.hpp mix64
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
[[noreturn]] void fail(const std::string &m) {
  std::fflush(stdout);
  std::fprintf(stderr, "splice_gre_check: %s\n", m.c_str());
  std::exit(1);
}
std::uint64_t le64(const unsigned char *b) {
  std::uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}
double since_s(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) fail("usage: splice_gre_check SOSD_FILE [LIMIT] [--thp]");
  bool thp = false;
  std::uint64_t limit = 0;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--thp")) thp = true;
    else limit = std::strtoull(argv[i], nullptr, 10);
  }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) fail(std::string("cannot open ") + argv[1]);
  unsigned char b[8];
  if (!in.read(reinterpret_cast<char *>(b), 8)) fail("short header");
  std::uint64_t count = le64(b);
  if (limit && limit < count) count = limit;
  std::vector<std::uint64_t> keys(count);
  for (std::uint64_t i = 0; i < count; ++i) {
    if (!in.read(reinterpret_cast<char *>(b), 8)) fail("short file");
    keys[i] = le64(b);
  }
  // GRE's load_keys: sort, unique, then bulk_load over the sorted pairs
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  std::vector<splice_gre::kv_t> kv(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) kv[i] = splice_gre::kv_t(keys[i], mix64(keys[i]));

  splice_gre::Handle *h = splice_gre::create(thp);
  auto t = std::chrono::steady_clock::now();
  splice_gre::bulk_load(h, kv.data(), kv.size());
  const double build_s = since_s(t);
  const splice::View v = *splice_gre::view(h);  // SpliceInterface's copy

  t = std::chrono::steady_clock::now();
  std::uint64_t found = 0, wrong = 0, val = 0;
  for (std::size_t i = 0; i < kv.size(); ++i) {
    if (!splice::get(v, kv[i].first, val)) continue;
    ++found;
    wrong += val != kv[i].second;
  }
  const double get_s = since_s(t);
  if (found != kv.size() || wrong) {
    std::printf("check_found: %llu of %zu, wrong payloads %llu\n", static_cast<unsigned long long>(found), kv.size(),
                static_cast<unsigned long long>(wrong));
    fail("stored keys not all found with their payload");
  }
  // Absent keys: a mix64 stream plus the ends of the domain and the neighbours of the stored range.
  std::vector<std::uint64_t> probe;
  for (std::uint64_t i = 0; probe.size() < 1000000; ++i) probe.push_back(mix64(i));
  const std::uint64_t top = std::numeric_limits<std::uint64_t>::max();
  probe.push_back(0);
  probe.push_back(1);
  probe.push_back(top);
  probe.push_back(top - 1);
  if (!keys.empty()) {
    probe.push_back(keys.front() - 1);
    probe.push_back(keys.back() + 1);
    for (std::size_t i = 0; i + 1 < keys.size(); i += 997)
      if (keys[i + 1] - keys[i] > 1) probe.push_back(keys[i] + 1);  // inside the bulk, between stored keys
  }
  std::uint64_t absent = 0, false_hits = 0;
  for (std::uint64_t x : probe) {
    if (std::binary_search(keys.begin(), keys.end(), x)) continue;
    ++absent;
    false_hits += splice::get(v, x, val);
  }
  std::printf("check_keys: %zu\n", kv.size());
  std::printf("check_found: %llu\n", static_cast<unsigned long long>(found));
  std::printf("check_absent: %llu\n", static_cast<unsigned long long>(absent));
  std::printf("check_absent_false_hits: %llu\n", static_cast<unsigned long long>(false_hits));
  std::printf("check_total_bytes_per_key: %.3f\n",
              kv.empty() ? 0.0 : double(splice_gre::total_bytes(h)) / double(kv.size()));
  std::printf("check_wall_build_s: %.3f\n", build_s);
  std::printf("check_get_ns_per_key: %.1f\n", kv.empty() ? 0.0 : 1e9 * get_s / double(kv.size()));  // local, not a result
  if (false_hits) fail("absent keys reported as found");
  splice_gre::destroy(h);
  std::printf("splice_gre_check: OK\n");
  return 0;
}
