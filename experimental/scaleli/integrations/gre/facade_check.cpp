// Facade check (C++17; includes only scaleli_gre.hpp, like GRE's TU). Builds a cell through the facade from the same
// records scaleli_bench loads (SOSD prefix, value mix64(row), stable sort, last write wins) and replays the read
// operations of a scaleli_bench --dump-trace CSV, printing the read digest (= the bench's result_checksum on a
// read-only trace) and the work counters. compare.py diffs this output against the bench JSON.
//   facade_check CELL SOSD_FILE LIMIT [TRACE_CSV] [--time]
// --time: ns per lookup through the facade's exported get() against the same finds looped inside the facade TU.
#include "scaleli_gre.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
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
  std::fprintf(stderr, "facade_check: %s\n", m.c_str());
  std::exit(1);
}
std::uint64_t le64(const unsigned char *b) {
  std::uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}
// scaleli::read_dataset(path, "sosd", 8, limit): 8-byte count header, little-endian uint64 keys, value mix64(row).
std::vector<scaleli_gre::kv_t> read_sosd(const char *path, std::uint64_t limit) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(std::string("cannot open ") + path);
  unsigned char b[8];
  if (!in.read(reinterpret_cast<char *>(b), 8)) fail("short header");
  std::uint64_t count = le64(b);
  if (limit && limit < count) count = limit;
  std::vector<scaleli_gre::kv_t> out;
  out.reserve(count);
  for (std::uint64_t i = 0; i < count; ++i) {
    if (!in.read(reinterpret_cast<char *>(b), 8)) fail("short file");
    out.emplace_back(le64(b), mix64(i));
  }
  return out;
}
double now_ns() {
  return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) fail("usage: facade_check CELL SOSD_FILE LIMIT [TRACE_CSV] [--time]");
  bool time = false;
  const char *trace = nullptr;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--time")) time = true;
    else trace = argv[i];
  }
  auto rows = read_sosd(argv[2], std::strtoull(argv[3], nullptr, 10));
  // scaleli::canonicalize: stable sort by key, the last input write of a key wins.
  std::stable_sort(rows.begin(), rows.end(), [](const scaleli_gre::kv_t &a, const scaleli_gre::kv_t &b) {
    return a.first < b.first;
  });
  std::size_t dst = 0;
  for (const auto &r : rows) {
    if (dst && rows[dst - 1].first == r.first) rows[dst - 1].second = r.second;
    else rows[dst++] = r;
  }
  rows.resize(dst);

  scaleli_gre::Handle *h = scaleli_gre::create(argv[1]);
  scaleli_gre::bulk_load(h, rows.data(), rows.size());
  std::printf("accounted_bytes: %lld\n", scaleli_gre::accounted_bytes(h));

  std::vector<std::uint64_t> keys;
  if (trace) {
    std::ifstream in(trace);
    if (!in) fail(std::string("cannot open ") + trace);
    std::string line;
    std::getline(in, line);  // header: operation,key,value,length
    while (std::getline(in, line)) {
      std::istringstream f(line);
      std::string op, key;
      std::getline(f, op, ',');
      std::getline(f, key, ',');
      if (op != "read_hit" && op != "read_miss") fail("only read traces are replayed, got " + op);
      keys.push_back(std::strtoull(key.c_str(), nullptr, 10));
    }
    scaleli_gre::Counters c{};
    const std::uint64_t d = scaleli_gre::replay_reads(h, keys.data(), keys.size(), &c);
    std::printf("reads: %zu\nchecksum: %llu\n", keys.size(), static_cast<unsigned long long>(d));
    std::printf("counters: {\"root_probes\":%llu,\"coordinate_probes\":%llu,\"fence_probes\":%llu,\"delta_probes\":%llu,"
                "\"key_at_calls\":%llu,\"decoded_keys\":%llu,\"codec_bytes_examined\":%llu,\"block_routes\":%llu,"
                "\"correction_distance\":%llu,\"max_correction_distance\":%llu,\"transform_calls\":%llu}\n",
                (unsigned long long)c.root_probes, (unsigned long long)c.coordinate_probes,
                (unsigned long long)c.fence_probes, (unsigned long long)c.delta_probes,
                (unsigned long long)c.key_at_calls, (unsigned long long)c.decoded_keys,
                (unsigned long long)c.codec_bytes_examined, (unsigned long long)c.block_routes,
                (unsigned long long)c.correction_distance, (unsigned long long)c.max_correction_distance,
                (unsigned long long)c.transform_calls);
  }
  if (time) {
    if (keys.empty())
      for (std::size_t i = 0; i < rows.size(); ++i) keys.push_back(rows[mix64(i) % rows.size()].first);
    double via = 1e300, direct = 1e300;  // best of 5 alternating repetitions
    std::uint64_t f1 = 0, f2 = 0;
    for (int rep = 0; rep < 5; ++rep) {
      double t = now_ns();
      std::uint64_t v = 0, found = 0;
      for (auto k : keys) found += scaleli_gre::get(h, k, &v);
      via = std::min(via, (now_ns() - t) / keys.size());
      f1 = found;
      t = now_ns();
      f2 = scaleli_gre::direct_reads(h, keys.data(), keys.size());
      direct = std::min(direct, (now_ns() - t) / keys.size());
    }
    if (f1 != f2) fail("facade and direct lookups disagree");
    std::printf("lookup_ns_facade: %.2f\nlookup_ns_direct: %.2f\nlookup_ns_overhead: %.2f\n", via, direct, via - direct);
  }
  scaleli_gre::destroy(h);
  return 0;
}
