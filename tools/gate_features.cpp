// SPDX-License-Identifier: MIT
// Print the gate features of GRE key files as JSON, one object per line, through the same code as gate.h
// (compute_features, nfl_switch). For gate/train_gate.py.
//
//   gate_features FILE...     FILE: a GRE binary key file (uint64 count, then the keys)
//
// The keys are sorted and deduplicated first, as GRE does before bulk_load (with --init_table_ratio=1 every key is
// bulk-loaded). The JSON adds file, keys_in_file, feature_ms (compute_features alone) and nflsw (NFL's switch).
// Build (standard library only):  c++ -std=c++17 -O2 -Iinclude tools/gate_features.cpp -o build/gate_features
#define GRET_FEATURES_ONLY
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gret/gate.h"

static bool read_gre(const char* path, std::vector<uint64_t>& keys, uint64_t& count) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  bool ok = std::fread(&count, sizeof count, 1, f) == 1;
  if (ok) {
    keys.resize(size_t(count));
    ok = std::fread(keys.data(), sizeof(uint64_t), keys.size(), f) == keys.size();
  }
  std::fclose(f);
  return ok;
}

static std::string json_string(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s GRE_KEY_FILE...\n", argv[0]);
    return 2;
  }
  int status = 0;
  for (int a = 1; a < argc; ++a) {
    std::vector<uint64_t> keys;
    uint64_t count = 0;
    if (!read_gre(argv[a], keys, count)) {
      std::fprintf(stderr, "gate_features: cannot read %s (uint64 count, then that many keys)\n", argv[a]);
      status = 1;
      continue;
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const auto t0 = std::chrono::steady_clock::now();
    const gret::GateFeatures f = gret::compute_features(keys.data(), keys.size());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("{\"file\": %s, \"keys_in_file\": %llu, %s, \"feature_ms\": %.3f, \"nflsw\": \"%s\"}\n",
                json_string(argv[a]).c_str(), (unsigned long long)count, f.json().c_str(), ms,
                gret::variant_name(gret::nfl_switch(f)));
    std::fflush(stdout);
  }
  return status;
}
