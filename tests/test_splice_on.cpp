// SPDX-License-Identifier: MIT
// SPLICE's parts on GRE's indexes (sp_<parts>_<base>, splice_on.h): every loaded key returns its own payload, for every
// parts string and base, on synthetic key sets (uniform, lognormal, far outliers at both ends, keys that tie as doubles,
// tiny sets), each with T's and V's decisions left to the timed gate and then forced both ways.
//   test_splice_on [KEYS...]   (GRE key files, optional: every name on each as well)
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static std::vector<uint64_t> make(const std::string& kind, size_t n) {
  std::mt19937_64 g(7);
  std::vector<uint64_t> k;
  if (kind == "uniform") { std::uniform_int_distribution<uint64_t> u(1, uint64_t(1) << 50); for (size_t i = 0; i < n; ++i) k.push_back(u(g)); }
  if (kind == "lognormal") { std::lognormal_distribution<double> l(0, 2); for (size_t i = 0; i < n; ++i) k.push_back(uint64_t(1e9 * l(g)) + 1); }
  if (kind == "outliers") {  // fb-like: a dense bulk and a few keys far above it (and one far below)
    std::uniform_int_distribution<uint64_t> u(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34));
    for (size_t i = 0; i < n; ++i) k.push_back(u(g));
    for (uint64_t j = 0; j < 5; ++j) k.push_back((uint64_t(1) << 62) + j * 1000003);
    k.push_back(3);
  }
  if (kind == "ties") {  // above 2^53, adjacent keys equal as doubles: raw LIPP cannot hold them on macOS
    for (size_t i = 0; i < n; ++i) k.push_back((uint64_t(1) << 60) + 3 * i);
  }
  if (kind == "tiny") { for (size_t i = 0; i < n; ++i) k.push_back(10 * i + 5); }
  std::sort(k.begin(), k.end());
  k.erase(std::unique(k.begin(), k.end()), k.end());
  return k;
}

static int check(const std::vector<uint64_t>& k, const std::string& label) {
  std::vector<std::pair<uint64_t, uint64_t>> kv(k.size());
  for (size_t i = 0; i < k.size(); ++i) { uint64_t v = 3 * k[i] + 1; if (v >= ~uint64_t(0) - 1) v = 7; kv[i] = {k[i], v}; }
  int bad = 0;
  std::vector<std::string> parts = {"t", "gt", "gtv", "tv", "g", "v", "gv"}, bases = {"btree", "alex", "pgm", "lipp"};
  auto split = [](const char* e, std::vector<std::string>& out) {  // SPON_PARTS / SPON_BASES: comma lists
    if (!e || !*e) return;
    out.clear();
    for (std::string s = e; !s.empty();) { const size_t c = s.find(','); out.push_back(s.substr(0, c)); s = c == std::string::npos ? "" : s.substr(c + 1); }
  };
  split(std::getenv("SPON_PARTS"), parts);
  split(std::getenv("SPON_BASES"), bases);
  const char* modes[][2] = {{"", ""}, {"raw", "0"}, {"t", "1"}};
  for (const std::string& b : bases)
    for (const std::string& pt : parts)
      for (const auto& md : modes) {
        const std::string name = "sp_" + pt + "_" + b;
        const bool has_t = pt.find('t') != std::string::npos;
        const bool ties = label == "ties" && b == "lipp" && LDBL_MANT_DIG < 64;
        if (ties && !has_t) continue;  // the raw LIPP cannot hold these keys: only names with T are meaningful
        if (ties && std::string(md[0]) == "raw") continue;
        setenv("SPLICE_ON_FORCE", md[0], 1);
        setenv("SPLICE_ON_V", md[1], 1);
        indexInterface<uint64_t, uint64_t>* idx = gret::make_index<uint64_t, uint64_t>(name);
        if (!idx) { std::printf("%s: not registered\n", name.c_str()); return 1; }
        std::vector<std::pair<uint64_t, uint64_t>> in(kv);
        Param p(1, 0); idx->init(&p); idx->bulk_load(in.data(), in.size(), &p);
        size_t miss = 0, wrong = 0;
        for (const auto& e : kv) { uint64_t v = 0; if (!idx->get(e.first, v, &p)) ++miss; else if (v != e.second) ++wrong; }
        if (miss + wrong) std::printf("FAIL %s on %s (force=%s v=%s): missing=%zu wrong=%zu of %zu\n", name.c_str(), label.c_str(), md[0], md[1], miss, wrong, kv.size());
        bad += miss + wrong > 0;
        delete idx;
      }
  return bad;
}

int main(int argc, char** argv) {
  setenv("SPLICE_TIME_N", "2048", 1);
  int bad = 0, cases = 0;
  for (const char* kind : {"uniform", "lognormal", "outliers", "ties"}) { bad += check(make(kind, 20000), kind); cases += 4 * 7 * 3; }
  for (size_t n : {size_t(3), size_t(200)}) { bad += check(make("tiny", n), "tiny" + std::to_string(n)); cases += 4 * 7 * 3; }
  for (int a = 1; a < argc; ++a) {
    FILE* f = std::fopen(argv[a], "rb"); uint64_t n = 0;
    if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
    std::vector<uint64_t> k(n); if (std::fread(k.data(), 8, n, f) != n) return 2; std::fclose(f);
    std::sort(k.begin(), k.end()); k.erase(std::unique(k.begin(), k.end()), k.end());
    bad += check(k, argv[a]); cases += 4 * 7 * 3;
  }
  std::printf("test_splice_on: %s (about %d cases, %d failed)\n", bad ? "FAIL" : "ok", cases, bad);
  return bad ? 1 : 0;
}
