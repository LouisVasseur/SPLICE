// SPDX-License-Identifier: MIT
// NFL's own flow (gret::NflFlow) on a GRE key file, against NFL's weights for it:
//   test_nfl KEYS WEIGHTS                -> exactness of nfl_btree/pgm/alex/lipp (distinct payloads 3k+1, every key)
//   test_nfl KEYS WEIGHTS --dump N OUT   -> N stride-sampled sorted keys and their z, binary (u64 key, f64 z), for
//                                           tests/nfl_parity.py
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<uint64_t> read_keys(const char* path) {
  FILE* f = std::fopen(path, "rb");
  uint64_t n = 0;
  if (!f || std::fread(&n, 8, 1, f) != 1) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(2); }
  std::vector<uint64_t> k(n);
  if (std::fread(k.data(), 8, n, f) != n) { std::fprintf(stderr, "short file %s\n", path); std::exit(2); }
  std::fclose(f);
  std::sort(k.begin(), k.end());
  k.erase(std::unique(k.begin(), k.end()), k.end());
  return k;
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: test_nfl KEYS WEIGHTS [--dump N OUT]\n"); return 2; }
  const std::vector<uint64_t> k = read_keys(argv[1]);
  if (argc == 6 && std::strcmp(argv[3], "--dump") == 0) {
    const gret::NflFlow f = gret::NflFlow::load(argv[2]);
    const size_t N = std::strtoull(argv[4], nullptr, 10), step = std::max<size_t>(1, k.size() / N);
    FILE* o = std::fopen(argv[5], "wb");
    size_t c = 0;
    for (size_t i = 0; i < k.size() && c < N; i += step, ++c) { const double z = f.z(k[i]); std::fwrite(&k[i], 8, 1, o); std::fwrite(&z, 8, 1, o); }
    std::fclose(o);
    std::printf("dumped %zu keys\n", c);
    return 0;
  }
  setenv("TRANSFORM_NFL", argv[2], 1);
  std::vector<std::pair<uint64_t, uint64_t>> kv(k.size());
  for (size_t i = 0; i < k.size(); ++i) kv[i] = {k[i], 3 * k[i] + 1};
  int bad_total = 0;
  for (const char* name : {"nfl_btree", "nfl_pgm", "nfl_alex", "nfl_lipp"}) {
    indexInterface<uint64_t, uint64_t>* idx = gret::make_index<uint64_t, uint64_t>(name);
    if (!idx) { std::printf("%s: not registered\n", name); return 1; }
    Param p(1, 0);
    idx->init(&p);
    idx->bulk_load(kv.data(), kv.size(), &p);
    size_t miss = 0, wrong = 0;
    for (const auto& e : kv) {
      uint64_t v = 0;
      if (!idx->get(e.first, v, &p)) ++miss;
      else if (v != e.second) ++wrong;
    }
    std::printf("%-10s n=%zu missing=%zu wrong_payload=%zu %s\n", name, kv.size(), miss, wrong, miss + wrong ? "FAIL" : "ok");
    std::fflush(stdout);
    bad_total += miss + wrong > 0;
  }
  return bad_total ? 1 : 0;
}
