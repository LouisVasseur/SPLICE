// SPDX-License-Identifier: MIT
// Virtual points (vp_<base>, <t>_vp_<base>): every key of a GRE key file returns its own payload (3k + 1), and on the
// comparison bases (btree, pgm) keys that were never loaded, virtual ones included, return false.
//   test_vp KEYS [NAMES...]
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: test_vp KEYS NAME...\n"); return 2; }
  FILE* f = std::fopen(argv[1], "rb"); uint64_t n = 0;
  if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
  std::vector<uint64_t> k(n); if (std::fread(k.data(), 8, n, f) != n) return 2; std::fclose(f);
  std::sort(k.begin(), k.end()); k.erase(std::unique(k.begin(), k.end()), k.end());
  std::vector<std::pair<uint64_t, uint64_t>> kv(k.size());
  for (size_t i = 0; i < k.size(); ++i) { uint64_t v = 3 * k[i] + 1; if (v >= ~uint64_t(0) - 1) v = 7; kv[i] = {k[i], v}; }  // keep clear of the reserved payloads
  int bad = 0;
  for (int a = 2; a < argc; ++a) {
    const std::string name = argv[a];
    indexInterface<uint64_t, uint64_t>* idx = gret::make_index<uint64_t, uint64_t>(name);
    if (!idx) { std::printf("%s: not registered\n", name.c_str()); return 1; }
    Param p(1, 0); idx->init(&p); idx->bulk_load(kv.data(), kv.size(), &p);
    size_t miss = 0, wrong = 0, neg = 0, fp = 0;
    for (const auto& e : kv) { uint64_t v = 0; if (!idx->get(e.first, v, &p)) ++miss; else if (v != e.second) ++wrong; }
    const bool cmp = name.find("btree") != std::string::npos || name.find("pgm") != std::string::npos;
    if (cmp && name.compare(0, 3, "vp_") == 0)  // raw keys only (a transform maps a non-key onto some key's t)
      for (size_t i = 0; i + 1 < k.size(); i += 7)
        for (uint64_t q = k[i] + 1; q < k[i + 1] && q < k[i] + 64; ++q) { uint64_t v = 0; ++neg; fp += idx->get(q, v, &p); }
    std::printf("%-16s n=%zu missing=%zu wrong=%zu non-keys=%zu false_pos=%zu %s\n", name.c_str(), kv.size(), miss, wrong, neg, fp,
                miss + wrong + fp ? "FAIL" : "ok");
    std::fflush(stdout); bad += miss + wrong + fp > 0; delete idx;
  }
  return bad ? 1 : 0;
}
