// Build one index on a key file and print only its build line: exact line counts and bytes, no lookups.
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
int main(int, char** v) {
  FILE* f = std::fopen(v[1], "rb");
  uint64_t n = 0;
  if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
  std::vector<uint64_t> k(n);
  if (std::fread(k.data(), 8, n, f) != n) return 2;
  std::fclose(f);
  if (!std::is_sorted(k.begin(), k.end())) std::sort(k.begin(), k.end());
  k.erase(std::unique(k.begin(), k.end()), k.end());
  n = k.size();
  std::vector<std::pair<uint64_t, uint64_t>> kv(n);
  for (uint64_t i = 0; i < n; ++i) kv[i] = {k[i], k[i]};
  std::vector<uint64_t>().swap(k);
  auto* idx = gret::make_index<uint64_t, uint64_t>(v[2]);
  Param p(1, 0);
  idx->init(&p);
  try {
    idx->bulk_load(kv.data(), n, &p);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gret: infeasible: %s\n", e.what());
    return 5;
  }
  return 0;
}
