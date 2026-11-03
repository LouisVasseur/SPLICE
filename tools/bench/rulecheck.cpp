// Every key of a file returns its payload and no key next to one that isn't loaded is found, with the rule active.
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"
#include <algorithm>
#include <cstdio>
#include <vector>
using u64 = uint64_t;
int main(int, char** v) {
  FILE* f = std::fopen(v[1], "rb"); u64 n = 0; if (!f || std::fread(&n, 8, 1, f) != 1) return 2; std::vector<u64> k(n); if (std::fread(k.data(), 8, n, f) != n) return 2; std::fclose(f);
  std::sort(k.begin(), k.end()); k.erase(std::unique(k.begin(), k.end()), k.end()); n = k.size();
  std::vector<std::pair<u64, u64>> kv(n); for (u64 i = 0; i < n; ++i) kv[i] = {k[i], k[i] * 2654435761u + 17};
  auto* idx = gret::make_index<u64, u64>(v[2]); Param p(1, 0); idx->init(&p); idx->bulk_load(kv.data(), n, &p);
  size_t miss = 0, wrong = 0, fp = 0, tested = 0;
  for (u64 i = 0; i < n; ++i) { u64 w = 0; if (!idx->get(k[i], w, &p)) ++miss; else if (w != k[i] * 2654435761u + 17) ++wrong;
    for (u64 x : {k[i] - 1, k[i] + 1}) if (!std::binary_search(k.begin(), k.end(), x)) { ++tested; fp += idx->get(x, w, &p); } }
  for (u64 x : {u64(0), ~u64(0), u64(1) << 63, k.back() + 1000000007}) if (!std::binary_search(k.begin(), k.end(), x)) { u64 w; ++tested; fp += idx->get(x, w, &p); }
  std::printf("rulecheck %s n=%llu missing=%zu wrong=%zu false_positives=%zu of %zu non-keys\n", v[2], (unsigned long long)n, miss, wrong, fp, tested);
  return miss + wrong + fp ? 1 : 0;
}
