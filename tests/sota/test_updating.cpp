// SPDX-License-Identifier: MIT
// alexu and btreeu (include/gret/sota/updating.h): update on stored and absent keys against a std::map, with get after each batch,
// and put and remove still behaving as GRE's wrappers do. Key sets without far outliers: raw ALEX's own predict overflows on
// those (see tests/run_tests.sh), which is not what is tested here.
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <vector>

int main() {
  std::mt19937_64 g(3);
  size_t bad = 0, cases = 0;
  for (const char* name : {"alexu", "btreeu"})
    for (size_t n : {size_t(1000), size_t(200000), size_t(2000000)}) {
      std::vector<uint64_t> keys;
      for (size_t i = 0; i < n; ++i) keys.push_back((g() >> 16) | 1);
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      std::vector<std::pair<uint64_t, uint64_t>> kv;
      for (uint64_t k : keys) kv.emplace_back(k, k * 7 + 1);
      std::unique_ptr<indexInterface<uint64_t, uint64_t>> idx(gret::make_index<uint64_t, uint64_t>(name));
      if (!idx) { std::fprintf(stderr, "%s: not registered\n", name); return 1; }
      idx->init();
      idx->bulk_load(kv.data(), kv.size());
      std::map<uint64_t, uint64_t> ref(kv.begin(), kv.end());
      size_t e = 0;
      auto check = [&](uint64_t k) {
        uint64_t v = 0;
        const bool got = idx->get(k, v);
        const auto it = ref.find(k);
        if (got != (it != ref.end()) || (got && v != it->second)) ++e;
      };
      for (int round = 0; round < 3; ++round) {
        for (size_t i = 0; i < keys.size() / 2 + 10; ++i) {
          const uint64_t k = keys[g() % keys.size()] + (g() % 4 == 0 ? 2 : 0);  // even keys are never stored
          const unsigned r = unsigned(g() % 8);
          const auto it = ref.find(k);
          if (r < 5) {
            const uint64_t v = g();
            if (idx->update(k, v) != (it != ref.end())) ++e;
            else if (it != ref.end()) it->second = v;
          } else if (r == 5) {
            if (idx->remove(k) != (it != ref.end())) ++e;
            else if (it != ref.end()) ref.erase(it);
          } else {
            const uint64_t v = g();
            if (idx->put(k, v) != (it == ref.end())) ++e;
            else if (it == ref.end()) ref[k] = v;
          }
        }
        for (uint64_t k : keys) check(k), check(k + 2);
      }
      if (e) std::fprintf(stderr, "FAIL %s n=%zu: %zu wrong\n", name, n, e);
      ++cases, bad += e != 0;
    }
  std::printf("test_updating: %s (%zu cases, %zu failed)\n", bad ? "FAILED" : "ok", cases, bad);
  return bad ? 1 : 0;
}
