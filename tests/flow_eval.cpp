// SPDX-License-Identifier: MIT
// Helper of tests/test_flow.sh: evaluates gret::MolFlow (include/gret/flow.h) on decimal keys.
//
//   flow_eval FLOW < keys        one key per line in, 'u T' per key out (u with %.17g)
//   flow_eval --check FLOW       exit 0 if FLOW loads, else print the error and exit 3 (ASan and UBSan exit with 1)
//   flow_eval --selftest         exp_small against std::exp on [-40, 40], u on extreme keys, the default flow
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include "gret/flow.h"

using gret::MolFlow;

static int selftest() {
  double worst = 0;
  for (long i = -4000000; i <= 4000000; ++i) {  // step 1e-5, plus the edges
    const double t = i * 1e-5, e = std::exp(t), r = std::fabs(MolFlow::exp_small(t) - e) / e;
    if (r > worst) worst = r;
  }
  std::printf("exp_small: max relative error %.3g on [-40, 40]\n", worst);
  int bad = worst < 2e-15 ? 0 : 1;
  const MolFlow f = MolFlow::parse("molflow 1 3 1000 2000\n0.5 0.25 1e-9\n0.25 0.5 0.1\n0.25 2 5e-300\n");
  const uint64_t ks[] = {0, 1, 999, 1000, 1500, 2000, 2001, uint64_t(1) << 53, UINT64_MAX - 1, UINT64_MAX};
  uint64_t prev = 0;
  for (uint64_t k : ks) {
    const double u = f.u(k);
    const uint64_t T = f(k);
    if (!std::isfinite(u) || u < 0 || u > 1 + 1e-15 || T > gret::kOutMax || T < prev) bad = 1;
    std::printf("k=%llu x=%.17g u=%.17g T=%llu\n", (unsigned long long)k, f.x_of(k), u, (unsigned long long)T);
    prev = T;
  }
  const MolFlow d;
  if (d(0) != 0 || d(UINT64_MAX) != 0 || d.cost_terms() != 0) bad = 1;
  std::printf("selftest: %s\n", bad ? "FAIL" : "ok");
  return bad;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--selftest") == 0) return selftest();
  if (argc == 3 && std::strcmp(argv[1], "--check") == 0) {
    try {
      const MolFlow f = MolFlow::load(argv[2]);
      std::printf("ok M=%zu lo=%llu hi=%llu bytes=%zu\n", f.M(), (unsigned long long)f.lo(),
                  (unsigned long long)f.hi(), f.bytes());
      return 0;
    } catch (const std::exception& e) {
      std::printf("rejected: %s\n", e.what());
      return 3;
    }
  }
  if (argc != 2) {
    std::fprintf(stderr, "usage: flow_eval FLOW < keys | flow_eval --check FLOW | flow_eval --selftest\n");
    return 2;
  }
  const MolFlow f = MolFlow::load(argv[1]);
  char line[64];
  while (std::fgets(line, sizeof line, stdin)) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long k = std::strtoull(line, &end, 10);
    if (end == line || errno == ERANGE) {
      std::fprintf(stderr, "flow_eval: bad key line '%s'\n", line);
      return 2;
    }
    std::printf("%.17g %llu\n", f.u(k), (unsigned long long)f(k));
  }
  return 0;
}
