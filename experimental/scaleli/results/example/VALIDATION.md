# Executed validation — 7 September 2026

This file records completed local execution, not planned CI or upstream results.

| Check | Result |
|---|---|
| GCC Release core and example CTest | Passed, 2/2 |
| Core assertion count | 762,890; all passed |
| Randomized differential configurations | 45; all passed |
| Python test methods (with parameterized subcases) | 12; all passed |
| AddressSanitizer + UndefinedBehaviorSanitizer Debug core/example CTest | Passed, 2/2 |
| Clang Release core/example CTest | Passed, 2/2 |
| Paired smoke benchmark jobs | 108/108, each differentially verified |
| Cross-variant matching | All matched trace fingerprints and result checksums |
| Layout invariance lab | Same keys, regions, ranks and blocks verified across policies |
| Dataset import/sampling | Synthetic uint32/uint64 fixtures passed; real bulk files not downloaded |
| Upstream ALEX/PGM bridge | Not compiled or benchmarked |
| Remote CI | Workflow provided, not executed by a remote service here |

Benchmark binary SHA-256: `6065d209ba168e01d4e3d95e5eec77cbec6adfac90400654462117c8c1f1f884`.

The SHA-256 matches the binary recorded by the supplied 108-job suite: **true**.

GCC and Clang ran on Linux x86-64. macOS, ARM64, MSVC and ISA-specialized codec paths were not tested. Source normal-distribution sampling may vary across standard libraries. The compiler warning in the Clang log is a conversion warning in a test diagnostic, not a failed test.

All timings are exploratory on a shared host. No CPU isolation, NUMA binding, fixed governor, LLC-size sweep or real-corpus performance claim is made. Two seeds are not enough for publication-level uncertainty estimates. Per-operation latency and throughput come from different fresh passes, and host scheduling noise can make their relative order disagree.

Raw output: `core_tests.txt`, `python_tests.txt`, `release_ctest.log`, `sanitize_tests.log`, `clang_tests.log`, `suite/results.jsonl`, `suite/environment.json`, `suite/summary.csv`, `layout/summary.json`. The separate `smoke.json` and `initial_layout.csv` are an earlier single shift-workload pilot, not the paired suite.
