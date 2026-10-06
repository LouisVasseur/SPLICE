# External benchmark integration status

## ALEX and Dynamic PGM: adapters supplied, not tested against upstream binaries

`external/bridge.cpp` is built only with `-DSCALELI_EXTERNAL=ON`. It uses PIMPL declarations from `include/scaleli/external.hpp`. The upstream bridge is a separate C++17 translation unit while the experimental engine uses C++20. This avoids forcing an older upstream header through the engine's language mode.

```sh
python3 tools/fetch_external.py
# Review repositories and licenses; retain external.lock.json.
cmake -S . -B build-external -DCMAKE_BUILD_TYPE=Release -DSCALELI_EXTERNAL=ON
cmake --build build-external -j2
./build-external/scaleli_bench --index alex --n 10000 --ops 5000 --profile churn --verify 1
./build-external/scaleli_bench --index pgm --n 10000 --ops 5000 --profile churn --verify 1
python3 tools/run_suite.py --binary build-external/scaleli_bench \
  --config configs/external.json --output results/upstream
```

These commands are an integration procedure, not a record of successful execution in the supplied environment. Compiler or API fixes may be needed for the revision you fetch.

### Semantics and accounting that must not be hidden

ALEX's `insert` is not upsert. The wrapper checks for an existing payload and overwrites it explicitly. Scan iterators return real key/value pairs. Construction uses sorted, unique input.

Dynamic PGM can use a sentinel payload value for deletion when the payload is arithmetic. The adapter wraps the value in a small struct, selecting the explicit-deletion-flag representation instead. This preserves the full uint64 value domain but changes the representation overhead relative to a scalar-only benchmark. The upstream maximum-key sentinel is still excluded in this adapter: insertion of UINT64_MAX is rejected; find/scan there are empty. Comparisons with this adapter must use the common restricted key domain, while core SCALE-LI still tests both uint64 endpoints.

PGM's upstream `size_in_bytes()` is not a measurement of all reserved capacity or allocator overhead. Its memory output is labeled estimated. ALEX's native model/data size accounting also has its own scope. Do not publish cross-index footprint ratios until a common capacity/allocator or isolated-RSS protocol is implemented.

## Other systems: integration targets, no local adapter advertised

| Target | Why it belongs | Required work before comparison |
|---|---|---|
| GRE | Dynamic ordered-index workload framework | Wrapper and C++20 facade in `gre/` (cells B, N, C, NC, J, Jg0; `gre/INTEGRATION.md`); facade-built index checked equal to scaleli_bench's on 1M prefixes; not yet wired into `tools/gre_lite.sh` |
| GRE: SPLICE-H (`gre_splice/`) | The read-only SPLICE index under GRE's protocol (names `splice`, `splice_thp`; chunk timers `trace_splice`, `trace_lipp`; harness baseline `nullindex`) | Wrapper, C++20 build facade and C++17 inline `get()` in `gre_splice/`, wired into `tools/gre_lite.sh` and `tools/gre_run.sh` (per-dataset plans from `splice_count`); checked locally on an x86-64 build under Rosetta (counts and correctness only). Timed runs: `SERVER.md` section 5b. `scaleli_splice` is SCALE-LI's Jg0 cell, not this index |
| RoBin | Uniform/prefix sampling, bulk-load size, sorted/shuffled insertion stress | Use its upstream scripts for a real reproduction; compare the local inspired grid separately |
| SOSD | Static sorted-key lookup benchmark and data provenance | Match equality/duplicate result contract; avoid equating it with a dynamic map benchmark |
| LIPP / SALI / LINE | Structural conflicts, adaptivity, maintenance and concurrency | Fetch official revisions; add explicit iterator/update capability checks |
| NFL / CSV | Transformation and virtual-point smoothing controls | Clean-room controls now run inside the core map (`transform.hpp`, `smoothing.hpp`, lesson 08). Still required for a real comparison: the official NFL binary (GPL-3, Intel MKL, AFLI) with its batched semantics, and a CSV host (ALEX/LIPP/SALI) since no CSV code is public |
| LA-vector / Elias–Fano | Compressed ordered-integer access/rank baselines | Static mode first; count key data plus directory and payload representation |
| LeCo / LICO | Learned residual and SIMD codec baselines | Port codec interface, not just published speedup; LICO's postings workload is not this map contract |
| SEA 2025 suite / SIMD S-tree | Strong traditional and compressed static baselines | Match hardware ISA, compiler and cache policy; include original uncompressed array costs |

The default test suite does not silently substitute home-grown algorithms under these published names.
