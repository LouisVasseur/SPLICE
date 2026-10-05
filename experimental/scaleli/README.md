# SCALE-LI — compression-aware learned-index research kit

**Version 0.1.0 · 7 September 2026**  
Single-threaded, in-memory, ordered `uint64 → uint64` map. Original C++20 implementation; Python experiment tooling.

This is a working experimental starting point, not a production index or a reproduction of NFL, CSV, ALEX, PGM, or LeCo. It tests whether **changing the encoded key layout** helps enough to justify decoding, indirection, and maintenance. Compression does **not** change logical key ranks.

Start with **[the technical specification](docs/TECHNICAL_SPEC.md)** (also supplied as a PDF). The specification separates paper claims, our analysis, implemented mechanisms, and proposed extensions.

## Build and test

Requirements: CMake ≥3.16, GCC or Clang with C++20 and native `__int128`, and Python ≥3.10. The default build has no downloaded dependencies. MSVC is not supported by this version. Linux/GCC was tested; Clang was also checked if listed in `results/example/VALIDATION.md`. macOS/Apple Clang is a target, not a tested platform in this release.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s tests -p test_tools.py -v

./build/scaleli_bench --n 50000 --ops 20000 \
  --distribution dense_sparse --profile read_heavy \
  --policy min_bytes --routing byte --verify 1
```

Each invocation emits one JSON object. Inputs, queries, and updates are prepared before timing. Throughput, individual-operation latency, correctness checking, and software work counters use separate fresh replays of the same trace. Timing includes synchronous compactions triggered by the measured operation. A final maintenance drain is measured separately rather than hidden.

## Run the controlled experiments

```sh
python3 tools/run_suite.py --config configs/smoke.json --output results/my_smoke
python3 tools/summarize.py results/my_smoke/results.jsonl
python3 tools/layout_lab.py --output results/my_layout
python3 tools/virtual_points_lab.py --exact
python3 tools/theory.py

# Learnability sweep: train per-dataset flows (seconds, stdlib), run, summarize, view.
python3 tools/prepare_flows.py --binary build/scaleli_bench --config configs/learnability.json
python3 tools/run_suite.py --binary build/scaleli_bench --config configs/learnability.json --output results/learnability
python3 tools/summarize.py results/learnability/results.jsonl --baseline packed_rank
python3 tools/sweep_report.py results/learnability/summary.csv
```

Single runs: `--flow weights.txt [--flow-bypass 0|1 --flow-min-gain 0.1]` and `--virtual-alpha 0.1`
(see `--help`). `--dump-keys file` writes the loaded key set in SOSD format for `tools/train_flow.py`.
The output JSON gains a `learnability` object (regions using the flow, virtual points, rank SSE before/after,
preprocessing nanoseconds, tail conflict degrees) and a `transform_calls` work counter.

`smoke.json` runs 108 paired cases: two synthetic datasets, three workload mixes, nine variants, two seeds. This is a correctness/sensitivity exercise, not a reliable hardware performance study. `research.json` is a larger synthetic sweep and has validation disabled for scale; run correctness tests before using it. `robustness.json` independently varies bulk-load sampling and held-out insertion order. That is a RoBin-inspired grid, **not** the RoBin benchmark itself.

The nine smoke variants are sorted vector, `std::map`, raw/binary, raw/rank, compressed/binary, compressed/rank, compressed/byte, selective-smoothing/byte, and adaptive-smoothing/byte. The same fixed logical block boundaries are used across the compression controls. Do not compare only compressed/byte against `std::map` and attribute the difference to smoothing.

## What is implemented

| Component | Status |
|---|---|
| Actual packed key blocks: raw, frame-of-reference, restarted delta/ULEB128, integer-linear residual | Implemented; tested |
| Exact `find`, `lower_bound`, ordered `scan(start, limit)`, upsert, delete | Implemented; tested |
| Binary, logical-rank, and encoded-byte routing; exact fence correction | Implemented; tested |
| Sorted per-region update buffers, tombstones, local synchronous rebuilds, splits | Implemented; tested |
| Minimum-byte policy; density-target smoothing heuristic; write-heat hysteresis | Implemented; heuristic, not optimal or latency-calibrated |
| Dataset manifest, validated SOSD-format import, local sampling, download tool | Local paths tested; bulk network fetch not executed successfully here |
| Official ALEX / Dynamic PGM source adapters | Written, opt-in, **not compiled or benchmarked here** |
| NFL-style key transform (official weight format, MKL-free inference, tail-conflict auto-switch) and stdlib trainer | Implemented as a model-feature control; tested; NOT the NFL/AFLI system |
| CSV-style virtual-point smoothing (Algorithm 1, closed-form greedy, per region) | Implemented as a rank-target control; tested; NOT hierarchical CSV |
| LeCo / LA-vector / LINE / LICO reproductions | Not implemented; sources and integration guidance supplied |
| Variable-length payloads, online region merging, SIMD, persistence, concurrency | Not implemented; specified as follow-on work |

Values are a separate uncompressed 8-byte column. The prediction target is the **encoded key arena**, not the address of an entire inline record. Arenas are contiguous **within each region**, not one globally contiguous mutable byte array. No hidden uncompressed full-key copy is retained inside the index.

## Real datasets

```sh
python3 tools/datasets.py list
python3 tools/datasets.py inspect data/dense_sparse_fixture_uint64 --dtype uint64
python3 tools/datasets.py fetch gre_genome --acknowledge-source
./build/scaleli_bench --data data/external/genome \
  --dtype uint64 --format sosd --profile read_heavy --verify 0
```

Check `datasets.py fetch --help` and the manifest for the actual destination and source terms. Downloading is opt-in and uses TLS validation, size caps, header checks, and SHA-256 provenance; SOSD files also carry upstream MD5 checks where available. Sources were verified in public upstream scripts; availability of the multi-gigabyte objects was not verified by a successful download. No real dataset or third-party paper is redistributed.

`--limit N` loads a **prefix**, not a representative sample. The sampling tool offers uniform, strided, and contiguous-window samples and records how local density may have changed. Keys are deduplicated with last-input-value-wins semantics and both source and unique cardinalities are reported. The bundled 16,384-key fixture is synthetic.

## AIDB 2026 hardness protocol lane

`tools/aidb_pipeline.py` applies the protocol of Zhang, Tang and Ailamaki, *How Hard Can Indexing Be? Principled
Dataset Hardness Measurement for Learned Indexes* (AIDB @ VLDB 2026), to **this repository's** clean-room controls
(NFL-style flow transform, CSV-style virtual points) on the ten GRE datasets (`books fb osm covid genome history
libio planet stack wise`, 200M distinct uint64 keys each, catalogued as `gre_<name>` in `configs/datasets.json`).
It is a **reduced-scale, single-host, clean-room control study**: it is not a reproduction of the paper, of NFL/AFLI
or of CSV, and its scores rank hardness metrics against *our* variants only, not the paper's seven indexes.

Scale, stated once so nobody has to infer it: hardness metrics (RMSE, ME, CD via LIPP's FMCD fit, PLA-32, PLA-4096)
are computed on the **full 200M keys** by `scaleli_hardness`; every index run uses a **2M-key uniform sample**
(seed 42) with 1M measured uniform lookups after 200K warm-up, three workload seeds, read-only, `load-ratio 1`,
`miss 0`. The paper bulk loads 200M keys and measures 100M lookups after 20M warm-up on a pinned Xeon core.

Cost of the hardness tool, measured on the real files (Apple M-series, 16 threads): about **3 s and 1.6 GB RSS per
sorted 200M-key file** (books: 3.1 s elapsed, PLA-32 262,604 and PLA-4096 97 segments, equal to the PGM-index
reference), bounded by the inherently sequential PLA-32 pass; `--flow` adds one double per key (another 1.6 GB).
Seven of the ten GRE files are **served unsorted** (`covid genome history libio planet stack wise`; GRE's own
loader sorts and de-duplicates at load time, `src/benchmark/benchmark.h`), so the `sort` step writes a sorted,
de-duplicated copy `data/external/gre/<name>.sorted` (1.6 GB each, 11.2 GB for the seven) with a parallel chunk
sort + merge (`hardness::parallel_sort`, ~3-4 s on 16 threads instead of ~17 s for `std::sort`); every later step
uses that copy and `provenance.json` records both the download's sha256 and, under `sort_audit`, the copy's sha256,
byte length and key count.

```sh
# the ten files must already be under data/external/gre/<name>; the pipeline never downloads
python3 tools/aidb_pipeline.py --wait            # polls until <name>.part files are gone, then runs everything
python3 tools/aidb_pipeline.py --jobs 4          # several scaleli_hardness processes at once (the sweep stays sequential)
python3 tools/aidb_pipeline.py --steps hardness,scores,report --force
python3 tools/aidb_pipeline.py --dry-run         # fixture + three synthetic sets, < 2 minutes, no network, results/aidb_dry/
```

Steps, each idempotent (outputs are kept unless `--force`): `verify-downloads` (header count, byte length, streamed
SHA-256 and mtime of the **download itself** → `provenance.json`), `sort` (`scaleli_hardness --sort-only` audit of
every file; writes and checksums `<name>.sorted` when the file is unsorted or has duplicates, see above), `sample`
(`tools/datasets.py sample --mode uniform --seed 42` on the sorted copy → `data/samples/<name>_2M_uniform_s42` plus
manifest; an existing sample is redrawn when its manifest's source path or source sha256 no longer match, when the
sample settings changed, or when its keys are not strictly ascending), `flows` (`tools/train_flow.py --monotone` on
each sample → `results/aidb/flows/<name>_2D2H2L.txt`, `training_report.json`), `hardness` (full file with the flow,
sample with flow and `--virtual-alpha 0.1`, sample with smoothing only → `hardness.json` in the C3 shape plus the
degeneracy fields `keys`, `duplicates`, `unordered_pairs`, `virtual_points`, `regions`; raw JSON under `hardness/`),
`sweep` (writes `configs/aidb.json`: the eight contracted variants `sorted_vector raw_rank packed_rank packed_rank_flow
packed_rank_flow_forced packed_rank_vp10 packed_rank_flow_vp10 packed_byte`; `--extra-variants fusion` adds the
opt-in `packed_rank_fusion_auto packed_rank_flow_costsel` controls after probing that `scaleli_bench --help` lists
`--fusion`; `tools/run_suite.py` then `tools/summarize.py --baseline packed_rank` → `results/aidb/sweep/`),
`throughput` (median over seeds of `throughput_ops_s/1e6` → `throughput.json`), `scores` (`tools/aidb_scores.py` →
`scores.json`: conformance and coverage of the 25 Table-2 metrics), `report` (`tools/aidb_report.py` →
`results/aidb/report.html`, inline SVG only; `--steps report` alone always regenerates the page).
`results/aidb/pipeline.log` has timestamps for every subprocess; a failing subprocess stops the pipeline with the
step name and a non-zero exit. Lesson 09 of the top-level walkthrough summarizes `throughput.json` and
`scores.json` when they exist and otherwise reports `NOT RUN`.

Idempotency details: JSON outputs are rewritten only when their content changes, so a repeat run touches nothing
but `pipeline.log` and `run.json` (written at the end of a successful run only). Every cached `scaleli_hardness` and
`train_flow.py` output records the sha256 of the executable/script that produced it (`*.command.json`), and
`sweep/identity.json` records the sweep config plus the sha256 of `scaleli_bench`; a different binary or config
makes the step redo (hardness, flows) or refuse until `--force` (sweep). Outputs that predate this fingerprinting
are kept with a WARNING line in the log.

Hardness scopes in `hardness.json`: `full`, `full_flow` (flow applied to all 200M keys, values sorted), `sample`,
`sample_flow`, `sample_csv` (virtual points on raw keys), `sample_flow_csv` (virtual points on transformed keys).
The flow is the stdlib stand-in trainer's 2D2H2L network trained on 4,096 keys of the sample, not the official BNAF.
A flow applied outside its training range saturates and maps many keys to the same double; the `duplicates` field
next to the transformed metrics (and the report's degeneracy table) shows how collapsed that sequence is.

Metric caveats, stated here because the report compares scopes: (1) CD on transformed/smoothed features adds
`1e-6 × mean gap` to U_T instead of LIPP's absolute `1e-6`, which would be a large fraction of U_T on O(1)-range
flow outputs (147 % on the books sample); those CD values are therefore not a literal LIPP fit, and
`fmcd.conflict_degree_lipp_epsilon` in `hardness_details.json` gives the literal-constant value (books sample: 9
vs 13). Raw-key CD uses LIPP's constant verbatim. (2) CD is `floor()` of a double product, as in LIPP's
`PREDICT_POS`, so keys within ~1000 units of a slot boundary are decided by rounding: CD carries ±1 uncertainty
across arithmetics (books 2M sample: 11 here and with LIPP's own evaluation, 10 with exact rationals); treat CD
differences of one as ties. (3) PLA counts widen keys to 128 bits so the PGM sentinel `(x_last + 1, n)` exists for
`x_last = 2^64 - 1`, where PGM's own `make_segmentation<uint64_t>` throws; PLA is translation-invariant in x, so the
count equals PGM's on the same keys shifted down. (4) `tools/aidb_scores.py` normalizes throughput by the
population std (ddof 0); a variant with identical throughput on every scored dataset has std 0, for which eq. (1)
is undefined, and is listed under `skipped` instead of being scored as perfectly conforming; null, string, NaN or
infinite throughput entries remove that dataset from every variant's corpus and are named in `dropped_datasets`.

## Use the index directly

```cpp
#include "scaleli/index.hpp"
#include <cassert>

int main() {
    scaleli::Config cfg;
    cfg.policy = scaleli::Policy::MinBytes;
    cfg.routing = scaleli::Routing::Byte;
    scaleli::Index index(cfg);
    index.bulk_load({{10, 100}, {20, 200}, {30, 300}});
    index.upsert(20, 999);             // false: replaced an existing value
    assert(index.find(20).value() == 999);
    auto rows = index.scan(15, 2);    // (20,999), (30,300)
    assert(rows.size() == 2);
    index.erase(10);
    index.maintain();                 // explicit synchronous debt drain
    index.validate();                // expensive debug oracle/invariant check
}
```

See `docs/METRICS.md`, `integrations/STATUS.md`, and `results/example/VALIDATION.md` before interpreting any timing or space number.

## Run SCALE-LI inside GRE

`integrations/gre/` registers the protocol cells B, N, C and NC, and the joint G+T+V root cells J and Jg0, as GRE
indexes (`scaleli_b` ... `scaleli_jg0`), behind a C++20 facade that builds each cell's Config with scaleli_bench's
own parser. What each cell is: [integrations/gre/README.md](integrations/gre/README.md#gre-cells). How the
`tools/gre_lite.sh` build and the `tools/gre_run.sh` runner take them: [INTEGRATION.md](integrations/gre/INTEGRATION.md).

## Repository map

```text
include/scaleli/   actual index, codecs, exact baselines, workload generation
src/              benchmark driver
integrations/     optional upstream adapters and integration constraints
tests/            randomized differential, codec, endpoint, and tooling tests
tools/            suites, paired summaries, geometry lab, dataset management
configs/          controlled variants, robustness grid, dataset catalog
data/             small synthetic fixture and its provenance
docs/             technical specification, metrics, source registry
results/example/  actual local raw results, summaries, validation logs, figures
```

The included runs show useful negative results: selective smoothing can make normalized byte geometry straighter without providing a better block prediction; compression can reduce memory while losing throughput. The research goal is to identify reproducible operating regions and an adaptive policy—not to assume universal dominance.

Original implementation and documentation are provided under MIT; third-party sources and datasets keep their own licenses. Read [THIRD_PARTY.md](THIRD_PARTY.md) before importing upstream code.
