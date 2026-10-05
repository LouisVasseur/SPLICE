# Reading guide (short): NFL, CSV, the AIDB hardness protocol, and what was run here

This is the condensed guide; the deep guide with equations, worked examples and per-dataset tables is READING_GUIDE.md.

Purpose: everything you need to understand and defend the experiments in the meeting, in reading order.
Paths are relative to `experimental/scaleli/` unless they start with `../`. Line numbers refer to the
files as of 2026-09-21 evening; grep the symbol names if they drift.

Time budget suggestion: papers 60 min (section 1), host index and controls 45 min (sections 2-3),
experiments and metrics 45 min (sections 4-6), results and caveats 30 min (sections 7-8).

---

## 1. The three papers

### 1.1 NFL: Robust Learned Index via Distribution Transformation (Wu et al., PVLDB 15(10), 2022)
Local copy: `../../sources/2205.11807v1.pdf`. Code: github.com/luffy06/NFL (GPL-3, Intel MKL, PyTorch trainer).

Read: section 3.1 (the two-stage framework; batching is a design decision stated here; 3.1.1 defines
conflict degree and tail conflict degree), 3.2 (the numerical flow: 3.2.1 feature expansion and Algorithm
3.1, 3.2.2 the switching mechanism and Table 2), 3.3 (AFLI: 3.3.1 structure, 3.3.2 operations and Algorithm
3.2; there is no section 3.4), 4.1 (setup; 4.1.3 gives the 2-input, 2-hidden, 2-layer B-NAF, the normal
target with variance 10^16 and batch size 256), 4.3 (tail latency: P99 is the 99th-percentile batch latency
divided by the batch size), Table 1 (ALEX with and without the flow), Table 2 (transform cost per key by
batch size).

What it claims: a numerical normalizing flow z = f(key), trained by maximum likelihood, makes the key
distribution near-uniform; an index designed for near-uniform keys (AFLI: model nodes, small buckets and
dense nodes sized by the measured conflict degree) then needs fewer levels and fewer collisions. A
"tail conflict degree" (Definition 3.2: the 99th-percentile per-position collision count under one linear
model) is computed on the raw and on the transformed keys at bulk load; the flow is switched off for the
whole dataset only if the transformed tail conflict degree is larger (3.2.2; YCSB, AMZN and WIKI end up
without the flow, 4.2). NFL processes requests in batches (3.1, justified by batching being common in
databases; batch size 256 in 4.1.3). Table 2 shows why this matters for the transform: the 2H2L flow costs
169.5 ns per key at batch 1 and 8.4 ns per key at batch 256.

What you should know that the paper does not stress: the shipped flow weights are tiny (input dimension
2, hidden 2, two layers, no bias: eight weights; features [x, x - floor(x)] with x = (key - min)/var);
the transform can be non-monotone, so NFL sorts by transformed key and looks up transformed keys;
all timing is measured with batched requests (batch 256): throughput is operations per second over
batched processing and P99 is the 99th-percentile batch latency divided by the batch size, so neither is
a per-singleton lookup measurement.

### 1.2 Learned Indexes with Distribution Smoothing via Virtual Points (Amarasinghe, Choudhury, Qi, Bailey, EDBT 2025; arXiv 2408.06134)
Local copy: `../../sources/2408.06134v3.pdf`. No public implementation exists (checked September 2026).

Read: section 3 (problem statement, equation 2, NP-hardness sketch), section 4 (single-model smoothing:
equation 4 with refit, candidate filtering by the sign of the loss derivative, Algorithm 1), section 5
(hierarchical CSV: Algorithm 2 and the cost condition, equation 22), section 6.1 (setup: ALEX, LIPP,
SALI hosts; alpha from 0.05 to 0.8), Figure 7 and Table 4 (query-time improvement for promoted keys, up to
34%; preprocessing seconds at alpha 0.1: 889 s fb, 1,423 covid, 2,297 osm, 2,902 genome on ALEX).

What it claims: insert "virtual points" (slots that hold no record) into a key set so that one linear
model, refitted, has lower squared error over real plus virtual points; budget lambda = alpha * n; greedy
insertion into the gaps between consecutive keys; in a hierarchy, smooth each parent-plus-subtree key set
bottom-up and replace the subtree by one new leaf when the acceptance test passes: for ALEX, equation 22
(search constant x expected searches + traversal constant x level) below a threshold c < 0; for LIPP and
SALI, which have no in-node search, simply a lower loss.

### 1.3 How Hard Can Indexing Be? Principled Dataset Hardness Measurement for Learned Indexes (Zhang, Tang, Ailamaki, AIDB@VLDB 2026)
Local copy: `~/Downloads/aidb26_11.pdf`.

Read: section 2.2 (the GRE metric and why it fails), section 3.2 (conformance and coverage, equations 1-3),
section 4.1 (the five scalar metrics, the six indexes, the ten datasets in Table 1, the workload), section
4.2 and Table 2 (results), Findings 1-7, section 5 (open problems).

Definitions you will be asked about:
- RMSE and ME: root mean square and maximum error of one least-squares line key -> rank over the whole set.
- CD (conflict degree): the largest number of keys mapped to one slot by LIPP's FMCD linear fit.
- PLA-32 and PLA-4096: number of segments of the optimal epsilon-bounded piecewise-linear approximation
  (O'Rourke 1981, as PGM-index computes it); GRE = the pair (PLA-32, PLA-4096).
- Harder-than: S_i is harder than S_j if every dimension is >= and at least one is >; comparable pairs C,
  incomparable U; coverage Cov = (|C| - |U|) / (|C| + |U|), with 45 pairs for ten datasets.
- Conformance: throughput normalized by its standard deviation over datasets; importance w = sigmoid of
  the signed gap; reward R over conforming pairs, penalty P over violating pairs; Conf_I = (R - P)/(R + P),
  Conf = mean over indexes.
- Workload: bulk load all 200M keys, then uniform random lookups drawn from the dataset's own keys (the
  paper's wording: "random lookups for all keys in the dataset"); 20M warm-up lookups, then 100M measured;
  a single pinned thread.

Their findings to keep in mind: conformance rises and coverage falls with dimensionality (F1); no metric is
near the frontier (F2); PLA-32 conforms for indexes with local correction (PGM, ALEX, FINEdex) and CD for
LIPP (F3); RMSE, ME and PLA-4096 conform well with none of the six indexes, and no scalar metric fits RMI
or XIndex (F4); two dimensions are the sweet spot (F5); adding an aligned scalar (CD or PLA-32) is what
buys conformance, adding misaligned ones does not (F6); the coverage cost of an added dimension is similar
whichever dimension is added, so it is not the deciding factor (F7).

---

## 2. The host index (what the experiments run inside)

`../../README.md` and `../../docs/APPROACHES.md` describe the workspace. The index is the SCALE-LI
experimental map: an ordered uint64 -> uint64 map, header-only C++20, `include/scaleli/`.

- `include/scaleli/index.hpp`: `struct Config` (line 43) holds every knob; `class Region` (line 87) is
  a run of `region_keys` (default 4096) keys split into blocks of `block_keys` (128) with packed codecs;
  `class Index` (line 327) owns the regions. Bulk load: `Index::bulk_load` (line 429); updates go to a
  per-region delta and are merged by `Index::compact` (line 402).
- Lookup path: `Index::locate_region` (line 348) finds the region (binary search over region fences, or
  the learned root of section 3.4); `Region::locate_block` (line 133) predicts a block from the region's
  linear model and corrects it exactly with an exponential search then a binary search over block fences;
  the block is then decoded and searched. Predictions are hints; correctness never depends on them.
- `include/scaleli/model.hpp`: `LinearModel` (line 11), `fit_xy` (line 26) fits arbitrary targets (ranks or slot
  ranks for the rank model, byte positions for the byte model) against an arbitrary double feature;
  `predict_x` (line 19).
- Benchmark driver: `src/benchmark.cpp`. Four passes per run: throughput (line 63; the timed loop; the
  "learnability" block is read from this index), per-operation latency (line 84; skipped with
  `--latency 0`), differential validation against std::map (line 98; `--verify 1`), software work
  counters on a fresh replay (line 111; `--instrument 1`). Output is one JSON object per run.
- Work counters (`include/scaleli/types.hpp`, `QueryStats`): `root_probes` (fence comparisons at the
  root), `coordinate_probes` (comparisons in the predicted-block binary search), `fence_probes`
  (comparisons in the exact block correction), `correction_distance` (blocks between predicted and true),
  `transform_calls` (flow evaluations). These are deterministic for a given index and trace.

Reproducibility features you can cite: identical query traces across variants (`trace_fingerprint` and
`result_checksum` in every JSON; `tools/run_suite.py` runs every variant on the same (dataset, profile,
seed) trace and `tools/summarize.py` pairs rows by (dataset, profile, seed, repeat) and rejects mismatched
fingerprints), a differential validation pass (available; switched off in the sweeps, see section 5), and
`tools/summarize.py` (paired geometric-mean speedups with a seed bootstrap).

---

## 3. What was implemented for this study (all clean-room)

### 3.1 NFL-style transform
- `include/scaleli/transform.hpp`: `FlowTransform::load` (line 35) reads the NFL text weight format;
  `transform` (line 64) evaluates the 2-input, 2-hidden tanh network with the [x, x - floor(x)] encoding
  and the sum decoder; `tail_conflict_degree` (line 90) is the paper's metric (one linear fit into
  1.5 n slots, 99th percentile of collisions, minus one).
- `tools/train_flow.py`: a stdlib stand-in trainer for the same eight-weight shape: `features` (line 25),
  `loss_and_grad` (line 48: 1-D change-of-variables negative log-likelihood under a standard normal,
  analytic gradients, a barrier keeping dz/dx > 0), `train` (line 87: Adam), `--monotone` zeroes the
  fractional-feature weights and their gradients. Not the official BNAF trainer.
- How the transform is used: as the model FEATURE. `Region::feature` (index.hpp line 106) returns the
  flow value when the region accepted the flow, else the normalized key; storage order stays the original
  key order, so scans and fences are untouched and a non-monotone flow only costs probes.
- NFL bypass per region: in `Region::rebuild` (line 199), keep the flow iff the tail conflict degree of
  the transformed features is at least `flow_min_gain` (10%) below that of the raw features
  (`--flow-bypass 1`, `--flow-min-gain 0.1`). The rule applies under `--fusion manual` (the default); the
  decision is made at bulk load and reused on compaction.

### 3.2 CSV-style virtual points
- `include/scaleli/smoothing.hpp`: `smooth_cdf` (line 47) is Algorithm 1 for one region: candidates in
  the gaps between consecutive features, closed-form OLS SSE from running sums (S_yy - S_xy^2/S_xx),
  ternary search inside a gap when the loss is decreasing at the left end and increasing at the right,
  early stop when nothing improves; budget alpha * n. Returns slot ranks for the real keys and the virtual
  feature values.
- Use in the index: virtual points move the rank TARGETS to slot ranks; each block records `slot_begin`;
  `Region::predicted_block` (line 118) compares predictions against slot boundaries. Zero key bytes;
  0.1 doubles per key of metadata at alpha 0.1. Compactions re-place the previous virtual points among
  the new keys instead of re-searching (`relearn_on_compaction=false`), because re-running the search on
  every compaction made the 10%-insert runs of the learnability sweep 116-232x slower
  (`results/learnability/summary.csv`, variant `packed_rank_vp10_relearn`).

### 3.3 The region-level selector (the "fusion")
`Region::rebuild`, the `Fusion::Auto` branch (line 239): candidates {raw, flow} x {ranks, smoothed
slots}; each candidate's cost is the mean number of probes the real `locate_block` spends on the region's
own keys, plus `flow_cost` (4 probe-equivalents) if the candidate uses the flow; keep the cheapest.
Options `--fusion auto --flow-cost 4`.

### 3.4 The learned root (root-level fusion)
`Index::fit_root` (line 362): one global linear model over the regions' first keys (line 370; an earlier
version fitted through region 0's sentinel fence at key 0, which is the artifact described in section 7);
candidates {raw, flow} x {ranks, virtual fences}; virtual fences come from `smooth_cdf` on the fence
features with budget `root_alpha` x regions and a slot -> region table for O(1) mapping; each candidate
is scored by the probes of the real root locate on fences and midpoints (+ flow_cost); kept only if it
beats the fence binary search. Options `--root model --root-alpha 4`.

### 3.5 Hardness tool (the paper's metrics)
`src/hardness.cpp` (CLI) and `include/scaleli/hardness.hpp`: `least_squares` (line 84; RMSE, ME),
`fmcd_fit` (line 168) and `conflict_degree` (line 218; a port of LIPP's bulk-load FMCD, MIT, that keeps
64-bit key differences exact in anchored form, so CD can differ by +-1 from LIPP's double arithmetic; for
transformed or smoothed double features it scales LIPP's absolute 1e-6 epsilon by the mean gap and emits
the literal-epsilon CD next to it; raw-key CD uses LIPP's constant verbatim), `pla_segments` (line 295;
PGM-index's optimal segmentation, Apache-2.0, re-typed with exact __int128 arithmetic for uint64 keys),
`compute_metrics` (line 323). Verified: PLA counts identical to PGM-index's `make_segmentation` (verbatim
header, commit c6fcf3d) for eps 0-4096 on the fixture, six synthetic sets, 5M-key prefixes of all ten GRE
files and the full 200M-key books and wise files (`results/aidb/verification/pla/`); RMSE/ME/CD against
independent Python oracles (`results/aidb/verification/metrics/vlib.py`).
Also `--sort-only 1 --write-sorted` (seven of the ten GRE files are served unsorted).

### 3.6 Scores, pipeline, reports
- `tools/aidb_scores.py`: equations 1-3 (`harder` line 82, `classify_pairs` line 87, `coverage` line 100,
  `conformance_of_variant` line 105); verified to 4e-16 against an independent implementation.
- `tools/aidb_pipeline.py`: steps `verify-downloads`, `sort`, `sample`, `flows`, `hardness`, `sweep`,
  `throughput`, `scores`, `report` (methods `step_*`, lines 346-562); idempotent; `--dry-run` uses the
  bundled fixture (`evidence/dense_sparse_fixture_uint64`) plus four synthetic key sets, one shuffled with
  duplicates.
- `tools/aidb_report.py` (HTML report per results directory), `tools/fusion_report.py` (H1-H3 verdicts
  and the hardness-move panel), `results/aidb/root_analysis.py` (root ablation), `results/aidb/
  make_figures.py` (the charts, slide and page renderers), `tools/summarize.py` (paired speedups).

Tests: `tests/test_main.cpp` (differential tests for every option, including flow, virtual points,
selector, learned root and parallel build), `tests/test_hardness.cpp`, `tests/test_tools.py`,
`tests/test_aidb_scores.py`, `tests/test_aidb_report.py`, `tests/test_aidb_pipeline.py` (46 Python tests).

---

## 4. Data and preparation

- Ten datasets of the paper, from the GRE mirror `https://www.cse.cuhk.edu.hk/mlsys/gre/<name>`
  (1,600,000,008 bytes each: 8-byte count + 200M uint64). Downloaded 2026-09-21; sha256 per file in
  `results/aidb/provenance.json`. `data/external/gre/`.
- Seven files (covid, genome, history, libio, planet, stack, wise) are stored unsorted; `data/external/
  gre/<name>.sorted` is the sorted copy (no duplicates in any file). Everything downstream uses it.
- Samples: `data/samples/<name>_2M_uniform_s42` (uniform random 2M of the 200M keys, seed 42) and
  `<name>_2M_window_s42` (one contiguous 2M-key window at a seeded offset). Uniform keeps the global
  shape but flattens local structure (the mean per-region tail conflict degree drops to 3-8 on eight of the
  ten samples; fb 8.5, osm 36); windows keep local structure (1-60) but lose the global shape. Both modes
  are run.
- Flows: one weight file per sample, `results/aidb/flows/<name>_2D2H2L.txt` and `results/aidb_window/
  flows/`, trained by `tools/train_flow.py --monotone` on 4,096 sampled keys, 200 steps (about 2.4 s each).

---

## 5. The experiments, in order

E1. Hardness on every dataset (full 200M keys and both samples), with the transformed keys and the
CSV-augmented samples as extra "scopes": `results/aidb/hardness.json`, `results/aidb_window/
hardness.json`, raw runs under `hardness/`. Scopes: full, full_flow, sample, sample_flow, sample_csv,
sample_flow_csv.

E2. Region-level sweep, uniform samples (`results/aidb/sweep/`): 10 datasets x 10 variants x 3 query
seeds, read-only, 1M lookups after 200k warm-up, 2M keys loaded; variants (see section 6.2). Repeated on
window samples (`results/aidb_window/sweep/`). 300 paired runs each.

E3. Final sweep (`results/aidb_final/sweep/`): 20 samples x 9 variants x 3 seeds (11, 29, 47), 5M lookups
after 500k warm-up, performance-core QoS, `build-final` binary. Adds the root ablation (raw root, flow root,
virtual fences, all candidates) and vp + root. Run twice: the first pass 19:16-20:11, then the five root
variants rerun 20:46-21:17 after the sentinel-fence fix with the binary rebuilt at 20:45 (sha256
c63ceab0...; `sweep/environment.json` still records the pre-fix binary 837daca7... because `--resume` keeps
the first record). The 307 pre-fix rows (300 root-variant rows plus 7 strays from earlier seed-61 and
fusion_auto attempts) are kept in `results.stale-root-before-fix.jsonl`.

E4. Granularity ablation (`results/aidb_granularity/`): regions of 4096, 16384, 32768 keys on five
datasets (fb, osm, planet, covid, genome), 15 variants x 2 seeds, all 150 runs complete (75 cells, two seeds
each; `summary_r4096.csv`, `summary_r16384.csv`, `summary_r32768.csv`). The selector chose the flow in 0 of
6,740 regions across the three sizes (4,890 + 1,230 + 620) and virtual points in 99-100%; forcing the flow
leaves fence probes equal to the control's within 0.05 probes at every size. The virtual-point saving
grows with region size because a single line fits a larger region worse: fence probes fall 9-24% at
4,096 keys, 9-36% at 16,384 and 7-39% at 32,768 (osm is always the smallest saving; e.g. fb 3.69 -> 2.25 at
32,768). Its `packed_rank_flow_vp10_r*` variants force the flow (`--flow-bypass 0`).

E5. Full scale (`results/aidb_fullscale/`): fb and planet at 200M keys, single seed 11, 2M measured lookups
after 200k warm-up, QoS on, `build-root` binary. Variants: the packed control, root raw, root flow, and
virtual-fence roots at two budgets: `--root-alpha 0.04` (about 1,950 virtual fences, the absolute count the
2M runs spent at alpha 4) on both datasets, and alpha 4 on fb. On planet, alpha 4 means 195k greedy rounds
over 48,828 fences (Algorithm 1 is O(budget x fences) per feature, single-threaded at the root) and did not
finish within 2 hours; those three planet runs are queued last with a 4-hour cap. Results of the 13 runs
done (single seed, so no intervals; `results/aidb_fullscale/root_analysis.txt`): both files load into 48,829
regions, so the fence binary search costs 15.66 root probes. fb: raw-key root 10.81; raw key + virtual
fences 3.42 with 973 fences (the greedy stopped at 973 whether the budget was 1,953 or 195k, so the two
budgets coincide on fb); total probes 25.8 -> 13.6 (-47%); the flow feature is never chosen (estimate 21.0
vs 10.8); region virtual points cut fence probes 5.14 -> 4.74 (-8%). planet: the raw-key root estimates 25.8
probes and the 1,950-fence root 25.2, both worse than binary, so both fall back; region virtual points cut
fence probes 3.99 -> 3.42 (-14%). Build: 6-7 s control, 10 s with root fences, 161-190 s with region
virtual points (smoothing at 200M keys on 16 threads). Metadata 0.57 -> 1.37 B/key with region virtual
points. Throughput: the identical fb fences structure measured three times gave 0.65, 0.72 and 0.81 Mops
against a 0.65 Mops control, a 25% spread at one seed, so no 200M-key time claim is made. planet at budget 4 x
regions (the first of the three queued runs, finished 00:59): the greedy spends the whole budget, 195,316
virtual fences after 2,068 s of single-threaded smoothing, for root probes 15.66 -> 13.69 (-13%; total 24.7
-> 22.7); one global line over planet's 48,829 fences is not smoothable at this budget, unlike fb.

Passes in E2-E5: every sweep ran the benchmark with `--verify 0 --latency 0 --instrument 1`, i.e. the
throughput pass and the counter pass only. The differential-validation and latency passes were not
executed in these runs; they are exercised by `tests/test_main.cpp` and by E7's tooling check.

E6. Conformance and coverage of the 25 metrics against our variants (`results/aidb/scores.json`,
`results/aidb_window/scores.json`, and the HTML reports).

E7. Verification (`results/aidb/verification/verdicts.json` and the verifiers' scripts): independent
recomputation of the ablation numbers, the cost decomposition, the "synergy" claim, an overclaim review
of the notes, and a clean build/test/dry-run of the tooling.

Equivalent commands (from `experimental/scaleli/`). The pipelines were actually run in step subsets and
resumed after interruptions; the scripts that ran are `results/aidb/run_chain.sh`, `run_chain2.sh`,
`run_chain3.sh`, `run_sweeps*.sh` and `prep_after_sort.sh`, with logs next to them.
```
# E1, E2, E6 on the uniform samples; ran in stages: --steps verify-downloads,sort ;
# --steps sample,flows,hardness --jobs 4 ; --steps sweep,throughput,scores,report --build-threads 16 --extra-variants fusion
python3 tools/aidb_pipeline.py --binary-dir ../../build-fusion/experimental/scaleli --jobs 4 --build-threads 16 --extra-variants fusion
# the same on the window samples
python3 tools/aidb_pipeline.py --binary-dir ../../build-fusion/experimental/scaleli --sample-mode window --results results/aidb_window --jobs 4 --build-threads 16 --extra-variants fusion
# E3 (run twice: the first pass, then the root variants again after the sentinel-fence fix)
python3 tools/run_suite.py --binary ../../build-final/experimental/scaleli/scaleli_bench --config results/aidb_final/config.json --output results/aidb_final/sweep --timeout 3600 --resume
python3 tools/summarize.py results/aidb_final/sweep/results.jsonl --baseline packed_rank
python3 results/aidb/root_analysis.py results/aidb_final/sweep/results.jsonl --json results/aidb_final/root_analysis.json
# E4
python3 tools/run_suite.py --binary ../../build-fusion/experimental/scaleli/scaleli_bench --config results/aidb_granularity/config.json --output results/aidb_granularity/sweep --timeout 3600 --resume
for r in 4096 16384 32768; do python3 tools/summarize.py results/aidb_granularity/sweep/results.jsonl --baseline packed_rank_r$r --output results/aidb_granularity/summary_r$r.csv; done
# E5 (config_a: both datasets, alpha 0.04 roots; config_b: fb at alpha 4; config_c1-c3: planet at alpha 4, 4 h cap)
python3 tools/run_suite.py --binary ../../build-root/experimental/scaleli/scaleli_bench --config results/aidb_fullscale/config_a.json --output results/aidb_fullscale/sweep --timeout 3600 --resume
python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl --json results/aidb_fullscale/root_analysis.json
# charts
python3 results/aidb/make_figures.py
```

---

## 6. Metrics and variants dictionary

### 6.1 Metrics
- fence probes per lookup: `work_counters.fence_probes / operations`; the exact correction work after the
  region model's prediction. Deterministic. This is what virtual points reduce.
- root probes per lookup: comparisons spent finding the region. 8.96 for binary search over 489 fences.
- total probes: root + fence + coordinate probes.
- correction distance: blocks between predicted and true block, per lookup.
- rank SSE before/after: OLS squared error of the region model on ranks vs on slots.
- tail conflict degree (raw, flow): NFL's metric per region (section 3.1).
- choices: per region, which candidate the selector chose (none, flow, vp, both).
- root_probes_binary/raw/flow/vp_raw/vp_flow: the root selector's candidate estimates (probes on fences
  and midpoints; flow candidates include the 4-probe charge).
- throughput_ops_s and paired speedup: lookups per second; ratio to the packed control on the same
  (dataset, seed) trace, geometric mean over seeds; the bracket is the 3-seed bootstrap = seed spread.
- hardness metrics: RMSE, ME, CD, PLA-32, PLA-4096 (section 1.3); conformance and coverage per metric.
- preprocessing: `smoothing_ns`, `transform_ns` (summed thread time), `build_ns` (wall clock).

### 6.2 Variants (name -> options)
- sorted_vector: plain binary search over the uncompressed array (no index).
- raw_rank: uncompressed blocks, rank routing.
- packed_rank: the control: packed codecs, per-region linear model on ranks, fence binary-search root.
- packed_rank_flow: + `--flow <weights> --flow-bypass 1` (NFL bypass decides per region).
- packed_rank_flow_forced: `--flow-bypass 0` (flow in every region).
- packed_rank_vp10: `--virtual-alpha 0.1`.
- packed_rank_flow_vp10: both, with the NFL bypass on (`--flow-bypass 1`); the E4 `packed_rank_flow_vp10_r*`
  variants force the flow (`--flow-bypass 0`).
- packed_rank_fusion_auto: `--fusion auto --flow ... --virtual-alpha 0.1` (per-region selector).
- packed_rank_flow_costsel: `--fusion auto` with the flow only (cost-based flow choice).
- packed_byte: byte routing (the repository's compression thesis; not part of this story).
- packed_rank_root_raw / _root_flow / _root_vf4 / _root_fusion: `--root model` with the raw key /
  with a flow file / with `--root-alpha 4` / with both (selector over all four candidates).
- packed_rank_vp10_root_fusion: region virtual points plus the root selector.
- packed_rank_root_vf004 / packed_rank_vp10_root_vf004 (E5 only): virtual-fence root at `--root-alpha 0.04`
  (about 1,950 fences at 200M keys), without / with region virtual points.

---

## 7. Results, with where each number lives

Verified numbers are in `results/aidb/MEETING_NOTES.md`; the charts are in the deck and in
`results/aidb/figures.html`.

1. Hardness agrees with the paper: history PLA-32/PLA-4096 = 105,468/468 and libio 145,808/639 match the
   values the paper prints; the coverage column equals Table 2 for every CD-free metric; CD-containing
   metrics differ by one of 45 pairs (paper prints no CD values).
2. Orthogonal axes: on full data the transform changes RMSE by -98% (fb) to +111% (stack) and PLA-32 by at
   most 5 segments; on the 2M samples virtual points cut PLA-32 43-61% on six datasets and raise RMSE 6-10%.
3. Virtual points alone: fence probes fall on all ten datasets in both sampling modes, by 5-24% (uniform:
   osm -8.6% to covid -24.0%; window: stack -4.7% to covid -20.8%; 17 of the 20 samples at -10% or more;
   e.g. fb 2.60 -> 2.18, osm 5.08 -> 4.64); correction distance -48 to -83% on nine uniform samples (osm
   -15%) and -47 to -86% on seven windows (osm -26%, planet -20%, fb -15%); throughput 0.92-1.06, mostly
   within noise; up to 0.8 B/key of metadata (0.74-0.80 on 18 of 20 samples, 0.68 libio_window, 0.25
   stack_window).
4. Transform alone: see the retraining study of 2026-09-22 (MEETING_NOTES section 1b, results/aidb_flowv2).
   The flows used below were inert (tail conflict degree unchanged on 9/10 datasets), so their null result is
   about our trainer. Retrained without the monotone constraint the transform reaches NFL's target of about 4
   on every dataset (osm 99 -> 6) but becomes non-monotone and then raises fence probes 9-192%; retrained
   with the constraint it changes nothing. With the inert flows: accepted in 0-9% of regions; forced into
   every region (one transform call per lookup), probe counts equal to the control's to three decimals;
   selector picks it in 0% of regions; at the root, never chosen once the root is fitted on real keys.
5. Region-level fusion: the selector picks virtual points in 96-100% of regions and "both" essentially
   never: no synergy at region scale.
6. Root: raw key + virtual fences cuts root probes 8.96 -> 2.1-2.8 on 16/20 samples (total probes -35 to
   -43% on those 16; -30%, -31% and -22% on osm_window, planet_window and planet_uniform; osm_uniform stays
   at 18.3); the first-pass "flow + fences synergy" was the sentinel-fence artifact and is withdrawn; the
   time effect is not resolvable: the fences root measures 0.99 pooled (0.89-1.08 per sample) while two
   measurements of the identical structure differ by up to 8% (10% for the binary-fallback root).
7. Cost: one flow evaluation on the lookup path costs 66 ns [56, 77] (about 16 probe-equivalents); it can
   save at most ~7 root probes (about 28 ns); NFL's batching (8 ns/key at batch 256) is the only regime in
   which the transform could pay here.
8. Binary search over the uncompressed array is 1.15-2.29x faster than the packed control at 2M keys:
   decoding and block work dominate lookups, which is why probe savings barely show in time.
9. Granularity (E4, complete): at 4,096 / 16,384 / 32,768 keys per region the selector picks the transform
   in 0 of 6,740 regions and a forced transform stays within 0.05 fence probes of the control; the
   virtual-point saving grows with region size (fb 2.60 -> 2.18 at 4,096, 3.69 -> 2.25 at 32,768; covid
   -24% -> -39%; osm stays at -7 to -9%). Chart: deck slide "results-granularity" and figures.html.
10. Full scale (E5): fb root 15.7 -> 3.4 probes with 973 virtual fences (total -47%); planet's root falls
   back at a 1,950-fence budget; region virtual points -8% (fb) and -14% (planet) fence probes.

---

## 8. Caveats and likely questions

- "Is this NFL / CSV?" No: clean-room controls inside our map. NFL's AFLI and its BNAF trainer are not
  run (GPL-3 code, MKL); CSV has no code, so ours is written from the paper and checked against an
  exhaustive toy oracle. The AIDB paper's six indexes are not run.
- "Why 2M-key samples?" The index runs at 2M for time; hardness is computed on the full 200M keys; 14 of
  the 16 full-scale runs are done (section 5, E5): at 200M keys fb's root goes from 15.7 to 3.4 probes with
  973 virtual fences; planet's root cannot be learned within a 1,950-fence budget and gains only 13% with
  195,316 fences (34 min of smoothing); two planet runs at budget 4 are still running
  (`results/aidb/run_chain3.log` has the live status). Uniform and window samples answer different questions and disagree on
  local hardness (fb window PLA-32 10,599 vs uniform 3,034).
- "How reliable is the timing?" Two measurements of an identical structure differ by up to 8% (10% for
  the binary-fallback root) on this laptop (performance/efficiency cores, shared host); per-dataset time
  effects below that are not resolvable. That is why mechanism claims use counters.
- "Why did the transform fail?" At region scale a per-region line already absorbs global curvature; at
  the root, once fitted on real keys, the raw key is already good and virtual fences finish the job; and
  the unbatched evaluation costs more than the probes it saves. It is a cost failure, not a prediction
  failure.
- "What would change the conclusion?" Batched lookups (NFL's regime), a higher-capacity or per-node
  transform, or a host where routing rather than decoding dominates.
- "What was verified and how?" Section 5, E7: five independent agents recomputed the ablation, the cost
  decomposition and the synergy claim from the raw run files and reviewed the notes; their reports and
  scripts are archived.
- "How expensive is CSV's preprocessing?" Algorithm 1 as written is O(budget x n) per model (each greedy
  round rescans every gap after the refit). Per region (4,096 keys, alpha 0.1) that is 12-32 us of thread
  time per key; at the root with alpha 4 it is 1956 rounds over 489 fences at 2M keys (0.6 s) but 195k
  rounds over 48,828 fences at 200M keys, which did not finish in 2 hours (section 5, E5). The CSV paper
  itself reports 889-2,902 s of preprocessing at alpha 0.1 on 200M keys.
- Open implementation question: one CD ordering differs from the paper's (paper prints no CD values).
