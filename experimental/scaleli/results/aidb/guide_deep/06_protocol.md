# 06 — The exact experimental protocol

**What this section gives you.** Everything that was actually run, precisely enough to rerun it: the benchmark binary option by option (how keys are loaded, how the query trace is generated and fingerprinted, what the four passes time and check, what every JSON field means), the two Python drivers that turn a config into paired runs and a summary (`tools/run_suite.py`, `tools/summarize.py`), the nine-step AIDB pipeline (`tools/aidb_pipeline.py`) with the exact parameters it fixed, the machine and the three build directories, the measured noise floor, one card per experiment E1–E7 (config, variants with the option string each expands to, seeds, dates, durations, row counts, caveats), a dictionary of every variant name, the reproduction commands in order, and the questions the supervisor is likely to ask. Every number is cited to a PDF, a source file or a result file; anything I could not confirm carries an `[unverified: …]` tag. All our components are clean-room CONTROLS inside the SCALE-LI experimental map, never reproductions of NFL, AFLI, CSV or the AIDB benchmark [aidb_pipeline.py:14-15]. Paths: S = `/Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli`; the repo root is two levels up.

Notation (shared with the other sections): keys k₁ < … < kₙ (uint64), rank r(kᵢ) = i − 1; region = 4,096 keys, block = 128 keys; root = the whole-range structure over region fences; probes = key comparisons counted by the software counters; λ = α·n is the virtual-point budget; D₉₉ the tail conflict degree.

---

## 1. The benchmark binary: `src/benchmark.cpp` (222 lines)

### 1.1 What one invocation does

One call of `scaleli_bench` = load keys → build a deterministic workload (initial set + operation trace) → run up to four separate passes on freshly built indexes → print ONE JSON object on stdout [benchmark.cpp:123-152]. The binary never writes result files itself; `tools/run_suite.py` captures stdout into `results.jsonl` (section 2).

### 1.2 Every command-line option (allow-list at benchmark.cpp:27; unknown options abort with `unknown option:`)

Options are `--name value` or `--name=value`; a bare `--name` means `1` [benchmark.cpp:25-26]. Flags accept only `0`/`1` [benchmark.cpp:33]. Defaults are those in the parsing code (benchmark.cpp:189-208), not the help text (which lists the same values).

| option | default | meaning (where read) |
|---|---|---|
| `--help` | off | prints the option summary and exits [benchmark.cpp:164-188] |
| `--index` | `scaleli` | `scaleli` (our region/block index), `sorted_vector` (plain binary search over a sorted array), `ordered_map` (std::map wrapper); `alex`/`pgm` only if built with `SCALELI_EXTERNAL` [benchmark.cpp:213-219] |
| `--n` | 50000 | number of synthetic keys when no `--data` is given [benchmark.cpp:208] |
| `--distribution` | `dense_sparse` | synthetic generator: linear, uniform, lognormal, dense_sparse, staircase, locally_hard, clustered, near_u64, duplicates [workload.hpp:14-32] |
| `--seed` | 42 | seed of the `std::mt19937_64` that drives BOTH the synthetic generator and the workload trace [workload.hpp:15, workload.hpp:77] |
| `--ops` | 20000 | number of operations in the trace [benchmark.cpp:203] |
| `--profile` | `read_only` | operation mix: read_only (read = 1), read_heavy (.8/.1/.05/.02/.03 read/insert/update/erase/scan), scan_heavy (.45/.05/scan .5), write_heavy (.2/.5/.2/.1), churn (.2/.3/.2/.3), append (.5/.5, insert_mode append), shift (.4/.4/.1/scan .1, insert_mode shift, moving_hotspot queries) [benchmark.cpp:194-201] |
| `--read --insert --update --erase --scan` | from profile | override the ratios; must be ≥ 0 and sum to 1 within 1e-8 [workload.hpp:61] |
| `--policy` | `min_bytes` | block codec policy: raw, min_bytes, smooth, adaptive, forced [benchmark.cpp:189] |
| `--codec` | `for` | codec used when policy = forced: raw, for, delta, linear [benchmark.cpp:189] |
| `--routing` | `byte` | how a key is located inside a region: binary (fence binary search), rank (linear rank model + exponential search on fences), byte [benchmark.cpp:189] |
| `--region-keys` | 4096 | keys per region [benchmark.cpp:189]; regions are cut every `region_keys` rows at bulk load [index.hpp:432] |
| `--block-keys` | 128 | keys per block [benchmark.cpp:189] |
| `--delta-limit` | 64 | per-region delta buffer size before compaction [benchmark.cpp:189] |
| `--restart` | 16 | restart interval of the delta codec [benchmark.cpp:190] |
| `--min-saving` | 0.05 | minimum fraction of bytes a codec must save to be chosen under min_bytes [benchmark.cpp:191] |
| `--smooth-scale`, `--hot-enter`, `--hot-exit`, `--heat-decay` | 1, .25, .05, .99 | adaptive-policy heat parameters [benchmark.cpp:193]; not used by any AIDB run |
| `--data` | (none) | path of a key file; when present `--n/--distribution` are ignored [benchmark.cpp:207-208] |
| `--format` | `sosd` | `sosd` = 8-byte little-endian count header then keys; `raw` = keys only [workload.hpp:37-48] |
| `--dtype` | `uint64` | `uint64` or `uint32` (4-byte keys are widened) [benchmark.cpp:207] |
| `--limit` | 0 | 0 = whole file; N > 0 = the first N keys of the file, a PREFIX, and the JSON then says `"dataset_scope":"prefix"` [workload.hpp:45, benchmark.cpp:125] |
| `--load-ratio` | 0.75 | fraction of the unique keys bulk-loaded; the rest are held out for inserts [workload.hpp:79-82] |
| `--bulk-sampling` | `uniform` | `uniform` = shuffle the unique keys with the seeded RNG before taking the loaded fraction; `prefix` = take the first fraction in key order [workload.hpp:78] |
| `--insert-order` | `shuffled` | order of held-out keys for inserts [workload.hpp:83-84] |
| `--insert-mode` | `random` | random, append, hotspot, shift [workload.hpp:119-127] |
| `--query-distribution` | `uniform` | uniform, hotspot (80% of reads in the first 10% of live keys), moving_hotspot (hot set moves to the last 10% after ops/2), zipf [workload.hpp:98-106] |
| `--zipf-theta` | 0.99 | Zipf exponent [workload.hpp:94-97] |
| `--miss` | 0.1 | probability that a read is turned into a read-miss on an absent key [workload.hpp:129] |
| `--scan-length` | 100 | rows returned per scan [benchmark.cpp:203] |
| `--flow` | (none) | NFL-format weight file; loads a `FlowTransform` and sets `cfg.flow` [benchmark.cpp:192] |
| `--flow-bypass` | 1 | 1 = per-region NFL-style auto-switch (keep the flow only if it lowers D₉₉ by ≥ `flow_min_gain`); 0 = force the flow into every region [index.hpp:281-282] |
| `--flow-min-gain` | 0.1 | relative D₉₉ reduction required by the bypass rule [index.hpp:281] |
| `--virtual-alpha` | 0 | CSV-style virtual-point budget per region, λ = α·(keys in region); 0 = off [benchmark.cpp:191] |
| `--relearn` | 0 | 1 = rerun the flow decision and smoothing on every compaction; 0 = reuse bulk-load decisions [benchmark.cpp:191] |
| `--fusion` | `manual` | `auto` = per-region cost-based choice among {none, flow, vp, both} scored by the real `locate_block` on the region's own keys [index.hpp:239-272] |
| `--flow-cost` | 4 | probe-equivalents charged per lookup to any candidate that uses the flow [index.hpp:264] |
| `--root` | `binary` | `binary` = fence binary search; `model` = one global linear model over region fences with candidates {raw, flow} × {ranks, virtual fences}, kept only if its estimated probes beat binary search [index.hpp:348-372] |
| `--root-alpha` | 0 | with `--root model`: virtual-fence budget = alpha × regions (alpha < 64) [benchmark.cpp:191, help text] |
| `--build-threads` | 1 | regions are built concurrently; `build_ns` becomes the wall clock of the parallel build; queries are always single-threaded [help text; benchmark.cpp:191] |
| `--qos` | 1 | macOS only: 1 = call `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` at the start of `main` so the process prefers performance cores; 0 = default class. Not a pinning guarantee [benchmark.cpp:158-163] |
| `--verify` | 1 | pass 3: differential validation against `OrderedMap` (std::map) [benchmark.cpp:98-110] |
| `--instrument` | 1 | pass 4: software counters on a fresh replay [benchmark.cpp:111-115] |
| `--latency` | 1 | pass 2: per-operation timings on a fresh replay; 0 skips it and emits every latency block with `count 0` [benchmark.cpp:84-97] |
| `--warmup` | 4096 | warm-up lookups before the timed loop in every pass [benchmark.cpp:44-48, benchmark.cpp:59] |
| `--dump-layout`, `--dump-trace`, `--dump-keys` | (none) | write the region/block layout CSV, the trace CSV (`operation,key,value,length`), or the loaded keys as a SOSD file [benchmark.cpp:116-120, 210-212] |

There is no `--sort` option in the benchmark: sortedness of a data file is handled by `canonicalize()` (stable sort + de-duplicate, later value wins) applied to every loaded input [types.hpp:38-44, workload.hpp:75]. The sorted on-disk copies of the seven unsorted GRE files come from `scaleli_hardness --sort-only 1 --write-sorted` in the pipeline (section 3).

### 1.3 How keys are loaded

`read_dataset(path, format, width, limit)` [workload.hpp:37-48]: checks that `(bytes − header) % width == 0`, reads the 8-byte count when `format == sosd`, checks that the count equals `(bytes − header)/width`, applies `--limit` as `min(count, limit)`, then reads every key little-endian and pairs it with the value `mix64(i)` (i = file position) [workload.hpp:46-47]. `make_workload` then sets `source_rows = input.size()`, canonicalizes (sort + dedupe) and sets `unique_rows` [workload.hpp:75]. For the GRE files `source_rows = unique_rows = 200,000,000` (no duplicates: provenance `sort_audit.duplicates = 0` for all ten [results/aidb/provenance.json]); for the samples both are 2,000,000 [results/aidb/sweep/results.jsonl, any row].

### 1.4 How the query trace is generated (workload.hpp:74-137)

```
rng = mt19937_64(seed)                                 # workload.hpp:77
if bulk_sampling == uniform: shuffle(input, rng)       # consumes RNG state even at load_ratio 1
load_n = max(1, floor(|input| * load_ratio))           # load_ratio 1 -> load_n = n
initial = canonicalize(input[0:load_n])                # sorted again, so the loaded set is sorted
heldout = keys input[load_n:]                          # empty at load_ratio 1
dynamic = insert+update+erase > 0                      # false for read_only
for op in 0..ops-1:
    p = unit(rng)                                      # U(0,1)
    if p < read:                                       # read_only: always
        key = sample_live(op)                          # uniform: i = rng() % n; key = initial[i]
        if unit(rng) < miss: kind = ReadMiss; key = absent(key)   # miss 0 -> never
        else kind = ReadHit
    ... (insert / update / erase / scan branches, unused in the AIDB runs)
```

So for the AIDB sweeps (`--profile read_only --load-ratio 1 --miss 0 --query-distribution uniform`) the trace is exactly `ops` uniformly random ReadHit lookups of loaded keys, drawn by `rng() % n` [workload.hpp:104] after two `unit(rng)` draws per operation. The trace depends only on (the key set, `--seed`, `--ops`, the profile/ratios) and NOT on any index option, which is what makes runs across variants paired. `absent()` produces near misses (`near+1+attempt` for 64 attempts, then random) [workload.hpp:108-113]; unused here because miss = 0.

**Trace fingerprint** [benchmark.cpp:54]: `h = 0; for each initial record r: h = mix64(h ^ digest_record(r)); for each op o: h = mix64(h ^ mix64(o.key) ^ mix64(o.value) ^ mix64(unsigned(o.kind)) ^ mix64(o.length))`, with `mix64` the splitmix64 finalizer [workload.hpp:11] and `digest_record(r) = mix64(r.first) ^ mix64(r.second + 17)` [workload.hpp:12]. It is printed as a decimal string (`"trace_fingerprint"`). Two runs with the same fingerprint executed the same operations on the same loaded records. Example: every fb_uniform run at seed 11 in the final sweep has fingerprint `12214875262882230774` [results/aidb_final/sweep/results.jsonl].

**Result checksum** [benchmark.cpp:80]: over the throughput pass, `throughput_digest = mix64(throughput_digest ^ r.checksum)` where for a read `r.checksum = mix64(*value) ^ mix64(key)` when found, 0 otherwise [benchmark.cpp:36-38]. Since values are `mix64(file position)`, a different index that returns a wrong value or misses a key produces a different checksum. Every pass recomputes it and the binary aborts if they differ ("fresh replays produced different results", "instrumentation changes results") [benchmark.cpp:96, 115]. Example: fb_uniform seed 11 → `17216331355546010123` for all nine variants [results/aidb_final/sweep/results.jsonl; verified by the recompute verifier: "one result_checksum across their variants (0 mismatches)", results/aidb/verification/verdicts.json entry 11].

**Warm-up** [benchmark.cpp:44-48]: `warmup(ix, w, n)` performs `min(n, |initial|)` lookups of `initial[(j·997) mod |initial|]` for j = 0, 1, …: a deterministic stride-997 walk over the loaded keys, not the trace, and not random. Its XOR of values goes into a `volatile` sink so it cannot be optimized away. `warmup_reads` in the JSON is `min(warm, |initial|)`.

### 1.5 The four passes (each on a freshly built index) [benchmark.cpp:58-115]

| pass | enabled by | what happens | what is measured |
|---|---|---|---|
| 1 throughput | always | `make()` → `bulk_load(initial)` timed → learnability block read → `memory()` → warm-up → timed loop over the whole trace, no per-op timer, no counters → `memory()`, `maintenance()`, `size()` → `maintain()` (drain) timed | `build_ns`, `elapsed_ns`, `throughput_ops_s = ops·10⁹/elapsed_ns`, `result_checksum`, `memory_before/after/after_drain`, `maintenance` |
| 2 latency | `--latency 1` | fresh index, warm-up, then `Clock::now()` around EVERY operation; digest must equal pass 1 | `latency_ns` per op kind (p50/p95/p99/max), `phase_latency_ns` (first/second half of the trace), `compaction_operation_latency_ns` |
| 3 verify | `--verify 1` | fresh index and a `std::map` reference; every op executed on both; any mismatch throws `differential mismatch at operation j`; then `validate()` and a full-range scan comparison | `verified: true` (or abort) |
| 4 instrument | `--instrument 1` | fresh index, warm-up, replay with a `QueryStats*` so every probe is counted; digest must equal pass 1 | `work_counters` |

All AIDB sweeps ran with `--verify 0 --latency 0 --instrument 1` [results/*/sweep/environment.json config.common], i.e. two bulk loads per run (pass 1 and pass 4), no per-operation timer in the timed pass, no differential check inside the sweeps. Differential verification with `--verify 1` was run separately on synthetic data by the final verifier (T7: `--n 20000 --distribution lognormal --ops 5000 --root model --root-alpha 4 --fusion auto --virtual-alpha 0.1 --verify 1` → `verified true`) [verdicts.json entry 15] and by the unit tests (46 Python tests + 4 ctest binaries) [verdicts.json entry 15, T3/T4]. Consequence of two bulk loads: for planet at full scale with root smoothing, `build_ns = 2,067.8 s` but the job took ≈ 70 min of wall clock (23:49:40 → 00:59:37 local) because pass 4 rebuilds the same index [results/aidb_fullscale/sweep/results.jsonl planet/packed_rank_root_vf4; results/aidb/run_chain3.log].

`clock_pair_p50_ns` is the median of 10,000 back-to-back `Clock::now()` pairs measured after the passes: 0 or 41 ns on this machine [benchmark.cpp:121; results/aidb_final/sweep/results.jsonl], i.e. the timer overhead reported (never subtracted).

### 1.6 The JSON output: every field (benchmark.cpp:123-152)

Top level (`schema_version` 1): `index`, `distribution`, `profile`, `dataset_path`, `dataset_scope` (`prefix` when `--limit` > 0 else `full_input`), `seed`, `source_rows`, `unique_rows`, `initial_rows` (loaded), `final_rows` (after the trace), `operations`, `trace_fingerprint`, `result_checksum`, `verified`, `instrumented`, `latency_pass`, `software_counters_supported` (true only for `--index scaleli`), `policy`, `routing`, `forced_codec`, `region_keys`, `block_keys`, `delta_limit`, `restart_interval`, `min_saving_fraction`, `smooth_scale`, `flow_weights` (path or ""), `flow_bypass`, `flow_min_gain`, `virtual_alpha`, `relearn_on_compaction`, `fusion`, `flow_cost`, `build_threads`, `root`, `root_alpha` (the last two only in binaries that have the root option; the build-fusion rows lack them), `learnability` (block, or `null` for non-scaleli indexes), `query_distribution`, `insert_mode`, `bulk_sampling`, `insert_order`, `load_ratio`, `scan_length`, `requested_miss_ratio`, `warmup_reads`, `build_ns`, `elapsed_ns`, `throughput_ops_s`, `scan_records_returned`, `scan_records_per_total_second`, `clock_pair_p50_ns`, `memory_before`, `memory_after`, `memory_after_drain`, `maintenance`, `latency_ns`, `phase_latency_ns`, `compaction_operation_latency_ns`, `work_counters`. `run_suite.py` appends `dataset`, `variant`, `repeat`, `command` [run_suite.py:69].

`memory_*` [benchmark.cpp:53; types.hpp:29-37]: `key_bytes` (encoded key arenas), `value_bytes` (8 per key), `metadata_bytes` (fences, block descriptors, models, virtual features, root tables), `delta_bytes`, `reserved_slack_bytes`, `accounted_bytes` = the sum, `estimated` (false: counted allocations, "excludes malloc overhead and process RSS"). Example (fb_uniform, packed_rank_vp10, seed 11): key 5,636,561 + value 16,000,000 + metadata 2,738,824 = 24,375,385 B = 12.19 B/key [results/aidb_final/sweep/results.jsonl].

`maintenance`: `compactions`, `splits`, `bytes_rewritten`, `max_rewrite_bytes`, `raw_bypasses`, `drain_ns`, `drain_additional_rewrite_bytes` [benchmark.cpp:144-146]; all zero in read-only runs except `drain_ns` (a few µs).

`work_counters` [types.hpp:17-24; increment sites]: `root_probes` (fence comparisons while locating the region: binary loop index.hpp:353 or the model's exponential search index.hpp:350), `coordinate_probes` (comparisons inside a block coordinate search, index.hpp:124), `fence_probes` (block-fence comparisons `blocks[i].last < k` in the region's locate, index.hpp:135), `delta_probes` (delta-buffer binary search, index.hpp:115), `key_at_calls` and `decoded_keys` (codec key accesses/decodes, codec.hpp:134-181), `codec_bytes_examined`, `block_routes` (one per block prediction, index.hpp:151), `correction_distance` (Σ |predicted block − actual block|, index.hpp:151) and `max_correction_distance`, `blocks_decoded_for_scan`, `transform_calls` (flow evaluations on the lookup path, transform.hpp:65). Per-operation values are these divided by `operations`; e.g. binary root: `root_probes/ops = 8.96` at 2M keys and 15.66 at 200M keys (section 8, Q5).

`learnability` [benchmark.cpp:66-76; index.hpp:472-493], read from the pass-1 index right after bulk load: `regions`, `flow_regions` (regions whose model uses the flow feature), `virtual_points` (Σ over regions), `keys`, `rank_sse_before/after` (Σ over regions of SSE of the linear fit before/after smoothing), `smoothing_ns` and `transform_ns` (Σ over regions of thread time, so with 16 build threads they exceed wall clock), `tail_conflicts_raw_mean`, `tail_conflicts_flow_mean` (mean D₉₉ over regions), `flow_bytes`, `choices {none, flow, vp, both}` (fusion-auto selections per region), `cost_none_mean`, `cost_selected_mean` (auto mode only), and the root block: `root_model` (true iff a learned root is in use), `root_flow` (root uses the flow feature), `root_vp` (root uses virtual fences), `root_virtual` (number of virtual fences), `root_probes_binary/raw/flow/vp_raw/vp_flow` (the selector's expected probes per lookup for each candidate, measured by the real root locate on the fences and fence midpoints; 0 for candidates not evaluated).

---

## 2. `tools/run_suite.py` and `tools/summarize.py`

### 2.1 run_suite.py (78 lines): config schema and execution

Config JSON keys [run_suite.py:39-58]:

- `datasets`: list of strings (then `{"name": s, "distribution": s}`) or objects `{"name", "data", "format", "dtype", "flow_weights", "flow_train_args", …}`; every key except `name`, `flow_weights`, `flow_train_args` becomes a `--key value` option [run_suite.py:53].
- `profiles`: list of profile names → `--profile`.
- `variants`: list of `{"name", option: value, …}`; every key except `name` becomes an option [run_suite.py:54]; booleans are printed as `1`/`0` [run_suite.py:61].
- `seeds` (default `[42]`) → `--seed`; `repeats` (default 1, or `--repeats`); `common`: options prepended to every job; `schedule_seed` (default 137).
- `"$flow"` substitution [run_suite.py:55-58]: a variant with `"flow": "$flow"` gets the dataset's `flow_weights` path resolved to an absolute path; missing file → `RuntimeError` telling you to run `tools/prepare_flows.py`.

Jobs = the cartesian product datasets × profiles × variants × seeds × repeats, shuffled by `random.Random(schedule_seed).shuffle` [run_suite.py:48-49], so variants are interleaved in time (a slow-down of the host affects all variants, not one). Each job runs `subprocess.run(command, timeout=--timeout)` (default 300 s; the AIDB runs passed 3600, 7200 or 14400 s); a non-zero exit writes `failed_command.json` and aborts [run_suite.py:62-66]. With `--resume`, jobs whose `(dataset, profile, variant, seed, repeat)` already appear in `results.jsonl` are skipped and new rows are appended [run_suite.py:36-42]; without it an existing `results.jsonl` is refused. `--cpu` pins with `taskset` on Linux only (never used here).

`environment.json` [run_suite.py:20-27]: `timestamp_utc`, `platform`, `machine`, `python`, `cpu` (output of `lscpu`, which is `"unavailable"` on macOS), `compiler` (`c++ --version`), `affinity` (null on macOS), `binary_sha256`, `git_commit` (`HEAD` here: the workspace has no commits), `warning` ("Shared-host exploratory results; CPU governor, NUMA binding and isolation not controlled by this script."), `config` (the whole config), `requested_cpu`. It is written once and NOT rewritten on `--resume` [run_suite.py:44]; this matters for E3 (section 5).

`results.jsonl`: one JSON object per job = the benchmark's JSON plus `dataset`, `variant`, `repeat`, `command` (the full argv) [run_suite.py:68-69].

### 2.2 summarize.py (75 lines): pairing, ratio, geometric mean, bootstrap, columns

- Pairing key: `(dataset, profile, seed, repeat)`; the script first asserts that every such group has exactly ONE `trace_fingerprint` across variants and raises `unmatched trace fingerprints across variants` otherwise [summarize.py:28-31].
- Ratio: for every run of a variant, `throughput_ops_s / throughput_ops_s of the baseline run with the same (dataset, profile, seed, repeat)` [summarize.py:38-40]. Baseline: `--baseline` (default `raw_rank`; the pipeline and all AIDB summaries used `packed_rank`, except the granularity summaries, which used `packed_rank_r4096/r16384/r32768`) [aidb_pipeline.py:531; run_chain.sh].
- Estimate: per seed, the mean of `log(ratio)` over repeats; then `exp(mean over seeds)` = the geometric mean of per-seed ratios [summarize.py:10-15].
- Bootstrap: `random.Random(42)`, 2000 resamples of the per-seed log-means WITH replacement (`rng.choices(logs, k=len(logs))`), `exp(mean)` of each; interval = the 2.5% and 97.5% quantiles, with `quantile(xs, q) = sorted(xs)[min(len−1, max(0, ceil(q·len) − 1))]` [summarize.py:8, 16-19]. With fewer than 2 seeds the bounds are NaN. With k = 3 seeds there are only 3³ = 27 equally likely resamples, so the 2.5% quantile is the smallest resample mean, i.e. the smallest per-seed ratio, and the 97.5% quantile is the largest: the interval IS the per-seed range, not a 95% confidence interval. Worked example (E3, fb_uniform, `packed_rank_root_vf4` vs `packed_rank`): per-seed ratios 1.0330, 1.0214, 1.1116 → geometric mean 1.0546, bootstrap [1.0214, 1.1116] [computed by results/aidb/guide_deep/scratch/protocol_examples.py from results/aidb_final/sweep/results.jsonl].
- Columns of `summary.csv` [summarize.py:41-59], all medians over the variant's runs unless stated: `dataset, profile, variant, runs, all_verified` (all rows had `verified`; false in every AIDB sweep because `--verify 0`), `throughput_ops_s_median`, `read_hit_p99_ns_median`, `insert_p99_ns_median`, `scan_p99_ns_median` (0 here: no latency pass), `initial_bytes_per_key` = accounted_bytes/initial_rows, `final_bytes_per_key`, `key_arena_ratio` = key_bytes/(8·initial_rows), `rewrite_bytes_per_operation`, `decode_work_per_operation` = decoded_keys/ops, `fence_probes_per_operation`, `root_probes_per_operation`, `root_model_fraction` (median of 1/0 per run of `learnability.root_model`), `correction_distance_per_operation`, `build_ns_per_key`, `flow_region_fraction` = flow_regions/regions, `virtual_points_per_key` = virtual_points/keys, `rank_sse_ratio` = rank_sse_after/rank_sse_before (1.0 when before = 0), `preprocess_ns_per_key` = (smoothing_ns + transform_ns)/keys, `fusion_both_fraction` = choices.both/regions, `fusion_flow_fraction` = (flow + both)/regions, `fusion_vp_fraction` = (vp + both)/regions, `est_probes_none`, `est_probes_selected` (auto-mode cost means), `baseline`, `paired_throughput_speedup`, `seed_bootstrap_low`, `seed_bootstrap_high`.

---

## 3. `tools/aidb_pipeline.py` (649 lines): the nine steps and their fixed parameters

Steps, in the enforced order [aidb_pipeline.py:35]: `verify-downloads, sort, sample, flows, hardness, sweep, throughput, scores, report`. Each checks its outputs first (`--force` redoes). Defaults [aidb_pipeline.py:600-640]: `--results results/aidb`, `--data-dir data/external/gre`, `--samples-dir data/samples`, `--sample 2000000`, `--sample-seed 42`, `--sample-mode uniform` (choices uniform, window, strided), `--ops 1000000`, `--warmup 200000`, `--seeds 11,29,47`, `--alpha 0.1`, `--region-keys 4096`, `--flow-sample 4096`, `--flow-steps 200`, `--binary-dir <repo>/build/experimental/scaleli`, `--jobs 1`, `--timeout 14400` s per subprocess, `--extra-variants` (only group `fusion`), `--build-threads` (omitted from the config unless given), `--wait/--poll-seconds 60`, `--dry-run`, `--steps`.

1. **verify-downloads** [aidb_pipeline.py:325-346]: the pipeline never downloads. For each dataset it requires `<data-dir>/<name>` with size `8 + 8·count`, compares the header count to the catalog count (200,000,000 for every GRE entry in configs/datasets.json), computes sha256 of the DOWNLOAD (never of the sorted copy) and writes `provenance.json[name] = {url, source, path, format, dtype, count, bytes, sha256, mtime_ns, retrieved_utc, verified_utc, role}`. Ran 15:17:17–15:17:27 UTC on 2026-09-21 (10.2 s) [results/aidb/pipeline.log].
2. **sort** [aidb_pipeline.py:352-381]: `scaleli_hardness --data <raw> --dtype uint64 --check-sorted 0 --sort-only 1 --write-sorted <name>.sorted`; the audit JSON (`sorted`, `duplicates`, `written`, `written_keys`, `sort_ns`, `threads`) goes to `results/aidb/sort/<name>.json` and into `provenance[name].sort_audit`, plus the sorted copy's own sha256/bytes/mtime. Result: books, fb, osm sorted with 0 duplicates; covid, genome, history, libio, planet, stack, wise unsorted with 0 duplicates → seven `.sorted` copies of 1,600,000,008 bytes each, ≈ 21 s per file, step total 151.5 s [results/aidb/pipeline.log 15:17:27–15:19:58 UTC]. Every later step uses `dataset_path(name)` = the `.sorted` copy when it exists, else the download [aidb_pipeline.py:278-283].
3. **sample** [aidb_pipeline.py:391-407]: `python3 tools/datasets.py sample <src> <dest> --n 2000000 --dtype uint64 --mode <mode> --seed 42`, dest = `data/samples/<name>_2M_<mode>_s42`, plus a `.manifest.json` (`source, source_count, source_sha256, mode, seed, sample_count, dtype, sample_sha256, warning`). Sampling rule [datasets.py:73-89]: `rng = random.Random(seed)`; `uniform`: `indices = sorted(rng.sample(range(total), n))` (2M of the 200M positions without replacement, kept in file order, hence sorted because the source is sorted); `window`: `start = rng.randrange(total − n + 1)` and `indices = range(start, start + n)`; `strided`: `(2i+1)·total/(2n)`. Because seed and total are the same for all ten datasets, every window sample is the SAME rank interval [171,644,825, 173,644,825) of its file (recomputed: `random.Random(42).randrange(198000001) = 171644825`; checked on disk: fb window first key 66,478,271,979 = fb[171,644,825], last 67,251,951,970 = fb[173,644,824]; planet 5,721,028,025 … 5,986,820,672 [guide_deep/scratch/protocol_examples.py; data/external/gre/fb, planet.sorted]). A sample is redrawn when the manifest's source path or sha256 no longer matches provenance, when size/mode/seed changed, or when the keys are not strictly ascending [aidb_pipeline.py:382-390]. Uniform samples were drawn 15:20:03–15:20:39 UTC (books/fb/osm earlier at 17:09 local from the sorted originals; the other seven after the sort step), window samples 16:11:17–16:11:32 UTC [pipeline.log of both result dirs].
4. **flows** [aidb_pipeline.py:409-421]: `python3 tools/train_flow.py <sample> --dtype uint64 --output results/<lane>/flows/<name>_2D2H2L.txt --sample 4096 --steps 200 --monotone` (exact command recorded in `<name>_training.json.command.json`). train_flow defaults that were NOT overridden: `--shifts 64`, `--lr 0.05`, `--barrier 1e-3`, `--seed 1000000007` [train_flow.py:60-64]. `--sample 4096` = 4,096 training keys uniformly sampled from the 2M-key sample; `--monotone` zeroes the fractional-feature weights (a stated deviation from NFL) [train_flow.py:65]. Output report per dataset: `keys, training_keys, steps, seed, shifts, monotone, best_nll, train_seconds, unordered_transformed_pairs, tail_conflict_degree_raw/transformed`; e.g. fb uniform: nll 4.1820, 2.39 s, D₉₉ 8 → 8 [results/aidb/flows/fb_training.json]. Whole step 71.5 s (uniform), 69.2 s (window). The flow is trained on the same keys it is later applied to (in-domain), which prepare_flows.py states must be reported [prepare_flows.py:7-9].
5. **hardness** [aidb_pipeline.py:423-503]: three `scaleli_hardness` jobs per dataset, common options `--dtype uint64 --region-keys 4096 --pla-eps 32,4096 --check-sorted 1`: (a) `full`: `--data <full file> --flow <flow>` → blocks `original` and `transformed`; (b) `sample_flow`: `--data <sample> --flow <flow> --virtual-alpha 0.1` → `original`, `transformed`, `smoothed` (smoothing applied to the TRANSFORMED features); (c) `sample_csv`: `--data <sample> --virtual-alpha 0.1` → `original`, `smoothed` (raw features). These map to the six scopes of `hardness.json`: `full`, `full_flow`, `sample`, `sample_flow`, `sample_flow_csv`, `sample_csv`, each with `rmse, max_error, conflict_degree, pla_32, pla_4096` (+ `keys`, and `duplicates/unordered_pairs` for flow scopes, `virtual_points/regions` for csv scopes) [aidb_pipeline.py:470-475]. `--jobs 4` ran four processes concurrently in the actual runs [prep_after_sort.sh]. Uniform lane: 132.3 s for all 30 jobs (15:21:50–15:24:03 UTC); window lane 105.1 s (the full-file jobs were reused from cache) [pipeline.log]. Details and timings in `hardness_details.json` (e.g. fb full 17.37 s elapsed incl. 0.81 s flow transform; fb sample_csv smoothing 11.7 s for 199,707 virtual points over 489 regions) [results/aidb/hardness_details.json].
6. **sweep** [aidb_pipeline.py:505-531]: writes the generated config to `configs/aidb.json` AND `results/<lane>/sweep/config.json` (byte-identical, checked with `cmp`), plus `identity.json` = {config, scaleli_bench sha256, path}. Config [aidb_pipeline.py:488-496]: `common = {load-ratio 1, miss 0, query-distribution uniform, ops, warmup, verify 0, instrument 1, latency 0, region-keys 4096, [build-threads]}`; `datasets = [{name, data: data/samples/<name>_2M_<mode>_s42, format sosd, dtype uint64, flow_weights, flow_train_args ["--monotone"]}]`; `profiles ["read_only"]`; `variants` = the eight contracted ones (+ the two fusion ones with `--extra-variants fusion`); `seeds [11,29,47]`; `repeats 1`; no `schedule_seed` (so 137). Then `run_suite.py --binary <bench> --config configs/aidb.json --output results/<lane>/sweep --timeout 14400 --resume`, then `summarize.py … --baseline packed_rank --output sweep/summary.csv`, then `sweep_report.py`. A pre-existing incomplete `results.jsonl` is renamed `results.partial-<timestamp>.jsonl` and the sweep restarts from zero (with `--resume` finding nothing) [aidb_pipeline.py:525-527]; a complete sweep with a different config or binary sha256 refuses to run without `--force`.
7. **throughput** [aidb_pipeline.py:533-541]: `throughput.json = {variant: {dataset: median over seeds of throughput_ops_s/1e6}}` after re-checking trace pairing; `throughput_detail.json` keeps every seed.
8. **scores** [aidb_pipeline.py:543-556]: `python3 tools/aidb_scores.py --hardness hardness.json --throughput throughput.json --output scores.json`; one block per scope with the 25 metrics (5 scalar, 10 two-dim, 10 three-dim) and per-variant conformance (section 5, E6).
9. **report** [aidb_pipeline.py:558-566]: `python3 tools/aidb_report.py --results <lane> --output report.html`; `--steps report` alone always regenerates.

`run.json` [aidb_pipeline.py:569-580] records the LAST pipeline invocation only: label, started/finished UTC, steps, datasets, settings (sample 2000000, sample_seed 42, sample_mode, ops 1000000, warmup 200000, seeds [11,29,47], alpha 0.1, region_keys 4096, flow_sample 4096, flow_steps 200, jobs, force, build_threads, extra_variants), variants, paths (binary_dir), sha256 of both binaries, and the scale note ("paper protocol is 200M keys, 20M warm-up and 100M measured lookups per run"). Caveat: the current `results/aidb/run.json` and `results/aidb_window/run.json` were written by `--steps report --force` invocations from `assemble.sh` at 21:49:40 UTC, so they say `extra_variants: []` and `build_threads: null` although the sweeps they sit next to were produced with `--extra-variants fusion --build-threads 16` (the `environment.json` config is authoritative for that) [results/aidb/run.json; results/aidb/sweep/environment.json; results/aidb/run_chain.sh].

`--extra-variants fusion` [aidb_pipeline.py:47-51] adds `packed_rank_fusion_auto` and `packed_rank_flow_costsel` and first checks that `scaleli_bench --help` mentions `--fusion`. `--sample-mode window` changes only the sample file names, the flow files and the results directory (`--results results/aidb_window`).

---

## 4. Machine, builds, QoS and the noise floor

**Machine** (measured now with `sysctl`, since `environment.json` records `cpu: "unavailable"` because `lscpu` does not exist on macOS): Apple M3 Max, 16 cores = 12 performance + 4 efficiency, 68,719,476,736 B (64 GiB) RAM, macOS 15.7.5 (24G624); `environment.json` of every sweep: `platform macOS-15.7.5-arm64-arm-64bit-Mach-O`, `machine arm64`, Python 3.14.6, `Apple clang version 17.0.0 (clang-1700.0.13.5)`, `affinity null`, `git_commit HEAD`, and the warning that governor/NUMA/isolation are uncontrolled [results/*/sweep/environment.json]. Compare the paper's testbed: two Intel Xeon Gold 5118 (12 cores/socket, 2.3 GHz), 384 GiB, Ubuntu 20.04.6, GCC 9.4.0 `-O3`, worker thread pinned to one core [AIDB §4.1, PDF page 4]. Our runs were NOT pinned; the only control is the QoS request.

**Builds** (all `CMAKE_BUILD_TYPE Release`, `CMAKE_CXX_FLAGS_RELEASE -O3 -DNDEBUG`, `/usr/bin/c++`, no `-march=native` because `SCALELI_NATIVE` is OFF by default [CMakeLists.txt:4,12-13; build-*/CMakeCache.txt]):

| build dir | scaleli_bench sha256 (prefix) | built (local time) | options in `--help` | used by |
|---|---|---|---|---|
| `build-fusion` | 3cc4fa88fedc1fd6… | 2026-09-21 17:18:13 | 18 lines; has `--fusion`, `--build-threads`, `--latency`; NO `--qos`, `--root`, `--root-alpha`; no QoS symbol linked (`nm -u` finds none) | E1 hardness (`scaleli_hardness` 1a414ce4…), E2 uniform + window sweeps, E4 granularity |
| `build-final` | at first 837daca73c112453… (environment.json of E3, built ≈ 19:14 local); now c63ceab046fe50b2… (rebuilt 20:45:24 after the root fix) | 21 lines; adds `--qos`, `--root`, `--root-alpha`; QoS symbol linked | E3: non-root rows from the first binary, root rows from the rebuilt one |
| `build-root` | c63ceab046fe50b2… (identical bytes to the rebuilt build-final), built 20:45:27 | same 21 lines | E5 full scale |

What differs between them is the source at build time: build-fusion predates the learned root and the QoS request; build-final/build-root include them and the root-fit fix (fit on the regions' first real keys instead of region 0's sentinel fence 0 [index.hpp:366-369]). `[unverified: the exact pre-fix build-final source is gone (the directory was rebuilt in place); only its sha256 837daca7… survives in results/aidb_final/sweep/environment.json]`.

**QoS** [benchmark.cpp:158-163]: the request is made unconditionally at the start of `main` unless `--qos 0`, but only by binaries compiled from a source that has it. So E2 and E4 (build-fusion) ran with the DEFAULT QoS class and no `--qos` option in their commands; E3 and E5 ran with `--qos 1` explicit and the call linked. The help text is explicit that this is "Not a pinning guarantee".

**Noise floor, three ways to see it (all from the result files):**

1. *Same structure, same trace, two variants.* In E3, `packed_rank_root_raw` falls back to binary search on books_uniform, osm_uniform, osm_window and planet_uniform (`root_model false`), so those runs execute exactly the same code path as `packed_rank` on the same trace; their work counters are byte-identical. The 12 throughput ratios range 0.824–1.029 (geometric means per dataset 0.903–0.948) [protocol_examples.py; results/aidb_final/sweep/results.jsonl]. The MEETING_NOTES figure "up to 8% (10% for the binary-fallback root)" is the geometric-mean view; per seed the deviation reaches 17.6%. Note `packed_rank_root_fusion` is NOT structurally identical to `packed_rank_root_vf4` even when the root chooses the same candidate, because the flow file also switches on the per-region bypass rule (flow_regions 1–43 of 489, transform_calls > 0), so that pair is not a clean repeat except on stack_window (3 pairs, 0.894–1.055).
2. *Same variant, different query seeds.* `packed_rank` max/min throughput over the three seeds: 1.005 (planet_uniform) to 1.221 (books_uniform), 13 of 20 samples above 1.06 [protocol_examples.py]. This is the "per seed up to 13%" spread of the notes (here it reaches 22% on books_uniform, seed spread mixes noise with genuine trace differences).
3. *The interrupted first uniform sweep.* `results/aidb/sweep/results.partial-20260921T183803.jsonl` holds 254 rows produced 15:24–16:10 UTC with the SAME commands as the final 300 rows, but concurrently with the window pipeline (four hardness processes, a sort) on the same host: identical counters, throughput ratios final/partial from 0.388 to 62.2, median 0.87. It is unusable as a repeat and shows why the sweeps were serialized afterwards (`run_chain.sh` runs one thing at a time). The 19-row window partial (`build-threads 1`, a first attempt) ranges 0.715–1.495.

Conclusion used throughout the guide: per-dataset throughput effects below ≈ 6% on the geometric mean, and ≈ 15–20% on single runs, are not resolvable on this host. Probe counts, memory and build decisions are deterministic and are the primary evidence.

---

## 5. Experiment cards

### E1 — Hardness metrics (full files and samples)

- **Purpose.** RMSE, ME, CD, PLA-32, PLA-4096 of the ten GRE datasets at full scale (the paper's scope), of the 2M samples, and of the same keys after our flow and after CSV-style virtual points, to place our controls on the paper's hardness axes.
- **Binary.** `scaleli_hardness` from build-fusion (sha256 1a414ce4…) [results/aidb/run.json binaries].
- **Inputs.** The ten downloads (200,000,000 keys, 1,600,000,008 bytes each; sha256 in provenance.json; URL `https://www.cse.cuhk.edu.hk/mlsys/gre/<name>`), the seven `.sorted` copies, the 2M samples, the flows.
- **Scopes and commands.** See section 3 step 5: `full` (`--flow`), `sample_flow` (`--flow --virtual-alpha 0.1`), `sample_csv` (`--virtual-alpha 0.1`), all with `--region-keys 4096 --pla-eps 32,4096 --check-sorted 1`. Window lane: same three jobs on the window samples (the full-file jobs are cache hits).
- **When / how long.** Uniform lane 15:21:50–15:24:03 UTC 2026-09-21 (132.3 s with `--jobs 4`); window lane 16:12:41–16:14:26 UTC (105.1 s) [pipeline.log of each lane]. Per file, e.g. fb full: 17.37 s elapsed, 14.35 s real / 30.88 s user in the timing log [hardness_details.json; results/aidb/hardness_timing/fb_full.log].
- **Outputs.** `results/aidb/hardness/<name>_{full,sample_flow,sample_csv}.json` (+ `.command.json` with the sha256 of the binary), `hardness.json` (6 scopes × 10 datasets × 5 metrics), `hardness_details.json` (FMCD slope/intercept/capacity/D/U_T, timings, virtual point counts: 184,990–199,707 per sample over 489 regions), the same under `results/aidb_window/`.
- **Rows.** 60 metric blocks per lane.
- **Caveats.** CD on transformed/smoothed features uses U_T + 1e-6·(mean gap) instead of LIPP's literal 1e-6 (documented; `fmcd.conflict_degree_lipp_epsilon` gives the literal value) [hardness --help; hardness_details cd_note]. CD carries ±1 rounding uncertainty. Flow scopes can be degenerate when the flow collapses keys onto equal doubles (the pipeline warns when duplicates > 50% of keys) [aidb_pipeline.py:495-497]. The sample scopes describe 1%-samples, not the corpus (manifest warning).

### E2 — Region-level sweeps on the uniform and window samples

- **Purpose.** Paired throughput, probes and memory of the eight contracted variants plus the two fusion controls on the ten 2M samples: does the NFL-style transform or CSV-style smoothing inside a 4,096-key region reduce work or time?
- **Config.** `configs/aidb.json` = `results/aidb/sweep/config.json` (uniform) and `results/aidb_window/sweep/config.json` (window), generated by the pipeline with `--jobs 4 --build-threads 16 --extra-variants fusion` [run_chain.sh]. `common`: `--load-ratio 1 --miss 0 --query-distribution uniform --ops 1000000 --warmup 200000 --verify 0 --instrument 1 --latency 0 --region-keys 4096 --build-threads 16`.
- **Datasets/samples.** `data/samples/<name>_2M_uniform_s42` with `results/aidb/flows/<name>_2D2H2L.txt`; `data/samples/<name>_2M_window_s42` with `results/aidb_window/flows/<name>_2D2H2L.txt`.
- **Variants** (exact expansion beyond the common options, `--profile read_only --seed S --data … --format sosd --dtype uint64`): `sorted_vector` → `--index sorted_vector`; `raw_rank` → `--policy raw --routing rank`; `packed_rank` → `--policy min_bytes --routing rank`; `packed_rank_flow` → `--policy min_bytes --routing rank --flow <abs path> --flow-bypass 1`; `packed_rank_flow_forced` → `… --flow <abs> --flow-bypass 0`; `packed_rank_vp10` → `… --virtual-alpha 0.1`; `packed_rank_flow_vp10` → `… --flow <abs> --flow-bypass 1 --virtual-alpha 0.1`; `packed_byte` → `--policy min_bytes --routing byte`; `packed_rank_fusion_auto` → `… --flow <abs> --virtual-alpha 0.1 --fusion auto`; `packed_rank_flow_costsel` → `… --flow <abs> --fusion auto`.
- **Seeds / ops / warm-up.** 11, 29, 47; 1,000,000 lookups after 200,000 warm-up reads; 2,000,000 keys loaded.
- **Binary.** build-fusion scaleli_bench 3cc4fa88… (no QoS request possible, see section 4).
- **When / how long.** Window sweep: 16:18:08–16:38:03 UTC (1,194.7 s for 300 runs); uniform clean rerun: 16:38:03–17:01:05 UTC (1,382.2 s for 300 runs) [pipeline.log; environment.json timestamps]. Local time = UTC + 2 (run_chain.log: WINDOW 18:16:19–18:38:03, UNIFORM 18:38:03–19:01:06).
- **Outputs.** `results/aidb{,_window}/sweep/{results.jsonl, summary.csv, environment.json, identity.json, config.json, sweep_report.html}`, `throughput.json`, `throughput_detail.json`, `scores.json`, `report.html`, `chart_data.json`, `fusion_report.html` (both lanes).
- **Rows.** 300 + 300 (10 datasets × 10 variants × 3 seeds). Same trace fingerprint for all ten variants of a (dataset, seed).
- **Caveats.** The uniform sweep was run three times: an 8-variant attempt at 15:24 UTC (contaminated by concurrent work, archived as `results.partial-20260921T183803.jsonl`, 254 rows), a refused attempt at 16:10 ("holds a sweep with different settings"), and the clean 10-variant run. The window lane had a 19-row partial with `build-threads 1` (`results.partial-20260921T181808.jsonl`). `run.json` in both lanes reflects a later `--steps report` call (section 3).

### E3 — Final sweep (QoS, 5M lookups, root ablation) — `results/aidb_final`

- **Purpose.** Repeat the region controls with more lookups and the performance-core QoS request, and add the learned-root ablation (raw root, flow root, virtual fences, fusion, region vp + root fusion) on all twenty samples.
- **Config.** `results/aidb_final/config.json`: `common` = `--load-ratio 1 --miss 0 --query-distribution uniform --ops 5000000 --warmup 500000 --verify 0 --instrument 1 --latency 0 --build-threads 16 --qos 1` (note: no `--region-keys`, so the default 4096); 20 datasets (`<name>_uniform` and `<name>_window`, flows from the matching lane); 9 variants; seeds 11, 29, 47; repeats 1.
- **Variants** (expansion beyond common/profile/seed/data): `sorted_vector`, `raw_rank`, `packed_rank`, `packed_rank_vp10` as in E2; `packed_rank_root_raw` → `--policy min_bytes --routing rank --root model`; `packed_rank_root_flow` → `… --root model --flow <abs>`; `packed_rank_root_vf4` → `… --root model --root-alpha 4`; `packed_rank_root_fusion` → `… --root model --root-alpha 4 --flow <abs>`; `packed_rank_vp10_root_fusion` → `… --virtual-alpha 0.1 --root model --root-alpha 4 --flow <abs>`.
- **Binary.** build-final. `environment.json` records sha256 837daca7… (the pre-fix build); the 300 root-variant rows were re-run with the rebuilt build-final (sha256 c63ceab0…, 20:45:24 local); the command path is the same string for all 540 rows.
- **Timeline (local).** 19:14:38 an 800-job attempt (10 variants incl. `packed_rank_fusion_auto`, 4 seeds incl. 61) wrote `environment.json` and a handful of rows, then was stopped; 19:16:43–20:11:18 `run.sh` ran the 9-variant × 3-seed grid with `--resume` ("[540/540]"); after the root-fit fix the root rows (300) plus the 7 stray rows (6 seed-61 runs + 1 `packed_rank_fusion_auto`) were moved to `results.stale-root-before-fix.jsonl` (307 rows, mtime 20:46:02) and the 300 root jobs re-run by `run_chain2.sh` 20:46:02–21:17:20 [results/aidb_final/run.sh, run.log; run_chain2.log; verdicts.json entry 11 for the 547-row inventory before the move].
- **What the fix changed.** Pre-fix `fit_root` fitted the raw root through region 0's `low_fence = 0` sentinel [index.hpp:432]; for a contiguous window whose first key is far from 0 the OLS line predicts ≈ rank 244 for every key, so the raw estimate was 14.1–14.4 on all windows and only the flow+fences candidate beat binary (the "synergy" that the synergy verifier refuted, verdicts.json entry 14). Post-fix the root is fitted on the regions' first real keys [index.hpp:366-370]. Example fb_window seed 11: raw estimate 14.34 → 3.58; `packed_rank_root_vf4` root probes 8.96 (fallback) → 2.27 with `root_vp true`; `packed_rank_root_fusion` 2.88 with `root_flow true` → 2.27 with `root_flow false` [results.stale-root-before-fix.jsonl vs results.jsonl].
- **Outputs.** `sweep/results.jsonl` (540 rows), `sweep/summary.csv` (baseline packed_rank), `root_analysis.json` (from `results/aidb/root_analysis.py`), `run.log`.
- **Caveats.** `assemble.sh` labels this lane "5 seeds": wrong, it is 3 seeds (verifier correction, verdicts.json entry 11). The 240 non-root rows come from the pre-fix binary; the non-root code paths are unaffected by the fix `[unverified: not re-run after the rebuild]`. `environment.json` describes the 800-job attempt, not the grid that was run.

### E4 — Granularity ablation — `results/aidb_granularity`

- **Purpose.** Does the transform start to matter when regions grow (4,096 → 16,384 → 32,768 keys), where a single per-region linear model can no longer absorb curvature?
- **Config.** `results/aidb_granularity/config.json`: `common` = `--load-ratio 1 --miss 0 --query-distribution uniform --ops 1000000 --warmup 200000 --verify 0 --instrument 1 --latency 0 --build-threads 16`; datasets fb, osm, planet, covid, genome (uniform samples, uniform flows); seeds 11, 29; 15 variants = 3 region sizes × 5 kinds:
  `packed_rank_r{R}` → `--region-keys R --policy min_bytes --routing rank`; `packed_rank_flow_forced_r{R}` → `… --flow <abs> --flow-bypass 0`; `packed_rank_vp10_r{R}` → `… --virtual-alpha 0.1`; `packed_rank_flow_vp10_r{R}` → `… --flow <abs> --flow-bypass 0 --virtual-alpha 0.1` (NOTE: forced flow, unlike E2's `packed_rank_flow_vp10` which uses the bypass); `packed_rank_fusion_auto_r{R}` → `… --flow <abs> --virtual-alpha 0.1 --fusion auto`; for R ∈ {4096, 16384, 32768}.
- **Binary.** build-fusion 3cc4fa88… (no QoS).
- **Timeline (local).** Started 19:01:06 (`run_chain.sh`, logged "GRANULARITY DONE" 19:14:33 but the grid was not complete), resumed in `run_chain2.sh` (after 21:17, interrupted) and `run_chain3.sh` 22:50:20–23:49:39, ending with 150 rows [run_chain*.log].
- **Outputs.** `sweep/results.jsonl` (150 rows = 5 × 15 × 2), `summary_r4096.csv`, `summary_r16384.csv`, `summary_r32768.csv` (each with baseline `packed_rank_r<R>`, so ratios are within a region size).
- **Caveats.** Only two seeds, so the bootstrap interval is the two-seed range. Regions per sample: 489 / 123 / 62 (2M/R rounded up).

### E5 — Full scale (200M keys) — `results/aidb_fullscale`

- **Purpose.** The paper's key count on two datasets: does the learned root and the region smoothing behave at 48,829 regions as at 489?
- **Configs.** Planned `config.json`: fb (`data/external/gre/fb`) and planet (`planet.sorted`), 6 variants (`packed_rank`, `packed_rank_root_raw`, `packed_rank_root_flow`, `packed_rank_root_vf4`, `packed_rank_root_fusion`, `packed_rank_vp10_root_fusion`), seed 11, `common` = `--load-ratio 1 --miss 0 --query-distribution uniform --ops 2000000 --warmup 200000 --verify 0 --instrument 1 --latency 0 --build-threads 16 --qos 1` (12 jobs). Split because the planet root smoothing at α_root = 4 is 195,316 greedy rounds over 48,828 fences (Algorithm 1 is O(budget × fences)) and did not finish in the 2 h timeout of `run_chain2.sh`: `config_a.json` (both datasets: `packed_rank`, `packed_rank_root_raw`, `packed_rank_root_flow`, `packed_rank_root_vf004` → `--root model --root-alpha 0.04`, `packed_rank_vp10_root_vf004` → `--virtual-alpha 0.1 --root model --root-alpha 0.04`; 0.04 × 48,829 = 1,953 fences, the absolute count the 2M runs spent at α = 4), `config_b.json` (fb only at α = 4: `root_vf4`, `root_fusion`, `vp10_root_fusion`), `config_c1/c2/c3.json` (planet at α = 4, one variant each, 4 h cap).
- **Binary.** build-root c63ceab0… (QoS linked, `--qos 1`).
- **Timeline (local).** `run_chain2.sh` 21:17:20 started the 12-job plan and wrote 2 rows (planet root_flow, fb root_fusion) before being stopped; `run_chain3.sh` 22:21:05 config_a → 22:42:17, config_b → 22:50:20 (13 rows), c1 23:49:40 → 00:59:37 (planet `root_vf4`: build 2,067.8 s, ×2 passes), c2 (planet `root_fusion`) started 00:59:37 and was still running when this section was written (PID 31890, elapsed 47:57 at 01:47 local); c3 not started [run_chain2.log, run_chain3.log, `ps`].
- **Outputs.** `sweep/results.jsonl` (14 rows), `sweep/summary.csv`, `root_analysis.json/.txt` (written 22:50:20, 13 rows; recompute with `python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl` to include the 14th).
- **Key numbers for the cards.** 48,829 regions; binary root 15.658 probes/lookup (emulated 15.6579); fb raw root 10.81, fences 3.42 with 973 virtual fences (greedy stops at 973 under both the 1,953 and the 195,316 budget), throughput 0.648 (binary) / 0.611 / 0.812 / 0.723 Mops for control / raw / fences / fusion, i.e. the same fences structure measured three times gave 0.65, 0.72, 0.81; planet: raw estimate 25.8 > 15.66 → fallback, 1,950-fence root estimate 25.2 → fallback, 195,316-fence root 13.75 estimate / 13.69 measured; region vp10 adds 161–190 s of build and 0.8 B/key [results/aidb_fullscale/sweep/results.jsonl].
- **Caveats.** Single seed; no throughput claim at 200M; `build_ns` is wall clock of a 16-thread build while `smoothing_ns` (2.4–2.9 × 10¹² ns for region vp) is summed thread time; each job builds twice (pass 1 + pass 4).

### E6 — Conformance/coverage scores — `results/aidb/scores.json`, `results/aidb_window/scores.json`

- **Purpose.** Apply the paper's eq. (1)–(3) to OUR variants: for each of the 6 hardness scopes and 25 metrics, coverage (fraction of comparable dataset pairs under the metric's partial order) and conformance (weighted agreement between "harder" and "lower throughput").
- **Inputs.** `hardness.json` (E1) and `throughput.json` (E2 medians in MOPS per variant × dataset). Command: `python3 tools/aidb_scores.py --hardness results/aidb/hardness.json --throughput results/aidb/throughput.json --output results/aidb/scores.json` [aidb_pipeline.py:547]. Defaults: `--ddof 0` (population std), `--min-datasets 3`.
- **Structure.** `scores.json[scope] = {metrics: {name: {dims, coverage, conformance, per_variant, comparable_pairs, incomparable_pairs, per_variant_detail, pairs}}, datasets, variants, skipped, dropped_datasets, ddof, normalized_throughput, hardness, note}`; 10 variants, 10 datasets, nothing skipped [results/aidb/scores.json]. Example: scope `sample`, metric PLA-32: coverage 1.0 (45/45 pairs comparable), conformance −0.397, per variant from −0.554 (raw_rank) to −0.194 (sorted_vector).
- **When.** 17:01:05 UTC (uniform), 16:38:03 UTC (window), 0.1 s each [pipeline.log].
- **Caveats.** Conformance measures whether OUR throughput ordering follows the metric; it says nothing about the paper's indexes. A variant with zero throughput std would be skipped, not scored +1 (fixer round 1, verdicts.json entry 10). The two "GRE" and "CD·PLA-32" lines printed by `walkthrough.py demo 09` come from a separate demo on `results/local`, not from this scores file [assembly1.log].

### E7 — Verification — `results/aidb/verification/verdicts.json` (16 entries)

Entries 0–4 are the builder reports of contracts C1–C5 (hardness binary, benchmark `--latency`/learnability capture, scorer, report, pipeline); entries 5–9 are the first-round verifiers (findings: unsorted GRE files → the `sort` step; provenance and idempotency gaps; the ten-variant deviation → `--extra-variants`; the ε rule for CD; std = 0 scoring); entry 10 is the fixer round. Entries 11–15 (workflow wf_26569070) verified the root-ablation claims on `results/aidb_final`:

| verifier (dir) | recomputed | verdict |
|---|---|---|
| recompute (`recompute/recompute.py`) | run counts, pairing (one fingerprint and one checksum per (dataset, seed)), probes per lookup, paired geometric speedups with seed bootstrap, sorted_vector vs every packed variant | pairing confirmed (0 mismatches); "5M × 5 seeds" wording refuted (3 seeds); C3a fences-only +6–12% refuted; C7 sorted_vector 1.15–2.3× confirmed |
| coststory (`coststory/coststory.py`) | `1e9/throughput == elapsed/ops` to 0.0000 ns; transform_calls per lookup = root_flow + flow_regions/regions within 0.0015; counters of root variants identical to packed_rank except root_probes and transform_calls; ns per probe by pooled regression | C4 probe cost needs qualification (≈ 4.0 ns/probe pooled, per-dataset spread is noise); C5 flow charge of 4 uncalibrated confirmed |
| synergy (`synergy/`) | exact Python emulation of `fit_root`'s estimator reproducing every stored estimate; counterfactual fit without the sentinel | C1 "synergy" REFUTED: artifact of region 0's sentinel fence → the fit was changed and E3's root rows re-run |
| claims (`final/claims.py`) + build/ctest/unittest/dry-run logs | fresh Release build, ctest 4/4, 46 unit tests, dry-run pipeline with `--extra-variants fusion`, `--help` option list, a `--verify 1` differential run | T1–T7 confirmed |
| pla (`pla/compare.py`), metrics (`metrics/gen.py`, `vlib.py`), scores (`scores/ref_scores.py`) | PLA segment counts vs a PGM-index reference on fixtures, synthetic sets and 5M-key prefixes; FMCD/CD oracle; an independent implementation of eq. (1)–(3) | agreement (PLA identical for ε ∈ {0,1,8,32,128,1024,4096}; scorer agreement 4e-16 per MEETING_NOTES §1) `[unverified: I did not re-run these scripts]` |

---

## 6. Variant dictionary (every name used anywhere → options → what it tests)

| variant | options beyond the common ones | what it isolates |
|---|---|---|
| `sorted_vector` | `--index sorted_vector` | plain binary search over the sorted uint64 array: the "no index" reference |
| `raw_rank` | `--policy raw --routing rank` | our region/block map with uncompressed blocks and rank routing: removes decoding cost only |
| `packed_rank` | `--policy min_bytes --routing rank` | the packed control (codec chosen per block by min bytes); baseline of every AIDB summary |
| `packed_byte` | `--policy min_bytes --routing byte` | packed with byte routing (E2 only) |
| `packed_rank_flow` | `+ --flow $flow --flow-bypass 1` | NFL-style transform under the per-region bypass rule (kept iff D₉₉ drops ≥ 10%) |
| `packed_rank_flow_forced` | `+ --flow $flow --flow-bypass 0` | transform forced into every region: upper bound of what it could change |
| `packed_rank_vp10` | `+ --virtual-alpha 0.1` | CSV-style virtual points, λ = 0.1 × 4096 ≈ 409 per region |
| `packed_rank_flow_vp10` | `+ --flow $flow --flow-bypass 1 --virtual-alpha 0.1` (E2) / `--flow-bypass 0` (E4) | both mechanisms stacked |
| `packed_rank_fusion_auto` | `+ --flow $flow --virtual-alpha 0.1 --fusion auto` | per-region cost-based choice among none/flow/vp/both by measured probes + 4-probe flow charge |
| `packed_rank_flow_costsel` | `+ --flow $flow --fusion auto` | cost-based choice between none and flow only (no vp) |
| `packed_rank_r{4096,16384,32768}` and the `_flow_forced_`, `_vp10_`, `_flow_vp10_`, `_fusion_auto_` forms | `--region-keys R` + as above | E4 region-size ablation |
| `packed_rank_root_raw` | `+ --root model` | learned root on the raw key feature, ranks as targets; binary fallback if not better |
| `packed_rank_root_flow` | `+ --root model --flow $flow` | root candidates {raw, flow} × ranks (regions also get the bypass flow) |
| `packed_rank_root_vf4` | `+ --root model --root-alpha 4` | root candidates {raw} × {ranks, virtual fences with budget 4 × regions} |
| `packed_rank_root_vf004` | `+ --root model --root-alpha 0.04` | E5 only: budget 0.04 × 48,829 ≈ 1,953 fences |
| `packed_rank_root_fusion` | `+ --root model --root-alpha 4 --flow $flow` | all four root candidates |
| `packed_rank_vp10_root_fusion` | `+ --virtual-alpha 0.1 --root model --root-alpha 4 --flow $flow` | region smoothing plus the full root selector |
| `packed_rank_vp10_root_vf004` | `+ --virtual-alpha 0.1 --root model --root-alpha 0.04` | E5 only |
| older learnability sweep (`results/learnability/summary.csv`): `packed_rank_vp05/vp30`, `packed_rank_vp10_relearn`, `packed_byte_flow` | `--virtual-alpha 0.05/0.3`, `--relearn 1`, `--routing byte --flow` | synthetic distributions, 20k keys, read_only + read_heavy; superseded by the AIDB lanes |

`$flow` always resolves to `results/aidb/flows/<name>_2D2H2L.txt` for uniform samples and `results/aidb_window/flows/<name>_2D2H2L.txt` for window samples (and to the uniform flow for the full files in E5) [config files].

---

## 7. Reproduction commands, in order (from S)

```
# 0. build (each build dir is a plain Release configure of the repo root)
cmake -S ../.. -B ../../build-final -DCMAKE_BUILD_TYPE=Release && cmake --build ../../build-final --parallel 4
#    (equivalent: the actual runs used build-fusion for E1/E2/E4, build-final for E3, build-root for E5;
#     build-fusion cannot be regenerated from the current source because the source has since gained --root/--qos)

# 1. data: place the ten GRE SOSD files under data/external/gre/<name> (the pipeline never downloads)

# 2. E1 + E2 uniform lane (verify-downloads, sort, sample, flows, hardness, sweep, throughput, scores, report)
python3 tools/aidb_pipeline.py --binary-dir ../../build-final/experimental/scaleli --jobs 4 --build-threads 16 --extra-variants fusion
#    equivalent: the actual run was staged (prep_sort.sh -> prep_after_sort.sh -> run_sweeps*.sh -> run_chain.sh)

# 3. E1 + E2 window lane
python3 tools/aidb_pipeline.py --binary-dir ../../build-final/experimental/scaleli --sample-mode window --results results/aidb_window --jobs 4 --build-threads 16 --extra-variants fusion

# 4. E3 final sweep
python3 tools/run_suite.py --binary ../../build-final/experimental/scaleli/scaleli_bench --config results/aidb_final/config.json --output results/aidb_final/sweep --timeout 3600
python3 tools/summarize.py results/aidb_final/sweep/results.jsonl --baseline packed_rank
python3 results/aidb/root_analysis.py results/aidb_final/sweep/results.jsonl --json results/aidb_final/root_analysis.json
#    equivalent: the actual run used --resume twice (section 5, E3)

# 5. E4 granularity
python3 tools/run_suite.py --binary ../../build-final/experimental/scaleli/scaleli_bench --config results/aidb_granularity/config.json --output results/aidb_granularity/sweep --timeout 3600
for r in 4096 16384 32768; do python3 tools/summarize.py results/aidb_granularity/sweep/results.jsonl --baseline packed_rank_r$r --output results/aidb_granularity/summary_r$r.csv; done

# 6. E5 full scale (A, B, then the planet alpha-4 runs with a 4 h cap each)
for c in a b c1 c2 c3; do python3 tools/run_suite.py --binary ../../build-root/experimental/scaleli/scaleli_bench --config results/aidb_fullscale/config_$c.json --output results/aidb_fullscale/sweep --timeout 14400 --resume; done
python3 tools/summarize.py results/aidb_fullscale/sweep/results.jsonl --baseline packed_rank --output results/aidb_fullscale/sweep/summary.csv
python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl --json results/aidb_fullscale/root_analysis.json > results/aidb_fullscale/root_analysis.txt

# 7. E6 scores are produced by step 2/3; to redo alone:
python3 tools/aidb_scores.py --hardness results/aidb/hardness.json --throughput results/aidb/throughput.json --output results/aidb/scores.json

# 8. cross-lane figures and reports
results/aidb/assemble.sh        # fusion_report.py over all summary.csv files, report.html of both lanes, manifest, walkthrough checks

# 9. worked numbers of this section
python3 results/aidb/guide_deep/scratch/protocol_examples.py
```

Timing budget from the logs: sort 2.5 min, samples + flows ≈ 2 min per lane, hardness ≈ 2 min per lane with 4 jobs, E2 ≈ 20–23 min per lane, E3 ≈ 55 min for 540 runs (+ 31 min for the root re-run), E4 ≈ 1 h 15 min spread over three sessions, E5 ≈ 30 min for A+B, 70 min for planet at α = 4, c2/c3 open.

---

## 8. Questions the supervisor may ask, with answers

1. **How do you know the traces are identical across variants?** The trace depends only on the key set, `--seed`, `--ops` and the profile [workload.hpp:74-137]; the binary prints `trace_fingerprint`, a splitmix64 hash over the loaded records and every operation [benchmark.cpp:54], and `summarize.py` refuses to pair runs whose fingerprints differ within a (dataset, profile, seed, repeat) group [summarize.py:28-31]. The recompute verifier found exactly one fingerprint AND one `result_checksum` per (dataset, seed) over all 540 final rows [verdicts.json entry 11].

2. **Why geometric means of ratios?** Throughput ratios are multiplicative; the geometric mean is symmetric under inverting the baseline (a 1.10 and a 0.909 average to 1.0, not 1.005), and averaging log-ratios per seed first gives each trace equal weight [summarize.py:10-15].

3. **What is the noise floor and how did you measure it?** From the result files, not assumed: identical code path and trace (`packed_rank_root_raw` in binary fallback vs `packed_rank`) gives 12 ratios in 0.824–1.029; the same variant across the three query seeds varies by up to 1.22×; the interrupted uniform sweep that overlapped other work varied up to 62× (section 4). Hence throughput differences under ≈ 6% on geometric means are not resolvable; probes, memory and selector decisions are deterministic and are the primary evidence.

4. **Why 5M lookups in the final sweep (and 1M before)?** The paper measures 100M lookups after 20M warm-up on 200M keys [AIDB §4.1]; on a 2M-key sample 1M lookups already touch every key ≈ 0.5 times and take 0.1–0.3 s, which is short enough for scheduler effects to dominate. 5M with 500k warm-up lengthens each timed loop to ≈ 1–2 s while keeping 540 runs under an hour; it is a compromise, not the paper's protocol, and the noise floor did not shrink much (section 4).

5. **Where does 8.96 root probes come from?** The fence binary search [index.hpp:352-353] counts one probe per loop iteration; 2,000,000/4,096 → 489 regions (last one 1,152 keys), so most lookups need 9 iterations (ceil(log₂ 489)) and a few 8; the size-weighted expectation is 8.9564, matching the measured 8.96 to the printed digits; at 200M keys 48,829 regions give 15.6579 vs measured 15.6578739875 [protocol_examples.py; results/aidb_fullscale/sweep/results.jsonl].

6. **Why load-ratio 1, miss 0, uniform queries?** To mirror the paper's read-only protocol ("bulk load the index and then perform random lookups for all keys") [AIDB §4.1]. No inserts means no compactions, so `relearn` and delta options are irrelevant and the counters are pure lookup work.

7. **What exactly is timed?** Only the loop over the trace in pass 1, with a monotonic `steady_clock` before and after, no per-operation timer and no counters [benchmark.cpp:80]; build, warm-up, memory accounting and the drain are outside it. Latency percentiles were disabled (`--latency 0`) so the only timing in every AIDB row is `elapsed_ns` (the cost-story verifier checked `1e9/throughput == elapsed/ops` to 0.0000 ns).

8. **Were results verified against a reference in the sweeps?** No: `--verify 0` in every sweep for speed. The differential check (std::map reference, every operation, plus final-state scan) was exercised in the unit tests and by the verifier's `--verify 1` run with root + fusion + virtual points [verdicts.json entry 15]; each sweep pass still cross-checks the result checksum between passes and aborts on mismatch [benchmark.cpp:96, 115].

9. **Why are the seven `.sorted` copies legitimate?** GRE serves those files unsorted and its own loader sorts and de-duplicates at load time [verdicts.json entry 6]; our audit found 0 duplicates in every file, so the sorted copy is the same multiset in key order; the copy's sha256 is in `provenance.json[name].sort_audit`, the download's own sha256 is kept separately, and the samples' manifests point at the copy they were drawn from.

10. **Are the window samples comparable across datasets?** They are all the rank interval [171,644,825; 173,644,825) of their file because `datasets.py` draws `start` from `random.Random(42)` with the same `total − n + 1` for every dataset (verified on fb and planet). They preserve local structure but not the global shape; the uniform samples do the opposite (manifest warning).

11. **Is the flow trained on the test keys?** Yes: 4,096 keys sampled from the same 2M sample the index is built from (in-domain), 200 Adam steps with the `--monotone` restriction; `prepare_flows.py` and the training report state this. For the 200M-key runs the 2M-sample flow was applied to the full file (see `flow_weights` in `results/aidb_fullscale/config*.json`).

12. **Why do the E2 runs lack `--qos` and `--root` in their JSON?** They were produced by the build-fusion binary, compiled before those options existed (its `--help` has 18 option lines vs 21, and `nm -u` shows no QoS symbol). Rows from that binary therefore have 62 top-level keys instead of 64. This is also why E2/E4 had no performance-core request while E3/E5 had one.

13. **What would you change to make the timing trustworthy?** Pin the process (not possible on macOS; on Linux `run_suite.py --cpu N` uses `taskset`), run repeats > 1 on a quiet host, use the paper's lookup count, and report the identical-structure spread next to every ratio; the probe counters already provide the deterministic story.

14. **How many runs and how long did the whole thing take?** E2 600 runs (≈ 43 min), E3 540 + 300 re-run (≈ 86 min), E4 150 (≈ 75 min), E5 14 done (≈ 2 h 10 min so far, planet α = 4 root fusion still running), E1 60 hardness jobs (≈ 4 min), all on 2026-09-21/22 local time; every wall-clock number above comes from `pipeline.log` and `run_chain*.log`.

15. **Which single file proves a given number?** `results/<lane>/sweep/results.jsonl` (one row per run, with the full command line), never the HTML; `summary.csv` and `root_analysis.json` are derived by the two 75-line scripts quoted in section 2 and can be regenerated in seconds.
