# Measurement contract and JSON schema

Schema version: **1**. All times are nanoseconds unless explicitly labeled seconds or operations/second. No wall-clock result is a claim of SOTA performance. The supplied host was shared and not isolated.

## Four passes, one trace

1. Build, warm up, replay without per-operation timers or software counters. Record elapsed time, result checksum, memory, and maintenance counters. Record an explicit maintenance drain after the measured replay.
2. Rebuild and replay with individual timers. Group latencies by actual operation type and by the first/second half of the trace. Group operations that triggered compaction separately. Verify the replay checksum against pass 1.
3. When `--verify 1`, rebuild and replay against an ordered-map oracle, checking each point result, mutation result, complete scan result, and final ordered contents. This pass is outside reported timing.
4. When `--instrument 1`, rebuild and replay with software counters. Verify its checksum against pass 1. Do not interpret the counters as hardware cache events.

`--latency 0` skips pass 2 entirely: `latency_pass` is then `false`, every `latency_ns.<type>` block is emitted with `count` 0 and zero-valued summaries, and `phase_latency_ns` / `compaction_operation_latency_ns` are likewise empty; throughput, checksum, memory and maintenance counters from pass 1 are unaffected. The `learnability` block (flow decision, tail conflict degrees, virtual points, rank SSE ratio) is read from the index bulk-loaded in pass 1, so it costs no extra bulk load and is identical whether or not the latency pass runs. The AIDB lane (`tools/aidb_pipeline.py`) runs every sweep with `--latency 0` because only `throughput_ops_s` enters its scores.

The timestamp-pair median is reported, not subtracted. Latency distributions include timer and dispatch/checksum overhead. Samples are retained in memory during the latency pass; allocation/scheduling noise remains possible. Throughput is closed-loop single-thread service throughput, not an open-loop queueing experiment. Warmup does not flush caches. Warmup accesses affect the adaptive controller's heat in the same way in each pass.

## Warm-up and steady state

A number measured on a cold index is not comparable to one measured on a warm index, and at 200M keys no uniform workload can ever make the ~2 GB structure cache-resident. "Warm" here therefore means three separate, individually checkable things.

1. **No paging during the timed pass.** `--prefault 1` (the default) reads one byte per 4096-byte span of every range the index owns, after `bulk_load` and before any timing, so any pages the OS reclaimed or compressed are paid up front. Measured on a 64 GB Apple M3 Max at 200M keys, it is a DRAM read sweep, not a paging fix: `bulk_load` has already written every page, the process shows 1-2 major faults with or without the sweep, and the 200M throughput does not change. Keep it for hosts under memory pressure, and check the fault counts there rather than assuming them. The walk does not cover `Config::flow` weights (a few hundred bytes). The sweep is read-only, so copy-on-write pages are not dirtied and the resident footprint reported by `memory_before` / `memory_after` is unchanged. The ranges walked are exactly those `memory()` accounts for: key arena, value column, block descriptors, delta, virtual features, the `Region` headers that hold the models and fences, the region pointer vector, and the root virtual-fence slot table. Ranges are taken at `size()`, not `capacity()`, so uninitialized slack past the last live byte is never read; after `rebuild()`'s `shrink_to_fit` that slack is near zero and `reserved_slack_bytes` reports it separately. `prefault_supported` is `false` for index types that cannot enumerate their allocations (`ordered_map`, whose `std::map` nodes are individually allocated); `prefault_bytes` is then 0. The direct external evidence for this knob is the major/minor fault count of the process, e.g. `/usr/bin/time -l`, not anything the benchmark prints.
2. **Warm-up queries from the measured distribution.** `--warmup-mode workload` (the default) draws warm-up keys from the same generator as the measured trace, with a different seed, at the same `--miss` ratio, so the caches, the branch predictors and the TLB reach the state the measurement will actually see. `--warmup-mode strided` is the historical behaviour: `--warmup N` lookups on the loaded keys strided by 997, a pattern no measured workload uses. Warm-up is read-only in both modes; only the measured trace mutates the index.
3. **Checkable steady state.** `--chunks K` (default 8) splits the timed replay into K equal spans and reports `throughput_chunks_ops_s`, the per-span throughput, plus `steady_state_ratio` = last span / median span. Do not use `steady_state_ratio` as a warm-up check: the cold-start transient sits in the *first* span, so a cold run still reports a ratio near 1. Compute first span / median of the remaining spans from `throughput_chunks_ops_s` instead. Measured at 200M keys with no warm-up, the first 1/16 of a 5M replay runs ~19% slow and the transient lasts at most ~375k lookups; a workload-mode warm-up of 200k lookups already hides it. Chunking does not change how `throughput_ops_s` is computed: `elapsed_ns` still spans the whole replay, and the only extra work inside the timed region is K additional timestamp pairs. `steady_state_ratio` is a noise detector as well as a warm-up detector; on a shared host a single dip in one span moves it without any warm-up being involved, so read the chunk curve, not only the ratio.

| Field | Definition / caveat |
|---|---|
| `warmup_reads` | Warm-up lookups actually issued, in either mode |
| `warmup_mode` | `workload` or `strided` |
| `prefault`, `prefault_supported` | Whether the sweep was requested, and whether this index type can enumerate its allocations |
| `prefault_bytes` | Bytes covered by the sweep; equals `accounted_bytes` minus `reserved_slack_bytes` for `scaleli` and `sorted_vector` |
| `prefault_ns` | Wall time of the sweep. Not part of `build_ns` or `elapsed_ns`; it is reported so its cost is never hidden |
| `chunks`, `throughput_chunks_ops_s` | K and the per-span throughput, in trace order |
| `steady_state_ratio` | Last span / median span. Blind to a slow first span, so not a warm-up check; use first span / median of the rest |

## Main fields

| Field | Definition / caveat |
|---|---|
| `source_rows`, `unique_rows`, `initial_rows`, `final_rows` | Cardinalities before deduplication, after deduplication, after bulk-load selection, and after replay |
| `trace_fingerprint` | Deterministic hash of initial key/value records and operations; decimal string avoids JSON floating-point truncation |
| `result_checksum` | Deterministic checksum of operation results; a mismatch is a hard failure |
| `verified` | Whether the full differential replay was run; **false is not a successful correctness result** |
| `dataset_scope` | `prefix` when `--limit` was used; otherwise the full provided input, not necessarily the source corpus |
| `build_ns` | Index construction after prepared input exists; preprocessing/trace generation is not included |
| `throughput_ops_s` | All completed operations / timed replay seconds, including mutations that rebuild regions |
| `latency_ns.<type>.count` | Actual sample count; absent types have zero counts and zero-valued summaries |
| `latency_ns.<type>.p50/p95/p99/max` | Nearest-rank individual-operation latency statistics, not batch averages |
| `phase_latency_ns` | Mixed-operation latencies for the two halves; do not mistake a changed mix for drift in a fixed operation |
| `compaction_operation_latency_ns` | End-to-end latency of operations whose maintenance counter increased |
| `scan_records_returned` | Actual records emitted, accounting for short scans near the end of the map |
| `scan_records_per_total_second` | Scan records / **entire workload** time; not pure scan throughput in a mixed workload |
| `clock_pair_p50_ns` | Median of 10,000 adjacent timestamp pairs, reported as diagnostic overhead |

For pure scan throughput set read, insert, update and erase to zero and scan to one. Record the explicit overrides, not only the profile name. Returning a scan allocates a result vector in every implementation; this common cost is included. A later iterator API must be compared separately.

## Memory scope

`memory_before`, `memory_after`, and `memory_after_drain` contain:

- `key_bytes`: live encoded arenas, including per-block headers and delta restart tables; raw baseline keys occupy eight bytes each.
- `value_bytes`: live uncompressed uint64 payload column, including base records later shadowed by tombstones until compaction.
- `metadata_bytes`: object sizes, block descriptors, models, root pointer capacity, configuration and counters. Descriptors contain exact fence keys; they are not free.
- `delta_bytes`: live update-buffer entries including padding and tombstones, not just key/value content.
- `reserved_slack_bytes`: actual vector capacities beyond live sizes. `shrink_to_fit()` is non-binding and not assumed to succeed.
- `accounted_bytes`: sum of the five components.
- `estimated`: true for external native-size approximations whose accounting differs.

Core accounting excludes malloc headers, allocator fragmentation, stack, temporary compaction buffers, workload vectors, source dataset copies, query output vectors, and process runtime. `std::map` uses an allocation-counting allocator for tree nodes, but its allocator-control object is not a general RSS measurement. Run each index in its own process and collect steady/peak RSS plus allocator statistics for publication-quality memory results. Include transient compaction memory separately rather than replacing the steady-state measure with it.

`key_arena_ratio = key_bytes / (8 × initial_rows)` is key-only and can exceed one because of headers. `accounted_bytes / live_rows` is full-index accounted bytes per live key. A tombstone-heavy index needs both measures.

## Work and maintenance counters

`root_probes`, `coordinate_probes`, `fence_probes`, `delta_probes`, and `key_at_calls` count logical comparisons/accesses in their respective code paths. `decoded_keys` and `codec_bytes_examined` are software decoding-work measures; header reads and CPU/cache behavior are not comprehensively modeled. `blocks_decoded_for_scan` counts scan block expansions.

`cache_lines` and `cache_lines_per_operation` count the distinct 64-byte lines each lookup has to reach, summed over lookups. They are a software locality model, not hardware cache misses: a line already resident from an earlier lookup is counted again. They are also **not run-to-run reproducible**, because a line identity is an actual heap address shifted right by six. Two runs of the same binary on the same input differ by a few percent purely from where the allocator happened to place the arenas (measured spread ~3% over three runs at n=60000). Compare medians of repeated runs, never single values, and never treat a small difference between two variants as a real difference. `lines_overflow` counts touches dropped because a single lookup exceeded the 128-line per-lookup set; a nonzero value means `cache_lines` is an undercount. Ranges read in bulk must be noted with `note_range`, which covers every line in the range; noting only a range's two endpoints is correct only for ranges of at most 64 bytes.

`correction_distance` is the sum of absolute block-index displacements from learned candidates to exact corrected results. Divide by `block_routes`, not by all operations. `max_correction_distance` is a worst observed displacement. Binary routing has no learned candidate, so these fields are zero and not comparable as model accuracy.

Maintenance `bytes_rewritten` counts newly written key-arena bytes plus eight bytes per rebuilt base payload. It excludes metadata writes, reads, allocator traffic, temporary materialization, and bytes merely moved in the delta. It is therefore a **logical lower-bound proxy**, not hardware write amplification. `max_rewrite_bytes` is the largest one-rebuild proxy. Compaction and split counters exclude initial bulk loading. `raw_bypasses` counts hot raw rebuild events, not distinct regions.

`drain_ns` and `drain_additional_rewrite_bytes` quantify pending maintenance after the timed run. Also report end-to-end amortized throughput using `(elapsed_ns + drain_ns)` when comparing sustainable workloads. Repeated updates to the same key may leave a tiny delta and postpone a representation transition indefinitely; the explicit drain reveals that condition.

## Statistical interpretation

`run_suite.py` randomizes job order, executes a fresh process per variant/seed/repeat, records commands, compiler/CPU metadata and the benchmark binary SHA-256, and refuses to overwrite results. Optional Linux `--cpu` uses `taskset`; it does not set NUMA placement, governor, or frequency.

`summarize.py` rejects unmatched trace fingerprints across variants. It reports per-group medians and paired throughput ratios against `raw_rank` by default. Ratios are summarized in log space, and bootstrap intervals resample **seed clusters**, not millions of individual queries. With fewer than two independent seeds, interval endpoints are NaN. The included two-seed smoke intervals are exploratory and should not be used for significance claims. Use at least five seeds, repeated process runs, predefined datasets, and confidence intervals for the actual claim being made.

The default benchmark does not report P99.9/P99.99, hardware PMU events, physical disk I/O, energy, queueing latency, or out-of-process memory peaks. Those are explicitly proposed extensions, not hidden in existing fields.
