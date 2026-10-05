# 200M before/after protocol for NFL and CSV, designed for measurement power (Designer 3)

Scope: how to measure B (before), N (after NFL), C (after CSV) and NC (after both) on the ten
200M-key AIDB datasets so that each reported difference can be told apart from noise, and so that the
differences that cannot be resolved are reported as bounds. I ran nothing CPU-heavy while writing this
(a timing pilot is running). Every number below comes from existing result files or from stdlib
arithmetic on them. The two helper scripts are in this folder: `noise.py` re-reads earlier runs and
`plan_calc.py` does the wall-clock and resolution arithmetic.

## 0. Bottom line

1. **Only throughput is noisy.** Memory, compression (B/key by component), work counters, root fences
   used, flow-region fraction and build time are deterministic, or their effects are 10-300x, so one
   build per cell settles them. Repeats are spent on throughput alone.
2. **The brief's repeat counts are one-sample numbers.** 13 / 4 / 1 repeats for 5 / 10 / 20 % equals
   7.85 (sigma/delta)^2 with sigma = 6.5 %. That formula assumes the baseline is known exactly. A
   before/after contrast has noise on both sides, so it needs **27 / 7 / 2 repeats per cell**. Pairing
   adjacent runs does not reduce this, because adjacent runs are uncorrelated on this host (§1).
3. **Most of the predicted 200M time effects are smaller than any feasible noise floor.** A 200M lookup
   costs 1.2-1.7 us, against about 0.44 us at 2M, so a probe saving is diluted 3-4x. Predicted effects:
   NFL as NFL applies it, under 2 %; NC vs C, the same; CSV, 2-7 % faster; NFL forced into every region,
   2-20 % slower; NFL x CSV interaction, about 0. Section 3 shows the derivation.
4. **The design therefore has two layers.** The counters, which are exact, show the mechanism.
   Throughput is reported as (a) per-dataset intervals, which can only resolve large effects, and (b) one
   effect pooled over the ten datasets. Pooling is the only route to a ±3 % interval overnight. Null
   cells are reported as equivalence bounds, not as "no significant difference".
5. **Every block includes an A/A cell B' (an identical rebuild of B).** It gives a live noise floor on
   every plot at no modelling cost, and it catches drift.
6. **Overnight plan, about 9.4 h:** 10 datasets, 6 cells (B, B', N, Nf, C, NC), n = 4 interleaved
   blocks, 5M ops per run, root budget capped at 0.4 x regions.
   - Per-dataset contrasts: 95 % CI about ±15 %.
   - Pooled over datasets: about ±3 %.
   - It can resolve the forced-NFL slowdown and a pooled CSV effect of 4 % or more. It cannot resolve any
     per-dataset CSV, NFL-faithful or interaction time effect.
7. **Full plan, about 49 h:**
   - Cheap cells: n = 20 everywhere, CI about ±4.3 % per dataset and ±1.3 % pooled.
   - planet/osm C and NC at the faithful root budget (alpha 4): n = 5. Each of these builds takes about
     38 min.
8. **Thermal confound (new).** Each CSV cell is timed right after a 160-2,000 s all-core build, while
   B is timed after a 7 s build. The fix is a longer warm-up (4M lookups, about 5 s) plus a pre-registered
   chunk-trend diagnostic (§5).

## 1. What the noise is (evidence)

| source | what | value |
|---|---|---|
| `results/aidb_fullscale/sweep/results.jsonl`, rows with identical structure (identical counters) | A/A at 200M, 2M ops, single seed | planet binary-like group 579.7 / 533.0 / 555.8 / 570.9 kops (CV 3.7 %, range 8.3 %); fb root-fences group 811.6 / 650.8 / 723.5 (CV 11.1 %, range 22 %); fb root-raw 611.0 / 597.3; fb vp+root 720.8 / 727.2. **Pooled CV 6.4 % (df 7)** |
| `scratchpad/verify/B/*.json` (fb, warm-up 200k, 2M ops, 5 repeats x prefault 0/1) | between-process CV, idle host | 6.0 % and 3.5 %, **pooled 4.9 % (df 8)** |
| `scratchpad/verify/A2` (cold, 6 repeats) | heavy tail | one run +30 % (`fb_pf1_r4`, 1.069 Mops vs median 0.80): range 43 % in n = 6 |
| chunks within one run (`throughput_chunks_ops_s`) | sub-second noise | CV 5-8 % per 250k-op chunk (warm), 11-14 % per 125k-op chunk |
| adjacent pf0/pf1 runs in verify/B and A2 | correlation of neighbours | none visible (pf0 791-922 vs pf1 793-862 in the same reps) |

What follows from these numbers:

- **Per-process noise dominates.** The between-run spread (5-6 %) is larger than what chunk noise
  averaged over a run predicts (6.5 %/sqrt(8) ≈ 2.3 % at 2M ops). The process-level component is
  therefore about 6 %, and adding ops per run barely helps: σ_run ≈ 6.4 % at 2M ops, 6.2 % at 5M,
  6.1 % at 10M. Only more processes, or the in-process harness in §8, reduce it.
- **Pairing protects against drift, not variance.** Neighbouring runs are uncorrelated, so
  Var(paired log-ratio) ≈ 2σ². Pairing in blocks still removes slow drift over the night, which is its
  job here.
- **The tails are heavy.** 44 % range in 22 repeats is about 1.7x what a normal with σ = 6.5 % gives
  (about 25 %). The estimator must therefore be robust (§9), and n = 1 is never a measurement (fb root
  fences went 0.65 → 0.81 Mops on one identical structure).
- **Per-run fixed cost is about 31 s + 1 s per million ops**, before the build. This is load plus
  workload construction: `verify/*.time` real 39-42 s at 2M ops with a 5 s build; `verify/M` 54.8 s
  (sorted_vector) and 73.2 s (packed, 19 s build) at 10M ops. A control run is therefore about 51 s at
  5M ops, not 7 s.

Resolution available (σ = 6.5 %, paired two-cell contrast, `plan_calc.py`):

| repeats n per cell | 95 % CI half-width, one dataset | MDE at 80 % power, one dataset | 95 % CI, pooled over 10 datasets (homogeneous effect) |
|---|---|---|---|
| 1 | none (no interval) | none | ±5.7 % |
| 3 | ±22.8 % | 27 % | ±3.3 % |
| 4 | ±14.6 % | 18.5 % | ±2.8 % |
| 5 | ±11.4 % | 14.9 % | ±2.5 % |
| 8 | ±7.7 % | 10.4 % | ±2.0 % |
| 10 | ±6.6 % | 9.0 % | ±1.8 % |
| 20 | ±4.3 % | 6.0 % | ±1.3 % |
| 27 | ±3.6 % | 5.1 % | ±1.1 % |

The interaction (NC − C) − (N − B) has standard error 2σ/√n: ±13 % per dataset at n = 4, ±6 % at
n = 20, and ±4 % (n = 4) or ±1.8 % (n = 20) pooled. The pooled column assumes the same effect on every
dataset. A random-effects interval is wider whenever datasets disagree, and that is the one to report.

## 2. Which metrics need repeats

| metric | noise | repeats needed | source in the bench JSON |
|---|---|---|---|
| key / value / metadata B/key, accounted bytes, key-arena ratio | none (byte-exact; within 2.0-2.7 % of steady-state RSS for packed layouts) | 1 | `memory_before` |
| virtual points, root virtual fences, flow regions, root choice, root_probes_* estimates | none (deterministic build) | 1 | `learnability` |
| probes, key_at, decoded keys, transform_calls, cache_lines per op | SE about 0.001 at 5M lookups | 1 instrumented run | `work_counters` |
| build time | about ±10 % (16 threads, shared host); effects of 7 → 190 → 2,000 s | free (every run rebuilds) | `build_ns` |
| throughput | σ ≈ 6.5 %, heavy-tailed | the subject of this document | `throughput_ops_s`, `throughput_chunks_ops_s` |

Two checks come at no cost:

- **Determinism check.** `learnability` and `memory_before` must be byte-identical across all repeats of
  a cell. If they are not, the build is not deterministic and the repeats are not repeats.
- **Correctness at 200M without `--verify`.** That pass would need a reference map of 200M entries.
  Instead, read-only lookups with miss 0 return a `result_checksum` that depends only on data and trace.
  It must be equal across B, B', N, Nf, C and NC within a block, because they share the seed.

## 3. Predicted effect sizes at 200M, and which contrasts can be resolved at all

A lookup costs 1/0.58-0.85 Mops = 1.2-1.7 us. The fullscale runs and `verify/M` show 0.58-0.85 Mops for
packed rank routing. Assumed unit costs:

- A root probe (48,829 fences = 390 KB, L2-resident after warm-up): 4-8 ns. The 2M joint verifier
  measured 28.3 ns ≈ 6.9 probes, i.e. 4.1 ns per probe.
- A region fence probe: anywhere from that up to a DRAM miss. Block descriptors total 100 MB, which is
  not cache-resident.
- The transform: 28-66 ns per lookup that uses it. That is 28.3 ns in the serial microbench and
  66 ns [56, 77] in-index (MEETING_NOTES §3). This is NFL's batch-1 regime; NFL batches to reach
  8.4 ns/key.

| contrast | counter change (200M where known, else 2M) | predicted Δ throughput | resolvable per dataset at n = 4 / n = 20? |
|---|---|---|---|
| **B' vs B (A/A)** | identical structure | 0 | it is the noise floor |
| **N vs B** (faithful flow, NFL's bypass) | flow in a fraction f of regions (2M: up to 157/489 = 32 % on planet; 200M inert flow: 508 of 48,829 regions on planet, 6 on fb) × (transform + extra fence probes, +9-192 % when forced) | under 2 % slower | no / no; report an equivalence bound |
| **Nf vs B** (`--flow-bypass 0`, forced) | every lookup pays the transform; fence probes +0.46 (osm) to +5.0 (fb) | about 2-20 % slower (largest on fb, planet) | fb/planet-like: borderline / yes; osm-like: no / no |
| **C vs B** (region vp 0.1 + root vf) | fb: root 10.81 → 3.42 (vs the binary 15.66), fence 5.14 → 4.74, correction 1.86 → 1.57; planet at alpha 4: root 15.66 → 13.69, fence 3.99 → 3.42 | about 2-7 % faster (root: 7-12 probes × 4-8 ns ≈ 30-100 ns; region about 0.4-0.6 probes) | no / borderline (≥ 6 % only); **pooled: yes if ≥ ~4 %** |
| **NC vs C** | with a non-monotone (faithful) flow the root vp+flow candidate is skipped (`include/scaleli/index.hpp:406`, `if(!std::is_sorted(...))continue`), so the NC root is identical to the C root and NC − C ≈ N − B | under 2 % | no / no; report an equivalence bound |
| **interaction** | root 2x2 interaction −0.36 probes (2M); region and root savings add to within 0.003 probes | about 0 | no / no; counters only |
| planet NC with a **monotone** flow (the only 200M counter-level interaction) | root 13.69 → 3.32 probes, 111,688 fences, transform_calls 1.01/op (`aidb_fullscale`, inert `planet_2D2H2L.txt`) | 10.4 probes × 4-8 ns saved minus 28-66 ns of transform: −2 % to +4 % | no at any feasible n; build 2,973-5,758 s per repeat |

Consequence: the experiment can resolve the following and nothing more.

- **Exactly:** what each method does to memory, compression and work (counters).
- **Statistically:** whether CSV makes the 200M index faster on average across the ten datasets (≥ ~4 %),
  and whether forcing NFL makes it slower.

Any slide that claims a per-dataset time effect of NFL-as-NFL-applies-it, or of combining the two
methods, is reading noise.

> **Note for the other designers.** The planet monotone-flow row conflicts with the established "flow
> chosen in 0 of 60 root cells". That result was measured at 2M. At 200M, planet's fused root did pick
> flow+fences (`results/aidb_fullscale/root_analysis.txt`: `flow+fences111688`). Counters show it; time
> cannot resolve it.

## 4. Cells

The power plan depends only on build cost and predicted Δ, so the definitions below are the measurement
team's proposal. Swapping in other definitions does not change the arithmetic.

Common flags for every cell:

```
--format sosd --dtype uint64 --load-ratio 1 --miss 0 --query-distribution uniform
--policy min_bytes --routing rank --root model
```

| cell | extra flags | build (16 threads) |
|---|---|---|
| B | none (root model on ranks; falls back to binary where it loses, e.g. planet) | ~7 s |
| B' | identical to B; separate process | ~7 s |
| N | `--flow flows_free/<d>_<cfg>.txt --flow-bypass 1` (NFL's own switch); flow also offered at the root (`--flow-cost 4`) | ~8 s (transform 6.4 s summed, parallel) |
| Nf | `--flow ... --flow-bypass 0` (forced into every region) | ~8 s |
| C | `--virtual-alpha 0.1 --root-alpha A` | 161-190 s + root smoothing |
| NC | C + `--flow ... --flow-bypass 1` (`--fusion manual`: the literal composition, so the 2x2 is a factorial) | about the same as C (faithful flow: no extra root candidate) |

Notes on the cells:

- **Flows.** Pick the faithful flow per dataset from `results/aidb_flowv2/flows_free/best.json` by
  dataset name and cfg: s512_t2000 for fb, osm and planet; s64_t2000 for the other seven. The `path`
  fields in that file point into a deleted scratchpad.
- **Data paths.** Use `<name>.sorted` for covid, genome, history, libio, planet, stack and wise.
- **Optional cells.** The fused selector (`--fusion auto`) and the joint bend (`--root-alpha 4
  --flow <d>_joint.txt --flow-cost 0`) can be added in the full plan. `--fusion auto` smooths both
  features and costs 2-4x the region smoothing (MEETING_NOTES §4).
- **Root budget A.**
  - Full plan: A = 4, as established. Root smoothing time ≈ 7.2e-8 s × R × (48,829 + R/2), where R is the
    number of fences used. This is calibrated on planet (R = 195,316 → 2,068 s) and fits fb
    (R = 973 → 3.4 s vs the measured +3.4 s).
  - At 2M, osm and planet saturated the budget (1,956 of 1,956). At 200M both should be assumed to take
    about 2,000+ s per build.
  - fb went from 1 fence (2M) to 973 (200M), so scaling is not predictable for the other eight; layer 1
    measures it.
  - Cut-down plan: A = 0.4 (budget 19,532 fences, at most about 82 s per smoothed candidate). This is
    identical to A = 4 wherever R < 19,532 (fb: 973), and it is a documented deviation on planet/osm. The
    A = 4 planet counters already exist in `aidb_fullscale` (13.69 / 3.32 probes) and are quoted from
    there, labelled as such.

## 5. Per-run protocol

Binary: `/Users/louisvasseur/Downloads/scaleli_sota/build-fs/scaleli_bench` (sha256 prefix
f02d4864f6ca82a9; record the full hash).

```
--ops 5000000 --chunks 20 --warmup 4000000 --warmup-mode workload --prefault 1
--latency 0 --verify 0 --instrument {1 in layer 1, else 0} --qos 1 --build-threads 16 --seed S
```

- **ops = 5M (timed for about 6.5 s).** 10M buys about 0.1 points of σ for +7 s per run, so it is not
  worth it. The trace costs 32 B/op, i.e. 160 MB. The established 2M ops is acceptable too. 5M is chosen
  so that 20 chunks of 250k exist for the warm-up indicator.
- **Warm-up: workload mode, 4M lookups (about 5 s).**
  - Coverage: 4M uniform lookups touch each of the 48,829 regions about 82 times and each of the 1.56M
    block descriptors about 2.6 times. Everything that can be cache-resident (root fences 390 KB, region
    objects 13.7 MB, slot-to-region table, virtual-fence models) is fully exercised. Steady state
    includes the descriptor, arena and value misses that no warm-up can remove.
  - 200k lookups already hide the cold first chunk (established), so 4M is there for a second reason. It
    gives the SoC about 5 s of single-threaded work between the all-core build and the clock. This
    partly equalises the thermal state between B (7 s build) and C/NC (160-2,000 s all-core build). Cost:
    about +4 s per run versus 1M.
- **Prefault 1.** Harmless and cheap. It does not change the answer (established). Record
  `prefault_ns`.
- **Warm-up indicator.** c1 = first chunk / median(chunks 2..20) − 1. Per-run noise at 250k-op chunks is
  ±15 % (verify/B: −16 … +19 %), so a single run's c1 means nothing; it is always pooled.
  - Pass criterion (pre-registered): the median c1 over all runs of a cell, pooled across datasets
    (about 40 runs, SE about 1.3 %), lies within ±3 %.
  - Per dataset with n = 4 (SE about 4 %), only a gross failure is detectable. The cold reference is
    about −17 % at 250k-op chunks (−34 % at 125k with `--warmup 0 --prefault 0`).
  - `steady_state_ratio` (last/median) is not used.
- **Chunk-trend diagnostic (thermal), pre-registered.** For each run, take the slope of log(chunk
  throughput) over chunk index 2..20. Compare the median slope of C/NC runs with B runs. A positive
  C − B slope difference means the post-build thermal state is leaking into timing. In that case, report
  C/NC throughput on chunks 11-20 only, as a sensitivity analysis, next to the full-run number.
- **Cold reference.** One `--warmup 0 --prefault 0` run of B per dataset in layer 1 (10 runs, about
  9 min). This is the slide the supervisor asked for: "this is no warm-up; this is ours."
- **Instrumented runs.** In layer 1 the run uses `--instrument 1`. The timed pass is pass 1, so the
  throughput is a valid repeat. The second build for counters happens after timing. This costs one extra
  build per cell, once: about 2.3 h at alpha 4 on each saturating dataset, and about 4 min at A = 0.4.
- **Seeds.** Use query seed S = 1000 + block index, the same within a block (paired traces, equal
  checksums) and different across blocks (averages over traces). The build is deterministic, so query
  seeds are the only replicate randomness. The flow's training seed is fixed by the weight file and is
  not replicated; that is a caveat, not something to measure overnight.
- **Guards, logged per run:**
  - Before starting, `pgrep -x scaleli_bench` must be empty and the 1-min `vm.loadavg` must be < 1.5;
    otherwise wait.
  - A 5 s background sampler records an "others" column. A run with another bench process alive during
    it is invalid and is re-queued at the end of its layer, never silently dropped.
  - Run the whole thing under `caffeinate -i`, on AC power. Log `pmset -g therm` once per layer.
  - Per-run timeout = 3 × the predicted build + 300 s. A timeout is a result ("did not build within X s")
    and gets its own row.
- **Serial only.** One 200M process at a time, ever. This includes the instrumented pass, which lives in
  the same process.

## 6. Interleaving and stopping

- **Block = one full set of the cells for one dataset, in a random order.** Draw a permutation per block
  from a logged RNG seed. With 6 cells and n = 4, a Williams square cannot be completed, so use random
  permutations. Their purpose is to break any order × time confound.
- **Layer = block r of all ten datasets.** Run layers in sequence. Within a layer, put the eight cheap
  datasets first and planet/osm last, so that an overrunning expensive build delays only itself.
- **Layer-based stopping.** After every layer, the dataset set is balanced, so stopping at a layer
  boundary because the clock ran out is safe. Stopping on results is not allowed. n is fixed in advance
  and is never extended because an interval "almost" excludes 1.
- **Order of datasets inside a layer is fixed.** Page cache holds all ten files (16 GB) next to a process
  peak of about 9.6 GB on a 64 GB host, so dataset switching costs no disk I/O after layer 1.

## 7. The two plans

Run-time model: fixed 31 s + 1 s/Mops, plus build, plus (ops + warm-up)/0.75 Mops. Control run about
51 s; cheap-dataset C run about 239 s; alpha-4 C run on planet/osm about 2,302 s.

| plan | cells | n | root budget | wall clock | per-dataset 95 % CI (contrast) | pooled 10-dataset 95 % CI |
|---|---|---|---|---|---|---|
| **Cut-down (overnight)** | B, B', N, Nf, C, NC + 10 cold refs | 4 | A = 0.4 on all ten (= A 4 where R < 19,532) | **9.4 h** (8.8 h without Nf) | ±14.6 % (MDE 18.5 %) | ±2.8 % (interaction ±4 %) |
| **Full** | same (+ optional fused / joint) | cheap cells 20 everywhere; planet/osm C, NC 5 | A = 4 | **≈ 49 h** (planet/osm C+NC alone 12.8 h); 84.6 h if planet/osm C/NC also get n = 20 | ±4.3 % (MDE 6 %); planet/osm C−B about ±9 % | ±1.3 % (interaction ±1.8 %) |

**Cut-down plan, what it can resolve:**

- All deterministic metrics for all 6 cells × 10 datasets: memory by component, compression, counters,
  build time, root fences, flow fraction, and the exact counter-level interaction.
- Warm-up adequacy (pooled c1 ±3 %) and the thermal diagnostic.
- Per dataset: only effects of about ≥ 19 %, which in practice means a forced-NFL slowdown on fb- or
  planet-like data.
- Pooled: a CSV speed-up of about ≥ 4 %, an NFL-forced slowdown of about ≥ 4 %, and equivalence bounds of
  about ±3 % for N − B and NC − C.

**Cut-down plan, what it cannot resolve:**

- Any per-dataset CSV time effect (predicted 2-7 %).
- Any per-dataset N, NC − C or interaction time effect.
- planet/osm CSV at the faithful alpha-4 budget (counters are quoted from aidb_fullscale for planet; osm
  at alpha 4 is unmeasured).
- Latency tails (`--latency 0`).

**Full plan, what it can resolve:**

- Per-dataset CSV effects of ≥ 6 % on eight datasets.
- Per-dataset Nf slowdowns of ≥ 6 %.
- Pooled equivalence for NFL and for combining, at ±1.3 %.
- The interaction at ±1.8 % pooled.

**Full plan, what it cannot resolve:**

- Per-dataset NFL-faithful effects (predicted < 2 %, below the 6 % MDE).
- The per-dataset interaction (±6 %).
- planet/osm C and NC below about 9-15 %.
- Anything about the planet monotone-flow NC cell in time (§3).

## 8. Optional power multiplier: an in-process A/B harness (needs a new scratch binary, after the pilot)

The existing binary rebuilds the index for every timed replay. On planet/osm, each repeat therefore
costs a 38 min build, and each comparison crosses a process boundary where about 6 % of noise lives.

A copy of `src/benchmark.cpp`, compiled in this scratchpad (no listed build touched; the sources are not
edited), could do the following:

- Build each cell once per process. Memory: 6 indexes × about 2.1-2.3 GB + 3.2 GB harness + 6.4 GB
  transient per build, about 23 GB peak.
- Alternate timed 250k-op chunks across the cells in a randomised round-robin.
- Read `proc_pid_rusage(RUSAGE_INFO_V4)` `ri_cycles` / `ri_instructions` around each chunk. This gives
  frequency-invariant cycles per lookup next to wall time.

If the per-process component is host state (frequency, background load), then in-process alternation
cancels it. The contrast noise falls to about 6.5 % × sqrt(2/40) ≈ 1.5 % per process, so per-dataset
intervals of about ±3 % become affordable at n = 4 processes, with the expensive builds paid once per
process instead of once per repeat. If the component is allocation layout, the alternation does not
help.

Decide with a 30 min A/A diagnostic before adopting it: two identical B builds in one process,
alternated, over 5 processes. If the in-process A/A σ is well below 4.4 %, adopt it. Do not mix its
numbers with the existing binary's in one table.

## 9. How to present intervals so nobody reads noise as signal

1. **Separate exact from estimated.** Deterministic metrics go in their own table with no error bars,
   labelled "exact; identical across n repeats (checked)". Throughput always carries n and a 95 %
   interval. A single-run ratio never appears on its own. The existing fullscale ratios (1.25x, 1.18x,
   0.92x) are single runs and must not be quoted.
2. **Estimator.** Use the paired log-ratio per block, d = ln T_cell − ln T_B.
   - Point estimate: the geometric-mean ratio exp(mean d).
   - Interval: t-interval with n − 1 df, cross-checked by a bootstrap over blocks.
   - Robust companion: Hodges-Lehmann median of d. If the two disagree by more than half the interval, an
     outlier is driving the result; say so.
   - No run is dropped except under the pre-registered "others > 0" contention rule.
3. **The A/A row comes first.** Every throughput table and plot shows B' vs B first. Its interval is the
   noise floor, drawn as a grey band behind all other cells. Also state in words on the slide: "two
   builds of the identical index differ by up to 22 % at 200M (fb); σ ≈ 6.5 % per run."
4. **Pre-registered verdicts with an equivalence margin δ** (δ = 5 % pooled; per dataset, δ = the
   achievable half-width, ±15 % at n = 4):
   - "faster" / "slower": the CI excludes 1.
   - "equivalent within ±δ": the CI lies inside [1−δ, 1+δ].
   - "inconclusive": everything else.
   - Never write "trend" or "slightly faster" for a CI that crosses 1.
5. **The headline is the pooled effect.** It is a random-effects mean of the per-dataset log-ratios over
   ten datasets, with a dataset-level bootstrap interval. Per-dataset intervals are descriptive. With
   10 datasets × 5 contrasts, 2-3 intervals will exclude 1 by chance at 95 %. If a per-dataset claim is
   made anyway, use Holm-adjusted intervals.
6. **Show every run.** Plot each block's ratio as a dot over the interval, with n printed. Use
   chunk-trace plots, chunk index × throughput, for one run per cell plus the cold reference; this is
   the warm-up evidence.
7. **Check against prediction.** Next to each measured interval, print the counter-predicted range from
   §3. A measured effect far outside what the counters can explain (e.g. +25 % for a 0.4-probe change)
   is noise until proven otherwise.
8. **Report nulls as results.** "NFL-as-applied changes 200M throughput by between −x % and +y % (95 %,
   pooled); its counters change by …" is the full result. Do not dress it up.

## 10. Caveats

- σ = 6.5 % comes from a few small groups: df 7 at 200M, df 8 at 2M ops on fb. It may differ by dataset
  or by cell; CSV cells may have a different thermal history. B' estimates it live in every layer, and
  the final intervals use the observed σ, not 6.5 %.
- The independence of chunk noise is assumed, not shown. If chunks are autocorrelated, ops per run
  matter even less.
- The ns-per-probe figures at 200M (4-8 ns root, up to DRAM latency in region) are assumptions taken
  from the 2M microbench and cache sizes. The predicted ranges in §3 inherit them. Use `cache_lines`
  from build-fs counters (not present in the build-root fullscale rows) to sharpen them.
- The root-smoothing time model is fitted on two points (planet, fb). Root budgets for the other eight
  datasets at 200M are unknown until layer 1. That is why the cut-down plan caps A at 0.4.
- The planet alpha-4 NC build was measured at 5,758 s against about 3,163 s expected from its parts.
  The fullscale runs may have shared the host with another 200M job; treat it as a worst case.
- Throughput is single-threaded batch-1 uniform lookups. NFL's own evaluation batches lookups, so
  NFL's transform cost here is its batch-1 cost by construction. State this whenever an N cell is
  shown.
- All absolute RSS and allocator effects are macOS-specific (the 790 MB size-class cliff hits only raw
  arenas, none of these cells).
