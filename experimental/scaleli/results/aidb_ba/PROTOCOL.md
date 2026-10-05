# Before vs after NFL, CSV and gap removal on SCALE-LI at 200M keys: experimental protocol (final, v1)

Status: design only, frozen by the procedure in 5.0. Nothing was built or run to write it (other experiments own the
machine). Every number is read from a cited file or derived in the text; predictions say so. This document supersedes
`PROTOCOL_draft.md` and, where they differ, `$NB/PLAN.md`; it reuses PLAN.md, `run_ba.py` and `analyze_ba.py`
everywhere else. The three reviews it answers are `review_stats.md`, `review_systems.md` and `review_supervisor.md`
(this folder); the CHANGE LOG at the end maps every issue to a change or a reasoned rejection.

```
REPO = /Users/louisvasseur/Downloads/scaleli_sota
S    = $REPO/experimental/scaleli
NB   = $S/results/aidb_ba        (PLAN.md, run_ba.py, analyze_ba.py, rss_ba.sh, hostmon.py, pilot_seed11.jsonl)
SP   = /private/tmp/claude-501/-Users-louisvasseur-Downloads-scaleli-sota/ac6d4251-6384-4db6-8aba-a17e1917a2ab/scratchpad
P    = $NB/proto                 (all outputs and the frozen copy of this protocol; persistent, not /tmp)
BFS  = $REPO/build-fs/scaleli_bench        (existing binary; night 0 and the D layer)
BPR  = $REPO/build-proto/scaleli_bench     (new binary; night 1)
```

---

## 0. One-page summary

**Question.** On the ten 200M-key AIDB datasets, what does each method change, before vs after, in (1) memory,
(2) input-space compression (empty key space squeezed out so one line fits better), (3) throughput after a proper
warm-up? Methods: NFL-style flow feature (N; Nm = monotone flow), CSV-style virtual points (C; Cr = root only),
gap removal (G at the root; G_key over all keys, AIDB metrics only), and combinations.

| layer | what | host | gives |
|---|---|---|---|
| D | one instrumented run per (cell, dataset), existing binary | any light load, never beside timing | memory, work counters, root decisions, build time, strata |
| CL | 48,829 root fences and all 200M keys; B, G/G_key, Nm, C | same | top-k gap share, line RMSE/ME, PLA, root probes |
| **P** (primary timing) | one process per (dataset, replicate): build every cell, common 60 s cool-down, 8 rounds, each cell once per round in random order: 1M warm-up + 1M timed | quiet Mac | within-process contrasts; build-heat confound removed by design |
| X (secondary timing) | separate processes on fb, osm, planet: B, B2, C, SV, COLD; AIDB-length arm (20M/100M) on fb | quiet Mac | AIDB-shaped check, cold reference |

**Warm-up.** From a fresh process the cold transient lasts at most 375k lookups (idle replication `$SP/verify/A2`,
n = 6). X: 60 s cool-down, prefault, then **4M workload-mode lookups** (at least 10x the transient). P: **1M re-warm
per visit** (2.7x). Verified on every run by a placebo-chunk test, by a COLD reference (predicted -6 to -20% in the
first 1/16) and by the AIDB-length arm.

| step | engineering | machine | quiet Mac |
|---|---|---|---|
| 0. host fix: AC and charged to 80% or more, `powermode` 2 (reads 1 today), UTM VM stopped | — | — | — |
| N0, night 0: Q0, E1 A/A and E0h heat screen on build-fs, E5 RSS | 1-1.5 h | 2.3 h | yes |
| D + CL | 2.5-3.5 h | 6.6 h | no |
| build-proto: P1 cool-down + go-gate, P2 telemetry, P6 interleave; runner v2 | 7.5-9.5 h | — | — |
| N1, night 1: R1, E1b, P (n = 3), X | — | 8.7 h | yes |
| analysis code (written blind, before any throughput is read) | 3 h | minutes | — |
| only if the 200M go-rule passes: P3 gap removal at the root, G and GCr timed | 5-8 h | +0.7 h | yes |

**Tier M** (meeting within about 2 days): steps 0, N0, D + CL; about 4-5 h of engineering. It delivers memory,
compression, work, the noise floor and the heat screen, but no method throughput. **Tier C**: everything; about
14-17 h of engineering and two nights.

| gate | pass | on failure |
|---|---|---|
| G0 host | AC, powermode 2, VM gone, Q0 passes | fix and repeat; no timing |
| G1 E1 (night 0) | 80% upper bound of sigma_AA <= 8% (sigma-hat <= 6.8%) | no Mac timing; Linux |
| G2 R1 | build-proto equals build-fs bit for bit at defaults; telemetry sane (P-core clock >= 3.8 GHz) | fix the binary |
| G3 E1b | in-process A/A sets n = 3-6 by rule | rule; fail-safe to the X design |
| G4 live A/A | pooled B2/B CI contains 1; <= 2 of 10 per-dataset CIs exclude 1; sigma-hat <= 8% | no throughput ratio shown |
| G5 controls | Nf slower, SV faster, COLD detectably cold, placebo warm-up within ±3% | table 6.10 |
| G6 determinism | checksums, signatures, separability | stop |

**Headline predictions.**
- Memory: N +0 B/key, because this index stores keys in key order. C +0.80 B/key accounted, +1.0 real. G +528 B
  in total.
- Compression: CSV leaves the inputs untouched. fb's empty space is 21 outlier gaps, so G_key straightens fb only at
  k >= 21 (RMSE 57.7M to about 141k). Nm squeezes that same space at key level; at the root, which never sees the
  outliers, it only bends the line (planet better, fb about 2x worse).
- Throughput: SV beats every learned cell; Nf is slower; Cr and C are faster only where the root adopts virtual
  fences (stratum A); N ≡ B and C ≡ Cr within ±3%.

---

## 1. Claims and how each is measured

### 1.1 Labels (used in every table, figure and slide)

| code | label | note |
|---|---|---|
| B | BEFORE: SCALE-LI, learned raw root (`--root model`; falls back to binary by itself) | the baseline is never the binary root B0 (the learned root alone was 1.17x on fb) |
| N | NFL-style flow feature (stand-in 1-D trainer), key-ordered index, batch 1 | not NFL's z-ordered system |
| Nf | NFL-style flow forced in every region | positive control (known extra work) |
| Nm | monotone flow (key-order-preserving NFL variant) | D and CL only |
| Cr | CSV-style virtual fences at the root only (**our extension**) | not in the CSV paper |
| C | CSV Algorithm 1 in regions + CSV-style virtual fences at the root (our extension) | headline Cr vs B and C vs Cr separately, not C vs B |
| G | k-gap shrink-to-median map at the root | timed only if the go-rule (4.3) passes |
| G_key | k-gap shrink-to-median map over all keys | AIDB metrics only; no throughput consequence (4.3) |
| SV | plain binary search over a sorted vector (16 B/key) | positive control and the non-learned reference; lead with it |

**Scope slide** (stated, not tested): published baselines (ALEX, PGM, LIPP) need a download approval and are
post-meeting (PLAN D7); AIDB's published numbers are quoted as context, labelled as different hardware. Range scans
are out of scope: none of N, C, G touches the key-ordered scan path, and faithful NFL breaks scans (39-385x read
amplification, `design_faithful.md`). Single thread, uniform reads, 16 KiB pages, one Mac.

### 1.2 Metric definitions (fixed before any run)

**Memory.** Accounted bytes per key by component (`memory_before` / `source_rows` = 200,000,000): key, value,
metadata, delta, slack. Real bytes per key on fb by the steady-state RSS method (memory_audit.md 2b, E5). Peak RSS is
never used.

**Input-space compression (method-agnostic).** A method that is monotone in the key defines a coordinate
u = phi_m(x): identity for B, Cr and C (CSV changes targets, not inputs); g for G and G_key; the flow for Nm. Two levels,
same key set for every method:
- **L_root**: the 48,829 root fences exactly as `fit_root` sees them (fence 0 = first key; fence j = row 4096·j of
  the canonicalised, deduplicated key array; index.hpp:394,456).
- **L_key**: all 200M keys (the AIDB level).

Measured in each coordinate:
- (a) **Top-k gap share** S_k(m) = (sum of the k largest adjacent gaps of u) / (u_max - u_min), k in {1, 16, 64}.
  It is defined for any monotone map, because gaps map to gaps. **Compression** = the fall S_k(B) - S_k(m).
  For G, also the span fraction removed, D_k = sum over selected gaps of w_i(1 - s_i).
- (b) **Outcome**: RMSE/n and ME/n of one least-squares line of rank on u; PLA-eps segment counts (eps 32 and 4096
  at L_key, 1 and 8 at L_root); CD at L_key; root model probes at L_root (C++ `learnability` or the validated Python
  replica).
- (c) **Target side** (CSV): region rank SSE before/after (`learnability.rank_sse_*`) and the hardness tool's `csv`
  scope, labelled "hardness-tool CSV construct" because it differs from the index's CSV (osm: 10.23M virtual points
  in the tool vs 19.97M in the index).

The faithful flow (N) is not monotone, so S_k in key order is undefined. Its AIDB metrics stay on the separate z-sorted
panel of `hardness_ba.md`.

**Work per lookup.** Comparisons/op = (root_probes + gap_probes + coordinate_probes + fence_probes + key_at_calls) /
operations. `gap_probes` exists only with P3 and is reported separately from root probes, since an L1-resident table
comparison is not a root probe. Also reported: transforms/op and cache lines/op; the latter is never a contrast alone
(±3% reproducibility).

**Throughput.** P: ln(throughput) of each 1M-lookup timed visit, combined within a process by the model in 6.2.
X: ln(`throughput_ops_s`) of the whole 5M replay. Contrasts are ratios of geometric means.

**Build time.** `build_ns` (wall, 16 threads), median over all timed processes of the cell; `smoothing_ns` for CPU.

**Telemetry (P2).**
- Effective GHz = Δri_cycles / on-CPU ns.
- P-core fraction = Δri_pcycles / Δri_cycles.
- Runnable fraction = Δri_runnable_time / wall.
- Latency probe: ns per dependent load in a 512 MB Sattolo cycle.

Every time is converted from mach ticks with `mach_timebase_info`.

### 1.3 Consistency checks (true by construction; verified, not "proved")

| # | check | why it must hold |
|---|---|---|
| K1 | key bytes/key identical for N, Nf, Nm, Cr, C, NC, G vs B on all 10 (fb 1.9340, osm 4.7138, planet 1.384) | blocks are encoded before any model (index.hpp:213-238) |
| K2 | accounted B/key of N, Nf, Nm = B; the 144 B flow is reported as `flow_bytes` | the flow lives outside the accounting. **Clause for the memory slide:** zero because the index stores keys in key order; NFL's own z-ordered layout (not built) is estimated at +30% B/key (`design_faithful.md`: fb 10.50 to about 13.3) |
| K3 | G adds 32·k_eff + 16 B (528 B at k = 16, 2.6e-6 B/key); 0 when the root falls back to binary | table size |
| K4 | additivity: NC = C; GCr = Cr + table | levels are independent |
| K5 | S_k(C) = S_k(Cr) = S_k(B) at both levels | CSV does not touch inputs |
| K6 | \|ΔPLA-eps\| <= 2k for G and G_key at every eps | proof in 1.5 |
| K7 | `result_checksum` equal across cells sharing (dataset, trace segment) | read-only lookups, miss 0 |

### 1.4 Deterministic measurements, with predictions (D and CL layers; exact, no statistics)

| # | metric | comparison | prediction | basis |
|---|---|---|---|---|
| H3 | accounted B/key | C vs B | +8 B x virtual_points + 4 B x root slots = **+0.799 B/key** where every region fills its 409-point budget; root fences +0.001-0.005 | fb: 19,970,703 VP = 409 x 48,828 + 51. **Rests on fb and osm only**; the hardness tool left 5 datasets under budget, so the D layer decides on the other 8 |
| H4 | real B/key (RSS, fb) | C vs B | real(C) - real(B) = 1.00 ± 0.02 B/key (0.799 accounted + 0.20 hidden `virtual_features` capacity) | (512 x 48,828 + 64 - 19,970,703) x 8 B ≈ 40 MB |
| H7 | S_16 at L_root | B across datasets | gap concentration highest on osm, books, planet; Spearman >= 0.7 with the 2M top-16 shares (osm 0.60 > books 0.27 > planet 0.12 > ... > fb 0.036) | **2M regime** (`$SP/threeblock`), about 488 fences; labelled as an extrapolation |
| H8 | RMSE/n, ME/n at L_root | G (k*) vs B | fall only where S_16 is high; < 5% change on fb, history, stack, wise | as H7 |
| H8k | RMSE/n, ME/n, S_k at L_key, fb | G_key vs B, k in {1, 4, 9, 16, 20, 21, 32, 64, 256} | k <= 9: RMSE/n stays >= 50% of 28.87% (the 9 gaps of 2.05e18 remain); k = 20: RMSE near fb_trim (<= 2 x 140.7k) but ME/n >= 20% (a 3.87e10 gap remains); **k in {21, 32}: RMSE within ±10% of 140.7k and ME within ±20% of 324.9k (fb_trim)**; k in {64, 256}: RMSE <= 1.2 x 140.7k; 10 <= k <= 19 not predicted (transition) | fb's top 21 gaps, read from the file tail: 9 x 2.05e18, 5 x 1.41e14, 6 x 7.04e13, 1 x 3.87e10; the bulk spans 7.73e10 (`hardness_ba.md` F5) |
| H8m | same, osm, books, planet | G_key vs B, k in {1, 4, 16, 64, 256} | RMSE falls where S_16(L_key) > 0.1; PLA within the 2k bound | — |
| H10 | S_k, RMSE/n, ME/n | Nm vs B | L_key: reproduces `hardness_ba` (fb RMSE 57.7M to 473.5k through the outliers, ME 1.0e8 to 1.14e9; planet RMSE 0.54x; PLA-32 within 4 segments). L_root (no fb outliers): fb RMSE about 2x worse (fb_trim mono 2.1x), planet about 0.5-0.6x | `hardness_ba.md` F4/F5 |
| H10r | root decision | N, Nm vs B | N's flow is never adopted for virtual fences (non-monotone, index.hpp:406); NFL's switch accepts 2-5% of regions; tail conflicts worse (fb 31.32 to 38.89) | pilot `tables.md` |
| H11 | rank SSE | C vs B | region rank SSE falls on all 10 (hardness tool: fb region RMSE 318 to 265, osm 519 to 404) | targets, not inputs |
| H12 | comparisons/op | N vs B | fence ±0.01, root unchanged, transforms 0.02-0.05/op | pilot |
| H13 | comparisons/op | Nf vs B | +1.00 transform/op, fence +0.01, lines/op +4 to +7 | pilot |
| H14 | comparisons/op | Cr, C vs B | root probes fall where virtual fences are adopted (fb root estimate 10.80 to 3.50 at alpha 0.04 in the pilot); C fence -0.4 to -0.6; osm root falls back | pilot |
| H15 | root model probes, table comparisons | G vs B at L_root, k in {1, 4, 16, 64} (`fences_ba.py`) | **the go-rule (4.3) fails on all 10**: on fb nothing to remove; on osm the learned root is already rejected (raw 24.91 vs binary 15.66 probes) and G would have to cut more than 9 probes plus the table | 2M: G lowered model probes by up to 3.5 (osm-u, k = 64) but never paid for its table (`threeblock/RESULT.md`) |
| H18 | separability (only with P3) | GC vs GCr (root) and C (regions), fb | exact equality | levels independent |
| H19 | build time | each cell vs B | N +0.5-1.0 s; C 28-70x B (184-451 s); Cr +3.6 s (fb) to <= 18 s; G < 0.1 s | pilot, `run_ba.SMOOTH` |
| H20 | AIDB five at L_key | B, Nm, C (tool construct), G_key | separate panel; NFL-faithful only in its z-sorted convention | `hardness_ba.md` |

### 1.5 Why G and G_key cannot move PLA by more than 2k (K6)

g is affine except at the 2k breakpoints at the start and end of each selected gap. Split any optimal eps-PLA
segmentation of the original sequence at those breakpoints. An affine reparametrisation of x preserves a vertical
error bound, so this is a feasible segmentation of the transformed sequence: PLA_new <= PLA_old + 2k. g is invertible
with the same breakpoints, so PLA_old <= PLA_new + 2k. Only the global-line metrics (RMSE, ME) and CD can move.

### 1.6 Statistical hypotheses (timed; P layer primary, X layer as a check)

Families are tested separately, each at a familywise α of 0.05 (Holm), because each answers a separate question the
supervisor asked (6.6). The strata come from the D layer and are frozen before any timing (4.4): **A** = datasets
where the root adopts virtual fences or root probes change by >= 0.5; **F** = the rest.

| family | # | comparison | prediction (per-dataset values frozen in `predictions.json` before T) | test |
|---|---|---|---|---|
| NFL | H22 | N vs B, pooled over datasets where N's signature differs | 0.997-1.000 (0.02-0.05 transforms x 28-66 ns < 0.3% of a lookup) | equivalence ±3% |
| CSV | H24a | C vs B, stratum A | faster; fb 1.03-1.10 | superiority |
| CSV | H25a | Cr vs B, stratum A | faster, about as much as C | superiority |
| CSV | H25 | C vs Cr, all datasets | 0.99-1.01 | equivalence ±3%, plus the share (C - Cr)/(C - B) with a Fieller CI |
| G (only with P3) | H26a, H26b | G vs B; GCr vs Cr | from the 200M counters: equivalence ±3% where the predicted \|Δ\| < 1.5%, superiority where >= 1.5% | as stated |
| mechanism | H28 | meta-regression Δ_d = a + β·p_d over every timed (cell, dataset) pair except SV | β > 0 (β ≈ 1 if counters explain time) | one-sided t |
| mechanism | H24b | C vs B, stratum F | 0.99-1.01 | equivalence ±3% |

**Validity conditions** (no α spent; if one fails, table 6.10 applies):

| # | condition | prediction |
|---|---|---|
| V1 (H21) | live A/A, B2 vs B | gate of 6.7 |
| V2 (H23) | Nf slower than B, pooled | 0.85-0.96 |
| V3 (H27) | SV faster than B, pooled | 1.15-1.50 |
| V4 | placebo warm-up check: pooled first-chunk deficit within ±3% (X); per-cell first-sub-chunk deficit within ±3% of B's (P) | about 0 |
| V5 | COLD detectably cold: one-sided 95% upper bound of the pooled first-1/16 deficit below -5% | -6 to -20% (at 62.5-125k chunks: -13 to -32%) |
| V6 | determinism (6.9) | exact |
| V7 | X only: C vs B effective GHz within ±1% (TOST) and latency probe within ±1% | equal |

**Descriptive** (no test):
- E0h heat screen;
- the AIDB-length interaction ln(CA/BA) - ln(C/B);
- P vs X C/B;
- interactions as bounds;
- NC (D layer only before the meeting).

**SESOI.** δ = 3% pooled: an effect below 3% would not change which index we recommend. It is a decision threshold,
not a mechanism size; the mechanisms (one flow evaluation 28-66 ns, 4-8 root probes) are 1.6-5.5% of a 1.2-1.7 µs
lookup. Every null therefore also prints its **achieved equivalence bound** max(|L90|, |U90|), and every superiority
result says whether its 95% lower bound clears 3%.

**Pilot data** (seed 11, `$SP/fullscale`, `pilot_seed11.jsonl`) set the predictions. It is hypothesis-generating and
is excluded from every confirmatory analysis. Night-0 data (E1, E0h) are calibration, also excluded.

---

## 2. Machine and host

### 2.1 The Mac

Facts read on 2026-10-02 (`sysctl`, `pmset`, the SDK headers):

- **Model**: Mac15,9 (16-inch, M3 Max), 64 GiB.
- **Cores and caches**: 12 performance cores in two clusters of 6, each cluster with its own 16 MiB L2; 4 efficiency
  cores; L1D 128 KiB per P-core.
- **Pages**: 16 KiB.
- **Power**:
  - `pmset -g custom` shows `powermode 1` under both the AC and the Battery profile. On MacBook Pros with High Power
    Mode the key means 0 = Automatic, 1 = Low Power, 2 = High Power; confirm in System Settings > Battery.
  - `pmset -g batt` showed battery power at 6% during the review; at writing it showed AC, 7% and charging.
  - hostmon never logged the power state, so capped clocks cannot be ruled out for any past 200M timing.
- **Telemetry available**: the SDK has `RUSAGE_INFO_V6` (`ri_cycles`, `ri_pcycles`, `ri_runnable_time`,
  `ri_user_ptime`, `ri_pageins`). Times are in mach ticks: 125/3 ns per tick, so convert with `mach_timebase_info`.
- **Scheduling**: macOS cannot pin a thread. `--qos 1` biases the scheduler toward P-cores, but guarantees neither a
  P-core nor one cluster.

### 2.2 Quiet-Mac checklist (before night 0 and night 1; undo it afterwards, 8.6)

1. **Power** (G0).
   - Charge to at least 80% and stay on the adapter: `pmset -g batt` must say "AC Power".
   - `sudo pmset -c powermode 2` (High Power on the adapter); check `pmset -g custom` shows `powermode 2` in the AC
     block and System Settings > Battery > Energy Mode shows High Power.
   - The old check `pmset -g | grep lowpowermode` prints nothing on this host and is dropped.
   - Record `pmset -g custom` and `pmset -g batt` in `host.txt`. Room temperature in the log; lid open, flat on a
     hard surface, same position all night.
2. **Stop the UTM/QEMU VM.** Shut the guest down, quit UTM; `pgrep -fl 'qemu|UTM'` prints nothing. It was in the
   top 3 CPU users in 98% of samples during every 200M measurement so far, and P needs about 30 GB of RAM.
3. **Quit heavy and periodic apps**: browsers, ChatGPT, Slack, Docker Desktop, LM Studio/ollama, Xcode, VS Code,
   DaVinci Resolve, Orca, and any agent session that runs jobs (the `$SP/threeblock` arms must have finished).
   Check with `pgrep -fl -i 'docker|lmstudio|ollama|python3'` (only hostmon and the runner) and
   `pgrep -x scaleli_bench; pgrep -x scaleli_hardness` (nothing). Never match `'scaleli_'`: it also matches the
   source tree `scaleli_proto` and any editor command line.
4. **Sleep and idle work.** `sudo pmset -a powernap 0`. The runner is wrapped in `caffeinate -ims` (not `-d`: the
   display may sleep, so WindowServer stops compositing). Pause automatic macOS and App Store updates.
5. **Spotlight, Time Machine, iCloud.** `sudo mdutil -a -i off`; `sudo tmutil disable`; pause iCloud Drive and
   Photos sync.
6. **Monitoring**: start hostmon v2 (2.3) under `taskpolicy -b`.
7. **Hands off** during calibration and timing. Remote login only to read logs.

`uptime` showed "10 users" in the pilot (critique.md 2a). On this personal Mac that counts login sessions (terminal
tabs and agents), not other people. The load average of 5-178 came from jobs on this machine, which step 3 stops.

### 2.3 hostmon v2 (replaces `hostmon.py`; stdlib only)

The old monitor used `ps -Ao pcpu`, a decaying average over up to a minute: a 2 s burst at 400% shows as about 30%.

**Every 5 s:**
- `ps -Ao pid=,pri=,time=,comm=`; difference the cumulative CPU time per PID between samples, which gives exact
  CPU-seconds per interval.
  - `other_cpu`: all processes except `scaleli_bench`, `scaleli_hardness` and hostmon itself.
  - `bg_cpu`: the subset with `pri <= 4`, approximately background QoS, which still competes for SLC and DRAM
    bandwidth.
  - The top 3 foreign processes, with pid.
- `n_bench`: `pgrep -x scaleli_bench` plus `pgrep -x scaleli_hardness`.
- `vm_stat` deltas: pageins, pageouts, swapins, swapouts, "Pages compressed", "Pages decompressed".

**Every 60 s:**
- `pmset -g batt` (AC or battery, %) and the AC-block `powermode` from `pmset -g custom`.
- The thermal pressure level, from `notifyutil -g com.apple.system.thermalpressurelevel`. R1 verifies that it reads
  a value; if not, a 10-line C helper using `notify_register_check` replaces it.
- The load average.

**Dropped:** `pmset -g therm`, which carries no information on Apple silicon, and `kernel_task` as a thermal signal
(that was Intel idle injection).

**Output:** `--out $P/hostmon.jsonl`, one JSON line per sample.

### 2.4 Q0 pass rule (10 idle minutes; gate G0)

- Delta-based `other_cpu`: median <= 25%, p95 <= 60%, max <= 150%.
- No foreign process above 30% in more than 2 samples.
- None of qemu, UTM, Docker, LM Studio or ollama in any sample.
- AC power, `powermode` 2, thermal pressure "nominal" in every sample.

The load average is logged but not gated.

### 2.5 Validity of a timed unit (outcome-blind)

A unit is an X run's timed window, or one P visit (warm-up plus timed lookups). All rules use only telemetry and
hostmon, never throughput.

**Quiet gate before timing (inside the process, P1).**
- Mechanism:
  - After the cool-down, and in P before every round, the binary writes `<go-dir>/ready` and blocks on the FIFO
    `<go-dir>/go`.
  - The runner writes `go` only when the last 6 hostmon samples (30 s) all have `other_cpu` < 60%, top foreign
    < 30%, `n_bench` = 1, AC power, powermode 2 and thermal pressure nominal.
- Effect: every cell is gated at the same position relative to its window. In the draft the gate ran before the
  process started, about 50 s before B's window but about 350 s before C's.
- Time-out: after 600 s the binary proceeds and sets `go_timeout = true`.

**A unit is invalid if any of these holds:**

| rule | condition |
|---|---|
| (a) | a hostmon interval overlapping the unit has `other_cpu` >= 100% or a foreign process >= 50% |
| (b) | another `scaleli_bench` or `scaleli_hardness` was alive (`pgrep -x`) |
| (c) | P-core fraction over the timed window < 0.99 |
| (d) | runnable time > 1% of the timed window |
| (e) | Δ`ri_pageins` > 0 in the timed window, or host swapins > 0 or decompressions > 1,000 pages in it |
| (f) | AC lost, powermode changed, or thermal pressure not nominal during the unit |
| (g) | `go_timeout`, a non-zero exit or a timeout (a timeout is also written as a result row) |

The thresholds in (c) and (d) become E1b's 1st and 99th percentiles if E1b's median P-core fraction is below 0.995
or its median runnable fraction is above 0.5%. That rule is pre-registered.

**Retries (keyed on (dataset, cell or "P", block or replicate, attempt)).**
- X: an invalid run is retried immediately, after the quiet gate, inside its (dataset, block) segment; at most 2
  retries, then it is marked missing.
- P: an invalid visit is missing (stage 1 of 6.2 handles gaps). A process is invalid if B ∪ B2 has fewer than 6
  valid rounds or any cell has fewer than 5; it is retried immediately once (attempt 2).
- A unit counts as done when it is valid or its attempts are exhausted.

**Pause and stop.**
- If 3 of the last 10 units are invalid, the runner pauses until hostmon shows 10 quiet minutes, and logs an alert.
- On battery power the runner stops after the current unit.

**Blinding.** The runner logs status, validity and wall time only, never throughput (today's `run_ba.py` prints
`thr=`; that is removed). Nobody reads P or X throughput until the analysis code is frozen (5.0).

### 2.6 Host fingerprint (`$P/host.txt`, once per session)

- `sw_vers`.
- `sysctl -n hw.model machdep.cpu.brand_string hw.memsize hw.pagesize hw.perflevel0.physicalcpu
  hw.perflevel0.cpusperl2 hw.perflevel0.l2cachesize hw.perflevel0.l1dcachesize hw.perflevel1.physicalcpu`.
- `pmset -g custom`, `pmset -g batt`.
- `shasum -a 256` of both binaries; `/usr/bin/c++ --version`.
- The patch diff of `scaleli_proto` against `S`.
- `taskpolicy -G -p <bench pid>`, once per run, in the run's `meta`; it proves the bench is not clamped to
  background.

### 2.7 Linux x86-64 replication (the AIDB setting; post-meeting unless G1 fails)

Use a dedicated machine with no other users and no VM neighbours. x86 only: on aarch64 Linux `long double` is
software quad precision, which makes `LinearModel::fit_xy` far slower and the models different.

| item | setting |
|---|---|
| isolation | kernel cmdline `isolcpus=2 nohz_full=2 rcu_nocbs=2`; core 2's SMT sibling offline (`echo 0 > /sys/devices/system/cpu/cpuN/online`) |
| interrupts | `systemctl stop irqbalance`; `for f in /proc/irq/*/smp_affinity; do echo <mask without core 2> > $f; done` (the single `echo > /proc/irq/*/...` of the draft writes one file only) |
| pinning | new flag `--pin-core 2` (patch P-linux): the build runs its 16 threads on the housekeeping cores (they inherit the main thread's affinity), then the main thread calls `sched_setaffinity({2})` after the threads join and before the cool-down. Never `taskset -c 2` on the process: all 16 build threads would share core 2 (a CSV build would take about 16 x 300 s). Check `cpu-migrations` = 0 over the window |
| memory | `numactl --cpunodebind=0 --membind=0`; `sysctl kernel.numa_balancing=0` |
| frequency | `cpupower frequency-set -g performance`; turbo off (`intel_pstate/no_turbo` = 1, or `cpufreq/boost` = 0 on AMD); uncore (mesh) pinned with min = max through the `intel_uncore_frequency` sysfs or MSR 0x620, since it sets memory latency; `cpupower idle-set -D 2` |
| pages | THP `never` (AIDB: hugepages disabled) for the primary run; a THP `always` sensitivity arm for SV and B |
| ASLR | on |
| perf | version >= 5.9 (for `--control`), recorded |
| cells | B, B2, N, Nf, Cr, C, SV, RAW on fb, osm, planet, books, n = 5. SV and RAW are what 4 KiB pages may change. Linux carries the AIDB-comparable absolute numbers and the SV-vs-index ranking |

Counters are scoped to the timed window by P4 (`--perf-ctl FIFO`) and listed in section 9.

### 2.8 Which result needs which machine

| result | Mac, any light load (never beside timing) | Mac, quiet | Linux x86, isolated |
|---|---|---|---|
| memory (accounted), key bytes, compression (CL), probes, transforms, root decisions, AIDB five | **primary** | — | cross-check only; sub-probe differences expected (80-bit `long double` on x86) |
| real RSS (E5) | — | **primary** | — |
| throughput (P, X) | — | **primary if G1 passes** | **primary if G1 fails**; otherwise a replication on 4 KiB pages |
| absolute Mops, "same protocol as AIDB" | — | — | **required** |
| hardware-counter mechanism (section 9) | — | with sudo: kperf / xctrace CPU Counters | Linux perf, more events (MLP) |

---

## 3. Warm-up and cool-down

### 3.1 What the warm-up must achieve

The timed window must start in the state that a long, uninterrupted stream of uniform lookups converges to. What
settles, and how fast:

1. **The hot set that is reused often enough to stay resident**, about 2-3 MB in all:
   - root fences and model, about 390 KB;
   - `regions_`, 48,829 x 8 B = 390 KB;
   - the last-level page tables of the index, 2.1-2.8 GB / 16 KiB x 8 B = 1.0-1.4 MB;
   - code and the branch-predictor tables.

   These settle within about 1e4-1e5 lookups.
2. **The structures that are not resident, by design.**
   - Region headers (13.7 MB), block descriptors, key arenas and values are DRAM misses in steady state.
   - Between two visits to the same region, a uniform stream brings in R x (21-50 lines/op) = 65-156 MB, far above
     the 16 MiB L2 and the system-level cache.
   - So the draft's coupon-collector bound (W >= R ln(100R) = 0.75M) did not describe cache state; it is withdrawn.
3. **LRU turnover**: the L2 turns over in 15-25k lookups and the SLC in about 1e5.
4. **Branch predictors**: 1e4 lookups of the same code path. Warm-up keys come from the same generator with a
   different seed (`seed ^ 0x5741524d`), so they train the same paths without replaying the timed keys.
5. **DVFS**: settles within milliseconds; P2 measures it per chunk.
6. **Pages**: all index pages were written during the build and are resident. Prefault moved 6 of 1.1M minor faults
   in the idle replication, so the cold transient is **not** a paging effect. Prefault stays on as harmless
   insurance with no claimed effect. Paging is detected directly (2.5 e).
7. **Thermal state equal across cells**: not the warm-up's job (3.6).

Because 1-5 have no exact model, **the warm-up length is set from measured transients**.

### 3.2 Evidence

- **Idle replication** (`$SP/verify/A2`; fb, `--warmup 0`, 2M ops, 16 chunks of 125k; n = 6 per arm; no peer job;
  re-analysed in `review_systems.md` 1.1):

  | | prefault 0 | prefault 1 |
  |---|---|---|
  | chunk 1 (0-125k), median | -19.4% (+3.0 to -32.0) | -5.8% (3 of 6 runs at -15 to -17%) |
  | chunk 2 (125k-250k) | -11 to -21% in all 6 runs | -10 to +19% |
  | first 1/16 at the X layer's 312.5k chunks (chunks 1, 2 and half of 3, against the median) | **median -14.4%** (-6.2 to -20.4) | median -3.7% (-8.0 to +5.2) |
  | minor faults | 1,104,188 | 1,104,194 |

  The prefault effect is not significant (Mann-Whitney U = 8, critical value 5). **The transient lasts at least
  250k and at most about 375k lookups.**
- **Contended set** (`$SP/fullscale` cold, n = 3, a peer 200M job alive in 237 of 241 samples): -34% in chunk 1.
  Kept as a footnote only; the idle data replace it.
- **2M workload-mode warm-up** (pilot, 24 runs, contended, prefault on in all 24): c1 = -0.3% (SE 2.0%).
- **A state flip is not a cold cache.** Pilot `R0sv` on fb had c1 = -0.305 after a 2M warm-up. P-core fraction,
  runnable time and the placebo test (3.5) separate the two.
- **`steady_state_ratio` (last/median) is broken.** It read 1.039 and 0.980 on runs whose first chunk was -34%.
  Not used.
- **Ops per process**: sigma was 6.4% at 2M ops, 6.2% at 5M and 6.1% at 10M. The noise is per process, so the design
  spends on processes and rounds, not on longer replays.

### 3.3 The sequence inside each timed process

**X layer (one cell per process; also the AIDB-length arm and COLD):**

```
load file + canonicalize + workload (~40 s)    ; latency-probe buffer (512 MB) allocated here, common to all cells
build (16 threads; build_ns)
learnability + memory_before
COOL-DOWN 60 s (sleep)                          [P1; identical for every cell]
GO-GATE (FIFO; runner writes go after 30 quiet seconds)   [P1]
snapshot #1: rusage V6, task events, host vm stats
probes (pre): LCG frequency probe (50M dependent ops) + latency probe (3M dependent loads)
prefault (one byte per page of every index allocation; insurance)
WARM-UP: 4,000,000 workload-mode lookups, seed ^ 0x5741524d     (COLD: 0, and prefault 0)
snapshot #2
TIMED: 5,000,000 lookups in 80 chunks of 62.5k; per chunk: wall, cycles, pcycles, instructions, user/sys ns, runnable ns
snapshot #3; probes (post)
```

**P layer (all cells in one process):**

```
load + canonicalize + workload; trace = 8 timed segments of 1M keys + 8 warm-up segments of 1M keys
build every cell in a random order (seed-derived; logged); per cell: build_ns, learnability, memory_before; prefault each
COMMON COOL-DOWN 60 s
for round r = 1..8:
    GO-GATE; probes (pre-round)
    for cell in a fresh random permutation (seed, r):
        snapshot; WARM-UP 1,000,000 lookups of warm-up segment r (the same keys for every cell in the round)
        snapshot; TIMED 1,000,000 lookups of timed segment r (the same keys for every cell in the round), 16 sub-chunks; snapshot
    probes (post-round)
```

### 3.4 Every number, and why

| parameter | value | why |
|---|---|---|
| warm-up mode | `workload` | same generator and distribution as the trace, different seed; `strided` warms a pattern no measured workload uses |
| X warm-up | **4,000,000 lookups** (4-5.5 s on learned cells) | >= 10x the 375k transient; the hot set needs <= 1e5. It is also PLAN.md's value, so pilot comparisons stay valid. W may only go up (to 8M) if V4 fails |
| P re-warm per visit | **1,000,000 lookups** | >= 2.7x the cold transient from a fresh process. A re-visit is warmer than a cold start: pages and page-table leaves were touched in earlier rounds, and only the 2-3 MB hot set was evicted by the previous cell |
| prefault | 1 (COLD: 0) | insurance with no claimed effect (3.1 item 6) |
| cool-down | **60 s fixed**, every cell, both layers | no calibration (T1 removed, 3.6). In P it only matters that it is common; in X it is a conservative prior (laptop heat sinks and fans settle within 1-2 min) |
| X timed ops | 5,000,000 in **80 chunks** (62.5k) | chunk 1 is 6-17% of the transient's length, so the placebo test sees a failure; re-aggregated to 16 chunks (312.5k) for continuity with PLAN.md and COLD |
| P timed ops per visit | 1,000,000 in 16 sub-chunks (62.5k) | 8 rounds give 8M timed lookups per cell per process, against 5M per X run |
| rounds R | 8 | the per-process mean is averaged over 8 round-paired contrasts; the cost is about 2.7 s per visit |
| statistic | X: whole 5M replay; P: whole 1M visit | pre-registered; first chunk or sub-chunk excluded only under the V4 rule |
| seeds | X: 1000 + block; P: 2000 + replicate; D: 1001 | D's trace equals X block 1's, so their checksums must match |

### 3.5 Checks that prove the warm-up worked

- **Placebo-chunk test.** It replaces c1 = chunk1/median(rest) - 1, which is biased because chunk rates are skewed.
  - For each run (X, 16-chunk aggregation) or visit (P, 16 sub-chunks), compute c_j = r_j / median(r_{-j}) - 1 for
    every chunk j, then w = c_1 - mean(c_2..c_16). By exchangeability E[w] = 0 in steady state.
  - X: the pooled mean w over all warm X runs must be within ±3% (TOST, run-level SE).
  - P: each cell's mean w must be within ±3% of B's (TOST, process-clustered SE). A per-cell difference is what
    could bias a contrast.
  - On failure, the primary statistic becomes chunks 2-16 (X) or sub-chunks 2-16 (P) for **every** cell, and the
    failure is reported.
- **COLD reference** (X layer, fb, osm and planet, all 3 blocks, 9 runs): V5 passes if the one-sided 95% upper bound
  of the pooled first-1/16 deficit is below -5% (prediction -6 to -20%; at 62.5k chunks -13 to -32%). With n = 9 a
  wrong "insensitive" verdict has probability below 0.04.
- **Re-warm curves** (P): the mean sub-chunk curve per cell, over all visits, is figure 2b.
- **Per-unit flags.** Units with w < -20% are flagged, never dropped. Flags are cross-tabulated with P-core fraction
  and runnable time to separate state flips from cold caches.
- **Migration dips.** With 80 chunks, the number of single-chunk dips per X run (likely moves between the two
  P-clusters) is reported as a diagnostic.

### 3.6 The build-heat confound

A CSV build runs 160-450 s at 100% of all cores, a B build 7 s, and in a separate-process design the timed pass starts
right after the build.

| layer | how the confound is handled |
|---|---|
| **P (primary)** | **removed by design**: every cell is built before one common cool-down, then cells are timed in random order within each round, so every cell's windows share the same thermal history, including C's build heat |
| X (secondary) | fixed 60 s cool-down for every cell; in-process go-gate; thermal pressure must be nominal (2.5 f). Measured, not assumed: effective GHz (V7, C vs B within ±1%) and the latency probe (V7, ±1%), which sees fabric, DRAM and TLB-walk state that core GHz misses; covariates S7 (ln GHz, ln latency) and S8 (predecessor's cell and build CPU-seconds). X's C/B is never pooled with P's |
| night 0, E0h | the size of the confound on throughput itself, with the existing binary: B on fb immediately after a books C-cell process (a 441 s all-core build, the worst case) vs B after 300 s of idle; 6 pairs, alternating, paired analysis. Descriptive; it also tells whether the pilot's per-process C/B could have been heat-biased |

The draft's T1 calibration (c* from {0, 30, 120} by GHz and cycles/op, n = 2) is removed. Core GHz is probably flat on
this chassis. cycles/op = ns/op x GHz is clock-dependent for memory-bound code. n = 2 made the choice random, and the
tested arms did not match the selectable ones. A state-gated cool-down (wait until the SoC temperature is within 2 °C
of its pre-build value) is an option, not the default (DECISIONS, item 9): it needs about 1 h of IOHID sensor code.

### 3.7 Deviations from AIDB's protocol (section 4.1: 20M warm-up, 100M measured, one pinned thread, x86 Linux)

| item | AIDB | here | why |
|---|---|---|---|
| warm-up | 20M lookups | X: 4M; P: 1M per visit | transient <= 375k; the AIDB-length arm tests whether this changes C/B |
| measured | 100M | X: 5M per process x n; P: 8 x 1M per cell per process x n | the noise is per process; more processes and rounds, not longer replays |
| placement | pinned core, Xeon Gold 5118, hugepages off | QoS 1, no pinning, 16 KiB pages | the machine available; Linux (2.7) carries AIDB comparability |
| lookups | "random lookups for all keys in the dataset" (draws or a permutation; ambiguous) | uniform draws with replacement over all 200M keys, miss 0 | at 100M of 200M the two differ negligibly for cache behaviour; stated on the slide |
| bulk load | all keys | `--load-ratio 1` | same |

**AIDB-length arm (X layer, night 1).**
- Cells: BA = B and CA = C with `--warmup 20000000 --ops 100000000 --chunks 100`, on fb, n = 3 each, separate
  processes as AIDB does.
- Estimate: the protocol x cell interaction ln(CA/BA) - ln(C/B_X) with its 95% CI, **descriptive**. At n = 3 its SE
  is about 3.5% at sigma 4%, so an equivalence test at ±3% has no power; the slide says so.
- Absolute Mops never cross machines.

---

## 4. The cells

### 4.1 Common flags

```
COMMON = --format sosd --dtype uint64 --load-ratio 1 --miss 0 --query-distribution uniform --profile read_only
         --qos 1 --build-threads 16 --verify 0 --latency 0
PK     = --policy min_bytes --routing rank
F(d)   = $S/results/aidb_flowv2/flows_free/<d>_<s512|s64>_t2000.txt   (s512 for fb, osm, planet; run_ba.PERIOD)
M(d)   = $S/results/aidb_flowv2/flows_monotone/<d>_mono.txt
data(d)= $S/data/external/gre/<d>   (<d>.sorted for covid genome history libio planet stack wise)

D (BFS) = COMMON --ops 5000000 --warmup 1000000 --warmup-mode workload --prefault 1 --chunks 16 --instrument 1 --seed 1001
X (BPR) = COMMON --ops 5000000 --warmup 4000000 --warmup-mode workload --prefault 1 --chunks 80 --instrument 0
          --seed $((1000+block)) --cooldown-s 60 --go-dir $P/go --cycles 1 --probe-latency-mb 512 --freq-probe 50000000
P (BPR) = COMMON --instrument 0 --seed $((2000+rep)) --interleave $P/cells/<d>.tsv --rounds 8 --visit-warmup 1000000
          --visit-ops 1000000 --sub-chunks 16 --prefault 1 --cooldown-s 60 --go-dir $P/go --cycles 1
          --probe-latency-mb 512 --freq-probe 50000000
```

The runner pads the environment (`SCALELI_PAD=xxx...`) so that len(argv) + len(env) is the same 8,192 bytes for every
X and D cell. Without it, N's long flow path shifts the initial stack, a pure layout artefact (Mytkowicz et al.,
ASPLOS 2009). In P, all cells share one argv.

### 4.2 Cell table

| cell | flags (added to the layer's line) | label (1.1) | D | CL | P | X |
|---|---|---|---|---|---|---|
| **B** | `PK --root model` | BEFORE | all 10 | ✓ | all 10 | fb, osm, planet |
| **B2** | as B, built separately | live A/A | — | — | all 10 | fb, osm, planet |
| **N** | B + `--flow F(d) --flow-bypass 1 --flow-cost 0` | NFL-style flow feature | all 10 | — | all 10 | — |
| **Nf** | B + `--flow F(d) --flow-bypass 0 --flow-cost 0` | flow forced (positive control) | all 10 | — | all 10 | — |
| **Nm** | B + `--flow M(d) --flow-bypass 1 --flow-cost 0` | monotone flow | all 10 | ✓ | — | — |
| **Cr** | B + `--root-alpha 0.1` | CSV-style root fences only (our extension) | all 10 | — | all 10 | — |
| **C** | B + `--virtual-alpha 0.1 --root-alpha 0.1` | CSV Alg. 1 in regions + root fences (our extension) | all 10 | ✓ | all 10 | fb, osm, planet |
| **C2** | as C, built separately | expensive-build A/A; CSV rebuild determinism | — | — | fb | — |
| **NC** | C + `--flow F(d) --flow-bypass 1 --flow-cost 0` | NFL + CSV | all 10 | — | session 2 | — |
| **SV** | `--index sorted_vector` | plain binary search (positive control) | all 10 | — | all 10 | fb, osm, planet |
| **RAW, RC** | `--policy raw --routing rank --root model` (+ `--root-alpha 0.1`) | codec reference (memory only) | fb | — | — | — |
| **B0** | `PK` | binary root, host default | fb | — | — | — |
| **COLD** | B with `--warmup 0 --prefault 0` | cold reference | — | — | — | fb, osm, planet, 3 blocks |
| **BA, CA** | B, C with `--warmup 20000000 --ops 100000000 --chunks 100` | AIDB-length arm | — | — | — | fb, n = 3 |
| **G, GCr** | B / Cr + `--root-gaps k*(d) --root-gaps-mode force` | k-gap map at the root | if P3 | prototype | if P3 | — |
| **GC** | C + `--root-gaps k*(d) --root-gaps-mode force` | separability (H18) | if P3, fb | — | — | — |
| **G_key** | hardness tool `--gap-key 1,4,16,64,256` (fb also 9,20,21,32) | k-gap map over all keys | — | ✓ | never | — |

**Cut from the draft:**
- the C++ k sweep G1-G256 and the selector cell Gs (the k sweep runs in Python on the fences);
- RAW and RC timing (compression is not about codecs; they stay in D as the memory reference);
- C4;
- the alpha sweep (session 2);
- NG and NCG.

**P memory per process:**
- 5 B-type indexes at 2.0-2.8 GB, C 2.2-3.0 GB, SV 3.2 GB, the loaded data 3.2 GB, trace and warm-up keys 0.3 GB;
  plus about 6.4 GB transient during each build.
- That is roughly 23-33 GB, plus 2.5 GB for C2 on fb and about 5 GB for G and GCr.
- It fits in 64 GB only with the VM stopped. R1-P6 records `phys_footprint` and requires zero swapins.

**Controls:**
- Negative: B2, C2, and every structural null (a cell whose D-layer signature equals B's on that dataset). Under P a
  structural null costs one cheap build and 8 visits (about 30 s per process), so it is kept as an extra A/A but
  excluded from every pool.
- Positive: Nf, SV, COLD.

### 4.3 Gap removal

**4.3.1 Root level at 200M, before any C++ (`fences_ba.py`, new; CL layer).**
- **extract**: writes `fences/<d>_raw.sosd` (48,829 keys).
  - If the D layer reports `unique_rows` = 200,000,000, it seeks to rows 4096·j. Otherwise it streams the sorted
    file and deduplicates (stdlib, about 3 min per dataset), because `canonicalize` removes duplicates
    (types.hpp:80-86) and fences index the deduplicated array.
  - Fence 0 is the first key (index.hpp:394).
- **gap --k 1,4,16,64**: runs `threeblock.gap_block` and `score` (the validated `fit_root` replica) on the 200M
  fences. Writes `fences/<d>_gaps.json`, with root model probes, table comparisons ceil(log2(k_eff+1)), D_k and S_k
  per k, and `fences/<d>_g<k>.sosd` (keys = round(g(x)·2^62), strictness asserted).
- **validate**: the replica's raw-root probes must equal the C++ `learnability.root_probes_raw` of the D-layer B run
  to ±0.01, and its binary cost `root_cost_binary`. A larger gap stops G reporting until it is explained.

**4.3.2 Go-rule and k selection** (frozen now; evaluated on `fences/*_gaps.json`; result in `$P/gorule.json`, hashed
before night 1).
- Price: a table comparison costs p = 0.25 root-probe equivalents. It touches a 528 B L1-resident array, while a root
  probe touches the 390 KB fence array. A sensitivity at p = 1 (the prototype's price) is reported.
- Per dataset: cost_G(k) = model probes(k) + p·ceil(log2(k+1)), and cost_B = min(raw model probes, binary probes),
  i.e. the root B actually uses.
- k*(d) = the argmin over k in {1, 4, 16, 64} of cost_G(k); ties go to the smaller k.
- **Go** if on at least one dataset cost_G(k*) <= cost_B - 1.0. That is, G must save at least one root probe
  equivalent, either by making the root adopt a model where B falls back, or by cutting model probes by more than
  its table.
- No-go (predicted, H15): P3 is not built. G is reported from the 200M counters (exact), and G's throughput as "not
  timed; the counters predict at most x%", with x computed from the price model in 4.4.

**4.3.3 P3: gap removal at the root in C++** (only on go; `include/scaleli/index.hpp`, `types.hpp`, `src/benchmark.cpp`;
4-6 h + 1-2 h validation).
1. `Config`: `std::size_t root_gaps = 0`; mode `Force` only (the selector variant Gs is cut).
2. `fit_root`:
   - Build the table on the raw feature x_j = tmp.normalized(fences[j]) by porting `threeblock.GapMap` exactly.
     Widths w_j = x_{j+1} - x_j; W_med = their median; take up to k largest gaps with w > W_med (ties to the lower
     index); s = W_med / w. Store sorted arrays a, w, s, C (cumulative shrink below), D, and scale = 1 - D.
   - g(x) = (x - C_j - (1 - s_j)·min(x - a_j, w_j)) / scale, with j the last a_j <= x, else x / scale. Assert it is
     strictly increasing on fences and fence midpoints.
   - Force mode offers the gapped candidates {g-raw ranks, g-raw vp} plus the binary fallback. The vp candidate runs
     `smooth_cdf` on g-features (monotone, so index.hpp:406 admits it): the single pass G then V.
   - Binary fallback clears the table.
3. `root_feature`: binary search over a (at most ceil(log2(k+1)) comparisons, each `++s->gap_probes`), then g(x).
4. Accounting: metadata += 32·k_eff + 16 B; `for_each_allocation` includes the table; `QueryStats.gap_probes`.
5. Output: flags `root-gaps`, `root-gaps-mode`; JSON `learnability.root_gaps`, `root_gap_shrink`,
   `root_probes_gap_raw`, `root_probes_gap_vp`; `work_counters.gap_probes`.
6. Correctness is automatic, since `locate_from_prediction` ends in an exact fence search: the checksum must equal B's.
7. Gate R1-G: C++ `root_probes_gap_raw` equals `fences_ba.py` model probes at 200M to ±0.01 for every k*(d), and the
   prototype at 2M to 1e-9 for 4 samples x k in {1, 4, 16, 64} (root alpha 0); `--verify 1` at 2M on all 10 samples.

**4.3.4 G_key: gap removal over all keys** (patch P5k in `src/hardness.cpp`, about 1-1.5 h; CL layer).
- `--gap-key K1,K2,...`:
  - on the tool's features (double(key - min), or the flow output with `--flow`), compute the adjacent gaps and
    their median (via `nth_element` on a copy; +1.6 GB);
  - select the top-K gaps with w > median (min-heap, O(n log K), ties to the lower index);
  - apply the same shrink-to-median map; assert u is non-decreasing (the double features of osm already contain
    ties);
  - run the existing metric block on u.

  JSON block `gapped[k]` holds k, k_eff, D_k, S_1/S_16/S_64 before and after, RMSE, ME, PLA-eps and CD.
- `--gap-share 1,16,64` adds S_k to every block (original, transformed, smoothed, gapped).
- `--fence-file` needs no change: fence files are ordinary SOSD files.
- **No throughput consequence.** fb's 21 outliers are rows 199,999,979 and above, all above the last root fence (row
  199,999,488), inside the last region (512 keys). The root never sees them, and the last region receives
  512/2e8 = 2.6e-6 of uniform lookups. The slide says this, so the AIDB panel does not imply a speed-up.

**4.3.5 What the 2M prototype shows (a different regime, labelled).** About 488 fences per sample, root model 2.2-8
probes; k = 16 is 3% of the gaps at 2M and 0.03% at 200M.
- G never paid for its table on any total: 0 of 160 arm A cells and 0 of 80 arm B cells (`threeblock/RESULT.md`).
- Largest model-probe cut: osm-u k = 64, 8.18 to 4.67, against a table of 7.
- Back-and-forth alternation beat a single pass by more than its instability band on only 2 of 20 samples, both
  without G. So G then V is a single pass.

**Region-level gap removal is out of scope.** The region coordinate step is a fixed 32-way search (index.hpp:126-133),
so model quality reaches only the fence term (about 5 of 29 comparisons).

### 4.4 Strata and per-dataset predictions (written from D and CL, hashed before night 1)

- **`strata.json`.**
  - A = {d : Cr's D-layer `root_vp` = true and `root_virtual` > 0, or |root probes/op(Cr) - root probes/op(B)|
    >= 0.5}; F = the rest. C has the same root as Cr (`fit_root` reads fences only), so C uses the same strata.
  - Signatures per (cell, d): accounted bytes, `flow_regions`, `virtual_points`, `root_model`, `root_flow`,
    `root_vp`, `root_virtual`, `root_gaps`, root, gap and fence probes/op. A cell is a **structural null** on d if its
    signature equals B's.
- **`predictions.json`**, per timed (cell, d):
  - the counter difference ΔW = c_cmp·Δ(root + coordinate + fence + key_at)/op + c_tab·Δgap/op + c_tr·Δtransforms/op,
    with c_cmp = 6 ns (range 4-8), c_tab = 1 ns, c_tr = 45 ns (range 28-66);
  - p_d = -ln(1 + ΔW / T0), with T0 = 1.40 µs (pilot fb B0: 1/0.731 Mops);
  - the interval from the range ends.

  p_d is the covariate of H28. The point predictions are also the "counter prediction" printed beside every
  estimate.

---

## 5. Schedule

### 5.0 Pre-registration freeze

1. **Freeze 1, before night 0.**
   ```
   mkdir -p $P/prereg && cp $SP/protocol/PROTOCOL.md $P/prereg/
   shasum -a 256 $P/prereg/PROTOCOL.md > $P/prereg/freeze1.sha256
   ```
   This copies the protocol out of `/private/tmp`, which is volatile.
2. **Freeze 2, before night 1 starts timing.** `strata.json`, `predictions.json`, `gorule.json`, `run_ba.py`, both
   binaries, the P cell files and the plan files are hashed into `$P/prereg/freeze2.sha256` and into every plan
   header.
3. **Analysis code.** `analyze_ba.py` v2 may be finished after night 1, but before anyone reads P or X throughput. Its
   hash goes into `freeze3.sha256`, and only then is `report_t.md` generated. The runner never prints throughput
   (2.5).
4. **Amendments** after freeze 1 are appended as dated entries in `$P/prereg/AMENDMENTS.md`, with a reason, and are
   always declared on the slides. Rule-driven choices (n, go-rule, strata) are not amendments.

### 5.1 Stages

| stage | what | eng (h) | machine (h) | host |
|---|---|---|---|---|
| S0 | host fix (2.2 items 1-2), freeze 1 | — | — | — |
| E-a | hostmon v2; `run_ba.py` presets `n0_aa`, `n0_heat`; E5 fix | 1-1.5 | — | — |
| N0 | night 0 (5.2) | — | 2.3 | quiet |
| E-b | `fences_ba.py`, P5k, `run_ba.py` preset `d`, D/CL tables in `analyze_ba.py` | 2.5-3.5 | — | — |
| D | D layer (5.3) | — | 5.1 | any light load; never beside timing |
| CL | compression layer (5.4) | — | 1.0-1.5 | same |
| E-c | build-proto P1, P2, P6; runner v2 (5.5) | 7.5-9.5 | — | — |
| E-d | P3, only if the go-rule passes | 5-8 | — | — |
| N1 | night 1 (5.6) | — | 8.7 (+0.7 with G) | quiet |
| N2 | night 2, only if E1b's rule gives n > 3 | — | 1.75 per extra replicate | quiet |
| A | `analyze_ba.py` v2, freeze 3, report | 3 | minutes | — |

**Tier M** is S0, E-a, N0, E-b, D and CL: about 4-5 h of engineering, 2.3 h quiet and 6.6 h any-load machine time.
**Tier C** is all of it.

### 5.2 Night 0 (existing binary `build-fs`; no patch needed)

| step | runs | settings | time | decides |
|---|---|---|---|---|
| Q0 | hostmon only, 10 min idle | 2.4 | 10 min | G0 |
| E1 A/A | fb, B x 10 and B2 x 10, random order (shuffle seed 20261002) | `COMMON --ops 5000000 --warmup 4000000 --warmup-mode workload --prefault 1 --chunks 80 --instrument 0 --seed 1001` + `PK --root model` | 20 min | G1 (6.7) |
| E0h heat screen | 6 pairs, arm order random within each pair: (i) heater = books C cell (`--ops 1000 --warmup 0`, a 441 s 16-thread build), then **B on fb immediately**; (ii) 300 s idle, then B on fb | B as in E1 | 91 min | descriptive (3.6) |
| E5 RSS | `rss_ba.sh` on fb, B and C, 10M ops | fixed script (8.4) | 15 min | H4 |

- **E1 outputs:** σ̂_AA (pooled within-cell SD of ln throughput, df 18), 1.4826 x MAD, and the one-sided 80% upper
  bound U80 = σ̂·sqrt(18/12.857) = 1.183·σ̂.
- **G1:** GO for Mac timing iff U80 <= 8%, i.e. σ̂ <= 6.76%. Otherwise Mac timing stops: Tier M is reported and
  throughput moves to Linux.
- **E1 also records** the per-chunk SD and the chunk-CV state classes (bimodality, critique 2a). They are
  descriptive, and they set the expectation for E1b.
- **E0h estimate:** the mean of the 6 paired differences ln B_heat - ln B_idle with its 95% CI. It resolves only
  large effects (about 6% or more at σ 4%). It is a screen: a null here does not prove absence. P removes the
  confound anyway.

### 5.3 D layer (`build-fs`, daytime)

- One run per (cell, dataset):
  - cells B, N, Nf, Nm, Cr, C, SV and NC on all 10 datasets; RAW, RC and B0 on fb;
  - plus G, GCr on all 10 and GC on fb, if P3 exists (then on `build-proto`).
- Cost: 74 s per cheap run, 110 s for Cr, about 671 s for C and NC (60 s + 2 builds), so 5.1 h; with G cells 5.7 h.
- `run_ba.py` writes `d.jsonl` with `throughput_ops_s` and the chunk rates **renamed** to
  `untimed_throughput_ops_s` and `untimed_chunks`. The D runs are instrumented, run under any load and have the wrong
  warm-up, so they can never enter a throughput table.
- Then: `strata.json`, `predictions.json` (4.4), and check 6.9-1 (D checksums equal X block 1 on fb, osm, planet),
  which runs after night 1.

### 5.4 CL layer (after D; minutes to an hour; never beside timing)

1. `fences_ba.py extract`, `gap --k 1,4,16,64`, `validate --det d.jsonl`, then `gorule`, which writes `gorule.json`.
2. For every fence file (raw, g<k*>, and the Nm coordinate via `--flow M(d)`):
   `scaleli_hardness --data <f> --dtype uint64 --pla-eps 1,8 --gap-share 1,16,64 --check-sorted 1 --threads 16`.
3. For every dataset at L_key (one invocation each; about 2-3 min):
   `scaleli_hardness --data data(d) --dtype uint64 --flow M(d) --gap-key 1,4,16,64,256 --gap-share 1,16,64
   --pla-eps 32,4096 --threads 16`. fb adds `--gap-key 9,20,21,32`.
4. Complete the hardness CSV scope: `$SP/nfl_ba/run2.sh csv stack wise`, plus fb with `--limit 199999979
   --virtual-alpha 0.1` (about 15 min).

### 5.5 Engineering E-c (before night 1)

In a copy of the sources: the git root is the home directory, so copy rather than branch, which leaves `build-fs` and
`S`'s MANIFEST untouched. Patches are specified in 8.1; the runner and analysis changes in 8.2-8.3.

| item | effort |
|---|---|
| P1 cool-down + go-dir FIFO gate | 0.5-1 h |
| P2 telemetry: V6 rusage with timebase conversion, phase snapshots, latency and LCG probes, 80 chunks, `--selftest-telemetry` | 2-2.5 h |
| P6 interleave harness | 3-4 h |
| runner v2: P and X jobs, attempt keys, go-writer thread, env pad, validity evaluation, blinding | 2 h |

### 5.6 Night 1 (quiet, unattended after calibration)

| order | step | time | decides |
|---|---|---|---|
| 1 | Q0 | 10 min | G0 |
| 2 | **R1**: fb and osm x {L0 = B, C2p = B + `--root-alpha 0.04`} at pilot settings (seed 11, `--warmup 2000000 --prefault 1`, 5M ops, instrument 1, cool-down 0), on BPR. Must equal `pilot_seed11.jsonl` in `result_checksum`, `trace_fingerprint`, `memory_before`, `learnability` minus `*_ns`, and `work_counters` minus `cache_lines*`. Plus `--selftest-telemetry`: on a 2 s busy loop, on-CPU fraction 1.00 ± 0.01, P-core fraction >= 0.99, effective GHz >= 3.8 and within 5% of the LCG probe, and the thermal-pressure read works | 9 min | G2 |
| 3 | **R1-P6**: one fb P process, all P cells, R = 1 | 9 min | per-visit checksums equal across cells; per-cell `memory_before` and `learnability` equal the D layer; `phys_footprint` <= 40 GB; zero swapins |
| 4 | **E1b**: fb, 5 P processes with cells {B, B2} only, R = 8 | 13 min | G3: n (below) |
| 5 | P replicate 1 (10 datasets, order shuffled per replicate) | 1.75 h | — |
| 6 | X block 1 (datasets shuffled; cells shuffled within a dataset and contiguous) | 43 min | — |
| 7-10 | P replicate 2, X block 2, P replicate 3, X block 3 | 4.9 h | — |
| 11 | BA/CA on fb (3 + 3, random order) | 37 min | — |

**E1b rule (G3).**
- s_P = the SD over the 5 processes of the per-process A/A contrast Δ_p(B2 - B) (stage 1 of 6.2). Also reported: the
  within-process round-to-round SD and the ratio s_P / (√2·σ̂_E1), the variance gain from interleaving.
- n = max(3, ceil((0.866·s_P / 1.02%)² / 10)), capped at 6. That gives n = 3 up to s_P = 6.4%, 4 at 7%, 5 at 8% and
  6 at 9%. (1.02% is the largest pooled SE that still gives at least 80% unadjusted TOST power at ±3% for a true 0;
  0.866 converts the A/A SD to a contrast against the mean of B and B2. At s_P = 5.7%, i.e. a per-run sigma of 4%,
  n = 3 gives 91% in the simulation of 6.12.)
- Replicates 4..n run on night 2 as complete (dataset, replicate) units. They are pre-planned, with no look in
  between.
- **Fail-safe:** if s_P > 1.5·√2·σ̂_E1 (interleaving clearly noisier than separate processes), P is demoted. Night 2
  then runs the X design on all 10 datasets with cells B, B2, N, Nf, Cr, C, SV x 3 blocks (about 10 h), and C/B is
  labelled "heat-mitigated, not removed" (3.6).

**Stopping.** n is fixed before any throughput is read. The night stops only at a unit boundary and for time, never
on results. Every complete (dataset, replicate) unit and every complete (dataset, block) segment enters the primary
analysis; only incomplete units are dropped.

### 5.7 Wall-clock

| item | model | total |
|---|---|---|
| P process | 40 s load + builds (5 cheap x 7 s + SV 5 + Cr 25 + C at `SMOOTH[d]` + 10) + 60 s cool-down + 8 rounds x 7 cells x 2.7 s + probes ≈ 610 s (fb 514 s, books 790 s) | 1.75 h per replicate (C2 on fb included; +0.2 h with G, GCr) |
| X run | 40 + build + 60 + 1 + 5.5 (warm-up) + 6.8 (timed) + 1 ≈ 122 s cheap; C + `SMOOTH[d]`; COLD 115 s | 43 min per block |
| BA / CA | 40 + 7 + 60 + 27 + 137 ≈ 273 s; CA + 184 s | 37 min |
| night 0 | 10 + 20 + 91 + 15 min | 2.3 h |
| night 1 | 0.7 h calibration + 5.25 h P (n = 3) + 2.15 h X + 0.6 h BA/CA | **8.7 h** |
| D + CL | 5.1 + 1.0-1.5 h | 6.1-6.6 h |

### 5.8 Session 2 (after the meeting; an independent replication, 6.6)

- **Stratum-A extension.** P processes with cells {B, B2, Cr, C} on every stratum-A dataset, until n = 8. Fresh
  seeds 3000 + replicate.
- **NC timed** (with B, B2 and C re-run in the same processes): the N x C interaction.
- G and GCr, if P3 was built late.
- The alpha sweep (PLAN `sweep`).
- The Linux replication (2.7).
- The mechanism layer (section 9).
- Per-dataset equivalence at ±5%.

Each session has its own complete units with its own B, B2 and C. Sessions are never pooled into one confirmatory
test.

---

## 6. Analysis (pre-registered)

### 6.1 Statistic per unit

- **X:** y = ln(`throughput_ops_s`) of the whole 5M replay, valid runs only.
- **P:** y_{pcr} = ln(throughput of the 1M timed lookups) of process p, cell c, round r, valid visits only.
- **Excluded from every throughput estimate:** COLD and the D layer. COLD is used only for V5.

### 6.2 P layer: two-stage within-process model (primary)

1. **Stage 1, per process.**
   - Fit y_{cr} = a_c + b_r + e by least squares over the valid visits. The round effect b_r absorbs host state
     shared within a round, because every cell in a round replays the same trace segment.
   - With no missing visits, â_c - â_B equals mean_r(y_{cr} - y_{Br}).
   - Output â_{pc}, centred within the process.
2. **Stage 2, per dataset d.**
   - Fit â_{pc} = μ_c + β_p + ε by least squares, with residual df (n - 1)(m - 1) (m cells; n = 3 and m = 7 give 12).
   - σ̂_d is the residual SD of dataset d.
   - Contrasts:
     - for a cell X: Δ_d(X) = μ̂_X - (μ̂_B + μ̂_B2)/2, SE_d = σ̂_d·sqrt(1.5/n);
     - A/A: Δ_d(B2) = μ̂_B2 - μ̂_B, SE_d = σ̂_d·sqrt(2/n);
     - C vs Cr: SE_d = σ̂_d·sqrt(2/n).
   - Per dataset, also report the per-cell residual SD and a Brown-Forsythe test across cells. A cell with p < 0.01
     gets its own variance in sensitivity S3.
3. **Variance decomposition** (reported): the within-process round residual SD, and the process x cell SD (the
   layout component). This is the interleaving gain.

### 6.3 X layer: block-adjusted model (secondary; also the per-process A/A check)

- Per dataset: y_{cb} = μ_c + β_b + ε over cells {B, B2, C, SV} and blocks b = 1-3, giving residual df 6.
- Contrasts as in 6.2. The segment ICC is reported.
- BA/CA: the unpaired difference ln(CA/BA) - (μ̂_C - μ̂_B̄) on fb, with its 95% CI, descriptive.
- S8 adds the predecessor's cell, its build CPU-seconds and "preceded by a C build" as covariates.
- X estimates are never pooled with P estimates. The P-vs-X difference in C/B on fb, osm and planet is reported
  descriptively.

### 6.4 Pooling over datasets

- **Estimand (primary): the equal-weight average over the ten AIDB datasets**, a fixed, enumerated set. It is
  restricted to the datasets in the hypothesis's scope:
  - stratum A or F where stated;
  - always excluding structural nulls, which are reported as extra A/A comparisons.
- θ̂ = (1/k)·Σ Δ_d and SE = sqrt(Σ SE_d²)/k. The df is Satterthwaite's, (Σ SE_d²)² / Σ(SE_d⁴/df_d). The 95% CI uses
  t at that df. The ratio is exp(θ̂).
- **Generalisation (secondary, labelled "for a new dataset like these"):**
  - DerSimonian-Laird random effects;
  - the 95% prediction interval θ̂_RE ± t_{k-2}·sqrt(τ̂² + SE_RE²);
  - τ̂ with a Q-profile CI.
  - I² is not used to switch wording, since the primary estimand is an average by definition.
- **Nulls.** Also report max_d |Δ_d| with Bonferroni 99.5% intervals over the k datasets: a null average can hide +5%
  and -5%.

### 6.5 Tests and verdicts

| verdict | rule |
|---|---|
| faster / slower | the 95% CI excludes 1 in the stated direction; the slide adds "by more than 3%" or "by less than 3%" according to whether the lower bound clears the 3% SESOI |
| equivalent within ±δ | TOST at α = 0.05 per side, i.e. the 90% CI inside [ln(1 - δ), ln(1 + δ)]; δ = 3% pooled, 5% per dataset (session 2 only); the achieved bound max(\|L90\|, \|U90\|) is always printed |
| inconclusive | everything else; "trend" and "slightly" are banned |

- **Predicted nulls** (H22, H25, H24b) are tested only by equivalence.
- **Share of C's gain due to the root.** ρ = (μ_C - μ_Cr)/(μ_C - μ_B̄), pooled over stratum A, with a 95% Fieller CI.
  Per dataset, Var(Δ1) = 2σ̂_d²/n, Var(Δ2) = 1.5σ̂_d²/n and Cov = σ̂_d²/n. If C > B's 95% lower bound is below 6%, the
  C ≡ Cr verdict cannot separate "all from the root" from "half from the root", and the slide shows the ρ interval
  instead of the TOST verdict. If the denominator is not significantly positive, ρ is reported as undefined.
- **Counter predictions.** Every estimate is reported beside its prediction from `predictions.json`. Nothing is
  withheld. A CI wholly outside the predicted interval is labelled "inconsistent with the counter prediction
  (mechanism unexplained)".
- **H28 meta-regression.**
  - Weighted least squares Δ_d(X) = a + β·p_d(X) with weights 1/SE_d², over every timed (cell, dataset) pair except
    SV. Structural nulls are included, at p = 0.
  - Cluster-robust (CR2) SE by dataset, t on k - 1 df; one-sided test of β > 0. a and β are reported with 95% CIs.
  - Studentized residuals > 3 are flagged as "inconsistent with counters".

### 6.6 Families and multiplicity

- **Families**, each at a familywise α of 0.05 with Holm; adjusted p-values are reported; for TOST,
  p = the max of the two one-sided p-values:
  - NFL {H22};
  - CSV {H24a, H25a, H25};
  - G {H26a, H26b} (only with P3);
  - mechanism {H28, H24b}.
- **No α is spent on the validity conditions V1-V7.** They gate interpretation (6.10).
- **Per dataset (descriptive overnight).**
  - "Equivalent on all ten" is an intersection-union test, so each dataset is tested at α with no correction.
  - "Equivalent (or faster) on these k datasets" needs Bonferroni over the k named.
  - A slide that names dataset-specific effects uses simultaneous intervals covering every interval on that slide:
    99.5% for 10, 99.94% for 80.
- **Interactions** ((GCr - Cr) - (G - B); in session 2, (NC - C) - (N - B)) are reported as bounds only.
- **Looks.** The night-1 data (plus pre-planned night-2 replicates) are the one confirmatory test of H22-H28.
  Session 2 is an independent replication, analysed alone, plus the session-2-only hypotheses. The combined estimate
  across sessions is descriptive.

### 6.7 Validity gates

- **G1 (E1, night 0):** U80(σ̂_AA) <= 8% (5.2). E1 is go/no-go only. The tests run whatever σ̂_live turns out to be,
  and each null prints its realised power at σ̂_live.
- **G4 (live A/A, V1), P layer.** All three must hold (simulated pass rate 0.94 when nothing is wrong, at n = 3 and
  sigma 4%; `power_sim_out.md`):
  - (i) the pooled B2/B 95% CI contains 1;
  - (ii) at most 2 of 10 per-dataset 95% CIs exclude 1 (P(>= 3 | null) = 1.16%);
  - (iii) σ̂_live = sqrt(mean over d of σ̂_d²), the per-process cell-level SD, is <= 8%.
- **Equivalence-style A/A (reported, not gated):** TOST with margin 3.6·SE_pooled(B2 - B). That is ±3.7% at sigma 4%
  and ±4.6% at 5%.
- **X layer A/A:** (i), and at most 1 of 3 per-dataset CIs excluding 1 (P(>= 2 | null) = 0.7%). If it fails, X's
  C/B is not shown; P is unaffected.
- **V2 and V3 (positive controls):** pooled Nf/B CI wholly below 1, and SV/B CI wholly above 1. If either fails, the
  protocol has not shown it can see an effect: estimates are reported, but no equivalence verdict is made.

### 6.8 Warm-up and thermal analysis

**Warm-up.**
- V4 placebo tests (3.5).
- COLD V5, with its per-dataset values and the 80-chunk curves.
- Per-cell re-warm curves in P.
- Flagged units cross-tabulated with telemetry.
- Migration dips per X run.

**Thermal, X layer.**
- V7: C vs B effective GHz (TOST ±1%, run-level SE) and latency probe ns/load (TOST ±1%).
- Descriptive: the slope of log chunk throughput over chunks 2-80 per cell. As a reference, a slope difference Δs
  moves the whole-replay mean by about 7.5·Δs per 16-chunk unit, so ±1% corresponds to about |Δs| <= 0.0013 at
  16-chunk resolution. Its SE is cluster-robust by run.
- Invalid-run rate by cell. If C's rate exceeds 2x B's, it is reported as a thermal signal, with S4.

**Thermal, P layer.**
- Effective GHz and the latency probe per round, by build-order position: descriptive, because heat is common by
  design.
- E0h (5.2), descriptive.

If V7 fails, X's C/B is labelled "thermally confounded". P's C/B is not affected.

### 6.9 Determinism checks (any failure stops the analysis until explained)

1. `result_checksum` must be identical:
   - in P, across all valid visits of a round (same segment), in every process;
   - in X, across cells within a block on a dataset;
   - between D and X block 1 (seed 1001).
2. `trace_fingerprint` must be identical across the same groups.
3. For each (dataset, cell), `memory_before` and `learnability` (excluding `*_ns`) must be identical in D, in every P
   process and in every X run. C2 must equal C, which is the CSV-greedy rebuild check.
4. `work_counters` (excluding `cache_lines*`) must be identical between instrumented runs of the same (dataset, cell,
   seed).
5. R1, R1-P6 and, with P3, R1-G must have passed.
6. With P3, separability on fb (H18): GC's root and gap probes equal GCr's, and GC's fence, coordinate and key_at
   probes equal C's.
7. `fences_ba.py` raw-root and binary probes must equal the D layer's to ±0.01.

### 6.10 What is reported when a gate fails

| failed | still reported | not reported | wording |
|---|---|---|---|
| G0 / Q0 | Tier M (D, CL, memory, RSS if quiet enough for E5) | all Mac timing | "host could not be made quiet" |
| G1 (U80 > 8%) | Tier M; E1 noise floor; E0h descriptively | Mac throughput; it moves to Linux | "host noise σ = x% exceeds the gate" |
| G2 R1 | Tier M (from build-fs) | anything from build-proto | "new binary not validated" |
| G3 fail-safe | X-design results on 10 datasets (night 2) | P results | "heat mitigated, not removed" |
| G4 live A/A (P) | deterministic results; the A/A diagnostics; X results if X's own A/A passes | every P throughput ratio | "A/A failed: the protocol cannot resolve throughput on this host" |
| V2/V3 positive controls | estimates with CIs | equivalence verdicts | "sensitivity not demonstrated" |
| V4 warm-up placebo | everything on sub-chunks/chunks 2-16 | whole-unit numbers as primary | "warm-up deficit observed; first chunk excluded" |
| V5 COLD | everything | the claim "the warm-up check is sensitive" | "cold reference not detected on this host" |
| V6 determinism | nothing until explained | — | — |
| V7 (X thermal) | P results; X's C/B labelled | an unqualified X C/B | "thermally confounded (X only)" |
| invalid rate > 10% of units | complete units, with the invalid list | units below the completeness rule | — |

### 6.11 Sensitivity analyses (always printed, never substituted)

| # | sensitivity |
|---|---|
| S1 | first chunk or sub-chunk excluded (chunks 2-16 / sub-chunks 2-16) |
| S1b | P without round 1 |
| S2 | median chunk rate (PLAN's `STAT=med`) |
| S3 | P: a single-stage mixed model with process and process-x-cell random effects, and per-cell variances for cells failing Brown-Forsythe; X: unpaired (PLAN's estimator) |
| S4 | invalid units included |
| S5 | units flagged by the placebo statistic or by state-flip telemetry excluded |
| S6 | Hodges-Lehmann on per-process contrasts. If it differs from the mean contrast by more than half the CI, the slide says that an outlier drives the result |
| S7 | ln(effective GHz) and ln(latency probe) as covariates (X per run; P per round) |
| S8 | X: predecessor covariates (6.3) |
| S9 | random effects and the prediction interval (6.4); with vs without structural nulls; per-replicate and per-block estimates; price sensitivity p = 1 for G |

### 6.12 Power (simulated; replaces the draft's table 5.4)

Source: `power_sim_out.md` and `power_sim_families_out.md`, 20,000 replicates.
- Datasets and effects: 10 datasets; C = Cr = +5% on 5 datasets and 0 elsewhere; Nf 0.85-0.96; SV +30%; nulls at 0
  ("base") or at -1% ("shift").
- Analysis: conditional pooling and per-method Holm, as specified here.
- Mapping: σ is the per-process cell-level SD, s_P/√2. The pooled (not stratified) C > B column is conservative:
  stratified within A the SE is about 1.3% at σ = 4%, n = 3, which gives roughly 90%.

| σ, n | A/A gate passes (true A/A) | C > B (pooled) | Cr > B | N ≡ B base / shift | C ≡ Cr | G ≡ B | GCr ≡ Cr |
|---|---|---|---|---|---|---|---|
| 3%, 3 | 0.94 | 0.94 | 0.94 | 1.00 / 0.92 | 0.96 | 0.99 | 0.97 |
| 4%, 3 | 0.94 | 0.71 | 0.71 | 0.91 / 0.73 | 0.69 | 0.88 | 0.77 |
| 5%, 3 | 0.94 | 0.46 | 0.45 | 0.69 / 0.53 | 0.28 | 0.56 | 0.38 |
| 4%, 5 | 0.94 | 0.93 | 0.93 | 0.99 / 0.90 | 0.95 | 0.99 | 0.96 |
| 5%, 5 | 0.94 | 0.75 | 0.74 | 0.93 / 0.76 | 0.74 | 0.91 | 0.80 |
| 4%, 8 | 0.94 | 0.99 | 0.99 | 1.00 / 0.98 | 1.00 | 1.00 | 1.00 |
| busy host (σ 9.6% mixture), 3 | 0.94 | 0.07 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 |

**What this means.**
- On a quiet Mac with per-process σ <= 4%, night 1 at n = 3 proves each predicted null with 70-90% probability.
- At σ = 5% only n = 5 does; E1b's rule (5.6) buys it on night 2.
- On a busy host only the positive controls can be shown.
- If interleaving cuts s_P well below √2·σ̂_E1, as the critique expects when the noise is host state, every row moves
  up.

---

## 7. Figures, in meeting order (none drawn from n = 1 timing)

1. **Method x metric matrix.**
   - Rows: N, Nm, Cr, C, G/G_key, NC; columns: memory, input-space compression (L_root, L_key), work, throughput,
     build.
   - Each cell holds "=", an arrow or a number, with the reason.
   - Claim: memory moves only through CSV metadata (plus G's 528 B). Compression is **measured** with the same S_k
     and line-fit metrics for every method, not true by definition: G_key and Nm squeeze fb's outlier space at key
     level, while CSV leaves inputs untouched and smooths targets. Throughput is the only statistical column.
   - Data: `d.jsonl`, `cl/*.json`.
2. **Protocol validity.**
   - Panels: (a) cold curves at 62.5k (A2 + COLD) against warm placebo statistics; (b) P re-warm curves per cell;
     (c) the E1 and E1b A/A distributions, with the interleaving variance gain; (d) E0h; (e) X GHz and latency probe
     by cell; (f) the night's hostmon other-CPU, powermode and thermal pressure.
   - Claim: every timed number is warm, the build heat is common by design in P (and measured in X), and the noise
     floor is σ = z%.
3. **Memory.**
   - Stacked B/key per dataset x cell (keys, values, base metadata, virtual points, root table), with RSS dots on fb.
   - Claim: NFL and G cost 0 B/key **because the index is key-ordered** (NFL's z-ordered layout, not built: about +30%
     B/key). CSV costs +0.80 accounted and 1.00 real B/key. Values are 76-80% of every bar.
4. **Input-space compression.**
   - Panels: (a) S_16 at L_root and at L_key per dataset for B, G(k*), G_key, Nm; (b) fb at L_key, RMSE/n and ME/n
     against k (1-256), with the k = 21 step marked (the 21 outlier gaps); (c) the root-fence CDF before and after G
     on osm and books; (d) region rank SSE for B and C; (e) |ΔPLA| against the 2k bound.
   - Claim: the measured version of "gap removal squeezes empty space". It works on fb only once all 21 outlier gaps
     are removed, it is invisible to PLA, and it has no index throughput consequence (4.3.4).
5. **Work per lookup.**
   - Stacked comparisons (key_at, coordinate, fence, gap table, root) for B, N, Nf, Cr, C, NC (+G, GCr), with
     transforms/op, the SV line and the 13.05 floor.
6. **Throughput forest plot, P layer.**
   - Per dataset and pooled (conditional, with the RE prediction interval greyed), 95% CI; the strata marked.
   - The A/A band from B2; the ±3% equivalence zone shaded; counter-prediction intervals as diamonds; SV and Nf rows;
     n on each row; every per-process contrast as a dot.
   - The headline rows are **Cr vs B (our root extension)** and **C vs Cr (CSV's own region use)**, plus the share ρ.
7. **Mechanism:** Δ_d against p_d (H28) with the fitted β and flagged residuals.
8. **Interactions** (bounds) and **P vs X** C/B on fb, osm and planet; the AIDB-length interaction (descriptive).
9. **Build time** (log axis): median and IQR per cell x dataset; the CSV paper's band of 889-2,902 s.
10. **AIDB hardness panel** (separate): B, Nm, G_key, NFL-faithful in its z-sorted convention, and CSV (labelled "the
    hardness tool's CSV construct").
11. **Backup slides:** the deviations table (3.7); the scope slide (1.1); the 2M prototype regime (4.3.5); the
    fb-outlier anatomy; the Linux plan.

---

## 8. Commands

### 8.1 Binary patches (in `$REPO/experimental/scaleli_proto`, a copy of `S`)

| patch | file | change | effort |
|---|---|---|---|
| **P1** | `src/benchmark.cpp` | `--cooldown-s S` (default 0): `sleep_for` right after `before=ix->memory();` in pass 1. `--go-dir DIR --go-timeout-s 600`: writes `DIR/ready` (pid, round) then blocks reading the FIFO `DIR/go` (poll with time-out); JSON `cooldown_s`, `go_wait_s`, `go_timeout`. Allowed-set updated | 0.5-1 h |
| **P2** | `src/benchmark.cpp` | `--cycles 1`: `proc_pid_rusage(getpid(), RUSAGE_INFO_V6, ...)` at build end, after cool-down, after warm-up, at every chunk boundary and after timing. Fields: user/system time, `ri_cycles`, `ri_instructions`, `ri_pcycles`, `ri_user_ptime`, `ri_system_ptime`, `ri_runnable_time`, `ri_pageins`, all converted with `mach_timebase_info`. Plus `getrusage` minflt/majflt, `task_info(TASK_EVENTS_INFO)`, `host_statistics64(HOST_VM_INFO64)` (swapins, compressions, decompressions), `TASK_VM_INFO.phys_footprint` and the thread count at each phase. Derived per chunk: `effective_ghz`, `pcore_fraction`, `runnable_fraction`, `oncpu_fraction`, `ipc`, `cycles_per_op`. `--freq-probe N` (LCG chain) and `--probe-latency-mb M` (Sattolo cycle over 64 B lines, allocated before the data load, 3M dependent loads) before prefault and after timing. `--selftest-telemetry` (2 s busy loop, prints the derived values, exits). On Linux: `perf_event_open` (cycles, instructions, user-only) and `CLOCK_THREAD_CPUTIME_ID` | 2-2.5 h |
| **P6** | `src/benchmark.cpp` (+ a small header) | `--interleave FILE` (TSV `name<TAB>flags`; common flags from argv), `--rounds R`, `--visit-warmup W`, `--visit-ops T`, `--sub-chunks K`. One workload: the trace is R x T keys and the warm-up keys R x W (`warmup_keys`, seed ^ 0x5741524d). Cells are built in a random order (`mt19937_64(seed ^ fnv1a(data path) ^ 0xB11D)`), each with `build_ns`, `memory_before`, `learnability` and prefault. Then the cool-down, and per round: go-gate, probes, permutation `mt19937_64(seed ^ r)` (logged; a test asserts the round permutations are not all equal), and per visit: snapshot, warm-up, snapshot, timed loop with K sub-chunk stamps, snapshot, checksum. Cells are type-erased through per-cell pointers to the **same templated loop as pass 1**: one indirect call per visit, none per lookup. JSON: `cells[]`, `rounds[]`, `visits[]`. No instrumented or verify pass in P | 3-4 h |
| P4 (Linux) | `src/benchmark.cpp` | `--perf-ctl FIFO`: `enable` just before the timed loop, `disable` just after | 0.5 h |
| P-linux | `src/benchmark.cpp` | `--pin-core N`: `sched_setaffinity({N})` on the main thread after the build threads join, before the cool-down | 0.5 h |
| **P5k** | `src/hardness.cpp` | `--gap-key`, `--gap-share` (4.3.4) | 1-1.5 h |
| P3 (go only) | `index.hpp`, `types.hpp`, `benchmark.cpp` | 4.3.3 | 4-6 h + 1-2 h |
| P7 (session 2) | `benchmark.cpp`, `index.hpp` | `--dependent 1` (each key XORed with an opaque zero derived from the previous result); `--skip-observe 1` (no `write_heat` store on reads; D layer only); `--kpc-events LIST` (kperf counters around the timed loop, sudo) | 2-3 h |

The P2 reads inside the timed window cost about 2 µs each: 81 per X run and 17 per P visit, under 0.01% of the
window. Every cell carries them.

### 8.2 `run_ba.py` v2

1. `BIN = os.environ.get('SCALELI_BIN', BPR)`; the night-0 and D presets set `SCALELI_BIN=$BFS`. When BIN is build-fs,
   `common()` omits every P1/P2/P6 flag (`--cooldown-s`, `--go-dir`, `--cycles`, `--probe-latency-mb`,
   `--freq-probe`), which build-fs would reject as unknown options. The plan header
   records the binary's sha256, `freeze2.sha256`, the shuffle seed and every rule-driven choice (n, k*, strata hash).
2. Jobs: `[dataset, cell_or_P, block_or_rep, layer, attempt, timeout, overrides]`, with layer in {N0, D, CAL, P, X}.
   For P jobs the runner writes `$P/cells/<d>.tsv` from `cells(d)` and the strata (structural nulls stay in).
3. `common(layer)` gives the lines of 4.1; COLD, BA, CA and E0h's heater use explicit overrides. A test asserts that
   the plan header lists COLD's `--prefault 0`.
4. Presets:
   - `n0_aa`, `n0_heat`: night 0 (5.2). In the heater arm, B starts within 2 s of the heater's exit, with no gate in
     between.
   - `d`: the D layer (`--with-g` adds G, GCr, GC).
   - `r1`, `r1p6`, `e1b`: calibration.
   - `night1`: P replicates 1-3 alternating with X blocks 1-3, then BA/CA (5.6).
   - `p_extra --reps 4..n`: night 2.
   - `x_all`: the fail-safe.
   - `s2_*`: session 2.
   - `cost <preset>` uses the 5.7 model.
5. `run()`:
   - Waits for `pgrep -x scaleli_bench` and `pgrep -x scaleli_hardness` to be empty.
   - Runs a **go-writer thread** that watches `$P/go/ready` and writes `go` to the FIFO when the 2.5 quiet rule
     holds.
   - Evaluates the validity rules 2.5 (a)-(g) after each unit, from the P2 stamps and hostmon, and writes `valid`,
     `invalid_reasons` and the telemetry summaries into `meta`.
   - Retries per 2.5.
   - `done()` keys on (dataset, cell, block, attempt) and counts a unit done only when it is valid or its attempts
     are exhausted.
   - Applies the pause rule, stops on battery, and keeps resumability.
   - Logs the predecessor's cell, build type and build CPU-seconds, plus `taskpolicy -G -p`.
   - Never prints throughput.
6. `show <cell> <d> [--layer D|X]` prints the exact command (`rss_ba.sh` uses it).
7. The D layer renames the throughput fields in `d.jsonl` (5.3).

### 8.3 `analyze_ba.py` v2 (written blind; frozen as freeze 3 before the report)

1. Modes:
   - `--quiet hostmon.jsonl --last 600` (Q0);
   - `--calib aa n0_aa.jsonl` (G1);
   - `--calib heat n0_heat.jsonl`;
   - `--regress r1.jsonl pilot_seed11.jsonl`;
   - `--calib p6 r1p6.jsonl --det d.jsonl`;
   - `--calib e1b e1b.jsonl --e1 n0_aa.jsonl` (prints n and the fail-safe verdict);
   - `--strata d.jsonl --cl cl/` (writes `strata.json` and `predictions.json`);
   - `--report --det d.jsonl --cl cl/ --t t.jsonl --hostmon hostmon.jsonl --figdata figdata.json` (P and X rows share
     `t.jsonl` and are separated by `meta.layer`).
2. Valid units only for the primary analysis; the invalid list with reasons.
3. The 6.2 two-stage P model and the 6.3 X model.
4. 6.4 pooling (conditional, Satterthwaite; RE, PI, Q-profile τ̂).
5. 6.5 verdicts, achieved bounds, SESOI clearance, Fieller ρ, the H28 meta-regression (CR2).
6. 6.6 per-family Holm.
7. Gates 6.7 and the 6.10 reporting logic.
8. 6.8 placebo, COLD, GHz and latency TOSTs.
9. 6.9 determinism checks.
10. The S1-S9 sensitivity analyses.
11. Deterministic and CL tables (signatures, S_k, RMSE/ME, PLA, probes).
12. `--figdata` for section 7.

### 8.4 Other scripts

- **hostmon v2**: section 2.3 (`--out`).
- **`fences_ba.py`** (stdlib; imports `$SP/threeblock/threeblock.py`): `extract`, `gap --k 1,4,16,64`,
  `validate --det d.jsonl`, `gorule --p 0.25` (writes `gorule.json`, with the p = 1 sensitivity alongside), and
  `gate200m --bin` (R1-G, only with P3).
- **`rss_ba.sh`**:
  - commands from `run_ba.py show <cell> fb --layer X` with `--ops 10000000 --cooldown-s 0` (on BFS at night 0:
    without the P1/P2 flags);
  - the plateau formula subtracts 3.20 GB + 32 B x ops + **8 B x warm-up lookups** (32 MB at 4M; the current script
    omits it, which leaves the B-vs-C difference unchanged but biases the absolute real-vs-accounted figure by 32 MB);
  - match processes with `pgrep -x`.

### 8.5 Command sequence

```zsh
REPO=/Users/louisvasseur/Downloads/scaleli_sota; S=$REPO/experimental/scaleli; NB=$S/results/aidb_ba
SP=/private/tmp/claude-501/-Users-louisvasseur-Downloads-scaleli-sota/ac6d4251-6384-4db6-8aba-a17e1917a2ab/scratchpad
P=$NB/proto; BFS=$REPO/build-fs/scaleli_bench; BPR=$REPO/build-proto/scaleli_bench
mkdir -p $P/prereg $P/go $P/cells $P/cl $P/fences; cd $P

# ---- S0: freeze 1 and host fix (2.2 items 1-2 by hand; sudo where shown) ---------------------------
cp $SP/protocol/PROTOCOL.md $P/prereg/ && shasum -a 256 $P/prereg/PROTOCOL.md > $P/prereg/freeze1.sha256
sudo pmset -c powermode 2 && pmset -g custom | sed -n '/AC Power/,$p' | grep powermode    # must print 2
pmset -g batt                                                                              # AC Power, >= 80%
pgrep -fl 'qemu|UTM'; pgrep -x scaleli_bench; pgrep -x scaleli_hardness                    # all empty

# ---- N0: night 0 on build-fs -------------------------------------------------------------------------
{ sw_vers; sysctl -n hw.model machdep.cpu.brand_string hw.memsize hw.pagesize hw.perflevel0.physicalcpu \
  hw.perflevel0.cpusperl2 hw.perflevel0.l2cachesize hw.perflevel0.l1dcachesize hw.perflevel1.physicalcpu; \
  pmset -g custom; pmset -g batt; /usr/bin/c++ --version; shasum -a 256 $BFS; } > host_n0.txt
taskpolicy -b python3 $NB/hostmon.py --out $P/hostmon.jsonl & echo $! > hostmon.pid
sleep 600; python3 $NB/analyze_ba.py --quiet hostmon.jsonl --last 600                      # G0 (Q0)
export SCALELI_BIN=$BFS
python3 $NB/run_ba.py plan n0_aa --shuffle-seed 20261002 > plan_n0_aa.json
caffeinate -ims python3 $NB/run_ba.py run plan_n0_aa.json n0_aa.jsonl --hostmon hostmon.jsonl
python3 $NB/analyze_ba.py --calib aa n0_aa.jsonl                                           # G1: GO iff U80 <= 8%
python3 $NB/run_ba.py plan n0_heat > plan_n0_heat.json
caffeinate -ims python3 $NB/run_ba.py run plan_n0_heat.json n0_heat.jsonl --hostmon hostmon.jsonl
zsh $NB/rss_ba.sh                                                                          # E5 (H4)
python3 $NB/analyze_ba.py --calib heat n0_heat.jsonl

# ---- D and CL (daytime; light load is fine, never beside a timing job) --------------------------------
python3 $NB/run_ba.py plan d > plan_d.json
nohup caffeinate -ims python3 $NB/run_ba.py run plan_d.json d.jsonl > run_d.log 2>&1 &
#   after run_d.log prints PLAN DONE:
python3 $NB/fences_ba.py extract --det d.jsonl --out fences && python3 $NB/fences_ba.py gap --k 1,4,16,64 --dir fences
python3 $NB/fences_ba.py validate --det d.jsonl --dir fences && python3 $NB/fences_ba.py gorule --p 0.25 --dir fences > gorule.json
H=$REPO/build-proto-h/scaleli_hardness     # build-fs hardness + P5k only (see 5.5); its sha256 goes into freeze 2
for f in fences/*_raw.sosd fences/*_g*.sosd; do $H --data $f --dtype uint64 --pla-eps 1,8 --gap-share 1,16,64 \
    --check-sorted 1 --threads 16 > cl/$(basename ${f%.sosd}).json; done
for d in books covid fb genome history libio osm planet stack wise; do                     # Nm at L_root
  $H --data fences/${d}_raw.sosd --dtype uint64 --flow $S/results/aidb_flowv2/flows_monotone/${d}_mono.txt \
     --pla-eps 1,8 --gap-share 1,16,64 --threads 16 > cl/${d}_root_mono.json; done
for d in books covid fb genome history libio osm planet stack wise; do                     # L_key: B, Nm, G_key
  K=1,4,16,64,256; [[ $d == fb ]] && K=1,4,9,16,20,21,32,64,256
  f=$S/data/external/gre/$d; [[ -f $f.sorted ]] && f=$f.sorted
  $H --data $f --dtype uint64 --flow $S/results/aidb_flowv2/flows_monotone/${d}_mono.txt --gap-key $K \
     --gap-share 1,16,64 --pla-eps 32,4096 --threads 16 > cl/${d}_key.json; done
zsh $SP/nfl_ba/run2.sh csv stack wise        # plus fb with --limit 199999979 --virtual-alpha 0.1 (same pattern)
python3 $NB/analyze_ba.py --strata d.jsonl --cl cl/                                         # strata.json, predictions.json

# ---- E-c: build-proto (5.5) ----------------------------------------------------------------------------
rsync -a --exclude data --exclude results $S/ $REPO/experimental/scaleli_proto/
#   apply P1, P2, P6 (P3 only if gorule.json says go; P5k already applied for $H) per 8.1
cmake -S $REPO/experimental/scaleli_proto -B $REPO/build-proto -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DSCALELI_NATIVE=OFF -DSCALELI_SANITIZE=OFF -DSCALELI_EXTERNAL=OFF
cmake --build $REPO/build-proto -j 8 && ctest --test-dir $REPO/build-proto
shasum -a 256 $BFS $BPR $REPO/build-proto/scaleli_hardness $NB/run_ba.py strata.json predictions.json gorule.json \
  > $P/prereg/freeze2.sha256                                                                # freeze 2

# ---- N1: night 1 (quiet; 2.2 checklist again) -----------------------------------------------------------
export SCALELI_BIN=$BPR
sleep 600; python3 $NB/analyze_ba.py --quiet hostmon.jsonl --last 600                      # G0
$BPR --selftest-telemetry                                                                  # G2, part 1
python3 $NB/run_ba.py plan r1 > plan_r1.json && python3 $NB/run_ba.py run plan_r1.json r1.jsonl --hostmon hostmon.jsonl
python3 $NB/analyze_ba.py --regress r1.jsonl $NB/pilot_seed11.jsonl                        # G2, part 2
python3 $NB/run_ba.py plan r1p6 > plan_r1p6.json && python3 $NB/run_ba.py run plan_r1p6.json r1p6.jsonl --hostmon hostmon.jsonl --go-dir $P/go
python3 $NB/analyze_ba.py --calib p6 r1p6.jsonl --det d.jsonl
python3 $NB/run_ba.py plan e1b > plan_e1b.json && python3 $NB/run_ba.py run plan_e1b.json e1b.jsonl --hostmon hostmon.jsonl --go-dir $P/go
N=$(python3 $NB/analyze_ba.py --calib e1b e1b.jsonl --e1 n0_aa.jsonl --print-n)            # G3
python3 $NB/run_ba.py plan night1 --n $N > plan_night1.json; python3 $NB/run_ba.py cost night1
nohup caffeinate -ims python3 $NB/run_ba.py run plan_night1.json t.jsonl --hostmon hostmon.jsonl --go-dir $P/go > run_t.log 2>&1 &
#   resumable: re-run the same line after any interruption

# ---- N2 (only if N > 3): python3 $NB/run_ba.py plan p_extra --reps 4..$N > plan_n2.json; run as above into t.jsonl

# ---- analysis (after freeze 3) ----------------------------------------------------------------------------
shasum -a 256 $NB/analyze_ba.py > $P/prereg/freeze3.sha256
python3 $NB/analyze_ba.py --report --det d.jsonl --cl cl/ --t t.jsonl --hostmon hostmon.jsonl --figdata figdata.json > report_t.md
kill $(cat hostmon.pid)
```

### 8.6 Restore the host

```zsh
sudo mdutil -a -i on; sudo tmutil enable; sudo pmset -a powernap 1     # only those that were on before
sudo pmset -c powermode 1                                              # or whatever pmset -g custom showed before
#   restart the UTM VM and the apps; resume iCloud sync and updates
```

**The hardness binary.** `$REPO/build-proto-h` is a copy of `S` with P5k only, built like build-proto. It exists so
CL can run before E-c. With P5k in build-proto, that copy can be used instead; the sha256 of whichever is used goes
into freeze 2.

---

## 9. Mechanism layer M: why plain binary search beats every learned cell (session 2; no gate depends on it)

Pilot, fb, n = 1: SV 1.011 Mops vs B0 0.731 (1.38x); SV, RAW and B gave 1.16, 0.96 and 0.73. Counters are measured
only over the timed window: P7 `--kpc-events` on the Mac, P4 on Linux. Cells SV, RAW, RC, B and C on fb and osm,
n = 3, single-cell processes with D-layer settings but uninstrumented.

**Mac, with sudo.** kperf through `/System/Library/PrivateFrameworks/kperf.framework`, or
`xctrace record --template "CPU Counters"` windowed by the P2 timestamps. The event database is
`/usr/share/kpep/cpu_100000c_2_72015832.plist` ("as3"): 8 configurable counters plus 2 fixed, and at most 3
`*_NONSPEC` events per pass (counter mask 0xe0). There is no L2 or SLC miss event, so DRAM misses are inferred.
- Pass A: `L1D_CACHE_MISS_LD_NONSPEC`, `L1D_TLB_MISS_NONSPEC`, `BRANCH_MISPRED_NONSPEC`, `L2_TLB_MISS_DATA`,
  `MMU_TABLE_WALK_DATA`, `MAP_STALL_DISPATCH`, `L1D_CACHE_WRITEBACK`, `LD_UNIT_UOP`.
- Pass B: `BRANCH_COND_MISPRED_NONSPEC`, `BRANCH_INDIR_MISPRED_NONSPEC`, `INST_INT_LD`, `MAP_DISPATCH_BUBBLE`,
  `L1D_CACHE_MISS_LD` (speculative minus retired gives wrong-path loads).

**Linux (Skylake-SP class).** cycles, instructions, `mem_load_retired.l1_miss/l2_miss/l3_miss`,
`dtlb_load_misses.walk_completed/walk_active`, `cycle_activity.stalls_l3_miss`,
`l1d_pend_miss.pending/pending_cycles` (memory-level parallelism), `br_misp_retired.all_branches`, topdown slots.
If perf reports less than 100% running time, split into passes.

| hypothesis | signature | manipulation |
|---|---|---|
| MLP: SV's misses overlap; the index walks a serial chain (root, `regions_`, Region, descriptor, arena, value) | SV: more L1D misses/op but fewer cycles per miss; higher pending-miss MLP | `--dependent 1` replay; if SV's lead shrinks, it is MLP |
| TLB: the index touches 4-6 unrelated pages per lookup | MMU walks/op, L2 TLB misses/op | Linux THP never vs always for SV and B |
| decode: codec dispatch and in-block decoding (8.02 comparisons over compressed keys) | instructions/op, indirect mispredicts | SV vs RAW vs B |
| write: every `find` stores `Region::write_heat` (index.hpp:111, called at :476), dirtying a header line per read | `L1D_CACHE_WRITEBACK`/op, B vs SV | `--skip-observe 1` (D only) |
| size: it is the hierarchy, not the algorithm | everything scales with n | B and SV at 2M, 20M and 200M; the crossover is the answer |

**Analysis.** Fit cycles/op ≈ a·instructions + b·L1D misses + c·walks + d·mispredicts across cells and datasets. The
calibration uses the latency probe at 64 B and 16 KiB strides. A fit that leaves SV's lead unexplained points to MLP.

---

## 10. Threats to validity

| # | threat | what the protocol does |
|---|---|---|
| 1 | host contention (VM, agents, nightly jobs, Spotlight, Time Machine, iCloud) | checklist 2.2; Q0; delta-based hostmon; in-process go-gate; outcome-blind invalidation; live A/A gate; G1 sends timing to Linux |
| 2 | **build-heat confound** | P: removed by design (common thermal history, round-randomised order). X: fixed cool-down, go-gate, GHz and latency-probe TOSTs, S7 and S8 covariates. E0h measures its size on throughput |
| 3 | low-power or capped clocks | powermode 2 on AC, logged per sample; R1 demands >= 3.8 GHz; AC loss or a powermode change invalidates |
| 4 | incomplete warm-up | X: 4M (>= 10x the measured transient); P: 1M re-warm; placebo tests; COLD reference; pre-registered first-chunk fallback; AIDB-length arm |
| 5 | per-process noise, two-state host mixture | P interleaving with round pairing; two-stage model; complete-unit analysis; HL and mixed-model sensitivities |
| 6 | E-core residency, descheduling, migration between P-clusters | P-core fraction and runnable time per chunk and visit (invalidating); dip counts; in P a migration hits all cells of a round alike |
| 7 | memory layout | random build order per P process; ASLR on; argv/env padding in X and D; one binary for all timed cells. Large allocations are page-aligned, so L1 aliasing is a fixed function of a cell's allocation sequence, which is legitimately part of the method. Optional layout-pad check (DECISIONS, item 13) |
| 8 | paging and allocator state | VM stopped; `ri_pageins`, swapins and decompressions in the window invalidate; `phys_footprint` logged |
| 9 | analytic flexibility | freezes 1-3; rule-driven n, k* and strata; blinding; amendments logged |
| 10 | structural nulls diluting pools | D-layer signatures; excluded from pools; reported as extra A/A |
| 11 | effect concentrated on a few datasets | strata A/F frozen before timing; stratified tests; meta-regression on counter predictions |
| 12 | instrumentation perturbing timing | counters only in D; P2 reads < 0.01% of the window, in every cell |
| 13 | chaotic CSV greedy (±0.36 root probes under 1e-6 perturbations) | C2 rebuild check; no claim rests on < 0.5 probes |
| 14 | construct validity (labels) | 1.1 labels on every slide; Cr vs B and C vs Cr headlined separately |
| 15 | compression true by definition | method-agnostic S_k and line-fit outcomes on the same key sets (1.2) |
| 16 | G tested where it cannot act (fb's outliers are invisible to the root) | G_key at L_key; the k >= 21 prediction; the no-throughput statement (4.3.4) |
| 17 | 2M-to-200M extrapolation | `fences_ba.py` at 200M before predictions and before P3; 2M labelled as a different regime |
| 18 | counters are not time (they mis-ranked SV) | counters are mechanism only; time is measured; table comparisons counted apart from root probes |
| 19 | accounted vs real memory | E5 RSS with the warm-up vector subtracted; hidden capacity predicted (H4) |
| 20 | external validity (one Mac, 16 KiB pages, uniform single-thread reads, batch 1) | the scope slide; Linux replication including SV and RAW; AIDB-length arm |
| 21 | flow seed not replicated | stated; Nm in D and CL; untrained-flow control on the AIDB panel |
| 22 | two looks; selective reporting | session 2 is an independent replication; every estimate reported with flags |
| 23 | non-random failures (C timeouts, C invalid more often) | the go-gate sits at the same position for every cell; immediate retries; invalid and missing counts per cell; S4 |

---

## Appendix A. Differences from PLAN.md

| item | PLAN.md | this protocol |
|---|---|---|
| compression | codec bytes (a null by construction) | input-space compression, method-agnostic (S_k, line fit, PLA) at L_root and L_key; G_key; Nm |
| methods | NFL, CSV | + gap removal (prototype at 200M; C++ only on go), Cr, Nm |
| timing design | separate processes, 3 blocks, fixed-effect pooling | P interleaved (primary) + X per-process (secondary), conditional pooling, per-method Holm, strata |
| warm-up | 4M, prefault 0 in `run_ba.py` (the pilot ran prefault 1) | X 4M + prefault 1; P 1M per visit; placebo test; COLD x 9 |
| thermal | 4M warm-up as partial equalisation + slope check | P removes the confound; X: 60 s cool-down, go-gate, GHz and latency TOSTs; E0h |
| host | other-CPU logged (`ps pcpu`) | power mode, VM stop, delta-CPU hostmon, telemetry-based invalidation, in-process go-gate |
| instrumented pass | in block 1 of the timing sweep | separate D layer on build-fs |
| NC | timed n = 1 | D only before the meeting; timed in session 2 |
| E1 | evening, 10 runs | night 0, 10 + 10 on build-fs, a go/no-go on the 80% upper bound; E1b decides n |
| D6 (interleaving) | deferred | adopted as the primary design |

---

## CHANGE LOG (review issue -> what changed)

Sources: **ST** = statistics review, **SY** = systems review, **SU** = supervisor review. Severity as the reviewer gave
it. Verdicts:
- **adopted**: as proposed;
- **adopted (modified)**: the problem is accepted, the fix differs, and the reason is given;
- **superseded**: made moot by another adopted change;
- **rejected**: with the reason.

| # | src, sev | issue | verdict | what changed (where) |
|---|---|---|---|---|
| 1 | ST blocker | live A/A gate needs TOST, passes 0.69 at σ 4% | adopted | G4 = pooled CI ∋ 1, <= 2/10 per-dataset CIs exclude 1, σ̂_live <= 8% (pass 0.94); equivalence-style A/A at 3.6·SE reported, not gated (6.7) |
| 2 | ST blocker | Holm over 7 mixed tests kills the nulls | adopted | Nf, SV moved to validity conditions with a 6.10 row; one Holm family per method; the 5.4 power table replaced by the simulation (6.6, 6.12) |
| 3 | ST major | RE pooling answers the wrong estimand | adopted | conditional equal-weight interval primary, Satterthwaite df; RE + prediction interval as "new dataset"; max\|Δ_d\| with 99.5% intervals for nulls (6.4) |
| 4 | ST major | the CSV effect is concentrated; pooling dilutes it | adopted | `strata.json` (A/F) from D, hashed before timing; H24a/H25a in A, H24b equivalence in F; meta-regression H28 on `predictions.json` (1.6, 4.4, 6.5) |
| 5 | ST major | Cr > B missing; C ≡ Cr uninformative | adopted | H25a Cr > B (A) in the CSV family; share ρ with a Fieller CI; slide rule when C > B's lower bound < 6% (6.5) |
| 6 | ST major | unpaired estimator ignores host state; end-of-block re-queue; partial-block rule | adopted (modified) | P: two-stage model with round and process effects (6.2); X: block-adjusted primary (6.3); immediate in-segment retry (2.5); every complete unit or segment is primary (5.6). Modified: under P the "segment" is the process, which the stage-2 model adjusts for directly |
| 7 | ST major | in-process interleaving dropped without a test | adopted (modified) | P6 is the **primary** T design (3.3, 8.1); E1b measures it (5.6). Modified decision rule: P is kept unless clearly noisier (s_P > 1.5·√2·σ̂_E1), not only when it halves the variance, because P is also the build-heat fix (SY blocker 20); E1b sets n instead |
| 8 | ST major | T1 picks c* at random; cycles/op clock-dependent; slope TOST powerless | superseded + adopted | T1 removed: P makes the cool-down common and X uses a fixed 60 s (3.6). GHz TOST ±1% kept as V7 (X); the slope is descriptive with an impact-unit reference (6.8); S7 ln GHz (+ latency probe) adopted |
| 9 | ST major | margin vs predictions inconsistent | adopted | per-dataset predictions from counters frozen before timing (4.4); δ = 3% stated as a decision SESOI; achieved bound always printed; SESOI clearance for superiority; N prediction corrected to 0.997-1.000, G from 200M counters (1.6, 6.5) |
| 10 | ST major | overnight + full = two looks | adopted (option a) | night 1 (+ pre-planned night-2 replicates) is the confirmatory test; session 2 is an independent replication; combined estimate descriptive (6.6, 5.8) |
| 11 | ST major | "unexplained, not claimed" is selective reporting | adopted | every estimate reported; "inconsistent with the counter prediction" label; studentized residual > 3 flag (6.5) |
| 12 | ST minor | E1 decided on a point σ; the 5-8% band discards valid tests | adopted | E1 is go/no-go only, on the 80% upper bound (SY 33); tests run at any σ̂_live with realised power printed; GHz SD, on-CPU, chunk-CV classes recorded (5.2, 6.7) |
| 13 | ST minor | COLD with 3 runs fails 15% of nights | adopted | COLD in all 3 X blocks (9 runs); upper-bound rule (3.5, V5) |
| 14 | ST minor | W1 gate noise-dominated | adopted | W1 dropped; W = 4M fixed and may only go up (3.4) |
| 15 | ST minor | one pooled σ̂ assumes equal variances | adopted | per-dataset σ̂_d from the stage-2 / block model; per-cell σ̂ and Brown-Forsythe; S3 with per-cell variances (6.2) |
| 16 | ST minor | per-dataset multiplicity unspecified | adopted | intersection-union for "all ten", Bonferroni over the named k, slide-wide simultaneous intervals (6.6) |
| 17 | ST minor | I² too imprecise | adopted | τ̂ with Q-profile CI; no I² trigger (6.4) |
| 18 | ST minor | protocol in /tmp, not frozen; pilot set predictions | adopted | freezes 1-3 with sha256, copied to `$P/prereg`; pilot and night-0 data excluded from confirmatory analyses (5.0, 1.6) |
| 19 | ST minor | carry-over and permutations | adopted | predecessor cell, build type and CPU-seconds logged; S8; per-round permutations asserted not all equal; B ∪ B2 baseline kept (6.3, 8.1-8.2) |
| 20 | SY blocker | build heat not neutralised | adopted (modified) | primary: P removes it (3.6). Fallback items: (b) latency probe adopted; (d) in-process FIFO go-gate adopted; (e) predecessor covariate adopted; (f) tested arms = selectable arms by removing T1 and fixing 60 s. **(a) BH preheat cell replaced** by E0h on night 0: the same question (heat on throughput) with the existing binary, using the worst-case books build, no code, before engineering. **(c) state-gated cool-down made optional** (DECISIONS 9): it needs sensor code, and P makes it unnecessary for the primary estimate |
| 21 | SY blocker | battery and powermode 1; the grep check is silent | adopted | checklist requires AC, >= 80%, `powermode` 2 checked through `pmset -g custom`; powermode and AC logged per sample; R1 requires >= 3.8 GHz (2.2, 2.3, 5.6) |
| 22 | SY major | separate processes keep heat and the two-state mixture | adopted | P6 design (R = 8, 1M + 1M, random build order, round pairing, per-visit V6 telemetry); E1b in-process A/A; reduced separate-process X layer kept as the external check, never mixed (3.3, 5.6, 6.3) |
| 23 | SY major | warm-up evidence from the contended set; page-level claim unsupported | adopted | 3.1-3.2 rebased on A2; page-level framing and -34% removed (footnote only); 80 chunks; COLD prediction -6 to -20% (-13 to -32% at 62.5-125k); prefault kept as insurance; paging measured directly |
| 24 | SY major | qos does not keep P-cores; V4 lacks a P/E split; mach-tick units | adopted | P2 uses RUSAGE_INFO_V6, P-core fraction and runnable time per chunk and visit; invalidation at < 0.99 / > 1%; `mach_timebase_info` conversion validated by the self-test; dip counts (2.5, 8.1) |
| 25 | SY major | hostmon uses decaying pcpu; therm uninformative; paging inferred | adopted | hostmon v2 with cumulative-time deltas, vm_stat deltas, powermode, thermal pressure; `pmset -g therm` and `kernel_task` dropped; in-binary fault/vm snapshots (2.3, 8.1) |
| 26 | SY major | "no PMU on the Mac" is wrong; write_heat unexamined | adopted (scheduled after the meeting) | section 9: kperf/xctrace passes A/B, Linux MLP events, manipulations (`--dependent`, THP, 2M/20M/200M, SV/RAW/B, `--skip-observe`), fit. It is session 2 because none of the three requested metrics needs it |
| 27 | SY major | runner: done() skips retries; re-queue breaks contiguity; sessions confounded; D throughput leak; pgrep | adopted | attempt-keyed done(), immediate retries, per-session complete units, renamed D throughput, `pgrep -x`, env padding (2.5, 5.3, 5.8, 8.2) |
| 28 | SY major | Linux taskset/NUMA/uncore/irq/perf | adopted | `--pin-core` after build, numa_balancing 0, uncore pinned, irqbalance stopped with a per-file loop, `idle-set -D 2`, perf >= 5.9, THP-always arm, SV and RAW included, x86 only (2.7) |
| 29 | SY minor | coupon-collector argument wrong | adopted | W justified empirically; hot set named; deep structures DRAM-resident by design (3.1) |
| 30 | SY minor | c1 biased; W1 powerless | adopted | placebo-chunk test; W1 dropped; cold arms as a figure (A2 + COLD) (3.5) |
| 31 | SY minor | ASLR low bits fixed; argv artefact | adopted | env padding; random build order in P; layout-pad check optional (DECISIONS 13) (4.1, 10) |
| 32 | SY minor | BA/CA rule always passes; sampling ambiguity; Mops do not transfer | adopted (modified) | the interaction ln(CA/BA) - ln(C/B) reported descriptively, with the power caveat; ambiguity stated; Linux carries AIDB comparability (3.7). **Rejected part:** running the AIDB-length arm inside P. AIDB measures one structure per process, so the comparability arm must be per-process |
| 33 | SY minor | E1 point σ; no expensive-build A/A; pilot had prefault on; state flips | adopted | U80 gate and 1.4826·MAD; C2 on fb in P; R1 passes `--prefault 1` and the prefault history is corrected (3.2, Appendix A); telemetry separates flips from cold starts |
| 34 | SY minor | caffeinate -d; background QoS; QoS clamp unverified | adopted | `caffeinate -ims`; `bg_cpu` logged; `taskpolicy -G -p` per run (2.2, 2.3, 2.6) |
| 35 | SU blocker | compression defined only for G; Nm missing | adopted | method-agnostic S_k + line-fit outcomes at L_root and L_key for B, G/G_key, Nm, C; Nm in D and CL; the Figure 1/4 claims say "measured" (1.2, 7) |
| 36 | SU blocker | root fences cannot see fb's 21 outliers | adopted (modified) | G_key in the hardness tool (P5k) in Tier M; no-throughput statement. **The supervisor's prediction (k = 2 gives RMSE ≈ 141k) is wrong**: reading the file tail shows 21 outlier gaps (9 x 2.05e18, 5 x 1.41e14, 6 x 7.04e13, 1 x 3.87e10), so fb_trim behaviour needs k >= 21. Pre-registered as H8k, with the k = 20 intermediate case |
| 37 | SU blocker | σ never measured quiet; E1 after 11-14 h of engineering | adopted | night 0 on build-fs before any engineering beyond hostmon (5.2); Linux to be arranged now (DECISIONS 4) |
| 38 | SU major | G predictions extrapolated from 2M | adopted | `fences_ba.py` at 200M in CL before predictions and before P3; 2M labelled a different regime; table comparisons reported separately (4.3) |
| 39 | SU major | timed G likely a structural null; k fixed from 2M | adopted | go-rule and k* rule frozen (4.3.2), with price p = 0.25 (sensitivity 1); predicted no-go (H15) |
| 40 | SU major | equal n everywhere; structural nulls timed | adopted (modified) | D before T, strata, stratified tests (#4). **Kept**: structural nulls are still timed in P, because under interleaving they cost about 30 s per process and serve as extra A/A; they stay out of every pool. **Moved**: n = 8 on stratum A goes to session 2, since night 1 cannot hold it |
| 41 | SU major | labels not faithful | adopted | 1.1 labels everywhere; Cr vs B and C vs Cr headlined separately (1.1, 7.6) |
| 42 | SU major | scope too large | adopted (modified) | full version, RAW/RC timing, C++ k sweep, Gs, W1, C4, NG, NCG cut; P3 conditional; E1 night 0, fences first, G_key, Nm, BA/CA added. **Tier M** (about 4-5 h of engineering) is the supervisor's budget. **Tier C needs 14-17 h** in all, because the build-heat and power blockers outrank scope: P6 and V6 telemetry are the cheapest fix that removes the confound, and P needs 5.25 h for all 10 datasets x 7 cells x n = 3, so P + X + calibration (8.7 h) fit one night where the draft's T layer alone took 9.6-11.9 h |
| 43 | SU minor | by-construction equalities listed as hypotheses; NFL 0 B/key needs a clause | adopted | consistency-check table K1-K7 with the key-order clause and the +30% z-order estimate (1.3, 7.3) |
| 44 | SU minor | AIDB-length replication only in the full version | adopted | BA/CA on fb, n = 3, on night 1 (3.7, 5.6) |
| 45 | SU minor | T1 heat source fb, not books; carry-over not logged | superseded + adopted | T1 removed; E0h uses books (441 s) as the heater; predecessor logged (S8) |
| 46 | SU minor | D6 conditional interleaving rule dropped | superseded | interleaving is the primary design; E1 chunk-step classes are still recorded (5.2) |
| 47 | SU minor | baselines and scans only in a threat row | adopted | scope box in 1.1; AIDB's published numbers as labelled context; SV leads |
| 48 | SU minor | hardness CSV panel is not the index's CSV; H3 rests on 2 datasets | adopted | label "hardness-tool CSV construct" (1.2, 7.10); H3 basis stated, D layer decides the other 8 |

**Found while finalising** (not raised by a reviewer):
- **Fence extraction.** `canonicalize` deduplicates keys (types.hpp:80-86), so fences index the deduplicated array.
  `fences_ba.py` therefore seeks only when `unique_rows` = 200M, and streams otherwise (4.3.1).
- **"10 users".** It counts login sessions on this personal Mac, not other people. The supervisor's "load from other
  people's processes" is corrected (2.2).
- **D on build-fs.** The D layer needs no patch, so it runs on build-fs. That moves 5.1 h of machine time ahead of
  the engineering. R1 guarantees build-proto reproduces it.
- **Flags build-fs would reject.** `rss_ba.sh` and every build-fs preset must omit the P1/P2/P6 flags (8.2 item 1).
- **Blinding.** The runner printed `thr=` in its log. Removed (2.5).
- **Flow weights and prefault.** The flow weights are not in `for_each_allocation` (SY 1.4), so prefault does not
  cover N and Nf. This is harmless: 144 B, warmed by the warm-up.

---

## DECISIONS FOR LOUIS (each with the recommended default)

| # | decision | recommended default | why |
|---|---|---|---|
| 1 | Tier M only, or Tier C | **Tier C if the meeting is at least 4-5 days away; otherwise Tier M for the meeting and Tier C right after** | Tier M gives memory, compression, work, noise floor and heat screen in about 4-5 h of engineering; Tier C adds method throughput |
| 2 | power mode for every timed session | **High Power on AC (`sudo pmset -c powermode 2`)**, restored afterwards | fixed clocks policy; fans ramp earlier, so less heat carry-over in X. Confirm what `powermode 1` means in System Settings > Battery |
| 3 | stop the UTM VM for night 0 and night 1 | **yes** | top-3 CPU user in 98% of samples; P needs about 30 GB |
| 4 | arrange the x86 Linux machine | **start now** (2.7) | it is primary if G1 fails, and it is needed for AIDB-comparable numbers anyway |
| 5 | P3 (C++ gap at the root) | **follow the go-rule** (predicted no-go) | 5-8 h for a predicted null; G is still reported exactly from 200M counters |
| 6 | go-rule price of a table comparison | **p = 0.25 root-probe equivalents**, p = 1 as sensitivity | L1 table vs L2/DRAM fence probes |
| 7 | NC (NFL x CSV) throughput | **session 2**; D-layer counters and memory before the meeting | a second CSV build per P process costs about 2.3 h a night |
| 8 | warm-up length | **4M in X, 1M per visit in P, plus the AIDB-length arm on fb** | alternative: 20M in X everywhere (+1.3 h a night) if the supervisor insists on AIDB's numbers |
| 9 | cool-down | **fixed 60 s** | alternative: wait until the SoC temperature is within 2 °C of its pre-build value (cap 300 s), about 1 h of IOHID code; only X would benefit |
| 10 | extra replicates if E1b gives n > 3 | **run night 2** (complete units, pre-planned) | otherwise the nulls are under-powered at σ 5% (6.12) |
| 11 | session 2 content | **stratum-A extension to n = 8, NC timed, alpha sweep, Linux, mechanism layer** | in that order of value |
| 12 | sudo for pmset, mdutil, tmutil (and kperf in session 2) | **yes, typed by you** | none of the checklist works without it |
| 13 | layout-randomisation check (B x 10 with a random 0-64 KiB pad) | **skip unless E1b shows s_P > √2·σ̂_E1** | P already randomises the build order |
| 14 | role of the pilot and night-0 data | **hypothesis-generating and calibration only** | predictions were set after seeing them |
| 15 | who may read throughput before freeze 3 | **nobody** (runner logs no throughput) | blinding is cheap here |
