# Before vs after NFL, CSV and gap removal on SCALE-LI at 200M keys: experimental protocol (draft)

Status: design only. Nothing was built or run to write this (other experiments own the machine). Every number
below is either read from an existing file (cited) or derived in the text. Where a number is a prediction it says so.

Paths used throughout:

```
REPO = /Users/louisvasseur/Downloads/scaleli_sota
S    = $REPO/experimental/scaleli
NB   = $S/results/aidb_ba                       (PLAN.md, run_ba.py, analyze_ba.py, rss_ba.sh, hostmon.py)
SP   = /private/tmp/claude-501/-Users-louisvasseur-Downloads-scaleli-sota/ac6d4251-6384-4db6-8aba-a17e1917a2ab/scratchpad
P    = $NB/proto                                (all outputs of this protocol; create it)
```

What this adds to `$NB/PLAN.md` (which it supersedes where they differ, and which it reuses everywhere else):

1. **Compression is redefined** as compression of the INPUT space (squeezing empty key space out so a line fits),
   not key-storage codecs. Key bytes stay in the memory column as a by-construction null.
2. **A third method, gap removal (G)**: a k-entry table that shrinks the k widest fence gaps before the root
   model predicts, plus its combinations with CSV and NFL.
3. **A thermal fix for the build-heat confound**: an explicit post-build cool-down for every cell, and a per-run
   frequency measurement that proves the cells were timed at the same clock.
4. **Deterministic metrics move to a separate layer (D)**, so the quiet overnight window is spent only on timing.
5. **Outcome-blind run validity** (host monitor plus on-CPU fraction), equivalence tests for every predicted null,
   random-effects pooling and Holm correction. All of it is pre-registered here.

---

## 0. One-page summary

**What we claim** (section 1 gives the exact hypotheses, H1-H27):

| metric | NFL (feature) | CSV (targets) | gap removal G (input axis, root) | kind of evidence |
|---|---|---|---|---|
| memory (B/key) | +0 (by construction) | +0.799 accounted, +1.00 real (fb) | +k x 32 B = 2.6e-6 B/key (null in practice) | exact; RSS check on fb |
| compression of the input space | none: the key axis is warped, not compressed, and key-order linearity gets worse | none on the input axis (D = 0); it smooths the targets instead | the only method that compresses it: D_k = fraction of key span removed; root-fence RMSE falls where gaps concentrate | exact |
| work per lookup (comparisons) | N: +0 (+0.02-0.05 transforms); Nf: +1 transform | root probes cut where root fences are adopted (fb 28.99 -> 21.2) | root model probes change by -1.5..+0.5; the table adds ceil(log2(k+1)) = 5 cheap comparisons | exact |
| throughput (ratio to B) | N equivalent to 1 (±3%); Nf 0.85-0.96 | pooled 1.01-1.06 | equivalent to 1 (±3%) | statistical |
| build time | +0.5-1.0 s | +160-450 s (16 threads) | +<0.1 s | exact (median of n runs) |

**Experiments, in order:**

1. **Engineering (once).** Build one new binary (`build-proto`) from a copy of the sources, with three patches:
   a cool-down, cycle and instruction telemetry, and gap removal at the root.
   - Regression gate R1: with defaults it must reproduce the pilot bit for bit.
   - Without the gap patch, G is reported from probe counts only, using the Python prototype on the 200M fences.
2. **Calibration, evening, quiet host (about 2.0 h).**
   - Q0: quiet-host check.
   - R1: regression.
   - E1: A/A noise, fb B vs B2, 10 + 10 runs.
   - T1: thermal calibration. It picks the cool-down c*.
   - W1: warm-up calibration. It confirms 4M lookups.
3. **D layer, deterministic.** One instrumented run per (cell, dataset), 10 datasets, about 6.6 h. It does not need
   a quiet host, but it must never run beside a timing run. Its outputs:
   - memory and key bytes;
   - probes, transforms, virtual points, root fences, gap shrink;
   - build time;
   - the checksum reference.
4. **T layer, timing, quiet host, overnight.** Three blocks: 9 cells x 10 datasets, plus fb references and cold
   references.
   - Wall-clock: about 9.6 h at c* = 30 s, 11.9 h at c* = 60 s.
   - Uninstrumented; full warm-up; cool-down c*.
5. **Extras (about 0.7 h).**
   - E5: steady-state RSS, fb B vs C.
   - Root-fence compression metrics from the 200M files.
   - Completion of the AIDB hardness CSV scope.
6. **Full version.** Five more timing blocks (n = 8), NC/GC/NG timed with n = 4, root-alpha-4 CSV on planet and osm,
   an AIDB-protocol replication (20M warm-up, 100M measured), and the deterministic extras.
   - About 47-53 h more machine time.
   - Optional Linux x86 replication.

**Total cost.**

| version | engineering (human) | machine time | of which needs a quiet host |
|---|---|---|---|
| Tier A (no gap patch; G probe-only) | ~6-7 h | ~15.5-17.5 h | ~10-12 h |
| Overnight (G implemented) | ~11-14 h | ~19-21 h | ~12-14 h |
| Full | same | overnight + ~47-53 h | overnight + ~35-42 h |

**Go/no-go gates** (what fails, what happens; full rules in 6.9):

| gate | when | pass | on failure |
|---|---|---|---|
| G0 regression R1 | after the build | new binary at defaults reproduces pilot checksums, fingerprints, memory, learnability and counters exactly; C++ gap probes match the prototype to 1e-9 at 2M | stop; fix the binary |
| G1 quiet host Q0 | evening | idle other-process CPU median <= 25%, p95 <= 60%, no VM | find and stop the source; repeat Q0 |
| G2 A/A noise E1 | evening | sigma_AA <= 5%: GO for everything | 5-8%: overnight, pooled claims only, no equivalence claims; > 8%: no throughput on this Mac (deterministic layer only, Linux for timing) |
| G3 thermal T1 | evening | a cool-down <= 120 s equalises frequency and cycles/op | use 120 s; label C-vs-B throughput "thermal covariate flagged" |
| G4 warm-up W1 | evening | 1M-lookup arm already within -5% at chunk 1; cold arm shows >= -15% | raise W to 8M lookups |
| G5 determinism | after D and T | checksums, fingerprints and signatures consistent; separability holds | stop; no numbers until explained |
| G6 acceptance | morning | live A/A, pooled warm-up, thermal, invalid-run rate <= 10% | section 6.9 says exactly what is still reported |

---

## 1. Claims to prove

### 1.1 Metric definitions (fixed before any run)

- **Memory.**
  - Accounted bytes per key by component, from `memory_before` over `source_rows` = 200,000,000: key, value,
    metadata, delta, slack.
  - Real bytes per key on fb by the steady-state RSS method (memory_audit.md 2b).
  - Peak RSS is never used.
- **Compression of the input space** (the supervisor's sense). Computed on the 48,829 root fences (fence j = key
  4096·j of the sorted file), where the root model lives.
  - (a) **D_k**: the fraction of the normalised key span removed by gap removal, D = sum over selected gaps of
    w_i (1 - s_i). It is 0 for NFL and CSV by construction.
  - (b) **Root-fence linearity**: the AIDB metrics computed on the fence sequence in the coordinate each method
    gives the root (`scaleli_hardness` on a fence file, see 5.6).
    - RMSE/n_f and ME/n_f of one least-squares line, in fence-rank units.
    - PLA-1 and PLA-8 segment counts.
  - (c) **Region-level smoothness**: sum over regions of `learnability.rank_sse_before/after` (CSV, NFL), and
    `tail_conflicts_*_mean`.
  - (d) The **AIDB five** on all 200M keys, as a separate panel: already measured in `hardness_ba.md`, plus G in the
    full version.
- **Work per lookup.**
  - comparisons/op = (root_probes + gap_probes + coordinate_probes + fence_probes + key_at_calls) / operations.
    `gap_probes` is new (patch P3) and is 0 without G.
  - Reported next to it: `transform_calls`/op and `cache_lines_per_operation`.
  - Cache lines are not reproducible to better than ±3% (fullscale caveat), so they are never a contrast on their own.
- **Throughput.** `throughput_ops_s` of the whole 5M-lookup timed replay (the pre-registered statistic, as in
  PLAN.md). Contrasts are ratios of geometric means.
- **Build time.** `build_ns` (wall clock, 16 threads), the median over all T runs of the cell; `smoothing_ns` for
  CPU-seconds.
- **Frequency** (new, patch P2). effective GHz = cycles / on-CPU ns over the timed window, and on-CPU fraction =
  on-CPU ns / wall ns.

### 1.2 Hypotheses

The starting point is PLAN.md section 5; G rows are new. "B" is always the learned raw root (`--root model`), never
the binary root (B0). The prediction bases are given after the tables.

**Exact claims** (one deterministic measurement each; verified by equality, not statistics):

| # | metric | comparison | predicted outcome |
|---|---|---|---|
| H1 | key B/key | N, Nf, C, G, all combinations vs B | identical to 4 decimals on all 10 (fb 1.9340, osm 4.7138, planet 1.384) |
| H2 | accounted B/key | N, Nf vs B | identical (the 144 B flow is outside the accounting; reported as `flow_bytes`) |
| H3 | accounted B/key | C vs B | metadata +8 B x virtual_points + 4 B x root slots: **+0.799 B/key** wherever every region saturates its 409-point budget (predicted: all 10); root fences +0.001-0.005 |
| H4 | real B/key (RSS) | C vs B, fb | real(C) - real(B) = **1.00 ± 0.02 B/key** (0.799 accounted + 0.20 hidden `virtual_features` capacity) |
| H5 | accounted B/key | G vs B | + (32·k_eff + 16) B = 528 B at k = 16 (2.6e-6 B/key); 0 when the root falls back to binary (table cleared) |
| H6 | accounted B/key | NC vs C, GC vs C, GCr vs Cr | additive: NC = C; GC = C + table; GCr = Cr + table |
| H7 | D_k (input compression) | G vs B | D_16 > 0 on all 10. Ordering follows the 2M top-16 gap share (osm 0.60 > books 0.27 > planet 0.12 > ... > fb 0.036): Spearman >= 0.7. D_16(200M) <= the 2M share on every dataset. Largest on osm |
| H8 | root-fence RMSE/n_f, ME/n_f | G vs B | fall on gap-concentrated sets (osm, books, planet); change < 5% on fb, history, stack, wise |
| H9 | PLA-eps on fences (and on all keys, full version) | G vs B | \|dPLA\| <= 2k = 32 segments at every eps, by construction (see below): local metrics cannot see G |
| H10 | input axis | N vs B | not compressed; non-monotone, so the root rejects it for virtual fences (index.hpp:406) on all 10; region tail conflicts worse (fb 31.32 -> 38.89, osm 25.79 -> 27.91); NFL's switch accepts 2-5% of regions |
| H11 | input axis / targets | C vs B | D = 0 on the input axis; region rank SSE falls on all 10 (region RMSE fb 318 -> 265, osm 519 -> 404 in hardness_ba) |
| H12 | comparisons/op | N vs B | fence ±0.01, root unchanged, transforms 0.02-0.05/op |
| H13 | comparisons/op | Nf vs B | +1.00 transform/op, fence +0.01, lines/op +4 to +7 |
| H14 | comparisons/op | C vs B | fence -0.4 to -0.6; fb root 10.80 -> 3.42 (comparisons 28.99 -> 21.2); osm root falls back |
| H15 | comparisons/op | G vs B | root model probes change by -1.5 to +0.5; gap_probes = 5/op; total comparisons rise on >= 8/10; the 13.05 floor is unchanged in every cell |
| H16 | comparisons/op | GCr vs Cr | root probes change by -1.6 to +0.2 (largest on osm, planet); total higher by 3.4-5 on >= 8/10 |
| H17 | root choice | Gs (selector, charged 1 probe per table comparison) vs B | the selector declines G on >= 8/10, so the layout equals B (a structural null and an extra A/A) |
| H18 | separability | GC vs (GCr root counters, C region counters), fb | exact equality: root and region levels are independent |
| H19 | build time | each cell vs B | N +0.5-1.0 s; C 28-70x B (184-451 s); Cr +3.6 s (fb) to <= 18 s; G +<0.1 s |
| H20 | AIDB five on all keys | N, C vs B | as measured in hardness_ba.md (no re-run except the missing CSV scopes); separate panel |

**Statistical claims** (95% CIs; every predicted null is tested for equivalence):

| # | metric | comparison | predicted outcome | test |
|---|---|---|---|---|
| H21 | throughput | B2 vs B (A/A, negative control) | 1.000 | gate: 95% CI contains 1 and TOST within ±3% |
| H22 | throughput | N vs B | 0.98-1.00 | **equivalence** within ±3% pooled |
| H23 | throughput | Nf vs B (positive control) | 0.85-0.96 | superiority (slower) |
| H24 | throughput | C vs B | pooled 1.01-1.06; fb 1.03-1.10; fall-back datasets 0.99-1.01 | superiority pooled; per dataset descriptive |
| H25 | throughput | C vs Cr (region virtual points alone) | 0.99-1.01 | **equivalence** ±3% pooled |
| H26 | throughput | G vs B; GCr vs Cr; NC vs C (full) | about 1.00 | **equivalence** ±3% pooled; interactions as bounds |
| H27 | throughput | SV vs B (positive control) | 1.15-1.50 faster | superiority; RC vs SV (fb) estimated two-sided, 0.90-1.10 |

**Warm-up and thermal checks** (statistical; they validate the protocol rather than make claims):

| # | metric | comparison | predicted outcome | test |
|---|---|---|---|---|
| W | first-chunk indicator c1 | warm runs (pooled) vs COLD | warm within ±3%; COLD -17 to -21% | equivalence; COLD is the sensitivity check |
| T | effective GHz, GHz slope | C vs B | equal within ±1%; slope difference within ±0.002/chunk | equivalence |

**Basis for the predictions.**

- **H1, H2, H10-H14 and the N, C throughput predictions:** PLAN.md section 5 and the pilot (`tables.md`):
  - blocks are encoded before models (index.hpp:213-238);
  - virtual points 19,970,703 = 409 x 48,828 + 51 on fb;
  - fb comparisons 10.80 + 5.03 + 5.14 + 8.02 = 28.99.
- **H3:** 19,970,703 x 8 B / 2e8 = 0.7988 B/key.
- **H4, hidden capacity:** (512 x 48,828 + 64 - 19,970,703) x 8 B ≈ 40 MB ≈ 0.20 B/key (critique.md 2c).
- **H7, H15, H16 come from the 2M prototype** (`$SP/threeblock/armB_parts`, root fences of 2M samples, CSV root at
  alpha 4). They are extrapolations to 200M, labelled as such.
  - At k = 16, G alone changed root model probes by -1.46 (history-w) to +0.45 (osm-u); the median is about -0.2.
  - G->V vs V changed them by -1.60 (osm-u) to +0.13.
  - Adding the 5-comparison table made the total worse than k = 0 in 20 of 20 samples at every k.
  - The priced greedy (`pool_*`) chose G in 2 of 320 configurations, both with a worse total.
- **Alternation buys nothing.** The arm A table (`armA_table.md`, still running) shows that rounds >= 2
  (alternating G, T, V) gain less than the CSV chaos band in almost all rows. So the protocol uses a single pass,
  G then V.
- **H9 (exact argument).** g is affine except at 2k breakpoints, the start and end of each selected gap; inside a
  gap the shrink is linear.
  - Splitting any optimal eps-PLA segmentation of the original sequence at those 2k points gives a feasible
    segmentation of the transformed sequence, because an affine reparametrisation of x preserves a vertical error
    bound. So PLA_new <= PLA_old + 2k.
  - g is invertible with the same breakpoints, so PLA_old <= PLA_new + 2k.
- **H22-H26, the equivalence margin of ±3%.** One flow evaluation (28-66 ns), or 4-8 root probes, is about 3% of a
  1.2-1.7 µs lookup. A smaller effect would not change any design decision.

---

## 2. Machine and host

### 2.1 The Mac (Apple M3 Max, 64 GiB)

Topology, read on 2026-10-02 with `sysctl`:

- 12 performance cores in **two clusters of 6, each with its own 16 MiB L2**, plus 4 efficiency cores.
- 16 KiB pages.

macOS cannot pin a thread. `--qos 1` (user-interactive) keeps the timed thread on performance cores but not on one
cluster. A migration between the clusters loses the L2 (threat 5 in section 9).

**Quiet-Mac checklist.** Do every step before Q0, in this order, and undo it afterwards (the restore list is at the
end of 8.5).

1. **Stop the UTM/QEMU VM.** Shut the guest down in UTM, then quit UTM.
   - `pgrep -fl 'qemu|UTM'` must print nothing.
   - The VM was in the top-3 CPU users in 98% of hostmon samples during every 200M measurement so far.
2. **Quit heavy and periodic apps.**
   - Browsers, ChatGPT, Slack, Docker Desktop, LM Studio / ollama, Xcode, VS Code, DaVinci Resolve and Orca.
   - Any agent session that runs jobs; the threeblock arms in `$SP/threeblock` must have finished.
   - `pgrep -fl -i 'docker|lmstudio|lms |ollama|python3|scaleli_'` must show only hostmon and the runner.
3. **Power.**
   - On the power adapter: `pmset -g batt` says "AC Power".
   - Low Power Mode off: `pmset -g | grep lowpowermode` gives 0.
   - Record `pmset -g` (including `powermode` where present) and keep it unchanged for the whole session.
   - Lid open, machine flat on a hard surface, same position all night. Write the room temperature into the log.
4. **Sleep and idle work.**
   - Power Nap off: `sudo pmset -a powernap 0`.
   - The runner is wrapped in `caffeinate -dims -w <runner pid>`.
   - Automatic macOS and App Store updates paused for the session.
5. **Spotlight and Time Machine.**
   - `sudo mdutil -a -i off`.
   - Time Machine: `sudo tmutil disable` (or turn automatic backups off in Settings).
   - Pause iCloud Drive and Photos sync.
6. **Monitoring.** Start `hostmon.py` at background priority, `taskpolicy -b`, so it stays on efficiency cores.
   - It samples every 5 s: other-process CPU, top-3 foreign processes, the count of `scaleli_*` processes,
     load average.
   - Every 60 s it also records `pmset -g therm` and `pmset -g batt` (patch in 8.4).
7. **Hands off.** Do not use the Mac during T or calibration runs. Remote login only to read logs.

**Q0 pass rule** (10 idle minutes, nothing else running):

- median other-process CPU <= 25%, p95 <= 60%, max <= 150%;
- no single foreign process above 30% in more than 2 samples;
- none of qemu, UTM, Docker, LM Studio or ollama in any sample;
- on AC power.

The load average is logged but not gated (macOS idles at 1-3).

**Per-run gating (runner).** All rules are outcome-blind: they are decided before throughput is read, from external
signals only.

- **Before each run (quiet gate).**
  - The last 6 hostmon samples (30 s) all have other CPU < 60% and top foreign process < 30%, and no other
    `scaleli_*` process.
  - Otherwise wait, polling every 15 s, and log the wait.
  - Never skip a job.
- **After each run (validity).** The run is **invalid** if any of these holds:
  - (a) any hostmon sample in [`t_warmup_start`, `t_timed_end`] shows other CPU >= 100% or a foreign process
    >= 50%. `kernel_task` is excluded here and reported separately as a thermal signal.
  - (b) another `scaleli_*` process was alive during the run.
  - (c) on-CPU fraction over the timed window < 0.99 (the descheduling detector, patch P2). The threshold is
    lowered to the 1st percentile of E1 if E1's median is below 0.995.
  - (d) `prefault_ns` > 1 s (paging; the normal value is 60-80 ms).
  - (e) AC power lost, or `pmset -g therm` reported a thermal warning during the run.
  - (f) non-zero exit or timeout. A timeout is recorded as a result row ("did not build in X s") and re-queued once.
- **What happens to an invalid run.** It is re-queued at the end of its block with the same seed, at most 2 retries,
  then marked missing.
- **Session pause.** If 3 of the last 10 runs are invalid, the runner pauses until hostmon shows 10 consecutive
  quiet minutes, and logs an alert line. It never aborts mid-block on its own.
- **Hard stop.** On battery power the runner stops after the current run.

**Host fingerprint**, recorded once per session into `$P/host.txt`:

- `sw_vers`;
- `sysctl -n machdep.cpu.brand_string hw.memsize hw.pagesize hw.perflevel0.physicalcpu hw.perflevel0.cpusperl2
  hw.perflevel0.l2cachesize hw.perflevel1.physicalcpu`;
- `pmset -g`, `pmset -g batt`, `pmset -g therm`;
- `shasum -a 256` of both binaries;
- `/usr/bin/c++ --version`;
- the patch diff of `scaleli_proto` against `S`.

### 2.2 The Linux alternative (x86-64, the AIDB setting)

Use a dedicated machine: no other users, no VM neighbours. The settings:

| item | setting | why |
|---|---|---|
| core isolation | kernel cmdline `isolcpus=2 nohz_full=2 rcu_nocbs=2`; move IRQs off core 2 (`echo <mask without core 2> > /proc/irq/*/smp_affinity`); take core 2's SMT sibling offline (`echo 0 > /sys/devices/system/cpu/cpuN/online`) | removes the scheduler, timer and interrupt noise that macOS cannot remove |
| pinning | `taskset -c 2` for the timed thread; `numactl --cpunodebind=0 --membind=0` | AIDB pins its worker; avoids cross-node memory |
| frequency | `cpupower frequency-set -g performance`; turbo off (`echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo`, or `echo 0 > .../cpufreq/boost` on AMD) | a fixed clock makes cycles and ns interchangeable |
| huge pages | THP `never` (`echo never > /sys/kernel/mm/transparent_hugepage/enabled` and `.../defrag`) | AIDB: "Hugepages are disabled"; a fixed policy for every cell |
| ASLR | left on | layout effects become noise across processes rather than a fixed bias (threat 6) |
| build | same CMake Release flags (`-O3 -DNDEBUG`, SCALELI_NATIVE OFF); record the compiler | — |
| bench | `--build-threads` = number of non-isolated cores; `--qos` is ignored on Linux | — |

**Counters to record** with `perf stat`, scoped to the timed window through patch P4 (`--perf-ctl`, a control FIFO;
perf is started with `-D -1 --control fifo:ctl,ack`):

- `cycles, instructions, branches, branch-misses, L1-dcache-load-misses, LLC-loads, LLC-load-misses, dTLB-loads,
  dTLB-load-misses, page-faults, context-switches, cpu-migrations`;
- if perf reports less than 100% running time for any event (multiplexing), split them into two passes.

Note: x86 `long double` is 80-bit, while it is 64-bit on Apple silicon. `LinearModel` accumulates in `long double`,
so fitted models, virtual-point placements and probe counts may differ at the sub-probe level between the
platforms. Lookup results do not, so the checksums must still match.

### 2.3 Which result needs which machine

| result | Mac, any load (never during T) | Mac, quiet | Linux x86, isolated |
|---|---|---|---|
| memory (accounted), key bytes, D_k, fence and region metrics, probes, transforms, virtual points, root choice, AIDB five | **primary** (D layer) | — | recomputed only as a cross-check; sub-probe differences are expected and not interpreted |
| build time | from T runs (quiet) | **primary** | — |
| real RSS (E5) | — | **primary** (the method is calibrated on macOS; the allocator cliff is macOS-specific) | not needed |
| throughput ratios H21-H27 | — | **primary**, if G2 passes (sigma_AA <= 5%; pooled-only at 5-8%) | replication on a second microarchitecture (4 KiB pages, fixed clock); **primary if G2 fails** |
| AIDB-comparable absolute Mops, "same protocol as AIDB" | — | — | **required** |
| hardware cache/TLB misses explaining a throughput effect | — | — (no PMU access without sudo; only cycles and instructions via rusage) | **required** |

---

## 3. The warm-up (and cool-down)

### 3.1 What it must achieve, from first principles

The index is 2.0-2.8 GB: 10-14 B/key x 2e8. A lookup reads the root, the region header, block descriptors, the key
arena and the value. Steady state is the state a long uninterrupted stream of uniform lookups converges to. The
timed window must start in it.

1. **Page residency and mapping.**
   - All index pages were written during the build, so they are resident. The 1-2 major faults per run measured with
     and without prefault confirm this.
   - The measured cold transient with `--warmup 0 --prefault 0` was -34% in the first 125k lookups. With
     `--prefault 1` it was -5% (fb) and about 0% (osm). So about 30 points of the cold penalty are page-level: a
     first-touch cost per page, removed by touching every page once. The exact mechanism (page-table or VM state)
     is not identified.
   - Prefault (one byte per 4 KiB, 59-78 ms) removes it, and so does any warm-up that touches every page.
2. **Hot metadata in cache and TLB.**
   - The structures every lookup reads: the `regions_` pointer vector (48,829 x 8 B = 390 KB), the Region headers
     (48,829 x 280 B = 13.7 MB), the root slot table (<= 1 MB), the gap table (528 B). Together about 14 MB, which
     fits one 16 MiB P-cluster L2.
   - Uniform lookups hit region j with probability 1/R, R = 48,829. After W lookups the expected fraction of regions
     never touched is e^(-W/R): 7.7% after 125k, 0.6% after 250k, 1e-9 after 1M.
   - Requirement: fewer than 0.01 untouched regions expected, i.e. W >= R ln(100 R) = **0.75M lookups**.
   - This coupon-collector curve reproduces the remaining 5% transient and its length (125k-250k lookups).
   - Block descriptors (100 MB), keys and values do not fit any cache. Their steady state is a miss.
3. **LRU equilibrium.** Each lookup brings in about 10-20 new lines (lines/op is 21-50). So a 16 MiB L2 (262k lines)
   turns over in 15-25k lookups and the system-level cache in about 1e5. This is reached long before item 2.
4. **Branch predictors.** The search branches are data-dependent and about 50% unpredictable by nature. The
   predictor reaches that equilibrium within about 1e4 lookups of the same code path. Warm-up queries come from
   the same generator with a different seed (`warmup_keys`, seed ^ 0x5741524d), so they train the same paths
   without replaying the measured keys.
5. **CPU frequency settled.** Apple DVFS ramps within milliseconds. Any warm-up of at least 1 s at full
   single-thread load leaves the timed core at its steady clock. Patch P2 measures this per chunk.
6. **Thermal state equal across cells.** This is not the warm-up's job and it cannot do it. A 4-5 s single-thread
   warm-up does not undo a 160-450 s all-core build. It is handled by a cool-down (3.4) and verified by measured
   frequency.

### 3.2 Evidence it rests on

- **Cold start without prefault** (`$SP/fullscale` cold/, K = 16 at 2M ops, so 125k-op chunks; per-run figures from
  the warm-up task output).
  - Chunk 1 was -34.4/-38.0/-34.3% on fb and -34.2/-39.7/-21.4% on osm.
  - Chunk 2 was about -8%; chunk 3 was back at the median.
  - Translated to 5M ops / 16 chunks (312.5k-op chunks): the extra time is
    125k/556k + 125k/806k - 2 x 125k/881k = 0.096 s on a 0.355 s chunk, a **-21%** first chunk. PLAN.md expected
    -17 to -19%.
- **With prefault and no warm-up:** fb -2.5/-5.1/-5.3%, osm -2.4/+6.0/+0.4%. Prefault cost 59 ms (fb) and 78 ms (osm).
- **With a 2M workload-mode warm-up** (pilot, 24 runs, fb and osm): c1 = **-0.3% (SE 2.0%)**.
- **Warm-up of 200k vs 2M** (fullscale matrix, contended host): no factor resolvable within a 28-30% spread.
- **`steady_state_ratio` (last/median) is broken as a warm-up indicator.** It read 1.039 and 0.980 on runs whose
  first chunk was -34%. It is not used. The indicator is c1 = chunk 1 / median(chunks 2..K) - 1.
- **Per-run c1 noise.** Its SD is about 10% at 312.5k-op chunks on the contended host: pilot range -0.31 to +0.18.
  One run's c1 means little; c1 is judged pooled.

### 3.3 The sequence inside every timed run (order is part of the protocol)

```
load file + canonicalize + workload (~40 s, untimed)
-> build (pass 1, 16 threads; build_ns)
-> learnability + memory_before
-> COOL-DOWN c* seconds (sleep; patch P1)                    [equalises the build heat]
-> freq probe (pre)                                           [P2]
-> prefault (one byte per 4 KiB of every index allocation)    [page level; prefault_ns = paging sentinel]
-> WARM-UP: 4,000,000 workload-mode lookups, seed^0x5741524d  [cache, TLB, predictors, clock]
-> TIMED: 5,000,000 uniform lookups, 16 chunks, cycles/instructions/on-CPU per chunk [P2]
-> freq probe (post)
-> (D layer only: instrumented pass on a fresh build)
```

### 3.4 Every number, and why

| parameter | value | justification |
|---|---|---|
| warm-up mode | `workload` | same generator and distribution as the trace, different seed (3.1 item 4); `strided` warms a pattern no measured workload uses |
| warm-up length | **4,000,000 lookups** | about 4.0-5.5 s on learned cells (0.73-1.0 Mops), 3.3-4.0 s on SV. Each region touched 82 times (needs >= 0.75M, 3.1 item 2), each 128-key block 2.6 times. 16x the observed 250k transient. The 2M arm already passed (-0.3% ± 2.0%). Same value as PLAN.md, so pilot comparisons stay valid. Cost over 1M: 3-4 s per run, about 0.3 h a night |
| warm-up measured in | lookups (seconds reported) | the requirement (coverage per region) counts lookups; seconds would give faster cells more coverage, which is harmless but no longer one number |
| prefault | **1** (PLAN.md had 0) | removes the about 30-point page-level part even if the warm-up were short. Costs 60-80 ms. `prefault_ns` becomes a free per-run paging sentinel (invalid above 1 s). It comes after the cool-down, so any page-out during the idle is caught and repaired before timing |
| cool-down c* | chosen by T1 from {0, 30, 120} s, applied identically to **every** cell; c* = max(10, smallest passing value); **60 s if T1 is not run**; 120 s and a flag if none passes | must exceed the die and heat-sink settling after a 160-450 s all-core build. Apple die temperatures fall within seconds; laptop heat sinks and fans within 1-2 min; 60 s is the conservative prior. Identical for all cells, so idle time itself is never confounded with the cell |
| timed ops | **5,000,000** | 16 chunks of 312.5k, so chunk 1 is longer than the transient and a failure would show as a deficit of about -21%. More ops do not help: sigma 6.4% at 2M, 6.2% at 5M, 6.1% at 10M ops, because the noise is per process. The timed window is about 5-7 s |
| chunks K | **16** | continuity with the COLD reference and the pilot; 312.5k-op chunks keep per-chunk noise moderate |
| statistic | whole replay `throughput_ops_s` | pre-registered in PLAN.md; chunks 2-16 is the warm-up sensitivity (6.10) |
| seeds | workload seed 1000 + block; warm-up seed derived by the binary | paired traces within a block; different traces across blocks |

### 3.5 The per-run check that proves it worked, and what happens to a failing run

The warm-up and frequency checks are computed for every T run. All runs are kept. **Flags never drop a run**, because
c1 is correlated with the outcome and dropping on it would bias. Only the outcome-blind rules of 2.1 invalidate runs.

| check | definition | per-run flag | cell / pooled gate | on gate failure |
|---|---|---|---|---|
| warm-up | c1 = chunk1 / median(chunks 2..16) - 1 | c1 < -max(0.20, 3·sd_chunk from E1) (the cold reference is about -0.2) | pooled mean c1 over all warm T runs within ±3% (TOST, SE about 0.6% at 290 runs); each cell's mean c1 > -5%; flagged runs <= 10% per cell | the primary statistic switches to chunks 2-16 for **all** cells (pre-registered fallback) and the failure is reported |
| clock settled | GHz(chunk 1) / median GHz(chunks 2..16) | outside ±2% | <= 10% flagged per cell | report; investigate DVFS |
| thermal | effective GHz over the timed window; slope of log chunk throughput over chunks 2-16 | none | C vs B: GHz ratio TOST ±1%; slope difference TOST ±0.002/chunk | C-vs-B throughput is labelled "thermally confounded"; cycles/op of C in T1 is reported |
| paging | `prefault_ns` | > 1 s means **invalid** (outcome-blind) | — | re-queue |

**COLD control** (`--warmup 0 --prefault 0`, fb, osm and planet, block 1). Prediction: c1 between -17 and -21%. If
COLD does not show at least -15%, the indicator is insensitive on this host and the warm-up claim is not made.

### 3.6 Where and why this deviates from AIDB (20M warm-up, 100M measured, pinned, x86 Linux)

| item | AIDB 4.1 | here | why |
|---|---|---|---|
| warm-up | 20M lookups | 4M lookups (4-5.5 s) + prefault + cool-down | coverage needs 0.75M; the transient is <= 250k; 20M would add about 20 s per run (+1.6 h a night) for no measurable state change |
| measured | 100M lookups (repeats unstated) | 5M per process x n independent processes (15M per cell per dataset at n = 3, 40M at n = 8) | noise is per process (σ hardly moves from 2M to 10M ops); only more processes shrink it |
| thread placement | pinned core, Xeon Gold 5118, hugepages off | macOS QoS, no pinning, 16 KiB pages | the available machine; Linux replication in 2.2 |
| lookups | random lookups of the dataset's keys | uniform with replacement over all 200M loaded keys, miss 0 | same distribution |
| bulk load | all keys | `--load-ratio 1` (all 200M; the default 0.75 would hold back 50M) | same |

**AIDB-protocol replication (full version).** Cells BA and CA = B and C with `--warmup 20000000 --ops 100000000
--chunks 100`, on fb and osm, n = 3.

- It tests whether the shortened protocol changes the C/B ratio.
- Pass: the BA/CA ratio's 95% CI overlaps the main sweep's C/B CI on the same dataset.
- Memory: trace 3.2 GB + warm-up keys 160 MB, well within 64 GB.

---

## 4. The cells

### 4.1 Common flags

All runs use the new binary `$REPO/build-proto/scaleli_bench` (section 8.1). Never mix binaries in a table.

```
COMMON  = --format sosd --dtype uint64 --load-ratio 1 --miss 0 --query-distribution uniform
          --profile read_only --ops 5000000 --warmup-mode workload --prefault 1 --chunks 16 --qos 1
          --build-threads 16 --verify 0 --latency 0 --cycles 1 --seed $((1000+block))
T layer = COMMON --warmup 4000000 --instrument 0 --cooldown-s $COOL
D layer = COMMON --warmup 1000000 --instrument 1 --cooldown-s 0           (block = 1, so seed 1001)
PK      = --policy min_bytes --routing rank
F(d)    = $S/results/aidb_flowv2/flows_free/<d>_<s512|s64>_t2000.txt    (period map in run_ba.py: s512 fb/osm/planet)
data(d) = $S/data/external/gre/<d>  (<d>.sorted for covid genome history libio planet stack wise)
```

D-layer runs use seed 1001 and 5M ops, the same trace as T block 1. So each D run's `result_checksum` must equal the
T block-1 checksum for that dataset.

### 4.2 Cell table

| cell | extra flags | what it is | layer, overnight | layer, full |
|---|---|---|---|---|
| **B** | `PK --root model` | BEFORE: learned raw root (falls back to binary by itself) | D, T x3 | T x8 |
| **B2** | identical to B | live A/A, negative control | T x3 | T x8 |
| **N** | B + `--flow F --flow-bypass 1 --flow-cost 0` | after NFL: faithful flow, NFL's own switch, root offered the flow free | D, T x3 | T x8 |
| **Nf** | B + `--flow F --flow-bypass 0 --flow-cost 0` | flow forced in every region; **positive control** (known extra work) | D, T x3 | T x8 |
| **C** | B + `--virtual-alpha 0.1 --root-alpha 0.1` | after CSV: Algorithm 1 in regions + virtual fences at the root, the paper's alpha | D, T x3 | T x8 |
| **Cr** | B + `--root-alpha 0.1` | CSV at the root only (cheap build). C vs Cr isolates the region virtual points | D, T x3 | T x8 |
| **NC** | C + `--flow F --flow-bypass 1 --flow-cost 0` | NFL + CSV | D | D, T x4 |
| **G** | B + `--root-gaps 16 --root-gaps-mode force` | after gap removal: k = 16 table, shrink to the median gap, root model fitted on g(x) | D, T x3 | T x8 |
| **Gs** | B + `--root-gaps 16 --root-gaps-mode auto` | gap removal offered to the root selector at 1 probe per table comparison (deployable version) | D | D (+T if its signature differs from B) |
| **G1, G4, G64, G256** | B + `--root-gaps k --root-gaps-mode force` | k sweep for the compression figure | D | D |
| **GCr** | Cr + `--root-gaps 16 --root-gaps-mode force` | gap removal then CSV at the root (single pass G -> V) | D, T x3 | T x8 |
| **GC** | C + `--root-gaps 16 --root-gaps-mode force` | gap removal + full CSV | D (fb: separability check H18) | D all, T x4 |
| **NG** | N + `--root-gaps 16 --root-gaps-mode force` | NFL (regions) + gap removal (root) | — | D, T x4 |
| **NCG** | NC + `--root-gaps 16 --root-gaps-mode force` | all three | — | D |
| **SV** | `--index sorted_vector` | plain binary search, 16 B/key; **positive control** | D, T x3 | T x8 |
| **RAW** | `--policy raw --routing rank --root model` | uncompressed blocks | D, T x3 fb only | D, T x8 all |
| **RC** | RAW + `--root-alpha 0.1` | raw blocks + CSV root (critique E2) | D, T x3 fb only | D, T x8 all |
| **B0** | `PK` | binary root, host-default reference | D fb | D all |
| **COLD** | B with `--warmup 0 --prefault 0` | cold reference for the warm-up figure | T block 1 (fb, osm, planet) | T blocks 1-3 |
| **C4** | B + `--virtual-alpha 0.1 --root-alpha 4` | CSV at the established root budget | — | D + T x3 (planet, osm) |
| **BA, CA** | B / C + `--warmup 20000000 --ops 100000000 --chunks 100` | AIDB-protocol replication | — | T x3 (fb, osm) |
| **Nm** | B + `--flow flows_monotone/<d>_mono.txt --flow-bypass 1 --flow-cost 0` | realisable monotone flow | — | D |
| sweep cells | PLAN.md `plan sweep` (Cr004, Cr04, Cv05, Cv2, Cr4) | deterministic CSV alpha sweep | — | D |

**Controls and what each controls for:**

- **Negative:**
  - B2: an identical rebuild.
  - Gs, wherever the selector declines G: a structural A/A.
  - Every structural null, i.e. a cell whose signature equals B's.
- **Positive:**
  - Nf, a known slowdown. If Nf vs B cannot be resolved, the protocol lacks the power for the C effect too.
  - SV, a known speed-up.
  - COLD, a known first-chunk deficit.
- **Reference:**
  - RAW, RC: the cost of codec compression.
  - B0: the binary root.

### 4.3 Gap removal: what exists, and what must be added

**What exists (Tier A, probe-only).** The Python prototype `$SP/threeblock/threeblock.py`:

- `gap_block`: the k largest gaps with width > median, shrink factor s_i = W_med/W_i, the map g(x), exact inverse,
  and strictness asserted.
- `score`: a replica of `fit_root` that counts probes, and `ceil(log2(k+1))` table probes.

It runs on root fences. For 200M, the new script `fences_ba.py` (8.4):

- extracts the 48,829 fences from each sorted file by seeking to key 4096·j;
- runs G for k in {0, 1, 4, 16, 64, 256};
- writes D_k, root model probes, gap probes, and the fence files for the hardness tool.

Its probe counts are validated against the C++ `learnability.root_probes_raw` of cell B from the D layer, to
±0.01 probes. **Tier A gives every exact G claim except H16-H18 and no throughput.** A CSV root on g-features needs
the C++ greedy; the Python greedy is O(budget x n) and infeasible at n = 48,829.

**What must be added for timing (Tier B, patch P3 in `include/scaleli/index.hpp`, `Config` at index.hpp:43, and
`src/benchmark.cpp`).**

1. **Config.**
   - `std::size_t root_gaps = 0`.
   - `enum GapMode {Force, Auto} root_gaps_mode = Force`.
   - `double gap_cost = 1.0` (probe-equivalents per table comparison, auto mode only).
2. **`fit_root`, building the table.**
   - After computing the raw feature x_j = tmp.normalized(fences[j]), if root_gaps > 0, build a `GapTable` by
     porting `threeblock.GapMap` exactly: widths w_j = x_{j+1} - x_j; W_med = median; select up to k largest with
     w > W_med (ties go to the lower index); s = W_med / w.
   - Store sorted arrays a (start), w, s, C (cumulative shrink below); total D; scale = 1 - D.
   - g(x) = (x - C_j - (1 - s_j)·min(x - a_j, w_j)) / scale, where j is the last a_j <= x; if there is none,
     x / scale.
   - Assert strictly increasing on fences and fence midpoints.
3. **`fit_root`, choosing the candidate.**
   - Candidates gain a `gapped` dimension, applied to the raw feature only: {raw, gapped-raw} x {ranks, vp}. The flow
     candidates are untouched.
   - **Force** mode offers only the gapped candidates plus the binary fallback (flow root candidates are dropped).
     **Auto** mode offers everything and charges gapped candidates `gap_cost·ceil(log2(k_eff+1))`.
   - The vp candidate runs `smooth_cdf` on g-features, which are monotone, so index.hpp:406 lets it through. This is
     the G -> V single pass.
   - If the selected root is binary, the table is cleared (0 bytes).
4. **`root_feature(k, s)`.** If the table is active: x = root_model_.normalized(k); binary search over a for j
   (k = 16: at most 5 comparisons, a fits two cache lines); return g(x). Each comparison adds `++s->gap_probes`, and
   `s->note_range(table, bytes)` records the lines.
5. **Accounting.**
   - `memory()`: metadata += 32·k_eff + 16 B.
   - `for_each_allocation`: include the table vectors.
   - `QueryStats` (types.hpp:17): new field `gap_probes`.
6. **Output (benchmark.cpp).**
   - Flags `root-gaps`, `root-gaps-mode` and `gap-cost` in the allowed set.
   - JSON `learnability`: `root_gaps` (k_eff), `root_gap_shrink` (D), `root_probes_gap_raw`, `root_probes_gap_vp`.
   - `work_counters.gap_probes`.
   - `--dump-root-gaps file` writes a, w, s, C, D for the hardness tool (P5).
7. **Correctness is automatic.** `locate_from_prediction` ends in an exact fence search, so any g gives correct
   lookups and only speed can change. The checksum must equal B's.

**Effort:** 4-6 h of code. Plus 1-2 h of validation:

- `ctest`;
- `--verify 1` at 2M on all 10 samples for G and GCr;
- **gate R1-G**: the C++ `root_probes_gap_raw` equals the prototype's `G.model_probes` (armB `k_<d>_uniform_<k>.json`)
  to 1e-9 for 4 samples x k in {1, 4, 16, 64}, with root alpha 0 (G alone; the prototype's V used alpha 4, so V
  is not compared).

**Region-level gap removal is out of scope, on purpose.** Inside a region the coordinate step is a fixed binary
search over 32 `slot_begin` values (index.hpp:126-133). Model quality reaches only the fence term (4-5 of about 29
comparisons), so a region-level table could move at most that term.

---

## 5. The run schedule

### 5.1 Order of stages

```
S0 engineering (P1, P2, P3; R1)            human 11-14 h (Tier A: 6-7 h, no P3)
S1 Q0 quiet check                          0.25 h   quiet
S2 calibration: R1 runs -> E1 -> T1 -> W1  1.8 h    quiet, attended (evening)
S3 D layer                                 6.6 h    any load, but never during S2/S4/S5 (best: the day before)
S4 T layer (overnight)                     9.6-11.9 h  quiet, unattended
S5 E5 RSS + fences + hardness completion   0.7 h    quiet for E5; the rest any load
S6 analysis                                minutes
```

Running S3 before S1/S2 is recommended. It shakes out crashes, flow-file paths and timeouts, and it gives real build
times for the T schedule.

### 5.2 Calibration (S1-S2), in this order

| step | runs | settings | time | output and decision |
|---|---|---|---|---|
| **Q0** | none; hostmon 10 min idle | — | 15 min | 2.1 pass rule (gate G1) |
| **R1** | fb, osm x {B, C2p = B + `--root-alpha 0.04`} at pilot settings (seed 11, `--warmup 2000000`, ops 5M, instrument 1, cooldown 0); plus R1-G: 2M samples x {B, G1, G4, G16, G64} | — | 10 min | equal to `pilot_seed11.jsonl` (L0, C2) in `result_checksum`, `trace_fingerprint`, `memory_before`, `learnability` minus `*_ns`, `work_counters` minus `cache_lines`; P2 fields present, effective GHz in [2.5, 4.5] and on-CPU ≈ 1 (else use the P2c frequency probe); R1-G as in 4.3 (gate G0) |
| **E1** A/A | fb: B x10 and B2 x10, random order, T-layer settings, cool-down 30 s (provisional) | T | 30 min | σ_AA = pooled within-cell SD of log throughput (df 18; a 4% estimate means a true 3.0-5.9%, χ² 95%); B2/B CI must contain 1; distribution of on-CPU fraction and per-chunk SD (gate G2). **Run first, so a noisy host fails fast** |
| **T1** thermal | fb: C x {cool-down 0, 30, 120 s} x 2; B x {0, 120 s} x 2 | T, no other change | 38 min | the C build itself (184 s, 16 threads) is the heat. For each cool-down: effective GHz and cycles/op of C against C at 120 s, and GHz against B. Pass at c: GHz within 1% of B **and** cycles/op of C within 3% of C at 120 s (cycles/op compared within the cell, because cells do different work). c* = max(10, smallest passing c); none passes: c* = 120 + flag (gate G3) |
| **W1** warm-up | fb, osm x {W0p0 (warmup 0, prefault 0), W0 (0, prefault 1), W250k, W1M, W4M} x 2; `--chunks 40` (125k-op chunks) | T with cool-down c* | 30 min | mean chunk curves per arm. Transient length L = last chunk where the W0p0 deficit < -3%. Pass: the W1M arm's chunk-1 deficit > -5% and the W0p0 arm < -15% (sensitivity). Else W = 8M (gate G4) |

### 5.3 D layer (deterministic; S3)

- One run per (cell, dataset), `--instrument 1`, seed 1001, 5M ops, cool-down 0, warm-up 1M.
- Cost model per run: about 60 s + 2 x build (the instrumented pass rebuilds the index, benchmark.cpp pass 4).

**Overnight D cells.**

- On all 10 datasets: B, N, Nf, SV, Cr, C, NC, plus G, Gs, G1, G4, G64, G256, GCr (Tier B).
- On fb only: RAW, RC, B0, GC.

**Time.**

- Cheap cell (build about 7 s): 74 s each.
- Cr/GCr (root smoothing <= 18 s): 110 s each.
- C/NC/GC: 60 + 2 x 296 s ≈ 671 s each.
- Per dataset: 10 x 74 + 2 x 110 + 2 x 671 = 2,302 s; x 10 = 23,020 s; fb extras 897 s. **Total 6.6 h.**
- Tier A (no G cells): 4.9 h.

**Full-version D additions (+11.2 h).**

- NCG, GC, NG, B0, Nm, RAW, RC on all 10 datasets.
- PLAN.md `sweep` (3.4 h).
- C4 on planet and osm (about 3 h; 2,068 s of root smoothing per build).

**What D produces**, all from the jsonl:

- memory by component;
- `learnability`: flow regions, virtual points, root model/flow/vp/virtual, root probe estimates, gap k and D, rank
  SSE, tail conflicts, smoothing and transform ns;
- `work_counters`: comparisons and their parts, gap probes, transforms, lines/op;
- `build_ns`;
- the checksum and fingerprint references for T block 1.

### 5.4 T layer (timing; S4)

- **Cells, overnight** (Tier B): B, B2, N, Nf, G, SV, Cr, GCr, C on all 10 datasets; RAW and RC on fb; COLD on fb,
  osm and planet in block 1 only.
- **Cells, overnight** (Tier A): the same without G and GCr.
- **Blocks.** One block is one complete replicate: every T cell on every dataset.
  - Overnight: 3 blocks.
  - Full: 8 blocks; NC, GC and NG in blocks 1-4; C4 and BA/CA in blocks 1-3.
- **Randomisation.**
  - Within a block, datasets are shuffled, and cells are shuffled within each dataset.
  - Shuffle seed 20261001 (as in PLAN.md), logged in the plan header.
  - Cells of one dataset stay contiguous, so that within-dataset contrasts sit close in time. Slow drift over the
    night then confounds datasets, which are never contrasted, not cells.
- **Serialisation.** Strictly one 200M process at a time.
  - The runner waits on `pgrep scaleli_bench|scaleli_hardness`.
  - The D layer, E5 and hardness never overlap T.
  - The instrumented pass never runs in T.
- **Seeds.**
  - Workload seed 1000 + block, shared by all cells in the block.
  - The warm-up seed is derived by the binary.
  - Flow weights and the gap rule are deterministic.
  - ASLR stays on.
- **Ops.** 5M timed, 4M warm-up, 16 chunks (3.4).
- **Stopping.** n is fixed in advance. Stop only at a block boundary, for time, never on results. A partial block is
  analysed only as a sensitivity check.

**Repeats and power.**

- Per-dataset contrast against the pooled B ∪ B2 baseline: SE = σ·sqrt(1/n + 1/(2n)).
- Pooled over k = 10 datasets with equal weights: SE / sqrt(10).
- With a homogeneous effect, the random-effects interval is about 1.15x the fixed one (t at 9 df).

| σ per run | n | per-dataset 95% half-width | pooled 95% half-width | pooled 90% (TOST) half-width | null provable at ±3% pooled if \|true\| < | per-dataset 90% (TOST) |
|---|---|---|---|---|---|---|
| 4% | 3 | ±5.5% | ±1.75% (RE ±2.0%) | ±1.5% (RE ±1.6%) | 1.4% | ±4.7% |
| 6% | 3 | ±8.3% | ±2.6% (RE ±3.0%) | ±2.2% (RE ±2.5%) | 0.5% | ±7.0% |
| 9.5% | 3 | ±13.2% | ±4.2% (RE ±4.8%) | ±3.5% (RE ±3.9%) | not provable | ±11% |
| 4% | 8 | ±3.4% | ±1.1% | ±0.9% | 2.1% | ±2.9% (a ±5% per-dataset null is provable if \|true\| < 2.1%) |
| 6% | 8 | ±5.1% | ±1.6% | ±1.4% | 1.6% | ±4.3% |
| 6% | 16 | ±3.6% | ±1.1% | ±1.0% | 2.0% | ±3.0% |

**What this means for the design.**

- **Equivalence claims overnight need σ <= 5%.** Hence gate G2. At the 9.5% measured with the VM running, no null can
  be proven at ±3%.
- **n = 3 overnight** resolves:
  - pooled effects of about 2-3% and up;
  - per-dataset effects of about 6-8% and up.

  That covers Nf, SV and the pooled C.
- **n = 8 in the full version** gives per-dataset ±3.4% and per-dataset equivalence at ±5%, if σ ≈ 4%. If E1 gives
  σ ≈ 6%, use n = 16 for the cheap cells (about +21 h).
- **Interactions.**
  - G x Cr, all four cells timed with n = 3: about ±2.7% pooled at σ = 4%, reported as a bound.
  - N x C needs NC timing, so full version only: about ±3% pooled.

### 5.5 Memory RSS validation (E5, S5)

- Runs: `rss_ba.sh` on fb, cells B and C (full version: planet B and C too).
  - 10M ops, cool-down 0, uninstrumented.
  - RSS sampled every 0.25 s; take the last plateau.
- Real index = plateau - 3.20 GB (`w.initial`) - 32 B x ops (trace) - **8 B x warm-up lookups (the warm-up key
  vector, 32 MB at 4M, which rss_ba.sh does not subtract today)**.
- Checks:
  - H4: real(C) - real(B) = 1.00 ± 0.02 B/key;
  - B's real/accounted ratio = 1.020-1.027 (memory_audit).
- About 10 min for fb, 15 more for planet. Quiet host (memory pressure from other processes would distort the
  plateau).

### 5.6 AIDB hardness and input-space compression metrics (S5)

1. **Already done, deterministic** (`hardness_ba.md`, `hardness_200m.json`, `$SP/nfl_ba/combined.json`): raw, free,
   mono, untr and csv scopes on all 10 datasets. They are reused, not re-run.
2. **Complete the CSV scope** on stack, wise and fb_trim: `$SP/nfl_ba/run2.sh csv stack wise`, plus fb with
   `--limit 199999979 --virtual-alpha 0.1`. About 15 min.
3. **Fence-level compression metrics** (new, Tier A, minutes):
   - `fences_ba.py extract` writes `fences/<d>_raw.sosd` (48,829 keys).
   - `fences_ba.py gap --k 1,4,16,64,256` writes `fences/<d>_g<k>.sosd`, keys = round(g(x)·2^62). Strictness is
     asserted, so the conversion is monotone.
   - It also writes `fences/<d>_gaps.json` with D_k, k_eff and the prototype's root model probes and gap probes.
   - Then on every fence file: `build-proto/scaleli_hardness --data <file> --dtype uint64 --pla-eps 1,8
     --check-sorted 1 --threads 16`. This gives RMSE, ME, CD, PLA-1 and PLA-8 on the fence sequence in the raw and
     g coordinates.
4. **Full version.** Hardness patch P5 (`--gap-table file`) computes the AIDB five on all 200M keys in the g
   coordinate, using the table dumped by P3. Check H9 (\|dPLA-32\|, \|dPLA-4096\| <= 32). About 10 x 30 s.

### 5.7 Gap-removal runs, in one place

| tier | run | where |
|---|---|---|
| A | `fences_ba.py` on 200M fences: D_k, root probes, gap probes, fence hardness | S5, minutes |
| A | the 2M arms (`$SP/threeblock`, finished or running): alternation vs single pass, priced greedy | quoted, not re-run |
| B | R1-G (C++ = prototype at 2M) | S2 |
| B | D: G, Gs, G1-G256, GCr (+ GC on fb; + GC, NG, NCG in full) | S3 |
| B | T: G, GCr x3 (x8 full); GC, NG x4 (full) | S4 |
| B | P5 full-key hardness | full only |

### 5.8 Wall-clock

Cost model, extending `run_ba.est`.

- **T run:** 52 s (load and workload 40, warm-up about 5, timed about 6, exit 1) + build + c*.
- **Builds:**
  - B-type 7 s;
  - Cr/GCr up to 25 s;
  - C 296 s on average: `SMOOTH[d]` + 10 + 18 s of root smoothing at alpha 0.1, 7.2e-8·R·(48,829 + R/2) with
    R <= 4,883.

**Overnight version.**

| stage | Tier B, c* = 30 s | Tier B, c* = 60 s | Tier A, c* = 30 s | Tier A, c* = 60 s |
|---|---|---|---|---|
| S0 engineering (human) | 11-14 h | — | 6-7 h | — |
| S1-S2 calibration | 2.0 h | 2.0 h | 1.9 h | 1.9 h |
| S3 D layer | 6.6 h | 6.6 h | 4.9 h | 4.9 h |
| S4 T layer (279 runs Tier B, 219 Tier A; +~5% re-queues) | **9.6 h** | **11.9 h** | 8.0 h | 9.8 h |
| S5 extras | 0.7 h | 0.7 h | 0.7 h | 0.7 h |
| **machine total** | **18.9 h** | **21.2 h** | 15.5 h | 17.3 h |

If c* = 120 s, S4 takes 16.4 h: run blocks 1-2 on night 1 and block 3 on night 2.

**Full version** (adds to the overnight version; c* = 30 s, 60 s in brackets).

| item | time |
|---|---|
| T blocks 4-8 for B, B2, N, Nf, G, SV, Cr, GCr, C on 10 datasets; RAW, RC on all 10 (blocks 1-8) | 19.8 h (24.9 h) |
| T NC, GC (blocks 1-4), NG (blocks 1-4) | 9.4 h (10.4 h) |
| T C4 planet/osm x3 (about 2,060 s of root smoothing per build); BA/CA fb/osm x3; COLD blocks 2-3 | 5.8 h (6.0 h) |
| D-full additions | 11.2 h |
| E5 planet, P5 hardness | 0.8 h |
| **full adds** | **≈ 47 h (≈ 53 h)**; with n = 16 for the cheap cells (σ ≈ 6%): +21 h |
| Linux replication (optional): B, B2, N, Nf, C, Cr, G, SV on fb, osm, planet, books, n = 5 | ≈ 10 h on that machine |

---

## 6. Analysis (pre-registered)

### 6.1 Statistic per run

y = ln(`throughput_ops_s`), the whole 5M replay, from valid T runs only.

Sensitivity statistics, reported next to the primary and never substituted after the fact (except under the
warm-up-gate rule):

- **S1 (warm-up):** ln(sum of ops over chunks 2-16 / sum of t_i over chunks 2-16), where t_i = ops_i / rate_i.
- **S2:** ln of the median chunk rate over chunks 2-16 (PLAN.md's `STAT=med`).
- **S3 (block-adjusted):** fit y = μ_cell + β_block per dataset by least squares and contrast the μ.
- **S4:** include invalid runs.
- **S5:** exclude c1-flagged runs.
- **S6:** Hodges-Lehmann median of the pairwise log differences. If it differs from the mean contrast by more than
  half the CI, say an outlier drives the result.

### 6.2 Per dataset

- For cell X vs base Y on dataset d: Δ_d = mean(y_X) - mean(y_Y).
- B2 is merged into the B baseline once the A/A gate passes. If it fails, no throughput is reported (6.6).
- SE_d = σ̂·sqrt(1/n_X + 1/n_Y). σ̂ is the pooled within-(dataset, cell) SD across all valid T runs (B2 counted as
  B), with df = Σ(n - 1). The 95% CI uses t at that df. The ratio is exp(Δ_d).
- PLAN.md's estimator is kept because adjacent runs were measured to be uncorrelated (design_measurement.md 1), so
  pairing within blocks buys no variance. The block-adjusted S3 covers drift.

### 6.3 Pooled over datasets

- θ̂ = mean of Δ_d over the datasets in S_X, the datasets where X's D-layer signature differs from Y's.
- **Structural nulls** (identical signature) are excluded from the pool and reported as extra A/A comparisons.
- The signature is: accounted bytes, `flow_regions`, `virtual_points`, `root_model`, `root_flow`, `root_vp`,
  `root_virtual`, `root_gaps`, `root_gap_shrink`, root probes/op, gap probes/op, fence probes/op.
- **Primary CI (random effects, equal weights):** θ̂ ± t_{k-1, 0.975}·sd(Δ_d)/sqrt(k). It includes heterogeneity
  between datasets.
- **Secondary (fixed):** θ̂ ± t_df·sqrt(Σ SE_d²)/k.
- Report Cochran's Q and I². Claims use the primary interval. With k < 4, use the fixed interval and say so.
- Estimand: the average effect over the ten AIDB datasets, each counted once. This matches AIDB's dataset-centred
  framing. It is not a law. If I² > 50%, the slide says "average over heterogeneous datasets".

### 6.4 Tests, margins, verdicts

| verdict | rule |
|---|---|
| faster / slower | the 95% CI excludes 1, in the stated direction |
| equivalent within ±δ | TOST at α = 0.05 each side, i.e. the 90% CI lies inside [ln(1-δ), ln(1+δ)]; **δ = 3% pooled; δ = 5% per dataset** (per-dataset equivalence only in the full version) |
| inconclusive | everything else; the words "trend" and "slightly" are banned |

- Predicted nulls (H22, H25, H26) are tested **only** by equivalence.
- Every measured interval is printed beside its counter-predicted range (section 1). A CI wholly outside that range
  is "unexplained" and is not claimed until the mechanism is found (critique.md 2b: an 18.5% speed-up from 1.2% less
  work).
- The thermal and warm-up checks use their own margins (3.5).

### 6.5 Multiple comparisons

- **Primary family F1:** the pooled tests of H22 (N ≡ B), H23 (Nf < B), H24 (C > B), H25 (C ≡ Cr), H26a (G ≡ B),
  H26b (GCr ≡ Cr), H27 (SV > B), and H26c (NC ≡ C, full only).
  - Holm-Bonferroni at a familywise α = 0.05.
  - For TOST, p = max of the two one-sided p-values.
  - Adjusted p-values are reported.
- **The A/A gate (H21)** is a validity condition, not a member of F1.
- **Per-dataset contrasts are descriptive.** Any slide that names a dataset-specific effect uses Bonferroni
  simultaneous 99.5% intervals over the 10 datasets.
- **Interactions** ((NC - C) - (N - B); (GCr - Cr) - (G - B)) are reported as bounds only.
- **Deterministic claims** are equalities and need no correction.

### 6.6 A/A acceptance rule

1. **Calibration E1:** σ̂_AA <= 5%: GO (all claims). 5-8%: GO, pooled claims only, no equivalence claims. > 8%: no
   throughput on this Mac. Also, the E1 B2/B 95% CI must contain 1.
2. **Live, in the main sweep:**
   - the pooled B2/B 95% CI contains 1;
   - **and** TOST at ±3% passes;
   - **and** at most 2 of 10 per-dataset B2/B CIs exclude 1. Under the null, P(>= 3 of 10) = 1.2%.
3. **On failure:** no throughput ratio is shown as a result. The report shows the A/A failure, the hostmon trace and
   the frequency data, and the deterministic results alone.

### 6.7 Warm-up and thermal analysis

- **Warm-up:**
  - per-run c1 and its flag;
  - the pooled mean c1 with SE and TOST at ±3%;
  - mean c1 per cell;
  - the COLD c1 per dataset;
  - the W1 curves.
- **Thermal:**
  - effective GHz per run;
  - C vs B GHz ratio (TOST ±1%), and the same for every cell vs B;
  - mean slope of log chunk throughput over chunks 2-16 per cell, C - B (TOST ±0.002/chunk);
  - `kernel_task` CPU during timed windows by cell;
  - invalid-run rate by cell. If C's rate is more than 2x B's, it is a thermal signal: re-analyse with S4 and say
    so.

### 6.8 Determinism checks (all must hold; any failure stops the analysis)

1. `result_checksum` is identical across all valid runs that share (dataset, seed, ops): every cell within a T
   block, and the D run against T block 1. Read-only lookups with miss 0 depend only on data and trace.
2. `trace_fingerprint` is identical across the same groups.
3. For each (dataset, cell): `memory_before` (all fields) and `learnability` (excluding `smoothing_ns` and
   `transform_ns`) are identical in every T run and in the D run.
4. `work_counters` (excluding `cache_lines` and `cache_lines_per_operation`) are identical between any two
   instrumented runs of the same (dataset, cell, seed).
5. R1 (binary regression) and R1-G (C++ = prototype) passed.
6. Separability on fb (H18):
   - GC's root and gap probes equal GCr's;
   - GC's fence, coordinate and key_at probes equal C's.
7. Tier A vs Tier B: `fences_ba.py` root probes agree with the C++ `root_probes_raw` / `root_probes_gap_raw` at 200M
   to ±0.01. A larger gap means a floating-point difference between the Python replica and C++, to be documented
   before any G probe count is quoted.

### 6.9 What is reported when a gate fails

| failed gate | still reported | not reported | wording |
|---|---|---|---|
| G0 regression | nothing from the new binary | everything | "binary not validated" |
| G1/G2 (σ > 8%) | all deterministic results; throughput moved to Linux | Mac throughput | "host noise σ = x% exceeds the 8% gate" |
| G2 (5-8%) | pooled superiority (Nf, C, SV); per-dataset descriptive | equivalence claims (N, G, Cr, NC nulls) | "not resolved", never "no effect" |
| G3 thermal | all, with C/B labelled "thermal covariate flagged"; T1 cycles/op shown | an unqualified C/B claim | — |
| G4 / warm-up gate in 3.5 | everything on statistic S1 (chunks 2-16) | whole-replay numbers as primary | "warm-up deficit observed; first chunk excluded" |
| G5 determinism | nothing until explained | — | — |
| live A/A | deterministic results | every throughput ratio | "A/A failed: the protocol cannot resolve throughput on this host" |
| invalid rate > 10% of runs | results from complete blocks, with the invalid list | blocks with > 2 missing cells | — |

### 6.10 Sensitivity list (always printed)

S1-S6; the fixed vs random-effects pooled interval; with vs without structural nulls; per-block estimates; the
predicted-range check.

---

## 7. Figures, in meeting order

Each figure states its claim and its data. None is drawn from n = 1 timing.

1. **Method x metric matrix.**
   - Rows NFL, CSV, G, NC, GCr; columns memory, input-space compression, work, throughput, build.
   - Each cell is an arrow, "=", or a number, with the code reason.
   - Claim: *only throughput is statistical; memory and compression move only through G (input axis) and CSV
     (metadata), by construction.*
   - Data: the D layer `d.jsonl` and the section 1 table.
2. **Warm-up, cool-down and noise floor (protocol validity).**
   - (a) Chunk curves, COLD vs warm, for fb, osm and planet (T block 1). (b) W1 deficit by warm-up length.
     (c) T1: GHz and cycles/op by cool-down. (d) E1 A/A distribution, and the night's hostmon other-CPU trace.
   - Claim: *every timed number is warm (pooled c1 = x% ± y), thermally equalised (C/B GHz = 1.00x), and the noise
     floor is σ = z%.*
   - Data: `t.jsonl` COLD + B, `cal_warm.jsonl`, `cal_therm.jsonl`, `cal_aa.jsonl`, `hostmon.jsonl`.
3. **Memory.**
   - Stacked B/key per dataset x cell (keys, values, base metadata, virtual points, root table, gap table), RSS dots
     on fb.
   - Claim: *NFL and G cost 0 B/key; CSV costs +0.80 accounted, 1.00 real (+8%); values are 76-80% of every bar.*
   - Data: `d.jsonl` memory_before; `rss/*.json`.
4. **Input-space compression.**
   - (a) D_k against k per dataset. (b) The root-fence CDF before and after G16 for osm, books and fb.
     (c) Root-fence RMSE/n_f and ME/n_f for B, G16, and Cr where defined. (d) Region rank SSE for B, N, C.
     (e) \|dPLA\| against the 2k bound.
   - Claim: *only gap removal compresses the key axis; it straightens the root CDF where gaps concentrate
     (osm, books, planet) and is invisible to local PLA metrics; NFL bends the axis without compressing it; CSV
     leaves it untouched and smooths the targets.*
   - Data: `fences/*.json`, `fences/*hardness*.json`, `d.jsonl` learnability.
5. **Work per lookup.**
   - Stacked comparisons (key_at, coordinate, fence, gap table, root) for B, N, Nf, G, Cr, GCr, C, NC, with
     transforms/op annotated, a SV line, and the 13.05 floor.
   - Claim: *CSV removes root probes where virtual fences are adopted; G trades about 0-1.5 root probes for 5 table
     comparisons; nobody touches the 13.05 floor.*
   - Data: `d.jsonl` work_counters.
6. **Throughput forest plot.**
   - Per dataset and pooled (random effects), 95% CI. Grey A/A band from B2. Shaded ±3% equivalence zone.
     Predicted-range diamonds. SV and RAW reference rows. n on every row; every run shown as a dot.
   - Claim: filled from the data against H21-H27. Predicted: Nf slower, C a few % faster pooled, N, G, GCr and C-vs-Cr
     equivalent within ±3%, SV faster than everything.
   - Data: `t.jsonl` + `report_t.md`.
7. **Interactions.** The 2x2 panels N x C and G x Cr in comparisons (exact) and in throughput (bounds).
   - Claim: *the methods compose additively; the levels are separable.*
   - Data: `d.jsonl`, `t.jsonl`.
8. **Build time** (log axis).
   - Median and IQR over T runs per cell x dataset; CSV paper band 889-2,902 s.
   - Claim: *CSV's gain costs 28-70x the base build; G and N are free.*
   - Data: `t.jsonl` build_ns.
9. **AIDB hardness**, a separate panel.
   - z-sorted NFL (with the untrained-sawtooth control), CSV slot space, and G (realisable, full version).
   - Claim (from hardness_ba.md): *NFL's AIDB gain exists only in z-sorted order; CSV raises PLA-32 on the 4
     hardest datasets; G moves only the global-line metrics.*
   - Data: `hardness_200m.json`, `$SP/nfl_ba/combined.json`, P5 output.
10. **Backup slides:**
    - the AIDB-protocol replication (BA/CA);
    - the Linux replication;
    - the CSV alpha sweep;
    - the gap k sweep;
    - the alternation-vs-single-pass result;
    - fb outliers;
    - the deviations table (3.6).

---

## 8. Commands

### 8.1 Binary patches (S0), in a copy of the sources

The git root is the home directory, so do not branch. Copy the tree instead, which also leaves `build-fs` and the
MANIFEST of `S` untouched.

| patch | file | change | effort |
|---|---|---|---|
| **P1** cool-down | `src/benchmark.cpp` | `--cooldown-s S` (double, default 0): `std::this_thread::sleep_for` right after `before=ix->memory();` in pass 1, before prefault. JSON `cooldown_s` | 0.5 h |
| **P2** telemetry | `src/benchmark.cpp` | (a) `std::chrono::system_clock` epoch seconds `t_build_end`, `t_warmup_start`, `t_timed_start`, `t_timed_end`. (b) `--cycles 1` (default 1): on macOS `proc_pid_rusage(getpid(), RUSAGE_INFO_V4, ...)` read before and after warm-up and at every chunk boundary, giving `chunk_cycles`, `chunk_instructions`, `chunk_cpu_ns` (`ri_user_time` + `ri_system_time`), `warmup_cycles`, `warmup_cpu_ns`, and derived `effective_ghz`, `oncpu_fraction`, `ipc`, `cycles_per_op`; on Linux `perf_event_open` (cycles, instructions; user-only) and `clock_gettime(CLOCK_THREAD_CPUTIME_ID)`. (c) `--freq-probe N` (default 50,000,000): an LCG dependent chain timed before prefault and after the timed loop, giving `freq_probe_ns_pre/post` (fallback if `ri_cycles` reads 0) | 1.5 h |
| **P3** gap removal | `include/scaleli/index.hpp`, `types.hpp`, `src/benchmark.cpp` | section 4.3 | 4-6 h + 1-2 h validation |
| **P4** perf control (Linux) | `src/benchmark.cpp` | `--perf-ctl FIFO`: write `enable\n` just before `t_timed_start` and `disable\n` just after the timed loop | 0.5 h |
| **P5** hardness gaps (full) | `src/hardness.cpp` | `--gap-table file` (from `--dump-root-gaps`): metrics on g(normalised key); monotone, no re-sort | 1 h |

The rusage reads inside the timed loop are 17 syscalls (about 1-2 µs each) in a 5-7 s window: less than 0.001%.
All cells carry them.

### 8.2 What must be added to `run_ba.py`

1. **Binary.** `BIN = os.environ.get('SCALELI_BIN', '$REPO/build-proto/scaleli_bench')`. Write its sha256 into the
   plan header.
2. **Jobs.** A job becomes `[dataset, cell, block, layer, timeout, overrides]`.
   - `layer` is in {T, D, CAL}.
   - `overrides` is a dict of flag -> value (`warmup`, `prefault`, `chunks`, `cooldown-s`, `ops`, `seed`) applied
     after `common()`.
3. **`common(layer, cool)`.**
   - T: `--warmup 4000000 --prefault 1 --instrument 0 --cooldown-s cool --cycles 1`.
   - D: `--warmup 1000000 --prefault 1 --instrument 1 --cooldown-s 0 --cycles 1` and seed 1001.
4. **New cells in `cells(d)`:** Cr, G, Gs, G1, G4, G64, G256, GCr, GC, NG, NCG, C2p (`--root-alpha 0.04`, for R1),
   BA, CA. COLD becomes `--warmup 0 --prefault 0` through overrides.
5. **Presets.** `regress`, `aa`, `therm`, `warm`, `d_overnight`, `d_tierA`, `d_full`, `t_overnight`, `t_tierA`,
   `t_full`, `aidb`, `sweep` (unchanged), `lin`.
   - `--cool C` is stored in the plan header and used by every T and CAL job.
   - The block structure is as in 5.4.
6. **`run()`.**
   - (a) A quiet gate before each job: read the last 6 lines of `--hostmon` jsonl.
   - (b) After each job, the validity rules of 2.1. They need the P2 timestamps and the hostmon samples inside them.
     Write `valid`, `invalid_reasons`, `oncpu`, `max_other_cpu`, `max_foreign` into `meta`.
   - (c) Re-queue invalid jobs at the end of their block (same seed), at most 2 retries.
   - (d) Pause rule: 3 invalid in the last 10.
   - (e) Stop on battery.
   - (f) Keep the existing resumability.
7. **Wall-clock model.** `est()` and `cost` gain the 5.8 model: c*, Cr builds 25 s, D = 60 + 2·build.
8. **`show`** accepts `--layer` and `--block` (rss_ba.sh depends on it).

### 8.3 What must be added to `analyze_ba.py`

1. **Inputs.** Accept `--det d.jsonl`, `--hostmon`, `--margin 0.03`, `--figdata out.json`, and modes `--regress`,
   `--calib aa|therm|warm`, `--quiet`.
2. **Valid runs only.** Use valid T runs for the primary analysis; list invalid runs with their reasons; add the
   S1-S6 statistics (`STAT=all|tail|med|block|all_runs|noflag|hl`).
3. **Signatures** come from the D layer, with gap fields. Structural-null detection uses them.
4. **Pooled estimates.** Random-effects equal-weight pooling (t at k-1), fixed-effect secondary, Q and I².
5. **Tests.** TOST p-values and verdicts (6.4), and Holm over family F1 (6.5).
6. **A/A rule** of 6.6, implemented as a gate. On failure the throughput section prints only the failure.
7. **Warm-up and thermal checks** of 3.5 and 6.7 (c1 flags, GHz, on-CPU, slopes, kernel_task by cell).
8. **Determinism checks** 1-7 of 6.8, including the D-vs-T block-1 checksum and GC separability.
9. **Tables.** The deterministic table gains `gap_probes`, the comparisons including gap, `root_gaps`,
   `root_gap_shrink`.
10. **`--figdata`** emits every series section 7 needs.
11. **Calibration modes:** `--calib aa` prints σ̂ with its χ² interval and the G2 verdict; `--calib therm` prints c*;
    `--calib warm` prints the transient length and the W decision.

### 8.4 Other scripts

- **`hostmon.py`.**
  - Add `--out`, `n_bench` (pgrep count of `scaleli_*`), and the PID of the top foreign process.
  - Every 60 s, add `therm` (`pmset -g therm` parsed) and `ac` (`pmset -g batt`).
  - Keep 5 s sampling.
  - Start it under `taskpolicy -b`.
- **`rss_ba.sh`.**
  - `show` with `--layer T` and `--cooldown-s 0`.
  - Subtract 8 B x warmup in the plateau formula.
  - Add planet in the full version.
- **`fences_ba.py`** (new, stdlib, imports `$SP/threeblock/threeblock.py`):
  - `extract`: seek to keys 4096·j of each sorted file, giving `fences/<d>_raw.sosd` and a json.
  - `gap --k 1,4,16,64,256`: G via `threeblock.gap_block`, D_k, prototype root and gap probes, and
    `fences/<d>_g<k>.sosd`.
  - `gate2m --bin`: R1-G.
  - `validate --det d.jsonl`: determinism check 7.

### 8.5 Command sequence

```zsh
REPO=/Users/louisvasseur/Downloads/scaleli_sota; S=$REPO/experimental/scaleli; NB=$S/results/aidb_ba
P=$NB/proto; mkdir -p $P; cd $P

# ---- S0 engineering ---------------------------------------------------------------------------
rsync -a --exclude data --exclude results $S/ $REPO/experimental/scaleli_proto/
#   apply P1, P2, P3 (and P4 on Linux, P5 for full) to $REPO/experimental/scaleli_proto per 8.1
cmake -S $REPO/experimental/scaleli_proto -B $REPO/build-proto -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DSCALELI_NATIVE=OFF -DSCALELI_SANITIZE=OFF -DSCALELI_EXTERNAL=OFF
cmake --build $REPO/build-proto -j 8 && ctest --test-dir $REPO/build-proto
shasum -a 256 $REPO/build-proto/scaleli_bench $REPO/build-proto/scaleli_hardness > $P/binaries.sha256
export SCALELI_BIN=$REPO/build-proto/scaleli_bench
pgrep -fl 'scaleli_bench|scaleli_hardness'            # must print nothing before every stage below

# ---- S3 D layer (daytime, any light load; never during S2/S4/S5) --------------------------------
python3 $NB/run_ba.py plan d_overnight > plan_d.json            # d_tierA without P3
nohup python3 $NB/run_ba.py run plan_d.json d.jsonl > run_d.log 2>&1 &
python3 $NB/fences_ba.py extract && python3 $NB/fences_ba.py gap --k 1,4,16,64,256    # Tier A, minutes
for f in fences/*.sosd; do $REPO/build-proto/scaleli_hardness --data $f --dtype uint64 --pla-eps 1,8 \
    --check-sorted 1 --threads 16 > ${f%.sosd}_hardness.json; done      # only after run_d.log says PLAN DONE

# ---- S1 quiet host (evening). Do the 2.1 checklist by hand first ------------------------------
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize hw.pagesize hw.perflevel0.physicalcpu \
  hw.perflevel0.cpusperl2 hw.perflevel0.l2cachesize hw.perflevel1.physicalcpu; pmset -g; pmset -g batt; \
  pmset -g therm; /usr/bin/c++ --version; cat binaries.sha256; } > host.txt
taskpolicy -b python3 $NB/hostmon.py --out $P/hostmon.jsonl & echo $! > hostmon.pid
caffeinate -dims -w $(cat hostmon.pid) &
sleep 600; python3 $NB/analyze_ba.py --quiet hostmon.jsonl --last 600          # G1

# ---- S2 calibration (attended) ----------------------------------------------------------------
python3 $NB/run_ba.py plan regress > plan_r1.json && python3 $NB/run_ba.py run plan_r1.json r1.jsonl --hostmon hostmon.jsonl
python3 $NB/analyze_ba.py --regress r1.jsonl $NB/pilot_seed11.jsonl && python3 $NB/fences_ba.py gate2m --bin $SCALELI_BIN   # G0
python3 $NB/run_ba.py plan aa --cool 30 > plan_aa.json && python3 $NB/run_ba.py run plan_aa.json cal_aa.jsonl --hostmon hostmon.jsonl
python3 $NB/analyze_ba.py --calib aa cal_aa.jsonl                               # G2: stop here if sigma > 8%
python3 $NB/run_ba.py plan therm > plan_th.json && python3 $NB/run_ba.py run plan_th.json cal_therm.jsonl --hostmon hostmon.jsonl
COOL=$(python3 $NB/analyze_ba.py --calib therm cal_therm.jsonl --print-cool)    # G3
python3 $NB/run_ba.py plan warm --cool $COOL > plan_w.json && python3 $NB/run_ba.py run plan_w.json cal_warm.jsonl --hostmon hostmon.jsonl
python3 $NB/analyze_ba.py --calib warm cal_warm.jsonl                           # G4 (prints W; default 4000000)

# ---- S4 T layer (overnight, unattended) -------------------------------------------------------
python3 $NB/run_ba.py cost t_overnight --cool $COOL                             # check it fits the night
python3 $NB/run_ba.py plan t_overnight --cool $COOL > plan_t.json               # t_tierA without P3
nohup caffeinate -dims python3 $NB/run_ba.py run plan_t.json t.jsonl --hostmon hostmon.jsonl > run_t.log 2>&1 &
#   resumable: re-run the same line after any interruption; invalid runs are re-queued automatically

# ---- S5 morning: RSS (quiet), hardness completion ---------------------------------------------
zsh $NB/rss_ba.sh
zsh $SP/nfl_ba/run2.sh csv stack wise        # plus fb with --limit 199999979 --virtual-alpha 0.1 (same script pattern)

# ---- S6 analysis ------------------------------------------------------------------------------
python3 $NB/analyze_ba.py --det d.jsonl --hostmon hostmon.jsonl --margin 0.03 --figdata figdata.json t.jsonl > report_t.md
for s in tail med block all_runs noflag hl; do STAT=$s python3 $NB/analyze_ba.py --det d.jsonl --hostmon hostmon.jsonl t.jsonl > report_t_$s.md; done
kill $(cat hostmon.pid)

# ---- restore the host -------------------------------------------------------------------------
sudo mdutil -a -i on; sudo tmutil enable; sudo pmset -a powernap 1    # only the ones that were on before
#   restart the UTM VM and the apps; resume iCloud sync and updates

# ---- full version (only if G2 gave sigma <= 6%) -----------------------------------------------
python3 $NB/run_ba.py plan d_full > plan_dfull.json && nohup python3 $NB/run_ba.py run plan_dfull.json d_full.jsonl > run_dfull.log 2>&1 &
python3 $NB/run_ba.py plan t_full --cool $COOL > plan_tfull.json    # contains the overnight blocks; run() skips finished jobs if pointed at t.jsonl
nohup caffeinate -dims python3 $NB/run_ba.py run plan_tfull.json t.jsonl --hostmon hostmon.jsonl > run_tfull.log 2>&1 &
```

**Fallback if no new binary can be built before the meeting.**

- Run PLAN.md's overnight plan unchanged on `build-fs`, with these differences:
  - add `Cr` (no patch needed);
  - set `--prefault 1`.
- Drop the thermal-equalisation claim. Report only PLAN's C-vs-B chunk-slope diagnostic.
- Report gap removal from Tier A (`fences_ba.py`) as probe counts only.

---

## 9. Threats to validity and what the protocol does about each

| # | threat | what the protocol does |
|---|---|---|
| 1 | Host contention (the UTM VM, nightly macOS jobs, Spotlight, Time Machine, iCloud) | quiet checklist and Q0; hostmon quiet gate before every run; outcome-blind invalidation and re-queue; live A/A with a pre-registered rule; G2 sends throughput to Linux if σ > 8% |
| 2 | Build-heat confound (CSV builds run 160-450 s at 100% of all cores; B's take 7 s) | identical cool-down c* for every cell, calibrated by T1 on the real heat source; per-run effective GHz and on-CPU fraction; C-vs-B GHz TOST ±1% and slope TOST; the instrumented rebuild never runs in T |
| 3 | Cold start or incomplete warm-up | prefault + 4M workload-mode warm-up (coverage bound 0.75M, transient <= 250k); per-run c1 flag; pooled TOST ±3%; COLD as a sensitivity check (-17 to -21%); pre-registered fallback to statistic S1 |
| 4 | Per-process noise, heavy tails, two-state host mixture | many independent processes; random-effects pooling; Hodges-Lehmann robustness (S6); every run plotted; a measured effect far outside its counter prediction is "unexplained", not claimed |
| 5 | Thread migration between the two 6-core P-clusters (separate 16 MiB L2s), which macOS cannot pin | randomisation turns it into noise, not bias; per-chunk cycles/op exposes mid-run steps; Linux pinned replication |
| 6 | Memory layout, ASLR, allocator effects (a fixed layout can bias one binary or cell) | separate processes with ASLR on; one binary for every cell; the raw-arena allocator cliff is documented (memory_audit) and affects only RAW/RC |
| 7 | Binary differences between cells | one binary (`build-proto`) for all T cells; R1 shows it equals `build-fs` at defaults |
| 8 | Drift over the night; order effects | randomised cell order in complete blocks; cells of a dataset contiguous; block-boundary stopping; block-adjusted S3 |
| 9 | Analytic flexibility | this document is the pre-registration: statistic, margins, pooling, Holm, gates and fallbacks fixed in advance; n never extended on results |
| 10 | Structural nulls pooled with real effects | D-layer signatures; identical-signature datasets leave the pool and become extra A/A |
| 11 | Instrumentation perturbing timing | counters only in the D layer; T uninstrumented; only 17 rusage reads inside the timed window |
| 12 | CSV greedy is chaotic (±0.36 root probes under 1e-6 perturbations) | bit-identical rebuilds are verified (6.8); sub-probe differences across platforms or compilers are not interpreted; no claim rests on < 0.5 probes |
| 13 | Construct validity: "NFL" and "CSV" are not the published systems | labels fixed: "NFL's feature in a key-ordered index" (no z-ordered storage, batch 1), "CSV Algorithm 1 as target relabelling" (no Algorithm 2, no physical gaps), "gap removal at the root"; z-sorted AIDB metrics on a separate panel |
| 14 | External validity: one Mac, 16 KiB pages, single-thread uniform reads, no scans or inserts, batch 1 | stated on every slide; Linux replication on 4 KiB pages; AIDB-protocol replication (BA/CA); scans and published baselines (ALEX, PGM) declared out of scope |
| 15 | Flow training seed not replicated (one weight file per dataset) | stated; Nm (monotone flow) in the full D layer; the untrained-sawtooth control on the hardness panel |
| 16 | Gap-removal design choices (k = 16, shrink to median, 1 probe per table comparison in the selector) | k fixed in advance, with a deterministic k sweep {1, 4, 16, 64, 256}; forced G reported whatever the selector does; Gs reported separately |
| 17 | Counters are not time (probes and lines mis-ranked SV before) | counters are mechanism only; time is measured directly; G's table comparisons are counted separately (`gap_probes`) because an L1 comparison is not a root probe |
| 18 | Accounted memory vs real memory | E5 RSS on fb (and planet in full); hidden `virtual_features` capacity predicted and tested; the warm-up key vector is subtracted |
| 19 | Prototype vs implementation mismatch for G | R1-G (C++ = Python to 1e-9 at 2M) and check 7 at 200M |
| 20 | Heterogeneity across datasets | random-effects intervals, I²; per-dataset forest plot; "average over datasets" wording when I² > 50% |
| 21 | Non-random failures (CSV timeouts on the slowest datasets) | timeouts are result rows, re-queued once, never silently dropped; per-cell invalid and missing counts reported |
| 22 | Results depending on the evening's c* or W choice | both are chosen by pre-registered rules from calibration, recorded in the plan header, and fixed for the whole sweep |

---

## Appendix A. Differences from PLAN.md, at a glance

| item | PLAN.md | this protocol | reason |
|---|---|---|---|
| compression | codec bytes (null by construction) + Elias-Fano panel | input-space compression (D_k, fence linearity, region SSE); key bytes stay in memory as a null | the supervisor's definition |
| methods | NFL, CSV | NFL, CSV, gap removal G (+ G x CSV, G x NFL) | requested |
| prefault | 0 | 1 | removes the page-level cold part regardless of W; `prefault_ns` as a paging sentinel |
| thermal | 4M warm-up as partial equalisation + slope check | cool-down c* for all cells + measured GHz + slope check | the build-heat confound needs a direct fix |
| instrumented pass | block 1 of the timing sweep | separate D layer | saves about 1.5-3.5 h of quiet time (the second builds, and NC's n = 1 timing); T never runs a second build |
| NC | timed n = 1 | D only overnight; timed n = 4 in full | n = 1 timing is uninformative |
| root-only CSV | in the sweep only | Cr timed | isolates the region virtual points (H25) and gives a cheap G x CSV 2x2 |
| pooling | fixed effect | random effects primary, fixed secondary | heterogeneity is expected (CSV adopted on fb, falls back on osm) |
| nulls | equivalence "bound only" | TOST ±3% pooled, ±5% per dataset, Holm | "not significant" does not prove "no effect" |
| run validity | other-CPU logged | outcome-blind invalidation + re-queue | contention handled before results are seen |
| binary | build-fs | build-proto (P1-P3), regression-tested | needs cool-down, telemetry, gap removal |
