# Reviewer 2 (systems benchmarking practice): review of PROTOCOL_draft.md

Scope: warm-up, cool-down and the build-heat confound first. Then host isolation, macOS limits, layout, allocator,
Linux, PMU counters for "why binary search wins", AIDB comparability, and the runner. Nothing was run for this
review except reading files, `sysctl`, `pmset -g`, the SDK header `sys/resource.h`, the PMU event database in
`/usr/share/kpep/`, and one stdlib re-analysis of 12 existing JSONs in `$SP/verify/A2`.

## Verdict

The structure is sound: separate deterministic and timing layers, outcome-blind invalidation, a live A/A, positive
and negative controls, and pre-registered gates. But the two things the supervisor asked for are not yet fixed:

1. **Warm-up.** 4M lookups is a defensible number. The argument for it is wrong, and part of the evidence is the
   contended n = 3 set, which the idle replication did not reproduce. The COLD sensitivity gate is set from that
   set, so it will fail about half the time on a quiet host and wrongly withdraw the warm-up claim.
2. **Build heat.** This is not neutralised. The calibration (T1) checks a proxy, core GHz, that is probably flat on
   this machine whatever the heat. It runs 2 runs per arm, and can pick cool-down values it never tested. The quiet
   gate is applied about 50 s before B's timed window but about 350 s before C's.

Both problems, and most of the per-process noise, go away by design if the T layer is moved to an **in-process
interleaved harness**. The critique asked for it, and PLAN.md D6 deferred it only to save engineering time. The
draft already spends 11-14 h building a new binary, so that reason is gone.

There is also a **host-state finding to act on before any run.** At the time of this review the Mac reported
Battery Power at 6%, discharging, with "Battery Warning: Early", and `powermode 1` under BOTH the AC and Battery
profiles. The draft's own check, `pmset -g | grep lowpowermode`, prints nothing on this host (rc = 1), so it would
pass without checking anything.

---

## 1. Warm-up (section 3)

### 1.1 The evidence base is the contended set; the idle replication disagrees (major)

Sections 3.1 item 1 and 3.2 rest on the warm-up agent's cold experiment: n = 3, with a peer 200M job alive in 237
of 241 samples. On that set the first chunk was -34% without prefault and -2 to -5% with it. The verifier re-ran it
with no peer scaleli process (`$SP/verify/A2`, fb, `--warmup 0`, K = 16 at 2M ops, so 125k-op chunks, n = 6 per
arm). The results, in the verifier's output and recomputed here from the 12 JSONs:

| | prefault 0 | prefault 1 |
|---|---|---|
| c1 at 125k (median) | -19.4% (range +3.0 to -32.0) | -5.8%, with 3 of 6 runs at -15 to -17% |
| chunk 2 at 125k | -11 to -21% in all 6 runs | -10 to +19% |
| first-1/16 deficit at the T layer's 312.5k-op chunk (chunks 1, 2 and half of 3 against the median) | **median -14.4%** (-6.2 to -20.4) | median -3.7% (-8.0 to +5.2) |
| minor faults (`time -l`) | 1,104,188 | 1,104,194 |

Consequences:

- **"About 30 points of the cold penalty are page-level" is not supported.** Prefault moves 6 minor faults out of
  1.1M and no major faults. Its effect on c1 is not significant (Mann-Whitney U = 8 against a critical value of 5).
  It also reached throughput 5-7% slower in two sets, again not significant. Whatever prefault does, it is not
  paging. Delete the "page-level" framing and the -34% / "30 points" numbers from 3.1-3.2 and section 0.
- **The COLD prediction (-17 to -21%) and gate (COLD must show at least -15%) are miscalibrated.** On the idle
  data the expected 312.5k-chunk deficit is about -14%, and 3 of 6 idle runs would fail the -15% gate. G4 and the
  3.5 rule would then say "the indicator is insensitive; the warm-up claim is not made". That is the wrong
  conclusion for a sound warm-up.
- **The transient is longer than the 125k-250k quoted.** Chunk 2 (125k-250k) is still 11-21% slow in every
  prefault-0 run, so it runs to at least 250k and possibly 375k. 4M is still 10-16x that, so the choice survives,
  but quote the margin honestly.

**Fixes.**

- (a) Rebase 3.1-3.2 on `$SP/verify/A2` and keep the contended set only as a footnote.
- (b) Record **80 chunks** (62.5k ops each) in every T run. Chunk timestamps cost nothing, and the P2 rusage
  reads cost about 2 µs each. Re-aggregate to 16 in analysis for continuity, but run the warm-up test at fine
  resolution.
- (c) Set the COLD prediction to "-6 to -20% at 312.5k; -13 to -32% at 62.5-125k", and set the sensitivity gate
  on the **pooled** COLD mean (at least -8% at 312.5k), not on per-run values.
- (d) Keep prefault = 1 only as harmless insurance that matches the binary's default, with no claimed effect.
  For paging, use direct counters (section 4.4), not `prefault_ns`.

### 1.2 The coupon-collector argument does not describe cache state (minor; the number survives)

3.1 item 2 sets W >= R ln(100R) = 0.75M so that every one of the 48,829 Region headers (13.7 MB) is touched.
Touching them does not keep them cached. Between two lookups of the same region, a uniform stream brings in about
R x (21 to 50 lines/op) = 1.0-2.4M new lines, which is 65-156 MB of traffic. That is far above the 16 MiB P-cluster
L2 (262k lines) and above the system-level cache too (not exposed by `sysctl`; about 48 MB is the figure usually
reported for M-Max parts, unverified here). So in steady state the Region headers are mostly misses, and a warm-up
cannot change that. The coupon curve matching the 125k-250k transient is a coincidence.

What actually warms, and is reused often enough to stay resident:

- the root fences and model (about 390 KB; about 6.1k lines, each reused every few thousand lookups);
- `regions_` (390 KB);
- the last-level page tables for the index (2.1-2.8 GB / 16 KiB x 8 B = 1.0-1.4 MB);
- the code and the predictor tables.

All of these settle within about 1e4-1e5 lookups, and so does LRU turnover. **Fix:** replace the 0.75M
requirement with "empirical transient <= 375k (A2); W = 4M is >= 10x; the hot set is root + `regions_` + page
tables, about 3 MB". Then state that the deeper structures are DRAM-resident in steady state by design.

### 1.3 The c1 statistic is biased and the W1 gate has no power (minor)

- **c1 compares one chunk with a median of 15, which is biased.** c1 = chunk1 / median(chunks 2..16) - 1. Chunk
  rates are skewed (occasional slow chunks), so E[c1] is not 0 even in steady state. **Fix:** a placebo-chunk
  test. Compute c_j = chunk_j / median(the other chunks) - 1 for every j, and test c_1 - mean(c_2..c_16) pooled.
  This is unbiased by construction.
- **W1 cannot decide anything at n = 2 per arm.** With 125k-chunk SD of 11-14% per run, the W1M arm's mean c1
  has an SE of about 9%. "Pass if > -5%" and "W0p0 < -15%" are coin flips. **Fix:**
  - Drop W1 as a gate.
  - Run the cold arms (W0p0, W0) at n >= 6 with 80 chunks as a curve figure.
  - Let the in-sweep pooled placebo test (about 290 runs) decide.
  - W may only go up (4M to 8M), never down.

### 1.4 Branch predictors, frequency and TLB: fine, with one addition

The workload-mode warm-up with a distinct seed is right. DVFS settles in milliseconds. TLB state is part of the hot
set above and settles in about 1e4 lookups. One gap: the FlowTransform weights are not in `for_each_allocation`
(verifier finding). They are tiny, and 4M lookups warm them, so this is harmless for T, but it means prefault
does not cover N and Nf.

---

## 2. The build-heat confound is not neutralised (blocker)

### 2.1 What T1 actually tests

- **GHz is probably the wrong proxy.** On the 16-inch chassis (`hw.model` = Mac15,9, the 16-inch M3 Max), a
  single P-core very likely runs at its maximum clock regardless of chassis temperature below throttling. So
  "GHz within 1% of B" will pass at c = 0 and T1 returns c* = max(10, 0) = 10 s, a value it never tested.
- **The workload is memory-latency-bound** (1.2-1.7 µs per lookup, several dependent DRAM misses). Its speed
  depends on fabric and DRAM state, not the core clock. Two heat-sensitive mechanisms are invisible to core GHz:
  - fabric or memory DVFS state;
  - the in-package LPDDR refresh rate, which rises with temperature.

  Both are plausible; their size here is unknown.
- **cycles/op is confounded by the clock.** For memory-bound code, cycles/op = (ns/op) x GHz. A hotter, lower
  clock lowers cycles/op for the same memory time. The "cycles/op of C within 3% of C at 120 s" criterion partly
  measures the thing it is meant to control.
- **Internal inconsistency:**
  - 3.4 says c* comes from {0, 30, 120}, with 60 if T1 is not run, and max(10, ·).
  - Section 0 and the abstract say 30 or 60.
  - The schedule is costed at 30 and 60.
  - 60 and 10 are never tested.
- **Power: n = 2 per arm.**

### 2.2 Two asymmetries the cool-down does not remove

- **The quiet gate is checked before the process starts.** B's timed window opens about 50 s after the gate. C's
  opens about 350 s after (40 s load + 296 s build + c*). Valid C runs are therefore "quiet at t - 350 s" and
  valid B runs "quiet at t - 50 s". Invalidation will hit C more often, and C's survivors are selected differently.
- **Heat carries over between runs.** A cell that follows a C run starts its own window only 40 + 7 + c* s after a
  300 s all-core load. Shuffling turns this into noise for the other cells, but C is always timed right after its
  own build.

### 2.3 Fixes, in order of preference

1. **Primary: remove the confound by design with an in-process interleaved T layer (new patch P6; section 3).**
   - Every cell of a dataset is built in one process, in random order.
   - Then one common cool-down.
   - Then all cells are timed in interleaved rounds.
   - Every cell's timed window has the same thermal history, including C's build heat.
2. **If the per-process design is kept**, add all of the following. Each is cheap.
   - **(a) A heat-matched control cell, BH.** BH = B + `--preheat-s 300`: 16 threads running a mixed integer and
     streaming loop for 300 s, right after the build, before the cool-down; about 0.5 h of code. BH vs B measures
     the confound **on throughput itself**, the outcome that matters, with the structure held fixed. Put it in T1
     (B and BH x {c = 30, 60, 120} x n >= 4) and in every T block on fb. Gate: BH/B equivalent within ±2% at c*.
     This replaces the GHz and cycles/op criteria.
   - **(b) A memory-latency probe next to the LCG frequency probe** (in P2).
     - What: a Sattolo random cycle over 512 MB-1 GB at 64 B granularity, 2-3M dependent loads (about 0.3 s).
     - When: before prefault and after the timed loop.
     - Allocation: allocated before the build, so its layout is common to all cells.
     - Output: ns/load. It captures fabric, DRAM and TLB-walk state, which core GHz cannot.
     - Equivalence: C vs B within ±1%.
     - Analysis: use it, pre-registered, as a covariate in sensitivity S7. It is a heat mediator, and adjusting
       for it is the intended correction.
   - **(c) A state-gated cool-down instead of a fixed one.**
     - Rule: wait until the SoC temperature is within 1-2 °C of the value read just before the build (cap 300 s),
       and log the wait as a per-run covariate.
     - Reading the temperature without sudo: the IOReport / IOHID sensor path that tools such as `macmon` use.
     - With sudo: `powermetrics -s thermal,cpu_power -i 5000` in hostmon (it also gives per-cluster frequency
       residency).
   - **(d) Move the quiet gate inside the process.** P1's sleep becomes "cool-down, then block on a FIFO `go`
     token". The runner writes `go` only when hostmon's last 30 s pass the quiet rule. Every cell is then gated in
     the same position relative to its window.
   - **(e) Record the predecessor's build CPU-seconds** in `meta` and add it as a covariate in a sensitivity
     analysis.
3. **Fix 3.4 so the arms tested equal the arms selectable:** {30, 60, 120}, or a state-gated rule. No max(10, ·).

---

## 3. Recommended primary T layer: in-process interleaving (P6) (major)

Why it is now the right call:

- **(i)** It removes the build-heat confound by construction (section 2).
- **(ii)** It cancels host state that persists over seconds. That is the two-state mixture the critique documented:
  1.4x steps inside one process, and osm runs at chunk CV 0.02-0.03 sitting at 0.87-0.89 Mops while all others
  sit at 0.67-0.78.
- **(iii)** It pays the 40 s load once per dataset instead of once per cell.
- **(iv)** design_measurement.md section 8 already specifies it, and the draft is building a new binary anyway.

**Design.** One process per (dataset, replicate):

1. Load data and generate the trace.
2. Build every T cell (B, B2, N, Nf, G, Cr, GCr, C, SV, and on fb RAW and RC) in a random order seeded by the
   replicate, each build timed.
3. Run the common cool-down (or the state gate) and the quiet-gate FIFO.
4. Run R = 8 rounds. In each round, visit every cell in a fresh random order. A visit is 1M untimed workload-mode
   warm-up lookups of that cell (at least 2.7x the longest transient seen; this re-warms the hot set lost when
   switching cells), then 1M timed lookups. Every cell in a round uses the same trace segment, giving pairing
   within the round, with 16 sub-chunks and per-visit rusage V6, cycles, P-core fraction and runnable time.
5. Estimand: the mean over rounds of ln(thr_X) - ln(thr_B), within the process. Then pool processes (n >= 3 per
   dataset), then datasets (random effects, as in 6.3).

**Cost.** About 40 s load + about 380 s of builds (C 296 s; 8 cheap builds about 80 s) + 60 s cool-down +
9 cells x 8 rounds x 2M lookups x about 1.2 µs (about 175 s). That is about 11 min per dataset-process, or about
1.9 h per replicate over 10 datasets. Three replicates take about 5.6 h, against 9.6-11.9 h for the draft's T layer,
with 24M timed lookups per cell per dataset instead of 15M.

**Memory.** About 8 x 2.2 GB indexes + SV 3.2 + `w.initial` 3.2 + the 6.4 GB bulk_load transient, roughly
31-33 GB. RAW adds 4.1 GB on fb. This fits in 64 GB only with the VM stopped. Check the C build's own peak in the
D layer first, and require zero `vm_stat` compressions during timing (4.4).

**Validation (replaces part of E1).**

- Run an in-process A/A: B and B2, built separately in one process, interleaved, over 5 processes.
- Adopt P6 as primary if the within-process sd of the pair difference is clearly below sqrt(2) x the
  separate-process sigma_AA.
- Keep a reduced separate-process T layer (n = 2 blocks, B, C, N, SV) as the external-validity and
  AIDB-shaped check.
- Never mix the two in one estimate.

**Effort.** About 3-4 h. Per-cell `Config` closures; `--cells cells.txt` with one flag string per line;
round-robin; JSON per visit.

---

## 4. Host isolation and macOS limits

### 4.1 Power state right now (blocker; cheap to fix)

`pmset -g batt` gave "Battery Power, 6%, discharging, 0:13 remaining, Battery Warning: Early". `pmset -g custom`
gave `powermode 1` for both "AC Power" and "Battery Power". On MacBook Pros with High Power Mode the key means
0 = Automatic, 1 = Low Power, 2 = High Power. Confirm in System Settings > Battery > Energy Mode. If it is Low
Power, every past 200M timing ran with capped clocks. hostmon never logged power state, so that cannot be ruled
out.

**Fixes.**

- The checklist must read `pmset -g custom` and require `powermode` = 2 (High Power: fans ramp earlier, which also
  reduces heat carry-over) or 0 on the AC profile, set with `sudo pmset -c powermode 2`.
- The `lowpowermode` grep must go, because it fails silently on this machine.
- Record `powermode` in `host.txt` and in every hostmon sample.
- R1's frequency check must see a P-core clock near the part's maximum (about 4 GHz), not merely inside
  [2.5, 4.5] GHz, a range that would accept a capped clock.
- Charge to at least 80% before the evening.

### 4.2 P-core vs E-core residency, descheduling and units (major)

`--qos 1` only biases the scheduler toward P-cores. A thread can still run on E-cores (2 cores' worth of slower
clock and a 4 MiB L2) when the P-clusters are busy. That is the most likely single cause of a 1.4x step inside one
run. The SDK on this host has `RUSAGE_INFO_V6` (`sys/resource.h` line 192) with `ri_runnable_time`,
`ri_user_ptime`, `ri_pcycles` and `ri_pinstructions`.

**Fixes to P2.**

- Use V6, not V4.
- Per chunk, record:
  - P-core fraction = `ri_pcycles / ri_cycles`;
  - `ri_runnable_time` (time ready to run but not running: a direct descheduling measure, better than on-CPU
    fraction).
- Invalidation (outcome-blind): P-core fraction < 0.99, or runnable time > 1% of the window.
- The time fields are in mach-absolute ticks on Apple silicon (125/3 ns per tick). Convert them with
  `mach_timebase_info`, and validate in R1 against wall time over a busy loop. Otherwise on-CPU fraction and GHz
  come out off by about 41.7x.
- Check in R1 that `ri_cycles` deltas over a chunk are non-zero and agree with the LCG probe.

**Cluster migration.**

- Moving between the two 6-core P-clusters, each with its own 16 MiB L2, cannot be prevented: thread affinity tags
  are ignored on Apple silicon.
- The per-chunk curve shows the cost of a migration (a re-warm dip of about one 62.5k chunk).
- With 80 chunks, the analysis can report the number of dips per run as an outcome-blind-ish diagnostic.
- Under P6 a migration hits all cells in a round alike.

### 4.3 hostmon measures the wrong thing at the wrong time (major)

`ps -Ao pcpu` on macOS is a decaying average over up to a minute. A 2 s burst at 400% shows up as about 30%,
spread over the next minute, so both the quiet gate and invalidation rule (a) lag and smooth.

**Fixes.**

- Sample cumulative CPU time (`ps -Ao pid=,time=,comm=`) and difference it per PID across samples. That gives
  exact CPU-seconds per 5 s interval, still stdlib-only. Alternatively, use the second sample of
  `top -l 2 -s 5 -stats pid,cpu,command`.
- Add `vm_stat` deltas (pageins, pageouts, compressions, decompressions, swapins) and `powermode` to every sample.
- Add the thermal pressure level: the `com.apple.system.thermalpressurelevel` notify key via a 10-line C helper,
  or `powermetrics -s thermal` with sudo.
- Drop `pmset -g therm`; it reports "No thermal warning level has been recorded" on Apple silicon and carries no
  information.
- Drop `kernel_task` as a thermal signal. That was Intel-Mac idle injection; Apple silicon throttles through DVFS.

### 4.4 Paging and allocator state: measure, do not infer (minor)

- **Paging.** Replace the `prefault_ns > 1 s` sentinel with direct per-phase counters read at four points: build
  end, after the cool-down, after the warm-up, after the timed window.
  - Counters: `getrusage` `ru_minflt` and `ru_majflt`, `task_info(TASK_EVENTS_INFO)` faults and pageins, and
    `host_statistics64` compressions and decompressions.
  - Rule: any decompression, swapin or pagein in the timed window invalidates the run.
- **Allocator.** C's build allocates and frees GBs of smoothing temporaries, so its heap is more fragmented than
  B's after a 7 s build. Without memory pressure that should not touch timing, but record
  `TASK_VM_INFO.phys_footprint` before the warm-up so a difference is visible.
- **Writes in read-only lookups.** Every `find` writes `Region::write_heat` (index.hpp:111, called from
  index.hpp:476). Read-only lookups therefore dirty a Region header line per lookup and cause write-backs. SV pays
  nothing comparable. This matters for 6 below, and it means "prefault is read-only" says nothing about steady
  state.

### 4.5 Display, nightly jobs and the rest of the checklist (minor)

- **Display.** `caffeinate -dims` keeps the display on, which keeps WindowServer compositing. Use `-ims` and let the
  display sleep, which does not stop the process.
- **Background daemons.** XProtect, mds, photoanalysisd and mediaanalysisd run at background QoS on E-cores, but
  still use SLC and DRAM bandwidth. hostmon's foreign-CPU figure should therefore be logged separately for
  background-QoS processes.
- **The rest of the checklist (2.1)** is good.

---

## 5. Memory layout, ASLR and processes (minor)

- **ASLR does not randomise what matters most.** Large allocations (>= 16 KiB) are page-aligned, so the low 14 bits
  of every value column, arena and descriptor array are the same in every run. L1 set aliasing (L1D is 128 KiB per
  P-core, `hw.perflevel0.l1dcachesize`) is therefore a fixed function of the cell's allocation sequence, not
  noise. L2 and SLC indexing is physical and does vary per run. Region arrays are allocated inside 16 build
  threads (index.hpp:459-464), so heap placement also depends on thread scheduling. Most of this is legitimately
  part of a method's effect: C allocates differently because it stores more.
- **argv length is a pure artefact.** N and Nf pass a long flow path that the others do not, which shifts the
  initial stack. This is the classic Mytkowicz et al. (ASPLOS 2009) artefact. **Fix:** the runner pads the
  environment (`SCALELI_PAD=xxx...`) so that len(argv) + len(env) is constant across cells. This costs nothing.
- **Layout sensitivity check (optional, 20 min in E1).**
  - B x 10 with `--layout-pad-seed s`: a random 0-64 KiB dummy allocation before load, plus a random small
    allocation between regions.
  - If sigma rises, layout is a component; then randomise the pad per run in T.

---

## 6. PMU counters that settle "why plain binary search beats every learned cell" (major)

The draft's 2.3 says hardware counters are "Linux only, no PMU access without sudo". That is not right. With sudo,
macOS exposes the core PMU through the private kperf/kpc interface, or through Instruments
(`xctrace record --template "CPU Counters"`).

This CPU (`hw.cpufamily` 0x72015832, database `/usr/share/kpep/cpu_100000c_2_72015832.plist`, name "as3") has
8 configurable counters plus 2 fixed. Events confirmed in that file include:

- `L1D_CACHE_MISS_LD_NONSPEC`;
- `L1D_TLB_MISS_NONSPEC`;
- `L2_TLB_MISS_DATA`;
- `MMU_TABLE_WALK_DATA`;
- `BRANCH_MISPRED_NONSPEC`, `BRANCH_COND_MISPRED_NONSPEC`, `BRANCH_INDIR_MISPRED_NONSPEC`;
- `MAP_STALL_DISPATCH`, `MAP_DISPATCH_BUBBLE`;
- `L1D_CACHE_WRITEBACK`;
- `INST_INT_LD`, `LD_UNIT_UOP`.

The `*_NONSPEC` events carry `counters_mask` 0xe0, so at most 3 fit in one pass. There is no L2 or SLC miss event,
so DRAM misses must be inferred.

**Mac passes** (each event normalised per lookup over the timed window only, through a P4-style enable/disable
around the loop):

- **Pass A:** `L1D_CACHE_MISS_LD_NONSPEC`, `L1D_TLB_MISS_NONSPEC`, `BRANCH_MISPRED_NONSPEC` (counters 5-7), plus
  `L2_TLB_MISS_DATA`, `MMU_TABLE_WALK_DATA`, `MAP_STALL_DISPATCH`, `L1D_CACHE_WRITEBACK`, `LD_UNIT_UOP`.
- **Pass B:** `BRANCH_COND_MISPRED_NONSPEC`, `BRANCH_INDIR_MISPRED_NONSPEC`, `INST_INT_LD`, plus
  `MAP_DISPATCH_BUBBLE` and `L1D_CACHE_MISS_LD`. The speculative minus the retired count gives wrong-path loads,
  a proxy for speculative prefetching by the binary search.
- Fixed counters: cycles and instructions.

**Linux x86 (Skylake-SP, like AIDB)**, added to the draft's generic list:

- `mem_load_retired.l1_miss`, `l2_miss` and `l3_miss`;
- `dtlb_load_misses.walk_completed` and `walk_active`;
- `cycle_activity.stalls_l3_miss`;
- **`l1d_pend_miss.pending` / `l1d_pend_miss.pending_cycles`**: memory-level parallelism, which is the single
  number most likely to explain the result;
- `br_misp_retired.all_branches`;
- `topdown` slots, where supported.

**Cells:** SV, RAW, RC, B, C on fb and osm, n = 3, in the D layer (counters perturb nothing if they are enabled
only for the window).

**Hypotheses to separate, with the manipulation that does it:**

| hypothesis | counter signature | manipulation (cheap) |
|---|---|---|
| H-mlp: SV's misses overlap (speculative binary search, independent lookups) while the index's are a serial pointer chain (root -> regions_ -> Region -> descriptor -> arena -> value) | SV: more L1D misses per op but fewer cycles per miss; higher l1d_pend_miss MLP | a `--dependent 1` replay mode, where each key depends on the previous result through a zero mask; if SV's lead shrinks or vanishes, it is MLP |
| H-tlb: the index touches 4-6 unrelated pages per lookup; SV's last ~10 levels share a page | MMU walks per op, L2 TLB misses per op | Linux THP never vs always for SV and B; macOS cannot change its page size |
| H-decode: codec dispatch and decoding (the 8.02 in-block comparisons over compressed keys) | instructions/op, indirect mispredicts/op | already present: SV vs RAW vs B decomposes "layout" from "decode" (fb n = 1: 1.16 / 0.96 / 0.73 Mops) |
| H-write: the `write_heat` store per lookup | `L1D_CACHE_WRITEBACK`/op of B vs SV | a one-line flag to skip `observe()` on reads, in the D layer only |
| H-size: it is the memory hierarchy, not the algorithm | everything scales with the key count | B and SV at 2M, 20M and 200M keys (the 2M samples exist); the crossover point is the answer |

**Converting counts to time.** Measure this host's dependent-DRAM latency and TLB-walk cost once (the section 2.3
latency probe at 64 B and at 16 KiB stride). Then fit cycles/op ≈ a·instr + b·L1Dmiss + c·walks + d·mispred
across cells and datasets. A fit that leaves SV's advantage unexplained points to MLP.

---

## 7. Linux replication (2.2) (major for Linux)

- **`taskset -c 2` applies to the whole process,** and threads inherit affinity. All 16 build threads would then
  share core 2: a CSV build on one core takes about 16 x 300 s. **Fix (P-linux, about 0.5 h):**
  - `--pin-core N`: the main thread runs on the housekeeping set during the build, its build threads inherit that,
    and it calls `sched_setaffinity({N})` after the threads join and before the cool-down.
  - Verify with `perf stat -e cpu-migrations` = 0 over the window.
- **Add `sysctl kernel.numa_balancing=0`.** On a 2-socket box, AutoNUMA page migration during timing is a known
  noise source.
- **Fix the uncore (mesh) frequency.** On Skylake-SP it is dynamic and sets memory latency. Pin it through the
  `intel_uncore_frequency` sysfs (min = max), or MSR 0x620. This is the Linux counterpart of the fabric-state
  concern on the Mac.
- **Stop `irqbalance`.** `echo mask > /proc/irq/*/smp_affinity` does not write to multiple files; loop over them.
  Limit deep C-states on the timed core (`cpupower idle-set -D 2`).
- **`perf stat --control fifo:` needs perf >= 5.9.** Record the version.
- **Add a THP = always arm** for SV and B (the TLB hypothesis). AIDB's own setting stays the primary.
- **The replication must include SV and RAW,** not only B, N, Nf, C, Cr and G. Whether binary search wins on
  4 KiB pages with a 1 MiB L2 is exactly what the Mac cannot say: 16 KiB pages give about 4x the TLB reach.
- **Use x86, not Linux on aarch64.** `LinearModel::fit_xy` accumulates in `long double`, which is IEEE quad
  emulated in software there: builds would be far slower and the models different.

---

## 8. Comparability with AIDB (20M warm-up / 100M measured, one pinned thread) (minor)

- **The sampling rule is ambiguous.** AIDB says "random lookups for all keys in the dataset". That reads as either
  uniform draws or a permutation. Say so; at 100M of 200M the two differ negligibly for cache behaviour.
- **The BA/CA pass rule always passes.** The rule is "the BA/CA ratio's CI overlaps the main sweep's C/B CI", and
  at n = 3 overlapping CIs pass almost always. **Fix:** test the protocol x cell interaction,
  ln(CA/BA) - ln(C/B), against an equivalence margin (±3%), or report it descriptively and say so. Under P6,
  AIDB length is cheap (100M timed per cell in one process), so put BA/CA inside the interleaved harness.
- **Only ratios carry over.** Absolute Mops cannot be compared across machines. The comparison needs 4 KiB pages
  and a pinned core (section 7). State this on the AIDB slide.

---

## 9. run_ba.py and the schedule: things that would bias a cell (major)

1. **Resumability blocks the re-queue.** `done()` keys on (dataset, cell, block). Once an invalid run is written
   to `t.jsonl`, the re-queue is skipped as already done. **Fix:** key on (dataset, cell, block, attempt), and
   count as done only rows with `valid` true or with attempts exhausted.
2. **Re-queueing "at the end of its block" breaks contiguity.** Cells of one dataset are kept contiguous
   deliberately. A re-run then lands hours later, after other datasets, and those retries are more often C
   (section 2.2). **Fix:** retry immediately after the quiet gate (it already waits for quiet), at most 2 tries.
3. **The full version confounds night with cell.** It adds cells to blocks 1-3 (RAW and RC on 9 datasets) and
   blocks 1-4 (NC, GC, NG) on a later session. Their baselines (B, C) for those blocks ran on the overnight session,
   and 6.2 compares unpaired means across the whole set. **Fix:**
   - Each session gets its own complete blocks, re-running B, B2 and C alongside any added cell.
   - Contrasts are estimated within a session, then combined.
   - "Run() skips finished jobs if pointed at t.jsonl" must not mix sessions.
4. **Chunks.** `--chunks 16` should become 80 (section 1.1).
5. **D-layer runs time pass 1 too.** Strip or rename `throughput_ops_s` in `d.jsonl` so it can never enter a
   throughput table (wrong warm-up, any load, instrumented process).
6. **COLD inherits `--prefault 0` from `common()` today.** Under the draft every other cell switches to prefault 1,
   so COLD must stay an explicit override, as the draft says. Add a test that the plan header lists it.
7. **`pgrep -f 'scaleli_bench|scaleli_hardness'`** also matches any shell or editor command line containing those
   strings. The draft's checklist pattern, `'scaleli_'`, matches the new source tree `scaleli_proto` too. Match
   on the executable path instead (`pgrep -x scaleli_bench`).
8. **The runner and hostmon** must start from the same login session as the quiet checklist, with `caffeinate`
   and no `taskpolicy` on the runner. Log `taskpolicy -G -p <bench pid>` once per run to prove the bench is not
   throttled (background clamp).

---

## 10. Smaller points

- **E1 decides on a point estimate.** sigma_AA from 20 runs over 30 min can read <= 5% when the truth is 6%. Gate
  on the one-sided 80% upper bound, or let the live A/A decide and treat E1 as a fail-fast. Report a robust sigma
  (1.4826 x MAD) next to the SD, because the mixture is heavy-tailed.
- **There is no A/A for an expensive build.** The live A/A covers B-type cells only, and heat matters for C. Add
  C vs C2 (an identical rebuild) on fb in 2 blocks. It doubles as the run-to-run check of the chaotic CSV greedy
  (threat 12).
- **The pilot already ran with prefault on.** `pilot_seed11.jsonl` has `prefault: true` in all 24 runs, so the
  "PLAN.md had 0" line in 3.4 and Appendix A describes `run_ba.py`, not the pilot. R1 at "pilot settings" must pass
  `--prefault 1`.
- **One mechanism slips through every chunk check: a run that starts and stays in a slow host state.** In the
  pilot, `R0sv` on fb had c1 = -0.305 after a 2M warm-up. That is a state flip, not a cold cache. The placebo-chunk
  test (1.3) and the P-core / runnable telemetry (4.2) are what separate these.

---

## Revised in-run sequence (per-process fallback; P6 uses the same steps per visit)

```
load + canonicalize + trace (untimed)              ; latency-probe buffer allocated here (common to all cells)
build (16 threads, build_ns)                        ; record SoC temp before/after (hostmon or in-process)
learnability + memory_before
[BH only] preheat 300 s all-core
cool-down: wait until temp <= pre-build + 2 °C (cap 300 s) AND runner writes "go" (quiet last 30 s)
counters snapshot #1 (rusage V6, task events, vm compressions)
freq probe (LCG) + latency probe (pointer chase)    ; pre
prefault (insurance; no claimed effect)
warm-up 4,000,000 workload-mode lookups, distinct seed
counters snapshot #2
TIMED 5,000,000 lookups, 80 chunks; per chunk: wall, cycles, pcycles, instructions, cpu_ns, runnable_ns
counters snapshot #3 ; freq probe + latency probe (post)
```

Validity (outcome-blind):

- P-core fraction >= 0.99;
- runnable time <= 1% of the window;
- zero pageins, swapins and decompressions in the window;
- hostmon CPU computed from deltas;
- on AC, with powermode unchanged.

Thermal acceptance:

- BH/B within ±2%;
- latency probe C vs B within ±1%;
- GHz reported but not gating.

Warm-up acceptance:

- the pooled placebo-chunk test within ±3%;
- COLD pooled at least -8% at 312.5k (or at least -13% at 125k).
