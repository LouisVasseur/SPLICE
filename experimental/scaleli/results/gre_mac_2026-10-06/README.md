# GRE baselines at 200M keys on the Mac (PREVIEW, 2026-10-06)

**These are preview numbers from a MacBook Pro (Apple M3 Max). The Mac is not the Atom.** They show whether GRE's
driver runs end to end at full scale on this machine, and they give a rough idea of the rankings. They do not replace the
Atom runs. Do not mix them with Atom numbers in a table or a ratio. `tools/gre_mac.md` lists the differences between
the two machines.

## What ran

- **Binary.** `tools/gre_mac.sh` built GRE natively for arm64. The source is GRE at
  e807edcef51df6732f07f94d4c797fb3897519ba with the committed `gre_lite.sh` (SPLICE 59d810f) run in edit-only mode,
  plus the arm64 patches. The compiler is Apple clang 17 with `-O3 -mcpu=native` (apple-m3).
  `warmup_pub/gre_mac_build.txt` records the patches and flags. The microbench sha256 is `6ddeaf85…030cc`.
- **Runner.** The committed `tools/gre_run.sh` (`git show HEAD:`), unmodified, ran under `nohup caffeinate -i` with
  `OMP_NUM_THREADS=1`. It ran on macOS without changes. Its Linux-only provenance sections (lscpu, free, nproc, ldd,
  git) print "(unavailable)", so `warmup_pub/provenance_mac.txt` holds the macOS equivalents.
  ```
  gre_run.sh results/gre_mac_2026-10-06/warmup_pub --gre <scratchpad>/gre_mac/GRE --data data/external/gre \
    --datasets fb,osm,books,stack.sorted --indexes sortedarray,pgm,lipp,btree,alex \
    --repeats 3 --warmup 20000000 --pin -1
  ```
- **Condition.** The workload is read-only and single-threaded. All 200M keys are bulk-loaded. Each run does 20M
  untimed warm-up lookups and then 100M timed lookups of loaded keys, with GRE's seed 1866. The runs are unpinned,
  because macOS cannot pin threads. That gives 3 repeats × 4 datasets × 5 indexes = 60 runs. fb, osm and books use
  the raw files, which are already sorted; stack uses `stack.sorted`.
- **Not run.** ART (artunsync) is left out of the Mac build because it needs SSE2. SCALE-LI and SPLICE cells were not
  part of this grid.
- **Time.** The grid ran from 04:33 to 07:05 CEST. There was one pause, from 04:43 to 05:19, followed by `--resume`.
  The pause is explained under Load below.

## Results per dataset

The columns are:
- **Mops/s:** GRE's `Throughput` over the 100M timed lookups.
- **index memory:** RSS after the bulk load minus RSS before it, per key. On macOS this is phys_footprint, measured
  after `malloc_zone_pressure_relief`.
- **build s:** the bulk-load time.
- **peak RSS:** the process peak, including GRE's own key and operation arrays.
- **correctness:** a run passes when it exits 0, reports a throughput, finds all 100M lookup keys
  (success_read = 100000000) and finds all 20M warm-up keys.

A `*` marks a run that started at a 1-minute load average of 8 or more, or beside another benchmark. Means include
these runs. `warmup_pub/report.md` is the full `gre_report.py` output, with CIs, unpurged memory and self-reported
memory.

### fb

| index | mean Mops/s | min | max | r1 / r2 / r3 | index memory B/key | build s | peak RSS GB | correctness |
|---|---|---|---|---|---|---|---|---|
| sortedarray | 0.901 | 0.783 | 0.998 | 0.998 / 0.922 / 0.783 | 16.01 | 0.90 | 11.7 | PASS 3/3 |
| pgm | 3.187 | 2.906 | 3.737 | 3.737 / 2.918 / 2.906 | 16.18 | 4.72 | 11.4 | PASS 3/3 |
| lipp | 6.296 | 6.002 | 6.527 | 6.527 / 6.358 / 6.002 | 134.53 | 7.35 | 38.5 | PASS 3/3 |
| btree | 1.423 | 1.210 | 1.840 | 1.840* / 1.210 / 1.219 | 19.38 | 1.15 | 12.0 | PASS 3/3 |
| alex | 1.988 | 1.854 | 2.231 | 2.231* / 1.854 / 1.880 | 24.01 | 41.04 | 13.0 | PASS 3/3 |

### osm

| index | mean Mops/s | min | max | r1 / r2 / r3 | index memory B/key | build s | peak RSS GB | correctness |
|---|---|---|---|---|---|---|---|---|
| sortedarray | 0.797 | 0.775 | 0.833 | 0.782 / 0.775 / 0.833 | 16.01 | 0.89 | 11.7 | PASS 3/3 |
| pgm | 2.911 | 2.467 | 3.625 | 3.625 / 2.642 / 2.467 | 16.12 | 4.67 | 11.4 | PASS 3/3 |
| lipp | - | - | - | - | - | - | - | FAIL 0/3: exit 137 (killed by macOS, out of memory) |
| btree | 1.298 | 1.224 | 1.444 | 1.444 / 1.225 / 1.224 | 19.38 | 1.19 | 12.0 | PASS 3/3 |
| alex | 2.487 | 2.194 | 3.055 | 3.055 / 2.213 / 2.194 | 24.51 | 38.56 | 13.1 | PASS 3/3 |

### books

| index | mean Mops/s | min | max | r1 / r2 / r3 | index memory B/key | build s | peak RSS GB | correctness |
|---|---|---|---|---|---|---|---|---|
| sortedarray | 0.904 | 0.779 | 1.149 | 1.149 / 0.779 / 0.785 | 16.01 | 0.87 | 11.7 | PASS 3/3 |
| pgm | 3.966 | 3.570 | 4.452 | 4.452 / 3.875 / 3.570 | 16.07 | 3.60 | 11.4 | PASS 3/3 |
| lipp | - | - | - | - | - | - | - | FAIL 0/3: exit 137 (killed by macOS, out of memory) |
| btree | 1.383 | 1.213 | 1.551 | 1.551 / 1.213 / 1.385 | 19.38 | 1.19 | 12.0 | PASS 3/3 |
| alex | 6.340 | 6.219 | 6.440 | 6.219 / 6.440 / 6.361 | 23.19 | 27.46 | 12.8 | PASS 3/3 |

### stack (stack.sorted)

| index | mean Mops/s | min | max | r1 / r2 / r3 | index memory B/key | build s | peak RSS GB | correctness |
|---|---|---|---|---|---|---|---|---|
| sortedarray | 0.839 | 0.777 | 0.930 | 0.808 / 0.777 / 0.930 | 16.01 | 0.83 | 11.7 | PASS 3/3 |
| pgm | 4.774 | 4.674 | 4.973 | 4.676 / 4.674 / 4.973 | 16.01 | 2.84 | 11.4 | PASS 3/3 |
| lipp | 26.817 | 22.744 | 33.103 | 22.744 / 24.606 / 33.103 | 44.82 | 1.92 | 20.2 | PASS 3/3 |
| btree | 1.498 | 1.253 | 1.777 | 1.777 / 1.253 / 1.463 | 19.38 | 1.06 | 12.0 | PASS 3/3 |
| alex | 8.975 | 8.216 | 9.761 | 9.761 / 8.216 / 8.949 | 23.44 | 25.25 | 12.9 | PASS 3/3 |

**Correctness.** 54 of 60 runs pass every check. Every run that finished found all 100M lookup keys and all 20M
warm-up keys.

**Memory method.** sortedarray measures 16.01 B/key on every dataset, against exactly 16 expected (gre_report: OK,
+0.06% to +0.08%).

**The six failures are all LIPP on osm and books.** macOS killed each of them with SIGKILL (exit 137) during the bulk
load. The memory log (`warmup_pub/memlog.txt`) shows LIPP reaching about 31 GB resident on osm. After that, the
compressor grew to about 37 GB and swap to more than 20 GB until the process was killed. The Jetsam report from the
same failure in the aborted first attempt (03:47) names microbench as the largest process, at a time of
vm-compressor-space shortage.

LIPP on osm and books needs more memory than this 64 GB Mac has free alongside its desktop apps; I did not measure
how much. LIPP on fb fits, with a 38.5 GB peak. The task note expected about 27 GB for fb, and 27 GB is fb's *index*
memory (134.5 B/key). Whether these LIPP cells fit on the Atom (64 GB, nothing else running) is a separate question;
`docs/MATCHING_NFL_CSV.md` already lists it as unchecked.

## Load and noise

- **Load at the start of each run (1 min).** Minimum 2.94, median 4.67, maximum 15.33. Desktop apps were running
  throughout: ChatGPT used about 70-80% of a core and WindowServer about 40%.
- **Another workflow was benchmarking.** A SPLICE workflow running at the same time started several microbench,
  splice_count and test runs, at up to 1400% CPU and load averages of 60-86. My first attempt at 03:31 overlapped it.
  I stopped that attempt and moved it out of the repo, to the scratchpad `run200/attempt1`. A LIPP run in it had been
  killed under the combined memory pressure.
- **The pause.** The second attempt began once the machine was quiet. At 04:43 the other workflow started again,
  including its own LIPP-on-osm run, so I paused the grid to avoid a double 30+ GB LIPP. I resumed with `--resume`
  at 05:19.
- **Disturbed runs.** In r1 on fb, btree started at load 15.3 and alex beside the other workflow; both are marked `*`.
  `osm__sortedarray__r1.log.prev` is the run cut off by the pause. Every other run started with no other benchmark
  running and a load below 8 (the highest was 7.77).
- **Repeat 1 was faster.** r1 is clearly faster than r2 and r3 for pgm, btree and alex on fb, osm and books; for
  example, fb btree measured 1.84, 1.21 and 1.22. The difference is not the disturbed runs: busy=0 for all the others.
  A single fb btree run after the grid gave 1.34 Mops/s; it is in the scratchpad, not here.
- **Possible causes, none confirmed.** Unpinned scheduling across P and E cores, the battery going from 36% to full
  during r1, and the other process's memory state.
- **Low Power Mode was on.** `NSProcessInfo.isLowPowerModeEnabled` was true for the whole grid (`pmset` powermode 1,
  on AC power), with thermal state nominal. This lowers clock speeds. I did not change this system setting.
- **What to use.** Use medians or the min/max range, not single runs. gre_report's per-dataset pooled SD of ln
  throughput is 0.13-0.15, which is about ±14%.

## Comparison with the Atom on fb (not like for like)

The Atom figures are the DIAS GRE fb baselines cited in `docs/SPLICE_DESIGN.md` §6 (4 KiB pages).

| index | Mac mean Mops/s | Atom Mops/s | Mac / Atom | Mac vs sortedarray | Atom vs sortedarray |
|---|---|---|---|---|---|
| LIPP | 6.296 | 1.11 | 5.67 | 6.99 | 2.43 |
| PGM | 3.187 | 1.17 | 2.72 | 3.54 | 2.56 |
| ART | (not built on arm64) | 1.00 | - | - | 2.19 |
| ALEX | 1.988 | 0.885 | 2.25 | 2.21 | 1.94 |
| B+tree | 1.423 | 0.747 | 1.90 | 1.58 | 1.63 |
| sorted array | 0.901 | 0.457 | 1.97 | 1 | 1 |

The Atom ranking is PGM 1.17 > LIPP 1.11 > ART 1.00 > ALEX 0.885 > B+tree 0.747 > sorted array 0.457. On the Mac it
is LIPP > PGM > ALEX > B+tree > sorted array. Four of the five indexes keep their relative order, and the Mac is
about 1.9-2.7 times faster on each of them. LIPP is the exception: it moves from tied with PGM to twice as fast as
PGM. Its ratio over the sorted array is 7.0 on the Mac and 2.4 on the Atom.

**The Mac is not the Atom.** The differences are listed below; see `tools/gre_mac.md` for details.
- **Pages:** 16 KiB on the Mac, 4 KiB on the Atom. Larger pages give much more TLB reach, which matters most for
  LIPP's 27 GB index.
- **Caches:** the M3 Max has a 128 KiB L1d and a 16 MiB L2 per P-cluster, and much lower memory latency.
- **Scheduling:** unpinned, a single thread, Low Power Mode on.
- **Libraries:** libc++ instead of libstdc++, so the same seed samples different lookup keys. The allocator is the
  macOS one rather than jemalloc.
- **Floating point:** hardware FMA, and 64-bit long double. LIPP's and ALEX's models use long double, so their
  structures can differ from the Atom's.
- **ART** is missing.

The Mac numbers cannot confirm or refute an Atom prediction. Use them only to check that the pipeline works and to
form hypotheses. In particular, check on the Atom whether LIPP's lead holds there; the 2.43 vs 2.56 Atom ratios say
it does not.

## Files

The `warmup_pub/` directory holds:
- From `gre_run.sh`: `runs.tsv`, `config.txt`, `provenance.txt`, `logs/` (60 logs and one `.prev`) and
  `gre_out.csv`.
- `report.md`: the committed `gre_report.py` run on this directory. It exits 1 because of the 6 LIPP failures.
- `gre_mac_build.txt`: the build record.
- `provenance_mac.txt`: macOS hardware, OS, power mode and swap.
- `memlog.txt`: a 30-second sample of load, swap, compressor and microbench RSS over both attempts.

**Exclude `warmup_pub/bin/`** (the microbench copy and the gre_lite build record) from anything committed. It is a
build artefact made from unlicensed GRE code and GPL-3 STX code. Nothing in this directory has been committed.
