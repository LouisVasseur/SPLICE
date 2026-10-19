# The Atom, fast: parallel pinned lanes

This file is about running the whole experiment set on the DIAS Atom in about 18 hours instead of about 41:

- the warm-up traces;
- the headline bench;
- the untimed counts.

Single runs keep the conditions of [PROTOCOL.md](PROTOCOL.md): one thread, pinned, the same seeds, flags and
checks. Up to K of them run at once, each on its own L2 module. K is chosen by a calibration that measures what the
co-runners do to the steady latency. `scripts/atom_batch.py` does the scheduling, through `./run.sh atom ...`.
Every run is still a `scripts/gre_run.sh` call, so its OUTDIR, log and checks are the usual ones.

Every time below is an estimate from the planning model in `atom_batch.py`, except where it says *DIAS measured*.
Each run's estimate is uncertain by about ±25-35%. The scheduler replaces estimates with measurements as runs end,
and `./run.sh atom status` shows how far it has got.

## What runs

| set | recipe | runs | conditions |
|---|---|---|---|
| (1) warm-up traces | `warmup` | 10 datasets x 10 `trace_<index>` x 2 rounds = 200 | W=0, 30M lookups, `GRE_TRACE=geo:64:65536 GRE_TRACE_CPU=1` |
| (2) headline | `bench` | 10 x 7 indexes x 3 rounds x (warm, cold) = 420, plus 12 solo anchors when K > 1 | warm: W=20M + 100M timed; cold: W=0, 100M |
| (3) untimed | `counts` | the 7 `.sorted` copies, then splice_count and nf_gate on 10 datasets | 16 threads, nothing timed beside them |

- Trace indexes: nullindex, splice, pgm, btree, alex, sortedarray, lipp, artunsync, scaleli_b, scaleli_n.
- Bench indexes (H7): splice, pgm, lipp, sortedarray, alex, btree, artunsync. To add scaleli_b and scaleli_n (H9),
  use `--indexes splice,pgm,lipp,sortedarray,alex,btree,artunsync,scaleli_b,scaleli_n`.
- In `run warmup,bench`, `--indexes` sets the bench list and `--trace-indexes` sets the trace list. Other options:
  `--datasets`, and `--rounds N` (which applies to every recipe of the call). `--ops`, `--warmup` and
  `--table-size` shrink every run, for tests of the machinery only (an R holds one condition).
- `warmup,bench` puts both sets in one queue, which packs best: short traces fill the gaps left by memory-heavy
  bench runs.

## Validity rules and how each is enforced

1. **One timed run per module, sibling idle.** The CPUs that share an L2 are a module (8 x 2 on the C3958), read
   from `/sys/devices/system/cpu/cpu*/cache/index*/shared_cpu_list` at start. The rules:
   - The module of the lowest CPU is the **house module** (0-1). The scheduler runs there, and so should your
     shells once the builds are done: `taskset -cp 0,1 $$`.
   - Lane j owns one other whole module. Its timed core is the module's first CPU: lane 1 is core 2, as in
     PROTOCOL.md, then 4, 6 and 8.
   - Only complete modules host lanes: every CPU online, allowed and not isolated. "Allowed" is the cpuset the kernel
     grants this user, not the shell's mask, so a shell on the house module still gets every lane.
     `--avoid-cpus` drops modules.
2. **Untimed threads never reach another lane.** Each run's whole process is confined to its lane's module plus the
   pool, using `sched_setaffinity` before exec. The pool is the modules that host no lane: 10-15 at K=4.
   - The process includes GRE's two `tbb::parallel_sort`s, which size their thread pool from this mask (8 threads at
     K=4), and the SPLICE/SCALE-LI build threads (16, the protocol's, kept so builds are unchanged).
   - GRE then pins the timed thread to the lane core (`--pin`).
   - So setups share only the pool and DRAM with timed runs. `OMP_NUM_THREADS=1` as always.
   - `gre_run.sh` counts online CPUs now (`getconf`), not `nproc`, which under a mask would reject core 8.
3. **Parallel timed runs share only DRAM, and the effect is measured first.** `run` refuses K > 1 unless
   `R/calib/verdict.json` admits that K and was measured with the same binary (sha256) on the same topology. The
   calibration is described in the next section.
4. **Memory admission: nothing swaps, and the OOM killer never acts.**
   - A run starts only if MemAvailable stays above the headroom (max 4 GB, 6% of MemTotal). The sum is: MemAvailable
     now, minus each running run's remaining growth, minus the new run's peak. A run's remaining growth is its
     predicted peak minus its RSS until it is pinned. After the pin it is 1 GB, for GRE's post-run work: read-only
     lookups allocate nothing.
   - Peaks start from the Mac logs plus margin, then come from each cell's measured VmHWM, plus 5% and 0.5 GB.
   - A head that does not fit reserves the time when it will. Others may start only if they end before that time,
     or leave it its memory (EASY backfill).
   - LIPP on osm, books and covid has never been measured anywhere, so it runs **alone** until its first run is
     measured.
   - The guard polls every 0.5 s. Below 3 GB of MemAvailable (with two or more runs) it kills the newest run still
     in setup or build. Below 1.5 GB, or below 3 GB while pages are being swapped out, it kills the newest run in
     setup or build, else the newest run. It waits 3 s between kills, so the memory of a killed run is freed first.
     Swap-out while memory is to spare (the kernel trimming other users' idle pages) kills nothing; every run records
     its swap-out (`pswpout` in the merged runs.tsv, listed in the report's Lanes section).
   - A killed run is retried alone once. A cell killed while alone does not fit; its other items are skipped and
     reported.
   - Only a run that passed counts as a measured peak. The VmHWM of a crashed or killed run only raises the estimate,
     so a LIPP cell that was never measured keeps running alone.
5. **Untimed multi-threaded work never runs beside timed runs.**
   - Builds, tests, smoke and `counts` run only while no timed run exists.
   - `counts` items are exclusive and wait for an empty machine.
   - `smoke`, `calib`, `run` and `counts` on one R share a lock, so they never overlap: a second one started while
     the first runs stops at once with "another atom_batch.py is running".
6. **Provenance for every run.** Each attempt writes `R/<arm>/cells/<ds>__<index>__r<round>/a<n>/`, a normal
   gre_run.sh OUTDIR, plus `atom.json`, which records:
   - lane, module, core, siblings, CPU mask, K and pool;
   - the runs alive at its start and their phase;
   - start, pin and end times on the monotonic clock;
   - MemAvailable at start and its minimum, and the swap-out delta;
   - foreign benchmark processes seen;
   - core and sibling busy % and interrupts per second over the timed window;
   - THP, governor, binary sha256 and boot id.

   `merge` adds the overlaps between runs. **solo** means that no other run was alive during the run's timed lookups
   and no foreign benchmark was seen. `sched/events.jsonl` logs every start, pin, end, guard action and pause.
7. **THP and governor are recorded, as in PROTOCOL.md 5b.**
   - `doctor` writes them to `R/machine.txt`, and every gre_run.sh provenance records them again.
   - Under THP `[always]`, every arm is named `*_thpalways`: there is no 4 KiB arm then.
8. **Foreign load pauses the campaign.** Every 5 s the scheduler looks for benchmark processes:
   - what counts: any `--keys_file=` process, splice_count, scaleli_*, nf_gate;
   - a process counts as foreign when it lacks this R's `ATOM_BATCH_ID` in its environment (other users' processes
     cannot be read, so they count as foreign);
   - while one runs, no new run starts and every overlapping run is tagged;
   - `--foreign tag` only records them.

   A calibration session that saw one waits for it to end and is run again (up to three times).
9. **Resume.**
   - Rerun the same command. Cells whose attempt passed every gre_run.sh check are skipped.
   - Attempts cut off by a kill close as `interrupted` and rerun. Orphaned runs of a killed scheduler are stopped
     first.
   - A failed run is retried once, on another lane (it waits for one rather than reuse the lane it failed on).
   - An exit-2 refusal of gre_run.sh (a missing file, an unknown index) ran nothing: three in a row stop everything,
     and after the fix the same command tries those items again.
   - One R holds one condition: a rerun with other `--ops`, `--warmup`, `--table-size` or trace settings is refused.

## The calibration

`calib` tests K = 4, then 3, then 2, and stops at the first that passes.

- **Probes.** `trace_pgm` and `trace_lipp` on fb run on lane 1, with W=20M, 200M lookups and fixed 65,536-call
  chunks. These are the DIAS reference cells, and LIPP fb has the largest TLB footprint of the cells DIAS has
  measured.
- **Loads.** Lanes 2..K loop real GRE runs under the campaign's masks: setups, builds and lookups of sortedarray,
  btree and pgm on wise, osm, genome, books and history. They all start at the probe's pin, so their setups overlap
  the first blocks (every setup at once, the heaviest case) and their lookups the later ones; a probe that started
  beside running loads would see them finish their setups during its own build and measure lookups only.
- **Load size.** A load is a full run when K-1 of them fit beside the probe in memory, otherwise a prefix of the
  file. The prefix has the same DRAM traffic per second and less memory (LIPP at K=4: a 50M-key prefix and 30M
  lookups).
- **Blocks.** From the probe's pin onward, the scheduler stops and resumes all load lanes together (SIGSTOP and
  SIGCONT) in 5 s blocks. Within each pair of blocks the order is random, from a seed. Solo and K-way blocks thus
  alternate inside one probe process, which removes the offset between whole runs. The round-to-round spread is
  4.9-6.1% (*DIAS measured*). Averaging it out with whole runs would take on the order of a hundred pairs or more
  (estimate).
- **Alignment.** `src/gre/trace_interface.h` now prints `trace_t0_mono_ns`, which places every chunk on the clock of
  the toggle log. A build that lacks the line falls back to the polled pin, within 20 ms.
- **Statistic.** For each pair, d = ln(median ns/lookup of the chunks in the K-way block) - ln(median in the solo
  block). Only chunks past the warm-up count, and chunks within 0.25 s of a switch are dropped. The inflation is
  exp(mean d) - 1, with a t interval over the pairs. The pairs whose K-way block had a load in setup or build are
  also reported separately.
- **Rule** (`--bound 1.0`, in %). K passes when both probes have at least 6 pairs, the upper 95% bound of the
  inflation is at most 1%, at least 3 pairs had a load in setup and they average at most 1% (setups run beside
  timed runs in the campaign, so they must be measured), and every chunk ran on the lane core.
- **Admissible K.** It is the largest passing K with no failing K below it. Smaller K are admissible too, since
  fewer co-runners share less DRAM. The result is written to `R/calib/verdict.md` and `verdict.json`, and `run`
  uses that K by default.
- **Run-level check inside the campaign.** At K > 1, each round adds solo warm runs of four anchor cells (pgm/fb,
  lipp/fb, sortedarray/osm, splice/fb). These run alone, all in one drain. `./run.sh atom anchors` pairs each with
  the same cell's parallel warm run of the same round. If the mean inflation exceeds the bound, stop the campaign
  and recalibrate (see the last section).

The 1% bound is measured on PGM and LIPP only. ALEX, ART and the B+tree are covered only through the
sortedarray and splice anchors.

## Expected wall time (estimates, K=4 needs the verdict)

| part | K=1 (serial) | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| (1) traces, 200 runs | 8.7 h | 4.6 h | 3.3 h | 2.8 h |
| (2) bench H7, 420 runs (+12 anchors) | 28.5 h | 16.2 h | 12.3 h | 11.1 h |
| (1)+(2) in one queue | 37.2 h | 20.4 h | 15.2 h | **13.4 h** |
| (1)+(2) with H9 (scaleli_b, scaleli_n) | 52.6 h | 28.1 h | 20.5 h | 18.9 h |

The K=4 run of (1)+(2) is detailed below:

- **Round ends.** Round 1 (half the traces, a third of the bench) ends at about 4.9 h, round 2 at 9.8 h and round 3
  at 13.4 h. Serially, round 1 alone takes 13.9 h.
- **Memory is the binding resource.** The schedule is at 90% of its memory-time lower bound (12.1 h).
  - The bound assumes 64 GB of MemAvailable, which leaves a 60 GB budget after the headroom.
  - With 58 GB of MemAvailable (a 54 GB budget), K=4 takes 15.7 h and gains nothing over K=3.
  - `doctor` prints MemAvailable, and the dry run takes `--mem-available` to replan.
- **Fixed costs**, the same in every schedule:
  - setup, build, test and smoke: about 0.75 h;
  - the lane smoke: about 2-5 min;
  - calibration: about 12 min if K=4 passes first, 35 min if every K is tested;
  - `counts`: about 2.9 h, plus about 20 min for the `.sorted` copies.
- **Total with H7:** about 18 h at K=4, against about 41 h strictly serial.

Print any of these for this machine:

```bash
./run.sh atom run warmup,bench --dry-run              # K=1..4 table, the schedule at the verdict's K, lower bounds
./run.sh atom run warmup,bench --dry-run --lanes 4    # the K=4 schedule, run by run
./run.sh atom calib --dry-run                         # sessions, load sizes, memory, time
./run.sh atom counts --dry-run
```

On the Mac the dry runs plan the Atom by default (8 modules of 2 CPUs, 64 GB available); `--topology 8x2
--mem-available 58` replans it with less memory, and `--topology 0-3/4-9/10-15 --mem-total 64 --mem-available 50`
plans the Mac's own L2 clusters (E cores, then two P clusters).

## Commands on the Atom, in order

Louis runs every command; nothing here contacts the server by itself. Before step 0, on the Mac:
`git push -u origin atom-fast`.

0. Checkout, with the conda env active. Keep the shell on all CPUs for now: the builds of step 1 use them.

   ```bash
   cd /tmp/louisvasseur/SPLICE && git status --short && git fetch && git checkout atom-fast
   ```

1. Builds and tests, with nothing timed running yet. This is the only time builds run. Then move the shell to the
   house module: everything started from it afterwards is light, and `atom_batch.py` gives each run its own mask.

   ```bash
   ./run.sh setup build gre && ./run.sh test && scripts/splice_asm_check.sh g++ | tail -3 && ./run.sh smoke
   taskset -cp 0,1 $$
   ```

   `gre_lite.sh` copies the new `trace_interface.h`, so the build prints `trace_t0_mono_ns`.

2. Results folder and machine state. Read `machine.txt` before going on:
   - THP: `[always]` renames the arms;
   - the governor;
   - the lanes for K=1..4;
   - MemAvailable;
   - the /tmp filesystem: tmpfs puts the datasets in RAM;
   - foreign benchmarks.

   ```bash
   export R=results/atom_$(date +%F)
   ./run.sh atom doctor --out $R
   ./run.sh atom run warmup,bench --out $R --dry-run | head -8
   ```

3. Lane smoke, about 2-5 min, untimed: 10M keys and 3M lookups on fb and wise, in up to 4 lanes. It checks on
   this machine what the Mac tests could only fake: each run's CPU mask as the kernel applied it, the pin seen from
   /proc (the setup/timed phase the memory admission relies on), VmHWM, the `ATOM_BATCH_ID` tag (our runs are never
   "foreign"), trace chunks on the lane core and `trace_t0_mono_ns` on the scheduler's clock (the calibration's
   alignment), then merge and `gre_report.py`. It must end with `Smoke: PASS` (details in `$R/smoke/SMOKE.md`).

   ```bash
   ./run.sh atom smoke --out $R
   ```

4. Calibrate (about 12-35 min), then read the verdict.

   ```bash
   git checkout -b atom-results-$(date +%F)
   nohup ./run.sh atom calib --out $R > $R.calib.out 2>&1 &
   tail -f $R.calib.out                # ends with the verdict table
   cat $R/calib/verdict.md             # "Admissible K = <K>"
   ```

   If the probes fail at every K, the verdict is K=1, and step 5 then runs the strictly serial protocol.

5. The campaign with that K, traces and bench in one queue. `--lanes` may be left out: it defaults to the verdict's
   K, and a K above it is refused. To run the sets separately, use `run warmup`, then `run bench`, with the same
   `--out` and K.

   ```bash
   K=<the admissible K>
   nohup ./run.sh atom run warmup,bench --out $R --lanes $K > $R.run.out 2>&1 &
   ```

6. Then, once step 5 has ended, the untimed counts on all 16 CPUs (a `counts` started while the campaign runs stops
   at once: they share a lock).

   ```bash
   nohup ./run.sh atom counts --out $R > $R.counts.out 2>&1 &
   ```

   The fastest variant chains steps 4-6 with no idle gap (create the results branch of step 4 first:
   `git checkout -b atom-results-$(date +%F)`). The campaign takes K from the verdict by itself. The counts
   start when the campaign ends, even if some runs failed for good (exit 1), but not after a refusal (exit 2) or a
   kill (exit 143).

   ```bash
   nohup sh -c "./run.sh atom calib --out $R && cat $R/calib/verdict.md && \
                { ./run.sh atom run warmup,bench --out $R; [ \$? -le 1 ]; } && \
                ./run.sh atom counts --out $R" > $R.out 2>&1 &
   ```

7. Watch.

   ```bash
   tail -f $R.out                      # (or $R.run.out) one line per start and end, pauses, guard actions
   ./run.sh atom status --out $R       # per arm: pass / fail / running, the last events
   ./run.sh atom anchors --out $R      # after each round: solo vs parallel, must stay within the bound
   free -g
   ```

8. Push after every round. /tmp is lost on reboot. `git add` is light and runs on the house module.

   ```bash
   nice -n 19 git add $R $R.*out && git commit -m "atom $(date +%F): <what finished>" && git push -u origin HEAD
   ```

   Binaries are never committed: `build*/` folders (`R/gre/build/`, `R/smoke/gre/build/`) and
   `results/**/bin/microbench` are ignored.

9. Reports, on the Atom (standard library) or on the Mac after the pull. The campaign already wrote
   `R/<arm>/merged/`; `./run.sh atom merge --out $R` rebuilds it.

   ```bash
   for a in warm cold warm_solo; do [ ! -d $R/$a/merged ] || ./run.sh report $R/$a/merged --splice-json results/splice_h > $R/$a/report.md; done
   ./run.sh atom anchors --out $R > $R/anchors.md
   ./run.sh atom merge --out $R --only parallel      # R/<arm>/merged_parallel: the runs that had company
   ./run.sh analyze $R/trace/merged --json $R/trace/warmup.json > $R/trace/warmup.md   # numpy (else on the Mac)
   ```

   At K > 1 almost every warm and cold run has company; the runs alone are the anchors (`warm_solo`), which
   `anchors.md` pairs with their parallel runs. `merge --only solo` writes `merged_solo` only for arms that have solo
   runs. `gre_report.py` checks each merged run against its own core, adds a Lanes section (solo and parallel
   counts, swap-out) and takes `--only solo|parallel`. `warmup_detect.py` reads the merged `logs/` unchanged. Under
   THP `[always]` the folders are `warm_thpalways` and so on.

## Outputs

```
R/machine.txt                      doctor
R/smoke/SMOKE.md                   the lane smoke (rebuilt by each smoke; never merged with the campaign)
R/calib/K<k>/<probe>_fb/           probe/ (gre_run.sh OUTDIR), load<j>/ (discarded), toggles.tsv, session.json
R/calib/verdict.md, verdict.json   admissible K
R/sched/events.jsonl               every start, pin, end, guard, pause; sched/lock while a scheduler runs
R/<arm>/cells/<ds>__<index>__r<round>/a<n>/   one gre_run.sh OUTDIR per attempt + atom.json
R/<arm>/merged[_solo|_parallel]/   gre_run.sh-shaped: runs.tsv (+ pin lane core mask k solo conc_* columns), logs/ links
R/untimed/                         splice_count/ (+ COUNTS.md), nfgate/<ds>/ and nfgate/cases.jsonl, sorted.txt
R/gre/build/microbench             the frozen binary every run hard-links (git-ignored)
```

Arms are `trace`, `warm`, `cold` and `warm_solo` (`*_thpalways` under THP `[always]`).

## Not done: further levers (each needs a decision and an A/B check)

- **Free GRE's unused key copies in read-only runs.** This saves about 3.2 GB per run, so a fifth lane fits, for
  about 15-20% less wall time. It is a GRE patch, and it changes the peak RSS column (index RSS is unchanged).
- **Cache the prepared harness** per (dataset, ops, seed), checked by sha256. Setup drops to about 20 s, and
  (1)+(2) to about 9 h at K=4. GRE would then no longer build its input in the process.
- **A gated mode, for when only the setups fail the calibration.** Setups would run only while no run is timed, and
  this needs a gate in the GRE patch. Not built: if the setup pairs fail, the verdict refuses that K and a smaller K
  or K=1 runs instead.

## When something goes wrong

- **The anchors fail.** Stop the chain: `pkill -f atom_batch.py`. The scheduler stops its runs (their attempts rerun
  later) and exits with 143, which ends the chain before the counts. Then run
  `rm -r $R/calib/K* && ./run.sh atom calib --out $R --all-k`, read the verdict and rerun `run` with the new K. Runs
  already done stay, tagged with their K and solo/parallel, so the report can separate them.
- **After a reboot.** Redo steps 0-1 at the same paths, `git pull` the results branch and rerun the same command.
  The binary is frozen again from the new build:
  - If its sha256 differs, the verdict no longer applies: recalibrate, or use `--lanes 1`.
  - `--trust-verdict` overrides this, and every run records that it did.
  - Overlaps are only computed within one boot.
- **LIPP on osm or books does not fit.** It runs alone first. If the guard kills it alone, the cell is reported as
  "does not fit" and its other items are skipped.
- **The lane smoke fails.** `$R/smoke/SMOKE.md` names the check. A mask that the kernel did not apply as asked, or
  the microbench's CPUs outside its lane, rules out K > 1 (setups could reach another lane's module): run
  `--lanes 1`, which keeps PROTOCOL.md's serial conditions. A pin that is never seen leaves the calibration without
  blocks (they start at the probe's pin), so the verdict is K=1; the serial campaign itself only loses packing
  (memory stays reserved at the predicted peak) and exact overlap columns. Trace chunks off the lane core or a
  missing `trace_t0_mono_ns` must be fixed (rebuild GRE) before calibrating.
- **A run fails its checks.** It is retried once on another lane, and the log is next to it (`gre_run.out`, `logs/`).
  The merged runs.tsv keeps every attempt; the passing one counts.
