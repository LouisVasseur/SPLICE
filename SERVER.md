# Running SPLICE on a server

This repository is the full SCALE-LI workspace: the original walkthrough package, the experimental
map in `experimental/scaleli/`, and every NFL / CSV / fusion / joint-design / gap-removal study run on it
(Sep 21 - Oct 5 2026). Datasets and build outputs are not committed; this page rebuilds them.

## Requirements

- Linux x86-64 or macOS; CMake >= 3.16; a C++20 compiler (GCC >= 11 or Clang >= 14); Python >= 3.10
  (standard library only); `zsh` (the run scripts use it); `curl`.
- About 28 GB free disk for the ten GRE datasets (16 GB) and the sorted copies of the seven unsorted ones (11.2 GB).
- 32 GB RAM or more: one 200M-key run holds ~3.2 GB of harness copies plus the index.
- Plots only: matplotlib (`results/aidb/*_figure.py`, `probes_four.py`, `throughput_four.py`).

## 1. Build (the protocol expects the binaries in `build-fs/`)

```bash
cmake -S experimental/scaleli -B build-fs -DCMAKE_BUILD_TYPE=Release
cmake --build build-fs -j
```

Release means `-O3 -DNDEBUG`, as on the Mac. `build-fs/scaleli_bench` and `build-fs/scaleli_hardness` are the two
binaries every script uses. To point a script at another build, set `SCALELI_BENCH` (`run_ba.py`, `run_pilot.py`)
or `SCALELI_HARDNESS` (`results/aidb_ba/run*.sh`).

## 2. Data

```bash
experimental/scaleli/data/external/gre/download.sh
cd experimental/scaleli
python3 tools/aidb_pipeline.py --binary-dir ../../build-fs --steps verify-downloads,sort,sample
```

The download (bash + curl) is resumable, runs four files in parallel (~1.6 GB each, from the GRE mirror) and ends by
checking every file against `SHA256SUMS`, the copies all Mac results were computed on. The pipeline then
writes `<name>.sorted` copies for the seven datasets GRE serves unsorted, and draws the 2M-key samples
(`data/samples/<name>_2M_uniform_s42`, seed 42) that the 2M-scale scripts read. The `.manifest.json` beside each
sample is committed, so a server sample can be checked against the one used on the Mac.

## 3. Validate the setup (before any timed run)

`validate.py` checks that the index and harness are compatible with the data, the metrics, NFL and CSV, the
baselines, this host and the protocol. It prints PASS / FAIL / WARN / UNKNOWN for every check, with the numbers
behind it; `experimental/scaleli/docs/CODE_DISSECTION.md` explains the code each check relies on.

```bash
cd experimental/scaleli/results/aidb_ba
python3 run_ba.py plan validity > validity_plan.json && python3 run_ba.py run validity_plan.json validity.jsonl
python3 run_ba.py plan verify > verify_plan.json && python3 run_ba.py run verify_plan.json verify.jsonl
python3 validate.py --runs validity.jsonl verify.jsonl --out validation
```

- `validity`: one instrumented run of the 8 configurations (B, Bb, N, Nf, Cr, C, NC, SV) on all 10 datasets with
  the protocol's D-layer settings, about 5 h. `--datasets fb,osm` gives a first pass in about 1 h.
- `verify`: the same 8 configurations on fb, checked answer by answer against `std::map` (about 35 min, about 20 GB RAM).
- `aa` (optional, quiet machine only): the night-0 A/A noise test, 10 x B and 10 x B2 on fb (about 25 min). Add
  `aa.jsonl` to `--runs`.
- Smoke test of the whole chain in minutes: add `--limit 1000000` to each `plan` (validate.py labels it SMOKE).
- validate.py also hashes the ten data files (cached after the first run), runs `ctest`, recomputes history and
  libio hardness against the AIDB paper (about a minute), and probes the binary for the P/X options and ALEX/PGM.
  Exit status: 0 gate passed, 1 a gate check failed, 2 gate incomplete.

Reading the verdicts: GATE must pass before any number counts, TIMING before any timed number counts, and SCOPE
says what the setup can show at all. As of 2026-10-05 the scope checks report that in-region learned routing cannot
beat binary routing even with a perfect model (HEAD-1), so region-level NFL/CSV effects are not a valid improvement
test; the root level is, on datasets where a learned root is adopted.

## 4. What to run

The plan for the next meeting (memory footprint, throughput with a cache warm-up, and input-space compression at
200M keys, before vs after NFL and CSV) is `experimental/scaleli/results/aidb_ba/PROTOCOL.md`. Start with
night 0 (section 5.2), which uses the existing binary and needs no patch. Section 2 of the protocol was written
for the Mac; on Linux use section 2.7 and replace the macOS tools (`pmset`, `/usr/bin/time -l`) with their Linux
equivalents (`cpupower frequency-info`, `/usr/bin/time -v`).

```bash
cd experimental/scaleli/results/aidb_ba
python3 run_ba.py show B fb            # prints the exact bench command for one cell
python3 run_ba.py plan aa > plan_aa.json && python3 run_ba.py run plan_aa.json aa.jsonl
```

## 5. GRE baselines

GRE (gre4index/GRE, VLDB 2022) is the reference harness for the baseline indexes. GRE has no licence, so it is
cloned and edited outside this repository; only the scripts below are committed.

Build (inside the activated conda environment), only while no timed run is active: the -O3 compile on all cores
would disturb it, and gre_lite.sh refuses while it sees one (`GRE_LITE_FORCE=1` overrides).

```bash
experimental/scaleli/tools/gre_lite.sh          # -> /tmp/louisvasseur/GRE/build/microbench and build/gre_lite_build.txt
```

Everything under `/tmp` (GRE, the conda env, the datasets) is lost if the server reboots: Ubuntu 18.04 empties
`/tmp` at boot unless the admin changed it. On DIAS there is no home directory (`HOME=/tmp/louisvasseur`), so `~` and
the results below are under `/tmp` too: push each results directory back to the repository as soon as it is done.

Smoke test on a 10M-key prefix of fb. GRE exits 0 even on a wrong file or index name, so check the output, not the
exit status: every line must show `success_read: 1000000` (`sortedarray` exists only in the patched build).

```bash
cd /tmp/louisvasseur/GRE && FB=/tmp/louisvasseur/SPLICE/experimental/scaleli/data/external/gre/fb
for idx in btree alex lipp pgm artunsync sortedarray; do
  OMP_NUM_THREADS=1 ./build/microbench --keys_file=$FB --keys_file_type=binary --read=1 --insert=0 \
      --operations_num=1000000 --table_size=10000000 --init_table_ratio=0.5 --thread_num=1 --memory \
      --index=$idx --output_path=smoke_fb_10m.csv > smoke_$idx.log 2>&1
  echo "$idx: $(grep -E '^(Throughput|Memory|success_read)' smoke_$idx.log | tr '\n' ' ')"
done
```

`tools/gre_run.sh` runs the timed grid: one process per run, single thread, read-only, every key bulk-loaded,
100M lookups, in the order repeat -> dataset -> index. It copies the binary into OUTDIR/bin first (its shared
libraries still come from the conda env, so do not update the env mid-grid), refuses to start while any GRE run
(`--keys_file=` on a command line, whatever the binary is called) or SPLICE benchmark runs, records provenance (CPU,
governors, memory, load, transparent huge pages, NUMA balancing, MALLOC_CONF, SPLICE and GRE versions, binary hash
and ldd, command line), notes the load and any other benchmark at the start of every run, and checks every run.

Three conditions, so that the warm-up and pinning effects can be told apart: GRE as published (no warm-up,
unpinned), pinned to core 2 without warm-up, and AIDB-style 20M untimed lookups then 100M measured on core 2. They
are interleaved in time, one repeat of each per round with the order rotated every round, so that drift on the
shared server does not fall on one condition: each call runs one more repeat of one condition (`--resume` with a
growing `--repeats` skips the repeats already done).

```bash
cd ~/SPLICE
R=~/gre_results; mkdir -p $R
D=fb,osm,books,covid,genome,history,libio,planet,stack,wise; I=btree,alex,pgm,artunsync,lipp,sortedarray
export R D I
nohup bash -c 'for k in 1 2 3; do
  for j in 0 1 2; do
    case $(( (k + j - 1) % 3 )) in
      0) c=published a="--warmup 0 --pin -1";;
      1) c=pinned    a="--warmup 0 --pin 2";;
      2) c=warmup    a="--warmup 20000000 --pin 2";;
    esac
    rc=0; experimental/scaleli/tools/gre_run.sh $R/$c --datasets $D --indexes $I $a --repeats $k --resume \
        >> $R/$c.out 2>&1 || rc=$?
    [ $rc != 2 ] || { echo "gre_run.sh refused, see $R/$c.out"; exit 2; }   # 1 = some runs failed: go on
  done
done' > $R/grid.out 2>&1 &
```

`--warmup`, `--pin` and the `sortedarray` index need the patched build of `gre_lite.sh`; the script checks the binary
before starting. `D=fb,osm` gives a first pass; time the first round before launching all ten datasets. Start the
grid only after any other GRE run (such as a copy of the unpatched binary) has finished: the first call refuses, and
the loop stops, while one is running.

What to expect: from a fresh process the cold transient lasts at most about 375k lookups (aidb_ba/PROTOCOL.md), under
0.4% of the 100M timed lookups, so the warm-up should change throughput by less than about 1%. Pinning (a migration
loses the L2 that Goldmont shares per module) and drift on the shared server can be larger; compare warmup with
pinned for the warm-up effect, and pinned with published for pinning. `--warmup` is for read-only runs only (the
script refuses it otherwise): with inserts it pre-trains ALEX's cost model. The patched binary also purges the
allocator before the timed region, which leaves read-only runs unchanged; with inserts it is only approximately
comparable with unpatched GRE.

Pages: SCALE-LI's protocol runs with transparent huge pages `never`
(`cat /sys/kernel/mm/transparent_hugepage/enabled`). If the server uses another setting and you cannot change it,
the report's provenance shows it; then only ratios to sortedarray, not absolute Mops/s, carry over to SCALE-LI
(which also runs with the performance governor and an isolated core).

Watching: `tail -f $R/grid.out $R/*.out` prints one line per run (throughput, success_read, index RSS, BUSY if
another benchmark was running when it started, FAILED and the reason if a check fails);
`column -t $R/published/runs.tsv | tail`; `free -g`. If the grid stops (kill, lost ssh session), start the same
block again: every call has `--resume`, so runs that passed every check are skipped and the others rerun (the old
log is kept as `*.log.prev`). After a reboot the env, GRE and the datasets under `/tmp` are gone: recreate them at
the same paths before resuming, since the binary copies in `$R/*/bin` load their libraries from the env. Killing
`gre_run.sh` also stops its current microbench; kill the `bash -c` loop first.

Report (standard library, Python >= 3.6):

```bash
for c in published pinned warmup; do python3 experimental/scaleli/tools/gre_report.py $R/$c > $R/$c/report.md; echo $c $?; done
```

It lists every run's checks (exit status, throughput printed, success_read == operations_num, every warm-up key
found, pinning confirmed; exit 1 if any fails), then per dataset and index: mean, min and max throughput in Mops/s,
the geometric-mean ratio to the reference index (`--ref`, default sortedarray) with a 95% CI, each cell's own
spread (flagged when it exceeds twice the pooled SD the CI assumes), index memory in bytes per key, peak RSS, and
build time. A run that started beside another benchmark is listed under Warnings.

Memory caveats:

- "index RSS" is the process RSS after bulk load minus before it, both after a jemalloc purge. It counts nodes,
  the index's own copies of keys and payloads, and allocator slack; GRE's own key arrays are allocated before and
  are excluded. RSS is page-granular, so per-key values only mean something at full scale, not on the smoke prefix.
  It is not SCALE-LI's "real B/key" (an unpurged RSS plateau during the run, `aidb_ba/memory_report.py`); the
  report's "unpurged" column is the closer match. "peak GB" (VmHWM) shows the headroom left in the 67.5 GB.
- GRE's `Memory:` is each wrapper's self-report: the STX B+tree reports 0, and ART excludes its 16-B key/payload
  records. Do not compare it across indexes.
- The sorted array needs exactly 16 B/key, and before building it allocates and frees another 16 B/key in small
  blocks that only the purge can release. OK (within 2%) therefore means that the RSS delta tracks one resident
  allocation and that the purge works; it does not prove that no other index keeps slack. WARN near +100% means the
  purge did not work (or jemalloc is not the process's malloc: check the ldd row), and every index RSS of that
  dataset then includes retained build temporaries.

What to push back: each OUTDIR without `bin/` (the binary; its hash and build record are in provenance.txt), plus
the `.out` file, under `experimental/scaleli/results/gre_<date>/` on a branch. Never the datasets or GRE itself.

```bash
G=experimental/scaleli/results/gre_$(date +%F); mkdir -p $G
for c in published pinned warmup; do rsync -a --exclude bin/ $R/$c/ $G/$c/ && cp $R/$c.out $G/; done; cp $R/grid.out $G/
git checkout -b gre-baselines-$(date +%F) && git add $G && git commit -m "GRE baselines $(date +%F)" && git push -u origin HEAD
```

## 5b. SPLICE-H (splice, splice_thp)

SPLICE-H is the read-only index of `experimental/scaleli/include/splice/` (design, evidence and kill criteria:
`experimental/scaleli/docs/SPLICE_DESIGN.md`). In GRE it is the index `splice`, plus `splice_thp` (the same layout
with `MADV_HUGEPAGE` on its arena). `scaleli_splice` is SCALE-LI's joint-root cell Jg0 from section 5, **not** this
index. Each dataset's cell (tier-1 knots, eps, entry geometry, slack) was chosen offline on the Mac by `splice_count`
over all 200M keys and is committed as `experimental/scaleli/results/splice_h/args/<ds>.args` (`<ds>.compact.args`
for the compact arm, absent for books, covid and osm, where no cell meets 16.5 B/key: `--splice-arm compact` refuses
those datasets); `gre_run.sh --splice-plan` hands it to the build, which then reproduces that layout with one
PLA pass (its `splice_layout_hash` must equal the one in `results/splice_h/<ds>.json`). Without the args files,
`--splice-plan none` builds with the defaults and the selection runs inside bulk_load (valid, but a longer build).

The protocol follows AIDB (`results/splice_h/design_evidence/warmup_verified.json`): 20M untimed lookups then 100M
timed, `OMP_NUM_THREADS=1` (gre_run.sh sets it; GRE's sampler depends on the OpenMP team size), one pinned core, a
warm headline plus a cold W=0 companion with the same seed, at least 3 interleaved rounds, and a within-process
check of the first-chunk transient. Every command below is run from the SPLICE checkout (`~/SPLICE`, i.e.
`/tmp/louisvasseur/SPLICE` on DIAS).

0) Machine state, once, before anything else:

```bash
R=~/splice_results; mkdir -p $R
{ date -u; cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag; grep -E 'AnonHugePages|HugePages_Total' /proc/meminfo; lscpu -e=CPU,CORE,SOCKET,NODE,CACHE; cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor; echo MALLOC_CONF=${MALLOC_CONF-}; } > $R/machine.txt
```

Pin to core 2. The CPU that shares core 2's L2 (the same L2 id in the CACHE column of `lscpu -e`) must stay idle:
Goldmont shares 2 MB of L2 per module, and the cost model prices exactly that L2. Use the performance governor.

THP decides what step 3 measures. With `enabled` = `[madvise]` or `[never]`, step 3 is the **4 KiB arm** (splice
never asks for huge pages; jemalloc's default does not either). With `[always]`, every anonymous mapping, the plain
`splice` arena included (2 MiB aligned), can get huge pages: step 3 is then a THP run for every index, there is no
4 KiB arm, and the 4 KiB kill rule below cannot be applied; name the OUTDIRs `warm_thpalways`/`cold_thpalways`
instead of `warm`/`cold` and ask for `madvise` before drawing the 4 KiB conclusion. gre_report.py warns when a
`splice` (not `splice_thp`) log reports `splice_arena_anon_huge_bytes` above 0.

1) Build (inside the activated conda environment, while no timed run is active). gre_lite.sh now builds `splice`,
`splice_thp`, `trace_splice`, `trace_lipp` and `nullindex` (an O(1) get() that touches no memory: the harness
baseline of step 8) next to the existing names; it stops with a CMake error if the checkout has no
`experimental/scaleli/include/splice/layout.hpp`. Then check the read path as compiled by this GCC (no call, division,
x87, FMA, BSF/BSR/LZCNT/TZCNT, indirect jump other than the segment-kind jump table, and no store except the
out-parameter and the stack):

```bash
experimental/scaleli/tools/gre_lite.sh
grep -E 'splice' /tmp/louisvasseur/GRE/build/gre_lite_build.txt   # cksums of the copied SPLICE files and headers
experimental/scaleli/tools/splice_asm_check.sh g++ > $R/asm_check.txt; tail -3 $R/asm_check.txt   # must end in PASS
```

2) Smoke test on a 10M fb prefix, with build defaults and no per-dataset plan:

```bash
cd /tmp/louisvasseur/GRE && FB=/tmp/louisvasseur/SPLICE/experimental/scaleli/data/external/gre/fb; for idx in splice splice_thp trace_splice; do SPLICE_ARGS= OMP_NUM_THREADS=1 ./build/microbench --keys_file=$FB --keys_file_type=binary --read=1 --insert=0 --operations_num=1000000 --table_size=10000000 --init_table_ratio=1 --thread_num=1 --memory --index=$idx --output_path=smoke_splice.csv > smoke_$idx.log 2>&1; echo "$idx: $(grep -E '^(Throughput|success_read|splice_total_bytes|splice_layout_hash|trace_chunks_n)' smoke_$idx.log | tr '\n' ' ')"; done
cd ~/SPLICE
```

Every line must show `success_read: 1000000`. A `splice_error:` line in a log names the problem (bad `SPLICE_ARGS`,
`--thread_num` other than 1, unsorted input); the process then exits with status 2.

3) Headline grid, warm (W=20M, pinned) and its cold companion (W=0, pinned), same seed, conditions alternating every
round:

```bash
cd ~/SPLICE; R=~/splice_results; mkdir -p $R; D=fb,osm,books,covid,genome,history,libio,planet,stack,wise; I=splice,pgm,lipp,sortedarray,alex,btree,artunsync; P=experimental/scaleli/results/splice_h/args; export R D I P
nohup bash -c 'for k in 1 2 3; do for j in 0 1; do case $(( (k + j) % 2 )) in 0) c=warm a="--warmup 20000000 --pin 2";; 1) c=cold a="--warmup 0 --pin 2";; esac; rc=0; experimental/scaleli/tools/gre_run.sh $R/$c --datasets $D --indexes $I $a --repeats $k --resume --splice-plan $P >> $R/$c.out 2>&1 || rc=$?; [ $rc != 2 ] || { echo "gre_run.sh refused, see $R/$c.out"; exit 2; }; done; done' > $R/grid.out 2>&1 &
```

Use 5 rounds (`k in 1 2 3 4 5`) if time allows; `--resume` skips what is done, so the same block with a larger range
continues the grid. gre_run.sh refuses to start (exit 2) if any `<ds>.args` file is missing or malformed, and checks
every splice log for `splice_index:` and the absence of `splice_error`. `D=fb,osm` with `k in 1` gives a first pass:
time it before launching all ten datasets.

4) THP arm, only if `/sys/kernel/mm/transparent_hugepage/enabled` shows `[madvise]` or `[always]`. It covers every
index of the arm (jemalloc's `thp:always` for the others, `MADV_HUGEPAGE` for splice_thp). Run it after the grid of
step 3, in the same shell (R, D and P exported):

```bash
cd ~/SPLICE; export MALLOC_CONF=thp:always; I=splice_thp,pgm,lipp,sortedarray; export I
nohup bash -c 'for k in 1 2 3; do c=thp_warm a="--warmup 20000000 --pin 2"; rc=0; experimental/scaleli/tools/gre_run.sh $R/$c --datasets $D --indexes $I $a --repeats $k --resume --splice-plan $P >> $R/$c.out 2>&1 || rc=$?; [ $rc != 2 ] || { echo "gre_run.sh refused, see $R/$c.out"; exit 2; }; done' > $R/thp.out 2>&1 &
unset MALLOC_CONF
grep -h splice_arena_anon_huge_bytes $R/thp_warm/logs/*__splice_thp__*.log; grep AnonHugePages /proc/meminfo
```

Record `splice_arena_anon_huge_bytes` (gre_report.py prints its share of the arena) and `AnonHugePages`.

5) Transient check (within process, paired; the trace_* indexes time every 1M get() calls and are never headline
numbers):

```bash
cd ~/SPLICE; for k in 1 2 3; do for W in 0 1000000 20000000; do experimental/scaleli/tools/gre_run.sh $R/transient_w$W --datasets fb,osm --indexes trace_splice,trace_lipp --warmup $W --pin 2 --repeats $k --resume --splice-plan $P >> $R/transient.out 2>&1; done; done
python3 experimental/scaleli/tools/gre_transient.py $R/transient_w0 $R/transient_w1000000 $R/transient_w20000000 > $R/transient.md
```

ACCEPT means: at W=20M the first timed chunk is within max(2 x chunk spread, 1%) of the run's median chunk, and at
W=0 the transient lasts at most 2 chunks (2M lookups). INVESTIGATE (THP collapse, frequency, page faults) rather than
enlarging the warm-up.

6) Reports:

```bash
for c in warm cold; do python3 experimental/scaleli/tools/gre_report.py $R/$c > $R/$c/report.md; python3 experimental/scaleli/tools/gre_report.py $R/$c --ref pgm --splice-json experimental/scaleli/results/splice_h > $R/$c/report_vs_pgm.md; done
```

Besides section 5's tables, every dataset with splice runs gets a SPLICE check: index RSS against the index's own
`splice_total_bytes` (OK within 3%, and 18-25 B/key unless the cell is compact), predicted E[D], and layout hashes
that must agree between repeats, between splice and splice_thp, and (with `--splice-json`) with the Mac's
`splice_count` result. Its findings are listed under Warnings.

7) Ablations (design section 8, row 10), measured like the headline (warm, pinned, paired with the same rounds).
The plan file fixes the cell; a global `SPLICE_ARGS` is appended after it and wins. `eps=0` makes bulk_load
bisect eps again where the ablation changes the fit (Hist-Tree knots, K1, exceptions); the others keep the plan's
eps. One OUTDIR per ablation, so `--resume` and the recorded `splice_args` never mix two of them:

```bash
cd ~/SPLICE; DA=fb,osm,books,stack; export DA
nohup bash -c 'for k in 1 2 3; do for ab in "histtree:tier1=histtree eps=0" "direct:mode=direct" "fenced:mode=fenced" "k16k:k1=16384 eps=0" "exc0:exc=0 eps=0"; do n=${ab%%:*}; d=$DA; [ $n = direct ] && d=books,stack; SPLICE_ARGS="${ab#*:}" experimental/scaleli/tools/gre_run.sh $R/abl_$n --datasets $d --indexes splice --warmup 20000000 --pin 2 --repeats $k --resume --splice-plan $P >> $R/abl.out 2>&1; done; done' > $R/abl_loop.out 2>&1 &
for n in histtree direct fenced k16k exc0; do python3 experimental/scaleli/tools/gre_report.py $R/abl_$n > $R/abl_$n/report.md; done
```

Compare each ablation's splice throughput with the warm grid's splice on the same dataset (step 3, same rounds).
The Mac counts already put the Hist-Tree knots within 0.01 lines of the learned knots on most datasets
(`results/splice_h/COUNTS.md`); a GRE difference inside the round-to-round spread confirms that the "learned
knots" claim is dead. `mode=direct` runs on books and stack only: on fb and osm the cap-bound DIRECT-only layout
costs about 160-250 D per lookup in the model (hours per run), and the counts already rule it out. Not implemented: the "static PGM with a parallel
last-mile window" ablation of the design.

8) Counter check (cost model, design section 8) on fb and stack. List the events first, then choose an L2-miss load
event and a D-side page-walk event (on Goldmont, for example `MEM_LOAD_UOPS_RETIRED.L2_MISS` and
`PAGE_WALKS.D_SIDE_COUNT` or `MEM_UOPS_RETIRED.DTLB_MISS_LOADS`; check every name against perf_events.txt). perf
counts the whole process, and GRE draws its 100M lookup keys by random reads of the 1.6 GB key array before the
timed loop (about one L2 miss and one walk per op), so the harness is subtracted with `nullindex`, whose get()
touches no memory, at the same op counts; the build is subtracted with a 1-op run of each:

```bash
perf list | grep -i -E 'walk|dtlb|l2_miss' > $R/perf_events.txt
EV=<l2-miss event>,<page-walk event>; MB=/tmp/louisvasseur/GRE/build/microbench; DATA=/tmp/louisvasseur/SPLICE/experimental/scaleli/data/external/gre
for ds in fb stack; do for idx in splice nullindex; do for ops in 100000000 1; do SPLICE_ARGS="$(cat $P/$ds.args)" SPLICE_BUILD_THREADS=16 OMP_NUM_THREADS=1 perf stat -x, -o $R/perf_${ds}_${idx}_$ops.csv -e cycles,instructions,$EV $MB --keys_file=$DATA/$ds --keys_file_type=binary --read=1 --insert=0 --operations_num=$ops --table_size=-1 --init_table_ratio=1 --thread_num=1 --memory --index=$idx --pin_core=2 --output_path=$R/perf_out.csv > $R/perf_${ds}_${idx}_$ops.log 2>&1; done; done; done
grep -h '^splice_stats' $R/perf_fb_splice_100000000.log $R/perf_stack_splice_100000000.log > $R/perf_expected.txt
```

Per lookup = [(splice at 100M - splice at 1) - (nullindex at 100M - nullindex at 1)] / 1e8. Expected, from the
`splice_stats` line of the same log (all per lookup, milli): L2-miss loads = (`dep_lines_milli` +
`par_lines_milli`)/1000 + the simulated `record_miss_milli` and `router_miss_milli`/1000 (the second line of a
2-line entry and the extra lines of a DIRECT window are real loads that miss); walks = `pages4k_milli`/1000. The
Mac's values for the committed cells are in `results/splice_h/COUNTS.md` ("Counter check" table). A deviation of
more than 0.3 per lookup in either count invalidates the count model; within it, the D weighting (parallel lines at
0.05 D, walks at their simulated miss probability) is what steps 3-5 test.

Decision rules (`docs/SPLICE_DESIGN.md` section 8):

- The headline is the warm 4 KiB arm; the cold arm is the GRE-as-published comparison.
- Adopt SPLICE-H only if the lower 95% CI bound of splice/pgm on fb (report_vs_pgm.md) is above 1.15.
- Kill the read-only claim if fb measures below 1.35 Mops/s on 4 KiB pages.
- Report the margins on the other datasets only after PGM and LIPP have run there, in the same rounds.
- Push results under `experimental/scaleli/results/splice_<date>/` without `bin/`:

```bash
G=experimental/scaleli/results/splice_$(date +%F); mkdir -p $G
for c in warm cold thp_warm transient_w0 transient_w1000000 transient_w20000000 abl_histtree abl_direct abl_fenced abl_k16k abl_exc0; do [ -d $R/$c ] && rsync -a --exclude bin/ $R/$c/ $G/$c/; done
cp $R/*.out $R/*.md $R/machine.txt $R/perf_* $G/ 2>/dev/null; git checkout -b splice-$(date +%F) && git add $G && git commit -m "SPLICE-H GRE runs $(date +%F)" && git push -u origin HEAD
```

Not implemented in this change: the rest of the calibration kit of the design (a pointer chase over 4 GB on 4 KiB
pages and on THP for D and the walk cost; k adjacent independent lines, k = 1..10; an lfence A/B on the get() loop)
and the static-PGM-with-window ablation. The O(1) harness index exists (`nullindex`): add it to step 3's index list
for one round to measure the harness residual. Until the kit exists, D, the walk costs and the overlap credit stay
estimates.

## Paths

Every script finds the repository root by walking up to `walkthrough.py` (override with `SPLICE_ROOT`), so it
runs from any clone and any working directory. Pipeline configs resolve paths from `experimental/scaleli/`.
Recorded outputs (`provenance.json`, `*.command.json`, `verdicts.json`, logs) still show the Mac paths they were
produced with; they are records, not inputs.

## Known issues

- `MANIFEST.sha256` covers the original walkthrough package only; `python3 walkthrough.py check` verifies it.
- The prefault sweep does not walk `Config::flow` weights (a few hundred bytes).
- `steady_state_ratio` cannot detect a cold start; use first span / median of the rest
  (`experimental/scaleli/docs/METRICS.md`).
- The X/P timing layers of the protocol need benchmark options that are not implemented yet (validate.py TIM-3);
  night 0 and the D layer run on the current binary.
- All 200M timings so far ran on a busy Mac that was probably in Low Power Mode, with a VM running beside them.
  Treat them as provisional; this server is where they get redone.

## Relaying results

Commit server outputs under `experimental/scaleli/results/` (one folder per run, e.g. `server_2026-10-06/`)
on a branch, and push. Never commit the datasets or `build-*` (`.gitignore` already excludes them).
