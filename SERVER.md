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
