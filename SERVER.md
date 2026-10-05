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

## 3. What to run

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
- All 200M timings so far ran on a busy Mac that was probably in Low Power Mode, with a VM running beside them.
  Treat them as provisional; this server is where they get redone.

## Relaying results

Commit server outputs under `experimental/scaleli/results/` (one folder per run, e.g. `server_2026-10-06/`)
on a branch, and push. Never commit the datasets or `build-*` (`.gitignore` already excludes them).
