# SPLICE

- **SPLICE-H** (`src/splice/`): a read-only learned index whose layout is chosen by exact counts of dependent DRAM
  lines and page walks, not by an error proxy ([docs/DESIGN.md](docs/DESIGN.md)).
- **SCALE-LI** (`src/scaleli/`): a compressed ordered map with NFL- and CSV-style controls and the joint G+T+V root.
- Both run inside GRE (gre4index/GRE, VLDB 2022) through `src/gre/` ([docs/GRE.md](docs/GRE.md)).
- Counts and correctness come from the Mac; timings come from the DIAS Atom (Intel Atom C3958, Goldmont).

Branch `main` @ ca8d80e is the archive (walkthrough, slides, papers, the aidb studies, scaleli_bench, the earlier
protocol); `main@ca8d80e:<path>` in the docs points there.

## Run

```bash
./run.sh setup          # build, datasets (+ .sorted copies and samples on the Mac), GRE
./run.sh test           # ctest + the Python test
./run.sh smoke          # every GRE index on a 10M-key prefix of fb: correctness only
./run.sh bench          # the headline: 20M warm-up + 100M timed lookups, 3 rounds
```

| Command | What it does |
|---|---|
| `setup [build\|data\|gre]...` | CMake build in `build/`; `data/external/gre/download.sh` (skips present files, checks sha256) and, on the Mac, `scripts/prepare_data.sh`; the GRE build. Default: all three. |
| `test` | `scripts/check.sh`: ctest and `src/tests/test_gre_report_splice.py`; without `data/samples` it skips `splice_core` and `splice_gre_check`. |
| `smoke [ARGS]` | `scripts/gre_run.sh` over every index of this GRE build on fb, 10M keys, 1M lookups; a run passes only if `success_read` equals the lookups. Not timed. |
| `bench [OUTDIR] [ARGS]` | `scripts/gre_run.sh` with the headline protocol; default OUTDIR `results/local/bench-<date>` (ignored by git), `--datasets fb,osm`, `--rounds 3`, `--indexes splice,pgm,lipp,sortedarray,alex,btree` (+`artunsync` on the Atom). Any gre_run.sh flag after it wins, e.g. `--warmup 0` for the cold arm. |
| `warmup --tag T [ARGS]` | `scripts/paired_run.sh`: the paired warm-up trace batch, into `results/paired/T/<machine>/`. |
| `analyze OUTDIR...` / `pair A B` | `scripts/warmup_detect.py analyze` / `pair` (numpy; matplotlib for `--plot`). |
| `report OUTDIR [ARGS]` | `scripts/gre_report.py` (e.g. `--ref pgm --splice-json results/splice_h`). |
| `counts [--datasets L] [--out D] [FLAGS]` | `build/splice_count` over all keys of each dataset (default all ten) into `D` (default `results/splice_h/`), then `D/COUNTS.md` (Mac: needs the `.sorted` copies). |
| `lane on\|off\|status\|run -- CMD` | `scripts/mac_lane.sh`: heavy apps to the efficiency cores (Mac only). |

`./run.sh help` prints the table. Long runs: `nohup ./run.sh bench OUT > OUT.out 2>&1 &`. Other tools, run directly:
`scripts/splice_asm_check.sh` (the get() code check), `scripts/splice_xarch.sh` (arm64 vs x86 layout hashes),
`scripts/gre_transient.py`, `scripts/splice_table.py`.

## Mac and Atom

`run.sh` picks the machine from `uname`. GRE lives outside the repository (no licence); set `GRE_DIR=DIR` or pass
`./run.sh --gre DIR COMMAND` to use another clone.

| | Mac (Darwin) | Atom (Linux) |
|---|---|---|
| GRE | `~/gre_mac/GRE`, built by `scripts/gre_mac.sh` (`GRE_MAC_SEED=<clone>` clones offline) | `/tmp/louisvasseur/GRE`, built by `scripts/gre_lite.sh` in the activated conda env |
| timed runs | unpinned, inside `scripts/mac_lane.sh`: previews, never mixed with Atom numbers ([docs/MAC.md](docs/MAC.md)) | pinned to core 2: the reference numbers ([docs/PROTOCOL.md](docs/PROTOCOL.md)) |
| data | raw files plus `.sorted` copies and the 2M samples | raw files only (GRE sorts on load) |

Heavy commands refuse to start beside a timed run (`RUN_FORCE=1`, `GRE_LITE_FORCE=1`, `GRE_RUN_BUSY`,
`PREPARE_DATA_FORCE=1` override). Requirements: CMake >= 3.16, GCC >= 11 or Clang >= 14, bash, curl; Python >= 3.6
(stdlib) for the reports, >= 3.8 for `prepare_data.sh`, numpy for `warmup_detect.py`; 16 GB of disk for the data
(+11.2 GB for the `.sorted` copies), 32 GB RAM.

## Layout

| Path | What |
|---|---|
| `run.sh` | the entry point |
| `src/` | all C++: `splice/`, `scaleli/` (headers, `-I src`), `gre/` (what GRE gets), `tools/` (`splice_count`, `scaleli_hardness`), `tests/` |
| `scripts/` | the shell and Python behind `run.sh` |
| `data/` | `external/gre/` (download script and checksums; datasets fetched, never committed), `samples/` (manifests), `flows/` (NFL weights), the test fixture |
| `results/` | `splice_h/` (the plans `gre_run.sh` reads, `args/`, and the counts), `splice_joint/` (the next G/T/V iteration: `window_suite/`, `prototypes/`), `gre_mac_2026-10-06/` (Mac baselines), `paired/<tag>/<machine>/` |
| `docs/` | [DESIGN](docs/DESIGN.md) (SPLICE-H), [PROTOCOL](docs/PROTOCOL.md) (the Atom), [MAC](docs/MAC.md), [GRE](docs/GRE.md), [METHODS](docs/METHODS.md) (matching NFL and CSV) |

## Results

`bench` and `smoke` write OUTDIRs outside git (`results/local/`, `$TMPDIR`). Keep a finished run with
`rsync -a --exclude bin/ OUT/ results/gre_<date>/` (or `results/splice_<date>/`) and commit it; `warmup` already
writes into `results/paired/<tag>/<machine>/` (commit and push on the Atom, then `./run.sh pair` on the Mac). Never
commit datasets, GRE/STX/NFL code, `microbench` or build directories.

## Atom checkout

Pull only when nothing is running (a running `paired_run.sh` calls `tools/gre_run.sh` every round). From the previous
`clean` layout nothing moves (`data/` keeps its paths): `git pull`, then `./run.sh setup build gre`.
From `main` (data under `experimental/scaleli/data/`, ignored): stash any edits `git status --short` lists, then
`git fetch && git checkout clean && mkdir -p data/external/gre data/samples && mv -n experimental/scaleli/data/external/gre/* data/external/gre/ && { mv -n experimental/scaleli/data/samples/* data/samples/ 2>/dev/null; true; }`,
and delete `experimental/` and `build-fs/` by hand once `find experimental -type f` lists no data
([docs/PROTOCOL.md](docs/PROTOCOL.md) section 0).

## License

MIT ([LICENSE](LICENSE)). Ported code, its licence texts, and what is never redistributed:
[THIRD_PARTY.md](THIRD_PARTY.md).
