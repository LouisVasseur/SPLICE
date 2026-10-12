# SPLICE

- **SPLICE-H** (`include/splice/`): a read-only learned index whose layout is chosen by exact counts of dependent
  DRAM lines and page walks, not by an error proxy ([docs/SPLICE_DESIGN.md](docs/SPLICE_DESIGN.md)).
- **SCALE-LI** (`include/scaleli/`): a compressed ordered map with NFL- and CSV-style controls and the joint G+T+V
  root ([integrations/gre/README.md](integrations/gre/README.md)).
- Both run inside GRE (gre4index/GRE, VLDB 2022) through `integrations/gre` and `integrations/gre_splice`.
- Counts and correctness come from the Mac; timings come from the DIAS Atom (Intel Atom C3958, Goldmont).

This branch (`clean`) holds only what runs. Branch `main` @ ca8d80e is the archive: walkthrough, slides, papers,
the aidb studies, scaleli_bench and the earlier protocol (`main@ca8d80e:<path>` in the docs points there).

## Layout

| Path | What |
|---|---|
| `include/` | `splice/` (SPLICE-H, header-only) and `scaleli/` (SCALE-LI; `hardness.hpp` holds the LIPP and PGM ports) |
| `src/hardness.cpp` | `scaleli_hardness`: writes the `.sorted` copies; ctest input |
| `integrations/` | GRE facades and interfaces that `tools/gre_lite.sh` copies into a GRE clone |
| `tools/` | GRE: `gre_lite.sh` (build), `gre_run.sh` (runner), `gre_report.py`, `gre_transient.py`, `warmup_detect.py` |
| | Mac: `gre_mac.sh` + [`gre_mac.md`](tools/gre_mac.md), `mac_lane.sh` (benchmark lane) |
| | SPLICE-H offline: `splice_count.cpp` (exact-count simulator), `splice_table.py`, `splice_asm_check.sh`, `splice_xarch.sh` |
| | data: `prepare_data.sh`, `datasets.py`; `check.sh` (cmake + ctest + the Python test) |
| `tests/`, `examples/` | the CMake tests and `test_gre_report_splice.py` |
| `data/` | `external/gre/` (fetched), `samples/` (generated; manifests tracked), `flows/` (NFL weights), the test fixture |
| `results/splice_h/` | the per-dataset plans `gre_run.sh` reads (`args/`), the count results, `COUNTS.md`, `design_evidence/` |
| `results/splice_joint/` | the next G/T/V iteration: diagnoses, hypotheses, `window_suite/`, `prototypes/paged.cpp` |
| `results/gre_mac_2026-10-06/` | reference Mac baselines (previews) |
| `docs/` | `SPLICE_DESIGN.md`, `MATCHING_NFL_CSV.md` |
| `SERVER.md` | the Atom protocol: sections 0, 5, 5b, 5w |

## Coming from main (the Atom checkout)

The datasets sit under `experimental/scaleli/data/` (ignored files, so they survive the checkout). After
`git fetch && git checkout clean`:

```bash
mkdir -p data/external/gre data/samples && mv -n experimental/scaleli/data/external/gre/* data/external/gre/ && { mv -n experimental/scaleli/data/samples/* data/samples/ 2>/dev/null; true; } && find experimental -type f | head
```

Then delete `experimental/` by hand once the `find` lists no data, and the stale `build-fs/`. Not
`mv experimental/scaleli/data/external data/`: `clean` already tracks `data/external/gre/{download.sh,SHA256SUMS}`
and `data/samples/*.manifest.json`, so that fails ("Directory not empty"). The `.sorted` copies move with the rest.
A Mac worktree moves nothing: per-file symlinks into `data/external/gre/` and `data/samples/` (ignored), or `--data`.

## Requirements

- CMake >= 3.16, a C++20 compiler (GCC >= 11 or Clang >= 14), bash, curl.
- Python >= 3.6, standard library only, for `gre_report.py`, `gre_transient.py` and `test_gre_report_splice.py`;
  Python >= 3.8 for `datasets.py` and `prepare_data.sh`; numpy (and matplotlib for `--plot`) for `warmup_detect.py`.
- Atom: an activated conda environment (`CONDA_PREFIX`) and micromamba (`MAMBA_EXE` or
  `/tmp/louisvasseur/bin/micromamba`); `gre_lite.sh` installs `tbb-devel` 2020.2 and `jemalloc` 5.4.0 into it.
- Disk: 16 GB for the raw files, +11.2 GB for the `.sorted` copies (Mac only). 32 GB RAM.

## Data

`data/external/gre/download.sh` (resumable, sha256-checked against `SHA256SUMS`), then on the Mac
`tools/prepare_data.sh` (the `.sorted` copies and the 20 2M-key samples, each checked against its tracked manifest).
Never commit data.

## Atom quickstart

```bash
cd /tmp/louisvasseur/SPLICE                    # conda env activated
tools/gre_lite.sh                              # -> /tmp/louisvasseur/GRE/build/microbench
tools/splice_asm_check.sh g++ | tail -3        # must end in PASS
# smoke tests: SERVER.md 5 and 5b step 2; then, e.g.
nohup tools/gre_run.sh ~/splice_results/warm --datasets fb,osm --indexes splice,pgm,lipp,sortedarray \
    --warmup 20000000 --pin 2 --splice-plan results/splice_h/args > ~/splice_results/warm.out 2>&1 &
python3 tools/gre_report.py ~/splice_results/warm --ref pgm --splice-json results/splice_h > report.md
```

The full protocol (interleaved rounds, cold arm, THP, transient, warm-up phases, ablations, counters) is
[SERVER.md](SERVER.md) sections 5, 5b and 5w.

## Mac quickstart

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
tools/prepare_data.sh build && ctest --test-dir build   # splice_core and splice_gre_check need data/samples
# counts: the splice_count loop of docs/SPLICE_DESIGN.md section 9
GRE_MAC_SEED=<existing GRE clone> tools/gre_mac.sh DEST
tools/mac_lane.sh run -- tools/gre_run.sh OUT --gre DEST --data data/external/gre --pin -1 --datasets fb --indexes splice,pgm
```

Mac numbers are previews ([tools/gre_mac.md](tools/gre_mac.md)); never mix them with Atom numbers.

## Next iteration

[results/splice_joint/README.md](results/splice_joint/README.md) and
[window_suite/WINDOWS.md](results/splice_joint/window_suite/WINDOWS.md). `window_mode.patch` applies with `patch -p1`
from the repository root (cleanly on this branch); rerun `validate.sh` against the current cells before the next
round. The H1 PAGED prototype is `results/splice_joint/prototypes/paged.cpp`.

## Results

OUTDIRs go under `~/` (= `/tmp/louisvasseur` on DIAS, lost on reboot). Copy each back with `rsync --exclude bin/` to
`results/gre_<date>/` or `results/splice_<date>/` on a branch cut from `clean`. Never datasets, GRE/STX/NFL code,
`microbench` or build directories.

## License

MIT ([LICENSE](LICENSE)). Ported code and what is never redistributed: [THIRD_PARTY.md](THIRD_PARTY.md) and
[LICENSES/](LICENSES/).
