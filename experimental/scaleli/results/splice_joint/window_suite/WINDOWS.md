# SPLICE-H window suite (counts only)

This suite measures windows of each dataset, priced as the 200M-key index, so that one SPLICE-H variant costs seconds instead of minutes. Every number here is a count or a model estimate (E[D] in D, 4 KiB pages, 2048 KB L2 arm). None is a timing. The suite was built on 2026-10-06 from the working tree as it was at 07:08; the source copy is `winbase/` and the pristine copy is `winorig/`.

## Files (all in this directory)

| file | what |
|---|---|
| `window_mode.patch` | every source change, as a diff against the copied tree (`patch -p1` from `experimental/scaleli`). It touches `include/splice/params.hpp`, `include/splice/cost.hpp` and `tools/splice_count.cpp`. `layout.hpp` (the read path) is untouched. |
| `run_suite.sh DIR OUT.json` | builds `DIR/build-win/splice_count` (Release, -O3) and runs the joint variant plus G0, T0 and V0 on the 11 windows. It writes one row per window and a summary. |
| `validate.sh DIR OUTDIR` | runs the full-scale chosen cell on the same windows without re-optimization, and prints residuals against `ref_full/<ds>.json`. |
| `ref_full/` | snapshot (07:08) of `results/splice_h/<ds>.json` and `args/`: the validation reference and the default cells. |
| `windows_contiguous.txt` | alternative window list of single contiguous blocks (`WINDOWS_FILE=...`). |
| `winsel/` | window selection evidence: log-gap stats per 12.5M block, a joint scan of all 16 blocks x 10 datasets, and stratified-window trials. |
| `suite_out/` | suite outputs: `winbase_j4.json` (one tree), `x5_*.json` (five trees at once) and `validate_winbase/`. |
| `winstats.cpp`, `pick_windows.py`, `scan_table.py`, `valtab.py` | scratch tools used for the window selection. They are not part of the patch. |

## Commands

```sh
J=/private/tmp/claude-501/-Users-louisvasseur-Downloads-scaleli-sota/ac6d4251-6384-4db6-8aba-a17e1917a2ab/scratchpad/joint
D=/Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/data/external/gre

# A tree for the suite: copy the sources and apply the patch. Anchor the excludes: a bare 'build*'
# also drops include/splice/build.hpp.
rsync -a /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/ $J/T/ --exclude /data --exclude /results --exclude /build --exclude '/build*'
(cd $J/T && patch -p1 < $J/window_mode.patch)

# The suite. Defaults: JOBS=4 processes, THREADS=1, SIM="sim=500000 warm=125000", REOPT=1.
# Cell args come from DIR/results/splice_h/args when present, else from ref_full/args (ARGS_DIR=... overrides).
$J/run_suite.sh $J/T $J/suite_out/T.json

# Validation of the window mode against the full-scale results (chosen cell, no re-optimization).
$J/validate.sh $J/T $J/suite_out/validate_T

# One window by hand: the joint and the three fair ablations, with the J line search and the checks.
$J/T/build-win/splice_count --dataset fb --data $D --window s16,12500000 --scale-as 200000000 \
  --args "$(cat $J/ref_full/args/fb.args) sim=500000 warm=125000" --variants joint,G0,T0,V0 --reopt --verify --parity --rows /dev/stdout
```

`--window OFF,LEN` takes a contiguous window at key index OFF. `--window end,LEN` takes the last LEN keys. `--window sC,LEN` takes C contiguous runs of LEN/C keys, one centred in each n/C stratum, concatenated (C must be a power of 2). Without `--window`, splice_count behaves exactly as before.

## What window mode scales (and why the result matches 200M)

`scale_as=200000000` (a new parameter, printed only when set) prices n window keys as a 200M-key index. The per-lookup counts (lines, pages, scans and fence depth) are local, so they need no change. Four global quantities shrink by n/scale_as:

- **Shared-L2 simulation.** Capacity is 2048 KB x n/scale_as, 16-way. The LRU now uses multiply-high set indexing, which is bit-identical to the old `>> (64 - lg)` for 2^lg sets. Under LRU, a line's hit rate depends on its access rate relative to capacity (Che's approximation). Records, router lines and page-table lines all scale with the key count, so their residency matches full scale. For 12.5M keys the L2 has exactly 128 sets.
- **Router top table.** It uses `router_bits - round(log2(scale_as/n))` bits, so knots per bucket (and child levels) stay as at 200M. Stratified windows span the whole key range at 1/C density, so they add log2(C) bits back.
- **Arena rounding.** The 2 MiB granule becomes 2 MiB x n/scale_as (a multiple of 16 KiB), which keeps bytes/key and the cap as at 200M.
- **Segment budget.** When eps is 0, k1 x n/scale_as is the bisection target. The suite passes eps explicitly from the args file.

Two settings apply only in window mode:

- `sim=500000 warm=125000`. On the 10 s16 windows, E[D] is within 2 milli-D of `sim=3000000 warm=1000000`.
- The reporting arms other than 4K/2048 are skipped (`ed_milli_all[1..3] = -1`). The 4K/2048 arm reuses the selection replay.

The full-scale path is unchanged:

- The 20 sample layout hashes from `--samples data/samples --hash-samples` are identical before and after the patch.
- A full-scale fb `--grid cell` gives layout hash 0x5f8f34fff5b3a481 and E[D] 4521/4649/2880/2880, both equal to `ref_full/fb.json`.
- `splice_tests` (20 samples x 3 modes, parity) and `splice_layout17` pass.

## Window choice

The windows are 12.5M keys (200M/16), not 10M, so that every scaled quantity is an exact power of 2: 128 L2 sets, a 7-bit router (11 for s16) and a 128 KiB granule. The windows are 11:

| window | why |
|---|---|
| `<ds>@s16+12500000` for all 10 datasets | 16 contiguous runs of 781,250 keys, one centred in each 12.5M stratum. Each run is 16-64 segments long, so the 1-64-key clustering and the segment-scale structure survive (not a uniform sample). All runs share one byte budget (one lambda), as at full scale. |
| `fb@end+12500000` | fb's contiguous tail. It holds the 21 top outliers that are fb's exceptions at 200M, so G has something to act on. |

### How the s16 windows were reached

1. **Log-gap picks: 3 of 10 failed.** The first picks were made before any count: for each dataset, the 12.5M block whose log-gap mean, sd, lag-1 autocorrelation and p99 are nearest the whole file's (`winsel/picks.txt`). Validation showed that books, libio and planet are heterogeneous at block scale. A scan of the joint over all 16 blocks of every dataset (`winsel/scan/`) gives these E[D] ranges:

   | dataset | blocks min-max | full scale |
   |---|---|---|
   | books | 2.51-6.10 | 3.304 |
   | libio | 2.37-3.73 | 2.729 |
   | planet | 2.72-4.60 | 4.281 |

   The block means are close to full scale (books 3.67, the others within 0.09 D). The log-gap picks missed books by +2.09, libio by +0.42 and planet by +0.32.

2. **No single contiguous block can represent books.** Its first half is all DIRECT at 20 B/key. Its second half is about 80% FENCED at the cap. At 200M one lambda moves bytes from the easy half to the hard half, and no single block reproduces that (no block is within 0.15 D and 0.05 DIRECT share).

3. **Stratified runs fixed it.** The s16 windows needed one fix: a router top table matched to the full key span (+log2 C bits). Without it every lookup took one extra child level, about +0.08 D. With the fix, all 10 datasets validate under one uniform rule with no per-dataset tuning (table below). s8 failed planet (+0.118 D, -0.063 DIRECT share). s32 also validated.

`windows_contiguous.txt` keeps the single-block alternative for the 9 datasets where a block validates. These blocks were re-picked by count after step 1, so they are not pre-registered. Residuals at sim=3M:

| block | E[D] residual | DIRECT share residual |
|---|---|---|
| fb 50M | +0.000 | |
| osm 0 | +0.122 | |
| covid 50M | -0.090 | |
| genome 112.5M | -0.001 | |
| history 175M | +0.050 | |
| libio 25M | +0.027 | -0.006 |
| planet 75M | -0.065 | -0.041 |
| stack 112.5M | -0.019 | |
| wise 100M | +0.045 | |
| books | none | none |

## Validation (`validate.sh`, chosen full-scale cell, sim=500K)

| window | full E[D] | window E[D] | resid | full DIRECT | window DIRECT | resid | lines full/win | walks full/win | B/key full/win | verify | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fb@s16 | 4.521 | 4.524 | +0.003 | 0.000 | 0.000 | +0.000 | 2.076/2.077 | 2.044/2.045 | 19.28/19.28 | True | True |
| fb@end | 4.521 | 4.521 | +0.000 | 0.000 | 0.000 | +0.000 | 2.076/2.076 | 2.044/2.045 | 19.28/19.28 | True | True |
| osm@s16 | 4.609 | 4.645 | +0.036 | 0.000 | 0.000 | +0.000 | 2.209/2.232 | 2.100/2.117 | 21.06/21.11 | True | True |
| books@s16 | 3.304 | 3.297 | -0.007 | 0.860 | 0.861 | +0.001 | 1.405/1.405 | 1.184/1.184 | 21.96/21.95 | True | True |
| covid@s16 | 2.577 | 2.523 | -0.054 | 0.996 | 0.998 | +0.003 | 1.070/1.071 | 1.025/1.021 | 21.81/21.81 | True | True |
| genome@s16 | 4.434 | 4.434 | +0.000 | 0.000 | 0.000 | +0.000 | 2.000/2.000 | 2.000/2.000 | 18.96/19.04 | True | True |
| history@s16 | 2.589 | 2.576 | -0.013 | 0.989 | 0.988 | -0.001 | 1.078/1.079 | 1.029/1.031 | 21.95/21.86 | True | True |
| libio@s16 | 2.729 | 2.814 | +0.085 | 0.918 | 0.910 | -0.008 | 1.139/1.163 | 1.098/1.108 | 21.95/21.99 | True | True |
| planet@s16 | 4.281 | 4.260 | -0.021 | 0.071 | 0.069 | -0.002 | 2.020/2.016 | 1.968/1.968 | 18.78/18.78 | True | True |
| stack@s16 | 2.334 | 2.329 | -0.005 | 1.000 | 1.000 | +0.000 | 1.000/1.000 | 1.000/1.000 | 21.24/21.00 | True | True |
| wise@s16 | 2.494 | 2.485 | -0.009 | 1.000 | 1.000 | +0.000 | 1.056/1.058 | 1.016/1.016 | 21.88/21.77 | True | True |

Every window is within 0.085 D of full scale, and every DIRECT share within 0.008. In every window, verify passes for all window keys plus 1M absent keys drawn inside the window's key range, and `get_impl<Count>` parity with the plan walker is exact. validate.sh takes 9 s wall (8 jobs).

## Fair ablations (`neutral()` in splice_count.cpp; extend it when G, T or V gain mechanisms)

| variant | neutral setting | re-optimized on J |
|---|---|---|
| joint | the dataset's chosen cell | the J line search below |
| G0 | `exc=0` (G's only mechanism in these sources) | the same search. When the joint kept 0 exceptions and G0 differs only by `exc`, the joint's row is copied (`same_as_joint`, deterministic). |
| T0 | `tier1=equal`: a knot every T keys, one slope per segment, T started at the joint's segment count | the same search over T |
| V0 | `mode=direct alpha=0 w=1`: no slack, no fences, a 1-line window | the same search (here T goes to eps 1-40) |

Every variant is re-optimized in four ways:

- **Per-segment selection.** The selection, lambda and the residency fixed point re-run.
- **J line search over the tier-1 resolution (`--reopt`).** It searches eps (or T) x 2^k on the full J, from k = 0, downhill until a step is worse.
- **Stage-1 screen.** A cheap stage-1 scan over k in [-8, 4] only screens out points more than 5 D above its best: long DIRECT scans take minutes to walk. Stage 1 cannot rank: it omits record misses and misranks small eps by up to 2.7 D (books eps 19: stage 1 says 2.93, the full J says 5.63).
- **Not searched:** V's cell parameters (entry, cbar, compact, alpha set) and the router parameters.

Each row carries:

- `d_X = E[D](X0) - E[D](joint)`
- verify and parity of every variant's chosen layout
- `outer_iters` of each variant

"Earns its place" means the mean d_X over the hard windows (fb s16, fb end, osm, genome, planet) is at least 0.05 D.

## Results on the current sources (`suite_out/winbase_j4.json`)

| window | E[D] joint | B/key | d_G | d_T | d_V | DIRECT | lines | walks | outer iters (J/G0/T0/V0) | joint eps (cell -> J search) | joint vs cell | verify | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| books@s16 | 3.079 | 21.99 | +0.000 | +0.128 | +4.810 | 0.833 | 1.407 | 1.209 | 1/1/1/1 | 154 -> 308 | -0.218 | True | True |
| covid@s16 | 2.474 | 21.82 | +0.000 | +0.115 | +4.555 | 0.998 | 1.070 | 1.023 | 1/1/1/1 | 169 -> 338 | -0.049 | True | True |
| fb@end | 4.514 | 19.27 | +0.394 | +0.000 | +5.786 | 0.000 | 2.079 | 2.046 | 1/1/1/1 | 2560 -> 10240 | -0.007 | True | True |
| fb@s16 | 4.513 | 19.27 | +0.000 | +0.000 | +5.895 | 0.000 | 2.077 | 2.045 | 1/1/1/1 | 2560 -> 20480 | -0.011 | True | True |
| genome@s16 | 4.426 | 19.04 | +0.000 | +0.084 | +10.529 | 0.000 | 2.000 | 2.000 | 1/1/1/1 | 1288 -> 5152 | -0.008 | True | True |
| history@s16 | 2.498 | 21.97 | +0.000 | +0.108 | +4.424 | 0.985 | 1.072 | 1.035 | 1/1/2/1 | 139 -> 278 | -0.078 | True | True |
| libio@s16 | 2.724 | 21.98 | +0.000 | +0.144 | +4.079 | 0.881 | 1.200 | 1.137 | 1/1/1/1 | 298 -> 596 | -0.090 | True | True |
| osm@s16 | 4.605 | 21.16 | +0.000 | +0.119 | +4.541 | 0.000 | 2.325 | 2.184 | 1/1/1/1 | 2704 -> 10816 | -0.040 | True | True |
| planet@s16 | 4.231 | 18.77 | +0.000 | +0.139 | +4.149 | 0.067 | 2.028 | 1.977 | 1/1/1/1 | 2608 -> 5216 | -0.029 | True | True |
| stack@s16 | 2.268 | 20.98 | +0.000 | +0.105 | +4.112 | 1.000 | 1.000 | 1.000 | 1/1/1/1 | 34 -> 272 | -0.061 | True | True |
| wise@s16 | 2.436 | 21.87 | +0.000 | +0.098 | +4.576 | 1.000 | 1.060 | 1.016 | 1/1/1/1 | 94 -> 188 | -0.049 | True | True |

Means over the hard windows:

| component | mean over hard windows | mean over all windows | at least 0.05 D? |
|---|---|---|---|
| G | 0.079 | 0.036 | yes, but only through fb@end |
| T | 0.068 | 0.095 | yes, but 0.000 on both fb windows |
| V | 6.18 | 5.22 | yes |

What these results say about the current sources:

- **The cell's eps is not J-optimal.** It was bisected to a K1 segment count, which is a proxy. Re-training eps on J doubles it on 6 of 11 windows, quadruples it on fb@end, genome and osm, and multiplies it by 8 on fb@s16 and stack. That gains 0.007-0.218 D (books: 0.218).
- **On fb, J is flat in T.** It ranges 4.513-4.524 for eps 2560-40960, and the J optimum is 16 segments per window. Equal-count knots tie the learned ones (d_T = 0.000 on both fb windows), so FENCED fences absorb everything T could do there.
- **G earns its place only in the outlier window.** G is only the end exceptions. It is +0.394 on fb@end and 0 on the 10 interior windows, where the G0 rows are copies of the joint.
- **The outer loop iterates once almost everywhere.** It ran once on 43 of 44 variant runs, and twice on history T0.

## Speed

| run | wall | CPU |
|---|---|---|
| one tree alone (JOBS=4, THREADS=1) | 100-108 s | 330 CPU-s (+8 s cmake build) |
| five trees at once (20 processes, while other sessions ran counts, load 17-44) | 158-171 s per tree | |

The five-tree run is under the 5-minute target. Rows are identical across the five trees and identical to the single run. Most windows take 25-50 CPU-s; fb@end is the longest because of its G0 search.
