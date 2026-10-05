# AIDB hardness, BEFORE vs AFTER, NFL and CSV, full 200M keys

Date 2026-10-01. Binary: `/Users/louisvasseur/Downloads/scaleli_sota/build-fs/scaleli_hardness` (built from
`experimental/scaleli/src/hardness.cpp`, not rebuilt). Options on every run: `--dtype uint64 --region-keys 4096
--pla-eps 32,4096 --check-sorted 1 --threads 16`, using `<name>.sorted` for covid genome history libio planet stack wise.
All runs were serial, one at a time; the runner waited until no other scaleli job was running. Raw JSON for each run:
`scratchpad/nfl_ba/out/<d>_<scope>.json` (stderr in `.err`). Combined: `scratchpad/nfl_ba/combined.json`. Scripts:
`run.sh`, `run2.sh`, `tab.py`, `analyze.py`, `mdtable.py`.

**Sanity check.** The BEFORE block matches `results/aidb/hardness.json` scope `full` exactly on all five metrics for all
ten datasets.

## 0. What "after" means in this tool (from the code; this decides how to read every number)

**NFL (`--flow`).** `hardness.cpp` evaluates z = flow(key) for every key, counts adjacent descents
(`unordered_pairs`), and if there are any it **sorts z** (`hardness.cpp:155`, `parallel_sort(transformed)`). It then
computes the metrics on sorted z with target y_i = i. That is **NFL's convention: rank in z order**. AFLI indexes keys
sorted by z. The tool does **not** fit z against key-order ranks. For a monotone flow the two conventions are the
same. For the faithful (non-monotone) flow, "after NFL" describes an index over a **permuted** key order. SCALE-LI
stores records in key order and uses z only as a model feature (`include/scaleli/transform.hpp:14-16`), so these
numbers are **not** what the host's models see. CD on z uses an epsilon of 1e-6 x mean gap on U_T rather than LIPP's
literal 1e-6, because z has an O(1) range. The literal value is in `fmcd.conflict_degree_lipp_epsilon`.

**CSV (`--virtual-alpha`).** `hardness.cpp:166-176` and `hardness.hpp:346-372`. The features are double(key − global
min) (or sorted z when `--flow` is also given; I never combined them here). They are cut into consecutive 4,096-key
regions, and each region gets `smooth_cdf` (`smoothing.hpp`, CSV Algorithm 1): a budget of floor(0.1·4096) = 409
virtual points, greedy insertion with an OLS refit after each one, and an early stop when no gap lowers the
region's SSE. The metrics are then computed on the **augmented sequence**: real and virtual features in feature order,
with target = **global slot index** (the per-region slot spaces concatenated). So n grows by the virtual-point count,
RMSE and ME are in slot units, and CD and PLA count virtual points as if they were keys. The region step is what the
index does: `index.hpp:312` calls the same `smooth_cdf`, on a region-normalised key (`index.hpp:242`). That is an
affine map of the same feature, identical in exact arithmetic but not bit-identical, and the greedy is known to be
chaotically sensitive. The **global** line over concatenated slots is an AIDB construct that has no counterpart
in the index. The virtual-point positions also matter for CD: unless the ternary search finds an interior minimum,
a candidate sits at lo + e or hi − e with e = 1e-3 x gap (`smoothing.hpp:78-87`), i.e. within 0.1% of a gap from a
real key.

**Normalisation used below.** RMSE/n and ME/n use n = sequence length (200M, or 200M + VP for CSV), so the slot-space
expansion does not read as "harder". For CSV vs BEFORE, compare CD with the literal epsilon, which is the same 1e-6
the raw block uses (column "extra"). For flows, the scaled CD is the comparable one.

**Scopes.** `raw` = BEFORE. `free` = AFTER NFL, faithful retrained non-monotone flow
(`results/aidb_flowv2/flows_free/<d>_<cfg>_t2000.txt`, with cfg from `flows_free/best.json`: s512 for osm, planet and fb,
s64 for the rest). `mono` = AFTER NFL with `flows_monotone/<d>_mono.txt`. `csv` = AFTER CSV, a = 0.1 per 4,096-key
region. `untr` is a **new control**: the *untrained* sawtooth flow. It has the trainer's initial weights,
`tools/train_flow.py --steps 0`, with the same normalisation, sawtooth period, sample and seed as the faithful flow
(`scratchpad/nfl_ba/flows_untrained/`). `fb_trim` = fb without its 21 outlier keys (`--limit 199999979`, see F5).

## 1. Results (full 200M keys)

