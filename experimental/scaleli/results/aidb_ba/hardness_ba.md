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

| dataset | scope | n (seq.) | RMSE | RMSE/n | ME | ME/n | CD | PLA-32 | PLA-4096 | extra |
|---|---|---|---|---|---|---|---|---|---|---|
| osm | raw | 200.0M | 24.18M | 12.09% | 67.38M | 33.7% | 4,107 | 661,115 | 5,495 |  |
| osm | free | 200.0M | 8.76M | 4.38% | 73.99M | 37.0% | 353 | 366,852 | 2,192 | descents=363, z-dups=2719, CD(lit 1e-6)=2446 |
| osm | untr | 200.0M | 24.65M | 12.32% | 72.93M | 36.5% | 968 | 490,374 | 3,343 | descents=359, z-dups=2729, CD(lit 1e-6)=12130 |
| osm | mono | 200.0M | 24.14M | 12.07% | 66.03M | 33.0% | 3,944 | 661,115 | 5,495 | descents=0, z-dups=3454, CD(lit 1e-6)=135478 |
| osm | csv | 210.2M | 25.07M | 11.93% | 70.48M | 33.5% | 4,045 | 710,009 | 5,678 | VP=10,226,156 (5.1% of n), region RMSE 519.0->403.6, smoothing 186s, CD(lit)=4081 |
| planet | raw | 200.0M | 29.77M | 14.89% | 60.46M | 30.2% | 21 | 613,597 | 2,314 |  |
| planet | free | 200.0M | 11.31M | 5.66% | 74.48M | 37.2% | 10 | 133,405 | 423 | descents=513, z-dups=0, CD(lit 1e-6)=178 |
| planet | untr | 200.0M | 22.18M | 11.09% | 39.91M | 20.0% | 15 | 274,731 | 866 | descents=513, z-dups=2, CD(lit 1e-6)=805 |
| planet | mono | 200.0M | 16.04M | 8.02% | 32.83M | 16.4% | 29 | 613,601 | 2,314 | descents=0, z-dups=0, CD(lit 1e-6)=2750 |
| planet | csv | 219.4M | 32.69M | 14.90% | 66.36M | 30.2% | 400 | 635,234 | 2,486 | VP=19,399,512 (9.7% of n), region RMSE 217.1->174.8, smoothing 370s, CD(lit)=400 |
| fb | raw | 200.0M | 57.74M | 28.87% | 100.00M | 50.0% | 110 | 1,055,308 | 1,687 |  |
| fb | free | 200.0M | 7.31M | 3.65% | 470.03M | 235.0% | 21 | 196,105 | 494 | descents=514, z-dups=20, CD(lit 1e-6)=197 |
| fb | untr | 200.0M | 15.85M | 7.93% | 39.28M | 19.6% | 40 | 441,605 | 1,038 | descents=513, z-dups=24, CD(lit 1e-6)=1019 |
| fb | mono | 200.0M | 473.5k | 0.24% | 1136.16M | 568.1% | 101 | 1,055,310 | 1,687 | descents=0, z-dups=20, CD(lit 1e-6)=1831 |
| fb | csv | 220.0M | 63.50M | 28.87% | 109.98M | 50.0% | 2,454 | 1,157,933 | 1,889 | VP=19,970,020 (10.0% of n), region RMSE 318.1->264.6, smoothing 301s, CD(lit)=322 |
| books | raw | 200.0M | 18.05M | 9.03% | 96.38M | 48.2% | 246 | 262,604 | 97 |  |
| books | free | 200.0M | 9.93M | 4.96% | 103.12M | 51.6% | 245 | 123,658 | 101 | descents=65, z-dups=357, CD(lit 1e-6)=250 |
| books | untr | 200.0M | 6.06M | 3.03% | 24.03M | 12.0% | 245 | 191,256 | 143 | descents=65, z-dups=303, CD(lit 1e-6)=323 |
| books | mono | 200.0M | 17.55M | 8.77% | 91.90M | 45.9% | 245 | 262,604 | 95 | descents=0, z-dups=271, CD(lit 1e-6)=381 |
| books | csv | 215.7M | 19.03M | 8.82% | 102.77M | 47.7% | 245 | 212,463 | 114 | VP=15,658,712 (7.8% of n), region RMSE 44.5->19.4, smoothing 441s, CD(lit)=246 |
| genome | raw | 200.0M | 7.53M | 3.77% | 19.62M | 9.8% | 585 | 1,290,208 | 1,426 |  |
| genome | free | 200.0M | 7.47M | 3.74% | 45.45M | 22.7% | 52 | 880,334 | 263 | descents=65, z-dups=0, CD(lit 1e-6)=1047 |
| genome | untr | 200.0M | 23.40M | 11.70% | 54.07M | 27.0% | 554 | 1,307,686 | 710 | descents=65, z-dups=4, CD(lit 1e-6)=7852 |
| genome | mono | 200.0M | 7.67M | 3.84% | 19.92M | 10.0% | 578 | 1,290,209 | 1,426 | descents=0, z-dups=0, CD(lit 1e-6)=10183 |
| genome | csv | 218.1M | 7.95M | 3.65% | 20.58M | 9.4% | 535 | 1,403,459 | 1,508 | VP=18,081,013 (9.0% of n), region RMSE 90.6->52.5, smoothing 207s, CD(lit)=547 |
| covid | raw | 200.0M | 1.80M | 0.90% | 8.13M | 4.1% | 27 | 81,908 | 850 |  |
| covid | free | 200.0M | 7.37M | 3.69% | 43.31M | 21.7% | 11 | 55,194 | 135 | descents=65, z-dups=61553, CD(lit 1e-6)=128 |
| covid | untr | 200.0M | 14.09M | 7.04% | 33.78M | 16.9% | 27 | 60,800 | 340 | descents=65, z-dups=61554, CD(lit 1e-6)=303 |
| covid | mono | 200.0M | 1.67M | 0.84% | 7.57M | 3.8% | 27 | 81,908 | 850 | descents=0, z-dups=61553, CD(lit 1e-6)=338 |
| covid | csv | 209.6M | 2.16M | 1.03% | 8.83M | 4.2% | 138 | 62,690 | 863 | VP=9,564,692 (4.8% of n), region RMSE 42.7->20.8, smoothing 244s, CD(lit)=138 |
| history | raw | 200.0M | 815.5k | 0.41% | 2.30M | 1.2% | 8 | 105,468 | 468 |  |
| history | free | 200.0M | 7.43M | 3.72% | 40.25M | 20.1% | 9 | 60,902 | 178 | descents=65, z-dups=0, CD(lit 1e-6)=123 |
| history | untr | 200.0M | 14.91M | 7.45% | 36.11M | 18.1% | 13 | 84,532 | 394 | descents=65, z-dups=4, CD(lit 1e-6)=356 |
| history | mono | 200.0M | 657.8k | 0.33% | 2.00M | 1.0% | 8 | 105,469 | 468 | descents=0, z-dups=0, CD(lit 1e-6)=177 |
| history | csv | 212.9M | 1.37M | 0.64% | 3.65M | 1.7% | 167 | 68,167 | 462 | VP=12,945,352 (6.5% of n), region RMSE 67.7->46.3, smoothing 227s, CD(lit)=175 |
| libio | raw | 200.0M | 3.45M | 1.72% | 11.83M | 5.9% | 2 | 145,808 | 639 |  |
| libio | free | 200.0M | 7.06M | 3.53% | 41.96M | 21.0% | 4 | 43,643 | 209 | descents=65, z-dups=0, CD(lit 1e-6)=109 |
| libio | untr | 200.0M | 18.89M | 9.44% | 58.13M | 29.1% | 9 | 89,569 | 439 | descents=65, z-dups=6, CD(lit 1e-6)=428 |
| libio | mono | 200.0M | 3.53M | 1.77% | 12.17M | 6.1% | 2 | 145,810 | 639 | descents=0, z-dups=0, CD(lit 1e-6)=184 |
| libio | csv | 212.5M | 3.20M | 1.50% | 11.16M | 5.3% | 392 | 140,221 | 631 | VP=12,476,915 (6.2% of n), region RMSE 100.0->66.3, smoothing 230s, CD(lit)=369 |
| stack | raw | 200.0M | 839.7k | 0.42% | 2.44M | 1.2% | 1 | 17,833 | 133 |  |
| stack | free | 200.0M | 7.40M | 3.70% | 38.47M | 19.2% | 2 | 10,569 | 129 | descents=65, z-dups=0, CD(lit 1e-6)=91 |
| stack | untr | 200.0M | 16.64M | 8.32% | 42.45M | 21.2% | 6 | 14,300 | 186 | descents=65, z-dups=3, CD(lit 1e-6)=314 |
| stack | mono | 200.0M | 977.5k | 0.49% | 2.75M | 1.4% | 1 | 17,833 | 132 | descents=0, z-dups=0, CD(lit 1e-6)=127 |
| wise | raw | 200.0M | 2.20M | 1.10% | 4.99M | 2.5% | 10 | 79,035 | 382 |  |
| wise | free | 200.0M | 7.58M | 3.79% | 41.75M | 20.9% | 10 | 53,112 | 126 | descents=65, z-dups=0, CD(lit 1e-6)=119 |
| wise | untr | 200.0M | 14.67M | 7.33% | 46.39M | 23.2% | 15 | 63,855 | 254 | descents=65, z-dups=4, CD(lit 1e-6)=381 |
| wise | mono | 200.0M | 2.07M | 1.03% | 4.80M | 2.4% | 9 | 79,035 | 382 | descents=0, z-dups=0, CD(lit 1e-6)=347 |
| fb_trim | raw | 200.0M | 140.7k | 0.07% | 324.9k | 0.2% | 113 | 1,055,307 | 1,687 |  |
| fb_trim | free | 200.0M | 7.31M | 3.65% | 37.96M | 19.0% | 11 | 196,104 | 493 | descents=513, z-dups=0, CD(lit 1e-6)=190 |
| fb_trim | mono | 200.0M | 297.7k | 0.15% | 766.0k | 0.4% | 102 | 1,055,309 | 1,687 | descents=0, z-dups=0, CD(lit 1e-6)=1949 |

Not run: **CSV scope on stack and wise** (and on fb_trim). The measured cost of the CSV scope is 186-441 s of
smoothing per dataset on 16 threads (osm 186, genome 207, history 227, libio 230, covid 244, fb 301, planet 370,
books 441), or 201-451 s wall per run. The host load average was 70-180 throughout, from other users' processes, not
from scaleli jobs. Eight CSV runs used the ~50-minute budget, and the chain was stopped before stack. Flow scopes
(free, mono, untr) cover all ten datasets.

## 2. Findings

### NFL, faithful flow (sorted-by-z convention)

**F1. Global error is homogenised, not reduced.** After the faithful flow, RMSE/n lies in a narrow band of
**3.5-5.7%** on all ten datasets (libio 3.53% to planet 5.66%). Before, it spans **0.41% (history) to 28.9% (fb)**.
RMSE therefore falls on osm (0.36x), planet (0.38x), fb (0.13x) and books (0.55x), is flat on genome (0.99x), and
**rises** on covid (4.1x), libio (2.0x), wise (3.5x), stack (8.8x) and history (9.1x). ME gets worse on **all ten**
(1.07x books to 17.5x history). The cause is in the trainer: it maximises likelihood under z ~ N(0,1)
(`tools/train_flow.py:49`), so sorted z is pushed toward a bounded Gaussian shape (z in about [-2.0, 2.6] on the s64
flows). A single line cannot fit a Gaussian-like CDF; a Monte-Carlo pure N(0,1) gives RMSE 6.1% of n. The global
AIDB error after NFL is therefore a property of the flow's target distribution more than of the dataset.

**F2. Local structure moves strongly, the opposite of the 2M result.** PLA-32 falls on **all ten** (0.19x fb, 0.22x
planet, 0.30x libio, 0.47x books, 0.56x osm, 0.58-0.68x the rest). PLA-4096 falls on eight (books 1.04x and stack
0.97x are flat). CD falls sharply where it was high: osm 4,107 → 353, genome 585 → 52, fb 110 → 21, planet 21 → 10,
covid 27 → 11. It rises on the three datasets whose CD was already 1-8 (history 8 → 9, libio 2 → 4, stack 1 → 2).

**F3. About half of that local gain is the permutation, not learning.** The untrained sawtooth (same features,
normalisation and period, initial weights, no training) already cuts PLA-32 to 0.42x on fb, 0.45x on planet, 0.61x on
libio, 0.73-0.81x on the rest, and genome 1.01x. It also cuts osm's CD 4,107 → 968. Measured on a log scale, the
untrained control accounts for **40-75% of the trained flow's PLA-32 reduction on 9/10 datasets** (osm 51%, planet
53%, fb 52%, books 42%, covid 75%, history 40%, libio 40%, stack 42%, wise 54%), and 0% on genome. Sorting by a
sawtooth feature interleaves 64 or 512 "teeth" of the key range, which averages their local CDFs and smooths the
merged sequence whatever the weights are. The trained flow adds the rest. A key-ordered index cannot collect either
part. This agrees with the selector choosing the flow in 0 of 6,740 regions.

**F4. Monotone flow: inert on local structure, as before.** PLA-32 changes by at most 4 segments (planet +4) and
PLA-4096 by at most 3. Kendall tau of the ME, PLA-32 and PLA-4096 orderings vs BEFORE is 1.000, and 0.956 for CD.
RMSE moves only through a global bend (planet 0.54x) and fb's outlier compression. This reproduces the inert
`full_flow` scope of `results/aidb/hardness.json` (same signs, similar magnitudes; e.g. planet RMSE 0.539x vs 0.522x
there).

**F5. fb's AIDB global error is 21 keys.** fb's bulk ends at 7.73e10. Above it sit 12 keys between 1.4e14 and
1.13e15 and 9 between 2.05e18 and 2^64−1. Dropping these **21 of 200M keys** takes RMSE from 57.7M (n/sqrt(12): the
least-squares line is flat) to **140,694 (0.07% of n, the lowest of all ten)**, and ME from 100M to 324,883. PLA-32
(1,055,307) and CD (113) are unchanged. On trimmed fb the faithful flow makes RMSE **52x worse** (7.31M) and ME 117x
worse; the monotone flow makes them 2.1x and 2.4x worse. Every "NFL cuts fb's RMSE" reading (0.13x here, 0.008x
monotone, 0.02x inert) is outlier clipping. fb's position as hardest by RMSE/ME in the AIDB ordering is an outlier
artefact. PLA and CD are robust to it.

### CSV, a = 0.1 per 4,096-key region (8 of 10 datasets)

**F6. CSV moves its own objective, which no AIDB metric measures.** The region-line RMSE, sqrt(SSE/n) over the
4,096-key regions (the quantity Algorithm 1 minimises and the index uses), falls **17-56%** on all eight: osm 519 →
404, planet 217 → 175, fb 318 → 265, books 44.5 → 19.4, genome 90.6 → 52.5, covid 42.7 → 20.8, history 67.7 → 46.3,
libio 100.0 → 66.3. Virtual points used are 4.8-10.0% of n (the budget is 10%).

**F7. AIDB local metrics after CSV: better on the easy half, worse on the hard half.** PLA-32 **rises** on the four
datasets with the highest raw PLA-32: genome +8.8%, fb +9.7%, osm +7.4%, planet +3.5%. It **falls** on the four
easier ones: books −19%, covid −24%, history −35%, libio −4%. At 2M uniform samples it fell 20-61% on 9/10, so the
2M result does not survive at full density on the hard datasets. This matches the contiguous-window result in
MEETING_NOTES section 1 (local structure preserved → no PLA-32 gain). PLA-4096 rises on 6/8 (+1.5% to +17.5%). CD
with the literal epsilon jumps where it was small: planet 21 → 400, fb 110 → 322, covid 27 → 138, history 8 → 175,
libio 2 → 369. It is flat on osm, books and genome. A plausible mechanism, not verified: virtual points placed
within 0.1% of a gap from a real key (`smoothing.hpp:78-87`) form near-coincident pairs that FMCD puts in one slot.

**F8. Global error after CSV.** After normalising by sequence length, RMSE/n and ME/n are within ±3% on osm, planet,
books and genome (fb is outlier-bound, 1.00x). They **rise** on the two smoothest datasets, history (+58% RMSE, +49%
ME) and covid (+15%), and fall 13% on libio. Virtual-point counts vary from region to region, so the global slot
map is no longer proportional to rank. That costs nothing where the global error is already large, and is visible
where it is tiny. The "+6-10% RMSE everywhere" seen at 2M was mostly the un-normalised slot expansion.

### Ordering of the datasets by hardness (Kendall tau, BEFORE vs AFTER, n/length-normalised RMSE and ME)

| metric | NFL faithful (10) | untrained control (10) | NFL monotone (10) | CSV (8) |
|---|---|---|---|---|
| RMSE | **0.244** | 0.244 | 0.511 | 1.000 |
| ME | 0.867 | **0.022** | 1.000 | 1.000 |
| CD | 0.933 | 0.978 | 0.956 | **0.429** |
| PLA-32 | 0.822 | 0.911 | 1.000 | 1.000 |
| PLA-4096 | 0.822 | 0.867 | 1.000 | 1.000 |

The faithful flow reorders mainly by RMSE. fb drops from hardest to 9th, history rises from easiest to 6th, and
planet becomes the hardest. Under PLA-32, libio drops from 6th to 9th and osm and fb swap. CSV keeps every ordering
except CD, where planet, fb and libio climb because virtual points raise their CD. The monotone flow changes only
the RMSE ordering, through fb's outliers.

## 3. Answer to the question

The 2M claim was "NFL moves global error (RMSE), CSV moves local structure (PLA-32)". At 200M with a faithful
flow, **it does not hold** in either half.
- **NFL (faithful, NFL's own sort-by-z convention)** moves **both** axes. It compresses RMSE into a 3.5-5.7%-of-n
  band set by the flow's Gaussian target (better for four datasets, worse for five, and much worse than raw for
  outlier-free fb). It also cuts PLA-32 by 32-81% and CD by up to 12x. Roughly half of the local gain comes from the
  z-order permutation alone (untrained control). None of it is available to an index that keeps records in key
  order.
- The 2M split came from three things: inert monotone flows (F4), uniform 1% sampling that removes the local
  structure CSV would have to fight (F7), and RMSE compared in un-normalised slot units (F8).
- **CSV** reduces its own region-level error by 17-56% on every dataset. On the AIDB metrics it is neutral to harmful
  on the four hardest datasets by PLA-32 (PLA-32 +3.5% to +9.7%, CD up to 19x). It helps PLA-32 only on the easier
  ones (books, covid, history, libio), and there it worsens global RMSE (history +58%).

**Consequence for the before/after experiment.** The five AIDB metrics are global or near-global properties of a
single sorted sequence. They do not see what either method changes inside SCALE-LI. For NFL they report a
permutation the host cannot use; for CSV they miss the region-level gain. For the meeting I would:
(1) show AIDB BEFORE plus NFL-monotone (the realisable flow) and label NFL-faithful as "NFL's convention, not
realisable in a key-ordered map";
(2) report CSV with region RMSE alongside AIDB, since that is the quantity that maps to fence probes (the 2M result
was 5-24% fewer probes);
(3) report fb with and without its 21 outliers.

## 4. Caveats
- The flows come from the clean-room stand-in trainer (1-D likelihood under N(0,1), 2,000 steps on a 16,384-key
  sample of a 2M uniform sample), not NFL's BNAF. A flow trained on the full 200M, or with the author shift of 1e6,
  could behave differently; F1's band is tied to this trainer's Gaussian target.
- The NFL convention in the tool (sort by z) differs from the host's (z as a feature, key-order records). Global
  AIDB metrics in key-order ranks are undefined for PLA/CD on a non-monotone feature, and the tool does not
  compute RMSE/ME in key order either.
- CSV metrics include virtual points as keys (n grows by 4.8-10%). Normalised and raw values are both in the table.
  The tool's feature (key − global min, as double) is an affine map of the index's region-normalised key: identical
  in exact arithmetic, not bit-identical. The greedy is chaotically sensitive.
- CD on z uses the scaled epsilon; CD for CSV is compared with the literal 1e-6 (both values are in the table). CD
  carries ±1 floor rounding.
- Raw uint64 above 2^53 loses precision in the double features (fb's outliers, osm); stored keys and the raw block
  are exact (__int128).
- One flow per dataset (best.json's pick, s512 or s64). The untrained control uses the trainer's seed-1000000007
  initial weights; another seed would move F3's shares.
- These are deterministic metrics, so the timings above are informational only. The host was heavily loaded (load
  average 70-180).
