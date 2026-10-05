
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
