# Adversarial verification of arms A (back-and-forth) and B (single pass) with G added

Verdict: **PARTIAL.** The arms are scored fairly, every state is monotone, and "G never pays on total"
holds even with the gap-table cost doubled. However, arm A's headline ("adding G does not change the
verdict on back-and-forth"; a clean negative) is too strong. In a handful of k=64 cells, rounds 2 and later
give gains that survive every perturbation of the whole pass, and in those cells the gap set really changes.
So the G<->T coupling does occur and does work there. It only repairs a poor round-1 gap selection, and the
result never beats the no-G optimum.

Scripts and data are in `verify/`: v_band.py, v_band_report.py, v_band.json, band_runs2/ (765 runs);
v_indep.py, v_indep_{A,B,Bheavy}.json; v_mono.py, v_mono.json; v_consist.py, v_consist.json.
threeblock.py, armA*, armB* and everything under experimental/scaleli were not modified.

## 1. Instability band of the whole single pass

How the band was measured: perturb the raw normalised input x -> x +/- eps*x^2 and x +/- eps*x^3 **before G**.
Then rerun arm A's exact guarded cycle. Round 1 of that cycle is the single pass. A `State` subclass applies the
perturbation; the identity run reproduces armA_runs bit for bit in all 45 cells. This was done for every arm-A
cell with a round>=2 gain of at least 0.01 (41 G cells and 4 k=0 cells), with eps = 1e-6 as the brief specifies
and also with the c4 family (eps in {1e-6, 3.33e-5, 1e-4, 1e-3}, 16 variants plus identity).

- **With eps = 1e-6 as specified, the band is exactly 0 in 34 of 41 G cells.** By that test, 40 of 41 gains
  "exceed the band". The cause: T is refitted by least squares after the perturbation, and G renormalises u, so
  a 1e-6 input-axis curvature is absorbed before V runs. The 1e-6 test is therefore not a useful null for a pass
  that contains T. (Last week's 1e-6 bands were for csv_only, which has no T to absorb the perturbation.)
- **With the c4 family on the whole pass,** the gain exceeds the band in 11 of 41 G cells.
- **Robustness of the gain itself:** the cycle's gain stays positive under all 17 perturbations, with this
  minimum over perturbations:

| cell | gain (id) | min gain over 17 perturbations | whole-pass c4 band | gap set changed in an accepted round (of 17) |
|---|---|---|---|---|
| books-u k=64 G->T->V | 0.537 | 0.426 | 0.133 | 17 |
| planet-u k=64 G->T->V | 0.230 | 0.225 | 0.219 | 17 |
| wise-u k=64 T->G->V | 0.216 | 0.004 | 0.032 | 16 |
| books-u k=16 G->T->V | 0.094 | 0.066 | 0.051 | 0 |
| libio-w k=64 G->T->V | 0.044 | 0.037 | 0.006 | 17 |
| books-u k=16 T->G->V | 0.043 | 0.043 | 0.129 | 17 |
| books-u k=4 G->T->V | 0.029 | 0.027 | 0.017 | 0 |
| books-u k=1 G->T->V | 0.019 | 0.019 | 0.020 | 0 |
| fb-u k=16 G->T->V | 0.012 | 0.001 | 0.044 | 2 |
| (control) books-u k=0 | 0.072 | 0.072 | 0.004 | - |
| (control) history-u k=0 | 0.024 | 0.012 | 0.021 | - |

  The other 30 G cells have a minimum gain of 0 under some perturbation, so they are consistent with greedy
  noise. Examples: osm-u k=1 G->T->V (arm A counts it as clearing every yardstick) has a median gain of 0.000
  and is positive in only 8 of 17 perturbations. genome-w k=16 T->G->V has a median of 0.058 and is positive
  in 10 of 17. fb-w k=1 T->G->V and fb-w k=4 T->G->V each have a minimum of 0.
- **Back-and-forth vs the best re-rolled single pass (min over 17 perturbed single passes):** the cycle is
  better in 19 of 41 G cells. Clear wins: books-u k=64 G->T->V (9.353 vs 9.847), planet-u k=64 G->T->V
  (9.670 vs 9.901) and wise-u k=64 T->G->V (9.564 vs 9.755).

Conclusion for item 1: on books-uniform and planet-uniform at k=64 (and marginally wise-u and libio-w at k=64),
back-and-forth beats the single pass by more than any band. This contradicts the reading that every
round>=2 gain is chaos. The gain undoes a poor round-1 gap choice: model probes end near the no-G joint result
(books 2.353 vs 2.349; planet 2.670 vs 2.835) and do not go below it.

## 2. Fairness

- Both arms call the same `threeblock.score`, with the same budget (alpha=4, lambda=4n) and the same charge,
  ceil(log2(k_eff+1)).
- **Arm A round 1 equals arm B's single pass exactly in all 160 cells** (G->T->V and T->G->V): same model
  probes, same gap set, same virtual count, same charge. Arm A's k=0 round 1 equals arm B's T->V, and equals last
  week's `sequential`, with a difference of 0.0 on all 10 uniform samples.
- **Independent rescoring with verify/indep.py.** Independent code was written for the gap map, the tanh
  evaluation and the factor selection; V uses indep.smooth_cdf, the fit uses indep.LM, and probes use
  indep.root_probes. All matched to the last bit:
  - Arm A: books-u k=64 G->T->V (round 1 2.890480, best 2.353122), fb-w k=1 T->G->V (2.492323 / 2.390993),
    genome-w k=16 T->G->V (2.707267 / 2.513818; its best round kept the previous slots via the guard, so those
    slots were taken from the replay), and planet-u k=64 G->T->V (2.900716 / 2.670420).
  - Arm B: books-u k=64 G->V 2.253838, planet-u k=64 T->G->V 2.846469, osm-w k=64 G->V 3.367451,
    osm-u k=64 G->T->V 4.670420 (1696 virtual fences), and osm-u k=1 G->V 8.014330 (1956 virtual fences).
  - Gap sets, virtual counts and charges all match.
- Asymmetries to note, none of them a scoring error:
  - Arm A's round guard selects on the evaluation metric, and arm B's poolV optimises the probe metric directly.
    Both are in-sample.
  - Arm B's "G helps model-only in 35/80" and "large gains only on osm-uniform" are measured against a no-G
    baseline that includes poolV. Against fixed-order no-G baselines (V, T->V), G lowers model probes in 52 of 80
    cells. The savings include osm-w k=64 0.523, books-u k=64 0.167 and planet-w k=64 0.158; books-u k=64 G->V
    vs V alone is 0.474. Every one of these is still below its table charge.
  - The 35-vs-36 count difference comes from arm B excluding the V-less T->G variant. That is fine.

## 3. Monotonicity

I recomputed f(g(x)) on fences plus floor midpoints (977 probe keys) with independent code for **every stored
arm-A round, 1,560 states** (from the stored gaps, factors and warps), and for **every arm-B fixed-order state
with G, 480 states** (gap sets and factors re-derived independently; all 480 gap sets match the stored ones).
There are 0 violations and no failed rounds. The smallest positive step is 1.4e-6 (osm-uniform). The 2 pool
cells that bought G were not re-derived independently. threeblock asserts monotonicity inside score() for them.

## 4. Gap charge

- No G cell wins in either arm at the prescribed charge, so doubling it cannot create a win: 0/160 (A) and
  0/80 (B) at 1x and at 2x.
- The relevant sensitivity runs the other way, and it matters:
  - Arm B's break-even charge factor is at most 0.50 (osm-u k=64 G->T->V saves 3.51 probes against 7).
  - In arm A, G beats the k=0 best in 1 cell at a 0.5x charge, 7 cells at 0.25x (all osm-uniform) and 96 cells
    at 0x (most of them inside noise).
  - So "G never pays" requires the gap table to cost at least about half of ceil(log2(k+1)) root probes.
- k_eff never changes between round 1 and the best round in any arm-A cell, so the round>=2 gains do not depend
  on the charge.

## 5. Coupling

Recounted from armA_runs, and confirmed: the gap set changed in an accepted round in 16 of 160 cells, in any
candidate round in 45, and in the runs without the guard in 47. books-u k=64 G->T->V swapped 53 of 64 gaps.
The coupling is robust where it matters: in books-u k=64, planet-u k=64 and libio-w k=64 G->T->V the gap set
changes in all 17 perturbed runs. In 5 of the 11 robust-gain cells the set never changes, so the gain is
T<->V coupling working in the G coordinate (as in last week's books k=0).

## Corrections to the arm reports

1. Arm A's "clean negative" should become: back-and-forth with G gives robust gains (larger than any
   whole-pass band, positive under all 17 perturbations) in a few cells, mainly books-u k=64 (0.537) and
   planet-u k=64 (0.230). These gains come with real gap-set changes. They repair a poor raw-x gap selection
   and never beat the no-G optimum.
2. Of arm A's 8 "clear every yardstick" cells, 4 are not robust under the whole-pass perturbation (minimum gain
   0): osm-u k=1 G->T->V, genome-w k=16 T->G->V, fb-w k=1 T->G->V and fb-w k=4 T->G->V. Three cells it did not
   flag are robust: books-u k=16 T->G->V, libio-w k=64 G->T->V and books-u k=4 G->T->V.
3. The brief's 1e-6 band is 0 for 34 of 41 passes that contain T, so it cannot serve as the chaos null for these
   arms. Use the c4 family on the whole pass.
4. "G never pays on total" is robust to a doubled charge but not to halving it (break-even about 0.5x on
   osm-uniform). Say so.
5. Arm B's model-only framing should also be given against fixed-order no-G baselines (52/80; osm-w k=64 -0.52,
   books-u k=64 G->V vs V -0.47), not only against the in-sample poolV.
