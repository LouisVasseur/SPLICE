# Does gap removal (G) change the back-and-forth verdict? (20 samples, k in {1,4,16,64}, lambda = 4n)

**Headline.** No. Once G is added, back-and-forth still does not beat a single pass in any way that matters.
On every one of the 20 samples the best back-and-forth configuration is the one *without* G (k=0). It beats
the best single pass by more than its instability band on only 2 samples (books-u by 0.072, history-u by 0.024),
and both are last week's T<->V effect. On total_charged it loses to CSV alone (V) on 20 of 20. Inside G
configurations, the cycle does give a few robust gains (largest: books-u k=64 G->T->V, 0.537 probes). These come
from a real G<->T coupling, but they only repair a poor round-1 gap choice and never reach the no-G optimum.

## Per-sample table (total_uncharged / total_charged, root probes; floor about 2.0)

Best single pass = min over arm B's fixed orders (baseline, T, V, T->V, G, G->T, T->G, G->V, G->T->V, T->G->V) and
all k. The pool is excluded here and covered in section 3. Back-and-forth (B&F) = best guarded cycle in arm A over
k in {0,1,4,16,64} and both orders; rN = selected round. Gain = gain from rounds >= 2 in that B&F cell. Band = c4
perturbation family; "wp" means the whole pass was perturbed before G (verifier), "V" means V was re-run at the
round-1 state (arm A; whole-pass bands exist only for cells with gain >= 0.01). Last column: best cycled G cell
on total_uncharged, and its distance from the sample's no-G optimum.

| sample | baseline | best single | order | B&F best | cell | B&F - single | gain r>=2 | band | > band | best G cell | vs no-G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| books-u | 11.076 | 2.421 / 9.321 | T->V | 2.349 / 9.249 | k=0 r3 | -0.072 | 0.072 | 0.004 wp | yes | 3.376 (k=1 GTV) | +1.027 |
| books-w | 2.320 | 2.299 / 9.199 | T->V | 2.296 / 9.196 | k=0 r2 | -0.003 | 0.003 | 0.025 V | no | 3.287 (k=1 GTV) | +0.991 |
| fb-u | 2.266 | 2.224 / 2.224 | V | 2.222 / 9.122 | k=0 r2 | -0.002 | 0.002 | 0.057 V | no | 3.248 (k=1 GTV) | +1.026 |
| fb-w | 3.584 | 2.409 / 2.409 | V | 2.412 / 9.312 | k=0 r1 | +0.003 | 0.000 | 0.095 V | no | 3.391 (k=1 TGV) | +0.982 |
| osm-u | 11.668 | 8.181 / 8.181 | V | 8.219 / 15.119 | k=0 r1 | +0.038 | 0.000 | 0.068 V | no | 8.930 (k=1 TGV) | +0.749 |
| osm-w | 10.555 | 3.559 / 3.559 | V | 3.512 / 10.412 | k=0 r3 | -0.047 | 0.137 | 0.165 wp | no | 4.459 (k=1 TGV) | +0.947 |
| covid-u | 4.642 | 2.416 / 9.316 | T->V | 2.416 / 9.316 | k=0 r1 | +0.000 | 0.000 | 0.147 V | no | 3.374 (k=1 TGV) | +0.958 |
| covid-w | 6.151 | 2.424 / 9.324 | T->V | 2.416 / 9.316 | k=0 r2 | -0.008 | 0.008 | 0.149 V | no | 3.399 (k=1 GTV) | +0.984 |
| genome-u | 8.635 | 2.774 / 9.674 | T->V | 2.774 / 9.674 | k=0 r1 | +0.000 | 0.000 | 0.198 V | no | 3.939 (k=1 GTV) | +1.165 |
| genome-w | 7.291 | 2.488 / 2.488 | V | 2.619 / 9.519 | k=0 r1 | +0.131 | 0.000 | 0.276 V | no | 3.573 (k=1 GTV) | +1.085 |
| history-u | 3.781 | 2.274 / 9.174 | T->V | 2.251 / 9.151 | k=0 r2 | -0.024 | 0.024 | 0.021 wp | yes (marginal) | 3.256 (k=1 GTV) | +1.005 |
| history-w | 3.849 | 2.250 / 9.150 | T->V | 2.244 / 9.144 | k=0 r2 | -0.006 | 0.006 | 0.041 V | no | 3.237 (k=1 GTV) | +0.994 |
| libio-u | 6.520 | 2.310 / 9.210 | T->V | 2.310 / 9.210 | k=0 r1 | +0.000 | 0.000 | 0.174 V | no | 3.465 (k=1 GTV) | +1.155 |
| libio-w | 3.624 | 2.409 / 2.409 | V | 2.382 / 9.282 | k=0 r2 | -0.028 | 0.059 | 0.095 wp | no | 3.426 (k=1 TGV) | +1.044 |
| planet-u | 12.595 | 2.836 / 9.736 | T->V | 2.835 / 9.735 | k=0 r2 | -0.001 | 0.001 | 0.202 V | no | 3.829 (k=1 GTV) | +0.994 |
| planet-w | 7.585 | 3.264 / 3.264 | V | 3.275 / 10.175 | k=0 r1 | +0.011 | 0.000 | 0.141 V | no | 4.238 (k=1 GTV) | +0.974 |
| stack-u | 3.316 | 2.314 / 2.314 | V | 2.316 / 9.216 | k=0 r1 | +0.002 | 0.000 | 0.044 V | no | 3.313 (k=1 GTV) | +0.999 |
| stack-w | 6.294 | 2.250 / 2.250 | V | 2.265 / 9.165 | k=0 r2 | +0.015 | 0.001 | 0.078 V | no | 3.328 (k=1 GTV) | +1.078 |
| wise-u | 5.745 | 2.667 / 9.567 | T->V | 2.667 / 9.567 | k=0 r1 | +0.000 | 0.000 | 0.252 V | no | 3.701 (k=1 GTV) | +1.034 |
| wise-w | 2.989 | 2.345 / 2.345 | V | 2.355 / 9.255 | k=0 r1 | +0.010 | 0.000 | 0.009 V | no | 3.343 (k=1 TGV) | +0.998 |

On total_charged the best single pass is V
in all 20 samples, and every B&F carries T's 6.9-probe charge, so B&F loses on charged 20/20. On 10 samples
(fb-u/w, osm-u/w, genome-w, libio-w, planet-w, stack-u/w, wise-w) the best single pass is V alone. Cycling needs
T, so it starts worse there: on osm-u, genome-w, fb-w, planet-w, stack-u/w and wise-w it never recovers.

## 3. Gap removal after its table cost; order; priced pool

- **G pays on no total, anywhere.** Arm A: 0 of 160 cells beat the sample's k=0 best on total_uncharged. Arm B: 0 of
  80 under total_uncharged, total_charged_pwl and total_charged. The nearest miss is osm-u k=1 T->G->V,
  +0.711 over arm A's k=0 best (+0.749 over V alone). Every other sample's best G cell is at least +0.947 over no-G
  (table, last column). G alone (no T, no V) never beats k=0 on any of the 20 samples.
- **Where G does straighten the model (model probes only, table ignored).** Against fixed-order no-G baselines, G lowers
  model probes in 52 of 80 arm-B cells. Largest: osm-u k=64 8.181 -> 4.670 (-3.511, table 7); osm-w k=64 -0.523;
  books-u k=64 G->V vs V -0.474; planet-w k=64 -0.158. All are below their table charge. osm-u is the only
  gap-dominated sample where this is large (top 64 gaps hold 88.7% of the key range). Contrary to the brief's expectation,
  window samples are smoother than uniform: no window gap exceeds 10x the median.
- **Order (G->T->V vs T->G->V).** It does not matter consistently. In arm A, T->G->V has the better round-1 total in
  27 of 80 cells and the better final total in 25 of 80. The mean difference is 0.044 and the max 0.343. In arm B
  (model probes), G->T->V wins 36, G->V 27 and T->G->V 17. The winner flips with k inside a sample (osm-u), and the
  median spread is 0.05. One asymmetry: 5 of the 6 robust cycle gains (next section) are G->T->V. That order picks
  gaps in raw x in round 1, and later rounds re-pick them in the warped coordinate.
- **Priced pool vs best fixed order.** On total_charged the pool (best of pool/poolV) beats the best fixed order on 18
  of 20 samples and ties on osm-u and planet-u, which are budget-bound at 1956 fences. On total_uncharged it wins 13,
  ties 1 and loses 6 (worst: planet-u, +0.581). **None of this comes from G.** The win is poolV's probe-direct V polish
  (books-u -0.368, genome-u -0.174, libio-u -0.133, others 0.002 to 0.097). It is in-sample by construction, and books-u
  sits at the edge of its band (0.361 listed, 0.435 c4). G was bought in 2 of 160 non-pwl pool runs (planet-u poolV,
  k=16 and k=64). Both lose under the true charge: 5.243 becomes 5.864 and 6.699. The 6.9 T was never bought. The
  0.45 piecewise-linear T was bought in 12 of 160 pwl runs.

## 4. Verifier corrections, applied

1. **Arm A's "clean negative" is replaced.** With G, back-and-forth gives gains over the single pass that stay positive under
   all 17 whole-pass perturbations in 9 G cells. In 6 of these the gain also exceeds the whole-pass c4 band
   (gain / min over perturbations / band):
   - books-u k=64 G->T->V: 0.537 / 0.426 / 0.133
   - planet-u k=64 G->T->V: 0.230 / 0.225 / 0.219
   - wise-u k=64 T->G->V: 0.216 / 0.004 / 0.032
   - books-u k=16 G->T->V: 0.094 / 0.066 / 0.051
   - libio-w k=64 G->T->V: 0.044 / 0.037 / 0.006
   - books-u k=4 G->T->V: 0.029 / 0.027 / 0.017

   Positive throughout but inside the band: books-u k=16 T->G->V (0.043 / 0.043 / 0.129), books-u k=1 G->T->V
   (0.019 / 0.019 / 0.020) and fb-u k=16 G->T->V (0.012 / 0.001 / 0.044).

   In the three k=64 G->T->V cells, the gap set changes in all 17 perturbed runs (books-u swaps 53 of 64 gaps).
   The cycle also beats the best of 17 re-rolled single passes: books-u 9.353 vs 9.847, planet-u 9.670 vs 9.901,
   wise-u 9.564 vs 9.755. So the G<->T coupling is real and pays inside G configurations. It undoes a poor raw-x
   round 1: books-u model probes go 2.890 -> 2.353, against the no-G joint result of 2.349. No such cell beats the
   no-G optimum on total.
2. **Arm A's "clears every yardstick" list is corrected.** Dropped, because their minimum gain over perturbations is 0:
   osm-u k=1 G->T->V (positive in 8 of 17), genome-w k=16 T->G->V, fb-w k=1 T->G->V and fb-w k=4 T->G->V. Added as
   robust: libio-w k=64 G->T->V, books-u k=16 T->G->V and books-u k=4 G->T->V.
3. **Chaos null.** The brief's eps=1e-6 band is exactly 0 in 34 of 41 passes that contain T, because the T refit and G's
   renormalisation absorb the perturbation, so 40 of 41 gains would trivially "exceed" it. This report uses the c4
   family (eps in {1e-6, 3.33e-5, 1e-4, 1e-3}, x^2 and x^3, both signs) applied to the whole pass before G. Under it,
   11 of 41 G-cell gains exceed the band. Last week's quoted bands mixed perturbation families.
4. **Charge dependence.** "G never pays on total" holds at 1x and 2x ceil(log2(k_eff+1)), but not at about 0.5x. Arm B's
   break-even factor is at most 0.50 (osm-u k=64: 3.51 saved against 7). In arm A, G beats the k=0 best in 1 cell at
   0.5x, in 7 at 0.25x (all osm-u) and in 96 at 0x (most within noise). The conclusion therefore needs the gap table to
   cost at least about half of ceil(log2(k+1)) root probes. k_eff never changes between round 1 and the best round, so
   the round >= 2 gains do not depend on the charge.
5. **Arm B's model-only framing.** G's model-only savings are reported against the fixed-order no-G baselines: 52 of 80
   cells (section 3), not only against the in-sample poolV (35 of 80).

**Caveats.** Window samples have no last-week reference, so their T, V and T->V numbers are new. Both arm A's round
guard and poolV select on the evaluation metric, so both are in-sample and optimistic. G uses select='current'
throughout. arm A's run without the guard beats the guarded stop in 22 of 180 cells (max 0.236). Fairness was checked:
arm A's round 1 equals arm B's single pass bit for bit in 160 of 160 cells, and the G-off results reproduce
joint_results.json at 0.0 on 40 cells. Monotonicity: 0 violations in 2,040 states.
