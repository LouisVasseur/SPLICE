# Review 1: statistics and inference, PROTOCOL_draft.md

Reviewer 1 (statistics and inference). I read the whole draft (1,165 lines). I also read PLAN.md (sections 4, 7 and 8),
design_measurement.md (sections 0-2, 8-10), critique.md (section 2) and `analyze_ba.py`. Nothing was built or
benchmarked. The only computation was a small numpy Monte Carlo (single thread, `nice 20`, seconds) of the draft's own
analysis rules. Its files are in this folder:

- `power_sim.py` and its output `power_sim_out.md`: the draft's rules against alternatives, at sigma = 4, 5, 6% and on a
  two-state "busy host" mixture, for n = 3 and n = 8;
- `power_sim_families.py` and its output `power_sim_families_out.md`: per-method Holm families, with n = 3, 5 and 8.

**Simulation model.**

- 10 datasets and the 9 T cells; log throughput per run = effect + iid process noise.
- Two noise models:
  - Gaussian sigma;
  - a "mix" model, built to match critique.md 2a: N(0, 3%) plus -20% with probability 0.3, which gives sd 9.6%.
- True effects:
  - Nf: 0.85-0.96, spread over the datasets;
  - C = Cr = +5% on 5 of the 10 datasets (root fences adopted) and 0 elsewhere, which pools to about +2.4%;
  - SV: +30%;
  - N, G, C-vs-Cr, GCr-vs-Cr: 0 in the "base" scenario, -1% in the "nullshift" scenario.
- Margin ±3%, 20,000 replicates.

All power figures below come from this simulation unless they are derived by hand in the text.

---

## 0. Verdict

The design layer is good: outcome-blind invalidation, structural-null exclusion, TOST for predicted nulls, a separate
deterministic layer, and pre-registered fallbacks. The inference layer has two blockers. As written, the overnight
protocol cannot prove what section 1 says it will, even on a quiet host:

1. **The live A/A gate (6.6, item 2) fails by chance** in 31% of nights at sigma = 4%, 60% at 5%, 81% at 6%, and 99% on
   the busy host. When it fails, the protocol shows no throughput ratio at all. The gate that "decides everything" is
   mostly a coin toss.
2. **Holm across all 7 hypotheses, applied to random-effects TOST with t on 9 df at n = 3**, leaves these powers at
   sigma = 4% (the GO case):

   | claim | power |
   |---|---|
   | N ≡ B | 54% |
   | G ≡ B | 53% |
   | C ≡ Cr | 35% |
   | GCr ≡ Cr | 36% |
   | C > B | 20% |

   Section 5.4's "null provable at ±3% if |true| < 1.4%" is the 50%-power boundary, computed without Holm, without the
   smaller Cr baseline, and without heterogeneity between datasets.

The fixes (section 2) cost no machine time:

- the conditional (fixed-set) estimand that the draft already states;
- Holm within each method;
- an A/A gate that does not need TOST;
- datasets stratified by mechanism.

With them, the sigma = 4%, n = 3 powers become:

| claim | power after the fixes |
|---|---|
| N ≡ B | 91% |
| G ≡ B | 88% |
| C ≡ Cr | 69% |
| GCr ≡ Cr | 77% |
| C > B | 71% |
| A/A gate pass, when nothing is wrong | 94% |

Raising the null cells to n = 5, or adopting in-process interleaving, is what makes the overnight claims reliable
(about 90-99%).

---

## 1. Power reality check (the table section 5.4 should contain)

All figures are probabilities of the stated verdict. In the column headers, "eq" means equivalent within ±3%.

**Draft rules: random-effects (RE) pooling, t on 9 df, Holm over the 7 hypotheses of F1, live A/A gate of 6.6.**

| sigma, n | A/A gate passes (true A/A) | C > B | N ≡ B | C ≡ Cr | G ≡ B | GCr ≡ Cr |
|---|---|---|---|---|---|---|
| 4%, n=3 | 0.69 | 0.20 | 0.54 | 0.35 | 0.53 | 0.36 |
| 4%, n=3, nulls really -1% | 0.68 | 0.18 | 0.36 | 0.32 | 0.37 | 0.24 |
| 5%, n=3 | 0.40 | 0.11 | 0.19 | 0.09 | 0.19 | 0.09 |
| 6%, n=3 | 0.19 | 0.08 | 0.06 | 0.03 | 0.06 | 0.02 |
| busy mix (9.6%), n=3 | 0.01 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4%, n=8 (full) | 0.93 | 0.63 | 1.00 | 0.98 | 1.00 | 0.98 |
| 6%, n=8 (full) | 0.78 | 0.28 | 0.70 | 0.52 | 0.70 | 0.52 |

**Recommended rules: conditional equal-weight pooling (FX), Holm within each method, A/A gate per fix B1.**

| sigma, n | A/A gate passes | C > B | Cr > B | N ≡ B | C ≡ Cr | G ≡ B | GCr ≡ Cr |
|---|---|---|---|---|---|---|---|
| 3%, n=3 | 0.94 | 0.94 | 0.94 | 1.00 | 0.96 | 0.99 | 0.97 |
| 4%, n=3 | 0.94 | 0.71 | 0.71 | 0.91 | 0.69 | 0.88 | 0.77 |
| 4%, n=3, nulls really -1% | 0.94 | 0.71 | 0.71 | 0.73 | 0.69 | 0.67 | 0.55 |
| 5%, n=3 | 0.94 | 0.46 | 0.45 | 0.69 | 0.28 | 0.56 | 0.38 |
| 4%, n=5 | 0.94 | 0.93 | 0.93 | 0.99 | 0.95 | 0.99 | 0.96 |
| 5%, n=5 | 0.94 | 0.75 | 0.74 | 0.93 | 0.74 | 0.91 | 0.80 |
| 4%, n=8 | 0.94 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| 6%, n=8 | 0.94 | 0.80 | 0.79 | 0.95 | 0.81 | 0.94 | 0.86 |

**What this means for "prove" in section 1, at the stated budget.**

| hypothesis | overnight, n=3, sigma 4% (draft rules) | with the fixes | needs |
|---|---|---|---|
| H21 A/A | gate passes 69% | 94% | fix B1 |
| H22 N ≡ B | 54% | 91% (73% if the true effect is -1%) | fixes; tighten the prediction (M7) |
| H23 Nf slower | ~100% | ~100% | — |
| H24 C > B pooled | 20% | 71%; ~85-90% stratified (M2) | M1, M2 |
| H25 C ≡ Cr | 35% | 69% | n = 5 for Cr, or interleaving |
| H26a G ≡ B | 53% | 88% | — |
| H26b GCr ≡ Cr | 36% | 77% | n = 5 for Cr and GCr |
| H26c NC ≡ C | full only | full: n = 4 NC vs n = 8 C, ≥ 95% at sigma 4% | — |
| H27 SV faster | ~100% | ~100% | — |
| W (pooled c1) | high (SE ≈ 0.6%) | — | — |
| T: GHz equivalence ±1% | unknown until E1 gives the per-run GHz SD | — | measure it in E1 |
| T: slope equivalence ±0.002 | ≈ 27% at chunk CV 6.5% | — | M6 |
| any equivalence claim at sigma ≥ 6% | ≤ 6% | ≤ 44% | do not attempt overnight |

The honest summary for the supervisor:

- Overnight, on a quiet Mac at sigma ≈ 4% with the fixes, the protocol is about 70-90% likely to prove each predicted
  null and the pooled CSV effect.
- At sigma ≥ 5% it is not, whatever the rules.
- On the busy host only the positive controls (Nf, SV) are provable.

---

## 2. Issues and fixes

### Blockers

**B1. The live A/A gate is a coin toss and it wipes every throughput result (6.6, item 2; 6.9 "live A/A").**

- The gate requires three things: the pooled B2/B 95% CI contains 1, AND TOST at ±3% passes, AND at most 2 of 10
  per-dataset CIs exclude 1.
- B2 vs B has n vs n = 3 vs 3. The pooled RE SE is 0.258·sigma·(t9 = 1.833), so the TOST half-width at sigma = 4% is
  1.89%. That leaves 1.1% of the ±3% margin for an estimate whose SE is 1.03%.
- Simulated pass rates when nothing is wrong: 0.69 at 4%, 0.40 at 5%, 0.19 at 6%, 0.01 on the busy host. The TOST term
  causes nearly all of the failures.

**Fix.**

- (a) Replace the gate with two conditions:
  - the pooled B2/B 95% CI contains 1 (conditional SE, see M1);
  - at most 2 of 10 per-dataset CIs exclude 1 (P(≥ 3 | null) = 1.15%).
  - It passes 94% of the time under a true A/A, and it still catches a systematic B-vs-B2 bias of about 3% or more.
- (b) If you want an equivalence-style A/A, set its margin from its own SE rather than borrowing ±3%:
  - 95% power at θ = 0 needs δ_AA ≥ (z_.95 + z_.975)·SE = 3.6·SE;
  - that is ±3.7% at sigma = 4% (conditional SE 1.03%), or ±4.6% at sigma = 5%.
  - Report it; do not gate on it.
- (c) Add a third gate term: σ̂_live ≤ 8% (the resource rule), so that a host that turned noisy overnight is caught.
- The B2-vs-B CI is the noise-floor band in figure 6 in every case.

**B2. The multiplicity scheme destroys the power of every predicted-null claim (6.5).**

- Holm across 7 hypotheses that mix 3 superiority tests with 4 TOSTs puts the first equivalence test at α/5 or α/4.
  t9 quantiles: 2.821 (one-sided 0.01) and 2.685 (one-sided 0.0125), against 1.833 at 0.05.
- TOST p-values are rarely tiny: an equivalence p is bounded below by how far the estimate sits from the margin. So the
  step-down usually stops at the first equivalence test, and the remaining ones fail with it.
- Result: N ≡ B falls from 85% unadjusted to 54%, and C ≡ Cr from 70% to 35% (RE, sigma 4%, n = 3).

**Fix.**

- Positive controls (H23 Nf, H27 SV) leave the family: they are validity checks like the A/A. If either fails, no
  throughput claim is made; add that row to 6.9.
- One Holm family per method, each at FWER 0.05:
  - NFL = {H22};
  - CSV = {H24 C > B, Cr > B (new, see M3), H25 C ≡ Cr};
  - G = {H26a, H26b}.
- Justification: each method is a separate scientific question that the supervisor asked about, and each slide makes
  one method's claim.
- If one familywise statement over all methods is required, use a serial gatekeeper (SV → Nf at α = 0.05, no α spent)
  followed by Holm over the 6 remaining hypotheses. That costs about 10-25 points of power against per-method
  families (see `power_sim_out.md`, column "gate+Holm-6").

### Major

**M1. The pooling method does not match the stated estimand, and it costs most of the power (6.3).**

- 6.3 defines the estimand as "the average effect over the ten AIDB datasets, each counted once". That is a fixed,
  enumerated set, not a sample from a population of datasets.
- The correct interval for it is the conditional one: θ̂ ± t_df·sqrt(Σ SE_d²)/k, which the draft calls "secondary".
- The RE interval θ̂ ± t_9·sd(Δ_d)/√k answers a different question: the mean effect on a hypothetical new dataset
  family. It pays twice for that:
  - t_9 instead of about 1.97;
  - the between-dataset spread τ.

**Fix.**

- Primary: the conditional, equal-weight interval for "average over these ten datasets", with per-dataset SEs from the
  block-adjusted model (M4).
- Generalisation: report the RE interval and, more usefully, the RE 95% prediction interval
  θ̂ ± t_{k-2}·sqrt(τ̂² + SE²) as "what a new dataset might show".
- Claim wording follows: "averaged over the ten AIDB datasets" (conditional) versus "for datasets like these" (RE/PI).
- For the predicted nulls also report max_d |Δ_d| with Bonferroni 99.5% intervals, because a null of the average can
  hide +5% and -5%.

**M2. A pooled C > B test dilutes a mechanism-concentrated effect (H24).**

- CSV can move throughput only where the root adopts virtual fences (the pilot shows fb adopted, osm fell back).
- With adoption on 5 of 10 datasets at +5%, the equal-weight pooled effect is about 2.4% and τ ≈ 2.6%.
- So the RE test sits at about 40% power even unadjusted, and only reaches about 51% at n = 8. More runs cannot fix
  heterogeneity under RE with k = 10.

**Fix.** The D layer runs before T, so stratify outcome-blind using it.

- Before the T layer starts, write `$P/strata.json` and hash it into the plan header. It holds:
  - the adopting stratum A: datasets where root_virtual > 0 or the root probes change by ≥ 0.5;
  - the fall-back stratum F.
- Pre-register:
  - (i) H24a: C > B within A (conditional pooled). With 5 datasets at +5% and sigma 4%, n = 3, the SE is about 1.3%,
    giving roughly 90% power.
  - (ii) H24b: C ≡ B within F at ±3%.
  - (iii) A mechanism meta-regression, Δ_d = β·p_d + e_d, over every timed (cell, dataset) pair, where
    p_d = predicted Δ log time from the D-layer counters × a pre-registered ns-per-probe range.
    - With p_d known before T, the test β > 0 has one degree of freedom of signal over about 60 contrasts.
    - It is the single most powerful test that "throughput follows work".
    - Also report β with its CI; β ≈ 1 means the counters explain time.

**M3. Cr > B is missing, yet the decomposition depends on it (H25, figure 7).**

- "C's gain comes from the root" needs Cr > B (root effect present) and C ≡ Cr (region virtual points add nothing).
  The draft tests only the second.

**Fix.** Add H25a, Cr > B, to the CSV family (stratum A as in M2).

- Note also that C ≡ Cr at ±3% is uninformative if C's effect is itself only about 3%. Report the share instead:
  (C - Cr)/(C - B), with a Fieller CI.
- Alternatively, set the C ≡ Cr margin to half the observed lower bound of C > B (pre-registered as a rule, not a
  number).

**M4. The noise has persistent host states, so the unpaired estimator is the wrong model (6.2; threats 4 and 8).**

- 6.2 keeps the unpaired estimator because "adjacent runs were measured to be uncorrelated". That evidence is about 5
  pf0/pf1 pairs ("none visible").
- critique.md 2a shows the opposite: three consecutive osm runs (15:11-15:15) all sat in the fast state (chunk CV
  0.02-0.03, 0.87-0.89 Mops), while every other run sat at 0.67-0.78.
- The protocol keeps a dataset's cells contiguous, about 25-35 minutes per (dataset, block) segment. So host state is
  shared within a segment and varies between segments.
- The unpaired analysis puts that segment variance into every contrast. That is conservative but wasteful, and the
  pooled σ̂ then mixes the two.

**Fix.**

- Make S3 primary. Per dataset, fit y = μ_cell + β_block by least squares; the residual SD gives the SE.
- With 9 cells × 3 blocks per dataset, the residual df is 16 per dataset and 160 pooled. This costs about 2 df per
  dataset and removes segment-level state.
- Report the segment ICC from the fit. If ICC > 0.3, also report the unpaired estimate as a sensitivity check.
- Two consequences for the runner:
  - (a) Re-queue an invalid run immediately, inside its segment, not at the end of the block (2.1). A run re-queued at
    the end leaves its block and its host state.
  - (b) If the night ends mid-block, every complete (dataset, block) segment enters the primary analysis. The stopping
    decision is outcome-blind and the segment is the analysis unit. Drop only incomplete segments, rather than the
    "partial block only as sensitivity" rule, which throws away up to a third of the data.

**M5. The strongest variance lever, in-process interleaving, is dropped without a test (5.4, threat 4).**

- design_measurement 8, critique 2a ("option (ii) stops being optional") and PLAN D6 all point at it.
- The draft adds a new binary (P1-P3, 11-14 h) but no interleave mode.
- If the between-process component is host state, per-process contrast noise falls to about 6.5%·sqrt(2/40) ≈ 1.5%.
  Per-dataset ±3% equivalence then becomes affordable overnight.
- It also removes the build-heat confound by construction: B and C are timed in the same process, after the same builds
  and the same cool-down.

**Fix.**

- Add P6, `--interleave cells`, about 3-4 h of engineering:
  - build all cheap cells (B, B2, N, G, Cr, GCr, Nf) in one process, in random order with a per-process seed (layout
    randomisation): about 7 × 2.3 GB + 3.2 GB = 19 GB;
  - then run R rounds, each round visiting the cells in a random order with 1M untimed workload lookups (the 0.75M
    coverage bound, re-warming after the switch) followed by 1M timed lookups.
- Add E1b to calibration (about 20 min): fb, 5 processes, B and B' alternated, 8 rounds.
- Pre-registered decision: adopt interleaving for the T layer if the SD of the per-process mean log(B'/B) is
  ≤ 0.5·√2·σ̂_E1. Otherwise keep the run-level design, because the variance is layout, not host state, and interleaving
  does not cancel it.
- Interleaved numbers are never mixed with run-level numbers in one table.

**M6. The thermal checks are noise-driven decisions (T1 in 5.2; 3.5 "thermal"; 6.7).**

- T1 chooses c* from n = 2 per arm. The "cycles/op of C within 3% of C at 120 s" criterion compares two means of 2
  runs. If cycles/op carries per-process noise like throughput (4-9.5%), the SE of that difference is about sigma
  (4-9.5%), so c* is chosen at random.
- cycles/op is also not frequency-invariant for memory-bound code: DRAM latency is fixed in ns, so cycles/op rises with
  the clock. The criterion therefore partly measures the thing it is meant to exclude.
- The slope TOST (±0.002/chunk) has low power. For one run over chunks 2-16, Σ(i - ī)² = 280, so the slope SE is
  CV/16.7, which is 0.0039 at chunk CV 6.5%. Over 30 runs per cell, the SE of the C-minus-B difference is about 0.001,
  and the TOST passes only when |estimate| < 0.00035, which happens about 27% of the time at a true 0. C/B would be
  labelled "thermally confounded" most nights for no reason. If chunk noise is autocorrelated (critique: steps inside
  runs), it is worse.

**Fix.**

- T1 decides on effective GHz only, n = 3 per arm. Pass at c if |mean GHz(C, c) - mean GHz(B)| < 1% and the per-run
  GHz SD from E1 is ≤ 0.5%. Otherwise fix c* = 60 s a priori.
- Keep cycles/op as descriptive.
- Make GHz the confirmatory thermal check: the C vs B TOST at ±1% over 30 vs 60 runs.
- Demote the slope to a descriptive diagnostic, or set its margin in impact units: a slope difference Δs changes the
  whole-replay mean by about 7.5·Δs, so a ±1% impact means |Δs| ≤ 0.0013. Compute its SE cluster-robustly at run level.
- Add a sensitivity statistic S7: the primary model plus ln(effective GHz) as a covariate. It is reported, never
  substituted.

**M7. The equivalence margin and the predicted ranges are not mutually consistent (1.2, "basis"; 6.4).**

- The ±3% margin is justified as "one flow evaluation or 4-8 root probes ≈ 3% of a lookup". That works out to
  28-66 ns of 1.2-1.7 µs, which is 1.6-5.5%. The positive mechanism and the margin are the same size.
- The pooled C effect is predicted at 1-6%, so by the protocol's own smallest effect of interest (SESOI), most of the
  predicted C range is "not decision-relevant".
- The N prediction (0.98-1.00) disagrees with the protocol's own counter arithmetic:
  - 0.02-0.05 transforms/op × 28-66 ns = 0.6-3.3 ns, which is under 0.3% of a lookup;
  - a true -2% would fail TOST at ±3% about 76% of the time.
- G's prediction is "about 1.00". The counters give:
  - 5 L1 table comparisons, about 1 ns each;
  - ±1.5 root probes × 4-8 ns;
  - so -1.4% to +0.8%.

**Fix.**

- Write the per-dataset predicted Δ from the D-layer counters before T (it feeds M2(iii)).
- Keep δ = 3% as the pre-registered SESOI, with a one-line decision rationale: "an effect below 3% would not change
  which index we recommend".
- Always print the achieved equivalence bound, max(|L90|, |U90|), so a reader can apply their own δ.
- Report C > B together with whether its CI clears the SESOI: "faster, and by more than 3%" versus "faster, by less
  than 3%".

**M8. Overnight followed by full is two looks at the same hypotheses, with no α plan (5.4 "Stopping"; 8.5 full version).**

- The full version re-runs `t_full` into the same `t.jsonl` and re-tests H22-H27 on the superset, after the overnight
  results have been seen and presented.
- The decision to continue depends only on sigma, a nuisance parameter, which is acceptable. The double testing is
  not.

**Fix (pick one and write it into 6.5).**

- (a) **Recommended.** The overnight analysis is the confirmatory test of H21-H27 at α = 0.05. The full version's blocks
  4-8 form an independent confirmatory replication, analysed alone, plus the full-only hypotheses (H26c, per-dataset
  ±5%). The combined estimate is reported descriptively.
- (b) A two-look group-sequential design with Lan-DeMets Pocock-type spending at information fraction 3/8:
  α1 = 0.05·ln(1 + 1.718·0.375) = 0.025 per TOST side at the interim. The final nominal α is about 0.035.
  O'Brien-Fleming spending would give α1 ≈ 0.0014 and kill the overnight claims, so do not use it.

**M9. "Unexplained, not claimed" is a selective-reporting rule (6.4).**

- A CI wholly outside its counter-predicted range is withheld until a mechanism is found. Results that agree with the
  prediction are reported; results that disagree are not.

**Fix.**

- Report every pre-registered estimate. Label disagreement as "inconsistent with the counter prediction (mechanism
  unexplained)".
- Use the M2(iii) meta-regression residual as the formal check, flagging studentized residuals > 3.
- critique.md's refusal of n = 1 pilot ratios is the right instinct. With n ≥ 3 and a valid CI, the defence against a
  spurious result is the CI and the A/A band, not suppression.

### Minor

**m1. The E1 gate is decided on a df-18 point estimate (5.2, 6.6 item 1).**

- P(σ̂ ≤ 5%) is 0.94 at a true 4%, 0.54 at 5%, and 0.18 at 6%.
- E1 is 20 consecutive runs on fb in 30 minutes. If host state persists, it underestimates the overnight sigma, which
  spans hours and ten datasets.
- TOST is self-calibrating: a noisy night simply fails it. So the "5-8%: no equivalence claims" rule discards valid
  evidence without protecting anything.

**Fix.**

- Use E1 only as a go/no-go resource decision: σ̂ > 8% means the timing moves to Linux.
- Let the pre-registered tests run whatever σ̂_live is.
- Print the realised power at σ̂_live next to each null claim.
- Separately, record the per-run GHz SD and on-CPU distribution in E1 (needed by M6) and check E1 for bimodality:
  report the chunk-CV classes as in critique 2a.

**m2. The COLD sensitivity control is underpowered (3.5).**

- With 3 runs, per-run c1 SD about 10%, and a true -21%, P(mean > -15%) = 0.15. So 15% of nights would declare the
  indicator "insensitive" and drop the warm-up claim.

**Fix.** Run COLD in all 3 blocks: 9 runs, P = 0.036. That is 6 more runs at about 90 s, about 9 minutes.
Alternatively, judge the rule on the one-sided 95% upper bound of COLD c1 being below -10%.

**m3. The W1 gate (G4) is noise-dominated.**

- 125k-op chunks have a per-chunk CV of 11-14%, and the W1M arm has only 4 runs, so the SE of mean c1 is about 7-8%.
  P(mean < -5% | true 0) ≈ 0.27, which would trigger W = 8M spuriously.

**Fix.** Keep W = 4M fixed (it is already 16x the observed transient and 5x the coverage bound) and make W1 descriptive.
Or require the rule to hold on the 90% upper bound with n = 4 per dataset per arm.

**m4. One pooled σ̂ across all (dataset, cell) groups assumes equal variance (6.2).**

- osm was noisier than fb, and C cells may carry thermal variance. Per-dataset CIs built from a 180-df pooled σ̂ are
  over-confident where variance is higher.

**Fix.** Use per-dataset residual SDs from the M4 model (df 16 each). Report per-cell σ̂ and a Brown-Forsythe test across
cells. The pooled conditional SE uses the per-dataset values.

**m5. Per-dataset equivalence in the full version needs multiplicity stated (6.4, 6.5).**

- "Equivalent on all ten" is an intersection-union test: each dataset at α, with no correction.
- "Equivalent on these k" needs Bonferroni over the claimed set.
- The 99.5% simultaneous intervals for named per-dataset effects must cover every (cell, dataset) interval on the
  slide, up to 80: 99.94%, not just 10.

**m6. I² at k = 10 and n = 3 is too imprecise to switch wording on (6.3).**

- Report τ̂ with a Q-profile CI instead. Drop the "I² > 50%" wording trigger, because the conditional estimand (M1) is
  an average by definition anyway.

**m7. The pre-registration must be frozen to count.**

- Record sha256 of PROTOCOL.md, analyze_ba.py, run_ba.py and strata.json in the plan header before E1 starts, and copy
  them out of /private/tmp (scratch is volatile).
- Declare the pilot data (seed 11, fullscale) as hypothesis-generating and excluded from every confirmatory analysis.
  The predictions were set after seeing them.

**m8. Carry-over and order.**

- Log the previous job's cell and its build type in `meta`. Add a sensitivity fit with "preceded by a C build" as a
  covariate.
- Check that every block draws a fresh permutation: one shuffle seed must not produce one shared order.
- The B ∪ B2 baseline (2n runs against 6 comparators) is already close to the √m-optimal allocation (√6 ≈ 2.4n). Keep
  it.

**m9. The k < 4 switch to the fixed interval (6.3)** is moot once M1 makes the conditional interval primary. Keep the
structural-null exclusion; state its k per hypothesis in the report.

---

## 3. Recommended replacement text (drop-in for 5.4 "Repeats and power" and 6.2-6.6)

1. **Model.**
   - Per dataset, fit y = ln throughput = μ_cell + β_block + ε, by least squares on the valid T runs (B2 coded as B
     after the A/A).
   - Contrast: Δ_d = μ_X - μ_Y. SE_d comes from the residual SD of dataset d (df 16 at n = 3).
2. **Estimand and pooling.**
   - Primary estimand: the equal-weight average over the datasets in S_X (the structural-null exclusion is kept),
     θ̂ = mean Δ_d, SE = sqrt(Σ SE_d²)/k, t on Σ df.
   - Secondary: RE and its 95% prediction interval, labelled "for a new dataset".
3. **Strata.** Written from the D layer and hashed before T: A (root adopts virtual fences) and F. H24 and Cr > B are
   tested in A; their F-stratum counterparts are equivalence tests.
4. **Hypotheses and families**, each at FWER 0.05 (Holm):
   - NFL {N ≡ B};
   - CSV {C > B (A), Cr > B (A), C ≡ Cr};
   - G {G ≡ B, GCr ≡ Cr}.
   - Validity conditions, outside any family: A/A, Nf < B, SV > B, the COLD deficit, pooled c1, C/B GHz ±1%. If any
     fails, the corresponding 6.9 row applies.
5. **Mechanism test.** The meta-regression of Δ over all timed (cell, dataset) pairs on the pre-written counter
   prediction p_d: H_mech: β > 0, one-sided α = 0.05, reported with its CI.
6. **A/A gate.** The pooled B2/B 95% CI contains 1, at most 2 of 10 per-dataset CIs exclude 1, and σ̂_live ≤ 8%.
7. **Reporting.**
   - For each null: the achieved equivalence bound and the realised power at σ̂_live.
   - For each superiority test: whether it clears the 3% SESOI.
   - Every estimate is reported, with flags; nothing is withheld.
8. **Repeats.**

   | E1 σ̂ | design |
   |---|---|
   | ≤ 4% | n = 3 for every cell (9.6-11.9 h, unchanged) |
   | 4-6% | B, N, G, Cr, GCr at n = 5; B2, Nf, C at n = 3; SV at n = 2. That is +1.8 h, so plan two nights (blocks 1-3 on night 1; blocks 4-5 of the null cells on night 2 as complete segments) |
   | > 6%, or E1b passes | interleaving (M5) |
   | > 8% | Linux |

9. **Full version.** Analysed as an independent replication of blocks 4-8 (M8a), plus the full-only hypotheses.

## 4. What does not need changing

- Outcome-blind invalidation, "flags never drop a run", and the S1-S6 sensitivity list.
- Structural-null exclusion based on deterministic signatures.
- TOST for nulls, and the ban on "trend".
- The separate deterministic layer: exact claims need no statistics, and the draft rightly does not decorate them.
- The 5M ops / 16 chunks choice: the evidence that sigma is per-process supports spending on processes, not ops.
- Pooled c1 with TOST ±3% (SE about 0.6%) as the warm-up validity check.
- The A/A and positive-control logic. Only the gate arithmetic and the families change.
