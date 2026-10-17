# NF-transform gate: should a dataset run the flow? (Mac, 2026-10-17)

The question: from a small sample of a dataset, decide quickly whether SCALE-LI should build with the NFL-style flow
(cell N = B + `--flow --flow-bypass 1 --flow-cost 0`) or without it (B).

Sources:
- the exact-count ground truth and the gate scores of `nf_gate`: `nfgate/cases.jsonl`. 90 cases: 10 datasets at full
  200M, plus 8 windows of 12.5M keys each; sample sizes m = 1024, 4096, 16384, 65536. Run at background QoS.
- the timed GRE pair `trace_scaleli_b` / `trace_scaleli_n` on every dataset: `traces/main`, analysed in
  `warmup/warmup.json`.
- `analysis/nfgate_eval.py` turns these into `nfgate_eval/nfgate_eval.{json,md}` and the figures. Every number below
  comes from `nfgate_eval.json` or `nfgate/cases.jsonl`.

## Ground truth and margin

- **D** = 2 root probes + coordinate + fence + key_at + delta probes per lookup. These are the dependent memory loads of
  the read path (`nf_gate.cpp`). **benefit_net** = (D_B - D_N - transform calls) / D_B, one probe-equivalent per
  transform call.
- **The flow helps** when benefit_net > delta and the memory overhead mem_rel < 1%. **delta = 1% of D.**
- Why 1%:
  - (a) benefit_net is never positive here. Over all 90 cases it lies between -1.95e-3 and 0.
  - (b) The raw benefit, without the transform charge, is positive in 17 cases, by at most 1.1e-6. That is 4e-5 probes
    per lookup, a rank-shift effect and not a gain. So delta must be above 0.
  - (c) A single timed run on this Mac cannot resolve less than a few percent. Across the 10 datasets the timed N/B
    steady ratio is +0.5% (mean), with an sd of 2.4% and a range of -2.2% to +5.0%. The counts say the probes are
    identical to within 0.2%, so that spread is run-to-run noise.
  - (d) N's build is 3-41% slower (full scale, 4 threads: B 33-45 s, N 42-52 s), so only a percent-level gain would
    pay for it.
- **Every conclusion below holds for any delta >= 0:** there are 0 positives in 90 cases (`positives`).
- **What the counts can show.** N's build picks its root with fit_root (raw, binary or flow, whichever costs least)
  and bypasses the flow per region. In all 90 cases it keeps B's root (`d_root` = 0), and the flow runs in at most
  6.3% of N's regions. So B and N differ by less than 1e-4 of D in probes, and benefit_net is the transform charge
  (it accounts for all of benefit_net to within 0.004 probes per lookup). A positive case needs fit_root itself to
  pick the flow root, and the root gate below computes fit_root's criterion on a sample of fences. Its agreement with
  this ground truth is therefore close to agreement by construction: it tests the sample estimate of the build's own
  root choice, not an independent outcome. The independent outcome is the timed pair, which cannot resolve less than
  a few percent.

## Timed verdict per dataset (GRE, one run each, battery + Low Power Mode)

| dataset | B steady ns | N steady ns | N speedup (steady) | N speedup (lower envelope) | within-run CI95 | GRE Mops/s ratio | verdict |
|---|---|---|---|---|---|---|---|
| fb | 1451 | 1381 | +5.0% | +0.4% | 4.0% | +4.5% | skip |
| osm | 1655 | 1590 | +4.1% | +3.1% | 6.5% | +3.8% | skip |
| books | 1625 | 1662 | -2.2% | +0.7% | 2.2% | -2.1% | skip |
| covid | 1604 | 1593 | +0.7% | -2.0% | 1.8% | +0.5% | skip |
| genome | 1580 | 1581 | -0.0% | +1.0% | 1.4% | +0.3% | skip |
| history | 1493 | 1500 | -0.5% | +0.0% | 1.6% | -0.6% | skip |
| libio | 1450 | 1476 | -1.7% | -5.2% | 1.4% | -1.7% | skip |
| planet | 1480 | 1488 | -0.6% | -3.1% | 1.6% | -0.7% | skip |
| stack | 1453 | 1473 | -1.3% | -0.5% | 1.5% | -1.1% | skip |
| wise | 1553 | 1528 | +1.6% | +6.2% | 1.7% | +1.0% | skip |

- The verdict is "use" only when N's steady speedup exceeds delta plus the within-run CI **and** the lower envelope
  (interference removed) agrees.
- The verdict is "skip" on all 10 datasets.
- fb's +5.0% just clears delta plus its CI (1% + 4.0%), so only the lower envelope (+0.4%) keeps it at "skip". The
  counts give identical probes (D_B = D_N = 39.79). It is a difference in machine state between the two runs, not the
  flow.
- nf_gate's builds ran on the efficiency cores beside the pairs of fb (all of N, the last ~20 s of B), osm, books,
  covid and genome (`WARMUP.md`, caveats). Those five pairs are less clean than the other five.
- Warm-up does not change this. B and N have the same cold transient: tau 1e4 to 1.6e5 lookups, 45-214 ms of excess
  (`WARMUP.md`).

## Gates (score s, higher = "use the flow"; use iff s > theta)

| gate | score | theta0 |
|---|---|---|
| kl16..kl1024 | KL(raw sample, uniform) - KL(flow sample, uniform), b bins | 0 |
| trend | log-gap trend share, raw - flow | 0 |
| nfl | NFL tail-conflict gain under one linear model | 0.10 (NFL's keep rule) |
| pla16, pla64 | ln(eps-PLA segments raw / flow) | 0 |
| w1, ks | W1 / KS to uniform, raw - flow | 0 |
| root | (min(raw, binary) - flow) / min: root probes predicted by fit_root's own least squares on m evenly spaced fences | 0 (the flow root wins) |
| pred, pred_net | root proxy + mini build of m/4096 regions; pred_net subtracts the predicted transform calls | delta |

- **Cost** is the score's own compute, plus applying the flow to the sample (plus sorting it, where the score needs
  the flow output sorted). Medians, in us, best of 3, measured on an efficiency core at background QoS.
- Drawing the sample is common to every gate and costs 134 / 328 / 1253 / 3517 us at m = 1024 / 4096 / 16384 / 65536
  (cold pages of the key array).
- **AUC of the decision is undefined:** there are no positive cases. In its place, the ranking quality is the
  **C-index** (Harrell's concordance; the AUC generalised to a continuous outcome) of s against benefit_net, plus
  Spearman's rho.
- **LODO:** for each dataset, theta is learned on the other 9 datasets' windows and tested on the held-out dataset's
  8 windows and its full case. Without positives, the most permissive threshold that is error-free on the training
  folds is the training maximum. A LODO false positive therefore means that the held-out dataset leaves the score
  range seen in training.

## Results (`nfgate_eval/nfgate_eval.md` has every gate at every m)

| gate | m | accuracy at theta0 (FP of 90) | accuracy, LODO (FP) | C-index | Spearman | cost (us) | agrees with GRE (full, LODO, of 10) |
|---|---|---|---|---|---|---|---|
| **root** | **1024** | **1.000 (0)** | 0.989 (1) | 0.59 | 0.27 | **331** | 10 |
| root | 4096 | 1.000 (0) | 0.989 (1) | 0.59 | 0.28 | 1138 | 10 |
| pred_net | 1024 / 4096 | 1.000 (0) | 1.000 (0) | 0.50 (constant score) | - | 1316 / 5224 | 10 |
| pred_net | 65536 | 1.000 (0) | 1.000 (0) | 0.61 | 0.44 | 71345 | 10 |
| nfl (NFL's rule) | 1024 | 0.900 (9) | 0.933 (6) | 0.44 | -0.18 | 309 | 9 |
| nfl | 65536 | 0.800 (18) | 0.933 (6) | 0.31 | -0.59 | 19974 | 9 |
| pla16 | 1024 | 0.822 (16) | 0.967 (3) | 0.50 | 0.00 | 496 | 9 |
| kl64 | 1024 | 0.622 (34) | 0.978 (2) | 0.48 | -0.02 | 114 | 9 |
| ks | 1024 | 0.567 (39) | 0.967 (3) | 0.48 | -0.04 | 164 | 10 |
| trend | 1024 | 0.056 (85) | 0.989 (1) | 0.71 | 0.58 | 233 | 10 |

**Chosen gate: the root proxy at m = 1024, theta0 = 0.** (`chosen`)
- **Selection rule.** The budget is 1 ms, about 0.002% of a full-scale build. Constant scores are excluded: a constant
  "always skip" cannot be told apart from a gate on this data. Among the rest, the rule picks the best accuracy at the
  natural threshold, then the best LODO accuracy, then the best C-index, then the lowest cost.
- **Accuracy.** It is right in 90/90 cases at theta0, with 0 false positives, and right on the full scale of all 10
  datasets. It agrees with the timed GRE verdict on 10/10. With no positive case, "right" only means "no false
  positive", which "always skip" also achieves. The 90 cases are 10 datasets with 9 related key sets each (the full
  set and 8 windows of it), not 90 independent tests.
- **Cost.** 331 us (full scale 380 us, windows 324 us), plus 134 us to sample.
- **Why it works.** It is the only score that measures where the flow could win.
  - The root model is fitted on fences, and fences are what fit_root uses. Its root probes match the built index:
    median error 0.019 probes at m = 1024 (0.003 at 65536), max 0.34 (`root_proxy_vs_build`). It picks the same
    raw-vs-binary root as the build in 86/90 cases.
  - The flow root always costs 6.8 to 17.7 probes more than the best of raw and binary (`flow_root_excess_probes`).
  - The region-level flow moves fence probes by at most 0.004 per lookup, and adds up to 0.063 transform calls per
    lookup (an fb window; 0.049 on fb full). The root decides.
- **Margin.** Its largest score is 0.82 IQR below the threshold (`headroom_iqr`). Every flatness score sits on both
  sides of its threshold.
- **Its one LODO false positive is an artefact of the tightest error-free threshold.** It is osm window 3, where
  the flow root costs 18.5 probes against 12.0 for binary (s = -0.54). It is the largest score of all 90 cases, so it
  sits above every threshold learned without osm.
- **Why not pred_net.**
  - At m <= 4096 the mini build holds a single region, so its score is identically 0, and it costs 4-16x as much as
    the root proxy.
  - At m = 65536 it ranks benefit_net better (C = 0.61) and has LODO 1.000, but costs 71 ms. It adds the region
    term, which here is 1e-4 of D.

**Per-dataset decisions (full 200M, root proxy, m = 1024):** skip on all 10 datasets, at theta0 and under LODO
(`gates.root.by_m.1024.per_dataset`). Scores: fb -1.48, osm -0.69, books -0.68, covid -0.68, genome -0.69,
history -0.70, libio -0.69, planet -0.68, stack -0.69, wise -0.68. In words, the flow root would cost 68-148% more
root probes than the root SCALE-LI already picks. That matches the ground truth (skip) and GRE (skip) on all 10.

**The flatness gates measure flatness, not probes.**
- NFL's own rule says "keep" on osm (6/8 windows and the full set) and on planet. At m = 1024 it has 9 false
  positives at theta0 and 6 under LODO.
- KL says the flow flattens the keys in 27-37 of the 90 cases, depending on bins and m. The largest values are on fb
  full (3.04) and osm, and are driven by outliers.
- trend has the best ranking (C = 0.71), but at theta0 it says "use" in 85/90 cases.
- None of these is a usable gate at its natural threshold. Under LODO they look better only because the threshold
  moves to the training maximum.

## Figures

| file | what |
|---|---|
| `nfgate_eval/nfgate_scatter.png` | score vs measured benefit_net for 8 gates at m = 1024. Colour and marker = dataset, outlined = full 200M, dashed = theta0, grey band = the 10 LODO thresholds |
| `nfgate_eval/nfgate_vs_m.png` | accuracy at theta0, LODO accuracy and C-index against m; accuracy against cost with the 1 ms budget |
| `nfgate_eval/nfgate_decisions.png` | per dataset x gate: full-scale decision under LODO, windows saying "use" (of 8), ground truth, GRE verdict |
| `nfgate_eval/nfgate_decisions_theta0.png` | the same at the natural thresholds |

## Limits

- **No positive case exists in this data.** None of the 10 datasets has a flow that wins the root. So the gate's
  sensitivity cannot be measured: what is measured is that it never says "use" when it should not, and that it tracks
  the only term that could flip.
- A positive control needs a flow that linearizes the fences better than raw. The dense_sparse test flow does not
  (B = N).
- **Ten datasets are a small sample.** 0 false positives on 10 datasets bounds the per-dataset false-positive rate
  only below 26% (one-sided 95%, Clopper-Pearson), and says nothing about false negatives.
- **Not a held-out evaluation of the choice.** The gate was chosen on the same 90 cases it is scored on (the rule
  ranks accuracy at theta0 first), and the root proxy was corrected once after a check on fb at 200M (commit 0eb7991:
  27.6 predicted root probes against 10.8 measured). fb is therefore not a held-out dataset for it. LODO holds out
  only the threshold.
- The gate costs were timed on an efficiency core; on a P-core they are expected to be lower. The timed B/N pair is
  one run each, on battery and in Low Power Mode.
