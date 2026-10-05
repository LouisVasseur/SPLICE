"""Write armA.md from armA.json (run armA_report.py first)."""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "armA.json")))
R = J["rows"]
T = open(os.path.join(HERE, "armA_table.md")).read()


def agg(rs):
    nb = [r for r in rs if r["band_spread"] is not None]
    nl = [r for r in rs if r["band_listed"] is not None]
    g = [r["gain_rounds2plus"] for r in rs]
    return "| %d | %d | %.3f | %.3f | %d/%d | %d/%d | %d/%d | %d | %d |" % (
        len(rs), sum(1 for x in g if x > 1e-6), st.mean(g), max(g),
        sum(1 for r in nb if r["exceeds_spread"]), len(nb), sum(1 for r in nb if r["exceeds_lucky"]), len(nb),
        sum(1 for r in nl if r["exceeds_listed"]), len(nl),
        sum(r["gapset_changed_in_accepted_round"] for r in rs), sum(r["gapset_changed_in_any_candidate"] for r in rs))


groups = [("k=0 control (T<->V only)", [r for r in R if r["k"] == 0])]
groups += [("all k>0", [r for r in R if r["k"] > 0])]
groups += [("k=%d" % k, [r for r in R if r["k"] == k]) for k in (1, 4, 16, 64)]
groups += [("order %s" % o, [r for r in R if r["k"] > 0 and r["order"] == o]) for o in ("GTV", "TGV")]
groups += [("%s samples, k>0" % m, [r for r in R if r["k"] > 0 and r["mode"] == m]) for m in ("uniform", "window")]
groups += [("k>0, gap set changed in an accepted round", [r for r in R if r["k"] > 0 and r["gapset_changed_in_accepted_round"]])]
aggtab = ["| group | cells | gain > 1e-6 | mean gain | max gain | gain > spread@r1 | gain > lucky@r1 | gain > listed band | gap set changed (accepted) | gap set changed (any candidate) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
aggtab += ["| %s %s" % (lab, agg(rs)) for lab, rs in groups]

K0R1 = {(r["dataset"], r["mode"]): r["round1"]["model_probes"] for r in R if r["k"] == 0}
BK = [r for r in R if r["dataset"] == "books" and r["mode"] == "uniform" and r["k"] == 64 and r["order"] == "GTV"][0]
strong = [r for r in R if r["k"] > 0 and r["exceeds_spread"] and r["exceeds_lucky"] and r["exceeds_listed"] in (True, None)]
strongtab = ["| cell | r1 total | best total (round) | gain | spread / lucky / listed | gap set changed | r1 model vs k=0 r1 model | best model vs k=0 best model | best total vs k=0 best total |", "|---|---|---|---|---|---|---|---|---|"]
for r in strong:
    strongtab.append("| %s-%s k=%d %s | %.3f | %.3f (%d) | %.3f | %.3f / %.3f / %s | %s | %+.3f | %+.3f | %+.3f |" % (
        r["dataset"], r["mode"], r["k"], r["order"], r["round1"]["total_uncharged"], r["best"]["total_uncharged"], r["best"]["round"],
        r["gain_rounds2plus"], r["band_spread"], r["band_lucky"], ("%.3f" % r["band_listed"]) if r["band_listed"] is not None else "-",
        "yes (%d of %d replaced)" % (r["gaps_swapped_best_vs_round1"] // 2, r["k"]) if r["gapset_changed_in_accepted_round"] else "no",
        r["round1"]["model_probes"] - K0R1[(r["dataset"], r["mode"])], r["delta_best_vs_k0_model"], r["delta_best_vs_k0_total"]))

beat_k0 = [r for r in R if r["k"] > 0 and r["delta_best_vs_k0_total"] < 0]
k0 = [r for r in R if r["k"] == 0]
G = [r for r in R if r["k"] > 0]
def cnt(rs, key):
    nb = [r for r in rs if r["band_spread"] is not None]
    return sum(1 for r in nb if r[key]), len(nb)
gs, gn = cnt(G, "exceeds_spread"); gl, _ = cnt(G, "exceeds_lucky")
ks, kn = cnt(k0, "exceeds_spread"); kl, _ = cnt(k0, "exceeds_lucky")
mind = min(G, key=lambda r: r["delta_best_vs_k0_total"])
sd = [r["delta_best_vs_k0_total"] for r in strong]
PAIRS = list(zip([r for r in R if r["k"] > 0 and r["order"] == "GTV"], [r for r in R if r["k"] > 0 and r["order"] == "TGV"]))
within = [r for r in strong if abs(r["delta_best_vs_k0_model"]) <= r["band_spread"]]
md = f"""# Arm A: back-and-forth over G, T and V

Question: once gap removal (G) is added to the warp (T) and the CSV virtual fences (V), do rounds 2 and later of a
block-coordinate cycle improve on the single coarse-to-fine pass by more than the CSV greedy's instability band?

**Answer: no, not in any way that matters.** G does not change last week's verdict on back-and-forth.

- Rounds 2 and later beat the round-1 chaos spread in {gs} of {gn} G cells ({100.0*gs/gn:.1f}%). The no-G control does so in {ks} of {kn} ({100.0*ks/kn:.1f}%).
  Against the "lucky re-roll" null the rates are {gl}/{gn} ({100.0*gl/gn:.0f}%) with G and {kl}/{kn} ({100.0*kl/kn:.0f}%) without.
- Where the gain does clear every yardstick ({len(strong)} cells), the cycle mostly **repairs a weak round 1**. It does not
  produce a usable advantage over the no-G optimum.
  - Model-only, {len(within)} of the {len(strong)} end within their own round-1 spread of the no-G optimum.
  - The rest are slightly better model-only ({", ".join("%s-%s k=%d %s: %+.3f against a spread of %.3f" % (r["dataset"], r["mode"], r["k"], r["order"], r["delta_best_vs_k0_model"], r["band_spread"]) for r in strong if r not in within)}).
    Even so, the gap table leaves them {", ".join("%+.2f" % r["delta_best_vs_k0_total"] for r in strong if r not in within)} probes worse on total.
  - The largest case is books-uniform k=64 (G->T->V), which gains {BK["gain_rounds2plus"]:.3f}. Round 1 picked the gaps in raw x,
    and the warp fitted after that made the model worse than no G at all ({BK["round1"]["model_probes"]:.3f} vs {K0R1[("books", "uniform")]:.3f}).
  - The cycle then replaced {BK["gaps_swapped_best_vs_round1"] // 2} of the 64 gaps and ended at model {BK["best"]["model_probes"]:.3f}.
    That is about last week's no-G joint result ({BK["k0_best_total"]:.3f}).
- **No G configuration beats its sample's no-G optimum on total_uncharged, on any of the 20 samples, at any k, in either order,
  after any number of rounds** ({len(beat_k0)} of 160 cells). The closest is {mind["dataset"]}-{mind["mode"]} k={mind["k"]} {mind["order"]}, at {mind["delta_best_vs_k0_total"]:+.3f} probes. The gap-table charge (1, 3, 5 or 7 probes for k = 1, 4, 16, 64) is
  never paid back. The only sample where G's model-only saving is large is osm-uniform: at k=64, model probes fall from 8.22 to 4.67.
  That saving is still smaller than the 7-probe table charge.

## Method (as run)

- **Module.** `threeblock.py`, imported and not modified. G uses select="current": gaps are ranked by width in the current
  composed coordinate f(g(x)), and an already-shrunk gap is measured unshrunk. V budget is 4x the number of fences (the deployed setting).
- **Round 1.** One pass, in the order G->T->V ("GTV") or T->G->V ("TGV").
- **Round r >= 2.** The same three blocks again, each applied to the current state. G re-selects in the current coordinate, T
  refits to the current slot targets in the current g-coordinate, and V re-runs the greedy with joint.py's keep-previous-slots guard.
- **Round guard.** This is the keep-previous rule lifted to the whole round. A round is accepted only if total_uncharged drops
  by more than 1e-6. The first rejected round stops the cycle, since the cycle is deterministic. At most 6 rounds are run,
  round 1 included.
- **Diagnostic "nostop" runs.** Same cycle, but with no round guard and no early stop. All 6 rounds run, and the best round is
  chosen afterwards, either by probes (optimistic) or by loss (last week's rule).
- **Control.** k=0 is run in the same harness. It reproduces last week's `sequential` exactly as round 1 on all 10 uniform samples.
  Its guarded best matches last week's `joint` on 8 of 10 samples. On osm and genome it is better, because there the probe-based guard
  keeps round 1 while last week's loss-based selection did not.
- **Noise yardsticks**, all measured at the round-1 state of each cell with `perturbed_V_spread` (c4 family: eps in
  {{1e-6, 3.33e-5, 1e-4, 1e-3}}, x^2 and x^3, both signs; 16 variants plus identity):
  - spread = max - min of model probes over the 17 variants.
  - lucky = base - min(variants). This is what re-rolling the greedy buys with no change to G or T, so it is the null for
    "a later round just got a lucky greedy draw".
  - listed = last week's quoted band. It exists for uniform samples only, and not for osm or planet.
- **Determinism.** Re-running genome-window k64 TGV, books-window k16 GTV and history-window k4 TGV reproduced every stored
  round bit for bit (`armA_extend.py`).
- **total_charged** adds the 6.9-probe tanh charge, which is constant once T is on. Every gain is therefore identical in
  charged and uncharged currency.

## Aggregate

{chr(10).join(aggtab)}

## Cells where rounds >= 2 clear every yardstick (spread, lucky, and listed where it exists)

{chr(10).join(strongtab)}

In every one of these cells the best total is {min(sd):.2f} to {max(sd):.2f} probes above that sample's no-G optimum. Model-only, {len(within)} of the {len(strong)} land within their own
round-1 spread of the no-G optimum.

## Did the gap set change after round 1?

- In accepted rounds the gap set changed in {sum(r["gapset_changed_in_accepted_round"] for r in R if r["k"] > 0)} of 160 cells.
  In any candidate round, accepted or rejected, it changed in {sum(r["gapset_changed_in_any_candidate"] for r in R if r["k"] > 0)}.
  In the no-guard runs it changed in {sum(r["gapset_changed_nostop"] for r in R if r["k"] > 0)}.
- Changes concentrate at k=64 and k=16, where the k-th largest gap is close to the median gap. Near that threshold, re-ranking
  after T can swap gaps in and out.
- At k=1 and k=4 the set changed in only {sum(r["gapset_changed_in_any_candidate"] for r in R if r["k"] in (1, 4))} of 80 cells
  (any candidate round). With so few gaps, only the top of the ranking matters, and T rarely re-orders the top.
- {sum(1 for r in strong if not r["gapset_changed_in_accepted_round"])} of the {len(strong)} cells that clear every yardstick kept their round-1
  gap set. Their gain is the T<->V coupling of last week's joint, acting in the G coordinate, not the new G<->T coupling.
- The cells whose gap set did change gained more on average than those whose set did not (see Aggregate). The G<->T coupling is
  real. What it buys is mainly undoing a poor round-1 selection.

## Other observations

- **Order matters little.** TGV has the better round-1 total in {sum(1 for a, b in PAIRS if b["round1"]["total_uncharged"] < a["round1"]["total_uncharged"] - 1e-9)} of 80 cells
  and the better final total in {sum(1 for a, b in PAIRS if b["best"]["total_uncharged"] < a["best"]["total_uncharged"] - 1e-9)} of 80.
  - The GTV and TGV finals differ by {st.mean(abs(a["best"]["total_uncharged"] - b["best"]["total_uncharged"]) for a, b in PAIRS):.3f} probes on average,
    which is inside a typical spread.
  - The largest difference is {max(abs(a["best"]["total_uncharged"] - b["best"]["total_uncharged"]) for a, b in PAIRS):.3f}.
- **The early stop sometimes quits too soon.** In {sum(1 for r in R if r["gain_nostop_postselected"] > r["gain_rounds2plus"] + 1e-9)} of 180 cells,
  the no-guard run (best round chosen afterwards by probes) finds a lower total than the guarded cycle. The largest case is
  osm-uniform k64 TGV: round 2 got worse, and rounds 3 and 4 recovered 0.236. That is post-hoc selection on the evaluation
  metric itself.
- **Selecting by loss is worse.** Choosing the no-guard round with the smallest least-squares loss (last week's rule) gives a
  total worse than round 1 in {sum(1 for r in R if r["gain_nostop_loss_selected"] < -1e-9)} of 180 cells. Loss and probes do not
  move together closely enough for the loss to choose the round.
- **The round cap rarely binds.** It was hit once (genome-window k64 TGV). Re-run with a cap of 12, that cell stops at round 7
  with no further gain, so the cap does not change its result (0.168, against a spread of 0.177).

## Per-cell table

Column notes:
- r1 = round 1 and gapT = gap-table probes. The best round is the last accepted one.
- gain = r1 total - best total, in uncharged probes.
- spread@r1 and lucky@r1 = the chaos yardsticks above.
- The gap set column reads accepted rounds / any candidate / nostop runs.
- virt = virtual-fence count.
- The last column gives the best result minus the no-G (k=0) best for the same sample, as total / model-only.

{T}

## Caveats

- **The band is measured once per cell.** It is taken at the round-1 state, with one perturbation family. A later round's
  state has its own band, not measured here. Running 148 cells against a one-sided threshold also produces a few exceedances
  by chance: compare the k=0 control rate, not zero.
- **The G selection rule is threeblock's first-order rule.** Gaps are ranked by width in f(g(x)), and selected gaps are
  shrunk to the median on the input axis. A different "current coordinate" rule could couple G and T more strongly or less.
- **The gap-table charge is the binary search only**, ceil(log2(k_eff+1)) probes. With a stricter charge, G's totals would
  only get worse. The back-and-forth verdict does not depend on the charge, because the charge is constant across rounds. The
  "G never pays" verdict does depend on it. The closest G cell is {mind["delta_best_vs_k0_total"]:+.3f} probes from the no-G
  optimum. If the table were free (model-only column), G would win clearly on osm-uniform
  ({min(r["delta_best_vs_k0_model"] for r in G if r["dataset"] == "osm" and r["mode"] == "uniform"):+.3f} at best), and elsewhere only by
  amounts comparable to the spread.
- **The T-first order (TGV) evaluates the warp on g(x) before it is refit in round 1.** That is the literal T->G->V order.
  The refit happens at the start of round 2.
- **Some of last week's listed bands came from a different perturbation subset** (see the harness report). The spread
  column here uses the c4 family consistently.
"""
open(os.path.join(HERE, "armA.md"), "w").write(md)
print("wrote armA.md", len(md))
