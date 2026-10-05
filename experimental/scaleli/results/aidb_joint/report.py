#!/usr/bin/env python3
"""Render joint_results.json as aligned tables and answer the four questions."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "joint_results.json")))
BUDGETS = ["0.5", "1", "4"]
out = []
w = out.append

w("ROOT PROBES PER LOOKUP over the 2n-1 fence+midpoint probe set (index.hpp:371-372, 389-392).")
w("Lower is better. 'binary' is the binary-search fallback the root must beat to be used at all.")
w("lambda = budget x (#fences).  No flow_cost penalty is charged to any column (see caveats).")
w("")
hdr = "%-8s %5s %6s %7s | %s | %6s" % (
    "dataset", "bin", "linear", "flowonly",
    " | ".join("%7s %7s %7s" % ("csv" + b, "seq" + b, "jnt" + b) for b in BUDGETS),
    "match")
w(hdr); w("-" * len(hdr))
for r in R:
    cells = []
    for b in BUDGETS:
        d = r["budgets"][b]
        cells.append("%7.2f %7.2f %7.2f" % (d["csv_only"], d["sequential"], d["joint"]))
    m = r["joint_budget_to_match_csv4x"]
    w("%-8s %5.2f %6.2f %7.2f | %s | %6s" % (
        r["dataset"], r["binary"], r["baseline_linear"], r["flow_only"],
        " | ".join(cells), ("<=%gx" % m if m is not None else "none")))
w("")
w("DELTAS at the deployed budget lambda = 4x")
hdr2 = "%-8s %9s %9s %9s %9s %9s" % ("dataset", "jnt-seq", "jnt-csv", "seq-csv", "flow-lin", "csv-lin")
w(hdr2); w("-" * len(hdr2))
for r in R:
    d = r["budgets"]["4"]
    w("%-8s %+9.3f %+9.3f %+9.3f %+9.3f %+9.3f" % (
        r["dataset"], d["joint"] - d["sequential"], d["joint"] - d["csv_only"],
        d["sequential"] - d["csv_only"], r["flow_only"] - r["baseline_linear"],
        d["csv_only"] - r["baseline_linear"]))
w("")
w("DIAGNOSTICS at lambda = 4x.  'greedy used' is SmoothingResult.rounds against the budget:")
w("equal to the budget means the budget BINDS, less means the greedy stopped early (P2).")
w("'r2+ share' is the fraction of the loss reduction bought by BCD rounds >= 2 (P3).")
hdr3 = "%-8s %6s %11s %11s %6s %5s %9s %10s %10s" % (
    "dataset", "budget", "csv used", "seq used", "rounds", "sel", "r2+ share", "loss seq", "loss joint")
w(hdr3); w("-" * len(hdr3))
for r in R:
    d = r["budgets"]["4"]
    w("%-8s %6d %11d %11d %6d %5d %9.4f %10.4g %10.4g" % (
        r["dataset"], d["lambda"], d["csv_rounds"], d["seq_greedy_rounds"],
        d["joint_rounds"], d.get("joint_selected_round", 0),
        d.get("rounds2plus_over_round1", 0.0),
        d.get("joint_loss_round1", 0.0), d.get("joint_loss_best", 0.0)))
w("")
w("SPACE (d6).  The objective has no space term, so comparing at matched lambda is")
w("misleading: the root slot table costs 4 bytes per slot ACTUALLY allocated")
w("(index.hpp:333,387) while the flow file is 10 doubles = 80 bytes.  At the deployed")
w("lambda = 4x, both probes and bytes:")
hdrS = "%-8s | %7s %6s %8s | %7s %6s %8s | %8s %8s" % (
    "dataset", "csv4", "csvVP", "bytes", "jnt4", "jntVP", "bytes", "d probe", "d bytes")
w(hdrS); w("-" * len(hdrS))
for r in R:
    n = r["n_fences"]; d = r["budgets"]["4"]
    cvp = d["csv_virtual"]; cb = 4 * (n + cvp)
    jvp = d["joint_virtual"]; jb = 4 * (n + jvp) + 80
    w("%-8s | %7.2f %6d %8d | %7.2f %6d %8d | %+8.2f %+8d" % (
        r["dataset"], d["csv_only"], cvp, cb, d["joint"], jvp, jb,
        d["joint"] - d["csv_only"], jb - cb))
w("")
w("P2 pre-check: greedy insertions USED vs budget on the raw feature (csv_only).")
w("The budget binds only where used == budget.")
hdrP = "%-8s %12s %12s %12s" % ("dataset", "lambda=0.5x", "lambda=1x", "lambda=4x")
w(hdrP); w("-" * len(hdrP))
for r in R:
    w("%-8s %12s %12s %12s" % (r["dataset"],
      "%d/%d" % (r["budgets"]["0.5"]["csv_rounds"], r["budgets"]["0.5"]["lambda"]),
      "%d/%d" % (r["budgets"]["1"]["csv_rounds"], r["budgets"]["1"]["lambda"]),
      "%d/%d" % (r["budgets"]["4"]["csv_rounds"], r["budgets"]["4"]["lambda"])))
w("")
w("THREE-ARM CONTROL for confound d5 (what changed: the objective, or the alternation?).")
w("A = flow trained on the NFL negative log-likelihood (tools/train_flow.py --monotone,")
w("    weights read from the workspace sweep), then CSV greedy -- the TRUE sequential baseline.")
w("B = flow trained on the downstream least-squares objective, then CSV greedy (one pass).")
w("C = the same, alternated (joint).  All at lambda = 4x.")
try:
    A = {r["dataset"]: r for r in json.load(open(os.path.join(HERE, "arm_a_results.json")))}
except Exception:
    A = {}
if A:
    hdrA = "%-8s %9s %9s %9s %9s %9s %9s" % ("dataset", "flowA", "flowB", "A:seq4", "B:seq4", "C:jnt4", "B-A")
    w(hdrA); w("-" * len(hdrA))
    for r in R:
        a = A.get(r["dataset"])
        if not a:
            continue
        d = r["budgets"]["4"]
        w("%-8s %9.2f %9.2f %9.2f %9.2f %9.2f %+9.2f" % (
            r["dataset"], a["flow_only"], r["flow_only"], a.get("sequential_4x", 0.0),
            d["sequential"], d["joint"], d["sequential"] - a.get("sequential_4x", 0.0)))
w("")
w("SUPPLEMENTARY: the same experiment with a per-unit BIAS inside each tanh")
w("(z = C1*tanh(g1*(x-m1)) + C2*tanh(g2*(x-m2)), +2 weights = 16 bytes), coarser grid.")
hdr4 = "%-8s %10s %10s %11s %11s" % ("dataset", "flow(sh)", "flow(bias)", "seq4(sh)", "seq4(bias)")
w(hdr4); w("-" * len(hdr4))
for r in R:
    if "bias_flow_only" not in r:
        continue
    w("%-8s %10.2f %10.2f %11.2f %11.2f" % (
        r["dataset"], r["flow_only"], r["bias_flow_only"],
        r["budgets"]["4"]["sequential"], r["bias_sequential_4x"]))
print("\n".join(out))

try:
    P = json.load(open(os.path.join(HERE, "prec_check.json")))
except Exception:
    P = []
if P:
    out2 = []
    out2.append("")
    out2.append("PRECISION CONTROL.  C++ smooth_cdf accumulates in long double; Python has only")
    out2.append("double, and the greedy's stopping rule compares at a relative tolerance of 1e-12.")
    out2.append("csv_only at lambda = 4x recomputed with a double-double accumulator:")
    out2.append("%-8s %10s %8s %10s %8s" % ("dataset", "double", "used", "dd", "used"))
    for r in P:
        out2.append("%-8s %10.3f %8d %10.3f %8d" % (
            r["dataset"], r["double"], r["double_rounds"], r["dd"], r["dd_rounds"]))
    print("\n".join(out2))
