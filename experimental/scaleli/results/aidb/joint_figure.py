#!/usr/bin/env python3
"""The joint-objective experiment: what the bend buys, and what alternating buys.

Needs matplotlib (outside the stdlib-only pipeline).
    /usr/bin/python3 results/aidb/joint_figure.py [--slide]

Source: results/aidb_joint/joint_results.json, the prototype run of 2026-09-22 on the 489 root
fences of each of the ten uniform 2M samples, scored with a replica of Index::fit_root's candidate
score (it reproduces osm's measured root probes to three decimals). No flow_cost is charged in
either panel; the deployed root charges 4.0 probes per lookup for using the transform.

Instability bands come from results/aidb_joint/verify/: the adversarial verifier perturbed the
feature by x +/- 1e-6*x^2 and x +/- 1e-6*x^3, changes too small to matter physically, and recorded
how far csv_only moved at the same budget.
"""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
J = HERE.parent / "aidb_joint"
GRAY, ORANGE, BLUE, TEAL, INKD = "#9aa5ad", "#eb6834", "#2a78d6", "#1baf7a", "#17242f"
INK, MUTED, GRID, BAND = "#17242f", "#6b7480", "#d8d7d0", "#e4e3dd"
SLIDE = "--slide" in sys.argv

rows = json.loads((J / "joint_results.json").read_text())
# bands the verifier measured (verify/c4.py); any it could not finish are filled from verify/bands_extra.json
BANDS = {"books": 0.361, "wise": 0.263, "genome": 0.197, "libio": 0.175,
         "covid": 0.086, "history": 0.050, "stack": 0.045, "fb": 0.000}
extra = J / "verify" / "bands_extra.json"
if extra.exists():
    for k, v in json.loads(extra.read_text()).items(): BANDS[k] = v["band"]

D = {}
for r in rows:
    b4 = r["budgets"]["4"]
    D[r["dataset"]] = {"binary": r["binary"], "linear": r["baseline_linear"], "flow": r["flow_only"],
                       "csv": b4["csv_only"], "seq": b4["sequential"], "joint": b4["joint"],
                       "band": BANDS.get(r["dataset"])}
order = sorted(D, key=lambda d: D[d]["linear"] - D[d]["flow"])
n = len(order)

fig = plt.figure(figsize=(13.33, 7.5) if SLIDE else (13.0, 7.4), dpi=200)
fig.patch.set_facecolor("white")
h = 0.545 if SLIDE else 0.50
axA = fig.add_axes([0.085, 0.235, 0.375, h]); axB = fig.add_axes([0.575, 0.235, 0.375, h])
FS = 11.5 if SLIDE else 10.8
for ax in (axA, axB):
    ax.set_facecolor("white")
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_yticks(range(n)); ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(length=0, labelsize=FS - 1, colors=MUTED)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)

# --- A: does the bend alone help?
axA.set_yticklabels(order, fontsize=FS, color=INK)
axA.axvline(D[order[0]]["binary"], color=GRAY, lw=1.6, ls=(0, (5, 4)), zorder=1)
for i, d in enumerate(order):
    r = D[d]
    axA.plot([r["flow"], r["linear"]], [i, i], color=GRAY, lw=2.6, zorder=2)
    axA.plot([r["linear"]], [i], "o", ms=8, color=GRAY, markeredgecolor="white", markeredgewidth=1.5, zorder=4)
    axA.plot([r["flow"]], [i], "o", ms=8, color=ORANGE, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    gain = r["linear"] - r["flow"]
    axA.text(17.3, i, f"{r['linear']:.1f} → {r['flow']:.1f}" if gain > 0.05 else "no change",
             va="center", ha="right", fontsize=FS - 1.6, color=INK if gain > 0.05 else MUTED)
axA.set_xlim(0, 17.5); axA.set_xticks([0, 4, 8, 12])
axA.text(D[order[0]]["binary"], n - 0.42, "  binary search 8.96", fontsize=FS - 2, color=MUTED, va="bottom")
axA.set_xlabel("root probes per lookup", fontsize=FS - 0.5, color=MUTED, labelpad=8)
axA.set_title("A monotone bend alone does move the root", fontsize=FS + 1.2, color=INK,
              loc="left", pad=10, fontweight="semibold")

# --- B: does combining, or alternating, help?
axB.set_yticklabels(order, fontsize=FS, color=INK)
for i, d in enumerate(order):
    r = D[d]
    if r["band"]:
        axB.add_patch(plt.Rectangle((r["csv"] - r["band"], i - 0.30), 2 * r["band"], 0.60,
                                    color=BAND, zorder=1))
    axB.plot([r["csv"]], [i], "o", ms=8.5, color=BLUE, markeredgecolor="white", markeredgewidth=1.5, zorder=4)
    axB.plot([r["seq"]], [i], "s", ms=6.5, color=TEAL, markeredgecolor="white", markeredgewidth=1.3, zorder=5)
    axB.plot([r["joint"]], [i], "D", ms=5.0, color=INKD, markeredgecolor="white", markeredgewidth=1.1, zorder=6)
    axB.text(10.35, i, f"{r['joint']-r['seq']:+.3f}", va="center", ha="right",
             fontsize=FS - 1.6, color=MUTED)
axB.set_xlim(1.9, 10.5); axB.set_xticks([2, 4, 6, 8])
axB.set_xlabel("root probes per lookup   ·   right column: joint minus sequential", fontsize=FS - 0.5, color=MUTED, labelpad=8)
axB.set_title("but combining adds little, and alternating nothing", fontsize=FS + 1.2,
              color=INK, loc="left", pad=10, fontweight="semibold")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=8.5, color=BLUE, label="CSV virtual fences alone"),
                    Line2D([], [], marker="s", ls="", ms=6.5, color=TEAL, label="bend, then CSV (sequential)"),
                    Line2D([], [], marker="D", ls="", ms=5.0, color=INKD, label="joint (alternating)"),
                    plt.Rectangle((0, 0), 1, 1, color=BAND, label="the greedy's own instability")],
           loc="lower left", bbox_to_anchor=(0.575, 0.045), ncol=2, frameon=False,
           fontsize=FS - 2.0, labelspacing=0.45, columnspacing=1.6)

head = 0.955
if not SLIDE:
    fig.text(0.085, head, "The joint objective, tested: the bend is real, the alternation is not",
             fontsize=15.3, color=INK, fontweight="semibold", va="top")
    head = 0.898
for i, line in enumerate([
        "Left: a single order-preserving tanh pair fitted to the least-squares objective, with ZERO virtual fences. Three datasets go from losing to binary search to beating it.",
        "Right: at the deployed budget of four times the number of fences, CSV alone against bend-then-CSV against alternating between them. The grey band is how far the greedy",
        "moves when the feature is perturbed by one part in a million, a change with no physical meaning. Every joint-minus-sequential difference is at most 0.072 probes.",
        "Neither panel charges for evaluating the transform. Measured on the critical path it costs 28.3 ns, about 6.9 probes, which is more than any column here saves."]):
    fig.text(0.085, head - i * 0.030, line, fontsize=9.6, color=MUTED, va="top")
fig.text(0.085, 0.012, "Source: results/aidb_joint/joint_results.json and verify/  ·  generated by results/aidb/joint_figure.py",
         fontsize=8, color=MUTED, va="bottom")

out = HERE / ("joint_slide.png" if SLIDE else "joint.png")
fig.savefig(out, facecolor="white")
print("wrote", out)
for d in order:
    r = D[d]
    print(f"  {d:<9} linear {r['linear']:6.2f} -> flow {r['flow']:6.2f} | csv {r['csv']:6.3f} seq {r['seq']:6.3f} "
          f"joint {r['joint']:6.3f}  joint-seq {r['joint']-r['seq']:+.3f}  band {r['band'] if r['band'] is not None else float('nan'):.3f}")
