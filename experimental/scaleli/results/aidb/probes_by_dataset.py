#!/usr/bin/env python3
"""Plot 1 of 3: the MECHANISM. Probes per lookup, the work each method removes.

Needs matplotlib (outside the stdlib-only pipeline).
    /usr/bin/python3 results/aidb/probes_by_dataset.py [--slide]

Probes are key comparisons counted by software counters: deterministic, identical on any machine,
and exactly the quantity the two methods are designed to change. Same four methods, same row order
and same colours as throughput_by_dataset.py, so the two read line by line.

Sources: results/aidb_final/root_analysis.json for the index variants; results/aidb_jointrun for
the joint design; plain binary search over n sorted keys costs log2(n) comparisons by construction.
"""
import json, math, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
S = HERE.parent.parent
BLUE, TEAL, ORANGE, GRAY = "#2a78d6", "#1baf7a", "#eb6834", "#9aa5ad"
INK, MUTED, GRID = "#17242f", "#6b7480", "#d8d7d0"
SLIDE = "--slide" in sys.argv

ra = json.loads((S / "results/aidb_final/root_analysis.json").read_text())["per_dataset"]
jr = json.loads((S / "results/aidb_jointrun/summary.json").read_text())
ORDER = ["osm", "planet", "books", "wise", "stack", "genome", "libio", "history", "fb", "covid"]
N_KEYS = 2_000_000
BINARY = math.log2(N_KEYS)          # comparisons a plain binary search performs

def total(d, v):
    r = ra[d + "_uniform"][v]
    return r["total_probes"]

def joint_total(d):
    # the joint run replaced the ROOT only; fence and coordinate probes are the control's
    c = ra[d + "_uniform"]["packed_rank"]
    return jr[d]["root"] + (c["total_probes"] - c["root_probes"])

fig = plt.figure(figsize=(13.33, 7.5) if SLIDE else (12.8, 7.4), dpi=200)
fig.patch.set_facecolor("white")
if SLIDE:
    ax = fig.add_axes([0.105, 0.265, 0.880, 0.625]); FS_DS, FS_BAR, FS_LEG, FS_TICK = 13.5, 10.2, 11.2, 11.5
else:
    ax = fig.add_axes([0.105, 0.215, 0.880, 0.545]); FS_DS, FS_BAR, FS_LEG, FS_TICK = 11.5, 8.6, 9.8, 10
ax.set_facecolor("white")

SERIES = [(lambda d: BINARY, GRAY, "No index at all: plain binary search, log2(2M) comparisons", +0.30),
          (lambda d: total(d, "packed_rank_root_flow"), ORANGE, "NFL only: the transform, at the root", +0.10),
          (lambda d: total(d, "packed_rank_vp10"), BLUE, "CSV only: virtual points, in the regions", -0.10),
          (lambda d: joint_total(d), TEAL, "Joint design: the learned bend + CSV virtual fences at the root", -0.30)]
h = 0.20
ctrl = {d: total(d, "packed_rank") for d in ORDER}
for fn, colour, label, off in SERIES:
    ys = [i + off for i in range(len(ORDER))]
    vals = [100 * (fn(d) / ctrl[d] - 1) for d in ORDER]
    ax.barh(ys, vals, height=h * 0.90, color=colour, label=label, zorder=3)
    for y, d, v in zip(ys, ORDER, vals):
        ax.text(v + (0.9 if v >= 0 else -0.9), y, f"{v:+.0f}%  ({ctrl[d]:.1f}→{fn(d):.1f})",
                va="center", ha="left" if v >= 0 else "right", fontsize=FS_BAR, color=MUTED, zorder=4)

ax.axvline(0, color=INK, lw=1.4, zorder=2)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=FS_DS, color=INK)
ax.set_xlim(-62, 62); ax.set_xticks([-50, -25, 0, 25, 50])
ax.set_xticklabels(["−50%", "−25%", "0", "+25%", "+50%"], fontsize=FS_TICK, color=MUTED)
ax.set_xlabel("change in total probes per lookup against the packed control   (further left is less work)",
              fontsize=FS_TICK + 0.5, color=MUTED, labelpad=9)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=1); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0, pad=10); ax.tick_params(axis="x", length=0)
ax.set_ylim(-0.62, len(ORDER) - 0.38)

if SLIDE:
    fig.legend(loc="lower left", bbox_to_anchor=(0.102, 0.085), ncol=1, frameon=False,
               fontsize=FS_LEG, labelspacing=0.45, handlelength=1.5, handleheight=1.0, borderpad=0.0)
    fig.text(0.102, 0.035, "Deterministic: the same index and the same query trace give the same count on any machine. No error bars because there is no error.",
             fontsize=11, color=INK, va="bottom", fontweight="semibold")
    fig.text(0.102, 0.008, "Total probes = root + block-correction + in-block comparisons. Plain binary search is shown at log2(2M) = 21.0 comparisons, which is MORE than the index spends.",
             fontsize=9.4, color=MUTED, va="bottom")
else:
    fig.text(0.105, 0.950, "Plot 1 of 3, the mechanism: work per lookup", fontsize=15.5,
             color=INK, fontweight="semibold", va="top")
    for i, line in enumerate([
            "Probes are key comparisons from the software counters, and they are deterministic: the same index and query trace give the same count anywhere. Hence no error bars.",
            "Total probes = root routing + block correction + in-block search. The joint design changed the root only, so its fence and in-block counts are the control's.",
            f"Plain binary search is shown at log2(2M) = {BINARY:.1f} comparisons. Note it does MORE comparisons than the index and is nonetheless faster, which is what plot 3 explains."]):
        fig.text(0.105, 0.893 - i * 0.031, line, fontsize=9.8, color=MUTED, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.102, 0.030), ncol=1, frameon=False, fontsize=FS_LEG,
               labelspacing=0.5, handlelength=1.5, handleheight=1.0, borderpad=0.0)
    fig.text(0.105, 0.010, "Source: results/aidb_final/root_analysis.json and results/aidb_jointrun/summary.json  ·  generated by results/aidb/probes_by_dataset.py",
             fontsize=8, color=MUTED, va="bottom")

out = HERE / ("probes_by_dataset_slide.png" if SLIDE else "probes_by_dataset.png")
fig.savefig(out, facecolor="white")
print("wrote", out)
for d in ORDER:
    print(f"  {d:<9} control {ctrl[d]:5.2f}  binary {BINARY:5.2f}  NFL {total(d,'packed_rank_root_flow'):5.2f}  "
          f"CSV {total(d,'packed_rank_vp10'):5.2f}  joint {joint_total(d):5.2f}")
