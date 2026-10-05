#!/usr/bin/env python3
"""Probes per lookup: NFL alone, CSV alone, the fused selector, and the joint design.

Needs matplotlib.  /usr/bin/python3 results/aidb/probes_four.py [--slide]

Sources: results/aidb_final/root_analysis.json (540 paired runs) for the first three;
results/aidb_jointrun/summary.json (90 runs) for the joint design, which changed the root only,
so its fence and in-block counts are the control's.
"""
import json, pathlib, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
S = HERE.parent.parent
ORANGE, BLUE, BLUE_L, TEAL = "#eb6834", "#2a78d6", "#86b6ef", "#1baf7a"
INK, MUTED, GRID, SURF = "#17242f", "#6b7480", "#d8d7d0", "#f4f4f1"
SLIDE = "--slide" in sys.argv

ra = json.loads((S / "results/aidb_final/root_analysis.json").read_text())["per_dataset"]
jr = json.loads((S / "results/aidb_jointrun/summary.json").read_text())
ORDER = ["osm", "planet", "books", "wise", "stack", "genome", "libio", "history", "fb", "covid"]
tot = lambda d, v: ra[d + "_uniform"][v]["total_probes"]
def joint_tot(d):
    c = ra[d + "_uniform"]["packed_rank"]
    return jr[d]["root"] + (c["total_probes"] - c["root_probes"])

SERIES = [(lambda d: tot(d, "packed_rank_root_flow"), ORANGE, "NFL alone: the transform at the root", +0.30),
          (lambda d: tot(d, "packed_rank_vp10"), BLUE, "CSV alone: virtual points in the regions", +0.10),
          (lambda d: tot(d, "packed_rank_vp10_root_fusion"), BLUE_L, "Fused selector: CSV in the regions AND at the root (NFL offered, declined)", -0.10),
          (joint_tot, TEAL, "Joint design: the learned bend + CSV virtual fences at the root", -0.30)]

fig = plt.figure(figsize=(13.33, 8.3) if SLIDE else (13.0, 8.3), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.105, 0.365, 0.880, 0.495]); ax.set_facecolor("white")
FS_DS, FS_BAR, FS_LEG, FS_TICK = (13.0, 9.8, 10.8, 11.0) if SLIDE else (11.5, 8.6, 9.8, 10)
h, ctrl = 0.20, {d: tot(d, "packed_rank") for d in ORDER}
for fn, colour, label, off in SERIES:
    ys = [i + off for i in range(len(ORDER))]
    vals = [100 * (fn(d) / ctrl[d] - 1) for d in ORDER]
    ax.barh(ys, vals, height=h * 0.90, color=colour, label=label, zorder=3)
    for y, d, v in zip(ys, ORDER, vals):
        ax.text(v - 0.8, y, f"{v:+.0f}%", va="center", ha="right",
                fontsize=FS_BAR + 0.4, color=MUTED, zorder=4)
ax.axvline(0, color=INK, lw=1.4, zorder=2)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=FS_DS, color=INK)
ax.set_xlim(-56, 3); ax.set_xticks([-50, -40, -30, -20, -10, 0])
ax.set_xticklabels(["−50%", "−40%", "−30%", "−20%", "−10%", "0"], fontsize=FS_TICK, color=MUTED)
ax.set_xlabel("change in probes per lookup against the packed control   (further left is less work)",
              fontsize=FS_TICK + 0.5, color=MUTED, labelpad=9)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=1); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0, pad=10); ax.tick_params(axis="x", length=0)
ax.set_ylim(-0.62, len(ORDER) - 0.38)

fig.legend(loc="lower left", bbox_to_anchor=(0.102, 0.250), ncol=2, frameon=False,
           fontsize=FS_LEG, labelspacing=0.42, handlelength=1.5, handleheight=1.0, borderpad=0.0)

# ---- what the metric is
fig.patches.append(plt.Rectangle((0.102, 0.035), 0.883, 0.175, transform=fig.transFigure,
                                 facecolor=SURF, edgecolor=GRID, lw=1.0, zorder=0))
fig.text(0.117, 0.192, "WHAT A PROBE IS", fontsize=FS_LEG - 0.2, color=MUTED, fontweight="semibold", va="top")
for i, line in enumerate([
        "One probe = one key comparison: is the key I want before or after this one? A single lookup makes many. Deterministic, so the count is the same on any machine.",
        "A lookup asks them in three rounds. ROOT: which chunk of the data, 8.96 by binary search, 2.1 with a learned root.  FENCE: correcting the block the model predicted, 2.2 to 5.1.",
        "COORDINATE: locating the block's entry, 5.03. The bars are the sum of those three; the control spends 16.2 to 19.1 of them.",
        "NOT included: the in-block search, a further 8.02 comparisons on every variant. Counting those too, the index does 24.2 to 27.1 against a plain binary search's 20.9.",
        "So the index makes MORE comparisons than no index at all. Probes measure how well the model guesses, not how long a lookup takes."]):
    fig.text(0.117, 0.168 - i * 0.0255, line, fontsize=FS_LEG - 1.6, color=INK, va="top")

if not SLIDE:
    fig.text(0.105, 0.968, "Work per lookup: each method alone, the fused selector, and the joint design",
             fontsize=15.5, color=INK, fontweight="semibold", va="top")
    fig.text(0.105, 0.925, "Ten 2M uniform samples, identical query traces. No error bars: the counts are exact.",
             fontsize=9.8, color=MUTED, va="top")
    fig.text(0.105, 0.010, "Source: results/aidb_final/root_analysis.json and results/aidb_jointrun/summary.json  ·  generated by results/aidb/probes_four.py",
             fontsize=8, color=MUTED, va="bottom")

out = HERE / ("probes_four_slide.png" if SLIDE else "probes_four.png")
fig.savefig(out, facecolor="white")
print("wrote", out)
for d in ORDER:
    print(f"  {d:<9} ctrl {ctrl[d]:5.2f}  NFL {tot(d,'packed_rank_root_flow'):5.2f}  CSV {tot(d,'packed_rank_vp10'):5.2f}  "
          f"fused {tot(d,'packed_rank_vp10_root_fusion'):5.2f}  joint {joint_tot(d):5.2f}")
