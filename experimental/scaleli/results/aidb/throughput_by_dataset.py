#!/usr/bin/env python3
"""Throughput per dataset, per method: the same layout as summary.png, but measuring time.

Needs matplotlib (outside the stdlib-only pipeline).
    /usr/bin/python3 results/aidb/throughput_by_dataset.py [--slide]

Source: results/aidb_final/root_analysis.json (540 paired runs, 5M lookups each, 3 query seeds,
performance-core QoS, identical query traces). Bars are the paired speedup against packed_rank,
expressed as a percentage change, so each dataset is compared only with itself.

Row order is IDENTICAL to summary.png so the two figures can be read line by line: the first
shows the work each method removes, this one shows what that does to the clock.
"""
import json, pathlib, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
S = HERE.parent.parent
BLUE, TEAL, ORANGE, GRAY = "#2a78d6", "#1baf7a", "#eb6834", "#9aa5ad"
INK, MUTED, GRID, BAND = "#17242f", "#6b7480", "#d8d7d0", "#e8e7e2"
SLIDE = "--slide" in sys.argv

ra = json.loads((S / "results/aidb_final/root_analysis.json").read_text())["per_dataset"]
# same order as summary.png (bottom row first, as barh draws index 0 at the bottom)
ORDER = ["osm", "planet", "books", "wise", "stack", "genome", "libio", "history", "fb", "covid"]
noise = 100 * max(abs(ra[d + "_uniform"]["packed_rank_root_vf4"]["speedup"] /
                      ra[d + "_uniform"]["packed_rank_root_fusion"]["speedup"] - 1) for d in ORDER)

SERIES = [("sorted_vector", GRAY, "No index at all: plain binary search over the sorted array", +0.30),
          ("packed_rank_root_flow", ORANGE, "NFL only: the transform, at the root", +0.10),
          ("packed_rank_vp10", BLUE, "CSV only: virtual points, in the regions", -0.10)]

# The joint design, now MEASURED. Its warp was exported to the NFL weight format (exact to the
# last bit against the prototype's own formula on all 4,890 fences) and run through the deployed
# fit_root with the transform charge set to zero, so the selector chooses on probes alone and the
# warp's real cost lands in the wall clock. results/aidb_jointrun/, 90 runs, same settings.
JOINT = json.loads((S / "results/aidb_jointrun/summary.json").read_text())

fig = plt.figure(figsize=(13.33, 7.5) if SLIDE else (12.8, 7.4), dpi=200)
fig.patch.set_facecolor("white")
if SLIDE:
    ax = fig.add_axes([0.105, 0.265, 0.880, 0.625]); FS_DS, FS_BAR, FS_LEG, FS_TICK = 13.5, 10.2, 11.2, 11.5
else:
    ax = fig.add_axes([0.105, 0.215, 0.880, 0.545]); FS_DS, FS_BAR, FS_LEG, FS_TICK = 11.5, 8.4, 9.8, 10
ax.set_facecolor("white")

ax.axvspan(-noise, noise, color=BAND, zorder=0)
h = 0.20
for key, colour, label, off in SERIES:
    ys = [i + off for i in range(len(ORDER))]
    vals = [100 * (ra[d + "_uniform"][key]["speedup"] - 1) for d in ORDER]
    ax.barh(ys, vals, height=h * 0.90, color=colour, label=label, zorder=3)  # measured
    for y, d, v in zip(ys, ORDER, vals):
        r = ra[d + "_uniform"]; c, m = r["packed_rank"]["mops"], r[key]["mops"]
        txt = f"{v:+.0f}%  ({c:.2f}→{m:.2f})"
        ax.text(v + (1.6 if v >= 0 else -1.6), y, txt, va="center",
                ha="left" if v >= 0 else "right", fontsize=FS_BAR, color=MUTED, zorder=4)

for i, d in enumerate(ORDER):
    j = JOINT[d]; v = 100 * (j["speedup"] - 1); y = i - 0.30
    ax.barh(y, v, height=h * 0.90, color=TEAL, zorder=3)
    ax.plot([100 * (j["lo"] - 1), 100 * (j["hi"] - 1)], [y, y], color=INK, lw=1.3, zorder=5)
    ax.text(v + (1.6 if v >= 0 else -1.6), y,
            f"{v:+.0f}%  ({j['mops_ctrl']:.2f}→{j['mops']:.2f})   {j['chosen']}",
            va="center", ha="left" if v >= 0 else "right", fontsize=FS_BAR, color=MUTED, zorder=4)

ax.axvline(0, color=INK, lw=1.4, zorder=2)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=FS_DS, color=INK)
ax.set_xlim(-44, 130); ax.set_xticks([-25, 0, 25, 50, 75, 100])
ax.set_xticklabels(["−25%", "0", "+25%", "+50%", "+75%", "+100%"], fontsize=FS_TICK, color=MUTED)
ax.set_xlabel("change in throughput against the same index without that component   (further right is faster)",
              fontsize=FS_TICK + 0.5, color=MUTED, labelpad=9)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=1); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0, pad=10); ax.tick_params(axis="x", length=0)
ax.set_ylim(-0.62, len(ORDER) - 0.38)
ax.text(noise + 2.0, -0.45, f"host noise floor, ±{noise:.0f}%", fontsize=FS_BAR + 0.6,
        color=MUTED, va="center", ha="left")

if SLIDE:
    fig.legend(loc="lower left", bbox_to_anchor=(0.085, 0.085), ncol=1, frameon=False,
               fontsize=FS_LEG, labelspacing=0.45, handlelength=1.5, handleheight=1.0, borderpad=0.0)
    fig.text(0.085, 0.035, "Only removing the index moves the clock. Every learned bar is inside the noise band, so the work reductions of the previous figure do not reach it.",
             fontsize=11, color=INK, va="bottom", fontweight="semibold")
    fig.text(0.085, 0.008, "Paired against the same index without that component, 5M lookups per run, 3 query seeds, identical query traces. Same row order as the previous figure.",
             fontsize=9.4, color=MUTED, va="bottom")
else:
    fig.text(0.105, 0.950, "The same methods in time, all four measured",
             fontsize=15.5, color=INK, fontweight="semibold", va="top")
    for i, line in enumerate([
            "Paired against the same index without that component, so each dataset is compared only with itself. 5M lookups per run, 3 query seeds, performance-core QoS.",
            f"The band is this host measuring one index twice: two variants that build an identical structure differ by up to {noise:.0f} percent. Every learned bar is inside it.",
            "The joint design was exported to the weight format and run through the deployed root selector with its transform charge set to zero, so the selector chooses on",
            "probes alone and the warp's real cost lands in the clock. It takes the warp on seven of ten datasets, and loses 6 to 14 percent on exactly those seven."]):
        fig.text(0.105, 0.893 - i * 0.031, line, fontsize=9.8, color=MUTED, va="top")
    ax.barh([-99], [0], color=TEAL,
            label="Joint design: the learned bend + CSV virtual fences at the root, measured")
    fig.legend(loc="lower left", bbox_to_anchor=(0.102, 0.030), ncol=1, frameon=False, fontsize=FS_LEG,
               labelspacing=0.5, handlelength=1.5, handleheight=1.0, borderpad=0.0)
    fig.text(0.105, 0.010, "Source: results/aidb_final/root_analysis.json  ·  generated by results/aidb/throughput_by_dataset.py",
             fontsize=8, color=MUTED, va="bottom")

out = HERE / ("throughput_by_dataset_slide.png" if SLIDE else "throughput_by_dataset.png")
fig.savefig(out, facecolor="white")
print("wrote", out, f"(noise {noise:.1f}%)")
for d in ORDER:
    r = ra[d + "_uniform"]
    print(f"  {d:<9} " + "  ".join(f"{k.split('_',2)[-1]:<14}{100*(r[k]['speedup']-1):+6.1f}%" for k, _, _, _ in SERIES))
