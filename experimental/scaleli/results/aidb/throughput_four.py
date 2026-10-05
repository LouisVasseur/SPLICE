#!/usr/bin/env python3
"""Throughput per lookup: NFL alone, CSV alone, the fused selector, and the joint design.

Needs matplotlib.  /usr/bin/python3 results/aidb/throughput_four.py [--slide]

Companion to probes_four.py: same four methods, same colours, same row order, so the two
figures can be read line by line. One shows the work removed, this one shows the time.

Sources: results/aidb_final/root_analysis.json (540 runs, 5M lookups x 3 seeds) for the first
three; results/aidb_jointrun/summary.json (90 runs, same settings, its own control) for the joint
design, which was exported to the weight format and run through the deployed root selector.
"""
import json, math, pathlib, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
S = HERE.parent.parent
ORANGE, BLUE, BLUE_L, TEAL = "#eb6834", "#2a78d6", "#86b6ef", "#1baf7a"
INK, MUTED, GRID, SURF, BAND = "#17242f", "#6b7480", "#d8d7d0", "#f4f4f1", "#e8e7e2"
SLIDE = "--slide" in sys.argv

ra = json.loads((S / "results/aidb_final/root_analysis.json").read_text())["per_dataset"]
jr = json.loads((S / "results/aidb_jointrun/summary.json").read_text())
ORDER = ["osm", "planet", "books", "wise", "stack", "genome", "libio", "history", "fb", "covid"]
noise = 100 * max(abs(ra[d + "_uniform"]["packed_rank_root_vf4"]["speedup"] /
                      ra[d + "_uniform"]["packed_rank_root_fusion"]["speedup"] - 1) for d in ORDER)
sp = lambda d, v: 100 * (ra[d + "_uniform"][v]["speedup"] - 1)

SERIES = [(lambda d: sp(d, "packed_rank_root_flow"), ORANGE, "NFL alone: learned root, transform offered", +0.30),
          (lambda d: sp(d, "packed_rank_vp10"), BLUE, "CSV alone: virtual points in the regions", +0.10),
          (lambda d: sp(d, "packed_rank_vp10_root_fusion"), BLUE_L, "Fused selector: CSV in the regions AND at the root", -0.10),
          (lambda d: 100 * (jr[d]["speedup"] - 1), TEAL, "Joint design: the learned bend + CSV virtual fences at the root", -0.30)]

fig = plt.figure(figsize=(13.33, 8.3) if SLIDE else (13.0, 8.3), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.105, 0.365, 0.880, 0.495]); ax.set_facecolor("white")
FS_DS, FS_BAR, FS_LEG, FS_TICK = (13.0, 9.8, 10.8, 11.0) if SLIDE else (11.5, 8.6, 9.8, 10)
h = 0.20
ax.axvspan(-noise, noise, color=BAND, zorder=0)
for fn, colour, label, off in SERIES:
    ys = [i + off for i in range(len(ORDER))]
    vals = [fn(d) for d in ORDER]
    ax.barh(ys, vals, height=h * 0.90, color=colour, label=label, zorder=3)
    for y, d, v in zip(ys, ORDER, vals):
        ax.text(v + (0.6 if v >= 0 else -0.6), y, f"{v:+.0f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=FS_BAR + 0.4, color=MUTED, zorder=4)
ax.axvline(0, color=INK, lw=1.4, zorder=2)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=FS_DS, color=INK)
ax.set_xlim(-22, 16); ax.set_xticks([-20, -10, 0, 10])
ax.set_xticklabels(["−20%", "−10%", "0", "+10%"], fontsize=FS_TICK, color=MUTED)
ax.set_xlabel("change in throughput against the same index without that component   (further right is faster)",
              fontsize=FS_TICK + 0.5, color=MUTED, labelpad=9)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=1); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0, pad=10); ax.tick_params(axis="x", length=0)
ax.set_ylim(-0.62, len(ORDER) - 0.38)
ax.text(noise + 0.8, -0.44, f"host noise floor, ±{noise:.0f}%", fontsize=FS_BAR, color=MUTED, va="center")

fig.legend(loc="lower left", bbox_to_anchor=(0.102, 0.250), ncol=2, frameon=False,
           fontsize=FS_LEG, labelspacing=0.42, handlelength=1.5, handleheight=1.0, borderpad=0.0)

fig.patches.append(plt.Rectangle((0.102, 0.035), 0.883, 0.175, transform=fig.transFigure,
                                 facecolor=SURF, edgecolor=GRID, lw=1.0, zorder=0))
fig.text(0.117, 0.192, "HOW THROUGHPUT IS SCORED", fontsize=FS_LEG - 0.2, color=MUTED,
         fontweight="semibold", va="top")
for i, line in enumerate([
        "Lookups completed per second, from the timed pass with no counters running. 5M lookups per run after a 500k warm-up, pinned to performance cores.",
        "Paired, not absolute: each variant is compared with the control on the SAME dataset and the SAME query trace, then the per-seed ratios are combined by geometric mean over 3 seeds.",
        "The band is measured, not assumed: two variants that build a bit-identical structure differ by up to 8% on this host, so nothing inside it is resolvable.",
        "The joint design comes from a separate 90-run sweep with its own control, paired the same way. Its transform charge was set to zero, so the selector chose on probes and paid the real cost in time.",
        "Read against the probe figure: the same methods that remove 36 to 44 percent of the work land inside the noise band here, and the joint design is measurably slower."]):
    fig.text(0.117, 0.168 - i * 0.0255, line, fontsize=FS_LEG - 1.6, color=INK, va="top")

if not SLIDE:
    fig.text(0.105, 0.968, "Time per lookup: each method alone, the fused selector, and the joint design",
             fontsize=15.5, color=INK, fontweight="semibold", va="top")
    fig.text(0.105, 0.925, "Ten 2M uniform samples, identical query traces. Same rows and colours as the probe figure.",
             fontsize=9.8, color=MUTED, va="top")
    fig.text(0.105, 0.010, "Source: results/aidb_final/root_analysis.json and results/aidb_jointrun/summary.json  ·  generated by results/aidb/throughput_four.py",
             fontsize=8, color=MUTED, va="bottom")

out = HERE / ("throughput_four_slide.png" if SLIDE else "throughput_four.png")
fig.savefig(out, facecolor="white")
print("wrote", out, f"(noise {noise:.1f}%)")
gm = lambda xs: math.exp(statistics.mean(math.log(1 + x / 100) for x in xs))
for fn, _, label, _ in SERIES:
    vals = [fn(d) for d in ORDER]
    print(f"  {label[:44]:<45} pooled {100*(gm(vals)-1):+5.1f}%   range {min(vals):+5.1f} to {max(vals):+5.1f}")
