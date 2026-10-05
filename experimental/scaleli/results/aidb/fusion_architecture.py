#!/usr/bin/env python3
"""Architecture of the fusion: where each method plugs into the lookup path.

Needs matplotlib.  /usr/bin/python3 results/aidb/fusion_architecture.py [--slide]
"""
import sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = pathlib.Path(__file__).resolve().parent
ORANGE, BLUE, TEAL, GRAY = "#eb6834", "#2a78d6", "#1baf7a", "#b9c0c6"
INK, MUTED, GRID, SURF = "#17242f", "#6b7480", "#d8d7d0", "#f4f4f1"
SLIDE = "--slide" in sys.argv

fig = plt.figure(figsize=(13.33, 7.5) if SLIDE else (13.0, 7.2), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, text, fc=SURF, ec=GRID, tc=INK, fs=11, weight="normal", lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, fontweight=weight, linespacing=1.45)

def arrow(x1, y1, x2, y2, colour=INK, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 color=colour, linewidth=lw, zorder=2,
                                 shrinkA=2, shrinkB=2))

FS = 11.5 if SLIDE else 10.8
FSS = 10 if SLIDE else 9.4

# ---- the lookup path, left to right
Y, H = 46, 11
stages = [(4.5, 9.5, "key"), (18, 15, "ROOT model\nover region fences"),
          (39, 13, "REGION model\nover the region's keys"), (58, 12, "BLOCK\ndecode + search"),
          (75, 9.5, "value")]
for x, w, label in stages:
    fill = SURF if label in ("key", "value") else "white"
    box(x, Y, w, H, label, fc=fill, fs=FS, weight="semibold" if "model" in label else "normal")
for i in range(len(stages) - 1):
    x1 = stages[i][0] + stages[i][1]; x2 = stages[i + 1][0]
    arrow(x1 + 0.6, Y + H / 2, x2 - 0.6, Y + H / 2)
ax.text(stages[1][0] + stages[1][1] / 2, Y - 2.4, "8.96 probes → 2.1", ha="center", va="top",
        fontsize=FSS, color=MUTED)
ax.text(stages[2][0] + stages[2][1] / 2, Y - 2.4, "2.2 – 5.1 probes", ha="center", va="top",
        fontsize=FSS, color=MUTED)
ax.text(stages[3][0] + stages[3][1] / 2, Y - 2.4, "8.0 comparisons", ha="center", va="top",
        fontsize=FSS, color=MUTED)

# ---- NFL above: bends the input
ax.add_patch(FancyBboxPatch((15, 65), 39, 13, boxstyle="round,pad=0.6,rounding_size=1.2",
                            facecolor="#fdf0ea", edgecolor=ORANGE, linewidth=1.6, zorder=3))
ax.text(34.5, 74.9, "NFL  ·  bends the INPUT", ha="center", va="center", fontsize=FS + 0.6,
        color=ORANGE, fontweight="semibold", zorder=4)
ax.text(34.5, 69.6, "replace the key with a learned monotone warp  z = f(key)\n"
                    "feeds the model; records stay in key order", ha="center", va="center",
        fontsize=FSS, color=INK, zorder=4, linespacing=1.5)
for x in (25.5, 45.5):
    arrow(x, 64.4, x, Y + H + 0.8, colour=ORANGE, lw=1.6)

# ---- CSV below: bends the output
ax.add_patch(FancyBboxPatch((15, 18), 39, 14, boxstyle="round,pad=0.6,rounding_size=1.2",
                            facecolor="#eaf1fb", edgecolor=BLUE, linewidth=1.6, zorder=3))
ax.text(34.5, 28.6, "CSV  ·  bends the OUTPUT", ha="center", va="center", fontsize=FS + 0.6,
        color=BLUE, fontweight="semibold", zorder=4)
ax.text(34.5, 23.0, "insert virtual points so the model's targets become\n"
                    "slot ranks instead of ranks; they hold no record",
        ha="center", va="center", fontsize=FSS, color=INK, zorder=4, linespacing=1.5)
for x in (25.5, 45.5):
    arrow(x, 32.6, x, Y - 0.8, colour=BLUE, lw=1.6)

# ---- the selector
ax.add_patch(FancyBboxPatch((60, 12), 36, 24, boxstyle="round,pad=0.8,rounding_size=1.2",
                            facecolor="white", edgecolor=TEAL, linewidth=1.8, zorder=3))
ax.text(78, 33.2, "THE SELECTOR  ·  run per model, at build time", ha="center", va="center",
        fontsize=FS, color=TEAL, fontweight="semibold", zorder=4)
cw, ch = 13.5, 5.2
for j, (tx, targ) in enumerate(((64.5, "ranks"), (80.5, "slot ranks"))):
    for i, (ty, feat, col) in enumerate(((23.6, "raw key", INK), (17.2, "z = f(key)", ORANGE))):
        both = (j == 1 and i == 1)
        box(tx, ty, cw, ch, f"{feat}\n→ {targ}", fc="#eafaf3" if both else SURF,
            ec=TEAL if both else GRID, tc=col, fs=FSS - 0.6, lw=1.6 if both else 1.0)
ax.text(78, 13.6, "score each by the probes the real lookup spends, + a charge for f\nkeep the cheapest;"
                  " at the root only if it beats binary search",
        ha="center", va="center", fontsize=FSS - 0.4, color=MUTED, zorder=4, linespacing=1.5)
ax.add_patch(FancyArrowPatch((60.5, 30), (52.6, 46.5), arrowstyle="-|>", mutation_scale=14,
                             color=TEAL, linewidth=1.7, zorder=5, shrinkA=3, shrinkB=3,
                             connectionstyle="arc3,rad=-0.28"))
ax.text(57.0, 37.0, "chooses\neach model", ha="center", va="center", fontsize=FSS - 0.6,
        color=TEAL, linespacing=1.4, zorder=6)

ax.text(4.5, 92.5, "The fusion, as architecture: two methods, two axes, one selector",
        fontsize=15.5, color=INK, fontweight="semibold", va="top")
for i, line in enumerate([
        "Every model in the index, the root and each region, is built the same way: pick a feature, pick targets, fit one line. NFL supplies an alternative feature, CSV supplies alternative targets.",
        "The aqua cell is the fusion proper: warped feature AND smoothed targets in one model. It is a real candidate, so synergy is measurable rather than assumed.",
        "Decided once at bulk load and reused on compaction. Predictions are hints only, so a wrong choice costs probes, never correctness."]):
    ax.text(4.5, 88.4 - i * 3.1, line, fontsize=FSS - 0.2, color=MUTED, va="top")
ax.text(4.5, 5.0, "Probe figures are measured medians over the ten 2M uniform samples (results/aidb_final). "
                  "The in-block stage is 8.0 comparisons that the probe counters do not include.",
        fontsize=FSS - 1.4, color=MUTED, va="center")

out = HERE / ("fusion_architecture_slide.png" if SLIDE else "fusion_architecture.png")
fig.savefig(out, facecolor="white")
print("wrote", out)
