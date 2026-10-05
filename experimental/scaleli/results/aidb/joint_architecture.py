#!/usr/bin/env python3
"""The joint design: one objective, two blocks, alternating.

Needs matplotlib.  /usr/bin/python3 results/aidb/joint_architecture.py [--slide]
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
FS = 11.6 if SLIDE else 11.0
FSS = 10.2 if SLIDE else 9.6

def box(x, y, w, h, title, body, ec, fc="white", tc=None, fs=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.7,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3))
    ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="center", fontsize=(fs or FS) + 0.4,
            color=tc or ec, fontweight="semibold", zorder=4)
    ax.text(x + w / 2, y + h / 2 - 2.2, body, ha="center", va="center", fontsize=FSS,
            color=INK, zorder=4, linespacing=1.5)

def arr(p1, p2, colour=INK, lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, color=colour,
                                 linewidth=lw, zorder=5, shrinkA=4, shrinkB=4,
                                 connectionstyle=f"arc3,rad={rad}"))

# objective
ax.add_patch(FancyBboxPatch((6, 74), 88, 12, boxstyle="round,pad=0.7,rounding_size=1.2",
                            facecolor=SURF, edgecolor=GRID, linewidth=1.4, zorder=3))
ax.text(50, 82.6, "ONE OBJECTIVE", ha="center", va="center", fontsize=FS, color=MUTED,
        fontweight="semibold", zorder=4)
ax.text(50, 77.6, "minimise   Σᵢ ( a·f(xᵢ) + b − sᵢ )²      over a monotone warp f,  a virtual-point set V with |V| ≤ λ,  and a refitted line (a, b)",
        ha="center", va="center", fontsize=FS + 0.2, color=INK, zorder=4)

# the two blocks
box(11, 40, 30, 22, "BLOCK A  ·  the warp  (NFL)",
    "hold the targets s fixed, fit f\n\n"
    "f(x) = C₁·tanh(g₁x) + C₂·tanh(g₂x)\n"
    "grid over the gains, exact 3×3\nleast-squares solve for C and (a, b)\n\n"
    "C ≥ 0, g > 0  ⇒  order preserving", ORANGE)
box(59, 40, 30, 22, "BLOCK B  ·  the slack  (CSV)",
    "hold the warp f fixed, choose V\n\n"
    "CSV greedy insertion in z-space:\nbest gap, refit, repeat to budget λ\n\n"
    "targets become slot ranks s\nO(1) per candidate from running sums", BLUE)
arr((41.5, 56), (58.5, 56), TEAL, 2.0, rad=-0.30)
arr((58.5, 45), (41.5, 45), TEAL, 2.0, rad=-0.30)
ax.text(50, 58.6, "new targets s", ha="center", va="center", fontsize=FSS - 0.4, color=TEAL)
ax.text(50, 42.4, "new feature z = f(x)", ha="center", va="center", fontsize=FSS - 0.4, color=TEAL)
ax.text(50, 50.6, "alternate until\nthe loss stops\nfalling", ha="center", va="center",
        fontsize=FSS - 0.4, color=TEAL, linespacing=1.4, fontweight="semibold")

# in / out
box(4, 24, 20, 11, "IN", "the 489 region fences\nkey → rank", GRID, SURF, tc=MUTED)
box(76, 24, 20, 11, "OUT", "one root model\nz → slot → region", GRID, SURF, tc=MUTED)
arr((14, 35.5), (18, 39.4), MUTED, 1.6)
arr((82, 39.4), (86, 35.5), MUTED, 1.6)

# guard
ax.add_patch(FancyBboxPatch((28, 24), 44, 11, boxstyle="round,pad=0.7,rounding_size=1.2",
                            facecolor="#eafaf3", edgecolor=TEAL, linewidth=1.5, zorder=3))
ax.text(50, 32.2, "THE GUARD  ·  why it converges", ha="center", va="center", fontsize=FSS + 0.6,
        color=TEAL, fontweight="semibold", zorder=4)
ax.text(50, 27.6, "the greedy cannot warm start, so it restarts from V = ∅ each round;\n"
                  "keep the previous slot ranks whenever the fresh pass is worse",
        ha="center", va="center", fontsize=FSS - 0.4, color=INK, zorder=4, linespacing=1.5)

# what happened
ax.add_patch(FancyBboxPatch((6, 6), 88, 13, boxstyle="round,pad=0.7,rounding_size=1.2",
                            facecolor="white", edgecolor=GRID, linewidth=1.4, zorder=3))
ax.text(50, 16.4, "WHAT IT ACTUALLY DID", ha="center", va="center", fontsize=FSS + 0.8,
        color=MUTED, fontweight="semibold", zorder=4)
ax.text(50, 10.8, "It converges: on books the loss falls 283 → 222 → 134.   Root probes reach 2.1 – 2.7 on nine of ten datasets.   Space: 41 virtual fences where CSV alone needs 662.\n"
                  "But alternating buys at most 0.072 probes over simply running the warp then the greedy once, and rounds ≥ 2 contribute 0.0 % of planet's loss reduction.",
        ha="center", va="center", fontsize=FSS - 0.2, color=INK, zorder=4, linespacing=1.6)

ax.text(4.5, 96.5, "The joint design: optimise the warp and the virtual points together, not in sequence",
        fontsize=15.5, color=INK, fontweight="semibold", va="top")
ax.text(4.5, 91.5, "Block coordinate descent over the two axes of the same fit. Each block is exact given the other, so the loss cannot rise.",
        fontsize=FSS, color=MUTED, va="top")
ax.text(4.5, 2.6, "Prior art: Li et al., arXiv:2101.00808 (CSV's reference 16) states this objective and solves it by block coordinate descent; the model class and the root setting are what is new here.",
        fontsize=FSS - 1.2, color=MUTED, va="center")

out = HERE / ("joint_architecture_slide.png" if SLIDE else "joint_architecture.png")
fig.savefig(out, facecolor="white")
print("wrote", out)
