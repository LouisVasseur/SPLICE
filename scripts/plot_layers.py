#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Every index with T, then G, then V added: one panel per base index, one line per dataset.

  scripts/plot_layers.py OUT.png --panel 'TITLE=RUNS:base,t,gt,gtv' [--panel ...] [--note TEXT]

Each --panel names its title, the runs.tsv it reads (run_parallel.sh's) and its four index names in ladder order;
RUNS may be a comma-free list joined by '+' (rows are pooled; the median over repeats is plotted), so a panel's base
can come from an earlier run of the same binary and concurrency. Panels share one log axis.
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DATASETS = ["fb", "genome", "osm", "planet", "books", "covid", "libio", "history", "stack", "wise"]
COLOURS = dict(zip(DATASETS, ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                              "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#5f6b77"]))
STEPS = [("base", ""), ("+ T", ""), ("+ G", ""), ("+ V", "")]
INK, MUTED, GRID = "#1f2328", "#57606a", "#e6e9ed"


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "Arial", "Helvetica Neue", "Helvetica"):
        if name in have:
            return name
    return "DejaVu Sans"


def load(paths):
    v = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["throughput_ops"]:
                    v[(r["dataset"], r["index"])].append(float(r["throughput_ops"]) / 1e6)
    return {k: statistics.median(x) for k, x in v.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    panels = []
    for spec in a.panel:
        title, rest = spec.split("=", 1)
        runs, names = rest.rsplit(":", 1)
        panels.append((title, load(runs.split("+")), names.split(",")))

    plt.rcParams.update({"font.family": font(), "font.size": 13, "axes.edgecolor": "#afb8c1",
                         "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED})
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n + 0.6, 6.4), dpi=200, sharey=True)
    fig.patch.set_facecolor("white")
    axes = [axes] if n == 1 else list(axes)
    shown = set()
    for ax, (title, val, names) in zip(axes, panels):
        xs = list(range(len(names)))
        for d in DATASETS:
            pts = [(x, val.get((d, nm))) for x, nm in zip(xs, names)]
            pts = [(x, y) for x, y in pts if y]
            if len(pts) < 2:
                continue
            shown.add(d)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOURS[d], lw=2.0, solid_capstyle="round", zorder=3)
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, color=COLOURS[d], edgecolor="white",
                       linewidth=1.3, zorder=4)
        ax.set_yscale("log", base=2)
        ax.set_xlim(-0.35, len(names) - 0.65)
        ax.set_xticks(xs)
        ax.set_xticklabels([s for s, _ in STEPS][:len(names)], fontsize=13, color=INK)
        ax.tick_params(axis="x", length=0, pad=8)
        ax.tick_params(axis="y", length=0, labelsize=12)
        ax.grid(axis="y", color=GRID, lw=1, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=15, fontweight="bold", color=INK, pad=12)
    ticks = [1, 2, 4, 8, 16, 32]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([str(t) for t in ticks])
    axes[0].minorticks_off()
    axes[0].set_ylim(1.0, 40)
    axes[0].set_ylabel("M lookups / s (log scale)", color=MUTED, fontsize=13, labelpad=10)
    handles = [Line2D([0], [0], color=COLOURS[d], lw=2.4, marker="o", markersize=6, markeredgecolor="white")
               for d in DATASETS if d in shown]
    fig.legend(handles, [d for d in DATASETS if d in shown], loc="upper center", ncol=10, frameon=False,
               fontsize=12.5, bbox_to_anchor=(0.5, 0.995), handlelength=1.6, columnspacing=1.4,
               labelcolor=INK)
    if a.note:
        fig.text(0.055, 0.018, a.note.replace(" | ", "\n"), fontsize=11, color=MUTED, ha="left", va="bottom",
                 linespacing=1.5)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.2 if "|" in a.note else 0.14, wspace=0.12)
    fig.savefig(a.out, facecolor="white")
    if a.out.endswith(".png"):
        fig.savefig(a.out[:-4] + ".pdf", facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
