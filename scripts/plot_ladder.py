#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SPLICE's attribution ladder as one line per dataset: GRE's indexes, then SPLICE's T, G and V added one at a time.

  scripts/plot_ladder.py RUNS_TSV OUT.png [--steps pgm,spl_t,spl_gt,splice] [--gre btree,alex,lipp] [--head RUNS]
                         [--note TEXT]

RUNS_TSV is run_parallel.sh's runs.tsv (dataset, index, repeat, throughput_ops, ...); the median over repeats is
plotted. --gre adds GRE's other indexes as columns before the ladder. --head RUNS takes GRE's indexes, the base step
and the last step (full SPLICE) from a one-at-a-time run instead; the middle steps then exist only in RUNS_TSV (a
concurrent ladder) and are drawn hollow, scaled by the last step's one-at-a-time / concurrent ratio per dataset.
"""
import argparse
import csv
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

DATASETS = ["fb", "genome", "osm", "planet", "books", "covid", "libio", "history", "stack", "wise"]
# one fixed hue per dataset (Tableau 10 order), so a dataset keeps its colour across figures
COLOURS = dict(zip(DATASETS, ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                              "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#5f6b77"]))
STEP_LABELS = {
    "btree": ("B+tree", ""),
    "alex": ("ALEX", ""),
    "lipp": ("LIPP", ""),
    "pgm": ("PGM", "same cutting algorithm"),
    "spl_t": ("+ T", "SPLICE structure, T only"),
    "spl_gt": ("+ G", "remove outliers"),
    "spl_tv": ("+ V", "add slots"),
    "splice": ("+ V", "add slots = SPLICE"),
    "splice3": ("+ V", "add slots = SPLICE"),
    "s3_t": ("+ T", "segments + top"),
    "s3_gt": ("+ G", "remove outliers"),
}
INK, MUTED, GRID = "#1f2328", "#57606a", "#e6e9ed"


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "Arial", "Helvetica Neue", "Helvetica"):
        if name in have:
            return name
    return "DejaVu Sans"


def load(path, steps):
    v = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["index"] in steps and r["throughput_ops"]:
                v[(r["dataset"], r["index"])].append(float(r["throughput_ops"]) / 1e6)
    return {k: statistics.median(x) for k, x in v.items()}


def spread(ys, gap):
    """Label positions (log10 space): the values, pushed apart to at least `gap`, as close to the values as possible."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for _ in range(200):
        moved = False
        for j in range(1, len(pos)):
            if pos[j] - pos[j - 1] < gap:
                d = (gap - (pos[j] - pos[j - 1])) / 2
                pos[j - 1] -= d
                pos[j] += d
                moved = True
        if not moved:
            break
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("out")
    ap.add_argument("--steps", default="pgm,spl_t,spl_gt,splice")
    ap.add_argument("--gre", default="")
    ap.add_argument("--head", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    steps = a.steps.split(",")
    gre = [g for g in a.gre.split(",") if g]
    lad = load(a.runs, steps + gre)
    head = load(a.head, steps + gre) if a.head else lad
    ds = [d for d in DATASETS if all((d, s) in lad for s in steps) and (d, steps[-1]) in head and (d, steps[0]) in head]
    cols = gre + steps
    derived = set(steps[1:-1]) if a.head else set()

    def value(d, c):  # (value, derived?) or None
        if c in gre or c in (steps[0], steps[-1]):
            return (head[(d, c)], False) if (d, c) in head else None
        if not a.head:
            return lad[(d, c)], False
        return lad[(d, c)] * head[(d, steps[-1])] / lad[(d, steps[-1])], True

    plt.rcParams.update({"font.family": font(), "font.size": 13, "axes.edgecolor": "#afb8c1",
                         "axes.linewidth": 1.0, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(13.5 if gre else 12, 7.2), dpi=200)
    fig.patch.set_facecolor("white")
    xs = list(range(len(cols)))
    k0 = len(gre)  # the ladder starts here
    if gre:  # GRE's indexes, the base included
        ax.axvspan(-0.5, k0 + 0.5, color="#f6f8fa", zorder=0, lw=0)
    for j, d in enumerate(ds):
        pts = [(x, value(d, c)) for x, c in zip(xs, cols)]
        # GRE's other indexes: dots only, spread a little sideways so that close values stay visible
        jit = (j - (len(ds) - 1) / 2) * 0.03
        pts = [(x + (jit if x < k0 else 0), v) for x, v in pts]
        pl = [(x, v[0], v[1]) for x, v in pts[k0:] if v]
        ax.plot([p[0] for p in pl], [p[1] for p in pl], color=COLOURS[d], lw=2.2, solid_capstyle="round", zorder=3)
        for x, y, der in [(x, v[0], v[1]) for x, v in pts if v]:
            ax.scatter([x], [y], s=46, color="white" if der else COLOURS[d], edgecolor=COLOURS[d] if der else "white",
                       linewidth=1.8 if der else 1.6, zorder=4)

    ax.set_yscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(1.0, 40)
    ax.set_ylabel("M lookups / s (log scale)", color=MUTED, fontsize=13, labelpad=10)
    ax.grid(axis="y", color=GRID, lw=1, zorder=1)
    ax.set_xlim(-0.5, len(cols) - 1 + 0.95)
    ax.set_xticks(xs)
    ax.set_xticklabels([""] * len(xs))
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0, labelsize=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for x, c in zip(xs, cols):
        top, sub = STEP_LABELS.get(c, (c, ""))
        ax.annotate(top, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top", fontsize=15, fontweight="bold", color=INK)
        ax.annotate(sub, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -36), textcoords="offset points",
                    ha="center", va="top", fontsize=12, color=MUTED)
    if gre:
        for x0, x1, t in ((-0.4, k0 + 0.4, "GRE's indexes"), (k0 + 0.6, len(cols) - 0.6, "SPLICE, one part at a time")):
            ax.annotate(t, ((x0 + x1) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=13, color=MUTED)

    # direct labels at the right end: name and final value in ink, a leader in the dataset's colour
    import math
    last = [math.log10(value(d, cols[-1])[0]) for d in ds]
    lab = spread(last, 0.058)
    xl = xs[-1] + 0.16
    for d, y0, y1 in zip(ds, last, lab):
        ax.plot([xs[-1] + 0.04, xl - 0.03], [10 ** y0, 10 ** y1], color=COLOURS[d], lw=1.2, zorder=2)
        ax.text(xl, 10 ** y1, f"{d}", va="center", ha="left", fontsize=12.5, color=INK)
        ax.text(xl + 0.46, 10 ** y1, f"{10 ** y0:.1f}", va="center", ha="left", fontsize=12.5, color=MUTED)

    if a.note:  # '|' starts a new line
        fig.text(0.07, 0.018, a.note.replace(" | ", "\n"), fontsize=11, color=MUTED, ha="left", va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.2 if "|" in a.note else 0.17)
    fig.savefig(a.out, facecolor="white")
    if a.out.endswith(".png"):
        fig.savefig(a.out[:-4] + ".pdf", facecolor="white")
    print("wrote", a.out, "datasets:", " ".join(ds))


if __name__ == "__main__":
    main()
