#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The meeting-5 results figure (F54) for each workload: the write-capable indexes as bars, each at its own memory and its best
setting, then SPLICE one part at a time (root cells, + T, + G, + V) as one line per dataset.

  scripts/plot_write_ladder.py RESULTS_DIR OUT_DIR

RESULTS_DIR holds the batches copied from Jed (results/jed200M_write_2026-11-21): batch16 is the ladder (one pass, one node),
batch20p1..p3 G's period rule on genome (three passes, another node; scaled to batch16 by the full index without the rule, which
both batches ran), the others the baselines. Writes F62..F65 (png and pdf) into OUT_DIR.
"""
import csv
import math
import os
import re
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

DATASETS = ["fb", "genome", "osm", "planet", "books", "covid", "libio", "history", "stack", "wise"]
COLOURS = dict(zip(DATASETS, ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                              "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#5f6b77"]))
INK, MUTED, GRID, PANEL = "#1f2328", "#57606a", "#e6e9ed", "#f6f8fa"
RE_TP = re.compile(r"Throughput = ([0-9.e+]+)")


def font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "Arial", "Helvetica Neue", "Helvetica"):  # Arial keeps bold; F54 was set in Inter in Figma
        if name in have:
            return name
    return "DejaVu Sans"


def load(root, dirs):
    """(dataset, full index name, mix) -> list of M ops/s; and the cells that timed out"""
    v, timeout = defaultdict(list), set()
    for d in dirs:
        p = os.path.join(root, d, "cells.tsv")
        if not os.path.exists(p):
            continue
        for row in csv.reader(open(p), delimiter="\t"):
            if len(row) < 5:
                continue
            if row[4] == "rc=124":
                timeout.add((row[0], row[1], row[2]))
            m = RE_TP.search(row[6]) if len(row) > 6 else None
            if row[4] == "rc=0" and m:
                v[(row[0], row[1], row[2])].append(float(m.group(1)) / 1e6)
    return v, timeout


def med(xs):
    return statistics.median(xs) if xs else None


def baselines(root):
    v, _ = load(root, ["pass2", "pass3", "batch1", "batch2", "batch3", "batch5", "batch7", "batch12", "batch12b", "batch12c",
                       "batch20b", "batch21"])
    by = defaultdict(list)
    for (d, n, mix), xs in v.items():
        by[(d, n.split("@")[0], mix)].extend(xs)
    out = {}
    for d in DATASETS:
        for mix in ("read", "upd50", "del50", "ins50"):
            def get(*names):
                vals = [med(by[(d, n, mix)]) for n in names if by.get((d, n, mix))]
                return max(vals) if vals else None
            out[(d, "ALEX", mix)] = get("alexu") if mix == "upd50" else get("alex")
            out[(d, "LIPP", mix)] = get("lipp")
            out[(d, "BLI", mix)] = get("bli", "bli16", "bli18", "bli20")
            out[(d, "SALI", mix)] = get("saliu" if mix.startswith("ins") else "sali")
            out[(d, "AFLI", mix)] = get("afli")
            out[(d, "DyTIS", mix)] = get("dytis")
            if mix == "read":
                out[(d, "RBS", mix)] = get("rbst28", "rbsstd28")
    return out


def step(name):
    if "AR_DEPTH=0" in name:
        return "root"
    if "TRANSFORM_G_E=0" in name:
        return "T"
    return "TG" if "AB_V=0" in name else "TGV"


def ladder(root):
    v, timeout = load(root, ["batch16"])
    lad, out_t = {}, set()
    for (d, n, mix), xs in v.items():
        if (mix == "ins50") != n.startswith("aw4") or "AB_BUDGET=20" not in n and "AB_BUDGET=30" not in n:
            continue
        if mix == "ins50" and "AB_BUDGET=30" not in n:
            continue
        lad[(d, step(n), mix)] = med(xs)
    for (d, n, mix) in timeout:
        out_t.add((d, step(n), mix))
    # G's period rule on genome: three passes on another node, scaled by the full index without the rule
    g, _ = load(root, ["batch20p1", "batch20p2", "batch20p3"])
    rule = {}
    for mix in ("read", "upd50", "del50", "ins50"):
        pick = lambda pref, vflag: med([x for (d, n, m), xs in g.items() if d == "genome" and m == mix and n.startswith(pref)
                                        and f"AB_V={vflag}" in n and ("AB_BUDGET=30" in n if mix == "ins50" else True) for x in xs])
        plain, gv, g0 = (pick("aw4" if mix == "ins50" else "ab4", 1), pick("gw4" if mix == "ins50" else "gp4", 1),
                         pick("gw4" if mix == "ins50" else "gp4", 0))
        ref = lad.get(("genome", "TGV", mix))
        if plain and gv and g0 and ref:
            s = ref / plain
            rule[mix] = (g0 * s, gv * s)
    return lad, out_t, rule


TITLES = {"read": ("read-only", "M lookups / s (log scale)", 20), "upd50": ("50% updates", "M operations / s (log scale)", 20),
          "del50": ("50% deletes", "M operations / s (log scale)", 20), "ins50": ("50% inserts", "M operations / s (log scale)", 30)}
FILES = {"read": "F62_write_ladder_read", "upd50": "F63_write_ladder_upd50", "del50": "F64_write_ladder_del50",
         "ins50": "F65_write_ladder_ins50"}


def spread(ys, gap):
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for _ in range(300):
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


def figure(mix, base, lad, out_t, rule, out_dir):
    title, ylab, budget = TITLES[mix]
    names = (["RBS"] if mix == "read" else []) + ["ALEX", "LIPP", "BLI", "SALI", "AFLI", "DyTIS"]
    labels = {"RBS": ("radix binary", "search, 2^28"), "ALEX": ("ALEX", "update: ours" if mix == "upd50" else ""),
              "LIPP": ("LIPP", ""), "BLI": ("BLI", "best setting"), "SALI": ("SALI", ""), "AFLI": ("AFLI", ""),
              "DyTIS": ("DyTIS", "")}
    names = [n for n in names if any(base.get((d, n, mix)) for d in DATASETS)]
    steps = ["root", "T", "TG", "TGV"]
    slab = {"root": ("root cells only", ""), "T": ("+ T", "directory"), "TG": ("+ G", "far keys aside,\nrule if it pays"),
            "TGV": ("+ V", "fitted lines")}
    ymin, ymax = (0.04 if any(k[2] == mix for k in out_t) else 0.125), 45
    plt.rcParams.update({"font.family": font(), "font.size": 13, "axes.edgecolor": "#afb8c1", "axes.linewidth": 1.0,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(16, 8.8), dpi=200)
    fig.patch.set_facecolor("white")
    nb = len(names)
    w = 0.8 / len(DATASETS)
    ax.axvspan(-0.55, nb - 0.45, color=PANEL, zorder=0, lw=0)
    below = []
    for i, n in enumerate(names):
        for j, d in enumerate(DATASETS):
            val = base.get((d, n, mix))
            if not val:
                continue
            x = i - 0.4 + w * (j + 0.5)
            if val < ymin:
                below.append(f"{n} {d} {val:.2f}")
                continue
            ax.bar([x], [val - ymin], bottom=ymin, width=w * 0.82, color=COLOURS[d], zorder=2, lw=0)
    x0 = nb + 0.35
    xs = [x0 + k for k in range(len(steps))]
    finals, timeouts = {}, []
    for d in DATASETS:
        pts = []
        for x, s in zip(xs, steps):
            v = lad.get((d, s, mix))
            if (d, s, mix) in out_t and v is None:
                timeouts.append((x, d))
                continue
            if v is not None:
                pts.append((x, v))
        solid = pts
        if d == "genome" and mix in rule:  # dashed: far keys aside only; solid: the rule the gate keeps
            gx = [(xs[2], rule[mix][0]), (xs[3], rule[mix][1])]
            dashed = [p for p in pts if p[0] >= xs[1]]
            ax.plot([p[0] for p in dashed], [p[1] for p in dashed], color=COLOURS[d], lw=1.8, ls=(0, (3, 2.5)), alpha=0.55, zorder=3)
            for x, y in dashed[1:]:
                ax.scatter([x], [y], s=46, color="white", edgecolor=COLOURS[d], linewidth=1.8, zorder=4)
            solid = [p for p in pts if p[0] <= xs[1]] + gx
        ax.plot([p[0] for p in solid], [p[1] for p in solid], color=COLOURS[d], lw=2.2, solid_capstyle="round", zorder=3)
        ax.scatter([p[0] for p in solid], [p[1] for p in solid], s=46, color=COLOURS[d], edgecolor="white", linewidth=1.6, zorder=4)
        if solid:
            finals[d] = solid[-1][1]
    for x, d in timeouts:  # a root cell that did not finish 100M operations in 45 minutes
        ax.scatter([x], [ymin * 1.08], s=60, marker="v", color=COLOURS[d], zorder=5)
        ax.annotate(f"{d}: did not finish\nin 45 min (< 0.04)", (x, ymin * 1.08), xytext=(10, 4), textcoords="offset points",
                    fontsize=11, color=COLOURS[d], va="bottom")
    ax.set_yscale("log", base=2)
    ticks = [t for t in (0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32) if t >= ymin]
    ax.set_yticks(ticks)
    ax.set_yticklabels([{0.0625: "0.06"}.get(t, f"{t:g}") for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylab, color=MUTED, fontsize=13, labelpad=10)
    ax.grid(axis="y", color=GRID, lw=1, zorder=1)
    ax.set_xlim(-0.6, xs[-1] + 1.15)
    ax.set_xticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=12)
    for i, n in enumerate(names):
        top, sub = labels[n]
        ax.annotate(top, (i, 0), xycoords=("data", "axes fraction"), xytext=(0, -14), textcoords="offset points", ha="center",
                    va="top", fontsize=15, fontweight="bold", color=INK)
        if sub:
            ax.annotate(sub, (i, 0), xycoords=("data", "axes fraction"), xytext=(0, -36), textcoords="offset points",
                        ha="center", va="top", fontsize=12, color=MUTED)
    for x, s in zip(xs, steps):
        top, sub = slab[s]
        ax.annotate(top, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -14), textcoords="offset points", ha="center",
                    va="top", fontsize=15, fontweight="bold", color=INK)
        ax.annotate(sub, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -36), textcoords="offset points", ha="center",
                    va="top", fontsize=12, color=MUTED)
    ax.annotate("Write-capable indexes, each at its own memory and best setting" if mix != "read" else
                "The strongest other indexes, each at its own memory and best setting",
                ((nb - 1) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, 8), textcoords="offset points", ha="center",
                va="bottom", fontsize=14, color=MUTED)
    ax.annotate(f"SPLICE at {budget} B/key, one part at a time", ((xs[0] + xs[-1]) / 2, 1), xycoords=("data", "axes fraction"),
                xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=14, color=MUTED)
    ds = [d for d in DATASETS if d in finals]
    last = [math.log10(finals[d]) for d in ds]
    lab = spread(last, 0.034 * math.log10(ymax / ymin))  # one label line, whatever the axis spans
    xl = xs[-1] + 0.2
    for d, y0, y1 in zip(ds, last, lab):
        ax.plot([xs[-1] + 0.05, xl - 0.03], [10 ** y0, 10 ** y1], color=COLOURS[d], lw=1.2, zorder=2)
        ax.text(xl, 10 ** y1, d, va="center", ha="left", fontsize=12.5, color=INK)
        ax.text(xl + 0.64, 10 ** y1, f"{finals[d]:.1f}", va="center", ha="left", fontsize=12.5, color=MUTED)
    if "genome" in finals and mix in rule:
        ax.text(xs[2] - 0.15, 1.7, "genome: G's period rule (solid);\ndashed: far keys aside only", fontsize=11.5,
                color=COLOURS["genome"], ha="left", va="top")
    notes = ["200M keys · GRE microbench, 100M operations a run, single thread, 4 KiB pages · bars: median over the runs of each "
             "index, which ran on several nodes; BLI at its fastest setting per dataset",
             "SPLICE: one node, one pass per step; genome's + G / + V with G's rule: 3 passes on a second node, scaled by the full "
             "index without the rule, run on both"]
    if mix == "ins50":
        notes[1] += " · inserts start from 100M bulk-loaded keys; GRE never reads an inserted key"
    if below:
        notes.append("below the axis: " + ", ".join(below))
    fig.text(0.055, 0.015, "\n".join(notes), fontsize=10.5, color=MUTED, ha="left", va="bottom", linespacing=1.55)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.93, bottom=0.215 if below else 0.2)
    out = os.path.join(out_dir, FILES[mix])
    fig.savefig(out + ".png", facecolor="white")
    fig.savefig(out + ".pdf", facecolor="white")
    plt.close(fig)
    return out + ".png", {d: round(finals[d], 2) for d in finals}


def main():
    root, out_dir = sys.argv[1], sys.argv[2]
    base = baselines(root)
    lad, out_t, rule = ladder(root)
    for mix in ("read", "upd50", "del50", "ins50"):
        path, fin = figure(mix, base, lad, out_t, rule, out_dir)
        print("wrote", path, fin)


if __name__ == "__main__":
    main()
