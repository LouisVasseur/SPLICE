#!/usr/bin/env python3
"""Throughput bar chart: baseline, each method alone, and the fused design.

Needs matplotlib (outside the stdlib-only pipeline).
    /usr/bin/python3 results/aidb/throughput_bars.py [--slide]

Source: results/aidb_final/root_analysis.json (540 paired runs, 5M lookups each, 3 query seeds,
performance-core QoS, identical query traces). Bars are the MEDIAN over the ten uniform 2M samples;
the whisker is the spread ACROSS DATASETS, which is dataset variation, not uncertainty.

The shaded band is the host's measurement noise, derived not assumed: packed_rank_root_vf4 and
packed_rank_root_fusion select an identical structure on every sample, so the difference between
those two measurements is this machine measuring one index twice.

NOTE: no external index (ALEX, PGM, LIPP) was ever benchmarked in this workspace. README.md lists
the ALEX and PGM adapters as "written, opt-in, not compiled or benchmarked here". The no-index
baseline below is a plain binary search over the uncompressed sorted array, which we did run.
"""
import json, math, pathlib, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
S = HERE.parent.parent
GRAY, INKBAR, ORANGE, BLUE, TEAL, PEND = "#9aa5ad", "#4a5568", "#eb6834", "#2a78d6", "#1baf7a", "#c3c2b7"
INK, MUTED, GRID, BAND = "#17242f", "#6b7480", "#d8d7d0", "#ebeae4"
SLIDE = "--slide" in sys.argv

ra = json.loads((S / "results/aidb_final/root_analysis.json").read_text())["per_dataset"]
DS = [d for d in sorted(ra) if d.endswith("_uniform")]
gm = lambda xs: math.exp(statistics.mean(math.log(x) for x in xs))
noise = max(abs(ra[d]["packed_rank_root_vf4"]["speedup"] / ra[d]["packed_rank_root_fusion"]["speedup"] - 1) for d in DS)

BARS = [
    ("sorted_vector", "No index\nplain binary search", GRAY),
    ("packed_rank", "Control\npacked index", INKBAR),
    ("packed_rank_root_flow", "NFL only\ntransform at root", ORANGE),
    ("packed_rank_vp10", "CSV only\npoints in regions", BLUE),
    ("packed_rank_root_vf4", "CSV only\nfences at root", TEAL),
    ("packed_rank_vp10_root_fusion", "FUSED\nCSV regions + root", BLUE),
    (None, "Joint design\nnot yet built", PEND),
]
vals = []
for key, label, colour in BARS:
    if key is None:
        vals.append((label, colour, None, None, None, None)); continue
    mo = [ra[d][key]["mops"] for d in DS if key in ra[d]]
    sp = [ra[d][key]["speedup"] for d in DS if key in ra[d]]
    vals.append((label, colour, statistics.median(mo), min(mo), max(mo), gm(sp)))

ctrl = next(v[2] for v in vals if v[0].startswith("Control"))
fig = plt.figure(figsize=(13.33, 7.5) if SLIDE else (12.6, 7.0), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.075, 0.235, 0.905, 0.50 if SLIDE else 0.52]); ax.set_facecolor("white")
ax.axhspan(ctrl * (1 - noise), ctrl * (1 + noise), color=BAND, zorder=0)
ax.axhline(ctrl, color=INK, lw=1.2, ls=(0, (5, 4)), zorder=2)
FS = 11.5 if SLIDE else 10.5
for i, (label, colour, med, lo, hi, sp) in enumerate(vals):
    if med is None:
        ax.bar(i, ctrl, color="white", edgecolor=PEND, lw=2.2, hatch="//", zorder=3)
        ax.text(i, ctrl * 0.5, "probes only\nso far", ha="center", va="center",
                fontsize=FS - 1.2, color=MUTED, style="italic")
        continue
    ax.bar(i, med, color=colour, zorder=3, width=0.66)
    ax.plot([i, i], [lo, hi], color=INK, lw=1.6, zorder=5)
    ax.plot([i - 0.09, i + 0.09], [lo, lo], color=INK, lw=1.6, zorder=5)
    ax.plot([i - 0.09, i + 0.09], [hi, hi], color=INK, lw=1.6, zorder=5)
    ax.text(i, hi + 0.12, f"{med:.2f}", ha="center", va="bottom", fontsize=FS + 0.5,
            color=INK, fontweight="semibold")
    if sp is not None and abs(sp - 1) > 1e-9:
        ax.text(i, hi + 0.42, f"x{sp:.2f}", ha="center", va="bottom", fontsize=FS - 1, color=MUTED)

ax.set_xticks(range(len(vals)))
ax.set_xticklabels([v[0] for v in vals], fontsize=FS, color=INK)
ax.set_ylabel("throughput, million lookups per second", fontsize=FS, color=MUTED)
ax.set_ylim(0, 7.0)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.spines["left"].set_color(GRID)
ax.tick_params(length=0, labelsize=FS - 0.5, colors=MUTED)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=1); ax.set_axisbelow(True)
ax.text(len(vals) - 0.45, ctrl * (1 + noise) + 0.06, f"host noise floor, ±{100*noise:.0f}%",
        ha="right", va="bottom", fontsize=FS - 1.4, color=MUTED)

head = 0.955
if not SLIDE:
    fig.text(0.075, head, "Throughput: only removing the index beats the index", fontsize=15.5,
             color=INK, fontweight="semibold", va="top")
    head = 0.893
for i, line in enumerate([
        "Median over the ten uniform 2M samples, 5M lookups per run, 3 query seeds, performance-core QoS, identical query traces.",
        "Whiskers are the spread across datasets, not uncertainty. The shaded band is this host measuring one index twice:",
        f"two variants that build an identical structure differ by up to {100*noise:.0f} percent, so every learned bar sits inside it.",
        "No external index was run here. The ALEX and PGM adapters exist in the tree but were never compiled, so the baseline is plain binary search."]):
    fig.text(0.075, head - i * 0.031, line, fontsize=9.7, color=MUTED, va="top")
fig.text(0.075, 0.012, "Source: results/aidb_final/root_analysis.json  ·  generated by results/aidb/throughput_bars.py",
         fontsize=8, color=MUTED, va="bottom")

out = HERE / ("throughput_bars_slide.png" if SLIDE else "throughput_bars.png")
fig.savefig(out, facecolor="white")
print("wrote", out, f"(noise {100*noise:.1f}%)")
for label, _, med, lo, hi, sp in vals:
    if med is None: print(f"  {label.replace(chr(10),' '):<34} pending"); continue
    print(f"  {label.replace(chr(10),' '):<34} median {med:5.3f} Mops  range {lo:.3f}-{hi:.3f}  paired x{sp:.3f}")
