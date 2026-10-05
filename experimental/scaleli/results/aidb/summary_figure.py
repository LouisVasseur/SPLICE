#!/usr/bin/env python3
"""One summary figure of the three measured reductions, built from the runs we actually made.

NOTE: this is the only script in the workspace that needs a third party library (matplotlib).
It sits under results/ and is outside the stdlib-only pipeline; nothing else imports it.
Run it with an interpreter that already has matplotlib, e.g.
    /usr/bin/python3 results/aidb/summary_figure.py

Sources, all from the corrected run of 2026-09-22:
  * results/aidb_flowv2/sweep/results.jsonl   140 runs, 10 datasets x 7 variants x 2 query seeds,
                                              2M uniform samples, identical traces, retrained flows.
  * results/aidb_flowv2/flowcheck_summary.json  tail conflict degree, raw keys vs the retrained
                                              unconstrained flow (NFL Definition 3.2).
Series:
  fence probes  packed_rank            -> packed_rank_vp10          (CSV virtual points per region)
  root probes   packed_rank (binary)   -> packed_rank_root_fusion   (CSV virtual fences at the root)
  NFL criterion raw keys               -> retrained flow            (paper metric, not index work)
"""
import json, collections, statistics, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
S = HERE.parent.parent
BLUE, TEAL, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
INK, MUTED, GRID = "#17242f", "#6b7480", "#d8d7d0"

rows = [json.loads(l) for l in (S / "results/aidb_flowv2/sweep/results.jsonl").open() if l.strip()]
by = collections.defaultdict(list)
for r in rows:
    by[(r["dataset"], r["variant"])].append(r)
fc = json.loads((S / "results/aidb_flowv2/flowcheck_summary.json").read_text())
work = lambda rs, k: statistics.mean(x["work_counters"][k] / x["operations"] for x in rs)

data = {}
for d in fc:
    c, vp, rf = by[(d, "packed_rank")], by[(d, "packed_rank_vp10")], by[(d, "packed_rank_root_fusion")]
    fence_a, fence_b = work(c, "fence_probes"), work(vp, "fence_probes")
    root_a, root_b = work(c, "root_probes"), work(rf, "root_probes")
    tcd_a, tcd_b = fc[d]["raw"], fc[d]["free"]
    data[d] = {
        "fence": (fence_a, fence_b, 100 * (fence_a - fence_b) / fence_a),
        "root": (root_a, root_b, 100 * (root_a - root_b) / root_a),
        "tcd": (tcd_a, tcd_b, 100 * (tcd_a - tcd_b) / tcd_a),
    }
order = sorted(data, key=lambda d: data[d]["fence"][2] + data[d]["root"][2])

SLIDE = "--slide" in sys.argv   # drop the title and shrink the prose: the slide supplies its own heading

if SLIDE:
    fig = plt.figure(figsize=(13.33, 7.5), dpi=200)
    ax = fig.add_axes([0.088, 0.245, 0.900, 0.685])
    FS_DS, FS_BAR, FS_LEG, FS_TICK = 13.5, 10.2, 11.2, 11.5
else:
    fig = plt.figure(figsize=(12.8, 7.4), dpi=200)
    ax = fig.add_axes([0.088, 0.215, 0.902, 0.60])
    FS_DS, FS_BAR, FS_LEG, FS_TICK = 11.5, 8.4, 9.8, 10
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
h = 0.26
series = [("root", TEAL, "Root probes per lookup   \u00b7   CSV virtual fences at the root", +h),
          ("fence", BLUE, "Exact fence probes per lookup   \u00b7   CSV virtual points per region", 0.0),
          ("tcd", ORANGE, "Tail conflict degree, NFL Def. 3.2   \u00b7   retrained transform", -h)]
for key, colour, label, off in series:
    ys = [i + off for i in range(len(order))]
    vals = [data[d][key][2] for d in order]
    ax.barh(ys, vals, height=h * 0.90, color=colour, label=label, zorder=3)
    for y, d, v in zip(ys, order, vals):
        a, b, _ = data[d][key]
        fmt = f"{a:.2f}\u2192{b:.2f}" if key != "tcd" else f"{a:g}\u2192{b:g}"
        ax.text(v + 1.2, y, f"{v:.0f}%  ({fmt})" if v >= 1 else f"no change  ({fmt})",
                va="center", ha="left", fontsize=FS_BAR, color=MUTED, zorder=4)

ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=FS_DS, color=INK)
ax.set_xlim(0, 132); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=FS_TICK, color=MUTED)
ax.set_xlabel("reduction against the same index without that component   (further right is better)",
              fontsize=FS_TICK + 0.5, color=MUTED, labelpad=9)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=0)
ax.set_ylim(-0.62, len(order) - 0.38)

if SLIDE:
    fig.legend(loc="lower left", bbox_to_anchor=(0.085, 0.085), ncol=1, frameon=False,
               fontsize=FS_LEG, labelspacing=0.45, handlelength=1.5, handleheight=1.0, borderpad=0.0)
    fig.text(0.085, 0.035,
             "Orange is NFL's own metric only: inside the index that same flow raises fence probes 9 to 192 percent, and is never chosen.",
             fontsize=11, color=ORANGE, va="bottom", fontweight="semibold")
    fig.text(0.085, 0.008,
             "Ten AIDB datasets, 2M uniform samples, identical query traces, two query seeds. Probe counts are deterministic software counters, not timings.",
             fontsize=9.4, color=MUTED, va="bottom")
else:
    fig.text(0.088, 0.945, "What actually works: three measured reductions on the ten AIDB hardness datasets",
             fontsize=15.5, color=INK, fontweight="semibold", va="top")
    for i, line in enumerate([
            "2M uniform samples, identical query traces, two query seeds. Probe counts are deterministic software counters, not timings.",
            "Virtual points and virtual fences cut work inside the index. The transform cuts only NFL's own metric: forced into the index,",
            "that same flow raises fence probes by 9 to 192 percent, and the cost-based selector never chooses it in any region."]):
        fig.text(0.088, 0.885 - i * 0.032, line, fontsize=10, color=MUTED, va="top")
    fig.legend(loc="lower left", bbox_to_anchor=(0.085, 0.035), ncol=1, frameon=False, fontsize=FS_LEG,
               labelspacing=0.5, handlelength=1.5, handleheight=1.0, borderpad=0.0)
    fig.text(0.088, 0.012, "Source: results/aidb_flowv2/sweep/results.jsonl (140 runs) and results/aidb_flowv2/flowcheck_summary.json  \u00b7  generated by results/aidb/summary_figure.py",
             fontsize=8, color=MUTED, va="bottom")
out = HERE / ("summary_slide.png" if SLIDE else "summary.png")
fig.savefig(out, facecolor="white")
print("wrote", out)
for d in order:
    v = data[d]
    print(f"  {d:<9} fence {v['fence'][0]:.2f}->{v['fence'][1]:.2f} ({v['fence'][2]:+.0f}%)  "
          f"root {v['root'][0]:.2f}->{v['root'][1]:.2f} ({v['root'][2]:+.0f}%)  "
          f"tcd {v['tcd'][0]}->{v['tcd'][1]} ({v['tcd'][2]:+.0f}%)")
