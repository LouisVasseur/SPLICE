#!/usr/bin/env python3
"""The fusion result: where a lookup's work goes, and what each component removes.

Needs matplotlib (the only script here that does; it sits under results/ outside the
stdlib-only pipeline and nothing imports it). Run with an interpreter that has it:
    /usr/bin/python3 results/aidb/fusion_figure.py

Source: results/aidb_final/root_analysis.json, built from results/aidb_final/sweep/results.jsonl
(540 paired runs, 20 samples x 9 variants x 3 query seeds, 5M lookups each, performance-core QoS,
root fitted on real first keys). The ten uniform 2M samples are shown.

Each lookup's probe count splits into three disjoint parts:
  root probes       comparisons to find the region   <- CSV virtual fences at the root remove these
  fence probes      comparisons to correct the block <- CSV virtual points per region remove these
  coordinate probes comparisons inside the block     <- neither component touches these
Control = packed_rank. Fused = packed_rank_vp10_root_fusion (both components at once).
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
S = HERE.parent.parent
TEAL, BLUE, GRAY = "#1baf7a", "#2a78d6", "#b9c0c6"
INK, MUTED, GRID = "#17242f", "#6b7480", "#d8d7d0"

ra = json.loads((S / "results/aidb_final/root_analysis.json").read_text())["per_dataset"]
parts = lambda r: (r["root_probes"], r["fence_probes"],
                   r["total_probes"] - r["root_probes"] - r["fence_probes"])

rows, resid = {}, []
for d in ra:
    if not d.endswith("_uniform"):
        continue
    name = d[:-8]
    c = parts(ra[d]["packed_rank"])
    vp = parts(ra[d]["packed_rank_vp10"])
    rt = parts(ra[d]["packed_rank_root_fusion"])
    both = parts(ra[d]["packed_rank_vp10_root_fusion"])
    predicted = sum(c) - (c[0] - rt[0]) - (c[1] - vp[1])   # each component's saving, added
    rows[name] = {"control": c, "both": both, "predicted": predicted, "measured": sum(both)}
    resid.append(abs(predicted - sum(both)))
order = sorted(rows, key=lambda n: (sum(rows[n]["control"]) - sum(rows[n]["both"])) / sum(rows[n]["control"]))
max_resid = max(resid)

fig = plt.figure(figsize=(12.8, 8.2), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.125, 0.185, 0.865, 0.605]); ax.set_facecolor("white")
bh = 0.34
for i, name in enumerate(order):
    r = rows[name]
    for row, (vals, y) in enumerate(((r["control"], i + 0.20), (r["both"], i - 0.20))):
        left = 0.0
        for val, colour in zip(vals, (TEAL, BLUE, GRAY)):
            ax.barh(y, val, left=left, height=bh, color=colour, zorder=3,
                    edgecolor="white", linewidth=1.1)
            left += val
        cut = 100 * (sum(r["control"]) - sum(r["both"])) / sum(r["control"])
        ax.text(left + 0.25, y, f"{sum(vals):.2f}" + ("" if row == 0 else f"   −{cut:.0f}%"),
                va="center", ha="left", fontsize=8.8, color=MUTED if row == 0 else INK, zorder=4)
    ax.text(-0.45, i + 0.20, "control", va="center", ha="right", fontsize=8.4, color=MUTED)
    ax.text(-0.45, i - 0.20, "fused", va="center", ha="right", fontsize=8.4, color=INK)

ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=11.5, color=INK)
ax.tick_params(axis="y", length=0, pad=52)
ax.set_xlim(0, 22.4); ax.set_xticks([0, 5, 10, 15, 20])
ax.set_xticklabels(["0", "5", "10", "15", "20"], fontsize=10, color=MUTED)
ax.set_xlabel("probes per lookup   (key comparisons, deterministic software counters)",
              fontsize=10.5, color=MUTED, labelpad=9)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
ax.set_ylim(-0.72, len(order) - 0.28)

fig.text(0.078, 0.955, "What the selector actually builds: CSV twice, at two levels, with NFL declined",
         fontsize=15, color=INK, fontweight="semibold", va="top")
for i, line in enumerate([
        "Ten uniform 2M samples, 540 paired runs, identical query traces. Each bar is one lookup's work, split into the three places probes are spent.",
        f"Both reductions are CSV: the same smoothing run on a region's own keys, and on the fences between regions. They compose to within {max_resid:.3f} probes on all ten.",
        "The NFL transform was offered to the root selector on every sample and declined on every one (17.7-17.9 estimated probes against 2.3-3.6 for the winner)."]):
    fig.text(0.078, 0.902 - i * 0.030, line, fontsize=9.8, color=MUTED, va="top")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (TEAL, BLUE, GRAY)],
           labels=["Root probes: find the region   ·   CSV smoothing over the region fences",
                   "Fence probes: correct the block   ·   CSV smoothing over a region's own keys",
                   "Coordinate probes: search inside the block   ·   untouched by CSV, and by NFL had it won"],
           loc="lower left", bbox_to_anchor=(0.076, 0.028), frameon=False, fontsize=9.6,
           labelspacing=0.5, handlelength=1.5, handleheight=1.0, borderpad=0.0)
fig.text(0.078, 0.010, "Source: results/aidb_final/root_analysis.json  ·  control = packed_rank, fused = packed_rank_vp10_root_fusion  ·  generated by results/aidb/fusion_figure.py",
         fontsize=8, color=MUTED, va="bottom")

out = HERE / "fusion.png"
fig.savefig(out, facecolor="white")
print("wrote", out, f"(max additivity residual {max_resid:.3f} probes)")
for n in order:
    r = rows[n]
    print(f"  {n:<9} control {sum(r['control']):6.2f}  fused {r['measured']:6.2f}  "
          f"predicted {r['predicted']:6.2f}  residual {abs(r['predicted']-r['measured']):.3f}")
