#!/usr/bin/env python3
"""Do the two methods help each other? The root's own 2x2, measured.

Needs matplotlib (outside the stdlib-only pipeline).
    /usr/bin/python3 results/aidb/interaction_figure.py [--slide]

Source: results/aidb_final/sweep/results.jsonl, the learnability block of packed_rank_root_fusion.
When the root selector runs it builds and costs FOUR candidates, which is a complete 2x2 factorial:
        factor A: the model's feature   raw key   or   NFL-style flow output
        factor B: the model's targets   plain ranks   or   CSV virtual-fence slots
Each cell's value is the probes the real root locate spends on the fences and fence midpoints,
so all four are measured the same way on the same data. Flow cells carry a constant +4.0
probe-equivalent charge for evaluating the transform, which cancels out of the interaction term.

interaction = (flow+fences - raw+fences) - (flow+ranks - raw+ranks)
  0        the two act independently: whatever the flow does, it does it just the same with or
           without virtual fences, so the combination is exactly the sum of the parts.
  < 0      synergy: virtual fences make the flow more useful than it was alone.
  > 0      interference.
Note this 2x2 requires a MONOTONE flow: with a non-monotone one the combination cell cannot be
built at all (index.hpp refuses it, a slot table needs an order-preserving feature).
"""
import json, collections, statistics, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
S = HERE.parent.parent
BLUE, ORANGE, GRAY, BAND = "#2a78d6", "#eb6834", "#b9c0c6", "#e8e7e2"
INK, MUTED, GRID = "#17242f", "#6b7480", "#d8d7d0"
SLIDE = "--slide" in sys.argv

by = collections.defaultdict(list)
for line in (S / "results/aidb_final/sweep/results.jsonl").open():
    r = json.loads(line); by[(r["dataset"], r["variant"])].append(r)

cells = []
for d in sorted({k[0] for k in by}):
    rs = by.get((d, "packed_rank_root_fusion"))
    if not rs: continue
    l = rs[0].get("learnability") or {}
    b, raw, flow, vr, vf = (l.get(k, 0) for k in ("root_probes_binary", "root_probes_raw",
                            "root_probes_flow", "root_probes_vp_raw", "root_probes_vp_flow"))
    if not vf: continue
    cells.append({"sample": d, "binary": b, "raw_rank": raw, "flow_rank": flow,
                  "raw_fence": vr, "flow_fence": vf, "inter": (vf - vr) - (flow - raw)})
inter = [c["inter"] for c in cells]
mean = lambda k: statistics.mean(c[k] for c in cells)
binary = mean("binary")
CHARGE = 4.0

fig = plt.figure(figsize=(13.33, 7.5) if SLIDE else (13.0, 7.4), dpi=200)
fig.patch.set_facecolor("white")
top = 0.545 if SLIDE else 0.50
axA = fig.add_axes([0.075, 0.235, 0.395, top]); axB = fig.add_axes([0.575, 0.235, 0.395, top])
for ax in (axA, axB):
    ax.set_facecolor("white")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["left"].set_color(GRID)
    ax.tick_params(length=0, labelsize=11 if SLIDE else 10, colors=MUTED)

# --- A: the interaction plot
for c in cells:
    axA.plot([0, 1], [c["raw_rank"], c["raw_fence"]], color=BLUE, lw=1.0, alpha=0.30, zorder=2)
    axA.plot([0, 1], [c["flow_rank"], c["flow_fence"]], color=ORANGE, lw=1.0, alpha=0.30, zorder=2)
axA.axhline(binary, color=GRAY, lw=1.6, ls=(0, (5, 4)), zorder=1)
axA.text(1.06, binary, f"binary search\n{binary:.2f}", va="center", ha="left",
         fontsize=9.6 if SLIDE else 9, color=MUTED)
for key_r, key_f, colour, name in ((("raw_rank"), ("raw_fence"), BLUE, "raw key"),
                                   (("flow_rank"), ("flow_fence"), ORANGE, "NFL flow feature")):
    ys = [mean(key_r), mean(key_f)]
    axA.plot([0, 1], ys, color=colour, lw=3.4, marker="o", ms=10, markeredgecolor="white",
             markeredgewidth=2, zorder=5, label=name)
    axA.text(-0.06, ys[0], f"{ys[0]:.1f}", va="center", ha="right", fontsize=10.5, color=colour, fontweight="semibold")
    axA.text(1.06, ys[1], f"{ys[1]:.1f}", va="center", ha="left", fontsize=10.5, color=colour, fontweight="semibold")
axA.set_xlim(-0.36, 1.42); axA.set_xticks([0, 1])
axA.set_xticklabels(["plain ranks", "+ CSV virtual fences"], fontsize=11.5 if SLIDE else 10.5, color=INK)
axA.set_ylabel("root probes per lookup", fontsize=11 if SLIDE else 10.5, color=MUTED)
axA.set_ylim(0, 17)
axA.legend(loc="upper right", frameon=False, fontsize=10.5 if SLIDE else 10)
axA.set_title("The two lines stay parallel", fontsize=12.5 if SLIDE else 11.8, color=INK,
              loc="left", pad=10, fontweight="semibold")

# --- B: the interaction term per sample
ys = list(range(len(cells)))
order = sorted(range(len(cells)), key=lambda i: cells[i]["inter"])
axB.axvspan(-1, 1, color=BAND, zorder=0)
axB.axvline(0, color=INK, lw=1.4, zorder=2)
for y, i in enumerate(order):
    axB.plot([cells[i]["inter"]], [y], "o", ms=7, color=BLUE if cells[i]["inter"] < 0 else ORANGE,
             markeredgecolor="white", markeredgewidth=1.3, zorder=4)
axB.set_yticks(ys)
axB.set_yticklabels([cells[i]["sample"].replace("_", " ") for i in order], fontsize=8.6 if SLIDE else 8.2, color=INK)
axB.set_xlim(-3, 3); axB.set_xticks([-2, -1, 0, 1, 2])
axB.set_xlabel("interaction, in probes per lookup", fontsize=11 if SLIDE else 10.5, color=MUTED, labelpad=8)
axB.set_ylim(-0.8, len(cells) - 0.2)
axB.xaxis.grid(True, color=GRID, lw=0.8, zorder=1); axB.set_axisbelow(True)
axB.set_title(f"and the interaction is nil: mean {statistics.mean(inter):+.2f} probes",
              fontsize=12.5 if SLIDE else 11.8, color=INK, loc="left", pad=10, fontweight="semibold")

head = 0.955
if not SLIDE:
    fig.text(0.075, head, "Do the two methods help each other? The root's own 2x2, measured on 20 samples",
             fontsize=15.5, color=INK, fontweight="semibold", va="top")
    head = 0.900
for i, line in enumerate([
        "The root selector builds and costs four candidates every run, which is a complete factorial: feature (raw key or NFL flow) x targets (plain ranks or CSV virtual fences).",
        f"Virtual fences help the raw key and the flow by the same amount, so the lines stay parallel: interaction {statistics.mean(inter):+.2f} probes, range {min(inter):+.2f} to {max(inter):+.2f}.",
        f"The flow's {CHARGE:.0f}-probe evaluation charge sits in both orange points and cancels from the interaction. Strip it out and the right-hand orange point",
        f"lands on the blue one ({mean('flow_fence')-CHARGE:.2f} against {mean('raw_fence'):.2f}): once virtual fences are in place, the transform has nothing left to add."]):
    fig.text(0.075, head - i * 0.030, line, fontsize=9.7, color=MUTED, va="top")
fig.text(0.075, 0.012, "Source: results/aidb_final/sweep/results.jsonl, learnability block of packed_rank_root_fusion  ·  generated by results/aidb/interaction_figure.py",
         fontsize=8, color=MUTED, va="bottom")

out = HERE / ("interaction_slide.png" if SLIDE else "interaction.png")
fig.savefig(out, facecolor="white")
print("wrote", out)
print(f"  cells complete on {len(cells)} samples")
print(f"  means: raw+ranks {mean('raw_rank'):.2f}  flow+ranks {mean('flow_rank'):.2f}  "
      f"raw+fences {mean('raw_fence'):.2f}  flow+fences {mean('flow_fence'):.2f}  (binary {binary:.2f})")
print(f"  flow+fences minus the {CHARGE:.0f}-probe charge = {mean('flow_fence')-CHARGE:.2f}, against raw+fences {mean('raw_fence'):.2f}")
print(f"  interaction: mean {statistics.mean(inter):+.3f}, range {min(inter):+.2f} to {max(inter):+.2f}, "
      f"{sum(1 for x in inter if abs(x) < 1)}/{len(inter)} within +-1 probe")
