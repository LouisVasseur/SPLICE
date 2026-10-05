#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Summarize raw-fence root budgets 4x (stored sweep) / 16x / 32x / 63x (direct runs) vs the fusion's flow+fences root."""
import json, collections
SWEEP = REPO + "/experimental/scaleli/results/aidb_final/sweep/results.jsonl"
BUD = REPO + "/build-verify-synergy/scratch/budget.jsonl"
OUT = REPO + "/build-verify-synergy/scratch/budget_summary.json"
fus = {}
for l in open(SWEEP):
    r = json.loads(l)
    if r["variant"] == "packed_rank_root_fusion" and r["seed"] == 11:
        L = r["learnability"]; fus[r["dataset"]] = {"binary": L["root_probes_binary"], "raw": L["root_probes_raw"], "vp_raw_4x": L["root_probes_vp_raw"],
                                                   "flow_corr": L["root_probes_flow"] - 4, "fusion_corr": L["root_probes_vp_flow"] - 4, "fusion_virtual": L["root_virtual"],
                                                   "fusion_measured": r["work_counters"]["root_probes"] / r["operations"], "regions": L["regions"]}
runs = collections.defaultdict(dict)
for l in open(BUD):
    r = json.loads(l); L = r["learnability"]
    key = ("flow" if r["with_flow"] else "raw", int(r["root_alpha_req"]))
    runs[r["dataset"]][key] = {"vp_raw": L["root_probes_vp_raw"], "vp_flow_corr": (L["root_probes_vp_flow"] - 4) if L["root_probes_vp_flow"] else None,
                               "chosen_virtual": L["root_virtual"], "chosen": ("flow+" if L["root_flow"] else "raw+") + ("fences" if L["root_vp"] else "ranks") if L["root_model"] else "binary",
                               "measured": r["work_counters"]["root_probes"] / r["operations"], "wall_s": r["wall_s"]}
out = {}
print(f"{'dataset':16s} {'bin':>5s} {'raw':>6s} | raw+fences est @ 4x / 16x / 32x / 63x (budget slots 1956/7824/15648/30807) | {'flow-4':>6s} {'fusion-4':>8s} {'fus virt':>8s} {'fus meas':>8s}")
for d in sorted(fus):
    if d not in runs: continue
    f = fus[d]; R = runs[d]
    cells = []
    for a in (4, 16, 32, 63):
        if a == 4: cells.append(f"{f['vp_raw_4x']:5.2f}")
        elif ("raw", a) in R: x = R[("raw", a)]; cells.append(f"{x['vp_raw']:5.2f}({x['wall_s']:.0f}s{',virt=' + str(x['chosen_virtual']) if x['chosen'] != 'binary' else ''})")
        else: cells.append("  -  ")
    out[d] = {"fusion_row": f, "runs": {f"{k[0]}_{k[1]}x": v for k, v in R.items()}}
    print(f"{d:16s} {f['binary']:5.2f} {f['raw']:6.2f} | {' / '.join(cells):58s} | {f['flow_corr']:6.2f} {f['fusion_corr']:8.2f} {f['fusion_virtual']:8d} {f['fusion_measured']:8.2f}")
json.dump(out, open(OUT, "w"), indent=1)
print("JSON ->", OUT)
