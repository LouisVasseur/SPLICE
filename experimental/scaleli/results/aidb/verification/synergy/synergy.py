#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Attack C1 / 'synergy': estimate-vs-measured, fence budget, uniform-vs-window, singles on windows."""
import json, statistics, collections, math
PATH = REPO + "/experimental/scaleli/results/aidb_final/sweep/results.jsonl"
OUT = REPO + "/build-verify-synergy/scratch/synergy.json"
GRID = {11, 29, 47}
rows = [json.loads(l) for l in open(PATH) if l.strip()]
by = collections.defaultdict(dict)
for r in rows:
    if r["seed"] in GRID: by[(r["dataset"], r["variant"])][r["seed"]] = r
datasets = sorted({d for d, _ in by})
uni = [d for d in datasets if d.endswith("_uniform")]; win = [d for d in datasets if d.endswith("_window")]

def chosen(L):
    if not L.get("root_model"): return "binary"
    return ("flow" if L.get("root_flow") else "raw") + "+" + ("fences" if L.get("root_vp") else "ranks")
def est(L, cand, corrected=True):
    """stored estimate; flow candidates carry +flow_cost inside the stored value"""
    key = {"binary": "root_probes_binary", "raw+ranks": "root_probes_raw", "flow+ranks": "root_probes_flow",
           "raw+fences": "root_probes_vp_raw", "flow+fences": "root_probes_vp_flow"}[cand]
    v = L.get(key, 0.0)
    if v == 0: return None
    if corrected and cand.startswith("flow"): v -= 4.0
    return v
def measured_root(sd):
    return statistics.median(r["work_counters"]["root_probes"] / r["operations"] for r in sd.values())
def xform(sd):
    return statistics.median(r["work_counters"].get("transform_calls", 0) / r["operations"] for r in sd.values())
def flowcost(sd):
    return {r.get("flow_cost") for r in sd.values()}

out = {"a_estimate_vs_measured": [], "b_budget": {}, "c_synergy": {}, "d_window_singles": {}}
print("=" * 100)
print("(a) ESTIMATE vs MEASURED root probes per lookup, every cell where a learned root was chosen (+ binary cells)")
print(f"{'dataset':16s} {'variant':30s} {'chosen':12s} {'stored':>7s} {'corr':>7s} {'measured':>9s} {'meas-corr':>10s} {'xform/op':>9s} flow_cost")
for (d, v), sd in sorted(by.items()):
    L = next(iter(sd.values())).get("learnability") or {}
    if not L: continue
    c = chosen(L)
    if c == "binary" and v != "packed_rank": continue  # fallbacks: nothing to compare beyond binary
    if v == "packed_rank":  # binary root never runs fit_root, so take the binary estimate from the fusion row of the same dataset
        L = next(iter(by[(d, "packed_rank_root_fusion")].values()))["learnability"]
    st = est(L, c, corrected=False); co = est(L, c, corrected=True); m = measured_root(sd)
    rec = {"dataset": d, "variant": v, "chosen": c, "stored_est": st, "corrected_est": co, "measured": m,
           "measured_minus_corrected": m - co, "measured_minus_stored": m - st, "transform_calls_per_op": xform(sd),
           "root_virtual": L.get("root_virtual"), "flow_cost": sorted(flowcost(sd))}
    out["a_estimate_vs_measured"].append(rec)
    print(f"{d:16s} {v:30s} {c:12s} {st:7.2f} {co:7.2f} {m:9.2f} {m-co:10.2f} {xform(sd):9.3f} {sorted(flowcost(sd))}")
# aggregate by candidate type
print("\nsummary of (measured - corrected estimate) by candidate type:")
agg = collections.defaultdict(list)
for rec in out["a_estimate_vs_measured"]: agg[rec["chosen"]].append(rec["measured_minus_corrected"])
for c, xs in agg.items():
    print(f"  {c:12s} n={len(xs):2d} min={min(xs):+.2f} median={statistics.median(xs):+.2f} max={max(xs):+.2f}")
out["a_summary"] = {c: {"n": len(xs), "min": min(xs), "median": statistics.median(xs), "max": max(xs)} for c, xs in agg.items()}

print("\n" + "=" * 100)
print("(b) FENCE BUDGET: vf4's raw+fences estimate vs fusion's raw+fences estimate; root_virtual; fusion flow+fences (corrected)")
print(f"{'dataset':16s} {'vf4 vp_raw':>10s} {'fus vp_raw':>10s} {'vf4 virt':>8s} {'fus virt':>8s} {'fus vp_flow-4':>13s} {'raw':>6s} {'flow-4':>7s} {'bin':>6s} {'vf4 measured':>12s}")
for d in datasets:
    Lv = next(iter(by[(d, "packed_rank_root_vf4")].values()))["learnability"]
    Lf = next(iter(by[(d, "packed_rank_root_fusion")].values()))["learnability"]
    rec = {"vf4_vp_raw": Lv["root_probes_vp_raw"], "fusion_vp_raw": Lf["root_probes_vp_raw"], "vf4_virtual": Lv["root_virtual"], "fusion_virtual": Lf["root_virtual"],
           "fusion_vp_flow_corrected": Lf["root_probes_vp_flow"] - 4, "raw": Lf["root_probes_raw"], "flow_corrected": Lf["root_probes_flow"] - 4, "binary": Lf["root_probes_binary"],
           "vf4_measured": measured_root(by[(d, "packed_rank_root_vf4")]), "vf4_chosen": chosen(Lv), "regions": Lf["regions"]}
    out["b_budget"][d] = rec
    print(f"{d:16s} {rec['vf4_vp_raw']:10.2f} {rec['fusion_vp_raw']:10.2f} {rec['vf4_virtual']:8d} {rec['fusion_virtual']:8d} {rec['fusion_vp_flow_corrected']:13.2f} {rec['raw']:6.2f} {rec['flow_corrected']:7.2f} {rec['binary']:6.2f} {rec['vf4_measured']:12.2f}  {rec['vf4_chosen']}")

print("\n" + "=" * 100)
print("(c) SYNERGY pattern per dataset from the fusion row's four candidates (charge-corrected probes; stored in brackets for flow)")
print("    syn_stored = fusion_stored < min(raw, flow_stored, fences) and < binary (notes' definition)")
print("    syn_corr   = same test on charge-corrected probes;  super = saving(fusion) > saving(flow)+saving(fences) vs raw+ranks")
print(f"{'dataset':16s} {'bin':>5s} {'raw':>6s} {'flow':>13s} {'fences':>7s} {'fusion':>13s} {'chosen':12s} syn_stored syn_corr  super  sv_flow sv_fence sv_fus")
for d in datasets:
    Lf = next(iter(by[(d, "packed_rank_root_fusion")].values()))["learnability"]
    b = Lf["root_probes_binary"]; raw = Lf["root_probes_raw"]; fl_s = Lf["root_probes_flow"]; fe = Lf["root_probes_vp_raw"]; fu_s = Lf["root_probes_vp_flow"]
    fl = fl_s - 4; fu = fu_s - 4
    syn_stored = fu_s < min(raw, fl_s, fe) and fu_s < b
    syn_corr = fu < min(raw, fl, fe) and fu < b
    sv_flow, sv_fence, sv_fus = raw - fl, raw - fe, raw - fu
    superadd = sv_fus > sv_flow + sv_fence
    rec = {"binary": b, "raw": raw, "flow_stored": fl_s, "flow_corr": fl, "fences": fe, "fusion_stored": fu_s, "fusion_corr": fu, "chosen": chosen(Lf),
           "synergy_stored_def": syn_stored, "synergy_corrected": syn_corr, "superadditive": superadd,
           "saving_flow": sv_flow, "saving_fences": sv_fence, "saving_fusion": sv_fus,
           "best_single_corr": min(raw, fl, fe), "fusion_beats_binary_corr": fu < b, "any_single_beats_binary_corr": min(raw, fl, fe) < b}
    out["c_synergy"][d] = rec
    print(f"{d:16s} {b:5.2f} {raw:6.2f} {fl:6.2f}[{fl_s:5.2f}] {fe:7.2f} {fu:6.2f}[{fu_s:5.2f}] {chosen(Lf):12s} {str(syn_stored):10s} {str(syn_corr):8s} {str(superadd):6s} {sv_flow:6.2f} {sv_fence:7.2f} {sv_fus:6.2f}")
for grp, name in ((uni, "uniform"), (win, "window")):
    n_s = sum(out["c_synergy"][d]["synergy_stored_def"] for d in grp); n_c = sum(out["c_synergy"][d]["synergy_corrected"] for d in grp); n_sup = sum(out["c_synergy"][d]["superadditive"] for d in grp)
    print(f"  {name}: synergy(stored def) on {n_s}/10, synergy(corrected) on {n_c}/10, super-additive on {n_sup}/10")
    out["c_synergy"][f"_count_{name}"] = {"stored_def": n_s, "corrected": n_c, "superadditive": n_sup}

print("\n" + "=" * 100)
print("(d) WINDOWS: did any single-candidate variant beat binary?  (root_raw / root_flow / root_vf4 rows, their own selector's choice)")
print(f"{'dataset':16s} {'root_raw':>14s} {'root_flow':>18s} {'root_vf4':>16s}   est raw / flow(stored,corr) / fences vs binary   | measured root probes raw/flow/vf4/packed")
for d in win + uni:
    recs = {}
    line = f"{d:16s}"
    for v in ("packed_rank_root_raw", "packed_rank_root_flow", "packed_rank_root_vf4"):
        L = next(iter(by[(d, v)].values()))["learnability"]; c = chosen(L)
        recs[v] = {"chosen": c, "est_raw": L["root_probes_raw"], "est_flow_stored": L["root_probes_flow"], "est_vp_raw": L["root_probes_vp_raw"], "binary": L["root_probes_binary"], "measured": measured_root(by[(d, v)])}
        line += f" {c:>16s}"
    Lf = next(iter(by[(d, "packed_rank_root_flow")].values()))["learnability"]; Lv = next(iter(by[(d, "packed_rank_root_vf4")].values()))["learnability"]
    line += f"   {Lf['root_probes_raw']:5.2f} / {Lf['root_probes_flow']:5.2f},{Lf['root_probes_flow']-4:5.2f} / {Lv['root_probes_vp_raw']:5.2f} vs {Lf['root_probes_binary']:5.2f}"
    line += f"   | {recs['packed_rank_root_raw']['measured']:5.2f}/{recs['packed_rank_root_flow']['measured']:5.2f}/{recs['packed_rank_root_vf4']['measured']:5.2f}/{measured_root(by[(d,'packed_rank')]):5.2f}"
    out["d_window_singles"][d] = recs
    print(line)
# consistency of estimates across variants (same fences => same raw / flow / vp_raw estimates?)
print("\nconsistency: are root_probes_raw / flow / vp_raw identical across the variants that computed them (per dataset)?")
incons = []
for d in datasets:
    vals = collections.defaultdict(set)
    for v in ("packed_rank_root_raw", "packed_rank_root_flow", "packed_rank_root_vf4", "packed_rank_root_fusion", "packed_rank_vp10_root_fusion"):
        L = next(iter(by[(d, v)].values()))["learnability"]
        for k in ("root_probes_binary", "root_probes_raw", "root_probes_flow", "root_probes_vp_raw", "root_probes_vp_flow"):
            if L.get(k, 0): vals[k].add(round(L[k], 6))
    bad = {k: sorted(s) for k, s in vals.items() if len(s) > 1}
    if bad: incons.append((d, bad))
print("  inconsistent:", incons if incons else "none (all candidate estimates identical wherever computed)")
out["estimate_consistency_across_variants"] = incons
json.dump(out, open(OUT, "w"), indent=1)
print("\nJSON ->", OUT)
