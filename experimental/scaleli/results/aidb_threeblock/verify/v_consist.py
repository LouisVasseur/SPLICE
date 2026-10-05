import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""VERIFIER items 2 (cross-arm consistency), 4 (charge sensitivity), 5 (coupling) from stored runs."""
import json, os, glob, math
TB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(os.path.join(TB, "armA.json")))
B = json.load(open(os.path.join(TB, "armB.json")))
Bs = {(s["dataset"], s["mode"]): s for s in B["samples"]}
JR = json.load(open(REPO + "/experimental/scaleli/results/aidb_joint/joint_results.json"))
out = {}
# --- A round 1 vs B single pass (same order) ---
mism = []; n = 0
for r in A["rows"]:
    if r["k"] == 0: continue
    b = Bs[(r["dataset"], r["mode"])]["per_k"][str(r["k"])]["variants"]["G->T->V" if r["order"] == "GTV" else "T->G->V"]
    run = json.load(open(os.path.join(TB, "armA_runs", "%s_%s_k%d_%s.json" % (r["dataset"], r["mode"], r["k"], r["order"]))))
    r1 = run["rounds"][0]
    n += 1
    if abs(r1["model_probes"] - b["model_probes"]) > 0 or r1["gaps"] != b["gaps"] or r1["virtual_count"] != b["virtual_count"] or r1["gap_table_probes"] != b["gap_table_probes"]:
        mism.append((r["dataset"], r["mode"], r["k"], r["order"], r1["model_probes"], b["model_probes"]))
out["A_round1_vs_B_singlepass"] = {"cells": n, "mismatches": mism}
# k0 control vs B T->V
mm0 = []
for r in A["rows"]:
    if r["k"] != 0: continue
    b = Bs[(r["dataset"], r["mode"])]["per_k"]["1"]["variants"]["T->V"]
    if r["round1"]["model_probes"] != b["model_probes"]:
        mm0.append((r["dataset"], r["mode"], r["round1"]["model_probes"], b["model_probes"]))
out["A_k0_round1_vs_B_TV"] = mm0
# --- charge sensitivity: does any G config beat the no-G best for charge factor c? break-even c ---
def ceil_log(k): return math.ceil(math.log2(k + 1)) if k else 0
be = []
for (d, m), s in Bs.items():
    for k, pk in s["per_k"].items():
        V = pk["variants"]
        noG = min(v["model_probes"] + (0) for nm, v in V.items() if v.get("gap_k", 0) == 0 and "model_probes" in v and nm in ("baseline", "V", "T->V", "T"))
        # total_uncharged without G (no T charge) and charged
        noG_unc = min(v["total_uncharged"] for nm, v in V.items() if "total_uncharged" in v and v.get("gap_k", 0) == 0)
        noG_ch = min(v["total_charged"] for nm, v in V.items() if "total_charged" in v and v.get("gap_k", 0) == 0)
        for nm, v in V.items():
            if "model_probes" not in v or v.get("gap_k", 0) == 0: continue
            gp = v["gap_table_probes"]
            for acc, ref, extra in (("uncharged", noG_unc, 0.0), ("charged", noG_ch, v["transform_charge_probes"] if "transform_charge_probes" in v else (6.9 if "T" in nm else 0.0))):
                save = ref - (v["model_probes"] + extra)
                c_be = save / gp if gp else float("inf")
                be.append((c_be, d, m, int(k), nm, acc, v["model_probes"], gp, ref))
be.sort(reverse=True)
out["B_breakeven_top"] = be[:12]
out["B_any_win_at_c1"] = [x for x in be if x[0] >= 1.0]
out["B_any_win_at_c2"] = [x for x in be if x[0] >= 2.0]
# A: best total under doubled charge vs k0 best (k0 has no charge -> unchanged)
k0 = {(r["dataset"], r["mode"]): r["best"]["total_uncharged"] for r in A["rows"] if r["k"] == 0}
winsA = {}
for c in (0.0, 0.25, 0.5, 1.0, 2.0):
    w = []
    for r in A["rows"]:
        if r["k"] == 0: continue
        run = json.load(open(os.path.join(TB, "armA_runs", "%s_%s_k%d_%s.json" % (r["dataset"], r["mode"], r["k"], r["order"]))))
        acc = [x for x in run["rounds"] if x.get("accepted")]
        best = min(x["model_probes"] + c * x["gap_table_probes"] for x in acc)
        if best < k0[(r["dataset"], r["mode"])]:
            w.append((r["dataset"], r["mode"], r["k"], r["order"], round(best, 4), round(k0[(r["dataset"], r["mode"])], 4)))
    winsA[str(c)] = w
out["A_G_beats_k0_at_charge_factor"] = {c: (len(w), w[:10]) for c, w in winsA.items()}
# A: does the round>=2 gain depend on the charge (k_eff changing)?
chg = []
for r in A["rows"]:
    if r["k"] == 0: continue
    if abs(r["round1"]["gap_table_probes"] - r["best"]["gap_table_probes"]) > 0 or r["round1"]["gap_k_eff"] != r["best"]["gap_k_eff"]:
        chg.append((r["dataset"], r["mode"], r["k"], r["order"], r["round1"]["gap_k_eff"], r["best"]["gap_k_eff"], r["round1"]["gap_table_probes"], r["best"]["gap_table_probes"], round(r["gain_rounds2plus"], 4), round(r["gain_rounds2plus_model_only"], 4)))
out["A_keff_or_charge_changed_r1_to_best"] = chg
# with doubled charge, does guarded selection differ? recompute gain under 2x charge from accepted-candidate series (approx: same accepted path)
# --- coupling: recount gap-set changes from raw runs ---
acc_chg = []; any_chg = []; ns_chg = []; swapped = {}
for p in sorted(glob.glob(os.path.join(TB, "armA_runs", "*.json"))):
    run = json.load(open(p))
    if run["k"] == 0: continue
    rs = [x for x in run["rounds"] if "gaps" in x]
    g1 = set(rs[0]["gaps"])
    tag = (run["dataset"], run["mode"], run["k"], run["order"])
    if run["nostop"]:
        if any(set(x["gaps"]) != g1 for x in rs[1:]): ns_chg.append(tag)
        continue
    if any(set(x["gaps"]) != g1 for x in rs[1:] if x["accepted"]): acc_chg.append(tag)
    if any(set(x["gaps"]) != g1 for x in rs[1:]): any_chg.append(tag)
    acc = [x for x in rs if x["accepted"]]
    swapped[str(tag)] = len(g1 - set(acc[-1]["gaps"]))
out["coupling"] = {"accepted": len(acc_chg), "any_candidate": len(any_chg), "nostop": len(ns_chg), "accepted_cells": acc_chg,
                   "books_u_64_GTV_swapped": swapped[str(("books", "uniform", 64, "GTV"))]}
# --- B claims: G helps model 35/80, total 0/80 ---
cnt = {"model_probes": 0, "total_uncharged": 0, "total_charged_pwl": 0, "total_charged": 0}
for (d, m), s in Bs.items():
    for k, pk in s["per_k"].items():
        V = pk["variants"]
        for acc in cnt:
            wg = min(v[acc] for nm, v in V.items() if acc in v and v.get("gap_k", 0) > 0)
            ng = min(v[acc] for nm, v in V.items() if acc in v and v.get("gap_k", 0) == 0)
            if wg < ng: cnt[acc] += 1
out["B_G_helps_counts_recomputed"] = cnt
# same, but baseline excludes in-sample probe polish (pool*), i.e. CSV/T only
cnt2 = {"model_probes": 0}; mp = []
for (d, m), s in Bs.items():
    for k, pk in s["per_k"].items():
        V = pk["variants"]
        wg = min(v["model_probes"] for nm, v in V.items() if "model_probes" in v and v.get("gap_k", 0) > 0 and not nm.startswith("pool"))
        ng = min(v["model_probes"] for nm, v in V.items() if "model_probes" in v and v.get("gap_k", 0) == 0 and not nm.startswith("pool"))
        if wg < ng: cnt2["model_probes"] += 1; mp.append((d, m, int(k), round(ng - wg, 4)))
mp.sort(key=lambda x: -x[3])
out["B_G_helps_model_fixed_orders_only"] = (cnt2, mp[:12])
json.dump(out, open(os.path.join(TB, "verify", "v_consist.json"), "w"), indent=1, default=str)
for k, v in out.items():
    print("==", k); print(json.dumps(v, default=str)[:2500])
