#!/usr/bin/env python3
"""Aggregate armB_parts/*.json -> armB.json + armB_tables.txt (the .md is written by hand from these)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import threeblock as tb  # noqa: E402

PARTS = os.path.join(HERE, "armB_parts")
KS = [1, 4, 16, 64]
FIXED = ["baseline", "G", "T", "V", "G->T->V", "T->G->V", "G->V", "T->V"]
POOLS = ["pool", "pool_pwl", "poolV", "poolV_pwl"]
FIELDS = ["model_probes", "gap_table_probes", "gap_k", "total_uncharged", "total_charged",
          "total_charged_pwl", "virtual_count"]
ACCTS = ["model_probes", "total_uncharged", "total_charged_pwl", "total_charged"]
OLD = {r["dataset"]: r for r in json.load(open(os.path.join(tb.AIDB, "joint_results.json")))}


def load(name):
    p = os.path.join(PARTS, name + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def slim(r):
    if r is None or "error" in r:
        return {"error": (r or {}).get("error", "missing")}
    o = {f: r[f] for f in FIELDS}
    o["gaps"] = r.get("gaps", [])
    if "pool_info" in r:
        o["pool_steps"] = r["pool_info"]["steps"]
        o["pool_objective_amortised"] = r["pool_info"]["pool_objective"]
        o["pool_priced_trace"] = r["pool_info"]["trace_priced"]
    return o


def main():
    out = {"meta": {
        "T_charge": tb.T_CHARGE, "T_charge_pwl": tb.T_CHARGE_PWL, "alpha": tb.ALPHA,
        "accountings": {
            "model_probes": "fit_root candidate score only",
            "total_uncharged": "model + gap table ceil(log2(k_eff+1)); T free",
            "total_charged_pwl": "total_uncharged + 0.45 if T on",
            "total_charged": "total_uncharged + 6.9 if T on"}},
        "samples": []}
    regress = []
    missing = []
    for m in tb.MODES:
        for d in tb.DATASETS:
            ind = load("indep_%s_%s" % (d, m))
            pe = load("pool_%s_%s_empty" % (d, m))
            pv = load("pool_%s_%s_fromV" % (d, m))
            if ind is None:
                missing.append("indep_%s_%s" % (d, m))
                continue
            if m == "uniform" and d in OLD:
                o = OLD[d]
                b = o["budgets"]["4"]
                for nm, ref in (("baseline", o["baseline_linear"]), ("T", o["flow_only"]),
                                ("V", b["csv_only"]), ("T->V", b["sequential"])):
                    regress.append((d, nm, ind[nm]["model_probes"] - ref))
            samp = {"dataset": d, "mode": m, "per_k": {}}
            if pe is not None and "v_phase" in pe:
                samp["pool_v_phase"] = pe["v_phase"]
            for k in KS:
                kk = load("k_%s_%s_%d" % (d, m, k))
                if kk is None:
                    missing.append("k_%s_%s_%d" % (d, m, k))
                    continue
                v = {}
                for nm in ("baseline", "T", "V", "T->V"):
                    v[nm] = slim(ind[nm])
                for nm in ("G", "G->T->V", "T->G->V", "G->V", "G->T", "T->G"):
                    v[nm] = slim(kk.get(nm))
                for src, tag in ((pe, "pool"), (pv, "poolV")):
                    if src is None or "error" in src:
                        if src is None:
                            missing.append("%s_%s_%s" % (tag, d, m))
                        continue
                    v[tag] = slim(src.get("k%d" % k))
                    v[tag + "_pwl"] = slim(src.get("k%d_pwl" % k))
                ent = {"variants": v, "overlap_G_vs_TG": kk.get("overlap_G_vs_TG")}
                reportable = [n for n in FIXED + POOLS if n in v and "error" not in v[n]]
                win = {}
                for a in ACCTS:
                    best = min(reportable, key=lambda n: (v[n][a], (FIXED + POOLS).index(n)))
                    fixed_best = min([n for n in reportable if n in FIXED],
                                     key=lambda n: (v[n][a], FIXED.index(n)))
                    withG = [n for n in reportable if v[n]["gap_k"] > 0]
                    noG = [n for n in reportable if v[n]["gap_k"] == 0]
                    bG = min(withG, key=lambda n: v[n][a]) if withG else None
                    bN = min(noG, key=lambda n: v[n][a])
                    win[a] = {"winner": best, "winner_value": v[best][a],
                              "fixed_order_winner": fixed_best,
                              "best_with_G": bG, "best_with_G_value": v[bG][a] if bG else None,
                              "best_without_G": bN, "best_without_G_value": v[bN][a],
                              "G_helps": (bG is not None and v[bG][a] < v[bN][a] - 1e-12),
                              "G_margin": (v[bG][a] - v[bN][a]) if bG else None}
                # order comparison among the three G orders on model probes
                gord = [n for n in ("G->T->V", "T->G->V", "G->V") if "error" not in v[n]]
                ent["G_order_winner_model"] = min(gord, key=lambda n: v[n]["model_probes"]) if gord else None
                ent["G_order_winner_charged"] = min(gord, key=lambda n: v[n]["total_charged"]) if gord else None
                ent["winners"] = win
                samp["per_k"][str(k)] = ent
            out["samples"].append(samp)
    out["regression_vs_joint_results"] = regress
    out["missing"] = missing
    json.dump(out, open(os.path.join(HERE, "armB.json"), "w"), indent=1)

    # ---------------- text tables ----------------
    L = []
    L.append("REGRESSION max |diff| vs joint_results.json (uniform, G off): %.1e over %d cells"
             % (max([abs(x[2]) for x in regress] or [0]), len(regress)))
    L.append("missing: %s" % missing)
    for s in out["samples"]:
        for k in KS:
            e = s["per_k"].get(str(k))
            if not e:
                continue
            L.append("")
            L.append("%s %s k=%d   overlap(G,T->G)=%s" % (s["dataset"], s["mode"], k, e["overlap_G_vs_TG"]))
            L.append("  %-10s %8s %5s %5s %9s %9s %9s %6s" % ("variant", "model", "gtab", "k_eff", "uncharged",
                                                            "chg_pwl", "charged", "virt"))
            for n in FIXED + POOLS + ["G->T", "T->G"]:
                r = e["variants"].get(n)
                if r is None:
                    continue
                if "error" in r:
                    L.append("  %-10s ERROR %s" % (n, r["error"][:80]))
                    continue
                L.append("  %-10s %8.3f %5.0f %5d %9.3f %9.3f %9.3f %6d" % (
                    n, r["model_probes"], r["gap_table_probes"], r["gap_k"], r["total_uncharged"],
                    r["total_charged_pwl"], r["total_charged"], r["virtual_count"]))
            for a in ACCTS:
                w = e["winners"][a]
                L.append("  [%s] winner=%s %.3f | bestG=%s %s | bestNoG=%s %.3f | G_helps=%s" % (
                    a, w["winner"], w["winner_value"], w["best_with_G"],
                    "%.3f" % w["best_with_G_value"] if w["best_with_G_value"] is not None else "-",
                    w["best_without_G"], w["best_without_G_value"], w["G_helps"]))
    open(os.path.join(HERE, "armB_tables.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:3]))


if __name__ == "__main__":
    main()
