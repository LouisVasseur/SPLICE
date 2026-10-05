"""Summaries over armB.json -> armB_analysis.json + armB_tables.md (pasted into armB.md)."""
import json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "armB.json")))
FIXED = ["baseline", "G", "T", "V", "G->T->V", "T->G->V", "G->V", "T->V"]
POOLS = ["pool", "pool_pwl", "poolV", "poolV_pwl"]
ACCTS = ["model_probes", "total_uncharged", "total_charged_pwl", "total_charged"]
BAND = {"books": 0.361, "wise": 0.263, "genome": 0.197, "libio": 0.175, "covid": 0.086,
        "history": 0.050, "stack": 0.045, "fb": 0.0}
out = {"G_helps_counts": {}, "winner_counts": {}, "fixed_winner_counts": {},
       "G_order_winner_model": collections.Counter(), "G_order_winner_charged": collections.Counter(),
       "pool_steps": [], "cells": 0, "breakeven": []}
for a in ACCTS:
    out["G_helps_counts"][a] = []
    out["winner_counts"][a] = collections.Counter()
    out["fixed_winner_counts"][a] = collections.Counter()
L = []
L.append("| sample | k | " + " | ".join(FIXED + ["pool", "poolV"]) + " | winner (charged) | winner (uncharged) | best G vs best no-G, model |")
L.append("|" + "---|" * (len(FIXED) + 7))
for s in A["samples"]:
    for k, e in s["per_k"].items():
        out["cells"] += 1
        v = e["variants"]
        for a in ACCTS:
            w = e["winners"][a]
            out["winner_counts"][a][w["winner"]] += 1
            out["fixed_winner_counts"][a][w["fixed_order_winner"]] += 1
            if w["G_helps"]:
                out["G_helps_counts"][a].append([s["dataset"], s["mode"], int(k), w["best_with_G"],
                                                  w["best_with_G_value"], w["best_without_G"],
                                                  w["best_without_G_value"]])
        out["G_order_winner_model"][e["G_order_winner_model"]] += 1
        out["G_order_winner_charged"][e["G_order_winner_charged"]] += 1
        wm = e["winners"]["model_probes"]
        gk = v[wm["best_with_G"]]["gap_table_probes"] if wm["best_with_G"] else None
        saving = wm["best_without_G_value"] - wm["best_with_G_value"] if wm["best_with_G"] else None
        out["breakeven"].append([s["dataset"], s["mode"], int(k), wm["best_with_G"], saving, gk])
        for p in ("pool", "pool_pwl", "poolV", "poolV_pwl"):
            r = v.get(p)
            if r and "error" not in r:
                tr = r["pool_priced_trace"]
                out["pool_steps"].append([s["dataset"], s["mode"], int(k), p, r["pool_steps"],
                                          tr[0][1], r["model_probes"], r["gap_k"], r["total_charged"],
                                          r["pool_objective_amortised"]])
        def cell(n):
            r = v.get(n)
            if r is None or "error" in r:
                return "err"
            t = "%.3f/%.3f" % (r["model_probes"], r["total_charged"])
            if r["gap_k"]:
                t += " g%d" % r["gap_k"]
            if r["virtual_count"]:
                t += " v%d" % r["virtual_count"]
            return t
        L.append("| %s-%s | %s | %s | %s | %s | %s |" % (
            s["dataset"], s["mode"][0], k, " | ".join(cell(n) for n in FIXED + ["pool", "poolV"]),
            e["winners"]["total_charged"]["winner"], e["winners"]["total_uncharged"]["winner"],
            "%+.3f (%s, table %d)" % (-saving, wm["best_with_G"], gk) if saving is not None else "-"))
for key in ("winner_counts", "fixed_winner_counts"):
    out[key] = {a: dict(c) for a, c in out[key].items()}
out["G_order_winner_model"] = dict(out["G_order_winner_model"])
out["G_order_winner_charged"] = dict(out["G_order_winner_charged"])
json.dump(out, open(os.path.join(HERE, "armB_analysis.json"), "w"), indent=1)
open(os.path.join(HERE, "armB_tables.md"), "w").write("\n".join(L) + "\n")
print(json.dumps({k: out[k] for k in ("cells", "winner_counts", "fixed_winner_counts",
                                      "G_order_winner_model", "G_order_winner_charged")}, indent=1))
for a in ACCTS:
    print(a, "G helps in", len(out["G_helps_counts"][a]), "cells")
    for x in out["G_helps_counts"][a][:200]:
        print("   ", x)
