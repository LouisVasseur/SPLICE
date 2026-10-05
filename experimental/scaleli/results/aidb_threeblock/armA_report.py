"""Aggregate armA_runs/ + armA_band/ (+ armA_extend/) into armA.json and armA_table.md.

Per (dataset, mode, k, order):
  round1          = the single coarse-to-fine pass (G->T->V or T->G->V)
  best (guarded)  = last accepted round of the guarded cycle (a round is accepted
                    only if total_uncharged drops by > 1e-6; first rejection stops)
  gain            = round1.total_uncharged - best.total_uncharged  (>= 0 by construction)
  nostop          = diagnostic run without the round guard; best taken post hoc
Noise yardsticks for V's chaotic greedy, all at the ROUND-1 state of that cell:
  listed   = last week's quoted band for the dataset (uniform samples only; none for osm/planet)
  spread   = max - min of model_probes over identity + 16 monotone perturbations
             (threeblock.perturbed_V_spread, c4 family)
  lucky    = base - min(variants): the improvement a pure re-roll of the greedy
             buys with NO change to G or T (the null for "a later round got lucky")
"""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import threeblock as tb

LISTED = {"books": 0.361, "wise": 0.263, "genome": 0.197, "libio": 0.175, "covid": 0.086,
          "history": 0.050, "stack": 0.045, "fb": 0.000}
OLD = {r["dataset"]: r for r in json.load(open(os.path.join(tb.AIDB, "joint_results.json")))}
KS = [0, 1, 4, 16, 64]


def loadruns():
    out = {}
    for p in glob.glob(os.path.join(HERE, "armA_runs", "*.json")):
        r = json.load(open(p))
        out[(r["dataset"], r["mode"], r["k"], r["order"], r.get("nostop", False))] = r
    return out


runs = loadruns()
bands = {}
for p in glob.glob(os.path.join(HERE, "armA_band", "*.json")):
    b = json.load(open(p))
    bands[(b["dataset"], b["mode"], b["k"], b["order"])] = b


def compact(x):
    if "total_uncharged" not in x:
        return x
    return {k: x[k] for k in ("round", "accepted", "model_probes", "gap_table_probes", "gap_k_eff",
                              "total_uncharged", "total_charged", "virtual_count", "sse", "gaps",
                              "gap_factors", "warp_units", "warp_sse", "notes")}


rows = []
for d in tb.DATASETS:
    for m in tb.MODES:
        k0 = runs[(d, m, 0, "GTV", False)]
        k0_best = [x for x in k0["rounds"] if x.get("accepted")][-1]
        for k in KS:
            for o in (["GTV"] if k == 0 else ["GTV", "TGV"]):
                g = runs[(d, m, k, o, False)]
                n = runs[(d, m, k, o, True)]
                assert not g["error"] and not n["error"], (d, m, k, o)
                G = [x for x in g["rounds"] if "total_uncharged" in x]
                N = [x for x in n["rounds"] if "total_uncharged" in x]
                assert len(G) == len([x for x in g["rounds"]]), "failed round in %s" % ((d, m, k, o),)
                r1 = G[0]
                assert r1["total_uncharged"] == N[0]["total_uncharged"]
                acc = [x for x in G if x["accepted"]]
                best = acc[-1]
                nb = min(N, key=lambda x: (x["total_uncharged"], x["round"]))
                nl = min(N, key=lambda x: (x["sse"], x["round"]))
                b = bands.get((d, m, k, o))
                row = {
                    "dataset": d, "mode": m, "k": k, "order": o,
                    "round1": {kk: r1[kk] for kk in ("model_probes", "gap_table_probes", "gap_k_eff",
                                                    "total_uncharged", "total_charged", "virtual_count")},
                    "best": {kk: best[kk] for kk in ("round", "model_probes", "gap_table_probes", "gap_k_eff",
                                                    "total_uncharged", "total_charged", "virtual_count")},
                    "gain_rounds2plus": r1["total_uncharged"] - best["total_uncharged"],
                    "gain_rounds2plus_model_only": r1["model_probes"] - best["model_probes"],
                    "rounds_run": len(G), "accepted_rounds": len(acc),
                    "hit_round_cap": len(acc) == 6,
                    "gapset_changed_in_accepted_round": any(x["gaps"] != r1["gaps"] for x in acc[1:]),
                    "gapset_changed_in_any_candidate": any(x["gaps"] != r1["gaps"] for x in G[1:]),
                    "gapset_changed_nostop": any(x["gaps"] != N[0]["gaps"] for x in N[1:]),
                    "gaps_swapped_best_vs_round1": len(set(best["gaps"]) ^ set(r1["gaps"])),
                    "warp_changed_after_round1": any(x["warp_units"] != r1["warp_units"] for x in acc[1:]),
                    "nostop_series": [x["total_uncharged"] for x in N],
                    "nostop_best": nb["total_uncharged"], "nostop_best_round": nb["round"],
                    "gain_nostop_postselected": r1["total_uncharged"] - nb["total_uncharged"],
                    "nostop_loss_selected": nl["total_uncharged"], "nostop_loss_selected_round": nl["round"],
                    "gain_nostop_loss_selected": r1["total_uncharged"] - nl["total_uncharged"],
                    "band_listed": LISTED.get(d) if m == "uniform" else None,
                    "band_spread": b["spread"] if b else None,
                    "band_maxdev": b["max_dev"] if b else None,
                    "band_lucky": (b["band_base"] - min(b["variants"].values())) if b else None,
                    "k0_best_total": k0_best["total_uncharged"],
                    "delta_best_vs_k0_total": best["total_uncharged"] - k0_best["total_uncharged"],
                    "delta_best_vs_k0_model": best["model_probes"] - k0_best["model_probes"],
                    "rounds_guarded": [compact(x) for x in g["rounds"]],
                }
                if b:
                    assert abs(b["round1_model_probes"] - r1["model_probes"]) < 1e-12, (d, m, k, o)
                gn = row["gain_rounds2plus"]
                for nm in ("listed", "spread", "lucky"):
                    v = row["band_" + nm]
                    row["exceeds_" + nm] = (gn > v) if v is not None else None
                if m == "uniform":
                    bud = OLD[d]["budgets"]["4"]
                    row["lastweek_best_uniform"] = min(bud["csv_only"], bud["sequential"], bud["joint"])
                rows.append(row)

ext = {}
for p in glob.glob(os.path.join(HERE, "armA_extend", "*.json")):
    r = json.load(open(p))
    ext[os.path.basename(p)] = [(x["round"], x["total_uncharged"], x["accepted"]) for x in r["rounds"]]

json.dump({"description": ("Arm A back-and-forth over G/T/V, every (dataset, mode, k, order). "
                           "total_uncharged = model_probes + gap_table_probes; total_charged adds the 6.9-probe tanh charge "
                           "(constant once T is on, so gains are identical in both). Bands measured at the round-1 state."),
           "max_rounds": 6, "tol": 1e-6, "alpha": tb.ALPHA, "select": "current",
           "rows": rows, "extend_check_max_rounds_12": ext,
           "raw_runs_dir": os.path.join(HERE, "armA_runs"), "band_dir": os.path.join(HERE, "armA_band")},
          open(os.path.join(HERE, "armA.json"), "w"), indent=1)


def f(x, p=3):
    return "-" if x is None else ("%.*f" % (p, x))


def yn(v):
    return {True: "YES", False: "no", None: "n/a"}[v]


lines = ["| sample | k | order | r1 model | r1 gapT | r1 total | best total (round) | gain r>=2 | listed band | spread@r1 | lucky@r1 | > spread | > lucky | gap set changed (acc/cand/nostop) | virt r1->best | best - k0best (total / model) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append("| %s-%s | %d | %s | %s | %d | %s | %s (%d) | %s | %s | %s | %s | %s | %s | %s/%s/%s | %d->%d | %+.3f / %+.3f |" % (
        r["dataset"], r["mode"][0], r["k"], r["order"], f(r["round1"]["model_probes"]), r["round1"]["gap_table_probes"],
        f(r["round1"]["total_uncharged"]), f(r["best"]["total_uncharged"]), r["best"]["round"], f(r["gain_rounds2plus"]),
        f(r["band_listed"]), f(r["band_spread"]), f(r["band_lucky"]), yn(r["exceeds_spread"]), yn(r["exceeds_lucky"]),
        "Y" if r["gapset_changed_in_accepted_round"] else "n", "Y" if r["gapset_changed_in_any_candidate"] else "n",
        "Y" if r["gapset_changed_nostop"] else "n", r["round1"]["virtual_count"], r["best"]["virtual_count"],
        r["delta_best_vs_k0_total"], r["delta_best_vs_k0_model"]))
open(os.path.join(HERE, "armA_table.md"), "w").write("\n".join(lines) + "\n")
print(len(rows), "rows;", sum(1 for r in rows if r["band_spread"] is None), "without band")
