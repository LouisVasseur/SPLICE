"""VERIFIER item 3: strict monotonicity of f(g(x)) on fences + floor midpoints,
recomputed with independent code for every stored arm-A round and every arm-B
fixed-order variant with G."""
import json, os, sys, glob
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); TB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from v_indep import my_gmap, my_warp, my_factors, I, S


def fences(d, m):
    return I.fences(I.load(os.path.join(S, "%s_2M_%s_s42" % (d, m))))


def check(f, idx, fac, units):
    o = f[0]; sp = float(max(1, f[-1] - o)); xs = [float(q - o) / sp for q in f]
    g = my_gmap(xs, idx, fac) if idx else (lambda x: x)
    w = my_warp(units) if units else (lambda u: u)
    pk = I.probe_keys(f); v = [w(g(float(q - o) / sp)) for q in pk]
    bad = [i for i in range(len(pk) - 1) if pk[i + 1] > pk[i] and not v[i + 1] > v[i]]
    ties = sum(1 for i in range(len(pk) - 1) if pk[i + 1] == pk[i])
    gaps_min = min(v[i + 1] - v[i] for i in range(len(pk) - 1) if pk[i + 1] > pk[i])
    return len(bad), ties, gaps_min


def job(dm):
    d, m = dm
    f = fences(d, m)
    out = {"A_states": 0, "A_bad": [], "B_states": 0, "B_bad": [], "min_step": float("inf"), "key_ties": None, "A_failed_rounds": []}
    for p in glob.glob(os.path.join(TB, "armA_runs", "%s_%s_k*.json" % (d, m))):
        run = json.load(open(p))
        for x in run["rounds"]:
            if "failed" in x:
                out["A_failed_rounds"].append((os.path.basename(p), x["round"], x["failed"])); continue
            nb, ties, mn = check(f, x["gaps"], x["gap_factors"], x["warp_units"])
            out["A_states"] += 1; out["key_ties"] = ties; out["min_step"] = min(out["min_step"], mn)
            if nb: out["A_bad"].append((os.path.basename(p), x["round"], nb))
    B = json.load(open(os.path.join(TB, "armB.json")))
    s = [s for s in B["samples"] if s["dataset"] == d and s["mode"] == m][0]
    o = f[0]; sp = float(max(1, f[-1] - o)); xs = [float(q - o) / sp for q in f]
    rw = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    for k, pk in s["per_k"].items():
        k = int(k)
        part = json.load(open(os.path.join(TB, "armB_parts", "k_%s_%s_%d.json" % (d, m, k))))
        for nm, v in pk["variants"].items():
            if nm in part:
                assert part[nm]["model_probes"] == v["model_probes"]
                v = part[nm]
            if "model_probes" not in v or v.get("gap_k", 0) == 0: continue
            if nm in ("G", "G->V", "G->T", "G->T->V"):
                sel, fac = my_factors(rw, k)
            elif nm in ("T->G", "T->G->V"):
                wf = my_warp(v["warp"]); z = [wf(x) for x in xs]
                sel, fac = my_factors([z[i + 1] - z[i] for i in range(len(z) - 1)], k)
            else:
                out.setdefault("B_unchecked", []).append((k, nm)); continue
            if sel != v["gaps"]:
                out.setdefault("B_gapset_mismatch", []).append((k, nm))
            nb, ties, mn = check(f, sel, fac, v["warp"])
            out["B_states"] += 1; out["min_step"] = min(out["min_step"], mn)
            if nb: out["B_bad"].append((k, nm, nb))
    return (d, m), out


if __name__ == "__main__":
    import threeblock as tb
    jobs = [(d, m) for d in tb.DATASETS for m in tb.MODES]
    res = {}
    with Pool(6) as p:
        for (d, m), o in p.imap_unordered(job, jobs):
            res["%s_%s" % (d, m)] = o
            print(d, m, o["A_states"], len(o["A_bad"]), o["B_states"], len(o["B_bad"]), "min_step=%.3g" % o["min_step"], o.get("B_unchecked"), o.get("B_gapset_mismatch"), o["A_failed_rounds"], flush=True)
    json.dump(res, open(os.path.join(HERE, "v_mono.json"), "w"), indent=1)
    print("TOTAL A states", sum(o["A_states"] for o in res.values()), "bad", sum(len(o["A_bad"]) for o in res.values()),
          "B states", sum(o["B_states"] for o in res.values()), "bad", sum(len(o["B_bad"]) for o in res.values()))
