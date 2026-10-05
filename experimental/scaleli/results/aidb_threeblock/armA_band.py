"""Chaos band of V at arm A's ROUND-1 state (threeblock.perturbed_V_spread,
c4 family eps in {1e-6, 3.33e-5, 1e-4, 1e-3}, x^2 and x^3, both signs).
Round-1 state is rebuilt deterministically (same blocks as armA.run_pass on
the initial state); its model_probes is checked against the armA record."""
import json, os, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import threeblock as tb
import armA
OUT = os.path.join(HERE, "armA_band")

def run(job):
    d, m, k, order = job
    path = os.path.join(OUT, "%s_%s_k%d_%s.json" % (d, m, k, order))
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    s, _ = armA.run_pass(tb.initial_state(d, m), k, order)
    sc = tb.score(s)
    b = tb.perturbed_V_spread(s)
    res = {"dataset": d, "mode": m, "k": k, "order": order, "round1_model_probes": sc["model_probes"],
           "band_base": b["base"], "max_dev": b["max_dev"], "spread": b["spread"], "variants": b["variants"],
           "seconds": time.time() - t0}
    json.dump(res, open(path, "w"), indent=1)
    return res

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    js = []
    for d in tb.DATASETS:
        for m in tb.MODES:
            for k in [0] + armA.KS:
                for o in (["GTV"] if k == 0 else armA.ORDERS):
                    js.append((d, m, k, o))
    if len(sys.argv) > 1:
        js = [j for j in js if j[0] in sys.argv[1:] or "%s_%s" % (j[0], j[1]) in sys.argv[1:]]
    heavy = {("osm", "uniform"): 0, ("planet", "uniform"): 1}
    js.sort(key=lambda j: heavy.get((j[0], j[1]), 2))
    with Pool(int(os.environ.get("NPROC", "8"))) as p:
        for i, r in enumerate(p.imap_unordered(run, js)):
            print("[%d/%d] %s %s k=%d %s spread=%.4f maxdev=%.4f %.0fs" % (i + 1, len(js), r["dataset"], r["mode"], r["k"], r["order"], r["spread"], r["max_dev"], r["seconds"]), flush=True)
