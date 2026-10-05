"""VERIFIER item 1: the greedy's own band for the WHOLE pass.
Perturb the raw normalised feature x -> p(x) = x +/- 1e-6 x^2, x +/- 1e-6 x^3
BEFORE G, then run arm A's exact guarded cycle (round 1 = the single pass;
rounds >= 2 = back-and-forth).  Identity uses threeblock's base State so it must
reproduce armA_runs bit for bit."""
import json, os, sys, time, traceback
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.path.dirname(HERE)
sys.path.insert(0, TB)
import threeblock as tb
import armA

OUT = os.path.join(HERE, "band_runs2")
PERT = {"id": None}
for _e in (1e-6, 3.33e-5, 1e-4, 1e-3):
    PERT["+x2@%g" % _e] = (lambda e: (lambda x: x + e * x * x))(_e)
    PERT["-x2@%g" % _e] = (lambda e: (lambda x: x - e * x * x))(_e)
    PERT["+x3@%g" % _e] = (lambda e: (lambda x: x + e * x * x * x))(_e)
    PERT["-x3@%g" % _e] = (lambda e: (lambda x: x - e * x * x * x))(_e)


class PState(tb.State):
    def __init__(self, f, pname, gaps=None, warp=None, slots=None, virt=0, log=None):
        tb.State.__init__(self, f, gaps=gaps, warp=warp, slots=slots, virt=virt, log=log)
        self.pname = pname
        p = PERT[pname]
        self.p = p
        self.xs = [p(x) for x in self.xs]

    def copy(self, **kw):
        d = dict(gaps=self.gaps, warp=self.warp, slots=self.slots, virt=self.virt, log=self.log)
        d.update(kw)
        return PState(self.f, self.pname, **d)

    def feature_fn(self):
        o, sp, p = self.origin, float(self.span), self.p
        g, w = self.gaps, self.warp
        def fn(k):
            u = p(float(k - o) / sp)
            if g is not None:
                u = g(u)
            if w is not None:
                u = w(u)
            return u
        return fn


def cycle(s0, k, order):
    cur = None; cur_sc = None; rounds = []
    for r in range(1, armA.MAX_ROUNDS + 1):
        base = s0 if cur is None else cur
        try:
            cand, _ = armA.run_pass(base, k, order)
            sc = tb.score(cand)
        except AssertionError as e:
            rounds.append({"round": r, "failed": str(e)}); break
        acc = True if cur is None else sc["total_uncharged"] < cur_sc["total_uncharged"] - armA.TOL
        rounds.append({"round": r, "accepted": acc, "model": sc["model_probes"], "gapT": sc["gap_table_probes"],
                       "k_eff": sc["gap_k"], "total": sc["total_uncharged"], "virt": sc["virtual_count"],
                       "gaps": list(cand.gaps.idx) if cand.gaps is not None else []})
        if not acc:
            break
        cur, cur_sc = cand, sc
    return rounds


def run(job):
    d, m, k, order, pn = job
    path = os.path.join(OUT, "%s_%s_k%d_%s_%s.json" % (d, m, k, order, pn))
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    res = {"dataset": d, "mode": m, "k": k, "order": order, "pert": pn}
    try:
        f = tb.load(d, m)
        s0 = tb.State(f) if pn == "id" else PState(f, pn)
        res["rounds"] = cycle(s0, k, order)
    except Exception:
        res["error"] = traceback.format_exc()
    res["seconds"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    A = json.load(open(os.path.join(TB, "armA.json")))
    cells = [(r["dataset"], r["mode"], r["k"], r["order"]) for r in A["rows"] if r["gain_rounds2plus"] >= 0.01]
    js = [c + (pn,) for c in cells for pn in PERT]
    heavy = {("osm", "uniform"): 0, ("planet", "uniform"): 1}
    js.sort(key=lambda j: heavy.get((j[0], j[1]), 2))
    print(len(cells), "cells", len(js), "jobs", flush=True)
    with Pool(int(os.environ.get("NPROC", "12"))) as p:
        for i, r in enumerate(p.imap_unordered(run, js)):
            rr = r.get("rounds", [])
            print("[%d/%d] %s %s k=%d %s %s r1=%s err=%s %.0fs" % (i + 1, len(js), r["dataset"], r["mode"], r["k"], r["order"], r["pert"],
                  rr[0].get("model") if rr else None, "error" in r, r["seconds"]), flush=True)
