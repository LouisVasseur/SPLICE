"""ARM A -- back-and-forth over the three blocks G, T, V.

Per (dataset, mode, k, order):
  round 1 : one coarse-to-fine pass, order "GTV" (G -> T -> V) or "TGV" (T -> G -> V)
  round r>=2 : the same block sequence again, each block in the CURRENT state:
      G re-selects gaps in the current composed coordinate z = f(g(x))
        (threeblock.apply_G select="current", already-shrunk gaps measured unshrunk),
      T refits the warp to the current targets in the current g-coordinate,
      V re-runs the CSV greedy in the current feature with joint.py's
        keep-previous-slots guard (apply_V guard=True).
  Round-level keep-previous guard (joint.py's guard lifted to the whole round):
      a round's candidate is accepted only if total_uncharged improves by more
      than TOL=1e-6; otherwise the previous state is kept and the cycle stops
      (the cycle is deterministic, so repeating a rejected round is pointless).
  Up to MAX_ROUNDS=6 rounds.  Every candidate (accepted or not) is logged.
DIAGNOSTIC "_nostop" runs drop the round guard and the early stop: all 6 rounds
are run, each continuing from the previous candidate, best taken post hoc.
k=0 (no G) is run as a control in order GTV (== T -> V cycle, last week's joint
with a probe-based stop instead of the loss-based one).
Only threeblock's public helpers are used; threeblock.py is not modified.
"""
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import threeblock as tb  # noqa: E402

MAX_ROUNDS = 6
TOL = 1e-6
KS = [1, 4, 16, 64]
ORDERS = ["GTV", "TGV"]
OUT = os.path.join(HERE, "armA_runs")


def gap_set(s):
    return list(s.gaps.idx) if s.gaps is not None else []


def run_pass(s, k, order):
    """one round: apply the three blocks in `order`.  Returns (state, notes)."""
    notes = []
    for b in order:
        if b == "G":
            s = tb.apply_G(s, k)                 # select="current" (sees T when present)
        elif b == "T":
            s = tb.apply_T(s)
            notes.append(s.log[-1][1])
        elif b == "V":
            s = tb.apply_V(s, alpha=tb.ALPHA, guard=True)
            notes.append("V kept_previous=%s" % s.log[-1][3])
    return s, notes


def record(r, s, sc, notes, accepted, t):
    return {
        "round": r,
        "accepted": accepted,
        "model_probes": sc["model_probes"],
        "gap_table_probes": sc["gap_table_probes"],
        "gap_k_eff": sc["gap_k"],
        "total_uncharged": sc["total_uncharged"],
        "total_charged": sc["total_charged"],
        "total_charged_pwl": sc["total_charged_pwl"],
        "virtual_count": sc["virtual_count"],
        "sse": sc["sse"],
        "gaps": gap_set(s),
        "gap_factors": list(s.gaps.s) if s.gaps is not None else [],
        "warp_units": [list(u) for u in s.warp.units] if s.warp is not None else None,
        "warp_sse": s.warp.sse if s.warp is not None else None,
        "notes": notes,
        "seconds": t,
    }


def run(job):
    d, m, k, order = job[:4]
    nostop = len(job) > 4 and job[4]
    path = os.path.join(OUT, "%s_%s_k%d_%s%s.json" % (d, m, k, order, "_nostop" if nostop else ""))
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    res = {"dataset": d, "mode": m, "k": k, "order": order, "nostop": nostop, "rounds": [], "error": None}
    try:
        s0 = tb.initial_state(d, m)
        res["baseline_linear"] = tb.score(s0)["model_probes"]
        cur = None
        cur_sc = None
        for r in range(1, MAX_ROUNDS + 1):
            t1 = time.time()
            base = s0 if cur is None else cur
            try:
                cand, notes = run_pass(base, k, order)
                sc = tb.score(cand)
            except AssertionError as e:
                res["rounds"].append({"round": r, "accepted": False, "failed": str(e)})
                break
            if cur is None:
                acc = True
            else:
                acc = sc["total_uncharged"] < cur_sc["total_uncharged"] - TOL
            res["rounds"].append(record(r, cand, sc, notes, acc, time.time() - t1))
            if nostop:
                # DIAGNOSTIC: no round guard, no early stop -- always continue
                # from the candidate for all MAX_ROUNDS; best is taken post hoc.
                cur, cur_sc = cand, sc
                continue
            if not acc:
                break
            cur, cur_sc = cand, sc
    except Exception:
        res["error"] = traceback.format_exc()
    res["seconds"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1)
    return res


def jobs():
    js = []
    for d in tb.DATASETS:
        for m in tb.MODES:
            js.append((d, m, 0, "GTV"))
            js.append((d, m, 0, "GTV", True))
            for k in KS:
                for o in ORDERS:
                    js.append((d, m, k, o))
                    js.append((d, m, k, o, True))
    heavy = {("osm", "uniform"): 0, ("planet", "uniform"): 1}
    js.sort(key=lambda j: heavy.get((j[0], j[1]), 2))
    return js


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    js = jobs()
    if len(sys.argv) > 1:
        js = [j for j in js if j[0] in sys.argv[1:]]
    n = 0
    with Pool(15) as p:
        for res in p.imap_unordered(run, js):
            n += 1
            last = [x for x in res["rounds"] if x.get("accepted")]
            print("[%d/%d] %s %s k=%d %s%s rounds=%d best=%s err=%s %.0fs" % (
                n, len(js), res["dataset"], res["mode"], res["k"], res["order"], "_nostop" if res["nostop"] else "", len(res["rounds"]),
                ("%.4f" % last[-1]["total_uncharged"]) if last else None,
                bool(res["error"]), res["seconds"]), flush=True)
