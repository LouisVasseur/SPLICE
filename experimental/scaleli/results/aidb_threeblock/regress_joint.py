"""Re-derive last week's E 'joint' (BCD with keep-previous-slots guard) from the
module's blocks only, to show the arms can express it: T -> (V -> T)* ."""
import json, sys
from multiprocessing import Pool
import threeblock as tb
OLD = {r["dataset"]: r for r in json.load(open(tb.AIDB + "/joint_results.json"))}

def run(d):
    s = tb.apply_T(tb.initial_state(d))
    best = None; prev = None
    for r in range(1, 7):
        s = tb.apply_V(s, guard=True)
        sc = tb.score(s)
        if best is None or sc["sse"] < best[0] - 1e-15:
            best = (sc["sse"], sc["model_probes"], r)
        s2 = tb.apply_T(s)
        lf = s2.warp.sse
        stop = prev is not None and not (lf < prev * (1 - 1e-7))
        prev = lf; s = s2
        if stop: break
    b = OLD[d]["budgets"]["4"]
    return d, best[1], b["joint"], best[2], b["joint_selected_round"]

if __name__ == "__main__":
    ds = sys.argv[1:] or ["books", "planet"]
    with Pool(len(ds)) as p:
        for d, new, old, r, ro in p.map(run, ds):
            print("%-7s joint new=%.6f old=%.6f diff=%+.2e round new=%d old=%d" % (d, new, old, new - old, r, ro))
