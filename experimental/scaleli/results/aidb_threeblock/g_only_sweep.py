"""Cheap preview, NOT an arm: G alone (no T, no V) at k = 0,1,4,16,64.
Reports model_probes and total_uncharged (= model + ceil(log2(k+1)))."""
import json
from multiprocessing import Pool
import threeblock as tb
KS = [0, 1, 4, 16, 64]
def run(dm):
    d, m = dm
    s0 = tb.initial_state(d, m)
    row = {"dataset": d, "mode": m}
    for k in KS:
        sc = tb.score(tb.apply_G(s0, k))
        row["k%d" % k] = (sc["model_probes"], sc["total_uncharged"], sc["gap_k"])
    return row
if __name__ == "__main__":
    jobs = [(d, m) for d in tb.DATASETS for m in tb.MODES]
    with Pool(10) as p:
        rows = p.map(run, jobs)
    print("%-8s %-7s " % ("dataset", "mode") + " ".join("%17s" % ("k=%d mdl/tot" % k) for k in KS))
    for r in rows:
        print("%-8s %-7s " % (r["dataset"], r["mode"]) + " ".join("%8.3f/%8.3f" % r["k%d" % k][:2] for k in KS))
    json.dump(rows, open("g_only_sweep.json", "w"), indent=1)
