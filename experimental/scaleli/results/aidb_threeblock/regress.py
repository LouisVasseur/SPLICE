import json, sys, time
from multiprocessing import Pool
import threeblock as tb
OLD = {r["dataset"]: r for r in json.load(open(tb.AIDB + "/joint_results.json"))}

def run(d):
    t = time.time()
    s0 = tb.initial_state(d, "uniform")
    sT = tb.apply_T(s0)
    sV = tb.apply_V(s0)
    sTV = tb.apply_V(sT)
    o = OLD[d]; b = o["budgets"]["4"]
    rows = []
    for name, st, ref in [("baseline_linear", s0, o["baseline_linear"]), ("flow_only", sT, o["flow_only"]),
                          ("csv_only", sV, b["csv_only"]), ("sequential", sTV, b["sequential"])]:
        sc = tb.score(st)
        rows.append((d, name, sc["model_probes"], ref, sc["model_probes"] - ref, sc["virtual_count"], st.log))
    return rows, time.time() - t

if __name__ == "__main__":
    ds = sys.argv[1:] or ["books", "planet"]
    with Pool(len(ds)) as p:
        res = p.map(run, ds)
    out = []
    for rows, sec in res:
        for r in rows:
            print("%-7s %-16s new=%.6f old=%.6f diff=%+.2e virt=%d log=%s" % r)
            out.append(dict(zip(["dataset", "method", "new", "old", "diff", "virt"], r[:6])))
        print("  seconds %.1f" % sec)
    json.dump(out, open("regression_%s.json" % "_".join(ds), "w"), indent=1)
