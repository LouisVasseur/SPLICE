"""Precision control: the C++ smooth_cdf accumulates in long double (64-bit
mantissa), Python only has double (53).  The greedy's stopping rule compares at
a relative tolerance of 1e-12, which is finer than the cancellation noise of the
double closed form, so the double run can stop early.  This re-runs csv_only at
lambda = 4x with probe_metric's double-double accumulator and reports both."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import probe_metric as pm, joint as J

def go(ds):
    k = pm.load_sample(os.path.join(J.SAMPLES, "%s_2M_uniform_s42" % ds))
    f = pm.fences(k, J.REGION_KEYS); x, o, sp = pm.raw_features(f)
    raw = lambda kk, _o=o, _s=float(sp): float(kk - _o) / _s
    out = {"dataset": ds}
    for tag, sm in (("double", J.smooth_cdf_fast(x, 4.0)),
                    ("dd", pm.smooth_cdf(x, 4.0, precision="dd"))):
        v = len(sm.virtual_features)
        out[tag] = J.score(f, raw, sm.slot, v) if v else J.score(f, raw)
        out[tag + "_rounds"] = sm.rounds
    return out

if __name__ == "__main__":
    from multiprocessing import Pool
    names = sys.argv[1:] or J.DATASETS
    with Pool(len(names)) as p:
        rows = p.map(go, names)
    for r in rows:
        print("%-8s double %7.3f (%5d)   dd %7.3f (%5d)" % (
            r["dataset"], r["double"], r["double_rounds"], r["dd"], r["dd_rounds"]), flush=True)
    json.dump(rows, open(os.path.join(HERE, "prec_check.json"), "w"), indent=1)
