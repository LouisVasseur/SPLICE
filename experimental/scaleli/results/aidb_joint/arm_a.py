#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""ARM A of the three-arm design (confound d5): the TRUE sequential baseline,
i.e. the NFL flow trained on the change-of-variables NEGATIVE LOG-LIKELIHOOD
(tools/train_flow.py --monotone), then CSV greedy in its z-space.

Arms B and C (SSE-trained warp with and without alternation) are joint.py's
flow_only / sequential / joint columns.  Comparing A with B isolates "train the
warp on the downstream loss instead of the likelihood"; comparing B with C
isolates "alternate the blocks".

Weight files come from the workspace's own monotone training sweep; this script
only READS them.  Evaluation follows include/scaleli/transform.hpp exactly.
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_metric as pm
import joint as J

FLOWDIR = REPO + "/experimental/scaleli/results/aidb_flowv2/flows_monotone"


def load_flow(path):
    tok = open(path).read().split()
    it = iter(tok)
    in_dim = int(next(it)); hidden = int(next(it)); layers = int(next(it))
    mean = float(next(it)); var = float(next(it))
    W = []
    for l in range(layers):
        rows = int(next(it)); cols = int(next(it))
        W.append((rows, cols, [float(next(it)) for _ in range(rows * cols)]))
    def fn(key):
        x = (float(key) - mean) / var
        a = [x, x - math.floor(x)][:in_dim]
        for li in range(len(W)):
            rows, cols, w = W[li]
            b = [sum(a[r] * w[r * cols + c] for r in range(rows)) for c in range(cols)]
            a = b if li + 1 == len(W) else [math.tanh(v) for v in b]
        z = sum(a)
        return z if math.isfinite(z) else 0.0
    return fn


def main():
    rows = []
    for name in J.DATASETS:
        p = os.path.join(FLOWDIR, "%s_mono.txt" % name)
        if not os.path.exists(p):
            continue
        fn = load_flow(p)
        keys = pm.load_sample(os.path.join(J.SAMPLES, "%s_2M_uniform_s42" % name))
        f = pm.fences(keys, J.REGION_KEYS)
        n = len(f)
        z = [fn(k) for k in f]
        strict = all(z[i + 1] > z[i] for i in range(n - 1))
        sorted_ok = all(z[i + 1] >= z[i] for i in range(n - 1))
        row = {"dataset": name, "strictly_increasing": strict, "sorted": sorted_ok,
               "flow_only": J.score(f, fn)}
        row["sse"] = J.ols_sse(z, [float(i) for i in range(n)])
        if sorted_ok:
            sm = J.smooth_cdf_fast(z, 4.0)
            v = len(sm.virtual_features)
            row["sequential_4x"] = J.score(f, fn, sm.slot, v) if v else row["flow_only"]
            row["virtual"] = v
            row["greedy_rounds"] = sm.rounds
        rows.append(row)
        print("%-8s strict=%-5s flow_only %7.3f  seq4x %7.3f  virt %5d" % (
            name, strict, row["flow_only"], row.get("sequential_4x", float("nan")),
            row.get("virtual", 0)), flush=True)
    json.dump(rows, open(os.path.join(HERE, "arm_a_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
