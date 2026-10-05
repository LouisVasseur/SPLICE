#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Run scaleli_bench directly with larger root fence budgets; append the JSON rows to scratch/budget.jsonl."""
import json, subprocess, sys, time, pathlib
BIN = REPO + "/build-final/experimental/scaleli/scaleli_bench"
ROOT = pathlib.Path(REPO + "/experimental/scaleli")
OUT = pathlib.Path(REPO + "/build-verify-synergy/scratch/budget.jsonl")
def run(ds, alpha, flow, ops=100000, warmup=10000, seed=11):
    name, kind = ds.rsplit("_", 1)
    data = ROOT / f"data/samples/{name}_2M_{kind}_s42"
    fw = ROOT / ("results/aidb/flows" if kind == "uniform" else "results/aidb_window/flows") / f"{name}_2D2H2L.txt"
    opts = {"load-ratio": 1, "miss": 0, "query-distribution": "uniform", "ops": ops, "warmup": warmup, "verify": 0, "instrument": 1, "latency": 0,
            "build-threads": 16, "qos": 1, "profile": "read_only", "seed": seed, "data": str(data), "format": "sosd", "dtype": "uint64",
            "policy": "min_bytes", "routing": "rank", "root": "model", "root-alpha": alpha}
    if flow: opts["flow"] = str(fw)
    cmd = [BIN]
    for k, v in opts.items(): cmd += ["--" + k, str(v)]
    t = time.time(); p = subprocess.run(cmd, text=True, capture_output=True, timeout=1800, cwd=ROOT)
    if p.returncode: print("FAILED", ds, alpha, flow, p.stderr[:500]); return None
    r = json.loads(p.stdout); r.update(dataset=ds, root_alpha_req=alpha, with_flow=flow, wall_s=time.time() - t)
    with OUT.open("a") as f: f.write(json.dumps(r) + "\n")
    L = r["learnability"]; w = r["work_counters"]
    print(f"{ds:16s} alpha={alpha:<3} flow={int(flow)} wall={time.time()-t:6.1f}s  est bin/raw/flow/vp_raw/vp_flow = {L['root_probes_binary']:.2f}/{L['root_probes_raw']:.2f}/{L['root_probes_flow']:.2f}/{L['root_probes_vp_raw']:.2f}/{L['root_probes_vp_flow']:.2f}  chosen model={L['root_model']} flow={L['root_flow']} vp={L['root_vp']} virt={L['root_virtual']}  measured root/op={w['root_probes']/r['operations']:.2f}", flush=True)
    return r
if __name__ == "__main__":
    for spec in sys.argv[1:]:
        ds, alpha, flow = spec.split(":"); run(ds, float(alpha), flow == "1")
