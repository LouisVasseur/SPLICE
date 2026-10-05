#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Exact emulation of fit_root's probe estimate (locate_from_prediction over fences + midpoints), raw vs flow,
with the sentinel fence 0 included in the fit (as built) and excluded from the fit (counterfactual)."""
import array, math, struct, json, pathlib, statistics, sys
sys.path.insert(0, REPO + "/build-verify-synergy/scratch")
from fence_residuals import load_flow
ROOT = pathlib.Path(REPO + "/experimental/scaleli")
SWEEP = ROOT / "results/aidb_final/sweep/results.jsonl"
OUT = REPO + "/build-verify-synergy/scratch/sentinel.json"
NAMES = ["books", "covid", "fb", "genome", "history", "libio", "osm", "planet", "stack", "wise"]
stored = {}
for l in open(SWEEP):
    r = json.loads(l)
    if r["variant"] == "packed_rank_root_fusion" and r["seed"] == 11:
        L = r["learnability"]; stored[r["dataset"]] = (L["root_probes_binary"], L["root_probes_raw"], L["root_probes_flow"] - 4)
def ols(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x); sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    s = sxy / sxx if sxx > 0 else 0.0; return s, my - s * mx
def locate(fences, y, k):
    n = len(fences); cnt = 0
    def ok(i):
        nonlocal cnt; cnt += 1; return fences[i] <= k
    p = 0 if y <= 0 else (n - 1 if y >= n - 1 else int(y))
    def last_true(lo, hi):
        while hi - lo > 1:
            m = lo + (hi - lo) // 2
            if ok(m): lo = m
            else: hi = m
        return lo
    if ok(p):
        lo, step, hi = p, 1, p + 1
        while hi < n and ok(hi): lo = hi; step *= 2; hi = min(n, p + step)
        last_true(lo, hi); return cnt
    hi, step = p, 1; lo = p - step if p > step else 0
    while lo > 0 and not ok(lo): hi = lo; step *= 2; lo = p - step if p > step else 0
    last_true(lo, hi); return cnt
def estimate(fences, feat, fit_from):
    """feat: key -> feature; fit_from: indices of fences used in the OLS fit; probes over all fences and midpoints"""
    x = [feat(f) for f in fences]; s, b = ols([x[i] for i in fit_from], [float(i) for i in fit_from])
    probes = []
    for j in range(len(fences)):
        probes.append(fences[j])
        if j + 1 < len(fences): probes.append(fences[j] + (fences[j + 1] - fences[j]) // 2)
    return statistics.mean(locate(fences, s * feat(k) + b, k) for k in probes)
def binary_est(fences):
    probes = []
    for j in range(len(fences)):
        probes.append(fences[j])
        if j + 1 < len(fences): probes.append(fences[j] + (fences[j + 1] - fences[j]) // 2)
    tot = 0
    for k in probes:
        lo, hi = 0, len(fences)
        while lo < hi:
            m = lo + (hi - lo) // 2; tot += 1
            if fences[m] <= k: lo = m + 1
            else: hi = m
    return tot / len(probes)
out = {}
print(f"{'dataset':16s} {'sent.gap':>8s} | {'bin':>5s} {'bin_emu':>7s} | raw: {'stored':>6s} {'emu':>6s} {'no-sent':>7s} | flow-4: {'stored':>6s} {'emu':>6s} {'no-sent':>7s}")
for kind in ("uniform", "window"):
    for nm in NAMES:
        d = f"{nm}_{kind}"; p = ROOT / f"data/samples/{nm}_2M_{kind}_s42"
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; keys = array.array("Q"); keys.fromfile(fh, n)
        fences = [0] + [keys[i] for i in range(4096, n, 4096)]
        origin, span = fences[0], max(1, fences[-1] - fences[0])
        raw = lambda k: (k - origin) / span
        flow = load_flow(ROOT / ("results/aidb/flows" if kind == "uniform" else "results/aidb_window/flows") / f"{nm}_2D2H2L.txt")
        allidx = list(range(len(fences))); nosent = allidx[1:]
        # counterfactual raw normalization without the sentinel: origin = first real fence
        o2, sp2 = fences[1], max(1, fences[-1] - fences[1]); raw2 = lambda k: (k - o2) / sp2
        e = {"sentinel_gap_fraction": fences[1] / fences[-1], "binary_stored": stored[d][0], "binary_emu": binary_est(fences),
             "raw_stored": stored[d][1], "raw_emu": estimate(fences, raw, allidx), "raw_fit_without_sentinel": estimate(fences, raw2, nosent),
             "flow_stored": stored[d][2], "flow_emu": estimate(fences, flow, allidx), "flow_fit_without_sentinel": estimate(fences, flow, nosent)}
        out[d] = e
        print(f"{d:16s} {e['sentinel_gap_fraction']:8.5f} | {e['binary_stored']:5.2f} {e['binary_emu']:7.2f} | {e['raw_stored']:6.2f} {e['raw_emu']:6.2f} {e['raw_fit_without_sentinel']:7.2f} | {e['flow_stored']:6.2f} {e['flow_emu']:6.2f} {e['flow_fit_without_sentinel']:7.2f}", flush=True)
json.dump(out, open(OUT, "w"), indent=1); print("JSON ->", OUT)
