#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Why the raw feature fails on windows: OLS root model over the 489 fences, residuals in regions, raw vs flow feature."""
import array, math, struct, json, pathlib, statistics
ROOT = pathlib.Path(REPO + "/experimental/scaleli")
OUT = REPO + "/build-verify-synergy/scratch/fence_residuals.json"
NAMES = ["books", "covid", "fb", "genome", "history", "libio", "osm", "planet", "stack", "wise"]
def load_flow(p):
    toks = open(p).read().split(); it = iter(toks)
    in_dim, hidden, layers = int(next(it)), int(next(it)), int(next(it)); mean, var = float(next(it)), float(next(it))
    Ws = []
    for l in range(layers):
        r, c = int(next(it)), int(next(it)); Ws.append((r, c, [float(next(it)) for _ in range(r * c)]))
    def f(k):
        x = (k - mean) / var; a = [x, x - math.floor(x)] if in_dim == 2 else [x]
        for l, (r, c, w) in enumerate(Ws):
            b = [sum(a[i] * w[i * c + j] for i in range(r)) for j in range(c)]
            a = b if l + 1 == layers else [math.tanh(v) for v in b]
        return sum(a)
    return f
def ols(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x); sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sxx if sxx > 0 else 0.0; return slope, my - slope * mx
def stats(x, y):
    s, b = ols(x, y); res = [abs(s * xi + b - yi) for xi, yi in zip(x, y)]
    return {"max_resid_regions": max(res), "rms_resid_regions": math.sqrt(sum(r * r for r in res) / len(res)),
            "median_resid_regions": statistics.median(res), "frac_within_1": sum(r <= 1 for r in res) / len(res),
            "mean_2log2": statistics.mean(2 * math.log2(r + 1) for r in res)}  # crude exponential-search proxy
out = {}
print(f"{'dataset':16s} | raw: max / rms / median resid (regions), frac<=1, 2log2 | flow: max / rms / median, frac<=1, 2log2 | monotone flow on fences?")
for kind in ("uniform", "window"):
    for nm in NAMES:
        d = f"{nm}_{kind}"; p = ROOT / f"data/samples/{nm}_2M_{kind}_s42"
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; keys = array.array("Q"); keys.fromfile(fh, n)
        fences = [0] + [keys[i] for i in range(4096, n, 4096)]
        ranks = list(range(len(fences)))
        origin, span = fences[0], max(1, fences[-1] - fences[0])
        xr = [(k - origin) / span for k in fences]
        flow = load_flow(ROOT / ("results/aidb/flows" if kind == "uniform" else "results/aidb_window/flows") / f"{nm}_2D2H2L.txt")
        xf = [flow(k) for k in fences]
        mono = all(xf[i] < xf[i + 1] for i in range(len(xf) - 1))
        sr, sf = stats(xr, ranks), stats(xf, ranks)
        out[d] = {"regions": len(fences), "raw": sr, "flow": sf, "flow_monotone_on_fences": mono, "key_span": span}
        print(f"{d:16s} | {sr['max_resid_regions']:6.1f} / {sr['rms_resid_regions']:6.1f} / {sr['median_resid_regions']:6.1f}, {sr['frac_within_1']:.2f}, {sr['mean_2log2']:5.1f} | {sf['max_resid_regions']:6.1f} / {sf['rms_resid_regions']:6.1f} / {sf['median_resid_regions']:6.1f}, {sf['frac_within_1']:.2f}, {sf['mean_2log2']:5.1f} | {mono}")
json.dump(out, open(OUT, "w"), indent=1); print("JSON ->", OUT)
