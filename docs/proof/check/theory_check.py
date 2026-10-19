"""Numbers quoted in docs/proof/PROOF.md, section "What the measurements say".

Read-only on the full 200M GRE datasets (data/external/gre, the .sorted copy where one exists, as
scripts/nf_gate.sh does) and the flows of data/flows. Nothing is timed. Run from anywhere:

    /usr/bin/python3 -I docs/proof/check/theory_check.py [m]        # default m = 65536
    -> docs/proof/check/theory_check_m<m>.json, and one line per dataset on stdout

Per dataset:
  sample  m keys at evenly spaced ranks floor(i (n-1)/(m-1)), nf_gate's formula (Theorem 5.3's sample).
  pieces  1 + number of descents of z = T(x) along key order (T monotone <=> pieces == 1).
  ks_*    max_i |u_i - i/(m-1)| with u the min-max normalized values: raw keys; the SORTED flow output
          (what nf_gate's 'ks' gate computes); the flow output in KEY order (Theorem 3.6: includes displacement).
          This statistic lies within 1/m below D_KS of the sample (PROOF.md, Section 7).
  disp    |rank of z_i among the z's - i| / (m-1), max and mean (Theorem 3.6's displacement).
  root    fit_root's least-squares line from feature to region index on min(m, R) evenly spaced region fences
          (nf_gate's root proxy), then the exact probe count of Index::locate_from_prediction for each sample
          key; checked against Theorem 4.3 (count <= 2 clog2(d+1) + 2), Corollary 4.4 (mean <= 2 log2(1+mean d) + 3)
          and the untruncated lower bound 2 log2(d+1) - 1. The JSON keys thm33_* and cor34_* keep the numbering of
          an earlier draft: they are Theorem 4.3 and Corollary 4.4 of PROOF.md.
  teeth   unit intervals of x = (key - mean)/var (the frac feature's pieces): keys per tooth, sign changes of
          T' inside a tooth, and the range of ln|T'| across each tooth (Lambda of Theorem 6.4), on a 2000-point grid.
  fold    regions of 4096 keys that straddle a tooth boundary, and the extra region tail conflict degree
          (transform.hpp tail_conflict_degree, sorted flow vs normalized raw) there, summed and divided by R.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "data", "external", "gre")
FLOWS = os.path.join(REPO, "data", "flows")
PERIOD = {"fb": 512, "osm": 512, "planet": 512}          # scripts/gre_run.sh period(): s512 for fb, osm, planet
DS = ["fb", "osm", "books", "covid", "genome", "history", "libio", "planet", "stack", "wise"]
REGION = 4096


def keys_of(ds):
    f = os.path.join(DATA, ds)
    if os.path.exists(f + ".sorted"):
        f += ".sorted"
    return np.memmap(f, dtype=np.uint64, mode="r", offset=8)


def load_flow(ds):
    tok = iter(open(os.path.join(FLOWS, f"{ds}_s{PERIOD.get(ds, 64)}_t2000.txt")).read().split())
    in_dim, hidden, layers = int(next(tok)), int(next(tok)), int(next(tok))
    mean, var = float(next(tok)), float(next(tok))
    W = []
    for _ in range(layers):
        r, c = int(next(tok)), int(next(tok))
        W.append(np.array([float(next(tok)) for _ in range(r * c)]).reshape(r, c))
    return in_dim, mean, var, W


def flow_apply(keys_u64, flow):
    """src/scaleli/transform.hpp FlowTransform::transform, vectorized."""
    in_dim, mean, var, W = flow
    x = (np.asarray(keys_u64).astype(np.float64) - mean) / var
    a = np.stack([x, x - np.floor(x)], axis=1) if in_dim == 2 else x[:, None]
    for l, w in enumerate(W):
        b = a @ w
        a = b if l + 1 == len(W) else np.tanh(b)
    z = a.sum(axis=1)
    z[~np.isfinite(z)] = 0.0
    return z


def ks_w1(v):
    lo, hi = v.min(), v.max()
    if not hi > lo:
        return 1.0, 0.5
    d = np.abs((v - lo) / (hi - lo) - np.arange(len(v)) / (len(v) - 1))
    return float(d.max()), float(d.mean())


def fit_line(xs, ys):
    """LinearModel::fit_xy's streaming least squares (long double is double on arm64)."""
    mx = my = xx = xy = 0.0
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        k = float(i + 1); dx, dy = xi - mx, yi - my
        mx += dx / k; my += dy / k; xx += dx * (xi - mx); xy += dx * (yi - my)
    sl = xy / xx if xx > 0 else 0.0
    return sl, my - sl * mx


def gallop(y, n, t):
    """Index::locate_from_prediction with ok(i) = (i <= t): (probe count, |p - t|)."""
    c = 0
    def ok(i):
        nonlocal c
        c += 1
        return i <= t
    def last_true(lo, hi):
        while hi - lo > 1:
            m = lo + (hi - lo) // 2
            if ok(m): lo = m
            else: hi = m
    p = 0 if y <= 0 else (n - 1 if y >= n - 1 else int(y))
    if ok(p):
        lo, step, hi = p, 1, p + 1
        while hi < n and ok(hi):
            lo = hi; step *= 2; hi = min(n, p + step)
        last_true(lo, hi)
    else:
        hi, step = p, 1
        lo = p - step if p > step else 0
        while lo > 0 and not ok(lo):
            hi = lo; step *= 2; lo = p - step if p > step else 0
        last_true(lo, hi)
    return c, abs(p - t)


def clog2(v):
    return 0 if v <= 1 else (int(v) - 1).bit_length()


def tail_conflict_degree(x, amp=1.5, tail=0.99):
    """src/scaleli/transform.hpp tail_conflict_degree."""
    n = len(x)
    if n < 2: return 0
    if not x[-1] > x[0]: return n - 1
    slope, _ = fit_line(x, range(n))
    if not slope > 0: slope = n / (x[-1] - x[0])
    ic = -slope * x[0] + 0.5
    size = max(1, min(int(n * amp), int(max(1.0, math.floor(slope * x[-1] + ic) + 1))))
    p = np.clip(np.floor(slope * np.asarray(x) + ic), 0, size - 1).astype(np.int64)
    counts = np.sort(np.unique(p, return_counts=True)[1])
    return int(counts[min(max(0, math.ceil(tail * len(counts)) - 1), len(counts) - 1)] - 1)


def one(ds, m):
    keys = keys_of(ds); n = len(keys); flow = load_flow(ds); _, mean, var, W = flow
    m = min(m, n)
    rank = np.array([i * (n - 1) // (m - 1) for i in range(m)], dtype=np.int64)
    s = np.array(keys[rank]); z = flow_apply(s, flow); x = (s - s[0]).astype(np.float64)
    out = {"n": n, "m": m, "flow": f"{ds}_s{PERIOD.get(ds, 64)}_t2000.txt", "pieces": 1 + int(np.sum(np.diff(z) < 0))}
    out["ks_raw"], out["w1_raw"] = ks_w1(x)
    out["ks_flow_sorted"], out["w1_flow_sorted"] = ks_w1(np.sort(z))
    out["ks_flow_keyorder"], out["w1_flow_keyorder"] = ks_w1(z)
    rz = np.empty(m, dtype=np.int64); rz[np.argsort(z, kind="stable")] = np.arange(m)
    disp = np.abs(rz - np.arange(m)) / (m - 1)
    out["disp_max"], out["disp_mean"] = float(disp.max()), float(disp.mean())
    # root
    R = (n + REGION - 1) // REGION; q = max(2, min(m, R))
    fj = np.array([i * (R - 1) // (q - 1) for i in range(q)], dtype=np.int64)
    fk = np.array(keys[fj * REGION]); t = rank // REGION
    out["R"] = R; out["binary_root"] = math.ceil(math.log2(R)); out["root"] = {}
    for name, ff, fs in (("raw", (fk - s[0]).astype(np.float64), x), ("flow", flow_apply(fk, flow), z)):
        sl, ic = fit_line(ff.tolist(), fj.astype(np.float64).tolist())
        y = sl * fs + ic
        cnt, d = map(np.array, zip(*(gallop(float(y[i]), R, int(t[i])) for i in range(m))))
        ub = np.array([2 * clog2(v + 1) + 2 for v in d])
        out["root"][name] = {"probes": float(cnt.mean()), "thm33_bound_mean": float(ub.mean()), "thm33_violations": int(np.sum(cnt > ub)),
                             "cor34_bound": float(2 * math.log2(1 + d.mean()) + 3), "untruncated_lower_mean": float(np.mean(np.where(d > 0, 2 * np.log2(d + 1.0) - 1, 2))),
                             "mean_d": float(d.mean()), "mean_d_over_R": float(d.mean() / R), "mean_log2_d1": float(np.mean(np.log2(d + 1.0)))}
    # teeth
    xs = (s.astype(np.float64) - mean) / var
    teeth, cnt = np.unique(np.floor(xs), return_counts=True)
    c = W[1].sum(axis=1); W1 = W[0]; ranges = []; sign = 0
    f_ = np.linspace(0, 1, 2001)[:-1]
    with np.errstate(over="ignore"):
        for k in teeth:
            b = np.outer(k + f_, W1[0]) + np.outer(f_, W1[1])
            dz = (np.cosh(b) ** -2) @ (c * (W1[0] + W1[1]))
            if dz.min() < 0 < dz.max(): sign += 1
            a = np.abs(dz[np.abs(dz) > 0])
            if len(a): ranges.append(float(np.log(a.max() / a.min())))
    kpt = n * cnt / m
    out["teeth"] = {"with_keys": int(len(teeth)), "keys_per_tooth_median": float(np.median(kpt)), "keys_per_tooth_max": float(kpt.max()),
                    "sign_changes": sign, "ln_dT_range_median": float(np.median(ranges)), "ln_dT_range_max": float(np.max(ranges))}
    # fold: regions straddling a tooth boundary
    first = np.array(keys[np.arange(R) * REGION]).astype(np.float64)
    last = np.array(keys[np.minimum(np.arange(R) * REGION + REGION - 1, n - 1)]).astype(np.float64)
    straddle = np.nonzero(np.floor((first - mean) / var) != np.floor((last - mean) / var))[0]
    extra = []
    for r in straddle:
        k = np.array(keys[r * REGION:min(n, (r + 1) * REGION)])
        xr = ((k - k[0]).astype(np.float64) / float(max(1, int(k[-1] - k[0])))).tolist()
        extra.append(tail_conflict_degree(np.sort(flow_apply(k, flow)).tolist()) - tail_conflict_degree(xr))
    out["fold"] = {"straddling_regions": int(len(straddle)), "extra_tail_conflicts_median": float(np.median(extra)) if extra else 0.0,
                   "extra_tail_conflicts_per_region": float(np.sum(extra) / R)}
    return out


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    res = {}
    for ds in DS:
        r = res[ds] = one(ds, m); a, b = r["root"]["raw"], r["root"]["flow"]
        print(f"{ds:8s} pieces {r['pieces']:4d} | KS raw {r['ks_raw']:.4f} flow sorted {r['ks_flow_sorted']:.4f} key order {r['ks_flow_keyorder']:.4f}"
              f" | disp mean {r['disp_mean']:.3f} | root raw {a['probes']:.2f} (d/R {a['mean_d_over_R']:.4f}) flow {b['probes']:.2f}"
              f" (d/R {b['mean_d_over_R']:.3f}, lower {b['untruncated_lower_mean']:.2f}) bin {r['binary_root']} viol {a['thm33_violations'] + b['thm33_violations']}"
              f" | teeth {r['teeth']['with_keys']} lnT' range max {r['teeth']['ln_dT_range_max']:.3f} sign {r['teeth']['sign_changes']}"
              f" | straddle {r['fold']['straddling_regions']} extra/R {r['fold']['extra_tail_conflicts_per_region']:.2f}", flush=True)
    with open(os.path.join(HERE, f"theory_check_m{m}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
