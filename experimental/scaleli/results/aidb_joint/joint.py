#!/usr/bin/env python3
"""Joint NFL-style input warp + CSV virtual points at the ROOT of the index.

Five methods, all scored with the SAME probe metric (probe_metric.root_probes,
a faithful replica of Index::fit_root's candidate score, index.hpp:389-392):

  A baseline_linear  one OLS line over (normalized fence, rank).
  B csv_only         CSV greedy virtual fences on the RAW fence feature.
  C flow_only        order-preserving 2-unit tanh warp, no virtual fences.
  D sequential       fit the warp (to plain ranks), then CSV greedy in z-space.
  E joint            block coordinate descent: alternate (fit warp given the
                     current slot targets) and (re-run CSV greedy given the
                     current warp).  Round 1 IS method D by construction, so
                     everything joint buys comes from rounds >= 2.

OBJECTIVE, and the fact that we changed it.  tools/train_flow.py minimises the
NFL change-of-variables negative log-likelihood (density matching to a standard
normal).  That is not the quantity the root cares about.  Methods C/D/E instead
minimise the downstream least-squares objective

      L(f, V) = min_{a,b} sum_i ( a*f(x_i) + b - s_i )^2                 (*)

over the REAL fences only, with s_i the slot ranks induced by the virtual set V
(s_i = i when V is empty).  This is deliberate: it is the objective of the
brief, and it is the one whose minimiser the probe count actually tracks.  The
consequence is that "joint vs sequential" here is a clean test of the
ALTERNATION only -- both use the same warp family, the same warp objective and
the same virtual-point operator.  The separate question "does training the warp
on (*) instead of the NLL help at all" is answered by the flow_only column.

WARP FAMILY.  The deployed NFL shape is 2 input dims / 2 hidden / 2 layers with
a sum decoder; with the fractional-feature weights zeroed (the monotone variant,
train_flow.py --monotone) it collapses exactly to

      z(x) = C1*tanh(g1*x) + C2*tanh(g2*x),      x = (key - f0) / span

because only the products (output weight * decoder sum) are identifiable.  We
keep C1, C2 >= 0 and g1, g2 > 0, which makes z order-preserving.  Given
(g1, g2) the fit of (C1, C2, b) against (*) is an exact 3x3 normal-equation
solve (non-negativity handled by trying the two active sets), so the warp block
is a 2-D grid search around a closed form -- no gradient descent, no learning
rate, no seed.

STRICT MONOTONICITY IS ENFORCED, NOT ASSUMED.  index.hpp:382 guards the slot
table with std::is_sorted, which ACCEPTS TIES; a saturating warp that collapses
several fences onto one double would pass that guard and then cap the achievable
accuracy forever.  Every warp candidate here is rejected unless z is STRICTLY
increasing in double over the fences.

Standard library only.
"""
from __future__ import annotations
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_metric as pm  # noqa: E402

SAMPLES = REPO + "/experimental/scaleli/data/samples"
DATASETS = ["books", "fb", "osm", "covid", "genome", "history", "libio", "planet", "stack", "wise"]
REGION_KEYS = 4096
ALPHAS = [0.5, 1.0, 4.0]                  # lambda = alpha * (#fences)
MATCH_LADDER = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]  # budgets tried for "match csv_only@4x"
MAX_ROUNDS = 6
REL_TOL = 1e-7


# ---------------------------------------------------------------------------
# fast CSV greedy: bit-identical to probe_metric.smooth_cdf (double precision),
# with the per-candidate arithmetic inlined.  Verified by verify.py.
# ---------------------------------------------------------------------------

def smooth_cdf_fast(x, alpha, max_ternary_steps=40):
    n = len(x)
    out = pm.SmoothingResult(n)
    seq_x = [float(v) for v in x]
    seq_v = [False] * n

    def total():
        sn = sx = sxx = sy = syy = sxy = 0.0
        for i in range(len(seq_x)):
            v = seq_x[i]
            fi = float(i)
            sn += 1.0
            sx += v
            sxx += v * v
            sy += fi
            syy += fi * fi
            sxy += v * fi
        return sn, sx, sxx, sy, syy, sxy

    def sse_of(sn, sx, sxx, sy, syy, sxy):
        if sn < 2:
            return 0.0
        cxx = sxx - sx * sx / sn
        cxy = sxy - sx * sy / sn
        cyy = syy - sy * sy / sn
        s = (cyy - cxy * cxy / cxx) if cxx > 0 else cyy
        return s if s > 0 else 0.0

    bn, bsx, bsxx, bsy, bsyy, bsxy = total()
    out.sse_before = sse_of(bn, bsx, bsxx, bsy, bsyy, bsxy)
    out.sse_after = out.sse_before
    budget = int(alpha * float(n))
    if n < 3 or not budget:
        return out

    current = out.sse_before
    for _ in range(budget):
        m = len(seq_x)
        suf_x = [0.0] * (m + 1)
        suf_y = [0.0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suf_x[i] = suf_x[i + 1] + seq_x[i]
            suf_y[i] = suf_y[i + 1] + float(i)

        best = current
        best_i = 0
        best_x = 0.0
        found = False
        nf = bn + 1.0
        for i in range(m - 1):
            lo = seq_x[i]
            hi = seq_x[i + 1]
            if not (hi > lo):
                continue
            # terms constant in xv for this gap
            cnt = float(m - (i + 1))
            yv = float(i + 1)
            sy_f = (bsy + cnt) + yv
            syy_f = (bsyy + (2.0 * suf_y[i + 1] + cnt)) + yv * yv
            sxy_b = bsxy + suf_x[i + 1]
            cyy = syy_f - sy_f * sy_f / nf

            def L(xv, _nf=nf, _bsx=bsx, _bsxx=bsxx, _sxy_b=sxy_b, _yv=yv,
                  _cyy=cyy, _syf=sy_f):
                sx = _bsx + xv
                sxx = _bsxx + xv * xv
                sxy = _sxy_b + xv * _yv
                cxx = sxx - sx * sx / _nf
                cxy = sxy - sx * _syf / _nf
                s = (_cyy - cxy * cxy / cxx) if cxx > 0 else _cyy
                return s if s > 0 else 0.0

            w = hi - lo
            e = w * 1e-3
            l0 = L(lo + e)
            l1 = L(lo + 2 * e)
            r1 = L(hi - 2 * e)
            r0 = L(hi - e)
            if l1 < l0 and r1 < r0:
                a, b = lo + e, hi - e
                it = 0
                while it < max_ternary_steps and b - a > e:
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    if L(m1) < L(m2):
                        b = m2
                    else:
                        a = m1
                    it += 1
                cx = (a + b) / 2.0
                cl = L(cx)
            elif l0 <= r0:
                cx, cl = lo + e, l0
            else:
                cx, cl = hi - e, r0
            if cl < best - 1e-12 * max(1.0, best):
                best, best_i, best_x, found = cl, i, cx, True
        if not found:
            break
        seq_x.insert(best_i + 1, best_x)
        seq_v.insert(best_i + 1, True)
        out.virtual_features.append(best_x)
        bn, bsx, bsxx, bsy, bsyy, bsxy = total()
        current = sse_of(bn, bsx, bsxx, bsy, bsyy, bsxy)
        out.rounds += 1

    k = 0
    for i in range(len(seq_x)):
        if not seq_v[i]:
            out.slot[k] = i
            k += 1
    out.sse_after = current
    return out


# ---------------------------------------------------------------------------
# the objective (*): OLS SSE of the refitted line over the REAL fences
# ---------------------------------------------------------------------------

def ols_sse(z, s):
    n = float(len(z))
    sx = sy = sxx = syy = sxy = 0.0
    for i in range(len(z)):
        a = z[i]
        b = s[i]
        sx += a
        sy += b
        sxx += a * a
        syy += b * b
        sxy += a * b
    cxx = sxx - sx * sx / n
    cxy = sxy - sx * sy / n
    cyy = syy - sy * sy / n
    v = (cyy - cxy * cxy / cxx) if cxx > 0 else cyy
    return v if v > 0 else 0.0


# ---------------------------------------------------------------------------
# the warp block: exact 3x3 solve inside a grid over the gains (and centres)
# ---------------------------------------------------------------------------

class WarpBasis:
    """Pre-tabulated tanh units over the normalized fences.

    unit u has parameters (gain g_u, centre m_u) and vector T_u[i] = tanh(g_u*(x_i - m_u)).
    Everything that does not depend on the targets -- sum T_u, <T_u, T_v>, and
    the strict-increase mask -- is computed once per dataset.
    """

    def __init__(self, xs, gains, centres=(0.0,)):
        self.xs = xs
        self.n = len(xs)
        self.params = [(g, m) for m in centres for g in gains]
        self.T = []
        self.ok = []
        for (g, m) in self.params:
            t = [math.tanh(g * (v - m)) for v in xs]
            self.T.append(t)
            self.ok.append(all(t[i + 1] > t[i] for i in range(len(t) - 1)))
        k = len(self.params)
        self.S = [sum(t) for t in self.T]
        self.Q = [[0.0] * k for _ in range(k)]
        for a in range(k):
            ta = self.T[a]
            for b in range(a, k):
                tb = self.T[b]
                v = 0.0
                for i in range(self.n):
                    v += ta[i] * tb[i]
                self.Q[a][b] = v
                self.Q[b][a] = v
        # only pairs where at least one unit is strictly increasing can give a
        # strictly increasing combination with non-negative coefficients
        self.pairs = [(a, b) for a in range(k) for b in range(a, k)
                      if self.ok[a] or self.ok[b]]

    def fit(self, s):
        """argmin over (unit pair, C1, C2 >= 0, b) of sum (C1 T_a + C2 T_b + b - s)^2,
        subject to the combination being strictly increasing in double.
        Returns (sse, [(gain, centre, coef), ...]) or None."""
        n = float(self.n)
        Ss = sum(s)
        P = []
        for a in range(len(self.params)):
            ta = self.T[a]
            v = 0.0
            for i in range(self.n):
                v += ta[i] * s[i]
            P.append(v)
        Sss = 0.0
        for v in s:
            Sss += v * v
        cyy = Sss - Ss * Ss / n

        def cov(a, b):
            return self.Q[a][b] - self.S[a] * self.S[b] / n

        def covs(a):
            return P[a] - self.S[a] * Ss / n

        cands = []
        for (a, b) in self.pairs:
            caa = cov(a, a)
            cbb = cov(b, b)
            cab = cov(a, b)
            cas = covs(a)
            cbs = covs(b)
            best = None
            if a != b:
                det = caa * cbb - cab * cab
                if det > 0:
                    C1 = (cbb * cas - cab * cbs) / det
                    C2 = (caa * cbs - cab * cas) / det
                    if C1 >= 0 and C2 >= 0:
                        r = cyy - (C1 * cas + C2 * cbs)
                        best = (r if r > 0 else 0.0, C1, C2)
            if best is None:
                opts = []
                if caa > 0:
                    C1 = cas / caa
                    if C1 >= 0:
                        r = cyy - C1 * cas
                        opts.append((r if r > 0 else 0.0, C1, 0.0))
                if b != a and cbb > 0:
                    C2 = cbs / cbb
                    if C2 >= 0:
                        r = cyy - C2 * cbs
                        opts.append((r if r > 0 else 0.0, 0.0, C2))
                if opts:
                    best = min(opts)
            if best is None:
                continue
            cands.append((best[0], a, b, best[1], best[2]))
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        # walk down the candidate list until one is strictly increasing
        for (sse, a, b, C1, C2) in cands[:400]:
            if C1 <= 0 and C2 <= 0:
                continue
            ta = self.T[a]
            tb = self.T[b]
            good = True
            prev = C1 * ta[0] + C2 * tb[0]
            for i in range(1, self.n):
                cur = C1 * ta[i] + C2 * tb[i]
                if not (cur > prev):
                    good = False
                    break
                prev = cur
            if good:
                out = []
                if C1 > 0:
                    out.append((self.params[a][0], self.params[a][1], C1))
                if C2 > 0:
                    out.append((self.params[b][0], self.params[b][1], C2))
                return (sse, out)
        return None


def make_flow_fn(units, origin, span):
    """feature_fn(key) -> z, valid for arbitrary keys (probe midpoints included)."""
    inv = 1.0 / float(span)
    tanh = math.tanh

    def fn(k, _u=units, _o=origin, _inv=inv):
        x = float(k - _o) * _inv
        z = 0.0
        for (g, m, c) in _u:
            z += c * tanh(g * (x - m))
        return z
    return fn


# ---------------------------------------------------------------------------
# scoring one configuration with the fit_root probe metric
# ---------------------------------------------------------------------------

def score(f, feature_fn, slots=None, virt=0):
    n = len(f)
    x = [feature_fn(k) for k in f]
    if slots is None:
        targets = [float(j) for j in range(n)]
        table = None
    else:
        targets = [float(s) for s in slots]
        table = pm.slot_to_region_table(slots, n, virt) if virt else None
    model = pm.LinearModel().fit_xy(f, x, targets)
    return pm.root_probes(f, lambda k: model.predict_x(feature_fn(k)), table, 0.0)


# ---------------------------------------------------------------------------
# the five methods on one dataset
# ---------------------------------------------------------------------------

GAINS = [math.exp(math.log(0.01) + (math.log(200.0) - math.log(0.01)) * i / 63.0) for i in range(64)]
GAINS_B = [math.exp(math.log(0.05) + (math.log(80.0) - math.log(0.05)) * i / 13.0) for i in range(14)]
CENTRES_B = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.95]


def run_dataset(name, do_bias=True):
    t_start = time.time()
    keys = pm.load_sample(os.path.join(SAMPLES, "%s_2M_uniform_s42" % name))
    f = pm.fences(keys, REGION_KEYS)
    n = len(f)
    origin = f[0]
    span = max(1, f[-1] - origin)
    xs = [float(k - origin) / float(span) for k in f]
    raw_fn = lambda k, _o=origin, _s=float(span): float(k - _o) / _s
    ranks = [float(j) for j in range(n)]

    res = {"dataset": name, "n_fences": n, "binary": pm.binary_probes(n)}

    # --- A baseline_linear -------------------------------------------------
    res["baseline_linear"] = score(f, raw_fn)
    res["linear_sse"] = ols_sse(xs, ranks)

    basis = WarpBasis(xs, GAINS)

    # --- C flow_only (warp fitted to PLAIN ranks, i.e. V empty) ------------
    fit0 = basis.fit(ranks)
    if fit0 is None:
        res["flow_only"] = res["baseline_linear"]
        res["flow_units"] = []
        res["flow_sse"] = res["linear_sse"]
        flow0 = raw_fn
        units0 = []
    else:
        sse0, units0 = fit0
        flow0 = make_flow_fn(units0, origin, span)
        res["flow_only"] = score(f, flow0)
        res["flow_units"] = units0
        res["flow_sse"] = sse0
    res["flow_sse_ratio"] = (res["flow_sse"] / res["linear_sse"]) if res["linear_sse"] > 0 else 1.0

    res["budgets"] = {}
    for alpha in ALPHAS:
        res["budgets"]["%g" % alpha] = one_budget(f, xs, raw_fn, basis, units0, flow0,
                                                  origin, span, alpha, ranks, res["flow_sse"])

    # --- smallest budget at which joint matches csv_only at 4x -------------
    target = res["budgets"]["4"]["csv_only"]
    matched = None
    cache = {}
    for a in MATCH_LADDER:
        key = "%g" % a
        if key in res["budgets"]:
            jv = res["budgets"][key]["joint"]
        else:
            r = one_budget(f, xs, raw_fn, basis, units0, flow0, origin, span, a, ranks,
                           res["flow_sse"], only_joint=True)
            cache[key] = r
            jv = r["joint"]
        if jv <= target + 1e-12:
            matched = a
            break
    res["match_extra"] = cache
    res["joint_budget_to_match_csv4x"] = matched

    # --- supplementary: the per-unit-bias warp family ----------------------
    if do_bias:
        bb = WarpBasis(xs, GAINS_B, CENTRES_B)
        fb = bb.fit(ranks)
        if fb is not None:
            sseb, unitsb = fb
            flowb = make_flow_fn(unitsb, origin, span)
            res["bias_flow_only"] = score(f, flowb)
            res["bias_flow_sse"] = sseb
            res["bias_units"] = unitsb
            rb = one_budget(f, xs, raw_fn, bb, unitsb, flowb, origin, span, 4.0, ranks, sseb)
            res["bias_sequential_4x"] = rb["sequential"]
            res["bias_joint_4x"] = rb["joint"]
            res["bias_rounds_4x"] = rb["joint_rounds"]

    res["seconds"] = time.time() - t_start
    return res


def one_budget(f, xs, raw_fn, basis, units0, flow0, origin, span, alpha, ranks,
               flow_sse=0.0, only_joint=False):
    n = len(f)
    out = {"alpha": alpha, "lambda": int(alpha * n)}

    # --- B csv_only --------------------------------------------------------
    if not only_joint:
        sm = smooth_cdf_fast(xs, alpha)
        v = len(sm.virtual_features)
        out["csv_only"] = score(f, raw_fn, sm.slot, v) if v else score(f, raw_fn)
        out["csv_virtual"] = v
        out["csv_rounds"] = sm.rounds

    # --- D sequential and E joint -----------------------------------------
    units = list(units0)
    flow = flow0
    cur_slots = None
    history = []
    best = None           # (loss, probes, round)
    seq_probes = None
    prev_loss = None
    for r in range(1, MAX_ROUNDS + 1):
        z = [flow(k) for k in f]
        if not all(z[i + 1] > z[i] for i in range(n - 1)):
            break
        sm = smooth_cdf_fast(z, alpha)
        v = len(sm.virtual_features)
        fresh = list(sm.slot)
        loss_fresh = ols_sse(z, [float(t) for t in fresh])
        kept = False
        if cur_slots is not None:
            loss_keep = ols_sse(z, [float(t) for t in cur_slots])
            if loss_keep <= loss_fresh:
                # MONOTONE GUARD: the greedy cannot warm start, so its fresh run
                # in the new z-space may be worse than the slot ranks we already
                # hold.  Slot ranks are feature-independent, so keeping them is a
                # legal configuration and makes the V block non-increasing.
                fresh = list(cur_slots)
                v = fresh[-1] + 1 - n
                loss_fresh = loss_keep
                kept = True
        cur_slots = fresh
        slots = fresh if v else None
        s = [float(t) for t in fresh]
        loss_after_V = loss_fresh
        probes = score(f, flow, slots, v) if v else score(f, flow)
        if r == 1:
            seq_probes = probes
            out["seq_virtual"] = v
            out["seq_greedy_rounds"] = sm.rounds
        history.append({"round": r, "loss": loss_after_V, "probes": probes,
                        "virtual": v, "greedy_rounds": sm.rounds,
                        "kept_previous_slots": kept,
                        "units": [list(u) for u in units]})
        if best is None or loss_after_V < best[0] - 1e-15:
            best = (loss_after_V, probes, r, v)
        # warp block: refit f against the current slot targets
        fit = basis.fit(s)
        if fit is None:
            break
        loss_after_f, new_units = fit
        history[-1]["loss_after_warp_refit"] = loss_after_f
        if prev_loss is not None and not (loss_after_f < prev_loss * (1.0 - REL_TOL)):
            prev_loss = loss_after_f
            units = new_units
            flow = make_flow_fn(units, origin, span)
            break
        prev_loss = loss_after_f
        units = new_units
        flow = make_flow_fn(units, origin, span)

    if not history:                      # warp collapsed: fall back to no warp
        out["sequential"] = out.get("csv_only", score(f, raw_fn))
        out["joint"] = out["sequential"]
        out["joint_rounds"] = 0
        out["joint_history"] = []
        out["joint_best_probe"] = out["sequential"]
        return out

    out["sequential"] = seq_probes
    out["joint"] = best[1]
    out["joint_selected_round"] = best[2]
    out["joint_virtual"] = best[3]
    out["joint_rounds"] = len(history)
    out["joint_best_probe"] = min(h["probes"] for h in history)
    out["joint_history"] = history
    # P3, the no-op falsifier: how much of the total loss reduction is bought by
    # rounds >= 2?  round 1 is exactly method D (sequential).
    L0 = flow_sse                          # L(f_0, V = empty): the warp alone
    L1 = history[0]["loss"]                # L(f_0, V_1): sequential
    out["joint_loss_flow_only"] = L0
    out["joint_loss_round1"] = L1
    out["joint_loss_best"] = best[0]
    out["joint_losses"] = [h["loss"] for h in history]
    g1 = L0 - L1
    g2 = L1 - best[0]
    out["round1_loss_gain"] = g1
    out["rounds2plus_loss_gain"] = g2
    out["rounds2plus_over_round1"] = (g2 / g1) if g1 > 0 else (0.0 if g2 <= 0 else float("inf"))
    return out


def main():
    only = sys.argv[1:] or DATASETS
    try:
        from multiprocessing import Pool
        with Pool(min(len(only), os.cpu_count() or 4)) as p:
            results = p.map(run_dataset, only)
    except Exception as exc:                    # pragma: no cover
        print("pool failed (%s), running serially" % exc, file=sys.stderr)
        results = [run_dataset(d) for d in only]
    results.sort(key=lambda r: DATASETS.index(r["dataset"]) if r["dataset"] in DATASETS else 99)
    with open(os.path.join(HERE, "joint_results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(render(results))


def render(results):
    lines = []
    hdr = ("%-8s %6s | %7s %7s | %8s %8s %8s %8s | %8s %8s %8s %8s | %8s %8s %8s %8s | %6s"
           % ("dataset", "binry", "linear", "flowonly", "csv.5", "seq.5", "joint.5", "d(j-s)",
              "csv1", "seq1", "joint1", "d(j-s)", "csv4", "seq4", "joint4", "d(j-s)", "match"))
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in results:
        row = ["%-8s %6.2f | %7.2f %8.2f" % (r["dataset"], r["binary"],
                                             r["baseline_linear"], r["flow_only"])]
        for a in ("0.5", "1", "4"):
            b = r["budgets"][a]
            row.append("| %8.2f %8.2f %8.2f %+8.2f" % (b["csv_only"], b["sequential"],
                                                       b["joint"], b["joint"] - b["sequential"]))
        m = r["joint_budget_to_match_csv4x"]
        row.append("| %6s" % ("%gx" % m if m is not None else "none"))
        lines.append(" ".join(row))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
