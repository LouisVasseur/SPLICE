#!/usr/bin/env python3
"""Warm-up and steady-state detection for GRE chunk traces (trace_<index> runs with GRE_TRACE=geo:F:C or fixed:C).

    python3 warmup_detect.py analyze OUTDIR [OUTDIR ...] [--eps 0.01] [--json FILE] [--plot DIR]
    python3 warmup_detect.py synth [--reps 40] [--seed 1] [--json FILE] [--plot DIR]

A trace has one point per chunk boundary (integrations/gre_splice/trace_interface.h): chunk k covers calls
[c_k, c_k+1) and takes t_k+1 - t_k ns, so y_k = (t_k+1 - t_k) / (c_k+1 - c_k) is its mean latency in ns per lookup.
The chunk after the last point ends after GRE's post-run work and is not used. Calls count from the first get(), so a
run with --warmup W traces its untimed warm-up too. Chunks double from F calls up to C and then stay at C: the first
lookups are resolved on a log scale, where cache and TLB warm-up live, and the rest at a fixed resolution.

Six detectors each return the call at which the steady phase starts:
  mser   MSER (White 1997), weighted by calls per chunk: the cut minimizing the squared standard error of the
         truncated mean, searched over the first half of the calls.
  pelt   PELT changepoints in the mean (Killick et al. 2012): Gaussian cost with a robust noise scale (isolated
         spikes removed first), MBIC penalty, segments of >= 3 chunks. The steady phase is the longest final run of
         segments equivalent to the last one, as in Barrett et al. (2017); equivalent = within max(eps, 3 SE).
  relax  stretched-exponential relaxation y(t) = mu + A exp(-(t/tau)^beta), weighted least squares over a log grid
         of tau and four betas with (mu, A) in closed form, then two robust (Huber) refits. Warm-up ends when the
         fitted transient is below eps * mu. This is the derivative criterion computed on the fit instead of on noisy
         data: |d ln y / d ln t| of the fit is plotted. A * tau * Gamma(1 + 1/beta) is the extra time the cold start
         costs in total.
  cov    windowed coefficient of variation (Georges et al. 2007): the first window of 10 equal chunks with
         CoV < 2%. The geometric prefix is merged into the first equal chunk.
  hw     Heidelberger-Welch (1983): drop 0, 10, ..., 50% of the equal chunks until a Cramer-von Mises test on the
         rest (spectral density at 0 from batch means) accepts stationarity at 5%.
  epoch  doubling epochs [2^j F, 2^(j+1) F) calls: warm once every later epoch's mean is within max(eps, 2 SE) of
         the reference (the chunks in the second half of the run).
The consensus cut is the median, in log space, of the cuts of mser, pelt, relax and epoch (cov and hw are baselines:
the synthetic validation shows them missing early transients). Steady state: the calls-weighted mean after the
consensus cut, with a 95% interval from 20 batch means; across runs, the mean of run means with a t interval and the
split of its variance into within-run and between-run parts (which says whether more runs or longer runs help).
The trace wrapper adds a counter to every get(), so these runs are never headline numbers.

synth generates series with the real chunk schedule and known warm-up ends (flat, exponential, stretched, step,
late slowdown) under two noise levels and scores every detector: |log10 error| of the cut and the bias that the
cut leaves in the steady mean. Needs numpy; --plot needs matplotlib.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

EPS = 0.01            # transient tolerance, relative to the steady mean
MIN_SEG = 3           # PELT: chunks per segment
COV_K, COV_THETA = 10, 0.02
HW_CRIT = 0.46136     # Cramer-von Mises, 5%
N_BATCH = 20
METHODS = ('mser', 'pelt', 'relax', 'cov', 'hw', 'epoch')
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042}

RE_POINT = re.compile(r'^trace_point: (\d+) (\d+) (-?\d+) (-?\d+)(?: (-?\d+))?\s*$')
RE_END = re.compile(r'^trace_end: (\d+) (-?\d+) (-?\d+)\s*$')
RE_SCHED = re.compile(r'^trace_schedule: (\S+)\s*$')
RE_WARM = re.compile(r'^gre_lite_patch: warmup_num=(\d+) pin_core=(-?\d+)\s*$')
RE_LOG = re.compile(r'^(?P<ds>.+?)__(?P<idx>.+?)__r(?P<r>\d+)\.log$')


def t975(df):
    if df < 1:
        return float('nan')
    keys = sorted(k for k in T975 if k <= df)
    return T975[keys[-1]] if df <= 30 else 1.96


def mad_sd(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return float('nan')
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


# ---------------------------------------------------------------- parsing

def parse_log(path):
    pts, end, sched, warm = [], None, None, 0
    with open(path, errors='replace') as f:
        for line in f:
            m = RE_POINT.match(line)
            if m:
                pts.append((int(m.group(2)), int(m.group(3)), int(m.group(4)),
                            int(m.group(5)) if m.group(5) is not None else -1))
                continue
            m = RE_END.match(line)
            if m:
                end = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                continue
            m = RE_SCHED.match(line)
            if m:
                sched = m.group(1)
                continue
            m = RE_WARM.match(line)
            if m:
                warm = int(m.group(1))
    if len(pts) < 8:
        return None
    m = RE_LOG.match(os.path.basename(path))
    c = np.array([p[0] for p in pts], float)
    t = np.array([p[1] for p in pts], float)
    f = np.array([p[2] for p in pts], float)
    u = np.array([p[3] for p in pts], int)
    run = {'path': path, 'schedule': sched, 'warmup': warm,
           'dataset': m.group('ds') if m else '?', 'index': m.group('idx') if m else '?',
           'round': int(m.group('r')) if m else 0,
           'start': c[:-1], 'n': np.diff(c), 'y': np.diff(t) / np.diff(c),
           'faults': np.diff(f) if (f >= 0).all() else None, 'cpu': u[:-1] if (u >= 0).all() else None,
           'calls': end[0] if end else int(c[-1])}
    return run


def find_runs(outdirs):
    runs = []
    for d in outdirs:
        logdir = os.path.join(d, 'logs') if os.path.isdir(os.path.join(d, 'logs')) else d
        for name in sorted(os.listdir(logdir)):
            if name.endswith('.log'):
                r = parse_log(os.path.join(logdir, name))
                if r:
                    runs.append(r)
    return runs


# ---------------------------------------------------------------- noise model and helpers

def equal_mask(n):
    return n == n.max()


def noise_model(n, y):
    """Two-component chunk variance: Var(y_k) = s_op2 / n_k + d2. s_op2 is the per-lookup part (it averages out with
    chunk size; first differences of the equal chunks), d2 the machine drift (frequency, other processes; it does
    not average out within a chunk; batch means of the second half of the run minus their per-lookup part, MAD
    across batches). A transient is only resolvable while it exceeds the drift, so the tolerance is
    eps_eff = max(eps, 2 sqrt(d2) / mean)."""
    eq = equal_mask(n)
    ye, cap = y[eq], float(n.max())
    if ye.size < 8:
        ye, cap = y, float(np.median(n))
    floor = (1e-3 * float(np.median(ye))) ** 2
    s_op2 = cap * max((mad_sd(np.diff(ye)) / math.sqrt(2.0)) ** 2, floor)
    half = ye[ye.size // 2:]
    b = max(int(math.sqrt(half.size)), 1)
    nb = half.size // b
    d2 = 0.0
    if nb >= 6:
        bm = half[:nb * b].reshape(nb, b).mean(axis=1)
        d2 = max(mad_sd(bm) ** 2 - s_op2 / (b * cap), 0.0)
    return s_op2, d2, float(np.median(half))


def chunk_var(n, nm):
    return nm[0] / n + nm[1]


def eps_eff(eps, nm):
    return max(eps, 2.0 * math.sqrt(nm[1]) / nm[2])


def despike(n, y, nm):
    """Isolated chunks more than 6 sigma from both neighbours' median (an interrupt or preemption inside one chunk) are
    replaced by that median for segmentation; the plots keep the raw values."""
    z = y.copy()
    spikes = []
    for i in range(y.size):
        lo, hi = max(0, i - 2), min(y.size, i + 3)
        nb = np.concatenate([y[lo:i], y[i + 1:hi]])
        if nb.size < 2:
            continue
        med = float(np.median(nb))
        if abs(y[i] - med) > 6.0 * math.sqrt(chunk_var(n[i], nm)) and abs(y[i] - med) > 0.05 * med:
            z[i] = med
            spikes.append(i)
    return z, spikes


def wmean(n, y):
    return float(np.sum(n * y) / np.sum(n))


def batch_ci(n, y, nb=N_BATCH):
    """Mean of the calls-weighted series with a 95% interval from nb contiguous batches of about equal calls."""
    if y.size < 4:
        return wmean(n, y), float('nan'), 0.0
    cum = np.cumsum(n)
    edges = np.searchsorted(cum, np.linspace(0, cum[-1], min(nb, y.size) + 1)[1:-1])
    parts = np.split(np.arange(y.size), edges)
    bm = np.array([wmean(n[p], y[p]) for p in parts if p.size])
    b = bm.size
    se = float(np.std(bm, ddof=1) / math.sqrt(b)) if b > 1 else float('nan')
    return wmean(n, y), t975(b - 1) * se, se


# ---------------------------------------------------------------- detectors

def det_mser(start, n, y, **kw):
    N = y.size
    w = n
    Sw = np.cumsum(w[::-1])[::-1]
    Swy = np.cumsum((w * y)[::-1])[::-1]
    Swyy = np.cumsum((w * y * y)[::-1])[::-1]
    cnt = np.arange(N, 0, -1).astype(float)
    m = Swy / Sw
    ss = np.maximum(Swyy - Sw * m * m, 0.0)
    stat = ss / (np.maximum(cnt - 1, 1) * Sw)
    total = start[-1] + n[-1]
    ok = (start <= 0.5 * total) & (cnt >= 10)
    if not ok.any():
        return None, {}
    d = int(np.argmin(np.where(ok, stat, np.inf)))
    return float(start[d]), {'chunk': d}


def _seg_cost(S0, S1, S2, a, b):
    return (S2[b] - S2[a]) - (S1[b] - S1[a]) ** 2 / (S0[b] - S0[a])


def pelt(n, y, nm, minseg=MIN_SEG):
    N = y.size
    w = 1.0 / chunk_var(n, nm)
    S0 = np.concatenate([[0.0], np.cumsum(w)])
    S1 = np.concatenate([[0.0], np.cumsum(w * y)])
    S2 = np.concatenate([[0.0], np.cumsum(w * y * y)])
    beta = 3.0 * math.log(max(N, 2))       # MBIC-like: conservative on noisy machines
    F = np.full(N + 1, np.inf)
    F[0] = -beta
    last = np.zeros(N + 1, int)
    R = np.array([0])
    for t in range(minseg, N + 1):
        cand = R[t - R >= minseg]
        if cand.size:
            vals = F[cand] + _seg_cost(S0, S1, S2, cand, t) + beta
            i = int(np.argmin(vals))
            F[t], last[t] = vals[i], cand[i]
            keep = (t - R < minseg) | (F[R] + np.where(t - R >= minseg,
                                                       _seg_cost(S0, S1, S2, R, np.full(R.size, t)), 0.0) <= F[t])
            R = R[keep]
        if np.isfinite(F[t]):
            R = np.append(R, t)
    cps, t = [], N
    while t > 0:
        cps.append(t)
        t = last[t]
    bounds = [0] + sorted(cps)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def det_pelt(start, n, y, nm=None, eps=EPS, **kw):
    z, spikes = despike(n, y, nm)
    segs = pelt(n, z, nm)
    e = eps_eff(eps, nm)
    info = []
    for a, b in segs:
        nn, zz = n[a:b], z[a:b]
        m = wmean(nn, zz)
        se_model = math.sqrt(nm[0] / nn.sum() + nm[1])      # drift does not average out within a segment
        se_emp = math.sqrt(max(np.sum(nn * (zz - m) ** 2), 0.0) / (max(b - a - 1, 1) * nn.sum()))
        info.append({'a': a, 'b': b, 'start': float(start[a]), 'end': float(start[b - 1] + n[b - 1]), 'mean': m,
                     'se': max(se_model, se_emp)})
    ref = info[-1]
    j0 = len(info) - 1
    for j in range(len(info) - 2, -1, -1):
        tol = max(e * ref['mean'], 3.0 * math.hypot(info[j]['se'], ref['se']))
        if abs(info[j]['mean'] - ref['mean']) <= tol:
            j0 = j
        else:
            break
    pre = info[:j0]
    total = float(start[-1] + n[-1])
    steady_calls = total - info[j0]['start']
    if j0 == 0:
        kind = 'flat'
    elif all(p['mean'] > ref['mean'] for p in pre):
        kind = 'warmup'
    elif all(p['mean'] < ref['mean'] for p in pre):
        kind = 'slowdown'
    else:
        kind = 'mixed'
    if steady_calls < 0.25 * total:
        kind = 'no steady state'
    return info[j0]['start'], {'segments': info, 'kind': kind, 'spikes': spikes}


def _relax_basis(start, n, tau, beta):
    nodes = np.array([0.125, 0.375, 0.625, 0.875])
    X = start[:, None] + n[:, None] * nodes[None, :]
    return np.exp(-(X[:, :, None] / tau[None, None, :]) ** beta).mean(axis=1)


def relax_fit(start, n, y, nm, w_extra=None):
    taus = np.logspace(1, 9.5, 171)
    w = 1.0 / chunk_var(n, nm) if w_extra is None else w_extra / chunk_var(n, nm)
    Sw, Sy, Syy = w.sum(), (w * y).sum(), (w * y * y).sum()
    best = None
    for beta in (0.35, 0.5, 0.7, 1.0):
        B = _relax_basis(start, n, taus, beta)
        Sb = (w[:, None] * B).sum(0)
        Sbb = (w[:, None] * B * B).sum(0)
        Sby = (w[:, None] * B * y[:, None]).sum(0)
        det = Sw * Sbb - Sb * Sb
        good = det > 1e-12 * Sw * np.maximum(Sbb, 1e-300)
        A = np.where(good, (Sw * Sby - Sb * Sy) / np.where(good, det, 1), 0.0)
        mu = (Sy - A * Sb) / Sw
        sse = Syy - 2 * mu * Sy - 2 * A * Sby + mu * mu * Sw + 2 * mu * A * Sb + A * A * Sbb
        sse = np.where(good, sse, np.inf)
        j = int(np.argmin(sse))
        if best is None or sse[j] < best['sse']:
            best = {'mu': float(mu[j]), 'A': float(A[j]), 'tau': float(taus[j]), 'beta': beta, 'sse': float(sse[j])}
    best['sse0'] = float(Syy - Sy * Sy / Sw)
    return best


def relax_curve(t, fit):
    return fit['mu'] + fit['A'] * np.exp(-(np.asarray(t, float) / fit['tau']) ** fit['beta'])


def relax_dlog(t, fit):
    """d ln y / d ln t of the fitted curve."""
    t = np.asarray(t, float)
    u = (t / fit['tau']) ** fit['beta']
    tr = fit['A'] * np.exp(-u)
    return -tr * fit['beta'] * u / (fit['mu'] + tr)


def det_relax(start, n, y, nm=None, eps=EPS, **kw):
    fit = relax_fit(start, n, y, nm)
    for _ in range(2):          # Huber reweighting against stalls
        r = (y - relax_curve(start + n / 2, fit)) / np.sqrt(chunk_var(n, nm))
        sc = max(mad_sd(r), 1e-9)
        h = np.minimum(1.0, 2.5 * sc / np.maximum(np.abs(r), 1e-12))
        fit = relax_fit(start, n, y, nm, h)
    N = y.size
    e = eps_eff(eps, nm)
    gain = fit['sse0'] - fit['sse']
    fit['eps_eff'] = e
    fit['significant'] = bool(gain > 9.0 * math.log(max(N, 2)) and abs(fit['A']) > 0.25 * e * abs(fit['mu']))
    total = float(start[-1] + n[-1])
    if not fit['significant'] or abs(fit['A']) <= e * abs(fit['mu']):
        cut = 0.0
    else:
        cut = fit['tau'] * math.log(abs(fit['A']) / (e * abs(fit['mu']))) ** (1.0 / fit['beta'])
    fit['excess_ns'] = fit['A'] * fit['tau'] * math.gamma(1.0 + 1.0 / fit['beta']) if fit['significant'] else 0.0
    fit['at_grid_edge'] = fit['tau'] >= 10 ** 9.4
    if cut > total:
        return None, fit
    return float(cut), fit


def merged_equal(start, n, y):
    """The equal chunks, with every smaller chunk before them merged into the first one (call 0)."""
    eq = np.where(equal_mask(n))[0]
    if eq.size == 0:
        return start, n, y
    f = eq[0]
    if f == 0:
        return start[eq], n[eq], y[eq]
    pre_n = n[:f].sum()
    pre_y = wmean(n[:f], y[:f])
    return (np.concatenate([[start[0]], start[eq]]), np.concatenate([[pre_n], n[eq]]),
            np.concatenate([[pre_y], y[eq]]))


def det_cov(start, n, y, k=COV_K, theta=COV_THETA, **kw):
    s, nn, yy = merged_equal(start, n, y)
    for i in range(0, yy.size - k + 1):
        win = yy[i:i + k]
        if np.std(win, ddof=1) / np.mean(win) < theta:
            return float(s[i]), {'theta': theta, 'k': k}
    return None, {'theta': theta, 'k': k}


def _cvm(x):
    m = x.size
    b = max(int(math.sqrt(m)), 2)
    nb = m // b
    if nb < 4:
        return float('inf')
    bm = x[:nb * b].reshape(nb, b).mean(axis=1)
    s0 = b * np.var(bm, ddof=1)
    if s0 <= 0:
        return 0.0
    bridge = (np.cumsum(x - x.mean())) / math.sqrt(m * s0)
    return float(np.mean(bridge ** 2))


def det_hw(start, n, y, **kw):
    s, nn, yy = merged_equal(start, n, y)
    tested = []
    for frac in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        i = int(frac * yy.size)
        stat = _cvm(yy[i:])
        tested.append((frac, stat))
        if stat < HW_CRIT:
            return float(s[i]), {'tested': tested}
    return None, {'tested': tested}


def det_epoch(start, n, y, nm=None, eps=EPS, **kw):
    total = float(start[-1] + n[-1])
    F0 = float(n[0])
    mid = start + n / 2
    ep = np.floor(np.log2(np.maximum(mid / F0, 1.0))).astype(int)
    ref_mask = start >= total / 2
    ref = wmean(n[ref_mask], y[ref_mask])
    ref_se = math.sqrt(nm[0] / n[ref_mask].sum() + nm[1])
    e = eps_eff(eps, nm)
    epochs = []
    for k in range(ep.min(), ep.max() + 1):
        msk = ep == k
        if not msk.any():
            continue
        nn, yy = n[msk], y[msk]
        m = wmean(nn, yy)
        se = math.sqrt(nm[0] / nn.sum() + nm[1])
        ok = abs(m - ref) <= max(e * ref, 2.0 * math.hypot(se, ref_se))
        epochs.append({'epoch': k, 'start': float(start[msk][0]), 'mean': m, 'se': se, 'ok': ok})
    j0 = len(epochs)
    for j in range(len(epochs) - 1, -1, -1):
        if epochs[j]['ok']:
            j0 = j
        else:
            break
    if j0 >= len(epochs):
        return None, {'epochs': epochs, 'ref': ref}
    return epochs[j0]['start'] if j0 > 0 else 0.0, {'epochs': epochs, 'ref': ref}


DETECTORS = {'mser': det_mser, 'pelt': det_pelt, 'relax': det_relax, 'cov': det_cov, 'hw': det_hw,
             'epoch': det_epoch}


CORE = ('mser', 'pelt', 'relax', 'epoch')   # the consensus; cov and hw are reported as baselines


def analyze_series(start, n, y, eps=EPS):
    nm = noise_model(n, y)
    out = {'noise': {'s_op2': nm[0], 'drift_sd': math.sqrt(nm[1]), 'eps_eff': eps_eff(eps, nm)}, 'cuts': {},
           'info': {}}
    for name in METHODS:
        cut, info = DETECTORS[name](start, n, y, nm=nm, eps=eps)
        out['cuts'][name] = cut
        out['info'][name] = info
    got = sorted(out['cuts'][m] for m in CORE if out['cuts'][m] is not None)
    # Cuts span decades, so the median is taken in log space (offset 1e3 calls so that 0 is a value).
    cons = float(10 ** np.median(np.log10(np.array(got) + 1e3)) - 1e3) if got else None
    if cons is not None and cons < 1.0:
        cons = 0.0
    out['consensus'] = cons
    out['range'] = (got[0], got[-1]) if got else None
    if cons is not None:
        msk = start >= cons
        if msk.sum() < 4:
            msk = np.ones_like(msk)
        m, half, se = batch_ci(n[msk], y[msk])
        out['steady'] = {'mean': m, 'ci95': half, 'se': se, 'calls': float(n[msk].sum())}
        cold = start < cons
        out['warmup_excess_ns'] = float(np.sum(n[cold] * (y[cold] - m))) if cold.any() else 0.0
        out['jitter'] = jitter_stats(start, n, y, cons)
    return out


def mixture2(x, iters=300):
    """1-D two-Gaussian mixture by EM, against one Gaussian by BIC. Returns None unless two states win."""
    x = np.asarray(x, float)
    if x.size < 30:
        return None
    lo, hi = np.percentile(x, [25, 75])
    m = np.array([lo, hi])
    sd = np.array([max(mad_sd(x), 1e-9)] * 2)
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        pdf = w / (sd * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x[:, None] - m) / sd) ** 2) + 1e-300
        r = pdf / pdf.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0)
        w = nk / x.size
        m = (r * x[:, None]).sum(axis=0) / nk
        sd = np.sqrt(np.maximum((r * (x[:, None] - m) ** 2).sum(axis=0) / nk, (1e-4 * np.median(x)) ** 2))
    pdf = w / (sd * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x[:, None] - m) / sd) ** 2) + 1e-300
    ll2 = float(np.log(pdf.sum(axis=1)).sum())
    s1 = float(np.std(x))
    ll1 = float(np.sum(-0.5 * ((x - x.mean()) / s1) ** 2 - math.log(s1 * math.sqrt(2 * math.pi))))
    bic1, bic2 = -2 * ll1 + 2 * math.log(x.size), -2 * ll2 + 5 * math.log(x.size)
    o = np.argsort(m)
    m, sd, w = m[o], sd[o], w[o]
    sep = abs(m[1] - m[0]) / math.sqrt((sd[0] ** 2 + sd[1] ** 2) / 2)      # Ashman's D
    if bic2 < bic1 - 10 and sep > 2 and w.min() > 0.03:
        return {'fast_mean': float(m[0]), 'fast_sd': float(sd[0]), 'slow_mean': float(m[1]), 'slow_sd': float(sd[1]),
                'slow_share': float(w[1]), 'ashman_d': float(sep)}
    return None


def jitter_stats(start, n, y, cut):
    """Jitter of the steady phase (the equal chunks after the cut): robust CoV, lag-1 autocorrelation (persistence
    of slow phases), and a two-state mixture when the chunk means are bimodal (on a shared or heterogeneous machine:
    fast and slow states, e.g. contention or core migration)."""
    msk = equal_mask(n) & (start >= (cut or 0))
    x = y[msk]
    if x.size < 10:
        return {}
    xc = x - x.mean()
    ac1 = float(np.sum(xc[1:] * xc[:-1]) / max(np.sum(xc * xc), 1e-300))
    return {'chunks': int(x.size), 'median': float(np.median(x)), 'robust_cov': mad_sd(x) / float(np.median(x)),
            'cov': float(np.std(x, ddof=1) / np.mean(x)), 'lag1': ac1, 'p01': float(np.percentile(x, 1)),
            'p99': float(np.percentile(x, 99)), 'states': mixture2(x)}


def lower_envelope(n, y, q=10.0, half=5):
    """Rolling q-th percentile of the equal chunks (window 2 * half + 1); the smaller chunks of the geometric prefix
    are single samples and stay as they are. Interference (other processes, core migration, interrupts) only adds
    time, so the lower envelope approximates the interference-free latency."""
    z = y.copy()
    eq = np.where(equal_mask(n))[0]
    ye = y[eq]
    z[eq] = [np.percentile(ye[max(0, i - half):i + half + 1], q) for i in range(ye.size)]
    return z


def core_map(spec):
    """CPU number -> core class. 'auto' on macOS: the efficiency cores are numbered first (checked on an M3 Max with
    pthread_cpu_number_np under taskpolicy -c background), then the performance clusters of hw.perflevel0.cpusperl2
    cores each. Elsewhere, or with 'none', every CPU is its own class ('cpu<N>'). Explicit: '0-3:E,4-9:P0,10-15:P1'."""
    if spec == 'auto' and sys.platform == 'darwin':
        import subprocess

        def sc(k):
            try:
                return int(subprocess.check_output(['sysctl', '-n', k]).decode().strip())
            except Exception:
                return 0
        ne, npf, per = sc('hw.perflevel1.logicalcpu'), sc('hw.perflevel0.logicalcpu'), sc('hw.perflevel0.cpusperl2')
        if ne and npf and per:
            m = {i: 'E' for i in range(ne)}
            for i in range(npf):
                m[ne + i] = 'P%d' % (i // per)
            return m
        return None
    if spec in ('auto', 'none', None, ''):
        return None
    m = {}
    for part in spec.split(','):
        ids, lab = part.split(':')
        a, _, b = ids.partition('-')
        for i in range(int(a), int(b or a) + 1):
            m[i] = lab
    return m


def core_stats(start, n, y, cpu, cut, cmap, states=None):
    """Where the steady chunks ran: share and mean latency per core class, migrations between consecutive chunks,
    and (when the steady phase has two states) the share of slow chunks per class."""
    if cpu is None:
        return None
    lab = np.array([(cmap or {}).get(int(c), 'cpu%d' % c) for c in cpu])
    msk = start >= (cut or 0)
    out = {'classes': {}, 'migrations': float(np.mean(cpu[msk][1:] != cpu[msk][:-1])) if msk.sum() > 1 else 0.0,
           'class_changes': float(np.mean(lab[msk][1:] != lab[msk][:-1])) if msk.sum() > 1 else 0.0}
    thr = 0.5 * (states['fast_mean'] + states['slow_mean']) if states else None
    for c in sorted(set(lab[msk])):
        k = msk & (lab == c)
        d = {'share': float(n[k].sum() / n[msk].sum()), 'mean': wmean(n[k], y[k]), 'chunks': int(k.sum())}
        if thr is not None:
            d['slow_share'] = float(np.mean(y[k] > thr))
        out['classes'][c] = d
    return out


# ---------------------------------------------------------------- across runs

def across_runs(results):
    ms = np.array([r['steady']['mean'] for r in results if r.get('steady')])
    ses = np.array([r['steady']['se'] for r in results if r.get('steady')])
    R = ms.size
    if R == 0:
        return {}
    out = {'runs': R, 'mean': float(ms.mean())}
    if R > 1:
        sd = float(np.std(ms, ddof=1))
        out['ci95'] = t975(R - 1) * sd / math.sqrt(R)
        within = float(np.mean(ses ** 2))
        out['var_within'] = within
        out['var_between'] = max(float(np.var(ms, ddof=1)) - within, 0.0)
        out['between_share'] = out['var_between'] / max(out['var_between'] + within, 1e-300)
    return out


def rhat(chains):
    chains = [np.asarray(c, float) for c in chains if len(c) >= 4]
    if len(chains) < 2:
        return float('nan')
    L = min(c.size for c in chains)
    halves = []
    for c in chains:
        c = c[-L:]
        halves += [c[:L // 2], c[L // 2:2 * (L // 2)]]
    X = np.array(halves)
    m, n = X.shape
    B = n * np.var(X.mean(axis=1), ddof=1)
    W = np.mean(np.var(X, axis=1, ddof=1))
    return float(math.sqrt(((n - 1) / n * W + B / n) / W))


# ---------------------------------------------------------------- synthetic validation

def schedule(F=64, C=65536, total=120000000):
    c, step, pts = 0, F, [0]
    while c < total:
        c += min(step, total - c)
        pts.append(c)
        step = min(step * 2, C)
    p = np.array(pts, float)
    return p[:-1], np.diff(p)


SCENARIOS = [
    # name, transient(t) relative to mu, true warm-up end (calls) for a tolerance e (relative)
    ('flat', lambda t: 0.0 * t, lambda e: 0.0),
    ('exp fast', lambda t: 1.0 * np.exp(-t / 2e3), lambda e: 2e3 * math.log(1.0 / e) if e < 1.0 else 0.0),
    ('exp mid', lambda t: 0.5 * np.exp(-t / 3e4), lambda e: 3e4 * math.log(0.5 / e) if e < 0.5 else 0.0),
    ('exp slow', lambda t: 0.3 * np.exp(-t / 1e6), lambda e: 1e6 * math.log(0.3 / e) if e < 0.3 else 0.0),
    ('stretched', lambda t: 1.0 * np.exp(-np.sqrt(t / 1e4)), lambda e: 1e4 * math.log(1.0 / e) ** 2 if e < 1 else 0.0),
    ('step', lambda t: np.where(t < 5e6, 0.25, 0.0), lambda e: 5e6 if e < 0.25 else 0.0),
    ('late slowdown', lambda t: np.where(t >= 6e7, 0.08, 0.0), lambda e: 6e7 if e < 0.08 else 0.0),
]
NOISE = {   # op: per-lookup sd / mu; drift: AR(1) over wall time (one step per 1M calls), sd / mu and rho per step;
            # stalls: Poisson per second of run time, each Exp(mean) ns (interrupts, preemption)
    'quiet': {'op': 1.0, 'drift': 0.004, 'rho': 0.5, 'stall_rate': 2.0, 'stall_ns': 3e4},
    'noisy': {'op': 1.5, 'drift': 0.03, 'rho': 0.9, 'stall_rate': 50.0, 'stall_ns': 1e5}}


def synth_series(rng, fn, noise, mu=150.0):
    start, n = schedule()
    nodes = np.array([0.125, 0.375, 0.625, 0.875])
    X = start[:, None] + n[:, None] * nodes[None, :]
    truth = mu * (1.0 + fn(X).mean(axis=1))
    per_op = rng.normal(0, noise['op'] * mu, size=n.size) / np.sqrt(n)
    grid = np.arange(0, start[-1] + n[-1] + 2e6, 1e6)
    d = np.zeros(grid.size)
    innov = rng.normal(0, noise['drift'] * mu * math.sqrt(1 - noise['rho'] ** 2), size=grid.size)
    d[0] = rng.normal(0, noise['drift'] * mu)
    for k in range(1, grid.size):
        d[k] = noise['rho'] * d[k - 1] + innov[k]
    drift = np.interp(start + n / 2, grid, d)
    dur_s = n * truth * 1e-9
    stalls = np.array([rng.exponential(noise['stall_ns'], size=k).sum() for k in rng.poisson(noise['stall_rate'] * dur_s)])
    return start, n, truth + per_op + drift + stalls / n, truth


def run_synth(reps, seed, eps=EPS):
    rng = np.random.default_rng(seed)
    rows, examples = [], {}
    for nname, noise in NOISE.items():
        e_true = max(eps, 2.0 * noise['drift'])
        for sname, fn, truth_at in SCENARIOS:
            true_cut = truth_at(e_true)
            for rep in range(reps):
                start, n, y, truth = synth_series(rng, fn, noise)
                res = analyze_series(start, n, y, eps)
                final = truth[-1]
                for mname in METHODS + ('consensus',):
                    cut = res['consensus'] if mname == 'consensus' else res['cuts'][mname]
                    if cut is None:
                        rows.append({'noise': nname, 'scenario': sname, 'method': mname, 'found': False})
                        continue
                    msk = start >= cut
                    steady = wmean(n[msk], y[msk]) if msk.sum() >= 4 else float('nan')
                    rows.append({'noise': nname, 'scenario': sname, 'method': mname, 'found': True,
                                 'cut': cut, 'true': true_cut,
                                 'log_err': abs(math.log10((cut + 1e3) / (true_cut + 1e3))),
                                 'bias_pct': 100.0 * (steady / final - 1.0)})
                if rep == 0:
                    examples[(nname, sname)] = {'start': start, 'n': n, 'y': y, 'truth': truth, 'res': res,
                                                'true_cut': true_cut}
    return rows, examples


def summarize_synth(rows):
    out = []
    for nname in NOISE:
        for sname, _, _ in SCENARIOS:
            for mname in METHODS + ('consensus',):
                rr = [r for r in rows if r['noise'] == nname and r['scenario'] == sname and r['method'] == mname]
                found = [r for r in rr if r['found']]
                out.append({'noise': nname, 'scenario': sname, 'method': mname,
                            'found_share': len(found) / max(len(rr), 1),
                            'median_log_err': float(np.median([r['log_err'] for r in found])) if found else None,
                            'median_abs_bias_pct': float(np.median([abs(r['bias_pct']) for r in found]))
                            if found else None})
    return out


# ---------------------------------------------------------------- plots

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1'
WARM, STEADY, FIT = '#eb6834', '#2a78d6', '#1c5cab'
MARK = {'mser': 'o', 'pelt': 's', 'relax': 'D', 'cov': '^', 'hw': 'v', 'epoch': 'P'}
LABEL = {'mser': 'MSER', 'pelt': 'PELT', 'relax': 'relaxation fit', 'cov': 'windowed CoV', 'hw': 'Heidelberger-Welch',
         'epoch': 'doubling epochs'}


def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'savefig.facecolor': 'white',
                         'figure.facecolor': 'white', 'legend.frameon': False})
    return plt


PROTOCOL_W = 20000000   # the AIDB-style warm-up of our protocol (SERVER.md 5b)


def _ylimits(ys, res):
    """Linear y unless the transient is over 3x the steady level, then log y."""
    st = res.get('steady', {}).get('mean') or float(np.median(np.concatenate(ys)))
    lo = min(float(np.percentile(y, 0.5)) for y in ys)
    hi = max(float(np.max(y[:16])) for y in ys)
    hi = max(hi, max(float(np.percentile(y, 99.5)) for y in ys))
    if hi > 3.0 * st:
        return 'log', max(lo * 0.8, 1e-3), hi * 1.3
    return 'linear', max(0.0, lo - 0.08 * (hi - lo)), hi + 0.08 * (hi - lo)


def _phase_axes(ax, start, n, ys, res, fit, xmin, xmax, show_legend):
    """Latency against calls (log x): raw chunk series, fitted relaxation, warm-up and steady phases."""
    cons, rng_ = res['consensus'], res['range']
    kind = res['info']['pelt']['kind']
    pre_label = 'pre-shift phase (run got slower)' if kind == 'slowdown' else 'warm-up phase'
    if cons is not None:
        if cons > xmin:
            ax.axvspan(xmin, cons, color=WARM, alpha=0.10, lw=0, label=pre_label)
        if rng_ and rng_[1] > rng_[0] and rng_[1] > xmin:
            ax.axvspan(max(rng_[0], xmin), rng_[1], facecolor='none', edgecolor=WARM, alpha=0.35, hatch='////', lw=0,
                       label='detectors disagree')
        st = res.get('steady')
        if st:
            x0 = max(cons, xmin)
            ax.axvspan(x0, xmax, color=STEADY, alpha=0.06, lw=0, label='steady phase')
            ax.hlines(st['mean'], x0, xmax, color=STEADY, lw=1.2, linestyles='--', label='steady mean')
            if st['ci95'] == st['ci95']:
                ax.fill_between([x0, xmax], st['mean'] - st['ci95'], st['mean'] + st['ci95'], color=STEADY, alpha=0.18,
                                lw=0)
        if cons > xmin:
            ax.axvline(cons, color=WARM, lw=1.4)
    ax.axvline(PROTOCOL_W, color=MUTED, lw=1.0, ls=(0, (4, 3)), label='protocol warm-up (20M)')
    for i, y in enumerate(ys):
        x = np.repeat(np.concatenate([start, [start[-1] + n[-1]]]), 2)[1:-1]
        ax.plot(np.maximum(x, xmin), np.repeat(y, 2), color=INK2, lw=0.7, alpha=0.55 if len(ys) > 1 else 0.8,
                label='chunk mean (each run)' if i == 0 else None)
    if fit is not None:
        tt = np.logspace(math.log10(xmin), math.log10(xmax), 400)
        ax.plot(tt, relax_curve(tt, fit), color=FIT, lw=2.0, label='relaxation fit')
    ax.set_xscale('log')
    ax.set_xlim(xmin, xmax)
    scale, lo, hi = _ylimits(ys, res)
    ax.set_yscale(scale)
    ax.set_ylim(lo, hi)
    if show_legend:
        ax.legend(loc='upper right', fontsize=7.5, ncol=1)


def _marginal(ax, ys, res, ylim, yscale):
    """Histogram of the steady chunks (all runs), sideways, sharing the phase panel's y axis."""
    cons = res['consensus'] or 0.0
    vals = np.concatenate([y[r_start >= cons] for y, r_start in ys])
    if vals.size:
        bins = np.logspace(math.log10(ylim[0]), math.log10(ylim[1]), 60) if yscale == 'log' else np.linspace(ylim[0], ylim[1], 60)
        ax.hist(vals, bins=bins, orientation='horizontal', color=STEADY, alpha=0.55, lw=0)
    stt = res.get('jitter', {}).get('states')
    if stt:
        for k, lab in (('fast_mean', 'fast'), ('slow_mean', 'slow')):
            ax.axhline(stt[k], color=INK2, lw=0.8, ls=':')
            ax.text(ax.get_xlim()[1], stt[k], ' ' + lab, va='center', ha='left', fontsize=7, color=INK2)
    ax.set_yscale(yscale)
    ax.set_ylim(*ylim)
    ax.tick_params(labelleft=False, labelbottom=False, length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('steady chunks', fontsize=7, color=INK2)


def _strip_axes(ax, cuts_list, xmin, xmax):
    """One row per detector, one dot per run at its cut (log x, aligned with the phase panel)."""
    for row, m in enumerate(METHODS):
        xs = [c[m] for c in cuts_list if c.get(m) is not None]
        miss = sum(1 for c in cuts_list if c.get(m) is None)
        if xs:
            ax.scatter(np.maximum(xs, xmin), [row] * len(xs), marker=MARK[m], s=26, color=INK2, zorder=3)
        if miss:
            ax.text(xmax, row, '  none x%d' % miss, va='center', ha='left', fontsize=7, color=MUTED)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([LABEL[m] for m in METHODS], fontsize=8)
    ax.set_ylim(len(METHODS) - 0.5, -0.5)
    ax.set_xscale('log')
    ax.set_xlim(xmin, xmax)
    ax.grid(axis='y', visible=False)


def _summary_line(key, g):
    pooled, acr = g['pooled'], g['across']
    st = pooled.get('steady', {})
    j = pooled.get('jitter', {})
    txt = '%s on %s: warm after %s lookups; steady %.1f ns' % (key[1].replace('trace_', ''), key[0],
                                                             _fmt_calls(pooled['consensus']),
                                                             acr.get('mean', st.get('mean', float('nan'))))
    if 'ci95' in acr:
        txt += ' +/- %.1f (%d runs)' % (acr['ci95'], acr['runs'])
    envg = g.get('envelope')
    if envg and envg.get('steady'):
        txt += '\nlower envelope: warm after %s, steady %.1f ns' % (_fmt_calls(envg['consensus']), envg['steady']['mean'])
    if j:
        txt += '; jitter: robust CoV %.1f%%, lag-1 %.2f' % (100 * j['robust_cov'], j['lag1'])
        if j.get('states'):
            txt += '; two states %.0f / %.0f ns (slow %.0f%%)' % (j['states']['fast_mean'], j['states']['slow_mean'],
                                                                100 * j['states']['slow_share'])
    return txt


def plot_group(plt, key, group, path, eps):
    """One figure per (dataset, index): phases with the steady distribution, derivative, detector strip."""
    runs, results, pooled = group['runs'], group['results'], group['pooled']
    xmin, xmax = 32.0, max(r['start'][-1] + r['n'][-1] for r in runs)
    has_f = all(r['faults'] is not None for r in runs)
    has_c = all(r['cpu'] is not None for r in runs)
    hr = [3.2] + ([0.35 + 0.18 * len(runs)] if has_c else []) + ([1.2] if has_f else []) + [1.4, 1.6]
    fig = plt.figure(figsize=(11.5, 1.25 * sum(hr)))
    gs = fig.add_gridspec(len(hr), 2, height_ratios=hr, width_ratios=[7, 1], hspace=0.14, wspace=0.03)
    row = 1
    ax = fig.add_subplot(gs[0, 0])
    start, n = runs[0]['start'], runs[0]['n']
    ys = [r['y'] for r in runs]
    _phase_axes(ax, start, n, ys, pooled, pooled['info']['relax'], xmin, xmax, True)
    envg = group['envelope']
    xe = np.repeat(np.concatenate([start, [start[-1] + n[-1]]]), 2)[1:-1]
    ax.plot(np.maximum(xe, xmin), np.repeat(group['env_y'], 2), color=INK, lw=1.0, ls=(0, (2, 1.5)),
            label='lower envelope (p10, interference removed)')
    if envg['consensus'] is not None and envg['consensus'] > xmin:
        ax.axvline(envg['consensus'], color=INK, lw=1.0, ls=(0, (2, 1.5)))
    ax.legend(loc='upper right', fontsize=7.5, ncol=1, frameon=True, framealpha=0.9, edgecolor='none')
    ax.set_ylabel('ns per lookup (chunk mean)')
    ax.tick_params(labelbottom=False)
    ax.text(0.0, 1.02, _summary_line(key, group), transform=ax.transAxes, va='bottom', fontsize=8.5, color=INK)
    axm = fig.add_subplot(gs[0, 1])
    _marginal(axm, [(r['y'], r['start']) for r in runs], pooled, ax.get_ylim(), ax.get_yscale())
    if has_c:
        axc = fig.add_subplot(gs[row, 0], sharex=ax)
        cmap = group.get('cmap') or {}
        labs = sorted({cmap.get(int(c), 'cpu%d' % c) for r in runs for c in r['cpu']})
        pal = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
        col = {lab: pal[i % len(pal)] for i, lab in enumerate(labs)}
        for i, r in enumerate(runs):
            lab = [cmap.get(int(c), 'cpu%d' % c) for c in r['cpu']]
            mid = r['start'] + r['n'] / 2
            for L in labs:
                sel = np.array([x == L for x in lab])
                if sel.any():
                    axc.vlines(np.maximum(mid[sel], xmin), i + 0.1, i + 0.9, colors=col[L], lw=0.6,
                               label=L if i == 0 else None)
        axc.set_yticks([i + 0.5 for i in range(len(runs))])
        axc.set_yticklabels(['run %d' % r['round'] for r in runs], fontsize=7)
        axc.set_ylim(0, len(runs))
        axc.grid(False)
        axc.set_ylabel('core', fontsize=8)
        axc.legend(loc='upper left', bbox_to_anchor=(1.0, 1.05), fontsize=7, frameon=False)
        axc.tick_params(labelbottom=False)
        row += 1
    if has_f:
        axf = fig.add_subplot(gs[row, 0], sharex=ax)
        for i, r in enumerate(runs):
            x = np.repeat(np.concatenate([r['start'], [r['start'][-1] + r['n'][-1]]]), 2)[1:-1]
            axf.plot(np.maximum(x, xmin), np.repeat(np.maximum(1e3 * r['faults'] / r['n'], 1e-3), 2), color=WARM, lw=0.8,
                     alpha=0.7, label='page faults per 1,000 lookups (getrusage)' if i == 0 else None)
        axf.set_yscale('log')
        axf.set_ylim(5e-4, None)
        axf.set_ylabel('faults / 1k')
        axf.legend(loc='upper right', fontsize=7.5)
        axf.tick_params(labelbottom=False)
        row += 1
    ax2 = fig.add_subplot(gs[row, 0], sharex=ax)
    fit = pooled['info']['relax']
    tt = np.logspace(math.log10(xmin), math.log10(xmax), 400)
    for i, r in enumerate(runs):
        mid = r['start'] + r['n'] / 2
        d = np.abs(np.diff(np.log(r['y'])) / np.diff(np.log(mid)))
        ax2.scatter(np.sqrt(mid[1:] * mid[:-1]), np.maximum(d, 1e-5), s=3, color=MUTED, alpha=0.35, lw=0,
                    label='raw finite differences' if i == 0 else None)
    ax2.plot(tt, np.maximum(np.abs(relax_dlog(tt, fit)), 1e-5), color=FIT, lw=2.0, label='derivative of the fit')
    e = pooled['noise']['eps_eff']
    ax2.axhline(e, color=WARM, lw=1.0, ls='--', label='tolerance %.3g (max of eps, 2 x drift)' % e)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-5, 30)
    ax2.set_ylabel('|d ln y / d ln t|')
    ax2.legend(loc='upper right', fontsize=7.5, ncol=3)
    ax2.tick_params(labelbottom=False)
    ax3 = fig.add_subplot(gs[row + 1, 0], sharex=ax)
    _strip_axes(ax3, [r['cuts'] for r in results] + [pooled['cuts']], xmin, xmax)
    ax3.axvline(PROTOCOL_W, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    if pooled['consensus'] and pooled['consensus'] > xmin:
        ax3.axvline(pooled['consensus'], color=WARM, lw=1.4)
    ax3.set_xlabel('lookups since the first get() (log scale); dots: each run and the pooled median series')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_overview(plt, groups, path, eps):
    keys = sorted(groups)
    k = len(keys)
    cols = 2 if k > 1 else 1
    rows = int(math.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.1 * rows), squeeze=False)
    for i, key in enumerate(keys):
        g = groups[key]
        ax = axes[i // cols][i % cols]
        runs = g['runs']
        xmin, xmax = 32.0, max(r['start'][-1] + r['n'][-1] for r in runs)
        _phase_axes(ax, runs[0]['start'], runs[0]['n'], [r['y'] for r in runs], g['pooled'],
                    g['pooled']['info']['relax'], xmin, xmax, i == 0)
        st = g['pooled'].get('steady', {})
        ax.set_title('%s: warm after %s, steady %.1f ns' % (key[1].replace('trace_', ''),
                     _fmt_calls(g['pooled']['consensus']), g['across'].get('mean', st.get('mean', float('nan')))),
                     fontsize=9, loc='left', color=INK)
        if i // cols == rows - 1 or (i + cols) >= k:
            ax.set_xlabel('lookups (log scale)')
        if i % cols == 0:
            ax.set_ylabel('ns per lookup')
    for j in range(k, rows * cols):
        axes[j // cols][j % cols].axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_synth(plt, examples, summary, outdir):
    names = [s[0] for s in SCENARIOS]
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 2.0 * len(names)), squeeze=False, sharex=True)
    for col, nname in enumerate(NOISE):
        for row, sname in enumerate(names):
            ex = examples[(nname, sname)]
            ax = axes[row][col]
            start, n = ex['start'], ex['n']
            mid = start + n / 2
            ax.plot(mid, ex['y'], color=INK2, lw=0.6, alpha=0.7)
            ax.plot(mid, ex['truth'], color=INK, lw=1.0)
            if ex['true_cut'] > 0:
                ax.axvline(ex['true_cut'], color=INK, lw=1.2, ls=':')
            res = ex['res']
            if res['consensus'] is not None:
                ax.axvspan(32, max(res['consensus'], 33), color=WARM, alpha=0.10, lw=0)
                ax.axvline(max(res['consensus'], 33), color=WARM, lw=1.2)
            ymin, ymax = ax.get_ylim()
            for i, m in enumerate(METHODS):
                c = res['cuts'][m]
                if c is not None:
                    ax.scatter([max(c, 40)], [ymax - (i + 1) * (ymax - ymin) * 0.08], marker=MARK[m], s=16,
                               color=INK2, zorder=4)
            ax.set_xscale('log')
            ax.set_xlim(32, 1.3e8)
            ax.set_title('%s, %s' % (sname, nname), fontsize=8, loc='left')
    for ax in axes[-1]:
        ax.set_xlabel('lookups (log scale)')
    handles = [plt.Line2D([], [], marker=MARK[m], ls='', color=INK2, label=LABEL[m]) for m in METHODS]
    handles += [plt.Line2D([], [], color=INK, ls=':', label='true warm-up end'),
                plt.Line2D([], [], color=WARM, label='consensus cut')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(outdir, 'synth_examples.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    meths = METHODS + ('consensus',)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
    for col, (nname, metric, xl) in enumerate([('quiet', 'median_log_err', 'median |log10 error| of the cut'),
                                               ('noisy', 'median_log_err', 'median |log10 error| of the cut'),
                                               ('quiet', 'median_abs_bias_pct', 'median |bias| of steady mean, %'),
                                               ('noisy', 'median_abs_bias_pct', 'median |bias| of steady mean, %')]):
        ax = axes[col]
        for row, sname in enumerate(names):
            for j, m in enumerate(meths):
                r = [s for s in summary if s['noise'] == nname and s['scenario'] == sname and s['method'] == m][0]
                v = r[metric]
                yy = row + (j - (len(meths) - 1) / 2) * 0.11
                if v is None:
                    ax.scatter([0], [yy], marker='x', s=14, color=MUTED)
                else:
                    ax.scatter([v], [yy], marker=MARK.get(m, '*'), s=18 if m != 'consensus' else 40,
                               color=INK2 if m != 'consensus' else WARM, zorder=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_title('%s machine' % nname, fontsize=9, loc='left')
        ax.set_xlabel(xl, fontsize=8)
        if 'log' in metric:
            ax.set_xlim(-0.05, 3.0)
        else:
            ax.set_xscale('symlog', linthresh=0.1)
    handles = [plt.Line2D([], [], marker=MARK[m], ls='', color=INK2, label=LABEL[m]) for m in METHODS]
    handles += [plt.Line2D([], [], marker='*', ls='', color=WARM, markersize=9, label='consensus (median)'),
                plt.Line2D([], [], marker='x', ls='', color=MUTED, label='no cut returned')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(outdir, 'synth_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _fmt_calls(c):
    if c is None:
        return 'never'
    if c < 1:
        return '0'
    for div, suf in ((1e9, 'G'), (1e6, 'M'), (1e3, 'K')):
        if c >= div:
            return '%.3g%s' % (c / div, suf)
    return '%d' % c


# ---------------------------------------------------------------- commands

def _jit_cells(j):
    if not j:
        return '- | - | -'
    st = j.get('states')
    return '%.1f%% | %.2f | %s' % (100 * j['robust_cov'], j['lag1'],
                                    ('%.0f / %.0f, %.0f%%' % (st['fast_mean'], st['slow_mean'], 100 * st['slow_share']))
                                    if st else 'one state')


def _core_cell(cores):
    cores = [c for c in cores if c]
    if not cores:
        return '-'
    agg = {}
    for c in cores:
        for k, d in c['classes'].items():
            agg.setdefault(k, []).append(d)
    cells = []
    for k in sorted(agg, key=lambda k: -sum(d['share'] for d in agg[k])):
        ds = agg[k]
        share = sum(d['share'] for d in ds) / len(cores)
        if share < 0.005:
            continue
        cell = '%s %.0f%% %.0f' % (k, 100 * share, float(np.mean([d['mean'] for d in ds])))
        sl = [d['slow_share'] for d in ds if 'slow_share' in d]
        if sl:
            cell += ' (slow %.0f%%)' % (100 * float(np.mean(sl)))
        cells.append(cell)
    return '; '.join(cells)


def _mig_cell(cores):
    cores = [c for c in cores if c]
    if not cores:
        return '-'
    return 'cpu %.0f%%, class %.0f%%' % (100 * float(np.mean([c['migrations'] for c in cores])),
                                       100 * float(np.mean([c['class_changes'] for c in cores])))


def cmd_analyze(a):
    runs = find_runs(a.outdirs)
    if not runs:
        print('no trace_point lines found (run trace_<index> with GRE_TRACE=geo:F:C)', file=sys.stderr)
        return 2
    groups = {}
    for r in runs:
        groups.setdefault((r['dataset'], r['index']), {'runs': []})['runs'].append(r)
    report = {'eps': a.eps, 'groups': []}
    lines = ['| dataset | index | runs | ' + ' | '.join(METHODS) + ' | consensus | tau (calls) | beta | excess (ms) |'
             ' steady ns/lookup | CI95 across runs | between-run share | R-hat | PELT kind | robust CoV | lag-1 |'
             ' two states (fast / slow ns, slow share) | envelope: warm after | envelope: steady ns |'
             ' steady chunks by core (share, mean ns, slow share) | migrations |',
             '|' + '---|' * (len(METHODS) + 19)]
    for key in sorted(groups):
        g = groups[key]
        g['runs'].sort(key=lambda r: r['round'])
        same = all(np.array_equal(r['n'], g['runs'][0]['n']) for r in g['runs'])
        g['results'] = [analyze_series(r['start'], r['n'], r['y'], a.eps) for r in g['runs']]
        if same:
            # The pooled series is the per-chunk median across runs: one cut and one fit per (dataset, index).
            ymed = np.median(np.array([r['y'] for r in g['runs']]), axis=0)
            g['pooled'] = analyze_series(g['runs'][0]['start'], g['runs'][0]['n'], ymed, a.eps)
        else:
            g['pooled'] = g['results'][0]
        base = g['runs'][0]
        g['env_y'] = lower_envelope(base['n'], ymed if same else base['y'])
        g['envelope'] = analyze_series(base['start'], base['n'], g['env_y'], a.eps)
        g['across'] = across_runs(g['results'])
        cons = g['pooled']['consensus']
        chains = [r['y'][r['start'] >= (cons or 0)] for r in g['runs']]
        g['rhat'] = rhat(chains)
        cmap = core_map(a.cores)
        g['cmap'] = cmap
        st = (g['pooled'].get('jitter') or {}).get('states')
        g['cores'] = [core_stats(r['start'], r['n'], r['y'], r['cpu'], res['consensus'], cmap,
                                 (res.get('jitter') or {}).get('states'))
                      for r, res in zip(g['runs'], g['results'])]
        fit = g['pooled']['info']['relax']
        per = {m: [res['cuts'][m] for res in g['results']] for m in METHODS}
        med = {m: (float(np.median([c for c in per[m] if c is not None])) if any(c is not None for c in per[m])
                   else None) for m in METHODS}
        acr = g['across']
        envg = g['envelope']
        lines.append('| %s | %s | %d | %s | %s | %.3g | %.2f | %.2f | %.2f | %s | %s | %.3f | %s | %s | %s | %.2f | %s | %s |' % (
            key[0], key[1], len(g['runs']), ' | '.join(_fmt_calls(med[m]) for m in METHODS), _fmt_calls(cons),
            fit['tau'], fit['beta'], fit['excess_ns'] / 1e6, acr.get('mean', float('nan')),
            ('%.2f' % acr['ci95']) if 'ci95' in acr else '-',
            ('%.2f' % acr['between_share']) if 'between_share' in acr else '-', g['rhat'],
            g['pooled']['info']['pelt']['kind'], _jit_cells(g['pooled'].get('jitter', {})),
            _fmt_calls(envg['consensus']), envg.get('steady', {}).get('mean', float('nan')), _core_cell(g['cores']),
            _mig_cell(g['cores'])))
        report['groups'].append({
            'dataset': key[0], 'index': key[1], 'runs': [r['path'] for r in g['runs']],
            'per_run_cuts': per, 'pooled_cuts': g['pooled']['cuts'], 'consensus': cons,
            'relax': {k: v for k, v in fit.items()}, 'steady_pooled': g['pooled'].get('steady'),
            'across_runs': acr, 'rhat': g['rhat'], 'pelt_kind': g['pooled']['info']['pelt']['kind'],
            'pelt_segments': g['pooled']['info']['pelt']['segments'],
            'warmup_excess_ns_pooled': g['pooled'].get('warmup_excess_ns'), 'jitter': g['pooled'].get('jitter'),
            'noise': g['pooled'].get('noise'),
            'cores': g['cores'],
            'envelope': {'cuts': envg['cuts'], 'consensus': envg['consensus'], 'steady': envg.get('steady'),
                         'noise': envg.get('noise'), 'relax': envg['info']['relax']}})
    print('Warm-up detection (eps = %g; calls = lookups since the first get(); cut = first call of the steady phase)'
          % a.eps)
    print()
    print('\n'.join(lines))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(report, f, indent=1, default=float)
    if a.plot:
        os.makedirs(a.plot, exist_ok=True)
        plt = _mpl()
        for key, g in groups.items():
            plot_group(plt, key, g, os.path.join(a.plot, 'phases_%s_%s.png' % key), a.eps)
        plot_overview(plt, groups, os.path.join(a.plot, 'phases_overview.png'), a.eps)
    return 0


def cmd_synth(a):
    rows, examples = run_synth(a.reps, a.seed, a.eps)
    summ = summarize_synth(rows)
    print('Synthetic validation: %d series per scenario and noise level; true warm-up end at max(eps = %g, 2 x drift sd)' % (a.reps, a.eps))
    print()
    print('| noise | scenario | ' + ' | '.join(METHODS) + ' | consensus |')
    print('|' + '---|' * (len(METHODS) + 3))
    for nname in NOISE:
        for sname, _, _ in SCENARIOS:
            cells = []
            for m in METHODS + ('consensus',):
                r = [s for s in summ if s['noise'] == nname and s['scenario'] == sname and s['method'] == m][0]
                if r['median_log_err'] is None:
                    cells.append('none')
                else:
                    cells.append('%.2f / %.2f%%%s' % (r['median_log_err'], r['median_abs_bias_pct'],
                                                     '' if r['found_share'] == 1 else ' (%.0f%% found)'
                                                     % (100 * r['found_share'])))
            print('| %s | %s | %s |' % (nname, sname, ' | '.join(cells)))
    print()
    print('Cells: median |log10 error| of the cut / median |bias| of the steady mean after the cut.')
    if a.json:
        with open(a.json, 'w') as f:
            json.dump({'summary': summ}, f, indent=1)
    if a.plot:
        os.makedirs(a.plot, exist_ok=True)
        plot_synth(_mpl(), examples, summ, a.plot)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd')
    pa = sub.add_parser('analyze')
    pa.add_argument('outdirs', nargs='+')
    pa.add_argument('--eps', type=float, default=EPS)
    pa.add_argument('--json')
    pa.add_argument('--plot')
    pa.add_argument('--cores', default='auto', help="CPU -> core class map: auto (macOS: E, P0, P1 ...), none, or "
                    "'0-3:E,4-9:P0,10-15:P1'")
    ps = sub.add_parser('synth')
    ps.add_argument('--reps', type=int, default=40)
    ps.add_argument('--seed', type=int, default=1)
    ps.add_argument('--eps', type=float, default=EPS)
    ps.add_argument('--json')
    ps.add_argument('--plot')
    a = p.parse_args()
    if a.cmd == 'analyze':
        return cmd_analyze(a)
    if a.cmd == 'synth':
        return cmd_synth(a)
    p.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
