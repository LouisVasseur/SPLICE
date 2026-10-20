#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Warm-up detection on trace_<name> runs (include/gret/warmup.h): where does per-chunk lookup latency stop moving?

    python3 scripts/warmup_detect.py OUT_DIR [--plot FILE.png] [--window 10]

OUT_DIR is a run_baselines.sh directory whose logs hold 'gret_trace <chunk> <calls> <ns_per_lookup>' lines. Two
detectors, both on the chunk series y_k (ns per lookup over chunk k), both "warm from the first chunk after which the
condition holds for every later window" (a suffix condition, so a late spike does not hide behind an early quiet run):
  variance    sliding-window coefficient of variation std/mean over w chunks, below theta;
  derivative  windowed relative slope |mean(next w) - mean(this w)| / (w * mean(this w)) per chunk, below delta;
  both thresholds are 1.5x the 99th percentile of the same statistic over the tail (the last 30% of the run, taken as
  stationary), so they are calibrated on the run's own noise rather than fixed.
A compact re-implementation of the two online methods of SPLICE's earlier warmup_detect.py, without its offline
consensus (MSER, PELT, relaxation fit, epochs). Prints, per run: the cut chunk and the lookups it covers for each
method, the mean before and after the cut, and the tail mean. Needs numpy (and matplotlib for --plot).
"""
import argparse
import glob
import os
import re

import numpy as np


def series(log):
    ys = [float(m.group(1)) for m in re.finditer(r'^gret_trace \d+ \d+ ([0-9.]+)$', open(log).read(), re.M)]
    return np.array(ys)


def suffix_first(ok):
    """First index i such that ok[j] holds for every j >= i (len(ok) if none)."""
    bad = np.nonzero(~ok)[0]
    return int(bad[-1]) + 1 if bad.size else 0


def detect(y, w):
    n = y.size
    m = np.array([y[k:k + w].mean() for k in range(n - w + 1)])
    s = np.array([y[k:k + w].std() for k in range(n - w + 1)])
    cov = s / m
    d = np.abs(m[w:] - m[:-w]) / (w * m[:-w])          # windowed relative slope per chunk
    t0, t1 = int(0.7 * cov.size), int(0.7 * d.size)    # the tail (last 30%): the stationary reference
    theta = 1.5 * float(np.quantile(cov[t0:], 0.99))   # thresholds calibrated on the tail's own statistics:
    delta = 1.5 * float(np.quantile(d[t1:], 0.99))     # 1.5x their 99th percentile, so noise alone rarely trips them
    cut_var = suffix_first(cov < theta)
    cut_der = suffix_first(d < delta)
    tail = y[int(0.7 * n):]
    return dict(theta=theta, delta=delta, cut_var=min(cut_var, n), cut_der=min(cut_der, n), tail=tail.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--plot')
    ap.add_argument('--choose-w', action='store_true', help='print "W <lookups>": the warm-up length for every cold run')
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--chunk', type=int, default=100000)
    a = ap.parse_args()
    chunk = a.chunk
    st = os.path.join(a.out, 'MANIFEST.txt')
    if os.path.exists(st):
        m = re.search(r'TRACE_CHUNK=(\d+)', open(st).read())
        if m: chunk = int(m.group(1))
    runs = {}
    for log in sorted(glob.glob(os.path.join(a.out, 'logs', '*.log'))):
        y = series(log)
        if y.size > 3 * a.window:
            runs[os.path.basename(log)[:-4]] = y
    print('%-36s %8s %10s %8s %10s %9s %9s %9s' % ('run', 'var_cut', 'var_calls', 'der_cut', 'der_calls', 'before', 'after', 'tail'))
    res = {}
    for name, y in runs.items():
        r = detect(y, a.window); res[name] = r
        c = max(r['cut_var'], r['cut_der'])
        before = y[:c].mean() if c else float('nan')
        print('%-36s %8d %10d %8d %10d %9.2f %9.2f %9.2f' % (name, r['cut_var'], r['cut_var'] * chunk, r['cut_der'],
              r['cut_der'] * chunk, before, y[c:].mean() if c < y.size else float('nan'), r['tail']))
    if a.choose_w:  # the protocol: derivative cut, unless the drift it marks is under 2% (then the variance cut); x1.5
        cuts = []
        for name, y in runs.items():
            if '.trace_w_' in name or '.trace_' not in name: continue
            r = res[name]; c = r['cut_der']
            before = y[:c].mean() if c else y.mean(); after = y[c:].mean() if c < y.size else y.mean()
            if abs(before - after) < 0.02 * after: c = r['cut_var']
            cuts.append(c * chunk)
        W = int(-(-1.5 * max(cuts + [0]) // 1e6) * 1e6) if cuts else 0
        print('W %d' % max(W, 1000000))
    if a.plot and runs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = sorted({re.sub(r'\.trace_(w_)?', '.trace_', n) for n in runs})
        fig, axs = plt.subplots(len(names), 1, figsize=(10, 2.4 * len(names)), sharex=True, squeeze=False)
        for ax, base in zip(axs[:, 0], names):
            for n, col, lab in ((base, '#1f5fa8', 'cold (GRE as published)'), (base.replace('.trace_', '.trace_w_'), '#c4561d', 'warmed (w_, untimed in bulk_load)')):
                if n not in runs: continue
                y = runs[n]; x = (np.arange(y.size) + 1) * chunk / 1e6
                ax.plot(x, y, lw=1.0, color=col, label=lab)
                r = res[n]
                if n == base:
                    ax.axvline(r['cut_var'] * chunk / 1e6, color='#2a8f6a', ls='--', lw=1, label='variance cut (cold)')
                    ax.axvline(r['cut_der'] * chunk / 1e6, color='#6e7781', ls=':', lw=1.2, label='derivative cut (cold)')
            ax.set_title(base.replace('.trace_', '  ').replace('.r1', ''), loc='left', fontsize=9)
            ax.set_ylabel('ns / lookup', fontsize=8)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axs[0, 0].legend(fontsize=7, loc='upper right', frameon=False, ncol=2)
        axs[-1, 0].set_xlabel('lookups (millions); chunks of %d' % chunk)
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print('plot:', a.plot)


if __name__ == '__main__':
    main()
