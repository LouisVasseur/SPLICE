#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SPLICE's results and attribution from a run_parallel.sh (or run_baselines.sh) directory.

    python3 scripts/splice_report.py OUT_DIR [--props OUT_DIR/dataset_props_200M.jsonl] [--plot FILE.png]

Reads OUT_DIR/runs.tsv (median throughput over repeats, spread = (max - min) / median), the SPLICE build lines
('gret: <name> ... E[lines]=...') in each run's log, and dataset properties from scripts/dataset_bounds.cpp. Prints
markdown: (1) throughput, every index; (2) SPLICE's build per dataset; (3) attribution: what each component adds, as
throughput ratios between ladder names that differ in one component; (4) SPLICE against the best GRE index, next to the
dataset properties. --plot writes a three-panel figure. Needs numpy and matplotlib for --plot only.
"""
import argparse
import json
import os
import re
import statistics as st

LADDER = ['spl_t', 'spl_gt', 'spl_tv', 'splice', 'splice_eq', 'spl_fixk']
BASES = ['lipp', 'alex', 'pgm', 'btree']
LAYERED = ['g_alex', 'pw512_alex', 'vp_alex']
STEPS = [  # (label, from, to): one component switched on
    ('G on T', 'spl_t', 'spl_gt'), ('V on T', 'spl_t', 'spl_tv'), ('V on G+T', 'spl_gt', 'splice'),
    ('G on T+V', 'spl_tv', 'splice'), ('PLA knots vs equi', 'splice_eq', 'splice'),
    ('adaptive vs fixed granularity', 'spl_fixk', 'splice'), ('all of G+V on T', 'spl_t', 'splice')]


def load_runs(out):
    rows = {}
    with open(os.path.join(out, 'runs.tsv')) as f:
        next(f)
        for line in f:
            c = line.rstrip('\n').split('\t')
            if len(c) < 8: continue
            d, i, r, thr, mem, ok, wall, log = c
            d = re.sub(r'_10M$', '', d)
            rows.setdefault((d, i), []).append(dict(thr=float(thr) / 1e6, mem=float(mem), log=os.path.join(out, log)))
    return rows


def build_stats(path):
    """The SPLICE build line of a GRE log, as a dict (empty when there is none)."""
    try:
        txt = open(path, errors='replace').read()
    except OSError:
        return {}
    m = None
    for m in re.finditer(r'^gret: (\S+) n=(\d+) (.*)$', txt, re.M): pass
    if not m: return {}
    out = {'n': int(m.group(2))}
    for k, v in re.findall(r'(\w+(?:\[\w+\])?|slots/key)=([^\s(]+)', m.group(3)):
        try: out[k] = float(v)
        except ValueError: out[k] = v
    return out


def med(v): return st.median(v) if v else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--props')
    ap.add_argument('--plot')
    a = ap.parse_args()
    rows = load_runs(a.out)
    props = {}
    pf = a.props or os.path.join(a.out, 'dataset_props_200M.jsonl')
    if os.path.exists(pf):
        for line in open(pf):
            p = json.loads(line); props[re.sub(r'_10M$', '', os.path.basename(p['file']))] = p
    ds = sorted({d for d, _ in rows}, key=lambda d: ['fb', 'genome', 'osm', 'planet', 'books', 'covid', 'libio',
                                                      'history', 'stack', 'wise'].index(d) if d in
                ['fb', 'genome', 'osm', 'planet', 'books', 'covid', 'libio', 'history', 'stack', 'wise'] else 99)
    idx = [i for i in LADDER + BASES + LAYERED if any((d, i) in rows for d in ds)]
    T = {(d, i): med([r['thr'] for r in rows[(d, i)]]) for (d, i) in rows}
    spread = {(d, i): (max(r['thr'] for r in rows[(d, i)]) - min(r['thr'] for r in rows[(d, i)])) / T[(d, i)]
              for (d, i) in rows if T[(d, i)] > 0}
    nrep = {k: len(v) for k, v in rows.items()}

    print('## Throughput (M lookups/s, median over repeats; repeats and spread in the cell)\n')
    print('| dataset | ' + ' | '.join(idx) + ' |')
    print('|---|' + '---|' * len(idx))
    for d in ds:
        cells = []
        for i in idx:
            if (d, i) not in T: cells.append('-'); continue
            best = max(T.get((d, j), 0) for j in idx)
            s = '%.2f' % T[(d, i)]
            if T[(d, i)] == best: s = '**' + s + '**'
            cells.append('%s <sub>%dx %.0f%%</sub>' % (s, nrep[(d, i)], 100 * spread.get((d, i), 0)))
        print('| %s | %s |' % (d, ' | '.join(cells)))

    print('\n## SPLICE builds (from the GRE logs: the build of each name, first repeat)\n')
    keys = ['knots', 'eps', 'segments', 'exceptions', 'slots/key', 'E[lines]', 'top_bytes', 'bytes_per_key', 'build_ms']
    print('| dataset | name | ' + ' | '.join(keys) + ' |')
    print('|---|---|' + '---|' * len(keys))
    B = {}
    for d in ds:
        for i in LADDER:
            if (d, i) not in rows: continue
            b = build_stats(rows[(d, i)][0]['log']); B[(d, i)] = b
            if not b: continue
            vals = []
            for k in keys:
                v = b.get(k, '')
                if k == 'top_bytes' and v != '': v = '%.1f MB' % (v / 1e6)
                elif isinstance(v, float): v = ('%.4g' % v)
                vals.append(str(v))
            print('| %s | %s | %s |' % (d, i, ' | '.join(vals)))

    print('\n## Attribution: throughput ratio when one component is switched on (median / median)\n')
    print('| dataset | ' + ' | '.join(s[0] for s in STEPS) + ' |')
    print('|---|' + '---|' * len(STEPS))
    for d in ds:
        cells = []
        for lab, f, t in STEPS:
            if (d, f) in T and (d, t) in T and T[(d, f)] > 0: cells.append('%.2fx' % (T[(d, t)] / T[(d, f)]))
            else: cells.append('-')
        print('| %s | %s |' % (d, ' | '.join(cells)))

    print('\n## SPLICE against the best GRE index, with dataset properties (after G)\n')
    print('| dataset | splice | best GRE index | ratio | splice B/key | best B/key | G peels | KL from linear | '
          'D_KS | PLA segs/1k (eps 4) | PLA segs/1k (eps 32) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    summ = []
    for d in ds:
        if (d, 'splice') not in T: continue
        bb = [(T[(d, i)], i) for i in BASES if (d, i) in T]
        if not bb: continue
        bt, bi = max(bb)
        sm = med([r['mem'] for r in rows[(d, 'splice')]]) / max(1, props.get(d, {}).get('n', 1))
        bm = med([r['mem'] for r in rows[(d, bi)]]) / max(1, props.get(d, {}).get('n', 1))
        p = props.get(d, {})
        kg = p.get('Kg', [float('nan')] * 6); n = p.get('n', float('nan'))
        r = T[(d, 'splice')] / bt
        summ.append((d, r, p))
        print('| %s | %.2f | %s %.2f | %.2fx | %.1f | %.1f | %s | %.3g | %.3g | %.2f | %.3f |' % (
            d, T[(d, 'splice')], bi, bt, r, sm, bm, p.get('e', '-'), p.get('KL_B', float('nan')), p.get('DB', float('nan')),
            1000 * kg[0] / n, 1000 * kg[3] / n))
    if a.plot: plot(a.plot, ds, T, idx, summ)


def plot(path, ds, T, idx, summ):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    ink, mute, grid = '#1f2328', '#6e7781', '#d0d7de'
    # Okabe-Ito order (colour-blind safe), fixed per entity
    col = {'splice': '#0072B2', 'lipp': '#E69F00', 'alex': '#009E73', 'pgm': '#CC79A7', 'btree': '#999999'}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={'width_ratios': [2.2, 2.2, 1.4]})
    # A: SPLICE and the GRE indexes
    ax = axs[0]; x = np.arange(len(ds)); names = [i for i in ['splice'] + BASES if i in idx]; w = 0.8 / len(names)
    for k, i in enumerate(names):
        v = [T.get((d, i), np.nan) for d in ds]
        ax.bar(x + (k - (len(names) - 1) / 2) * w, v, w * 0.92, color=col.get(i, mute), label=i)
    ax.set_xticks(x, ds, rotation=30, ha='right', color=ink); ax.set_ylabel('M lookups/s', color=ink)
    ax.set_title('A  SPLICE and GRE\'s indexes (median)', loc='left', fontsize=10, color=ink)
    ax.legend(frameon=False, fontsize=8, ncol=len(names))
    # B: the ladder, relative to T alone
    ax = axs[1]
    steps = [('spl_gt', '+G'), ('spl_tv', '+V'), ('splice', '+G+V'), ('splice_eq', 'equi knots'), ('spl_fixk', 'fixed K')]
    shades = ['#56B4E9', '#0072B2', '#003f6b', '#D55E00', '#F0E442']
    w = 0.8 / len(steps)
    for k, (i, lab) in enumerate(steps):
        v = [T.get((d, i), np.nan) / T.get((d, 'spl_t'), np.nan) for d in ds]
        ax.bar(x + (k - (len(steps) - 1) / 2) * w, v, w * 0.92, color=shades[k], label=lab)
    ax.axhline(1, color=mute, lw=1)
    ax.set_xticks(x, ds, rotation=30, ha='right', color=ink); ax.set_ylabel('throughput / T alone (spl_t)', color=ink)
    ax.set_title('B  Attribution: each component against T alone', loc='left', fontsize=10, color=ink)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    # C: SPLICE / best GRE index against local roughness
    ax = axs[2]
    for d, r, p in summ:
        if not p: continue
        xr = 1000 * p['Kg'][0] / p['n']
        ax.scatter(xr, r, s=36, color=col['splice'])
        ax.annotate(d, (xr, r), textcoords='offset points', xytext=(4, 3), fontsize=8, color=ink)
    ax.axhline(1, color=mute, lw=1); ax.set_xscale('log')
    ax.set_xlabel('PLA segments per 1k keys at eps = 4 (after G)', color=ink); ax.set_ylabel('splice / best GRE index', color=ink)
    ax.set_title('C  Gain against local roughness', loc='left', fontsize=10, color=ink)
    for ax in axs:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color=grid, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    print('\nplot:', path)


if __name__ == '__main__':
    main()
