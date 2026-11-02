#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Slide figures for the warm-up traces: where each detector of scripts/warmup_detect.py says warm-up ends.

    python3 scripts/plot_warmup.py FB200M_DIR TEN_M_DIR OUT_DIR [--window-200m 10] [--window-10m 25]

FB200M_DIR  results/warm_fb200M_2026-10-20 (fb at 200M keys: trace_{alex,lipp} cold, trace_w_{alex,lipp} warmed)
TEN_M_DIR   results/warm_10M_2026-10-20 ({fb,genome,stack}_10M, same runs)
OUT_DIR     where F33_warmup_fb200M.{png,pdf} and F34_warmup_10M.{png,pdf} are written

Each directory holds '<dataset>.trace_<index>.r1.trace' files of 'gret_trace <chunk> <calls> <ns per lookup>' lines.
The cuts are the ones warmup_detect.py computes: series(), suffix_first() and detect() are imported from it, the two
statistics drawn in the strips are recomputed with detect()'s own lines, and every cut is asserted equal to detect()'s.
Before plotting, every run listed in a directory's detect.txt is re-detected and compared (chunk and calls, both
methods); a mismatch stops the script. The windows that reproduce the committed detect.txt files are w = 10 chunks
at 200M and w = 25 chunks at 10M, hence the defaults.

Panels: latency per chunk against calls (log x, log y), cold run dark, warmed run faint gray; derivative cut solid
#d55e00, variance cut dashed #0072b2, each shading its warm-up span and drawing the mean after its cut. Strips below:
the derivative statistic |mean(next w) - mean(this w)| / (w * mean(this w)) and the variance statistic (std/mean over
w chunks), each with its threshold (1.5 x its 99th percentile over the last 30% of the run). A statistic indexed k
(the window starting at chunk k) is drawn at chunk k, so the last point above the threshold sits just left of the cut.
Needs numpy and matplotlib.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warmup_detect import detect, series, suffix_first  # noqa: E402  (the ground truth for both cuts)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea, VPacker  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

INK, MUTED, GRID, SPINE = '#1f2328', '#57606a', '#d8dee4', '#8c959f'
COLD, WARM = '#2d333b', '#c2c8cf'
DER = dict(color='#d55e00', ls='-', name='derivative')
VAR = dict(color='#0072b2', ls=(0, (5, 2.5)), name='variance')
FS = 11.5          # body text; nothing in the figures is below 11 pt

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Arial', 'DejaVu Sans'], 'font.size': FS, 'pdf.fonttype': 42,
    'axes.edgecolor': SPINE, 'axes.linewidth': 0.8, 'axes.labelcolor': INK, 'axes.labelsize': FS,
    'xtick.color': SPINE, 'ytick.color': SPINE, 'xtick.labelcolor': MUTED, 'ytick.labelcolor': MUTED,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
})


# ---------------------------------------------------------------- data and detection (exactly warmup_detect.py's)
def load(path):
    y = series(path)
    calls = np.array([int(m.group(1)) for m in re.finditer(r'^gret_trace \d+ (\d+) [0-9.]+$', open(path).read(), re.M)])
    chunk = int(calls[0])
    if calls.size != y.size or np.any(np.diff(calls) != chunk):
        sys.exit('%s: calls column is not uniform chunks' % path)
    return y, chunk


def criteria(y, w):
    """detect()'s statistics, line for line, plus its cuts (asserted equal to detect()'s own)."""
    n = y.size
    m = np.array([y[k:k + w].mean() for k in range(n - w + 1)])
    s = np.array([y[k:k + w].std() for k in range(n - w + 1)])
    cov = s / m
    d = np.abs(m[w:] - m[:-w]) / (w * m[:-w])
    t0, t1 = int(0.7 * cov.size), int(0.7 * d.size)
    theta = 1.5 * float(np.quantile(cov[t0:], 0.99))
    delta = 1.5 * float(np.quantile(d[t1:], 0.99))
    r = detect(y, w)
    cut_var, cut_der = min(suffix_first(cov < theta), n), min(suffix_first(d < delta), n)
    assert (cut_var, cut_der, theta, delta) == (r['cut_var'], r['cut_der'], r['theta'], r['delta'])
    return dict(cov=cov, d=d, theta=theta, delta=delta, cut_var=cut_var, cut_der=cut_der)


def check(dirpath, w):
    """Re-detect every run of detect.txt; print the comparison; return the rows."""
    ref = os.path.join(dirpath, 'detect.txt')
    rows = []
    if not os.path.exists(ref):
        print('%s: no detect.txt, nothing to check' % dirpath)
        return rows
    for line in open(ref).read().splitlines()[1:]:
        p = line.split()
        y, chunk = load(os.path.join(dirpath, p[0] + '.trace'))
        r = detect(y, w)
        got = (r['cut_var'], r['cut_var'] * chunk, r['cut_der'], r['cut_der'] * chunk)
        want = tuple(int(v) for v in p[1:5])
        rows.append((p[0], got, want, got == want))
    print('%s  (w = %d)' % (dirpath, w))
    print('  %-34s %8s %10s %8s %10s   %s' % ('run', 'var_cut', 'var_calls', 'der_cut', 'der_calls', 'detect.txt'))
    for name, got, want, ok in rows:
        print('  %-34s %8d %10d %8d %10d   %s' % ((name,) + got + ('match' if ok else 'MISMATCH %s' % (want,),)))
    return rows


# ---------------------------------------------------------------- formatting helpers
def fmt_calls(c):
    if c >= 1e6:
        return ('%.2f' % (c / 1e6)).rstrip('0').rstrip('.') + 'M'
    return '%.0fk' % (c / 1e3)


def log_ticks(lo, hi, height_in, min_gap_in=0.24):
    """Plain-number ticks on a log axis: the coarsest 1-2-5-like set giving >= 3 ticks at least min_gap_in apart."""
    best = []
    for subs in ((1,), (1, 2, 5), (1, 2, 3, 5), (1, 1.5, 2, 3, 4, 5, 6, 8), (1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9)):
        t = sorted({round(s * 10.0 ** e, 6) for e in range(-1, 6) for s in subs if lo <= s * 10.0 ** e <= hi})
        if len(t) < 2:
            continue
        gap = np.diff(np.log(t)).min() / np.log(height_in[1] / height_in[0]) * height_in[2]
        if gap >= min_gap_in:
            best = t
            if len(t) >= 3:
                break
    return best


def axes_height_in(ax):
    return ax.get_position().height * ax.figure.get_figheight()


def style(ax):
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(which='both', length=3, width=0.6)
    ax.tick_params(which='minor', length=2)


def sample(spec, lw=1.8):
    da = DrawingArea(18, 10, 0, 0)
    da.add_artist(Line2D([0, 18], [5, 5], color=spec['color'], ls=spec['ls'], lw=lw))
    return da


ROW0_IN, ROW_STEP_IN = 0.22, 0.29   # label rows: inches below the top of the main panel


def place_labels(ax, items):
    """Direct labels at the top of the cut lines: line sample(s) + ink text beside the line. items: list of
    (anchor_lo_x, anchor_hi_x or None, specs, text); x None = no line (pinned to the left or right edge). Each takes the
    first slot (row 0 right, row 0 left, row 1 right, ...) that stays inside the panel, overlaps no placed label and
    crosses no other cut line; then the same with the text broken after its colon (two rows); failing both, the first
    in-panel one-row slot, with a white backing. Returns the number of rows used."""
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(rend)
    hin = axes_height_in(ax)
    lines_px = [ax.transData.transform((x, 1))[0] for it in items for x in it[:2] if x is not None]

    def build(it, row, side, two):
        lo, hi, specs, text = it
        kids = [sample(sp) for sp in specs]
        if two:   # 'name:' with the sample on the first line, the rest under it, both flush against the line
            head, rest = text.split(': ', 1)
            first = [TextArea(head, textprops=dict(color=INK, size=FS))]
            first = kids + first if side != 'left' else first + kids[::-1]
            box = VPacker(children=[HPacker(children=first, align='center', pad=0, sep=4),
                                    TextArea(rest, textprops=dict(color=INK, size=FS))],
                          align='right' if side == 'left' else 'left', pad=0, sep=5)
        else:
            txt = TextArea(text, textprops=dict(color=INK, size=FS))
            kids = kids + [txt] if side != 'left' else [txt] + kids[::-1]
            box = HPacker(children=kids, align='center', pad=0, sep=4)
        yf = 1 - (ROW0_IN + (row + 0.5 * two) * ROW_STEP_IN) / hin
        if side == 'pin':
            xy, coords, align, off = (0.012, yf), 'axes fraction', (0, 0.5), (0, 0)
        elif side == 'pinright':
            xy, coords, align, off = (0.988, yf), 'axes fraction', (1, 0.5), (0, 0)
        elif side == 'right':
            xy, coords, align, off = (hi if hi is not None else lo, yf), ('data', 'axes fraction'), (0, 0.5), (5, 0)
        else:
            xy, coords, align, off = (lo, yf), ('data', 'axes fraction'), (1, 0.5), (-5, 0)
        ab = AnnotationBbox(box, xy, xycoords=coords, box_alignment=align, xybox=off, boxcoords='offset points',
                            frameon=False, pad=0.2, zorder=8)
        ax.add_artist(ab)
        return ab, ab.get_window_extent(rend)

    def attempt(order):
      placed, rows, artists, crossings = [], 0, [], 0
      for it in order:
        sides = ('pin', 'pinright') if it[0] is None else ('right', 'left')
        passes = [(True, False), (True, True), (False, False)]   # (no crossing required, two-line text)
        choice = None
        for strict, two in passes:
            for r in range(3):
                for sd in sides:
                    span = set(range(r, r + 1 + two))
                    ab, bb = build(it, r, sd, two)
                    inside = bb.x0 >= axbb.x0 - 1 and bb.x1 <= axbb.x1 + 1
                    free = all(bb.x1 + 4 < q.x0 or bb.x0 - 4 > q.x1 or not (span & qs) for qs, q in placed)
                    crosses = any(bb.x0 < xp < bb.x1 for xp in lines_px)
                    if inside and free and (not strict or not crosses):
                        choice = (span, ab, bb, crosses)
                        break
                    ab.remove()
                if choice:
                    break
            if choice:
                break
        if choice is None:
            ab, bb = build(it, 2, sides[0], False)
            choice = ({2}, ab, bb, True)
        span, ab, bb, crosses = choice
        if crosses:   # a label that must cross a line gets a white backing so the text stays legible
            ab.patch.set(fc='white', ec='none', alpha=0.92, boxstyle='square,pad=0.2')
            ab.frameon = True
        placed.append((span, bb))
        artists.append(ab)
        crossings += crosses
        rows = max(rows, max(span) + 1)
      return rows, artists, crossings

    best = None
    for order in (items, sorted(items, key=lambda it: it[0] is None)):   # as given, then lines first
        rows, artists, crossings = attempt(order)
        if best is None or crossings < best[2]:
            if best is not None:
                for ab in best[1]:
                    ab.remove()
            best = (rows, artists, crossings)
        else:
            for ab in artists:
                ab.remove()
        if best[2] == 0:
            break
    return best[0]



# ---------------------------------------------------------------- one panel: main + two criterion strips
SMOOTH = 0.15  # --smooth: running-median width in log(calls); 0 draws the raw chunks only

def panel(fig, cell, cold, warm, chunk, w, title_left, title_right, show_xlabel, ratios):
    sub = cell.subgridspec(3, 1, height_ratios=ratios, hspace=0.10)
    ax = fig.add_subplot(sub[0])
    sd = fig.add_subplot(sub[1], sharex=ax)
    sv = fig.add_subplot(sub[2], sharex=ax)
    y = cold
    n = y.size
    c = criteria(y, w)
    xm = (np.arange(n) + 0.5) * chunk            # chunk k spans [k*chunk, (k+1)*chunk) calls
    xl, xr = 0.38 * chunk, n * chunk * 1.06
    for a in (ax, sd, sv):
        a.set_xscale('log')
        a.set_xlim(xl, xr)
        style(a)
    for a in (ax, sd, sv):
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_calls(v)))
        a.xaxis.set_minor_formatter(NullFormatter())
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(sd.get_xticklabels(), visible=False)
    if show_xlabel:
        sv.set_xlabel('calls (lookups so far, log scale)')

    # main: latency per chunk
    # the lines drawn: a running median whose window is a fixed width in log(calls) (chunk i: chunks i/(1+s)..i(1+s)),
    # so the early transient keeps its resolution and the late Mac noise is smoothed; the raw chunks stay, faint.
    # The cuts are computed on the raw series either way
    def rmed(v):
        if SMOOTH <= 0: return v
        out = np.empty(v.size)
        for i in range(v.size):
            lo_, hi_ = int((i + 0.5) / (1 + SMOOTH)), int((i + 0.5) * (1 + SMOOTH)) + 1
            out[i] = np.median(v[max(0, lo_):max(hi_, lo_ + 1)])
        return out
    if warm is not None:
        wx = (np.arange(warm.size) + 0.5) * chunk
        if SMOOTH > 0: ax.plot(wx, warm, color=WARM, lw=0.6, alpha=0.35, zorder=2)
        ax.plot(wx, rmed(warm), color=WARM, lw=1.5, zorder=2)
    if SMOOTH > 0: ax.plot(xm, y, color=COLD, lw=0.6, alpha=0.22, zorder=3)
    ax.plot(xm, rmed(y), color=COLD, lw=1.7, zorder=3)
    ax.set_yscale('log')
    ax.set_ylabel('ns / lookup (log)')
    ax.set_title(title_left, loc='left', color=INK, fontsize=13, fontweight='semibold', pad=6)
    if title_right:
        ax.set_title(title_right, loc='right', color=MUTED, fontsize=FS, pad=6)

    # cuts, shading, mean after each cut, and the two criterion strips
    cuts = {'der': c['cut_der'], 'var': c['cut_var']}
    specs = {'der': DER, 'var': VAR}
    for key, a_s, stat, thr in (('der', sd, c['d'], c['delta']), ('var', sv, c['cov'], c['theta'])):
        spec, k = specs[key], cuts[key]
        xs = (np.arange(stat.size) + 0.5) * chunk   # statistic k = the window starting at chunk k
        floor = thr / 60
        a_s.plot(xs, np.maximum(stat, floor), color=spec['color'], lw=1.0, zorder=3)
        a_s.axhline(thr, color=MUTED, lw=1.1, ls=(0, (1, 1.6)), zorder=4)
        a_s.set_yscale('log')
        slo = max(stat[stat > 0].min() if np.any(stat > 0) else floor, floor) * 0.8
        a_s.set_ylim(slo, max(stat.max() * 2.2, thr * 3))
        a_s.yaxis.set_major_locator(FixedLocator([]))
        a_s.yaxis.set_minor_locator(FixedLocator([]))
        a_s.grid(False, axis='y')
        a_s.set_ylabel(spec['name'], rotation=0, ha='right', va='center', labelpad=8, color=INK)
        if 0 < k < n:   # the last window over the threshold: the reason the cut is here
            a_s.plot([xs[k - 1]], [max(stat[k - 1], floor)], 'o', ms=6.5, mfc='white', mec=spec['color'], mew=1.6,
                     zorder=6, clip_on=False)
        if 0 < k <= n:
            for a in (ax, sd, sv):
                a.axvline(k * chunk, color=spec['color'], ls=spec['ls'], lw=1.7, zorder=5)
            ax.axvspan(xl, k * chunk, color=spec['color'], alpha=0.05, lw=0, zorder=1)
            a_s.axvspan(xl, k * chunk, color=spec['color'], alpha=0.06, lw=0, zorder=1)
        if k < n:
            ax.hlines(y[k:].mean(), k * chunk if k else xl, xr, color=spec['color'], ls=spec['ls'], lw=1.1, zorder=4)

    # direct labels at the top of each cut line
    kd, kv = cuts['der'], cuts['var']
    xd, xv = (kd * chunk if 0 < kd <= n else None), (kv * chunk if 0 < kv <= n else None)

    def where(k):
        return 'never settles in the run' if k >= n else 'warm after %s calls' % fmt_calls(k * chunk)

    if kd == 0 and kv == 0:
        items = [(None, None, [DER, VAR], 'neither method fires: warm from the first chunk')]
    elif xd is not None and xv is not None and abs(kd - kv) <= w:   # within one window: say so, anchor outside both
        lo_, hi_ = min(xd, xv), max(xd, xv)
        items = [(lo_, hi_, [DER], 'derivative: %s' % where(kd)),
                 (lo_, hi_, [VAR], 'variance: %s calls, coincides' % fmt_calls(kv * chunk))]
    else:
        items = []
        for key, x in (('der', xd), ('var', xv)):
            k = cuts[key]
            t = '%s: %s' % (specs[key]['name'], 'never fires, warm from the first chunk' if k == 0 else where(k))
            items.append((x, None, [specs[key]], t))
    rows = place_labels(ax, items)

    # y range: data at the bottom, the label rows in a reserved band at the top
    allv = np.concatenate([y] + ([warm] if warm is not None else []))
    lo, hi = allv.min() * 0.95, allv.max() * 1.04
    hin = axes_height_in(ax)
    frac = (ROW0_IN + (rows - 1) * ROW_STEP_IN + 0.2) / hin
    top = np.exp(np.log(hi) + frac / (1 - frac) * np.log(hi / lo))
    ax.set_ylim(lo, top)
    ax.yaxis.set_major_locator(FixedLocator(log_ticks(lo, hi, (lo, top, hin))))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: '%g' % v))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    return c


def legend(fig, x, y):
    h = [Line2D([], [], color=COLD, lw=1.6), Line2D([], [], color=WARM, lw=2.2),
         Line2D([], [], color=DER['color'], ls=DER['ls'], lw=1.8), Line2D([], [], color=VAR['color'], ls=VAR['ls'], lw=1.8),
         Line2D([], [], color=MUTED, ls=(0, (1, 1.6)), lw=1.3),
         Line2D([], [], ls='', marker='o', ms=6.5, mfc='white', mec=MUTED, mew=1.6)]
    lab = ['cold run', 'warmed run (untimed warm-up first)', 'derivative cut', 'variance cut', 'threshold',
           'last window above threshold']
    fig.legend(h, lab, loc='upper left', bbox_to_anchor=(x, y), ncol=6, frameon=False, fontsize=FS, labelcolor=INK,
               handlelength=2.2, columnspacing=1.3, handletextpad=0.6, borderaxespad=0)


def runs_of(dirpath, ds, idx):
    f = os.path.join(dirpath, '%s.trace_%s.r1.trace' % (ds, idx))
    g = os.path.join(dirpath, '%s.trace_w_%s.r1.trace' % (ds, idx))
    cold, chunk = load(f)
    warm = load(g)[0] if os.path.exists(g) else None
    return cold, warm, chunk


def save(fig, out, stem):
    for ext, kw in (('png', dict(dpi=200)), ('pdf', {})):
        p = os.path.join(out, stem + '.' + ext)
        fig.savefig(p, facecolor='white', **kw)
        print('wrote', p)
    plt.close(fig)


def summary(tag, ds, idx, c, chunk):
    print('  %-22s der_cut %5d (%10d calls)   var_cut %5d (%10d calls)   delta %.4g theta %.4g'
          % ('%s %s %s' % (tag, ds, idx), c['cut_der'], c['cut_der'] * chunk, c['cut_var'], c['cut_var'] * chunk,
             c['delta'], c['theta']))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('fb200m_dir')
    ap.add_argument('ten_m_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--window-200m', type=int, default=10, help='w in chunks for the 200M traces (detect.txt: 10)')
    ap.add_argument('--window-10m', type=int, default=25, help='w in chunks for the 10M traces (detect.txt: 25)')
    ap.add_argument('--smooth', type=float, default=0.15, help='running-median width in log(calls), 0 = raw only')
    a = ap.parse_args()
    global SMOOTH
    SMOOTH = a.smooth
    os.makedirs(a.out_dir, exist_ok=True)

    bad = [r for d, w in ((a.fb200m_dir, a.window_200m), (a.ten_m_dir, a.window_10m)) for r in check(d, w) if not r[3]]
    if bad:
        sys.exit('cuts differ from detect.txt for %d run(s); not plotting' % len(bad))

    # F33: fb at 200M, ALEX and LIPP
    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(1, 2, left=0.105, right=0.985, top=0.855, bottom=0.135, wspace=0.27)
    w = a.window_200m
    print('F33 cuts:')
    for j, (idx, nm) in enumerate((('alex', 'ALEX'), ('lipp', 'LIPP'))):
        cold, warm, chunk = runs_of(a.fb200m_dir, 'fb', idx)
        c = panel(fig, gs[0, j], cold, warm, chunk, w, nm, 'fb, 200M keys', True, (4.6, 1.15, 1.15))
        summary('200M', 'fb', idx, c, chunk)
    legend(fig, 0.105, 0.975)
    fig.text(0.105, 0.018, 'Source: GRE chunk traces (%s), %d chunks of %s lookups; cuts from warmup_detect.py, w = %d '
             'chunks, thresholds 1.5 × p99 of the last 30%%.' % (os.path.basename(os.path.normpath(a.fb200m_dir)),
                                                                   cold.size, fmt_calls(chunk), w),
             color=MUTED, fontsize=11)
    save(fig, a.out_dir, 'F33_warmup_fb200M')

    # F34: 10M, rows fb / genome / stack, columns ALEX / LIPP
    fig = plt.figure(figsize=(13.5, 11.5))
    gs = fig.add_gridspec(3, 2, left=0.105, right=0.985, top=0.925, bottom=0.07, wspace=0.27, hspace=0.30)
    w = a.window_10m
    print('F34 cuts:')
    for i, ds in enumerate(('fb', 'genome', 'stack')):
        for j, (idx, nm) in enumerate((('alex', 'ALEX'), ('lipp', 'LIPP'))):
            cold, warm, chunk = runs_of(a.ten_m_dir, ds + '_10M', idx)
            c = panel(fig, gs[i, j], cold, warm, chunk, w, '%s · %s' % (ds, nm), '10M keys' if i == 0 else '',
                      i == 2, (3.4, 0.9, 0.9))
            summary('10M', ds, idx, c, chunk)
    legend(fig, 0.105, 0.99)
    fig.text(0.105, 0.012, 'Source: GRE chunk traces (%s), %d chunks of %s lookups; cuts from warmup_detect.py, w = %d '
             'chunks, thresholds 1.5 × p99 of the last 30%%.' % (os.path.basename(os.path.normpath(a.ten_m_dir)),
                                                                   cold.size, fmt_calls(chunk), w),
             color=MUTED, fontsize=11)
    save(fig, a.out_dir, 'F34_warmup_10M')


if __name__ == '__main__':
    main()
