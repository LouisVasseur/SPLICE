#!/usr/bin/env python3
"""Cross-dataset warm-up summary of the GRE trace batch (traces/main, traces/lipp) from warmup_detect.py analyze.

    python3 analysis/warmup_eval.py summary  [--warm warmup/warmup.json] [--out warmup]   (from results/<tag>)
    python3 analysis/warmup_eval.py overview [--jobs 4] [--out warmup/overview]

summary: heatmaps of the consensus and online cuts (dataset x index), relaxation tau against index pages, cold excess,
protocol-warm-up coverage and the shortest safe warm-up per index; core-cluster switches per run from the
GRE_TRACE_CPU column (a cut at a P0 <-> P1 switch is a machine event, not the index warming up). Writes
OUT/warmup_eval.json, OUT/warmup_eval.md and OUT/warmup_*.png. overview: warmup_detect.plot_overview per dataset.
"""
import argparse
import json
import math
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'scripts'))
import warmup_detect as wd  # noqa: E402

DS = ('fb', 'osm', 'books', 'covid', 'genome', 'history', 'libio', 'planet', 'stack', 'wise')
IX = ('nullindex', 'splice', 'pgm', 'btree', 'alex', 'lipp', 'scaleli_b', 'scaleli_n')
LOGDIRS = ('traces/main/logs', 'traces/lipp/logs')
PAGE = 16384          # macOS arm64 page
TOTAL = 30000000      # lookups per run (--ops)
SWITCH_MIN = 8        # chunks a run must stay on the new cluster for a switch to count (brief hops are noise)
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
MARK = ['o', 's', '^', 'D', 'v', 'P', 'X', '<']


def base(ds):
    return ds.replace('.sorted', '')


def log_path(g):
    for d in LOGDIRS:
        p = os.path.join(d, os.path.basename(g['runs'][0]))
        if os.path.exists(p):
            return p
    return None


def log_field(path, key):
    rx = re.compile(r'^%s: (\S+)' % re.escape(key))
    for line in open(path, errors='replace'):
        m = rx.match(line)
        if m:
            return m.group(1)
    return None


def switches(run, cmap):
    """Calls at which the run moved to another core class and stayed there >= SWITCH_MIN chunks."""
    if run['cpu'] is None:
        return []
    lab = [(cmap or {}).get(int(c), 'cpu%d' % c) for c in run['cpu']]
    out, cur, i = [], lab[0], 1
    while i < len(lab):
        if lab[i] != cur and len(set(lab[i:i + SWITCH_MIN])) == 1 and i + SWITCH_MIN <= len(lab):
            out.append({'call': float(run['start'][i]), 'from': cur, 'to': lab[i]})
            cur = lab[i]
        i += 1
    return out


def summarize(warm):
    W = json.load(open(warm))
    cmap = wd.core_map('auto')
    rows = {}
    for g in W['groups']:
        ds, ix = base(g['dataset']), g['index'].replace('trace_', '')
        lp = log_path(g)
        run = wd.parse_log(lp)
        rss = log_field(lp, 'index_rss_bytes')
        r = g['relax']
        cons = g['consensus']
        sw = switches(run, cmap)
        # a cut within a factor 1.5 of a persistent cluster switch is attributed to the switch
        at_sw = cons is not None and cons > 0 and any(cons / 1.5 <= s['call'] <= cons * 1.5 for s in sw)
        relax_cut = g['pooled_cuts'].get('relax')
        cold = bool(r['significant'] and r['A'] > 0 and not r['drift'])
        st = g['steady_pooled'] or {}
        cls = {}
        for c in (g['cores'] or [None])[0:1]:
            if c:
                cls = {k: {'share': v['share'], 'mean': v['mean']} for k, v in c['classes'].items()}
        rows[(ds, ix)] = {
            'dataset': ds, 'index': ix, 'log': lp, 'consensus': cons, 'cuts': g['pooled_cuts'],
            'online_cut': g['pooled_cuts'].get('online'), 'online_stop': g['online']['pooled_stop'],
            'pelt_kind': g['pelt_kind'], 'no_steady': g['pelt_kind'] == 'no steady state',
            'tau': r['tau'], 'beta': r['beta'], 'A': r['A'], 'mu': r['mu'], 'relax_significant': bool(r['significant']),
            'relax_at_grid_edge': bool(r['at_grid_edge']), 'cold_transient': cold,
            'relax_cut': relax_cut, 'excess_ms_fit': r['excess_ns'] / 1e6 if cold else 0.0,
            'excess_ms_fit_signed': r['excess_ns'] / 1e6,
            'excess_ms_empirical': (g['warmup_excess_ns_pooled'] or 0.0) / 1e6,
            'steady_ns': st.get('mean'), 'steady_ci95': st.get('ci95'),
            'index_rss_bytes': int(rss) if rss else None, 'pages': int(rss) / PAGE if rss else None,
            'switches': sw, 'cut_at_switch': at_sw, 'steady_by_class': cls,
            # what the index itself needs: the cold transient's end (relax) when there is one, else nothing
            'index_warmup': relax_cut if cold and relax_cut is not None else 0.0}
    return rows


def per_index(rows):
    out = {}
    for ix in IX:
        R = [r for (d, i), r in rows.items() if i == ix]
        if not R:
            continue
        cons = [r['consensus'] for r in R if r['consensus'] is not None]
        nosw = [r['consensus'] for r in R if r['consensus'] is not None and not r['cut_at_switch']]
        iw = [(r['index_warmup'], r['dataset']) for r in R]
        good = [(r['pages'], r['tau']) for r in R if r['cold_transient'] and not r['relax_at_grid_edge'] and r['pages']]
        fit = None
        if len(good) >= 3 and np.ptp(np.log10([p for p, _ in good])) >= 0.3:
            x, y = np.log10([p for p, _ in good]), np.log10([t for _, t in good])
            sl, ic = np.polyfit(x, y, 1)
            fit = {'slope': float(sl), 'n': len(good)}
        out[ix] = {
            'datasets': len(R), 'max_consensus': max(cons) if cons else None,
            'max_consensus_dataset': max(((r['consensus'], r['dataset']) for r in R if r['consensus'] is not None),
                                         default=(None, None))[1],
            'max_consensus_no_switch': max(nosw) if nosw else None,
            'max_index_warmup': max(iw)[0], 'max_index_warmup_dataset': max(iw)[1],
            'cold_transients': sum(r['cold_transient'] for r in R),
            'max_online_stop': max((r['online_stop'] for r in R if r['online_stop'] is not None), default=None),
            'online_never_sure': sum(r['online_stop'] is None for r in R),
            'median_tau': float(np.median([r['tau'] for r in R if r['cold_transient']])) if any(r['cold_transient'] for r in R) else None,
            'touches_per_lookup_k': float(np.median([p / t for p, t in good])) if good else None,
            'tau_vs_pages_fit': fit,
            'excess_ms_fit_median': float(np.median([r['excess_ms_fit'] for r in R])),
            'covered_by_20M': all((r['consensus'] or 0) <= wd.PROTOCOL_W for r in R),
            'no_steady_runs': sum(r['no_steady'] for r in R)}
    return out


def fmt(c):
    return '-' if c is None else wd._fmt_calls(c)


def heat(ax, M, txt, mark, vmin=1e3, vmax=3e7, cmap=None, title='', cols=IX):
    import matplotlib.colors as mc
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if isinstance(v, float) and math.isnan(v):
                fc = 'white'
                ax.add_patch(__import__('matplotlib').patches.Rectangle((j, i), 0.96, 0.92, facecolor='white',
                             edgecolor=wd.GRID, lw=0.8))
            else:
                fc = cmap(mc.LogNorm(vmin, vmax)(min(max(v, vmin), vmax)))
                ax.add_patch(__import__('matplotlib').patches.Rectangle((j, i), 0.96, 0.92, color=fc, lw=0))
            lum = 0.2126 * fc[0] + 0.7152 * fc[1] + 0.0722 * fc[2] if not isinstance(fc, str) else 1
            ax.text(j + 0.48, i + 0.40, txt[i][j], ha='center', va='center', fontsize=7.5,
                    color='white' if lum < 0.5 else wd.INK)
            if mark[i][j]:
                ax.text(j + 0.48, i + 0.74, mark[i][j], ha='center', va='center', fontsize=6.3,
                        color='white' if lum < 0.5 else wd.INK2)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(DS), 0)
    ax.set_xticks([j + 0.48 for j in range(len(cols))])
    ax.set_xticklabels(cols, rotation=30, ha='right', fontsize=8)
    ax.set_yticks([i + 0.46 for i in range(len(DS))])
    ax.set_yticklabels(DS, fontsize=8)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9, loc='left', color=wd.INK)


def plots(rows, pix, out):
    plt = wd._mpl()
    import matplotlib.colors as mc
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    seq = mc.LinearSegmentedColormap.from_list('warm', ['#fdf1eb', '#f6b796', wd.WARM, '#8a3410'])
    # (1) consensus / online cut heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, key, title in ((axes[0], 'consensus', 'consensus cut (log median of mser, pelt, relax, epoch)'),
                           (axes[1], 'online_cut', 'online (streaming) cut; small: lookups it had to see')):
        M = np.full((len(DS), len(IX)), np.nan, dtype=object)
        T = [['' for _ in IX] for _ in DS]
        K = [['' for _ in IX] for _ in DS]
        for i, d in enumerate(DS):
            for j, x in enumerate(IX):
                r = rows.get((d, x))
                if r is None:
                    T[i][j] = 'not run' if not (x == 'lipp' and d == 'covid') else 'build failed'
                    M[i, j] = float('nan')
                    continue
                v = r[key]
                if v is None:
                    M[i, j] = float('nan')
                    T[i][j] = 'none'
                    K[i][j] = 'never sure' if key == 'online_cut' else 'no cut'
                    continue
                M[i, j] = float(v)
                T[i][j] = fmt(v)
                if key == 'consensus':
                    K[i][j] = ' '.join(f for f, c in (('S', r['cut_at_switch']), ('n', r['no_steady'])) if c)
                else:
                    K[i][j] = 'stop ' + fmt(r['online_stop'])
        heat(ax, M, T, K, cmap=seq, title=title)
    sm = ScalarMappable(norm=mc.LogNorm(1e3, 3e7), cmap=seq)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label('warm-up end, lookups (log; 0 shown at 1K)', fontsize=8)
    fig.text(0.0, -0.02, 'S: the cut sits at a persistent P0 <-> P1 cluster switch of the run (a machine event); '
             'n: PELT finds no level lasting 25% of the run. 30M lookups per run, battery + Low Power Mode, one run '
             'per cell.', fontsize=8, color=wd.INK2)
    fig.savefig(os.path.join(out, 'warmup_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # (3) relaxation tau vs index pages
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    allp = [r['pages'] for r in rows.values() if r['pages'] and r['index'] != 'nullindex']
    pp = np.logspace(math.log10(min(allp) / 2), math.log10(max(allp) * 2), 50)
    for k in (1, 4, 16, 64):
        ax.plot(pp, pp / k, color=wd.GRID if k != 4 else wd.MUTED, lw=1.0, ls='--', zorder=1)
        ax.text(pp[-1], pp[-1] / k, ' k=%d' % k, fontsize=7.5, color=wd.INK2, va='center')
    hs = []
    for j, x in enumerate(IX[1:]):
        R = [r for (d, i), r in rows.items() if i == x and r['pages']]
        for r in R:
            good = r['cold_transient'] and not r['relax_at_grid_edge']
            ax.scatter(r['pages'], r['tau'], s=46, marker=MARK[j], color=PAL[j] if good else 'white',
                       edgecolor=PAL[j], lw=1.2, zorder=3)
            if x == 'lipp':
                ax.annotate(r['dataset'], (r['pages'], r['tau']), xytext=(4, 3), textcoords='offset points',
                            fontsize=7, color=wd.INK2)
        f = pix.get(x, {}).get('tau_vs_pages_fit')
        k = pix.get(x, {}).get('touches_per_lookup_k')
        hs.append(Line2D([], [], marker=MARK[j], color=PAL[j], ls='', ms=7,
                         label='%s%s%s' % (x, (', k~%.1f' % k) if k else '', (', slope %.2f' % f['slope']) if f else '')))
    hs.append(Line2D([], [], marker='o', mfc='white', mec=wd.INK2, ls='', ms=7, label='no cold transient (hollow)'))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('index pages P = index_rss_bytes / 16 KiB')
    ax.set_ylabel('relaxation tau (lookups)')
    ax.legend(handles=hs, fontsize=7.5, loc='upper left')
    ax.set_title('first-touch model: tau = P / k (k distinct new pages per lookup); dashed: k = 1, 4, 16, 64',
                 fontsize=9, loc='left', color=wd.INK)
    fig.savefig(os.path.join(out, 'warmup_tau_vs_pages.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # (4) cold excess (ms): A tau Gamma(1 + 1/beta) of the relaxation fit, cold transients only
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    M = np.full((len(DS), len(IX)), np.nan, dtype=object)
    T = [['' for _ in IX] for _ in DS]
    K = [['' for _ in IX] for _ in DS]
    for i, d in enumerate(DS):
        for j, x in enumerate(IX):
            r = rows.get((d, x))
            if r is None:
                T[i][j] = 'not run' if not (x == 'lipp' and d == 'covid') else 'build failed'
                M[i, j] = float('nan')
                continue
            M[i, j] = max(r['excess_ms_fit'], 1.0)
            T[i][j] = '%.0f' % r['excess_ms_fit'] if r['cold_transient'] else '0'
            K[i][j] = 'emp %.0f' % r['excess_ms_empirical']
    heat(ax, M, T, K, vmin=1.0, vmax=1e4, cmap=seq,
         title='cold excess, ms (fit A tau Gamma(1+1/beta); 0 = no cold transient); small: empirical sum before the cut')
    sm = ScalarMappable(norm=mc.LogNorm(1.0, 1e4), cmap=seq)
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01).set_label('ms (log)', fontsize=8)
    fig.savefig(os.path.join(out, 'warmup_cold_excess.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # (5) protocol coverage: per index, each dataset's consensus cut, index transient end, online stop; 20M line
    fig, ax = plt.subplots(figsize=(11, 4.8))
    xmin = 3e3
    for j, x in enumerate(IX):
        R = [r for (d, i), r in rows.items() if i == x]
        for r in R:
            jit = (DS.index(r['dataset']) - 4.5) * 0.045
            c = r['consensus']
            if c is not None:
                ax.scatter(max(c, xmin), j + jit, s=26, color=wd.WARM if not r['cut_at_switch'] else 'white',
                           edgecolor=wd.WARM, lw=1.1, zorder=3)
            if r['index_warmup']:
                ax.scatter(max(r['index_warmup'], xmin), j + jit, s=30, marker='D', color=wd.STEADY, zorder=4)
            if r['online_stop'] is not None:
                ax.scatter(max(r['online_stop'], xmin), j + jit, s=40, marker='|', color=wd.ONLINE, lw=1.6, zorder=2)
            else:
                ax.scatter(TOTAL, j + jit, s=40, marker='>', color=wd.ONLINE, zorder=2)
        p = pix.get(x)
        if p:
            ax.text(5.2e7, j, 'index %s / consensus %s' % (fmt(p['max_index_warmup']), fmt(p['max_consensus'])),
                    fontsize=7.5, va='center', color=wd.INK2)
    ax.axvline(wd.PROTOCOL_W, color=wd.MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.axvline(TOTAL, color=wd.GRID, lw=1.0)
    ax.text(wd.PROTOCOL_W, -0.45, ' 20M protocol warm-up', fontsize=7.5, color=wd.INK2, ha='right')
    ax.set_xscale('log')
    ax.set_xlim(xmin, 4.5e7)
    ax.set_yticks(range(len(IX)))
    ax.set_yticklabels(IX, fontsize=8)
    ax.set_ylim(len(IX) - 0.5, -0.6)
    ax.set_xlabel('lookups (log); one mark per dataset')
    ax.grid(axis='y', visible=False)
    hs = [Line2D([], [], marker='D', color=wd.STEADY, ls='', label='index cold transient end (relax fit, eps 1%)'),
          Line2D([], [], marker='o', color=wd.WARM, ls='', label='consensus cut'),
          Line2D([], [], marker='o', mfc='white', mec=wd.WARM, ls='', label='consensus cut at a cluster switch'),
          Line2D([], [], marker='|', color=wd.ONLINE, ls='', ms=9, mew=1.6, label='online stop (> : never sure)')]
    ax.legend(handles=hs, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.45, -0.13), ncol=4)
    ax.set_title('Does the 20M warm-up cover every transient? right: shortest safe warm-up per index '
                 '(index transient / consensus, max over datasets)', fontsize=9, loc='left', color=wd.INK)
    fig.savefig(os.path.join(out, 'warmup_coverage.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_md(rows, pix, path):
    L = ['| dataset | index | consensus | relax cut | online cut | online stop | tau | beta | cold | excess ms (fit) |'
         ' excess ms (emp.) | steady ns | pages | cut at switch | PELT kind |', '|' + '---|' * 15]
    for d in DS:
        for x in IX:
            r = rows.get((d, x))
            if r is None:
                continue
            L.append('| %s | %s | %s | %s | %s | %s | %.3g | %.2f | %s | %.0f | %.0f | %.1f | %s | %s | %s |' % (
                d, x, fmt(r['consensus']), fmt(r['relax_cut']), fmt(r['online_cut']), fmt(r['online_stop']),
                r['tau'], r['beta'], 'yes' if r['cold_transient'] else '-', r['excess_ms_fit'],
                r['excess_ms_empirical'], r['steady_ns'] or float('nan'),
                '%.3g' % r['pages'] if r['pages'] else '-', 'yes' if r['cut_at_switch'] else '-', r['pelt_kind']))
    L += ['', '| index | datasets | cold transients | median tau | k = P / tau | tau-vs-P slope | max index transient |'
          ' max consensus | max consensus w/o switch | max online stop | online never sure | runs w/o steady level |'
          ' 20M covers consensus |', '|' + '---|' * 13]
    for x, p in pix.items():
        f = p['tau_vs_pages_fit']
        L.append('| %s | %d | %d | %s | %s | %s | %s (%s) | %s (%s) | %s | %s | %d | %d | %s |' % (
            x, p['datasets'], p['cold_transients'], '%.3g' % p['median_tau'] if p['median_tau'] else '-',
            '%.2f' % p['touches_per_lookup_k'] if p['touches_per_lookup_k'] else '-',
            '%.2f (n=%d)' % (f['slope'], f['n']) if f else '-', fmt(p['max_index_warmup']),
            p['max_index_warmup_dataset'], fmt(p['max_consensus']), p['max_consensus_dataset'],
            fmt(p['max_consensus_no_switch']), fmt(p['max_online_stop']), p['online_never_sure'], p['no_steady_runs'],
            'yes' if p['covered_by_20M'] else 'no'))
    open(path, 'w').write('\n'.join(L) + '\n')


def _overview_job(args):
    ds, files, out = args
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            os.symlink(os.path.abspath(f), os.path.join(td, os.path.basename(f)))
        groups = wd.analyze_groups([td], wd.EPS, 'auto')
        plt = wd._mpl()
        wd.plot_overview(plt, groups, os.path.join(out, 'overview_%s.png' % ds), wd.EPS)
    return ds


def overview(jobs, out):
    os.makedirs(out, exist_ok=True)
    work = []
    for d in DS:
        files = []
        for ld in LOGDIRS:
            for name in sorted(os.listdir(ld)):
                m = wd.RE_LOG.match(name)
                if m and base(m.group('ds')) == d:
                    files.append(os.path.join(ld, name))
        work.append((d, files, out))
    from multiprocessing import Pool
    with Pool(jobs) as p:
        for ds in p.imap_unordered(_overview_job, work):
            print('overview', ds, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('summary', 'overview'))
    ap.add_argument('--warm', default='warmup/warmup.json')
    ap.add_argument('--out', default='warmup')
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    if a.cmd == 'overview':
        overview(a.jobs, a.out)
        return
    rows = summarize(a.warm)
    pix = per_index(rows)
    os.makedirs(a.out, exist_ok=True)
    rep = {'rows': [r for r in rows.values()], 'per_index': pix, 'protocol_w': wd.PROTOCOL_W, 'page': PAGE}
    json.dump(rep, open(os.path.join(a.out, 'warmup_eval.json'), 'w'), indent=1, default=float)
    write_md(rows, pix, os.path.join(a.out, 'warmup_eval.md'))
    plots(rows, pix, a.out)


if __name__ == '__main__':
    main()
