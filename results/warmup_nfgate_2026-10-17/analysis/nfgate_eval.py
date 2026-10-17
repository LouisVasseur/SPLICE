#!/usr/bin/env python3
"""Evaluate the NF-transform gates of nf_gate (nfgate/cases.jsonl) against the exact-count ground truth and the
timed GRE pair trace_scaleli_b / trace_scaleli_n (warmup/warmup.json + traces/main/logs).

    python3 analysis/nfgate_eval.py [--margin 0.01] [--out nfgate_eval]     (run from results/warmup_nfgate_2026-10-17)

Every gate is a score s (higher = "the flow helps") and a threshold: use the flow iff s > theta. theta0 is the gate's
natural zero (NFL: its 10% keep rule; pred: the margin); LODO learns theta on 9 datasets' windows and tests on the
10th (windows + full). With no positive in the training folds the most permissive threshold that keeps the training
folds error-free is the training maximum, so that is theta_LODO. Writes OUT.json, OUT.md and nfgate_*.png.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'scripts'))
from warmup_detect import INK, INK2, MUTED, GRID, WARM, STEADY, ONLINE, _mpl  # noqa: E402  (repo palette and rcParams)

DS = ('fb', 'osm', 'books', 'covid', 'genome', 'history', 'libio', 'planet', 'stack', 'wise')
MS = (1024, 4096, 16384, 65536)
# 8 validated hues (warmup_detect's core palette) + two inks; a marker per dataset so identity is never colour alone
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948', INK2, MUTED]
MARK = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h']


def sl(x):
    return math.log(x) if x > 0 else float('nan')


# name -> (label, score(case, G), theta0, cost_ns(G), uses the flow output sorted)
def gates():
    g = {}
    for b in (16, 64, 256, 1024):
        g['kl%d' % b] = ('KL to uniform, %d bins (raw - flow)' % b,
                         (lambda c, G, b=b: G['kl'][str(b)]['diff'] if str(b) in G['kl'] else None), 0.0,
                         (lambda G, b=b: G['kl'][str(b)]['ns'] + G['flow_ns'] if str(b) in G['kl'] else None))
    g['trend'] = ('log-gap trend share (raw - flow)', lambda c, G: G['trend_share']['diff'], 0.0,
                  lambda G: G['trend_share']['ns'] + G['flow_ns'] + G['flow_sort_ns'])
    g['nfl'] = ('NFL tail-conflict gain (keep at 10%)', lambda c, G: G['tail_conflict']['gain'], 0.10,
                lambda G: G['tail_conflict']['ns'] + G['flow_ns'] + G['flow_sort_ns'])
    for e in (16, 64):
        g['pla%d' % e] = ('eps-PLA segments, eps %d: ln(raw / flow)' % e,
                          (lambda c, G, e=e: -sl(G['pla'][str(e)]['ratio'])), 0.0,
                          (lambda G, e=e: G['pla'][str(e)]['ns'] + G['flow_ns'] + G['flow_sort_ns']))
    g['w1'] = ('W1 to uniform (raw - flow)', lambda c, G: G['w1']['diff'], 0.0,
               lambda G: G['w1ks_ns'] + G['flow_ns'] + G['flow_sort_ns'])
    g['ks'] = ('KS to uniform (raw - flow)', lambda c, G: G['ks']['diff'], 0.0,
               lambda G: G['w1ks_ns'] + G['flow_ns'] + G['flow_sort_ns'])
    # signed: (best root without the flow - flow root) / best; > 0 iff the flow root would win
    g['root'] = ('root proxy: (min(raw, binary) - flow) / min', lambda c, G: (G['root_proxy']['B'] - G['root_proxy']['flow'])
                 / G['root_proxy']['B'], 0.0, lambda G: G['root_proxy']['ns'] + G['flow_ns'])
    g['pred'] = ('root proxy + mini build: predicted benefit', lambda c, G: G['pred']['benefit'], None,
                 lambda G: G['root_proxy']['ns'] + G['mini']['ns'] + G['flow_ns'])
    # benefit_net's charge: one probe-equivalent per transform; flow-region share from the mini build, +1 if the root flows
    g['pred_net'] = ('pred, net of transform calls', lambda c, G: G['pred']['benefit'] - (
        G['mini']['flow_chunks'] / max(G['mini']['chunks'], 1) + (G['root_proxy']['N'] < G['root_proxy']['B'])) / G['pred']['D_B'],
                     None, lambda G: G['root_proxy']['ns'] + G['mini']['ns'] + G['flow_ns'])
    return g


def score(sf, c, k):
    if k not in c['gate']:
        return float('nan')
    v = sf(c, c['gate'][k])
    return float('nan') if v is None else v


def spearman(x, y):
    def rk(a):
        a = np.asarray(a, float)
        o = np.argsort(a, kind='mergesort')
        r = np.empty(len(a))
        r[o] = np.arange(len(a))
        for v in np.unique(a):   # average ranks on ties
            m = a == v
            r[m] = r[m].mean()
        return r
    rx, ry = rk(x), rk(y)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def cindex(s, y):
    """Concordance (Harrell's C): P(s_i > s_j | y_i > y_j), score ties 1/2; the AUC of a continuous outcome."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    num = den = 0.0
    for i in range(len(s)):
        m = y[i] > y
        den += m.sum()
        num += (s[i] > s[m]).sum() + 0.5 * (s[i] == s[m]).sum()
    return float(num / den) if den else float('nan')


def auc(s, lab):
    s, lab = np.asarray(s, float), np.asarray(lab, bool)
    if lab.all() or not lab.any():
        return None
    return cindex(s, lab.astype(float))


def gre_pairs(warm_json, logdir):
    """trace_scaleli_b vs trace_scaleli_n per dataset: steady mean (consensus cut), envelope steady, warm-up, GRE Mops/s."""
    out = {}
    if not os.path.exists(warm_json):
        return out
    W = json.load(open(warm_json))
    for g in W['groups']:
        ds = g['dataset'].replace('.sorted', '')
        if g['index'] not in ('trace_scaleli_b', 'trace_scaleli_n'):
            continue
        st = g['steady_pooled'] or {}
        env = (g['envelope'] or {}).get('steady') or {}
        tp = None
        log = os.path.join(logdir, os.path.basename(g['runs'][0]))
        for line in open(log, errors='replace'):
            m = re.match(r'^Throughput = (\d+)', line)
            if m:
                tp = int(m.group(1)) / 1e6
        out.setdefault(ds, {})[g['index'].replace('trace_scaleli_', '')] = {
            'steady_ns': st.get('mean'), 'ci95': st.get('ci95'), 'env_ns': env.get('mean'), 'warm': g['consensus'],
            'online_cut': g['pooled_cuts'].get('online'), 'mops': tp, 'log': g['runs'][0]}
    for ds, p in out.items():
        if 'b' in p and 'n' in p:
            b, n = p['b'], p['n']
            p['speedup'] = b['steady_ns'] / n['steady_ns'] - 1          # > 0: N faster
            p['speedup_env'] = b['env_ns'] / n['env_ns'] - 1 if b['env_ns'] and n['env_ns'] else None
            # within-run only (one run each): the 95% half-width of the ratio from the two batch-means intervals
            p['speedup_ci95'] = (1 + p['speedup']) * math.hypot(b['ci95'] / b['steady_ns'], n['ci95'] / n['steady_ns'])
            p['speedup_mops'] = n['mops'] / b['mops'] - 1 if b['mops'] and n['mops'] else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='nfgate/cases.jsonl')
    ap.add_argument('--warm', default='warmup/warmup.json')
    ap.add_argument('--logs', default='traces/main/logs')
    ap.add_argument('--margin', type=float, default=0.01)
    ap.add_argument('--out', default='nfgate_eval')
    ap.add_argument('--budget-us', type=float, default=1000.0, help='gate cost budget (median, sampling excluded)')
    a = ap.parse_args()
    C = [json.loads(l) for l in open(a.cases)]
    C = [c for c in C if c['dataset'] in DS]
    G = gates()
    bn = np.array([c['gt']['benefit_net'] for c in C])
    bg = np.array([c['gt']['benefit'] for c in C])
    truth = (bn > a.margin) & np.array([c['gt']['mem_rel'] < 0.01 for c in C])
    dsa = np.array([c['dataset'] for c in C])
    full = np.array([c['case'] == 'full' for c in C])
    rep = {'margin': a.margin, 'cases': len(C), 'positives': int(truth.sum()),
           'benefit_net': [float(bn.min()), float(bn.max())], 'benefit': [float(bg.min()), float(bg.max())],
           'positives_at_margin0_benefit': int((bg > 0).sum()), 'gates': {}}
    for name, (label, sf, th0, cf) in G.items():
        for m in MS:
            k = str(m)
            s = np.array([score(sf, c, k) for c in C], float)
            if np.isnan(s).all():
                continue
            ok = ~np.isnan(s)
            cost = np.array([cf(c['gate'][k]) if k in c['gate'] and cf(c['gate'][k]) is not None else np.nan for c in C], float)
            th = a.margin if th0 is None else th0
            dec0 = s > th
            # LODO: theta = max over the other 9 datasets' windows (the most permissive error-free threshold)
            thl, decl = {}, np.zeros(len(C), bool)
            for d in DS:
                tr = (dsa != d) & ~full & ok
                thl[d] = float(np.max(s[tr]))
                te = dsa == d
                decl[te] = s[te] > thl[d]
            per = {}
            for d in DS:
                i_full = np.where((dsa == d) & full)[0]
                w = (dsa == d) & ~full
                per[d] = {'full_score': float(s[i_full[0]]) if len(i_full) else None,
                          'full_use_theta0': bool(dec0[i_full[0]]) if len(i_full) else None,
                          'full_use_lodo': bool(decl[i_full[0]]) if len(i_full) else None,
                          'theta_lodo': thl[d], 'windows_use_theta0': int(dec0[w].sum()),
                          'windows_use_lodo': int(decl[w].sum()), 'windows': int(w.sum())}
            # how far the gate is from flipping: the largest score in units of the score's own spread
            spread = float(np.subtract(*np.percentile(s[ok], [75, 25]))) or float(np.std(s[ok])) or 1.0
            rep['gates'].setdefault(name, {'label': label, 'theta0': th, 'by_m': {}})['by_m'][k] = {
                'n': int(ok.sum()),
                'acc_theta0': float(np.mean(dec0[ok] == truth[ok])), 'fp_theta0': int((dec0 & ~truth & ok).sum()),
                'acc_theta0_windows': float(np.mean((dec0 == truth)[ok & ~full])),
                'acc_theta0_full': float(np.mean((dec0 == truth)[ok & full])),
                'acc_lodo': float(np.mean(decl[ok] == truth[ok])), 'fp_lodo': int((decl & ~truth & ok).sum()),
                'acc_lodo_full': float(np.mean((decl == truth)[ok & full])),
                'auc_decision': auc(s[ok], truth[ok]),
                'cindex_benefit_net': cindex(s[ok], bn[ok]), 'spearman_benefit_net': spearman(s[ok], bn[ok]),
                'spearman_benefit_net_windows': spearman(s[ok & ~full], bn[ok & ~full]),
                'headroom_iqr': float((th - np.max(s[ok])) / spread),
                'cost_us_median': float(np.nanmedian(cost) / 1e3), 'cost_us_full': float(np.nanmedian(cost[full]) / 1e3),
                'cost_us_windows': float(np.nanmedian(cost[~full]) / 1e3),
                'sample_us_median': float(np.median([c['gate'][k]['sample_ns'] for c in C if k in c['gate']]) / 1e3),
                'constant': bool(np.nanstd(s) == 0),
                'per_dataset': per, 'scores': [None if math.isnan(x) else float(x) for x in s]}
    # root proxy accuracy against the build's measured root probes
    rp = {}
    for m in MS:
        k = str(m)
        e = [abs(c['gate'][k]['root_proxy']['B'] - c['B']['per_lookup']['root_probes']) for c in C if k in c['gate']]
        agree = [(c['gate'][k]['root_proxy']['raw'] < c['gate'][k]['root_proxy']['binary']) == (c['B']['root'] == 'raw')
                 for c in C if k in c['gate']]
        rp[k] = {'median_abs_err_probes': float(np.median(e)), 'max_abs_err_probes': float(np.max(e)),
                 'root_choice_agree': int(sum(agree)), 'n': len(agree)}
    rep['root_proxy_vs_build'] = rp
    ex = [c['N']['root_probes_fit']['flow'] - min(c['N']['root_probes_fit']['raw'], c['N']['root_probes_fit']['binary'])
          for c in C if c['N']['root_probes_fit']['flow'] > 0]
    rep['flow_root_excess_probes'] = [float(min(ex)), float(max(ex))] if ex else None
    rep['gre'] = gre_pairs(a.warm, a.logs)
    rep['case_index'] = [{'dataset': c['dataset'], 'case': c['case'], 'window': c['window'],
                          'benefit_net': c['gt']['benefit_net'], 'benefit': c['gt']['benefit'],
                          'flow_regions': c['N']['flow_regions'], 'regions': c['N']['regions'],
                          'transform_calls': c['N']['per_lookup']['transform_calls'], 'D_B': c['gt']['D_B']} for c in C]
    # GRE verdict per dataset (one run each): "use" only if N beats B by margin + the within-run CI on the steady
    # mean AND the lower envelope agrees (the open Mac's slow state can favour either run)
    for d, p in rep['gre'].items():
        if 'speedup' in p:
            lim = rep['margin'] + p['speedup_ci95']
            env = p['speedup_env'] if p['speedup_env'] is not None else p['speedup']
            p['verdict'] = ('use' if p['speedup'] > lim and env > rep['margin'] else
                            'slower' if p['speedup'] < -lim and env < -rep['margin'] else 'skip')
    for g in rep['gates'].values():
        for r in g['by_m'].values():
            r['agree_gre_full'] = sum(((rep['gre'].get(d, {}).get('verdict') == 'use') == pd['full_use_lodo'])
                                      for d, pd in r['per_dataset'].items() if 'verdict' in rep['gre'].get(d, {}))
    # choice: within the cost budget and with a score that moves at all (a constant score is "always skip", which no
    # data here can tell from a gate), the best accuracy at theta0, then LODO, then concordance, then the lowest cost
    cand = []
    for name, g in rep['gates'].items():
        for k, r in g['by_m'].items():
            ci = r['cindex_benefit_net'] if r['cindex_benefit_net'] == r['cindex_benefit_net'] else 0.0
            elig = r['cost_us_median'] <= a.budget_us and not r['constant']
            cand.append((not elig, -r['acc_theta0'], -r['acc_lodo'], -round(ci, 2), r['cost_us_median'], name, k))
    cand.sort()
    rep['budget_us'] = a.budget_us
    rep['ranking'] = [{'gate': c[5], 'm': int(c[6]), 'eligible': not c[0], 'acc_theta0': -c[1], 'acc_lodo': -c[2],
                       'cindex': -c[3], 'cost_us': c[4]} for c in cand]
    rep['chosen'] = rep['ranking'][0]
    with open(a.out + '.json', 'w') as f:
        json.dump(rep, f, indent=1)
    write_md(rep, a.out + '.md')
    plots(rep, C, bn, dsa, full)
    print(json.dumps(rep['chosen']))


def write_md(rep, path):
    L = ['| gate | m | theta0 | acc (theta0) | FP (theta0) | acc LODO | FP LODO | C-index vs benefit_net |'
         ' Spearman | headroom (IQR) | cost us (median) | sample us |', '|' + '---|' * 12]
    for name, g in rep['gates'].items():
        for k, r in g['by_m'].items():
            L.append('| %s | %s | %.3g | %.3f | %d | %.3f | %d | %.2f | %.2f | %.2f | %.1f | %.0f |' % (
                name, k, g['theta0'], r['acc_theta0'], r['fp_theta0'], r['acc_lodo'], r['fp_lodo'],
                r['cindex_benefit_net'], r['spearman_benefit_net'], r['headroom_iqr'], r['cost_us_median'],
                r['sample_us_median']))
    open(path, 'w').write('\n'.join(L) + '\n')


def plots(rep, C, bn, dsa, full):
    plt = _mpl()
    from matplotlib.lines import Line2D
    m0 = str(rep['chosen']['m'])
    show = ['kl64', 'trend', 'nfl', 'pla16', 'pla64', 'ks', 'root', 'pred_net']
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), squeeze=False)
    for ax, name in zip(axes.flat, show):
        g = rep['gates'][name]
        r = g['by_m'][m0]
        s = np.array([np.nan if x is None else x for x in r['scores']])
        for i, d in enumerate(DS):
            for isfull in (False, True):
                sel = (dsa == d) & (full == isfull)
                ax.scatter(s[sel], 100 * bn[sel], s=70 if isfull else 26, marker=MARK[i], color=PAL[i],
                           edgecolor=INK if isfull else 'white', lw=1.0 if isfull else 0.6, zorder=3 if isfull else 2)
        ax.axvline(g['theta0'], color=WARM, lw=1.4, ls='--')
        th = [v['theta_lodo'] for v in r['per_dataset'].values()]
        ax.axvspan(min(th), max(th), color=MUTED, alpha=0.18, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title('%s\nC = %.2f, acc %.2f (theta0) / %.2f (LODO), %.0f us' % (
            g['label'], r['cindex_benefit_net'], r['acc_theta0'], r['acc_lodo'], r['cost_us_median']),
            fontsize=8.5, loc='left', color=INK)
        ax.tick_params(labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('measured benefit_net (% of D)')
    for ax in axes[-1]:
        ax.set_xlabel('gate score (higher = flow helps)')
    h = [Line2D([], [], marker=MARK[i], color=PAL[i], ls='', ms=7, label=d) for i, d in enumerate(DS)]
    h += [Line2D([], [], marker='o', color=MUTED, mec=INK, ls='', ms=9, label='full 200M (outlined)'),
          Line2D([], [], color=WARM, ls='--', label='natural threshold theta0'),
          Line2D([], [], color=MUTED, lw=6, alpha=0.4, label='LODO thresholds (10 folds)')]
    fig.legend(handles=h, loc='lower center', ncol=13, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.0, 1.0, 'm = %s sampled keys. Use-the-flow region (benefit_net > %.0f%%) is empty: the measured '
             'benefit_net is %.3f%% to %.3f%%, so every point right of a threshold is a false positive.'
             % (m0, 100 * rep['margin'], 100 * rep['benefit_net'][0], 100 * rep['benefit_net'][1]), fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig('nfgate_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    names = list(rep['gates'])
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for gi, name in enumerate(names):
        g = rep['gates'][name]
        ms = sorted(int(k) for k in g['by_m'])
        col, mk = PAL[gi % len(PAL)], MARK[gi % len(MARK)]
        ls = '-' if gi < len(PAL) else ':'
        a0 = [g['by_m'][str(m)]['acc_theta0'] for m in ms]
        al = [g['by_m'][str(m)]['acc_lodo'] for m in ms]
        ci = [g['by_m'][str(m)]['cindex_benefit_net'] for m in ms]
        cost = [g['by_m'][str(m)]['cost_us_median'] for m in ms]
        axes[0].plot(ms, a0, color=col, marker=mk, ls=ls, lw=1.6, ms=6, label=name)
        axes[1].plot(ms, al, color=col, marker=mk, ls=ls, lw=1.6, ms=6)
        axes[2].plot(ms, ci, color=col, marker=mk, ls=ls, lw=1.6, ms=6)
        axes[3].plot(cost, a0, color=col, marker=mk, ls=ls, lw=1.2, ms=6)
    axes[0].set_ylabel('decision accuracy, natural threshold (90 cases)')
    axes[1].set_ylabel('decision accuracy, LODO threshold')
    axes[2].set_ylabel('C-index of score vs benefit_net')
    axes[2].axhline(0.5, color=MUTED, lw=0.8, ls='--')
    axes[3].set_ylabel('decision accuracy, natural threshold')
    for ax in axes[:3]:
        ax.set_xscale('log', base=2)
        ax.set_xlabel('sample size m (keys)')
    axes[3].set_xscale('log')
    axes[3].axvline(rep['budget_us'], color=WARM, lw=1.0, ls='--')
    axes[3].set_xlabel('gate cost, us (median, efficiency core; sampling excluded)')
    ch = rep['chosen']
    axes[3].annotate('chosen: %s, m=%d' % (ch['gate'], ch['m']), (ch['cost_us'], ch['acc_theta0']),
                     xytext=(8, -14), textcoords='offset points', fontsize=8, color=INK)
    axes[3].text(rep['budget_us'], 0.05, ' budget %.0f us' % rep['budget_us'], fontsize=7.5, color=INK2)
    axes[0].legend(fontsize=7.5, ncol=2, loc='lower left')
    fig.tight_layout()
    fig.savefig('nfgate_vs_m.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # per-dataset decisions at the chosen m: full 200M decision (LODO theta) and window false positives
    k = str(ch['m'])
    gl = [n for n in names if k in rep['gates'][n]['by_m']]
    gre = rep['gre']
    cols = ['truth', 'GRE timed'] + gl
    for mode, key, wkey, fn in (('LODO threshold', 'full_use_lodo', 'windows_use_lodo', 'nfgate_decisions.png'),
                                ('natural threshold theta0', 'full_use_theta0', 'windows_use_theta0',
                                 'nfgate_decisions_theta0.png')):
        decision_table(plt, rep, gl, gre, cols, k, mode, key, wkey, fn)


def decision_table(plt, rep, gl, gre, cols, k, mode, key, wkey, fn):
    fig, ax = plt.subplots(figsize=(1.0 + 0.82 * len(cols), 0.42 * len(DS) + 1.6))
    for i, d in enumerate(DS):
        row = []
        row.append(('skip', 'benefit_net\n%.1e' % [c['benefit_net'] for c in rep['case_index'] if c['dataset'] == d and c['case'] == 'full'][0]))
        p = gre.get(d, {})
        sp = p.get('speedup')
        if sp is None:
            row.append(('na', '-'))
        else:
            v = p['verdict']
            row.append((v, '%+.1f%% (env %+.1f%%)' % (100 * sp, 100 * p['speedup_env'])))
        for n in gl:
            pd = rep['gates'][n]['by_m'][k]['per_dataset'][d]
            row.append(('use' if pd[key] else 'skip', '%d/%d' % (pd[wkey], pd['windows'])))
        for j, (v, txt) in enumerate(row):
            fc = {'use': '#fbd9c9', 'skip': '#dbe8f8', 'slower': '#e6e5e1', 'na': 'white'}[v]
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=fc, lw=0))
            ax.text(j + 0.48, i + 0.32, v, ha='center', va='center', fontsize=8, color=INK)
            ax.text(j + 0.48, i + 0.68, txt, ha='center', va='center', fontsize=6.5, color=INK2)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(DS), 0)
    ax.set_xticks([j + 0.48 for j in range(len(cols))])
    ax.set_xticklabels(cols, rotation=40, ha='right', fontsize=8)
    ax.set_yticks([i + 0.46 for i in range(len(DS))])
    ax.set_yticklabels(DS, fontsize=8)
    ax.grid(False)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title('Full 200M decision per gate at m = %s, %s (small: windows saying "use", of 8). '
                 'GRE timed: N vs B steady speedup, one run each' % (k, mode), fontsize=8.5, loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
