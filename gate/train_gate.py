#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Train the gate (include/gret/gate.h) from timed labels: leave-one-dataset-out, then include/gret/gate_model.h.

    python3 gate/train_gate.py LABELS_DIR FEATURES.jsonl --out-dir DIR [--header include/gret/gate_model.h]
                               [--bases "lipp btree"] [--no-veto-lipp]

LABELS_DIR is a run_baselines.sh directory (runs.tsv) with, per base B, the indexes B (raw), pw512_B, pw4096_B and
nf_B on every dataset; FEATURES.jsonl is tools/gate_features output for the same key files (matched by basename).

Labels: per base and dataset, the median throughput of each variant over its repeats; the oracle is the variant with
the highest median. A variant the gate cannot build is not a candidate: raw LIPP where gate.h vetoes it (more than
0.1% of the keys above 2^53 with a 64-bit long double, i.e. macOS arm64; --no-veto-lipp for x86 labels), where
run_baselines.sh does not run it either. The regret of a choice is 100 (1 - thr(choice) / thr(oracle)), in %.

Models, per base, each fitted by exhaustive search for the least total regret on its training datasets:
  const   one variant everywhere (a depth-0 tree);
  rule    a threshold rule on one feature: f <= t -> leaf A, else leaf B;
  tree2   a depth-2 tree; each child is a leaf or a threshold rule, so it also covers const and rule.
A leaf is (pick, fallback): fallback (pw512 or pw4096) is built where pick is vetoed, as in gate.h. Thresholds lie
between two adjacent training values: the geometric mean for the positive, heavy-tailed features (LOG_FEATURES), the
mean otherwise. Ties in training regret go to fewer splits, then the wider gap between the two training values (in
the feature's scale, over its training range), then GateFeatures::kNames order, then the lower threshold.
Validation: leave-one-dataset-out (LODO) over the datasets; the model type with the least mean LODO regret (ties:
the simpler) is refitted on every dataset and written to the header. The header keeps a '*' tree, the documented
placeholder (raw, pw512 where raw is vetoed), for the bases without labels (alex, pgm, artunsync).
Baselines, scored on the same labels: always-raw (pw512 where raw is vetoed, the placeholder), always-pw512,
always-pw4096, always-nf, and NFL's switch (nflsw: pw512 if conflict_pw512 <= conflict_raw, else raw).

Writes into --out-dir: labels.tsv, features.jsonl (the rows used), lodo.tsv, choices.tsv, model.json, tables.md;
and the header. Python 3.6+, stdlib only.
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import OrderedDict

VARIANTS = ['raw', 'pw512', 'pw4096', 'nf']        # gret::Variant, in order
FALLBACKS = ['pw512', 'pw4096']                     # a leaf's fallback (never nf: it needs a flow file)
FEATURES = ['log2_range', 'central_share', 'ks_uniform', 'pla4_per_1k', 'pla64_per_1k', 'conflict_raw',
            'conflict_pw512', 'gap_cv', 'collide_pw512']  # GateFeatures::kNames, in order
LOG_FEATURES = {'pla4_per_1k', 'pla64_per_1k', 'conflict_raw', 'conflict_pw512', 'gap_cv'}
EPS = 1e-9  # regret ties (in % points summed over datasets)


def die(msg):
    sys.exit('train_gate.py: ' + msg)


def tilde(path):  # $HOME -> ~ in what goes into the repo
    home = os.path.expanduser('~')
    path = os.path.abspath(path)
    return '~' + path[len(home):] if path.startswith(home + os.sep) else path


def index_name(base, v):
    return base if v == 'raw' else v + '_' + base


# ---- data

def read_labels(labels_dir):
    """{(dataset, index): [throughput, ...]} from runs.tsv (run_baselines.sh writes only successful runs)."""
    cells = OrderedDict()
    with open(os.path.join(labels_dir, 'runs.tsv')) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['success_read'] == '' or float(r['throughput_ops']) <= 0:
                continue
            cells.setdefault((r['dataset'], r['index']), []).append(float(r['throughput_ops']))
    return cells


def read_features(path):
    rows = OrderedDict()
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[os.path.basename(r['file'])] = r
    return rows


def raw_vetoed(base, feat, veto_lipp):
    """gate.h's RawLimits on the platform the labels come from (see the docstring)."""
    if base == 'lipp' and veto_lipp and feat['above_2_53'] * 1000 > feat['n']:
        return True
    if base in ('alex', 'pgm') and feat['n'] > 0 and feat['max'] == 2 ** 64 - 1:
        return True
    return False


class Dataset:
    def __init__(self, name, base, feat, thr, vetoed):
        self.name, self.base, self.feat = name, base, feat
        self.f = feat['features']
        self.thr = thr                  # {variant: median throughput} of the buildable variants
        self.vetoed = vetoed            # raw cannot be built (gate.h's veto)
        self.oracle = max(VARIANTS, key=lambda v: (thr.get(v, -1), -VARIANTS.index(v)))

    def buildable(self, v):
        return v in self.thr and not (v == 'raw' and self.vetoed)

    def realize(self, leaf):
        """The variant gate.h builds for leaf (pick, fallback): pick, else fallback, else pw512."""
        pick, fb = leaf
        if self.buildable(pick):
            return pick
        return fb if self.buildable(fb) else 'pw512'

    def regret(self, v):
        if v not in self.thr:
            die('%s %s: no label for %s' % (self.name, self.base, v))
        return 100.0 * (1.0 - self.thr[v] / self.thr[self.oracle])


# ---- models: a tree is {'feature', 'threshold', 'le', 'gt'} or a leaf {'leaf': [pick, fallback]}

def holds(f, feature, thr):  # gate.h: feature <= threshold; NaN fails
    x = f[feature]
    return x == x and x <= thr


def predict(tree, f):
    while 'leaf' not in tree:
        tree = tree['le'] if holds(f, tree['feature'], tree['threshold']) else tree['gt']
    return tuple(tree['leaf'])


def splits(tree):
    return 0 if 'leaf' in tree else 1 + splits(tree['le']) + splits(tree['gt'])


def best_leaf(ds):
    best = None
    for pick in VARIANTS:
        for fb in FALLBACKS:
            cost = sum(d.regret(d.realize((pick, fb))) for d in ds)
            if best is None or cost < best[0] - EPS:  # ties: VARIANTS order, then pw512 first
                best = (cost, {'leaf': [pick, fb]})
    return best


def midpoint(feature, a, b):
    t = math.sqrt(a * b) if feature in LOG_FEATURES and a > 0 else (a + b) / 2
    return t if a <= t < b else a


def scaled(feature, x):
    return math.log(x) if feature in LOG_FEATURES and x > 0 else x


def candidate_splits(ds):
    """(feature index, threshold, normalized gap, left, right) over every feature and adjacent value pair."""
    for fi, feature in enumerate(FEATURES):
        vals = sorted(set(d.f[feature] for d in ds))
        if len(vals) < 2:
            continue
        span = scaled(feature, vals[-1]) - scaled(feature, vals[0])
        for a, b in zip(vals, vals[1:]):
            t = midpoint(feature, a, b)
            gap = (scaled(feature, b) - scaled(feature, a)) / span if span > 0 else 0.0
            yield fi, t, gap, [d for d in ds if holds(d.f, feature, t)], [d for d in ds if not holds(d.f, feature, t)]


def fit(ds, depth):
    """Least-regret tree of at most `depth` splits per path; returns (cost, tree)."""
    best_cost, best_tree = best_leaf(ds)
    best_key = (best_cost, 0, 0.0, -1, 0.0)
    if depth == 0 or len(ds) < 2:
        return best_cost, best_tree
    for fi, t, gap, le, gt in candidate_splits(ds):
        (cl, tl), (cg, tg) = fit(le, depth - 1), fit(gt, depth - 1)
        tree = {'feature': FEATURES[fi], 'threshold': t, 'le': tl, 'gt': tg}
        key = (cl + cg, splits(tree), -gap, fi, t)
        if key[0] < best_key[0] - EPS or (abs(key[0] - best_key[0]) <= EPS and key[1:] < best_key[1:]):
            best_key, best_tree = key, tree
    return best_key[0], best_tree


MODELS = OrderedDict([('const', 0), ('rule', 1), ('tree2', 2)])  # name -> depth, simplest first


def describe(tree):
    if 'leaf' in tree:
        pick, fb = tree['leaf']
        return pick if pick != 'raw' else 'raw|' + fb
    return '(%s <= %.6g ? %s : %s)' % (tree['feature'], tree['threshold'], describe(tree['le']), describe(tree['gt']))


# ---- the header

def tree_to_cpp(base, tree):
    """gate.h's GateTree: nodes {root, child if root holds, child if it fails}, leaves 2 (root failed) + (child)."""
    def child(t):  # (feature, threshold, leaf if holds, leaf if fails)
        if 'leaf' in t:
            return None, 0.0, t['leaf'], t['leaf']
        if 'leaf' not in t['le'] or 'leaf' not in t['gt']:
            die('tree deeper than 2')
        return t['feature'], t['threshold'], t['le']['leaf'], t['gt']['leaf']
    if 'leaf' in tree:
        f = [None, None, None]; thr = [0.0, 0.0, 0.0]; leaves = [tree['leaf']] * 4
    else:
        f1, t1, a, b = child(tree['le'])
        f2, t2, c, d = child(tree['gt'])
        f = [tree['feature'], f1, f2]; thr = [tree['threshold'], t1, t2]; leaves = [a, b, c, d]
    fs = ', '.join('"%s"' % x if x else 'nullptr' for x in f)
    ts = ', '.join('%.17g' % x for x in thr)
    ls = ['{Variant::%s, Variant::%s}' % tuple(l) for l in leaves]
    return '    {"%s", {%s},\n     {%s},\n     {%s, %s,\n      %s, %s}},' % ((base, fs, ts) + tuple(ls))


def chunks(s, n):  # adjacent C++ string literals of at most n characters, split after a space where possible
    out = []
    while len(s) > n:
        cut = s.rfind(' ', 0, n) + 1 or n
        out.append(s[:cut])
        s = s[cut:]
    return out + [s]


def write_header(path, trees, source, notes):
    lines = [
        '// SPDX-License-Identifier: MIT',
        "// The gate's decision model, included by gate.h after it defines Variant, GateLeaf and GateTree (do not include it",
        '// directly). GENERATED by gate/train_gate.py: do not edit; rerun it on new labels and rebuild (EXTRA=transforms).',
        '//',
        '// Format (C++17, namespace gret, both inline constexpr):',
        '//   kGateModelSource  one string: what trained the model (labels, validation).',
        "//   kGateTrees[]      one GateTree per base; the first entry whose base matches wins, \"*\" matches any base.",
        '//     {"base", {f_root, f_true, f_false}, {thr_root, thr_true, thr_false},',
        '//      {{pick, fallback}, {pick, fallback}, {pick, fallback}, {pick, fallback}}}',
        '//     a test is feature <= thr (nullptr always holds); leaves (root, child) = (holds, holds), (holds, fails),',
        '//     (fails, holds), (fails, fails); fallback is built where pick cannot be (raw vetoed, nf without a flow).',
        '//',
    ] + ['// ' + n if n else '//' for n in notes] + [
        '#pragma once',
        '',
        'namespace gret {',
        '',
        'inline constexpr const char* kGateModelSource =',
    ] + ['    "%s"' % c.replace('\\', '\\\\').replace('"', '\\"') for c in chunks(source, 100)]
    lines[-1] += ';'
    lines += ['', 'inline constexpr GateTree kGateTrees[] = {']
    for base, tree in trees:
        lines.append(tree_to_cpp(base, tree))
    lines += ['};', '', '}  // namespace gret', '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


PLACEHOLDER = {'leaf': ['raw', 'pw512']}


# ---- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('labels_dir')
    ap.add_argument('features')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--header', help='write the model here (include/gret/gate_model.h)')
    ap.add_argument('--bases', default='lipp btree')
    ap.add_argument('--no-veto-lipp', action='store_true', help='labels from an 80-bit long double (x86)')
    a = ap.parse_args()
    labels_dir = os.path.abspath(a.labels_dir)
    runs = os.path.join(labels_dir, 'runs.tsv')
    cells, feats = read_labels(labels_dir), read_features(a.features)
    names = list(OrderedDict.fromkeys(d for d, _ in cells))
    missing = [d for d in names if d not in feats]
    if missing:
        die('no features for %s in %s' % (' '.join(missing), a.features))
    for name in names:  # the C++ switch (gate_features) and nflsw below must agree
        f = feats[name]['features']
        if feats[name]['nflsw'] != ('pw512' if f['conflict_pw512'] <= f['conflict_raw'] else 'raw'):
            die('%s: nflsw in %s disagrees with the features' % (name, a.features))
    os.makedirs(a.out_dir, exist_ok=True)
    bases = a.bases.split()
    out = {'labels': tilde(runs), 'features': tilde(a.features), 'bases': {}}
    lab = open(os.path.join(a.out_dir, 'labels.tsv'), 'w')
    lab.write('base\tdataset\tvariant\tindex\trepeats\tmedian_mops\tmin_mops\tmax_mops\tspread_pct\toracle\n')
    data = {}
    for base in bases:
        ds = []
        for name in names:
            feat = feats[name]
            vetoed = raw_vetoed(base, feat, not a.no_veto_lipp)
            thr, stats = {}, {}
            for v in VARIANTS:
                t = cells.get((name, index_name(base, v)))
                if not t:
                    if not (v == 'raw' and vetoed):
                        die('%s: no successful %s runs in %s' % (name, index_name(base, v), runs))
                    continue
                if v == 'raw' and vetoed:
                    die('%s: raw %s ran although the gate vetoes it' % (name, base))
                thr[v] = statistics.median(t)
                stats[v] = (len(t), min(t), max(t))
            d = Dataset(name, base, feat, thr, vetoed)
            ds.append(d)
            for v in VARIANTS:
                if v in thr:
                    n, lo, hi = stats[v]
                    lab.write('%s\t%s\t%s\t%s\t%d\t%.3f\t%.3f\t%.3f\t%.1f\t%s\n' % (
                        base, name, v, index_name(base, v), n, thr[v] / 1e6, lo / 1e6, hi / 1e6,
                        100 * (hi - lo) / thr[v], 'yes' if v == d.oracle else ''))
                else:
                    lab.write('%s\t%s\t%s\t%s\t0\t\t\t\t\tvetoed\n' % (base, name, v, index_name(base, v)))
        data[base] = ds
    lab.close()
    with open(os.path.join(a.out_dir, 'features.jsonl'), 'w') as f:
        for name in names:
            f.write(json.dumps(feats[name], sort_keys=True) + '\n')

    lodo_rows, choice_rows, md, final = [], [], [], []
    for base in bases:
        ds = data[base]
        res = OrderedDict()  # model -> [(dataset, realized variant, regret, fold model)]
        for m, depth in MODELS.items():
            res[m] = []
            for i, d in enumerate(ds):
                _, tree = fit(ds[:i] + ds[i + 1:], depth)
                v = d.realize(predict(tree, d.f))
                res[m].append((d.name, v, d.regret(v), describe(tree)))
        fixed = OrderedDict([
            ('always-raw', lambda d: d.realize(('raw', 'pw512'))),
            ('always-pw512', lambda d: 'pw512'),
            ('always-pw4096', lambda d: 'pw4096'),
            ('always-nf', lambda d: 'nf'),
            ('nflsw', lambda d: d.realize(('pw512' if d.f['conflict_pw512'] <= d.f['conflict_raw'] else 'raw',
                                           'raw'))),
        ])
        for m, rule in fixed.items():
            res[m] = [(d.name, rule(d), d.regret(rule(d)), '') for d in ds]
        summary = OrderedDict()
        for m, r in res.items():
            reg = [x[2] for x in r]
            summary[m] = {'mean_regret_pct': statistics.mean(reg), 'worst_regret_pct': max(reg),
                          'share_of_oracle_pct': statistics.mean(100 - x for x in reg),
                          'oracle_hits': sum(1 for x, d in zip(r, ds) if x[1] == d.oracle), 'datasets': len(ds)}
            for (name, v, g, fm) in r:
                lodo_rows.append((base, m, name, v, g, fm))
        chosen = min(MODELS, key=lambda m: (round(summary[m]['mean_regret_pct'], 9), MODELS[m]))
        cost, tree = fit(ds, MODELS[chosen])
        final.append((base, tree))
        fin = [(d.name, d.realize(predict(tree, d.f))) for d in ds]
        for (name, v), d in zip(fin, ds):
            nfl = dict((x[0], x[1]) for x in res['nflsw'])[name]
            choice_rows.append((base, name, d.oracle, v, d.regret(v), dict((x[0], x[1]) for x in res[chosen])[name],
                                nfl))
        out['bases'][base] = {
            'oracle': OrderedDict((d.name, d.oracle) for d in ds),
            'summary': summary, 'chosen_model': chosen, 'final_tree': tree, 'final_tree_text': describe(tree),
            'final_in_sample_mean_regret_pct': cost / len(ds),
            'final_choices': OrderedDict(fin),
            'lodo': OrderedDict((m, [{'dataset': x[0], 'choice': x[1], 'regret_pct': x[2], 'fold_model': x[3]}
                                     for x in r]) for m, r in res.items()),
        }
        # tables.md
        md.append('### %s\n' % base)
        md.append('| dataset | oracle | ' + ' | '.join(VARIANTS) + ' | margin over 2nd |')
        md.append('|---' * (len(VARIANTS) + 3) + '|')
        for d in ds:
            cells_ = []
            for v in VARIANTS:
                if v not in d.thr:
                    cells_.append('vetoed')
                    continue
                lo, hi = min(cells[(d.name, index_name(base, v))]), max(cells[(d.name, index_name(base, v))])
                s = '%.2f (%.0f%%)' % (d.thr[v] / 1e6, 100 * (hi - lo) / d.thr[v])
                cells_.append('**%s**' % s if v == d.oracle else s)
            second = sorted(d.thr.values())[-2] if len(d.thr) > 1 else d.thr[d.oracle]
            md.append('| %s | %s | %s | %.1f%% |' % (d.name, d.oracle, ' | '.join(cells_),
                                                     100 * (d.thr[d.oracle] / second - 1)))
        md.append('\nLODO (const, rule, tree2) and fixed baselines, regret in % of the oracle\'s median throughput:\n')
        md.append('| model | mean regret | worst regret | mean share of oracle | picks the oracle |')
        md.append('|---|---|---|---|---|')
        for m, s in summary.items():
            md.append('| %s%s | %.2f%% | %.2f%% | %.2f%% | %d/%d |' % (
                m, ' (chosen)' if m == chosen else '', s['mean_regret_pct'], s['worst_regret_pct'],
                s['share_of_oracle_pct'], s['oracle_hits'], s['datasets']))
        md.append('\nPer-dataset choices (LODO for const/rule/tree2):\n')
        md.append('| dataset | oracle | ' + ' | '.join(res) + ' |')
        md.append('|---' * (len(res) + 2) + '|')
        for i, d in enumerate(ds):
            md.append('| %s | %s | %s |' % (d.name, d.oracle, ' | '.join(
                '%s (%.1f%%)' % (res[m][i][1], res[m][i][2]) for m in res)))
        md.append('\nFinal model (%s, fitted on all %d datasets, in-sample mean regret %.2f%%): `%s`\n' % (
            chosen, len(ds), cost / len(ds), describe(tree)))

    with open(os.path.join(a.out_dir, 'lodo.tsv'), 'w') as f:
        f.write('base\tmodel\tdataset\tchoice\tregret_pct\tfold_model\n')
        for r in lodo_rows:
            f.write('%s\t%s\t%s\t%s\t%.3f\t%s\n' % r)
    with open(os.path.join(a.out_dir, 'choices.tsv'), 'w') as f:
        f.write('base\tdataset\toracle\tfinal_model_choice\tfinal_regret_pct\tlodo_choice\tnflsw_choice\n')
        for r in choice_rows:
            f.write('%s\t%s\t%s\t%s\t%.3f\t%s\t%s\n' % r)
    with open(os.path.join(a.out_dir, 'model.json'), 'w') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(a.out_dir, 'tables.md'), 'w') as f:
        f.write('Labels: median Mops/s over repeats (spread = (max - min) / median); oracle in bold. Source: %s\n\n'
                % tilde(runs) + '\n'.join(md) + '\n')
    if a.header:
        src = 'gate/train_gate.py on %s (%d datasets); LODO-chosen per base: %s' % (
            tilde(runs), len(names), ', '.join('%s=%s' % (b, out['bases'][b]['chosen_model']) for b in bases))
        notes = ['Trained on %s' % tilde(runs), 'with the features of %s.' % tilde(a.features)]
        for b in bases:
            s = out['bases'][b]
            notes.append('%s: %s, LODO mean regret %.2f%% (worst %.2f%%): %s' % (
                b, s['chosen_model'], s['summary'][s['chosen_model']]['mean_regret_pct'],
                s['summary'][s['chosen_model']]['worst_regret_pct'], s['final_tree_text']))
        notes.append("'*': the bases without labels (alex, pgm, artunsync) keep the placeholder: raw, pw512 where vetoed.")
        write_header(a.header, final + [('*', PLACEHOLDER)], src, notes)
    sys.stdout.write(open(os.path.join(a.out_dir, 'tables.md')).read())


if __name__ == '__main__':
    main()
