#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check gate_<base> and nflsw_<base> runs of a GRE binary against train_gate.py's model and labels.

    python3 gate/check_gate.py RUN_DIR MODEL_DIR [--bases "lipp btree"]

RUN_DIR is a run_baselines.sh directory with gate_<base> and nflsw_<base> runs on the labelled datasets; MODEL_DIR is
train_gate.py's --out-dir (model.json, labels.tsv, features.jsonl). For every run it reads the gate's stderr line
(gret: gate|nflsw <base> choice=<c> features=[...] feature_ms=<ms>) and checks:
  - the choice equals the model's final choice for that dataset (nflsw: NFL's switch on the same features);
  - the printed features (%.6g) match features.jsonl within 1e-5 relative: bulk_load saw the same keys;
then prints the run's throughput against the label median of the variant it chose (ratio, and whether it lies in the
label's [min, max] widened by 5%). Exits 1 if a choice or a feature differs. Python 3.6+, stdlib only.
"""
import argparse
import csv
import json
import os
import re
import sys

LINE = re.compile(r'^gret: (gate|nflsw) (\S+) choice=(\S+) features=\[([^\]]*)\] feature_ms=([0-9.]+)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir')
    ap.add_argument('model_dir')
    ap.add_argument('--bases', default='lipp btree')
    a = ap.parse_args()
    model = json.load(open(os.path.join(a.model_dir, 'model.json')))
    feats = {}
    for line in open(os.path.join(a.model_dir, 'features.jsonl')):
        r = json.loads(line)
        feats[os.path.basename(r['file'])] = r['features']
    labels = {}
    for r in csv.DictReader(open(os.path.join(a.model_dir, 'labels.tsv')), delimiter='\t'):
        if r['median_mops']:
            labels[(r['base'], r['dataset'], r['variant'])] = tuple(float(r[k]) for k in
                                                                     ('median_mops', 'min_mops', 'max_mops'))
    bad = 0
    print('mode\tbase\tdataset\tchoice\texpected\tfeatures\tfeature_ms\tmops\tlabel_median\tratio\tin_label_range')
    with open(os.path.join(a.run_dir, 'runs.tsv')) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            mode, _, base = r['index'].partition('_')
            if mode not in ('gate', 'nflsw') or base not in a.bases.split():
                continue
            d = r['dataset']
            m = None
            for line in open(os.path.join(a.run_dir, r['log'])):
                m = LINE.match(line) or m
            if not m or m.group(1) != mode or m.group(2) != base:
                print('%s\t%s\t%s\tno gate line in %s' % (mode, base, d, r['log']))
                bad += 1
                continue
            choice = m.group(3)
            b = model['bases'][base]
            expected = b['final_choices'][d] if mode == 'gate' else \
                next(x['choice'] for x in b['lodo']['nflsw'] if x['dataset'] == d)
            printed = dict(kv.split('=') for kv in m.group(4).split())
            fok = all(abs(float(v) - feats[d][k]) <= 1e-5 * max(abs(feats[d][k]), 1e-300) for k, v in printed.items())
            fok = fok and set(printed) == set(feats[d])
            bad += (choice != expected) + (not fok)
            mops = float(r['throughput_ops']) / 1e6
            med, lo, hi = labels[(base, d, choice)]
            print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%s' % (
                mode, base, d, choice, expected if expected != choice else 'same', 'same' if fok else 'DIFFER',
                m.group(5), mops, med, mops / med, 'yes' if 0.95 * lo <= mops <= 1.05 * hi else 'no'))
    print('check_gate: %s' % ('OK' if bad == 0 else '%d mismatches' % bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
