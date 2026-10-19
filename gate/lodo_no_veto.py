#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""How much of train_gate.py's LODO result is the veto: the same LODO with the veto taken out of the models.

    python3 gate/lodo_no_veto.py LABELS_DIR FEATURES.jsonl [--bases "lipp btree"]

train_gate.py applies gate.h's raw veto outside the model: a leaf that picks raw where raw cannot be built gets its
fallback. Here a model's raw pick is built as chosen, and on a dataset without a raw label (vetoed, or not run) it
scores 100% regret, so a model must find the vetoed datasets from the features alone. Prints the mean and worst LODO
regret per base and model type, and the datasets with non-zero regret. Python 3.6+, stdlib only.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_gate as tg  # noqa: E402


def realize(self, leaf):  # pick as chosen; a missing non-raw label falls back as in train_gate.py
    pick, fb = leaf
    if pick == 'raw' or pick in self.thr:
        return pick
    return fb if fb in self.thr else 'pw512'


def regret(self, v):
    return 100.0 if v not in self.thr else 100.0 * (1.0 - self.thr[v] / self.thr[self.oracle])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('labels_dir')
    ap.add_argument('features')
    ap.add_argument('--bases', default='lipp btree')
    a = ap.parse_args()
    tg.Dataset.realize, tg.Dataset.regret = realize, regret
    cells, feats = tg.read_labels(a.labels_dir), tg.read_features(a.features)
    names = list(dict.fromkeys(d for d, _ in cells))
    print('base\tmodel\tmean_regret_pct\tworst_regret_pct\tnonzero')
    for base in a.bases.split():
        ds = []
        for name in names:
            thr, spread = {}, {}
            for v in tg.VARIANTS:
                t = cells.get((name, tg.index_name(base, v)))
                if t:
                    thr[v] = statistics.median(t)
                    spread[v] = 100 * (max(t) - min(t)) / thr[v]
            ds.append(tg.Dataset(name, base, feats[name], thr, spread, 'raw' not in thr))
        for m, depth in tg.MODELS.items():
            out = []
            for i, d in enumerate(ds):
                _, tree = tg.fit(ds[:i] + ds[i + 1:], depth)
                v = d.realize(tg.predict(tree, d.f))
                out.append((d.name, v, d.regret(v)))
            r = [x[2] for x in out]
            print('%s\t%s\t%.2f\t%.2f\t%s' % (base, m, statistics.mean(r), max(r),
                                              ' '.join('%s=%s(%.1f%%)' % x for x in out if x[2] > 0) or '-'))


if __name__ == '__main__':
    main()
