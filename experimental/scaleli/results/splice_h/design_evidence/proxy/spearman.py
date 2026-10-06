#!/usr/bin/env python3
"""Mean Spearman correlation, over the ten datasets, of each error proxy in surr.json with the counted lines (Lg)
of the eight models of each dataset (average ranks for ties; a proxy that is constant on a dataset has no ranking
there and is left out of that mean). Standard library only.

    python3 spearman.py [surr.json]
"""
import json
import math
import os
import sys


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra))
    sb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surr.json')
    with open(path) as f:
        d = json.load(f)
    names = {'log2': 'log2 error', 'mse': 'MSE', 'CDmax': 'LIPP conflict degree (max)',
             'CDtail99': 'NFL tail conflict (99th pct)', 'slotcoll': 'slot collisions', 'lineexcess': 'line excess'}
    for k in ('log2', 'mse', 'CDmax', 'CDtail99', 'slotcoll', 'lineexcess'):
        per = {}
        for ds, models in d.items():
            m = sorted(models)
            per[ds] = spearman([models[x][k] for x in m], [models[x]['Lg'] for x in m])
        ok = [v for v in per.values() if v is not None]
        print('%-30s mean %+.3f over %d datasets; %s' % (names[k], sum(ok) / len(ok) if ok else float('nan'), len(ok),
              ', '.join('%s %s' % (ds, 'n/a' if v is None else '%+.2f' % v) for ds, v in sorted(per.items()))))


if __name__ == '__main__':
    main()
