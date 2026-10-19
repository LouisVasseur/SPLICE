#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Median throughput (Mops/s) and memory (MB) per dataset x index from run_baselines.sh's runs.tsv.

    python3 scripts/summarize.py OUT_DIR       writes OUT_DIR/summary.tsv and prints the table

The spread column is (max - min) / median over repeats, in percent. Python 3.6, stdlib only.
"""
import csv
import statistics
import sys
from collections import OrderedDict

out = sys.argv[1]
cells = OrderedDict()
with open(out + '/runs.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        cells.setdefault((r['dataset'], r['index']), []).append((float(r['throughput_ops']), int(r['memory_bytes'])))
if not cells:
    sys.exit('summarize.py: no successful runs in %s/runs.tsv' % out)
datasets = list(OrderedDict.fromkeys(d for d, _ in cells))
indexes = list(OrderedDict.fromkeys(i for _, i in cells))
rows = []
for (d, i), v in cells.items():
    thr = [t for t, _ in v]
    med = statistics.median(thr)
    rows.append((d, i, len(v), med / 1e6, 100 * (max(thr) - min(thr)) / med, statistics.median(m for _, m in v) / 1e6))
with open(out + '/summary.tsv', 'w') as f:
    f.write('dataset\tindex\trepeats\tmedian_mops\tspread_pct\tmemory_mb\n')
    for r in rows:
        f.write('%s\t%s\t%d\t%.3f\t%.1f\t%.1f\n' % r)
by = {(r[0], r[1]): r for r in rows}
w = max(len(d) for d in datasets)
print('median Mops/s (spread %% over repeats); %s' % out)
print(' ' * w + ''.join('%16s' % i for i in indexes))
for d in datasets:
    print(d.ljust(w) + ''.join('%9.2f (%3.0f%%)' % (by[d, i][3], by[d, i][4]) if (d, i) in by else '%16s' % '-'
                               for i in indexes))
print('memory MB (GRE self-reported; 0 = not reported by the index: GRE\'s B+tree, and the transform indexes on it)')
for d in datasets:
    print(d.ljust(w) + ''.join('%16.1f' % by[d, i][5] if (d, i) in by else '%16s' % '-' for i in indexes))
