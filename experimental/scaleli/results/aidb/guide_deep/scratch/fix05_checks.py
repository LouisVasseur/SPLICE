#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Recomputations behind the section-05 fixes (stdlib only). Run from S:
   python3 results/aidb/guide_deep/scratch/fix05_checks.py
Every number quoted in the fixed section 05 that is not a direct file value comes from here."""
import json, math, csv
from collections import Counter

S = REPO + '/experimental/scaleli/'

# 1. Comparison counts of the lower-bound loop `while(lo<hi){m=lo+(hi-lo)/2; if(x<=y) lo=m+1; else hi=m;}`
#    (index.hpp:123-127 over #blocks boundaries, index.hpp:160 over count keys)
def loop_cost(n, ans):
    lo, hi, c = 0, n, 0
    while lo < hi:
        m = lo + (hi - lo) // 2; c += 1
        if m < ans: lo = m + 1
        else: hi = m
    return c
for n in (32, 9, 128):
    cs = Counter(loop_cost(n, a) for a in range(n + 1))
    print(f'lower-bound loop over n={n}: outcomes={dict(sorted(cs.items()))} min={min(cs)} max={max(cs)}')

# 2. Flow forward pass + per-key NLL term (train_flow.py:31-40, 48-55) on results/aidb/flows/fb_2D2H2L.txt
W0 = [[0.0058556031582764, 0.0058592806847908], [0.0, 0.0]]
W1 = [[1.2798795760698392, 0.8812895293543010], [1.4785700705830542, 1.0331745748154413]]
mean, var, barrier = 15162980.0, 1207648189.703125, 1e-3
def forward(key):
    x = (key - mean) / var; f = [x, x - math.floor(x)]
    u = [f[0] * W0[0][h] + f[1] * W0[1][h] for h in range(2)]
    h = [math.tanh(t) for t in u]; v = [sum(W1[k]) for k in range(2)]
    a = [W0[0][k] + W0[1][k] for k in range(2)]; s = [1 - t * t for t in h]
    z = sum(h[k] * v[k] for k in range(2)); d = sum(s[k] * a[k] * v[k] for k in range(2))
    return x, u, h, v, a, s, z, d
for key in (38737396981, 77308811965):
    x, u, h, v, a, s, z, d = forward(key)
    t1, t2, t3 = 0.5 * z * z, -math.log(d), barrier / d
    print(f'key {key}: x={x:.12f} u={u} v={v} a={a} s={s} z={z:.10f} dz/dx={d:.9f} 1/(dz/dx)={1/d:.3f}')
    print(f'   NLL term = {t1:.6f} + {t2:.6f} + {t3:.6f} = {t1+t2+t3:.6f}')
print('best_nll (fb_training.json):', json.load(open(S + 'results/aidb/flows/fb_training.json'))['best_nll'])

# 3. O(1) sums update of smoothing.hpp:64-70 on features [1,2,3,10,11,12], gap i=2, x_v=6.5
n, Sx, Sxx, Sy, Syy, Sxy = 6, 39, 379, 15, 55, 142
cnt = n - (2 + 1); suf_y = 3 + 4 + 5; suf_x = 10 + 11 + 12
Sy2, Syy2, Sxy2 = Sy + cnt, Syy + 2 * suf_y + cnt, Sxy + suf_x
print('after shift: S_y', Sy2, 'S_yy', Syy2, 'S_xy', Sxy2)
n2, Sx2, Sxx2, Sy3, Syy3, Sxy3 = n + 1, Sx + 6.5, Sxx + 6.5 ** 2, Sy2 + 3, Syy2 + 9, Sxy2 + 6.5 * 3
cxx, cxy, cyy = Sxx2 - Sx2 ** 2 / n2, Sxy3 - Sx2 * Sy3 / n2, Syy3 - Sy3 ** 2 / n2
print('after add:', n2, Sx2, Sxx2, Sy3, Syy3, Sxy3, 'centred', cxx, cxy, cyy, 'SSE', cyy - cxy ** 2 / cxx)
xs = [1, 2, 3, 6.5, 10, 11, 12]; ys = list(range(7)); mx = sum(xs) / 7; my = sum(ys) / 7
w = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs); b = my - w * mx
print('brute-force refit SSE', sum((y - w * x - b) ** 2 for x, y in zip(xs, ys)))

# 4. Memory per key, fb packed_rank vs packed_rank_vp10 (results/aidb/sweep/results.jsonl, seed 11)
rows = [json.loads(l) for l in open(S + 'results/aidb/sweep/results.jsonl') if l.strip()]
for r in rows:
    if r['dataset'] == 'fb' and r['seed'] == 11 and r['variant'] in ('packed_rank', 'packed_rank_vp10'):
        m = r['memory_before']; print(r['variant'], m, {k: m[k] / 2e6 for k in ('key_bytes', 'value_bytes', 'metadata_bytes', 'accounted_bytes')})
# 5. flow_regions and tail conflicts per dataset, packed_rank_flow seed 11
for r in rows:
    if r['variant'] == 'packed_rank_flow' and r['seed'] == 11:
        l = r['learnability']; print(r['dataset'], 'flow_regions', l['flow_regions'], f"{100*l['flow_regions']/l['regions']:.1f}%", 'D99 raw/flow', round(l['tail_conflicts_raw_mean'], 2), round(l['tail_conflicts_flow_mean'], 2))
# 6. smoothing_ns fusion_auto vs vp10
for r in rows:
    if r['seed'] == 11 and r['dataset'] in ('fb', 'osm') and r['variant'] in ('packed_rank_fusion_auto', 'packed_rank_vp10'):
        print(r['dataset'], r['variant'], 'smoothing_s', r['learnability']['smoothing_ns'] / 1e9)
# 7. learnability read_only ratios
by = {}
for r in csv.DictReader(open(S + 'results/learnability/summary.csv')):
    if r['profile'] == 'read_only': by[(r['dataset'], r['variant'])] = (float(r['throughput_ops_s_median']), float(r['paired_throughput_speedup']))
for d in sorted({k[0] for k in by}):
    a, b = by[(d, 'packed_rank_vp10')], by[(d, 'packed_rank_vp10_relearn')]
    print(d, 'paired speedup vp10', round(a[1], 4), 'relearn', round(b[1], 4), 'relearn/vp10 throughput', round(b[0] / a[0], 4))
# 8. tail conflict degree readings on the worked example
counts = sorted([2, 4, 3, 6, 1, 1, 1, 1, 1]); m = len(counts)
print('counts', counts, 'code idx', math.ceil(0.99 * m) - 1, '->', counts[math.ceil(0.99 * m) - 1] - 1,
      '| paper t=INT(m*0.99)=', int(m * 0.99), 'ascending t-th (1-based) ->', counts[int(m * 0.99) - 1] - 1,
      'descending t-th largest ->', sorted(counts, reverse=True)[int(m * 0.99) - 1] - 1)
