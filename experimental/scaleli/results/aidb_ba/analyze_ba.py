#!/usr/bin/env python3
"""Analyse run_ba.py output. Stdlib only.  analyze_ba.py out.jsonl [more.jsonl ...] > report.md

Pre-registered statistic: per run, whole-replay throughput_ops_s (pilot osm A/A: sd 9.5%, vs 10.1% for
the median of chunks 2..16, which is reported as a sensitivity check; chunk 1 = warm-up check).
Set STAT=med in the environment to switch the primary statistic.
Contrast = difference of mean log throughput, cell vs baseline (B and B2 pooled once the A/A passes).
sigma = pooled within-(dataset, cell) sd of log throughput. 95% CI uses t with the pooled df.
A dataset where the cell's deterministic signature equals B's is a structural null and is left out
of the pooled estimate (it is reported as an extra A/A instead).
"""
import json, math, os, statistics as st, sys
STAT = os.environ.get('STAT', 'all')
from collections import defaultdict

T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}
def t975(df):
    if df <= 0: return float('nan')
    for k in sorted(T975):
        if df <= k: return T975[k]
    return 1.96

rows = []
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l); m = r.get('ba') or r.get('pilot')
        ch = r.get('throughput_chunks_ops_s') or []
        if len(ch) < 3: continue
        rest = st.median(ch[1:])
        rows.append(dict(d=m['dataset'], c=m['cell'], b=m.get('block', m.get('seed')), inst=m.get('instrument', 1),
                         thr=(rest if STAT == 'med' else r['throughput_ops_s']), thr_med=rest, warm=ch[0] / rest - 1,
                         slope=st.linear_regression(range(len(ch) - 1), [math.log(x) for x in ch[1:]]).slope,
                         ocpu=(m.get('other_cpu_start', float('nan')) + m.get('other_cpu_end', float('nan'))) / 2,
                         r=r))
base_of = lambda c: 'B'

def sig(r):
    """Deterministic signature of the built structure (memory + learnability + work per op)."""
    mb, L, w = r['memory_before'], r.get('learnability') or {}, r.get('work_counters') or {}
    ops = r['operations']
    return (mb['accounted_bytes'], L.get('flow_regions'), L.get('virtual_points'), L.get('root_model'),
            L.get('root_vp'), L.get('root_virtual'), L.get('root_flow'),
            round(w.get('root_probes', 0) / ops, 3), round(w.get('fence_probes', 0) / ops, 3))

print(f'# Before/after NFL & CSV: analysis (statistic: {"median of chunks 2-16" if STAT == "med" else "whole replay"})\n')
# ---- deterministic table (instrumented runs) --------------------------------------------------
print('## Deterministic metrics (one instrumented build per cell)\n')
print('| dataset | cell | key B/key | meta B/key | total B/key | root | coord | fence | key_at | comparisons | transform/op | lines/op | build s | smooth CPU-s | flow regions | virtual pts | root vf | root flow |')
print('|' + '---|' * 18)
det = {}
for x in sorted((x for x in rows if x['inst']), key=lambda x: (x['d'], x['c'])):
    r = x['r']; mb = r['memory_before']; n = r['source_rows']; w = r['work_counters']; o = r['operations']; L = r.get('learnability') or {}
    if (x['d'], x['c']) in det: continue
    det[(x['d'], x['c'])] = sig(r)
    pr = [w['root_probes'] / o, w['coordinate_probes'] / o, w['fence_probes'] / o, w['key_at_calls'] / o]
    print(f"| {x['d']} | {x['c']} | {mb['key_bytes']/n:.4f} | {mb['metadata_bytes']/n:.4f} | {mb['accounted_bytes']/n:.3f} | "
          + ' | '.join(f'{p:.2f}' for p in pr) + f" | {sum(pr):.2f} | {w['transform_calls']/o:.3f} | {w['cache_lines_per_operation']:.1f} | "
          f"{r['build_ns']/1e9:.1f} | {L.get('smoothing_ns', 0)/1e9:.0f} | {L.get('flow_regions', '-')} | {L.get('virtual_points', '-')} | "
          f"{L.get('root_virtual', '-')} | {L.get('root_flow', '-')} |")

# ---- correctness: checksums agree within (dataset, block) ---------------------------------------
bad = defaultdict(set)
for x in rows: bad[(x['d'], x['b'])].add(x['r']['result_checksum'])
nbad = [k for k, v in bad.items() if len(v) > 1]
print(f"\nChecksum agreement within (dataset, block): {len(bad) - len(nbad)}/{len(bad)} groups agree" + (f'; DISAGREE: {nbad}' if nbad else ''))

# ---- noise ------------------------------------------------------------------------------------
timed = [x for x in rows if x['c'] != 'COLD']
g = defaultdict(list)
for x in timed: g[(x['d'], 'B' if x['c'] == 'B2' else x['c'])].append(math.log(x['thr']))
ss = sum(sum((v - st.mean(l)) ** 2 for v in l) for l in g.values() if len(l) > 1)
df = sum(len(l) - 1 for l in g.values() if len(l) > 1)
sigma = math.sqrt(ss / df) if df else float('nan')
print(f'\n## Noise\n\nPooled within-cell sd of log throughput: **{100*sigma:.1f}%** (df {df}). '
      f'Median other-process CPU during runs: {st.median(x["ocpu"] for x in timed):.0f}%.')

def contrast(cell, base, d):
    a = [math.log(x['thr']) for x in timed if x['d'] == d and x['c'] == cell]
    b = [math.log(x['thr']) for x in timed if x['d'] == d and (x['c'] == base or (base == 'B' and x['c'] == 'B2' and cell != 'B2'))]
    if cell == 'B2': b = [math.log(x['thr']) for x in timed if x['d'] == d and x['c'] == 'B']
    if not a or not b: return None
    est = st.mean(a) - st.mean(b); se = sigma * math.sqrt(1 / len(a) + 1 / len(b))
    return est, se, len(a), len(b)

print('\n## Throughput contrasts (ratio = exp(mean log cell - mean log baseline), 95% CI)\n')
print('| contrast | ' + ' | '.join(sorted({x["d"] for x in timed})) + ' | pooled (non-null datasets) |')
print('|' + '---|' * (2 + len({x["d"] for x in timed})))
ds = sorted({x['d'] for x in timed})
for cell, base in [('B2', 'B'), ('N', 'B'), ('Nf', 'B'), ('C', 'B'), ('NC', 'C'), ('NC', 'B'), ('SV', 'B'), ('RAW', 'B'), ('RC', 'SV'), ('B', 'B0')]:
    cells_txt = []; pool = []
    for d in ds:
        c = contrast(cell, base, d)
        if not c: cells_txt.append('-'); continue
        e, se, na, nb = c; h = t975(df) * se
        null = (d, cell) in det and (d, base) in det and det[(d, cell)] == det[(d, base)] and cell != 'B2'
        cells_txt.append(f"{math.exp(e):.3f} [{math.exp(e-h):.3f},{math.exp(e+h):.3f}] n={na}/{nb}" + (' NULL' if null else ''))
        if not null: pool.append((e, se))
    if pool:
        pe = st.mean(e for e, _ in pool); pse = math.sqrt(sum(s * s for _, s in pool)) / len(pool); h = t975(df) * pse
        ptxt = f'{math.exp(pe):.3f} [{math.exp(pe-h):.3f},{math.exp(pe+h):.3f}] k={len(pool)}'
    else: ptxt = '-'
    print(f'| {cell} vs {base} | ' + ' | '.join(cells_txt) + f' | {ptxt} |')

# interaction (NC - C) - (N - B), per dataset
print('\nInteraction (NC-C)-(N-B), log units:')
for d in ds:
    a, b, c_, e = (contrast('NC', 'C', d), contrast('N', 'B', d), None, None)
    if a and b: print(f'  {d}: {a[0]-b[0]:+.3f} +/- {t975(df)*math.sqrt(a[1]**2+b[1]**2):.3f}')

# ---- warm-up ----------------------------------------------------------------------------------
w = [x['warm'] for x in timed]
print(f'\n## Warm-up\n\nFirst chunk / median(rest) - 1, pooled over {len(w)} warm runs: {100*st.mean(w):+.1f}% '
      f'(SE {100*st.stdev(w)/math.sqrt(len(w)):.1f}%; pass if within +/-3%).')
for x in rows:
    if x['c'] == 'COLD': print(f"  cold {x['d']}: first chunk {100*x['warm']:+.1f}% vs rest")
sl = defaultdict(list)
for x in timed: sl[x['c']].append(x['slope'])
print('Mean log-throughput slope per chunk (chunks 2-16), by cell (thermal check, C should match B): '
      + ', '.join(f'{c} {1000*st.mean(v):+.2f}e-3' for c, v in sorted(sl.items())))
