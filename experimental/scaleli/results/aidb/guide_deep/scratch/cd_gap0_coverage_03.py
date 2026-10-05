"""Guide section 03 sensitivity check: coverage of the 25 Table-2 metrics when CD is
recomputed with L = N (gap 0, 'keys per rank') instead of LIPP's L = 2N (gap 1), using
the RMSE/ME/PLA values of results/aidb/hardness.json (full scope) and the CD values of
scratch/fmcd_gap_03.full.jsonl. Standard library only."""
import json, itertools, os
S = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(S, '..', '..'))
h = json.load(open(os.path.join(R, 'hardness.json')))
rows = [json.loads(l) for l in open(os.path.join(S, 'fmcd_gap_03.full.jsonl')) if l.strip()]
cd = {}
for r in rows:
    name = os.path.basename(r['file']).split('.')[0]
    cd.setdefault(name, {})[r['gap']] = r
names = sorted(h)
paper_cov = {'RMSE':1.00,'ME':1.00,'CD':1.00,'PLA-32':1.00,'PLA-4096':1.00,'PLA-32·PLA-4096':0.47,'RMSE·ME':0.82,'RMSE·CD':0.47,'RMSE·PLA-32':0.60,'RMSE·PLA-4096':0.42,'ME·CD':0.56,'ME·PLA-32':0.60,'ME·PLA-4096':0.42,'CD·PLA-32':0.60,'CD·PLA-4096':0.42,'RMSE·ME·CD':0.42,'RMSE·ME·PLA-32':0.51,'RMSE·ME·PLA-4096':0.33,'RMSE·CD·PLA-32':0.33,'RMSE·CD·PLA-4096':0.16,'RMSE·PLA-32·PLA-4096':0.24,'ME·CD·PLA-32':0.38,'ME·CD·PLA-4096':0.20,'ME·PLA-32·PLA-4096':0.24,'CD·PLA-32·PLA-4096':0.24}
def table(gap):
    t = {}
    for n in names:
        b = h[n]['full']
        t[n] = {'RMSE': b['rmse'], 'ME': b['max_error'], 'CD': cd[n][gap]['CD'] if gap in cd.get(n, {}) else b['conflict_degree'], 'PLA-32': b['pla_32'], 'PLA-4096': b['pla_4096']}
    return t
def harder(a, b, dims): return all(a[k] >= b[k] for k in dims) and any(a[k] > b[k] for k in dims)
def cov(t, dims):
    C = U = 0; inc = []
    for a, b in itertools.combinations(names, 2):
        if harder(t[a], t[b], dims) or harder(t[b], t[a], dims): C += 1
        else: U += 1; inc.append((a, b))
    return (C - U) / (C + U), C, U, inc
scal = ['RMSE', 'ME', 'CD', 'PLA-32', 'PLA-4096']
metrics = [[s] for s in scal] + [list(c) for c in itertools.combinations(scal, 2)] + [list(c) for c in itertools.combinations(scal, 3)]
print('CD per dataset: gap1 (L=2N) vs gap0 (L=N), D and CD')
for n in names:
    g1 = cd[n].get(1); g0 = cd[n].get(0)
    print(f"  {n:8s} L=2N: D={g1['D'] if g1 else '?'} CD={g1['CD'] if g1 else '?'}   L=N: D={g0['D'] if g0 else '?'} CD={g0['CD'] if g0 else '?'}   ratio={g0['CD']/g1['CD'] if g0 and g1 else float('nan'):.2f}")
for gap in (1, 0):
    t = table(gap); print(f'\n=== coverage with CD at gap {gap} (L = {"2N" if gap else "N"}) ===')
    mism = 0
    for dims in metrics:
        name = '·'.join(dims); c, C, U, inc = cov(t, dims)
        flag = '' if abs(c - paper_cov[name]) < 0.005 else f'   != paper {paper_cov[name]:.2f}'
        if flag: mism += 1
        if 'CD' in dims: print(f'  {name:24s} Cov={c:.3f} (|C|={C}, |U|={U}){flag}')
    print(f'  CD-containing metrics disagreeing with Table 2: {mism}')
t1, t0 = table(1), table(0)
print('\npairs ordered differently by CD at L=2N vs L=N:')
for a, b in itertools.combinations(names, 2):
    s1 = (t1[a]['CD'] > t1[b]['CD']) - (t1[a]['CD'] < t1[b]['CD']); s0 = (t0[a]['CD'] > t0[b]['CD']) - (t0[a]['CD'] < t0[b]['CD'])
    if s1 != s0: print(f'  ({a}, {b}): L=2N {t1[a]["CD"]} vs {t1[b]["CD"]}; L=N {t0[a]["CD"]} vs {t0[b]["CD"]}')
print('\nCD-scalar orderings: L=2N', sorted(names, key=lambda n: t1[n]['CD']), '\n                     L=N ', sorted(names, key=lambda n: t0[n]['CD']))
