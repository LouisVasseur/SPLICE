import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json, math, statistics, random, collections
D = json.load(open('coststory.json'))
noflow, noflow_all, flow, sham = D['noflow'], D['noflow_all'], D['flow'], D['sham']
cell = D['cell']
fl = [p for p in flow if p['vA'] in ('packed_rank_root_fusion','packed_rank_vp10_root_fusion')]
# 1. joint OLS: dns = a - b*S + F*T over group A (raw/vf4, T=0) + group B (T~1), dataset-cluster bootstrap
def ols(ps):
    # columns: 1, -S, dT
    X = [[1.0, -p['S'], p['dT']] for p in ps]; y = [p['dns'] for p in ps]
    n=3; XtX=[[sum(X[i][r]*X[i][c] for i in range(len(X))) for c in range(n)] for r in range(n)]; Xty=[sum(X[i][r]*y[i] for i in range(len(X))) for r in range(n)]
    # solve 3x3 by Gaussian elimination
    A=[row[:]+[Xty[r]] for r,row in enumerate(XtX)]
    for c in range(n):
        piv=max(range(c,n),key=lambda r:abs(A[r][c])); A[c],A[piv]=A[piv],A[c]
        for r in range(n):
            if r!=c:
                f=A[r][c]/A[c][c]; A[r]=[a-f*b for a,b in zip(A[r],A[c])]
    return [A[r][n]/A[r][r] for r in range(n)]
allp = noflow + fl
a,b,F = ols(allp)
print(f"JOINT OLS on {len(allp)} pairs (groupA raw/vf4 + groupB fusion pairs): dns = {a:+.1f} - {b:.2f}*S + {F:.1f}*T  => flow cost {F:.1f} ns/call = {F/b:.1f} probe-equivalents")
rng=random.Random(5); ds=sorted({p['dataset'] for p in allp}); byd=collections.defaultdict(list)
for p in allp: byd[p['dataset']].append(p)
B=[]
for _ in range(4000):
    s=[q for d in rng.choices(ds,k=len(ds)) for q in byd[d]]
    try: B.append(ols(s))
    except ZeroDivisionError: pass
def ci(xs): xs=sorted(xs); return xs[int(.025*len(xs))], xs[int(.975*len(xs))-1]
print("   dataset-cluster bootstrap 95%: a", ci([x[0] for x in B]), " b", ci([x[1] for x in B]), " F", ci([x[2] for x in B]), " F/b", ci([x[2]/x[1] for x in B if x[1]>0.3]))
allp2 = noflow_all + flow
a2,b2,F2 = ols(allp2); print(f"JOINT OLS on all {len(allp2)} model-root pairs (incl. region-flow ones): dns = {a2:+.1f} - {b2:.2f}*S + {F2:.1f}*T => {F2/b2:.1f} probe-eq")
# 2. slot-table size effect within group A: dns = a - b*S + c*slot_bytes/1000
def ols2(ps):
    X=[[1.0,-p['S'],p['slotk']] for p in ps]; y=[p['dns'] for p in ps]; n=3
    XtX=[[sum(X[i][r]*X[i][c] for i in range(len(X))) for c in range(n)] for r in range(n)]; Xty=[sum(X[i][r]*y[i] for i in range(len(X))) for r in range(n)]
    A=[row[:]+[Xty[r]] for r,row in enumerate(XtX)]
    for c in range(n):
        piv=max(range(c,n),key=lambda r:abs(A[r][c])); A[c],A[piv]=A[piv],A[c]
        for r in range(n):
            if r!=c: f=A[r][c]/A[c][c]; A[r]=[x-f*z for x,z in zip(A[r],A[c])]
    return [A[r][n]/A[r][r] for r in range(n)]
for p in noflow_all: p['slotk'] = cell[f"{p['dataset']}|{p['vA']}"]['root_virtual']*4/1000.0 if cell[f"{p['dataset']}|{p['vA']}"]['chosen'].endswith('fences') else 0.0
vf = [p for p in noflow_all if p['vA'] in ('packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_vp10_root_fusion')]
a3,b3,c3 = ols2(vf); print(f"SLOT-TABLE effect among raw+fences roots (n={len(vf)}): dns = {a3:+.1f} - {b3:.2f}*S + {c3:+.2f} ns per KB of slot table (tables 0-7.8 KB); flow-root tables are 5.4-6.8 KB (virt 1339-1706)")
rng=random.Random(9); ds=sorted({p['dataset'] for p in vf}); byd=collections.defaultdict(list)
for p in vf: byd[p['dataset']].append(p)
C=[]
for _ in range(4000):
    s=[q for d in rng.choices(ds,k=len(ds)) for q in byd[d]]
    try: C.append(ols2(s)[2])
    except ZeroDivisionError: pass
print("   95% bootstrap for the per-KB coefficient:", ci(C), "=> at 6 KB:", tuple(round(6*x,1) for x in ci(C)), "ns")
# 3. reproduce the notes' numbers: median over seeds, b=3.3
print("\nNOTES-STYLE (median dns over seeds, b = 3.3 ns/probe): per-dataset flow cost F = med(dns) + 3.3*S for root_fusion vs packed_rank")
byd=collections.defaultdict(list)
for p in flow:
    if p['vA']=='packed_rank_root_fusion': byd[p['dataset']].append(p)
vals=[]
for d,qs in sorted(byd.items()):
    m=statistics.median(q['dns'] for q in qs); S=qs[0]['S']; vals.append((d, m, S, m+3.3*S))
    print(f"   {d:<16} med dns={m:+6.1f} (seeds {', '.join(f'{q['dns']:+.0f}' for q in qs)}) S={S:.2f} F={m+3.3*S:5.1f} ns = {(m+3.3*S)/3.3:4.1f} probe-eq")
# per-dataset ns/probe with medians (vf4 only)
byd=collections.defaultdict(list)
for p in noflow:
    if p['vA']=='packed_rank_root_vf4': byd[p['dataset']].append(p)
med_b=[]
for d,qs in sorted(byd.items()):
    m=statistics.median(q['dns'] for q in qs); S=qs[0]['S']
    if S>0.5: med_b.append((d,-m/S))
print("   vf4 per-dataset ns/probe with median dns:", ", ".join(f"{d} {v:+.1f}" for d,v in med_b), "=> median", round(statistics.median(v for _,v in med_b),2))
# 4. vf4 paired gmean speedups (calibrated-selector proxy) and sham gmeans
import itertools
P=REPO + '/experimental/scaleli/results/aidb_final/sweep/results.jsonl'
rows=[json.loads(l) for l in open(P) if l.strip()]; idx={(r['dataset'],r['variant'],r['seed']):r for r in rows}
print("\nvf4 vs packed_rank paired gmean (min,max per-seed ratio) and chosen root:")
for d in sorted({r['dataset'] for r in rows}):
    rs=[(idx[(d,'packed_rank_root_vf4',s)]['throughput_ops_s']/idx[(d,'packed_rank',s)]['throughput_ops_s']) for s in (11,29,47)]
    g=math.exp(statistics.mean(map(math.log,rs))); print(f"   {d:<16} {g:.3f} [{min(rs):.3f},{max(rs):.3f}] {cell[d+'|packed_rank_root_vf4']['chosen']} S={8.956-cell[d+'|packed_rank_root_vf4']['root']:.2f}")
# 5. calibrated charge and what it implies
print(f"\nCalibrated charge = F/b: joint OLS {F/b:.1f}; pooled (66.4 ns / 4.07) = {66.4/4.07:.1f}; range over b in [2.6,5.5] and F in [56,77]: {56/5.5:.1f} .. {77/2.6:.1f}")
print("Max root saving possible with 489 regions: 8.96 - 2.08 (best observed learned root) = 6.88 probes = at 4.07 ns -> 28 ns; the flow costs ~66 ns: it cannot pay at the root of a 2M-key / 489-region index under any selector.")
