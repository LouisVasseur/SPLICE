#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Cost-story check (C4-C6) on results/aidb_final/sweep/results.jsonl. stdlib only."""
import json, math, statistics, random, collections, sys
P = REPO + '/experimental/scaleli/results/aidb_final/sweep/results.jsonl'
rows = [json.loads(l) for l in open(P) if l.strip()]
SEEDS = {11, 29, 47}
rows = [r for r in rows if r['seed'] in SEEDS and r['variant'] != 'packed_rank_fusion_auto']
CTR = ['root_probes','fence_probes','coordinate_probes','delta_probes','key_at_calls','decoded_keys','codec_bytes_examined','block_routes','correction_distance','transform_calls']
def per(r, k): return r['work_counters'][k] / r['operations']
def ns(r): return r['elapsed_ns'] / r['operations']
def chosen(r):
    l = r.get('learnability') or {}
    if r['variant'] == 'sorted_vector': return 'n/a'
    if not l.get('root_model'): return 'binary'
    return ('flow+' if l.get('root_flow') else 'raw+') + ('fences' if l.get('root_vp') else 'ranks')
idx = {(r['dataset'], r['variant'], r['seed']): r for r in rows}
datasets = sorted({r['dataset'] for r in rows}); variants = sorted({r['variant'] for r in rows})
out = {}
# sanity: throughput vs elapsed
mx = max(abs(1e9 / r['throughput_ops_s'] - ns(r)) for r in rows)
print(f"max |1e9/throughput - elapsed/ops| = {mx:.4f} ns  (n={len(rows)} rows on seeds 11/29/47)")

# ---------- 1. transform_calls per lookup by variant and root choice ----------
print("\n== transform_calls per lookup (T) by variant, split by root_flow ==")
T_summary = collections.defaultdict(list)
for r in rows:
    l = r.get('learnability') or {}
    T_summary[(r['variant'], bool(l.get('root_flow')))].append((per(r,'transform_calls'), l.get('flow_regions',0)/max(1,l.get('regions',1))))
for k in sorted(T_summary):
    v = T_summary[k]; ts = [x for x,_ in v]; fr = [y for _,y in v]
    print(f"  {k[0]:<30} root_flow={str(k[1]):<5} n={len(v):>3} T min/med/max = {min(ts):.4f}/{statistics.median(ts):.4f}/{max(ts):.4f}   flow_regions/regions min/max = {min(fr):.4f}/{max(fr):.4f}")
# per-row check T ~= root_flow + flow_regions/regions
dev = []
for r in rows:
    l = r.get('learnability') or {}
    if r['variant']=='sorted_vector': continue
    pred = (1.0 if l.get('root_flow') else 0.0) + l.get('flow_regions',0)/max(1,l.get('regions',1))
    dev.append((abs(per(r,'transform_calls')-pred), r['dataset'], r['variant'], r['seed'], per(r,'transform_calls'), pred))
dev.sort(reverse=True)
print(f"  max |T - (root_flow + flow_regions/regions)| = {dev[0][0]:.4f} at {dev[0][1:]}")

# ---------- 2. counter identity vs packed_rank ----------
print("\n== counters of the root variants vs packed_rank on the same (dataset, seed): max abs diff per lookup ==")
for v in ['packed_rank_root_raw','packed_rank_root_flow','packed_rank_root_vf4','packed_rank_root_fusion']:
    diffs = collections.defaultdict(list)
    for d in datasets:
        for s in sorted(SEEDS):
            a = idx.get((d,v,s)); b = idx.get((d,'packed_rank',s))
            if not a or not b: continue
            for k in CTR: diffs[k].append(per(a,k)-per(b,k))
    print(f"  {v:<28} " + "  ".join(f"{k}:[{min(diffs[k]):+.3f},{max(diffs[k]):+.3f}]" for k in ['fence_probes','coordinate_probes','key_at_calls','decoded_keys','codec_bytes_examined','correction_distance']))
    print(f"  {'':<28} root_probes:[{min(diffs['root_probes']):+.3f},{max(diffs['root_probes']):+.3f}]  transform_calls:[{min(diffs['transform_calls']):+.3f},{max(diffs['transform_calls']):+.3f}]")
print("  same for packed_rank_vp10_root_fusion vs packed_rank_vp10:")
diffs = collections.defaultdict(list)
for d in datasets:
    for s in sorted(SEEDS):
        a = idx.get((d,'packed_rank_vp10_root_fusion',s)); b = idx.get((d,'packed_rank_vp10',s))
        if not a or not b: continue
        for k in CTR: diffs[k].append(per(a,k)-per(b,k))
print("  " + "  ".join(f"{k}:[{min(diffs[k]):+.3f},{max(diffs[k]):+.3f}]" for k in ['root_probes','fence_probes','coordinate_probes','key_at_calls','decoded_keys','codec_bytes_examined','transform_calls']))

# ---------- 3. memory ----------
print("\n== metadata_bytes: variant minus packed_rank (bytes), root_virtual, slot-table bytes 4*(regions+virt) ==")
memtab = {}
for d in datasets:
    b = idx.get((d,'packed_rank',11))
    line = [f"{d:<16} packed meta={b['memory_before']['metadata_bytes']:>9}"]
    for v in ['packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_vp10','packed_rank_vp10_root_fusion']:
        a = idx.get((d,v,11))
        if not a: continue
        l = a['learnability']; dm = a['memory_before']['metadata_bytes']-b['memory_before']['metadata_bytes']
        slot = 4*(l['regions']+l['root_virtual']) if l.get('root_model') and l.get('root_vp') else 0
        memtab[(d,v)] = dict(dmeta=dm, root_virtual=l['root_virtual'], slot_bytes=slot, chosen=chosen(a), meta=a['memory_before']['metadata_bytes'], accounted=a['memory_before']['accounted_bytes'])
        line.append(f"{v.replace('packed_rank_','')}: d={dm:+7d} virt={l['root_virtual']:>5} slot={slot:>5} {chosen(a):<11}")
    print("  " + " | ".join(line))

# ---------- 4. per-cell table ----------
print("\n== per (dataset, variant): chosen root, T, probes/lookup (root, fence, coord, total), ns/lookup by seed 11/29/47 ==")
cell = {}
for d in datasets:
    for v in variants:
        rs = [idx[(d,v,s)] for s in sorted(SEEDS) if (d,v,s) in idx]
        if not rs: continue
        r0 = rs[0]
        cell[(d,v)] = dict(chosen=chosen(r0), T=statistics.mean(per(r,'transform_calls') for r in rs),
            root=statistics.mean(per(r,'root_probes') for r in rs), fence=statistics.mean(per(r,'fence_probes') for r in rs),
            coord=statistics.mean(per(r,'coordinate_probes') for r in rs),
            ns={r['seed']: ns(r) for r in rs}, est=[(r0.get('learnability') or {}).get(k,0) for k in ('root_probes_binary','root_probes_raw','root_probes_flow','root_probes_vp_raw','root_probes_vp_flow')],
            root_virtual=(r0.get('learnability') or {}).get('root_virtual',0), meta=r0['memory_before']['metadata_bytes'],
            transform_ns_per_key=(r0.get('learnability') or {}).get('transform_ns',0)/max(1,(r0.get('learnability') or {}).get('keys',1)))
        c = cell[(d,v)]
        print(f"  {d:<16}{v:<30}{c['chosen']:<12}T={c['T']:.3f} root={c['root']:5.2f} fence={c['fence']:5.2f} coord={c['coord']:5.2f} tot={c['root']+c['fence']+c['coord']:5.2f}  ns=" + " ".join(f"{c['ns'].get(s,float('nan')):6.1f}" for s in sorted(SEEDS)) + f"  est b/r/f/vr/vf={'/'.join(f'{x:.2f}' for x in c['est'])}")

# ---------- 5. paired deltas ----------
def pairs(vA, vB, cond):
    res = []
    for d in datasets:
        for s in sorted(SEEDS):
            a = idx.get((d,vA,s)); b = idx.get((d,vB,s))
            if not a or not b: continue
            la = a.get('learnability') or {}
            if not cond(la): continue
            res.append(dict(dataset=d, seed=s, vA=vA, vB=vB, dns=ns(a)-ns(b), ratio=a['throughput_ops_s']/b['throughput_ops_s'],
                            S=per(b,'root_probes')-per(a,'root_probes'), Stot=sum(per(b,k) for k in ['root_probes','fence_probes','coordinate_probes','delta_probes'])-sum(per(a,k) for k in ['root_probes','fence_probes','coordinate_probes','delta_probes']),
                            dT=per(a,'transform_calls')-per(b,'transform_calls'), ns_base=ns(b), chosen=chosen(a),
                            dmeta=a['memory_before']['metadata_bytes']-b['memory_before']['metadata_bytes']))
    return res
sham = pairs('packed_rank_root_raw','packed_rank', lambda l: not l.get('root_model')) + pairs('packed_rank_root_vf4','packed_rank', lambda l: not l.get('root_model'))
shamflow = pairs('packed_rank_root_flow','packed_rank', lambda l: not l.get('root_model'))
noflow = pairs('packed_rank_root_raw','packed_rank', lambda l: l.get('root_model') and not l.get('root_flow')) + pairs('packed_rank_root_vf4','packed_rank', lambda l: l.get('root_model') and not l.get('root_flow'))
noflow_all = noflow + pairs('packed_rank_root_flow','packed_rank', lambda l: l.get('root_model') and not l.get('root_flow')) + pairs('packed_rank_root_fusion','packed_rank', lambda l: l.get('root_model') and not l.get('root_flow')) + pairs('packed_rank_vp10_root_fusion','packed_rank_vp10', lambda l: l.get('root_model') and not l.get('root_flow'))
flow = pairs('packed_rank_root_fusion','packed_rank', lambda l: l.get('root_flow')) + pairs('packed_rank_root_flow','packed_rank', lambda l: l.get('root_flow')) + pairs('packed_rank_vp10_root_fusion','packed_rank_vp10', lambda l: l.get('root_flow'))

def summ(xs):
    xs = sorted(xs); n = len(xs)
    return f"n={n} mean={statistics.mean(xs):+.1f} sd={statistics.pstdev(xs):.1f} min={xs[0]:+.1f} p25={xs[n//4]:+.1f} med={statistics.median(xs):+.1f} p75={xs[3*n//4]:+.1f} max={xs[-1]:+.1f}"
print("\n== NOISE FLOOR: sham pairs (root variant fell back to binary => identical lookup path to packed_rank), delta ns/lookup and ratio ==")
print("  root_raw/root_vf4 fallback vs packed_rank: dns " + summ([p['dns'] for p in sham]))
print("      ratio: " + summ([100*(p['ratio']-1) for p in sham]) + " (percent)")
print("  root_flow fallback (region-level flow only, T<=0.09) vs packed_rank: dns " + summ([p['dns'] for p in shamflow]))
print("      ratio: " + summ([100*(p['ratio']-1) for p in shamflow]) + " (percent)")
for p in sorted(sham+shamflow, key=lambda p:(p['dataset'],p['vA'],p['seed'])): print(f"     {p['dataset']:<16}{p['vA']:<24}s{p['seed']:<3} dns={p['dns']:+6.1f} ratio={p['ratio']:.3f} dT={p['dT']:.3f} dmeta={p['dmeta']:+d}")

print("\n== GROUP A: model root WITHOUT flow (raw feature), paired vs packed_rank: probes saved S, dns, ns per probe saved ==")
for p in sorted(noflow_all, key=lambda p:(p['vA'],p['dataset'],p['seed'])):
    print(f"  {p['vA']:<30}{p['dataset']:<16}s{p['seed']:<3}{p['chosen']:<11} S={p['S']:5.2f} Stot={p['Stot']:5.2f} dns={p['dns']:+6.1f} ns/probe={(-p['dns']/p['S'] if p['S']>0.05 else float('nan')):+6.1f} dT={p['dT']:.3f} base={p['ns_base']:.0f}")
def fit(ps, intercept=True):
    # least squares dns = a - b*S  (returns a, b)
    xs = [-p['S'] for p in ps]; ys = [p['dns'] for p in ps]; n=len(xs)
    if intercept:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        b = sxy/sxx; a = my - b*mx
    else:
        b = sum(x*y for x,y in zip(xs,ys))/sum(x*x for x in xs); a = 0.0
    return a, b
def boot(ps, intercept, nb=4000, seed=7):
    rng = random.Random(seed); ds = sorted({p['dataset'] for p in ps}); byd = collections.defaultdict(list)
    for p in ps: byd[p['dataset']].append(p)
    A=[];B=[]
    for _ in range(nb):
        sample = [q for d in rng.choices(ds, k=len(ds)) for q in byd[d]]
        try: a,b = fit(sample, intercept)
        except ZeroDivisionError: continue
        A.append(a); B.append(b)
    A.sort(); B.sort(); n=len(A)
    return (A[int(.025*n)], A[int(.975*n)-1]), (B[int(.025*n)], B[int(.975*n)-1])
for name, ps in [('raw/vf4 only (T=0)', noflow), ('all no-flow model roots (incl. root_flow/fusion/vp10_fusion with region flow T<=0.09)', noflow_all)]:
    a0,b0 = fit(ps, False); a1,b1 = fit(ps, True)
    (alo,ahi),(blo,bhi) = boot(ps, True); (_,_),(b0lo,b0hi) = boot(ps, False)
    ratio_pool = -sum(p['dns'] for p in ps)/sum(p['S'] for p in ps)
    print(f"  FIT {name}: n={len(ps)}; through origin: ns/probe = {b0:.2f} [{b0lo:.2f},{b0hi:.2f}] (dataset-cluster bootstrap); with intercept: dns = {a1:+.1f} [{alo:+.1f},{ahi:+.1f}] - {b1:.2f} [{blo:.2f},{bhi:.2f}] * S; pooled sum(-dns)/sum(S) = {ratio_pool:.2f}")
    per_ds = collections.defaultdict(list)
    for p in ps: per_ds[(p['vA'],p['dataset'])].append(p)
    vals = []
    for k, qs in sorted(per_ds.items()):
        S = statistics.mean(q['S'] for q in qs); dns = statistics.mean(q['dns'] for q in qs)
        if S > 0.5: vals.append((k, S, dns, -dns/S))
    print("     per-dataset (mean over seeds) ns/probe: " + ", ".join(f"{k[1]}({k[0].replace('packed_rank_','')}) S={S:.1f} {v:+.1f}" for k,S,dns,v in vals))
    vv = sorted(v for *_,v in vals); print(f"     per-dataset ns/probe distribution: min={vv[0]:+.1f} med={statistics.median(vv):+.1f} max={vv[-1]:+.1f}; per-seed ns/probe (S>0.5): " + summ([-p['dns']/p['S'] for p in ps if p['S']>0.5]))

print("\n== GROUP B: model root WITH flow (T~1), paired vs the flow-free counterpart ==")
a_hat, b_hat = fit(noflow, True); b_org = fit(noflow, False)[1]
print(f"  using group-A raw/vf4 fit: a={a_hat:+.1f} ns fixed, b={b_hat:.2f} ns/probe (with intercept); b_origin={b_org:.2f}")
print("  F(b) = implied flow cost per lookup = dns + b*S (a=0);  F_ab = dns - a_hat + b_hat*S;  per call = /dT")
for p in sorted(flow, key=lambda p:(p['vA'],p['dataset'],p['seed'])):
    print(f"  {p['vA']:<30}{p['dataset']:<16}s{p['seed']:<3}{p['chosen']:<12}S={p['S']:5.2f} dns={p['dns']:+6.1f} ratio={p['ratio']:.3f} dT={p['dT']:.3f} | F(b=0)={p['dns']:+6.1f} F(4)={p['dns']+4*p['S']:+6.1f} F(8)={p['dns']+8*p['S']:+6.1f} F_ab={p['dns']-a_hat+b_hat*p['S']:+6.1f} | per call F(4)/dT={(p['dns']+4*p['S'])/p['dT']:+6.1f}")
for vA in ['packed_rank_root_fusion','packed_rank_vp10_root_fusion','packed_rank_root_flow']:
    ps = [p for p in flow if p['vA']==vA]
    if not ps: continue
    print(f"  SUMMARY {vA} (n={len(ps)}): dns " + summ([p['dns'] for p in ps]))
    print(f"      S " + summ([p['S'] for p in ps]) + f";  ratio% " + summ([100*(p['ratio']-1) for p in ps]))
    for b in (0, 4, 8, b_org):
        print(f"      F(b={b:.2f}) per lookup: " + summ([p['dns']+b*p['S'] for p in ps]))
    print(f"      F_ab (a={a_hat:+.1f}, b={b_hat:.2f}): " + summ([p['dns']-a_hat+b_hat*p['S'] for p in ps]))
    per_ds = collections.defaultdict(list)
    for p in ps: per_ds[p['dataset']].append(p)
    print("      per dataset mean dns / S / F(4) / F(8): " + "; ".join(f"{d} {statistics.mean(q['dns'] for q in qs):+.0f}/{statistics.mean(q['S'] for q in qs):.1f}/{statistics.mean(q['dns']+4*q['S'] for q in qs):+.0f}/{statistics.mean(q['dns']+8*q['S'] for q in qs):+.0f}" for d,qs in sorted(per_ds.items())))
# combined fusion vs packed and vp10_fusion vs vp10 pooled, dataset-cluster bootstrap of mean dns and mean F(b)
def cboot(ps, f, nb=4000, seed=3):
    rng = random.Random(seed); ds = sorted({p['dataset'] for p in ps}); byd = collections.defaultdict(list)
    for p in ps: byd[p['dataset']].append(p)
    M = sorted(statistics.mean(f(q) for d in rng.choices(ds,k=len(ds)) for q in byd[d]) for _ in range(nb))
    return statistics.mean(f(p) for p in ps), M[int(.025*nb)], M[int(.975*nb)-1]
fl = [p for p in flow if p['vA'] in ('packed_rank_root_fusion','packed_rank_vp10_root_fusion')]
print(f"  POOLED (root_fusion vs packed_rank and vp10_root_fusion vs vp10, n={len(fl)}): mean dns = {cboot(fl, lambda p:p['dns'])[0]:+.1f} [{cboot(fl, lambda p:p['dns'])[1]:+.1f},{cboot(fl, lambda p:p['dns'])[2]:+.1f}]; mean S={statistics.mean(p['S'] for p in fl):.2f}; mean dT={statistics.mean(p['dT'] for p in fl):.3f}")
for b in (0,4,8):
    m,lo,hi = cboot(fl, lambda p, b=b: (p['dns']+b*p['S'])/p['dT'])
    print(f"     implied ns per flow call at b={b}: {m:.1f} [{lo:.1f},{hi:.1f}]  => probe-equivalents at that b: {(m/b if b else float('nan')):.1f}")
m,lo,hi = cboot(fl, lambda p: (p['dns']-a_hat+b_hat*p['S'])/p['dT'])
print(f"     implied ns per flow call with group-A fit (a={a_hat:+.1f}, b={b_hat:.2f}): {m:.1f} [{lo:.1f},{hi:.1f}]")
# joint bootstrap: resample datasets in group A and group B, propagate a,b
rng = random.Random(11); dsA = sorted({p['dataset'] for p in noflow}); dsB = sorted({p['dataset'] for p in fl}); byA=collections.defaultdict(list); byB=collections.defaultdict(list)
for p in noflow: byA[p['dataset']].append(p)
for p in fl: byB[p['dataset']].append(p)
J=[]; J0=[]
for _ in range(4000):
    sa = [q for d in rng.choices(dsA,k=len(dsA)) for q in byA[d]]; sb = [q for d in rng.choices(dsB,k=len(dsB)) for q in byB[d]]
    try: a,b = fit(sa, True); b0 = fit(sa, False)[1]
    except ZeroDivisionError: continue
    J.append(statistics.mean((q['dns']-a+b*q['S'])/q['dT'] for q in sb)); J0.append(statistics.mean((q['dns']+b0*q['S'])/q['dT'] for q in sb))
J.sort(); J0.sort()
print(f"     joint bootstrap (A and B resampled): flow ns/call with intercept model [{J[int(.025*len(J))]:.1f},{J[int(.975*len(J))-1]:.1f}]; origin model [{J0[int(.025*len(J0))]:.1f},{J0[int(.975*len(J0))-1]:.1f}]")

# ---------- 6. C5 thresholds ----------
print("\n== C5: charge thresholds. est_vp_flow and est_flow INCLUDE the +4 charge (index.hpp:389). c* = charge at which every flow candidate stops beating min(binary, raw, vp_raw) ==")
print("   (a flow candidate is chosen iff probes_flow + c < min(binary, raw, vp_raw); so c* = min(binary,raw,vp_raw) - (est - 4))")
cstar = {}
for d in datasets:
    c = cell.get((d,'packed_rank_root_fusion'))
    if not c: continue
    bn, raw, fl_, vr, vf = c['est']
    best_noflow = min(x for x in (bn, raw, vr) if x>0)
    cs = []
    if vf>0: cs.append(('flow+fences', best_noflow-(vf-4)))
    if fl_>0: cs.append(('flow+ranks', best_noflow-(fl_-4)))
    cm = max(v for _,v in cs); cstar[d]=cm
    dec10 = 'binary' if best_noflow==bn else ('raw+fences' if best_noflow==vr else 'raw+ranks')
    print(f"  {d:<16} chosen={c['chosen']:<12} est bin={bn:.2f} raw={raw:.2f} flow={fl_:.2f} vp_raw={vr:.2f} vp_flow={vf:.2f} -> uncharged vp_flow={vf-4:.2f}; c*={cm:5.2f} ({', '.join(f'{n}:{v:.2f}' for n,v in cs)}); at charge 10 -> {dec10}; measured root probes fusion={c['root']:.2f}")
fs = {d:v for d,v in cstar.items() if cell[(d,'packed_rank_root_fusion')]['chosen'].startswith('flow')}
print(f"  max c* over datasets where the flow was chosen: {max(fs.values()):.2f} ({max(fs,key=fs.get)}); min: {min(fs.values()):.2f} ({min(fs,key=fs.get)})")
print(f"  => any charge > {max(fs.values()):.2f} flips every flow selection to the flow-free best; charge 10 is sufficient but not the threshold.")
print("  max possible root saving at 489 regions = binary 8.96 - ~2.5 probes ~ 6.5 probes < any charge >= 7, so a calibrated charge >= 10 can never select the flow at the root on these 2M samples.")

# ---------- 7. build-time transform cost ----------
print("\n== transform_ns per key at build (learnability.transform_ns / keys), runs with a flow file ==")
tk = collections.defaultdict(list)
for r in rows:
    l = r.get('learnability') or {}
    if l.get('transform_ns',0)>0: tk[r['variant']].append(l['transform_ns']/l['keys'])
for v,xs in sorted(tk.items()): print(f"  {v:<30} " + summ(xs))

json.dump(dict(sham=sham, shamflow=shamflow, noflow=noflow, noflow_all=noflow_all, flow=flow, cstar=cstar, cell={f"{d}|{v}":c for (d,v),c in cell.items()}, memtab={f"{d}|{v}":c for (d,v),c in memtab.items()}), open(REPO + '/build-verify-coststory/scratch/coststory.json','w'), indent=1, default=str)
