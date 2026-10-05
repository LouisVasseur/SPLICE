import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json, collections, math, statistics
P=REPO + '/experimental/scaleli/results/aidb_final/sweep/results.jsonl'
rows=[json.loads(l) for l in open(P) if l.strip()]
MAIN={'packed_rank','packed_rank_root_raw','packed_rank_root_flow','packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_vp10','packed_rank_vp10_root_fusion','raw_rank','sorted_vector'}
rows=[r for r in rows if r['variant'] in MAIN and r['seed'] in (11,29,47)]
by=collections.defaultdict(list)
for r in rows: by[(r['dataset'],r['variant'])].append(r)
# pairing check
fp=collections.defaultdict(set)
for r in rows: fp[(r['dataset'],r['seed'])].add(r['trace_fingerprint'])
print("PAIRING: groups",len(fp),"groups with >1 fingerprint:",sum(1 for v in fp.values() if len(v)>1))
dup=[(k,len(v)) for k,v in by.items() if len(v)!=3]
print("groups with runs!=3:",dup)
def w(r,k): return r['work_counters'][k]/r['operations']
def med(rs,f): return statistics.median(f(r) for r in rs)
def one(rs,f):
    vals=sorted(set(round(f(r),6) for r in rs)); return vals
def chosen(l):
    if not l.get('root_model'): return 'binary'
    return ('flow' if l.get('root_flow') else 'raw')+'+'+('fences' if l.get('root_vp') else 'ranks')
def paired(d,v,base='packed_rank'):
    b={r['seed']:r for r in by[(d,base)]}; rs=[]
    for r in by[(d,v)]:
        if r['seed'] in b: rs.append(r['throughput_ops_s']/b[r['seed']]['throughput_ops_s'])
    if not rs: return float('nan'),[]
    return math.exp(statistics.mean(math.log(x) for x in rs)), rs
datasets=sorted({d for d,_ in by})
BIN=8.96
print("\n=== C1/C2: root fusion + vf4 per dataset ===")
hdr=f"{'dataset':<16}{'fus.chosen':>12}{'root/op bin':>12}{'root/op fus':>12}{'est bin':>8}{'raw':>6}{'flow':>6}{'vf':>6}{'fus':>6}{'syn':>5} | {'vf4.chosen':>11}{'vf4 root/op':>12}{'vf4 est':>8}"
print(hdr)
c1={}; c2={}
for d in datasets:
    b=by[(d,'packed_rank')]; f=by[(d,'packed_rank_root_fusion')]; v=by[(d,'packed_rank_root_vf4')]
    lf=f[0]['learnability']; lv=v[0]['learnability']
    assert all(chosen(r['learnability'])==chosen(lf) for r in f), d
    e=[lf['root_probes_binary'],lf['root_probes_raw'],lf['root_probes_flow'],lf['root_probes_vp_raw'],lf['root_probes_vp_flow']]
    singles=[x for x in e[1:4] if x>0]
    syn = e[4]>0 and e[4]<min(singles) and e[4]<e[0] and all(x>e[0] for x in singles)
    c1[d]=(chosen(lf),med(b,lambda r:w(r,'root_probes')),med(f,lambda r:w(r,'root_probes')),e,syn)
    c2[d]=(chosen(lv),med(v,lambda r:w(r,'root_probes')),lv['root_probes_vp_raw'],lv['root_probes_raw'])
    print(f"{d:<16}{chosen(lf):>12}{c1[d][1]:12.2f}{c1[d][2]:12.2f}{e[0]:8.2f}{e[1]:6.1f}{e[2]:6.1f}{e[3]:6.1f}{e[4]:6.1f}{str(syn):>5} | {chosen(lv):>11}{c2[d][1]:12.2f}{lv['root_probes_vp_raw']:8.2f}")
win=[d for d in datasets if d.endswith('_window')]; uni=[d for d in datasets if d.endswith('_uniform')]
print("\nC1 window: fusion chose flow+fences on", sum(1 for d in win if c1[d][0]=='flow+fences'),"/",len(win),"; not:",[ (d,c1[d][0]) for d in win if c1[d][0]!='flow+fences'])
fw=[c1[d][2] for d in win if c1[d][0]=='flow+fences']; print("   measured root/op where chosen: min %.2f max %.2f"%(min(fw),max(fw)),"; synergy(all singles>binary & fusion<binary) count:",sum(1 for d in win if c1[d][4]),"/",len(win))
print("   baseline root/op on windows:", sorted(set(round(c1[d][1],2) for d in win)))
print("C2 uniform: vf4 chosen raw+fences on", sum(1 for d in uni if c2[d][0]=='raw+fences'),"/",len(uni), "; per-dataset root/op:", {d.split('_')[0]:round(c2[d][1],2) for d in uni})
print("   uniform in [2.1,3.3]:", [d.split('_')[0] for d in uni if 2.1<=c2[d][1]<=3.3], "count", sum(1 for d in uni if 2.1<=c2[d][1]<=3.3))
print("   window vf4 (raw feature) chosen:", {d.split('_')[0]:c2[d][0] for d in win})
print("   root_raw variant on windows chosen:", {d.split('_')[0]:chosen(by[(d,'packed_rank_root_raw')][0]['learnability']) for d in win})
print("\n=== C3: paired throughput vs packed_rank (geomean of per-seed ratios; [min,max] per-seed) ===")
c3={}
for d in datasets:
    line=f"{d:<16}"
    for v in ('packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_vp10','packed_rank_vp10_root_fusion','raw_rank','sorted_vector'):
        g,rs=paired(d,v); c3[(d,v)]=(g,min(rs),max(rs))
        line+=f" {v.replace('packed_rank_','')[:12]:>12}={g:5.3f}[{min(rs):.2f},{max(rs):.2f}]"
    print(line)
print("\nC3 fences-only where model root chosen (uniform):", {d.split('_')[0]:round(c3[(d,'packed_rank_root_vf4')][0],3) for d in uni if c2[d][0]!='binary'})
print("C3 flow-fused root on windows:", {d.split('_')[0]:round(c3[(d,'packed_rank_root_fusion')][0],3) for d in win}, " range where flow chosen:", (round(min(c3[(d,'packed_rank_root_fusion')][0] for d in win if c1[d][0]=='flow+fences'),3), round(max(c3[(d,'packed_rank_root_fusion')][0] for d in win if c1[d][0]=='flow+fences'),3)))
print("C3 fusion elsewhere (uniform):", {d.split('_')[0]:(c1[d][0],round(c3[(d,'packed_rank_root_fusion')][0],3)) for d in uni})
tp={}
for d in win:
    b=by[(d,'packed_rank')]; f=by[(d,'packed_rank_root_fusion')]
    tot=lambda rs: med(rs,lambda r:w(r,'root_probes')+w(r,'fence_probes')+w(r,'coordinate_probes'))
    tp[d]=(tot(b),tot(f),1-tot(f)/tot(b))
print("C3 total probes window base->fusion (reduction):", {d.split('_')[0]:(round(a,1),round(b_,1),f"{c*100:.0f}%") for d,(a,b_,c) in tp.items()})
print("C3 region vp alone:", {d.replace('_uniform','_u').replace('_window','_w'):round(c3[(d,'packed_rank_vp10')][0],3) for d in datasets})
print("C3 vp+root fusion (uniform):", {d.split('_')[0]:round(c3[(d,'packed_rank_vp10_root_fusion')][0],3) for d in uni})
print("\n=== C7: sorted_vector vs packed_rank and vs every packed variant ===")
sv=[c3[(d,'sorted_vector')][0] for d in datasets]; print("sorted_vector vs packed_rank: min %.3f max %.3f"%(min(sv),max(sv)), {d:round(c3[(d,'sorted_vector')][0],2) for d in datasets})
allmin=1e9; allmax=0
for d in datasets:
    for v in ('packed_rank','packed_rank_root_raw','packed_rank_root_flow','packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_vp10','packed_rank_vp10_root_fusion'):
        g,rs=paired(d,'sorted_vector',base=v); allmin=min(allmin,g); allmax=max(allmax,g)
print("sorted_vector vs every packed variant: geomean range %.3f .. %.3f"%(allmin,allmax))
print("\n=== C6: transform_calls/op and metadata by variant (window datasets) ===")
for d in win:
    line=f"{d:<16}"
    for v in ('packed_rank','packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_root_flow'):
        rs=by[(d,v)]; line+=f" {v.replace('packed_rank','pr')[:14]:>14}: tc/op={med(rs,lambda r:w(r,'transform_calls')):.2f} metaB/key={med(rs,lambda r:r['memory_before']['metadata_bytes']/r['initial_rows']):.3f} tns={rs[0]['learnability']['transform_ns']/1e6:.0f}ms"
    print(line)
# non-root counters identical between root variants and baseline?
print("\nnon-root counters identical to baseline per (dataset,seed)?")
diffs=collections.Counter()
for d in datasets:
    b={r['seed']:r for r in by[(d,'packed_rank')]}
    for v in ('packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_root_raw','packed_rank_root_flow'):
        for r in by[(d,v)]:
            bb=b[r['seed']]
            for k in ('fence_probes','coordinate_probes','decoded_keys','key_at_calls','block_routes','codec_bytes_examined'):
                if r['work_counters'][k]!=bb['work_counters'][k]: diffs[(v,k)]+=1
print("  mismatches:",dict(diffs) or "none")
tc_pos=[(r['dataset'],r['variant']) for r in rows if r['work_counters']['transform_calls']>0]
print("  variants with transform_calls>0:", collections.Counter(v for _,v in tc_pos))
print("  root_flow true <=> transform_calls>0:", all((r['learnability'].get('root_flow',False))==(r['work_counters']['transform_calls']>0) for r in rows if 'root' in r['variant']))
print("\n=== C4: cost decomposition (paired per (dataset,seed)) ===")
# raw-feature root pairs (vf4 & root_raw where model chosen): dt = a * dprobes
xs=[];ys=[]
for d in datasets:
    b={r['seed']:r for r in by[(d,'packed_rank')]}
    for v in ('packed_rank_root_vf4','packed_rank_root_raw'):
        for r in by[(d,v)]:
            if not r['learnability'].get('root_model') or r['learnability'].get('root_flow'): continue
            bb=b[r['seed']]; dt=1e9/r['throughput_ops_s']-1e9/bb['throughput_ops_s']; dp=w(r,'root_probes')-w(bb,'root_probes')
            xs.append(dp); ys.append(dt)
a=sum(x*y for x,y in zip(xs,ys))/sum(x*x for x in xs)
print(f"raw-root pairs n={len(xs)}: ns per root probe (through-origin LS) = {a:.2f}; per-pair ratio median={statistics.median(y/x for x,y in zip(xs,ys)):.2f}, IQR=[{statistics.quantiles([y/x for x,y in zip(xs,ys)],n=4)[0]:.2f},{statistics.quantiles([y/x for x,y in zip(xs,ys)],n=4)[2]:.2f}]")
# bootstrap over datasets? simple pair bootstrap
import random
rng=random.Random(1); boots=[]
for _ in range(2000):
    idx=[rng.randrange(len(xs)) for _ in xs]; bx=[xs[i] for i in idx]; byy=[ys[i] for i in idx]; boots.append(sum(x*y for x,y in zip(bx,byy))/sum(x*x for x in bx))
boots.sort(); print(f"   bootstrap 95%: [{boots[50]:.2f}, {boots[1949]:.2f}]")
# flow pairs: dt = a*dprobes + c*tc ; solve c with a fixed
cs=[]; saved=[]
for d in datasets:
    b={r['seed']:r for r in by[(d,'packed_rank')]}
    for v in ('packed_rank_root_fusion','packed_rank_root_flow','packed_rank_vp10_root_fusion'):
        for r in by[(d,v)]:
            if not r['learnability'].get('root_flow'): continue
            bb=b[r['seed']]; dt=1e9/r['throughput_ops_s']-1e9/bb['throughput_ops_s']; dp=w(r,'root_probes')-w(bb,'root_probes'); tc=w(r,'transform_calls')
            if v=='packed_rank_vp10_root_fusion': continue  # different region counters
            cs.append((d,v,r['seed'],dt,dp,tc,(dt-a*dp)/tc)); saved.append(-dp)
c_vals=[c[-1] for c in cs]
print(f"flow-root pairs n={len(cs)}: flow eval ns median={statistics.median(c_vals):.1f} mean={statistics.mean(c_vals):.1f} min={min(c_vals):.1f} max={max(c_vals):.1f}; probe-equivalents (at {a:.1f} ns/probe) median={statistics.median(c_vals)/a:.1f}; root probes saved range {min(saved):.2f}..{max(saved):.2f}; tc/op values {sorted(set(round(c[5],2) for c in cs))}")
print("   net ns/op delta (fusion vs base) median=%.1f"%statistics.median(c[3] for c in cs))
print("\n=== C5: selector flip threshold on flow charge ===")
# selector: candidate cost = est (+ charge for flow ones). Current charge 4 already included in est? check: does est flow include the 4?
# We compute: fusion chosen iff vp_flow_est_excl + charge < min(binary, raw, vp_raw). Infer excl = est - 4 assuming the recorded estimate includes the charge.
for incl in (True,False):
    thr={}
    for d in datasets:
        l=by[(d,'packed_rank_root_fusion')][0]['learnability']
        if not l.get('root_flow'): continue
        e_bin=l['root_probes_binary']; e_raw=l['root_probes_raw']; e_vraw=l['root_probes_vp_raw']; e_vflow=l['root_probes_vp_flow']-(4 if incl else 0)
        best_nonflow=min(x for x in (e_bin,e_raw,e_vraw) if x>0)
        thr[d]=round(best_nonflow-e_vflow,2)  # charge at which fusion ties best non-flow
    print(f"   (assuming recorded flow estimates {'INCLUDE' if incl else 'EXCLUDE'} the 4-probe charge) charge at which fusion ties best non-flow candidate, per dataset:", thr, "-> max:", max(thr.values()))
