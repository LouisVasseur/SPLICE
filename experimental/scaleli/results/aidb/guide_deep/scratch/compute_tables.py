#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Compute every table of 07_results.md from the raw result files (stdlib only).
Writes markdown fragments to scratch/tables/<name>.md and a JSON with the key numbers."""
import json, math, statistics, collections, random, os, csv, itertools
S = REPO + "/experimental/scaleli"
OUT = f"{S}/results/aidb/guide_deep/scratch/tables"
os.makedirs(OUT, exist_ok=True)
DS = ['books','fb','osm','covid','genome','history','libio','planet','stack','wise']
KEY = {}

def w(name, text):
    open(f"{OUT}/{name}.md","w").write(text)

def pct(a,b):
    return (b-a)/a*100 if a else float('nan')

def fmt(x, d=2):
    if isinstance(x,str): return x
    if x is None or (isinstance(x,float) and math.isnan(x)): return "n/a"
    if abs(x) >= 1e6 and d<=3: return f"{x:,.0f}"
    return f"{x:,.{d}f}"

def table(header, rows):
    out = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows: out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out

def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def paired(rows_v, rows_b):
    by = {r["seed"]: r for r in rows_b}
    ratios = [(r["seed"], r["throughput_ops_s"]/by[r["seed"]]["throughput_ops_s"]) for r in rows_v if r["seed"] in by]
    if not ratios: return float('nan'), float('nan'), float('nan'), 0
    logs = [math.log(x) for _,x in ratios]; est = math.exp(statistics.mean(logs))
    if len(logs)<2: return est, float('nan'), float('nan'), len(logs)
    rng = random.Random(42); boot = sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(2000))
    return est, boot[int(0.025*len(boot))], boot[int(0.975*len(boot))-1], len(logs)

# ---------------------------------------------------------------- E1 hardness
HU = json.load(open(f"{S}/results/aidb/hardness.json"))
HW = json.load(open(f"{S}/results/aidb_window/hardness.json"))
M = [("rmse","RMSE"),("max_error","ME"),("conflict_degree","CD"),("pla_32","PLA-32"),("pla_4096","PLA-4096")]
hdr = ["dataset"] + [f"{n} raw" for _,n in M]
r1=[]; r2=[]
for d in DS:
    f=HU[d]['full']; g=HU[d]['full_flow']
    r1.append([d]+[fmt(f[k],0) if k!='rmse' else fmt(f[k],1) for k,_ in M])
    r2.append([d]+[f"{fmt(g[k],1 if k=='rmse' else 0)} ({pct(f[k],g[k]):+.2f}%)" for k,_ in M])
w("e1_full_raw", table(hdr, r1))
w("e1_full_flow", table(["dataset"]+[f"{n} flow (Δ%)" for _,n in M], r2))
# sample scopes uniform and window
for tag,H in (("uniform",HU),("window",HW)):
    for k,n in M:
        rr=[]
        for d in DS:
            s=H[d]
            base=s['sample'][k]
            rr.append([d]+[f"{fmt(s[sc][k],1 if k=='rmse' else 0)}" + ("" if sc=='sample' else f" ({pct(base,s[sc][k]):+.1f}%)") for sc in ('sample','sample_flow','sample_csv','sample_flow_csv')])
        w(f"e1_{tag}_{k}", table(["dataset","sample","sample_flow (Δ%)","sample_csv (Δ%)","sample_flow_csv (Δ%)"], rr))
KEY['e1'] = {d:{sc:HU[d][sc] for sc in HU[d]} for d in DS}
KEY['e1_window'] = {d:{sc:HW[d][sc] for sc in HW[d]} for d in DS}
# window full scopes are the same files? check
KEY['window_full_equals_uniform_full'] = all(HW[d]['full']==HU[d]['full'] for d in DS)

# ---------------------------------------------------------------- E2 region sweeps
def region_tables(path, tag):
    R = rows(path); by=collections.defaultdict(list)
    for r in R: by[(r['dataset'],r['variant'])].append(r)
    V = ['packed_rank','packed_rank_flow','packed_rank_flow_forced','packed_rank_vp10','packed_rank_flow_vp10','packed_rank_fusion_auto','packed_rank_flow_costsel','packed_byte','raw_rank','sorted_vector']
    SH = {'packed_rank':'control','packed_rank_flow':'flow (bypass)','packed_rank_flow_forced':'flow forced','packed_rank_vp10':'vp10','packed_rank_flow_vp10':'flow_vp10','packed_rank_fusion_auto':'fusion_auto','packed_rank_flow_costsel':'flow_costsel','packed_byte':'packed_byte','raw_rank':'raw_rank','sorted_vector':'sorted_vector'}
    def wc(r,k): return r['work_counters'][k]/r['operations']
    def med(rs,f): return statistics.median(f(r) for r in rs)
    out={}
    for d in DS:
        out[d]={}
        for v in V:
            rs=by.get((d,v),[])
            if not rs: continue
            L=rs[0].get('learnability') or {}
            base=by[(d,'packed_rank')]
            est,lo,hi,n = (1.0,1.0,1.0,3) if v=='packed_rank' else paired(rs,base)
            regs=L.get('regions',0) or 1
            ch=L.get('choices',{}) or {}
            out[d][v]=dict(fence=med(rs,lambda r:wc(r,'fence_probes')), coord=med(rs,lambda r:wc(r,'coordinate_probes')), root=med(rs,lambda r:wc(r,'root_probes')),
                corr=med(rs,lambda r:wc(r,'correction_distance')), tc=med(rs,lambda r:wc(r,'transform_calls')), flow_frac=L.get('flow_regions',0)/regs,
                choices={k:ch.get(k,0) for k in ('none','flow','vp','both')}, regions=L.get('regions',0), vp=L.get('virtual_points',0), keys=L.get('keys',1),
                sse_b=L.get('rank_sse_before',0), sse_a=L.get('rank_sse_after',0), bpk=med(rs,lambda r:r['memory_before']['accounted_bytes']/r['initial_rows']),
                meta=med(rs,lambda r:r['memory_before']['metadata_bytes']/r['initial_rows']), build_ms=med(rs,lambda r:r['build_ns']/1e6), smooth_s=L.get('smoothing_ns',0)/1e9, transform_s=L.get('transform_ns',0)/1e9,
                tcr=L.get('tail_conflicts_raw_mean',0), tcf=L.get('tail_conflicts_flow_mean',0), mops=med(rs,lambda r:r['throughput_ops_s']/1e6), sp=est, lo=lo, hi=hi, n=len(rs),
                cost_none=L.get('cost_none_mean',0), cost_sel=L.get('cost_selected_mean',0), seeds=sorted(r['seed'] for r in rs), ops=rs[0]['operations'])
    # Table A: fence probes per lookup by variant
    VA=['packed_rank','packed_rank_flow','packed_rank_flow_forced','packed_rank_vp10','packed_rank_flow_vp10','packed_rank_fusion_auto','packed_rank_flow_costsel']
    t=[]
    for d in DS:
        o=out[d]; c=o['packed_rank']['fence']
        t.append([d]+[fmt(o[v]['fence'],3) for v in VA]+[f"{pct(c,o['packed_rank_vp10']['fence']):+.1f}%", f"{pct(c,o['packed_rank_fusion_auto']['fence']):+.1f}%"])
    w(f"e2_{tag}_fence", table(["dataset"]+[SH[v] for v in VA]+["vp10 Δ","fusion Δ"], t))
    # Table B: coordinate probes, correction distance, transform calls
    t=[]
    for d in DS:
        o=out[d]
        t.append([d, fmt(o['packed_rank']['coord'],3), fmt(o['packed_rank_vp10']['coord'],3), fmt(o['packed_rank']['corr'],3), fmt(o['packed_rank_vp10']['corr'],3), f"{pct(o['packed_rank']['corr'],o['packed_rank_vp10']['corr']):+.1f}%",
                  fmt(o['packed_rank_flow']['tc'],4), fmt(o['packed_rank_flow_forced']['tc'],4), fmt(o['packed_rank_flow_vp10']['tc'],4), fmt(o['packed_rank_fusion_auto']['tc'],4)])
    w(f"e2_{tag}_coord", table(["dataset","coord control","coord vp10","corr.dist control","corr.dist vp10","Δ","transform_calls flow","forced","flow_vp10","fusion_auto"], t))
    # Table C: acceptance / choices / tail conflicts
    t=[]
    for d in DS:
        o=out[d]; f=o['packed_rank_flow']; fa=o['packed_rank_fusion_auto']; cs=o['packed_rank_flow_costsel']; regs=fa['regions']
        t.append([d, regs, fmt(f['tcr'],2), fmt(f['tcf'],2), f"{f['flow_frac']*100:.2f}% ({round(f['flow_frac']*regs)})",
                  f"{fa['choices']['none']}/{fa['choices']['flow']}/{fa['choices']['vp']}/{fa['choices']['both']}", f"{cs['choices']['none']}/{cs['choices']['flow']}", fmt(fa['cost_none'],3), fmt(fa['cost_sel'],3), fmt(cs['cost_none'],3), fmt(cs['cost_sel'],3)])
    w(f"e2_{tag}_choices", table(["dataset","regions","D99 raw mean","D99 flow mean","flow accepted (bypass)","fusion_auto none/flow/vp/both","costsel none/flow","fusion cost none","fusion cost selected","costsel cost none","costsel cost selected"], t))
    # Table D: SSE, memory, build, preprocessing
    t=[]
    for d in DS:
        o=out[d]; c=o['packed_rank']; v=o['packed_rank_vp10']; fl=o['packed_rank_flow_forced']; fa=o['packed_rank_fusion_auto']
        t.append([d, f"{c['sse_b']:.4g}", f"{v['sse_a']:.4g}", f"{v['sse_a']/v['sse_b']:.3f}", f"{fl['sse_a']/fl['sse_b']:.3f}" if fl['sse_b'] else "n/a", fmt(v['vp']/v['keys'],4), fmt(c['bpk'],2), fmt(v['bpk'],2), fmt(v['meta']-c['meta'],3), fmt(c['build_ms'],1), fmt(v['build_ms'],1), fmt(fa['build_ms'],1), fmt(v['smooth_s'],2), fmt(fa['smooth_s'],2), fmt(fl['transform_s']*1e9/fl['keys'],1)])
    w(f"e2_{tag}_sse", table(["dataset","rank SSE control","slot SSE vp10","ratio vp10","ratio flow forced","vp/key","B/key control","B/key vp10","meta Δ B/key","build ms control","build ms vp10","build ms fusion","smoothing s vp10 (thread)","smoothing s fusion","transform ns/key forced"], t))
    # Table E: throughput and speedups
    VT=['packed_rank','packed_rank_flow','packed_rank_flow_forced','packed_rank_vp10','packed_rank_flow_vp10','packed_rank_fusion_auto','packed_rank_flow_costsel','packed_byte','raw_rank','sorted_vector']
    t=[]
    for d in DS:
        o=out[d]
        t.append([d, fmt(o['packed_rank']['mops'],3)]+[f"{o[v]['sp']:.3f} [{o[v]['lo']:.2f}, {o[v]['hi']:.2f}]" for v in VT[1:]])
    w(f"e2_{tag}_speed", table(["dataset","control Mops"]+[SH[v] for v in VT[1:]], t))
    return out
E2U = region_tables(f"{S}/results/aidb/sweep/results.jsonl","uniform")
E2W = region_tables(f"{S}/results/aidb_window/sweep/results.jsonl","window")
# summary stats for readings
def sav(E):
    return {d: pct(E[d]['packed_rank']['fence'],E[d]['packed_rank_vp10']['fence']) for d in DS}
KEY['e2_vp_saving_uniform']=sav(E2U); KEY['e2_vp_saving_window']=sav(E2W)
KEY['e2_forced_maxdiff']={t:max(abs(E[d]['packed_rank_flow_forced']['fence']-E[d]['packed_rank']['fence']) for d in DS) for t,E in (('uniform',E2U),('window',E2W))}
KEY['e2_flow_choice']={t:{d:E[d]['packed_rank_fusion_auto']['choices'] for d in DS} for t,E in (('uniform',E2U),('window',E2W))}
KEY['e2_costsel_choice']={t:{d:E[d]['packed_rank_flow_costsel']['choices'] for d in DS} for t,E in (('uniform',E2U),('window',E2W))}
KEY['e2_corr_saving']={t:{d:pct(E[d]['packed_rank']['corr'],E[d]['packed_rank_vp10']['corr']) for d in DS} for t,E in (('uniform',E2U),('window',E2W))}
KEY['e2_speed_vp10']={t:{d:(E[d]['packed_rank_vp10']['sp'],E[d]['packed_rank_vp10']['lo'],E[d]['packed_rank_vp10']['hi']) for d in DS} for t,E in (('uniform',E2U),('window',E2W))}
KEY['e2_flow_frac']={t:{d:E[d]['packed_rank_flow']['flow_frac'] for d in DS} for t,E in (('uniform',E2U),('window',E2W))}

# ---------------------------------------------------------------- E3 final sweep
RA = json.load(open(f"{S}/results/aidb_final/root_analysis.json"))['per_dataset']
FR = rows(f"{S}/results/aidb_final/sweep/results.jsonl")
byf=collections.defaultdict(list)
for r in FR: byf[(r['dataset'],r['variant'])].append(r)
V3=['packed_rank','packed_rank_root_raw','packed_rank_root_flow','packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_vp10','packed_rank_vp10_root_fusion']
SH3={'packed_rank':'binary root (control)','packed_rank_root_raw':'raw root','packed_rank_root_flow':'flow root','packed_rank_root_vf4':'root fences','packed_rank_root_fusion':'root fusion','packed_rank_vp10':'region vp10','packed_rank_vp10_root_fusion':'vp10 + root fusion'}
samples=sorted(RA.keys())
t=[]
for s in samples:
    for v in V3:
        x=RA[s].get(v);
        if not x: continue
        e=x['est']
        t.append([s if v=='packed_rank' else "", SH3[v], fmt(x['root_probes'],2), fmt(x['fence_probes'],2), fmt(x['total_probes'],2), f"{pct(RA[s]['packed_rank']['total_probes'],x['total_probes']):+.1f}%", x['chosen'], x['root_virtual'],
                  f"{e[0]:.2f}/{e[1]:.2f}/{e[2]:.2f}/{e[3]:.2f}/{e[4]:.2f}", fmt(x['build_s'],3), fmt(x['meta_bytes_per_key'],4), fmt(x['mops'],3), f"{x['speedup']:.3f} [{x['lo']:.2f}, {x['hi']:.2f}]"])
w("e3_root", table(["sample","variant","root/op","fence/op","total/op","total Δ","chosen","virt. fences","est bin/raw/flow/vf/fus","build s","meta B/key","Mops","speedup [seed spread]"], t))
# compact per-sample summary
t=[]
for s in samples:
    b=RA[s]['packed_rank']; f=RA[s]['packed_rank_root_fusion']; vf=RA[s]['packed_rank_root_vf4']; vp=RA[s]['packed_rank_vp10']; vpf=RA[s]['packed_rank_vp10_root_fusion']; raw=RA[s]['packed_rank_root_raw']
    t.append([s, fmt(b['root_probes'],2), fmt(raw['root_probes'],2), raw['chosen'], fmt(vf['root_probes'],2), vf['root_virtual'], vf['chosen'], fmt(f['root_probes'],2), f['chosen'], fmt(b['total_probes'],2), fmt(f['total_probes'],2), f"{pct(b['total_probes'],f['total_probes']):+.1f}%", fmt(vpf['total_probes'],2), f"{pct(b['total_probes'],vpf['total_probes']):+.1f}%"])
w("e3_summary", table(["sample","binary root/op","raw root/op","raw chosen","fences root/op","virt","fences chosen","fusion root/op","fusion chosen","total control","total fusion","Δ","total vp10+fusion","Δ"], t))
KEY['e3']={s:{v:{k:RA[s][v][k] for k in ('root_probes','fence_probes','total_probes','chosen','root_virtual','est','speedup','lo','hi','build_s','meta_bytes_per_key','mops')} for v in RA[s]} for s in samples}
# stale rows
ST = rows(f"{S}/results/aidb_final/sweep/results.stale-root-before-fix.jsonl")
bys=collections.defaultdict(list)
for r in ST: bys[(r['dataset'],r['variant'])].append(r)
t=[]
for s in samples:
    for v in ['packed_rank_root_raw','packed_rank_root_flow','packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_vp10_root_fusion']:
        rs=[r for r in bys.get((s,v),[]) if r['seed'] in (11,29,47)]
        if not rs: continue
        L=rs[0]['learnability']; wc=lambda r,k:r['work_counters'][k]/r['operations']
        chosen = "-" if not L.get('root_model') else ("flow+" if L.get('root_flow') else "raw+")+("fences" if L.get('root_vp') else "ranks")
        if not L.get('root_model'): chosen="binary(fallback)"
        base=byf[(s,'packed_rank')]
        est,lo,hi,n=paired(rs,base)
        if v in ('packed_rank_root_fusion','packed_rank_vp10_root_fusion'):
            t.append([s, SH3[v], chosen, fmt(statistics.median(wc(r,'root_probes') for r in rs),2), fmt(statistics.median(wc(r,'transform_calls') for r in rs),3), L.get('root_virtual',0), f"{L.get('root_probes_binary',0):.2f}/{L.get('root_probes_raw',0):.2f}/{L.get('root_probes_flow',0):.2f}/{L.get('root_probes_vp_raw',0):.2f}/{L.get('root_probes_vp_flow',0):.2f}", f"{est:.3f} [{lo:.2f}, {hi:.2f}]"])
w("e3_stale", table(["sample","variant (PRE-FIX rows)","chosen","root/op","transform_calls/op","virt","est bin/raw/flow/vf/fus","speedup vs (post-fix) control"], t))
# stale summary numbers
stale_chosen=collections.Counter()
for s in samples:
    rs=[r for r in bys.get((s,'packed_rank_root_fusion'),[]) if r['seed']==11]
    if rs:
        L=rs[0]['learnability']; stale_chosen[("flow+" if L.get('root_flow') else "raw+")+("fences" if L.get('root_vp') else "ranks") if L.get('root_model') else 'binary']+=1
KEY['e3_stale_fusion_choices']=dict(stale_chosen)
KEY['e3_stale_n']=len(ST); KEY['e3_stale_seeds']=sorted({r['seed'] for r in ST}); KEY['e3_stale_variants']=dict(collections.Counter(r['variant'] for r in ST))
# post-fix flow chosen count over 3 root variants with flow
flowcells=0; flowchosen=0
for s in samples:
    for v in ('packed_rank_root_flow','packed_rank_root_fusion','packed_rank_vp10_root_fusion'):
        for r in byf[(s,v)]:
            flowcells+=1; flowchosen+= 1 if (r['learnability'] or {}).get('root_flow') else 0
KEY['e3_flow_cells']=(flowcells,flowchosen)
# consistency across seeds of root probes
KEY['e3_seed_root_spread']=max(max(r['work_counters']['root_probes']/r['operations'] for r in byf[(s,v)])-min(r['work_counters']['root_probes']/r['operations'] for r in byf[(s,v)]) for s in samples for v in V3)

# ---------------------------------------------------------------- E4 granularity
GR = rows(f"{S}/results/aidb_granularity/sweep/results.jsonl")
byg=collections.defaultdict(list)
for r in GR: byg[(r['dataset'],r['variant'])].append(r)
GD=['fb','osm','planet','covid','genome']
t=[]; KEY['e4']={}
tot_regions=0; tot_flow=0; tot_vp=0; maxforced=0
for R in (4096,16384,32768):
    for d in GD:
        def g(v,k):
            rs=byg[(d,f"{v}_r{R}")]; return statistics.median(r['work_counters'][k]/r['operations'] for r in rs)
        c=g('packed_rank','fence_probes'); vp=g('packed_rank_vp10','fence_probes'); ff=g('packed_rank_flow_forced','fence_probes'); fv=g('packed_rank_flow_vp10','fence_probes'); fa=g('packed_rank_fusion_auto','fence_probes')
        L=byg[(d,f"packed_rank_fusion_auto_r{R}")][0]['learnability']; ch=L['choices']; regs=L['regions']
        base=byg[(d,f"packed_rank_r{R}")]
        sp=paired(byg[(d,f"packed_rank_vp10_r{R}")],base); spf=paired(byg[(d,f"packed_rank_flow_forced_r{R}")],base); spa=paired(byg[(d,f"packed_rank_fusion_auto_r{R}")],base)
        rootp=g('packed_rank','root_probes')
        t.append([R, d, regs, fmt(rootp,2), fmt(c,3), fmt(vp,3), f"{pct(c,vp):+.1f}%", fmt(ff,3), f"{ff-c:+.3f}", fmt(fv,3), fmt(fa,3), f"{ch['none']}/{ch['flow']}/{ch['vp']}/{ch['both']}", f"{sp[0]:.3f} [{sp[1]:.2f}, {sp[2]:.2f}]", f"{spf[0]:.3f}", f"{spa[0]:.3f}"])
        tot_regions+=regs; tot_flow+=ch['flow']+ch['both']; tot_vp+=ch['vp']+ch['both']; maxforced=max(maxforced,abs(ff-c))
        KEY['e4'][f"{d}_r{R}"]=dict(control=c,vp10=vp,forced=ff,flow_vp10=fv,fusion=fa,choices=ch,regions=regs,root=rootp)
w("e4", table(["region keys","dataset","regions","root/op (binary)","fence control","fence vp10","Δ","fence flow forced","forced − control","fence flow_vp10 (forced)","fence fusion_auto","choices none/flow/vp/both","vp10 speedup","forced speedup","fusion speedup"], t))
KEY['e4_totals']=dict(regions=tot_regions,flow=tot_flow,vp=tot_vp,max_forced_diff=maxforced, runs=len(GR))
# 16384 and 32768 fusion_auto: transform calls
# ---------------------------------------------------------------- E5 full scale
FS = rows(f"{S}/results/aidb_fullscale/sweep/results.jsonl")
t=[]; KEY['e5']=[]
order=['packed_rank','packed_rank_root_raw','packed_rank_root_flow','packed_rank_root_vf004','packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_vp10_root_vf004','packed_rank_vp10_root_fusion']
for d in ('fb','planet'):
    base=[r for r in FS if r['dataset']==d and r['variant']=='packed_rank'][0]
    for v in order:
        for r in FS:
            if r['dataset']!=d or r['variant']!=v: continue
            L=r['learnability']; wc=lambda k:r['work_counters'][k]/r['operations']
            chosen = "binary" if v=='packed_rank' else ("binary(fallback)" if not L.get('root_model') else ("flow+" if L.get('root_flow') else "raw+")+("fences" if L.get('root_vp') else "ranks"))
            e=[L.get(k,0) for k in ('root_probes_binary','root_probes_raw','root_probes_flow','root_probes_vp_raw','root_probes_vp_flow')]
            tot=wc('root_probes')+wc('fence_probes')+wc('coordinate_probes'); btot=sum(base['work_counters'][k] for k in ('root_probes','fence_probes','coordinate_probes'))/base['operations']
            rec=dict(dataset=d,variant=v,mops=r['throughput_ops_s']/1e6,ratio=r['throughput_ops_s']/base['throughput_ops_s'],build_s=r['build_ns']/1e9,root=wc('root_probes'),fence=wc('fence_probes'),coord=wc('coordinate_probes'),total=tot,dtotal=pct(btot,tot),chosen=chosen,virt=L.get('root_virtual',0),est=e,meta=r['memory_before']['metadata_bytes']/r['initial_rows'],bpk=r['memory_before']['accounted_bytes']/r['initial_rows'],regions=L['regions'],vp=L.get('virtual_points',0),smooth_s=L.get('smoothing_ns',0)/1e9,transform_s=L.get('transform_ns',0)/1e9,tc=wc('transform_calls'),seed=r['seed'],ops=r['operations'])
            KEY['e5'].append(rec)
            t.append([d, v, fmt(rec['mops'],3), f"{rec['ratio']:.3f}", fmt(rec['build_s'],1), fmt(rec['root'],2), fmt(rec['fence'],2), fmt(rec['coord'],2), fmt(tot,2), f"{rec['dtotal']:+.1f}%", chosen, rec['virt'], "/".join(f"{x:.2f}" for x in e), fmt(rec['meta'],3), fmt(rec['smooth_s'],0), fmt(rec['tc'],4)])
w("e5", table(["dataset","variant","Mops","ratio vs control (1 seed)","build s","root/op","fence/op","coord/op","total/op","total Δ","chosen","virt. fences","est bin/raw/flow/vf/fus","meta B/key","smoothing s (thread)","transform_calls/op"], t))
KEY['e5_n']=len(FS)

# ---------------------------------------------------------------- E6 scores
PAPER = {  # Table 2 of the AIDB paper (Conf, Cov columns only; ConfI per index omitted)
 'RMSE':(0.19,1.00),'ME':(0.15,1.00),'CD':(0.28,1.00),'PLA-32':(0.50,1.00),'PLA-4096':(0.08,1.00),
 'PLA-32·PLA-4096':(0.71,0.47),'RMSE·ME':(0.25,0.82),'RMSE·CD':(0.60,0.47),'RMSE·PLA-32':(0.64,0.60),'RMSE·PLA-4096':(0.52,0.42),'ME·CD':(0.49,0.56),'ME·PLA-32':(0.61,0.60),'ME·PLA-4096':(0.50,0.42),'CD·PLA-32':(0.69,0.60),'CD·PLA-4096':(0.55,0.42),
 'RMSE·ME·CD':(0.59,0.42),'RMSE·ME·PLA-32':(0.63,0.51),'RMSE·ME·PLA-4096':(0.50,0.33),'RMSE·CD·PLA-32':(0.83,0.33),'RMSE·CD·PLA-4096':(0.70,0.16),'RMSE·PLA-32·PLA-4096':(0.73,0.24),'ME·CD·PLA-32':(0.78,0.38),'ME·CD·PLA-4096':(0.66,0.20),'ME·PLA-32·PLA-4096':(0.69,0.24),'CD·PLA-32·PLA-4096':(0.78,0.24)}
SCU=json.load(open(f"{S}/results/aidb/scores.json")); SCW=json.load(open(f"{S}/results/aidb_window/scores.json"))
def scoretab(SC, scope, tag):
    m=SC[scope]['metrics']; t=[]; match=0; cdmm=[]
    for name,(pc,pv) in PAPER.items():
        x=m.get(name)
        if x is None:
            for k in m:
                if set(k.split('·'))==set(name.split('·')): x=m[k]; break
        cov=x['coverage']; conf=x['conformance']
        ok = abs(round(cov,2)-pv)<0.005
        match+=ok
        d=x['per_variant']
        t.append([name, f"{pc:.2f}", f"{pv:.2f}", f"{conf:.3f}", f"{cov:.3f}", x['comparable_pairs'], x['incomparable_pairs'], "yes" if ok else f"no ({cov-pv:+.3f})", f"{d['packed_rank']:.3f}", f"{d['packed_rank_vp10']:.3f}", f"{d.get('packed_rank_flow_forced',float('nan')):.3f}", f"{d['sorted_vector']:.3f}"])
        if not ok: cdmm.append((name,cov,pv,x['comparable_pairs'],x['incomparable_pairs']))
    w(f"e6_{tag}_{scope}", table(["metric","paper Conf","paper Cov","our Conf (mean over variants)","our Cov","|C|","|U|","Cov matches paper","Conf_I packed_rank","Conf_I vp10","Conf_I flow forced","Conf_I sorted_vector"], t))
    return match, cdmm
KEY['e6']={}
for tag,SC in (('uniform',SCU),('window',SCW)):
    for scope in SC:
        mt,mm=scoretab(SC,scope,tag); KEY['e6'][f"{tag}/{scope}"]=dict(matches=mt,mismatches=mm)
# per-variant conformance for the scalar metrics, full scope, uniform
m=SCU['full']['metrics']; vs=list(m['RMSE']['per_variant'].keys())
t=[[v]+[f"{m[k]['per_variant'][v]:.3f}" for k in ('RMSE','ME','CD','PLA-32','PLA-4096')] for v in vs]
w("e6_variants_full", table(["variant","RMSE","ME","CD","PLA-32","PLA-4096"], t))
# hardness orderings (ranks) per scalar for full data
t=[]
for k,n in M:
    order=sorted(DS,key=lambda d:HU[d]['full'][k])
    t.append([n," < ".join(order)])
w("e6_orderings", table(["metric (full 200M keys, raw)","easiest → hardest"], t))

# ---------------------------------------------------------------- Throughput (final sweep)
V8=['packed_rank_vp10','packed_rank_root_raw','packed_rank_root_flow','packed_rank_root_vf4','packed_rank_root_fusion','packed_rank_vp10_root_fusion','raw_rank','sorted_vector']
t=[]
for s in samples:
    t.append([s]+[f"{RA[s][v]['speedup']:.3f} [{RA[s][v]['lo']:.2f}, {RA[s][v]['hi']:.2f}]" for v in V8])
w("e8_speed", table(["sample"]+[SH3.get(v,v) for v in V8], t))
# pooled geometric means
pooled={}
for v in V8:
    logs=[]
    for s in samples:
        for r in byf[(s,v)]:
            b=[x for x in byf[(s,'packed_rank')] if x['seed']==r['seed']]
            if b: logs.append(math.log(r['throughput_ops_s']/b[0]['throughput_ops_s']))
    pooled[v]=(math.exp(statistics.mean(logs)), math.exp(min(logs)), math.exp(max(logs)), len(logs))
KEY['e8_pooled']=pooled
# noise band: identical structures measured twice: root_vf4 vs root_fusion where both chose raw+fences (post-fix), root_raw vs packed_rank when fallback
same=[]; fb=[]
for s in samples:
    for r in byf[(s,'packed_rank_root_fusion')]:
        q=[x for x in byf[(s,'packed_rank_root_vf4')] if x['seed']==r['seed']][0]
        if r['learnability'].get('root_vp') and q['learnability'].get('root_vp') and not r['learnability'].get('root_flow') and abs(r['work_counters']['root_probes']-q['work_counters']['root_probes'])<1:
            same.append((s,r['seed'],r['throughput_ops_s']/q['throughput_ops_s']))
    for r in byf[(s,'packed_rank_root_raw')]:
        if not r['learnability'].get('root_model'):
            q=[x for x in byf[(s,'packed_rank')] if x['seed']==r['seed']][0]
            fb.append((s,r['seed'],r['throughput_ops_s']/q['throughput_ops_s']))
KEY['e8_noise_same_structure']=dict(n=len(same),min=min(x[2] for x in same),max=max(x[2] for x in same),pairs=same)
KEY['e8_noise_fallback']=dict(n=len(fb),min=min(x[2] for x in fb),max=max(x[2] for x in fb),pairs=fb)
# per-sample geometric mean of the same-structure ratio
gs=collections.defaultdict(list)
for s,seed,r in same: gs[s].append(math.log(r))
KEY['e8_noise_same_per_sample']={s:math.exp(statistics.mean(v)) for s,v in gs.items()}
gf=collections.defaultdict(list)
for s,seed,r in fb: gf[s].append(math.log(r))
KEY['e8_noise_fallback_per_sample']={s:math.exp(statistics.mean(v)) for s,v in gf.items()}
# ns per lookup and decoding ablation
t=[]
for s in samples:
    def ns(v): return statistics.median(1e9/r['throughput_ops_s'] for r in byf[(s,v)])
    p=ns('packed_rank'); rr=ns('raw_rank'); sv=ns('sorted_vector')
    t.append([s, fmt(p,1), fmt(rr,1), fmt(sv,1), f"{RA[s]['sorted_vector']['speedup']:.3f}", f"{RA[s]['raw_rank']['speedup']:.3f}", f"{(p-rr)/(p-sv)*100:.0f}%", f"{(rr-sv)/(p-sv)*100:.0f}%", fmt(RA[s]['packed_rank']['meta_bytes_per_key'],2), fmt(statistics.median(r['memory_before']['accounted_bytes']/r['initial_rows'] for r in byf[(s,'packed_rank')]),2), fmt(statistics.median(r['memory_before']['accounted_bytes']/r['initial_rows'] for r in byf[(s,'raw_rank')]),2)])
w("e8_decode", table(["sample","ns/lookup packed_rank","ns/lookup raw_rank","ns/lookup sorted_vector","sorted_vector ×","raw_rank ×","gap closed by removing decoding","gap remaining (routing)","meta B/key packed","B/key packed","B/key raw"], t))
KEY['e8_sv_range']=(min(RA[s]['sorted_vector']['speedup'] for s in samples), max(RA[s]['sorted_vector']['speedup'] for s in samples))
KEY['e8_rr_range']=(min(RA[s]['raw_rank']['speedup'] for s in samples), max(RA[s]['raw_rank']['speedup'] for s in samples))
# region-level vp10 paired speedups final
KEY['e8_vp10_final']={s:(RA[s]['packed_rank_vp10']['speedup'],RA[s]['packed_rank_vp10']['lo'],RA[s]['packed_rank_vp10']['hi']) for s in samples}
KEY['e8_vp10_final_include1']=sum(1 for s in samples if RA[s]['packed_rank_vp10']['lo']<=1.0<=RA[s]['packed_rank_vp10']['hi'])
KEY['e8_vf4_final']={s:(RA[s]['packed_rank_root_vf4']['speedup'],RA[s]['packed_rank_root_vf4']['lo'],RA[s]['packed_rank_root_vf4']['hi']) for s in samples}
KEY['e8_root_fusion_final']={s:(RA[s]['packed_rank_root_fusion']['speedup'],RA[s]['packed_rank_root_fusion']['lo'],RA[s]['packed_rank_root_fusion']['hi']) for s in samples}
# transform_calls in final flow variants & flow fraction
KEY['e3_transform_calls']={s:{v:statistics.median(r['work_counters']['transform_calls']/r['operations'] for r in byf[(s,v)]) for v in ('packed_rank_root_flow','packed_rank_root_fusion','packed_rank_vp10_root_fusion')} for s in samples}
KEY['e3_ns_per_lookup']={s:{v:statistics.median(1e9/r['throughput_ops_s'] for r in byf[(s,v)]) for v in V3+['raw_rank','sorted_vector']} for s in samples}

json.dump(KEY, open(f"{S}/results/aidb/guide_deep/scratch/key_numbers.json","w"), indent=1, default=str)
print("done", len(os.listdir(OUT)), "tables")
