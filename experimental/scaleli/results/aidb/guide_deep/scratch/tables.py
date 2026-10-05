import json, math, collections, struct
names=['books','fb','osm','covid','genome','history','libio','planet','stack','wise']
HU=json.load(open('results/aidb/hardness.json')); HW=json.load(open('results/aidb_window/hardness.json'))
DU=json.load(open('results/aidb/hardness_details.json')); DW=json.load(open('results/aidb_window/hardness_details.json'))
P=json.load(open('results/aidb/provenance.json')); ST=json.load(open('results/aidb/guide_deep/scratch/data_stats.json'))
TU=json.load(open('results/aidb/flows/training_report.json')); TW=json.load(open('results/aidb_window/flows/training_report.json'))
def f(x,nd=1): return f'{x:,.{nd}f}'
def i(x): return f'{int(x):,}'
def pct(a,b): return f'{100*(b-a)/a:+.1f}%'
out=[]
# T1 provenance
out.append('## T1 provenance')
out.append('| dataset | served sorted? | download sha256 | .sorted sha256 | uniform sample sha256 | window sample sha256 |')
out.append('|---|---|---|---|---|---|')
for d in names:
    p=P[d]; wm=json.load(open(f'data/samples/{d}_2M_window_s42.manifest.json'))
    srt='yes' if p['sort_audit']['sorted'] else 'no (sorted copy written)'
    out.append(f"| {d} | {srt} | `{p['sha256']}` | {'`'+p['sample']['source_sha256']+'`' if not p['sort_audit']['sorted'] else '(same file)'} | `{p['sample']['sample_sha256']}` | `{wm['sample_sha256']}` |")
# T2 master full table sorted by PLA-32
out.append('\n## T2 master full-file table sorted by PLA-32')
out.append('| rank | dataset | RMSE | ME | CD | PLA-32 | PLA-4096 | RMSE/n | ME/n |')
out.append('|---|---|---|---|---|---|---|---|---|')
for r,d in enumerate(sorted(names,key=lambda d:HU[d]['full']['pla_32']),1):
    b=HU[d]['full']; mark=' (paper: h_l=105k, h_g=468)' if d=='history' else (' (paper: h_l=146k, h_g=639)' if d=='libio' else '')
    out.append(f"| {r} | {d}{mark} | {f(b['rmse'])} | {f(b['max_error'])} | {i(b['conflict_degree'])} | {i(b['pla_32'])} | {i(b['pla_4096'])} | {b['rmse']/2e8:.4f} | {b['max_error']/2e8:.4f} |")
# T3 key stats
out.append('\n## T3 key statistics (full sorted files)')
out.append('| dataset | min | max | range = max-min | log2(range) | p50 | p99 | p99.99 | mean gap = range/(n-1) |')
out.append('|---|---|---|---|---|---|---|---|---|')
for d in names:
    s=ST[d]; mn=s['pct']['0']; mx=s['pct']['1.0']; rg=mx-mn
    out.append(f"| {d} | {i(mn)} | {i(mx)} | {i(rg)} | {math.log2(rg):.2f} | {i(s['pct']['0.5'])} | {i(s['pct']['0.99'])} | {i(s['pct']['0.9999'])} | {rg/(2e8-1):,.3f} |")
# T4 sample hardness
for label,H,D in (('uniform',HU,DU),('window',HW,DW)):
    out.append(f'\n## T4 {label} sample hardness')
    out.append('| dataset | scope | keys | RMSE | ME | CD | PLA-32 | PLA-4096 | extra |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for d in names:
        for sc in ('sample','sample_flow','sample_csv','sample_flow_csv'):
            b=H[d][sc]; ex=''
            if sc=='sample_flow': ex=f"dup={D[d]['sample_flow'].get('duplicates','n/a')}, unordered={D[d]['sample_flow'].get('unordered_pairs')}"
            if sc in ('sample_csv','sample_flow_csv'): ex=f"vp={i(D[d][sc]['virtual_points'])}, regions={D[d][sc]['regions']}"
            keys=b.get('keys', D[d]['sample']['keys'] + (D[d][sc]['virtual_points'] if 'csv' in sc else 0))
            out.append(f"| {d} | {sc} | {i(keys)} | {f(b['rmse'])} | {f(b['max_error'])} | {i(b['conflict_degree'])} | {i(b['pla_32'])} | {i(b['pla_4096'])} | {ex} |")
# T5 tail conflicts per region from sweeps
def tails(res):
    rows=[json.loads(l) for l in open(res+'/sweep/results.jsonl')]
    raw={}; flow={}; fr={}; vp={}
    for r in rows:
        L=r.get('learnability'); 
        if not L: continue
        d=r['dataset']; v=r['variant']
        if v=='raw_rank': raw[d]=L['tail_conflicts_raw_mean']
        if v=='packed_rank_flow_forced': flow[d]=L['tail_conflicts_flow_mean']
        if v=='packed_rank_flow': fr[d]=L['flow_regions']
        if v=='packed_rank_vp10': vp[d]=L['virtual_points']
    return raw,flow,fr,vp
ru,fu,fru,vpu=tails('results/aidb'); rw,fw,frw,vpw=tails('results/aidb_window')
out.append('\n## T5 mean per-region tail conflict degree (489 regions of 4,096 keys), from sweep learnability blocks')
out.append('| dataset | uniform raw | uniform flow (forced) | uniform regions where auto-switch kept the flow | window raw | window flow (forced) | window flow regions kept |')
out.append('|---|---|---|---|---|---|---|')
for d in names:
    out.append(f"| {d} | {ru[d]:.4f} | {fu[d]:.4f} | {fru[d]}/489 | {rw[d]:.4f} | {fw[d]:.4f} | {frw[d]}/489 |")
# whole-sample tail CD from training report
out.append('\n## T6 flow training (train_flow.py --monotone, 4,096 training keys, 200 Adam steps)')
out.append('| dataset | mode | best NLL | train s | whole-sample tail CD raw | tail CD transformed | unordered pairs |')
out.append('|---|---|---|---|---|---|---|')
for lab,T in (('uniform',TU),('window',TW)):
    for r in T:
        out.append(f"| {r['dataset']} | {lab} | {r['best_nll']:.4f} | {r['train_seconds']:.2f} | {r['tail_conflict_degree_raw']} | {r['tail_conflict_degree_transformed']} | {r['unordered_transformed_pairs']} |")
# T7 flow effect
out.append('\n## T7 effect of the flow transform')
out.append('| dataset | scope | RMSE before | RMSE after | dRMSE | CD before | CD after | PLA-32 before | PLA-32 after | dPLA-32 | PLA-4096 before | after |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
for d in names:
    for lab,H,a,b in (('full (uniform-trained flow)',HU,'full','full_flow'),('uniform sample',HU,'sample','sample_flow'),('window sample',HW,'sample','sample_flow')):
        x=H[d][a]; y=H[d][b]
        out.append(f"| {d} | {lab} | {f(x['rmse'])} | {f(y['rmse'])} | {pct(x['rmse'],y['rmse'])} | {i(x['conflict_degree'])} | {i(y['conflict_degree'])} | {i(x['pla_32'])} | {i(y['pla_32'])} | {pct(x['pla_32'],y['pla_32'])} | {i(x['pla_4096'])} | {i(y['pla_4096'])} |")
# T8 csv effect
out.append('\n## T8 effect of CSV-style virtual points (alpha=0.1 per region of 4,096)')
out.append('| dataset | mode | virtual points | of budget 199,707 | RMSE sample | RMSE sample_csv | dRMSE | CD | CD csv | PLA-32 | PLA-32 csv | dPLA-32 | PLA-4096 | csv |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for d in names:
    for lab,H,D in (('uniform',HU,DU),('window',HW,DW)):
        x=H[d]['sample']; y=H[d]['sample_csv']; vp=D[d]['sample_csv']['virtual_points']
        out.append(f"| {d} | {lab} | {i(vp)} | {100*vp/199707:.1f}% | {f(x['rmse'])} | {f(y['rmse'])} | {pct(x['rmse'],y['rmse'])} | {i(x['conflict_degree'])} | {i(y['conflict_degree'])} | {i(x['pla_32'])} | {i(y['pla_32'])} | {pct(x['pla_32'],y['pla_32'])} | {i(x['pla_4096'])} | {i(y['pla_4096'])} |")
# T9 ratios full/sample
out.append('\n## T9 full-file / sample ratios (n ratio = 100)')
out.append('| dataset | RMSE full/uniform | ME full/uniform | CD full/uniform | PLA-32 full/uniform | PLA-4096 full/uniform | RMSE full/window | PLA-32 full/window | CD full / CD window |')
out.append('|---|---|---|---|---|---|---|---|---|')
for d in names:
    F=HU[d]['full']; U=HU[d]['sample']; W=HW[d]['sample']
    out.append(f"| {d} | {F['rmse']/U['rmse']:.2f} | {F['max_error']/U['max_error']:.2f} | {F['conflict_degree']/U['conflict_degree']:.2f} | {F['pla_32']/U['pla_32']:.1f} | {F['pla_4096']/U['pla_4096']:.1f} | {F['rmse']/W['rmse']:.1f} | {F['pla_32']/W['pla_32']:.1f} | {F['conflict_degree']/W['conflict_degree']:.2f} |")
# T10 uniform vs window sample
out.append('\n## T10 uniform vs window sample, raw keys')
out.append('| dataset | uniform min | uniform max | window min | window max | window span / full range | uniform PLA-32 | window PLA-32 | uniform CD | window CD | uniform RMSE | window RMSE | uniform tailCD | window tailCD |')
out.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for d in names:
    U=HU[d]['sample']; W=HW[d]['sample']; du=DU[d]['sample']; dw=DW[d]['sample']; s=ST[d]; rg=s['pct']['1.0']-s['pct']['0']
    out.append(f"| {d} | {i(du['min'])} | {i(du['max'])} | {i(dw['min'])} | {i(dw['max'])} | {100*(dw['max']-dw['min'])/rg:.3f}% | {i(U['pla_32'])} | {i(W['pla_32'])} | {i(U['conflict_degree'])} | {i(W['conflict_degree'])} | {f(U['rmse'])} | {f(W['rmse'])} | {ru[d]:.2f} | {rw[d]:.2f} |")
# constants
out.append(f"\nn/sqrt(12) = {2e8/math.sqrt(12):,.3f}; fb full rmse = {HU['fb']['full']['rmse']:,.3f}; ratio = {HU['fb']['full']['rmse']/(2e8/math.sqrt(12)):.6f}")
out.append(f"budget: 488*floor(0.1*4096)+floor(0.1*1152) = {488*409+115}")
# fb flow worked example
w=open('results/aidb/flows/fb_2D2H2L.txt').read().split('\n'); mean,var=map(float,w[1].split('\t')); w0=[float(x) for x in (w[3]+'\t'+w[4]).split('\t') if x]; w1=[float(x) for x in (w[6]+'\t'+w[7]).split('\t') if x]
out.append(f"fb flow: mean={mean}, var={var}, w0={w0}, w1={w1}, span=mean+64*var={mean+64*var}")
for k in (97995, 38752453239, 77308811965):
    x=(k-mean)/var; u=[x*w0[0],x*w0[1]]; h=[math.tanh(u[0]),math.tanh(u[1])]; v=[w1[0]+w1[1],w1[2]+w1[3]]; z=h[0]*v[0]+h[1]*v[1]
    out.append(f"  k={k}: x={x:.6f} u={u[0]:.6f},{u[1]:.6f} h={h[0]:.6f},{h[1]:.6f} v={v[0]:.6f},{v[1]:.6f} z={z:.6f}")
open('results/aidb/guide_deep/scratch/tables.md','w').write('\n'.join(out)); print('\n'.join(out))
