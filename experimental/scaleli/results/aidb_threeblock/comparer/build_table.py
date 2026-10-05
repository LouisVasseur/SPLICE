"""Comparer: build the 20-row per-sample table for RESULT.md from armA.json, armB.json, verify/v_band.json.
Reads only; computes nothing new beyond min/argmin over stored numbers."""
import json, os
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = json.load(open(f'{D}/armA.json'))['rows']
B = json.load(open(f'{D}/armB.json'))['samples']
VB = {c['cell']: c for c in json.load(open(f'{D}/verify/v_band.json'))}
FIXED = ['baseline','T','V','T->V','G','G->T->V','T->G->V','G->V','G->T','T->G']
POOL = ['pool','poolV','pool_pwl','poolV_pwl']
DS = 'books fb osm covid genome history libio planet stack wise'.split()
def lab(var, k):
    return var if 'G' not in var.replace('pool','') or k is None else f'{var} k{k}'
out = []
for ds in DS:
  for mode in ['uniform','window']:
    s = [x for x in B if x['dataset']==ds and x['mode']==mode][0]
    base = s['per_k']['1']['variants']['baseline']['total_uncharged']
    cand = []; pool = []
    for k, pk in s['per_k'].items():
        for var, r in pk['variants'].items():
            usesG = r['gap_k'] > 0 or var in ('G','G->T->V','T->G->V','G->V','G->T','T->G')
            kk = int(k) if usesG else None
            if var in FIXED: cand.append((r['total_uncharged'], r['total_charged'], var, kk, r['model_probes']))
            elif var in POOL: pool.append((r['total_uncharged'], r['total_charged'], var, int(k), r['model_probes'], r['gap_k']))
    su = min(cand, key=lambda t: t[0]); sc = min(cand, key=lambda t: t[1])
    pu = min(pool, key=lambda t: t[0]); pc = min(pool, key=lambda t: t[1])
    rows = [r for r in A if r['dataset']==ds and r['mode']==mode]
    bu = min(rows, key=lambda r: (r['best']['total_uncharged'], r['k']))
    bc = min(rows, key=lambda r: (r['best']['total_charged'], r['k']))
    gr = [r for r in rows if r['k']>0]
    gbest = min(gr, key=lambda r: r['best']['total_uncharged'])
    cell = f"{ds}-{mode[0]} k={bu['k']} {bu['order']}"
    vb = VB.get(cell)
    if vb: band, src, mn = vb['band_c4'], 'wp', vb['gain_min_over_perts']
    else: band, src, mn = bu['band_spread'], 'V', None
    gain = bu['gain_rounds2plus']
    out.append(dict(sample=f'{ds}-{mode[0]}', baseline=base,
        single_unch=su[0], single_unch_charged=su[1], single_unch_cfg=lab(su[2], su[3]),
        single_charged=sc[1], single_charged_cfg=lab(sc[2], sc[3]),
        pool_unch=pu[0], pool_unch_cfg=pu[2], pool_charged=pc[1], pool_charged_cfg=pc[2],
        bf_unch=bu['best']['total_uncharged'], bf_charged=bu['best']['total_charged'],
        bf_cell=f"k={bu['k']} r{bu['best']['round']}", bf_round1_unch=bu['round1']['total_uncharged'],
        bf_best_charged_any=bc['best']['total_charged'],
        gain=gain, band=band, band_src=src, gain_min_over_perts=mn, exceeds=gain > band,
        gbest_unch=gbest['best']['total_uncharged'], gbest_cell=f"k={gbest['k']} {gbest['order']}",
        gbest_gain=gbest['gain_rounds2plus'],
        gbest_minus_nog=gbest['best']['total_uncharged']-bu['best']['total_uncharged']))
json.dump(out, open(f'{D}/comparer/per_sample.json','w'), indent=1)
for o in out:
    print(f"{o['sample']:10s} base {o['baseline']:.3f} | single {o['single_unch']:.3f}/{o['single_unch_charged']:.3f} {o['single_unch_cfg']:8s} | chg-best {o['single_charged']:.3f} {o['single_charged_cfg']} | pool {o['pool_unch']:.3f} {o['pool_unch_cfg']} / {o['pool_charged']:.3f} {o['pool_charged_cfg']} | BF {o['bf_unch']:.3f}/{o['bf_charged']:.3f} {o['bf_cell']} r1 {o['bf_round1_unch']:.3f} anychg {o['bf_best_charged_any']:.3f} | gain {o['gain']:.3f} band {o['band']:.3f}{o['band_src']} min {o['gain_min_over_perts']} {o['exceeds']} | G {o['gbest_unch']:.3f} {o['gbest_cell']} gain {o['gbest_gain']:.3f} +{o['gbest_minus_nog']:.3f}")
