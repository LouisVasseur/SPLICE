import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import sys, time, random, json, math
sys.path.insert(0,REPO + '/experimental/scaleli/tools')
import train_flow as T
from pathlib import Path

name = sys.argv[1]
SAMP = Path(REPO + '/experimental/scaleli/data/samples')/f'{name}_2M_uniform_s42'
OUT = Path(REPO + '/experimental/scaleli/results/aidb_flowv2/training_sweep/out')

keys = T.read_keys(SAMP,'uint64')
N = len(keys)
rng = random.Random(1000000007)
# held-out: 250k keys drawn with a different seed
hrng = random.Random(777)
held = sorted(hrng.sample(keys, 250000))
raw = [(k-held[0])/max(1,held[-1]-held[0]) for k in held]
tcd_raw = T.tail_conflict_degree(raw)

def trainset(m):
    return sorted(rng.sample(keys,m)) if m < N else list(keys)

CFG = []
def add(sample,steps,shifts,lr,mono,tag):
    CFG.append(dict(sample=sample,steps=steps,shifts=shifts,lr=lr,monotone=mono,tag=tag))

# A: baseline reproduction (what aidb_pipeline.py used)
add(4096,200,64,0.05,True,'baseline-pipeline')
add(4096,200,64,0.05,False,'baseline-nomono')
# B: main grid, 2000 steps
for sh in (8,64,512,65536):
    for mo in (True,False):
        add(4096,2000,sh,0.05,mo,'grid')
for sh in (64,512):
    for mo in (True,False):
        add(4096,2000,sh,0.3,mo,'grid-bigLR')
# C: long training
add(4096,10000,64,0.05,False,'long')
add(4096,10000,512,0.3,False,'long')
# D: more training data
add(65536,200,64,0.05,False,'data65k')
add(65536,200,512,0.3,False,'data65k')
add(200000,200,64,0.05,False,'data200k')
add(200000,200,512,0.3,False,'data200k')

cache={}
res=[]
for c in CFG:
    if c['sample'] not in cache: cache[c['sample']]=trainset(c['sample'])
    tk=cache[c['sample']]
    t0=time.time()
    try:
        flow,info=T.train(tk,c['shifts'],c['steps'],c['lr'],1000000007,1e-3,log=lambda s:None,monotone=c['monotone'])
        el=time.time()-t0
        z=sorted(flow.transform(float(k)) for k in held)
        tcd=T.tail_conflict_degree(z)
        unord=sum(1 for i in range(1,len(held)) if flow.transform(float(held[i]))<flow.transform(float(held[i-1])))
        r=dict(c, nll=info['best_nll'], tcd_raw=tcd_raw, tcd_flow=tcd,
               ratio=(tcd+1)/(tcd_raw+1), seconds=round(el,2),
               unordered=unord, w0=flow.w0, w1=flow.w1, mean=flow.mean, var=flow.var)
    except Exception as e:
        r=dict(c, error=str(e), tcd_raw=tcd_raw, seconds=round(time.time()-t0,2))
    res.append(r)
    print(name, json.dumps({k:v for k,v in r.items() if k not in ('w0','w1','mean','var')}), flush=True)

ok=[r for r in res if 'error' not in r]
best=min(ok,key=lambda r:(r['tcd_flow'], r['nll']))
fl=T.Flow(best['w0'],best['w1'],best['mean'],best['var'])
fl.save(OUT/f'{name}_2D2H2L_best.txt')
json.dump(dict(dataset=name,keys=N,held_out=len(held),tcd_raw=tcd_raw,best=best,runs=res),
          open(OUT/f'{name}_sweep.json','w'), indent=1)
print('DONE',name,'best',best['tag'],best['tcd_flow'],'raw',tcd_raw, flush=True)
