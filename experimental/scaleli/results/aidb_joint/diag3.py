#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
# Is the SSE-optimal in-family flow also the flow that frees the most virtual-point budget?
import struct, math, json
from pathlib import Path
DATA = Path(REPO + '/experimental/scaleli/data/samples')
REGION = 4096
def read_keys(p):
    b=p.read_bytes(); n=struct.unpack_from('<Q',b)[0]
    return list(struct.unpack_from('<%dQ'%n,b,8))
def slope_bound(z,e):
    n=len(z); best=0.0; k=int(math.ceil(2*e))
    for i in range(n):
        zi=z[i]
        for j in range(i+1+k,n):
            d=z[j]-zi
            if d>0:
                v=(j-i-2*e)/d
                if v>best: best=v
    return best
def B(z,e): return slope_bound(z,e)*(z[-1]-z[0])-(len(z)-1)-2*e
def estar(z,lam):
    lo,hi=0.0,float(len(z))
    if B(z,hi)>lam: return hi
    for _ in range(30):
        mid=(lo+hi)/2
        if B(z,mid)>lam: lo=mid
        else: hi=mid
    return hi
def mono_ok(r,g0,g1,steps=2000):
    prev=None
    for t in range(steps+1):
        x=t/steps; v=math.tanh(g0*x)+r*math.tanh(g1*x)
        if prev is not None and v<prev-1e-15: return False
        prev=v
    return True
gs=[10**(-2+6*k/13) for k in range(14)]
rs=[-0.99,-0.9,-0.7,-0.5,-0.3,-0.1,0.0,0.3,1.0,3.0,10.0]
out={}
for name in ('planet','osm','books'):
    keys=read_keys(DATA/('%s_2M_uniform_s42'%name)); f=[keys[i] for i in range(0,len(keys),REGION)]
    n=len(f); origin=f[0]; span=max(1,f[-1]-origin)
    x=[(k-origin)/span for k in f]; lam=4*n
    cands=[]
    for g0 in gs:
        t0=[math.tanh(g0*xi) for xi in x]
        for g1 in gs:
            t1=[math.tanh(g1*xi) for xi in x]
            for r in rs:
                if not mono_ok(r,g0,g1): continue
                z=[t0[k]+r*t1[k] for k in range(n)]
                if not (z[-1]>z[0]): continue
                cands.append((B(z,4.0),g0,g1,r))
    cands.sort()
    top=[]
    for c in cands[:8]:
        _,g0,g1,r=c
        z=[math.tanh(g0*xi)+r*math.tanh(g1*xi) for xi in x]
        top.append({'B4':c[0],'estar':estar(z,lam),'g0':g0,'g1':g1,'r':r})
    out[name]={'n':n,'lambda':lam,'best_B_directed':top[:4],'n_monotone_candidates':len(cands)}
    print(name,json.dumps(out[name]),flush=True)
Path(REPO + '/experimental/scaleli/results/aidb_joint/diag3.json').write_text(json.dumps(out,indent=1))
