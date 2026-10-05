#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
# Tie-aware version of the budget bound + collapse audit.
import struct, math, json
from pathlib import Path
DATA=Path(REPO + '/experimental/scaleli/data/samples')
REGION=4096
NAMES=['books','fb','osm','covid','genome','history','libio','planet','stack','wise']
def read_keys(p):
    b=p.read_bytes(); n=struct.unpack_from('<Q',b)[0]
    return list(struct.unpack_from('<%dQ'%n,b,8))
def tie_floor(z):
    """If m fences share one feature value they share one prediction, so max error >= (m-1)/2
    no matter how many virtual points are spent."""
    best=1; run=1
    for i in range(1,len(z)):
        if z[i]==z[i-1]: run+=1
        else:
            if run>best: best=run
            run=1
    if run>best: best=run
    return (best-1)/2.0, best
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
    tf,_=tie_floor(z)
    lo,hi=0.0,float(len(z))
    if B(z,hi)>lam: return hi
    for _ in range(30):
        mid=(lo+hi)/2
        if B(z,mid)>lam: lo=mid
        else: hi=mid
    return max(hi,tf)
def solve3(M,v):
    A=[row[:]+[v[i]] for i,row in enumerate(M)]
    for c in range(3):
        p=max(range(c,3),key=lambda r:abs(A[r][c]))
        if abs(A[p][c])<1e-18: return None
        A[c],A[p]=A[p],A[c]; pv=A[c][c]
        for r in range(3):
            if r==c: continue
            fq=A[r][c]/pv
            for k in range(c,4): A[r][k]-=fq*A[c][k]
    return [A[i][3]/A[i][i] for i in range(3)]
def strict_mono(C0,C1,g0,g1,x):
    z=[C0*math.tanh(g0*xi)+C1*math.tanh(g1*xi) for xi in x]
    return all(z[i+1]>z[i] for i in range(len(z)-1)), z
grid=[10**(-2+6*k/47) for k in range(48)]
out={}
for name in NAMES:
    keys=read_keys(DATA/('%s_2M_uniform_s42'%name)); f=[keys[i] for i in range(0,len(keys),REGION)]
    n=len(f); origin=f[0]; span=max(1,f[-1]-origin)
    x=[(k-origin)/span for k in f]; y=[float(i) for i in range(n)]; lam=4*n; sy=sum(y)
    T={g:[math.tanh(g*xi) for xi in x] for g in grid}
    best=None; best_collapse=None
    for ia,g0 in enumerate(grid):
        t0=T[g0]
        for g1 in grid[ia:]:
            t1=T[g1]
            s00=s01=s11=s0=s1=s0y=s1y=0.0
            for k in range(n):
                a0=t0[k]; a1=t1[k]; yy=y[k]
                s00+=a0*a0; s01+=a0*a1; s11+=a1*a1; s0+=a0; s1+=a1; s0y+=a0*yy; s1y+=a1*yy
            c=solve3([[s00,s01,s0],[s01,s11,s1],[s0,s1,float(n)]],[s0y,s1y,sy])
            if c is None: continue
            C0,C1,Bc=c
            ok,z=strict_mono(C0,C1,g0,g1,x)
            sse=sum((z[k]+Bc-y[k])**2 for k in range(n))
            if not ok:
                if best_collapse is None or sse<best_collapse[0]: best_collapse=(sse,g0,g1,C0,C1)
                continue
            if best is None or sse<best[0]: best=(sse,g0,g1,C0,C1,[v+Bc for v in z])
    tf_raw,run_raw=tie_floor(x)
    row={'n':n,'lambda':lam,'raw_estar':estar(x,lam),'raw_tie_run':run_raw}
    if best:
        sse,g0,g1,C0,C1,z=best
        tf,run=tie_floor(z)
        row.update({'flow_estar':estar(z,lam),'flow_rms':math.sqrt(sse/n),'flow_tie_run':run,'g0':g0,'g1':g1,'ratio':C1/C0})
    else:
        row['flow_estar']=None
    out[name]=row; print(name,json.dumps(row),flush=True)
Path(REPO + '/experimental/scaleli/results/aidb_joint/diag4.json').write_text(json.dumps(out,indent=1))
