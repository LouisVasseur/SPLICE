#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
# Does allowing a BIAS inside each tanh (two independently placed bends) rescue osm?
# Deployed NFL format has no bias, so both units share the centre x=0 (= the data minimum).
import struct, math, json
from pathlib import Path
DATA=Path(REPO + '/experimental/scaleli/data/samples')
REGION=4096
def read_keys(p):
    b=p.read_bytes(); n=struct.unpack_from('<Q',b)[0]
    return list(struct.unpack_from('<%dQ'%n,b,8))
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
    run=1; mrun=1
    for i in range(1,len(z)):
        if z[i]==z[i-1]: run+=1; mrun=max(mrun,run)
        else: run=1
    lo,hi=0.0,float(len(z))
    if B(z,hi)>lam: return hi
    for _ in range(28):
        mid=(lo+hi)/2
        if B(z,mid)>lam: lo=mid
        else: hi=mid
    return max(hi,(mrun-1)/2.0)
gs=[10**(-1+5*k/11) for k in range(12)]
ms=[k/11.0 for k in range(12)]
out={}
for name in ('osm','planet','books'):
    keys=read_keys(DATA/('%s_2M_uniform_s42'%name)); f=[keys[i] for i in range(0,len(keys),REGION)]
    n=len(f); origin=f[0]; span=max(1,f[-1]-origin)
    x=[(k-origin)/span for k in f]; y=[float(i) for i in range(n)]; sy=sum(y); lam=4*n
    cache={}
    def basis(g,m):
        key=(g,m)
        if key not in cache: cache[key]=[math.tanh(g*(xi-m)) for xi in x]
        return cache[key]
    best=None
    for g0 in gs:
        for m0 in ms:
            t0=basis(g0,m0)
            for g1 in gs:
                for m1 in ms:
                    t1=basis(g1,m1)
                    s00=s01=s11=s0=s1=s0y=s1y=0.0
                    for k in range(n):
                        a0=t0[k]; a1=t1[k]; yy=y[k]
                        s00+=a0*a0; s01+=a0*a1; s11+=a1*a1; s0+=a0; s1+=a1; s0y+=a0*yy; s1y+=a1*yy
                    c=solve3([[s00,s01,s0],[s01,s11,s1],[s0,s1,float(n)]],[s0y,s1y,sy])
                    if c is None: continue
                    C0,C1,Bc=c
                    z=[C0*t0[k]+C1*t1[k]+Bc for k in range(n)]
                    if not all(z[i+1]>z[i] for i in range(n-1)): continue
                    sse=sum((z[k]-y[k])**2 for k in range(n))
                    if best is None or sse<best[0]: best=(sse,g0,m0,g1,m1,z)
    sse,g0,m0,g1,m1,z=best
    mx=max(abs(z[k]-y[k]) for k in range(n))
    out[name]={'rms_bias_flow':math.sqrt(sse/n),'maxerr_bias_flow':mx,'estar_bias_flow':estar(z,lam),
               'g0':g0,'m0':m0,'g1':g1,'m1':m1}
    print(name,json.dumps(out[name]),flush=True)
Path(REPO + '/experimental/scaleli/results/aidb_joint/diag5.json').write_text(json.dumps(out,indent=1))
