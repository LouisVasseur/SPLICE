#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import struct, math, json, sys
from pathlib import Path
DATA = Path(REPO + '/experimental/scaleli/data/samples')
NAMES = ['books','fb','osm','covid','genome','history','libio','planet','stack','wise']
REGION = 4096
def read_keys(p):
    b = p.read_bytes(); n = struct.unpack_from('<Q', b)[0]
    return list(struct.unpack_from('<%dQ' % n, b, 8))
def solve3(M, v):
    A = [row[:] + [v[i]] for i, row in enumerate(M)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-18: return None
        A[c], A[p] = A[p], A[c]; pv = A[c][c]
        for r in range(3):
            if r == c: continue
            f = A[r][c]/pv
            for k in range(c, 4): A[r][k] -= f*A[c][k]
    return [A[i][3]/A[i][i] for i in range(3)]
def slope_bound(z, e):
    n = len(z); best = 0.0; k = int(math.ceil(2*e))
    for i in range(n):
        zi = z[i]
        for j in range(i+1+k, n):
            d = z[j]-zi
            if d > 0:
                v = (j-i-2*e)/d
                if v > best: best = v
    return best
def B(z, e):
    return slope_bound(z, e)*(z[-1]-z[0]) - (len(z)-1) - 2*e
def estar(z, lam):
    """smallest max-error e achievable with at most lam virtual points (bisection on the bound)"""
    lo, hi = 0.0, float(len(z))
    if B(z, hi) > lam: return hi
    for _ in range(40):
        mid = (lo+hi)/2
        if B(z, mid) > lam: lo = mid
        else: hi = mid
    return hi
def mono(C0, C1, g0, g1, steps=4000):
    prev = None
    for t in range(steps+1):
        x = t/steps
        v = C0*math.tanh(g0*x) + C1*math.tanh(g1*x)
        if prev is not None and v < prev - 1e-15: return False
        prev = v
    return True
def best_flow(x, y, grid, require_mono):
    n = len(x); T = {g: [math.tanh(g*xi) for xi in x] for g in grid}
    sy = sum(y); best = None
    for ia, g0 in enumerate(grid):
        t0 = T[g0]
        for g1 in grid[ia:]:
            t1 = T[g1]
            s00=s01=s11=s0=s1=s0y=s1y=0.0
            for k in range(n):
                a0=t0[k]; a1=t1[k]; yy=y[k]
                s00+=a0*a0; s01+=a0*a1; s11+=a1*a1; s0+=a0; s1+=a1; s0y+=a0*yy; s1y+=a1*yy
            c = solve3([[s00,s01,s0],[s01,s11,s1],[s0,s1,float(n)]], [s0y,s1y,sy])
            if c is None: continue
            C0,C1,Bc = c
            if require_mono and not mono(C0,C1,g0,g1): continue
            z=[C0*t0[k]+C1*t1[k]+Bc for k in range(n)]
            sse=sum((z[k]-y[k])**2 for k in range(n))
            if best is None or sse<best[0]: best=(sse,g0,g1,C0,C1,Bc)
    return best
grid=[10**(-2+6*k/47) for k in range(48)]
lam_mult=4
res={}
for name in NAMES:
    keys=read_keys(DATA/('%s_2M_uniform_s42'%name)); f=[keys[i] for i in range(0,len(keys),REGION)]
    n=len(f); origin=f[0]; span=max(1,f[-1]-origin)
    x=[(k-origin)/span for k in f]; y=[float(i) for i in range(n)]
    lam=lam_mult*n
    bf=best_flow(x,y,grid,True)
    sse,g0,g1,C0,C1,Bc=bf
    z=[C0*math.tanh(g0*xi)+C1*math.tanh(g1*xi)+Bc for xi in x]
    r={'n':n,'lambda':lam,'estar_raw':estar(x,lam),'estar_flow':estar(z,lam),
       'estar_raw_nobudget':estar(x,0),'estar_flow_nobudget':estar(z,0),
       'B1_raw':B(x,1),'B1_flow':B(z,1),'g0':g0,'g1':g1,'ratio':C1/C0 if C0 else None,'sse_mono':sse}
    res[name]=r; print(name, json.dumps(r), flush=True)
Path(REPO + '/experimental/scaleli/results/aidb_joint/diag2.json').write_text(json.dumps(res,indent=1))
