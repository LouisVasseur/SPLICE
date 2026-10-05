#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
# Diagnostics for the joint (flow + virtual-point) objective at the ROOT.
# Standard library only. Nothing here writes into the scaleli tree.
import struct, math, json, sys
from pathlib import Path

DATA = Path(REPO + '/experimental/scaleli/data/samples')
NAMES = ['books','fb','osm','covid','genome','history','libio','planet','stack','wise']
REGION = 4096

def read_keys(p):
    b = p.read_bytes(); n = struct.unpack_from('<Q', b)[0]
    return list(struct.unpack_from('<%dQ' % n, b, 8))

def fences(keys):
    return [keys[i] for i in range(0, len(keys), REGION)]

def fit_line(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sxx = sum((xi-mx)**2 for xi in x); sxy = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
    a = sxy/sxx if sxx > 0 else 0.0
    return a, my - a*mx

def errs(x, y, a, b):
    r = [a*xi + b - yi for xi, yi in zip(x, y)]
    return max(abs(v) for v in r), math.sqrt(sum(v*v for v in r)/len(r))

def budget_bound(z, e):
    """Rigorous lower bound on the number of virtual points needed so that SOME
    refitted line over slot ranks has max |a z_i + b - s_i| <= e.
    Derivation: for any i<j, s_j-s_i >= j-i and a(z_j-z_i) = (s_j-s_i)+(err_j-err_i)
    >= (j-i)-2e, so a >= max_{i<j} (j-i-2e)/(z_j-z_i); and #virtual = (s_last-s_first)
    -(n-1) >= a*range - 2e - (n-1)."""
    n = len(z); best = 0.0
    for i in range(n):
        zi = z[i]
        for j in range(i+1+int(2*e), n):
            d = z[j]-zi
            if d > 0:
                v = (j-i-2*e)/d
                if v > best: best = v
    return best*(z[-1]-z[0]) - (n-1) - 2*e

def best_flow(x, y, grid):
    """z = c0*tanh(g0*x) + c1*tanh(g1*x); target y ~ a*z+b. Since a absorbs the c
    scale, fit y ~ C0*t0 + C1*t1 + b by exact linear least squares for each (g0,g1)."""
    n = len(x)
    T = {}
    for g in grid:
        T[g] = [math.tanh(g*xi) for xi in x]
    best = None
    for ia, g0 in enumerate(grid):
        t0 = T[g0]
        for g1 in grid[ia:]:
            t1 = T[g1]
            # normal equations for [t0, t1, 1]
            s00 = s01 = s11 = s0 = s1 = s0y = s1y = 0.0
            for k in range(n):
                a0 = t0[k]; a1 = t1[k]; yy = y[k]
                s00 += a0*a0; s01 += a0*a1; s11 += a1*a1
                s0 += a0; s1 += a1; s0y += a0*yy; s1y += a1*yy
            sy = sum(y)
            M = [[s00, s01, s0], [s01, s11, s1], [s0, s1, float(n)]]
            v = [s0y, s1y, sy]
            c = solve3(M, v)
            if c is None: continue
            C0, C1, B = c
            z = [C0*t0[k] + C1*t1[k] + B for k in range(n)]
            sse = sum((z[k]-y[k])**2 for k in range(n))
            if best is None or sse < best[0]:
                best = (sse, g0, g1, C0, C1, B)
    return best

def solve3(M, v):
    import copy
    A = [row[:] + [v[i]] for i, row in enumerate(M)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-18: return None
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        for r in range(3):
            if r == c: continue
            f = A[r][c]/pv
            for k in range(c, 4): A[r][k] -= f*A[c][k]
    return [A[i][3]/A[i][i] for i in range(3)]

grid = [10**(-2 + 6*k/47) for k in range(48)]  # 1e-2 .. 1e4, 48 points
out = {}
for name in NAMES:
    p = DATA / ('%s_2M_uniform_s42' % name)
    if not p.exists():
        print('missing', p, file=sys.stderr); continue
    keys = read_keys(p)
    f = fences(keys)
    n = len(f)
    origin = f[0]; span = max(1, f[-1]-origin)
    x = [(k-origin)/span for k in f]           # the index's raw root feature
    y = [float(i) for i in range(n)]           # plain ranks
    a, b = fit_line(x, y)
    mx_lin, rms_lin = errs(x, y, a, b)
    bf = best_flow(x, y, grid)
    sse, g0, g1, C0, C1, B = bf
    z = [C0*math.tanh(g0*xi) + C1*math.tanh(g1*xi) + B for xi in x]
    mono_fences = all(z[i+1] > z[i] for i in range(n-1))
    # monotone on a fine grid over the whole key range?
    fine = [i/4000.0 for i in range(4001)]
    zf = [C0*math.tanh(g0*t) + C1*math.tanh(g1*t) + B for t in fine]
    mono_fine = all(zf[i+1] >= zf[i] for i in range(len(fine)-1))
    a2, b2 = fit_line(z, y)
    mx_flow, rms_flow = errs(z, y, a2, b2)
    row = {'n': n, 'line_maxerr': mx_lin, 'line_rms': rms_lin,
           'flow_maxerr': mx_flow, 'flow_rms': rms_flow, 'g0': g0, 'g1': g1,
           'C0': C0, 'C1': C1, 'mono_fences': mono_fences, 'mono_range': mono_fine}
    for e in (1, 4, 16):
        row['B_raw_e%d' % e] = budget_bound(x, e)
        row['B_flow_e%d' % e] = budget_bound(z, e) if mono_fences else None
    out[name] = row
    print(name, json.dumps(row), flush=True)

Path(REPO + '/experimental/scaleli/results/aidb_joint/diag.json').write_text(json.dumps(out, indent=1))
