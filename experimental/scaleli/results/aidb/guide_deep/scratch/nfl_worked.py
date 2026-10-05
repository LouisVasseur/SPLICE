#!/usr/bin/env python3
"""Worked numeric examples for 01_nfl.md (standard library only)."""
import math

print("=== A. Conflict degree / tail conflict degree on 10 keys ===")
keys = [10, 11, 12, 13, 30, 31, 55, 80, 81, 100]
n = len(keys); ranks = list(range(n))
mx = sum(keys)/n; my = sum(ranks)/n
sxx = sum((k-mx)**2 for k in keys); sxy = sum((k-mx)*(r-my) for k, r in zip(keys, ranks))
a = sxy/sxx; b = my - a*mx
print(f"n={n} keys={keys}")
print(f"least squares rank ~ a*key + b: a={a:.6f} b={b:.6f}")
# Paper: positions = M(x_i) rounded to integer (Alg 3.2 says 'rounding'); we show round-half-up
pos_round = [math.floor(a*k + b + 0.5) for k in keys]
# our control: intercept rescaled so first key lands at 0 (transform.hpp:96), floor, capacity 1.5n
alpha = 1.5
intercept = -a*keys[0] + 0.5
max_size = max(1, min(int(n*alpha), int(max(1.0, math.floor(a*keys[-1] + intercept) + 1))))
pos_ours = [min(max(int(math.floor(a*k + intercept)), 0), max_size-1) for k in keys]
print("key   rank  a*k+b     pos(paper,round)  pos(ours,floor,shifted)")
for k, r, pr, po in zip(keys, ranks, pos_round, pos_ours):
    print(f"{k:4d}  {r:4d}  {a*k+b:8.3f}  {pr:6d}  {po:6d}")
def conflict_table(pos, label):
    from collections import Counter
    c = Counter(pos); m = len(c)
    print(f"-- {label}: occupied positions m={m}")
    for j in sorted(c): print(f"   position {j}: D_j = {c[j]}")
    D = sorted(c.values())
    gamma = 0.99; t = int(m*gamma)
    print(f"   sorted D_j ascending: {D}")
    print(f"   t = INT(m*gamma) = INT({m}*0.99) = {t}")
    print(f"   D99 (paper, t-th value in ascending order, 1-based) = {D[t-1]}")
    print(f"   D99 (paper, t-th LARGEST) = {sorted(D, reverse=True)[t-1]}  <- alternative reading")
    idx = min(len(D)-1, max(0, math.ceil(gamma*len(D)) - 1))
    print(f"   ours: idx = ceil(0.99*{m})-1 = {idx}; counts[idx]-1 = {D[idx]-1}  (collisions, i.e. keys minus one)")
    print(f"   max D_j = {max(D)}")
conflict_table(pos_round, "paper rounding")
conflict_table(pos_ours, "our floor/shift with capacity 1.5n (max_size=%d)" % max_size)

print("\n=== B. Algorithm 3.1 feature expansion ===")
def expand(x, d, theta):
    vec = []
    xi = math.floor(x); xf = x - math.floor(x)
    vec.append(xi)
    for k in range(1, d-1):
        vec.append(xi)              # line 10 as printed (adds x_int again)
        xf = xf*theta; xi = math.floor(xf); xf = xf - xi
    vec.append(xf)
    return vec
def expand_intended(x, d, theta):
    vec = []
    xi = math.floor(x); xf = x - math.floor(x)
    vec.append(xi)
    for k in range(1, d-1):
        xf = xf*theta; xi = math.floor(xf); xf = xf - xi
        vec.append(xi)              # likely intended: add the NEW integer digit
    vec.append(xf)
    return vec
X = [1000, 1500, 2500, 4000]; theta = 10
mu = min(X); sigma = (max(X)-min(X))/theta
print(f"X={X} theta={theta} mu={mu} sigma=(max-min)/theta={sigma}")
for x in X:
    xn = (x-mu)/sigma
    print(f"key {x}: x_norm={xn:.4f}  d=2 -> {expand(xn,2,theta)}   d=4 (as printed) -> {expand(xn,4,theta)}   d=4 (digit reading) -> {expand_intended(xn,4,theta)}")
x = 2.71828
print(f"x=2.71828 theta=10 d=4 digit reading -> {expand_intended(x,4,10)}")
print(f"our code encoder for x=2.71828 (in_dim=2): [x, x-floor(x)] = [{x}, {x-math.floor(x):.5f}]")

print("\n=== C. Forward pass of results/aidb/flows/fb_2D2H2L.txt ===")
mean = 15162980.0; var = 1207648189.703125
w0 = [0.0058556031582764, 0.0058592806847908, 0.0, 0.0]   # row-major 2x2: row0 = x feature, row1 = frac feature
w1 = [1.2798795760698392, 0.8812895293543010, 1.4785700705830542, 1.0331745748154413]
def fwd(key):
    x = (key-mean)/var; f = (x, x-math.floor(x))
    u = [f[0]*w0[0] + f[1]*w0[2], f[0]*w0[1] + f[1]*w0[3]]
    h = [math.tanh(u[0]), math.tanh(u[1])]
    out = [h[0]*w1[0] + h[1]*w1[2], h[0]*w1[1] + h[1]*w1[3]]
    z = out[0] + out[1]
    v = [w1[0]+w1[1], w1[2]+w1[3]]
    a_ = [w0[0]+w0[2], w0[1]+w0[3]]
    dzdx = (1-h[0]**2)*a_[0]*v[0] + (1-h[1]**2)*a_[1]*v[1]
    return x, f, u, h, out, z, dzdx
span = var*64
print(f"mean={mean} var={var}  -> key span mean..mean+64*var = {mean:.0f}..{mean+span:.0f}")
for key in [15162980, 15162980 + span/4, 15162980 + span/2, 15162980 + 3*span/4, 15162980 + span]:
    x, f, u, h, out, z, d = fwd(key)
    print(f"key={key:.0f} x={x:.4f} feat=({f[0]:.4f},{f[1]:.4f}) u=({u[0]:.5f},{u[1]:.5f}) h=({h[0]:.5f},{h[1]:.5f}) out=({out[0]:.5f},{out[1]:.5f}) z={z:.5f} dz/dx={d:.5f}")
# linearity check: compare z with a straight line between endpoints
z0 = fwd(mean)[5]; z1 = fwd(mean+span)[5]
maxdev = 0
for i in range(0, 65):
    key = mean + span*i/64; z = fwd(key)[5]; lin = z0 + (z1-z0)*i/64; maxdev = max(maxdev, abs(z-lin))
print(f"z(min)={z0:.5f} z(max)={z1:.5f}; max |z - chord| over 65 grid points = {maxdev:.5f} (relative to range {z1-z0:.5f}: {maxdev/(z1-z0):.4f})")

print("\n=== D. Change of variables in 1-D ===")
for x in [0.1, 0.5, 2.0]:
    z = 1-math.exp(-x); dz = math.exp(-x)
    print(f"X~Exp(1): x={x} z=F(x)={z:.5f} dz/dx={dz:.5f} -> p_Z(z)*|dz/dx| = 1*{dz:.5f} = p_X(x)=e^-x={math.exp(-x):.5f}")
print("NLL per key under N(0,1) target: 0.5*z^2 - log|dz/dx| + 0.5*log(2*pi); e.g. z=0.3, dz/dx=2 ->", 0.5*0.09 - math.log(2) + 0.5*math.log(2*math.pi))

print("\n=== E. Table 2 arithmetic ===")
t2 = {"2H2L(8)":[169.53,40.60,15.28,9.52,8.38,7.40,7.29],"2H4L(16)":[384.84,83.05,34.75,24.00,21.66,19.81,19.63],"4H3L(32)":[320.15,77.52,33.80,24.91,23.52,22.21,22.00],"4H4L(48)":[463.81,113.31,49.40,36.93,35.07,33.13,32.73]}
bs = [1,8,32,128,256,1024,2048]
for k, v in t2.items():
    print(f"{k}: batch1/batch256 = {v[0]/v[4]:.2f}x ; batch1/batch2048 = {v[0]/v[6]:.2f}x ; batch256 vs 2H2L = {v[4]/t2['2H2L(8)'][4]:.2f}x")
print("parameter counts: 2D2H2L =", 2*2+2*2, "; 2H4L =", 2*2*4, "; 4H3L =", 2*4+4*4+4*2, "; 4H4L =", 2*4+4*4+4*4+4*2)
print("throughput ceiling from transform alone at batch 1 (2H2L): 1e9/169.53 = %.2f Mops; at batch 256: %.1f Mops" % (1e9/169.53/1e6, 1e9/8.38/1e6))

print("\n=== F. Table 1 arithmetic ===")
t1 = {"LLT":dict(err=(925487063,118833075),pred=(124831692,101108381),thr=(8.21,11.57),h=(4,3),ah=(2.30,2.01)),
      "FB":dict(err=(928113206,453003864),pred=(492112591,100462387),thr=(4.13,9.16),h=(11,3),ah=(5.91,2.02))}
for k, v in t1.items():
    print(f"{k}: throughput x{v['thr'][1]/v['thr'][0]:.2f}; predictions x{v['pred'][0]/v['pred'][1]:.2f} fewer; errors x{v['err'][0]/v['err'][1]:.2f} fewer; errors per prediction {v['err'][0]/v['pred'][0]:.2f} -> {v['err'][1]/v['pred'][1]:.2f}")

print("\n=== G. Table 3 arithmetic ===")
t3L = dict(LTD=(8,4),LLT=(146,4),LGN=(14,4),YCSB=(3,4),AMZN=(4,4),FB=(386,4),WIKI=(2,4))
t3R = dict(LTD=(7,4),LLT=(147,5),LGN=(13,4),YCSB=(3,4),AMZN=(4,4),FB=(454,4),WIKI=(1,4))
for k in t3L:
    r, f = t3L[k]; print(f"{k}: load raw {r} -> flow {f} (x{r/f:.1f}); run raw {t3R[k][0]} -> flow {t3R[k][1]}; flow larger? {f>r}; flow strictly smaller? {f<r}")

print("\n=== H. Bulk load arithmetic ===")
print("13.42 s / 100M keys = %.1f ns/key overall; 77%% transform = %.2f s = %.1f ns/key" % (13.42e9/100e6, 0.77*13.42, 0.77*13.42e9/100e6))
print("Table 2 batch-256 2H2L: 8.38 ns/key -> 100M keys = %.2f s" % (8.38*100e6/1e9))
