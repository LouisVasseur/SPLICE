#!/usr/bin/env python3
"""Additional worked numeric examples for 01_nfl.md (fixer pass; standard library only).
Run from the experimental/scaleli directory."""
import math, csv, struct, random

print("=== I. Sawtooth: a 2D2H2L net with a NON-ZERO fractional row is not monotone ===")
mean = 15162980.0; var = 1207648189.703125
w1 = [1.2798795760698392, 0.8812895293543010, 1.4785700705830542, 1.0331745748154413]
def fwd(x, w0):
    f = (x, x - math.floor(x))
    u = [f[0]*w0[0] + f[1]*w0[2], f[0]*w0[1] + f[1]*w0[3]]
    h = [math.tanh(u[0]), math.tanh(u[1])]
    z = h[0]*(w1[0]+w1[1]) + h[1]*(w1[2]+w1[3])
    v = [w1[0]+w1[1], w1[2]+w1[3]]
    a_ = [w0[0]+w0[2], w0[1]+w0[3]]
    d = (1-h[0]**2)*a_[0]*v[0] + (1-h[1]**2)*a_[1]*v[1]
    return f, u, h, z, d
w0_real = [0.0058556031582764, 0.0058592806847908, 0.0, 0.0]
w0_hyp  = [0.0058556031582764, 0.0058592806847908, 0.5, 0.5]   # hypothetical: fractional row = [0.5, 0.5]
print("weights as in fb_2D2H2L.txt except W0 row 1 = [0.5, 0.5] (hypothetical, --monotone off)")
print("   x      key            frac     u1       u2       z(real file)  z(hypothetical)  dz/dx(hyp, a.e.)")
for x in [15.9, 15.99, 16.0, 16.01, 16.5]:
    fr, ur, hr, zr, dr = fwd(x, w0_real); fh, uh, hh, zh, dh = fwd(x, w0_hyp)
    print(f"{x:6.2f}  {mean+x*var:14.0f}  {fh[1]:.5f}  {uh[0]:.5f}  {uh[1]:.5f}  {zr:.5f}       {zh:.5f}          {dh:.5f}")
# jump at the integer x=16: z(16-) - z(16+)
zm = fwd(16 - 1e-9, w0_hyp)[3]; zp = fwd(16.0, w0_hyp)[3]
print(f"jump at x=16: z(16^-) - z(16^+) = {zm - zp:.5f}")
u16 = [16*w0_hyp[0], 16*w0_hyp[1]]
v = [w1[0]+w1[1], w1[2]+w1[3]]
jump_formula = sum(v[h]*(math.tanh(u16[h] + w0_hyp[2+h]) - math.tanh(u16[h])) for h in range(2))
print(f"closed form sum_h v_h*(tanh(u_h + W0[1,h]) - tanh(u_h)) with u_h = 16*W0[0,h]: {jump_formula:.5f}")
print("=> with a non-zero fractional row the composed map key -> z decreases at every integer of x (sawtooth);")
print("   train_flow.py --monotone zeroes W0[1,:], which is why fb_training.json has unordered_transformed_pairs = 0")

print("\n=== J. Jacobian formula check (analytic vs central finite difference), hypothetical weights, x=15.5 ===")
x = 15.5; eps = 1e-6
d_an = fwd(x, w0_hyp)[4]; d_fd = (fwd(x+eps, w0_hyp)[3] - fwd(x-eps, w0_hyp)[3])/(2*eps)
print(f"analytic dz/dx = {d_an:.8f}; finite difference = {d_fd:.8f}; |diff| = {abs(d_an-d_fd):.2e}")
print(f"dz/dkey = dz/dx / var = {d_an/var:.6e} per key unit")

print("\n=== K. Why sigma^2 = 1e16 is 'flat' ===")
for z in [1e3, 1e6, 1e8]:
    print(f"|z| = {z:.0e}: data term z^2/(2 sigma^2) = {z*z/(2*1e16):.3e}; log-density offset ln sqrt(2 pi 1e16) = {0.5*math.log(2*math.pi*1e16):.3f}")
print("our trainer: sigma = 1, so for z ~ 1 the data term is 0.5, of the same order as -log|dz/dx|")

print("\n=== L. Batched P99 vs per-request P99 (illustration) ===")
lat = [10000] + [50]*255
batch = sum(lat); print(f"batch of 256: one 10,000 ns lookup + 255 x 50 ns = {batch} ns; reported as {batch}/256 = {batch/256:.4f} ns per operation")
srt = sorted(lat); print(f"per-request P99 of the same trace = {srt[math.ceil(0.99*256)-1]} ns; max = {srt[-1]} ns")

print("\n=== M. Control vs forced flow, fence probes per operation (results/aidb/sweep/summary.csv) ===")
rows = list(csv.DictReader(open('results/aidb/sweep/summary.csv')))
def get(d, v, col): return [r[col] for r in rows if r['dataset']==d and r['variant']==v][0]
print("dataset  packed_rank  packed_rank_flow  packed_rank_flow_forced  forced-control  flow_region_fraction  preprocess_ns_per_key(flow)")
for d in ['books','covid','fb','genome','history','libio','osm','planet','stack','wise']:
    c = float(get(d,'packed_rank','fence_probes_per_operation')); f = float(get(d,'packed_rank_flow','fence_probes_per_operation')); ff = float(get(d,'packed_rank_flow_forced','fence_probes_per_operation'))
    print(f"{d:8s} {c:.6f}     {f:.6f}          {ff:.6f}                 {ff-c:+.6f}       {float(get(d,'packed_rank_flow','flow_region_fraction')):.4f}                {float(get(d,'packed_rank_flow','preprocess_ns_per_key')):.2f}")

print("\n=== N. Training subsample of fb_2M_uniform_s42 (train_flow.py: seed 1000000007, sample 4096) ===")
b = open('data/samples/fb_2M_uniform_s42','rb').read(); n = struct.unpack_from('<Q', b)[0]
keys = sorted(set(struct.unpack_from(f'<{n}Q', b, 8)))
rng = random.Random(1000000007); tk = sorted(rng.sample(keys, 4096))
print(f"full sample: n={len(keys)} min={keys[0]} max={keys[-1]}")
print(f"training subsample: min={tk[0]} max={tk[-1]} var=(max-min)/64={(tk[-1]-tk[0])/64}")
print(f"full-sample keys below the subsample min: {sum(1 for k in keys if k < tk[0])} (x < 0); above the subsample max: {sum(1 for k in keys if k > tk[-1])} (x > 64)")
print(f"x for the smallest full-sample key: {(keys[0]-tk[0])/((tk[-1]-tk[0])/64):.6f}; for the largest: {(keys[-1]-tk[0])/((tk[-1]-tk[0])/64):.6f}")

print("\n=== O. Small arithmetic used in the text ===")
print(f"Table 2: (8.38-7.29)/8.38 = {(8.38-7.29)/8.38:.4f}; 8.38/7.29 = {8.38/7.29:.4f}; 169.53/7.29 = {169.53/7.29:.2f}; 169.53/8.38 = {169.53/8.38:.2f}")
print(f"fb regions: 41/489 = {41/489:.4f}; transform_ns 59765592 / 2e6 keys = {59765592/2e6:.2f} ns/key")
print(f"fb fence probes forced - control = {2.599933-2.599879:.2e}")
print(f"2H2L with biases would have 2*2+2 + 2*2+2 = {2*2+2+2*2+2} parameters; without: {2*2+2*2}")
