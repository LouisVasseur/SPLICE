#!/usr/bin/env python3
"""Worked example for CSV Algorithm 1 on 7 integer keys. Standard library only.
Loss = OLS SSE over the augmented set K u V with targets = positions 0..n+|V|-1 (CSV eq. 4/5).
"""
from fractions import Fraction as F
import itertools

def ols(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    w = sxy/sxx if sxx else 0; b = my - w*mx
    sse = sum((w*x+b-y)**2 for x,y in zip(xs,ys))
    return w, b, sse

def loss_aug(seq):
    """seq = sorted list of (x, is_virtual). Targets are positions."""
    xs=[F(x) for x,_ in seq]; ys=[F(i) for i in range(len(seq))]
    return ols(xs, ys)

def loss_real_only(seq):
    w,b,_ = loss_aug(seq)
    return sum((w*F(x)+b-F(i))**2 for i,(x,v) in enumerate(seq) if not v)

keys=[2,3,4,5,12,13,14]
seq=[(k,False) for k in keys]
w,b,sse = loss_aug(seq)
print("keys", keys)
print("n =", len(keys), " ranks 0..6")
print("initial OLS: w =", w, "=", float(w), " b =", b, "=", float(b))
for i,k in enumerate(keys):
    p = w*k+b
    print(f"  k={k:2d} rank={i} pred={float(p):.4f} residual={float(p-i):+.4f} sq={float((p-i)**2):.4f}")
print("SSE(K) =", sse, "=", float(sse))

# Evaluate every integer candidate in (min,max) not in K (paper's candidate set), one virtual point
print("\n--- one virtual point: every integer candidate ---")
cands=[]
for v in range(min(keys)+1, max(keys)):
    if v in keys: continue
    aug=sorted(seq+[(v,True)], key=lambda t:t[0])
    w2,b2,s2 = loss_aug(aug)
    cands.append((s2,v,w2,b2,aug))
    print(f"  kv={v:2d} -> loss {float(s2):.4f}  (w={float(w2):.4f}, b={float(b2):.4f})")
best=min(cands)
s2,v,w2,b2,aug=best
print("\nbest integer candidate kv =", v, " loss =", s2, "=", float(s2))
print("slots after insertion (key -> slot):")
for i,(x,virt) in enumerate(aug):
    print(f"  slot {i}: {'VIRTUAL' if virt else 'key'} {x}  pred={float(w2*x+b2):.4f} residual={float(w2*x+b2-i):+.4f}")
print("loss over real keys only (paper's L_f'(K)):", float(loss_real_only(aug)))
print("loss over real+virtual (paper's L_f'(K u V)):", float(s2))

# Sub-sequence structure: continuous candidate runs (gaps)
print("\n--- sub-sequences (integer runs between consecutive keys) ---")
for a,bk in zip(keys, keys[1:]):
    run=list(range(a+1,bk))
    if run: print(f"  gap ({a},{bk}): candidates {run} length {len(run)}")
    else: print(f"  gap ({a},{bk}): empty")

# derivative sign test at endpoints of the length>2 run, via finite difference of the real-valued loss
def loss_real_valued(v):
    aug=sorted(seq+[(v,True)], key=lambda t:t[0])
    return loss_aug(aug)[2]
def dloss(v, h=F(1,1000)):
    return (loss_real_valued(v+h)-loss_real_valued(v-h))/(2*h)
print("\n--- derivative sign test on the run 6..11 (endpoints 6 and 11) ---")
for e in (6, 11):
    print(f"  dL/dkv at kv={e}: {float(dloss(F(e))):+.5f}")
# continuous minimiser inside (5,12) by ternary search on Fractions -> floats
lo,hi=5.001,11.999
def lf(v): return float(loss_real_valued(F(v).limit_denominator(10**9)))
for _ in range(200):
    m1=lo+(hi-lo)/3; m2=hi-(hi-lo)/3
    if lf(m1)<lf(m2): hi=m2
    else: lo=m1
print(f"  continuous minimiser inside the gap: kv* = {(lo+hi)/2:.4f}, loss = {lf((lo+hi)/2):.4f}")

# Greedy for budget lambda = floor(alpha n) with alpha=0.5 -> 3
print("\n--- greedy, alpha = 0.5, lambda = floor(0.5*7) = 3 ---")
cur=list(seq); cur_loss=sse; V=[]
for rnd in range(3):
    best=None
    present={x for x,_ in cur}
    for v in range(min(keys)+1, max(keys)):
        if v in present: continue
        aug=sorted(cur+[(v,True)], key=lambda t:t[0]); s=loss_aug(aug)[2]
        if s < cur_loss and (best is None or s<best[0]): best=(s,v,aug)
    if best is None:
        print(f"  round {rnd+1}: no candidate lowers the loss -> stop (Alg.1 line 27)"); break
    cur_loss,v,cur=best; V.append(v)
    print(f"  round {rnd+1}: insert kv={v}, loss -> {float(cur_loss):.4f}")
print("V =", V)
w3,b3,s3=loss_aug(cur)
print("final model w =", float(w3), " b =", float(b3))
for i,(x,virt) in enumerate(cur):
    print(f"  slot {i}: {'VIRTUAL' if virt else 'key'} {x} pred={float(w3*x+b3):.3f}")
print("final loss (K u V):", float(s3), " real-only:", float(loss_real_only(cur)))
print("reduction:", f"{100*(1-float(s3)/float(sse)):.2f}%")

# Exhaustive for budget 3
print("\n--- exhaustive, at most 3 virtual points among the integer pool ---")
pool=[v for v in range(min(keys)+1,max(keys)) if v not in keys]
bestx=(sse,())
count=0
for c in range(1,4):
    for comb in itertools.combinations(pool,c):
        aug=sorted(seq+[(v,True) for v in comb], key=lambda t:t[0]); s=loss_aug(aug)[2]; count+=1
        if s<bestx[0]: bestx=(s,comb)
print("subsets examined:", count, " best:", bestx[1], " loss:", float(bestx[0]))

# Paper's toy keys from tools/virtual_points_lab.py default (NOT the paper's figure)
print("\n--- lab default keys 1,2,3,4,5,10,20,26,27,30 (budget 3) for cross-check ---")
lab=[1,2,3,4,5,10,20,26,27,30]; seql=[(k,False) for k in lab]
print("before:", float(loss_aug(seql)[2]))
