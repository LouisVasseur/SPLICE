#!/usr/bin/env python3
"""Numeric checks for the fixer pass on 02_csv.md (stdlib only)."""
from fractions import Fraction as F
import math, itertools

keys=[2,3,4,5,12,13,14]; n=len(keys); y0=list(range(n))
print("== eq. 9 with original vs shifted ranks (k_v = 9, y_v = 4)")
ys=[0,1,2,3,5,6,7]
print(" original: (sum y_orig + n)/(n+1) =", (sum(y0)+n)/(n+1))
print(" shifted : (sum y_shift + n)/(n+1) =", (sum(ys)+n)/(n+1))
print(" true (n+1)-point mean of 0..7 =", sum(range(8))/8)

print("== eq. 14 lower limit: sum(keys[3:]) =", sum(keys[3:]), " sum(keys[4:]) =", sum(keys[4:]))

# secant / chord intercept of L' between 6 and 11
Lp6=-1.132330; Lp11=0.886966
kstar=6+(11-6)*(-Lp6)/(Lp11-Lp6)
print("== secant zero of L' between 6 and 11:", round(kstar,6))

# OLS helpers
def ols(xs,ys):
    m=len(xs); mx=sum(xs)/m; my=sum(ys)/m
    sxx=sum((x-mx)**2 for x in xs); syy=sum((y-my)**2 for y in ys); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    w=sxy/sxx; b=my-w*mx; sse=sum((w*x+b-y)**2 for x,y in zip(xs,ys))
    rho=sxy/math.sqrt(sxx*syy)
    return w,b,sse,rho,syy
w,b,sse,rho,syy=ols([F(k) for k in keys],[F(i) for i in range(n)])
print("== rho before:", round(float(rho),6), "rho^2", round(float(rho**2),6), "S_yy", syy, "S_yy(1-rho^2)=", float(syy*(1-rho**2)), "SSE", float(sse))
aug=sorted(keys+[9]); w2,b2,sse2,rho2,syy2=ols([F(k) for k in aug],[F(i) for i in range(8)])
print("== rho after k_v=9:", round(float(rho2),6), "rho^2", round(float(rho2**2),6), "S_yy", syy2, "S_yy(1-rho^2)=", float(syy2*(1-rho2**2)), "SSE", float(sse2))
aug3=sorted(keys+[8,9,10]); w3,b3,sse3,rho3,syy3=ols([F(k) for k in aug3],[F(i) for i in range(10)])
print("== rho after {8,9,10}:", round(float(rho3),6), "SSE", float(sse3))

# Algorithm-1 candidate set per greedy round: derivative signs at run endpoints
def loss_with(seqx, v):
    xs=sorted(seqx+[v]); return ols([F(x).limit_denominator(10**9) for x in xs],[F(i) for i in range(len(xs))])[2]
def dloss(seqx,v,h=F(1,1000)):
    return float((loss_with(seqx,v+h)-loss_with(seqx,v-h))/(2*h))
cur=list(keys)
for rnd in range(1,4):
    print(f"== round {rnd}: base set {cur}")
    C=[]
    for a,bk in zip(cur,cur[1:]):
        run=list(range(a+1,bk))
        if not run: continue
        if len(run)<=2:
            print(f"   run {run}: length {len(run)} <= 2 -> both endpoints kept (line 8)"); C+=run
        else:
            d1=dloss(cur,F(run[0])); d2=dloss(cur,F(run[-1]))
            print(f"   run {run[0]}..{run[-1]}: L'({run[0]})={d1:+.4f}, L'({run[-1]})={d2:+.4f}", end=' ')
            if d1*d2<0:
                lo,hi=float(run[0]),float(run[-1])
                for _ in range(100):
                    m1=lo+(hi-lo)/3; m2=hi-(hi-lo)/3
                    if loss_with(cur,F(m1).limit_denominator(10**9))<loss_with(cur,F(m2).limit_denominator(10**9)): hi=m2
                    else: lo=m1
                ks=(lo+hi)/2; sec=run[0]+(run[-1]-run[0])*(-d1)/(d2-d1)
                print(f"-> sign change; ternary min {ks:.4f}, secant {sec:.4f}, nearest int {round(ks)}"); C.append(round(ks))
            else:
                print("-> same sign; both endpoints kept"); C+=[run[0],run[-1]]
    losses={c:float(loss_with(cur,c)) for c in C}
    print("   C =",C," losses:",{c:round(l,4) for c,l in losses.items()})
    best=min(losses,key=losses.get); cur=sorted(cur+[best]); print("   accept",best)

print("== Table 2 percentages: greedy", 100*(8.327-2.293)/8.327, "exhaustive", 100*(8.327-2.118)/8.327)
print("== Table 3/4 growth: ALEX OSM", 81620/988, "ALEX FB", 48737/247, "LIPP Genome", 15709/1155, "LIPP OSM", 13019/1217, "ALEX Covid", 4955/609, "ALEX Genome", 9777/1356, "LIPP FB", 2228/589)
print("== PLA-32 drops:", {k:round(1-b/a,4) for k,(a,b) in dict(history=(1067,411),books=(602,236),planet=(3456,2767),genome=(2045,1627),libio=(1256,614),stack=(593,319),wise=(787,436),fb=(3034,1718),covid=(1192,892)).items()})
print("== per-region: regions", math.ceil(2_000_000/4096), "budget/region", math.floor(0.1*4096), "gaps", 4095, "cand evals", 409*4095)
print("   books: 185327 vp / 489 regions =", 185327/489, " smoothing_ns/region ms =", 16575434458/489/1e6, " per key us =", 16575434458/2e6/1e3, " x16 threads per key us =", 16*16575434458/2e6/1e3)
print("   200M regions:", math.ceil(200_000_000/4096), " 4x =", 4*math.ceil(200_000_000/4096))
