#!/usr/bin/env python3
# Replays the deterministic selector against the exhaustive timed grid.
#   replay.py BUDGET_ROOT [G_ROOT]
# The selector sees only what a build can know: exact lines, levels, directory bytes, the memory cap and the scan requirement. It never
# sees a timing from the dataset it is choosing for: the cost weights are refit with that dataset left out.
# Oracle: the fastest arm actually timed for that dataset and cap, within what the requirement allows.
import csv,math,sys,collections,glob,os
import numpy as np
root=sys.argv[1]; groot=sys.argv[2] if len(sys.argv)>2 else None
ALL="fb genome osm planet books covid libio history stack wise".split()
def rows(p):
    return list(csv.DictReader(open(p),delimiter="\t")) if os.path.exists(p) else []
timed=collections.defaultdict(list)
for sub in ("bud","fin","rep"):
    for r in rows(f"{root}/{sub}/table.tsv"):
        if not r["Mops"]: continue
        k=(r["dataset"],int(float(r["spk"])),r["index"],int(r["leaf"]),r["tau"],int(r["cap"]))
        timed[k].append(r)
if groot:
    for sub in ("real",):
        for r in rows(f"{groot}/{sub}/table.tsv"):
            if not r["Mops"]: continue
            k=(r["dataset"],int(float(r["spk"])),"gd4",int(r["leaf"]),r["tau"],int(r["cap"]))
            timed[k].append(r)
def med(v): return float(np.median(v))
A={}
for k,rs in timed.items():
    A[k]=dict(ds=k[0],b=k[1],idx=k[2],leaf=k[3],tau=k[4],v=k[5],mops=med([float(r["Mops"]) for r in rs]),bkey=med([float(r["Bkey"]) for r in rs]),
              lines=med([float(r["lines"]) for r in rs if r["lines"]] or [float("nan")]),levels=med([float(r["levels"]) for r in rs if r["levels"]] or [1.0]),
              dirmb=med([float(r["dir_MB"]) for r in rs if r["dir_MB"]] or [0.5]),n=len(rs))
ORDER={"ab4","gr4","gd4"}; HASH={"hf4"}; PART={"ph4"}
def feats(a):
    rule=1.0 if a["idx"] in("gr4","gd4") else 0.0
    hashed=1.0 if a["idx"]=="hf4" else 0.0
    return [1.0,a["lines"]-1,a["levels"]-1,math.log2(max(a["dirmb"],1.0)),rule,hashed]
arms=list(A.values())
usable=[a for a in arms if not math.isnan(a["lines"])]
def fit(train):
    X=np.array([feats(a) for a in train]); y=np.array([1e3/a["mops"] for a in train])
    w=1/y   # relative error
    beta,*_=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)
    return beta
beta=fit(usable)
print("cost weights on all data: ns = %.1f + %.1f (lines-1) + %.1f (levels-1) + %.2f log2(dirMB) + %.1f rule + %.1f hash"%tuple(beta))
def regret(pick,oracle): return 1-pick["mops"]/oracle["mops"]
SCAN={"none":ORDER|HASH|PART,"unordered":ORDER|PART,"sorted":ORDER}
tot=collections.defaultdict(list)
print("\nselector against the timed oracle, weights fit without the dataset, top pick and with one timed check between the top two:")
print(f"{'ds':8s} {'cap':>3s} {'scan':10s} {'oracle':26s} {'M/s':>6s} {'pick':26s} {'M/s':>6s} {'regret':>7s} {'top2':>7s}")
def name(a): return f"{a['idx']} T{a['leaf']} t{a['tau']} V{a['v']}"
for ds in ALL:
    beta_o=fit([a for a in usable if a["ds"]!=ds])
    for b in (20,23,25):
        cand=[a for a in usable if a["ds"]==ds and a["b"]==b and a["bkey"]<=b+0.1]
        if not cand: continue
        for scan,allowed in SCAN.items():
            c=[a for a in cand if a["idx"] in allowed]
            if not c: continue
            o=max(c,key=lambda a:a["mops"])
            ranked=sorted(c,key=lambda a:float(np.dot(beta_o,feats(a))))
            p=ranked[0]; two=max(ranked[:2],key=lambda a:a["mops"])
            r1,r2=regret(p,o),regret(two,o)
            tot[scan].append((r1,r2))
            if scan=="sorted" or (scan=="none" and r1>0.01):
                print(f"{ds:8s} {b:3d} {scan:10s} {name(o):26s} {o['mops']:6.2f} {name(p):26s} {p['mops']:6.2f} {100*r1:6.1f}% {100*r2:6.1f}%")
print()
for scan,v in tot.items():
    a=np.array(v)
    print(f"scan={scan:10s} cells={len(v):3d} mean regret {100*a[:,0].mean():5.2f}% worst {100*a[:,0].max():5.1f}% | with one timed check: mean {100*a[:,1].mean():5.2f}% worst {100*a[:,1].max():5.1f}%")
# what a fixed choice would have lost, as baselines for the selector
print("\nfixed choices, order-keeping (scan=sorted):")
for label,f in (("T64 t3 V0 everywhere",lambda a:a["idx"]=="ab4" and a["leaf"]==64 and a["tau"]=="3" and a["v"]==0),
                ("T64 t3 V1 everywhere",lambda a:a["idx"]=="ab4" and a["leaf"]==64 and a["tau"]=="3" and a["v"]==1)):
    out=[]
    for ds in ALL:
        for b in (20,23,25):
            cand=[a for a in usable if a["ds"]==ds and a["b"]==b and a["idx"] in ORDER and a["bkey"]<=b+0.1]
            pick=[a for a in cand if f(a)]
            if cand and pick: out.append(regret(pick[0],max(cand,key=lambda a:a["mops"])))
    if out: print(f"  {label:24s} cells={len(out):3d} mean {100*np.mean(out):5.2f}% worst {100*np.max(out):5.1f}%")
# T against V against both, at each budget, on hard sets
print("\nattribution at equal bytes: M/s of each arm, median of passes")
for ds in ALL:
    for b in (20,23,25):
        c=[a for a in arms if a["ds"]==ds and a["b"]==b]
        if not c: continue
        base=[a for a in c if a["idx"]=="ab4" and a["leaf"]==64 and a["tau"]=="3" and a["v"]==0]
        fitonly=[a for a in c if a["idx"]=="ab4" and a["leaf"]==64 and a["tau"]=="3" and a["v"]==1]
        v0=[a for a in c if a["idx"]=="ab4" and a["v"]==0]
        v1=[a for a in c if a["idx"]=="ab4" and a["v"]==1]
        g=[a for a in c if a["idx"] in("gr4","gd4")]
        f=lambda L:(max(L,key=lambda a:a["mops"]) if L else None)
        s=lambda a:(f"{a['mops']:6.2f}@{a['bkey']:5.2f} T{a['leaf']}" if a else "-")
        hf=f([a for a in c if a["idx"]=="hf4"]); ph=f([a for a in c if a["idx"]=="ph4"])
        print(f"{ds:8s} {b:3d}  base {s(f(base)):22s} fit {s(f(fitonly)):22s} bestT {s(f(v0)):22s} both {s(f(v1)):22s} G {s(f(g)):22s} ph4 {s(ph):22s} hf4 {s(hf)}")
