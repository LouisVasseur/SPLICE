#!/usr/bin/env python3
# The memory-speed frontier of one dataset over every timed arm: pareto.py DATASET ROOT...   Order-keeping and any-order frontiers apart.
import csv,sys,glob,collections,os
import numpy as np
ds=sys.argv[1]
T=collections.defaultdict(list)
for root in sys.argv[2:]:
    for p in glob.glob(root+"/table.tsv")+glob.glob(root+"/*/table.tsv"):
        tag=os.path.basename(os.path.dirname(p))
        for r in csv.DictReader(open(p),delimiter="\t"):
            if r["dataset"]!=ds or not r["Mops"]: continue
            idx=r["index"]
            if idx=="gr4" and "/real" in p: idx="gd4"
            if idx=="gr4" and "/hand" in p: idx="hand_rule"
            T[(idx,r["spk"],r["tau"],r["leaf"],r["cap"])].append((float(r["Mops"]),float(r["Bkey"])))
P=[(float(np.median([x[1] for x in v])),float(np.median([x[0] for x in v])),k,len(v)) for k,v in T.items()]
P.sort()
def front(items):
    out=[];best=-1
    for b,m,k,n in items:
        if m>best: out.append((b,m,k,n)); best=m
    return out
ORDER={"ab4","gr4","gd4","hand_rule","alex","lipp","pgm","btree","splice"}
for label,sel in (("order-keeping",lambda k:k[0] in ORDER),("any order",lambda k:True)):
    print(label,"frontier for",ds)
    for b,m,k,n in front([x for x in P if sel(x[2])]):
        print(f"  {b:6.2f} B/key  {m:6.2f} M/s  {k[0]} cap={k[1]} tau={k[2]} leaf={k[3]} v={k[4]}  ({n} passes)")
