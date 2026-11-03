#!/usr/bin/env python3
# What to time on the finished code, with no model deciding what the oracle may contain. A cell with at least as many lines and levels, as large a
# directory and as bad a tail as another at the same budget and coordinate is skipped, and a sample of the skipped cells is timed anyway, against
# the timed cell that justified skipping it, to test that. A fitted cell has its unfitted twin's directory and levels and never more lines or a worse
# tail, by construction, so unfitted cells are dominated too unless the counts say otherwise; those that attribution needs are timed regardless: the
# coarse base, leaf 8, leaf 4 on the hard sets, and the twin of each fitted cell the frozen weights like best. Cells of the discovered rule at
# leaves 8 and 16 come from the G batch.
import csv,os,random,sys
root=sys.argv[1]
HARD="fb genome osm planet books".split()
ALL="fb genome osm planet books covid libio history stack wise".split()
A,B,C,D=map(float,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"gate_weights.txt")).read().split())
rows=[r for r in csv.DictReader(open(root+"/counts.tsv"),delimiter="\t") if r["lines"]]
rows.sort(key=lambda r:(r["ds"],r["idx"],int(r["budget"]),int(r["leaf"]),r["tau"],r["v"]))
for r in rows:
    r["L"],r["E"],r["S"],r["P"],r["M"]=float(r["lines"]),float(r["levels"]),float(r["dir_bytes"]),int(r["p999"]),int(r["max"])
    r["cost"]=A+B*(r["L"]-1)+C*(r["E"]-1)
key=lambda r:(r["ds"],r["idx"],r["budget"])
twin={(key(r),r["leaf"],r["tau"],r["v"]):r for r in rows}
def beats(q,r):
    ok=q["L"]<=r["L"]+0.002 and q["E"]<=r["E"]+0.005 and q["S"]<=r["S"]+2097152 and q["P"]<=r["P"] and q["M"]<=r["M"]
    return ok and (q["L"]<r["L"]-0.002 or q["E"]<r["E"]-0.005 or q["S"]<r["S"]-2097152 or q["P"]<r["P"] or q["M"]<r["M"] or (q["leaf"],q["tau"])<(r["leaf"],r["tau"]))
groups={}
for r in rows: groups.setdefault(key(r)+(r["v"],),[]).append(r)
best_v1={}
for r in rows:
    if r["v"]=="1" and (key(r) not in best_v1 or r["cost"]<best_v1[key(r)]["cost"]): best_v1[key(r)]=r
done=lambda r:r["idx"]=="gr4" and r["leaf"] in("8","16") and r["tau"]=="2"
broken=0
keep,skip=[],[]
for g in groups.values():
    for r in g:
        t=twin.get((key(r),r["leaf"],r["tau"],"1")) if r["v"]=="0" else None
        must=(r["idx"]=="ab4" and r["leaf"]=="64" and r["tau"]=="3")
        if r["v"]=="0":
            bv=best_v1.get(key(r))
            must=must or (r["leaf"]=="8" and r["tau"]=="2") or (r["ds"] in HARD and r["leaf"]=="4" and r["tau"]=="2") or (bv and bv["leaf"]==r["leaf"] and bv["tau"]==r["tau"])
            if t and (t["L"]>r["L"]+1e-4 or t["M"]>r["M"] or t["P"]>r["P"]): must=True; broken+=1
            r["ref"]=t
            dom=True
        else:
            q=next((q for q in g if q is not r and beats(q,r)),None)
            r["ref"]=q
            dom=q is not None
            must=must or (r["ds"] in HARD and r["idx"]=="ab4" and r["leaf"]=="4" and r["tau"]=="2")
        (keep if must or not dom else skip).append(r)
keep=[r for r in keep if not done(r)]
kept={id(r) for r in keep}
skip=[r for r in skip if not done(r)]
# the audit: cells whose reference is itself timed, dominated fitted cells on the hard sets first
usable=[r for r in skip if r.get("ref") is not None and (id(r["ref"]) in kept or done(r["ref"]))]
random.seed(20261103)
s1=[r for r in usable if r["v"]=="1" and r["ds"] in HARD]; s2=[r for r in usable if not (r["v"]=="1" and r["ds"] in HARD)]
audit=random.sample(s1,min(16,len(s1)))
audit+=random.sample(s2,min(24-len(audit),len(s2)))
cells=[]
with open(root+"/picked.tsv","w") as g:
    for why,rs in (("front",keep),("audit",audit)):
        for r in rs:
            cells.append(f"{r['idx']} {r['ds']} {r['budget']} {r['leaf']} {r['tau']} {r['v']} 1\n")
            q=r.get("ref")
            g.write("\t".join([why,r["idx"],r["ds"],r["budget"],r["leaf"],r["tau"],r["v"],(q["leaf"]+"/"+q["tau"]+"/"+q["v"]) if q else "-"])+"\n")
n=0
for ds in ALL:
    for b in (20,23,25):
        cells.append(f"hf4 {ds} {b} 0 0 0 1\n"); cells.append(f"ph4 {ds} {b} 65536 0 0 1\n"); n+=2
    cells.append(f"ph4 {ds} 20 1024 0 0 1\n"); n+=1
random.shuffle(cells)  # so that no family only ever runs beside itself
open(root+"/cells.txt","w").writelines(cells)
print(len(rows),"counted cells,",len(keep),"on the front,",len(skip),"skipped,",len(audit),"of them audited,",n,"hash and partition arms,",len(cells),"to time;",broken,"twins where the fit's guarantee failed")
