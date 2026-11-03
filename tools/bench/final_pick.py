#!/usr/bin/env python3
# What to time on the finished code, with no model choosing. A cell with at least as many lines and levels and at least as large a
# directory as another at the same budget and coordinate cannot be faster but for noise, so it is skipped, and a random sample of the
# skipped cells is timed anyway to test that. A fitted cell has its unfitted twin's directory and levels and never more lines, by
# construction, so unfitted cells are dominated too: only those that attribution needs are timed, the coarse base, leaf 8 and, on the
# hard sets, leaf 4. Cells of the discovered rule at leaves 8 and 16 come from the G batch.
import csv,random,sys
root=sys.argv[1]
HARD="fb genome osm planet books".split()
ALL="fb genome osm planet books covid libio history stack wise".split()
rows=[r for r in csv.DictReader(open(root+"/counts.tsv"),delimiter="\t") if r["lines"]]
for r in rows:
    r["L"],r["E"],r["D"]=float(r["lines"]),float(r["levels"]),float(r["dir_bytes"])
groups={}
for r in rows: groups.setdefault((r["ds"],r["idx"],r["budget"],r["v"]),[]).append(r)
keep,skip=[],[]
def attribution(r):
    return (r["leaf"]=="64" and r["tau"]=="3") or (r["leaf"]=="8" and r["tau"]=="2") or (r["ds"] in HARD and r["leaf"]=="4" and r["tau"]=="2")
for k,g in groups.items():
    for r in g:
        if r["v"]=="0" and not attribution(r):
            skip.append(r)
            continue
        dom=any(q is not r and q["L"]<=r["L"]+0.002 and q["E"]<=r["E"]+0.005 and q["D"]<=r["D"]+2097152 and
                (q["L"]<r["L"]-0.002 or q["E"]<r["E"]-0.005 or q["D"]<r["D"]-2097152 or (q["leaf"],q["tau"])<(r["leaf"],r["tau"])) for q in g)
        must=(r["idx"]=="ab4" and r["leaf"]=="64" and r["tau"]=="3") or (r["ds"] in HARD and r["idx"]=="ab4" and r["leaf"]=="4" and r["tau"]=="2")
        (keep if must or not dom else skip).append(r)
done=lambda r:r["idx"]=="gr4" and r["leaf"] in("8","16") and r["tau"]=="2"
keep=[r for r in keep if not done(r)]
skip=[r for r in skip if not done(r)]
random.seed(20261103)
audit=random.sample(skip,min(24,len(skip)))
with open(root+"/cells.txt","w") as f, open(root+"/picked.tsv","w") as g:
    for why,rs in (("front",keep),("audit",audit)):
        for r in rs:
            f.write(f"{r['idx']} {r['ds']} {r['budget']} {r['leaf']} {r['tau']} {r['v']} 1\n")
            g.write(f"{why}\t{r['idx']}\t{r['ds']}\t{r['budget']}\t{r['leaf']}\t{r['tau']}\t{r['v']}\n")
    n=0
    for ds in ALL:
        for b in (20,23,25):
            f.write(f"hf4 {ds} {b} 0 0 0 1\nph4 {ds} {b} 65536 0 0 1\n"); n+=2
        f.write(f"ph4 {ds} 20 1024 0 0 1\n"); n+=1
print(len(rows),"counted cells,",len(keep),"on the front,",len(skip),"skipped,",len(audit),"of them audited,",n,"hash and partition arms,",len(keep)+len(audit)+n,"to time")
