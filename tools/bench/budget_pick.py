
# Picks the timed arms from the exact counts: per dataset and byte budget, the coarse base, the fit alone, the finest splits and the
# lowest predicted cost with and without the fit. Predicted cost is the fit from earlier runs; the timing decides.
import csv,math,sys
root=sys.argv[1]
HARD="fb genome osm planet books".split()
ALL="fb genome osm planet books covid libio history stack wise".split()
rows=list(csv.DictReader(open(root+"/counts.tsv"),delimiter="\t"))
C={}
for r in rows:
    if not r["lines"]: continue
    r["cost"]=33+33*(float(r["lines"])-1)+25*(float(r["levels"])-1)+3.5*math.log2(max(float(r["dir_bytes"])/1048576,1))
    C[(r["ds"],r["idx"],int(r["budget"]),int(r["leaf"]),r["tau"],int(r["v"]))]=r
def best(ds,idx,b,v):
    c=[r for k,r in C.items() if k[0]==ds and k[1]==idx and k[2]==b and k[5]==v]
    return min(c,key=lambda r:r["cost"]) if c else None
arms=[]
def add(idx,ds,b,leaf,tau,v,why):
    t=(idx,ds,b,leaf,tau,v)
    if not any(a[:6]==t for a in arms): arms.append(t+(why,))
for ds in ALL:
    for b in (20,23,25):
        if ("ab4",)and (ds,"ab4",b,64,"3",0) in C: add("ab4",ds,b,64,"3",0,"base")
        if (ds,"ab4",b,64,"3",1) in C: add("ab4",ds,b,64,"3",1,"fit")
        r=best(ds,"ab4",b,1)
        if r: add("ab4",ds,b,int(r["leaf"]),r["tau"],1,"both")
        add("hf4",ds,b,0,"0",0,"hash")
        add("ph4",ds,b,65536,"0",0,"partition")
        if ds in HARD:
            r=best(ds,"ab4",b,0)
            if r: add("ab4",ds,b,int(r["leaf"]),r["tau"],0,"finer")
            for v in (0,1):
                if (ds,"ab4",b,4,"2",v) in C: add("ab4",ds,b,4,"2",v,"leaf4")
            add("ph4",ds,b,1024,"0",0,"partition")
        if ds=="genome":
            for v in (0,1):
                r=best(ds,"gr4",b,v)
                if r: add("gr4",ds,b,int(r["leaf"]),r["tau"],v,"rule")
arms.sort(key=lambda a:(a[1],a[2],a[0],a[3]))
with open(root+"/arms.txt","w") as f:
    for a in arms: f.write(" ".join(map(str,a))+"\n")
print(len(arms),"arms")
