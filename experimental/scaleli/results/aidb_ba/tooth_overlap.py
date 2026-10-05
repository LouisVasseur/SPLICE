import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
# Weights-only analysis of the faithful (non-monotone) flows: no data read, no index built.
# z(x) = sum_c out_c, x=(key-mean)/var in [0,s]; hidden = tanh(x*W0[0]+frac(x)*W0[1]); out = hidden @ W1.
# For each tooth i (x in [i,i+1)) z is evaluated on a grid; we report how many teeth overlap a given z,
# i.e. how many distinct key bands a z-ordered block interleaves.
import math,json,sys
D=REPO + "/experimental/scaleli/results/aidb_flowv2/flows_free/"
best={"books":64,"fb":512,"osm":512,"covid":64,"genome":64,"history":64,"libio":64,"planet":512,"stack":64,"wise":64}
def load(p):
    t=open(p).read().split()
    i=0
    def nx():
        nonlocal i; v=t[i]; i+=1; return v
    ind,hid,lay=int(nx()),int(nx()),int(nx()); mean,var=float(nx()),float(nx())
    W=[]
    for l in range(lay):
        r,c=int(nx()),int(nx()); W.append([[float(nx()) for _ in range(c)] for _ in range(r)])
    return W
def z(W,x):
    a=[x,x-math.floor(x)]
    for l,M in enumerate(W):
        b=[sum(a[r]*M[r][c] for r in range(len(M))) for c in range(len(M[0]))]
        a=b if l==len(W)-1 else [math.tanh(v) for v in b]
    return sum(a)
out={}
for d,s in best.items():
    W=load(D+f"{d}_s{s}_t2000.txt")
    G=50
    lo=[];hi=[]
    for i in range(s):
        zs=[z(W,i+(g+0.5)/G) for g in range(G)]
        mono=all(zs[k]<zs[k+1] for k in range(G-1))
        lo.append(min(zs));hi.append(max(zs))
    zmin,zmax=min(lo),max(hi)
    # average over a uniform grid of z of the number of teeth covering it, weighted by tooth density
    # (each tooth equally populated: uniform keys); mean coverage seen by a key = sum_i len_i^2/(sum overlap) approx:
    M=2000; cov=[0]*M
    for i in range(s):
        a=int((lo[i]-zmin)/(zmax-zmin)*(M-1)); b=int((hi[i]-zmin)/(zmax-zmin)*(M-1))
        for k in range(a,b+1): cov[k]+=1
    # key-weighted mean coverage: a key in tooth i at z sees cov(z); integrate over teeth uniformly
    tot=0;cnt=0
    for i in range(s):
        a=int((lo[i]-zmin)/(zmax-zmin)*(M-1)); b=int((hi[i]-zmin)/(zmax-zmin)*(M-1))
        for k in range(a,b+1): tot+=cov[k];cnt+=1
    within=(hi[0]-lo[0]); shift=(lo[0]-lo[-1])/(s-1)
    out[d]={"shifts":s,"tooth_z_width":round(within,4),"tooth_to_tooth_shift":round(shift,6),
            "max_teeth_overlapping":max(cov),"mean_teeth_seen_by_a_key":round(tot/cnt,1),
            "monotone_within_tooth":mono}
    print(d,out[d])
json.dump(out,open("tooth_overlap.json","w"),indent=1)
