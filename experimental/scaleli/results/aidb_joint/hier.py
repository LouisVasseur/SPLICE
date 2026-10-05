import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import struct,sys,math
BASE=REPO + "/experimental/scaleli/data/samples/"
DS="books fb osm covid genome history libio planet stack wise".split()
RK=4096

def load(name):
    with open(BASE+name+"_2M_uniform_s42","rb") as f:
        b=f.read()
    n=struct.unpack_from("<Q",b,0)[0]
    keys=struct.unpack_from("<%dQ"%n,b,8)
    return keys

def fences(keys):
    return [keys[i] for i in range(0,len(keys),RK)]

def probe_keys(fx):
    ps=[]
    for j,f in enumerate(fx):
        ps.append(f)
        if j+1<len(fx): ps.append(f+(fx[j+1]-f)//2)
    return ps

def last_true_count(ok,lo,hi,cnt):
    # mirrors locate_from_prediction's final lower/last_true binary search; ok counts probes
    while lo<hi:
        m=lo+(hi-lo)//2
        if ok(m): lo=m+1
        else: hi=m
    return lo

def locate_count(fx,pred,k,n):
    cnt=[0]
    def ok(i):
        cnt[0]+=1
        return fx[i]<=k
    p=0 if pred<=0 else (n-1 if pred>=n-1 else int(pred))
    if ok(p):
        lo=p;step=1;hi=p+1
        while hi<n and ok(hi):
            lo=hi;step*=2;hi=min(n,p+step)
    else:
        hi=p;step=1;lo=p-step if p>step else 0
        while lo>0 and not ok(lo):
            hi=lo;step*=2;lo=p-step if p>step else 0
    # last_true(ok,lo,hi): while(hi-lo>1)
    a,b=lo,hi
    while b-a>1:
        m=a+(b-a)//2
        if ok(m): a=m
        else: b=m
    return cnt[0]

def fit(xs,ys):
    n=len(xs)
    mx=sum(xs)/n; my=sum(ys)/n
    sxx=sum((x-mx)**2 for x in xs); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    a=sxy/sxx if sxx>0 else 0.0
    return a, my-a*mx

def binary_probes(fx,ps,n):
    tot=0
    for k in ps:
        lo,hi=0,n
        while lo<hi:
            m=lo+(hi-lo)//2; tot+=1
            if fx[m]<=k: lo=m+1
            else: hi=m
    return tot/len(ps)

def flat_linear(fx,ps,n):
    xs=[float(x) for x in fx]; ys=[float(i) for i in range(n)]
    a,b=fit(xs,ys)
    return sum(locate_count(fx,a*float(k)+b,k,n) for k in ps)/len(ps)

def hierarchical(fx,ps,n,S):
    # CSV Alg-2 style / two-level: top linear model over fences maps key -> bucket,
    # each bucket has its own linear model fitted on its fence slice.
    xs=[float(x) for x in fx]
    a0,b0=fit(xs,[float(i) for i in range(n)])
    def bucket_of(k):
        y=a0*float(k)+b0
        bi=int(y*S/n)
        return 0 if bi<0 else (S-1 if bi>=S else bi)
    # assign fences to buckets, fit per-bucket line on (key -> global index)
    members=[[] for _ in range(S)]
    for i,f in enumerate(fx): members[bucket_of(f)].append(i)
    mods=[]
    for bi in range(S):
        m=members[bi]
        if len(m)>=2:
            mods.append(fit([float(fx[i]) for i in m],[float(i) for i in m]))
        elif len(m)==1:
            mods.append((0.0,float(m[0])))
        else:
            mods.append(None)
    # empty bucket: fall back to the global model
    tot=0
    for k in ps:
        bi=bucket_of(k); md=mods[bi]
        if md is None: pred=a0*float(k)+b0
        else: pred=md[0]*float(k)+md[1]
        tot+=locate_count(fx,pred,k,n)
    return tot/len(ps)

print("%-8s %7s %7s %7s %7s %7s %7s"%("ds","n","binary","linear","h/8","h/32","h/64"))
for d in DS:
    keys=load(d); fx=fences(keys); n=len(fx); ps=probe_keys(fx)
    row=[binary_probes(fx,ps,n), flat_linear(fx,ps,n)]
    for S in (8,32,64): row.append(hierarchical(fx,ps,n,S))
    print("%-8s %7d "%(d,n)+" ".join("%7.3f"%v for v in row))
