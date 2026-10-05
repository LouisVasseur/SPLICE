"""Independent transcription of index.hpp fit_root scoring + smoothing.hpp greedy.
Written from the C++ headers, NOT from the prototype's probe_metric.py."""
import struct, math
from array import array

def load(path):
    with open(path,'rb') as fh:
        (n,)=struct.unpack('<Q',fh.read(8))
        a=array('Q'); a.fromfile(fh,n)
    return a.tolist()

def fences(keys,rk=4096):
    return [keys[i] for i in range(0,len(keys),rk)]

def probe_keys(f):
    n=len(f); out=[]
    for j in range(n):
        out.append(f[j])
        if j+1<n: out.append(f[j]+(f[j+1]-f[j])//2)
    return out

# ---- LinearModel (model.hpp) ----
class LM:
    def __init__(self): self.origin=0; self.span=1; self.slope=0.0; self.intercept=0.0
    def normalized(self,k):
        return float(k-self.origin)/float(self.span) if k>=self.origin else -float(self.origin-k)/float(self.span)
    def predict_x(self,x):
        y=self.slope*x+self.intercept
        return y if math.isfinite(y) else 0.0
    def fit_xy(self,keys,x,t):
        self.origin=keys[0]; self.span=max(1,keys[-1]-keys[0])
        mx=my=xx=xy=0.0
        for i in range(len(x)):
            xi=float(x[i]); y=float(t[i]); nn=float(i+1)
            dx=xi-mx; dy=y-my
            mx+=dx/nn; my+=dy/nn
            xx+=dx*(xi-mx); xy+=dx*(y-my)
        self.slope=(xy/xx) if xx>0 else 0.0
        self.intercept=my-self.slope*mx
        return self

# ---- locate (index.hpp:337-347) with probe counting ----
def locate_count(f,n,k,y,table):
    cnt=0
    def ok(i):
        nonlocal cnt
        cnt+=1
        return True if i==0 else f[i]<=k
    if table:
        m=len(table)-1
        slot = 0 if y<=0 else (m if y>=float(m) else int(y))
        p=table[slot]
    else:
        p = 0 if y<=0 else (n-1 if y>=float(n-1) else int(y))
    if ok(p):
        lo,step,hi=p,1,p+1
        while hi<n and ok(hi):
            lo=hi; step*=2; hi=min(n,p+step)
    else:
        hi,step=p,1
        lo=p-step if p>step else 0
        while lo>0 and not ok(lo):
            hi=lo; step*=2; lo=p-step if p>step else 0
    while hi-lo>1:
        mm=lo+(hi-lo)//2
        if ok(mm): lo=mm
        else: hi=mm
    return cnt

def root_probes(f,predict,table=None):
    n=len(f); pk=probe_keys(f); tot=0
    for k in pk: tot+=locate_count(f,n,k,predict(k),table)
    return tot/float(len(pk))

def binary_probes(f):
    n=len(f); pk=probe_keys(f); tot=0
    for k in pk:
        lo,hi=0,n
        while lo<hi:
            m=lo+(hi-lo)//2; tot+=1
            if m==0 or f[m]<=k: lo=m+1
            else: hi=m
    return tot/float(len(pk))

def slot_table(slot,n,virt):
    tab=[0]*(n+virt)
    for j in range(n):
        end = slot[j+1] if j+1<n else n+virt
        for s in range(slot[j],end): tab[s]=j
    return tab

# ---- smooth_cdf, literal transcription of smoothing.hpp (double precision) ----
def smooth_cdf(x,alpha,max_ternary_steps=40,tie=1e-12):
    n=len(x)
    slot=list(range(n))
    seq=[(float(v),False) for v in x]
    def total():
        sn=sx=sxx=sy=syy=sxy=0.0
        for i in range(len(seq)):
            v=seq[i][0]; yy=float(i)
            sn+=1.0; sx+=v; sxx+=v*v; sy+=yy; syy+=yy*yy; sxy+=v*yy
        return [sn,sx,sxx,sy,syy,sxy]
    def sse(s):
        sn,sx,sxx,sy,syy,sxy=s
        if sn<2: return 0.0
        cxx=sxx-sx*sx/sn; cxy=sxy-sx*sy/sn; cyy=syy-sy*sy/sn
        v=(cyy-cxy*cxy/cxx) if cxx>0 else cyy
        return v if v>0 else 0.0
    base=total(); before=sse(base)
    budget=int(alpha*float(n))
    vf=[]
    if n<3 or not budget:
        return slot,vf,before,before,0
    m=len(seq)
    suf_x=[0.0]*(m+1); suf_y=[0.0]*(m+1)
    def rebuild():
        nonlocal suf_x,suf_y
        m=len(seq)
        suf_x=[0.0]*(m+1); suf_y=[0.0]*(m+1)
        for i in range(m-1,-1,-1):
            suf_x[i]=suf_x[i+1]+seq[i][0]; suf_y[i]=suf_y[i+1]+float(i)
    def loss_at(i,xv):
        sn,sx,sxx,sy,syy,sxy=base
        cnt=float(len(seq)-(i+1))
        sy=sy+cnt; syy=syy+2.0*suf_y[i+1]+cnt; sxy=sxy+suf_x[i+1]
        yy=float(i+1)
        sn+=1.0; sx+=xv; sxx+=xv*xv; sy+=yy; syy+=yy*yy; sxy+=xv*yy
        return sse([sn,sx,sxx,sy,syy,sxy])
    current=before; rounds=0
    for _ in range(budget):
        rebuild()
        best=current; best_i=0; best_x=0.0; found=False
        for i in range(len(seq)-1):
            lo=seq[i][0]; hi=seq[i+1][0]
            if not (hi>lo): continue
            w=hi-lo; e=w*1e-3
            l0=loss_at(i,lo+e); l1=loss_at(i,lo+2*e); r1=loss_at(i,hi-2*e); r0=loss_at(i,hi-e)
            if l1<l0 and r1<r0:
                a,b=lo+e,hi-e; it=0
                while it<max_ternary_steps and b-a>e:
                    m1=a+(b-a)/3.0; m2=b-(b-a)/3.0
                    if loss_at(i,m1)<loss_at(i,m2): b=m2
                    else: a=m1
                    it+=1
                cx=(a+b)/2.0; cl=loss_at(i,cx)
            elif l0<=r0: cx,cl=lo+e,l0
            else: cx,cl=hi-e,r0
            if cl<best-tie*max(1.0,best):
                best=cl; best_i=i; best_x=cx; found=True
        if not found: break
        seq.insert(best_i+1,(best_x,True)); vf.append(best_x)
        base=total(); current=sse(base); rounds+=1
    k=0
    for i in range(len(seq)):
        if not seq[i][1]: slot[k]=i; k+=1
    return slot,vf,before,current,rounds
