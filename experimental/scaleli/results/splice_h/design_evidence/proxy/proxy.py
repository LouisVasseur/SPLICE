# Counts (no timing): how do SSE, log2-error, NLL and lines-touched rank the same CDF models?
import numpy as np, sys, json, os
SAMP="/Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/data/samples"
DS=["books","fb","osm","covid","genome","history","libio","planet","stack","wise"]
rng=np.random.default_rng(1)
def load(d):
    with open(os.path.join(SAMP,f"{d}_2M_uniform_s42"),"rb") as f:
        n=int(np.frombuffer(f.read(8),"<u8")[0]); k=np.frombuffer(f.read(8*n),"<u8")
    return np.unique(k)
def pwl(xs,kx,ky):  # monotone piecewise linear through knots (kx strictly incr)
    return np.interp(xs,kx,ky)
def models(x):
    n=len(x); xf=(x-x[0]).astype(np.float64); r=np.arange(n,dtype=np.float64)
    out={}
    a,b=np.polyfit(xf,r,1); out["LS1"]=(a*xf+b, None)
    for K in (64,512,4096):
        idx=np.unique(np.linspace(0,n-1,K+1).round().astype(int))
        out[f"Qknot{K}"]=(pwl(xf,xf[idx],r[idx]), (xf[idx],r[idx]))
        kx=np.linspace(0,xf[-1],K+1); ky=np.searchsorted(xf,kx).astype(float)
        out[f"Wknot{K}"]=(pwl(xf,kx,ky),(kx,ky))
    # discontinuous LS per equal-count region
    for K in (512,):
        p=np.empty(n); idx=np.linspace(0,n,K+1).astype(int)
        for s,e in zip(idx[:-1],idx[1:]):
            if e-s<2: p[s:e]=r[s:e]; continue
            a,b=np.polyfit(xf[s:e],r[s:e],1); p[s:e]=a*xf[s:e]+b
        out[f"LSseg{K}"]=(p,None)
    return xf,r,out
def nll(xf,knots,n):
    if knots is None: return float("nan")
    kx,ky=knots; seg=np.clip(np.searchsorted(kx,xf,side="right")-1,0,len(kx)-2)
    dens=(ky[seg+1]-ky[seg])/np.maximum(kx[seg+1]-kx[seg],1e-300)/n
    dens=np.maximum(dens,1e-300); return float(-np.mean(np.log(dens)))
def exp_lines(pred,true,n,per_line=4,m=40000):
    sel=rng.choice(len(true),size=min(m,len(true)),replace=False)
    p=np.clip(np.floor(pred[sel]).astype(np.int64),0,n-1); t=true[sel].astype(np.int64)
    lines=[p//per_line]; probes=1
    # exponential search outward toward t, then binary search; record lines of every probe
    d=t-p; s=np.sign(d); s[s==0]=1; ad=np.abs(d)
    step=np.ones_like(p); lo=np.zeros_like(p); done=ad==0
    while not done.all():
        nxt=np.where(done,p,np.clip(p+s*step,0,n-1)); lines.append(np.where(done,lines[0],nxt//per_line))
        reached=(~done)&(step>=ad); lo=np.where((~done)&~reached,step,lo)
        hi_final=np.where(reached,step,0)
        # binary search in (lo,step]: count probes ~ log2(step-lo)
        if reached.any():
            L=lo[reached].copy();H=hi_final[reached].copy();T=ad[reached];P=p[reached];S=s[reached]
            extra=[]
            while True:
                act=H-L>1
                if not act.any(): break
                M=(L+H)//2; ln=np.where(act,(P+S*M)//per_line,(P+S*H)//per_line); extra.append((reached.nonzero()[0],ln))
                ge=M>=T; H=np.where(act&ge,M,H); L=np.where(act&~ge,M,L)
            for ix,ln in extra:
                col=lines[0].copy(); col[ix]=ln; lines.append(col)
        done=done|reached; step=step*2
    Lm=np.stack(lines,1); Lm.sort(1); distinct=1+(np.diff(Lm,1)!=0).sum(1)
    return float(distinct.mean())
def gapped_lines(pred,n,alpha=0.25,per_line=4):
    S=(1+alpha); h=np.floor(pred*S).astype(np.int64); h=np.maximum(h,0)
    pos=h.copy()
    # order-preserving placement: pos_i=max(h_i,pos_{i-1}+1)  (cummax trick)
    pos=np.maximum.accumulate(h-np.arange(n))+np.arange(n)
    lines=np.abs(pos//per_line-h//per_line)+1
    return float(lines.mean()), float((pos//per_line!=h//per_line).mean())
if __name__!="__main__": raise SystemExit
res={}
for d in DS:
    x=load(d); n=len(x); xf,r,M=models(x); res[d]={}
    for name,(pred,kn) in M.items():
        e=pred-r
        res[d][name]=dict(mse=float(np.mean(e*e)),log2=float(np.mean(np.log2(np.abs(e)+1))),
            maxe=float(np.abs(e).max()),nll=nll(xf,kn,n),lines_sorted=exp_lines(pred,r,n),
            gapped=gapped_lines(np.clip(pred,0,n-1),n))
    print(d,flush=True)
json.dump(res,open(os.path.join(os.path.dirname(__file__),"proxy.json"),"w"),indent=1)
