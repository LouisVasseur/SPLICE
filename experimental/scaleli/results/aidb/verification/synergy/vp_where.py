#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Port of smooth_cdf's greedy (first R rounds) on the root fences with the raw feature: where do the virtual fences go?"""
import array, struct, pathlib, math, sys
ROOT = pathlib.Path(REPO + "/experimental/scaleli")
class Sums:
    __slots__ = ("n","sx","sxx","sy","syy","sxy")
    def __init__(s): s.n=s.sx=s.sxx=s.sy=s.syy=s.sxy=0.0
    def add(s,x,y): s.n+=1; s.sx+=x; s.sxx+=x*x; s.sy+=y; s.syy+=y*y; s.sxy+=x*y
    def copy(s):
        t=Sums(); t.n,t.sx,t.sxx,t.sy,t.syy,t.sxy=s.n,s.sx,s.sxx,s.sy,s.syy,s.sxy; return t
    def sse(s):
        if s.n<2: return 0.0
        cxx=s.sxx-s.sx*s.sx/s.n; cxy=s.sxy-s.sx*s.sy/s.n; cyy=s.syy-s.sy*s.sy/s.n
        v = cyy-cxy*cxy/cxx if cxx>0 else cyy
        return v if v>0 else 0.0
def greedy(x, rounds):
    seq=[(v,False) for v in x]
    def total():
        s=Sums()
        for i,(v,_) in enumerate(seq): s.add(v,float(i))
        return s
    base=total(); current=base.sse(); sse0=current; where=[]
    for r in range(rounds):
        m=len(seq); suf_x=[0.0]*(m+1); suf_y=[0.0]*(m+1)
        for i in range(m-1,-1,-1): suf_x[i]=suf_x[i+1]+seq[i][0]; suf_y[i]=suf_y[i+1]+float(i)
        def loss_at(i,xv):
            s=base.copy(); cnt=float(m-(i+1)); s.sy+=cnt; s.syy+=2*suf_y[i+1]+cnt; s.sxy+=suf_x[i+1]; s.add(xv,float(i+1)); return s.sse()
        best=current; bi=0; bx=0.0; found=False
        for i in range(m-1):
            lo,hi=seq[i][0],seq[i+1][0]
            if not hi>lo: continue
            w=hi-lo; e=w*1e-3
            l0=loss_at(i,lo+e); l1=loss_at(i,lo+2*e); r1=loss_at(i,hi-2*e); r0=loss_at(i,hi-e)
            if l1<l0 and r1<r0:
                a,b=lo+e,hi-e
                for _ in range(40):
                    if b-a<=e: break
                    m1=a+(b-a)/3; m2=b-(b-a)/3
                    if loss_at(i,m1)<loss_at(i,m2): b=m2
                    else: a=m1
                cx=(a+b)/2; cl=loss_at(i,cx)
            elif l0<=r0: cx,cl=lo+e,l0
            else: cx,cl=hi-e,r0
            if cl<best-1e-12*max(1.0,best): best=cl; bi=i; bx=cx; found=True
        if not found: break
        # which original gap? count real fences at or before position bi
        real_before=sum(1 for j in range(bi+1) if not seq[j][1])
        where.append(real_before-1)  # gap index g means between real fence g and g+1
        seq.insert(bi+1,(bx,True)); base=total(); current=base.sse()
    return sse0, current, where
for nm in sys.argv[1:]:
    p=ROOT/f"data/samples/{nm}_2M_window_s42"
    with open(p,"rb") as fh:
        n=struct.unpack("<Q",fh.read(8))[0]; keys=array.array("Q"); keys.fromfile(fh,n)
    fences=[0]+[keys[i] for i in range(4096,n,4096)]
    span=fences[-1]; x=[f/span for f in fences]
    R=150
    sse0,sse1,where=greedy(x,R)
    in_sentinel=sum(1 for g in where if g==0)
    print(f"{nm}_window raw feature: first {len(where)} virtual fences -> {in_sentinel} in the sentinel gap [0, first key), {len(where)-in_sentinel} elsewhere; SSE {sse0:.3e} -> {sse1:.3e}")
    # same on the raw feature normalized from the first real fence (sentinel excluded from the sequence)
    x2=[(f-fences[1])/(fences[-1]-fences[1]) for f in fences[1:]]
    sse0,sse1,where=greedy(x2,R)
    print(f"{nm}_window raw feature without sentinel: {len(where)} virtual fences placed (stops early if fewer than {R}); SSE {sse0:.3e} -> {sse1:.3e}; distinct gaps used {len(set(where))}")
