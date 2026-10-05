import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
# Double-double (two-float) accumulator: ~32 significant digits, beats x87 long double.
import math, probe_metric as pm

def tsum(a,b):
    s=a+b; bb=s-a; err=(a-(s-bb))+(b-bb); return s,err
def dd_add(hi,lo,x):
    s,e=tsum(hi,x); e+=lo; s,e2=tsum(s,e); return s,e2
def dd_val(hi,lo): return hi+lo

class DDSums:
    __slots__=("n","sx","sxx","sy","syy","sxy")
    def __init__(self,other=None):
        if other is None:
            self.n=0.0; self.sx=(0.0,0.0); self.sxx=(0.0,0.0); self.sy=(0.0,0.0); self.syy=(0.0,0.0); self.sxy=(0.0,0.0)
        else:
            self.n=other.n; self.sx=other.sx; self.sxx=other.sxx; self.sy=other.sy; self.syy=other.syy; self.sxy=other.sxy
    def add(self,x,y):
        self.n+=1.0
        self.sx=dd_add(*self.sx,x); self.sxx=dd_add(*self.sxx,x*x)
        self.sy=dd_add(*self.sy,y); self.syy=dd_add(*self.syy,y*y); self.sxy=dd_add(*self.sxy,x*y)
    def bump(self,attr,v): setattr(self,attr,dd_add(*getattr(self,attr),v))
    def sse(self):
        if self.n<2: return 0.0
        n=self.n; sx=dd_val(*self.sx); sxx=dd_val(*self.sxx); sy=dd_val(*self.sy); syy=dd_val(*self.syy); sxy=dd_val(*self.sxy)
        # centered recomputation from dd sums
        cxx=sxx-sx*sx/n; cxy=sxy-sx*sy/n; cyy=syy-sy*sy/n
        s=(cyy-cxy*cxy/cxx) if cxx>0 else cyy
        return s if s>0 else 0.0

pm.Sums=DDSums   # monkeypatch
BASE=REPO + "/experimental/scaleli/data/samples/%s_2M_uniform_s42"
for d,exp in [("fb",2.08)]:
    keys=pm.load_sample(BASE%d); f=pm.fences(keys)
    o=f[0]; sp=max(1,f[-1]-o); feat=lambda k: float(k-o)/float(sp)
    import time; t=time.time(); c,v=pm.candidate_cost(f,feat,root_alpha=4.0)
    print("%s dd: virt=%d cost=%.3f expected=%.2f (%.0fs)"%(d,v,c,exp,time.time()-t))
