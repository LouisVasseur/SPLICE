import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import math,os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import indep as I
S=REPO + "/experimental/scaleli/data/samples"
ds,shape,eps=sys.argv[1],sys.argv[2],float(sys.argv[3])
keys=I.load(os.path.join(S,"%s_2M_uniform_s42"%ds))
f=I.fences(keys); n=len(f); o=f[0]; sp=float(max(1,f[-1]-o))
g={'id':lambda x:x,'-x3':lambda x:x-eps*x**3,'+x3':lambda x:x+eps*x**3,
   '-x2':lambda x:x-eps*x*x,'+x2':lambda x:x+eps*x*x}[shape]
feat=lambda k: g(float(k-o)/sp)
xs=[feat(k) for k in f]
slot,vf,_,_,_=I.smooth_cdf(xs,4.0)
t=[float(s) for s in slot]; tab=I.slot_table(slot,n,len(vf)) if vf else None
m=I.LM().fit_xy(f,xs,t)
print("%s %s eps=%g -> probes=%.4f v=%d"%(ds,shape,eps,I.root_probes(f,lambda k: m.predict_x(feat(k)),tab),len(vf)))
