import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json,math,os,sys,time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import indep as I
S=REPO + "/experimental/scaleli/data/samples"
def setup(ds):
    keys=I.load(os.path.join(S,"%s_2M_uniform_s42"%ds))
    f=I.fences(keys); n=len(f); o=f[0]; sp=max(1,f[-1]-o)
    return f,n,o,float(sp)
def score(f,feat,slot=None,virt=0):
    n=len(f); x=[feat(k) for k in f]
    if slot is None: t=[float(j) for j in range(n)]; tab=None
    else: t=[float(s) for s in slot]; tab=I.slot_table(slot,n,virt) if virt else None
    m=I.LM().fit_xy(f,x,t)
    return I.root_probes(f,lambda k: m.predict_x(feat(k)),tab)
ALPHA=4.0
for ds in sys.argv[1:]:
    f,n,o,sp=setup(ds)
    variants=[
      ("raw x            ", lambda k,o=o,sp=sp: float(k-o)/sp, {}),
      ("affine 1*x+0.3   ", lambda k,o=o,sp=sp: float(k-o)/sp+0.3, {}),
      ("affine 3*x       ", lambda k,o=o,sp=sp: 3.0*(float(k-o)/sp), {}),
      ("affine 1e-3*x    ", lambda k,o=o,sp=sp: 1e-3*(float(k-o)/sp), {}),
      ("affine 7*x-2     ", lambda k,o=o,sp=sp: 7.0*(float(k-o)/sp)-2.0, {}),
      ("x*(1+1e-12)      ", lambda k,o=o,sp=sp: (float(k-o)/sp)*(1.0+1e-12), {}),
      ("tie 1e-11        ", lambda k,o=o,sp=sp: float(k-o)/sp, {"tie":1e-11}),
      ("tie 1e-13        ", lambda k,o=o,sp=sp: float(k-o)/sp, {"tie":1e-13}),
      ("ternary 30 steps ", lambda k,o=o,sp=sp: float(k-o)/sp, {"max_ternary_steps":30}),
    ]
    print("== %s  alpha=%g"%(ds,ALPHA))
    vals=[]
    for name,feat,kw in variants:
        xs=[feat(k) for k in f]
        slot,vf,_,_,_=I.smooth_cdf(xs,ALPHA,**kw)
        p=score(f,feat,slot,len(vf)) if vf else score(f,feat)
        vals.append(p)
        print("   %s probes=%.4f  v=%d"%(name,p,len(vf)))
    print("   -> csv_only spread over mathematically-equivalent runs: %.4f (min %.4f max %.4f)"%(max(vals)-min(vals),min(vals),max(vals)))
