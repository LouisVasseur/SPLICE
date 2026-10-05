import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""On datasets where the fitted warp is pinned at the grid FLOOR (gain 0.01,
i.e. the objective wants the identity), the 'sequential' win must be a response
to an arbitrary infinitesimal curvature.  Probe a family of such curvatures."""
import math,os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import indep as I
S=REPO + "/experimental/scaleli/data/samples"
def score(f,feat,slot=None,virt=0):
    n=len(f); x=[feat(k) for k in f]
    if slot is None: t=[float(j) for j in range(n)]; tab=None
    else: t=[float(s) for s in slot]; tab=I.slot_table(slot,n,virt) if virt else None
    m=I.LM().fit_xy(f,x,t)
    return I.root_probes(f,lambda k: m.predict_x(feat(k)),tab)
ALPHA=4.0
for ds in sys.argv[1:]:
    keys=I.load(os.path.join(S,"%s_2M_uniform_s42"%ds))
    f=I.fences(keys); o=f[0]; sp=float(max(1,f[-1]-o))
    fam=[("identity", lambda x: x)]
    for eps in (1e-6,1e-4):
        for nm,g in (("-x^3",lambda x,e=eps: x-e*x*x*x),
                     ("+x^3",lambda x,e=eps: x+e*x*x*x),
                     ("-x^2",lambda x,e=eps: x-e*x*x),
                     ("+x^2",lambda x,e=eps: x+e*x*x)):
            fam.append(("%s eps=%.3g"%(nm,eps), g))
    print("== %s alpha=%g  (fitted warp is gain-floor => objective wants identity)"%(ds,ALPHA))
    vals=[]
    for nm,g in fam:
        feat=lambda k,g=g,o=o,sp=sp: g(float(k-o)/sp)
        xs=[feat(k) for k in f]
        if any(xs[i+1]<=xs[i] for i in range(len(xs)-1)):
            print("   %-16s NON-MONOTONE, skipped"%nm); continue
        slot,vf,_,_,_=I.smooth_cdf(xs,ALPHA)
        p=score(f,feat,slot,len(vf)) if vf else score(f,feat)
        if nm!="identity": vals.append(p)
        print("   %-16s probes=%.4f v=%d"%(nm,p,len(vf)))
    print("   -> spread of arbitrary infinitesimal warps: %.4f  (min %.4f max %.4f)"%(max(vals)-min(vals),min(vals),max(vals)))
