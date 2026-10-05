import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Robustness of the HEADLINE columns (flow_only, sequential) to the warp's
exact fitted gains, plus a held-out fit (warp fitted on half the fences)."""
import json,math,os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,REPO + "/experimental/scaleli/results/aidb_joint")
import indep as I
from joint import WarpBasis, GAINS
S=REPO + "/experimental/scaleli/data/samples"
R={r['dataset']:r for r in json.load(open(REPO + "/experimental/scaleli/results/aidb_joint/joint_results.json"))}
def score(f,feat,slot=None,virt=0):
    n=len(f); x=[feat(k) for k in f]
    if slot is None: t=[float(j) for j in range(n)]; tab=None
    else: t=[float(s) for s in slot]; tab=I.slot_table(slot,n,virt) if virt else None
    m=I.LM().fit_xy(f,x,t)
    return I.root_probes(f,lambda k: m.predict_x(feat(k)),tab)
for ds in sys.argv[1:]:
    keys=I.load(os.path.join(S,"%s_2M_uniform_s42"%ds))
    f=I.fences(keys); n=len(f); o=f[0]; sp=float(max(1,f[-1]-o))
    u=R[ds]['flow_units']
    print("== %s  linear=%.4f  reported flow_only=%.4f seq4=%.4f csv4=%.4f"%(
        ds,R[ds]['baseline_linear'],R[ds]['flow_only'],R[ds]['budgets']['4']['sequential'],R[ds]['budgets']['4']['csv_only']))
    for mult in (0.7,0.85,0.95,1.0,1.05,1.15,1.4):
        uu=[(g*mult,mm,c) for (g,mm,c) in u]
        feat=lambda k,uu=uu: sum(c*math.tanh(g*(float(k-o)/sp-mm)) for (g,mm,c) in uu)
        zs=[feat(k) for k in f]
        if any(zs[i+1]<=zs[i] for i in range(n-1)):
            print("   gain x%.2f  NON-MONOTONE"%mult); continue
        fo=score(f,feat)
        line="   gain x%.2f  flow_only=%.4f"%(mult,fo)
        if mult in (0.85,1.0,1.15):
            slot,vf,_,_,_=I.smooth_cdf(zs,4.0)
            sq=score(f,feat,slot,len(vf)) if vf else fo
            line+="   seq4=%.4f (v=%d)"%(sq,len(vf))
        print(line)
    # held-out: fit the warp on every other fence, evaluate on the full root
    sub=[f[i] for i in range(0,n,2)]
    xs_sub=[float(k-o)/sp for k in sub]
    b=WarpBasis(xs_sub,GAINS)
    fit=b.fit([float(j) for j in range(len(sub))])
    if fit is None:
        print("   held-out fit: none")
    else:
        _,unitsH=fit
        featH=lambda k,uu=unitsH: sum(c*math.tanh(g*(float(k-o)/sp-mm)) for (g,mm,c) in uu)
        zs=[featH(k) for k in f]
        ok=all(zs[i+1]>zs[i] for i in range(n-1))
        foH=score(f,featH) if ok else float('nan')
        slot,vf,_,_,_=I.smooth_cdf(zs,4.0)
        sqH=score(f,featH,slot,len(vf)) if vf else foH
        print("   HELD-OUT warp (fitted on 245 of 489 fences): units=%s flow_only=%.4f seq4=%.4f (v=%d) mono=%s"%(
            [[round(x,4) for x in uu] for uu in unitsH],foH,sqH,len(vf),ok))
