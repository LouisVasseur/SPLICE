import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Does the prototype's own inlined 'fast' greedy match its own literal
transcription (probe_metric.smooth_cdf)?  joint.py's docstring claims
'bit-identical ... Verified by verify.py' -- and verify.py does not exist."""
import json,math,os,sys
sys.path.insert(0,REPO + "/experimental/scaleli/results/aidb_joint")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import indep as I, probe_metric as pm, joint as J
S=REPO + "/experimental/scaleli/data/samples"
R={r['dataset']:r for r in json.load(open(REPO + "/experimental/scaleli/results/aidb_joint/joint_results.json"))}
ds=sys.argv[1]; alpha=float(sys.argv[2]); use_flow=sys.argv[3]=="flow"
keys=I.load(os.path.join(S,"%s_2M_uniform_s42"%ds))
f=I.fences(keys); n=len(f); o=f[0]; sp=float(max(1,f[-1]-o))
u=R[ds]['flow_units']
feat=(lambda k: sum(c*math.tanh(g*(float(k-o)/sp-m)) for (g,m,c) in u)) if use_flow else (lambda k: float(k-o)/sp)
x=[feat(k) for k in f]
a=J.smooth_cdf_fast(x,alpha); b=pm.smooth_cdf(x,alpha)
va=len(a.virtual_features); vb=len(b.virtual_features)
same=(list(a.slot)==list(b.slot))
def sc(slot,v):
    t=[float(s) for s in slot]; tab=I.slot_table(slot,n,v) if v else None
    m=I.LM().fit_xy(f,x,t)
    return I.root_probes(f,lambda k: m.predict_x(feat(k)),tab)
print("%s alpha=%g %s: fast v=%d probes=%.4f | literal v=%d probes=%.4f | slots identical=%s"%(
    ds,alpha,"flow" if use_flow else "raw",va,sc(a.slot,va),vb,sc(b.slot,vb),same))
