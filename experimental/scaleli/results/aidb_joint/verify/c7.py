import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json,math,os,sys
sys.path.insert(0,REPO + "/experimental/scaleli/results/aidb_joint")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import indep as I, probe_metric as pm, joint as J
S=REPO + "/experimental/scaleli/data/samples"
R={r['dataset']:r for r in json.load(open(REPO + "/experimental/scaleli/results/aidb_joint/joint_results.json"))}
ds="osm"
keys=I.load(os.path.join(S,"%s_2M_uniform_s42"%ds))
f=I.fences(keys); n=len(f); o=f[0]; sp=max(1,f[-1]-o)
xs=[float(k-o)/float(sp) for k in f]
print("JSON flow_units full precision:",R[ds]['flow_units'])
# refit the warp exactly as run_dataset does
basis=J.WarpBasis(xs,J.GAINS)
fit=basis.fit([float(j) for j in range(n)])
sse0,units0=fit
print("refit units:",units0)
flow0=J.make_flow_fn(units0,o,sp)
z=[flow0(k) for k in f]
sm=J.smooth_cdf_fast(z,4.0); v=len(sm.virtual_features)
print("v=",v)
p_pm=J.score(f,flow0,sm.slot,v)
tab_pm=pm.slot_to_region_table(sm.slot,n,v); tab_me=I.slot_table(sm.slot,n,v)
print("tables identical:",tab_pm==tab_me)
m=I.LM().fit_xy(f,z,[float(s) for s in sm.slot])
p_me=I.root_probes(f,lambda k: m.predict_x(flow0(k)),tab_me)
print("prototype score()=%.4f   my score=%.4f   reported sequential=%.4f"%(p_pm,p_me,R[ds]['budgets']['4']['sequential']))
