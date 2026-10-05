import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import random, probe_metric as pm
BASE=REPO + "/experimental/scaleli/data/samples/%s_2M_uniform_s42"
EXP={"fb":2.08,"books":2.60,"osm":8.18}
random.seed(42)
for d in ["fb","books","osm"]:
    keys=pm.load_sample(BASE%d); f=pm.fences(keys); n=len(f)
    o=f[0]; sp=max(1,f[-1]-o); feat=lambda k: float(k-o)/float(sp)
    x=[feat(k) for k in f]
    sm=pm.smooth_cdf(x,4.0); virt=len(sm.virtual_features)
    tgt=[float(s) for s in sm.slot]; tab=pm.slot_to_region_table(sm.slot,n,virt) if virt else None
    m=pm.LinearModel().fit_xy(f,x,tgt)
    insample=pm.root_probes(f, lambda k: m.predict_x(feat(k)), tab)
    # out-of-sample: real keys drawn uniformly from the dataset (what a workload does)
    qs=random.sample(range(len(keys)),200000)
    c=[0]
    for qi in qs:
        k=keys[qi]; ok=pm._ok_factory(f,k,c)
        pm.locate_from_prediction(ok, m.predict_x(feat(k)), n, tab)
    print("%-6s virt=%5d  in-sample(fence+mid)=%7.3f   workload(real keys)=%7.3f   quoted=%.2f"
          %(d,virt,insample,c[0]/len(qs),EXP[d]), flush=True)
