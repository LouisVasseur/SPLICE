import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import time, probe_metric as pm
BASE=REPO + "/experimental/scaleli/data/samples/%s_2M_uniform_s42"
EXP={"books":2.60,"osm":8.18,"fb":2.08}
for d in ["fb","books","osm"]:
    keys=pm.load_sample(BASE%d); f=pm.fences(keys)
    origin=f[0]; span=max(1,f[-1]-origin)
    feat=lambda k: float(k-origin)/float(span)
    t=time.time(); c,v=pm.candidate_cost(f,feat,root_alpha=4.0); dt=time.time()-t
    print("%-6s alpha=4  virt=%5d  cost=%7.3f  expected=%.2f  (%.0fs)"%(d,v,c,EXP[d],dt), flush=True)
