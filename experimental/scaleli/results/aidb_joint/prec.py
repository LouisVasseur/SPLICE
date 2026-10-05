import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import probe_metric as pm
BASE=REPO + "/experimental/scaleli/data/samples/%s_2M_uniform_s42"
for d in ["fb","books","osm"]:
    keys=pm.load_sample(BASE%d); f=pm.fences(keys)
    x,o,s=pm.raw_features(f)
    S=pm.Sums()
    for i,xi in enumerate(x): S.add(xi,float(i))
    # magnitudes that cancel in the closed form
    print("%-6s sse=%12.6g  syy=%10.6g  sy^2/n=%10.6g  cyy=%10.6g  noise~%.3g"%(
        d,S.sse(),S.syy,S.sy*S.sy/S.n,S.syy-S.sy*S.sy/S.n, S.syy*2.22e-16))
