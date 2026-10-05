import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import probe_metric as pm
BASE=REPO + "/experimental/scaleli/data/samples/%s_2M_uniform_s42"
EXP={"fb":2.3,"history":3.8,"covid":4.6,"wise":5.7,"libio":6.5,"genome":8.6,"books":11.1,"osm":11.7,"planet":12.6}
print("%-9s %5s %8s %8s"%("dataset","n","linear","expected"))
for d in ["fb","history","stack","covid","wise","libio","genome","books","osm","planet"]:
    keys=pm.load_sample(BASE%d)
    f=pm.fences(keys)
    x,origin,span=pm.raw_features(f)
    m=pm.LinearModel().fit_xy(f,x,[float(j) for j in range(len(f))])
    c=pm.root_probes(f, lambda k,m=m,o=origin,s=span: m.predict_x(float(k-o)/float(s)))
    print("%-9s %5d %8.3f %8s"%(d,len(f),c,EXP.get(d,"-")))
print("binary baseline:", pm.binary_probes(489))
