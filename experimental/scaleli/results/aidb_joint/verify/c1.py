import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json,math,os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import indep as I
S=REPO + "/experimental/scaleli/data/samples"
R={r['dataset']:r for r in json.load(open(REPO + "/experimental/scaleli/results/aidb_joint/joint_results.json"))}
print("%-8s %8s %8s | %8s %8s %8s | %8s %8s %8s | mono(probe keys)"%("ds","binMe","binThey","linMe","linThey","d","flowMe","flowThey","d"))
for ds,r in R.items():
    keys=I.load(os.path.join(S,"%s_2M_uniform_s42"%ds))
    f=I.fences(keys); n=len(f); o=f[0]; sp=max(1,f[-1]-o)
    xs=[float(k-o)/float(sp) for k in f]
    raw=lambda k: float(k-o)/float(sp)
    m=I.LM().fit_xy(f,xs,[float(j) for j in range(n)])
    lin=I.root_probes(f,lambda k: m.predict_x(raw(k)))
    u=r['flow_units']
    def flow(k,u=u,o=o,sp=float(sp)):
        x=float(k-o)/sp
        return sum(c*math.tanh(g*(x-mm)) for (g,mm,c) in u)
    z=[flow(k) for k in f]
    mf=I.LM().fit_xy(f,z,[float(j) for j in range(n)])
    fl=I.root_probes(f,lambda k: mf.predict_x(flow(k)))
    pk=I.probe_keys(f); zp=[flow(k) for k in pk]
    inv=sum(1 for i in range(len(zp)-1) if zp[i+1]<zp[i])
    ties=sum(1 for i in range(len(zp)-1) if zp[i+1]==zp[i])
    bin_me=I.binary_probes(f)
    print("%-8s %8.4f %8.4f | %8.4f %8.4f %+7.4f | %8.4f %8.4f %+7.4f | inv=%d ties=%d"%(
        ds,bin_me,r['binary'],lin,r['baseline_linear'],lin-r['baseline_linear'],fl,r['flow_only'],fl-r['flow_only'],inv,ties))
