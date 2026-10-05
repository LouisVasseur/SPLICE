import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json,math,os,sys,time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import indep as I
S=REPO + "/experimental/scaleli/data/samples"
R={r['dataset']:r for r in json.load(open(REPO + "/experimental/scaleli/results/aidb_joint/joint_results.json"))}
def setup(ds):
    keys=I.load(os.path.join(S,"%s_2M_uniform_s42"%ds))
    f=I.fences(keys); n=len(f); o=f[0]; sp=max(1,f[-1]-o)
    xs=[float(k-o)/float(sp) for k in f]
    return f,n,o,float(sp),xs
def score(f,feat,slot=None,virt=0):
    n=len(f); x=[feat(k) for k in f]
    if slot is None:
        t=[float(j) for j in range(n)]; tab=None
    else:
        t=[float(s) for s in slot]; tab=I.slot_table(slot,n,virt) if virt else None
    m=I.LM().fit_xy(f,x,t)
    return I.root_probes(f,lambda k: m.predict_x(feat(k)),tab)
for ds in sys.argv[1:]:
    t0=time.time()
    f,n,o,sp,xs=setup(ds); r=R[ds]
    raw=lambda k,o=o,sp=sp: float(k-o)/sp
    u=r['flow_units']
    def flow(k,u=u,o=o,sp=sp):
        x=float(k-o)/sp
        return sum(c*math.tanh(g*(x-mm)) for (g,mm,c) in u)
    b4=r['budgets']['4']
    slot,vf,sb,sa,rounds=I.smooth_cdf(xs,4.0)
    csv=score(f,raw,slot,len(vf)) if vf else score(f,raw)
    z=[flow(k) for k in f]
    slotz,vfz,_,_,_=I.smooth_cdf(z,4.0)
    seq=score(f,flow,slotz,len(vfz)) if vfz else score(f,flow)
    print("%-8s csv4 me=%.4f(v=%d) they=%.4f(v=%d) d=%+.4f | seq4 me=%.4f(v=%d) they=%.4f(v=%d) d=%+.4f   [%.0fs]"%(
        ds,csv,len(vf),b4['csv_only'],b4['csv_virtual'],csv-b4['csv_only'],
        seq,len(vfz),b4['sequential'],b4['seq_virtual'],seq-b4['sequential'],time.time()-t0))
