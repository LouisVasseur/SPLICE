import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json,glob,os,re,statistics as st,math
R=REPO + '/experimental/scaleli/results/aidb_warmup'
def real(tf):
    try:
        s=open(tf).read(); m=re.search(r'([\d.]+) real',s); return float(m.group(1)) if m else None
    except: return None
groups={}
for f in sorted(glob.glob(R+'/verify/*/*.json')+glob.glob(R+'/fullscale/cold/*.json')+glob.glob(R+'/fullscale/final/*.json')+glob.glob(R+'/verify/*.json')+glob.glob(R+'/fullscale/*.json')):
    try:j=json.load(open(f))
    except: continue
    if 'throughput_chunks_ops_s' not in j: 
        if 'throughput_ops_s' in j: print('nochunk',os.path.relpath(f,R),round(j['throughput_ops_s']),j.get('operations'),j.get('warmup_reads'),round(j['build_ns']/1e9,1),real(f[:-5]+'.time'))
        continue
    c=j['throughput_chunks_ops_s']; rest=c[1:]; m=st.median(rest)
    cv=st.pstdev(rest)/st.mean(rest) if len(rest)>1 else 0
    key=(os.path.dirname(os.path.relpath(f,R)),os.path.basename(j['dataset_path']),j['warmup_reads'],j['prefault'],j['operations'],j['chunks'],j.get('routing'),j.get('policy'))
    groups.setdefault(key,[]).append((j['throughput_ops_s'],c[0]/m-1,cv,j['build_ns']/1e9,real(f[:-5]+'.time'),os.path.basename(f),m))
allrel=[]
for k,v in groups.items():
    t=[x[0] for x in v]
    print(k,'n=',len(v))
    for x in v: print('   %-22s tput=%8.0f c1=%+6.1f%% chunkCV=%5.2f%% build=%5.1fs real=%s medrest=%8.0f'%(x[5],x[0],x[1]*100,x[2]*100,x[3],x[4],x[6]))
    if len(v)>1:
        mu=st.mean(t); sd=st.stdev(t)
        print('   between-run mean=%.0f sd=%.0f CV=%.2f%% range=%.1f%%  mean within-run chunkCV=%.2f%%'%(mu,sd,sd/mu*100,(max(t)-min(t))/mu*100,st.mean(x[2] for x in v)*100))
