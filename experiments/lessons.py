"""Numbered meeting experiments. Python standard library only.
Teaching mechanisms are explicitly not named as complete paper reproductions.
"""
from __future__ import annotations
import bisect, hashlib, importlib.util, json, math, os, platform, random, statistics, struct, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'evidence/dense_sparse_fixture_uint64'
CORE=ROOT/'build/experimental/scaleli/scaleli_bench'
RS=ROOT/'build/radix_walkthrough'
TITLES={
 '01':'2018: rank prediction and a two-stage teaching model',
 '02':'2020: RadixSpline source-derived native experiment',
 '03':'2022: NFL published evidence and batching cost',
 '04':'Compression: real encodings in the experimental map',
 '05':'2024/25: virtual-point teaching experiment and CSV evidence',
 '06':'Project controls: byte geometry and update cost',
 '07':'Evidence audit: archived suite and user Mac pilot',
 '08':'Learnability controls: NFL-style transform and CSV-style virtual points',
 '09':'AIDB 2026 hardness protocol: our controls on the ten GRE datasets (reduced scale)'}

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def load_keys():
 b=FIXTURE.read_bytes();n=struct.unpack_from('<Q',b)[0]
 if len(b)!=8+8*n:raise ValueError('fixture length mismatch')
 a=list(struct.unpack_from(f'<{n}Q',b,8))
 if a!=sorted(set(a)):raise ValueError('fixture must be sorted and distinct')
 return a

def jsonrun(args,output):
 p=subprocess.run([str(x) for x in args],cwd=ROOT,capture_output=True,text=True,timeout=180)
 (output.with_suffix('.command.json')).write_text(json.dumps([str(x) for x in args],indent=2))
 if p.returncode:
  output.with_suffix('.stderr.txt').write_text(p.stderr);raise RuntimeError(p.stderr[-3000:])
 r=json.loads(p.stdout);output.write_text(json.dumps(r,indent=2));return r

def model(a,ys):
 origin=a[0];scale=max(1,a[-1]-origin);x=[(k-origin)/scale for k in a]
 mx=statistics.mean(x);my=statistics.mean(ys)
 xx=sum((v-mx)**2 for v in x)
 slope=sum((v-mx)*(y-my) for v,y in zip(x,ys))/xx if xx else 0
 return lambda k:slope*((k-origin)/scale)+(my-slope*mx)

def exact_correction(a,k,pred):
 """Teaching hint + expanding window + exact lower_bound; no bounded-error claim."""
 h=min(len(a)-1,max(0,int(pred)));w=1
 while True:
  lo=max(0,h-w);hi=min(len(a),h+w+1)
  if (lo==0 or a[lo-1]<k) and (hi==len(a) or a[hi]>=k):
   return bisect.bisect_left(a,k,lo,hi)
  w*=2

def lesson01(out):
 a=load_keys();n=len(a);rank=list(range(n));single=model(a,rank);leaves=16
 choose=lambda k:min(leaves-1,max(0,int(single(k)*leaves/n)))
 buckets=[[] for _ in range(leaves)]
 for i,k in enumerate(a):buckets[choose(k)].append((k,i))
 models=[model([k for k,i in b],[i for k,i in b]) if b else single for b in buckets]
 two=lambda k:models[choose(k)](k)
 one_err=[abs(single(k)-i) for i,k in enumerate(a)];two_err=[abs(two(k)-i) for i,k in enumerate(a)]
 q=a+[k+1 for k in a[::4]]+[0,a[-1]+1]
 for k in q:
  if exact_correction(a,k,two(k))!=bisect.bisect_left(a,k):raise AssertionError('teaching correction mismatch')
 sweep=[]
 for count in [1,16,128,512]:
  pick=lambda k:min(count-1,max(0,int(single(k)*count/n)))
  groups=[[] for _ in range(count)]
  for i,k in enumerate(a):groups[pick(k)].append((k,i))
  ms=[model([k for k,i in b],[i for k,i in b]) if b else single for b in groups]
  err=[abs(ms[pick(k)](k)-i) for i,k in enumerate(a)]
  sweep.append({'leaf_models':count,'mae_rank':statistics.mean(err),'max_error_rank':max(err)})
 samples=[{'key':a[i],'rank':i,'single_prediction':single(a[i]),'two_stage_prediction':two(a[i])} for i in range(0,n,64)]
 return {'status':'TEACHING DEMO, not original RMI','dataset':'bundled synthetic dense/sparse fixture','keys':n,
  'single_model_mae_rank':statistics.mean(one_err),'two_stage_mae_rank':statistics.mean(two_err),
  'single_model_max_rank_error':max(one_err),'two_stage_max_rank_error':max(two_err),
  'capacity_sweep':sweep,'root_models':1,'leaf_models':leaves,'exact_query_checks':len(q),'verified':True,'samples':samples,
  'interpretation':'More local models can reduce fitting error. This does not measure native throughput or reproduce the published RMI architecture/training.'}

def lesson02(out):
 return jsonrun([RS,FIXTURE],out/'native_radix_raw.json')

def lesson03(out):
 p=json.loads((ROOT/'evidence/paper_numbers.json').read_text());t=p['01'];batch=p['02']
 rows=[dict(r,ratio=r['with_NF_Mops_s']/r['without_NF_Mops_s']) for r in t['rows']]
 return {'status':'PUBLISHED NUMBERS + recomputed arithmetic; no NF run','rows':rows,'batch_transform':batch['rows'],
  'source':'sources/2205.11807v1.pdf, Table 1 p.3; Table 2 p.6; latency definition p.10',
  'verified':all(r['with_NF_Mops_s']>0 and r['without_NF_Mops_s']>0 for r in rows),
  'warning':'Table 1 does not state its operation mix or batch size. Do not silently assign settings from later experiments. Table 2 is transform-only average cost.'}

def core_run(out,name,policy,routing,profile='read_only',seed=42,load=1,extra=None):
 args=[CORE,'--data',FIXTURE,'--dtype','uint64','--format','sosd','--profile',profile,'--ops','10000',
 '--seed',str(seed),'--load-ratio',str(load),'--policy',policy,'--routing',routing,'--verify','1','--instrument','1','--warmup','1024']
 if extra:args+=extra
 return jsonrun(args,out/f'{name}_{seed}.json')

def lesson04(out):
 variants=[('raw_rank','raw','rank'),('packed_rank','min_bytes','rank'),('packed_byte','min_bytes','byte'),('packed_binary','min_bytes','binary')]
 rows=[]
 for seed in (11,29,47):
  group=[]
  for name,pol,route in variants:
   r=core_run(out,name,pol,route,seed=seed);r['variant']=name;rows.append(r);group.append(r)
  if len({r['trace_fingerprint'] for r in group})!=1 or len({r['result_checksum'] for r in group})!=1:raise AssertionError('unpaired compression controls')
 def summary(name):
  r=[x for x in rows if x['variant']==name]
  return {'variant':name,'bytes_per_record':statistics.median(x['memory_before']['accounted_bytes']/x['initial_rows'] for x in r),
  'throughput_ops_s':statistics.median(x['throughput_ops_s'] for x in r),
  'read_p99_ns':statistics.median(x['latency_ns']['read_hit']['p99'] for x in r)}
 return {'status':'LOCAL EXPERIMENTAL MAP; not LeCo/LA-vector benchmark','dataset':'bundled synthetic fixture','keys':16384,
 'source_rows':16384,'operations_per_trace':10000,'seeds':[11,29,47],'load_ratio':1,'verified':all(x['verified'] for x in rows),
 'trace_pairing_verified':True,'summary':[summary(v[0]) for v in variants],
 'warning':'Three small process runs; memory includes values and metadata. Not comparable to the keys-only contract of lesson 02. No statistical superiority claim.'}

def lesson05(out):
 vp=load_module('virtual_points',ROOT/'experimental/scaleli/tools/virtual_points_lab.py')
 a=[1,2,3,4,5,10,20,26,27,30];g=vp.greedy(a,3);e=vp.exhaustive(a,3);before=vp.fit_loss(a)[0]
 if not e['sse_all']<=g['sse_all']<=before:raise AssertionError('toy optimum ordering failure')
 return {'status':'TEACHING DEMO + separately labeled paper data; not optimized CSV','keys':a,'budget':3,'before_sse':before,
 'greedy':g,'exhaustive':e,'verified':True,
 'published_csv':json.loads((ROOT/'evidence/paper_numbers.json').read_text())['03'],
 'warning':'These ten keys and budget three are ours, not Figure 2 or Table 2 from the paper. SSE_all fits augmented ranks; original-key SSE is separate. No virtual key becomes a user record.'}

def lesson06(out):
 lm=load_module('layout',ROOT/'experimental/scaleli/tools/layout_lab.py');rows=[];identity=None;rank_mse=None
 for p in ['raw','min_bytes','smooth']:
  layout=out/f'layout_{p}.csv';r=core_run(out,'geometry_'+p,p,'byte',extra=['--dump-layout',str(layout)])
  v=lm.analyze(layout);current=v.pop('_identity')
  if identity is not None and current!=identity:raise AssertionError('keys/partitions changed')
  if rank_mse is not None and abs(v['rank_normalized_mse']-rank_mse)>1e-14:raise AssertionError('rank changed')
  identity=current;rank_mse=v['rank_normalized_mse'];v['policy']=p;rows.append(v)
 mutation=[]
 for p in ['raw','min_bytes']:
  r=core_run(out,'updates_'+p,p,'byte',profile='read_heavy',load=.75)
  mutation.append({'policy':p,'trace_fingerprint':r['trace_fingerprint'],'checksum':r['result_checksum'],'verified':r['verified'],
   'insert_p50_ns':r['latency_ns']['insert']['p50'],'insert_p99_ns':r['latency_ns']['insert']['p99'],
   'compactions':r['maintenance']['compactions'],'drain_ns':r['maintenance']['drain_ns']})
 if mutation[0]['trace_fingerprint']!=mutation[1]['trace_fingerprint'] or mutation[0]['checksum']!=mutation[1]['checksum']:raise AssertionError('mutation traces not paired')
 return {'status':'LOCAL EXPERIMENTAL MAP, not prior SOTA','verified':True,'logical_rank_invariance_verified':True,
 'geometry':rows,'update_controls':mutation,
 'warning':'Geometry has full initial fixture; dynamic experiment has 75% initial load. Geometry error and mutation timing are not one combined metric.'}

def lesson07(out):
 p=subprocess.run([sys.executable,str(ROOT/'experiments/audit_archived.py')],cwd=ROOT,capture_output=True,text=True,timeout=60)
 (out/'audit.txt').write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError(p.stderr)
 return {'status':'ARCHIVED RESULTS AUDIT, not a fresh performance run','verified':True,'audit_output':p.stdout,
 'mac_pilot':json.loads((ROOT/'evidence/mac_run_transcribed.json').read_text()),
 'warning':'Mac numbers are transcribed from the user, not measured in this runtime. Archived Linux results use a different generated dataset from the shipped binary fixture.'}

def lesson08(out):
 # Train a small NFL-shaped transform on the fixture in this run (seeded, stdlib), then
 # pair four learnability variants against the packed/rank control on identical traces.
 flow=out/'fixture_flow_2D2H2L.txt'
 p=subprocess.run([sys.executable,str(ROOT/'experimental/scaleli/tools/train_flow.py'),str(FIXTURE),'--output',str(flow),'--sample','4096','--steps','200','--monotone'],cwd=ROOT,capture_output=True,text=True,timeout=600)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 training=json.loads(p.stdout)
 variants=[('packed_rank',[]),('flow_bypass',['--flow',str(flow),'--flow-bypass','1']),('flow_forced',['--flow',str(flow),'--flow-bypass','0']),
  ('virtual_0.05',['--virtual-alpha','0.05']),('virtual_0.1',['--virtual-alpha','0.1']),('flow_bypass_virtual_0.1',['--flow',str(flow),'--flow-bypass','1','--virtual-alpha','0.1'])]
 rows=[]
 for seed in (11,29,47):
  group=[]
  for name,extra in variants:
   r=core_run(out,name,'min_bytes','rank',seed=seed,extra=extra);r['variant']=name;rows.append(r);group.append(r)
  if len({r['trace_fingerprint'] for r in group})!=1 or len({r['result_checksum'] for r in group})!=1:raise AssertionError('unpaired learnability controls')
 def summary(name):
  r=[x for x in rows if x['variant']==name];l=lambda x,k:x['learnability'][k]
  return {'variant':name,'throughput_ops_s':statistics.median(x['throughput_ops_s'] for x in r),
   'read_p99_ns':statistics.median(x['latency_ns']['read_hit']['p99'] for x in r),
   'fence_probes_per_op':statistics.median(x['work_counters']['fence_probes']/x['operations'] for x in r),
   'correction_per_op':statistics.median(x['work_counters']['correction_distance']/x['operations'] for x in r),
   'flow_regions':statistics.median(l(x,'flow_regions') for x in r),'regions':statistics.median(l(x,'regions') for x in r),
   'virtual_points':statistics.median(l(x,'virtual_points') for x in r),
   'rank_sse_ratio':statistics.median(l(x,'rank_sse_after')/l(x,'rank_sse_before') if l(x,'rank_sse_before')>0 else 1 for x in r),
   'preprocess_ms':statistics.median((l(x,'smoothing_ns')+l(x,'transform_ns'))/1e6 for x in r),
   'bytes_per_record':statistics.median(x['memory_before']['accounted_bytes']/x['initial_rows'] for x in r)}
 return {'status':'LOCAL EXPERIMENTAL MAP with clean-room NFL-style transform and CSV-style smoothing; NOT NFL/AFLI or CSV reproductions','dataset':'bundled synthetic fixture','keys':16384,
  'operations_per_trace':10000,'seeds':[11,29,47],'verified':all(x['verified'] for x in rows),'trace_pairing_verified':True,
  'flow_training':training,'summary':[summary(v[0]) for v in variants],
  'warning':'Compactions reuse the bulk-load flow decision and re-place existing virtual points; --relearn 1 repays the full search per rebuild (see the sweep). The transform is an 8-parameter tanh network trained here in seconds by a stand-in trainer (monotone variant: fractional-feature weights zeroed), used only as the model feature; records stay in key order. Virtual points change rank targets only and cost no arena bytes. The NFL auto-switch (tail conflict degree) decides per region. Read-only, single host, three seeds: no superiority claim.'}

AIDB=ROOT/'experimental/scaleli/results/aidb'
AIDB_COMMAND='python3 experimental/scaleli/tools/aidb_pipeline.py'
def lesson09(out):
 # Summarize the AIDB 2026 hardness-protocol lane if tools/aidb_pipeline.py has produced it. Nothing is
 # measured here: throughput.json (median Mops/s per variant and dataset, 2M-key uniform samples) and
 # scores.json (conformance/coverage of the 25 Table-2 metrics against OUR variants) are read back only.
 tp=AIDB/'throughput.json';sp=AIDB/'scores.json';html=AIDB/'report.html'
 if not(tp.exists() and sp.exists()):
  return {'status':'NOT RUN: '+AIDB_COMMAND,'verified':False,'datasets':[],'variants':[],'metrics':[],'throughput':[],'report_html':str(html),
   'warning':'experimental/scaleli/results/aidb/throughput.json and scores.json are absent. Run '+AIDB_COMMAND+' once the ten GRE downloads are complete (--wait polls; --dry-run exercises the lane on the bundled fixture without network). Nothing in this lane reproduces the AIDB 2026 paper, NFL/AFLI or CSV.'}
 T=json.loads(tp.read_text());S=json.loads(sp.read_text())
 variant='packed_rank' if 'packed_rank' in T else next(iter(T));per=T[variant];datasets=sorted(per)
 if not datasets:raise ValueError('throughput.json has no datasets for '+variant)
 scope='full' if 'full' in S else next(iter(S));metrics=S[scope].get('metrics',{})
 def pick(names):
  for n in names:
   if n in metrics:return n,metrics[n]
  return None,None
 rows=[]
 for label,names in [('PLA-32',('PLA-32',)),('CD',('CD',)),('GRE (PLA-32·PLA-4096)',('GRE','GRE (PLA-32·PLA-4096)','PLA-32·PLA-4096')),('CD·PLA-32',('CD·PLA-32',))]:
  name,m=pick(names);g=lambda k:(m or {}).get(k,float('nan'))
  rows.append({'metric':label,'found_as':name,'coverage':g('coverage'),'conformance':g('conformance'),'conformance_variant':((m or {}).get('per_variant') or {}).get(variant,float('nan')),
   'comparable_pairs':(m or {}).get('comparable_pairs'),'incomparable_pairs':(m or {}).get('incomparable_pairs')})
 run=json.loads((AIDB/'run.json').read_text()) if (AIDB/'run.json').exists() else {};st=run.get('settings',{})
 scale=f"{st.get('sample','?')}-key uniform samples, {st.get('ops','?')} lookups after {st.get('warmup','?')} warm-up, seeds {st.get('seeds','?')}" if st else 'see results/aidb/run.json'
 best=max(datasets,key=lambda d:per[d]);worst=min(datasets,key=lambda d:per[d])
 covered=set(S[scope].get('datasets',datasets))
 return {'status':'SUMMARY OF results/aidb: reduced-scale, single-host, clean-room controls; NOT a reproduction of the AIDB 2026 paper, NFL/AFLI or CSV',
  'verified':all(d in covered for d in datasets) and variant in (S[scope].get('variants') or [variant]),
  'datasets':datasets,'variants':list(T),'variant':variant,'scope':scope,'scopes':list(S),'scale':scale,'seeds':st.get('seeds',[]),
  'throughput':[{'dataset':d,'mops':per[d]} for d in datasets],'best':{'dataset':best,'mops':per[best]},'worst':{'dataset':worst,'mops':per[worst]},
  'all_variants':[{'dataset':d,**{v:T[v].get(d,float('nan')) for v in T}} for d in datasets],'metrics':rows,
  'report_html':str(html),'report_link':os.path.relpath(html,out.parent) if html.exists() else None,
  'warning':'Hardness metrics come from the full files (scope "'+scope+'"); throughput comes from '+scale+', not the paper\'s 200M keys and 100M lookups. The learned-index column is OUR experimental map and its NFL-style/CSV-style controls, not the seven indexes of the paper. Scores therefore describe how well each metric orders hardness for these controls only.'}

def run_lesson(id,out):
 out.mkdir(parents=True,exist_ok=True)
 r=globals()['lesson'+id](out);r['lesson']=id;r['title']=TITLES[id];r['environment']={'system':platform.platform(),'python':sys.version.split()[0],
 'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest()}
 for name,binary in [('radix',RS),('experimental',CORE)]:
  if binary.exists():r['environment'][name+'_binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
 (out/'result.json').write_text(json.dumps(r,indent=2));return r
