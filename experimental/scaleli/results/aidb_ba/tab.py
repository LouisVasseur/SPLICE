import json,os,sys,math
O=sys.argv[1]
ds="osm planet fb books genome covid history libio stack wise".split()
def blk(b): return dict(rmse=b['rmse'],me=b['max_error'],cd=b['conflict_degree'],p32=b['pla']['32'],p4k=b['pla']['4096'],n=b['keys'])
rows={}
def load(d,s,tag):
  p=f"{O}/{d}_{s}.json"
  if not os.path.exists(p): return {}
  j=json.load(open(p)); r={}
  r['raw'+tag]=blk(j['original']); r['raw'+tag]['dup']=j['duplicates']
  if j['transformed']:
    t=j['transformed']; b=blk(t); b.update(unord=t['unordered_pairs'],tdup=t['duplicates'],cdlit=t['fmcd']['conflict_degree_lipp_epsilon'],zmin=t['min'],zmax=t['max']); r[s]=b
  if j['smoothed']:
    t=j['smoothed']; b=blk(t); n0=j['keys']
    b.update(vp=t['virtual_points'],sm_s=t['smoothing_ns']/1e9,reg_rmse_before=math.sqrt(t['rank_sse_before']/n0),reg_rmse_after=math.sqrt(t['rank_sse_after']/t['keys']),cdlit=t['fmcd']['conflict_degree_lipp_epsilon']); r[s]=b
  return r
for d in ds:
  r={}
  for s in ('free','mono','untr','csv'): r.update(load(d,s,''))
  rows[d]=r
fb={}
for s in ('trimfree','trimmono','trimcsv'): fb.update(load('fb',s,'_trim'))
if fb: rows['fb_trim']={('raw' if k=='raw_trim' else k.replace('trim','')):v for k,v in fb.items()}
json.dump(rows,open(f"{O}/../combined.json","w"),indent=1)
for d,r in rows.items():
  for s,b in r.items(): print(d,s,{k:(round(v,4) if isinstance(v,float) else v) for k,v in b.items()})
