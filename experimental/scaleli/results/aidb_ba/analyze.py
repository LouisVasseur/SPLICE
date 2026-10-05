import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json,itertools
B=REPO + "/experimental/scaleli/results/aidb_ba"
R=json.load(open(B+"/combined.json")); H=json.load(open(REPO + "/experimental/scaleli/results/aidb/hardness.json"))
M=['rmse','me','cd','p32','p4k']; HM={'rmse':'rmse','me':'max_error','cd':'conflict_degree','p32':'pla_32','p4k':'pla_4096'}
ds=[d for d in R if d!='fb_trim']
def tau(a,b):
  c=dd=0
  for i,j in itertools.combinations(range(len(a)),2):
    s=(a[i]-a[j])*(b[i]-b[j]); c+=s>0; dd+=s<0
  return (c-dd)/(len(a)*(len(a)-1)/2)
def norm(r,s,m):
  b=r[s]; v=b[m]
  return v/b['n'] if m in('rmse','me') else v
def ranks(have,s,m):
  o=sorted(have,key=lambda d:-norm(R[d],s,m)); return ' > '.join(o)
out={}
for s in ('free','mono','untr','csv'):
  have=[d for d in ds if s in R[d]]
  if len(have)<3: continue
  print(f'\n## scope {s} (n={len(have)})')
  for m in M:
    a=[norm(R[d],'raw',m) for d in have]; b=[norm(R[d],s,m) for d in have]
    t=tau(a,b); out[(s,m)]=t
    print(f" {m:5s} tau={t:+.3f}\n   raw: {ranks(have,'raw',m)}\n   {s}: {ranks(have,s,m)}")
  # also raw-scope tau restricted to same subset vs full_flow (inert) for reference
print('\n## ratios after/before (rmse,me per sequence length; cd, pla raw counts)')
for d in R:
  line=[f'{d:8s}']
  for s in ('free','untr','mono','csv'):
    if s in R[d]: line.append(s+': '+' '.join(f"{m}={norm(R[d],s,m)/norm(R[d],'raw',m):.3g}" for m in M))
  print(' | '.join(line))
print('\n## inert full_flow (hardness.json) vs our mono, ratio to raw')
for d in ds:
  print(d,' '.join(f"{m}: inert={H[d]['full_flow'][HM[m]]/H[d]['full'][HM[m]]:.3g} mono={R[d]['mono'][m]/R[d]['raw'][m]:.3g}" for m in M))
