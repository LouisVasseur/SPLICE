import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json,math
B=REPO + "/experimental/scaleli/results/aidb_ba"
R=json.load(open(B+"/combined.json"))
lab={'raw':'BEFORE (raw keys)','free':'AFTER NFL, faithful flow (sort by z)','untr':'control: UNTRAINED sawtooth flow (sort by z)','mono':'AFTER NFL, monotone flow','csv':'AFTER CSV, a=0.1 per 4,096-key region'}
def f(v,big=True):
  if v>=1e6: return f"{v/1e6:.2f}M"
  if v>=1e4: return f"{v/1e3:.1f}k"
  return f"{v:,.0f}" if v>=100 else f"{v:.3g}"
print("| dataset | scope | n (seq.) | RMSE | RMSE/n | ME | ME/n | CD | PLA-32 | PLA-4096 | extra |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for d,r in R.items():
  for s in ('raw','free','untr','mono','csv'):
    if s not in r: continue
    b=r[s]; ex=''
    if s in('free','untr','mono'): ex=f"descents={b['unord']}, z-dups={b['tdup']}, CD(lit 1e-6)={b['cdlit']}"
    if s=='csv': ex=f"VP={b['vp']:,} ({100*b['vp']/(b['n']-b['vp']):.1f}% of n), region RMSE {b['reg_rmse_before']:.1f}->{b['reg_rmse_after']:.1f}, smoothing {b['sm_s']:.0f}s, CD(lit)={b['cdlit']}"
    print(f"| {d} | {s} | {b['n']/1e6:.1f}M | {f(b['rmse'])} | {100*b['rmse']/b['n']:.2f}% | {f(b['me'])} | {100*b['me']/b['n']:.1f}% | {b['cd']:,} | {b['p32']:,} | {b['p4k']:,} | {ex} |")
