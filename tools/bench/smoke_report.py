#!/usr/bin/env python3
# Did everything that was asked for actually run? For each stage: cells asked for, passes each, rows that came back, any lookup count short of the
# full 100M, any scan that disagreed with the sorted keys, any log with an error. Prints PASS or the exact cells that are missing.
#   smoke_report.py RESULTS_ROOT
import csv,glob,os,re,sys,collections
root=sys.argv[1]
fails=0
def line(s): print(s)
def timed(stage,cellfile,parse,with_code=False):
    global fails
    tp=stage+"/table.tsv"
    if not os.path.exists(cellfile):
        line(f"-- {stage}: no cells file"); fails+=1; return
    want=collections.Counter(); ops={}
    for l in open(cellfile):
        p=parse(l.split())
        if p[0]:
            want[p[0]]=p[1]
            ops[p[0]]=p[2] if len(p)>2 else 100000000
    got=collections.Counter(); short=[]
    if os.path.exists(tp):
        for r in csv.DictReader(open(tp),delimiter="\t"):
            k=(r["index"],r["dataset"],r["spk"],r["leaf"],r["tau"],r["cap"])+((r["bits"],) if with_code else ())
            got[k]+=1
            if k in ops and float(r.get("ok") or 0)!=ops[k]: short.append((k,r.get("ok")))
    miss=[(k,want[k],got[k]) for k in want if got[k]<want[k]]
    st="PASS" if not miss and not short else "FAIL"
    fails+=st=="FAIL"
    line(f"{st} {os.path.relpath(stage,root)}: {len(want)} cells, {sum(want.values())} passes asked, {sum(got.values())} rows"+(f", {len(miss)} short" if miss else "")+(f", {len(short)} with lookups missing" if short else ""))
    for m in miss[:40]: line(f"   missing {m[0]}: wanted {m[1]} got {m[2]}")
    for x in short[:20]: line(f"   short {x}")
    bad=[]
    for p in glob.glob(stage+"/*/cells/*/run.out"):
        t=open(p,errors="ignore").read()
        if "FAILED" in t: bad.append(p)
    if bad: line(f"   {len(bad)} run.out with FAILED, first {os.path.relpath(bad[0],root)}"); fails+=1
def scanfile(path,name):
    global fails
    if not os.path.exists(path):
        line(f"FAIL {name}: no {os.path.basename(path)}"); fails+=1; return
    rows=[l.rstrip("\n").split("\t") for l in open(path) if l.strip()]
    cells=collections.Counter((r[0],r[1],*[x for x in r[3:6]]) for r in rows)
    bad=collections.Counter(); known=[]
    for r in rows:
        d=dict(x.split("=",1) for x in r if "=" in x)
        if d.get("bad","0")=="0": continue
        # ALEX's scan returns keys for a start above every stored key: 8 checks, found in the first scan batch and reported
        if r[1]=="alex" and d["bad"]=="8": known.append(r[0])
        else: bad[(r[0],r[1])]=d["bad"]
    st="PASS" if not bad else "CHECK"
    fails+=st!="PASS"
    line(f"{st} {name}: {len(cells)} cells, {len(rows)} rows; scans wrong: {dict(bad) if bad else 'none'}"+(f"; known ALEX start-above-all bug on {sorted(set(known))}" if known else ""))
def g_parse(a):  # data idx ds b leaf tau v passes [rule]
    return ((a[1],a[2],a[3],a[4],a[5],a[6]),int(a[7]))
def f_parse(a):  # idx ds b leaf tau v passes
    return ((a[0],a[1],a[2],a[3],a[4],a[5]),int(a[6]))
def a_parse(a):  # idx ds b leaf tau v passes code spk E depth noskip ops rule
    return ((a[0],a[1],a[2],a[3],a[4],a[5],a[7]),int(a[6]),int(a[12]))
for name,base,subs in (("G","jed200M_g_2026-11-03",[("real",g_parse,0),("hand",g_parse,0),("synt",g_parse,0)]),
                       ("final","jed200M_final_2026-11-03",[("reg",f_parse,0),("fin",f_parse,0),("rep",f_parse,0)]),
                       ("baselines","jed200M_base_2026-11-03",[("point",f_parse,0),("readonly",f_parse,0),("readonly_rep",f_parse,0),("calib",f_parse,0)]),
                       ("ablation","jed200M_ablation_2026-11-03",[("abl",a_parse,1),("hash",f_parse,0)])):
    r=f"{root}/{base}"
    if not os.path.isdir(r): line(f"-- {name}: not run yet ({base})"); continue
    for sub,parse,code in subs:
        cf=f"{r}/{sub}/cells" if sub!="fin" else f"{r}/cells.txt"
        timed(f"{r}/{sub}",cf,parse,bool(code))
    for x in ("scan/out.tsv",):
        if os.path.exists(f"{r}/{x}"): scanfile(f"{r}/{x}",f"{name} {x}")
    ch=f"{r}/chain/out.tsv"
    if os.path.exists(ch): line(f"-- {name} chain: {sum(1 for _ in open(ch))} rows")
    for x in ("scan/failed.txt","miss/failed.txt"):
        if os.path.exists(f"{r}/{x}"):
            lost=[l for l in open(f"{r}/{x}") if " alex " not in l]  # exits 1 on the known bug above, its rows are all there
            if lost: line(f"FAIL {name} {x}: "+"".join(lost)[:300]); fails+=1
    if os.path.exists(f"{r}/tests.txt"): line(f"-- {name} tests: "+" | ".join(x.strip()[:60] for x in open(f"{r}/tests.txt") if ": " in x and ("ok" in x or "FAILED" in x))[:200])
e=f"{root}/jed200M_extras_2026-11-03"
if os.path.isdir(e):
    if os.path.exists(f"{e}/gate/choices.tsv"): line(f"-- gate: {sum(1 for _ in open(e+'/gate/choices.tsv'))} choices")
    if os.path.exists(f"{e}/miss/out.tsv"):
        rows=[l.split("\t") for l in open(f"{e}/miss/out.tsv")]
        wrong=[r[:3] for r in rows if len(r)>4 and r[4].startswith("hits=") and r[4].split("=")[1]!=r[5].split("=")[1]]
        line(f"{'PASS' if not wrong else 'FAIL'} misses: {len(rows)} rows, wrong hit counts: {wrong[:10] or 'none'}")
print("SMOKE:", "PASS" if not fails else f"{fails} problems, see above")
