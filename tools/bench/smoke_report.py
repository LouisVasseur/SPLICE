#!/usr/bin/env python3
# Did everything that was asked for actually run? For each stage: cells asked for, passes each, rows that came back, any lookup count short of the
# full 100M, any scan that disagreed with the sorted keys, any log with an error. Prints PASS or the exact cells that are missing.
#   smoke_report.py RESULTS_ROOT
import csv,glob,os,re,sys,collections
root=sys.argv[1]
fails=0
def line(s): print(s)
def timed(stage,cellfile,parse):
    global fails
    tp=stage+"/table.tsv"
    if not os.path.exists(cellfile):
        line(f"-- {stage}: no cells file"); fails+=1; return
    want=collections.Counter()
    for l in open(cellfile):
        k,passes=parse(l.split())
        if k: want[k]=passes
    got=collections.Counter(); short=[]
    if os.path.exists(tp):
        for r in csv.DictReader(open(tp),delimiter="\t"):
            k=(r["index"],r["dataset"],r["spk"],r["leaf"],r["tau"],r["cap"])
            got[k]+=1
            if r.get("ok") not in("100000000","100000000.0"): short.append((k,r.get("ok")))
    miss=[(k,want[k],got[k]) for k in want if got[k]<want[k]]
    st="PASS" if not miss and not short else "FAIL"
    fails+=st=="FAIL"
    line(f"{st} {os.path.relpath(stage,root)}: {len(want)} cells, {sum(want.values())} passes asked, {sum(got.values())} rows"+(f", {len(miss)} short" if miss else "")+(f", {len(short)} with lookups missing" if short else ""))
    for m in miss[:40]: line(f"   missing {m[0]}: wanted {m[1]} got {m[2]}")
    for s in short[:20]: line(f"   short {s}")
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
    bad=collections.Counter()
    for r in rows:
        d=dict(x.split("=",1) for x in r if "=" in x)
        if d.get("bad","0")!="0": bad[(r[0],r[1])]=d["bad"]
    st="PASS" if not bad else "CHECK"
    fails+=st!="PASS"
    line(f"{st} {name}: {len(cells)} cells, {len(rows)} rows; scans wrong: {dict(bad) if bad else 'none'}")
def g_parse(a):  # data idx ds b leaf tau v passes [rule]
    return ((a[1],a[2],a[3],a[4],a[5],a[6]),int(a[7]))
def f_parse(a):  # idx ds b leaf tau v passes
    return ((a[0],a[1],a[2],a[3],a[4],a[5]),int(a[6]))
for name,base,subs in (("G","jed200M_g_2026-11-03",[("real",g_parse),("hand",g_parse),("synt",g_parse)]),
                       ("final","jed200M_final_2026-11-03",[("reg",f_parse),("fin",None),("rep",f_parse)]),
                       ("baselines","jed200M_base_2026-11-03",[("point",f_parse),("readonly",f_parse),("readonly_rep",f_parse)]),
                       ("ablation","jed200M_ablation_2026-11-03",[("abl",f_parse)])):
    r=f"{root}/{base}"
    if not os.path.isdir(r): line(f"-- {name}: not run yet ({base})"); continue
    for sub,parse in subs:
        cf=f"{r}/{sub}/cells" if sub!="fin" else f"{r}/cells.txt"
        timed(f"{r}/{sub}",cf,parse or f_parse)
    for s in ("scan/out.tsv","readonly_scan/out.tsv","ablation_scan/out.tsv"):
        if os.path.exists(f"{r}/{s}"): scanfile(f"{r}/{s}",f"{name} {s}")
    ch=f"{r}/chain/out.tsv"
    if os.path.exists(ch): line(f"-- {name} chain: {sum(1 for _ in open(ch))} rows")
    if os.path.exists(f"{r}/tests.txt"): line(f"-- {name} tests: "+" | ".join(x.strip()[:60] for x in open(f"{r}/tests.txt") if ": " in x and ("ok" in x or "FAILED" in x))[:200])
e=f"{root}/jed200M_extras_2026-11-03"
if os.path.isdir(e):
    timed(f"{e}/picks",f"{e}/picks/cells",lambda a:(None,0)) if False else None
    if os.path.exists(f"{e}/gate/choices.tsv"): line(f"-- gate: {sum(1 for _ in open(e+'/gate/choices.tsv'))} choices")
    if os.path.exists(f"{e}/miss/out.tsv"):
        rows=[l.split("\t") for l in open(f"{e}/miss/out.tsv")]
        wrong=[r[:3] for r in rows if len(r)>4 and r[4].startswith("hits=") and r[4].split("=")[1]!=r[5].split("=")[1]]
        line(f"{'PASS' if not wrong else 'FAIL'} misses: {len(rows)} rows, wrong hit counts: {wrong[:10] or 'none'}")
print("SMOKE:", "PASS" if not fails else f"{fails} problems, see above")
