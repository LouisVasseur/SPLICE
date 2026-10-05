#!/usr/bin/env python3
"""Offline HTML view of a sweep summary (tools/summarize.py output). Standard library only.

One section per dataset/profile pair with bar charts for throughput, exact fence
probes per operation, rank-SSE ratio and virtual points per key, plus the full
summary table. No plotting package is needed; the SVG is generated inline.
"""
from __future__ import annotations
import argparse, csv, html, json, pathlib

def esc(x): return html.escape(str(x))
def bars(labels, values, title, unit, digits=3):
    mx=max(values) if max(values, default=0)>0 else 1
    out=[f'<figure><figcaption>{esc(title)}</figcaption><svg viewBox="0 0 900 {len(values)*30+10}" role="img">']
    for i,(label,v) in enumerate(zip(labels,values)):
        y=5+i*30; w=520*v/mx
        out+=[f'<text x="4" y="{y+17}" font-size="14">{esc(label)}</text>',f'<rect x="250" y="{y}" width="{w:.1f}" height="22" fill="currentColor" opacity=".85"/>',
              f'<text x="{256+w:.1f}" y="{y+16}" font-size="13">{v:.{digits}g} {esc(unit)}</text>']
    return ''.join(out)+'</svg></figure>'

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('summary', type=pathlib.Path); p.add_argument('--output', type=pathlib.Path)
    a=p.parse_args(); rows=list(csv.DictReader(a.summary.open())); out=a.output or a.summary.with_name('sweep_report.html')
    env=a.summary.with_name('environment.json'); envtxt=json.loads(env.read_text()) if env.exists() else {}
    groups={}
    for r in rows: groups.setdefault((r['dataset'],r['profile']),[]).append(r)
    fields=['variant','runs','all_verified','throughput_ops_s_median','paired_throughput_speedup','seed_bootstrap_low','seed_bootstrap_high','read_hit_p99_ns_median',
            'fence_probes_per_operation','correction_distance_per_operation','rank_sse_ratio','virtual_points_per_key','flow_region_fraction','preprocess_ns_per_key','initial_bytes_per_key','build_ns_per_key']
    sections=[]
    for (dataset,profile),g in sorted(groups.items()):
        names=[r['variant'] for r in g]; f=lambda k:[float(r[k]) for r in g]
        s=f'<section><h2>{esc(dataset)} · {esc(profile)}</h2>'
        s+=bars(names,[v/1e6 for v in f('throughput_ops_s_median')],'Median throughput','Mops/s')
        s+=bars(names,f('fence_probes_per_operation'),'Exact fence probes per operation (lower = better prediction)','probes/op')
        s+=bars(names,f('rank_sse_ratio'),'Rank-model SSE after / before smoothing (1 = unchanged)','')
        s+=bars(names,f('virtual_points_per_key'),'Virtual points per key (space proxy; costs metadata only)','')
        s+=bars(names,f('flow_region_fraction'),'Fraction of regions where the auto-switch kept the transform','')
        s+='<table><thead><tr>'+''.join(f'<th>{esc(k)}</th>' for k in fields)+'</tr></thead><tbody>'
        for r in g: s+='<tr>'+''.join('<td>'+esc(f'{float(r[k]):.4g}' if r[k].replace('.','',1).replace('-','',1).replace('e','',1).replace('+','',1).isdigit() else r[k])+'</td>' for k in fields)+'</tr>'
        sections.append(s+'</tbody></table></section>')
    text='<!doctype html><html lang="en"><meta charset="utf-8"><title>SCALE-LI sweep · learnability</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#17242f}section{border:1px solid #dde4e8;border-radius:8px;padding:22px;margin:22px 0;background:#fff}figure{margin:14px 0}figcaption{font-weight:600;margin-bottom:4px}svg{width:100%;color:#176771}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:6px;border-bottom:1px solid #dbe1e5;text-align:left}th{background:#f1f5f6}.warn{border-left:3px solid #a96420;padding-left:14px}</style>'
    text+=f'<h1>Sweep summary: {esc(a.summary)}</h1><p class="warn">{esc(envtxt.get("warning",""))} Baseline for paired speedups: {esc(rows[0]["baseline"] if rows else "")}. Flow and virtual-point variants are clean-room NFL/CSV-style controls on the SCALE-LI map, not the original systems.</p>'
    text+=''.join(sections)+f'<p><small>{esc(envtxt.get("platform",""))} · {esc(envtxt.get("timestamp_utc",""))}</small></p></html>'
    out.write_text(text); print(out)
if __name__=='__main__': main()
