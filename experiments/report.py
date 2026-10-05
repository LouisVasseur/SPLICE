"""Offline HTML results: standard library only, no CDN, plotting install, or server."""
from __future__ import annotations
import html,json
from pathlib import Path
from .lessons import TITLES

def esc(x):return html.escape(str(x))
def table(rows,fields):
 return '<table><thead><tr>'+''.join('<th>'+esc(label)+'</th>' for key,label in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(f'{row.get(key):,.4g}' if isinstance(row.get(key),float) else row.get(key,''))+'</td>' for key,label in fields)+'</tr>' for row in rows)+'</tbody></table>'
def bars(labels,values,title,unit):
 mx=max(values) if max(values,default=0)>0 else 1
 parts=[f'<svg viewBox="0 0 900 {len(values)*58+65}" role="img"><title>{esc(title)}</title>']
 for i,(label,v) in enumerate(zip(labels,values)):
  y=35+i*58;w=480*v/mx
  parts += [f'<text x="8" y="{y+19}" font-size="17">{esc(label)}</text>',f'<rect x="220" y="{y}" width="{w}" height="30" rx="2" fill="currentColor"/>',f'<text x="{230+w}" y="{y+21}" font-size="16">{v:,.4g} {esc(unit)}</text>']
 return ''.join(parts)+'</svg>'
def details(r):
 id=r['lesson'];s=''
 if id=='01':
  s=bars(['Single affine','Two-stage toy'],[r['single_model_mae_rank'],r['two_stage_mae_rank']],'Stored-key mean absolute rank error','ranks')
  s+=f'<p>{r["exact_query_checks"]:,} exact correction checks passed. This is a geometric teaching experiment, not a timing benchmark.</p>'
 elif id=='02':
  s=bars(['Binary lower_bound','RadixSpline-derived'],[r['binary_ns_per_query_median'],r['radix_ns_per_query_median']],'Median of five warmed trace replays','ns/query')
  s+=f'<p>Keys: {r["keys"]:,}; queries: {r["queries"]:,}; actually absent queries: {r["absent_queries"]:,}. Key array: {r["key_bytes"]:,} bytes; additional spline+radix metadata: {r["auxiliary_index_bytes"]:,} bytes. Maximum correction window: {r["max_search_window_keys"]} keys. No payloads or updates.</p>'
 elif id=='03':
  s=table(r['rows'],[('dataset','Paper dataset'),('without_NF_Mops_s','ALEX without NF, Mops/s'),('with_NF_Mops_s','With NF, Mops/s'),('ratio','Ratio')])
  s+=table(r['batch_transform'],[('batch_size','Batch size'),('two_hidden_two_layer_ns_per_key','Transform-only ns/key')])
  s+='<p>Source: NFL Table 1 (p.3), Table 2 (p.6). Regenerating this view recalculates reported ratios; it does not execute or train NFL.</p>'
 elif id=='04':
  s=table(r['summary'],[('variant','Matched variant'),('bytes_per_record','Accounted B/record'),('throughput_ops_s','Operations/s'),('read_p99_ns','Hit P99 ns')])
  s+='<p>Read-only, all 16,384 fixture records loaded; 10,000 operations per seed; three seeds. These are custom-map ablations, not ALEX/PGM/LeCo results.</p>'
 elif id=='05':
  s=bars(['Before','Greedy toy','Exhaustive toy'],[r['before_sse'],r['greedy']['sse_all'],r['exhaustive']['sse_all']],'Augmented-set SSE in the independent teaching toy','SSE')
  s+=f'<p>Original keys: {esc(r["keys"])}. Budget = {r["budget"]}. Greedy inserts {esc(r["greedy"]["virtual_points"])}; exhaustive search checks {r["exhaustive"]["subsets_examined"]:,} subsets. The source paper uses different keys/budget for its figure.</p>'
  s+='<p>Separately, CSV reports up to 34% less query time on <strong>promoted keys</strong>. It repeats each queried key 100 times. See paper p.9–10, not the toy timing.</p>'
 elif id=='06':
  s=table(r['geometry'],[('policy','Layout'),('bytes_per_key','Encoded key B/key'),('byte_normalized_mse','Normalized byte MSE'),('byte_mean_block_error','Initial block error')])
  s+=table(r['update_controls'],[('policy','Update control'),('insert_p50_ns','Insert median ns'),('insert_p99_ns','Insert P99 ns'),('compactions','Compactions'),('drain_ns','Drain ns')])
  s+='<p>Verified invariant: logical rank model unchanged across layouts. Geometry and update tables are different workloads.</p>'
 elif id=='08':
  sm=r['summary'];names=[x['variant'] for x in sm]
  s=bars(names,[x['fence_probes_per_op'] for x in sm],'Exact fence probes per operation (lower is better; identical traces)','probes/op')
  s+=bars(names,[x['throughput_ops_s']/1e6 for x in sm],'Median throughput over three seeds','Mops/s')
  s+=table(sm,[('variant','Variant'),('flow_regions','Regions using flow'),('virtual_points','Virtual points'),('rank_sse_ratio','Rank SSE after/before'),('preprocess_ms','Preprocess ms'),('read_p99_ns','Hit P99 ns'),('bytes_per_record','Accounted B/record')])
  t=r['flow_training'];s+=f'<p>Flow trained in this run: {t["training_keys"]:,} sampled keys, {t["steps"]} steps, {t["train_seconds"]:.1f} s; global tail conflict degree {t["tail_conflict_degree_raw"]} → {t["tail_conflict_degree_transformed"]}; unordered transformed pairs: {t["unordered_transformed_pairs"]}. Per-region auto-switch decides whether the flow is used.</p>'
 elif id=='09':
  if r['status'].startswith('NOT RUN'):
   s=f'<p>Not run yet. Generate this lane with <code>{esc(r["status"][8:].strip())}</code> once the ten GRE downloads are complete (<code>--dry-run</code> exercises it on the bundled fixture without network). Nothing in it reproduces the AIDB 2026 paper, NFL/AFLI or CSV.</p>'
  else:
   tp=r['throughput'];s=bars([x['dataset'] for x in tp],[x['mops'] for x in tp],f'{r["variant"]}: median lookup throughput per dataset ({r["scale"]})','Mops/s')
   s+=table(r['metrics'],[('metric','Hardness metric'),('coverage','Cov'),('conformance','Conf (mean over our variants)'),('conformance_variant','Conf for '+r['variant']),('comparable_pairs','Comparable pairs'),('incomparable_pairs','Incomparable pairs')])
   link=f'<a href="{esc(r["report_link"])}">full AIDB lane report (hardness space before/after the transform and after smoothing, all 25 metrics, provenance)</a>' if r.get('report_link') else 'run the report step of the pipeline for the full HTML view'
   s+=f'<p>Datasets: {esc(", ".join(r["datasets"]))}. Best/worst {esc(r["variant"])} throughput: {esc(r["best"]["dataset"])} {r["best"]["mops"]:.2f} / {esc(r["worst"]["dataset"])} {r["worst"]["mops"]:.2f} Mops/s. Hardness scope <code>{esc(r["scope"])}</code>; variants {esc(", ".join(r["variants"]))}. See the {link}.</p>'
 else:s='<pre>'+esc(r['audit_output'])+'</pre>'
 return s

def build(root:Path,out:Path):
 sections=[]
 for id,title in TITLES.items():
  p=out/id/'result.json'
  if not p.exists():continue
  r=json.loads(p.read_text());notice=r.get('warning',r.get('caution',r.get('interpretation','')))
  sections.append(f'<section id="L{id}"><p class="eyebrow">L{id} · {esc(r["status"])}</p><h2>{esc(title)}</h2>{details(r)}<p class="notice">{esc(notice)}</p><p>Repeat: <code>python3 walkthrough.py demo {id}</code> · <a href="{id}/result.json">Raw result and environment</a></p><details><summary>Inspect all fields</summary><pre>{esc(json.dumps(r,indent=2))}</pre></details></section>')
 mode='LOCAL RUN' if out.name!='saved' else 'SAVED REHEARSAL OUTPUT'
 text='''<!doctype html><html lang="en"><meta charset="utf-8"><title>SCALE-LI · numbered results</title><style>
:root{font-family:system-ui,sans-serif;color:#17242f;background:#f6f8fa}body{max-width:1080px;margin:40px auto;padding:0 24px}h1{font-size:38px}h2{font-size:27px}p{line-height:1.55}nav{position:sticky;top:0;background:#fff;padding:16px;border-bottom:1px solid #ccd4da}nav a{margin-right:20px;color:#145a61}section{background:white;margin:25px 0;padding:28px;border:1px solid #dde4e8;border-radius:8px;scroll-margin-top:70px}.eyebrow{font-size:12px;letter-spacing:.08em;font-weight:700;color:#145a61}.notice{border-left:3px solid #a96420;padding-left:15px}table{border-collapse:collapse;width:100%;font-size:15px;margin:20px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #dbe1e5}th{background:#f1f5f6}pre{overflow:auto;max-height:480px;font-size:12px;line-height:1.5;padding:15px;background:#f2f5f7}code{font-size:14px}svg{width:100%;color:#176771}a{color:#176771}footer{font-size:13px;margin:30px 0}@media print{nav,details{display:none}section{break-inside:avoid}body{max-width:none}}</style>'''
 text+=f'<h1>SCALE-LI · results you can step through</h1><p class="eyebrow">{mode}</p><p>Numbers are separated into published evidence, teaching demonstrations, and native prototype measurements. A successful arithmetic check is not a paper reproduction.</p><nav>'+''.join(f'<a href="#L{x}">L{x}</a>' for x in TITLES)+'</nav>'
 text+=''.join(sections)+'<footer>Meeting guide: ../../docs/REHEARSAL.md · Scope and sources: ../../docs/APPROACHES.md · Timing is machine-dependent.</footer></html>'
 (out/'report.html').write_text(text);return out/'report.html'
