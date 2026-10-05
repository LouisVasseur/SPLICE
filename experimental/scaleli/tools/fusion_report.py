#!/usr/bin/env python3
"""Fusion evidence view: does transform + virtual points beat each part alone? Standard library only.

Reads one or more sweep summaries produced by tools/summarize.py (--baseline packed_rank) and
renders a single HTML page with inline SVG that tests three hypotheses per dataset:
  H1  where the flow is accepted (flow_region_fraction > 0), flow+vp has fewer fence probes than flow alone and vp alone;
  H2  where the flow is rejected, flow+vp equals vp alone (graceful bypass: no loss);
  H3  the per-region cost-based selector (fusion auto) is never worse than the best single variant
      within the seed-bootstrap interval of the paired throughput speedup.
Every verdict is computed from the numbers in the CSV; nothing is inferred. Speedups are paired on
identical traces and bootstrapped over query seeds (see summarize.py). Label every figure as a
reduced-scale, single-host control study.

usage: fusion_report.py LABEL=path/to/summary.csv [LABEL=path ...] --output fusion_report.html
"""
from __future__ import annotations
import argparse, csv, html, pathlib

CONTROL = "packed_rank"; FLOW = "packed_rank_flow"; VP = "packed_rank_vp10"; BOTH = "packed_rank_flow_vp10"; AUTO = "packed_rank_fusion_auto"
SINGLES = [FLOW, VP]
def esc(x): return html.escape(str(x))
def f(r, k, d=0.0):
    try: return float(r[k])
    except (KeyError, ValueError, TypeError): return d

def load(path: pathlib.Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    out = {}
    for r in rows: out.setdefault((r["dataset"], r["profile"]), {})[r["variant"]] = r
    return out

def bars(groups, series, value, lo=None, hi=None, unit="", title="", baseline_line=None):
    """Grouped bars: groups = dataset names, series = variant names; value(row)->float, lo/hi optional CI accessors."""
    W, H, left, bottom = 960, 260, 60, 60; n = len(groups); m = max(1, len(series)); gw = (W - left - 20) / max(1, n); bw = gw / (m + 1)
    vals = [value(r) for g in groups for r in g[1] if r is not None]; vmax = max(vals + [hi(r) for g in groups for r in g[1] if r is not None and hi]) if vals else 1
    vmax = vmax * 1.15 if vmax > 0 else 1; sy = (H - bottom - 20) / vmax
    palette = ["#9aa5ad", "#176771", "#a96420", "#4b6d9c", "#7a4b9c", "#2f8f5b", "#c2412c"]
    s = [f'<figure><figcaption>{esc(title)}</figcaption><svg viewBox="0 0 {W} {H}" role="img">']
    for t in range(5):
        y = H - bottom - (H - bottom - 20) * t / 4; v = vmax * t / 4
        s.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W-20}" y2="{y:.1f}" stroke="#e3e8eb"/><text x="{left-6}" y="{y+4:.1f}" font-size="11" text-anchor="end">{v:.3g}</text>')
    if baseline_line is not None:
        y = H - bottom - baseline_line * sy; s.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W-20}" y2="{y:.1f}" stroke="#17242f" stroke-dasharray="4 3"/>')
    for gi, (gname, rows) in enumerate(groups):
        x0 = left + gi * gw
        for si, r in enumerate(rows):
            if r is None: continue
            v = value(r); x = x0 + (si + 0.5) * bw; y = H - bottom - v * sy
            s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.9:.1f}" height="{max(0.0, v*sy):.1f}" fill="{palette[si % len(palette)]}"><title>{esc(series[si])}: {v:.4g} {esc(unit)}</title></rect>')
            if lo and hi:
                yl = H - bottom - lo(r) * sy; yh = H - bottom - hi(r) * sy; xc = x + bw * 0.45
                s.append(f'<line x1="{xc:.1f}" y1="{yl:.1f}" x2="{xc:.1f}" y2="{yh:.1f}" stroke="#17242f" stroke-width="1.2"/>')
        s.append(f'<text x="{x0+gw/2:.1f}" y="{H-bottom+16}" font-size="12" text-anchor="middle">{esc(gname)}</text>')
    for si, name in enumerate(series):
        s.append(f'<rect x="{left+si*150}" y="{H-24}" width="12" height="12" fill="{palette[si % len(palette)]}"/><text x="{left+si*150+16}" y="{H-14}" font-size="11">{esc(name)}</text>')
    return "".join(s) + "</svg></figure>"

def hardness_moves(hardness: dict, title: str) -> str:
    """Orthogonal moves in hardness space: x = PLA-32 (local, log), y = RMSE (global, log); arrows from the sample
    to sample_flow (transform), sample_csv (virtual points) and sample_flow_csv (fusion)."""
    import math
    W, H, left, bottom = 960, 420, 70, 50
    pts = []
    for d, scopes in hardness.items():
        if "sample" not in scopes: continue
        base = scopes["sample"]; pts.append((d, base, {k: scopes[k] for k in ("sample_flow", "sample_csv", "sample_flow_csv") if k in scopes}))
    if not pts: return f"<p>{esc(title)}: no sample-scope hardness available.</p>"
    xs = [m["pla_32"] for _, b, o in pts for m in [b, *o.values()] if m.get("pla_32", 0) > 0]; ys = [m["rmse"] for _, b, o in pts for m in [b, *o.values()] if m.get("rmse", 0) > 0]
    lx0, lx1 = math.log10(min(xs)) - 0.1, math.log10(max(xs)) + 0.1; ly0, ly1 = math.log10(min(ys)) - 0.1, math.log10(max(ys)) + 0.1
    X = lambda v: left + (math.log10(max(v, 1e-9)) - lx0) / (lx1 - lx0) * (W - left - 20); Y = lambda v: H - bottom - (math.log10(max(v, 1e-9)) - ly0) / (ly1 - ly0) * (H - bottom - 20)
    col = {"sample_flow": "#176771", "sample_csv": "#a96420", "sample_flow_csv": "#7a4b9c"}
    s = [f'<figure><figcaption>{esc(title)}</figcaption><svg viewBox="0 0 {W} {H}" role="img"><defs>']
    for k, c in col.items(): s.append(f'<marker id="m_{k}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{c}"/></marker>')
    s.append("</defs>")
    for e in range(int(math.floor(lx0)), int(math.ceil(lx1)) + 1):
        x = X(10 ** e); s.append(f'<line x1="{x:.1f}" y1="20" x2="{x:.1f}" y2="{H-bottom}" stroke="#e3e8eb"/><text x="{x:.1f}" y="{H-bottom+16}" font-size="11" text-anchor="middle">1e{e}</text>')
    for e in range(int(math.floor(ly0)), int(math.ceil(ly1)) + 1):
        y = Y(10 ** e); s.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W-20}" y2="{y:.1f}" stroke="#e3e8eb"/><text x="{left-6}" y="{y+4:.1f}" font-size="11" text-anchor="end">1e{e}</text>')
    s.append(f'<text x="{(W+left)/2:.0f}" y="{H-8}" font-size="12" text-anchor="middle">PLA-32 segments of the 2M-key sample (local hardness, log scale)</text>')
    s.append(f'<text transform="translate(14,{H/2:.0f}) rotate(-90)" font-size="12" text-anchor="middle">RMSE of one linear model (global hardness, log scale)</text>')
    for d, b, others in pts:
        x0, y0 = X(b["pla_32"]), Y(b["rmse"])
        for k, m in others.items():
            x1, y1 = X(m["pla_32"]), Y(m["rmse"])
            s.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{col[k]}" stroke-width="1.6" marker-end="url(#m_{k})"><title>{esc(d)} {esc(k)}: PLA-32 {b["pla_32"]:.0f}→{m["pla_32"]:.0f}, RMSE {b["rmse"]:.3g}→{m["rmse"]:.3g}</title></line>')
        s.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="4" fill="#17242f"/><text x="{x0+6:.1f}" y="{y0-6:.1f}" font-size="12">{esc(d)}</text>')
    for i, (k, c) in enumerate(col.items()):
        s.append(f'<line x1="{left+i*230}" y1="12" x2="{left+i*230+30}" y2="12" stroke="{c}" stroke-width="2"/><text x="{left+i*230+36}" y="16" font-size="11">{esc(k)}</text>')
    return "".join(s) + "</svg></figure>"

def verdicts(table: dict) -> list[dict]:
    out = []
    for (dataset, profile), v in sorted(table.items()):
        if CONTROL not in v: continue
        row = {"dataset": dataset, "profile": profile}
        def probes(name): return f(v[name], "fence_probes_per_operation") if name in v else None
        def speed(name): return (f(v[name], "paired_throughput_speedup"), f(v[name], "seed_bootstrap_low"), f(v[name], "seed_bootstrap_high")) if name in v else None
        accepted = FLOW in v and f(v[FLOW], "flow_region_fraction") > 0
        row["flow_accepted_fraction"] = f(v[FLOW], "flow_region_fraction") if FLOW in v else float("nan")
        pf, pv, pb, pc = probes(FLOW), probes(VP), probes(BOTH), probes(CONTROL)
        if accepted and None not in (pf, pv, pb): row["H1"] = "supported" if pb < min(pf, pv) else "not supported"; row["H1_detail"] = f"probes/op: control {pc:.2f}, flow {pf:.2f}, vp {pv:.2f}, both {pb:.2f}"
        elif not accepted and None not in (pv, pb): row["H2"] = "supported" if abs(pb - pv) <= 0.02 * max(pv, 1e-9) else "not supported"; row["H2_detail"] = f"flow rejected in every region; probes/op vp {pv:.2f} vs both {pb:.2f}"
        if AUTO in v:
            sa = speed(AUTO); best = max(((speed(n), n) for n in SINGLES + [BOTH] if n in v), key=lambda t: t[0][0], default=None)
            if sa and best:
                sb, bn = best; row["H3"] = "supported" if sa[2] >= sb[1] else "not supported"  # auto's upper CI reaches best single's lower CI
                row["H3_detail"] = f"auto {sa[0]:.3f} [{sa[1]:.2f},{sa[2]:.2f}] vs best single {bn} {sb[0]:.3f} [{sb[1]:.2f},{sb[2]:.2f}]"
                row["auto_choices"] = f"flow {f(v[AUTO],'fusion_flow_fraction'):.0%} · vp {f(v[AUTO],'fusion_vp_fraction'):.0%} · both {f(v[AUTO],'fusion_both_fraction'):.0%}"
                row["auto_est_probes"] = f"{f(v[AUTO],'est_probes_none'):.2f} → {f(v[AUTO],'est_probes_selected'):.2f}"
        out.append(row)
    return out

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="LABEL=path/to/summary.csv"); p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--hardness", action="append", default=[], help="LABEL=path/to/hardness.json (pipeline output) for the hardness-space panel")
    a = p.parse_args(); parts = []
    import json
    for spec in a.hardness:
        label, _, path = spec.partition("=")
        try: parts.append("<section>" + hardness_moves(json.loads(pathlib.Path(path).read_text()), f"{label}: where the transform moves a dataset (global RMSE) vs where virtual points move it (local PLA-32)") + "</section>")
        except (OSError, ValueError) as e: parts.append(f"<section><p>{esc(label)}: hardness panel unavailable ({esc(e)})</p></section>")
    for spec in a.inputs:
        label, _, path = spec.partition("="); table = load(pathlib.Path(path)); vs = verdicts(table)
        variants = [CONTROL, FLOW, "packed_rank_flow_forced", VP, BOTH, AUTO, "packed_rank_flow_costsel", "packed_rank_root", "packed_rank_vp10_root", "packed_rank_fusion_auto_root", "raw_rank", "sorted_vector"]
        present = [x for x in variants if any(x in v for v in table.values())]
        groups = [(d, [v.get(x) for x in present]) for (d, prof), v in sorted(table.items()) if prof == "read_only"] or [(d, [v.get(x) for x in present]) for (d, prof), v in sorted(table.items())]
        s = f'<section><h2>{esc(label)}</h2><p class="small">{esc(path)}</p>'
        s += bars(groups, present, lambda r: f(r, "paired_throughput_speedup"), lambda r: f(r, "seed_bootstrap_low"), lambda r: f(r, "seed_bootstrap_high"), "×", "Paired throughput speedup vs packed_rank (whiskers: seed bootstrap 95%)", baseline_line=1.0)
        s += bars(groups, present, lambda r: f(r, "fence_probes_per_operation"), None, None, "probes/op", "Exact fence probes per operation (lower = better prediction; identical traces)")
        s += bars(groups, present, lambda r: f(r, "read_hit_p99_ns_median"), None, None, "ns", "Read-hit P99 latency (median over seeds; 0 when the latency pass was skipped)")
        s += '<table><thead><tr><th>dataset</th><th>profile</th><th>flow accepted (regions)</th><th>H1 fusion &lt; each alone</th><th>H2 graceful bypass</th><th>H3 auto ≥ best single</th><th>auto choices</th><th>auto est. probes none→selected</th></tr></thead><tbody>'
        for r in vs:
            s += "<tr>" + "".join(f"<td>{esc(r.get(k, ''))}{('<br><span class=small>' + esc(r.get(k + '_detail', '')) + '</span>') if r.get(k + '_detail') else ''}</td>" for k in ("dataset", "profile", "flow_accepted_fraction", "H1", "H2", "H3", "auto_choices", "auto_est_probes")) + "</tr>"
        s += "</tbody></table>"
        cols = ["variant", "runs", "throughput_ops_s_median", "paired_throughput_speedup", "seed_bootstrap_low", "seed_bootstrap_high", "fence_probes_per_operation", "correction_distance_per_operation", "rank_sse_ratio", "virtual_points_per_key", "flow_region_fraction", "fusion_flow_fraction", "fusion_vp_fraction", "fusion_both_fraction", "est_probes_none", "est_probes_selected", "preprocess_ns_per_key", "initial_bytes_per_key"]
        for (d, prof), v in sorted(table.items()):
            s += f"<details><summary>{esc(d)} · {esc(prof)}: full summary rows</summary><table><thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr></thead><tbody>"
            for name in present:
                if name in v: s += "<tr>" + "".join(f"<td>{esc(v[name].get(c, ''))[:10]}</td>" for c in cols) + "</tr>"
            s += "</tbody></table></details>"
        parts.append(s + "</section>")
    css = "body{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#17242f}section{border:1px solid #dde4e8;border-radius:8px;padding:22px;margin:22px 0;background:#fff}figure{margin:14px 0}figcaption{font-weight:600;margin-bottom:4px}svg{width:100%}table{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0}th,td{padding:6px;border-bottom:1px solid #dbe1e5;text-align:left;vertical-align:top}th{background:#f1f5f6}.small{font-size:11px;color:#5a6670}.warn{border-left:3px solid #a96420;padding-left:14px}"
    head = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Fusion evidence</title><style>' + css + '</style><h1>Fusion evidence: NFL-style transform + CSV-style virtual points</h1>'
            '<p class="warn">Reduced-scale, single-host control study inside the SCALE-LI experimental map. The transform is a clean-room 8-parameter monotone tanh network trained here; virtual points follow CSV Algorithm 1 per region; "fusion auto" picks per region among none / flow / virtual points / both by expected probes on its own keys. Speedups are paired on identical query traces and bootstrapped over query seeds. None of this is a reproduction of NFL, AFLI or CSV.</p>'
            '<p>H1: where the flow is accepted, flow+vp has fewer fence probes than flow alone and vp alone. H2: where the flow is rejected everywhere, flow+vp equals vp alone. H3: the auto selector is at least as fast as the best single variant within the bootstrap interval.</p>')
    a.output.write_text(head + "".join(parts) + "</html>"); print(a.output)
if __name__ == "__main__": main()
