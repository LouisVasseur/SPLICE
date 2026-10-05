#!/usr/bin/env python3
"""Self-contained HTML report for the AIDB-style dataset-hardness protocol applied to the
SCALE-LI NFL-style transform and CSV-style virtual-point controls. Standard library only;
every figure is inline SVG and the page loads no external resource.

Inputs, all read from --results and all optional (a missing or unreadable file yields a
"not available" notice in its section, so the page always renders):
  hardness.json      {"<dataset>": {"<scope>": {"rmse","max_error","conflict_degree","pla_32","pla_4096"}}}
  throughput.json    {"<variant>": {"<dataset>": median MOPS}}
  scores.json        {"<scope>": {"metrics": {"<metric>": {"dims","coverage","conformance","per_variant",
                      "comparable_pairs","incomparable_pairs"}}, "datasets": [...], "variants": [...]}}
  provenance.json    {"<dataset>": {"url","sha256","bytes","retrieved_utc","sample": {...}, "sort_audit": {...}}}
  sweep/summary.csv  tools/summarize.py output;   flows/training_report.json  tools/prepare_flows.py style
  hardness/*.json    raw scaleli_hardness output (transform/smoothing cost);   pipeline.log (tail shown)
  hardness_details.json  per-scope degeneracy fields (duplicates, unordered pairs, FMCD epsilon) when hardness.json lacks them

Every number is a reduced-scale, single-host, clean-room control on the SCALE-LI map. Nothing here
reproduces Zhang, Tang and Ailamaki (AIDB@VLDB 2026), NFL/AFLI or CSV.
"""
from __future__ import annotations
import argparse, csv, datetime, html, itertools, json, math, pathlib

PAPER_ORDER = ['books', 'fb', 'osm', 'covid', 'genome', 'history', 'libio', 'planet', 'stack', 'wise']
SCALARS = [('rmse', 'RMSE'), ('max_error', 'ME'), ('conflict_degree', 'CD'), ('pla_32', 'PLA-32'), ('pla_4096', 'PLA-4096')]
METRICS = ['·'.join(c) for k in (1, 2, 3) for c in itertools.combinations([n for _, n in SCALARS], k)]  # 25 names, Table 2 order
SCOPE_LABEL = {'full': 'full keys, raw', 'full_flow': 'full keys + NFL-style flow', 'sample': 'sample, raw', 'sample_flow': 'sample + NFL-style flow',
               'sample_csv': 'sample + CSV-style virtual points', 'sample_flow_csv': 'sample + flow + virtual points'}
HEADINGS = ['1. Protocol and dataset provenance', '2. Hardness space before and after the controls', '3. Throughput per dataset',
            '4. Conformance and coverage of the 25 metrics', '5. Hardness shift per dataset', '6. Preprocessing and cost', '7. Reproduce']
PALETTE = ['#176771', '#a96420', '#4b6ea8', '#8c3f5d', '#5b8c3a', '#b0862b', '#606a72', '#2f9f8f', '#c05a3c', '#7a5cad']
FLOW, CSV, INK, GRID = '#176771', '#a96420', '#17242f', '#e3e8eb'
PROTOCOL = ('Reduced-scale, single-host, clean-room controls. The five scalar hardness metrics (least-squares RMSE and ME, conflict degree of the '
            'FMCD fit, optimal PLA segment counts at eps 32 and 4096) are computed on the full key sets; the index runs use 2M-key uniform samples '
            'with uniform lookups (1M measured after 200k warm-up per seed by default, median of three seeds) on the SCALE-LI map and its NFL-style '
            'transform and CSV-style virtual-point variants. Nothing here reproduces the AIDB paper, NFL/AFLI or CSV, and the paper\'s indexes '
            '(RMI, PGM, ALEX, LIPP, XIndex, FINEdex) are not run. Timing is machine-dependent. Seven GRE files (covid, genome, history, libio, '
            'planet, stack, wise) are served unsorted; hardness, samples and index runs use a sorted, de-duplicated copy whose sha256 is in the '
            'provenance table.')
CD_NOTE = ('CD caveats: on transformed and smoothed features the FMCD fit adds 1e-6 x mean gap to U_T instead of LIPP\'s absolute 1e-6 (which would '
           'be a large fraction of U_T on O(1)-range flow outputs), so those CD values are not what a literal LIPP fit gives; the literal value is kept '
           'in hardness_details.json as fmcd.conflict_degree_lipp_epsilon. Raw-key CD uses LIPP\'s constant verbatim. In every scope CD is floor() of a '
           'double product, as in LIPP, so it carries +-1 rounding uncertainty: treat CD differences of one as ties when ranking datasets.')
_ids = itertools.count()

def esc(x): return html.escape(str(x))
def load(path, kind='json'):
    """Parsed file content, or None when the file is missing or unreadable (never fatal)."""
    try:
        if kind == 'json': return json.loads(path.read_text())
        if kind == 'csv': return list(csv.DictReader(path.open(newline='')))
        return path.read_text()
    except (OSError, ValueError): return None
def dig(obj, *keys):
    for k in keys: obj = obj.get(k) if isinstance(obj, dict) else None
    return obj
def num(x):
    try: v = float(x)
    except (TypeError, ValueError): return None
    return v if math.isfinite(v) else None
def fmt(x, digits=4):
    if x is None or x == '': return 'n/a'
    if isinstance(x, bool): return str(x)
    v = num(x)
    if v is None: return 'n/a' if str(x).lower() in ('nan', 'inf', '-inf') else esc(x)
    return f'{int(v):,}' if v == int(v) and abs(v) < 1e15 else f'{v:,.{digits}g}'
def pct(before, after):
    b, a = num(before), num(after)
    return 'n/a' if b is None or a is None or b == 0 else f'{(a - b) / abs(b) * 100:+.1f}%'
def notice(text): return f'<p class="notice">{esc(text)}</p>'
def unavailable(what): return notice(f'Not available: {what}.')
def table(headers, rows):
    return ('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows) + '</tbody></table></div>')
def shade(v):
    x = num(v)
    return fmt(v) if x is None else f'<span class="shade" style="background:rgba(23,103,113,{max(0.0, min(1.0, x)) * .4:.2f})">{x:.2f}</span>'
def order(names):
    known = [d for d in PAPER_ORDER if d in names]
    return known + sorted(n for n in names if n not in known)
def base_name(name, known):
    """Map a sweep dataset name such as books_2M_uniform_s42 back to the paper dataset it samples."""
    return name if name in known else next((k for k in sorted(known, key=len, reverse=True) if name.startswith(k)), name)

def axis(values, p0, p1, fixed=None):
    """(value -> pixel, [(tick, label)], scale name). log10 when positive values span more than two decades."""
    vals = [v for v in map(num, values) if v is not None]
    lo, hi = fixed if fixed else ((min(vals), max(vals)) if vals else (0.0, 1.0))
    if not fixed and lo > 0 and hi / lo > 100:
        k0 = math.floor(math.log10(lo)); k1 = max(math.ceil(math.log10(hi)), k0 + 1)
        ticks = [(10.0 ** k, f'{10 ** k:,}' if 0 <= k <= 9 else f'1e{k}') for k in range(k0, k1 + 1)]
        return (lambda v: p0 + (p1 - p0) * (math.log10(max(v, 1e-300)) - k0) / (k1 - k0)), ticks, 'log10 scale'
    if hi == lo: lo, hi = lo - (abs(lo) or 1), hi + (abs(hi) or 1)
    raw = (hi - lo) / 5; mag = 10.0 ** math.floor(math.log10(raw)); step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    t0 = math.floor(lo / step) * step; n = int(round((math.ceil(hi / step) * step - t0) / step))
    ticks = [(round(t0 + i * step, 10), f'{round(t0 + i * step, 10):,.6g}') for i in range(n + 1)]
    return (lambda v: p0 + (p1 - p0) * (v - t0) / (n * step)), ticks, 'linear scale'

def place(labels, W, H):
    """Greedy label placement: [(px, py, text, size)] -> [(x, y, anchor)], avoiding earlier labels and the canvas edge."""
    boxes, out = [], []
    for px, py, text, size in labels:
        w, h = .58 * size * len(text), size
        for dx, dy, anchor in ((6, -5, 'start'), (-6, -5, 'end'), (6, size + 4, 'start'), (-6, size + 4, 'end'), (6, -5 - size, 'start'), (-6, -5 - size, 'end'), (6, 2 * size + 6, 'start'), (-6, 2 * size + 6, 'end')):
            x, y = px + dx, py + dy; x0 = x - w if anchor == 'end' else x; box = (x0, y - h, x0 + w, y)
            if box[0] >= 0 and box[2] <= W and box[1] >= 0 and box[3] <= H and not any(b[0] < box[2] and box[0] < b[2] and b[1] < box[3] and box[1] < b[3] for b in boxes): break
        boxes.append(box); out.append((x, y, anchor))
    return out

def scatter(title, xlab, ylab, series, arrows=(), fixed=None, frontier=False, size=11, narrow=False):
    """series: [(name, color, [(x, y, label)])]; arrows: [(name, color, x0, y0, x1, y1)] drawn base -> target."""
    W, H, L, R, T, B = 640, 430, 80, 52, 44, 64
    pts = [(p[0], p[1]) for _, _, ps in series for p in ps]
    xs = [p[0] for p in pts] + [a[2] for a in arrows] + [a[4] for a in arrows]; ys = [p[1] for p in pts] + [a[3] for a in arrows] + [a[5] for a in arrows]
    fx, xt, xscale = axis(xs, L, W - R, fixed); fy, yt, yscale = axis(ys, H - B, T, fixed); uid = next(_ids)
    marker = {c: i for i, c in enumerate(dict.fromkeys(a[1] for a in arrows))}
    out = [f'<figure class="{"narrow" if narrow else ""}"><figcaption>{esc(title)}</figcaption><svg viewBox="0 0 {W} {H}" role="img" font-size="12"><title>{esc(title)}</title><defs>'
           + ''.join(f'<marker id="m{uid}-{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="{c}"/></marker>' for c, i in marker.items()) + '</defs>']
    out += [f'<line x1="{fx(v):.1f}" y1="{T}" x2="{fx(v):.1f}" y2="{H - B}" stroke="{GRID}"/><text x="{fx(v):.1f}" y="{H - B + 16}" text-anchor="middle">{esc(l)}</text>' for v, l in xt]
    out += [f'<line x1="{L}" y1="{fy(v):.1f}" x2="{W - R}" y2="{fy(v):.1f}" stroke="{GRID}"/><text x="{L - 6}" y="{fy(v) + 4:.1f}" text-anchor="end">{esc(l)}</text>' for v, l in yt]
    out.append(f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" fill="none" stroke="#9aa5ad"/>'
               f'<text x="{(L + W - R) / 2}" y="{H - 28}" text-anchor="middle" font-weight="600">{esc(xlab)} ({xscale})</text>'
               f'<text transform="translate(16 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle" font-weight="600">{esc(ylab)} ({yscale})</text>')
    for name, color, x0, y0, x1, y1 in arrows:
        ax, ay, bx, by = fx(x0), fy(y0), fx(x1), fy(y1); d = math.hypot(bx - ax, by - ay) or 1.0
        out.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx - 5 * (bx - ax) / d:.1f}" y2="{by - 5 * (by - ay) / d:.1f}" stroke="{color}" stroke-width="1.6" marker-end="url(#m{uid}-{marker[color]})"><title>{esc(name)}</title></line>')
    dots = [(fx(p[0]), fy(p[1]), p[2], name, color, p[0], p[1], p[3] if len(p) > 3 else p[2]) for name, color, ps in series for p in ps]
    for (px, py, label, name, color, x, y, tip), (tx, ty, anchor) in zip(dots, place([(px, py, label, size) for px, py, label, *_ in dots], W, H)):
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{color}"><title>{esc(tip)} \u00b7 {esc(name)}: {fmt(x)}, {fmt(y)}</title></circle>'
                   f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="{size}" text-anchor="{anchor}">{esc(label)}</text>')
    if frontier:  # non-dominated points (maximise both axes), Fig. 4 style
        best, front = -math.inf, []
        for x, y in sorted(pts, key=lambda p: (-p[0], -p[1])):
            if y > best: best = y; front.append((x, y))
        out.append('<polyline fill="none" stroke="#8a969e" stroke-dasharray="5 4" points="' + ' '.join(f'{fx(x):.1f},{fy(y):.1f}' for x, y in front) + '"><title>approximate Pareto frontier</title></polyline>')
    x = L
    for n, c, kind in [(n, c, 'dot') for n, c, _ in series] + [(n, c, 'arrow') for n, c in dict.fromkeys((a[0], a[1]) for a in arrows)]:
        out.append((f'<circle cx="{x + 5}" cy="{T - 14}" r="4.5" fill="{c}"/>' if kind == 'dot' else
                    f'<line x1="{x}" y1="{T - 14}" x2="{x + 12}" y2="{T - 14}" stroke="{c}" stroke-width="1.6" marker-end="url(#m{uid}-{marker[c]})"/>')
                   + f'<text x="{x + 17}" y="{T - 10}">{esc(n)}</text>'); x += 26 + 6.5 * len(n)
    return ''.join(out) + '</svg></figure>'

def grouped_bars(title, ylab, groups, series, value, ref=None):
    """groups: [name]; series: [(name, color)]; value(group, series) -> (v, lo, hi) or None; lo/hi draw whiskers when numeric."""
    W, H, L, R, T, B = 960, 420, 70, 20, 72, 60
    cells = {(g, s): value(g, s) for g in groups for s, _ in series}
    ys = [0.0] + [x for c in cells.values() if c for x in c if num(x) is not None] + ([ref] if ref is not None else [])
    fy, yt, yscale = axis(ys, H - B, T); gw = (W - L - R) / max(1, len(groups)); bw = gw * .8 / max(1, len(series))
    out = [f'<figure><figcaption>{esc(title)}</figcaption><svg viewBox="0 0 {W} {H}" role="img" font-size="12"><title>{esc(title)}</title>']
    out += [f'<line x1="{L}" y1="{fy(v):.1f}" x2="{W - R}" y2="{fy(v):.1f}" stroke="{GRID}"/><text x="{L - 6}" y="{fy(v) + 4:.1f}" text-anchor="end">{esc(l)}</text>' for v, l in yt]
    out.append(f'<text transform="translate(16 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle" font-weight="600">{esc(ylab)} ({yscale})</text>')
    if ref is not None: out.append(f'<line x1="{L}" y1="{fy(ref):.1f}" x2="{W - R}" y2="{fy(ref):.1f}" stroke="{CSV}" stroke-dasharray="6 4"/>')
    for gi, g in enumerate(groups):
        out.append(f'<text x="{L + (gi + .5) * gw:.1f}" y="{H - B + 18}" text-anchor="middle">{esc(g)}</text>')
        for si, (s, color) in enumerate(series):
            c = cells[g, s]
            if not c or num(c[0]) is None: continue
            v, lo, hi = num(c[0]), num(c[1]), num(c[2]); x = L + gi * gw + gw * .1 + si * bw; y0, y1 = fy(0), fy(v); cx = x + (bw - 1) / 2
            out.append(f'<rect x="{x:.1f}" y="{min(y0, y1):.1f}" width="{bw - 1:.1f}" height="{abs(y0 - y1):.1f}" fill="{color}"><title>{esc(g)} · {esc(s)}: {fmt(v)}</title></rect>')
            if lo is not None and hi is not None:
                out.append(f'<path d="M{cx:.1f},{fy(lo):.1f}V{fy(hi):.1f}M{cx - 3:.1f},{fy(lo):.1f}h6M{cx - 3:.1f},{fy(hi):.1f}h6" stroke="{INK}" fill="none"><title>{esc(g)} · {esc(s)}: [{fmt(lo)}, {fmt(hi)}]</title></path>')
    lx, ly = L, T - 44
    for s, color in series:
        if lx + 26 + 6.5 * len(s) > W - R: lx, ly = L, ly + 16
        out.append(f'<rect x="{lx}" y="{ly}" width="11" height="11" fill="{color}"/><text x="{lx + 15}" y="{ly + 10}">{esc(s)}</text>'); lx += 26 + 6.5 * len(s)
    return ''.join(out) + f'<line x1="{L}" y1="{fy(0):.1f}" x2="{W - R}" y2="{fy(0):.1f}" stroke="#9aa5ad"/></svg></figure>'

def sec_provenance(prov):
    if not prov: return unavailable('provenance.json (download verification: url, sha256, bytes, retrieval time, sample manifest)')
    rows, size, mode = [], ('sample_count', 'keys', 'count', 'size', 'n'), ('mode', 'method')
    hidden = size + mode + ('seed', 'source', 'source_sha256', 'sample_sha256', 'path', 'dtype', 'warning')  # shown in their own columns or noise
    for d in order(prov):
        p = dig(prov, d) or {}; s = dig(p, 'sample') or {}
        a = dig(p, 'sort_audit') or {}; on_disk = 'n/a' if a.get('sorted') is None else ('yes' if a.get('sorted') else 'NO')
        copy = ((esc(str(a['sha256'])[:12]) if a.get('sha256') else 'sha256 pending') + f' ({fmt(a.get("written_keys"))} keys)') if a.get('written') else ('none needed' if a.get('sorted') else 'n/a')
        rows.append([esc(d), f'<a href="{esc(p.get("url", ""))}">{esc(p.get("url", "n/a"))}</a>', esc(str(p.get('sha256', 'n/a'))[:12]), fmt(p.get('bytes')), fmt(p.get('count', p.get('keys'))),
                     esc(p.get('retrieved_utc', 'n/a')), on_disk, fmt(a.get('duplicates')), copy,
                     fmt(next((s[k] for k in size if k in s), None)), esc(next((s[k] for k in mode if k in s), 'n/a')), fmt(s.get('seed')),
                     esc(json.dumps({k: v for k, v in s.items() if k not in hidden}, default=str)[:90] if s else 'n/a')])
    return (notice('sha256 (prefix) is the download itself. "sorted on disk" and "duplicates" come from the sort audit (scaleli_hardness --sort-only); when the file is '
                   'unsorted or has duplicates every later step uses the sorted copy whose sha256 prefix and key count are in "sorted copy".')
            + table(['dataset', 'url', 'sha256 (prefix)', 'bytes', 'keys', 'retrieved (UTC)', 'sorted on disk', 'duplicates', 'sorted copy', 'sample keys', 'sample mode', 'seed', 'other sample fields'], rows))

def sec_hardness_space(hard):
    if not hard: return unavailable('hardness.json (scaleli_hardness summaries per dataset and scope)')
    def pt(d, scope, xk, yk):
        x, y = num(dig(hard, d, scope, xk)), num(dig(hard, d, scope, yk)); return None if x is None or y is None else (x, y)
    out = [notice('Arrows start at the raw keys and end where the control moved the dataset: teal = NFL-style flow (metrics on the sorted transformed '
                  'values), orange = CSV-style virtual points (metrics on the augmented slot sequence). Axes switch to log10 when values span more than two decades. '
                  'A flow that collapses many keys onto equal feature values makes the transformed sequence degenerate; section 5 lists the duplicate counts next to the metrics.'),
           notice(CD_NOTE)]
    for xk, xl, yk, yl, kind in [('pla_32', 'PLA-32 segments', 'pla_4096', 'PLA-4096 segments', 'GRE metric, Fig. 2 style'), ('pla_32', 'PLA-32 segments', 'conflict_degree', 'CD, keys per FMCD position', 'CD·PLA-32, Fig. 6 style')]:
        row = []
        for base, targets in [('full', [('full_flow', FLOW)]), ('sample', [('sample_flow', FLOW), ('sample_csv', CSV)])]:
            ps, arrows = [], []
            for d in order(hard):
                p = pt(d, base, xk, yk)
                if p is None: continue
                ps.append((p[0], p[1], d))
                arrows += [(f'→ {SCOPE_LABEL[s]}', c, p[0], p[1], q[0], q[1]) for s, c in targets if (q := pt(d, s, xk, yk))]
            row.append(scatter(f'{kind} — {SCOPE_LABEL[base]}', xl, yl, [(SCOPE_LABEL[base], INK, ps)], arrows) if ps else unavailable(f'scope "{base}" with {xl} and {yl} in hardness.json'))
        out.append('<div class="row">' + ''.join(row) + '</div>')
    return ''.join(out)

def sec_throughput(tp, sweep, known):
    out = []
    if tp:
        variants = list(tp); datasets = order({d for v in variants for d in (dig(tp, v) or {})}); series = [(v, PALETTE[i % len(PALETTE)]) for i, v in enumerate(variants)]
        out.append(grouped_bars('Median lookup throughput per dataset and variant (uniform lookups on the key samples, median over seeds)', 'MOPS', datasets, series,
                                lambda g, s: (x, None, None) if (x := num(dig(tp, s, g))) is not None else None))
        out.append(table(['dataset'] + [f'{v} (MOPS)' for v in variants], [[esc(d)] + [fmt(dig(tp, v, d)) for v in variants] for d in datasets]))
    else: out.append(unavailable('throughput.json (median MOPS per variant and dataset)'))
    if sweep:
        rows = [dict(r, dataset=base_name(r.get('dataset') or '', known)) for r in sweep if isinstance(r, dict)]
        groups = order({r['dataset'] for r in rows}); variants = list(dict.fromkeys(r.get('variant') or '' for r in rows)); cell = {}
        for r in rows: cell.setdefault((r['dataset'], r.get('variant')), r)
        out.append(grouped_bars(f'Paired throughput speedup vs {rows[0].get("baseline", "baseline")} (geometric mean over seeds; whiskers = seed-bootstrap 95% interval; dashed = parity)',
                                'speedup', groups, [(v, PALETTE[i % len(PALETTE)]) for i, v in enumerate(variants)],
                                lambda g, s: tuple(cell[g, s].get(k) for k in ('paired_throughput_speedup', 'seed_bootstrap_low', 'seed_bootstrap_high')) if (g, s) in cell else None, ref=1.0))
        cols = ['dataset', 'profile', 'variant', 'runs', 'throughput_ops_s_median', 'paired_throughput_speedup', 'seed_bootstrap_low', 'seed_bootstrap_high',
                'fence_probes_per_operation', 'rank_sse_ratio', 'virtual_points_per_key', 'flow_region_fraction', 'preprocess_ns_per_key']
        out.append(table(cols, [[fmt(r.get(c)) for c in cols] for r in rows]))
    else: out.append(unavailable('sweep/summary.csv (paired speedups, seed-bootstrap intervals and work counters)'))
    return ''.join(out)

def sec_scores(scores):
    if not scores: return unavailable('scores.json (conformance and coverage per scope)')
    out = [notice('Conf_I per variant, Conf = mean over variants, Cov = (|C| - |U|) / (|C| + |U|). Shading follows the value. Variants are SCALE-LI controls, not the paper\'s indexes.')]
    for scope in [s for s in SCOPE_LABEL if s in scores] + sorted(s for s in scores if s not in SCOPE_LABEL):
        metrics = dig(scores, scope, 'metrics') or {}; variants = dig(scores, scope, 'variants') or sorted({v for m in metrics.values() for v in (dig(m, 'per_variant') or {})})
        names = [m for m in METRICS if m in metrics] + sorted(m for m in metrics if m not in METRICS)
        out.append(f'<h3>Scope {esc(scope)} ({esc(SCOPE_LABEL.get(scope, scope))}) — datasets: {esc(", ".join(map(str, dig(scores, scope, "datasets") or [])) or "n/a")}</h3>')
        dims = lambda m: len(dig(metrics, m, 'dims') or m.split('·'))
        out.append(table(['metric', 'dims'] + [f'Conf {v}' for v in variants] + ['Conf', 'Cov', '|C|', '|U|'],
                         [[esc(m), dims(m)] + [shade(dig(metrics, m, 'per_variant', v)) for v in variants] + [shade(dig(metrics, m, 'conformance')), shade(dig(metrics, m, 'coverage')),
                          fmt(dig(metrics, m, 'comparable_pairs')), fmt(dig(metrics, m, 'incomparable_pairs'))] for m in names]))
        if scope in ('full', 'sample'):
            groups, coincident, seen = {}, [], {}
            for m in names:  # metrics sharing one (coverage, conformance) point collapse into one marker labelled ×n; the tooltip and the table below list them
                x, y = num(dig(metrics, m, 'coverage')), num(dig(metrics, m, 'conformance'))
                if x is None or y is None: continue
                key = (dims(m), round(x, 4), round(y, 4))
                if key in seen: seen[key].append(m)
                else: seen[key] = [m]
            for (k, x, y), ms in seen.items():
                if len(ms) == 1: groups.setdefault(k, []).append((x, y, ms[0], ms[0]))
                else: groups.setdefault(k, []).append((x, y, f'×{len(ms)}', ', '.join(ms))); coincident.append((k, x, y, ms))
            out.append(scatter(f'Conformance vs coverage, scope {scope} (Fig. 4 style; dashed = approximate Pareto frontier; ×n = n metrics at the same point)', 'coverage', 'conformance',
                               [(f'{k}-dim', {1: FLOW, 2: CSV, 3: '#4b6ea8'}.get(k, '#8a969e'), groups[k]) for k in sorted(groups)], fixed=(-1, 1), frontier=True, size=10, narrow=True) if groups else unavailable(f'numeric scores for scope {scope}'))
            if coincident:
                out.append('<details><summary>Coincident metrics in the scatter (' + str(sum(len(ms) for *_, ms in coincident)) + ' metrics on ' + str(len(coincident)) + ' shared points)</summary>'
                           + table(['dims', 'coverage', 'conformance', 'metrics'], [[k, fmt(x), fmt(y), esc(', '.join(ms))] for k, x, y, ms in sorted(coincident, key=lambda t: (t[0], -t[1], -t[2]))]) + '</details>')
    return ''.join(out)

def sec_shift(hard, details=None):
    if not hard: return unavailable('hardness.json')
    present = {s for d in hard for s in (dig(hard, d) or {})}; get = lambda d, s, k: dig(hard, d, s, k)
    cols = [(s, lambda d, k, s=s: fmt(get(d, s, k))) for s in ('full', 'sample') if s in present]
    for b, a in [('full', 'full_flow'), ('sample', 'sample_flow'), ('sample', 'sample_csv'), ('sample_flow', 'sample_flow_csv')]:
        if a in present: cols += [(a, lambda d, k, a=a: fmt(get(d, a, k))), (f'Δ {a} vs {b}', lambda d, k, a=a, b=b: pct(get(d, b, k), get(d, a, k)))]
    out = (notice('Relative change = (after - before) / before. A negative change means the control made the key-to-rank map easier under that metric; PLA counts are segments, CD is keys per position.')
           + notice(CD_NOTE)
           + table(['dataset', 'metric'] + [c for c, _ in cols], [[esc(d), name] + [f(d, k) for _, f in cols] for d in order(hard) for k, name in SCALARS]))
    # Degeneracy of the transformed / smoothed sequences: a flow can map many keys to one double (saturation), so
    # the metrics above then describe a collapsed sequence. Read from hardness.json blocks, else hardness_details.json.
    field = lambda d, s, k: dig(hard, d, s, k) if dig(hard, d, s, k) is not None else dig(details or {}, d, s, k)
    rows = []
    for d in order(hard):
        for s in ('full_flow', 'sample_flow', 'sample_csv', 'sample_flow_csv'):
            if s not in (dig(hard, d) or {}): continue
            keys = field(d, s, 'keys') if field(d, s, 'keys') is not None else dig(details or {}, d, 'full' if s.startswith('full') else 'sample', 'keys')
            dup, unordered = field(d, s, 'duplicates'), field(d, s, 'unordered_pairs')
            share = None if num(dup) is None or not num(keys) else num(dup) / num(keys)
            fm = dig(details or {}, d, s, 'fmcd') or {}
            rows.append([esc(d), esc(SCOPE_LABEL.get(s, s)), fmt(keys), fmt(dup), 'n/a' if share is None else f'{share:.1%}', fmt(unordered), fmt(field(d, s, 'virtual_points')), fmt(field(d, s, 'regions')),
                         fmt(get(d, s, 'conflict_degree')), fmt(fm.get('conflict_degree_lipp_epsilon')), fmt(fm.get('ut_epsilon'))])
    if rows:
        out += ('<h3>Degeneracy of the transformed and smoothed sequences</h3>'
                + notice('"duplicates" counts feature values equal to their predecessor after the flow (a saturated flow collapses most keys, and the metrics then describe that collapsed sequence); '
                         '"unordered pairs" counts descents the sort had to fix for a non-monotone flow. "CD (LIPP 1e-6)" is the literal-epsilon fit for comparison with the scaled-epsilon CD.')
                + table(['dataset', 'scope', 'keys', 'duplicates', 'duplicates / keys', 'unordered pairs', 'virtual points', 'regions', 'CD (scaled eps)', 'CD (LIPP 1e-6)', 'U_T epsilon used'], rows))
    return out

def sec_cost(flows, raw, sweep, hard):
    out = []
    if isinstance(flows, dict): flows = [dict(v, dataset=k) for k, v in flows.items() if isinstance(v, dict)]
    flows = [r for r in (flows or []) if isinstance(r, dict)]
    if flows:
        cols = [('dataset', 'dataset'), ('training_keys', 'training keys'), ('steps', 'steps'), ('train_seconds', 'train s'), ('best_nll', 'best NLL'), ('tail_conflict_degree_raw', 'tail CD raw'),
                ('tail_conflict_degree_transformed', 'tail CD flow'), ('unordered_transformed_pairs', 'unordered pairs'), ('monotone', 'monotone')]
        out.append('<h3>NFL-style flow training (stdlib stand-in trainer, not BNAF)</h3>' + table([l for _, l in cols], [[fmt(r.get(k)) for k, _ in cols] for r in flows]))
        out.append(grouped_bars('Flow training seconds per dataset', 'seconds', [str(r.get('dataset')) for r in flows], [('train_seconds', FLOW)],
                                lambda g, s: next(((r.get(s), None, None) for r in flows if str(r.get('dataset')) == g), None)))
    else: out.append(unavailable('flows/training_report.json (training seconds and NLL per dataset)'))
    runs = [(stem, r.get('keys'), dig(r, 'transformed') or {}, dig(r, 'smoothed') or {}) for stem, r in sorted(raw.items())]  # raw scaleli_hardness output
    runs += [(f'{d} \u00b7 {sc}', m.get('keys'), m, m) for d in order(hard or {}) for sc in SCOPE_LABEL if isinstance(m := dig(hard, d, sc), dict) and any(k in m for k in ('virtual_points', 'smoothing_ns', 'transform_ns'))]
    rows, ms = [], lambda v: None if num(v) is None else num(v) / 1e6
    for run, keys, t, s in runs:
        if any(k in t or k in s for k in ('transform_ns', 'virtual_points', 'smoothing_ns')):
            rows.append([esc(run), fmt(keys), fmt(ms(t.get('transform_ns'))), fmt(t.get('unordered_pairs')), fmt(s.get('regions')), fmt(s.get('virtual_points')),
                         fmt(None if num(s.get('virtual_points')) is None or not num(keys) else num(s.get('virtual_points')) / num(keys)), fmt(ms(s.get('smoothing_ns')))])
    if rows: out.append('<h3>Transform and smoothing cost (scaleli_hardness runs and hardness.json blocks)</h3>' + table(['run', 'keys', 'transform ms', 'unordered pairs', 'regions', 'virtual points', 'virtual points / key', 'smoothing ms'], rows))
    elif sweep:
        cols = ['dataset', 'variant', 'virtual_points_per_key', 'flow_region_fraction', 'preprocess_ns_per_key']
        out.append('<h3>Preprocessing inside the index runs (sweep summary)</h3>' + table(cols, [[fmt(r.get(c)) for c in cols] for r in sweep if isinstance(r, dict) and (num(r.get('preprocess_ns_per_key')) or 0) > 0]))
    else: out.append(unavailable('smoothing cost (no hardness/*.json raw runs and no sweep/summary.csv)'))
    return ''.join(out)

def sec_reproduce(log):
    cmds = ['cd experimental/scaleli', 'cmake -S ../.. -B ../../build-aidb -DCMAKE_BUILD_TYPE=Release && cmake --build ../../build-aidb --parallel 4',
            'python3 tools/aidb_pipeline.py --wait            # every step, idempotent: verify-downloads, sort, sample, flows, hardness, sweep, throughput, scores, report',
            'python3 tools/aidb_pipeline.py --steps report    # regenerate this page only',
            'python3 tools/aidb_pipeline.py --force           # redo every step',
            'python3 tools/aidb_pipeline.py --dry-run         # fixture + synthetic keys into results/aidb_dry/ (under two minutes)',
            'python3 tools/aidb_report.py --results results/aidb --output results/aidb/report.html', 'python3 -m unittest tests.test_aidb_report -v']
    out = ('<pre>' + esc('\n'.join(cmds)) + '</pre><p>Downloads are never started by the pipeline; it only verifies and waits for the ten GRE key files. The sort step writes a sorted, '
           'de-duplicated copy (<code>&lt;name&gt;.sorted</code>, 1.6 GB each) for the seven GRE files served unsorted and every later step uses it. See '
           '<code>tools/aidb_pipeline.py --help</code> for the build-directory, sample-size, ops, warm-up and <code>--extra-variants</code> options.</p>')
    if log: out +='<details><summary>pipeline.log (last 40 lines)</summary><pre>' + esc('\n'.join(log.splitlines()[-40:])) + '</pre></details>'
    return out

def build(results):
    hard, tp, scores, prov = (load(results / n) for n in ('hardness.json', 'throughput.json', 'scores.json', 'provenance.json'))
    hard, tp, scores, prov = (x if isinstance(x, dict) else None for x in (hard, tp, scores, prov))
    sweep = load(results / 'sweep' / 'summary.csv', 'csv'); flows = load(results / 'flows' / 'training_report.json'); log = load(results / 'pipeline.log', 'text')
    raw = {p.stem: r for p in sorted((results / 'hardness').glob('*.json')) if isinstance(r := load(p), dict)}
    details = load(results / 'hardness_details.json'); details = details if isinstance(details, dict) else None
    known = set(prov or {}) | set(hard or {}) | {d for v in (tp or {}).values() for d in (v if isinstance(v, dict) else {})}
    bodies = [sec_provenance(prov), sec_hardness_space(hard), sec_throughput(tp, sweep, known), sec_scores(scores), sec_shift(hard, details), sec_cost(flows, raw, sweep, hard), sec_reproduce(log)]
    style = ('body{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#17242f;background:#f6f8fa}h1{font-size:30px}h2{font-size:22px;margin-top:0}'
             'h3{font-size:16px;margin:18px 0 6px}p{line-height:1.5}.eyebrow{font-size:12px;letter-spacing:.08em;font-weight:700;color:#145a61}'
             'nav{position:sticky;top:0;background:#fff;padding:10px 14px;border-bottom:1px solid #ccd4da;z-index:1}nav a{margin-right:16px;color:#145a61;font-size:14px}'
             'section{background:#fff;border:1px solid #dde4e8;border-radius:8px;padding:22px;margin:22px 0;scroll-margin-top:60px}.notice{border-left:3px solid #a96420;padding-left:14px}'
             'figure{margin:14px 0}figure.narrow{max-width:720px}figcaption{font-weight:600;margin-bottom:4px;font-size:14px}svg{width:100%;color:#176771;background:#fff}.row{display:flex;gap:16px;flex-wrap:wrap}'
             '.row>figure,.row>.notice{flex:1 1 440px;min-width:0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0}'
             'th,td{padding:5px 6px;border-bottom:1px solid #dbe1e5;text-align:left;white-space:nowrap}th{background:#f1f5f6}.shade{display:block;padding:2px 4px;border-radius:3px}'
             'pre{overflow:auto;font-size:12px;line-height:1.5;padding:12px;background:#f2f5f7}a{color:#176771}footer{font-size:13px;margin:30px 0}')
    text = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>AIDB-style hardness protocol · NFL/CSV-style controls</title>'
            f'<style>{style}</style></head><body><h1>AIDB-style hardness protocol · NFL/CSV-style controls</h1>'
            f'<p class="eyebrow">REDUCED-SCALE CLEAN-ROOM CONTROLS · generated {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC} · {esc(results)}</p>{notice(PROTOCOL)}'
            '<nav>' + ''.join(f'<a href="#s{i}">{esc(h.split(". ", 1)[1])}</a>' for i, h in enumerate(HEADINGS, 1)) + '</nav>'
            + ''.join(f'<section id="s{i}"><h2>{esc(h)}</h2>{b}</section>' for i, (h, b) in enumerate(zip(HEADINGS, bodies), 1))
            + '<footer>Metric definitions follow Zhang, Tang and Ailamaki, AIDB@VLDB 2026 (protocol only; no result of that paper is reproduced here). Scope and sources: docs/APPROACHES.md.</footer></body></html>')
    return text

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=pathlib.Path, default=pathlib.Path('results/aidb')); p.add_argument('--output', type=pathlib.Path)
    a = p.parse_args(); out = a.output or a.results / 'report.html'
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(build(a.results)); print(out)
if __name__ == '__main__': main()
