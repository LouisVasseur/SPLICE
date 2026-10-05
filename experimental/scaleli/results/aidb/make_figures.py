#!/usr/bin/env python3
"""Build the result figures from results/aidb/chart_data.json (stdlib only).

Two renderers share one chart model:
  * slide mode  : the Slides artifact format (SVG marks; every label is an HTML <p>, since fonts never
                  load inside an SVG there) -> one <section> file per figure under <deck>/project/slides/
  * page mode   : results/aidb/figures.html, a standalone page with SVG <text> labels, hover tooltips
                  (<title>) and a table twin under each chart.
Chart forms follow the data's job: dumbbells for before->after per item, vectors for direction of change
in a two-metric space, stacked bars for part-to-whole, dot plots for ordered states, a strip plot with a
noise band for paired speedups. Palette: three validated categorical slots (blue #2a78d6, orange #eb6834,
aqua #1baf7a) plus text and chrome tokens; text never wears a data colour.
"""
from __future__ import annotations
import json, math, pathlib, sys, html

HERE = pathlib.Path(__file__).resolve().parent
DATA = json.loads((HERE / "chart_data.json").read_text())
DS = DATA["datasets"]
BLUE, ORANGE, AQUA, BLUE_L = "#2a78d6", "#eb6834", "#1baf7a", "#86b6ef"
INK, INK2, MUTED, GRID, AXIS, SURF, BAND = "#17242f", "#4a5568", "#6b7480", "#e1e0d9", "#c3c2b7", "#fbfbf8", "#ececea"
GRAY = "#9aa5ad"
TF = "'IBM Plex Sans', Arial, sans-serif"; HF = "'Source Serif 4', Georgia, serif"

def esc(s): return html.escape(str(s))

# ------------------------------------------------------------------ primitives (both modes)
class Fig:
    """An SVG plot area plus labels; labels become <p> (slide) or <text> (page)."""
    def __init__(self, w, h): self.w, self.h, self.svg, self.labels, self.axis_title = w, h, [], [], None
    def axis(self, x, y, text):  # page mode draws it; slide mode moves it into the panel caption (pinned-label budget)
        self.axis_title = (x, y, text)
    def rect(self, x, y, w, h, fill, rx=0, title=None, opacity=1):
        t = f"<title>{esc(title)}</title>" if title else ""
        self.svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0,w):.1f}" height="{max(0,h):.1f}" rx="{rx}" fill="{fill}" opacity="{opacity}">{t}</rect>')
    def bar(self, x0, x1, yc, fill, thick=18, title=None):
        # thin bar, square at the baseline (x0), 4px rounded data end
        w = x1 - x0
        if w >= 8: self.rect(x0, yc - thick/2, w, thick, fill, rx=4, title=title); self.rect(x0, yc - thick/2, min(6, w/2), thick, fill)
        else: self.rect(x0, yc - thick/2, w, thick, fill, title=title)
    def line(self, x1, y1, x2, y2, stroke, width=2, title=None, dash=None):
        t = f"<title>{esc(title)}</title>" if title else ""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"{d}>{t}</line>')
    def dot(self, x, y, fill, r=6, title=None, hollow=False):
        t = f"<title>{esc(title)}</title>" if title else ""
        if hollow: self.svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r+3}" fill="none" stroke="{fill}" stroke-width="2.5">{t}</circle>')
        else: self.svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{SURF}" stroke-width="2">{t}</circle>')
    def arrow(self, x1, y1, x2, y2, stroke, title=None):
        L = math.hypot(x2-x1, y2-y1)
        if L < 4: self.dot(x1, y1, stroke, r=4, title=title); return
        ux, uy = (x2-x1)/L, (y2-y1)/L; hx, hy = x2 - ux*10, y2 - uy*10
        t = f"<title>{esc(title)}</title>" if title else ""
        self.svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{stroke}" stroke-width="2.2" stroke-linecap="round">{t}</line>')
        px, py = -uy, ux
        self.svg.append(f'<polygon points="{x2:.1f},{y2:.1f} {hx+px*5:.1f},{hy+py*5:.1f} {hx-px*5:.1f},{hy-py*5:.1f}" fill="{stroke}"/>')
    def label(self, x, y, text, size=20, anchor="start", color=INK2, weight=400):
        self.labels.append((x, y, text, size, anchor, color, weight))
    def render(self, mode, aria):
        body = "".join(self.svg)
        if mode == "page":
            labels = self.labels + ([(self.axis_title[0], self.axis_title[1], self.axis_title[2], 18, "middle", MUTED, 400)] if self.axis_title else [])
            texts = "".join(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{s}" text-anchor="{ {"start":"start","middle":"middle","end":"end"}[a] }" fill="{c}" font-weight="{w}" font-family="system-ui, sans-serif">{esc(t)}</text>' for x,y,t,s,a,c,w in labels)
            return f'<svg viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" role="img" aria-label="{esc(aria)}">{body}{texts}</svg>'
        ps = []
        for x, y, t, s, a, c, w in self.labels:
            width = max(40, int(0.62 * s * len(t)) + 12)
            left = x if a == "start" else (x - width/2 if a == "middle" else x - width)
            ps.append(f'<p style="position:absolute; left:{max(0,left):.0f}px; top:{max(0,y - s*0.85):.0f}px; width:{width}px; font-size:{s}px; line-height:1.1; color:{c}; font-weight:{w}; text-align:{ {"start":"left","middle":"center","end":"right"}[a] }; white-space:nowrap">{esc(t)}</p>')
        return f'<div style="position:relative; width:{self.w}px; height:{self.h}px"><svg aria-label="{esc(aria)}" viewBox="0 0 {self.w} {self.h}" style="width:{self.w}px; height:{self.h}px">{body}</svg>{"".join(ps)}</div>'

def legend(items, mode, size=20):
    """items: (color, text, kind) kind in dot|ring|bar|band"""
    if mode == "page":
        sw = {"dot": lambda c: f'<span style="display:inline-block;width:12px;height:12px;border-radius:6px;background:{c};margin-right:6px"></span>',
              "ring": lambda c: f'<span style="display:inline-block;width:10px;height:10px;border-radius:7px;border:2.5px solid {c};margin-right:6px"></span>',
              "bar": lambda c: f'<span style="display:inline-block;width:18px;height:10px;border-radius:3px;background:{c};margin-right:6px"></span>',
              "band": lambda c: f'<span style="display:inline-block;width:18px;height:12px;background:{c};margin-right:6px"></span>'}
        return '<div class="legend">' + "".join(f'<span>{sw[k](c)}{esc(t)}</span>' for c, t, k in items) + "</div>"
    out = []
    for c, t, k in items:
        if k == "dot": sw = f'<div style="width:14px; height:14px; border-radius:7px; background:{c}"></div>'
        elif k == "ring": sw = f'<div style="width:10px; height:10px; border-radius:7px; border:3px solid {c}"></div>'
        elif k == "band": sw = f'<div style="width:20px; height:14px; background:{c}"></div>'
        else: sw = f'<div style="width:20px; height:10px; border-radius:3px; background:{c}"></div>'
        out.append(f'<div style="display:flex; flex-direction:row; gap:8px; align-items:center">{sw}<p style="font-size:{size}px; color:{INK2}">{esc(t)}</p></div>')
    return f'<div style="display:flex; flex-direction:row; gap:28px; flex-wrap:wrap">{"".join(out)}</div>'

# ------------------------------------------------------------------ chart A: dumbbells of fence probes
def chart_probes(mode, panel_w=760):
    panels = []
    for pmode in ("uniform", "window"):
        rows = DATA["probes"][pmode]; lo, hi = 1.5, 5.5; L, R = 130, panel_w - 190; rh = 44; n = len(DS)
        f = Fig(panel_w, 40 + n*rh + 40); X = lambda v: L + (v-lo)/(hi-lo)*(R-L)
        for t in (2, 3, 4, 5):
            f.line(X(t), 30, X(t), 30 + n*rh, GRID, 1)
            if t in (2, 5): f.label(X(t), 30 + n*rh + 26, str(t), 18, "middle", MUTED)
        f.axis(X(3.5), 30 + n*rh + 52, "exact fence probes per lookup (2 to 5)")
        for i, d in enumerate(DS):
            y = 30 + i*rh + rh/2; r = rows[d]; c, ff, v = r["control"], r["forced_flow"], r["vp"]
            f.label(L - 12, y + 6, d, 20, "end", INK)
            f.line(X(c), y, X(v), y, GRAY, 3)
            f.dot(X(ff), y, ORANGE, r=6, hollow=True, title=f"{d} {pmode}: transform forced into every region {ff:.2f}")
            f.dot(X(c), y, GRAY, title=f"{d} {pmode}: control {c:.2f}")
            f.dot(X(v), y, BLUE, title=f"{d} {pmode}: virtual points {v:.2f}")
            f.label(R + 14, y + 6, f"{c:.2f} → {v:.2f}  ({100*(v-c)/c:+.0f}%)", 19, "start", INK2)
        panels.append((f"{pmode} 2M samples", f, f"Dumbbell chart of fence probes per lookup, {pmode} samples"))
    return panels

# ------------------------------------------------------------------ chart B: hardness-space vectors
def chart_moves(mode):
    def panel(title, pts, W=800, H=560):
        # pts: list of (name, base(x,y), [(kind,color,(x,y))...]) in raw units; log-log
        xs = [p[1][0] for p in pts] + [m[2][0] for p in pts for m in p[2]]; ys = [p[1][1] for p in pts] + [m[2][1] for p in pts for m in p[2]]
        lx0, lx1 = math.floor(math.log10(min(xs))*2)/2 - 0.1, math.ceil(math.log10(max(xs))*2)/2 + 0.1
        ly0, ly1 = math.floor(math.log10(min(ys))*2)/2 - 0.1, math.ceil(math.log10(max(ys))*2)/2 + 0.1
        L, R, T, B = 110, W - 30, 24, H - 70
        X = lambda v: L + (math.log10(v)-lx0)/(lx1-lx0)*(R-L); Y = lambda v: B - (math.log10(v)-ly0)/(ly1-ly0)*(B-T)
        f = Fig(W, H)
        for e in range(math.ceil(lx0), math.floor(lx1)+1):
            f.line(X(10**e), T, X(10**e), B, GRID, 1); f.label(X(10**e), B + 26, f"10^{e}", 18, "middle", MUTED)
        for e in range(math.ceil(ly0), math.floor(ly1)+1):
            f.line(L, Y(10**e), R, Y(10**e), GRID, 1); f.label(L - 8, Y(10**e) + 6, f"10^{e}", 18, "end", MUTED)
        f.line(L, B, R, B, AXIS, 1); f.line(L, T, L, B, AXIS, 1)
        f.axis((L+R)/2, B + 52, "x: PLA-32 segments, local (log) · y: RMSE, global (log)")
        for name, (bx, by), moves in pts:
            for kind, color, (mx, my) in moves: f.arrow(X(bx), Y(by), X(mx), Y(my), color, title=f"{name}: {kind}: PLA-32 {bx:.0f}→{mx:.0f}, RMSE {by:.3g}→{my:.3g}")
            f.dot(X(bx), Y(by), INK, r=5, title=f"{name}: PLA-32 {bx:.0f}, RMSE {by:.3g}")
            f.label(X(bx) + 8, Y(by) - 8, name, 18, "start", INK, 600)
        return title, f, f"Vectors in hardness space: {title}"
    pts_u = []
    for d in DS:
        h = DATA["hardness_sample"][d]["uniform"]; b = (h["sample"]["pla_32"], h["sample"]["rmse"])
        pts_u.append((d, b, [("transform", ORANGE, (h["sample_flow"]["pla_32"], h["sample_flow"]["rmse"])), ("virtual points", BLUE, (h["sample_csv"]["pla_32"], h["sample_csv"]["rmse"])), ("both", AQUA, (h["sample_flow_csv"]["pla_32"], h["sample_flow_csv"]["rmse"]))]))
    pts_f = []
    for d in DS:
        h = DATA["hardness_full"][d]; pts_f.append((d, (h["full"]["pla_32"], h["full"]["rmse"]), [("transform", ORANGE, (h["full_flow"]["pla_32"], h["full_flow"]["rmse"]))]))
    return [panel("2M uniform samples: three moves per dataset", pts_u), panel("200M keys: the transform's move", pts_f)]

# ------------------------------------------------------------------ chart C: selector choice shares
def chart_choices(mode, panel_w=760):
    panels = []
    for pmode in ("uniform", "window"):
        rows = DATA["choices"][pmode]; L, R = 130, panel_w - 110; rh = 40; n = len(DS)
        f = Fig(panel_w, 30 + n*rh + 44); X = lambda v: L + v*(R-L)
        for t in (0, .25, .5, .75, 1):
            f.line(X(t), 22, X(t), 22 + n*rh, GRID, 1)
            if t in (0, .5, 1): f.label(X(t), 22 + n*rh + 24, f"{t*100:.0f}%", 18, "middle", MUTED)
        f.axis(X(.5), 22 + n*rh + 50, "share of regions choosing each option")
        for i, d in enumerate(DS):
            y = 22 + i*rh + rh/2; r = rows[d]; x = 0.0
            f.label(L - 12, y + 6, d, 20, "end", INK)
            for key, col in (("vp", BLUE), ("both", AQUA), ("flow", ORANGE), ("none", GRAY)):
                v = r[key]
                if v <= 0: continue
                w = X(x+v) - X(x); f.rect(X(x) + (1 if x > 0 else 0), y - 10, max(0, w - 2), 20, col, rx=3, title=f"{d} {pmode}: {key} {100*v:.1f}%"); x += v
            f.label(R + 12, y + 6, f"{100*r['vp']:.0f}% points", 18, "start", INK2)
        panels.append((f"{pmode} 2M samples", f, f"Stacked bars of selector choices per region, {pmode} samples"))
    return panels

# ------------------------------------------------------------------ chart D: root probes dot plot
def chart_root(mode, panel_w=760):
    panels = []
    for pmode in ("uniform", "window"):
        L, R = 130, panel_w - 200; rh = 44; n = len(DS); lo, hi = 0, 10
        f = Fig(panel_w, 30 + n*rh + 44); X = lambda v: L + (v-lo)/(hi-lo)*(R-L)
        for t in (0, 2, 4, 6, 8, 10):
            f.line(X(t), 22, X(t), 22 + n*rh, GRID, 1)
            if t in (0, 4, 8): f.label(X(t), 22 + n*rh + 24, str(t), 18, "middle", MUTED)
        f.axis(X(5), 22 + n*rh + 50, "root probes per lookup (binary search over fences = 8.96)")
        for i, d in enumerate(DS):
            y = 22 + i*rh + rh/2; rec = DATA["root"][f"{d}_{pmode}"]
            b = rec["packed_rank"]["root_probes"]; raw = rec["packed_rank_root_raw"]["root_probes"]; fen = rec["packed_rank_root_vf4"]["root_probes"]
            f.label(L - 12, y + 6, d, 20, "end", INK)
            f.line(X(fen), y, X(b), y, GRAY, 3)
            f.dot(X(b), y, GRAY, title=f"{d} {pmode}: binary search {b:.2f}")
            f.dot(X(raw), y, BLUE_L, title=f"{d} {pmode}: learned root, raw key {raw:.2f}" + (" (fell back to binary)" if abs(raw-b) < 1e-6 else ""))
            f.dot(X(fen), y, BLUE, title=f"{d} {pmode}: raw key + virtual fences {fen:.2f}")
            f.label(R + 14, y + 6, f"{b:.1f} → {fen:.1f}", 19, "start", INK2)
        panels.append((f"{pmode} 2M samples", f, f"Dot plot of root probes per lookup, {pmode} samples"))
    return panels

# ------------------------------------------------------------------ chart E: cost bars + acceptance
def chart_cost(mode):
    c = DATA["cost"]; W, H = 900, 300; L, R = 330, W - 120; lo, hi = 0, 180
    f = Fig(W, H); X = lambda v: L + (v-lo)/(hi-lo)*(R-L)
    for t in (0, 50, 100, 150): f.line(X(t), 16, X(t), 16 + 5*46, GRID, 1); f.label(X(t), 16 + 5*46 + 24, f"{t} ns", 18, "middle", MUTED)
    f.axis(X(90), 16 + 5*46 + 50, "nanoseconds per lookup")
    rows = [("root probes the transform saves", (c["probes_saved"][0]*c["ns_per_probe"][0], c["probes_saved"][1]*c["ns_per_probe"][0]), BLUE, f"{c['probes_saved'][0]:.1f}-{c['probes_saved'][1]:.1f} probes × {c['ns_per_probe'][0]} ns"),
            ("one flow evaluation, unbatched (measured)", (c["flow_eval_ns"][1], c["flow_eval_ns"][2]), ORANGE, f"{c['flow_eval_ns'][0]} ns [{c['flow_eval_ns'][1]}, {c['flow_eval_ns'][2]}]"),
            ("net loss per lookup (measured)", (c["net_loss_ns"][1], c["net_loss_ns"][2]), ORANGE, f"{c['net_loss_ns'][0]} ns [{c['net_loss_ns'][1]}, {c['net_loss_ns'][2]}], −11.6%"),
            ("NFL paper, batch 1 (Table 2)", (c["nfl_batch_ns"]["1"], c["nfl_batch_ns"]["1"]), GRAY, f"{c['nfl_batch_ns']['1']} ns"),
            ("NFL paper, batch 256 (Table 2)", (c["nfl_batch_ns"]["256"], c["nfl_batch_ns"]["256"]), GRAY, f"{c['nfl_batch_ns']['256']} ns")]
    for i, (name, (a, b), col, lab) in enumerate(rows):
        y = 16 + i*46 + 23
        f.label(L - 12, y + 6, name, 19, "end", INK)
        if abs(b - a) < 1e-9: f.bar(X(0), X(min(a, hi)), y, col, title=f"{name}: {lab}")
        else: f.bar(X(0), X(min((a+b)/2, hi)), y, col, title=f"{name}: {lab}"); f.line(X(a), y, X(min(b, hi)), y, INK, 2); f.line(X(a), y-6, X(a), y+6, INK, 2); f.line(X(min(b,hi)), y-6, X(min(b,hi)), y+6, INK, 2)
        f.label(X(min(b, hi)) + 10, y + 6, lab, 18, "start", INK2)
    acc = Fig(700, 300); accL, accR = 110, 700 - 40; rh = 26; n = len(DS)
    Xa = lambda v: accL + v/0.10*(accR-accL)
    for t in (0, .025, .05, .075, .10): acc.line(Xa(t), 10, Xa(t), 10 + n*rh, GRID, 1); acc.label(Xa(t), 10 + n*rh + 22, f"{t*100:.1f}%", 16, "middle", MUTED)
    acc.axis(Xa(.05), 10 + n*rh + 44, "regions where NFL's bypass rule accepted the transform")
    for i, d in enumerate(DS):
        y = 10 + i*rh + rh/2; u = DATA["flow_accept"]["uniform"][d]; w = DATA["flow_accept"]["window"][d]
        acc.label(accL - 10, y + 5, d, 17, "end", INK)
        acc.bar(Xa(0), Xa(min(u, .1)), y - 6, ORANGE, thick=9, title=f"{d} uniform: {100*u:.1f}%"); acc.bar(Xa(0), Xa(min(w, .1)), y + 6, GRAY, thick=9, title=f"{d} window: {100*w:.1f}%")
    return [("Per-lookup cost of the transform against what it can save", f, "Bar chart of per-lookup nanoseconds"), ("How often the transform is accepted at region level (orange uniform, gray window)", acc, "Bars of transform acceptance per dataset")]

# ------------------------------------------------------------------ chart F: throughput strip plot
def chart_time(mode):
    variants = [("packed_rank_vp10", "region virtual points"), ("packed_rank_root_vf4", "root: raw key + virtual fences"), ("packed_rank_root_fusion", "root: selector (same structure, 2nd run)"), ("packed_rank_vp10_root_fusion", "region points + root"), ("raw_rank", "uncompressed rank routing"), ("sorted_vector", "binary search, no index")]
    W, H = 1300, 60 + len(variants)*56 + 60; L, R = 420, W - 160; lo, hi = math.log10(0.8), math.log10(2.6)
    f = Fig(W, H); X = lambda v: L + (math.log10(v)-lo)/(hi-lo)*(R-L)
    top, bot = 30, 30 + len(variants)*56
    f.rect(X(0.92), top, X(1.08)-X(0.92), bot-top, BAND, title="±8%: largest spread between two measurements of an identical structure")
    for t in (0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5): f.line(X(t), top, X(t), bot, GRID, 1); f.label(X(t), bot + 26, f"{t:g}×", 18, "middle", MUTED)
    f.line(X(1), top, X(1), bot, INK, 1.5)
    f.axis(X(1.35), bot + 52, "paired throughput vs the packed control with a binary root (log scale); one dot per dataset sample")
    for i, (v, name) in enumerate(variants):
        y = 30 + i*56 + 28; f.label(L - 14, y + 6, name, 20, "end", INK)
        sp = DATA["speedups"].get(v, {}); vals = sorted(x[0] for x in sp.values())
        for smp, (s, lo_, hi_) in sp.items(): f.dot(X(min(max(s, 0.8), 2.6)), y, BLUE if v not in ("raw_rank", "sorted_vector") else GRAY, r=6, title=f"{smp}: {s:.3f} [{lo_:.2f}, {hi_:.2f}]")
        if vals:
            gm = math.exp(sum(math.log(x) for x in vals)/len(vals)); f.line(X(gm), y-16, X(gm), y+16, INK, 2.5)
            f.label(R + 16, y + 6, f"geo-mean {gm:.2f}×", 19, "start", INK2)
    return [("Time effects are inside the noise band, except removing compression", f, "Strip plot of paired speedups per variant")]

# ------------------------------------------------------------------ chart G: hardness bars
def chart_hardness(mode):
    order = sorted(DS, key=lambda d: DATA["hardness_full"][d]["full"]["pla_32"], reverse=True)
    def panel(key, title, lo, hi, W=560):
        L, R = 110, W - 140; rh = 40; n = len(order); f = Fig(W, 22 + n*rh + 50)
        X = lambda v: L + (math.log10(max(v, 1))-lo)/(hi-lo)*(R-L)
        for e in range(math.ceil(lo), math.floor(hi)+1): f.line(X(10**e), 14, X(10**e), 14 + n*rh, GRID, 1); f.label(X(10**e), 14 + n*rh + 24, f"10^{e}", 17, "middle", MUTED)
        f.axis(X(10**((lo+hi)/2)), 14 + n*rh + 48, title)
        for i, d in enumerate(order):
            y = 14 + i*rh + rh/2; v = DATA["hardness_full"][d]["full"][key]
            f.label(L - 10, y + 6, d, 19, "end", INK); f.bar(X(10**lo), X(v), y, BLUE, thick=16, title=f"{d}: {v:,.0f}")
            f.label(X(v) + 8, y + 6, f"{v:,.0f}" + ("  = paper" if d in ("history", "libio") and key == "pla_32" else ""), 17, "start", INK2)
        return f
    return [("PLA-32 segments (local hardness), full 200M keys", panel("pla_32", "PLA-32 segments (log)", 4, 6.2), "Bars of PLA-32 per dataset"),
            ("Conflict degree (LIPP FMCD), full 200M keys", panel("conflict_degree", "conflict degree (log)", 0, 3.7), "Bars of conflict degree per dataset")]

# ------------------------------------------------------------------ chart H: granularity small multiples
def chart_granularity(mode, panel_w=330):
    """One panel per dataset; rows are region sizes; paired thin bars control (gray) vs virtual points (blue)."""
    G = DATA["granularity"]; sizes = G["sizes"]; panels = []
    big = mode == "slide"; rh = 118 if big else 52; th = 22 if big else 12; fs = 22 if big else 18
    for k, d in enumerate(G["order"]):
        L, R = 100, panel_w - 84; f = Fig(panel_w, 12 + 3*rh + 54); lo, hi = 0, 11
        X = lambda v: L + (v-lo)/(hi-lo)*(R-L)
        for t in (0, 5, 10):
            f.line(X(t), 12, X(t), 12 + 3*rh, GRID, 1); f.label(X(t), 12 + 3*rh + 26, str(t), 18, "middle", MUTED)
        if mode == "page" and k == 0: f.axis(X(5.5), 12 + 3*rh + 50, "fence probes per lookup; rows = keys per region")
        for i, sz in enumerate(sizes):
            y = 12 + i*rh + rh/2; r = G["rows"][d][sz]; c, v, ff = r["control"], r["vp"], r["forced_flow"]
            f.label(L - 12, y + 7, f"{int(sz):,}", fs, "end", INK)
            f.bar(X(0), X(c), y - th*0.75, GRAY, thick=th, title=f"{d}, {int(sz):,}-key regions: control {c:.2f} (forced transform {ff:.2f})")
            f.bar(X(0), X(v), y + th*0.75, BLUE, thick=th, title=f"{d}, {int(sz):,}-key regions: virtual points {v:.2f}")
            f.label(X(max(c, v)) + 10, y + 7, f"{100*(v-c)/c:+.0f}%", fs, "start", INK2)
        panels.append((d, f, f"Paired bars of fence probes by region size, {d}"))
    return panels

# ------------------------------------------------------------------ chart I: the transform, before and after retraining
def chart_flowcheck(mode, panel_w=760):
    """Two dumbbell panels: what retraining the transform buys on NFL's own metric, and what it costs in our index."""
    F = DATA["flowcheck"]; rows = F["rows"]; order = F["order"]; n = len(order)
    # A: NFL's own criterion, tail conflict degree of the whole sample (log axis)
    L, R = 120, panel_w - 168; rh = 44
    f = Fig(panel_w, 26 + n*rh + 54); hi = math.log10(F["tcd_axis_max"])
    X = lambda v: L + math.log10(max(v, 1))/hi*(R - L)
    for t in (1, 10, 100):
        if t > F["tcd_axis_max"]: continue
        f.line(X(t), 20, X(t), 20 + n*rh, GRID, 1); f.label(X(t), 20 + n*rh + 24, str(t), 18, "middle", MUTED)
    f.line(X(4), 20, X(4), 20 + n*rh, AQUA, 2)
    f.axis(X(10), 20 + n*rh + 50, "tail conflict degree of the sample (log); aqua line = NFL's stated post-transform target of 4")
    for i, d in enumerate(order):
        y = 20 + i*rh + rh/2; r = rows[d]; raw, mono, free = r["raw"], r["mono"], r["free"]
        f.label(L - 12, y + 6, d, 20, "end", INK)
        if free != raw: f.line(X(min(raw, free)), y, X(max(raw, free)), y, GRAY, 3)
        f.dot(X(raw), y, GRAY, title=f"{d}: raw keys {raw}")
        f.dot(X(mono), y, ORANGE, r=6, hollow=True, title=f"{d}: monotone flow, the only kind our map can use: {mono}")
        f.dot(X(free), y, ORANGE, title=f"{d}: unconstrained flow, NFL's own design: {free}")
        if mode == "page" or free < raw: f.label(R + 14, y + 6, f"{raw} → {free}" if free < raw else f"{raw}, unchanged", 19, "start", INK2)
    pa = ("What retraining buys: NFL's criterion is met", f, "Dumbbell of tail conflict degree per dataset")
    # B: what it costs inside our index, exact fence probes per lookup
    L2, R2 = 120, panel_w - 168; g = Fig(panel_w, 26 + n*rh + 54); lo2, hi2 = 1.5, F["probe_axis_max"]
    X2 = lambda v: L2 + (min(v, hi2) - lo2)/(hi2 - lo2)*(R2 - L2)
    for t in (2, 8):
        g.line(X2(t), 20, X2(t), 20 + n*rh, GRID, 1); g.label(X2(t), 20 + n*rh + 24, str(t), 18, "middle", MUTED)
    g.axis(X2((lo2 + hi2)/2), 20 + n*rh + 50, "exact fence probes per lookup; left is better")
    for i, d in enumerate(order):
        y = 20 + i*rh + rh/2; r = rows[d]
        c, fl, vp = r.get("probes_control"), r.get("probes_flow"), r.get("probes_vp")
        g.label(L2 - 12, y + 6, d, 20, "end", INK)
        if c is None: continue
        g.line(X2(c), y, X2(max(fl or c, c)), y, GRAY, 3)
        if mode == "page" and vp is not None: g.dot(X2(vp), y, BLUE, title=f"{d}: virtual points {vp:.2f}")
        g.dot(X2(c), y, GRAY, title=f"{d}: control, no transform {c:.2f}")
        if fl is not None: g.dot(X2(fl), y, ORANGE, title=f"{d}: retrained unconstrained flow {fl:.2f}" + (" (off the axis)" if fl > hi2 else ""))
        if fl is not None: g.label(R2 + 14, y + 6, f"{100*(fl-c)/c:+.0f}%", 19, "start", INK2)
    pb = ("What it costs: a non-monotone feature in a key-ordered map", g, "Dumbbell of fence probes per lookup")
    return [pa, pb]


# ------------------------------------------------------------------ slide + page assembly
def slide(id_, eyebrow, title, body, footer, notes):
    return f'''<section id="{id_}" data-transition="fade" style="background:#f6f8fa; color:{INK}; font-family:{TF}; padding:96px 96px 150px; display:flex; flex-direction:column; gap:20px">
<p style="font-size:22px; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:#176771">{esc(eyebrow)}</p>
<h2 style="font-family:{HF}; font-size:52px; font-weight:600; line-height:1.1; color:{INK}">{esc(title)}</h2>
{body}
<p style="position:absolute; left:96px; bottom:56px; width:1728px; font-size:20px; color:{INK2}">{esc(footer)}</p>
<aside>{esc(notes)}</aside>
</section>
'''
def panel_block(panels, mode, gap=32):
    inner = "".join(f'<div style="flex:1; display:flex; flex-direction:column; gap:6px"><h3 style="font-size:24px; font-weight:600; color:{INK2}">{esc(t)}{esc(" · " + fig.axis_title[2]) if fig.axis_title else ""}</h3>{fig.render(mode, aria)}</div>' for t, fig, aria in panels)
    return f'<div style="display:flex; flex-direction:row; gap:{gap}px">{inner}</div>'

def build_slides(deck_dir: pathlib.Path):
    S = deck_dir / "project/slides"; S.mkdir(parents=True, exist_ok=True); m = "slide"
    S.joinpath("results-csv.html").write_text(slide("results-csv", "5 · Results: CSV-style virtual points alone",
        "Virtual points cut exact fence probes on every dataset; a forced transform moves nothing",
        legend([(GRAY, "control (binary root, no transform)", "dot"), (ORANGE, "transform forced into every region", "ring"), (BLUE, "virtual points, budget 10 percent", "dot")], m) + panel_block(chart_probes(m), m),
        "Metric: exact fence probes per lookup from the software counters (deterministic, identical query traces); results/aidb/sweep and results/aidb_window/sweep, 300 paired runs each.",
        "Read left to right: every gray dot moves to a blue dot; the orange ring sits exactly on the gray dot on all ten datasets, which is the transform doing nothing at region scale. This is the mechanism metric, not a timing."))
    G = DATA["granularity"]
    S.joinpath("results-granularity.html").write_text(slide("results-granularity", "5 · Results: CSV-style virtual points at coarser regions",
        "The saving from virtual points grows with region size: one line fits a larger region worse, and virtual points repair most of it",
        legend([(GRAY, "control: one linear model per region", "bar"), (BLUE, "virtual points, budget 10 percent", "bar")], m) + f'<p style="font-size:22px; color:{INK2}">Bars: exact fence probes per lookup (0 to 11). Rows: keys per region, 4,096 / 16,384 / 32,768. Labels: change from control to virtual points.</p>' + panel_block(chart_granularity(m), m, gap=24),
        f"Metric: exact fence probes per lookup, mean of two query seeds; results/aidb_granularity, {G['runs']} runs (fb, osm, planet, covid, genome × regions of 4,096 / 16,384 / 32,768 keys × 15 variants × 2 seeds). At every size the selector chose the transform in {G['flow_chosen']} of {G['regions_total']:,} regions and a forced transform stayed within 0.05 probes of the control.",
        "Read each panel top to bottom: as regions grow, the gray control bar grows because one linear model per region fits worse, and the blue bar grows much less. osm is the exception at every size: its local structure is too irregular for a 10 percent budget. This is the scale argument for CSV: the coarser the model, the more the virtual points buy; the transform still buys nothing."))
    S.joinpath("connect.html").write_text(slide("connect", "3 · Why they connect",
        "The two methods move a dataset along different hardness axes",
        legend([(ORANGE, "transform (NFL-style)", "bar"), (BLUE, "virtual points (CSV-style)", "bar"), (AQUA, "both", "bar")], m) + panel_block(chart_moves(m), m),
        "Axes: PLA-32 segments (the local metric the paper found predictive for local-correction indexes) and RMSE of one linear fit (global). Log scales; arrows go from the dataset to its transformed or augmented version; results/aidb/hardness.json.",
        "Orange arrows are vertical: the transform changes global shape, not local segment count. Blue arrows are horizontal: virtual points change local structure, not the global fit. Aqua is close to the vector sum. Right panel: at 200M keys the transform's arrow is again vertical."))
    S.joinpath("results-fusion-region.html").write_text(slide("results-fusion-region", "7 · Results with fusion: regions",
        "Given a free choice per region, the selector takes virtual points almost everywhere and the transform nowhere",
        legend([(BLUE, "virtual points", "bar"), (AQUA, "both", "bar"), (ORANGE, "transform", "bar"), (GRAY, "none", "bar")], m) + panel_block(chart_choices(m), m),
        "Metric: share of the 489 regions per sample choosing each option, scored by the probes the real locate routine spends on the region's own keys; learnability.choices in the sweep outputs.",
        "A stacked bar that is entirely one colour is the point: the selector, given every combination, never finds a region where the transform helps. Both is never chosen, so there is no region-level synergy."))
    S.joinpath("results-fusion-root.html").write_text(slide("results-fusion-root", "7 · Results with fusion: root",
        "At the root, virtual fences on the raw key take routing from 9 probes to 2 or 3; the flow feature is never chosen",
        legend([(GRAY, "binary search over fences", "dot"), (BLUE_L, "learned root, raw key only", "dot"), (BLUE, "raw key + virtual fences (budget 4× regions)", "dot")], m) + panel_block(chart_root(m), m),
        "Metric: measured root probes per lookup (deterministic); corrected fit on real first keys; results/aidb_final/root_analysis.json, 540 paired runs. The flow feature was offered as a candidate on every sample and chosen on none.",
        "Light blue shows how far the raw key alone gets; dark blue adds virtual fences. On the windows the raw key alone is often weak (planet, genome) and the fences fix it; on uniform samples the raw key is often enough. The transform candidate never wins, so it is not drawn."))
    S.joinpath("results-nfl.html").write_text(slide("results-nfl", "5 · Results: NFL-style transform alone",
        "The transform saves at most 4 to 6 root probes (about 20 ns) and costs 66 ns per unbatched lookup",
        panel_block(chart_cost(m), m),
        "Left: paired per-seed decomposition, 105 run pairs; whiskers are dataset-cluster bootstrap intervals; NFL's own batch costs from its Table 2. Right: NFL's bypass rule applied per region (orange uniform, gray window); results/aidb/verification/coststory.",
        "This is why batching is the whole game for the transform: at batch 256 the same fusion would gain a few percent; at batch 1 it loses 12 percent. The right panel shows the transform rarely even passes NFL's own acceptance test inside a region."))
    S.joinpath("results-time.html").write_text(slide("results-time", "7 · Results with fusion: time",
        "In wall-clock time the learnability variants sit in or at the edge of the noise band; only removing compression is clearly outside it",
        legend([(BLUE, "learnability variant, one dot per sample", "dot"), (GRAY, "reference variants", "dot"), (BAND, "±8 percent: largest spread between two runs of one identical structure", "band")], m) + panel_block(chart_time(m), m),
        "Metric: paired throughput vs the packed control, geometric mean over three query seeds on identical traces, 5M lookups per run, performance-core QoS; results/aidb_final/sweep/summary.csv. Tick marks are the geometric mean over the 20 samples.",
        "Honest reading: the probe reductions are real and deterministic, but on this host and at 2M keys they do not move the clock outside the band that two identical runs already span. Uncompressed routing and plain binary search do, which says decoding dominates."))
    S.joinpath("hardness.html").write_text(slide("hardness", "4 · Benchmark datasets and properties",
        "Our hardness numbers match the values the paper prints and span two decades across the ten datasets",
        panel_block(chart_hardness(m), m) + f'<p style="font-size:22px; line-height:1.4; color:{INK2}">Coverage of every CD-free metric equals the paper’s Table 2, so our RMSE, ME, PLA-32 and PLA-4096 orderings agree with the paper’s; CD-containing metrics differ by exactly one of 45 pairs (the paper prints no CD values). PLA counts are identical to PGM-index’s segmentation; the scorer matches an independent implementation of equations 1 to 3.</p>',
        "Full 200M-key files; PLA-32 via O’Rourke / PGM-index segmentation; CD via LIPP’s FMCD fit; results/aidb/hardness.json. Bars sorted by PLA-32; log axes because the values span two decades.",
        "Sorted bars on a log axis so the two-decade spread is visible. history and libio are the two datasets whose values the paper prints, and they match to the digit."))
    S.joinpath("methods").with_suffix(".html").write_text(slide("methods", "Appendix",
        "How the plots were made",
        f'''<div style="display:flex; flex-direction:row; gap:28px">
<div style="flex:1; display:flex; flex-direction:column; gap:10px; background:{SURF}; padding:28px; border:1px solid #dde4e8; border-radius:12px"><h3 style="font-size:28px; font-weight:600">Metric before form</h3><ul style="font-size:22px; line-height:1.45; color:{INK2}; padding:0 0 0 26px"><li>Mechanism claims use deterministic software counters (fence probes, root probes, selector choices), never timing</li><li>Complementarity uses the paper’s own hardness axes: PLA-32 (local) against RMSE (global)</li><li>Time uses paired ratios on identical traces, with the noise band measured on identical structures</li></ul></div>
<div style="flex:1; display:flex; flex-direction:column; gap:10px; background:{SURF}; padding:28px; border:1px solid #dde4e8; border-radius:12px"><h3 style="font-size:28px; font-weight:600">Form by the data’s job</h3><ul style="font-size:22px; line-height:1.45; color:{INK2}; padding:0 0 0 26px"><li>Before → after per dataset: dumbbells and dot plots</li><li>Direction of change in two metrics: vectors on log-log axes</li><li>Share of regions: stacked bars; distribution of paired speedups: strip plot with the noise band</li><li>Hardness spanning decades: sorted bars on a log axis</li></ul></div>
<div style="flex:1; display:flex; flex-direction:column; gap:10px; background:{SURF}; padding:28px; border:1px solid #dde4e8; border-radius:12px"><h3 style="font-size:28px; font-weight:600">Colour and marks</h3><ul style="font-size:22px; line-height:1.45; color:{INK2}; padding:0 0 0 26px"><li>Three categorical slots, fixed roles: blue = virtual points, orange = transform, aqua = both; gray = control; text never wears a data colour</li><li>Palette validated for colour-vision deficiency (worst adjacent ΔE 9.2) and contrast on the card surface; aqua carries direct labels</li><li>Thin marks, 2-pixel surface rings and gaps, hairline solid gridlines, selective labels</li></ul></div>
</div>
<p style="font-size:22px; line-height:1.4; color:{INK2}">Every chart is generated from results/aidb/chart_data.json by results/aidb/make_figures.py (standard library only); the same code renders an interactive page with hover values and a table twin under each chart at results/aidb/figures.html.</p>''',
        "Data: results/aidb, results/aidb_window, results/aidb_final; verification archive results/aidb/verification.",
        "If asked why a plot looks the way it does, the answer is on this slide: metric first, then form, then colour, each chosen by a rule rather than taste."))

def build_page(out: pathlib.Path):
    m = "page"; sections = []
    def sec(title, sub, panels, table):
        sections.append(f'<section><h2>{esc(title)}</h2><p class="sub">{esc(sub)}</p><div class="panels">' + "".join(f'<figure><figcaption>{esc(t)}</figcaption>{fig.render(m, aria)}</figure>' for t, fig, aria in panels) + f'</div><details><summary>Table view</summary>{table}</details></section>')
    def tbl(header, rows): return "<table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr>" + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows) + "</table>"
    sec("Virtual points cut exact fence probes on every dataset; a forced transform moves nothing", "Dumbbell per dataset: control → virtual points; ring = transform forced into every region.", chart_probes(m),
        tbl(["dataset", "mode", "control", "forced transform", "virtual points"], [(d, pm, f"{DATA['probes'][pm][d]['control']:.2f}", f"{DATA['probes'][pm][d]['forced_flow']:.2f}", f"{DATA['probes'][pm][d]['vp']:.2f}") for pm in ("uniform", "window") for d in DS]))
    sec("The two methods move a dataset along different hardness axes", "Vectors in (PLA-32, RMSE) space; left: 2M uniform samples, right: full 200M keys (transform only).", chart_moves(m),
        tbl(["dataset", "scope", "PLA-32", "RMSE"], [(d, sc, f"{DATA['hardness_sample'][d]['uniform'][sc]['pla_32']:.0f}", f"{DATA['hardness_sample'][d]['uniform'][sc]['rmse']:.4g}") for d in DS for sc in ("sample", "sample_flow", "sample_csv", "sample_flow_csv")] + [(d, sc, f"{DATA['hardness_full'][d][sc]['pla_32']:,.0f}", f"{DATA['hardness_full'][d][sc]['rmse']:.4g}") for d in DS for sc in ("full", "full_flow")]))
    sec("Given a free choice per region, the selector takes virtual points almost everywhere and the transform nowhere", "Stacked share of regions per option.", chart_choices(m),
        tbl(["dataset", "mode", "points", "both", "transform", "none"], [(d, pm, *(f"{100*DATA['choices'][pm][d][k]:.1f}%" for k in ("vp", "both", "flow", "none"))) for pm in ("uniform", "window") for d in DS]))
    sec("At the root, virtual fences on the raw key take routing from 9 probes to 2 or 3; the flow feature is never chosen", "Dot plot of measured root probes per lookup.", chart_root(m),
        tbl(["sample", "binary", "raw root", "raw + fences", "virtual fences"], [(s, f"{r['packed_rank']['root_probes']:.2f}", f"{r['packed_rank_root_raw']['root_probes']:.2f}", f"{r['packed_rank_root_vf4']['root_probes']:.2f}", r['packed_rank_root_vf4']['virtual']) for s, r in DATA["root"].items()]))
    sec("The transform saves at most 4 to 6 root probes (about 20 ns) and costs 66 ns per unbatched lookup", "Per-lookup cost decomposition and NFL's own batch costs; acceptance of the transform per region.", chart_cost(m),
        tbl(["quantity", "value"], [("ns per root probe saved", "4.1 [2.6, 5.5]"), ("flow evaluation per lookup", "66 ns [56, 77]"), ("net loss per lookup", "47 ns [40, 53]"), ("NFL Table 2 batch 1 / 256", "169.5 / 8.4 ns per key")]))
    sec("In wall-clock time the learnability variants sit in or at the edge of the noise band; only removing compression is clearly outside it", "Strip plot of paired speedups; band = ±8%, the largest spread between two runs of one identical structure.", chart_time(m),
        tbl(["variant", "sample", "speedup", "seed spread"], [(v, s, f"{x[0]:.3f}", f"[{x[1]:.2f}, {x[2]:.2f}]") for v, sp in DATA["speedups"].items() for s, x in sp.items()]))
    sec("Retraining the transform: NFL's criterion is met, but only by a flow our index cannot use", "Left: tail conflict degree per dataset, raw keys against the retrained flow; the hollow ring is the monotone flow, the only kind a key-ordered map can use. Right: what the unconstrained flow costs in exact fence probes, against virtual points on the same samples.", chart_flowcheck(m),
        tbl(["dataset", "raw", "monotone flow", "unconstrained flow", "inverted pairs", "control probes", "flow probes", "virtual-point probes"], [(d, r["raw"], r["mono"], r["free"], r["unord"], f'{r["probes_control"]:.2f}' if r.get("probes_control") else "-", f'{r["probes_flow"]:.2f}' if r.get("probes_flow") else "-", f'{r["probes_vp"]:.2f}' if r.get("probes_vp") else "-") for d in DATA["flowcheck"]["order"] for r in [DATA["flowcheck"]["rows"][d]]]))
    sec("The saving from virtual points grows with region size", "One panel per dataset; rows are keys per region; gray = control, blue = virtual points at a 10% budget; mean of two query seeds.", chart_granularity(m),
        tbl(["dataset", "keys per region", "control", "forced transform", "virtual points", "change"], [(d, f"{int(sz):,}", f"{r['control']:.2f}", f"{r['forced_flow']:.2f}", f"{r['vp']:.2f}", f"{100*(r['vp']-r['control'])/r['control']:+.0f}%") for d in DATA["granularity"]["order"] for sz in DATA["granularity"]["sizes"] for r in [DATA["granularity"]["rows"][d][sz]]]))
    sec("Our hardness numbers match the values the paper prints and span two decades", "Sorted bars, log axes.", chart_hardness(m),
        tbl(["dataset", "PLA-32", "PLA-4096", "CD", "RMSE"], [(d, f"{DATA['hardness_full'][d]['full']['pla_32']:,}", f"{DATA['hardness_full'][d]['full']['pla_4096']:,}", f"{DATA['hardness_full'][d]['full']['conflict_degree']:,}", f"{DATA['hardness_full'][d]['full']['rmse']:.4g}") for d in DS]))
    css = f"body{{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:{INK};background:#f9f9f7}}section{{background:{SURF};border:1px solid #e1e0d9;border-radius:10px;padding:22px;margin:22px 0}}h2{{font-size:24px;margin:0 0 4px}}.sub{{color:{INK2};margin:0 0 12px}}.panels{{display:flex;gap:24px;flex-wrap:wrap}}figure{{margin:0}}figcaption{{font-weight:600;color:{INK2};margin-bottom:4px}}svg{{max-width:100%;height:auto}}.legend span{{margin-right:22px;font-size:14px;color:{INK2}}}table{{border-collapse:collapse;font-size:13px;margin-top:8px}}th,td{{padding:4px 10px;border-bottom:1px solid #e1e0d9;text-align:left;font-variant-numeric:tabular-nums}}details{{margin-top:8px}}"
    out.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Fusion figures</title><style>{css}</style><h1 style="font-size:28px">NFL-style transform, CSV-style virtual points, and their fusion: figures</h1><p class="sub">Generated by results/aidb/make_figures.py from results/aidb/chart_data.json. Hover any mark for its value; each chart has a table view.</p>' + "".join(sections) + "</html>")

if __name__ == "__main__":
    deck = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if deck: build_slides(deck); print("slides written to", deck / "project/slides")
    build_page(HERE / "figures.html"); print("page written to", HERE / "figures.html")
