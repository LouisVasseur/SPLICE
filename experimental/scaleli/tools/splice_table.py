#!/usr/bin/env python3
"""COUNTS.md for SPLICE-H: one row per dataset from splice_count's results/splice_h/<ds>.json.

    python3 tools/splice_table.py results/splice_h > results/splice_h/COUNTS.md

Reads the schema splice_count writes: "index" (stats of the chosen fast cell, materialised and verified),
"chosen" {fast, compact}, "fits" (one eps bisection per K1), "stage_wall_s", "stage2", "ablations", "verify",
"parity", "perturb", "compact_index" and "timings". Tables: the chosen cells and their exact counts, stability
(stage-1-only choice, fixed point, perturbation), the comparison with the synthesis' predicted table and its own
cell, the expected perf counters, the build passes, and the ablations. All values are counts and
model estimates from the Mac, never timings of lookups. Standard library only; runs on Python 3.6.
"""
import argparse
import json
import os
import re
import sys

DATASETS = ['fb', 'osm', 'books', 'covid', 'genome', 'history', 'libio', 'planet', 'stack', 'wise']
ARMS = ['4k_2048', '4k_1024', 'thp_2048', 'thp_1024']   # Stats::ed_milli order
ARM_NAMES = ['4K/2048', '4K/1024', 'THP/2048', 'THP/1024']
# The synthesis' predicted table, verbatim (splice_synthesis.md section 5, "Other datasets"): expected mode and
# E[D] on 4 KiB pages. A range is (lo, hi); 'about x' has no range and gets a signed difference instead of a verdict.
# Its E[D] has no compute term; ours includes d_c_direct or d_c_fenced (0.30 / 0.45 D per lookup), so the
# comparison subtracts them.
PREDICTED = {
    'stack': ('DIRECT, \u03b1 0.25, 1.04 lines', '2.2-2.4', (2.2, 2.4)),
    'wise': ('DIRECT, 1.31-1.64 lines', '2.3-2.7', (2.3, 2.7)),
    'history': ('DIRECT, 1.46-1.91 lines', '2.4-2.8', (2.4, 2.8)),
    'covid': ('DIRECT, 1.51-1.94 lines', '2.5-2.8', (2.5, 2.8)),
    'libio': ('70-87% DIRECT (2.51-2.83 D before tier-1 misses)', '2.6-3.0', (2.6, 3.0)),
    'books': ('47-59% DIRECT (3.09-3.32 D)', '3.1-3.5', (3.1, 3.5)),
    'planet': ('FENCED, over-capacity 0.074', '3.9-4.0', (3.9, 4.0)),
    'genome': ('FENCED, over-capacity 0.0009', 'about 3.8', 3.8),
    'fb': ('FENCED, over-capacity 0.134, plus 21 exceptions', '4.0-4.1', (4.0, 4.1)),
    'osm': ('FENCED, over-capacity 0.191', 'about 4.2', 4.2),
}
# Over-capacity shares the synthesis quotes at C = 228 (adv/advsim.cpp: a radix-trie coarse model, not eps-PLA).
PRED_OVERCAP = {'fb': 0.134, 'osm': 0.191, 'planet': 0.074, 'genome': 0.0009}
BASELINE = {'fast': (16384, 2, 24, 0), 'compact': (16384, 2, 24, 1)}
D_C_DIRECT, D_C_FENCED = 300, 450


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def milli(x, nd=3):
    x = num(x)
    return '-' if x is None else '%.*f' % (nd, x / 1000.0)


def share(x):
    x = num(x)
    return '-' if x is None else '%.3f' % (x / 1e6)


def arm_list(v):
    """E[D] of the four arms (milli-D) from a list (Stats) or a dict keyed 4k_2048 ... (cells)."""
    if isinstance(v, list) and len(v) == 4:
        return v
    if isinstance(v, dict):
        return [v.get(k) for k in ARMS]
    return [None] * 4


def args_of(cell):
    return (cell or {}).get('splice_args') or ''


def cell_tag(args):
    """The grid coordinates of a cell from its args line (k1, entry, cbar, compact; mode/tier1/exc if not default)."""
    kv = dict(t.split('=', 1) for t in args.split() if '=' in t)
    out = ['k1=%s' % kv.get('k1', '?'), 'e%s' % kv.get('entry', '?'), 'c%s' % kv.get('cbar', '?')]
    if kv.get('compact') == '1':
        out.append('compact')
    for k, d in (('mode', 'auto'), ('tier1', 'pla'), ('exc', '64')):
        if kv.get(k, d) != d:
            out.append('%s=%s' % (k, kv[k]))
    return ' '.join(out)


def compute_milli(ix):
    """Per-lookup compute constant included in the model's E[D] (mode-share weighted)."""
    d, f = num(ix.get('direct_key_ppm')), num(ix.get('fenced_key_ppm'))
    if d is None or f is None:
        return None
    return (d * D_C_DIRECT + f * D_C_FENCED) / 1e6


def load(d):
    names = [x for x in DATASETS if os.path.exists(os.path.join(d, x + '.json'))]
    names += sorted(f[:-5] for f in os.listdir(d) if f.endswith('.json') and f[:-5] not in names) \
        if os.path.isdir(d) else []
    out, bad = [], []
    for ds in names:
        try:
            with open(os.path.join(d, ds + '.json'), encoding='utf-8') as f:
                out.append((ds, json.load(f)))
        except (IOError, OSError, ValueError) as e:
            bad.append('%s: %s' % (ds, e))
    return out, bad


def kv_of(args):
    return dict(t.split('=', 1) for t in args.split() if '=' in t)


def key_of(args):
    kv = kv_of(args)
    try:
        return (int(kv['k1']), int(kv['entry']), int(kv['cbar']), int(kv['compact']))
    except (KeyError, ValueError):
        return None


def stage1_choice(j, arm):
    """The cell the 0.3-D rule picks from the stage-1 scorings alone (default residency constants, no L2
    simulation): what the layout would be without the fixed point."""
    cells = [c for c in j.get('cells') or [] if c.get('arm') == arm]
    key = lambda c: (c.get('k1'), c.get('entry'), c.get('cbar'), c.get('compact'))
    bl = [c for c in cells if key(c) == BASELINE[arm]]
    feas = [c for c in cells if not c.get('cap_violated')]
    if not feas:
        return None
    best = min(feas, key=lambda c: c.get('ed_milli'))
    if not bl or bl[0].get('cap_violated'):
        return key(best)
    return key(best) if best.get('ed_milli') + 300 <= bl[0].get('ed_milli') else key(bl[0])


def tag_of_key(k):
    return '-' if k is None else 'k1=%d e%d c%d%s' % (k[0], k[1], k[2], ' compact' if k[3] else '')


def counts(cell):
    return (cell or {}).get('counts') or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dir', help='results/splice_h')
    a = ap.parse_args()
    runs, bad = load(a.dir)
    o = ['# SPLICE-H counts (generated by tools/splice_table.py from results/splice_h/*.json)', '',
         'Counts and model estimates over all keys of each dataset (the `*.sorted` files where GRE\'s raw file is '
         'unsorted), from `splice_count --verify --parity --perturb` on the Mac; none of these is a timing. "lines" are '
         'dependent DRAM lines per lookup (+ lines issued in parallel), "walks" distinct 4 KiB pages per lookup, both '
         'exact counts over every stored key of the chosen fast cell (the materialised index; `get_impl<Count>` '
         'parity with the plan walker is exact where marked). E[D] is the model\'s expected serialized DRAM accesses '
         'per lookup (D, about 105-110 ns on the Atom, an estimate) for 4 KiB or 2 MiB (THP) pages and 2048 or 1024 KB '
         'of L2; it includes the per-mode compute constant (0.30 D DIRECT, 0.45 D FENCED). B/key is the index\'s '
         'total_bytes (arena rounded up to 2 MiB, records, router, exceptions), the quantity the bytes cap bounds. '
         'See docs/SPLICE_DESIGN.md.', '']

    o += ['## Chosen cells', '',
          '| dataset | n | exc | chosen fast cell | why | eps | segments | DIRECT share | FENCED share | over-capacity share '
          '| lines (+par) | walks | ' + ' | '.join('E[D] ' + k for k in ARM_NAMES) + ' | B/key fast | compact cell '
          '| B/key compact | E[D] compact 4K/2048 | verify (fast; compact) | parity |',
          '|' + '---|' * 22]
    for ds, j in runs:
        ix = j.get('index') or {}
        cix = j.get('compact_index') or {}
        ch = j.get('chosen') or {}
        fast, comp = ch.get('fast') or {}, ch.get('compact') or {}
        fc, cc = fast.get('cell') or {}, comp.get('cell') or {}
        ed = arm_list(ix.get('ed_milli'))

        def ver_of(v):
            return '-' if not v else ('%d/%d, %d absent false' % (v.get('found', 0), v.get('n', 0), v.get('absent_false', 0))
                                      if v.get('pass') else 'FAIL')
        ver = ver_of(j.get('verify'))
        if j.get('compact_verify'):
            ver += '; ' + ('pass' if j['compact_verify'].get('pass') else 'FAIL')
        par = j.get('parity') or {}
        par = '-' if not par else ('exact' if par.get('equal') else 'MISMATCH')
        cbpk = cix.get('bytes_per_key_milli', cc.get('bytes_per_key_milli'))
        o.append('| ' + ' | '.join([
            ds, str(j.get('n', '-')), str(ix.get('exceptions', '-')), '`%s`' % cell_tag(args_of(fc)), fast.get('why', '-'),
            str(ix.get('eps', '-')), str(ix.get('segments', '-')), share(ix.get('direct_key_ppm')),
            share(ix.get('fenced_key_ppm')), share(ix.get('overcap_key_ppm')),
            '%s (+%s)' % (milli(ix.get('dep_lines_milli')), milli(ix.get('par_lines_milli'))), milli(ix.get('pages4k_milli'))]
            + [milli(x) for x in ed] + [
            milli(ix.get('bytes_per_key_milli'), 2),
            '`%s`%s' % (cell_tag(args_of(cc)), ' (infeasible: no cell meets the cap; no args file)'
                        if comp.get('feasible') is False else ''), milli(cbpk, 2),
            milli(arm_list(cc.get('ed_milli'))[0]), ver, par]) + ' |')

    o += ['', '## Stability and the residency simulation', '',
          '"stage-1-only choice" is the cell the 0.3-D rule picks from the stage-1 scorings alone (default residency '
          'constants, no L2 simulation, segment-local pages); where it equals the chosen cell, the shared-L2 '
          'simulation priced and reported the layout but did not change it. "outer" is the fixed point\'s accepted '
          'iterations (1 = the re-selection with simulated residency was identical or not better by > 0.01 D). '
          '"perturb": the whole grid rerun on t\' = (t + 1e-6 t^2)/(1 + 1e-6) of the normalised keys; it passes if the '
          'chosen fast cell is the same and E[D] moves by at most 10 milli-D. "cut traces": sampled lookups of the '
          'chosen cell whose access trace exceeded the 1024-id cap (their simulated misses are understated).', '',
          '| dataset | chosen fast | stage-1-only choice (fast) | same | chosen compact | stage-1-only (compact) | same '
          '| outer (fast) | E[D] per iterate | perturb: same cell, dE[D] milli | cut traces |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for ds, j in runs:
        ch = j.get('chosen') or {}
        fk = key_of(args_of((ch.get('fast') or {}).get('cell')))
        ck = key_of(args_of((ch.get('compact') or {}).get('cell'))) if (ch.get('compact') or {}).get('feasible') else None
        s1f, s1c = stage1_choice(j, 'fast'), stage1_choice(j, 'compact')
        fcell = (ch.get('fast') or {}).get('cell') or {}
        pt = j.get('perturb')
        pts = '-' if not pt else '%s, %+d%s' % ('yes' if pt.get('same_cell') else 'NO', int(pt.get('ed_delta_milli', 0)),
                                                '' if pt.get('pass') else ' (FAIL)')
        o.append('| %s | `%s` | `%s` | %s | `%s` | `%s` | %s | %s | %s | %s | %s/%s |' % (
            ds, tag_of_key(fk), tag_of_key(s1f), 'yes' if fk == s1f else 'no', tag_of_key(ck), tag_of_key(s1c),
            'yes' if ck == s1c else 'no', fcell.get('outer_iters', '-'),
            ', '.join(str(x) for x in fcell.get('ed_hist') or []) or '-', pts,
            fcell.get('trace_truncated', '-'), fcell.get('trace_keys', '-')))

    o += ['', '## Against the synthesis\' predicted table', '',
          'Predicted: splice_synthesis.md section 5, quoted verbatim (estimates from the prototypes, before the merged '
          'design existed). "E[D] excl. compute" removes the mode-weighted compute constant. The two accountings still '
          'differ: the counted E[D] also charges simulated record and router misses, 0.08 D per router child level, '
          '0.09 D per page walk that hits L2 and 0.05 D per parallel line, which the synthesis\' table folded into its '
          'per-step ns or left out, so agreement within about 0.3 D is the most this comparison can show. "synthesis '
          'cell" is the configuration the predictions describe for FENCED data (K1 16384, 2-line entries, cbar 24, fast '
          'lines: C = 228, FENCED only, uncapped), counted like for like; it is compared with the prediction only where the '
          'synthesis predicted FENCED (fb, osm, planet, genome). The chosen cell is usually a different one.',
          '',
          '| dataset | predicted mode | predicted E[D] 4K | chosen cell: DIRECT / FENCED share | chosen: lines | chosen: '
          'E[D] excl. compute | chosen vs prediction | synthesis cell: over-capacity (predicted) | synthesis cell: lines '
          '| synthesis cell: E[D] excl. compute | synthesis cell vs prediction |', '|---|---|---|---|---|---|---|---|---|---|---|']

    def verdict(p, ex):
        if p is None or ex is None:
            return '-'
        r = p[2]
        if isinstance(r, tuple):
            return 'inside' if r[0] - 0.005 <= ex <= r[1] + 0.005 else ('below' if ex < r[0] else 'above')
        return '%+.2f D from %s' % (ex - r, p[1])

    for ds, j in runs:
        ix = j.get('index') or {}
        p = PREDICTED.get(ds)
        ed = num(arm_list(ix.get('ed_milli'))[0])
        cm = compute_milli(ix)
        ex = None if ed is None or cm is None else (ed - cm) / 1000.0
        sc = None
        for ab in j.get('ablations') or []:
            if ab.get('ablation') == 'synthesis_cell':
                sc = ab.get('cell') or {}
        sx = None
        if sc:
            sed = num(arm_list(sc.get('ed_milli'))[0])
            sx = None if sed is None else (sed - D_C_FENCED) / 1000.0
        po = PRED_OVERCAP.get(ds)
        o.append('| %s | %s | %s | %s / %s | %s | %s | %s | %s | %s | %s | %s |' % (
            ds, p[0] if p else '-', p[1] if p else '-', share(ix.get('direct_key_ppm')), share(ix.get('fenced_key_ppm')),
            milli(ix.get('dep_lines_milli')), '-' if ex is None else '%.3f' % ex, verdict(p, ex),
            '-' if not sc else '%s (%s)' % (share(counts(sc).get('overcap_key_ppm')), '-' if po is None else po),
            '-' if not sc else milli(counts(sc).get('dep_lines_milli')), '-' if sx is None else '%.3f' % sx,
            verdict(p, sx) if p and p[0].startswith('FENCED') else 'n/a (predicted mode is DIRECT)'))
    o += ['', 'The synthesis\' over-capacity shares come from `design_evidence/adv/advsim.cpp`, whose coarse model is a '
          'dyadic radix trie with a linear map per leaf; the implementation\'s tier-1 is eps-PLA (K1 16384). The '
          'spec\'s sanity band was fb 0.134 +/-0.02 and osm 0.191 +/-0.03.']

    o += ['', '## Counter check: what perf should see (SERVER.md 5b step 8)', '',
          'Per lookup of the chosen fast cell, from the materialised index: L2-missing loads = dependent + parallel '
          'arena lines (every line of a DIRECT window and both lines of a 2-line entry are loads) + simulated record and '
          'router misses (4 KiB, 2048 KB arm); walks = distinct 4 KiB pages. GRE\'s own per-op traffic is subtracted '
          'on the server with `nullindex`.', '',
          '| dataset | dep lines | par lines | record + router misses | expected L2-miss loads | expected walks |',
          '|---|---|---|---|---|---|']
    for ds, j in runs:
        ix = j.get('index') or {}
        dep, par = num(ix.get('dep_lines_milli')), num(ix.get('par_lines_milli'))
        rr = (num(ix.get('record_miss_milli')) or 0) + (num(ix.get('router_miss_milli')) or 0)
        tot = None if dep is None or par is None else dep + par + rr
        o.append('| %s | %s | %s | %s | %s | %s |' % (ds, milli(dep), milli(par), milli(rr), milli(tot),
                                                   milli(ix.get('pages4k_milli'))))

    o += ['', '## Build passes and wall time', '',
          'Grid (splice_count, pruned): PLA passes of the three eps bisections (K1 4096 / 8192 / 16384, run '
          'concurrently), candidate passes (one DIRECT placement pass per alpha and K1, one FENCED pass per K1 x entry x '
          'cbar), stage-2 fixed-point cells (the arms share their finalists), stage-3 ablations, total wall time on 16 '
          'threads (target 600 s; it includes the perturbed rerun, verify and parity). GRE replay: what bulk_load does '
          'with the chosen args file (eps explicit: one PLA pass), from the materialised index.', '',
          '| dataset | PLA passes (4k/8k/16k) | eps (4k/8k/16k) | candidate passes | stage-2 cells | ablations | grid wall s '
          '| fit / candidates / stage 2 / stage 3 s | GRE replay: PLA passes, outer iterations | replay build s (Mac) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for ds, j in runs:
        fits = j.get('fits') or []
        sw = j.get('stage_wall_s') or {}
        ix = j.get('index') or {}
        tm = j.get('timings') or {}
        bns = num(ix.get('build_ns'))
        o.append('| %s | %s | %s | %d | %d | %d | %s%s | %s | %s, %s | %s |' % (
            ds, '/'.join(str(f.get('passes', '-')) for f in fits), '/'.join(str(f.get('eps', '-')) for f in fits),
            len(sw.get('candidate_passes') or {}), len(j.get('stage2') or []), len(j.get('ablations') or []),
            tm.get('wall_s', '-'), ' (over target)' if tm.get('over_target') else '',
            ' / '.join(str(sw.get(k, '-')) for k in ('fit', 'candidates', 'stage2', 'stage3')),
            ix.get('passes', '-'), ix.get('outer_iters', '-'), '-' if bns is None else '%.1f' % (bns / 1e9)))

    o += ['', '## Ablations (at the chosen fast cell, each with the fixed point)', '',
          'Delta = ablation minus the chosen fast cell, in dependent lines and in E[D] (4 KiB, 2048 KB), from the exact '
          'plan walk; positive means the ablation is worse. tier1=histtree keeps everything but the knot placement: '
          'Hist-Tree count-split knots instead of eps-PLA knots, with the same per-leaf integer linear model, so it is '
          'not model-free. The synthesis\' rule is in lines: the "learned knots" claim dies where the Hist-Tree knots '
          'are within 0.01 lines (or within GRE noise). An E[D] gap of a few hundredths of a D is about 1% of the '
          'lookup, inside the 4.9-6.1% round-to-round spread measured on DIAS. synthesis_cell is not an ablation of the '
          'chosen cell (see the table above).', '',
          '| dataset | ablation | delta lines | delta E[D] | lines | E[D] 4K/2048 | B/key | segments | learned-knots verdict |',
          '|---|---|---|---|---|---|---|---|---|']
    for ds, j in runs:
        ix = j.get('index') or {}
        base_lines = num(ix.get('dep_lines_milli'))
        base_ed = num(arm_list(ix.get('ed_milli'))[0])
        for ab in j.get('ablations') or []:
            c = ab.get('cell') or {}
            ln = num(counts(c).get('dep_lines_milli'))
            dl = None if ln is None or base_lines is None else ln - base_lines
            dd = num(ab.get('ed_delta_milli'))
            v = ''
            if ab.get('ablation') == 'tier1=histtree' and dl is not None:
                v = ('killed: within 0.01 lines' if abs(dl) <= 10 else
                     ('lines favour PLA by %.3f; E[D] gap %s, below the 4.9%% round spread' % (
                         dl / 1000.0, '-' if dd is None or not base_ed else '%+.3f D (%.1f%%)' % (dd / 1000.0, 100.0 * abs(dd) / base_ed))
                      if dl > 0 else 'killed: Hist-Tree has fewer lines'))
            o.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                ds, ab.get('ablation', '?'), '-' if dl is None else '%+.3f' % (dl / 1000.0),
                '-' if dd is None else '%+.3f' % (dd / 1000.0), milli(ln), milli(arm_list(c.get('ed_milli'))[0]),
                milli(c.get('bytes_per_key_milli'), 2), ab.get('segments', '-'), v))

    o += ['', '## Layout hashes', '', '| dataset | fast layout_hash | fast full_hash | compact layout_hash |',
          '|---|---|---|---|']
    for ds, j in runs:
        o.append('| %s | `%s` | `%s` | %s |' % (ds, j.get('layout_hash', '-'), j.get('full_hash', '-'),
                                             '`%s`' % j['compact_layout_hash'] if j.get('compact_layout_hash') else '-'))
    if bad:
        o.append('\nNot read: ' + '; '.join(bad))
    if not runs:
        o.append('(no <dataset>.json in %s)' % a.dir)
    text = re.sub(r'[ \t]+\n', '\n', '\n'.join(o) + '\n')
    sys.stdout.buffer.write(text.encode('utf-8'))
    return 0 if runs and not bad else 1


if __name__ == '__main__':
    sys.exit(main())
