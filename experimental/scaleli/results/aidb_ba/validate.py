#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
__doc__ = """Setup validation gate: is the SCALE-LI index + harness compatible with the data, the metrics, the two methods
(NFL-style flow, CSV-style virtual points), the baselines, this host and PROTOCOL.md? Stdlib only, Python >= 3.6.

Every check prints PASS, FAIL, WARN, UNKNOWN (an input is missing) or INFO, with the numbers behind it, under three
verdicts:
  GATE    the setup is internally consistent: right answers, right data, exact accounting, methods isolated
  TIMING  extra conditions before a timed number counts: A/A noise gate, warm-up, quiet host, P/X options
  SCOPE   what the setup can show at all: headroom per level, the methods' host assumptions, baselines

Inputs (run serially on the measurement host; see SERVER.md):
  python3 run_ba.py plan validity > validity_plan.json && python3 run_ba.py run validity_plan.json validity.jsonl
  python3 run_ba.py plan verify   > verify_plan.json   && python3 run_ba.py run verify_plan.json verify.jsonl
  python3 run_ba.py plan aa       > aa_plan.json       && python3 run_ba.py run aa_plan.json aa.jsonl      (night 0)
  python3 rss_ba.py                                                                  (real memory, fb B and C)
  python3 validate.py --runs validity.jsonl verify.jsonl aa.jsonl --out validation

Exit status: 0 = gate passed, 1 = a gate check failed, 2 = gate incomplete (some input missing).
Check definitions and code references: docs/CODE_DISSECTION.md and PROTOCOL.md (K1-K7, 6.7, 6.9).
"""
import argparse, glob, hashlib, json, math, os, platform, re, statistics, subprocess, sys, textwrap

S = REPO + '/experimental/scaleli'
GRE = S + '/data/external/gre'
HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get('SCALELI_BUILD_DIR') or REPO + '/build-fs'
BENCH = os.environ.get('SCALELI_BENCH') or os.environ.get('SCALELI_BIN') or BUILD + '/scaleli_bench'
HARD = os.environ.get('SCALELI_HARDNESS') or BUILD + '/scaleli_hardness'
DATASETS = ['books', 'covid', 'fb', 'genome', 'history', 'libio', 'osm', 'planet', 'stack', 'wise']
SORTED = {'covid', 'genome', 'history', 'libio', 'planet', 'stack', 'wise'}
FULL_ROWS = 200000000
SEED_D = 1001
PAPER_PLA = {'history': (105468, 468), 'libio': (145808, 639)}   # PLA-32 / PLA-4096 as printed in the AIDB 2026 paper
MAC_HARDNESS = S + '/results/aidb/hardness.json'                 # full-scope metrics computed on the Mac (2026-09-21)
FLOWCHECK = S + '/results/aidb_flowv2/flowcheck_summary.json'      # NFL tail-conflict check on the 2M samples (Mac)
PX_OPTIONS = ['cooldown-s', 'go-dir', 'cycles', 'probe-latency-mb', 'freq-probe',
              'interleave', 'rounds', 'visit-warmup', 'visit-ops', 'sub-chunks']   # PROTOCOL 4.1 X/P lines (patches P1/P2/P6)
Z20 = -0.8416212335729143            # standard normal 20% quantile (Wilson-Hilferty chi-square quantile)

D_EXPECT = {'prefault': True, 'chunks': 16, 'warmup_mode': 'workload', 'instrumented': True, 'seed': SEED_D,
            'operations': 5000000, 'load_ratio': 1, 'requested_miss_ratio': 0, 'query_distribution': 'uniform',
            'profile': 'read_only', 'verified': False}                      # PROTOCOL 4.1, D line
SCALELI = {'index': 'scaleli', 'policy': 'min_bytes', 'root': 'model', 'build_threads': 16}
CELL_EXPECT = {
    'B':  dict(SCALELI, routing='rank', root_alpha=0, virtual_alpha=0, flow_weights=''),
    'Bb': dict(SCALELI, routing='binary', root_alpha=0, virtual_alpha=0, flow_weights=''),
    'N':  dict(SCALELI, routing='rank', root_alpha=0, virtual_alpha=0, flow_bypass=True, flow_cost=0),
    'Nf': dict(SCALELI, routing='rank', root_alpha=0, virtual_alpha=0, flow_bypass=False, flow_cost=0),
    'Cr': dict(SCALELI, routing='rank', root_alpha=0.1, virtual_alpha=0, flow_weights=''),
    'C':  dict(SCALELI, routing='rank', root_alpha=0.1, virtual_alpha=0.1, flow_weights=''),
    'NC': dict(SCALELI, routing='rank', root_alpha=0.1, virtual_alpha=0.1, flow_bypass=True, flow_cost=0),
    'SV': {'index': 'sorted_vector'},
}
FLOW_CELLS = {'N', 'Nf', 'NC', 'Nm'}
CONFIG_KEYS = ('index', 'policy', 'routing', 'forced_codec', 'region_keys', 'block_keys', 'delta_limit', 'restart_interval',
               'min_saving_fraction', 'smooth_scale', 'flow_weights', 'flow_bypass', 'flow_min_gain', 'virtual_alpha',
               'relearn_on_compaction', 'fusion', 'flow_cost', 'root', 'root_alpha')   # everything that shapes the built index

CHECKS, CLAIMS = [], []
def add(cid, kind, title, status, evidence, detail=()):
    CHECKS.append({'id': cid, 'kind': kind, 'title': title, 'status': status, 'evidence': evidence, 'detail': list(detail)})

def sh(cmd, timeout=120, cwd=None):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout, cwd=cwd)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.SubprocessError) as e:
        return None, '', str(e)

def rd(path):
    try:
        with open(path) as f: return f.read().strip()
    except OSError: return None

# ---------------------------------------------------------------- run records (output of run_ba.py run)
def dataset_of(r):
    m = r.get('ba') or {}
    if m.get('dataset'): return m['dataset']
    b = os.path.basename(r.get('dataset_path') or '')
    return b[:-len('.sorted')] if b.endswith('.sorted') else b

def cell_of(r):
    m = r.get('ba') or {}
    if m.get('cell'): return m['cell']
    if r.get('index') != 'scaleli': return {'sorted_vector': 'SV'}.get(r.get('index'), r.get('index'))
    flow, va, ra = r.get('flow_weights') or '', float(r.get('virtual_alpha') or 0), float(r.get('root_alpha') or 0)
    if r.get('policy') == 'raw': return 'RC' if ra > 0 else 'RAW'
    if r.get('root') != 'model': return 'B0'
    if r.get('routing') == 'binary': return 'Bb'
    if 'flows_monotone' in flow: return 'Nm'
    if va > 0: return 'NC' if flow else 'C'
    if flow: return 'N' if r.get('flow_bypass') else 'Nf'
    return 'Cr' if ra > 0 else 'B'

def layer_of(r):
    m = r.get('ba') or {}
    if m.get('layer'): return m['layer']
    if r.get('verified'): return 'V'
    if r.get('instrumented') and r.get('chunks') == 16 and r.get('seed') == SEED_D: return 'D'
    if r.get('chunks') == 80 and not r.get('instrumented'): return 'E1'
    return 'other'

def per_op(r):
    if not r.get('instrumented'): return None
    w, n = r.get('work_counters') or {}, float(r.get('operations') or 0)
    if not n: return None
    g = lambda k: (w.get(k) or 0) / n
    o = {'root': g('root_probes'), 'coord': g('coordinate_probes'), 'fence': g('fence_probes'),
         'key_at': g('key_at_calls'), 'delta': g('delta_probes'), 'transform': g('transform_calls')}
    o['total'] = o['root'] + o['coord'] + o['fence'] + o['key_at'] + o['delta']
    return o

def load(paths):
    good, bad = [], []
    for p in paths:
        for path, out in ((p, good), (p + '.failed', bad)):
            if not os.path.exists(path): continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try: r = json.loads(line)
                    except ValueError: continue
                    m = r.get('ba') or {}
                    r['_file'] = os.path.basename(path)
                    r['_ds'] = dataset_of(r) if 'dataset_path' in r else m.get('dataset', '?')
                    r['_cell'] = cell_of(r) if 'index' in r else m.get('cell', '?')
                    r['_layer'] = layer_of(r) if 'index' in r else m.get('layer', '?')
                    r['_op'] = per_op(r) if 'index' in r else None
                    out.append(r)
    return good, bad

def tkey(r):   # records with the same tkey replayed the same trace on the same loaded rows
    return (r.get('dataset_path'), r.get('seed'), r.get('operations'), r.get('profile'), r.get('query_distribution'),
            r.get('requested_miss_ratio'), r.get('load_ratio'), r.get('initial_rows'))

def primaries(recs):
    """One instrumented record per (dataset, cell), all from ONE trace per dataset (preferring the D layer), so that
    cross-cell equalities are exact."""
    by_ds, prim = {}, {}
    for r in recs:
        if r['_op'] is not None: by_ds.setdefault(r['_ds'], []).append(r)
    for d, rs in by_ds.items():
        groups = {}
        for r in rs: groups.setdefault(tkey(r), []).append(r)
        best = max(groups.values(), key=lambda g: (len(set(x['_cell'] for x in g)), sum(x['_layer'] == 'D' for x in g)))
        for r in sorted(best, key=lambda x: x['_layer'] != 'D'):
            prim.setdefault((d, r['_cell']), r)
    return prim

def tag(r): return '%s/%s' % (r['_ds'], r['_cell'])
def md(r): return r['memory_before']['metadata_bytes']
def L(r): return r.get('learnability') or {}
def fmt(x): return '{:,}'.format(x)

# ---------------------------------------------------------------- GATE: data
def chk_data_hash(skip):
    title = 'GRE files are byte-identical to the copies every result was computed on (SHA256SUMS)'
    sums = GRE + '/SHA256SUMS'
    if skip: return add('DATA-1', 'gate', title, 'UNKNOWN', 'skipped (--no-hash)')
    if not os.path.exists(sums): return add('DATA-1', 'gate', title, 'UNKNOWN', 'missing ' + sums)
    expected = dict(reversed(l.split()) for l in open(sums) if l.strip())
    cache_path = GRE + '/.sha256_cache.json'
    try: cache = json.load(open(cache_path))
    except (OSError, ValueError): cache = {}
    ok, bad, missing, hashed = [], [], [], 0
    for name in sorted(expected):
        path = GRE + '/' + name
        if not os.path.exists(path): missing.append(name); continue
        st = os.stat(path); stamp = '%d:%d' % (st.st_size, st.st_mtime_ns)
        c = cache.get(name) or {}
        if c.get('stamp') == stamp: digest = c['sha256']
        else:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 23), b''): h.update(chunk)
            digest = h.hexdigest(); cache[name] = {'stamp': stamp, 'sha256': digest}; hashed += 1
        (ok if digest == expected[name] else bad).append(name)
    try:
        with open(cache_path, 'w') as f: json.dump(cache, f, indent=1)
    except OSError: pass
    ev = '%d/%d match (%d hashed now, the rest cached by size+mtime)' % (len(ok), len(expected), hashed)
    if bad: ev += '; MISMATCH: ' + ', '.join(bad)
    if missing: ev += '; missing: ' + ', '.join(missing)
    add('DATA-1', 'gate', title, 'FAIL' if bad else ('WARN' if missing else 'PASS'), ev)

def chk_rows(recs):
    title = 'each dataset loads as exactly 200,000,000 unique keys (nothing de-duplicated or truncated)'
    seen = {}
    for r in recs:
        if 'source_rows' in r:
            seen.setdefault(r['_ds'], set()).add((r['source_rows'], r['unique_rows'], r['initial_rows'], r.get('dataset_scope')))
    if not seen: return add('DATA-2', 'gate', title, 'UNKNOWN', 'no run records')
    ok, bad, prefix = [], [], []
    for d, vals in sorted(seen.items()):
        for s, u, i, scope in sorted(vals, key=str):
            if scope == 'prefix': prefix.append('%s (%s rows)' % (d, fmt(i)))
            elif s == u == i == FULL_ROWS: ok.append(d)
            else: bad.append('%s: source %s, unique %s, loaded %s' % (d, fmt(s), fmt(u), fmt(i)))
    missing = [d for d in DATASETS if d not in seen]
    ev = 'exact on: ' + (', '.join(sorted(set(ok))) or 'none')
    if prefix: ev += '; SMOKE (--limit prefix): ' + ', '.join(sorted(set(prefix)))
    if missing: ev += '; not run yet: ' + ', '.join(missing)
    add('DATA-2', 'gate', title, 'FAIL' if bad else ('WARN' if prefix else ('UNKNOWN' if missing else 'PASS')), ev, bad)

# ---------------------------------------------------------------- GATE: runs and correctness
def chk_failed(bad):
    title = 'every planned run completed (no crash, timeout, missing data file or answer mismatch)'
    if not bad: return add('RUN-0', 'gate', title, 'PASS', 'no .failed records')
    lines = ['%s/%s [%s]: %s' % (m.get('dataset'), m.get('cell'), m.get('layer', '?'), str(m.get('status'))[:200].replace('\n', ' '))
             for m in ((r.get('ba') or {}) for r in bad)]
    add('RUN-0', 'gate', title, 'FAIL', '%d failed runs' % len(bad), lines)

def chk_verify(recs, bad):
    title = 'every configuration returns the same answers as std::map on all lookups plus a full scan (--verify 1)'
    v = [r for r in recs if r['_layer'] == 'V']
    vf = [r for r in bad if (r.get('ba') or {}).get('layer') == 'V']
    if not v and not vf:
        return add('CORR-1', 'gate', title, 'UNKNOWN', 'no verify runs (run_ba.py plan verify)')
    ok = sorted(set(tag(r) for r in v if r.get('verified') is True))
    notv = sorted(set(tag(r) for r in v if r.get('verified') is not True))
    fails = ['%s/%s: %s' % (m.get('dataset'), m.get('cell'), str(m.get('status'))[:160]) for m in ((r.get('ba') or {}) for r in vf)]
    add('CORR-1', 'gate', title, 'FAIL' if fails or notv else 'PASS', 'verified: ' + (', '.join(ok) or 'none'),
        fails + ['not verified: ' + x for x in notv])

def chk_checksums(recs):
    title = 'all configurations answer identically: result_checksum and trace_fingerprint agree across cells, layers and SV (K7, 6.9-1/2)'
    groups = {}
    for r in recs:
        if 'result_checksum' in r: groups.setdefault(tkey(r), []).append(r)
    multi = dict((k, g) for k, g in groups.items() if len(g) > 1)
    if not multi: return add('CORR-2', 'gate', title, 'UNKNOWN', 'no trace was replayed by two runs')
    bad, n = [], 0
    for k, g in multi.items():
        n += len(g)
        if len(set(r['result_checksum'] for r in g)) > 1 or len(set(r['trace_fingerprint'] for r in g)) > 1:
            by = {}
            for r in g: by.setdefault(r['result_checksum'], []).append('%s/%s' % (r['_cell'], r['_layer']))
            bad.append('%s seed %s: %s' % (os.path.basename(k[0] or '?'), k[1], ' | '.join('..%s: %s' % (c[-6:], ','.join(v)) for c, v in by.items())))
    cells = ' '.join(sorted(set(r['_cell'] for g in multi.values() for r in g)))
    add('CORR-2', 'gate', title, 'FAIL' if bad else 'PASS',
        ('%d of %d shared traces disagree' % (len(bad), len(multi))) if bad else
        '%d runs over %d shared traces agree (cells: %s)' % (n, len(multi), cells), bad)

def chk_counted(recs):
    title = 'counters come from a replay that reproduced the timed answers (the run aborts otherwise)'
    inst = [r for r in recs if r.get('instrumented')]
    if not inst: return add('CORR-3', 'gate', title, 'UNKNOWN', 'no instrumented runs (run_ba.py plan validity)')
    add('CORR-3', 'gate', title, 'PASS', '%d instrumented runs completed; counter_digest == throughput_digest is enforced in src/benchmark.cpp pass 4' % len(inst))

# ---------------------------------------------------------------- GATE: metrics
def chk_prefault(recs):
    title = 'memory accounting covers exactly the allocations the prefault sweep walks (prefault_bytes = accounted - slack)'
    rs = [r for r in recs if r.get('prefault') and r.get('prefault_supported')]
    if not rs: return add('MET-1', 'gate', title, 'UNKNOWN', 'no prefaulted runs')
    bad, warn = [], []
    for r in rs:
        m = r['memory_before']; gap = m['accounted_bytes'] - m['reserved_slack_bytes'] - r['prefault_bytes']
        if gap == 0: continue
        n_reg = L(r).get('regions') or 0
        (warn if 0 < gap <= 8 * n_reg else bad).append('%s: accounted-slack exceeds prefault_bytes by %s B' % (tag(r), fmt(gap)))
    ex = rs[0]
    add('MET-1', 'gate', title, 'FAIL' if bad else ('WARN' if warn else 'PASS'),
        '%d runs; e.g. %s: %s = %s' % (len(rs), tag(ex), fmt(ex['prefault_bytes']), fmt(ex['memory_before']['accounted_bytes'])), bad + warn)

def chk_values(recs):
    title = 'values cost exactly 8 B/key in every SCALE-LI cell; SV stores exactly 16 B/key'
    bad, n = [], 0
    for r in recs:
        m = r.get('memory_before')
        if not m: continue
        n += 1; rows = r['initial_rows']
        if r.get('index') == 'scaleli' and m['value_bytes'] != 8 * rows: bad.append('%s: value_bytes %s for %s rows' % (tag(r), fmt(m['value_bytes']), fmt(rows)))
        if r.get('index') == 'sorted_vector' and m['key_bytes'] + m['value_bytes'] != 16 * rows: bad.append('%s: %s B for %s rows' % (tag(r), fmt(m['key_bytes'] + m['value_bytes']), fmt(rows)))
    if not n: return add('MET-2', 'gate', title, 'UNKNOWN', 'no run records')
    add('MET-2', 'gate', title, 'FAIL' if bad else 'PASS', '%d runs checked' % n, bad)

def chk_csv_accounting(prim):
    title = "CSV's memory is exactly 8 B per stored virtual point (C - Cr) and 4 B per root slot (Cr - B)"
    ev, bad = [], []
    for d in DATASETS:
        B, Cr, C = prim.get((d, 'B')), prim.get((d, 'Cr')), prim.get((d, 'C'))
        if not (B and Cr and C): continue
        vp, lr = L(C).get('virtual_points', 0), L(Cr)
        table = 4 * (lr.get('regions', 0) + lr.get('root_virtual', 0)) if lr.get('root_vp') else 0
        d1, d2, rows = md(C) - md(Cr), md(Cr) - md(B), C['initial_rows']
        if d1 != 8 * vp: bad.append('%s: C-Cr metadata %s B vs 8 x %s points' % (d, fmt(d1), fmt(vp)))
        if d2 != table: bad.append('%s: Cr-B metadata %s B vs root table %s B' % (d, fmt(d2), fmt(table)))
        ev.append('%s +%.3f B/key (%s points) +%.5f B/key (table)' % (d, d1 / rows, fmt(vp), d2 / rows))
    if not ev: return add('MET-3', 'gate', title, 'UNKNOWN', 'needs B, Cr and C on the same trace')
    add('MET-3', 'gate', title, 'FAIL' if bad else 'PASS', '; '.join(ev[:4]) + (' ...' if len(ev) > 4 else ''), bad)

def chk_counter_sanity(recs):
    title = 'counters behave as the code says: 32-way block search ~5, 128-key block ~7-8, binary root ~log2(regions), SV ~log2(n)'
    bad, n = [], 0
    for r in recs:
        o = r['_op']
        if not o: continue
        n += 1; t = tag(r)
        if r.get('index') == 'sorted_vector':
            lg = math.log2(max(2, r['initial_rows']))
            if not lg - 0.5 <= o['root'] <= lg + 1.0: bad.append('%s: %.2f comparisons vs log2(n) %.2f' % (t, o['root'], lg))
            continue
        if r.get('index') != 'scaleli': continue
        lb, lk, l = math.log2(max(2, r['region_keys'] // r['block_keys'])), math.log2(r['block_keys']), L(r)
        if r.get('routing') == 'binary':
            if o['coord'] != 0 or not lb - 0.5 <= o['fence'] <= lb + 1.0: bad.append('%s: binary routing coord %.2f fence %.2f vs log2 %.2f' % (t, o['coord'], o['fence'], lb))
        elif not lb - 0.5 <= o['coord'] <= lb + 1.0: bad.append('%s: coordinate probes %.2f vs log2(blocks) %.2f' % (t, o['coord'], lb))
        if not lk - 0.5 <= o['key_at'] <= lk + 2.0: bad.append('%s: key_at %.2f vs log2(block) %.2f' % (t, o['key_at'], lk))
        if not l.get('root_model') and l.get('regions', 0) > 1:
            lr = math.log2(l['regions'])
            if not lr - 0.5 <= o['root'] <= lr + 1.0: bad.append('%s: binary root %.2f vs log2(regions) %.2f' % (t, o['root'], lr))
        if o['delta']: bad.append('%s: %.3f delta probes in a read-only run' % (t, o['delta']))
    if not n: return add('MET-4', 'gate', title, 'UNKNOWN', 'no instrumented runs')
    add('MET-4', 'gate', title, 'FAIL' if bad else 'PASS', '%d instrumented runs within bounds' % (n - len(bad)), bad)

def run_hardness(datasets):
    out = {}
    for d in datasets:
        path = GRE + '/' + d + ('.sorted' if d in SORTED and os.path.exists(GRE + '/' + d + '.sorted') else '')
        if not os.path.exists(path): out[d] = {'error': 'missing ' + path}; continue
        presorted = d not in SORTED or path.endswith('.sorted')     # GRE serves seven files unsorted; sort those in memory
        rc, so, se = sh([HARD, '--data', path, '--dtype', 'uint64', '--region-keys', '4096', '--pla-eps', '32,4096',
                         '--check-sorted', '1' if presorted else '0'], timeout=1800)
        if rc != 0: out[d] = {'error': (se or so or 'rc=%s' % rc).strip()[-300:]}; continue
        j = json.loads(so); o = j['original']
        out[d] = {'keys': j['keys'], 'sorted': j['sorted'], 'duplicates': j['duplicates'], 'rmse': o['rmse'], 'max_error': o['max_error'],
                  'conflict_degree': o['conflict_degree'], 'pla_32': o['pla'].get('32'), 'pla_4096': o['pla'].get('4096'), 'path': path}
    return out

def chk_hardness(hard):
    t5 = 'hardness metrics reproduce the AIDB paper (history, libio PLA-32/PLA-4096)'
    t6 = 'hardness metrics are identical to the Mac run (same data, same code): PLA and CD exact, RMSE/ME within 1e-6'
    if hard is None:
        add('MET-5', 'gate', t5, 'UNKNOWN', 'skipped (--hardness none)'); return add('MET-6', 'gate', t6, 'UNKNOWN', 'skipped (--hardness none)')
    errs = ['%s: %s' % (d, h['error']) for d, h in hard.items() if 'error' in h]
    ok5, bad5 = [], []
    for d, (p32, p4k) in PAPER_PLA.items():
        h = hard.get(d)
        if not h or 'error' in h: continue
        (ok5 if (h['pla_32'], h['pla_4096']) == (p32, p4k) else bad5).append('%s %s/%s (paper %s/%s)' % (d, fmt(h['pla_32']), fmt(h['pla_4096']), fmt(p32), fmt(p4k)))
    add('MET-5', 'gate', t5, 'FAIL' if bad5 else ('PASS' if ok5 else 'UNKNOWN'), '; '.join(ok5 + bad5) or 'history/libio not computed', errs)
    try: mac = json.load(open(MAC_HARDNESS))
    except (OSError, ValueError): return add('MET-6', 'gate', t6, 'UNKNOWN', 'missing ' + MAC_HARDNESS)
    ok6, warn6, bad6 = [], [], []
    for d, h in sorted(hard.items()):
        m = (mac.get(d) or {}).get('full')
        if 'error' in h or not m: continue
        exact = all(h[k] == m[k] for k in ('pla_32', 'pla_4096', 'conflict_degree')) and h['keys'] == FULL_ROWS and h['duplicates'] == 0
        rel = max(abs(h[k] - m[k]) / max(1e-300, abs(m[k])) for k in ('rmse', 'max_error'))
        line = '%s: PLA %s/%s CD %s, RMSE rel diff %.1e' % (d, fmt(int(h['pla_32'])), fmt(int(h['pla_4096'])), int(h['conflict_degree']), rel)
        (ok6 if exact and rel <= 1e-6 else (warn6 if exact and rel <= 1e-3 else bad6)).append(line)
    add('MET-6', 'gate', t6, 'FAIL' if bad6 else ('WARN' if warn6 else ('PASS' if ok6 else 'UNKNOWN')), '; '.join(ok6 + warn6 + bad6) or 'nothing computed', errs)

def plateau(path):   # steady-state RSS (bytes) of a pass, as in rss_ba.sh
    v = [int(l.split()[1]) * 1024 for l in open(path) if l.strip()]; n, start = len(v), 0
    for i in range(n - 1, 0, -1):
        if v[i] < 0.90 * v[i - 1] and n - i >= 12: start = i; break
    seg = v[start:]; hi = max(seg)
    while seg and seg[-1] < 0.85 * hi: seg.pop()
    return max(seg)

def chk_rss(rss_dir):
    title = 'accounted bytes match real memory (steady-state RSS minus harness copies) within 3% (E5, H4)'
    files = sorted(glob.glob(os.path.join(rss_dir or '', '*__B.json'))) if rss_dir else []
    if not files: return add('MET-7', 'gate', title, 'UNKNOWN', 'no RSS runs in %s (python3 rss_ba.py)' % rss_dir)
    ev, bad, real = [], [], {}
    for jf in sorted(glob.glob(os.path.join(rss_dir, '*__*.json'))):
        name = os.path.basename(jf)[:-5]; rf = os.path.join(rss_dir, name + '.rss')
        if not os.path.exists(rf): continue
        j = json.load(open(jf)); rows = j['initial_rows']
        harness = 16 * rows + 32 * j['operations'] + 8 * j.get('warmup_reads', 0)    # w.initial + trace + warm-up keys
        first = [int(l.split()[1]) * 1024 for l in open(rf) if l.strip()][:1]
        if not first: continue
        real[name] = (plateau(rf) - first[0] - harness, j['memory_before']['accounted_bytes'], rows)   # first sample = process before loading
    warn, small = [], False
    for name, (r_, acc, rows) in sorted(real.items()):
        err = (r_ - acc) / acc
        ev.append('%s real %.3f vs accounted %.3f B/key (%+.1f%%)' % (name, r_ / rows, acc / rows, 100 * err))
        small |= rows != FULL_ROWS
        if name.endswith('__B'): (bad if abs(err) > 0.06 else warn if abs(err) > 0.03 else []).append(name)
        elif abs(err) > 0.03: warn.append(name)
    detail = []
    for name in sorted(real):
        c = name[:-3] + '__C'
        if name.endswith('__B') and c in real:
            (rb, ab, n), (rc_, ac, _) = real[name], real[c]
            ev.append('%s: C - B real %+.3f vs accounted %+.3f B/key (PROTOCOL H4 predicted +1.00 +- 0.02)' % (name[:-3], (rc_ - rb) / n, (ac - ab) / n))
    if warn: detail.append('accounted bytes understate real memory for %s: the accounting omits virtual_features capacity and allocator '
                           'retention (CODE_DISSECTION.md), so report memory of those cells from RSS' % ', '.join(warn))
    if small:   # tens of MB of fixed process overhead swamp a small index; the method is calibrated at 200M (memory_audit.md)
        return add('MET-7', 'gate', title, 'WARN', '; '.join(ev) + '; reduced-scale run: not meaningful below 200M keys')
    add('MET-7', 'gate', title, 'FAIL' if bad else ('WARN' if warn else 'PASS'), '; '.join(ev), detail)

# ---------------------------------------------------------------- GATE: isolation (PROTOCOL K-checks)
def pairs(prim, a, b):
    return [(d, prim[(d, a)], prim[(d, b)]) for d in DATASETS if (d, a) in prim and (d, b) in prim]

def chk_isolation(prim):
    t1 = 'key bytes are identical in B, Bb, N, Nf, Cr, C, NC (blocks are encoded before any model; K1)'
    rows, bad = 0, []
    for d in DATASETS:
        kb = dict((c, prim[(d, c)]['memory_before']['key_bytes']) for c in ('B', 'Bb', 'N', 'Nf', 'Cr', 'C', 'NC') if (d, c) in prim)
        if len(kb) < 2: continue
        rows += 1
        if len(set(kb.values())) > 1: bad.append('%s: %s' % (d, kb))
    ex = [(d, prim[(d, 'B')]['memory_before']['key_bytes'] / prim[(d, 'B')]['initial_rows']) for d in DATASETS if (d, 'B') in prim][:3]
    add('ISO-1', 'gate', t1, 'FAIL' if bad else ('PASS' if rows else 'UNKNOWN'),
        '%d datasets; key B/key %s' % (rows, ', '.join('%s %.4f' % e for e in ex)) if rows else 'needs two cells on one trace', bad)

    t2 = 'the flow adds nothing to accounted memory (N, Nf = B) and is reported as flow_bytes (K2)'
    ps, bad = pairs(prim, 'N', 'B') + pairs(prim, 'Nf', 'B'), []
    for d, x, b in ps:
        if x['memory_before']['accounted_bytes'] != b['memory_before']['accounted_bytes'] or not L(x).get('flow_bytes'):
            bad.append('%s/%s: %s vs B %s, flow_bytes %s' % (d, x['_cell'], fmt(x['memory_before']['accounted_bytes']), fmt(b['memory_before']['accounted_bytes']), L(x).get('flow_bytes')))
    add('ISO-2', 'gate', t2, 'FAIL' if bad else ('PASS' if ps else 'UNKNOWN'), '%d pairs' % len(ps) if ps else 'needs N/Nf and B', bad)

    t3 = 'CSV levels are independent: C and Cr share the root exactly; Cr changes nothing below the root'
    keys = ('root_model', 'root_flow', 'root_vp', 'root_virtual', 'root_probes_binary', 'root_probes_raw', 'root_probes_vp_raw')
    ps, bad = pairs(prim, 'C', 'Cr'), []
    for d, c, cr in ps:
        diff = [k for k in keys if L(c).get(k) != L(cr).get(k)]
        if diff or c['_op']['root'] != cr['_op']['root']: bad.append('%s: C vs Cr root differs (%s; root/op %.4f vs %.4f)' % (d, ','.join(diff), c['_op']['root'], cr['_op']['root']))
    ps2 = pairs(prim, 'Cr', 'B')
    for d, cr, b in ps2:
        if any(cr['_op'][k] != b['_op'][k] for k in ('coord', 'fence', 'key_at')): bad.append('%s: Cr changed region counters vs B' % d)
    n = len(ps) + len(ps2)
    add('ISO-3', 'gate', t3, 'FAIL' if bad else ('PASS' if n else 'UNKNOWN'), '%d C/Cr and %d Cr/B pairs' % (len(ps), len(ps2)) if n else 'needs B, Cr and C', bad)

    t4 = 'binary in-region routing (Bb) changes only the block search: root and in-block counters equal B'
    ps, bad = pairs(prim, 'Bb', 'B'), []
    for d, bb, b in ps:
        if bb['_op']['root'] != b['_op']['root'] or bb['_op']['key_at'] != b['_op']['key_at'] or bb['_op']['coord'] != 0: bad.append('%s: Bb root %.4f key_at %.4f coord %.4f vs B %.4f %.4f' % (d, bb['_op']['root'], bb['_op']['key_at'], bb['_op']['coord'], b['_op']['root'], b['_op']['key_at']))
    add('ISO-4', 'gate', t4, 'FAIL' if bad else ('PASS' if ps else 'UNKNOWN'), '%d pairs' % len(ps) if ps else 'needs Bb (run_ba.py plan validity)', bad)

    t5 = 'flow evaluations happen only where a flow is configured; forced flow (Nf) evaluates it on every lookup'
    bad, n = [], 0
    for (d, c), r in prim.items():
        n += 1; tr = r['_op']['transform']
        if c not in FLOW_CELLS and tr: bad.append('%s/%s: %.3f transforms/lookup without a flow' % (d, c, tr))
        if c == 'Nf' and tr < 1.0 - 1e-9: bad.append('%s/Nf: %.3f transforms/lookup (forced flow should give >= 1)' % (d, tr))
        if c == 'N' and (tr > 0) != (L(r).get('flow_regions', 0) > 0 or bool(L(r).get('root_flow'))): bad.append('%s/N: transforms %.3f but flow_regions %s' % (d, tr, L(r).get('flow_regions')))
    add('ISO-5', 'gate', t5, 'FAIL' if bad else ('PASS' if n else 'UNKNOWN'), '%d runs' % n, bad)

def chk_determinism(recs):
    title = 'repeat runs of one configuration are identical: memory, learnability (excl. *_ns) and counters (6.9-3/4)'
    groups = {}
    for r in recs:
        if 'memory_before' in r: groups.setdefault((tkey(r), r['_cell']) + tuple(str(r.get(k)) for k in CONFIG_KEYS), []).append(r)
    multi = [g for g in groups.values() if len(g) > 1]
    if not multi: return add('ISO-6', 'gate', title, 'UNKNOWN', 'no configuration ran twice (verify and D layers of one cell do)')
    strip = lambda l: dict((k, v) for k, v in (l or {}).items() if not k.endswith('_ns'))
    cnt = lambda r: dict((k, v) for k, v in (r.get('work_counters') or {}).items() if not k.startswith('cache_lines') and k != 'lines_overflow')
    bad = []
    for g in multi:
        r0 = g[0]
        for r in g[1:]:
            if r['memory_before'] != r0['memory_before'] or strip(L(r)) != strip(L(r0)): bad.append('%s: memory/learnability differ between %s and %s' % (tag(r), r0['_layer'], r['_layer']))
            if r.get('instrumented') and r0.get('instrumented') and cnt(r) != cnt(r0): bad.append('%s: counters differ between repeats' % tag(r))
    add('ISO-6', 'gate', title, 'FAIL' if bad else 'PASS', '%d repeated configurations' % len(multi), bad)

# ---------------------------------------------------------------- GATE: fidelity of the implementations
def chk_fidelity(prim):
    t1 = "NFL-style switch: forced flow (Nf) is used in every region; N adopts it where tail conflicts fall >= 10%"
    ev, bad = [], []
    for (d, c), r in sorted(prim.items()):
        l = L(r)
        if c == 'Nf' and l.get('flow_regions') != l.get('regions'): bad.append('%s/Nf: flow in %s of %s regions' % (d, l.get('flow_regions'), l.get('regions')))
        if c == 'N' and l.get('regions'): ev.append('%s %.1f%%' % (d, 100.0 * l.get('flow_regions', 0) / l['regions']))
    n = sum(1 for (d, c) in prim if c in ('N', 'Nf'))
    add('FID-1', 'gate', t1, 'FAIL' if bad else ('PASS' if n else 'UNKNOWN'), ('N adopts the flow in: ' + ', '.join(ev)) if ev else ('%d runs' % n if n else 'needs N/Nf'), bad)

    t2 = 'CSV Algorithm 1 runs as specified: SSE never rises, points stay within floor(alpha x region keys) per region'
    ev, bad, n = [], [], 0
    for (d, c), r in sorted(prim.items()):
        if c not in ('C', 'NC'): continue
        n += 1; l = L(r); va, rk = float(r['virtual_alpha']), r['region_keys']
        R, keys, vp = l.get('regions', 0), l.get('keys', 0), l.get('virtual_points', 0)
        budget = (R - 1) * int(va * rk) + int(va * (keys - rk * (R - 1))) if R else 0
        if not 0 < vp <= budget: bad.append('%s/%s: %s points, budget %s' % (d, c, fmt(vp), fmt(budget)))
        if l.get('rank_sse_after', 0) > l.get('rank_sse_before', 0): bad.append('%s/%s: SSE rose' % (d, c))
        if c == 'C': ev.append('%s %s/%s points, SSE x%.2f' % (d, fmt(vp), fmt(budget), l['rank_sse_after'] / max(1e-300, l['rank_sse_before'])))
    add('FID-2', 'gate', t2, 'FAIL' if bad else ('PASS' if n else 'UNKNOWN'), '; '.join(ev[:3]) + (' ...' if len(ev) > 3 else '') if ev else ('%d runs' % n if n else 'needs C/NC'), bad)

    t3 = 'CSV root fences stay within floor(root_alpha x regions) and are kept only if they beat the other root candidates'
    ev, bad, n = [], [], 0
    for (d, c), r in sorted(prim.items()):
        if c not in ('Cr', 'C'): continue
        n += 1; l = L(r); cap = int(float(r['root_alpha']) * l.get('regions', 0))
        if l.get('root_vp'):
            if not 0 < l.get('root_virtual', 0) <= cap: bad.append('%s/%s: %s fences, budget %s' % (d, c, l.get('root_virtual'), cap))
            if c == 'Cr': ev.append('%s adopted (%s fences)' % (d, fmt(l['root_virtual'])))
        elif c == 'Cr': ev.append('%s not adopted' % d)
    add('FID-3', 'gate', t3, 'FAIL' if bad else ('PASS' if n else 'UNKNOWN'), '; '.join(ev) if ev else ('%d runs' % n if n else 'needs Cr/C'), bad)

# ---------------------------------------------------------------- GATE: host and build, protocol flags
def cmake_cache():
    out = {}
    for line in (rd(BUILD + '/CMakeCache.txt') or '').splitlines():
        m = re.match(r'^([A-Za-z0-9_]+):[A-Z]+=(.*)$', line)
        if m: out[m.group(1)] = m.group(2)
    return out

def chk_build():
    title = 'the binary is a Release (-O3 -DNDEBUG) build of THIS clone, newer than every source file, without sanitizers'
    c = cmake_cache()
    if not c or not os.path.exists(BENCH): return add('HOST-1', 'gate', title, 'FAIL', 'no build at %s (cmake -S experimental/scaleli -B build-fs -DCMAKE_BUILD_TYPE=Release)' % BUILD)
    bad, warn = [], []
    if c.get('CMAKE_BUILD_TYPE') != 'Release': bad.append('CMAKE_BUILD_TYPE=%s' % c.get('CMAKE_BUILD_TYPE'))
    flags = c.get('CMAKE_CXX_FLAGS_RELEASE', '')
    if '-O3' not in flags or '-DNDEBUG' not in flags: bad.append('release flags "%s"' % flags)
    if c.get('SCALELI_SANITIZE') == 'ON': bad.append('sanitizers on')
    if c.get('SCALELI_NATIVE') == 'ON': warn.append('-march=native: not comparable with the Mac binary')
    home = c.get('CMAKE_HOME_DIRECTORY', '')
    if home and os.path.realpath(home) != os.path.realpath(S): warn.append('built from %s, not this clone' % home)
    srcs = glob.glob(S + '/include/scaleli/*.hpp') + glob.glob(S + '/src/*.cpp') + [S + '/CMakeLists.txt']
    newest = max(srcs, key=os.path.getmtime)
    if os.path.getmtime(newest) > os.path.getmtime(BENCH): bad.append('%s is newer than the binary: rebuild' % os.path.relpath(newest, REPO))
    cxx = c.get('CMAKE_CXX_COMPILER', 'c++'); rc, so, se = sh([cxx, '--version'], timeout=20)
    ver = (so or se).splitlines()[0] if (so or se) else cxx
    add('HOST-1', 'gate', title, 'FAIL' if bad else ('WARN' if warn else 'PASS'),
        '%s; %s; external baselines %s' % (ver, flags, c.get('SCALELI_EXTERNAL', '?')), bad + warn)

def chk_tests(skip):
    title = 'unit tests pass on this build (ctest)'
    if skip: return add('HOST-2', 'gate', title, 'UNKNOWN', 'skipped (--no-tests)')
    rc, so, se = sh(['ctest', '--output-on-failure'], timeout=900, cwd=BUILD)
    if rc is None: return add('HOST-2', 'gate', title, 'UNKNOWN', 'ctest not runnable: %s' % se[:120])
    summary = [l.strip() for l in so.splitlines() if 'tests passed' in l or 'tests failed' in l]
    add('HOST-2', 'gate', title, 'PASS' if rc == 0 else 'FAIL', summary[-1] if summary else 'rc=%s' % rc, [] if rc == 0 else so.splitlines()[-15:])

def host_info():
    i = {'platform': platform.platform(), 'python': platform.python_version(), 'cpus': os.cpu_count(), 'loadavg': list(os.getloadavg())}
    if sys.platform.startswith('linux'):
        cpu = [l.split(':', 1)[1].strip() for l in (rd('/proc/cpuinfo') or '').splitlines() if l.startswith('model name')]
        i['cpu'] = cpu[0] if cpu else '?'
        mem = dict((l.split(':')[0], int(l.split()[1]) * 1024) for l in (rd('/proc/meminfo') or '').splitlines() if l.split()[1:2] and l.split()[1].isdigit())
        i['mem_total'], i['mem_available'] = mem.get('MemTotal'), mem.get('MemAvailable')
        i['governors'] = sorted(set(rd(f) for f in glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor')) - {None})
        nt, boost = rd('/sys/devices/system/cpu/intel_pstate/no_turbo'), rd('/sys/devices/system/cpu/cpufreq/boost')
        i['turbo'] = None if nt is None and boost is None else (nt == '0' if nt is not None else boost == '1')
        i['smt'] = rd('/sys/devices/system/cpu/smt/active')
        i['numa_nodes'] = len(glob.glob('/sys/devices/system/node/node[0-9]*'))
        i['users'] = sorted(set(l.split()[0] for l in sh(['who'], 10)[1].splitlines() if l.strip()))
        i['top'] = [l.strip() for l in sh(['ps', '-eo', 'pcpu,user,comm', '--sort=-pcpu'], 10)[1].splitlines()[1:6]]
    elif sys.platform == 'darwin':
        i['cpu'] = sh(['sysctl', '-n', 'machdep.cpu.brand_string'], 10)[1].strip()
        i['mem_total'] = int(sh(['sysctl', '-n', 'hw.memsize'], 10)[1].strip() or 0)
        i['power'] = [l.strip() for l in sh(['pmset', '-g'], 10)[1].splitlines() if 'powermode' in l or 'lowpowermode' in l]
    return i

def chk_host(info):
    t4 = 'enough RAM: >= 32 GB (verify pass ~20 GB; P layer 23-33 GB per PROTOCOL 4.2)'
    m = info.get('mem_total')
    if not m: add('HOST-4', 'gate', t4, 'UNKNOWN', 'memory size not readable')
    else: add('HOST-4', 'gate', t4, 'PASS' if m >= 32e9 else ('WARN' if m >= 24e9 else 'FAIL'), '%.1f GB total%s' % (m / 1e9, (', %.1f GB available' % (info['mem_available'] / 1e9)) if info.get('mem_available') else ''))
    t3 = 'host is quiet and frequency-stable (governor, turbo, load, other users); timing has no CPU pinning on Linux'
    warn, ev = [], [info.get('cpu', '?'), '%s CPUs' % info.get('cpus'), 'load %.2f' % info['loadavg'][0]]
    if sys.platform.startswith('linux'):
        g = info.get('governors') or []
        ev.append('governor %s' % ('/'.join(g) or '?'))
        if g and g != ['performance']: warn.append('set the performance governor (sudo cpupower frequency-set -g performance)')
        if info.get('turbo'): warn.append('turbo/boost is on: clock varies with temperature and co-runners')
        if info.get('numa_nodes', 0) > 1: warn.append('%d NUMA nodes: bind memory and CPU to one node (numactl -N 0 -m 0)' % info['numa_nodes'])
        if len(info.get('users', [])) > 1: warn.append('other users logged in: %s' % ', '.join(info['users']))
        warn.append('scaleli_bench sets no CPU affinity on Linux (--qos is macOS-only; pinning is patch P-linux)')
    if info['loadavg'][0] > 1.0: warn.append('load average %.2f: something else is running' % info['loadavg'][0])
    for p in info.get('power', []):
        if p.split()[-1] != '0': warn.append('macOS %s (Low Power Mode caps clocks)' % p)
    add('HOST-3', 'timing', t3, 'WARN' if warn else 'PASS', ', '.join(ev), warn + ['top: ' + t for t in info.get('top', [])])

def chk_protocol_flags(recs):
    title = 'deterministic (D-layer) runs used exactly the PROTOCOL 4.1/4.2 settings for their cell'
    rs = [r for r in recs if r['_layer'] == 'D']
    if not rs: return add('PROT-1', 'gate', title, 'UNKNOWN', 'no D-layer runs (run_ba.py plan validity)')
    bad, smoke = [], 0
    for r in rs:
        exp = dict(D_EXPECT, **CELL_EXPECT.get(r['_cell'], {}))
        if r['_cell'] == 'SV': exp.pop('build_threads', None)
        diff = ['%s=%s (want %s)' % (k, r.get(k), v) for k, v in sorted(exp.items()) if r.get(k) != v and not (isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(r.get(k), (int, float)) and abs(r.get(k) - v) < 1e-12)]
        if r['_cell'] in FLOW_CELLS and not r.get('flow_weights'): diff.append('no flow weights')
        if r.get('warmup_reads') != min(1000000, r.get('initial_rows', 0)): diff.append('warmup_reads=%s' % r.get('warmup_reads'))
        if r.get('dataset_scope') == 'prefix': smoke += 1
        if diff: bad.append('%s: %s' % (tag(r), '; '.join(diff)))
    status = 'FAIL' if bad else ('WARN' if smoke else 'PASS')
    add('PROT-1', 'gate', title, status, '%d D-layer runs%s' % (len(rs), (', %d with --limit (smoke)' % smoke) if smoke else ''), bad)

# ---------------------------------------------------------------- TIMING
def chk_timing(recs):
    e1 = [r for r in recs if r['_layer'] == 'E1' and r.get('throughput_ops_s')]
    t1 = 'A/A noise gate G1 (night 0, E1): one-sided 80% upper bound of the run-to-run SD <= 8%'
    yb = [math.log(r['throughput_ops_s']) for r in e1 if r['_cell'] == 'B']
    yb2 = [math.log(r['throughput_ops_s']) for r in e1 if r['_cell'] == 'B2']
    if len(yb) < 2 or len(yb2) < 2:
        add('TIM-1', 'timing', t1, 'UNKNOWN', 'needs >= 2 B and >= 2 B2 E1 runs (run_ba.py plan aa)')
    else:
        df = len(yb) + len(yb2) - 2
        ss = sum((y - statistics.mean(yb)) ** 2 for y in yb) + sum((y - statistics.mean(yb2)) ** 2 for y in yb2)
        sd = math.sqrt(ss / df); chi2 = df * (1 - 2.0 / (9 * df) + Z20 * math.sqrt(2.0 / (9 * df))) ** 3
        u80 = sd * math.sqrt(df / chi2)
        add('TIM-1', 'timing', t1, 'PASS' if u80 <= 0.08 else 'FAIL',
            'sigma %.2f%%, U80 %.2f%% (df %d); B2/B %.3f; n = %d + %d' % (100 * sd, 100 * u80, df, math.exp(statistics.mean(yb2) - statistics.mean(yb)), len(yb), len(yb2)))
    t2 = 'warm-up worked: the first chunk is no slower than the median of the rest (pooled within 3%; V4)'
    dev = []
    for r in e1:
        c = r.get('throughput_chunks_ops_s') or []
        if len(c) >= 3: dev.append(c[0] / statistics.median(c[1:]) - 1)
    if len(dev) < 2: add('TIM-2', 'timing', t2, 'UNKNOWN', 'needs E1 runs with chunk rates')
    else:
        m, se = statistics.mean(dev), statistics.stdev(dev) / math.sqrt(len(dev))
        status = 'FAIL' if abs(m) > 0.03 else ('PASS' if abs(m) + 1.96 * se <= 0.03 else 'WARN')
        add('TIM-2', 'timing', t2, status, 'first chunk %+.2f%% +- %.2f%% (95%%) over %d runs' % (100 * m, 196 * se, len(dev)))
    t4 = 'timed runs were not disturbed by other work on the host'
    if not e1: add('TIM-4', 'timing', t4, 'UNKNOWN', 'no timed runs')
    else:
        noisy = ['%s rep %s: other CPU %.0f%% -> %.0f%%, load %.1f' % (tag(r), (r.get('ba') or {}).get('rep'), (r.get('ba') or {}).get('other_cpu_start', 0), (r.get('ba') or {}).get('other_cpu_end', 0), ((r.get('ba') or {}).get('loadavg_start') or [0])[0])
                 for r in e1 if max((r.get('ba') or {}).get('other_cpu_start', 0), (r.get('ba') or {}).get('other_cpu_end', 0)) > 50]
        add('TIM-4', 'timing', t4, 'WARN' if noisy else 'PASS', '%d of %d timed runs saw > 50%% CPU used by other processes' % (len(noisy), len(e1)), noisy)

def probe_options():
    missing, present = [], []
    for o in PX_OPTIONS:
        rc, so, se = sh([BENCH, '--' + o, '1', '--n', '64', '--ops', '16', '--warmup', '0', '--chunks', '1', '--verify', '0', '--latency', '0', '--instrument', '0'], timeout=60)
        (missing if 'unknown option' in (se + so) else present).append(o)
    return missing, present

def chk_px(skip):
    title = 'the binary implements the X/P timing options of PROTOCOL 4.1 (needed before night 1, not night 0)'
    if skip or not os.path.exists(BENCH): return add('TIM-3', 'timing', title, 'UNKNOWN', 'not probed')
    missing, present = probe_options()
    add('TIM-3', 'timing', title, 'FAIL' if missing else 'PASS',
        ('missing: --' + ' --'.join(missing) + ' (patches P1/P2/P6, PROTOCOL 8.1)') if missing else 'all present')

# ---------------------------------------------------------------- SCOPE
def chk_headroom(prim):
    t1 = 'region level: can a better model beat the no-model alternative? (perfect model vs binary routing over blocks)'
    ev, ok, no = [], [], []
    for d in DATASETS:
        B = prim.get((d, 'B'))
        if not B: continue
        o, bb = B['_op'], prim.get((d, 'Bb'))
        bpr = max(2, B['region_keys'] // B['block_keys'])
        perfect = o['coord'] + 2 - 1.0 / bpr                 # an exact block prediction costs 2 fence probes (1 for block 0)
        binary, src = (bb['_op']['fence'], 'measured') if bb else (o['coord'], 'est.')
        C, Cr = prim.get((d, 'C')), prim.get((d, 'Cr'))
        csv = ('; CSV moves the fence count by %+.2f' % (C['_op']['fence'] - Cr['_op']['fence'])) if C and Cr else ''
        ev.append('%s: learned %.2f (coord %.2f + fence %.2f), perfect >= %.2f, binary %.2f (%s)%s' % (d, o['coord'] + o['fence'], o['coord'], o['fence'], perfect, binary, src, csv))
        (ok if perfect < binary - 1e-9 else no).append(d)
    if not ev: add('HEAD-1', 'scope', t1, 'UNKNOWN', 'needs B (and Bb) runs')
    else:
        add('HEAD-1', 'scope', t1, 'FAIL' if not ok else ('PASS' if not no else 'WARN'),
            'no headroom on %d/%d datasets: even a perfect region model costs more comparisons than binary search over the blocks' % (len(no), len(ev)) if no else 'headroom on all datasets', ev)
        if no: CLAIMS.append('region level: NFL/CSV effects are NOT a valid improvement test on %s (a perfect model still loses to binary routing)' % ', '.join(no))
    t2 = 'root level: headroom between binary search over the region fences and a perfect root model'
    ev, null, adopted = [], [], []
    for d in DATASETS:
        B = prim.get((d, 'B'))
        if not B: continue
        l = L(B); R = l.get('regions', 0)
        if R < 2: continue
        binary, perfect = l.get('root_probes_binary') or math.log2(R), 2 - 1.0 / R
        Cr = prim.get((d, 'Cr'))
        learned = l.get('root_model') or (Cr and L(Cr).get('root_vp'))
        (adopted if learned else null).append(d)
        ev.append('%s: binary %.2f, perfect %.2f, B %.2f%s%s' % (d, binary, perfect, B['_op']['root'], (', Cr %.2f' % Cr['_op']['root']) if Cr else '', '' if learned else ' (no learned root adopted: structural null)'))
    if not ev: add('HEAD-2', 'scope', t2, 'UNKNOWN', 'needs B runs')
    else:
        add('HEAD-2', 'scope', t2, 'PASS' if adopted else 'FAIL', 'learned root adopted on %d/%d datasets' % (len(adopted), len(ev)), ev)
        if adopted: CLAIMS.append('root level: methods are testable on %s' % ', '.join(adopted))
        if null: CLAIMS.append('root level: structural null (binary root kept by every candidate) on %s' % ', '.join(null))
    t3 = 'share of each lookup that no model can change (block search + in-block search), in B'
    ev = ['%s: root %.2f + coord %.2f + fence %.2f + key_at %.2f = %.2f, fixed %.0f%%' % (d, o['root'], o['coord'], o['fence'], o['key_at'], o['total'], 100 * (o['coord'] + o['key_at']) / o['total'])
          for d, o in ((d, prim[(d, 'B')]['_op']) for d in DATASETS if (d, 'B') in prim)]
    add('HEAD-3', 'scope', t3, 'INFO' if ev else 'UNKNOWN', ev[0] if ev else 'needs B runs', ev[1:])

def chk_assumptions(prim):
    add('ASM-1', 'scope', "NFL's host assumption: the index stores and searches keys in transformed (z) order", 'FAIL',
        'violated by design: records stay in key order; the switch scores sorted z but the model is fit on key-ordered z (index.hpp:287,290,319; transform.hpp:14-16)')
    ev, root_ok = [], []
    for d in DATASETS:
        C = prim.get((d, 'C'))
        if not C: continue
        if L(C).get('root_vp'): root_ok.append(d)
        ev.append('%s: slot -> block costs %.2f comparisons' % (d, C['_op']['coord']))
    add('ASM-2', 'scope', "CSV's host assumption: a virtual point is an empty slot, so the model's slot is a physical position", 'FAIL',
        'regions: violated (dense blocks; the slot is translated by a binary search over 32 descriptors, index.hpp:121-134); root: satisfied by the O(1) slot table (index.hpp:362-365) where adopted: %s' % (', '.join(root_ok) or 'none yet'), ev)

def chk_baselines(prim, recs, skip_probe):
    with_b = [d for d in DATASETS if (d, 'B') in prim]
    sv = [d for d in with_b if (d, 'SV') in prim]
    add('BASE-1', 'scope', 'plain binary search (SV) ran on the same data and trace as every SCALE-LI cell',
        'PASS' if with_b and len(sv) == len(with_b) else ('UNKNOWN' if not with_b else 'WARN'), 'SV on %d/%d datasets' % (len(sv), len(with_b)))
    title = 'published learned indexes (ALEX, dynamic PGM) run through the same harness'
    ext = sorted(set(r.get('index') for r in recs if r.get('index') in ('alex', 'pgm')))
    avail = []
    if not skip_probe and os.path.exists(BENCH):
        for name in ('pgm', 'alex'):
            rc, so, se = sh([BENCH, '--index', name, '--n', '2000', '--ops', '200', '--warmup', '0', '--chunks', '1', '--verify', '0', '--latency', '0', '--instrument', '0'], timeout=60)
            if rc == 0: avail.append(name)
    status = 'PASS' if ext else ('WARN' if avail else 'FAIL')
    add('BASE-2', 'scope', title, status, 'runs: %s; built into the binary: %s%s' % (', '.join(ext) or 'none', ', '.join(avail) or 'none',
        '' if avail else ' (configure with -DSCALELI_EXTERNAL=ON after fetching ALEX and PGM-index into experimental/scaleli/external)'),
        ['ALEX adapter memory() reports metadata only (key/value bytes 0): not comparable on memory'] if 'alex' in ext + avail else [])
    add('BASE-3', 'scope', 'standard non-learned baselines beyond binary search (B+tree / ART, as in SOSD and GRE)', 'FAIL',
        'none in the harness: only sorted_vector and std::map')

def chk_flow_fidelity():
    title = "NFL's transform reproduces the paper where the paper defines it (tail conflicts on the 2M samples; recorded on the Mac)"
    try: f = json.load(open(FLOWCHECK))
    except (OSError, ValueError): return add('FID-4', 'scope', title, 'UNKNOWN', 'missing ' + FLOWCHECK)
    ok = [d for d in DATASETS if d in f and f[d]['free'] <= 6 and f[d]['free'] <= f[d]['raw']]
    same = [d for d in DATASETS if d in f and f[d]['mono'] == f[d]['raw']]
    add('FID-4', 'scope', title, 'PASS' if len(ok) == len([d for d in DATASETS if d in f]) else 'FAIL',
        'free flow <= 6 tail conflicts on %d/%d (e.g. osm %s -> %s); monotone flow leaves them unchanged on %d/%d' % (len(ok), len(f), f.get('osm', {}).get('raw'), f.get('osm', {}).get('free'), len(same), len(f)))

# ---------------------------------------------------------------- report
KINDS = [('gate', 'GATE: the setup is internally consistent (needed before any result counts)'),
         ('timing', 'TIMING: extra conditions before a timed number counts'),
         ('scope', 'SCOPE: what this setup can and cannot show')]

def verdict(kind):
    st = [c['status'] for c in CHECKS if c['kind'] == kind]
    if kind == 'scope': return 'see claims'
    if 'FAIL' in st: return 'FAIL'
    if 'UNKNOWN' in st: return 'INCOMPLETE'
    return 'PASS with warnings' if 'WARN' in st else 'PASS'

def render_text(info, inputs):
    w = textwrap.TextWrapper(width=118, initial_indent=' ' * 17, subsequent_indent=' ' * 17)
    out = ['SCALE-LI setup validation  (%s; %s)' % (info.get('cpu', '?'), info['platform']),
           'inputs: %s' % inputs]
    if any('SMOKE' in c['evidence'] for c in CHECKS): out.append('*** SMOKE MODE: runs used --limit; these are not the 200M-key setup ***')
    for kind, head in KINDS:
        out += ['', '%s  [%s]' % (head, verdict(kind))]
        for c in CHECKS:
            if c['kind'] != kind: continue
            out.append('  %-7s %-7s %s' % (c['id'], c['status'], c['title']))
            out += w.wrap(c['evidence'])
            for d in c['detail'][:12]: out += w.wrap('- ' + d)
            if len(c['detail']) > 12: out.append(' ' * 17 + '- ... %d more' % (len(c['detail']) - 12))
    out += ['', 'CLAIMS this setup supports / excludes:'] + ['  - ' + c for c in CLAIMS] if CLAIMS else []
    out += ['', 'VERDICT: gate %s | timing %s' % (verdict('gate'), verdict('timing'))]
    return '\n'.join(out)

def render_md(info, inputs):
    esc = lambda s: str(s).replace('|', '\\|')
    out = ['# SCALE-LI setup validation', '', 'Host: %s, %s. Inputs: %s.' % (info.get('cpu', '?'), info['platform'], inputs), '']
    for kind, head in KINDS:
        out += ['## %s: %s' % (head, verdict(kind)), '', '| ID | Status | Check | Evidence |', '|---|---|---|---|']
        for c in CHECKS:
            if c['kind'] == kind:
                ev = esc(c['evidence']) + ''.join('<br>- ' + esc(d) for d in c['detail'][:12])
                out.append('| %s | **%s** | %s | %s |' % (c['id'], c['status'], esc(c['title']), ev))
        out.append('')
    if CLAIMS: out += ['## Claims', ''] + ['- ' + c for c in CLAIMS] + ['']
    return '\n'.join(out)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='*', default=None, help='run_ba.py output files (default: validity/verify/aa.jsonl next to this script)')
    ap.add_argument('--hardness', default='history,libio', help="datasets to recompute with scaleli_hardness, or 'none'")
    ap.add_argument('--rss', default=os.path.join(HERE, 'rss'), help='rss_ba.py output directory')
    ap.add_argument('--no-hash', action='store_true', help='skip the sha256 of the ten GRE files')
    ap.add_argument('--no-tests', action='store_true', help='skip ctest')
    ap.add_argument('--no-probe', action='store_true', help='do not execute the benchmark binary to probe options/baselines')
    ap.add_argument('--out', help='write OUT.json and OUT.md as well')
    a = ap.parse_args()
    runs = a.runs if a.runs is not None else [p for p in (os.path.join(HERE, n) for n in ('validity.jsonl', 'verify.jsonl', 'aa.jsonl')) if os.path.exists(p)]
    recs, bad = load(runs)
    prim = primaries(recs)
    info = host_info()
    inputs = '%d run records from %s, %d failed' % (len(recs), ', '.join(os.path.basename(p) for p in runs) or 'no files', len(bad))
    chk_data_hash(a.no_hash); chk_rows(recs)
    chk_failed(bad); chk_verify(recs, bad); chk_checksums(recs); chk_counted(recs)
    chk_prefault(recs); chk_values(recs); chk_csv_accounting(prim); chk_counter_sanity(recs)
    chk_hardness(None if a.hardness == 'none' else run_hardness([d for d in a.hardness.split(',') if d]))
    chk_rss(a.rss)
    chk_isolation(prim); chk_determinism(recs); chk_fidelity(prim)
    chk_build(); chk_tests(a.no_tests); chk_host(info); chk_protocol_flags(recs)
    chk_timing(recs); chk_px(a.no_probe)
    chk_headroom(prim); chk_assumptions(prim); chk_baselines(prim, recs, a.no_probe); chk_flow_fidelity()
    order = dict((k, i) for i, (k, _) in enumerate(KINDS))
    CHECKS.sort(key=lambda c: order[c['kind']])
    text = render_text(info, inputs)
    print(text)
    if a.out:
        with open(a.out + '.json', 'w') as f:
            json.dump({'host': info, 'inputs': inputs, 'verdicts': dict((k, verdict(k)) for k, _ in KINDS), 'claims': CLAIMS, 'checks': CHECKS}, f, indent=1)
        with open(a.out + '.md', 'w') as f: f.write(render_md(info, inputs))
    g = verdict('gate')
    sys.exit(1 if g == 'FAIL' else (2 if g == 'INCOMPLETE' else 0))

if __name__ == '__main__':
    main()
