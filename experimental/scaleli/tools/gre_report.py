#!/usr/bin/env python3
"""Markdown report for a gre_run.sh OUTDIR: correctness of every run, throughput, index memory and build time.

    python3 gre_report.py OUTDIR [--ref sortedarray|btree|...] [--splice-json DIR] > OUTDIR/report.md

Reads OUTDIR/runs.tsv (the last row per repeat, dataset and index counts), OUTDIR/config.txt, OUTDIR/provenance.txt
and each log. Logs from the unpatched GRE (no gre_lite_patch, rss_* or build_ns lines) are fine; those columns stay
empty. Throughput ratios are geometric: exp(mean ln T_index - mean ln T_ref), with a 95% CI from the pooled
within-cell SD of ln throughput over all cells of the dataset (df = runs - cells). That assumes equal variance in
every cell; each cell's own SD is printed, and a cell whose SD exceeds twice the pooled one is flagged, since its CI
is then too narrow. Only runs that pass every correctness check enter the statistics. Exit status 1 if any run
fails a check, 2 if OUTDIR has no runs.
SPLICE-H runs (splice, splice_thp, trace_splice) add a per-dataset SPLICE check: index RSS against the index's own
splice_total_bytes (within 3%, and 18-25 B/key unless the cell is compact), predicted E[D], the AnonHugePages share of
splice_thp's arena, and layout hashes that must agree between repeats (and, with --splice-json DIR, with
splice_count's DIR/<ds>.json). Its findings go to Warnings; they never change the exit status. Without SPLICE runs
the report is unchanged.
Standard library only; runs on Python 3.6.
"""
import argparse
import json
import math
import os
import re
import sys

# Two-sided 95% Student t; a df between entries uses the next smaller one (wider interval).
T95 = [(1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571), (6, 2.447), (7, 2.365), (8, 2.306),
       (9, 2.262), (10, 2.228), (11, 2.201), (12, 2.179), (13, 2.160), (14, 2.145), (15, 2.131), (16, 2.120),
       (17, 2.110), (18, 2.101), (19, 2.093), (20, 2.086), (21, 2.080), (22, 2.074), (23, 2.069), (24, 2.064),
       (25, 2.060), (26, 2.056), (27, 2.052), (28, 2.048), (29, 2.045), (30, 2.042), (40, 2.021), (60, 2.000),
       (120, 1.980)]
SORTED_ARRAY_BPK = 16.0   # 8-B key + 8-B payload per slot
MEM_TOL = 0.02
SD_FLAG = 2.0             # a cell's own SD of ln throughput above this multiple of the pooled SD is flagged

CONTRACT = ['rss_before_build_bytes', 'rss_after_build_bytes', 'index_rss_bytes', 'build_ns', 'pinned_core',
            'warmup_success_read', 'warmup_ns', 'rss_after_run_bytes', 'success_read',
            'rss_after_build_unpurged_bytes', 'rss_peak_bytes']
RE_CONTRACT = re.compile(r'^(%s): (-?\d+)\s*$' % '|'.join(CONTRACT))
RE_PATCH = re.compile(r'^gre_lite_patch: warmup_num=(\d+) pin_core=(-?\d+)\s*$')
RE_TABLE = re.compile(r'^Table size is (\d+), Init table size is (\d+)\s*$')
RE_TPUT = re.compile(r'^Throughput = (\d+)\s*$')
RE_MEM = re.compile(r'^Memory: (-?\d+)\s*$')
# SPLICE-H facade lines (integrations/gre_splice/README.md) and the trace_* chunk timer
SPLICE_IDX = ('splice', 'splice_thp', 'trace_splice')
SPLICE_BPK = (18.0, 25.0)  # fast cells: 22 B/key cap plus allocator and page slack; compact cells are exempt
SPLICE_TOL = 0.03          # index RSS against the self-reported splice_total_bytes
RE_SPLICE = re.compile(r'^(splice_total_bytes|splice_arena_bytes|splice_arena_anon_huge_bytes|splice_predicted_ed_milli|'
                       r'splice_bulk_load_ns|splice_hash_ns|splice_keys|trace_chunks_n): (-?\d+)\s*$')
RE_SPLICE_HEX = re.compile(r'^(splice_layout_hash|splice_full_hash): (0x[0-9a-fA-F]+)\s*$')
RE_SPLICE_TEXT = re.compile(r'^(splice_args_effective|splice_index): (.*?)\s*$')
RE_FLAG = re.compile(r'^(keys_file|read|insert|operations_num|table_size|init_table_ratio|index|thread_num|seed) = (.*?)\s*$')


def t95(df):
    t = None
    for d, v in T95:
        if d <= df:
            t = v
    return t


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def parse_log(path):
    r = {'throughputs': [], 'flags': {}}
    try:
        f = open(path, encoding='utf-8', errors='replace')
    except (IOError, OSError):
        return None
    with f:
        for line in f:
            line = line.rstrip('\n')
            m = RE_TPUT.match(line)
            if m:
                r['throughputs'].append(int(m.group(1)))
                continue
            m = RE_CONTRACT.match(line)
            if m:
                r[m.group(1)] = int(m.group(2))
                continue
            m = RE_MEM.match(line)
            if m:
                r['memory'] = int(m.group(1))
                continue
            m = RE_PATCH.match(line)
            if m:
                r['warmup_num'], r['pin_core'] = int(m.group(1)), int(m.group(2))
                continue
            m = RE_TABLE.match(line)
            if m:
                r['table_size'], r['init_table_size'] = int(m.group(1)), int(m.group(2))
                continue
            m = RE_FLAG.match(line)
            if m and m.group(1) not in r['flags']:
                r['flags'][m.group(1)] = m.group(2)
                continue
            if line.startswith('splice_') or line.startswith('trace_'):
                m = RE_SPLICE.match(line)
                if m:
                    r[m.group(1)] = int(m.group(2))
                    continue
                m = RE_SPLICE_HEX.match(line) or RE_SPLICE_TEXT.match(line)
                if m:
                    r[m.group(1)] = m.group(2)
                    continue
            if 'splice_error' in line:
                r['splice_error'] = line.strip()
    if r['throughputs']:
        r['throughput'] = r['throughputs'][-1]
    return r


def read_kv(path):
    kv = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if '=' in line:
                    k, v = line.rstrip('\n').split('=', 1)
                    kv[k] = v
    return kv


def read_sections(path):
    secs = []   # [(name, [lines])], repeated names kept (resumed runs)
    if os.path.exists(path):
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('## '):
                    secs.append((line[3:].strip(), []))
                elif secs:
                    secs[-1][1].append(line)
    return secs


def as_int(s, default=None):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def as_float(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def load_runs(outdir):
    path = os.path.join(outdir, 'runs.tsv')
    if not os.path.exists(path):
        return None
    last, order, retried = {}, [], 0
    with open(path, encoding='utf-8', errors='replace') as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            if not line.strip():
                continue
            row = dict(zip(header, line.rstrip('\n').split('\t')))
            key = (row.get('repeat'), row.get('dataset'), row.get('index'))
            if key not in last:
                order.append(key)
            else:
                retried += 1
            last[key] = row
    return [last[k] for k in order], retried


def check_run(row, log, cfg):
    """Problems of one run as (check, message) pairs; empty means it counts."""
    p = []
    rc = as_int(row.get('exit_code'))
    if rc != 0:
        p.append(('exit', 'exit %s' % row.get('exit_code')))
    if log is None:
        p.append(('throughput', 'log missing'))
        return p
    if not log['throughputs']:
        p.append(('throughput', 'no Throughput line'))
    elif len(log['throughputs']) > 1:
        p.append(('throughput', '%d results in one log' % len(log['throughputs'])))
    idx = log['flags'].get('index')
    if idx is not None and idx != row.get('index'):
        p.append(('other', 'log is for index %s' % idx))
    if 'init_table_size' not in log:
        p.append(('other', 'no Table size line'))
    ops = as_int(log['flags'].get('operations_num'))
    read_only = as_float(log['flags'].get('read')) == 1.0 and as_float(log['flags'].get('insert'), 0.0) == 0.0
    if read_only:
        sr = log.get('success_read')
        if sr is None or ops is None or sr != ops:
            p.append(('success_read', 'success_read %s != operations_num %s' % (sr, ops)))
    want_w = as_int(cfg.get('warmup'), 0)
    want_pin = as_int(cfg.get('pin'), -1)
    w = log.get('warmup_num')
    if w is None:
        if want_w > 0 or want_pin >= 0:
            p.append(('warmup', 'no gre_lite_patch line (unpatched binary?)'))
        w = 0
    elif w != want_w and 'warmup' in cfg:
        p.append(('warmup', 'warmup_num %d, requested %d' % (w, want_w)))
    if w > 0:
        if log.get('warmup_success_read') != w:
            p.append(('warmup', 'warmup_success_read %s != warmup_num %d' % (log.get('warmup_success_read'), w)))
        if log.get('warmup_ns') is None:
            p.append(('warmup', 'no warmup_ns'))
    if want_pin >= 0 and log.get('pinned_core') != want_pin:
        p.append(('pin', 'pinning to core %d not confirmed' % want_pin))
    if row.get('index') in SPLICE_IDX:
        if 'splice_index' not in log:
            p.append(('other', 'no splice_index line'))
        if 'splice_error' in log:
            p.append(('other', log['splice_error']))
    if (row.get('index') or '').startswith('trace_') and 'trace_chunks_n' not in log:
        p.append(('other', 'no trace_chunks_n line'))
    return p


def fmt(x, nd=3):
    return '' if x is None else ('%.*f' % (nd, x))


def provenance_md(secs, cfg, out):
    get = {}
    for name, lines in secs:
        get.setdefault(name, lines)
    def first(name, pred=lambda l: True):
        for l in get.get(name, []):
            if l.strip() and l.strip() != '(unavailable)' and pred(l):
                return l.strip()
        return ''
    cpu = first('lscpu', lambda l: l.startswith('Model name'))
    cpu = cpu.split(':', 1)[1].strip() if ':' in cpu else ''
    gov = '; '.join(' '.join(l.split()) for l in get.get('governors', []) if l.strip() and l != '(unavailable)')
    mem = first('free', lambda l: l.startswith('Mem:'))
    build = [l for l in get.get('gre_lite_build', []) if l.strip()]
    libs = re.compile(r'jemalloc|tbb|libstdc\+\+')
    # ldd of the binary that ran (gre_run.sh), else of the build (gre_lite_build.txt)
    ldd = [l.strip() for l in get.get('ldd', []) if libs.search(l)]
    if not ldd:
        ldd = [l[len('ldd: '):] for l in build if l.startswith('ldd: ')]
    thp = '; '.join(l.strip() for l in get.get('thp', []) if l.strip() and l.strip() != '(unavailable)')
    rows = [('date (UTC)', first('date')), ('host', first('hostname')), ('kernel', first('uname')),
            ('CPU', cpu), ('cores (nproc)', first('nproc')), ('governors', gov), ('free -g', ' '.join(mem.split())),
            ('load average at start', first('loadavg')), ('SPLICE', first('splice')),
            ('GRE build', build[0] if build else ''),
            ('ALEX', next((l for l in build if l.startswith('ALEX_USE_LZCNT')), '')),
            ('compiler', next((l for l in build if 'gcc' in l.lower() or 'clang' in l.lower()), '')),
            ('-march', ' '.join(next((l for l in build if l.strip().startswith('-march=')), '').split())),
            # the purge behind index RSS needs jemalloc as the process's malloc
            ('runtime libraries (ldd)', '; '.join(ldd)),
            ('transparent huge pages', thp), ('numa_balancing', first('numa_balancing')),
            ('jemalloc options', first('malloc_conf')),
            ('binary sha256', first('binary'))]
    resumed = [lines for name, lines in secs if name == 'resumed']
    if resumed:
        rows.append(('resumed', ', '.join(l[0].strip() for l in resumed if l)))
    w, pin = as_int(cfg.get('warmup'), 0), as_int(cfg.get('pin'), -1)
    cond = 'with warm-up' if w > 0 else 'GRE as published' if pin < 0 else 'pinned, no warm-up'
    keys = ['datasets', 'indexes', 'repeats', 'ops', 'warmup', 'pin', 'init_ratio', 'table_size', 'read', 'insert', 'extra']
    rows.append(('condition', '%s (%s)' % (cond, ', '.join('%s=%s' % (k, cfg[k]) for k in keys if k in cfg))))
    out.append('## Provenance\n')
    out.append('| | |\n|---|---|')
    for k, v in rows:
        out.append('| %s | %s |' % (k, (v or '(not recorded)').replace('|', '\\|')))
    cmd = first('command')
    if cmd:
        out.append('\nCommand: `%s`' % cmd)
    out.append('')


def load_md(runs, out):
    """Load and other benchmarks at the start of each run (runs.tsv columns load1, busy)."""
    loads = sorted(x for x in (as_float(r.get('load1')) for r in runs) if x is not None)
    if loads:
        out.append('Load average (1 min) at the start of the runs: min %.2f, median %.2f, max %.2f.'
                   % (loads[0], loads[len(loads) // 2], loads[-1]))
    busy = [r for r in runs if as_int(r.get('busy'), 0) > 0]
    if busy:
        out.append('%d runs started while another benchmark was running: %s.' % (len(busy), ', '.join(
            'r%s %s %s' % (r['repeat'], r['dataset'], r['index']) for r in busy)))
    if loads or busy:
        out.append('')
    return busy


def splice_md(ds, cells, a, cfg, out, warn):
    """SPLICE check of one dataset: memory against the self-report, predicted E[D], THP share, layout hashes."""
    present = [ix for ix in SPLICE_IDX if ix in cells]
    if not present:
        return
    want = None   # splice_count's layout hash of the chosen cell (--splice-json)
    if a.splice_json:
        path = os.path.join(a.splice_json, ds + '.json')
        try:
            with open(path, encoding='utf-8') as f:
                j = json.load(f)
            arm = cfg.get('splice_arm', 'fast')
            if arm == 'compact':
                want = j.get('compact_layout_hash') or ((j.get('chosen') or {}).get('compact') or {}).get('layout_hash')
            else:
                want = j.get('layout_hash')
            if isinstance(want, int):
                want = '0x%016x' % want
            if not want:
                warn.append('%s: --splice-json: %s has no layout hash for the %s arm; the identity check was not made'
                            % (ds, path, arm))
        except (IOError, OSError, ValueError) as e:
            warn.append('%s: --splice-json: cannot read %s (%s)' % (ds, path, e))
    out.append('\nSPLICE check (index RSS against the index\'s own splice_total_bytes, tolerance %.0f%%; %g-%g B/key '
               'unless the cell is compact):\n' % (100 * SPLICE_TOL, SPLICE_BPK[0], SPLICE_BPK[1]))
    out.append('| index | runs | index RSS B/key | self-reported B/key | deviation | predicted E[D] (D) | '
               'bulk load s | hash s | AnonHugePages share | layout hash | verdict |')
    out.append('|---|---|---|---|---|---|---|---|---|---|---|')
    hashes = {}
    for ix in present:
        rs = [r['parsed'] for r in cells[ix]]
        def per_key(field):
            v = [p[field] / float(p['init_table_size']) for p in rs if field in p and p.get('init_table_size')]
            return mean(v)
        def avg(field, scale=1.0):
            v = [p[field] / scale for p in rs if field in p]
            return mean(v)
        rss, selfr = per_key('index_rss_bytes'), per_key('splice_total_bytes')
        compact = any('compact=1' in p.get('splice_args_effective', '').split() for p in rs)
        dev = rss / selfr - 1 if rss is not None and selfr else None
        problems = []
        if dev is None:
            problems.append('no index RSS or splice_total_bytes')
        elif abs(dev) > SPLICE_TOL:
            problems.append('index RSS %.3f B/key is %+.2f%% off its splice_total_bytes %.3f B/key'
                            % (rss, 100 * dev, selfr))
        if rss is not None and not compact and not SPLICE_BPK[0] <= rss <= SPLICE_BPK[1]:
            problems.append('index RSS %.3f B/key outside %g-%g B/key' % (rss, SPLICE_BPK[0], SPLICE_BPK[1]))
        huge = ''
        sh = [p['splice_arena_anon_huge_bytes'] / float(p['splice_arena_bytes']) for p in rs
              if p.get('splice_arena_anon_huge_bytes', -1) >= 0 and p.get('splice_arena_bytes')]
        if ix == 'splice_thp':
            huge = fmt(mean(sh), 3) if sh else 'not reported'
        elif sh and max(sh) > 0:
            # THP enabled=[always] gives the plain index huge pages too: this is then not a 4 KiB arm
            huge = fmt(mean(sh), 3)
            problems.append('arena has AnonHugePages (share %.3f) without MADV_HUGEPAGE: THP is [always], so this '
                            'run is not a 4 KiB-page arm' % max(sh))
        hs = sorted(set(p['splice_layout_hash'].lower() for p in rs if 'splice_layout_hash' in p))
        hashes[ix] = hs
        if len(hs) > 1:
            problems.append('layout hash differs between repeats (%s)' % ', '.join(hs))
        if want and hs and hs != [want.lower()]:
            problems.append('layout hash %s != splice_count %s/%s.json %s%s'
                            % (', '.join(hs), a.splice_json, ds, want.lower(),
                               ' (global SPLICE_ARGS was set: %s)' % cfg['splice_args'] if cfg.get('splice_args') else ''))
        ed = avg('splice_predicted_ed_milli', 1000.0)
        hcol = ', '.join(hs) or '-'
        if want and hs == [want.lower()]:
            hcol += ' (= splice_count)'
        out.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            ix + (' (compact)' if compact else ''), len(rs), fmt(rss, 3), fmt(selfr, 3),
            '' if dev is None else '%+.2f%%' % (100 * dev), fmt(ed, 3), fmt(avg('splice_bulk_load_ns', 1e9), 2),
            fmt(avg('splice_hash_ns', 1e9), 2), huge or 'n/a', hcol, 'WARN' if problems else 'OK'))
        for m in problems:
            warn.append('%s %s: SPLICE check: %s' % (ds, ix, m))
    # splice_thp differs from splice only by MADV_HUGEPAGE: the layout must be the same
    if hashes.get('splice') and hashes.get('splice_thp') and hashes['splice'] != hashes['splice_thp']:
        warn.append('%s: splice and splice_thp layout hashes differ (%s vs %s); they must build one layout'
                    % (ds, ', '.join(hashes['splice']), ', '.join(hashes['splice_thp'])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('outdir')
    ap.add_argument('--ref', help='reference index for throughput ratios (default sortedarray, else btree)')
    ap.add_argument('--splice-json', help='splice_count results directory (results/splice_h): compare layout hashes')
    a = ap.parse_args()
    runs, retried = load_runs(a.outdir) or ([], 0)
    if not runs:
        sys.stderr.write('gre_report.py: no runs in %s/runs.tsv\n' % a.outdir)
        return 2
    cfg = read_kv(os.path.join(a.outdir, 'config.txt'))
    secs = read_sections(os.path.join(a.outdir, 'provenance.txt'))

    def ordered(cfg_key, field):
        seen = [x for x in cfg.get(cfg_key, '').split(',') if x]
        for r in runs:
            if r[field] not in seen:
                seen.append(r[field])
        return [x for x in seen if any(r[field] == x for r in runs)]
    datasets, indexes = ordered('datasets', 'dataset'), ordered('indexes', 'index')
    ref = a.ref or ('sortedarray' if 'sortedarray' in indexes else 'btree' if 'btree' in indexes else indexes[0])

    for r in runs:
        r['parsed'] = parse_log(os.path.join(a.outdir, r.get('log', '')))
        r['problems'] = check_run(r, r['parsed'], cfg)
    nfail = sum(1 for r in runs if r['problems'])
    planned = None
    if cfg.get('repeats') and cfg.get('datasets') and cfg.get('indexes'):
        planned = as_int(cfg['repeats'], 0) * len(cfg['datasets'].split(',')) * len(cfg['indexes'].split(','))

    out = ['# GRE baselines: %s\n' % os.path.basename(os.path.normpath(a.outdir))]
    provenance_md(secs, cfg, out)

    out.append('## Correctness\n')
    out.append('%d runs%s; %d fail a check%s. A run counts only if it exits 0, prints its throughput, finds every '
               'looked-up key when read-only (success_read == operations_num), finds every warm-up key '
               '(warmup_success_read == warmup_num) and, when pinning was requested, reports the core.\n'
               % (len(runs), ' of %d planned' % planned if planned else '', nfail,
                  ' (%d earlier attempts superseded by a --resume rerun; their logs are *.log.prev)' % retried
                  if retried else ''))
    want_w = as_int(cfg.get('warmup'), 0)
    out.append('| dataset | index | runs | exit 0 | throughput | success_read | warm-up | pin | status |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for ds in datasets:
        for ix in indexes:
            cell = [r for r in runs if r['dataset'] == ds and r['index'] == ix]
            if not cell:
                continue
            def n_ok(word):  # runs that pass this check
                return '%d/%d' % (sum(1 for r in cell if not any(c == word for c, _ in r['problems'])), len(cell))
            bad = any(r['problems'] for r in cell)
            out.append('| %s | %s | %d | %s | %s | %s | %s | %s | %s |' % (
                ds, ix, len(cell), n_ok('exit'), n_ok('throughput'), n_ok('success_read'),
                n_ok('warmup') if want_w > 0 else 'n/a', n_ok('pin') if as_int(cfg.get('pin'), -1) >= 0 else 'n/a',
                'FAIL' if bad else 'PASS'))
    failed = [r for r in runs if r['problems']]
    if failed:
        out.append('\nFailed runs:\n')
        for r in failed:
            out.append('- r%s %s %s (`%s`): %s' % (r['repeat'], r['dataset'], r['index'], r.get('log', ''),
                                                  '; '.join(m for _, m in r['problems'])))
    out.append('')
    busy = load_md(runs, out)

    out.append('## Results per dataset\n')
    out.append('Throughput in Mops/s over the runs that pass every check (GRE times one pass over the operations '
               'per run). "vs %s" is the geometric-mean ratio with its 95%% CI. Index RSS = RSS after bulk load '
               'minus RSS before it (both after an allocator purge), per bulk-loaded key; "self-reported" is GRE\'s `Memory:` per key. '
               '"run growth" is RSS after the timed run minus RSS after the build, per key. "unpurged" is RSS right '
               'after bulk load, before the purge, minus RSS before the build, per key (build temporaries the '
               'allocator still holds included). "peak GB" is the process\'s peak RSS (VmHWM), GRE\'s own arrays '
               'included. "SD ln" is the cell\'s own SD of ln throughput; * marks one above %.0fx the pooled SD, whose '
               'CI assumes equal variance and is then too narrow.\n' % (ref, SD_FLAG))
    warn = []
    any_w = any(r['parsed'] and r['parsed'].get('warmup_num') for r in runs)
    for ds in datasets:
        good = [r for r in runs if r['dataset'] == ds and not r['problems']]
        cells = {}
        for r in good:
            cells.setdefault(r['index'], []).append(r)
        lns = dict((ix, [math.log(r['parsed']['throughput']) for r in rs]) for ix, rs in cells.items())
        ss = sum(sum((x - mean(v)) ** 2 for x in v) for v in lns.values())
        df = sum(len(v) for v in lns.values()) - len(lns)
        sd = math.sqrt(ss / df) if df > 0 else None
        inits = sorted(set(r['parsed']['init_table_size'] for r in good))
        out.append('### %s\n' % ds)
        out.append('Bulk-loaded keys: %s; pooled SD of ln throughput %s (df %d, t %s).\n' % (
            ', '.join(str(x) for x in inits) or '-', fmt(sd, 4) or 'n/a', df, fmt(t95(df), 3) if df > 0 else 'n/a'))
        if len(inits) > 1:
            warn.append('%s: bulk-loaded key count differs between runs (%s)' % (ds, inits))
        head = ['index', 'runs', 'mean Mops/s', 'min', 'max', 'SD ln', 'vs %s [95%% CI]' % ref, 'index RSS B/key',
                'unpurged B/key', 'self-reported B/key', 'run growth B/key', 'peak GB', 'build s']
        if any_w:
            head.append('warm-up Mops/s')
        out.append('| ' + ' | '.join(head) + ' |')
        out.append('|' + '---|' * len(head))
        for ix in indexes:
            rs = cells.get(ix)
            if not rs:
                if any(r['dataset'] == ds and r['index'] == ix for r in runs):
                    out.append('| %s | 0 | (every run failed) |' % ix + ' |' * (len(head) - 3))
                continue
            tp = [r['parsed']['throughput'] / 1e6 for r in rs]
            v = lns[ix]
            own_sd = math.sqrt(sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None
            sd_s = fmt(own_sd, 4)
            if own_sd is not None and sd and own_sd > SD_FLAG * sd:
                sd_s += ' *'
                warn.append('%s %s: SD of ln throughput %.4f is %.1fx the pooled %.4f; its CI is too narrow'
                            % (ds, ix, own_sd, own_sd / sd, sd))
            if ix == ref:
                ratio = '1 (reference)'
            elif ref in lns:
                d = mean(lns[ix]) - mean(lns[ref])
                ratio = '%.3f' % math.exp(d)
                if sd is not None:
                    h = t95(df) * sd * math.sqrt(1.0 / len(lns[ix]) + 1.0 / len(lns[ref]))
                    ratio += ' [%.3f, %.3f]' % (math.exp(d - h), math.exp(d + h))
            else:
                ratio = '(no %s runs)' % ref

            def per_key(f):
                v = [f(r['parsed']) / float(r['parsed']['init_table_size']) for r in rs
                     if f(r['parsed']) is not None and r['parsed']['init_table_size']]
                return mean(v)
            rss = per_key(lambda p: p.get('index_rss_bytes'))
            selfrep = per_key(lambda p: p.get('memory'))
            growth = per_key(lambda p: p['rss_after_run_bytes'] - p['rss_after_build_bytes']
                             if 'rss_after_run_bytes' in p and 'rss_after_build_bytes' in p else None)
            unpurged = per_key(lambda p: p['rss_after_build_unpurged_bytes'] - p['rss_before_build_bytes']
                               if 'rss_after_build_unpurged_bytes' in p and 'rss_before_build_bytes' in p else None)
            peaks = [r['parsed']['rss_peak_bytes'] / 1e9 for r in rs if r['parsed'].get('rss_peak_bytes', -1) >= 0]
            peak = max(peaks) if peaks else None
            build = mean([r['parsed']['build_ns'] / 1e9 for r in rs if 'build_ns' in r['parsed']])
            sr = fmt(selfrep, 2)
            if ix == 'btree' and selfrep == 0:
                sr = '0 (not reported)'
            row = [ix, str(len(rs)), fmt(mean(tp)), fmt(min(tp)), fmt(max(tp)), sd_s, ratio, fmt(rss, 2),
                   fmt(unpurged, 2), sr, fmt(growth, 2), fmt(peak, 2), fmt(build, 2)]
            if any_w:
                wm = [p['warmup_num'] / (p['warmup_ns'] / 1e9) / 1e6 for p in (r['parsed'] for r in rs)
                      if p.get('warmup_num') and p.get('warmup_ns')]
                row.append(fmt(mean(wm)))
            out.append('| ' + ' | '.join(row) + ' |')
        if 'sortedarray' in cells:
            v = [r['parsed']['index_rss_bytes'] / float(r['parsed']['init_table_size']) for r in cells['sortedarray']
                 if 'index_rss_bytes' in r['parsed'] and r['parsed']['init_table_size']]
            if v:
                dev = mean(v) / SORTED_ARRAY_BPK - 1
                verdict = 'OK' if abs(dev) <= MEM_TOL else 'WARN'
                out.append('\nMemory-method check: sortedarray index RSS %.3f B/key against %.0f B/key expected '
                           '(%+.2f%%, tolerance %.0f%%): %s' % (mean(v), SORTED_ARRAY_BPK, 100 * dev, 100 * MEM_TOL,
                                                                 verdict))
                if verdict == 'WARN':
                    # +100%: the 16 B/key of small blocks freed by its probe all stayed resident (the purge failed,
                    # or jemalloc is not the process's malloc)
                    warn.append('%s: sortedarray index RSS %.3f B/key is off 16 B/key by %+.2f%%: the allocator '
                                'purge did not release all freed memory (about +100%% if it released none), so the '
                                'index RSS of this dataset includes retained build temporaries' % (ds, mean(v), 100 * dev))
            else:
                out.append('\nMemory-method check: sortedarray ran but its logs have no index_rss_bytes.')
        splice_md(ds, cells, a, cfg, out, warn)
        out.append('')

    out.append('## Notes\n')
    out.append('- Index RSS counts what the index holds after bulk load: nodes, its own copies of keys and payloads, '
               'and allocator slack. It needs the gre_lite.sh patch; unpatched logs leave it empty.')
    out.append('- GRE\'s self-reported `Memory:` is each wrapper\'s own accounting, taken after the timed run: the STX '
               'B+tree wrapper reports 0, and ART (artunsync) excludes the 16-B key/payload records it points to.')
    out.append('- A sorted array of (key, payload) pairs needs exactly 16 B/key, which is what the memory-method '
               'check compares against. Before building, sortedarray also allocates and frees 16 B/key in small '
               'blocks, so OK means both that the RSS delta tracks one resident allocation and that the purge '
               'released freed small blocks. It does not show that an index\'s RSS holds no other slack.')
    out.append('- Index RSS (purged, around the build) is not SCALE-LI\'s "real B/key" (aidb_ba/memory_report.py: '
               'unpurged RSS plateau during the run minus nominal buffers). The closer match is "unpurged"; compare '
               'like with like only.')
    out.append('- Throughput CIs pool the SD over the cells of a dataset (equal variance). With few repeats a cell '
               'flagged * has a CI that understates its spread; read its min and max.')
    if any(r['index'] in SPLICE_IDX for r in runs):
        out.append('- SPLICE: the self-reported splice_total_bytes is the arena (2 MiB rounded, every page written at '
                   'build) plus records, router and exceptions, so index RSS should match it within a few percent. '
                   'Predicted E[D] is splice_count\'s model in serialized DRAM accesses (4 KiB pages, 2 MB L2), an '
                   'estimate, not a measurement. trace_* runs time every 1M get() calls and are not headline numbers.')
    if busy:
        warn.append('%d runs started beside another benchmark process; their timings may be disturbed' % len(busy))
    if warn:
        out.append('\n## Warnings\n')
        out.extend('- WARN ' + w for w in warn)
    # bytes, not print: Python 3.6 under LANG=C would encode stdout as ASCII
    sys.stdout.buffer.write(('\n'.join(out) + '\n').encode('utf-8'))
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
