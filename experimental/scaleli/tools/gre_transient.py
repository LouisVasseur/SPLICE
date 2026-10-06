#!/usr/bin/env python3
"""Within-process warm-up transient check for gre_run.sh OUTDIRs of the trace_* indexes (SERVER.md 5b).

    python3 gre_transient.py OUTDIR [OUTDIR ...] [--accept-w 20000000] > transient.md

trace_splice and trace_lipp (integrations/gre_splice/trace_interface.h) time every 1,000,000 get() calls of a run:
'trace_chunk: k ns' is the time from the start of call k*1M to the start of call (k+1)*1M. The first W calls are
the untimed warm-up (--warmup W, which must be a multiple of 1M), so the timed chunks are k = W/1M .. W/1M + ops/1M - 2;
the last chunk is dropped, because it ends after GRE's post-run work. Per run, against the run's own median chunk:
  deficit        = first timed chunk / median - 1 (positive: the first 1M lookups were slower)
  spread         = 1.4826 * MAD / median (the robust chunk-to-chunk spread)
  transient_len  = the number of leading chunks before the run settles: the first chunk from which SETTLE (5)
                   consecutive chunks are within max(2 spread, 1%) of the median
  late outliers  = chunks after that outside the band. A 2-sigma band is crossed by about 5% of chunks by noise
                   alone, so 'every later chunk within the band' would call nearly every 99-chunk run a transient;
                   isolated late outliers are listed, not counted as transient.
The comparison is paired within one process, so per-process noise (6% between processes on the Mac) cancels.
Verdict per (dataset, index): accept W = --accept-w iff the mean deficit of its runs is <= max(2 x mean spread, 1%)
and the transient at W = 0 lasts at most 2 chunks (2M lookups) in every W = 0 run; otherwise INVESTIGATE (THP
collapse, CPU frequency, page faults) rather than absorbing it into a larger warm-up. Only the trace_* runs of an
OUTDIR are read; they are never headline numbers (the wrapper adds a counter to every get()).
Exit status 2 if no run has trace lines. Standard library only; runs on Python 3.6.
"""
import argparse
import os
import re
import sys

CHUNK = 1000000
TOL_MIN = 0.01        # 1%: below this the chunk timer and the clock are not trusted to resolve a deficit
MAX_TRANSIENT = 2     # chunks of 1M lookups at W = 0
SETTLE = 5            # consecutive in-band chunks that end a transient

RE_CHUNK = re.compile(r'^trace_chunk: (\d+) (-?\d+)\s*$')
RE_INT = re.compile(r'^(trace_chunk_size|trace_chunks_n|trace_calls|success_read|warmup_success_read): (\d+)\s*$')
RE_PATCH = re.compile(r'^gre_lite_patch: warmup_num=(\d+) pin_core=(-?\d+)\s*$')
RE_FLAG = re.compile(r'^(operations_num|index|keys_file) = (.*?)\s*$')


def median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def parse_log(path):
    r = {'chunks': {}, 'flags': {}}
    try:
        f = open(path, encoding='utf-8', errors='replace')
    except (IOError, OSError):
        return None
    with f:
        for line in f:
            m = RE_CHUNK.match(line)
            if m:
                r['chunks'][int(m.group(1))] = int(m.group(2))
                continue
            m = RE_INT.match(line)
            if m:
                r[m.group(1)] = int(m.group(2))
                continue
            m = RE_PATCH.match(line)
            if m:
                r['warmup_num'] = int(m.group(1))
                continue
            m = RE_FLAG.match(line)
            if m and m.group(1) not in r['flags']:
                r['flags'][m.group(1)] = m.group(2)
    return r


def load_runs(outdir):
    """The last runs.tsv row per (repeat, dataset, index), as gre_report.py counts them."""
    path = os.path.join(outdir, 'runs.tsv')
    if not os.path.exists(path):
        return []
    last, order = {}, []
    with open(path, encoding='utf-8', errors='replace') as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            if not line.strip():
                continue
            row = dict(zip(header, line.rstrip('\n').split('\t')))
            key = (row.get('repeat'), row.get('dataset'), row.get('index'))
            if key not in last:
                order.append(key)
            last[key] = row
    return [last[k] for k in order]


def analyse(log):
    """Timed chunks and their statistics, or (None, reason)."""
    if log.get('trace_chunk_size', CHUNK) != CHUNK:
        return None, 'trace_chunk_size %d, expected %d' % (log['trace_chunk_size'], CHUNK)
    w = log.get('warmup_num', 0)
    try:
        ops = int(log['flags'].get('operations_num', ''))
    except ValueError:
        return None, 'no operations_num'
    if w % CHUNK or ops % CHUNK:
        return None, 'warm-up %d and operations %d must be multiples of %d' % (w, ops, CHUNK)
    k0 = w // CHUNK
    ks = list(range(k0, k0 + ops // CHUNK - 1))  # the last timed chunk ends after GRE's post-run work: dropped
    if len(ks) < 3:
        return None, 'fewer than 3 timed chunks'
    missing = [k for k in ks if k not in log['chunks']]
    if missing:
        return None, '%d timed chunks missing (first %d)' % (len(missing), missing[0])
    c = [float(log['chunks'][k]) for k in ks]
    m = median(c)
    if not m or m <= 0:
        return None, 'non-positive median chunk'
    spread = 1.4826 * median([abs(x - m) for x in c]) / m
    tol = max(2 * spread, TOL_MIN)
    inb = [abs(x / m - 1) <= tol for x in c]
    tl = 0
    while tl < len(c) and not all(inb[tl:tl + SETTLE]):
        tl += 1
    late = sum(1 for b in inb[tl:] if not b)
    return {'w': w, 'ops': ops, 'n': len(c), 'median_ns': m, 'deficit': c[0] / m - 1, 'spread': spread, 'tol': tol,
            'transient_len': tl, 'late': late, 'first': [x / m - 1 for x in c[:5]]}, None


def pct(x):
    return '' if x is None else '%+.2f%%' % (100 * x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('outdirs', nargs='+')
    ap.add_argument('--accept-w', type=int, default=20000000, help='warm-up to accept or reject (default 20M)')
    a = ap.parse_args()
    groups, skipped, order = {}, [], []
    for outdir in a.outdirs:
        for row in load_runs(outdir):
            if not (row.get('index') or '').startswith('trace_'):
                continue
            where = '%s r%s %s %s' % (os.path.basename(os.path.normpath(outdir)), row.get('repeat'), row.get('dataset'),
                                      row.get('index'))
            if row.get('exit_code') != '0':
                skipped.append((where, 'exit %s' % row.get('exit_code')))
                continue
            log = parse_log(os.path.join(outdir, row.get('log', '')))
            if log is None:
                skipped.append((where, 'log missing'))
                continue
            st, why = analyse(log)
            if st is None:
                skipped.append((where, why))
                continue
            st['where'] = where
            st['ok_reads'] = log.get('success_read') == st['ops']
            key = (row['dataset'], row['index'])
            if key not in groups:
                groups[key] = {}
                order.append(key)
            groups[key].setdefault(st['w'], []).append(st)
    out = ['# Warm-up transient check (trace_* chunk timer)\n',
           'Chunks of %d lookups, each against its own run\'s median chunk. deficit = first timed chunk / median - 1; '
           'spread = 1.4826 MAD / median; transient = leading chunks before %d in a row are within max(2 spread, %.0f%%) of '
           'the median; late outliers = later chunks outside that band. Accept W = %d iff '
           'its mean deficit <= max(2 x mean spread, %.0f%%) and every W = 0 run has a transient of at most %d '
           'chunks.\n' % (CHUNK, SETTLE, 100 * TOL_MIN, a.accept_w, 100 * TOL_MIN, MAX_TRANSIENT)]
    if not groups:
        out.append('No trace_* run with chunk lines in %s.' % ', '.join(a.outdirs))
    verdicts = []
    for ds, ix in order:
        g = groups[(ds, ix)]
        out.append('## %s %s\n' % (ds, ix))
        out.append('| W | run | chunks | median ms/chunk | Mops/s | deficit | spread | transient (chunks) | '
                   'late outliers | first 5 chunks vs median | success_read |')
        out.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for w in sorted(g):
            for st in g[w]:
                out.append('| %d | %s | %d | %.3f | %.3f | %s | %.2f%% | %d | %d | %s | %s |' % (
                    w, st['where'], st['n'], st['median_ns'] / 1e6, CHUNK / (st['median_ns'] / 1e9) / 1e6,
                    pct(st['deficit']), 100 * st['spread'], st['transient_len'], st['late'],
                    ' '.join(pct(x) for x in st['first']),
                    'ok' if st['ok_reads'] else 'MISMATCH'))
        for w in sorted(g):
            rs = g[w]
            out.append('\nW = %d: %d runs, mean deficit %s, mean spread %.2f%%, transient %s chunks.'
                       % (w, len(rs), pct(mean([s['deficit'] for s in rs])), 100 * mean([s['spread'] for s in rs]),
                          ', '.join(str(s['transient_len']) for s in rs)))
        acc, cold = g.get(a.accept_w), g.get(0)
        if not acc or not cold:
            v = 'INCOMPLETE (needs runs at W = 0 and W = %d)' % a.accept_w
        else:
            md, ms = mean([s['deficit'] for s in acc]), mean([s['spread'] for s in acc])
            tl = max(s['transient_len'] for s in cold)
            fails = []
            if md > max(2 * ms, TOL_MIN):
                fails.append('mean deficit %s at W = %d exceeds max(2 x spread, 1%%) = %.2f%%'
                             % (pct(md), a.accept_w, 100 * max(2 * ms, TOL_MIN)))
            if tl > MAX_TRANSIENT:
                fails.append('transient at W = 0 lasts %d chunks (%dM lookups), more than %d'
                             % (tl, tl, MAX_TRANSIENT))
            v = ('ACCEPT W = %d' % a.accept_w if not fails else
                 'INVESTIGATE (THP collapse, frequency, page faults): ' + '; '.join(fails))
        verdicts.append((ds, ix, v))
        out.append('\n**Verdict: %s**\n' % v)
    if verdicts:
        out.append('## Summary\n')
        out.append('| dataset | index | verdict |')
        out.append('|---|---|---|')
        for ds, ix, v in verdicts:
            out.append('| %s | %s | %s |' % (ds, ix, v))
        out.append('')
    if skipped:
        out.append('## Runs not used\n')
        for where, why in skipped:
            out.append('- %s: %s' % (where, why))
    sys.stdout.buffer.write(('\n'.join(out) + '\n').encode('utf-8'))
    return 0 if groups else 2


if __name__ == '__main__':
    sys.exit(main())
