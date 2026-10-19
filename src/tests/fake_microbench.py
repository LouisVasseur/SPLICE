#!/usr/bin/env python3
# SPDX-License-Identifier: MIT (SPLICE)
"""A stand-in for GRE's microbench in src/tests/test_atom_batch.py: it prints the lines gre_run.sh and gre_report.py
check (gre_lite_patch warmup_num pin_core, success_read, the trace_ chunk lines) without loading or timing anything.
gre_run.sh reads the index list from the binary: (gre_lite: nullindex pgm btree sortedarray lipp trace_pgm trace_lipp trace_nullindex)
FAKE_MB_SECONDS (default 0.2) is how long the 'lookups' take; trace_ indexes print one point per 10 ms of it.
"""
import os
import sys
import time


def main():
    f = {}
    for x in sys.argv[1:]:
        if x.startswith('--'):
            k, _, v = x[2:].partition('=')
            f[k] = v
    ops = int(f.get('operations_num', '1000'))
    w = int(f.get('warmup_num', '0'))
    pin = int(f.get('pin_core', '-1'))
    idx = f.get('index', '')
    n = 1000
    print('gre_lite_patch: warmup_num=%d pin_core=%d' % (w, pin))
    print('Table size is %d, Init table size is %d' % (n, n))
    print('index = %s' % idx)
    print('operations_num = %d' % ops)
    print('read = 1')
    print('insert = 0')
    print('rss_before_build_bytes: 1000000')
    print('rss_after_build_bytes: 1016000')
    print('index_rss_bytes: %d' % (16 * n))
    print('build_ns: 1000000')
    if pin >= 0:
        print('pinned_core: %d' % pin)
    secs = float(os.environ.get('FAKE_MB_SECONDS', '0.2'))
    pts = []
    t_start = time.monotonic()
    time.sleep(secs)
    if idx.startswith('trace_') and os.environ.get('GRE_TRACE'):
        # one point every 10 ms of the real interval (sleep wakes late at background QoS, so they are laid out
        # afterwards); the warm-up is instant here, so every chunk after the first is timed
        t_end = time.monotonic()
        n_pts = max(2, int((t_end - t_start) / 0.01))
        pts.append((0, int(t_start * 1e9)))
        for i in range(1, n_pts + 1):
            pts.append((w + ops * i // n_pts, int((t_start + (t_end - t_start) * i / n_pts) * 1e9)))
    if w:
        print('warmup_success_read: %d' % w)
        print('warmup_ns: 1000')
    print('Begin running')
    print('Throughput = %d' % int(ops / max(secs, 1e-3)))
    # FAKE_MB_FAIL_ONCE=DIR: the first run of each index named in FAKE_MB_FAIL_INDEX misses keys (a failed check)
    fail_dir = os.environ.get('FAKE_MB_FAIL_ONCE')
    marker = os.path.join(fail_dir or '.', '%s__%s__w%d' % (os.path.basename(f.get('keys_file', '')), idx, w))
    if fail_dir and idx == os.environ.get('FAKE_MB_FAIL_INDEX') and not os.path.exists(marker):
        open(marker, 'w').close()
        ops -= 1
    print('success_read: %d' % ops)
    print('rss_after_run_bytes: 1016000')
    print('rss_peak_bytes: 2000000000')
    if pts:
        # strictly increasing calls, as the real wrapper
        out, last = [], -1
        for c, t in pts:
            if c > last:
                out.append((c, t))
                last = c
        t0 = out[0][1]
        print('trace_schedule: %s' % os.environ['GRE_TRACE'])
        print('trace_calls: %d' % out[-1][0])
        print('trace_chunks_n: %d' % len(out))
        for i, (c, t) in enumerate(out):
            print('trace_point: %d %d %d -1 %d' % (i, c, t - t0, pin))
        print('trace_end: %d %d -1' % (out[-1][0], out[-1][1] - t0))
        print('trace_t0_mono_ns: %d' % t0)
    print('Memory: %d' % (16 * n))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
