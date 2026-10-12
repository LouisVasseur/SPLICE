#!/usr/bin/env python3
"""Tests of the SPLICE-H additions to tools/gre_report.py and of tools/gre_transient.py, on synthetic gre_run.sh
OUTDIRs (no GRE, no data). Standard library only, Python 3.6+:

    python3 tests/test_gre_report_splice.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), 'tools')
REPORT = os.path.join(TOOLS, 'gre_report.py')
TRANSIENT = os.path.join(TOOLS, 'gre_transient.py')
N = 10000000  # bulk-loaded keys of the synthetic runs


def log_text(index, tput=2000000, rss_bpk=20.0, self_bpk=20.2, layout='0x00000000000000aa', warmup=0, ops=1000000,
             compact=0, splice_lines=True, huge=-1, chunks=None, error=False):
    head = ['keys_file = /data/fb', 'read = 1', 'insert = 0', 'operations_num = %d' % ops, 'table_size = -1',
            'init_table_ratio = 1', 'thread_num = 1', 'index = %s' % index, 'seed = 1866',
            'gre_lite_patch: warmup_num=%d pin_core=-1' % warmup,
            'Table size is %d, Init table size is %d' % (N, N), 'rss_before_build_bytes: 1000000']
    body = []
    if error:
        body.append('splice_error: SPLICE_ARGS: unknown key bogus')
    elif splice_lines:
        body += ['splice_index: %s' % ('splice_thp' if index == 'splice_thp' else 'splice'),
                 'splice_args_effective: k1=16384 eps=873 compact=%d thp=%d' % (compact, index == 'splice_thp'),
                 'splice_build_threads: 16', 'splice_keys: %d' % N, 'splice_bulk_load_ns: 4000000000',
                 'splice_hash_ns: 500000000', 'splice_layout_hash: %s' % layout,
                 'splice_full_hash: 0x00000000000000bb', 'splice_total_bytes: %d' % int(self_bpk * N),
                 'splice_arena_bytes: %d' % int((self_bpk - 0.2) * N), 'splice_arena_anon_huge_bytes: %d' % huge,
                 'splice_predicted_ed_milli: 4050', 'splice_stats: {"n":%d}' % N]
    body += ['rss_after_build_unpurged_bytes: %d' % int(1000000 + rss_bpk * N),
             'rss_after_build_bytes: %d' % int(1000000 + rss_bpk * N), 'index_rss_bytes: %d' % int(rss_bpk * N),
             'build_ns: 5000000000']
    if warmup:
        body += ['warmup_success_read: %d' % warmup, 'warmup_ns: 1000000000']
    body += ['Begin running', 'Finish running', 'rss_after_run_bytes: %d' % int(1000000 + rss_bpk * N),
             'rss_peak_bytes: 4000000000']
    if chunks is not None:
        body += ['trace_chunk_size: 1000000', 'trace_calls: %d' % (warmup + ops), 'trace_chunks_n: %d' % len(chunks)]
        body += ['trace_chunk: %d %d' % (k, c) for k, c in enumerate(chunks)]
    if splice_lines and not error and index in ('splice', 'splice_thp', 'trace_splice'):
        body.append('splice_total_bytes_after_run: %d' % int(self_bpk * N))
    if not error:
        body += ['Throughput = %d' % tput, 'Memory: %d' % int(self_bpk * N), 'success_read: %d' % ops,
                 'success_insert: 0']
    return '\n'.join(head + body) + '\n'


def make_outdir(root, name, runs, cfg_extra=''):
    """runs: [(repeat, dataset, index, log_text, exit_code)]"""
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, 'logs'))
    ds = []
    ix = []
    rows = ['repeat\tdataset\tindex\texit_code\twall_s\tlog\tload1\tbusy']
    reps = 0
    for r, dset, idx, text, rc in runs:
        rel = 'logs/%s__%s__r%d.log' % (dset, idx, r)
        with open(os.path.join(d, rel), 'w') as f:
            f.write(text)
        rows.append('%d\t%s\t%s\t%d\t1.0\t%s\t0.10\t0' % (r, dset, idx, rc, rel))
        if dset not in ds:
            ds.append(dset)
        if idx not in ix:
            ix.append(idx)
        reps = max(reps, r)
    with open(os.path.join(d, 'runs.tsv'), 'w') as f:
        f.write('\n'.join(rows) + '\n')
    with open(os.path.join(d, 'config.txt'), 'w') as f:
        f.write('datasets=%s\nindexes=%s\nops=1000000\nwarmup=%d\npin=-1\nread=1\ninsert=0\n%srepeats=%d\n'
                % (','.join(ds), ','.join(ix), runs[0][3].count('warmup_success_read') and 20000000 or 0,
                   cfg_extra, reps))
    return d


def run(script, *args):
    p = subprocess.Popen([sys.executable, script] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    out, err = p.communicate()
    return p.returncode, out, err


class ReportSplice(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_no_splice_output_unchanged(self):
        """Without SPLICE runs the report is byte-identical to the committed gre_report.py (HEAD)."""
        d = make_outdir(self.tmp, 'base', [(r, 'fb', ix, log_text(ix, splice_lines=False, rss_bpk=16.0), 0)
                                           for r in (1, 2) for ix in ('sortedarray', 'pgm', 'lipp')])
        rc, new, _ = run(REPORT, d)
        self.assertEqual(rc, 0)
        self.assertNotIn('SPLICE check', new)
        try:
            old = subprocess.check_output(['git', 'show', 'HEAD:./gre_report.py'], cwd=TOOLS, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest('git HEAD version of gre_report.py not available')
        path = os.path.join(self.tmp, 'gre_report_head.py')
        with open(path, 'wb') as f:
            f.write(old)
        for extra in ([], ['--ref', 'pgm']):
            rc_old, out_old, _ = run(path, d, *extra)
            rc_new, out_new, _ = run(REPORT, d, *extra)
            self.assertEqual((rc_old, out_old), (rc_new, out_new))

    def test_memory_check_ok_and_warn(self):
        d = make_outdir(self.tmp, 'mem', [
            (1, 'fb', 'splice', log_text('splice', rss_bpk=20.4, self_bpk=20.2), 0),
            (2, 'fb', 'splice', log_text('splice', rss_bpk=20.4, self_bpk=20.2), 0),
            (1, 'fb', 'splice_thp', log_text('splice_thp', rss_bpk=20.4, self_bpk=20.2, huge=180000000), 0),
            (1, 'osm', 'splice', log_text('splice', rss_bpk=26.0, self_bpk=20.2), 0),         # off by 29%, > 25
            (1, 'osm', 'splice_thp', log_text('splice_thp', rss_bpk=16.5, self_bpk=16.4, compact=1), 0),  # compact
            (1, 'fb', 'sortedarray', log_text('sortedarray', splice_lines=False, rss_bpk=16.0), 0),
            (1, 'osm', 'sortedarray', log_text('sortedarray', splice_lines=False, rss_bpk=16.0), 0)])
        rc, out, _ = run(REPORT, d)
        self.assertEqual(rc, 0, out)  # warnings never change the exit status
        fb = out.split('### fb')[1].split('### osm')[0]
        osm = out.split('### osm')[1].split('## Notes')[0]
        self.assertIn('SPLICE check', fb)
        self.assertRegex(fb, r'\| splice \| 2 \| 20\.400 \| 20\.200 \| \+0\.99% \| 4\.050 \| 4\.00 \| 0\.50 \| n/a \| '
                             r'0x00000000000000aa \| OK \|')
        self.assertRegex(fb, r'\| splice_thp \| 1 \| .* \| 0\.900 \| 0x00000000000000aa \| OK \|')  # 180M / 200M
        self.assertRegex(osm, r'\| splice \| 1 \| 26\.000 .* WARN \|')
        self.assertRegex(osm, r'\| splice_thp \(compact\) \| 1 \| 16\.500 .* OK \|')
        warn = out.split('## Warnings')[1]
        self.assertIn('osm splice: SPLICE check: index RSS 26.000 B/key is +28.71% off', warn)
        self.assertIn('osm splice: SPLICE check: index RSS 26.000 B/key outside 18-25 B/key', warn)
        self.assertNotIn('fb splice', warn)

    def test_hashes_and_splice_json(self):
        d = make_outdir(self.tmp, 'hash', [
            (1, 'fb', 'splice', log_text('splice', layout='0x00000000000000aa'), 0),
            (2, 'fb', 'splice', log_text('splice', layout='0x00000000000000ab'), 0),
            (1, 'fb', 'splice_thp', log_text('splice_thp', layout='0x00000000000000aa'), 0),
            (1, 'osm', 'trace_splice', log_text('trace_splice', layout='0x00000000000000cc', chunks=[10, 10]), 0)],
            cfg_extra='splice_arm=fast\n')
        js = os.path.join(self.tmp, 'json')
        os.makedirs(js)
        with open(os.path.join(js, 'fb.json'), 'w') as f:
            json.dump({'dataset': 'fb', 'layout_hash': '0x00000000000000AA'}, f)
        with open(os.path.join(js, 'osm.json'), 'w') as f:
            json.dump({'dataset': 'osm', 'layout_hash': '0x00000000000000cc'}, f)
        rc, out, _ = run(REPORT, d, '--splice-json', js)
        self.assertEqual(rc, 0, out)
        warn = out.split('## Warnings')[1]
        self.assertIn('fb splice: SPLICE check: layout hash differs between repeats', warn)
        self.assertIn('!= splice_count', warn)
        self.assertIn('fb: splice and splice_thp layout hashes differ', warn)
        self.assertNotIn('osm', warn.replace('osm.json', ''))
        self.assertIn('0x00000000000000cc (= splice_count)', out)

    def test_compact_hash_and_thp_always(self):
        """--splice-arm compact reads compact_layout_hash and warns when the JSON has none; a plain splice arena
        with AnonHugePages (THP [always]) is flagged as not a 4 KiB arm."""
        d = make_outdir(self.tmp, 'carm', [
            (1, 'fb', 'splice', log_text('splice', layout='0x00000000000000dd', compact=1, rss_bpk=16.0,
                                         self_bpk=16.0, huge=40000000), 0),
            (1, 'osm', 'splice', log_text('splice', layout='0x00000000000000ee', compact=1, rss_bpk=16.0,
                                          self_bpk=16.0), 0)],
            cfg_extra='splice_arm=compact\n')
        js = os.path.join(self.tmp, 'cjson')
        os.makedirs(js)
        with open(os.path.join(js, 'fb.json'), 'w') as f:
            json.dump({'dataset': 'fb', 'layout_hash': '0x00000000000000aa',
                       'compact_layout_hash': '0x00000000000000dd'}, f)
        with open(os.path.join(js, 'osm.json'), 'w') as f:
            json.dump({'dataset': 'osm', 'layout_hash': '0x00000000000000cc'}, f)
        rc, out, _ = run(REPORT, d, '--splice-json', js)
        self.assertEqual(rc, 0, out)
        self.assertIn('0x00000000000000dd (= splice_count)', out)
        warn = out.split('## Warnings')[1]
        self.assertIn('has no layout hash for the compact arm', warn)
        self.assertIn('fb splice: SPLICE check: arena has AnonHugePages (share 0.253', warn)
        self.assertNotIn('osm splice: SPLICE check: arena has AnonHugePages', warn)

    def test_failed_splice_runs(self):
        d = make_outdir(self.tmp, 'fail', [
            (1, 'fb', 'splice', log_text('splice', error=True), 2),
            (1, 'fb', 'trace_lipp', log_text('trace_lipp', splice_lines=False), 0),   # no trace chunk lines
            (1, 'fb', 'trace_splice', log_text('trace_splice', chunks=[5, 5]), 0)])
        rc, out, _ = run(REPORT, d)
        self.assertEqual(rc, 1)
        self.assertIn('splice_error: SPLICE_ARGS: unknown key bogus', out)
        self.assertIn('r1 fb trace_lipp (`logs/fb__trace_lipp__r1.log`): no trace_chunks_n line', out)
        self.assertNotIn('fb trace_splice (`', out)


class Transient(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    @staticmethod
    def chunks(w, ops, first, base=100000000):
        """W/1M warm-up chunks, then ops/1M timed ones whose first len(first) carry the given excess."""
        c = [3 * base] * (w // 1000000)
        for k in range(ops // 1000000):
            c.append(int(round(base * (1 + (first[k] if k < len(first) else 0.0)))))
        return c

    def outdirs(self, cold_first, warm_first):
        out = []
        for w, first in ((0, cold_first), (20000000, warm_first)):
            runs = [(r, 'fb', 'trace_splice', log_text('trace_splice', warmup=w, ops=20000000,
                                                       chunks=self.chunks(w, 20000000, first)), 0) for r in (1, 2, 3)]
            out.append(make_outdir(self.tmp, 'w%d' % w, runs))
        return out

    def test_accept(self):
        rc, out, _ = run(TRANSIENT, *self.outdirs([0.30, 0.05], [0.002]))
        self.assertEqual(rc, 0)
        self.assertIn('**Verdict: ACCEPT W = 20000000**', out)
        self.assertIn('| 0 | w0 r1 fb trace_splice | 19 | 100.000 | 10.000 | +30.00% |', out)

    def test_long_cold_transient(self):
        rc, out, _ = run(TRANSIENT, *self.outdirs([0.30, 0.20, 0.10, 0.05], [0.0]))
        self.assertIn('INVESTIGATE', out)
        self.assertIn('transient at W = 0 lasts 4 chunks', out)

    def test_warm_deficit(self):
        rc, out, _ = run(TRANSIENT, *self.outdirs([0.0], [0.08]))
        self.assertIn('INVESTIGATE', out)
        self.assertIn('mean deficit +8.00% at W = 20000000', out)

    def test_no_trace_runs(self):
        d = make_outdir(self.tmp, 'none', [(1, 'fb', 'pgm', log_text('pgm', splice_lines=False), 0)])
        rc, out, _ = run(TRANSIENT, d)
        self.assertEqual(rc, 2)


if __name__ == '__main__':
    unittest.main()
