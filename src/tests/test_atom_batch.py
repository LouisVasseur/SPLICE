#!/usr/bin/env python3
"""Tests of scripts/atom_batch.py (the Atom lane scheduler) that run in seconds on any machine, standard library only:
the sysfs topology reader, lane layouts, the admission (memory, exclusive items, round barrier, backfill) in the dry
run, the calibration analysis and verdict rules on synthetic chunk traces with planted effects (setup-only
slowdowns, drift, migrations, missing clock lines, mixed binaries), the merge read by gre_report.py (and
warmup_detect.py when numpy is installed), end-to-end runs through gre_run.sh with a fake microbench (retries,
resume, refusals, one condition per R, the lane smoke and its Linux checks on faked records; ATOM_BATCH_TEST=1: no
CPU masks or /proc on macOS).

    python3 src/tests/test_atom_batch.py
"""
import contextlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import atom_batch as A  # noqa: E402


def sysfs_tree(d, pairs, online=None, isolated=''):
    """A /sys/devices/system/cpu with one L2 per pair of CPUs."""
    cpus = sorted(c for p in pairs for c in p)
    with open(os.path.join(d, 'online'), 'w') as f:
        f.write(online or A.cpulist(cpus))
    with open(os.path.join(d, 'isolated'), 'w') as f:
        f.write(isolated + '\n')
    for p in pairs:
        for c in p:
            for idx, (level, shared) in enumerate(((1, str(c)), (1, str(c)), (2, A.cpulist(p)))):
                x = os.path.join(d, 'cpu%d' % c, 'cache', 'index%d' % idx)
                os.makedirs(x)
                with open(os.path.join(x, 'level'), 'w') as f:
                    f.write('%d\n' % level)
                with open(os.path.join(x, 'shared_cpu_list'), 'w') as f:
                    f.write(shared + '\n')


class Args(object):
    datasets = indexes = rounds = None
    no_anchors = False
    lanes = None
    avoid_cpus = ''


class Topology(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_atom_pairs(self):
        sysfs_tree(self.d, [(2 * i, 2 * i + 1) for i in range(8)])
        mods, iso = A.read_modules(self.d)
        self.assertEqual([m['cpus'] for m in mods], [[2 * i, 2 * i + 1] for i in range(8)])
        lay = A.Layout(mods, 4)
        self.assertEqual(lay.house, [0, 1])
        self.assertEqual(lay.lanes, [[2, 3], [4, 5], [6, 7], [8, 9]])
        self.assertEqual(lay.pool, [10, 11, 12, 13, 14, 15])
        L = lay.lane(0)
        self.assertEqual((L['core'], L['siblings'], A.cpulist(L['mask'])), (2, [3], '2-3,10-15'))
        # no lane's mask ever touches another lane's module
        for j in range(4):
            for i in range(4):
                if i != j:
                    self.assertFalse(set(lay.lane(j)['mask']) & set(lay.lanes[i]))
        self.assertEqual(A.Layout(mods, 1).lane(0)['core'], 2)   # docs/PROTOCOL.md's core 2

    def test_split_numbering_and_partial_module(self):
        # siblings numbered n and n+8; CPU 11 not allowed (a cpuset): its module cannot host a lane
        sysfs_tree(self.d, [(i, i + 8) for i in range(8)])
        allowed = set(range(16)) - {11}
        mods, _ = A.read_modules(self.d, allowed)
        lay = A.Layout(mods, 3)
        self.assertEqual(lay.house, [0, 8])
        self.assertEqual(lay.lanes, [[1, 9], [2, 10], [4, 12]])
        self.assertIn(3, lay.pool)
        self.assertNotIn(11, lay.pool)

    def test_isolated_and_too_many_lanes(self):
        sysfs_tree(self.d, [(0, 1), (2, 3), (4, 5)], isolated='4-5')
        mods, iso = A.read_modules(self.d)
        self.assertEqual(iso, [4, 5])
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                A.Layout(mods, 2)

    def test_cpulist(self):
        for s in ('0', '0-3', '2-3,10-15', '1,3,5'):
            self.assertEqual(A.cpulist(A.parse_cpulist(s)), s)

    def test_explicit_topology(self):
        # the Mac: E cluster 0-3 (house), P clusters 4-9 and 10-15
        mods, _ = A.fake_modules('0-3/4-9/10-15')
        lay = A.Layout(mods, 2)
        self.assertEqual((lay.house, lay.lanes, lay.pool), ([0, 1, 2, 3], [[4, 5, 6, 7, 8, 9], list(range(10, 16))], []))
        self.assertEqual(A.max_lanes(mods, []), 2)
        self.assertEqual(A.max_lanes(A.fake_modules('8x2')[0], [8]), 6)
        for bad in ('0-3/2-5', 'x', '0-3//'):
            if bad == '0-3//':
                self.assertEqual(len(A.fake_modules(bad)[0]), 1)
                continue
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    A.fake_modules(bad)


class Admission(unittest.TestCase):
    def items(self, recipes, k, **kw):
        a = Args()
        for x, v in kw.items():
            setattr(a, x, v)
        return A.recipe_items(recipes, a, k, 'madvise')

    def check(self, sched, budget, pred):
        events = sorted([(s, 1, it) for s, e, _, it in sched] + [(e, -1, it) for s, e, _, it in sched],
                        key=lambda x: (x[0], x[1]))
        live = []
        for t, kind, it in events:
            if kind == -1:
                live.remove(it)
                continue
            live.append(it)
            if len(live) > 1:
                self.assertLessEqual(sum(pred.peak(x) for x in live), budget + 1e-6)
                for x in live:
                    self.assertFalse(A.exclusive(x, set()) and x['why'] == 'solo anchor', x['id'])
                    self.assertNotEqual(x['kind'], 'cmd')
        # soft round barrier: nothing of round r+1 starts before every item of round r has started
        last_start = {}
        first_start = {}
        for s, e, _, it in sched:
            r = it['round']
            last_start[r] = max(last_start.get(r, 0), s)
            first_start[r] = min(first_start.get(r, 1e18), s)
        for r in last_start:
            if r + 1 in first_start:
                self.assertGreaterEqual(first_start[r + 1], last_start[r] - 1e-6)

    def test_campaign_k4(self):
        pred = A.Predictor()
        its = self.items(['warmup', 'bench'], 4)
        self.assertEqual(len(its), 200 + 420 + 12)
        sched, total = A.simulate(A.order_items(its, pred, 8), 4, 60.0, pred, 8)
        self.assertEqual(len(sched), len(its))
        self.check(sched, 60.0, pred)
        serial, total1 = A.simulate(A.order_items(self.items(['warmup', 'bench'], 1), pred, 14), 1, 60.0, pred, 14)
        self.assertLess(total, total1 / 2.0)        # the whole point
        # the first round leads with the solo anchors, then the largest memory x time cells (LIPP)
        firsts = [it['id'] for s, e, _, it in sched[:6]]
        self.assertTrue(all('warm_solo' in x for x in firsts[:4]), firsts)
        self.assertIn('lipp', firsts[4])

    def test_small_memory_serializes_lipp(self):
        pred = A.Predictor()
        its = self.items(['bench'], 4, datasets='fb,stack', indexes='lipp,pgm', rounds=1)
        sched, _ = A.simulate(A.order_items(its, pred, 8), 4, 50.0, pred, 8)
        self.check(sched, 50.0, pred)

    def test_uncertain_cell_alone_until_measured(self):
        pred = A.Predictor()
        its = self.items(['bench'], 2, datasets='osm', indexes='lipp,pgm', rounds=1, no_anchors=True)
        sched, _ = A.simulate(A.order_items(its, pred, 12), 2, 60.0, pred, 12)
        first = [x for x in sched if x[3]['idx'] == 'lipp'][0]
        overl = [x for x in sched if x is not first and x[0] < first[1] and x[1] > first[0]]
        self.assertEqual(overl, [])

    def test_predictor_learns(self):
        it = A.gre_item('warm', 'fb', 'pgm', 1, 100000000, 20000000)
        pred = A.Predictor([{'item': it, 'status': 'pass', 'wall_s': 300.0, 'rss_peak_gb': 20.0, 'mask': list(range(8))}])
        self.assertAlmostEqual(pred.wall(it, 8), 300.0)
        self.assertAlmostEqual(pred.peak(it), 20.0 * 1.05 + 0.5, places=6)
        other = A.gre_item('warm', 'osm', 'pgm', 1, 100000000, 20000000)
        self.assertGreater(pred.wall(other, 8), A.model_wall(other, 8))   # the index ratio carries over

    def test_backfill_respects_reservation(self):
        pred = A.Predictor()
        big = A.gre_item('warm', 'fb', 'lipp', 1, 100000000, 20000000)
        short = A.gre_item('trace', 'fb', 'trace_nullindex', 1, 30000000, 0)
        long_ = A.gre_item('warm', 'fb', 'sortedarray', 1, 100000000, 20000000)
        running = [{'item': long_, 'end': 200.0, 'reserved': 20.0, 'lane': 0}]
        # the head (LIPP, ~45 GB) does not fit in 30 GB free; it fits once the running run ends at 200 s. The
        # sortedarray run (~350 s) would delay it; the nullindex trace (~90 s) ends before.
        sel = A.select([big, long_, short], running, [1, 2], 30.0, 0.0, pred, 8, {}, set())
        self.assertEqual(sel[0]['id'], short['id'])
        sel = A.select([big, long_, short], running, [1, 2], 30.0, 150.0, pred, 8, {}, set())
        self.assertIsNone(sel)


class Calibration(unittest.TestCase):
    def synth(self, effect, setup_extra=0.0, pairs=20, block=5.0, seed=1):
        rng = random.Random(seed)
        blocks, t, pts, calls = [], 100.0, [], 0
        t0 = t
        chunk = 0.06
        warm_calls = 300000
        state_order = []
        for p in range(pairs):
            o = ['off', 'on'] if rng.random() < 0.5 else ['on', 'off']
            for st in o:
                blocks.append({'pair': p, 'state': st, 'start': t, 'end': t + block, 'setup': p % 2})
                state_order.append((t, t + block, st, p % 2))
                t += block
        # chunks of 65536 calls; latency 850 ns, x(1+effect) while on (+setup_extra when a load is in setup)
        tt = t0
        pts.append((0, 0, 2))
        while tt < t:
            st = [x for x in state_order if x[0] <= tt < x[1]]
            lat = 850e-9 * (1 + rng.gauss(0, 0.01))
            if st and st[0][2] == 'on':
                lat *= 1 + effect + (setup_extra if st[0][3] else 0.0)
            tt += 65536 * lat
            calls += 65536
            pts.append((calls, int((tt - t0) * 1e9), 2))
        return pts, t0, warm_calls, blocks

    def synth2(self, effect, setup_pairs=None, setup_effect=0.0, drift=0.0, pairs=20, cpu=2, migrate_after=None, seed=1,
               noise=0.01, block=5.0):
        """synth with control: the pairs whose K-way block had a load in setup (default the first half, as when the
        loads start together at the probe's pin) and their extra slowdown, a linear drift of the latency (fraction per
        second), and chunks that ran on another CPU after a time."""
        if setup_pairs is None:
            setup_pairs = range(pairs // 2)
        rng = random.Random(seed)
        blocks, t, pts, calls = [], 100.0, [(0, 0, cpu)], 0
        t0 = t
        for p in range(pairs):
            o = ['off', 'on'] if rng.random() < 0.5 else ['on', 'off']
            for st in o:
                blocks.append({'pair': p, 'state': st, 'start': t, 'end': t + block,
                               'setup': 1 if p in setup_pairs else 0})
                t += block
        tt = t0
        while tt < t:
            b = [x for x in blocks if x['start'] <= tt < x['end']]
            lat = 850e-9 * (1 + rng.gauss(0, noise)) * (1 + drift * (tt - t0))
            if b and b[0]['state'] == 'on':
                lat *= 1 + effect + (setup_effect if b[0]['setup'] else 0.0)
            tt += 65536 * lat
            calls += 65536
            c = cpu if migrate_after is None or tt - t0 < migrate_after else cpu + 1
            pts.append((calls, int((tt - t0) * 1e9), c))
        return pts, t0, 300000, blocks

    def write_session(self, R, k, name, synth, sha='abc', t0_line=True, pin_off=0.0):
        pts, t0, w, blocks = synth
        sd = os.path.join(R, 'calib', 'K%d' % k, name)
        os.makedirs(os.path.join(sd, 'probe', 'logs'))
        ix = 'trace_' + name.split('_')[0]
        with open(os.path.join(sd, 'probe', 'logs', 'fb__%s__r1.log' % ix), 'w') as f:
            f.write('gre_lite_patch: warmup_num=%d pin_core=2\n' % w)
            for i, (c, tn, cpu) in enumerate(pts):
                f.write('trace_point: %d %d %d -1 %d\n' % (i, c, tn, cpu))
            if t0_line:
                f.write('trace_t0_mono_ns: %d\n' % int(t0 * 1e9))
        with open(os.path.join(sd, 'toggles.tsv'), 'w') as f:
            f.write('pair\tstate\tmono_start\tmono_end\tload_setup\tload_timed\tload_idle\n')
            for b in blocks:
                f.write('%d\t%s\t%.6f\t%.6f\t%d\t0\t0\n' % (b['pair'], b['state'], b['start'], b['end'], b['setup']))
        A.write_json(os.path.join(sd, 'session.json'), {
            'status': 'done', 'k': k, 'probe': {'ds': 'fb', 'idx': ix}, 'block_s': 5.0, 'mono_pin': t0 + pin_off,
            'binary_sha256': sha, 'topology': [[0, 1]]})

    def verdict_of(self, sessions):
        """sessions: [(k, name, synth2 kwargs, write_session kwargs)] -> (admissible K, verdict)."""
        R = tempfile.mkdtemp()
        try:
            for k, name, kw, wkw in sessions:
                self.write_session(R, k, name, self.synth2(**kw), **wkw)
            v = A.make_verdict(R, 1.0)
            return v['admissible_k'], v
        finally:
            shutil.rmtree(R)

    def test_verdict_rules(self):
        def both(k, kw, wkw=None):
            return [(k, 'pgm_fb', dict(kw, seed=10 * k + 1), wkw or {}),
                    (k, 'lipp_fb', dict(kw, seed=10 * k + 2), wkw or {})]
        # K=4 passes at once: the calibration stops there
        self.assertEqual(self.verdict_of(both(4, {'effect': 0.002}))[0], 4)
        # 3% at K=4, 1.5% at K=3, 0.1% at K=2: K=2
        self.assertEqual(self.verdict_of(both(4, {'effect': 0.03}) + both(3, {'effect': 0.015}) +
                                         both(2, {'effect': 0.001}))[0], 2)
        # a K that passes above one that fails is a red flag: serial
        self.assertEqual(self.verdict_of(both(3, {'effect': 0.001}) + both(2, {'effect': 0.03}))[0], 1)
        # one probe fails: the K fails
        self.assertEqual(self.verdict_of([(4, 'pgm_fb', {'effect': 0.001}, {}),
                                          (4, 'lipp_fb', {'effect': 0.02}, {})])[0], 1)
        # 0.8% planted with noisy chunks and few pairs: the point estimate is below 1%, the upper bound decides
        k, v = self.verdict_of(both(4, {'effect': 0.008, 'noise': 0.03, 'pairs': 8}))
        his = [r['all']['hi'] for t in v['tested'] for r in t['probes']]
        self.assertGreater(max(his), 1.0)
        self.assertEqual(k, 1)
        # only the pairs with a load in setup are slow (+1.5% in 4 of 20): the mean passes, the setup rule fails
        k, v = self.verdict_of(both(4, {'effect': 0.0, 'setup_pairs': (1, 6, 11, 16), 'setup_effect': 0.015}))
        self.assertEqual(k, 1)
        self.assertIn('load in setup', v['text'])
        # slow drift (+5% over the 200 s) does not bias the paired estimate
        k, v = self.verdict_of(both(4, {'effect': 0.003, 'drift': 0.05 / 200}))
        self.assertEqual(k, 4)
        for t in v['tested']:
            for r in t['probes']:
                self.assertTrue(0.0 < r['all']['pct'] < 0.6, r['all'])
        # the probe migrated off its core: fail
        k, v = self.verdict_of(both(4, {'effect': 0.0, 'migrate_after': 100.0}))
        self.assertEqual(k, 1)
        self.assertIn('probe ran on CPUs', v['text'])
        # too few pairs: fail
        self.assertEqual(self.verdict_of(both(4, {'effect': 0.0, 'pairs': 4}))[0], 1)
        # no K-way block ever had a load in setup: the setups beside a timed run are untested, so the K fails
        k, v = self.verdict_of(both(4, {'effect': 0.0, 'setup_pairs': (3, 9)}))
        self.assertEqual(k, 1)
        self.assertIn('setups beside a timed run are untested', v['text'])
        # no trace_t0_mono_ns (an older build): the polled pin aligns the chunks, within 20 ms
        self.assertEqual(self.verdict_of(both(4, {'effect': 0.002}, {'t0_line': False, 'pin_off': 0.015}))[0], 4)
        # sessions of two binaries vouch for none
        k, v = self.verdict_of([(4, 'pgm_fb', {'effect': 0.0}, {'sha': 'a'}), (4, 'lipp_fb', {'effect': 0.0}, {'sha': 'b'})])
        self.assertEqual(k, 1)
        self.assertIn('different binaries', v['text'])

    def test_planted_effects(self):
        for eff, lo, hi in ((0.0, -0.4, 0.4), (0.005, 0.2, 0.8), (0.02, 1.6, 2.4)):
            pts, t0, w, blocks = self.synth(eff)
            an = A.analyze_session(pts, t0, w, blocks)
            m, a, b, n = an['all']
            self.assertEqual(n, 20)
            self.assertTrue(lo < A.pct(m) < hi, (eff, A.pct(m)))
            self.assertLess(A.pct(a), 100 * eff + 0.3)
            self.assertGreater(A.pct(b), 100 * eff - 0.3)
            self.assertEqual(an['cpus'], [2])

    def test_verdict(self):
        R = tempfile.mkdtemp()
        try:
            for k, eff in ((4, 0.03), (3, 0.002)):
                for name in ('pgm_fb', 'lipp_fb'):
                    sd = os.path.join(R, 'calib', 'K%d' % k, name)
                    os.makedirs(os.path.join(sd, 'probe', 'logs'))
                    pts, t0, w, blocks = self.synth(eff, seed=k)
                    ix = 'trace_' + name.split('_')[0]
                    with open(os.path.join(sd, 'probe', 'logs', 'fb__%s__r1.log' % ix), 'w') as f:
                        f.write('gre_lite_patch: warmup_num=%d pin_core=2\n' % w)
                        for i, (c, tn, cpu) in enumerate(pts):
                            f.write('trace_point: %d %d %d -1 %d\n' % (i, c, tn, cpu))
                        f.write('trace_t0_mono_ns: %d\n' % int(t0 * 1e9))
                    with open(os.path.join(sd, 'toggles.tsv'), 'w') as f:
                        f.write('pair\tstate\tmono_start\tmono_end\tload_setup\tload_timed\tload_idle\n')
                        for b in blocks:
                            f.write('%d\t%s\t%.6f\t%.6f\t%d\t0\t0\n' % (b['pair'], b['state'], b['start'], b['end'],
                                                                      b['setup']))
                    A.write_json(os.path.join(sd, 'session.json'), {
                        'status': 'done', 'k': k, 'probe': {'ds': 'fb', 'idx': ix}, 'block_s': 5.0,
                        'binary_sha256': 'abc', 'topology': [[0, 1]]})
            v = A.make_verdict(R, 1.0)
            self.assertEqual(v['admissible_k'], 3)
            self.assertIn('Admissible K = 3', v['text'])
            self.assertTrue(os.path.exists(os.path.join(R, 'calib', 'verdict.md')))
        finally:
            shutil.rmtree(R)


def fake_gre(d):
    os.makedirs(os.path.join(d, 'build'))
    mb = os.path.join(d, 'build', 'microbench')
    with open(os.path.join(HERE, 'fake_microbench.py')) as f:
        body = f.read().split('\n', 1)[1]
    with open(mb, 'w') as f:      # this interpreter: its monotonic clock is the one the scheduler reads
        f.write('#!%s\n%s' % (sys.executable, body))
    os.chmod(mb, 0o755)
    with open(os.path.join(d, 'build', 'gre_lite_build.txt'), 'w') as f:
        f.write('fake GRE for tests\n')
    return mb


def fake_data(d, names):
    os.makedirs(d)
    for n in names:
        with open(os.path.join(d, n), 'wb') as f:
            f.write((1000).to_bytes(8, 'little'))
            for i in range(1000):
                f.write((10 * i).to_bytes(8, 'little'))


def quiet(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = fn(*args)
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


@unittest.skipIf(shutil.which('bash') is None, 'needs bash')
class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.gre = os.path.join(self.d, 'GRE')
        self.mb = fake_gre(self.gre)
        self.data = os.path.join(self.d, 'data')
        fake_data(self.data, ['fb', 'osm', 'wise', 'genome', 'books', 'history'])
        self.R = os.path.join(self.d, 'R')
        os.environ['ATOM_BATCH_TEST'] = '1'
        os.environ['ATOM_BATCH_TEST_PIN'] = '1'
        os.environ['FAKE_MB_SECONDS'] = '0.1'

    def tearDown(self):
        os.environ.pop('ATOM_BATCH_TEST', None)
        os.environ.pop('ATOM_BATCH_TEST_PIN', None)
        os.environ.pop('FAKE_MB_SECONDS', None)
        shutil.rmtree(self.d)

    def args(self, *x):
        return list(x) + ['--out', self.R, '--gre', self.gre, '--data', self.data, '--topology', '8x2',
                          '--mem-available', '64', '--mem-total', '67.5', '--poll', '0.05', '--foreign', 'tag']

    def test_run_merge_report(self):
        # K=2 needs a verdict made with this binary on this topology
        os.makedirs(os.path.join(self.R, 'calib'))
        rc, out, err = quiet(A.main, self.args('bench', '--datasets', 'fb,osm', '--indexes', 'pgm,btree',
                                                '--rounds', '1', '--lanes', '2'))
        self.assertEqual(rc, 2)
        self.assertIn('needs a calibration verdict', err)
        A.write_json(os.path.join(self.R, 'calib', 'verdict.json'), {
            'admissible_k': 2, 'binary_sha256': A.sha256(self.mb),
            'topology': [list(range(2 * i, 2 * i + 2)) for i in range(8)]})
        rc, out, err = quiet(A.main, self.args('bench', '--datasets', 'fb,osm', '--indexes', 'pgm,btree',
                                                '--rounds', '1', '--lanes', '2', '--no-anchors'))
        self.assertEqual(rc, 0, out + err)
        recs = A.attempt_records(self.R)
        self.assertEqual(len(recs), 8)
        self.assertTrue(all(r['status'] == 'pass' for r in recs), [r['status'] for r in recs])
        self.assertEqual(sorted(set(r['core'] for r in recs)), [2, 4])
        merged = os.path.join(self.R, 'warm', 'merged')
        rows = A.read(os.path.join(merged, 'runs.tsv')).splitlines()
        self.assertEqual(len(rows), 5)
        self.assertIn('pin=per-run', A.read(os.path.join(merged, 'config.txt')))
        rep = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'gre_report.py'), merged],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(rep.returncode, 0, rep.stdout + rep.stderr)
        self.assertIn('## Lanes', rep.stdout)
        self.assertIn('| fb | pgm | 1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | PASS |', rep.stdout)
        # a second call resumes: nothing left to run
        rc, out, err = quiet(A.main, self.args('bench', '--datasets', 'fb,osm', '--indexes', 'pgm,btree',
                                                '--rounds', '1', '--lanes', '2', '--no-anchors'))
        self.assertEqual(rc, 0)
        self.assertIn('8 items: 8 done', out)
        self.assertEqual(len(A.attempt_records(self.R)), 8)
        # merge --only solo leaves the runs that had nothing else alive during their timed lookups
        rc, out, err = quiet(A.main, ['merge', '--out', self.R, '--only', 'solo'])
        self.assertEqual(rc, 0)

    def verdict(self, k):
        os.makedirs(os.path.join(self.R, 'calib'), exist_ok=True)
        A.write_json(os.path.join(self.R, 'calib', 'verdict.json'), {
            'admissible_k': k, 'binary_sha256': A.sha256(self.mb),
            'topology': [list(range(2 * i, 2 * i + 2)) for i in range(8)]})

    def test_retry_on_another_lane_and_recover(self):
        self.verdict(2)
        os.environ['FAKE_MB_FAIL_ONCE'] = self.d
        os.environ['FAKE_MB_FAIL_INDEX'] = 'btree'
        try:
            # an attempt left behind by a killed scheduler: closed as interrupted, then rerun as a2
            os.makedirs(os.path.join(self.R, 'cold', 'cells', 'fb__pgm__r1', 'a1'))
            rc, out, err = quiet(A.main, self.args('bench', '--datasets', 'fb', '--indexes', 'pgm,btree', '--rounds',
                                                    '1', '--lanes', '2', '--no-anchors'))
        finally:
            os.environ.pop('FAKE_MB_FAIL_ONCE')
            os.environ.pop('FAKE_MB_FAIL_INDEX')
        self.assertEqual(rc, 0, out + err)
        self.assertIn('1 interrupted attempts closed', out)
        recs = dict((r['_dir'][len(self.R) + 1:], r) for r in A.attempt_records(self.R))
        self.assertEqual(recs['cold/cells/fb__pgm__r1/a1']['status'], 'interrupted')
        self.assertEqual(recs['cold/cells/fb__pgm__r1/a2']['status'], 'pass')
        for arm in ('warm', 'cold'):
            a1, a2 = recs['%s/cells/fb__btree__r1/a1' % arm], recs['%s/cells/fb__btree__r1/a2' % arm]
            self.assertEqual((a1['status'], a2['status']), ('fail', 'pass'))
            self.assertNotEqual(a1['lane'], a2['lane'])          # retried on the other lane
        rows = A.read(os.path.join(self.R, 'warm', 'merged', 'runs.tsv')).splitlines()
        btree = [r.split('\t') for r in rows if '\tbtree\t' in r]
        self.assertEqual([(x[5], x[14]) for x in btree], [('logs/fb__btree__r1.log.a1', 'fail'),
                                                           ('logs/fb__btree__r1.log', 'pass')])
        rep = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'gre_report.py'),
                              os.path.join(self.R, 'warm', 'merged')], stdout=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(rep.returncode, 0, rep.stdout)          # the passing attempt counts
        self.assertIn('1 earlier attempts superseded by a retry', rep.stdout)

    def test_refusals_stop_the_campaign(self):
        rc, out, err = quiet(A.main, self.args('bench', '--datasets', 'covid,stack', '--indexes', 'pgm,btree',
                                                '--rounds', '1', '--lanes', '1'))
        self.assertEqual(rc, 2)                                  # no such data files: gre_run.sh exits 2
        self.assertIn('refused three runs in a row', err)
        self.assertEqual(set(r['status'] for r in A.attempt_records(self.R)), {'refused'})
        self.assertFalse(os.path.exists(os.path.join(self.R, 'sched', 'lock')))

    def test_refused_items_rerun_after_the_fix(self):
        quiet(A.main, self.args('bench', '--datasets', 'covid', '--indexes', 'pgm', '--rounds', '1', '--lanes', '1',
                                '--no-anchors'))
        self.assertEqual(set(r['status'] for r in A.attempt_records(self.R)), {'refused'})
        fake_data(os.path.join(self.d, 'data2'), ['covid'])       # the fix: the data file is there now
        for n in ('covid',):
            shutil.copy(os.path.join(self.d, 'data2', n), os.path.join(self.data, n))
        rc, out, err = quiet(A.main, self.args('bench', '--datasets', 'covid', '--indexes', 'pgm', '--rounds', '1',
                                                '--lanes', '1', '--no-anchors'))
        self.assertEqual(rc, 0, out + err)
        self.assertIn('2 items: 0 done, 0 failed for good, 2 to run', out)

    def test_one_condition_per_out(self):
        rc, out, err = quiet(A.main, self.args('warmup', '--datasets', 'fb', '--indexes', 'pgm', '--rounds', '1',
                                                '--lanes', '1', '--ops', '2000'))
        self.assertEqual(rc, 0, out + err)
        rc, out, err = quiet(A.main, self.args('warmup', '--datasets', 'fb', '--indexes', 'pgm', '--rounds', '1',
                                                '--lanes', '1'))
        self.assertEqual(rc, 2)
        self.assertIn('use another --out', err)

    def test_smoke_runs_and_checks(self):
        rc, out, err = quiet(A.main, ['smoke', '--out', self.R, '--gre', self.gre, '--data', self.data, '--topology',
                                      '0-3/4-9/10-15', '--foreign', 'tag', '--poll', '0.05'])
        self.assertEqual(rc, 0, out + err)
        self.assertIn('Smoke: PASS', A.read(os.path.join(self.R, 'smoke', 'SMOKE.md')))
        recs = [r for r in A.attempt_records(os.path.join(self.R, 'smoke'))]
        self.assertEqual(sorted(set(r['item']['arm'] for r in recs)), ['smoke', 'smoke_trace'])
        self.assertEqual(A.attempt_records(self.R), [])            # never mixed with the campaign's cells
        # the Linux checks on records as the Atom would write them
        good = dict(recs[0], mask_applied=True, mask_seen=A.cpulist(recs[0]['mask']), mono_pin=1.0, hwm_gb=1.2,
                    foreign=[], mb_pid=4242)
        rows, ok = A.smoke_checks([good], True)
        self.assertTrue(ok, rows)
        for bad, why in ((dict(good, mask_seen='0-15'), 'microbench CPUs'), (dict(good, mono_pin=None), 'pin never'),
                         (dict(good, mask_applied=False), 'mask not applied'), (dict(good, hwm_gb=0.0), 'VmHWM'),
                         (dict(good, foreign=['4242:microbench']), 'own runs'), (dict(good, status='fail'), 'status')):
            rows, ok = A.smoke_checks([bad], True)
            self.assertFalse(ok, why)
            self.assertIn(why, rows[0])
        tr = [r for r in recs if r['item']['idx'].startswith('trace_')][0]
        t0 = A.parse_trace_log(os.path.join(tr['_dir'], 'logs', 'fb__%s__r1.log' % tr['item']['idx']))[1]
        tr = dict(tr, mask_applied=True, mask_seen=A.cpulist(tr['mask']), mono_pin=t0 - 0.05, hwm_gb=1.0, foreign=[])
        rows, ok = A.smoke_checks([tr], True)
        self.assertTrue(ok, rows)
        self.assertIn('trace t0 - pin poll +0.050 s', rows[0])
        rows, ok = A.smoke_checks([dict(tr, core=99)], True)        # the fake reports the core it was given
        self.assertIn('trace chunks ran on CPUs', rows[0])
        rows, ok = A.smoke_checks([dict(tr, mono_pin=t0 - 30)], True)
        self.assertIn('trace clock', rows[0])

    def test_lock(self):
        os.makedirs(os.path.join(self.R, 'sched', 'lock'))
        with open(os.path.join(self.R, 'sched', 'lock', 'pid'), 'w') as f:
            f.write('%d\n' % os.getpid())                     # a live scheduler
        rc, out, err = quiet(A.main, self.args('bench', '--datasets', 'fb', '--indexes', 'pgm', '--rounds', '1',
                                                '--lanes', '1'))
        self.assertEqual(rc, 2)
        self.assertIn('is running on', err)

    def test_untimed_commands_run_alone_in_order(self):
        machine = A.Machine(A.main.__globals__['argparse'].Namespace(
            topology='8x2', sysfs='', mem_total=67.5, mem_available=64.0))
        order = os.path.join(self.d, 'order.txt')
        its = [A.cmd_item('sort', ds, ['sh', '-c', 'echo %s >> %s; sleep 0.05' % (ds, order)]) for ds in ('b', 'a')]
        its.append(A.cmd_item('nf_gate', 'c', ['sh', '-c', 'exit 3']))

        class Arg(object):
            retries, foreign, build_threads, splice_plan, trust_verdict, poll = 0, 'tag', 16, None, False, 0.02
            data = self.data
        s = A.Scheduler(Arg(), self.R, machine, A.Layout(machine.mods, 1), its, A.Predictor())
        s.frozen, s.sha = '', 'untimed'
        with contextlib.redirect_stdout(io.StringIO()):
            rc = s.run()
        self.assertEqual(rc, 1)                                   # the failing command fails for good
        self.assertEqual(A.read(order).split(), ['b', 'a'])       # the given order, one at a time
        st = dict((r['item']['id'], r['status']) for r in A.attempt_records(self.R))
        self.assertEqual(st, {'untimed/b__sort': 'pass', 'untimed/a__sort': 'pass', 'untimed/c__nf_gate': 'fail'})

    def test_warmup_merge_feeds_warmup_detect(self):
        rc, out, err = quiet(A.main, self.args('warmup', '--datasets', 'fb', '--indexes', 'pgm,nullindex', '--rounds',
                                                '2', '--lanes', '1'))
        self.assertEqual(rc, 0, out + err)
        merged = os.path.join(self.R, 'trace', 'merged')
        names = sorted(os.listdir(os.path.join(merged, 'logs')))
        self.assertEqual(names, ['fb__trace_nullindex__r1.log', 'fb__trace_nullindex__r2.log', 'fb__trace_pgm__r1.log',
                                 'fb__trace_pgm__r2.log'])
        self.assertIn('trace_env=GRE_TRACE=geo:64:65536 GRE_TRACE_CPU=1', A.read(os.path.join(merged, 'config.txt')))
        rep = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'gre_report.py'), merged, '--only', 'solo'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(rep.returncode, 0, rep.stdout + rep.stderr)     # K=1: every run is solo
        self.assertIn('4 runs solo', rep.stdout)
        try:
            import numpy  # noqa: F401
            import warmup_detect as W
        except ImportError:
            return
        runs = W.find_runs([merged])
        self.assertEqual(sorted((r['dataset'], r['index'], r['round']) for r in runs),
                         [('fb', 'trace_nullindex', 1), ('fb', 'trace_nullindex', 2), ('fb', 'trace_pgm', 1),
                          ('fb', 'trace_pgm', 2)])

    def test_calib_mechanics(self):
        os.environ['FAKE_MB_SECONDS'] = '6'     # 10 ms chunks: 0.3 s blocks hold enough of them
        rc, out, err = quiet(A.main, self.args('calib', '--k', '2', '--probes', 'pgm/fb', '--block-s', '0.3',
                                                '--calib-ops', '1000000'))
        self.assertEqual(rc, 0, out + err)
        sd = os.path.join(self.R, 'calib', 'K2', 'pgm_fb')
        sj = A.load_json(os.path.join(sd, 'session.json'))
        self.assertEqual(sj['status'], 'done')
        tog = A.read_toggles(os.path.join(sd, 'toggles.tsv'))
        self.assertGreaterEqual(len(tog), 6)
        self.assertEqual(set(b['state'] for b in tog), {'on', 'off'})
        v = A.load_json(os.path.join(self.R, 'calib', 'verdict.json'))
        self.assertIn(v['admissible_k'], (1, 2))
        self.assertEqual(v['tested'][0]['probes'][0]['status'], 'done')
        self.assertGreaterEqual(v['tested'][0]['probes'][0]['pairs'], 3)


class FakeProc(object):
    """/proc and /sys contents for the Linux-only readers, through atom_batch.read / read_bytes and os.listdir."""

    def __init__(self, files):
        self.files = files

    def read(self, path, default=None):
        return self.files.get(path, default)

    def read_bytes(self, path):
        v = self.files.get(path)
        return v.encode() if isinstance(v, str) else v

    def listdir(self, path):
        assert path == '/proc'
        return sorted(set(p.split('/')[2] for p in self.files if p.startswith('/proc/') and p.split('/')[2].isdigit()))

    def patch(self):
        from unittest import mock
        return [mock.patch.object(A, 'read', self.read), mock.patch.object(A, 'read_bytes', self.read_bytes),
                mock.patch.object(A.os, 'listdir', self.listdir)]


@contextlib.contextmanager
def patched(ps):
    for p in ps:
        p.start()
    try:
        yield
    finally:
        for p in ps:
            p.stop()


class LinuxReaders(unittest.TestCase):
    def test_parsers(self):
        fp = FakeProc({
            '/proc/meminfo': 'MemTotal:       65919000 kB\nMemAvailable:   61000000 kB\nHugePages_Total:       0\n',
            '/proc/stat': 'cpu  1 2 3 4 5 6 7 8 0 0\ncpu0 10 0 10 70 10 0 0 0 0 0\ncpu1 1 1 1 1 1 1 1 1 0 0\n',
            '/proc/interrupts': '           CPU0       CPU1\n  0:         10          2   IO-APIC   2-edge  timer\n'
                                'LOC:        100        200   Local timer interrupts\nERR:          0\n',
            '/proc/vmstat': 'pgpgin 1\npswpout 77\n',
            '/proc/42/stat': '42 (micro bench) (x) S 7 42 42 0 -1',
            '/proc/43/stat': '43 (microbench) R 9 43 43 0 -1',
        })
        with patched(fp.patch()):
            mi = A.meminfo()
            self.assertEqual(mi['MemAvailable'], 61000000 * 1024)
            self.assertEqual(mi['HugePages_Total'], 0)
            j = A.cpu_jiffies()
            self.assertEqual(j[0], (100, 80))
            self.assertEqual(A.interrupts(), {0: 110, 1: 202})
            self.assertEqual(A.pswpout(), 77)
            self.assertEqual(A.proc_table(), {42: (7, 'micro bench) (x'), 43: (9, 'microbench')})

    def test_foreign_and_ours(self):
        cid = 'abc123'
        fp = FakeProc({
            '/proc/10/cmdline': b'/tmp/R/warm/cells/x/a1/bin/microbench\0--keys_file=/d/fb\0--index=pgm\0',
            '/proc/10/environ': b'HOME=/tmp\0ATOM_BATCH_ID=abc123\0',
            '/proc/11/cmdline': b'/other/bin/microbench\0--keys_file=/d/osm\0',
            '/proc/11/environ': None,                        # another user's process: unreadable
            '/proc/12/cmdline': b'tail\0-f\0splice_count.log\0',  # mentions a name, is no benchmark
            '/proc/13/cmdline': b'/x/build/splice_count\0--dataset\0fb\0',
            '/proc/13/environ': b'ATOM_BATCH_ID=zzz\0',     # another campaign's
        })
        with patched(fp.patch()):
            ours, foreign = A.benchmark_procs(cid)
        self.assertEqual(ours, [10])
        self.assertEqual(foreign, ['11:microbench', '13:splice_count'])

    def stub_sched(self):
        class M(object):
            linux = True
            fake_topo = False
            allowed = list(range(16))

            def mem_available(self):
                return 50.0

        class S(object):
            pass
        s = S()
        s.machine, s.R, s.cid, s.events = M(), '/nonexistent', 'abc', []
        s.layout = A.Layout(A.fake_modules('8x2')[0], 4)
        s.event = lambda kind, run=None, **kw: s.events.append(kind)
        return s

    def test_pin_detection(self):
        s = self.stub_sched()
        it = A.gre_item('warm', 'fb', 'pgm', 1, 100000000, 20000000)
        run = A.Run(s, it, 1, 1)
        run.info = s.layout.lane(1)

        class P(object):
            pid = 100

            def poll(self):
                return None
        run.p = P()
        state = {'cpus': '4-5,10-15'}
        from unittest import mock

        def status(pid, tid=None):
            return {'VmRSS': '4000000 kB', 'VmHWM': '9000000 kB', 'Cpus_allowed_list': state['cpus']}
        with mock.patch.object(A, 'proc_status', status), mock.patch.object(A, 'cpu_jiffies', lambda: {}), \
                mock.patch.object(A, 'interrupts', lambda: {}):
            run.poll({200: (100, 'microbench'), 300: (1, 'microbench')}, 10.0)
            self.assertEqual(run.mb, 200)
            self.assertIsNone(run.mono_pin)
            self.assertAlmostEqual(run.rss, 4000000 * 1024 / 1e9)
            state['cpus'] = '4'                        # gre_lite_pin moved the timed thread to lane 2's core
            run.poll({200: (100, 'microbench')}, 12.5)
            self.assertEqual(run.mono_pin, 12.5)
            self.assertEqual(run.phase(), 'timed')
            self.assertEqual(s.events, ['pin'])

    def test_guard(self):
        s = self.stub_sched()
        g = A.Scheduler.__new__(A.Scheduler)
        g.machine, g.swap_hist, g.last_guard, g.event = s.machine, [], -1e9, s.event
        killed = []

        def mk(lane, t0, pinned):
            r = A.Run(s, A.gre_item('warm', 'fb', 'pgm', 1, 1, 0), lane, 1)
            r.info, r.mono_start, r.mono_pin = s.layout.lane(lane), t0, (t0 + 1 if pinned else None)
            r.kill = lambda why, r=r: (killed.append(r.lane), setattr(r, 'guarded', why))
            return r
        running = {0: mk(0, 1.0, True), 1: mk(1, 5.0, False), 2: mk(2, 3.0, False)}
        from unittest import mock
        with mock.patch.object(A, 'pswpout', lambda: 0), contextlib.redirect_stdout(io.StringIO()):
            g.guard(running, 10.0, 100.0)
            self.assertEqual(killed, [])                    # plenty of memory
            g.guard(running, 2.5, 101.0)
            self.assertEqual(killed, [1])                   # soft: the newest run still in setup
            g.guard(running, 2.5, 102.0)
            self.assertEqual(killed, [1])                   # 3 s between kills: the first one frees memory
            g.guard(running, 1.0, 105.0)
            self.assertEqual(killed, [1, 2])                # hard: setups first, never twice the same run


class Shell(unittest.TestCase):
    def test_bash_syntax(self):
        for sh in ('/bin/bash', shutil.which('bash')):
            if sh and os.path.exists(sh):
                for f in ('run.sh', 'scripts/gre_run.sh'):
                    self.assertEqual(subprocess.call([sh, '-n', os.path.join(ROOT, f)]), 0, (sh, f))

    def test_dry_run(self):
        d = tempfile.mkdtemp()
        try:
            rc, out, err = quiet(A.main, ['run', 'warmup,bench', '--dry-run', '--lanes', '4', '--topology', '8x2',
                                          '--mem-available', '64', '--mem-total', '67.5', '--out', d])
            self.assertEqual(rc, 0, err)
            self.assertIn('total at K=4: 632 runs', out)
            rc, out, err = quiet(A.main, ['counts', '--dry-run', '--out', d, '--data', d])
            self.assertEqual(rc, 0, err)
            lines = [l for l in out.splitlines() if 'untimed/' in l]
            first = [l.split()[3] for l in lines[:3]]
            self.assertEqual(first, ['untimed/covid__sort', 'untimed/genome__sort', 'untimed/history__sort'])
        finally:
            shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
