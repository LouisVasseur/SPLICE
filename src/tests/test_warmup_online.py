#!/usr/bin/env python3
"""Tests of the online (streaming) warm-up detector in scripts/warmup_detect.py on synthetic chunk series. Needs numpy
(skipped without it, like warmup_detect.py itself):

    python3 src/tests/test_warmup_online.py
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'scripts'))
try:
    import numpy as np
    import warmup_detect as W
except ImportError:          # numpy missing: warmup_detect.py cannot run here either
    W = None


@unittest.skipIf(W is None, 'numpy not installed')
class OnlineWarmup(unittest.TestCase):
    def series(self, fn, seed=3, noise='quiet'):
        return W.synth_series(np.random.default_rng(seed), fn, W.NOISE[noise])

    def test_no_look_ahead(self):
        # The chunks after the stop must not change anything: cut and stop are the same on the truncated trace.
        start, n, y, _ = self.series(lambda t: 0.5 * np.exp(-t / 3e4))
        cut, info = W.det_online(start, n, y)
        self.assertIsNotNone(cut)
        k = int(np.searchsorted(start + n, info['stop'])) + 1
        cut2, info2 = W.det_online(start[:k], n[:k], y[:k])
        self.assertEqual((cut, info['stop']), (cut2, info2['stop']))
        self.assertLessEqual(cut, info['stop'])

    def test_flat_stops_early_near_zero(self):
        start, n, y, _ = self.series(lambda t: 0.0 * t)
        cut, info = W.det_online(start, n, y)
        self.assertLess(cut, 1000)            # the first chunks are 64-512 lookups: one or two may be off by chance
        self.assertLess(info['seen_share'], 0.05)

    def test_transient_end_found(self):
        # exp mid: 0.5 exp(-t / 3e4) is below eps = 1% after 3e4 ln 50 = 1.17e5 calls
        start, n, y, _ = self.series(lambda t: 0.5 * np.exp(-t / 3e4))
        cut, info = W.det_online(start, n, y)
        true = 3e4 * math.log(0.5 / W.EPS)
        self.assertLess(abs(math.log10((cut + 1e3) / (true + 1e3))), 0.5)
        self.assertGreater(info['stop'], cut)
        self.assertLess(info['seen_share'], 0.1)

    def test_in_analyze_not_in_consensus(self):
        start, n, y, _ = self.series(lambda t: 0.0 * t)
        res = W.analyze_series(start, n, y)
        self.assertIn('online', res['cuts'])
        self.assertNotIn('online', W.CORE)


if __name__ == '__main__':
    unittest.main()
