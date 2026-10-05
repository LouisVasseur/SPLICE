"""tools/aidb_report.py: renders with full synthetic inputs, with an empty results dir and with corrupt files."""
from __future__ import annotations
import csv, html.parser, importlib.util, json, pathlib, subprocess, sys, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('aidb_report', ROOT / 'tools' / 'aidb_report.py'); REPORT = importlib.util.module_from_spec(spec); spec.loader.exec_module(REPORT)
DATASETS = ['books', 'fb', 'osm', 'wise']; VARIANTS = ['packed_rank', 'packed_rank_flow', 'packed_rank_vp10']
SWEEP_COLS = ['dataset', 'profile', 'variant', 'runs', 'all_verified', 'throughput_ops_s_median', 'read_hit_p99_ns_median', 'fence_probes_per_operation', 'correction_distance_per_operation',
              'rank_sse_ratio', 'virtual_points_per_key', 'flow_region_fraction', 'preprocess_ns_per_key', 'initial_bytes_per_key', 'build_ns_per_key', 'baseline', 'paired_throughput_speedup',
              'seed_bootstrap_low', 'seed_bootstrap_high']

class Check(html.parser.HTMLParser):
    """Collects <h2> text and verifies that container tags open and close in order."""
    TRACKED = {'html', 'head', 'body', 'section', 'table', 'thead', 'tbody', 'tr', 'td', 'th', 'svg', 'figure', 'div', 'nav', 'details', 'pre', 'span', 'a', 'p', 'h2', 'h3'}
    def __init__(self): super().__init__(); self.h2, self.stack, self.errors, self.in_h2 = [], [], [], False
    def handle_starttag(self, tag, attrs):
        if tag in self.TRACKED: self.stack.append(tag)
        if tag == 'h2': self.in_h2 = True; self.h2.append('')
    def handle_endtag(self, tag):
        if tag in self.TRACKED: self.stack.pop() if self.stack and self.stack[-1] == tag else self.errors.append(tag)
        if tag == 'h2': self.in_h2 = False
    def handle_data(self, data):
        if self.in_h2: self.h2[-1] += data

def synthetic(d: pathlib.Path) -> None:
    hard, tp, prov, scores = {}, {v: {} for v in VARIANTS}, {}, {}
    for i, ds in enumerate(DATASETS):
        base = {'rmse': 1e5 * (i + 1), 'max_error': 3e6 * (i + 1), 'conflict_degree': 7 * (i + 1) + 1, 'pla_32': 100000 * (i + 1) ** 4, 'pla_4096': 300 * (i + 1)}
        scale = lambda f, extra={}: dict({k: v * f for k, v in base.items()}, **extra)
        hard[ds] = {'full': base, 'full_flow': scale(.6), 'sample': scale(.01), 'sample_flow': scale(.005), 'sample_csv': scale(.004, {'virtual_points': 2000, 'smoothing_ns': 3e7})}
        if i == 0: hard[ds]['sample_flow_csv'] = scale(.003)
        for j, v in enumerate(VARIANTS): tp[v][ds] = 10.0 + i + 2 * j
        prov[ds] = {'url': f'https://example.invalid/{ds}', 'sha256': '0123abcd' * 8, 'bytes': 1600000008, 'count': 200000000, 'retrieved_utc': '2026-09-21T00:00:00Z', 'sample': {'sample_count': 2000000, 'mode': 'uniform', 'seed': 42, 'source_count': 200000000, 'path': f'data/samples/{ds}'},
                    'sort_audit': {'sorted': True, 'duplicates': 0, 'written': None} if i % 2 == 0 else {'sorted': False, 'duplicates': 3, 'written': f'/x/{ds}.sorted', 'written_keys': 199999997, 'sha256': 'feedbeef' * 8, 'bytes': 1599999984}}
        hard[ds]['full_flow'].update({'keys': 200000000, 'duplicates': 120000000 * (i == 1), 'unordered_pairs': 0}); hard[ds]['sample_flow'].update({'keys': 2000000, 'duplicates': 5, 'unordered_pairs': 2})
    for scope in ('full', 'sample'):
        scores[scope] = {'datasets': DATASETS, 'variants': VARIANTS, 'metrics': {m: {'dims': m.split('·'), 'coverage': 1 - .4 * (len(m.split('·')) - 1), 'conformance': (k % 7) / 7 - .3,
                         'per_variant': {v: (k + j) % 5 / 5 - .5 for j, v in enumerate(VARIANTS)}, 'comparable_pairs': 6 - k % 4, 'incomparable_pairs': k % 4} for k, m in enumerate(REPORT.METRICS)}}
    for name, obj in (('hardness.json', hard), ('throughput.json', tp), ('provenance.json', prov), ('scores.json', scores)): (d / name).write_text(json.dumps(obj))
    (d / 'sweep').mkdir(); (d / 'flows').mkdir(); (d / 'hardness').mkdir()
    with (d / 'sweep' / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_COLS); w.writeheader()
        for i, ds in enumerate(DATASETS):
            for j, v in enumerate(VARIANTS):
                w.writerow(dict.fromkeys(SWEEP_COLS, 0.5) | {'dataset': f'{ds}_2M_uniform_s42', 'profile': 'read_only', 'variant': v, 'runs': 3, 'all_verified': 'True', 'throughput_ops_s_median': (10 + i + j) * 1e6,
                           'baseline': 'packed_rank', 'paired_throughput_speedup': 1 + .1 * j, 'seed_bootstrap_low': 'nan' if j == 1 else .95 + .1 * j, 'seed_bootstrap_high': 1.05 + .1 * j, 'virtual_points_per_key': .1 * (j == 2), 'preprocess_ns_per_key': 12.5 * (j > 0)})
    (d / 'flows' / 'training_report.json').write_text(json.dumps([{'dataset': ds, 'training_keys': 4096, 'steps': 200, 'train_seconds': 2.3 + i, 'best_nll': 1.5 - .1 * i, 'tail_conflict_degree_raw': 40, 'tail_conflict_degree_transformed': 9, 'unordered_transformed_pairs': 0, 'monotone': True} for i, ds in enumerate(DATASETS)]))
    (d / 'hardness_details.json').write_text(json.dumps({ds: {'sample_flow': {'fmcd': {'conflict_degree_lipp_epsilon': 13, 'ut_epsilon': 6.8e-7}}} for ds in DATASETS}))
    (d / 'hardness' / 'books_sample_flow_csv.json').write_text(json.dumps({'data': 'x', 'keys': 2000000, 'transformed': {'transform_ns': 5e8, 'unordered_pairs': 0}, 'smoothed': {'virtual_points': 20000, 'smoothing_ns': 8e8, 'regions': 489}}))
    (d / 'pipeline.log').write_text('2026-09-21T00:00:00Z start\n2026-09-21T00:01:00Z done\n')

def parse(text: str) -> Check:
    c = Check(); c.feed(text); c.close(); return c

class ReportTests(unittest.TestCase):
    def check(self, text):
        c = parse(text); self.assertEqual(c.h2, REPORT.HEADINGS); self.assertEqual(c.errors, []); self.assertEqual(c.stack, [])
        self.assertIn('clean-room', text); self.assertNotIn('<script', text); self.assertNotIn('src=', text); self.assertNotIn('<link', text); return c
    def test_full_inputs_via_cli(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d); synthetic(root); out = root / 'report.html'
            p = subprocess.run([sys.executable, str(ROOT / 'tools' / 'aidb_report.py'), '--results', str(root), '--output', str(out)], capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr); text = out.read_text(); self.check(text)
            self.assertNotIn('Not available', text); self.assertIn('log10 scale', text); self.assertIn('linear scale', text)
            self.assertIn('marker-end', text); self.assertIn('Pareto', text); self.assertEqual(text.count('<section'), 7)
            for m in ('RMSE·CD·PLA-32', 'PLA-32·PLA-4096', 'Δ sample_flow_csv vs sample_flow', 'books_sample_flow_csv', 'books \u00b7 sample_csv', 'pipeline.log', '2,000,000', '200,000,000'): self.assertIn(m, text)
            for m in ('sorted on disk', 'feedbeef', '199,999,997', 'none needed', 'Degeneracy of the transformed', '60.0%', 'CD (LIPP 1e-6)', 'conflict_degree_lipp_epsilon', 'Coincident metrics', '×'): self.assertIn(m, text)
            self.assertLess(text.count('>RMSE·ME·CD<'), 3)  # coincident 3-dim metrics collapse into one ×n marker instead of stacked labels
            self.assertEqual(len(REPORT.METRICS), 25)
    def test_empty_results_dir(self):
        with tempfile.TemporaryDirectory() as d:
            text = REPORT.build(pathlib.Path(d) / 'missing'); self.check(text); self.assertGreaterEqual(text.count('Not available'), 6)
    def test_corrupt_and_partial_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d); (root / 'hardness.json').write_text('{not json'); (root / 'throughput.json').write_text(json.dumps({'packed_rank': {'books': 12.5, 'fb': None}}))
            (root / 'scores.json').write_text(json.dumps({'full': {'metrics': {'CD': {'coverage': 1, 'conformance': 'nan'}}}})); (root / 'provenance.json').write_text('[]')
            text = REPORT.build(root); self.check(text); self.assertIn('12.5', text); self.assertIn('Not available: hardness.json', text)
if __name__ == '__main__': unittest.main()
