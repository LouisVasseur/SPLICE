from __future__ import annotations
import importlib.util, json, math, os, pathlib, struct, subprocess, sys, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
DATA=load('datasets');VIRTUAL=load('virtual_points_lab');THEORY=load('theory');SUMMARY=load('summarize');FLOW=load('train_flow')

class ToolsTests(unittest.TestCase):
    def test_feasibility(self):
        self.assertTrue(THEORY.slope_interval([1,2],[1,1],[8,8])['feasible_in_continuous_relaxation'])
        self.assertFalse(THEORY.slope_interval([1,10],[1,1],[8,8])['feasible_in_continuous_relaxation'])
        with self.assertRaises(ValueError):THEORY.slope_interval([0],[1],[8])
    def test_uniform_scaling_not_smoothing(self):
        x=[1,2,3,40,80];y=[0,1,2,3,4]
        def loss(y):
            mx=sum(x)/len(x);my=sum(y)/len(y);a=sum((u-mx)*(v-my) for u,v in zip(x,y))/sum((u-mx)**2 for u in x);b=my-a*mx
            return sum(((a*u+b-v)/(max(y)-min(y)))**2 for u,v in zip(x,y))
        self.assertAlmostEqual(loss(y),loss([.125*v+11 for v in y]),places=13)
    def test_virtual_greedy(self):
        keys=[1,2,3,4,5,10,20,26,27,30];before=VIRTUAL.fit_loss(keys)[0]
        greedy=VIRTUAL.greedy(keys,3);exact=VIRTUAL.exhaustive(keys,3)
        self.assertLessEqual(greedy['sse_all'],before);self.assertLessEqual(exact['sse_all'],greedy['sse_all']+1e-9)
        self.assertFalse(set(keys)&set(greedy['virtual_points']))
    def test_dataset_sampling_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            src=pathlib.Path(d)/'a';src.write_bytes(struct.pack('<Q',100)+b''.join(struct.pack('<Q',i*8) for i in range(100)))
            report=DATA.inspect(src,'uint64');self.assertTrue(report['sorted']);self.assertEqual(report['count'],100)
            for mode in ['uniform','window','strided']:
                dest=pathlib.Path(d)/mode;DATA.sample(src,dest,20,'uint64',mode,13);self.assertEqual(DATA.inspect(dest,'uint64')['count'],20)
            src.write_bytes(src.read_bytes()[:-1])
            with self.assertRaises(ValueError):DATA.header(src,'uint64')
    def test_flow_trainer_and_conflicts(self):
        self.assertEqual(FLOW.tail_conflict_degree([float(i) for i in range(500)]),0)
        self.assertGreater(FLOW.tail_conflict_degree([i*1e-6 if i<450 else float(i) for i in range(500)]),0)
        keys=[1000+int(math.exp(i/60)*7) for i in range(400)];keys=sorted(set(keys))
        flow,info=FLOW.train(keys,shifts=16,steps=30,lr=0.05,seed=3,barrier=1e-3,log=lambda s:None,monotone=True)
        self.assertTrue(math.isfinite(info['best_nll']));self.assertLessEqual(info['best_nll'],info['history'][0]+1e-9)
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'w.txt';flow.save(path);lines=path.read_text().splitlines()
            self.assertEqual(lines[0].split(),['2','2','2']);self.assertEqual(len(lines),8)
    def test_cluster_bootstrap(self):
        e,l,h=SUMMARY.paired_interval([(1,2),(1,2),(2,2),(2,2)]);self.assertEqual(e,2);self.assertEqual(l,2);self.assertEqual(h,2)

class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary=pathlib.Path(os.environ.get('SCALELI_BENCH',ROOT/'build/scaleli_bench'))
        if not cls.binary.exists():raise RuntimeError('build scaleli_bench first, or set SCALELI_BENCH')
    def run_bench(self,*args):
        p=subprocess.run([str(self.binary),'--n','2048','--ops','1200','--seed','7',*args],capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stderr);return json.loads(p.stdout)
    def test_all_workloads(self):
        for profile in ['read_only','read_heavy','scan_heavy','write_heavy','churn','append','shift']:
            with self.subTest(profile=profile):self.assertTrue(self.run_bench('--profile',profile,'--policy','adaptive')['verified'])
    def test_query_distributions(self):
        for distribution in ['uniform','zipf','hotspot','moving_hotspot']:
            with self.subTest(distribution=distribution):self.assertTrue(self.run_bench('--profile','churn','--query-distribution',distribution)['verified'])
    def test_bulk_sampling_and_order(self):
        for sampling in ["uniform","prefix"]:
            for order in ["shuffled","sorted"]:
                r=self.run_bench("--profile","write_heavy","--bulk-sampling",sampling,"--insert-order",order,"--load-ratio","0.1")
                self.assertTrue(r["verified"]);self.assertEqual(r["bulk_sampling"],sampling);self.assertEqual(r["insert_order"],order)
    def test_pairing(self):
        results=[self.run_bench('--index',i,'--profile','read_heavy') for i in ['scaleli','ordered_map','sorted_vector']]
        self.assertEqual(len({r['trace_fingerprint'] for r in results}),1);self.assertEqual(len({r['result_checksum'] for r in results}),1)
    def test_full_width_and_duplicates(self):
        for distribution in ['near_u64','duplicates','linear','locally_hard','staircase']:
            with self.subTest(distribution=distribution):self.assertTrue(self.run_bench('--distribution',distribution)['verified'])
    def test_cli_rejects_invalid_inputs(self):
        for args in [['--read','0.5'],['--dtype','bad','--data','x'],['--n','-1'],['--unknown','1']]:
            p=subprocess.run([str(self.binary),*args],capture_output=True,text=True);self.assertNotEqual(p.returncode,0)
    def test_learnability_options(self):
        with tempfile.TemporaryDirectory() as d:
            keys=pathlib.Path(d)/'k.sosd';weights=pathlib.Path(d)/'w.txt'
            self.assertTrue(self.run_bench('--distribution','lognormal','--dump-keys',str(keys))['verified'])
            p=subprocess.run([sys.executable,str(ROOT/'tools/train_flow.py'),str(keys),'--output',str(weights),'--sample','512','--steps','20'],capture_output=True,text=True,timeout=300)
            self.assertEqual(p.returncode,0,p.stderr)
            for args in (['--virtual-alpha','0.1'],['--flow',str(weights),'--flow-bypass','0'],['--flow',str(weights),'--virtual-alpha','0.1','--fusion','auto'],['--flow',str(weights),'--virtual-alpha','0.2','--routing','byte'],['--flow',str(weights),'--policy','adaptive','--profile','churn']):
                with self.subTest(args=args):
                    r=self.run_bench('--distribution','lognormal',*args);self.assertTrue(r['verified']);self.assertIn('learnability',r)
                    if '--virtual-alpha' in args and '--fusion' not in args:self.assertGreater(r['learnability']['virtual_points'],0)
                    if '--virtual-alpha' in args:self.assertLessEqual(r['learnability']['rank_sse_after'],r['learnability']['rank_sse_before']+1e-9)
                    if '--flow-bypass' in args:self.assertEqual(r['learnability']['flow_regions'],r['learnability']['regions'])
                    if '--fusion' in args:
                        ch=r['learnability']['choices'];self.assertEqual(sum(ch.values()),r['learnability']['regions']);self.assertLessEqual(r['learnability']['cost_selected_mean'],r['learnability']['cost_none_mean']+1e-9)
            p=subprocess.run([str(self.binary),'--n','100','--virtual-alpha','1.5'],capture_output=True,text=True);self.assertNotEqual(p.returncode,0)
    def test_latency_skip(self):
        # --latency 0 skips the per-operation replay: empty latency blocks, throughput and learnability still reported.
        r=self.run_bench('--profile','read_heavy','--latency','0','--virtual-alpha','0.1')
        self.assertFalse(r['latency_pass']);self.assertTrue(r['verified']);self.assertGreater(r['throughput_ops_s'],0)
        self.assertEqual(len(r['latency_ns']),6)
        for name,block in r['latency_ns'].items():
            with self.subTest(op=name):self.assertEqual(block,{'count':0,'p50':0,'p95':0,'p99':0,'max':0})
        self.assertEqual([b['count'] for b in r['phase_latency_ns']],[0,0]);self.assertEqual(r['compaction_operation_latency_ns']['count'],0)
        self.assertIsInstance(r['learnability'],dict);self.assertEqual(r['learnability']['keys'],r['initial_rows']);self.assertGreater(r['learnability']['virtual_points'],0)
        rows=[dict(r,dataset='d',variant=v,repeat=0) for v in ('a','b')]
        summary=SUMMARY.summarize(rows,'a');self.assertEqual(len(summary),2)
        self.assertEqual(summary[0]['read_hit_p99_ns_median'],0);self.assertGreater(summary[0]['throughput_ops_s_median'],0);self.assertEqual(summary[0]['paired_throughput_speedup'],1)
        full=self.run_bench('--profile','read_heavy');self.assertTrue(full['latency_pass']);self.assertGreater(full['latency_ns']['read_hit']['count'],0)
        self.assertEqual(full['trace_fingerprint'],r['trace_fingerprint']);self.assertEqual(full['result_checksum'],r['result_checksum'])
        for index in ('sorted_vector','ordered_map'):
            with self.subTest(index=index):self.assertIsNone(self.run_bench('--index',index,'--latency','0')['learnability'])
        p=subprocess.run([str(self.binary),'--n','100','--latency','2'],capture_output=True,text=True);self.assertNotEqual(p.returncode,0)
    def test_u32_u64_import(self):
        with tempfile.TemporaryDirectory() as d:
            for dtype,w,fmt in [('uint32',4,'I'),('uint64',8,'Q')]:
                path=pathlib.Path(d)/dtype;path.write_bytes(struct.pack('<Q',1000)+b''.join(struct.pack('<'+fmt,i*16) for i in range(1000)))
                r=self.run_bench('--data',str(path),'--dtype',dtype);self.assertEqual(r['unique_rows'],1000);self.assertTrue(r['verified'])
if __name__=='__main__':unittest.main()
