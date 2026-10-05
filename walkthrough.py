#!/usr/bin/env python3
"""Offline numbered supervisor-meeting walkthrough. No third-party Python packages."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments.lessons import TITLES,run_lesson,load_keys
from experiments.report import build

def execute(command,logname):
 p=subprocess.run([str(x) for x in command],cwd=ROOT,text=True,capture_output=True)
 (ROOT/'results/local').mkdir(parents=True,exist_ok=True)
 (ROOT/'results/local'/logname).write_text(p.stdout+p.stderr)
 print(p.stdout[-5000:]);
 if p.returncode:raise RuntimeError(p.stderr[-4000:])

def check():
 if sys.version_info<(3,10):raise RuntimeError('Python 3.10 or newer required')
 expected=json.loads((ROOT/'evidence/dense_sparse_fixture_uint64.manifest.json').read_text())['sha256']
 actual=hashlib.sha256((ROOT/'evidence/dense_sparse_fixture_uint64').read_bytes()).hexdigest()
 if actual!=expected:raise RuntimeError('fixture checksum mismatch')
 print(f'PASS fixture SHA-256; {len(load_keys()):,} distinct sorted synthetic keys')
 from experiments.lessons import lesson05
 lesson05(ROOT/'results/saved');print('PASS virtual-point toy objective and exhaustive ceiling')
 execute([sys.executable,'experiments/audit_archived.py'],'archive_audit.log')
 manifest=ROOT/'MANIFEST.sha256'
 if manifest.exists():
  count=0
  for line in manifest.read_text().splitlines():
   digest,name=line.split('  ',1);p=ROOT/name
   if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise RuntimeError('package file changed: '+name)
   count+=1
  print(f'PASS package manifest: {count} files')
 print('Evidence checks passed. These are not a substitute for compiling/testing the native code.')

def prepare():
 check()
 for program in ['cmake','ctest']:
  if not shutil.which(program):raise RuntimeError(f'{program} not found. On macOS install CMake; see docs/REHEARSAL.md')
 execute(['cmake','-S','.', '-B','build','-DCMAKE_BUILD_TYPE=Release','-DSCALELI_EXTERNAL=OFF','-DSCALELI_NATIVE=OFF'],'configure.log')
 execute(['cmake','--build','build','--parallel','2'],'build.log')
 execute(['ctest','--test-dir','build','--output-on-failure'],'ctest.log')
 env=os.environ.copy();env['SCALELI_BENCH']=str(ROOT/'build/experimental/scaleli/scaleli_bench')
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'experimental/scaleli/tests'),'-p','test_tools.py','-v'],cwd=ROOT/'experimental/scaleli',env=env,text=True,capture_output=True)
 (ROOT/'results/local/python_tests.log').write_text(p.stdout+p.stderr);print(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError('Python tests failed; inspect results/local/python_tests.log')
 print('READY: build and tests passed. Next: python3 walkthrough.py demo all')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['list','check','prepare','demo','show']);p.add_argument('lesson',nargs='?',default='all');p.add_argument('--output',default='results/local')
 args=p.parse_args();out=ROOT/args.output
 if args.action=='list':
  for id,title in TITLES.items():print(id,title)
 elif args.action=='check':check()
 elif args.action=='prepare':prepare()
 elif args.action=='show':
  base=ROOT/'results/saved'
  if args.lesson!='all':
   path=base/args.lesson/'result.json';print(path.read_text())
  print('Offline fallback:',base/'report.html')
 else:
  ids=list(TITLES) if args.lesson=='all' else [args.lesson.zfill(2)]
  for id in ids:
   if id not in TITLES:raise ValueError('unknown lesson; use list')
   print('\n'+id+' — '+TITLES[id],flush=True);start=time.perf_counter()
   r=run_lesson(id,out/id); print('PASS | '+r['status']+' | '+f'{time.perf_counter()-start:.2f}s',flush=True)
   if id=='02':print(f"Binary: {r['binary_ns_per_query_median']:.2f} ns/query; RadixSpline-derived: {r['radix_ns_per_query_median']:.2f} ns/query; window <= {r['max_search_window_keys']} keys")
   if id=='05':print(f"Augmented SSE: {r['before_sse']:.4f} -> greedy {r['greedy']['sse_all']:.4f}; exhaustive {r['exhaustive']['sse_all']:.4f}")
   if id=='09':
    if r['status'].startswith('NOT RUN'):print('  '+r['status'])
    else:
     for m in r['metrics']:print(f"  {m['metric']:<24} Cov {m['coverage']:.2f}  Conf {m['conformance']:.2f}  Conf[{r['variant']}] {m['conformance_variant']:.2f}")
   if id=='08':
    for x in r['summary']:print(f"  {x['variant']:<24} fence/op {x['fence_probes_per_op']:.2f}  SSE ratio {x['rank_sse_ratio']:.3f}  flow regions {x['flow_regions']:.0f}/{x['regions']:.0f}  virtual {x['virtual_points']:.0f}  {x['throughput_ops_s']/1e6:.2f} Mops/s")
  print('\nResults:',build(ROOT,out))
if __name__=='__main__':
 try:main()
 except Exception as e:print('STOP:',e,file=sys.stderr);sys.exit(1)
