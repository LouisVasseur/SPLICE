#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Serial before/after NFL & CSV pilot at 200M keys. Resumable; one bench at a time."""
import json, os, subprocess, sys, time
OUT = os.path.dirname(os.path.abspath(__file__))
JSONL = os.path.join(OUT, 'pilot.jsonl')
BIN = _os.environ.get("SCALELI_BENCH") or REPO + '/build-fs/scaleli_bench'  # override with your build
S = REPO + '/experimental/scaleli'
DATA = {'fb': f'{S}/data/external/gre/fb', 'osm': f'{S}/data/external/gre/osm', 'planet': f'{S}/data/external/gre/planet.sorted'}
FREE = {d: f'{S}/results/aidb_flowv2/flows_free/{d}_s512_t2000.txt' for d in DATA}
JOINT = {d: f'{S}/results/aidb_joint/flows_root/{d}_joint.txt' for d in DATA}
COMMON = ['--format', 'sosd', '--dtype', 'uint64', '--load-ratio', '1', '--miss', '0', '--query-distribution', 'uniform',
          '--profile', 'read_only', '--ops', '5000000', '--warmup', '2000000', '--warmup-mode', 'workload', '--prefault', '1',
          '--chunks', '16', '--qos', '1', '--build-threads', '16', '--verify', '0', '--latency', '0', '--instrument', '1']
PK = ['--policy', 'min_bytes', '--routing', 'rank']
def cells(d):
    F, J = FREE[d], JOINT[d]
    return {
        'B0':   PK,
        'L0':   PK + ['--root', 'model'],                      # extra control: learned root, no NFL / no CSV
        'N1':   PK + ['--flow', F, '--flow-bypass', '0'],
        'N2':   PK + ['--flow', F, '--flow-bypass', '1'],
        'N3':   PK + ['--root', 'model', '--flow', F, '--flow-cost', '0'],
        'C1':   PK + ['--virtual-alpha', '0.1'],
        'C2':   PK + ['--root', 'model', '--root-alpha', '0.04'],
        'F1':   PK + ['--virtual-alpha', '0.1', '--root', 'model', '--root-alpha', '0.04', '--flow', F],
        'J1':   PK + ['--root', 'model', '--root-alpha', '4', '--flow', J, '--flow-cost', '0'],
        'J1s':  PK + ['--root', 'model', '--root-alpha', '0.04', '--flow', J, '--flow-cost', '0'],  # extra: joint bend at the C2 budget
        'R0raw': ['--policy', 'raw', '--routing', 'rank'],
        'R0sv': ['--index', 'sorted_vector'],
    }
def done_keys():
    k = set()
    if os.path.exists(JSONL):
        for l in open(JSONL):
            r = json.loads(l); k.add((r['pilot']['dataset'], r['pilot']['cell'], r['pilot']['seed']))
    return k
def others():
    p = subprocess.run(['pgrep', '-f', 'scaleli_bench|scaleli_hardness'], capture_output=True, text=True).stdout.split()
    return [x for x in p if int(x) != os.getpid()]
def run(dataset, cell, seed, timeout):
    if (dataset, cell, seed) in done_keys():
        print('skip', dataset, cell, seed, flush=True); return
    waited = 0
    while others():
        time.sleep(15); waited += 15
    cmd = [BIN, '--data', DATA[dataset], '--seed', str(seed)] + COMMON + cells(dataset)[cell]
    la0 = os.getloadavg(); t0 = time.time()
    print(time.strftime('%H:%M:%S'), 'start', dataset, cell, seed, flush=True)
    status, rec = 'ok', None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0: status = f'rc={p.returncode}: {p.stderr[-500:]}'
        else: rec = json.loads(p.stdout)
    except subprocess.TimeoutExpired:
        status = f'timeout>{timeout}s'
    wall = time.time() - t0
    meta = {'dataset': dataset, 'cell': cell, 'seed': seed, 'status': status, 'wall_s': wall, 'loadavg_start': la0,
            'loadavg_end': os.getloadavg(), 'waited_for_other_bench_s': waited, 'other_bench_at_end': others(),
            'nice': os.nice(0), 'started': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t0)), 'cmd': cmd}
    out = rec or {}
    out['pilot'] = meta
    with open(JSONL if rec else os.path.join(OUT, 'pilot_failed.jsonl'), 'a') as f: f.write(json.dumps(out) + '\n')
    print(time.strftime('%H:%M:%S'), 'end', dataset, cell, seed, status, f'{wall:.0f}s',
          f"thr={rec['throughput_ops_s']:.0f}" if rec else '', flush=True)
if __name__ == '__main__':
    # usage: run_pilot.py PLAN.json  where plan = [[dataset, cell, seed, timeout], ...]
    for d, c, s, t in json.load(open(sys.argv[1])):
        run(d, c, s, t)
    print('PLAN DONE', flush=True)
