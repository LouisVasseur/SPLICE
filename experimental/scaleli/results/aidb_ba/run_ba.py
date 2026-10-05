#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Before/after NFL & CSV at 200M keys. Serial, resumable, stdlib only.

  run_ba.py plan  <preset> [--shuffle-seed S] > plan.json     presets: aa, overnight, full, sweep, rss_check
  run_ba.py run   plan.json out.jsonl                          runs jobs one at a time, skips finished ones
  run_ba.py show  <cell> <dataset>                             prints the exact bench command

Every job is [dataset, cell, block, instrument, timeout]. All cells of one block share the workload
seed (1000 + block), so result_checksum must agree across cells inside a block.
"""
import json, os, random, subprocess, sys, time

BIN = _os.environ.get("SCALELI_BENCH") or REPO + '/build-fs/scaleli_bench'  # override with your build
S = REPO + '/experimental/scaleli'
GRE = f'{S}/data/external/gre'
DATASETS = ['books', 'covid', 'fb', 'genome', 'history', 'libio', 'osm', 'planet', 'stack', 'wise']
SORTED = {'covid', 'genome', 'history', 'libio', 'planet', 'stack', 'wise'}
PERIOD = {'books': 's64', 'covid': 's64', 'fb': 's512', 'genome': 's64', 'history': 's64', 'libio': 's64',
          'osm': 's512', 'planet': 's512', 'stack': 's64', 'wise': 's64'}  # best.json pick per dataset

def data(d): return f'{GRE}/{d}.sorted' if d in SORTED else f'{GRE}/{d}'
def free(d): return f'{S}/results/aidb_flowv2/flows_free/{d}_{PERIOD[d]}_t2000.txt'
def mono(d): return f'{S}/results/aidb_flowv2/flows_monotone/{d}_mono.txt'

def common(warmup=4_000_000, ops=5_000_000):
    return ['--format', 'sosd', '--dtype', 'uint64', '--load-ratio', '1', '--miss', '0',
            '--query-distribution', 'uniform', '--profile', 'read_only', '--ops', str(ops),
            '--warmup', str(warmup), '--warmup-mode', 'workload', '--prefault', '0', '--chunks', '16',
            '--qos', '1', '--build-threads', '16', '--verify', '0', '--latency', '0']

PK = ['--policy', 'min_bytes', '--routing', 'rank']
def cells(d):
    F = free(d)
    B = PK + ['--root', 'model']                     # BEFORE: learned raw root (falls back to binary itself)
    nfl = ['--flow', F, '--flow-bypass', '1', '--flow-cost', '0']
    C = B + ['--virtual-alpha', '0.1', '--root-alpha', '0.1']   # CSV paper alpha at both levels
    return {
        'B': B, 'B2': B,                              # B2 = identical rebuild, live A/A
        'N': B + nfl,                                 # after NFL: faithful flow, NFL's own switch, root offered it free
        'Nf': B + ['--flow', F, '--flow-bypass', '0', '--flow-cost', '0'],   # stress: flow forced in every region
        'C': C,                                       # after CSV
        'NC': C + nfl,                                # after both
        'SV': ['--index', 'sorted_vector'],           # reference: plain binary search, 16 B/key
        'RAW': ['--policy', 'raw', '--routing', 'rank', '--root', 'model'],          # reference: no compression
        'RC': ['--policy', 'raw', '--routing', 'rank', '--root', 'model', '--root-alpha', '0.1'],  # E2
        'B0': PK,                                     # host default / pilot baseline (binary root)
        'COLD': B,                                    # B with no warm-up (warm-up figure)
        'Nm': B + ['--flow', mono(d), '--flow-bypass', '1', '--flow-cost', '0'],   # realisable monotone flow
        'Cv05': B + ['--virtual-alpha', '0.05'], 'Cv1': B + ['--virtual-alpha', '0.1'], 'Cv2': B + ['--virtual-alpha', '0.2'],
        'Cr004': B + ['--root-alpha', '0.04'], 'Cr01': B + ['--root-alpha', '0.1'],
        'Cr04': B + ['--root-alpha', '0.4'], 'Cr4': B + ['--root-alpha', '4'],
        'C4': B + ['--virtual-alpha', '0.1', '--root-alpha', '4'],   # side cell, planet/osm only
    }

def command(d, cell, block):
    warm = 0 if cell == 'COLD' else 4_000_000
    return [BIN, '--data', data(d), '--seed', str(1000 + block)] + common(warm) + cells(d)[cell]

EXPENSIVE = {'C', 'NC', 'Cv05', 'Cv1', 'Cv2', 'C4'}
def tmo(cell, inst):
    if cell == 'C4' or cell == 'Cr4': return 9000
    if cell in EXPENSIVE: return 3600 if inst else 1800
    return 900

def plan(preset, rng):
    jobs = []
    def add(block, per_dataset, inst_cells=()):
        for d, cl in per_dataset:
            cl = cl[:]; rng.shuffle(cl)
            for c in cl:
                inst = 1 if c in inst_cells else 0
                jobs.append([d, c, block, inst, tmo(c, inst)])
    def shuffled(lst): l = lst[:]; rng.shuffle(l); return l
    if preset == 'aa':            # E1: is run-level timing viable on this host? ~12 min
        for b in range(1, 6): add(b, [('fb', ['B', 'B2'])])
    elif preset == 'e2':          # E2: can any learned layout beat binary search? ~15 min
        for b in range(1, 8): add(b, [('fb', ['RC', 'SV'])])
    elif preset in ('overnight', 'full'):
        core1 = ['B', 'B2', 'N', 'Nf', 'C', 'NC', 'SV']
        inst1 = {'B', 'N', 'Nf', 'C', 'NC', 'SV'}
        extra1 = ['RAW', 'RC', 'B0', 'Nm'] if preset == 'full' else []
        blk = [(d, core1 + extra1 + (['RAW', 'RC'] if preset == 'overnight' and d == 'fb' else [])
                + (['COLD'] if d in ('fb', 'osm', 'planet') else [])) for d in shuffled(DATASETS)]
        add(1, blk, inst1 | {'RAW', 'RC', 'B0', 'Nm'})
        last = 3 if preset == 'overnight' else 16
        for b in range(2, last + 1):
            per = []
            for d in shuffled(DATASETS):
                cl = ['B', 'B2', 'N', 'Nf', 'SV']
                if preset == 'overnight':
                    cl += ['C'] + (['RAW', 'RC'] if d == 'fb' else [])
                else:
                    cl += ['RAW', 'RC'] + (['C'] if b <= 8 else []) + (['NC'] if b <= 4 else [])
                per.append((d, cl))
            add(b, per)
        if preset == 'full':
            for b in range(1, 4): add(b, [(d, ['C4']) for d in ('planet', 'osm')], {'C4'} if b == 1 else set())
    elif preset == 'sweep':       # deterministic alpha sweep, one instrumented build per cell
        for d in DATASETS:
            cl = ['Cr004', 'Cr04']                    # alpha 0.1 at both levels is cell C (block 1)
            if d in ('fb', 'osm', 'planet'): cl += ['Cv05', 'Cv2']
            if d in ('fb', 'osm'): cl += ['Cr4']      # planet alpha 4 already in results/aidb_fullscale
            add(1, [(d, cl)], set(cl))
    else:
        sys.exit(f'unknown preset {preset}')
    return jobs

def others():
    p = subprocess.run(['pgrep', '-f', 'scaleli_bench|scaleli_hardness'], capture_output=True, text=True).stdout.split()
    return [x for x in p if int(x) != os.getpid()]

def other_cpu():
    tot = 0.0
    for l in subprocess.run(['ps', '-Ao', 'pcpu=,comm='], capture_output=True, text=True).stdout.splitlines():
        c, _, name = l.strip().partition(' ')
        try:
            if 'scaleli_bench' not in name: tot += float(c)
        except ValueError: pass
    return tot

def done(out):
    k = set()
    if os.path.exists(out):
        for l in open(out):
            m = json.loads(l)['ba']; k.add((m['dataset'], m['cell'], m['block']))
    return k

def run(planfile, out):
    jobs = json.load(open(planfile))
    for i, (d, c, b, inst, timeout) in enumerate(jobs):
        if (d, c, b) in done(out):
            continue
        while others(): time.sleep(15)          # never two 200M jobs at once
        cmd = command(d, c, b) + ['--instrument', str(inst)]
        meta = {'dataset': d, 'cell': c, 'block': b, 'instrument': inst, 'cmd': cmd,
                'loadavg_start': os.getloadavg(), 'other_cpu_start': other_cpu(),
                'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
        print(time.strftime('%H:%M:%S'), f'[{i+1}/{len(jobs)}] start', d, c, b, inst, flush=True)
        t0 = time.time(); rec = None
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            meta['status'] = 'ok' if p.returncode == 0 else f'rc={p.returncode}: {p.stderr[-400:]}'
            if p.returncode == 0: rec = json.loads(p.stdout)
        except subprocess.TimeoutExpired:
            meta['status'] = f'timeout>{timeout}s'
        meta.update(wall_s=time.time() - t0, loadavg_end=os.getloadavg(), other_cpu_end=other_cpu())
        r = rec or {}; r['ba'] = meta
        with open(out if rec else out + '.failed', 'a') as f: f.write(json.dumps(r) + '\n')
        print(time.strftime('%H:%M:%S'), 'end', d, c, b, meta['status'], f"{meta['wall_s']:.0f}s",
              f"thr={rec['throughput_ops_s']:.0f}" if rec else '', flush=True)
    print('PLAN DONE', flush=True)

if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'plan':
        seed = int(a[a.index('--shuffle-seed') + 1]) if '--shuffle-seed' in a else 20261001
        print(json.dumps(plan(a[1], random.Random(seed))))
    elif a[0] == 'run':
        run(a[1], a[2])
    elif a[0] == 'show':
        print(' '.join(command(a[2], a[1], 1) + ['--instrument', '1']))

# ---- wall-clock estimate (pilot: cheap run ~70 s with instrument at 5M ops; fixed ~31 s + 1 s per M ops)
SMOOTH = {'fb': 184, 'osm': 200, 'planet': 384, 'books': 451, 'genome': 222, 'covid': 256, 'history': 237,
          'libio': 241, 'stack': 300, 'wise': 300}   # fb/osm: bench build_ns; others: hardness CSV wall (upper side)
def est(d, c, inst):
    build = 7
    if c in ('C', 'NC', 'Cv1'): build = SMOOTH[d] + 10
    if c == 'Cv05': build = SMOOTH[d] * 0.5 + 10
    if c == 'Cv2': build = SMOOTH[d] * 2 + 10
    if c in ('Cr004', 'Cr01'): build = 25
    if c == 'Cr04': build = 90
    if c == 'Cr4': build = 96 if d == 'fb' else 2068
    if c == 'C4': build = SMOOTH[d] + 2068 + 300
    fixed = 40 + 5 + 7          # load + workload, warm-up, 5M timed ops
    return fixed + build + (build + 10 if inst else 0)

if __name__ == '__main__' and sys.argv[1] == 'cost':
    for p in sys.argv[2:]:
        j = plan(p, random.Random(20261001)); t = sum(est(d, c, i) for d, c, b, i, _ in j)
        from collections import Counter
        n = Counter(c for _, c, _, _, _ in j)
        print(f'{p:10s} jobs={len(j):4d} est={t/3600:5.1f} h  per-cell runs (all datasets): {dict(n)}')
