#!/usr/bin/env python3
# SPDX-License-Identifier: MIT (SPLICE)
"""Run the GRE experiment set on the DIAS Atom in parallel pinned lanes, as fast as the validity rules allow.

    atom_batch.py doctor  [--out R]                          topology, lanes, memory, THP -> R/machine.txt
    atom_batch.py smoke   [--out R] [--lanes K]              real lanes, untimed, 10M keys -> R/smoke/SMOKE.md
    atom_batch.py calib   [--out R] [--k 4,3,2] [--dry-run]   solo vs K-way on the probe cells -> R/calib/
    atom_batch.py verdict [--out R]                          admissible K from R/calib -> R/calib/verdict.json
    atom_batch.py run RECIPES [--out R] [--lanes K] [--rounds N] [--datasets L] [--indexes L] [--dry-run]
    atom_batch.py counts  [--out R] [--dry-run]               untimed: .sorted copies, splice_count, nf_gate
    atom_batch.py status | merge [--only solo|parallel] | anchors   [--out R]

RECIPES: warmup (trace_<index>, W=0, 30M lookups, GRE_TRACE=geo:64:65536, 2 rounds), bench (warm W=20M and cold W=0,
100M lookups, 3 rounds, plus solo anchor runs when K > 1), or warmup,bench in one queue (packs best). Defaults:
the ten datasets, the indexes of docs/ATOM_FAST.md. R defaults to the newest results/atom_* (else atom_<today>).

Lanes (read from sysfs at run time, never assumed): CPUs that share an L2 form a module (Goldmont: 2 cores, 2 MB).
The module of the lowest CPU is the house module: this scheduler and Louis's shells. Lane j owns one whole other
module; its timed core is the module's first CPU (core 2 for lane 1, as docs/PROTOCOL.md) and its sibling stays idle
while the lane is timed. Each run's whole process (GRE's TBB sorts, the SPLICE/SCALE-LI build threads) is confined to
its lane's module plus the pool (the modules that host no lane), so no run ever touches another lane's module; GRE
then pins the timed thread to the lane's core (--pin). Untimed multi-threaded work therefore shares only the pool
and DRAM with timed runs, which is what the calibration measures before any K > 1 is used.

Admission: a run starts on a free lane when MemAvailable minus the remaining growth of the running runs (predicted
peak minus RSS before the pin, 1 GB after it) minus its predicted peak stays above the headroom (max 4 GB, 6% of
MemTotal); a head that does not fit reserves the time when it will (EASY backfill: others may start only if they end
before it or leave it room). Rounds are a soft barrier (round r+1 starts once every item of round r has started),
largest memory x time first, so LIPP leads each round. Exclusive items (solo anchors, LIPP cells never measured,
guard-killed retries, untimed counts) wait for an empty machine. A memory guard kills the newest run still in setup
when MemAvailable falls below 3 GB, and below 1.5 GB or on swap-out the newest run whatever its phase if none is in
setup, so the kernel's OOM killer never acts; the run is retried alone once.

Provenance: every attempt is one gre_run.sh OUTDIR, R/<arm>/cells/<ds>__<index>__r<round>/a<n>/, plus atom.json:
lane, module, core, sibling, CPU mask, K, the runs alive and their phase at the start, pin time (the timed thread's
affinity changing to its core, polled from /proc), MemAvailable (start, min), swap-out, foreign benchmark processes,
sibling and core busy share and interrupts over the timed window, THP, governor, binary sha256. `merge` turns the
cells into one gre_run.sh-shaped OUTDIR per arm (runs.tsv with lane, core, concurrency and solo columns, logs/ as
links) that gre_report.py and warmup_detect.py read unchanged; `--only solo|parallel` separates the two.

Calibration (calib, verdict): for K in --k (largest first), probes trace_pgm and trace_lipp on fb (W=20M, 200M
lookups, fixed 65,536-call chunks) run on lane 1 while lanes 2..K loop real GRE runs (setups, builds, lookups) under
the campaign's masks, started together at the probe's pin. The scheduler stops and resumes the load lanes together
(SIGSTOP/SIGCONT) in 5 s blocks, in seeded random order within pairs, so solo and K-way alternate inside one probe
process: the pairing removes the run-to-run offset that a solo-vs-parallel comparison of whole runs would need
hundreds of runs to average out. Inflation = exp(mean over pairs of ln(median chunk latency, K-way block / solo
block)) - 1, with a t interval. K is admissible when its upper 95% bound is at most --bound (1%) for both probes and
at least 3 pairs whose K-way block had a load in setup or build do not exceed it on average. `run` refuses K > 1
without such a verdict for the same binary. `smoke` runs a few minutes of real lanes (10M keys) and checks the
Linux-only parts: masks, pin, VmHWM, the tag, trace chunks on the lane core and on the scheduler's clock.

Python 3.6, standard library only (it runs on the Atom). Linux for real runs; anywhere for --dry-run, merge,
verdict, anchors and status (ATOM_BATCH_TEST=1 lets the tests drive fake binaries on macOS).
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRE_RUN = os.path.join(ROOT, 'scripts', 'gre_run.sh')
DATASETS = ('fb', 'osm', 'books', 'covid', 'genome', 'history', 'libio', 'planet', 'stack', 'wise')
RAW_SORTED = ('fb', 'osm', 'books')       # the other seven raw files are unsorted: GRE sorts 200M keys on load
BENCH_IX = ('splice', 'pgm', 'lipp', 'sortedarray', 'alex', 'btree', 'artunsync')
TRACE_IX = ('nullindex', 'splice', 'pgm', 'btree', 'alex', 'sortedarray', 'lipp', 'artunsync', 'scaleli_b', 'scaleli_n')
ANCHORS = (('fb', 'pgm'), ('fb', 'lipp'), ('osm', 'sortedarray'), ('fb', 'splice'))
PROBES = (('fb', 'pgm'), ('fb', 'lipp'))  # the DIAS reference cells; LIPP has the largest TLB footprint measured there
LOAD_DS = ('wise', 'osm', 'genome', 'books', 'history')
LOAD_IX = ('sortedarray', 'btree', 'pgm')  # real cells, chosen for a small peak so K-1 of them fit beside a probe
TRACE_ENV = {'GRE_TRACE': 'geo:64:65536', 'GRE_TRACE_CPU': '1'}
PROBE_ENV = {'GRE_TRACE': 'fixed:65536', 'GRE_TRACE_CPU': '1'}
LOAD_SIZES = ((-1, 100000000, 20000000), (100000000, 100000000, 20000000), (50000000, 100000000, 20000000),
              (50000000, 30000000, 0), (25000000, 30000000, 0))   # (table_size, ops, warmup) of a load run
CLEAN_ENV = ('GRE_TRACE', 'GRE_TRACE_FAULTS', 'GRE_TRACE_CPU', 'SPLICE_ARGS', 'SCALELI_ARGS', 'GRE_RUN_BUSY')
NEVER = '[a]tom_batch_no_such_process'    # GRE_RUN_BUSY: the scheduler does the busy check itself (foreign only)
BENCH_NAMES = ('splice_count', 'scaleli_hardness', 'scaleli_bench', 'nf_gate')
GB = 1e9
OPS_GB = 20e-9                            # peak RSS per lookup in the operation list (Mac logs: 16 B + sampling)
GUARD_GB, GUARD_HARD_GB = 3.0, 1.5
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df):
    keys = [k for k in sorted(T975) if k <= df]
    return T975[keys[-1]] if keys else float('nan')


def die(msg, code=2):
    sys.stderr.write('atom_batch.py: %s\n' % msg)
    sys.exit(code)


def note(msg):
    sys.stdout.write('%s %s\n' % (time.strftime('%F %T'), msg))
    sys.stdout.flush()


def mono():
    return time.monotonic()  # CLOCK_MONOTONIC on Linux: the clock of the trace's steady_clock (trace_t0_mono_ns)


def read(path, default=None):
    try:
        with open(path, errors='replace') as f:
            return f.read()
    except (IOError, OSError):
        return default


def read_bytes(path):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except (IOError, OSError):
        return None


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return default


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def hms(s):
    s = int(round(s))
    return '%d:%02d:%02d' % (s // 3600, s % 3600 // 60, s % 60)


def cpulist(cpus):
    """[2, 3, 10, 11, 12] -> '2-3,10-12' (taskset -c / sysfs format)."""
    out, cpus = [], sorted(cpus)
    i = 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else '%d-%d' % (cpus[i], cpus[j]))
        i = j + 1
    return ','.join(out)


def parse_cpulist(s):
    out = []
    for part in (s or '').strip().split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ------------------------------------------------------------------------------------------------ planning model
# Atom estimates (docs/ATOM_FAST.md): wall seconds per run, warm (W=20M + 100M) / cold (W=0, 100M) / trace (W=0, 30M),
# setups on 14 CPUs, datasets in DATASETS order. Every estimate is replaced by measurements as runs finish.
def _w(s):
    return [tuple(float(x) for x in c.split('/')) for c in s.split()]


WALL = {
    'pgm': _w('198/179/110 225/202/120 171/155/100 176/162/112 231/209/131 179/164/113 180/166/114 214/194/124 '
              '172/158/110 176/162/112'),
    'splice': _w('188/174/123 196/181/127 166/157/127 169/162/138 192/179/132 168/161/137 172/164/138 202/189/143 '
                 '160/154/132 166/159/136'),
    'lipp': _w('237/232/160 266/257/173 266/257/173 178/181/141 258/252/180 207/207/155 180/184/144 248/242/170 '
               '148/154/123 209/209/156'),
    'alex': _w('357/332/245 331/308/231 218/208/174 284/275/245 254/240/190 296/287/257 304/294/259 308/291/232 '
               '214/206/179 268/260/232'),
    'btree': _w('245/216/113 ' * 3 + '262/232/128 ' * 7),
    'sortedarray': _w('345/299/137 ' * 3 + '359/313/151 ' * 7),
    'artunsync': _w('279/257/178 ' * 3 + '293/271/192 ' * 7),
    'scaleli_b': _w('451/391/182 ' * 3 + '501/435/205 ' * 7),
    'scaleli_n': _w('454/394/185 ' * 3 + '523/452/212 ' * 7),
    'nullindex': _w('81/78/68 ' * 3 + '95/92/82 ' * 7),
}
# Peak RSS (VmHWM) in GB of a warm run (120M operations); Mac logs plus margin, LIPP per dataset.
PEAK = {'pgm': 12.65, 'splice': 13.4, 'sortedarray': 12.9, 'btree': 13.4, 'alex': 14.5, 'scaleli_b': 15.9,
        'scaleli_n': 15.9, 'artunsync': 19.0, 'nullindex': 9.1,
        'lipp': dict(zip(DATASETS, (42.1, 56.0, 56.0, 29.0, 46.3, 31.0, 25.7, 45.2, 22.1, 31.7)))}
# Never measured anywhere (the 64 GB Mac could not run them, or LIPP's RT_ASSERT stopped covid): alone until measured.
UNCERTAIN = (('lipp', 'osm'), ('lipp', 'books'), ('lipp', 'covid'))
# Extra setup seconds when GRE's two parallel_sorts get fewer CPUs (sorted raw file, unsorted raw file).
SETUP_PENALTY = ((1, 60, 115), (2, 38, 76), (4, 15, 30), (8, 4, 8), (14, 0, 0))
CMD_WALL = {'sort': 150.0, 'splice_count': 900.0, 'nf_gate': 150.0}


def base_index(idx):
    return idx[6:] if idx.startswith('trace_') else idx


def setup_penalty(ncpu, ds):
    col = 1 if ds in RAW_SORTED else 2
    pts = SETUP_PENALTY
    if ncpu >= pts[-1][0]:
        return 0.0
    for (a, pa, ua), (b, pb, ub) in zip(pts, pts[1:]):
        if a <= ncpu <= b:
            va, vb = (pa, pb) if col == 1 else (ua, ub)
            return va + (vb - va) * (ncpu - a) / float(b - a)
    return float(pts[0][col])


def model_wall(it, ncpu=14):
    if it['kind'] == 'cmd':
        return CMD_WALL.get(it['step'], 300.0)
    b = base_index(it['idx'])
    row = WALL.get(b, WALL['pgm'])[DATASETS.index(it['ds'])] if it['ds'] in DATASETS else WALL.get(b, WALL['pgm'])[0]
    warm, cold, trace = row
    lat = max((cold - trace) / 70e6, 1e-7)  # s per lookup from the 70M lookups between the trace and cold runs
    wall = cold + (it['ops'] + it['warmup'] - 100e6) * lat
    if it.get('table_size', -1) > 0:          # a prefix of the file: setup and build shrink with it
        setup = max(cold - 100e6 * lat, 30.0)
        wall -= setup * (1.0 - min(1.0, it['table_size'] / 200e6))
    return max(wall + setup_penalty(ncpu, it['ds']), 5.0)


def model_peak(it):
    if it['kind'] == 'cmd':
        return 0.0
    b = base_index(it['idx'])
    p = PEAK.get(b, PEAK['pgm'] + 2.0)
    if isinstance(p, dict):
        p = p.get(it['ds'], max(p.values()))
    fixed = p - 120e6 * OPS_GB
    if it.get('table_size', -1) > 0:
        fixed = max(0.5, fixed * min(1.0, it['table_size'] / 200e6))
    return fixed + (it['ops'] + it['warmup']) * OPS_GB


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


class Predictor(object):
    """Model estimates, replaced by measurements: a cell's own measured wall and peak, else the model times the
    measured/model ratio of the same index."""

    def __init__(self, records=()):
        self.walls, self.peaks, self.rw, self.rp, self.floors = {}, {}, {}, {}, {}
        for r in records:
            self.add(r)

    def add(self, rec):
        it = rec.get('item') or {}
        if it.get('kind') != 'gre' or it.get('table_size', -1) > 0:
            return
        ncpu = len(rec.get('mask') or []) or 14
        w = rec.get('wall_s')
        if rec.get('status') == 'pass' and w:
            key = (it['idx'], it['ds'], it['ops'], it['warmup'])
            self.walls.setdefault(key, []).append(float(w))
            self.rw.setdefault(it['idx'], []).append(float(w) / model_wall(it, ncpu))
        g = rec.get('rss_peak_gb') or rec.get('hwm_gb')
        if g:
            norm = float(g) - (it['ops'] + it['warmup']) * OPS_GB
            k = (base_index(it['idx']), it['ds'])
            if rec.get('status') == 'pass':
                self.peaks[k] = max(self.peaks.get(k, 0.0), norm)
                self.rp.setdefault(k[0], []).append(float(g) / model_peak(it))
            else:
                # a crashed or killed run stopped short of its peak: its VmHWM only raises the estimate, and the cell
                # stays unmeasured (a LIPP cell that was never measured keeps running alone)
                self.floors[k] = max(self.floors.get(k, 0.0), norm)

    def measured_peak(self, it):
        return (base_index(it['idx']), it['ds']) in self.peaks

    def wall(self, it, ncpu=14):
        if it['kind'] == 'cmd':
            return model_wall(it)
        m = self.walls.get((it['idx'], it['ds'], it['ops'], it['warmup']))
        if m:
            return sum(m) / len(m)
        r = median(self.rw.get(it['idx'], [])) or 1.0
        return model_wall(it, ncpu) * r

    def peak(self, it):
        """Reservation in GB: 5% and 0.5 GB above the measured or predicted VmHWM."""
        if it['kind'] == 'cmd':
            return 0.0
        k = (base_index(it['idx']), it['ds'])
        if k in self.peaks and it.get('table_size', -1) <= 0:
            p = self.peaks[k] + (it['ops'] + it['warmup']) * OPS_GB
        else:
            p = model_peak(it) * max(1.0, median(self.rp.get(k[0], [])) or 1.0)
        if k in self.floors and it.get('table_size', -1) <= 0:
            p = max(p, self.floors[k] + (it['ops'] + it['warmup']) * OPS_GB)
        return p * 1.05 + 0.5 + it.get('peak_extra', 0.0)


# ------------------------------------------------------------------------------------------------ machine
def read_modules(sysfs='/sys/devices/system/cpu', allowed=None):
    """L2 modules [(cpu, ...)], complete flags and isolated CPUs. A module is complete when all its CPUs are online,
    allowed and not isolated: only complete modules can host a lane (its sibling must be ours to keep idle)."""
    online = parse_cpulist(read(os.path.join(sysfs, 'online')) or '')
    if not online:
        raise RuntimeError('no CPUs in %s/online' % sysfs)
    iso = set(parse_cpulist(read(os.path.join(sysfs, 'isolated')) or ''))
    groups = {}
    for c in online:
        l2 = None
        for d in sorted(glob.glob(os.path.join(sysfs, 'cpu%d' % c, 'cache', 'index*'))):
            if (read(os.path.join(d, 'level')) or '').strip() == '2':
                l2 = tuple(parse_cpulist(read(os.path.join(d, 'shared_cpu_list')) or str(c)))
                break
        groups.setdefault(l2 or (c,), []).append(c)
    mods = []
    for l2, cs in sorted(groups.items()):
        usable = [c for c in cs if (allowed is None or c in allowed) and c not in iso]
        if usable:
            mods.append({'cpus': sorted(usable), 'complete': sorted(usable) == sorted(l2) and set(l2) <= set(online)})
    return mods, sorted(iso)


def fake_modules(spec):
    """'8x2' (the Atom: 8 modules of 2 CPUs) or explicit modules '0-3/4-9/10-15' (the Mac: E cluster, 2 P clusters)."""
    m = re.match(r'^(\d+)x(\d+)$', spec or '')
    if m:
        n, w = int(m.group(1)), int(m.group(2))
        return [{'cpus': list(range(i * w, (i + 1) * w)), 'complete': True} for i in range(n)], []
    try:
        mods = [sorted(parse_cpulist(x)) for x in (spec or '').split('/') if x.strip()]
    except ValueError:
        mods = []
    flat = [c for x in mods for c in x]
    if not mods or not all(mods) or len(flat) != len(set(flat)):
        die("--topology must look like 8x2 (modules x CPUs per module) or 0-3/4-9/10-15 (one CPU list per module)")
    return [{'cpus': x, 'complete': True} for x in mods], []


class Layout(object):
    """House module (lowest CPU), K lane modules in ascending order, and the pool (the other complete modules)."""

    def __init__(self, mods, k, avoid=()):
        mods = [m for m in mods if not set(m['cpus']) & set(avoid)]
        if not mods:
            die('no usable CPU module')
        self.house = min(mods, key=lambda m: m['cpus'][0])['cpus']
        rest = [m for m in mods if m['cpus'] != self.house]
        lanes = [m['cpus'] for m in rest if m['complete']]
        if k > len(lanes):
            die('K=%d lanes asked, but only %d complete L2 modules are free of the house module %s'
                % (k, len(lanes), cpulist(self.house)))
        self.k = k
        self.lanes = lanes[:k]
        self.pool = sorted(c for m in rest if m['cpus'] not in self.lanes for c in m['cpus'])
        self.all = sorted(c for m in mods for c in m['cpus'])

    def lane(self, j):
        mod = self.lanes[j]
        return {'lane': j + 1, 'module': mod, 'core': mod[0], 'siblings': mod[1:],
                'mask': sorted(mod + self.pool)}

    def describe(self):
        parts = ['lane %d: core %d (module %s)' % (j + 1, m[0], cpulist(m)) for j, m in enumerate(self.lanes)]
        return '%s; pool %s; house %s; each run confined to its lane module + pool (%d CPUs)' % (
            ', '.join(parts), cpulist(self.pool) or '-', cpulist(self.house), len(self.lanes[0]) + len(self.pool))

    def as_json(self):
        return {'k': self.k, 'house': self.house, 'lanes': self.lanes, 'pool': self.pool}


def meminfo():
    s = read('/proc/meminfo')
    if s is None:
        return None
    d = {}
    for line in s.splitlines():
        k, _, v = line.partition(':')
        p = v.split()
        if p and p[0].isdigit():
            d[k] = int(p[0]) * (1024 if len(p) > 1 and p[1] == 'kB' else 1)
    return d


def pswpout():
    for line in (read('/proc/vmstat') or '').splitlines():
        if line.startswith('pswpout '):
            return int(line.split()[1])
    return 0


def proc_table():
    """pid -> (ppid, comm); empty without /proc."""
    out = {}
    try:
        names = os.listdir('/proc')
    except OSError:
        return out
    for n in names:
        if not n.isdigit():
            continue
        s = read('/proc/%s/stat' % n)
        if not s:
            continue
        r = s.rfind(')')
        f = s[r + 2:].split()
        if len(f) > 1:
            out[int(n)] = (int(f[1]), s[s.find('(') + 1:r])
    return out


def proc_status(pid, tid=None):
    s = read('/proc/%d/task/%d/status' % (pid, tid) if tid else '/proc/%d/status' % pid)
    if s is None:
        return None
    d = {}
    for line in s.splitlines():
        k, _, v = line.partition(':')
        d[k] = v.strip()
    return d


def kb_gb(v):
    try:
        return int(v.split()[0]) * 1024 / GB
    except (AttributeError, ValueError, IndexError):
        return None


def cpu_jiffies():
    d = {}
    for line in (read('/proc/stat') or '').splitlines():
        if line.startswith('cpu') and line[3:4].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:9]]
            d[int(f[0][3:])] = (sum(v), v[3] + v[4])   # total, idle + iowait
    return d


def interrupts():
    s = read('/proc/interrupts')
    if not s:
        return {}
    lines = s.splitlines()
    cpus = [int(x[3:]) for x in lines[0].split() if x.startswith('CPU')]
    tot = dict((c, 0) for c in cpus)
    for line in lines[1:]:
        f = line.split()[1:1 + len(cpus)]
        for c, x in zip(cpus, f):
            if x.isdigit():
                tot[c] += int(x)
    return tot


def thp():
    v = (read('/sys/kernel/mm/transparent_hugepage/enabled') or '').strip()
    m = re.search(r'\[(\w+)\]', v)
    return m.group(1) if m else (v or 'unknown')


def governor(core):
    return (read('/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor' % core) or 'unknown').strip()


def boot_id():
    return (read('/proc/sys/kernel/random/boot_id') or 'unknown').strip()


def load1():
    s = read('/proc/loadavg')
    if s:
        return float(s.split()[0])
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def benchmark_procs(cid):
    """(ours, foreign) benchmark processes: GRE (--keys_file=) or SPLICE's own tools, as run.sh's busy_pids. A process
    is ours when its environment carries ATOM_BATCH_ID=<cid>; one whose environment cannot be read is foreign."""
    ours, foreign = [], []
    tag = ('ATOM_BATCH_ID=%s' % cid).encode()
    try:
        names = os.listdir('/proc')
    except OSError:
        return ours, foreign
    me = os.getpid()
    for n in names:
        if not n.isdigit() or int(n) == me:
            continue
        raw = read_bytes('/proc/%s/cmdline' % n)
        if not raw:
            continue
        argv = raw.split(b'\0')
        a0 = os.path.basename(argv[0].decode('utf-8', 'replace'))
        if not (a0 in BENCH_NAMES or any(a.startswith(b'--keys_file=') for a in argv[1:])):
            continue
        env = read_bytes('/proc/%s/environ' % n)
        if env is not None and tag in env.split(b'\0'):
            ours.append(int(n))
        else:
            foreign.append('%s:%s' % (n, a0))
    return ours, foreign


def tagged_pids(cid):
    """Every process carrying ATOM_BATCH_ID=<cid> (orphans of a killed scheduler)."""
    out = []
    tag = ('ATOM_BATCH_ID=%s' % cid).encode()
    try:
        names = os.listdir('/proc')
    except OSError:
        return out
    for n in names:
        if n.isdigit() and int(n) != os.getpid():
            env = read_bytes('/proc/%s/environ' % n)
            if env and tag in env.split(b'\0'):
                out.append(int(n))
    return out


def fs_type(path):
    best, typ = '', 'unknown'
    for line in (read('/proc/mounts') or '').splitlines():
        f = line.split()
        if len(f) > 2 and (path == f[1] or path.startswith(f[1].rstrip('/') + '/')) and len(f[1]) > len(best):
            best, typ = f[1], f[2]
    return typ


def allowed_cpus(sysfs):
    """The CPUs this user may run on: the cpuset, not this shell's mask. A shell moved to the house module
    (taskset -cp 0,1 $$) would otherwise leave no lane. Widening our own affinity to every online CPU, the kernel
    grants exactly the cpuset's share of it (sched_setaffinity intersects with the cpuset); the scheduler then
    confines itself to the house module again."""
    try:
        online = parse_cpulist(read(os.path.join(sysfs, 'online')) or '')
        if online:
            os.sched_setaffinity(0, set(online))
    except (OSError, ValueError):
        pass
    return sorted(os.sched_getaffinity(0))


class Machine(object):
    """What the scheduler reads about the machine; overridable for --dry-run and tests."""

    def __init__(self, a):
        self.linux = os.path.isdir('/proc/self') and sys.platform.startswith('linux')
        self.allowed = allowed_cpus(a.sysfs) if hasattr(os, 'sched_getaffinity') else None
        self.fake_topo = bool(a.topology) or not self.linux    # a fake layout: never apply its CPU masks
        if a.topology:
            self.mods, self.iso = fake_modules(a.topology)
            self.topo_src = 'fake %s' % a.topology
        elif self.linux:
            self.mods, self.iso = read_modules(a.sysfs, set(self.allowed) if self.allowed else None)
            self.topo_src = a.sysfs
        else:
            self.mods, self.iso = fake_modules('8x2')
            self.topo_src = 'fake 8x2 (not Linux: the Atom layout)'
        mi = meminfo() if self.linux and a.mem_available is None else None
        self.fake_mem = mi is None
        self.mem_total = a.mem_total if a.mem_total else (mi['MemTotal'] / GB if mi else 67.5)
        self.mem_avail0 = a.mem_available if a.mem_available is not None else (mi['MemAvailable'] / GB if mi else 64.0)
        self.headroom = max(4.0, 0.06 * self.mem_total)

    def mem_available(self):
        if self.fake_mem:
            return self.mem_avail0
        mi = meminfo()
        return mi['MemAvailable'] / GB if mi else self.mem_avail0

    def topology_key(self):
        return [m['cpus'] for m in self.mods]


# ------------------------------------------------------------------------------------------------ items and recipes
def gre_item(arm, ds, idx, rnd, ops, warmup, env=None, exclusive=False, table_size=-1, why=''):
    return {'kind': 'gre', 'arm': arm, 'ds': ds, 'idx': idx, 'round': rnd, 'ops': ops, 'warmup': warmup,
            'env': dict(env or {}), 'exclusive': exclusive, 'table_size': table_size, 'why': why,
            'id': '%s/%s__%s__r%d' % (arm, ds, idx, rnd)}


def cmd_item(step, name, argv, rnd=0):
    cmd_item.seq += 1
    return {'kind': 'cmd', 'arm': 'untimed', 'step': step, 'ds': name, 'idx': step, 'round': rnd, 'argv': argv,
            'seq': cmd_item.seq,
            'exclusive': True, 'ops': 0, 'warmup': 0, 'env': {}, 'table_size': -1, 'why': 'untimed',
            'id': 'untimed/%s__%s' % (name, step)}


cmd_item.seq = 0


def item_dir(R, it):
    cell = '%s__%s' % (it['ds'], it['idx']) if it['kind'] == 'cmd' else '%s__%s__r%d' % (it['ds'], it['idx'], it['round'])
    return os.path.join(R, it['arm'], 'cells', cell)


def arm_name(base, thp_mode):
    # docs/PROTOCOL.md 5b: under THP [always] there is no 4 KiB arm; name the folders so nobody reads them as one
    return base + ('_thpalways' if thp_mode == 'always' else '')


def csv_list(s, default):
    return [x for x in s.split(',') if x] if s else list(default)


def recipe_items(names, a, k, thp_mode):
    items = []
    dss = csv_list(a.datasets, DATASETS)
    for ds in dss:
        if ds not in DATASETS:
            die('unknown dataset %s' % ds)
    # --ops / --warmup / --table-size shrink every run: a smoke of the machinery, never the protocol's numbers
    ops = getattr(a, 'ops', None)
    w = getattr(a, 'warmup', None)
    w = 20000000 if w is None else w
    ts = getattr(a, 'table_size', None) or -1
    for name in names:
        if name == 'warmup':
            # --trace-indexes; a warmup-only run also takes --indexes (in warmup,bench it is the bench's list)
            tl = getattr(a, 'trace_indexes', None) or (a.indexes if names == ['warmup'] else None)
            ixs = ['trace_' + base_index(i) for i in csv_list(tl, TRACE_IX)]
            for r in range(1, (a.rounds or 2) + 1):
                for ds in dss:
                    for ix in ixs:
                        items.append(gre_item(arm_name('trace', thp_mode), ds, ix, r, ops or 30000000, 0, TRACE_ENV,
                                              table_size=ts))
        elif name == 'bench':
            ixs = csv_list(a.indexes, BENCH_IX)
            for r in range(1, (a.rounds or 3) + 1):
                if k > 1 and not a.no_anchors:
                    for ds, ix in ANCHORS:
                        if ds in dss and ix in ixs:
                            items.append(gre_item(arm_name('warm_solo', thp_mode), ds, ix, r, ops or 100000000, w,
                                                  exclusive=True, why='solo anchor', table_size=ts))
                for ds in dss:
                    for ix in ixs:
                        items.append(gre_item(arm_name('warm', thp_mode), ds, ix, r, ops or 100000000, w,
                                              table_size=ts))
                        items.append(gre_item(arm_name('cold', thp_mode), ds, ix, r, ops or 100000000, 0,
                                              table_size=ts))
        else:
            die("unknown recipe '%s' (warmup, bench)" % name)
    return items


def counts_items(R, a, data):
    hard = os.path.join(ROOT, 'build', 'scaleli_hardness')
    items = []
    dss = csv_list(a.datasets, DATASETS)
    for ds in dss:
        if ds not in RAW_SORTED and not os.path.exists(os.path.join(data, ds + '.sorted')):
            # prepare_data.sh's sort, without its sample step (Python >= 3.8): splice_count and nf_gate read <ds>.sorted
            items.append(cmd_item('sort', ds, ['bash', '-c', 'set -e; "$0" --data "$1" --dtype uint64 --check-sorted 0 '
                                                            '--sort-only 1 --write-sorted "$1.sorted.part" > /dev/null; '
                                                            'mv "$1.sorted.part" "$1.sorted"', hard,
                                                os.path.join(data, ds)]))
    for ds in dss:
        # run.sh counts' flags (kept in step with it); its COUNTS.md table is written once at the end
        items.append(cmd_item('splice_count', ds, [os.path.join(ROOT, 'build', 'splice_count'), '--dataset', ds, '--data',
                                                   data, '--out', os.path.join(R, 'untimed', 'splice_count'),
                                                   '--threads', '16', '--grid', 'pruned', '--verify', '--parity',
                                                   '--perturb']))
    for ds in dss:
        items.append(cmd_item('nf_gate', ds, [os.path.join(ROOT, 'scripts', 'nf_gate.sh'),
                                              os.path.join(ROOT, 'build', 'nf_gate'),
                                              os.path.join(R, 'untimed', 'nfgate', ds), ds, '--threads', '16']))
    return items


def order_items(items, pred, ncpu):
    """Rounds in order; within a round exclusive items first (one drain), then cells by memory x time, largest first
    (LIPP leads), a cell's arms adjacent with their order alternating by round."""
    cell_w = {}
    for it in items:
        key = (it['round'], it['ds'], base_index(it['idx']))
        cell_w[key] = max(cell_w.get(key, 0.0), pred.peak(it) * pred.wall(it, ncpu))

    def key(it):
        if it['kind'] == 'cmd':
            return (it['round'], 0, 0.0, '', '', it['seq'])   # untimed steps keep their order (sort before use)
        ck = (it['round'], it['ds'], base_index(it['idx']))
        arm = it['arm']
        flip = it['round'] % 2 == 0
        arm_rank = {'warm': 1, 'cold': 2, 'trace': 3}.get(arm.split('_')[0], 0)
        if flip and arm_rank in (1, 2):
            arm_rank = 3 - arm_rank
        return (it['round'], 0 if it['exclusive'] else 1, -cell_w[ck], it['ds'], it['idx'], arm_rank)
    return sorted(items, key=key)


# ------------------------------------------------------------------------------------------------ admission
def exclusive(it, measured):
    """Alone on the machine: solo anchors, untimed work, guard retries, and cells whose peak was never measured."""
    return it['exclusive'] or ((base_index(it['idx']), it['ds']) in UNCERTAIN and
                               (base_index(it['idx']), it['ds']) not in measured)


def select(pending, running, free_lanes, free_mem, now, pred, ncpu, history, measured, k=None):
    """Next (item, lane) to start, 'drain' (an exclusive head waits for an empty machine) or None.
    running: dicts with item, end (predicted), reserved (GB)."""
    # a retry waits for a lane other than the one it failed on (when there is one)
    pending = [it for it in pending if pick_lane(it, free_lanes, history, k) is not None] if free_lanes else []
    if not free_lanes or not pending:
        return None
    if any(exclusive(r['item'], measured) for r in running):
        return None          # an exclusive run has the machine to itself until it ends
    head = pending[0]
    if exclusive(head, measured):
        return 'drain' if running else (head, pick_lane(head, free_lanes, history))
    ph = pred.peak(head)
    if ph <= free_mem:
        return head, pick_lane(head, free_lanes, history)
    if not running:
        return head, pick_lane(head, free_lanes, history)   # alone: the guard decides whether it fits
    # EASY backfill: the head reserves the earliest time enough memory frees; others may go only if they end by then
    # or leave the head its memory at that time.
    m, t_res = free_mem, None
    for r in sorted(running, key=lambda r: r['end']):
        m += r['reserved']
        if m >= ph:
            t_res = r['end']
            break
    if t_res is None:
        return 'drain'       # it fits only on an empty machine: backfilling would starve it
    for c in pending[1:]:
        if exclusive(c, measured):
            continue
        pc = pred.peak(c)
        if pc > free_mem:
            continue
        if now + pred.wall(c, ncpu) <= t_res or m - pc >= ph:
            return c, pick_lane(c, free_lanes, history)
    return None


def pick_lane(it, free_lanes, history, k=None):
    """Prefer a lane this cell has not used yet (lane effects average out over rounds), else the lowest. None: every
    free lane is one this item failed on, and another lane exists (k lanes in all): wait for it."""
    used = history.get((it['ds'], it['idx']), set())
    avoid = set(it.get('avoid_lanes', ()))
    if k is not None and avoid and all(j in avoid for j in free_lanes) and len(avoid & set(range(k))) < k:
        return None
    for pref in (lambda j: j not in used and j not in avoid, lambda j: j not in avoid, lambda j: True):
        for j in sorted(free_lanes):
            if pref(j):
                return j
    return sorted(free_lanes)[0]


def eligible(queue):
    """Soft round barrier: only the lowest round that still has items waiting."""
    if not queue:
        return []
    r = min(it['round'] for it in queue)
    return [it for it in queue if it['round'] == r]


def simulate(items, k, budget, pred, ncpu):
    """Dry run: the same admission on predicted walls and peaks. Returns [(start, end, lane, item)] and makespan."""
    queue = list(items)
    running, sched, history, now = [], [], {}, 0.0
    measured = set(pred.peaks)
    while queue or running:
        free = [j for j in range(k) if all(r['lane'] != j for r in running)]
        free_mem = budget - sum(r['reserved'] for r in running)
        sel = select(eligible(queue), running, free, free_mem, now, pred, ncpu, history, measured)
        if sel is not None and sel != 'drain':
            it, lane = sel
            queue.remove(it)
            w = pred.wall(it, ncpu if it['kind'] == 'gre' else 16)
            running.append({'item': it, 'lane': lane, 'start': now, 'end': now + w, 'reserved': pred.peak(it)})
            history.setdefault((it['ds'], it['idx']), set()).add(lane)
            continue
        if not running:
            break
        r = min(running, key=lambda r: r['end'])
        now = r['end']
        running.remove(r)
        if r['item']['kind'] == 'gre':
            measured.add((base_index(r['item']['idx']), r['item']['ds']))
        sched.append((r['start'], r['end'], r['lane'], r['item']))
    return sorted(sched, key=lambda s: s[0]), now


# ------------------------------------------------------------------------------------------------ attempts on disk
def attempt_records(R):
    out = []
    for p in glob.glob(os.path.join(R, '*', 'cells', '*', 'a*', 'atom.json')):
        rec = load_json(p)
        if rec:
            rec['_dir'] = os.path.dirname(p)
            out.append(rec)
    return out


def attempt_dirs(cell_dir):
    out = []
    for d in glob.glob(os.path.join(cell_dir, 'a*')):
        m = re.match(r'^a(\d+)$', os.path.basename(d))
        if m and os.path.isdir(d):
            out.append((int(m.group(1)), d))
    return sorted(out)


def parse_attempt(adir, it):
    """gre_run.sh's row and the numbers of its log (gre_report.py's fields)."""
    res = {}
    rows = (read(os.path.join(adir, 'runs.tsv')) or '').splitlines()
    if len(rows) > 1:
        h = rows[0].split('\t')
        row = dict(zip(h, rows[-1].split('\t')))
        res['exit_code'] = row.get('exit_code')
        res['run_wall_s'] = float(row['wall_s']) if row.get('wall_s') else None
    if it['kind'] != 'gre':
        return res
    log = os.path.join(adir, 'logs', '%s__%s__r1.log' % (it['ds'], it['idx']))
    pats = {'throughput': r'^Throughput = (\d+)', 'rss_peak_bytes': r'^rss_peak_bytes: (-?\d+)',
            'build_ns': r'^build_ns: (\d+)', 'warmup_ns': r'^warmup_ns: (\d+)', 'pinned_core': r'^pinned_core: (-?\d+)',
            'success_read': r'^success_read: (\d+)', 'index_rss_bytes': r'^index_rss_bytes: (-?\d+)',
            'trace_t0_mono_ns': r'^trace_t0_mono_ns: (-?\d+)'}
    comp = dict((k, re.compile(v)) for k, v in pats.items())
    for line in (read(log) or '').splitlines():
        if line[:1] not in 'Trbwpsit':
            continue
        for k, c in comp.items():
            m = c.match(line)
            if m:
                res[k] = int(m.group(1))
    if res.get('rss_peak_bytes', -1) > 0:
        res['rss_peak_gb'] = res['rss_peak_bytes'] / GB
    return res


# ------------------------------------------------------------------------------------------------ live runs
class Run(object):
    """One attempt: gre_run.sh (or an untimed command) in its own session, confined to its CPU mask."""

    def __init__(self, sched, it, lane, attempt):
        self.s, self.it, self.lane, self.attempt = sched, it, lane, attempt
        self.dir = os.path.join(item_dir(sched.R, it), 'a%d' % attempt)
        self.mb = self.mb_seen = None
        self.mono_pin = None
        self.rss = 0.0
        self.hwm = 0.0
        self.guarded = None
        self.guard_alone = False
        self.foreign = set()
        self.mem_min = None
        self.snap_pin = None
        self.mask_seen = None     # the microbench's CPU list at first sight (before the pin): the mask really applied
        self.threads_max = 0      # its thread count: TBB sizes its pool from the mask
        self.tagged_seen = False  # the foreign scan saw it as ours (ATOM_BATCH_ID in its environment)
        self.pin = -1

    def phase(self):
        if self.it['kind'] == 'cmd':
            return 'untimed'
        return 'timed' if self.mono_pin is not None else 'setup'

    def start(self, others):
        s, it = self.s, self.it
        os.makedirs(self.dir)
        if it['kind'] == 'gre':
            L = s.layout.lane(self.lane)
            self.info = L
            mask = L['mask']
            self.pin = pin_for(s.machine, L)
            cmd = ['bash', GRE_RUN, self.dir, '--gre', s.frozen, '--data', s.data, '--datasets', it['ds'],
                   '--indexes', it['idx'], '--repeats', '1', '--ops', str(it['ops']), '--warmup', str(it['warmup']),
                   '--pin', str(self.pin), '--build-threads', str(s.a.build_threads)]
            if it.get('table_size', -1) > 0:
                cmd += ['--table-size', str(it['table_size'])]
            if s.a.splice_plan:
                cmd += ['--splice-plan', s.a.splice_plan]
        else:
            self.info = {'lane': 0, 'module': [], 'core': -1, 'siblings': [], 'mask': s.machine.allowed or []}
            self.pin = -1
            mask = s.machine.allowed
            cmd = it['argv']
        env = dict(os.environ)
        for v in CLEAN_ENV:
            env.pop(v, None)
        env.update(it.get('env') or {})
        env.update({'ATOM_BATCH_ID': s.cid, 'ATOM_BATCH_ATTEMPT': os.path.relpath(self.dir, s.R),
                    'GRE_RUN_BUSY': NEVER, 'GRE_RUN_BIN_LINK': '1', 'OMP_NUM_THREADS': '1'})
        self.env_set = sorted(k for k in (it.get('env') or {}))
        self.out = open(os.path.join(self.dir, 'gre_run.out' if it['kind'] == 'gre' else 'out.log'), 'w')
        aff = set(mask) if (mask and hasattr(os, 'sched_setaffinity') and not s.machine.fake_topo) else None

        def pre():
            if aff:
                os.sched_setaffinity(0, aff)
        self.mono_start = mono()
        self.t_start = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        self.mem_start = s.machine.mem_available()
        self.swap_start = pswpout()
        self.load_start = load1()
        self.others = others
        self.p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=self.out, stderr=subprocess.STDOUT,
                                  start_new_session=True, preexec_fn=pre)
        self.mask_applied = bool(aff)
        s.event('start', self, mask=cpulist(mask or []), others=others)

    def find_mb(self, table):
        for pid, (ppid, comm) in table.items():
            if ppid == self.p.pid and comm == 'microbench':
                self.mb = self.mb_seen = pid
                return

    def poll(self, table, now):
        if self.it['kind'] == 'gre' and s_linux(self.s):
            if self.mb is None or self.mb not in table:
                self.mb = None
                self.find_mb(table)
            if self.mb is not None:
                st = proc_status(self.mb)
                if st:
                    self.rss = kb_gb(st.get('VmRSS')) or self.rss
                    self.hwm = max(self.hwm, kb_gb(st.get('VmHWM')) or 0.0)
                    try:
                        self.threads_max = max(self.threads_max, int(st.get('Threads', '0')))
                    except ValueError:
                        pass
                if self.mono_pin is None:
                    ts = proc_status(self.mb, self.mb)
                    if ts and self.mask_seen is None and ts.get('Cpus_allowed_list') != str(self.info['core']):
                        self.mask_seen = ts.get('Cpus_allowed_list')
                    if ts and ts.get('Cpus_allowed_list') == str(self.info['core']):
                        self.mono_pin = now   # gre_lite_pin: the timed thread moved to its core, right before warm-up
                        self.snap_pin = (cpu_jiffies(), interrupts(), now)
                        self.s.event('pin', self)
        return self.p.poll()

    def kill(self, why):
        self.guarded = why
        try:
            os.killpg(self.p.pid, signal.SIGKILL)
        except OSError:
            pass

    def finish(self, rc):
        s, it = self.s, self.it
        self.out.close()
        now = mono()
        res = parse_attempt(self.dir, it)
        if self.guarded:
            status = 'guard_killed'
        elif rc == 0:
            status = 'pass'
        elif rc == 2:
            status = 'refused'
        else:
            status = 'fail'
        rec = {'item': it, 'attempt': self.attempt, 'campaign': s.cid, 'segment': s.sha[:12], 'binary_sha256': s.sha,
               'boot_id': s.boot, 'lane': self.info['lane'], 'module': self.info['module'], 'core': self.info['core'],
               'siblings': self.info['siblings'], 'mask': self.info['mask'], 'mask_applied': self.mask_applied,
               'k': s.layout.k if it['kind'] == 'gre' else 0, 'pool': s.layout.pool, 'build_threads': s.a.build_threads,
               't_start': self.t_start, 'mono_start': self.mono_start, 'mono_pin': self.mono_pin, 'mono_end': now,
               'mb_pid': self.mb_seen,
               'wall_s': round(now - self.mono_start, 2), 'exit_code': rc, 'status': status, 'guard': self.guarded,
               'guard_alone': self.guard_alone,
               'others_at_start': self.others, 'mem_available_start_gb': round(self.mem_start, 2),
               'mem_available_min_gb': round(self.mem_min, 2) if self.mem_min is not None else None,
               'pswpout_delta': pswpout() - self.swap_start, 'foreign': sorted(self.foreign),
               'foreign_max': len(self.foreign), 'load1_start': self.load_start, 'hwm_gb': round(self.hwm, 3),
               'thp': s.thp, 'governor': governor(self.info['core']) if self.info['core'] >= 0 else '',
               'env_set': self.env_set, 'trust_verdict': bool(s.a.trust_verdict), 'pin': self.pin,
               'mask_seen': self.mask_seen, 'threads_max': self.threads_max, 'tagged_seen': self.tagged_seen}
        rec.update(res)
        rec['timed_window'] = timed_window(rec)
        if self.snap_pin and it['kind'] == 'gre':
            j0, i0, t0 = self.snap_pin
            j1, i1 = cpu_jiffies(), interrupts()
            c = self.info['core']

            def busy(cpu):
                a, b = j0.get(cpu), j1.get(cpu)
                if not a or not b or b[0] <= a[0]:
                    return None
                return round(100.0 * (1.0 - (b[1] - a[1]) / float(b[0] - a[0])), 2)
            rec['core_busy_pct'] = busy(c)
            rec['sibling_busy_pct'] = max([busy(x) or 0.0 for x in self.info['siblings']] or [0.0])
            dt = max(now - t0, 1e-3)
            rec['irq_per_s_core'] = round((i1.get(c, 0) - i0.get(c, 0)) / dt, 1)
            rec['irq_per_s_siblings'] = round(sum(i1.get(x, 0) - i0.get(x, 0) for x in self.info['siblings']) / dt, 1)
        write_json(os.path.join(self.dir, 'atom.json'), rec)
        s.event('end', self, status=status, rc=rc, wall_s=rec['wall_s'])
        return rec


def s_linux(s):
    return s.machine.linux


def pin_for(machine, L):
    # gre_lite_pin is Linux only: elsewhere (the Mac, ATOM_BATCH_TEST=1) GRE prints no pinned_core and gre_run.sh would
    # fail every pinned run, so those runs go unpinned and say so (atom.json pin = -1). The tests' fake microbench
    # reports any core it is given (ATOM_BATCH_TEST_PIN=1).
    return L['core'] if machine.linux or os.environ.get('ATOM_BATCH_TEST_PIN') == '1' else -1


def timed_window(rec):
    """[start, end] of the timed lookups on the monotonic clock: from the pin (warm-up follows it at once, then GRE's
    10 ms TSCNS calibration) and the log's warmup_ns and throughput; without a pin, back from the process exit."""
    it = rec['item']
    if it.get('kind') != 'gre':
        return [rec['mono_start'], rec['mono_end']]
    tp = rec.get('throughput')
    dur = it['ops'] / float(tp) if tp else None
    if rec.get('mono_pin') is not None:
        a = rec['mono_pin'] + rec.get('warmup_ns', 0) / 1e9 + 0.01
        b = min(a + dur, rec['mono_end']) if dur else rec['mono_end']
        return [a, b]
    if dur:
        return [max(rec['mono_start'], rec['mono_end'] - 1.0 - dur), rec['mono_end']]
    return [rec['mono_start'], rec['mono_end']]


# ------------------------------------------------------------------------------------------------ the scheduler
class Scheduler(object):
    def __init__(self, a, R, machine, layout, items, pred):
        self.a, self.R, self.machine, self.layout, self.pred = a, R, machine, layout, pred
        self.items = items
        self.cid = campaign_id(R)
        self.boot = boot_id()
        self.thp = thp()
        self.data = a.data
        self.sched_dir = os.path.join(R, 'sched')
        os.makedirs(self.sched_dir, exist_ok=True)
        self.events = open(os.path.join(self.sched_dir, 'events.jsonl'), 'a')
        self.nofit = set()       # (index, dataset) killed by the guard while running alone: does not fit, skip the rest
        self.swap_hist, self.last_guard = [], -1e9

    def event(self, kind, run=None, **kw):
        e = {'t': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'mono': round(mono(), 3), 'event': kind}
        if run is not None:
            e.update({'item': run.it['id'], 'attempt': run.attempt, 'lane': run.info['lane']})
        e.update(kw)
        self.events.write(json.dumps(e, sort_keys=True) + '\n')
        self.events.flush()
        os.fsync(self.events.fileno())

    def run(self):
        a = self.a
        pending, done, failed = [], 0, []
        records = attempt_records(self.R)
        by_item = {}
        for r in records:
            by_item.setdefault(r['item']['id'], []).append(r)
            if r.get('status') == 'guard_killed' and r.get('guard_alone'):
                self.nofit.add((base_index(r['item']['idx']), r['item']['ds']))
        for it in self.items:
            recs = by_item.get(it['id'], [])
            for r in recs:
                # one R, one condition: a rerun with other sizes or trace settings would count the old runs as done
                ri = r['item']
                if r['status'] != 'interrupted' and any(ri.get(f) != it.get(f) for f in ('ops', 'warmup', 'table_size',
                                                                                            'env')):
                    die('%s already holds %s with ops %s, warmup %s, table size %s, env %s; this command asks for %s, '
                        '%s, %s, %s: use another --out' % (self.R, it['id'], ri.get('ops'), ri.get('warmup'),
                                                            ri.get('table_size'), ri.get('env'), it['ops'],
                                                            it['warmup'], it['table_size'], it['env']))
            if any(r['status'] == 'pass' for r in recs):
                done += 1
                continue
            # a refusal (gre_run.sh exit 2) ran nothing: after the setup is fixed, a rerun tries the item again
            fails = [r for r in recs if r['status'] == 'fail']
            guards = [r for r in recs if r['status'] == 'guard_killed']
            if len(fails) > a.retries or len(guards) > 1 or (guards and fails):
                failed.append(it['id'])
                continue
            it = dict(it)
            if guards:
                it['exclusive'], it['why'] = True, 'retry alone after the memory guard'
                it['peak_extra'] = max(0.0, (guards[-1].get('hwm_gb') or 0) + 2.0 - self.pred.peak(it))
            if fails:
                it['avoid_lanes'] = [r['lane'] - 1 for r in fails]
                it['retries_used'] = len(fails)
            if guards:
                it['guard_retry'] = True
            if (base_index(it['idx']), it['ds']) in self.nofit:
                failed.append(it['id'])
                continue
            if exclusive(it, set(self.pred.peaks)) and not it['exclusive']:
                it['why'] = 'peak never measured: alone'
            pending.append(it)
        ncpu = len(self.layout.lane(0)['mask'])
        queue = order_items(pending, self.pred, ncpu)
        note('%d items: %d done, %d failed for good, %d to run on K=%d (%s)'
             % (len(self.items), done, len(failed), len(queue), self.layout.k, self.layout.describe()))
        self.event('campaign', k=self.layout.k, layout=self.layout.as_json(), items=len(self.items), todo=len(queue),
                   sha=self.sha, thp=self.thp)
        running = {}             # lane index -> Run
        history = {}
        for r in records:
            if r.get('lane') and r['item'].get('kind') == 'gre':
                history.setdefault((r['item']['ds'], r['item']['idx']), set()).add(r['lane'] - 1)
        refused_streak, paused, last_scan, foreign = 0, None, -1e9, []
        try:
            return self.loop(queue, running, history, failed, ncpu, refused_streak, paused, last_scan, foreign)
        finally:
            self.events.close()
            if running:          # killed or crashed: stop the runs; their attempts close as interrupted on restart
                note('stopping %d running attempts' % len(running))
                for r in running.values():
                    try:
                        os.killpg(r.p.pid, signal.SIGTERM)
                    except OSError:
                        pass

    def loop(self, queue, running, history, failed, ncpu, refused_streak, paused, last_scan, foreign):
        a = self.a
        while queue or running:
            now = mono()
            table = proc_table() if self.machine.linux else {}
            mem = self.machine.mem_available()
            for lane, run in list(running.items()):
                rc = run.poll(table, now)
                run.mem_min = mem if run.mem_min is None else min(run.mem_min, mem)
                run.foreign.update(foreign)
                if rc is None:
                    continue
                del running[lane]
                rec = run.finish(rc)
                self.pred.add(rec)
                st = rec['status']
                tag = '%s a%d lane %d: %s, %.0f s' % (run.it['id'], run.attempt, run.info['lane'], st, rec['wall_s'])
                note(tag + ('' if st == 'pass' else ' (%s)' % run.out.name))
                refused_streak = refused_streak + 1 if st == 'refused' else 0
                if refused_streak >= 3:
                    self.stop_all(running, 'three refusals in a row')
                    die('gre_run.sh refused three runs in a row (exit 2): fix the setup, then rerun the same command')
                if st == 'guard_killed' and run.guard_alone:
                    cell = (base_index(run.it['idx']), run.it['ds'])
                    self.nofit.add(cell)
                    drop = [q for q in queue if (base_index(q['idx']), q['ds']) == cell]
                    note('%s/%s does not fit in memory even alone: %d more items of it skipped' % (cell + (len(drop),)))
                    failed.extend(q['id'] for q in drop)
                    queue = [q for q in queue if q not in drop]
                again = self.requeue(run, rec)
                if again and (base_index(again['idx']), again['ds']) not in self.nofit:
                    queue.append(again)
                    queue = order_items(queue, self.pred, ncpu)
                elif st != 'pass':
                    failed.append(run.it['id'])
            self.guard(running, mem, now)
            if now - last_scan >= 5.0:
                last_scan = now
                ours, foreign = benchmark_procs(self.cid) if self.machine.linux else ([], [])
                for r in running.values():
                    if r.mb is not None and r.mb in ours:
                        r.tagged_seen = True
            if foreign and a.foreign == 'wait':
                if paused is None:
                    paused = now
                    note('paused: foreign benchmark processes %s' % ', '.join(foreign))
                    self.event('pause', foreign=foreign)
            else:
                if paused is not None:
                    note('resumed after %.0f s' % (now - paused))
                    self.event('resume', paused_s=round(now - paused, 1))
                    paused = None
                while queue:
                    # a pinned run is past its build: read-only lookups allocate nothing, so it keeps its RSS (+1 GB
                    # for GRE's post-run work); before the pin it may still grow to its predicted peak
                    rlist = [{'item': r.it, 'end': max(r.mono_start + self.pred.wall(r.it, ncpu), now + 30.0),
                              'reserved': r.rss + 1.0 if r.mono_pin is not None and r.rss > 0 else
                              max(self.pred.peak(r.it), r.rss + 0.5), 'lane': l} for l, r in running.items()]
                    base = mem + sum(r.rss for r in running.values())
                    free_mem = base - self.machine.headroom - sum(r['reserved'] for r in rlist)
                    free = [j for j in range(self.layout.k) if j not in running]
                    sel = select(eligible(queue), rlist, free, free_mem, now, self.pred, ncpu, history,
                                 set(self.pred.peaks), self.layout.k)
                    if sel is None or sel == 'drain':
                        break
                    it, lane = sel
                    queue.remove(it)
                    prev = attempt_dirs(item_dir(self.R, it))
                    run = Run(self, it, lane, (prev[-1][0] + 1) if prev else 1)
                    others = [{'item': r.it['id'], 'lane': r.info['lane'], 'phase': r.phase()} for r in running.values()]
                    run.start(others)
                    running[lane] = run
                    history.setdefault((it['ds'], it['idx']), set()).add(lane)
                    note('start %s a%d on lane %d (core %s), %d running, est %.0f s, peak %.1f GB%s'
                         % (it['id'], run.attempt, run.info['lane'], run.info['core'], len(running),
                            self.pred.wall(it, ncpu), self.pred.peak(it), ', ' + it['why'] if it.get('why') else ''))
            time.sleep(a.poll)
        note('campaign done: %d failed for good%s' % (len(failed), (': ' + ', '.join(failed)) if failed else ''))
        self.event('done', failed=failed)
        return 1 if failed else 0

    def requeue(self, run, rec):
        it, st = dict(run.it), rec['status']
        if st == 'guard_killed' and not run.it.get('guard_retry'):
            it.update({'exclusive': True, 'why': 'retry alone after the memory guard', 'guard_retry': True})
            it['peak_extra'] = max(0.0, (rec.get('hwm_gb') or 0) + 2.0 - self.pred.peak(run.it))
            return it
        if st == 'fail' and it.get('retries_used', 0) < self.a.retries:
            it['retries_used'] = it.get('retries_used', 0) + 1
            it['avoid_lanes'] = list(it.get('avoid_lanes', [])) + [run.lane]
            return it
        return None

    def guard(self, running, mem, now):
        """MemAvailable below 3 GB: kill the newest run still in setup or build (if more than one runs); below
        1.5 GB, or below 3 GB while pages are swapped out: the newest run in setup or build, else the newest run. The
        kernel's OOM killer never gets to choose. A killed run frees its memory within a second or two: no second
        kill before that. Swap-out with memory to spare (the kernel trimming other users' idle pages) kills nothing:
        each run records its swap-out (pswpout_delta) for the report."""
        if not running or not self.machine.linux:
            return
        self.swap_hist = [x for x in self.swap_hist if now - x[0] <= 5.0] + [(now, pswpout())]
        swapping = self.swap_hist[-1][1] - self.swap_hist[0][1] > 4096     # 16 MB of 4 KiB pages within 5 s
        hard = mem < GUARD_HARD_GB or (swapping and mem < GUARD_GB)
        if now - self.last_guard < 3.0 or (not hard and (mem >= GUARD_GB or len(running) < 2)):
            return
        alive = [r for r in running.values() if not r.guarded]
        cands = sorted(alive, key=lambda r: (r.phase() != 'setup', -r.mono_start))
        if not hard:
            cands = [r for r in cands if r.phase() == 'setup']
        if cands:
            self.last_guard = now
            v = cands[0]
            why = 'MemAvailable %.1f GB%s' % (mem, ', swap-out' if swapping else '')
            note('memory guard: killing %s a%d (%s)' % (v.it['id'], v.attempt, why))
            self.event('guard', v, why=why)
            v.guard_alone = len(running) == 1
            v.kill(why)

    def stop_all(self, running, why):
        for r in running.values():
            r.kill(why)


def campaign_id(R):
    return hashlib.sha1(os.path.realpath(R).encode()).hexdigest()[:12]


class Lock(object):
    """One scheduler per R: calib, the campaign and the untimed counts never overlap."""

    def __init__(self, R):
        self.d = os.path.join(R, 'sched', 'lock')

    def __enter__(self):
        os.makedirs(os.path.dirname(self.d), exist_ok=True)
        try:
            os.mkdir(self.d)
        except OSError:
            pid = (read(os.path.join(self.d, 'pid')) or '').strip()
            if pid.isdigit() and pid_alive(int(pid)):
                die('another atom_batch.py (pid %s) is running on %s; wait for it (calib, runs and counts never '
                    'overlap)' % (pid, os.path.dirname(os.path.dirname(self.d))))
            shutil.rmtree(self.d, ignore_errors=True)
            os.mkdir(self.d)
        with open(os.path.join(self.d, 'pid'), 'w') as f:
            f.write('%d\n' % os.getpid())
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.d, ignore_errors=True)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def recover(R, cid, linux):
    """After a killed scheduler: stop its orphaned runs and close their attempts as interrupted (they do not count
    as failures and rerun as new attempts)."""
    if linux:
        pids = tagged_pids(cid)
        if pids:
            note('stopping %d orphaned processes of a previous scheduler' % len(pids))
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for p in pids:
                    try:
                        os.kill(p, sig)
                    except OSError:
                        pass
                time.sleep(3)
    n = 0
    for d in glob.glob(os.path.join(R, '*', 'cells', '*', 'a*')):
        if os.path.isdir(d) and not os.path.exists(os.path.join(d, 'atom.json')):
            arm, _, cell, att = os.path.relpath(d, R).split(os.sep)
            write_json(os.path.join(d, 'atom.json'), {'status': 'interrupted', 'attempt': int(att[1:]) if
                                                      att[1:].isdigit() else -1, 'item': {'id': arm + '/' + cell,
                                                                                          'arm': arm}})
            n += 1
    if n:
        note('%d interrupted attempts closed (they rerun)' % n)


def freeze_binary(R, gre, refreeze=False):
    """The campaign's one binary: R/gre/build/microbench (git-ignored), copied once; every OUTDIR hard-links it."""
    src = os.path.join(gre, 'build', 'microbench')
    dst_dir = os.path.join(R, 'gre', 'build')
    dst = os.path.join(dst_dir, 'microbench')
    if not os.path.exists(src) and not os.path.exists(dst):
        die('no GRE build at %s (./run.sh setup gre)' % src)
    if os.path.exists(dst):
        if os.path.exists(src) and sha256(src) != sha256(dst):
            if not refreeze:
                note('note: %s differs from the frozen copy %s; the campaign keeps the frozen copy (--refreeze to '
                     'switch: a new segment, and K > 1 needs a new calibration)' % (src, dst))
            else:
                os.replace(dst, dst + '.' + sha256(dst)[:12])
    if not os.path.exists(dst):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, dst)
        b = os.path.join(gre, 'build', 'gre_lite_build.txt')
        if os.path.exists(b):
            shutil.copy2(b, os.path.join(dst_dir, 'gre_lite_build.txt'))
    return os.path.join(R, 'gre'), sha256(dst)


def resolve_k(a, R, machine, sha, dry):
    v = load_json(os.path.join(R, 'calib', 'verdict.json'))
    if a.lanes is not None:
        k = a.lanes
    else:
        k = v['admissible_k'] if v else 1
    if k < 1:
        die('--lanes must be >= 1')
    if k > 1 and not dry:
        if not v:
            die('K=%d needs a calibration verdict (%s/calib/verdict.json): ./run.sh atom calib, then ./run.sh atom '
                'verdict; or --lanes 1 (strictly serial)' % (k, R))
        if k > v['admissible_k']:
            die('K=%d exceeds the admissible K=%d of %s/calib/verdict.json' % (k, v['admissible_k'], R))
        if v.get('binary_sha256') != sha and not a.trust_verdict:
            die('the verdict was measured with binary %s, the campaign runs %s: recalibrate (or --trust-verdict, '
                'recorded in every run)' % (str(v.get('binary_sha256'))[:12], sha[:12]))
        if v.get('topology') != machine.topology_key():
            die('the verdict was measured on another CPU topology; recalibrate')
    return k, v


def default_out(a):
    if a.out:
        return os.path.abspath(a.out)
    # folders only: results/atom_<date>.run.out and friends sort after their folder
    cands = sorted(d for d in glob.glob(os.path.join(ROOT, 'results', 'atom_*')) if os.path.isdir(d))
    R = cands[-1] if cands else os.path.join(ROOT, 'results', 'atom_' + datetime.date.today().isoformat())
    return R


def can_run(machine):
    if machine.linux or os.environ.get('ATOM_BATCH_TEST') == '1':
        return
    die('real runs need Linux (the Atom); on this machine use --dry-run')


# ------------------------------------------------------------------------------------------------ commands
def cmd_run(a, recipes):
    R = default_out(a)
    machine = Machine(a)
    pred = Predictor(attempt_records(R) if os.path.isdir(R) else ())
    thp_mode = thp() if machine.linux else 'madvise'
    counts = recipes == ['counts']
    if a.dry_run:
        k, v = (1, None) if counts else resolve_k(a, R, machine, None, True)
        items = counts_items(R, a, a.data) if counts else recipe_items(recipes, a, k, thp_mode)
        if not items:
            die('nothing to run')
        return dry_run(a, R, machine, items, k, pred, recipes, thp_mode)
    can_run(machine)
    with Lock(R):
        if counts:
            gre_dir, sha, k, v = '', 'untimed', 1, None
        else:
            gre_dir, sha = freeze_binary(R, a.gre, a.refreeze)
            k, v = resolve_k(a, R, machine, sha, False)
        layout = Layout(machine.mods, k, parse_cpulist(a.avoid_cpus))
        items = counts_items(R, a, a.data) if counts else recipe_items(recipes, a, k, thp_mode)
        recover(R, campaign_id(R), machine.linux)
        if machine.allowed and hasattr(os, 'sched_setaffinity') and not machine.fake_topo:
            os.sched_setaffinity(0, set(layout.house))  # the scheduler itself stays on the house module
        note('out %s, binary %s, THP %s, K=%d%s' % (R, sha[:12], thp_mode, k,
                                                   ' (verdict of %s)' % v.get('date', '?') if v else ''))
        s = Scheduler(a, R, machine, layout, items, pred)
        s.frozen, s.sha = gre_dir, sha
        rc = s.run()
    if not counts:
        for arm in sorted(set(it['arm'] for it in items)):
            merge_arm(R, arm, None)
        note('merged: %s' % ', '.join(sorted(set(os.path.join(R, it['arm'], 'merged') for it in items))))
    else:
        finish_counts(R, a.data)
    return rc


def finish_counts(R, data):
    d = os.path.join(R, 'untimed')
    sc = os.path.join(d, 'splice_count')
    if glob.glob(os.path.join(sc, '*.json')):
        with open(os.path.join(sc, 'COUNTS.md'), 'w') as f:
            rc = subprocess.call([sys.executable, os.path.join(ROOT, 'scripts', 'splice_table.py'), sc], stdout=f)
        note('COUNTS.md: %s' % ('written' if rc == 0 else 'splice_table.py failed (rerun it on the Mac)'))
    sums = []
    for p in sorted(glob.glob(os.path.join(data, '*.sorted'))):
        sums.append('%s  %s' % (sha256(p), os.path.basename(p)))
    if sums:
        with open(os.path.join(d, 'sorted.txt'), 'w') as f:
            f.write('\n'.join(sums) + '\n')
    nf = sorted(glob.glob(os.path.join(d, 'nfgate', '*', 'cases.jsonl')))
    if nf:
        with open(os.path.join(d, 'nfgate', 'cases.jsonl'), 'w') as out:
            for p in nf:
                out.write(read(p) or '')
    note('untimed results in %s' % d)


def dry_run(a, R, machine, items, k, pred, recipes, thp_mode):
    budget = machine.mem_avail0 - machine.headroom
    print('dry run of %s: out %s, topology %s, MemTotal %.1f GB, MemAvailable %.1f GB, headroom %.1f GB, budget '
          '%.1f GB%s' % ('+'.join(recipes), R, machine.topo_src, machine.mem_total, machine.mem_avail0,
                          machine.headroom, budget, ', THP %s' % thp_mode))
    avoid = parse_cpulist(a.avoid_cpus)
    cmd = items[0]['kind'] == 'cmd'

    def plan(kk, its):
        lay = Layout(machine.mods, kk, avoid)
        ncpu = 16 if cmd else len(lay.lane(0)['mask'])
        sched, total = simulate(order_items(its, pred, ncpu), kk, budget, pred, ncpu)
        return lay, sched, total
    lay, sched, total = plan(k, items)
    if not cmd:
        rows = []
        for kk in range(1, min(4, max_lanes(machine.mods, avoid)) + 1):
            _, sc, tot = plan(kk, items if kk == k else recipe_items(recipes, a, kk, thp_mode))
            rows.append((kk, len(sc), tot, sum(e - s for s, e, _, _ in sc)))
        print('\nK  runs   wall      run time   speed-up vs K=1')
        for kk, n, tot, work in rows:
            print('%d  %4d   %8s  %9s  %.2fx' % (kk, n, hms(tot), hms(work), rows[0][2] / tot if tot else 0))
    print('\nschedule at K=%d: %s' % (k, 'one at a time, all CPUs, nothing timed beside it' if cmd else lay.describe()))
    print('%9s %9s %4s  %-40s %7s %7s' % ('start', 'end', 'lane', 'item', 'est s', 'peak GB'))
    for s, e, lane, it in sched:
        print('%9s %9s %4d  %-40s %7.0f %7.1f%s' % (hms(s), hms(e), lane + 1, it['id'], e - s, pred.peak(it),
                                                    '  (%s)' % it['why'] if it.get('why') else ''))
    arms = {}
    for s, e, lane, it in sched:
        x = arms.setdefault(it['arm'], [0, 0.0, 1e18, 0.0])
        x[0] += 1
        x[1] += e - s
        x[2] = min(x[2], s)
        x[3] = max(x[3], e)
    print('\nper arm: ' + '; '.join('%s %d runs, %s of run time' % (arm, x[0], hms(x[1])) for arm, x in sorted(arms.items())))
    if not cmd and sched:
        # no schedule can beat the larger of: all run time over K lanes, all GB x seconds over the memory budget
        lane_lb = sum(e - s for s, e, _, _ in sched) / float(k)
        mem_lb = sum((e - s) * pred.peak(it) for s, e, _, it in sched) / budget
        print('lower bounds at K=%d: lanes %s, memory %s; this schedule is %.0f%% of the larger'
              % (k, hms(lane_lb), hms(mem_lb), 100.0 * max(lane_lb, mem_lb) / total))
    print('total at K=%d: %d runs, wall %s (%.1f h)%s' % (k, len(sched), hms(total), total / 3600.0,
                                                        '' if cmd else
                                                        '; calibration not included (./run.sh atom calib --dry-run)'))
    if k > 1 and not load_json(os.path.join(R, 'calib', 'verdict.json')):
        print('note: K=%d needs a calibration verdict before it can run for real' % k)
    return 0


def cmd_doctor(a):
    R = default_out(a)
    m = Machine(a)
    out = ['date_utc: %s' % time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'host: %s' % os.uname()[1], 'uname: %s' % ' '.join(os.uname()),
           'cpu: %s' % next((l.split(':', 1)[1].strip() for l in (read('/proc/cpuinfo') or '').splitlines()
                             if l.startswith('model name')), 'unknown'),
           'topology (%s): %s' % (m.topo_src, ' '.join('[%s]%s' % (cpulist(x['cpus']), '' if x['complete'] else '(partial)')
                                                     for x in m.mods)),
           'allowed CPUs: %s' % (cpulist(m.allowed) if m.allowed else 'unknown'),
           'isolated CPUs: %s' % (cpulist(m.iso) or 'none'),
           'MemTotal %.1f GB, MemAvailable %.1f GB, headroom %.1f GB' % (m.mem_total, m.mem_available(), m.headroom)]
    mi = meminfo() or {}
    out.append('swap: total %.1f GB, free %.1f GB; AnonHugePages %.2f GB' % (
        mi.get('SwapTotal', 0) / GB, mi.get('SwapFree', 0) / GB, mi.get('AnonHugePages', 0) / GB))
    out.append('thp: enabled %s, defrag %s' % (
        (read('/sys/kernel/mm/transparent_hugepage/enabled') or 'unknown').strip(),
        (read('/sys/kernel/mm/transparent_hugepage/defrag') or 'unknown').strip()))
    govs = {}
    for c in (m.allowed or []):
        g = governor(c)
        govs.setdefault(g, []).append(c)
    out.append('governors: %s' % ('; '.join('%s on %s' % (g, cpulist(cs)) for g, cs in sorted(govs.items())) or 'unknown'))
    for name, path in (('cgroup cpu quota', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),
                       ('cgroup memory limit', '/sys/fs/cgroup/memory/memory.limit_in_bytes'),
                       ('cgroup cpuset', '/sys/fs/cgroup/cpuset/cpuset.cpus'), ('cgroup v2 cpu.max', '/sys/fs/cgroup/cpu.max'),
                       ('cgroup v2 memory.max', '/sys/fs/cgroup/memory.max')):
        v = read(path)
        if v is not None:
            out.append('%s: %s' % (name, v.strip()))
    st = os.statvfs(ROOT)
    out.append('filesystem of %s: %s, %.1f GB free' % (ROOT, fs_type(os.path.realpath(ROOT)), st.f_bavail * st.f_frsize / GB))
    if fs_type(os.path.realpath(a.data)) == 'tmpfs':
        out.append('WARNING: the datasets are on tmpfs: they occupy RAM, so fewer lanes fit')
    out.append('HugePages_Total: %s; MALLOC_CONF=%s' % (mi.get('HugePages_Total', 'unknown'),
                                                        os.environ.get('MALLOC_CONF', '(unset)')))
    if shutil.which('lscpu'):
        ls = subprocess.Popen(['lscpu', '-e=CPU,CORE,SOCKET,NODE,CACHE'], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, universal_newlines=True).communicate()[0]
        out.append('lscpu -e:\n' + ls.rstrip())
    out.append('page size: %d' % os.sysconf('SC_PAGE_SIZE'))
    out.append('python: %s' % sys.version.split()[0])
    i0 = interrupts()
    if i0:
        time.sleep(2.0)
        i1 = interrupts()
        out.append('interrupts/s per CPU (2 s): %s' % ' '.join('%d:%d' % (c, (i1[c] - i0.get(c, 0)) / 2) for c in sorted(i1)))
    out.append('load1: %s' % load1())
    _, foreign = benchmark_procs(campaign_id(R)) if m.linux else ([], [])
    out.append('benchmark processes running now: %s' % (', '.join(foreign) or 'none'))
    for k in range(1, min(4, max_lanes(m.mods, parse_cpulist(a.avoid_cpus))) + 1):
        out.append('K=%d: %s' % (k, Layout(m.mods, k, parse_cpulist(a.avoid_cpus)).describe()))
    txt = '\n'.join(out) + '\n'
    sys.stdout.write(txt)
    if m.linux:
        os.makedirs(R, exist_ok=True)
        with open(os.path.join(R, 'machine.txt'), 'w') as f:
            f.write(txt)
        note('wrote %s' % os.path.join(R, 'machine.txt'))
    return 0


# ------------------------------------------------------------------------------------------------ smoke
SMOKE_IX = ('lipp', 'pgm', 'btree', 'splice', 'artunsync')
SMOKE_TRACE = ('trace_pgm', 'trace_lipp')


def binary_indexes(path):
    """gre_lite.sh compiles its index list into the binary: '(gre_lite: alex lipp ... sortedarray)'."""
    m = re.search(rb'gre_lite: ([a-z0-9_ ]*)\)', read_bytes(path) or b'')
    return m.group(1).decode().split() if m else []


def max_lanes(mods, avoid):
    mods = [m for m in mods if not set(m['cpus']) & set(avoid)]
    if not mods:
        return 0
    house = min(mods, key=lambda m: m['cpus'][0])['cpus']
    return sum(1 for m in mods if m['complete'] and m['cpus'] != house)


def smoke_checks(recs, linux):
    """(rows, ok): what the Linux-only parts did in a real run. The pass of gre_run.sh's checks includes the pin."""
    rows, ok = [], True
    pids = set(r.get('mb_pid') for r in recs if r.get('mb_pid'))
    for r in sorted(recs, key=lambda r: r.get('mono_start') or 0):
        it = r['item']
        bad, info = [], []
        if r.get('status') != 'pass':
            bad.append('status %s (exit %s)' % (r.get('status'), r.get('exit_code')))
        if linux:
            want = cpulist(r.get('mask') or [])
            if not r.get('mask_applied'):
                bad.append('mask not applied')
            elif r.get('mask_seen') is not None and r['mask_seen'] != want:
                bad.append('microbench CPUs %s, mask %s' % (r['mask_seen'], want))
            elif r.get('mask_seen') is None:
                info.append('mask not seen (the run ended before a poll)')
            if r.get('mono_pin') is None:
                bad.append('pin never seen: phases unknown (memory admission and overlaps fall back)')
            if not r.get('hwm_gb'):
                bad.append('no VmHWM read')
            own = [x for x in (r.get('foreign') or []) if x.split(':')[0].isdigit() and int(x.split(':')[0]) in pids]
            if own:
                bad.append('own runs counted as foreign: %s' % ', '.join(own))
            info.append('threads max %s on %d CPUs' % (r.get('threads_max'), len(r.get('mask') or [])))
            if it['idx'].startswith('trace_'):
                pts, t0, warm, core = parse_trace_log(os.path.join(r['_dir'], 'logs', '%s__%s__r1.log'
                                                                   % (it['ds'], it['idx'])))
                cpus = sorted(set(c for _, _, c in pts if c >= 0))
                if cpus != [r.get('core')]:
                    bad.append('trace chunks ran on CPUs %s, not only %s' % (cpus, r.get('core')))
                if t0 is None:
                    bad.append('no trace_t0_mono_ns (rebuild GRE: ./run.sh setup gre); calib falls back to the pin poll')
                elif r.get('mono_pin') is not None and abs(t0 - r['mono_pin']) > 2.0:
                    bad.append('trace clock %.1f s off the scheduler clock' % (t0 - r['mono_pin']))
                elif r.get('mono_pin') is not None:
                    info.append('trace t0 - pin poll %+.3f s' % (t0 - r['mono_pin']))
        ok = ok and not bad
        rows.append('| %s | %s | %s | %s | %s | %s |' % (it['id'], r.get('lane'), r.get('core'), cpulist(r.get('mask') or []),
                                                     '; '.join(info) or '-', 'FAIL: ' + '; '.join(bad) if bad else 'PASS'))
    return rows, ok


def cmd_smoke(a):
    """A few minutes of real lanes, untimed (10M keys, 3M lookups): what the Mac tests could only fake (CPU masks,
    the pin seen from /proc, VmHWM, the ATOM_BATCH_ID tag, trace_t0_mono_ns on the scheduler's clock), then merge
    and gre_report.py on the result. R/smoke/ is rebuilt each time and never merged with the campaign."""
    R = default_out(a)
    S = os.path.join(R, 'smoke')
    machine = Machine(a)
    can_run(machine)
    avoid = parse_cpulist(a.avoid_cpus)
    k = min(a.lanes or 4, max_lanes(machine.mods, avoid))
    if k < 1:
        die('no complete L2 module outside the house module')
    with Lock(R):
        if os.path.isdir(S):
            shutil.rmtree(S)
        os.makedirs(S)
        gre_dir, sha = freeze_binary(S, a.gre, True)     # the current build, whatever R/gre holds
        known = binary_indexes(os.path.join(gre_dir, 'build', 'microbench'))
        dss = [d for d in csv_list(a.datasets, ('fb', 'wise')) if os.path.exists(os.path.join(a.data, d))]
        if not dss:
            die('no smoke dataset in %s' % a.data)
        items = []
        # 3M lookups: a timed phase of a few seconds, so the 0.2 s poll sees each pin
        for ds in dss:
            for ix in [x for x in SMOKE_IX if x in known]:
                items.append(gre_item('smoke', ds, ix, 1, 3000000, 300000, table_size=10000000))
        for ix in [x for x in SMOKE_TRACE if x in known]:
            items.append(gre_item('smoke_trace', dss[0], ix, 1, 3000000, 0, TRACE_ENV, table_size=10000000))
        a.splice_plan = 'none'                           # the plans are for all 200M keys
        a.poll = min(a.poll, 0.2)
        layout = Layout(machine.mods, k, avoid)
        recover(S, campaign_id(S), machine.linux)
        if machine.allowed and hasattr(os, 'sched_setaffinity') and not machine.fake_topo:
            os.sched_setaffinity(0, set(layout.house))
        note('smoke: %d runs on K=%d (%s), binary %s' % (len(items), k, layout.describe(), sha[:12]))
        s = Scheduler(a, S, machine, layout, items, Predictor())
        s.frozen, s.sha = gre_dir, sha
        s.run()
    recs = [r for r in attempt_records(S) if (r.get('item') or {}).get('kind') == 'gre']
    rows, ok = smoke_checks(recs, machine.linux)
    lines = ['# Lane smoke (%s)' % S, '', 'K=%d; %s; binary %s; %s' % (k, layout.describe(), sha[:12],
                                                                        'Linux checks on' if machine.linux else
                                                                        'not Linux: masks, pin and /proc not checked'),
             '', '| run | lane | core | mask | seen | verdict |', '|---|---|---|---|---|---|'] + rows + ['']
    for r in recs:
        r['_key'] = r['_dir']
    conc = max([0] + [o['conc_alive_max'] for o in overlaps(recs).values()])
    if k > 1 and conc < 1:
        ok = False
        lines.append('FAIL: no two runs were ever alive at once: the lanes did not run in parallel')
    else:
        lines.append('At most %d other runs alive during a run\'s lookups.' % conc)
    for arm in ('smoke', 'smoke_trace'):
        out = merge_arm(S, arm, None)
        if out:
            rep = subprocess.Popen([sys.executable, os.path.join(ROOT, 'scripts', 'gre_report.py'), out],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            txt = rep.communicate()[0]
            with open(os.path.join(S, arm, 'report.md'), 'w') as f:
                f.write(txt)
            lines.append('gre_report.py %s: exit %d%s' % (os.path.relpath(out, S), rep.returncode,
                                                         '' if rep.returncode == 0 else ' (FAIL)'))
            ok = ok and rep.returncode == 0
    lines += ['', 'Smoke: %s' % ('PASS' if ok else 'FAIL')]
    with open(os.path.join(S, 'SMOKE.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ calibration
def calib_sessions(a, machine, pred):
    ks = [int(x) for x in a.k.split(',') if x]
    probes = [tuple(p.split('/')) for p in a.probes.split(',')] if a.probes else [(ix, ds) for ds, ix in PROBES]
    budget = machine.mem_avail0 - machine.headroom
    out = []
    for k in sorted(set(ks), reverse=True):
        if k < 2:
            continue
        lay = Layout(machine.mods, k, parse_cpulist(a.avoid_cpus))
        for ix, ds in probes:
            probe = gre_item('calib', ds, 'trace_' + base_index(ix), 1, a.calib_ops, 20000000, PROBE_ENV)
            room = budget - pred.peak(probe)
            loads = []
            for j in range(1, k):
                dss = [LOAD_DS[(j - 1 + i) % len(LOAD_DS)] for i in range(len(LOAD_DS))]
                ixs = [LOAD_IX[(j - 1 + i) % len(LOAD_IX)] for i in range(len(LOAD_IX))]
                loads.append({'lane': j, 'datasets': dss, 'indexes': ixs})
            # The largest load that fits K-1 times beside the probe: full runs, else a prefix of the file (setups and
            # lookups at the same DRAM traffic per second, less memory), else shorter operation lists.
            size, need = None, None
            for ts, ops, w in LOAD_SIZES:
                need = max(pred.peak(gre_item('l', d, i, 1, ops, w, table_size=ts)) for d in LOAD_DS for i in LOAD_IX)
                if (k - 1) * need <= room:
                    size = {'table_size': ts, 'ops': ops, 'warmup': w}
                    break
            out.append({'k': k, 'probe': probe, 'layout': lay, 'loads': loads, 'size': size,
                        'name': 'K%d/%s_%s' % (k, base_index(ix), ds), 'mem': pred.peak(probe) + (k - 1) * need})
    return out


def load_desc(size):
    if size is None:
        return 'DOES NOT FIT: skipped'
    return '%s, %dM lookups%s' % ('full size' if size['table_size'] < 0 else '%dM-key prefix' % (size['table_size'] // 1000000),
                                  size['ops'] // 1000000, ', W=%dM' % (size['warmup'] // 1000000) if size['warmup'] else '')


def cmd_calib(a):
    R = default_out(a)
    if a.lanes:
        a.k = ','.join(str(x) for x in range(a.lanes, 1, -1))
    machine = Machine(a)
    pred = Predictor()
    sessions = calib_sessions(a, machine, pred)
    if a.dry_run:
        tot = 0.0
        print('calibration dry run: out %s, topology %s, budget %.1f GB, bound %.2f%%, %d-s blocks'
              % (R, machine.topo_src, machine.mem_avail0 - machine.headroom, a.bound, a.block_s))
        for s in sessions:
            w = pred.wall(s['probe'], len(s['layout'].lane(0)['mask'])) + 20
            tot += w
            print('%-14s probe %-28s %d load lanes (%s), mem %.1f GB, est %s' % (
                s['name'], s['probe']['id'], s['k'] - 1, load_desc(s['size']), s['mem'], hms(w)))
        first_k = max(s['k'] for s in sessions) if sessions else 0
        t1 = sum(pred.wall(s['probe']) + 20 for s in sessions if s['k'] == first_k)
        print('total %s if every K is tested; %s if K=%d passes first (the sessions stop at the first admissible K)'
              % (hms(tot), hms(t1), first_k))
        return 0
    can_run(machine)
    with Lock(R):
        gre_dir, sha = freeze_binary(R, a.gre, a.refreeze)
        recover(R, campaign_id(R), machine.linux)      # orphans of a killed calibration would co-run with the probe
        if machine.allowed and hasattr(os, 'sched_setaffinity') and not machine.fake_topo:
            house = Layout(machine.mods, 1, parse_cpulist(a.avoid_cpus)).house
            os.sched_setaffinity(0, set(house))
        rng = random.Random(a.seed)
        for k in sorted(set(s['k'] for s in sessions), reverse=True):
            for s in [x for x in sessions if x['k'] == k]:
                sd = os.path.join(R, 'calib', s['name'])
                if (load_json(os.path.join(sd, 'session.json')) or {}).get('status') == 'done':
                    note('%s already done' % s['name'])
                elif s['size'] is None:
                    note('%s: the probe and %d load lanes do not fit in memory; skipped' % (s['name'], k - 1))
                    os.makedirs(sd, exist_ok=True)
                    write_json(os.path.join(sd, 'session.json'), {'status': 'does_not_fit', 'k': k})
                else:
                    for _ in range(3):   # someone else's benchmark ran during the blocks: wait for it, run it again
                        calib_session(a, R, machine, s, sd, gre_dir, sha, rng)
                        if (load_json(os.path.join(sd, 'session.json')) or {}).get('status') != 'foreign':
                            break
            res = analyze_k(R, k, a.bound)
            note('K=%d: %s' % (k, 'admissible' if res and res['admissible'] else 'not admissible'))
            if a.stop_at_pass and res and res['admissible']:
                break     # every smaller K shares less DRAM: no need to test it
    v = make_verdict(R, a.bound)
    print(v['text'])
    return 0


def calib_session(a, R, machine, s, sd, gre_dir, sha, rng):
    lay, cid = s['layout'], campaign_id(R)
    waited = False
    while machine.linux and a.foreign == 'wait' and benchmark_procs(cid)[1]:
        if not waited:
            note('%s: waiting for foreign benchmark processes to end: %s' % (s['name'], ', '.join(benchmark_procs(cid)[1])))
            waited = True
        time.sleep(10)
    if os.path.isdir(sd):
        shutil.rmtree(sd)
    os.makedirs(sd)
    foreign, last_scan = set(), -1e9
    procs = []

    def launch(name, lane, datasets, indexes, ops, warmup, repeats, env, ts):
        L = lay.lane(lane)
        d = os.path.join(sd, name)
        cmd = ['bash', GRE_RUN, d, '--gre', gre_dir, '--data', a.data, '--datasets', ','.join(datasets),
               '--indexes', ','.join(indexes), '--repeats', str(repeats), '--ops', str(ops), '--warmup', str(warmup),
               '--pin', str(pin_for(machine, L)), '--build-threads', str(a.build_threads)]
        if ts and ts > 0:
            cmd += ['--table-size', str(ts)]
        if a.splice_plan:
            cmd += ['--splice-plan', a.splice_plan]
        e = dict(os.environ)
        for v in CLEAN_ENV:
            e.pop(v, None)
        e.update(env)
        e.update({'ATOM_BATCH_ID': cid, 'GRE_RUN_BUSY': NEVER, 'GRE_RUN_BIN_LINK': '1', 'OMP_NUM_THREADS': '1'})
        aff = set(L['mask']) if hasattr(os, 'sched_setaffinity') and not machine.fake_topo else None

        def pre():
            if aff:
                os.sched_setaffinity(0, aff)
        out = open(d + '.out', 'w')
        p = subprocess.Popen(cmd, cwd=ROOT, env=e, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                             preexec_fn=pre)
        procs.append((p, out))
        return p, L
    pr = s['probe']
    probe, PL = launch('probe', 0, [pr['ds']], [pr['idx']], pr['ops'], pr['warmup'], 1, pr['env'], -1)
    loads = []
    sz = s['size']

    def start_loads():
        # at the probe's pin, all together: their setups (read, sorts, shuffle, copies, build) then overlap the first
        # blocks and their lookups the later ones, so both conditions get pairs, with every load alive in every pair
        for ld in s['loads']:
            p, L = launch('load%d' % (ld['lane'] + 1), ld['lane'], ld['datasets'], ld['indexes'], sz['ops'],
                          sz['warmup'], 200, {}, sz['table_size'])
            loads.append((p, L))
    note('%s: probe %s on core %d, then %d load lanes at its pin (%s)' % (s['name'], pr['id'], PL['core'],
                                                                         len(s['loads']), load_desc(sz)))
    t_start = mono()
    blocks, t_pin, state, status = [], None, 'on', 'done'

    def set_state(on):
        for p, _ in loads:
            try:
                os.killpg(p.pid, signal.SIGCONT if on else signal.SIGSTOP)
            except OSError:
                pass
        return mono()

    def load_phases():
        table = proc_table()
        n = {'setup': 0, 'timed': 0, 'idle': 0}
        for p, L in loads:
            mb = [pid for pid, (pp, comm) in table.items() if pp == p.pid and comm == 'microbench']
            if not mb:
                n['idle'] += 1
                continue
            ts_ = proc_status(mb[0], mb[0]) or {}
            n['timed' if ts_.get('Cpus_allowed_list') == str(L['core']) else 'setup'] += 1
        return n
    try:
        # wait for the probe's pin (its warm-up starts at once); without /proc, start right away (tests)
        mb, last = None, -1e9
        while t_pin is None and probe.poll() is None:
            if machine.fake_topo:     # tests: the fake binary never pins
                t_pin = mono()
                break
            if mb is None and mono() - last > 0.5:   # find the microbench (a /proc scan) twice a second
                last = mono()
                mb = next((pid for pid, (pp, comm) in proc_table().items()
                           if pp == probe.pid and comm == 'microbench'), None)
            if mb is not None:
                ts_ = proc_status(mb, mb)
                if ts_ is None:
                    mb = None
                elif ts_.get('Cpus_allowed_list') == str(PL['core']):
                    t_pin = mono()
            if machine.mem_available() < GUARD_GB:
                status = 'aborted_memory'
                break
            time.sleep(0.02)
        if t_pin is not None and status == 'done' and probe.poll() is None:
            start_loads()
        pair = 0
        while probe.poll() is None and status == 'done':
            order = ['off', 'on'] if rng.random() < 0.5 else ['on', 'off']
            for st in order:
                if probe.poll() is not None:
                    break
                t0 = set_state(st == 'on')
                state = st
                mid = None
                while mono() - t0 < a.block_s and probe.poll() is None:
                    if machine.linux and mono() - last_scan > 5.0:
                        last_scan = mono()
                        foreign.update(benchmark_procs(cid)[1])
                    if mid is None and mono() - t0 >= a.block_s / 2.0:
                        mid = load_phases() if machine.linux else {'setup': 0, 'timed': 0, 'idle': 0}
                    if machine.linux and machine.mem_available() < GUARD_HARD_GB:
                        status = 'aborted_memory'
                        break
                    time.sleep(0.05)
                if mid is None:
                    mid = load_phases() if machine.linux else {'setup': 0, 'timed': 0, 'idle': 0}
                blocks.append({'pair': pair, 'state': st, 'start': t0, 'end': mono(), 'phases': mid})
                if status != 'done':
                    break
            pair += 1
    except BaseException:
        status = 'interrupted'   # killed: stop the probe too; no session.json, so the session reruns
        raise
    finally:
        set_state(True)
        if status != 'done':
            try:
                os.killpg(probe.pid, signal.SIGTERM)
            except OSError:
                pass
        for p, _ in loads:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                pass
        deadline = mono() + 20
        for p, _ in loads:
            while p.poll() is None and mono() < deadline:
                time.sleep(0.1)
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
        rc = probe.wait()
        for _, out in procs:
            out.close()
    with open(os.path.join(sd, 'toggles.tsv'), 'w') as f:
        f.write('pair\tstate\tmono_start\tmono_end\tload_setup\tload_timed\tload_idle\n')
        for b in blocks:
            ph = b['phases']
            f.write('%d\t%s\t%.6f\t%.6f\t%s\t%s\t%s\n' % (b['pair'], b['state'], b['start'], b['end'],
                                                       ph.get('setup', ''), ph.get('timed', ''), ph.get('idle', '')))
    if rc != 0 and status == 'done':
        status = 'probe_failed'
    if foreign and status == 'done' and a.foreign == 'wait':
        status = 'foreign'       # someone else's benchmark ran during the blocks: rerun this session
    write_json(os.path.join(sd, 'session.json'), {
        'foreign': sorted(foreign),
        'status': status, 'k': s['k'], 'probe': pr, 'probe_exit': rc, 'core': PL['core'], 'mono_pin': t_pin,
        'mono_start': t_start, 'layout': lay.as_json(), 'loads': s['loads'], 'load_size': s['size'],
        'block_s': a.block_s, 'seed': a.seed, 'binary_sha256': sha, 'boot_id': boot_id(), 'thp': thp(),
        'topology': machine.topology_key(), 'date': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    note('%s: %s, %d blocks' % (s['name'], status, len(blocks)))


def parse_trace_log(path):
    pts, t0, warm, core = [], None, 0, None
    for line in (read(path) or '').splitlines():
        if line.startswith('trace_point: '):
            f = line.split()
            pts.append((int(f[2]), int(f[3]), int(f[5]) if len(f) > 5 else -1))
        elif line.startswith('trace_t0_mono_ns: '):
            t0 = int(line.split()[1]) / 1e9
        elif line.startswith('gre_lite_patch: '):
            m = re.search(r'warmup_num=(\d+) pin_core=(-?\d+)', line)
            if m:
                warm, core = int(m.group(1)), int(m.group(2))
    return pts, t0, warm, core


def read_toggles(path):
    rows = (read(path) or '').splitlines()
    out = []
    if not rows:
        return out
    h = rows[0].split('\t')
    for line in rows[1:]:
        r = dict(zip(h, line.split('\t')))
        out.append({'pair': int(r['pair']), 'state': r['state'], 'start': float(r['mono_start']),
                    'end': float(r['mono_end']), 'setup': int(r['load_setup'] or 0) if r.get('load_setup') else 0})
    return out


def mean_ci(ds):
    n = len(ds)
    if n == 0:
        return None
    m = sum(ds) / float(n)
    if n < 2:
        return m, None, None, n
    sd = math.sqrt(sum((x - m) ** 2 for x in ds) / (n - 1))
    h = t975(n - 1) * sd / math.sqrt(n)
    return m, m - h, m + h, n


def pct(x):
    return None if x is None else 100.0 * (math.exp(x) - 1.0)


def analyze_session(pts, t0, warm, blocks, guard=0.25, min_chunks=10):
    """Per pair: d = ln(median chunk ns/lookup in the K-way block) - ln(median in the solo block). Chunks are
    [t_i, t_i+1) of the trace, placed on the monotonic clock by t0; a chunk counts in a block only if it lies inside
    it with `guard` seconds to spare on both sides (a switch takes effect within a scheduling tick), and only past
    the warm-up calls. The last chunk ends after GRE's post-run work and is never used."""
    chunks = []
    for (c0, ta, cpu), (c1, tb, _) in zip(pts, pts[1:]):
        if c0 >= warm and c1 > c0:
            chunks.append((t0 + ta / 1e9, t0 + tb / 1e9, (tb - ta) / float(c1 - c0), cpu))
    meds = {}
    for b in blocks:
        v = [lat for s, e, lat, _ in chunks if s >= b['start'] + guard and e <= b['end'] - guard]
        if len(v) >= min_chunks:
            meds[(b['pair'], b['state'])] = (median(v), len(v), b)
    d_all, d_setup, d_quiet = [], [], []
    for p in sorted(set(b['pair'] for b in blocks)):
        on, off = meds.get((p, 'on')), meds.get((p, 'off'))
        if on and off:
            d = math.log(on[0]) - math.log(off[0])
            d_all.append(d)
            (d_setup if on[2].get('setup', 0) > 0 else d_quiet).append(d)
    cpus = sorted(set(c for _, _, _, c in chunks if c >= 0))
    off_med = [meds[k][0] for k in meds if k[1] == 'off']
    return {'pairs': len(d_all), 'chunks': len(chunks), 'all': mean_ci(d_all), 'setup': mean_ci(d_setup),
            'quiet': mean_ci(d_quiet), 'cpus': cpus, 'solo_ns': median(off_med) if off_med else None}


def analyze_k(R, k, bound):
    sess = sorted(glob.glob(os.path.join(R, 'calib', 'K%d' % k, '*', 'session.json')))
    if not sess:
        return None
    res = {'k': k, 'probes': [], 'admissible': True}
    for p in sess:
        sj = load_json(p) or {}
        name = os.path.basename(os.path.dirname(p))
        row = {'name': name, 'status': sj.get('status')}
        res['probes'].append(row)
        if sj.get('status') != 'done':
            res['admissible'] = False
            row['why'] = 'session %s' % sj.get('status')
            continue
        pr = sj['probe']
        log = os.path.join(os.path.dirname(p), 'probe', 'logs', '%s__%s__r1.log' % (pr['ds'], pr['idx']))
        pts, t0, warm, core = parse_trace_log(log)
        pin = sj.get('mono_pin')
        if t0 is None or (pin is not None and abs(t0 - pin) > 2.0):
            # no trace_t0_mono_ns (older build), or a clock that is not the scheduler's: the polled pin (the first
            # warm-up get follows it at once), within the 20 ms poll
            row['t0'] = 'pin poll' if t0 is None else 'pin poll (trace clock off by %.1f s)' % (t0 - pin)
            t0 = pin
        if not pts or t0 is None:
            res['admissible'] = False
            row['why'] = 'no trace in %s' % log
            continue
        an = analyze_session(pts, t0, warm, read_toggles(os.path.join(os.path.dirname(p), 'toggles.tsv')),
                             guard=min(0.25, 0.1 * float(sj.get('block_s', 5.0))))
        row.update({'pairs': an['pairs'], 'solo_ns': an['solo_ns'], 'cpus': an['cpus']})
        for cls in ('all', 'setup', 'quiet'):
            ci = an[cls]
            row[cls] = None if ci is None else {'n': ci[3], 'pct': pct(ci[0]), 'lo': pct(ci[1]), 'hi': pct(ci[2])}
        why = []
        if an['pairs'] < 6:
            why.append('only %d pairs' % an['pairs'])
        a_ = row['all']
        if a_ and (a_['hi'] is None or a_['hi'] > bound):
            why.append('upper bound %.2f%% > %.2f%%' % (a_['hi'] if a_['hi'] is not None else float('nan'), bound))
        st = row['setup']
        if st and st['n'] >= 3 and st['pct'] > bound:
            why.append('pairs with a load in setup: %+.2f%% > %.2f%%' % (st['pct'], bound))
        if not st or st['n'] < 3:
            # the campaign runs setups beside timed runs: a K whose setups were never measured is not shown harmless
            why.append('only %d pairs with a load in setup (3 needed): the setups beside a timed run are untested'
                       % (st['n'] if st else 0))
        if an['cpus'] and core is not None and core >= 0 and an['cpus'] != [core]:
            why.append('probe ran on CPUs %s, not only %d' % (an['cpus'], core))
        if why:
            res['admissible'] = False
            row['why'] = '; '.join(why)
    return res


def make_verdict(R, bound):
    ks = sorted(set(int(m.group(1)) for m in (re.match(r'^K(\d+)$', os.path.basename(d))
                                               for d in glob.glob(os.path.join(R, 'calib', 'K*'))) if m))
    tested = [x for x in (analyze_k(R, k, bound) for k in ks) if x]
    ok = [t['k'] for t in tested if t['admissible']]
    fails = [t['k'] for t in tested if not t['admissible']]
    best = 1                 # the largest passing K with no failing K below it (a smaller K failing is a red flag)
    for k_ in sorted(ok):
        if all(f > k_ for f in fails):
            best = k_
    shas, topology = set(), None
    for p in sorted(glob.glob(os.path.join(R, 'calib', 'K*', '*', 'session.json'))):
        sj = load_json(p) or {}
        if sj.get('binary_sha256'):
            shas.add(sj['binary_sha256'])
        topology = topology or sj.get('topology')
    sha = shas.pop() if len(shas) == 1 else None
    if len(shas) > 0:        # sessions of two binaries: none of them may vouch for the campaign's binary
        best, ok = 1, []
    lines = ['# Calibration verdict (%s)' % os.path.join(R, 'calib'), '',
             'Bound: the K-way steady latency of each probe may exceed its solo latency by at most %.2f%% (upper 95%% '
             'bound over block pairs within one process), and pairs whose K-way block had a load in setup or build '
             'may not exceed it on average.' % bound, '',
             '| K | probe | pairs | inflation % [95% CI] | with a load in setup (n) | loads in lookups only (n) | solo ns | verdict |',
             '|---|---|---|---|---|---|---|---|']
    for t in tested:
        for r in t['probes']:
            def f(c):
                if not c:
                    return '-'
                if c['lo'] is None:
                    return '%+.2f (n %d)' % (c['pct'], c['n'])
                return '%+.2f [%+.2f, %+.2f] (n %d)' % (c['pct'], c['lo'], c['hi'], c['n'])
            lines.append('| %d | %s | %s | %s | %s | %s | %s | %s |' % (
                t['k'], r['name'], r.get('pairs', '-'), f(r.get('all')), f(r.get('setup')), f(r.get('quiet')),
                '%.0f' % r['solo_ns'] if r.get('solo_ns') else '-', 'FAIL: ' + r['why'] if r.get('why') else 'PASS'))
    lines.append('')
    if sha is None and tested:
        lines.append('The sessions were made with different binaries: recalibrate with one (rm -r calib/K*).')
    lines.append('Admissible K = %d%s. Every smaller K is admissible too (fewer co-runners share less DRAM).'
                 % (best, '' if ok else ' (no K >= 2 passed: run strictly serially)'))
    lines.append('Next: ./run.sh atom run warmup,bench --out %s --lanes %d' % (R, best))
    v = {'admissible_k': best, 'bound_pct': bound, 'tested': tested, 'binary_sha256': sha, 'topology': topology,
         'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'text': '\n'.join(lines)}
    os.makedirs(os.path.join(R, 'calib'), exist_ok=True)
    write_json(os.path.join(R, 'calib', 'verdict.json'), v)
    with open(os.path.join(R, 'calib', 'verdict.md'), 'w') as f:
        f.write(v['text'] + '\n')
    return v


def cmd_verdict(a):
    R = default_out(a)
    v = make_verdict(R, a.bound)
    print(v['text'])
    return 0


# ------------------------------------------------------------------------------------------------ merge
def overlaps(recs):
    """Per attempt: other attempts alive (any phase) and in their timed window during its timed window."""
    spans = [(r['_key'], r.get('boot_id'), r['mono_start'], r['mono_end'], r['timed_window']) for r in recs
             if r.get('mono_start') is not None and r.get('mono_end') is not None]
    out = {}
    for key, boot, s, e, (ta, tb) in spans:
        alive, timed, ov = [], [], 0.0
        for k2, b2, s2, e2, (ta2, tb2) in spans:
            if k2 == key or b2 != boot:
                continue
            o = min(tb, e2) - max(ta, s2)
            if o > 0:
                alive.append((max(ta, s2), min(tb, e2)))
                ov += o
            if min(tb, tb2) - max(ta, ta2) > 0:
                timed.append((max(ta, ta2), min(tb, tb2)))
        out[key] = {'conc_alive_max': max_depth(alive), 'conc_timed_max': max_depth(timed), 'overlap_s': round(ov, 1)}
    return out


def max_depth(iv):
    ev = sorted([(a, 1) for a, b in iv] + [(b, -1) for a, b in iv], key=lambda x: (x[0], x[1]))
    d = m = 0
    for _, x in ev:
        d += x
        m = max(m, d)
    return m


def merge_arm(R, arm, only):
    recs = [r for r in attempt_records(R) if r.get('status') != 'interrupted' and (r.get('item') or {}).get('kind') == 'gre']
    for r in recs:
        r['_key'] = r['_dir']
    ov = overlaps(recs)
    mine = [r for r in recs if r['item']['arm'] == arm]
    if not mine:
        return None
    for r in mine:
        o = ov.get(r['_key'], {})
        r.update(o)
        r['solo'] = 1 if (o.get('conc_alive_max', 0) == 0 and not r.get('foreign_max')) else 0
    if only:
        want = 1 if only == 'solo' else 0
        mine = [r for r in mine if r['status'] == 'pass' and r['solo'] == want]
    out = os.path.join(R, arm, 'merged' + ('_' + only if only else ''))
    if os.path.isdir(out):
        shutil.rmtree(out)
    if not mine:
        return None
    os.makedirs(os.path.join(out, 'logs'))
    groups = {}
    for r in mine:
        it = r['item']
        groups.setdefault((it['round'], it['ds'], it['idx']), []).append(r)
    cols = ['repeat', 'dataset', 'index', 'exit_code', 'wall_s', 'log', 'load1', 'busy', 'pin', 'lane', 'core', 'mask',
            'k', 'attempt', 'status', 'conc_start', 'conc_alive_max', 'conc_timed_max', 'overlap_s', 'solo',
            'sibling_busy_pct', 'core_busy_pct', 'mem_available_min_gb', 'pswpout', 'segment']
    lines = ['\t'.join(cols)]
    order = dict((d, i) for i, d in enumerate(DATASETS))
    dss, ixs, cfgs, rounds = [], [], [], 0
    for key in sorted(groups, key=lambda k: (k[0], order.get(k[1], 99), k[2])):
        rs = sorted(groups[key], key=lambda r: (r['status'] == 'pass', r['attempt']))   # the passing one last
        for i, r in enumerate(rs):
            it = r['item']
            # the bare name only for a passing attempt: warmup_detect.py reads every *.log, so a cell that never
            # passed keeps its logs as *.log.a<n> (runs.tsv still lists them, and gre_report.py fails the cell)
            last = i == len(rs) - 1 and r['status'] == 'pass'
            name = '%s__%s__r%d.log' % (it['ds'], it['idx'], it['round']) + ('' if last else '.a%d' % r['attempt'])
            src = os.path.join(r['_dir'], 'logs', '%s__%s__r1.log' % (it['ds'], it['idx']))
            if os.path.exists(src):
                os.symlink(os.path.relpath(src, os.path.join(out, 'logs')), os.path.join(out, 'logs', name))
            row = [it['round'], it['ds'], it['idx'], r.get('exit_code') if r.get('exit_code') is not None else '',
                   r.get('run_wall_s') or r.get('wall_s'), 'logs/' + name, r.get('load1_start'), r.get('foreign_max', 0),
                   r.get('pin', r.get('core')), r.get('lane'), r.get('core'), cpulist(r.get('mask') or []), r.get('k'),
                   r['attempt'],
                   r['status'], len(r.get('others_at_start') or []), r.get('conc_alive_max'), r.get('conc_timed_max'),
                   r.get('overlap_s'), r.get('solo'), r.get('sibling_busy_pct'), r.get('core_busy_pct'),
                   r.get('mem_available_min_gb'), r.get('pswpout_delta'), r.get('segment')]
            lines.append('\t'.join('' if x is None else str(x) for x in row))
            cfgs.append(read(os.path.join(r['_dir'], 'config.txt')) or '')
        if key[1] not in dss:
            dss.append(key[1])
        if key[2] not in ixs:
            ixs.append(key[2])
        rounds = max(rounds, key[0])
    with open(os.path.join(out, 'runs.tsv'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    # config.txt: what every attempt shares; datasets, indexes and repeats of the arm; pin per run (runs.tsv)
    kv = {}
    for c in cfgs:
        for line in c.splitlines():
            if '=' in line:
                k_, v_ = line.split('=', 1)
                kv.setdefault(k_, set()).add(v_)
    keys = []
    for c in cfgs:            # every key any attempt recorded, in gre_run.sh's order
        for line in c.splitlines():
            if '=' in line and line.split('=', 1)[0] not in keys:
                keys.append(line.split('=', 1)[0])
    fixed = {'datasets': ','.join(sorted(dss, key=lambda d: order.get(d, 99))), 'indexes': ','.join(ixs),
             'pin': 'per-run', 'repeats': str(rounds)}
    with open(os.path.join(out, 'config.txt'), 'w') as f:
        for k_ in keys:
            v_ = fixed.get(k_)
            if v_ is None:
                vs = kv.get(k_, set())
                v_ = vs.pop() if len(vs) == 1 else 'per-run'
            f.write('%s=%s\n' % (k_, v_))
    shas = sorted(set(r.get('binary_sha256') for r in mine if r.get('binary_sha256')))
    prov = read(os.path.join(mine[0]['_dir'], 'provenance.txt')) or ''
    ks = sorted(set(r.get('k') for r in mine))
    with open(os.path.join(out, 'provenance.txt'), 'w') as f:
        f.write(prov)
        f.write('\n## atom_batch\nmerged %d attempts of arm %s from %s/%s/cells (provenance above: the first attempt)\n'
                'K: %s; cores: %s; solo runs: %d, parallel: %d\nbinaries: %s\nboot ids: %s\n'
                % (len(mine), arm, R, arm, ','.join(str(x) for x in ks),
                   ','.join(str(x) for x in sorted(set(r.get('core') for r in mine))),
                   sum(1 for r in mine if r['solo']), sum(1 for r in mine if not r['solo']), ' '.join(shas),
                   ' '.join(sorted(set(str(r.get('boot_id')) for r in mine)))))
        if len(shas) > 1:
            f.write('WARNING: %d binaries (segments): compare runs within a segment\n' % len(shas))
    b = os.path.join(R, 'gre', 'build', 'gre_lite_build.txt')
    os.makedirs(os.path.join(out, 'bin'))
    src_b = os.path.join(mine[0]['_dir'], 'bin', 'gre_lite_build.txt')
    if os.path.exists(src_b) or os.path.exists(b):
        shutil.copy2(src_b if os.path.exists(src_b) else b, os.path.join(out, 'bin', 'gre_lite_build.txt'))
    return out


def cmd_merge(a):
    R = default_out(a)
    arms = sorted(set(r['item']['arm'] for r in attempt_records(R) if (r.get('item') or {}).get('kind') == 'gre'))
    if a.arms:
        arms = [x for x in a.arms.split(',') if x]
    for arm in arms:
        out = merge_arm(R, arm, a.only)
        if out:
            print(out)
    return 0


def cmd_anchors(a):
    """Run-level check of the calibration inside the campaign: each anchor cell's solo warm run against its parallel
    warm run of the same round, paired by round: latency inflation = tp_solo / tp_parallel - 1."""
    R = default_out(a)
    recs = [r for r in attempt_records(R) if r.get('status') == 'pass']
    for r in recs:
        r['_key'] = r['_dir']
    ov = overlaps(recs)
    by = {}
    for r in recs:
        it = r['item']
        arm = it['arm'].replace('_thpalways', '')
        if arm in ('warm', 'warm_solo') and r.get('throughput'):
            by.setdefault((it['ds'], it['idx'], it['round']), {})[arm] = r
    ds_ = []
    lines = ['| cell | round | solo Mops/s | parallel Mops/s | parallel run overlapped (max others) | inflation % |',
             '|---|---|---|---|---|---|']
    for key in sorted(by):
        x = by[key]
        if 'warm' in x and 'warm_solo' in x:
            d = math.log(x['warm_solo']['throughput']) - math.log(x['warm']['throughput'])
            o = ov.get(x['warm']['_key'], {})
            ds_.append(d)
            lines.append('| %s/%s | %d | %.3f | %.3f | %s | %+.2f |' % (
                key[1], key[0], key[2], x['warm_solo']['throughput'] / 1e6, x['warm']['throughput'] / 1e6,
                o.get('conc_alive_max', '-'), pct(d)))
    ci = mean_ci(ds_)
    print('\n'.join(lines))
    if ci and ci[1] is not None:
        # stop when the mean exceeds the bound (its interval is then mostly above it too)
        verdict = 'PASS' if pct(ci[0]) <= a.bound else 'FAIL'
        print('\nmean inflation %+.2f%% [%+.2f, %+.2f] over %d pairs (bound %.2f%%): %s%s' % (
            pct(ci[0]), pct(ci[1]), pct(ci[2]), ci[3], a.bound, verdict,
            '' if verdict == 'PASS' else ': stop, recalibrate (docs/ATOM_FAST.md)'))
    else:
        print('\nfewer than 2 anchor pairs so far')
    return 0


def cmd_status(a):
    R = default_out(a)
    recs = attempt_records(R)
    by = {}
    for r in recs:
        it = r.get('item') or {}
        x = by.setdefault(it.get('arm', '?'), {})
        x[r.get('status', '?')] = x.get(r.get('status', '?'), 0) + 1
    running = [d for d in glob.glob(os.path.join(R, '*', 'cells', '*', 'a*'))
               if os.path.isdir(d) and not os.path.exists(os.path.join(d, 'atom.json'))]
    print('out %s' % R)
    for arm in sorted(by):
        print('%-20s %s' % (arm, ', '.join('%s %d' % kv for kv in sorted(by[arm].items()))))
    print('running now: %s' % (', '.join(os.path.relpath(d, R) for d in sorted(running)) or 'none'))
    lock = read(os.path.join(R, 'sched', 'lock', 'pid'))
    print('scheduler: %s' % ('pid %s' % lock.strip() if lock else 'not running'))
    ev = (read(os.path.join(R, 'sched', 'events.jsonl')) or '').splitlines()[-5:]
    for e in ev:
        print('  ' + e)
    return 0


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd')
    sub.required = True

    def common(p):
        p.add_argument('--out', help='results folder R (default: newest results/atom_*, else results/atom_<today>)')
        p.add_argument('--gre', default='/tmp/louisvasseur/GRE')
        p.add_argument('--data', default=os.path.join(ROOT, 'data', 'external', 'gre'))
        p.add_argument('--topology', help='fake L2 topology (dry runs, tests): MxN such as 8x2 (the Atom), or one CPU '
                       'list per module such as 0-3/4-9/10-15 (the Mac)')
        p.add_argument('--sysfs', default='/sys/devices/system/cpu')
        p.add_argument('--mem-total', type=float, help='GB (dry runs)')
        p.add_argument('--mem-available', type=float, help='GB (dry runs; default: /proc/meminfo, else 64)')
        p.add_argument('--avoid-cpus', default='', help='never use the modules of these CPUs (e.g. a CPU with heavy IRQs)')
        p.add_argument('--dry-run', action='store_true')
        p.add_argument('--build-threads', type=int, default=16, help="SPLICE/SCALE-LI build threads (the protocol's 16)")
        p.add_argument('--splice-plan', help='gre_run.sh --splice-plan (default: its own, results/splice_h/args)')
        p.add_argument('--refreeze', action='store_true', help='replace the frozen binary R/gre/build/microbench')
        p.add_argument('--bound', type=float, default=1.0, help='admissible steady-latency inflation, %% (default 1)')
        p.add_argument('--poll', type=float, default=0.5)
        p.add_argument('--foreign', choices=('wait', 'tag'), default='wait',
                       help='a benchmark outside this campaign is running: pause new starts (wait) or only record it')
        return p
    for name in ('run', 'warmup', 'bench', 'counts'):
        p = common(sub.add_parser(name))
        if name == 'run':
            p.add_argument('recipes', help='warmup, bench or warmup,bench')
        p.add_argument('--lanes', '--max-lanes', dest='lanes', type=int, help='K (default: the calibration verdict, else 1)')
        p.add_argument('--rounds', type=int)
        p.add_argument('--datasets')
        p.add_argument('--indexes', help='bench: %s; warmup alone: %s' % (','.join(BENCH_IX), ','.join(TRACE_IX)))
        p.add_argument('--trace-indexes', help='the warmup recipe\'s indexes in a combined run (default %s)'
                       % ','.join(TRACE_IX))
        p.add_argument('--retries', type=int, default=1)
        p.add_argument('--no-anchors', action='store_true')
        p.add_argument('--trust-verdict', action='store_true', help='accept a verdict made with another binary')
        p.add_argument('--ops', type=int, help='lookups per run (default 100M bench, 30M traces); small runs only')
        p.add_argument('--warmup', type=int, help='untimed lookups before the warm arm (default 20M)')
        p.add_argument('--table-size', type=int, help='keys loaded per run (default: the whole file); small runs only')
    p = common(sub.add_parser('calib'))
    p.add_argument('--k', default='4,3,2', help='the K to test, largest first')
    p.add_argument('--lanes', '--max-lanes', dest='lanes', type=int, help='test K = LANES, LANES-1, ..., 2')
    p.add_argument('--probes', help='index/dataset,... (default pgm/fb,lipp/fb)')
    p.add_argument('--calib-ops', type=int, default=200000000)
    p.add_argument('--block-s', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--all-k', dest='stop_at_pass', action='store_false', help='test every K even after one passes')
    for name in ('verdict', 'doctor', 'status', 'anchors'):
        common(sub.add_parser(name))
    p = common(sub.add_parser('smoke'))
    p.add_argument('--lanes', type=int, help='K (default: 4, or as many as the machine has)')
    p.add_argument('--datasets', help='default fb,wise (one sorted and one unsorted raw file)')
    p.add_argument('--retries', type=int, default=0)
    p = common(sub.add_parser('merge'))
    p.add_argument('--only', choices=('solo', 'parallel'))
    p.add_argument('--arms')
    a = ap.parse_args(argv)
    for attr, val in (('trust_verdict', False), ('lanes', None), ('retries', 1), ('no_anchors', False),
                      ('foreign', 'wait'), ('rounds', None), ('datasets', None), ('indexes', None)):
        if not hasattr(a, attr):
            setattr(a, attr, val)
    a.data = os.path.abspath(a.data)

    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)     # kill: stop the runs and release the lock (nohup covers SIGHUP)
    if a.cmd == 'run':
        recipes = [x for x in a.recipes.replace('+', ',').split(',') if x]
        return cmd_run(a, recipes)
    if a.cmd in ('warmup', 'bench'):
        return cmd_run(a, [a.cmd])
    if a.cmd == 'counts':
        return cmd_run(a, ['counts'])
    return {'calib': cmd_calib, 'verdict': cmd_verdict, 'doctor': cmd_doctor, 'status': cmd_status,
            'merge': cmd_merge, 'anchors': cmd_anchors, 'smoke': cmd_smoke}[a.cmd](a)


if __name__ == '__main__':
    try:
        rc = main()
        sys.stdout.flush()
    except BrokenPipeError:
        # `... --dry-run | head`: the reader left; say nothing more and exit as SIGPIPE would (141 also ends the
        # documented chain before the counts, should a scheduler ever lose its stdout)
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        rc = 141
    sys.exit(rc)
