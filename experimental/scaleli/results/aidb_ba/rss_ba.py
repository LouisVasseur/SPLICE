#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
__doc__ = """E5: steady-state RSS of B and C, input of validate.py MET-7. Python port of rss_ba.sh (no zsh needed).

  python3 rss_ba.py [--dataset fb] [--cells B,C] [--limit N] [--out DIR]

Runs each cell once with the v1 command of run_ba.py (as rss_ba.sh does) at 10M lookups without counters, and samples
the process RSS every 0.25 s into DIR/<dataset>__<cell>.rss ("seconds rss_kB") beside the run's JSON. validate.py
turns the plateau into real bytes per key: plateau - first sample - 16 B x rows - 32 B x lookups - 8 B x warm-up keys.
Serial; waits for other scaleli jobs like run_ba.py. Stdlib only, Python >= 3.6.
"""
import argparse, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ba

def rss_kb(pid):
    try:
        with open('/proc/%d/status' % pid) as f:            # Linux
            for line in f:
                if line.startswith('VmRSS:'): return int(line.split()[1])
    except OSError: pass
    out = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         universal_newlines=True).stdout.strip()            # macOS
    return int(out) if out.isdigit() else None

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--dataset', default='fb')
    ap.add_argument('--cells', default='B,C')
    ap.add_argument('--limit', type=int, help='first N rows only (smoke test)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rss'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for cell in a.cells.split(','):
        out = os.path.join(a.out, '%s__%s' % (a.dataset, cell))
        if os.path.exists(out + '.json') and os.path.getsize(out + '.json'):
            print('skip', cell); continue
        while run_ba.others(): time.sleep(5)
        cmd = run_ba.command(a.dataset, cell, 1)
        cmd[cmd.index('--ops') + 1] = '10000000'
        cmd += ['--instrument', '0'] + (['--limit', str(a.limit)] if a.limit else [])
        print(time.strftime('%H:%M:%S'), cell, ' '.join(cmd), flush=True)
        t0 = time.time()
        with open(out + '.json.tmp', 'w') as so, open(out + '.err', 'w') as se, open(out + '.rss', 'w') as rf:
            p = subprocess.Popen(cmd, stdout=so, stderr=se)
            while p.poll() is None:
                r = rss_kb(p.pid)
                if r: rf.write('%.3f %d\n' % (time.time() - t0, r)); rf.flush()
                time.sleep(0.25)
        if p.returncode == 0: os.replace(out + '.json.tmp', out + '.json')
        print(time.strftime('%H:%M:%S'), cell, 'rc=%d' % p.returncode, '%.0fs' % (time.time() - t0), flush=True)

if __name__ == '__main__':
    main()
