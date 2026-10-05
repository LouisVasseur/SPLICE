#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
__doc__ = """Memory footprint before vs after NFL and CSV, in bytes per key. Stdlib only, Python >= 3.6.

  python3 memory_report.py memory.jsonl [more.jsonl ...] [--rss DIR] > memory_report.md

Accounted bytes come from each run's memory_before (taken right after the build): keys (encoded blocks) + values
(8 B/key) + metadata (headers, descriptors, root table, CSV virtual points) + delta + slack, divided by the number of
keys. N must equal B exactly (the flow lives outside the index). --rss DIR (rss_ba.py output) adds real memory:
steady-state RSS - first sample - 16 B x keys - 32 B x lookups - 8 B x warm-up keys.
"""
import glob, json, os, sys

CELLS = ['B', 'N', 'C', 'NC', 'SV']

def plateau(path):   # steady-state RSS (bytes) of a pass, as in rss_ba.sh / validate.py
    v = [int(l.split()[1]) * 1024 for l in open(path) if l.strip()]; n, start = len(v), 0
    for i in range(n - 1, 0, -1):
        if v[i] < 0.90 * v[i - 1] and n - i >= 12: start = i; break
    seg = v[start:]; hi = max(seg)
    while seg and seg[-1] < 0.85 * hi: seg.pop()
    return max(seg), v[0]

def main():
    args = sys.argv[1:]
    rss_dir = None
    if '--rss' in args:
        k = args.index('--rss'); rss_dir = args[k + 1]; del args[k:k + 2]
    best = {}
    for path in args:
        for line in open(path):
            r = json.loads(line)
            m = r.get('ba') or {}
            if 'memory_before' not in r: continue
            d, c = m.get('dataset'), m.get('cell')
            if c == 'B2': c = 'B'
            if (d, c) not in best or m.get('layer') == 'M': best[(d, c)] = r
    if not best: sys.exit('no runs with memory_before in ' + ', '.join(args))
    per = lambda r, k: r['memory_before'][k] / r['source_rows']
    print('# Memory footprint, bytes per key (accounted, right after the build)\n')
    print('| dataset | B total | keys | values | metadata | N − B | C − B | NC − B | plain sorted array |')
    print('|---|---|---|---|---|---|---|---|---|')
    notes = []
    for d in sorted({d for d, _ in best}):
        B = best.get((d, 'B'))
        if not B: continue
        if B.get('dataset_scope') == 'prefix': notes.append('%s: --limit run (%s keys), not the 200M setup' % (d, format(B['source_rows'], ',')))
        tot = lambda r: per(r, 'accounted_bytes')
        diff = lambda c: ('%+.3f' % (tot(best[(d, c)]) - tot(B))) if (d, c) in best else '–'
        sv = ('%.3f' % tot(best[(d, 'SV')])) if (d, 'SV') in best else '16.000 (by construction)'
        print('| %s | **%.3f** | %.3f | %.3f | %.3f | %s | %s | %s | %s |' % (
            d, tot(B), per(B, 'key_bytes'), per(B, 'value_bytes'), per(B, 'metadata_bytes'), diff('N'), diff('C'), diff('NC'), sv))
        if (d, 'N') in best and best[(d, 'N')]['memory_before']['accounted_bytes'] != B['memory_before']['accounted_bytes']:
            notes.append('%s: N differs from B (expected identical)' % d)
    if rss_dir:
        print('\n## Real memory (RSS) vs accounted\n')
        print('| run | real B/key | accounted B/key | gap |')
        print('|---|---|---|---|')
        for jf in sorted(glob.glob(os.path.join(rss_dir, '*__*.json'))):
            name = os.path.basename(jf)[:-5]; rf = os.path.join(rss_dir, name + '.rss')
            if not os.path.exists(rf): continue
            j = json.load(open(jf)); rows = j['initial_rows']
            top, first = plateau(rf)
            real = top - first - 16 * rows - 32 * j['operations'] - 8 * j.get('warmup_reads', 0)
            acc = j['memory_before']['accounted_bytes']
            print('| %s | %.3f | %.3f | %+.1f%% |' % (name, real / rows, acc / rows, 100.0 * (real - acc) / acc))
    print('\nNFL adds no stored bytes (N − B = 0); CSV adds 8 B per virtual point plus 4 B per root slot.')
    for n in notes: print('- ' + n)

if __name__ == '__main__':
    main()
