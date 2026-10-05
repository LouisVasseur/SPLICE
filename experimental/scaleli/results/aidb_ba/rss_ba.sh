#!/bin/zsh
# E5: steady-state RSS of B vs C on fb (recipe from ../fullscale/memory_audit.md section 2b).
# Plateau RSS - 3.20 GB (w.initial) - 32 B x ops (w.trace) = real index bytes. Serial; waits for other jobs.
set -u; zmodload zsh/datetime
SC=${0:A:h}; mkdir -p $SC/rss
for cell in B C; do
  out=$SC/rss/fb__$cell; [[ -s $out.json ]] && { echo "skip $cell"; continue; }
  while pgrep -f "scaleli_bench|scaleli_hardness" >/dev/null; do sleep 5; done
  cmd=($(python3 $SC/run_ba.py show $cell fb | sed 's/--ops 5000000/--ops 10000000/; s/--instrument 1/--instrument 0/'))
  echo "=== $(date +%T) $cell"; $cmd > $out.json 2> $out.err & pid=$!; : > $out.rss; t0=$EPOCHREALTIME
  while kill -0 $pid 2>/dev/null; do r=$(ps -o rss= -p $pid | tr -d ' '); [[ -n "$r" ]] && echo "$((EPOCHREALTIME-t0)) $r" >> $out.rss; sleep 0.25; done
  wait $pid; echo "rc=$?"
done
python3 - "$SC/rss" <<'PY'
import json, os, sys
d = sys.argv[1]
def plateau(f):
    v = [int(l.split()[1]) * 1024 for l in open(f) if l.strip()]; n = len(v); start = 0
    for i in range(n - 1, 0, -1):
        if v[i] < 0.90 * v[i - 1] and n - i >= 12: start = i; break
    seg = v[start:]; hi = max(seg)
    while seg and seg[-1] < 0.85 * hi: seg.pop()
    return max(seg)
for c in ('B', 'C'):
    j = json.load(open(f'{d}/fb__{c}.json')); m = j['memory_before']; n = j['source_rows']
    real = plateau(f'{d}/fb__{c}.rss') - 3_200_000_000 - 32 * j['operations']
    print(f"fb {c}: accounted {m['accounted_bytes']/n:.3f} B/key, real {real/n:.3f} B/key, unaccounted {(real-m['accounted_bytes'])/1e6:+.0f} MB")
PY
