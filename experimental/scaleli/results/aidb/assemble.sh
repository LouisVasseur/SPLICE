#!/bin/zsh
# Final assembly after all experiment stages: reports, manifest, walkthrough checks.
set -e
cd "$(dirname "$0")/../../../.."  # repo root (scaleli_sota)
S=experimental/scaleli
inputs=()
for lab in "uniform 2M=results/aidb/sweep/summary.csv" "window 2M=results/aidb_window/sweep/summary.csv" \
           "granularity r4096=results/aidb_granularity/summary_r4096.csv" "granularity r16384=results/aidb_granularity/summary_r16384.csv" \
           "granularity r32768=results/aidb_granularity/summary_r32768.csv" "full scale 200M=results/aidb_fullscale/sweep/summary.csv" \
           "final: QoS-pinned, 5M lookups, 5 seeds (uniform+window)=results/aidb_final/sweep/summary.csv"; do
  f=${lab#*=}; [ -f "$S/$f" ] && inputs+=("$lab") || echo "skip missing $f"
done
( cd $S && python3 tools/fusion_report.py "${inputs[@]}" --hardness "uniform 2M=results/aidb/hardness.json" --hardness "window 2M=results/aidb_window/hardness.json" --output results/aidb/fusion_report.html )
( cd $S && python3 tools/aidb_pipeline.py --binary-dir ../../build-fusion/experimental/scaleli --steps report --force >/dev/null 2>&1 || echo "uniform report step failed" )
( cd $S && python3 tools/aidb_pipeline.py --binary-dir ../../build-fusion/experimental/scaleli --sample-mode window --results results/aidb_window --steps report --force >/dev/null 2>&1 || echo "window report step failed" )
python3 - <<'PY'
import hashlib,pathlib
root=pathlib.Path('.')
old=[l.split('  ',1)[1] for l in pathlib.Path('MANIFEST.sha256').read_text().splitlines() if l.strip()]
new=[str(p) for p in pathlib.Path('experimental/scaleli').rglob('*') if p.is_file() and not any(part in ('results','data','__pycache__','build') for part in p.parts) and p.suffix in ('.py','.cpp','.hpp','.json','.md','.sh','.txt','.css','.html','.pdf')]
new+=[str(p) for p in pathlib.Path('experiments').glob('*.py')]+['docs/APPROACHES.md','README.md','walkthrough.py','.gitignore','CMakeLists.txt']
paths=sorted({p for p in set(old)|set(new) if pathlib.Path(p).is_file()})
pathlib.Path('MANIFEST.sha256').write_text(''.join(f'{hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()}  {p}\n' for p in paths))
print('manifest entries:',len(paths))
PY
python3 walkthrough.py check 2>&1 | tail -2
python3 walkthrough.py demo 09 2>&1 | tail -4
echo "ASSEMBLY DONE $(date +%T)"
