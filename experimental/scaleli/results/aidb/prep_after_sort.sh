#!/bin/zsh
cd "$(dirname "$0")/../.."
until grep -q -E "\[sort\] done|StepError|Traceback" results/aidb/prep_sort.log; do sleep 5; done
if grep -q -E "StepError|Traceback" results/aidb/prep_sort.log; then echo "SORT AUDIT FAILED"; exit 1; fi
python3 - <<'PY'
import json,pathlib
prov=json.load(open('results/aidb/provenance.json'))
for d,e in prov.items():
    a=e.get('sort_audit',{})
    if a.get('sorted') is False or a.get('duplicates'):
        for p in pathlib.Path('data/samples').glob(f'{d}_2M_*'): p.unlink(); print('removed stale sample', p.name)
        for p in pathlib.Path('results/aidb/flows').glob(f'{d}_*'): p.unlink(); print('removed stale flow', p.name)
PY
python3 tools/aidb_pipeline.py --binary-dir ../../build-fusion/experimental/scaleli --steps sample,flows,hardness --jobs 4 2>&1 | tail -3
echo "PREP DONE $(date +%T)"
