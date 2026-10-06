#!/usr/bin/env bash
# validate.sh DIR OUTDIR -- window E[D] / mode shares of the full-scale chosen cell (no re-optimization)
# against ref_full/<ds>.json; prints the residual table. Same windows and settings as run_suite.sh.
set -euo pipefail
DIR=$(cd "$1" && pwd); OUTD=$2; HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-/Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/data/external/gre}
ARGS_DIR=${ARGS_DIR:-$HERE/ref_full/args}; SIM=${SIM:-"sim=500000 warm=125000"}; JOBS=${JOBS:-8}
BIN=$DIR/build-win/splice_count; [ -x "$BIN" ] || BIN=$DIR/build/splice_count
mkdir -p "$OUTD"
printf '%s\n' "fb s16" "fb end" osm\ s16 books\ s16 covid\ s16 genome\ s16 history\ s16 libio\ s16 planet\ s16 stack\ s16 wise\ s16 |
  xargs -P "$JOBS" -L 1 bash -c '"'"$BIN"'" --dataset "$0" --data "'"$DATA"'" --window "$1,12500000" --scale-as 200000000 \
    --args "$(cat "'"$ARGS_DIR"'/$0.args") '"$SIM"'" --variants joint --threads 1 --verify --parity --rows "'"$OUTD"'/$0_$1.json"'
python3 - "$HERE" "$OUTD" <<'PY'
import json, sys
here, d = sys.argv[1:]
print("| window | full E[D] | window E[D] | resid | full DIRECT | window DIRECT | resid | lines full/win | walks full/win | B/key full/win | verify | parity |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for ds, w in [("fb","s16"),("fb","end"),("osm","s16"),("books","s16"),("covid","s16"),("genome","s16"),("history","s16"),("libio","s16"),("planet","s16"),("stack","s16"),("wise","s16")]:
    c = json.load(open(f"{here}/ref_full/{ds}.json"))["chosen"]["fast"]["cell"]
    v = json.load(open(f"{d}/{ds}_{w}.json"))["variants"][0]
    fe, we = c["ed_milli"]["4k_2048"]/1000, v["ed_milli"]/1000; fd, wd = c["direct_key_ppm"]/1e6, v["direct_key_ppm"]/1e6
    print(f"| {ds}@{w} | {fe:.3f} | {we:.3f} | {we-fe:+.3f} | {fd:.3f} | {wd:.3f} | {wd-fd:+.3f} | {c['counts']['dep_lines_milli']/1000:.3f}/{v['dep_lines_milli']/1000:.3f} | {sum(c['counts']['pages4k_milli'])/1000:.3f}/{v['pages4k_milli']/1000:.3f} | {c['bytes_per_key_milli']/1000:.2f}/{v['bytes_per_key_milli']/1000:.2f} | {v['verify']['pass']} | {v['parity']} |")
PY
