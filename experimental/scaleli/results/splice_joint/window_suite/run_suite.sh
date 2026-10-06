#!/usr/bin/env bash
# run_suite.sh DIR OUT.json -- SPLICE-H window suite (counts only, never timings) for the source tree DIR
# (a copy of experimental/scaleli with window_mode.patch applied). Runs, on every window of WINDOWS below,
# the joint cell and the three fair ablations (G0, T0, V0: that component neutral, the other two
# re-optimized on J) and writes one row per window plus a summary to OUT.json. See WINDOWS.md.
#
# Environment (all optional):
#   DATA     dataset dir (default: the repo's data/external/gre; read-only)
#   ARGS_DIR per-dataset chosen-cell args (<ds>.args); default DIR/results/splice_h/args, else the
#            2026-10-06 snapshot next to this script (ref_full/args)
#   JOBS     concurrent splice_count processes (default 4: five trees at once fit 16 cores + headroom)
#   THREADS  threads per process (default 1)
#   SIM      extra args appended to every cell (default "sim=500000 warm=125000")
#   REOPT    1 (default) = line search of the tier-1 resolution on J per variant; 0 = the cell as given
#   WINDOWS_FILE  alternative window list ("ds off,len" per line)
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: run_suite.sh DIR OUT.json" >&2; exit 2; }
DIR=$(cd "$1" && pwd); OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:-/Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/data/external/gre}
ARGS_DIR=${ARGS_DIR:-}
if [ -z "$ARGS_DIR" ]; then
  if [ -d "$DIR/results/splice_h/args" ]; then ARGS_DIR=$DIR/results/splice_h/args; else ARGS_DIR=$HERE/ref_full/args; fi
fi
JOBS=${JOBS:-4}; THREADS=${THREADS:-1}; SIM=${SIM:-"sim=500000 warm=125000"}; REOPT=${REOPT:-1}
SCALE_AS=200000000

# 12.5M = 200M/16 keys: every scaled quantity (L2 sets, router top bits, arena granule) is an exact power of 2.
# sC = C contiguous runs, one per stratum (WINDOWS.md); fb also gets its contiguous tail (the 21 outliers).
WINDOWS_DEFAULT="fb s16,12500000
fb end,12500000
osm s16,12500000
books s16,12500000
covid s16,12500000
genome s16,12500000
history s16,12500000
libio s16,12500000
planet s16,12500000
stack s16,12500000
wise s16,12500000"
if [ -n "${WINDOWS_FILE:-}" ]; then WINDOWS=$(grep -v '^#' "$WINDOWS_FILE" | grep .); else WINDOWS=$WINDOWS_DEFAULT; fi

BIN=$DIR/build-win/splice_count
t0=$(date +%s)
if [ ! -x "$BIN" ] || [ -n "$(find "$DIR/include/splice" "$DIR/tools/splice_count.cpp" -newer "$BIN" 2>/dev/null | head -1)" ]; then
  cmake -S "$DIR" -B "$DIR/build-win" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$DIR/build-win" --target splice_count -j 4 >/dev/null
fi
"$BIN" --help | grep -q -- '--window' || { echo "run_suite: $BIN has no --window (apply window_mode.patch)" >&2; exit 2; }
tb=$(date +%s)

WORK=$(mktemp -d "${TMPDIR:-/tmp}/splice_suite.XXXXXX")
# Two processes per window: joint, G0, T0 (T0 matches the joint's segment count) and V0 (independent).
# The joint,G0,T0 chains are the long ones (three line searches): they start first.
for vars in joint,G0,T0 V0; do
  while read -r ds win; do
    [ -f "$ARGS_DIR/$ds.args" ] || { echo "run_suite: no $ARGS_DIR/$ds.args" >&2; exit 2; }
    echo "$ds $win ${ds}_${win%%,*} $vars"
  done <<< "$WINDOWS"
done > "$WORK/jobs"
export BIN DATA ARGS_DIR SIM REOPT THREADS SCALE_AS WORK
run_one() {
  local ds=$1 win=$2 tag=$3 vars=$4 ro=()
  [ "$REOPT" = 1 ] && ro=(--reopt)
  "$BIN" --dataset "$ds" --data "$DATA" --window "$win" --scale-as "$SCALE_AS" --args "$(cat "$ARGS_DIR/$ds.args") $SIM" \
    --variants "$vars" --threads "$THREADS" "${ro[@]}" --verify --parity --rows "$WORK/$tag.${vars//,/_}.json" 2> "$WORK/$tag.${vars//,/_}.err"
}
export -f run_one
xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _ < "$WORK/jobs" || echo "run_suite: some jobs failed (see errors in OUT)" >&2
te=$(date +%s)

python3 - "$WORK" "$OUT" "$DIR" "$ARGS_DIR" "$((tb - t0))" "$((te - tb))" "$JOBS" "$THREADS" "$SIM" "$REOPT" <<'EOF'
import json, sys, glob, os
work, out, tree, argsdir, tbuild, trun, jobs, threads, sim, reopt = sys.argv[1:]
HARD = {"fb", "osm", "genome", "planet"}
wins = {}
for f in sorted(glob.glob(f"{work}/*.json")):
    j = json.load(open(f))
    w = wins.setdefault(j["window"], {"window": j["window"], "dataset": j["dataset"], "variants": {}, "wall_s": 0.0})
    w["wall_s"] += j["wall_s"]
    for v in j["variants"]: w["variants"][v["variant"]] = v
errs = [open(f).read().strip() for f in glob.glob(f"{work}/*.err") if open(f).read().strip()]
rows = []
for name, w in wins.items():
    V = w["variants"]; jv = V.get("joint")
    if not jv: continue
    ed = lambda x: x["ed_milli"] / 1000
    d = lambda k: round(ed(V[k]) - ed(jv), 3) if k in V else None
    allv = [V[k] for k in ("joint", "G0", "T0", "V0") if k in V]
    rows.append({
        "window": name, "hard": w["dataset"] in HARD, "ed": ed(jv), "bytes_per_key": jv["bytes_per_key_milli"] / 1000,
        "d_G": d("G0"), "d_T": d("T0"), "d_V": d("V0"),
        "direct_share": jv["direct_key_ppm"] / 1e6, "lines": jv["dep_lines_milli"] / 1000, "walks": jv["pages4k_milli"] / 1000,
        "outer_iters": jv["outer_iters"], "segments": jv["segments"], "eps": jv["eps"], "exceptions": jv["exceptions"],
        "outer_iters_all": {k: V[k]["outer_iters"] for k in V},
        "ed_all": {k: ed(V[k]) for k in V}, "bytes_per_key_all": {k: V[k]["bytes_per_key_milli"] / 1000 for k in V},
        "cap_violated": {k: V[k]["cap_violated"] for k in V},
        "verify": all(x.get("verify", {}).get("pass", False) for x in allv),
        "parity": all(x.get("parity", False) for x in allv),
        "wall_s": round(w["wall_s"], 1), "variants": V})
rows.sort(key=lambda r: r["window"])
summary = {}
for c in ("d_G", "d_T", "d_V"):
    h = [r[c] for r in rows if r["hard"] and r[c] is not None]
    a = [r[c] for r in rows if r[c] is not None]
    summary[c] = {"mean_hard": round(sum(h) / len(h), 3) if h else None, "mean_all": round(sum(a) / len(a), 3) if a else None,
                  "earns_its_place": (sum(h) / len(h) >= 0.05) if h else None}
res = {"tree": tree, "args_dir": argsdir, "sim": sim, "reopt": reopt == "1", "jobs": int(jobs), "threads": int(threads),
       "build_s": int(tbuild), "run_s": int(trun), "hard_windows": sorted(r["window"] for r in rows if r["hard"]),
       "all_verify": all(r["verify"] for r in rows), "all_parity": all(r["parity"] for r in rows),
       "errors": errs, "summary": summary, "rows": rows}
json.dump(res, open(out, "w"), indent=1)
print(f"run_suite: {len(rows)} windows, build {tbuild} s, run {trun} s, verify {res['all_verify']}, parity {res['all_parity']} -> {out}")
for r in rows:
    print(f"  {r['window']:28s} ed {r['ed']:.3f} B/key {r['bytes_per_key']:.2f} dG {r['d_G']} dT {r['d_T']} dV {r['d_V']} dir {r['direct_share']:.3f} lines {r['lines']:.3f} walks {r['walks']:.3f} it {r['outer_iters']} {r['wall_s']}s")
print("  summary", json.dumps(summary))
EOF
rm -r "$WORK"
