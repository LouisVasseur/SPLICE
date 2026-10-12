#!/usr/bin/env bash
# SPDX-License-Identifier: MIT (SPLICE)
# Mac-side data preparation after data/external/gre/download.sh: the .sorted copies and the 20 2M-key samples
# (data/samples/<ds>_2M_<uniform|window>_s42) that splice_count, splice_core and splice_gre_check read.
#
# Usage: tools/prepare_data.sh [BUILD_DIR]      (default build; needs BUILD_DIR/scaleli_hardness, Python >= 3.8)
#
# Every sample is checked against its tracked manifest, never rewritten: the draw is deterministic (seed 42), so a
# matching sample_sha256 and source_sha256 prove this machine's sample is the one the counts were made on. With all
# files already present (or symlinked) it only verifies and writes nothing. The Atom needs none of this (GRE sorts
# on load).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${1:-$ROOT/build}
case $BUILD in /*) ;; *) BUILD=$PWD/$BUILD;; esac
DATA=$ROOT/data/external/gre
SAMPLES=$ROOT/data/samples
HARD=$BUILD/scaleli_hardness
BUSY='[-]-keys_file=|splice_count|scaleli_hardness'    # as gre_run.sh: never beside a timed run
if [ "${PREPARE_DATA_FORCE:-0}" != 1 ] && command -v pgrep >/dev/null 2>&1 && pgrep -f "$BUSY" >/dev/null; then
    echo "prepare_data.sh: a benchmark is running (pgrep -fl '$BUSY'); run when it is done, or set PREPARE_DATA_FORCE=1" >&2; exit 1
fi
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
UNSORTED="covid genome history libio planet stack wise"

# GRE serves these unsorted; splice_count (needs_sorted) and 14 of the samples read the sorted, de-duplicated copy.
for ds in $UNSORTED; do
    [ -e "$DATA/$ds.sorted" ] && continue
    [ -x "$HARD" ] || { echo "prepare_data.sh: $HARD missing (cmake --build $BUILD)" >&2; exit 1; }
    echo "sorting $ds"
    "$HARD" --data "$DATA/$ds" --dtype uint64 --check-sorted 0 --sort-only 1 --write-sorted "$DATA/$ds.sorted.part" > /dev/null
    mv "$DATA/$ds.sorted.part" "$DATA/$ds.sorted"
done

TMP=$(mktemp -d "${TMPDIR:-/tmp}/prepare_data.XXXXXX"); trap 'rm -rf "$TMP"' EXIT
ok=0
for ds in fb osm books $UNSORTED; do
    src=$DATA/$ds; case " $UNSORTED " in *" $ds "*) src=$DATA/$ds.sorted;; esac
    for mode in uniform window; do
        name=${ds}_2M_${mode}_s42; dest=$SAMPLES/$name; man=$dest.manifest.json
        want=$(field "$man" sample_sha256)
        if [ -e "$dest" ] && [ "$(sha "$dest")" = "$want" ]; then ok=$((ok + 1)); continue; fi
        [ -e "$dest" ] && { echo "prepare_data.sh: $dest differs from its manifest; move it away first" >&2; exit 1; }
        echo "drawing $name from $(basename "$src")"
        python3 "$ROOT/tools/datasets.py" sample "$src" "$TMP/$name" --n 2000000 --dtype uint64 --mode $mode --seed 42 > /dev/null
        if [ "$(field "$TMP/$name.manifest.json" sample_sha256)" != "$want" ] \
            || [ "$(field "$TMP/$name.manifest.json" source_sha256)" != "$(field "$man" source_sha256)" ]; then
            echo "prepare_data.sh: $name does not reproduce its tracked manifest (wrong or unsorted source?)" >&2; exit 1
        fi
        mv "$TMP/$name" "$dest"; ok=$((ok + 1))
    done
done
echo "prepare_data.sh: $ok/20 samples match their manifests"
