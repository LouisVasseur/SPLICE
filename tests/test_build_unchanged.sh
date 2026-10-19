#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# The baselines path is unchanged: in a temporary GRE clone, scripts/build_gre.sh without EXTRA must leave every GRE
# source (competitor.h above all) byte-identical to what the baselines' build_gre.sh makes, and EXTRA=transforms may
# only add the two registry lines to competitor.h (Linux: and one include_directories line to CMakeLists.txt).
#
#   tests/test_build_unchanged.sh [REF]   REF: the git ref of the baselines scripts (default gre-baselines)
#
#   GRE_SEED=DIR   the local GRE clone to clone from (default $HOME/gre-baselines/GRE; only read)
#   BASE_GRE=DIR   the baselines build to compare competitor.h with too (default $HOME/gre-baselines/GRE; only read;
#                  skipped if absent)
#   FULL=1         compile all three builds for real (minutes each); default: edits only, the compile step is stubbed
#                  (macOS GRE_CLANG=/usr/bin/false, Linux CONDA_PREFIX unset), so build_gre.sh stops right there
#   KEEP_TMP=1     keep the temporary clone
# Also checks add_transforms.py's anchors on the other platform's trim (Linux: alex,lipp,pgm,btree,artunsync with
# --cmake; macOS: alex,lipp,pgm,btree). And that this branch contains REF: the comparison runs REF's scripts, so a
# baselines fix that never reached this branch would otherwise pass unseen. Refuses while a timed run is active
# (exit 2). Exit 1 on any difference.
set -euo pipefail
REF=${1:-gre-baselines}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GRE_SEED=${GRE_SEED:-$HOME/gre-baselines/GRE}
BASE_GRE=${BASE_GRE:-$HOME/gre-baselines/GRE}
if pgrep -f '[s]cratchpad/mac_run.sh' >/dev/null 2>&1 || pgrep -f '[-]-keys_file=' >/dev/null 2>&1; then
    echo "test_build_unchanged.sh: a timed run is active (pgrep -fl 'mac_run.sh|--keys_file='); run it afterwards" >&2
    exit 2; fi
OS=$(uname -s)
T=$(mktemp -d "${TMPDIR:-/tmp}/gre-unchanged.XXXXXX")
[ "${KEEP_TMP:-0}" = 1 ] && echo "keeping $T" || trap 'rm -rf "$T"' EXIT
G=$T/gre; C=src/competitor/competitor.h
mkdir -p "$T/ref/scripts"
for f in build_gre.sh trim_gre.py; do git -C "$ROOT" show "$REF:scripts/$f" > "$T/ref/scripts/$f"; done
chmod +x "$T/ref/scripts/build_gre.sh"
bad=0
no() { echo "FAIL: $*"; bad=1; }
git -C "$ROOT" merge-base --is-ancestor "$REF" HEAD || no "this branch does not contain $REF: merge it first"

# build LABEL SCRIPT [VAR=value ...]: run a build_gre.sh into $G, then snapshot the edited sources into $T/LABEL/
build() {
    local label=$1 script=$2; shift 2
    local stub=(); [ "${FULL:-0}" = 1 ] || { [ "$OS" = Darwin ] && stub=(GRE_CLANG=/usr/bin/false) || stub=(-u CONDA_PREFIX); }
    local st=0
    rm -f "$G/build/BUILD_MANIFEST.txt"  # each label's own (a stubbed build writes none; EXTRA refuses a baselines one)
    env -u EXTRA ${stub[@]+"${stub[@]}"} GRE_SEED="$GRE_SEED" "$@" bash "$script" "$G" > "$T/$label.log" 2>&1 || st=$?
    if [ "${FULL:-0}" = 1 ]; then [ "$st" = 0 ] || no "$label: build failed ($T/$label.log)"
    elif ! grep -q '^== 3/4 build' "$T/$label.log"; then no "$label: stopped before the build step ($T/$label.log)"; fi
    mkdir -p "$T/$label"
    cp "$G/$C" "$T/$label/competitor.h"; cp "$G/CMakeLists.txt" "$T/$label/CMakeLists.txt"
    (cd "$G" && git diff) > "$T/$label/gre.diff"
    (cd "$G" && git diff -- . ':!src/competitor/competitor.h' ':!CMakeLists.txt') > "$T/$label/other.diff"
    (cd "$G/src/competitor/alex/src" && git diff) > "$T/$label/alex.diff"
    if [ -d "$G/build/mac/include" ]; then cp -R "$G/build/mac/include" "$T/$label/shims"; fi
    if [ -f "$G/build/BUILD_MANIFEST.txt" ]; then cp "$G/build/BUILD_MANIFEST.txt" "$T/$label/"; fi
    echo "$label: $(grep -E '^(trim_gre|add_transforms)\.py: ' "$T/$label.log" | paste -sd ';' -)"
}
# added A B: the lines B adds to A, failing if B removes or changes any line
added() { { diff "$1" "$2" || [ $? = 1 ]; } | awk '/^</ { bad = 1 } /^>/ { print substr($0, 3) } END { exit bad }'; }
same() { cmp -s "$1" "$2" || no "$3 differs: diff $1 $2"; }

build ref "$T/ref/scripts/build_gre.sh"
build plain "$ROOT/scripts/build_gre.sh"
build extra "$ROOT/scripts/build_gre.sh" EXTRA=transforms

# 1. without EXTRA: byte-identical to the baselines scripts, every edited file
for f in competitor.h CMakeLists.txt gre.diff alex.diff; do same "$T/ref/$f" "$T/plain/$f" "no EXTRA vs $REF: $f"; done
[ ! -d "$T/ref/shims" ] || diff -r "$T/ref/shims" "$T/plain/shims" > /dev/null || no "no EXTRA vs $REF: macOS shims differ"
if [ "${FULL:-0}" = 1 ]; then  # the manifests may differ in the repo line and the binary hash, not in the indexes
    [ "$(grep '^indexes: ' "$T/ref/BUILD_MANIFEST.txt")" = "$(grep '^indexes: ' "$T/plain/BUILD_MANIFEST.txt")" ] ||
        no "no EXTRA vs $REF: manifest indexes"
    ! grep -q '^extra: ' "$T/plain/BUILD_MANIFEST.txt" || no "no EXTRA: manifest has an extra line"
    grep -q '^extra: transforms' "$T/extra/BUILD_MANIFEST.txt" || no "EXTRA: manifest has no extra line"; fi
if [ -f "$BASE_GRE/$C" ]; then same "$BASE_GRE/$C" "$T/plain/competitor.h" "no EXTRA vs the build in $BASE_GRE: competitor.h"
else echo "(no $BASE_GRE/$C: skipped the comparison with the baselines build)"; fi

# 2. EXTRA=transforms: exactly the two registry lines (Linux: plus the include dir), nothing else in GRE
INCL='#include "gret/registry.h"'
CHAIN='  else if ((index = gret::make_index<KEY_TYPE, PAYLOAD_TYPE>(index_type)) != nullptr) {}'
want=$(printf '%s\n%s' "$INCL" "$CHAIN")
got=$(added "$T/plain/competitor.h" "$T/extra/competitor.h") || no "EXTRA removes or changes competitor.h lines"
[ "$got" = "$want" ] || no "EXTRA adds to competitor.h: $got"
[ "$(grep -B1 -xF "$INCL" "$T/extra/competitor.h" | head -n 1)" = '#include "btree/btree.h"' ] ||
    no "registry include is not right after btree.h"
[ "$(grep -A1 -xF "$CHAIN" "$T/extra/competitor.h" | tail -n 1)" = '  else {' ] ||
    no "registry branch is not right before '  else {'"
if [ "$OS" = Linux ]; then
    got=$(added "$T/plain/CMakeLists.txt" "$T/extra/CMakeLists.txt") || no "EXTRA removes CMakeLists.txt lines"
    [ "$got" = "include_directories(\"$ROOT/include\")" ] || no "EXTRA adds to CMakeLists.txt: $got"
else same "$T/plain/CMakeLists.txt" "$T/extra/CMakeLists.txt" "EXTRA on macOS: CMakeLists.txt"; fi
same "$T/plain/alex.diff" "$T/extra/alex.diff" "EXTRA: ALEX edits"
same "$T/plain/other.diff" "$T/extra/other.diff" "EXTRA: GRE edits outside competitor.h and CMakeLists.txt"

# 3. the other platform's trim: anchors still unique, same two lines (Linux trim: plus the CMake line)
if [ "$OS" = Darwin ]; then OTHER=alex,lipp,pgm,btree,artunsync; XC=--cmake; else OTHER=alex,lipp,pgm,btree; XC=--no-cmake; fi
(cd "$G" && git checkout -q -- "$C" CMakeLists.txt)
python3 "$T/ref/scripts/trim_gre.py" "$G" "$OTHER" > /dev/null
cp "$G/$C" "$T/other.competitor.h"; cp "$G/CMakeLists.txt" "$T/other.CMakeLists.txt"
python3 "$ROOT/scripts/add_transforms.py" "$G" "$ROOT" $XC > /dev/null || no "add_transforms.py on the $OTHER trim"
got=$(added "$T/other.competitor.h" "$G/$C") || no "$OTHER trim: lines removed"
[ "$got" = "$want" ] || no "$OTHER trim: competitor.h gets: $got"
got=$(added "$T/other.CMakeLists.txt" "$G/CMakeLists.txt") || no "$OTHER trim: CMakeLists.txt lines removed"
if [ "$XC" = --cmake ]; then [ "$got" = "include_directories(\"$ROOT/include\")" ] || no "$OTHER trim: CMake gets: $got"
else [ -z "$got" ] || no "$OTHER trim: CMakeLists.txt changed"; fi
python3 "$ROOT/scripts/add_transforms.py" "$G" "$ROOT" $XC > /dev/null 2>&1 && no "add_transforms.py applied twice"

if [ "$bad" = 0 ]; then echo "PASS: no EXTRA = $REF byte for byte; EXTRA=transforms adds only the registry lines"
else echo "logs and snapshots: $T (KEEP_TMP=1 keeps them)"; exit 1; fi
