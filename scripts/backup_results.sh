#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Versioned, read-only snapshots of result directories, outside the repository, while runs are going.
#
#   scripts/backup_results.sh [--every MIN] [--keep N] [--dest DIR] RESULT_DIR [RESULT_DIR...]
#
# Every MIN minutes (default 10) each RESULT_DIR is copied to DEST/<its name>/<UTC time>/ (default DEST
# $HOME/backups), with rsync --link-dest: files unchanged since the previous snapshot are hard links, so a snapshot
# costs only what changed (a few KB of runs.tsv and new logs). Each snapshot is made read-only, gets SHA256SUMS of its
# runs.tsv, MANIFEST.txt and SETTINGS, and DEST/<name>/latest points at the newest one. The newest N (default 144,
# one day at 10 min) are kept. A snapshot is skipped when nothing changed. One final snapshot is taken when stopped
# (Ctrl-C, kill). Run it beside the benchmark, e.g.:
#   nohup scripts/backup_results.sh results/atom_2026-10-19_full results/atom_warm_cold > ~/backup.log 2>&1 &
# Restore: copy DEST/<name>/latest/ back (cp -a), then rerun the same run_baselines.sh command; it resumes.
set -uo pipefail
EVERY=10 KEEP=144 DEST=$HOME/backups
while [ $# -gt 0 ]; do
    case $1 in
        --every) EVERY=$2; shift 2 ;;
        --keep) KEEP=$2; shift 2 ;;
        --dest) DEST=$2; shift 2 ;;
        *) break ;;
    esac
done
[ $# -gt 0 ] || { sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
command -v rsync >/dev/null || { echo "backup_results.sh: needs rsync" >&2; exit 2; }
SRCS=()
for d in "$@"; do SRCS+=("$(cd "$d" 2>/dev/null && pwd || echo "$d")"); done
sha() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

snap() {
    local src name dir prev stamp new
    for src in "${SRCS[@]}"; do
        [ -d "$src" ] || { echo "$(date -u +%FT%TZ) $src: not there yet"; continue; }
        name=$(basename "$src"); dir=$DEST/$name; mkdir -p "$dir"
        prev=$(readlink "$dir/latest" 2>/dev/null || true)
        # skip when nothing changed since the previous snapshot
        if [ -n "$prev" ] && [ -d "$dir/$prev" ] && rsync -anc --delete --exclude=SHA256SUMS --out-format='%n' "$src/" "$dir/$prev/" | grep -q . ; then :
        elif [ -n "$prev" ] && [ -d "$dir/$prev" ]; then continue; fi
        stamp=$(date -u +%Y%m%dT%H%M%SZ); new=$dir/$stamp
        rsync -a ${prev:+--link-dest="$dir/$prev"} "$src/" "$new/" || { echo "$(date -u +%FT%TZ) $name: rsync failed" >&2; continue; }
        (cd "$new" && sha $(ls runs.tsv MANIFEST.txt SETTINGS 2>/dev/null) > SHA256SUMS 2>/dev/null)
        chmod -R a-w "$new"
        ln -sfn "$stamp" "$dir/latest"
        echo "$(date -u +%FT%TZ) $name: snapshot $stamp ($(($(wc -l < "$new/runs.tsv" 2>/dev/null || echo 1) - 1)) rows)"
        ls -1d "$dir"/20*Z 2>/dev/null | sort -r | tail -n +"$((KEEP + 1))" | while read -r old; do chmod -R u+w "$old"; rm -rf "$old"; done
    done
}

trap 'snap; exit 0' INT TERM HUP
echo "$(date -u +%FT%TZ) backing up ${SRCS[*]} to $DEST every $EVERY min, keeping $KEEP"
while true; do snap; sleep $((EVERY * 60)) & wait $!; done
