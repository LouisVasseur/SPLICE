#!/usr/bin/env bash
# SPDX-License-Identifier: MIT (SPLICE)
# Benchmark lane for an Apple-silicon Mac: macOS cannot reserve cores (no cpusets, and affinity hints are ignored),
# but a process in the darwin background class (setpriority PRIO_DARWIN_BG, `taskpolicy -b -p`) is kept on the
# efficiency cores and its I/O is throttled. Moving the heavy interactive apps there for the length of a batch leaves
# the performance cores to the benchmark. Every demoted process is restored with `taskpolicy -B -p` when the batch
# ends, also on Ctrl-C, kill or an error. Only processes of the current user under the listed .app bundles are
# touched (MAC_LANE_APPS, '|'-separated, overrides the list); new helper processes are demoted every 30 s.
#
# Usage:
#   scripts/mac_lane.sh run -- COMMAND [ARGS...]   demote, run COMMAND, restore (exit status of COMMAND)
#   scripts/mac_lane.sh on | off | status          by hand; `off` restores what `on` recorded
set -uo pipefail
APPS=${MAC_LANE_APPS:-'/Applications/Discord.app/|/Applications/ChatGPT.app/|/Applications/WhatsApp.app/|/Applications/Google Chrome.app/'}
STATE=${MAC_LANE_STATE:-${TMPDIR:-/tmp}/mac_lane.pids}
[ "$(uname -s)" = Darwin ] || { echo "mac_lane.sh: macOS only" >&2; exit 2; }

targets() { pgrep -u "$(id -u)" -f "$APPS" 2>/dev/null | grep -vx "$$" || true; }

demote() {   # demote every target not yet recorded; append it to STATE
  local p
  touch "$STATE"
  for p in $(targets); do
    grep -qx "$p" "$STATE" && continue
    if taskpolicy -b -p "$p" 2>/dev/null; then echo "$p" >> "$STATE"; fi
  done
}

restore() {
  local p n=0
  [ -f "$STATE" ] || return 0
  for p in $(sort -u "$STATE"); do
    kill -0 "$p" 2>/dev/null && taskpolicy -B -p "$p" 2>/dev/null && n=$((n + 1))
  done
  rm -f "$STATE"
  echo "mac_lane: restored $n processes"
}

status() {
  local p
  echo "mac_lane: $( [ -f "$STATE" ] && wc -l < "$STATE" | tr -d ' ' || echo 0) demoted (state $STATE)"
  for p in $(targets); do
    printf '  %6s %s %s\n' "$p" "$(grep -qx "$p" "$STATE" 2>/dev/null && echo background || echo normal)" \
      "$(ps -o comm= -p "$p" | sed 's|.*/||' | cut -c1-60)"
  done
}

case "${1:-}" in
  on) demote; echo "mac_lane: $(wc -l < "$STATE" | tr -d ' ') processes on the efficiency cores";;
  off) restore;;
  status) status;;
  run)
    shift; [ "${1:-}" = -- ] && shift
    # On battery or in Low Power Mode the performance cores run 1.3-2x slower (seen 2026-10-12): not a timing run.
    if { pmset -g batt | grep -q 'Battery Power' || pmset -g | grep -Eq '^ *(lowpowermode|powermode) +1'; } &&
       [ "${MAC_ALLOW_BATTERY:-0}" != 1 ]; then
      echo "mac_lane: on battery or in Low Power Mode; plug in and turn Low Power Mode off (MAC_ALLOW_BATTERY=1 overrides)" >&2
      exit 2
    fi
    [ $# -gt 0 ] || { echo "usage: mac_lane.sh run -- COMMAND [ARGS...]" >&2; exit 2; }
    trap 'restore' EXIT
    trap 'exit 130' INT TERM HUP
    demote; echo "mac_lane: $(wc -l < "$STATE" | tr -d ' ') processes on the efficiency cores; running: $*"
    ( while sleep 30; do demote; done ) & keeper=$!
    "$@"; rc=$?
    kill "$keeper" 2>/dev/null; wait "$keeper" 2>/dev/null
    exit $rc;;
  *) sed -n '3,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2;;
esac
