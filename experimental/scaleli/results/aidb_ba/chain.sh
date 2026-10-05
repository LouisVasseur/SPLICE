#!/bin/zsh
REPO=${SPLICE_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}  # repo root (scripts live in experimental/scaleli/results/<dir>/)
until grep -q PHASE_DONE $REPO/experimental/scaleli/results/aidb_ba/csv_phase.log; do sleep 5; done
$REPO/experimental/scaleli/results/aidb_ba/run2.sh untrained osm planet fb books genome covid history libio stack wise
$REPO/experimental/scaleli/results/aidb_ba/run2.sh fbtrim x
$REPO/experimental/scaleli/results/aidb_ba/run2.sh csv covid history libio stack wise
