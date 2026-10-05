#!/bin/zsh
REPO=${SPLICE_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}  # repo root (scripts live in experimental/scaleli/results/<dir>/)
$REPO/experimental/scaleli/results/aidb_ba/run2.sh fbtrim x
$REPO/experimental/scaleli/results/aidb_ba/run2.sh csv covid history libio stack wise
