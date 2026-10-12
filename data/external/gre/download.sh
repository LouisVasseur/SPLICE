#!/usr/bin/env bash
# Resumable parallel download of the ten AIDB/GRE datasets (source: github.com/gre4index/GRE datasets/download.sh),
# then a check against SHA256SUMS: the sha256 of the copies every result in this repo was computed on.
cd "$(dirname "$0")" || exit 1
RETRY_ALL=; curl -sS --retry-all-errors -V >/dev/null 2>&1 && RETRY_ALL=--retry-all-errors  # curl >= 7.71 only
export RETRY_ALL
one() {
  name=$1; url="https://www.cse.cuhk.edu.hk/mlsys/gre/$name"
  if [ -f "$name" ]; then echo "$(date +%T) $name already present"; return 0; fi
  echo "$(date +%T) start $name"
  if curl -sS -L --fail -C - --retry 8 --retry-delay 15 $RETRY_ALL -o "$name.part" "$url"; then
    mv "$name.part" "$name"; echo "$(date +%T) done $name $(wc -c < "$name" | tr -d " ") bytes"
  else echo "$(date +%T) FAILED $name"; fi
}
export -f one
printf '%s\n' covid genome libio osm books fb history planet stack wise | xargs -P 4 -I{} bash -c 'one {}'
echo "$(date +%T) checking sha256 (about a minute)"
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
