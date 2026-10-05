#!/bin/zsh
# Resumable parallel download of the ten AIDB/GRE datasets (source: github.com/gre4index/GRE datasets/download.sh).
cd "$(dirname "$0")"
one() {
  name=$1; url="https://www.cse.cuhk.edu.hk/mlsys/gre/$name"
  if [ -f "$name" ]; then echo "$(date +%T) $name already present"; return 0; fi
  echo "$(date +%T) start $name"
  if curl -sS -L --fail -C - --retry 8 --retry-delay 15 --retry-all-errors -o "$name.part" "$url"; then
    mv "$name.part" "$name"; echo "$(date +%T) done $name $(wc -c < "$name" | tr -d " ") bytes"
  else echo "$(date +%T) FAILED $name"; fi
}
export -f one 2>/dev/null
for n in covid genome libio osm books fb history planet stack wise; do echo $n; done | xargs -P 4 -I{} zsh -c 'source ./download.sh.fn; one {}'
