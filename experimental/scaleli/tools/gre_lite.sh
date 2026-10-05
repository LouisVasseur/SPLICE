#!/usr/bin/env bash
# Build a trimmed GRE (gre4index/GRE, Wongkham et al., VLDB 2022) next to SPLICE, for machines without AVX2 or MKL.
#
# The full suite needs AVX2/BMI2 (HOT) and Intel MKL (XIndex, FINEdex) and cannot compile on the DIAS Atom (no AVX).
# This build keeps the indexes the NFL / CSV comparison needs: ALEX, LIPP, dynamic PGM, STX B+tree and ART (unsync).
# GRE ships no licence file, so it is cloned OUTSIDE this repository and only edited locally; nothing of it is committed here.
# On CPUs without LZCNT/BMI1 (the Atom is Goldmont) ALEX is built with its own documented switch ALEX_USE_LZCNT 0, which
# only changes the gap search during inserts. Every local edit is recorded in DEST/build/gre_lite_build.txt.
#
# Usage, inside the activated conda environment (see SERVER.md):
#   experimental/scaleli/tools/gre_lite.sh [DEST]          # default DEST=/tmp/louisvasseur/GRE
# Then, for example (single thread, read-only, half the keys bulk-loaded as in GRE/NFL):
#   DEST/build/microbench --keys_file=<SOSD file> --keys_file_type=binary --read=1 --insert=0 --operations_num=10000000 \
#       --table_size=-1 --init_table_ratio=0.5 --thread_num=1 --memory --index=lipp --output_path=out.csv
# Lookups sample bulk-loaded keys, so a correct read-only run prints success_read equal to operations_num.
# --table_size=N reads only the first N keys of the file (a prefix), for smoke tests.
set -euo pipefail
DEST=${1:-/tmp/louisvasseur/GRE}
GRE_SHA=e807edcef51df6732f07f94d4c797fb3897519ba          # GRE master as of 2022-11-09
: "${CONDA_PREFIX:?activate the conda environment first (SERVER.md)}"
MAMBA=${MAMBA_EXE:-/tmp/louisvasseur/bin/micromamba}

echo "== 1/4 dependencies: TBB 2020 (GRE's FindTBB reads tbb_stddef.h, gone in oneTBB 2021+) and jemalloc"
"$MAMBA" install -y -p "$CONDA_PREFIX" -c conda-forge "tbb-devel=2020.2" jemalloc

echo "== 2/4 GRE at $GRE_SHA with only the needed submodules"
[ -d "$DEST/.git" ] || git clone https://github.com/gre4index/GRE.git "$DEST"
cd "$DEST"
git checkout -q "$GRE_SHA"
git submodule update --init src/competitor/alex/src src/competitor/lipp/src src/competitor/pgm/src \
    src/competitor/btree/src src/competitor/artsync/src

echo "== 3/4 local edits: ALEX's LZCNT switch, trimmed index registry and build file (originals kept as *.full)"
# ALEX's insert-time gap search (closest_gap) uses the LZCNT/TZCNT instructions. ALEX's own setting for CPUs without them
# is ALEX_USE_LZCNT 0 (alex_nodes.h: "If your hardware does not support lzcnt/tzcnt ... set this to 0"), a bit-by-bit
# gap scan; lookups never call it. Set it from what -march=native enables, starting from the pristine file on every run.
ALEX_NODES=src/competitor/alex/src/src/core/alex_nodes.h
git -C src/competitor/alex/src checkout -q -- src/core/alex_nodes.h
MACROS=$("${CXX:-c++}" -march=native -dM -E -x c++ /dev/null)
if grep -qw __LZCNT__ <<<"$MACROS" && grep -qw __BMI__ <<<"$MACROS"; then ALEX_LZCNT=1
else ALEX_LZCNT=0; sed 's/^#define ALEX_USE_LZCNT 1$/#define ALEX_USE_LZCNT 0/' "$ALEX_NODES" > "$ALEX_NODES.tmp"
     mv "$ALEX_NODES.tmp" "$ALEX_NODES"; fi
grep -q "^#define ALEX_USE_LZCNT $ALEX_LZCNT\$" "$ALEX_NODES" || { echo "ALEX_USE_LZCNT line not found in $ALEX_NODES" >&2; exit 1; }
echo "   ALEX_USE_LZCNT=$ALEX_LZCNT (CPU $(grep -qw __LZCNT__ <<<"$MACROS" && echo has || echo lacks) LZCNT)"
[ -f src/competitor/competitor.h.full ] || cp src/competitor/competitor.h src/competitor/competitor.h.full
[ -f CMakeLists.txt.full ] || cp CMakeLists.txt CMakeLists.txt.full
cat > src/competitor/competitor.h <<'EOF'
// Trimmed by SPLICE gre_lite.sh: ALEX, LIPP, PGM, STX B+tree, ART only (no AVX2 / MKL needed). Original: competitor.h.full
#include "./indexInterface.h"
#include "./alex/alex.h"
#include "./artsync/artunsync.h"
#include "./lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "iostream"

template<class KEY_TYPE, class PAYLOAD_TYPE>
indexInterface<KEY_TYPE, PAYLOAD_TYPE> *get_index(std::string index_type) {
  indexInterface<KEY_TYPE, PAYLOAD_TYPE> *index;
  if (index_type == "alex") index = new alexInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else if (index_type == "lipp") index = new LIPPInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else if (index_type == "pgm") index = new pgmInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else if (index_type == "btree") index = new BTreeInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else if (index_type == "artunsync") index = new ARTUnsynchronizedInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else { std::cout << "Could not find a matching index called " << index_type << " (gre_lite: alex lipp pgm btree artunsync).\n"; exit(0); }
  return index;
}
EOF
cat > CMakeLists.txt <<'EOF'
# Trimmed by SPLICE gre_lite.sh: no MKL, HOT, Masstree or Wormhole. Original: CMakeLists.txt.full
cmake_minimum_required(VERSION 3.14)
project(GRE_lite)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(OpenMP REQUIRED)
find_package(JeMalloc REQUIRED)
find_package(TBB REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${TBB_INCLUDE_DIRS} ${JEMALLOC_INCLUDE_DIR})
# -include cstdint: 2020-era headers rely on transitive <cstdint>, which GCC 13+ no longer provides
add_compile_options(-faligned-new -march=native -g -O3 -include cstdint)
add_executable(microbench ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/microbench.cpp)
target_link_libraries(microbench PUBLIC OpenMP::OpenMP_CXX ${JEMALLOC_LIBRARIES} ${TBB_LIBRARIES})
EOF

echo "== 4/4 build"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
      -DTBB_ROOT_DIR="$CONDA_PREFIX" -DJEMALLOC_ROOT_DIR="$CONDA_PREFIX"
cmake --build build -j
{ echo "GRE $GRE_SHA, trimmed by SPLICE gre_lite.sh: competitor.h and CMakeLists.txt replaced (originals *.full)"
  git submodule status src/competitor/{alex,lipp,pgm,btree,artsync}/src
  echo "ALEX_USE_LZCNT=$ALEX_LZCNT (1 = as shipped)"
  "${CXX:-c++}" --version | sed -n 1p
  "${CXX:-c++}" -march=native -Q --help=target | grep -E '^ +-march=' | tr -s ' ' || true
} > build/gre_lite_build.txt
cat build/gre_lite_build.txt
echo "built: $DEST/build/microbench   (indexes: alex lipp pgm btree artunsync)"
