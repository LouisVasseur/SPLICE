#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Build GRE (gre4index/GRE, Wongkham et al., VLDB 2022) at a pinned commit, with only its own competitors.
#
#   scripts/build_gre.sh [DEST]        default DEST=$HOME/gre-baselines/GRE (outside this repository)
#   EXTRA=transforms scripts/build_gre.sh [DEST]   default DEST=$HOME/gre-transform/GRE; adds the transform indexes
#
# Linux x86-64 (reference, e.g. the DIAS Atom): inside an activated conda environment. Installs TBB 2020.2 and jemalloc
#   5.4.0 into it, builds with GRE's own CMake. Indexes: alex lipp pgm btree artunsync.
# macOS arm64 (preview only): Apple clang, no dependencies. Indexes: alex lipp pgm btree (ART needs SSE2).
#
# GRE has no licence, so it is cloned to DEST and edited there; nothing of it is committed here. Edits, all
# build-enabling, none in GRE's driver or timed loop (src/benchmark/benchmark.h, microbench.cpp):
#   both:  scripts/trim_gre.py drops the competitors that need AVX2/MKL from competitor.h and CMakeLists.txt.
#          ALEX_USE_LZCNT 0 when the CPU lacks LZCNT/BMI1 (ALEX's own documented switch; insert-time gap search only).
#   macOS: rdtsc -> CNTVCT_EL0 (tscns.h); x86 asm compiled out (ALEX's CPUID, GRE utils.h cmpxchg, used by XIndex only);
#          shims for x86intrin popcount, single-thread OpenMP, tbb::parallel_sort -> std::sort, empty jemalloc.h.
# Every edit starts from the pristine file, and DEST/build/BUILD_MANIFEST.txt records the commit, edits and compiler.
#   GRE_SEED=DIR      clone GRE and its 5 submodules from a local GRE clone instead of GitHub
#   GRE_FORCE=1       build even while a benchmark runs
#   EXTRA=transforms  after the trim, scripts/add_transforms.py registers include/gret/registry.h in competitor.h
#                     (Linux: and include_directories in CMakeLists.txt; macOS: -I REPO/include); recorded in the
#                     manifest. Without EXTRA nothing differs from the baselines build (tests/test_build_unchanged.sh).
set -euo pipefail
EXTRA=${EXTRA:-}
case "$EXTRA" in ''|transforms) ;; *) echo "EXTRA must be 'transforms' or unset, not '$EXTRA'" >&2; exit 1 ;; esac
if [ -n "$EXTRA" ]; then
    DEST=${1:-$HOME/gre-transform/GRE}
    # the baselines clone is what run_baselines.sh times by default: never put the transforms into it
    if [ -d "$DEST" ] && [ -d "$HOME/gre-baselines/GRE" ] && \
       [ "$(cd "$DEST" && pwd -P)" = "$(cd "$HOME/gre-baselines/GRE" && pwd -P)" ]; then
        echo "EXTRA=$EXTRA: $DEST is the baselines clone; build into another DEST" >&2; exit 1; fi
else DEST=${1:-$HOME/gre-baselines/GRE}; fi
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
GRE_SHA=e807edcef51df6732f07f94d4c797fb3897519ba          # GRE master as of 2022-11-09
SUBS="src/competitor/alex/src src/competitor/lipp/src src/competitor/pgm/src src/competitor/btree/src src/competitor/artsync/src"
OS=$(uname -s)
case "$OS/$(uname -m)" in
  Linux/x86_64) KEEP=alex,lipp,pgm,btree,artunsync ;;
  Darwin/arm64) KEEP=alex,lipp,pgm,btree ;;
  *) echo "build_gre.sh: Linux x86-64 or macOS arm64 only" >&2; exit 1 ;;
esac
if [ "${GRE_FORCE:-0}" != 1 ] && pgrep -f '[-]-keys_file=' >/dev/null 2>&1; then
    echo "a GRE benchmark is running (pgrep -fl -- --keys_file=); build after it, or GRE_FORCE=1" >&2; exit 1
fi

echo "== 1/4 GRE $GRE_SHA -> $DEST"
if [ ! -d "$DEST/.git" ]; then
    mkdir -p "$(dirname "$DEST")"
    if [ -n "${GRE_SEED:-}" ]; then
        git clone -q --no-checkout "$GRE_SEED" "$DEST"
        for p in $SUBS; do git -C "$DEST" config "submodule.$p.url" "$GRE_SEED/$p"; done
    else
        git clone -q https://github.com/gre4index/GRE.git "$DEST"
    fi
fi
cd "$DEST"
git checkout -q "$GRE_SHA"
git -c protocol.file.allow=always submodule update -q --init $SUBS
# pristine sources before every edit (idempotent)
git checkout -q -- .
for p in $SUBS; do git -C "$p" checkout -q -- .; done

echo "== 2/4 trim to $KEEP, ALEX LZCNT switch"
python3 "$HERE/trim_gre.py" "$DEST" "$KEEP"
XINC=()
if [ "$EXTRA" = transforms ]; then python3 "$HERE/add_transforms.py" "$DEST" "$ROOT"; XINC=(-I"$ROOT/include"); fi
ALEX_NODES=src/competitor/alex/src/src/core/alex_nodes.h
if [ "$OS" = Darwin ]; then ALEX_LZCNT=0
else MACROS=$("${CXX:-c++}" -march=native -dM -E -x c++ /dev/null)
     if grep -qw __LZCNT__ <<<"$MACROS" && grep -qw __BMI__ <<<"$MACROS"; then ALEX_LZCNT=1; else ALEX_LZCNT=0; fi; fi
[ "$ALEX_LZCNT" = 1 ] || { sed 's/^#define ALEX_USE_LZCNT 1$/#define ALEX_USE_LZCNT 0/' $ALEX_NODES > $ALEX_NODES.tmp
                           mv $ALEX_NODES.tmp $ALEX_NODES; }
grep -q "^#define ALEX_USE_LZCNT $ALEX_LZCNT\$" $ALEX_NODES || { echo "ALEX_USE_LZCNT line not found" >&2; exit 1; }
echo "   ALEX_USE_LZCNT=$ALEX_LZCNT"

# edit FILE before|after|replace ANCHOR < lines: ANCHOR must be a whole line occurring exactly once in FILE.
edit() {
    local n; n=$(grep -cxF -- "$3" "$1" || true)
    [ "$n" = 1 ] || { echo "anchor found $n times in $1: $3" >&2; exit 1; }
    cat > "$1.add"
    WHERE=$2 ANCHOR=$3 awk 'FILENAME == ARGV[1] { add = add $0 "\n"; next }
        $0 == ENVIRON["ANCHOR"] && ENVIRON["WHERE"] == "before" { printf "%s", add }
        $0 == ENVIRON["ANCHOR"] && ENVIRON["WHERE"] == "replace" { printf "%s", add; next }
        { print }
        $0 == ENVIRON["ANCHOR"] && ENVIRON["WHERE"] == "after" { printf "%s", add }' "$1.add" "$1" > "$1.tmp"
    mv "$1.tmp" "$1"; rm -f "$1.add"
}

echo "== 3/4 build"
mkdir -p build
if [ "$OS" = Linux ]; then
    : "${CONDA_PREFIX:?activate the conda environment first}"
    # TBB 2020: GRE's FindTBB reads tbb_stddef.h, gone in oneTBB 2021+. Installed only if the environment lacks them.
    if [ -f "$CONDA_PREFIX/include/tbb/tbb_stddef.h" ] && [ -f "$CONDA_PREFIX/include/jemalloc/jemalloc.h" ]; then
        echo "   tbb 2020 and jemalloc already in $CONDA_PREFIX"
    else
        # micromamba's shell hook defines a function, invisible here: also look in the usual install places
        MAMBA=${MAMBA_EXE:-}
        for m in micromamba mamba conda "$HOME/bin/micromamba" "$HOME/.local/bin/micromamba"; do
            [ -n "$MAMBA" ] && break; MAMBA=$(command -v "$m" 2>/dev/null || true)
        done
        [ -n "$MAMBA" ] || { echo "no micromamba/mamba/conda found: set MAMBA_EXE=/path/to/micromamba" >&2; exit 1; }
        "$MAMBA" install -y -q -p "$CONDA_PREFIX" -c conda-forge "tbb-devel=2020.2" "jemalloc=5.4.0" > build/deps.log 2>&1 \
            || { tail -n 30 build/deps.log; echo "dependency install failed ($MAMBA)" >&2; exit 1; }
    fi
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
          -DTBB_ROOT_DIR="$CONDA_PREFIX" -DJEMALLOC_ROOT_DIR="$CONDA_PREFIX" > build/cmake.log 2>&1 \
        || { tail -n 30 build/cmake.log; echo "cmake configure failed" >&2; exit 1; }
    cmake --build build -j "$(nproc)" --target microbench > build/compile.log 2>&1 || { tail -n 30 build/compile.log; exit 1; }
    COMPILER=$("$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' build/CMakeCache.txt)" --version | sed -n 1p)
    FLAGS="GRE CMakeLists.txt (Release): $(grep '^add_compile_options' CMakeLists.txt)"
else
    CLANG=${GRE_CLANG:-/usr/bin/clang++}
    W=build/mac; mkdir -p $W/include/tbb $W/include/jemalloc
    edit src/tscns.h replace '  static int64_t rdtsc() { return __builtin_ia32_rdtsc(); }' <<'EOF'
#if defined(__aarch64__)  // build_gre.sh: generic timer CNTVCT_EL0, readable at EL0; TSCNS calibrates it in init()
  static int64_t rdtsc() { int64_t t; asm volatile("mrs %0, cntvct_el0" : "=r"(t)); return t; }
#else
  static int64_t rdtsc() { return __builtin_ia32_rdtsc(); }
#endif
EOF
    AB=src/competitor/alex/src/src/core/alex_base.h
    edit $AB before '    asm volatile("cpuid"' <<'EOF'
#if !defined(__aarch64__)  // build_gre.sh
EOF
    edit $AB after '                 : "a"(i), "c"(j));' <<'EOF'
#else  // build_gre.sh: no CPUID on arm64; only read by asserts of the ALEX_USE_LZCNT 1 paths
    regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
EOF
    edit src/benchmark/utils.h before 'inline uint64_t cmpxchg(uint64_t *object, uint64_t expected,' <<'EOF'
#if !defined(__aarch64__)  // build_gre.sh: x86 asm, used only by XIndex (not built)
EOF
    edit src/benchmark/utils.h before '#endif  // HELPER_H' <<'EOF'
#endif  // build_gre.sh
EOF
    cat > $W/include/x86intrin.h <<'EOF'
// build_gre.sh (MIT): <x86intrin.h> on arm64; ALEX uses only _mm_popcnt_u64 once ALEX_USE_LZCNT is 0.
#pragma once
static inline long long _mm_popcnt_u64(unsigned long long x) { return __builtin_popcountll(x); }
EOF
    cat > $W/include/omp.h <<'EOF'
// build_gre.sh (MIT): single-thread OpenMP (Apple clang has none; the pragmas are ignored), as OMP_NUM_THREADS=1.
#pragma once
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_max_threads() { return 1; }
EOF
    cat > $W/include/tbb/parallel_sort.h <<'EOF'
// build_gre.sh (MIT): tbb::parallel_sort -> std::sort (setup only, outside the timed region).
#pragma once
#include <algorithm>
namespace tbb {
template <class It> void parallel_sort(It first, It last) { std::sort(first, last); }
template <class It, class Cmp> void parallel_sort(It first, It last, const Cmp &cmp) { std::sort(first, last, cmp); }
}
EOF
    printf '%s\n' '// build_gre.sh (MIT): empty; benchmark.h includes jemalloc.h but uses nothing from it.' '#pragma once' \
        > $W/include/jemalloc/jemalloc.h
    # GRE's add_compile_options with -march=native -> -mcpu=native (the host core), plus CMake's Release flags.
    # __float128: none on arm64 (pgm_metric.h, --dataset_statistic only); random_shuffle: removed in C++17 libc++.
    FLAGS="-faligned-new -mcpu=native -g -O3 -include cstdint -O3 -DNDEBUG -std=gnu++17 -D_LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE -D__float128=long\\ double -I$W/include${XINC[*]+ ${XINC[*]}}"
    "$CLANG" -faligned-new -mcpu=native -g -O3 -include cstdint -O3 -DNDEBUG -std=gnu++17 \
        -D_LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE '-D__float128=long double' -I$W/include ${XINC[@]+"${XINC[@]}"} \
        src/benchmark/microbench.cpp -o build/microbench 2> build/compile.log || { tail -n 30 build/compile.log; exit 1; }
    COMPILER=$("$CLANG" --version | sed -n 1p)
fi

echo "== 4/4 manifest"
INDEXES=$(grep -oE '^  (else )?if ?\( ?index_type == "[a-z0-9_]+"' src/competitor/competitor.h | sed 's/.*"\(.*\)"/\1/' | paste -sd ' ' -)
{ echo "GRE $GRE_SHA ($(git config remote.origin.url)), built by GRE-baselines build_gre.sh"
  echo "repo: $(git -C "$HERE" rev-parse -q --verify HEAD || echo uncommitted)$(git -C "$HERE" diff --quiet 2>/dev/null || echo ' (dirty)')"
  echo "indexes: $INDEXES"
  echo "ALEX_USE_LZCNT: $ALEX_LZCNT"
  if [ "$EXTRA" = transforms ]; then
      echo "extra: transforms (scripts/add_transforms.py, $ROOT/include; include/gret cksum $(cat "$ROOT"/include/gret/*.h | cksum))"
      echo "transform indexes: pw<K>_B (K=2..65536) nf_B gate_B nflsw_B, B in ${KEEP//,/ }"
  fi
  for p in $SUBS; do echo "submodule $p: $(git -C "$p" rev-parse HEAD)"; done
  echo "edits (diff cksum): GRE $(git diff | cksum); alex $(git -C src/competitor/alex/src diff | cksum)"
  git diff --stat | sed 's/^/  /'
  echo "compiler: $COMPILER"
  echo "flags: $FLAGS"
  if [ "$OS" = Linux ]; then
      echo "host: $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) cpus, $(uname -r)"
      echo "deps: tbb-devel 2020.2, jemalloc 5.4.0 (conda-forge, $CONDA_PREFIX)"
  else
      echo "host: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.perflevel0.physicalcpu)P+$(sysctl -n hw.perflevel1.physicalcpu)E, macOS $(sw_vers -productVersion)"
      echo "PREVIEW build: libc++ (Atom: libstdc++), so --seed samples other operation lists; no jemalloc, no OpenMP"
  fi
  echo "microbench sha256: $( (command -v sha256sum >/dev/null && sha256sum build/microbench || shasum -a 256 build/microbench) | cut -c1-16)"
} > build/BUILD_MANIFEST.txt
cat build/BUILD_MANIFEST.txt
echo "built: $DEST/build/microbench"
