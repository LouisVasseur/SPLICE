#!/usr/bin/env bash
# SPDX-License-Identifier: MIT (SPLICE)
# Native arm64 macOS build of the gre_lite GRE, for PREVIEW numbers only; the reference numbers come from the Atom.
# See gre_mac.md for what differs from the Atom build.
#
# Runs a copy of the COMMITTED tools/gre_lite.sh in edit-only mode (so in-progress edits of gre_lite.sh cannot break
# it), then patches the GRE clone for arm64 and builds build/microbench with Apple clang. GRE has no licence and the
# STX B+tree is GPL-3: all of it stays in DEST; the repository gets only this script and gre_mac.md.
#
# Usage: experimental/scaleli/tools/gre_mac.sh [DEST]      (bash 3.2 is enough)
#   DEST                 GRE clone to create or reuse (default below)
#   GRE_MAC_SEED=DIR     if DEST does not exist, clone GRE and its 5 submodules from this local GRE clone instead of
#                        GitHub (offline); every file gre_lite.sh or this script edits is reset from git first anyway
#   GRE_MAC_LITE=FILE    gre_lite.sh to run instead of HEAD's (e.g. the working-tree one, to test new registrations)
# Every arm64 edit starts from the pristine file (git checkout), so the script is idempotent.
set -euo pipefail
DEST=${1:-$HOME/gre_mac/GRE}                              # outside the repo: GRE and STX must never be committed
SCALELI_DIR=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$SCALELI_DIR" rev-parse --show-toplevel)
GRE_SHA=e807edcef51df6732f07f94d4c797fb3897519ba          # must match gre_lite.sh
SUBS="src/competitor/alex/src src/competitor/lipp/src src/competitor/pgm/src src/competitor/btree/src src/competitor/artsync/src"
CLANG=${GRE_MAC_CLANG:-/usr/bin/clang++}

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "gre_mac.sh: for arm64 macOS only (Linux: gre_lite.sh)" >&2; exit 1; }

echo "== 1/5 GRE clone: $DEST"
if [ ! -d "$DEST/.git" ]; then
    mkdir -p "$(dirname "$DEST")"
    if [ -n "${GRE_MAC_SEED:-}" ]; then
        git clone -q --no-checkout "$GRE_MAC_SEED" "$DEST"
        # submodule URLs point at the seed's checked-out submodules; 'submodule init' keeps URLs already configured
        for p in $SUBS; do git -C "$DEST" config "submodule.$p.url" "$GRE_MAC_SEED/$p"; done
        git -C "$DEST" checkout -q "$GRE_SHA"
        # protocol.file.allow: git >= 2.38 refuses local-path submodule clones by default
        git -C "$DEST" -c protocol.file.allow=always submodule update -q --init $SUBS
    else
        git clone https://github.com/gre4index/GRE.git "$DEST"   # as gre_lite.sh does
    fi
fi
W=$DEST/build/gre_mac        # shims, compiler wrapper, gre_lite copy, objects (build/ is in GRE's .gitignore)
mkdir -p "$W/bin" "$W/include/tbb" "$W/include/jemalloc" "$W/obj"

echo "== 2/5 gre_lite.sh, edit-only"
if [ -n "${GRE_MAC_LITE:-}" ]; then cp "$GRE_MAC_LITE" "$W/gre_lite.sh"; LITE_FROM="$GRE_MAC_LITE"
else git -C "$REPO" show HEAD:experimental/scaleli/tools/gre_lite.sh > "$W/gre_lite.sh"
     LITE_FROM="HEAD $(git -C "$REPO" rev-parse HEAD):experimental/scaleli/tools/gre_lite.sh"; fi
# The copy lives outside tools/, so its own dirname-based SCALELI_DIR would be wrong: point it at this checkout.
LINE='SCALELI_DIR=$(cd "$(dirname "$0")/.." && pwd)   # experimental/scaleli of this SPLICE checkout (SCALE-LI cells)'
[ "$(grep -cxF -- "$LINE" "$W/gre_lite.sh" || true)" = 1 ] || { echo "SCALELI_DIR line not found once in gre_lite.sh" >&2; exit 1; }
LINE="$LINE" awk '$0 == ENVIRON["LINE"] { print "SCALELI_DIR=\"$GRE_MAC_SCALELI_DIR\"  # gre_mac.sh"; next } { print }' \
    "$W/gre_lite.sh" > "$W/gre_lite.sh.tmp" && mv "$W/gre_lite.sh.tmp" "$W/gre_lite.sh"
grep -q "^GRE_SHA=$GRE_SHA" "$W/gre_lite.sh" || { echo "gre_lite.sh pins another GRE commit than $GRE_SHA" >&2; exit 1; }
# arm64 clang reads -march=native as a generic v8.6-A (target-cpu apple-m1); -mcpu=native is the host (apple-m3).
# gre_lite.sh's CPU check runs "$CXX -march=native": route it through the same flags as the build.
cat > "$W/bin/c++" <<EOF
#!/bin/sh
# gre_mac.sh: $CLANG with -march=native mapped to -mcpu=native
for a; do shift; case \$a in -march=native) set -- "\$@" -mcpu=native ;; *) set -- "\$@" "\$a" ;; esac; done
exec $CLANG "\$@"
EOF
chmod +x "$W/bin/c++"
GRE_LITE_EDIT_ONLY=1 GRE_MAC_SCALELI_DIR="$SCALELI_DIR" CXX="$W/bin/c++" bash "$W/gre_lite.sh" "$DEST" > "$W/gre_lite.log"
tail -n 1 "$W/gre_lite.log"
cd "$DEST"
grep -q '^#define ALEX_USE_LZCNT 0$' src/competitor/alex/src/src/core/alex_nodes.h \
    || { echo "expected ALEX_USE_LZCNT 0 on arm64" >&2; exit 1; }

echo "== 3/5 arm64 edits (each from the pristine file)"
# edit FILE before|after|replace|delete ANCHOR < lines: ANCHOR must be a whole line occurring exactly once in FILE.
edit() {
    local n
    n=$(grep -cxF -- "$3" "$1" || true)
    [ "$n" = 1 ] || { echo "anchor found $n times in $1: $3" >&2; exit 1; }
    cat > "$1.add"   # a file, not NR == FNR: that test misfires when the added text is empty (delete)
    WHERE=$2 ANCHOR=$3 awk 'FILENAME == ARGV[1] { add = add $0 "\n"; next }
        $0 == ENVIRON["ANCHOR"] && ENVIRON["WHERE"] == "before" { printf "%s", add }
        $0 == ENVIRON["ANCHOR"] && ENVIRON["WHERE"] == "replace" { printf "%s", add; next }
        $0 == ENVIRON["ANCHOR"] && ENVIRON["WHERE"] == "delete" { next }
        { print }
        $0 == ENVIRON["ANCHOR"] && ENVIRON["WHERE"] == "after" { printf "%s", add }' "$1.add" "$1" > "$1.tmp"
    mv "$1.tmp" "$1"; rm -f "$1.add"
}
# (1) TSCNS (MIT, Meng Rao) reads the x86 TSC. On arm64 read the generic timer instead; TSCNS still calibrates its
# ticks against CLOCK_REALTIME in init(). No isb: like rdtsc, the read is not serialising.
git checkout -q -- src/tscns.h
edit src/tscns.h replace '  static int64_t rdtsc() { return __builtin_ia32_rdtsc(); }' <<'EOF'
#if defined(__aarch64__)  // gre_mac.sh: generic timer CNTVCT_EL0 (CNTFRQ 1 GHz on an M3 Max), readable at EL0
  static int64_t rdtsc() { int64_t t; asm volatile("mrs %0, cntvct_el0" : "=r"(t)); return t; }
#else
  static int64_t rdtsc() { return __builtin_ia32_rdtsc(); }
#endif
EOF
# (2) ALEX (MIT) CPUID class: x86 inline asm, which arm64 cannot compile even unused. Only cpu_supports_bmi() reads it,
# inside asserts of the ALEX_USE_LZCNT 1 paths (compiled out here, and NDEBUG).
ALEX_BASE=src/competitor/alex/src/src/core/alex_base.h
git -C src/competitor/alex/src checkout -q -- src/core/alex_base.h
edit "$ALEX_BASE" before '    asm volatile("cpuid"' <<'EOF'
#if !defined(__aarch64__)  // gre_mac.sh
EOF
edit "$ALEX_BASE" after '                 : "a"(i), "c"(j));' <<'EOF'
#else  // gre_mac.sh: no CPUID on arm64
    regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
EOF
# (2b) GRE's utils.h: cmpxchg/cmpxchgb are x86 asm with x86-only constraints; only XIndex (not built) calls them.
git checkout -q -- src/benchmark/utils.h
edit src/benchmark/utils.h before 'inline uint64_t cmpxchg(uint64_t *object, uint64_t expected,' <<'EOF'
#if !defined(__aarch64__)  // gre_mac.sh: x86 asm, used only by XIndex
EOF
edit src/benchmark/utils.h before '#endif  // HELPER_H' <<'EOF'
#endif  // gre_mac.sh
EOF
# (3) ART (artunsync) needs SSE2 (N16.cpp): not built on arm64. Drop it from gre_lite's registry and name list.
CH=src/competitor/competitor.h
edit "$CH" delete '#include "./artsync/artunsync.h"' </dev/null
edit "$CH" delete '  else if (index_type == "artunsync") index = new ARTUnsynchronizedInterface<KEY_TYPE, PAYLOAD_TYPE>;' </dev/null
sed 's/^\(  else { std::cout << "Could not find.*(gre_lite: .*\) artunsync /\1 /' "$CH" > "$CH.tmp" && mv "$CH.tmp" "$CH"
! grep -q artunsync "$CH" || { echo "artunsync still in $CH" >&2; exit 1; }
sed -i '' '1a\
// gre_mac.sh: artunsync (ART) removed, it needs SSE2.
' "$CH"
# (3b) Memory probe (gre_lite_patch.h, ours, rewritten by gre_lite.sh on every run). macOS malloc returns freed pages
# with MADV_FREE_REUSABLE: they leave the footprint at once but stay in resident_size until the kernel needs them, so
# resident_size after a purge still counts freed build temporaries (sortedarray's probe: ~22 B/key instead of 16).
# phys_footprint is the macOS figure that drops on free, the analogue of VmRSS after gre_lite's jemalloc purge.
GP=src/benchmark/gre_lite_patch.h
edit $GP replace '  mach_task_basic_info_data_t info;' <<'EOF'
  task_vm_info_data_t info;  // gre_mac.sh: phys_footprint, not resident_size (freed pages stay resident)
EOF
edit $GP replace '  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;' <<'EOF'
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
EOF
edit $GP replace '  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)' <<'EOF'
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
EOF
edit $GP replace '  return static_cast<long long>(info.resident_size);' <<'EOF'
  return static_cast<long long>(info.phys_footprint);
EOF
edit $GP delete '  struct rusage ru;' </dev/null
edit $GP replace '  return getrusage(RUSAGE_SELF, &ru) == 0 ? static_cast<long long>(ru.ru_maxrss) : -1;  // bytes on macOS' <<'EOF'
  task_vm_info_data_t info;  // gre_mac.sh: footprint peak, to match gre_lite_rss_bytes
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  return task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS
      ? static_cast<long long>(info.ledger_phys_footprint_peak) : -1;
EOF
# (4) Shims for what arm64 macOS lacks (our code, written whole on every run): x86intrin.h for ALEX's popcount,
# single-thread OpenMP and TBB, and an empty jemalloc.h (benchmark.h includes it, uses nothing from it).
cat > "$W/include/x86intrin.h" <<'EOF'
// gre_mac.sh (MIT): stands in for <x86intrin.h> on arm64. ALEX uses only _mm_popcnt_u64 once ALEX_USE_LZCNT is 0.
#pragma once
#if defined(__x86_64__) || defined(__i386__)
#error "gre_mac shim: arm64 only"
#endif
static inline long long _mm_popcnt_u64(unsigned long long x) { return __builtin_popcountll(x); }
EOF
cat > "$W/include/omp.h" <<'EOF'
// gre_mac.sh (MIT): single-thread OpenMP (Apple clang has no OpenMP; without -fopenmp the pragmas are ignored).
// Thread 0 only, as with OMP_NUM_THREADS=1 on the Atom: GRE's sampler seeds seed + 0, and every index here gets the
// same operation list. Not the Atom's list: libc++ and libstdc++ implement uniform_int_distribution and std::shuffle
// differently, so the same --seed samples other keys (and, with --init_table_ratio < 1, bulk-loads another subset).
#pragma once
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_max_threads() { return 1; }
EOF
cat > "$W/include/tbb/parallel_sort.h" <<'EOF'
// gre_mac.sh (MIT): tbb::parallel_sort -> std::sort (setup only; not in the timed region).
#pragma once
#include <algorithm>
namespace tbb {
template <class It> void parallel_sort(It first, It last) { std::sort(first, last); }
template <class It, class Cmp> void parallel_sort(It first, It last, const Cmp &cmp) { std::sort(first, last, cmp); }
}
EOF
cat > "$W/include/jemalloc/jemalloc.h" <<'EOF'
// gre_mac.sh (MIT): NOT jemalloc, intentionally empty. MALLCTL_ARENAS_ALL stays undefined, so gre_lite_purge() takes
// only its macOS branch (malloc_zone_pressure_relief on the system allocator).
#pragma once
EOF

echo "== 4/5 build (Apple clang, -mcpu=native)"
# Mirror gre_lite's CMakeLists.txt without CMake (its find_package(OpenMP/JeMalloc/TBB) cannot succeed here).
# Stop if it sets anything this mirror does not reproduce.
CM=CMakeLists.txt
KNOWN_OPTS='add_compile_options(-faligned-new -march=native -g -O3 -include cstdint)'
[ "$(grep -c '^add_compile_options' $CM)" = 1 ] && grep -qxF "$KNOWN_OPTS" $CM || { echo "$CM: global options changed" >&2; exit 1; }
ODD=$(grep -E '^[[:space:]]*(add_definitions|add_compile_definitions|target_compile_definitions|target_compile_features|set_source_files_properties|set_target_properties|target_compile_options|target_include_directories|include_directories|set\(CMAKE_CXX_FLAGS)' $CM \
      | sed 's/[[:space:]]*#.*$//' \
      | grep -vE '^set_target_properties\([a-z0-9_]+ PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON\)$' \
      | grep -vE '^target_compile_options\([a-z0-9_]+ PRIVATE -ffp-contract=off\)$' \
      | grep -vE '^target_include_directories\([a-z0-9_]+ PRIVATE \$\{SCALELI_INCLUDE\}\)$' \
      | grep -vxF 'include_directories(${TBB_INCLUDE_DIRS} ${JEMALLOC_INCLUDE_DIR})' || true)
[ -z "$ODD" ] || { printf '%s: settings gre_mac.sh does not mirror:\n%s\n' "$CM" "$ODD" >&2; exit 1; }
LIBS=$(sed -n 's/^add_library(\([a-z0-9_]*\) STATIC .*/\1/p' $CM)
for l in $LIBS; do   # every SPLICE library: C++20, -ffp-contract=off, SCALELI_INCLUDE, linked into microbench
    for pat in "set_target_properties($l PROPERTIES CXX_STANDARD 20" "target_compile_options($l PRIVATE -ffp-contract=off)" \
               "target_include_directories($l PRIVATE \${SCALELI_INCLUDE})" "target_link_libraries(microbench PUBLIC $l)"; do
        grep -qF "$pat" $CM || { echo "$CM: library $l lacks '$pat'" >&2; exit 1; }
    done
done
TUS=$(grep -oE '\$\{CMAKE_CURRENT_SOURCE_DIR\}/src/[^ )]+\.cpp' $CM | sed 's|^\${CMAKE_CURRENT_SOURCE_DIR}/||')
echo "$TUS" | grep -qx src/benchmark/microbench.cpp || { echo "$CM: no microbench.cpp" >&2; exit 1; }
[ "$(echo "$TUS" | grep -vxc src/benchmark/microbench.cpp)" = "$(echo "$LIBS" | grep -c .)" ] \
    || { echo "$CM: expected one source per add_library" >&2; exit 1; }
# GRE's options with -march=native -> -mcpu=native, plus CMake's Release flags; CMake's default CXX_EXTENSIONS gives gnu++.
COMMON="-faligned-new -mcpu=native -g -O3 -include cstdint -O3 -DNDEBUG"
# __float128: arm64 clang has none; only pgm_metric.h (--dataset_statistic) uses it, and long double is double here.
# random_shuffle (removed in C++17; libstdc++ keeps it, Apple libc++ needs this switch): data_shift and insert key sets.
MBF="-std=gnu++17 -D_LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE"
# The SPLICE-registering gre_lite gives microbench the SCALE-LI include path too (splice/layout.hpp inlines into it).
grep -qE '^target_include_directories\(microbench PRIVATE \$\{SCALELI_INCLUDE\}\)' $CM && MBF="$MBF -I$SCALELI_DIR/include"
FAC="-std=gnu++20 -ffp-contract=off -I$SCALELI_DIR/include"
OBJS=""
: > "$W/compile.log"
for tu in $TUS; do
    [ "$tu" = src/benchmark/microbench.cpp ] && continue
    o="$W/obj/$(basename "$tu" .cpp).o"
    echo "   $tu (gnu++20)"
    "$CLANG" $COMMON $FAC -c "$tu" -o "$o" 2>>"$W/compile.log"
    OBJS="$OBJS $o"
done
echo "   src/benchmark/microbench.cpp (gnu++17; GRE's warnings in $W/compile.log)"
"$CLANG" $COMMON $MBF '-D__float128=long double' -I"$W/include" -c src/benchmark/microbench.cpp \
    -o "$W/obj/microbench.o" 2>>"$W/compile.log"
"$CLANG" "$W/obj/microbench.o" $OBJS -o build/microbench

echo "== 5/5 build record"
INDEXES=$(sed -n 's/.*(gre_lite: \([a-z0-9_ ]*\)).*/\1/p' "$CH")
{ echo "GRE $GRE_SHA, native arm64 macOS build by SPLICE gre_mac.sh: PREVIEW numbers only (see tools/gre_mac.md)"
  echo "gre_lite.sh: $LITE_FROM, cksum $(cksum < "$W/gre_lite.sh") (copy, SCALELI_DIR line pointed at $SCALELI_DIR)"
  echo "GRE origin: $(git config remote.origin.url)"
  echo "indexes: $INDEXES"
  echo "arm64 edits (each from the pristine file):"
  echo "  src/tscns.h: rdtsc -> CNTVCT_EL0 on aarch64; diff cksum $(git diff -- src/tscns.h | cksum)"
  echo "  src/benchmark/utils.h: cmpxchg asm compiled out on aarch64; diff cksum $(git diff -- src/benchmark/utils.h | cksum)"
  echo "  $ALEX_BASE: CPUID asm compiled out on aarch64; diff cksum $(git -C src/competitor/alex/src diff -- src/core/alex_base.h | cksum)"
  echo "  $GP: macOS RSS = phys_footprint, peak = ledger_phys_footprint_peak; cksum $(cksum < $GP)"
  echo "  $CH: gre_lite's file minus artunsync; cksum $(cksum < "$CH")"
  for f in x86intrin.h omp.h tbb/parallel_sort.h jemalloc/jemalloc.h; do echo "  shim $f: cksum $(cksum < "$W/include/$f")"; done
  echo "compiler: $("$CLANG" --version | sed -n 1p) ($CLANG)"
  echo "target-cpu: $("$CLANG" -### -mcpu=native -x c++ -c /dev/null -o /dev/null 2>&1 | tr ' ' '\n' | grep -A1 -e '"-target-cpu"' | tail -n 1 | tr -d '"')"
  # Not visible in the flags, and each changes what is measured relative to the Atom (gre_mac.md):
  echo "C++ library: libc++ (Atom: libstdc++), so --seed gives other operation lists than on the Atom"
  echo "FP: hardware FMA (std::fma in SCALE-LI is one instruction; clang contracts a*x+b in GRE's TU), long double = double"
  echo "flags (all TUs): $COMMON"
  echo "flags microbench.cpp: $MBF '-D__float128=long double' -I$W/include"
  echo "flags $(echo "$TUS" | grep -vx src/benchmark/microbench.cpp | paste -sd ' ' -): $FAC"
  echo "host: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.perflevel0.physicalcpu)P+$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0)E cores, page $(sysctl -n hw.pagesize) B, P-core L1d $(sysctl -n hw.perflevel0.l1dcachesize) B, L2 $(sysctl -n hw.perflevel0.l2cachesize) B, RAM $(sysctl -n hw.memsize) B, macOS $(sw_vers -productVersion)"
  # the file differs per link (LC_UUID, debug-map timestamps); its code and data do not
  echo "microbench code+data sha1: $( (otool -t build/microbench; otool -d build/microbench) | grep -v '^build/' | shasum | cut -c1-16)"
  otool -L build/microbench | sed -n '2,$p' | sed 's/^[[:space:]]*/linked: /'
  echo "--- gre_lite_build.txt (edit-only run) ---"
  cat build/gre_lite_build.txt
} > build/gre_mac_build.txt
cat build/gre_mac_build.txt
echo "built: $DEST/build/microbench   (indexes: $INDEXES)"
