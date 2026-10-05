#!/usr/bin/env python3
"""Apply INTEGRATION.md's edits to tools/gre_lite.sh and tools/gre_run.sh (SCALE-LI cells in GRE).

    python3 apply_integration.py gre_lite.sh FILE [--check]
    python3 apply_integration.py gre_run.sh FILE [--check]

Each edit is anchored on text, never on a line number, and every anchor must occur exactly once; otherwise nothing is
written and the failing anchor is printed (then apply that edit by hand from INTEGRATION.md). Running it twice is
refused: the first edit's marker is already present. --check reports without writing. Python 3.6+, stdlib only.
"""
import sys

# (kind, anchor, text): 'before'/'after' insert whole lines next to the line equal to anchor (whitespace included);
# 'sub' replaces the substring anchor with text.
LITE = [
    ('after', "# plus 'sortedarray', our own binary-search baseline.",
     "# SCALE-LI (SPLICE): scaleli_b scaleli_n scaleli_c scaleli_cr scaleli_nc scaleli_j scaleli_jg0 scaleli_splice, the protocol cells of\n"
     "# results/aidb_ba/run_ba.py plus the joint G+T+V root, from integrations/gre/ (README.md 'GRE cells'). They\n"
     "# read SCALELI_FLOW (per-dataset flow file, cells _n _nc _j _jg0) and SCALELI_BUILD_THREADS (default 16)."),
    ('before', 'cd "$DEST"',
     'SCALELI_DIR=$(cd "$(dirname "$0")/.." && pwd)   # experimental/scaleli of this SPLICE checkout (SCALE-LI cells)'),
    ('sub', "// Trimmed by SPLICE gre_lite.sh: ALEX, LIPP, PGM, STX B+tree, ART only (no AVX2 / MKL needed), plus SPLICE's sortedarray.",
     "// Trimmed by SPLICE gre_lite.sh: ALEX, LIPP, PGM, STX B+tree, ART only (no AVX2 / MKL needed), plus SPLICE's sortedarray\n"
     "// and SCALE-LI (scaleli_*)."),
    ('after', '#include "./sortedarray/sortedarray.h"', '#include "./scaleli/scaleli_interface.h"'),
    ('after', '  else if (index_type == "sortedarray") index = new SortedArrayInterface<KEY_TYPE, PAYLOAD_TYPE>;',
     '  else if (index_type == "scaleli_b") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("B");\n'
     '  else if (index_type == "scaleli_n") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("N");\n'
     '  else if (index_type == "scaleli_c") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("C");\n'
     '  else if (index_type == "scaleli_nc") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("NC");\n'
     '  else if (index_type == "scaleli_j") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("J");\n'
     '  else if (index_type == "scaleli_jg0") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("Jg0");\n'
     '  else if (index_type == "scaleli_splice") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("Jg0");\n'
     '  else if (index_type == "scaleli_cr") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("Cr");'),
    ('sub', '(gre_lite: alex lipp pgm btree artunsync sortedarray)',
     '(gre_lite: alex lipp pgm btree artunsync sortedarray scaleli_b scaleli_n scaleli_c scaleli_cr scaleli_nc scaleli_j scaleli_jg0 scaleli_splice)'),
    ('after', 'target_link_libraries(microbench PUBLIC OpenMP::OpenMP_CXX ${JEMALLOC_LIBRARIES} ${TBB_LIBRARIES})',
     '# SPLICE SCALE-LI: C++20 in its own TU (GRE\'s TU is C++17 and ALEX needs that), linked into microbench.\n'
     '# -ffp-contract=off: with -march=native on an FMA host, GCC would fuse multiply-adds the portable scaleli_bench does not.\n'
     'set(SCALELI_INCLUDE "" CACHE PATH "SPLICE experimental/scaleli/include")\n'
     'if(NOT EXISTS "${SCALELI_INCLUDE}/scaleli/index.hpp")\n'
     '  message(FATAL_ERROR "SCALELI_INCLUDE must point to SPLICE experimental/scaleli/include")\n'
     'endif()\n'
     'find_package(Threads REQUIRED)\n'
     'add_library(scaleli_gre STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/scaleli/scaleli_gre.cpp)\n'
     'set_target_properties(scaleli_gre PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)\n'
     'target_include_directories(scaleli_gre PRIVATE ${SCALELI_INCLUDE})\n'
     'target_compile_options(scaleli_gre PRIVATE -ffp-contract=off)\n'
     'target_link_libraries(scaleli_gre PUBLIC Threads::Threads)\n'
     'target_link_libraries(microbench PUBLIC scaleli_gre)'),
    ('after', 'mkdir -p src/competitor/sortedarray',
     '# SCALE-LI wrapper and facade (SPLICE, MIT), copied whole on every run; the core headers are read in place.\n'
     'mkdir -p src/competitor/scaleli\n'
     'cp "$SCALELI_DIR"/integrations/gre/scaleli_interface.h "$SCALELI_DIR"/integrations/gre/scaleli_gre.hpp \\\n'
     '   "$SCALELI_DIR"/integrations/gre/scaleli_gre.cpp src/competitor/scaleli/'),
    ('sub', '-DTBB_ROOT_DIR="$CONDA_PREFIX" -DJEMALLOC_ROOT_DIR="$CONDA_PREFIX"',
     '-DTBB_ROOT_DIR="$CONDA_PREFIX" -DJEMALLOC_ROOT_DIR="$CONDA_PREFIX" \\\n'
     '          -DSCALELI_INCLUDE="$SCALELI_DIR/include"'),
    ('after', '  echo "  src/competitor/sortedarray/sortedarray.h: SPLICE, cksum $(cksum < src/competitor/sortedarray/sortedarray.h)"',
     '  for f in src/competitor/scaleli/*; do echo "  $f: SPLICE, cksum $(cksum < "$f")"; done\n'
     '  echo "  SCALE-LI headers $SCALELI_DIR/include/scaleli: cksum $(cat "$SCALELI_DIR"/include/scaleli/*.hpp | cksum)"\n'
     '  echo "  SPLICE $(git -C "$SCALELI_DIR" rev-parse HEAD 2>/dev/null || echo unknown)'
     ' ($(git -C "$SCALELI_DIR" status --porcelain -- . 2>/dev/null | wc -l | tr -d \' \') uncommitted paths in experimental/scaleli)"'),
    ('sub', '(indexes: alex lipp pgm btree artunsync sortedarray)',
     '(indexes: alex lipp pgm btree artunsync sortedarray scaleli_b scaleli_n scaleli_c scaleli_cr scaleli_nc scaleli_j scaleli_jg0 scaleli_splice)'),
]

RUN = [
    ('after', '#       [--table-size -1] [--read 1 --insert 0] [--resume] [-- extra microbench flags]',
     '#       [--flows DIR] [--build-threads 16]   (SCALE-LI cells scaleli_b _n _c _nc _j _jg0, integrations/gre/)'),
    ('after', 'DATASETS= INDEXES= REPEATS=3 OPS=100000000 WARMUP=0 INIT_RATIO=1 PIN=-1 TABLE_SIZE=-1 READ=1 INSERT=0 RESUME=0',
     '# SCALE-LI: the per-dataset NFL flow files (tracked in SPLICE) and run_ba.py\'s 16 build threads\n'
     'FLOWS=$(cd "$(dirname "$0")/.." && pwd)/results/aidb_flowv2/flows_free\n'
     'BUILD_THREADS=16'),
    ('after', '    --insert) INSERT=$2; shift 2;;',
     '    --flows) FLOWS=$2; shift 2;;\n'
     '    --build-threads) BUILD_THREADS=$2; shift 2;;'),
    ('after', 'range --repeats "$REPEATS" 1 1000',
     'int --build-threads "$BUILD_THREADS"; range --build-threads "$BUILD_THREADS" 1 1024'),
    ('after', 'extra=${EXTRA[*]+${EXTRA[*]}}"',
     '# SCALE-LI cells: flow file per dataset with run_ba.py\'s PERIOD; recorded only when a scaleli_* index runs, so\n'
     '# OUTDIRs of the baseline indexes resume as before.\n'
     'period() { case $1 in fb|osm|planet) echo s512;; *) echo s64;; esac; }\n'
     'flow_cell() { case $1 in scaleli_n|scaleli_nc|scaleli_j|scaleli_jg0|scaleli_splice) return 0;; *) return 1;; esac; }\n'
     'case ",$INDEXES," in *,scaleli_*) CONFIG="$CONFIG\n'
     'flows=$FLOWS\n'
     'build_threads=$BUILD_THREADS\n'
     'scaleli_args=${SCALELI_ARGS-}";; esac\n'
     'for idx in "${IX[@]}"; do\n'
     '  flow_cell "$idx" || continue\n'
     '  for ds in "${DS[@]}"; do\n'
     '    f=$FLOWS/${ds}_$(period "$ds")_t2000.txt\n'
     '    [ -f "$f" ] && [ -r "$f" ] || die "$idx needs the flow file $f (--flows DIR)"\n'
     '  done\n'
     'done'),
    ('sub', "'gre_lite: [a-z ]*)'", "'gre_lite: [a-z0-9_ ]*)'"),
    ('after', '  [ "$PIN" = -1 ] || [ "$pc" = "$PIN" ] || bad="${bad:+$bad, }not pinned to core $PIN"',
     '  case $log in *__scaleli_*)  # the facade prints scaleli_cell after bulk_load, scaleli_error on any failure\n'
     '    grep -q \'^scaleli_cell: \' "$log" || bad="${bad:+$bad, }no scaleli_cell line"\n'
     '    ! grep -q \'scaleli_error\' "$log" || bad="${bad:+$bad, }scaleli_error";; esac'),
    ('before', '      echo "cmd: OMP_NUM_THREADS=1 ${cmd[*]}" > "$log"',
     '      runenv=(OMP_NUM_THREADS=1)\n'
     '      case $idx in scaleli_*) runenv+=(SCALELI_BUILD_THREADS="$BUILD_THREADS");; esac\n'
     '      if flow_cell "$idx"; then runenv+=(SCALELI_FLOW="$FLOWS/${ds}_$(period "$ds")_t2000.txt"); fi'),
    ('sub', '      echo "cmd: OMP_NUM_THREADS=1 ${cmd[*]}" > "$log"',
     '      echo "cmd: ${runenv[*]} ${cmd[*]}" > "$log"'),
    ('sub', '      OMP_NUM_THREADS=1 "${cmd[@]}" >> "$log" 2>&1 & child=$!',
     '      env "${runenv[@]}" "${cmd[@]}" >> "$log" 2>&1 & child=$!'),
]

EDITS = {'gre_lite.sh': LITE, 'gre_run.sh': RUN}


def apply(text, edits):
    errors = []
    first = edits[0][2].split('\n')[0]
    if first in text:
        return text, ['already applied: "%s" is present' % first]
    for kind, anchor, add in edits:
        if kind == 'sub':
            n = text.count(anchor)
            if n != 1:
                errors.append('substring found %d times: %s' % (n, anchor))
                continue
            text = text.replace(anchor, add)
            continue
        lines = text.split('\n')
        hits = [i for i, l in enumerate(lines) if l == anchor]
        if len(hits) != 1:
            errors.append('anchor line found %d times: %s' % (len(hits), anchor))
            continue
        i = hits[0] + (1 if kind == 'after' else 0)
        lines[i:i] = add.split('\n')
        text = '\n'.join(lines)
    return text, errors


def main():
    args = [a for a in sys.argv[1:] if a != '--check']
    if len(args) != 2 or args[0] not in EDITS:
        sys.exit(__doc__)
    with open(args[1]) as f:
        text = f.read()
    out, errors = apply(text, EDITS[args[0]])
    for e in errors:
        print('apply_integration: %s: %s' % (args[1], e), file=sys.stderr)
    if errors:
        sys.exit(1)
    if '--check' in sys.argv:
        print('%s: all %d edits apply' % (args[1], len(EDITS[args[0]])))
        return
    with open(args[1], 'w') as f:
        f.write(out)
    print('%s: %d edits applied' % (args[1], len(EDITS[args[0]])))


if __name__ == '__main__':
    main()
