#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build-enabling trim of a pristine GRE clone: keep only the competitors that compile without AVX2 or MKL.

    python3 trim_gre.py GRE_DIR alex,lipp,pgm,btree,artunsync

Only deletes lines of GRE's own src/competitor/competitor.h and CMakeLists.txt (plus one 'else if' -> 'if' and one
compile flag), each anchored on an exact line that must occur once, so no GRE text is written from scratch and any
change upstream stops the build. GRE's driver and timed loop (src/benchmark/) are not touched. Python 3.6, stdlib.
"""
import re
import sys

root, keep = sys.argv[1], set(sys.argv[2].split(','))
# The header each kept competitor needs (GRE's own include lines); every other competitor include is dropped.
HEADER = {'alex': '#include "./alex/alex.h"', 'lipp': '#include "./lipp/lipp.h"', 'pgm': '#include "pgm/pgm.h"',
          'btree': '#include "btree/btree.h"', 'artunsync': '#include "./artsync/artunsync.h"'}
ALWAYS = {'#include"./indexInterface.h"', '#include "iostream"'}


def die(msg):
    sys.exit('trim_gre.py: ' + msg)


def once(lines, text):
    n = sum(1 for l in lines if l == text)
    if n != 1:
        die('expected exactly one line %r, found %d (GRE changed?)' % (text, n))
    return lines.index(text)


p = root + '/src/competitor/competitor.h'
lines = open(p).read().split('\n')
for k in keep:
    if k not in HEADER:
        die('no known header for %s' % k)
    once(lines, HEADER[k])
want = ALWAYS | set(HEADER[k] for k in keep)
lines = [l for l in lines if not (l.startswith('#include') and l not in want)]
out, i, kept = [], 0, []
block = re.compile(r'^  (else )?if ?\( ?index_type == "([a-z0-9_]+)"\) \{$')
while i < len(lines):
    m = block.match(lines[i])
    if m:  # a 3-line registration: the test, 'index = new ...;', '}'
        if lines[i + 2] != '  }':
            die('unexpected registration block at line %d' % (i + 1))
        if m.group(2) in keep:
            kept.append(m.group(2))
            out.extend(lines[i:i + 3])
        i += 3
        continue
    out.append(lines[i])
    i += 1
if sorted(kept) != sorted(keep):
    die('registrations kept %s, wanted %s' % (sorted(kept), sorted(keep)))
first = next(j for j, l in enumerate(out) if block.match(l))
out[first] = out[first].replace('  else if', '  if', 1)  # the chain must start with 'if' after drops
open(p, 'w').write('\n'.join(out))

p = root + '/CMakeLists.txt'
lines = open(p).read().split('\n')
for text in ('find_package(MKL REQUIRED)', 'include_directories(${MKL_INCLUDE_DIRS})',
             'add_subdirectory(src/competitor/wormhole_u64)', 'add_subdirectory(src/competitor/masstree)',
             '        WORMHOLEU64', '        MASSTREE'):
    del lines[once(lines, text)]
link = 'target_link_libraries(microbench PUBLIC OpenMP::OpenMP_CXX ${JEMALLOC_LIBRARIES} ${MKL_LIBRARIES} ${TBB_LIBRARIES})'
lines[once(lines, link)] = link.replace(' ${MKL_LIBRARIES}', '')
opts = 'add_compile_options(-faligned-new -march=native -g -O3)'
# GCC 11+ no longer pulls <cstdint> in transitively, and some competitor headers rely on it.
lines[once(lines, opts)] = opts.replace('-O3)', '-O3 -include cstdint)')
open(p, 'w').write('\n'.join(lines))
print('trim_gre.py: kept %s' % ','.join(sorted(keep)))
