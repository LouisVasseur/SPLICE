#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Register the transform indexes (include/gret/registry.h) in a GRE clone trimmed by trim_gre.py.

    python3 add_transforms.py GRE_DIR REPO_ROOT [--cmake | --no-cmake]

Run by build_gre.sh with EXTRA=transforms, right after trim_gre.py. Adds two lines to src/competitor/competitor.h:
'#include "gret/registry.h"' after '#include "btree/btree.h"' (the last competitor include, so every base interface is
declared before registry.h), and one 'else if' asking gret::make_index right before GRE's "Could not find" branch, so
GRE's own names still match first and an unknown name still exits as before. With --cmake (the default on Linux) it
also adds include_directories("REPO_ROOT/include") to CMakeLists.txt; the macOS build has no CMake and passes
-I REPO_ROOT/include itself. Each anchor must be a whole line occurring exactly once. Python 3.6, stdlib.
"""
import os
import sys

INCLUDE = '#include "gret/registry.h"'
CHAIN = '  else if ((index = gret::make_index<KEY_TYPE, PAYLOAD_TYPE>(index_type)) != nullptr) {}'


def die(msg):
    sys.exit('add_transforms.py: ' + msg)


def once(lines, text):
    n = sum(1 for l in lines if l == text)
    if n != 1:
        die('expected exactly one line %r, found %d (not trimmed, or GRE changed?)' % (text, n))
    return lines.index(text)


args = [a for a in sys.argv[1:] if not a.startswith('--')]
flags = [a for a in sys.argv[1:] if a.startswith('--')]
if len(args) != 2 or not set(flags) <= {'--cmake', '--no-cmake'} or len(flags) > 1:
    die('usage: add_transforms.py GRE_DIR REPO_ROOT [--cmake | --no-cmake]')
root, repo = args[0], os.path.abspath(args[1])
cmake = flags == ['--cmake'] or (not flags and sys.platform.startswith('linux'))
if not os.path.isfile(os.path.join(repo, 'include', 'gret', 'registry.h')):
    die('no %s/include/gret/registry.h' % repo)

p = root + '/src/competitor/competitor.h'
lines = open(p).read().split('\n')
if INCLUDE in lines or CHAIN in lines:
    die('%s already has the transform registry (build_gre.sh restores the pristine file first)' % p)
i = once(lines, '#include "btree/btree.h"')
late = [l for l in lines[i + 1:] if l.startswith('#include') and l != '#include "iostream"']
if late:
    die('competitor includes after btree.h would be invisible to registry.h: %s' % late)
lines.insert(i + 1, INCLUDE)
j = once(lines, '  else {')
if 'Could not find a matching index' not in lines[j + 1]:
    die("'  else {' is not GRE's \"Could not find\" branch: %r" % lines[j + 1])
lines.insert(j, CHAIN)
open(p, 'w').write('\n'.join(lines))

if cmake:
    inc = os.path.join(repo, 'include')
    if any(c in inc for c in '"\\$;'):
        die('include path %r needs CMake escaping' % inc)
    p = root + '/CMakeLists.txt'
    lines = open(p).read().split('\n')
    lines.insert(once(lines, 'include_directories(${JEMALLOC_INCLUDE_DIR})') + 1, 'include_directories("%s")' % inc)
    open(p, 'w').write('\n'.join(lines))
print('add_transforms.py: registry in competitor.h%s' % (', include dir in CMakeLists.txt' if cmake else ''))
