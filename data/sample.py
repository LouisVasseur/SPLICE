#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stride samples of the GRE key files: n evenly spaced keys of each file's sorted distinct keys, as GRE files.

    python3 data/sample.py SRC_DIR OUT_DIR [--n 10000000] [--datasets "covid genome ..."] [--force]

For each dataset NAME (default: the ten GRE files) it writes OUT_DIR/NAME_10M (the suffix follows --n: 10M, 2M, 500k):
the keys of rank floor(i (U - 1) / (n - 1)), i = 0 .. n-1, among the U sorted distinct keys (GRE sorts and dedups
before bulk loading). So the sample keeps the CDF shape and the exact minimum and maximum (fb's outliers up to
2^64-1), unlike --table_size, which reads a file prefix (for osm, books and fb, stored sorted, the 5% smallest keys).
Format: a uint64 count, then the keys, little-endian, sorted (GRE shuffles them on load). OUT_DIR/SAMPLES.txt holds
the sha256 of each sample (sha256sum -c format); a sample already listed there with the same hash is not redone.

Dev tool: needs numpy (on the Mac, /usr/bin/python3). Memory: one copy of a source file's keys (1.6 GB for 200M);
sorting and deduplication are in place, the scans in chunks. Heavy enough to disturb a timed run: run it between runs.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

DATASETS = 'covid genome libio osm books fb history planet stack wise'
CHUNK = 1 << 24  # keys per chunk in the scans (128 MB of temporaries)


def die(msg):
    sys.exit('sample.py: ' + msg)


def tag(n):
    for d, s in ((10 ** 9, 'G'), (10 ** 6, 'M'), (10 ** 3, 'k')):
        if n % d == 0:
            return '%d%s' % (n // d, s)
    return str(n)


def load(path):
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        count = int(np.fromfile(f, dtype='<u8', count=1)[0]) if size >= 8 else -1
        if size != 8 * (count + 1):
            die('%s: %d bytes, not 8 + 8 x count (count %d): not a GRE uint64 file' % (path, size, count))
        return np.fromfile(f, dtype='<u8', count=count)


def is_sorted(a):
    for s in range(0, len(a) - 1, CHUNK):  # windows overlap by one key, so every adjacent pair is compared
        e = min(s + CHUNK + 1, len(a))
        if not np.all(a[s + 1:e] >= a[s:e - 1]):
            return False
    return True


def dedup_sorted(a):
    """Moves the distinct keys of sorted a to its front, chunk by chunk; returns how many there are."""
    w, prev = 0, None
    for s in range(0, len(a), CHUNK):
        c = a[s:s + CHUNK]
        keep = np.empty(len(c), dtype=bool)
        keep[0] = prev is None or c[0] != prev
        np.not_equal(c[1:], c[:-1], out=keep[1:])
        prev = int(c[-1])  # read before the write below, which may overwrite this chunk (w <= s)
        kept = c[keep]
        a[w:w + len(kept)] = kept
        w += len(kept)
    return w


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('src_dir')
ap.add_argument('out_dir')
ap.add_argument('--n', type=int, default=10000000, help='keys per sample (default 10000000)')
ap.add_argument('--datasets', default=DATASETS, help='space-separated file names in SRC_DIR (default: the ten)')
ap.add_argument('--force', action='store_true', help='redo samples already listed in SAMPLES.txt')
a = ap.parse_args()
if a.n < 1:
    die('--n must be positive')
os.makedirs(a.out_dir, exist_ok=True)
names = a.datasets.split()
for d in names:
    if not os.path.isfile(os.path.join(a.src_dir, d)):
        die('no %s in %s' % (d, a.src_dir))
listing = os.path.join(a.out_dir, 'SAMPLES.txt')
sums = {}
if os.path.exists(listing):
    for line in open(listing):
        h, name = line.rstrip('\n').split('  ', 1)
        sums[name] = h


def write_listing():
    with open(listing + '.part', 'w') as f:
        for name in sorted(sums):
            f.write('%s  %s\n' % (sums[name], name))
    os.replace(listing + '.part', listing)


for d in names:
    out = '%s_%s' % (d, tag(a.n))
    dst = os.path.join(a.out_dir, out)
    if not a.force and out in sums and os.path.exists(dst) and sha256(dst) == sums[out]:
        print('%s: present (%s)' % (out, sums[out][:16]))
        continue
    t0 = time.time()
    keys = load(os.path.join(a.src_dir, d))
    was_sorted = is_sorted(keys)
    if not was_sorted:
        keys.sort()  # in place (introsort), no second copy
    u = dedup_sorted(keys)
    if u < a.n:
        die('%s has %d distinct keys, fewer than --n %d' % (d, u, a.n))
    pos = np.arange(a.n, dtype=np.int64)
    if a.n > 1:
        if a.n * (u - 1) >= 2 ** 63:
            die('n x U overflows int64')
        pos = pos * (u - 1) // (a.n - 1)  # strictly increasing since U >= n: n distinct keys, first and last kept
    sample = keys[pos]
    del keys
    assert len(sample) == a.n and (a.n == 1 or np.all(sample[1:] > sample[:-1]))
    with open(dst + '.part', 'wb') as f:
        np.array([a.n], dtype='<u8').tofile(f)
        sample.astype('<u8', copy=False).tofile(f)
    os.replace(dst + '.part', dst)
    sums[out] = sha256(dst)
    write_listing()
    print('%s: %d of %d distinct keys (%s), min %d max %d, %.1f s, sha256 %s' % (
        out, a.n, u, 'stored sorted' if was_sorted else 'sorted here', sample[0], sample[-1], time.time() - t0,
        sums[out][:16]))
