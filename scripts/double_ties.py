#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Count the adjacent pairs of a GRE key file's sorted distinct keys that are equal as IEEE doubles.

    python3 scripts/double_ties.py GRE_KEY_FILE        prints the count (gate.h's GateFeatures::double_ties)

Raw LIPP fails exactly when this is > 0 and long double is double (macOS arm64): two keys that convert to the same
long double land in the same slot at every level, so its bulk load recurses until memory runs out (or RT_ASSERT).
Only keys above 2^53 can tie. run_baselines.sh calls this on macOS for raw lipp on file names other than GRE's ten.
numpy if it imports (1.6 GB for 200M keys); else the standard library, which handles sorted files of a few 10M keys.
"""
import sys
from array import array

TWO53 = 1 << 53


def count_numpy(path):
    import numpy as np
    k = np.unique(np.fromfile(path, dtype='<u8', offset=8))  # sorted, distinct, as GRE bulk-loads them
    i = max(int(np.searchsorted(k, TWO53, side='right')) - 1, 0)
    d = k[i:].astype(np.float64)  # round to nearest, as LIPP's (long double) cast on arm64
    return int(np.count_nonzero(d[1:] == d[:-1]))


def count_stdlib(path):
    with open(path, 'rb') as f:
        n = array('Q', f.read(8))[0]
        k = array('Q')
        k.frombytes(f.read(8 * n))
    if sys.byteorder != 'little':
        k.byteswap()
    if any(k[j] >= k[j + 1] for j in range(len(k) - 1)):
        k = array('Q', sorted(set(k)))
    lo, hi = 0, len(k)  # first key above 2^53
    while lo < hi:
        mid = (lo + hi) // 2
        if k[mid] > TWO53:
            hi = mid
        else:
            lo = mid + 1
    return sum(1 for j in range(max(lo - 1, 0), len(k) - 1) if float(k[j]) == float(k[j + 1]))


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: double_ties.py GRE_KEY_FILE')
    try:
        import numpy  # noqa: F401
        c = count_numpy(sys.argv[1])
    except ImportError:
        c = count_stdlib(sys.argv[1])
    print(c)


if __name__ == '__main__':
    main()
