#!/usr/bin/env python3
"""Check that a SCALE-LI index built through the GRE facade is the index scaleli_bench builds for the same cell.

    python3 compare.py BENCH_JSON LOG [LOG ...]

BENCH_JSON: scaleli_bench output for the cell on the same keys (--load-ratio 1, --limit = GRE's --table_size).
LOG: a GRE microbench log (index scaleli_*) or a facade_check log. Checked, exactly:
  scaleli_memory       == memory_before (all fields)
  scaleli_learnability == learnability (timing fields excluded: smoothing_ns, transform_ns, root_joint_build_ns)
  facade_check only:   checksum == result_checksum (read-only trace), counters == work_counters (same names)
  GRE only:            Memory == scaleli_accounted_bytes, success_read == operations_num on a read-only run
Exit status 1 on any mismatch. Python 3.6+, stdlib only (runs on the server's system python).
"""
import json
import sys

TIMING = {'smoothing_ns', 'transform_ns', 'root_joint_build_ns'}


def lines(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            for sep in (': ', ' = '):
                if sep in line:
                    k, v = line.split(sep, 1)
                    out.setdefault(k.strip(), v.strip())
                    break
    return out


def diff(a, b, where, skip=()):
    bad = []
    for k in sorted(set(a) | set(b)):
        if k in skip:
            continue
        if k not in a or k not in b:
            bad.append('%s.%s only in %s' % (where, k, 'bench' if k in a else 'log'))
        elif a[k] != b[k]:
            bad.append('%s.%s: bench %r != log %r' % (where, k, a[k], b[k]))
    return bad


def check(bench, path):
    kv = lines(path)
    bad = []
    if 'scaleli_error' in open(path).read():
        bad.append('scaleli_error in log')
    if 'scaleli_memory' not in kv or 'scaleli_learnability' not in kv:
        return bad + ['no scaleli_memory / scaleli_learnability lines (not a scaleli_* run?)']
    bad += diff(bench['memory_before'], json.loads(kv['scaleli_memory']), 'memory')
    learn = json.loads(kv['scaleli_learnability'])
    bad += diff(bench['learnability'], learn, 'learnability', TIMING)
    if 'checksum' in kv:  # facade_check
        if kv['checksum'] != bench['result_checksum']:
            bad.append('checksum %s != result_checksum %s' % (kv['checksum'], bench['result_checksum']))
        counters = json.loads(kv['counters'])
        wc = {k: v for k, v in bench['work_counters'].items() if k in counters}
        bad += diff(wc, counters, 'work_counters')
    if 'Memory' in kv:  # GRE
        if kv['Memory'] != kv.get('scaleli_accounted_bytes'):
            bad.append('GRE Memory %s != scaleli_accounted_bytes %s' % (kv['Memory'], kv.get('scaleli_accounted_bytes')))
        ops = kv.get('operations_num')
        if kv.get('read', '1') in ('1', '1.0') and ops is not None and kv.get('success_read') != ops:
            bad.append('success_read %s != operations_num %s' % (kv.get('success_read'), ops))
    l = bench['learnability']
    root = 'binary' if not l['root_model'] else ('joint' if learn.get('root_joint') else
                                                'flow' if l['root_flow'] else 'raw') + ('_vp' if l['root_vp'] else '')
    if kv.get('scaleli_root') != root:
        bad.append('scaleli_root %s != bench root %s' % (kv.get('scaleli_root'), root))
    return bad, root, kv


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        bench = json.load(f)
    failed = 0
    for path in sys.argv[2:]:
        res = check(bench, path)
        if isinstance(res, list):
            bad, root, kv = res, '?', {}
        else:
            bad, root, kv = res
        failed += bool(bad)
        print('%s %s: root %s, accounted %s B%s' % ('FAIL' if bad else 'OK  ', path, root,
                                                   kv.get('scaleli_accounted_bytes', '?'),
                                                   ', checksum equal' if 'checksum' in kv and not bad else ''))
        for b in bad:
            print('    ' + b)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
