#!/usr/bin/env python3
"""Check the supplied evidence without compiling or running any benchmark.
Uses only the Python standard library. Timings are from archived measurements.
"""
from __future__ import annotations
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def main() -> None:
    evidence = ROOT / 'evidence'
    rows = [json.loads(line) for line in (evidence / 'archived_suite_results.jsonl').read_text().splitlines() if line.strip()]
    require(len(rows) == 108, 'Expected 108 archived runs')
    require(all(row['verified'] for row in rows), 'A run was not verified')
    groups: dict[tuple, list] = {}
    for row in rows:
        key = (row['dataset'], row['profile'], row['seed'], row['repeat'])
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        require(len(group) == 9, f'{key}: expected nine variants')
        require(len({r['trace_fingerprint'] for r in group}) == 1, f'{key}: trace mismatch')
        require(len({r['result_checksum'] for r in group}) == 1, f'{key}: result checksum mismatch')
    print(f'PASS: {len(rows)} verified runs; {len(groups)} groups with matched traces and checksums')
    summaries = list(csv.DictReader((evidence / 'archived_suite_summary.csv').open()))
    print('\nCard 05: dense_sparse / read_only (medians of two runs)')
    print(f'{"Variant":<19} {"bytes/record":>13} {"Mops/s":>10} {"hit P99 ns":>12}')
    for summary in summaries:
        if summary['dataset'] != 'dense_sparse' or summary['profile'] != 'read_only':
            continue
        group = [r for r in rows if r['dataset'] == 'dense_sparse' and r['profile'] == 'read_only' and r['variant'] == summary['variant']]
        require(len(group) == 2, 'Expected two seed runs')
        speed = statistics.median(r['throughput_ops_s'] for r in group)
        memory = statistics.median(r['memory_before']['accounted_bytes'] / r['initial_rows'] for r in group)
        tail = statistics.median(r['latency_ns']['read_hit']['p99'] for r in group)
        require(math.isclose(speed, float(summary['throughput_ops_s_median']), rel_tol=1e-10), 'Summary throughput mismatch')
        require(math.isclose(memory, float(summary['initial_bytes_per_key']), rel_tol=1e-10), 'Summary memory mismatch')
        require(tail == float(summary['read_hit_p99_ns_median']), 'Summary latency mismatch')
        print(f'{summary["variant"]:<19} {memory:13.3f} {speed/1e6:10.3f} {tail:12.1f}')
    layout = json.loads((evidence / 'archived_layout_summary.json').read_text())
    errors = [r['rank_normalized_mse'] for r in layout['results']]
    require(layout['logical_rank_invariance_verified'], 'Rank invariance not marked verified')
    require(max(errors) - min(errors) < 1e-14, 'Rank geometry changed')
    for row in layout['results']:
        require(math.isclose(row['bytes_per_key'], row['key_bytes'] / row['keys']), 'Layout size calculation mismatch')
    print('\nPASS: Card 06 stored size calculations and recorded logical-rank invariance')
    mac = json.loads((evidence / 'mac_run_transcribed.json').read_text())
    require(sum(v['count'] for v in mac['latency_ns'].values()) == mac['operations'], 'Mac operation counts do not sum')
    require(mac['initial_rows'] + mac['latency_ns']['insert']['count'] - mac['latency_ns']['erase']['count'] == mac['final_rows'], 'Mac cardinality mismatch')
    for field in ['memory_before', 'memory_after', 'memory_after_drain']:
        part = mac[field]
        require(sum(part[k] for k in ['key_bytes','value_bytes','metadata_bytes','delta_bytes','reserved_slack_bytes']) == part['accounted_bytes'], 'Memory components do not sum')
    drained_speed = mac['operations'] * 1e9 / (mac['elapsed_ns'] + mac['maintenance']['drain_ns'])
    print(f'PASS: Card 07 counts, cardinality and memory; final-drain-inclusive throughput = {drained_speed:,.1f} ops/s')
    print('\nNo benchmarks were run. Paper table values remain manually transcribed source evidence, not reproduced results.')

if __name__ == '__main__':
    try:
        main()
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        raise SystemExit(1)
