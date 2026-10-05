# Real memory on the Mac (fb, 200M keys, 2026-10-05)

`python3 rss_ba.py` on the Apple M3 Max (`build-fs`, v1 command, 10M lookups, no counters). Read with
`python3 validate.py --rss rss_mac_2026-10-05` (check MET-7).

| cell | real B/key | accounted B/key | gap |
|---|---|---|---|
| B | 10.734 | 10.504 | +2.2% |
| C | 11.942 | 11.304 | +5.6% |
| C - B | +1.208 | +0.800 | PROTOCOL H4 predicted +1.00 +- 0.02 |

Real = steady-state RSS minus the process's first sample, the 16 B/key input copy, the 32 B/lookup trace and the
8 B warm-up keys. B's accounting is within 3%; C's understates real memory (unaccounted `virtual_features` capacity
and allocator retention), so CSV's memory cost should be reported from RSS: about 1.2 B/key on fb, not 0.8.
