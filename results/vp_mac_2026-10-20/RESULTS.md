# Virtual points (V) on the Mac, 10M-key samples, 2026-10-20

One repeat per cell (earlier runs on this setup repeated within a median 1.1%), OPS=10M, GRE e807edc + gre-transform
45d08f5, AC power, High Power mode, other apps on the efficiency cores. V = CSV's Algorithm 1 on regions of 256 keys,
alpha 0.1 (about 6-10% virtual keys, about 1.5-2.3 s per 10M keys at bulk load), in front of the unchanged index.

| dataset | lipp | vp_lipp | alex | vp_alex | pw512_alex | pw512_vp_alex | btree | vp_btree |
|---|---|---|---|---|---|---|---|---|
| covid | 47.38 | 47.11 | 16.10 | 16.58 | 13.09 | 12.99 | 6.12 | 6.09 |
| genome | 13.10 | 13.45 | 13.86 | 13.75 | 9.37 | 9.40 | 6.27 | 6.08 |
| libio | 51.04 | 49.81 | 15.99 | 15.55 | 12.76 | 12.19 | 6.28 | 6.14 |
| osm | 13.99 | 14.39 | 8.68 | 8.86 | 6.79 | 7.36 | 6.23 | 6.19 |
| books | - | - | 16.23 | 16.62 | 12.20 | 12.30 | 6.30 | 6.17 |
| fb | 17.01 | 17.05 | 5.93 | 6.15 | 7.59 | 7.55 | 6.20 | 6.12 |
| history | 48.60 | 48.02 | 16.26 | 16.34 | 13.11 | 13.26 | 6.29 | 6.11 |
| planet | 16.60 | 15.13 | 11.72 | 12.01 | 8.11 | 9.35 | 6.20 | 6.12 |
| stack | 49.76 | 49.18 | 16.53 | 16.68 | 13.17 | 13.15 | 6.31 | 6.00 |
| wise | 43.80 | 43.68 | 16.97 | 15.94 | 13.22 | 12.63 | 6.21 | 6.08 |

(M lookups/s; raw LIPP does not run on books on macOS.)

- V alone changes uniform-lookup throughput by -9% to +4% (LIPP: planet -9%, genome and osm +3%; ALEX +-6%); the
  B+tree control loses 1-5% (more keys, nothing to gain). Mostly within noise: no clear win.
- T then V helps ALEX over T alone where the data is globally hard: planet +15%, osm +8%; but pw512_vp_alex stays below
  raw ALEX there (9.35 vs 11.72, 7.36 vs 8.68).
- Consistent with CSV's own report that random-key lookup time is about unchanged; CSV's gains (up to 34%) are
  measured on promoted keys only, and come from Algorithm 2 rebuilding subtrees inside the index, which this layer
  does not do. vp_alex is exact except raw ALEX's own 2^64-1 sentinel key on fb (as raw ALEX).
