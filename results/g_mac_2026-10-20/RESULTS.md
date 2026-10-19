# G (outlier exception list) on the Mac, 10M-key samples, 2026-10-20

G = SPLICE-H's exception list in front of an unchanged GRE index (include/gret/exceptions.h): at most 64 keys peeled
at each end while the span exceeds 2x the 64/64-trimmed span, one range check per lookup. Peeled: fb 1 key (2^64-1),
osm 128 (64 + 64), every other dataset 0. OPS=10M, 1 repeat (runs.tsv); LIPP re-timed with the base lookup inlined
into the wrapper, 2 repeats on four datasets (runs_lipp_flatten.tsv). AC power, High Power mode, apps on E-cores.

| dataset | alex | g_alex | g_pw512_alex | lipp | g_lipp | pgm | g_pgm | btree | g_btree |
|---|---|---|---|---|---|---|---|---|---|
| covid | 15.37 | 15.67 | 12.30 | 45.90 | 38.60 | 8.47 | 8.43 | 6.06 | 5.97 |
| genome | 12.91 | 13.15 | 8.82 | 11.87 | 11.72 | 8.67 | 8.70 | 6.36 | 6.28 |
| libio | 15.77 | 15.43 | 12.23 | 49.59 | 38.60 | 8.62 | 8.61 | 6.32 | 6.44 |
| osm | 8.57 | 8.32 | 7.14 | 13.41 | 12.48 | 7.87 | 7.78 | 6.40 | 6.08 |
| books | 16.63 | 15.32 | 12.18 | - | - | 9.73 | 9.65 | 6.41 | 6.36 |
| **fb** | **6.29** | **8.85** | **9.10** | 15.64 | 14.86 | (fails) | 8.46 | 6.14 | 6.23 |
| history | 15.84 | 15.70 | 12.36 | 47.93 | 39.81 | 9.03 | 9.07 | 6.15 | 5.94 |
| planet | 11.61 | 10.91 | 7.36 | 15.95 | 14.53 | 8.29 | 8.39 | 6.18 | 5.93 |
| stack | 16.88 | 16.33 | 12.54 | 48.83 | 40.76 | 9.37 | 9.34 | 6.29 | 6.27 |
| wise | 16.94 | 16.40 | 12.58 | 43.61 | 37.94 | 9.31 | 9.04 | 6.20 | 5.79 |

- ALEX on fb: G alone +41% (6.29 -> 8.85; 8.57 -> ~1.7 levels), close to removing the key by hand (+48%); G then T
  (pw512) +45%. T alone gave +22%: most of T's fb gain was the outlier.
- Sentinel check: fb with 2^64-1 replaced by 2^64-2 keeps raw ALEX at 8.57 levels, so ALEX's fb depth is the
  outlier's range effect, not its 2^64-1 sentinel. G also makes ALEX exact on that key and lets PGM load fb (PGM
  reserves 2^64-1).
- Elsewhere G peels nothing and costs nothing measurable on ALEX, PGM and B+tree (0.92-1.02x, one repeat).
- LIPP: G costs ~10% on its ~20 ns lookups even with nothing peeled (0.86-1.01x with the base inlined, was
  0.78-0.87x through a direct call): the wrapper's range check and fields, which LIPP's short lookups notice. The same
  per-layer cost is inside every T and V number for LIPP. A gate should drop G where nothing is peeled.
