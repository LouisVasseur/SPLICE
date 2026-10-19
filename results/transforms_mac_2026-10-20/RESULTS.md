# Transforms on the Mac, 10M-key samples, 2026-10-20

Median throughput (M lookups/s) over 2 repeats; GRE e807edc + gre-transform 5d876300b798b781 nfl_dir=/Users/louisvasseur/gre-transform/nfl10M flow_dir=/Users/louisvasseur/gre-transform/flows; OPS=10M, one process at a time, AC power, High Power mode, other apps on the efficiency cores. KL = KL(raw keys || uniform on [min,max]), 1000 bins, nats.

| dataset | KL raw | alex | pw512_alex | pw4096_alex | nf_alex | nfl_alex | lipp | nfl_lipp | btree | nfl_btree |
|---|---|---|---|---|---|---|---|---|---|---|
| covid | 0.041 | 16.66 | 13.21 | 13.28 | 3.60 | 4.43 | 47.58 | 4.09 | 6.22 | 4.26 |
| genome | 0.140 | 13.88 | 9.34 | 11.08 | 3.37 | 4.35 | 13.14 | 4.11 | 6.12 | 4.33 |
| libio | 0.041 | 16.04 | 12.89 | 12.80 | 3.43 | 4.51 | 48.19 | 3.97 | 6.17 | 4.20 |
| osm | 2.338 | 8.73 | 6.74 | 6.37 | 2.93 | 4.34 | 13.91 | 3.63 | 6.32 | 4.36 |
| books | 0.383 | 17.07 | 12.70 | 12.59 | 3.63 | 4.49 | - | 3.78 | 6.17 | 4.28 |
| fb | 6.908 | 6.30 | 7.68 | 6.46 | 2.85 | 4.37 | 17.28 | 4.06 | 6.22 | 4.36 |
| history | 0.003 | 15.97 | 13.21 | 13.21 | 3.69 | 4.46 | 48.80 | 4.07 | 6.35 | 4.38 |
| planet | 0.874 | 11.43 | 8.13 | 8.39 | 3.08 | 4.45 | 16.90 | 4.16 | 6.32 | 4.23 |
| stack | 0.008 | 17.32 | 13.26 | 13.40 | 3.59 | 4.50 | 50.83 | 4.04 | 6.33 | 4.40 |
| wise | 0.043 | 17.04 | 13.38 | 13.26 | 3.51 | 4.44 | 44.80 | 4.07 | 6.17 | 4.37 |

Repeat spread (max-min)/median: median 1.1%, max 16.8% over 89 cells.

## KL gate, leave-one-dataset-out

Per base: candidate rules "raw" and "transform with variant V iff KL > t"; V and t chosen on the 9 training datasets by least total regret; regret = 1 - thr(choice)/thr(best variant), on the held-out dataset.

- **alex**: oracle fb=pw512_alex. LODO mean regret 1.8%, worst 18.0%. Final rule on all datasets: pw512_alex iff KL > 4.623.
  - LODO choices: covid:alex, genome:alex, libio:alex, osm:alex, books:alex, fb:alex (best pw512_alex, -18%), history:alex, planet:alex, stack:alex, wise:alex
- **lipp**: oracle raw everywhere. LODO mean regret 0.0%, worst 0.0%. Final rule on all datasets: always raw.
  - LODO choices: covid:lipp, genome:lipp, libio:lipp, osm:lipp, fb:lipp, history:lipp, planet:lipp, stack:lipp, wise:lipp
- **btree**: oracle raw everywhere. LODO mean regret 0.0%, worst 0.0%. Final rule on all datasets: always raw.
  - LODO choices: covid:btree, genome:btree, libio:btree, osm:btree, books:btree, fb:btree, history:btree, planet:btree, stack:btree, wise:btree

## Closing notes (flow part parked here)

What holds (10M samples, one Mac):
- All three transforms are exact in the wrapper (every key returns its own payload; tests/test_index.cpp,
  tests/test_nfl.cpp) and do what they are built for: pw4096 brings KL from linear to ~0 on all ten datasets, pw512 to
  <= 0.10, our monotone NF to <= 0.40 (osm), and NFL's own flow (NflFlow, trained with NFL's trainer and settings)
  reproduces NFL's inference to 6e-16.
- A transform pays only where (1) the keys are far from linear (KL), (2) that shape is what makes the index deep, and
  (3) the transform is cheap per lookup. Only ALEX on fb meets all three: 8.6 -> 1-2 levels with any transform, and
  pw512 is 22-23% faster than raw ALEX (6.30 -> 7.68 M/s, both repeats). LIPP and B+tree are slower with every transform.
- Against NFL (PVLDB 2022): its depth result for ALEX on FB (11 -> 3 levels) reproduces; its 2.2x speedup does not with
  its flow evaluated one key at a time (-31%; ~230 ns per lookup here, ~170 ns in NFL's Table 2 at batch 1). NFL's
  numbers come from batched inference (~8 ns per key at batch 256), which GRE's one-lookup-at-a-time driver cannot do.

Caveats:
- One timed win (ALEX on fb), so the KL gate's threshold rests on one positive: leave-one-dataset-out misses fb.
- 10M stride samples on an Apple M3 Max (arm64, 70 W adapter); not yet at 200M keys or on the Atom (x86).
- Not tested: AFLI (NFL's own index), batched inference, the remaining GRE indexes with nfl_.

Parked (next, in order): ALEX raw vs pw512 / nfl on fb, osm, planet at 200M keys on the Atom; AFLI; batched lookups.
NFL's weights for these runs: ~/gre-transform/nfl10M/<dataset>_10M.nfl, trained from ~/nfl/work (NFL fe71172, outside
this repository: GPL-3).
