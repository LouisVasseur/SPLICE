Warm-up detection (eps = 0.01; calls = lookups since the first get(); cut = first call of the steady phase)

| dataset | index | runs | mser | pelt | relax | cov | hw | epoch | online | online stop | consensus | tau (calls) | beta | excess (ms) | steady ns/lookup | CI95 across runs | between-run share | R-hat | PELT kind | robust CoV | lag-1 | two states (fast / slow ns, slow share) | envelope: warm after | envelope: steady ns | steady chunks by core (share, mean ns, slow share) | migrations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fb | trace_alex | 3 | 393K | 65.5K | 256K | 6.82M | 30M | 262K | 1.18M | 27.3M | 385K | 2.24e+05 | 1.00 | 210.11 | 540.42 | 85.01 | 0.83 | 1.034 | no steady state | 22.8% | 0.86 | 409 / 658, 46% | 224K | 434.30 | - | - |
| fb | trace_btree | 3 | 0 | 119M | 774K | 17.4M | 0 | 1.05M | 2.1M | 18.5M | 662K | 1.12e+06 | 1.00 | 418.98 | 717.13 | 315.20 | 0.97 | 1.132 | no steady state | 16.1% | 0.80 | 567 / 886, 37% | 490K | 567.91 | - | - |
| fb | trace_lipp | 3 | 6.09M | 96.4M | 5.59M | 19.7M | 0 | 4.19M | 5.7M | 51.1M | 5.56M | 1e+06 | 0.70 | 8191.07 | 210.58 | 17.17 | 0.00 | 1.005 | warmup | 33.5% | 0.88 | 146 / 234, 67% | 5.34M | 176.37 | - | - |
| fb | trace_nullindex | 3 | 0 | 107M | 4.46M | 786K | 6M | 33.6M | 64 | 2.82M | 0 | 3.16e+09 | 1.00 | 0.00 | 3.59 | 1.15 | 0.98 | 1.202 | flat | 15.4% | 0.69 | 3 / 4, 73% | 0 | 3.20 | - | - |
| fb | trace_pgm | 3 | 0 | 0 | 0 | 16.2M | 0 | 0 | 1.31M | 16.3M | 0 | 2.82e+06 | 1.00 | 0.00 | 318.15 | 81.44 | 0.94 | 1.042 | flat | 18.5% | 0.89 | 269 / 395, 33% | 31.9K | 274.73 | - | - |
| fb | trace_sortedarray | 3 | 0 | 73.8M | 492K | 32.7M | 0 | 32.7K | 524K | 8.13M | 32.7K | 3.16e+09 | 1.00 | 0.00 | 1053.89 | 89.07 | 0.38 | 1.010 | no steady state | 15.7% | 0.70 | 840 / 1260, 38% | 75.8K | 815.95 | - | - |
| fb | trace_splice | 3 | 0 | 0 | 389K | 2.1M | 0 | 65.5K | 4.65M | 41.4M | 383K | 2.51e+06 | 0.35 | 2770.12 | 314.95 | 13.10 | 0.00 | 1.008 | flat | 33.3% | 0.89 | 234 / 371, 56% | 1.25M | 265.13 | - | - |
