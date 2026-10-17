Paired warm-up comparison: A = open (results/warmup_mac_2026-10-12/fb_w0_cpu_open), B = lane (results/warmup_mac_2026-10-12/fb_w0_cpu_lane); eps = 0.01

| dataset | index | runs A/B | warm after A | warm after B | online A (cut / stop) | online B (cut / stop) | tau A | tau B | excess A (ms) | excess B (ms) | steady A ns | steady B ns | B/A | robust CoV A | robust CoV B | states A | states B | kind A | kind B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fb | lipp | 3/2 | 10.6M | 10.1M | 9.76M / 87.6M | 4.98M / 44.6M | 5.01e+05 | 4.47e+05 | 7752.6 | 8359.6 | 211.2 +/- 57.3 | 285.6 +/- 57.2 | 1.35 +/- 0.46 | 37.4% | 14.5% | 150/251 (54%) | 272/475 (6%) | warmup | no steady state |
| fb | nullindex | 3/2 | 0 | 52.2M | 0 / 5.24M | 0 / 15.7M | 3.16e+09 | 3.16e+09 | 0.0 | 0.0 | 3.2 +/- 0.7 | 3.3 +/- 2.4 | 1.04 +/- 0.78 | 16.7% | 5.3% | 3/3 (78%) | one | flat | no steady state |
| fb | pgm | 3/2 | 0 | 0 | 0 / 17.8M | 0 / 3.87M | 3.16e+09 | 1.41e+07 | 0.0 | 0.0 | 358.3 +/- 30.4 | 373.8 +/- 46.6 | 1.04 +/- 0.16 | 24.6% | 13.5% | 286/419 (53%) | 299/383 (89%) | flat | no steady state |
| fb | splice | 3/2 | 250K | 371K | 262K / 22.5M | 131K / 3.47M | 1.26e+05 | 1.78e+05 | 74.5 | 59.9 | 350.4 +/- 43.5 | 389.0 +/- 279.8 | 1.11 +/- 0.81 | 19.5% | 10.6% | 280/426 (50%) | one | warmup | no steady state |
