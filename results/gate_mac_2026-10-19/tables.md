Labels: median Mops/s over repeats (spread = (max - min) / median); oracle in bold. Source: results/gate_mac_2026-10-19/labels/runs.tsv; features: results/gate_mac_2026-10-19/features_10M.jsonl; raw LIPP veto: more than 0.1% of the keys above 2^53 (--veto-rule count).

Caveat: labels from 10M-key samples on one Mac (Apple M3 Max, on AC power), ten datasets, GRE's uniform read-only lookups; leave-one-dataset-out is the only validation.

### lipp

| dataset | oracle | raw | pw512 | pw4096 | nf | margin over 2nd |
|---|---|---|---|---|---|---|
| covid_10M | pw512 | vetoed | **28.64 (3%)** | 28.19 (2%) | 5.79 (3%) | 1.6% (noise tie) |
| genome_10M | raw | **12.54 (6%)** | 7.82 (9%) | 7.77 (2%) | 2.28 (0%) | 60.3% |
| libio_10M | raw | **49.49 (2%)** | 28.29 (9%) | 26.74 (2%) | 5.79 (4%) | 74.9% |
| osm_10M | pw512 | vetoed | **8.20 (4%)** | 8.00 (7%) | 2.78 (4%) | 2.5% (noise tie) |
| books_10M | pw512 | vetoed | **20.82 (4%)** | 20.60 (3%) | 5.18 (5%) | 1.0% (noise tie) |
| fb_10M | raw | **16.84 (0%)** | 6.92 (1%) | 6.55 (7%) | 2.63 (6%) | 143.3% |
| history_10M | raw | **48.81 (6%)** | 29.12 (3%) | 28.26 (0%) | 5.81 (2%) | 67.6% |
| planet_10M | raw | **15.82 (5%)** | 11.65 (5%) | 11.46 (5%) | 3.62 (3%) | 35.8% |
| stack_10M | raw | **48.79 (7%)** | 29.74 (2%) | 28.78 (1%) | 5.84 (3%) | 64.1% |
| wise_10M | raw | **43.51 (1%)** | 29.59 (1%) | 27.95 (2%) | 5.83 (0%) | 47.1% |

LODO (const, rule, tree2) and fixed baselines, regret in % of the oracle's median throughput:

| model | mean regret | worst regret | mean share of oracle | picks the oracle (on noise ties) |
|---|---|---|---|---|
| const (chosen) | 0.00% | 0.00% | 100.00% | 10/10 (3) |
| rule | 0.00% | 0.00% | 100.00% | 10/10 (3) |
| tree2 | 0.00% | 0.00% | 100.00% | 10/10 (3) |
| always-raw | 0.00% | 0.00% | 100.00% | 10/10 (3) |
| always-pw512 | 27.71% | 58.91% | 72.29% | 3/10 (3) |
| always-pw4096 | 29.67% | 61.12% | 70.33% | 0/10 (0) |
| always-nf | 81.53% | 88.30% | 18.47% | 0/10 (0) |
| nflsw | 27.71% | 58.91% | 72.29% | 3/10 (3) |
| nflsw-strict | 11.73% | 58.91% | 88.27% | 7/10 (3) |

The model type is chosen by this same LODO (no nested LODO): its LODO figure is optimistic once the types differ. Every model type makes the same choices as always-raw on every dataset: on these labels LODO cannot tell const, rule and tree2 apart, and the gate adds nothing over the hand-written rule (raw, pw512 where raw is vetoed).

Per-dataset choices (LODO for const/rule/tree2):

| dataset | oracle | const | rule | tree2 | always-raw | always-pw512 | always-pw4096 | always-nf | nflsw | nflsw-strict |
|---|---|---|---|---|---|---|---|---|---|---|
| covid_10M | pw512 | pw512 (0.0%) | pw512 (0.0%) | pw512 (0.0%) | pw512 (0.0%) | pw512 (0.0%) | pw4096 (1.6%) | nf (79.8%) | pw512 (0.0%) | pw512 (0.0%) |
| genome_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (37.6%) | pw4096 (38.1%) | nf (81.8%) | pw512 (37.6%) | raw (0.0%) |
| libio_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (42.8%) | pw4096 (46.0%) | nf (88.3%) | pw512 (42.8%) | raw (0.0%) |
| osm_10M | pw512 | pw512 (0.0%) | pw512 (0.0%) | pw512 (0.0%) | pw512 (0.0%) | pw512 (0.0%) | pw4096 (2.5%) | nf (66.1%) | pw512 (0.0%) | pw512 (0.0%) |
| books_10M | pw512 | pw512 (0.0%) | pw512 (0.0%) | pw512 (0.0%) | pw512 (0.0%) | pw512 (0.0%) | pw4096 (1.0%) | nf (75.1%) | pw512 (0.0%) | pw512 (0.0%) |
| fb_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (58.9%) | pw4096 (61.1%) | nf (84.4%) | pw512 (58.9%) | pw512 (58.9%) |
| history_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (40.3%) | pw4096 (42.1%) | nf (88.1%) | pw512 (40.3%) | raw (0.0%) |
| planet_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (26.4%) | pw4096 (27.5%) | nf (77.1%) | pw512 (26.4%) | pw512 (26.4%) |
| stack_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (39.0%) | pw4096 (41.0%) | nf (88.0%) | pw512 (39.0%) | raw (0.0%) |
| wise_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (32.0%) | pw4096 (35.8%) | nf (86.6%) | pw512 (32.0%) | pw512 (32.0%) |

Final model (const, fitted on all 10 datasets, in-sample mean regret 0.00%): `raw|pw512`

### btree

| dataset | oracle | raw | pw512 | pw4096 | nf | margin over 2nd |
|---|---|---|---|---|---|---|
| covid_10M | raw | **6.07 (0%)** | 5.43 (4%) | 5.25 (5%) | 3.87 (1%) | 11.7% |
| genome_10M | raw | **6.02 (2%)** | 5.32 (1%) | 5.19 (1%) | 3.87 (2%) | 13.2% |
| libio_10M | raw | **6.04 (1%)** | 5.40 (2%) | 5.27 (3%) | 3.86 (4%) | 11.8% |
| osm_10M | raw | **6.03 (2%)** | 5.14 (0%) | 4.79 (2%) | 3.86 (3%) | 17.4% |
| books_10M | raw | **5.95 (2%)** | 5.40 (1%) | 5.14 (2%) | 3.83 (3%) | 10.2% |
| fb_10M | raw | **5.99 (1%)** | 4.91 (1%) | 4.53 (2%) | 3.90 (3%) | 22.1% |
| history_10M | raw | **5.91 (5%)** | 5.48 (4%) | 5.21 (1%) | 3.90 (2%) | 8.0% |
| planet_10M | raw | **6.10 (4%)** | 5.21 (3%) | 5.07 (1%) | 3.83 (1%) | 17.1% |
| stack_10M | raw | **6.05 (2%)** | 5.26 (3%) | 5.12 (3%) | 3.84 (1%) | 15.0% |
| wise_10M | raw | **6.05 (1%)** | 5.41 (2%) | 5.19 (4%) | 3.77 (2%) | 11.8% |

LODO (const, rule, tree2) and fixed baselines, regret in % of the oracle's median throughput:

| model | mean regret | worst regret | mean share of oracle | picks the oracle (on noise ties) |
|---|---|---|---|---|
| const (chosen) | 0.00% | 0.00% | 100.00% | 10/10 (0) |
| rule | 0.00% | 0.00% | 100.00% | 10/10 (0) |
| tree2 | 0.00% | 0.00% | 100.00% | 10/10 (0) |
| always-raw | 0.00% | 0.00% | 100.00% | 10/10 (0) |
| always-pw512 | 12.04% | 18.08% | 87.96% | 0/10 (0) |
| always-pw4096 | 15.72% | 24.40% | 84.28% | 0/10 (0) |
| always-nf | 35.98% | 37.63% | 64.02% | 0/10 (0) |
| nflsw | 12.04% | 18.08% | 87.96% | 0/10 (0) |
| nflsw-strict | 6.72% | 18.08% | 93.28% | 5/10 (0) |

The model type is chosen by this same LODO (no nested LODO): its LODO figure is optimistic once the types differ. Every model type makes the same choices as always-raw on every dataset: on these labels LODO cannot tell const, rule and tree2 apart, and the gate adds nothing over the hand-written rule (raw, pw512 where raw is vetoed).

Per-dataset choices (LODO for const/rule/tree2):

| dataset | oracle | const | rule | tree2 | always-raw | always-pw512 | always-pw4096 | always-nf | nflsw | nflsw-strict |
|---|---|---|---|---|---|---|---|---|---|---|
| covid_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (10.5%) | pw4096 (13.5%) | nf (36.3%) | pw512 (10.5%) | raw (0.0%) |
| genome_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (11.7%) | pw4096 (13.8%) | nf (35.7%) | pw512 (11.7%) | raw (0.0%) |
| libio_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (10.6%) | pw4096 (12.8%) | nf (36.1%) | pw512 (10.6%) | raw (0.0%) |
| osm_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (14.8%) | pw4096 (20.6%) | nf (36.0%) | pw512 (14.8%) | pw512 (14.8%) |
| books_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (9.3%) | pw4096 (13.7%) | nf (35.6%) | pw512 (9.3%) | pw512 (9.3%) |
| fb_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (18.1%) | pw4096 (24.4%) | nf (34.8%) | pw512 (18.1%) | pw512 (18.1%) |
| history_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (7.4%) | pw4096 (12.0%) | nf (34.0%) | pw512 (7.4%) | raw (0.0%) |
| planet_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (14.6%) | pw4096 (16.9%) | nf (37.2%) | pw512 (14.6%) | pw512 (14.6%) |
| stack_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (13.0%) | pw4096 (15.3%) | nf (36.5%) | pw512 (13.0%) | raw (0.0%) |
| wise_10M | raw | raw (0.0%) | raw (0.0%) | raw (0.0%) | raw (0.0%) | pw512 (10.5%) | pw4096 (14.2%) | nf (37.6%) | pw512 (10.5%) | pw512 (10.5%) |

Final model (const, fitted on all 10 datasets, in-sample mean regret 0.00%): `raw|pw512`

