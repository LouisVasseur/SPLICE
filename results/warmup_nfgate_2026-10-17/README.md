# Warm-up on every dataset and an NF-transform gate (Mac, 2026-10-17)

These are Mac previews, not headline numbers. **Power: every timed run was on battery in Low Power Mode**
(`traces/POWER.txt`: `POWER=battery-lowpower`). A 45-minute power gate never saw AC, so the batches ran with
`MAC_ALLOW_BATTERY=1`. Branch `exp-warmup-nfgate`; nothing pushed. Timed runs never overlapped each other, but nf_gate
(efficiency cores, background QoS) ran beside 30 of the 70 main runs (WARMUP.md, caveats).

## What ran

| step | where | output |
|---|---|---|
| Timed GRE traces: `trace_<index>`, 30M lookups, `--warmup 0`, `GRE_TRACE=geo:64:65536 GRE_TRACE_CPU=1`, one run per cell, inside `scripts/mac_lane.sh` | P-cores, one run at a time | `traces/main` (10 datasets x nullindex, splice, pgm, btree, alex, scaleli_b, scaleli_n: 70/70 ok), `traces/lipp` (8 datasets, 7/8 ok: covid.sorted fails LIPP's own `RT_ASSERT isfinite(node->model.a)`) |
| `nf_gate`: exact B vs N counters plus sample gate scores, 10 datasets x (full 200M + 8 windows of 12.5M) | E-cores, 4 threads, 33 min | `nfgate/cases.jsonl`, `nfgate/<ds>.log` |
| Online warm-up detector, validated on synthetic and fb traces (commit f2020dd) | E-cores | `online/README.md` |
| `warmup_detect.py analyze traces/main traces/lipp --json --plot` | E-cores, 7 min 48 s | `warmup/warmup.{md,json}`, `warmup/plots/` |
| `analysis/warmup_eval.py summary` and `overview` | E-cores | `warmup/warmup_eval.{md,json}`, `warmup/*.png`, `warmup/overview/` |
| `analysis/nfgate_eval.py` (run from `nfgate_eval/`) | E-cores | `nfgate_eval/nfgate_eval.{md,json}`, `nfgate_eval/*.png` |

Details: [WARMUP.md](WARMUP.md) and [NFGATE.md](NFGATE.md).

## Key findings

1. **The 20M protocol warm-up covers every index's cold transient.**
   - The longest is LIPP's, 10.3M lookups (wise). Its tau is 3.2e5-1e6, and it costs 0.2-8.2 s of excess time.
   - Every other index is warm by 2.15M: SPLICE 2.15M, PGM 1.62M, ALEX 1.37M, SCALE-LI N 1.05M, B+tree 825K,
     SCALE-LI B 729K.
   - Shortest safe warm-up with a 2x margin: 20M for LIPP, 5M for SPLICE, PGM and ALEX, 2M for B+tree and SCALE-LI.
     Source: `warmup/warmup_eval.json` `per_index`.
2. **First touch explains the order of magnitude of tau, not its proportionality.**
   - k = pages / tau = 1.6-3.9 distinct new pages per lookup, for ALEX, LIPP, PGM, SPLICE and SCALE-LI.
   - LIPP's tau grows as P^0.49 over 9-31 GB (n = 7), not linearly. Its three biggest indexes are also under memory
     pressure (3-5 us steady).
   - B+tree's transient (k ~ 42) is cache and TLB warm-up, not first touch.
3. **On battery and in Low Power Mode, most late cuts are machine events, not warm-up.**
   - The harness alone gets cut at 4-17M (9/10 datasets).
   - PELT finds no level lasting 25% of the run in 53/77 runs.
   - 56/77 runs switch between the P0 and P1 clusters. 10 cuts sit at a switch, and the two clusters are equally fast
     (P1/P0 latency 0.994 median).
   - The online detector stops after a median of 4.5M lookups. It is never sure within 30M in 8/77 runs.
4. **The flow never helps on these datasets, at any margin of 0 or more.**
   - In 90/90 cases, benefit_net lies between -1.95e-3 and 0 (exact counts). The margin is 1% of D, justified in
     NFGATE.md.
   - That is largely by construction: N's build keeps B's root in all 90 cases and bypasses the flow in at least 94%
     of regions, so the counts differ only by the transform charge.
   - Timed, N vs B is within noise on all 10 datasets: mean +0.5%, sd 2.4%. fb's +5.0% is not confirmed by the lower
     envelope (+0.4%) and has identical probe counts.
5. **Chosen gate: the root proxy at m = 1024 keys, threshold 0** (`nfgate_eval/nfgate_eval.json` `chosen`).
   - It costs 331 us on an efficiency core, plus 134 us to sample.
   - It is right in 90/90 cases (0 false positives), right at full scale on 10/10 datasets, and agrees with the timed
     GRE verdict on 10/10. Under LODO it is right in 89/90.
   - With no positive case this counts only false positives, which "always skip" also avoids. The 90 cases are 10
     datasets of 9 related key sets each, and the gate computes the build's own root criterion on a sample, so the
     count agreement is close to agreement by construction (NFGATE.md, ground truth and limits).
   - It skips the flow on every dataset, because the flow root costs 6.8-17.7 probes more than raw or binary.
   - The flatness gates fail at their natural thresholds:
     - NFL's keep rule: 9 false positives, mostly on osm.
     - KL: 27-37 false positives.
     - Trend share: 85 false positives.
   - Limit: no dataset has a flow that wins, so the gate's sensitivity is unmeasured. A positive control needs a flow
     that beats the raw root.

## Figures

| file | shows |
|---|---|
| `warmup/warmup_heatmap.png` | consensus and online warm-up cut, dataset x index (log), with machine-event marks |
| `warmup/overview/overview_<ds>.png` | every index's phases, one figure per dataset |
| `warmup/warmup_tau_vs_pages.png` | relaxation tau against index pages, with tau = P/k lines |
| `warmup/warmup_cold_excess.png` | cold excess (ms) per dataset x index |
| `warmup/warmup_coverage.png` | per index: cuts, transient ends and online stops against 20M, plus the shortest safe warm-up |
| `nfgate_eval/nfgate_scatter.png` | gate score against measured benefit_net, by dataset, with thresholds |
| `nfgate_eval/nfgate_vs_m.png` | accuracy and C-index against sample size, accuracy against cost |
| `nfgate_eval/nfgate_decisions.png`, `nfgate_decisions_theta0.png` | per-dataset decisions of each gate, with ground truth and GRE |
| `online/...` | online detector validation (see `online/README.md`) |

## Reproduce (from this folder, on the efficiency cores)

```
taskpolicy -c background /usr/bin/python3 ../../scripts/warmup_detect.py analyze traces/main traces/lipp \
    --json warmup/warmup.json --plot warmup/plots > warmup/warmup.md
taskpolicy -c background /usr/bin/python3 analysis/warmup_eval.py summary
taskpolicy -c background /usr/bin/python3 analysis/warmup_eval.py overview --jobs 4 --out warmup/overview
cd nfgate_eval && taskpolicy -c background /usr/bin/python3 ../analysis/nfgate_eval.py --cases ../nfgate/cases.jsonl \
    --warm ../warmup/warmup.json --logs ../traces/main/logs --out nfgate_eval
```
