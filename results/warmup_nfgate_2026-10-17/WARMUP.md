# Warm-up on every dataset (Mac preview, 2026-10-17)

**These are Mac previews, not headline numbers.** Every run was on battery in Low Power Mode
(`traces/POWER.txt`: `POWER=battery-lowpower`, 87% -> 63% over the two batches). Each run was a `trace_<index>` run
(a counter in every get()) inside `scripts/mac_lane.sh`, with one run per cell and 30M lookups. Compare shapes and
orders of magnitude, not absolute latencies.

## What ran

- Timed batch (`traces/main`, `traces/lipp`). `gre_run.sh --repeats 1 --ops 30000000 --warmup 0 --pin -1` with
  `GRE_TRACE=geo:64:65536 GRE_TRACE_CPU=1`, inside the lane.
  - main: 10 datasets x {nullindex, splice, pgm, btree, alex, scaleli_b, scaleli_n}, 70/70 ok.
  - lipp: 8 datasets (osm and books were not in the LIPP batch). 7/8 ok. covid.sorted failed inside LIPP's own
    bulk_load: `RT_ASSERT isfinite(node->model.a)`, `lipp.h:490`. That assert exits 0, so `runs.tsv` shows exit_code 0;
    gre_run.sh's own check (no throughput line) counts it as failed (`r1_lipp end rc=1` in `POWER.txt`).
- Analysis, on the efficiency cores (`taskpolicy -c background`):
  - `warmup_detect.py analyze traces/main traces/lipp --json warmup/warmup.json --plot warmup/plots`, with the
    online detector. Table in `warmup/warmup.md`, 77 per-run figures in `warmup/plots/`. Took 7 min 48 s.
  - `analysis/warmup_eval.py summary` builds the cross-dataset tables and figures: `warmup/warmup_eval.{json,md}`.
  - `analysis/warmup_eval.py overview --jobs 4` builds `warmup/overview/overview_<ds>.png`.

The numbers below come from `warmup/warmup_eval.json` (`rows`, `per_index`) unless another file is named.

## Figures

| file | what |
|---|---|
| `warmup/warmup_heatmap.png` | (1) consensus cut and online cut, dataset x index, log colour. S = the cut sits at a P0 <-> P1 cluster switch; n = PELT finds no steady level |
| `warmup/overview/overview_<ds>.png` | (2) per dataset: every index's phases (`plot_overview`) |
| `warmup/warmup_tau_vs_pages.png` | (3) relaxation tau against index pages P = index_rss_bytes / 16 KiB, with the lines tau = P/k |
| `warmup/warmup_cold_excess.png` | (4) cold excess in ms, fit and empirical |
| `warmup/warmup_coverage.png` | (5) per index: each dataset's cut, its cold-transient end and its online stop, against the 20M protocol warm-up |
| `warmup/plots/phases_<ds>_<index>.png` | per run: phases, cores, derivative, detector strip |

## Findings

**1. The index's own cold transient is short, except for LIPP.** "Index transient" is the end of the relaxation fit's
cold transient: the fit is significant, A > 0, and the fitted excess falls below 1% of the steady mean. Its maximum
over the datasets, per index:

| index | runs with a cold transient | median tau (lookups) | max transient end | cold excess, ms (median; range) |
|---|---|---|---|---|
| LIPP | 7/7 | 6.3e5 | 10.3M (wise) | 4892; 210 (stack) to 8153 (planet) |
| SPLICE | 7/10 | 8.9e4 | 2.15M (genome) | 20 to 107, 0 on fb/osm/stack |
| PGM | 3/10 | 1.0e5 | 1.62M (history) | 0, except books 27, history 40, wise 1 |
| ALEX | 10/10 | 1.9e5 | 1.37M (history) | 86 to 239 |
| B+tree | 9/10 | 5.6e3 | 825K (wise) | 18 to 191 |
| SCALE-LI B | 10/10 | 3.4e4 | 729K (stack) | 101 to 211 |
| SCALE-LI N | 10/10 | 4.7e4 | 1.05M (planet) | 45 to 214 |
| harness (nullindex) | 0/10 | - | 0 | 0 |

- SCALE-LI B and N have the same cold transient (tau 1e4 to 1.6e5, 45-214 ms). The flow adds no warm-up of its own.
- On fb, SPLICE again has no cold transient (consensus 21.9K), as on 2026-10-12. It is not flat, though: it has the
  dip described next. On 7 other datasets it has a small transient (20-107 ms).
- Not every early phase is a monotone decay. The relaxation fit comes out rising (A < 0) on 7 index runs: PGM on fb,
  genome, libio and stack, SPLICE on osm and stack, and B+tree on history. PGM on osm and SPLICE on fb have the same
  shape without a significant fit (`overview/overview_{fb,osm,genome,stack}.png`). The early chunks get faster, then
  latency steps up by 15-45% at about 2e4-3e5 lookups. The monotone fit counts this as no cold transient (0 ms, with
  a negative empirical excess). It is over by about 5e5 lookups (libio/pgm's later dip at 1-2M is a machine burst),
  so the warm-ups below cover it. It is not first touch, which only adds time at the start; its cause is open.

**2. The first-touch model (tau ~ P/k) holds in order of magnitude, but not as a proportion.**
`warmup_tau_vs_pages.png`, `per_index[*].touches_per_lookup_k`.
- k = P/tau is the number of distinct new pages a lookup touches. It comes out at 1.6 (ALEX), 1.8 (LIPP), 2.0 (PGM),
  3.0 (SPLICE), 3.2 (SCALE-LI N) and 3.9 (SCALE-LI B). These are plausible page counts per lookup.
- Only LIPP's page count varies enough across datasets to test proportionality: 9-31 GB, 0.55M-1.9M pages. Its fitted
  slope of log tau against log P is 0.49 (n = 7), not 1. So tau grows with P, but less than linearly.
  - The three biggest LIPP indexes (fb, genome, planet, 27-31 GB) also run 3.1-5.0 us per lookup in steady state,
    against 60-183 ns on the others. About 1.8 GB of swap was in use after the batch (a reading not saved with the
    traces). Their transient mixes first touch with compressor and swap traffic.
- Every other index has about the same P on every dataset (200M keys each; P varies less than 2x), so no slope is fitted.
- B+tree's k is 42. Its tau (median 5.6K lookups; 2.2K-25K on 7 of its 9 cold runs, 4.5e5-5e5 on libio and stack)
  is mostly too fast to be page first-touch: that is cache and TLB warm-up of the upper levels.

**3. On this machine, most late cuts are machine events, not warm-up.**
- The harness alone (`nullindex`, 3.4 ns per lookup) gets a consensus cut of 4-17M in 9 of 10 datasets, and 16.8M in
  five of them.
- PELT finds no level lasting 25% of the run in 53/77 runs (the n marks).
- 56/77 runs move between the two performance clusters at least once and stay there (P0 <-> P1, at least 8 chunks).
  In 10 runs the consensus cut sits at such a switch (the S marks). For example, osm/pgm is cut at 16.8M, while its
  relaxation fit finds no transient and the core strip shows the switch at about 17M
  (`plots/phases_osm_trace_pgm.png`).
- The two clusters are equally fast: the steady latency ratio P1/P0 has a median of 0.994 (IQR 0.975-1.006, 50 runs
  with at least 10% on each). A switch costs a re-warm of the 16 MB L2, not a level change.
- Battery plus Low Power Mode also gives short fast and slow bursts after about 5M lookups in most runs
  (`overview/overview_fb.png`).
- Consensus cuts above about 15M are at the detectors' limit for a 30M run. MSER searches the first half, and the
  epoch reference is the second half.

**4. Online detector.** On the open Mac it stopped after a median of 4.5M lookups (p90 17.8M). It was never sure
within 30M in 8/77 runs: LIPP on history, libio and wise, ALEX on history, libio and stack, splice/stack and
btree/planet. It needed more than 20M in 2 more runs. Its cut is a median 0.38 decades from the cold-transient end
(n = 49) and 0.49 decades from the consensus (n = 59). That is weaker than on the fb traces of 2026-10-12, because
there is one run per cell and the drift is larger on battery, and because its parameters were partly tuned on those
fb traces (`online/README.md`), so they were never a held-out test.

**5. The 20M protocol warm-up covers every index transient seen here.**
- The longest is LIPP's, 10.3M (wise). Every other index is warm by 2.15M.
- Every consensus cut is at or below 16.8M. But in a 30M run the detectors cannot place a cut much later than that,
  so this half of the check is weak.

| index | shortest safe warm-up (max index transient) | suggested, with a 2x margin | max consensus (includes machine shifts) |
|---|---|---|---|
| LIPP | 10.3M | 20M | 10.1M |
| SPLICE | 2.15M | 5M | 12.4M |
| PGM | 1.62M | 5M | 16.8M (9.5M without the cluster-switch cuts) |
| ALEX | 1.37M | 5M | 15.8M (8.4M) |
| SCALE-LI N | 1.05M | 2M | 7.0M (3.1M) |
| B+tree | 825K | 2M | 3.5M |
| SCALE-LI B | 729K | 2M | 6.7M (1.7M) |

- Keep 20M as the protocol warm-up: it is needed for LIPP and costs little elsewhere.
- What this Mac needs on top is a quieter machine: AC power, no Low Power Mode, and a pinned cluster. A longer
  warm-up does not help.
- A per-index warm-up of 2-5M would be safe for everything except LIPP, but only on a machine without the drift above.

## Caveats

- There is one run per cell, so there are no across-run intervals: `across_runs` is empty and the `between-run share`
  column in `warmup/warmup.md` shows `-`.
- All runs were on battery and in Low Power Mode. The timed agent's 45-minute power gate never saw AC.
- Timed runs never overlapped each other: one at a time, `busy` = 0 in both `runs.tsv`. But other work shared the
  machine, on the efficiency cores at background QoS as planned. nf_gate (4 threads, builds of up to 200M keys) ran
  from 21:00:30 to 21:33:17, alongside 30 of the 70 main runs (fb scaleli_b's last ~20 s through genome scaleli_n).
  The end of the online validation (file times up to 20:57) ran alongside fb's first runs. `busy` matches only GRE and
  SPLICE's own benchmarks, so it shows neither.
- A rough, unscripted check puts the step-up of the early dip (finding 1) 4-40 ms into the lookup loop, both for PGM
  and SPLICE (2e4-1.3e5 lookups) and for 7 of the 10 harness steps from 2.6 to about 3.4 ns (1.4M-14M lookups; on
  genome and history it steps far earlier, on fb the check found none). That fits a machine step in time, such as a
  frequency change on battery, better than index warm-up, but it has not been tested.
- GRE_TRACE_FAULTS was not set, so first touch cannot be confirmed from page-fault counts. For that, run LIPP and ALEX
  again with `GRE_TRACE_FAULTS=1`.
