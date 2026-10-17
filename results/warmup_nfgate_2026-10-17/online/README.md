# Online warm-up detector: validation (Mac, efficiency cores, 2026-10-17)

`scripts/warmup_detect.py` detector `online` (class `OnlineWarmup`): it reads the chunk trace in order, never looks
ahead, and stops as soon as it is sure. For each chunk it looks at the window [t/9, t] of the chunks seen so far:

- **derivative**: a weighted line of ln(latency) against log10(lookups). The slope is the relative change per decade.
  Its SE comes from the two-component noise model, inflated for residual chi^2 and lag-1 autocorrelation. The test is
  an equivalence test, |slope| + 1.64 SE <= ln(10) min(eps_eff, 0.2) per decade, which equals |d ln y / d ln t| <= eps_eff.
- **variance**: the two halves of the window, each fitted by its own line, must have the same residual variance
  standardized by the noise model (|ln F| within 2.58 sigma, effective degrees of freedom).
- the noise model (per-lookup part + drift, hence eps_eff) is learned on the chunks seen so far: it is refitted
  whenever their count grows by 1/8, and the drift is also read inside the window.

The run is warm after 3 consecutive passing chunks. **stop** is the call at which that happens (the lookups the
detector had to see). The cut is the window start, walked back over chunks already seen while they stay within
max(eps_eff, 3 sigma) of the window mean. Cost: about 0.7 ms per chunk of 65,536 lookups in Python on an efficiency
core, worst case. The parameters (R = 3, M = 3, Z = 1.64, SMAX = 0.2) were tuned on a separate synthetic draw (seed 7,
20 reps) and on these fb traces. The table below uses seed 1, 40 reps.

## Synthetic suite (`synth/`, `warmup_detect.py synth --reps 40 --jobs 4`)

Per method: the median over the 14 cells of each cell's median |log10 error| of the cut.

| method | all scenarios | resolvable (no step, late slowdown) | worst resolvable cell |
|---|---|---|---|
| consensus (mser, pelt, relax, epoch) | 0.06 | 0.06 | 0.17 |
| online | 0.16 | 0.06 | 0.22 |
| mser / pelt / relax / epoch | 0.05 / 0.26 / 0.09 / 0.08 | 0.10 / 0.34 / 0.01 / 0.06 | 0.53 / 0.80 / 0.16 / 0.19 |

- On the resolvable scenarios, online is as accurate as the consensus and better than MSER and PELT. It saw a median
  0.2-2% of the 120M run before stopping (exp slow: 13-15%), 5-10x the true warm-up end for slow transients. For the
  fastest ones, the minimum window (about 0.2-2M lookups) sets that ratio.
- **It cannot see step or late slowdown.** A long flat stretch is followed by a shift, so it stops on the flat
  stretch (|log10 error| 3.7-4.8, 1-3.7% bias). No detector without look-ahead can see these. That is why online
  stays out of the consensus.

## Real traces (fb, `results/warmup_mac_2026-10-12/`; `fb_w0*/`, `open_vs_lane.*`)

Pooled (median across runs) cut. Online vs consensus:

| folder | lipp | alex | splice | pgm | nullindex |
|---|---|---|---|---|---|
| fb_w0 (open) | 5.83M vs 5.56M | 459K vs 385K | 65.5K vs 383K | 0 vs 0 | 0 vs 0 |
| fb_w0_cpu_open | 9.76M vs 10.6M | - | 262K vs 250K | 0 vs 0 | 0 vs 0 |
| fb_w0_cpu_lane | 4.98M vs 10.1M | - | 131K vs 371K | 0 vs 0 | 0 vs 52.2M |

The online stop for LIPP is 40-88M lookups. The cut is found at 5-10M, but on the Mac's jitter it takes about 9x
longer to be sure, which is beyond the 20M protocol warm-up. Splice, pgm and the harness stop at 1-30M. Per run, the
open machine's two-state jitter can block the walk-back, so some flat indexes get a conservative cut of 1-3M. That
does not bias the steady mean; it only shortens the steady phase.
