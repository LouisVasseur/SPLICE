# Before vs after NFL and CSV on SCALE-LI at 200M keys: the plan

Scratch folder: `NB=/Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/results/aidb_ba`.
Binary: `build-fs/scaleli_bench`, used as is (not rebuilt). Scripts written for this plan: `run_ba.py` (cells, plans,
serial resumable runner, cost estimate), `analyze_ba.py` (tables, sigma, CIs, A/A, warm-up, checksums), `rss_ba.sh` (E5).
`analyze_ba.py` was smoke-tested on the pilot jsonl. No 200M job was run to write this plan.

---

## 1. The question, restated so it can be answered

**What it means here.** SCALE-LI always stores the same records in key order, in the same packed 128-key blocks.
Each block is encoded *before* any feature or target is computed (index.hpp:213-235; models from :238). Neither method
touches the records:

* **NFL** changes the model's **feature**: z = f(key) replaces the normalised key in a region model (or at the root).
* **CSV** changes the model's **targets**: slot ranks with up to 0.1·n virtual points per region (Algorithm 1), and
  virtual fences at the root.

So "after X" can move only four things:

* **Memory:** model metadata only.
* **Compression:** nothing.
* **Throughput:** only through root probes, fence probes and the transform call. Two terms are fixed by construction
  and give a 13.05-comparison floor on every lookup: the coordinate search (5.03) and the in-block search (8.02).
* **Build time.**

**The four cells, plus controls.** All cells share `--policy min_bytes --routing rank`.

* **B (before):** adds `--root model`. This is a learned raw root, which falls back to binary search by itself when
  binary search is cheaper (osm, planet).
* **N (after NFL):** B plus the faithful flow, NFL's own switch, and the root offered the flow at cost 0.
* **C (after CSV):** B plus `--virtual-alpha 0.1 --root-alpha 0.1`, the paper's alpha at both levels.
* **NC (after both):** C plus the flow, as in N.
* **Controls:**
  * Nf: the flow forced into every region.
  * B2: an identical rebuild of B, used as a live A/A control.
  * SV: plain binary search.
  * RAW: uncompressed blocks.
  * RC: raw blocks plus root fences.
  * B0: the old baseline with a binary-search root.

**What it cannot mean.**

* **It cannot mean NFL as published.** AFLI stores records sorted by z, batches lookups (169.5 → 8.4 ns/key at batch
  256) and evaluates no scans. A z-ordered SCALE-LI would break all three codecs (codec.hpp:86,199), the key fences,
  scans and the CSV root (index.hpp:406). It would cost 15-20 h of engineering plus about 5.3 h of runs, for a
  predicted +30% bytes per key (design 1).
* **It cannot mean CSV as published.** There is no Algorithm 2 and no physical gaps (LIPP), so virtual points never
  become slots.
* **It cannot mean "the datasets got easier" in the AIDB sense.** Those metrics see the faithful NFL flow only in
  z-sorted order, and count CSV's virtual points as keys.
* **It cannot mean "learned beats binary search."** Binary search is a reference line, not a cell.

## 2. What the pilot and the hardness recompute already show

All pilot timings are from one seed (11) on fb and osm. planet was never reached. Pilot B0 used a binary-search root,
so here **"B" means pilot L0**, and the pilot ratios are recomputed against it.

**NFL (faithful flow, as the host realises it)**

| quantity | fb | osm | all 10 (hardness) |
|---|---|---|---|
| regions given the flow by NFL's switch | 2,408 / 48,829 (4.9%) | 967 (2.0%) | — |
| fence probes, B → N | 5.14 → 5.14 | 4.77 → 4.77 | — |
| root | flow estimate 26.86 vs raw 10.80 probes → rejected (root_flow=False) | root falls back to binary search (15.66) | — |
| forced (N1 vs B0) | +1.00 transform/lookup, lines 42.5 → 49.5, fence 5.16 | +1.00, lines 43.6 → 47.6, fence 4.78 | — |
| tail conflict in key order, raw → flow | 31.32 → 38.89 (worse) | 25.79 → 27.91 (worse) | — |
| memory / key bytes (B/key) | 10.504 / 1.934, identical | 13.284 / 4.714, identical | identical by construction |
| throughput, n=1 (pilot N3 / L0) | 1.021 | 0.875 (both layouts are B plus 967 regions) | — |
| AIDB PLA-32, z-sorted order | — | — | −32 to −81%; the untrained sawtooth gets 40-75% of the log gain on 9/10 datasets |
| AIDB, monotone (realisable) flow | — | — | PLA-32 moves by ≤ 4 segments (inert) |
| AIDB RMSE/n, faithful flow | — | — | 3.5-5.7% on every dataset (an artefact of the trainer's N(0,1) target); fb RMSE/ME come entirely from 21 outlier keys |

**CSV (Algorithm 1, regions + root)**

| quantity | fb | osm | other |
|---|---|---|---|
| virtual points | 19,970,703 = 409/region = the full budget in 100% of regions | same | — |
| metadata B/key | 0.570 → 1.369 (+0.799); plus ~0.20 hidden vector capacity (512 − 409 per region) ≈ +1.0 real | same | — |
| total B/key | 10.504 → 11.303 (+7.6%) | 13.284 → 14.083 (+6.0%) | — |
| key bytes | identical (1.934) | identical (4.714) | — |
| fence probes | 5.14 → 4.74 | 4.77 → 4.26 | planet 3.99 → 3.42 |
| root fences (root-alpha 0.04) | 973 fences; root 10.80 → 3.42 | 0 fences (falls back) | planet at alpha 4: 195,316 fences; 15.66 → 13.69 probes, 2,068 s |
| comparisons (root + coord + fence + key_at) | 28.99 → 21.20 (−27%) | 33.47 → 32.96 (−1.5%) | — |
| build | 6.5 → 184 s (2,826 CPU-s) | 7.0 → 200 s (3,077 CPU-s) | paper: 889-2,902 s at a = 0.1 |
| throughput, n=1 | C2/L0 = 1.070; C1/B0 = 1.185 is refused (1.2% less work) | C1/B0 0.914 and C2/B0 0.931 are refused (C2 is the same layout as B0) | — |
| AIDB | region RMSE 318 → 265; PLA-32 +9.7% | region RMSE 519 → 404; PLA-32 +7.4% | PLA-32 +3.5% (planet) to +8.8% (genome) on the four hardest; −4 to −35% on books, covid, history, libio |

**Noise floor and checks (the pilot is all we have at 200M)**

| item | value |
|---|---|
| osm, 7 runs of identical layout (B0, L0, C2, J1s, N2, N3, N3 seed 29) | 0.698-0.892 Mops, range 1.277x, sd(log) **9.5%** whole replay, 10.1% as the median of chunks 2-16 (recomputed here) |
| brief / design 3 sigma | 6.5% / 6.4% ("idle host" premise, which is false) |
| within-run state steps | about 1.4x (fb B0 chunks 0.82-0.91, then 0.66-0.74 Mops) |
| host | load average 5-178 during the pilot. hostmon (229 samples, 15:45-18:32): a QEMU VM was in the top-3 CPU users in 98% of samples; median other-process CPU 345%, max 1,626% |
| warm-up (2M workload warm-up + prefault) | first chunk / median of the rest −1 = **−0.3% (SE 2.0%)** over 24 pilot runs: the warm-up already passes |
| correctness | result_checksum is identical across all 12 fb cells and all 11 osm cells (seed 11) |
| binary search | SV/B0 1.382 (fb), 1.489 (osm); SV/L0 1.181 (fb). Every learned cell is slower |
| raw vs packed | raw is +6.19 B/key on fb (16.695 vs 10.504); R0raw/B0 1.216 (fb), 1.226 (osm), n=1 |

## 3. Current failure points, ranked by how much each would hurt in the meeting

1. **Single-seed ratios on a shared host.**
   - Every pilot ratio has n=1. Sigma is 9.5%, not 6.5%, and the host carries a VM.
   - The supervisor has already refused F1 1.436x, C1 1.185x, N1 1.172x and C2 0.931x.
   - Showing any of them ends the discussion. Fix: retire them all. Timing only comes from the protocol in §4, with a
     live A/A cell.
2. **Wrong baseline.**
   - The treatments used `--root model` and were compared with a binary-search root (B0).
   - The learned root alone is 1.171x on fb, which was being booked to CSV or NFL.
   - Fix: B = `--root model` (pilot L0). B0 stays only as the "host default" reference.
3. **Binary search beats every learned cell** (1.18-1.49x, one seed).
   - fb C2 makes fewer comparisons (21.6 vs 27.66) and touches fewer lines (21.2 vs 25.5) than binary search, yet is
     9% slower.
   - If this is not on a slide, the supervisor will put it there. Fix: show SV as a line on every throughput figure,
     and run E2 (raw blocks + root fences vs SV).
4. **Nulls by construction presented as results.**
   - NFL × compression, NFL × memory and CSV × compression cannot move (index.hpp:213-238).
   - Fix: one "null by construction" row with the code reason. The compression panel becomes packed vs raw vs an
     Elias-Fano bound instead.
5. **"NFL" is not NFL, and some cells are mislabelled.**
   - The faithful flow acts as a feature in key order, so within a region it is a near-identity bend: one tooth spans
     about 95-763 regions. The root rejects non-monotone features (index.hpp:406).
   - Pilot N3 "NFL at root" and J1 "joint bend" both have root_flow=False.
   - The "0 of 6,740 regions" figure is from 2M. At 200M the switch accepts 4.9% (fb) and 2.0% (osm) of regions.
   - Fix: label the cell "NFL's feature in a key-ordered index". Put NFL's own convention (z-sorted AIDB metrics,
     untrained-sawtooth control) on a separate panel.
6. **CSV understated and under-explored.**
   - Its memory cost is +0.80 B/key accounted but about 1.0 real.
   - One alpha, which binds in 100% of regions, is not an evaluation.
   - Root fences fall back on osm (0.04) and planet (about 1,950 budget).
   - Fix: E5 RSS check, the alpha sweep (`sweep` preset), and stating the fallbacks.
7. **Coverage.** Only 2 of 10 datasets are timed, planet is missing, and there are no CIs, no range scans at 200M and
   no published learned index (the ALEX/PGM adapters were never fetched). Fix: all ten datasets in §4. State scans
   and published baselines as out of scope.
8. **Three different "before" comparison counts are in circulation** (33.8, 29.00, 28.81). Fix: one definition,
   root + coord + fence + key_at. B0 = 33.84 and B = 28.99 on fb.
9. **Compression range in the brief.** "4-6x" holds only for fb and planet; osm is 1.70x (4.714 B/key). The
   hardness panel also counts virtual points as keys and is outlier-bound on fb. Fix: quote per-dataset numbers and
   report fb with and without its 21 outliers.

## 4. The experiment

**Run settings, common to every cell** (`run_ba.py`):

```
--format sosd --dtype uint64 --load-ratio 1 --miss 0 --query-distribution uniform --profile read_only --ops 5000000 --warmup 4000000 --warmup-mode workload --prefault 0 --chunks 16 --qos 1 --build-threads 16 --verify 0 --latency 0
```

* **Warm-up:** 4M workload-mode lookups, about 5 s, covering each region about 82 times. The pilot's 2M already
  passes (−0.3%). Prefault does not change the answer.
* **Cold reference:** a COLD cell (warm-up 0) on fb, osm and planet, for the figure only.
* **Ops:** 5M per run. More ops barely help: sigma 6.4% → 6.1% from 2M to 10M ops, and only more processes do.
* **Instrumentation:** the instrumented pass rebuilds the index (benchmark.cpp:173), so `--instrument 1` is used only
  on block 1. That block gives every deterministic number, and its pass-1 throughput is still uninstrumented, so it
  counts as a timing repeat. Every other block uses `--instrument 0`, one build per run.
* **Seeds:** the workload seed is 1000 + block, shared by all cells in a block. result_checksum must then agree
  within each (dataset, block), which checks correctness without the slow verify pass.
* **Run order:**
  * Blocks run one after another.
  * Within a block, datasets are shuffled, and cells are shuffled within each dataset (fixed shuffle seed).
  * Strictly serial: the runner waits on `pgrep scaleli_bench|scaleli_hardness`.
  * other-process CPU is logged at the start and end of each run.
* **Statistic** (pre-registered, `analyze_ba.py`):
  * Per run: whole-replay `throughput_ops_s`. The median of chunks 2-16 is a sensitivity check (`STAT=med`).
  * Per dataset: the contrast is the difference of mean log throughput.
  * Sigma is pooled within cells, and the 95% CI uses t.
  * The pooled effect is the mean over datasets whose deterministic signature differs from B's. Identical
    signatures are structural nulls and are reported as extra A/A comparisons.
  * Thermal check: the per-chunk slope for C must match B's (C is timed straight after a 3-7 min all-core build).
* **Deterministic metrics** (one build per cell, block 1): accounted B/key by component; root, coordinate, fence and
  key_at probes; transform calls and cache lines per operation; build_ns and smoothing CPU-s; flow regions, virtual
  points, root fences and root_flow. Plus E5 RSS on fb.

**Overnight version.** `run_ba.py plan overnight`: 199 runs, an estimated **8.2 h**, plus E1/E2/E5 at about
0.6 h.

* Block 1 (instrumented): B, B2, N, Nf, C, NC and SV on all ten datasets; RAW and RC on fb; COLD on fb, osm and
  planet.
* Blocks 2-3: B, B2, N, Nf, C and SV on all ten datasets; RAW and RC on fb.
* Repeats per dataset:
  * 3 each for B, B2, N, Nf, C and SV.
  * 1 for NC.
* NC gets one repeat because NC − C is predicted under 2% and no overnight budget resolves it.

**Full version.** `plan full`: 1,269 runs, an estimated **38.1 h**, plus `plan sweep` at **3.4 h**.

* Block 1 as in the overnight version, plus RAW, RC, B0 and Nm (monotone flow) on all ten datasets.
* Blocks 2-16: B, B2, N, Nf, SV, RAW and RC.
* C in blocks 2-8 and NC in blocks 2-4.
* Side cell C4 (root-alpha 4): 3 repeats on planet and osm (2,068 s of root smoothing per build).
* Repeats per dataset: 16 each for B, B2, N, Nf, SV, RAW and RC; 8 for C; 4 for NC.
* Sweep (deterministic):
  * root-alpha 0.04 and 0.4 on all ten datasets;
  * region alpha 0.05 and 0.2 on fb, osm and planet;
  * root-alpha 4 on fb and osm (planet reuses `results/aidb_fullscale`).

**What each version can resolve.** 95% half-width in log units at sigma = 9.5%. Multiply by about 0.42 if E1 shows
sigma ≈ 4% with the VM stopped.

| contrast | prediction | overnight, per dataset | overnight, pooled over 10 | full, per dataset | full, pooled |
|---|---|---|---|---|---|
| B2 vs B (A/A) | 1.000 | ±15% | ±4.8% | ±6.6% | ±2.1% |
| N vs B | 0.98-1.00 | ±13% | ±4.2% | ±5.7% | ±1.8% |
| Nf vs B | 0.85-0.96 | ±13% (only if ≤ 0.87) | ±4.2% (**yes**) | ±5.7% (most datasets) | ±1.8% (**yes**) |
| C vs B | 1.03-1.10 where root fences are adopted; ~1.00 where the root falls back | ±13% (no) | ±4.2% (borderline) | ±7.4% (fb borderline) | ±2.3% (**yes** if ≥ 3%) |
| NC vs C | 0.98-1.00 | ±21% | ±6.8% | ±11% | ±3.6% (equivalence bound only) |
| interaction (NC−C)−(N−B) | ≈ 0 | ±25% | ±8% | ±13% | ±4.1% (equivalence bound only) |
| SV vs B | 1.15-1.50 | ±13% (**yes** on most) | **yes** | **yes** | **yes** |
| RC vs SV (fb, E2 + blocks) | 0.90-1.10 | ±8% (10 vs 10 runs) | — | ±6.6% | — |
| memory, compression, counters, build time | exact | exact | — | exact | — |

Pooling assumes one shared effect. If the per-dataset effects disagree, quote the pooled value as an average, not as a
law.

## 5. Predicted outcomes, written down before running

Every cell is compared with B on the same dataset. Accounted B/key is from `memory()`.

| method × metric | prediction | basis |
|---|---|---|
| NFL × compression | key B/key identical to B to 4 decimals (fb 1.9340, osm 4.7138, planet 1.384) | blocks are encoded before the model (index.hpp:213-238); 16/16 earlier runs agree |
| NFL × memory | identical; the flow is 144 B and not counted | pilot N1-N3 |
| NFL × work | N: fence ±0.01, transform 0.02-0.05/lookup, root unchanged (rejected or falls back). Nf: +1.00 transform, +4 to +7 lines/op, fence +0.01 | pilot fb/osm |
| NFL × throughput | N 0.98-1.00 (null; equivalence bound); Nf 0.85-0.96, pooled resolvable | transform 28-66 ns of a 1.2-1.7 µs lookup |
| NFL × build | +0.5-1.0 s (7.2-7.8 s vs 6.5), plus offline training | pilot |
| CSV × compression | identical key B/key | as above; C1/C2/F1 identical |
| CSV × memory | +0.799 B/key accounted on every dataset whose region budget saturates (expected: all); root fences +0.001-0.005; RSS +≈1.0 B/key (+0.20 hidden capacity) | pilot; capacity arithmetic; E5 confirms |
| CSV × work | fence −0.4 to −0.6; root 10.80 → 3.42 on fb; root falls back on osm (and probably planet) at alpha 0.1; comparisons fb 28.99 → 21.2 | pilot; aidb_fullscale |
| CSV × throughput | fb 1.03-1.10 (pilot point estimate 1.070); fall-back datasets 0.99-1.01 (region virtual points alone ≈ 1.002); pooled 1.01-1.06 | probe deltas × ns/probe |
| CSV × build | 184-451 s (16 threads), +≤ 25 s root at alpha 0.1; 28-70x B | pilot; hardness CSV walls |
| both × memory, compression | memory = C (+0.000); compression identical | — |
| both × work | NC = C plus 2-5% flow regions; root identical to C (non-monotone flow barred, index.hpp:406) | pilot F1 vs C1/C2 |
| both × throughput | NC/C 0.98-1.00; interaction ≈ 0 | — |
| A/A (B2/B) | CI contains 1.000; if not, the protocol is broken and no timing is reported | — |
| warm-up | pooled first-chunk indicator within ±3%; COLD first chunk −17 to −19% | pilot −0.3%; brief and design 3 |
| references | SV 1.15-1.50 faster than B; RAW 1.1-1.25 faster at +6.2 B/key (fb); RC vs SV unknown (0.90-1.10) | pilot n=1 |

**Elias-Fano reference for the compression figure.** The bound is 2 + ceil(log2(U/n)) bits per key, from each file's
first and last key; there is no select index.

| dataset | bound (B/key) | SCALE-LI packed keys (B/key) |
|---|---|---|
| planet | 1.00 | 1.384 |
| fb bulk (without its 21 outliers) | 1.38 | 1.934 |
| osm | 4.75 | 4.714 |

The other seven bounds were computed the same way here: genome 1.75, history 1.00, libio 0.50, stack 0.375,
wise 2.25 and covid 3.875 B/key. books (4.75) and fb (4.875) are set by their top outliers.

## 6. Figures, in meeting order, each with its one-sentence claim

1. **Method × metric matrix** (code-reason cells). *Only throughput can move, and only through root and fence
   probes; compression is untouched by both methods and memory by NFL, by construction.*
2. **Warm-up** (chunk traces, COLD vs warm, for fb/osm/planet; pooled indicator). *A 4M-lookup workload warm-up
   removes the 17-19% cold first chunk; every later number is warm.*
3. **Memory** (stacked B/key per dataset × cell: keys, values, metadata, virtual points, hidden capacity; RSS dots on
   fb). *NFL costs 0 B/key; CSV costs +0.80 B/key accounted, about +1.0 real (≈ +8%), all of it in metadata that only
   the update path reads.*
4. **Compression** (key B/key per dataset: packed vs raw vs Elias-Fano bound, plus codec survey from
   `memory_audit.md`). *Packing saves 37% (fb) to 40% (planet) of total bytes, only 20% on osm, and is identical under NFL
   and CSV; it sits 38-40% above a simple Elias-Fano bound on planet and fb.*
5. **Work per lookup** (stacked root/coord/fence/key_at for B, N, Nf, C, NC; SV line; 13.05 floor). *CSV removes
   root probes where root fences are adopted (fb 29.0 → 21.2); NFL removes nothing; 13.05 comparisons per lookup are
   out of reach of both.*
6. **Throughput forest plot** (per dataset and pooled, 95% CI, with an A/A band from B2; SV and RAW as references).
   *Claim to be filled from the data and judged against §5; predicted: only forced NFL (slower) and pooled CSV (a few
   percent faster) resolve, and binary search is faster than every cell.*
7. **Build time** (log axis). *CSV's gain costs 184-451 s per build on 16 threads, 28-70x the base; planet's root at
   alpha 4 costs 2,068 s for 13% fewer root probes.*
8. **AIDB hardness**, a panel separate from the 2x2. *NFL's hardness gain exists only in z-sorted order, and half of
   it comes from an untrained sawtooth; CSV lowers its own region RMSE by 17-56% but raises PLA-32 on the four
   hardest datasets, so these metrics cannot see either method as the host uses it.*
9. Backup slides:
   - the CSV alpha sweep;
   - NFL tooth geometry (one tooth spans about 95-763 regions);
   - the fb outliers (RMSE 57.7M → 140,694 without 21 keys);
   - the faithful-NFL cost estimate (+30% B/key, 39-385x scan read amplification) as the reason it was not built.

## 7. Decisions that are Louis's, each with a recommended default

| # | decision | recommended default |
|---|---|---|
| D1 | Present NFL as "NFL's feature in a key-ordered index", or build z-ordered NFL (15-20 h + 5.3 h) | restricted, labelled; give the cost estimate as the reason |
| D2 | Baseline = learned root (B), not binary root (B0) | B; show B0 as the host-default reference |
| D3 | CSV alpha for the 2x2 | the paper's 0.1 at both levels; sweep 0.04/0.4/4 at the root and 0.05/0.2 in regions as determinism-only backup; time root-alpha 4 only in the full version (planet, osm) |
| D4 | Quiet the host (stop the QEMU VM, ChatGPT and other apps) for the runs | yes; without it sigma ≈ 9.5% and only pooled effects resolve |
| D5 | Overnight (8.2 h) vs full (38.1 h + 3.4 h sweep) | overnight before the meeting; full only if E1 shows sigma ≤ 5% |
| D6 | Compile a scratch copy of the bench that interleaves cells inside one process | not before the meeting; reconsider if E1 shows within-run 1.4x steps persist with the VM stopped |
| D7 | Fetch and run ALEX/PGM (a download that needs approval) | no for this meeting; state it as missing |
| D8 | Report CSV memory for a static index (drop virtual_features, needs a code change) | report +0.80 accounted, +0.20 hidden, and say static would be +0.001-0.005, not measured |
| D9 | Show pilot n=1 ratios | no; retire them all |
| D10 | O(1) slot → block table (3-4 h, −5.03 probes on every arm) | not before the meeting; never booked as a CSV gain |
| D11 | Finish the hardness CSV scope on stack, wise and trimmed fb (~10-15 min) | optional; it does not change any conclusion |

## 8. Commands to run, in order

Run as Louis, from `$NB`. Everything is serial: never start a step while `pgrep -fl scaleli_` prints anything.

```zsh
NB=/Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/results/aidb_ba
cd $NB; pgrep -fl 'scaleli_bench|scaleli_hardness'      # must print nothing
# D4: stop the QEMU VM and other heavy apps by hand, then log host load for the whole session
nohup python3 hostmon.py >/dev/null 2>&1 & echo $! > hostmon.pid

# E1  A/A on fb, 10 runs, ~12 min. GO if sigma <= 5%; 5-10%: run overnight, report pooled only; >10%: stop, see D6
python3 run_ba.py plan aa > plan_aa.json && python3 run_ba.py run plan_aa.json aa.jsonl
python3 analyze_ba.py aa.jsonl | sed -n '/## Noise/,/## Throughput/p'

# E2  raw blocks + root fences vs binary search on fb, 14 runs, ~13 min
python3 run_ba.py plan e2 > plan_e2.json && python3 run_ba.py run plan_e2.json e2.jsonl
python3 analyze_ba.py e2.jsonl > report_e2.md

# Overnight 2x2, 199 runs, ~8.2 h (resumable: rerun the same line after any interruption)
python3 run_ba.py cost overnight
python3 run_ba.py plan overnight > plan_overnight.json
nohup python3 run_ba.py run plan_overnight.json ba.jsonl > run_ba.log 2>&1 &
tail -f run_ba.log                                       # failures go to ba.jsonl.failed

# E5  steady-state RSS, fb B vs C, ~10 min (after the overnight run, never during it)
zsh rss_ba.sh

# Analysis: primary + sensitivity
python3 analyze_ba.py ba.jsonl > report_ba.md
STAT=med python3 analyze_ba.py ba.jsonl > report_ba_med.md
kill $(cat hostmon.pid)

# Second night (optional, D3): deterministic alpha sweep, ~3.4 h
python3 run_ba.py plan sweep > plan_sweep.json && nohup python3 run_ba.py run plan_sweep.json sweep.jsonl > run_sweep.log 2>&1 &

# Full version (only on D5 = full): ~38 h, same runner, separate output
python3 run_ba.py plan full > plan_full.json && nohup python3 run_ba.py run plan_full.json full.jsonl > run_full.log 2>&1 &
```

**Acceptance checks before any number goes on a slide:**

* Checksums agree in every (dataset, block).
* The B2/B CI contains 1.
* The pooled warm-up indicator is within ±3%.
* C's chunk slope matches B's.
* Deterministic signatures are identical across repeats of a cell.
* Every throughput ratio shown carries its n and its CI.
