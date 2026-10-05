# Supervisor critique: "before vs after NFL and CSV" at 200M

Reviewer stance: hostile. I asked for three things on the full 200M datasets with a proper warm-up: memory footprint, throughput, compression. I was handed a hardness study, three designs and, buried in the scratch folder, a one-seed timing pilot (`pilot_nice5_attempt1.jsonl`, `tables.md`, `summary.json`: fb 12 cells, osm 11 cells, seed 11 only, planet never reached). None of the three designs used that pilot. I did.

Everything below comes from code reading (`include/scaleli/index.hpp`, `smoothing.hpp`, `src/benchmark.cpp`), the existing result files and the pilot JSON. I ran no benchmark.

---

## 1. Is "before vs after NFL and CSV" the right framing for my three metrics?

No. Of the six method x metric pairs, three cannot move by construction, one is plain accounting, and only two involve timing.

| pair | can it move? | why (code) | measured at 200M (pilot, seed 11) |
|---|---|---|---|
| NFL x compression | **null by construction** | `Region::rebuild` encodes every 128-key block (index.hpp:213-235) before any feature or target is computed (index.hpp:238 on) | key B/key 1.9340 fb, 4.7138 osm in all N cells |
| NFL x memory | **null by construction** | the flow is 144 B and is not even counted in metadata | 10.504 / 13.284 B/key, identical |
| CSV x compression | **null by construction** | same encode-first order; virtual points have no physical slot | key B/key identical in C1, C2, F1 |
| CSV x memory | deterministic, +8 B per virtual point (index.hpp:340) | `virtual_features` is read only by the compaction re-placement path (index.hpp:296-307) | +0.799 B/key (fb +7.6%, osm +6.0%); root fences +0.001 |
| NFL x throughput | can move only through the transform cost and fence probes | coordinate search is always a full binary search over 32 `slot_begin` values (index.hpp:126-133), so model quality reaches only the fence term | fence probes 5.14 -> 5.14 (switch) / 5.16 (forced); forced adds 1.0 transform and +7 cache lines per lookup (42.5 -> 49.5) |
| CSV x throughput | can move through fence and root probes | same | region: fence 5.14 -> 4.74 (fb), 4.77 -> 4.26 (osm), i.e. -1.2% / -1.5% of 33.85 / 33.48 comparisons; root: fb 10.80 -> 3.42 with 973 fences; osm root falls back (0 fences) |

What follows from this:
- **"Compression before vs after NFL/CSV" is an empty column.** My compression question is about the codecs, and neither method touches them. The compression contrast I actually want is {packed, raw} x {binary root, learned root}, plus a non-learned compressed reference (section 5). The pilot already has half of it: R0raw (uncompressed blocks) costs +6.19 B/key on fb (16.695 vs 10.504, +59%) and +3.41 on osm. On both datasets it was faster than packed (1.216x, 1.226x, one seed each).
- **"Memory before vs after NFL" is an empty column**, and so is the NFL half of the "combined" cell.
- **The "before" baseline is wrong for every root cell.** B0 uses a binary-search root. C2, N3, F1, J1 and J1s all use `--root model`. Comparing them with B0 credits CSV and NFL with the learned root: fb root probes go 15.66 -> 10.80 with no CSV and no NFL (L0). The CSV-attributable root saving is L0 -> C2 (10.80 -> 3.42), not B0 -> C2. Throughput shows the same confound: fb L0 alone is 1.171x B0 (one seed), so C2's 1.252x leaves 1.069x attributable to CSV, and N3's 1.196x leaves 1.021x attributable to NFL. Each design also counts "before" differently: design 1 uses 33.8 comparisons (binary root), design 2 uses 29.00 (learned raw root), and the pilot table uses 28.81 (no coordinate term). That is three numbers for the same fb index. Pick one definition (root + coord + fence + key_at) and one baseline (the same root type as the treatment).
- **The 2x2 needs four levels, not four cells.** The pilot found two more things that break the clean 2x2:
  - **"NFL at root" (N3) did not use NFL at the root.** fb `root_flow=False` (flow estimate 26.86 probes vs raw 10.80), and the osm root fell back to binary. N3 = L0 plus flow in 2,408 regions.
  - **J1 "joint bend, root alpha 4" did not use the bend.** fb `root_flow=False`, 973 fences, 4 flow regions, so J1 has the same layout as C2. Its 1.317x is not a joint-design result.

---

## 2. Do the pilot numbers survive the noise floor? What I refuse.

### 2a. The noise is worse than the brief says, and it has two states

Look at runs whose index layout is identical:
- **osm:** B0, L0 (root fell back), C2 (root fell back, 0 fences), J1s (82 flow regions), N2/N3 (967 flow regions, 0.020 transforms/lookup), N3 seed 29. Their throughputs are 0.750, 0.886, 0.698, 0.743, 0.892, 0.775 and 0.734 Mops: a 1.277x range, sd(log) = 9.5% (df 6).
- **Pooled with the small fb A/A groups** (C2/J1/J1s; L0/N3; B0/N2): sd is about 7.5% per run, not 6.5%.
- **The throughput is a mixture of two host states.** The osm runs with chunk CV of 0.02-0.03 (L0, N1, N2, consecutive at 15:11-15:15) all sit at 0.87-0.89 Mops. Every other osm run has chunk CV of 0.08-0.14 and sits at 0.67-0.78. Within single runs the per-chunk series steps between levels:
  - fb B0 runs at 0.82-0.91 for chunks 1-4, then 0.66-0.74;
  - fb sorted_vector runs at 0.80, 0.82, then 1.15-1.25, drops to 0.73 at chunk 11, then recovers.

  That is a step of about 1.4x inside one process, larger than any predicted method effect.
- **The machine is shared.** Load average ran from 5 to 178 during the pilot (fb F1 started at 19 and ended at 178), and `uptime` shows 10 users. The brief's "idle host" premise is false.

Consequences:
- **Design 3's repeat counts (27/7/2) are too low.** They scale by (9.5/6.4)^2 = 2.2, to about 60 / 15 / 4 repeats per cell for 5 / 10 / 20% effects. And independent repeats are the wrong tool against a two-state mixture.
- **Run-level ratios are invalid.** The fix is (i) to classify chunks by state, or (ii) to interleave cells inside one process, as in design 3's optional upgrade, so both arms see the same state. Option (ii) stops being optional.
- **The warm-up indicator is confounded.** "First chunk / median of rest" mixes cold cache with state switching. fb sorted_vector scores 0.69, but its chunk 11 drops just as far mid-run. So "a 200,000-lookup warm-up already hides the cold start" is unverified at 200M. Even the pilot's 2M-lookup warm-up gives first/median values of 0.89-1.18 across fb cells, which cannot be told apart from the state flips.

### 2b. Claims I refuse outright

| claim (source) | number | why refused |
|---|---|---|
| CSV region virtual points speed up fb (pilot C1) | 1.185x | work changes by -0.40 of 33.85 comparisons (-1.2%) and 42.5 -> 42.3 lines/op. An 18.5% speedup from 1.2% less work is impossible. This is the clearest proof that single-seed ratios are noise. |
| NFL forced into every region speeds up osm (pilot N1) | 1.172x | it does more work: +1 transform/lookup, +4 lines/op, fence probes +0.01. It ran in the fast host state. |
| CSV root fences slow osm (pilot C2) | 0.931x | the osm root fell back: 0 fences, the same layout as B0. This is an A/A difference. |
| "both" beats CSV-root on fb (pilot F1 vs C2) | 1.436 vs 1.252 | they differ by 0.39 comparisons and 2,408 flow regions; F1 ran while load climbed to 178. |
| NFL at root (N3 1.196x), joint design (J1 1.317x) | - | mislabelled: neither used a flow at the root (`root_flow=False`). |
| CSV root fences on fb (existing sweep: 1.253x and 1.117x; pilot C2: 1.252x) | - | the existing sweep also has the identical structure (973 fences, 3.42 probes) at alpha 0.04 at 1.005x. The same structure has given 1.005x, 1.117x and 1.25x. Direction plausible, magnitude unknown. |
| planet root fusion "synergy" (root_analysis.txt) | 1.175x, root 15.66 -> 3.32 | one seed; uses the inert monotone flow (a global bend, not NFL); 2,973 s build. Interesting as a probe count, not as a throughput claim. |
| Hardness F1: "NFL makes RMSE/n 3.5-5.7%" | - | a property of the stand-in trainer's N(0,1) target (train_flow.py:49), not of NFL. |
| Hardness F2/F3: NFL cuts PLA-32 32-81% | - | measured after sorting by z; a key-ordered index can never see it. The untrained sawtooth already gives 40-75% of it (F3). |
| Hardness F7: CSV raises PLA-32 / CD | - | the metric counts virtual points as keys. The CD jump has a mechanism in the code: boundary-minimum virtual points are placed at exactly lo+1e-3*w or hi-1e-3*w (smoothing.hpp:77,87), within 0.1% of a gap from a real key. How many points land on boundaries is not measured. In any case this metric is not one of mine. |

### 2c. Claims I accept

- **All deterministic numbers:** key B/key, value B/key, metadata, probe and line counters, and build times to about 10%. `result_checksum` agrees across all cells within a seed (fb 16164573662621542249, osm 108634...), so correctness holds at 200M.
- **Compressed key size:** 4.14x fb, 1.70x osm, 5.8x planet (existing).
- **CSV memory cost, with a correction.** Every region saturates its budget: 19,970,703 points / 48,829 regions = 409.0 = floor(0.1 x 4,096). `virtual_features` grows by `push_back` with no reserve (smoothing.hpp:92), so under 2x growth its capacity is 512. `memory()` counts `size()` and leaves `virtual_features` out of `reserved_slack_bytes` (index.hpp:338-343). Hidden cost: (512 x 48,829 - 19,970,703) x 8 B = 40.2 MB = 0.20 B/key. CSV's real memory cost is therefore about 1.0 B/key, not 0.80 (+25%). Design 2 predicted 0.1-0.2; this is the arithmetic. Confirm it with RSS.
- **Binary search beats every learned cell.** I accept this provisionally because four independent measurements agree:
  - 2M: 1.67x;
  - memory_audit at 200M: 1.14-1.16 vs 0.73-0.76 Mops;
  - pilot: 1.38x fb, 1.49x osm;
  - the best learned cell, C2 on fb, makes 21.6 comparisons and touches 21.2 lines/op, against 27.66 and 25.5 for sorted_vector, and is still 9% slower.

  Counting comparisons or cache lines does not predict throughput here. That points to the instruction-bound result in the brief, and neither NFL nor CSV attacks it.

---

## 3. Faithful or strawman?

**NFL is a strawman by construction.** The NFL authors would raise four objections:
1. **It is not their index.** AFLI stores and searches records in z-order and batches lookups (169.5 -> 8.4 ns/key from batch 1 to 256). Here z is only a region-model feature over key-ordered records, at batch 1.
2. **It is not their flow.** The flows come from a 1-D stand-in trainer: 2,000 steps on a 16,384-key sample of a 2M sample, which is 0.008% of the 200M keys.
3. **At region scale the faithful flow is a near-identity.** A tooth is 1/s of the normalised key range. At s = 512 the mean tooth holds about 390,625 keys, about 95 regions of 4,096 keys; at s = 64 about 763 regions. Fewer than 512/48,829 (about 1%) of regions straddle a tooth boundary. Inside a region, the "faithful non-monotone" flow is a smooth monotone bend, which is the monotone flow already shown to be inert.

   The pilot confirms this. NFL's own switch accepts the flow in 2,408 fb regions (4.9%) and 967 osm regions (2.0%), more than the tooth boundaries, with zero fence-probe change. On average the flow makes NFL's own criterion worse in key order: tail conflict degree 31.32 -> 38.89 on fb and 25.79 -> 27.91 on osm.
4. **The one place the sawtooth could act is blocked.** That place is the root (48,829 fences). The root slot table rejects non-monotone features (index.hpp:406), and the raw-feature root estimate is 26.86-27.06 probes against 10.80-24.91 without the flow.

Verdict: present this as "a flow feature in a key-ordered compressed map has no lever", a negative result explained by mechanism. Do not label the column "after NFL". Design 1 is right not to build z-ordered SCALE-LI. But its +30% bytes forecast is a first/last-key estimate, and for fb it rests on an untested tanh-saturation assumption. The z-order FOR-width audit is a few minutes of stdlib Python on the 2M samples, not 3-4 h. Do it; it is the only way NFL can appear in my compression column at all (by destroying it).

**CSV is a restricted version: Algorithm 1 only, used as target relabelling.** The CSV authors would object as follows:
1. **Virtual points never become physical gaps here.** CSV's gain in LIPP/SALI comes from fewer conflicts and shallower trees. Here virtual points only renumber `slot_begin`, and the coordinate step is a fixed 5.03-probe binary search over 32 blocks whatever the model's quality. Model quality reaches only the fence term (4-5 probes of about 33.5), so the region mechanism is architecturally capped at about 14% of comparisons even with a perfect model. It achieved 1.2-1.5%.
2. **Alpha = 0.1 is the binding constraint.** The budget saturates in 100% of regions, so the greedy never reaches its own stop rule (smoothing.hpp:91). A single alpha is not an evaluation; an alpha sweep is (memory vs fence probes).
3. **Algorithm 2 (hierarchical merge) is absent.** It has little to act on in a balanced two-level map; design 1's argument holds.
4. **The root use (virtual fences) is our invention**, not theirs. It is the only place CSV pays off: fb 10.80 -> 3.42 probes for +0.001 B/key and +3.6 s. On osm it does nothing (fallback) and on planet it costs 2,068 s.

Verdict: a fair test of "CSV Algorithm 1 as a model-target smoother". Label it so. The build cost is consistent with the paper: 2,825 CPU-s on fb (184 s wall on 16 threads, 29x the 6.4 s build) against the paper's 889-2,902 s.

**Design-level errors:**
- **Design 2 claims "NFL+CSV = CSV by construction, free A/A control".** As run (F1, no `--fusion auto`), the default `flow_bypass=1` gives the flow to 2,408 fb regions (choices 0/0/46,421/2,408). Fence probes match (4.74) but the layout does not. The A/A claim holds only under `--fusion auto`, which has never been run at 200M with the faithful flow.
- **Design 3 pools across ten datasets.** That is meaningless when the effect is structurally zero on some of them. osm C2 has 0 fences, so its true effect is exactly 0, and pooling it with fb's 973-fence root averages a null with a real effect. Pool only over datasets whose layout actually differs, and estimate sigma from the measured A/A (9.5%), not 6.4%.
- **Design 1:** its 17-19 comparisons for z-ordered NFL ignore that key-ordered fences no longer exist in z-order. Its "0.57 B/key static CSV" figure needs a code change that was not made, so it cannot be reported as measured.

---

## 4. The cheapest experiments that would change my mind

Ordered by cost. All are serial on build-fs with existing flags, except E4.

- **E0. Deterministic table for all ten datasets (about 1 h, no repeats needed).**
  - Cells B0, L0, C1, C2 (root alpha 0.04), R0raw, R0sv, with `--ops 1000000 --instrument 1`.
  - Builds are about 7 + 7 + 190 + 10 + 6 + 6 s, plus about 31 s fixed per run.
  - Gives memory, compression and work counters on 10/10 datasets. Today I have 2/10, plus 3 from the older sweep.
  - Changes my mind if: CSV's fence saving exceeds 3 comparisons on any dataset. Today the best is 0.51.
- **E1. Host-state A/A (about 15 min).**
  - Run B0 on fb 10 times back to back, logging per-chunk throughput, the 1-minute load average and core type. `taskpolicy`/QoS already sets interactive; log `powermetrics` if allowed.
  - Changes my mind if: per-chunk states disappear when the load is under about 10, so a quiet window or another machine gives sigma under 3%. Otherwise the run-level design is dead and the in-process interleaved binary (design 3's upgrade) becomes mandatory.
- **E2. Can any learned layout beat binary search? (about 20 min)**
  - fb: `--policy raw --root model --root-alpha 0.04` (raw blocks plus CSV root, 21.6 comparisons) against R0sv (27.66) and R0raw.
  - 7 interleaved repeats each (21 runs x about 70 s), state-filtered.
  - Changes my mind if: raw+C2 beats sorted_vector by more than 10%. Then the learned index has a throughput story and the price is compression, which is a real trade-off to show the committee. If it is still slower, no NFL/CSV variant matters for throughput in this host, and the talk becomes "compression vs speed", not "before/after NFL/CSV".
- **E3. Compression trade-off (about 35 min).**
  - B0 vs R0raw, 7 repeats, on fb and planet.
  - The pilot says compression saves 37% of bytes on fb for about 18% throughput. Confirm or kill it. This is the answer to my compression question.
- **E4. NFL z-order compression audit (minutes, stdlib Python, 2M samples).**
  - Sort each 2M sample by faithful z, chunk into 128-key blocks, compute FOR bit width and delta width.
  - Changes my mind if: z-ordered blocks pack within 20% of key-ordered ones. That would contradict design 1's estimate (FOR width = the dataset's bulk key range, +30% total).
- **E5. CSV memory honesty (one build, about 4 min).**
  - Steady-state RSS of C1 vs B0 on fb.
  - Predicted gap: +0.80 accounted, about +1.0 real.

---

## 5. What is missing that I would expect at this point

1. **A single published learned index on the same data. There is none.**
   - ALEX and Dynamic PGM adapters exist (`src/benchmark.cpp:297-298`, CMake `SCALELI_EXTERNAL`), but `external/` was never fetched. `integrations/STATUS.md` says outright that the procedure is "not a record of successful execution".
   - RadixSpline is vendored in `third_party/radix_spline` and used only by a walkthrough.
   - For a read-only 200M point-lookup comparison, static PGM and RadixSpline are the minimum; ALEX if updates are claimed. Fetching them is a download from upstream and needs Louis's go-ahead.
   - Without them, "learned index 0.73x of binary search" is a statement about SCALE-LI, not about learned indexes.
2. **A non-learned compressed baseline for the compression column.** A textbook Elias-Fano bound, 2 + ceil(log2(U/n)) bits per key from each file's first/last key:
   - planet: 1.00 B/key vs SCALE-LI's 1.384 (+38%);
   - fb bulk without the 21 outliers: 1.38 vs 1.934 (+40%; with outliers EF is 4.88);
   - osm: 4.75 vs 4.714 (at the bound).

   So "4-6x compression" should be shown against EF / LA-vector / a plain FOR-packed sorted array, which needs no model. Right now no reference says whether 1.93 B/key is good.
3. **Coverage.** Timing exists for 2 of 10 datasets at one seed. Planet, the one dataset with an interesting root result, is missing from the pilot. `run_main.log` stops at "16:19:50 start fb B0 11"; the main run never finished.
4. **Intervals.** Every reported throughput ratio has n = 1 and `cv: null`.
5. **Range scans.** Key order plus packed blocks is the host's whole reason to exist, and NFL's faithful version breaks scans. Not one scan number exists at 200M.
6. **An alpha sweep for CSV**, and a region-size sweep at 200M (at 2M the saving rose from 16% to 39% going from 4k to 32k keys).
7. **Cost per lookup in instructions, not comparisons.** The established IPC result says the index is instruction-bound. Comparisons and lines/op both mis-rank sorted_vector (section 2c). Report ns/lookup next to instructions/lookup on one calibration run, or stop presenting comparison counts as a proxy for speed.
8. **The hardness study** is careful and reproduces `hardness.json` exactly, but it answers none of my three questions. Keep two lines from it:
   - fb's RMSE/ME come entirely from 21 outlier keys (F5);
   - an untrained sawtooth gives 40-75% of NFL's PLA-32 gain (F3).

   Drop the rest from the meeting.

---

## What I want at the next meeting (one slide each)

1. **Memory and compression, 10 datasets, deterministic (E0).** Stacked B/key with key / value / metadata, with columns for B0, raw, CSV, sorted array and the EF bound. Note in the caption that NFL is identical to B0 by construction.
2. **Work per lookup**, root + coord + fence + key_at, for B0, L0, C1, C2, N-forced and R0sv. CSV is measured against L0, not B0.
3. **Throughput only where the counters say something can move:** C2 vs L0 on datasets with fences, raw vs packed, and the binary-search reference. Use in-process interleaving or state-filtered chunks, with a live A/A band. State "NFL: no lever (mechanism)" in one line instead of a bar.
4. **One published baseline (PGM or RadixSpline)**, or an explicit admission that there is none.
