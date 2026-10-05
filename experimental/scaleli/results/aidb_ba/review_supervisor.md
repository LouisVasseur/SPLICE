# Review 3 (the supervisor): do the claims match the evidence?

Reviewed: `PROTOCOL_draft.md` (1,165 lines, read in full). I also checked it against `PLAN.md`, `critique.md`,
`tables.md`, `hardness_ba.md`, `$SP/threeblock/armB_head.md` and `armA_table.md`, and the root code in
`include/scaleli/index.hpp:386-425,456`. I ran nothing.

## Verdict in five lines

1. **Memory: accepted.** Exact accounting on 10 datasets plus one RSS check on fb is the evidence I would ask for.
   Two wording fixes are needed (I-7, I-8).
2. **Throughput: well designed, but it rests on a condition nobody has observed yet,** namely a quiet Mac with
   sigma at or below 5%. That number is measured only after 11-14 h of engineering. Measure it first, with the
   binary that already exists (I-3).
3. **Compression: the evidence is next to the claim, not on it.** The input-space metric D_k is defined so that only
   gap removal can score above zero, so "only G compresses the key axis" is true by definition. That contradicts
   `hardness_ba.md`, where the monotone NFL flow squeezes fb's empty space (I-1).
4. **Gap removal gets a test only at the root, and on 2M-sample predictions.** The root fences cannot see fb's 21
   outlier keys, so the one place gap removal obviously works is never tested before the full version (I-2, I-4,
   I-5).
5. **The scope is too large for one meeting.** Cut the full version, the codec cells, the C++ gap patch (unless a
   cheap 200M probe check earns it), and the C++ k sweep. Move three cheap items into the overnight (section 6).

---

## 1. Claim by claim: does the protocol produce evidence that would convince me?

| claim (section 1) | evidence the protocol produces | convinced? | what is missing |
|---|---|---|---|
| H1, H2: key bytes and accounted bytes unchanged by NFL | D-layer equality | yes, as a **consistency check**. It is a null by construction (blocks are encoded before any model, index.hpp:213-238). | Call it a check, not a hypothesis. On the memory slide, say why it is zero: the index keeps key order. NFL's own z-ordered storage is estimated at about +30% B/key (`design_faithful.md`, not built). |
| H3, H4: CSV +0.80 accounted, +1.00 real | D layer on 10 datasets; RSS on fb | yes | "Saturates on all 10" is based on fb and osm only. In the hardness tool, the same Algorithm 1 left osm, covid, history and libio well under budget (VP 4.8-6.5% of n). Keep the prediction, but say it rests on 2 datasets. |
| H5, H6: G costs 528 B; costs add up | exact | yes (trivial) | none |
| H7, H8: D_k > 0; root-fence RMSE falls with G | `fences_ba.py` on 200M fences | **adjacent** | D_k is an input to the method, not an outcome, and it is defined only for G. The outcome is "a line fits better". That has to be measured the same way for G, the monotone NFL flow and B (I-1). |
| H9: \|dPLA\| <= 2k | proof plus the hardness tool | yes, the proof is correct (a piecewise-affine map with 2k breakpoints preserves vertical error inside each piece). | none |
| H10: NFL does not compress the input axis | the faithful flow is non-monotone, so the root refuses it | **no.** This only shows that the root's monotonicity rule rejects that particular flow. `hardness_ba.md` F4/F5: the **monotone** flow takes planet's RMSE to 0.54x and fb's to 0.008x. The fb part is exactly "squeezing out the empty space below 21 outliers", which is the supervisor's definition of compression. | Put Nm (monotone flow) in the overnight D layer and measure it with the same metric as G (I-1). |
| H11: CSV leaves the input axis alone and smooths targets | region rank SSE | yes, and the framing is honest | none |
| H12-H14: work counts for N, Nf, C | D layer | yes; the definition root + gap + coordinate + fence + key_at is right | none |
| H15, H16: G changes root probes by -1.5..+0.5 and costs 5 comparisons | **2M-sample prototype**, with 200M checked only after the C++ patch | **no, not as a 200M prediction** (I-4) | Run `fences_ba.py` at 200M before freezing predictions and before writing P3. |
| H17, H18: selector declines G; levels are separable | D layer | yes | none |
| H19: build time | median of the T runs | yes | none |
| H21: A/A | E1 plus a live A/A gate | yes, and this is the best part of the draft | Run E1 first (I-3). |
| H22, H25, H26: nulls (N, C vs Cr, G, GCr) | TOST at ±3% pooled | yes **if** sigma <= 5%. At 9.5%, nothing can be claimed, and the draft says so. | Do not time cells whose D-layer signature equals B's on that dataset (I-6). |
| H24: CSV is faster (pooled 1.01-1.06; fb 1.03-1.10) | random-effects pool | **only partly.** The predicted effect sits on fb (root 10.80 -> 3.42 probes). Elsewhere the root falls back or barely moves, so the pool averages one real effect with near-nulls. With n = 3, a per-dataset effect on fb resolves only at about 6-8%. | Spend the repeats where the effect is: B, Cr and C at n = 8 on fb and on whichever datasets the D layer shows adopting root fences (I-6). |
| H27: SV is faster | positive control | yes | Lead with it: plain binary search beats every learned cell by 1.38-1.49x. Every before/after ratio has to be read against that. |
| W, T: warm-up and thermal checks | c1, GHz, slope, COLD | yes | See section 4. |

## 2. Nulls by construction: presented honestly?

Mostly yes. The one-page table marks NFL x memory "+0 (by construction)", and §4 of PLAN plus threat 13 explain
why. Three things still mislead:

- **Figure 1's claim**, "memory and compression move only through G (input axis) and CSV (metadata), by
  construction", is circular for compression. The metric that makes it true (D_k) is defined only for G.
- **Figure 3's claim**, "NFL and G cost 0 B/key", is true for this index. It needs the clause "because the index
  stores records in key order; NFL's z-ordered layout, not built, is estimated at +30% B/key". Without that clause a
  reader takes it as a property of NFL.
- **H1, H2, H5 and H6 are numbered as hypotheses "to prove".** They are equalities the code guarantees. List them
  under "consistency checks (by construction)". They should not share a table with the claims that carry
  information, or the slide count overstates the findings.

CSV x key bytes is handled honestly: it is in H1 as a null, and the compression column redefines compression away
from codecs. Good.

## 3. Is input-space compression given a real test?

### 3.1 At the root: partly

The root is the right place. It is the only global model in SCALE-LI, and the region coordinate step is a fixed
32-way binary search (index.hpp:126-133). So the draft is right to skip region-level G. But:

- **Where G could matter, the root does not use a model.** G should help where gaps concentrate. On the 2M top-16
  gap share that is osm 0.60, books 0.27 and planet 0.12. But osm's learned root already falls back to binary
  (`tables.md` L0: raw 24.91 probes vs binary 15.66). G in force mode would have to cut more than 9 probes plus its
  5 table comparisons just to be adopted, or the table is cleared and G = B. On fb, where the root model *is* used
  (10.80), the top-16 gap share is the lowest of all ten (0.036).
  - So the timed G cell may be a structural null on most datasets, and those are excluded from the pool. The draft
    never shows which datasets are left. The D layer, or better `fences_ba.py`, answers this in minutes, and it
    should do so before the C++ patch is written.
- **The 2M prototype is not a 200M predictor for the root (I-4).** At 2M the root has about 488 fences and about
  2.2-8 model probes (`armA_table.md`: fb-u 2.22, osm-u 8.22). At 200M it has 48,829 fences, with fb at 10.80 and
  osm at 24.91. k = 16 is 3% of the gaps at 2M and 0.03% at 200M.
  - The "-1.5..+0.5 probes" in H15/H16 and "total worse in 20/20" cannot be carried over. The direction might hold;
    the magnitude is unknown.
  - The prototype also charges one table comparison (a 528 B, L1-resident array) the same as one root model probe
    (the 390 KB fence array). That price is unfavourable to G, and threat 17 admits as much.
- **k = 16 is fixed from the 2M result.** Pre-register a selection rule instead. For example: time the k from
  {1, 4, 16, 64} that minimises (root model probes + table comparisons) on the 200M fences, per dataset, or the
  smallest k that makes the root adopt a model where B falls back. Otherwise G is timed at a k chosen for a
  different regime.

### 3.2 At the global (AIDB) level: not tested before the full version, and the planned test would miss fb

- **The root fences cannot see fb's 21 outlier keys.** Fences are rows[4096·j] (index.hpp:456, 394). The last
  region starts at row 199,999,488. The 21 outliers are rows 199,999,979 and above, so all of them sit inside the
  last region, above the last fence.
  - The fence sequence on fb is pure bulk. `fences_ba.py` and P3 never see the outlier gap.
  - P5 as specified (`--gap-table` from `--dump-root-gaps`) applies the **root's** table to all 200M keys. On fb it
    would therefore **not** remove the outlier gap.
  - So the draft's fb predictions (H8: "< 5% change on fb"; Figure 9: "G moves only the global-line metrics") are
    consistent, but only because the test cannot reach the supervisor's own example.
- **The fix is cheap and should be in the overnight.** Add a key-level variant, G_key: the hardness tool selects the
  top-k adjacent gaps over **all** keys in one O(n) pass with a heap, applies the same shrink-to-median map, and
  recomputes the AIDB five.
  - Prediction: on fb, k = 2 (the 7.7e10 -> 1.4e14 and 1.1e15 -> 2.0e18 gaps) takes RMSE from 57.7M to about the
    fb_trim value of 141k, ME from 100M to about 325k, and PLA-32 and PLA-4096 change by at most 2k. That is the
    clean before/after compression result the supervisor is asking for.
  - On osm, books and planet, report D_k and RMSE/ME for k in {1, 4, 16, 64, 256}.
  - Cost: about 1 h of code in `hardness.cpp` and about 10 x 30 s of runs.
- **Then say what it means for the index.** G_key is an AIDB-metric result. The index's root never sees those keys,
  so G_key has no throughput consequence in SCALE-LI. State that on the slide, rather than letting the AIDB panel
  imply a speed-up.

### 3.3 The compression metric must be method-agnostic (I-1)

Gap removal and a monotone NFL flow are the same kind of object: a monotone reparametrisation of the key axis
placed before a linear model. G is piecewise-linear with 2k knots; the flow is learned. "G compresses, NFL warps" is
a distinction of words, not of measurement. Use, for every method on the same key set (the 200M fences for the root
level, all 200M keys for the AIDB level):

- (a) the **top-k gap share in the method's coordinate**. This works for any monotone map, because gaps map to gaps.
  D_k is then the drop in this share, defined for B, G, Nm and NFL-monotone alike, and 0 for CSV because CSV leaves
  the inputs alone.
- (b) the **outcome**: RMSE/n and ME/n of one least-squares line, and root model probes, in that coordinate.

Then the compression column reads honestly. G and Nm both squeeze the input axis (by different amounts, measured),
the faithful NFL flow is not monotone so the root cannot use it, and CSV leaves the inputs alone and smooths the
targets.

## 4. Warm-up and thermal: accepted, with two changes

The warm-up design is the strongest part of the draft. It has a first-principles bound (0.75M lookups to touch every
region), measured evidence (c1 = -0.3% ± 2% at 2M; cold -21% at 5M/16), prefault turned on as a paging alarm, flags
that never drop runs, and a pre-registered fallback to chunks 2-16. I accept 4M lookups. But:

- **The supervisor asked for "a proper warm-up", and will compare it with AIDB's 20M + 100M.** The draft's argument
  for 4M is sound. Still, the cheapest way to close the question is to run the BA/CA replication (fb, B and C,
  n = 3) in the **overnight**, not the full version. It costs about 0.7 h. Alternatively, adopt 20M everywhere for
  about +1.3 h a night. Do one of the two.
- **The thermal calibration T1 uses fb's C build (184 s) as the heat source, but the worst builds are books (441 s)
  and planet (370 s).** Calibrate on books. Also log the previous run's build seconds as a covariate. The heat a run
  inherits from the cell before it is randomised but not removed, and the S3 block adjustment does not see it.
- **PLAN.md D6 carried a condition that the draft dropped:** "reconsider in-process interleaving if E1 shows
  within-run 1.4x steps persist with the VM stopped". Keep it as gate G2b, judged on the per-chunk SD in E1.

## 5. What is missing

| item | expected? | recommendation |
|---|---|---|
| **A quiet-machine noise figure** | yes, first | It is planned (E1) but sits after 11-14 h of engineering. It needs no new binary: B and B2 on build-fs, 10 + 10 runs, with the VM stopped, in about 45 min including Q0. Run it on night 0. The machine is shared: `critique.md` reports 10 users and load averages of 70-180 from other people's processes. If Q0 cannot pass, every throughput claim moves to Linux, so start arranging the x86 Linux box now. Do not leave it as "optional". |
| Published baselines (ALEX, PGM, LIPP) | yes, eventually | Not for this meeting: fetching them needs a download approval, and PLAN D7 already declared them out of scope. On the summary slide, give SV as the non-learned reference and quote AIDB's published numbers on the same datasets as context, labelled as different hardware. |
| Range scans | no, for these claims | NFL-feature, CSV and G do not touch the scan path in key order. State that as out of scope with this reason. Faithful NFL breaks scans (39-385x read amplification), which is part of why it was not built. |
| Monotone NFL (Nm) as the key-order-preserving NFL arm | **yes** | Move it from the full D layer to the overnight D layer: 10 builds of about 7 s, about 12 min. It is the only way NFL can appear in the input-compression column. |
| fb outlier test (G_key) | **yes** | See 3.2. |
| A per-dataset map of where G and Cr change the root | yes | It comes free from the D layer. Print it before T and prune T by it. |

## 6. Scope: achievable before the meeting? What I would cut first

As drafted, the overnight version is 11-14 h of engineering, 19-21 h of machine time (12-14 h of it quiet) and an
evening of attended calibration. It also depends on a quiet-host condition never observed on this machine. Unless
the meeting is at least a week away, that is too much. Cut, in this order:

1. **The whole full version** (47-53 h). It is post-meeting by definition.
2. **RAW and RC in T.** The supervisor said compression is not about codecs. Keep RAW in the D layer as a memory
   reference only.
3. **P3, the C++ gap patch (Tier B), and with it the timed G and GCr cells** (5-8 h of engineering and 60 runs).
   - Reinstate them only if a pre-registered go-rule passes on the 200M fences from `fences_ba.py`. For example: on
     at least one dataset, G makes the root adopt a model where B falls back, or it cuts root model probes by more
     than its table comparisons.
   - The current evidence (2M, charged) predicts a null. Timing a predicted null across 10 datasets is the most
     expensive thing in the plan for the least information.
4. **The C++ k sweep (G1-G256) and Gs.** Do the k sweep in Python on the fences.
5. **W1** (30 min of quiet time). Its question is already answered by the pilot's c1 = -0.3% ± 2%, COLD in block 1,
   and the live c1 gate.
6. **Equal-n timing of structural nulls.** Drop T cells whose D-layer signature equals B's on that dataset, and give
   the time to B, Cr and C at n = 8 where root fences are adopted.

Add (cheap):

- E1 on night 0;
- `fences_ba.py` at 200M before any C++;
- G_key in the hardness tool;
- Nm in the D layer;
- BA/CA on fb, n = 3.

The resulting plan: about 3-4 h of engineering (P1, P2, the P5 key-level variant), a D layer of about 5 h, and a T
layer of about 8.5 h at c* = 30 s. That is 7 cells (B, B2, N, Nf, Cr, C, SV) x 10 datasets x 3, plus COLD, plus
BA/CA. It fits one quiet night, if night 0 shows the night can be quiet.

## 7. Wording of method labels (faithfulness)

- **"N / after NFL"** should be **"NFL-style flow feature (stand-in 1-D trainer, 16k-key sample), key-ordered,
  batch 1"** on every table and figure, not only in threat 13. Use "Nm: monotone flow" for the key-order-preserving
  arm.
- **"C / after CSV"** should be **"CSV Algorithm 1 in regions + CSV-style virtual fences at the root (our
  extension)"**. Most of C's predicted speed-up is the root mechanism, which is not in the CSV paper
  (`critique.md` 3.4). Headline Cr vs B ("our root extension") and C vs Cr ("CSV's own region use") separately,
  rather than C vs B.
- **"G / gap removal"** should be **"k-gap shrink-to-median map at the root (k = 16)"**, and "G_key" for the AIDB
  version. Say that it is an input-space reparametrisation of the same family as a monotone flow.
- **The hardness "after CSV" panel is not the index's CSV.** The tool uses double(key - global min) features. The
  index uses region-normalised features and placed 19.97M virtual points on osm, against 10.23M in the tool. Label
  the panel accordingly.

## 8. Issue index (referenced above)

| id | severity | one line |
|---|---|---|
| I-1 | blocker | The input-compression metric (D_k) is defined only for G, so "NFL/CSV do not compress" holds by definition. This contradicts the monotone-flow results in hardness_ba F4/F5. Use a method-agnostic gap share plus line-fit outcome, and add Nm. |
| I-2 | blocker | The root fences exclude fb's 21 outliers (all in the last region, above fence 48,828). Neither root G nor P5-with-root-table can test the supervisor's flagship example. Add key-level G_key in the hardness tool, in the overnight. |
| I-3 | blocker | Every throughput claim depends on sigma_AA <= 5% on a shared Mac where the VM is present in 98% of samples. E1 is scheduled after 11-14 h of engineering. Run Q0 + E1 on build-fs on night 0, and line up Linux now. |
| I-4 | major | H7/H15/H16 extrapolate from 2M samples (about 488 fences, 2-8 probes) to 200M (48,829 fences, fb 10.80, osm 24.91 with binary fallback). Run fences_ba.py at 200M before freezing predictions and before P3. |
| I-5 | major | Timed G is likely a structural null where gaps concentrate (osm falls back to binary), and G has nothing to act on where the root is learned (fb's gap share is 0.036). k = 16 is fixed from 2M. Pre-register a go-rule and a k-selection rule. |
| I-6 | major | Equal n = 3 over 10 datasets x 9 cells. The one predicted positive effect (C on fb, via the root) is unresolved per dataset at n = 3. Prune structural nulls from T using D, and move repeats to B/Cr/C where root fences are adopted. |
| I-7 | major | Labels. "After CSV" includes our root extension, which carries most of the effect. "After NFL" is a stand-in flow feature in a key-ordered index. Apply the labels in every table and figure. |
| I-8 | minor | By-construction equalities (H1, H2, H5, H6) are listed as hypotheses. "NFL costs 0 B/key" needs the key-order clause. |
| I-9 | major | Scope. As drafted, 11-14 h of engineering plus 19-21 h of machine time. Cut the full version, RAW/RC timing, P3 (unless the go-rule passes), the C++ k sweep, Gs and W1. Add E1 on night 0, G_key, Nm, and BA/CA on fb. |
| I-10 | minor | AIDB-protocol replication (20M/100M) is full-version only, although the supervisor explicitly asked about the warm-up. Run BA/CA on fb in the overnight. |
| I-11 | minor | T1 calibrates heat on fb (184 s build), not on the worst case (books 441 s). Carry-over heat from the previous cell is not logged. |
| I-12 | minor | PLAN D6's conditional in-process interleaving rule was dropped. Restore it as a gate on E1's per-chunk SD. |
| I-13 | minor | Published baselines and range scans are only mentioned in threat 14. Put them on the scope slide with the reason; quote AIDB's published numbers as context. |
| I-14 | minor | The hardness "after CSV" panel differs from the index's CSV (osm VP 10.2M vs 20.0M). H3 "saturates on all 10" rests on 2 datasets. |
