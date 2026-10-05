# Designer 1: what a faithful before/after for NFL and CSV would take (2026-10-01)

Scope: what SCALE-LI would need to run each method as its paper describes it, what that costs, what it breaks, and what it would plausibly do to memory, compression and throughput at 200M keys. Then a separate build-or-scope decision for NFL and for CSV.
No benchmark, build or hardness job was run, because a 200M timing pilot was running. The only computations were these:
(a) a weights-only analysis of the ten faithful flows (`tooth_overlap.py` / `tooth_overlap.json` in this folder; it reads no data and takes milliseconds);
(b) reading the first and last key of each 200M file;
(c) a Poisson calculation.
Everything else comes from code and existing result files, cited by path:line.

S = /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli

---

## 0. Three facts that decide the design (new, from files already on disk)

**F1. The faithful NFL flows behave like a modular hash, not a CDF straightener.** Each of the ten `flows_free/<d>_s{64,512}_t2000.txt` files has the same structure:
- Layer 0 weights on x are about -4e-4 (s512) or -3.4e-3 (s64). The weights on frac(x) are about 0.16-0.33.
- The tanh arguments stay in [-0.25, 0.33], the near-linear regime.
- So z is approximately C*((a-b)*frac(x) - b*floor(x)), with x = (key-mean)/var spanning [0, s].
- Within one "tooth" (one unit of x, i.e. range/s in key space) z is monotone. Successive teeth are shifted down by only 0.0034-0.042 in z, against a tooth width of 2.0-2.7.

Sorting by z therefore interleaves the teeth. Assuming equally populated teeth, a key's z neighbourhood holds keys from a mean of 38.7-49.3 of the 64 teeth (books, covid, genome, history, libio, stack, wise). For the s512 flows it holds keys from 342-385 of 512 teeth (fb 384.7, osm 342.5, planet 364.1) [tooth_overlap.json].

This is why D99 drops. A uniformly random hash scored by our evaluator has D99 = 3:
- capacity is about n because `max_size` is min(1.5n, fitted span) [transform.hpp:107];
- the occupied-slot counts are Poisson(1), with P(count <= 3 | occupied) = 0.970 and P(count <= 4 | occupied) = 0.994.

The faithful flows reach 3 on 7/10 datasets, 4 on genome and 6 on osm [aidb_flowv2/flowcheck_summary.json]. They are at or above the no-learning hash floor. NFL's "around 4" (Table 3) matches that floor too.

Consequence: a z-ordered store is close to a hash-partitioned store. Its honest control is a hash order (for example z = frac(key*phi)), not the key-ordered map.

**F2. The CSV memory cost in our host belongs to the update path, not to the lookup path.** `virtual_features` is read only by the compaction re-placement [index.hpp:296-311] and counted in `memory()` [index.hpp:340]. Lookups use `rank_model` and `blocks[].slot_begin` only [index.hpp:121-134].

At 200M, fb metadata is 114,063,088 B without virtual points and 274,027,920 B with them. The difference of 159,964,832 B equals the 19,970,703 virtual points × 8 B exactly [aidb_fullscale/sweep/results.jsonl].

So "after CSV" memory is +0.80 B/key if the index supports compaction, and +0.00 B/key for a static (read-only) build. Compression is unchanged by construction: key_bytes are identical at 386,797,735 B (fb) and 276,840,814 B (planet) with and without virtual points.

**F3. The largest probe term the CSV variants leave alone is the coordinate search.** Every 200M variant spends 5.03 coordinate probes per lookup [results.jsonl work_counters], which is log2(32 blocks). That search maps a predicted slot to a block by binary search over `slot_begin` [index.hpp:126-133]. For comparison:
- CSV's region-level saving is 0.40 fence probes on fb (5.14 -> 4.74) and 0.57 on planet (3.99 -> 3.42).
- In CSV's own hosts a predicted slot IS the storage position, so there is no such search.

The faithful element of "virtual points are slots" is an O(1) slot-to-block table, like the root's `root_slot_to_region_` [index.hpp:353, 410-411]. That table also helps the before arm (block = floor(rank/128) without virtual points), so it is a host fix, not a CSV effect.

**Correction to the brief.** The brief says the selector picks the flow in "0 of 60 root cells". That holds at 2M only. At 200M on planet, `packed_rank_root_fusion` chose **flow+fences**:
- root probes 3.32, against 13.69 for fences alone and 15.66 for binary search;
- 111,688 virtual fences and 2,973 s of build;
- 1.01 transform calls per lookup;
- single-seed speedup 1.175, inside the 44% noise range.

That run used the old INERT monotone planet flow (results/aidb/flows/planet_2D2H2L.txt) with flow_cost 4 [aidb_fullscale/root_analysis.txt, verdict "synergy=True"].
Mechanism: at 200M a root probe is a pointer chase. One lookup is 1/0.58 Mops = 1.72 us for about 33 comparisons, about 50 ns per comparison, so a 28-66 ns transform costs only about 0.6-1.3 probes. A flow_cost fixed at 4 is too high at this scale, not too low.

---

## 1. NFL, faithful

### 1.1 What "faithful" means
NFL computes z for every key and builds AFLI over pairs **sorted by z** [NFL Alg 3.1-3.2; guide_deep/01_nfl.md 3.7]. A lookup transforms the query (batched at 256) and searches for its z. AFLI's model nodes place keys exactly, conflicts go to buckets of at most 6 keys, and dense nodes are binary searched.
The paper evaluates point lookups and inserts only, with no range scans. The switch is one dataset-wide D99 comparison.

Our host instead uses z only as a region or root feature over key-ordered records [transform.hpp:14-16; index.hpp:106-109, 280-291].

### 1.2 Two faithful options considered

**N-A: run the official NFL/AFLI.**
- License: GPL-3, so it can be run but not copied into the MIT repo.
- Inference uses Intel MKL, which is x86-only, and this host is an Apple M3 Max (arm64). Faithful timing needs a port of the MKL calls (cblas/vdTanh to Accelerate) or Rosetta, which would make timing meaningless.
- Training is PyTorch B-NAF on a GPU, which violates the stdlib-only rule. Our stand-in weights are format-compatible.
- AFLI has no compression, and its footprint is 2.26x ALEX [NFL 4.4.3]. The compression column would be N/A and the memory column would not be comparable to ours.
- Effort: 3-4 days. **Rejected.** It does not answer the supervisor's question about our index.

**N-B: z-ordered SCALE-LI (the meaningful faithful variant).** What to implement, in scratch, with repo headers read-only:
1. **Bulk load** [replaces index.hpp:453-468].
   - Compute z for every key: 200M × ~30 ns = 6 s, more with 16 threads.
   - Sort (z, key, value) by z: about 30-40 s at 200M.
   - Cut into 4,096-pair regions with z fences (double).
   - z ties: NFL assumes a 1-to-1 map; ties need an equal-z run scan.
2. **Codec.** `encode_block` rejects unsorted input [codec.hpp:86] and `decode_block` re-checks [codec.hpp:199].
   - FOR and Linear use `keys[i]-keys.front()` [codec.hpp:100, 105], which underflows out of order.
   - Delta varints `keys[i]-keys[i-1]` [codec.hpp:126], which needs zig-zag coding.
   - Needed: raw plus FOR with a min base. Delta is pointless (see 1.3).
3. **Locate.**
   - Root model over z fences (the existing `locate_from_prediction` on doubles).
   - Region model over z with block z-fences in place of `blocks[i].last < k` [index.hpp:140].
   - In-block search on z needs one of two things. Either store z per key (+8 B/key), or compute the flow on every probed key (8 key_at × 28-66 ns = 220-530 ns per lookup). The alternative is a linear key-equality scan of the 128-key block.
4. **Batching.** Transform 256 queries, then search them. Fairness then requires batching the before arm too. Otherwise the experiment measures batching or memory-level parallelism, not NFL. The design needs a 2×2 of {batched, unbatched} × {flow, none}.
5. **Range scans.** A [a, a+100] scan lies in one tooth (a tooth holds about 390k keys at s512, about 3.1M at s64). Its z interval also contains keys from the ~39-385 other interleaved teeth. A correct scan must read and filter 39-385× the records, or it is declared unsupported, which is NFL's own scope.
6. **Updates and compaction.** Drop them and declare the experiment read-only. `rebuild`, `set_delta` and the delta merge in `scan_into` are all key-ordered [index.hpp:171-199].
7. **CSV root fences cannot combine with it in the current code.** The slot table requires monotone features [index.hpp:406]. Region smoothing on unsorted features silently skips non-increasing gaps [smoothing.hpp:76]. In z-order the feature is sorted, so this is fine, but it is new code.

**Effort for N-B.**
- Engineering: about 15-20 h (2-2.5 days). Load/sort/partition 3 h, codec 2 h, locate with 2 in-block modes 4 h, batched driver plus memory accounting plus warm-up parity with build-fs 3 h, tests against a reference on 2M samples (misses, z ties) 3 h, checking the 2M-trained flows' D99 at full scale 1 h.
- Machine: 10 datasets × {before, after} × {batched, unbatched} × 4 repeats (10% resolution at sigma 6.5%) × about 2 min = **about 5.3 h serial**.
- Total: about 3 days including analysis.

### 1.3 What faithful NFL would plausibly do (predictions, labelled as such)

**Compression and memory: a clean, large negative.**
- In key order a 128-key block spans about 128 mean gaps.
- In z order a block draws from about min(128, 39-385) distinct teeth, so its FOR width is about the bit width of the dataset's bulk key range. This width does not depend on n, unlike key order.
- Measured first and last keys of the 200M files give these bulk range widths: books 63, fb 37 (the core range; the upper outliers saturate the tanh and sort to one end), osm 64, covid 57, genome 39, history 34, libio 29, planet 34, stack 28, wise 43 bits.
- Predicted key bytes per key (width/8 + 0.125 header): books 8.0, fb ~4.75, osm 8.1, covid 7.3, genome 5.0, history 4.4, libio 3.8, planet 4.4, stack 3.6, wise 5.5.
- Measured in key order today: fb 1.93 and planet 1.38.
- So the key arena grows **2.5x on fb and 3.2x on planet**, from a compression ratio of 4-6x down to 1.0-2.2x.
- Totals: fb 10.50 -> about 13.3 B/key, planet 9.95 -> about 13.0, both about +30%. If z is stored for the in-block search, add +8 B/key.
- Delta coding does not recover anything: consecutive z-ordered keys come from random teeth, so |delta| is about range/3.

**Point-lookup throughput: plausibly no better than restricted CSV already gives.** Comparisons per lookup:

| Arm | root | coordinate | fence | key_at | total |
|---|---|---|---|---|---|
| fb before | 15.66 | 5.03 | 5.14 | 8.02 | 33.8 |
| fb, restricted CSV root fences | 3.42 | 5.03 | 4.74 | 8.02 | 21.2 |
| faithful z-order (prediction) | ~2-3 | 5 | ~1.5-2.5 | 8 | ~17-19 |

The faithful arm also pays one transform. The transform is 28-66 ns, about 2-4% of a 1.5-1.7 us lookup at 200M, so NFL's batching argument (169.5 -> 8.4 ns) moves at most about 4% here.
Predicted: within about +-20% of the restricted-CSV arm, and +0 to +30% against the plain before arm. At sigma 6.5% that needs at least 4 repeats per cell to resolve 10%. The identical fb fences structure has already shown 0.65/0.72/0.81 Mops at one seed.

**Range scans:** 39-385x read amplification, or unsupported.

**NFL + CSV under faithful NFL.** CSV should add about nothing: z is already near-uniform, so the linear-model SSE that CSV attacks is near its floor. Predicted combined ≈ NFL alone. In the restricted host the two compose exactly; that is already established.

### 1.4 Decision for NFL: **do not build N-B for this meeting. Present the restricted version with its scope stated, and add a cheap faithful-layout audit.**

Why:
- The faithful version's memory and compression outcome is predictable from F1 and the key ranges, and it is negative.
- Its throughput outcome is unlikely to be resolvable on this host at the cost.
- It breaks scans, delta coding and updates.

**What to present instead (about 3-4 h, CPU-light, after the pilot ends).** A Python-stdlib "z-order audit" on the existing 2M uniform samples (data/samples/<d>_2M_uniform_s42). For each dataset:
1. Transform with the faithful flow and sort by z. That is 2M flow evaluations in pure Python, about 3-5 s per dataset.
2. Compute per-128-key-block FOR width (bit_width(max-min)) and the zig-zag delta varint size. This gives the faithful key bytes per key, which (per F1) carries over to 200M.
3. Compute D99 of the z order, of a multiplicative-hash order (frac(key*0.6180339887)) and of the key order.
4. Count teeth per block, which measures F1 directly.

This turns the memory/compression column of "after NFL (faithful)" into measured numbers and tests the hash interpretation. Throughput stays a prediction, stated as such. Predicted audit result: hash-order D99 = 3 = flow-order D99 on 7/10 datasets, and z-order FOR widths within about 1 bit of the bulk range width.

---

## 2. CSV, faithful

### 2.1 What our host implements and what it does not
- **Implemented: Algorithm 1 per model** [smoothing.hpp:47-97], per region on slot targets [index.hpp:311-315] and at the root as virtual fences with an O(1) slot-to-region table [index.hpp:403-412].
- **Fidelity differences.**
  - O(lambda·n) instead of the paper's claimed O(n+lambda). That claim is unjustified per guide_deep/02_csv.md 4.5.
  - Real-valued candidates instead of integer ones.
  - Loss probes plus ternary search instead of the derivative chord.
  - Virtual points are slot offsets, not empty array slots.
- **Missing: Algorithm 2.** It is a bottom-up merge of a node and its subtree into one smoothed leaf when a cost condition holds (eq. 22 for ALEX; "holds more keys" for LIPP and SALI). It runs inside ALEX, LIPP and SALI.

### 2.2 What faithful CSV would require here

**C-A: Algorithm 2 analogue in SCALE-LI.** "Collect subtree, smooth, rebuild as one leaf" maps to merging k adjacent 4,096-key regions into one larger smoothed region, accepted when eq. 22 holds: search_const × expected fence+coordinate probes + traversal_const × root probes falls.
- Implement:
  - a variable partition in bulk_load (Region already accepts any size: splits at index.hpp:431, validate allows up to 16M);
  - candidate merges at 2/4/8x;
  - the cost model with measured constants;
  - a slot-to-block table (F3).
- Effort: about 10-12 h.
- Machine: smoothing is O(alpha·n²/regions) [granularity prep_us/key at 4k/16k/32k: fb 18.0/90.5/207.7, planet 18.1/72.0/137.5]. Evaluating three granularities at 200M is about 200M × 230-320 us / 16 threads = **47-66 min per dataset**, so about 8-10 h serial for ten datasets, plus timing runs.
- Breaks nothing.
- Prediction from the granularity ablation [aidb_granularity/summary_r*.csv]:
  - Against a **binary** root, bigger smoothed leaves win: root+fence probes fb 11.14 (4k) -> 8.25 (32k), planet 11.56 -> 10.14, covid 11.29 -> 9.53. They lose on osm (13.60 -> 15.38) and are flat on genome.
  - Against the **virtual-fence root** that already exists, the merge can save at most the 2.1-3.4 root probes left, while fence probes rise (planet 2.60 -> 4.15 at 32k).
  - So: within about +-1 probe of the restricted arm, negative on planet, genome and osm.
  - Memory: +0.8 B/key (0 static). Compression unchanged. Throughput unresolvable.
- The underlying reason is structural. Algorithm 2's lever is promoting keys out of deep, unbalanced subtrees. In our balanced two-level host every key has the same path (root -> region -> block), so there is nothing to promote. Root virtual fences already attack the traversal term that Algorithm 2 targets.

**C-B: CSV in its own host (LIPP).** A LIPP header exists locally (/Users/louisvasseur/Downloads/scaleli_sota/build-verify-metrics/scratch/lipp.h, used for the FMCD check; its license must be checked).
- Implement Alg 1 + Alg 2 with the LIPP cost condition. "Reconstruct" is underspecified in the paper (guide 5.2), so it has to be designed.
- Effort: 3-4 days.
- Machine: CSV's own LIPP preprocessing at alpha 0.1 is 337-2,329 s per dataset [CSV Table 3], so about 4-8 h for ten datasets plus 200M LIPP builds.
- It produces no compression metric (LIPP stores raw keys in gapped arrays), and it is not our index. **Rejected for this meeting.**

**C-C: cheap faithful-izations (hours, not days).**
1. Report memory both ways (F2). No run is needed: metadata minus 8 B × virtual points. A static build could free `virtual_features`; that is a one-line change, to be verified once later.
2. State that compression is unchanged by construction (identical key_bytes).
3. Slot-to-block table (F3): about 3-4 h. It removes about 5 coordinate probes per lookup on **both** arms. It is CSV's position semantics, but it is a host improvement and must not be booked as a CSV gain.

### 2.3 Decision for CSV: **present the restricted version (Algorithm 1 per region + root virtual fences) with its scope stated. Do not build Algorithm 2 or a LIPP host for this meeting.**

Effort: 0 engineering hours. The runs are the ones the experiment needs anyway.

Predicted and partly measured outcome at 200M:
- **Region virtual points.** Fence probes fb 5.14 -> 4.74, planet 3.99 -> 3.42. Memory +0.80 B/key (0.57 -> 1.37), or +0 static. Compression unchanged. Build +155-185 s at 16 threads.
- **Root fences.** fb root 15.66 -> 3.42 (973 fences, +3.4 s build). Planet needs budget 4 (13.69, 2,068 s) or flow+fences (3.32, 2,973 s).
- **Throughput.** Inside noise unless at least 4 repeats per cell (the fb fences triple 0.65/0.72/0.81 Mops).

Scope sentence for the slide: "CSV Algorithm 1 applied per region (4,096 keys) and to the root's region fences. Algorithm 2 (hierarchical merge) is not implemented: the host is a balanced two-level map with no deep subtrees to promote. The root virtual fences target the same traversal cost. Virtual points are slot offsets, not stored gaps."

---

## 3. Summary table

| Method / version | Implement | Eng. effort | 200M machine (serial) | Breaks | Memory (B/key, fb / planet) | Key compression | Point throughput | Scans | Decision |
|---|---|---|---|---|---|---|---|---|---|
| Before (packed, binary root) | none | 0 | ~3 min/run | n/a | 10.50 / 9.95 (measured) | 4.1x / 5.8x (measured) | 0.648 / 0.580 Mops (1 seed) | ok | baseline |
| NFL restricted (z as model feature, key order) | none | 0 | runs only | nothing (flow refused by selector) | +0 (+144 B) | unchanged | region: selector 0/6,740; root: planet 200M picks flow+fences (3.32 root probes) | ok | **present, scope stated** |
| NFL faithful N-B (z-ordered SCALE-LI, batched, read-only) | z-sort load, min-base codec, z fences, in-block z search, batched driver | 15-20 h | ~5.3 h (2x2 batching × 4 reps × 10 ds) | range scans (39-385x read amp.), delta/FOR/Linear codecs (codec.hpp:86,100,126), updates, CSV root fences (index.hpp:406) | ~13.3 / ~13.0 (+30%), +8 more if z stored | 4.1x -> ~1.7x fb, 5.8x -> ~1.8x planet (est.) | predicted ±20% of restricted-CSV arm; unresolvable <10% at <4 reps | broken | **do not build**; run 3-4 h z-order audit on 2M samples instead |
| NFL official (AFLI binary) | MKL->Accelerate port, PyTorch training | 3-4 days | 4-8 h | stdlib rule, no compression | AFLI ~2.26x ALEX, not comparable | N/A | NFL vs AFLI only | not evaluated | reject |
| CSV restricted (Alg 1 regions + root fences) | none | 0 | +155-185 s build/run (regions), +3 s fb root, +2,068 s planet root | nothing | 11.30 / 10.75 compaction-ready; 10.50 / 9.95 static | unchanged (identical key_bytes) | fence -0.40 / -0.57; root 15.66 -> 3.42 fb | ok | **present, scope stated** |
| CSV Alg 2 analogue (adaptive region merge, eq. 22) | variable partition, cost model, slot->block table | 10-12 h | 8-10 h smoothing + timing | nothing | as restricted | unchanged | within ±1 probe of restricted; negative on planet/osm/genome | ok | do not build |
| CSV in LIPP (paper host) | Alg 1+2 inside LIPP | 3-4 days | 4-8 h | not our index | LIPP gapped arrays, not comparable | N/A | ~30% on promoted keys per paper | n/a | reject |
| Host fix: O(1) slot->block table (both arms) | per-region table | 3-4 h | reruns | nothing | +~0.03 | unchanged | -5.03 coordinate probes/lookup on every arm | ok | optional, not a CSV effect |

---

## 4. Caveats
- F1 is a weights-only analysis that assumes equally populated teeth. Real data leaves teeth empty (osm has 321 inversions, not 512), so the interleaving counts are upper-side estimates. The z-order audit measures them.
- The flows were trained on 2M uniform samples with our stand-in 1-D trainer (no biases, base sigma 1, not 1e16). The author B-NAF could place its weights differently. The input encoding [x, frac(x)] and the sum decoder are NFL's, and the sawtooth mechanism is the paper's.
- Predicted z-order key bytes use the bulk key range from the files' first and last keys. fb assumes its upper outliers (keys above about 7.7e10, the 2M sample max) sort to one end because tanh saturates. That is not measured.
- Every throughput number at 200M is single-seed (sigma about 6.5%, 44% range across 22 repeats). No time claim is made here.
- The planet 200M flow+fences result uses the INERT monotone flow and one seed. It changes the brief's "0 of 60 root cells" (a 2M statement), not the region-level conclusion.
- The random-hash D99 = 3 is a Poisson(1) calculation under our evaluator's capacity (about n). NFL's own capacity rule may differ.
