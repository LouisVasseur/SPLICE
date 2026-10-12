# SPLICE-H: the read-only design to implement

**What this is.** This design merges the parts of the four designs that held up under critique. Each segment of the key space is placed in one of two modes. **DIRECT** uses model-placed slots and fits Poisson-like or near-linear data. **FENCED** uses a model-addressed 128-B directory entry whose fences point to the exact data line, and fits clustered data. A cache-resident integer top sits above both modes. Every choice is made by an exact count over all 200M keys of the dependent DRAM lines and page walks per lookup. Metadata is priced by simulating the shared L2.

**Status.** All ns and Mops/s figures below are estimates built from calibrated constants. Only the counts and the fb baselines on DIAS are measured. "Critique" refers to the four adversarial reviews in the brief.

---

## 1. Architecture

**Cost currency.** We count serialized DRAM accesses, D. The measured sorted array (2.18 µs, about 18.3 serialized accesses) caps D at about 119 ns, which gives **D = 100-120 ns, best estimate 105-110**. The 130-170 values used in some designs contradict that measurement. Other figures used throughout:
- L2 hit: 17-19 cycles, about 9 ns. L1 hit: 3 cycles. Clock fixed at 2.0 GHz.
- With 4 KiB pages, a walk into a multi-GB array costs about 0.8-0.95 D. This comes from an LRU simulation that includes the real cold-line stream (critique of D2: directory PTE hit 0.19, data PTE hit 0.06).
- With 2 MiB pages, a walk costs 10-30 ns.
- Sources: Intel ARK C3958; Agner Table 16.2; Intel Opt. Manual Vol.2 Table 6-7; https://www.7-cpu.com/cpu/Goldmont.html

| Component | Contents | Size and residency | Why |
|---|---|---|---|
| **E. Exception list** | Up to 64 keys peeled at each end, so the remaining span is within 2x of the 64/64-trimmed span. Reached through one rarely taken range check. | 1-2 L1 lines | On fb this catches exactly the 21 outliers, including 2^64-1. Raw radix otherwise puts all 200M fb keys in one bucket (splice_wk/radix.json). Never use UINT64_MAX as a sentinel. |
| **R. Router** | Count-split dyadic radix trie over the knot keys, Hist-Tree style. Top level 2^11-2^12 entries of u16/u32 (8-16 KiB, L1). A child table with its own shift is created where a bucket holds more than 8 knots. Ends in a one-line scan of at most 8 keys (PCMPGTQ, SSE4.2). | ≤ about 16 KiB in L1, child tables in L2. Counted inside the L2 budget. | A raw 2^12 radix is 36.5x the ideal bucket on osm and leaves 237 candidates; planet leaves 25.8. Count-split depth on osm: 74% of keys at depth 1, 14% at depth 2. Only shifts are used, no clz/BSR (BSR is 10 cycles; there is no LZCNT). |
| **K. Tier-1 segments** | Optimal ε-PLA (the int128 hull in hardness.hpp), with ε bisected to fit K1. One merged **16-B record** per segment: knot key, fixed-point slope m with pre-shift, base, mode bit. The exact bit packing is checked by static_assert. | **K1 ≤ 16k, ≤ 256 KB including keys** | Separate knot-key and record arrays (24 B per entry), or K1 = 32k, add +0.33 to +0.66 D per lookup of tier-1 misses (critique l2sim). Merged 16k records miss 0.155 D with a 2 MB L2 and 0.42 D with 1 MB. On FENCED datasets the selector may choose about 1-5k segments: entries holding more than C keys are about as common with 1,166 leaves as with 18,384 (fb 0.1245 vs 0.1173), so finer tier-1 resolution buys nothing there. |
| **Prediction** | `u = (x - k_j)` as unsigned integer subtraction, then `>> pre`, then a 64x64 to 128-bit MUL by m, then `>> 32`, then `+ base`. | 8-12 cycles (MUL r64: 6 cycles) | No libm, std::fma, DIVSD (34 cycles), DIV r64 (14-43 cycles), x87 or u64-to-double conversion. Predictions are bit-identical on Mac and Atom. The pre-shift is mandatory: books spans 2^63, and 5 fb keys are ≥ 2^63, which would overflow an int64 subtraction. |
| **DIRECT segment** | ceil((1+α_j)·n_j) 16-B {key, payload} slots, line-aligned. Placement is order-preserving: forward cumulative max, then a backward clamp. Each gap copies its neighbour, so the array stays sorted and scannable. Lookup issues loads for w_j ∈ {1,2,4} adjacent lines together, compares with pcmpeqq and movmskpd, then scans right in the rare overflow case. | Arena (see below) | Easy data at 200M: stack 1.04 lines; wise 1.31 (α=0.5) to 1.64 (α=0.25); history 1.46-1.91; covid 1.51-1.94 (sequential line counts, impl/*.json). In windows with w ≤ 4 the counts are 1.000-1.034 (memfirst/final_w4.jsonl). |
| **FENCED segment** | The segment model selects a **128-B directory entry**: two lines in one block, loaded together. An entry holds a u64 base key, a u32 first-line offset counted in 64-B lines (reaches 256 GB), a fence shift, flags, and 56 biased u16 fences, one per data line, so at most 57 lines or 228 keys. A **branch-free count over all 56 fences** (pcmpgtw, packsswb, pmovmskb, popcnt) gives the exact data line. Ambiguous u16 fences fall back to the previous line, which is in the same page. Entries over capacity become inner entries pointing to child entries. osm needs a second inner level for entries above 13,224 keys (0.13% of keys). | Directory about 5.3 B/key (fb about 1.07 GB). Data lines hold 4 keys then 4 payloads. | At 200M with C = 228, the share of keys whose entry exceeds capacity: fb 0.134, osm 0.191, planet 0.074, genome 0.0009, all others ≤ 0.0003 (critique advsim). 56% of fb keys and 47% of osm keys sit in entries of more than 17 lines, so the fence count must not branch. |
| **Arena** | One allocation, 64-B and 2 MB aligned, holding all slots, directory entries and data lines. Pages are pre-touched at build. MADV_HUGEPAGE is behind a flag (off by default). | 3.8-4.8 GB | No per-region heap objects, no separate bitmap or payload arrays, no stores on the read path. Those cost LIPP, ALEX and SCALE-LI one dependent miss each. |

**Lookup path.**
- Exception check, then router (L1/L2), then tier-1 record (L2), then the integer prediction.
- DIRECT: slot window → compare.
- FENCED: entry (2 lines) → fence count → data line → compare, plus a child entry for the 0-19% of keys in over-capacity entries.
- Common path on 4 KiB pages: **DIRECT ≈ 2.2-2.4 D; FENCED ≈ 3.75 D** (0.81 D directory walk + 1 D entry + 0.94 D data walk + 1 D data line).

---

## 2. Learning signal

**Objective.** For a parameter set θ:

J(θ) = E over stored keys of [ D · (N_lines + P_walk-miss + P_table-miss) + τ · extra parallel lines + c_compute ] + λ · bytes/key

- θ covers: K1 and ε, mode per segment, α_j, w_j, entry width (1 or 2 lines), cbar ∈ {16, 24, 32}, compact on/off, and α ∈ {0.25, 0.5}.
- **The terms that never depend on cache state** are counted exactly in one O(n) dry run of the real integer layout: N_lines, fence depth, over-capacity share, distinct pages, and page straddles. Spill must carry across segment boundaries. GRE looks up stored keys uniformly, so this count equals the workload's expectation and there is no generalisation gap.
- **Residency terms** (P_walk-miss and the tier-1 and router misses) come from one **shared 16-way LRU simulation of L2**. It covers every hot table plus the cold stream of each lookup (entry lines, data line, both PTE lines), at both 2 MB and 1 MB effective L2. Independent per-table step charges underestimated E[D] by 0.3-0.8 D in critique.
- τ ≈ 3.3 ns per extra line within a batch: 64 B over one 19.2 GB/s DDR4-2400 channel, arithmetic. It applies only up to the 8-WCB cap. **A 2-line entry is budgeted at 1.0-1.1 D and a 4-line window at 1.1-1.3 D** until DIAS confirms.

**Why this objective and not the alternatives:**
- SOSD: cache misses explain lookup time (R² = 0.955). Once misses are included, log2 error is not significant (https://arxiv.org/abs/2006.12804).
- On a model-placed layout, every error proxy mis-ranks models: mean Spearman 0.75 for log2 error, 0.77 for MSE, 0.33 for LIPP's conflict degree, -0.16 for NFL's tail conflict, and down to 0.07 on history (proxy/surr.json).
- NLL moved 0.01 nats while lines went from 9.9 to 3.2 on books.
- The repo's joint loss, flow loss and root-probe currency all decoupled from cost.

**How it is optimised.**
1. Fit deterministically, once, in this order: exceptions, then tier-1 ε-PLA with ε bisected (about 9 O(n) passes), then router.
2. For each segment, enumerate the candidates and choose by multiple-choice selection with a Lagrangian (λ for bytes).
3. Because residency couples segments, wrap step 2 in an outer fixed-point loop: choose, recompute footprints, re-simulate L2, choose again.
4. There is no G/T/V alternation. Joint alternation was refuted: at most 0.072 probes gained, at about 2,000 s per round at 200M.
5. A refinement is accepted only if it saves more than 0.01 lines per lookup. A grid configuration replaces the baseline only if predicted E[D] drops by at least 0.3 D. Never use a strict '<': that rule turned the fb 10M tie into a loss.

---

## 3. Learning protocol

- **Data.** All 200M keys, read from the `*.sorted` files. The raw GRE files for covid, genome, history, libio, planet, stack and wise are unsorted: about 50% descents in planet, genome and covid. Never use 2M uniform samples, which flatten clustering. Windows are for unit tests only: osm's over-capacity share is 0.097 in windows against 0.191 at full scale, and books' window extrapolation was off by 2.2x.
- **Cost of a count pass.** About 9 s per dataset per configuration on the Mac (mlp/fullsim.cpp; bookkeeping, not a benchmark).
- **Grid per dataset.** K1 ∈ {about 1-5k, 8k, 16k}, α ∈ {0.25, 0.5}, w ∈ {1, 2, 4}, entry width ∈ {1, 2}, cbar ∈ {16, 24, 32}, compact ∈ {off, on where the line span is below 2^32}. That is about 10^2 cells, scored at about 9 s each, so minutes per dataset. Prune it as SPLICE-M's critique advises: its 300-450-pass builder was estimated at about 3,000-4,500 CPU-seconds per dataset.
- **Selection.** Minimum J subject to a stated bytes cap. The cap is either ≤ 22 B/key (fast) or ≤ 16.5 B/key (compact). **Report the whole Pareto front of bytes against lines.**
- **Validation, with no hold-out needed for read-only:**
  1. A SPLICE_COUNT build of get() must match the simulator's lines, fence depth and pages exactly on all keys. Residency terms cannot be counted inside get(), so they are checked with perf counters.
  2. All 200M keys must return their payloads.
  3. A perturbation test of x ± 1e-6·x² must leave the choice unchanged.
  4. Mac and x86 builds must be byte-identical. No long double anywhere.
- **Build budget.** Target ≤ 2x PGM's build time. Build in 3 streaming passes after the fit (D2 plan). Reuse GRE's already-sorted array: one aligned copy, no stable_sort.

---

## 4. G, T and V: three scales, one owner each

| Block | What it owns in SPLICE-H | Evidence that it stays complementary |
|---|---|---|
| **G** (gaps, outliers) | The exception list, ε-PLA knot breaks at holes (the hull places them automatically), and the count-split router. **No gap table.** | A G table was chosen in 0 of 240 charged cells. G-64 changes radix steps by ≤ 0.01 on 7 of 10 datasets, and computing the compressed coordinate costs more than it saves (critique of the RO design). |
| **T** (slow trend) | Fixed-point linear slopes in the ≤ 256 KB tier-1, at scales above about 3k keys. Curvature is off. Piecewise-exponential fitting with PELT and l1 trend filtering is kept only as an optional knot initializer, accepted only if it wins the count. | The exp/line segment ratio is already 0.792 on pure Poisson data, so the extra parameter mostly fits noise. Real curvature gains are 4-15%, and only on genome, fb, osm, books-w and planet-u. A mass-fitted cubic raised the fb-window max error from 2,093 to 11,728. Every transcendental warp measured lost 7-14% of throughput. |
| **V** (1-64-key clustering, collisions) | DIRECT: slack α_j and fetch width w_j. FENCED: equi-depth u16 fences (deterministic virtual boundaries), entry width and capacity, and child entries. Replaces CSV's greedy. | The fine log-gap residual holds 67-99% of variance everywhere. Changing tier-1 resolution leaves the over-capacity share unchanged, so T and V act at disjoint scales. CSV's greedy cost O(α·n²), drifted by ±0.36 probes under perturbation, and took 2,068 s at the planet root. |
| **Writes (later)** | V becomes reservations inside the same pages: one spare slot per line plus one overflow line per entry. The **elastic density (piecewise exponential with l1 trend filtering, by likelihood) becomes the insert forecast** that sizes those reservations. Add local re-selection when a segment's counted D grows by at least 0.3, and a delta/LSM buffer for distribution shift (ALEX lost 52% going covid → osm). | This is the only place where likelihood is the right signal: forecasting where inserts will land. |

---

## 5. Per-lookup budget for fb at 200M (4 KiB pages, no cross-lookup overlap)

| Step | ns | Basis |
|---|---|---|
| GRE loop, virtual get, exception check | 5-15 | benchmark.h:365-406 (light loop, latency_sample off) |
| Router L1 load, then tier-1 record from L2 | 10-13 | L1 3 cycles, L2 17-19 cycles (Agner, 7-cpu); about 0 D of misses at about 43 KB of tables |
| Integer prediction | 5-7 | MUL 6 cycles plus shifts (Agner) |
| Directory walk (0.81 D + about 10) | 91-107 | adv/pte.cpp, directory PTE hit 0.19 |
| Entry: 2 lines issued together (1.0-1.1 D) | 100-132 | D; 2 of 8 WCBs (Intel Table 6-7) |
| Branch-free 56-fence count | 8-12 | pcmpgtw, pmovmskb, POPCNT 3 cycles (Agner) |
| Data walk (0.94 D + about 10) | 104-123 | data PTE hit 0.06 |
| Data line | 100-120 | D |
| Compare and select | 6-10 | SSE compare, L1 load |
| Over-capacity child entry, 0.134 × about 1.9 D | 25-31 | advsim, C = 228 |
| **Subtotal** | **454-570** | |
| Residual not explained by the competitor fits | 0-100 | critique; must be measured |
| **Total** | **454-670 ns, i.e. 1.5-2.2 Mops/s** | about 1.3-1.9x PGM's measured 852 ns |

**Variants of the fb estimate:**
- Up to about 0.9 D of credit from overlapping consecutive lookups would raise the ceiling to about 2.6 Mops/s. It is not counted, because Goldmont's walker concurrency and its re-issue of uTLB-miss loads at retirement are unverified.
- With 2 MiB pages, both walks drop to 10-30 ns: about 270-490 ns, 2.0-3.7 Mops/s. PGM also gains about 1 D under THP (unmeasured).
- Memory: fb fast 22.4 B/key, compact 19.3 B/key (cbar 24, 2-line entries).

**Other datasets.** Formula: ns = E[D] × (100-120) + 40-70 of compute, plus a 0-100 residual. These are absolute predictions only, because PGM was measured on DIAS for fb alone.

| Dataset | Expected mode (counts) | E[D], 4 KiB | Mops/s | PGM L0 |
|---|---|---|---|---|
| stack | DIRECT, α 0.25, 1.04 lines | 2.2-2.4 | 2.2-3.8 | 0.87 MB, cached (PGM at its strongest) |
| wise | DIRECT, 1.31-1.64 lines | 2.3-2.7 | 2.0-3.7 | 3.9 MB, near-cached |
| history | DIRECT, 1.46-1.91 lines | 2.4-2.8 | 2.0-3.6 | 4.9 MB, near-cached |
| covid | DIRECT, 1.51-1.94 lines | 2.5-2.8 | 2.0-3.4 | 3.7 MB, near-cached |
| libio | 70-87% DIRECT (2.51-2.83 D before tier-1 misses) | 2.6-3.0 | 1.9-3.3 | 4.7 MB, near-cached |
| books | 47-59% DIRECT (3.09-3.32 D); FENCED part has 0 over-capacity, fast format only (compact coverage 5%) | 3.1-3.5 | 1.7-2.9 | 12.3 MB, misses |
| planet | FENCED, over-capacity 0.074 | 3.9-4.0 | 1.5-2.3 | 22.7 MB, misses |
| genome | FENCED, over-capacity 0.0009; needs 2-line entries (1-line gives 0.215) | about 3.8 | 1.6-2.4 | 50.5 MB, misses |
| fb | FENCED, over-capacity 0.134, plus 21 exceptions | 4.0-4.1 | 1.5-2.2 | 33.9 MB, misses |
| osm | FENCED, over-capacity 0.191; router depth 1-2 (+9-18 ns) | about 4.2 | 1.45-2.1 | 21.6 MB, misses |

---

## 6. Predicted ranking against the measured baselines (fb, DIAS, 4 KiB)

**SPLICE-H 1.5-2.2** > PGM 1.17 ≈ LIPP 1.11 (6% gap, inside the 4.9-6.1% round spread) > ART 1.00 > ALEX 0.885 > B+tree 0.747 > sorted array 0.457 ≥ SCALE-LI (expected 0.3-0.45).

| Comparison | Confidence | Basis |
|---|---|---|
| SPLICE-H beats PGM by more than 15% on fb | **Likely** | Fails only if D ≈ 170 ns and the residual is about 250 ns at the same time, which gives about 1.2, a tie. |
| SPLICE-H above LIPP, ART, ALEX, B+tree on fb | Likely | Follows from the PGM comparison. |
| Order among the competitors | Measured | PGM vs LIPP is not resolved. |
| Margins on the other nine datasets | **Unknown** | Expect larger margins on the five where PGM's L0 misses cache, and the smallest on stack. |

---

## 7. What is new, against prior art

**Not new; cite these.**

| Mechanism | Prior work |
|---|---|
| Model-placed slots | LIPP (https://arxiv.org/abs/2104.05520), DILI (https://arxiv.org/abs/2304.08817) |
| Order-preserving learned hashing and collision theory | Sabek et al. (https://www.vldb.org/pvldb/vol16/p532-sabek.pdf), LeMonHash (https://arxiv.org/abs/2304.11012) |
| 64-B metadata line over data blocks, latency cost model | CARMI (https://arxiv.org/abs/2103.00858) |
| Count-split radix | Hist-Tree (https://www.vldb.org/cidrdb/2021/hist-tree-those-who-ignore-it-are-doomed-to-learn.html) |
| Radix plus spline | RadixSpline (https://arxiv.org/abs/2004.14541), PLEX |
| ε-PLA under a space budget | PGM multicriteria (https://pgm.di.unipi.it/) |
| Segment-to-page layout | FITing-tree (cited from memory, not checked) |
| Correction layer | Shift-Table (https://arxiv.org/abs/2101.10457) |
| Cost-driven knob and structure search | AirIndex, PGM++ (https://arxiv.org/abs/2410.00846), CAM |
| Priced fit with reserved gaps | Li et al. MDL (https://arxiv.org/abs/2101.00808) |
| Error-aligned loss | LER |
| Collision bound | ICDT 2025 Rényi-2 bound |
| floor(L·F) dispatch | RMI |

**Defensible claim, which is narrow.** Each segment chooses between model-placed slack and rank fences. Knot budget, mode, slack, fetch width and entry geometry are chosen by an **exact full-scale count of dependent DRAM lines and page walks**, with metadata priced by a **shared-L2 residency simulation**, on an integer-only lookup path.

**What the evidence does not support.** The read-only evidence does **not** support "one jointly learned G+T+V CDF" as the source of the gain. A Hist-Tree router with no model, over the same directory, is the ablation that decides whether any learned part earns its place.

---

## 8. Implementation plan

Nothing is edited until Louis approves. All paths are under `experimental/scaleli/`.

1. **`tools/splice_count.cpp`, written first.** Merge scratchpad `mlp/fullsim.cpp`, `impl/layout.cpp` (DIRECT and fence modes only), `adv/advsim.cpp`, `adv/pte.cpp` and `review/l2sim2.cpp`. Give it a shared-L2 LRU at 2 MB and 1 MB, the fixed-point outer loop, and JSON output. Run it on all 10 sorted files. Its output, under `results/splice_h/`, is the paper's count table and the target for count parity.
2. **`include/splice/layout.hpp`, C++17.** POD arrays (Exceptions, Router, Seg16, Entry128, Line, Slot) and an inline `noexcept` `get()`. No repo headers: all `scaleli/*.hpp` are C++20 and cannot enter GRE's C++17 TU.
3. **`include/splice/build.hpp` and `cost.hpp`, C++20.**
   - Reuse `hardness.hpp`'s OptimalPLA (lines 230-311), adding an emit-segments variant next to `pla_segments()`.
   - Reuse `parallel_sort` and the 16-thread pattern.
   - `cost.hpp` is shared with `splice_count.cpp` so the build and the offline numbers cannot diverge.
   - **Do not reuse** `index.hpp`'s region/block path, `codec.hpp`'s `key_at`, `model.hpp`'s `std::fma`, `joint.hpp`'s tanh warp and gap table, or `smoothing.hpp`.
4. **Parameters.** An own `SpliceParams`, parsed from env `SPLICE_ARGS`, the way `gre_run.sh` passes `SCALELI_FLOW`. Never touch `Config`: `static_assert(sizeof(Config)==152)` is at index.hpp:90.
5. **GRE entry.** Add `integrations/gre_splice/`:
   - `splice_gre.cpp` (C++20 TU) builds and returns the POD through an opaque handle.
   - `splice_interface.h` (C++17) includes `layout.hpp` so `get()` inlines.
   - `thread_num` must be 1. `put`/`update` return false for now.
   - In `gre_lite.sh`, register `splice` and `splice_thp`, and retire the `scaleli_splice → Jg0` alias (gre_lite.sh:131).
   - Add the names to `gre_run.sh` and `gre_report.py`, with a memory check of 18-25 B/key.
6. **Warm-up hook.** No index-specific hook is needed: there is no state on the read side. `bulk_load` pre-touches every arena page and reads the router and tier-1 lines once. Use GRE's `--warmup_num 20M --pin_core` on a core whose module sibling is idle, since L2 is probably shared per core pair.
7. **Tests, `tests/test_splice.cpp`.**
   - Every stored key in the 20 2M samples returns its payload. 1M absent keys return false.
   - n ∈ {0, 1, 2, 3, 4, 5, 56, 57, 228, 229, 13,224, 13,225}.
   - Keys 0 and UINT64_MAX; the fb outliers spliced into a window; osm holes; a run of 10^6 stack keys.
   - Line span ≥ 2^32 (shifted fences, and the previous-line fallback on ties).
   - The 2-level inner path.
   - Duplicate keys throw.
   - Count parity, and a Mac/x86 byte-identical build.

---

## 9. Measurements that confirm or kill it

All of these are run by Louis on DIAS. The thresholds below are decision rules, not predictions.

1. **THP state.** Read `/sys/kernel/mm/transparent_hugepage/{enabled,defrag}`, jemalloc's `opt.thp`, and `AnonHugePages`. This decides which arm is primary.
2. **Pointer chase over 4 GB, on 4 KiB pages and on THP**, giving D and the walk cost.
   - **Kill** if D ≥ about 170 ns and the residual from item 5 is ≥ about 250 ns: fb then ties PGM.
3. **k adjacent independent lines, k = 1..10.**
   - If 2 lines cost about 2 D, switch to 1-line entries. fb's over-capacity share then rises to 0.408: recount before going further.
4. **lfence A/B on the get() loop** measures real cross-lookup overlap. Report the fenced variant as SOSD recommends.
5. **GRE with an O(1) fake index** measures harness overhead. That is the residual term.
6. **perf counters** (L2 misses, dTLB walks per lookup) on fb and stack. Expect about 2 walks plus about 2.13 DRAM lines on fb, and about 1 walk plus about 1.04 lines on stack. A deviation above 0.3 D means the cost model is invalid.
7. **PGM and LIPP on all 10 datasets** before any per-dataset ratio is stated.
8. **GRE runs.** 20M warm-up, 100M lookups, ≥ 3 interleaved rounds, gre_report.py intervals. **Adopt only with a margin above 15%.**
   - **Kill the read-only claim** if fb measures below 1.35 Mops/s on 4 KiB pages.
9. **Ablations:**
   - A Hist-Tree router with no model over the same directory. **Kill the "learned" claim** if it is within 0.01 lines or within noise.
   - Static PGM with a parallel last-mile window.
   - DIRECT-only and FENCED-only.
   - K1 at 16k vs 4k.
   - fb with and without its 21 outliers.

---

## 10. Open uncertainties

- **L2 size.** Whether L2 is 2 MB per core pair (likely; WikiChip was unreachable) or effectively 1 MB. At 1 MB, tier-1 misses rise from 0.155 to 0.42 D. Goldmont's replacement policy is unknown.
- **DRAM latency.** No published DRAM or L2 latency for the C3958. D is back-solved from three throughput numbers.
- **Overlap.** Page-walker concurrency, re-issue of uTLB-miss loads at retirement, and whether GCC 15 keeps the lookup near 95-105 µops against the 78-entry ROB. Together these decide the 0 to 0.9 D overlap credit.
- **Unexplained 250-300 ns.** The competitor fits leave this term unexplained. It may also apply to SPLICE-H, which is why the 0-100 ns residual is carried.
- **Memory.** 19-22.5 B/key fast against PGM's 16.18. Compact gets under 16.2 only on stack, wise, history and libio. Compact covers 55% of lines on osm and 5% on books.
- **Unproven at full scale.** The mixed-mode results for books and libio are re-runs without the dropped RADIX mode, before tier-1 misses. No full end-to-end count of the merged design exists yet, which is step 1 of the plan.
- **RO alternative.** The rank-addressed packed-record design (16.1-16.4 B/key) stays a candidate arm only if item 3 shows batches of 5-8 lines cost about 1 D plus 3.3 ns per line. As specified, it had record overflow, a residency problem on fb and an infeasible case on osm.
- **Writes are not designed.** GRE's losing corner is hard data with at least 50% writes, and easy-to-hard shift.

Scratchpad sources: `/private/tmp/claude-501/-Users-louisvasseur-Downloads-scaleli-sota/ac6d4251-6384-4db6-8aba-a17e1917a2ab/scratchpad/{mlp,adv,impl,review,review_m,review_ro,memfirst,splice_wk,paged,loggap,proxy,gm_model}/`. Nothing in the repository was modified, and DIAS was not contacted.