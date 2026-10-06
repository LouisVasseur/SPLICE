# SPLICE-H: a count-optimised read-only learned index

Design and decision document, 2026-10-06. It describes what `include/splice/` implements, why each choice was
made, what evidence backs it, and which measurements would kill it.

**Sources and conventions**
- The design is the synthesis of a 15-agent design workflow (four designs, four adversarial critiques):
  `results/splice_h/design_evidence/splice_synthesis.md`. The prototype sources and outputs it cites are copied under
  `results/splice_h/design_evidence/` (its README names each file's origin). Every repository path below is relative
  to `experimental/scaleli/`.
- **D** is one serialized DRAM access, the cost currency (section 2). "count" means an exact count from a dry run or a
  simulation (not a timing); "measured" means a timing on DIAS (Intel Atom C3958, Goldmont); "estimate" means a
  value derived from calibrated constants and not measured. All Mops/s and ns figures for SPLICE-H are estimates
  until SERVER.md section 5b has run.
- Status labels in the ledger (section 6): **count (repo)** = reproducible from a repository file; **measured** =
  DIAS timing; **paper, verified** = checked against the paper and code at a cited commit
  (`design_evidence/warmup_verified.json`); **paper, cited** = cited by the design workflow, not re-read for this
  document; **unverified** = recollection, to be checked before publication.

---

## 1. What SPLICE-H is, and what it is not

SPLICE-H is a read-only in-memory index for sorted unique uint64 keys with uint64 payloads. Each segment of the key
space is stored in one of two modes: **DIRECT**, model-placed slots read with a window of 1, 2 or 4 adjacent lines,
for Poisson-like or near-linear data; or **FENCED**, a model-addressed 64-B or 128-B directory entry whose 24 or 56
quantised fences give the exact data line, for clustered data. A cache-resident integer top (an exception list, a
count-split radix router and one merged 16-B record per segment) sits above both. Every choice (knot budget, mode,
slack, fetch width, entry geometry) is made by an **exact full-scale count of dependent DRAM lines and page walks
per lookup**, with the hot metadata priced by a **shared-L2 residency simulation**, and the lookup path is
integer-only, so the layout is bit-identical on arm64 and x86.

**What it is not.**
- It is not SCALE-LI and not any SCALE-LI cell. In GRE, `scaleli_splice` (an alias of `scaleli_jg0`) is SCALE-LI's
  joint G+T+V root cell Jg0 (`integrations/gre/README.md`), kept under that name because earlier server runs use it.
  The SPLICE-H index is the GRE index `splice` (and `splice_thp`).
- It does not claim one jointly learned G+T+V CDF (section 7).
- It does not support writes yet; `put`, `update` and `remove` return false in GRE (write plan: section 10).

## 2. Architecture

### 2.1 Components (implemented bit layouts)

| Component | Implemented layout (`include/splice/layout.hpp`, C++17, little-endian, static_asserted) | Residency | Why |
|---|---|---|---|
| **E. Exception list** | `Exc {u64 key, val}`, sorted, at most 2 x 64 entries. Up to `exc` (64) keys peeled at each end so the remaining span is within 2x of the 64/64-trimmed span; reached through one predictable range check `x - lo > span`. | 1-2 L1 lines | fb: exactly 21 outliers, including 2^64-1; without them all 200M fb keys share one top radix bucket (`design_evidence/splice_wk/radix.json`). No key value is a sentinel. |
| **R. Router** | `rtop[2^R]` u32 (R = 11): bit31 = child, else leaf `cnt:4 | j0:27`. Child nodes `RNode {u64 lo; u32 off; u8 shift, bits}` over `rsub[]`, created where a bucket holds more than L = 8 knots, recursively (no depth cap: each level consumes at least one key bit, and a width-1 bucket holds one knot). Counted on osm at K1 16384: 0/1/2/3/>=4 child levels for 4.7/75.3/18.6/1.3/0.05% of keys (`results/splice_h/osm.json` "fits"); at most 3 levels for every chosen cell. Ends in a branch-free masked scan of up to 8 records (pcmpgtq on SSE4.2, scalar elsewhere, same result). | top 8 KiB in L1, children in L2 | Raw 2^12 radix is 36.5x the ideal bucket on osm (`design_evidence/review_m/spans.json`). Shifts only: no BSR/LZCNT (Goldmont has no LZCNT; BSR is slow). |
| **K. Tier-1 records** | `Seg16 {u64 key; u32 m; u32 meta}`, meta = `base:24` (region start in 512-B units, 8 GiB reach) `| pre:6 | kind:2` (0 = FENCED, 1/2/3 = DIRECT with w = 1/2/4). `seg[K]` is a sentinel holding the end of section A; 8 zero records pad the router scan. | K1 <= 16k records, 256 KB, L2 | Merged 16-B records at K1 16k miss 0.155 D per lookup with a 2 MB L2 and 0.42 D with 1 MB; separate key and record arrays, or K1 32k, add 0.33-0.66 D (`design_evidence/review/l2sim2.cpp`). |
| **Prediction** | `u = min((x - key) >> pre, 2^32-1)`; `p = (u * m) >> 32`: one 64-bit IMUL, no 128-bit MUL. | registers | No libm, FMA, DIV, x87 or u64-to-double: bit-identical on Mac and Atom. `pre` keeps u < 2^32 (books spans 2^63; five fb keys are >= 2^63). |
| **DIRECT region** | `Line {u64 k[4]; u64 v[4]}` slots, `R_j = roundup8(max(ceil(U/4), last+1) + w - 1)` lines. Order-preserving placement (forward cumulative max); gaps copy their right neighbour. Lookup: window line `h = min(p >> 2, R - w)`, all w lines compared with pcmpeqq/movmskpd with no branch between them, then a right scan inside the region in the rare overflow case. | arena | stack 1.039 lines at alpha 0.25 (`design_evidence/impl/stack.json`); windows of w <= 4 give 1.000-1.034 (`design_evidence/memfirst/final_w4.jsonl`). |
| **FENCED region** | Top-level entries `Entry128 {EntryHdr; i16 f[56]}` (2 lines, 128-B aligned, up to 57 data lines = 228 keys) or `Entry64 {EntryHdr; i16 f[24]}` (1 line, 25 lines = 100 keys). `EntryHdr {u64 base; u32 line_off; u8 shift, flags; u16 nlines}`. Fence count is branch-free over all fences (pcmpgtw, packsswb, pmovmskb, popcnt). Over-capacity entries become inner entries with up to F+1 children (two levels reach 57 x 228 = 12,996 keys per inner entry with 2-line entries). Ties fall back to the previous line/child. Data lines: fast `Line` (4 keys) or, when the entry spans < 2^32, compact `CLine {u32 r[5]; u32 pad; u64 v[5]}` (5 keys). | arena | At C = 228 the share of keys in over-capacity entries is fb 0.127 and osm 0.096 counted with the implemented eps-PLA tier-1 (K1 16384, FENCED only; `results/splice_h/COUNTS.md`), against 0.134 and 0.191 in the radix-trie prototype (`design_evidence/adv/advsim.cpp`); 56% of fb keys sit in entries of more than 17 lines, so the fence count must not branch. |
| **Arena** | One `mmap`, 2 MiB aligned and rounded, below 8 GiB. Line 0 is the empty line (copies of the first key). Section A: per segment, the DIRECT slot lines or the FENCED top-level entries, each region 512-B aligned. Section B: FENCED child entries and data lines, depth-first. `splice_thp`: `madvise(MADV_HUGEPAGE)` before the fill. | 3.8-4.8 GB (estimate) | No per-region heap objects, no bitmaps, no stores on the read path; those cost LIPP, ALEX and SCALE-LI one dependent miss each. |

### 2.2 Lookup path

```
 x ──► x - lo > span ? ──yes──► binary search in Exc[] (fb: 21 keys)                      L1
        │no
        ▼
 rtop[(x-lo) >> rshift] ──child──► RNode / rsub (0-3 levels) ──► leaf (j0, cnt)               L1 / L2
        ▼
 masked scan of seg[j0+1 .. j0+8] keys (<= x) ──► j;  seg[j], seg[j+1].base (region end)       L2
        ▼
 u = min((x - key) >> pre, 2^32-1);  p = (u * m) >> 32                                      registers
        ▼
 kind > 0 (DIRECT, w lines)                     kind == 0 (FENCED)
   h = min(p >> 2, R - w)                         e = min(p, Q - 1)
   load lines h .. h+w-1 together  ── 1 D ──      load entry e (1 or 2 lines together)  ── walk + 1 D ──
   pcmpeqq / movmskpd: hit -> payload             fence count -> child (inner, 0-19% of keys) or data line
   miss and k[3] < x: scan right (rare)           load data line  ── walk + 1 D ──; compare (fast or compact)
```

The page walk (4 KiB pages) is charged per distinct page per lookup. Common path on 4 KiB pages, estimate: **DIRECT
about 2.2-2.4 D; FENCED about 3.75 D** (0.81 D directory walk + 1 D entry + 0.94 D data walk + 1 D data line).

### 2.3 Cost currency and the fb budget

**D = 100-120 ns, best estimate 105-110.** The measured GRE sorted array (0.457 Mops/s, about 2.18 us per lookup on
DIAS) needs about 18.3 serialized DRAM accesses per lookup in a Goldmont cache and TLB model
(`design_evidence/gm_model/bsearch_sim.py`), which caps D near 119 ns; the 130-170 ns used in some early designs
contradicts it. L2 hit 17-19 cycles (about 9 ns), L1 3 cycles, 2.0 GHz (Agner Fog's tables; Intel optimization
manual vol. 2; 7-cpu.com/cpu/Goldmont.html; paper, cited). A walk into a multi-GB array on 4 KiB pages costs 0.8-0.95
D (`design_evidence/adv/pte.cpp`: directory PTE hit 0.19, data PTE hit 0.06); on 2 MiB pages 10-30 ns (estimate).

Per-lookup budget for fb at 200M, 4 KiB pages, no overlap between lookups. **All rows are estimates.**

| Step | ns | Basis |
|---|---|---|
| GRE loop, virtual get, exception check | 5-15 | GRE benchmark.h timed loop (latency sampling off) |
| Router L1 load, then tier-1 record from L2 | 10-13 | L1 3 cycles, L2 17-19 cycles |
| Integer prediction | 5-7 | IMUL plus shifts |
| Directory walk (0.81 D + about 10) | 91-107 | `design_evidence/adv/pte.cpp` |
| Entry: 2 lines issued together (1.0-1.1 D) | 100-132 | D; 2 of 8 write-combining/fill buffers |
| Branch-free 56-fence count | 8-12 | pcmpgtw, pmovmskb, POPCNT 3 cycles |
| Data walk (0.94 D + about 10) | 104-123 | `design_evidence/adv/pte.cpp` |
| Data line | 100-120 | D |
| Compare and select | 6-10 | SSE compare |
| Over-capacity child entry, 0.134 x about 1.9 D | 25-31 | `design_evidence/adv/advsim.cpp`, C = 228 |
| **Subtotal** | **454-570** | |
| Residual not explained by the competitor fits | 0-100 | must be measured (O(1) fake index, section 8) |
| **Total** | **454-670 ns, 1.5-2.2 Mops/s** | about 1.3-1.9x PGM's measured 852 ns on fb |

With 2 MiB pages both walks drop to 10-30 ns: about 270-490 ns (estimate; PGM also gains about 1 D under THP,
unmeasured).

## 3. Learning signal

For a parameter set theta (K1 and eps, mode per segment, slack alpha_j, window w_j, entry width, cbar, compact):

```
J(theta) = E over stored keys of [ D * (N_dep_lines + P_walk_miss + P_table_miss) + tau * N_par_lines + c_mode ]
           + lambda * bytes/key
```

- **Exact terms** (no cache state): dependent and parallel lines, fence depth, tie steps, the over-capacity share,
  distinct 4 KiB and 2 MiB pages per lookup. They are counted in one pass over the real integer layout for every
  stored key. GRE looks up stored keys uniformly, so this count is the workload's expectation: no generalisation
  gap. `get_impl<Count>` must reproduce the plan walker's totals exactly (parity, section 4.4).
- **Residency terms**: the page-walk miss probability per region (P_Ad DIRECT pages, P_Ae entry pages, P_B section-B
  pages) and the router, record and exception miss rates, from **one shared 16-way LRU simulation of L2** that
  replays 3M sampled lookups (after 1M of warm-up) with every hot table, the GRE operation stream, the cold arena lines
  and the PTE/PDE lines in program order, at 2 MB and 1 MB. Independent per-table step charges underestimated E[D] by
  0.3-0.8 D in the critiques (`design_evidence/review/l2sim2.cpp`).
- **Constants** (`splice/params.hpp`, milli-D): tau 50 per extra parallel line (a 2-line entry costs 1.05 D, a
  4-line window 1.15 D, inside the 1.0-1.1 and 1.1-1.3 budgets), walk-hit 90 (THP 150), router child level 80, compute
  300 (DIRECT) / 450 (FENCED) / 200 (exception), initial P_Ad / P_Ae / P_B 940 / 810 / 940.
- **Integer arithmetic.** J, lambda and every probability are integers in milli-D, with an integer lambda bisection,
  so the selection does not depend on floating-point summation order, thread count or architecture, and the layout
  hash can be compared strictly across arm64 and x86 (deviation 12).

**Why counts and not error proxies.**
- SOSD: cache misses explain lookup time (R^2 = 0.955), and once misses are in the model, log2 error is not
  significant (Marcus et al., VLDB 2020, arXiv 2006.12804; paper, cited).
- On a model-placed layout every error proxy mis-ranks models: mean Spearman 0.75 for log2 error, 0.77 for MSE and
  0.38 for LIPP's conflict degree, and log2 error falls to 0.07 on history; NFL's tail conflict is constant (ranks
  nothing) on 6 of 10 datasets (`design_evidence/proxy/surr.json`, recomputed by `proxy/spearman.py`; the synthesis'
  0.33 and -0.16 for the last two could not be reproduced and are not used).
  For plain sorted-array search, log2 error does rank models like the lines touched (Spearman 1.00), which is why
  the proxies looked adequate in earlier work.
- NLL moved 0.003 nats while lines went from 9.9 to 3.2 on books (`design_evidence/proxy/proxy.json`, Qknot64 vs
  Qknot4096).
- The repository's own joint loss, flow loss and root-probe currency all decoupled from cost
  (`results/aidb_joint`, `results/aidb_threeblock`).

## 4. Learning protocol

### 4.1 Data and fit order
- All 200M keys, read from the sorted files (`data/external/gre/<ds>` for fb, osm, books; `<ds>.sorted` for covid,
  genome, history, libio, planet, stack, wise; the raw GRE files of those seven have about 50% descents). Never the
  2M samples: windows misrepresent clustering (books' window extrapolation was off by 2.2x in the design study), and
  the selection must count the arena the lookups will touch. The design also cited osm's over-capacity share, 0.097
  in windows against 0.191 at 200M, but that 0.191 came from the radix-trie prototype; the implemented eps-PLA tier-1
  counts 0.096 at 200M (same geometry), so osm is not evidence for this point. Samples are for unit tests only.
- Deterministic fit, once: exceptions (rule of `design_evidence/mlp/fullsim.cpp`) -> tier-1 optimal eps-PLA (the
  repository's Apache-2.0 port `scaleli::hardness::OptimalPLA<unsigned __int128>`, used unmodified) with eps
  bisected in integers until the segment count fits K1 -> count-split router over the knots -> per-segment candidate
  evaluation -> multiple-choice selection with a Lagrangian on bytes -> outer fixed point on residency.
- No G/T/V alternation (section 5): joint alternation gained at most 0.072 root probes on the 2M samples, and one more
  round would cost about 2,000 s at the 200M root (an extrapolation in `results/aidb_joint/workflow_result.txt`,
  bracketing the 2,068 s measured for the CSV planet root).

### 4.2 Selection
- Per segment, candidates DIRECT (alpha in {0.25, 0.5}) x (w in {1, 2, 4}) and FENCED; one multiple-choice knapsack
  per segment, cost `sum_keys J + lambda * bytes`, lower candidate index wins ties.
- The bytes cap is 22.0 B/key (fast arm) or 16.5 B/key (compact arm) on the index's `total_bytes`: the arena
  rounded up to its 2 MiB mapping, plus records, router and exceptions (the selection computes exactly what
  `make_plan` allocates, and `make_plan` checks the two are equal). lambda is the smallest integer meeting it
  (cap_violated is reported if none does by 2^20).
- Outer fixed point: select with the current residency constants, plan, simulate L2, re-select. A new selection is
  accepted only if E[D] drops by more than 10 milli (0.01 lines per lookup); at most 4 rounds.

### 4.3 The pruned grid (`tools/splice_count.cpp`)
- Stage 1 (count only, default residency constants): K1 in {4096, 8192, 16384} (3 concurrent eps bisections), one
  DIRECT pass per K1, one FENCED pass per (K1, entry in {1, 2}, cbar in {16, 24, 32}), each yielding fast and
  compact counts: 36 cells, each scored for both arms.
- Stage 2: residency fixed point for the baseline cells (K1 16384, entry 2, cbar 24; fast lines for the fast arm,
  compact for the compact arm) and the 2 best stage-1 cells per arm, with 4 reporting simulations ({4 KiB, THP} x
  {2048, 1024} KB). The arms share these finalists: each also runs under the other arm's cap when stage 1 found it
  feasible there (at most 12 fixed points). Stage-1 E[D] uses default residency constants and can misrank cells by
  about 0.4 D (1-line entries above all), so a cell that fits 16.5 B/key must also compete in the fast arm: without
  this, genome and planet chose a fast cell that was worse on both E[D] and bytes than their compact cell.
- Stage 3, ablations at the chosen fast cell: DIRECT only, FENCED only, Hist-Tree knots (same per-leaf linear model),
  exc=0; plus the synthesis' own cell (K1 16384, 2-line entries, cbar 24, fast lines, FENCED only, uncapped) counted
  like for like with its predictions (`results/splice_h/COUNTS.md`).
- **Acceptance margins**: a grid cell replaces the baseline only if its E[D] is at least 0.3 D (300 milli) lower
  and it is feasible; never a strict '<' (that rule turned the fb 10M tie into a loss in SCALE-LI's history).
- Budget: about 42 PLA passes, 21 candidate passes, at most about 50 plan walks and 50 LRU simulations of 4M
  lookups per dataset; target 10 minutes per dataset on the Mac with 16 threads (wall time per stage is in the
  JSON), against about 10^2 cells x 9 s unpruned and the 3,000-4,500 CPU-s estimated for SPLICE-M's builder.

### 4.4 Validation (no hold-out is needed for a read-only index over its own keys)
1. **Parity**: `get_impl<Count>` over all keys equals the plan walker's counts exactly (lines, depth, ties, 4 KiB
   and 2 MiB pages) on fb, osm and stack.
2. **Verify**: all 200M keys return their payloads; 10M absent keys return false.
3. **Perturbation**: `t' = (t + 1e-6 t^2) / (1 + 1e-6)` on the normalised key t (ends fixed; deviation 17), the
   whole pruned grid rerun on all 200M perturbed keys (`splice_count --perturb`): the chosen fast cell must be the
   same and E[D] must move by at most 10 milli. Results per dataset: `results/splice_h/COUNTS.md`, "Stability".
4. **Cross-architecture**: arm64 and x86 (`clang++ -arch x86_64 -march=goldmont`, Rosetta) give the same layout
   hash on the 20 samples and on full fb and osm (`tools/splice_xarch.sh`).
5. **GRE identity**: the GRE build of the chosen cell prints `splice_layout_hash` equal to `results/splice_h/<ds>.json`
   (`gre_report.py --splice-json`); the hash skips payload words, so GRE's constant payload does not matter.
6. **Fairness of get()**: `tools/splice_asm_check.sh` rejects call, div, int-to-float conversions, x87, FMA, BSF/BSR,
   LZCNT/TZCNT and string stores in the compiled lookup, also behind a `rep`, `lock` or `notrack` prefix (clang writes
   TZCNT as `rep bsf` for targets without BMI1), any jump out of the function, and any indirect jump that is not a
   local jump table (the table must be referenced in the same basic block; clang spells it `jmpq *%reg`, GCC
   `jmp *%reg` or `notrack jmp`). Stores pass only if their base is the stack (%rsp, or %rbp when a frame pointer is
   set up) or a register that holds the out parameter on every path to the store: a must-dataflow over the function's
   CFG (jump-table edges included) that follows the pointer through copies, spills and reloads. The checker fails on
   injected stores through the arena, the View, a global and a pointer that is the out parameter on one path only, on
   an indirect call, a tail call and TZCNT (self-test in the review notes, section 11). The one indirect jump in the
   compiled get() is the jump table of the segment-kind switch; the only non-stack store is the payload.

### 4.5 Build in GRE and fairness of the timed region
GRE bulk-loads its sorted, unique pairs; the facade builds over them in place (stride 2, no copy, no sort). With
the per-dataset args file the cell and eps are fixed, so bulk_load runs one PLA pass, the candidate passes, at most
4 plan walks and simulations, and one fill; its time is `splice_bulk_load_ns` (target <= 2x PGM's build time, not a
gate). **The build is multi-threaded and the competitors' are not:** splice builds on `SPLICE_BUILD_THREADS`
(gre_run.sh `--build-threads`, default 16, as for the SCALE-LI cells), while gre_run.sh sets `OMP_NUM_THREADS=1`, so
PGM's and the others' bulk loads run on one thread. GRE's `build_ns` for splice also includes the layout-hash pass
(`splice_hash_ns`, excluded from `splice_bulk_load_ns`). A build-time comparison with PGM is fair only from a round
with `--build-threads 1` and the global `SPLICE_ARGS=hash=0`; the lookup numbers do not depend on either. **Pre-touch, documented:** the fill writes every arena byte (so every page is resident before the timed
region, as for any index whose bulk load writes its nodes), `splice_thp` calls `madvise` before writing, and the
build ends by reading the router and record lines once. SPLICE-H has no read-side state, so this is identical with
and without GRE's warm-up and moves no lookup work out of the timed region. `get()` performs no store except the
payload of a hit, no allocation and no cross-TU call (`integrations/gre_splice/README.md`).

## 5. G, T and V: three scales, one owner each

| Block | What it owns in SPLICE-H | Evidence that it stays complementary |
|---|---|---|
| **G** (gaps, outliers) | The exception list, eps-PLA knot breaks at holes (the hull places them), and the count-split router. No gap table. | A G table was chosen in 0 of 240 charged cells (`results/aidb_threeblock`); G-64 changes radix steps by at most 0.01 on 7 of 10 datasets, and its compressed coordinate costs more than it saves (critique of the RO design). |
| **T** (slow trend) | Fixed-point linear slopes in the 256-KB tier-1, at scales above about 3k keys. No curvature. | Piecewise-exponential vs linear eps-PLA gives 0.792 as many segments even on pure Poisson data (`design_evidence/loggap/seg_eps4.txt`: 20,603 vs 26,005), so the extra parameter mostly fits noise. Real curvature gains are 4-15%, only on genome, fb, osm, books-w and planet-u; every transcendental warp measured lost 7-14% of throughput. |
| **V** (1-64-key clustering) | DIRECT slack alpha_j and window w_j; FENCED equi-depth fences, entry width and capacity, child entries. | The fine log-gap residual holds 67-99% of the variance everywhere, and changing the radix trie's resolution left fb's over-capacity share almost unchanged (0.1245 with 1,166 leaves vs 0.1173 with 18,384). The tier-1 *model* does move it, though: at the same C = 228, eps-PLA knots give osm 0.096 against the radix trie's 0.191 (fb 0.127 vs 0.134), so T and V are not on fully disjoint scales on osm. CSV's greedy is O(alpha n^2), drifted by +/-0.36 probes under perturbation and took 2,068 s at the planet root. |
| **Writes (later)** | Reservations inside the same pages; the elastic density becomes the insert forecast only. | Section 10. |

## 6. Literature and evidence ledger

One row per decision. Repository paths are relative to `experimental/scaleli/`.

| Decision | Evidence | Source | Status |
|---|---|---|---|
| Cost currency D = 100-120 ns | Sorted array 2.18 us = about 18.3 serialized accesses; PGM 852 ns = about 8 | `results/splice_h/design_evidence/gm_model/bsearch_sim.py`; DIAS GRE baselines (fb) | count (repo) + measured |
| Page walk 0.81 / 0.94 D on 4 KiB pages | Directory PTE hit 0.19, data PTE hit 0.06 in a shared L2 LRU with the cold stream | `results/splice_h/design_evidence/adv/pte.cpp` | count (repo) |
| Merged 16-B records, K1 <= 16k | 0.155 D (2 MB L2) / 0.42 D (1 MB) of tier-1 misses; separate arrays or K1 32k +0.33-0.66 D | `results/splice_h/design_evidence/review/l2sim2.cpp` | count (repo) |
| Exception list (64 per end) | Raw radix puts all 200M fb keys in one bucket; fb has exactly 21 outliers | `results/splice_h/design_evidence/splice_wk/radix.json`, `mlp/fullsim.cpp` | count (repo) |
| Count-split router | osm raw 2^12 radix 36.5x ideal, 237 candidates; implemented router on osm at K1 16384: 1 child level for 75.3% of keys, 2 for 18.6%, 3+ for 1.4% (`results/splice_h/osm.json`) | `results/splice_h/design_evidence/review_m/spans.json`; Hist-Tree (Crotty, CIDR 2021, https://www.vldb.org/cidrdb/2021/hist-tree-those-who-ignore-it-are-doomed-to-learn.html) | count (repo); paper, cited |
| Optimal eps-PLA under a space budget | eps bisected to K1; eps at K1 about 32k: fb 873, osm 840 | `results/splice_h/design_evidence/impl/fb.json`, `osm.json`; PGM-index (Ferragina and Vinciguerra, VLDB 2020; multicriteria tuner); port `include/scaleli/hardness.hpp` | count (repo); paper, cited (`warmup_verified.json` checked only PGM's measurement protocol) |
| Integer predictor, no FMA/DIV/LZCNT | DIVSD 34 cycles, MUL r64 6, no FMA, no LZCNT/BMI1 on Goldmont; the CSV long-double divergence in this repository | Agner Fog instruction tables; Intel optimization manual vol. 2; `docs/MATCHING_NFL_CSV.md` | paper, cited |
| DIRECT slots, alpha and window w | stack 1.04 lines (alpha 0.25), wise 1.31-1.64, history 1.46-1.91, covid 1.51-1.94 sequential; w <= 4 windows 1.000-1.034 | `results/splice_h/design_evidence/impl/*.json`, `memfirst/final_w4.jsonl`; LIPP (arXiv 2104.05520), DILI (arXiv 2304.08817), Sabek et al. (PVLDB 16, p532), LeMonHash (arXiv 2304.11012) | count (repo); papers, cited |
| FENCED entries, C = 228, child entries | Prototype (radix-trie tier-1): over-capacity share fb 0.134, osm 0.191, planet 0.074, genome 0.0009; keys in entries > 68 keys: fb 0.559. Implemented (eps-PLA tier-1, same geometry): fb 0.127, osm 0.096, planet 0.047, genome 0.001 (`results/splice_h/<ds>.json`, ablation `synthesis_cell`) | `results/splice_h/design_evidence/adv/advsim.cpp`, `impl/f_*.json`, `review/fence_chk2.cpp`; CARMI (arXiv 2103.00858); FITing-tree | count (repo); CARMI cited; FITing-tree **unverified** (from memory) |
| Shared-L2 residency simulation | Independent per-table charges underestimate E[D] by 0.3-0.8 D | `results/splice_h/design_evidence/review/l2sim2.cpp`, `adv/pte.cpp` | count (repo) |
| Lagrangian multiple-choice selection | Cost-driven structure search | AirIndex; PGM++ (arXiv 2410.00846); CAM | papers, cited |
| No G/T/V alternation | Joint vs sequential at most 0.072 root probes (2M samples); about 2,000 s per extra round at the 200M root (extrapolated) | `results/aidb_joint/workflow_result.txt` | count (repo); the 2,000 s is an estimate |
| No gap table | G pays in 0 of 240 cells (RESULT.md section 3: 0/160 + 0/80), but only while the gap table is charged at least about half of ceil(log2(k+1)) root probes (RESULT.md item 4: at 0.25x and 0x it pays in 7 and 96 cells); G-64 changes radix steps by <= 0.01 on 7 of 10 datasets | `results/aidb_threeblock/RESULT.md`; design critique (`design_evidence/splice_synthesis.md` section 4) | count (repo) |
| Linear T only | exp/line segments 0.792 on Poisson; curvature gains 4-15%; warps lost 7-14% | `results/splice_h/design_evidence/loggap/seg_eps4.txt` | count (repo) |
| Margins 0.01 D, 0.3 D, 15% | fb 10M tie lost to a strict '<'; DIAS round spread 4.9-6.1% | SCALE-LI history (`results/aidb_*`); DIAS GRE baselines | measured |
| Full scale only | books window extrapolation off by 2.2x (synthesis section 3); the osm 0.097-vs-0.191 example compared a window with the radix-trie prototype, and the implementation counts 0.096 at 200M, so it no longer supports the rule | synthesis section 3; `results/splice_h/osm.json` | count (repo); the books figure is the synthesis' |
| Counts, not error proxies | SOSD R^2 0.955; proxy Spearman 0.75 (log2) / 0.77 (MSE) / 0.38 (LIPP conflict degree), 0.07 on history; NFL tail conflict constant on 6 of 10; NLL 0.003 nats vs lines 9.9 -> 3.2 | arXiv 2006.12804; `results/splice_h/design_evidence/proxy/surr.json`, `proxy/spearman.py`, `proxy/proxy.json` | paper, cited; count (repo) |
| Warm-up protocol | AIDB: 20M untimed then 100M timed, pinned; GRE's sampler depends on OMP_NUM_THREADS; SOSD cold 2-2.5x slower | `results/splice_h/design_evidence/warmup_verified.json` (12 papers and harnesses checked at cited commits) | paper, verified |
| Dispatch floor(L F) | RMI | Kraska et al., SIGMOD 2018 (arXiv 1712.01208) | paper, cited (only its measurement protocol was verified) |
| Radix plus spline | RadixSpline; PLEX | RadixSpline arXiv 2004.14541; PLEX cited by the design workflow | paper, cited (only RadixSpline's measurement protocol was verified) |
| Correction layer | Shift-Table | arXiv 2101.10457 | **unverified** (recollection) |
| Priced fit with reserved gaps | MDL learned index | Li et al., arXiv 2101.00808 | paper, cited |
| Error-aligned loss; collision bound | LER; ICDT 2025 Renyi-2 bound | design workflow | **unverified** |

Measured numbers are only the DIAS GRE baselines on fb (sorted array 0.457, PGM 1.17, LIPP 1.11, ART 1.00, ALEX
0.885, B+tree 0.747 Mops/s) and the timings in the cited repository results. Everything else above is a count or a
paper.

## 7. Novelty limits

**Not new; cite these.**

| Mechanism | Prior work |
|---|---|
| Model-placed slots | LIPP (arXiv 2104.05520), DILI (arXiv 2304.08817) |
| Order-preserving learned hashing, collision theory | Sabek et al. (PVLDB 16, p532), LeMonHash (arXiv 2304.11012) |
| 64-B metadata line over data blocks, latency cost model | CARMI (arXiv 2103.00858) |
| Count-split radix | Hist-Tree (CIDR 2021) |
| Radix plus spline | RadixSpline (arXiv 2004.14541), PLEX |
| eps-PLA under a space budget | PGM-index multicriteria (pgm.di.unipi.it) |
| Segment-to-page layout | FITing-tree (cited from memory, not checked) |
| Correction layer | Shift-Table (arXiv 2101.10457) |
| Cost-driven knob and structure search | AirIndex, PGM++ (arXiv 2410.00846), CAM |
| Priced fit with reserved gaps | Li et al. MDL (arXiv 2101.00808) |
| Error-aligned loss; collision bound | LER; ICDT 2025 Renyi-2 bound |
| floor(L F) dispatch | RMI |

**The defensible claim is narrow.** Each segment chooses between model-placed slack and rank fences, and the knot
budget, mode, slack, fetch width and entry geometry are chosen by an exact full-scale count of dependent DRAM lines
and page walks, on an integer-only lookup path. The shared-L2 residency simulation prices the hot metadata and
reports E[D] per arm, but on these data it did not change any per-segment selection: every chosen cell's fixed point
stopped after its first iterate (the re-selection with simulated residency was identical or within 0.01 D), and the
0.3-D rule applied to the stage-1 scorings alone (default residency constants, no simulation) picks the same fast
cell on 8 of 10 datasets and the same compact cell on every dataset that has one except genome. On genome and planet the stage-2 ranking (exact plan pages plus simulated
residency) moved the choice to a different cbar or entry width of the same K1. So the simulation is a pricing and
reporting term here, not a demonstrated source of layout gains; a claim that it improves layouts needs data where
it changes the selection.

**Not claimed.** The read-only evidence does not support "one jointly learned G+T+V CDF" as the source of any gain.
The `tier1=histtree` ablation (Hist-Tree count-split knot placement instead of eps-PLA knots; the router, the
directory and the per-leaf integer linear model are the same, so it is not model-free) decides whether the learned
knots earn their place: **kill the "learned" claim** if it is within 0.01 lines of the PLA tier-1, or within
measurement noise in GRE.

**Counted outcome (Mac, all 200M keys, `results/splice_h/COUNTS.md`, rule applied in lines as the synthesis states
it).** Hist-Tree knots minus PLA knots, dependent lines per lookup at the chosen fast cell: fb +0.000, genome +0.004,
books +0.001, covid +0.006, history +0.002, stack +0.001, wise -0.001 (within 0.01 lines: the claim is killed on these
7), osm +0.049, libio +0.027, planet +0.027. On those three the E[D] gap is +0.019, -0.018 and +0.030 D (0.4-0.7% of
the lookup; libio's E[D] even favours Hist-Tree), far inside the 4.9-6.1% round-to-round spread measured on DIAS,
so by the "or within noise" clause the claim is dead on all 10 datasets unless a GRE run resolves a sub-1% gap.
What the counts support is the selection itself (mode, slack, window, entry geometry by exact count), not a learned
CDF.

## 8. Predictions and kill criteria

Predicted ranking on fb (DIAS, 4 KiB pages), estimates: **SPLICE-H 1.5-2.2 Mops/s** > PGM 1.17 = LIPP 1.11 (6%
apart, inside the 4.9-6.1% round spread) > ART 1.00 > ALEX 0.885 > B+tree 0.747 > sorted array 0.457 >= SCALE-LI
(0.3-0.45 expected).

| Comparison | Confidence | Basis |
|---|---|---|
| SPLICE-H beats PGM by more than 15% on fb | Likely | Fails only if D is about 170 ns and the residual about 250 ns at the same time (gives about 1.2, a tie) |
| SPLICE-H above LIPP, ART, ALEX, B+tree on fb | Likely | Follows from the PGM comparison |
| Order among the competitors | Measured | PGM vs LIPP not resolved |
| Margins on the other nine datasets | Unknown | Larger where PGM's level 0 misses cache (books, planet, genome, fb, osm); smallest on stack |

Expected E[D] (4 KiB pages, estimate): stack 2.2-2.4, wise 2.3-2.7, history 2.4-2.8, covid 2.5-2.8, libio 2.6-3.0,
books 3.1-3.5, genome about 3.8, planet 3.9-4.0, fb 4.0-4.1, osm about 4.2. The counted values are in
`results/splice_h/COUNTS.md`.

Counted (chosen fast cell, 4 KiB / 2048 KB, compute constant removed; the model, not a timing): stack 2.03, wise
2.19, covid 2.28, history 2.29, libio 2.42, books 2.98, planet 3.70, genome 3.73, fb 4.07, osm 4.16
(`results/splice_h/COUNTS.md`). This is not a confirmation of the synthesis' numbers, for two reasons. The chosen
cells are not the ones the predictions describe (fb and planet chose K1 4096 with compact lines, osm K1 8192 with
1-line entries, genome K1 4096 with 1-line entries), and the counted E[D] also charges simulated record and router
misses, router child levels, walk hits and parallel lines, which the synthesis' table folded differently. Counted
like for like at the synthesis' own FENCED cell (K1 16384, 2-line entries, cbar 24, C = 228, uncapped), E[D]
excluding compute is fb 4.35 (predicted 4.0-4.1), osm 4.35 (about 4.2), genome 4.17 (about 3.8) and planet 4.26
(3.9-4.0): 0.15-0.37 D above every FENCED prediction, while fb's dependent lines (2.126) match the synthesis' 2.13.
The selection recovers that gap and more by choosing other cells. The DIRECT and mixed datasets are 0.1-0.3 D below
their predictions because per-segment windows (w up to 4) bring them to 1.00-1.41 dependent lines against the
1.04-1.94 sequential lines the prediction used. Every dataset verifies (200M keys, 10M absent keys; compact arms
too) and `get_impl<Count>` equals the plan walker exactly. Mode shares: fb, osm, genome all FENCED; planet 93%
FENCED; books 86% DIRECT (predicted 47-59%); covid, history, libio, stack, wise 92-100% DIRECT. The fast cap of
22 B/key binds on fb, where the baseline cell (K1 16k, uncompacted lines) is forced into long DIRECT scans (27.7 D);
the chosen cell uses compact lines at 19.28 B/key. On genome and planet the best cell fits the compact cap too, so
both arms build the same layout (15.60 and 16.30 B/key). The compact arm (16.5 B/key) has no feasible cell on books,
covid and osm, so those datasets get no `<ds>.compact.args`. The perturbed grid (all 200M keys) chose the same fast
cell on all 10 datasets with |dE[D]| <= 1 milli.

Measurements that confirm or kill it (SERVER.md section 5b; thresholds are decision rules, not predictions):

| # | Measurement | Rule |
|---|---|---|
| 1 | THP state (`transparent_hugepage/{enabled,defrag}`, jemalloc `thp`, AnonHugePages) | Decides which arm is primary; the 4 KiB warm arm is the headline |
| 2 | Pointer chase over 4 GB, 4 KiB and THP (D, walk cost) | **Kill** if D >= about 170 ns together with a residual >= about 250 ns: fb then ties PGM |
| 3 | k adjacent independent lines, k = 1..10 | If 2 lines cost about 2 D: switch to 1-line entries (`entry=1`); fb's over-capacity share then rises to 0.408: recount |
| 4 | lfence A/B on the get() loop | Measures cross-lookup overlap (0 to 0.9 D credit); report the fenced variant |
| 5 | GRE with an O(1) fake index (`nullindex`, implemented) | Harness residual |
| 6 | perf: L2-missing loads and D-side walks per lookup, fb and stack, with GRE's harness subtracted by `nullindex` runs at the same op counts (SERVER.md 5b step 8) | Expect the committed cell's dependent + parallel lines + simulated record/router misses, and its pages4k (per dataset in `results/splice_h/COUNTS.md`, "Counter check"); not the synthesis' 2.13 lines, which was another cell and left out the parallel second entry line. A deviation above 0.3 per lookup in either count invalidates the count model |
| 7 | PGM and LIPP on all 10 datasets | Required before any per-dataset ratio is stated |
| 8 | GRE runs: 20M warm-up, 100M lookups, >= 3 interleaved rounds | **Adopt only** if the lower 95% CI bound of splice/pgm on fb is above 1.15. **Kill the read-only claim** if fb is below 1.35 Mops/s on 4 KiB pages |
| 9 | Transient check (trace_splice, trace_lipp; W in {0, 1M, 20M}) | Accept W = 20M only if the first timed chunk is within max(2 x spread, 1%) and the W = 0 transient is at most 2M lookups; else investigate |
| 10 | Ablations (SERVER.md 5b step 7): Hist-Tree knots, DIRECT-only (books, stack), FENCED-only, K1 16k vs the plan, fb with and without its 21 outliers (`exc=0`); static PGM with a parallel last-mile window is **not implemented** | Hist-Tree within 0.01 lines or within noise kills the "learned" claim |

Items 2-4 of the calibration kit and the static-PGM ablation are not implemented in this change; item 5 is
(`nullindex`). Until the kit runs, D, the walk costs and the overlap credit stay estimates.

## 9. Reproduction

Mac (counts and correctness; timings are meaningless there):

```bash
cmake -S experimental/scaleli -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build                                    # scaleli_* tests unchanged, plus splice_core, splice_layout17, splice_gre_check
D=experimental/scaleli/data/external/gre
for ds in fb osm books covid genome history libio planet stack wise; do
  build/splice_count --dataset $ds --data $D --out experimental/scaleli/results/splice_h --threads 16 --grid pruned --verify --parity --perturb
done
experimental/scaleli/tools/splice_xarch.sh [--full]       # arm64 vs x86 (Rosetta) layout hashes
experimental/scaleli/tools/splice_asm_check.sh            # get() code: no call/div/x87/fma/bsr/lzcnt, stores listed
python3 experimental/scaleli/tools/splice_table.py experimental/scaleli/results/splice_h > experimental/scaleli/results/splice_h/COUNTS.md
python3 experimental/scaleli/tests/test_gre_report_splice.py
```

Layout of `results/splice_h/`: `<ds>.json` (n, exceptions, per-K1 eps and segments, stage-1 cells, stage-2/3 cells
with the fixed point, Pareto front of bytes/key against E[D], chosen fast and compact cells, layout hash, verify,
parity, perturbation, timings), `args/<ds>.args` and `args/<ds>.compact.args` (one line each, eps explicit),
`COUNTS.md` (generated) and `design_evidence/`.

Server (DIAS): `SERVER.md` section 5b: build with `tools/gre_lite.sh`, smoke test, warm and cold grids with
`tools/gre_run.sh --splice-plan`, the THP arm, the transient check (`tools/gre_transient.py`), reports
(`tools/gre_report.py --ref pgm --splice-json results/splice_h`) and the perf counter check.

## 10. Open uncertainties and the write plan

- **L2 size and policy.** 2 MB per core pair (likely; not confirmed) or effectively 1 MB: tier-1 misses rise from
  0.155 to 0.42 D. Goldmont's replacement policy is unknown; the simulator assumes true LRU.
- **DRAM latency.** No published DRAM or L2 latency for the C3958; D is back-solved from three throughput numbers.
- **Overlap.** Page-walker concurrency, re-issue of uTLB-miss loads at retirement, and whether GCC 15 keeps the
  lookup near 95-105 uops against the 78-entry ROB decide the 0 to 0.9 D overlap credit.
- **Unexplained 250-300 ns** in the competitor fits; it may apply to SPLICE-H too (hence the 0-100 ns residual).
- **Memory.** Counted (index total_bytes): fast arm 15.6-22.0 B/key (genome 15.60, planet 16.30, fb 19.28, the
  DIRECT-heavy datasets 21.1-22.0) against PGM's 16.18; the compact arm meets 16.5 B/key on fb, genome, history,
  libio, planet, stack and wise (15.6-16.5) and on no cell of books, covid or osm. These were estimates in the
  synthesis (19-22.5 fast; compact only on stack, wise, history, libio).
- **Mixed-mode datasets** (books, libio) were estimated by re-runs without the dropped RADIX mode, before tier-1
  misses; the implemented count (`results/splice_h`) is the first end-to-end count of the merged design.
- **RO alternative.** The rank-addressed packed-record design (16.1-16.4 B/key) stays a candidate only if
  measurement 3 shows batches of 5-8 lines cost about 1 D plus 3.3 ns per line.
- **Writes (not designed).** GRE's losing corner is hard data with >= 50% writes, and easy-to-hard shift. Plan:
  reservations inside the same pages (one spare slot per DIRECT line, one overflow line per FENCED entry); the
  elastic density (piecewise exponential with l1 trend filtering, fitted by likelihood) becomes the insert forecast
  that sizes them, the only place where likelihood is the right signal; local re-selection when a segment's counted
  D grows by at least 0.3; a delta/LSM buffer for distribution shift (ALEX lost 52% going covid -> osm).

## 11. Deviations from the synthesis

The implementation spec fixed these departures from `design_evidence/splice_synthesis.md`:

1. **Prediction** uses a 32-bit slope m capped at 2^32-1 with `u = min((x-k) >> pre, 2^32-1)`, one 64-bit IMUL and
   `>> 32`, not a 64x64->128 MUL: pre keeps u < 2^32, it is the arithmetic of `mlp/fullsim.cpp` that produced the
   counts, a slope above 1 unit per key unit adds nothing for unique integer keys, and the clamp keeps absent keys in
   the arena.
2. **Seg16 packing**: key u64, m u32, meta u32 = base:24 (512-B units, 8 GiB) | pre:6 | kind:2 (FENCED or DIRECT
   w in {1,2,4}); regions 512-B aligned (at most 448 B waste per segment, < 0.04 B/key at K1 16k).
3. **Region end** from the adjacent record seg[j+1] (sentinel seg[K], 8 padding records) clamps the DIRECT window and
   the FENCED entry index for absent keys; 16 B leave no room for U_j. Stored keys never trigger the clamp.
4. **DIRECT placement** is a forward cumulative max only; spill extends the segment's own region and the lookup
   only scans right. Every spill line is counted and charged in bytes.
5. **DIRECT slots** use the same 64-B `Line {k[4], v[4]}` as FENCED fast data lines: same bytes, two aligned key
   loads plus pcmpeqq per line, one compare routine for both modes and for Count parity.
6. **1-line entries hold 24 fences** (C = 100 fast, 125 compact), not 25; inner capacity with 2-line entries is
   57 x 228 = 12,996 keys, not 13,224 (which mixed the 57- and 56-fence geometries). Fences load as 3 or 7 aligned
   8-lane vectors.
7. **Section B**: FENCED child entries and data lines live after all segment regions, so section A holds only
   top-level entries and slot lines and the entry clamp follows from the region size.
8. **hardness.hpp is not edited**: the segment-emitting PLA driver is in `include/splice/cost.hpp` over the public
   `OptimalPLA<unsigned __int128>::add_point`, keeping SCALE-LI's headers (and gre_lite's recorded cksum) unchanged.
9. **`scaleli_splice` stays** an alias of Jg0 in gre_lite.sh (tonight's server run uses it); the documentation says
   it is SCALE-LI's joint-root cell, not this index.
10. **Transient check** uses new registry indexes `trace_splice` and `trace_lipp`: a decorator timestamps every 1M
    get() calls (before the call) and prints chunk times from `memory_consumption()`. GRE's `--latency_sample` sorts
    its samples (order lost), and changing benchmark.h would change gre_lite's frozen edits.
11. **K1 grid** {4096, 8192, 16384}; lower values can be passed by args if E[D] still falls at 4096.
12. **Integer selection arithmetic** (milli-D, integer lambda bisection; margins 10 and 300 milli) for bit-identical
    layouts on arm64 and x86.
13. **Selection-time pages** use segment-local offsets (regions treated as page-aligned); only the final plan walk
    uses real offsets, and the outer loop re-evaluates.
14. **Router leaf scan** reads up to 8 merged 16-B records (2-3 lines, masked) instead of one line of 8 knot keys;
    `router_leaf=4` is available if the residency simulation shows the extra lines matter.
15. **SPLICE_ARGS** are key=value tokens; gre_run.sh passes the per-dataset plan file (`--splice-plan`,
    `--splice-arm`) with eps explicit, then the global SPLICE_ARGS.
16. **splice and splice_thp share one layout**, selected on the 4 KiB arm; splice_thp only adds MADV_HUGEPAGE, so
    the page-size effect is isolated. The THP E[D] is still reported.
17. **Perturbation** is applied to the normalised key `t = (x-lo)/(hi-lo)`: `t' = (t + 1e-6 t^2) / (1 + 1e-6)`
    (the division keeps both ends fixed), mapped back and forced strictly increasing (1e-6 x^2 on raw 64-bit keys
    overflows). Offline only (`splice_count --perturb`, `tests/test_splice.cpp`), in double.
18. **Compact data lines** are per leaf entry: compact iff the entry's key span is below 2^32, residual = key -
    entry.base, `CLine {u32 r[5], u32 pad, u64 v[5]}`.

Implementation notes beyond the spec (GRE tooling): `gre_run.sh` also records the cksum of the plan files in the
OUTDIR config, so a resumed grid never mixes two plans; `gre_transient.py` ends a transient at the first run of 5
consecutive in-band chunks and lists later isolated outliers separately (with a 2-sigma band, about 5% of chunks
cross it by noise alone, so "every later chunk in band" would flag nearly every run); `SpliceInterface::get` is
marked `flatten` so GCC inlines the whole lookup into GRE's TU regardless of size heuristics.

Integration fixes (after both halves met): `low_index` in layout.hpp hides the `popcount((m & -m) - 1)` idiom from
the optimiser with an empty asm, because clang (and GCC) rewrote it into cttz, which -march=goldmont lowers to
`rep bsf`; the asm check had missed it because it did not strip the `rep` prefix. `splice_count` writes no args
file for an arm whose chosen cell violates its bytes cap (the JSON says `"feasible": false`), so `gre_run.sh
--splice-plan` refuses that dataset and arm instead of running a cap-violating layout; the Pareto front breaks
ties between the two arms' scorings of one layout deterministically and records `cap_violated`.

Review fixes (after the integration):
- **Stage 2 shares finalists between the arms** (section 4.3). Before, genome and planet committed a fast cell that
  was worse than their own compact cell on both E[D] (+0.25 and +0.14 D) and bytes (+3.4 and +2.5 B/key), because
  stage-1 E[D] ranked 1-line-entry cells about 0.4 D too high and the arms never compared their stage-2 results.
  Both now build the compact cell in both arms; the other eight datasets keep their cells and layout hashes.
- **The bytes cap bounds total_bytes exactly**, the 2 MiB arena rounding included (wise's compact arm had built
  16.505 B/key under a 16.5 cap while reporting `cap_violated: 0`; it is now 16.46).
- **Access traces for the L2 simulation keep 1024 ids per lookup** (was 192, silently cut) and count cuts
  (`trace_truncated` per fixed point in the JSON). Only cap-bound DIRECT-only ablations (95-255 D per lookup) still
  exceed the cap; no chosen or stage-2 cell does.
- **The compact arm is materialised, verified and hashed** by splice_count (`compact_layout_hash`,
  `compact_verify`), so `gre_report.py --splice-json` checks it under `--splice-arm compact` and warns when a JSON
  has no hash for the arm.
- **The synthesis' own cell is counted** (stage-3 entry `synthesis_cell`), which replaced comparisons between
  different geometries: osm's over-capacity at C = 228 is 0.096 with eps-PLA knots, outside the spec's sanity band
  (0.191 +/-0.03) because that figure came from the radix-trie prototype; fb is 0.127 (inside 0.134 +/-0.02).
- **The perturbation check ran at full scale** (`splice_count --perturb` on all 10 datasets): same fast cell
  everywhere, |dE[D]| <= 1 milli.
- **splice_asm_check.sh** now strips `rep`/`lock`/`notrack` prefixes before matching, rejects every indirect jump
  that is not a local jump table (clang's `jmpq *` had escaped the old pattern) and every jump out of the function,
  and decides stores with a must-dataflow over the CFG instead of a path-insensitive copy set (which had accepted
  stores through %rdi and %rsi). Self-test: injected stores through the arena, the View, a global and a pointer that
  is the out parameter on one path only, an indirect call, a tail call and TZCNT all FAIL; the real probe PASSes
  with one out store and the kind-switch jump table.
- **`nullindex`** (an O(1) get() that touches no memory) was added to gre_lite.sh's registry, so the perf check
  of SERVER.md 5b subtracts GRE's per-op sampling (one random read of the 1.6 GB key array per op, before the timed
  loop) and compares against dependent + parallel lines, not the synthesis' 2.13 dependent lines of another cell.
- **gre_report.py** warns when a plain `splice` arena has AnonHugePages (THP `[always]`: not a 4 KiB arm).
- **Documentation**: the Hist-Tree rule is applied in lines (section 7); the ablation is described as knot
  placement with the same per-leaf model; the residency simulation is described as pricing, not as a source of
  layout changes; ledger rows that only had their measurement protocol verified now say "paper, cited"; the proxy
  Spearman values are recomputed from `design_evidence/proxy/surr.json` (`proxy/spearman.py`); the router has no
  depth cap (osm reaches 3 child levels on its chosen cell); the build-thread asymmetry is stated (section 4.5);
  SERVER.md 5b gained the THP rule, the asm check on the server's GCC, an ablation step and the corrected perf step.
