# Matching NFL and CSV: how the original authors test, where SCALE-LI differs, and what to do next

Decision document, 2026-10-05.

Written from a research run whose findings were checked claim by claim against the code and papers (146 claims: 129 confirmed, 17 corrected before this text was written). Effort and run-time figures are estimates.

**Sources**
- NFL code: luffy06/NFL @ `fe71172` (GPL-3.0).
- NFL paper: arXiv 2205.11807v1 (local PDF), cited as **NFLp**.
- CSV paper: EDBT 2025 camera-ready, pp. 668-680, doi 10.48786/edbt.2025.54, cited as **CSVe**. The arXiv v3 local PDF is cited as **CSVa**.
- GRE: gre4index/GRE @ `e807edc`. Paper arXiv 2207.02900, cited as **GREp**.
- SOSD: learnedsystems/SOSD @ `f52f4cb`. Paper arXiv 2006.12804, cited as **SOSDp**.
- AIDB 2026 hardness paper: local PDF, cited as **AIDBp**.

**Conventions**
- External code was read online only. Nothing external was cloned, built or run.
- IDs in brackets ([N8], [G6], [L2]...) link to permalinks or local files. The full list is in section 6.
- "(unconfirmed)" means a statement was not verified against a primary source.
- "(estimate)" means one of my effort or run-time guesses. Measure before relying on it.

---

## 0. Summary

1. **Neither paper's result can be tested inside SCALE-LI.**
   - NFL's gain comes from fewer conflicts in AFLI. AFLI stores keys sorted by the transformed value z, puts each key at an exact predicted slot (no search inside a node), and uses buckets of at most 6 keys plus child nodes [N3][N5][N6][N8]. The flow's cost is spread over batches of 256 keys [N21] (NFLp Table 2: 8.38 ns/key at batch 256 vs 169.53 ns at batch 1).
   - CSV's gain comes from moving deep keys up ("promoting" them) in gapped hierarchical indexes (ALEX, LIPP, SALI). It is measured per promoted key (CSVe §5, §6.1).
   - SCALE-LI has no z-order, no batching, no gaps and a fixed depth. It also has no headroom inside regions: a perfect model still loses to binary search over the 32 blocks [L3][L4].
2. **NFL can be reproduced exactly.** The official code is GPL-3, needs Intel MKL, and runs on x86 only, which in practice means the Atom. Its driver already contains every baseline the paper used [N20][N22].
3. **CSV can only be reimplemented, not reproduced.** No code is public (CSVe has no artifact statement [C4]), and about 14 details are unreported (section 5). The natural hosts are MIT-licensed: microsoft/ALEX and Jiacheng-WU/lipp.
4. **SCALE-LI has no external baselines.** GRE is the standard driver for that. `tools/gre_lite.sh` builds a trimmed GRE for the Atom [L1]; its first Atom build failed on ALEX's LZCNT/TZCNT, fixed in f0f39aa with ALEX's own `ALEX_USE_LZCNT 0` (inserts only), and the rebuild is pending. Our current read-only protocol is closest to AIDB's: full bulk load, uniform lookups, warm-up, one pinned thread. Ours uses 5M lookups after 1-4M warm-up; AIDB uses 100M after 20M.
5. **New finding: NFL's key conversion changes the datasets.** NFL turns uint64 keys into doubles by keeping only their **last 15 decimal digits** [N15].
   - In our local GRE copies, 99.97% of `books` keys are ≥ 1e15 (max 9.22e18), and 100% of `osm` and `covid` keys are. `fb` and `genome` are below 1e15 (but `fb`'s maximum key is 2^64−1) [L8].
   - So NFL's "AMZN" is very likely books **mod 1e15**. This is an inference, but NFL's books weights file normalises over [0, ~1e15) [N31], which agrees.
   - Running NFL on the AIDB datasets therefore needs an explicit key-conversion decision (option a2).
6. **Recommendation.**
   - Before the meeting: run the official NFL/AFLI unmodified on the Atom (3 datasets × 2 mixes), run GRE-lite read-only baselines, and reframe SCALE-LI's claims.
   - After the meeting: the full NFL grid, then SCALE-LI inside GRE, then CSV inside LIPP and then ALEX, then NFL on the AIDB datasets.

Three questions should be kept apart at the meeting:
- **Q1.** Do NFL and CSV reproduce in their own architectures? Answered by options (a) and (b).
- **Q2.** Where does SCALE-LI stand against standard indexes? Answered by option (c).
- **Q3.** What is SCALE-LI's own contribution (the learned root plus CSV-inspired virtual fences)? Answered by option (d).

---

## 1. How the originals run their tests

### 1.1 NFL / AFLI (Wu et al., VLDB 2022)

| Aspect | What they do | Evidence |
|---|---|---|
| Code, licence | luffy06/NFL @ fe71172 (2023-11-01), GPL-3.0. The baselines are 2021 forks: luffy06/ALEX af1265d (MIT), luffy06/lipp 9cfb83a (MIT), luffy06/PGM-index 71ec013 (Apache-2.0); each adds statistics counters. Google cpp-btree 1.0.1 comes as a tarball. | [N25], GitHub API |
| Architecture: flow | Keys are min-max scaled to [0, shifts] (1e6 by default, 1e5 for longitudes, 1e7 for fb). Features are [x, x − floor x]. The network is B-NAF with 2 inputs, 2 tanh hidden units and 2 outputs, no biases; z is the sum of the two outputs (8 weight entries). The training code makes every unmasked weight positive with exp. C++ inference uses MKL cblas_dgemm and vdTanh. | [N10][N13][N14]; NFLp Alg. 3.1 p.6 |
| Architecture: how the flow is used | All bulk keys are transformed (batches of 4196), sorted by z, and AFLI is built on z. The payload is the original (key, value). Lookups compare z only, with an absolute epsilon of about 2.2e-16. | [N1][N3][N18]; NFLp Fig. 4 |
| Switch rule | The flow is kept only if the 99th-percentile tail conflict degree of a least-squares rank line fitted on z is at least 10% lower than on the raw keys. The 10% margin is an integer, so it is 0 when the original value is below 10. The winning value becomes the bucket cap, clamped to [1,6]. Per the paper the flow is off for YCSB, AMZN and WIKI. | [N2][N4][N9]; NFLp p.6, Table 3 p.11 |
| AFLI index | Model node: a linear model plus a slot array of capacity min(predicted span, 2n) and 2 bitmaps (empty, data, bucket, child), with no search inside the node. Bucket: at most 6 keys, unsorted, scanned linearly. A conflict above the cap creates a child node. Dense node: sorted array with binary search, used only when no model fits. Inserts go slot → bucket → child; nodes are never retrained. The paper's gapped dense node exists only in commits up to Aug 2022. | [N5][N6][N7][N8][N30]; NFLp §3.3 |
| Harness | One binary, `benchmark <index> <batch> <workload.bin> float64 <config>`, for nfl, afli, lipp, alex, pgm-index and btree. All share the same batch loop. | [N20][N22] |
| Datasets | 7 datasets of about 200M keys. longitudes, longlat and ycsb come from the ALEX Google Drive. books, fb, wiki-ts and lognormal come from SOSD. Paper names: LTD, LLT, YCSB, AMZN, FB, WIKI, LGN. | [N23]; NFLp §4.1.1 |
| Key and value types | double keys and int64 payloads (mt19937_64, seed 1e9+7). uint64 keys are cut to their last 15 digits, then sorted and deduplicated. books is therefore effectively mod 1e15 (see summary point 5). | [N15][N31][L8] |
| Workload | 50% of keys are bulk-loaded, with min and max forced in. The rest are inserted in shuffled order. Each batch of 256 holds int(256·r) lookups and then inserts: 256/0, 204/52, 51/205, 0/256. Lookups follow scrambled Zipf 0.99 over the keys present at that time. The generators are seeded from std::random_device, so a regenerated workload differs. There are no updates, deletes or scans. | [N16][N17]; NFLp p.9 |
| Operations | N − N/2 ≈ 100M requests per workload (about 390,625 batches). Inserts: about 0, 20.3M, 80.1M and 100M for the four mixes. | [N16] |
| Warm-up | None. Each repeat is a fresh process that rebuilds the index. | [N32][N22] |
| Timer and timed region | high_resolution_clock around each batch of 256. Inside: every op, with lookup values summed. Copying the batch is outside the timer. NFL additionally times the flow transform as "T" (normalise, 2 dgemm, vdTanh, sum; MKL sequential). | [N20][N21][N28] |
| Repetitions | 3 processes per cell in the code; the paper says 5. Arithmetic mean; the std is computed but not printed. | [N22] L16; [N26]; NFLp p.9 |
| Metrics | Throughput = ops·1e3 / Σ(T+I) ns, in Mops/s. P50/75/99/99.5/99.99/max are per-batch times divided by 256, not single-op latencies. Index size is an analytic count (nodes, slot arrays, full bucket capacity, flow). Bulk-load time is split into transform and index. | [N19][N9]; NFLp §4.3 |
| Parameters | Batch 256; bucket cap ≤ 6; α = 2. ALEX, LIPP and B-tree use defaults. PGM uses ε = 16, but base, buffer_level and index_level are tuned per mix: 64; 32/1/5; 4/3/9; 2/4/26. | [N24][N5] |
| Flow training | PyTorch float64, default device cuda:0. Data: a 10% sample (at least 10k) of the 100R bulk keys, 3 rounds. Adam (amsgrad) lr 0.1, clip 0.1, batch 4096, at most 15 epochs. Target N(0,1); the paper says variance 1e16. The committed weights predate the training code. | [N11][N33][N12]; NFLp p.6, p.9 |
| Hardware | i7-10700 2.9 GHz, 64 GB, Ubuntu 20.04, GCC 9.3.0 -O3 (the repo uses -O3 -march=native), single thread. RTX 3080 for training. | NFLp §4.1.3 p.9; [N27] |
| Headline numbers | Read-only, on average: 2.34× LIPP, 2.46× ALEX, 3.82× PGM, 7.45× B-tree. Read-heavy: +72.22% over LIPP and +101.05% over ALEX. Table 3 tail conflicts: FB 386→4, LLT 146→4. | NFLp pp.9-11 |
| Driver caveats | LIPP is called with at(), which skips the key check. The forks add counters on the hot path. The val_sum accumulator is never read, so the compiler could drop lookups (unverified). | [N20]; fork comparisons |

### 1.2 CSV (Amarasinghe, Choudhury, Qi, Bailey, EDBT 2025)

| Aspect | What they do | Evidence |
|---|---|---|
| Code | None public: not in arXiv v1-v3, not in EDBT, not on the authors' pages, not found on GitHub. The harness is "based on" Bachfischer/LogarithmicErrorRegression @ 640a4f0. That repo has no licence and only ALEX and PGM, and it is a poisoning microbenchmark. | CSVe §6 p.675; [C4][C1] |
| Version to match | EDBT: 6 datasets, 25 repeats per key, CSV preprocessing on 32 threads, NFL and GI baselines, absolute units. arXiv v3: 4 datasets, 100 repeats, percentages. | CSVe §6.1 p.676 vs CSVa §6.1 p.9 |
| Architecture | CSV is not an index. It is a one-time pass after bulk load on ALEX, LIPP or SALI, working bottom-up. For each node with a subtree it collects the keys, runs Algorithm 1 with λ_i = α·n_i, and if cost < c replaces the subtree with a single new leaf; that promotes keys. Virtual points take real storage and act as gaps that absorb later inserts. CSV is never re-run after inserts. The root is level 1. LIPP and SALI start at the "second level"; ALEX starts at the bottom. The pass stops at level 2. | CSVe §5 pp.674-675, Alg. 2; §6.1 p.676 |
| Algorithm 1 | Greedy over runs of integers that are not keys. Runs of ≤ 2 points: keep both. Longer runs: the endpoint derivative signs decide between the two endpoints and the interior minimum. Each round picks the candidate with the lowest OLS-SSE. Stops at λ points or when the loss stops falling. Claimed cost O(λ+n). | CSVe §4 pp.671-674; CSVa Alg. 1 p.7 |
| Cost rule | LIPP/SALI: the loss itself. ALEX: search_const × expected log2 error + traversal_const × level < c, with c < 0. The constants are sampled at runtime. No values are reported. | CSVe §5.1 eq. 16 p.675; [C5][C6] |
| Datasets | Facebook, Covid, OSM, Genome, Books and Uniform (sparse integers), 200M keys each, deduplicated. Books and Uniform are omitted for ALEX. | CSVe §6.1 p.676 |
| Key and value types | Unique integer keys. Bit width and payload type are not stated. | CSVe §3 pp.670-671 |
| Workloads | Read-only: bulk-load everything, run CSV, then query. Read-write: bulk-load a random half, run CSV once, insert the other half in 5 random batches of 0.1n (20M each at 200M), and query promoted keys after each batch. | CSVe §6.1 p.676; §6.3 p.679 |
| Query set | Every promoted key, i.e. every originally level-≥3 key that moved up. A random-key panel (Figs. 8 and 12) uses an unreported number of keys. | CSVe §6.1 p.676 |
| Timing | Each key is timed separately, the query repeated 25 times back to back and averaged. Timer and timed region are not stated. | CSVe §6.1 p.676 |
| Warm-up | Not reported. The back-to-back repeats warm each key. | — |
| Threads | CSV preprocessing on 32 threads; everything else single-threaded. | CSVe §6 p.675 |
| Parameters | α ∈ {0.05, 0.1, 0.2, 0.4, 0.8}, default 0.1. Scalability runs at 12.5M to 100M, made by dropping every j-th key. | CSVe §6.1 p.676 |
| Repetitions | Not reported. One value per bar, no error bars. | CSVe Figs. 6-12 |
| Metrics | Total time saved on promoted keys (ns); average ns vs α = 0; promoted count among level-≥3 keys; storage bytes; node reduction; insert ns; preprocessing seconds. | CSVe §6.1 p.676 |
| Hardware | Ubuntu 20.04.5 VM, AMD EPYC 7763 (64 cores), 128 GB. Compiler not reported. | CSVe §6 p.675 |
| Headline numbers | Up to 34% faster on promoted keys (LIPP, SALI); up to about 60% of promotable keys promoted (fb); storage overhead < 10% typically, ≤ 31% worst case; random-key time about unchanged. LIPP CSV preprocessing 76-956 s (Table 2); ALEX 110-20,158 s (Table 3). | CSVe pp.676-678 |

### 1.3 GRE (Wongkham et al., VLDB 2022): the standard driver for updatable indexes

| Aspect | What they do | Evidence |
|---|---|---|
| Code, licence | gre4index/GRE @ e807edc. No LICENSE file. Submodule licences are mixed: STX B+tree, FINEdex and Wormhole are GPL-3.0. | GitHub API; [G2][G14] |
| Architecture | One driver, `Benchmark<uint64,uint64>`. Each index implements an 8-method interface (bulk_load, get, put, update, remove, scan, init, memory_consumption) and is registered by name. | [G1][G2][G8] |
| Indexes | alex, alexol, lipp, lippol, pgm (DynamicPGM, ε16, defaults), xindex, finedex, btree (STX), btreeolc, artolc, artunsync, hot, hotrowex, masstree, wormhole_u64. No SALI, RMI, NFL or CSV. | [G2][G13] |
| Datasets | 10 datasets of 200M unique uint64 keys: books, fb, osm, covid, genome, stack, wise, libio, history, planet. CUHK mirror, no checksums. These are the same files we use. | [G15]; GREp Table 2 |
| Preparation | Load, sort, deduplicate, shuffle (mt19937, seed 1866). Bulk-load the first init_table_ratio of the keys (default 0.5), sorted. Payload is the constant 123456789. | [G3] |
| Workloads (paper) | Read-only: bulk-load 200M, then 800M lookups. 80/20, 50/50 and 20/80 lookup/insert, and write-only: bulk-load 100M, then insert the other 100M. Also delete, range, data-shift and YCSB A/B/C. | GREp §3.3 p.5; [G16] |
| Operation generation | All ops are pre-generated. Each op type is drawn independently against cumulative ratios. Lookups are uniform with replacement (or scrambled Zipf 0.99), drawn only from bulk-loaded keys. Inserts follow shuffled order. Generation stops when the insert keys run out. | [G5][G9][G10] |
| Warm-up | None. | [G6] |
| Timer and timed region | rdtsc (TSCNS, calibrated), from after an omp barrier to the end of `omp for schedule(dynamic,10000)`. Loading, generation and bulk load are not timed. | [G6][G11] |
| Threads | OpenMP. The paper uses 1 and 24 threads (one socket, hyper-threading off). The code does no pinning. | [G6]; GREp §4 |
| Repetitions | 1 per invocation; the paper reports the mean of 3. | GREp p.5; [G17] |
| Metrics | Throughput in ops/s. Optional 1% latency sampling, which runs inside the timed loop. memory_consumption() after the run (returns 0 for 7 wrappers, including btree). Dataset hardness as PLA segment count. | [G7][G6] |
| Hardware | 4× Xeon Platinum 8268 (96 cores), 768 GB, gcc 8.3.0 -O3. Build needs MKL, TBB 2020 and jemalloc, with -march=native. | GREp §3.4; [G14] |
| Caveats | The uniform sample depends on the OpenMP thread count. ALEX update() is a no-op. The Zipf generator uses YCSB's zeta for 10^10 items, so it only approximates Zipf 0.99. | [G9][G12][G10] |
| Our status | `tools/gre_lite.sh` builds alex, lipp, pgm, btree and artunsync without AVX2 or MKL (Atom rebuild pending). The Atom (Goldmont) lacks LZCNT/BMI1, so ALEX is built with its documented `ALEX_USE_LZCNT 0`, which changes only the insert-time gap search. All five wrappers' get() check that the key exists, so `success_read == operations_num` is a valid read-only correctness check. HOT needs AVX2/BMI2; XIndex and FINEdex need MKL. | [L1] |

### 1.4 SOSD (Marcus et al., VLDB 2020): the standard read-only harness

| Aspect | What they do | Evidence |
|---|---|---|
| Code, licence | learnedsystems/SOSD @ f52f4cb, GPL-3.0. | GitHub API |
| Architecture | Static. The data is a sorted array of {key, position}. An index returns a SearchBound [start, stop), and the harness runs std::lower_bound inside it. | [S1][S3] |
| Index contract | Build() returns ns; EqualityLookup(key) returns a bound; size() reports index bytes excluding the data array. Indexes register via `benchmark_64_<x>`. | [S5][S6][S13] |
| Datasets | books, fb, osm_cellids and wiki_ts at 200M uint64, plus synthetic sets. books and osm at 200M are downsampled from 800M by taking every 4th key. Downloads are md5-checked. | [S10][S11] |
| Workload | 10M positive lookups, uniform over unique keys (FastRandom, seed 42). | [S4]; SOSDp p.6 |
| Warm-up | None explicit; warm-cache tight loop. | [S3]; SOSDp p.1 |
| Timer and timed region | high_resolution_clock around the whole loop: lookup, last-mile lower_bound, payload sum and result check. | [S2][S3] |
| Threads | 1, pinned to core 0. | [S7] |
| Repetitions | `-r` defaults to 1. The CSV output reports the median of the per-run mean ns/lookup. | [S7][S8][S12] |
| Metrics | ns/lookup, index size, build ns. Also a `--pareto` size sweep and a cold-cache mode. | [S8][S9] |
| Hardware | Xeon Gold 6230 2.1 GHz, 256 GB. | SOSDp §4.1 p.5 |

### 1.5 AIDB 2026 (Zhang, Tang, Ailamaki): the hardness paper

| Aspect | What they do | Evidence |
|---|---|---|
| Code | None found. Their "common codebase" is not named. | AIDBp §4.1 p.4 |
| Indexes | RMI, PGM, ALEX, LIPP, XIndex, FINEdex; default configurations plus bug fixes. | AIDBp §4.1 p.4 |
| Datasets | The same 10 as GRE, 200M unique uint64 keys each. | AIDBp Table 1 p.4 |
| Workload | Read-only. Bulk load, then random lookups over all keys ("uniform read-only workload over the same dataset"). That the bulk load is full is inferred. | AIDBp §4.1 p.4, §2.2 p.2 |
| Operations | 20M warm-up lookups, then 100M measured lookups. Throughput in MOPS. | AIDBp §4.1 p.4 |
| Timer, repetitions, sampling | Not stated. | — |
| Threads | 1 worker, pinned to a fixed core. | AIDBp §4.1 p.4 |
| Metrics | Throughput normalised by its std across datasets. Conformance per index = (R−P)/(R+P), with sigmoid weights. Coverage = (#comparable − #incomparable)/C(10,2). Hardness metrics: RMSE, ME, CD (LIPP's FMCD), PLA-32, PLA-4096 and their combinations. | AIDBp §3.2 pp.3-4 |
| Hardware | 2× Xeon Gold 5118, 384 GiB, Ubuntu 20.04.6 container on kernel 6.8.0-57, hugepages off, GCC 9.4.0 -O3. | AIDBp §4.1 p.4 |
| Calibration targets | LIPP 7.1 MOPS on libio vs 5.6 on history. GRE metric: conformance 0.71, coverage 0.47. PLA-32: conformance 0.50, coverage 1.00. | AIDBp p.2, Table 2 p.5 |

---

## 2. Where our setup differs

| Dimension | Originals | SCALE-LI now | Consequence for validity |
|---|---|---|---|
| Index architecture | NFL: AFLI with exact-position slot arrays, buckets ≤ 6 and child nodes; depth grows with conflicts (NFLp Table 1: ALEX max height 11 vs 3 on FB with the flow) [N5][N8]. CSV: gapped hierarchical ALEX, LIPP, SALI (CSVe §6.1). | Fixed 4,096-key regions with a linear model each; 128-key compressed blocks; binary search over 32 block descriptors and inside the block; no gaps; fixed depth. | The quantity both methods reduce (conflicts, depth) is constant in SCALE-LI by design. Even a perfect region model loses to binary search over the blocks [L4]; the region-level cap is about 14% of comparisons, and 1.2-1.5% was achieved [L3]. Our negative results describe SCALE-LI, not NFL or CSV. |
| Where NFL's transform enters | A global key transform; keys re-sorted by z; AFLI built on z; lookups by z [N1][N3]. | Input feature of each region's linear model (and a root candidate); records not re-sorted by z. | We test "flow as a regression feature", which is a different hypothesis. The NFL mechanism (z-order makes conflicts uniform, so fewer buckets and children) is absent. |
| Flow cost and batching | Batch 256, MKL dgemm, timed as T: 8.38 ns/key (NFLp Table 2) [N21]. | Batch 1: 66 ns per evaluation, CI [56, 77] [L6]. | Our flow costs about 8× NFL's batched figure. A throughput loss in our harness says nothing about NFL's batched setting. |
| Flow model and training | B-NAF with all-positive weights, N(0,1) target, inputs scaled to [0, shifts], their PyTorch trainer [N11][N13][N14]. C++ uses the unfloored x as first feature [N10]. | Clean-room 2-input 2-layer tanh flow with our own trainer. It can read luffy06's weight format. | Even the flow is not theirs. Their committed weights exist for fb and books only among our datasets, and their books weights apply to books mod 1e15. |
| Switch rule | Global: the 99th-percentile tail conflict must drop by at least 10% [N2]. | Per region, "NFL-style". | Different decision unit. NFL's switch decides on the whole key set. |
| Where CSV's points live | Physical gaps in a rebuilt subtree node; keys promoted upward; gaps absorb inserts (CSVe §4 p.671, §6.3 p.679). | Changed model targets inside dense regions (no physical gaps) [L3], plus virtual fences at the root with an O(1) slot→region table. | The empty-slot assumption holds only at our root. Our flat structure has no analogue of "promoted keys", CSV's main metric. |
| CSV algorithm | Algorithm 1 plus Algorithm 2 (hierarchical merge); claimed O(λ+n). | Algorithm 1 only, O(λ·n) [L2]. | Our build times are not comparable with CSVe Tables 2-3, and our Algorithm 1 is too slow for large subtrees. |
| Key and value types | NFL: double keys, int64 values, uint64 cut to the last 15 digits [N15]. GRE, SOSD, AIDB: uint64. CSV: integer, width unstated. | uint64 → uint64; values = mix64(row). | NFL's AMZN is not our books [L8]. Comparing with NFL's books numbers is invalid. |
| Datasets | NFL: 7 datasets, overlapping ours only on fb (plus books mod 1e15). CSV: 6, of which 5 are in the GRE mirror. GRE and AIDB: our 10. | GRE/AIDB 10. | NFL's flow helped mainly where tail conflicts were high (FB 386, LLT 146, LGN 14, LTD 8; NFLp Table 3). Only fb is in our set. |
| Workload shape | NFL: 50% bulk, 4 read/insert mixes, Zipf 0.99. CSV: full bulk with promoted-key queries, or half bulk plus 5 insert batches. GRE: read-only with full bulk, mixes with half bulk, uniform. AIDB: full bulk, uniform, read-only. | Full bulk, read-only, uniform over stored keys. | Closest to AIDB and GRE read-only. NFL's write-mix claims (read-heavy +72%/+101%) and CSV's insert-absorption claims are untested by us. |
| Operation count | NFL ≈ 100M; GRE 800M read-only; SOSD 10M; AIDB 100M after 20M warm-up; CSV = #promoted × 25. | 5M after 1-4M warm-up. | Our 5M is 20× fewer than AIDB; this matters for tail stability and for comparing absolute MOPS. |
| Warm-up | NFL, GRE: none. SOSD: none explicit. CSV: not reported. AIDB: 20M. | 1-4M, from the same distribution. | NFL and GRE include the cold start in their numbers, diluted over 100M-800M ops. Matching them means no warm-up; matching AIDB means 20M. |
| Timed region and timer | NFL: per-batch high_resolution_clock, transform included [N21]. GRE: rdtsc over one omp loop [G6]. SOSD: whole loop including last-mile search and check [S3]. CSV: per key ×25, timer unknown. | steady_clock around the whole replay. | Similar to GRE and SOSD. NFL's "P99" is a per-op average within a batch, not a per-op latency; our numbers are not comparable with it. |
| Repetitions and aggregation | NFL: mean of 3 (paper says 5), no spread. GRE: mean of 3. SOSD: median. CSV: single values. AIDB: not stated. | Ratio vs baseline with CIs over 3 blocks. | Ours is stricter. When matching, also report their aggregate (the plain mean of 3). |
| Metrics | Throughput, batch-latency percentiles, analytic sizes (NFL); promoted-key time saved, promoted count, storage, node reduction, preprocessing (CSV); throughput, latency, post-run memory (GRE); ns/lookup and index-only size (SOSD); MOPS, conformance, coverage (AIDB). | Accounted bytes/key, comparisons per lookup, throughput ratio, hardness metrics. | No original reports comparisons per lookup. Our bytes/key includes values; SOSD's size excludes data; GRE's size is end-to-end after the run. Only like-for-like definitions are comparable. |
| Baselines | NFL: AFLI, LIPP, ALEX, PGM (tuned per mix), B-tree. CSV: host vs host+CSV (plus NFL and GI). GRE: 15 indexes. AIDB: 6. | None built. ALEX and PGM adapters exist but are untested [L7]; no B+tree or ART. | SCALE-LI has no external anchor. Binary search beat every learned cell 1.18-1.49× (n = 1) [L5]. |
| Machine and compiler | i7-10700 / GCC 9.3; EPYC 7763 VM; 4× Xeon 8268 / gcc 8.3; Xeon Gold 6230; 2× Xeon Gold 5118 / GCC 9.4. All x86, all with AVX2. | Atom C3958 (no AVX [L1]), Ubuntu 18.04, conda GCC 15; busy M3 Max (ARM). | Absolute numbers are not comparable, and cache differences may shift ratios. MKL-dependent code (NFL, XIndex, FINEdex) runs only on the Atom. Record `lscpu` with every result. |
| Threads | All measurements single-threaded (CSV preprocessing uses 32). | Single-threaded lookups; 16-thread build. | Matches. |

---

## 3. How to match them

### Licences and environment constraints (apply to every option)

| Component | Licence | What we may do |
|---|---|---|
| NFL code, scripts, flow_weights | GPL-3.0 | Run it and publish numbers. Any patch is GPL-derived: keep it outside the MIT repo. Do not commit their weight files. |
| NFL's forks of ALEX, lipp, PGM | MIT, MIT, Apache-2.0 | Usable with notices, but the 2021 forks carry hot-path counters. Prefer upstream for new work. |
| GRE | No licence | Use locally only. Never commit GRE files or patched copies; `gre_lite.sh` already keeps it outside [L1]. Our own wrapper code is ours. |
| A GRE binary built with STX btree, FINEdex or Wormhole | GPL-3.0 parts | Do not distribute the binary. |
| microsoft/ALEX, Jiacheng-WU/lipp, cds-ruc/SALI | MIT | Can be forked into the MIT repo with notices (CSV hosts). |
| SOSD | GPL-3.0 | Run only. |
| Bachfischer harness | No licence | Read only; its poisoning argmax is buggy anyway [C3]. |
| CSV paper | CC BY-NC-ND 4.0 | Implement from its description; do not copy text or figures. |
| Intel MKL | Intel proprietary (licence terms unconfirmed) | Install from conda; do not vendor. |

- **x86 and ISA.** MKL is x86-only, so NFL cannot run on the M3 Max. The Atom has no AVX [L1].
  - NFL's GCC flags are `-O3 -march=native` [N27]; the AVX2 flag appears only on the MSVC branch.
  - Whether conda's MKL picks a working code path on this CPU is unconfirmed. Check with `MKL_VERBOSE=1`.
- **GCC 15.** 2021-era headers miss `<cstdint>`. `gre_lite.sh` adds `-include cstdint` for this [L1], and the same is likely needed for NFL and its forks (unconfirmed).

### (a) Run the official NFL and AFLI as-is, outside our MIT repo. Use for Q1 (NFL).

**Steps**
1. Use the Atom. Create a working directory outside the repo, e.g. `/tmp/louisvasseur/NFL` (the same convention as `gre_lite.sh`). Clone luffy06/NFL and `git checkout fe71172c326e2a0abb7e31d41b8383ff04672796`.
2. Install dependencies in the conda environment: MKL (`mkl`, `mkl-devel`, `mkl-include`), Boost headers (common.h includes boost/optional), CMake ≥ 3.12 and OpenMP [N29].
   - NFL finds MKL through its bundled `lib/libmkl/MKLConfig.cmake`, sequential by default [N28].
   - If that fails with conda's MKL (unconfirmed), edit the CMake file locally to link `mkl_intel_lp64 mkl_sequential mkl_core` explicitly.
3. Run `bash scripts/bootstrap.sh`. It clones the 3 forks at their pinned SHAs and fetches cpp-btree-1.0.1 from the Google Code archive [N25] (whether that URL still works is unconfirmed). It then builds `benchmark`, `gen`, `format` and `nf_convert`.
4. Get the data for the meeting subset and place it in NFL's `data/`:
   - `fb_200M_uint64` and `wiki_ts_200M_uint64` from SOSD's Dataverse, with md5 checks [S10].
   - `longlat-200M.bin.data` from the ALEX Google Drive link printed by `generate_workloads.sh` [N23].
   - Do not substitute our GRE `fb` copy unless its bytes match SOSD's (unconfirmed).
   - Do not run SOSD's full download.sh: it also fetches the 800M books and osm files.
5. Format and generate exactly as the script does (generate_workloads.sh L138/L143 and L153/L158):
   - `build/format <root> fb_200M uint64`
   - `build/format <root> wiki_ts_200M uint64`
   - `build/format <root> longlat-200M float64`
   - Then for each dataset name and R ∈ {100, 80}: `build/gen workload <root>/data <root>/workloads keyset <name> float64 zipf 256 0.5 <R> 1`
   - **Archive a sha256 of every `.bin`.** The lookups are seeded from std::random_device [N16], so a regenerated file is a different workload.
6. Run `bash scripts/generate_configs.sh` [N24]. It writes the per-mix PGM settings and points NFL at the committed `flow_weights/*_2D2H2L_weights.txt`.
7. As benchmark.sh does [N22], run 3 times per cell: `build/benchmark <algo> 256 workloads/<w>.bin float64 configs/<algo>_<w>.in` for algo ∈ {nfl, afli, lipp, alex, pgm-index, btree}.
   - One process at a time, on an idle machine.
   - `taskset -c 2` is optional; label it as our addition.
8. Aggregate with their `load_result.py` (mean of 3) and also with our CI method. Compare against:
   - NFLp Table 3 switch decisions: flow on for fb and longlat, off for wiki.
   - The read-only ratios 2.34×, 2.46×, 3.82× and 7.45×.
   - The NFL/AFLI ratio, which isolates the flow's contribution.

**Effort (estimate)**
- 1-1.5 days of human time, mostly MKL and build fixes.
- Disk: about 4.8 GB per workload file (24-byte `Request<double, int64>` [N18] × ~200M records). About 29 GB for the 6-file subset; about 134 GB for the full 28-file grid. Check `df` on the Atom.
- Machine time: time one `nfl` run and one `btree` run first; B-tree should be slowest (7.45× below NFL in the paper). Guess: minutes per process, so 108 processes overnight and the full 504 in 1-2 days.

**Risks**
- GPL-3: keep everything outside the repo.
- MKL dispatch on a no-AVX CPU (unconfirmed).
- The Atom is much slower than an i7-10700, so compare ratios, not Mops/s.
- Known paper-vs-code gaps that you inherit: 3 vs 5 repeats; N(0,1) vs variance 1e16; dense node changed; 10% switch margin.
- Driver asymmetries: LIPP `at()` skips the key check; the forks add counters; `val_sum` is never read.
- Workloads are not reproducible across regenerations.

**Result**
- An exact reproduction check of NFL's own claims, in its own architecture and harness, on our hardware.
- NFL vs AFLI is the clean answer to "does the flow help where it was designed to help?".

### (a2) NFL on the GRE/AIDB datasets. Use for Q1 extended to our datasets.

**Problem.** AFLI and the driver use double keys. NFL's converter keeps only the last 15 decimal digits [N15], and our checks show this would scramble books, osm and covid [L8]. A plain cast to double instead loses integer precision above 2^53 (≈ 9.0e15), so close keys merge and are deduplicated away.

**Choices.** Label whichever you pick.
- (i) NFL's own rule: their method, but a different dataset.
- (ii) Cast and deduplicate: count and report the lost keys.
- (iii) Port AFLI and NFL to uint64: a GPL code change, kept outside the repo. Normalisation and the epsilon compare are double-based.

**Training.**
- Generate the training keys with `scripts/prepare_keys_for_flow.sh` (a zsh script; the README misnames it), which calls `nf_convert`.
- Run `train/train_flow.sh` from inside `train/` (it uses relative paths).
- `--device` defaults to `cuda:0` [N33] and `train_flow.sh` does not pass it [N12], so call `numerical_flow.py` with `--device=cpu`. PyTorch's MPS backend on the M3 has no float64, so use the CPU on either machine.
- Dependencies: torch, numpy, scipy, matplotlib, seaborn, tqdm, psutil, POT.
- The weight file format is plain text [N31], so training on the Mac and copying the file to the Atom works.

**Effort (estimate):** 2-4 days, plus CPU training time (the paper took 38 s on an RTX 3080; our CPU time is unknown).

**Risks:** the dataset changes under (i) and (ii); GPL under (iii); we cannot rebuild their weights exactly, because the committed weights predate the training code.

**Result:** NFL in its own architecture on the hardness datasets, which is what the AIDB-style analysis actually needs.

### (b) Integrate CSV into LIPP, then ALEX, clean-room (no official code exists). Use for Q1 (CSV).

**Steps**
1. Hosts:
   - Jiacheng-WU/lipp @ fe6ca49 (MIT) first, then microsoft/ALEX @ 4370da6 (MIT). SALI (cds-ruc/SALI @ dba9157, MIT) is optional.
   - If the measurements will run inside GRE, start from GRE's pinned host versions instead: pohchaichon/lipp a1189c3 (which carries GRE's bug fixes) and ALEX f4365a7.
2. Write a fast Algorithm 1. Our `smoothing.hpp` is O(λ·n) and says so [L2]. A level-2 LIPP subtree of 10^6 keys at α = 0.1 would need about 10^11 candidate evaluations. Target the paper's O(n+λ) (CSVe §4), or at least O((n+λ) log n).
3. Write Algorithm 2: root = level 1, bottom-up, stop at level 2.
   - LIPP and SALI start at the "second level". The wording is ambiguous, so implement both readings and report both.
   - ALEX starts at the bottom level.
   - LIPP accept rule: accept if SSE falls (no threshold is reported).
   - ALEX accept rule: search_const × E[log2 error] + traversal_const × level < c, with c < 0. Sample the constants at runtime using ALEX's own estimator [C5][C6].
4. Unreported design choices (fix them, document them, and label the work "reimplementation"):
   - The rebuilt node's model is OLS over the ranks of K∪V.
   - Virtual points become empty slots: LIPP NONE items, ALEX gaps.
   - Remaining LIPP conflicts are handled by LIPP's own child creation.
   - The real-valued minimum point is rounded to an integer.
5. Measure the CSVe way:
   - Datasets: fb, covid, osm, genome and books from our local GRE copies, plus a sparse uniform set (SOSD `gen_uniform.py --sparse`; which file CSV used is unconfirmed).
   - Deduplicate, bulk-load everything, and log each key's level before and after CSV.
   - Queries are the promoted keys, each timed separately with 25 back-to-back repeats, averaged. Use a calibrated TSC timer, written ourselves (GRE's tscns.h has no licence).
   - α ∈ {0, 0.05, 0.1, 0.2, 0.4, 0.8}.
   - Report all 7 CSVe metrics.
   - Read-write run: half bulk, CSV once, 5 insert batches of 0.1n, query after each batch.
   - Add a random-key panel and preprocessing time with 16 threads (the paper used 32).

**Effort (estimate):** LIPP 1-1.5 weeks; ALEX 1 more week; measurement 2-4 days of machine time. The paper's LIPP CSV pass alone took 76-956 s per (dataset, α) on 32 EPYC threads.

**Risks**
- About 14 unreported details (section 5), so this can never be called "a reproduction".
- LIPP at 200M is the largest index in GRE's size comparisons; whether it fits in 64 GB on the Atom is unconfirmed.
- No GPL is involved (MIT hosts), so this work can live in our MIT repo with upstream notices.

**Result:** the first independent measurement of CSV in its intended hosts. It tests the "promoted keys get faster" mechanism that SCALE-LI cannot express.

### (c) Use GRE's driver and workloads for all SCALE-LI measurements. Use for Q2.

**Steps**
1. Build with `tools/gre_lite.sh` (GRE @ e807edc; alex, lipp, pgm, btree, artunsync) [L1].
   - For AIDB's index set, try adding xindex and finedex with conda MKL. They need MKL, not AVX2 [L1]; whether they build on the Atom is unconfirmed.
   - RMI exists only in SOSD.
2. Write a GRE wrapper for SCALE-LI: the 8 methods of `indexInterface` [G1].
   - SCALE-LI is C++20 and GRE is C++17, so use the C++17 PIMPL bridge pattern already in `integrations/` [L7].
   - The wrapper lives in our repo; GRE stays outside.
3. Fix the harness variables:
   - Set `OMP_NUM_THREADS` to a fixed value, because the uniform sample depends on it [G9]. Keep the default seed 1866.
   - Use `--memory` for sizes. btree reports 0.
   - Never use `--latency_sample` in throughput runs; it adds rdtsc calls inside the timed loop [G6].
4. Workloads:
   - GRE paper read-only: `--init_table_ratio=1 --read=1 --operations_num=800000000`; start at 100M on the Atom.
   - Mixes 80/20, 50/50, 20/80 and write-only: `--init_table_ratio=0.5` with `--read`/`--insert`, and operations_num large enough that generation ends on insert exhaustion [G5].
   - AIDB shape: read-only, init 1.0, 100M measured. GRE has no warm-up, so the 20M warm-up needs either a local patch (kept outside the repo) or our own harness run at 20M + 100M.
5. 3 runs per cell. Report the mean (GREp p.5) and our CIs.

**Effort (estimate):** wrapper 1-2 days; 2-4 days of machine time for 10 datasets × 6 indexes × 5 workloads × 3 runs.

**Risks**
- No licence: never commit GRE.
- Lookups only target bulk-loaded keys.
- ALEX update() is a no-op.
- **NFL cannot be measured faithfully inside GRE.** GRE's `get()` is per key, so there is no batching and the flow costs about NFL's batch-1 figure (169.53 ns/key) or our 66 ns [L6], not 8.38 ns. GRE is also uint64-only.

**Result:** SCALE-LI against ALEX, LIPP, PGM, B+tree and ART under a published protocol, on the same datasets. This fixes the validation finding that we have no published baselines.

### (d) Keep SCALE-LI only for the root-level / joint method. Use for Q3.

**Steps**
- Stop presenting region-level NFL and CSV effects as tests of NFL and CSV; the validator and critique already show there is no headroom [L3][L4].
- Present SCALE-LI's own design: a learned root, CSV-inspired virtual fences, and an optional flow feature.
- Keep the region-level result as an explained negative result: "a flow feature in a key-ordered compressed map has no lever" [L3].
- Measure the design end-to-end through (c).

**Effort:** 0.5-1 day of writing; measurement comes from (c).

**Risks:** on the 2M samples, root virtual fences cut total probes by 39-44% [L6], but on the noisy Mac this has not shown up as time beyond noise. Binary search beat every learned cell (n = 1) [L5]. This needs the Atom and GRE baselines.

**Result:** a contribution claim that does not depend on NFL's or CSV's assumptions.

### (e) Ask the authors (cheap, run in parallel). You send these yourself.

- **NFL** (shangyuwu2-c@my.cityu.edu.hk [N29]): 5 vs 3 repetitions; variance 1e16 vs N(0,1); provenance of the committed weights; MKL version; whether AMZN was intentionally books mod 1e15.
- **CSV** (Amarasinghe, Qi): code; the values of c and the cost constants; how virtual slots are represented; the timer; the meaning of "second level"; which uniform file was used.
- **AIDB** (Zhang, Tang, Ailamaki): the "common codebase", repetitions, and how lookups were sampled. `gre_lite.sh` calls our machine "the DIAS Atom" [L1], so that group may be reachable directly (an inference).

**Effort:** about 1 hour.

### Not recommended now

- **NFL inside GRE:** no batching and uint64 keys, so it would not be NFL's method (see (c)).
- **SOSD for SCALE-LI:** its static search-bound contract and index-only size definition do not fit a compressed key-value map, and the code is GPL. It is useful only for static read-only baselines such as RMI and RS.

---

## 4. Recommendation

### Plan for the next supervisor meeting (about 2-3 working days plus 1-2 nights on the Atom; estimate)

1. **(e)** Send the three author queries (1 h).
2. **(a)** Build the official NFL @ fe71172 on the Atom, outside the repo. Confirm MKL works with `MKL_VERBOSE=1` (4-8 h).
3. **(a)** Fetch `fb_200M_uint64` and `wiki_ts_200M_uint64` (SOSD Dataverse, md5) and `longlat-200M.bin.data` (ALEX Google Drive) (1-2 h plus downloads).
   - fb: flow on (386→4), and the one dataset shared with ours.
   - longlat: flow on (146→4).
   - wiki: flow off, the control.
4. **(a)** Format and generate the 100R and 80R workloads for the three datasets (about 29 GB), and hash every file.
5. **(a)** Overnight: 6 indexes × 6 workloads × 3 repeats = 108 processes, sequential, idle machine.
6. **(c, minimal)** Same machine: GRE-lite single-thread read-only with `--init_table_ratio=1 --read=1 --operations_num=100000000`, indexes alex, lipp, pgm, btree and artunsync. Run fb first and as many of the 10 datasets as fit, 3 runs each, fixed `OMP_NUM_THREADS`.
7. **(d)** Reframe the slides using section 2's table: "their architecture / our architecture / what we can claim".

**What to show at the meeting**
- NFL vs AFLI vs the four baselines in the authors' own harness, against the paper's numbers.
- Standard-baseline throughput on our machine.
- The differences table.
- The books mod 1e15 finding.

**What to decide at the meeting:** which of Q1, Q2 and Q3 the thesis answers.
- If NFL ≤ AFLI on fb and longlat on the Atom, report a non-reproduction (with the hardware caveat) and wait for the author's reply before starting (a2).

### Plan after the meeting (ordered; about 4-6 weeks; estimate)

1. **(a)** Full NFL grid: 7 datasets × 4 mixes × 6 indexes × 3 repeats. Books needs SOSD's 800M file plus downsample.py [S11]; lognormal needs SOSD's generator. 1 day of human time, 1-2 days of machine time, about 134 GB of disk.
2. **(c)** SCALE-LI GRE wrapper, then GRE's read-only workload, three read/insert mixes and write-only on all 10 datasets. 1-2 days plus 2-4 days of machine time.
3. **(c, AIDB variant)** 20M warm-up + 100M measured, pinned. Use a local GRE patch kept outside the repo, or our harness scaled to those counts. 0.5 day plus machine time.
4. **(b)** CSV in LIPP: fast Algorithm 1, Algorithm 2, per-key ×25 timing; promoted-key and random-key panels; α sweep; read-write run. 1-1.5 weeks plus 2-3 days of machine time.
5. **(b)** CSV in ALEX with the runtime-sampled cost model. 1 week.
6. **(a2)** NFL on the AIDB datasets with a documented key conversion and flows retrained on CPU. 2-4 days.
7. Optional: XIndex and FINEdex in GRE via MKL, to cover AIDB's index set except RMI. 1 day (unconfirmed it builds).
8. Write-up with strict labels:
   - "official NFL, GPL code run unmodified"
   - "CSV reimplementation after CSVe, unreported details listed"
   - "SCALE-LI, own design, measured in GRE"

---

## 5. Unconfirmed or open items

- **NFL**
  - Paper vs code: 5 vs 3 repeats; latent variance 1e16 vs N(0,1); gapped vs plain dense node; 10% switch margin.
  - The committed weights cannot be regenerated: no training logs, and the files predate the training code.
  - "NFL's AMZN = books mod 1e15" is inferred from [N15], the books weights normalisation [N31] and our GRE copy [L8]. Byte-identity of GRE books with SOSD `books_200M_uint64` is not checked.
  - Not checked: whether the `val_sum` dead-code risk affects any index; whether the cpp-btree tarball URL still works; whether MKL runs on the Atom and whether conda MKL ships `mkl_link_tool`.
  - Exact post-deduplication key counts per dataset are not reported.
- **CSV**
  - Unreported: the values of c, search_const and traversal_const; the LIPP/SALI accept rule; how virtual slots are represented; rounding of the minimum point; how LIPP conflicts are handled after a merge; the meaning of "second level".
  - Also unreported: timer and timed region; warm-up; key width and payload type; seeds and number of runs; the size of the random query set; which uniform file was used; the NFL and GI versions used.
  - EDBT is internally inconsistent: §1 says four datasets, and Algorithm 1 ends with "return C".
- **AIDB:** the harness identity, repetitions, lookup sampling, timer and pinned core are all unstated. That the bulk load is full is an inference.
- **GRE:** the paper's tuned PGM settings are not in the wrapper. The NUMA and pinning setup was applied outside the code.
- **Our machines:** the Atom's cache hierarchy and whether LIPP at 200M fits in 64 GB are unchecked. All effort and run-time figures in this document are estimates.

---

## 6. Evidence index

| ID | File and lines | Link |
|---|---|---|
| N1 | NFL src/nfl/nfl.h L12-150 (wrapper; lookups by z) | [N1] |
| N2 | nfl.h L26-91 (switch rule and constants) | [N2] |
| N3 | nfl.h L71-105 (bulk transform, sort by z) | [N3] |
| N4 | src/afli/conflicts.h L35-123 (conflict degree) | [N4] |
| N5 | src/afli/afli_nodes.h L24-64 (node types, defaults) | [N5] |
| N6 | afli_nodes.h L105-130 (lookup) | [N6] |
| N7 | afli_nodes.h L132-279 (insert, update, delete) | [N7] |
| N8 | afli_nodes.h L315-398 (build) | [N8] |
| N9 | src/afli/afli.h L23-169 (bucket size, size accounting) | [N9] |
| N10 | src/models/bnaf.h L91-173 (MKL inference) | [N10] |
| N11 | train/numerical_flow.py L44-283 (training) | [N11] |
| N12 | train/train_flow.sh L11-61 | [N12] |
| N13 | train/models/BNAF/bnaf.py L246-269 (positive weights) | [N13] |
| N14 | train/distribution_transformer.py L23-133 | [N14] |
| N15 | src/util/format_data.cc L60-81 (last-15-digit cut) | [N15] |
| N16 | src/util/data_generator.cc L18-92 (workload generation) | [N16] |
| N17 | src/util/zipf.h L8-42 | [N17] |
| N18 | src/util/common.h L33-61 (Request struct, epsilon compare) | [N18] |
| N19 | src/util/common.h L279-363 (metrics) | [N19] |
| N20 | src/benchmark/benchmark.h L224-505 (timed loops) | [N20] |
| N21 | benchmark.h L540-579 (NFL T and I timing) | [N21] |
| N22 | scripts/benchmark.sh L13-61 | [N22] |
| N23 | scripts/generate_workloads.sh L21-158 | [N23] |
| N24 | scripts/generate_configs.sh L36-73 | [N24] |
| N25 | scripts/bootstrap.sh L15-37 | [N25] |
| N26 | scripts/load_result.py L45-117 | [N26] |
| N27 | CMakeLists.txt L5-23 | [N27] |
| N28 | lib/libmkl/MKLConfig.cmake L188-192 | [N28] |
| N29 | README.md L6-65 | [N29] |
| N30 | 54f5b9b src/afli/afli.h L104-122 (old gapped dense node) | [N30] |
| N31 | flow_weights/books-200M-0R-zipf_2D2H2L_weights.txt L1-8 | [N31] |
| N32 | benchmark.h L136-206 (driver; no warm-up) | [N32] |
| N33 | train/numerical_flow.py L135 (`--device` default cuda:0) | [N33] |
| C1-C3 | Bachfischer benchmark.cpp L16-44; alex_benchmark.h L37-91; poisoning.h L110-226 | [C1] [C2] [C3] |
| C4 | Qi's papers.bib L299-324 (no code entry for CSV) | [C4] |
| C5-C6 | ALEX alex_nodes.h L731-752; alex_base.h L228-249 | [C5] [C6] |
| C7 | CSV EDBT camera-ready PDF | [C7] |
| G1-G17 | GRE files at e807edc (see links) | [G1] [G2] [G3] [G4] [G5] [G6] [G7] [G8] [G9] [G10] [G11] [G12] [G13] [G14] [G15] [G16] [G17] |
| S1-S13 | SOSD files at f52f4cb (see links) | [S1] [S2] [S3] [S4] [S5] [S6] [S7] [S8] [S9] [S10] [S11] [S12] [S13] |
| L1 | /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/tools/gre_lite.sh L1-12 (no AVX on the Atom; GRE kept outside the repo) | local |
| L2 | /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/include/scaleli/smoothing.hpp L3-13 (O(λ·n); the paper's O(n+λ) not reproduced) | local |
| L3 | /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/results/aidb_ba/critique.md L91-96 | local |
| L4 | /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/results/aidb_ba/validate.py L638-655 (perfect-model headroom check) | local |
| L5 | /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/results/aidb_ba/PLAN.md L114 (binary search 1.18-1.49×, n = 1) | local |
| L6 | /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/results/aidb/MEETING_NOTES.md L183 (probes −39 to −44%), L238 (66 ns [56, 77]) | local |
| L7 | /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/integrations/STATUS.md (ALEX and PGM adapters untested; bridge pattern) | local |
| L8 | Key-range check on /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/data/external/gre/{books,fb,osm,covid.sorted,genome.sorted}, run 2026-10-05 (header, first and last key, every 1000th key). books: max 9,223,372,036,854,784,874, 99.97% ≥ 1e15. osm, covid: 100%. fb: 0%, max 2^64−1. genome: 0%. | local |
| Papers | NFLp /Users/louisvasseur/Downloads/scaleli_sota/sources/2205.11807v1.pdf; CSVa /Users/louisvasseur/Downloads/scaleli_sota/sources/2408.06134v3.pdf; AIDBp /Users/louisvasseur/Downloads/aidb26_11.pdf; CSVe [C7]; GREp arXiv 2207.02900; SOSDp arXiv 2006.12804 | — |

[N1]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/nfl/nfl.h#L12-L150
[N2]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/nfl/nfl.h#L26-L91
[N3]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/nfl/nfl.h#L71-L105
[N4]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/afli/conflicts.h#L35-L123
[N5]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/afli/afli_nodes.h#L24-L64
[N6]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/afli/afli_nodes.h#L105-L130
[N7]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/afli/afli_nodes.h#L132-L279
[N8]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/afli/afli_nodes.h#L315-L398
[N9]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/afli/afli.h#L23-L169
[N10]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/models/bnaf.h#L91-L173
[N11]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/train/numerical_flow.py#L44-L283
[N12]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/train/train_flow.sh#L11-L61
[N13]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/train/models/BNAF/bnaf.py#L246-L269
[N14]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/train/distribution_transformer.py#L23-L133
[N15]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/util/format_data.cc#L60-L81
[N16]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/util/data_generator.cc#L18-L92
[N17]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/util/zipf.h#L8-L42
[N18]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/util/common.h#L33-L61
[N19]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/util/common.h#L279-L363
[N20]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/benchmark/benchmark.h#L224-L505
[N21]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/benchmark/benchmark.h#L540-L579
[N22]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/scripts/benchmark.sh#L13-L61
[N23]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/scripts/generate_workloads.sh#L21-L158
[N24]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/scripts/generate_configs.sh#L36-L73
[N25]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/scripts/bootstrap.sh#L15-L37
[N26]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/scripts/load_result.py#L45-L117
[N27]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/CMakeLists.txt#L5-L23
[N28]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/lib/libmkl/MKLConfig.cmake#L188-L192
[N29]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/README.md#L6-L65
[N30]: https://github.com/luffy06/NFL/blob/54f5b9b6f9fbc49a44e132f49e9eb4fb9283ca74/src/afli/afli.h#L104-L122
[N31]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/flow_weights/books-200M-0R-zipf_2D2H2L_weights.txt#L1-L8
[N32]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/src/benchmark/benchmark.h#L136-L206
[N33]: https://github.com/luffy06/NFL/blob/fe71172c326e2a0abb7e31d41b8383ff04672796/train/numerical_flow.py#L135
[C1]: https://github.com/Bachfischer/LogarithmicErrorRegression/blob/640a4f044b1815326a19fb20214fd9a855dd64c4/benchmark.cpp#L16-L44
[C2]: https://github.com/Bachfischer/LogarithmicErrorRegression/blob/640a4f044b1815326a19fb20214fd9a855dd64c4/src/helpers/alex_benchmark.h#L37-L91
[C3]: https://github.com/Bachfischer/LogarithmicErrorRegression/blob/640a4f044b1815326a19fb20214fd9a855dd64c4/src/poisoning.h#L110-L226
[C4]: https://github.com/jianzhongqi/jianzhongqi.github.io/blob/bd9b406c733347a8519ece620eff368e5b68c742/_bibliography/papers.bib#L299-L324
[C5]: https://github.com/microsoft/ALEX/blob/4370da6aa8b509fdc9b0d2c49faa0624b0078589/src/core/alex_nodes.h#L731-L752
[C6]: https://github.com/microsoft/ALEX/blob/4370da6aa8b509fdc9b0d2c49faa0624b0078589/src/core/alex_base.h#L228-L249
[C7]: https://openproceedings.org/2025/conf/edbt/paper-204.pdf
[G1]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/competitor/indexInterface.h#L5-L41
[G2]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/competitor/competitor.h#L1-L81
[G3]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/benchmark/benchmark.h#L131-L179
[G4]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/benchmark/benchmark.h#L203-L234
[G5]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/benchmark/benchmark.h#L239-L297
[G6]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/benchmark/benchmark.h#L299-L372
[G7]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/benchmark/benchmark.h#L391-L507
[G8]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/benchmark/benchmark.h#L512-L525
[G9]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/benchmark/utils.h#L196-L232
[G10]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/benchmark/zipf.h#L13-L50
[G11]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/tscns.h#L28-L65
[G12]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/competitor/alex/alex.h#L22-L54
[G13]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/src/competitor/pgm/pgm.h#L22-L32
[G14]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/CMakeLists.txt#L8-L42
[G15]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/datasets/download.sh#L1-L11
[G16]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/README.md#L26-L58
[G17]: https://github.com/gre4index/GRE/blob/e807edcef51df6732f07f94d4c797fb3897519ba/script/plot.py#L85-L109
[S1]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/util.h#L19-L40
[S2]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/util.h#L69-L91
[S3]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/benchmark.h#L193-L278
[S4]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/generate.cc#L14-L130
[S5]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/competitors/base.h#L5-L12
[S6]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/benchmark.cc#L77-L104
[S7]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/benchmark.cc#L112-L180
[S8]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/benchmark.h#L335-L410
[S9]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/benchmark.h#L114-L147
[S10]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/scripts/download.sh#L50-L72
[S11]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/downsample.py#L5-L30
[S12]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/scripts/execute.sh#L4-L38
[S13]: https://github.com/learnedsystems/SOSD/blob/f52f4cba01dfcd37f1574551fccef00198863b88/competitors/pgm_index.h#L16-L62