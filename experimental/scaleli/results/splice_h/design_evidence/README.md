# SPLICE-H design evidence

Small prototype sources and their outputs from the SPLICE-H design workflow (Oct 2026, scratchpad of the design
session), copied here so that `docs/SPLICE_DESIGN.md` can cite repository paths. They are our own code and outputs
(MIT, like the rest of SPLICE); none of them contains GRE, STX or NFL code. Every number is a **count** from a
simulation or a dry run on the Mac, never a timing. The prototypes used double-precision predictions in places; the
implemented index (`include/splice/`) is integer-only, so its counts (`results/splice_h/<ds>.json`) supersede these
where they differ. Absolute paths inside the files are the Mac paths they were produced with: records, not inputs.

| File | Origin (design scratchpad) | What it shows | Cited for |
|---|---|---|---|
| `splice_synthesis.md` | `splice_synthesis.md` | The SPLICE-H synthesis of the 15-agent design workflow (four designs, four critiques). | The design as a whole |
| `warmup_verified.json` | `warmup_verified.json` | Warm-up and measurement protocols of RMI, SOSD, PGM, RadixSpline, ALEX, LIPP, NFL, CSV, GRE, CARMI, TLI and AIDB, verified against the papers and the code at cited commits, plus the corrected protocol for GRE. | SERVER.md 5b, warm-up protocol |
| `impl/<ds>.json` | `impl/layout.cpp` runs at 200M | Per dataset: tier-1 knots and eps at K1 about 32k, router bucket statistics, lines per lookup of DIRECT slot placement (alpha 0.25 and 0.5) and of fence modes, B/key. stack: 1.039 lines at alpha 0.25 (20.0 B/key); fb: DIRECT alone 88 lines, fences 1.000. | DIRECT counts, eps at K1 32k (fb 873, osm 840) |
| `impl/f_<ds>.json` | `impl/layout.cpp`, fence variants | The same statistics for the fenced (directory) modes with depth. | FENCED depth |
| `memfirst/final_w4.jsonl` | `memfirst/tl2.cpp` | Lines per lookup for fetch windows of 1, 2 and 4 lines (alpha 0.25, K 16384): covid 1.027, history 1.034 at w = 4. | Fetch width w |
| `review/l2sim2.cpp` | critique of the tier-1 layout | Shared 16-way L2 LRU over router, knot keys, 16-B records, bucket and data PTE lines. Merged records at K1 16k: 0.155 misses per lookup at 2 MB, 0.42 at 1 MB. | Merged 16-B records, K1 <= 16k |
| `review/fence_chk2.cpp` | critique of the fence geometry | Full-200M checks of the fenced layout: last segments, bucket-size tail, depth >= 3, u32-span fallback, tie cost. | Fence ties, compact eligibility |
| `review/lay2.cpp` | critique of the RO layout | Count-only simulation of the SPLICE-RO layout at 200M (greedy PLA tier-1, DIRECT and fences per segment). | Mixed-mode counts |
| `adv/advsim.cpp` | adversarial review | Count-split radix router plus directory occupancy at 200M: share of keys in entries above capacity C (fb 0.134, osm 0.191 at C = 228; keys_cnt_gt68 0.559). | Over-capacity share, child entries |
| `adv/pte.cpp` | adversarial review | L2 LRU residency of page-table lines for the D2 access stream on 4 KiB pages: directory PTE hit 0.19, data PTE hit 0.06, i.e. walks of 0.81 D and 0.94 D. | Page-walk cost, P_Ae and P_Ad/P_B defaults |
| `mlp/fullsim.cpp` | SPLICE-MLP design | Full-scale count of exceptions plus a count-split dyadic trie with fixed-point leaf models; the integer prediction (`u = (x-k) >> pre`, 32-bit slope) and the exception rule used by `include/splice`. | Exceptions, router, integer model |
| `splice_wk/radix.json` | SPLICE workload study | Raw radix bucket sizes with and without peeled extreme keys: without exceptions all 200M fb keys share one top bucket. | Exception list |
| `review_m/spans.json` | critique of SPLICE-M | Per-dataset key spans and raw radix load: osm's 2^12 radix is 36.5x the ideal bucket. | Count-split router |
| `proxy/surr.json` | proxy study | Error proxies (log2 error, MSE, LIPP conflict degree, NFL tail conflict) against counted lines (Lg) of eight models per dataset on model-placed layouts. `proxy/spearman.py` (added here, stdlib) recomputes the mean Spearman: log2 0.750, MSE 0.774, LIPP conflict degree 0.376, NFL tail conflict undefined on 6 of 10 datasets (constant) and 0.72 on the other 4; log2 0.07 on history. The synthesis quoted 0.33 and -0.16 for the last two; the script that produced them was not kept (`surr.py` is a 2-line stub), so the doc cites the recomputed values. | Why counts, not error proxies |
| `proxy/proxy.json`, `proxy/proxy.py` | proxy study | The same models on sorted-array search: NLL, log2 error and lines touched per model. books: Qknot64 NLL 43.2879 nats at 9.92 lines vs Qknot4096 43.2846 at 3.21 lines (NLL moves 0.003 nats while lines fall 3x). `proxy.py` needs numpy and reads the 2M samples; it is a record, not a server tool. | NLL vs lines |
| `loggap/seg_eps4.txt` | log-gap study | Segments of eps-PLA with lines vs piecewise exponential at eps 4 on the 2M samples: Poisson reference 20,603 exp vs 26,005 line segments (ratio 0.792). | Linear T only |
| `gm_model/bsearch_sim.py` | Goldmont model | Counted cache and TLB events of std::lower_bound over 200M 16-B records on a Goldmont-like hierarchy: about 18.3 serialized accesses, which with the measured 2.18 us sorted array bounds D at about 119 ns. | D = 100-120 ns |
