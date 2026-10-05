# 05 — The host index and the clean-room implementations, with worked examples

**What this section gives you.** This is the "how does the code actually work" section. It walks through the SCALE-LI experimental map (the host index that every control runs inside: regions of 4,096 keys, blocks of 128 keys, packed codecs, fences, delta, compaction, parallel bulk load), the exact lookup path and the precise meaning of every software counter, then each clean-room control in turn: the affine `LinearModel`, the NFL-style flow transform and its stand-in trainer, the bypass rule, the CSV-style smoothing (`smooth_cdf`) with a full derivation of the O(1) closed-form loss, how the index consumes virtual points, the per-region selector (`Fusion::Auto`), the learned root (`Index::fit_root`, including the sentinel-fence bug), the hardness tool (least squares, LIPP's FMCD, PGM's optimal PLA) and the AIDB scorer. Every mechanism comes with a worked numeric example that was actually computed (Python, or a tiny C++ program compiled against the headers; scripts are under `results/aidb/guide_deep/scratch/`). Everything here is a clean-room CONTROL inside the SCALE-LI experimental map, written from the papers' descriptions; nothing is a reproduction of NFL/AFLI, CSV or the AIDB benchmark. Notation: keys k₁ < … < kₙ (uint64), rank r(kᵢ) = i − 1 ∈ {0..n−1}, linear model f(k) = w·φ(k) + b with feature φ, slot s(kᵢ) = rank + number of virtual points before kᵢ, budget λ = α·n, tail conflict degree D₉₉, region = 4,096 keys, block = 128 keys.

Code paths are relative to `S = /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli`. Line numbers refer to the files as read on 2026-09-22.

---

## 1. The SCALE-LI experimental map (the host)

### 1.1 What it stores

The index maps `Key = uint64` to `Value = uint64` (`Record = pair<Key,Value>`) [types.hpp:11-13]. Input rows are canonicalised (stable sort by key, last duplicate wins) [types.hpp:38-47]. The structure is two levels:

- `Index` owns `vector<unique_ptr<Region>> regions_`, a `Config`, a root model and a slot→region table [index.hpp:327-334].
- `Region` owns `low_fence` (the smallest key it can hold), `vector<BlockDescriptor> blocks`, a byte `arena` holding all encoded key blocks back to back, `vector<Value> values` (uncompressed, one per key, in rank order), a sorted `delta` of pending writes, two `LinearModel`s (`rank_model`, `byte_model`), heat statistics and the learnability diagnostics (`flow_used`, `tail_conflicts_raw/flow`, `virtual_points`, `virtual_features`, `rank_sse_before/after`, `smoothing_ns`, `transform_ns`, `choice`, `cost_none`, `cost_selected`) [index.hpp:87-104].

`Config` defaults: `region_keys=4096, block_keys=128, delta_limit=64, restart_interval=16, routing=Byte, policy=MinBytes, min_saving_fraction=0.05`, learnability knobs `flow=nullptr, flow_bypass=true, flow_min_gain=0.10, virtual_alpha=0, relearn_on_compaction=false, fusion=Manual, root=Binary, root_alpha=0, flow_cost=4.0, build_threads=1` [index.hpp:43-65]. `validate()` enforces `virtual_alpha ∈ [0,1)`, `flow_min_gain ∈ [0,1]`, `root_alpha ∈ [0,64)`, `1 ≤ block_keys ≤ region_keys ≤ 2²⁴`, `delta_limit ≤ 2·region_keys` [index.hpp:66-77]. The benchmark binary maps these to `--region-keys --block-keys --routing rank|byte|binary --policy raw|min_bytes|... --flow --flow-bypass --flow-min-gain --virtual-alpha --relearn --fusion manual|auto --flow-cost --root binary|model --root-alpha --build-threads` [benchmark.cpp:189-193]. The ten uniform-sweep variants are `sorted_vector` (a different index, `--index sorted_vector`), `raw_rank` (`--routing rank --policy raw`), `packed_rank` (`--routing rank --policy min_bytes`), `packed_byte` (`--routing byte --policy min_bytes`, the only variant with byte routing), and six learnability variants that add flags on top of `packed_rank`: `packed_rank_flow` (`--flow W --flow-bypass 1`), `packed_rank_flow_forced` (`--flow-bypass 0`), `packed_rank_vp10` (`--virtual-alpha 0.1`), `packed_rank_flow_vp10` (flow + bypass + α = 0.1), `packed_rank_fusion_auto` (flow + α = 0.1 + `--fusion auto`), `packed_rank_flow_costsel` (flow + `--fusion auto`, no virtual points) [results/aidb/sweep/environment.json config.variants; results.jsonl has these 10 variant names]. So every scaleli variant except `packed_byte` uses rank routing, every packed variant uses `min_bytes`, and only `raw_rank` uses `raw`. The root sweeps (`results/aidb_final`, `results/aidb_fullscale`) add `packed_rank_root_{raw,flow,vf4,fusion}`, `packed_rank_vp10_root_fusion`, `packed_rank_root_vf004`, `packed_rank_vp10_root_vf004` (`--root model`, `--root-alpha 4` or `0.04`) [results/aidb_final/config.json; results/aidb_fullscale/config*.json].

### 1.2 Regions and blocks; bulk load

`bulk_load` cuts the canonical row vector into consecutive slices of `region_keys` (so 2,000,000 keys → ⌈2,000,000/4096⌉ = 489 regions, the last one holding 2,000,000 − 488·4096 = 1,152 keys; 200,000,000 keys → 48,829 regions). Region j gets `low_fence = rows[j·4096].first`, except region 0 whose fence is the sentinel 0 [index.hpp:432]. Each region is built independently by `Region::rebuild`; with `build_threads = W > 1` a static interleaved partition (thread w builds regions w, w+W, …) runs in a thread pool, exceptions are captured and rethrown, and the result is bit-identical to the serial build because each rebuild touches only its own `Region` plus read-only config/flow [index.hpp:433-441]. Test `fusion_auto` proves determinism: `dump_layout` output for `build_threads ∈ {1, 4}` must be byte-identical [test_main.cpp:143-149]. After the regions exist, `fit_root()` runs [index.hpp:443]. Timings (`build_ns`) are the wall clock of the parallel build [benchmark.cpp:65]; the AIDB sweeps used `--build-threads 16` [results/aidb/sweep/environment.json config.common].

Inside `rebuild` [index.hpp:199-317], keys are cut into blocks of `block_keys` (128; 4,096/128 = 32 blocks per full region). For each block:

1. `encode_block(part, Codec::Raw)` is always produced first [index.hpp:209].
2. Under `Policy::MinBytes` (the default, called "packed"), the three packed codecs `For`, `Delta`, `Linear` are tried; a candidate replaces the current choice iff it saves at least `min_saving_fraction` = 5 % against RAW (`size ≤ raw·0.95`) AND has a strictly smaller score (= its byte size under MinBytes; under `Policy::Smooth`/`Adaptive` the score is |size − target| with a per-block byte target computed from the block's key-span share) [index.hpp:211-226]. The target as coded [index.hpp:205, 216-219]:

```
span   = keys.back() − keys.front()                       (region key span; 1 if one key)
left   = 0 if first block, else ((keys[begin−1] − keys[0]) + (keys[begin] − keys[0]))/2
right  = span if last block, else ((keys[end−1] − keys[0]) + (keys[end] − keys[0]))/2
target = (16·⌈n/block_keys⌉ + 8·n) · (right − left)/max(1, span) · smooth_scale
```
i.e. the region's RAW byte budget (16-byte headers + 8 bytes per key) is shared out in proportion to the key-span each block covers, measured between the midpoints to its neighbours, so dense intervals get smaller byte targets. Two guards: the whole codec trial is skipped when the region is flagged `hot` by the `Adaptive` policy (`!hot` at index.hpp:211: a hot region keeps Raw blocks, the "raw bypass"), and `validate()` additionally requires `flow_cost ≥ 0` and finite [index.hpp:68].
3. `Policy::Raw` skips step 2; `Policy::Forced` uses `forced_codec` [index.hpp:210].
4. A `BlockDescriptor{first, last, rank_begin, count, offset, length, codec, slot_begin = rank_begin}` is appended and the bytes go to the arena [index.hpp:227-229].

The four codecs [codec.hpp:14,83-130]. Every block starts with a 16-byte header of which only three fields are written for every codec: byte 0 = codec tag, bytes 4-7 = count (LE32), bytes 8-15 = first key (LE64) [codec.hpp:88-90]. Byte 1 (bit width w) is written only by `For` and `Linear` [codec.hpp:97,111], bytes 2-3 (restart interval, LE16) only by `Delta` [codec.hpp:119], and `Linear` stores 16 extra bytes after the header (`span` at bytes 16-23, `lo` at 24-31) so its bit-packed payload starts at byte 32 = bit 256, whereas `For`'s starts at byte 16 = bit 128 [codec.hpp:112-116,99]:

| Codec | Payload | Random access `key_at(i)` |
|---|---|---|
| `Raw` | 8 bytes per key, little endian | direct [codec.hpp:136-138] |
| `For` (frame of reference) | `w = bit_width(last − first)` bits per key, values `k − first` bit-packed [codec.hpp:96-101] | `get_bits` at bit 128 + i·w [codec.hpp:139-143] |
| `Linear` | residual `r_i = (k_i − first) − interpolation(span, i, n)` with exact `__int128` interpolation `⌊span·i/(n−1)⌋`; stores `span`, `lo`, and `w = bit_width(hi − lo)` bits per key at bit 256 + i·w; falls back to Raw if residuals exceed int64 [codec.hpp:102-117,80-82] | recompute interpolation + residual [codec.hpp:144-150] |
| `Delta` | restart every 16 keys: an 8-byte absolute key, then LEB128 varints of consecutive differences; a 4-byte offset table per restart group [codec.hpp:118-127] | jump to the restart group, then decode ≤ 15 varints [codec.hpp:151-161] |

`key_at` increments `key_at_calls`, `decoded_keys` and `codec_bytes_examined` [codec.hpp:134-160]. Test `codecs()` round-trips every codec over 12 sizes × 7 key shapes (including keys at 2⁶⁴−1 and all-equal keys) [test_main.cpp:13-39].

### 1.3 Fences, delta, compaction

- **Fences.** Region-level: `low_fence` (region j owns keys in [low_fence_j, low_fence_{j+1})); `validate()` throws "fence ownership" if violated [index.hpp:503-507]. Block-level: `BlockDescriptor::first/last` are exact stored keys; the block search compares against `last` [index.hpp:135].
- **Delta.** Writes never touch the arena. `upsert`/`erase` locate the region, then `set_delta` inserts or overwrites a `{key, value, deleted}` entry in a sorted vector [index.hpp:164-168,449-459]. `find` checks the delta first by binary search (`delta_probes`) [index.hpp:113-117,156-157].
- **Compaction.** When `delta.size() ≥ delta_limit` (64), `compact(j)` materialises base+delta (`scan_into` over all blocks, merging deletions), and either rebuilds the region in place or, if more than `2·region_keys` rows, splits it into two regions (the right one gets `low_fence = rows[mid].first`) and refits the root [index.hpp:402-425]. The previous region is passed as `previous` so the rebuild can reuse the flow decision and re-place the virtual points (section 6). `maintain()` compacts every region [index.hpp:467].

### 1.4 The lookup path, step by step

`Index::find(k)` [index.hpp:445-447]:

```
1. j = locate_region(k)                         -> root_probes
2. r = regions_[j]; r.observe(false)            (heat bookkeeping)
3. Region::find(k):
   a. d = delta_lower(k)                        -> delta_probes; hit in delta returns immediately
   b. b = locate_block(k):
        i.   p = predicted_block(k)             -> transform_calls (if flow), coordinate_probes
        ii.  exponential bracket + binary correction on blocks[].last
                                                -> fence_probes, block_routes, correction_distance
   c. binary search inside block b by key_at()  -> key_at_calls, decoded_keys, codec_bytes_examined
   d. return values[b.rank_begin + lo] if key matches
```

**Step 1 — `Index::locate_region`** [index.hpp:348-357]. With `Root::Binary` (or a root model that was not accepted) it is a plain binary search over `regions_[m]->low_fence <= k`, incrementing `root_probes` once per comparison [index.hpp:352-354]; the answer is the last region whose fence ≤ k. With an accepted `Root::Model` it calls `locate_from_prediction(ok, root_model_.predict_x(root_feature(k)), n, &root_slot_to_region_)` [index.hpp:356]: the prediction y is turned into a starting region p exactly as follows [index.hpp:341-343]:

```
with a slot table of size m+1 = n_regions + virt:   slot = 0 if y ≤ 0;  m if y ≥ m;  else ⌊y⌋ (size_t truncation)
                                                    p = table[slot]
without a table:                                    p = 0 if y ≤ 0;  n−1 if y ≥ n−1;  else ⌊y⌋
```
The table is filled at fit time so that every slot in [sm.slot[j], sm.slot[j+1]) maps to region j (the last region owns the slots up to n + virt) [index.hpp:386-387]: a virtual fence inserted between fences j and j+1 therefore belongs to region j. Numeric instance: the fb_uniform root chose raw + 1 virtual fence over 489 regions, so the table has 490 `uint32` entries = 1,960 B, which is exactly the `slot=1960` metadata delta of `root_vf4` in the cost-story table [results/aidb_final/sweep/results.jsonl fb_uniform packed_rank_root_vf4 root_virtual = 1; results/aidb/verification/coststory/coststory.txt metadata table]. Note the two search semantics: the root's predicate is `ok(i) = low_fence_i ≤ k` and the answer is `last_true` (the largest true index), whereas the block search's predicate is `less(i) = last_i < k` and the answer is `lower` (the smallest false index) [index.hpp:337-339,135-137]. Then `ok(p)` is tested; if true, exponential probing upwards (step 1, 2, 4, …) until `ok` fails, then `last_true` binary search; if false, exponential probing downwards, then `last_true` [index.hpp:340-347,337-339]. Every `ok(i)` call increments `root_probes` [index.hpp:350].

**Step 3.b.i — `Region::predicted_block`** [index.hpp:118-129]. `f = feature(k)` (either `rank_model.normalized(k)` = (k − origin)/span, or `(*flow)(k)` when `flow_used`, which counts a `transform_calls`) [index.hpp:106; transform.hpp:65]; `y = rank_model.predict_x(f)` under `Routing::Rank` (`byte_model` under `Routing::Byte`); then a binary search over the blocks' `slot_begin` (rank routing) or `offset` (byte routing) for the last block whose boundary ≤ y, each comparison incrementing `coordinate_probes` [index.hpp:122-127]. The result is a *hint*, never trusted.

**Step 3.b.ii — `Region::locate_block`** [index.hpp:133-154]. `less(i) := blocks[i].last < k` and every call increments `fence_probes` [index.hpp:135]. `Routing::Binary` simply binary-searches all blocks [index.hpp:139]. Otherwise, with prediction p:

```
if less(p):                       # answer is to the right
    lo = p+1; hi = lo; step = 1
    while hi < n and less(hi): lo = hi+1; step = min(n, 2*step); hi = min(n, p+step)
    out = lower(lo, hi < n ? hi+1 : n)     # binary search in the bracket
elif p == 0 or less(p-1): out = p         # exact hit
else:                             # answer is to the left
    hi = p; step = 1; lo = p - step
    while lo > 0 and not less(lo): hi = lo; step = min(n, 2*step); lo = p > step ? p-step : 0
    if less(lo): lo += 1
    out = lower(lo, hi+1)
correction_distance += |out - p|; max_correction_distance = max(...); block_routes += 1
```
[index.hpp:140-153]. The exponential phase costs O(log d) probes for a prediction that is d blocks off, then a binary search inside a bracket of ≤ 2d blocks: total O(log d), and correctness never depends on the model (test `adversarial_routes` sets `slope=0, intercept=±1e15` and still requires exact answers [test_main.cpp:77-79]).

**Step 3.c — in-block search** [index.hpp:159-161]: lower-bound binary search over `count ≤ 128` keys with `key_at` [index.hpp:160], then one more `key_at` for the hit check `if(lo<b.count && key_at(bytes(b),lo,s)==k)` [index.hpp:161]. The loop `while(lo<hi){m=lo+(hi−lo)/2; if(key_at(m)<k) lo=m+1; else hi=m;}` over 128 keys makes 7 or 8 calls depending on where the answer lands (simulated over all 129 outcomes: 127 cost 7, 2 cost 8 [scratch/fix05_checks.py]), so a successful lookup costs **8 or 9** `key_at` calls, never fewer than 8; the AIDB runs measure `key_at_calls/op = 8.016` on every variant [results/aidb_final/sweep/results.jsonl, fb_uniform packed_rank, seed 11].

### 1.5 The counters, precisely

`QueryStats` [types.hpp:17-24] is filled only when a non-null pointer is passed; the benchmark collects them in a separate pass (pass 4) so they never contaminate timing, and it checks that the instrumented replay produces the same result checksum as the timed replay [benchmark.cpp:111-115].

| Counter | Incremented at | Meaning |
|---|---|---|
| `root_probes` | `locate_region`: binary search comparison [index.hpp:353] or each `ok(i)` of the learned root [index.hpp:350] | number of *region fence comparisons* (`low_fence <= k`) per operation |
| `coordinate_probes` | `predicted_block` binary search over `slot_begin`/`offset` [index.hpp:124] | comparisons of the model's real-valued prediction y against block boundaries; the lower-bound loop of index.hpp:123-127 over 32 boundaries has 33 possible outcomes and costs 5 or 6 comparisons (31 outcomes cost 5, 2 cost 6; the 9-block last region costs 3 or 4) [scratch/fix05_checks.py], which is why the measured mean is 5.029, slightly above 5 [results/aidb_final/sweep/results.jsonl fb_uniform packed_rank seed 11]. It never grows with the model error |
| `fence_probes` | `locate_block::less` [index.hpp:135] | comparisons of the query key against a block's exact `last` key during exponential bracketing + binary correction; the *first* probe is always made (checking `less(p)`), so a perfect prediction costs 1 or 2 fence probes |
| `correction_distance` | `locate_block` after the search [index.hpp:151] | Σ over operations of |found block − predicted block| (`max_correction_distance` is the max) |
| `block_routes` | same place | number of `locate_block` calls (denominator for the mean correction) |
| `transform_calls` | `FlowTransform::transform` [transform.hpp:65] | number of flow evaluations (region feature and/or root feature) |
| `delta_probes` | `delta_lower` [index.hpp:115] | comparisons in the delta binary search |
| `key_at_calls`, `decoded_keys`, `codec_bytes_examined` | `key_at`/`get_bits`/`get_varint` [codec.hpp:134,137,142,149,154,158,52,66] | decode work in the block search |
| `blocks_decoded_for_scan` | `scan_into` [index.hpp:177] | blocks materialised by scans |

**Determinism.** All counters are functions of the built structure and the query trace only; they contain no timing. Because the build is deterministic (section 1.2) and the trace is seeded, the same command yields identical counters (the benchmark compares the throughput digest with the instrumented replay's digest and throws "instrumentation changes results" on a mismatch [benchmark.cpp:115]; the latency-replay comparison at benchmark.cpp:96 runs only when `--latency` is set, and every AIDB sweep passes `--latency 0` [results/aidb/sweep/environment.json config.common; every `command` in results.jsonl], so in these sweeps only the two digests are compared). The cost-story verification confirms that root variants differ from `packed_rank` *only* in `root_probes` and `transform_calls` (max abs diff per lookup on `fence_probes`, `coordinate_probes`, `key_at_calls`, `correction_distance` = ±0.000) [results/aidb/verification/coststory/coststory.txt]. What is NOT deterministic: `smoothing_ns`, `transform_ns`, `build_ns`, `throughput_ops_s` (wall-clock).

**Worked example (counters on a toy index).** `scratch/index_demo.cpp` builds 32 keys (0,10,…,230 then 240 + 1000·(i−24)) with `region_keys=16, block_keys=4, routing=Rank, policy=Raw`: 2 regions, 8 blocks. Measured:

```
find(0)    -> root_probes=2 coordinate_probes=3 fence_probes=1 correction=0 key_at=4
find(70)   -> root_probes=2 coordinate_probes=2 fence_probes=2 correction=0 key_at=3
find(230)  -> root_probes=1 coordinate_probes=2 fence_probes=2 correction=0 key_at=3
find(2240) -> root_probes=1 coordinate_probes=2 fence_probes=3 correction=1 key_at=3
find(7240) -> root_probes=1 coordinate_probes=2 fence_probes=2 correction=0 key_at=3
```
Reading: 2 regions → the fence binary search makes 1 or 2 comparisons; 4 blocks per region → 2-3 coordinate probes; key 2240 sits in the sparse tail of region 1 where the linear model misses by one block, so one extra `fence_probe` and `correction_distance = 1`. On the real 2M fb sample, `packed_rank` measures per lookup `root_probes 8.956` (≈ log₂ 489 = 8.93), `coordinate_probes 5.029`, `fence_probes 2.600`, `correction_distance 0.257`, `key_at_calls 8.016` [results/aidb_final/sweep/results.jsonl, fb_uniform, seed 11].

---

## 2. `LinearModel` (model.hpp)

```
struct LinearModel { Key origin=0; uint64 span=1; double slope=0, intercept=0; ... }
normalized(k) = (k − origin)/span      if k ≥ origin, else −(origin − k)/span      [model.hpp:15-18]
predict_x(x)  = fma(slope, x, intercept), 0 if non-finite                         [model.hpp:19-22]
```

`fit_xy(keys, x, targets)` [model.hpp:26-42] sets `origin = keys.front()`, `span = max(1, keys.back() − origin)` (integers subtracted *before* conversion to double, so precision near 2⁶⁴ is kept), then a single-pass centred (Welford) accumulation in `long double`:

```
for i: n=i+1; dx = xᵢ − mx; dy = yᵢ − my; mx += dx/n; my += dy/n; xx += dx·(xᵢ − mx); xy += dx·(yᵢ − my)
slope = xy/xx (0 if xx ≤ 0);  intercept = my − slope·mx
```
which is exactly closed-form OLS: w = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)², b = ȳ − w·x̄. `fit(keys, targets)` is the same with x = normalized(k) [model.hpp:43-60].

**Targets used.** In `rebuild`, `rank_model.fit_xy(keys, x, slots)` and `byte_model.fit_xy(keys, x, positions)` [index.hpp:312]: the rank model's target is the *slot rank* (= rank when no virtual points; rank + #virtual points before the key otherwise), the byte model's target is the byte coordinate of the key's first encoded bit inside the arena (`desc.offset + key_positions[j]`) [index.hpp:228]. Features x are normalized keys or flow outputs (section 3). The hardness tool uses plain rank targets (section 9).

**Worked example.** Keys/features {1,2,3,10,11,12}, ranks {0..5}: x̄ = 6.5, ȳ = 2.5, Σ(x−x̄)² = 125.5, Σ(x−x̄)(y−ȳ) = 44.5 → w = 0.354581673, b = 0.195219124; `predict_x(10) = 3.741` (true rank 3). With normalized features (origin 1, span 11, φ = {0, 0.0909, 0.1818, 0.8182, 0.9091, 1}) the fit is w = 3.900398, b = 0.549801: the same line in different units [scratch/index_demo.cpp, block A].

---

## 3. NFL-style transform (transform.hpp) and the stand-in trainer (tools/train_flow.py)

### 3.1 What the paper does and what we re-implement

NFL [Wu et al., PVLDB 2022, §3.2] transforms the keys with a small numerical normalizing flow so that the transformed distribution is near-uniform, then builds AFLI on the transformed keys. The transformation pipeline is Algorithm 3.1: scaled min-max normalisation μ = min(X), σ = (max(X) − min(X))/θ, x_norm = (x − μ)/σ (line 2-4); an encoder that expands x into d features by repeatedly splitting integral and fractional parts (lines 5-17); the flow F on the d-dimensional vector (line 18); a decoder that sums the output vector back to one number (lines 19-22) [NFL §3.2.1, Alg. 3.1]. The deployed flows are small ("2H2L" = hidden 2, 2 layers, 8 parameters, 169.53 ns per key at batch size 1 and 7.29 ns at batch 2048 on their MKL implementation) [NFL Table 2]. That the network is a tanh network is NOT stated in the paper (the word does not occur in the PDF text); it comes from `transform.hpp`'s description of the official repository's C++ side ("evaluates that file as a small tanh network on Intel MKL") [transform.hpp:6-7; unverified against the paper: activation not stated there]. The paper's own training statement is split: §3.2.2 says only that the flow "takes time to train (about 38 seconds)" [NFL §3.2.2, p. 6]; the architecture is in the experimental setup: "we train a modified B-NAF in PyTorch. The B-NAF is set to two layers, two input dimensions, two hidden dimensions, and a normal distribution with the variance of 10¹⁶ as the latent distribution", trained on an RTX 3080 (10 GB) with "We only sample 10% bulk-loaded keys for three times to train the NF. We set the batch size to 256" [NFL §4.1, p. 9]. Our stand-in for comparison: 4,096 uniformly sampled keys of the 2M sample, 200 Adam steps, lr 0.05, 2.39 s on CPU [results/aidb/flows/fb_training.json; train_flow.py:119-127], and the scale factor θ (Alg. 3.1) is our `--shifts 64`, whereas the trainer's help text records the author default of 10⁶ for 200M keys [train_flow.py:119; unverified: author code not in the workspace]. The official repository (github.com/luffy06/NFL, GPL-3) exports a text weight file that its C++ side evaluates [transform.hpp:3-10]. We copy no code: `transform.hpp` re-implements only *inference* of that text format, and `train_flow.py` is a stand-in trainer that writes the same format [transform.hpp:1-19; train_flow.py:1-14]. Differences stated in the header: no MKL, no batching; the transformed value is used **only as the model feature** (records stay in key order, so fences/scans/exactness hold even for a non-monotone network, whereas NFL sorts by transformed key); keys are converted to double before the transform (precision above 2⁵³ is lost in the feature only) [transform.hpp:12-19].

### 3.2 The weight file ("2D2H2L")

`FlowTransform::load` [transform.hpp:35-53] reads: `in_dim hidden layers` then `mean var`, then for each layer `rows cols` followed by rows×cols weights row-major; shapes are checked (`layer 0: in_dim×hidden`, last layer: `hidden×in_dim`), `in_dim ∈ {1,2}`, `layers ≤ 16`, `hidden ≤ 64`, `var > 0`. "2D2H2L" = input dimension 2, hidden 2, 2 layers, no bias. `save` writes the same layout with 17 significant digits [transform.hpp:54-61]. Whether the official author files carry exactly this header layout is stated in the header comment but not independently checked here [unverified: no author weight file in the workspace].

`results/aidb/flows/fb_2D2H2L.txt` (trained by `tools/train_flow.py` on the fb 2M uniform sample with `--sample 4096 --steps 200 --monotone`, seed 1000000007, 2.39 s, best NLL 4.182016 [results/aidb/flows/fb_training.json, fb_training.json.command.json]):

```
2	2	2                                    in_dim=2 hidden=2 layers=2
15162980.0	1207648189.703125            mean, var
2	2                                        W0 (2×2): row 0 = weights of feature x, row 1 = weights of the fractional feature
0.0058556031582764	0.0058592806847908
0.0000000000000000	0.0000000000000000        <- --monotone zeroes row 1
2	2                                        W1 (2×2)
1.2798795760698392	0.8812895293543010
1.4785700705830542	1.0331745748154413
```
`mean` = the smallest of the 4,096 *sampled* training keys (not the dataset minimum: the sample minimum is 97,995 [scratch/flow_demo.py]) because `train()` sets `mean = float(keys[0])` on the training subset; `var = max(1, (train_keys[-1] − train_keys[0])/shifts)` with `shifts = 64` [train_flow.py:89,119]. So x = (key − mean)/var spans ≈ [0, 64] on the training keys (Algorithm 3.1's θ is our `--shifts`).

### 3.3 The feature and the network, exactly as coded

`transform(key)` [transform.hpp:64-80]:

```
x   = (key − mean) / var
a   = [x, x − ⌊x⌋]                          (in_dim = 2: the repository's "partition" encoder as described in transform.hpp:68)
for each layer l with matrix W_l (rows×cols):
    b_c = Σ_r a_r · W_l[r][c]
    a   = tanh(b)  for all but the last layer; a = b on the last layer
z   = Σ_c a_c                                ("sum" decoder); 0 if non-finite
```
**Encoder: paper versus code.** NFL Algorithm 3.1 with d = 2 does *not* produce [x, x − ⌊x⌋]. Its lines 6-8 set x_int = INT(x), x_float = x − INT(x) and add x_int to the vector; the loop of lines 9-14 runs d − 2 = 0 times; line 15 adds x_float [NFL Alg. 3.1, p. 6]. So the paper's 2-D vector is [INT(x_norm), x_norm − INT(x_norm)] (integral part, fractional part), whose *sum* equals x_norm but whose first coordinate is a 64-step staircase over the key range (x_norm ∈ [0, 64] with θ = 64). Our code follows transform.hpp:68's reading of the official repository's "partition" encoder, [x, x − floor(x)], which keeps the full value in the first slot [transform.hpp:68-69; unverified: the official encoder is not in the workspace, only the header author's description of it]. The consequence matters for `--monotone`: with W0 row 1 zeroed only the first feature carries information, so keeping x (rather than INT(x)) is what makes z continuous and strictly increasing in the key; with the paper's INT(x) in the first slot a monotone 2D2H2L flow would be a 64-step staircase and the tail conflict degree would be catastrophic. For 2D2H2L the code computes z(k) = Σ_c Σ_h tanh(u_h)·W1[h][c] = Σ_h v_h·tanh(u_h) with u_h = W0[0][h]·x + W0[1][h]·(x − ⌊x⌋) and v_h = Σ_c W1[h][c]. With `--monotone`, W0[1][·] = 0 so u_h = W0[0][h]·x and z is strictly increasing in the key. In the index, φ(k) = z(k) replaces the normalized key when the region uses the flow [index.hpp:106].

**Worked example (fb, by hand with Python; `scratch/flow_demo.py`).** v₀ = 1.27987958 + 0.88128953 = 2.16116911; v₁ = 1.47857007 + 1.03317457 = 2.51174465. Take the sample's median key k = keys[1,000,000] = 38,737,396,981:

```
x   = (38737396981 − 15162980) / 1207648189.703125 = 32.064167636867
frac= 0.064167636867 (unused: its weights are 0)
u₀  = 0.0058556031582764 · x = 0.187755041282 ;  tanh(u₀) = 0.185579469285
u₁  = 0.0058592806847908 · x = 0.187872958109 ;  tanh(u₁) = 0.185693322595
layer 1 pre-activations: [0.185579·1.279880 + 0.185693·1.478570, 0.185579·0.881290 + 0.185693·1.033175]
                       = [0.512079961573, 0.355402862763]   (no tanh on the last layer)
z   = 0.512079961573 + 0.355402862763 = 0.867482824336
```
Compare the raw feature φ_raw = (k − k_min)/(k_max − k_min) = 0.5010729. The sample's first key (97,995) maps to x = −0.01247 → z = −0.000341; the last key (77,308,811,965) to x = 64.0034 → z = 1.674188. Sampling every 1,000th key gives z ∈ [−0.000341, 1.673468], strictly increasing (`unordered_transformed_pairs = 0` in the training report). The pre-activations are at most 0.3750 (u₁ = 0.3750142 at the last key, u₀ = 0.3747788 [scratch/flow_demo.py; scratch/fix05_checks.py]), i.e. in the near-linear part of tanh (tanh(0.375) = 0.3584, slope 1 − tanh² = 0.872): this flow is a gentle concave bend, not a strong re-shaping, which is why the 2M-sample tail conflict degree is 8 before and 8 after the transform [results/aidb/flows/fb_training.json].

### 3.4 `tail_conflict_degree` (transform.hpp:90-106) — NFL Definitions 3.1/3.2

Paper: the conflict degree of position j under model M is D_j = |{xᵢ : M(xᵢ) = j}| (Def. 3.1, eq. 5); over the m positions with D_j > 0, with tail percent γ = 0.99 and t = INT(m·γ), the tail conflict degree D^γ is the t-th largest conflict degree (Def. 3.2); The paper lists its two uses verbatim: "It could be useful in: 1) determine the execution of flow (see Section 3.2.2); 2) determine the capacity threshold of a node (see Section 3.3.1)" [NFL §3.1, Def. 3.2 → §3.2.2 and §3.3.1]. The paper example: 1000 positions, t = INT(1000 × 0.99) = 990, "the 990-th larger conflict degree", where "INT represents the flooring operation" [NFL Def. 3.2].

Our function on a sorted feature vector x (n ≥ 2; if x_n ≤ x₁ it returns n − 1):

```
1. OLS slope of rank on x (Welford); if slope ≤ 0, slope = n/(x_n − x₁)
2. intercept = −slope·x₁ + 0.5                      (first key maps near position 0, as NFL rescales)
3. max_size = max(1, min(⌊1.5·n⌋, max(1, ⌊slope·x_n + intercept⌋ + 1)))   ("1.5·n slots")
4. p_i = clamp(⌊slope·x_i + intercept⌋, 0, max_size − 1)                 (floor)
5. counts = run lengths of equal consecutive p_i  (these are exactly the m positions with D_j > 0)
6. sort counts ascending; idx = max(0, ⌈0.99·m⌉ − 1); return counts[min(idx, m−1)] − 1
```
Two conventions to keep in mind: (i) sorting ascending and taking index ⌈0.99 m⌉−1 selects the 99th-percentile *smallest*, which is the same element as the ⌊0.01 m⌋+1-th largest — the paper's wording "t-th larger" with t = INT(0.99 m) is ambiguous between the two readings, ours takes the upper tail (the one that "indicates the upper bound of the conflicts for most positions") [NFL Def. 3.2]; (ii) the final "− 1" reports *extra* keys per position, so D₉₉ = 0 means no conflict (a uniform sequence gives 0: test `learnability` [test_main.cpp:100]).

**Worked example** (`scratch/tcd_demo.py`, same arithmetic as the C++): x = {0,1,…,9, 9.1, 9.2, 9.3, 9.4, 9.5, 20, 30, 40, 50, 60}, n = 20. OLS slope = 0.287944413, intercept = 0.5, ⌊slope·60 + 0.5⌋ + 1 = 18 < ⌊1.5·20⌋ = 30 → max_size = 18. Positions: [0,0,1,1,1,1,2,2,2,3,3,3,3,3,3,6,9,12,14,17]. Run lengths: {2,4,3,6,1,1,1,1,1} → sorted {1,1,1,1,1,2,3,4,6}, m = 9, idx = ⌈8.91⌉ − 1 = 8 → counts[8] − 1 = 6 − 1 = **5**. The C++ returns 5 as well [scratch/index_demo.cpp, block E]. **Which percentile is meant.** The paper takes t = INT(m·γ) (floor) and the "t-th larger" element; the code takes index ⌈m·γ⌉ − 1 in ascending order [transform.hpp:104-105], i.e. the ⌈m·γ⌉-th smallest. For m = 1000 both give the 990th smallest (t = 990 1-based ⇔ 0-based 989) and coincide; for m·γ not an integer they differ by one element. On this m = 9 example the readings diverge: code ⌈8.91⌉ = 9th smallest = 6 → **5**; paper's t = INT(8.91) = 8 read as 8th smallest = counts[7] = 4 → **3**; read as 8th *largest* (descending {6,4,3,2,1,1,1,1,1}) = 1 → **0** [scratch/fix05_checks.py]. The code deliberately takes the upper-tail (ascending, ceil) reading because the definition says the quantity "indicates the upper bound of the conflicts for most positions" [NFL §3.1]; with m in the hundreds per region the one-element difference is immaterial, but say so if asked.

Real numbers (per-region means over 489 regions, `tail_conflicts_raw_mean` / `tail_conflicts_flow_mean`, variant `packed_rank_flow`, seed 11) for all ten 2M uniform samples: wise 3.03 / 3.03, fb 8.448 / 8.464, genome 4.23 / 4.23, books 3.28 / 3.28, planet 7.66 / 7.65, libio 3.37 / 3.36, stack 3.01 / 3.01, covid 3.16 / 3.14, history 3.10 / 3.10, osm 36.16 / 36.98 [results/aidb/sweep/results.jsonl, learnability; scratch/fix05_checks.py]. The transform changes the mean tail conflict degree by less than 1 % everywhere except osm (+2.3 %, worse). The Python trainer's whole-sample D₉₉ for fb is 8 → 8 [results/aidb/flows/fb_training.json].

### 3.5 The stand-in trainer (tools/train_flow.py)

Not a BNAF. It maximises the likelihood of the keys under z ~ N(0,1) through the 1-D change-of-variables formula p_X(x) = p_Z(z(x))·|dz/dx|, i.e. it minimises

  NLL = (1/n) Σ_keys [ ½·z² − log(dz/dx) + barrier/(dz/dx) ]   (constant ½log 2π and log var dropped) [train_flow.py:48-55]

with, for the 2D2H2L shape, z = Σ_h v_h·h_h, h_h = tanh(u_h), s_h = 1 − h_h², and dz/dx = Σ_h s_h·a_h·v_h where a_h = W0[0][h] + W0[1][h] (since d(x − ⌊x⌋)/dx = 1 almost everywhere) [train_flow.py:31-40]. If dz/dx ≤ 0 the loss switches to ½z² + 50 − 10·(dz/dx) (a linear push-back) [train_flow.py:55]. The `barrier/(dz/dx)` term (default 1e-3) keeps the derivative positive, i.e. keeps the transform monotone. Gradients are analytic (∂z/∂W0[i][h] = v_h s_h f_i, ∂(dz/dx)/∂W0[i][h] = v_h((−2h_h s_h f_i)a_h + s_h), ∂z/∂W1[h][c] = h_h, ∂(dz/dx)/∂W1[h][c] = s_h a_h) [train_flow.py:57-66]. **Scaled parametrisation:** the two x-feature weights are learned as θ = w·shifts (so `scale0 = [1/shifts, 1/shifts, 1, 1]`) to prevent a single Adam step from saturating the tanh, "the official code relies on BNAF weight normalisation for this" [train_flow.py:90-93,105-109]. **Adam** with β₁ = 0.9, β₂ = 0.999, ε = 1e-8, lr 0.05, 200 steps, bias-corrected; the weights with the best loss seen are kept [train_flow.py:98-111]. **`--monotone`** zeroes W0 row 1 and its gradient, "deviation from NFL, which keeps them" [train_flow.py:67,96,124]. Initialisation is seeded (`abs(gauss(1.5, 0.5))/shifts` for row 0, `abs(gauss(0.1, 0.05))` for row 1, `abs(gauss(1, 0.2))` for W1) [train_flow.py:93-95]. Training keys: `--sample 4096` uniformly sampled from the de-duplicated key set with the same seed [train_flow.py:127-128]. The report JSON records `best_nll`, `unordered_transformed_pairs` on *all* keys, and D₉₉ raw vs transformed [train_flow.py:132-139]. The pipeline trains one flow per dataset on the 2M sample (`step_flows`) [aidb_pipeline.py:435-447].

**Worked example (the loss on one key; same fb flow and median key as 3.3; `scratch/fix05_checks.py`).** v = [2.16116911, 2.51174465] (row sums of W1), a_h = W0[0][h] + W0[1][h] = [0.0058556032, 0.0058592807] (row 1 is zero), h = tanh(u) = [0.185579469, 0.185693323], s_h = 1 − h² = [0.96556026, 0.96551799]:

```
dz/dx = Σ_h s_h·a_h·v_h = 0.96556026·0.0058556032·2.16116911 + 0.96551799·0.0058592807·2.51174465 = 0.026428660
per-key NLL term = ½z² − log(dz/dx) + 10⁻³/(dz/dx)
                 = ½·0.867483² + (−log 0.026428660) + 0.001/0.026428660
                 = 0.376263 + 3.633306 + 0.037838 = 4.047407
```
for comparison `best_nll` = 4.182016 is the mean of this term over the 4,096 training keys [results/aidb/flows/fb_training.json]; at the last key (77,308,811,965: z = 1.674188, dz/dx = 0.023858) the term is 1.401452 + 3.735616 + 0.041914 = 5.178982. Why NLL ≈ 4 rather than ≈ 1.4 (= ½ + ½log 2π, the entropy of a standard normal): the −log(dz/dx) term dominates because z spans only ≈ 1.67 while x spans 64, so dz/dx ≈ 1/38 and −log(1/38) ≈ 3.6. The likelihood is measured in normalized-x units (the constant log var of the key→x scaling is dropped, train_flow.py:50), so the absolute value of the NLL is not comparable to NFL's; only differences between candidate weights matter to the optimiser.

---

## 4. The bypass rule ("manual fusion")

NFL: "the NFL first tries to transform the input keys, and computes the tail conflict degree based on the input keys and the transformed keys, respectively. If the latter tail conflict is larger, the NFL determines not to use the Numerical NF" [NFL §3.2.2]. Our per-region rule, in `Region::rebuild`, `Fusion::Manual` branch [index.hpp:273-285]:

```
z = flow(keys)                                      (transform_ns measured)
D99_flow = tail_conflict_degree(sorted z)           (sorted because the flow need not be monotone)
D99_raw  = tail_conflict_degree(normalized keys)    [index.hpp:237]
gain = D99_flow < D99_raw  and  (D99_raw − D99_flow) ≥ flow_min_gain · D99_raw     [index.hpp:281]
flow_used = !flow_bypass or gain                                                     [index.hpp:282]
if flow_used: x = z
```
With the defaults `flow_bypass = true, flow_min_gain = 0.10` a region keeps the flow only if its tail conflict degree falls by at least 10 % (the paper's rule is "not larger"; the 10 % margin is ours). `--flow-bypass 0` (variant `packed_rank_flow_forced`) forces the flow everywhere. **Reuse on compaction:** if `previous` is given and `relearn_on_compaction` is false, the previous region's `flow_used` and `tail_conflicts_*` are copied and the transform is evaluated only if the flow is in use [index.hpp:274-276]; `tail_conflicts_raw` is likewise copied [index.hpp:238].

Measured outcome on the 2M uniform samples (variant `packed_rank_flow`, 489 regions): the flow is kept (`flow_regions`, seed 11) in wise 4, fb 41, genome 8, books 26, planet 43, libio 21, stack 1, covid 17, history 13, osm 7 of the 489 regions, i.e. from 0.2 % (stack) to 8.8 % (planet) [results/aidb/sweep/results.jsonl, variant packed_rank_flow; scratch/fix05_checks.py]. Consistently, `transform_calls/op` = 0.084 on fb (= 41/489 = 0.0838, since a lookup pays one transform iff its region uses the flow) and 1.000 for the forced variant [same file; verified in coststory: max |T − (root_flow + flow_regions/regions)| = 0.0015].

---

## 5. CSV-style smoothing (smoothing.hpp), line by line

### 5.1 The paper's problem

CSV [Amarasinghe et al., EDBT 2025] keeps the index structure and instead changes the *key set*: it inserts virtual points into the sorted keys so that the CDF becomes easier to fit by a linear model. The loss is the SSE of the indexing function(s), L_F(K) = Σᵢ Σ_{k∈Kᵢ} (fᵢ(k) − rank(k))² (eq. 2); with a budget λ = α·n, α ∈ (0,1), the single-segment problem is argmin_{Vᵢ,w,b} L_{f_{w,b}}(Kᵢ ∪ Vᵢ) s.t. |Vᵢ| ≤ λ (eq. 4) — note that the line is *refitted* on real and virtual points and the virtual points themselves count in the loss (eq. 5: Σᵢ(w kᵢ + b − yᵢ)² + (w k_v + b − y_v)²) [CSV §3-4]. The general problem is NP-hard (reduction from knapsack) [CSV §3]. Algorithm 1 `CDF_smoothing` is a greedy: while |V| < λ, for each sub-sequence of candidate positions take the endpoints (or, when the first-order derivative of the loss changes sign between the endpoints, the interior minimum), evaluate the loss of every candidate, pick the best, stop if it does not lower the previous loss (line 27-28), insert it and repeat [CSV Alg. 1, lines 5-31]. Candidates are restricted to (min K, max K) and never coincide with an existing key [CSV §4.2]. There is no public implementation ("checked September 2026") [smoothing.hpp:7-8].

### 5.2 `Sums` and the closed-form SSE

```
struct Sums { long double n, sx, sxx, sy, syy, sxy;   add(x,y): ++n; sx+=x; sxx+=x²; sy+=y; syy+=y²; sxy+=xy;
  sse(): if n<2 → 0; cxx = sxx − sx²/n; cxy = sxy − sx·sy/n; cyy = syy − sy²/n;
         s = cxx>0 ? cyy − cxy²/cxx : cyy; return max(0, s) }                         [smoothing.hpp:35-44]
```
Derivation: for OLS w = S_xy/S_xx, b = ȳ − w x̄ (centred sums S_xx = Σ(x−x̄)², S_xy = Σ(x−x̄)(y−ȳ), S_yy = Σ(y−ȳ)²), the residual sum of squares is

  SSE = Σ(yᵢ − w xᵢ − b)² = Σ((yᵢ−ȳ) − w(xᵢ−x̄))² = S_yy − 2w S_xy + w² S_xx = S_yy − S_xy²/S_xx,

and the centred sums come from the raw sums by S_xx = Σx² − (Σx)²/n etc. This is CSV's eq. 10-16 in a compact form: the loss for a candidate can be evaluated from six running sums without refitting w, b explicitly [CSV §4.1].

### 5.3 `loss_at(i, x_v)`: why inserting one point is O(1)

Working sequence `seq` = (feature, is_virtual) in feature order; the target of position j is simply j (so slot ranks are sequence positions) [smoothing.hpp:51-55]. `base` holds the sums of the current sequence. Inserting a virtual point after position i (target i+1) shifts the target of every later element j ≥ i+1 by +1. The updated sums are, with cnt = |seq| − (i+1) elements shifted,

```
S_y'  = S_y + cnt                                        (each shifted y grows by 1)
S_yy' = S_yy + Σ_{j≥i+1} ((j+1)² − j²) = S_yy + 2·Σ_{j≥i+1} j + cnt = S_yy + 2·suf_y[i+1] + cnt
S_xy' = S_xy + Σ_{j≥i+1} x_j·1 = S_xy + suf_x[i+1]
then add the new point (x_v, i+1): n'=n+1, S_x += x_v, S_xx += x_v², S_y += i+1, S_yy += (i+1)², S_xy += x_v(i+1)
loss = sse(S')
```
[smoothing.hpp:64-70]. `suf_x[j] = Σ_{t≥j} x_t` and `suf_y[j] = Σ_{t≥j} t` are suffix sums rebuilt once per round (O(|seq|)) [smoothing.hpp:59-63], so each candidate evaluation is O(1).

**Numeric instance of the O(1) update** (the 6-key example of 5.5, base sums n = 6, S_x = 39, S_xx = 379, S_y = 15, S_yy = 55, S_xy = 142; insert x_v = 6.5 after position i = 2, i.e. in the gap (3, 10)) [scratch/fix05_checks.py]:

```
cnt      = 6 − (2+1) = 3 shifted elements (positions 3, 4, 5)
suf_y[3] = 3 + 4 + 5 = 12          suf_x[3] = 10 + 11 + 12 = 33
shift:   S_y' = 15 + 3 = 18        S_yy' = 55 + 2·12 + 3 = 82       S_xy' = 142 + 33 = 175
add (6.5, 3): n' = 7, S_x' = 45.5, S_xx' = 421.25, S_y' = 21, S_yy' = 91, S_xy' = 194.5
centred: cxx = 421.25 − 45.5²/7 = 125.5   cxy = 194.5 − 45.5·21/7 = 58   cyy = 91 − 21²/7 = 28
SSE      = 28 − 58²/125.5 = 1.1952191235
```
identical to a brute-force OLS refit of the seven points (1,0),(2,1),(3,2),(6.5,3),(10,4),(11,5),(12,6): 1.1952191235. A round examines every gap: 4 loss evaluations for the endpoint test, and for a gap that enters the ternary search ≤ 2·40 more inside the loop plus 1 final evaluation of the midpoint candidate (`cl = loss_at(i, cx)`, smoothing.hpp:86), i.e. ≤ 85 evaluations per gap [smoothing.hpp:78,82-86], so one round is O(|seq|) and the whole run with budget λ is **O(λ·n)** loss evaluations; "the paper's O(n + λ) claim is not reproduced" [smoothing.hpp:11-13]. (CSV's complexity paragraph says the loss over K is computed once in O(n) and reused; the per-round cost over the candidate set is what dominates here.)

### 5.4 The per-gap search (four-sample endpoint test, ternary search)

For each gap (lo = seq[i].x, hi = seq[i+1].x) with hi > lo (gaps of zero width are skipped, so a virtual point never equals a key), w = hi − lo, e = 10⁻³·w:

```
l0 = loss(lo+e), l1 = loss(lo+2e), r1 = loss(hi−2e), r0 = loss(hi−e)
if l1 < l0 and r1 < r0:   # loss decreasing at the left end and increasing at the right end: interior minimum
    ternary search on [lo+e, hi−e] for ≤ max_ternary_steps=40 iterations or until b−a ≤ e; candidate = midpoint
else: candidate = the better of lo+e and hi−e
```
[smoothing.hpp:75-87]. The paper uses the *sign of the derivative* at the two endpoints for the same decision (Alg. 1 lines 13-22, "if the sign … are opposite, there is a local minimum"; convexity within a sub-sequence is cited from [11]) [CSV §4.2]; we sample the loss just inside the ends instead of computing L'. Three further deviations from CSV Algorithm 1, stated explicitly: (a) **candidate domain.** CSV works on integer keys ("To simplify the discussion, we use integer index keys, while our techniques also apply to real number index keys when they can be scaled up to become integers" [CSV §3]) and its candidate virtual points are the integer values strictly between consecutive keys (Fig. 3: "the segment formed by 21 to 25 is between index keys 20 and 26 (the index keys themselves are not considered to be candidate virtual points)" [CSV §4.2, Fig. 3]); Algorithm 1 line 7-8 append both endpoints directly when `E[i].second − E[i].first ≤ 1`, so a sub-sequence of ≤ 2 candidates skips the derivative test entirely, and between two *adjacent* integer keys there is no candidate at all [CSV Alg. 1 lines 7-8]. `smooth_cdf` instead works on real-valued features and always tries lo + 10⁻³·w and hi − 10⁻³·w in every gap of positive width [smoothing.hpp:77-78], which is why it produces non-integer virtual features such as 6.4994 on integer keys (5.5). (b) **interior minimum.** CSV's `minimum_point(M[i].first, M[i].second)` is "calculated by using the two partial derivative values" [CSV §4.2; Alg. 1 lines 20-21]; ours is a ≤ 40-step ternary search on the sampled loss [smoothing.hpp:82-86]. (c) **acceptance margin.** `best` starts at `current` and every candidate must satisfy `cl < best − 10⁻¹²·max(1, best)`, with `best` updated as the round progresses [smoothing.hpp:74,88]: the relative margin is applied against the best candidate seen so far in the round, not only against the current loss, and a round that finds no such candidate breaks the loop (`found == false`, Algorithm 1 line 27) [smoothing.hpp:90]; otherwise the point is inserted, `base` is recomputed from scratch (O(|seq|)) and `rounds` increments [smoothing.hpp:91-92]. At the end, `out.slot[k]` = position of the k-th real key in the final sequence (rank + virtual points before it) and `out.virtual_features` = the inserted values in insertion order; `sse_before` is the OLS SSE of the ORIGINAL sequence (computed at smoothing.hpp:56 before any insertion, no virtual points), while `sse_after` is the SSE of the augmented sequence after the last accepted insertion (`current`, smoothing.hpp:71,92,95). So a ratio `sse_after/sse_before` compares an n-point fit with an (n + v)-point fit whose targets are slots, not ranks; the header comment's "augmented-set OLS SSE" applies to `sse_after` only [smoothing.hpp:30].

**Bounds.** `alpha ∈ [0, 64)` is enforced here (the root uses `root_alpha` up to 64 fences per region fence), while `Config::validate` caps `virtual_alpha < 1` for regions [smoothing.hpp:48; index.hpp:67,69]: CSV requires α ∈ (0,1) "to retain a linear space overhead" [CSV §3]. `budget = ⌊α·n⌋`; nothing is done if n < 3 or budget = 0 [smoothing.hpp:57-58].

### 5.5 WORKED EXAMPLE on 6 keys, features [1, 2, 3, 10, 11, 12]

Base sums: n = 6, S_x = 39, S_xx = 379, S_y = 15, S_yy = 55, S_xy = 142. Centred: S_xx^c = 379 − 39²/6 = 125.5, S_xy^c = 142 − 39·15/6 = 44.5, S_yy^c = 55 − 15²/6 = 17.5. **SSE_before = 17.5 − 44.5²/125.5 = 1.7211155378** (direct check with w = 0.35458, b = 0.19522: 1.7211155378 ✓) [scratch/smooth_py.py; C++ `Sums::sse` gives 1.7211155378 too, scratch/index_demo.cpp block A].

Per gap (e = 10⁻³·width; losses via `loss_at`, cross-checked against a brute-force refit on the augmented sequence — identical to 6 decimals):

| gap i | (lo, hi) | e | loss(lo+e) | loss(lo+2e) | loss(hi−2e) | loss(hi−e) | interior? |
|---|---|---|---|---|---|---|---|
| 0 | (1, 2) | 0.001 | 3.427440 | 3.427522 | 3.632328 | 3.632664 | no (rising) |
| 1 | (2, 3) | 0.001 | 3.632510 | 3.632021 | 3.265183 | 3.264944 | no (falling) |
| 2 | (3, 10) | 0.007 | 3.257072 | 3.249449 | 3.249449 | 3.257072 | **yes** |
| 3 | (10, 11) | 0.001 | 3.264944 | 3.265183 | 3.632021 | 3.632510 | no |
| 4 | (11, 12) | 0.001 | 3.632664 | 3.632328 | 3.427522 | 3.427440 | no |

Only the big gap qualifies for a ternary search; it converges in 18 iterations to x_v = 6.499363 (loss 1.1952191978; the exact minimiser is the midpoint 6.5 with loss 1.1952191235 — the search stops when the bracket is narrower than e = 0.007). Every other gap's best endpoint (≥ 3.26) is *worse than the current SSE 1.72*: inserting a point in a small gap only shifts the later targets up and hurts. Best = gap 2, so with α = 0.2 (budget ⌊1.2⌋ = 1) the C++ returns

```
alpha=0.20 budget=1 rounds=1 sse_before=1.7211155378 sse_after=1.1952191978
slots = 0 1 2 4 5 6      virtual = 6.4993630796
```
[scratch/smooth_demo.cpp]. Slots: keys 10, 11, 12 moved from ranks 3,4,5 to slots 4,5,6 — the augmented sequence is (1,0),(2,1),(3,2),(6.499,3),(10,4),(11,5),(12,6) and its OLS SSE is 1.19522 (Python recomputation ✓). Larger budgets keep filling the same gap: α = 0.4 → virtual {6.4994, 7.5031}, SSE 0.97892; α = 0.5 → adds 4.9698, SSE 0.51954; α = 0.9 (budget 5) → adds 8.5625 and 4.1837, SSE 0.17773 and slots 0 1 2 8 9 10 [same program]. The unit test's vector {1,2,3,4,5,10,20,26,27,30} at α = 0.3 gives 3 rounds, SSE 7.8405 → 4.5861, virtual {16.599, 14.781, 22.110} [scratch/smooth_demo.cpp; test_main.cpp:92-96 checks the invariants].

Real data: on the fb 2M sample at α = 0.1 the budget is 488·⌊409.6⌋ + ⌊115.2⌋ = 199,707 and exactly 199,707 points were inserted (`rounds = 199707`: early stop never fired), per-region summed SSE 3.798×10⁹ → 5.873×10⁸ (ratio 0.1546) in the hardness tool [results/aidb/hardness/fb_sample_csv.json smoothed], and 3.798×10⁹ → 5.866×10⁸ (0.1544) inside the index [results/aidb/sweep/results.jsonl fb packed_rank_vp10 learnability] — the two differ by 0.1 %. This is floating point, not an algorithmic difference: the OLS SSE and the greedy's relative grid (e = 10⁻³·gap) are invariant under an affine change of the feature, so the tool (features `double(key − global_min)`, up to 7.7×10¹⁰ [hardness.cpp:170]) and the index (per-region features `(k − origin)/span` in [0, 1] [index.hpp:236]) run the same algorithm and should agree exactly. The proof is that `sse_before`, which involves no greedy choice, already differs: 3,798,262,711.26 (tool) vs 3,798,262,931.22 (index). With raw-scale features S_xx ≈ 10²⁵ and the centred sums cxx = S_xx − S_x²/n [smoothing.hpp:40] cancel catastrophically in `long double`, which on Apple arm64 is only 64-bit ("Apple arm64 has long double == double" [hardness.hpp:14]); the index's [0, 1] features do not suffer from this. The 0.1 % is therefore a precision artefact of the tool's feature scale, and the index's number is the more accurate one [results/aidb/hardness/fb_sample_csv.json smoothed.rank_sse_before; results/aidb/sweep/results.jsonl fb packed_rank_vp10 learnability.rank_sse_before]. Smoothing cost 11.7 s (tool, 16 threads) / 36.9 s summed over the build threads (index) for 2M keys; at 200M keys it is 2,407 s of summed thread time (fb `packed_rank_vp10_root_vf004`, `smoothing_ns`) [results/aidb_fullscale/sweep/results.jsonl].

---

## 6. How the index uses virtual points

1. **Targets, not records.** `smooth_cdf(x, virtual_alpha)` runs on the region's features [index.hpp:305]; `slots[i] = sm.slot[i]` become the rank model's targets [index.hpp:307,312]; each block stores `slot_begin = sm.slot[b.rank_begin]` (the slot of its first key) [index.hpp:308,83]. Nothing is written to the arena or to `values`; `virtual_points` is only counted and the feature values are kept in `virtual_features` [index.hpp:306].
2. **Lookup.** `predicted_block` binary-searches the prediction y against `blocks[m].slot_begin` [index.hpp:125-126]: the model now predicts in slot space and the block boundaries are expressed in slot space, so the extra "room" created by the virtual points is consistent on both sides. `locate_block`'s fence correction is unchanged (exact `last` keys) [index.hpp:135].
3. **Memory.** `Region::memory()` counts `virtual_features.size()·sizeof(double)` in `metadata_bytes` [index.hpp:320]. At α = 0.1 that is ≤ 0.1 doubles per key = 0.8 bytes/key. Measured: fb vp10 adds 199,707·8 = 1,597,656 bytes of metadata for 2M keys = 0.7988 B/key. To place that, the full fb `packed_rank` memory at bulk load (seed 11, `memory_before`) is: `key_bytes` 5,636,561 (2.82 B/key of packed arena, against 8 B/key raw), `value_bytes` 16,000,000 (8 B/key), `metadata_bytes` 1,141,040 (0.57 B/key: Σ over the 489 regions of sizeof(Region) + blocks·sizeof(BlockDescriptor) with 15,625 descriptors in total, plus the Index-level bookkeeping [index.hpp:320,495]), `accounted_bytes` 22,777,601 = 11.39 B/key; `packed_rank_vp10` has `metadata_bytes` 2,738,696 = 1,141,040 + 1,597,656 and `accounted_bytes` 24,375,257 = 12.19 B/key, i.e. +7.0 % [results/aidb/sweep/results.jsonl fb seed 11; scratch/fix05_checks.py]. Beware that the cost-story metadata table, built from the `aidb_final` sweep, prints `packed meta = 1141168`, 128 B more than the uniform sweep's 1,141,040: `Index::memory()` adds `sizeof(Index)` and the root's slot table [index.hpp:495], and the root fields were added to `Index` between the two builds [unverified: the two binaries were not diffed]. Cite one sweep consistently; the vp10 delta of +1,597,656 B is identical in both.
4. **Compaction reuse** (`relearn_on_compaction = false`, `previous` given, features sorted) [index.hpp:289-303]: keep the previous `virtual_features` that fall strictly inside (x.front(), x.back()), sort them, and merge them with the new keys in one pass to assign slots — O(n + v), no search; compute SSE before/after with `Sums`; **drop them all if the SSE would not improve** (`after.sse() <= before.sse()` required, else `slots = ranks`) [index.hpp:300-301]. With `relearn` the full `smooth_cdf` reruns. The test `learnability` covers both paths and asserts `rank_sse_after ≤ rank_sse_before` after 2,000 mixed operations [test_main.cpp:109-122].
5. **Why reuse:** in the synthetic learnability sweep (read_heavy profile: 80 % reads, 10 % inserts, 5 % updates, 2 % erases, 3 % scans [benchmark.cpp:196]), relearning on every compaction (`packed_rank_vp10_relearn`) dropped throughput from 1,063,104 to 9,145 ops/s on clustered (116.3×), 921,942 → 3,981 on dense_sparse (231.6×), 1,047,678 → 6,510 on locally_hard (160.9×), 945,537 → 7,323 on lognormal (129.1×); insert p99 went from ~0.2 ms to 14-56 ms. Read-only throughput is unaffected within noise, as expected, since no compaction ever runs: the file's `paired_throughput_speedup` versus `packed_rank` on the read_only profile is 1.0004 / 1.0059 (clustered), 1.0279 / 1.0320 (dense_sparse), 1.0588 / 1.0145 (locally_hard), 0.9897 / 0.9842 (lognormal) for `packed_rank_vp10` / `packed_rank_vp10_relearn`, i.e. 0.98-1.06; the direct relearn/vp10 throughput ratios (medians 4,682,376/4,653,687, 3,005,787/2,935,709, 3,831,908/3,996,803, 4,237,439/4,397,696 ops/s) are 1.006, 1.024, 0.959, 0.964, i.e. 0.96-1.02 [results/learnability/summary.csv, read_only rows; scratch/fix05_checks.py].

---

## 7. The per-region selector (`Fusion::Auto`) [index.hpp:239-272]

The condition is `c.fusion == Fusion::Auto && !reuse && keys.size() > 2` with `reuse = previous && !c.relearn_on_compaction` [index.hpp:201,239]. So the selector runs at bulk load (no `previous`), and it also re-runs on every compaction rebuild (which does pass `previous`) when `relearn_on_compaction` is true; it is skipped only when a previous region is given AND relearn is off, in which case the previous decision and virtual points are reused (section 6.4). Two facts about Auto mode that are easy to get wrong: (1) the NFL bypass rule of section 4 is NOT applied; `tail_conflicts_flow` is still computed on the sorted z, but only for reporting [index.hpp:248], and the flow is chosen purely by the probe cost below; (2) `smooth_cdf` runs once per *base* candidate, i.e. once on the raw features and once on the flow features [index.hpp:252-257], so the build pays the O(λ·n) search twice: on fb, `packed_rank_fusion_auto` reports `smoothing_ns` = 124.15 s of summed thread time against 36.94 s for `packed_rank_vp10` (osm: 104.83 s vs 27.75 s) [results/aidb/sweep/results.jsonl seed 11]. That doubling (and a bit more, since the flow candidate's smoothing is on the same n) is a cost of the selector design, not of the structure it chooses. Candidates, in this order:

```
cands[0] = {flow=false, vp=false, feat=normalized keys, slots=ranks}
cands[1] = {flow=true,  vp=false, feat=z(keys),         slots=ranks}          (if c.flow; transform_ns measured; D99_flow computed on sorted z)
cands[2] = {flow=false, vp=true,  feat=normalized keys, slots=smooth_cdf(...).slot}   (if virtual_alpha > 0)
cands[3] = {flow=true,  vp=true,  feat=z(keys),         slots=smooth_cdf(z).slot}     (if both)
```
[index.hpp:243-257]. For each candidate the *real* structure is fitted (`rank_model.fit_xy(keys, feat, slots)`, `slot_begin` per block) and the real `locate_block` is run on every key of the region under `Routing::Rank` with a `QueryStats` collector; the cost is

  cost = (fence_probes + coordinate_probes)/n + (flow ? flow_cost : 0)

[index.hpp:258-265]. Root probes and decode work are excluded (identical across candidates), and the routing is forced to Rank even if the run uses byte routing. The minimum is taken with a strict `<` scan from index 0, so ties go to the earlier candidate: raw ranks beat everything at equal cost, flow beats virtual points at equal cost [index.hpp:266]. `cost_none = cands[0].cost`, `cost_selected = best.cost`; the chosen candidate's features, slots, virtual features and SSEs are installed and `targets_done` skips the manual branch [index.hpp:267-272]. `choice = (flow_used ? 1 : 0) | (virtual_points ? 2 : 0)` [index.hpp:313] and `learnability()` counts regions per choice as `choices: {none, flow, vp, both}` [index.hpp:488; benchmark.cpp:74]. `flow_cost` is a probe-equivalent per lookup charged to the transform, default 4, "measure it; default is a placeholder" [index.hpp:64-65]; the measured transform costs 66 ns [56, 77] per evaluation [results/aidb/MEETING_NOTES.md:114-115].

**Worked example** (`scratch/index_demo.cpp`, block C): 128 keys (96 quadratic then 32 sparse), `region_keys = 64, block_keys = 8, virtual_alpha = 0.1`, a toy flow. Both regions choose `vp`: `cost_none_mean = 6.0391` probes per lookup, `cost_selected_mean = 5.9297`, 12 virtual points, D₉₉ raw 16.5 vs flow 11.0 — yet the flow is not chosen even with `flow_cost = 0`, because the tail conflict degree is not what is being minimised; the measured probe count is. **Real data:** on the fb 2M sample (`packed_rank_fusion_auto`, α = 0.1, flow_cost 4) all 489 regions choose `vp`, `cost_none_mean = 7.627`, `cost_selected_mean = 7.208` probes (fence + coordinate) per lookup; the measured counters agree: fence 2.602 → 2.183 with coordinate 5.029-5.030 (7.63 → 7.21). On osm 470 regions choose vp, 19 choose none (10.097 → 9.657). With the flow alone as an option (`packed_rank_flow_costsel`), zero regions take it on all five datasets shown [results/aidb/sweep/results.jsonl]. Test `fusion_auto` asserts Σchoices = regions, `cost_selected ≤ cost_none`, and that `flow_cost = 100` is never chosen [test_main.cpp:124-141].

---

## 8. The learned root (`Index::fit_root`) [index.hpp:358-401]

**Which keys.** For n ≥ 2 regions, fences[j] = the *first real key* of region 0 for j = 0 and `low_fence` (= first key at bulk load) for j ≥ 1; ranks = j [index.hpp:369-370]. **The sentinel-fence bug:** an earlier version used `low_fence` for all regions, i.e. the sentinel 0 for region 0. For a contiguous window whose keys start far from 0 (e.g. 9·10¹²), the point (0, 0) is a huge leverage outlier: the raw least-squares line through it is useless, so the raw candidate's estimated cost exceeded binary search and was rejected, while the flow's tanh compression of the outlier let the flow-based candidates look *relatively* useful. The complete pre-fix candidate table (estimated root probes per lookup, seed 11, binary = 8.956; the same estimates appear in every root variant of that sweep) shows that the plain flow candidate did NOT win either: only the vp_flow combination beat binary search [results/aidb_final/sweep/results.stale-root-before-fix.jsonl, learnability.root_probes_*, root_virtual; work_counters.root_probes/operations]:

| sample (pre-fix) | raw | flow | vp_raw | vp_flow | chosen | measured `root_probes/op` |
|---|---|---|---|---|---|---|
| fb_window | 14.339 | 14.066 | 14.230 | **7.000** | flow + 1,386 virtual fences (`root_flow = true`, `root_virtual = 1386`) | 2.882 |
| books_window | 14.241 | 13.955 | 13.902 | **6.990** | flow + 1,374 fences | 2.930 |
| osm_window | 14.378 | 15.632 | 14.358 | 10.510 | none (binary) | 8.956 |

Read it this way: the sentinel point (0, 0) ruined raw AND vp_raw alike (both ≈ 14.2-14.4: no number of virtual fences can repair a line pinned to an outlier 9·10¹² away), whereas the tanh squashed the outlier's feature so that fences could still straighten the flow's CDF, hence vp_flow ≈ 7. After the fix (fences[0] = first real key) the same seed gives [results/aidb_final/sweep/results.jsonl]:

| sample (post-fix) | raw | flow | vp_raw | vp_flow | chosen | measured `root_probes/op` |
|---|---|---|---|---|---|---|
| fb_window | 3.584 | 7.860 | **2.409** (34 fences) | 6.490 | raw + fences | 2.273 |
| books_window | 2.320 | 8.115 | **2.301** (1 fence) | 6.331 | raw + fences | 2.102 |
| osm_window | 10.555 | 14.382 | **3.559** (984 fences) | 7.648 | raw + fences | 3.435 |

fb_uniform is unchanged (raw 2.266, vp_raw 2.224) because its keys start near 0 [same files]. **Warning about `coststory.txt`:** the cost-story verification file was generated on the PRE-fix sweep (its mtime, Sep 21 20:45, precedes the refit `results.jsonl` at 21:17 and matches `results.stale-root-before-fix.jsonl` at 20:46), so its root columns contradict this section (e.g. "fb_window packed_rank_root_fusion flow+fences ... est b/r/f/vr/vf = 8.96/14.34/14.07/14.23/7.00" and "fb_window root_vf4: d = +0 binary") [results/aidb/verification/coststory/coststory.txt per-(dataset, variant) table and metadata table; `ls -lT`]. Its `P` constant already points at `results/aidb_final/sweep/results.jsonl` [coststory.py:4], so rerunning it refreshes those rows; it was not rerun here because the verification directory is outside this guide's write scope. What is unaffected by the staleness: the vp10 metadata delta (+1,597,656 B), the invariant that root variants differ from `packed_rank` only in `root_probes` and `transform_calls`, and the osm/planet 1,956-fence rows (uniform samples, whose keys start near 0). So the "flow + fences synergy" of the first pass was the artefact; after the fix the flow feature wins 0 of 60 candidate cells [results/aidb/MEETING_NOTES.md:101-106]. The regression test builds 4,000 keys from 9·10¹² and requires the raw root to beat binary search [test_main.cpp:169-171].

**Probe set for the cost estimate.** 2n − 1 keys: every fence and every midpoint fence_j + (fence_{j+1} − fence_j)/2 [index.hpp:371-372].

**Candidates** {raw, flow} × {ranks, virtual fences} [index.hpp:375-396]:
- feature x_j = `tmp.normalized(fence_j)` (tmp fitted on the fences: origin = first fence, span = last − first) or `(*flow)(fence_j)`;
- with virtual fences (`root_alpha > 0`, n > 2, x sorted): `smooth_cdf(x, root_alpha)` with budget ⌊root_alpha·n⌋; targets = slots; `table` of size n + virt maps every slot to its region (slot j.. of region j) [index.hpp:381-388]; the candidate is skipped if no virtual fence was inserted;
- `model.fit_xy(fences, x, targets)`; cost = (Σ over the 2n−1 probe keys of the `ok()` calls made by the real `locate_from_prediction`)/(2n−1) + (flow ? flow_cost : 0) [index.hpp:389-392];
- the binary-search cost on the same probe keys is measured too [index.hpp:397].
The best candidate replaces binary search only if its cost is strictly lower [index.hpp:399-400]; `learnability()` reports `root_probes_binary/raw/flow/vp_raw/vp_flow`, `root_model`, `root_flow`, `root_vp`, `root_virtual` [index.hpp:490-491].

**Refit.** `fit_root()` runs at the end of `bulk_load` and after every *split* [index.hpp:443,418]; a compaction without split keeps the fences, so the model remains valid; exactness is guaranteed by the correction in any case.

**Numbers.** 2M samples (489 regions; binary cost 8.956): fb_uniform raw 2.266, vp_raw 2.224 (1 virtual fence), flow 8.139, vp_flow 6.349 → chosen raw+fences, measured `root_probes/op` 2.079; fb_window raw 3.584, vp_raw 2.409 (34 fences), flow 7.860 → raw+fences, measured 2.273; osm_uniform: 1,956 virtual fences (= full budget 4·489) [results/aidb_final/sweep/results.jsonl; coststory metadata table]. Slot-table bytes = 4·(489 + virt), e.g. 9,780 B on osm/planet [coststory]. **200M keys** (48,829 regions, binary 15.658 ≈ log₂ 48,829 = 15.58): fb raw 10.802, vp_raw 3.497 with 973 virtual fences (search stopped early), measured 3.419/op; build 9.98 s vs 6.63 s for `packed_rank`. planet raw 25.80, flow 27.87 (both rejected), vp_raw 13.748 with 195,316 virtual fences = the full budget ⌊4·48,829⌋ → chosen, measured 13.685/op, **build 2,067.8 s** vs 6.26 s [results/aidb_fullscale/sweep/results.jsonl]. **Why this build number is wall clock while the region figure (2,407 s, section 5.5) is summed thread time:** `fit_root()` runs on the calling thread after the worker pool has joined [index.hpp:441-443], so its `smooth_cdf` over 48,829 fences is single-threaded and its whole cost lands in `build_ns`; it is *not* included in `learnability.smoothing_ns`, which only sums the per-region `r->smoothing_ns` [index.hpp:485] (planet `packed_rank_root_vf4` reports `smoothing_ns = 0` next to its 2,067.8 s build). Region smoothing, by contrast, runs inside the 16 build threads and is reported as the SUM of per-region times, so fb `packed_rank_vp10_root_vf004`'s 2,407.4 s of thread time is ≈ 150 s of wall clock, consistent with its `build_ns` of 161.0 s [results/aidb_fullscale/sweep/results.jsonl]. Complexity: 195,316 rounds × O(|seq|) with |seq| growing from 48,829 to 244,145, i.e. of the order of 10¹⁰ gap examinations (≥ 4 loss evaluations each) — our arithmetic from section 5.3, consistent with the 34 minutes observed [unverified: not profiled]. At `root_alpha = 0.04` (budget 1,953) fb again stops at 973 fences (same result, build 9.99 s) while planet's estimate 25.18 loses to binary search [same file, `packed_rank_root_vf004`].

---

## 9. The hardness tool (hardness.hpp, src/hardness.cpp)

AIDB [Zhang, Tang, Ailamaki, AIDB@VLDB 2026, §4.1] evaluates five scalar metrics on sorted keys, all O(N): RMSE and ME of a least-squares line, CD = "the maximum number of keys mapped to the same rank by a linear model fitted on the dataset using FMCD by Wu et al." (LIPP), PLA-32 and PLA-4096 = the number of segments of the optimal ε-bounded PLA (the two dimensions of the GRE metric), plus their 2- and 3-way compositions [AIDB §4.1]. The performance measure is "the index throughput under a uniform read-only workload over the same dataset used to populate the index" [AIDB §2.2 "Measuring Dataset Hardness"; §3.1 is the "Strawman Measurement" subsection].

- **`least_squares<T>(x)`** [hardness.hpp:84-106]: features are x − x_min (integers subtracted as `unsigned __int128`, exact), Welford OLS of rank i on the feature, then RMSE = √(Σeᵢ²/n), ME = max|eᵢ|. Hand case {0,1,3}: slope 9/14 = 0.642857, intercept 1/7, residuals −1/7, 3/14, −1/14 → ME = 3/14 = 0.214286, RMSE = √(1/42) = 0.154303 [test_hardness.cpp:128-131; scratch/index_demo.cpp block D]. fb full: RMSE 57,735,023.67, ME 99,999,995.54 [results/aidb/hardness.json full.fb].
- **`fmcd_fit` / `conflict_degree`** [hardness.hpp:108-227]: a port of LIPP's `build_tree_bulk_fmcd` (MIT, commit fe6ca49) [hardness.hpp:15-22]. LIPP's FMCD ("fastest minimum conflict degree") looks for the smallest window D such that a line with slope 1/U_T, U_T = (keys[size−1−D] − keys[D])/(L − 2) + 10⁻⁶, spreads the keys over L = size·(gap+1) slots with at most D-ish collisions: `i = 0, D = 1; while i < size−1−D: advance i while keys[i+D] − keys[i] ≥ U_T; if stuck, D += 1 and recompute U_T; stop when D·3 > size` [hardness.hpp:185-192]. gap = 1 (size ≥ 10⁶), 2 (≥ 10⁵), 5 otherwise [hardness.hpp:109-113]. Model: a = 1/U_T, base = L/2, anchor = (keys[size−1−D] + keys[D])/2 kept as an exact `__int128` `anchor2 = 2·anchor` so that position(key) = a·(key − anchor) + base uses exact 64-bit differences (LIPP itself computes a·key + b in long double, exact only on 80-bit platforms) [hardness.hpp:114-117,138-147,194-198]. If D·3 > size, the fallback fits the line through the two tertile midpoints [hardness.hpp:199-212]. `predict` = floor of the double product, clamped to [0, L−1] (LIPP's PREDICT_POS) [hardness.hpp:138-147]; **CD** = the longest run of equal predictions (sorted keys give non-decreasing slots) [hardness.hpp:218-227]. **Epsilon scaling:** LIPP's absolute +10⁻⁶ on U_T is used verbatim for integer keys; for double features (flow outputs, range O(1)) it would be up to 147 % of U_T, so it is scaled to 10⁻⁶·(range/(n−1)); the literal value is emitted alongside as `conflict_degree_lipp_epsilon` [hardness.hpp:149-167; hardness.cpp:160]. Rounding caveat: floor of a double product makes CD ±1-uncertain near slot boundaries (books 2M sample: 11 here and with LIPP's double arithmetic, 10 with exact rationals) [hardness.hpp:119-127]. Hand-traced example (test `cd_cases`, reproduced in block D): 50 consecutive keys then 950 keys 100 apart from 10⁶: gap 5, L = 6,000, D = 50, U_T = 89,900/5,998 + 10⁻⁶ = 14.98833, a = 0.0667186, base 3,000, anchor 1,044,950; the cluster is clamped to slot 0 (predict(cl[0]) = predict(cl[49]) = 0, predict(cl[50]) = 1, predict(cl[51]) = 7) → **CD = 50** [test_hardness.cpp:96-102]. fb full: gap 1, capacity 4·10⁸, D = 114, U_T = 193.27, CD = 110 [results/aidb/hardness_details.json fb.full_original.fmcd; hardness.json full.fb.conflict_degree].
- **`pla_segments<T>(in, ε)`** [hardness.hpp:229-311]: `OptimalPLA` is O'Rourke's 1981 streaming algorithm as PGM-index implements it (`OptimalPiecewiseLinearModel::add_point`, Apache-2.0, commit c6fcf3d): it maintains the upper and lower convex hulls of the strips [y−ε, y+ε] and the rectangle of feasible slopes; a point outside starts a new segment [hardness.hpp:252-284]. Integral keys are widened to `unsigned __int128` so PGM's sentinel (x_last + 1, n) exists even at 2⁶⁴ − 1, and cross products are `__int128` (exact) [hardness.hpp:230-249,287-294]. Duplicates follow PGM's rule (a run of equal keys is one point; (x+1, rank of the last duplicate) is added if there is room) [hardness.hpp:304-309]. Count = breaks + 1. Examples: a perfect line → 1 segment for ε ≥ 1 (2 at ε = 0 because the sentinel is off the line); the staircase of 10 runs of 100 consecutive keys 10⁶ apart → 10 segments at ε = 32, 1 at ε = 4,096 [test_hardness.cpp:67-75; block D].
- **`compute_metrics`** runs the four passes (LS, FMCD+CD, one PLA per ε) as `std::async` tasks [hardness.hpp:323-333]; results are identical serial vs parallel [test_hardness.cpp:136-138].
- **CLI** `scaleli_hardness --data PATH --dtype uint64|uint32 [--limit N] [--flow W.txt] [--virtual-alpha A] [--region-keys 4096] [--pla-eps 32,4096] [--check-sorted 0|1] [--write-sorted PATH] [--sort-only 1] [--threads T] [--verbose 1]` [hardness.cpp:84-106]: reads SOSD (8-byte count header) [hardness.cpp:41-58]; audits sortedness and duplicates (seven GRE files are served unsorted: covid, genome, history, libio, planet, stack, wise; the pipeline writes `<name>.sorted`) [hardness.cpp:121-135; aidb_pipeline.py:22-27]; `original` block on raw keys; with `--flow`, `transformed` block on sorted z(keys) (sorting is needed only if the flow is non-monotone; `unordered_pairs` is reported); with `--virtual-alpha`, `smoothed` block on the per-region augmented sequence produced by `smooth_regions` (real and virtual features in order, so it has n + v entries; features are z(keys) with `--flow`, else `double(key − min)`) [hardness.cpp:143-182; hardness.hpp:335-377]. Cost: ~3 s and 1.6 GB RSS for a sorted 200M-key file on 16 threads [hardness.cpp:104]; books full took 3.13 s [results/aidb/verification/pla/books_full_hardness.json elapsed_ns].
- **Verification.** PLA against PGM-index's own `make_segmentation` on the full 200M-key books file: PGM 262,604 / 97 segments, ours 262,604 / 97 [results/aidb/verification/pla/books_full_ref.json vs books_full_hardness.json]; 320 random small instances against a DP oracle [test_hardness.cpp:83-90]. Least squares against an exact-rational Python oracle (`vlib.py`, `Fraction`/`Decimal`): relative error 5.3×10⁻¹⁵ (RMSE) and 2.5×10⁻¹⁴ (ME) on the books 2M sample, 2.7×10⁻¹⁴ / 5.5×10⁻¹³ on the fb 5M prefix; CD: books sample cpp 11 vs exact-rational 10 (the documented ±1), fb 5M 94 = 94 [results/aidb/verification/metrics/realdata.log].

---

## 10. The scorer (tools/aidb_scores.py)

Protocol [AIDB §3.2, eqs. 1-3]: normalise each index's throughput p̂_I(S) = p_I(S)/std_I over the datasets; S_i is *harder* than S_j under a (multi-dimensional) metric iff h_k(S_i) ≥ h_k(S_j) on every dimension and > on at least one (ties on all dimensions are incomparable); importance w = sigmoid(δp) with δp = p̂(harder) − p̂(easier); a comparable pair is conforming when δp ≤ 0 and violating when δp > 0; R_I = Σ w over conforming, P_I = Σ w over violating; Conf_I = (R_I − P_I)/(R_I + P_I) (eq. 1), Conf = mean over indexes (eq. 2); Cov = (|C| − |U|)/(|C| + |U|) with |C| + |U| = C(n, 2) (eq. 3).

Code: `harder(hi, hj, dims)` = `all(≥) and any(>)` [aidb_scores.py:82-84]; `classify_pairs` splits the C(n,2) unordered pairs into (harder, easier) and incomparable [87-97]; `coverage` [100-102]; `conformance_of_variant(p_hat, comparable)` accumulates reward/penalty with `sigmoid(gap)` and returns Conf_I (0 when R + P = 0) [105-120]; `normalize` divides by the population std (ddof = 0 default; the paper does not state the ddof) [67-79]; `score_scope` builds the dataset corpus as the intersection of the hardness scope and every scored variant's throughput, skips variants with < 3 datasets or with zero throughput std (eq. 1 undefined) and drops non-finite entries with a reason [153-211]; 25 metrics = 5 scalars + C(5,2) = 10 pairs + C(5,3) = 10 triples, "PLA-32·PLA-4096" aliased "GRE" [45-58]. Scopes: `full, full_flow, sample, sample_flow, sample_csv, sample_flow_csv` (the metrics on 200M keys, on the flow-transformed 200M keys, on the 2M sample, after flow, after smoothing, after both) [49]. **Protocol deviation to state up front:** in EVERY scope the throughput side comes from the 2M uniform samples (every sweep row has `dataset_path = data/samples/<name>_2M_uniform_s42` and `initial_rows = 2,000,000` [results/aidb/sweep/results.jsonl]; `throughput.json` is built from that sweep [aidb_pipeline.py:159-175]). Scope `full` therefore pairs hardness measured on the 200M-key files with throughput measured on 2M-key samples of them, whereas the AIDB paper measures both on the same 200M-key dataset [AIDB §2.2]. Only `results/aidb_fullscale` (fb and planet, single seed) measures throughput at 200M keys, and it is not scored. Consequence for reading Conf: scope `sample` is the internally consistent one (hardness and throughput on the same 2M keys); scope `full` tests whether the full-file hardness ordering predicts the sample throughput ordering, which is a weaker, cross-scale statement and should be presented as such.

**How our throughputs get in.** `aidb_pipeline.py step_throughput` reads `sweep/results.jsonl`, checks that every (dataset, profile, seed, repeat) has a single `trace_fingerprint` across variants (paired traces), and writes `throughput.json = {variant: {dataset: median over seeds of throughput_ops_s/1e6}}` [aidb_pipeline.py:159-175,539-546]; `step_scores` calls `aidb_scores.py --hardness hardness.json --throughput throughput.json --output scores.json` [548-553]. The "indexes" of the paper are replaced by our ten control variants (`sorted_vector, raw_rank, packed_rank, packed_rank_flow, packed_rank_flow_forced, packed_rank_vp10, packed_rank_flow_vp10, packed_byte, packed_rank_fusion_auto, packed_rank_flow_costsel`) [results/aidb/throughput.json].

**Worked example (scope `full`, metric RMSE, variant `packed_rank`; computed with the module).** Throughput (MOPS): books 2.144, covid 2.217, fb 2.638, genome 2.476, history 2.816, libio 2.824, osm 1.932, planet 2.789, stack 2.942, wise 2.482; population std 0.32038 → p̂ = books 6.692, covid 6.921, fb 8.234, genome 7.728, history 8.789, libio 8.816, osm 6.030, planet 8.705, stack 9.182, wise 7.747. RMSE (full): fb 57.7M > planet 29.8M > osm 24.2M > books 18.1M > genome 7.5M > libio 3.4M > wise 2.2M > covid 1.8M > stack 0.84M > history 0.82M — all distinct, so all 45 pairs are comparable (Cov = 1.0). Pair (fb harder than books): δp = 8.234 − 6.692 = +1.543 → violating, w = sigmoid(1.543) = 0.824 (fb is *faster* although harder); pair (books harder than covid): δp = −0.229 → conforming, w = 0.443. Totals: R = 7.358 over 29 conforming pairs, P = 11.984 over 16 violating → Conf_packed_rank = (7.358 − 11.984)/(19.342) = **−0.239**. Averaging over the ten variants gives Conf = −0.317 (per variant from −0.565 for `sorted_vector` to −0.147 for `packed_byte`) [results/aidb/scores.json full.RMSE]. For RMSE·CD, 32 pairs are comparable and 13 incomparable → Cov = (32 − 13)/45 = 0.422 [scratch computation with `classify_pairs`].

---

## 11. Tests (what is checked, by name)

- `tests/test_main.cpp` (8 groups, run as one binary): `codecs` (round trips, bit widths 0-64, corrupt input), `endpoints` (empty index, duplicates, keys 0 and 2⁶⁴−1 across all routings/policies), `differential` (45 configurations × 3,500 random ops vs `OrderedMap`, with `maintain`/`validate` every 127 ops; requires splits), `adversarial_routes` (corrupted models must still answer exactly), `maintenance_and_memory` (adaptive raw bypass, memory accounting, `dump_layout`, invalid config throws), `learnability` (smoothing invariants: SSE never increases, virtual points inside the range and never equal to keys, strictly increasing slots; D₉₉ = 0 on uniform, > 0 on a cluster; hand-checked 2D2H2L forward pass 0.5·2.75 + 0.25·0.75 → tanh sums; text round trip; differential with flow + smoothing × bypass × relearn × routing; `transform_calls > 0` when forced), `fusion_auto` (Σchoices = regions, `cost_selected ≤ cost_none`, no flow when absent or when flow_cost = 100, no vp when α = 0, differential, parallel-build layout identical), `learned_root` (`root_probes_binary > 0`, chosen estimate < binary, virtual fences > 0 when chosen, exact lookups over all keys and 9,973-spaced probes, splits refit, the far-from-zero window regression) [test_main.cpp:13-183].
- `tests/test_hardness.cpp`: `exactness_near_2_64` (1,000 keys below 2⁶⁴: RMSE < 1e-9, PLA = 1, FMCD D = 1, CD = 1, and the same keys as doubles collapse to CD = 1000), `pla_cases` (line, staircase 10/1, duplicates, 320 DP-oracle differentials, monotone in ε), `cd_cases` (cluster D = 50/CD = 50, tertile fallback with hand-derived slots, uniform CD = 1, gap table, scaled epsilon on doubles), `least_squares_cases` ({0,1,3} hand values, parallel = serial), `smoothing_cases` (`smooth_regions`: 8 regions, sorted output, every real feature survives, thread count irrelevant, unsorted input throws), `parallel_sort_cases` (equals `std::sort` for 6 sizes × 8 thread counts), `cli_case` (JSON well-formed, `--flow --virtual-alpha 0.1 --region-keys 1024` → 16 regions with virtual points, `--limit`, unknown option rejected, `--sort-only --write-sorted` on a shuffled file with duplicates, `conflict_degree_lipp_epsilon` present) [test_hardness.cpp:47-214].
- `tests/test_tools.py`: `ToolsTests.test_feasibility, test_uniform_scaling_not_smoothing, test_virtual_greedy` (greedy ≤ baseline, exhaustive ≤ greedy on 10 keys), `test_dataset_sampling_and_corruption` (uniform/window/strided samplers), `test_flow_trainer_and_conflicts` (D₉₉ 0 vs > 0; monotone training lowers NLL; 8-line weight file), `test_cluster_bootstrap`; `BenchmarkTests.test_all_workloads, test_query_distributions, test_bulk_sampling_and_order, test_pairing` (identical trace fingerprint and result checksum across indexes), `test_full_width_and_duplicates, test_cli_rejects_invalid_inputs, test_learnability_options` (trains a flow on dumped keys and runs vp/flow/fusion/byte/adaptive variants; virtual points > 0, SSE non-increasing, forced flow in every region, Σchoices = regions; α = 1.5 rejected), `test_latency_skip, test_u32_u64_import` [test_tools.py].
- `tests/test_aidb_scores.py` (21 test methods: `HandBuiltExample` 6, `ProtocolProperties` 6, `DatasetSelection` 7, `CommandLine` 2 [test_aidb_scores.py:31-291]): `test_normalization, test_scalar_rmse, test_two_dim_rmse_me_trades_coverage_for_conformance, test_three_dim_with_tied_dimension_matches_two_dim, test_all_ties_are_incomparable, test_sample_std_changes_scale_not_sign, test_scalar_metric_has_full_coverage_when_values_are_distinct, test_adding_a_dimension_never_increases_coverage, test_perfectly_conforming_ordering_scores_one_and_reversed_minus_one, test_zero_std_variant_is_skipped_not_scored, test_importance_is_sigmoid_of_signed_gap, test_canonical_metric_names, test_nan_and_inf_throughput_are_dropped_with_a_reason, test_null_throughput_shrinks_the_corpus_and_says_so, test_string_throughput_skips_the_variant_with_the_real_reason, test_cli_accepts_nan_in_throughput_json, test_variant_with_fewer_than_three_datasets_is_skipped, test_datasets_are_the_intersection_of_hardness_and_every_scored_variant, test_scopes_are_independent_and_missing_fields_drop_a_dataset, test_end_to_end_writes_contract_shape_and_prints_table, test_rejects_unknown_option_and_missing_scope`.
- `tests/test_aidb_pipeline.py`: `test_contract_variants_and_common_block, test_extra_variants_and_build_threads_are_opt_in, test_dry_dataset_urls_and_report_step_detection, test_median_throughput_orders_contract_variants_first, test_write_json_leaves_identical_files_untouched, test_sosd_helpers_and_metric_block, test_sample_stale_reason_and_cached_run_identity`. `tests/test_aidb_report.py`: `test_full_inputs_via_cli, test_empty_results_dir, test_corrupt_and_partial_inputs`.

---

## 12. Questions the supervisor may ask, with answers

1. **"Why do virtual points cost zero key bytes?"** Because they are never stored as records. `smooth_cdf` returns only slot ranks and feature values; the index uses the slots as the rank model's targets and writes one `slot_begin` per block. Lookups compare the model's prediction against `slot_begin` and then correct on the exact `last` keys of the blocks, so the arena, values and scans are untouched [index.hpp:83,125,307-308,312]. The only cost is 8 bytes per virtual point of `virtual_features` kept for compaction reuse = 0.8 B/key at α = 0.1, measured 1,597,656 B for 199,707 points [coststory]. CSV materialises gaps in its host indexes (ALEX/LIPP) and uses them for inserts: "we include V_i in the loss calculation, such that the storage space allocated to the virtual points can be used to accumulate data insertions, with minimized prediction errors when querying the inserted data points" [CSV §4, after eq. 4]; we deliberately do not.

2. **"Why is the selector's cost a probe count and not time?"** Time on this host is noisy at the level of the effect (two measurements of the identical structure differ by up to 8-10 % [MEETING_NOTES.md:107-111]), whereas the probe counts are deterministic, reproducible across seeds and comparable across candidates because everything except routing is held fixed. The selector runs the *real* `locate_block` on the region's own keys, so it measures the work the index will do, not a proxy such as SSE or D₉₉ [index.hpp:240-242,262-264]. The one time-like quantity, the transform, enters as `flow_cost` probe-equivalents (default 4, measured 66 ns per call ≈ a few probes) [index.hpp:64; MEETING_NOTES.md:114-115].

3. **"What exactly is a fence probe?"** One evaluation of `blocks[i].last < k` inside `Region::locate_block` [index.hpp:135]. It is the exact-key comparison that verifies or corrects the model's block hint: at least one per lookup (checking the predicted block), two for a hit that needs the left neighbour checked, and O(log d) for a prediction d blocks off. Root fence comparisons (`low_fence <= k`) are counted separately as `root_probes` [index.hpp:350,353].

4. **"Is your transform the NFL flow?"** No. Same deployed *shape* (2D2H2L, features [x, x − ⌊x⌋], tanh, sum decoder) and the same text weight format, but a stand-in trainer (1-D change-of-variables likelihood with Adam, not a BNAF), `--monotone` zeroes the fractional-feature weights, and the output is used as a model feature only; NFL sorts by the transformed key and builds AFLI on it [transform.hpp:1-19; train_flow.py:1-14].

5. **"Why does the flow not help at the region level?"** With 4,096-key regions each linear model already absorbs the global curvature; the tail conflict degree barely moves (fb 8.45 → 8.46 per region, osm 36.2 → 37.0), so the 10 %-gain bypass keeps it in 1-9 % of regions, and the cost-based selector never picks it at flow_cost 4 [results/aidb/sweep/results.jsonl]. The granularity ablation (regions 4,096 / 16,384 / 32,768) tested whether larger regions change this, and they do not: forcing the flow changes `fence_probes/op` negligibly at every region size (fb: 2.6010 vs 2.6010 at 4,096; 3.2104 vs 3.2094 at 16,384; 3.6888 vs 3.6922 at 32,768; osm: 5.0777 vs 5.0780; 8.3355 vs 8.3361; 10.1188 vs 10.1200, forced-flow vs plain `packed_rank`), `correction_distance/op` likewise (osm 18.17 vs 18.17 at 32,768), and `fusion_auto` picks the flow in 0 regions at all three sizes (`flow_region_fraction` = 0.0, `virtual_points_per_key` ≈ 0.095-0.100). So larger regions do raise the raw probe count (fb fence 2.60 → 3.69, osm 5.08 → 10.12: one line per region fits worse), but the flow still does not help; only virtual points do (fb `fusion_auto` fence 2.18-2.25 at all sizes) [results/aidb_granularity/summary_r4096.csv, summary_r16384.csv, summary_r32768.csv, columns fence_probes_per_operation, correction_distance_per_operation, flow_region_fraction, virtual_points_per_key; config.json].

6. **"What did the sentinel-fence bug do?"** It fitted the root's raw least-squares line through (0, region 0) while the real keys started at e.g. 9·10¹² (window samples), so the raw root looked worse than binary search (est. 14.3 vs 8.96 on fb_window) while the tanh-saturated flow feature was less damaged (14.07): the fusion candidate "won" for the wrong reason. Fitting on the first real key gives raw 3.58, flow 7.86; the flow then never wins [results/aidb_final/sweep/results.stale-root-before-fix.jsonl vs results.jsonl; index.hpp:366-370].

7. **"Where is the O(1) in CSV's loss evaluation, and why is your algorithm O(λ·n)?"** Inserting one point after position i shifts every later target by one; with suffix sums of x and of the positions the six OLS sums update in O(1) (S_y += cnt, S_yy += 2·suf_y + cnt, S_xy += suf_x) and SSE = S_yy − S_xy²/S_xx (centred) follows in O(1) [smoothing.hpp:64-70]. But every round still scans every gap (≥ 4 evaluations each) and the budget is λ = α·n rounds, hence O(λ·n) overall; the paper's O(n + λ) is not reproduced [smoothing.hpp:11-13]. This is why smoothing 200M keys costs ~2,400 s of thread time and why 195,316 virtual fences at the root took 2,068 s [results/aidb_fullscale].

8. **"How is CD computed and why can it differ from LIPP by one?"** It is LIPP's FMCD fit (window search on D and U_T, a = 1/U_T, base L/2, tertile fallback) followed by the longest run of equal floor(a·(k − anchor) + base) slots. The differences are kept as exact 128-bit integers, but the final floor of a double product decides keys within ~10³ units of a slot boundary by rounding, so CD is LIPP-faithful, not bit-reproducible: books 2M gives 11 (ours, and LIPP-style doubles) vs 10 (exact rationals) [hardness.hpp:119-127; verification/metrics/realdata.log]. On double features the +10⁻⁶ epsilon is scaled by the mean gap; the literal-epsilon value is reported beside it [hardness.hpp:149-159].

9. **"Are the PLA counts really PGM's?"** Yes, verified on the full books file: 262,604 (ε = 32) and 97 (ε = 4,096) both from PGM's `make_segmentation` and from ours [verification/pla/books_full_ref.json, books_full_hardness.json], plus 320 DP-oracle differentials [test_hardness.cpp:83-90]. Ours additionally survives x_last = 2⁶⁴ − 1 by widening to 128 bits [hardness.hpp:287-294].

10. **"Why does smoothing lower the per-region SSE by 85 % on fb but raise the whole-sequence RMSE?"** The regions minimise their own OLS SSE (3.80×10⁹ → 5.87×10⁸ summed over 489 regions), while the hardness tool's `smoothed` block fits ONE line over the concatenated 2.2M-entry sequence (RMSE 1,448 → 1,591) [results/aidb/hardness.json sample vs sample_csv]. A per-region optimum does not imply a better global line; the index only ever uses the per-region models, and its measured `fence_probes/op` fall from 2.60 to 2.18 with `correction_distance/op` 0.257 → 0.088 [results/aidb/sweep/results.jsonl fb packed_rank vs packed_rank_vp10].

11. **"What happens to the virtual points on inserts?"** Nothing until the region's delta reaches 64 entries; then the compaction re-places the stored virtual feature values among the new keys in O(n + v), recomputes the SSE with and without them, and drops them if they no longer help; the greedy search is not re-run unless `--relearn 1` [index.hpp:289-303]. Re-running it made the 10 %-insert workloads 116-232× slower [results/learnability/summary.csv].

12. **"Why is the parallel bulk load safe and deterministic?"** Regions are independent: each `rebuild` writes only its own `Region` and reads a const `Config`/`FlowTransform`; the partition is static (thread w builds regions w, w+W, …); exceptions are captured per thread and rethrown; the resulting layout is byte-identical for 1 and 4 threads in the test [index.hpp:433-441; test_main.cpp:143-149]. Queries stay single-threaded.

13. **"Why measure throughput on the same keys that were loaded, with no misses?"** Because that is the AIDB protocol ("the index throughput under a uniform read-only workload over the same dataset used to populate the index") [AIDB §2.2]; the paper's own workload is "random lookups for all keys in the dataset" with "a warm-up phase of 20 million lookups and then a measurement phase of 100 million lookups" on 200M-key datasets [AIDB §4.1 "Workloads and Measurements"]. Our sweeps all set `load-ratio 1, miss 0, query-distribution uniform, latency 0, instrument 1, build-threads 16` and differ in size: the uniform/window 2M sweeps use ops 1,000,000, warmup 200,000, seeds 11/29/47 [results/aidb/sweep/environment.json config.common]; `aidb_final` uses ops 5,000,000, warmup 500,000, qos 1, seeds 11/29/47 [results/aidb_final/config.json]; `aidb_fullscale` uses ops 2,000,000, warmup 200,000, qos 1, single seed 11 on the 200M-key files [results/aidb_fullscale/config.json]; `aidb_granularity` uses ops 1,000,000, warmup 200,000, seeds 11/29 [results/aidb_granularity/config.json]. And, as said in section 10, all scored throughputs are on the 2M samples, not on the 200M-key files.

14. **"Which numbers in the learnability block are wall-clock and therefore not reproducible?"** `smoothing_ns`, `transform_ns` (summed across build threads), `build_ns`, and of course `throughput_ops_s`. Everything else (regions, flow_regions, virtual_points, SSEs, tail conflicts, choices, cost estimates, root estimates) is deterministic [benchmark.cpp:69-78; index.hpp:482-493].

15. **"What is the difference between `coordinate_probes` and `fence_probes`, and why is the former always ≈ 5?"** Coordinate probes are the lower-bound binary search of the real-valued prediction against the 32 block boundaries in slot (or byte) space: 5 or 6 comparisons depending on where the prediction lands (31 of the 33 outcomes cost 5, 2 cost 6; measured mean 5.029), and independent of model *accuracy* only in the sense that the count never grows with the error [index.hpp:122-127; scratch/fix05_checks.py]; fence probes are the exact-key comparisons that fix the hint and grow with the correction distance [index.hpp:135-150]. A better model shows up in `fence_probes` and `correction_distance`, never in `coordinate_probes`.
