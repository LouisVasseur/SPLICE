# 04. The ten datasets and our samples

**What this section gives you.** Everything about the data behind the SCALE-LI AIDB lane: where the ten
200M-key files come from and how we checked them (URLs, sizes, sha256, sorting, duplicates); what each
dataset's keys are according to the AIDB 2026 paper and, where those papers describe them, NFL and CSV;
the exact key ranges and full-file hardness numbers (RMSE, ME, CD, PLA-32, PLA-4096) of every dataset,
with and without the NFL-style transform; a master table sorted by PLA-32 with the two values the AIDB
paper prints marked; the exact recipe for the two 2M-key samples (uniform, seed 42; contiguous window at
a seeded offset) and what each sampling mode preserves or destroys, calibrated against a control sample
of a perfectly linear file; every sample's hardness in the four sample scopes and its mean per-region
tail conflict degree; how each sample's flow was trained and what the transform does to the metrics; how
the CSV-style augmented scopes were produced; and a list of questions the supervisor may ask, with
answers. Every number is copied from the PDFs, the code or the result files, cited inline in brackets;
worked numbers were computed with the scratch scripts under `results/aidb/guide_deep/scratch/`
(`data_stats.py`, `tables.py`, `linear_uniform_hardness.json`). Our index components are clean-room
CONTROLS inside the SCALE-LI experimental map; nothing here is a reproduction of NFL, AFLI, CSV or the
AIDB benchmark.

Notation: keys k₁ < … < kₙ (uint64), rank r(kᵢ) = i − 1 ∈ {0..n−1}; linear model f(k) = w·φ(k) + b with
feature φ (raw key, or flow output z(k)); slot s(kᵢ) = rank plus the number of virtual points before kᵢ;
virtual budget λ = α·n; conflict degree D and tail conflict degree D₉₉; region = 4,096 keys; block =
128 keys.

---

## 1. Provenance: the ten GRE files

### 1.1 Where the bytes come from

- URL pattern: `https://www.cse.cuhk.edu.hk/mlsys/gre/<name>` for `<name>` ∈ {books, fb, osm, covid,
  genome, history, libio, planet, stack, wise} [results/aidb/provenance.json `<name>.url`]. The list and
  the URL come from the GRE benchmark's own `datasets/download.sh`
  (`https://github.com/gre4index/GRE/blob/master/datasets/download.sh`) [provenance.json `<name>.source`];
  our resumable copy of that script is `data/external/gre/download.sh` (curl with `--retry 8`, four
  parallel downloads).
- Why these ten: they are the ten datasets of Table 1 of the AIDB paper [AIDB Table 1, p.4], which the
  paper describes as "a superset of the datasets used in recent work on learned indexes", each containing
  "200 million 64-bit unsigned integer keys without duplicates" [AIDB §4.1, "Datasets"]. GRE (Wongkham
  et al., PVLDB 2022, AIDB ref. [27]) is where the paper's hardness baseline (PLA-32 / PLA-4096) comes
  from [AIDB §2.2].
- File format ("sosd"): an 8-byte little-endian count followed by count × 8-byte little-endian uint64
  keys [tools/datasets.py `header`, `sample`]. Size check: 8 + 8 × 200,000,000 = 1,600,000,008 bytes,
  which is the `bytes` field of all ten entries [provenance.json `<name>.bytes`] and the on-disk size
  (`ls -l data/external/gre/`).
- Retrieved 2026-09-21 between 14:47 and 14:59 UTC; sha256 verified 15:17 UTC the same day
  [provenance.json `retrieved_utc`, `verified_utc`].

### 1.2 Sortedness, duplicates and the `.sorted` copies

The GRE loader sorts at load time, and seven of the ten files are served **unsorted on disk**: covid,
genome, history, libio, planet, stack, wise [tools/aidb_pipeline.py:42 `GRE_UNSORTED`, audited
2026-09-21; hardness.hpp:50-53]. The pipeline's `sort` step runs `scaleli_hardness --sort-only 1
--write-sorted` on every file, records `sort_audit = {sorted, duplicates, written, written_keys}` and
writes `data/external/gre/<name>.sorted` for the unsorted ones [aidb_pipeline.py:374-380;
src/hardness.cpp:93-95]. The sort is a chunked `std::sort` + pairwise `std::inplace_merge`, identical in
result to `std::sort` [hardness.hpp:57-77]. Every step after `verify-downloads` uses the sorted copy when
it exists, else the download [aidb_pipeline.py:278-283].

Audit result: `sorted: true` for books, fb, osm; `sorted: false, written_keys: 200000000` for the other
seven; **`duplicates: 0` for all ten** [provenance.json `<name>.sort_audit`]. So "without duplicates"
[AIDB §4.1] holds for the files as served, and n = 200,000,000 exactly in every full-file metric below.

**T1. Checksums.** Download sha256 [provenance.json `<name>.sha256`]; sorted-copy sha256 [provenance.json
`<name>.sample.source_sha256`, which is the sample step's hash of the file it read]; sample sha256
[`data/samples/<name>_2M_{uniform,window}_s42.manifest.json` `sample_sha256`].

| dataset | served sorted? | download sha256 | .sorted sha256 | uniform sample sha256 | window sample sha256 |
|---|---|---|---|---|---|
| books | yes | `c71eb78eebb746f0e525e47d3d7e6b097f680c45cf17f1ead1d1f22a8950f939` | (same file) | `c2484df60c8b2f3100c408330453e0aca321ba5af6d0d23312a63bdd567f3eca` | `76fb950be335b5bcf4ee01fe615edacd3581441385d130df07f3ed571b97884a` |
| fb | yes | `22d5fd6f608e528c2ab60b77d4592efa5765516b75a75350f564feb85d573415` | (same file) | `7ecf95742784e7f3ef7bae5c5f3ac5c5924126a023927cb90e371d0074aeac06` | `a26c9b3809d2089a0efeff0ec90cdcc84a40fdc81203b0a4c7600433e0858b8c` |
| osm | yes | `1d1f5681cbfcd3774de112533dccc4065b58b74adf09684e9ad47298d1caa9e0` | (same file) | `b832a46ba727d96dd16acd70e909b4dc9a232dad0d116b850a2373a865d84ceb` | `7c3c37cb82936d9741a4451de9aeba95d00f95191a1631c89611862c79d9f67c` |
| covid | no (sorted copy written) | `96834e0f0a9a697ac39ec30f37316b9cf092056bcd9332e59ce131e891c0a3a8` | `cdc34a55f609691b57e316fd11c46253477df63b1296a7f05314e701d7c342ba` | `5dee4e179c89b1e59ddba6578bc96614267d3bfc3e91190acb33447f405ff268` | `62128272374874002adc29b6fddccd9ee408b278f602e7b028ed959167356644` |
| genome | no (sorted copy written) | `6862bff81727ec2bd0df54e9e08c60758ef8b3ee9bd800ffdc4eb68bd7680daa` | `2b0b5ae57a5cd00f504eb8226801ab8311cec1a2e3fb0ec3594b5bb38457c739` | `fb507fcd79f8bddfb0d2b93f1f087e02c9eed8f2c7cc573a994e5289cf606c24` | `61dc34adb69499a2c439240c3591b689a9fd967ce61e615ca3fbef6479651a2a` |
| history | no (sorted copy written) | `2556096e02969666fc4f48737330b1ec2788588481f54181816563f193efedd4` | `c2662355dd437e5d6f4a8e581a18cee07c2b2c184634afe05dd8a0e59311cd42` | `186080bdf75a972ba156edbc681fa9c49f0458bf7f61f9b1037cfcacc86ee662` | `997e2743bdbc42f5f5e31a8c978d29d2905b51141876898d9db5390a9e70d55e` |
| libio | no (sorted copy written) | `0d0c99a064a1142c48f4a5887429f272113ed8daa2f407072f3d5c2f18b2975b` | `f9f868375134273548644708f42365b4d3e0b952b5ce211f205b822e9440aea0` | `9156ef37030a4861635ec9f621b7678b7f058c2ce68400e4b67f0a9de7fdb237` | `4f911d41d6c4579014c4d7e745075e6bb33b68ce6759d8923642c4335e0a8362` |
| planet | no (sorted copy written) | `0e4a90c81e6e36f198aa674e9773c366e55e27430b208780d559a25e97324ef5` | `d6680d134075890173d5a9d5f2af991eb13ed794da153b5a67d90cec98a4628b` | `3a0fcf47ba903c33e9f6c6ed9bdc233918489751dbc0b9a41b0f5e205125c045` | `e15c214776a67511ec4211f7c888776cb71ebcb165b4569bf426fc9aa1a7c8e2` |
| stack | no (sorted copy written) | `34f9a3ae69ac17ed80a054d3f8ed9cebf0d86dca61eb502ca634daea9469a09d` | `1e261e45f0058731afb84a2caa37fec957e49fa5c4c622333696f43a1530bacc` | `1cb323ad7b26ed8bb637684f2a12b5558e6a052520504eafa82b0644c490abdb` | `7b3764a00f6e4f94883127deb13e31c919afb0a1855fdd87d0e0ff60b02b120a` |
| wise | no (sorted copy written) | `87e34fb1503881b19e2f172fee4855919544432e79b96f6bceed33e5c9805dbb` | `6353bad843367ceda822189c223b6ecea76cd606d824f9483d2a45fb7afa7f88` | `b5b5240959dc12f187b487b25c1d1dc3e08c715994d7a85827effeb2eadf0709` | `9d038460726c16aad533cee4d1a9527937735e8890a9a0e0bdb82215008abd6a` |

Note the window-sample sha256 values are not in `provenance.json` (the provenance entry records the
uniform sample only, because `results/aidb/` was the uniform run); they come from the window manifests.

---

## 2. The metrics used below, exactly as computed

All five hardness metrics follow the AIDB paper's list [AIDB §4.1 "Metrics under Test"] and are computed
by `scaleli_hardness` on the sorted feature sequence x₀ ≤ … ≤ xₙ₋₁ with rank target yᵢ = i
[hardness.hpp:1-10]. Raw uint64 keys are handled with exact `__int128` differences; flow outputs and
smoothed slots are doubles [hardness.hpp:31-34; src/hardness.cpp:100].

```
RMSE, ME   least-squares line y ≈ a·(x − x₀) + b over all n points (Welford accumulation);
           RMSE = sqrt( Σᵢ (i − a·(xᵢ−x₀) − b)² / n ),  ME = maxᵢ |i − a·(xᵢ−x₀) − b|     [hardness.hpp:84-106]

CD         LIPP's FMCD bulk-load model (Wu et al. 2021, lipp.h build_tree_bulk_fmcd), capacity
           L = n·(gap+1) with gap = 1 for n ≥ 1e6 (so L = 2n), search for the smallest D such that
           every D-apart pair of keys is at least U_T = (k[n−1−D] − k[D])/(L−2) + ε apart, ε = 1e-6
           on integer keys; slot(k) = floor(a·(k − anchor) + L/2) clamped to [0, L−1] with a = 1/U_T;
           CD = the longest run of consecutive keys with the same slot.       [hardness.hpp:109-227]
           On real-valued features ε is scaled to 1e-6 × mean gap (documented deviation from LIPP's
           absolute 1e-6, which would dominate O(1)-range flow outputs).      [hardness.hpp:151-166]

PLA-ε      number of segments of the optimal ε-bounded piecewise-linear approximation (O'Rourke 1981),
           counted exactly as PGM-index's make_segmentation does, including the (x_last+1, n) sentinel;
           ε ∈ {32, 4096} = the GRE local/global dimensions.                  [hardness.hpp:230-312]

tail CD    NFL Definitions 3.1/3.2: fit one line feature→rank, floor predictions into
D₉₉        [0, 1.5·n) slots, count keys per occupied slot, take the value at the 99th percentile of the
           occupied-slot counts and subtract one (NFL: "t = INT(m × γ)", γ = 0.99 [NFL §3.1.1]).
                                                              [transform.hpp:84-108; train_flow.py 71-86]
```

The pipeline computes the metric blocks with `--region-keys 4096 --pla-eps 32,4096 --check-sorted 1`
[aidb_pipeline.py:451]. Scopes in `hardness.json` map to `scaleli_hardness` runs as follows
[aidb_pipeline.py:449-454, 472-474]:

| scope | run | input | block |
|---|---|---|---|
| `full` | `--data <full> --flow <flow>` | 200M sorted keys | `original` |
| `full_flow` | same run | z(k) of the 200M keys through the SAMPLE-trained flow | `transformed` |
| `sample` | `--data <sample> --flow <flow> --virtual-alpha 0.1` | 2M sample keys | `original` |
| `sample_flow` | same run | z(k) of the 2M keys | `transformed` |
| `sample_flow_csv` | same run | z(k) + virtual points per region | `smoothed` |
| `sample_csv` | `--data <sample> --virtual-alpha 0.1` (no flow) | double(k − min) + virtual points | `smoothed` |

CD carries a ±1 rounding uncertainty (floor of a double product, as in LIPP) [hardness.hpp:119-127;
hardness_details.json `cd_note`]; treat CD differences of one as ties.

---

## 3. The ten datasets, one by one

Descriptions are quoted from Table 1 of the AIDB paper [AIDB Table 1, p.4] with the paper's source
reference; anything beyond that is either from the CSV or NFL papers (cited) or measured by us on the
files (`scratch/data_stats.py`, `hardness_details.json`). Key statistics (min, max, percentiles) are from
the sorted files [scratch/data_stats.json]; "p50" is the key at rank ⌊0.5·(n−1)⌋. Full-file metrics are
[results/aidb/hardness.json `<name>.full` / `.full_flow`]; the flow in `full_flow` is the one trained on
the 2M **uniform** sample (Section 6).

**T3. Key statistics of the ten sorted files** [scratch/data_stats.json].

| dataset | min | max | range = max−min | log₂(range) | p50 | p99 | p99.99 | mean gap = range/(n−1) |
|---|---|---|---|---|---|---|---|---|
| books | 0 | 9,223,372,036,854,784,874 | 9,223,372,036,854,784,874 | 63.00 | 1,920,975,649,179,691,136 | 7,690,015,317,089,125,888 | 9,132,078,801,956,411,904 | 46,116,860,414.858 |
| fb | 1 | 18,446,744,073,709,551,615 | 18,446,744,073,709,551,614 | 64.00 | 38,752,453,239 | 76,546,751,873 | 77,299,521,445 | 92,233,720,829.716 |
| osm | 33,246,697,004,540,789 | 13,748,550,930,623,082,253 | 13,715,304,233,618,541,464 | 63.57 | 5,170,332,552,509,244,517 | 10,737,454,289,360,892,995 | 13,686,332,700,453,684,633 | 68,576,521,510.975 |
| covid | 1,344,795,470,900,715,522 | 1,446,626,459,003,486,210 | 101,830,988,102,770,688 | 56.50 | 1,391,398,680,826,101,761 | 1,443,581,128,888,127,494 | 1,446,589,962,841,907,209 | 509,154,943.060 |
| genome | 2,489,750 | 446,640,429,470 | 446,637,939,720 | 38.70 | 280,552,045,122 | 445,347,172,802 | 446,612,275,232 | 2,233.190 |
| history | 83 | 9,178,997,263 | 9,178,997,180 | 33.10 | 4,473,873,432 | 9,085,796,222 | 9,178,114,938 | 45.895 |
| libio | 21,335 | 527,206,042 | 527,184,707 | 28.97 | 286,981,400 | 523,018,907 | 527,161,946 | 2.636 |
| planet | 1 | 9,178,997,250 | 9,178,997,249 | 33.10 | 776,139,014 | 8,988,035,135 | 9,177,243,630 | 45.895 |
| stack | 1 | 238,071,030 | 238,071,029 | 27.83 | 121,403,258 | 235,463,774 | 238,041,730 | 1.190 |
| wise | 8,796,093,034,805 | 17,592,186,032,162 | 8,796,092,997,357 | 43.00 | 13,104,235,008,794 | 17,530,874,982,592 | 17,591,561,268,342 | 43,980.465 |

A quick global-linearity test: for a perfectly linear file p50 sits at 50 % of the range. Measured
(p50 − min)/range: stack 0.510, history 0.487, libio 0.544, covid 0.458, wise 0.490, genome 0.628, osm
0.375, books 0.208, planet 0.085, fb 2.1e-9 (fb's range is set by five outliers, see below).

### 3.1 books — "Amazon book sales popularity" [AIDB Table 1, source [7] = SOSD, Kipf et al. 2019]

- NFL calls the same source "the amazon (AMZN) dataset ... book sale popularity data on the Amazon [35]"
  (Kaggle sales-rank data) [NFL §4.1.1]. What the integer encodes beyond that is not described in the
  papers.
- Keys 0 … 9,223,372,036,854,784,874 (= 2⁶³ + 9,066); 245 keys are ≥ 2⁶³ and 30,130,154 are ≥ 2⁶²
  [scratch/data_stats.json `books.count_ge`]. The distribution is strongly skewed towards small keys:
  p50 at 20.8 % of the range, p90 at 58 %.
- Full: RMSE 18,053,637.5, ME 96,380,337.9, CD 246, PLA-32 262,604, PLA-4096 **97** [hardness.json
  `books.full`]. RMSE/n = 0.090 and ME/n = 0.48: a single line is a poor global fit, yet PLA-4096 = 97 is
  the smallest of all ten: the CDF is smooth and concave (few global pieces) but far from a line.
- With the uniform-sample flow: RMSE 15,985,128.0 (−11.5 %), ME 78,944,811.5, CD 245, PLA-32 262,604,
  PLA-4096 94 [`books.full_flow`]. The flow's tanh bends the concave CDF a little straighter; PLA is
  untouched because a monotone reparametrisation cannot change the number of ε-segments unless it
  creates or removes collinearity (it changed PLA-32 by 0 to 5 segments on every dataset).
- Character: globally curved and skewed, moderately clustered (CD 246), locally moderate. The uniform
  sample looks trivially easy (PLA-32 602, at the sampling-noise floor of Section 5.3), the window does
  not (PLA-32 5,789, mean per-region tail CD 10.76).

### 3.2 fb — "Upsampled Facebook user ID" [AIDB Table 1, source [7] = SOSD]

- NFL: "an upsampled version of a Facebook user ID dataset [36]" [NFL §4.1.1]; CSV: "200 million integer
  Facebook user IDs [30]" [CSV §6.1 "Datasets"]. The upsampling procedure is not described in the three
  papers.
- Keys 1 … 18,446,744,073,709,551,615 = 2⁶⁴ − 1. **The range is set by a handful of outliers**: only 21
  keys are ≥ 2⁴⁰ (1.1e12), 9 are ≥ 2⁵⁶ and 5 are ≥ 2⁶³ [scratch/data_stats.json `fb.count_ge`]; the last
  five keys are 10,248,691,552,019,458,048; 12,298,204,682,441,981,952; 14,347,717,812,864,503,808;
  16,397,230,943,287,027,712; 18,446,744,073,709,551,615, spaced by 2,049,513,130,422,523,904 ≈ 2⁶⁴/9
  [`fb.tail`, `fb.top_gaps`]. The other 199,999,979 keys lie in [1, 1.1e12]; p99.99 = 77,299,521,445.
- Full: RMSE 57,735,023.7, ME 99,999,995.5, CD 110, PLA-32 1,055,308, PLA-4096 1,687 [`fb.full`].
  **Worked check**: with five keys at ~10¹⁹ and everything else below 10¹², the least-squares line is
  pinned by the outliers and is essentially constant over the bulk; the residual of a constant predictor
  against ranks 0..n−1 has RMSE n/√12 = 200,000,000/3.4641 = 57,735,026.9 and ME n/2 = 100,000,000. The
  measured values are 57,735,023.7 (ratio 1.000000) and 99,999,995.5. So fb's RMSE and ME say nothing
  about the 199,999,979 ordinary keys; they measure the outliers.
- With the uniform-sample flow: RMSE 1,161,273.7 (−98.0 %), ME 352,904,667.5 (larger than n: the flow's
  saturated tanh maps the five outliers to z ≈ v₀+v₁ = 4.67 while the bulk spans z ∈ [0, 1.67], so the
  line now fits the bulk and the outliers become the residual), CD 110, PLA-32 1,055,308, PLA-4096 1,687
  [`fb.full_flow`].
- Character: a near-linear body (uniform-sample RMSE 1,448.2 is the smallest of all ten, 5× the noise
  floor) with extreme outliers and strong **local** irregularity: window PLA-32 10,599, window CD 96,
  mean per-region tail CD 31.41 (second-highest after genome's window). CSV's Figure 5 calls Facebook
  "easy" with "almost globally linear CDFs" [CSV §6.1]; the AIDB Figure 2 places it at the top of the
  PLA-32 axis [AIDB Fig. 2]. Both are right: they look at different scales.

### 3.3 osm — "Uniformly sampled OpenStreetMap locations" [AIDB Table 1, source [7] = SOSD]

- CSV: "200 million locations randomly sampled from OpenStreetMap and represented as Google S2 [26] cell
  IDs [24]" [CSV §6.1]; CSV also calls OSM and Genome "hard datasets" [CSV §6.1].
- Keys 33,246,697,004,540,789 … 13,748,550,930,623,082,253 (log₂ range 63.57). Extremely clustered: p25 =
  5.06e18 and p75 = 6.09e18, i.e. half of all keys sit in 7.5 % of the range (S2 cells over land masses).
- Full: RMSE 24,177,498.5, ME 67,384,214.8, **CD 4,107** (by far the highest), PLA-32 661,115, **PLA-4096
  5,495** (highest) [`osm.full`]. FMCD's fit: U_T = 3.41e10 key units per slot, D = 4,106 [hardness_details
  `osm.full_original.fmcd`]: 4,107 keys fall in one 3.4e10-wide slot somewhere.
- With the flow: RMSE 23,977,602.0 (−0.8 %), CD 3,943, PLA unchanged [`osm.full_flow`]: a 2×2 tanh
  cannot un-cluster a stepped CDF.
- Character: globally hard (steps and plateaus; the only dataset whose uniform-sample CD stays in the
  thousands: 1,102), moderately hard locally (window PLA-32 4,919, tail CD 9.66: inside one 0.144 %-wide
  slice of the key range the cells are fairly regular).

### 3.4 covid — "Uniformly sampled Tweet ID with tag COVID-19" [AIDB Table 1, source [16] = Lopez & Gallemore 2021]

- CSV: "200 million integer tweet IDs randomly sampled from tweets tagged with 'Covid-19' [18]"; CSV
  calls it an easy dataset with an almost globally linear CDF and, zoomed in, the one dataset that does
  not "deviate from linear CDFs at local level" [CSV §6.1]. What the tweet-ID integer encodes is not
  described in the papers.
- Keys 1,344,795,470,900,715,522 … 1,446,626,459,003,486,210, range 1.018e17 (log₂ 56.5), p50 at 45.8 %
  of the range.
- Full: RMSE 1,795,722.5, ME 8,133,078.0, CD 27, PLA-32 81,908, PLA-4096 850 [`covid.full`]. With the
  flow: RMSE 1,272,460.5 (−29.1 %), CD 27, PLA-32 81,909, PLA-4096 850 [`covid.full_flow`].
- Character: near-linear globally (RMSE/n = 0.009) and the easiest locally together with stack, history
  and wise: window PLA-32 827, mean per-region tail CD 3.02 in the window and 3.16 in the uniform sample
  (the noise floor is 3.00). The window's CD 24 and the full CD 27 agree (ratio 1.12): the clusters that
  set CD are local, so a window sees them at full resolution.

### 3.5 genome — "Loci pairs in human chromosomes" [AIDB Table 1, source [21] = Rao et al. 2014, Cell]

- CSV: "200 million entries of loci pairs in human chromosomes represented as integers [25]", a "hard"
  dataset that deviates from linearity "at local level, especially Genome" [CSV §6.1]. The integer
  encoding of a locus pair is not described in the papers.
- Keys 2,489,750 … 446,640,429,470 (log₂ 38.7), mean gap 2,233; p50 at 62.8 % of the range.
- Full: RMSE 7,531,940.1, ME 19,622,982.5, CD 585, **PLA-32 1,290,208** (highest of all ten), PLA-4096
  1,426 [`genome.full`]. With the flow: RMSE 8,390,195.9 (+11.4 %: the transform hurts here), CD 587,
  PLA-32 1,290,208, PLA-4096 1,425 [`genome.full_flow`].
- Character: only mildly curved globally (RMSE/n 0.038) but the roughest locally: window PLA-32 12,719
  (highest window value), window CD 171, mean per-region tail CD **60.41** (highest anywhere). The
  uniform sample hides all of it (PLA-32 2,045, tail CD 4.23).

### 3.6 history — "History node ID in OpenStreetMap" [AIDB Table 1, source [5] = Google Cloud OSM]

- Not described further in the papers. Keys 83 … 9,178,997,263 (log₂ 33.1), mean gap 45.9, p50 at 48.7 %
  of the range: history and planet (below) share the same ID space (their maxima differ by 13) but have
  very different shapes.
- Full: RMSE 815,542.7, ME 2,303,085.2, CD 8, **PLA-32 105,468, PLA-4096 468** [`history.full`]. These are
  exactly the two values the AIDB paper prints for history, "h_l = 105k, h_g = 468" [AIDB §2.2], which is
  the strongest evidence that our metric code and the paper's agree on the same file. With the flow:
  RMSE 432,676.2 (−46.9 %), CD 8, PLA-32 105,469, PLA-4096 467 [`history.full_flow`].
- Character: near-linear globally (RMSE/n 0.0041, lowest with stack) and easy locally (window PLA-32 943,
  tail CD 3.10 in both modes). Note the AIDB paper's point: LIPP does 5.6 MOPS on this "relatively easy"
  dataset versus 7.1 MOPS on the "relatively hard" libio [AIDB §2.2].

### 3.7 libio — "Repository ID from libraries.io" [AIDB Table 1, source [14] = libraries.io data]

- Not described further. Keys 21,335 … 527,206,042 (log₂ 29.0), mean gap **2.636**: about 38 % of the
  integers in the range are present, so the sequence is close to consecutive IDs with holes.
- Full: RMSE 3,445,772.9, ME 11,828,007.2, **CD 2** (lowest with stack), **PLA-32 145,808, PLA-4096 639**
  [`libio.full`], matching the paper's "h_l = 146k, h_g = 639" [AIDB §2.2]. With the flow: RMSE
  4,072,387.5 (+18.2 %), CD 2, PLA-32 145,811, PLA-4096 638 [`libio.full_flow`].
- Character: mildly curved globally (p50 at 54 %, RMSE/n 0.017) but so dense that FMCD never collides:
  window CD 2, mean per-region tail CD 1.61 (window) — the second-easiest local structure after stack.

### 3.8 planet — "Planet ID in OpenStreetMap" [AIDB Table 1, source [5] = Google Cloud OSM]

- Not described further. Keys 1 … 9,178,997,250 (same ID space as history), but p50 = 776,139,014 sits at
  8.5 % of the range and p75 at 33.8 %: most IDs are small (a heavily front-loaded, convex-then-flat CDF).
- Full: RMSE 29,771,622.2 (RMSE/n 0.149, second-highest after fb), ME 60,462,049.5, CD 21, PLA-32 613,597,
  PLA-4096 2,314 [`planet.full`]. With the flow: RMSE 15,545,175.8 (−47.8 %, the largest genuine
  improvement: a tanh bends a front-loaded CDF the right way), CD 32, PLA-32 613,602, PLA-4096 2,313
  [`planet.full_flow`].
- Character: globally curved, locally rough as well (window PLA-32 8,498, window CD 49, mean per-region
  tail CD 35.78). Uniform-sample tail CD 7.66 is the third-highest of the uniform samples.

### 3.9 stack — "Vote ID from StackOverflow" [AIDB Table 1, source [22] = archive.org stackexchange dump]

- Not described further. Keys 1 … 238,071,030, mean gap **1.190**: 84 % of all integers in the range are
  keys; the file is almost the sequence 1, 2, 3, … with 16 % holes (head: 1, 2, 3, 4, 5).
- Full: RMSE 839,727.0, ME 2,444,700.0, **CD 1, PLA-32 17,833, PLA-4096 133** (all three the lowest of
  the ten) [`stack.full`]. With the flow: RMSE 1,769,181.3 (+110.7 %: the flow can only add curvature to
  an already straight CDF), CD 1, PLA-32 17,833, PLA-4096 129 [`stack.full_flow`].
- Character: the easiest dataset at every scale. Window: CD 1, PLA-32 144, mean per-region tail CD
  exactly 1.0000; the CSV-style smoother could place only 62,282 of the 199,707 budgeted virtual points
  in the window (31 %) because there was nothing left to straighten (Section 7).

### 3.10 wise — "Partition key from the data returned by the Wide-field Infrared Survey Explorer (WISE)" [AIDB Table 1, source [28] = Wright et al. 2010]

- Not described further. Keys 8,796,093,034,805 … 17,592,186,032,162: the whole file sits inside
  [2⁴³, 2⁴⁴) (2⁴³ = 8,796,093,022,208, 2⁴⁴ = 17,592,186,044,416), min = 2⁴³ + 12,597, max = 2⁴⁴ − 12,254.
  p50 at 49.0 % of the range.
- Full: RMSE 2,200,401.6, ME 4,988,460.6, CD 10, PLA-32 79,035, PLA-4096 382 [`wise.full`]. With the
  flow: RMSE 1,511,720.5 (−31.3 %), CD 9, PLA-32 79,036, PLA-4096 382 [`wise.full_flow`].
- Character: near-linear globally (RMSE/n 0.011) and easy locally (window PLA-32 805, tail CD 3.01).

---

## 4. Master table (full files, sorted by PLA-32)

**T2.** All values [results/aidb/hardness.json `<name>.full`]; n = 200,000,000. The two datasets whose
GRE values the AIDB paper prints are marked; both match to the printed precision [AIDB §2.2].

| rank | dataset | RMSE | ME | CD | PLA-32 (h_l) | PLA-4096 (h_g) | RMSE/n | ME/n |
|---|---|---|---|---|---|---|---|---|
| 1 | stack | 839,727.0 | 2,444,700.0 | 1 | 17,833 | 133 | 0.0042 | 0.0122 |
| 2 | wise | 2,200,401.6 | 4,988,460.6 | 10 | 79,035 | 382 | 0.0110 | 0.0249 |
| 3 | covid | 1,795,722.5 | 8,133,078.0 | 27 | 81,908 | 850 | 0.0090 | 0.0407 |
| 4 | **history** (paper: h_l = 105k, h_g = 468) | 815,542.7 | 2,303,085.2 | 8 | **105,468** | **468** | 0.0041 | 0.0115 |
| 5 | **libio** (paper: h_l = 146k, h_g = 639) | 3,445,772.9 | 11,828,007.2 | 2 | **145,808** | **639** | 0.0172 | 0.0591 |
| 6 | books | 18,053,637.5 | 96,380,337.9 | 246 | 262,604 | 97 | 0.0903 | 0.4819 |
| 7 | planet | 29,771,622.2 | 60,462,049.5 | 21 | 613,597 | 2,314 | 0.1489 | 0.3023 |
| 8 | osm | 24,177,498.5 | 67,384,214.8 | 4,107 | 661,115 | 5,495 | 0.1209 | 0.3369 |
| 9 | fb | 57,735,023.7 | 99,999,995.5 | 110 | 1,055,308 | 1,687 | 0.2887 | 0.5000 |
| 10 | genome | 7,531,940.1 | 19,622,982.5 | 585 | 1,290,208 | 1,426 | 0.0377 | 0.0981 |

Reading the table: the five metrics rank the datasets differently, which is the AIDB paper's whole
point ("there is no metric that simultaneously achieves ideal conformance and ideal coverage" [AIDB §4.2]). By PLA-32 genome > fb > osm > planet > books;
by CD osm ≫ genome > books > fb; by RMSE fb > planet > osm > books ≫ genome; by PLA-4096 osm > planet >
fb > genome. The paper's own Figure 2 (GRE plane) and Figure 6 (CD·PLA-32 plane) show the same ten
points [AIDB Figs. 2, 6]; the exact coordinates of the other eight datasets are not printed in the
paper, so only history and libio can be cross-checked digit by digit.

The same ten metrics after the uniform-sample flow (`full_flow`) are in T7 (Section 6.3).

---

## 5. The samples

### 5.1 Exactly how the two samples are drawn

Both samples are produced by `tools/datasets.py sample <src> <dest> --n 2000000 --dtype uint64 --mode
{uniform|window} --seed 42`, invoked from the pipeline's `sample` step [aidb_pipeline.py:418-433, 430].
The source `<src>` is the sorted copy for the seven unsorted files and the download itself for books,
fb, osm [aidb_pipeline.py:278-283; provenance.json `<name>.sample.source`].

```
# tools/datasets.py:73-89 (verbatim logic)
total = 200_000_000; n = 2_000_000; rng = random.Random(seed)          # seed = 42
if mode == 'uniform':  indices = sorted(rng.sample(range(total), n))    # 2M distinct ranks, no replacement
elif mode == 'strided': indices = [(2*i+1)*total // (2*n) for i in range(n)]   # exists, not used
else:                   start = rng.randrange(total - n + 1)            # 'window'
                        indices = range(start, start + n)
write 8-byte count, then key[i] for i in indices  (mmap of the source, so keys keep source order = sorted)
manifest = {source, source_count, source_sha256, mode, seed, sample_count, dtype, sample_sha256, warning}
```

- **Uniform, seed 42**: 2,000,000 distinct ranks chosen without replacement from 0..199,999,999; every key
  has inclusion probability p = 0.01. The first sampled ranks are 132, 203, 373, 403, 407 and the last
  are 199,999,700, 199,999,709, 199,999,874; the mean rank gap is 99.99992 [scratch/data_stats.py, same
  RNG call]. Because the RNG is seeded identically for every dataset, **the same 2M ranks are taken from
  all ten files**.
- **Window, seed 42**: `random.Random(42).randrange(198,000,001)` = **171,644,825**, so every window is
  ranks 171,644,825 … 173,644,824, i.e. the slice starting at the 85.82 % quantile of each file
  [scratch/data_stats.py]. Verified for all ten datasets: the window sample's first and last keys equal
  `key[171,644,825]` and `key[173,644,824]` of the sorted source file [scratch/data_stats.json
  `<name>.window_matches = true`]. The manifests do not store the offset (they store mode, seed, counts
  and hashes) [data/samples/*_window_s42.manifest.json]; the offset is reproducible from the seed.
- Each sample file is 8 + 8 × 2,000,000 = 16,000,008 bytes; `sample_count: 2000000`, `dtype: uint64`,
  and the manifest carries the warning "A derived sample is NOT the full benchmark corpus; sampling
  changes local hardness." [manifest.json].
- Sample min/max: **T10** below [hardness_details.json `<name>.sample.min/max` in both result
  directories].

**T10. Uniform vs window samples, raw keys** (metrics [hardness.json `<name>.sample`]; tail CD = mean
per-region tail conflict degree from the sweeps, Section 5.4).

| dataset | uniform min | uniform max | window min | window max | window span / full range | uniform PLA-32 | window PLA-32 | uniform CD | window CD | uniform RMSE | window RMSE | uniform tail CD | window tail CD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| books | 3,241 | 9,223,372,036,854,781,810 | 4,735,252,013,344,826,368 | 4,880,676,988,429,512,704 | 1.577% | 602 | 5,789 | 11 | 27 | 180,591.3 | 1,709.1 | 3.28 | 10.76 |
| fb | 97,995 | 77,308,811,965 | 66,478,271,979 | 67,251,951,970 | 0.000% | 3,034 | 10,599 | 24 | 96 | 1,448.2 | 8,202.5 | 8.45 | 31.41 |
| osm | 42,262,281,243,463,475 | 13,748,274,210,235,485,307 | 9,771,619,518,438,332,541 | 9,791,353,970,024,941,583 | 0.144% | 7,099 | 4,919 | 1,102 | 152 | 241,476.1 | 118,136.5 | 36.16 | 9.66 |
| covid | 1,344,795,518,640,259,075 | 1,446,626,324,517,429,249 | 1,428,280,948,953,600,004 | 1,429,257,627,939,184,641 | 0.959% | 1,192 | 827 | 8 | 24 | 18,073.1 | 24,143.5 | 3.16 | 3.02 |
| genome | 46,509,831 | 446,638,295,046 | 417,037,340,916 | 419,441,740,061 | 0.538% | 2,045 | 12,719 | 77 | 171 | 75,352.0 | 48,827.0 | 4.23 | 60.41 |
| history | 81,021 | 9,178,989,449 | 7,808,774,263 | 7,898,935,777 | 0.982% | 1,067 | 943 | 7 | 8 | 8,104.9 | 9,651.7 | 3.10 | 3.10 |
| libio | 29,518 | 527,205,800 | 462,326,661 | 466,941,114 | 0.875% | 1,256 | 1,057 | 8 | 2 | 34,472.4 | 8,854.0 | 3.37 | 1.61 |
| planet | 457 | 9,178,968,583 | 5,721,028,025 | 5,986,820,672 | 2.896% | 3,456 | 8,498 | 26 | 49 | 297,556.5 | 42,645.1 | 7.66 | 35.78 |
| stack | 299 | 238,070,846 | 204,327,766 | 206,662,194 | 0.981% | 593 | 144 | 8 | 1 | 8,395.8 | 25,321.9 | 3.01 | 1.00 |
| wise | 8,796,104,780,969 | 17,592,181,572,984 | 16,325,355,473,712 | 16,408,098,198,591 | 0.941% | 787 | 805 | 8 | 8 | 21,926.3 | 5,551.0 | 3.03 | 3.01 |

Two things to notice immediately. (i) The fb uniform sample's maximum is 77,308,811,965 < 2⁴⁰: none of
fb's 21 outliers was drawn (with p = 0.01 per key the chance of drawing none is 0.99²¹ = 0.81), so the
uniform sample of fb has none of the global pathology of the full file. (ii) For a file with a linear
CDF, a 1 %-of-ranks window covers 1 % of the key range; the "window span / full range" column is
therefore a local-density indicator: covid, history, stack, wise, libio ≈ 0.9–1 % (uniform density where
the window sits), planet 2.9 % (sparse there), books 1.6 %, osm 0.14 % and genome 0.54 % (dense there),
fb 0.000 % (the window sits in the bulk, the range is the outliers).

### 5.2 What the two modes preserve and destroy: derivation

Let R(k) be a key's rank in the full file (n_F = 2·10⁸) and R_s(k) its rank in a sample.

*Uniform sample.* R_s(k) counts how many of the R(k) smaller keys were drawn, so R_s(k) ~ Binomial(R(k),
p) with p = 0.01: mean p·R(k), standard deviation √(R(k)·p·(1−p)) ≤ √(2·10⁸ · 0.0099) = 1,407 ranks.
Hence:

- Global shape is preserved and **RMSE and ME scale by exactly p**: the least-squares residual of the
  sample is p × the full residual plus binomial noise. Measured full/uniform ratios [T9]: RMSE 99.36–
  100.62 and ME 98.52–100.85 on nine datasets; fb is the exception (39,866 and 27,538) because the sample
  lost the outliers.
- Local structure at the 32-rank scale is destroyed: an ε = 32 corridor in sample ranks is a ±3,200-rank
  corridor in the full file, and on top of that the binomial jitter alone forces segments. The
  sampling-noise floor (Section 5.3) is PLA-32 = 535 for a perfectly linear file; books (602), stack
  (593), wise (787), history (1,067), covid (1,192), libio (1,256) all sit within 2.4× of it.
- CD is not scale-free either: FMCD's slot width U_T = (k[n−1−D] − k[D])/(2n − 2) is 100× wider on the
  sample (same key range, n/100), so CD measures cluster density at a 100× coarser key scale. Full/uniform
  CD ratios range from 0.12 (stack) to 22 (books) [T9]; only osm keeps a CD in the thousands (1,102).
- Consequence: the uniform samples are a **global-shape** benchmark. The mean per-region tail CD is
  3.0–3.4 on six of the ten uniform samples, i.e. at the noise floor of 3.00 (Section 5.3); only fb
  (8.45), osm (36.16), planet (7.66) and genome (4.23) keep anything above it [T5].

*Window sample.* Ranks are exact (R_s = R − 171,644,825), so every local statistic is measured at full
resolution, but the sample sees 1 % of the key range at one place:

- PLA-32 is preserved up to how typical the slice is: window PLA-32 × 100 vs full PLA-32 gives ratios
  full/window of 99.6 (fb), 99.0 (covid), 101.4 (genome), 98.2 (wise), 111.8 (history), 123.8 (stack),
  134.4 (osm), 137.9 (libio), 72.2 (planet), 45.4 (books) [T9]. For fb, covid, genome and wise the window
  is a faithful 1/100 of the file; for books and planet (non-stationary, front-loaded CDFs) the 85.8 %
  quantile slice is rougher than average.
- CD is preserved when the clusters that set it are local: full/window CD ratios 1.00 (history, libio,
  stack), 1.12 (covid), 1.15 (fb), 1.25 (wise), but 27 (osm), 9.1 (books), 3.4 (genome), 0.43 (planet)
  [T9].
- Global shape is destroyed: every window RMSE is tiny in absolute terms (1,709 books; 5,551 wise) and
  unrelated to the full RMSE (full/window RMSE ratio 33 to 10,564 [T9]).

**T9. Full-file / sample ratios** (n ratio = 100) [computed from hardness.json of both runs].

| dataset | RMSE full/uniform | ME full/uniform | CD full/uniform | PLA-32 full/uniform | PLA-4096 full/uniform | RMSE full/window | PLA-32 full/window | CD full/window |
|---|---|---|---|---|---|---|---|---|
| books | 99.97 | 100.19 | 22.36 | 436.2 | 12.1 | 10,563.5 | 45.4 | 9.11 |
| fb | 39,865.58 | 27,538.25 | 4.58 | 347.8 | 1,687.0 | 7,038.7 | 99.6 | 1.15 |
| osm | 100.12 | 100.21 | 3.73 | 93.1 | 80.8 | 204.7 | 134.4 | 27.02 |
| covid | 99.36 | 100.66 | 3.38 | 68.7 | 94.4 | 74.4 | 99.0 | 1.12 |
| genome | 99.96 | 99.84 | 7.60 | 630.9 | 79.2 | 154.3 | 101.4 | 3.42 |
| history | 100.62 | 98.52 | 1.14 | 98.8 | 156.0 | 84.5 | 111.8 | 1.00 |
| libio | 99.96 | 100.42 | 0.25 | 116.1 | 63.9 | 389.2 | 137.9 | 1.00 |
| planet | 100.05 | 100.02 | 0.81 | 177.5 | 178.0 | 698.1 | 72.2 | 0.43 |
| stack | 100.02 | 100.85 | 0.12 | 30.1 | 33.2 | 33.2 | 123.8 | 1.00 |
| wise | 100.35 | 100.21 | 1.25 | 100.4 | 18.2 | 396.4 | 98.2 | 1.25 |

**The fb example.** Full PLA-32 1,055,308 (rank 9 of 10). Uniform sample: PLA-32 3,034, CD 24, RMSE
1,448 — the "easiest-looking" body of all ten. Window: PLA-32 **10,599** (×100 = 1,059,900 ≈ full),
CD 96, RMSE 8,202, mean per-region tail CD 31.41 [T10]. The uniform sample turned fb into a near-linear
file with a little noise; the window kept the upsampled-ID roughness that makes fb hard for LIPP-style
slot mapping [NFL Table 1 on facebook; AIDB Fig. 6 on fb's CD].

**The osm example.** Full CD 4,107, PLA-4096 5,495 (the hardest global structure). Uniform sample: CD
1,102, PLA-32 7,099 (the highest uniform PLA-32), tail CD 36.16 (the highest uniform tail CD): the global
clustering survives sampling because clusters spanning 10¹⁷ key units are 100× wider than the sample's
slot width. Window: CD 152, PLA-32 4,919, tail CD 9.66: inside 0.144 % of the range the S2 cells are more
regular than fb's IDs. So osm is "hard" in the uniform lane and only "medium" in the window lane, while
fb and genome are the opposite — the two lanes are two different benchmarks and are reported separately
everywhere in this guide.

### 5.3 Control: the sampling-noise floor

To know how much of a uniform sample's hardness is sampling noise, we sampled a *perfectly linear* file
with the same RNG: keys = the 2M sampled ranks themselves (`sorted(random.Random(42).sample(range(2e8),
2e6))`, so the "full file" is 0, 1, …, 199,999,999) and ran the same `scaleli_hardness` binary with the
same flags [scratch/linear_uniform_2M_s42, scratch/linear_uniform_hardness.json]:

| metric | linear-file uniform sample (noise floor) | smallest real uniform sample | largest |
|---|---|---|---|
| RMSE | 289.5 | fb 1,448.2 | planet 297,556.5 |
| ME | 732.5 | fb 3,631.3 | books 961,972.8 |
| CD | 7 | history 7 | osm 1,102 |
| PLA-32 | 535 | stack 593 | osm 7,099 |
| PLA-4096 | 1 | fb 1 | osm 68 |
| mean per-region tail CD (489 regions) | 3.002 (min 3, max 4) | stack 3.008 | osm 36.16 |
| whole-sample tail CD | 3 | covid/history/libio/stack/wise 3 | osm 99 |

So for the uniform samples: PLA-32 below ~600, CD ≤ 8 and per-region tail CD ≈ 3 mean "indistinguishable
from a linear file with sampling noise"; those numbers should not be interpreted as dataset structure.
The FMCD fit on the control gives U_T = 49.9995 ranks per slot with D = 7 [linear_uniform_hardness.json
`original.fmcd`]: seven sampled ranks land in the same 50-rank slot somewhere in 2M draws, which is the
Poisson tail, not structure. (The CSV smoother on this control sample: PLA-32 535 → 225 with 179,985
virtual points; Section 7.)

### 5.4 Sample hardness in the four sample scopes

**T4a. Uniform samples** [results/aidb/hardness.json `<name>.{sample,sample_flow,sample_csv,
sample_flow_csv}`; `vp`, `regions`, `unordered` from hardness_details.json]. The uniform run's details do
not record the transformed-duplicates count (field absent; the window run's do, see T4b); `unordered = 0`
everywhere means the monotone flows kept key order.

| dataset | scope | keys | RMSE | ME | CD | PLA-32 | PLA-4096 | extra |
|---|---|---|---|---|---|---|---|---|
| books | sample | 2,000,000 | 180,591.3 | 961,972.8 | 11 | 602 | 8 |  |
| books | sample_flow | 2,000,000 | 159,882.7 | 787,945.4 | 9 | 602 | 8 | unordered=0 |
| books | sample_csv | 2,185,327 | 196,846.2 | 1,048,403.0 | 10 | 236 | 8 | vp=185,327, regions=489 |
| books | sample_flow_csv | 2,185,445 | 174,288.9 | 858,949.9 | 9 | 239 | 8 | vp=185,445, regions=489 |
| fb | sample | 2,000,000 | 1,448.2 | 3,631.3 | 24 | 3,034 | 1 |  |
| fb | sample_flow | 2,000,000 | 11,584.6 | 30,693.4 | 24 | 3,034 | 3 | unordered=0 |
| fb | sample_csv | 2,199,707 | 1,591.2 | 3,959.5 | 23 | 1,718 | 1 | vp=199,707, regions=489 |
| fb | sample_flow_csv | 2,199,707 | 12,742.1 | 33,743.0 | 23 | 1,721 | 3 | vp=199,707, regions=489 |
| osm | sample | 2,000,000 | 241,476.1 | 672,432.6 | 1,102 | 7,099 | 68 |  |
| osm | sample_flow | 2,000,000 | 239,472.1 | 589,442.9 | 1,100 | 7,099 | 68 | unordered=0 |
| osm | sample_csv | 2,199,605 | 265,521.5 | 739,686.0 | 1,099 | 7,491 | 76 | vp=199,605, regions=489 |
| osm | sample_flow_csv | 2,199,396 | 263,263.9 | 648,294.0 | 1,099 | 7,468 | 76 | vp=199,396, regions=489 |
| covid | sample | 2,000,000 | 18,073.1 | 80,794.0 | 8 | 1,192 | 9 |  |
| covid | sample_flow | 2,000,000 | 12,906.4 | 47,120.6 | 8 | 1,193 | 9 | unordered=0 |
| covid | sample_csv | 2,198,862 | 19,819.2 | 88,707.4 | 10 | 892 | 11 | vp=198,862, regions=489 |
| covid | sample_flow_csv | 2,198,904 | 14,147.7 | 51,691.0 | 11 | 892 | 9 | vp=198,904, regions=489 |
| genome | sample | 2,000,000 | 75,352.0 | 196,543.7 | 77 | 2,045 | 18 |  |
| genome | sample_flow | 2,000,000 | 83,934.3 | 214,604.9 | 82 | 2,045 | 18 | unordered=0 |
| genome | sample_csv | 2,199,707 | 82,874.1 | 216,173.7 | 71 | 1,627 | 19 | vp=199,707, regions=489 |
| genome | sample_flow_csv | 2,199,707 | 92,313.6 | 236,037.8 | 73 | 1,629 | 19 | vp=199,707, regions=489 |
| history | sample | 2,000,000 | 8,104.9 | 23,376.3 | 7 | 1,067 | 3 |  |
| history | sample_flow | 2,000,000 | 4,290.7 | 11,084.8 | 7 | 1,068 | 3 | unordered=0 |
| history | sample_csv | 2,195,413 | 8,628.3 | 24,888.9 | 21 | 411 | 3 | vp=195,413, regions=489 |
| history | sample_flow_csv | 2,195,568 | 4,846.6 | 12,761.0 | 21 | 412 | 3 | vp=195,568, regions=489 |
| libio | sample | 2,000,000 | 34,472.4 | 117,779.9 | 8 | 1,256 | 10 |  |
| libio | sample_flow | 2,000,000 | 40,748.0 | 138,497.3 | 7 | 1,255 | 10 | unordered=0 |
| libio | sample_csv | 2,195,924 | 37,442.8 | 128,489.6 | 21 | 614 | 11 | vp=195,924, regions=489 |
| libio | sample_flow_csv | 2,195,953 | 44,356.9 | 151,183.9 | 16 | 614 | 11 | vp=195,953, regions=489 |
| planet | sample | 2,000,000 | 297,556.5 | 604,478.5 | 26 | 3,456 | 13 |  |
| planet | sample_flow | 2,000,000 | 155,367.6 | 337,958.1 | 33 | 3,455 | 15 | unordered=0 |
| planet | sample_csv | 2,199,656 | 327,255.8 | 664,813.1 | 37 | 2,767 | 15 | vp=199,656, regions=489 |
| planet | sample_flow_csv | 2,199,699 | 170,882.5 | 371,498.2 | 33 | 2,771 | 15 | vp=199,699, regions=489 |
| stack | sample | 2,000,000 | 8,395.8 | 24,241.8 | 8 | 593 | 4 |  |
| stack | sample_flow | 2,000,000 | 17,705.8 | 42,459.6 | 7 | 594 | 4 | unordered=0 |
| stack | sample_csv | 2,184,990 | 9,036.9 | 26,127.3 | 7 | 319 | 4 | vp=184,990, regions=489 |
| stack | sample_flow_csv | 2,184,637 | 19,180.3 | 45,948.0 | 8 | 321 | 4 | vp=184,637, regions=489 |
| wise | sample | 2,000,000 | 21,926.3 | 49,780.1 | 8 | 787 | 21 |  |
| wise | sample_flow | 2,000,000 | 15,026.4 | 38,641.2 | 7 | 787 | 21 | unordered=0 |
| wise | sample_csv | 2,194,693 | 24,008.5 | 54,783.4 | 9 | 436 | 22 | vp=194,693, regions=489 |
| wise | sample_flow_csv | 2,194,876 | 16,401.8 | 42,370.8 | 8 | 430 | 22 | vp=194,876, regions=489 |

**T4b. Window samples** [results/aidb_window/hardness.json, hardness_details.json].

| dataset | scope | keys | RMSE | ME | CD | PLA-32 | PLA-4096 | extra |
|---|---|---|---|---|---|---|---|---|
| books | sample | 2,000,000 | 1,709.1 | 4,261.7 | 27 | 5,789 | 1 |  |
| books | sample_flow | 2,000,000 | 11,384.6 | 30,804.2 | 26 | 5,789 | 3 | dup=0, unordered=0 |
| books | sample_csv | 2,199,707 | 1,875.1 | 4,565.1 | 26 | 4,730 | 1 | vp=199,707, regions=489 |
| books | sample_flow_csv | 2,199,707 | 12,517.3 | 33,812.0 | 26 | 4,730 | 3 | vp=199,707, regions=489 |
| fb | sample | 2,000,000 | 8,202.5 | 24,280.7 | 96 | 10,599 | 18 |  |
| fb | sample_flow | 2,000,000 | 11,037.8 | 33,896.0 | 89 | 10,599 | 18 | dup=0, unordered=0 |
| fb | sample_csv | 2,199,707 | 9,015.3 | 26,484.2 | 85 | 11,034 | 20 | vp=199,707, regions=489 |
| fb | sample_flow_csv | 2,199,707 | 12,139.2 | 37,214.4 | 86 | 11,033 | 20 | vp=199,707, regions=489 |
| osm | sample | 2,000,000 | 118,136.5 | 219,857.8 | 152 | 4,919 | 44 |  |
| osm | sample_flow | 2,000,000 | 111,995.7 | 213,508.7 | 164 | 4,918 | 44 | dup=0, unordered=0 |
| osm | sample_csv | 2,199,707 | 129,938.9 | 241,721.1 | 141 | 4,856 | 49 | vp=199,707, regions=489 |
| osm | sample_flow_csv | 2,199,707 | 123,184.6 | 234,738.0 | 149 | 4,859 | 49 | vp=199,707, regions=489 |
| covid | sample | 2,000,000 | 24,143.5 | 49,834.7 | 24 | 827 | 9 |  |
| covid | sample_flow | 2,000,000 | 17,664.6 | 34,341.4 | 23 | 827 | 9 | dup=526, unordered=0 |
| covid | sample_csv | 2,197,391 | 26,482.5 | 54,690.5 | 22 | 490 | 9 | vp=197,391, regions=489 |
| covid | sample_flow_csv | 2,197,565 | 19,341.2 | 37,597.6 | 22 | 489 | 9 | vp=197,565, regions=489 |
| genome | sample | 2,000,000 | 48,827.0 | 98,852.6 | 171 | 12,719 | 14 |  |
| genome | sample_flow | 2,000,000 | 45,938.1 | 89,886.7 | 160 | 12,719 | 14 | dup=0, unordered=0 |
| genome | sample_csv | 2,199,707 | 53,700.7 | 108,703.0 | 142 | 12,337 | 14 | vp=199,707, regions=489 |
| genome | sample_flow_csv | 2,199,707 | 50,523.0 | 98,766.3 | 167 | 12,330 | 14 | vp=199,707, regions=489 |
| history | sample | 2,000,000 | 9,651.7 | 21,773.6 | 8 | 943 | 6 |  |
| history | sample_flow | 2,000,000 | 18,579.0 | 48,044.9 | 8 | 943 | 6 | dup=0, unordered=0 |
| history | sample_csv | 2,194,763 | 10,512.4 | 23,859.3 | 21 | 422 | 6 | vp=194,763, regions=489 |
| history | sample_flow_csv | 2,194,819 | 20,400.1 | 52,812.8 | 20 | 418 | 6 | vp=194,819, regions=489 |
| libio | sample | 2,000,000 | 8,854.0 | 21,656.5 | 2 | 1,057 | 9 |  |
| libio | sample_flow | 2,000,000 | 13,617.6 | 44,065.3 | 2 | 1,058 | 9 | dup=0, unordered=0 |
| libio | sample_csv | 2,170,213 | 9,257.0 | 23,767.3 | 31 | 941 | 9 | vp=170,213, regions=489 |
| libio | sample_flow_csv | 2,170,632 | 15,157.8 | 48,427.9 | 20 | 944 | 9 | vp=170,632, regions=489 |
| planet | sample | 2,000,000 | 42,645.1 | 81,355.0 | 49 | 8,498 | 35 |  |
| planet | sample_flow | 2,000,000 | 47,239.9 | 100,454.0 | 47 | 8,498 | 35 | dup=0, unordered=0 |
| planet | sample_csv | 2,199,707 | 46,896.6 | 89,150.9 | 124 | 8,628 | 38 | vp=199,707, regions=489 |
| planet | sample_flow_csv | 2,199,707 | 51,950.2 | 110,460.9 | 190 | 8,636 | 37 | vp=199,707, regions=489 |
| stack | sample | 2,000,000 | 25,321.9 | 48,379.9 | 1 | 144 | 3 |  |
| stack | sample_flow | 2,000,000 | 34,350.2 | 73,343.2 | 1 | 146 | 4 | dup=0, unordered=0 |
| stack | sample_csv | 2,062,282 | 23,747.7 | 45,364.7 | 6 | 86 | 3 | vp=62,282, regions=489 |
| stack | sample_flow_csv | 2,062,394 | 33,100.3 | 71,408.8 | 10 | 88 | 4 | vp=62,394, regions=489 |
| wise | sample | 2,000,000 | 5,551.0 | 11,282.6 | 8 | 805 | 4 |  |
| wise | sample_flow | 2,000,000 | 11,518.2 | 29,991.5 | 7 | 807 | 3 | dup=0, unordered=0 |
| wise | sample_csv | 2,195,396 | 6,107.5 | 12,504.7 | 16 | 315 | 4 | vp=195,396, regions=489 |
| wise | sample_flow_csv | 2,195,323 | 12,666.0 | 32,988.4 | 10 | 317 | 4 | vp=195,323, regions=489 |

One caveat about the **window run's `full_flow` scope** (not shown in the tables above, but present in
`results/aidb_window/hardness.json <name>.full_flow`): there the 200M-key file is pushed through a flow
trained on a 2M window that spans 0.14–2.9 % of the key range. Outside the window, the tanh is
saturated, so most keys collapse onto duplicate feature values (`duplicates` = 39,473,722 for books up to
168,160,900 for osm; 0 for planet) and the block is degenerate (CD = 200,000,000 for osm; RMSE ≈ 44–45 M
on all ten) [results/aidb_window/hardness_details.json `<name>.full_flow`]. The pipeline logs a warning
when duplicates exceed half the keys [aidb_pipeline.py:491-493]. Use the uniform run's `full_flow` for
"flow applied to the full file"; the window run's `full_flow` only documents that a locally trained flow
cannot be applied globally.

### 5.5 Per-region tail conflict degree (the index's own learnability signal)

At bulk load, every region of 4,096 keys fits its own line, normalises keys to x ∈ [0,1] over the region
(`LinearModel::normalized`, model.hpp:15) and computes NFL's tail conflict degree D₉₉ on x
(`tail_conflicts_raw`); with a flow it also computes D₉₉ on the sorted z(k) (`tail_conflicts_flow`)
[index.hpp:231-238, 245-248, 280]. The auto-switch keeps the flow for a region only if D₉₉(z) < D₉₉(x)
and the gain is at least `flow_min_gain` = 10 % of D₉₉(x) [index.hpp:281]. The sweep records the mean
over the 489 regions in `learnability.tail_conflicts_raw_mean` / `tail_conflicts_flow_mean`
[src/benchmark.cpp:72]. With 2,000,000 keys, regions = ⌈2,000,000 / 4,096⌉ = 489 (488 full regions and a
last one of 1,152 keys) [hardness.hpp:348].

**T5.** Mean per-region tail CD: raw from the `raw_rank` rows, flow from the `packed_rank_flow_forced`
rows (`--flow-bypass 0`, flow applied to all 489 regions); "kept" = `flow_regions` of the auto-switch
variant `packed_rank_flow` [results/aidb/sweep/results.jsonl and results/aidb_window/sweep/results.jsonl,
`learnability`; identical across the three query seeds since bulk load is deterministic].

| dataset | uniform raw | uniform flow (forced) | uniform regions where the auto-switch kept the flow | window raw | window flow (forced) | window regions kept |
|---|---|---|---|---|---|---|
| books | 3.2843 | 3.2843 | 26/489 | 10.7648 | 10.8037 | 4/489 |
| fb | 8.4479 | 8.4642 | 41/489 | 31.4110 | 31.4274 | 12/489 |
| osm | 36.1636 | 36.9816 | 7/489 | 9.6626 | 9.6462 | 21/489 |
| covid | 3.1575 | 3.1391 | 17/489 | 3.0184 | 3.0307 | 1/489 |
| genome | 4.2270 | 4.2290 | 8/489 | 60.4090 | 60.3824 | 9/489 |
| history | 3.1022 | 3.1043 | 13/489 | 3.1002 | 3.1084 | 2/489 |
| libio | 3.3722 | 3.3620 | 21/489 | 1.6074 | 1.6012 | 5/489 |
| planet | 7.6646 | 7.6483 | 43/489 | 35.7812 | 35.6708 | 17/489 |
| stack | 3.0082 | 3.0082 | 1/489 | 1.0000 | 1.0000 | 0/489 |
| wise | 3.0327 | 3.0348 | 4/489 | 3.0061 | 3.0041 | 1/489 |

Reading: (a) the noise floor is 3.00 (Section 5.3), so six uniform samples and four windows are at the
floor; (b) a single global 2×2 flow changes the per-region tail CD by less than 1 % on every sample
(largest change: osm uniform 36.16 → 36.98, worse) — at region scale a global monotone transform is
nearly affine, and D₉₉ is invariant under affine maps; (c) the auto-switch consequently keeps the flow
in 0–43 of 489 regions. This is the data-side reason the flow variants are neutral in the throughput
sweeps (see the results section of this guide).

---

## 6. The flows

### 6.1 What was trained, on what, how

One weight file per sample: `results/aidb/flows/<name>_2D2H2L.txt` (uniform) and
`results/aidb_window/flows/<name>_2D2H2L.txt` (window), produced by
`tools/train_flow.py <sample> --dtype uint64 --output <weights> --sample 4096 --steps 200 --monotone`
[aidb_pipeline.py:441; run.json `flow_sample: 4096, flow_steps: 200`]. The trainer:

- reads the 2M sample, de-duplicates and sorts it (`read_keys`), draws **4,096 training keys** uniformly
  with `random.Random(1000000007)` [train_flow.py `main`, `--seed` default 1000000007];
- model = NFL's deployed shape: x = (k − mean)/var with mean = smallest training key and var =
  (max − min)/shifts, shifts = 64 (NFL's author default is 1e6 for 200M keys [train_flow.py `--shifts`
  help]); features φ(x) = [x, x − ⌊x⌋]; hidden u = φ·W₀ (2×2), h = tanh(u); z = Σ_h h_h · (W₁[h,0] +
  W₁[h,1]) (2×2 output, sum decoder) [train_flow.py `Flow.forward`; transform.hpp reads the same
  format];
- `--monotone` zeroes the two weights on the fractional feature, so z is monotone in k (a documented
  deviation from NFL, which keeps them) [train_flow.py `--monotone` help; `loss_and_grad`];
- loss = 1-D change-of-variables negative log-likelihood under N(0,1) with a barrier keeping dz/dx > 0:
  L = mean over training keys of ½z² − log(dz/dx) + barrier/(dz/dx), barrier = 1e-3; Adam, lr 0.05,
  200 steps, best-loss weights kept [train_flow.py `loss_and_grad`, `train`]. This is "a reproducible,
  dependency-free stand-in, not a BNAF" [train_flow.py docstring; training_report.json `note`].

Weight file format (fb uniform) [results/aidb/flows/fb_2D2H2L.txt]:

```
2	2	2                                  # depth, hidden, output
15162980.0000000000000000	1207648189.7031250000000000     # mean, var  (var·64 = 77,289,484,141 = training span)
2	2
0.0058556031582764	0.0058592806847908                      # W0 row 0: x-feature weights (≈ 0.375/64)
0.0000000000000000	0.0000000000000000                      # W0 row 1: fractional-feature weights = 0 (--monotone)
2	2
1.2798795760698392	0.8812895293543010                      # W1 row 0
1.4785700705830542	1.0331745748154413                      # W1 row 1
```

**Worked transform** (fb uniform flow, computed with the formulas above [scratch/tables.py]):
v = (W₁[0,0]+W₁[0,1], W₁[1,0]+W₁[1,1]) = (2.161169, 2.511745).

| key k | x = (k − mean)/var | u = x·W₀[0,·] | h = tanh u | z = h·v |
|---|---|---|---|---|
| 97,995 (sample min) | −0.012475 | (−0.000073, −0.000073) | (−0.000073, −0.000073) | −0.000341 |
| 38,752,453,239 (full p50) | 32.076635 | (0.187828, 0.187946) | (0.185650, 0.185764) | 0.867812 |
| 77,308,811,965 (sample max) | 64.003449 | (0.374779, 0.375014) | (0.358165, 0.358370) | 1.674188 |

So on fb the learned map is z ≈ 0.026·x with a slight concave bend (h/u = 0.956 at the top): nearly
affine, which is why it changes CD and PLA by nothing and per-region tail CD by < 1 %. For the five
outlier keys of the full file x ≈ 1.5e10, u ≈ 8.8e7 and tanh saturates at 1, giving z = v₀ + v₁ =
4.672914 for all five.

**T6. Training report** [results/aidb/flows/training_report.json, results/aidb_window/flows/
training_report.json]. "Whole-sample tail CD" is the trainer's own D₉₉ on the whole 2M sample, raw
(min-max normalised keys) and transformed (sorted z).

| dataset | mode | best NLL | train s | whole-sample tail CD raw | tail CD transformed | unordered pairs |
|---|---|---|---|---|---|---|
| books | uniform | 3.6104 | 2.41 | 4 | 4 | 0 |
| fb | uniform | 4.1820 | 2.39 | 8 | 8 | 0 |
| osm | uniform | 3.9238 | 2.38 | 99 | 99 | 0 |
| covid | uniform | 4.1279 | 2.38 | 3 | 3 | 0 |
| genome | uniform | 4.3172 | 2.35 | 5 | 5 | 0 |
| history | uniform | 4.1648 | 2.36 | 3 | 3 | 0 |
| libio | uniform | 4.2237 | 2.38 | 3 | 3 | 0 |
| planet | uniform | 3.3771 | 2.36 | 18 | 9 | 0 |
| stack | uniform | 4.1902 | 2.39 | 3 | 3 | 0 |
| wise | uniform | 4.1711 | 2.39 | 3 | 3 | 0 |
| books | window | 4.1814 | 2.37 | 11 | 11 | 0 |
| fb | window | 4.1721 | 2.34 | 31 | 31 | 0 |
| osm | window | 4.1628 | 2.34 | 15 | 14 | 0 |
| covid | window | 4.1543 | 2.35 | 3 | 3 | 0 |
| genome | window | 4.1544 | 2.33 | 67 | 67 | 0 |
| history | window | 4.1913 | 2.35 | 3 | 3 | 0 |
| libio | window | 4.1874 | 2.35 | 2 | 2 | 0 |
| planet | window | 4.1799 | 2.34 | 52 | 52 | 0 |
| stack | window | 4.2061 | 2.34 | 1 | 1 | 0 |
| wise | window | 4.1783 | 2.37 | 3 | 3 | 0 |

Training takes 2.33–2.41 s per sample (pure Python, 4,096 keys × 200 steps). The only whole-sample tail
CD improvement is planet uniform, 18 → 9 (its front-loaded CDF is the one shape a monotone tanh can
straighten). Compare NFL's claim that after its flow "the conflict degrees can be kept around a low value
(e.g., around 4 for the tail conflict degree)" [NFL §3.3, p.6]: our 2×2 stand-in with 4,096
training keys does not achieve that on osm (99), fb window (31), genome window (67) or planet window
(52), and this guide never claims otherwise.

### 6.2 Effect of the transform on the hardness metrics

**T7.** Before/after for the full file (uniform-trained flow), the uniform sample and the window sample
[hardness.json `<name>.full` vs `.full_flow`, `.sample` vs `.sample_flow` in both runs]. Percentages are
(after − before)/before.

| dataset | scope | RMSE before | RMSE after | ΔRMSE | CD before | CD after | PLA-32 before | PLA-32 after | ΔPLA-32 | PLA-4096 before | after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| books | full (uniform-trained flow) | 18,053,637.5 | 15,985,128.0 | −11.5% | 246 | 245 | 262,604 | 262,604 | +0.0% | 97 | 94 |
| books | uniform sample | 180,591.3 | 159,882.7 | −11.5% | 11 | 9 | 602 | 602 | +0.0% | 8 | 8 |
| books | window sample | 1,709.1 | 11,384.6 | +566.1% | 27 | 26 | 5,789 | 5,789 | +0.0% | 1 | 3 |
| fb | full (uniform-trained flow) | 57,735,023.7 | 1,161,273.7 | −98.0% | 110 | 110 | 1,055,308 | 1,055,308 | +0.0% | 1,687 | 1,687 |
| fb | uniform sample | 1,448.2 | 11,584.6 | +699.9% | 24 | 24 | 3,034 | 3,034 | +0.0% | 1 | 3 |
| fb | window sample | 8,202.5 | 11,037.8 | +34.6% | 96 | 89 | 10,599 | 10,599 | +0.0% | 18 | 18 |
| osm | full (uniform-trained flow) | 24,177,498.5 | 23,977,602.0 | −0.8% | 4,107 | 3,943 | 661,115 | 661,115 | +0.0% | 5,495 | 5,495 |
| osm | uniform sample | 241,476.1 | 239,472.1 | −0.8% | 1,102 | 1,100 | 7,099 | 7,099 | +0.0% | 68 | 68 |
| osm | window sample | 118,136.5 | 111,995.7 | −5.2% | 152 | 164 | 4,919 | 4,918 | −0.0% | 44 | 44 |
| covid | full (uniform-trained flow) | 1,795,722.5 | 1,272,460.5 | −29.1% | 27 | 27 | 81,908 | 81,909 | +0.0% | 850 | 850 |
| covid | uniform sample | 18,073.1 | 12,906.4 | −28.6% | 8 | 8 | 1,192 | 1,193 | +0.1% | 9 | 9 |
| covid | window sample | 24,143.5 | 17,664.6 | −26.8% | 24 | 23 | 827 | 827 | +0.0% | 9 | 9 |
| genome | full (uniform-trained flow) | 7,531,940.1 | 8,390,195.9 | +11.4% | 585 | 587 | 1,290,208 | 1,290,208 | +0.0% | 1,426 | 1,425 |
| genome | uniform sample | 75,352.0 | 83,934.3 | +11.4% | 77 | 82 | 2,045 | 2,045 | +0.0% | 18 | 18 |
| genome | window sample | 48,827.0 | 45,938.1 | −5.9% | 171 | 160 | 12,719 | 12,719 | +0.0% | 14 | 14 |
| history | full (uniform-trained flow) | 815,542.7 | 432,676.2 | −46.9% | 8 | 8 | 105,468 | 105,469 | +0.0% | 468 | 467 |
| history | uniform sample | 8,104.9 | 4,290.7 | −47.1% | 7 | 7 | 1,067 | 1,068 | +0.1% | 3 | 3 |
| history | window sample | 9,651.7 | 18,579.0 | +92.5% | 8 | 8 | 943 | 943 | +0.0% | 6 | 6 |
| libio | full (uniform-trained flow) | 3,445,772.9 | 4,072,387.5 | +18.2% | 2 | 2 | 145,808 | 145,811 | +0.0% | 639 | 638 |
| libio | uniform sample | 34,472.4 | 40,748.0 | +18.2% | 8 | 7 | 1,256 | 1,255 | −0.1% | 10 | 10 |
| libio | window sample | 8,854.0 | 13,617.6 | +53.8% | 2 | 2 | 1,057 | 1,058 | +0.1% | 9 | 9 |
| planet | full (uniform-trained flow) | 29,771,622.2 | 15,545,175.8 | −47.8% | 21 | 32 | 613,597 | 613,602 | +0.0% | 2,314 | 2,313 |
| planet | uniform sample | 297,556.5 | 155,367.6 | −47.8% | 26 | 33 | 3,456 | 3,455 | −0.0% | 13 | 15 |
| planet | window sample | 42,645.1 | 47,239.9 | +10.8% | 49 | 47 | 8,498 | 8,498 | +0.0% | 35 | 35 |
| stack | full (uniform-trained flow) | 839,727.0 | 1,769,181.3 | +110.7% | 1 | 1 | 17,833 | 17,833 | +0.0% | 133 | 129 |
| stack | uniform sample | 8,395.8 | 17,705.8 | +110.9% | 8 | 7 | 593 | 594 | +0.2% | 4 | 4 |
| stack | window sample | 25,321.9 | 34,350.2 | +35.7% | 1 | 1 | 144 | 146 | +1.4% | 3 | 4 |
| wise | full (uniform-trained flow) | 2,200,401.6 | 1,511,720.5 | −31.3% | 10 | 9 | 79,035 | 79,036 | +0.0% | 382 | 382 |
| wise | uniform sample | 21,926.3 | 15,026.4 | −31.5% | 8 | 7 | 787 | 787 | +0.0% | 21 | 21 |
| wise | window sample | 5,551.0 | 11,518.2 | +107.5% | 8 | 7 | 805 | 807 | +0.2% | 4 | 3 |


How to read T7:

- **PLA-32 and PLA-4096 are invariant** under the monotone flow (changes of 0 to 5 segments out of up to
  1.29 M). That is expected: a strictly monotone reparametrisation of the x-axis preserves the ordering
  and only bends line segments slightly; PGM's optimal segmentation sees the same rank sequence.
- **CD moves by at most a few percent** (osm full 4,107 → 3,943; planet 21 → 32): the flow output is
  near-affine at cluster scale, and the CD deviation ε on real features is documented.
- **RMSE changes a lot and in both directions**, because RMSE is about global curvature and a 2×2 tanh
  is a global curvature knob: it helps front-loaded or convex CDFs (planet −47.8 %, history −46.9 %,
  wise −31.3 %, covid −29.1 %, books −11.5 %), does nothing on stepped CDFs (osm −0.8 %), and hurts
  already-straight sequences (stack +110.7 %, libio +18.2 %, genome +11.4 %); on the windows, which are
  nearly linear locally, it hurts seven of ten (books +566 %, wise +108 %, history +93 %). fb full −98 %
  is the outlier-squashing effect explained in Section 3.2, not a real gain for the bulk (fb uniform
  sample: +700 %).
- The uniform-sample rows track the full-file rows to three significant figures (books −11.5/−11.5,
  covid −29.1/−28.6, history −46.9/−47.1, planet −47.8/−47.8, wise −31.3/−31.5): the flow was trained
  on 4,096 keys of the uniform sample, and the uniform sample has the full file's global shape
  (Section 5.2). This is the data-level reason our full-file `full_flow` scope can be trusted for RMSE
  but tells you nothing new about PLA or CD.

---

## 7. The CSV-augmented scopes (`sample_csv`, `sample_flow_csv`)

### 7.1 How they are produced

`scaleli_hardness --virtual-alpha 0.1 --region-keys 4096` computes a third metric block, `smoothed`, on
"the CSV-style augmented sequence (alpha × region virtual points per region; features are the
transformed keys with --flow, else double(key − min))" [src/hardness.cpp:87-88, 166-181]; `--alpha 0.1`
is the pipeline default, "CSV-style virtual-point budget per region" [aidb_pipeline.py:606], the same
α = 0.1 default the CSV paper uses ("We vary the smoothing threshold, α, from 0.05 to 0.8, with a default
value of 0.1" [CSV §6.1 "Parameters"]). The full 200M files are not smoothed (cost O(α·n·region_keys)
loss evaluations, "meant for samples (a few million keys), not 200M keys" [hardness.hpp:341-345]).

The smoother is our clean-room reading of CSV Algorithm 1 (no public implementation exists as of
September 2026) [smoothing.hpp header]:

```
smooth_regions(features f[0..n), region_keys R = 4096, α = 0.1):          # hardness.hpp:346-378
  regions = ⌈n / R⌉ = 489 for n = 2,000,000 (488 regions of 4,096 keys + one of 1,152)
  for each region independently (threads):  smooth_cdf(f[b .. b+len), α)   # smoothing.hpp
      budget λ = ⌊α · len⌋  (= 409 for a full region, 115 for the last one)
      seq = the region's features, targets = positions 0..len−1
      repeat up to λ times:
          for every gap (seq[i], seq[i+1]) with seq[i+1] > seq[i]:
              probe the OLS SSE of (seq ∪ {virtual point at x_v inserted after i}) at
              x_v = lo+e, lo+2e, hi−2e, hi−e  (e = 1e-3 · gap);  if the loss decreases from the left AND
              from the right, ternary-search the interior minimum (≤ 40 steps), else take the better end
          if no candidate lowers the SSE (beyond 1e-12 relative): stop      # CSV Alg. 1 line 27
          insert the best virtual point; every later target shifts by +1
      return slot(k) = rank + #virtual points before k, and the virtual features
  concatenate the per-region augmented sequences: position j = slot j of the concatenated slot spaces
```

Loss = OLS residual SSE over real AND virtual points (CSV's equation 4), evaluated in O(1) per candidate
from running and suffix sums, so one round is O(gaps) and a region costs O(λ·R) [smoothing.hpp header,
`loss_at`]. Virtual points are slot offsets only: "never stored as records and never affect logical rank"
[smoothing.hpp header]. The metrics of the `smoothed` block are then computed on the augmented sequence
of n + (virtual points) features with target = augmented position, i.e. **the slot sequence**
[src/hardness.cpp:172-176].

**Budget arithmetic** (verified against the files): 488 × ⌊0.1 × 4,096⌋ + ⌊0.1 × 1,152⌋ = 488 × 409 + 115
= **199,707** = the `virtual_points` count on every dataset that used its whole budget (fb, genome,
planet-window, books-window, osm-window, …) [hardness_details.json `<name>.sample_csv.virtual_points`].
Smaller counts mean the early stop fired in some regions: the loss could not be lowered any more. That
happens where the CDF is already straight: stack window used 62,282 (31.2 %), libio window 170,213
(85.2 %), stack uniform 184,990, books uniform 185,327; and on the linear control sample 179,985
[linear_uniform_hardness.json `smoothed.virtual_points`].

### 7.2 Effect per dataset

**T8.** `sample` vs `sample_csv` [hardness.json both runs; vp from hardness_details.json].

| dataset | mode | virtual points | of budget 199,707 | RMSE sample | RMSE sample_csv | ΔRMSE | CD | CD csv | PLA-32 | PLA-32 csv | ΔPLA-32 | PLA-4096 | csv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| books | uniform | 185,327 | 92.8% | 180,591.3 | 196,846.2 | +9.0% | 11 | 10 | 602 | 236 | −60.8% | 8 | 8 |
| books | window | 199,707 | 100.0% | 1,709.1 | 1,875.1 | +9.7% | 27 | 26 | 5,789 | 4,730 | −18.3% | 1 | 1 |
| fb | uniform | 199,707 | 100.0% | 1,448.2 | 1,591.2 | +9.9% | 24 | 23 | 3,034 | 1,718 | −43.4% | 1 | 1 |
| fb | window | 199,707 | 100.0% | 8,202.5 | 9,015.3 | +9.9% | 96 | 85 | 10,599 | 11,034 | +4.1% | 18 | 20 |
| osm | uniform | 199,605 | 99.9% | 241,476.1 | 265,521.5 | +10.0% | 1,102 | 1,099 | 7,099 | 7,491 | +5.5% | 68 | 76 |
| osm | window | 199,707 | 100.0% | 118,136.5 | 129,938.9 | +10.0% | 152 | 141 | 4,919 | 4,856 | −1.3% | 44 | 49 |
| covid | uniform | 198,862 | 99.6% | 18,073.1 | 19,819.2 | +9.7% | 8 | 10 | 1,192 | 892 | −25.2% | 9 | 11 |
| covid | window | 197,391 | 98.8% | 24,143.5 | 26,482.5 | +9.7% | 24 | 22 | 827 | 490 | −40.7% | 9 | 9 |
| genome | uniform | 199,707 | 100.0% | 75,352.0 | 82,874.1 | +10.0% | 77 | 71 | 2,045 | 1,627 | −20.4% | 18 | 19 |
| genome | window | 199,707 | 100.0% | 48,827.0 | 53,700.7 | +10.0% | 171 | 142 | 12,719 | 12,337 | −3.0% | 14 | 14 |
| history | uniform | 195,413 | 97.8% | 8,104.9 | 8,628.3 | +6.5% | 7 | 21 | 1,067 | 411 | −61.5% | 3 | 3 |
| history | window | 194,763 | 97.5% | 9,651.7 | 10,512.4 | +8.9% | 8 | 21 | 943 | 422 | −55.2% | 6 | 6 |
| libio | uniform | 195,924 | 98.1% | 34,472.4 | 37,442.8 | +8.6% | 8 | 21 | 1,256 | 614 | −51.1% | 10 | 11 |
| libio | window | 170,213 | 85.2% | 8,854.0 | 9,257.0 | +4.6% | 2 | 31 | 1,057 | 941 | −11.0% | 9 | 9 |
| planet | uniform | 199,656 | 100.0% | 297,556.5 | 327,255.8 | +10.0% | 26 | 37 | 3,456 | 2,767 | −19.9% | 13 | 15 |
| planet | window | 199,707 | 100.0% | 42,645.1 | 46,896.6 | +10.0% | 49 | 124 | 8,498 | 8,628 | +1.5% | 35 | 38 |
| stack | uniform | 184,990 | 92.6% | 8,395.8 | 9,036.9 | +7.6% | 8 | 7 | 593 | 319 | −46.2% | 4 | 4 |
| stack | window | 62,282 | 31.2% | 25,321.9 | 23,747.7 | −6.2% | 1 | 6 | 144 | 86 | −40.3% | 3 | 3 |
| wise | uniform | 194,693 | 97.5% | 21,926.3 | 24,008.5 | +9.5% | 8 | 9 | 787 | 436 | −44.6% | 21 | 22 |
| wise | window | 195,396 | 97.8% | 5,551.0 | 6,107.5 | +10.0% | 8 | 16 | 805 | 315 | −60.9% | 4 | 4 |

How to read T8 (and why the global metrics look worse while the local one looks better):

- **RMSE/ME rise by ≈ +10 %**, i.e. by exactly the factor (n + λ)/n = 1.1 when the budget is used up.
  The global line is fitted to slots, not ranks; inserting 10 % more targets stretches the y-axis by 1.1
  and every residual with it. This is not a degradation, it is the unit change; RMSE/(n+λ) is unchanged.
  Where fewer points were inserted the rise is smaller (history uniform +6.5 %, stack window −6.2 % with
  31 % of the budget: there the smoother actually straightened the window a little).
- **PLA-32 falls sharply on the smooth datasets**: −60.8 % books, −61.5 % history, −60.9 % wise window,
  −51.1 % libio, −46.2 % stack, −43.4 % fb uniform, −40.7 % covid window. The per-region smoother
  straightens the slot sequence inside each 4,096-key region, so a 32-slot corridor covers more keys.
  On the linear control sample it does the same (535 → 225 [linear_uniform_hardness.json]): the gain on
  those datasets is mostly the removal of sampling jitter, not of dataset structure.
- **PLA-32 does not fall where the roughness is stronger than 10 % of the keys can absorb**: fb window
  +4.1 %, osm uniform +5.5 %, planet window +1.5 %, genome window −3.0 %, osm window −1.3 %. This is the
  data-side limit of a fixed α: CSV's own claim is a per-key improvement for "the keys in deeper levels"
  [CSV Abstract], not a global PLA reduction.
- **CD goes up on the easy datasets** (history 7 → 21, libio 8 → 21 uniform / 2 → 31 window, planet window
  49 → 124, wise window 8 → 16): the smoother is greedy on the OLS loss, and a virtual point placed 1e-3
  of a gap away from a real key (the `lo+e` / `hi−e` candidates) creates a feature pair closer than
  FMCD's slot width, so LIPP-style slot mapping sees a new "cluster". The index does not use CD to place
  keys (virtual points are slot offsets), but the metric records it honestly. On the hard datasets CD
  drops slightly (osm 1,102 → 1,099, genome 171 → 142, fb window 96 → 85).
- **PLA-4096 is unchanged or +1–8**: the smoother works inside 4,096-key regions, and the region
  boundaries are exactly where an ε = 4,096 segmentation can already absorb everything.

`sample_flow_csv` (flow first, then virtual points on z) behaves like `sample_csv` on every dataset
(T4a/T4b): the two mechanisms act on different scales and do not interact at the metric level.

---

## 8. Questions the supervisor may ask, with answers

1. **Why 2M-key samples and not the full 200M keys?** The full files are used for the hardness metrics
   (`full`, `full_flow`: 200,000,000 keys each, exact `__int128` arithmetic, ~17 s per file
   [hardness_details.json `full.elapsed_ns` = 17.4 s for fb]). The index sweeps run on 2M-key samples
   because (a) the CSV-style smoother costs O(α·n·R) loss evaluations and the pipeline documents it as
   meant for "a few million keys, not 200M" [hardness.hpp:344-345]; (b) the sweep is 10 datasets × 10
   variants × 3 seeds = 300 runs per lane, and each run bulk-loads and instruments every probe; (c) the
   AIDB paper's own protocol ("bulk load ... then perform random lookups for all keys", 20M warm-up +
   100M measured lookups, two Xeon Gold 5118 sockets [AIDB §4.1]) is a server protocol we did not
   attempt to reproduce on a laptop. `results/aidb_fullscale/` holds the separate full-scale
   configurations (see the results section). The sample size is `--sample 2000000` [aidb_pipeline.py:601;
   run.json].

2. **Is a uniform 1 % sample representative?** For global shape, yes, exactly: RMSE and ME scale by
   p = 0.01 to within 1 % on nine datasets (T9), and the flow's effect on RMSE is the same on the sample
   and the full file to three digits (T7). For local structure, no: an ε = 32 corridor in the sample is
   ±3,200 full ranks plus binomial jitter of up to 1,407 ranks, so PLA-32 collapses to a noise floor of
   ~535 segments (Section 5.3) and per-region tail CD to 3.0 on six datasets. That is exactly why the
   window lane exists.

3. **Is a contiguous 2M window representative?** For local structure it is a faithful 1/100 of the file
   on fb, covid, genome, wise (window PLA-32 × 100 within 2 % of full PLA-32) and within 40 % on osm,
   libio, stack, history; on the non-stationary books and planet the 85.8 % quantile slice is 1.4–2.2×
   rougher than the file average (T9). For global shape it is useless by construction (1 % of the key
   range at one place). We report both lanes separately and never average them.

4. **Why is the window at rank 171,644,825 for every dataset?** Because `random.Random(42).randrange
   (198,000,001)` is evaluated with the same seed for every dataset [tools/datasets.py:81]; the seed is
   `--sample-seed 42` [aidb_pipeline.py:602]. The offset is not in the manifest but is reproducible, and
   we verified that each window's first/last keys equal the source keys at ranks 171,644,825 and
   173,644,824 [scratch/data_stats.json]. A different seed would give a different slice; we did not run a
   multi-offset study (see the "unverified" list at the end of this guide).

5. **Which datasets are hard, and why?** It depends on the scale you ask about, which is the AIDB
   paper's thesis [AIDB §4.2]. Globally (RMSE/ME/PLA-4096): fb (five outliers up to 2⁶⁴ − 1 make the
   least-squares line useless: RMSE = n/√12 exactly), planet (front-loaded CDF, RMSE/n 0.149), osm
   (stepped, clustered S2 cells: CD 4,107, PLA-4096 5,495), books (concave, skewed). Locally (window
   PLA-32, per-region tail CD): genome (12,719; 60.4), fb (10,599; 31.4), planet (8,498; 35.8), books
   (5,789; 10.8). Easy at every scale: stack (CD 1, 84 % dense integers), then wise, covid, history,
   libio (per-region tail CD ≈ 3 or below, PLA-32 windows 144–1,057).

6. **Your history and libio numbers: do they match the paper?** Yes, digit for digit at the paper's
   precision: history PLA-32 105,468 / PLA-4096 468 vs "h_l = 105k, h_g = 468", libio 145,808 / 639 vs
   "h_l = 146k, h_g = 639" [AIDB §2.2; hardness.json]. The paper prints no other dataset's coordinates
   (only Figures 2 and 6), so the other eight cannot be cross-checked numerically.

7. **Why is fb's RMSE exactly 57.7 M?** Because 199,999,979 of its keys are below 1.1·10¹² and five are
   above 10¹⁹, so the least-squares line is flat over the bulk and the residual against ranks 0..n−1 is
   that of a constant predictor: RMSE = n/√12 = 57,735,026.9 (measured 57,735,023.7), ME = n/2 (measured
   99,999,995.5). The uniform sample happened to draw none of the 21 keys ≥ 2⁴⁰ (probability 0.99²¹ =
   0.81), which is why the sample's RMSE is 1,448 and the ratio is 39,866 instead of 100. Nothing in our
   code special-cases fb.

8. **Did you sort or de-duplicate the data, and is that legitimate?** Seven files are served unsorted;
   GRE's own loader sorts at load time [aidb_pipeline.py:22-23]. Our `sort` step sorts them once,
   audits duplicates (0 in all ten) and stores the sorted copy with its own sha256 [provenance.json
   `sort_audit`]. No key was added or removed (`written_keys: 200000000`), so the sorted files are the
   same key sets the paper describes ("200 million 64-bit unsigned integer keys without duplicates"
   [AIDB §4.1]).

9. **Does the flow change the hardness metrics?** PLA-32/PLA-4096: no (monotone reparametrisation;
   changes ≤ 5 segments). CD: by a few percent. RMSE: yes, ±50 %, in both directions, because a 2×2
   tanh is a global curvature knob (helps planet/history/wise/covid, hurts stack/libio/genome and almost
   every window). Per-region tail CD: < 1 % (T5), so the NFL-style auto-switch keeps the flow in only
   0–43 of 489 regions. None of this contradicts NFL, which trains a much larger BNAF on the full key
   set; it says that our small clean-room control cannot reproduce NFL's "around 4" tail conflict
   degree on osm, fb-window, genome-window or planet-window.

10. **What does α = 0.1 mean concretely and why do some samples have fewer than 199,707 virtual
    points?** λ = ⌊0.1 × region size⌋ virtual slots per 4,096-key region: 409 per full region, 115 in
    the 1,152-key tail region, 199,707 in total. Fewer means the greedy loop stopped because no candidate
    lowered the region's OLS SSE (CSV Algorithm 1's stop condition) — e.g. stack window used 62,282
    because the sequence is almost 1, 2, 3, …. The budget is per region, never global, and virtual points
    are never materialised as records.

11. **Why do the CSV scopes show higher RMSE and CD if smoothing is supposed to help?** RMSE rises by
    the slot-count factor 1.1 (units, not fit quality); CD rises on smooth datasets because the greedy
    OLS placement puts virtual points 1e-3 of a gap from real keys, which FMCD reads as clusters. The
    metric that measures what CSV targets, PLA-32, drops by 20–60 % on the smooth datasets and by 0–5 %
    on the rough ones (fb window, osm, planet window). The index's own signal (rank SSE before/after in
    the sweep's learnability block) drops by 16–98 % (osm uniform 2.72e11 → 2.29e11 = −16 %; wise window
    1.27e9 → 6.17e7 = −95 %; stack window −98 %) [results/*/sweep/results.jsonl `learnability.rank_sse_before/after`].

12. **How were "duplicates after the transform" handled?** The trainer zeroes the fractional-feature
    weights (`--monotone`), so z is monotone and `unordered_pairs = 0` on all 20 samples. On the window
    lane, covid's flow still produced 526 duplicate z values among 2M keys (tanh resolution on a 1e17
    range compressed to 1 % of the range) [results/aidb_window/hardness_details.json
    `covid.sample_flow.duplicates`]; `scaleli_hardness` reports duplicates and PGM-style PLA handles
    them with the `next(x)` rule [hardness.hpp:303-308]. Applying a window-trained flow to the full file
    collapses up to 84 % of the keys onto duplicates and is reported as degenerate (Section 5.4 caveat).

13. **What would you run next on the data side?** (a) Several window offsets per dataset (seeds) to put
    error bars on the window lane; (b) the `strided` sampler, which exists in the code but was not used,
    as a third lane that keeps global shape without binomial jitter; (c) larger flows (more hidden
    units, all keys) to test whether NFL's "around 4" tail conflict degree is reachable on osm and
    genome; (d) a full-file CSV smoothing run with a bounded-window implementation, since the current
    O(λ·R) smoother is the reason the `*_csv` scopes exist only for samples.

---

## Unverified items in this section

- [unverified: not stated in the papers] what integer each dataset's key encodes beyond the one-line
  Table 1 description (books "sales popularity" value, fb upsampling procedure, covid tweet-ID layout,
  genome locus-pair encoding, wise "partition key", history/planet "node ID" vs "planet ID"); the CSV
  paper's "Google S2 cell IDs" for osm is the only extra detail available.
- [unverified: observed only] the five fb keys ≥ 2⁶³ spaced by ≈ 2⁶⁴/9 are in the file as served (sha256
  verified); whether they are intentional sentinels of the SOSD generator or artefacts is not stated in
  the papers.
- [unverified: single seed] all "window" statements are for one offset (rank 171,644,825, seed 42); no
  multi-offset variance was measured.
- [unverified: field absent] transformed-duplicate counts for the uniform lane's `sample_flow` and
  `full_flow` blocks are not recorded in `results/aidb/hardness_details.json` (the window lane records
  them); `unordered_pairs = 0` is recorded for both lanes.
