# Historical approach map and implementation status

This repository is a meeting-ready literature and experiment walkthrough. It is NOT
an exhaustive, commit-pinned reproduction of all SOTA systems. There is no
cross-paper speed leaderboard. Years below refer to publication where established;
the uploaded CSV text has inconsistent date labels, recorded rather than reconciled.

| Period | Approach | Mechanism and question | What runs here | What is missing |
|---|---|---|---|---|
| Conventional | Sorted array / binary search; ordered map | Exact search without fitted predictors | Native controls and reference checks | Optimized B+tree/ART comparison |
| 2018 | RMI / The Case for Learned Index Structures | Learn key-to-position; hierarchical model dispatch + correction | L01 is an original two-stage teaching model | Authors' RMI generator/training and benchmark |
| 2019 | FITing-Tree | Error-bounded piecewise-linear segments and cost-based error choice | Literature discussion; no FITing executable | Authors' implementation and dynamic reproduction |
| 2020 | PGM-index | Optimal PLA segments, recursively indexed; dynamic and compressed-model variants | Optional adapter source inherited under experimental/scaleli | Upstream checkout, compile, full semantic tests and results |
| 2020 | RadixSpline | One-pass spline corridor + radix table + bounded final search | L02: compiled source-derived transcription of three MIT headers | Byte-identical commit-pinned checkout and full SOSD reproduction |
| 2020 | ALEX | Gapped data nodes, model-guided search, cost-driven structural adaptation | Optional adapter source inherited; no ALEX timing | Upstream compile, semantic tests, common-workload measurements |
| 2021 | LIPP | Predicted placement; collisions handled through additional nodes | Literature only | Upstream integration |
| 2021 | LA-vector | Learned lossless representation with exact rank/select | Literature only | Official compressed-integer comparison |
| 2022 | NFL / AFLI | Numerical normalizing flow changes input distribution; AFLI handles transformed keys | L03 recomputes ratios from published tables. L08 runs a clean-room, MKL-free inference of the official weight format (2D2H2L tanh network, github.com/luffy06/NFL, GPL-3, code not copied), the tail-conflict-degree auto-switch per region, and a stdlib stand-in trainer; the transform is a model feature only | Official BNAF training, AFLI buckets/dense nodes, batched transformation semantics, author weights on real SOSD data |
| 2022 | GRE / Are Updatable Learned Indexes Ready? | Common dynamic evaluation and distribution-shift tests | Workload factors discussed; not a GRE reproduction | Run official suite under agreed scope |
| 2023 | SALI | Probability-model-based adaptation, based on LIPP | Mentioned as a CSV host and dynamic continuation | Upstream integration |
| 2024 | LeCo | Learn serial correlations; store exact residuals with random access | L04 uses our scalar codecs as a control, NOT LeCo | Author partitioning/codecs/RocksDB reproduction |
| 2024/25 attached version | CSV | Insert virtual points; refit; select subtree reconstructions | L05 independent 10-key greedy/exhaustive teaching experiment. L08 runs Algorithm 1 written from the paper (no public code exists) in closed form per region; virtual points move rank targets to slot ranks and cost no arena bytes | Hierarchical Algorithm 2 (cost-model merges), gap materialization for inserts, the authors' ALEX/LIPP/SALI hosts |
| 2025 | SEA integer-index comparison | Compressed, learned and optimized traditional methods | Literature only | Implement its stronger static controls |
| 2025 article / SIGMOD 2026 venue context | RoBin | Change initial sampling, load size and insertion order | Factors appear in inherited robustness config | Original RoBin suite is not run |
| 2026 | LINE | Group-enhanced leaves, cache-optimized tree, incremental group migration | Literature only | Upstream replication |
| 2026 | AIDB hardness metrics (Zhang, Tang, Ailamaki, AIDB@VLDB) | Conformance/coverage scores for dataset-hardness metrics: RMSE, ME, CD (LIPP FMCD fit), PLA-32/PLA-4096 and their 2-/3-way compositions | L09 and experimental/scaleli/tools/aidb_pipeline.py: the scalar metrics on the ten GRE datasets (full 200M keys), the scores computed against OUR control variants at reduced scale (2M-key samples, 1M lookups), and an HTML view of hardness space before/after the NFL-style transform and after CSV-style smoothing | The paper's seven indexes (RMI, PGM, ALEX, LIPP, XIndex, FINEdex, ...), its 100M-lookup runs and testbed; any claim about those indexes |
| Current project | Selective physical compression | Does the physical encoded-key mapping help exact search after overhead? | L04/L06 custom exact compressed map, clearly experimental | No new-method or superiority claim |

## Learnability lane (lesson 08 and configs/learnability.json)

Both controls live inside the experimental map's `Region::rebuild`. The NFL-style flow
replaces the model feature (`z = flow(key)` instead of the normalized key) and the
NFL auto-switch keeps it only where the region's tail conflict degree falls by at
least 10%. The CSV-style smoothing replaces rank targets with slot ranks after greedy
virtual-point insertion with budget alpha times region keys. Records, fences, codecs
and scans are untouched, so exact semantics hold even for a non-monotone transform.
Per-lookup transform cost is inside the measured latency. The sweep pairs every
variant on identical traces; `tools/sweep_report.py` renders it without matplotlib.

## AIDB hardness lane (lesson 09 and experimental/scaleli/tools/aidb_pipeline.py)

The pipeline verifies the ten GRE downloads (never downloading itself), audits their
sortedness (seven of the ten, `covid genome history libio planet stack wise`, are served
unsorted, so a sorted, de-duplicated `<name>.sorted` copy is written and checksummed and
every later step uses it), draws a 2M-key uniform sample of each, trains the stdlib
stand-in flow on the sample, runs `scaleli_hardness` on the full file (raw and
flow-transformed keys) and on the sample (raw, flow, and CSV-style virtual points with
alpha 0.1), pairs the eight contracted control variants of the experimental map on the
samples (`--extra-variants fusion` adds two opt-in cost-based fusion controls), and feeds
median throughput plus the hardness scopes to `tools/aidb_scores.py` for the paper's
conformance and coverage scores. CD on transformed/smoothed features uses a gap-scaled
U_T epsilon rather than LIPP's absolute 1e-6 (the literal value is emitted alongside), and
every CD carries +-1 rounding uncertainty; see experimental/scaleli/README.md.
Everything is a reduced-scale, single-host, clean-room control: the hardness numbers are
on the real 200M-key files, but the throughput column is our map on 2M-key samples, so
the scores say how well each metric orders hardness *for our controls*, not for the
indexes in the paper. Lesson 09 only reads `results/aidb/throughput.json` and
`scores.json` back; when they are absent it reports NOT RUN instead of failing.

## The three runnable lanes are intentionally different

1. L01 and L05: Python teaching demonstrations. Geometry/objective calculations,
   not native throughput measurements of RMI/CSV.
2. L02: native static lower_bound, keys only. Source-derived RadixSpline and
   binary search use the same fixture/query trace. No payloads or updates.
3. L04/L06: custom mutable compressed key/value map. Eight-byte values,
   metadata and overlays are included in its accounted storage.

Never place L02 and L04 memory numbers on an unlabeled common axis: one is a
keys-only search index, the other is a full key/value map.

## External source retrieval

The original opt-in fetcher remains under experimental/scaleli/tools/fetch_external.py.
It requires network access, downloads official ALEX/PGM repositories, records actual
commit IDs, and does not automatically establish compile/semantic parity. Do not run
new downloads or integration work during the meeting. Read its STATUS.md first.

## Data

All new local demos use the shipped 16,384-key dense/sparse synthetic fixture.
Its SHA-256 and exact binary representation are included in evidence/. It is not a
SOSD real dataset, not a sample of Facebook/OSM, and not bit-identical to the old
C++ generated fixture used in archived measurements.

The SOSD/GRE acquisition recipes in experimental/scaleli/configs/datasets.json (fifteen
entries, including the ten AIDB 2026 / GRE datasets as `gre_<name>`) are starting points,
not evidence of completed real-data experiments; the AIDB lane above records provenance
(URL, SHA-256, byte length, retrieval time, sample manifest) when it runs. For the next
milestone, agree on a matched static contract first, acquire an approved real dataset,
record license/source/checksum, then select a documented window or full corpus.
