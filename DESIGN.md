# Transform builds on the GRE baselines (branch gre-transform)

This branch sits on top of `gre-baselines` and adds one layer: GRE's own indexes built on *transformed* keys
(NFL-style: Wu et al., PVLDB 2022), plus a learned per-dataset gate that picks the transform. Nothing from SPLICE or
SCALE-LI is used. The baselines path must stay unchanged: without `EXTRA=transforms`, `scripts/build_gre.sh` builds
exactly the baseline binary, with the same `competitor.h`.

## Contract

- **Read-only point lookups.** This is GRE's `--read=1 --insert=0` workload.
  - `get()` of a key that was bulk-loaded must return its own payload, always.
  - `put`, `update` and `remove` return false; `scan` returns 0. A transform is not order-preserving in general
    (NF rounding), and NFL itself serves point operations only.
  - A lookup of a key that was never loaded may return a false positive. This is documented, not hidden.
- **Exactness does not rely on GRE's check.** GRE gives every key the payload 123456789, so `success_read` cannot
  catch a lookup that returns another key's payload. Our tests must load distinct payloads and check each one.
- **The transform's output range is `[0, 2^52)`** (`gret::kOutBits = 52`) on every machine. Outputs are therefore exact
  doubles, the double-based models in LIPP and ALEX behave the same on arm64 (64-bit `long double`) and x86, and LIPP
  does not hit its macOS failure above 2^53.

## Files

C++17 headers in `include/gret/`, namespace `gret`, MIT, no dependencies beyond the standard library and GRE's
competitor headers.

### `include/gret/pwlcdf.h`: the direct transform `PwlCdf`

- **Build.** `static PwlCdf build(const uint64_t* sorted, size_t n, unsigned K)` takes the sorted, distinct build keys.
  - It places K+1 knots at equi-rank positions `x_j = sorted[round(j (n-1)/K)]`, dropping duplicate x.
  - Each knot gets `y_j = round(j/K * (2^52 - 1))`, so the outputs fill `[0, 2^52)`.
- **Evaluate.** `uint64_t operator()(uint64_t k) const`.
  - The formula is `y_j + ((k - x_j) * mul_j) >> 64`, with `mul_j = floor((y_{j+1} - y_j) << 64 / (x_{j+1} - x_j))` in
    `unsigned __int128`.
  - The product stays below `(y_{j+1} - y_j) 2^64` inside a segment, so it cannot overflow.
  - Keys below `x_0` give `y_0`, and keys at or above `x_K` give `y_K`.
  - The result is non-decreasing in k and exact on every machine.
- **Knot search.** Branch-free, with a fixed number of steps; record it in `search_steps()`. A faster layout
  (interleaved records, or a small radix table on the top bits) is welcome, but it must stay exact.
- **Size.** `size_t bytes() const`.

### `include/gret/flow.h`: the monotone normalizing flow `MolFlow`

- **The flow.** A mixture-of-logistics CDF: `u(x) = sum_m w_m * sigmoid((x - mu_m) / s_m)`, with `w_m > 0`,
  `sum w = 1` and `s_m > 0`. It is monotone by construction: the elementwise flow layer of Flow++, trained here with a
  flat (uniform) latent by maximum likelihood, the target `docs/proof` argues for.
- **Input.** `x = (k - lo) / (hi - lo)`, computed as `double(k - lo)` (and `-double(lo - k)` for `k < lo`) divided by
  `double(hi - lo)`. `lo` and `hi` are stored in the file: by default the 0.1% and 99.9% quantiles of the training
  sample, so that outliers (fb reaches 2^64-1) do not crush the bulk's resolution.
- **Output.** `T(k) = clamp(floor(u * 2^52), 0, 2^52 - 1)`.
- **File format.** Text:
  - a header line `molflow 1 M lo hi`;
  - then M lines `w mu s`, written with `%.17g`.
- **Loading.** `static MolFlow load(path)` validates everything: finite values, `w > 0`, `s > 0`, and
  `|sum w - 1| < 1e-9`, then renormalizes.
- **Evaluation.** Plain loops, written so that clang and gcc can vectorize them. `bytes()`, and `cost_terms()`
  returning M.

### `transform/train_flow.py`

- **Model and data.** torch, float64, CPU. It reads a GRE key file (a uint64 count, then the keys) and takes a sample
  of the sorted keys: 2M by default, stride or random with a fixed seed.
- **Training.** Adam on `-mean log du/dx`. Parameters are softmax logits for w, `mu` (initialized at sample
  quantiles), and softplus-raw for s (initialized from the quantile spacing). Defaults: M=32, about 2,000 steps,
  minibatches. Seed everything.
- **Outputs.** The flow file, plus a JSON report with:
  - training NLL;
  - KS of `u` to uniform on the whole sample;
  - the number of decreasing pairs of `T` on the sorted sample;
  - the collision fraction of `T` on the sample;
  - the wall time.
- **Threads.** `--threads` (default 4).

### `include/gret/transform_index.h`: the wrapper `TransformIndex<K, P, Base, Tr>`

It implements GRE's `indexInterface<K, P>`. `Base` is a GRE interface class such as `LIPPInterface<uint64_t,
uint64_t>`; `Tr` is `PwlCdf` or `MolFlow`.

- **`bulk_load`** (GRE passes the keys sorted):
  1. Build or load `Tr`. `PwlCdf` is built from the input keys with `K = TRANSFORM_K` (env, default 512). `MolFlow`
     is loaded from the file named by `TRANSFORM_FLOW` (env); stop with a clear message if it is missing.
  2. Compute `t_i = Tr(k_i)` and sort `(t_i, i)` by t. Sorting is needed: NF rounding can make `t` locally
     non-monotone.
  3. Every group of equal `t` with two or more members is a **collision group**. All of its members go into
     `overflow_` (raw key and payload, sorted by key), and their `t` goes into `collided_` (sorted, unique).
  4. The remaining unique `(t, payload)` pairs are bulk-loaded into `base_`.
  5. Print one line to stderr: `gret: <name> n=... unique=... collided_keys=... groups=... build_ms=...`.
- **`get(k)`:**
  1. Compute `t = Tr(k)`.
  2. If `collided_` is non-empty and `t` is in it, binary-search `overflow_` by raw key.
  3. Otherwise return `base_.get(t, val)`.

  The common path costs one predictable branch.
- **`memory_consumption()`** is the base's own figure plus `Tr.bytes()` plus the overflow and collided arrays.

### `include/gret/gate.h`: the gate `GateIndex<K, P, Base>`

- **What it chooses.** At `bulk_load`, from features of the keys, it picks one of `raw`, `pw512`, `pw4096` and `nf`
  (`nf` only if a flow file is present), then builds that variant. It prints
  `gret: gate <base> choice=<c> features=[...] feature_ms=...`.
- **Features** (`gret::GateFeatures compute_features(const uint64_t* sorted, size_t n)`). Cheap: computed from a
  stride sample of at most 65,536 keys, plus the exact min and max.
  - log2 of the key range;
  - the share of the key range covered by the central 99.8% (outliers);
  - KS to uniform on the sample;
  - optimal ε-PLA segment counts on the sample at ε = 4 and 64 (an O(n) shrinking-cone or optimal PLA), per 1,000
    sample keys;
  - NFL's tail conflict degree (99th percentile, amplification 1.5) of the raw sample, and of the sample transformed
    by a `PwlCdf` with K=512 built on the sample;
  - the coefficient of variation of the gaps;
  - the collision fraction of `PwlCdf(512)` on the sample.
- **The decision model.** The parameters live in `include/gret/gate_model.h`, generated by `gate/train_gate.py`.
  Until it is trained, the model is a documented placeholder that always chooses `raw`.
- **A non-learned baseline, `nflsw_<base>`.** NFL's switch: use `pw512` if its tail conflict is at most the raw one,
  else `raw`.

### `include/gret/registry.h`

`template<class K, class P> indexInterface<K, P>* make_index(const std::string& name)` returns nullptr for unknown
names.

- **Names.** `pw<K>_<base>` (any K from 2 to 65536, e.g. `pw512_lipp`, `pw4096_btree`), `nf_<base>`, `gate_<base>`,
  `nflsw_<base>`.
- **Bases.** `lipp`, `btree`, `alex`, `pgm`, and `artunsync` only where GRE's registry has it, i.e. not on macOS. Use
  `#ifdef`, or detect it through the include guard of `artunsync.h`.

## Build integration (the baselines path stays unchanged)

- **`scripts/add_transforms.py GRE_DIR REPO_ROOT`** makes anchored edits, like `trim_gre.py`:
  - It adds `#include "gret/registry.h"` after `#include "btree/btree.h"`.
  - It inserts the line `  else if ((index = gret::make_index<KEY_TYPE, PAYLOAD_TYPE>(index_type)) != nullptr) {}`
    right before the `  else {` of the "Could not find" branch.
  - On Linux it adds `include_directories(REPO_ROOT/include)` to `CMakeLists.txt`.
  - Each anchor must occur exactly once.
- **`scripts/build_gre.sh`:**
  - With `EXTRA=transforms`, it runs `add_transforms.py` after the trim, adds `-I$ROOT/include` to the macOS compile,
    and records the extra in `BUILD_MANIFEST.txt`.
  - Without it, the behaviour is unchanged. Test this: `competitor.h` must be identical.
- **`scripts/run_baselines.sh`:**
  - With `FLOW_DIR` set, it exports `TRANSFORM_FLOW=$FLOW_DIR/<dataset>.flow` per dataset.
  - `TRANSFORM_K` passes through.
  - Add both to `SETTINGS` and `MANIFEST.txt`.
- **`data/sample.py SRC_DIR OUT_DIR --n 10000000`** writes stride samples of the sorted keys as GRE files named
  `<name>_10M`, plus `SAMPLES.txt` with their sha256.
  - Sampling preserves the CDF shape and avoids the sorted-prefix problem of `--table_size`.
  - It uses numpy (a dev tool; say so).

## Tests (`tests/`, one command: `tests/run_tests.sh GRE_DIR`)

All tests compile with `-O1 -g -fsanitize=address,undefined`, using the same include paths and shims as
`build_gre.sh`.

- **`test_pwlcdf.cpp`** runs on synthetic sets:
  - **Inputs:**
    - dense integers;
    - random 64-bit keys;
    - keys above 2^53;
    - outliers up to 2^64-1 (an fb-like shape);
    - n = 1, 2 and 3;
    - K greater than n;
    - two clusters far apart.
  - **Checks:**
    - non-decreasing on sorted random queries, including outside the build range;
    - `T(x_j) = y_j`;
    - every output below 2^52;
    - an exact match with a long-double/__int128 reference.
- **`test_index.cpp`:**
  - **Coverage:** every base (`btree`, `pgm`, `alex` and `lipp`; `artunsync` on Linux), with `pw` (K = 512 and
    4096) and with `nf` (a synthetic flow file).
  - **Exactness:** bulk-load n keys with distinct payloads (`3k + 1`), then `get()` every key and check its payload.
    Sizes: 100k keys, plus 2M keys from each real dataset's sample when available.
  - **Collision path:** force collisions with a flow that saturates and with near-duplicate clusters. Check that
    overflow returns the right payloads.
  - **Memory:** check that `memory_consumption()` is positive.
- **`test_flow.sh`** trains a tiny flow on synthetic data and checks:
  - C++ against torch evaluation: `|du| < 1e-12`;
  - monotonicity on a sorted sample;
  - that the loader rejects malformed files.
- **GRE-level smoke:** every new index name on the 10M samples, through `run_baselines.sh`; each must reach
  `success_read == ops`.

## The gate (`gate/`)

- **Labels** come from Mac timings on the 10M samples.
  - Variants: raw, pw512, pw4096 and nf, on lipp and btree (ALEX and PGM stay off for now), all ten datasets.
  - Settings: `OPS=10000000`, 3 repeats. Run under `caffeinate -ims ~/Downloads/SPLICE-clean/scripts/mac_lane.sh run
    --`, on AC power with Low Power Mode off, one process at a time.
  - On the Mac, raw LIPP fails on covid, osm and books (keys above 2^53). Treat it as a failed variant there: the gate
    must not choose it.
- **Features:** `tools/gate_features.cpp` (the same code as `gate.h`) writes per-dataset JSON.
- **Training:** `gate/train_gate.py` uses leave-one-dataset-out. Candidate models are a threshold rule on one feature
  and a depth-2 tree.
  - **Report:** for each base, the regret against the per-dataset oracle (the best measured variant, median
    throughput) and the share of oracle throughput. Compare with always-raw, always-pw512, always-nf and NFL's
    switch.
  - **Output:** `include/gret/gate_model.h`, and `results/gate_mac_<date>/REPORT.md` with every number traced to
    `runs.tsv`.
- **Caveat to state:** labels from 10M samples on one Mac; ten datasets; LODO is the only validation.

## Rules for whoever implements this

- Work in `~/Downloads/GRE-transform` (branch `gre-transform`).
- Never touch `~/Downloads/GRE-baselines` or `~/gre-baselines/GRE`. Build into your own GRE clone:
  `GRE_SEED=$HOME/gre-baselines/GRE scripts/build_gre.sh <your dir>`.
- Never push. Never ssh. Never change system settings.
- While `pgrep -f scratchpad/mac_run.sh` finds a process, a timed baseline run is in progress. Do not compile (except
  single small test files under `taskpolicy -b nice -n 20`) and do not run any benchmark until it has finished.
- Timed runs only under the lane, on AC power with Low Power Mode off (`pmset -g`: powermode is not 1).
- Data: the GRE key files are in `~/Downloads/scaleli_sota/experimental/scaleli/data/external/gre/` (ignore the
  `*.sorted` files there). Python with numpy and torch is `/usr/bin/python3`: never pass `-I` or `-s`, because the
  packages are in the user site.
