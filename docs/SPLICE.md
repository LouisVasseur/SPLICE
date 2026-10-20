# SPLICE on GRE (branch `splice-gre`)

SPLICE is a read-only, exact learned index, written as one GRE index (`include/gret/splice.h`, `SpliceIndex`). It is
built from three components, each a relaxation of the key set that the model has to fit:

- **G, reduce:** keys leave the model's key set.
- **T, the top:** the model itself.
- **V, expand:** slots are added to the key set.

Every decision is an exact count made at build time, never a timing. Two measured constants price what the count
cannot see.

## The design, top-down

| level | decision | how it is made |
|---|---|---|
| global | **G**: peel end outliers | `peel_outliers` (exceptions.h): at most 64 per end, while the span exceeds 2x the trimmed span. Exact at build. Peeled keys go to a small sorted array behind one range check. Nothing peeled means nothing paid: the check reads two fields. |
| global | **T, knots**: where segments start | Optimal eps-PLA over (key, rank), using PGM-index's `make_segmentation`. Each segment keeps its own fitted line, with \|error\| <= eps. The ablation `splice_eq` uses K equi-rank knots instead (the pw transform's), with the same K. |
| global | **T, granularity**: eps, hence the number of segments | Candidates eps = 1, 2, 4, ..., 256. Each is laid out by V, and the least of E[lines] + 0.02 x slots/key + 0.15 x log2(top / 1 MiB) wins. |
| segment | **V, expansion** e in {1, 1.25, 1.5, 2, 3, 4} | Per segment, the least of (exact lines touched by a lookup of each of its keys) + 0.02 x slots. The price per slot is then raised, by bisection, to the least value that keeps the index within 32 bytes per key. Ties go to the smaller e. e = 1 is the packed array, i.e. no V. |
| line | search | Read the predicted line: 4 slots, counted without branches. Then read further, at most the segment's exact maximum displacement. |

**Placement (V).** Each key goes to the first free slot of its predicted line, or later. The predicted slot is
`b + floor(e (r0 + a (key - x0) - f))`. An empty slot holds a copy of the next key and its payload, so the slots
stay sorted and a scan stops at the first key >= its own. Segment bases are line-aligned, and the slot array is
64-byte aligned, so "lines touched" is exact.

**Top.** `PwlCdf::locator` over the segment starts finds a key's segment: a radix bucket, then a few lifting steps
over the starts. Segment entries are 32 bytes in a 64-byte-aligned array, so an entry never straddles two lines.

**Robustness.**
- A segment line that would extrapolate far past its own keys is replaced by its chord. This happens to the last
  interval when G is off and the last key is far away (fb, 2^64 - 1).
- Keys must be unique, as GRE's driver guarantees: it sorts and deduplicates before bulk-loading
  (`benchmark.h:133`).

### Measured constants (the only non-exact parts)

| constant | value | what it prices | measured |
|---|---|---|---|
| space price | 0.02 lines per slot | V's memory: an extra slot per key must save 0.02 lines per lookup | 10M keys: stack keeps 37.9 M/s at 20 B/key instead of 32 |
| top price | 0.15 lines per doubling past 1 MiB | a top that leaves the cache: misses first, then TLB misses and page walks | 10M (genome: K 16k → 65k → 262k cost +6 and +12 ns at about 30 ns per line). 200M (genome, fb, planet): the eps that wins at each |
| cap | 32 bytes per key | total memory | twice the 16-byte key/payload pairs |

Two tried alternatives were dropped:
- **Pricing the top as 2 lines times its uncached share.** It priced a 30 MB top at 10M as free, chose eps = 1–2,
  and halved throughput (genome 13.3 against 28.7 M/s).
- **A second learned level in the top** (a root PLA over the segment starts at eps2 = 2, then an 8-wide branchless
  window). It was slower at both sizes. At 10M: fb 14.0 against 19.2 M/s. At 200M: fb 4.43 against 6.45, planet
  6.87 against 8.95, genome 4.43 against 5.02. The root itself (about 1M segments) misses in cache.

## What the theory says each component can and cannot do

The proof development (`SPLICE-exp`, `docs/proof/lean/NFLProof`, no axioms; see `PROOF.md`) bounds the components.
A derivation from it, checked three ways (Lean faithfulness, applicability to this code, numbers on the ten
samples), gives the following:

- **T cannot beat the within-segment deviation.**
  - `kpiece_composite` (Limits.lean:45): a K-piece affine transform followed by one line is a K-segment PLA of the raw
    keys, with the same errors.
  - `cv_floor` and `two_run` (Limits.lean:113, :71): the run-span dispersion inside a piece forces an error floor
    that no affine model of that piece can avoid.
  - Hence SPLICE's top is a PLA chosen per dataset (it puts segments where the data needs them) instead of a
    transform in front of another index. It also means T's coarse-shape statistics, KL from linear among them,
    cannot predict the gain: KL measures what T removes, not what costs the index.
- **G is exact and changes the normalisation.**
  - `ks_sandwich` (Error.lean:240): n D_KS - 1 <= the endpoint line's error <= n D_KS, before and after peeling.
  - On fb, peeling the one key 2^64 - 1 takes D_KS from 1.000 to 0.0019 (533x).
  - Every coarse statistic must be computed after G: fb's KL from linear is 6.9 before G and 0.001 after.
- **V lowers a span's error by at most half the keys it adds.**
  - Inserting A keys into a span lowers any piecewise-affine error over it by at most A/2. This lemma is derived
    from the shift argument; it is not in Lean.
  - So V is a fine-scale lever, and its cost is space. That is why its decision is per segment and priced per slot.
- **Not covered by the theory:**
  - model-based placement into gapped slots, which is what V does here;
  - the mapping from lines to time;
  - cache, TLB and memory-level parallelism.

  These are measured, and the timings below are the test.

## How the gates err on "use"

- **G** is used whenever its rule peels a key.
- **V** is used in a segment whenever it lowers the exact line count by more than the slot price, and the price is
  small (0.02 lines per slot).
- **T's** knots and granularity are always the count's best, so every component is used whenever the count says it
  does not hurt.

For the layered compositions in front of GRE's own indexes (`g_`, `pw<K>_`, `vp_`), the derivation found no
certificate that T helps an unmodified ALEX, LIPP or PGM. After G, every T cell measured at 10M loses. So those
layers are kept as context, not as SPLICE.

## The attribution ladder (registry.h)

| name | G | T granularity | knots | V |
|---|---|---|---|---|
| `spl_t` | – | adaptive | PLA | – (e = 1) |
| `spl_gt` | yes | adaptive | PLA | – |
| `spl_tv` | – | adaptive | PLA | yes |
| `splice` | yes | adaptive | PLA | yes |
| `splice_eq` | yes | the same K as splice | equi-rank | yes |
| `spl_fixk` | yes | fixed: n / 610 keys per segment | PLA | yes |

How to read it:
- **G's effect:** `spl_t` → `spl_gt` and `spl_tv` → `splice`.
- **V's effect:** `spl_t` → `spl_tv` and `spl_gt` → `splice`.
- **Knot placement:** `splice_eq` → `splice`.
- **Adaptive granularity:** `spl_fixk` → `splice`.

## Running

```bash
tests/run_tests.sh ~/gre-splice/GRE      # test_splice_edge: adversarial sets, every ladder name (ASan/UBSan and -O3)
build/tests/test_splice-O3 KEYS splice spl_t spl_gt spl_tv splice_eq spl_fixk   # a real key file
GRE_SEED=~/gre-baselines/GRE EXTRA=transforms scripts/build_gre.sh ~/gre-splice/GRE
J=3 PASSES="1 2" GRE_DIR=~/gre-splice/GRE DATA_DIR=... scripts/run_parallel.sh OUT_DIR OUT_DIR/cells.txt
c++ -O3 -std=c++17 scripts/dataset_bounds.cpp -o build/dataset_bounds   # dataset properties, one JSON line per file
/usr/bin/python3 scripts/splice_report.py OUT_DIR --props OUT_DIR/dataset_props_200M.jsonl --plot OUT_DIR/attribution.png
```

Results: `results/mac200M_2026-10-20/` (RESULTS.md, attribution.png).

## Limits

- **One machine.** An Apple M3 Max, with three single-thread runs at once (`run_parallel.sh`, J = 3). macOS cannot
  pin cores, so these timings compare with each other, not with one-at-a-time numbers. The Atom (x86) is not run
  yet.
- **GRE as published:** cold, read-only, uniform lookups of existing keys. Raw LIPP does not run on covid, osm and
  books on macOS, where long double is double.
- **The two prices** were calibrated on this machine, on a few datasets.
- **Not done:**
  - interior G (exceptions inside segments). A variant that peeled into a hash reached a mean of 1.31 lines only
    with 27–43% of the keys in the hash;
  - updates;
  - range scans.
