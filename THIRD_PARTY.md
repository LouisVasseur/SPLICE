# Third-party material

**Our code is MIT** ([LICENSE](LICENSE)). The MIT licence of this repository does not relicense anything below.

## Code ported into this repository

Both ports are in `include/scaleli/hardness.hpp`, used by `src/hardness.cpp` and, through `include/splice/cost.hpp`,
by every SPLICE-H build (`splice_count`, the GRE `splice` indexes, the tests).

| Port | Upstream | Licence |
|---|---|---|
| LIPP FMCD: `build_tree_bulk_fmcd`, `compute_gap_count`, `PREDICT_POS` | github.com/Jiacheng-WU/lipp `src/core/lipp.h`, commit fe6ca4954f00875482f9e4dd63b34dae2384d23b | MIT, Copyright (c) 2021 Jiacheng-WU: [LICENSES/MIT-LIPP.txt](LICENSES/MIT-LIPP.txt) |
| PGM-index `OptimalPiecewiseLinearModel::add_point` / `make_segmentation` | github.com/gvinciguerra/PGM-index `include/pgm/piecewise_linear_model.hpp`, commit c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407 | Apache-2.0, Copyright (c) 2018 Giorgio Vinciguerra: [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) |

Modification notice (Apache-2.0 section 4(b)): both ports are modified. They are re-typed for exact uint64
arithmetic (unsigned/signed `__int128` differences and cross products; `long double` only for the final slopes and
residual products), so raw keys are never rounded to double. The `hardness.hpp` header lists what was taken and
how it changed. PGM-index ships no NOTICE file.

## Re-implemented, not copied

- `include/scaleli/transform.hpp` is a clean-room re-implementation of NFL's inference path (luffy06/NFL, GPL-3).
  None of NFL's code is copied; it reads the same 2D2H2L text weight format. `data/flows/*.txt` are our own trained
  weights (`data/flows/README.md`), not NFL's author models.
- Ideas cited, not code: CSV, NFL, Hist-Tree, ALEX, LIPP, PGM-index, GRE and the AIDB 2026 hardness metrics.

## Never in this repository

- GRE (gre4index/GRE, no licence file), the STX B+tree (GPL-3), ALEX, LIPP, PGM-index, ART and NFL.
  `tools/gre_lite.sh` and `tools/gre_mac.sh` clone GRE at e807edc **outside** the repository and edit it there.
- The files those scripts write into the GRE clone are ours (MIT): `integrations/gre/*`, `integrations/gre_splice/*`,
  `gre_lite_patch.h`, `sortedarray.h`, the trimmed `competitor.h` dispatch and the trimmed `CMakeLists.txt`. They
  name GRE's interface classes and anchor on single GRE lines; the dispatch and CMake text is the interoperability
  glue needed to register the indexes (one `include`/`new` line per index, the build statements), not GRE logic.
- `results/**/bin/microbench` (a GRE + STX binary) is ignored; `bin/gre_lite_build.txt` next to it is a text
  record of our build and is committed.

## Data and papers

- The GRE/SOSD datasets are fetched by `data/external/gre/download.sh` under their own terms and never committed;
  neither are their `.sorted` copies or the 2M samples. `data/samples/*.manifest.json` are metadata (hashes,
  modes, seeds). `data/dense_sparse_fixture_uint64` is synthetic and ours.
- Papers are cited by arXiv id (NFL 2205.11807, CSV 2408.06134) or venue and are not redistributed.
