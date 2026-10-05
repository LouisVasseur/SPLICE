# SCALE-LI in GRE

GRE (gre4index/GRE @ e807edc, Wongkham et al., VLDB 2022) is the benchmark harness the learned-index literature
uses, NFL's and CSV's papers included. These files let GRE's `microbench` run SCALE-LI as six more indexes, next to
ALEX, LIPP, PGM, the B+tree, ART and `sortedarray` from `tools/gre_lite.sh`. How to wire them into the build:
[INTEGRATION.md](INTEGRATION.md).

| File | What it is |
|---|---|
| `scaleli_gre.hpp` | Facade header, C++11-compatible, std types only: an opaque handle with bulk_load/get/put/update/remove/scan/memory. |
| `scaleli_gre.cpp` | The facade, C++20. Builds a cell's `Config` with scaleli_bench's own parser (`scaleli/cli.hpp`) and prints the `scaleli_*` stats lines. |
| `scaleli_interface.h` | GRE's `indexInterface<K,P>` for the cells, C++17. Copied into GRE's `src/competitor/scaleli/`; it includes GRE's `../indexInterface.h` at build time, and nothing of GRE is copied here. |
| `facade_check.cpp` | C++17 driver (target `scaleli_gre_check`): builds a cell through the facade from the records scaleli_bench loads and replays a `--dump-trace` read trace. |
| `compare.py` | Python 3.6+, stdlib only: checks that a facade or GRE log describes the index scaleli_bench builds (memory, learnability, root, checksum, counters). |
| `apply_integration.py` | Applies INTEGRATION.md's edits to `tools/gre_lite.sh` and `tools/gre_run.sh`, anchored on text. |

Why a separate translation unit: GRE compiles its whole benchmark as one C++17 TU, and ALEX in that TU uses
`std::allocator` members that C++20 removed. SCALE-LI is C++20, so it sits behind a handle with a std-types-only
header, compiled as its own C++20 library and linked into `microbench`. Each GRE operation already pays one virtual
call; the facade adds one direct call into the other TU, where `Index::find` is inlined.

## GRE cells

The GRE index names are lowercase because `gre_run.sh` accepts only `[a-z0-9_]` index names. Each name is a cell of
`results/aidb_ba/run_ba.py`'s `cells(d)` plus the joint root, with every cell built with
`--build-threads $SCALELI_BUILD_THREADS` (default 16, the protocol's).

| GRE index | Cell | Flags (scaleli_bench) | What it is |
|---|---|---|---|
| `scaleli_b` | B | `--policy min_bytes --routing rank --root model` | SCALE-LI before NFL and CSV: compressed blocks, rank models, a learned root over the region fences (binary search if the root model does not save probes). |
| `scaleli_n` | N | B + `--flow $SCALELI_FLOW --flow-bypass 1 --flow-cost 0` | Our NFL: the dataset's NFL flow is the region-model feature where NFL's own 10% tail-conflict switch keeps it, and the root is offered the flow candidate for free. |
| `scaleli_c` | C | B + `--virtual-alpha 0.1 --root-alpha 0.1` | Our CSV: CSV Algorithm 1 virtual points in every region (alpha 0.1) and virtual fences at the root, with a uint32 slot-to-region table. |
| `scaleli_cr` | Cr | B + `--root-alpha 0.1` | CSV at the root only: virtual fences and the slot-to-region table, regions as B. The reference for SPLICE vs CSV alone at the root. |
| `scaleli_nc` | NC | C + N's flow flags | Both, applied in sequence. |
| `scaleli_j` | J | NC + `--root-joint-rounds 6 --root-joint-order gtv --root-joint-kmin 0 --root-joint-kmax 64 --root-joint-gap-charge 1` | Our simultaneous learned compression and virtual-points model, with NFL: NC's regions bit for bit, plus a joint G+T+V root candidate. |
| `scaleli_jg0` | Jg0 | J with `--root-joint-gap-charge 0` | The same joint root, with the gap table offered for free, so measured throughput decides whether learned compression pays. |
| `scaleli_splice` | Jg0 | the same as `scaleli_jg0` | SPLICE as reported: compression and warp offered free at selection (N's convention), GRE measures their real cost. At gap charge 1 (J) compression is never chosen on the samples, so J tests only joint T+V. |

The joint root (J, Jg0) is one least-squares model of the root fences, fitted by alternating three blocks with a
guard on the root's probe score (results/aidb_threeblock, arm A):

- **G**, learned compression of the input axis: the k widest fence gaps, among those wider than the median gap, are
  shrunk to the median width through a table of k entries (32 B each).
- **T**, an order-preserving transform (NFL's two-unit tanh decoder): at most two units with non-negative weights,
  gains from a 64-point log grid on [0.01, 200]. It is fitted to the current targets and must stay strictly
  increasing on the fences and fence midpoints.
- **V**, CSV virtual fences at the root, with budget `root_alpha` (0.1, as in NC). The previous virtual fences are
  kept if they fit no worse.

Each round runs G, T, V in order on the current state. The joint candidate is scored like fit_root's other root
candidates: the probes the real root lookup spends on the fences and fence midpoints, plus `gap_charge` ×
ceil(log2(k+1)) for the gap-table search. Round 1 is always accepted; later rounds are accepted only if they lower
the score by more than 1e-6, for at most 6 rounds. The cycle runs for k in {0, 1, 4, 16, 64} and keeps the
lowest-scoring k, ties going to the smaller k. The winner is appended after NC's four root candidates and adopted
only if it is strictly cheaper than all of them and than binary search. J against NC therefore isolates joint
against sequential at the root: the regions are identical, and when the joint candidate loses, J's index is NC's
plus a few hundred bytes of fit diagnostics (`root_joint_bytes`).

Why these settings, from the prototypes:

- **6 rounds, tolerance 1e-6, keep-previous V guard.** These are arm A's guarded settings verbatim (armA.py
  MAX_ROUNDS, TOL). With k = 0, round 1 is exactly the sequential T then V pass, so J can never score worse than
  sequential.
- **Order GTV.** Order has no consistent effect: TGV ends better in 25 of 80 cells, with a mean difference of 0.044
  probes. Five of the six robust cycle gains are GTV (aidb_threeblock/RESULT.md section 3). `--root-joint-order tgv`
  stays available through `SCALELI_ARGS`.
- **k grid {0, 1, 4, 16, 64}, minimum total.** The prototypes' best is the minimum over k with k = 0 included. k is
  absolute because the table's search cost, ceil(log2(k+1)), does not grow with n.
- **V budget `root_alpha` 0.1, not the prototypes' 4.** At equal budgets J against NC isolates jointness. At 200M,
  alpha 4 costs about 2,000 s per root V run, and the cycle can need 30 runs.
- **T charge `flow_cost` 0.** This is N's convention: the transform is offered free and throughput decides. J's T
  is fitted from scratch and never reads the NFL weight file, because the free flow is not monotone and cannot feed
  a slot table.
- **Gap charge 1 in J, 0 in Jg0.** At charge 1 the score is the prototypes' `total_uncharged`, and on the 2M
  samples G is then never chosen (k = 0 on 20 of 20), so J measures the joint T and V fit against the sequential
  one. A 64-entry table lives in L1 while fence probes chase pointers, so a 1:1 charge is a modelling guess. Jg0
  removes it; on the samples it then picks G on 19 of 20, so Jg0 is the cell where learned compression is actually
  exercised.

Every scaleli_* run prints, once after bulk_load, `scaleli_cell`, `scaleli_config` (the exact scaleli_bench flags;
pasting them into scaleli_bench rebuilds the same index), `scaleli_flow_file`, `scaleli_build_threads`,
`scaleli_bulk_load_ns`, `scaleli_accounted_bytes`, `scaleli_memory` (scaleli_bench's `memory_before`),
`scaleli_root` (binary, raw, raw_vp, flow, flow_vp, joint or joint_vp), `scaleli_flow_regions`,
`scaleli_virtual_points`, `scaleli_root_virtual` and `scaleli_learnability` (scaleli_bench's learnability block,
including the `root_joint_*` diagnostics for J and Jg0).
