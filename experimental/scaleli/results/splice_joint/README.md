# SPLICE joint G+T+V: diagnoses, ranked hypotheses and the window suite (2026-10-06)

Working assumption (Louis): G (learned compression: exceptions, holes, spills), T (the model: knots, slopes) and V
(virtual points: slack, fences) can work together on one exact-count loss J, and each must earn its place. An
ablation that costs about nothing is a defect in how they are combined, not a fact about the data.

All numbers here are counts (dependent DRAM lines, page walks, E[D]), never timings. The run stopped after the
diagnosis and the window suite, before the hypothesis rounds, because the agents' weekly usage limit was reached.
Paths under `/private/tmp/...` in these files are the session scratchpad where the work ran; the files are copied here.

## What the diagnoses found (`diagnoses.json`, merged in `hypotheses.json`)

- **The layout is the blocker, not the data.** On fb, osm, genome and planet every segment is FENCED, and
  `make_plan` (include/splice/cost.hpp) puts all directory entries in one region and all data lines after it. Every
  FENCED lookup therefore pays 2 dependent lines and 2 page walks, a floor that G, T and V cannot move. Their only
  lever is the over-capacity residue (fb 0.08, osm 0.21, planet 0.09, genome 0.0004), which is why the T ablation
  (Hist-Tree knots) lands within 0.03 D and G (exceptions) only shows on fb's router (+0.24 D).
- **Each component is trained on a proxy or frozen:** T's knots come from eps-PLA and only K1 is selected by J,
  and each segment uses the endpoint chord. G is a span rule on the two ends, with at most 64 exceptions per end. V
  is one cbar or alpha per cell. The outer fixed point varies only the three L2 page-miss scalars, so it stops after
  one iteration. The old ablations are not fair: the other components are not re-optimized.
- **Bytes are free at every chosen hard cell** (lambda = 0), so nothing turns bytes into fewer lines or walks.

## Ranked hypotheses (`hypotheses.json`)

1. **H1 PAGED mode.** Co-locate each directory entry with its group's data lines in the same 4 KiB page; groups
   that do not fit spill (G) to a shared area. T (knots, slope) and V (slack per page) decide what fits, and G decides
   what spills, so all three trade on J. Counted on all 200M keys with a scratch prototype (default constants):
   fb 4.52 -> 3.65-3.95 D and osm 4.61 -> 3.51-3.80 D at 19.9-20.7 B/key; books 3.30 -> 2.79; stack unchanged
   (stays DIRECT). genome and planet are estimates only (about -0.5 to -0.8 D).
2. H2 co-located FENCED fallback (diag A: fb -0.43, osm -0.38 to -0.44, genome -0.58 to -0.83, planet -0.73 D, near
   the 22 B/key cap).
3. H3 T trained on J (knot moves scored by the exact count). 4. H4 G's spill set trained on J. 5. H5 per-segment V
   density. 6. H6 order-free DIRECT with a stash. 7. H7 an outer loop over G, T and V. 8. H8 fair ablations.
   9. H9 tier-1 capacity for shape. 10. H10 router exceptions chosen by J.

## Window suite (`window_suite/`, `window_suite.json`)

12.5M-key windows (200M/16; each is 16 stratified contiguous runs, plus fb's contiguous tail for its 21 outliers),
priced as at 200M. Validated: every window is within 0.085 D of the full-scale E[D] of the chosen cell and within
0.008 of its DIRECT share; verify and parity pass. `run_suite.sh DIR OUT.json` runs the joint cell and the three
fair ablations (G0, T0, V0, the other two re-optimized) for a source tree in about 3 minutes, five trees at once.
`window_mode.patch` adds the window mode and ablation switches to params.hpp, cost.hpp and splice_count.cpp; it was
made against the tree before the review-fix stage and needs rebasing on the committed SPLICE-H.

## Next

Rebase `window_mode.patch`, implement H1 with G, T and V trained on J (H3-H5, H7), and accept a change only if the
fair ablations show each component worth at least 0.05 D on the hard windows; then confirm at 200M and time in GRE.
