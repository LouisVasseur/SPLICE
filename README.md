# SCALE-LI: chronological learned-index walkthrough

Start with **docs/REHEARSAL.md** and **slides/SCALE-LI_SOTA_Walkthrough.pdf**.
Open **results/saved/report.html** to see the saved evidence immediately.
To build, fetch the datasets and run the 200M-key protocol on another machine, see **SERVER.md**.

This is a literature-to-experiment workspace for a first supervisor meeting, not a
claim that a new custom index is the SOTA. Each slide identifies published evidence,
a teaching demonstration, or a native experiment. Full upstream reproduction of
ALEX/PGM/NFL/CSV remains incomplete; see **docs/APPROACHES.md**.

## Offline workflow (Python 3.10+, CMake, C++ compiler)

```bash
python3 walkthrough.py check
python3 walkthrough.py prepare
python3 walkthrough.py demo all
```

Open results/local/report.html. For a short live session run lessons 02, 05 and 06.
No pip packages, internet connection, Docker or GPU are needed for these commands.
The PPTX contains speaker notes; the PDF is a presentation fallback. The optional
notebook gives a cell-by-cell interface (Jupyter is optional, not auto-installed).

## Navigation

| Lesson | Topic | Evidence/implementation |
|---|---|---|
| 01 | Predict rank, then specialize | Independent two-stage teaching model |
| 02 | Bounded spline search | Native RadixSpline source-derived transcription |
| 03 | Transform the keys | NFL published tables and arithmetic, not NF execution |
| 04 | Compress with exact access | Custom map controls, not LeCo reproduction |
| 05 | Insert virtual points | Independent greedy/exhaustive toy plus separate CSV evidence |
| 06 | Geometry and maintenance | Custom compression/routing/update ablations |
| 07 | Audit prior numbers | Archived paired trace results and the user's Mac transcript |
| 08 | Learnability controls | Clean-room NFL-style key transform and CSV-style virtual points inside the experimental map; paired traces, not NFL/CSV reproductions |
| 09 | Dataset hardness protocol (AIDB 2026) | Reduced-scale, clean-room application of the AIDB 2026 hardness/conformance protocol to our NFL-style and CSV-style controls on the ten GRE datasets (`experimental/scaleli/tools/aidb_pipeline.py`); summary only until that pipeline has run; not a reproduction of the paper |

The historical slide path also covers FITing-Tree, PGM, ALEX, LIPP, LA-vector, LeCo,
CSV, and more recent benchmarking/robustness/LINE work. A paper's inclusion does not
mean its implementation is integrated. Sources and status are in docs/APPROACHES.md.

## Files

- `walkthrough.py`: all preparation and lesson commands.
- `experiments/lessons.py`: inspectable definitions and correctness checks.
- `native/radix_walkthrough.cpp`: native baseline wrapper and differential tests.
- `third_party/radix_spline/`: MIT source-derived transcription and provenance.
- `experimental/scaleli/`: original custom prototype, isolated from the literature. Now also hosts
  `include/scaleli/transform.hpp` (NFL-format flow inference, MKL-free), `include/scaleli/smoothing.hpp`
  (CSV Algorithm 1 in closed form), `tools/train_flow.py` (stdlib flow trainer), `tools/prepare_flows.py`,
  `configs/learnability.json` (sweep) and `tools/sweep_report.py` (stdlib HTML/SVG view of any sweep).
  `tools/aidb_pipeline.py` (AIDB 2026 hardness lane: verify downloads, sample, train flows, `scaleli_hardness`, paired
  sweep, conformance/coverage scores, HTML report; `--dry-run` needs no network) feeds lesson 09.
- `evidence/`: source tables, fixture hashes, validation logs and previous raw data.
- `results/saved/`: recorded, immutable rehearsal outputs used in the deck.
- `results/local/`: your own reruns, not used to alter the deck automatically.
- `slides/`: presentation, speaker notes and its construction script.
- `sources/`: user-provided NFL and CSV PDFs; other sources are linked in the registry.

## Measurement limits

The new live dataset is 16,384 distinct synthetic uint64 keys. L02 tests static
lower_bound on a keys-only array, while L04/L06 include uint64 values and metadata.
Do not compare those footprints without adapting the contracts. Timings are tiny
single-host demonstrations, not publication estimates. Full real-data/SOTA work is
an explicit next milestone. See evidence/VALIDATION.md for what actually passed.

## License

New harness and teaching code: MIT (LICENSE). RadixSpline-derived source retains
its original MIT notice. The experimental prototype retains its own notice. Source
papers are user-provided reference materials and are not relicensed by this repo.
