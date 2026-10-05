# Tomorrow: a rehearsal, not a live integration session

## What to open

Use this NEW folder, scaleli_sota, not the older Downloads/scaleli directory.
The old Codebase Reading Guide explains the custom engine; it is not the
chronological presentation.

- slides/SCALE-LI_SOTA_Walkthrough.pptx — editable deck with speaker notes.
- slides/SCALE-LI_SOTA_Walkthrough.pdf — easiest offline presentation fallback.
- results/saved/report.html — all seven saved numbered results; no setup needed.
- results/local/report.html — your own rerun after the commands below.
- Walkthrough.ipynb — optional cell-by-cell execution; not needed for the meeting.
- docs/APPROACHES.md — exact algorithm coverage and remaining gaps.

## Tonight: allow 30–45 minutes for preparation

Extract the ZIP. In Terminal, type `cd ` (including the space), drag the extracted
scaleli_sota folder into Terminal, and press Return. That establishes the correct
path without assuming how your browser names extracted folders.

Run one command at a time:

```bash
python3 walkthrough.py check
python3 walkthrough.py prepare
python3 walkthrough.py demo all
open results/local/report.html
open slides/SCALE-LI_SOTA_Walkthrough.pdf
```

`check` validates the fixture and archived evidence; it does not benchmark anything.
`prepare` configures/builds the C++ programs, runs three CTest entries and twelve
Python tests. Stop on errors. `demo all` runs all seven small lessons and builds
an offline result page. No third-party Python packages or data downloads are needed.
The shipped results were produced on Linux, not your Mac: compare within your own
run, and expect timing differences. Do not promise a particular runtime.

On a Mac with missing compiler tools, use `xcode-select --install` and finish the
installation first. With Homebrew already installed, `brew install cmake` provides
CMake and CTest. If your shell cannot find Homebrew, use the existing system setup
instructions rather than running arbitrary installer commands during the meeting.
Python 3.10 or newer is required. Full Xcode, PyTorch, Docker and a GPU are unnecessary.

## Rehearse the first 20 slides in 25–30 minutes

| Slides | Time | One sentence to say | Action |
|---|---:|---|---|
| 1–3 | 2 min | This is a history and baseline workspace; papers and local experiments are separated. | Show the evidence legend and roadmap. |
| 4–5 | 3 min | A sorted array turns indexing into rank prediction, but local irregularity remains. | Open L01; show error versus model count. |
| 6–7 | 2 min | FITing-Tree and PGM turn error tolerance into an explicit design parameter. | Explain the literature; do not claim those systems ran. |
| 8 | 3 min | RadixSpline combines a small directory and spline with exact final search. | Run demo 02; inspect the window size and measured timings. |
| 9 | 2 min | ALEX and LIPP address updates using different data placement and conflict handling. | Explain the source mechanisms; no native reproduction claim. |
| 10–11 | 3 min | NFL changes the input, and its query-time transformation must pay for itself. | Show L03 and paper tables; no training live. |
| 12–13 | 3 min | Compression with random access is prior work; smaller bytes do not automatically mean faster lookup. | Show L04 measured map controls, separate from LeCo. |
| 14–15 | 3 min | CSV adds positions; a small teaching example lets us see the fit change. | Run demo 05; name the augmented-set objective. |
| 16–17 | 3 min | Our project should test useful search work and update costs, not only fitting error. | Show L06; run it only if there is time. |
| 18–20 | 3 min | Recent work makes robustness and maintenance part of the baseline, not optional polish. | Agree on the next real-data/upstream milestone. |

## Only three live commands

```bash
python3 walkthrough.py demo 02
python3 walkthrough.py demo 05
python3 walkthrough.py demo 06
```

Refresh results/local/report.html after a command. L02 is the native static search
experiment. L05 is an independently written virtual-point teaching toy, NOT CSV.
L06 is the custom compression geometry/update comparison, NOT a published SOTA index.
Everything else can be shown from saved output. The command shown on each slide
uses these exact lesson IDs. The deck's numeric charts use results/saved/ and will
not silently update when you rerun locally; label a differing Mac result as your run.

## If something fails

Stop running code. Open results/saved/report.html and the PDF deck. Explain:
"These are recorded runs with their environment; my local rerun needs a setup fix."
Do not debug installations, fetch upstream dependencies, or download 200-million-key
files during the meeting. The saved JSON and logs remain inspectable offline.

## Opening script

"I organized existing approaches around three choices: how to predict positions,
how to maintain the index when data changes, and how to change or compress the
representation. The deck links papers to small, inspectable experiments. Some are
teaching demonstrations and some are measured native code; I have not reproduced
all published systems. I want to use this baseline to agree on the next experiments,
not to commit prematurely to a new index design."

## Closing decision

Agree on the mandatory implementations (for example ALEX, PGM, RadixSpline and an
optimized conventional baseline), the first real datasets, and the initial API
contract. A practical next checkpoint is a commit-pinned static comparison on one
shared real corpus, followed by dynamic point/scan/update tests. Present this as a
proposed milestone, not an already completed reproduction.

## Questions you should be able to answer

- Which numbers are published? NFL/CSV source results; L03 recomputes arithmetic.
- Which code is genuinely source-derived? The small RadixSpline transcription,
  with MIT attribution and disclosed lack of a pinned upstream commit.
- Are the local data real? No: a fixed, checksummed synthetic dataset.
- Does compression alter logical rank? No, for the same stored keys.
- Has speed superiority been established? No. The small experiments show trade-offs.
- What exactly passed? Read evidence/VALIDATION.md and the recorded test logs.
