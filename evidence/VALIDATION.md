# Validation of this meeting release

Executed in this working Linux environment, 8 September 2026. No claim of macOS
validation of the new release is made. The user's earlier Mac pilot belongs to the
previous custom prototype, not this newly packaged repository.

## Executed and passed

- GCC Release configure and build of the new root project.
- Three CTest entries: radix_differential, scaleli_core, scaleli_example.
- Native RadixSpline-derived differential test: 1,159,528 comparisons with exact
  std::lower_bound. Covers empty/singleton wrapper cases, keys and neighboring
  missing values, randomized dense/sparse inputs, upper uint64-domain inputs,
  four error-tolerance settings. There is no fallback hiding mismatches.
- Original custom core: 762,890 assertions, as printed in core_assertions.log.
- Twelve Python unittest methods for the existing driver/tooling.
- Independent Clang Release build and all three CTest entries.
- AddressSanitizer + UndefinedBehaviorSanitizer on the new RadixSpline-derived
  differential test. This is not a new sanitizer audit of every package component.
- All seven numbered lessons executed; native trace checks enabled in their
  measured C++ runs. L04 controls pair trace fingerprints and checksums per seed;
  L06 verifies same-key/partition rank invariance and paired update traces.
- Eight code cells in Walkthrough.ipynb executed successfully using a Python kernel.
- Archived 108-run suite audited for per-group trace/checksum consistency; its
  performance figures are old measurements, not rerun by that audit.
- 24-slide PPTX exported to PDF; all pages rendered and visually reviewed.
- A fresh extraction of the distribution ZIP successfully ran `prepare` and all
  seven demos with the same documented commands; see clean_extraction_validation.log.

Logs: build.log, ctest.log, core_assertions.log, prepare_validation.log,
radix_sanitizer.log, clang_validation.log, walkthrough_validation.log.
Native environment and binary hashes are included in each lesson result.

## Source status

RadixSpline is an attributed source-derived transcription of three official MIT
headers, not a verified byte-identical upstream checkout. A Git commit could not
be fetched in this environment. See third_party/radix_spline/PROVENANCE.md.

ALEX and PGM adapter sources are inherited but their upstream implementations are
not integrated or validated here. Original RMI, FITing-Tree, LIPP, LA-vector, LeCo,
NFL, optimized CSV, RoBin and LINE are not locally reproduced. The deck marks
literature/teaching/native status separately.

## Data and statistical limits

New experiments use a 16,384-key synthetic binary fixture with a recorded SHA-256.
No real SOSD/GRE corpus was downloaded or benchmarked. L02 is static keys-only
lower_bound with five warmed same-process repeats. L04 is a full key/value map
with three seeds on tiny workloads. These runs are useful demonstrations, not
publication-quality estimates, independent statistical superiority tests, or
an exhaustive SOTA benchmark. The full upstream common-data suite is a remaining
milestone, not a completed deliverable hidden behind the word 'verified'.
