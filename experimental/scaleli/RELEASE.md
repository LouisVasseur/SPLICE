# Release 0.1.0

This release contains a working dependency-free core, not a claim of a completed SOTA comparison.

- Canonical specification: `docs/TECHNICAL_SPEC.md` (approximately 10,700 words).
- Rendered specification: `docs/TECHNICAL_SPEC.pdf` (38 pages); all pages rendered and visually inspected, and text-bound checks passed.
- Executed test record: `results/example/VALIDATION.md`.
- Actual paired experimental outputs: `results/example/suite/`.
- Source registry: `docs/SOURCES.json` and the specification bibliography.
- Integrity: `MANIFEST.sha256` covers packaged files except itself.

Large external datasets, third-party implementations, compilers, compiled binaries, and font files are not bundled. The synthetic fixture, original source, documentation, experiment configurations and actual local results are included.

The initial public-upstream fetch is not frozen by the release. The optional fetch script captures a resolved commit lock for subsequent runs; that lock must be reviewed and retained by the experimenter.
