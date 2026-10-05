# RadixSpline source-derived transcription

Source: https://github.com/learnedsystems/RadixSpline
Files inspected through the web tool on 8 September 2026:
- https://raw.githubusercontent.com/learnedsystems/RadixSpline/master/include/rs/common.h
- https://raw.githubusercontent.com/learnedsystems/RadixSpline/master/include/rs/builder.h
- https://raw.githubusercontent.com/learnedsystems/RadixSpline/master/include/rs/radix_spline.h
- https://raw.githubusercontent.com/learnedsystems/RadixSpline/master/LICENSE

The container could not fetch a Git checkout. These three small headers were
transcribed from the official publicly displayed source, retaining its classes,
algorithm, parameters and arithmetic. Whitespace/comments were shortened and
standard includes were made explicit. This is a SOURCE-DERIVED TRANSCRIPTION,
not a byte-identical or commit-pinned upstream release. Repository checksums
pin the delivered local text; they do not certify equality to an upstream commit.

The actual radix table, spline corridor builder, interpolation and bounded
correction are used. Native tests compare the resulting lower_bound against
std::lower_bound. The benchmark wrapper handles empty/singleton arrays outside
the original builder (which computes clz(diff)). Do not present timing as a
reproduction of the paper's complete benchmark. No ALEX/PGM/NFL/CSV upstream run
is implied. A future release should replace these files with a pinned Git checkout
and rerun parity and compiler checks before making upstream-comparison claims.

License: MIT; see LICENSE. Citation: Kipf et al., RadixSpline: a single-pass
learned index, aiDM 2020. https://arxiv.org/abs/2004.14541
