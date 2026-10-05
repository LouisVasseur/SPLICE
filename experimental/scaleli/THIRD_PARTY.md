# Third-party material and reproducibility

The default source distribution contains original implementation code and a synthetic fixture. It does not redistribute the user's two papers, real benchmark datasets, or upstream implementations.

The specification cites research ideas without representing this code as their official implementation. In particular, the simple integer-linear residual codec is not LeCo; the tiny virtual-point lab is not the optimized CSV implementation; the affine model is not the error-bounded PGM algorithm.

ALEX, PGM-index, GRE, SOSD, RoBin, NFL, LA-vector, LeCo, LINE, LICO, and the SEA comparison suite are external projects. Their licenses and dataset terms must be read in the exact fetched revisions. MIT licensing of this kit does not relicense those projects or their data.

`tools/fetch_external.py` fetches the official ALEX and PGM-index repositories, checks origin/dirty-tree state, and writes exact resolved commits to `external.lock.json`. The first fetch is **not predetermined** by this release: it captures the available upstream revision at the time you run it. Archive that lock and use `--locked` to reproduce it. Review the code and licenses before compiling. The script does not execute upstream shell scripts.

Bulk network downloads did not succeed in the build environment used for this release. Public source scripts and API declarations were read through web access. Therefore upstream adapters are delivered as reviewed, uncompiled integration code, and dataset locations as source-verified acquisition recipes, not as successfully downloaded artifacts.

The synthetic fixture and generated experimental results may be redistributed under the kit license. The PDF and Markdown bibliography identifies all researched primary sources. No training or query data from personal user accounts is included.
