# Changelog

## v1.0.1 — 2026-11-24 (bug fix)
- Command-line default of `-outlierRatio`: typo `0.9` → `0.09`.
  - The typo had been in the command-line default only, since the first upload (2019-07-31, `8a21d04`).
    The paper (Algorithm 1 and §IV: γ = 0.09), the help text and `IDEA.fit()` all use 0.09, and the paper
    experiments were run with 0.09. **The published results are not affected.**
  - Because of the typo, running the README example commands with `v1.0-paper` as-is marked almost every point
    as an outlier (`-1`): mean ARI 0.115 instead of 0.922 over the 20 bundled datasets.
  - No other code change. With default options the output equals `v1.0-paper` with `-outlierRatio 0.09` on all
    20 bundled datasets. Use this version to reproduce the paper.

## v1.0-paper — published version (commit `b4964c5`)
Public implementation of IDEA (Ahn et al., IEEE BigData 2021, pp. 2791–2800, doi:10.1109/BigData52589.2021.9671953).
