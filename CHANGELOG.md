# Changelog

Versioning follows [Semantic Versioning](https://semver.org/):
- patch (x.y.**Z**): bug fixes
- minor (x.**Y**.0): new features or performance improvements that keep the default results unchanged
- major (**X**.0.0): algorithm changes that change the default results

To reproduce the paper (IEEE BigData 2021), use `v1.0.1` (= `v1.0-paper` + one typo fix).

## v1.1.0 — 2026-11-24

The algorithm of `v1.0-paper` / `v1.0.1` is unchanged; only its implementation was optimised. With the
default options, the cluster and linkage outputs are **byte-identical** to `v1.0.1` on all 20 bundled
datasets. Graph chunking methods were added as options.

### 1. Options: original vs. recommended

| Option | Original `v1.0-paper` (paper / code) | Recommended in `v1.1.0` | Notes |
|---|---|---|---|
| `-nNeighbor` (α) | 10 | 10 | number of neighbours of the WNN graph |
| WNN λ (`l`) | 0.5 | 0.5 | weight between the rank term and the neighbourness term |
| `-nChunk` (β) | auto = 80 + ⌊log₂N⌋ | auto | |
| `-nIteration` (τ) | auto = 2 + ⌊√k⌋ | auto | number of recursive tree divisions |
| `-outlierRatio` (γ) | **0.09** (paper, `IDEA.fit()`, help text); the command-line default was a typo (0.9) | **0.09** | typo fixed in `v1.0.1` |
| `-chunkingMethod` | `metis` (rMETIS: recursive METIS bisection + 5 rounds of silhouette-based shatter / reassignment) | **`metis`** | the new methods have lower mean accuracy (table below) |
| linkage ensemble | auto (11 methods) | auto (11 methods) | average, weighted, single, complete, deltaRev, deltaModularity, minPossibleEdgeCountNormalized, minNodeNormalized, minEdgeCountNormalized, minEdgeWeightNormalized, minModularityWeightNormalized |
| forbidden-pair re-search | 10 rounds | 10 rounds | |
| numba | — | **recommended** | without it the pure-Python path is used (same results, ~2x slower) |

Alternatives for specific cases:
- Distance-matrix input (D01 format): `-chunkingMethod leiden_mod` was best on D01 (ARI 0.823 → 0.872).
  It drops on D02 (sparse distance graph, 0.922 → 0.673), so it is not a general recommendation.
- Speed first: `-chunkingMethod louvain` is the fastest (51 s vs. 60 s in total), with mean ARI 0.015 lower.

### 2. Accuracy (ARI, 20 bundled datasets)

All 20 datasets were run concurrently on one machine (`bench/run_bench.py -j 20`). The first two columns are
the original code. The "metis (default)" column is byte-identical to the `v1.0.1` column.

| Data | `v1.0-paper` as-is (typo 0.9) | `v1.0.1` (paper setting, 0.09) | metis (default) | leiden_cpm | leiden_mod | louvain | spectral | metis_kway |
|---|---|---|---|---|---|---|---|---|
| A01 | 0.993 | 0.993 | 0.993 | 0.993 | 0.989 | 0.989 | 0.968 | 0.982 |
| A02 | 0.000 | 0.938 | 0.938 | 0.931 | 0.938 | 0.936 | 0.910 | 0.909 |
| A03 | 0.000 | 0.969 | 0.969 | 0.965 | 0.965 | 0.968 | 0.943 | 0.926 |
| A04 | 0.000 | 0.927 | 0.927 | 0.915 | 0.917 | 0.911 | 0.902 | 0.906 |
| A05 | 0.015 | 0.703 | 0.703 | 0.686 | 0.682 | 0.688 | 0.712 | 0.666 |
| A06 | 0.000 | 0.605 | 0.605 | 0.605 | 0.594 | 0.597 | 0.581 | 0.576 |
| B01 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.829 | 0.474 |
| B02 | -0.001 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B03 | -0.001 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 0.999 |
| B04 | -0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B05 | 0.002 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B06 | 0.066 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| C01 | -0.001 | 0.970 | 0.970 | 0.973 | 0.972 | 0.972 | 0.957 | 0.890 |
| C02 | -0.000 | 0.866 | 0.866 | 0.866 | 0.866 | 0.866 | 0.859 | 0.858 |
| C03 | -0.000 | 0.898 | 0.898 | 0.894 | 0.904 | 0.902 | 0.662 | 0.774 |
| C04 | 0.205 | 0.948 | 0.948 | 0.949 | 0.948 | 0.948 | 0.716 | 0.947 |
| C05 | 0.002 | 0.946 | 0.946 | 0.945 | 0.945 | 0.944 | 0.878 | 0.858 |
| C06 | 0.002 | 0.941 | 0.941 | 0.941 | 0.938 | 0.939 | 0.850 | 0.940 |
| D01 | 0.012 | 0.823 | 0.823 | 0.815 | 0.872 | 0.832 | 0.821 | 0.626 |
| D02 | 0.011 | 0.922 | 0.922 | 0.626 | 0.673 | 0.651 | 0.748 | 0.656 |
| **mean ARI** | **0.1153** | **0.9223** | **0.9223** | **0.9051** | **0.9100** | **0.9071** | **0.8668** | **0.8493** |
| **mean NMI** | 0.1729 | 0.9349 | 0.9349 | 0.9239 | 0.9295 | 0.9273 | 0.9004 | 0.8878 |
| total time (s) | 272 | 275 | 60 | 60 | 64 | 51 | 111 | 84 |

Summary:
- Running the README example commands with the command-line default of `v1.0-paper` (typo, 0.9) gives a mean
  ARI of 0.115; with the paper setting 0.09 (= default of `v1.0.1`) it is 0.922.
- None of the new chunking methods beats the original rMETIS (`metis`) on average. On the A, B and C datasets
  most differences are within ±0.01; the gap comes mainly from D02, where the silhouette-based shatter /
  reassignment of rMETIS helps on the sparse biological network.
- `spectral` and `metis_kway` drop considerably on non-convex / noisy data (e.g. B01, C03).

### 3. Speed

| | `v1.0.1` | `v1.1.0` | |
|---|---|---|---|
| 20 bundled datasets, total (`metis`) | 275 s | 60 s | 4.6x |
| WNN graph construction (20 datasets + 4 synthetic inputs) | 7.5 s | 2.7 s | 2.8x |
| synthetic 30,000 points (k = 12), one run | 23 s | 6.9 s | 3.3x |

### 4. Changes

**Performance (results unchanged)**
- `hierarchicalClusteringAgent.fit`: each merge updates only the neighbours of the merged pair (adjacency sets)
  instead of scanning every live cluster; per-method update functions are bound once; the merge prefix replayed
  between rounds is cached. Ties are resolved in exactly the original `set` iteration order.
- `IDEA_kernel.py` (optional, numba): the same agglomeration, compiled; used for up to 512 clusters.
  Order-sensitive cases (tied weights, exhausted candidate heap, NaN, division by zero) fall back to the exact
  Python path.
- `chunking`: shattered nodes are reassigned with a heap instead of re-sorting the list after every pick
  (same pick order).
- `WKNNgraph`: the WNN weighting (rank, neighbourness, node-average scaling) is vectorised with numpy. The graph
  is identical to the original, including dict key order and value types (`np.float64`): distance ties follow
  the original `set_edge` iteration order, the square uses Python's `**`, node means use `np.add.reduce`
  (= `np.mean`).
- METIS is called through `pymetis.CSRAdjacency` (the `xadj/adjncy` form is deprecated, removal planned for 2027).

**Features**
- `-chunkingMethod` choices: `metis` (default), `metis_kway`, `leiden_cpm`, `leiden_mod`, `louvain`,
  `spectral`, `average` (`IDEA_chunking.py`).
  - Leiden / Louvain search the resolution so that the number of chunks is close to nChunk (needs python-igraph).
  - Common post-processing: chunks are split into connected components; if there are more than nChunk pieces,
    the smallest piece is merged into the neighbouring piece it shares the most edge weight with; if there are
    fewer, the largest chunk is bisected with METIS.
  - Graphs with several connected components use the components as chunks, as in the original.
- `metis_kway`: on sparse graphs the k-way parts can split into far more connected pieces than nChunk
  (D02: 2,433 pieces for 85 chunks, which made the ensemble step effectively not terminate); the same merge is applied.

**Bug fixes**
- Command-line default of `-outlierRatio`: typo `0.9` → `0.09` (same fix as `v1.0.1`, see below).
- `chunking`, branch `nChunk <= 1`: `for node in enumerate(...)` stored tuples as keys.

**Verification and tools**
- `tests/test_equivalence.py`: randomised comparison with the original code of the WNN graph, the shattered-node
  reassignment, and the chunk- and node-level agglomeration agents (11 linkage methods, with merge prefixes and
  forbidden pairs). Integer weights are used on purpose to create many ties.
- `bench/run_bench.py`: runs the 20 datasets; time, ARI, NMI, byte comparison with reference outputs.
- `bench/compare_methods.py`: builds the comparison table above from several runs.

### Reproduction

```bash
git worktree add ../IDEA_v101 v1.0.1           # paper code (+ typo fix)
python bench/run_bench.py ../IDEA_v101 out_v101 -j 20
python bench/run_bench.py .            out_metis -j 20 --ref out_v101
for m in leiden_cpm leiden_mod louvain spectral metis_kway; do
  python bench/run_bench.py . out_$m -j 20 --extra "-chunkingMethod $m"; done
python bench/compare_methods.py metis=out_metis leiden_cpm=out_leiden_cpm ...
python tests/test_equivalence.py --orig ../IDEA_v101/IDEA.py
```

## v1.0.1 — 2026-11-24 (bug fix, branch `release/1.0.x`)
- Command-line default of `-outlierRatio`: typo `0.9` → `0.09`.
  - The typo had been in the command-line default only, since the first upload (2019-07-31, `8a21d04`).
    The paper (Algorithm 1 and §IV: γ = 0.09), the help text and `IDEA.fit()` all use 0.09, and the paper
    experiments were run with 0.09. **The published results are not affected.**
  - Because of the typo, running the README example commands with `v1.0-paper` as-is marked almost every point
    as an outlier (`-1`).
  - No other code change. With default options the output equals `v1.0-paper` with `-outlierRatio 0.09` on all
    20 bundled datasets.
- The `v1.0-paper` tag is left unchanged (published tags are not moved); its release notes point to `v1.0.1`.

## v1.0-paper — published version (commit `b4964c5`)

Public implementation of IDEA (Ahn et al., IEEE BigData 2021, pp. 2791–2800, doi:10.1109/BigData52589.2021.9671953).
