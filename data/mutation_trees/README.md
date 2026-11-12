# HA1+HA2 amino-acid mutation trees (from Nextstrain April-1 snapshots)

Built by `python3 -I scripts/build_mutation_trees.py data/nextstrain_trees data/mutation_trees`.

## Definition
- Node = an HA1+HA2 protein genotype on the ancestry of the sampled tips. Edge = >=1 HA1/HA2 AA mutation.
- Nextstrain edges with no HA1/HA2 AA mutation (synonymous-only, SigPep-only, or none) are collapsed into the parent.
- Mutations to '-' / 'X' (gaps, ambiguity) are ignored.
- Root node genotype = Nextstrain root sequence plus any mutations on the Nextstrain root node (in 2024+ exports these are reference->root changes).
- `n_recent` = number of Nextstrain leaves mapping to the node with collection date in
  [Oct 1 of previous year, snapshot date]. Dates: the explicit `date` string (2019 v1) or the augur `num_date` point estimate (v2; ambiguous dates use augur's estimate).
- Distinct nodes can share a genotype (convergence/reversion); they are kept separate (`meta.stats.genotypes_repeated_in_tree`).

## Files
- `<subtype>/<year>.json`: `{meta, nodes}`. Each node: `id, parent, children, mutations` (edge, e.g. `HA1:K160T`),
  `HA1, HA2` (full AA sequences), `mutations_from_reference`, `n_leaves, n_recent, recent_leaves, leaves, nextstrain_nodes`.
  Node 0 is the root. Node ids are per-tree; match genotypes across years by sequence.
- `references/<subtype>_HA1_HA2.fasta`: common reference per subtype (the reference sequence of the 2024-2026 builds).
- `summary.json`: per-tree stats.

## Coordinates / reference checks
- HA1/HA2 AA lengths are identical across all years (H3N2 329/221, H1N1pdm 327/222), and root sequences differ by <=6 residues, so AA numbering is shared across years.
  The H3N2 nucleotide reference changed in 2023 (1701 -> 1737 nt, HA1 start 49 -> 66), which doesn't affect the AA coordinates.
- Every tree was validated by checking every nucleotide and AA mutation's from-state against the reconstructed parent sequence (0 mismatches).

## Root-sequence provenance
- 2019, 2021-2024: `*_root-sequence.json` uploaded with the build. 2025-2026: `root_sequence` embedded in the main JSON.
- 2020: no root-sequence existed on S3 at build time (deleted 2019-11-10, re-added 2020-12). Used the 2019-08-09 version,
  repaired from the tree: the root state at each position = from-state of the first mutation on every root-to-tip path
  (no conflicts). H3N2: 9 nuc fixes; H1N1pdm: 1 nuc fix. Unmutated positions agree with both the 2019 and 2021 roots.
