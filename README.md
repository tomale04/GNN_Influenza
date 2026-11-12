# Forecasting seasonal influenza HA with graph neural networks

Goal: given the Nextstrain HA phylogeny available on April 1 of year *t*, forecast the HA1+HA2 amino-acid
genotype composition of the next season (*t+1*), in the spirit of Huddleston et al. (2020, eLife 9:e60067).
Forecasts only reweight genotypes observed in season *t* (no mutation forecasting) and are scored by the
1-Wasserstein (earth mover's) distance with Hamming ground distance, in amino acids (AA).

**Status:** framework, baselines, regressions and a residual GNN are in place. The best model so far is a
node-feature regression plus a blended, tree-message-passing GNN (7.81 AA vs 8.62 for the naive baseline; see Results).
The GNN's improvement over the regression alone (−0.14 AA) is not statistically resolvable on the 10 test pairs.

---

## 1. Data

### 1.1 Nextstrain trees (`data/nextstrain_trees/`)
- Seasonal-flu HA 12y builds for **H3N2** and **H1N1pdm**, snapshot as of **April 1, 2019–2026** (16 trees).
- Source: the public, versioned S3 bucket `nextstrain-data` (backs data.nextstrain.org). For each year,
  the latest version uploaded on or before April 1 23:59 UTC; `manifest.json` records S3 version IDs, URLs and sha256.
- Pre-2019 snapshots aren't publicly available: S3 versioning starts in May 2018, the Wayback Machine never captured the
  data JSONs, and the other candidate buckets are private.
- Script: `scripts/download_trees.py`.

### 1.2 Root sequences and coordinates
- The root sequence comes from the `*_root-sequence.json` sidecar or the embedded `root_sequence`.
- **2020 repair:** no root-sequence existed on S3 at build time. The stale 2019-08 version was repaired from the tree
  itself: the from-state of the first mutation at each site on every root-to-tip path gives the root state.
  This took 9 nucleotide fixes for H3N2 and 1 for H1N1pdm, with no conflicts.
- Every tree validates with 0 from-state mismatches for every nucleotide and AA mutation.
- HA1/HA2 AA coordinates are identical across all years (H3N2 329/221 aa, H1N1pdm 327/222 aa).
  The common per-subtype reference is the 2024–2026 build reference: `data/mutation_trees/references/`.

### 1.3 Mutation trees (`data/mutation_trees/<subtype>/<year>.json`; `scripts/build_mutation_trees.py`)
- **Nodes** are HA1+HA2 protein genotypes; **edges** carry ≥1 HA1/HA2 AA mutation.
- Nextstrain edges with no HA1/HA2 AA mutation are collapsed: synonymous changes, signal-peptide changes and
  mutations to/from gap (`-`) or `X` are ignored.
- Each node stores its full HA1/HA2 sequence, its mutations from the reference, all of its leaves, and `n_recent`.
  `n_recent` counts leaves collected from Oct 1 of the previous year to the snapshot (the current season).
- Leaf dates come from Nextstrain: the explicit `date` string for 2019 (auspice v1), otherwise `num_date`.

### 1.4 Annotations (`data/annotations/sites.json`)
- Epitope sites, from github.com/nextstrain/seasonal-flu `config/distance_maps`:
  - H3N2: Luksza & Lässig (2014), 49 HA1 sites.
  - H3N2: Koel et al. (2013), 7 sites.
  - H1N1pdm: Caton (1982) antigenic sites, 50 HA1 sites.

---

## 2. Forecasting task and evaluation (`scripts/forecast_framework.py`)

- **Season *t*:** samples collected Oct 1 (*t*−1) to Apr 1 (*t*). The current population is the set of genotypes with
  `n_recent > 0` in tree *t*, with frequency x ∝ `n_recent`.
- **Forecast:** a weight for each season-*t* genotype. Truth is the season-(*t*+1) genotype frequencies from tree *t*+1.
- **Metric:** exact 1-Wasserstein distance (linear program, SciPy HiGHS), with Hamming distance over HA1+HA2 AA as the ground distance.
- **Training surrogate:** map each season-(*t*+1) sample to its nearest season-*t* genotype(s) by Hamming distance,
  splitting ties equally. This gives target weights y. Models are fit with the multinomial cross-entropy −Σ y log p.
  Predicting y exactly attains the Wasserstein lower bound (the "oracle bound").
- **Model form** (Huddleston 2020, Eq. 1): p_v ∝ b_v · exp(f_v), where b is the current frequency x (or a time-discounted share)
  and f is a learned fitness.
- **Test set:** 10 pairs, **H3N2 and H1N1pdm, season *t* → *t*+1 for targets 2022–2026**, always from the actual
  April-1 snapshots (tree *t* → tree *t*+1). Every result below is on these same 10 pairs.
- **Significance:** paired bootstrap (10,000 resamples) over the test pairs.

### 2.1 Training and validation data: historical within-tree pairs (`scripts/hist/`)
For the test pair with target *T*, everything is built from tree *T*−1 (the latest available) only:
- **Historical views** (`hist/views.py`): tree *T*−1 pruned to leaves collected on or before Apr 1 of an origin year *o* ≤ *T*−2.
  - The future season (*o*+1) comes from the same tree's own leaves.
  - LBI is recomputed on the pruned time tree (`scripts/lbi.py`, augur algorithm, τ = 0.5, window = 0.5).
    It reproduces Nextstrain's stored LBI exactly for the 2023–2026 builds; Nextstrain used different settings for 2019–2022.
  - An origin is used only if seasons *o*−1, *o* and *o*+1 each have ≥30 leaves.
  - This gives 17–20 training views per fold across both subtypes.
- **Validation:** the 6 most recent historical origins, each scored by a fit on views with earlier origins only.
- **Leakage:** no input to any view postdates its cutoff. Residual leakage in historical views is limited to tree *T*−1's
  topology, internal-node genotypes and dates, which were inferred with later tips. Test pairs are unaffected.
- Snapshot views reproduce the original pipeline exactly (pairs, targets, features); this was checked in code.

---

## 3. Models

### 3.1 Baselines and regressions (`scripts/hist/pipeline.py`; snapshot-protocol versions in `scripts/run_regression.py`)
All regressions use p ∝ b · exp(θᵀz):
- They're fit by L-BFGS on the convex cross-entropy with an anisotropic L2 penalty: λ_allele on the allele block, λ_feat on dense features.
- They're pooled across subtypes: dense coefficients are shared, allele coefficients are subtype-specific.
- Hyperparameters are chosen per test target by mean validation Wasserstein distance. The forecasts of the top 5 configurations are averaged
  (selection protocol adapted from `mlr_mom_K.pdf`).

| Name | Features z |
|---|---|
| naive | none: p = current frequency |
| lbi | LBI (two variants, see §3.2) |
| mom | momentum: log π(τ_fast) − log π(τ_slow), centred |
| mlr_mom | allele one-hot (site-alleles with frequency ≥ 0.005) + momentum |
| **nodereg** | **the GNN's 12 base node features + momentum + LBI** |
| full | nodereg + allele one-hot |

π(τ) is the exponentially discounted genotype share over a 5-year lookback. The grid covers (τ_fast, τ_slow) and the base b ∈ {x, π(τ_fast)}.

### 3.2 GNN (`scripts/models/gnn.py`, features in `scripts/hist/views.py`)

**Graph.** The mutation tree (pruned view) at the origin. All nodes are included, including ancestral genotypes and genotypes
with no current samples. Each parent→child edge is used in both directions, with separate weights.

**Node features (17, standardised with training statistics).** Windows are relative to the origin:
W0 = Oct 1 (*o*−1) to the cutoff (current season), W1 = the previous Apr–Sep, W2 = the previous Oct–Mar.

| # | Feature | Description |
|---|---|---|
| 1–3 | `freq_W0/W1/W2` | node's own leaf count in the window ÷ window total |
| 4–6 | `logc_W0/W1/W2` | log(1 + own leaf count) in the window |
| 7 | `has_leaves` | node has ≥1 sampled leaf |
| 8–9 | `yrs_since_first/last` | years from the node's first/last sample to the cutoff |
| 10 | `rel_depth` | (# AA mutations from reference − frequency-weighted mean over current genotypes) / 10 |
| 11 | `is_candidate` | has current-season samples (receives a forecast) |
| 12 | `is_root` | root indicator |
| 13–14 | `mom_0.25_1`, `mom_0.5_2` | momentum: log-ratio of discounted shares at (fast, slow) = (0.25, 1) and (0.5, 2) years |
| 15 | `lbi_leaf` | mean LBI over the node's current-season (else all) leaves; z-scored over current genotypes |
| 16 | `lbi_top` | LBI of the Nextstrain node where the genotype arises; z-scored |
| 17 | `has_lookback` | node has samples in the 5-year lookback |

Node features are node-local by design. Clade-level information must come from message passing.

**Edge features.** Each AA mutation on an edge is described by 15 features:

| Group | Features |
|---|---|
| Location | in HA1; at an epitope site (Luksza for H3N2, Caton for H1N1pdm); at a Koel site (H3N2) |
| Chemistry | BLOSUM62 / 4; Δ Kyte–Doolittle hydropathy / 4.5; Δ charge; Δ side-chain volume / 100 |
| Tree context | reversion (new residue already seen at that site on the lineage); log(# other edges with the identical mutation) (homoplasy) |
| Population context | frequency of the new residue and of the old residue at the site in W0, W1 and W2 (6 features) |

Each mutation goes through a 2-layer MLP. Edge embedding = Σ over the edge's mutations + a linear term in log(1 + # mutations).
A learned position/amino-acid embedding variant was also tested; see Results.

**Architecture** (basic edge-conditioned message passing):
```
h0          = MLP(node features)                                   hidden size 16
K = 4 layers:
  m_down(v) = MLP_down,k([h_parent(v), e_edge])                    message from the parent
  m_up(v)   = Σ_children c MLP_up,k([h_c, e_edge])                 messages from the children
  h        <- LayerNorm(h + MLP_k([h, m_down, m_up]))
fitness g_v = MLP_out(h_v)                    last layer zero-initialised
```
- K = 0 is the ablation with no message passing (a per-node MLP on the same features).
- All MLPs are 2-layer with GELU and dropout 0.2.

**Residual + blend.** The GNN is stacked on the regression rather than replacing it:
```
p_reg  = averaged forecast of the regression's top-5 configs
p_gnn  ∝ p_reg · exp(g)                       g = 0 at init  ->  p_gnn = p_reg at epoch 0
p      = (1 − w) · p_reg + w · p_gnn
```
- Training: AdamW (lr 1e-3, weight decay 1e-2), full batch over all training views, cross-entropy against the nearest-neighbour targets.
  Each season pair is weighted equally. The model is an ensemble of 5 seeds, averaging g.
- Selection: (epoch e ∈ {0, 10, …, 150}, w ∈ {0, 0.1, …, 1}) minimises mean validation Wasserstein distance over all validation pairs.
  Ties go to the smaller w. Because e = 0 or w = 0 recovers the regression, the selection can always fall back to it.

---

## 4. Results (10 test pairs, both subtypes, targets 2022–2026; AA)

### 4.1 Current protocol (historical within-tree training/validation; `results/hist_kval6/`)

| Model | Mean | H3N2 | H1N1pdm | vs naive (95% CI) | Wins vs naive |
|---|---|---|---|---|---|
| **nodereg + GNN (K=4), blended** | **7.81** | 8.07 | 7.56 | **−0.80 [−1.33, −0.30]** | 7/10 |
| nodereg + GNN (K=0), blended | 7.87 | 8.04 | 7.69 | −0.75 [−1.37, −0.19] | 7/10 |
| full + GNN (K=4), blended | 7.88 | 7.97 | 7.78 | −0.74 [−1.17, −0.31] | 7/10 |
| nodereg | 7.95 | 8.04 | 7.86 | −0.67 [−1.13, −0.20] | 8/10 |
| full + GNN (K=0), blended | 7.95 | 8.05 | 7.85 | −0.67 [−1.34, −0.11] | 7/10 |
| full | 8.02 | 8.05 | 7.98 | −0.60 [−1.12, −0.12] | 7/10 |
| lbi | 8.22 | 8.05 | 8.39 | −0.40 [−0.64, −0.16] | 8/10 |
| mom | 8.49 | 8.33 | 8.65 | −0.13 [−0.41, +0.23] | 7/10 |
| mlr_mom | 8.59 | 8.30 | 8.87 | −0.03 [−0.34, +0.31] | 6/10 |
| naive | 8.62 | 8.58 | 8.66 | — | — |
| oracle bound (best possible reweighting) | 3.52 | | | | |

**What the GNN adds over its regression (paired):**
- nodereg + K=4: −0.14 [−0.69, +0.42], wins/losses 4/4.
- nodereg + K=0: −0.09 [−0.29, +0.05], 5/5.
- With message passing, validation distance improves by 0.5–0.9 AA over the regression in 4 of 5 folds; without it, by at most 0.15.
  So the tree carries signal on validation, but on test the K=4 gains are high-variance: large wins in the 2025–2026 folds, losses in the 2023–2024 folds.
- The selected epoch is at or near the 150-epoch ceiling in most folds, so the training-length limit is binding.

**Do the regressors help as GNN node inputs?** (`scripts/hist/compare_node_ablation.py`, `scripts/hist/perm_importance.py`)

| GNN node inputs | Val EMD (50 val pairs, w=1 @ 150 epochs) | vs all 17 | GNN vs regression on val | Test |
|---|---|---|---|---|
| all 17 | 5.64 | — | −0.45 [−0.78, −0.13], better in 36/50 | 7.81 |
| 12 basic (no momentum/LBI) | 5.78 | +0.14 [−0.03, +0.34] | −0.31 [−0.63, +0.00] | 7.80 |
| structure only (3 flags) | 5.93 | **+0.30 [+0.11, +0.50]** | −0.15 [−0.40, +0.10] | 8.15 |

- Removing all regressors from the GNN's inputs significantly hurts validation, even though the regression is the offset.
- Permutation importance: current-season abundance (`logc_W0`, `freq_W0`) matters most (the only intervals excluding 0),
  followed by `lbi_top`, `has_leaves` and recency. Momentum contributes about 0.

### 4.2 Earlier experiments (snapshot-only training, 1 → 3 validation seasons)
- **Position×AA linear fitness** (`run_linear_fitness.py`): 8.17 (Wasserstein fit) / 8.26 (cross-entropy fit).
  It gains about 1 AA on H3N2 and nothing on H1N1pdm, and overfits.
- **Plain GNNs** (`run_gnn.py`; `results/gnn`, `results/gnn_v2`): all within noise of naive (8.38–8.79).
  - Ablations on that protocol: learned position/AA embeddings alone did no better than naive;
    biological features ≥ embeddings; pooled > per-subtype.
  - Adding the full sequence as node input (`*_bioseq_*`) didn't help either.
- **Residual / blended GNN on snapshot-only training** (`run_gnn_residual.py`, `run_gnn_blend.py`): K=4 improved
  validation but hurt test (+0.3 to +0.4 AA). This was traced to validation being dominated by COVID-era seasons
  (2020→21, 2021→22), which motivated the historical-pair protocol.
- **ne_star side comparison** (H3N2 only, 4 pairs; `results/nestar_fair_comparison.csv`): Nextstrain's `ne_star` and LBI
  each beat naive by about 0.5 AA in 4/4 pairs. No model significantly beats `ne_star` there.

---

## 5. Caveats
- **Only 10 test pairs.** Model differences smaller than about 0.5 AA aren't resolvable. Treat rankings within the top group as provisional.
- **Nextstrain subsampling:** seasons contain 14–101 genotypes, so the "observed" next-season frequencies are noisy.
- **Historical views** have mild residual leakage (tree topology and ancestral states inferred with later tips) and slightly better
  sampling than a real-time forecast, because later trees include late-submitted sequences.
  This affects training and validation only.
- **The forecast support is restricted to season-*t* genotypes** (the project specification). The `mlr_mom_K` note uses a 5-year
  lookback support instead, which lowers the oracle bound.

## 6. Possible next steps
1. Raise the GNN epoch ceiling (the selected epoch is often at 150).
2. Lineage-additive fitness (f_v = sum of learned per-edge effects along the root path) and soft (distance-kernel) training targets.
3. More test pairs, e.g. historical-view test pairs from earlier trees, to tighten confidence intervals.

---

## 7. Reproducing

```bash
python3 -I scripts/download_trees.py data/nextstrain_trees                     # data (S3 versioned objects)
python3 -I scripts/build_mutation_trees.py data/nextstrain_trees data/mutation_trees
python3 scripts/run_baselines.py                                               # naive / uniform / oracle
cd scripts
python3 -m hist.pipeline naive|lbi|mom|mlr_mom|nodereg|full                    # regressions (historical protocol)
python3 -m hist.pipeline nodereg 4      # + residual blended GNN, K=4 (K=0 for the no-message-passing ablation)
```
Requirements: Python 3.9, numpy, scipy, pandas, torch, biopython. Results are written to `results/`.

## 8. Layout
```
v0/                        previous contents of this repository (exploration notebook, July snapshots, flag reports), kept as-is
data/nextstrain_trees/     raw April-1 Nextstrain JSONs + manifest
data/mutation_trees/       HA1+HA2 AA mutation trees, references, summary
data/annotations/          epitope / Koel / Caton site lists (+ source configs)
literature/                Huddleston 2020, Huddleston & Bedford 2025
scripts/
  download_trees.py, s3_versions.py, auspice_io.py, build_mutation_trees.py   data pipeline
  forecast_framework.py    task, EMD, nearest-neighbour targets, rolling folds
  lbi.py                   LBI (augur algorithm), supports pruning
  hist/views.py            pruned historical tree views, node/edge features, graphs
  hist/pipeline.py         current protocol: regressions + residual blended GNN
  models/gnn.py            GNN architecture;  models/graph_data.py  snapshot graph features
  models/regression.py, models/node_features.py, models/linear_fitness.py
  run_*.py, compare_*.py   earlier experiments and comparison tables
results/                   CSVs (per test pair) and selection logs for every experiment
```
