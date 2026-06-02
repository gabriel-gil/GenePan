# GenePan User Manual

## Overview

`GenePan` is a Python tool for identifying tissue-specific only-tumor
(`T-gene`) and only-normal (`N-gene`) expression patterns from cohort-level
FPKM data. For a selected cancer dataset, GenePan discovers gene pools,
constructs compact perfect gene panels, exports binary matrices for downstream
network construction, and can query the classification status of individual
genes or complete gene sets.

GenePan is intended to be used in two main ways:

- Run a complete cohort analysis for one tissue or cancer type.
- Query one gene or a user-supplied gene set against the same discovery rules.

## Credits And Citation

Code authors: Gabriel Gil, Augusto González, and Julio César Drake.

If you use CChains in scientific work, please cite the following manuscript:
G. Gil, C. Carricarte, J. C. Drake-Pérez, Y. Perera, A. Gonzalez. Highly
specific and sensitive gene panels for cancer screening: First application of
only-normal and only-tumor genes. Tumor Discovery 2025, 4(3), 58-69.
https://doi.org/10.36922/TD025190035

## Scientific Pipeline

The standard GenePan workflow proceeds as follows:

1. Read cohort metadata from `sample.xls`.
2. Read the per-sample `*.FPKM.txt` expression files.
3. Separate samples into normal and tumor classes.
4. Evaluate each gene on the raw FPKM expression scale.
5. Identify T-gene and N-gene families from class-exclusive expression
   intervals.
6. Merge family-specific genes into all-T-gene and all-N-gene pools.
7. Build compact perfect panels from the requested gene pools.
8. Export gene pools, panels, summaries, and binary sample matrices.

In the default configuration, GenePan does not apply an expression detection
floor or detection-frequency prefilter before discovery. These filters can be
enabled explicitly when needed.

## Input Data

Choose a tissue folder such as `TCGA-PRAD`. The folder should contain:

- `sample.xls`
  Sample metadata. GenePan uses this file to identify normal and tumor samples.
- many `*.FPKM.txt` files
  Per-sample expression files. Each file contains FPKM values for the measured
  genes in one sample.

The gene annotation file `ensembl.txt` is distributed with GenePan and is used
to map Ensembl identifiers to standard gene names.

The current class split is:

- `Normal`
  Samples whose type is exactly `Solid Tissue Normal`.
- `Tumor`
  Every other sample type in the cohort, including metastatic samples.

## Main Concepts

- `T-gene`
  A gene with tumor-exclusive expression behavior.
- `N-gene`
  A gene with normal-exclusive expression behavior.
- `Only-T-above`, `Only-T-below`, `Only-T-outside`, `Only-T-inside`
  Tumor-facing families.
- `Only-N-above`, `Only-N-below`, `Only-N-outside`, `Only-N-inside`
  Normal-facing families.
- `Gene pool`
  The complete set of genes belonging to one family or merged family group.
- `Perfect gene panel`
  A compact gene panel that covers all target samples in the training cohort.

## Standard Command-Line Modes

Set these paths once in PowerShell:

```powershell
$PY = "C:\Users\gabri\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$ROOT = "C:\Users\gabri\Documents\Codex\2026-04-28\could-you-reach-a-github-private"
$DATA = "C:\Users\gabri\GenePan\TCGA-PRAD"
```

Run a complete cohort analysis:

```powershell
& $PY "$ROOT\genepan.py" $DATA --output-dir "$ROOT\genepan_output"
```

Query one gene by standard gene name, full Ensembl ID, or bare Ensembl ID:

```powershell
& $PY "$ROOT\genepan.py" $DATA --query-gene EPHA10
```

Query a full gene set from a text, CSV, or TSV file:

```powershell
& $PY "$ROOT\genepan.py" $DATA `
  --query-gene-set "$ROOT\my_gene_set.tsv" `
  --output-dir "$ROOT\my_gene_set_query"
```

The first column of the gene-set file is interpreted as a standard gene
symbol/name, full Ensembl ID, or bare Ensembl ID. Matching is case-insensitive.

Compute all gene pools:

```powershell
& $PY "$ROOT\genepan.py" $DATA --compute-all-gene-pools
```

Compute one specific gene pool:

```powershell
& $PY "$ROOT\genepan.py" $DATA --compute-gene-pool Only-T-above
```

Compute one specific perfect panel:

```powershell
& $PY "$ROOT\genepan.py" $DATA --compute-panel Only-T-above
```

## Main Output Files

The files most directly needed to interpret a standard GenePan run are:

- `panels_all.tsv`
  All discovered family-specific panel entries.
- `t_gene_pool.tsv`
  Merged all-T-gene pool.
- `n_gene_pool.tsv`
  Merged all-N-gene pool.
- `selected_t_gene_panel.tsv`
  Perfect panel derived from the merged T-gene pool.
- `selected_n_gene_panel.tsv`
  Perfect panel derived from the merged N-gene pool.
- `summary.json`
  Run summary with gene counts, panel coverage, and parameter values.
- `gene_set_query.tsv`
  Written when `--query-gene-set` is used. It contains one row per
  gene-family membership.

## Binary Matrix Outputs

GenePan writes both gene-level and compressed binary matrices. These matrices
are mainly needed for downstream network construction and for checking the
binary discretization used by GenePan.

The binary meaning is category-specific:

- T-gene matrices
  `1` means that the tumor sample is active in the tumor-exclusive expression
  interval for that gene.
- N-gene matrices
  `0` means that the normal sample is active in the normal-exclusive expression
  interval for that gene, and `1` means that it is not active in that interval.

The CChains-ready files are the strictly necessary binary-matrix outputs for
network construction:

- `T_Network/data/sample.txt`, `T_Network/data/names.csv`
  Compressed T-gene binary matrix and matching bare Ensembl identifiers.
- `N_Network/data/sample.txt`, `N_Network/data/names.csv`
  Compressed N-gene binary matrix and matching bare Ensembl identifiers.

The remaining matrix files are auxiliary outputs for inspection, validation,
or alternative downstream use:

- `sampleT_gene_level.txt`
  Binary matrix for all T-genes across tumor samples.
- `sampleN_gene_level.txt`
  Binary matrix for all N-genes across normal samples.
- `namesT_gene_level.csv`, `namesN_gene_level.csv`
  Gene names in the row order of the gene-level matrices.
- `sampleT_unique_patterns.txt`, `sampleN_unique_patterns.txt`
  Binary matrices after collapsing identical rows.
- `namesT_unique_patterns.csv`, `namesN_unique_patterns.csv`
  Representative gene names in the row order of the unique-pattern matrices.
- `T_gene_order.tsv`, `N_gene_order.tsv`
  Gene identifiers, names, and row sums for the gene-level matrices.
- `T_unique_pattern_order.tsv`, `N_unique_pattern_order.tsv`
  Mapping from each unique-pattern row back to the first source gene row.

## Gene Query Output

Gene query reports include:

- whether the gene is a `T-gene`, `N-gene`, `NT-gene`, or unclassified
- the specific families to which it belongs
- the relevant expression thresholds
- the normal and tumor activation counts
- the normal and tumor activation frequencies
- whether each family membership passes the required frequency criterion

## Optional Modes

Run with explicit parameter values:

```powershell
& $PY "$ROOT\genepan.py" $DATA `
  --low-fpkm 0.6 `
  --detection-floor-fpkm 0.0 `
  --min-detected-fraction-normal 0.0 `
  --min-detected-fraction-tumor 0.0 `
  --pvalue-tumor 0.1 `
  --pvalue-normal 0.05 `
  --split-pvalue 0.01 `
  --split-probability 0.1 `
  --delta-threshold-tumor 1.0 `
  --delta-threshold-normal 1.0 `
  --panel-tie-breaking-priority first
```

Run with stricter expression-detection filtering:

```powershell
& $PY "$ROOT\genepan.py" $DATA `
  --detection-floor-fpkm 0.1 `
  --min-detected-fraction-normal 0.05 `
  --min-detected-fraction-tumor 0.10
```

Run with the alternative perfect-panel tie-breaking priority:

```powershell
& $PY "$ROOT\genepan.py" $DATA `
  --panel-tie-breaking-priority most_redundant_then_first
```

## Stability Analysis Workflows

GenePan can also be used as the discovery engine for downstream stability
analyses under cohort subsampling and synthetic-sample augmentation. These
analyses are connected to GenePan but are intentionally separated from the
core single-cohort run.

The generic runner is:

- `stability_analysis.py`

It uses four cohort-construction keywords:

- `source`
  `real`, `kde`, or `pca`.
- `action`
  `augment`, `subsample`, or `none`.
- `side`
  `normal`, `tumor`, `balanced`, or `none`.
- `counts`
  Explicit comma-separated sample counts, such as `10,20,30,40,50`.

The scientific analysis is selected separately:

- `gene_set_overlap`
  Computes recall, precision, and Jaccard for T-gene and/or N-gene sets.
- `block_structure`
  Measures retained original N-genes, N-gene recall, number of binary-pattern
  blocks, largest block size, and block-size distributions.
- `loevinger_edges`
  Builds the Loevinger edge set for each perturbed cohort and compares it with
  the original full-cohort edge set by recall, precision, and Jaccard. The
  threshold is controlled by `--loevinger-threshold`.

Execution controls are also separate:

- `replicates`
  Number of independent realizations per count.
- `seed`
  Base seed used to derive deterministic per-job seeds.
- `seed-stride`
  Offset between replicate seeds. This is mainly useful when reproducing a
  previous run that used a specific seed convention.
- `subsample-design`
  For real-sample subsampling, `independent` draws each count separately.
  `cumulative` starts from a fixed base subset and progressively adds samples
  from the remaining bank. Cumulative subsampling is useful when the scientific
  question is how stability changes as the same replicate cohort is enlarged.
  The optional `--cumulative-base-seed` and
  `--cumulative-base-replicate-offset` arguments reproduce a previously used
  base-subset schedule.
- `resume`
  Skip already completed replicate/count rows.
- `max-workers`
  Parallel worker count.

Examples:

```powershell
& $PY "$ROOT\stability_analysis.py" `
  --source pca `
  --action augment `
  --side normal `
  --counts 10,20,30,40,50 `
  --analysis block_structure `
  --gene-category N-gene `
  --replicates 10 `
  --seed 7732026 `
  --output-dir "$ROOT\stability_pca_normal_blocks_output" `
  --figure-dir "$ROOT\stability_pca_normal_blocks_figures"
```

```powershell
& $PY "$ROOT\stability_analysis.py" `
  --source real `
  --action subsample `
  --side normal `
  --counts 10,20,30,40,50 `
  --analysis block_structure `
  --gene-category N-gene `
  --replicates 10 `
  --seed 8272026 `
  --output-dir "$ROOT\stability_real_normal_blocks_output" `
  --figure-dir "$ROOT\stability_real_normal_blocks_figures"
```

```powershell
& $PY "$ROOT\stability_analysis.py" `
  --source real `
  --action subsample `
  --side tumor `
  --counts 52 `
  --analysis gene_set_overlap `
  --gene-category both `
  --replicates 100 `
  --seed 9102026 `
  --output-dir "$ROOT\stability_real_tumor_52_output" `
  --figure-dir "$ROOT\stability_real_tumor_52_figures"
```

For a cumulative real-tumor sweep:

```powershell
& $PY "$ROOT\stability_analysis.py" `
  --source real `
  --action subsample `
  --side tumor `
  --subsample-design cumulative `
  --counts 52,104,156,208,260,312,364,416,468 `
  --cumulative-base-count 52 `
  --cumulative-base-seed 9242026 `
  --cumulative-base-replicate-offset 1 `
  --analysis loevinger_edges `
  --gene-category T-gene `
  --loevinger-threshold 0.5 `
  --replicates 10 `
  --seed 10242026 `
  --seed-stride 1 `
  --output-dir "$ROOT\stability_cumulative_tumor_edges_output" `
  --figure-dir "$ROOT\stability_cumulative_tumor_edges_figures"
```

```powershell
& $PY "$ROOT\stability_analysis.py" `
  --source real `
  --action subsample `
  --side normal `
  --counts 10,20,30,40,50 `
  --analysis loevinger_edges `
  --gene-category N-gene `
  --loevinger-threshold 0.5 `
  --replicates 10 `
  --seed 9302026 `
  --output-dir "$ROOT\stability_real_normal_edges_output" `
  --figure-dir "$ROOT\stability_real_normal_edges_figures"
```

The stability workflows are responsible for:

- rerunning GenePan discovery on perturbed cohorts
- comparing T-gene and N-gene sets across realizations
- evaluating block structure in discretized N-gene matrices
- producing publication-ready stability figures

These analyses should be run through their own executable scripts or namelist
configuration. They should record all random seeds so that completed runs can
be reproduced exactly.

Before using stability outputs as final results, rerun them with the same
random seeds after any refactor that changes cohort construction,
discretization, or output aggregation.

## Testing

The test suite uses `pytest` and should be run from the GenePan project root.

If `pytest` is not available in the selected Python environment, install it
first:

```powershell
& $PY -m pip install pytest
```

Then run:

```powershell
& $PY -m pytest
```

In the bundled runtime used during this validation pass, `pytest` was not
installed by default, so tests could not be executed until that dependency is
added.

## Optional Parameter Reference

Most users can keep the default parameters. The most commonly adjusted
parameters are:

- `low_fpkm`
  Low-expression threshold used in the family-classification logic.
- `detection_floor_fpkm`
  Optional preprocessing floor. Values below this level are treated as zero
  before checking whether enough samples express the gene.
- `min_detected_fraction_normal`
  Optional normal-side detection cutoff. A gene is removed only when it is
  below this cutoff and also below the tumor-side cutoff.
- `min_detected_fraction_tumor`
  Optional tumor-side detection cutoff. A gene is removed only when it is
  below this cutoff and also below the normal-side cutoff.
- `pvalue_tumor`
  Minimum tumor fraction required inside a tumor-exclusive interval.
- `pvalue_normal`
  Minimum normal fraction required inside a normal-exclusive interval.
- `split_pvalue`
  Significance threshold for mixed-direction split tests.
- `split_probability`
  Probability parameter for mixed-direction rules.
- `delta_threshold_tumor`
  Minimum inside-interval size for `Only-T-inside`.
- `delta_threshold_normal`
  Minimum inside-interval size for `Only-N-inside`.
- `panel_tie_breaking_priority`
  Perfect-panel tie priority. The default is `first`. The alternative
  `most_redundant_then_first` first ranks candidates by the number of target
  samples covered by each gene.

## Excel Note

If GenePan cannot read `sample.xls`, install an `.xls` reader such as `xlrd`
in the selected Python environment.

## Validation Note

Warning: GenePan outputs are not comparable unless the input cohort and all
analysis parameters are held fixed. Changes in preprocessing, significance
thresholds, binary discretization rules, panel tie-breaking priority, sample
ordering, or gene ordering can change gene pools, panels, and binary matrices.
Always report the parameter set and random seeds used for a run.
