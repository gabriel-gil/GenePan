# GenePan Developer Guide

## Purpose

This guide is for maintaining and extending the current Python implementation.
The user manual explains how to run GenePan; this guide explains where behavior
lives in the code and which internal contracts should not be broken.

Release modules:

- `genepan.py`
  Core scientific engine and command-line entry point.
- `stability_analysis.py`
  Downstream stability-analysis runner.
- `tests/test_genepan.py`
  Current test suite.

No graphical interface is part of the current release.

## Developer Requirements

Use Python 3.10 or newer with:

- `numpy`
- `pandas`
- `matplotlib`
- `xlrd` for legacy `.xls` sample sheets
- `pytest` for running the test suite

Install the usual development dependencies from PowerShell with:

```powershell
& $PY -m pip install numpy pandas matplotlib xlrd pytest
```

## Core Engine Internals

Main classes in `genepan.py`:

- `GenePanParameters`
  Runtime parameters for preprocessing, thresholds, inside intervals, and
  panel tie-breaking.
- `PreparedCohort`
  Prepared expression matrix, sample indices, gene identifiers, and threshold
  metadata used by discovery and query methods.
- `PanelEntry`
  One gene-family membership with thresholds and gene identity.
- `StageResult`
  Family-specific entries and their discovery masks.
- `GeneCategoryScan`
  Category-level active and exported binary matrices.
- `MixSelection`
  Perfect-panel selection result, including selected entries, addition order,
  target support counts, and coverage.
- `GenePanResult`
  Full in-memory output from a complete run.
- `GeneSetResult`
  Labeled output for pool/panel query modes.
- `GenePan`
  Main orchestration class.

## Core Engine Flow By Function

- `_load_tcga_expression_cohort(...)`
  Reads sample metadata, expression files, and gene annotations.
- `_prepare_analysis_cohort(...)`
  Loads or builds a cached `PreparedCohort`.
- `_prepare_analysis_cohort_from_loaded_data(...)`
  Builds a `PreparedCohort` from already loaded matrix data. Stability analyses
  use this heavily for perturbed cohorts.
- `_build_family_stage(...)`
  Dispatches family discovery and returns a `StageResult`.
- `_deduplicated_gene_ids_for_category(...)`
  Returns T-gene or N-gene sets for overlap analyses.
- `build_gene_category_scan(...)`
  Builds active and exported binary matrices for all genes in a category.
- `compute_all_gene_pools(...)`, `compute_specific_gene_pool(...)`,
  `compute_specific_panel(...)`
  Implement command-line pool and panel query modes.
- `query_gene_directly(...)`, `query_gene_set_directly(...)`
  Implement direct gene and gene-set assessment.
- `write_outputs(...)`
  Writes public output tables and CChains-ready binary matrix folders.

## Binary Matrix Internal Contract

`GeneCategoryScan` contains two matrices:

- `active_patterns`
  Biological activity in the class-exclusive interval.
- `binary_patterns`
  Exported matrix encoding used by downstream workflows.

The encoding is category-specific:

- T-gene: `binary_patterns == active_patterns`.
- N-gene: `binary_patterns == 1 - active_patterns`.

This is deliberate. CChains-ready N-gene matrices therefore use `0` for active
normal-exclusive behavior and `1` for non-active behavior. Do not change this
without updating downstream network construction and validation data.

## CChains Export Contract

`_write_gene_category_sample_matrices(...)` writes:

- gene-level inspection matrices
- unique-pattern inspection matrices
- CChains-ready `T_Network/data` and `N_Network/data` folders

The CChains-ready folders must contain full gene-level matrices:

- `sample.txt`
  One binary row per T-gene or N-gene, without collapsing identical patterns.
- `names.csv`
  Bare Ensembl identifiers in exactly the same row order as `sample.txt`.

CChains handles compression of identical binary patterns into block nodes
itself. GenePan may still write unique-pattern inspection files, but those are
not the CChains input contract.

## Stability Runner Internals

`stability_analysis.py` must stay downstream of `genepan.py`. It may call
GenePan methods, but it should not change core GenePan behavior for a specific
robustness experiment.

Configuration dataclasses:

- `CohortConfig`
  `source`, `action`, `side`, `counts`, and balanced-count interpretation.
- `AnalysisConfig`
  Analyzer name, gene category, reference, and Loevinger threshold.
- `ExecutionConfig`
  Replicates, seed, worker count, and resume behavior.
- `OutputConfig`
  Output directories and figure-writing toggle.
- `StabilityConfig`
  Full run configuration.
- `LoadedContext`
  Loaded raw cohort, samplers, sample indices, reference sets/edges, and the
  global gene-index map used for edge comparisons.

Main flow:

1. `_parse_args()` builds `StabilityConfig`.
2. `_load_context()` loads the original cohort and builds reference gene/edge sets.
3. `_build_perturbed_cohort()` constructs one replicate/count cohort.
4. `_run_one_job()` dispatches to the selected analyzer.
5. `_write_tables()` writes raw and summary TSVs.
6. `_seed_report()` records reproducibility metadata.

Current analyzers:

- `_gene_set_overlap_metrics(...)`
  Recall, precision, and Jaccard for T-gene and/or N-gene sets.
- `_block_structure_metrics(...)`
  N-gene recall and binary-pattern block structure.
- `_loevinger_edge_metrics(...)`
  Full edge-set recall, precision, Jaccard, node count, and edge density.

Implementation notes:

- Real subsampling supports `independent` and `cumulative` designs. The
  cumulative design keeps a replicate-specific base subset and adds further
  samples from the remaining bank in a deterministic order.
- Loevinger edge keys must be converted from local matrix row positions to
  global gene-identity keys before intersections are computed. Local row keys
  are only valid inside one scan; they are not comparable across cohorts whose
  discovered gene sets differ.

## Adding Stability Behavior

Add new stability behavior inside `stability_analysis.py` unless there is a
strong reason not to.

For a new analyzer:

1. Add the analyzer name to the CLI choices.
2. Implement a function that accepts `StabilityConfig`, `LoadedContext`, and a
   `PreparedCohort`.
3. Return flat raw metrics and optional distribution rows.
4. Let `_write_tables()` handle raw and summary output.
5. Add a lightweight unit test for the metric formula.

For a new cohort source:

1. Add the source to the CLI choices.
2. Prepare any sampler in `_load_context()`.
3. Add sampling logic to `_sample_synthetic()` or `_build_perturbed_cohort()`.
4. Preserve deterministic seeding.
5. Add a smoke test and a small unit test.

## Testing Strategy

Tests live in `tests/test_genepan.py` and use `pytest`.

Run tests after every source-code change before committing. At minimum, run the
smoke tests for ordinary code edits. Run consistency checks and regression
tests whenever input parsing, output writing, binary matrices, gene ordering,
sample ordering, stability analyses, or numerical thresholds are touched.

The current suite mostly contains lightweight smoke tests and internal
consistency checks. It covers:

- default parameters
- threshold calculations
- family discovery on controlled matrices
- perfect-panel selection
- direct gene and gene-set query behavior
- output serialization
- prepared-cohort caching
- packed Hamming and Loevinger helper consistency
- synthetic sampler behavior
- stability-runner metric formulas and configuration helpers

Recommended test categories:

- `smoke_tests`
  Toy examples used to check that functions are callable, dependencies are
  wired correctly, and the main code paths run. These are programming-facing
  tests and should be fast.
- `consistency_checks`
  Structural checks of inputs and outputs, such as matching row counts between
  `sample.txt` and `names.csv`, expected sample counts, expected gene-order
  tables, and CChains export contracts. These checks do not require old
  validated outputs.
- `regression_tests`
  Artifact-based comparisons against previously validated outputs. A test
  should only be called a regression test if it compares generated output to a
  committed or explicitly supplied reference artifact within a stated
  tolerance. Regression tests may be divided into fast and slow subsets.
- `benchmarks`
  Timing and performance checks. Benchmarks should report runtime and hardware
  context when possible. They should not be mixed with correctness tests unless
  a very broad performance guard is intentionally added.

Suggested commands:

```powershell
# Install test dependency if needed.
& $PY -m pip install pytest

# Run all currently available tests.
& $PY -m pytest

# Run the smoke-test layer after ordinary source edits.
& $PY -m pytest -m smoke_tests

# Run structural contract checks before committing output or cache changes.
& $PY -m pytest -m consistency_checks

# This marker name is reserved for artifact-based tests.
& $PY -m pytest -m regression_tests
```

The repository currently does not include full real-cohort artifact regression
fixtures for every workflow. Until those fixtures are added, full external
validation against completed PRAD or multi-cancer outputs must be run as a
separate validation analysis and documented in the commit or release notes.

## Validation Invariants

When changing code, preserve and report:

- input cohort
- sample order
- gene order
- preprocessing settings
- frequency thresholds
- binary discretization rules
- panel tie-breaking priority
- random seeds and selected sample identities

Changing any of these can alter gene pools, panels, matrices, network edges,
and stability summaries.

## Development Principles

- Keep `genepan.py` as the core scientific engine.
- Keep cohort perturbation and robustness workflows in `stability_analysis.py`.
- Do not change core GenePan behavior to support one downstream experiment.
- Keep raw calculation outputs separate from figure-generation artifacts.
- Preserve deterministic seed reporting for stochastic analyses.
- Add focused tests for new scientific rules, metrics, or sampling behavior.
- Update both manuals when public behavior changes.
