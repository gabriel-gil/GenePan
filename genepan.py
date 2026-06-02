"""GenePan translated from Mathematica into a documented Python workflow.

This module keeps the scientific logic from the original notebook, but wraps it
in a structure that is easier to read, test, and extend:

1. `GenePan` loads the TCGA-style data set.
2. It rebuilds the eight panel families from the notebook.
3. It computes the merged T-gene and N-gene gene pools.
4. It applies greedy perfect-panel selection.

The code intentionally favors explicit variable names and step-by-step helpers
over dense one-liners so that colleagues coming from Mathematica, biology, or
other non-Python backgrounds can follow the translation comfortably.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelEntry:
    """One panel-family rule for a single gene."""

    gene_index: int
    gene_id: str
    gene_name: str
    reference: float
    panel_code: int
    threshold_low: float
    threshold_high: float | None

    @property
    def threshold_kind(self) -> str:
        return "range" if self.threshold_high is not None else "single"

    @property
    def panel_family_name(self) -> str:
        """Return the paper-style label for the panel family."""

        mapping = {
            1: "Only-T-above",
            2: "Only-N-above",
            3: "Only-T-below",
            4: "Only-N-below",
            5: "Only-T-outside",
            6: "Only-N-outside",
            7: "Only-T-inside",
            8: "Only-N-inside",
        }
        return mapping[self.panel_code]

    @property
    def notebook_panel_name(self) -> str:
        """Return the internal family-code identifier."""

        return f"panel{self.panel_code}"


@dataclass(frozen=True)
class StageResult:
    """Output of one panel-building stage.

    `entries` holds the metadata for each accepted gene rule.
    `masks` is the corresponding Boolean matrix with shape:
    `n_entries x n_samples`.
    """

    entries: list[PanelEntry]
    masks: np.ndarray


@dataclass(frozen=True)
class MixSelection:
    """Greedy perfect panel extracted from a broader gene pool.

    In the language of the manuscript, this is the Python counterpart of a
    minimal perfect-panel construction obtained through a formal-concept
    reduct-like procedure.
    """

    selected_positions: list[int]
    selected_entries: list[PanelEntry]
    selected_support_counts: list[int]
    covered_fraction: float
    target_size: int


@dataclass(frozen=True)
class GenePanResult:
    """All relevant results produced by one GenePan run."""

    all_entries: list[PanelEntry]
    tumor_entries: list[PanelEntry]
    normal_entries: list[PanelEntry]
    selected_tumor_mix: MixSelection
    selected_normal_mix: MixSelection
    selected_tumor_mix_most_redundant_then_first: MixSelection
    selected_normal_mix_most_redundant_then_first: MixSelection
    reference: np.ndarray
    log2_values: np.ndarray
    sample_types: list[str]
    gene_ids: list[str]
    gene_names: list[str]
    parameters: GenePanParameters
    t_gene_scan: GeneCategoryScan | None = None
    n_gene_scan: GeneCategoryScan | None = None

    @property
    def t_gene_pool(self) -> list[PanelEntry]:
        return self.tumor_entries

    @property
    def n_gene_pool(self) -> list[PanelEntry]:
        return self.normal_entries

    @property
    def selected_t_gene_panel(self) -> MixSelection:
        return self.selected_tumor_mix

    @property
    def selected_n_gene_panel(self) -> MixSelection:
        return self.selected_normal_mix


@dataclass(frozen=True)
class PreparedCohort:
    """Prepared cohort state shared by direct queries and full runs.

    `log2_values` is kept as a legacy field name for compatibility with the
    older Python translation and its tests. In the current notebook-faithful
    raw-expression workflow it simply carries the same raw analysis matrix as
    `filtered_values`.
    """

    filtered_values: np.ndarray
    reference: np.ndarray
    log2_values: np.ndarray
    limits: np.ndarray
    sample_types: list[str]
    normal_idx: np.ndarray
    tumor_idx: np.ndarray
    gene_ids: list[str]
    gene_names: list[str]
    tumor_threshold: int
    normal_threshold: int


@dataclass(frozen=True)
class GeneQueryMembership:
    """One family-specific characterization returned by a gene query."""

    gene_category: str
    panel_family_name: str
    notebook_panel_name: str
    notebook_panel_code: int
    threshold_kind: str
    threshold_low: float
    threshold_high: float | None
    exclusive_interval_low: float | None
    exclusive_interval_high: float | None
    n_activation_count: int
    n_activation_frequency: float
    t_activation_count: int
    t_activation_frequency: float
    passes_significance: bool


@dataclass(frozen=True)
class GeneQueryResult:
    """Scientific summary for one queried gene."""

    gene_name: str
    gene_id: str | None
    gene_class: str
    memberships: list[GeneQueryMembership]


@dataclass(frozen=True)
class GeneSetResult:
    """A named gene set returned by a pool or panel computation mode."""

    set_kind: str
    gene_category: str
    family_name: str
    entries: list[PanelEntry]

    @property
    def header(self) -> str:
        return f"{len(self.entries)}-gene {self.gene_category} {self.set_kind} ({self.family_name})"


@dataclass(frozen=True)
class SyntheticSimilarityPoint:
    """One augmented-cohort similarity measurement.

    This is used for synthetic-stability experiments where the original cohort
    is enlarged with fictitious tumor and/or normal samples, the T-gene or
    N-gene families are rediscovered, and the resulting gene sets are compared
    against the original cohort through a Jaccard-style similarity index.
    """

    synthetic_tumor_count: int
    synthetic_normal_count: int
    replicate_index: int
    random_seed_used: int
    original_gene_count: int
    augmented_gene_count: int
    intersection_size: int
    union_size: int
    jaccard_similarity: float


@dataclass(frozen=True)
class GeneCategoryScan:
    """Gene-level binary and activity patterns for one cohort and one category.

    `active_patterns` always encodes biological activity in the exclusion
    interval:

    - T-gene: 1 means active in the T-exclusive interval
    - N-gene: 1 means active in the N-exclusive interval

    `binary_patterns` is the user-facing matrix encoding:

    - T-gene: 1 active, 0 inactive
    - N-gene: 1 non-active, 0 active
    """

    gene_category: str
    gene_ids: list[str]
    gene_names: list[str]
    active_patterns: np.ndarray
    binary_patterns: np.ndarray


@dataclass(frozen=True)
class SyntheticProfileSampler:
    """Prepared one-dimensional KDE sampler for one class-by-gene matrix."""

    class_values: np.ndarray
    bandwidth: np.ndarray
    all_zero_mask: np.ndarray
    constant_mask: np.ndarray


@dataclass(frozen=True)
class PCASyntheticProfileSampler:
    """Prepared PCA-Gaussian sampler for one class-by-gene matrix.

    `components` has shape `(k, p)` where `k` is the latent dimension and `p`
    is the number of genes.
    `reference` has shape `(p,)` and stores the fixed normal reference profile.
    `transformed_mean` has shape `(p,)` and stores the mean log-fold profile.
    `latent_mean` and `latent_std` both have shape `(k,)`.
    """

    reference: np.ndarray
    transformed_mean: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    latent_mean: np.ndarray
    latent_std: np.ndarray
    all_zero_mask: np.ndarray
    constant_mask: np.ndarray
    latent_dim: int


@dataclass(frozen=True)
class GenePanParameters:
    """User-facing configuration for the GenePan workflow."""

    low_fpkm: float = 0.6
    interval_margin_fpkm: float = 0.1
    detection_floor_fpkm: float = 0.0
    min_detected_fraction_normal: float = 0.0
    min_detected_fraction_tumor: float = 0.0
    pseudocount_fpkm: float = 0.1
    pvalue_tumor: float = 0.1
    pvalue_normal: float = 0.05
    split_pvalue: float = 0.01
    split_probability: float = 0.1
    delta_threshold_tumor: float = 1.0
    delta_threshold_normal: float = 1.0
    panel_tie_breaking_priority: str = "first"


class GenePan:
    """Main entry point for the GenePan workflow.

    Parameters let users reproduce the standard configuration or explore
    alternatives without editing source code.
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        parameters: GenePanParameters | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.parameters = parameters or GenePanParameters()
        self.cache_dir = Path(__file__).resolve().parent / ".genepan_cache"
        self._prepared_cohort_cache: PreparedCohort | None = None

    def run(self) -> GenePanResult:
        """Execute the full pipeline from raw files to perfect panels."""

        cohort = self._prepare_analysis_cohort()
        stages = self._build_all_family_stages(cohort)

        # Panels 1-4 correspond to one-sided dysregulation patterns in the
        # manuscript: "only-x-above" and "only-x-below".
        #
        # The concept class defines the baseline interval.  The rough class is
        # the class that is allowed to cross that boundary and therefore gives
        # the resulting family its scientific label.  For example,
        # `Only-T-above` means tumor samples rise above the normal baseline.
        panel1 = stages[1]
        panel2 = stages[2]
        panel3 = stages[3]
        panel4 = stages[4]
        panel5 = stages[5]
        panel6 = stages[6]
        panel7 = stages[7]
        panel8 = stages[8]

        tumor_mix_entries = panel1.entries + panel3.entries + panel5.entries + panel7.entries
        normal_mix_entries = panel2.entries + panel4.entries + panel6.entries + panel8.entries
        tumor_mix_masks = self._stack_masks([panel1.masks, panel3.masks, panel5.masks, panel7.masks], cohort.filtered_values.shape[0])
        normal_mix_masks = self._stack_masks([panel2.masks, panel4.masks, panel6.masks, panel8.masks], cohort.filtered_values.shape[0])
        t_gene_scan = self.build_gene_category_scan(cohort=cohort, gene_category="T-gene", stages=stages)
        n_gene_scan = self.build_gene_category_scan(cohort=cohort, gene_category="N-gene", stages=stages)

        # Keep the broader software inventory of all T-gene-derived and
        # N-gene-derived gene pools, even though the current manuscript
        # emphasizes only a subset of them in the main text.
        #
        # The notebook's explicit perfect-panel block works directly on regT
        # rows across tumor samples. The N-gene side is dual: regN rows across
        # normal samples.
        selected_tumor_mix = self._construct_formal_concept_reduct(
            entries=tumor_mix_entries,
            masks=tumor_mix_masks,
            target_idx=cohort.tumor_idx,
            tie_breaking_priority="first",
        )
        selected_tumor_mix_most_redundant_then_first = self._construct_formal_concept_reduct(
            entries=tumor_mix_entries,
            masks=tumor_mix_masks,
            target_idx=cohort.tumor_idx,
            tie_breaking_priority="most_redundant_then_first",
        )
        selected_normal_mix = self._construct_formal_concept_reduct(
            entries=normal_mix_entries,
            masks=normal_mix_masks,
            target_idx=cohort.normal_idx,
            tie_breaking_priority="first",
        )
        selected_normal_mix_most_redundant_then_first = self._construct_formal_concept_reduct(
            entries=normal_mix_entries,
            masks=normal_mix_masks,
            target_idx=cohort.normal_idx,
            tie_breaking_priority="most_redundant_then_first",
        )

        return GenePanResult(
            all_entries=tumor_mix_entries + normal_mix_entries,
            tumor_entries=tumor_mix_entries,
            normal_entries=normal_mix_entries,
            selected_tumor_mix=selected_tumor_mix,
            selected_normal_mix=selected_normal_mix,
            selected_tumor_mix_most_redundant_then_first=selected_tumor_mix_most_redundant_then_first,
            selected_normal_mix_most_redundant_then_first=selected_normal_mix_most_redundant_then_first,
            reference=cohort.reference,
            log2_values=cohort.log2_values,
            sample_types=cohort.sample_types,
            gene_ids=cohort.gene_ids,
            gene_names=cohort.gene_names,
            parameters=self.parameters,
            t_gene_scan=t_gene_scan,
            n_gene_scan=n_gene_scan,
        )

    def _prepare_analysis_cohort(self) -> PreparedCohort:
        """Load, preprocess, and normalize the cohort once.

        The heavy part of repeated GenePan queries is not the family logic. It
        is reading hundreds of text files and rebuilding the dense cohort
        matrix. We therefore keep a two-level cache:

        1. in-memory for repeated calls inside one Python process
        2. binary on disk for fast reuse across later runs
        """

        if self._prepared_cohort_cache is not None:
            return self._prepared_cohort_cache

        cached = self._load_prepared_cohort_from_binary_cache()
        if cached is not None:
            self._prepared_cohort_cache = cached
            return cached

        cohort = self._compute_prepared_analysis_cohort()
        self._prepared_cohort_cache = cohort
        self._write_prepared_cohort_to_binary_cache(cohort)
        return cohort

    def _compute_prepared_analysis_cohort(self) -> PreparedCohort:
        """Compute the prepared cohort from the raw cohort files.

        The current identification of T-gene and N-gene families follows the
        raw-expression logic seen in the notebook, not the older log-fold path
        from the first Python translation. We therefore keep the manuscript's
        optional detection-floor and support filters, but the matrix that
        enters family discovery remains on the raw FPKM scale.
        """

        raw_values, gene_ids, gene_names, sample_types = self._load_tcga_expression_cohort()
        return self._prepare_analysis_cohort_from_loaded_data(
            raw_values,
            gene_ids,
            gene_names,
            sample_types,
        )

    def _prepare_analysis_cohort_from_loaded_data(
        self,
        raw_values: np.ndarray,
        gene_ids: list[str],
        gene_names: list[str],
        sample_types: list[str],
    ) -> PreparedCohort:
        """Prepare a cohort from an already loaded expression matrix.

        This helper is shared by the standard file-backed pipeline and by the
        synthetic-sample experiments, where we start from an in-memory
        augmented raw-expression cohort rather than rereading hundreds of text
        files from disk.
        """

        normal_idx, tumor_idx = self._split_samples(sample_types)
        filtered_values, kept_gene_ids, kept_gene_names = self._apply_manuscript_preprocessing_constraints(
            raw_values,
            gene_ids,
            gene_names,
            normal_idx=normal_idx,
            tumor_idx=tumor_idx,
        )
        reference = np.full(filtered_values.shape[1], np.nan, dtype=np.float64)
        log2_values = filtered_values.copy()
        limits = np.full(filtered_values.shape[1], self.parameters.low_fpkm, dtype=np.float64)
        n_samples = raw_values.shape[0]
        tumor_threshold = self._minimum_significant_support_threshold(
            len(tumor_idx),
            n_samples,
            pvalue=self.parameters.pvalue_tumor,
        )
        normal_threshold = self._minimum_significant_support_threshold(
            len(normal_idx),
            n_samples,
            pvalue=self.parameters.pvalue_normal,
        )
        return PreparedCohort(
            filtered_values=filtered_values,
            reference=reference,
            log2_values=log2_values,
            limits=limits,
            sample_types=sample_types,
            normal_idx=normal_idx,
            tumor_idx=tumor_idx,
            gene_ids=kept_gene_ids,
            gene_names=kept_gene_names,
            tumor_threshold=tumor_threshold,
            normal_threshold=normal_threshold,
        )

    def _cohort_cache_signature(self) -> str:
        """Build a compact signature for the current data files and parameters."""

        parts = [
            "raw-discovery-v2",
            str(self.base_dir.resolve()),
            json.dumps(
                {key: value for key, value in self.parameters.__dict__.items() if key != "panel_tie_breaking_priority"},
                sort_keys=True,
            ),
        ]
        tracked_paths = [self.base_dir / "sample.xls", self.base_dir / "ensembl.txt"]
        tracked_paths.extend(sorted(self.base_dir.glob("*.FPKM.txt")))
        for path in tracked_paths:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def _cohort_cache_path(self) -> Path:
        """Return the binary cache path for the current cohort signature."""

        return self.cache_dir / f"{self._cohort_cache_signature()}.npz"

    def _load_prepared_cohort_from_binary_cache(self) -> PreparedCohort | None:
        """Load a prepared cohort from a binary cache when available."""

        cache_path = self._cohort_cache_path()
        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                return PreparedCohort(
                    filtered_values=cached["filtered_values"],
                    reference=cached["reference"],
                    log2_values=cached["log2_values"],
                    limits=cached["limits"],
                    sample_types=cached["sample_types"].astype(str).tolist(),
                    normal_idx=cached["normal_idx"].astype(np.int64),
                    tumor_idx=cached["tumor_idx"].astype(np.int64),
                    gene_ids=cached["gene_ids"].astype(str).tolist(),
                    gene_names=cached["gene_names"].astype(str).tolist(),
                    tumor_threshold=int(cached["tumor_threshold"][0]),
                    normal_threshold=int(cached["normal_threshold"][0]),
                )
        except (OSError, ValueError, zipfile.BadZipFile):
            return None

    def _write_prepared_cohort_to_binary_cache(self, cohort: PreparedCohort) -> None:
        """Persist the prepared cohort as a binary cache for later reuse."""

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._cohort_cache_path()
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as handle:
                temp_path = Path(handle.name)
            np.savez_compressed(
                temp_path,
                filtered_values=cohort.filtered_values,
                reference=cohort.reference,
                log2_values=cohort.log2_values,
                limits=cohort.limits,
                sample_types=np.array(cohort.sample_types, dtype=str),
                normal_idx=cohort.normal_idx,
                tumor_idx=cohort.tumor_idx,
                gene_ids=np.array(cohort.gene_ids, dtype=str),
                gene_names=np.array(cohort.gene_names, dtype=str),
                tumor_threshold=np.array([cohort.tumor_threshold], dtype=np.int64),
                normal_threshold=np.array([cohort.normal_threshold], dtype=np.int64),
            )
            os.replace(temp_path, cache_path)
        except OSError:
            # Cache persistence is an optimization only. If the filesystem does
            # not allow writes here, we keep the in-memory cache and continue.
            return

    def _build_all_family_stages(self, cohort: PreparedCohort) -> dict[int, StageResult]:
        """Compute the notebook families from one prepared cohort.

        The current classification path follows `carcinogenesis.nb` for
        T-gene and N-gene discovery:

        - one-sided families use raw-expression margins of `Â± 0.1`
        - outside families mean both tails pass simultaneously
        - inside families are not part of that notebook's classification
          stage, so they are empty here
        """

        return {panel_code: self._build_family_stage(cohort=cohort, panel_code=panel_code) for panel_code in range(1, 9)}

    def _load_tcga_expression_cohort(self) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        """Load all expression files and translate gene identifiers.

        The original notebook imported every sample independently and appended
        the second column into a giant list.  Here we preallocate a dense NumPy
        matrix once and fill it row by row, which is much faster and clearer.
        """

        sample_sheet = self._read_sample_sheet(self.base_dir / "sample.xls")
        filenames = sample_sheet.iloc[:, 0].astype(str).tolist()
        sample_types = sample_sheet.iloc[:, 3].astype(str).tolist()

        expression_paths = [self.base_dir / name for name in filenames]
        first = pd.read_csv(expression_paths[0], sep="\t", header=None, names=["gene_id", "value"])
        gene_ids = first["gene_id"].astype(str).tolist()

        matrix = np.empty((len(expression_paths), len(gene_ids)), dtype=np.float64)
        matrix[0, :] = first["value"].to_numpy(dtype=np.float64)
        for idx, path in enumerate(expression_paths[1:], start=1):
            frame = pd.read_csv(path, sep="\t", header=None, names=["gene_id", "value"])
            matrix[idx, :] = frame["value"].to_numpy(dtype=np.float64)

        gene_names = self._translate_gene_names(gene_ids)
        return matrix, gene_ids, gene_names, sample_types

    def compute_t_gene_similarity_with_synthetic_cohort(
        self,
        *,
        synthetic_tumor_counts: list[int],
        synthetic_normal_count: int = 0,
        replicates: int = 1,
        random_seed: int = 0,
    ) -> list[SyntheticSimilarityPoint]:
        """Compare original and augmented T-gene sets through Jaccard overlap."""

        return self.compute_gene_similarity_with_synthetic_cohort(
            gene_category="T-gene",
            synthetic_tumor_counts=synthetic_tumor_counts,
            synthetic_normal_counts=[synthetic_normal_count],
            replicates=replicates,
            random_seed=random_seed,
        )

    def compute_gene_similarity_with_synthetic_cohort(
        self,
        *,
        gene_category: str,
        synthetic_tumor_counts: list[int],
        synthetic_normal_counts: list[int],
        replicates: int = 1,
        random_seed: int = 0,
    ) -> list[SyntheticSimilarityPoint]:
        """Compare original and augmented gene sets through Jaccard overlap.

        The comparison can be performed for either the T-gene or N-gene set.
        Synthetic tumors and normals are generated independently from the
        empirical per-gene class-specific raw-expression distributions.
        """

        if replicates < 1:
            raise RuntimeError("replicates must be at least 1.")
        if any(count < 0 for count in synthetic_tumor_counts):
            raise RuntimeError("synthetic_tumor_counts must be non-negative.")
        if any(count < 0 for count in synthetic_normal_counts):
            raise RuntimeError("synthetic_normal_counts must be non-negative.")

        raw_values, gene_ids, gene_names, sample_types = self._load_tcga_expression_cohort()
        normal_idx, tumor_idx = self._split_samples(sample_types)
        baseline_cohort = self._prepare_analysis_cohort_from_loaded_data(raw_values, gene_ids, gene_names, sample_types)
        original_gene_ids = self._deduplicated_gene_ids_for_category(baseline_cohort, gene_category=gene_category)

        normal_values = raw_values[normal_idx, :]
        tumor_values = raw_values[tumor_idx, :]
        normal_sampler = self._prepare_synthetic_profile_sampler(normal_values)
        tumor_sampler = self._prepare_synthetic_profile_sampler(tumor_values)
        points: list[SyntheticSimilarityPoint] = []
        seed_sequence = np.random.SeedSequence(random_seed)
        child_sequences = seed_sequence.spawn(
            len(synthetic_tumor_counts) * len(synthetic_normal_counts) * replicates
        )
        child_index = 0

        for synthetic_tumor_count in synthetic_tumor_counts:
            for synthetic_normal_count in synthetic_normal_counts:
                for replicate_index in range(replicates):
                    child_seed = int(child_sequences[child_index].generate_state(1, dtype=np.uint64)[0])
                    child_index += 1
                    rng = np.random.default_rng(child_seed)
                    sampled_tumors = self._sample_synthetic_profiles(
                        synthetic_count=synthetic_tumor_count,
                        rng=rng,
                        sampler=tumor_sampler,
                    )
                    sampled_normals = self._sample_synthetic_profiles(
                        synthetic_count=synthetic_normal_count,
                        rng=rng,
                        sampler=normal_sampler,
                    )

                    augmented_blocks = [raw_values]
                    augmented_sample_types = list(sample_types)
                    if sampled_tumors.size:
                        augmented_blocks.append(sampled_tumors)
                        augmented_sample_types.extend(["Synthetic Tumor"] * synthetic_tumor_count)
                    if sampled_normals.size:
                        augmented_blocks.append(sampled_normals)
                        augmented_sample_types.extend(["Solid Tissue Normal"] * synthetic_normal_count)

                    augmented_values = np.vstack(augmented_blocks)
                    augmented_cohort = self._prepare_analysis_cohort_from_loaded_data(
                        augmented_values,
                        gene_ids,
                        gene_names,
                        augmented_sample_types,
                    )
                    augmented_gene_ids = self._deduplicated_gene_ids_for_category(augmented_cohort, gene_category=gene_category)
                    intersection_size = len(original_gene_ids & augmented_gene_ids)
                    union_size = len(original_gene_ids | augmented_gene_ids)
                    points.append(
                        SyntheticSimilarityPoint(
                            synthetic_tumor_count=synthetic_tumor_count,
                            synthetic_normal_count=synthetic_normal_count,
                            replicate_index=replicate_index,
                            random_seed_used=child_seed,
                            original_gene_count=len(original_gene_ids),
                            augmented_gene_count=len(augmented_gene_ids),
                            intersection_size=intersection_size,
                            union_size=union_size,
                            jaccard_similarity=float(intersection_size) / float(union_size) if union_size else 1.0,
                        )
                    )

        return points

    def _deduplicated_gene_ids_for_category(self, cohort: PreparedCohort, *, gene_category: str) -> set[str]:
        """Return the deduplicated T-gene or N-gene ids for one cohort."""

        stages = self._build_all_family_stages(cohort)
        entries = self._entries_for_gene_category_from_stages(stages, gene_category=gene_category)
        return {entry.gene_id for entry in entries}

    def build_gene_category_scan(
        self,
        *,
        cohort: PreparedCohort,
        gene_category: str,
        stages: dict[int, StageResult] | None = None,
        relevant_sample_idx: np.ndarray | None = None,
    ) -> GeneCategoryScan:
        """Build reusable gene-level patterns for one cohort and gene category.

        Genes can belong to more than one family inside a broader T-gene or
        N-gene category. For the category-level scan we therefore collapse
        family memberships by gene id and OR the active exclusion-interval
        patterns across all families belonging to that category.
        """

        if stages is None:
            stages = self._build_all_family_stages(cohort)

        if relevant_sample_idx is None:
            relevant_sample_idx = cohort.tumor_idx if gene_category == "T-gene" else cohort.normal_idx

        sample_count = len(relevant_sample_idx)
        active_by_gene: dict[str, np.ndarray] = {}
        name_by_gene: dict[str, str] = {}

        for panel_code in self._panel_codes_for_gene_category(gene_category):
            stage = stages[panel_code]
            for entry in stage.entries:
                gene_mask = active_by_gene.setdefault(entry.gene_id, np.zeros(sample_count, dtype=bool))
                full_mask = self._exclusive_interval_mask_from_values(
                    entry=entry,
                    values=cohort.filtered_values[:, entry.gene_index],
                    reference=float(cohort.reference[entry.gene_index]),
                    normal_idx=cohort.normal_idx,
                    tumor_idx=cohort.tumor_idx,
                )
                gene_mask |= full_mask[relevant_sample_idx]
                name_by_gene.setdefault(entry.gene_id, entry.gene_name)

        sorted_gene_ids = sorted(active_by_gene)
        gene_names = [name_by_gene[gene_id] for gene_id in sorted_gene_ids]
        if sorted_gene_ids:
            active_patterns = np.vstack([active_by_gene[gene_id] for gene_id in sorted_gene_ids]).astype(np.uint8, copy=False)
        else:
            active_patterns = np.zeros((0, sample_count), dtype=np.uint8)

        if gene_category == "T-gene":
            binary_patterns = active_patterns.copy()
        elif gene_category == "N-gene":
            binary_patterns = (1 - active_patterns).astype(np.uint8, copy=False)
        else:
            raise RuntimeError("gene_category must be 'T-gene' or 'N-gene'.")

        return GeneCategoryScan(
            gene_category=gene_category,
            gene_ids=sorted_gene_ids,
            gene_names=gene_names,
            active_patterns=active_patterns,
            binary_patterns=binary_patterns,
        )

    def _entries_for_gene_category_from_stages(
        self,
        stages: dict[int, StageResult],
        *,
        gene_category: str,
    ) -> list[PanelEntry]:
        """Collect all stage entries belonging to one broader gene category."""

        entries: list[PanelEntry] = []
        for panel_code in self._panel_codes_for_gene_category(gene_category):
            entries.extend(stages[panel_code].entries)
        return entries

    @staticmethod
    def _panel_codes_for_gene_category(gene_category: str) -> list[int]:
        """Return the notebook family codes belonging to one broader category."""

        if gene_category == "T-gene":
            return [1, 3, 5, 7]
        if gene_category == "N-gene":
            return [2, 4, 6, 8]
        raise RuntimeError("gene_category must be 'T-gene' or 'N-gene'.")

    @staticmethod
    def _sample_synthetic_profiles(
        *,
        synthetic_count: int,
        rng: np.random.Generator,
        class_values: np.ndarray | None = None,
        sampler: SyntheticProfileSampler | None = None,
        pca_sampler: PCASyntheticProfileSampler | None = None,
    ) -> np.ndarray:
        """Sample synthetic class-specific profiles from per-gene KDEs.

        This builds one smooth one-dimensional distribution per gene and class:

        - draw synthetic values from that distribution
        - keep genes independent across the synthetic profile

        A synthetic value is generated by drawing one observed class-side value
        and then perturbing it with Gaussian noise whose bandwidth is estimated
        from that gene's class-side sample.
        """

        if pca_sampler is not None:
            if sampler is not None or class_values is not None:
                raise ValueError("Supply only pca_sampler, or KDE sampler/class_values, not both.")
            return GenePan._sample_pca_profiles(
                synthetic_count=synthetic_count,
                rng=rng,
                sampler=pca_sampler,
            )

        if sampler is None:
            if class_values is None:
                raise ValueError("Either class_values or sampler must be supplied.")
            sampler = GenePan._prepare_synthetic_profile_sampler(class_values)
        elif class_values is not None:
            raise ValueError("Supply either class_values or sampler, not both.")

        if synthetic_count == 0:
            return np.zeros((0, sampler.class_values.shape[1]), dtype=np.float64)

        class_values = sampler.class_values
        n_samples, n_genes = class_values.shape
        sampled_indices = rng.integers(0, n_samples, size=(synthetic_count, n_genes))
        gene_positions = np.broadcast_to(np.arange(n_genes, dtype=np.int64), (synthetic_count, n_genes))
        sampled_values = class_values[sampled_indices, gene_positions].astype(np.float64, copy=True)
        noise = rng.normal(loc=0.0, scale=1.0, size=(synthetic_count, n_genes))
        sampled_values += noise * sampler.bandwidth[np.newaxis, :]

        if np.any(sampler.constant_mask):
            sampled_values[:, sampler.constant_mask] = class_values[0, sampler.constant_mask]
        if np.any(sampler.all_zero_mask):
            sampled_values[:, sampler.all_zero_mask] = 0.0
        return sampled_values

    @staticmethod
    def _sample_pca_profiles(
        *,
        synthetic_count: int,
        rng: np.random.Generator,
        sampler: PCASyntheticProfileSampler,
    ) -> np.ndarray:
        """Sample synthetic profiles from a PCA-Gaussian latent model."""

        n_genes = sampler.mean.shape[0]
        if synthetic_count == 0:
            return np.zeros((0, n_genes), dtype=np.float64)

        latent_dim = sampler.latent_dim
        latent_noise = rng.normal(
            loc=sampler.latent_mean[np.newaxis, :],
            scale=sampler.latent_std[np.newaxis, :],
            size=(synthetic_count, latent_dim),
        )
        transformed = sampler.transformed_mean[np.newaxis, :] + latent_noise @ sampler.components
        sampled_values = sampler.reference[np.newaxis, :] * np.exp(transformed) - 0.01
        sampled_values = np.maximum(sampled_values, 0.0)
        if np.any(sampler.constant_mask):
            sampled_values[:, sampler.constant_mask] = sampler.mean[sampler.constant_mask]
        if np.any(sampler.all_zero_mask):
            sampled_values[:, sampler.all_zero_mask] = 0.0
        return sampled_values.astype(np.float64, copy=False)

    @staticmethod
    def _prepare_synthetic_profile_sampler(class_values: np.ndarray) -> SyntheticProfileSampler:
        """Prepare the cached KDE bandwidth machinery for one class matrix."""

        class_values = class_values.astype(np.float64, copy=False)
        n_samples = class_values.shape[0]
        n_genes = class_values.shape[1]
        std = np.std(class_values, axis=0, ddof=1) if n_samples > 1 else np.zeros(n_genes, dtype=np.float64)
        q75 = np.percentile(class_values, 75, axis=0)
        q25 = np.percentile(class_values, 25, axis=0)
        iqr_sigma = (q75 - q25) / 1.34
        scale = np.minimum(std, iqr_sigma)
        scale = np.where(np.isfinite(scale), scale, 0.0)
        scale = np.where(scale > 0.0, scale, std)
        bandwidth = 0.9 * scale * (float(n_samples) ** (-0.2)) if n_samples > 1 else np.zeros(n_genes, dtype=np.float64)
        all_zero_mask = np.all(class_values == 0.0, axis=0)
        constant_mask = np.all(class_values == class_values[[0], :], axis=0)
        return SyntheticProfileSampler(
            class_values=class_values,
            bandwidth=bandwidth,
            all_zero_mask=all_zero_mask,
            constant_mask=constant_mask,
        )

    @staticmethod
    def _prepare_pca_profile_sampler(
        class_values: np.ndarray,
        *,
        reference_values: np.ndarray,
        latent_dim: int = 10,
    ) -> PCASyntheticProfileSampler:
        """Prepare a low-rank PCA-Gaussian sampler for one class matrix.

        Mathematical model:

        - X in R^{n x p} is the class-specific expression matrix.
        - r in R^p is the fixed normal reference profile.
        - Y = log(X + 0.01) - log(r) is the log-fold matrix.
        - Y_c = Y - mu_y is centered data.
        - PCA yields components V_k in R^{k x p}.
        - Z = Y_c V_k^T in R^{n x k} are latent coordinates.
        - We fit a diagonal Gaussian on Z: z_j ~ N(m_j, s_j^2).
        - New samples are reconstructed by:
          y_new = mu_y + z_new V_k
          x_new = r * exp(y_new) - 0.01.
        """

        class_values = class_values.astype(np.float64, copy=False)
        reference_values = reference_values.astype(np.float64, copy=False)
        n_samples, n_genes = class_values.shape
        latent_dim = int(min(max(latent_dim, 1), n_samples, n_genes))
        reference = np.exp(np.mean(np.log(reference_values + 0.01), axis=0))
        log_fold = np.log(class_values + 0.01) - np.log(reference)[np.newaxis, :]
        transformed_mean = np.mean(log_fold, axis=0)
        centered = log_fold - transformed_mean[np.newaxis, :]

        # SVD is numerically stable and equivalent to PCA on centered data.
        # centered = U S Vt, so principal directions are rows of Vt.
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:latent_dim, :].astype(np.float64, copy=False)
        latent = centered @ components.T
        if latent.shape[0] > 1:
            latent_mean = np.mean(latent, axis=0)
            latent_std = np.std(latent, axis=0, ddof=1)
        else:
            latent_mean = np.zeros(latent_dim, dtype=np.float64)
            latent_std = np.ones(latent_dim, dtype=np.float64)
        latent_std = np.where(np.isfinite(latent_std) & (latent_std > 0.0), latent_std, 1.0)

        all_zero_mask = np.all(class_values == 0.0, axis=0)
        constant_mask = np.all(class_values == class_values[[0], :], axis=0)
        return PCASyntheticProfileSampler(
            reference=reference,
            transformed_mean=transformed_mean,
            mean=np.mean(class_values, axis=0),
            components=components,
            latent_mean=latent_mean,
            latent_std=latent_std,
            all_zero_mask=all_zero_mask,
            constant_mask=constant_mask,
            latent_dim=latent_dim,
        )

    def _apply_manuscript_preprocessing_constraints(
        self,
        raw_values: np.ndarray,
        gene_ids: list[str],
        gene_names: list[str],
        *,
        normal_idx: np.ndarray,
        tumor_idx: np.ndarray,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Apply the manuscript-inspired preprocessing defaults.

        The paper introduces stronger default constraints than the old notebook:

        1. Expression values below a small detection floor are treated as zero.
        2. Genes detected below both the normal and tumor support cutoffs are
           removed from the analysis.

        We keep those as defaults while preserving user control through the
        parameter model.
        """

        floored = raw_values.copy()
        floored[floored < self.parameters.detection_floor_fpkm] = 0.0

        normal_detected = np.mean(floored[normal_idx, :] > 0.0, axis=0)
        tumor_detected = np.mean(floored[tumor_idx, :] > 0.0, axis=0)

        keep_mask = ~(
            (normal_detected < self.parameters.min_detected_fraction_normal)
            & (tumor_detected < self.parameters.min_detected_fraction_tumor)
        )

        kept_values = floored[:, keep_mask]
        kept_gene_ids = [gene_id for gene_id, keep in zip(gene_ids, keep_mask, strict=False) if keep]
        kept_gene_names = [gene_name for gene_name, keep in zip(gene_names, keep_mask, strict=False) if keep]
        return kept_values, kept_gene_ids, kept_gene_names

    def _read_sample_sheet(self, path: Path) -> pd.DataFrame:
        """Read `sample.xls`.

        The repository uses legacy `.xls` format.  We first try the standard
        pandas path.  If the required Excel engine is missing, we provide a
        Windows-only fallback that asks Excel itself to export the first sheet
        to CSV.
        """

        errors: list[str] = []
        try:
            return pd.read_excel(path, header=None).iloc[1:, :].reset_index(drop=True)
        except Exception as exc:
            errors.append(f"pandas.read_excel failed: {exc}")

        if os.name == "nt":
            try:
                return self._read_sample_sheet_via_excel(path)
            except Exception as exc:
                errors.append(f"Windows Excel fallback failed: {exc}")

        message = (
            f"Unable to read legacy Excel file {path}. "
            f"Install an .xls-capable engine such as xlrd, or run on Windows with Excel available. "
            f"Details: {' | '.join(errors)}"
        )
        raise RuntimeError(message)

    def _read_sample_sheet_via_excel(self, path: Path) -> pd.DataFrame:
        """Fallback reader for legacy `.xls` files on Windows.

        This is intentionally isolated in one helper so the scientific code
        stays separate from platform-specific file conversion details.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            script = f"""
$ErrorActionPreference = 'Stop'
$path = '{str(path).replace("'", "''")}'
$out = '{str(csv_path).replace("'", "''")}'
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
try {{
    $workbook = $excel.Workbooks.Open($path)
    $worksheet = $workbook.Worksheets.Item(1)
    $worksheet.SaveAs($out, 6)
    $workbook.Close($false)
}} finally {{
    if ($workbook) {{ [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($workbook) }}
    if ($worksheet) {{ [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($worksheet) }}
    $excel.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
}}
"""
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
                text=True,
            )
            return pd.read_csv(csv_path, header=None).iloc[1:, :].reset_index(drop=True)

    def _translate_gene_names(self, gene_ids: Iterable[str]) -> list[str]:
        """Map Ensembl identifiers to human-readable gene names."""

        translation_path = self.base_dir / "ensembl.txt"
        translation = pd.read_csv(translation_path, sep="\t")
        lookup = dict(zip(translation.iloc[:, 0].astype(str), translation.iloc[:, 1].fillna("-").astype(str)))

        names: list[str] = []
        for gene_id in gene_ids:
            bare = gene_id.split(".", 1)[0]
            name = lookup.get(bare, "-")
            names.append(name if name != "-" else bare)
        return names

    def _split_samples(self, sample_types: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return zero-based normal and tumor sample indices."""

        # The notebook uses a binary distinction only:
        # `Solid Tissue Normal` versus everything else.
        # That means metastatic samples belong to the tumor side as well.
        return self._split_sample_types_static(sample_types)

    @staticmethod
    def _split_sample_types_static(sample_types: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return zero-based normal and tumor sample indices."""

        normal_idx = np.array([i for i, sample in enumerate(sample_types) if sample == "Solid Tissue Normal"], dtype=np.int64)
        tumor_idx = np.array([i for i, sample in enumerate(sample_types) if sample != "Solid Tissue Normal"], dtype=np.int64)
        if len(normal_idx) == 0 or len(tumor_idx) == 0:
            raise RuntimeError("Expected both normal and tumor samples in sample metadata.")
        return normal_idx, tumor_idx

    def _panel_family_name(self, panel_code: int) -> str:
        """Return the paper-facing family label for one notebook panel code."""

        return {
            1: "Only-T-above",
            2: "Only-N-above",
            3: "Only-T-below",
            4: "Only-N-below",
            5: "Only-T-outside",
            6: "Only-N-outside",
            7: "Only-T-inside",
            8: "Only-N-inside",
        }[panel_code]

    def _panel_code_from_family_name(self, family_name: str) -> int:
        """Translate a paper-facing family label back to the notebook code."""

        mapping = {self._panel_family_name(panel_code): panel_code for panel_code in range(1, 9)}
        try:
            return mapping[family_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Unknown family {family_name!r}. Expected one of: {', '.join(mapping)} or All-T/All-N."
            ) from exc

    def _build_family_stage(self, *, cohort: PreparedCohort, panel_code: int) -> StageResult:
        """Build exactly one family from the prepared cohort."""

        if panel_code in {1, 2, 3, 4, 5, 6}:
            return self._build_family_stage_fast(cohort=cohort, panel_code=panel_code)
        return self._build_family_stage_for_single_gene(
            cohort=cohort,
            panel_code=panel_code,
            values=cohort.filtered_values,
            limits=cohort.limits,
            gene_ids=cohort.gene_ids,
            gene_names=cohort.gene_names,
            reference=cohort.reference,
        )

    def _build_family_stage_fast(self, *, cohort: PreparedCohort, panel_code: int) -> StageResult:
        """Fast whole-cohort family builder for the current carcinogenesis rules.

        The optimized path uses a nested strategy:

        1. a cheap first/last prescreen derived from class extrema
        2. direct threshold-aware discovery based on the `(threshold + 1)`-th
           rough-class order statistic

        For the current one-sided and outside families, that threshold-aware
        rule is the actual membership rule, so no extra confirmation pass is
        needed after the nested screen.
        """

        values = cohort.filtered_values
        normal_values = values[cohort.normal_idx, :]
        tumor_values = values[cohort.tumor_idx, :]
        margin = self.parameters.interval_margin_fpkm
        normal_max = normal_values.max(axis=0)
        tumor_max = tumor_values.max(axis=0)
        normal_min = normal_values.min(axis=0)
        tumor_min = tumor_values.min(axis=0)

        tumor_k = cohort.tumor_threshold + 1
        normal_k = cohort.normal_threshold + 1
        kth_tumor_largest = np.partition(tumor_values, tumor_values.shape[0] - tumor_k, axis=0)[tumor_values.shape[0] - tumor_k, :]
        kth_tumor_smallest = np.partition(tumor_values, tumor_k - 1, axis=0)[tumor_k - 1, :]
        kth_normal_largest = np.partition(normal_values, normal_values.shape[0] - normal_k, axis=0)[normal_values.shape[0] - normal_k, :]
        kth_normal_smallest = np.partition(normal_values, normal_k - 1, axis=0)[normal_k - 1, :]

        if panel_code == 1:
            qualifying = (tumor_max > (normal_max + margin)) & (kth_tumor_largest > (normal_max + margin))
            thresholds_low = normal_max + margin
            thresholds_high = None
            masks = values[:, qualifying].T > thresholds_low[qualifying, None]
        elif panel_code == 2:
            qualifying = (normal_max > (tumor_max + margin)) & (kth_normal_largest > (tumor_max + margin))
            thresholds_low = tumor_max + margin
            thresholds_high = None
            masks = values[:, qualifying].T > thresholds_low[qualifying, None]
        elif panel_code == 3:
            qualifying = (tumor_min < (normal_min - margin)) & (kth_tumor_smallest < (normal_min - margin))
            thresholds_low = normal_min - margin
            thresholds_high = None
            masks = values[:, qualifying].T < thresholds_low[qualifying, None]
        elif panel_code == 4:
            qualifying = (normal_min < (tumor_min - margin)) & (kth_normal_smallest < (tumor_min - margin))
            thresholds_low = tumor_min - margin
            thresholds_high = None
            masks = values[:, qualifying].T < thresholds_low[qualifying, None]
        elif panel_code == 5:
            qualifying = (
                (tumor_max > (normal_max + margin))
                & (tumor_min < (normal_min - margin))
                & (kth_tumor_largest > (normal_max + margin))
                & (kth_tumor_smallest < (normal_min - margin))
            )
            thresholds_low = normal_min - margin
            thresholds_high = normal_max + margin
            masks = (values[:, qualifying].T < thresholds_low[qualifying, None]) | (
                values[:, qualifying].T > thresholds_high[qualifying, None]
            )
        elif panel_code == 6:
            qualifying = (
                (normal_max > (tumor_max + margin))
                & (normal_min < (tumor_min - margin))
                & (kth_normal_largest > (tumor_max + margin))
                & (kth_normal_smallest < (tumor_min - margin))
            )
            thresholds_low = tumor_min - margin
            thresholds_high = tumor_max + margin
            masks = (values[:, qualifying].T < thresholds_low[qualifying, None]) | (
                values[:, qualifying].T > thresholds_high[qualifying, None]
            )
        else:
            raise RuntimeError(f"Fast family builder does not support panel code {panel_code}.")

        gene_positions = np.flatnonzero(qualifying)
        entries = [
            PanelEntry(
                gene_index=int(gene_index),
                gene_id=cohort.gene_ids[gene_index],
                gene_name=cohort.gene_names[gene_index],
                reference=float(cohort.reference[gene_index]),
                panel_code=panel_code,
                threshold_low=float(thresholds_low[gene_index]),
                threshold_high=None if thresholds_high is None else float(thresholds_high[gene_index]),
            )
            for gene_index in gene_positions.tolist()
        ]
        final_masks = self._final_activation_masks_for_entries(cohort=cohort, entries=entries)
        return StageResult(entries=entries, masks=final_masks)

    def _build_family_stage_for_single_gene(
        self,
        *,
        cohort: PreparedCohort,
        panel_code: int,
        values: np.ndarray,
        limits: np.ndarray,
        gene_ids: list[str],
        gene_names: list[str],
        reference: np.ndarray,
    ) -> StageResult:
        """Build one panel family for either a whole cohort or one selected gene."""

        if panel_code == 1:
            return self._discover_one_sided_dysregulation_panel(
                values,
                limits,
                concept_idx=cohort.normal_idx,
                rough_idx=cohort.tumor_idx,
                threshold_count=cohort.tumor_threshold,
                gene_ids=gene_ids,
                gene_names=gene_names,
                reference=reference,
                panel_code=1,
                direction="up",
            )
        if panel_code == 2:
            return self._discover_one_sided_dysregulation_panel(
                values,
                limits,
                concept_idx=cohort.tumor_idx,
                rough_idx=cohort.normal_idx,
                threshold_count=cohort.normal_threshold,
                gene_ids=gene_ids,
                gene_names=gene_names,
                reference=reference,
                panel_code=2,
                direction="up",
            )
        if panel_code == 3:
            return self._discover_one_sided_dysregulation_panel(
                values,
                limits,
                concept_idx=cohort.normal_idx,
                rough_idx=cohort.tumor_idx,
                threshold_count=cohort.tumor_threshold,
                gene_ids=gene_ids,
                gene_names=gene_names,
                reference=reference,
                panel_code=3,
                direction="down",
            )
        if panel_code == 4:
            return self._discover_one_sided_dysregulation_panel(
                values,
                limits,
                concept_idx=cohort.tumor_idx,
                rough_idx=cohort.normal_idx,
                threshold_count=cohort.normal_threshold,
                gene_ids=gene_ids,
                gene_names=gene_names,
                reference=reference,
                panel_code=4,
                direction="down",
            )
        if panel_code == 5:
            return self._discover_two_tailed_dysregulation_panel(
                values,
                limits,
                concept_idx=cohort.normal_idx,
                rough_idx=cohort.tumor_idx,
                threshold_count=cohort.tumor_threshold,
                gene_ids=gene_ids,
                gene_names=gene_names,
                reference=reference,
                panel_code=5,
            )
        if panel_code == 6:
            return self._discover_two_tailed_dysregulation_panel(
                values,
                limits,
                concept_idx=cohort.tumor_idx,
                rough_idx=cohort.normal_idx,
                threshold_count=cohort.normal_threshold,
                gene_ids=gene_ids,
                gene_names=gene_names,
                reference=reference,
                panel_code=6,
            )
        if panel_code == 7:
            return self._discover_inside_dysregulation_panel(
                values,
                limits,
                concept_idx=cohort.normal_idx,
                rough_idx=cohort.tumor_idx,
                threshold_count=cohort.tumor_threshold,
                gene_ids=gene_ids,
                gene_names=gene_names,
                reference=reference,
                panel_code=7,
                delta_threshold=self.parameters.delta_threshold_tumor,
            )
        if panel_code == 8:
            return self._discover_inside_dysregulation_panel(
                values,
                limits,
                concept_idx=cohort.tumor_idx,
                rough_idx=cohort.normal_idx,
                threshold_count=cohort.normal_threshold,
                gene_ids=gene_ids,
                gene_names=gene_names,
                reference=reference,
                panel_code=8,
                delta_threshold=self.parameters.delta_threshold_normal,
            )
        raise RuntimeError(f"Unsupported notebook panel code: {panel_code}")

    def _discover_one_sided_dysregulation_panel(
        self,
        values: np.ndarray,
        limits: np.ndarray,
        *,
        concept_idx: np.ndarray,
        rough_idx: np.ndarray,
        threshold_count: int,
        gene_ids: list[str],
        gene_names: list[str],
        reference: np.ndarray,
        panel_code: int,
        direction: str,
    ) -> StageResult:
        """Build panels 1-4 from the notebook.

        The `concept_idx` samples define the baseline interval.  The
        `rough_idx` samples are the ones tested for dysregulation outside that
        interval, so the rough class determines whether the resulting gene pool
        is T-oriented or N-oriented.

        `direction="up"` reproduces the `ta` / `na` rule from
        `carcinogenesis.nb`.
        `direction="down"` reproduces the `tb` / `nb` rule from
        `carcinogenesis.nb`.
        """

        entries: list[PanelEntry] = []
        masks: list[np.ndarray] = []
        n_genes = values.shape[1]

        for gene_index in range(n_genes):
            concept_values = values[concept_idx, gene_index]
            rough_values = values[rough_idx, gene_index]
            margin = self.parameters.interval_margin_fpkm

            if direction == "up":
                threshold = float(np.max(concept_values)) + margin
                aberrations = rough_values[rough_values > threshold]
                if len(aberrations) <= threshold_count:
                    continue
                mask = values[:, gene_index] > threshold
            else:
                threshold = float(np.min(concept_values)) - margin
                aberrations = rough_values[rough_values < threshold]
                if len(aberrations) <= threshold_count:
                    continue
                mask = values[:, gene_index] < threshold

            entries.append(
                PanelEntry(
                    gene_index=gene_index,
                    gene_id=gene_ids[gene_index],
                    gene_name=gene_names[gene_index],
                    reference=float(reference[gene_index]),
                    panel_code=panel_code,
                    threshold_low=threshold,
                    threshold_high=None,
                )
            )
            masks.append(mask)

        final_masks = self._final_activation_masks_from_values(values=values, entries=entries)
        return StageResult(entries=entries, masks=final_masks)

    def _discover_two_tailed_dysregulation_panel(
        self,
        values: np.ndarray,
        limits: np.ndarray,
        *,
        concept_idx: np.ndarray,
        rough_idx: np.ndarray,
        threshold_count: int,
        gene_ids: list[str],
        gene_names: list[str],
        reference: np.ndarray,
        panel_code: int,
    ) -> StageResult:
        """Build panels 5-6.

        These panels reproduce the `to` / `no` logic from
        `carcinogenesis.nb`: both the lower and upper opposite-class tails
        must individually exceed the required class fraction beyond a raw
        margin of `Â± 0.1` around the concept range.
        """

        entries: list[PanelEntry] = []
        masks: list[np.ndarray] = []
        n_genes = values.shape[1]

        for gene_index in range(n_genes):
            concept_values = values[concept_idx, gene_index]
            rough_values = values[rough_idx, gene_index]
            margin = self.parameters.interval_margin_fpkm
            threshold_down = float(np.min(concept_values)) - margin
            threshold_up = float(np.max(concept_values)) + margin

            abunder = rough_values[rough_values < threshold_down]
            abover = rough_values[rough_values > threshold_up]
            if len(abunder) <= threshold_count or len(abover) <= threshold_count:
                continue

            mask = (values[:, gene_index] < threshold_down) | (values[:, gene_index] > threshold_up)

            entries.append(
                PanelEntry(
                    gene_index=gene_index,
                    gene_id=gene_ids[gene_index],
                    gene_name=gene_names[gene_index],
                    reference=float(reference[gene_index]),
                    panel_code=panel_code,
                    threshold_low=threshold_down,
                    threshold_high=threshold_up,
                )
            )
            masks.append(mask)

        final_masks = self._final_activation_masks_from_values(values=values, entries=entries)
        return StageResult(entries=entries, masks=final_masks)

    def _discover_inside_dysregulation_panel(
        self,
        values: np.ndarray,
        limits: np.ndarray,
        *,
        concept_idx: np.ndarray,
        rough_idx: np.ndarray,
        threshold_count: int,
        gene_ids: list[str],
        gene_names: list[str],
        reference: np.ndarray,
        panel_code: int,
        delta_threshold: float,
    ) -> StageResult:
        """Build panels 7-8.

        `carcinogenesis.nb` does not use an inside family during its
        T-gene / N-gene classification stage. We therefore return an empty
        stage here to keep the public family inventory stable while making the
        classification logic notebook-faithful.
        """

        del values, limits, concept_idx, rough_idx, threshold_count, gene_ids, gene_names, reference, panel_code, delta_threshold
        return StageResult(entries=[], masks=np.zeros((0, 0), dtype=bool))

    def _final_activation_masks_for_entries(self, *, cohort: PreparedCohort, entries: list[PanelEntry]) -> np.ndarray:
        """Build notebook-style final activation masks for accepted entries."""

        return self._final_activation_masks_from_values(values=cohort.filtered_values, entries=entries)

    def _final_activation_masks_from_values(self, *, values: np.ndarray, entries: list[PanelEntry]) -> np.ndarray:
        """Use unshifted exclusive intervals after margin-based discovery."""

        masks = [
            self._exclusive_interval_mask_from_values(
                entry=entry,
                values=values[:, entry.gene_index],
                reference=entry.reference,
                normal_idx=np.array([], dtype=int),
                tumor_idx=np.array([], dtype=int),
            )
            for entry in entries
        ]
        return self._as_mask_array(masks, values.shape[0])

    def query_gene(self, result: GenePanResult, gene_name: str) -> GeneQueryResult:
        """Return class membership and activation frequencies for one gene."""

        normal_idx, tumor_idx = self._split_samples(result.sample_types)
        sample_size = len(result.sample_types)
        matching_entries = [entry for entry in result.all_entries if entry.gene_name == gene_name]

        memberships: list[GeneQueryMembership] = []
        gene_id: str | None = None
        for entry in matching_entries:
            gene_id = entry.gene_id
            mask, interval_low, interval_high = self._exclusive_interval_mask(
                entry=entry,
                result=result,
                normal_idx=normal_idx,
                tumor_idx=tumor_idx,
            )
            n_count = int(np.sum(mask[normal_idx]))
            t_count = int(np.sum(mask[tumor_idx]))
            gene_category = "T-gene" if entry.panel_family_name.startswith("Only-T") else "N-gene"
            threshold_count = (
                self._minimum_significant_support_threshold(
                    len(tumor_idx),
                    sample_size,
                    pvalue=self.parameters.pvalue_tumor,
                )
                if gene_category == "T-gene"
                else self._minimum_significant_support_threshold(
                    len(normal_idx),
                    sample_size,
                    pvalue=self.parameters.pvalue_normal,
                )
            )
            activation_count = t_count if gene_category == "T-gene" else n_count
            memberships.append(
                GeneQueryMembership(
                    gene_category=gene_category,
                    panel_family_name=entry.panel_family_name,
                    notebook_panel_name=entry.notebook_panel_name,
                    notebook_panel_code=entry.panel_code,
                    threshold_kind=entry.threshold_kind,
                    threshold_low=entry.threshold_low,
                    threshold_high=entry.threshold_high,
                    exclusive_interval_low=interval_low,
                    exclusive_interval_high=interval_high,
                    n_activation_count=n_count,
                    n_activation_frequency=float(n_count) / float(len(normal_idx)) if len(normal_idx) else 0.0,
                    t_activation_count=t_count,
                    t_activation_frequency=float(t_count) / float(len(tumor_idx)) if len(tumor_idx) else 0.0,
                    passes_significance=activation_count > threshold_count,
                )
            )

        categories = sorted({membership.gene_category for membership in memberships})
        if categories == ["N-gene", "T-gene"]:
            gene_class = "NT-gene"
        elif categories:
            gene_class = categories[0]
        else:
            gene_class = "unclassified"

        return GeneQueryResult(
            gene_name=gene_name,
            gene_id=gene_id,
            gene_class=gene_class,
            memberships=memberships,
        )

    def query_gene_directly(self, gene_name: str) -> GeneQueryResult:
        """Assess one gene directly without constructing the full gene pools."""

        cohort = self._prepare_analysis_cohort()
        matches = self._matching_gene_indices(cohort=cohort, query=gene_name)
        if not matches:
            return GeneQueryResult(gene_name=gene_name, gene_id=None, gene_class="unclassified", memberships=[])

        if len(matches) > 1:
            raise RuntimeError(
                f"Gene name {gene_name!r} is ambiguous after identifier translation. "
                "Please query by a unique symbol or extend the interface to support gene identifiers explicitly."
            )

        gene_index = matches[0]
        memberships = self._direct_family_memberships_for_gene(cohort=cohort, gene_index=gene_index)
        categories = sorted({membership.gene_category for membership in memberships})
        if categories == ["N-gene", "T-gene"]:
            gene_class = "NT-gene"
        elif categories:
            gene_class = categories[0]
        else:
            gene_class = "unclassified"
        return GeneQueryResult(
            gene_name=cohort.gene_names[gene_index],
            gene_id=cohort.gene_ids[gene_index],
            gene_class=gene_class,
            memberships=memberships,
        )

    def query_gene_set_directly(self, gene_names: Iterable[str]) -> list[GeneQueryResult]:
        """Assess every gene in a user-supplied gene set."""

        return [self.query_gene_directly(gene_name) for gene_name in gene_names]

    @staticmethod
    def _matching_gene_indices(*, cohort: PreparedCohort, query: str) -> list[int]:
        """Match user queries by standard gene symbol/name or Ensembl ID."""

        normalized_query = query.strip()
        bare_query = normalized_query.split(".", 1)[0]
        normalized_query_lower = normalized_query.lower()
        bare_query_lower = bare_query.lower()
        matches: list[int] = []
        for idx, (gene_id, gene_name) in enumerate(zip(cohort.gene_ids, cohort.gene_names, strict=False)):
            gene_id_lower = gene_id.lower()
            if (
                gene_name.lower() == normalized_query_lower
                or gene_id_lower == normalized_query_lower
                or gene_id_lower.split(".", 1)[0] == bare_query_lower
            ):
                matches.append(idx)
        return matches

    def compute_all_gene_pools(self) -> dict[str, GeneSetResult]:
        """Compute the eight family-specific gene pools plus the two merged pools."""

        cohort = self._prepare_analysis_cohort()
        stages = self._build_all_family_stages(cohort)
        results: dict[str, GeneSetResult] = {}
        for panel_code, stage in stages.items():
            family_name = stage.entries[0].panel_family_name if stage.entries else self._panel_family_name(panel_code)
            gene_category = "T-gene" if family_name.startswith("Only-T") else "N-gene"
            results[family_name] = GeneSetResult(
                set_kind="pool",
                gene_category=gene_category,
                family_name=family_name,
                entries=stage.entries,
            )
        results["All-T"] = GeneSetResult(
            set_kind="pool",
            gene_category="T-gene",
            family_name="All-T",
            entries=stages[1].entries + stages[3].entries + stages[5].entries + stages[7].entries,
        )
        results["All-N"] = GeneSetResult(
            set_kind="pool",
            gene_category="N-gene",
            family_name="All-N",
            entries=stages[2].entries + stages[4].entries + stages[6].entries + stages[8].entries,
        )
        return results

    def compute_specific_gene_pool(self, family_name: str) -> GeneSetResult:
        """Compute one requested pool without forcing the other families."""

        if family_name in {"All-T", "All-N"}:
            return self.compute_all_gene_pools()[family_name]
        cohort = self._prepare_analysis_cohort()
        panel_code = self._panel_code_from_family_name(family_name)
        stage = self._build_family_stage(cohort=cohort, panel_code=panel_code)
        gene_category = "T-gene" if family_name.startswith("Only-T") else "N-gene"
        return GeneSetResult(
            set_kind="pool",
            gene_category=gene_category,
            family_name=family_name,
            entries=stage.entries,
        )

    def compute_specific_panel(self, family_name: str) -> GeneSetResult:
        """Compute one perfect panel from its correlative pool."""

        gene_category, selection = self.compute_panel_selection(family_name)
        return GeneSetResult("panel", gene_category, family_name, selection.selected_entries)

    def compute_panel_selection(
        self,
        family_name: str,
        *,
        tie_breaking_priority: str | None = None,
    ) -> tuple[str, MixSelection]:
        """Compute one perfect panel and keep its addition-order metadata."""

        cohort = self._prepare_analysis_cohort()
        if family_name == "All-T":
            stages = self._build_all_family_stages(cohort)
            entries = stages[1].entries + stages[3].entries + stages[5].entries + stages[7].entries
            masks = self._stack_masks([stages[1].masks, stages[3].masks, stages[5].masks, stages[7].masks], cohort.filtered_values.shape[0])
            selection = self._construct_formal_concept_reduct(
                entries=entries,
                masks=masks,
                target_idx=cohort.tumor_idx,
                tie_breaking_priority=tie_breaking_priority,
            )
            return "T-gene", selection
        if family_name == "All-N":
            stages = self._build_all_family_stages(cohort)
            entries = stages[2].entries + stages[4].entries + stages[6].entries + stages[8].entries
            masks = self._stack_masks([stages[2].masks, stages[4].masks, stages[6].masks, stages[8].masks], cohort.filtered_values.shape[0])
            selection = self._construct_formal_concept_reduct(
                entries=entries,
                masks=masks,
                target_idx=cohort.normal_idx,
                tie_breaking_priority=tie_breaking_priority,
            )
            return "N-gene", selection

        panel_code = self._panel_code_from_family_name(family_name)
        stage = self._build_family_stage(cohort=cohort, panel_code=panel_code)
        target_idx = cohort.tumor_idx if family_name.startswith("Only-T") else cohort.normal_idx
        selection = self._construct_formal_concept_reduct(
            entries=stage.entries,
            masks=stage.masks,
            target_idx=target_idx,
            tie_breaking_priority=tie_breaking_priority,
        )
        gene_category = "T-gene" if family_name.startswith("Only-T") else "N-gene"
        return gene_category, selection

    def _direct_family_memberships_for_gene(self, *, cohort: PreparedCohort, gene_index: int) -> list[GeneQueryMembership]:
        """Evaluate one gene against the eight notebook families only."""

        values = cohort.filtered_values[:, [gene_index]]
        limits = cohort.limits[[gene_index]]
        gene_id = [cohort.gene_ids[gene_index]]
        gene_name = [cohort.gene_names[gene_index]]
        reference = cohort.reference[[gene_index]]
        memberships: list[GeneQueryMembership] = []

        for panel_code in range(1, 9):
            stage = self._build_family_stage_for_single_gene(
                cohort=cohort,
                panel_code=panel_code,
                values=values,
                limits=limits,
                gene_ids=gene_id,
                gene_names=gene_name,
                reference=reference,
            )
            if not stage.entries:
                continue
            entry = stage.entries[0]
            mask = self._exclusive_interval_mask_from_values(
                entry=entry,
                values=values[:, 0],
                reference=float(reference[0]),
                normal_idx=cohort.normal_idx,
                tumor_idx=cohort.tumor_idx,
            )
            interval_low, interval_high = self._exclusive_interval_bounds_from_values(
                entry=entry,
                values=values[:, 0],
                normal_idx=cohort.normal_idx,
                tumor_idx=cohort.tumor_idx,
            )
            n_count = int(np.sum(mask[cohort.normal_idx]))
            t_count = int(np.sum(mask[cohort.tumor_idx]))
            gene_category = "T-gene" if entry.panel_family_name.startswith("Only-T") else "N-gene"
            threshold_count = cohort.tumor_threshold if gene_category == "T-gene" else cohort.normal_threshold
            memberships.append(
                GeneQueryMembership(
                    gene_category=gene_category,
                    panel_family_name=entry.panel_family_name,
                    notebook_panel_name=entry.notebook_panel_name,
                    notebook_panel_code=entry.panel_code,
                    threshold_kind=entry.threshold_kind,
                    threshold_low=entry.threshold_low,
                    threshold_high=entry.threshold_high,
                    exclusive_interval_low=interval_low,
                    exclusive_interval_high=interval_high,
                    n_activation_count=n_count,
                    n_activation_frequency=float(n_count) / float(len(cohort.normal_idx)) if len(cohort.normal_idx) else 0.0,
                    t_activation_count=t_count,
                    t_activation_frequency=float(t_count) / float(len(cohort.tumor_idx)) if len(cohort.tumor_idx) else 0.0,
                    passes_significance=(t_count if gene_category == "T-gene" else n_count) > threshold_count,
                )
            )
        return memberships

    def _exclusive_interval_mask(
        self,
        *,
        entry: PanelEntry,
        result: GenePanResult,
        normal_idx: np.ndarray,
        tumor_idx: np.ndarray,
    ) -> tuple[np.ndarray, float | None, float | None]:
        """Return the exclusive-interval mask and its interval bounds."""

        values = result.log2_values[:, entry.gene_index]
        interval_low, interval_high = self._exclusive_interval_bounds_from_values(
            entry=entry,
            values=values,
            normal_idx=normal_idx,
            tumor_idx=tumor_idx,
        )

        if interval_high is None:
            mask = values > interval_low
        elif interval_low is None:
            mask = values < interval_high
        elif entry.panel_family_name in {"Only-T-outside", "Only-N-outside"}:
            mask = (values < interval_low) | (values > interval_high)
        else:
            mask = (values > interval_low) & (values <= interval_high)
        return mask, interval_low, interval_high

    def _exclusive_interval_mask_from_values(
        self,
        *,
        entry: PanelEntry,
        values: np.ndarray,
        reference: float,
        normal_idx: np.ndarray,
        tumor_idx: np.ndarray,
    ) -> np.ndarray:
        """Return the exclusive-interval mask for one gene directly from values."""

        interval_low, interval_high = self._exclusive_interval_bounds_from_values(
            entry=entry,
            values=values,
            normal_idx=normal_idx,
            tumor_idx=tumor_idx,
        )
        if interval_high is None:
            return values > interval_low
        if interval_low is None:
            return values < interval_high
        if entry.panel_family_name in {"Only-T-outside", "Only-N-outside"}:
            return (values < interval_low) | (values > interval_high)
        return (values > interval_low) & (values <= interval_high)

    def _exclusive_interval_bounds_from_values(
        self,
        *,
        entry: PanelEntry,
        values: np.ndarray,
        normal_idx: np.ndarray,
        tumor_idx: np.ndarray,
    ) -> tuple[float | None, float | None]:
        """Recover the biologically meaningful exclusive interval bounds."""

        margin = self.parameters.interval_margin_fpkm
        if entry.panel_family_name == "Only-T-above":
            return entry.threshold_low - margin, None
        if entry.panel_family_name == "Only-N-above":
            return entry.threshold_low - margin, None
        if entry.panel_family_name == "Only-T-below":
            return None, entry.threshold_low + margin
        if entry.panel_family_name == "Only-N-below":
            return None, entry.threshold_low + margin
        if entry.panel_family_name == "Only-T-outside":
            return entry.threshold_low + margin, entry.threshold_high - margin
        if entry.panel_family_name == "Only-N-outside":
            return entry.threshold_low + margin, entry.threshold_high - margin
        return entry.threshold_low, entry.threshold_high

    def _construct_formal_concept_reduct(
        self,
        *,
        entries: list[PanelEntry],
        masks: np.ndarray,
        target_idx: np.ndarray,
        tie_breaking_priority: str | None = None,
    ) -> MixSelection:
        """Build a greedy perfect panel with a configurable tie priority.

        `first` scans candidates in their current gene-pool order and keeps
        the first strict improvement when additional coverage ties.
        `most_redundant_then_first` scans candidates by decreasing single-gene
        support, so ties in additional coverage prefer the candidate with more
        redundancy relative to already covered samples; remaining ties still use the
        first encountered candidate.
        """

        if not entries:
            return MixSelection(
                selected_positions=[],
                selected_entries=[],
                selected_support_counts=[],
                covered_fraction=0.0,
                target_size=len(target_idx),
            )

        candidate_positions = self._rough_set_admissible_positions(masks=masks, concept_idx=target_idx)
        if len(candidate_positions) == 0:
            return MixSelection(
                selected_positions=[],
                selected_entries=[],
                selected_support_counts=[],
                covered_fraction=0.0,
                target_size=len(target_idx),
            )

        tie_priority = tie_breaking_priority or self.parameters.panel_tie_breaking_priority
        support_sizes = masks[candidate_positions][:, target_idx].sum(axis=1)
        if tie_priority == "most_redundant_then_first":
            ordered_candidates = candidate_positions[np.argsort(-support_sizes, kind="stable")].tolist()
        elif tie_priority == "first":
            ordered_candidates = candidate_positions.tolist()
        else:
            raise ValueError("panel_tie_breaking_priority must be either 'first' or 'most_redundant_then_first'.")

        selected: list[int] = []
        current_coverage = 0
        target_size = len(target_idx)
        iteration = 1
        flag = 1

        while current_coverage < target_size and iteration <= len(ordered_candidates) and flag != 0:
            best_position = -1
            best_total = current_coverage
            selected_set = set(selected)
            for position in ordered_candidates:
                if position in selected_set:
                    continue
                possible_coverage = self._rough_set_extent(
                    masks=masks,
                    selected_positions=selected + [position],
                    concept_idx=target_idx,
                )
                if possible_coverage > best_total:
                    best_position = position
                    best_total = int(possible_coverage)
            flag = best_total - current_coverage

            if flag != 0 and best_position >= 0:
                selected.append(best_position)
            current_coverage = best_total
            iteration += 1

        covered_fraction = float(current_coverage) / float(target_size) if target_size else 0.0
        selected_support_counts = [int(masks[position][target_idx].sum()) for position in selected]
        return MixSelection(
            selected_positions=selected,
            selected_entries=[entries[pos] for pos in selected],
            selected_support_counts=selected_support_counts,
            covered_fraction=covered_fraction,
            target_size=target_size,
        )

    @staticmethod
    def _rough_set_admissible_positions(*, masks: np.ndarray, concept_idx: np.ndarray) -> np.ndarray:
        """Notebook `RA`: rows whose discovered samples stay inside the concept."""

        concept_mask = np.zeros(masks.shape[1], dtype=bool)
        concept_mask[concept_idx] = True
        return np.flatnonzero(np.all(~masks[:, ~concept_mask], axis=1))

    @staticmethod
    def _rough_set_extent(*, masks: np.ndarray, selected_positions: list[int], concept_idx: np.ndarray) -> int:
        """Notebook `RE`: number of concept samples discovered by the union."""

        if not selected_positions:
            return 0
        combined = np.any(masks[selected_positions], axis=0)
        return int(np.sum(combined[concept_idx]))

    def _minimum_significant_support_threshold(self, class_size: int, sample_size: int, *, pvalue: float) -> int:
        """Return the minimum accepted support threshold for one class.

        The downstream GenePan rule is `count > threshold`.  For the current
        project semantics, `pvalue` is used as a minimum accepted fraction of
        the relevant class:

        - tumor side: `pvalue_tumor = 0.1` means fewer than 10 percent of
          tumor samples is not enough
        - normal side: `pvalue_normal = 0.05` means fewer than 5 percent of
          normal samples is not enough

        So we convert the requested minimum fraction into the largest integer
        threshold that still enforces that strict `count > threshold` rule.
        """

        del sample_size
        minimum_count = math.ceil(class_size * pvalue)
        return max(0, minimum_count - 1)

    def _minimum_significant_subsplit_threshold(self, size: int, *, pvalue: float) -> int:
        """Translate `pvaluethresholdplus`.

        In the notebook this is the first cumulative Binomial threshold whose
        lower-tail probability exceeds the chosen p-value.

        In manuscript terms this controls whether both sides of a two-tailed or
        inside pattern are sufficiently populated to justify the proposed
        expression dysregulation pattern.
        """

        cumulative = math.exp(size * math.log(1.0 - self.parameters.split_probability))
        for selected in range(1, size + 1):
            cumulative += math.exp(
                self._log_choose(size, selected)
                + selected * math.log(self.parameters.split_probability)
                + (size - selected) * math.log(1.0 - self.parameters.split_probability)
            )
            if cumulative > pvalue:
                return selected
        return size

    @staticmethod
    def _log_choose(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float("-inf")
        return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)

    @staticmethod
    def _as_mask_array(masks: list[np.ndarray], sample_size: int) -> np.ndarray:
        if not masks:
            return np.zeros((0, sample_size), dtype=bool)
        return np.vstack(masks).astype(bool, copy=False)

    @staticmethod
    def _stack_masks(mask_sets: list[np.ndarray], sample_size: int) -> np.ndarray:
        non_empty = [mask for mask in mask_sets if mask.size]
        if not non_empty:
            return np.zeros((0, sample_size), dtype=bool)
        return np.vstack(non_empty).astype(bool, copy=False)


def panel_entries_to_frame(entries: list[PanelEntry]) -> pd.DataFrame:
    """Convert panel dataclasses into a tidy export table.

    The exported table uses manuscript-facing labels so that downstream users
    can compare the software output directly against the paper and
    supplementary material.
    """

    rows = []
    for entry in entries:
        rows.append(
            {
                "gene_index_1based": entry.gene_index + 1,
                "gene_id": entry.gene_id,
                "gene_name": entry.gene_name,
                "legacy_reference_value": entry.reference,
                "panel_family_name": entry.panel_family_name,
                "gene_category": "T-gene" if "Only-T" in entry.panel_family_name else "N-gene",
                "notebook_panel_name": entry.notebook_panel_name,
                "notebook_panel_code": entry.panel_code,
                "threshold_kind": entry.threshold_kind,
                "threshold_low": entry.threshold_low,
                "threshold_high": entry.threshold_high,
            }
        )
    return pd.DataFrame(rows)


def panel_selection_to_frame(selection: MixSelection) -> pd.DataFrame:
    """Convert a perfect panel to a table in greedy addition order."""

    frame = panel_entries_to_frame(selection.selected_entries)
    if frame.empty:
        return frame
    frame.insert(0, "addition_order", np.arange(1, len(frame) + 1, dtype=int))
    frame["target_activation_count"] = selection.selected_support_counts
    frame["target_activation_frequency"] = [
        count / selection.target_size if selection.target_size else 0.0 for count in selection.selected_support_counts
    ]
    return frame


def panel_selection_frequency_order_frame(selection: MixSelection) -> pd.DataFrame:
    """Return a perfect panel sorted by decreasing individual frequency."""

    frame = panel_selection_to_frame(selection)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["target_activation_count", "addition_order"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def write_outputs(result: GenePanResult, output_dir: str | Path) -> None:
    """Persist GenePan outputs in tabular form."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    panel_entries_to_frame(result.all_entries).to_csv(output / "panels_all.tsv", sep="\t", index=False)

    t_gene_pool_frame = panel_entries_to_frame(result.t_gene_pool)
    n_gene_pool_frame = panel_entries_to_frame(result.n_gene_pool)
    selected_t_gene_panel_frame = panel_selection_to_frame(result.selected_t_gene_panel)
    selected_n_gene_panel_frame = panel_selection_to_frame(result.selected_n_gene_panel)
    selected_t_gene_panel_frequency_frame = panel_selection_frequency_order_frame(result.selected_t_gene_panel)
    selected_n_gene_panel_frequency_frame = panel_selection_frequency_order_frame(result.selected_n_gene_panel)
    selected_t_gene_panel_most_redundant_then_first_frame = panel_selection_to_frame(result.selected_tumor_mix_most_redundant_then_first)
    selected_n_gene_panel_most_redundant_then_first_frame = panel_selection_to_frame(result.selected_normal_mix_most_redundant_then_first)
    selected_t_gene_panel_most_redundant_then_first_frequency_frame = panel_selection_frequency_order_frame(result.selected_tumor_mix_most_redundant_then_first)
    selected_n_gene_panel_most_redundant_then_first_frequency_frame = panel_selection_frequency_order_frame(result.selected_normal_mix_most_redundant_then_first)

    t_gene_pool_frame.to_csv(output / "t_gene_pool.tsv", sep="\t", index=False)
    n_gene_pool_frame.to_csv(output / "n_gene_pool.tsv", sep="\t", index=False)
    selected_t_gene_panel_frame.to_csv(
        output / "selected_t_gene_panel.tsv",
        sep="\t",
        index=False,
    )
    selected_n_gene_panel_frame.to_csv(
        output / "selected_n_gene_panel.tsv",
        sep="\t",
        index=False,
    )
    selected_t_gene_panel_frame.to_csv(output / "selected_t_gene_panel_addition_order.tsv", sep="\t", index=False)
    selected_n_gene_panel_frame.to_csv(output / "selected_n_gene_panel_addition_order.tsv", sep="\t", index=False)
    selected_t_gene_panel_frequency_frame.to_csv(output / "selected_t_gene_panel_frequency_order.tsv", sep="\t", index=False)
    selected_n_gene_panel_frequency_frame.to_csv(output / "selected_n_gene_panel_frequency_order.tsv", sep="\t", index=False)
    selected_t_gene_panel_frame.to_csv(output / "selected_t_gene_panel_first_addition_order.tsv", sep="\t", index=False)
    selected_n_gene_panel_frame.to_csv(output / "selected_n_gene_panel_first_addition_order.tsv", sep="\t", index=False)
    selected_t_gene_panel_frequency_frame.to_csv(output / "selected_t_gene_panel_first_frequency_order.tsv", sep="\t", index=False)
    selected_n_gene_panel_frequency_frame.to_csv(output / "selected_n_gene_panel_first_frequency_order.tsv", sep="\t", index=False)
    selected_t_gene_panel_most_redundant_then_first_frame.to_csv(output / "selected_t_gene_panel_most_redundant_then_first_addition_order.tsv", sep="\t", index=False)
    selected_n_gene_panel_most_redundant_then_first_frame.to_csv(output / "selected_n_gene_panel_most_redundant_then_first_addition_order.tsv", sep="\t", index=False)
    selected_t_gene_panel_most_redundant_then_first_frequency_frame.to_csv(output / "selected_t_gene_panel_most_redundant_then_first_frequency_order.tsv", sep="\t", index=False)
    selected_n_gene_panel_most_redundant_then_first_frequency_frame.to_csv(output / "selected_n_gene_panel_most_redundant_then_first_frequency_order.tsv", sep="\t", index=False)

    if result.t_gene_scan is not None and result.n_gene_scan is not None:
        _write_gene_category_sample_matrices(result, output)

    # Legacy compatibility outputs from the earlier Python translation.
    t_gene_pool_frame.to_csv(output / "panels_tumor_mix.tsv", sep="\t", index=False)
    n_gene_pool_frame.to_csv(output / "panels_normal_mix.tsv", sep="\t", index=False)
    selected_t_gene_panel_frame.to_csv(output / "selected_tumor_mix.tsv", sep="\t", index=False)
    selected_n_gene_panel_frame.to_csv(output / "selected_normal_mix.tsv", sep="\t", index=False)

    summary = {
        "all_panels": len(result.all_entries),
        "all_t_gene_pool_entries": len(result.t_gene_pool),
        "all_n_gene_pool_entries": len(result.n_gene_pool),
        "selected_t_gene_panel": len(result.selected_t_gene_panel.selected_entries),
        "selected_n_gene_panel": len(result.selected_n_gene_panel.selected_entries),
        "selected_t_gene_panel_coverage": result.selected_t_gene_panel.covered_fraction,
        "selected_n_gene_panel_coverage": result.selected_n_gene_panel.covered_fraction,
        "selected_t_gene_panel_most_redundant_then_first": len(result.selected_tumor_mix_most_redundant_then_first.selected_entries),
        "selected_n_gene_panel_most_redundant_then_first": len(result.selected_normal_mix_most_redundant_then_first.selected_entries),
        "selected_t_gene_panel_most_redundant_then_first_coverage": result.selected_tumor_mix_most_redundant_then_first.covered_fraction,
        "selected_n_gene_panel_most_redundant_then_first_coverage": result.selected_normal_mix_most_redundant_then_first.covered_fraction,
        # Legacy compatibility keys.
        "all_t_gene_panels": len(result.t_gene_pool),
        "all_n_gene_panels": len(result.n_gene_pool),
        "selected_tumor_mix": len(result.selected_t_gene_panel.selected_entries),
        "selected_normal_mix": len(result.selected_n_gene_panel.selected_entries),
        "selected_tumor_mix_coverage": result.selected_t_gene_panel.covered_fraction,
        "selected_normal_mix_coverage": result.selected_n_gene_panel.covered_fraction,
        "parameters": {
            "low_fpkm": result.parameters.low_fpkm,
            "detection_floor_fpkm": result.parameters.detection_floor_fpkm,
            "min_detected_fraction_normal": result.parameters.min_detected_fraction_normal,
            "min_detected_fraction_tumor": result.parameters.min_detected_fraction_tumor,
            "pseudocount_fpkm": result.parameters.pseudocount_fpkm,
            "pvalue_tumor": result.parameters.pvalue_tumor,
            "pvalue_normal": result.parameters.pvalue_normal,
            "split_pvalue": result.parameters.split_pvalue,
            "split_probability": result.parameters.split_probability,
            "delta_threshold_tumor": result.parameters.delta_threshold_tumor,
            "delta_threshold_normal": result.parameters.delta_threshold_normal,
            "panel_tie_breaking_priority": result.parameters.panel_tie_breaking_priority,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_gene_category_sample_matrices(result: GenePanResult, output: Path) -> None:
    """Write CChains-style binary matrices for merged T- and N-gene scans."""

    assert result.t_gene_scan is not None
    assert result.n_gene_scan is not None

    t_matrix = result.t_gene_scan.binary_patterns
    n_matrix = result.n_gene_scan.binary_patterns
    _write_binary_matrix(output / "sampleT_gene_level.txt", t_matrix)
    _write_binary_matrix(output / "sampleN_gene_level.txt", n_matrix)
    _write_name_csv(output / "namesT_gene_level.csv", result.t_gene_scan.gene_names)
    _write_name_csv(output / "namesN_gene_level.csv", result.n_gene_scan.gene_names)

    t_unique, t_unique_source_rows = _unique_rows_preserve_order(t_matrix)
    n_unique, n_unique_source_rows = _unique_rows_preserve_order(n_matrix)
    _write_binary_matrix(output / "sampleT_unique_patterns.txt", t_unique)
    _write_binary_matrix(output / "sampleN_unique_patterns.txt", n_unique)
    _write_name_csv(output / "namesT_unique_patterns.csv", [result.t_gene_scan.gene_names[index] for index in t_unique_source_rows])
    _write_name_csv(output / "namesN_unique_patterns.csv", [result.n_gene_scan.gene_names[index] for index in n_unique_source_rows])
    _write_cchains_data_bundle(
        output / "T_Network" / "data",
        t_unique,
        [result.t_gene_scan.gene_ids[index].split(".", 1)[0] for index in t_unique_source_rows],
    )
    _write_cchains_data_bundle(
        output / "N_Network" / "data",
        n_unique,
        [result.n_gene_scan.gene_ids[index].split(".", 1)[0] for index in n_unique_source_rows],
    )

    _write_gene_scan_order(output / "T_gene_order.tsv", result.t_gene_scan, t_matrix)
    _write_gene_scan_order(output / "N_gene_order.tsv", result.n_gene_scan, n_matrix)
    _write_unique_pattern_order(output / "T_unique_pattern_order.tsv", result.t_gene_scan, t_unique_source_rows)
    _write_unique_pattern_order(output / "N_unique_pattern_order.tsv", result.n_gene_scan, n_unique_source_rows)


def _write_binary_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix.astype(np.uint8, copy=False), fmt="%d", delimiter=" ")


def _write_name_csv(path: Path, gene_names: Iterable[str]) -> None:
    pd.Series(list(gene_names), dtype=str).to_csv(path, index=False, header=False)


def _write_cchains_data_bundle(path: Path, matrix: np.ndarray, gene_names: Iterable[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_binary_matrix(path / "sample.txt", matrix)
    _write_name_csv(path / "names.csv", gene_names)


def _unique_rows_preserve_order(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    seen: set[bytes] = set()
    rows: list[np.ndarray] = []
    source_rows: list[int] = []
    for row_index, row in enumerate(matrix):
        key = np.ascontiguousarray(row).tobytes()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row.copy())
        source_rows.append(row_index)
    if rows:
        return np.vstack(rows).astype(np.uint8, copy=False), source_rows
    return np.zeros((0, matrix.shape[1]), dtype=np.uint8), source_rows


def _write_gene_scan_order(path: Path, scan: GeneCategoryScan, matrix: np.ndarray) -> None:
    pd.DataFrame(
        {
            "row_index": np.arange(len(scan.gene_ids), dtype=int),
            "gene_id": scan.gene_ids,
            "gene_id_base": [gene_id.split(".", 1)[0] for gene_id in scan.gene_ids],
            "gene_name": scan.gene_names,
            "binary_sum": matrix.sum(axis=1).astype(int),
        }
    ).to_csv(path, sep="\t", index=False)


def _write_unique_pattern_order(path: Path, scan: GeneCategoryScan, source_rows: list[int]) -> None:
    pd.DataFrame(
        {
            "unique_row_index": np.arange(len(source_rows), dtype=int),
            "source_gene_row_index": source_rows,
            "source_gene_id": [scan.gene_ids[index] for index in source_rows],
            "source_gene_id_base": [scan.gene_ids[index].split(".", 1)[0] for index in source_rows],
            "source_gene_name": [scan.gene_names[index] for index in source_rows],
        }
    ).to_csv(path, sep="\t", index=False)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""

    parser = argparse.ArgumentParser(
        description="Run GenePan to discover T-gene and N-gene pools, perfect panels, and binary sample matrices."
    )
    parser.add_argument("base_dir", help="Directory containing sample.xls and the per-sample FPKM expression files.")
    parser.add_argument(
        "--output-dir",
        default="genepan_output",
        help="Directory where GenePan output tables, summaries, and binary matrices will be written.",
    )
    parser.add_argument("--low-fpkm", type=float, default=0.6, help="Low-expression threshold used by the family-classification logic.")
    parser.add_argument("--detection-floor-fpkm", type=float, default=0.0, help="Optional detection floor: values below this are treated as zero before expression-frequency filtering. The default is 0.0.")
    parser.add_argument("--min-detected-fraction-normal", type=float, default=0.0, help="Optional normal-side detection cutoff; genes below this and also below the tumor-side cutoff are removed. The default is 0.0.")
    parser.add_argument("--min-detected-fraction-tumor", type=float, default=0.0, help="Optional tumor-side detection cutoff; genes below this and also below the normal-side cutoff are removed. The default is 0.0.")
    parser.add_argument("--pseudocount-fpkm", type=float, default=0.1, help="Legacy compatibility parameter. The current raw-expression discovery path does not use a pseudocount.")
    parser.add_argument("--pvalue-tumor", type=float, default=0.1, help="Minimum tumor fraction required inside a tumor-exclusive interval.")
    parser.add_argument("--pvalue-normal", type=float, default=0.05, help="Minimum normal fraction required inside a normal-exclusive interval.")
    parser.add_argument("--split-pvalue", type=float, default=0.01, help="Significance threshold for the mixed-direction split checks.")
    parser.add_argument(
        "--split-probability",
        type=float,
        default=0.1,
        help="Probability parameter for mixed-direction split checks.",
    )
    parser.add_argument(
        "--delta-threshold-tumor",
        type=float,
        default=1.0,
        help="Minimum inside interval size for tumor-facing inside panels.",
    )
    parser.add_argument(
        "--delta-threshold-normal",
        type=float,
        default=1.0,
        help="Minimum inside interval size for normal-facing inside panels.",
    )
    parser.add_argument(
        "--query-gene",
        help="Directly assess one gene without constructing the full pools or panels, and print its GenePan classification and activation frequencies as JSON.",
    )
    parser.add_argument(
        "--query-gene-set",
        type=Path,
        help="Assess every gene listed in a text/CSV/TSV file. The first column is interpreted as a standard gene symbol/name, full Ensembl ID, or bare Ensembl ID.",
    )
    parser.add_argument(
        "--compute-all-gene-pools",
        action="store_true",
        help="Compute and print all eight family-specific gene pools plus the merged All-T and All-N pools.",
    )
    parser.add_argument(
        "--compute-gene-pool",
        help="Compute one specific gene pool, for example Only-T-above, Only-N-below, Only-N-outside, Only-N-inside, All-T, or All-N.",
    )
    parser.add_argument(
        "--compute-panel",
        help="Compute one specific perfect panel from its correlative pool, for example Only-T-above, Only-N-below, Only-N-outside, Only-N-inside, All-T, or All-N.",
    )
    parser.add_argument(
        "--panel-tie-breaking-priority",
        choices=["first", "most_redundant_then_first"],
        default="first",
        help=(
            "Tie-breaking priority for perfect-panel selection. 'first' keeps the first "
            "candidate encountered when additional coverage ties; "
            "'most_redundant_then_first' first prefers the candidate with greater "
            "redundancy relative to already covered samples."
        ),
    )
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()
    parameters = GenePanParameters(
        low_fpkm=args.low_fpkm,
        detection_floor_fpkm=args.detection_floor_fpkm,
        min_detected_fraction_normal=args.min_detected_fraction_normal,
        min_detected_fraction_tumor=args.min_detected_fraction_tumor,
        pseudocount_fpkm=args.pseudocount_fpkm,
        pvalue_tumor=args.pvalue_tumor,
        pvalue_normal=args.pvalue_normal,
        split_pvalue=args.split_pvalue,
        split_probability=args.split_probability,
        delta_threshold_tumor=args.delta_threshold_tumor,
        delta_threshold_normal=args.delta_threshold_normal,
        panel_tie_breaking_priority=args.panel_tie_breaking_priority,
    )
    engine = GenePan(args.base_dir, parameters=parameters)
    if args.query_gene:
        query = engine.query_gene_directly(args.query_gene)
        print(json.dumps(_gene_query_to_json_dict(query), indent=2))
        return
    if args.query_gene_set:
        gene_names = _read_gene_set_file(args.query_gene_set)
        queries = engine.query_gene_set_directly(gene_names)
        print(json.dumps([_gene_query_to_json_dict(query) for query in queries], indent=2))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _gene_query_results_to_frame(queries).to_csv(output_dir / "gene_set_query.tsv", sep="\t", index=False)
        return
    if args.compute_all_gene_pools:
        for family_name, gene_set in engine.compute_all_gene_pools().items():
            print(_format_gene_set_output(gene_set))
        return
    if args.compute_gene_pool:
        print(_format_gene_set_output(engine.compute_specific_gene_pool(args.compute_gene_pool)))
        return
    if args.compute_panel:
        print(_format_gene_set_output(engine.compute_specific_panel(args.compute_panel)))
        return
    run_pipeline(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        parameters=parameters,
    )


def run_pipeline(
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    parameters: GenePanParameters | None = None,
    low_fpkm: float = 0.6,
    detection_floor_fpkm: float = 0.0,
    min_detected_fraction_normal: float = 0.0,
    min_detected_fraction_tumor: float = 0.0,
    pseudocount_fpkm: float = 0.1,
    pvalue_tumor: float = 0.1,
    pvalue_normal: float = 0.05,
    split_pvalue: float = 0.01,
    split_probability: float = 0.1,
    delta_threshold_tumor: float = 1.0,
    delta_threshold_normal: float = 1.0,
    panel_tie_breaking_priority: str = "first",
) -> GenePanResult:
    """Convenience wrapper shared by the CLI and tests."""

    genepan = GenePan(
        base_dir,
        parameters=parameters
        or GenePanParameters(
            low_fpkm=low_fpkm,
            detection_floor_fpkm=detection_floor_fpkm,
            min_detected_fraction_normal=min_detected_fraction_normal,
            min_detected_fraction_tumor=min_detected_fraction_tumor,
            pseudocount_fpkm=pseudocount_fpkm,
            pvalue_tumor=pvalue_tumor,
            pvalue_normal=pvalue_normal,
            split_pvalue=split_pvalue,
            split_probability=split_probability,
            delta_threshold_tumor=delta_threshold_tumor,
            delta_threshold_normal=delta_threshold_normal,
            panel_tie_breaking_priority=panel_tie_breaking_priority,
        ),
    )
    result = genepan.run()
    write_outputs(result, output_dir)
    return result


def _gene_query_to_json_dict(query: GeneQueryResult) -> dict[str, object]:
    """Serialize a gene query result into a JSON-friendly dictionary."""

    return {
        "gene_name": query.gene_name,
        "gene_id": query.gene_id,
        "gene_class": query.gene_class,
        "memberships": [
            {
                "gene_category": membership.gene_category,
                "panel_family_name": membership.panel_family_name,
                "notebook_panel_name": membership.notebook_panel_name,
                "notebook_panel_code": membership.notebook_panel_code,
                "threshold_kind": membership.threshold_kind,
                "threshold_low": membership.threshold_low,
                "threshold_high": membership.threshold_high,
                "exclusive_interval_low": membership.exclusive_interval_low,
                "exclusive_interval_high": membership.exclusive_interval_high,
                "n_activation_count": membership.n_activation_count,
                "n_activation_frequency": membership.n_activation_frequency,
                "t_activation_count": membership.t_activation_count,
                "t_activation_frequency": membership.t_activation_frequency,
                "passes_significance": membership.passes_significance,
            }
            for membership in query.memberships
        ],
    }


def _read_gene_set_file(path: Path) -> list[str]:
    """Read a user gene-set file, taking the first non-empty token per row."""

    genes: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        token = stripped.replace(",", "\t").split("\t", 1)[0].strip()
        if token and token.lower() not in {"gene", "gene_id", "gene_name", "symbol"}:
            genes.append(token)
    return genes


def _gene_query_results_to_frame(queries: list[GeneQueryResult]) -> pd.DataFrame:
    """Flatten query results into one row per gene-family membership."""

    rows: list[dict[str, object]] = []
    for query in queries:
        if not query.memberships:
            rows.append(
                {
                    "query_gene_name": query.gene_name,
                    "gene_id": query.gene_id,
                    "gene_class": query.gene_class,
                    "gene_category": "",
                    "panel_family_name": "",
                    "threshold_kind": "",
                    "threshold_low": np.nan,
                    "threshold_high": np.nan,
                    "exclusive_interval_low": np.nan,
                    "exclusive_interval_high": np.nan,
                    "n_activation_count": 0,
                    "n_activation_frequency": 0.0,
                    "t_activation_count": 0,
                    "t_activation_frequency": 0.0,
                    "passes_significance": False,
                }
            )
            continue
        for membership in query.memberships:
            rows.append(
                {
                    "query_gene_name": query.gene_name,
                    "gene_id": query.gene_id,
                    "gene_class": query.gene_class,
                    "gene_category": membership.gene_category,
                    "panel_family_name": membership.panel_family_name,
                    "threshold_kind": membership.threshold_kind,
                    "threshold_low": membership.threshold_low,
                    "threshold_high": membership.threshold_high,
                    "exclusive_interval_low": membership.exclusive_interval_low,
                    "exclusive_interval_high": membership.exclusive_interval_high,
                    "n_activation_count": membership.n_activation_count,
                    "n_activation_frequency": membership.n_activation_frequency,
                    "t_activation_count": membership.t_activation_count,
                    "t_activation_frequency": membership.t_activation_frequency,
                    "passes_significance": membership.passes_significance,
                }
            )
    return pd.DataFrame(rows)


def _format_gene_set_output(gene_set: GeneSetResult) -> str:
    """Render a gene pool or panel with the requested descriptive header."""

    lines = [gene_set.header]
    if gene_set.entries:
        lines.extend(entry.gene_name for entry in gene_set.entries)
    return "\n".join(lines)


if __name__ == "__main__":
    main()

