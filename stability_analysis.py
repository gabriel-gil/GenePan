"""Generic downstream stability-analysis runner for GenePan.

This module deliberately lives outside `genepan.py`. It treats GenePan as the
core discovery engine and adds a configurable layer for cohort perturbation,
metric calculation, result writing, and reproducible execution.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import genepan


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = Path(r"C:\Users\gabri\GenePan\TCGA-PRAD")
DEFAULT_SAMPLE_SHEET = WORKSPACE / "sample_sheet_from_excel.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE / "stability_analysis_output"
DEFAULT_FIGURE_DIR = WORKSPACE / "stability_analysis_figures"
DEFAULT_MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))
POPCOUNT_TABLE = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class CohortConfig:
    source: str
    action: str
    side: str
    counts: tuple[int, ...]
    balanced_count_mode: str = "total"
    subsample_design: str = "independent"
    cumulative_base_count: int | None = None
    cumulative_base_seed: int | None = None
    cumulative_base_replicate_offset: int = 0


@dataclass(frozen=True)
class AnalysisConfig:
    analysis: str
    gene_category: str
    reference: str = "original_full_cohort"
    loevinger_threshold: float = 0.5


@dataclass(frozen=True)
class ExecutionConfig:
    replicates: int
    seed: int
    seed_stride: int
    max_workers: int
    resume: bool


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path
    figure_dir: Path
    write_figures: bool


@dataclass(frozen=True)
class StabilityConfig:
    cohort: CohortConfig
    analysis: AnalysisConfig
    execution: ExecutionConfig
    outputs: OutputConfig
    base_dir: Path


@dataclass(frozen=True)
class LoadedContext:
    engine: genepan.GenePan
    raw_values: np.ndarray
    gene_ids: list[str]
    gene_names: list[str]
    sample_types: list[str]
    normal_idx: np.ndarray
    tumor_idx: np.ndarray
    normal_sampler: Any | None
    tumor_sampler: Any | None
    reference_gene_sets: dict[str, set[str]]
    reference_gene_lists: dict[str, list[str]]
    global_gene_index: dict[str, int]
    reference_edge_sets: dict[str, set[int]]
    fixed_subsample_bases: dict[int, np.ndarray]


def main() -> None:
    config = _parse_args()
    config.outputs.output_dir.mkdir(parents=True, exist_ok=True)
    config.outputs.figure_dir.mkdir(parents=True, exist_ok=True)

    context = _load_context(config)
    raw_path = config.outputs.output_dir / f"{_run_stem(config)}_raw.tsv"
    summary_path = config.outputs.output_dir / f"{_run_stem(config)}_summary.tsv"
    seed_path = config.outputs.output_dir / f"{_run_stem(config)}_seed_report.json"
    cluster_path = config.outputs.output_dir / f"{_run_stem(config)}_block_distribution.tsv"

    existing = _load_existing(raw_path) if config.execution.resume else []
    done = {(int(row["replicate_index"]), int(row["count"])) for row in existing}
    jobs = [
        (replicate_index, count)
        for replicate_index in range(config.execution.replicates)
        for count in config.cohort.counts
        if (replicate_index, count) not in done
    ]
    print(
        f"[stability] source={config.cohort.source} action={config.cohort.action} "
        f"side={config.cohort.side} analysis={config.analysis.analysis} "
        f"counts={list(config.cohort.counts)} replicates={config.execution.replicates} "
        f"pending_jobs={len(jobs)}",
        flush=True,
    )

    rows = list(existing)
    distribution_rows = _load_existing(cluster_path) if config.execution.resume else []

    def run_job(replicate_index: int, count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return _run_one_job(
            config=config,
            context=context,
            replicate_index=replicate_index,
            count=count,
        )

    if jobs:
        with ThreadPoolExecutor(max_workers=config.execution.max_workers) as pool:
            futures = {pool.submit(run_job, rep, count): (rep, count) for rep, count in jobs}
            for future in as_completed(futures):
                row, dist = future.result()
                rows.append(row)
                distribution_rows.extend(dist)
                _write_tables(
                    rows=rows,
                    distribution_rows=distribution_rows,
                    raw_path=raw_path,
                    summary_path=summary_path,
                    cluster_path=cluster_path,
                    config=config,
                )
                print(f"[stability] completed={futures[future]} total_rows={len(rows)}", flush=True)

    _write_tables(
        rows=rows,
        distribution_rows=distribution_rows,
        raw_path=raw_path,
        summary_path=summary_path,
        cluster_path=cluster_path,
        config=config,
    )
    seed_path.write_text(
        json.dumps(_seed_report(config, context), indent=2),
        encoding="utf-8",
    )
    if config.outputs.write_figures:
        _write_placeholder_figure_summary(summary_path, config.outputs.figure_dir / f"{_run_stem(config)}_figure_manifest.json")
    print(f"[stability] wrote {raw_path}", flush=True)


def _parse_args() -> StabilityConfig:
    parser = argparse.ArgumentParser(description="Run downstream GenePan stability analyses.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--source", choices=["real", "kde", "pca"], required=True)
    parser.add_argument("--action", choices=["augment", "subsample", "none"], required=True)
    parser.add_argument("--side", choices=["normal", "tumor", "balanced", "none"], required=True)
    parser.add_argument("--counts", required=True, help="Comma-separated sample-count schedule, e.g. 0,10,20,30.")
    parser.add_argument("--balanced-count-mode", choices=["total", "per_side"], default="total")
    parser.add_argument("--subsample-design", choices=["independent", "cumulative"], default="independent")
    parser.add_argument("--cumulative-base-count", type=int)
    parser.add_argument("--cumulative-base-seed", type=int)
    parser.add_argument("--cumulative-base-replicate-offset", type=int, default=0)
    parser.add_argument("--analysis", choices=["gene_set_overlap", "block_structure", "loevinger_edges"], required=True)
    parser.add_argument("--gene-category", choices=["T-gene", "N-gene", "both"], default="both")
    parser.add_argument("--reference", choices=["original_full_cohort"], default="original_full_cohort")
    parser.add_argument("--loevinger-threshold", type=float, default=0.5)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-stride", type=int, default=100_000)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    counts = tuple(_parse_counts(args.counts))
    if not counts:
        raise RuntimeError("--counts cannot be empty.")
    if args.source == "real" and args.action == "augment":
        raise RuntimeError("source=real with action=augment is ambiguous; use action=subsample for real-sample scans.")
    if args.source in {"kde", "pca"} and args.action == "subsample":
        raise RuntimeError("Synthetic sources support action=augment, not action=subsample.")
    if args.side == "balanced" and args.action != "augment":
        raise RuntimeError("side=balanced is currently defined for synthetic augmentation.")
    if args.subsample_design == "cumulative":
        if args.source != "real" or args.action != "subsample":
            raise RuntimeError("--subsample-design cumulative requires source=real and action=subsample.")
        if args.side != "tumor":
            raise RuntimeError("--subsample-design cumulative currently supports side=tumor.")
        if args.cumulative_base_count is None:
            raise RuntimeError("--cumulative-base-count is required for cumulative subsampling.")
        if min(counts) != args.cumulative_base_count:
            raise RuntimeError("--counts must include the cumulative base count as the first/smallest point.")
    if args.analysis == "block_structure" and args.gene_category not in {"N-gene", "both"}:
        raise RuntimeError("block_structure is currently defined for N-genes.")
    if args.analysis == "loevinger_edges" and args.gene_category == "both":
        raise RuntimeError("loevinger_edges requires --gene-category T-gene or N-gene.")

    return StabilityConfig(
        cohort=CohortConfig(
            source=args.source,
            action=args.action,
            side=args.side,
            counts=counts,
            balanced_count_mode=args.balanced_count_mode,
            subsample_design=args.subsample_design,
            cumulative_base_count=args.cumulative_base_count,
            cumulative_base_seed=args.cumulative_base_seed,
            cumulative_base_replicate_offset=args.cumulative_base_replicate_offset,
        ),
        analysis=AnalysisConfig(
            analysis=args.analysis,
            gene_category=args.gene_category,
            reference=args.reference,
            loevinger_threshold=args.loevinger_threshold,
        ),
        execution=ExecutionConfig(
            replicates=args.replicates,
            seed=args.seed,
            seed_stride=args.seed_stride,
            max_workers=args.max_workers,
            resume=args.resume,
        ),
        outputs=OutputConfig(
            output_dir=args.output_dir,
            figure_dir=args.figure_dir,
            write_figures=not args.no_figures,
        ),
        base_dir=args.base_dir,
    )


def _load_context(config: StabilityConfig) -> LoadedContext:
    engine = _make_engine(config.base_dir)
    raw_values, gene_ids, gene_names, sample_types = engine._load_tcga_expression_cohort()
    normal_idx, tumor_idx = engine._split_samples(sample_types)
    normal_values = raw_values[normal_idx, :]
    tumor_values = raw_values[tumor_idx, :]

    normal_sampler = None
    tumor_sampler = None
    if config.cohort.source == "kde":
        normal_sampler = engine._prepare_synthetic_profile_sampler(normal_values)
        tumor_sampler = engine._prepare_synthetic_profile_sampler(tumor_values)
    elif config.cohort.source == "pca":
        normal_sampler = engine._prepare_pca_profile_sampler(normal_values, reference_values=normal_values, latent_dim=10)
        tumor_sampler = engine._prepare_pca_profile_sampler(tumor_values, reference_values=normal_values, latent_dim=10)

    baseline = engine._prepare_analysis_cohort_from_loaded_data(raw_values, gene_ids, gene_names, sample_types)
    reference_gene_sets = {
        "T-gene": engine._deduplicated_gene_ids_for_category(baseline, gene_category="T-gene"),
        "N-gene": engine._deduplicated_gene_ids_for_category(baseline, gene_category="N-gene"),
    }
    reference_gene_lists = {
        category: sorted(gene_ids)
        for category, gene_ids in reference_gene_sets.items()
    }
    global_gene_index = {_canonical_gene_id(gene_id): idx for idx, gene_id in enumerate(gene_ids)}
    reference_edge_sets: dict[str, set[int]] = {}
    if config.analysis.analysis == "loevinger_edges":
        category = config.analysis.gene_category
        reference_scan = engine.build_gene_category_scan(cohort=baseline, gene_category=category)
        reference_edge_sets[category] = _loevinger_global_edge_keys(
            scan=reference_scan,
            threshold=config.analysis.loevinger_threshold,
            global_gene_index=global_gene_index,
        )
    fixed_subsample_bases = _fixed_subsample_bases(config, normal_idx=normal_idx, tumor_idx=tumor_idx)
    return LoadedContext(
        engine=engine,
        raw_values=raw_values,
        gene_ids=gene_ids,
        gene_names=gene_names,
        sample_types=sample_types,
        normal_idx=normal_idx,
        tumor_idx=tumor_idx,
        normal_sampler=normal_sampler,
        tumor_sampler=tumor_sampler,
        reference_gene_sets=reference_gene_sets,
        reference_gene_lists=reference_gene_lists,
        global_gene_index=global_gene_index,
        reference_edge_sets=reference_edge_sets,
        fixed_subsample_bases=fixed_subsample_bases,
    )


def _make_engine(base_dir: Path) -> genepan.GenePan:
    parameters = genepan.GenePanParameters(
        low_fpkm=0.6,
        interval_margin_fpkm=0.1,
        detection_floor_fpkm=0.0,
        min_detected_fraction_normal=0.0,
        min_detected_fraction_tumor=0.0,
        pvalue_tumor=0.1,
        pvalue_normal=0.05,
    )
    engine = genepan.GenePan(base_dir, parameters=parameters)
    if DEFAULT_SAMPLE_SHEET.exists():
        engine._read_sample_sheet = lambda path: pd.read_csv(DEFAULT_SAMPLE_SHEET)
    return engine


def _fixed_subsample_bases(
    config: StabilityConfig,
    *,
    normal_idx: np.ndarray,
    tumor_idx: np.ndarray,
) -> dict[int, np.ndarray]:
    if config.cohort.subsample_design != "cumulative":
        return {}
    if config.cohort.cumulative_base_count is None:
        raise RuntimeError("cumulative_base_count is required for cumulative subsampling.")
    base_seed = config.cohort.cumulative_base_seed
    if base_seed is None:
        base_seed = config.execution.seed
    source_idx = normal_idx if config.cohort.side == "normal" else tumor_idx
    bases: dict[int, np.ndarray] = {}
    for replicate_index in range(config.execution.replicates):
        seed_replicate = replicate_index + config.cohort.cumulative_base_replicate_offset
        random_seed = base_seed + seed_replicate * config.execution.seed_stride
        rng = np.random.default_rng(random_seed)
        bases[replicate_index] = np.sort(
            rng.choice(source_idx, size=config.cohort.cumulative_base_count, replace=False).astype(np.int64, copy=False)
        )
    return bases


def _run_one_job(
    *,
    config: StabilityConfig,
    context: LoadedContext,
    replicate_index: int,
    count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    random_seed = _job_seed(config.execution.seed, config.execution.seed_stride, replicate_index, count, config.cohort.subsample_design)
    rng = np.random.default_rng(random_seed)
    cohort, sample_identity = _build_perturbed_cohort(
        config=config,
        context=context,
        replicate_index=replicate_index,
        count=count,
        rng=rng,
    )
    base_row = {
        "source": config.cohort.source,
        "action": config.cohort.action,
        "side": config.cohort.side,
        "analysis": config.analysis.analysis,
        "count": count,
        "replicate_index": replicate_index,
        "random_seed_used": random_seed,
        **sample_identity,
    }
    if config.analysis.analysis == "gene_set_overlap":
        return {**base_row, **_gene_set_overlap_metrics(config, context, cohort)}, []
    if config.analysis.analysis == "block_structure":
        metrics, distribution = _block_structure_metrics(config, context, cohort)
        dist_rows = [
            {
                **base_row,
                **row,
            }
            for row in distribution
        ]
        return {**base_row, **metrics}, dist_rows
    if config.analysis.analysis == "loevinger_edges":
        return {**base_row, **_loevinger_edge_metrics(config, context, cohort)}, []
    raise RuntimeError(f"Unsupported analysis: {config.analysis.analysis}")


def _build_perturbed_cohort(
    *,
    config: StabilityConfig,
    context: LoadedContext,
    replicate_index: int,
    count: int,
    rng: np.random.Generator,
) -> tuple[genepan.PreparedCohort, dict[str, str | int]]:
    if config.cohort.action == "none":
        cohort = context.engine._prepare_analysis_cohort_from_loaded_data(
            context.raw_values,
            context.gene_ids,
            context.gene_names,
            context.sample_types,
        )
        return cohort, {"normal_count": len(context.normal_idx), "tumor_count": len(context.tumor_idx)}

    if config.cohort.action == "augment":
        synthetic_normal_count, synthetic_tumor_count = _synthetic_counts(config, count)
        synthetic_blocks: list[np.ndarray] = []
        synthetic_types: list[str] = []
        if synthetic_normal_count:
            synthetic_blocks.append(_sample_synthetic(context, "normal", synthetic_normal_count, rng, config.cohort.source))
            synthetic_types.extend(["Solid Tissue Normal"] * synthetic_normal_count)
        if synthetic_tumor_count:
            synthetic_blocks.append(_sample_synthetic(context, "tumor", synthetic_tumor_count, rng, config.cohort.source))
            synthetic_types.extend(["Primary Tumor"] * synthetic_tumor_count)
        values = context.raw_values if not synthetic_blocks else np.vstack([context.raw_values, *synthetic_blocks])
        sample_types = list(context.sample_types) + synthetic_types
        cohort = context.engine._prepare_analysis_cohort_from_loaded_data(values, context.gene_ids, context.gene_names, sample_types)
        return cohort, {
            "normal_count": len(context.normal_idx) + synthetic_normal_count,
            "tumor_count": len(context.tumor_idx) + synthetic_tumor_count,
            "synthetic_normal_count": synthetic_normal_count,
            "synthetic_tumor_count": synthetic_tumor_count,
        }

    if config.cohort.action == "subsample":
        if config.cohort.source != "real":
            raise RuntimeError("Subsampling currently requires source=real.")
        if config.cohort.side == "normal":
            if count > len(context.normal_idx):
                raise RuntimeError(f"Cannot select {count} normals from {len(context.normal_idx)} available normals.")
            selected_normal = np.sort(rng.choice(context.normal_idx, size=count, replace=False))
            selected_idx = np.concatenate([selected_normal, context.tumor_idx])
            sample_types = ["Solid Tissue Normal"] * len(selected_normal) + ["Primary Tumor"] * len(context.tumor_idx)
            identity = ",".join(str(int(idx)) for idx in selected_normal)
        elif config.cohort.side == "tumor":
            if count > len(context.tumor_idx):
                raise RuntimeError(f"Cannot select {count} tumors from {len(context.tumor_idx)} available tumors.")
            if config.cohort.subsample_design == "cumulative":
                selected_tumor = _cumulative_subsample_for_job(
                    base=context.fixed_subsample_bases[replicate_index],
                    source_idx=context.tumor_idx,
                    count=count,
                    rng=rng,
                )
            else:
                selected_tumor = np.sort(rng.choice(context.tumor_idx, size=count, replace=False))
            selected_idx = np.concatenate([context.normal_idx, selected_tumor])
            sample_types = ["Solid Tissue Normal"] * len(context.normal_idx) + ["Primary Tumor"] * len(selected_tumor)
            identity = ",".join(str(int(idx)) for idx in selected_tumor)
        else:
            raise RuntimeError("Real subsampling currently supports side=normal or side=tumor.")
        cohort = context.engine._prepare_analysis_cohort_from_loaded_data(
            context.raw_values[selected_idx, :],
            context.gene_ids,
            context.gene_names,
            sample_types,
        )
        return cohort, {
            "normal_count": int(np.sum(np.array(sample_types) == "Solid Tissue Normal")),
            "tumor_count": int(np.sum(np.array(sample_types) != "Solid Tissue Normal")),
            "selected_source_indices": identity,
        }

    raise RuntimeError(f"Unsupported action: {config.cohort.action}")


def _sample_synthetic(
    context: LoadedContext,
    side: str,
    count: int,
    rng: np.random.Generator,
    source: str,
) -> np.ndarray:
    sampler = context.normal_sampler if side == "normal" else context.tumor_sampler
    if source == "kde":
        return context.engine._sample_synthetic_profiles(synthetic_count=count, rng=rng, sampler=sampler)
    if source == "pca":
        return context.engine._sample_synthetic_profiles(synthetic_count=count, rng=rng, pca_sampler=sampler)
    raise RuntimeError(f"Synthetic sampling requires source=kde or source=pca, got {source!r}.")


def _synthetic_counts(config: StabilityConfig, count: int) -> tuple[int, int]:
    if config.cohort.side == "normal":
        return count, 0
    if config.cohort.side == "tumor":
        return 0, count
    if config.cohort.side == "balanced":
        if config.cohort.balanced_count_mode == "per_side":
            return count, count
        normal_count = count // 2
        tumor_count = count - normal_count
        return normal_count, tumor_count
    raise RuntimeError("Synthetic augmentation requires side=normal, side=tumor, or side=balanced.")


def _cumulative_subsample_for_job(
    *,
    base: np.ndarray,
    source_idx: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    additional_count = count - len(base)
    if additional_count < 0:
        raise RuntimeError("Cumulative subsampling count cannot be smaller than the base count.")
    if additional_count == 0:
        return np.sort(base)
    base_set = set(base.astype(int).tolist())
    remaining = np.array([idx for idx in source_idx if int(idx) not in base_set], dtype=np.int64)
    ordered_remaining = rng.permutation(remaining)
    return np.sort(np.concatenate([base, ordered_remaining[:additional_count]]))


def _gene_set_overlap_metrics(
    config: StabilityConfig,
    context: LoadedContext,
    cohort: genepan.PreparedCohort,
) -> dict[str, float | int]:
    categories = ["T-gene", "N-gene"] if config.analysis.gene_category == "both" else [config.analysis.gene_category]
    metrics: dict[str, float | int] = {}
    for category in categories:
        reference = context.reference_gene_sets[category]
        observed = context.engine._deduplicated_gene_ids_for_category(cohort, gene_category=category)
        prefix = "t_gene" if category == "T-gene" else "n_gene"
        metrics.update(_set_metrics(prefix, reference, observed))
    return metrics


def _block_structure_metrics(
    config: StabilityConfig,
    context: LoadedContext,
    cohort: genepan.PreparedCohort,
) -> tuple[dict[str, float | int], list[dict[str, int]]]:
    metrics, distribution = _target_n_gene_block_metrics(
        engine=context.engine,
        cohort=cohort,
        target_n_gene_ids=context.reference_gene_lists["N-gene"],
    )
    non_singleton = np.array(
        [
            row["block_size"]
            for row in distribution
            for _ in range(row["block_count"])
            if row["block_size"] > 1
        ],
        dtype=float,
    )
    metrics["mean_non_singleton_block_size"] = float(non_singleton.mean()) if non_singleton.size else 0.0
    return metrics, distribution


def _target_n_gene_block_metrics(
    *,
    engine: genepan.GenePan,
    cohort: genepan.PreparedCohort,
    target_n_gene_ids: list[str],
) -> tuple[dict[str, float | int], list[dict[str, int]]]:
    gene_index = {gene_id: idx for idx, gene_id in enumerate(cohort.gene_ids)}
    present_pairs = [(gene_id, gene_index[gene_id]) for gene_id in target_n_gene_ids if gene_id in gene_index]
    if not present_pairs:
        return _empty_n_gene_block_metrics(len(target_n_gene_ids), 0, len(cohort.normal_idx)), []

    positions = np.array([idx for _gene_id, idx in present_pairs], dtype=np.int64)
    values = cohort.filtered_values[:, positions]
    normal_values = values[cohort.normal_idx, :]
    tumor_values = values[cohort.tumor_idx, :]
    normal_k = cohort.normal_threshold + 1
    if normal_k > normal_values.shape[0]:
        return _empty_n_gene_block_metrics(len(target_n_gene_ids), len(present_pairs), len(cohort.normal_idx)), []

    margin = engine.parameters.interval_margin_fpkm
    normal_max = normal_values.max(axis=0)
    tumor_max = tumor_values.max(axis=0)
    normal_min = normal_values.min(axis=0)
    tumor_min = tumor_values.min(axis=0)
    kth_normal_largest = np.partition(normal_values, normal_values.shape[0] - normal_k, axis=0)[normal_values.shape[0] - normal_k, :]
    kth_normal_smallest = np.partition(normal_values, normal_k - 1, axis=0)[normal_k - 1, :]

    active = np.zeros((len(present_pairs), len(cohort.normal_idx)), dtype=bool)
    retained = np.zeros(len(present_pairs), dtype=bool)

    above = (normal_max > (tumor_max + margin)) & (kth_normal_largest > (tumor_max + margin))
    if np.any(above):
        retained |= above
        active[above, :] |= normal_values[:, above].T > (tumor_max[above] + margin)[:, None]

    below = (normal_min < (tumor_min - margin)) & (kth_normal_smallest < (tumor_min - margin))
    if np.any(below):
        retained |= below
        active[below, :] |= normal_values[:, below].T < (tumor_min[below] - margin)[:, None]

    outside = (
        (normal_max > (tumor_max + margin))
        & (normal_min < (tumor_min - margin))
        & (kth_normal_largest > (tumor_max + margin))
        & (kth_normal_smallest < (tumor_min - margin))
    )
    if np.any(outside):
        retained |= outside
        active[outside, :] |= (normal_values[:, outside].T < (tumor_min[outside] - margin)[:, None]) | (
            normal_values[:, outside].T > (tumor_max[outside] + margin)[:, None]
        )

    retained_count = int(np.sum(retained))
    if retained_count == 0:
        return _empty_n_gene_block_metrics(len(target_n_gene_ids), len(present_pairs), len(cohort.normal_idx)), []

    binary = (~active[retained, :]).astype(np.uint8, copy=False)
    _unique_patterns, _inverse, counts = np.unique(binary, axis=0, return_inverse=True, return_counts=True)
    distinct_count = int(len(counts))
    largest = int(counts.max()) if counts.size else 0
    singleton = int(np.sum(counts == 1))
    distribution = [
        {
            "block_size": int(size),
            "block_count": int(np.sum(counts == size)),
            "gene_count_in_block_size": int(size * np.sum(counts == size)),
        }
        for size in sorted(set(counts.astype(int).tolist()))
    ]
    metrics = {
        "reference_n_gene_count": len(target_n_gene_ids),
        "present_original_n_gene_count": len(present_pairs),
        "retained_original_n_gene_count": retained_count,
        "n_gene_recall": retained_count / len(target_n_gene_ids) if target_n_gene_ids else 1.0,
        "distinct_block_count": distinct_count,
        "largest_block_size": largest,
        "singleton_block_count": singleton,
        "mean_block_size": float(retained_count) / float(distinct_count) if distinct_count else 0.0,
        "normal_sample_count": len(cohort.normal_idx),
    }
    return metrics, distribution


def _empty_n_gene_block_metrics(reference_count: int, present_count: int, normal_count: int) -> dict[str, float | int]:
    return {
        "reference_n_gene_count": reference_count,
        "present_original_n_gene_count": present_count,
        "retained_original_n_gene_count": 0,
        "n_gene_recall": 0.0,
        "distinct_block_count": 0,
        "largest_block_size": 0,
        "singleton_block_count": 0,
        "mean_block_size": 0.0,
        "normal_sample_count": normal_count,
    }


def _loevinger_edge_metrics(
    config: StabilityConfig,
    context: LoadedContext,
    cohort: genepan.PreparedCohort,
) -> dict[str, float | int]:
    category = config.analysis.gene_category
    reference = context.reference_edge_sets[category]
    scan = context.engine.build_gene_category_scan(cohort=cohort, gene_category=category)
    observed = _loevinger_global_edge_keys(
        scan=scan,
        threshold=config.analysis.loevinger_threshold,
        global_gene_index=context.global_gene_index,
    )
    shared = reference & observed
    union = reference | observed
    possible_edges = len(scan.gene_ids) * max(len(scan.gene_ids) - 1, 0)
    prefix = "t_gdn" if category == "T-gene" else "n_gdn"
    return {
        f"{prefix}_reference_edge_count": len(reference),
        f"{prefix}_observed_edge_count": len(observed),
        f"{prefix}_shared_edge_count": len(shared),
        f"{prefix}_edge_recall": len(shared) / len(reference) if reference else 0.0,
        f"{prefix}_edge_precision": len(shared) / len(observed) if observed else 0.0,
        f"{prefix}_edge_jaccard": len(shared) / len(union) if union else 0.0,
        f"{prefix}_observed_node_count": len(scan.gene_ids),
        f"{prefix}_observed_edge_density": len(observed) / possible_edges if possible_edges else 0.0,
    }


def _loevinger_global_edge_keys(
    *,
    scan: genepan.GeneCategoryScan,
    threshold: float,
    global_gene_index: dict[str, int],
) -> set[int]:
    local_pair_keys = _loevinger_pair_keys(
        scan.active_patterns,
        threshold=threshold,
    )
    local_count = len(scan.gene_ids)
    global_count = len(global_gene_index)
    local_to_global = np.array(
        [global_gene_index[_canonical_gene_id(gene_id)] for gene_id in scan.gene_ids],
        dtype=np.int64,
    )
    return {
        int(local_to_global[local_key // local_count] * global_count + local_to_global[local_key % local_count])
        for local_key in local_pair_keys
    }


def _pack_binary_patterns(patterns: np.ndarray) -> np.ndarray:
    if patterns.ndim != 2:
        raise ValueError("patterns must be a 2D matrix")
    if patterns.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.packbits(patterns.astype(np.uint8, copy=False), axis=1, bitorder="little")


def _loevinger_pair_keys(active_patterns: np.ndarray, *, threshold: float = 0.5, block_size: int = 128) -> set[int]:
    packed = _pack_binary_patterns(active_patterns.astype(np.uint8, copy=False))
    gene_count = packed.shape[0]
    sample_count = active_patterns.shape[1]
    if gene_count <= 1 or sample_count == 0:
        return set()

    activation_counts = POPCOUNT_TABLE[packed].sum(axis=1, dtype=np.int32)
    activation_frequencies = activation_counts.astype(np.float64) / float(sample_count)
    qualifying_pairs: set[int] = set()

    for start in range(0, gene_count, block_size):
        stop = min(start + block_size, gene_count)
        block = packed[start:stop]
        and_bytes = np.bitwise_and(block[:, None, :], packed[None, :, :])
        coactivation = POPCOUNT_TABLE[and_bytes].sum(axis=2, dtype=np.int32)
        count_i = activation_counts[start:stop].astype(np.float64)[:, None]
        p_j = activation_frequencies[None, :]
        denominator = 1.0 - p_j
        with np.errstate(divide="ignore", invalid="ignore"):
            conditional = np.divide(
                coactivation.astype(np.float64),
                count_i,
                out=np.zeros_like(coactivation, dtype=np.float64),
                where=count_i > 0.0,
            )
            h_values = np.divide(
                conditional - p_j,
                denominator,
                out=np.full_like(conditional, -np.inf, dtype=np.float64),
                where=denominator > 0.0,
            )

        qualifying = h_values > threshold
        row_positions = np.arange(start, stop, dtype=np.int64)
        qualifying[np.arange(stop - start), row_positions] = False
        row_idx, col_idx = np.where(qualifying)
        for local_row, col in zip(row_idx.tolist(), col_idx.tolist(), strict=False):
            i = start + local_row
            qualifying_pairs.add(i * gene_count + col)

    return qualifying_pairs


def _canonical_gene_id(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0]


def _set_metrics(prefix: str, reference: set[str], observed: set[str]) -> dict[str, float | int]:
    shared = reference & observed
    union = reference | observed
    return {
        f"{prefix}_reference_count": len(reference),
        f"{prefix}_observed_count": len(observed),
        f"{prefix}_shared_count": len(shared),
        f"{prefix}_recall": len(shared) / len(reference) if reference else 0.0,
        f"{prefix}_precision": len(shared) / len(observed) if observed else 0.0,
        f"{prefix}_jaccard": len(shared) / len(union) if union else 0.0,
    }


def _write_tables(
    *,
    rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    raw_path: Path,
    summary_path: Path,
    cluster_path: Path,
    config: StabilityConfig,
) -> None:
    if not rows:
        return
    raw = pd.DataFrame(rows).drop_duplicates(["replicate_index", "count"], keep="last")
    raw = raw.sort_values(["count", "replicate_index"], kind="stable")
    raw.to_csv(raw_path, sep="\t", index=False)
    numeric_cols = [
        col for col in raw.columns
        if col not in {"source", "action", "side", "analysis", "selected_source_indices"}
        and pd.api.types.is_numeric_dtype(raw[col])
    ]
    metric_cols = [col for col in numeric_cols if col not in {"count", "replicate_index", "random_seed_used"}]
    summary = raw.groupby("count")[metric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary.to_csv(summary_path, sep="\t", index=False)
    if distribution_rows:
        pd.DataFrame(distribution_rows).drop_duplicates(
            ["replicate_index", "count", "block_size"],
            keep="last",
        ).sort_values(["count", "replicate_index", "block_size"], kind="stable").to_csv(cluster_path, sep="\t", index=False)


def _seed_report(config: StabilityConfig, context: LoadedContext) -> dict[str, Any]:
    if config.cohort.subsample_design == "cumulative":
        seed_formula = "seed + replicate_index * seed_stride"
    else:
        seed_formula = "seed + replicate_index * seed_stride + count"
    return {
        "cohort": asdict(config.cohort),
        "analysis": asdict(config.analysis),
        "execution": asdict(config.execution),
        "base_dir": str(config.base_dir),
        "original_normal_count": int(len(context.normal_idx)),
        "original_tumor_count": int(len(context.tumor_idx)),
        "seed_formula": seed_formula,
        "loevinger_threshold": config.analysis.loevinger_threshold,
    }


def _write_placeholder_figure_summary(summary_path: Path, manifest_path: Path) -> None:
    manifest_path.write_text(
        json.dumps(
            {
                "status": "summary_ready",
                "summary_path": str(summary_path),
                "note": "Publication-style figure rendering is intentionally separated and can be regenerated from the summary/raw TSV files.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path, sep="\t").to_dict("records")


def _parse_counts(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _job_seed(seed: int, seed_stride: int, replicate_index: int, count: int, subsample_design: str) -> int:
    if subsample_design == "cumulative":
        return int(seed + replicate_index * seed_stride)
    return int(seed + replicate_index * seed_stride + count)


def _run_stem(config: StabilityConfig) -> str:
    category = config.analysis.gene_category.lower().replace("-", "_")
    return f"{config.cohort.source}_{config.cohort.action}_{config.cohort.side}_{config.analysis.analysis}_{category}"


if __name__ == "__main__":
    main()
