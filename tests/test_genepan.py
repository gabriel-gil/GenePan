from __future__ import annotations

import numpy as np
from pathlib import Path
import shutil

import pytest

import genepan
import stability_analysis


smoke = pytest.mark.smoke_tests


def make_engine() -> genepan.GenePan:
    return genepan.GenePan(".")


@smoke
def test_default_parameters_match_standard_setup() -> None:
    parameters = genepan.GenePanParameters()

    assert parameters.interval_margin_fpkm == 0.1
    assert parameters.detection_floor_fpkm == 0.0
    assert parameters.min_detected_fraction_normal == 0.0
    assert parameters.min_detected_fraction_tumor == 0.0
    assert parameters.pvalue_tumor == 0.1
    assert parameters.pvalue_normal == 0.05
    assert parameters.panel_tie_breaking_priority == "first"


@smoke
def test_coverage_thresholds_enforce_minimum_sample_support() -> None:
    engine = make_engine()

    normal_threshold = engine._minimum_significant_support_threshold(class_size=52, sample_size=551, pvalue=0.05)
    tumor_threshold = engine._minimum_significant_support_threshold(class_size=499, sample_size=551, pvalue=0.1)

    # The effective support condition is `count > threshold`, so these values
    # enforce at least 3/52 normal samples and at least 50/499 tumor samples.
    assert normal_threshold == 2
    assert tumor_threshold == 49


@smoke
def test_binomial_split_threshold_is_positive_lower_tail_cutoff() -> None:
    engine = make_engine()

    threshold = engine._minimum_significant_subsplit_threshold(10, pvalue=0.01)

    assert threshold >= 1
    assert threshold <= 10


def test_preprocessing_excludes_only_double_low_detection_genes() -> None:
    engine = make_engine()
    engine.parameters = genepan.GenePanParameters(
        detection_floor_fpkm=0.1,
        min_detected_fraction_normal=0.5,
        min_detected_fraction_tumor=0.5,
    )
    raw_values = np.array(
        [
            [0.2, 0.2, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    filtered_values, kept_ids, kept_names = engine._apply_manuscript_preprocessing_constraints(
        raw_values,
        ["normal_only", "tumor_only", "double_low"],
        ["NormalOnly", "TumorOnly", "DoubleLow"],
        normal_idx=np.array([0, 1]),
        tumor_idx=np.array([2, 3]),
    )

    assert kept_ids == ["normal_only", "tumor_only"]
    assert kept_names == ["NormalOnly", "TumorOnly"]
    assert filtered_values.shape == (4, 2)


@smoke
def test_single_side_up_family_detects_opposite_class_overexpression() -> None:
    engine = make_engine()
    values = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.8, 0.25],
            [0.9, 0.35],
        ],
        dtype=float,
    )
    limits = np.array([-2.0, -2.0], dtype=float)

    result = engine._discover_one_sided_dysregulation_panel(
        values,
        limits,
        concept_idx=np.array([0, 1]),
        rough_idx=np.array([2, 3]),
        threshold_count=0,
        gene_ids=["g0", "g1"],
        gene_names=["G0", "G1"],
        reference=np.array([1.0, 1.0]),
        panel_code=1,
        direction="up",
    )

    assert len(result.entries) == 1
    assert result.entries[0].gene_id == "g0"
    assert result.entries[0].threshold_low == 0.30000000000000004
    np.testing.assert_array_equal(result.masks[0], np.array([False, False, True, True]))


@smoke
def test_perfect_panel_can_define_the_complementary_concept() -> None:
    engine = make_engine()
    entries = [
        genepan.PanelEntry(0, "g0", "TumorDeregA", 1.0, 1, 0.1, None),
        genepan.PanelEntry(1, "g1", "TumorDeregB", 1.0, 1, 0.2, None),
        genepan.PanelEntry(2, "g2", "NormalOnly", 1.0, 1, 0.3, None),
    ]
    masks = np.array(
        [
            [False, False, True, False],
            [False, False, False, True],
            [True, True, False, False],
        ],
        dtype=bool,
    )

    selection = engine._construct_formal_concept_reduct(
        entries=entries,
        masks=masks,
        target_idx=np.array([0, 1]),
    )

    assert selection.covered_fraction == 1.0
    assert [entry.gene_name for entry in selection.selected_entries] == ["NormalOnly"]


@smoke
def test_outside_family_builds_two_sided_rule(monkeypatch) -> None:
    engine = make_engine()
    monkeypatch.setattr(engine, "_minimum_significant_subsplit_threshold", lambda size, pvalue: 0)

    values = np.array(
        [
            [0.0],
            [1.0],
            [0.5],
            [0.7],
            [-0.2],
            [1.8],
        ],
        dtype=float,
    )
    limits = np.array([-1.0], dtype=float)

    result = engine._discover_two_tailed_dysregulation_panel(
        values,
        limits,
        concept_idx=np.array([0, 1, 2, 3]),
        rough_idx=np.array([4, 5]),
        threshold_count=0,
        gene_ids=["g0"],
        gene_names=["G0"],
        reference=np.array([1.0]),
        panel_code=5,
    )

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert np.isclose(entry.threshold_low, -0.1)
    assert np.isclose(entry.threshold_high, 1.1)
    np.testing.assert_array_equal(result.masks[0], np.array([False, False, False, False, True, True]))


def test_inside_family_currently_returns_empty_stage(monkeypatch) -> None:
    engine = make_engine()
    monkeypatch.setattr(engine, "_minimum_significant_subsplit_threshold", lambda size, pvalue: 0)

    values = np.array(
        [
            [0.0],
            [0.2],
            [2.0],
            [2.2],
            [0.8],
            [1.4],
        ],
        dtype=float,
    )
    limits = np.array([-1.0], dtype=float)

    result = engine._discover_inside_dysregulation_panel(
        values,
        limits,
        concept_idx=np.array([0, 1, 2, 3]),
        rough_idx=np.array([4, 5]),
        threshold_count=0,
        gene_ids=["g0"],
        gene_names=["G0"],
        reference=np.array([1.0]),
        panel_code=7,
        delta_threshold=1.0,
    )

    assert result.entries == []
    assert result.masks.shape == (0, 0)


@smoke
def test_greedy_cover_prefers_rules_that_only_hit_target_samples() -> None:
    engine = make_engine()
    entries = [
        genepan.PanelEntry(0, "g0", "G0", 1.0, 1, 0.1, None),
        genepan.PanelEntry(1, "g1", "G1", 1.0, 1, 0.2, None),
        genepan.PanelEntry(2, "g2", "G2", 1.0, 1, 0.3, None),
    ]
    masks = np.array(
        [
            [True, True, False, False],
            [False, False, True, False],
            [True, False, False, True],  # invalid for target [0, 1, 2] because it hits sample 3
        ],
        dtype=bool,
    )

    selection = engine._construct_formal_concept_reduct(
        entries=entries,
        masks=masks,
        target_idx=np.array([0, 1, 2]),
    )

    assert selection.covered_fraction == 1.0
    assert selection.selected_positions == [0, 1]
    assert [entry.gene_id for entry in selection.selected_entries] == ["g0", "g1"]


@smoke
def test_panel_entries_to_frame_serializes_threshold_shapes() -> None:
    entries = [
        genepan.PanelEntry(0, "g0", "G0", 1.0, 1, 0.5, None),
        genepan.PanelEntry(1, "g1", "G1", 2.0, 7, 0.4, 1.2),
    ]

    frame = genepan.panel_entries_to_frame(entries)

    assert list(frame["threshold_kind"]) == ["single", "range"]
    assert list(frame["panel_family_name"]) == ["Only-T-above", "Only-T-inside"]
    assert frame.loc[1, "threshold_high"] == 1.2


@smoke
def test_query_gene_reports_nt_memberships_and_activation_frequencies() -> None:
    engine = make_engine()
    result = genepan.GenePanResult(
        all_entries=[
            genepan.PanelEntry(0, "g0", "GENE", 1.0, 1, 0.5, None),
            genepan.PanelEntry(0, "g0", "GENE", 1.0, 4, 0.3, None),
        ],
        tumor_entries=[
            genepan.PanelEntry(0, "g0", "GENE", 1.0, 1, 0.5, None),
        ],
        normal_entries=[
            genepan.PanelEntry(0, "g0", "GENE", 1.0, 4, 0.3, None),
        ],
        selected_tumor_mix=genepan.MixSelection([], [], [], 0.0, 0),
        selected_normal_mix=genepan.MixSelection([], [], [], 0.0, 0),
        selected_tumor_mix_most_redundant_then_first=genepan.MixSelection([], [], [], 0.0, 0),
        selected_normal_mix_most_redundant_then_first=genepan.MixSelection([], [], [], 0.0, 0),
        reference=np.array([1.0]),
        log2_values=np.array([[0.0], [0.6], [1.0], [0.2]], dtype=float),
        sample_types=["Solid Tissue Normal", "Primary Tumor", "Primary Tumor", "Solid Tissue Normal"],
        gene_ids=["g0"],
        gene_names=["GENE"],
        parameters=genepan.GenePanParameters(low_fpkm=0.1),
    )

    query = engine.query_gene(result, "GENE")

    assert query.gene_class == "NT-gene"
    assert len(query.memberships) == 2

    t_membership = next(item for item in query.memberships if item.gene_category == "T-gene")
    n_membership = next(item for item in query.memberships if item.gene_category == "N-gene")

    assert t_membership.panel_family_name == "Only-T-above"
    assert t_membership.t_activation_count == 2
    assert t_membership.t_activation_frequency == 1.0
    assert t_membership.n_activation_count == 0

    assert n_membership.panel_family_name == "Only-N-below"
    assert n_membership.n_activation_count == 2
    assert n_membership.n_activation_frequency == 1.0
    assert n_membership.t_activation_count == 0


@smoke
def test_query_gene_directly_checks_one_gene_without_full_run(monkeypatch) -> None:
    engine = make_engine()
    cohort = genepan.PreparedCohort(
        filtered_values=np.array([[1.0], [2.0], [2.2], [1.0]], dtype=float),
        reference=np.array([1.0]),
        log2_values=np.array([[1.0], [2.0], [2.2], [1.0]], dtype=float),
        limits=np.array([-2.0]),
        sample_types=["Solid Tissue Normal", "Primary Tumor", "Primary Tumor", "Solid Tissue Normal"],
        normal_idx=np.array([0, 3]),
        tumor_idx=np.array([1, 2]),
        gene_ids=["g0"],
        gene_names=["GENE"],
        tumor_threshold=0,
        normal_threshold=0,
    )
    monkeypatch.setattr(engine, "_prepare_analysis_cohort", lambda: cohort)

    query = engine.query_gene_directly("GENE")

    assert query.gene_class in {"T-gene", "NT-gene"}
    t_membership = next(membership for membership in query.memberships if membership.panel_family_name == "Only-T-above")
    assert t_membership.t_activation_count == 2
    assert t_membership.n_activation_count == 0


@smoke
def test_compute_specific_gene_pool_formats_requested_header(monkeypatch) -> None:
    engine = make_engine()
    cohort = genepan.PreparedCohort(
        filtered_values=np.array([[1.0]], dtype=float),
        reference=np.array([1.0]),
        log2_values=np.array([[1.0]], dtype=float),
        limits=np.array([-2.0]),
        sample_types=["Primary Tumor"],
        normal_idx=np.array([0]),
        tumor_idx=np.array([0]),
        gene_ids=["g0"],
        gene_names=["GENE"],
        tumor_threshold=0,
        normal_threshold=0,
    )
    monkeypatch.setattr(engine, "_prepare_analysis_cohort", lambda: cohort)
    monkeypatch.setattr(
        engine,
        "_build_family_stage",
        lambda cohort, panel_code: genepan.StageResult(
            entries=[genepan.PanelEntry(0, "g0", "GENE", 1.0, panel_code, 0.5, None)],
            masks=np.array([[True]], dtype=bool),
        ),
    )

    gene_set = engine.compute_specific_gene_pool("Only-T-above")

    assert gene_set.header == "1-gene T-gene pool (Only-T-above)"
    assert genepan._format_gene_set_output(gene_set) == "1-gene T-gene pool (Only-T-above)\nGENE"


def test_cchains_export_uses_full_gene_level_matrices(tmp_path: Path) -> None:
    result = genepan.GenePanResult(
        all_entries=[],
        tumor_entries=[],
        normal_entries=[],
        selected_tumor_mix=genepan.MixSelection([], [], [], 0.0, 0),
        selected_normal_mix=genepan.MixSelection([], [], [], 0.0, 0),
        selected_tumor_mix_most_redundant_then_first=genepan.MixSelection([], [], [], 0.0, 0),
        selected_normal_mix_most_redundant_then_first=genepan.MixSelection([], [], [], 0.0, 0),
        reference=np.array([], dtype=float),
        log2_values=np.zeros((0, 0), dtype=float),
        sample_types=[],
        gene_ids=[],
        gene_names=[],
        parameters=genepan.GenePanParameters(),
        t_gene_scan=genepan.GeneCategoryScan(
            gene_category="T-gene",
            gene_ids=["ENSGT000001.1", "ENSGT000002.1"],
            gene_names=["TG1", "TG2"],
            active_patterns=np.array([[1, 0], [0, 1]], dtype=np.uint8),
            binary_patterns=np.array([[1, 0], [0, 1]], dtype=np.uint8),
        ),
        n_gene_scan=genepan.GeneCategoryScan(
            gene_category="N-gene",
            gene_ids=["ENSGN000001.1", "ENSGN000002.1", "ENSGN000003.1"],
            gene_names=["NG1", "NG2", "NG3"],
            active_patterns=np.array([[1, 0], [1, 0], [0, 1]], dtype=np.uint8),
            binary_patterns=np.array([[0, 1], [0, 1], [1, 0]], dtype=np.uint8),
        ),
    )

    genepan._write_gene_category_sample_matrices(result, tmp_path)

    t_sample = np.loadtxt(tmp_path / "T_Network" / "data" / "sample.txt", dtype=np.uint8)
    n_sample = np.loadtxt(tmp_path / "N_Network" / "data" / "sample.txt", dtype=np.uint8)
    t_names = (tmp_path / "T_Network" / "data" / "names.csv").read_text(encoding="utf-8").splitlines()
    n_names = (tmp_path / "N_Network" / "data" / "names.csv").read_text(encoding="utf-8").splitlines()

    np.testing.assert_array_equal(t_sample, result.t_gene_scan.binary_patterns)
    np.testing.assert_array_equal(n_sample, result.n_gene_scan.binary_patterns)
    assert t_names == ["ENSGT000001", "ENSGT000002"]
    assert n_names == ["ENSGN000001", "ENSGN000002", "ENSGN000003"]


def test_prepare_analysis_cohort_reuses_in_memory_cache(monkeypatch) -> None:
    engine = make_engine()
    cache_dir = Path("tests") / "cache_runtime_in_memory"
    shutil.rmtree(cache_dir, ignore_errors=True)
    engine.cache_dir = cache_dir
    cohort = genepan.PreparedCohort(
        filtered_values=np.array([[1.0]], dtype=float),
        reference=np.array([1.0]),
        log2_values=np.array([[0.0]], dtype=float),
        limits=np.array([-1.0]),
        sample_types=["Solid Tissue Normal"],
        normal_idx=np.array([0]),
        tumor_idx=np.array([0]),
        gene_ids=["g0"],
        gene_names=["GENE"],
        tumor_threshold=49,
        normal_threshold=2,
    )
    calls = {"count": 0}

    def fake_compute() -> genepan.PreparedCohort:
        calls["count"] += 1
        return cohort

    monkeypatch.setattr(engine, "_compute_prepared_analysis_cohort", fake_compute)
    monkeypatch.setattr(engine, "_cohort_cache_signature", lambda: "memory-cache-test")

    first = engine._prepare_analysis_cohort()
    second = engine._prepare_analysis_cohort()

    assert calls["count"] == 1
    assert first is second


def test_prepare_analysis_cohort_loads_binary_cache(monkeypatch) -> None:
    engine_writer = make_engine()
    cache_dir = Path("tests") / "cache_runtime_disk"
    shutil.rmtree(cache_dir, ignore_errors=True)
    engine_writer.cache_dir = cache_dir
    cohort = genepan.PreparedCohort(
        filtered_values=np.array([[1.0, 2.0]], dtype=float),
        reference=np.array([1.0, 2.0]),
        log2_values=np.array([[0.0, 1.0]], dtype=float),
        limits=np.array([-1.0, -2.0]),
        sample_types=["Solid Tissue Normal"],
        normal_idx=np.array([0]),
        tumor_idx=np.array([0]),
        gene_ids=["g0", "g1"],
        gene_names=["G0", "G1"],
        tumor_threshold=49,
        normal_threshold=2,
    )
    monkeypatch.setattr(engine_writer, "_cohort_cache_signature", lambda: "disk-cache-test")
    engine_writer._write_prepared_cohort_to_binary_cache(cohort)

    engine_reader = make_engine()
    engine_reader.cache_dir = cache_dir
    monkeypatch.setattr(engine_reader, "_cohort_cache_signature", lambda: "disk-cache-test")
    monkeypatch.setattr(
        engine_reader,
        "_compute_prepared_analysis_cohort",
        lambda: (_ for _ in ()).throw(AssertionError("binary cache should have been used first")),
    )

    loaded = engine_reader._prepare_analysis_cohort()

    assert loaded.sample_types == cohort.sample_types
    assert loaded.gene_ids == cohort.gene_ids
    np.testing.assert_allclose(loaded.log2_values, cohort.log2_values)


def test_packed_loevinger_matches_dense_binary_matrix() -> None:
    active_patterns = np.array(
        [
            [1, 0, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    dense_keys = _dense_loevinger_pair_keys(active_patterns, threshold=0.5)
    packed_keys = stability_analysis._loevinger_pair_keys(active_patterns, threshold=0.5, block_size=2)

    assert packed_keys == dense_keys


@smoke
def test_kde_sampler_generates_continuous_values_for_variable_genes() -> None:
    class_values = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [8.0, 0.0],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(1234)

    sampled = genepan.GenePan._sample_synthetic_profiles(
        class_values=class_values,
        synthetic_count=16,
        rng=rng,
    )

    observed_first_gene = set(class_values[:, 0].tolist())
    assert any(float(value) not in observed_first_gene for value in sampled[:, 0])
    assert np.all(sampled[:, 1] == 0.0)


@smoke
def test_kde_sampler_keeps_constant_genes_constant() -> None:
    class_values = np.array(
        [
            [3.5, 7.0],
            [3.5, 7.0],
            [3.5, 7.0],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(7)

    sampled = genepan.GenePan._sample_synthetic_profiles(
        class_values=class_values,
        synthetic_count=8,
        rng=rng,
    )

    np.testing.assert_allclose(sampled[:, 0], 3.5)
    np.testing.assert_allclose(sampled[:, 1], 7.0)


def test_fast_family_stage_matches_exact_builder_on_toy_cohort() -> None:
    engine = make_engine()
    cohort = genepan.PreparedCohort(
        filtered_values=np.array(
            [
                [0.2, 0.6, 1.0],
                [0.3, 0.7, 1.1],
                [1.0, 0.4, 1.2],
                [1.1, 0.3, 1.3],
            ],
            dtype=float,
        ),
        reference=np.array([1.0, 1.0, 1.0]),
        log2_values=np.array(
            [
                [0.2, 0.6, 1.0],
                [0.3, 0.7, 1.1],
                [1.0, 0.4, 1.2],
                [1.1, 0.3, 1.3],
            ],
            dtype=float,
        ),
        limits=np.array([-2.0, -2.0, -2.0]),
        sample_types=["Solid Tissue Normal", "Solid Tissue Normal", "Primary Tumor", "Primary Tumor"],
        normal_idx=np.array([0, 1]),
        tumor_idx=np.array([2, 3]),
        gene_ids=["g0", "g1", "g2"],
        gene_names=["G0", "G1", "G2"],
        tumor_threshold=0,
        normal_threshold=0,
    )

    for panel_code in [1, 2, 3, 4, 5, 6]:
        fast = engine._build_family_stage(cohort=cohort, panel_code=panel_code)
        exact = engine._build_family_stage_for_single_gene(
            cohort=cohort,
            panel_code=panel_code,
            values=cohort.filtered_values,
            limits=cohort.limits,
            gene_ids=cohort.gene_ids,
            gene_names=cohort.gene_names,
            reference=cohort.reference,
        )

        assert [entry.gene_id for entry in fast.entries] == [entry.gene_id for entry in exact.entries]
        assert [entry.threshold_low for entry in fast.entries] == [entry.threshold_low for entry in exact.entries]
        assert [entry.threshold_high for entry in fast.entries] == [entry.threshold_high for entry in exact.entries]
        np.testing.assert_array_equal(fast.masks, exact.masks)


def test_build_gene_category_scan_uses_margin_free_discretization() -> None:
    engine = make_engine()
    cohort = genepan.PreparedCohort(
        filtered_values=np.array(
            [
                [0.20],
                [0.55],
            ],
            dtype=float,
        ),
        reference=np.array([1.0]),
        log2_values=np.array(
            [
                [0.20],
                [0.55],
            ],
            dtype=float,
        ),
        limits=np.array([-2.0]),
        sample_types=["Solid Tissue Normal", "Primary Tumor"],
        normal_idx=np.array([0]),
        tumor_idx=np.array([1]),
        gene_ids=["g0"],
        gene_names=["GENE"],
        tumor_threshold=0,
        normal_threshold=0,
    )
    stage = genepan.StageResult(
        entries=[genepan.PanelEntry(0, "g0", "GENE", 1.0, 1, 0.3, None)],
        masks=np.array([[False, True]], dtype=bool),
    )
    stages = {
        1: stage,
        2: genepan.StageResult([], np.zeros((0, 2), dtype=bool)),
        3: genepan.StageResult([], np.zeros((0, 2), dtype=bool)),
        4: genepan.StageResult([], np.zeros((0, 2), dtype=bool)),
        5: genepan.StageResult([], np.zeros((0, 2), dtype=bool)),
        6: genepan.StageResult([], np.zeros((0, 2), dtype=bool)),
        7: genepan.StageResult([], np.zeros((0, 2), dtype=bool)),
        8: genepan.StageResult([], np.zeros((0, 2), dtype=bool)),
    }

    scan = engine.build_gene_category_scan(
        cohort=cohort,
        gene_category="T-gene",
        stages=stages,
    )

    # Discovery threshold is 0.3 (= normal max + 0.1). The discretization
    # should instead use the true exclusion boundary 0.2 (= normal max).
    np.testing.assert_array_equal(scan.active_patterns, np.array([[1]], dtype=np.uint8))


@smoke
def test_stability_count_parser_and_balanced_total_mode() -> None:
    counts = stability_analysis._parse_counts("0,10,20")
    assert counts == [0, 10, 20]

    config = stability_analysis.StabilityConfig(
        cohort=stability_analysis.CohortConfig(
            source="pca",
            action="augment",
            side="balanced",
            counts=(25,),
            balanced_count_mode="total",
        ),
        analysis=stability_analysis.AnalysisConfig(analysis="gene_set_overlap", gene_category="both"),
        execution=stability_analysis.ExecutionConfig(replicates=1, seed=1, seed_stride=1, max_workers=1, resume=False),
        outputs=stability_analysis.OutputConfig(output_dir=Path("."), figure_dir=Path("."), write_figures=False),
        base_dir=Path("."),
    )

    assert stability_analysis._synthetic_counts(config, 25) == (12, 13)


@smoke
def test_stability_set_metrics_reports_overlap_fractions() -> None:
    metrics = stability_analysis._set_metrics(
        "t_gene",
        reference={"a", "b", "c"},
        observed={"b", "c", "d"},
    )

    assert metrics["t_gene_reference_count"] == 3
    assert metrics["t_gene_observed_count"] == 3
    assert metrics["t_gene_shared_count"] == 2
    assert metrics["t_gene_recall"] == 2 / 3
    assert metrics["t_gene_precision"] == 2 / 3
    assert metrics["t_gene_jaccard"] == 2 / 4


def test_stability_loevinger_edge_metrics_uses_overlap_fractions(monkeypatch) -> None:
    context = type(
        "Context",
        (),
        {
            "reference_edge_sets": {"T-gene": {1, 2, 3}},
            "engine": type("Engine", (), {"build_gene_category_scan": lambda self, cohort, gene_category: "scan"})(),
        },
    )()
    config = stability_analysis.StabilityConfig(
        cohort=stability_analysis.CohortConfig(source="real", action="subsample", side="tumor", counts=(2,)),
        analysis=stability_analysis.AnalysisConfig(
            analysis="loevinger_edges",
            gene_category="T-gene",
            loevinger_threshold=0.5,
        ),
        execution=stability_analysis.ExecutionConfig(replicates=1, seed=1, seed_stride=1, max_workers=1, resume=False),
        outputs=stability_analysis.OutputConfig(output_dir=Path("."), figure_dir=Path("."), write_figures=False),
        base_dir=Path("."),
    )
    monkeypatch.setattr(stability_analysis, "_loevinger_pair_keys", lambda active_patterns, threshold: {2, 3, 4, 5})

    class Scan:
        gene_ids = ["g0", "g1", "g2"]
        active_patterns = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.uint8)

    context.engine.build_gene_category_scan = lambda cohort, gene_category: Scan()

    metrics = stability_analysis._loevinger_edge_metrics(config, context, cohort=None)

    assert metrics["t_gdn_reference_edge_count"] == 3
    assert metrics["t_gdn_observed_edge_count"] == 4
    assert metrics["t_gdn_shared_edge_count"] == 2
    assert metrics["t_gdn_edge_recall"] == 2 / 3
    assert metrics["t_gdn_edge_precision"] == 2 / 4
    assert metrics["t_gdn_edge_jaccard"] == 2 / 5


def _dense_loevinger_pair_keys(active_patterns: np.ndarray, *, threshold: float) -> set[int]:
    gene_count, sample_count = active_patterns.shape
    if gene_count <= 1 or sample_count == 0:
        return set()

    active = active_patterns.astype(np.uint8, copy=False)
    activation_counts = active.sum(axis=1).astype(np.float64)
    activation_frequencies = activation_counts / float(sample_count)
    keys: set[int] = set()
    for i in range(gene_count):
        if activation_counts[i] == 0:
            continue
        for j in range(gene_count):
            if i == j or activation_frequencies[j] >= 1.0:
                continue
            coactivation = np.logical_and(active[i], active[j]).sum()
            conditional = coactivation / activation_counts[i]
            h_value = (conditional - activation_frequencies[j]) / (1.0 - activation_frequencies[j])
            if h_value > threshold:
                keys.add(i * gene_count + j)
    return keys
