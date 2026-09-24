"""Tests for table and fuelling metrics."""

import numpy as np
import pytest

from vetuner.drive_cycle import generate_drive_cycle
from vetuner.metrics import evaluate_afr, evaluate_table
from vetuner.ve_surface import BASE_MAP_SURFACE, TRUE_SURFACE, ve_table


def test_identical_tables_score_zero():
    table = ve_table()
    metrics = evaluate_table(table, table)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.max_abs_error == pytest.approx(0.0)


def test_uniform_offset_gives_matching_rmse():
    truth = ve_table()
    metrics = evaluate_table(truth + 0.05, truth)
    assert metrics.rmse == pytest.approx(5.0)
    assert metrics.mean_abs_error == pytest.approx(5.0)


def test_coverage_split_reports_both_groups():
    truth = ve_table()
    estimate = truth.copy()
    estimate[0, :] += 0.10

    counts = np.ones(truth.shape, dtype=int)
    counts[0, :] = 0

    metrics = evaluate_table(estimate, truth, counts)
    assert metrics.rmse_visited == pytest.approx(0.0)
    assert metrics.rmse_unvisited == pytest.approx(10.0)
    assert metrics.n_visited == truth.size - truth.shape[1]


def test_true_table_achieves_near_target_fuelling():
    cycle = generate_drive_cycle()
    metrics = evaluate_afr(ve_table(params=TRUE_SURFACE), cycle)
    assert metrics.mean_abs_pct < 0.25
    assert metrics.fraction_within_2pct > 0.99


def test_base_map_fuelling_error_is_substantial():
    cycle = generate_drive_cycle()
    metrics = evaluate_afr(ve_table(params=BASE_MAP_SURFACE), cycle)
    assert metrics.mean_abs_pct > 5.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_table(np.ones((3, 3)), np.ones((4, 4)))