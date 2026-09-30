"""
Unit Tests for Day 2 Phase 3: Model Evaluation Module.
=====================================================
Verifies:
1. compute_metrics mathematical accuracy.
2. format_comparison difference and percentage calculations.
3. ModelEvaluator test journey extraction and isolation.
4. Correct structure and schema in generated model_metrics.json.
"""

import json
import pytest
import numpy as np
from pathlib import Path

from backend.ml.evaluate_model import (
    compute_metrics,
    format_comparison,
    ModelEvaluator,
    DEFAULT_METRICS_OUTPUT_PATH,
)


def test_compute_metrics():
    """Verify MAE and RMSE calculations on known vectors."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 16.0, 35.0])

    # Errors: [2, 4, 5] -> MAE = (2 + 4 + 5) / 3 = 3.667
    # Squared errors: [4, 16, 25] -> MSE = 45 / 3 = 15 -> RMSE = sqrt(15) = 3.873
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == 3.667
    assert metrics["rmse"] == 3.873

    # Empty array guard
    empty_metrics = compute_metrics(np.array([]), np.array([]))
    assert empty_metrics["mae"] == 0.0
    assert empty_metrics["rmse"] == 0.0


def test_format_comparison_ml_wins():
    """Verify format_comparison when ML has lower error than Baseline."""
    b_m = {"mae": 50.0, "rmse": 70.0}
    m_m = {"mae": 10.0, "rmse": 15.0}

    comp = format_comparison(b_m, m_m, sample_count=100)
    assert comp["sample_count"] == 100
    assert comp["baseline_mae"] == 50.0
    assert comp["ml_mae"] == 10.0
    assert comp["absolute_diff_mae"] == -40.0
    assert comp["percentage_improvement"] == 80.0
    assert comp["winner"] == "ML"


def test_format_comparison_baseline_wins():
    """Verify format_comparison when Baseline has lower error than ML."""
    b_m = {"mae": 20.0, "rmse": 30.0}
    m_m = {"mae": 30.0, "rmse": 45.0}

    comp = format_comparison(b_m, m_m, sample_count=50)
    assert comp["sample_count"] == 50
    assert comp["baseline_mae"] == 20.0
    assert comp["ml_mae"] == 30.0
    assert comp["absolute_diff_mae"] == 10.0
    assert comp["percentage_improvement"] == -50.0
    assert comp["winner"] == "BASELINE"


def test_extract_test_journey_ids():
    """Verify ModelEvaluator extracts 40 test journeys matching 20% test size."""
    evaluator = ModelEvaluator(test_size=0.20, random_seed=42)
    test_ids = evaluator.extract_test_journey_ids()
    assert len(test_ids) == 40
    assert all(isinstance(jid, str) and jid.startswith("J_") for jid in test_ids)


def test_model_metrics_json_schema():
    """Verify that data/processed/model_metrics.json exists and conforms to required schema."""
    assert DEFAULT_METRICS_OUTPUT_PATH.exists()

    with open(DEFAULT_METRICS_OUTPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "summary" in data
    assert data["summary"]["total_test_journeys"] == 40
    assert data["summary"]["total_evaluated_samples"] > 0

    assert "by_horizon" in data
    for horizon in ["1_station_ahead", "3_stations_ahead", "5_stations_ahead"]:
        assert horizon in data["by_horizon"]
        h_data = data["by_horizon"][horizon]
        for field in ["sample_count", "baseline_mae", "baseline_rmse", "ml_mae", "ml_rmse", "percentage_improvement", "winner"]:
            assert field in h_data

    assert "by_disruption" in data
    for d in ["no_disruption", "with_disruption"]:
        assert d in data["by_disruption"]

    assert "overall" in data
    assert "winner" in data["overall"]


def test_load_latest_metrics_success():
    """Verify load_latest_metrics loads real metrics and metadata successfully."""
    from backend.ml.evaluate_model import load_latest_metrics
    data = load_latest_metrics()
    assert data["status"] == "AVAILABLE"
    assert data["is_available"] is True
    assert data["model_name"] == "Dynamic Train ETA XGBoost Regressor"
    assert data["model_version"] == "1.0.0"
    assert data["total_training_samples"] == 4915
    assert data["test_samples"] == 2430
    assert data["baseline_mae"] == pytest.approx(74.106, 0.01)
    assert data["ml_mae"] == pytest.approx(28.275, 0.01)
    assert "1_station_ahead" in data["metrics_by_horizon"]
    assert "no_disruption" in data["metrics_by_disruption_status"]


def test_load_latest_metrics_missing_file():
    """Verify load_latest_metrics handles missing file without crashing."""
    from backend.ml.evaluate_model import load_latest_metrics
    data = load_latest_metrics(metrics_path=Path("/invalid/path.json"))
    assert data["status"] == "UNAVAILABLE"
    assert data["is_available"] is False
    assert data["baseline_mae"] is None
    assert "not found" in data["message"].lower()

