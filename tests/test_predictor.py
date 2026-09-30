"""
Unit Tests for Day 2 Phase 2: XGBoost ETA Model and Predictor Service.
======================================================================
Verifies:
1. Model metadata serialization and schema compliance.
2. Journey-level train/test partitioning with zero data leakage.
3. Dataset validation guards.
4. End-to-end model training execution.
5. Prediction service inference on TrainFeatures, dicts, and batches.
6. Non-negative output clamping and physical consistency.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from backend.ml.model_metadata import (
    ModelMetadata,
    EvaluationMetrics,
    DatasetSplitInfo,
    PRD_FEATURE_NAMES,
)
from backend.ml.train_model import (
    validate_dataset,
    encode_features,
    split_by_journey,
    train_eta_model,
    TARGET_COLUMN,
    JOURNEY_ID_COLUMN,
)
from backend.ml.predictor import ETAPredictor, DetailedETAPrediction
from backend.features.feature_builder import TrainFeatures


@pytest.fixture
def sample_dataset_df():
    """Generates a synthetic DataFrame with 4 journeys and known values for testing."""
    rows = []
    for j_idx in range(4):
        journey_id = f"J_TEST_{j_idx}"
        for step in range(10):
            dist = 200.0 - (step * 20.0)
            sched = 120.0 - (step * 12.0)
            actual = sched + (j_idx * 5.0)  # Predictable target
            rows.append({
                JOURNEY_ID_COLUMN: journey_id,
                "train_id": 12028,
                "route_id": 1,
                "station": "NDLS",
                "target_station": "CNB",
                "timestamp": "2026-10-01T10:00:00+00:00",
                "distance_to_go_km": dist,
                "scheduled_time_to_go_min": sched,
                "num_intermediate_halts": 1,
                "current_delay_min": 10.0,
                "delay_trend_3pt": 1.5,
                "active_speed_restriction_flag": 0,
                "congestion_score_downstream": 0.1,
                "hist_avg_delay_this_section": 0.0,
                "hist_recovery_rate_section": 0.0,
                "weather_flag": 0,
                "day_type": 0,
                TARGET_COLUMN: actual,
                "disruption_present": 0,
                "is_synthetic": True,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_train_features():
    """Returns a valid TrainFeatures instance for inference tests."""
    return TrainFeatures(
        train_number="12302",
        target_station_code="CNB",
        distance_to_go_km=145.0,
        scheduled_time_to_go_min=85.0,
        num_intermediate_halts=0,
        current_delay_min=12.0,
        delay_trend_3pt=1.0,
        active_speed_restriction_flag=0,
        congestion_score_downstream=0.0,
        hist_avg_delay_this_section=0.0,
        hist_recovery_rate_section=0.0,
        weather_flag=0,
        day_type=0,
    )


def test_metadata_serialization(tmp_path):
    """Verify ModelMetadata can serialize to JSON and deserialize back with identical fields."""
    meta_path = tmp_path / "test_meta.json"
    metrics = EvaluationMetrics(mae=5.2, rmse=7.8, r2_score=0.97, baseline_mae_comparison=28.4)
    split_info = DatasetSplitInfo(
        total_samples=100,
        train_samples=80,
        test_samples=20,
        total_journeys=10,
        train_journeys=8,
        test_journeys=2,
    )
    metadata = ModelMetadata(
        dataset_info=split_info,
        metrics=metrics,
        hyperparameters={"max_depth": 5, "learning_rate": 0.05},
        feature_importance_gain={"distance_to_go_km": 50000.0},
    )

    metadata.save(meta_path)
    assert meta_path.exists()

    loaded = ModelMetadata.load(meta_path)
    assert loaded.model_name == metadata.model_name
    assert loaded.metrics.mae == 5.2
    assert loaded.metrics.rmse == 7.8
    assert loaded.dataset_info.train_samples == 80
    assert loaded.features == PRD_FEATURE_NAMES


def test_journey_level_split_isolation(sample_dataset_df):
    """Verify that split_by_journey strictly isolates journeys between train and test sets."""
    train_df, test_df, train_j, test_j = split_by_journey(
        sample_dataset_df, test_size=0.25, random_seed=42
    )

    # 1. Zero journey overlap
    train_journeys = set(train_df[JOURNEY_ID_COLUMN].unique())
    test_journeys = set(test_df[JOURNEY_ID_COLUMN].unique())
    assert train_journeys.isdisjoint(test_journeys), "Leakage detected between train and test journeys"

    # 2. Complete partition of rows
    assert len(train_df) + len(test_df) == len(sample_dataset_df)
    assert len(train_j) + len(test_j) == 4
    assert len(test_j) == 1  # 25% of 4 journeys = 1 journey


def test_validation_guards(sample_dataset_df):
    """Verify dataset validation rejects missing columns, nulls, and empty inputs."""
    # 1. Empty dataframe
    with pytest.raises(ValueError, match="empty"):
        validate_dataset(pd.DataFrame())

    # 2. Missing journey_id
    bad_df = sample_dataset_df.drop(columns=[JOURNEY_ID_COLUMN])
    with pytest.raises(ValueError, match="Missing required grouping column"):
        validate_dataset(bad_df)

    # 3. Missing target column
    bad_df2 = sample_dataset_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="Missing target column"):
        validate_dataset(bad_df2)

    # 4. Missing PRD feature
    bad_df3 = sample_dataset_df.drop(columns=["distance_to_go_km"])
    with pytest.raises(ValueError, match="missing required PRD feature"):
        validate_dataset(bad_df3)

    # 5. Null values
    bad_df4 = sample_dataset_df.copy()
    bad_df4.loc[0, "current_delay_min"] = np.nan
    with pytest.raises(ValueError, match="null/NaN"):
        validate_dataset(bad_df4)


def test_end_to_end_model_training(sample_dataset_df, tmp_path):
    """Verify end-to-end model training produces artifacts, metadata, and valid metrics."""
    data_path = tmp_path / "toy_data.csv"
    model_path = tmp_path / "toy_model.json"
    meta_path = tmp_path / "toy_meta.json"

    sample_dataset_df.to_csv(data_path, index=False)

    booster, metadata = train_eta_model(
        dataset_path=data_path,
        model_output_path=model_path,
        metadata_output_path=meta_path,
        test_size=0.25,
        random_seed=42,
        hyperparameters={"n_estimators": 20, "max_depth": 3},
    )

    assert model_path.exists()
    assert meta_path.exists()
    assert metadata.metrics.mae >= 0.0
    assert metadata.metrics.rmse >= 0.0
    assert metadata.dataset_info.test_journeys == 1
    assert metadata.dataset_info.train_journeys == 3


def test_predictor_inference_with_train_features(sample_train_features):
    """Verify ETAPredictor loads trained production model and produces valid predictions."""
    predictor = ETAPredictor()

    minutes = predictor.predict(sample_train_features)
    assert isinstance(minutes, float)
    assert minutes >= 0.0
    # Sanity: scheduled time is 85 min, distance is 145 km -> remaining time should be reasonable
    assert 20.0 <= minutes <= 200.0


def test_predictor_inference_with_dict(sample_train_features):
    """Verify ETAPredictor can ingest raw dictionary inputs."""
    predictor = ETAPredictor()
    feat_dict = sample_train_features.to_model_input_dict()

    minutes = predictor.predict(feat_dict)
    assert isinstance(minutes, float)
    assert minutes >= 0.0


def test_predictor_detailed_response(sample_train_features):
    """Verify predict_detailed returns complete typed structure with model info."""
    predictor = ETAPredictor()
    detailed = predictor.predict_detailed(sample_train_features)

    assert isinstance(detailed, DetailedETAPrediction)
    assert detailed.minutes_to_next_station >= 0.0
    assert detailed.train_number == "12302"
    assert detailed.target_station_code == "CNB"
    assert detailed.model_name == "Dynamic Train ETA XGBoost Regressor"
    assert detailed.is_synthetic_model is True
    assert "distance_to_go_km" in detailed.features_used


def test_predictor_batch_dataframe(sample_train_features):
    """Verify vectorized predict_batch produces predictions matching row count."""
    predictor = ETAPredictor()

    feat_dict = sample_train_features.to_model_input_dict()
    df = pd.DataFrame([feat_dict, feat_dict, feat_dict])

    preds = predictor.predict_batch(df)
    assert len(preds) == 3
    assert all(p >= 0.0 for p in preds)
    assert preds[0] == preds[1] == preds[2]


def test_predictor_missing_feature_guard():
    """Verify ETAPredictor raises ValueError when required features are missing."""
    predictor = ETAPredictor()
    incomplete_dict = {"distance_to_go_km": 100.0}

    with pytest.raises(ValueError, match="Missing required feature keys"):
        predictor.predict(incomplete_dict)
