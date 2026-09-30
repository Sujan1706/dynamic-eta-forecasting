"""
Training Pipeline for Dynamic Train ETA XGBoost Regression Model.
================================================================
Trains an XGBoost gradient-boosted regression model on the synthetic train-running dataset.
Predicts remaining travel time (minutes_to_next_station) using the 11 PRD features.

Enforces:
1. Journey-level train/test partitioning to prevent checkpoint data leakage.
2. Complete column and schema validation.
3. Deterministic reproducibility via fixed random seed.
4. Evaluation with MAE, RMSE, and comparison to naive schedule baseline.
5. Export of trained model artifact and model metadata.
"""

import os
import argparse
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import xgboost as xgb

from backend.ml.model_metadata import (
    ModelMetadata,
    EvaluationMetrics,
    DatasetSplitInfo,
    PRD_FEATURE_NAMES,
    DATA_DISCLAIMER,
)

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed"
    / "synthetic_train_eta_dataset.csv"
)
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "eta_xgboost_model.json"
DEFAULT_METADATA_PATH = DEFAULT_MODELS_DIR / "eta_xgboost_metadata.json"

TARGET_COLUMN = "actual_remaining_minutes"
JOURNEY_ID_COLUMN = "journey_id"


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validates dataset integrity, required columns, and missing values.
    Raises ValueError if requirements are not met.
    """
    if df.empty:
        raise ValueError("Dataset is empty. Cannot train model on empty DataFrame.")

    # 1. Verify journey grouping column
    if JOURNEY_ID_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required grouping column '{JOURNEY_ID_COLUMN}' for journey-level splitting."
        )

    # 2. Verify target column
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}' in dataset.")

    # 3. Verify all 11 PRD features
    missing_features = [f for f in PRD_FEATURE_NAMES if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Dataset is missing required PRD feature columns: {missing_features}"
        )

    # 4. Check for null or NaN values
    null_counts = df[PRD_FEATURE_NAMES + [TARGET_COLUMN, JOURNEY_ID_COLUMN]].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise ValueError(
            f"Found null/NaN values in training columns: {cols_with_nulls.to_dict()}"
        )

    # 5. Check target range validity
    if (df[TARGET_COLUMN] <= 0).any():
        non_positive_count = (df[TARGET_COLUMN] <= 0).sum()
        raise ValueError(
            f"Found {non_positive_count} non-positive values in target '{TARGET_COLUMN}'."
        )


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and encodes categorical and numerical features.
    Ensures correct data types for XGBoost ingestion.
    """
    df_copy = df.copy()

    # Validate day_type is categorical integer (0=Weekday, 1=Weekend, 2=Holiday)
    if "day_type" in df_copy.columns:
        valid_day_types = {0, 1, 2}
        unique_types = set(df_copy["day_type"].unique())
        if not unique_types.issubset(valid_day_types):
            raise ValueError(
                f"Invalid day_type values found: {unique_types - valid_day_types}. Expected {valid_day_types}."
            )
        df_copy["day_type"] = df_copy["day_type"].astype(int)

    # Ensure binary flags are integers
    for flag_col in ["active_speed_restriction_flag", "weather_flag", "num_intermediate_halts"]:
        if flag_col in df_copy.columns:
            df_copy[flag_col] = df_copy[flag_col].astype(int)

    # Ensure continuous features are floats
    for cont_col in [
        "distance_to_go_km",
        "scheduled_time_to_go_min",
        "current_delay_min",
        "delay_trend_3pt",
        "congestion_score_downstream",
        "hist_avg_delay_this_section",
        "hist_recovery_rate_section",
    ]:
        if cont_col in df_copy.columns:
            df_copy[cont_col] = df_copy[cont_col].astype(float)

    return df_copy


def split_by_journey(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Partitions dataset into train and test splits at the journey level.
    Guarantees that all checkpoints from any given journey stay strictly within
    either the train set or test set, preventing time-series leakage.
    """
    # Extract unique journeys as a Python list
    unique_journeys = sorted(list(df[JOURNEY_ID_COLUMN].unique()))
    if len(unique_journeys) < 2:
        raise ValueError(
            f"Cannot split dataset with only {len(unique_journeys)} journey(s)."
        )

    # Deterministic shuffle using fixed seed
    rng = random.Random(random_seed)
    shuffled_journeys = list(unique_journeys)
    rng.shuffle(shuffled_journeys)

    num_test_journeys = max(1, int(round(len(shuffled_journeys) * test_size)))
    test_journey_ids = set(shuffled_journeys[:num_test_journeys])
    train_journey_ids = set(shuffled_journeys[num_test_journeys:])

    # Guarantee zero journey leakage
    assert train_journey_ids.isdisjoint(test_journey_ids), "Journey split leakage detected!"

    train_df = df[df[JOURNEY_ID_COLUMN].isin(train_journey_ids)].copy()
    test_df = df[df[JOURNEY_ID_COLUMN].isin(test_journey_ids)].copy()

    return train_df, test_df, sorted(list(train_journey_ids)), sorted(list(test_journey_ids))


def train_eta_model(
    dataset_path: Optional[Path | str] = None,
    model_output_path: Optional[Path | str] = None,
    metadata_output_path: Optional[Path | str] = None,
    test_size: float = 0.20,
    random_seed: int = 42,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Tuple[xgb.Booster, ModelMetadata]:
    """
    End-to-end training routine: loads data, splits by journey, trains XGBoost,
    evaluates on test journeys, and saves model artifacts + metadata.
    """
    data_file = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    model_file = Path(model_output_path) if model_output_path else DEFAULT_MODEL_PATH
    meta_file = Path(metadata_output_path) if metadata_output_path else DEFAULT_METADATA_PATH

    if not data_file.exists():
        raise FileNotFoundError(f"Training dataset not found at: {data_file}")

    # 1. Load dataset
    df = pd.read_csv(data_file)
    validate_dataset(df)
    df = encode_features(df)

    # 2. Split train/test at journey level
    train_df, test_df, train_j_ids, test_j_ids = split_by_journey(
        df, test_size=test_size, random_seed=random_seed
    )

    X_train = train_df[PRD_FEATURE_NAMES]
    # Residual Target Formulation: Net delay deviation from timetable schedule
    # actual_remaining_minutes = scheduled_time_to_go_min + residual_delay
    y_train_residual = (train_df[TARGET_COLUMN] - train_df["scheduled_time_to_go_min"]).values.astype(float)

    X_test = test_df[PRD_FEATURE_NAMES]
    y_test_residual = (test_df[TARGET_COLUMN] - test_df["scheduled_time_to_go_min"]).values.astype(float)
    y_test_actual = test_df[TARGET_COLUMN].values.astype(float)

    # 3. Create XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train_residual, feature_names=PRD_FEATURE_NAMES)
    dtest = xgb.DMatrix(X_test, label=y_test_residual, feature_names=PRD_FEATURE_NAMES)

    # 4. Default hyperparameters optimized for tabular ETA forecasting
    default_params: Dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": ["mae", "rmse"],
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "gamma": 0.1,
        "seed": random_seed,
    }
    if hyperparameters:
        default_params.update(hyperparameters)

    num_boost_round = int(default_params.pop("n_estimators", 200))

    # 5. Train model
    booster = xgb.train(
        params=default_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=False,
    )

    # 6. Evaluate predictions on test set: reconstructed actual remaining minutes
    raw_residual_preds = booster.predict(dtest)
    preds = np.maximum(0.0, test_df["scheduled_time_to_go_min"].values + raw_residual_preds)

    mae = float(np.mean(np.abs(y_test_actual - preds)))
    rmse = float(np.sqrt(np.mean((y_test_actual - preds) ** 2)))

    # Compute R^2 score
    ss_total = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
    ss_res = np.sum((y_test_actual - preds) ** 2)
    r2_score = float(1.0 - (ss_res / ss_total)) if ss_total > 0 else 0.0

    # Benchmark comparison: Naive timetable + current delay baseline on same test set
    if "scheduled_time_to_go_min" in test_df.columns and "current_delay_min" in test_df.columns:
        naive_preds = test_df["scheduled_time_to_go_min"] + test_df["current_delay_min"]
        baseline_mae = float(np.mean(np.abs(y_test_actual - naive_preds.values)))
    else:
        baseline_mae = None

    # 7. Extract feature importance
    importance_gain = {
        feat: float(booster.get_score(importance_type="gain").get(feat, 0.0))
        for feat in PRD_FEATURE_NAMES
    }
    importance_weight = {
        feat: float(booster.get_score(importance_type="weight").get(feat, 0.0))
        for feat in PRD_FEATURE_NAMES
    }

    # Sort importances descending by gain
    importance_gain = dict(sorted(importance_gain.items(), key=lambda item: item[1], reverse=True))

    # 8. Save model artifact
    model_file.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_file))

    # 9. Build and save metadata
    eval_metrics = EvaluationMetrics(
        mae=round(mae, 3),
        rmse=round(rmse, 3),
        r2_score=round(r2_score, 4),
        baseline_mae_comparison=round(baseline_mae, 3) if baseline_mae is not None else None,
    )

    split_info = DatasetSplitInfo(
        total_samples=len(df),
        train_samples=len(train_df),
        test_samples=len(test_df),
        total_journeys=len(train_j_ids) + len(test_j_ids),
        train_journeys=len(train_j_ids),
        test_journeys=len(test_j_ids),
        test_ratio=test_size,
        random_seed=random_seed,
    )

    full_hyperparams = dict(default_params)
    full_hyperparams["num_boost_round"] = num_boost_round

    metadata = ModelMetadata(
        model_name="Dynamic Train ETA XGBoost Regressor",
        model_version="1.0.0",
        model_type="XGBoost Regressor (Tree-based Gradient Boosting)",
        target_column=TARGET_COLUMN,
        target_formulation="residual",
        prediction_label="minutes_to_next_station",
        features=PRD_FEATURE_NAMES,
        categorical_features=["day_type"],
        dataset_info=split_info,
        metrics=eval_metrics,
        hyperparameters=full_hyperparams,
        feature_importance_gain=importance_gain,
        feature_importance_weight=importance_weight,
        model_file_name=model_file.name,
        is_synthetic=True,
        disclaimer=DATA_DISCLAIMER,
    )
    metadata.save(meta_file)

    return booster, metadata


def main():
    """CLI entry point for training the XGBoost ETA regression model."""
    parser = argparse.ArgumentParser(description="Train XGBoost ETA Regression Model")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to input dataset CSV")
    parser.add_argument("--model-out", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to save model file")
    parser.add_argument("--meta-out", type=str, default=str(DEFAULT_METADATA_PATH), help="Path to save metadata JSON")
    parser.add_argument("--test-size", type=float, default=0.20, help="Ratio of journeys reserved for test (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  DYNAMIC TRAIN ETA FORECASTING: DAY 2 ML TRAINING (XGBOOST)")
    print("=" * 65)
    print(f"Loading dataset from: {args.data}")

    booster, metadata = train_eta_model(
        dataset_path=args.data,
        model_output_path=args.model_out,
        metadata_output_path=args.meta_out,
        test_size=args.test_size,
        random_seed=args.seed,
    )

    print("\n--- Training & Split Summary ---")
    print(f"Total samples:     {metadata.dataset_info.total_samples}")
    print(f"Train samples:     {metadata.dataset_info.train_samples} ({metadata.dataset_info.train_journeys} journeys)")
    print(f"Test samples:      {metadata.dataset_info.test_samples} ({metadata.dataset_info.test_journeys} journeys)")
    print(f"Journey-level test ratio: {args.test_size * 100:.0f}%")
    print(f"Random seed:       {args.seed}")

    print("\n--- Test Evaluation Metrics ---")
    print(f"MAE:               {metadata.metrics.mae:.3f} minutes")
    print(f"RMSE:              {metadata.metrics.rmse:.3f} minutes")
    print(f"R^2 Score:         {metadata.metrics.r2_score:.4f}")
    if metadata.metrics.baseline_mae_comparison is not None:
        print(f"Baseline MAE:      {metadata.metrics.baseline_mae_comparison:.3f} minutes")
        improvement = ((metadata.metrics.baseline_mae_comparison - metadata.metrics.mae) / metadata.metrics.baseline_mae_comparison) * 100.0
        print(f"Error Reduction:   {improvement:.1f}% vs. baseline heuristic")

    print("\n--- Artifact Paths ---")
    print(f"Model saved to:    {args.model_out}")
    print(f"Metadata saved to: {args.meta_out}")

    print("\n--- Feature Importance Summary (Gain) ---")
    print(f"{'Feature':<32} {'Importance (Gain)':<20}")
    print("-" * 52)
    for feat, gain in metadata.feature_importance_gain.items():
        print(f"{feat:<32} {gain:>15.2f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
