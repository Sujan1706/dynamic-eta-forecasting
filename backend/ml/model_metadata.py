"""
Model Metadata and Serialization Schema for Dynamic Train ETA Forecasting.
========================================================================
Tracks model provenance, training hyperparameters, journey-level evaluation metrics,
and feature definitions for the XGBoost ETA regression model.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


DATA_DISCLAIMER = (
    "SYNTHETIC SIMULATED DATASET ONLY. Timings, routes, disruptions, and model predictions "
    "are generated for MVP machine learning evaluation and DO NOT represent real historical "
    "Indian Railways operational logs."
)

PRD_FEATURE_NAMES = [
    "distance_to_go_km",
    "scheduled_time_to_go_min",
    "num_intermediate_halts",
    "current_delay_min",
    "delay_trend_3pt",
    "active_speed_restriction_flag",
    "congestion_score_downstream",
    "hist_avg_delay_this_section",
    "hist_recovery_rate_section",
    "weather_flag",
    "day_type",
]


class EvaluationMetrics(BaseModel):
    """Evaluation metrics computed on holdout test journeys."""
    mae: float = Field(description="Mean Absolute Error in minutes")
    rmse: float = Field(description="Root Mean Squared Error in minutes")
    r2_score: Optional[float] = Field(default=None, description="Coefficient of determination R^2")
    baseline_mae_comparison: Optional[float] = Field(
        default=None,
        description="MAE of naive baseline (scheduled + delay) on the same test set",
    )


class DatasetSplitInfo(BaseModel):
    """Information on journey-level train/test partitioning."""
    total_samples: int
    train_samples: int
    test_samples: int
    total_journeys: int
    train_journeys: int
    test_journeys: int
    test_ratio: float = 0.20
    random_seed: int = 42


class ModelMetadata(BaseModel):
    """Complete provenance and configuration metadata for the trained ETA model."""
    model_name: str = "Dynamic Train ETA XGBoost Regressor"
    model_version: str = "1.0.0"
    model_type: str = "XGBoost Regressor (Tree-based Gradient Boosting)"
    target_column: str = "actual_remaining_minutes"
    target_formulation: str = Field(
        default="residual",
        description="'residual' for schedule deviation (actual - scheduled), or 'direct'",
    )
    prediction_label: str = "minutes_to_next_station"
    features: List[str] = Field(default_factory=lambda: list(PRD_FEATURE_NAMES))
    categorical_features: List[str] = Field(default_factory=lambda: ["day_type"])
    trained_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_info: DatasetSplitInfo
    metrics: EvaluationMetrics
    hyperparameters: Dict[str, Any]
    feature_importance_gain: Dict[str, float] = Field(default_factory=dict)
    feature_importance_weight: Dict[str, float] = Field(default_factory=dict)
    model_file_name: str = "eta_xgboost_model.json"
    is_synthetic: bool = True
    disclaimer: str = DATA_DISCLAIMER

    def save(self, file_path: Path | str) -> None:
        """Serializes metadata to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, file_path: Path | str) -> "ModelMetadata":
        """Loads and deserializes metadata from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model metadata file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
