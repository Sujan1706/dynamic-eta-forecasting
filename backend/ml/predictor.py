"""
Reusable ETA Prediction Service using Trained XGBoost Model.
===========================================================
Provides fast, typed inference for predicting remaining travel time
(minutes_to_next_station) from either a TrainFeatures instance,
a feature dictionary, or batched DataFrames.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field

from backend.ml.model_metadata import ModelMetadata, PRD_FEATURE_NAMES
from backend.features.feature_builder import TrainFeatures


DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "eta_xgboost_model.json"
DEFAULT_METADATA_PATH = DEFAULT_MODELS_DIR / "eta_xgboost_metadata.json"


class DetailedETAPrediction(BaseModel):
    """Structured response container for ML ETA forecasts."""
    minutes_to_next_station: float = Field(
        description="Predicted remaining transit time in minutes to the next station"
    )
    target_station_code: Optional[str] = Field(
        default=None, description="Target station code if provided in input"
    )
    train_number: Optional[str] = Field(
        default=None, description="Train identifier if provided in input"
    )
    model_name: str = Field(description="Name of the machine learning model")
    model_version: str = Field(description="Version of the trained model")
    prediction_source: str = "xgboost_ml"
    features_used: Dict[str, Any] = Field(description="Input features supplied to the model")
    is_synthetic_model: bool = True


class ETAPredictor:
    """
    Production-grade XGBoost prediction service for Dynamic Train ETA forecasting.
    Loads the serialized model artifact and metadata and computes robust inferences.
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        metadata_path: Optional[Union[Path, str]] = None,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.metadata_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH

        self._booster: Optional[xgb.Booster] = None
        self._metadata: Optional[ModelMetadata] = None
        self._feature_names = PRD_FEATURE_NAMES

        # Eagerly load model on instantiation if artifact exists
        if self.model_path.exists():
            self._load()

    def _load(self) -> None:
        """Loads XGBoost Booster and metadata from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained ETA model file not found at: {self.model_path}. "
                f"Please run 'python -m backend.ml.train_model' first."
            )

        self._booster = xgb.Booster()
        self._booster.load_model(str(self.model_path))

        if self.metadata_path.exists():
            self._metadata = ModelMetadata.load(self.metadata_path)
            self._feature_names = self._metadata.features
        else:
            self._metadata = None
            self._feature_names = PRD_FEATURE_NAMES

    @property
    def booster(self) -> xgb.Booster:
        """Returns the loaded XGBoost Booster instance."""
        if self._booster is None:
            self._load()
        return self._booster

    @property
    def metadata(self) -> Optional[ModelMetadata]:
        """Returns the loaded model metadata if available."""
        if self._metadata is None and self.metadata_path.exists():
            self._metadata = ModelMetadata.load(self.metadata_path)
        return self._metadata

    def _extract_feature_vector(
        self, features_input: Union[TrainFeatures, Dict[str, Any]]
    ) -> Tuple[List[float], Dict[str, Any], Optional[str], Optional[str]]:
        """
        Extracts ordered 11 PRD features from TrainFeatures or dict,
        validating required keys.
        """
        train_number = None
        target_station = None

        if isinstance(features_input, TrainFeatures):
            train_number = features_input.train_number
            target_station = features_input.target_station_code
            raw_dict = features_input.to_model_input_dict()
        elif isinstance(features_input, dict):
            train_number = features_input.get("train_number")
            target_station = features_input.get("target_station_code") or features_input.get("target_station")
            raw_dict = features_input
        else:
            raise TypeError(
                f"Unsupported feature input type: {type(features_input)}. "
                f"Expected TrainFeatures or Dict[str, Any]."
            )

        # Validate that all 11 required features are present
        missing = [f for f in self._feature_names if f not in raw_dict]
        if missing:
            raise ValueError(f"Missing required feature keys for ETA prediction: {missing}")

        vector = [float(raw_dict[f]) for f in self._feature_names]
        ordered_dict = {f: raw_dict[f] for f in self._feature_names}
        return vector, ordered_dict, train_number, target_station

    def _is_residual_model(self) -> bool:
        """Determines if the loaded model predicts residual delay or direct remaining time."""
        if self.metadata is not None:
            return getattr(self.metadata, "target_formulation", "residual") == "residual"
        return True

    def predict(
        self, features: Union[TrainFeatures, Dict[str, Any], pd.DataFrame]
    ) -> float:
        """
        Predicts remaining travel time in minutes to the next station.
        Guarantees non-negative prediction.

        Returns:
            minutes_to_next_station (float, rounded to 1 decimal place)
        """
        if isinstance(features, pd.DataFrame):
            batch_preds = self.predict_batch(features)
            return batch_preds[0] if batch_preds else 0.0

        vector, ordered_dict, _, _ = self._extract_feature_vector(features)
        feature_matrix = np.array([vector], dtype=float)
        dmatrix = xgb.DMatrix(feature_matrix, feature_names=self._feature_names)

        raw_pred = float(self.booster.predict(dmatrix)[0])
        if self._is_residual_model():
            sched_time = float(ordered_dict.get("scheduled_time_to_go_min", 0.0))
            clamped_pred = max(0.0, sched_time + raw_pred)
        else:
            clamped_pred = max(0.0, raw_pred)
        return round(clamped_pred, 1)

    def predict_detailed(
        self, features: Union[TrainFeatures, Dict[str, Any]]
    ) -> DetailedETAPrediction:
        """
        Returns a rich structured prediction response containing metadata,
        model version, and the input feature payload.
        """
        vector, features_used, train_num, target_stn = self._extract_feature_vector(features)
        feature_matrix = np.array([vector], dtype=float)
        dmatrix = xgb.DMatrix(feature_matrix, feature_names=self._feature_names)

        raw_pred = float(self.booster.predict(dmatrix)[0])
        if self._is_residual_model():
            sched_time = float(features_used.get("scheduled_time_to_go_min", 0.0))
            minutes = round(max(0.0, sched_time + raw_pred), 1)
        else:
            minutes = round(max(0.0, raw_pred), 1)

        model_name = self.metadata.model_name if self.metadata else "Dynamic Train ETA XGBoost Regressor"
        model_ver = self.metadata.model_version if self.metadata else "1.0.0"

        return DetailedETAPrediction(
            minutes_to_next_station=minutes,
            target_station_code=target_stn,
            train_number=train_num,
            model_name=model_name,
            model_version=model_ver,
            features_used=features_used,
            is_synthetic_model=True,
        )

    def predict_batch(
        self,
        features_list: Union[List[TrainFeatures], List[Dict[str, Any]], pd.DataFrame],
    ) -> List[float]:
        """
        Performs vectorized high-throughput inference across multiple observation samples.
        """
        if isinstance(features_list, pd.DataFrame):
            missing = [f for f in self._feature_names if f not in features_list.columns]
            if missing:
                raise ValueError(f"DataFrame is missing required feature columns: {missing}")
            X = features_list[self._feature_names].values.astype(float)
        elif isinstance(features_list, list):
            if not features_list:
                return []
            vectors = [self._extract_feature_vector(item)[0] for item in features_list]
            X = np.array(vectors, dtype=float)
        else:
            raise TypeError(f"Unsupported batch input type: {type(features_list)}")

        dmatrix = xgb.DMatrix(X, feature_names=self._feature_names)
        raw_preds = self.booster.predict(dmatrix)

        if self._is_residual_model():
            if isinstance(features_list, pd.DataFrame):
                sched_times = features_list["scheduled_time_to_go_min"].values.astype(float)
            else:
                sched_times = np.array([
                    float(item.scheduled_time_to_go_min if hasattr(item, "scheduled_time_to_go_min") else item["scheduled_time_to_go_min"])
                    for item in features_list
                ], dtype=float)
            clamped_preds = np.maximum(0.0, sched_times + raw_preds)
        else:
            clamped_preds = np.maximum(0.0, raw_preds)

        return [round(float(p), 1) for p in clamped_preds]
