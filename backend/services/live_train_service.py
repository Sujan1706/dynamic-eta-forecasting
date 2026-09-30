"""
Live Train Lookup Service.
==========================
Coordinates the clean service flow:
  train number
      ↓
  Railway API (RailRadarClient)
      ↓
  Normalized TrainRunningState
      ↓
  Feature Builder (FeatureBuilder)
      ↓
  Baseline ETA + ML ETA (BaselineETAService + MultiStationETAService)

Enforces:
1. Strict environment-only secret handling (never leaks RAILRADAR_API_KEY).
2. Clean separation of concerns (networking -> normalization -> feature extraction -> ETA prediction).
3. Robust error handling without unhandled crashes.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.services.schemas import TrainRunningState
from backend.services.railway_api_client import (
    RailRadarClient,
    RailwayAPIError,
    TrainNotFoundError,
    sanitize_secret,
)
from backend.services.baseline_eta import BaselineETAService
from backend.services.multi_station_eta import MultiStationETAService
from backend.features.feature_builder import FeatureBuilder

logger = logging.getLogger("dynamic_eta.live_train_service")


class ConfidenceRangeDetails(BaseModel):
    """Confidence interval details around the ML ETA prediction."""
    lower_bound: datetime = Field(description="Earliest expected arrival datetime")
    upper_bound: datetime = Field(description="Latest expected arrival datetime")
    margin_minutes: float = Field(description="Heuristic uncertainty margin in minutes (+/-)")


class LiveTrainResult(BaseModel):
    """
    Standard response payload for live train lookup.
    Fulfills all requirements of GET /live/train/{train_number}.
    """
    train_number: str = Field(description="Unique train identifier/number")
    train_name: str = Field(description="Name of the train")
    status: str = Field(description="Operational status (e.g. RUNNING, COMPLETED)")
    current_station: Optional[str] = Field(default=None, description="Current or last passed station code")
    next_station: Optional[str] = Field(default=None, description="Immediate next scheduled station code")
    current_delay: float = Field(default=0.0, description="Current train delay in minutes")
    speed: Optional[float] = Field(default=None, description="Current instantaneous speed in km/h if available")
    segment_progress: Optional[float] = Field(default=None, description="Current inter-station segment progress (0.0 to 1.0) if available")
    timestamp: datetime = Field(description="Timestamp of the telemetry update")
    baseline_eta: Optional[datetime] = Field(default=None, description="Baseline predicted arrival datetime at next station")
    ml_eta: Optional[datetime] = Field(default=None, description="ML predicted arrival datetime at next station")
    confidence_range: Optional[ConfidenceRangeDetails] = Field(default=None, description="Confidence interval details")
    data_source: str = Field(default="LIVE_API", description="Telemetry data source")

    # Freshness and cache provenance (Requirements 1, 3, 4)
    last_updated: Optional[datetime] = Field(default=None, description="Timestamp when telemetry was last observed/updated")
    cached_at: Optional[datetime] = Field(default=None, description="Timestamp when snapshot was placed in cache")
    is_cached: bool = Field(default=False, description="True if telemetry was served from in-process cache")
    cache_age_seconds: Optional[float] = Field(default=None, description="Age of cached snapshot in seconds")
    cache_ttl_seconds: Optional[int] = Field(default=None, description="Configured cache TTL in seconds")
    is_stale: bool = Field(default=False, description="True if serving stale telemetry during rate limit cooldown")

    # ML Features and Provenance
    features: Optional[Dict[str, Any]] = Field(default=None, description="The 11 computed model features")
    feature_provenance: Optional[Dict[str, str]] = Field(
        default=None,
        description="Feature source categorization ('live data', 'historical data', 'simulator/default source')",
    )
    estimated_features: Optional[List[str]] = Field(
        default=None,
        description="Features that used fallback or estimated defaults because they are unavailable in live telemetry",
    )


class LiveTrainLookupService:
    """
    Executes the clean end-to-end live train lookup pipeline:
    RailRadar live API -> normalized TrainRunningState -> FeatureBuilder -> Baseline & ML ETAs.
    """

    def __init__(
        self,
        railway_client: Optional[RailRadarClient] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        baseline_service: Optional[BaselineETAService] = None,
        multi_station_service: Optional[MultiStationETAService] = None,
    ):
        self._railway_client = railway_client
        self.feature_builder = feature_builder or FeatureBuilder()
        self.baseline_service = baseline_service or BaselineETAService()
        if multi_station_service is None:
            try:
                self.multi_station_service = MultiStationETAService(
                    baseline_service=self.baseline_service,
                    feature_builder=self.feature_builder,
                )
            except Exception:
                self.multi_station_service = None
        else:
            self.multi_station_service = multi_station_service

    @property
    def railway_client(self) -> RailRadarClient:
        if self._railway_client is None:
            self._railway_client = RailRadarClient()
        return self._railway_client

    def lookup_live_train(
        self,
        train_number: str,
        db: Optional[Session] = None,
    ) -> LiveTrainResult:
        """
        Executes the live train lookup flow:
        1. Queries Railway API for live telemetry.
        2. Receives normalized TrainRunningState.
        3. Extracts features via FeatureBuilder.
        4. Calculates Baseline ETA and ML ETA for the next upcoming station.
        5. Returns structured LiveTrainResult.
        """
        clean_number = str(train_number).strip()

        # Step 1 & 2: Railway API -> Normalized TrainRunningState
        state = self.railway_client.fetch_live_train(clean_number)

        # Manage DB session lifecycle if not passed
        session_created = False
        if db is None:
            try:
                from backend.database.connection import SessionLocal
                db = SessionLocal()
                session_created = True
            except Exception:
                pass

        try:
            return self._compute_live_result(state, db=db)
        finally:
            if session_created and db is not None:
                db.close()

    def _compute_live_result(
        self,
        state: TrainRunningState,
        db: Optional[Session] = None,
    ) -> LiveTrainResult:
        """Computes Baseline ETA and ML ETA from normalized state and builds LiveTrainResult."""
        clean_number = state.train_number
        baseline_eta: Optional[datetime] = None
        ml_eta: Optional[datetime] = None
        confidence_range: Optional[ConfidenceRangeDetails] = None
        features_dict: Optional[Dict[str, Any]] = None
        feature_prov_dict: Optional[Dict[str, str]] = None
        est_features_list: Optional[List[str]] = None

        target_station = state.next_station_code

        # If train is completed or has no next station, ETAs are not applicable
        if target_station and state.status.upper() != "COMPLETED":
            # Step 3: Feature Builder
            try:
                feat = self.feature_builder.build(
                    train_state=state,
                    target_station_code=target_station,
                    db=db,
                )
                features_dict = feat.to_model_input_dict()
                feature_prov_dict = feat.feature_provenance
                est_features_list = feat.estimated_features

                logger.info(
                    "Connected live train %s to feature pipeline (target: %s):\n"
                    "  - live data: %s\n"
                    "  - historical data: %s\n"
                    "  - simulator/default source: %s",
                    clean_number,
                    target_station,
                    ", ".join(feat.live_features) if feat.live_features else "none",
                    ", ".join(feat.historical_features) if feat.historical_features else "none",
                    ", ".join(feat.default_features) if feat.default_features else "none",
                )
            except Exception as exc:
                logger.warning(
                    "Feature extraction skipped for train '%s': %s",
                    clean_number,
                    sanitize_secret(str(exc)),
                )

            # Step 4: Baseline ETA
            try:
                base_pred = self.baseline_service.predict_station(
                    train_state=state,
                    target_station_code=target_station,
                    db=db,
                )
                if base_pred:
                    baseline_eta = base_pred.baseline_eta
            except Exception as exc:
                logger.warning(
                    "Baseline ETA computation failed for train '%s' at '%s': %s",
                    clean_number,
                    target_station,
                    sanitize_secret(str(exc)),
                )

            # Ensure baseline ETA aligns directly with live timestamp and scheduled_time_to_go_min:
            # current timestamp + scheduled_time_to_go_min + predicted_delay
            if feat is not None and state.timestamp is not None:
                sched_min = float(feat.scheduled_time_to_go_min)
                curr_delay = float(state.current_delay_minutes or 0.0)
                dist_km = float(feat.distance_to_go_km)
                rec_rate = getattr(self.baseline_service, "recovery_rate_per_100km", 2.0)
                raw_buf = (dist_km / 100.0) * rec_rate
                pred_delay = max(0.0, curr_delay - min(curr_delay, raw_buf)) if curr_delay > 0.0 else curr_delay
                baseline_eta = state.timestamp + timedelta(minutes=sched_min + pred_delay)

            # Step 4: ML ETA (Chained XGBoost)
            if self.multi_station_service is not None:
                try:
                    station_pred = self.multi_station_service.predict_station_eta(
                        train_state=state,
                        target_station_code=target_station,
                        db=db,
                    )
                    if station_pred:
                        ml_eta = station_pred.predicted_eta
                        confidence_range = ConfidenceRangeDetails(
                            lower_bound=station_pred.confidence_lower_bound,
                            upper_bound=station_pred.confidence_upper_bound,
                            margin_minutes=station_pred.uncertainty_margin_minutes,
                        )
                        if baseline_eta is None:
                            baseline_eta = station_pred.baseline_eta
                except Exception as exc:
                    logger.warning(
                        "ML ETA computation failed for train '%s' at '%s': %s",
                        clean_number,
                        target_station,
                        sanitize_secret(str(exc)),
                    )
                    # Graceful ML fallback: use baseline ETA if ML fails
                    if baseline_eta is not None and ml_eta is None:
                        ml_eta = baseline_eta

        # Freshness and cache metadata (Requirements 1, 3, 4)
        is_cached = False
        cached_at = None
        cache_age = None
        cache_ttl = None
        is_stale = False

        client_cache = getattr(self.railway_client, "cache", None) if self.railway_client else None
        if client_cache:
            try:
                from backend.services.cache import LiveTrainCache, CacheEntry
                if isinstance(client_cache, LiveTrainCache):
                    entry = client_cache.get_entry(clean_number)
                    if isinstance(entry, CacheEntry):
                        cached_at = entry.cached_at
                        cache_age = entry.age_seconds
                        cache_ttl = client_cache.ttl_seconds
                        is_stale = entry.is_stale
                        is_cached = (entry.hit_count > 0)
            except Exception:
                pass

        return LiveTrainResult(
            train_number=state.train_number,
            train_name=state.train_name,
            status=state.status,
            current_station=state.current_station_code,
            next_station=state.next_station_code,
            current_delay=float(state.current_delay_minutes or 0.0),
            speed=float(state.speed_kmh) if state.speed_kmh is not None else None,
            segment_progress=float(state.segment_progress) if state.segment_progress is not None else None,
            timestamp=state.timestamp,
            last_updated=state.timestamp,
            baseline_eta=baseline_eta,
            ml_eta=ml_eta,
            confidence_range=confidence_range,
            data_source="LIVE_API",
            features=features_dict,
            feature_provenance=feature_prov_dict,
            estimated_features=est_features_list,
            cached_at=cached_at,
            is_cached=is_cached,
            cache_age_seconds=cache_age,
            cache_ttl_seconds=cache_ttl,
            is_stale=is_stale,
        )
