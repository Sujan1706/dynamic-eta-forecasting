"""
Unit & Integration Tests for Live Train Telemetry Feature Engineering and ML Pipeline.
======================================================================================
Validates all requirements:
1. Uses existing FeatureBuilder.
2. Maps available live API fields into existing features.
3. For features unavailable from live API:
   - uses existing historical/default feature provider
   - clearly marks them as estimated/default where appropriate
4. Does not create fake real-world values (neutral defaults: 0 for TSR, 0.0 for congestion, 0 for weather).
5. Does not retrain the model using live data yet (uses existing XGBoost artifact).
6. Calculates:
   - baseline ETA
   - ML ETA
   - confidence range
7. Emits structured logging showing feature categorization:
   - live data
   - historical data
   - simulator/default source
8. Trained model architecture and 11-feature contract remain 100% unchanged.
"""

import logging
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from backend.main import app
from backend.services.schemas import TrainRunningState
from backend.features.feature_builder import (
    FeatureBuilder,
    TrainFeatures,
    SectionStatsProvider,
)
from backend.simulator.journey import StationStop
from backend.services.baseline_eta import BaselineETAService
from backend.services.multi_station_eta import MultiStationETAService
from backend.services.live_train_service import LiveTrainLookupService
from backend.services.railway_api_client import RailRadarClient
from backend.database.connection import SessionLocal


@pytest.fixture
def sample_route_stops():
    """Corridor stops: NDLS -> CNB -> PRYJ -> DDU -> HWH."""
    return [
        StationStop(
            sequence=1,
            station_id=1,
            station_code="NDLS",
            station_name="New Delhi",
            distance_from_source_km=0.0,
            scheduled_departure="16:55",
            scheduled_stop_minutes=0.0,
        ),
        StationStop(
            sequence=2,
            station_id=2,
            station_code="CNB",
            station_name="Kanpur Central",
            distance_from_source_km=440.0,
            scheduled_arrival="21:35",
            scheduled_departure="21:40",
            scheduled_stop_minutes=5.0,
        ),
        StationStop(
            sequence=3,
            station_id=3,
            station_code="PRYJ",
            station_name="Prayagraj Junction",
            distance_from_source_km=635.0,
            scheduled_arrival="23:43",
            scheduled_departure="23:45",
            scheduled_stop_minutes=2.0,
        ),
        StationStop(
            sequence=4,
            station_id=4,
            station_code="DDU",
            station_name="Pt. DD Upadhyaya Junction",
            distance_from_source_km=788.0,
            scheduled_arrival="01:42",
            scheduled_departure="01:52",
            scheduled_stop_minutes=10.0,
        ),
        StationStop(
            sequence=5,
            station_id=5,
            station_code="HWH",
            station_name="Howrah Junction",
            distance_from_source_km=1447.0,
            scheduled_arrival="09:55",
            scheduled_departure="09:55",
            scheduled_stop_minutes=0.0,
        ),
    ]


@pytest.fixture
def live_train_state():
    """Live TrainRunningState representing real-time telemetry from external API."""
    return TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Howrah Rajdhani Express",
        status="RUNNING",
        current_station_code="CNB",
        current_station_sequence=2,
        current_delay_minutes=18.5,
        previous_station_code="NDLS",
        next_station_code="PRYJ",
        next_station_distance_km=195.0,
        segment_progress=0.35,
        speed_kmh=118.0,
        timestamp=datetime(2026, 9, 28, 22, 10, tzinfo=timezone.utc),
        source="external_api",
    )


# ---------------------------------------------------------------------------
# 1. Feature Builder Mapping & Provenance Tests
# ---------------------------------------------------------------------------

def test_feature_builder_maps_live_fields(live_train_state, sample_route_stops, caplog):
    """
    Verify FeatureBuilder maps live API fields to PRD features and tracks provenance:
    - live data
    - historical data
    - simulator/default source
    """
    builder = FeatureBuilder()

    with caplog.at_level(logging.INFO):
        feat = builder.build(
            train_state=live_train_state,
            target_station_code="PRYJ",
            route_stations=sample_route_stops,
        )

    # 1. Verify mapped live values
    assert feat.current_delay_min == 18.5
    assert feat.train_number == "12302"
    assert feat.target_station_code == "PRYJ"
    assert feat.distance_to_go_km == 195.0  # Mapped from live next_station_distance_km
    assert feat.num_intermediate_halts == 0  # CNB -> PRYJ has 0 intermediate stops
    assert feat.scheduled_time_to_go_min > 0.0

    # 2. Verify provenance categorization
    assert "current_delay_min" in feat.live_features
    assert "distance_to_go_km" in feat.live_features
    assert "scheduled_time_to_go_min" in feat.live_features
    assert "num_intermediate_halts" in feat.live_features
    assert "day_type" in feat.live_features

    # 3. Verify historical section features
    assert "hist_avg_delay_this_section" in feat.historical_features
    assert "hist_recovery_rate_section" in feat.historical_features
    assert feat.hist_avg_delay_this_section > 0.0
    assert feat.hist_recovery_rate_section > 0.0

    # 4. Verify unavailable features marked as default/estimated without fake values
    assert feat.active_speed_restriction_flag == 0
    assert feat.congestion_score_downstream == 0.0
    assert feat.weather_flag == 0
    assert feat.delay_trend_3pt == 0.0

    assert "active_speed_restriction_flag" in feat.default_features
    assert "congestion_score_downstream" in feat.default_features
    assert "weather_flag" in feat.default_features
    assert "delay_trend_3pt" in feat.default_features

    assert "active_speed_restriction_flag" in feat.estimated_features
    assert "congestion_score_downstream" in feat.estimated_features
    assert "weather_flag" in feat.estimated_features
    assert "delay_trend_3pt" in feat.estimated_features

    # 5. Verify logging output contains the required three categories
    log_text = caplog.text
    assert "live data" in log_text
    assert "historical data" in log_text
    assert "simulator/default source" in log_text


def test_live_with_recent_delays_uses_live_trend(live_train_state, sample_route_stops):
    """Verify that when recent delay observations exist for live train, trend is classified as live data."""
    builder = FeatureBuilder()
    feat = builder.build(
        train_state=live_train_state,
        target_station_code="PRYJ",
        route_stations=sample_route_stops,
        recent_delays=[10.0, 14.0, 18.5],
    )

    assert feat.delay_trend_3pt == 4.25  # (18.5 - 10.0) / 2
    assert "delay_trend_3pt" in feat.live_features
    assert "delay_trend_3pt" not in feat.estimated_features


def test_simulator_features_provenance(sample_route_stops):
    """Verify that simulator train state attributes features to simulator/default source."""
    builder = FeatureBuilder()
    sim_state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Howrah Rajdhani Express",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        current_delay_minutes=5.0,
        timestamp=datetime.now(timezone.utc),
        source="simulator",
    )

    feat = builder.build(sim_state, "CNB", route_stations=sample_route_stops)
    assert feat.feature_provenance["current_delay_min"] == "simulator/default source"
    assert feat.feature_provenance["distance_to_go_km"] == "simulator/default source"
    assert feat.feature_provenance["hist_avg_delay_this_section"] == "historical data"


# ---------------------------------------------------------------------------
# 2. Model Input Architecture Unchanged
# ---------------------------------------------------------------------------

def test_model_input_dict_and_feature_list_contracts(live_train_state, sample_route_stops):
    """Verify model input format contains exactly 11 numeric features matching training."""
    builder = FeatureBuilder()
    feat = builder.build(live_train_state, "PRYJ", route_stations=sample_route_stops)

    model_dict = feat.to_model_input_dict()
    assert len(model_dict) == 11
    expected_keys = [
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
    assert list(model_dict.keys()) == expected_keys
    for k, v in model_dict.items():
        assert isinstance(v, (int, float)), f"Feature {k} must be numeric float/int"

    feat_list = feat.to_feature_list()
    assert len(feat_list) == 11
    assert feat_list == list(model_dict.values())


# ---------------------------------------------------------------------------
# 3. Calculation of Baseline ETA, ML ETA, Confidence Range from Live State
# ---------------------------------------------------------------------------

def test_live_train_eta_calculations(live_train_state, sample_route_stops):
    """
    Verify that live TrainRunningState is processed by:
    - BaselineETAService -> Baseline ETA
    - MultiStationETAService -> ML ETA + Confidence Range
    """
    baseline_service = BaselineETAService()
    base_pred = baseline_service.predict_station(
        train_state=live_train_state,
        target_station_code="PRYJ",
        route_stations=sample_route_stops,
    )

    assert base_pred is not None
    assert base_pred.baseline_eta is not None
    assert base_pred.scheduled_eta is not None
    # Baseline ETA reflects delay minus recovery
    assert base_pred.baseline_eta >= base_pred.scheduled_eta

    multi_station_service = MultiStationETAService(
        baseline_service=baseline_service,
        feature_builder=FeatureBuilder(),
    )
    stn_pred = multi_station_service.predict_station_eta(
        train_state=live_train_state,
        target_station_code="PRYJ",
        route_stations=sample_route_stops,
    )

    assert stn_pred is not None
    assert stn_pred.predicted_eta is not None
    assert stn_pred.baseline_eta is not None
    assert stn_pred.confidence_lower_bound is not None
    assert stn_pred.confidence_upper_bound is not None
    assert stn_pred.confidence_lower_bound <= stn_pred.predicted_eta <= stn_pred.confidence_upper_bound
    assert stn_pred.uncertainty_margin_minutes > 0.0


# ---------------------------------------------------------------------------
# 4. LiveTrainLookupService End-to-End Flow & Logging
# ---------------------------------------------------------------------------

def test_live_train_lookup_service_e2e(live_train_state, caplog):
    """
    Verify complete LiveTrainLookupService flow from live state to:
    - FeatureBuilder
    - Baseline ETA
    - ML ETA
    - Confidence Range
    - Feature Provenance & Logging
    """
    mock_client = MagicMock(spec=RailRadarClient)
    mock_client.fetch_live_train.return_value = live_train_state
    mock_client.cache = None

    service = LiveTrainLookupService(railway_client=mock_client)

    db = SessionLocal()
    try:
        with caplog.at_level(logging.INFO):
            result = service.lookup_live_train("12302", db=db)

        assert result.train_number == "12302"
        assert result.status == "RUNNING"
        assert result.current_station == "CNB"
        assert result.next_station == "PRYJ"
        assert result.current_delay == 18.5
        assert result.speed == 118.0
        assert result.segment_progress == 0.35

        # Predictions
        assert result.baseline_eta is not None
        assert result.ml_eta is not None
        assert result.confidence_range is not None
        assert result.confidence_range.lower_bound <= result.ml_eta <= result.confidence_range.upper_bound

        # Feature provenance in result
        assert result.features is not None
        assert len(result.features) == 11
        assert result.feature_provenance is not None
        assert result.feature_provenance["current_delay_min"] == "live data"
        assert result.feature_provenance["hist_avg_delay_this_section"] == "historical data"
        assert result.feature_provenance["active_speed_restriction_flag"] == "simulator/default source"

        # Check logs
        assert "live data" in caplog.text
        assert "historical data" in caplog.text
        assert "simulator/default source" in caplog.text
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 5. API Endpoint Verification: GET /live/train/{train_number}
# ---------------------------------------------------------------------------

def test_api_live_train_returns_features_and_provenance(live_train_state):
    """Verify GET /live/train/{train_number} returns baseline, ML ETA, confidence, and features."""
    with patch.object(RailRadarClient, "fetch_live_train", return_value=live_train_state):
        with TestClient(app) as client:
            resp = client.get("/live/train/12302")
            assert resp.status_code == 200
            data = resp.json()

            assert data["train_number"] == "12302"
            assert data["baseline_eta"] is not None
            assert data["ml_eta"] is not None
            assert data["confidence_range"] is not None
            assert data["confidence_range"]["margin_minutes"] > 0

            # Features and provenance
            assert "features" in data
            assert data["features"] is not None
            assert len(data["features"]) == 11
            assert "feature_provenance" in data
            assert data["feature_provenance"]["current_delay_min"] == "live data"
            assert data["feature_provenance"]["hist_avg_delay_this_section"] == "historical data"
            assert "estimated_features" in data
            assert "active_speed_restriction_flag" in data["estimated_features"]
