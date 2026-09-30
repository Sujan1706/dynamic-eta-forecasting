"""
Integration tests for Day 3 Phase 1: Real-Time Railway Data Integration.
Proves end-to-end:
  LIVE_API (RailRadar telemetry)
      ↓
  Normalized TrainRunningState (source='external_api')
      ↓
  Existing ETA Pipeline:
      - BaselineETAService (static timetable + delay - recovery)
      - FeatureBuilder (11 PRD kinematic and situational features)
      - MultiStationETAService (Chained segment-by-segment XGBoost regression)
      ↓
  FastAPI Endpoints:
      - GET /train/{train_id}?data_source=LIVE_API
      - GET /train/{train_id}/eta/{station_code}?data_source=LIVE_API
      - GET /trains?data_source=LIVE_API
      - Automatic controlled fallback to SIMULATOR on external API disruption
"""

import os
from datetime import date, datetime, timezone
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.connection import SessionLocal
from backend.services.schemas import TrainRunningState
from backend.services.railway_api_client import (
    RailRadarClient,
    ServiceUnavailableError,
    APITimeoutError,
    AuthenticationError,
    TrainNotFoundError,
    RateLimitExceededError,
    MalformedResponseError,
)
from backend.services.baseline_eta import BaselineETAService
from backend.services.multi_station_eta import MultiStationETAService
from backend.features.feature_builder import FeatureBuilder


@pytest.fixture
def client():
    """FastAPI TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session():
    """Provides a database session for route inspection."""
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def sample_railradar_api_response():
    """Documented RailRadar API live train payload for train 12302."""
    return {
        "success": True,
        "data": {
            "trainNumber": "12302",
            "trainName": "Howrah Rajdhani Express",
            "startDate": "2026-09-28",
            "status": "RUNNING",
            "currentLocation": {
                "stationCode": "CNB",
                "sequence": 2,
                "delayMinutes": 22.0,
                "segmentProgress": 0.40,
                "speedKmh": 105.0,
            },
            "previousStation": {
                "stationCode": "NDLS",
            },
            "nextHalt": {
                "stationCode": "PRYJ",
                "distance": 195.0,
            },
            "lastUpdatedAt": "2026-09-28T16:00:00+00:00",
        },
    }


@pytest.fixture
def live_running_state(sample_railradar_api_response):
    """Normalized TrainRunningState produced from the sample RailRadar response."""
    client = RailRadarClient(api_key="test_env_key")
    return client.normalize_payload(sample_railradar_api_response, "12302")


# ---------------------------------------------------------------------------
# 1. Pipeline Verification: LIVE_API -> normalized state -> ETA Pipeline
# ---------------------------------------------------------------------------

def test_live_api_normalized_state_through_eta_services(live_running_state, db_session):
    """
    Direct pipeline test:
    Validates that a live RailRadar payload normalizes into TrainRunningState
    and feeds directly into BaselineETAService and MultiStationETAService
    without requiring any model retraining or schema alterations.
    """
    # 1. Verify normalized state properties
    assert isinstance(live_running_state, TrainRunningState)
    assert live_running_state.source == "external_api"
    assert live_running_state.train_number == "12302"
    assert live_running_state.current_station_code == "CNB"
    assert live_running_state.current_delay_minutes == 22.0
    assert live_running_state.next_station_code == "PRYJ"

    # 2. Feed into BaselineETAService
    baseline_service = BaselineETAService()
    baseline_pred = baseline_service.predict_station(
        train_state=live_running_state,
        target_station_code="PRYJ",
        db=db_session,
    )
    assert baseline_pred is not None
    assert baseline_pred.target_station == "PRYJ"
    assert baseline_pred.current_delay == 22.0
    assert baseline_pred.baseline_eta > baseline_pred.scheduled_eta

    # 3. Feed into MultiStationETAService (Chained XGBoost)
    feature_builder = FeatureBuilder()
    multi_service = MultiStationETAService(
        baseline_service=baseline_service,
        feature_builder=feature_builder,
    )
    station_pred = multi_service.predict_station_eta(
        train_state=live_running_state,
        target_station_code="PRYJ",
        db=db_session,
    )
    assert station_pred is not None
    assert station_pred.station_code == "PRYJ"
    assert station_pred.predicted_eta is not None
    assert station_pred.confidence_lower_bound <= station_pred.predicted_eta <= station_pred.confidence_upper_bound


# ---------------------------------------------------------------------------
# 2. FastAPI End-to-End Integration with LIVE_API
# ---------------------------------------------------------------------------

def test_api_train_details_with_live_data(client, live_running_state):
    """
    Verify GET /train/{train_id}?data_source=LIVE_API:
    1. Queries live RailRadar client.
    2. Returns 200 with source='external_api'.
    3. Produces chained ML ETAs and confidence ranges.
    4. Confirms data_source_mode='LIVE_API' and is_fallback=False.
    """
    with patch.dict(os.environ, {"RAILRADAR_API_KEY": "rr_live_secret_key"}):
        with patch.object(RailRadarClient, "fetch_live_train", return_value=live_running_state):
            resp = client.get("/train/12302?data_source=LIVE_API")
            assert resp.status_code == 200
            data = resp.json()

            assert data["train_number"] == "12302"
            assert data["data_source_mode"] == "LIVE_API"
            assert data["is_fallback"] is False
            assert data["fallback_reason"] is None

            # Current state comes from external API
            state = data["current_state"]
            assert state["source"] == "external_api"
            assert state["current_station_code"] == "CNB"
            assert state["current_delay_minutes"] == 22.0
            assert state["next_station_code"] == "PRYJ"

            # Upcoming multi-station predictions generated using live state
            assert len(data["upcoming_stations"]) > 0
            first_upcoming = data["upcoming_stations"][0]
            assert first_upcoming["station_code"] == "PRYJ"
            assert first_upcoming["scheduled_eta"] is not None
            assert first_upcoming["baseline_eta"] is not None
            assert first_upcoming["ml_eta"] is not None
            assert first_upcoming["confidence_range"]["margin_minutes"] > 0


def test_api_single_station_eta_with_live_data(client, live_running_state):
    """
    Verify GET /train/{train_id}/eta/{station_code}?data_source=LIVE_API:
    Calculates dynamic ML ETA and Baseline ETA for target station using live telemetry.
    """
    with patch.dict(os.environ, {"RAILRADAR_API_KEY": "rr_live_secret_key"}):
        with patch.object(RailRadarClient, "fetch_live_train", return_value=live_running_state):
            resp = client.get("/train/12302/eta/PRYJ?data_source=LIVE_API")
            assert resp.status_code == 200
            data = resp.json()

            assert data["train_number"] == "12302"
            assert data["target_station"] == "PRYJ"
            assert data["current_delay_minutes"] == 22.0
            assert data["data_source_mode"] == "LIVE_API"
            assert data["is_fallback"] is False
            assert data["ml_status"] == "AVAILABLE"
            assert data["ml_eta"] is not None
            assert data["baseline_eta"] is not None


def test_api_list_trains_with_live_data(client, live_running_state):
    """
    Verify GET /trains?data_source=LIVE_API:
    Lists all trains with live running telemetry and computed next-station ETA.
    """
    with patch.dict(os.environ, {"RAILRADAR_API_KEY": "rr_live_secret_key"}):
        with patch.object(RailRadarClient, "fetch_live_train", return_value=live_running_state):
            resp = client.get("/trains?data_source=LIVE_API")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] >= 1
            train_item = next(t for t in data["trains"] if t["train_number"] == "12302")
            assert train_item["data_source_mode"] == "LIVE_API"
            assert train_item["current_state"]["source"] == "external_api"
            assert train_item["current_delay_minutes"] == 22.0


# ---------------------------------------------------------------------------
# 3. End-to-End Fallback from LIVE_API to SIMULATOR
# ---------------------------------------------------------------------------

def test_api_fallback_on_live_api_503(client):
    """
    Verify that when RailRadar returns HTTP 503 Service Unavailable,
    the API does not fail with 500/503.
    Instead, it returns 200 with fallback to SIMULATOR and logs a sanitized error.
    """
    err = ServiceUnavailableError("RailRadar service temporarily unavailable (HTTP 503): maintenance")
    with patch.dict(os.environ, {"RAILRADAR_API_KEY": "super_secret_token_123"}):
        with patch.object(RailRadarClient, "fetch_live_train", side_effect=err):
            resp = client.get("/train/12302?data_source=LIVE_API")
            assert resp.status_code == 200
            data = resp.json()

            # Provenance indicates controlled fallback
            assert data["data_source_mode"] == "SIMULATOR"
            assert data["is_fallback"] is True
            assert data["fallback_reason"] is not None
            assert "HTTP 503" in data["fallback_reason"]
            # Sensitive key MUST NOT appear anywhere in the response
            assert "super_secret_token_123" not in resp.text

            # Still produced valid ETA predictions using simulator state
            assert data["current_state"]["source"] == "simulator"
            assert data["current_state"]["train_number"] == "12302"
            assert len(data["upcoming_stations"]) > 0


def test_api_fallback_on_live_api_timeout(client):
    """
    Verify that when RailRadar request times out,
    the API smoothly falls back to SIMULATOR.
    """
    err = APITimeoutError("Live railway API request timed out after 5.0 seconds")
    with patch.dict(os.environ, {"RAILRADAR_API_KEY": "super_secret_token_123"}):
        with patch.object(RailRadarClient, "fetch_live_train", side_effect=err):
            resp = client.get("/train/12302/eta/PRYJ?data_source=LIVE_API")
            assert resp.status_code == 200
            data = resp.json()

            assert data["data_source_mode"] == "SIMULATOR"
            assert data["is_fallback"] is True
            assert "timed out" in data["fallback_reason"].lower()
            assert data["ml_status"] == "AVAILABLE"
            assert "super_secret_token_123" not in resp.text


# ---------------------------------------------------------------------------
# 4. System Diagnostics Endpoint
# ---------------------------------------------------------------------------

def test_system_data_source_endpoint(client):
    """Verify GET /system/data-source reveals mode and masked key status."""
    with patch.dict(os.environ, {"DATA_SOURCE_MODE": "SIMULATOR", "RAILRADAR_API_KEY": "dummy_key"}):
        resp = client.get("/system/data-source")
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured_mode"] == "SIMULATOR"
        assert "SIMULATOR" in data["supported_modes"]
        assert "LIVE_API" in data["supported_modes"]
        assert data["has_api_key"] is True
        # Never reveals key
        assert "dummy_key" not in resp.text
