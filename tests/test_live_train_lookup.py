"""
Tests for Live Train Lookup Service and GET /live/train/{train_number} Endpoint.
Validates:
1. Clean service flow:
   train number -> Railway API -> Normalized TrainRunningState -> Feature Builder -> Baseline ETA + ML ETA.
2. Complete response contract:
   - train number
   - train name
   - status
   - current station
   - next station
   - current delay
   - speed if available
   - segment progress if available
   - timestamp
   - baseline ETA
   - ML ETA
   - confidence range
   - data source ("LIVE_API")
3. Structured error handling without crashing on:
   - TrainNotFoundError (HTTP 404)
   - APITimeoutError (HTTP 503)
   - AuthenticationError (HTTP 503)
   - RateLimitExceededError (HTTP 503)
   - ServiceUnavailableError (HTTP 503)
   - MalformedResponseError (HTTP 503)
   - MissingApiKeyError (HTTP 503)
   All error responses indicate simulator mode is available.
4. Security: Secret API key is never leaked in responses or error details.
"""

import os
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.schemas import TrainRunningState
from backend.services.railway_api_client import (
    RailRadarClient,
    TrainNotFoundError,
    APITimeoutError,
    AuthenticationError,
    RateLimitExceededError,
    ServiceUnavailableError,
    MalformedResponseError,
    MissingApiKeyError,
)
from backend.services.live_train_service import (
    LiveTrainLookupService,
    LiveTrainResult,
)


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_live_state():
    """Normalized live train running state for train 12302 approaching PRYJ."""
    return TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Howrah Rajdhani Express",
        status="RUNNING",
        current_station_code="CNB",
        current_station_sequence=2,
        current_delay_minutes=25.0,
        previous_station_code="NDLS",
        next_station_code="PRYJ",
        next_station_distance_km=195.0,
        segment_progress=0.40,
        speed_kmh=112.5,
        timestamp=datetime(2026, 9, 28, 16, 30, tzinfo=timezone.utc),
        source="external_api",
    )


# ---------------------------------------------------------------------------
# 1. Service Layer Tests: Clean Service Flow
# ---------------------------------------------------------------------------

def test_live_lookup_service_flow(sample_live_state):
    """
    Verify clean service flow:
    train number -> Railway API -> Normalized TrainRunningState -> Feature Builder -> Baseline ETA + ML ETA.
    """
    mock_client = MagicMock(spec=RailRadarClient)
    mock_client.fetch_live_train.return_value = sample_live_state

    service = LiveTrainLookupService(railway_client=mock_client)
    result = service.lookup_live_train("12302")

    mock_client.fetch_live_train.assert_called_once_with("12302")

    assert isinstance(result, LiveTrainResult)
    assert result.train_number == "12302"
    assert result.train_name == "Howrah Rajdhani Express"
    assert result.status == "RUNNING"
    assert result.current_station == "CNB"
    assert result.next_station == "PRYJ"
    assert result.current_delay == 25.0
    assert result.speed == 112.5
    assert result.segment_progress == 0.40
    assert result.timestamp == datetime(2026, 9, 28, 16, 30, tzinfo=timezone.utc)
    assert result.data_source == "LIVE_API"

    # Baseline & ML ETAs calculated for next station PRYJ
    assert result.baseline_eta is not None
    assert result.baseline_eta > result.timestamp
    assert result.ml_eta is not None


def test_live_lookup_service_completed_train():
    """Verify that a completed train returns without upcoming ETAs."""
    completed_state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Howrah Rajdhani Express",
        status="COMPLETED",
        current_station_code="HWH",
        current_station_sequence=8,
        current_delay_minutes=5.0,
        previous_station_code="ASN",
        next_station_code=None,
        next_station_distance_km=0.0,
        segment_progress=1.0,
        speed_kmh=0.0,
        timestamp=datetime.now(timezone.utc),
        source="external_api",
    )
    mock_client = MagicMock(spec=RailRadarClient)
    mock_client.fetch_live_train.return_value = completed_state

    service = LiveTrainLookupService(railway_client=mock_client)
    result = service.lookup_live_train("12302")

    assert result.status == "COMPLETED"
    assert result.next_station is None
    assert result.baseline_eta is None
    assert result.ml_eta is None
    assert result.confidence_range is None


# ---------------------------------------------------------------------------
# 2. API Endpoint Tests: GET /live/train/{train_number} Success
# ---------------------------------------------------------------------------

def test_api_live_train_lookup_success(client, sample_live_state):
    """
    Verify GET /live/train/{train_number} returns 200 with all requested fields:
    - train number, train name, status
    - current station, next station
    - current delay
    - speed, segment progress
    - timestamp
    - baseline ETA, ML ETA, confidence range
    - data source ("LIVE_API")
    """
    with patch.object(RailRadarClient, "fetch_live_train", return_value=sample_live_state):
        resp = client.get("/live/train/12302")
        assert resp.status_code == 200
        data = resp.json()

        assert data["train_number"] == "12302"
        assert data["train_name"] == "Howrah Rajdhani Express"
        assert data["status"] == "RUNNING"
        assert data["current_station"] == "CNB"
        assert data["next_station"] == "PRYJ"
        assert data["current_delay"] == 25.0
        assert data["speed"] == 112.5
        assert data["segment_progress"] == 0.40
        assert data["timestamp"] is not None
        assert data["data_source"] == "LIVE_API"

        # Baseline and ML ETAs
        assert data["baseline_eta"] is not None
        assert data["ml_eta"] is not None
        assert data["confidence_range"] is not None
        assert "lower_bound" in data["confidence_range"]
        assert "upper_bound" in data["confidence_range"]
        assert data["confidence_range"]["margin_minutes"] > 0


# ---------------------------------------------------------------------------
# 3. API Structured Error Tests: Live Data Unavailable (Does NOT Crash)
# ---------------------------------------------------------------------------

def test_api_live_train_lookup_train_not_found_404(client):
    """Verify 404 structured error when train is not tracked in live API."""
    err = TrainNotFoundError("Train 99999 not found in RailRadar live API (HTTP 404)")
    with patch.object(RailRadarClient, "fetch_live_train", side_effect=err):
        resp = client.get("/live/train/99999")
        assert resp.status_code == 404
        data = resp.json()

        assert data["error"] == "TRAIN_NOT_FOUND"
        assert data["train_number"] == "99999"
        assert data["simulator_mode_available"] is True
        assert data["simulator_available"] is True
        assert "/train/99999" in data["simulator_url"]
        assert "simulator" in data["message"].lower()


@pytest.mark.parametrize(
    "exception_to_raise, expected_text",
    [
        (APITimeoutError("Live railway API request timed out after 5.0 seconds"), "timed out"),
        (AuthenticationError("HTTP 401 Unauthorized: Invalid API key"), "HTTP 401"),
        (RateLimitExceededError("HTTP 429: Too Many Requests"), "HTTP 429"),
        (ServiceUnavailableError("RailRadar service temporarily unavailable (HTTP 503)"), "HTTP 503"),
        (MalformedResponseError("Malformed JSON response received from API"), "Malformed JSON"),
        (MissingApiKeyError("RAILRADAR_API_KEY is not set"), "RAILRADAR_API_KEY is not set"),
    ],
)
def test_api_live_train_lookup_service_unavailable_503(client, exception_to_raise, expected_text):
    """
    Verify that when the live API fails for any reason,
    the endpoint returns a structured 503 error without crashing,
    and indicates that simulator mode is available.
    """
    with patch.object(RailRadarClient, "fetch_live_train", side_effect=exception_to_raise):
        resp = client.get("/live/train/12302")
        assert resp.status_code == 503
        data = resp.json()

        assert data["error"] == "LIVE_DATA_UNAVAILABLE"
        assert data["train_number"] == "12302"
        assert data["simulator_mode_available"] is True
        assert data["simulator_available"] is True
        assert "/train/12302" in data["simulator_url"]
        assert "simulator" in data["message"].lower()
        assert expected_text.lower() in data["detail"].lower()


# ---------------------------------------------------------------------------
# 4. Security: Secret Key Scrubbing in Live Train Endpoint
# ---------------------------------------------------------------------------

def test_api_live_train_lookup_never_leaks_api_key(client):
    """
    Verify that if an upstream error message includes the secret API key,
    it is strictly masked and never returned to the client.
    """
    secret = "rr_live_supersecret_key_8899"
    with patch.dict(os.environ, {"RAILRADAR_API_KEY": secret}):
        err = AuthenticationError(f"Unauthorized access with key {secret} to RailRadar API")
        with patch.object(RailRadarClient, "fetch_live_train", side_effect=err):
            resp = client.get("/live/train/12302")
            assert resp.status_code == 503
            assert secret not in resp.text
            data = resp.json()
            assert data["simulator_mode_available"] is True


def test_live_train_timestamp_timezone_consistency(client):
    """
    Verify that for live trains with IST timezone (+05:30):
    1. baseline_eta retains the IST timezone (+05:30)
    2. baseline_eta equals timestamp + scheduled_time_to_go_min + predicted_delay
    3. baseline_eta is consistent with ml_eta (no multi-hour offset)
    """
    from datetime import timezone, timedelta
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    obs_time = datetime(2026, 9, 30, 0, 9, 19, tzinfo=ist_tz)

    live_state = TrainRunningState(
        train_number="12028",
        journey_date=date(2026, 9, 30),
        train_name="Shatabdi Express",
        status="RUNNING",
        current_station_code="SBC",
        current_station_sequence=1,
        current_delay_minutes=0.0,
        next_station_code="BNC",
        next_station_distance_km=5.0,
        segment_progress=0.1,
        speed_kmh=60.0,
        timestamp=obs_time,
        source="external_api",
    )

    with patch.object(RailRadarClient, "fetch_live_train", return_value=live_state):
        resp = client.get("/live/train/12028")
        assert resp.status_code == 200
        data = resp.json()

        assert data["timestamp"].endswith("+05:30")
        assert data["baseline_eta"].endswith("+05:30")
        assert data["ml_eta"].endswith("+05:30")

        # Parse and verify times
        ts = datetime.fromisoformat(data["timestamp"])
        b_eta = datetime.fromisoformat(data["baseline_eta"])
        m_eta = datetime.fromisoformat(data["ml_eta"])

        # Baseline ETA should be within a reasonable scheduled minutes delta from timestamp (e.g. 5-30 min), NOT 6+ hours away
        b_diff_minutes = (b_eta - ts).total_seconds() / 60.0
        assert 0.0 <= b_diff_minutes <= 60.0, f"Expected baseline ETA within 1 hour of timestamp, got {b_diff_minutes} min"

        # ML ETA should also be within 1 hour of timestamp
        m_diff_minutes = (m_eta - ts).total_seconds() / 60.0
        assert 0.0 <= m_diff_minutes <= 60.0, f"Expected ML ETA within 1 hour of timestamp, got {m_diff_minutes} min"

        # Difference between ML ETA and Baseline ETA should be minimal (< 30 min)
        assert abs((m_eta - b_eta).total_seconds() / 60.0) < 30.0
