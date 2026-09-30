import os
import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import httpx
import pytest
from pydantic import ValidationError

from backend.services.schemas import TrainRunningState
from backend.services.railway_api_client import (
    RailRadarClient,
    MissingApiKeyError,
    AuthenticationError,
    TrainNotFoundError,
    RateLimitExceededError,
    APITimeoutError,
    MalformedResponseError,
    ServiceUnavailableError,
    NormalizedLiveTrain,
    sanitize_secret,
)


@pytest.fixture
def mock_live_response():
    return {
        "success": True,
        "data": {
            "trainNumber": "12302",
            "trainName": "Howrah Rajdhani Express",
            "startDate": "2026-09-26",
            "status": "RUNNING",
            "currentLocation": {
                "stationCode": "CNB",
                "sequence": 2,
                "delayMinutes": 15.5,
                "segmentProgress": 0.35,
                "speedKmh": 110.0,
            },
            "previousStation": {
                "stationCode": "NDLS",
            },
            "nextHalt": {
                "stationCode": "PRYJ",
                "distance": 195.0,
            },
            "lastUpdatedAt": "2026-09-26T21:40:00+05:30",
        },
    }


def test_missing_api_key():
    """Verify that MissingApiKeyError is raised when RAILRADAR_API_KEY is unset."""
    with patch.dict(os.environ, {}, clear=True):
        client = RailRadarClient(api_key=None)
        with pytest.raises(MissingApiKeyError) as exc_info:
            client.fetch_live_train("12302")
        assert "RAILRADAR_API_KEY is not set" in str(exc_info.value)


def test_successful_fetch_and_normalization(mock_live_response):
    """Verify correct normalization of RailRadar payload into TrainRunningState with all 14 fields."""
    client = RailRadarClient(api_key="mock_secret_key_123")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_live_response

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        result = client.fetch_live_train("12302")

        # Verify HTTP call
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        called_headers = mock_get.call_args[1]["headers"]
        assert called_url.endswith("/v1/trains/12302/live")
        assert called_headers["X-API-Key"] == "mock_secret_key_123"

        # Verify Normalized fields (all 14 fields)
        assert isinstance(result, TrainRunningState)
        assert isinstance(result, NormalizedLiveTrain)  # Backward compat alias
        assert result.train_number == "12302"
        assert result.journey_date == date(2026, 9, 26)
        assert result.train_name == "Howrah Rajdhani Express"
        assert result.status == "RUNNING"
        assert result.current_station_code == "CNB"
        assert result.current_station_sequence == 2
        assert result.current_delay_minutes == 15.5
        assert result.previous_station_code == "NDLS"
        assert result.next_station_code == "PRYJ"
        assert result.next_station_distance_km == 195.0
        assert result.segment_progress == 0.35
        assert result.speed_kmh == 110.0
        assert isinstance(result.timestamp, datetime)
        assert result.source == "external_api"


def test_flat_payload_normalization():
    """Verify that flat/snake_case payloads are properly parsed into TrainRunningState."""
    client = RailRadarClient(api_key="mock_secret_key_123")
    flat_data = {
        "train_number": "12952",
        "journey_date": "2026-09-26",
        "train_name": "Mumbai Rajdhani",
        "status": "RUNNING",
        "current_station_code": "ST",
        "current_station_sequence": 5,
        "delay_minutes": 10.0,
        "previous_station_code": "BRC",
        "next_station_code": "BVI",
        "next_station_distance_km": 230.0,
        "segment_progress": 0.5,
        "speed_kmh": 95.0,
        "last_updated": "2026-09-26T18:00:00Z",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = flat_data

    with patch("httpx.Client.get", return_value=mock_resp):
        state = client.fetch_live_train("12952")
        assert state.train_number == "12952"
        assert state.journey_date == date(2026, 9, 26)
        assert state.train_name == "Mumbai Rajdhani"
        assert state.status == "RUNNING"
        assert state.current_station_code == "ST"
        assert state.current_station_sequence == 5
        assert state.current_delay_minutes == 10.0
        assert state.previous_station_code == "BRC"
        assert state.next_station_code == "BVI"
        assert state.next_station_distance_km == 230.0
        assert state.segment_progress == 0.5
        assert state.speed_kmh == 95.0
        assert state.source == "external_api"


def test_train_running_state_source_validation():
    """Verify that TrainRunningState strictly enforces valid source literals."""
    valid_kwargs = {
        "train_number": "12302",
        "journey_date": date(2026, 9, 26),
        "train_name": "Howrah Rajdhani",
        "status": "RUNNING",
        "timestamp": datetime.now(),
        "source": "simulator",
    }
    state = TrainRunningState(**valid_kwargs)
    assert state.source == "simulator"

    valid_kwargs["source"] = "external_api"
    state2 = TrainRunningState(**valid_kwargs)
    assert state2.source == "external_api"

    # Invalid source should raise ValidationError
    valid_kwargs["source"] = "invalid_source"
    with pytest.raises(ValidationError):
        TrainRunningState(**valid_kwargs)


def test_http_401_unauthorized():
    """Verify handling of invalid or expired API key (HTTP 401)."""
    client = RailRadarClient(api_key="bad_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized: Invalid API key"

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(AuthenticationError) as exc_info:
            client.fetch_live_train("12302")
        assert "HTTP 401" in str(exc_info.value)


def test_http_404_not_found():
    """Verify handling when requested train is not found (HTTP 404)."""
    client = RailRadarClient(api_key="valid_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "Not Found: Train 99999 not tracked"

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(TrainNotFoundError) as exc_info:
            client.fetch_live_train("99999")
        assert "HTTP 404" in str(exc_info.value)
        assert "99999" in str(exc_info.value)


def test_http_429_rate_limit():
    """Verify handling of rate-limiting/quota exhaustion (HTTP 429)."""
    client = RailRadarClient(api_key="valid_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Too Many Requests"

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(RateLimitExceededError) as exc_info:
            client.fetch_live_train("12302")
        assert "HTTP 429" in str(exc_info.value)


def test_timeout_handling():
    """Verify handling of connection/request timeouts."""
    client = RailRadarClient(api_key="valid_key", timeout_seconds=1.0)

    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Connection timed out")):
        with pytest.raises(APITimeoutError) as exc_info:
            client.fetch_live_train("12302")
        assert "timed out" in str(exc_info.value).lower()


def test_malformed_json_handling():
    """Verify handling when the server returns non-JSON or HTML error page."""
    client = RailRadarClient(api_key="valid_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html>502 Bad Gateway</html>"
    mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "doc", 0)

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(MalformedResponseError) as exc_info:
            client.fetch_live_train("12302")
        assert "Malformed JSON" in str(exc_info.value)


def test_debug_file_output(mock_live_response, tmp_path):
    """Verify that raw JSON response is dumped to file only when explicitly enabled."""
    client = RailRadarClient(api_key="valid_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_live_response

    debug_file = tmp_path / "debug_output.json"
    assert not debug_file.exists()

    with patch("httpx.Client.get", return_value=mock_resp):
        # 1. Fetch with debug enabled
        client.fetch_live_train("12302", debug_output_file=debug_file)
        assert debug_file.exists()

        with open(debug_file, "r") as f:
            dumped = json.load(f)
        assert dumped["data"]["trainNumber"] == "12302"


def test_http_503_service_unavailable():
    """Verify handling of RailRadar temporary service unavailability (HTTP 503)."""
    client = RailRadarClient(api_key="valid_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable: Down for maintenance"

    with patch("httpx.Client.get", return_value=mock_resp):
        with pytest.raises(ServiceUnavailableError) as exc_info:
            client.fetch_live_train("12302")
        assert "HTTP 503" in str(exc_info.value)
        assert "temporarily unavailable" in str(exc_info.value).lower()


def test_sanitize_secret():
    """Verify that sanitize_secret completely strips API keys from strings."""
    secret = "rr_live_supersecret12345"
    sample_text = f"Failed to connect using key {secret} at https://api.railradar.in?key={secret}"

    cleaned = sanitize_secret(sample_text, secret=secret)
    assert secret not in cleaned
    assert "***MASKED_API_KEY***" in cleaned


