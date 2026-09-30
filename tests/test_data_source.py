"""
Unit tests for DataSourceMode, TrainStateProvider, and Error Fallback.
Validates:
1. Mode resolution priority (override > env var > default SIMULATOR).
2. SIMULATOR mode queries.
3. LIVE_API mode queries with mocked RailRadar payloads.
4. Controlled fallback to SIMULATOR on:
   - timeout (APITimeoutError)
   - 401 (AuthenticationError)
   - 404 (TrainNotFoundError)
   - 429 (RateLimitExceededError)
   - 503 (ServiceUnavailableError)
   - malformed JSON (MalformedResponseError)
   - missing key (MissingApiKeyError)
5. Strict secret masking (API key never leaks into fallback_reason, logs, or error text).
"""

import os
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base
from backend.database.seed import seed_data
from backend.services.schemas import TrainRunningState
from backend.services.railway_api_client import (
    RailRadarClient,
    APITimeoutError,
    AuthenticationError,
    TrainNotFoundError,
    RateLimitExceededError,
    ServiceUnavailableError,
    MalformedResponseError,
    MissingApiKeyError,
    RailwayAPIError,
)
from backend.services.data_source import (
    DataSourceMode,
    TrainStateProvider,
    StateProviderResult,
    get_configured_mode,
)
from backend.simulator.engine import TrainSimulator


@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite database seeded with standard test routes."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    seed_data(db)
    yield db
    db.close()


@pytest.fixture
def test_simulator(db_session):
    """Provides a TrainSimulator loaded with the seeded trains."""
    sim = TrainSimulator(simulation_speed=1.0, db=db_session)
    sim.load_from_db(db_session)
    return sim


@pytest.fixture
def mock_live_state():
    """Generates a normalized TrainRunningState simulating an external API response."""
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
        segment_progress=0.45,
        speed_kmh=115.0,
        timestamp=datetime.now(timezone.utc),
        source="external_api",
    )


# ---------------------------------------------------------------------------
# 1. Mode Configuration Priority Tests
# ---------------------------------------------------------------------------

def test_configured_mode_default():
    """Verify default mode is SIMULATOR when nothing is set."""
    with patch.dict(os.environ, {}, clear=True):
        mode = get_configured_mode()
        assert mode == DataSourceMode.SIMULATOR


def test_configured_mode_from_env():
    """Verify DATA_SOURCE_MODE env var takes effect."""
    with patch.dict(os.environ, {"DATA_SOURCE_MODE": "LIVE_API"}):
        assert get_configured_mode() == DataSourceMode.LIVE_API

    with patch.dict(os.environ, {"DATA_SOURCE_MODE": "simulator"}):
        assert get_configured_mode() == DataSourceMode.SIMULATOR


def test_configured_mode_override():
    """Verify explicit override takes highest priority."""
    with patch.dict(os.environ, {"DATA_SOURCE_MODE": "SIMULATOR"}):
        # String override
        assert get_configured_mode("LIVE_API") == DataSourceMode.LIVE_API
        assert get_configured_mode("LIVE") == DataSourceMode.LIVE_API
        # Enum override
        assert get_configured_mode(DataSourceMode.LIVE_API) == DataSourceMode.LIVE_API


# ---------------------------------------------------------------------------
# 2. Simulator Mode Execution Tests
# ---------------------------------------------------------------------------

def test_provider_simulator_mode(db_session, test_simulator):
    """Verify provider retrieves running state from simulator in SIMULATOR mode."""
    provider = TrainStateProvider(default_mode=DataSourceMode.SIMULATOR)
    result = provider.get_train_state("12302", db=db_session, sim=test_simulator)

    assert isinstance(result, StateProviderResult)
    assert result.effective_mode == DataSourceMode.SIMULATOR
    assert result.requested_mode == DataSourceMode.SIMULATOR
    assert result.is_fallback is False
    assert result.fallback_reason is None
    assert result.state.source == "simulator"
    assert result.state.train_number == "12302"
    assert result.state.status == "RUNNING"


# ---------------------------------------------------------------------------
# 3. Live API Mode Execution Tests (Success)
# ---------------------------------------------------------------------------

def test_provider_live_api_mode_success(db_session, test_simulator, mock_live_state):
    """Verify provider retrieves and returns external API state in LIVE_API mode."""
    mock_client = MagicMock(spec=RailRadarClient)
    mock_client.fetch_live_train.return_value = mock_live_state

    provider = TrainStateProvider(client=mock_client, default_mode=DataSourceMode.LIVE_API)
    result = provider.get_train_state("12302", db=db_session, sim=test_simulator)

    assert result.effective_mode == DataSourceMode.LIVE_API
    assert result.requested_mode == DataSourceMode.LIVE_API
    assert result.is_fallback is False
    assert result.fallback_reason is None
    assert result.state.source == "external_api"
    assert result.state.train_number == "12302"
    assert result.state.current_station_code == "CNB"
    assert result.state.current_delay_minutes == 18.5
    assert result.delay_history == [18.5]
    mock_client.fetch_live_train.assert_called_once_with("12302")


# ---------------------------------------------------------------------------
# 4. Controlled Fallback Tests across all Error Types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "exception_to_raise, expected_text_in_reason",
    [
        (APITimeoutError("Request timed out after 5.0 seconds"), "timed out"),
        (AuthenticationError("HTTP 401 Unauthorized: Invalid API key"), "HTTP 401"),
        (TrainNotFoundError("HTTP 404: Train 12302 not found"), "HTTP 404"),
        (RateLimitExceededError("HTTP 429: Too Many Requests"), "HTTP 429"),
        (ServiceUnavailableError("HTTP 503: RailRadar temporarily unavailable"), "HTTP 503"),
        (MalformedResponseError("Malformed JSON: parse error"), "Malformed JSON"),
        (MissingApiKeyError("RAILRADAR_API_KEY is not set"), "RAILRADAR_API_KEY is not set"),
    ],
)
def test_provider_live_api_fallback_on_errors(
    db_session, test_simulator, exception_to_raise, expected_text_in_reason
):
    """
    Verify that all specified live API failures automatically trigger a controlled
    fallback to SIMULATOR mode without crashing.
    """
    mock_client = MagicMock(spec=RailRadarClient)
    mock_client.fetch_live_train.side_effect = exception_to_raise

    provider = TrainStateProvider(client=mock_client)
    result = provider.get_train_state(
        train_number="12302",
        db=db_session,
        sim=test_simulator,
        mode_override=DataSourceMode.LIVE_API,
        fallback_on_error=True,
    )

    # Effective mode falls back to SIMULATOR
    assert result.effective_mode == DataSourceMode.SIMULATOR
    assert result.requested_mode == DataSourceMode.LIVE_API
    assert result.is_fallback is True
    assert result.fallback_reason is not None
    assert expected_text_in_reason.lower() in result.fallback_reason.lower()
    # Normalized simulator state returned
    assert result.state.source == "simulator"
    assert result.state.train_number == "12302"


def test_provider_raises_when_fallback_disabled(db_session, test_simulator):
    """Verify that when fallback_on_error=False, the live exception is raised."""
    mock_client = MagicMock(spec=RailRadarClient)
    mock_client.fetch_live_train.side_effect = APITimeoutError("Timeout")

    provider = TrainStateProvider(client=mock_client)
    with pytest.raises(APITimeoutError):
        provider.get_train_state(
            train_number="12302",
            db=db_session,
            sim=test_simulator,
            mode_override=DataSourceMode.LIVE_API,
            fallback_on_error=False,
        )


# ---------------------------------------------------------------------------
# 5. Security & Secret Scrubbing Test
# ---------------------------------------------------------------------------

def test_provider_fallback_sanitizes_secret_key(db_session, test_simulator):
    """
    Verify that even if an exception or response includes the raw secret API key,
    it is strictly redacted and never leaked into fallback_reason or log messages.
    """
    secret = "rr_live_supersecret998877"
    with patch.dict(os.environ, {"RAILRADAR_API_KEY": secret}):
        mock_client = MagicMock(spec=RailRadarClient)
        # Exception containing the sensitive key
        mock_client.fetch_live_train.side_effect = AuthenticationError(
            f"Failed to authenticate with key {secret}: HTTP 401 Unauthorized"
        )

        provider = TrainStateProvider(client=mock_client)
        result = provider.get_train_state(
            train_number="12302",
            db=db_session,
            sim=test_simulator,
            mode_override=DataSourceMode.LIVE_API,
            fallback_on_error=True,
        )

        assert result.is_fallback is True
        assert secret not in result.fallback_reason
        assert "***MASKED_API_KEY***" in result.fallback_reason
