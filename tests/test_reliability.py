"""
Comprehensive Reliability Test Suite.
=====================================
Tests and validates robust error handling, structured error responses,
and graceful degradation across the 11 required failure scenarios:

1. External API unavailable
2. API key invalid
3. API rate limited
4. Train not found
5. Station not found
6. ML model missing
7. Database unavailable
8. Simulator stopped
9. Empty station arrivals
10. Network timeout
11. Malformed external API data

Enforces:
- Backend returns structured errors
- Application does not crash (no unhandled 500s)
- Baseline ETA remains available when ML fails
- Simulator mode remains available when external API fails
"""

import os
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

from backend.main import (
    app,
    get_db,
    get_active_simulator,
    get_multi_station_service,
    get_metrics_filepath,
    get_metadata_filepath,
)
from backend.database.connection import Base
from backend.database.seed import seed_data
from backend.simulator.engine import TrainSimulator, SimulatorStoppedError
from backend.simulator.events import EventType
from backend.services.schemas import TrainRunningState
from backend.services.data_source import TrainStateProvider, DataSourceMode
from backend.services.multi_station_eta import MultiStationETAService
from backend.services.railway_api_client import (
    RailRadarClient,
    RailwayAPIError,
    MissingApiKeyError,
    AuthenticationError,
    TrainNotFoundError,
    RateLimitExceededError,
    APITimeoutError,
    MalformedResponseError,
    ServiceUnavailableError,
)


@pytest.fixture
def reliability_env():
    """Sets up an in-memory SQLite database and isolated simulator."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    seed_data(db)

    sim = TrainSimulator(simulation_speed=1.0, db=db)
    sim.load_from_db(db)

    def override_get_db():
        try:
            yield db
        finally:
            pass

    def override_get_simulator():
        return sim

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_active_simulator] = override_get_simulator

    with TestClient(app) as client:
        yield client, db, sim

    app.dependency_overrides.clear()
    db.close()


# ---------------------------------------------------------------------------
# 1. External API Unavailable (503 / Network Error)
# ---------------------------------------------------------------------------

def test_1_external_api_unavailable_returns_structured_error_and_simulator_fallback(reliability_env):
    """
    Scenario 1: External API unavailable (HTTP 503 / network connection failure).
    - /live/train/{number} returns structured 503 indicating simulator mode is available.
    - /train/{number}?data_source=LIVE_API performs controlled fallback to SIMULATOR mode.
    """
    client, _, _ = reliability_env
    exc = ServiceUnavailableError("RailRadar service temporarily unavailable (HTTP 503)")

    with patch.object(RailRadarClient, "fetch_live_train", side_effect=exc):
        # 1. Live endpoint returns structured error
        resp = client.get("/live/train/12302")
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"] == "LIVE_DATA_UNAVAILABLE"
        assert data["simulator_mode_available"] is True
        assert data["simulator_available"] is True
        assert "/train/12302" in data["simulator_url"]
        assert "simulator" in data["message"].lower()

        # 2. Main train details endpoint automatically falls back to SIMULATOR
        resp_train = client.get("/train/12302?data_source=LIVE_API")
        assert resp_train.status_code == 200
        train_data = resp_train.json()
        assert train_data["data_source_mode"] == "SIMULATOR"
        assert train_data["is_fallback"] is True
        assert train_data["baseline_eta"] is not None
        assert train_data["ml_eta"] is not None


# ---------------------------------------------------------------------------
# 2. API Key Invalid (401 / Missing Key)
# ---------------------------------------------------------------------------

def test_2_api_key_invalid_never_leaks_secret_and_falls_back(reliability_env):
    """
    Scenario 2: API key invalid (HTTP 401) or missing.
    - Returns structured error indicating live data unavailable.
    - Never leaks the secret key in response or error details.
    - Preserves simulator fallback.
    """
    client, _, _ = reliability_env
    secret_key = "super_secret_test_key_xyz_123"

    with patch.dict(os.environ, {"RAILRADAR_API_KEY": secret_key}):
        exc = AuthenticationError(f"HTTP 401 Unauthorized for key: {secret_key}")
        with patch.object(RailRadarClient, "fetch_live_train", side_effect=exc):
            resp = client.get("/live/train/12302")
            assert resp.status_code == 503
            assert secret_key not in resp.text
            assert "***MASKED_API_KEY***" in resp.text
            data = resp.json()
            assert data["simulator_mode_available"] is True

            # Train detail view falls back cleanly
            resp_train = client.get("/train/12302?data_source=LIVE_API")
            assert resp_train.status_code == 200
            assert secret_key not in resp_train.text
            assert resp_train.json()["data_source_mode"] == "SIMULATOR"


# ---------------------------------------------------------------------------
# 3. API Rate Limited (429)
# ---------------------------------------------------------------------------

def test_3_api_rate_limited_returns_structured_error_and_serves_cache_if_available(reliability_env):
    """
    Scenario 3: API rate limit reached (HTTP 429).
    - When rate limited, returns structured error indicating cooldown and simulator availability.
    - Does not crash the application.
    """
    client, _, _ = reliability_env
    exc = RateLimitExceededError("Rate limit exceeded (HTTP 429): Quota reached on RailRadar API.")

    with patch.object(RailRadarClient, "fetch_live_train", side_effect=exc):
        resp = client.get("/live/train/12302")
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"] == "LIVE_DATA_UNAVAILABLE"
        assert data["simulator_mode_available"] is True
        assert "HTTP 429" in data["detail"] or "rate limit" in data["detail"].lower()


# ---------------------------------------------------------------------------
# 4. Train Not Found (404)
# ---------------------------------------------------------------------------

def test_4_train_not_found_returns_structured_404(reliability_env):
    """
    Scenario 4: Train not found.
    - /train/99999 returns structured 404 with TRAIN_NOT_FOUND error code.
    - /train/99999/eta/PRYJ returns structured 404.
    - /live/train/99999 returns structured 404 indicating simulator availability.
    - Does not crash.
    """
    client, _, _ = reliability_env

    # 1. Main train detail endpoint
    resp = client.get("/train/99999")
    assert resp.status_code == 404
    data = resp.json()
    assert data["error"] == "TRAIN_NOT_FOUND"
    assert "not found" in data["detail"].lower()

    # 2. Train station ETA endpoint
    resp_eta = client.get("/train/99999/eta/PRYJ")
    assert resp_eta.status_code == 404
    assert resp_eta.json()["error"] == "TRAIN_NOT_FOUND"

    # 3. Live train lookup
    err = TrainNotFoundError("Train 99999 not found in live API")
    with patch.object(RailRadarClient, "fetch_live_train", side_effect=err):
        resp_live = client.get("/live/train/99999")
        assert resp_live.status_code == 404
        live_data = resp_live.json()
        assert live_data["error"] == "TRAIN_NOT_FOUND"
        assert live_data["simulator_mode_available"] is True


# ---------------------------------------------------------------------------
# 5. Station Not Found (404 / 400)
# ---------------------------------------------------------------------------

def test_5_station_not_found_returns_structured_error(reliability_env):
    """
    Scenario 5: Station not found.
    - /station/UNKNOWN_CODE/arrivals returns structured 404 with STATION_NOT_FOUND.
    - /train/12302/eta/UNKNOWN_CODE returns structured 400 with STATION_NOT_ON_ROUTE.
    - Does not crash.
    """
    client, _, _ = reliability_env

    # 1. Station arrivals for unknown station
    resp_stn = client.get("/station/UNKNOWN_CODE/arrivals")
    assert resp_stn.status_code == 404
    data_stn = resp_stn.json()
    assert data_stn["error"] == "STATION_NOT_FOUND"
    assert "not found" in data_stn["detail"].lower()

    # 2. Train ETA for station not on route
    resp_eta = client.get("/train/12302/eta/MAS")
    assert resp_eta.status_code == 400
    data_eta = resp_eta.json()
    assert data_eta["error"] == "STATION_NOT_ON_ROUTE"
    assert "not on the route" in data_eta["detail"].lower()


# ---------------------------------------------------------------------------
# 6. ML Model Missing (Graceful Degradation to Baseline ETA)
# ---------------------------------------------------------------------------

def test_6_ml_model_missing_preserves_baseline_eta(reliability_env):
    """
    Scenario 6: ML model file missing or predictor failure.
    - Baseline ETA remains 100% available across all endpoints.
    - /train/{id} returns upcoming stations with Baseline ETAs.
    - /train/{id}/eta/{station} returns HTTP 200 with ml_status='UNAVAILABLE' and valid baseline_eta.
    - /station/{code}/arrivals continues operating using Baseline ETAs.
    - /model/metrics returns is_available=False without crashing.
    """
    client, _, _ = reliability_env

    class BrokenMLPredictor:
        def predict(self, *args, **kwargs):
            raise FileNotFoundError("eta_xgboost_model.json not found on disk")

    class FailingMLEtaService:
        def predict_upcoming_stations(self, *args, **kwargs):
            raise RuntimeError("XGBoost booster failed to load artifact")

        def predict_station_eta(self, *args, **kwargs):
            raise RuntimeError("XGBoost booster failed to load artifact")

    # 1. Single station ETA when ML fails
    app.dependency_overrides[get_multi_station_service] = lambda: FailingMLEtaService()
    try:
        resp_eta = client.get("/train/12302/eta/PRYJ")
        assert resp_eta.status_code == 200
        eta_data = resp_eta.json()
        assert eta_data["ml_status"] == "UNAVAILABLE"
        assert eta_data["baseline_eta"] is not None
        assert eta_data["scheduled_eta"] is not None

        # 2. Train detail view when ML fails: upcoming_stations still populated with Baseline ETAs
        resp_train = client.get("/train/12302")
        assert resp_train.status_code == 200
        train_data = resp_train.json()
        assert train_data["baseline_eta"] is not None
        assert len(train_data["upcoming_stations"]) > 0
        first_upcoming = train_data["upcoming_stations"][0]
        assert first_upcoming["baseline_eta"] is not None
        assert first_upcoming["scheduled_eta"] is not None

        # 3. Station arrivals when ML fails: uses Baseline ETA
        resp_arr = client.get("/station/PRYJ/arrivals")
        assert resp_arr.status_code == 200
        arr_data = resp_arr.json()
        assert arr_data["total_arrivals"] >= 1
        assert arr_data["arrivals"][0]["baseline_eta"] is not None

    finally:
        app.dependency_overrides.pop(get_multi_station_service, None)

    # 4. Metrics endpoint when metrics file is missing
    app.dependency_overrides[get_metrics_filepath] = lambda: Path("/nonexistent/metrics.json")
    app.dependency_overrides[get_metadata_filepath] = lambda: Path("/nonexistent/metadata.json")
    try:
        resp_metrics = client.get("/model/metrics")
        assert resp_metrics.status_code == 200
        metrics_data = resp_metrics.json()
        assert metrics_data["is_available"] is False
        assert metrics_data["status"] == "UNAVAILABLE"
    finally:
        app.dependency_overrides.pop(get_metrics_filepath, None)
        app.dependency_overrides.pop(get_metadata_filepath, None)


# ---------------------------------------------------------------------------
# 7. Database Unavailable (Connection / Operational Error)
# ---------------------------------------------------------------------------

def test_7_database_unavailable_returns_structured_503(reliability_env):
    """
    Scenario 7: Database unavailable (OperationalError / connection loss).
    - Global exception handler intercepts SQLAlchemyError.
    - Returns structured 503 response with DATABASE_UNAVAILABLE.
    - Does not emit raw Python traceback or unhandled 500 crash.
    """
    client, _, _ = reliability_env

    def failing_get_db():
        raise OperationalError("sqlite3.OperationalError: database is locked", params=None, orig=None)

    app.dependency_overrides[get_db] = failing_get_db
    try:
        resp = client.get("/train/12302")
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"] == "DATABASE_UNAVAILABLE"
        assert "database is temporarily unavailable" in data["message"].lower()
    finally:
        app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# 8. Simulator Stopped (Engine Inactive / Paused)
# ---------------------------------------------------------------------------

def test_8_simulator_stopped_returns_structured_503(reliability_env):
    """
    Scenario 8: Simulator stopped or paused.
    - Simulator operations raise SimulatorStoppedError.
    - /train/{id} returns structured 503 with SIMULATOR_STOPPED.
    - /station/{code}/arrivals returns structured 503 with SIMULATOR_STOPPED.
    - /simulate/event returns structured 503 with SIMULATOR_STOPPED.
    - /demo/reset returns structured 503 with SIMULATOR_STOPPED.
    - Resuming simulator restores normal operation immediately.
    """
    client, _, sim = reliability_env

    # Stop the simulation engine
    sim.stop()
    assert sim.is_running is False

    # 1. Train details returns structured 503
    resp_train = client.get("/train/12302")
    assert resp_train.status_code == 503
    assert resp_train.json()["error"] == "SIMULATOR_STOPPED"

    # 2. Station arrivals returns structured 503
    resp_stn = client.get("/station/PRYJ/arrivals")
    assert resp_stn.status_code == 503
    assert resp_stn.json()["error"] == "SIMULATOR_STOPPED"

    # 3. Disruption injection returns structured 503
    resp_evt = client.post(
        "/simulate/event",
        json={"train_id": "12302", "event_type": "SIGNAL_HALT", "delay_minutes": 10.0},
    )
    assert resp_evt.status_code == 503
    assert resp_evt.json()["error"] == "SIMULATOR_STOPPED"

    # 4. Demo reset returns structured 503
    resp_demo = client.post("/demo/reset")
    assert resp_demo.status_code == 503
    assert resp_demo.json()["error"] == "SIMULATOR_STOPPED"

    # 5. Restarting simulator restores 200 responses
    sim.start()
    assert sim.is_running is True

    resp_restored = client.get("/train/12302")
    assert resp_restored.status_code == 200
    assert resp_restored.json()["train_number"] == "12302"


# ---------------------------------------------------------------------------
# 9. Empty Station Arrivals
# ---------------------------------------------------------------------------

def test_9_empty_station_arrivals_handled_cleanly(reliability_env):
    """
    Scenario 9: Empty station arrivals board.
    - Returns HTTP 200 with total_arrivals=0 and arrivals=[].
    - Does not crash or return null.
    """
    client, _, _ = reliability_env

    # NDLS is route origin; no trains approach from further north
    resp = client.get("/station/NDLS/arrivals")
    assert resp.status_code == 200
    data = resp.json()
    assert data["station_code"] == "NDLS"
    assert data["total_arrivals"] == 0
    assert data["arrivals"] == []


# ---------------------------------------------------------------------------
# 10. Network Timeout
# ---------------------------------------------------------------------------

def test_10_network_timeout_returns_structured_error_and_simulator_fallback(reliability_env):
    """
    Scenario 10: Network timeout when calling external live API.
    - /live/train/{number} returns structured 503 with simulator availability.
    - /train/{number}?data_source=LIVE_API gracefully falls back to simulator.
    """
    client, _, _ = reliability_env
    exc = APITimeoutError("Connection to RailRadar API timed out after 10.0s for train 12302.")

    with patch.object(RailRadarClient, "fetch_live_train", side_effect=exc):
        # Live endpoint
        resp = client.get("/live/train/12302")
        assert resp.status_code == 503
        data = resp.json()
        assert data["simulator_mode_available"] is True
        assert "timed out" in data["detail"].lower()

        # Fallback to simulator
        resp_train = client.get("/train/12302?data_source=LIVE_API")
        assert resp_train.status_code == 200
        assert resp_train.json()["data_source_mode"] == "SIMULATOR"
        assert resp_train.json()["is_fallback"] is True


# ---------------------------------------------------------------------------
# 11. Malformed External API Data
# ---------------------------------------------------------------------------

def test_11_malformed_external_api_data_handled_cleanly(reliability_env):
    """
    Scenario 11: External API returns malformed payload.
    - Malformed payload (non-dict, corrupted types, HTML body) is caught.
    - /live/train/{number} returns structured 503 without crashing.
    - /train/{number}?data_source=LIVE_API gracefully falls back to simulator.
    """
    client, _, _ = reliability_env
    exc = MalformedResponseError("Malformed JSON response received from API: <html>502 Bad Gateway</html>")

    with patch.object(RailRadarClient, "fetch_live_train", side_effect=exc):
        # Live endpoint
        resp = client.get("/live/train/12302")
        assert resp.status_code == 503
        data = resp.json()
        assert data["simulator_mode_available"] is True
        assert "malformed" in data["detail"].lower()

        # Fallback to simulator
        resp_train = client.get("/train/12302?data_source=LIVE_API")
        assert resp_train.status_code == 200
        assert resp_train.json()["data_source_mode"] == "SIMULATOR"
        assert resp_train.json()["is_fallback"] is True
