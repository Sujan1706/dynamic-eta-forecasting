"""
End-to-End Validation Test Suite for Dynamic ETA Forecasting.
=============================================================
Formally validates:
SCENARIO A — DEMO MODE:
  1. Start simulator
  2. Start FastAPI
  3. Start frontend
  4. Select prepared train
  5. Show scheduled ETA
  6. Show baseline ETA
  7. Show ML ETA
  8. Inject signal halt
  9. Verify train delay changes
  10. Verify features update
  11. Verify baseline ETA changes
  12. Verify ML ETA changes
  13. Verify station board updates
  14. Verify passenger view updates

SCENARIO B — LIVE API MODE:
  1. Start FastAPI
  2. Request one live train
  3. Normalize the external response
  4. Generate features
  5. Calculate baseline ETA
  6. Calculate ML ETA
  7. Display result in dashboard
  8. Verify last-updated timestamp
  9. Test API failure fallback
"""

import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app, get_active_simulator
from backend.services.schemas import TrainRunningState
from backend.services.railway_api_client import RailRadarClient, ServiceUnavailableError
from backend.services.baseline_eta import BaselineETAService
from backend.services.multi_station_eta import MultiStationETAService
from backend.features.feature_builder import FeatureBuilder
from backend.database.connection import SessionLocal


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session():
    db = SessionLocal()
    yield db
    db.close()


# ==============================================================================
# SCENARIO A — DEMO MODE
# ==============================================================================

def test_scenario_a_demo_mode_e2e(client, db_session):
    """
    Complete 14-step E2E validation of Scenario A: Deterministic Demo Mode.
    """
    # --------------------------------------------------------------------------
    # 1. Start simulator
    # --------------------------------------------------------------------------
    sim = get_active_simulator()
    assert sim is not None, "Simulator instance must be initialized"
    sim.start()
    assert sim.is_running is True, "Simulator must be in running state"

    # --------------------------------------------------------------------------
    # 2. Start FastAPI
    # --------------------------------------------------------------------------
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["service"] == "dynamic-eta-forecasting"

    # --------------------------------------------------------------------------
    # 3. Start frontend (verify API contracts required for Next.js frontend pages)
    # --------------------------------------------------------------------------
    sys_ds = client.get("/system/data-source")
    assert sys_ds.status_code == 200
    assert "configured_mode" in sys_ds.json()

    # --------------------------------------------------------------------------
    # 4. Select prepared train
    # --------------------------------------------------------------------------
    reset_resp = client.post("/demo/reset")
    assert reset_resp.status_code == 200
    init_state = reset_resp.json()
    assert init_state["status"] in ("SUCCESS", "PREPARED_INITIAL_STATE")
    assert init_state["seed"] == 42
    assert init_state["train_number"] == "12302"
    assert init_state["current_station"] == "CNB"
    assert init_state["next_station"] == "PRYJ"
    assert init_state["current_delay_minutes"] == 2.0
    assert init_state["current_speed_kmh"] == 110.0
    assert init_state["train_status"] == "RUNNING"

    # Also verify train details endpoint consumed by frontend dashboard
    train_pre = client.get("/train/12302").json()
    assert train_pre["train_number"] == "12302"
    assert train_pre["current_station"] == "CNB"

    # --------------------------------------------------------------------------
    # 5. Show scheduled ETA
    # --------------------------------------------------------------------------
    assert init_state["scheduled_eta"] is not None
    scheduled_eta_init = datetime.fromisoformat(init_state["scheduled_eta"])
    assert scheduled_eta_init is not None
    assert train_pre["scheduled_eta"] is not None

    # --------------------------------------------------------------------------
    # 6. Show baseline ETA
    # --------------------------------------------------------------------------
    assert init_state["baseline_eta"] is not None
    baseline_eta_init = datetime.fromisoformat(init_state["baseline_eta"])
    assert baseline_eta_init is not None
    assert train_pre["baseline_eta"] is not None

    # --------------------------------------------------------------------------
    # 7. Show ML ETA
    # --------------------------------------------------------------------------
    assert init_state["ml_eta"] is not None
    ml_eta_init = datetime.fromisoformat(init_state["ml_eta"])
    assert ml_eta_init is not None
    assert init_state.get("confidence_margin_minutes") is not None
    assert init_state["confidence_margin_minutes"] > 0
    # Also verify dashboard confidence bounds
    assert train_pre["confidence_range"] is not None
    assert "lower_bound" in train_pre["confidence_range"]
    assert "upper_bound" in train_pre["confidence_range"]

    # --------------------------------------------------------------------------
    # 8. Inject signal halt
    # --------------------------------------------------------------------------
    action_resp = client.post("/demo/action/signal_halt")
    assert action_resp.status_code == 200
    action_data = action_resp.json()
    assert action_data["status"] == "SUCCESS"
    assert action_data["action_id"] == "signal_halt"

    # --------------------------------------------------------------------------
    # 9. Verify train delay changes
    # --------------------------------------------------------------------------
    assert action_data["previous_delay_minutes"] == 2.0
    assert action_data["new_delay_minutes"] == 17.0
    assert action_data["delay_delta_minutes"] == 15.0
    assert action_data["new_status"] == "HALTED"
    assert action_data["speed_kmh"] == 0.0

    # --------------------------------------------------------------------------
    # 10. Verify features update
    # --------------------------------------------------------------------------
    eta_detail_resp = client.get("/train/12302/eta/PRYJ")
    assert eta_detail_resp.status_code == 200
    eta_detail = eta_detail_resp.json()
    features = eta_detail.get("features", {})
    assert len(features) == 11, "Must contain all 11 model features"
    assert features["current_delay_min"] == 17.0
    assert features["distance_to_go_km"] == pytest.approx(165.75, abs=0.5)
    assert features["scheduled_time_to_go_min"] > 0
    assert features["num_intermediate_halts"] == 0
    assert "feature_provenance" in eta_detail
    provenance = eta_detail["feature_provenance"]
    assert len(provenance) == 11

    # --------------------------------------------------------------------------
    # 11. Verify baseline ETA changes
    # --------------------------------------------------------------------------
    baseline_eta_post = datetime.fromisoformat(action_data["baseline_eta"])
    # Baseline ETA must be pushed back by delay increase
    assert baseline_eta_post > baseline_eta_init, (
        f"Baseline ETA ({baseline_eta_post}) must be later than initial ({baseline_eta_init})"
    )
    baseline_shift_minutes = (baseline_eta_post - baseline_eta_init).total_seconds() / 60.0
    assert baseline_shift_minutes > 10.0, f"Baseline ETA must shift significantly: {baseline_shift_minutes}m"
    assert baseline_shift_minutes == pytest.approx(12.86, abs=1.5)

    # --------------------------------------------------------------------------
    # 12. Verify ML ETA changes
    # --------------------------------------------------------------------------
    ml_eta_post = datetime.fromisoformat(action_data["ml_eta"])
    assert ml_eta_post != ml_eta_init, (
        f"ML ETA must change after event injection: initial={ml_eta_init}, post={ml_eta_post}"
    )
    ml_delta_minutes = abs((ml_eta_post - ml_eta_init).total_seconds()) / 60.0
    assert ml_delta_minutes >= 1.0, f"ML ETA change must be substantive: {ml_delta_minutes}m"

    # --------------------------------------------------------------------------
    # 13. Verify station board updates
    # --------------------------------------------------------------------------
    stn_board_resp = client.get("/station/PRYJ/arrivals")
    assert stn_board_resp.status_code == 200
    stn_board = stn_board_resp.json()
    assert stn_board["station_code"] == "PRYJ"
    arrivals = [a for a in stn_board["arrivals"] if a["train_number"] == "12302"]
    assert len(arrivals) == 1, "Train 12302 must appear on PRYJ station board"
    arr_12302 = arrivals[0]
    assert arr_12302["current_delay_minutes"] == 17.0
    arr_ml_eta = datetime.fromisoformat(arr_12302["ml_eta"])
    assert arr_ml_eta == ml_eta_post
    assert arr_12302["ml_status"] == "AVAILABLE"
    assert arr_12302["confidence_lower"] is not None
    assert arr_12302["confidence_upper"] is not None

    # --------------------------------------------------------------------------
    # 14. Verify passenger view updates
    # --------------------------------------------------------------------------
    # 14a. Train details view
    train_view_resp = client.get("/train/12302")
    assert train_view_resp.status_code == 200
    train_view = train_view_resp.json()
    assert train_view["train_number"] == "12302"
    assert train_view["current_delay_minutes"] == 17.0
    assert train_view["current_state"]["status"] == "HALTED"
    assert train_view["current_state"]["speed_kmh"] == 0.0
    assert len(train_view["active_events"]) >= 1
    assert any(e["event_type"] == "SIGNAL_HALT" for e in train_view["active_events"])

    # 14b. Single-station passenger ETA endpoint
    passenger_eta_resp = client.get("/train/12302/eta/PRYJ")
    assert passenger_eta_resp.status_code == 200
    passenger_eta = passenger_eta_resp.json()
    assert passenger_eta["target_station"] == "PRYJ"
    assert passenger_eta["current_delay_minutes"] == 17.0
    pass_ml_eta = datetime.fromisoformat(passenger_eta["ml_eta"])
    assert pass_ml_eta == ml_eta_post
    assert passenger_eta["confidence_lower"] is not None
    assert passenger_eta["confidence_upper"] is not None


# ==============================================================================
# SCENARIO B — LIVE API MODE
# ==============================================================================

def test_scenario_b_live_api_mode_e2e(client, db_session):
    """
    Complete 9-step E2E validation of Scenario B: Live Railway API Mode.
    """
    # --------------------------------------------------------------------------
    # 1. Start FastAPI
    # --------------------------------------------------------------------------
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    # Sample external RailRadar API response
    sample_external_payload = {
        "success": True,
        "data": {
            "trainNumber": "12302",
            "trainName": "Howrah Rajdhani Express",
            "startDate": "2026-09-29",
            "status": "RUNNING",
            "currentLocation": {
                "stationCode": "CNB",
                "sequence": 2,
                "delayMinutes": 24.5,
                "segmentProgress": 0.45,
                "speedKmh": 102.0,
            },
            "previousStation": {
                "stationCode": "NDLS",
            },
            "nextHalt": {
                "stationCode": "PRYJ",
                "distance": 195.0,
            },
            "lastUpdatedAt": "2026-09-29T13:30:00+00:00",
        },
    }

    # --------------------------------------------------------------------------
    # 2. Request one live train
    # --------------------------------------------------------------------------
    with patch.object(RailRadarClient, "fetch_live_train") as mock_fetch:
        # Step 3 helper normalization
        client_instance = RailRadarClient(api_key="test_key_e2e")
        normalized_state = client_instance.normalize_payload(sample_external_payload, "12302")
        mock_fetch.return_value = normalized_state

        resp = client.get("/live/train/12302")
        assert resp.status_code == 200
        live_result = resp.json()

    # --------------------------------------------------------------------------
    # 3. Normalize the external response
    # --------------------------------------------------------------------------
    assert live_result["train_number"] == "12302"
    assert live_result["train_name"] == "Howrah Rajdhani Express"
    assert live_result["status"] == "RUNNING"
    assert live_result["current_station"] == "CNB"
    assert live_result["next_station"] == "PRYJ"
    assert live_result["current_delay"] == 24.5
    assert live_result["speed"] == 102.0
    assert live_result["segment_progress"] == 0.45
    assert live_result["data_source"] == "LIVE_API"

    # --------------------------------------------------------------------------
    # 4. Generate features
    # --------------------------------------------------------------------------
    features = live_result.get("features")
    assert features is not None
    assert len(features) == 11, "Must generate exactly 11 PRD features"
    assert features["current_delay_min"] == 24.5
    assert features["distance_to_go_km"] == 195.0
    assert features["scheduled_time_to_go_min"] > 0
    assert features["num_intermediate_halts"] == 0

    provenance = live_result.get("feature_provenance")
    assert provenance is not None
    assert provenance["current_delay_min"] == "live data"
    assert provenance["distance_to_go_km"] == "live data"
    assert provenance["scheduled_time_to_go_min"] == "live data"
    assert provenance["hist_avg_delay_this_section"] == "historical data"
    assert provenance["weather_flag"] == "simulator/default source"

    # --------------------------------------------------------------------------
    # 5. Calculate baseline ETA
    # --------------------------------------------------------------------------
    assert live_result["baseline_eta"] is not None
    baseline_eta = datetime.fromisoformat(live_result["baseline_eta"])
    assert baseline_eta is not None
    assert baseline_eta > datetime.fromisoformat(live_result["timestamp"])

    # --------------------------------------------------------------------------
    # 6. Calculate ML ETA
    # --------------------------------------------------------------------------
    assert live_result["ml_eta"] is not None
    ml_eta = datetime.fromisoformat(live_result["ml_eta"])
    assert ml_eta is not None

    confidence_range = live_result.get("confidence_range")
    assert confidence_range is not None
    lower_bound = datetime.fromisoformat(confidence_range["lower_bound"])
    upper_bound = datetime.fromisoformat(confidence_range["upper_bound"])
    assert lower_bound <= ml_eta <= upper_bound
    assert confidence_range["margin_minutes"] > 0

    # --------------------------------------------------------------------------
    # 7. Display result in dashboard
    # --------------------------------------------------------------------------
    # Query /train/{train_id}?data_source=LIVE_API (used by Dashboard)
    with patch.object(RailRadarClient, "fetch_live_train", return_value=normalized_state):
        dash_resp = client.get("/train/12302?data_source=LIVE_API")
        assert dash_resp.status_code == 200
        dash_data = dash_resp.json()
        assert dash_data["train_number"] == "12302"
        assert dash_data["data_source_mode"] == "LIVE_API"
        assert dash_data["is_fallback"] is False
        assert dash_data["current_delay_minutes"] == 24.5
        assert dash_data["baseline_eta"] is not None
        assert dash_data["ml_eta"] is not None
        assert len(dash_data["upcoming_stations"]) >= 1

    # --------------------------------------------------------------------------
    # 8. Verify last-updated timestamp
    # --------------------------------------------------------------------------
    assert live_result["last_updated"] is not None
    last_updated = datetime.fromisoformat(live_result["last_updated"])
    assert last_updated is not None
    assert dash_data["last_updated"] is not None

    # --------------------------------------------------------------------------
    # 9. Test API failure fallback
    # --------------------------------------------------------------------------
    # When external API throws 503 Service Unavailable, provider falls back to SIMULATOR
    with patch.object(RailRadarClient, "fetch_live_train", side_effect=ServiceUnavailableError("RailRadar 503")):
        fallback_resp = client.get("/train/12302?data_source=LIVE_API")
        assert fallback_resp.status_code == 200, "Dashboard request must succeed with fallback"
        fb_data = fallback_resp.json()
        assert fb_data["train_number"] == "12302"
        assert fb_data["data_source_mode"] == "SIMULATOR", "Must switch effective mode to SIMULATOR"
        assert fb_data["is_fallback"] is True, "is_fallback flag must be True"
        assert "fallback_reason" in fb_data and fb_data["fallback_reason"] is not None
        assert fb_data["baseline_eta"] is not None, "Baseline ETA must remain available during fallback"
        assert fb_data["ml_eta"] is not None, "ML ETA must remain available during fallback"
