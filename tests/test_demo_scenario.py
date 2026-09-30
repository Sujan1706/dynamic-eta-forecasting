"""
Tests for Deterministic Hackathon Demo Mode.
Validates:
1. POST /demo/reset (resets scenario to seed 42, NDLS-HWH corridor, train 12302 at CNB)
2. GET /demo/scenario (returns current scenario state and available actions)
3. Reproducibility across resets with identical seed
4. Deterministic action execution:
   - signal_halt: +15m, speed = 0, status = HALTED
   - congestion: +10m, speed = 49.5, status = RUNNING
   - speed_restriction: +8m, speed = 30.0, status = RUNNING
5. Downstream propagation across FastAPI endpoints:
   - GET /train/12302 (Train details & current state)
   - GET /station/PRYJ/arrivals (Station board for next station)
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    """FastAPI TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


def test_demo_reset(client):
    """Verify POST /demo/reset deterministically initializes train 12302 at CNB."""
    resp = client.post("/demo/reset")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] in ("SUCCESS", "PREPARED_INITIAL_STATE")
    assert data["seed"] == 42
    assert data["train_number"] == "12302"
    assert data["current_station"] == "CNB"
    assert data["next_station"] == "PRYJ"
    assert data["current_delay_minutes"] == 2.0
    assert data["current_speed_kmh"] == 110.0
    assert data["train_status"] == "RUNNING"
    assert data["baseline_eta"] is not None
    assert data["ml_eta"] is not None
    assert len(data["available_actions"]) == 3


def test_demo_scenario_get(client):
    """Verify GET /demo/scenario retrieves the active demo state."""
    # Ensure initialized
    client.post("/demo/reset")

    resp = client.get("/demo/scenario")
    assert resp.status_code == 200
    data = resp.json()

    assert data["train_number"] == "12302"
    assert data["seed"] == 42
    assert data["current_station"] == "CNB"
    assert data["next_station"] == "PRYJ"
    assert data["distance_to_next_km"] > 0
    assert len(data["available_actions"]) == 3


def test_demo_reproducibility(client):
    """Verify that multiple resets produce identical deterministic state and predictions."""
    resp1 = client.post("/demo/reset")
    assert resp1.status_code == 200
    d1 = resp1.json()

    # Modify state by triggering an action
    client.post("/demo/action/signal_halt")

    # Reset again
    resp2 = client.post("/demo/reset")
    assert resp2.status_code == 200
    d2 = resp2.json()

    assert d1["current_delay_minutes"] == d2["current_delay_minutes"] == 2.0
    assert d1["current_speed_kmh"] == d2["current_speed_kmh"] == 110.0
    assert d1["train_status"] == d2["train_status"] == "RUNNING"
    assert d1["baseline_eta"] == d2["baseline_eta"]
    assert d1["ml_eta"] == d2["ml_eta"]


def test_demo_action_signal_halt(client):
    """Verify signal_halt sets delay to 17m, status to HALTED, and speed to 0."""
    client.post("/demo/reset")

    resp = client.post("/demo/action/signal_halt")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert data["action_id"] == "signal_halt"
    assert data["previous_delay_minutes"] == 2.0
    assert data["new_delay_minutes"] == 17.0
    assert data["delay_delta_minutes"] == 15.0
    assert data["new_status"] == "HALTED"
    assert data["speed_kmh"] == 0.0
    assert data["baseline_eta"] is not None
    assert data["ml_eta"] is not None


def test_demo_action_congestion(client):
    """Verify congestion sets speed to 49.5 and increments delay by 10m."""
    client.post("/demo/reset")

    resp = client.post("/demo/action/congestion")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert data["action_id"] == "congestion"
    assert data["previous_delay_minutes"] == 2.0
    assert data["new_delay_minutes"] == 12.0
    assert data["delay_delta_minutes"] == 10.0
    assert data["new_status"] == "RUNNING"
    assert data["speed_kmh"] == 49.5


def test_demo_action_speed_restriction(client):
    """Verify speed_restriction sets speed to 30.0 and increments delay by 8m."""
    client.post("/demo/reset")

    resp = client.post("/demo/action/speed_restriction")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert data["action_id"] == "speed_restriction"
    assert data["previous_delay_minutes"] == 2.0
    assert data["new_delay_minutes"] == 10.0
    assert data["delay_delta_minutes"] == 8.0
    assert data["new_status"] == "RUNNING"
    assert data["speed_kmh"] == 30.0


def test_demo_invalid_action(client):
    """Verify unknown action returns 400 Bad Request."""
    client.post("/demo/reset")
    resp = client.post("/demo/action/unknown_action_xyz")
    assert resp.status_code == 400
    assert "Invalid demo action" in resp.json()["detail"]


def test_demo_action_sequence_cumulative(client):
    """Verify sequence of events executes cleanly and accumulates delays properly."""
    client.post("/demo/reset")

    # Step 1: Signal Halt (+15m) -> 17.0m
    r1 = client.post("/demo/action/signal_halt")
    assert r1.status_code == 200
    assert r1.json()["new_delay_minutes"] == 17.0
    assert r1.json()["new_status"] == "HALTED"
    assert r1.json()["speed_kmh"] == 0.0

    # Step 2: Line Congestion (+10m) -> 27.0m, transitions to RUNNING
    r2 = client.post("/demo/action/congestion")
    assert r2.status_code == 200
    assert r2.json()["new_delay_minutes"] == 27.0
    assert r2.json()["new_status"] == "RUNNING"
    assert r2.json()["speed_kmh"] == 49.5

    # Step 3: Speed Restriction (+8m) -> 35.0m
    r3 = client.post("/demo/action/speed_restriction")
    assert r3.status_code == 200
    assert r3.json()["new_delay_minutes"] == 35.0
    assert r3.json()["new_status"] == "RUNNING"
    assert r3.json()["speed_kmh"] == 30.0


def test_demo_end_to_end_propagation_across_views(client):
    """
    Comprehensive end-to-end integration test:
    Simulator → state update → feature update → baseline recalculation
    → ML prediction → FastAPI → dashboard & station board.
    """
    # 1. Reset scenario
    reset_resp = client.post("/demo/reset")
    assert reset_resp.status_code == 200

    # 2. Check Train Details endpoint (consumed by Dashboard & Passenger View)
    t_resp = client.get("/train/12302")
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert t_data["current_station"] == "CNB"
    assert t_data["current_delay_minutes"] == 2.0
    assert t_data["current_state"]["status"] == "RUNNING"
    initial_ml_eta = t_data["ml_eta"]
    assert initial_ml_eta is not None

    # 3. Check Station Board for next station PRYJ
    stn_resp = client.get("/station/PRYJ/arrivals")
    assert stn_resp.status_code == 200
    stn_data = stn_resp.json()
    arrivals_12302 = [a for a in stn_data["arrivals"] if a["train_number"] == "12302"]
    assert len(arrivals_12302) == 1
    assert arrivals_12302[0]["current_delay_minutes"] == 2.0

    # 4. Inject Signal Halt via Demo Action
    action_resp = client.post("/demo/action/signal_halt")
    assert action_resp.status_code == 200
    halt_data = action_resp.json()
    assert halt_data["new_delay_minutes"] == 17.0
    assert halt_data["new_status"] == "HALTED"
    assert halt_data["speed_kmh"] == 0.0

    # 5. Verify Train Details endpoint reflects new delay and status immediately
    t_resp2 = client.get("/train/12302")
    assert t_resp2.status_code == 200
    t_data2 = t_resp2.json()
    assert t_data2["current_delay_minutes"] == 17.0
    assert t_data2["current_state"]["status"] == "HALTED"
    assert t_data2["current_state"]["speed_kmh"] == 0.0
    # Delay increased by 15m, baseline ETA is pushed back
    assert t_data2["baseline_eta"] > t_data["baseline_eta"]
    assert t_data2["ml_eta"] is not None

    # 6. Verify Station Board for PRYJ immediately reflects 17m delay and updated arrival
    stn_resp2 = client.get("/station/PRYJ/arrivals")
    assert stn_resp2.status_code == 200
    stn_data2 = stn_resp2.json()
    arrivals_halt = [a for a in stn_data2["arrivals"] if a["train_number"] == "12302"]
    assert len(arrivals_halt) == 1
    assert arrivals_halt[0]["current_delay_minutes"] == 17.0
    assert arrivals_halt[0]["baseline_eta"] > arrivals_12302[0]["baseline_eta"]
    assert arrivals_halt[0]["ml_eta"] is not None

    # 7. Inject Line Congestion
    action_resp2 = client.post("/demo/action/congestion")
    assert action_resp2.status_code == 200
    cong_data = action_resp2.json()
    assert cong_data["new_delay_minutes"] == 27.0
    assert cong_data["new_status"] == "RUNNING"
    assert cong_data["speed_kmh"] == 49.5

    # 8. Verify Train Details and PRYJ Station Board again
    t_resp3 = client.get("/train/12302")
    assert t_resp3.json()["current_delay_minutes"] == 27.0
    assert t_resp3.json()["current_state"]["status"] == "RUNNING"

    stn_resp3 = client.get("/station/PRYJ/arrivals")
    arrivals_cong = [a for a in stn_resp3.json()["arrivals"] if a["train_number"] == "12302"]
    assert arrivals_cong[0]["current_delay_minutes"] == 27.0
    assert arrivals_cong[0]["baseline_eta"] > arrivals_halt[0]["baseline_eta"]
