import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

from backend.database.connection import Base, get_db
from backend.database.seed import seed_data
from backend.database.models import Event, Journey, Train
from pathlib import Path
from backend.main import (
    app,
    get_active_simulator,
    get_multi_station_service,
    get_metrics_filepath,
    get_metadata_filepath,
)
from backend.simulator.engine import TrainSimulator
from backend.simulator.events import EventType


@pytest.fixture
def test_setup():
    """Sets up an in-memory SQLite database and isolated simulator for API tests."""
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

    # Override dependencies
    def override_get_db():
        try:
            yield db
        finally:
            pass

    def override_get_simulator():
        return sim

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_active_simulator] = override_get_simulator

    client = TestClient(app)
    yield client, db, sim

    app.dependency_overrides.clear()
    db.close()


def test_endpoint_health(test_setup):
    """Verify GET /health returns 200 and expected status."""
    client, _, _ = test_setup
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "dynamic-eta-forecasting"
    assert data["version"] == "0.1.0"


def test_endpoint_get_trains(test_setup):
    """Verify GET /trains returns all configured trains with their running states."""
    client, _, _ = test_setup
    response = client.get("/trains")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 8
    trains = data["trains"]
    train_numbers = [t["train_number"] for t in trains]
    assert "12302" in train_numbers
    assert "12952" in train_numbers

    # Verify running state attached
    rajdhani = next(t for t in trains if t["train_number"] == "12302")
    assert rajdhani["current_state"] is not None
    assert rajdhani["current_state"]["train_number"] == "12302"
    assert rajdhani["current_state"]["source"] == "simulator"


def test_endpoint_get_train_details(test_setup):
    """Verify GET /train/{train_id} by train number and integer ID."""
    client, _, _ = test_setup

    # By train number
    resp_num = client.get("/train/12302")
    assert resp_num.status_code == 200
    data = resp_num.json()
    assert data["train_number"] == "12302"
    assert data["name"] == "Howrah Rajdhani Express"
    assert len(data["route_stations"]) == 8
    assert data["route_stations"][0]["station_code"] == "NDLS"
    assert data["route_stations"][-1]["station_code"] == "HWH"

    # By integer ID
    train_id = data["id"]
    resp_id = client.get(f"/train/{train_id}")
    assert resp_id.status_code == 200
    assert resp_id.json()["train_number"] == "12302"

    # 404 for nonexistent train
    resp_404 = client.get("/train/99999")
    assert resp_404.status_code == 404
    assert "not found" in resp_404.json()["detail"].lower()


def test_endpoint_get_train_eta_internal_consistency(test_setup):
    """Verify GET /train/{id}/eta/{station} returns consistent ETA predictions and features."""
    client, _, sim = test_setup

    # Inject a 12-minute delay on Train 12302
    sim.inject_event("12302", EventType.SIGNAL_HALT, delay_minutes=12.0)

    response = client.get("/train/12302/eta/PRYJ")
    assert response.status_code == 200
    data = response.json()

    assert data["train_number"] == "12302"
    assert data["target_station"] == "PRYJ"
    assert data["current_station"] == "CNB"
    # Seed journey has 14.0 min delay + 12.0 min injected = 26.0 min
    assert data["current_delay_minutes"] == 26.0

    # 1. Prediction verification
    pred = data["prediction"]
    assert pred["target_station"] == "PRYJ"
    assert pred["method"] == "BASELINE"
    assert pred["current_delay"] == 26.0
    # CNB (440km) to PRYJ (635km) = 195km
    assert pred["distance_to_go_km"] == 195.0
    # Baseline ETA = ScheduledArrival + CurrentDelay - Recovery
    assert pred["baseline_eta"] is not None
    assert pred["scheduled_eta"] is not None

    # 2. Features verification
    feat = data["features"]
    assert feat["distance_to_go_km"] == 195.0
    # CNB arr 21:35 to PRYJ arr 23:43 = 128.0 min
    assert feat["scheduled_time_to_go_min"] == 128.0
    assert feat["num_intermediate_halts"] == 0
    assert feat["current_delay_min"] == 26.0
    assert feat["active_speed_restriction_flag"] == 0
    assert feat["hist_avg_delay_this_section"] == 8.5

    # 3. Off-route station raises 400
    resp_bad = client.get("/train/12302/eta/MAS")  # Chennai is not on Delhi-Howrah line
    assert resp_bad.status_code == 400
    assert "not on the route" in resp_bad.json()["detail"].lower()


def test_endpoint_get_train_details_ten_fields_contract(test_setup):
    """
    Verify GET /train/{train_id} returns all 10 required PRD fields:
    1. Current train state
    2. Current station
    3. Current delay
    4. Current timestamp
    5. Upcoming stations
    6. Scheduled ETA
    7. Baseline ETA
    8. ML ETA
    9. Confidence range
    10. Segment predictions
    """
    client, _, _ = test_setup
    resp = client.get("/train/12302")
    assert resp.status_code == 200
    data = resp.json()

    # 1. Current train state
    assert "current_state" in data
    assert data["current_state"] is not None
    assert data["current_state"]["train_number"] == "12302"
    assert data["current_state"]["status"] == "RUNNING"

    # 2. Current station
    assert "current_station" in data
    assert data["current_station"] == data["current_state"]["current_station_code"]

    # 3. Current delay
    assert "current_delay" in data
    assert isinstance(data["current_delay"], (int, float))
    assert data["current_delay"] == data["current_state"]["current_delay_minutes"]

    # 4. Current timestamp
    assert "current_timestamp" in data
    assert data["current_timestamp"] is not None

    # 5. Upcoming stations
    assert "upcoming_stations" in data
    assert isinstance(data["upcoming_stations"], list)
    assert len(data["upcoming_stations"]) > 0

    # 6. Scheduled ETA (next station)
    assert "scheduled_eta" in data
    assert data["scheduled_eta"] is not None

    # 7. Baseline ETA (next station)
    assert "baseline_eta" in data
    assert data["baseline_eta"] is not None

    # 8. ML ETA (next station)
    assert "ml_eta" in data
    assert data["ml_eta"] is not None

    # 9. Confidence range (next station)
    assert "confidence_range" in data
    conf = data["confidence_range"]
    assert conf is not None
    assert "lower_bound" in conf
    assert "upper_bound" in conf
    assert "margin_minutes" in conf
    assert conf["margin_minutes"] > 0.0

    # 10. Segment predictions
    assert "segment_predictions" in data
    assert isinstance(data["segment_predictions"], list)
    assert len(data["segment_predictions"]) > 0
    seg = data["segment_predictions"][0]
    assert "from_station_code" in seg
    assert "to_station_code" in seg
    assert "predicted_transit_minutes" in seg

    # Also verify upcoming stations have their own per-station fields
    first_up = data["upcoming_stations"][0]
    assert first_up["station_code"] == "PRYJ"  # Seed journey is at CNB, next is PRYJ
    assert first_up["scheduled_eta"] == data["scheduled_eta"]
    assert first_up["baseline_eta"] == data["baseline_eta"]
    assert first_up["ml_eta"] == data["ml_eta"]
    assert first_up["confidence_range"]["margin_minutes"] == conf["margin_minutes"]
    assert len(first_up["segment_predictions"]) >= 1

    # Verify uncertainty margin grows with distance/segments ahead
    last_up = data["upcoming_stations"][-1]
    assert last_up["segments_ahead"] > first_up["segments_ahead"]
    assert last_up["confidence_range"]["margin_minutes"] > first_up["confidence_range"]["margin_minutes"]


def test_endpoint_train_details_segment_predictions_contract(test_setup):
    """
    Regression test for segment predictions data contract:
    Asserts that GET /train/{train_id} returns all required canonical and alias
    fields on every segment prediction, including distance_km and segment_distance_km,
    for both Rajdhani (12302) and Shatabdi (12028).
    """
    client, _, _ = test_setup
    for train_num in ["12302", "12028"]:
        resp = client.get(f"/train/{train_num}")
        assert resp.status_code == 200
        data = resp.json()
        assert "segment_predictions" in data
        assert len(data["segment_predictions"]) > 0

        for seg in data["segment_predictions"]:
            # Canonical & alias keys must be present and non-null
            assert "segment_distance_km" in seg and seg["segment_distance_km"] is not None
            assert "distance_km" in seg and seg["distance_km"] is not None
            assert seg["segment_distance_km"] == seg["distance_km"]
            assert seg["distance_km"] > 0

            assert "scheduled_transit_minutes" in seg and seg["scheduled_transit_minutes"] is not None
            assert "scheduled_minutes" in seg and seg["scheduled_minutes"] is not None
            assert seg["scheduled_transit_minutes"] == seg["scheduled_minutes"]
            assert seg["scheduled_minutes"] > 0

            assert "predicted_transit_minutes" in seg and seg["predicted_transit_minutes"] is not None
            assert "predicted_minutes" in seg and seg["predicted_minutes"] is not None
            assert seg["predicted_transit_minutes"] == seg["predicted_minutes"]

            assert "baseline_transit_minutes" in seg and seg["baseline_transit_minutes"] is not None
            assert "baseline_minutes" in seg and seg["baseline_minutes"] is not None
            assert seg["baseline_transit_minutes"] == seg["baseline_minutes"]

            assert "segment_index" in seg and seg["segment_index"] is not None
            assert "segment_order" in seg and seg["segment_order"] is not None
            assert seg["segment_index"] == seg["segment_order"]

            assert "from_station_code" in seg and seg["from_station_code"]
            assert "to_station_code" in seg and seg["to_station_code"]




@pytest.mark.parametrize("invalid_id", [
    "99999",            # Non-existent numeric ID
    "0",                # Zero ID
    "-1",               # Negative ID
    "-999",             # Negative multi-digit ID
    "NONEXISTENT",      # Unknown string code
    "TRAIN_XYZ",        # Unknown train identifier
    "@@@$$$",           # Special characters
    "   ",              # Whitespace string
])
def test_endpoint_get_train_details_invalid_ids(test_setup, invalid_id):
    """Verify GET /train/{train_id} returns 404 for diverse invalid train IDs."""
    client, _, _ = test_setup
    resp = client.get(f"/train/{invalid_id}")
    assert resp.status_code == 404
    detail = resp.json()["detail"].lower()
    assert "not found" in detail


def test_endpoint_get_train_details_with_injected_disruption(test_setup):
    """Verify that operational disruptions affect current delay, baseline ETA, and ML ETA."""
    client, _, sim = test_setup

    # Inject 25-minute delay on Train 12952 (Mumbai Rajdhani, currently at KOTA)
    sim.inject_event("12952", EventType.SIGNAL_HALT, delay_minutes=25.0)

    resp = client.get("/train/12952")
    assert resp.status_code == 200
    data = resp.json()

    # Seed delay was 6.0 min + 25.0 min injected = 31.0 min
    assert data["current_delay"] == 31.0
    assert data["current_state"]["current_delay_minutes"] == 31.0
    assert data["current_station"] == "KOTA"

    # Next stop is RTM
    assert data["upcoming_stations"][0]["station_code"] == "RTM"
    assert data["baseline_eta"] is not None
    assert data["ml_eta"] is not None
    assert data["confidence_range"]["margin_minutes"] == 5.0


def test_endpoint_get_train_details_completed_journey(test_setup):
    """Verify that when a journey is completed (at final station), upcoming_stations is empty."""
    client, _, sim = test_setup

    # Move train journey to completed state
    journey = sim.get_journey("12302")
    journey.status = "COMPLETED"
    journey.current_station_index = len(journey.stops) - 1  # At HWH

    resp = client.get("/train/12302")
    assert resp.status_code == 200
    data = resp.json()

    assert data["current_state"]["status"] == "COMPLETED"
    assert data["current_station"] == "HWH"
    assert data["upcoming_stations"] == []
    assert data["scheduled_eta"] is None
    assert data["baseline_eta"] is None
    assert data["ml_eta"] is None
    assert data["confidence_range"] is None
    assert data["segment_predictions"] == []


def test_endpoint_get_train_station_eta_required_fields(test_setup):
    """
    Verify GET /train/{train_id}/eta/{station_code} returns all required fields:
    - train_id
    - train_number
    - target_station
    - current_station
    - scheduled_eta
    - baseline_eta
    - ml_eta
    - confidence_lower
    - confidence_upper
    - current_delay_minutes
    - prediction_timestamp
    - model_version
    - ml_status
    """
    client, _, _ = test_setup
    resp = client.get("/train/12302/eta/PRYJ")
    assert resp.status_code == 200
    data = resp.json()

    assert "train_id" in data and isinstance(data["train_id"], int)
    assert data["train_number"] == "12302"
    assert data["target_station"] == "PRYJ"
    assert data["current_station"] == "CNB"
    assert data["scheduled_eta"] is not None
    assert data["baseline_eta"] is not None
    assert data["ml_eta"] is not None
    assert data["confidence_lower"] is not None
    assert data["confidence_upper"] is not None
    assert isinstance(data["current_delay_minutes"], (int, float))
    assert data["prediction_timestamp"] is not None
    assert data["model_version"] == "1.0.0"
    assert data["ml_status"] == "AVAILABLE"
    assert data["ml_error"] is None

    # Confidence interval sanity
    assert data["confidence_lower"] <= data["confidence_upper"]


def test_endpoint_get_train_station_eta_by_integer_id(test_setup):
    """Verify lookup by integer train ID."""
    client, db, _ = test_setup
    resp = client.get("/train/1/eta/PRYJ")
    assert resp.status_code == 200
    data = resp.json()
    assert data["train_id"] == 1
    assert data["train_number"] == "12302"
    assert data["target_station"] == "PRYJ"


def test_endpoint_get_train_station_eta_multi_station_ahead(test_setup):
    """Verify multi-station ahead ETA computation (e.g. GAYA is 3 stops ahead from CNB)."""
    client, _, _ = test_setup
    resp = client.get("/train/12302/eta/GAYA")
    assert resp.status_code == 200
    data = resp.json()

    assert data["target_station"] == "GAYA"
    assert data["ml_eta"] is not None
    assert data["baseline_eta"] is not None
    assert data["segments_ahead"] == 3
    assert data["confidence_lower"] <= data["confidence_upper"]


def test_endpoint_get_train_station_eta_fallback_when_ml_fails(test_setup):
    """
    Verify that if ML prediction fails:
    - returns baseline ETA
    - clearly indicates ML status as unavailable
    - does not crash the API (HTTP 200)
    """
    client, _, _ = test_setup

    class FailingMLService:
        def predict_station_eta(self, *args, **kwargs):
            raise RuntimeError("Simulated XGBoost engine inference failure")

    app.dependency_overrides[get_multi_station_service] = lambda: FailingMLService()
    try:
        resp = client.get("/train/12302/eta/PRYJ")
        assert resp.status_code == 200
        data = resp.json()

        assert data["train_number"] == "12302"
        assert data["target_station"] == "PRYJ"
        assert data["ml_status"] == "UNAVAILABLE"
        assert "Simulated XGBoost engine inference failure" in data["ml_error"]
        assert data["ml_eta"] is None
        assert data["confidence_lower"] is None
        assert data["confidence_upper"] is None

        # Baseline ETA is safely returned
        assert data["baseline_eta"] is not None
        assert data["scheduled_eta"] is not None
    finally:
        app.dependency_overrides.pop(get_multi_station_service, None)


def test_endpoint_get_train_station_eta_invalid_train(test_setup):
    """Verify 404 for invalid train IDs."""
    client, _, _ = test_setup
    resp = client.get("/train/99999/eta/PRYJ")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_endpoint_get_train_station_eta_invalid_station(test_setup):
    """Verify 400 for off-route station."""
    client, _, _ = test_setup
    resp = client.get("/train/12302/eta/MAS")
    assert resp.status_code == 400
    assert "not on the route" in resp.json()["detail"].lower()


def test_endpoint_get_station_arrivals_success(test_setup):
    """Verify GET /station/{station_code}/arrivals returns valid arrivals list."""
    client, _, _ = test_setup
    resp = client.get("/station/PRYJ/arrivals")
    assert resp.status_code == 200
    data = resp.json()

    assert data["station_code"] == "PRYJ"
    assert data["station_name"] == "Prayagraj Junction"
    assert data["current_timestamp"] is not None
    assert data["window_hours"] == 24.0
    assert data["total_arrivals"] >= 1
    assert isinstance(data["arrivals"], list)

    first = data["arrivals"][0]
    assert "train_id" in first and isinstance(first["train_id"], int)
    assert "train_number" in first
    assert "train_name" in first
    assert "train_type" in first
    assert "origin_station_code" in first
    assert "destination_station_code" in first
    assert "current_station" in first
    assert isinstance(first["current_delay_minutes"], (int, float))
    assert first["distance_to_go_km"] > 0
    assert first["segments_ahead"] >= 1

    # ETAs & confidence
    assert first["scheduled_eta"] is not None
    assert first["baseline_eta"] is not None
    assert first["ml_eta"] is not None
    assert first["confidence_lower"] is not None
    assert first["confidence_upper"] is not None
    assert first["confidence_range"] is not None
    assert first["confidence_range"]["margin_minutes"] > 0
    assert first["ml_status"] == "AVAILABLE"
    assert isinstance(first["minutes_to_arrival"], (int, float))


def test_endpoint_get_station_arrivals_sorted_by_ml_eta(test_setup):
    """
    Verify approaching trains are sorted primarily by predicted ML ETA.
    Train 12302 is at CNB (1 hop, 195km away).
    Train 12306 is at NDLS (2 hops, 635km away).
    Train 12302 must be predicted first.
    """
    client, _, _ = test_setup
    resp = client.get("/station/PRYJ/arrivals")
    assert resp.status_code == 200
    data = resp.json()
    arrivals = data["arrivals"]

    assert len(arrivals) >= 2
    # Verify sorting primarily by ML ETA
    for i in range(len(arrivals) - 1):
        assert arrivals[i]["ml_eta"] <= arrivals[i + 1]["ml_eta"]

    # 12302 (at CNB) should precede 12306 (at NDLS)
    assert arrivals[0]["train_number"] == "12302"
    assert arrivals[0]["segments_ahead"] < arrivals[1]["segments_ahead"]


def test_endpoint_get_station_arrivals_window_filter(test_setup):
    """Verify window_hours filters upcoming arrivals correctly."""
    client, _, _ = test_setup

    # Very small window (0.1 hours = 6 min): no train arriving in 6 min
    resp_tight = client.get("/station/PRYJ/arrivals?window_hours=0.1")
    assert resp_tight.status_code == 200
    data_tight = resp_tight.json()
    assert data_tight["total_arrivals"] == 0
    assert data_tight["arrivals"] == []

    # Large window (24 hours): includes all approaching trains
    resp_wide = client.get("/station/PRYJ/arrivals?window_hours=24.0")
    assert resp_wide.status_code == 200
    data_wide = resp_wide.json()
    assert data_wide["total_arrivals"] >= 1


def test_endpoint_get_station_arrivals_no_approaching_trains(test_setup):
    """Verify origin station with no approaching trains returns empty list."""
    client, _, _ = test_setup
    # NDLS is the origin station on Route 1 & 2; no trains approach NDLS from further north
    resp = client.get("/station/NDLS/arrivals")
    assert resp.status_code == 200
    data = resp.json()
    assert data["station_code"] == "NDLS"
    assert data["total_arrivals"] == 0
    assert data["arrivals"] == []


def test_endpoint_get_station_arrivals_unknown_station(test_setup):
    """Verify 404 for unknown or nonexistent station code."""
    client, _, _ = test_setup
    resp = client.get("/station/NONEXISTENT/arrivals")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()

    resp2 = client.get("/station/XYZ999/arrivals")
    assert resp2.status_code == 404
    assert "not found" in resp2.json()["detail"].lower()


def test_endpoint_get_station_arrivals_case_insensitive(test_setup):
    """Verify station code is handled case-insensitively."""
    client, _, _ = test_setup
    resp = client.get("/station/pryj/arrivals")
    assert resp.status_code == 200
    assert resp.json()["station_code"] == "PRYJ"


def test_endpoint_get_station_arrivals_ml_fallback(test_setup):
    """Verify that when ML prediction fails, station arrivals fall back to baseline without crashing."""
    client, _, _ = test_setup

    class FailingMLService:
        def predict_station_eta(self, *args, **kwargs):
            raise RuntimeError("Simulated XGBoost station arrivals inference error")

    app.dependency_overrides[get_multi_station_service] = lambda: FailingMLService()
    try:
        resp = client.get("/station/PRYJ/arrivals")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_arrivals"] >= 1
        for item in data["arrivals"]:
            assert item["ml_status"] == "UNAVAILABLE"
            assert item["ml_eta"] is None
            assert item["baseline_eta"] is not None
            assert item["scheduled_eta"] is not None
    finally:
        app.dependency_overrides.pop(get_multi_station_service, None)


def test_endpoint_simulate_event_success(test_setup):
    """Verify POST /simulate/event applies event, updates delay, persists to DB, and returns updated ETAs."""
    client, db, sim = test_setup
    payload = {
        "train_id": "12302",
        "event_type": "SIGNAL_HALT",
        "delay_minutes": 15.0,
        "severity": "HIGH",
        "metadata": {"location": "Signal Post 42", "reason": "Aspect red"},
    }
    response = client.post("/simulate/event", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Event details
    assert data["event"]["event_type"] == "SIGNAL_HALT"
    assert data["event"]["delay_minutes"] == 15.0
    assert data["event"]["severity"] == "HIGH"
    assert data["event"]["metadata"]["location"] == "Signal Post 42"

    # Train state
    state = data["train_state"]
    assert state["train_number"] == "12302"
    assert state["current_delay_minutes"] >= 15.0
    assert state["status"] == "HALTED"
    assert state["speed_kmh"] == 0.0

    # ETAs
    assert data["baseline_eta"] is not None
    assert data["updated_baseline_eta"] is not None
    assert data["ml_eta"] is not None
    assert data["updated_ml_eta"] is not None
    assert data["ml_status"] == "AVAILABLE"
    assert data["confidence_range"] is not None

    # SQLite persistence verification
    train_record = db.query(Train).filter(Train.train_number == "12302").first()
    assert train_record is not None
    journey = db.query(Journey).filter(Journey.train_id == train_record.id).first()
    assert journey is not None
    db.refresh(journey)
    assert journey.current_delay_minutes >= 15.0
    assert journey.status == "HALTED"

    ev = db.query(Event).filter(Event.journey_id == journey.id).order_by(Event.id.desc()).first()
    assert ev is not None
    assert ev.event_type == "SIGNAL_HALT"
    assert ev.delay_minutes == 15.0
    assert ev.severity == "HIGH"


def test_endpoint_simulate_event_all_supported_types(test_setup):
    """Verify all 5 supported event types succeed with case insensitivity."""
    client, db, sim = test_setup
    supported_types = [
        "SIGNAL_HALT",
        "CONGESTION",
        "SPEED_RESTRICTION",
        "UNSCHEDULED_HALT",
        "WEATHER",
    ]
    for ev_type in supported_types:
        resp = client.post(
            "/simulate/event",
            json={
                "train_id": "12302",
                "event_type": ev_type.lower(),
                "delay_minutes": 5.0,
            },
        )
        assert resp.status_code == 200, f"Failed for {ev_type}: {resp.text}"
        data = resp.json()
        assert data["event"]["event_type"] == ev_type


def test_endpoint_simulate_event_invalid_event_type(test_setup):
    """Verify 422 Unprocessable Entity for invalid or unsupported event types."""
    client, _, _ = test_setup
    payload = {
        "train_id": "12302",
        "event_type": "ALIEN_ATTACK",
        "delay_minutes": 10.0,
    }
    resp = client.post("/simulate/event", json=payload)
    assert resp.status_code == 422


def test_endpoint_simulate_event_invalid_train_id(test_setup):
    """Verify 404 for unknown train ID."""
    client, _, _ = test_setup
    payload = {
        "train_id": "99999",
        "event_type": "SIGNAL_HALT",
        "delay_minutes": 10.0,
    }
    resp = client.post("/simulate/event", json=payload)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_endpoint_simulate_event_negative_delay(test_setup):
    """Verify 422 for negative delay_minutes."""
    client, _, _ = test_setup
    payload = {
        "train_id": "12302",
        "event_type": "SIGNAL_HALT",
        "delay_minutes": -5.0,
    }
    resp = client.post("/simulate/event", json=payload)
    assert resp.status_code == 422


def test_endpoint_simulate_event_ml_fallback(test_setup):
    """Verify that when ML prediction fails, /simulate/event gracefully returns baseline without crashing."""
    client, _, _ = test_setup

    class FailingMLService:
        def predict_station_eta(self, *args, **kwargs):
            raise RuntimeError("Injected ML failure")

    app.dependency_overrides[get_multi_station_service] = lambda: FailingMLService()
    try:
        resp = client.post(
            "/simulate/event",
            json={
                "train_id": "12302",
                "event_type": "CONGESTION",
                "delay_minutes": 8.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ml_status"] == "UNAVAILABLE"
        assert data["baseline_eta"] is not None
        assert data["ml_eta"] == data["baseline_eta"]
    finally:
        app.dependency_overrides.pop(get_multi_station_service, None)


def test_endpoint_get_model_metrics_success(test_setup):
    """Verify GET /model/metrics loads real metrics and returns all expected fields."""
    client, _, _ = test_setup
    resp = client.get("/model/metrics")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "AVAILABLE"
    assert data["is_available"] is True
    assert data["model_name"] == "Dynamic Train ETA XGBoost Regressor"
    assert data["model_version"] == "1.0.0"
    assert data["training_timestamp"] is not None
    assert data["total_training_samples"] == 4915
    assert data["test_samples"] == 2430

    # Overall MAE and RMSE
    assert data["baseline_mae"] == pytest.approx(74.106, 0.01)
    assert data["ml_mae"] == pytest.approx(28.275, 0.01)
    assert data["baseline_rmse"] == pytest.approx(102.008, 0.01)
    assert data["ml_rmse"] == pytest.approx(48.091, 0.01)

    # Metrics by horizon
    assert "metrics_by_horizon" in data
    horizons = data["metrics_by_horizon"]
    assert "1_station_ahead" in horizons
    assert "3_stations_ahead" in horizons
    assert "5_stations_ahead" in horizons
    assert horizons["1_station_ahead"]["sample_count"] == 1210
    assert horizons["1_station_ahead"]["ml_mae"] == pytest.approx(5.167, 0.01)
    assert horizons["1_station_ahead"]["winner"] == "ML"
    assert data["by_horizon"] == horizons

    # Metrics by disruption status
    assert "metrics_by_disruption_status" in data
    disruptions = data["metrics_by_disruption_status"]
    assert "no_disruption" in disruptions
    assert "with_disruption" in disruptions
    assert disruptions["no_disruption"]["sample_count"] == 1585
    assert disruptions["with_disruption"]["sample_count"] == 845
    assert disruptions["no_disruption"]["winner"] == "ML"
    assert disruptions["with_disruption"]["winner"] == "ML"
    assert data["by_disruption"] == disruptions
    assert data["metrics_by_disruption"] == disruptions


def test_endpoint_get_model_metrics_unavailable_when_file_missing(test_setup):
    """Verify GET /model/metrics returns structured UNAVAILABLE response without crashing when metrics file is missing."""
    client, _, _ = test_setup
    nonexistent_path = Path("/path/to/nonexistent/metrics_file.json")

    app.dependency_overrides[get_metrics_filepath] = lambda: nonexistent_path
    try:
        resp = client.get("/model/metrics")
        assert resp.status_code == 200  # Must not crash!
        data = resp.json()
        assert data["status"] == "UNAVAILABLE"
        assert data["is_available"] is False
        assert data["baseline_mae"] is None
        assert data["ml_mae"] is None
        assert data["baseline_rmse"] is None
        assert data["ml_rmse"] is None
        assert data["metrics_by_horizon"] is None
        assert data["metrics_by_disruption_status"] is None
        assert "not found" in data["message"].lower()
    finally:
        app.dependency_overrides.pop(get_metrics_filepath, None)


def test_endpoint_get_model_metrics_corrupted_json(test_setup, tmp_path):
    """Verify GET /model/metrics handles corrupted JSON gracefully without crashing."""
    client, _, _ = test_setup
    corrupt_file = tmp_path / "corrupt_metrics.json"
    corrupt_file.write_text("{ this is corrupted invalid json content !!!")

    app.dependency_overrides[get_metrics_filepath] = lambda: corrupt_file
    try:
        resp = client.get("/model/metrics")
        assert resp.status_code == 200  # Must not crash!
        data = resp.json()
        assert data["status"] == "UNAVAILABLE"
        assert data["is_available"] is False
        assert data["baseline_mae"] is None
        assert "failed to load" in data["message"].lower()
    finally:
        app.dependency_overrides.pop(get_metrics_filepath, None)


def test_endpoint_get_model_metrics_both_files_missing(test_setup):
    """Verify graceful handling when both metrics and metadata are missing."""
    client, _, _ = test_setup
    app.dependency_overrides[get_metrics_filepath] = lambda: Path("/missing/metrics.json")
    app.dependency_overrides[get_metadata_filepath] = lambda: Path("/missing/metadata.json")
    try:
        resp = client.get("/model/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "UNAVAILABLE"
        assert data["is_available"] is False
        assert data["total_training_samples"] is None
        assert data["baseline_mae"] is None
    finally:
        app.dependency_overrides.pop(get_metrics_filepath, None)
        app.dependency_overrides.pop(get_metadata_filepath, None)





