from datetime import date, datetime, timezone, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.connection import Base
from backend.database.seed import seed_data
from backend.services.schemas import TrainRunningState
from backend.services.baseline_eta import BaselineETAService, BaselineETAPrediction
from backend.simulator.journey import StationStop


@pytest.fixture
def sample_stops():
    """Northern corridor route stations spanning evening into next morning."""
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
            scheduled_arrival="01:42",  # Overnight arrival next day
            scheduled_departure="01:52",
            scheduled_stop_minutes=10.0,
        ),
        StationStop(
            sequence=5,
            station_id=5,
            station_code="GAYA",
            station_name="Gaya Junction",
            distance_from_source_km=993.0,
            scheduled_arrival="04:10",  # Overnight arrival next day
            scheduled_departure="04:13",
            scheduled_stop_minutes=3.0,
        ),
    ]


@pytest.fixture
def in_memory_db():
    """In-memory SQLite database populated with the 3-route seed network."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    seed_data(session)
    yield session
    session.close()


def test_scenario_1_on_time_train(sample_stops):
    """Scenario 1: On-time train predicting next station ETA equals scheduled ETA with zero recovery."""
    service = BaselineETAService()
    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Rajdhani",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        next_station_code="CNB",
        current_delay_minutes=0.0,
        segment_progress=0.0,
        speed_kmh=110.0,
        timestamp=datetime(2026, 9, 28, 16, 55, tzinfo=timezone.utc),
        source="simulator",
    )

    pred = service.predict_next_station(state, route_stations=sample_stops)

    assert isinstance(pred, BaselineETAPrediction)
    assert pred.train_number == "12302"
    assert pred.target_station == "CNB"
    assert pred.scheduled_eta == datetime(2026, 9, 28, 21, 35, tzinfo=timezone.utc)
    assert pred.baseline_eta == datetime(2026, 9, 28, 21, 35, tzinfo=timezone.utc)
    assert pred.current_delay == 0.0
    assert pred.recovery_applied == 0.0
    assert pred.predicted_delay == 0.0
    assert pred.method == "BASELINE"
    assert isinstance(pred.generated_timestamp, datetime)


def test_scenario_2_accumulated_delay_with_partial_recovery(sample_stops):
    """Scenario 2: Train with 25 min delay partially recovered by distance buffer."""
    service = BaselineETAService(recovery_rate_per_100km=2.5)
    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Rajdhani",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        next_station_code="PRYJ",
        current_delay_minutes=25.0,
        segment_progress=0.0,
        speed_kmh=110.0,
        timestamp=datetime(2026, 9, 28, 17, 20, tzinfo=timezone.utc),
        source="simulator",
    )

    pred = service.predict_station(state, "PRYJ", route_stations=sample_stops)

    # PRYJ distance is 635.0 km
    # Recovery buffer = (635.0 / 100.0) * 2.5 = 15.88 minutes
    # Predicted delay = 25.0 - 15.88 = 9.12 minutes
    assert pred.distance_to_go_km == 635.0
    assert pred.current_delay == 25.0
    assert pred.recovery_applied == pytest.approx(15.88, abs=0.02)
    assert pred.predicted_delay == pytest.approx(9.12, abs=0.02)

    expected_scheduled = datetime(2026, 9, 28, 23, 43, tzinfo=timezone.utc)
    assert pred.scheduled_eta == expected_scheduled
    assert pred.baseline_eta == pytest.approx(
        expected_scheduled + timedelta(minutes=9.12),
        abs=timedelta(seconds=2),
    )


def test_scenario_3_small_delay_full_recovery(sample_stops):
    """Scenario 3: Small delay (4 min) fully absorbed by recovery buffer; does not arrive earlier than schedule."""
    service = BaselineETAService(recovery_rate_per_100km=2.5)
    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Rajdhani",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        next_station_code="PRYJ",
        current_delay_minutes=4.0,  # 4 min delay is less than 15.88 min buffer
        segment_progress=0.0,
        speed_kmh=110.0,
        timestamp=datetime(2026, 9, 28, 17, 0, tzinfo=timezone.utc),
        source="simulator",
    )

    pred = service.predict_station(state, "PRYJ", route_stations=sample_stops)

    # Recovery applied capped at current_delay (4.0)
    assert pred.current_delay == 4.0
    assert pred.recovery_applied == 4.0
    assert pred.predicted_delay == 0.0
    assert pred.baseline_eta == pred.scheduled_eta


def test_scenario_4_overnight_multi_station_chaining(sample_stops):
    """Scenario 4: Multi-station chaining correctly computes dates across midnight."""
    service = BaselineETAService()
    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Rajdhani",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        next_station_code="CNB",
        current_delay_minutes=15.0,
        segment_progress=0.0,
        speed_kmh=110.0,
        timestamp=datetime(2026, 9, 28, 17, 10, tzinfo=timezone.utc),
        source="simulator",
    )

    preds = service.predict_upcoming_stations(state, route_stations=sample_stops)

    # Should predict all 4 upcoming stations: CNB, PRYJ, DDU, GAYA
    assert len(preds) == 4
    target_codes = [p.target_station for p in preds]
    assert target_codes == ["CNB", "PRYJ", "DDU", "GAYA"]

    # 1. CNB is same evening (2026-09-28)
    assert preds[0].scheduled_eta.date() == date(2026, 9, 28)
    assert preds[0].scheduled_eta == datetime(2026, 9, 28, 21, 35, tzinfo=timezone.utc)

    # 2. PRYJ is late night (2026-09-28 23:43)
    assert preds[1].scheduled_eta == datetime(2026, 9, 28, 23, 43, tzinfo=timezone.utc)

    # 3. DDU is early morning of NEXT day (2026-09-29 01:42)
    assert preds[2].scheduled_eta.date() == date(2026, 9, 29)
    assert preds[2].scheduled_eta == datetime(2026, 9, 29, 1, 42, tzinfo=timezone.utc)
    assert preds[2].baseline_eta.date() == date(2026, 9, 29)

    # 4. GAYA is next morning (2026-09-29 04:10)
    assert preds[3].scheduled_eta.date() == date(2026, 9, 29)
    assert preds[3].scheduled_eta == datetime(2026, 9, 29, 4, 10, tzinfo=timezone.utc)

    # Monotonically increasing ETAs
    for i in range(len(preds) - 1):
        assert preds[i].baseline_eta < preds[i + 1].baseline_eta


def test_scenario_5_physical_feasibility_guard(sample_stops):
    """Scenario 5: Physical guard prevents impossible arrival times by enforcing maximum permissible speed."""
    # Max speed 120 km/h -> 2.0 km/min
    service = BaselineETAService(max_permissible_speed_kmh=120.0)

    # Current time is 21:30. CNB scheduled arrival is 21:35 (only 5 minutes away)
    # But train is at NDLS (440 km away)
    # Physically covering 440 km at 120 km/h takes at least 220 minutes (3h 40m)
    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Rajdhani",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        next_station_code="CNB",
        current_delay_minutes=0.0,
        segment_progress=0.0,
        speed_kmh=100.0,
        timestamp=datetime(2026, 9, 28, 21, 30, tzinfo=timezone.utc),  # Late timestamp
        source="simulator",
    )

    pred = service.predict_station(state, "CNB", route_stations=sample_stops)

    # Minimum travel time: (440 / 120) * 60 = 220 minutes
    earliest_feasible = datetime(2026, 9, 28, 21, 30, tzinfo=timezone.utc) + timedelta(minutes=220.0)
    assert pred.baseline_eta >= earliest_feasible


def test_scenario_6_completed_journey(sample_stops):
    """Scenario 6: Completed train yields None for next station and empty list for upcoming stations."""
    service = BaselineETAService()
    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Rajdhani",
        status="COMPLETED",
        current_station_code="GAYA",
        current_station_sequence=5,
        next_station_code=None,
        timestamp=datetime.now(timezone.utc),
        source="simulator",
    )

    assert service.predict_next_station(state, route_stations=sample_stops) is None
    assert service.predict_upcoming_stations(state, route_stations=sample_stops) == []


def test_scenario_7_sqlite_integration(in_memory_db):
    """Scenario 7: Full integration querying routes and trains directly from SQLite."""
    service = BaselineETAService()
    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Howrah Rajdhani Express",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        next_station_code="CNB",
        current_delay_minutes=18.0,
        segment_progress=0.0,
        speed_kmh=110.0,
        timestamp=datetime(2026, 9, 28, 17, 15, tzinfo=timezone.utc),
        source="simulator",
    )

    # 1. Predict next station directly from DB
    next_pred = service.predict_next_station(state, db=in_memory_db)
    assert next_pred is not None
    assert next_pred.target_station == "CNB"
    assert next_pred.target_station_name == "Kanpur Central"
    assert next_pred.target_sequence == 2
    assert next_pred.current_delay == 18.0
    assert next_pred.method == "BASELINE"

    # 2. Predict all upcoming stops down to HWH (7 upcoming stations in 8-station route)
    all_preds = service.predict_upcoming_stations(state, db=in_memory_db)
    assert len(all_preds) == 7
    final_pred = all_preds[-1]
    assert final_pred.target_station == "HWH"
    assert final_pred.target_station_name == "Howrah Junction"
    assert final_pred.target_sequence == 8
    # Route distance to HWH is 1450.0 km
    assert final_pred.distance_to_go_km == 1450.0
    # Next day arrival at 09:55
    assert final_pred.scheduled_eta == datetime(2026, 9, 29, 9, 55, tzinfo=timezone.utc)
    # Recovery buffer for 1450km = (1450/100)*2.5 = 36.25 min -> recovers all 18 min delay
    assert final_pred.recovery_applied == 18.0
    assert final_pred.predicted_delay == 0.0
    assert final_pred.baseline_eta == final_pred.scheduled_eta
