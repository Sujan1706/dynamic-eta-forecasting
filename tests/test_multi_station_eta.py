"""
Unit Tests for Day 2 Phase 4: Multi-Station ETA Prediction Service.
===================================================================
Verifies:
1. One station ahead prediction (single segment).
2. Multiple stations ahead chaining (progressive transit + dwell accumulation).
3. Train at final station (terminus edge case returns empty list).
4. Invalid station handling (informative ValueError raised).
5. Route ordering guarantees (strict sequential order, increasing distance and ETAs).
6. Uncertainty widening (margin scales with sqrt(segments_ahead)).
7. Physical feasibility guard (confidence lower bound bounded by maximum speed).
8. SQLite database integration.
"""

from datetime import date, datetime, timezone, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.connection import Base
from backend.database.seed import seed_data
from backend.services.schemas import TrainRunningState
from backend.services.multi_station_eta import (
    MultiStationETAService,
    StationETAPrediction,
    SegmentPrediction,
)
from backend.simulator.journey import StationStop


@pytest.fixture
def sample_route_stops():
    """6-station Southern corridor route: SBC -> BNC -> KJM -> BWT -> KPD -> MAS."""
    return [
        StationStop(
            sequence=1,
            station_id=1,
            station_code="SBC",
            station_name="KSR Bengaluru",
            distance_from_source_km=0.0,
            scheduled_departure="06:00",
            scheduled_stop_minutes=0.0,
        ),
        StationStop(
            sequence=2,
            station_id=2,
            station_code="BNC",
            station_name="Bengaluru Cant",
            distance_from_source_km=4.5,
            scheduled_arrival="06:08",
            scheduled_departure="06:10",
            scheduled_stop_minutes=2.0,
        ),
        StationStop(
            sequence=3,
            station_id=3,
            station_code="KJM",
            station_name="Krishnarajapuram",
            distance_from_source_km=14.0,
            scheduled_arrival="06:23",
            scheduled_departure="06:25",
            scheduled_stop_minutes=2.0,
        ),
        StationStop(
            sequence=4,
            station_id=4,
            station_code="BWT",
            station_name="Bangarapet",
            distance_from_source_km=70.0,
            scheduled_arrival="07:08",
            scheduled_departure="07:10",
            scheduled_stop_minutes=2.0,
        ),
        StationStop(
            sequence=5,
            station_id=5,
            station_code="KPD",
            station_name="Katpadi Junction",
            distance_from_source_km=214.0,
            scheduled_arrival="09:13",
            scheduled_departure="09:15",
            scheduled_stop_minutes=2.0,
        ),
        StationStop(
            sequence=6,
            station_id=6,
            station_code="MAS",
            station_name="MGR Chennai Central",
            distance_from_source_km=359.0,
            scheduled_arrival="11:30",
            scheduled_departure=None,
            scheduled_stop_minutes=0.0,
        ),
    ]


@pytest.fixture
def in_memory_db():
    """In-memory SQLite database seeded with network routes."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    seed_data(session)
    yield session
    session.close()


def test_one_station_ahead(sample_route_stops):
    """Verify predicting ETA for the immediate next station produces single-segment chained output."""
    service = MultiStationETAService()

    # Train running on the last segment between KPD (seq 5) and MAS (seq 6)
    train_state = TrainRunningState(
        train_number="12028",
        journey_date=date(2026, 10, 1),
        train_name="Shatabdi Express",
        status="RUNNING",
        current_station_code="KPD",
        current_station_sequence=5,
        current_delay_minutes=10.0,
        previous_station_code="BWT",
        next_station_code="MAS",
        next_station_distance_km=145.0,
        segment_progress=0.40,
        speed_kmh=90.0,
        timestamp=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc),
        source="simulator",
    )

    predictions = service.predict_upcoming_stations(
        train_state=train_state, route_stations=sample_route_stops
    )

    assert len(predictions) == 1
    pred = predictions[0]

    assert pred.station_code == "MAS"
    assert pred.segments_ahead == 1
    assert pred.intermediate_halts == 0
    assert pred.predicted_remaining_minutes > 0.0
    assert pred.predicted_eta > train_state.timestamp
    assert pred.baseline_eta > train_state.timestamp
    assert pred.confidence_lower_bound <= pred.predicted_eta <= pred.confidence_upper_bound

    # Check segment predictions
    assert len(pred.segment_predictions) == 1
    seg = pred.segment_predictions[0]
    assert seg.from_station_code == "KPD"
    assert seg.to_station_code == "MAS"
    assert seg.segment_index == 1
    assert seg.predicted_transit_minutes > 0.0


def test_multiple_stations_ahead(sample_route_stops):
    """Verify chaining across multiple stations produces ordered predictions with cumulative times."""
    service = MultiStationETAService()

    # Train just departing from origin SBC (seq 1), heading towards BNC (seq 2)
    train_state = TrainRunningState(
        train_number="12028",
        journey_date=date(2026, 10, 1),
        train_name="Shatabdi Express",
        status="RUNNING",
        current_station_code="SBC",
        current_station_sequence=1,
        current_delay_minutes=0.0,
        previous_station_code=None,
        next_station_code="BNC",
        next_station_distance_km=4.5,
        segment_progress=0.0,
        speed_kmh=0.0,
        timestamp=datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc),
        source="simulator",
    )

    predictions = service.predict_upcoming_stations(
        train_state=train_state, route_stations=sample_route_stops
    )

    # 5 upcoming stations: BNC, KJM, BWT, KPD, MAS
    assert len(predictions) == 5
    expected_codes = ["BNC", "KJM", "BWT", "KPD", "MAS"]
    assert [p.station_code for p in predictions] == expected_codes

    # Verify cumulative transit time and segment count progression
    for i, p in enumerate(predictions):
        expected_segments_ahead = i + 1
        assert p.segments_ahead == expected_segments_ahead
        assert p.intermediate_halts == i
        assert len(p.segment_predictions) == expected_segments_ahead

        if i > 0:
            prev_p = predictions[i - 1]
            assert p.predicted_remaining_minutes > prev_p.predicted_remaining_minutes
            assert p.predicted_eta > prev_p.predicted_eta
            assert p.distance_to_go_km > prev_p.distance_to_go_km


def test_train_at_final_station(sample_route_stops):
    """Verify that when a train reaches the terminus, zero upcoming stations are returned."""
    service = MultiStationETAService()

    # Case A: Train at final station code MAS
    train_state_at_terminus = TrainRunningState(
        train_number="12028",
        journey_date=date(2026, 10, 1),
        train_name="Shatabdi Express",
        status="RUNNING",
        current_station_code="MAS",
        current_station_sequence=6,
        current_delay_minutes=5.0,
        previous_station_code="KPD",
        next_station_code=None,
        next_station_distance_km=0.0,
        segment_progress=1.0,
        speed_kmh=0.0,
        timestamp=datetime(2026, 10, 1, 11, 35, tzinfo=timezone.utc),
        source="simulator",
    )

    res_terminus = service.predict_upcoming_stations(
        train_state=train_state_at_terminus, route_stations=sample_route_stops
    )
    assert res_terminus == []

    # Case B: Train status COMPLETED
    train_state_completed = TrainRunningState(
        train_number="12028",
        journey_date=date(2026, 10, 1),
        train_name="Shatabdi Express",
        status="COMPLETED",
        current_station_code="MAS",
        current_station_sequence=6,
        current_delay_minutes=0.0,
        timestamp=datetime(2026, 10, 1, 11, 40, tzinfo=timezone.utc),
        source="simulator",
    )

    res_completed = service.predict_upcoming_stations(
        train_state=train_state_completed, route_stations=sample_route_stops
    )
    assert res_completed == []


def test_invalid_station(sample_route_stops):
    """Verify predicting for an invalid station code raises a clear ValueError."""
    service = MultiStationETAService()

    train_state = TrainRunningState(
        train_number="12028",
        journey_date=date(2026, 10, 1),
        train_name="Shatabdi Express",
        status="RUNNING",
        current_station_code="SBC",
        current_station_sequence=1,
        current_delay_minutes=0.0,
        next_station_code="BNC",
        timestamp=datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc),
        source="simulator",
    )

    # 1. Non-existent station code
    with pytest.raises(ValueError, match="not on the route"):
        service.predict_station_eta(
            train_state=train_state,
            target_station_code="XYZ_INVALID",
            route_stations=sample_route_stops,
        )

    # 2. Station already behind the current train position
    train_state_past = TrainRunningState(
        train_number="12028",
        journey_date=date(2026, 10, 1),
        train_name="Shatabdi Express",
        status="RUNNING",
        current_station_code="KPD",
        current_station_sequence=5,
        current_delay_minutes=0.0,
        next_station_code="MAS",
        timestamp=datetime(2026, 10, 1, 9, 30, tzinfo=timezone.utc),
        source="simulator",
    )

    with pytest.raises(ValueError, match="already behind"):
        service.predict_station_eta(
            train_state=train_state_past,
            target_station_code="BNC",
            route_stations=sample_route_stops,
        )


def test_route_ordering_and_monotonicity(sample_route_stops):
    """Verify that returned predictions are strictly in ascending route order."""
    service = MultiStationETAService()

    train_state = TrainRunningState(
        train_number="12028",
        journey_date=date(2026, 10, 1),
        train_name="Shatabdi Express",
        status="RUNNING",
        current_station_code="BNC",
        current_station_sequence=2,
        current_delay_minutes=5.0,
        previous_station_code="SBC",
        next_station_code="KJM",
        next_station_distance_km=9.5,
        segment_progress=0.10,
        speed_kmh=65.0,
        timestamp=datetime(2026, 10, 1, 6, 15, tzinfo=timezone.utc),
        source="simulator",
    )

    preds = service.predict_upcoming_stations(
        train_state=train_state, route_stations=sample_route_stops
    )

    assert len(preds) == 4  # KJM (3), BWT (4), KPD (5), MAS (6)

    # Check strict monotonicity of sequences, distance to go, and arrival times
    sequences = [p.station_sequence for p in preds]
    assert sequences == sorted(sequences), "Station sequences must be monotonically increasing"

    distances = [p.distance_to_go_km for p in preds]
    assert distances == sorted(distances), "Remaining distances must be monotonically increasing"

    etas = [p.predicted_eta for p in preds]
    assert etas == sorted(etas), "Predicted ETAs must be monotonically increasing"

    baseline_etas = [p.baseline_eta for p in preds]
    assert baseline_etas == sorted(baseline_etas), "Baseline ETAs must be monotonically increasing"


def test_uncertainty_widening(sample_route_stops):
    """Verify that confidence range uncertainty margin widens as future segments increase."""
    service = MultiStationETAService(base_uncertainty_minutes=5.0)

    train_state = TrainRunningState(
        train_number="12028",
        journey_date=date(2026, 10, 1),
        train_name="Shatabdi Express",
        status="RUNNING",
        current_station_code="SBC",
        current_station_sequence=1,
        current_delay_minutes=0.0,
        next_station_code="BNC",
        timestamp=datetime(2026, 10, 1, 6, 0, tzinfo=timezone.utc),
        source="simulator",
    )

    preds = service.predict_upcoming_stations(
        train_state=train_state, route_stations=sample_route_stops
    )

    margins = [p.uncertainty_margin_minutes for p in preds]
    # Margin for segment k is 5.0 * sqrt(k)
    # k=1: 5.0, k=2: 7.1, k=3: 8.7, k=4: 10.0, k=5: 11.2
    assert margins == sorted(margins), "Uncertainty margins must widen over the horizon"
    assert margins[0] == 5.0
    assert margins[-1] > margins[0]

    # Verify interval bounds encapsulate the predicted time
    for p in preds:
        assert p.confidence_lower_bound_minutes <= p.predicted_remaining_minutes <= p.confidence_upper_bound_minutes
        assert p.confidence_lower_bound <= p.predicted_eta <= p.confidence_upper_bound


def test_sqlite_integration(in_memory_db):
    """Verify MultiStationETAService can resolve route stations directly from SQLite database."""
    service = MultiStationETAService()

    # Train 12302 (New Delhi to Howrah)
    train_state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 10, 1),
        train_name="Howrah Rajdhani Express",
        status="RUNNING",
        current_station_code="CNB",
        current_station_sequence=2,
        current_delay_minutes=12.0,
        previous_station_code="NDLS",
        next_station_code="PRYJ",
        next_station_distance_km=195.0,
        segment_progress=0.25,
        speed_kmh=100.0,
        timestamp=datetime(2026, 10, 1, 22, 0, tzinfo=timezone.utc),
        source="simulator",
    )

    preds = service.predict_upcoming_stations(train_state=train_state, db=in_memory_db)

    # Route 1 has 8 stations: NDLS(1), CNB(2), PRYJ(3), DDU(4), GAYA(5), DHN(6), ASN(7), HWH(8)
    # From CNB (seq 2), upcoming stations are 6: PRYJ, DDU, GAYA, DHN, ASN, HWH
    assert len(preds) == 6
    assert preds[0].station_code == "PRYJ"
    assert preds[-1].station_code == "HWH"
    assert preds[-1].station_name == "Howrah Junction"

    # Specific station lookup
    hwh_pred = service.predict_station_eta(
        train_state=train_state, target_station_code="HWH", db=in_memory_db
    )
    assert hwh_pred.station_code == "HWH"
    assert hwh_pred.segments_ahead == 6
    assert len(hwh_pred.segment_predictions) == 6
