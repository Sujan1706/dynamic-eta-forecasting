from datetime import date, datetime, timezone, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.connection import Base
from backend.database.models import Train, Event
from backend.database.seed import seed_data
from backend.services.schemas import TrainRunningState
from backend.simulator.events import EventType, SimulationEvent
from backend.simulator.journey import SimulatedJourney, StationStop
from backend.simulator.engine import TrainSimulator


@pytest.fixture
def in_memory_db():
    """Provides an isolated, in-memory SQLite database populated with seed network."""
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


@pytest.fixture
def sample_stops():
    """Provides 3 ordered synthetic stops for granular isolated testing."""
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
            distance_from_source_km=100.0,
            scheduled_arrival="18:35",
            scheduled_departure="18:40",
            scheduled_stop_minutes=5.0,
        ),
        StationStop(
            sequence=3,
            station_id=3,
            station_code="PRYJ",
            station_name="Prayagraj Junction",
            distance_from_source_km=250.0,
            scheduled_arrival="20:30",
            scheduled_departure=None,
            scheduled_stop_minutes=0.0,
        ),
    ]


def test_load_from_sqlite(in_memory_db):
    """Verify loading trains, routes, and stations directly from SQLite."""
    sim = TrainSimulator(db=in_memory_db)
    loaded_count = sim.load_from_db()

    assert loaded_count == 8  # 3 routes total 8 trains in seed data
    assert "12302" in sim.journeys
    assert "12952" in sim.journeys
    assert "12028" in sim.journeys

    j_rajdhani = sim.get_journey("12302")
    assert j_rajdhani.train_name == "Howrah Rajdhani Express"
    assert len(j_rajdhani.stops) == 8
    assert j_rajdhani.stops[0].station_code == "NDLS"
    assert j_rajdhani.stops[-1].station_code == "HWH"


def test_train_movement(sample_stops):
    """Verify that ticking advances the train along the segment and updates distance/progress."""
    journey = SimulatedJourney(
        train_id=1,
        train_number="12302",
        train_name="Rajdhani",
        train_type="RAJDHANI",
        route_id=1,
        stops=sample_stops,
        nominal_speed_kmh=100.0,
        sim_time=datetime(2026, 9, 26, 10, 0, tzinfo=timezone.utc),
    )

    assert journey.segment_progress == 0.0
    assert journey.next_station_distance_km == 100.0
    assert journey.status == "RUNNING"
    assert journey.current_speed_kmh == 100.0

    # Advance 1800 seconds (30 minutes) at 100 km/h -> 50 km distance covered
    journey.advance(dt_seconds=1800.0)

    assert journey.segment_progress == pytest.approx(0.5, abs=0.01)
    assert journey.next_station_distance_km == pytest.approx(50.0, abs=0.5)
    assert journey.current_station.station_code == "NDLS"
    assert journey.next_station.station_code == "CNB"
    assert journey.sim_time == datetime(2026, 9, 26, 10, 30, tzinfo=timezone.utc)


def test_delay_increase(sample_stops):
    """Verify controlled delay increase via direct event injection and speed restrictions."""
    journey = SimulatedJourney(
        train_id=1,
        train_number="12302",
        train_name="Rajdhani",
        train_type="RAJDHANI",
        route_id=1,
        stops=sample_stops,
        nominal_speed_kmh=100.0,
        current_delay_minutes=0.0,
    )

    # 1. Injected event with explicit delay
    event = SimulationEvent(
        event_type=EventType.UNSCHEDULED_HALT,
        delay_minutes=15.0,
        severity="HIGH",
    )
    journey.inject_event(event)

    assert journey.current_delay_minutes == 15.0
    assert journey.status == "HALTED"
    assert journey.current_speed_kmh == 0.0

    # 2. Delay accumulation under SPEED_RESTRICTION
    # Move under restriction: nominal 100 km/h, restricted to 50 km/h for 1 hour (3600s)
    # Expected lost distance = 50 km -> 30 minutes of lost time added to delay
    journey.active_events.clear()
    journey.status = "RUNNING"
    restriction = SimulationEvent(
        event_type=EventType.SPEED_RESTRICTION,
        delay_minutes=0.0,
        metadata={"speed_limit_kmh": 50.0},
        remaining_duration_seconds=3600.0,
    )
    journey.inject_event(restriction)
    assert journey.current_speed_kmh == 50.0

    journey.advance(dt_seconds=3600.0)
    assert journey.current_delay_minutes == pytest.approx(45.0, abs=0.5)  # 15 + 30 = 45


def test_event_injection_all_types(in_memory_db, sample_stops):
    """Verify that all 5 event types modify running state deterministically and record to SQLite."""
    sim = TrainSimulator(db=in_memory_db)
    journey = SimulatedJourney(
        train_id=1,
        train_number="12302",
        train_name="Rajdhani",
        train_type="RAJDHANI",
        route_id=1,
        stops=sample_stops,
        nominal_speed_kmh=100.0,
        journey_id=1,  # matches seeded journey
    )
    sim.register_journey(journey)

    # 1. SIGNAL_HALT
    ev1 = sim.inject_event("12302", EventType.SIGNAL_HALT, delay_minutes=10.0, db=in_memory_db)
    assert journey.status == "HALTED"
    assert journey.current_speed_kmh == 0.0
    assert journey.current_delay_minutes == 10.0

    # 2. CONGESTION
    journey.active_events.clear()
    ev2 = sim.inject_event(
        "12302",
        EventType.CONGESTION,
        delay_minutes=5.0,
        metadata={"speed_factor": 0.4},
        db=in_memory_db,
    )
    assert journey.current_speed_kmh == pytest.approx(40.0, abs=0.1)
    assert journey.current_delay_minutes == 15.0

    # 3. SPEED_RESTRICTION
    journey.active_events.clear()
    ev3 = sim.inject_event(
        "12302",
        EventType.SPEED_RESTRICTION,
        delay_minutes=4.0,
        metadata={"speed_limit_kmh": 35.0},
        db=in_memory_db,
    )
    assert journey.current_speed_kmh == 35.0
    assert journey.current_delay_minutes == 19.0

    # 4. UNSCHEDULED_HALT
    journey.active_events.clear()
    ev4 = sim.inject_event("12302", EventType.UNSCHEDULED_HALT, delay_minutes=20.0, db=in_memory_db)
    assert journey.status == "HALTED"
    assert journey.current_speed_kmh == 0.0
    assert journey.current_delay_minutes == 39.0

    # 5. WEATHER
    journey.active_events.clear()
    ev5 = sim.inject_event(
        "12302",
        EventType.WEATHER,
        delay_minutes=8.0,
        metadata={"speed_factor": 0.6, "max_speed_kmh": 50.0},
        db=in_memory_db,
    )
    assert journey.current_speed_kmh == 50.0
    assert journey.current_delay_minutes == 47.0

    # Verify SQLite DB persistence
    db_events = in_memory_db.query(Event).filter(Event.journey_id == 1).all()
    logged_types = [e.event_type for e in db_events]
    for expected in ["SIGNAL_HALT", "CONGESTION", "SPEED_RESTRICTION", "UNSCHEDULED_HALT", "WEATHER"]:
        assert expected in logged_types


def test_station_transition(sample_stops):
    """Verify sequential station arrival, dwell time, departure, and final terminus completion."""
    journey = SimulatedJourney(
        train_id=1,
        train_number="12302",
        train_name="Rajdhani",
        train_type="RAJDHANI",
        route_id=1,
        stops=sample_stops,
        nominal_speed_kmh=100.0,
    )

    assert journey.current_station.station_code == "NDLS"
    assert journey.current_sequence == 1
    assert journey.previous_station is None
    assert journey.next_station.station_code == "CNB"

    # Step 1: Advance by 1.1 hours (3960s) at 100 km/h -> 110 km (segment is 100 km)
    # Train reaches and transitions to Station 2 (CNB)!
    journey.advance(dt_seconds=3960.0)

    assert journey.current_station.station_code == "CNB"
    assert journey.current_sequence == 2
    assert journey.previous_station.station_code == "NDLS"
    assert journey.next_station.station_code == "PRYJ"
    assert journey.segment_progress == 0.0
    # CNB has scheduled_stop_minutes=5.0 -> 300s dwell
    assert journey.dwell_time_remaining_seconds == 300.0
    assert journey.current_speed_kmh == 0.0

    # Step 2: Advance dwell time (300s) -> leaves dwell, resumes cruise
    journey.advance(dt_seconds=300.0)
    assert journey.dwell_time_remaining_seconds == 0.0
    assert journey.current_speed_kmh == 100.0

    # Step 3: Advance 2.0 hours (7200s) -> 200 km (segment to PRYJ is 150 km)
    # Reaches final terminus PRYJ!
    journey.advance(dt_seconds=7200.0)

    assert journey.current_station.station_code == "PRYJ"
    assert journey.current_sequence == 3
    assert journey.previous_station.station_code == "CNB"
    assert journey.next_station is None
    assert journey.status == "COMPLETED"
    assert journey.segment_progress == 1.0
    assert journey.current_speed_kmh == 0.0
    assert journey.estimated_next_station_arrival is None


def test_normalized_state_generation(sample_stops):
    """Verify that simulator produces the exact normalized TrainRunningState schema."""
    journey = SimulatedJourney(
        train_id=1,
        train_number="12302",
        train_name="Howrah Rajdhani Express",
        train_type="RAJDHANI",
        route_id=1,
        stops=sample_stops,
        nominal_speed_kmh=110.0,
        current_delay_minutes=12.5,
        sim_time=datetime(2026, 9, 26, 17, 30, tzinfo=timezone.utc),
    )

    state = journey.to_train_running_state()

    # 1. Type validation
    assert isinstance(state, TrainRunningState)

    # 2. Strict field verification (all 14 schema fields)
    assert state.train_number == "12302"
    assert state.journey_date == date.today()
    assert state.train_name == "Howrah Rajdhani Express"
    assert state.status == "RUNNING"
    assert state.current_station_code == "NDLS"
    assert state.current_station_sequence == 1
    assert state.current_delay_minutes == 12.5
    assert state.previous_station_code is None
    assert state.next_station_code == "CNB"
    assert state.next_station_distance_km == 100.0
    assert state.segment_progress == 0.0
    assert state.speed_kmh == 110.0
    assert state.timestamp == datetime(2026, 9, 26, 17, 30, tzinfo=timezone.utc)
    assert state.source == "simulator"


def test_configurable_simulation_speed_and_ticks(sample_stops):
    """Verify configurable simulation speed multiplier and tick advancement."""
    sim = TrainSimulator(
        simulation_speed=10.0,
        start_time=datetime(2026, 9, 26, 8, 0, tzinfo=timezone.utc),
    )
    journey = SimulatedJourney(
        train_id=1,
        train_number="12028",
        train_name="Shatabdi",
        train_type="SHATABDI",
        route_id=1,
        stops=sample_stops,
        nominal_speed_kmh=120.0,
        sim_time=sim.sim_time,
    )
    sim.register_journey(journey)

    # 1 tick of 60 seconds at 10x speed advances 600 simulated seconds (10 minutes)
    states = sim.tick(step_seconds=60.0)

    assert sim.sim_time == datetime(2026, 9, 26, 8, 10, tzinfo=timezone.utc)
    state = states["12028"]
    # 10 minutes at 120 km/h covers 20 km (20% of 100km segment)
    assert state.segment_progress == pytest.approx(0.20, abs=0.01)
    assert state.next_station_distance_km == pytest.approx(80.0, abs=1.0)


def test_estimated_next_station_arrival(sample_stops):
    """Verify dynamic next-station arrival estimation factoring remaining distance and speed."""
    journey = SimulatedJourney(
        train_id=1,
        train_number="12302",
        train_name="Rajdhani",
        train_type="RAJDHANI",
        route_id=1,
        stops=sample_stops,
        nominal_speed_kmh=100.0,
        sim_time=datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc),
    )

    # At start: 100 km remaining at 100 km/h = 1 hour (60 minutes)
    eta = journey.estimated_next_station_arrival
    assert eta is not None
    assert eta == datetime(2026, 9, 26, 13, 0, tzinfo=timezone.utc)

    # Inject a 15-minute halt event
    journey.inject_event(SimulationEvent(EventType.SIGNAL_HALT, delay_minutes=15.0))
    eta_delayed = journey.estimated_next_station_arrival
    assert eta_delayed == datetime(2026, 9, 26, 13, 15, tzinfo=timezone.utc)
