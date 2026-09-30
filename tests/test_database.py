import pytest
from datetime import date
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from backend.database.models import Base, Station, Route, RouteStation, Train, Journey, Event
from backend.database.init_db import init_db
from backend.database.seed import seed_data, DATA_DISCLAIMER
from backend.database.repository import (
    StationRepository,
    RouteRepository,
    TrainRepository,
    JourneyRepository,
    EventRepository,
)


@pytest.fixture
def db_session():
    """Provides a fresh isolated in-memory SQLite database session for each test."""
    test_engine = create_engine("sqlite:///:memory:")
    with test_engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


def test_init_db():
    """Verify that all 6 required entities exist in the database after initialization."""
    test_engine = create_engine("sqlite:///:memory:")
    init_db(target_engine=test_engine)

    inspector = inspect(test_engine)
    tables = inspector.get_table_names()

    required_tables = ["stations", "routes", "route_stations", "trains", "journeys", "events"]
    for table in required_tables:
        assert table in tables, f"Expected table '{table}' to be created"


def test_seed_3_routes_network(db_session):
    """
    Verify the 3-route development/simulation network:
    - 20 realistic stations across 3 regional corridors
    - 3 distinct routes with 6 to 8 ordered stations each
    - Correct approximate distances, scheduled arrival/departure, and stop times
    """
    result = seed_data(db=db_session)
    assert result["status"] == "seeded"
    assert result["stations_count"] == 20
    assert result["routes_count"] == 3
    assert result["trains_count"] == 8
    assert result["is_simulation_data"] is True
    assert "DEVELOPMENT/SIMULATION DATA ONLY" in result["disclaimer"]

    # 1. Verify Total Stations
    stations = StationRepository.get_all(db_session)
    assert len(stations) == 20

    # 2. Verify Route 1: Northern-Eastern Trunk (NDLS -> HWH, 8 stations)
    r1_id = result["route_ids"][0]
    r1 = RouteRepository.get_by_id(db_session, r1_id)
    assert r1.source_station.code == "NDLS"
    assert r1.destination_station.code == "HWH"
    r1_stops = RouteRepository.get_route_stations(db_session, r1_id)
    assert len(r1_stops) == 8
    assert [s.station.code for s in r1_stops] == ["NDLS", "CNB", "PRYJ", "DDU", "GAYA", "DHN", "ASN", "HWH"]
    # Check sequence ordering and distance monotonicity
    for i, stop in enumerate(r1_stops):
        assert stop.sequence == i + 1
        if i > 0:
            assert stop.distance_from_source_km > r1_stops[i - 1].distance_from_source_km
    assert r1_stops[0].distance_from_source_km == 0.0
    assert r1_stops[-1].distance_from_source_km == 1450.0

    # 3. Verify Route 2: Western Trunk (NDLS -> MMCT, 7 stations)
    r2_id = result["route_ids"][1]
    r2 = RouteRepository.get_by_id(db_session, r2_id)
    assert r2.source_station.code == "NDLS"
    assert r2.destination_station.code == "MMCT"
    r2_stops = RouteRepository.get_route_stations(db_session, r2_id)
    assert len(r2_stops) == 7
    assert [s.station.code for s in r2_stops] == ["NDLS", "MTJ", "KOTA", "RTM", "BRC", "ST", "MMCT"]
    for i, stop in enumerate(r2_stops):
        assert stop.sequence == i + 1
    assert r2_stops[-1].distance_from_source_km == 1385.0

    # 4. Verify Route 3: Southern Corridor (SBC -> MAS, 6 stations)
    r3_id = result["route_ids"][2]
    r3 = RouteRepository.get_by_id(db_session, r3_id)
    assert r3.source_station.code == "SBC"
    assert r3.destination_station.code == "MAS"
    r3_stops = RouteRepository.get_route_stations(db_session, r3_id)
    assert len(r3_stops) == 6
    assert [s.station.code for s in r3_stops] == ["SBC", "BNC", "KJM", "BWT", "KPD", "MAS"]
    for i, stop in enumerate(r3_stops):
        assert stop.sequence == i + 1
    assert r3_stops[-1].distance_from_source_km == 359.0

    # 5. Verify Stopping times and timings exist on intermediate stops
    cnb_stop = r1_stops[1]
    assert cnb_stop.station.code == "CNB"
    assert cnb_stop.scheduled_arrival == "21:35"
    assert cnb_stop.scheduled_departure == "21:40"
    assert cnb_stop.scheduled_stop_minutes == 5.0

    # 6. Verify Active Simulated Journeys and Disruption Events
    active_journeys = JourneyRepository.get_active(db_session)
    assert len(active_journeys) == 2
    for j in active_journeys:
        events = EventRepository.get_by_journey(db_session, j.id)
        assert len(events) >= 1
        assert events[0].event_metadata.get("data_source") == "SIMULATION_DATA"


def test_seed_idempotency(db_session):
    """Calling seed_data a second time must be a safe no-op without duplicating records."""
    res1 = seed_data(db=db_session)
    assert res1["status"] == "seeded"

    res2 = seed_data(db=db_session)
    assert res2["status"] == "already_seeded"

    assert len(StationRepository.get_all(db_session)) == 20


def test_foreign_key_constraints(db_session):
    """Verify that foreign key violations raise IntegrityError."""
    invalid_train = Train(
        train_number="99999",
        name="Ghost Train",
        train_type="EXPRESS",
        route_id=9999,  # Non-existent route
    )
    db_session.add(invalid_train)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_repository_journey_progress(db_session):
    """Verify journey progress updates via repository methods."""
    seed_data(db=db_session)
    j = JourneyRepository.get_active(db_session)[0]

    # Advance journey to next station PRYJ with updated delay
    pryj = StationRepository.get_by_code(db_session, "PRYJ")
    updated = JourneyRepository.update_progress(
        db=db_session,
        journey_id=j.id,
        current_station_id=pryj.id,
        current_sequence=3,
        current_delay_minutes=18.5,
        status="RUNNING",
    )

    assert updated.current_station_id == pryj.id
    assert updated.current_sequence == 3
    assert updated.current_delay_minutes == 18.5
