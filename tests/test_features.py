from datetime import date, datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.connection import Base
from backend.database.seed import seed_data
from backend.features.feature_builder import (
    FeatureBuilder,
    TrainFeatures,
    SectionStatsProvider,
    compute_cumulative_schedule_minutes,
)
from backend.services.schemas import TrainRunningState
from backend.simulator.events import SimulationEvent, EventType
from backend.simulator.journey import StationStop


@pytest.fixture
def sample_stops():
    """Synthetic Northern trunk corridor with known timetable across midnight."""
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
            scheduled_arrival="01:42",  # Next day arrival across midnight
            scheduled_departure="01:52",
            scheduled_stop_minutes=10.0,
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


def test_feature_schema_typing():
    """Verify that TrainFeatures strictly validates the 11 core ML features and export methods."""
    features = TrainFeatures(
        train_number="12302",
        target_station_code="CNB",
        distance_to_go_km=440.0,
        scheduled_time_to_go_min=280.0,
        num_intermediate_halts=0,
        current_delay_min=10.0,
        delay_trend_3pt=2.5,
        active_speed_restriction_flag=0,
        congestion_score_downstream=0.15,
        hist_avg_delay_this_section=8.5,
        hist_recovery_rate_section=2.5,
        weather_flag=0,
        day_type=0,
    )

    d = features.to_model_input_dict()
    assert len(d) == 11
    assert "distance_to_go_km" in d
    assert "day_type" in d
    assert d["num_intermediate_halts"] == 0

    lst = features.to_feature_list()
    assert len(lst) == 11
    assert lst[0] == 440.0  # distance_to_go_km
    assert lst[-1] == 0     # day_type


def test_cumulative_schedule_midnight_handling(sample_stops):
    """Verify exact cumulative scheduled minute calculation handling midnight crossings."""
    sched = compute_cumulative_schedule_minutes(sample_stops)

    # NDLS origin = 0.0 min
    assert sched["NDLS"] == 0.0
    # CNB arrival: 16:55 to 21:35 = 4h 40m = 280 min
    assert sched["CNB"] == 280.0
    # PRYJ arrival: 16:55 to 23:43 = 6h 48m = 408 min
    assert sched["PRYJ"] == 408.0
    # DDU arrival: 16:55 to 01:42 (next day) = 8h 47m = 527 min
    assert sched["DDU"] == 527.0


def test_static_features_known_inputs(sample_stops):
    """Verify known input/output calculations for distance, timetable, and halts."""
    builder = FeatureBuilder()

    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),  # Monday
        train_name="Rajdhani",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        current_delay_minutes=0.0,
        segment_progress=0.0,
        speed_kmh=110.0,
        timestamp=datetime(2026, 9, 28, 16, 55, tzinfo=timezone.utc),
        source="simulator",
    )

    # 1. Target CNB (next halt, seq 2)
    feat_cnb = builder.build(state, "CNB", route_stations=sample_stops)
    assert feat_cnb.distance_to_go_km == 440.0
    assert feat_cnb.scheduled_time_to_go_min == 280.0
    assert feat_cnb.num_intermediate_halts == 0

    # 2. Target PRYJ (seq 3)
    feat_pryj = builder.build(state, "PRYJ", route_stations=sample_stops)
    assert feat_pryj.distance_to_go_km == 635.0
    assert feat_pryj.scheduled_time_to_go_min == 408.0
    assert feat_pryj.num_intermediate_halts == 1  # CNB is intermediate

    # 3. Target DDU (seq 4, across midnight)
    feat_ddu = builder.build(state, "DDU", route_stations=sample_stops)
    assert feat_ddu.distance_to_go_km == 788.0
    assert feat_ddu.scheduled_time_to_go_min == 527.0
    assert feat_ddu.num_intermediate_halts == 2  # CNB and PRYJ are intermediate


def test_segment_progress_interpolation(sample_stops):
    """Verify accurate distance and time deduction when train is en route (progress = 0.25)."""
    builder = FeatureBuilder()

    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Rajdhani",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        current_delay_minutes=5.0,
        segment_progress=0.25,  # 25% of NDLS->CNB (440km) = 110km covered
        speed_kmh=110.0,
        timestamp=datetime(2026, 9, 28, 17, 55, tzinfo=timezone.utc),
        source="simulator",
    )

    feat_cnb = builder.build(state, "CNB", route_stations=sample_stops)
    # Remaining distance: 440 - 110 = 330 km
    assert feat_cnb.distance_to_go_km == 330.0
    # Remaining scheduled time: 280 - (0.25 * 280) = 280 - 70 = 210 min
    assert feat_cnb.scheduled_time_to_go_min == 210.0
    assert feat_cnb.current_delay_min == 5.0

    feat_pryj = builder.build(state, "PRYJ", route_stations=sample_stops)
    # Remaining distance to PRYJ: 635 - 110 = 525 km
    assert feat_pryj.distance_to_go_km == 525.0
    # Remaining scheduled time: 408 - 70 = 338 min
    assert feat_pryj.scheduled_time_to_go_min == 338.0


def test_delay_and_trend_calculation(sample_stops):
    """Verify 3-point delay slope calculation for expanding, recovering, and stationary delays."""
    builder = FeatureBuilder()

    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Rajdhani",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        current_delay_minutes=15.0,
        segment_progress=0.0,
        speed_kmh=110.0,
        timestamp=datetime.now(timezone.utc),
        source="simulator",
    )

    # Expanding delay: history [0, 5, 15] -> slope = (15 - 0) / 2 = 7.5 min
    feat1 = builder.build(
        state,
        "CNB",
        route_stations=sample_stops,
        recent_delays=[0.0, 5.0, 15.0],
    )
    assert feat1.current_delay_min == 15.0
    assert feat1.delay_trend_3pt == 7.5

    # Recovering delay: history [20, 15, 10] -> slope = (10 - 20) / 2 = -5.0 min
    state.current_delay_minutes = 10.0
    feat2 = builder.build(
        state,
        "CNB",
        route_stations=sample_stops,
        recent_delays=[20.0, 15.0, 10.0],
    )
    assert feat2.current_delay_min == 10.0
    assert feat2.delay_trend_3pt == -5.0

    # No history fallback: 0.0
    feat3 = builder.build(state, "CNB", route_stations=sample_stops, recent_delays=None)
    assert feat3.delay_trend_3pt == 0.0


def test_events_and_disruptions_features(sample_stops):
    """Verify speed restriction, congestion score, and weather flags derived from active events."""
    builder = FeatureBuilder()

    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Rajdhani",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        current_delay_minutes=0.0,
        timestamp=datetime.now(timezone.utc),
        source="simulator",
    )

    # 1. Speed Restriction event active
    ev_tsr = SimulationEvent(EventType.SPEED_RESTRICTION, delay_minutes=5.0)
    feat_tsr = builder.build(state, "CNB", route_stations=sample_stops, active_events=[ev_tsr])
    assert feat_tsr.active_speed_restriction_flag == 1
    assert feat_tsr.congestion_score_downstream == 0.0
    assert feat_tsr.weather_flag == 0

    # 2. Congestion event active with speed factor = 0.4 -> congestion score = 0.6
    ev_cong = SimulationEvent(EventType.CONGESTION, delay_minutes=5.0, metadata={"speed_factor": 0.4})
    feat_cong = builder.build(state, "CNB", route_stations=sample_stops, active_events=[ev_cong])
    assert feat_cong.congestion_score_downstream == 0.6

    # 3. Weather event active
    ev_weather = SimulationEvent(EventType.WEATHER, delay_minutes=10.0)
    feat_weather = builder.build(state, "CNB", route_stations=sample_stops, active_events=[ev_weather])
    assert feat_weather.weather_flag == 1


def test_day_type_classification(sample_stops):
    """Verify day type: 0=Weekday (Mon-Fri), 1=Weekend (Sat-Sun), 2=Holiday."""
    builder = FeatureBuilder()

    def make_state(d: date) -> TrainRunningState:
        return TrainRunningState(
            train_number="12302",
            journey_date=d,
            train_name="Rajdhani",
            status="RUNNING",
            current_station_code="NDLS",
            timestamp=datetime.now(timezone.utc),
            source="simulator",
        )

    # Monday 2026-09-28 -> Weekday (0)
    feat_mon = builder.build(make_state(date(2026, 9, 28)), "CNB", route_stations=sample_stops)
    assert feat_mon.day_type == 0

    # Saturday 2026-09-26 -> Weekend (1)
    feat_sat = builder.build(make_state(date(2026, 9, 26)), "CNB", route_stations=sample_stops)
    assert feat_sat.day_type == 1

    # Sunday 2026-09-27 -> Weekend (1)
    feat_sun = builder.build(make_state(date(2026, 9, 27)), "CNB", route_stations=sample_stops)
    assert feat_sun.day_type == 1

    # Explicit Holiday override -> Holiday (2)
    feat_hol = builder.build(
        make_state(date(2026, 9, 28)),
        "CNB",
        route_stations=sample_stops,
        is_holiday=True,
    )
    assert feat_hol.day_type == 2


def test_sqlite_integration(in_memory_db):
    """Verify building features querying route and station data directly from SQLite."""
    builder = FeatureBuilder()

    state = TrainRunningState(
        train_number="12302",  # Howrah Rajdhani in seed data
        journey_date=date(2026, 9, 28),
        train_name="Howrah Rajdhani Express",
        status="RUNNING",
        current_station_code="NDLS",
        current_station_sequence=1,
        current_delay_minutes=12.0,
        segment_progress=0.0,
        speed_kmh=110.0,
        timestamp=datetime.now(timezone.utc),
        source="simulator",
    )

    # Build features to downstream station HWH directly from DB
    features = builder.build(state, "HWH", db=in_memory_db)
    assert features.train_number == "12302"
    assert features.target_station_code == "HWH"
    # Seed route distance to HWH is 1450.0 km
    assert features.distance_to_go_km == 1450.0
    # In seed data: NDLS dep 16:55, HWH arr 09:55 next day -> 17 hours = 1020 minutes
    assert features.scheduled_time_to_go_min == 1020.0
    # Intermediate halts between NDLS (seq 1) and HWH (seq 8) = 8 - 1 - 1 = 6 halts
    assert features.num_intermediate_halts == 6
    assert features.current_delay_min == 12.0
    # Day 1 MVP defaults
    assert features.hist_avg_delay_this_section == 8.5
    assert features.hist_recovery_rate_section == 2.5


def test_invalid_target_station(sample_stops):
    """Verify that targeting an invalid or off-route station raises ValueError."""
    builder = FeatureBuilder()
    state = TrainRunningState(
        train_number="12302",
        journey_date=date(2026, 9, 28),
        train_name="Rajdhani",
        status="RUNNING",
        current_station_code="NDLS",
        timestamp=datetime.now(timezone.utc),
        source="simulator",
    )

    with pytest.raises(ValueError) as exc_info:
        builder.build(state, "MAS", route_stations=sample_stops)
    assert "not on the route" in str(exc_info.value)
