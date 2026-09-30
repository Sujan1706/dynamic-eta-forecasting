"""
Deterministic Hackathon Demo Mode Manager.
=========================================
Coordinates an isolated, 100% reproducible demonstration scenario for hackathon judges and evaluators.

Predefined Scenario:
- Train: 12302 ("Howrah Rajdhani Express")
- Route: Northern-Eastern Trunk Line (NDLS -> CNB -> PRYJ -> DDU -> GAYA -> DHN -> ASN -> HWH)
- Checkpoint: Passing Kanpur Central (CNB) bound for Prayagraj Junction (PRYJ)
- Predefined State:
  - Current Station: CNB (Sequence 2)
  - Next Station: PRYJ (195.0 km to go)
  - Segment Progress: 15%
  - Nominal Speed: 110.0 km/h
  - Headway Delay: 2.0 minutes
  - Status: RUNNING
  - Random Seed: 42 (Guarantees identical feature vectors and inference)

Supported Demo Actions:
1. Signal Halt (+15m, speed = 0 km/h, status = HALTED)
2. Congestion (+10m, speed = 49.5 km/h, status = RUNNING)
3. Speed Restriction (+8m, speed = 30.0 km/h, status = RUNNING)

Pipeline Flow:
Simulator -> State Update -> Feature Update -> Baseline Recalculation -> ML Prediction -> FastAPI -> Dashboard -> Station Board -> Passenger View
"""

import random
from datetime import datetime, date, time, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import numpy as np
from sqlalchemy.orm import Session, joinedload

from backend.database.models import Train, Route, RouteStation, Station, Journey, Event
from backend.services.schemas import TrainRunningState
from backend.simulator.engine import TrainSimulator
from backend.simulator.journey import SimulatedJourney, StationStop
from backend.simulator.events import SimulationEvent, EventType
from backend.services.baseline_eta import BaselineETAService, BaselineETAPrediction
from backend.services.multi_station_eta import MultiStationETAService, StationETAPrediction
from backend.features.feature_builder import FeatureBuilder

DEMO_SEED = 42
DEMO_TRAIN_NUMBER = "12302"
DEMO_TRAIN_NAME = "Howrah Rajdhani Express"
DEMO_CURRENT_STATION_CODE = "CNB"
DEMO_NEXT_STATION_CODE = "PRYJ"
DEMO_INITIAL_DELAY_MINUTES = 2.0
DEMO_NOMINAL_SPEED_KMH = 110.0
DEMO_SEGMENT_PROGRESS = 0.15
DEMO_DISTANCE_TO_NEXT_KM = 195.0

# 3 Predefined Demo Actions
DEMO_ACTIONS = {
    "signal_halt": {
        "action_id": "signal_halt",
        "step_number": 1,
        "name": "1. Inject Signal Halt",
        "event_type": "SIGNAL_HALT",
        "delay_minutes": 15.0,
        "severity": "HIGH",
        "location": "CNB-PRYJ Interlocking Section (Signal 42)",
        "reason": "Signal aspect held at Danger due to track circuit drop",
        "description": "Emergency stop at interlocking signal. Speed drops to 0 km/h, status becomes HALTED, and accumulated delay increases by +15 min.",
        "expected_status": "HALTED",
        "expected_speed_kmh": 0.0,
        "metadata": {
            "location": "CNB-PRYJ Interlocking Section (Signal 42)",
            "reason": "Signal aspect held at Danger due to track circuit drop",
            "source": "HACKATHON_DEMO",
            "speed_limit_kmh": 0.0,
        },
    },
    "congestion": {
        "action_id": "congestion",
        "step_number": 2,
        "name": "2. Inject Congestion",
        "event_type": "CONGESTION",
        "delay_minutes": 10.0,
        "severity": "MEDIUM",
        "location": "PRYJ Outer Junction Approach",
        "reason": "Freight bottleneck on Down main line ahead of junction",
        "description": "Corridor traffic congestion. Train resumes forward motion at 45% cruise speed (~49.5 km/h), accumulating +10 min delay.",
        "expected_status": "RUNNING",
        "expected_speed_kmh": 49.5,
        "metadata": {
            "location": "PRYJ Outer Junction Approach",
            "reason": "Freight bottleneck on Down main line ahead of junction",
            "source": "HACKATHON_DEMO",
            "speed_factor": 0.45,
        },
    },
    "speed_restriction": {
        "action_id": "speed_restriction",
        "step_number": 3,
        "name": "3. Inject Speed Restriction",
        "event_type": "SPEED_RESTRICTION",
        "delay_minutes": 8.0,
        "severity": "LOW",
        "location": "Subedarganj Caution Order Zone",
        "reason": "Ballast tamping and track maintenance work (30 km/h ceiling)",
        "description": "Engineering caution order enforced. Maximum permissible speed capped at 30 km/h, accumulating +8 min delay.",
        "expected_status": "RUNNING",
        "expected_speed_kmh": 30.0,
        "metadata": {
            "location": "Subedarganj Caution Order Zone",
            "reason": "Ballast tamping and track maintenance work (30 km/h ceiling)",
            "source": "HACKATHON_DEMO",
            "speed_limit_kmh": 30.0,
        },
    },
}


class DemoActionInfo(BaseModel):
    action_id: str = Field(description="Unique action slug: signal_halt, congestion, speed_restriction")
    step_number: int = Field(description="Recommended sequential demonstration step (1, 2, 3)")
    name: str = Field(description="Human readable display name")
    event_type: str = Field(description="Simulation event type: SIGNAL_HALT, CONGESTION, SPEED_RESTRICTION")
    delay_minutes: float = Field(description="Injected delay increment in minutes")
    severity: str = Field(description="Operational severity level: HIGH, MEDIUM, LOW")
    location: str = Field(description="Track sector / checkpoint where disruption takes place")
    reason: str = Field(description="Real-world railway operational cause")
    description: str = Field(description="Brief explanation of simulated kinematics and expected outcome")
    expected_status: str = Field(description="Expected train running state status after event injection")
    expected_speed_kmh: Optional[float] = Field(default=None, description="Expected instantaneous speed after event injection")


class DemoScenarioResponse(BaseModel):
    status: str = Field(description="Current demo scenario operational status")
    seed: int = Field(default=DEMO_SEED, description="Deterministic random seed used for reproducibility")
    train_number: str = Field(description="Demonstration train identifier")
    train_name: str = Field(description="Demonstration train name")
    train_type: str = Field(description="Train classification category")
    current_station: str = Field(description="Current station code")
    next_station: str = Field(description="Target upcoming station code")
    distance_to_next_km: float = Field(description="Distance in km to next station")
    current_delay_minutes: float = Field(description="Current accumulated delay in minutes")
    current_speed_kmh: float = Field(description="Current instantaneous speed in km/h")
    segment_progress: float = Field(description="Fractional progress across current segment (0.0 to 1.0)")
    train_status: str = Field(description="Running status: RUNNING or HALTED")
    scheduled_eta: Optional[str] = Field(default=None, description="Timetable scheduled arrival ISO string")
    baseline_eta: Optional[str] = Field(default=None, description="Baseline heuristic estimated arrival ISO string")
    ml_eta: Optional[str] = Field(default=None, description="XGBoost ML predicted arrival ISO string")
    confidence_margin_minutes: Optional[float] = Field(default=None, description="ML uncertainty margin in minutes (+/-)")
    active_events: List[Dict[str, Any]] = Field(default_factory=list, description="Currently active operational disruption events")
    delay_history: List[float] = Field(default_factory=list, description="Historical checkpoint delay progression")
    upcoming_stations_count: int = Field(description="Total remaining stations to route terminus")
    available_actions: List[DemoActionInfo] = Field(description="List of predefined 1-click demonstration actions")
    message: str = Field(description="Operator scenario instruction or reset message")


class DemoActionResult(BaseModel):
    status: str = Field(default="SUCCESS", description="Action execution status")
    action_id: str = Field(description="Executed action identifier")
    action_name: str = Field(description="Human readable action name")
    train_number: str = Field(description="Target train number")
    previous_delay_minutes: float = Field(description="Delay before action was applied")
    new_delay_minutes: float = Field(description="Delay after action was applied")
    delay_delta_minutes: float = Field(description="Net delay increment applied")
    previous_status: str = Field(description="Running status before action")
    new_status: str = Field(description="Running status after action")
    speed_kmh: float = Field(description="New instantaneous speed in km/h")
    baseline_eta: str = Field(description="Updated baseline ETA ISO string")
    ml_eta: str = Field(description="Updated XGBoost ML ETA ISO string")
    confidence_margin_minutes: float = Field(description="ML uncertainty window in minutes (+/-)")
    target_station: str = Field(description="Upcoming target station code")
    active_events_count: int = Field(description="Number of currently active disruptions")
    message: str = Field(description="Execution summary explaining model and state reaction")


class DemoScenarioManager:
    """
    Coordinates the deterministic hackathon demo scenario.
    Provides reproducible reset and 1-click execution of the 3 prepared actions.
    """

    def __init__(
        self,
        baseline_service: BaselineETAService,
        multi_station_service: MultiStationETAService,
        seed: int = DEMO_SEED,
    ):
        self.baseline_service = baseline_service
        self.multi_station_service = multi_station_service
        self.seed = seed

    def set_seed(self) -> None:
        """Sets fixed random seed for 100% deterministic reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    def reset_scenario(self, db: Session, sim: TrainSimulator) -> DemoScenarioResponse:
        """
        Resets Train 12302 in SQLite and Simulator to the exact predefined baseline state.
        Clears all injected events, restores 2.0m initial delay, 110 km/h speed, and CNB station.
        Recalculates baseline and ML ETAs deterministically with seed 42.
        """
        self.set_seed()

        train = (
            db.query(Train)
            .options(
                joinedload(Train.route)
                .joinedload(Route.route_stations)
                .joinedload(RouteStation.station)
            )
            .filter(Train.train_number == DEMO_TRAIN_NUMBER)
            .first()
        )
        if not train or not train.route or not train.route.route_stations:
            raise ValueError(f"Train '{DEMO_TRAIN_NUMBER}' or its route not found in database.")

        # Ordered stops
        sorted_rs = sorted(train.route.route_stations, key=lambda rs: rs.sequence)
        stops = [
            StationStop(
                sequence=rs.sequence,
                station_id=rs.station_id,
                station_code=rs.station.code,
                station_name=rs.station.name,
                distance_from_source_km=rs.distance_from_source_km,
                scheduled_arrival=rs.scheduled_arrival,
                scheduled_departure=rs.scheduled_departure,
                scheduled_stop_minutes=rs.scheduled_stop_minutes or 0.0,
            )
            for rs in sorted_rs
        ]

        # Target CNB station (sequence 2)
        cnb_stop = next((s for s in stops if s.station_code == DEMO_CURRENT_STATION_CODE), stops[0])

        sim_time = getattr(sim, "sim_time", None) or datetime.now(timezone.utc)
        if sim_time.tzinfo is None:
            sim_time = sim_time.replace(tzinfo=timezone.utc)
        journey_date = sim_time.date()

        # 1. Reset SQLite Journey and delete previous Events
        db_journey = (
            db.query(Journey)
            .filter(Journey.train_id == train.id, Journey.journey_date == journey_date)
            .first()
        )
        if not db_journey:
            # Fallback to any journey for this train
            db_journey = db.query(Journey).filter(Journey.train_id == train.id).first()

        if db_journey:
            # Delete any previous events
            db.query(Event).filter(Event.journey_id == db_journey.id).delete()
            db_journey.current_station_id = cnb_stop.station_id
            db_journey.current_sequence = cnb_stop.sequence
            db_journey.current_delay_minutes = DEMO_INITIAL_DELAY_MINUTES
            db_journey.status = "RUNNING"
            db_journey.journey_date = journey_date
        else:
            db_journey = Journey(
                train_id=train.id,
                journey_date=journey_date,
                current_station_id=cnb_stop.station_id,
                current_sequence=cnb_stop.sequence,
                current_delay_minutes=DEMO_INITIAL_DELAY_MINUTES,
                status="RUNNING",
            )
            db.add(db_journey)

        db.commit()
        db.refresh(db_journey)

        # 2. Reset Simulator in-memory journey
        sim_journey = SimulatedJourney(
            train_id=train.id,
            train_number=DEMO_TRAIN_NUMBER,
            train_name=train.name,
            train_type=train.train_type,
            route_id=train.route_id,
            stops=stops,
            journey_id=db_journey.id,
            journey_date=journey_date,
            current_sequence=cnb_stop.sequence,
            current_delay_minutes=DEMO_INITIAL_DELAY_MINUTES,
            status="RUNNING",
            nominal_speed_kmh=DEMO_NOMINAL_SPEED_KMH,
            sim_time=sim_time,
        )

        sim_journey.current_station_index = 1  # 0=NDLS, 1=CNB
        sim_journey.segment_progress = DEMO_SEGMENT_PROGRESS
        sim_journey.current_speed_kmh = DEMO_NOMINAL_SPEED_KMH
        sim_journey.delay_history = [0.0, 1.0, DEMO_INITIAL_DELAY_MINUTES]
        sim_journey.active_events = []
        sim_journey.status = "RUNNING"

        sim.journeys[DEMO_TRAIN_NUMBER] = sim_journey

        # 3. Compute baseline and ML ETAs
        train_state = sim_journey.to_train_running_state()

        base_pred = self.baseline_service.predict_station(
            train_state=train_state,
            target_station_code=DEMO_NEXT_STATION_CODE,
            db=db,
        )

        ml_pred = self.multi_station_service.predict_station_eta(
            train_state=train_state,
            target_station_code=DEMO_NEXT_STATION_CODE,
            db=db,
        )

        all_upcoming = self.multi_station_service.predict_upcoming_stations(
            train_state=train_state,
            db=db,
        )

        actions_list = [
            DemoActionInfo(**act)
            for act in sorted(DEMO_ACTIONS.values(), key=lambda a: a["step_number"])
        ]

        return DemoScenarioResponse(
            status="PREPARED_INITIAL_STATE",
            seed=self.seed,
            train_number=DEMO_TRAIN_NUMBER,
            train_name=train.name,
            train_type=train.train_type,
            current_station=DEMO_CURRENT_STATION_CODE,
            next_station=DEMO_NEXT_STATION_CODE,
            distance_to_next_km=DEMO_DISTANCE_TO_NEXT_KM,
            current_delay_minutes=DEMO_INITIAL_DELAY_MINUTES,
            current_speed_kmh=DEMO_NOMINAL_SPEED_KMH,
            segment_progress=DEMO_SEGMENT_PROGRESS,
            train_status="RUNNING",
            scheduled_eta=ml_pred.scheduled_eta.isoformat() if ml_pred.scheduled_eta else None,
            baseline_eta=base_pred.baseline_eta.isoformat() if base_pred.baseline_eta else None,
            ml_eta=ml_pred.predicted_eta.isoformat() if ml_pred.predicted_eta else None,
            confidence_margin_minutes=round(ml_pred.uncertainty_margin_minutes, 1),
            active_events=[],
            delay_history=list(sim_journey.delay_history),
            upcoming_stations_count=len(all_upcoming),
            available_actions=actions_list,
            message=(
                f"Deterministic Demo Scenario initialized with seed {self.seed}. "
                f"Train {DEMO_TRAIN_NUMBER} is passing {DEMO_CURRENT_STATION_CODE} bound for {DEMO_NEXT_STATION_CODE} "
                f"with initial headway delay of +{DEMO_INITIAL_DELAY_MINUTES:.1f}m at {DEMO_NOMINAL_SPEED_KMH:.0f} km/h. "
                "Ready for judge event injection."
            ),
        )

    def get_scenario(self, db: Session, sim: TrainSimulator) -> DemoScenarioResponse:
        """Returns the current operational state of the demonstration scenario."""
        journey = sim.journeys.get(DEMO_TRAIN_NUMBER)
        if not journey:
            # Auto-reset if not loaded
            return self.reset_scenario(db, sim)

        train = db.query(Train).filter(Train.train_number == DEMO_TRAIN_NUMBER).first()
        train_state = journey.to_train_running_state()
        target_station = train_state.next_station_code or DEMO_NEXT_STATION_CODE

        base_pred = self.baseline_service.predict_station(
            train_state=train_state,
            target_station_code=target_station,
            db=db,
        )

        ml_pred = self.multi_station_service.predict_station_eta(
            train_state=train_state,
            target_station_code=target_station,
            db=db,
        )

        all_upcoming = self.multi_station_service.predict_upcoming_stations(
            train_state=train_state,
            db=db,
        )

        active_events = [
            {
                "event_type": ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type),
                "delay_minutes": float(ev.delay_minutes),
                "severity": ev.severity,
                "metadata": ev.metadata,
                "timestamp": ev.created_at.isoformat() if hasattr(ev, "created_at") and ev.created_at else None,
            }
            for ev in journey.active_events
        ]

        actions_list = [
            DemoActionInfo(**act)
            for act in sorted(DEMO_ACTIONS.values(), key=lambda a: a["step_number"])
        ]

        return DemoScenarioResponse(
            status="ACTIVE",
            seed=self.seed,
            train_number=DEMO_TRAIN_NUMBER,
            train_name=train.name if train else DEMO_TRAIN_NAME,
            train_type=train.train_type if train else "RAJDHANI",
            current_station=train_state.current_station_code or DEMO_CURRENT_STATION_CODE,
            next_station=target_station,
            distance_to_next_km=train_state.next_station_distance_km or 0.0,
            current_delay_minutes=train_state.current_delay_minutes,
            current_speed_kmh=train_state.speed_kmh or 0.0,
            segment_progress=train_state.segment_progress,
            train_status=train_state.status,
            scheduled_eta=ml_pred.scheduled_eta.isoformat() if ml_pred.scheduled_eta else None,
            baseline_eta=base_pred.baseline_eta.isoformat() if base_pred.baseline_eta else None,
            ml_eta=ml_pred.predicted_eta.isoformat() if ml_pred.predicted_eta else None,
            confidence_margin_minutes=round(ml_pred.uncertainty_margin_minutes, 1),
            active_events=active_events,
            delay_history=list(journey.delay_history),
            upcoming_stations_count=len(all_upcoming),
            available_actions=actions_list,
            message=f"Train {DEMO_TRAIN_NUMBER} is in status '{train_state.status}' with +{train_state.current_delay_minutes:.1f}m delay.",
        )

    def execute_action(
        self,
        action_key: str,
        db: Session,
        sim: TrainSimulator,
    ) -> DemoActionResult:
        """
        Executes one of the 3 prepared demo actions deterministically.
        Updates simulator state, writes to SQLite, extracts features, and recalculates Baseline and ML ETAs.
        """
        self.set_seed()

        # Map action key (supports 'signal_halt', '1', 'congestion', '2', 'speed_restriction', '3')
        clean_key = str(action_key).lower().strip().replace(" ", "_")
        key_alias = {
            "1": "signal_halt",
            "step_1": "signal_halt",
            "signal": "signal_halt",
            "halt": "signal_halt",
            "2": "congestion",
            "step_2": "congestion",
            "traffic": "congestion",
            "3": "speed_restriction",
            "step_3": "speed_restriction",
            "speed": "speed_restriction",
            "restriction": "speed_restriction",
        }
        resolved_key = key_alias.get(clean_key, clean_key)

        if resolved_key not in DEMO_ACTIONS:
            valid = list(DEMO_ACTIONS.keys()) + ["1", "2", "3"]
            raise ValueError(f"Invalid demo action '{action_key}'. Must be one of: {valid}")

        action = DEMO_ACTIONS[resolved_key]

        journey = sim.journeys[DEMO_TRAIN_NUMBER]
        prev_delay = float(journey.current_delay_minutes)
        prev_status = str(journey.status)

        # When moving from Halt to Congestion or Speed Restriction, resolve conflicting halt events so motion resumes
        if resolved_key in ("congestion", "speed_restriction"):
            journey.active_events = [ev for ev in journey.active_events if not ev.is_halt]
            if journey.status == "HALTED":
                journey.status = "RUNNING"

        # 1. Inject event through TrainSimulator
        sim_event = sim.inject_event(
            train_number=DEMO_TRAIN_NUMBER,
            event_type=action["event_type"],
            delay_minutes=action["delay_minutes"],
            severity=action["severity"],
            metadata=action["metadata"],
            db=db,
        )

        # 2. Persist updated journey state to SQLite
        sim.sync_to_db(db)

        # 3. Extract updated normalized TrainRunningState
        updated_state = sim.get_state(DEMO_TRAIN_NUMBER)
        target_station = updated_state.next_station_code or DEMO_NEXT_STATION_CODE

        # 4. Feature update & Baseline Recalculation
        base_pred = self.baseline_service.predict_station(
            train_state=updated_state,
            target_station_code=target_station,
            db=db,
        )

        # 5. ML Prediction (XGBoost inference)
        ml_pred = self.multi_station_service.predict_station_eta(
            train_state=updated_state,
            target_station_code=target_station,
            db=db,
        )

        new_delay = float(updated_state.current_delay_minutes)
        delta_delay = round(new_delay - prev_delay, 1)

        return DemoActionResult(
            status="SUCCESS",
            action_id=resolved_key,
            action_name=action["name"],
            train_number=DEMO_TRAIN_NUMBER,
            previous_delay_minutes=round(prev_delay, 1),
            new_delay_minutes=round(new_delay, 1),
            delay_delta_minutes=delta_delay,
            previous_status=prev_status,
            new_status=updated_state.status,
            speed_kmh=round(updated_state.speed_kmh or 0.0, 1),
            baseline_eta=base_pred.baseline_eta.isoformat(),
            ml_eta=ml_pred.predicted_eta.isoformat(),
            confidence_margin_minutes=round(ml_pred.uncertainty_margin_minutes, 1),
            target_station=target_station,
            active_events_count=len(journey.active_events),
            message=(
                f"Action '{action['name']}' applied deterministically. "
                f"Status: {prev_status} -> {updated_state.status}, "
                f"Speed: {updated_state.speed_kmh:.1f} km/h, "
                f"Delay: +{prev_delay:.1f}m -> +{new_delay:.1f}m (+{delta_delay:.1f}m). "
                f"Baseline ETA: {base_pred.baseline_eta.strftime('%H:%M:%S')}, "
                f"ML ETA: {ml_pred.predicted_eta.strftime('%H:%M:%S')} (±{ml_pred.uncertainty_margin_minutes:.1f}m)."
            ),
        )
