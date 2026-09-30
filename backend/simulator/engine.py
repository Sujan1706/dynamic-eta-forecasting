from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session, joinedload

from backend.database.models import Train, Route, RouteStation, Station, Journey, Event
from backend.services.schemas import TrainRunningState
from backend.simulator.events import SimulationEvent, EventType
from backend.simulator.journey import SimulatedJourney, StationStop


class SimulatorStoppedError(Exception):
    """Raised when an operation is requested on a stopped or uninitialized simulator."""
    pass


class TrainSimulator:
    """
    Synthetic train-running simulation engine.
    Orchestrates multiple simulated train journeys along their designated routes,
    supports configurable simulation speeds, discrete tick advancement,
    controlled disruption event injection, and emits normalized TrainRunningState telemetry.
    """

    def __init__(
        self,
        simulation_speed: float = 1.0,
        start_time: Optional[datetime] = None,
        db: Optional[Session] = None,
    ):
        if simulation_speed <= 0:
            raise ValueError("simulation_speed must be positive.")
        self.simulation_speed = float(simulation_speed)
        self.sim_time = start_time or datetime.now(timezone.utc)
        self.db = db
        self.journeys: Dict[str, SimulatedJourney] = {}
        self.is_running: bool = True

    def stop(self) -> None:
        """Stops/pauses the simulation engine."""
        self.is_running = False

    def start(self) -> None:
        """Resumes/starts the simulation engine."""
        self.is_running = True

    def set_simulation_speed(self, speed: float) -> None:
        """Sets the simulation speed multiplier (e.g., 1.0 = real-time, 10.0 = 10x, 60.0 = 1 min/sec)."""
        if speed <= 0:
            raise ValueError("Simulation speed must be greater than zero.")
        self.simulation_speed = float(speed)

    def register_journey(self, journey: SimulatedJourney) -> None:
        """Registers a simulated train journey into the engine."""
        self.journeys[journey.train_number] = journey

    def get_journey(self, train_number: str) -> Optional[SimulatedJourney]:
        """Retrieves a simulated journey by train number."""
        return self.journeys.get(str(train_number).strip())

    def load_from_db(
        self,
        db: Optional[Session] = None,
        train_numbers: Optional[List[str]] = None,
        journey_date: Optional[date] = None,
    ) -> int:
        """
        Loads trains, routes, and route stations from SQLite.
        Links or creates Journey records in the database.
        Returns the number of journeys loaded.
        """
        session = db or self.db
        if not session:
            raise ValueError("Database session required to load trains and routes from SQLite.")

        target_date = journey_date or (self.sim_time.date() if hasattr(self, "sim_time") and self.sim_time else datetime.now(timezone.utc).date())

        query = session.query(Train).options(
            joinedload(Train.route)
            .joinedload(Route.route_stations)
            .joinedload(RouteStation.station)
        )
        if train_numbers:
            cleaned = [str(n).strip() for n in train_numbers]
            query = query.filter(Train.train_number.in_(cleaned))

        trains = query.all()
        loaded_count = 0

        for train in trains:
            if not train.route or not train.route.route_stations:
                continue

            # Build ordered list of station stops
            sorted_route_stations = sorted(train.route.route_stations, key=lambda rs: rs.sequence)
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
                for rs in sorted_route_stations
            ]

            # Find or create Journey in SQLite
            db_journey = (
                session.query(Journey)
                .filter(Journey.train_id == train.id, Journey.journey_date == target_date)
                .first()
            )
            if not db_journey:
                first_station_id = stops[0].station_id if stops else None
                db_journey = Journey(
                    train_id=train.id,
                    journey_date=target_date,
                    current_station_id=first_station_id,
                    current_sequence=1,
                    current_delay_minutes=0.0,
                    status="RUNNING",
                )
                session.add(db_journey)
                session.commit()
                session.refresh(db_journey)

            sim_journey = SimulatedJourney(
                train_id=train.id,
                train_number=train.train_number,
                train_name=train.name,
                train_type=train.train_type,
                route_id=train.route_id,
                stops=stops,
                journey_id=db_journey.id,
                journey_date=target_date,
                current_sequence=db_journey.current_sequence,
                current_delay_minutes=db_journey.current_delay_minutes,
                status=db_journey.status,
                sim_time=self.sim_time,
            )

            self.register_journey(sim_journey)
            loaded_count += 1

        return loaded_count

    def tick(
        self,
        step_seconds: float = 60.0,
        auto_sync_db: bool = False,
        db: Optional[Session] = None,
    ) -> Dict[str, TrainRunningState]:
        """
        Advances the simulation by one discrete tick.
        Advances all registered journeys by (step_seconds * simulation_speed) simulated seconds.
        Returns a mapping of train_number -> normalized TrainRunningState.
        """
        if not self.is_running:
            raise SimulatorStoppedError("Train simulator is currently stopped.")

        effective_seconds = step_seconds * self.simulation_speed
        self.sim_time += timedelta(seconds=effective_seconds)

        states: Dict[str, TrainRunningState] = {}
        for number, journey in self.journeys.items():
            journey.advance(effective_seconds)
            states[number] = journey.to_train_running_state()

        if auto_sync_db:
            self.sync_to_db(db)

        return states

    def inject_event(
        self,
        train_number: str,
        event_type: str | EventType,
        delay_minutes: float = 0.0,
        severity: str = "MEDIUM",
        metadata: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
        db: Optional[Session] = None,
    ) -> SimulationEvent:
        """
        Injects an operational disruption event into the specified train's journey.
        Modifies delay and running state reproducibly, and optionally logs to SQLite.
        """
        if not self.is_running:
            raise SimulatorStoppedError("Cannot inject event: Train simulator is currently stopped.")

        journey = self.get_journey(train_number)
        if not journey:
            raise KeyError(f"Train '{train_number}' is not currently managed by the simulator.")

        event = SimulationEvent(
            event_type=event_type,
            delay_minutes=delay_minutes,
            severity=severity,
            metadata=metadata or {},
            remaining_duration_seconds=duration_seconds or 0.0,
            created_at=self.sim_time,
        )

        journey.inject_event(event)

        # Persist event in SQLite if session available
        session = db or self.db
        if session and journey.journey_id:
            db_event = Event(
                journey_id=journey.journey_id,
                timestamp=self.sim_time,
                event_type=event.event_type.value,
                delay_minutes=event.delay_minutes,
                severity=event.severity,
                event_metadata=event.metadata,
            )
            session.add(db_event)
            session.commit()

        return event

    def get_state(self, train_number: str) -> TrainRunningState:
        """Returns the current normalized TrainRunningState for a specific train."""
        if not self.is_running:
            raise SimulatorStoppedError("Train simulator is currently stopped.")
        journey = self.get_journey(train_number)
        if not journey:
            raise KeyError(f"Train '{train_number}' is not found in the simulator.")
        return journey.to_train_running_state()

    def get_all_states(self) -> List[TrainRunningState]:
        """Returns normalized TrainRunningState for all managed trains."""
        return [journey.to_train_running_state() for journey in self.journeys.values()]

    def sync_to_db(self, db: Optional[Session] = None) -> None:
        """Synchronizes current in-memory journey states back to SQLite."""
        session = db or self.db
        if not session:
            return

        for journey in self.journeys.values():
            if not journey.journey_id:
                continue

            curr_stop = journey.current_station
            db_journey = session.query(Journey).filter(Journey.id == journey.journey_id).first()
            if db_journey:
                db_journey.current_station_id = curr_stop.station_id
                db_journey.current_sequence = curr_stop.sequence
                db_journey.current_delay_minutes = journey.current_delay_minutes
                db_journey.status = journey.status

        session.commit()


def main():
    """Quick CLI runner for local inspection and demo of the simulator."""
    import argparse
    from backend.database.connection import SessionLocal
    from backend.database.seed import seed_data

    parser = argparse.ArgumentParser(description="Synthetic Train-Running Simulator CLI")
    parser.add_argument("--speed", type=float, default=10.0, help="Simulation speed multiplier (default: 10x)")
    parser.add_argument("--ticks", type=int, default=5, help="Number of ticks to run (default: 5)")
    parser.add_argument("--step-seconds", type=float, default=60.0, help="Seconds per tick (default: 60s)")
    args = parser.parse_args()

    session = SessionLocal()
    try:
        seed_data(session)
        sim = TrainSimulator(simulation_speed=args.speed, db=session)
        loaded = sim.load_from_db()
        print(f"\n[Simulator] Loaded {loaded} trains from SQLite.")
        print(f"[Simulator] Running {args.ticks} ticks at {args.speed}x speed...\n")

        for tick_idx in range(1, args.ticks + 1):
            states = sim.tick(step_seconds=args.step_seconds, auto_sync_db=True)
            print(f"--- Tick {tick_idx} (Sim Time: {sim.sim_time.strftime('%H:%M:%S')}) ---")
            for t_num, state in states.items():
                print(
                    f"Train {state.train_number} ({state.train_name}) | "
                    f"Loc: {state.current_station_code} (Seq {state.current_station_sequence}) -> "
                    f"Next: {state.next_station_code or 'TERMINUS'} ({state.next_station_distance_km} km) | "
                    f"Prog: {state.segment_progress:.1%} | Spd: {state.speed_kmh} km/h | "
                    f"Delay: +{state.current_delay_minutes:.1f}m | Status: {state.status}"
                )

        # Demo event injection
        train_num = list(sim.journeys.keys())[0]
        print(f"\n[Simulator] Injecting SIGNAL_HALT (+15 min) on Train {train_num}...")
        sim.inject_event(train_num, "SIGNAL_HALT", delay_minutes=15.0, severity="HIGH")
        state_after = sim.get_state(train_num)
        state_after.display()

    finally:
        session.close()


if __name__ == "__main__":
    main()
