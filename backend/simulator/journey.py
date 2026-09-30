from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional
import math

from backend.services.schemas import TrainRunningState
from backend.simulator.events import SimulationEvent, EventType


@dataclass
class StationStop:
    """Represents a scheduled station stop along a train's route."""
    sequence: int
    station_id: int
    station_code: str
    station_name: str
    distance_from_source_km: float
    scheduled_arrival: Optional[str] = None
    scheduled_departure: Optional[str] = None
    scheduled_stop_minutes: float = 0.0


DEFAULT_NOMINAL_SPEEDS = {
    "RAJDHANI": 110.0,
    "SHATABDI": 100.0,
    "SUPERFAST": 90.0,
    "EXPRESS": 75.0,
}


class SimulatedJourney:
    """
    Maintains and advances the dynamic running state of a train journey along its route.
    Tracks station transitions, segment progress, instantaneous speed, delays, and ETA.
    """

    def __init__(
        self,
        train_id: int,
        train_number: str,
        train_name: str,
        train_type: str,
        route_id: int,
        stops: List[StationStop],
        journey_id: Optional[int] = None,
        journey_date: Optional[date] = None,
        current_sequence: int = 1,
        current_delay_minutes: float = 0.0,
        status: str = "RUNNING",
        nominal_speed_kmh: Optional[float] = None,
        sim_time: Optional[datetime] = None,
    ):
        if not stops:
            raise ValueError("A journey must have at least one station stop.")

        self.journey_id = journey_id
        self.train_id = train_id
        self.train_number = str(train_number).strip()
        self.train_name = train_name.strip()
        self.train_type = train_type.upper().strip()
        self.route_id = route_id
        self.stops = sorted(stops, key=lambda s: s.sequence)
        self.journey_date = journey_date or date.today()

        # Speeds
        if nominal_speed_kmh is not None and nominal_speed_kmh > 0:
            self.nominal_speed_kmh = float(nominal_speed_kmh)
        else:
            self.nominal_speed_kmh = DEFAULT_NOMINAL_SPEEDS.get(self.train_type, 80.0)

        self.current_speed_kmh = self.nominal_speed_kmh

        # Position tracking
        self.current_delay_minutes = max(0.0, float(current_delay_minutes))
        self.status = status.upper().strip()

        # Map current sequence to station index
        seq_indices = {s.sequence: idx for idx, s in enumerate(self.stops)}
        self.current_station_index = seq_indices.get(current_sequence, 0)
        self.segment_progress = 0.0
        self.dwell_time_remaining_seconds = 0.0

        # Disruption events
        self.active_events: List[SimulationEvent] = []
        self.delay_history: List[float] = [self.current_delay_minutes]

        # Simulation clock
        self.sim_time = sim_time or datetime.now(timezone.utc)

        # Check if already at or past destination
        if self.current_station_index >= len(self.stops) - 1:
            self.status = "COMPLETED"
            self.segment_progress = 1.0
            self.current_speed_kmh = 0.0

    @property
    def current_sequence(self) -> int:
        return self.stops[self.current_station_index].sequence

    @property
    def current_station(self) -> StationStop:
        return self.stops[self.current_station_index]

    @property
    def previous_station(self) -> Optional[StationStop]:
        if self.current_station_index > 0:
            return self.stops[self.current_station_index - 1]
        return None

    @property
    def next_station(self) -> Optional[StationStop]:
        if self.current_station_index < len(self.stops) - 1:
            return self.stops[self.current_station_index + 1]
        return None

    @property
    def next_station_distance_km(self) -> Optional[float]:
        if self.status == "COMPLETED" or not self.next_station:
            return 0.0
        curr_dist = self.current_station.distance_from_source_km
        next_dist = self.next_station.distance_from_source_km
        segment_total_km = max(0.1, next_dist - curr_dist)
        remaining_km = max(0.0, (1.0 - self.segment_progress) * segment_total_km)
        return round(remaining_km, 2)

    @property
    def estimated_next_station_arrival(self) -> Optional[datetime]:
        """Calculates estimated arrival time at the next scheduled halt."""
        if self.status == "COMPLETED" or not self.next_station:
            return None

        rem_km = self.next_station_distance_km or 0.0

        # Travel speed estimate
        effective_speed = self.current_speed_kmh if self.current_speed_kmh > 5.0 else self.nominal_speed_kmh
        travel_hours = rem_km / effective_speed
        travel_minutes = travel_hours * 60.0

        # Remaining halt/dwell buffers
        buffer_minutes = self.dwell_time_remaining_seconds / 60.0
        for ev in self.active_events:
            if ev.is_halt:
                buffer_minutes += ev.remaining_duration_seconds / 60.0

        total_est_minutes = travel_minutes + buffer_minutes
        return self.sim_time + timedelta(minutes=total_est_minutes)

    def inject_event(self, event: SimulationEvent) -> None:
        """
        Injects a disruption event, deterministically modifying running state and delay.
        """
        self.active_events.append(event)
        self.current_delay_minutes += event.delay_minutes
        self.delay_history.append(round(self.current_delay_minutes, 1))

        if event.is_halt:
            self.current_speed_kmh = 0.0
            self.status = "HALTED"
        else:
            self._update_speed()

    def _update_speed(self) -> None:
        """Re-evaluates instantaneous speed against all currently active events."""
        if any(ev.is_halt for ev in self.active_events):
            self.current_speed_kmh = 0.0
            self.status = "HALTED"
            return

        if self.dwell_time_remaining_seconds > 0.0:
            self.current_speed_kmh = 0.0
            return

        if self.status == "COMPLETED":
            self.current_speed_kmh = 0.0
            return

        # Restore status to RUNNING if no halts
        if self.status == "HALTED":
            self.status = "RUNNING"

        target_speed = self.nominal_speed_kmh
        for ev in self.active_events:
            target_speed = min(target_speed, ev.calculate_effective_speed(self.nominal_speed_kmh))

        self.current_speed_kmh = max(0.0, target_speed)

    def advance(self, dt_seconds: float) -> None:
        """
        Advances the journey state by dt_seconds (simulated seconds).
        Moves the train along its segment, handles dwell times and station transitions.
        """
        if self.status == "COMPLETED" or dt_seconds <= 0.0:
            self.sim_time += timedelta(seconds=dt_seconds)
            return

        # 1. Advance simulation clock
        self.sim_time += timedelta(seconds=dt_seconds)

        # 2. Ensure instantaneous speed reflects active events for this tick
        self._update_speed()

        time_available_for_travel = dt_seconds

        # 3. Handle station dwell time if stopped at a station
        if self.dwell_time_remaining_seconds > 0.0:
            dwell_spent = min(time_available_for_travel, self.dwell_time_remaining_seconds)
            self.dwell_time_remaining_seconds -= dwell_spent
            time_available_for_travel -= dwell_spent
            self.current_speed_kmh = 0.0

            if self.dwell_time_remaining_seconds <= 0.0 and self.status != "HALTED":
                # Resumed departure from station
                self._update_speed()

        # 4. Advance along the current segment (if travel time available and not halted)
        if time_available_for_travel > 0.0 and self.current_speed_kmh > 0.0:
            curr_stop = self.current_station
            next_stop = self.next_station

            if not next_stop:
                self.status = "COMPLETED"
                self.segment_progress = 1.0
                self.current_speed_kmh = 0.0
            else:
                segment_length_km = max(0.1, next_stop.distance_from_source_km - curr_stop.distance_from_source_km)

                # Actual distance covered in time_available_for_travel
                distance_covered_km = self.current_speed_kmh * (time_available_for_travel / 3600.0)

                # Track schedule delay if moving below nominal cruise speed
                if self.current_speed_kmh < self.nominal_speed_kmh:
                    nominal_dist = self.nominal_speed_kmh * (time_available_for_travel / 3600.0)
                    lost_distance = max(0.0, nominal_dist - distance_covered_km)
                    lost_minutes = (lost_distance / self.nominal_speed_kmh) * 60.0
                    self.current_delay_minutes += lost_minutes

                current_km_on_segment = self.segment_progress * segment_length_km
                new_km_on_segment = current_km_on_segment + distance_covered_km

                if new_km_on_segment < segment_length_km:
                    # Still moving along the segment
                    self.segment_progress = min(0.9999, new_km_on_segment / segment_length_km)
                    self.status = "RUNNING"
                else:
                    # Station Transition: Train arrives at next station!
                    self.current_station_index += 1
                    arrived_station = self.stops[self.current_station_index]

                    if self.current_station_index >= len(self.stops) - 1:
                        # Reached final destination!
                        self.status = "COMPLETED"
                        self.segment_progress = 1.0
                        self.current_speed_kmh = 0.0
                        self.dwell_time_remaining_seconds = 0.0
                    else:
                        # Arrived at intermediate station stop
                        self.segment_progress = 0.0
                        self.dwell_time_remaining_seconds = arrived_station.scheduled_stop_minutes * 60.0
                        if self.dwell_time_remaining_seconds > 0.0:
                            self.current_speed_kmh = 0.0

        # 5. Decrement active event durations and expire completed events
        surviving_events = []
        for ev in self.active_events:
            ev.remaining_duration_seconds -= dt_seconds
            if ev.remaining_duration_seconds > 0.0:
                surviving_events.append(ev)
        self.active_events = surviving_events

        # 6. Recalculate speed for the end-of-tick state
        self._update_speed()

    def to_train_running_state(self) -> TrainRunningState:
        """
        Converts the dynamic journey state into the standard normalized TrainRunningState schema.
        Ensures 100% interoperability with external API live feeds.
        """
        curr_stop = self.current_station
        prev_stop = self.previous_station
        next_stop = self.next_station

        prev_code = prev_stop.station_code if prev_stop else None
        next_code = next_stop.station_code if next_stop else None
        next_dist = self.next_station_distance_km if next_stop else None

        return TrainRunningState(
            train_number=self.train_number,
            journey_date=self.journey_date,
            train_name=self.train_name,
            status=self.status,
            current_station_code=curr_stop.station_code,
            current_station_sequence=curr_stop.sequence,
            current_delay_minutes=round(self.current_delay_minutes, 2),
            previous_station_code=prev_code,
            next_station_code=next_code,
            next_station_distance_km=next_dist,
            segment_progress=round(min(max(self.segment_progress, 0.0), 1.0), 4),
            speed_kmh=round(self.current_speed_kmh, 1),
            timestamp=self.sim_time,
            source="simulator",
        )
