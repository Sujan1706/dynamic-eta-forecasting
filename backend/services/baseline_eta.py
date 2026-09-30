from datetime import datetime, date, time, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from backend.database.models import Train, Route, RouteStation, Station
from backend.services.schemas import TrainRunningState
from backend.features.feature_builder import compute_cumulative_schedule_minutes


class BaselineETAPrediction(BaseModel):
    """
    Structured baseline ETA prediction for an upcoming station stop.
    Calculated via static timetable + current delay - scheduled recovery buffer.
    Clearly labeled as BASELINE.
    """
    train_number: str = Field(description="Unique train identifier")
    target_station: str = Field(description="Target station code")
    target_station_name: Optional[str] = Field(default=None, description="Target station name")
    target_sequence: int = Field(description="Sequence number of target station along route")
    scheduled_eta: datetime = Field(description="Timetable scheduled arrival datetime")
    baseline_eta: datetime = Field(description="Baseline predicted arrival datetime")
    current_delay: float = Field(description="Current train delay at time of prediction in minutes")
    recovery_applied: float = Field(description="Scheduled recovery buffer applied in minutes")
    predicted_delay: float = Field(description="Expected remaining delay at target station in minutes")
    distance_to_go_km: float = Field(description="Remaining route distance to target station in km")
    method: str = Field(default="BASELINE", description="Forecasting method: strictly 'BASELINE'")
    generated_timestamp: datetime = Field(description="Timestamp when prediction was generated")

    @property
    def target_station_code(self) -> str:
        """Alias for target_station code."""
        return self.target_station

    @property
    def scheduled_arrival(self) -> datetime:
        """Alias for scheduled_eta."""
        return self.scheduled_eta


DEFAULT_RECOVERY_RATE_PER_100KM = 2.5  # Standard IR recovery margin: 2.5 min per 100km
DEFAULT_MAX_SPEED_KMH = 130.0          # Physical feasibility guard upper bound


class BaselineETAService:
    """
    Zero-dependency Baseline ETA forecasting service.
    Implements the PRD baseline heuristic:
        ETA = ScheduledArrival + CurrentDelay - ScheduledRecoveryBuffer
    Always returns a valid, physically bounded prediction even when ML models are unavailable.
    """

    def __init__(
        self,
        recovery_rate_per_100km: float = DEFAULT_RECOVERY_RATE_PER_100KM,
        max_permissible_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
    ):
        self.recovery_rate_per_100km = max(0.0, float(recovery_rate_per_100km))
        self.max_permissible_speed_kmh = max(10.0, float(max_permissible_speed_kmh))

    def predict_next_station(
        self,
        train_state: TrainRunningState,
        route_stations: Optional[List[Any]] = None,
        db: Optional[Session] = None,
    ) -> Optional[BaselineETAPrediction]:
        """
        Calculates baseline ETA for the immediate next scheduled station.
        Returns None if train has completed its journey or has no next station.
        """
        if not train_state.next_station_code or train_state.status.upper() == "COMPLETED":
            return None

        return self.predict_station(
            train_state=train_state,
            target_station_code=train_state.next_station_code,
            route_stations=route_stations,
            db=db,
        )

    def predict_upcoming_stations(
        self,
        train_state: TrainRunningState,
        route_stations: Optional[List[Any]] = None,
        db: Optional[Session] = None,
    ) -> List[BaselineETAPrediction]:
        """
        Calculates baseline ETA for all remaining upcoming stations along the route until terminus.
        """
        if train_state.status.upper() == "COMPLETED":
            return []

        stops = self._resolve_route_stops(train_state.train_number, route_stations, db)
        if not stops:
            return []

        # Find current sequence index
        curr_code = (train_state.current_station_code or "").upper().strip()
        curr_seq = train_state.current_station_sequence or 1

        curr_idx = None
        for idx, stop in enumerate(stops):
            code = getattr(stop, "station_code", getattr(stop, "code", "")).upper().strip()
            seq = getattr(stop, "sequence", getattr(stop, "seq", idx + 1))
            if code == curr_code or seq == curr_seq:
                curr_idx = idx
                break

        if curr_idx is None:
            curr_idx = 0

        # All stops strictly after current station index
        upcoming_stops = stops[curr_idx + 1:]
        predictions = []

        for stop in upcoming_stops:
            code = getattr(stop, "station_code", getattr(stop, "code", "")).upper().strip()
            try:
                pred = self.predict_station(
                    train_state=train_state,
                    target_station_code=code,
                    route_stations=stops,
                )
                predictions.append(pred)
            except Exception:
                continue

        return predictions

    def predict_station(
        self,
        train_state: TrainRunningState,
        target_station_code: str,
        route_stations: Optional[List[Any]] = None,
        db: Optional[Session] = None,
    ) -> BaselineETAPrediction:
        """
        Calculates the baseline ETA prediction for a specific target station.
        Handles overnight crossings, current accumulated delays, and physical feasibility bounds.
        """
        target_code = target_station_code.upper().strip()
        stops = self._resolve_route_stops(train_state.train_number, route_stations, db)
        if not stops:
            raise ValueError(f"No route stations found for train '{train_state.train_number}'.")

        # Map stations
        stop_map: Dict[str, Tuple[int, Any]] = {}
        for idx, s in enumerate(stops):
            code = getattr(s, "station_code", getattr(s, "code", "")).upper().strip()
            stop_map[code] = (idx, s)

        if target_code not in stop_map:
            raise ValueError(
                f"Target station '{target_code}' is not on the route for train '{train_state.train_number}'. "
                f"Available route stations: {list(stop_map.keys())}"
            )

        target_idx, target_stop = stop_map[target_code]
        target_seq = int(getattr(target_stop, "sequence", getattr(target_stop, "seq", target_idx + 1)))
        target_name = getattr(target_stop, "station_name", getattr(target_stop, "name", target_code))

        # Current station position
        curr_code = (train_state.current_station_code or "").upper().strip()
        if curr_code in stop_map:
            curr_idx, curr_stop = stop_map[curr_code]
        else:
            curr_seq = train_state.current_station_sequence or 1
            curr_idx = max(0, min(len(stops) - 1, curr_seq - 1))
            curr_stop = stops[curr_idx]

        curr_dist = float(getattr(curr_stop, "distance_from_source_km", getattr(curr_stop, "dist_km", 0.0)))
        target_dist = float(getattr(target_stop, "distance_from_source_km", getattr(target_stop, "dist_km", 0.0)))

        # Segment distance and interpolation
        progress = max(0.0, min(1.0, float(train_state.segment_progress or 0.0)))
        is_live = str(train_state.source or "").lower() in ("external_api", "live_api", "live")
        now_dt = train_state.timestamp
        ref_tz = now_dt.tzinfo if now_dt.tzinfo is not None else timezone.utc
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=ref_tz)

        if curr_idx < len(stops) - 1:
            next_stop = stops[curr_idx + 1]
            next_dist = float(getattr(next_stop, "distance_from_source_km", getattr(next_stop, "dist_km", curr_dist)))
            seg_dist = max(0.0, next_dist - curr_dist)
            if is_live and train_state.next_station_distance_km is not None and seg_dist > 0:
                live_dist_remaining = max(0.0, float(train_state.next_station_distance_km))
                dist_covered_on_seg = max(0.0, seg_dist - min(live_dist_remaining, seg_dist))
                eff_prog = dist_covered_on_seg / seg_dist if seg_dist > 0 else 0.0
            else:
                dist_covered_on_seg = progress * seg_dist
                eff_prog = progress
        else:
            dist_covered_on_seg = 0.0
            eff_prog = progress
            seg_dist = 0.0

        current_position_km = curr_dist + dist_covered_on_seg
        distance_to_go_km = max(0.0, round(target_dist - current_position_km, 2))

        # Timetable scheduled arrival calculation (handling overnight crossings)
        cumulative_times = compute_cumulative_schedule_minutes(stops)
        target_sched_cumulative_min = cumulative_times.get(target_code, 0.0)
        curr_sched_time = cumulative_times.get(curr_code, 0.0)

        if curr_idx < len(stops) - 1:
            next_code_clean = getattr(next_stop, "station_code", getattr(next_stop, "code", ""))
            next_sched_time = cumulative_times.get(next_code_clean, curr_sched_time)
            seg_time = max(0.0, next_sched_time - curr_sched_time)
            time_covered_on_seg = eff_prog * seg_time
        else:
            time_covered_on_seg = 0.0

        curr_position_time = curr_sched_time + time_covered_on_seg
        sched_time_to_go_min = max(0.0, round(target_sched_cumulative_min - curr_position_time, 2))

        # Base origin datetime
        origin_stop = stops[0]
        dep_time_str = getattr(origin_stop, "scheduled_departure", None) or getattr(origin_stop, "scheduled_arrival", "00:00")
        try:
            dep_parts = dep_time_str.strip().split(":")
            origin_dep_time = time(int(dep_parts[0]), int(dep_parts[1]))
        except Exception:
            origin_dep_time = time(0, 0)

        journey_d = train_state.journey_date
        if isinstance(journey_d, datetime):
            journey_d = journey_d.date()
        elif not isinstance(journey_d, date):
            journey_d = date.today()

        if is_live:
            scheduled_arrival_dt = now_dt + timedelta(minutes=sched_time_to_go_min)
        else:
            origin_base_dt = datetime.combine(journey_d, origin_dep_time).replace(tzinfo=ref_tz)
            scheduled_arrival_dt = origin_base_dt + timedelta(minutes=target_sched_cumulative_min)

        # Calculate Recovery Buffer
        # Buffer proportional to remaining distance: (distance_to_go / 100) * recovery_rate
        raw_recovery_buffer = (distance_to_go_km / 100.0) * self.recovery_rate_per_100km

        # Train delay
        current_delay = round(float(train_state.current_delay_minutes or 0.0), 2)

        # Recovery can only recover accumulated delay (cannot cause arrival before scheduled timetable)
        if current_delay > 0.0:
            recovery_applied = round(min(current_delay, raw_recovery_buffer), 2)
            predicted_delay = round(current_delay - recovery_applied, 2)
        else:
            recovery_applied = 0.0
            predicted_delay = current_delay

        # Baseline ETA formula:
        # For live telemetry: current timestamp + scheduled_time_to_go_min + predicted_delay
        # For simulator: ScheduledArrival + CurrentDelay - RecoveryApplied
        if is_live:
            baseline_eta = now_dt + timedelta(minutes=sched_time_to_go_min + predicted_delay)
        else:
            baseline_eta = scheduled_arrival_dt + timedelta(minutes=predicted_delay)

        # Physical Feasibility Guard: ETA >= CurrentTime + (Distance / MaxSpeed)
        min_travel_minutes = (distance_to_go_km / self.max_permissible_speed_kmh) * 60.0
        earliest_feasible_eta = now_dt + timedelta(minutes=min_travel_minutes)

        if baseline_eta < earliest_feasible_eta:
            baseline_eta = earliest_feasible_eta
            predicted_delay = max(
                0.0,
                round((baseline_eta - scheduled_arrival_dt).total_seconds() / 60.0, 2),
            )

        return BaselineETAPrediction(
            train_number=train_state.train_number,
            target_station=target_code,
            target_station_name=target_name,
            target_sequence=target_seq,
            scheduled_eta=scheduled_arrival_dt,
            baseline_eta=baseline_eta,
            current_delay=current_delay,
            recovery_applied=recovery_applied,
            predicted_delay=predicted_delay,
            distance_to_go_km=distance_to_go_km,
            method="BASELINE",
            generated_timestamp=datetime.now(ref_tz),
        )

    def _resolve_route_stops(
        self,
        train_number: str,
        route_stations: Optional[List[Any]],
        db: Optional[Session],
    ) -> List[Any]:
        """Resolves ordered route stops from argument or SQLite query."""
        if route_stations:
            return sorted(route_stations, key=lambda s: getattr(s, "sequence", getattr(s, "seq", 0)))

        if db:
            train = (
                db.query(Train)
                .options(
                    joinedload(Train.route)
                    .joinedload(Route.route_stations)
                    .joinedload(RouteStation.station)
                )
                .filter(Train.train_number == str(train_number).strip())
                .first()
            )
            if train and train.route and train.route.route_stations:
                stops = sorted(train.route.route_stations, key=lambda rs: rs.sequence)
                for s in stops:
                    if hasattr(s, "station") and s.station:
                        setattr(s, "station_code", s.station.code)
                        setattr(s, "station_name", s.station.name)
                return stops

        return []
