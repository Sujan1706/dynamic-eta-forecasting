"""
Multi-Station ETA Prediction Service.
====================================
Predicts dynamic arrival ETAs for all upcoming stations along a train's route
by chaining segment-by-segment XGBoost regression inferences.

Architecture:
1. Predicts transit minutes for the current active segment (checkpoint -> next station)
   using the trained XGBoost model.
2. Forward-propagates the simulated train state (arrival timestamp, accumulated delay,
   and scheduled dwell buffer) to subsequent route segments.
3. Chains segment transit times: Segment 1 + Segment 2 + Segment 3 + ...
4. For each upcoming station, provides:
   - predicted_eta (datetime)
   - baseline_eta (datetime from Day 1 BaselineETAService)
   - confidence_lower_bound (datetime)
   - confidence_upper_bound (datetime)
   - segment_predictions (granular segment-by-segment breakdown)

Confidence Range:
Implements a transparent heuristic uncertainty approach for the MVP:
  margin = base_uncertainty * sqrt(segments_ahead)
Uncertainty widens as the forecast horizon extends into future segments.
DISCLAIMER: This confidence interval is an uncalibrated MVP heuristic
and is NOT a statistically validated quantile or conformal prediction interval.
"""

from datetime import datetime, date, time, timedelta, timezone
from typing import List, Dict, Optional, Any, Tuple
import math
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session, joinedload
from backend.database.models import Train, Route, RouteStation
from backend.services.schemas import TrainRunningState
from backend.services.baseline_eta import BaselineETAService, BaselineETAPrediction
from backend.features.feature_builder import FeatureBuilder, compute_cumulative_schedule_minutes
from backend.ml.predictor import ETAPredictor


BASE_SEGMENT_UNCERTAINTY_MINUTES = 5.0  # Heuristic base margin derived from single-segment test MAE (~5.2 min)
MAX_PERMISSIBLE_SPEED_KMH = 130.0       # Physical kinematic ceiling guard


class SegmentPrediction(BaseModel):
    """Granular prediction for a single inter-station route segment."""
    segment_index: int = Field(description="1-based index of this segment along the upcoming path")
    segment_order: Optional[int] = Field(default=None, description="Alias for segment_index for frontend compatibility")
    from_station_code: str = Field(description="Departure station code for this segment")
    to_station_code: str = Field(description="Arrival station code for this segment")
    to_station_name: Optional[str] = Field(default=None, description="Arrival station name")
    segment_distance_km: float = Field(description="Distance of this segment in km")
    distance_km: Optional[float] = Field(default=None, description="Alias for segment_distance_km for frontend compatibility")
    scheduled_transit_minutes: float = Field(description="Timetable scheduled travel time for this segment")
    scheduled_minutes: Optional[float] = Field(default=None, description="Alias for scheduled_transit_minutes for frontend compatibility")
    baseline_transit_minutes: Optional[float] = Field(default=None, description="Baseline heuristic transit time in minutes")
    baseline_minutes: Optional[float] = Field(default=None, description="Alias for baseline_transit_minutes for frontend compatibility")
    predicted_transit_minutes: float = Field(description="XGBoost predicted travel minutes for this segment")
    predicted_minutes: Optional[float] = Field(default=None, description="Alias for predicted_transit_minutes for frontend compatibility")
    scheduled_dwell_minutes: float = Field(description="Scheduled stop/dwell time at arrival station in minutes")
    cumulative_transit_minutes: float = Field(description="Cumulative travel minutes from observation checkpoint")
    predicted_arrival_time: datetime = Field(description="Predicted arrival datetime at to_station")
    predicted_departure_time: datetime = Field(description="Predicted departure datetime from to_station")

    @model_validator(mode="before")
    @classmethod
    def populate_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Distance aliases
            if "segment_distance_km" in data and "distance_km" not in data:
                data["distance_km"] = data["segment_distance_km"]
            elif "distance_km" in data and "segment_distance_km" not in data:
                data["segment_distance_km"] = data["distance_km"]

            # Order / index aliases
            if "segment_index" in data and "segment_order" not in data:
                data["segment_order"] = data["segment_index"]
            elif "segment_order" in data and "segment_index" not in data:
                data["segment_index"] = data["segment_order"]

            # Scheduled transit minutes aliases
            if "scheduled_transit_minutes" in data and "scheduled_minutes" not in data:
                data["scheduled_minutes"] = data["scheduled_transit_minutes"]
            elif "scheduled_minutes" in data and "scheduled_transit_minutes" not in data:
                data["scheduled_transit_minutes"] = data["scheduled_minutes"]

            # Predicted transit minutes aliases
            if "predicted_transit_minutes" in data and "predicted_minutes" not in data:
                data["predicted_minutes"] = data["predicted_transit_minutes"]
            elif "predicted_minutes" in data and "predicted_transit_minutes" not in data:
                data["predicted_transit_minutes"] = data["predicted_minutes"]

            # Baseline transit minutes aliases
            if "baseline_transit_minutes" in data and "baseline_minutes" not in data:
                data["baseline_minutes"] = data["baseline_transit_minutes"]
            elif "baseline_minutes" in data and "baseline_transit_minutes" not in data:
                data["baseline_transit_minutes"] = data["baseline_minutes"]
            elif "baseline_transit_minutes" not in data and "baseline_minutes" not in data:
                sched = data.get("scheduled_transit_minutes", data.get("scheduled_minutes", 0.0))
                data["baseline_transit_minutes"] = sched
                data["baseline_minutes"] = sched
        return data


class StationETAPrediction(BaseModel):
    """Complete multi-station ETA prediction for an upcoming station stop."""
    train_number: str = Field(description="Unique train identifier")
    station_code: str = Field(description="Target station code")
    station_name: Optional[str] = Field(default=None, description="Target station name")
    station_sequence: int = Field(description="Sequence number of station along the route")
    distance_to_go_km: float = Field(description="Total remaining route distance from current position in km")
    segments_ahead: int = Field(description="Number of inter-station segments ahead (1=next stop, 2=second, etc.)")
    intermediate_halts: int = Field(description="Number of intermediate scheduled halts between train and target station")

    scheduled_eta: datetime = Field(description="Timetable scheduled arrival datetime")
    baseline_eta: datetime = Field(description="Baseline heuristic predicted arrival datetime")
    predicted_eta: datetime = Field(description="XGBoost chained predicted arrival datetime")
    confidence_lower_bound: datetime = Field(description="Heuristic confidence lower bound (earliest expected arrival)")
    confidence_upper_bound: datetime = Field(description="Heuristic confidence upper bound (latest expected arrival)")

    predicted_remaining_minutes: float = Field(description="XGBoost chained predicted minutes to arrive at this station")
    baseline_remaining_minutes: float = Field(description="Baseline heuristic predicted minutes to arrive at this station")
    confidence_lower_bound_minutes: float = Field(description="Lower bound remaining minutes")
    confidence_upper_bound_minutes: float = Field(description="Upper bound remaining minutes")
    uncertainty_margin_minutes: float = Field(description="Plus/minus uncertainty window in minutes (+/- margin)")

    method: str = Field(default="XGBOOST_CHAINED", description="Forecasting method: strictly 'XGBOOST_CHAINED'")
    confidence_disclaimer: str = Field(
        default="Uncalibrated heuristic MVP uncertainty interval widening with horizon (+/- base * sqrt(segments_ahead))",
        description="Disclaimer on confidence calibration",
    )
    generated_timestamp: datetime = Field(description="Timestamp when prediction was produced")
    segment_predictions: List[SegmentPrediction] = Field(
        default_factory=list, description="Chained segment breakdown leading to this station"
    )


class MultiStationETAService:
    """
    Multi-Station ETA Forecasting Service.
    Produces chained segment-by-segment XGBoost predictions for all upcoming stations.
    """

    def __init__(
        self,
        predictor: Optional[ETAPredictor] = None,
        baseline_service: Optional[BaselineETAService] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        base_uncertainty_minutes: float = BASE_SEGMENT_UNCERTAINTY_MINUTES,
        max_speed_kmh: float = MAX_PERMISSIBLE_SPEED_KMH,
    ):
        self.predictor = predictor or ETAPredictor()
        self.baseline_service = baseline_service or BaselineETAService()
        self.feature_builder = feature_builder or FeatureBuilder()
        self.base_uncertainty_minutes = max(1.0, float(base_uncertainty_minutes))
        self.max_speed_kmh = max(10.0, float(max_speed_kmh))

    def predict_upcoming_stations(
        self,
        train_state: TrainRunningState,
        route_stations: Optional[List[Any]] = None,
        db: Optional[Session] = None,
    ) -> List[StationETAPrediction]:
        """
        Calculates chained XGBoost ETA predictions for every upcoming station along the route.
        Returns an empty list if the train has already completed its journey or is at terminus.
        """
        if train_state.status.upper() == "COMPLETED":
            return []

        stops = self._resolve_route_stops(train_state.train_number, route_stations, db)
        if not stops:
            return []

        # Find current station position along the route
        curr_code = (train_state.current_station_code or "").upper().strip()
        curr_seq = train_state.current_station_sequence or 1

        curr_idx: Optional[int] = None
        for idx, s in enumerate(stops):
            code = getattr(s, "station_code", getattr(s, "code", "")).upper().strip()
            seq = getattr(s, "sequence", getattr(s, "seq", idx + 1))
            if code == curr_code or seq == curr_seq:
                curr_idx = idx
                break

        if curr_idx is None:
            curr_idx = 0

        # If train is already at or past final station, no upcoming stations remain
        if curr_idx >= len(stops) - 1:
            return []

        upcoming_stops = stops[curr_idx + 1:]
        if not upcoming_stops:
            return []

        # Reference timestamps and timetables
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

        obs_time = train_state.timestamp
        ref_tz = obs_time.tzinfo if obs_time.tzinfo is not None else timezone.utc
        if obs_time.tzinfo is None:
            obs_time = obs_time.replace(tzinfo=ref_tz)

        is_live = str(train_state.source or "").lower() in ("external_api", "live_api", "live")
        origin_base_dt = datetime.combine(journey_d, origin_dep_time).replace(tzinfo=ref_tz)
        cumulative_times = compute_cumulative_schedule_minutes(stops)

        # Track train simulation state forward segment by segment
        curr_stop = stops[curr_idx]
        curr_dist = float(getattr(curr_stop, "distance_from_source_km", getattr(curr_stop, "dist_km", 0.0)))
        progress = max(0.0, min(1.0, float(train_state.segment_progress or 0.0)))

        next_immediate_stop = stops[curr_idx + 1]
        next_dist = float(getattr(next_immediate_stop, "distance_from_source_km", getattr(next_immediate_stop, "dist_km", curr_dist)))
        dist_covered_active_seg = progress * max(0.0, next_dist - curr_dist)
        current_train_pos_km = curr_dist + dist_covered_active_seg

        curr_code_clean = getattr(curr_stop, "station_code", getattr(curr_stop, "code", ""))
        curr_sched_time = cumulative_times.get(curr_code_clean, 0.0)
        next_code_clean = getattr(next_immediate_stop, "station_code", getattr(next_immediate_stop, "code", ""))
        next_sched = cumulative_times.get(next_code_clean, curr_sched_time)
        time_cov = progress * max(0.0, next_sched - curr_sched_time)
        curr_position_time = curr_sched_time + time_cov

        sim_current_time = obs_time
        sim_current_delay = max(0.0, float(train_state.current_delay_minutes or 0.0))
        accumulated_transit_minutes = 0.0

        chained_segment_records: List[SegmentPrediction] = []
        station_predictions: List[StationETAPrediction] = []

        for j, target_stop in enumerate(upcoming_stops):
            from_stop = stops[curr_idx + j]
            segments_ahead = j + 1
            to_code = getattr(target_stop, "station_code", getattr(target_stop, "code", "")).upper().strip()
            from_code = getattr(from_stop, "station_code", getattr(from_stop, "code", "")).upper().strip()
            to_name = getattr(target_stop, "station_name", getattr(target_stop, "name", to_code))
            to_seq = int(getattr(target_stop, "sequence", getattr(target_stop, "seq", curr_idx + j + 2)))

            to_dist = float(getattr(target_stop, "distance_from_source_km", getattr(target_stop, "dist_km", 0.0)))
            from_dist = float(getattr(from_stop, "distance_from_source_km", getattr(from_stop, "dist_km", 0.0)))

            # Distance for this single segment
            if j == 0:
                seg_distance = max(0.1, to_dist - current_train_pos_km)
                active_state = train_state
            else:
                seg_distance = max(0.1, to_dist - from_dist)
                # Construct forwarded state at departure from previous station
                active_state = TrainRunningState(
                    train_number=train_state.train_number,
                    journey_date=train_state.journey_date,
                    train_name=train_state.train_name,
                    status="RUNNING",
                    current_station_code=from_code,
                    current_station_sequence=int(getattr(from_stop, "sequence", getattr(from_stop, "seq", curr_idx + j + 1))),
                    current_delay_minutes=sim_current_delay,
                    previous_station_code=getattr(stops[curr_idx + j - 1], "station_code", None) if (curr_idx + j > 0) else None,
                    next_station_code=to_code,
                    next_station_distance_km=seg_distance,
                    segment_progress=0.0,
                    speed_kmh=train_state.speed_kmh,
                    timestamp=sim_current_time,
                    source=train_state.source,
                )

            # Scheduled segment transit minutes
            sched_to = cumulative_times.get(to_code, 0.0)
            sched_from = cumulative_times.get(from_code, 0.0)
            if j == 0:
                scheduled_segment_min = max(0.5, (1.0 - progress) * max(0.5, sched_to - sched_from))
            else:
                scheduled_segment_min = max(0.5, sched_to - sched_from)

            # 1. XGBoost inference for this individual segment (with graceful baseline fallback)
            is_ml_fallback = False
            try:
                feat = self.feature_builder.build(
                    train_state=active_state,
                    target_station_code=to_code,
                    route_stations=stops,
                )
                pred_seg_minutes = self.predictor.predict(feat)
            except Exception:
                is_ml_fallback = True
                avg_speed = max(20.0, min(130.0, float(active_state.speed_kmh or 60.0)))
                pred_seg_minutes = max(scheduled_segment_min, (seg_distance / avg_speed) * 60.0)

            # Kinematic physical feasibility guard for this segment
            min_segment_minutes = (seg_distance / self.max_speed_kmh) * 60.0
            pred_seg_minutes = max(pred_seg_minutes, min_segment_minutes)

            # Predicted arrival at to_stop
            predicted_arrival_dt = sim_current_time + timedelta(minutes=pred_seg_minutes)
            accumulated_transit_minutes += pred_seg_minutes

            # Scheduled dwell time at arrival station
            is_terminus = (curr_idx + j + 1 == len(stops) - 1)
            scheduled_dwell = 0.0 if is_terminus else float(getattr(target_stop, "scheduled_stop_minutes", 0.0) or 0.0)
            predicted_departure_dt = predicted_arrival_dt + timedelta(minutes=scheduled_dwell)

            # Update simulated state for next loop iteration
            if is_live:
                sched_rem_to = max(0.0, sched_to - curr_position_time)
                sched_arrival_dt = obs_time + timedelta(minutes=sched_rem_to)
            else:
                sched_arrival_dt = origin_base_dt + timedelta(minutes=sched_to)
            new_delay = max(0.0, (predicted_arrival_dt - sched_arrival_dt).total_seconds() / 60.0)
            sim_current_delay = round(new_delay, 2)
            sim_current_time = predicted_departure_dt

            # Record this segment's prediction
            seg_record = SegmentPrediction(
                segment_index=segments_ahead,
                segment_order=segments_ahead,
                from_station_code=from_code,
                to_station_code=to_code,
                to_station_name=to_name,
                segment_distance_km=round(seg_distance, 2),
                distance_km=round(seg_distance, 2),
                scheduled_transit_minutes=round(scheduled_segment_min, 1),
                scheduled_minutes=round(scheduled_segment_min, 1),
                baseline_transit_minutes=round(scheduled_segment_min, 1),
                baseline_minutes=round(scheduled_segment_min, 1),
                predicted_transit_minutes=round(pred_seg_minutes, 1),
                predicted_minutes=round(pred_seg_minutes, 1),
                scheduled_dwell_minutes=round(scheduled_dwell, 1),
                cumulative_transit_minutes=round(accumulated_transit_minutes, 1),
                predicted_arrival_time=predicted_arrival_dt,
                predicted_departure_time=predicted_departure_dt,
            )
            chained_segment_records.append(seg_record)

            # 2. Baseline heuristic prediction for this target station
            base_pred = self.baseline_service.predict_station(
                train_state=train_state,
                target_station_code=to_code,
                route_stations=stops,
                db=db,
            )
            baseline_arrival_dt = base_pred.baseline_eta
            baseline_rem_min = max(0.0, (baseline_arrival_dt - obs_time).total_seconds() / 60.0)

            # 3. Transparent Heuristic Confidence Range
            # Uncertainty widens with the square root of segments ahead: margin = base * sqrt(k)
            uncertainty_margin = round(self.base_uncertainty_minutes * math.sqrt(segments_ahead), 1)

            total_dist_to_go = max(0.1, to_dist - current_train_pos_km)
            min_feasible_total_min = (total_dist_to_go / self.max_speed_kmh) * 60.0
            lower_bound_min = max(min_feasible_total_min, round(accumulated_transit_minutes - uncertainty_margin, 1))
            upper_bound_min = round(accumulated_transit_minutes + uncertainty_margin, 1)

            confidence_lower_dt = obs_time + timedelta(minutes=lower_bound_min)
            confidence_upper_dt = obs_time + timedelta(minutes=upper_bound_min)

            station_pred = StationETAPrediction(
                train_number=train_state.train_number,
                station_code=to_code,
                station_name=to_name,
                station_sequence=to_seq,
                distance_to_go_km=round(total_dist_to_go, 2),
                segments_ahead=segments_ahead,
                intermediate_halts=max(0, segments_ahead - 1),
                scheduled_eta=sched_arrival_dt,
                baseline_eta=baseline_arrival_dt,
                predicted_eta=predicted_arrival_dt,
                confidence_lower_bound=confidence_lower_dt,
                confidence_upper_bound=confidence_upper_dt,
                predicted_remaining_minutes=round(accumulated_transit_minutes, 1),
                baseline_remaining_minutes=round(baseline_rem_min, 1),
                confidence_lower_bound_minutes=lower_bound_min,
                confidence_upper_bound_minutes=upper_bound_min,
                uncertainty_margin_minutes=uncertainty_margin,
                method="BASELINE_FALLBACK" if is_ml_fallback else "XGBOOST_CHAINED",
                confidence_disclaimer=(
                    "ML model unavailable. Operating in kinematic baseline fallback mode."
                    if is_ml_fallback
                    else f"Uncalibrated heuristic MVP uncertainty interval widening with horizon (+/- {self.base_uncertainty_minutes}m * sqrt({segments_ahead}) = +/- {uncertainty_margin}m)"
                ),
                generated_timestamp=datetime.now(ref_tz),
                segment_predictions=list(chained_segment_records),
            )
            station_predictions.append(station_pred)

            # Dwell time also adds to the running clock for subsequent segment arrivals
            if not is_terminus:
                accumulated_transit_minutes += scheduled_dwell

        return station_predictions

    def predict_station_eta(
        self,
        train_state: TrainRunningState,
        target_station_code: str,
        route_stations: Optional[List[Any]] = None,
        db: Optional[Session] = None,
    ) -> StationETAPrediction:
        """
        Calculates the chained XGBoost ETA prediction for a single specific upcoming station.
        Raises ValueError if the station is invalid or not in the train's upcoming route.
        """
        target_code = target_station_code.upper().strip()
        all_upcoming = self.predict_upcoming_stations(
            train_state=train_state, route_stations=route_stations, db=db
        )

        for pred in all_upcoming:
            if pred.station_code == target_code:
                return pred

        # If not found, provide informative error
        stops = self._resolve_route_stops(train_state.train_number, route_stations, db)
        all_codes = [getattr(s, "station_code", getattr(s, "code", "")).upper().strip() for s in stops]

        if target_code not in all_codes:
            raise ValueError(
                f"Station '{target_code}' is not on the route for train '{train_state.train_number}'. "
                f"Valid stations: {all_codes}"
            )
        else:
            raise ValueError(
                f"Station '{target_code}' is already behind the current train position "
                f"('{train_state.current_station_code}')."
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
