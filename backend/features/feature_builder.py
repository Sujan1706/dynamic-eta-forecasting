import logging
from datetime import date, datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from backend.database.models import Train, Route, RouteStation, Station
from backend.services.schemas import TrainRunningState
from backend.simulator.events import EventType

logger = logging.getLogger("dynamic_eta.features")


class TrainFeatures(BaseModel):
    """
    11-feature engineered tabular representation for Dynamic Train ETA Forecasting.
    Conforms to the PRD Section 4 ML feature contract.
    """
    train_number: str = Field(description="Unique train identifier")
    target_station_code: str = Field(description="Target downstream station code")

    # 1. Static features
    distance_to_go_km: float = Field(description="Remaining route distance to target station in km")
    scheduled_time_to_go_min: float = Field(description="Timetable-implied scheduled travel minutes to target station")
    num_intermediate_halts: int = Field(description="Number of scheduled stops between current position and target station")

    # 2. Dynamic state features
    current_delay_min: float = Field(description="Current train delay in minutes at last reported checkpoint")
    delay_trend_3pt: float = Field(description="Delay slope/delta across last 3 checkpoints in minutes")
    active_speed_restriction_flag: int = Field(description="1 if TSR / speed restriction active on upcoming section, else 0")
    congestion_score_downstream: float = Field(description="Traffic density / congestion score on upcoming track section (0.0 to 1.0)")

    # 3. Historical section features (Day 1 MVP synthetic defaults)
    hist_avg_delay_this_section: float = Field(description="Historical average delay for section in minutes (Day 1 MVP default)")
    hist_recovery_rate_section: float = Field(description="Historical recovery pace in minutes recovered per 100km (Day 1 MVP default)")

    # 4. Contextual features
    weather_flag: int = Field(description="1 if adverse weather (fog, heavy rain) active, else 0")
    day_type: int = Field(description="Day type classification: 0=Weekday, 1=Weekend, 2=Holiday")

    # 5. Provenance metadata (tracking source: live data, historical data, simulator/default source)
    feature_provenance: Dict[str, str] = Field(
        default_factory=dict,
        description="Source mapping for each feature ('live data', 'historical data', 'simulator/default source')",
    )
    estimated_features: List[str] = Field(
        default_factory=list,
        description="List of features that used fallback or estimated defaults because they are unavailable in live telemetry",
    )

    @property
    def live_features(self) -> List[str]:
        """List of feature names derived from live data."""
        return [f for f, src in self.feature_provenance.items() if src == "live data"]

    @property
    def historical_features(self) -> List[str]:
        """List of feature names derived from historical data."""
        return [f for f, src in self.feature_provenance.items() if src == "historical data"]

    @property
    def default_features(self) -> List[str]:
        """List of feature names derived from simulator or default fallbacks."""
        return [f for f, src in self.feature_provenance.items() if src == "simulator/default source"]

    def to_model_input_dict(self) -> Dict[str, Any]:
        """Returns the dictionary containing only the 11 numeric features for model training/inference."""
        return {
            "distance_to_go_km": self.distance_to_go_km,
            "scheduled_time_to_go_min": self.scheduled_time_to_go_min,
            "num_intermediate_halts": self.num_intermediate_halts,
            "current_delay_min": self.current_delay_min,
            "delay_trend_3pt": self.delay_trend_3pt,
            "active_speed_restriction_flag": self.active_speed_restriction_flag,
            "congestion_score_downstream": self.congestion_score_downstream,
            "hist_avg_delay_this_section": self.hist_avg_delay_this_section,
            "hist_recovery_rate_section": self.hist_recovery_rate_section,
            "weather_flag": self.weather_flag,
            "day_type": self.day_type,
        }

    def to_feature_list(self) -> List[float]:
        """Returns ordered list of 11 numeric features matching ML training column order."""
        return list(self.to_model_input_dict().values())


class SectionStatsProvider:
    """
    Provides section-level historical operational statistics.

    DAY 1 MVP NOTICE:
    Section average delays and recovery rates currently use synthetic reference defaults
    for MVP simulator validation and baseline testing.
    DO NOT treat these defaults as real historical Indian Railways operational logs.
    On Day 2, this provider will be populated from historical training parquet datasets.
    """
    DEFAULT_AVG_DELAY_MIN = 8.5
    DEFAULT_RECOVERY_RATE_PER_100KM = 2.5

    def __init__(self, section_stats_map: Optional[Dict[str, Dict[str, float]]] = None):
        self._stats = section_stats_map or {}

    def get_hist_avg_delay(self, source_code: str, target_code: str) -> float:
        key = f"{source_code.upper().strip()}_{target_code.upper().strip()}"
        if key in self._stats:
            return float(self._stats[key].get("avg_delay_min", self.DEFAULT_AVG_DELAY_MIN))
        return self.DEFAULT_AVG_DELAY_MIN

    def get_hist_recovery_rate(self, source_code: str, target_code: str) -> float:
        key = f"{source_code.upper().strip()}_{target_code.upper().strip()}"
        if key in self._stats:
            return float(self._stats[key].get("recovery_rate_per_100km", self.DEFAULT_RECOVERY_RATE_PER_100KM))
        return self.DEFAULT_RECOVERY_RATE_PER_100KM


def compute_cumulative_schedule_minutes(stops: List[Any]) -> Dict[str, float]:
    """
    Computes cumulative scheduled timetable minutes from the origin station,
    correctly handling midnight (24h) crossings along the route sequence.
    """
    cumulative: Dict[str, float] = {}
    day_offset_minutes = 0.0
    prev_minutes: Optional[float] = None

    for stop in stops:
        code = getattr(stop, "station_code", getattr(stop, "code", ""))
        arr = getattr(stop, "scheduled_arrival", None)
        dep = getattr(stop, "scheduled_departure", None)
        time_str = arr or dep

        if not time_str:
            cumulative[code] = 0.0
            continue

        try:
            parts = time_str.strip().split(":")
            raw_minutes = int(parts[0]) * 60.0 + int(parts[1])
        except (ValueError, IndexError):
            cumulative[code] = 0.0
            continue

        # If time has rolled backward by more than 6 hours, we crossed midnight
        if prev_minutes is not None and raw_minutes < (prev_minutes - 360.0):
            day_offset_minutes += 1440.0

        current_cumulative = day_offset_minutes + raw_minutes
        cumulative[code] = current_cumulative

        # Update previous time marker using departure if available
        dep_str = dep or time_str
        try:
            dep_parts = dep_str.strip().split(":")
            prev_minutes = int(dep_parts[0]) * 60.0 + int(dep_parts[1])
        except (ValueError, IndexError):
            prev_minutes = raw_minutes

    # Normalize so origin departure is 0.0 minutes
    if stops:
        origin_code = getattr(stops[0], "station_code", getattr(stops[0], "code", ""))
        origin_base = cumulative.get(origin_code, 0.0)
        for c in cumulative:
            cumulative[c] = max(0.0, cumulative[c] - origin_base)

    return cumulative


class FeatureBuilder:
    """
    Day 1 Feature Engineering Pipeline.
    Calculates the 11 core ML features for any train and upcoming target station.
    Accepts normalized TrainRunningState telemetry from either the synthetic simulator or external API.
    """

    def __init__(self, stats_provider: Optional[SectionStatsProvider] = None):
        self.stats_provider = stats_provider or SectionStatsProvider()

    def build(
        self,
        train_state: TrainRunningState,
        target_station_code: str,
        route_stations: Optional[List[Any]] = None,
        db: Optional[Session] = None,
        recent_delays: Optional[List[float]] = None,
        active_events: Optional[List[Any]] = None,
        is_holiday: bool = False,
        speed_restriction_flag: Optional[int] = None,
        congestion_score: Optional[float] = None,
        weather_flag: Optional[int] = None,
    ) -> TrainFeatures:
        """
        Builds the complete 11-feature TrainFeatures vector for (train_state, target_station_code).
        """
        target_code = target_station_code.upper().strip()

        # 1. Resolve ordered route stops (from provided list or SQLite database)
        stops = self._resolve_route_stops(train_state.train_number, route_stations, db)
        if not stops:
            raise ValueError(f"No route stations found for train number '{train_state.train_number}'.")

        # Index stops by code and sequence
        stop_map: Dict[str, Tuple[int, Any]] = {}
        for idx, stop in enumerate(stops):
            code = getattr(stop, "station_code", getattr(stop, "code", "")).upper().strip()
            stop_map[code] = (idx, stop)

        if target_code not in stop_map:
            raise ValueError(
                f"Target station '{target_code}' is not on the route for train '{train_state.train_number}'. "
                f"Available route stations: {list(stop_map.keys())}"
            )

        target_idx, target_stop = stop_map[target_code]

        # Locate current position
        curr_code = (train_state.current_station_code or "").upper().strip()
        if curr_code in stop_map:
            curr_idx, curr_stop = stop_map[curr_code]
        else:
            # Fallback by current_station_sequence
            seq = train_state.current_station_sequence or 1
            curr_idx = max(0, min(len(stops) - 1, seq - 1))
            curr_stop = stops[curr_idx]

        # 2. Static distance and timetable calculations
        curr_dist = float(getattr(curr_stop, "distance_from_source_km", getattr(curr_stop, "dist_km", 0.0)))
        target_dist = float(getattr(target_stop, "distance_from_source_km", getattr(target_stop, "dist_km", 0.0)))

        cumulative_times = compute_cumulative_schedule_minutes(stops)
        curr_code_clean = getattr(curr_stop, "station_code", getattr(curr_stop, "code", ""))
        target_code_clean = getattr(target_stop, "station_code", getattr(target_stop, "code", ""))
        curr_sched_time = cumulative_times.get(curr_code_clean, 0.0)
        target_sched_time = cumulative_times.get(target_code_clean, 0.0)

        # Handle segment progression
        progress = max(0.0, min(1.0, float(train_state.segment_progress or 0.0)))

        # Check if train telemetry is from live API or simulator
        is_live = str(train_state.source or "").lower() in ("external_api", "live_api", "live")
        feature_provenance: Dict[str, str] = {}
        estimated_features: List[str] = []

        if target_idx < curr_idx:
            # Target station is already in the past
            distance_to_go_km = 0.0
            scheduled_time_to_go_min = 0.0
            num_intermediate_halts = 0
        elif target_idx == curr_idx:
            # Currently at target station
            distance_to_go_km = 0.0
            scheduled_time_to_go_min = 0.0
            num_intermediate_halts = 0
        else:
            # Target is downstream
            # Distance across immediate next segment
            if curr_idx < len(stops) - 1:
                next_stop = stops[curr_idx + 1]
                next_dist = float(getattr(next_stop, "distance_from_source_km", getattr(next_stop, "dist_km", curr_dist)))
                seg_dist = max(0.0, next_dist - curr_dist)

                # Requirement 2: Map live next_station_distance_km if provided
                if train_state.next_station_distance_km is not None and seg_dist > 0:
                    live_dist_remaining = max(0.0, float(train_state.next_station_distance_km))
                    if target_idx == curr_idx + 1:
                        distance_to_go_km = round(min(live_dist_remaining, seg_dist), 2)
                    else:
                        distance_to_go_km = round(min(live_dist_remaining, seg_dist) + max(0.0, target_dist - next_dist), 2)

                    dist_covered_on_seg = max(0.0, seg_dist - min(live_dist_remaining, seg_dist))
                    effective_progress = dist_covered_on_seg / seg_dist if seg_dist > 0 else 0.0
                else:
                    dist_covered_on_seg = progress * seg_dist
                    curr_position_km = curr_dist + dist_covered_on_seg
                    distance_to_go_km = max(0.0, round(target_dist - curr_position_km, 2))
                    effective_progress = progress

                # Timetable across immediate next segment
                next_code_clean = getattr(next_stop, "station_code", getattr(next_stop, "code", ""))
                next_sched_time = cumulative_times.get(next_code_clean, curr_sched_time)
                seg_time = max(0.0, next_sched_time - curr_sched_time)
                time_covered_on_seg = effective_progress * seg_time
            else:
                dist_covered_on_seg = 0.0
                time_covered_on_seg = 0.0
                distance_to_go_km = 0.0

            curr_position_time = curr_sched_time + time_covered_on_seg
            scheduled_time_to_go_min = max(0.0, round(target_sched_time - curr_position_time, 2))
            num_intermediate_halts = max(0, target_idx - curr_idx - 1)

        # 1. Positional features provenance
        pos_source = "live data" if is_live else "simulator/default source"
        feature_provenance["distance_to_go_km"] = pos_source
        feature_provenance["scheduled_time_to_go_min"] = pos_source
        feature_provenance["num_intermediate_halts"] = pos_source

        # 2. Dynamic state features
        current_delay_min = round(float(train_state.current_delay_minutes or 0.0), 2)
        feature_provenance["current_delay_min"] = "live data" if is_live else "simulator/default source"

        # Delay trend over last 3 checkpoints
        delay_trend_3pt = self._calculate_delay_trend(current_delay_min, recent_delays)
        if recent_delays and len(recent_delays) >= 2:
            feature_provenance["delay_trend_3pt"] = "live data" if is_live else "simulator/default source"
        else:
            feature_provenance["delay_trend_3pt"] = "simulator/default source"
            if is_live:
                estimated_features.append("delay_trend_3pt")

        # Active speed restriction flag (Requirement 3 & 4: no fake values, clear defaults)
        if speed_restriction_flag is not None:
            active_tsr = 1 if speed_restriction_flag else 0
            feature_provenance["active_speed_restriction_flag"] = "live data" if is_live else "simulator/default source"
        elif active_events:
            active_tsr = 1 if any(getattr(e, "event_type", "") in ("SPEED_RESTRICTION", EventType.SPEED_RESTRICTION) for e in active_events) else 0
            feature_provenance["active_speed_restriction_flag"] = "simulator/default source"
        else:
            active_tsr = 0
            feature_provenance["active_speed_restriction_flag"] = "simulator/default source"
            if is_live:
                estimated_features.append("active_speed_restriction_flag")

        # Congestion score downstream (Requirement 3 & 4: no fake values, clear defaults)
        if congestion_score is not None:
            cong_score = max(0.0, min(1.0, float(congestion_score)))
            feature_provenance["congestion_score_downstream"] = "live data" if is_live else "simulator/default source"
        elif active_events:
            cong_ev = next(
                (e for e in active_events if getattr(e, "event_type", "") in ("CONGESTION", EventType.CONGESTION)),
                None,
            )
            if cong_ev:
                meta = getattr(cong_ev, "metadata", {}) or {}
                factor = float(meta.get("speed_factor", 0.5))
                cong_score = round(max(0.0, min(1.0, 1.0 - factor)), 2)
            else:
                cong_score = 0.0
            feature_provenance["congestion_score_downstream"] = "simulator/default source"
        else:
            cong_score = 0.0
            feature_provenance["congestion_score_downstream"] = "simulator/default source"
            if is_live:
                estimated_features.append("congestion_score_downstream")

        # Weather flag (Requirement 3 & 4: no fake values, clear defaults)
        if weather_flag is not None:
            w_flag = 1 if weather_flag else 0
            feature_provenance["weather_flag"] = "live data" if is_live else "simulator/default source"
        elif active_events:
            w_flag = 1 if any(getattr(e, "event_type", "") in ("WEATHER", EventType.WEATHER) for e in active_events) else 0
            feature_provenance["weather_flag"] = "simulator/default source"
        else:
            w_flag = 0
            feature_provenance["weather_flag"] = "simulator/default source"
            if is_live:
                estimated_features.append("weather_flag")

        # 3. Historical section statistics (Requirement 3: use existing historical feature provider)
        hist_avg_delay = self.stats_provider.get_hist_avg_delay(curr_code_clean, target_code_clean)
        hist_recovery = self.stats_provider.get_hist_recovery_rate(curr_code_clean, target_code_clean)
        feature_provenance["hist_avg_delay_this_section"] = "historical data"
        feature_provenance["hist_recovery_rate_section"] = "historical data"

        # 4. Contextual feature (Day type classification: 0=Weekday, 1=Weekend, 2=Holiday)
        day_type = self._determine_day_type(train_state.journey_date, is_holiday)
        feature_provenance["day_type"] = "live data" if is_live else "simulator/default source"

        # Requirement: Logging showing which features came from:
        # - live data
        # - historical data
        # - simulator/default source
        live_list = [f for f, src in feature_provenance.items() if src == "live data"]
        hist_list = [f for f, src in feature_provenance.items() if src == "historical data"]
        default_list = [f for f, src in feature_provenance.items() if src == "simulator/default source"]

        logger.info(
            "Feature mapping for train %s -> %s (source=%s):\n"
            "  - live data: %s\n"
            "  - historical data: %s\n"
            "  - simulator/default source: %s%s",
            train_state.train_number,
            target_code,
            train_state.source,
            ", ".join(live_list) if live_list else "none",
            ", ".join(hist_list) if hist_list else "none",
            ", ".join(default_list) if default_list else "none",
            f" (estimated/default: {', '.join(estimated_features)})" if estimated_features else "",
        )

        return TrainFeatures(
            train_number=train_state.train_number,
            target_station_code=target_code,
            distance_to_go_km=distance_to_go_km,
            scheduled_time_to_go_min=scheduled_time_to_go_min,
            num_intermediate_halts=num_intermediate_halts,
            current_delay_min=current_delay_min,
            delay_trend_3pt=delay_trend_3pt,
            active_speed_restriction_flag=active_tsr,
            congestion_score_downstream=cong_score,
            hist_avg_delay_this_section=hist_avg_delay,
            hist_recovery_rate_section=hist_recovery,
            weather_flag=w_flag,
            day_type=day_type,
            feature_provenance=feature_provenance,
            estimated_features=estimated_features,
        )

    def _resolve_route_stops(
        self,
        train_number: str,
        route_stations: Optional[List[Any]],
        db: Optional[Session],
    ) -> List[Any]:
        """Resolves route stops from explicit argument or SQLite query."""
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
                # Attach station_code helper attribute if RouteStation object
                for s in stops:
                    if hasattr(s, "station") and s.station:
                        setattr(s, "station_code", s.station.code)
                return stops

        return []

    def _calculate_delay_trend(
        self,
        current_delay: float,
        recent_delays: Optional[List[float]],
    ) -> float:
        """
        Calculates delay slope across last 3 reporting checkpoints:
        slope = (Delay_t - Delay_{t-2}) / 2
        """
        if not recent_delays:
            return 0.0

        history = list(recent_delays)
        # If current_delay not in history, append it
        if not history or history[-1] != current_delay:
            history.append(current_delay)

        if len(history) >= 3:
            slope = (history[-1] - history[-3]) / 2.0
        elif len(history) == 2:
            slope = history[-1] - history[-2]
        else:
            slope = 0.0

        return round(slope, 2)

    def _determine_day_type(self, journey_date: date, is_holiday: bool) -> int:
        """
        Determines day type:
        0 = Weekday (Mon-Fri)
        1 = Weekend (Sat-Sun)
        2 = Holiday
        """
        if is_holiday:
            return 2

        if isinstance(journey_date, datetime):
            d = journey_date.date()
        elif isinstance(journey_date, date):
            d = journey_date
        else:
            d = date.today()

        # Monday=0, Sunday=6
        if d.weekday() in (5, 6):
            return 1
        return 0
