"""
Synthetic Training Dataset Generator for Dynamic Train ETA Forecasting.
========================================================================
DISCLAIMER:
This module generates SYNTHETIC SUPERVISED TRAINING DATA using the discrete-event
train running simulator and feature engineering pipeline.
All delays, disruptions, and timetables are synthetic reference data for MVP ML
development and DO NOT represent real historical Indian Railways operational logs.
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime, date, time, timedelta, timezone
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
from sqlalchemy.orm import Session, joinedload

from backend.database.connection import SessionLocal
from backend.database.init_db import init_db
from backend.database.models import Train, Route, RouteStation, Station
from backend.services.schemas import TrainRunningState
from backend.simulator.events import EventType, SimulationEvent
from backend.features.feature_builder import FeatureBuilder, TrainFeatures

DATA_DISCLAIMER = (
    "SYNTHETIC SIMULATED DATASET ONLY. Timings, routes, and disruptions are generated "
    "by the synthetic train-running simulator for MVP model development and DO NOT "
    "represent real historical Indian Railways operational logs."
)

PRD_FEATURE_COLUMNS = [
    "distance_to_go_km",
    "scheduled_time_to_go_min",
    "num_intermediate_halts",
    "current_delay_min",
    "delay_trend_3pt",
    "active_speed_restriction_flag",
    "congestion_score_downstream",
    "hist_avg_delay_this_section",
    "hist_recovery_rate_section",
    "weather_flag",
    "day_type",
]

DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


class SyntheticDatasetGenerator:
    """
    Generates realistic supervised learning datasets for dynamic train ETA forecasting.
    Simulates train journeys across routes, injects stochastic operational disruptions,
    extracts the 11 PRD features at intermediate checkpoints, and calculates the true
    ground-truth actual remaining minutes to the next station.
    """

    def __init__(
        self,
        db: Optional[Session] = None,
        seed: int = 42,
        disruption_probability: float = 0.30,
    ):
        self.db = db
        self.seed = seed
        self.disruption_probability = max(0.0, min(1.0, float(disruption_probability)))
        self.feature_builder = FeatureBuilder()
        self.rng = random.Random(seed)

    def generate_dataset(
        self,
        num_days: int = 20,
        samples_per_segment: int = 5,
        output_csv_path: Optional[Path] = None,
        output_metadata_path: Optional[Path] = None,
        base_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Runs multi-day fast-forward simulations across all configured routes and trains.
        Captures observations at multiple progress checkpoints and computes true target:
        actual_remaining_minutes = arrival_timestamp - observation_timestamp.
        """
        # Re-seed RNG for deterministic reproducibility
        self.rng = random.Random(self.seed)
        from backend.simulator.journey import SimulatedJourney

        start_date = base_date or date(2026, 10, 1)
        trains_data = self._load_trains_and_routes()

        if not trains_data:
            raise ValueError("No trains or routes found in SQLite database. Seed database first.")

        all_rows: List[Dict[str, Any]] = []
        total_journeys = 0
        total_disruptions = 0

        # Progress thresholds for sampling observations along each segment
        if samples_per_segment == 5:
            sample_thresholds = [0.0, 0.20, 0.40, 0.60, 0.80]
        elif samples_per_segment == 3:
            sample_thresholds = [0.0, 0.35, 0.70]
        else:
            step = 1.0 / max(1, samples_per_segment)
            sample_thresholds = [round(i * step, 2) for i in range(samples_per_segment)]

        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            is_weekend = current_date.weekday() in (5, 6)
            is_holiday = (day_offset % 7 == 3)  # Synthetic holiday every 7 days

            for train_info in trains_data:
                total_journeys += 1
                train_id = train_info["train_number"]
                route_id = train_info["route_id"]
                stops = train_info["stops"]
                train_type = train_info["train_type"]
                train_name = train_info["name"]

                journey_id = f"J_{train_id}_{current_date.strftime('%Y%m%d')}"

                # Initial departure time from origin
                origin_stop = stops[0]
                dep_str = origin_stop.scheduled_departure or "16:00"
                dep_parts = dep_str.strip().split(":")
                origin_dep_time = time(int(dep_parts[0]), int(dep_parts[1]))
                sim_start_dt = datetime.combine(current_date, origin_dep_time).replace(tzinfo=timezone.utc)

                # Stochastic initial delay (80% on-time, 20% slight departure delay)
                initial_delay = 0.0
                if self.rng.random() < 0.20:
                    initial_delay = round(self.rng.uniform(1.0, 6.0), 1)

                journey = SimulatedJourney(
                    train_id=train_info["id"],
                    train_number=train_id,
                    train_name=train_name,
                    train_type=train_type,
                    route_id=route_id,
                    stops=stops,
                    journey_id=total_journeys,
                    journey_date=current_date,
                    current_delay_minutes=initial_delay,
                    sim_time=sim_start_dt,
                )

                recent_delays: List[float] = [initial_delay]

                # Run journey segment by segment
                for seg_idx in range(len(stops) - 1):
                    curr_stop = stops[seg_idx]
                    next_stop = stops[seg_idx + 1]

                    # 1. Determine if disruption occurs on this segment
                    disruption_on_seg = self.rng.random() < self.disruption_probability
                    active_event: Optional[SimulationEvent] = None

                    if disruption_on_seg:
                        total_disruptions += 1
                        ev_type = self.rng.choice([
                            EventType.SIGNAL_HALT,
                            EventType.CONGESTION,
                            EventType.SPEED_RESTRICTION,
                            EventType.UNSCHEDULED_HALT,
                            EventType.WEATHER,
                        ])

                        if ev_type == EventType.SIGNAL_HALT:
                            delay_m = round(self.rng.uniform(6.0, 22.0), 1)
                            active_event = SimulationEvent(
                                event_type=ev_type,
                                delay_minutes=delay_m,
                                severity="HIGH" if delay_m > 15 else "MEDIUM",
                                metadata={"cause": "Signal interlocking failure"},
                            )
                        elif ev_type == EventType.CONGESTION:
                            delay_m = round(self.rng.uniform(4.0, 15.0), 1)
                            factor = round(self.rng.uniform(0.35, 0.65), 2)
                            active_event = SimulationEvent(
                                event_type=ev_type,
                                delay_minutes=delay_m,
                                severity="MEDIUM",
                                metadata={"speed_factor": factor},
                            )
                        elif ev_type == EventType.SPEED_RESTRICTION:
                            delay_m = round(self.rng.uniform(3.0, 10.0), 1)
                            limit = float(self.rng.choice([30.0, 45.0, 50.0]))
                            active_event = SimulationEvent(
                                event_type=ev_type,
                                delay_minutes=delay_m,
                                severity="LOW" if limit >= 45 else "MEDIUM",
                                metadata={"speed_limit_kmh": limit},
                            )
                        elif ev_type == EventType.UNSCHEDULED_HALT:
                            delay_m = round(self.rng.uniform(5.0, 18.0), 1)
                            active_event = SimulationEvent(
                                event_type=ev_type,
                                delay_minutes=delay_m,
                                severity="HIGH",
                                metadata={"cause": "Emergency brake / technical check"},
                            )
                        elif ev_type == EventType.WEATHER:
                            delay_m = round(self.rng.uniform(8.0, 25.0), 1)
                            factor = round(self.rng.uniform(0.5, 0.7), 2)
                            active_event = SimulationEvent(
                                event_type=ev_type,
                                delay_minutes=delay_m,
                                severity="MEDIUM",
                                metadata={"speed_factor": factor, "max_speed_kmh": 50.0},
                            )

                    # Trigger disruption early or mid-segment
                    inject_threshold = self.rng.choice([0.10, 0.25, 0.40])
                    event_injected = False

                    segment_observations: List[Dict[str, Any]] = []
                    sampled_thresholds = set()

                    # Fast-forward ticks on this segment
                    dt_step = 60.0  # 1 simulated minute per advance step
                    while journey.current_station_index == seg_idx and journey.status != "COMPLETED":
                        # Check event injection
                        if active_event and not event_injected and journey.segment_progress >= inject_threshold:
                            journey.inject_event(active_event)
                            event_injected = True

                        # Check observation sampling
                        for target_p in sample_thresholds:
                            if target_p not in sampled_thresholds and journey.segment_progress >= target_p:
                                state = journey.to_train_running_state()

                                # Build 11 PRD features
                                feat = self.feature_builder.build(
                                    train_state=state,
                                    target_station_code=next_stop.station_code,
                                    route_stations=stops,
                                    active_events=journey.active_events,
                                    is_holiday=is_holiday,
                                    recent_delays=recent_delays,
                                )

                                row = {
                                    "journey_id": journey_id,
                                    "train_id": train_id,
                                    "route_id": route_id,
                                    "station": curr_stop.station_code,
                                    "target_station": next_stop.station_code,
                                    "timestamp": state.timestamp.isoformat(),
                                    # 11 PRD features
                                    "distance_to_go_km": feat.distance_to_go_km,
                                    "scheduled_time_to_go_min": feat.scheduled_time_to_go_min,
                                    "num_intermediate_halts": feat.num_intermediate_halts,
                                    "current_delay_min": feat.current_delay_min,
                                    "delay_trend_3pt": feat.delay_trend_3pt,
                                    "active_speed_restriction_flag": feat.active_speed_restriction_flag,
                                    "congestion_score_downstream": feat.congestion_score_downstream,
                                    "hist_avg_delay_this_section": feat.hist_avg_delay_this_section,
                                    "hist_recovery_rate_section": feat.hist_recovery_rate_section,
                                    "weather_flag": feat.weather_flag,
                                    "day_type": feat.day_type,
                                    # Target and flags
                                    "actual_remaining_minutes": 0.0,
                                    "disruption_present": 1 if disruption_on_seg else 0,
                                    "is_synthetic": 1,
                                    "_sample_dt": state.timestamp,
                                }
                                segment_observations.append(row)
                                sampled_thresholds.add(target_p)

                        journey.advance(dt_step)

                    # Arrival at next_stop reached!
                    arrival_dt = journey.sim_time

                    # Retrospectively assign true target: actual remaining minutes to arrival
                    for obs in segment_observations:
                        elapsed_seconds = (arrival_dt - obs["_sample_dt"]).total_seconds()
                        obs["actual_remaining_minutes"] = max(0.1, round(elapsed_seconds / 60.0, 2))
                        del obs["_sample_dt"]  # Clean up internal helper

                    all_rows.extend(segment_observations)

                    # Track recent delay trend
                    recent_delays.append(journey.current_delay_minutes)
                    if len(recent_delays) > 5:
                        recent_delays.pop(0)

                    # Complete station dwell time if intermediate halt
                    if next_stop.scheduled_stop_minutes > 0.0 and journey.status != "COMPLETED":
                        journey.advance(next_stop.scheduled_stop_minutes * 60.0)

        df = pd.DataFrame(all_rows)

        # Save dataset to disk if path specified or use default processed dir
        if output_csv_path is not None:
            csv_file = output_csv_path
            meta_file = output_metadata_path or (csv_file.parent / f"{csv_file.stem}_metadata.json")
        else:
            csv_file = DEFAULT_PROCESSED_DIR / "synthetic_train_eta_dataset.csv"
            meta_file = output_metadata_path or (DEFAULT_PROCESSED_DIR / "dataset_metadata.json")

        csv_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_file, index=False)

        # Compute metadata statistics
        self._write_metadata(df, meta_file, num_days, total_journeys, total_disruptions)

        return df

    def _write_metadata(
        self,
        df: pd.DataFrame,
        meta_path: Path,
        num_days: int,
        total_journeys: int,
        total_disruptions: int,
    ) -> None:
        """Writes dataset provenance and distribution statistics to JSON metadata file."""
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        target_series = df["actual_remaining_minutes"]
        meta_info = {
            "dataset_name": "Synthetic Train Running ETA Supervised Dataset",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": DATA_DISCLAIMER,
            "is_synthetic": True,
            "random_seed": self.seed,
            "num_days_simulated": num_days,
            "total_journeys": total_journeys,
            "total_samples": len(df),
            "disruption_samples": int((df["disruption_present"] == 1).sum()),
            "normal_samples": int((df["disruption_present"] == 0).sum()),
            "feature_columns": PRD_FEATURE_COLUMNS,
            "target_column": "actual_remaining_minutes",
            "target_distribution": {
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "mean": float(round(target_series.mean(), 2)),
                "median": float(round(target_series.median(), 2)),
                "std": float(round(target_series.std(), 2)),
                "p25": float(round(target_series.quantile(0.25), 2)),
                "p75": float(round(target_series.quantile(0.75), 2)),
            },
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2)

    def _load_trains_and_routes(self) -> List[Dict[str, Any]]:
        """Loads trains and ordered station stops from SQLite."""
        session = self.db or SessionLocal()
        own_session = self.db is None

        try:
            from backend.simulator.journey import StationStop

            trains = (
                session.query(Train)
                .options(
                    joinedload(Train.route)
                    .joinedload(Route.route_stations)
                    .joinedload(RouteStation.station)
                )
                .order_by(Train.train_number.asc())
                .all()
            )

            result = []
            for t in trains:
                if not t.route or not t.route.route_stations:
                    continue
                sorted_rs = sorted(t.route.route_stations, key=lambda rs: rs.sequence)
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
                result.append({
                    "id": t.id,
                    "train_number": t.train_number,
                    "name": t.name,
                    "train_type": t.train_type,
                    "route_id": t.route_id,
                    "stops": stops,
                })
            return result
        finally:
            if own_session:
                session.close()


def main():
    """CLI runner to generate synthetic training dataset."""
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic Train ETA Dataset Generator")
    parser.add_argument("--days", type=int, default=20, help="Number of synthetic days to simulate (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--samples-per-seg", type=int, default=5, help="Samples per segment (default: 5)")
    args = parser.parse_args()

    print(f"\n[Dataset Generator] Initializing generator with seed={args.seed}, days={args.days}...")
    init_db()
    gen = SyntheticDatasetGenerator(seed=args.seed)
    df = gen.generate_dataset(num_days=args.days, samples_per_segment=args.samples_per_seg)

    print(f"\n[Dataset Generator] Successfully generated {len(df)} samples across {args.days} days.")
    print(f"Output saved to: data/processed/synthetic_train_eta_dataset.csv")
    print(f"Metadata saved to: data/processed/dataset_metadata.json")
    print("\nTarget Distribution (actual_remaining_minutes):")
    print(df["actual_remaining_minutes"].describe())
    print("\nDisruption Breakdown:")
    print(df["disruption_present"].value_counts())


if __name__ == "__main__":
    main()
