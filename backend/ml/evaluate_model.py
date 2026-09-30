"""
Structured Model Evaluation Module: Baseline Heuristic vs. Machine Learning.
=============================================================================
Compares the Day 1 Baseline ETA Service against the Day 2 XGBoost ETA Regressor
on the held-out test journeys across multiple forecasting horizons:
  - 1 station ahead (immediate next station)
  - 3 stations ahead (intermediate corridor horizon)
  - 5 stations ahead (distant terminus horizon, where route length supports it)
And operational conditions:
  - No disruption (normal cruising)
  - With disruption (active signal halt, congestion, speed restriction, weather)

Saves results to data/processed/model_metrics.json and prints human-readable tables.
Never manufactures or alters metrics.
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime, date, time, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.database.init_db import init_db
from backend.services.baseline_eta import BaselineETAService
from backend.features.feature_builder import FeatureBuilder
from backend.ml.predictor import ETAPredictor
from backend.ml.dataset_generator import SyntheticDatasetGenerator
from backend.ml.train_model import split_by_journey, DEFAULT_DATASET_PATH
from backend.simulator.events import SimulationEvent, EventType

DEFAULT_METRICS_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "processed"
    / "model_metrics.json"
)

DEFAULT_METADATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "models"
    / "eta_xgboost_metadata.json"
)

DATA_DISCLAIMER = (
    "SYNTHETIC SIMULATED DATASET ONLY. Timings, routes, disruptions, and model predictions "
    "are generated for MVP machine learning evaluation and DO NOT represent real historical "
    "Indian Railways operational logs."
)


def load_latest_metrics(
    metrics_path: Optional[Path | str] = None,
    metadata_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Loads latest model evaluation metrics and training metadata.
    Returns a structured dictionary without crashing if files are missing or unreadable.
    """
    m_path = Path(metrics_path) if metrics_path else DEFAULT_METRICS_OUTPUT_PATH
    meta_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH

    model_name = "Dynamic Train ETA XGBoost Regressor"
    model_version = "1.0.0"
    training_timestamp = None
    total_training_samples = None
    test_samples = None

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            model_name = meta.get("model_name", model_name)
            model_version = meta.get("model_version", model_version)
            training_timestamp = meta.get("trained_at")
            dataset_info = meta.get("dataset_info", {})
            total_training_samples = dataset_info.get("train_samples")
            test_samples = dataset_info.get("test_samples")
        except Exception:
            pass

    if not m_path.exists():
        return {
            "status": "UNAVAILABLE",
            "is_available": False,
            "model_name": model_name,
            "model_version": model_version,
            "training_timestamp": training_timestamp,
            "total_training_samples": total_training_samples,
            "test_samples": test_samples,
            "baseline_mae": None,
            "ml_mae": None,
            "baseline_rmse": None,
            "ml_rmse": None,
            "metrics_by_horizon": None,
            "by_horizon": None,
            "metrics_by_disruption_status": None,
            "metrics_by_disruption": None,
            "by_disruption": None,
            "cross_tabulation": None,
            "overall": None,
            "evaluation_timestamp": None,
            "message": f"Evaluation metrics file not found at {m_path}. Run evaluate_model.py to generate metrics.",
            "disclaimer": DATA_DISCLAIMER,
        }

    try:
        with open(m_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    except Exception as e:
        return {
            "status": "UNAVAILABLE",
            "is_available": False,
            "model_name": model_name,
            "model_version": model_version,
            "training_timestamp": training_timestamp,
            "total_training_samples": total_training_samples,
            "test_samples": test_samples,
            "baseline_mae": None,
            "ml_mae": None,
            "baseline_rmse": None,
            "ml_rmse": None,
            "metrics_by_horizon": None,
            "by_horizon": None,
            "metrics_by_disruption_status": None,
            "metrics_by_disruption": None,
            "by_disruption": None,
            "cross_tabulation": None,
            "overall": None,
            "evaluation_timestamp": None,
            "message": f"Failed to load evaluation metrics: {str(e)}",
            "disclaimer": DATA_DISCLAIMER,
        }

    eval_timestamp = eval_data.get("evaluation_timestamp")
    overall = eval_data.get("overall", {})
    by_horizon = eval_data.get("by_horizon", {})
    by_disruption = eval_data.get("by_disruption", {})
    cross_tab = eval_data.get("cross_tabulation", {})
    summary = eval_data.get("summary", {})

    evaluated_samples = (
        overall.get("sample_count")
        or summary.get("total_evaluated_samples")
        or test_samples
    )
    baseline_mae = overall.get("baseline_mae")
    ml_mae = overall.get("ml_mae")
    baseline_rmse = overall.get("baseline_rmse")
    ml_rmse = overall.get("ml_rmse")

    dataset_info_dict = meta.get("dataset_info", {}) if meta_path.exists() else {}
    total_journeys = dataset_info_dict.get("total_journeys")
    train_journeys = dataset_info_dict.get("train_journeys")
    test_journeys = summary.get("total_test_journeys") or dataset_info_dict.get("test_journeys")

    return {
        "status": "AVAILABLE",
        "is_available": True,
        "model_name": model_name,
        "model_version": model_version,
        "training_timestamp": training_timestamp,
        "total_training_samples": total_training_samples,
        "test_samples": evaluated_samples,
        "total_journeys": total_journeys,
        "train_journeys": train_journeys,
        "test_journeys": test_journeys,
        "dataset_info": dataset_info_dict,
        "summary": summary,
        "baseline_mae": baseline_mae,
        "ml_mae": ml_mae,
        "baseline_rmse": baseline_rmse,
        "ml_rmse": ml_rmse,
        "metrics_by_horizon": by_horizon,
        "by_horizon": by_horizon,
        "metrics_by_disruption_status": by_disruption,
        "metrics_by_disruption": by_disruption,
        "by_disruption": by_disruption,
        "cross_tabulation": cross_tab,
        "overall": overall,
        "evaluation_timestamp": eval_timestamp,
        "message": "Evaluation metrics loaded successfully.",
        "disclaimer": eval_data.get("disclaimer", DATA_DISCLAIMER),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE and RMSE given true and predicted values."""
    if len(y_true) == 0:
        return {"mae": 0.0, "rmse": 0.0}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": round(mae, 3), "rmse": round(rmse, 3)}


def format_comparison(
    baseline_metrics: Dict[str, float],
    ml_metrics: Dict[str, float],
    sample_count: int,
) -> Dict[str, Any]:
    """Formats comparison statistics, absolute differences, and percentage improvements."""
    b_mae = baseline_metrics["mae"]
    b_rmse = baseline_metrics["rmse"]
    m_mae = ml_metrics["mae"]
    m_rmse = ml_metrics["rmse"]

    abs_diff = round(m_mae - b_mae, 3)
    if b_mae > 0:
        pct_improvement = round(((b_mae - m_mae) / b_mae) * 100.0, 2)
    else:
        pct_improvement = 0.0

    winner = "ML" if m_mae < b_mae else ("BASELINE" if b_mae < m_mae else "TIE")

    return {
        "sample_count": sample_count,
        "baseline_mae": b_mae,
        "baseline_rmse": b_rmse,
        "ml_mae": m_mae,
        "ml_rmse": m_rmse,
        "absolute_diff_mae": abs_diff,
        "percentage_improvement": pct_improvement,
        "winner": winner,
    }


class ModelEvaluator:
    """
    Evaluates Baseline ETA vs. ML ETA across multiple horizons and operational regimes
    using the exact held-out test journeys.
    """

    def __init__(
        self,
        dataset_path: Optional[Path | str] = None,
        model_path: Optional[Path | str] = None,
        metadata_path: Optional[Path | str] = None,
        test_size: float = 0.20,
        random_seed: int = 42,
    ):
        self.dataset_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        self.test_size = test_size
        self.random_seed = random_seed

        # Initialize engines
        self.baseline_service = BaselineETAService()
        self.feature_builder = FeatureBuilder()
        self.predictor = ETAPredictor(model_path=model_path, metadata_path=metadata_path)
        self.gen = SyntheticDatasetGenerator(seed=random_seed)

    def extract_test_journey_ids(self) -> List[str]:
        """Loads dataset and extracts the strictly held-out test journey IDs."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        _, _, _, test_j_ids = split_by_journey(
            df, test_size=self.test_size, random_seed=self.random_seed
        )
        return test_j_ids

    def gather_evaluation_records(
        self,
        test_journey_ids: List[str],
        num_days: int = 25,
        base_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Replays the simulation across the held-out test journeys, capturing observations
        at horizons of 1, 3, and 5 stations ahead.
        """
        test_j_set = set(test_journey_ids)
        trains_data = self.gen._load_trains_and_routes()
        from backend.simulator.journey import SimulatedJourney

        rng = random.Random(self.random_seed)
        start_date = base_date or date(2026, 10, 1)

        eval_records: List[Dict[str, Any]] = []
        total_journeys_count = 0

        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            is_holiday = (day_offset % 7 == 3)

            for train_info in trains_data:
                total_journeys_count += 1
                train_id = train_info["train_number"]
                journey_id = f"J_{train_id}_{current_date.strftime('%Y%m%d')}"
                stops = train_info["stops"]

                origin_stop = stops[0]
                dep_str = origin_stop.scheduled_departure or "16:00"
                dep_parts = dep_str.strip().split(":")
                origin_dep_time = time(int(dep_parts[0]), int(dep_parts[1]))
                sim_start_dt = datetime.combine(current_date, origin_dep_time).replace(tzinfo=timezone.utc)

                initial_delay = 0.0
                if rng.random() < 0.20:
                    initial_delay = round(rng.uniform(1.0, 6.0), 1)

                journey = SimulatedJourney(
                    train_id=train_info["id"],
                    train_number=train_id,
                    train_name=train_info["name"],
                    train_type=train_info["train_type"],
                    route_id=train_info["route_id"],
                    stops=stops,
                    journey_id=total_journeys_count,
                    journey_date=current_date,
                    current_delay_minutes=initial_delay,
                    sim_time=sim_start_dt,
                )

                recent_delays = [initial_delay]
                is_test_journey = (journey_id in test_j_set)

                segment_events = []
                for seg_idx in range(len(stops) - 1):
                    disruption_on_seg = rng.random() < 0.30
                    active_event = None
                    if disruption_on_seg:
                        ev_type = rng.choice([
                            EventType.SIGNAL_HALT,
                            EventType.CONGESTION,
                            EventType.SPEED_RESTRICTION,
                            EventType.UNSCHEDULED_HALT,
                            EventType.WEATHER,
                        ])
                        if ev_type == EventType.SIGNAL_HALT:
                            delay_m = round(rng.uniform(6.0, 22.0), 1)
                            active_event = SimulationEvent(
                                event_type=ev_type,
                                delay_minutes=delay_m,
                                severity="HIGH" if delay_m > 15 else "MEDIUM",
                                metadata={"cause": "Signal failure"},
                            )
                        elif ev_type == EventType.CONGESTION:
                            delay_m = round(rng.uniform(4.0, 15.0), 1)
                            factor = round(rng.uniform(0.35, 0.65), 2)
                            active_event = SimulationEvent(
                                event_type=ev_type,
                                delay_minutes=delay_m,
                                severity="MEDIUM",
                                metadata={"speed_factor": factor},
                            )
                        elif ev_type == EventType.SPEED_RESTRICTION:
                            delay_m = round(rng.uniform(3.0, 10.0), 1)
                            limit = float(rng.choice([30.0, 45.0, 50.0]))
                            active_event = SimulationEvent(
                                event_type=ev_type,
                                delay_minutes=delay_m,
                                severity="LOW" if limit >= 45 else "MEDIUM",
                                metadata={"speed_limit_kmh": limit},
                            )
                        elif ev_type == EventType.UNSCHEDULED_HALT:
                            delay_m = round(rng.uniform(5.0, 18.0), 1)
                            active_event = SimulationEvent(
                                event_type=ev_type,
                                delay_minutes=delay_m,
                                severity="HIGH",
                                metadata={"cause": "Brake check"},
                            )
                        elif ev_type == EventType.WEATHER:
                            delay_m = round(rng.uniform(8.0, 25.0), 1)
                            factor = round(rng.uniform(0.5, 0.7), 2)
                            active_event = SimulationEvent(
                                event_type=ev_type,
                                delay_minutes=delay_m,
                                severity="MEDIUM",
                                metadata={"speed_factor": factor, "max_speed_kmh": 50.0},
                            )

                    inject_threshold = rng.choice([0.10, 0.25, 0.40])
                    segment_events.append((active_event, inject_threshold))

                station_arrival_times: Dict[int, datetime] = {}
                sample_thresholds = [0.0, 0.20, 0.40, 0.60, 0.80]
                checkpoint_snapshots: List[Dict[str, Any]] = []

                for seg_idx in range(len(stops) - 1):
                    curr_stop = stops[seg_idx]
                    next_stop = stops[seg_idx + 1]
                    active_event, inject_threshold = segment_events[seg_idx]
                    event_injected = False
                    sampled_thresholds = set()

                    while journey.current_station_index == seg_idx and journey.status != "COMPLETED":
                        if active_event and not event_injected and journey.segment_progress >= inject_threshold:
                            journey.inject_event(active_event)
                            event_injected = True

                        for target_p in sample_thresholds:
                            if target_p not in sampled_thresholds and journey.segment_progress >= target_p:
                                if is_test_journey:
                                    state = journey.to_train_running_state()
                                    checkpoint_snapshots.append({
                                        "seg_idx": seg_idx,
                                        "state": state,
                                        "obs_time": journey.sim_time,
                                        "disruption_present": 1 if active_event is not None else 0,
                                        "recent_delays": list(recent_delays),
                                        "active_events": list(journey.active_events),
                                        "is_holiday": is_holiday,
                                    })
                                sampled_thresholds.add(target_p)

                        journey.advance(60.0)

                    station_arrival_times[seg_idx + 1] = journey.sim_time
                    recent_delays.append(round(journey.current_delay_minutes, 2))
                    if len(recent_delays) > 3:
                        recent_delays.pop(0)

                    if next_stop.scheduled_stop_minutes > 0.0 and journey.status != "COMPLETED":
                        journey.advance(next_stop.scheduled_stop_minutes * 60.0)

                # Process snapshots for held-out test journeys
                if is_test_journey:
                    for snap in checkpoint_snapshots:
                        seg_idx = snap["seg_idx"]
                        state = snap["state"]
                        obs_time = snap["obs_time"]
                        disrupt = snap["disruption_present"]

                        for stations_ahead in [1, 3, 5]:
                            target_stop_idx = seg_idx + stations_ahead
                            # Only include if route length supports this horizon
                            if target_stop_idx < len(stops):
                                target_stop = stops[target_stop_idx]
                                arr_time = station_arrival_times[target_stop_idx]
                                actual_rem_min = (arr_time - obs_time).total_seconds() / 60.0

                                # 1. ML Prediction
                                feat = self.feature_builder.build(
                                    train_state=state,
                                    target_station_code=target_stop.station_code,
                                    route_stations=stops,
                                    active_events=snap["active_events"],
                                    is_holiday=snap["is_holiday"],
                                    recent_delays=snap["recent_delays"],
                                )
                                ml_pred = self.predictor.predict(feat)

                                # 2. Baseline Prediction
                                base_pred_obj = self.baseline_service.predict_station(
                                    train_state=state,
                                    target_station_code=target_stop.station_code,
                                    route_stations=stops,
                                )
                                base_rem_min = (base_pred_obj.baseline_eta - obs_time).total_seconds() / 60.0
                                base_rem_min = max(0.0, base_rem_min)

                                eval_records.append({
                                    "journey_id": journey_id,
                                    "train_id": train_id,
                                    "stations_ahead": stations_ahead,
                                    "disruption_present": disrupt,
                                    "actual_remaining_minutes": actual_rem_min,
                                    "baseline_remaining_minutes": base_rem_min,
                                    "ml_remaining_minutes": ml_pred,
                                })

        return pd.DataFrame(eval_records)

    def evaluate(
        self,
        output_json_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """
        Runs complete evaluation, computes segmented metrics, and serializes
        to model_metrics.json.
        """
        out_file = Path(output_json_path) if output_json_path else DEFAULT_METRICS_OUTPUT_PATH
        out_file.parent.mkdir(parents=True, exist_ok=True)

        test_j_ids = self.extract_test_journey_ids()
        eval_df = self.gather_evaluation_records(test_j_ids)

        # 1. By forecasting horizon (1, 3, 5 stations ahead)
        by_horizon: Dict[str, Any] = {}
        for sa in [1, 3, 5]:
            sub = eval_df[eval_df["stations_ahead"] == sa]
            b_m = compute_metrics(
                sub["actual_remaining_minutes"].values,
                sub["baseline_remaining_minutes"].values,
            )
            m_m = compute_metrics(
                sub["actual_remaining_minutes"].values,
                sub["ml_remaining_minutes"].values,
            )
            key = f"{sa}_station_ahead" if sa == 1 else f"{sa}_stations_ahead"
            by_horizon[key] = format_comparison(b_m, m_m, len(sub))

        # 2. By disruption condition (No Disruption vs With Disruption)
        by_disruption: Dict[str, Any] = {}
        for d, key in [(0, "no_disruption"), (1, "with_disruption")]:
            sub = eval_df[eval_df["disruption_present"] == d]
            b_m = compute_metrics(
                sub["actual_remaining_minutes"].values,
                sub["baseline_remaining_minutes"].values,
            )
            m_m = compute_metrics(
                sub["actual_remaining_minutes"].values,
                sub["ml_remaining_minutes"].values,
            )
            by_disruption[key] = format_comparison(b_m, m_m, len(sub))

        # 3. Cross-tabulation (Horizon x Disruption)
        cross_tab: Dict[str, Any] = {}
        for sa in [1, 3, 5]:
            for d, d_lbl in [(0, "no_disruption"), (1, "with_disruption")]:
                sub = eval_df[(eval_df["stations_ahead"] == sa) & (eval_df["disruption_present"] == d)]
                b_m = compute_metrics(
                    sub["actual_remaining_minutes"].values,
                    sub["baseline_remaining_minutes"].values,
                )
                m_m = compute_metrics(
                    sub["actual_remaining_minutes"].values,
                    sub["ml_remaining_minutes"].values,
                )
                key = f"{sa}_stations_ahead_{d_lbl}"
                cross_tab[key] = format_comparison(b_m, m_m, len(sub))

        # 4. Overall aggregate
        b_all = compute_metrics(
            eval_df["actual_remaining_minutes"].values,
            eval_df["baseline_remaining_minutes"].values,
        )
        m_all = compute_metrics(
            eval_df["actual_remaining_minutes"].values,
            eval_df["ml_remaining_minutes"].values,
        )
        overall_comp = format_comparison(b_all, m_all, len(eval_df))

        results = {
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "disclaimer": DATA_DISCLAIMER,
            "summary": {
                "total_test_journeys": len(test_j_ids),
                "total_evaluated_samples": len(eval_df),
                "random_seed": self.random_seed,
            },
            "by_horizon": by_horizon,
            "by_disruption": by_disruption,
            "cross_tabulation": cross_tab,
            "overall": overall_comp,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Pretty prints the evaluation report to stdout."""
    print("\n" + "=" * 95)
    print("  DAY 2 PHASE 3: BASELINE VS. ML ETA EVALUATION REPORT (HELD-OUT TEST SET)")
    print("=" * 95)
    print(f"Total Test Journeys:  {results['summary']['total_test_journeys']}")
    print(f"Total Test Samples:   {results['summary']['total_evaluated_samples']}")
    print(f"Evaluation Timestamp: {results['evaluation_timestamp']}")
    print("-" * 95)
    print(f"{'Category / Horizon':<26} {'Count':<7} {'Base MAE':<10} {'ML MAE':<10} {'Diff':<10} {'Improvement':<13} {'Winner':<10}")
    print("-" * 95)

    # 1. By Horizon
    print("[1] BY FORECASTING HORIZON:")
    for key, data in results["by_horizon"].items():
        label = key.replace("_", " ").title()
        pct_str = f"{data['percentage_improvement']:+.1f}%"
        diff_str = f"{data['absolute_diff_mae']:+.2f}m"
        print(f"  {label:<24} {data['sample_count']:<7} {data['baseline_mae']:<10.2f} {data['ml_mae']:<10.2f} {diff_str:<10} {pct_str:<13} {data['winner']:<10}")

    print("\n[2] BY OPERATIONAL REGIME (ALL HORIZONS):")
    for key, data in results["by_disruption"].items():
        label = key.replace("_", " ").title()
        pct_str = f"{data['percentage_improvement']:+.1f}%"
        diff_str = f"{data['absolute_diff_mae']:+.2f}m"
        print(f"  {label:<24} {data['sample_count']:<7} {data['baseline_mae']:<10.2f} {data['ml_mae']:<10.2f} {diff_str:<10} {pct_str:<13} {data['winner']:<10}")

    print("\n[3] CROSS-TABULATION (HORIZON x DISRUPTION):")
    for key, data in results["cross_tabulation"].items():
        label = key.replace("_", " ").title()
        pct_str = f"{data['percentage_improvement']:+.1f}%"
        diff_str = f"{data['absolute_diff_mae']:+.2f}m"
        print(f"  {label:<24} {data['sample_count']:<7} {data['baseline_mae']:<10.2f} {data['ml_mae']:<10.2f} {diff_str:<10} {pct_str:<13} {data['winner']:<10}")

    print("-" * 95)
    ov = results["overall"]
    ov_pct = f"{ov['percentage_improvement']:+.1f}%"
    ov_diff = f"{ov['absolute_diff_mae']:+.2f}m"
    print(f"  {'OVERALL AGGREGATE':<24} {ov['sample_count']:<7} {ov['baseline_mae']:<10.2f} {ov['ml_mae']:<10.2f} {ov_diff:<10} {ov_pct:<13} {ov['winner']:<10}")
    print("=" * 95 + "\n")


def main():
    """CLI entry point for model evaluation."""
    init_db()
    evaluator = ModelEvaluator()
    results = evaluator.evaluate()
    print_evaluation_report(results)
    print(f"Full metrics saved to: {DEFAULT_METRICS_OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
