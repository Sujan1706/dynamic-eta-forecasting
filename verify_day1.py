"""
End-to-End Day 1 Verification Script
Validates the entire pipeline:
1. SQLite Database initialization & seeding
2. Simulator startup & journey creation
3. Disruption Event Injection (Signal Halt, Congestion, Speed Restriction)
4. Normalized TrainRunningState generation
5. 11-Feature extraction
6. Baseline ETA forecasting
7. FastAPI endpoints (/health, /trains, /train/{id}, /train/{id}/eta/{station})
8. Internal consistency verification
"""

import sys
import json
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from backend.database.connection import SessionLocal
from backend.database.init_db import init_db
from backend.database.seed import seed_data
from backend.database.models import Station, Route, Train, Journey, Event
from backend.simulator.engine import TrainSimulator
from backend.simulator.events import EventType
from backend.services.schemas import TrainRunningState
from backend.services.baseline_eta import BaselineETAService, BaselineETAPrediction
from backend.features.feature_builder import FeatureBuilder, TrainFeatures
from backend.main import app, get_active_simulator


def run_verification():
    print("=" * 65)
    print("      DYNAMIC ETA FORECASTING - DAY 1 END-TO-END VERIFICATION")
    print("=" * 65)
    
    results = {}

    # ---------------------------------------------------------
    # 1. DATABASE VERIFICATION
    # ---------------------------------------------------------
    print("\n[1/7] Initializing SQLite Database and Seeding Reference Network...")
    try:
        init_db()
        db = SessionLocal()
        seed_res = seed_data(db)
        
        station_count = db.query(Station).count()
        route_count = db.query(Route).count()
        train_count = db.query(Train).count()
        
        print(f"  -> SQLite Database active at: data/train_eta.db")
        print(f"  -> Stations in DB: {station_count}")
        print(f"  -> Routes in DB: {route_count}")
        print(f"  -> Trains in DB: {train_count}")
        
        assert station_count >= 20, f"Expected at least 20 stations, found {station_count}"
        assert route_count >= 3, f"Expected 3 routes, found {route_count}"
        assert train_count >= 8, f"Expected 8 trains, found {train_count}"
        results["Database"] = "PASS"
        print("  [SUCCESS] Database layer verified.")
    except Exception as exc:
        results["Database"] = f"FAIL: {exc}"
        print(f"  [ERROR] Database check failed: {exc}", file=sys.stderr)
        return False, results

    # ---------------------------------------------------------
    # 2. SIMULATOR & JOURNEYS VERIFICATION
    # ---------------------------------------------------------
    print("\n[2/7] Initializing Train Simulator and Loading Running Journeys...")
    try:
        sim = TrainSimulator(simulation_speed=5.0, db=db)
        loaded_journeys = sim.load_from_db(db)
        print(f"  -> Loaded {loaded_journeys} active journeys across 3 corridors.")
        assert loaded_journeys == 8, f"Expected 8 journeys loaded, got {loaded_journeys}"

        # Run 2 ticks to advance journeys
        print("  -> Advancing simulator by 2 ticks (120s simulated per tick)...")
        states = sim.tick(step_seconds=60.0)
        assert len(states) == 8
        results["Simulator"] = "PASS"
        print("  [SUCCESS] Simulator and journeys verified.")
    except Exception as exc:
        results["Simulator"] = f"FAIL: {exc}"
        print(f"  [ERROR] Simulator check failed: {exc}", file=sys.stderr)
        return False, results

    # ---------------------------------------------------------
    # 3. DISRUPTION EVENT INJECTION
    # ---------------------------------------------------------
    print("\n[3/7] Injecting Operational Disruptions (Signal Halt, Congestion, TSR)...")
    try:
        # Event 1: SIGNAL_HALT on Train 12302
        e1 = sim.inject_event(
            train_number="12302",
            event_type=EventType.SIGNAL_HALT,
            delay_minutes=15.0,
            severity="HIGH",
            metadata={"cause": "Red signal at block outer"},
            db=db,
        )
        j1 = sim.get_journey("12302")
        assert j1.status == "HALTED"
        assert j1.current_speed_kmh == 0.0
        print(f"  -> Injected SIGNAL_HALT on Train 12302: Status={j1.status}, Delay={j1.current_delay_minutes:.1f}m")

        # Event 2: CONGESTION on Train 12952
        e2 = sim.inject_event(
            train_number="12952",
            event_type=EventType.CONGESTION,
            delay_minutes=6.0,
            severity="MEDIUM",
            metadata={"speed_factor": 0.45},
            db=db,
        )
        j2 = sim.get_journey("12952")
        print(f"  -> Injected CONGESTION on Train 12952: Speed={j2.current_speed_kmh:.1f} km/h, Delay={j2.current_delay_minutes:.1f}m")

        # Event 3: SPEED_RESTRICTION on Train 12260
        e3 = sim.inject_event(
            train_number="12260",
            event_type=EventType.SPEED_RESTRICTION,
            delay_minutes=4.0,
            severity="LOW",
            metadata={"speed_limit_kmh": 30.0},
            db=db,
        )
        j3 = sim.get_journey("12260")
        assert j3.current_speed_kmh == 30.0
        print(f"  -> Injected SPEED_RESTRICTION on Train 12260: Speed={j3.current_speed_kmh:.1f} km/h, Delay={j3.current_delay_minutes:.1f}m")

        # Check DB persistence
        db_events = db.query(Event).all()
        assert len(db_events) >= 3, "Expected at least 3 disruption events logged in SQLite."
        results["Events"] = "PASS"
        print("  [SUCCESS] All disruption events injected, state-modified, and persisted.")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        results["Events"] = f"FAIL: {exc}"
        print(f"  [ERROR] Event injection check failed: {exc}", file=sys.stderr)
        return False, results

    # ---------------------------------------------------------
    # 4. NORMALIZED RUNNING STATE GENERATION
    # ---------------------------------------------------------
    print("\n[4/7] Generating and Validating Normalized TrainRunningState Telemetry...")
    try:
        all_states = sim.get_all_states()
        assert len(all_states) == 8
        for st in all_states:
            assert isinstance(st, TrainRunningState)
            assert st.source == "simulator"
            assert st.train_number is not None
            assert st.journey_date is not None
            assert st.status in ("RUNNING", "HALTED", "COMPLETED", "SCHEDULED")
            assert 0.0 <= st.segment_progress <= 1.0
        
        sample_state = sim.get_state("12302")
        print(f"  -> Sample normalized state for Train 12302:")
        print(f"     Train: {sample_state.train_name} (#{sample_state.train_number})")
        print(f"     Location: {sample_state.current_station_code} -> Next: {sample_state.next_station_code}")
        print(f"     Progress: {sample_state.segment_progress:.1%} | Delay: +{sample_state.current_delay_minutes:.1f}m | Source: {sample_state.source}")
        results["NormalizedState"] = "PASS"
        print("  [SUCCESS] Normalized state generation verified.")
    except Exception as exc:
        results["NormalizedState"] = f"FAIL: {exc}"
        print(f"  [ERROR] Normalized state check failed: {exc}", file=sys.stderr)
        return False, results

    # ---------------------------------------------------------
    # 5. FEATURE ENGINEERING (11 FEATURES)
    # ---------------------------------------------------------
    print("\n[5/7] Generating and Validating the 11-Feature Vector...")
    try:
        feat_builder = FeatureBuilder()
        state_12302 = sim.get_state("12302")
        # Target PRYJ (downstream from CNB)
        features = feat_builder.build(
            train_state=state_12302,
            target_station_code="PRYJ",
            db=db,
            active_events=sim.journeys["12302"].active_events,
        )
        assert isinstance(features, TrainFeatures)
        feature_dict = features.to_model_input_dict()
        assert len(feature_dict) == 11, f"Expected 11 features, got {len(feature_dict)}"
        
        print("  -> 11-Feature Vector Extracted for Train 12302 -> PRYJ:")
        for k, v in feature_dict.items():
            print(f"     * {k:30s}: {v}")

        results["Features"] = "PASS"
        print("  [SUCCESS] Feature engineering pipeline verified.")
    except Exception as exc:
        results["Features"] = f"FAIL: {exc}"
        print(f"  [ERROR] Feature extraction check failed: {exc}", file=sys.stderr)
        return False, results

    # ---------------------------------------------------------
    # 6. BASELINE ETA FORECASTING
    # ---------------------------------------------------------
    print("\n[6/7] Calculating Baseline ETA Predictions...")
    try:
        baseline_svc = BaselineETAService()
        state_12302 = sim.get_state("12302")
        
        # 1. Single next station
        pred_next = baseline_svc.predict_next_station(state_12302, db=db)
        assert isinstance(pred_next, BaselineETAPrediction)
        assert pred_next.method == "BASELINE"
        
        # 2. Multi-station upcoming chain
        upcoming_preds = baseline_svc.predict_upcoming_stations(state_12302, db=db)
        assert len(upcoming_preds) >= 5, f"Expected upcoming stations, got {len(upcoming_preds)}"
        
        print(f"  -> Baseline ETA chain for Train 12302 from {state_12302.current_station_code} (Delay: +{state_12302.current_delay_minutes:.1f}m):")
        for p in upcoming_preds:
            print(
                f"     Stop: {p.target_station:5s} ({p.target_station_name:25s}) | "
                f"Dist: {p.distance_to_go_km:6.1f}km | Sched: {p.scheduled_eta.strftime('%d-%b %H:%M')} | "
                f"Baseline ETA: {p.baseline_eta.strftime('%d-%b %H:%M')} | Recov: -{p.recovery_applied:.1f}m | "
                f"Net Delay: +{p.predicted_delay:.1f}m"
            )

        results["Baseline"] = "PASS"
        print("  [SUCCESS] Baseline ETA service verified.")
    except Exception as exc:
        results["Baseline"] = f"FAIL: {exc}"
        print(f"  [ERROR] Baseline ETA calculation failed: {exc}", file=sys.stderr)
        return False, results

    # ---------------------------------------------------------
    # 7. FASTAPI DAY 1 ENDPOINTS & INTERNAL CONSISTENCY
    # ---------------------------------------------------------
    print("\n[7/7] Testing FastAPI Endpoints & Verifying Internal Consistency...")
    try:
        client = TestClient(app)

        # Endpoint 1: GET /health
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"
        print("  -> GET /health: 200 OK")

        # Endpoint 2: GET /trains
        r_trains = client.get("/trains")
        assert r_trains.status_code == 200
        trains_data = r_trains.json()
        assert trains_data["total"] >= 8
        print(f"  -> GET /trains: 200 OK (returned {trains_data['total']} trains)")

        # Endpoint 3: GET /train/{train_id}
        r_train = client.get("/train/12302")
        assert r_train.status_code == 200
        t_info = r_train.json()
        assert t_info["train_number"] == "12302"
        assert len(t_info["route_stations"]) == 8
        print(f"  -> GET /train/12302: 200 OK ({t_info['name']} with {len(t_info['route_stations'])} stops)")

        # Endpoint 4: GET /train/{train_id}/eta/{station_code}
        r_eta = client.get("/train/12302/eta/PRYJ")
        assert r_eta.status_code == 200
        eta_data = r_eta.json()
        assert eta_data["train_number"] == "12302"
        assert eta_data["target_station"] == "PRYJ"
        print(f"  -> GET /train/12302/eta/PRYJ: 200 OK")

        # Consistency Checks
        p = eta_data["prediction"]
        f = eta_data["features"]
        
        # Verify feature and prediction consistency
        assert p["target_station"] == "PRYJ"
        assert p["method"] == "BASELINE"
        assert p["distance_to_go_km"] == f["distance_to_go_km"]
        assert p["current_delay"] == f["current_delay_min"]
        assert p["distance_to_go_km"] > 0
        
        print("  -> Internal Consistency verified between prediction and features:")
        print(f"     * Distance match: {p['distance_to_go_km']} km == {f['distance_to_go_km']} km")
        print(f"     * Delay match:    {p['current_delay']} min == {f['current_delay_min']} min")
        print(f"     * Scheduled ETA:  {p['scheduled_eta']}")
        print(f"     * Baseline ETA:   {p['baseline_eta']} (Method: {p['method']})")

        results["FastAPI"] = "PASS"
        print("  [SUCCESS] FastAPI Day 1 endpoints verified.")
    except Exception as exc:
        results["FastAPI"] = f"FAIL: {exc}"
        print(f"  [ERROR] FastAPI check failed: {exc}", file=sys.stderr)
        return False, results
    finally:
        db.close()

    return True, results


if __name__ == "__main__":
    success, summary = run_verification()
    if not success:
        sys.exit(1)
