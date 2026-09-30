"""
DEVELOPMENT / SIMULATION REFERENCE DATA
=======================================
DISCLAIMER:
The following station names, codes, distances, and timetables are DEVELOPMENT/SIMULATION
DATA for local MVP testing and simulator validation. They are realistic approximations
designed for discrete-event train running simulations and DO NOT represent actual live
or official Indian Railways operational schedules.
"""

from datetime import date, datetime, timezone
from sqlalchemy.orm import Session
from backend.database.connection import SessionLocal
from backend.database.models import Station, Route, RouteStation, Train, Journey, Event
from backend.database.init_db import init_db

DATA_DISCLAIMER = (
    "DEVELOPMENT/SIMULATION DATA ONLY. Timings, routes, and halts are synthetic "
    "approximations for MVP simulator development and do not represent actual live "
    "railway operational timetables."
)

# --------------------------------------------------------------------------------------
# Stations: 20 realistic Indian Railways stations across 3 regional trunk corridors
# --------------------------------------------------------------------------------------
STATIONS_DATA = [
    # Route 1: Northern - Eastern Trunk (NDLS - HWH)
    {"code": "NDLS", "name": "New Delhi", "latitude": 28.6424, "longitude": 77.2195},
    {"code": "CNB", "name": "Kanpur Central", "latitude": 26.4539, "longitude": 80.3507},
    {"code": "PRYJ", "name": "Prayagraj Junction", "latitude": 25.4358, "longitude": 81.8463},
    {"code": "DDU", "name": "Pt. DD Upadhyaya Junction", "latitude": 25.2818, "longitude": 83.1190},
    {"code": "GAYA", "name": "Gaya Junction", "latitude": 24.8020, "longitude": 84.9995},
    {"code": "DHN", "name": "Dhanbad Junction", "latitude": 23.7957, "longitude": 86.4304},
    {"code": "ASN", "name": "Asansol Junction", "latitude": 23.6889, "longitude": 86.9661},
    {"code": "HWH", "name": "Howrah Junction", "latitude": 22.5839, "longitude": 88.3426},

    # Route 2: Northern - Western Trunk (NDLS - MMCT) - NDLS is shared
    {"code": "MTJ", "name": "Mathura Junction", "latitude": 27.4924, "longitude": 77.6737},
    {"code": "KOTA", "name": "Kota Junction", "latitude": 25.2235, "longitude": 75.8711},
    {"code": "RTM", "name": "Ratlam Junction", "latitude": 23.3364, "longitude": 75.0373},
    {"code": "BRC", "name": "Vadodara Junction", "latitude": 22.3107, "longitude": 73.1812},
    {"code": "ST", "name": "Surat", "latitude": 21.2049, "longitude": 72.8406},
    {"code": "MMCT", "name": "Mumbai Central", "latitude": 18.9696, "longitude": 72.8193},

    # Route 3: Southern Corridor (SBC - MAS)
    {"code": "SBC", "name": "KSR Bengaluru City", "latitude": 12.9784, "longitude": 77.5695},
    {"code": "BNC", "name": "Bengaluru Cantt", "latitude": 12.9934, "longitude": 77.5982},
    {"code": "KJM", "name": "Krishnarajapuram", "latitude": 12.9982, "longitude": 77.6775},
    {"code": "BWT", "name": "Bangarapet Junction", "latitude": 12.9880, "longitude": 78.2047},
    {"code": "KPD", "name": "Katpadi Junction", "latitude": 12.9698, "longitude": 79.1360},
    {"code": "MAS", "name": "MGR Chennai Central", "latitude": 13.0827, "longitude": 80.2755},
]

# --------------------------------------------------------------------------------------
# 3 Routes with 6–8 ordered stations each
# --------------------------------------------------------------------------------------
ROUTES_CONFIG = [
    {
        "name": "Northern-Eastern Trunk Line (New Delhi to Howrah)",
        "source": "NDLS",
        "destination": "HWH",
        "stations": [
            {"code": "NDLS", "seq": 1, "dist_km": 0.0, "arr": None, "dep": "16:55", "stop_min": 0.0},
            {"code": "CNB", "seq": 2, "dist_km": 440.0, "arr": "21:35", "dep": "21:40", "stop_min": 5.0},
            {"code": "PRYJ", "seq": 3, "dist_km": 635.0, "arr": "23:43", "dep": "23:45", "stop_min": 2.0},
            {"code": "DDU", "seq": 4, "dist_km": 788.0, "arr": "01:42", "dep": "01:52", "stop_min": 10.0},
            {"code": "GAYA", "seq": 5, "dist_km": 993.0, "arr": "04:10", "dep": "04:13", "stop_min": 3.0},
            {"code": "DHN", "seq": 6, "dist_km": 1195.0, "arr": "06:40", "dep": "06:45", "stop_min": 5.0},
            {"code": "ASN", "seq": 7, "dist_km": 1253.0, "arr": "07:35", "dep": "07:37", "stop_min": 2.0},
            {"code": "HWH", "seq": 8, "dist_km": 1450.0, "arr": "09:55", "dep": None, "stop_min": 0.0},
        ],
        "trains": [
            {"train_number": "12302", "name": "Howrah Rajdhani Express", "train_type": "RAJDHANI"},
            {"train_number": "12306", "name": "Kolkata Rajdhani Express", "train_type": "RAJDHANI"},
            {"train_number": "12260", "name": "Sealdah Duronto Express", "train_type": "SUPERFAST"},
        ],
    },
    {
        "name": "Western Trunk Line (New Delhi to Mumbai Central)",
        "source": "NDLS",
        "destination": "MMCT",
        "stations": [
            {"code": "NDLS", "seq": 1, "dist_km": 0.0, "arr": None, "dep": "16:30", "stop_min": 0.0},
            {"code": "MTJ", "seq": 2, "dist_km": 141.0, "arr": "17:58", "dep": "18:00", "stop_min": 2.0},
            {"code": "KOTA", "seq": 3, "dist_km": 465.0, "arr": "21:30", "dep": "21:40", "stop_min": 10.0},
            {"code": "RTM", "seq": 4, "dist_km": 730.0, "arr": "01:25", "dep": "01:30", "stop_min": 5.0},
            {"code": "BRC", "seq": 5, "dist_km": 992.0, "arr": "04:45", "dep": "04:53", "stop_min": 8.0},
            {"code": "ST", "seq": 6, "dist_km": 1122.0, "arr": "06:15", "dep": "06:20", "stop_min": 5.0},
            {"code": "MMCT", "seq": 7, "dist_km": 1385.0, "arr": "08:35", "dep": None, "stop_min": 0.0},
        ],
        "trains": [
            {"train_number": "12952", "name": "Mumbai Rajdhani Express", "train_type": "RAJDHANI"},
            {"train_number": "12954", "name": "August Kranti Rajdhani", "train_type": "RAJDHANI"},
            {"train_number": "12926", "name": "Paschim Superfast Express", "train_type": "SUPERFAST"},
        ],
    },
    {
        "name": "Southern Corridor (Bengaluru to Chennai Central)",
        "source": "SBC",
        "destination": "MAS",
        "stations": [
            {"code": "SBC", "seq": 1, "dist_km": 0.0, "arr": None, "dep": "06:00", "stop_min": 0.0},
            {"code": "BNC", "seq": 2, "dist_km": 4.0, "arr": "06:10", "dep": "06:12", "stop_min": 2.0},
            {"code": "KJM", "seq": 3, "dist_km": 14.0, "arr": "06:23", "dep": "06:25", "stop_min": 2.0},
            {"code": "BWT", "seq": 4, "dist_km": 70.0, "arr": "07:08", "dep": "07:10", "stop_min": 2.0},
            {"code": "KPD", "seq": 5, "dist_km": 229.0, "arr": "09:08", "dep": "09:10", "stop_min": 2.0},
            {"code": "MAS", "seq": 6, "dist_km": 359.0, "arr": "11:00", "dep": None, "stop_min": 0.0},
        ],
        "trains": [
            {"train_number": "12028", "name": "KSR Bengaluru - MGR Chennai Shatabdi", "train_type": "SHATABDI"},
            {"train_number": "12610", "name": "Chennai Central Express", "train_type": "EXPRESS"},
        ],
    },
]


def seed_data(db: Session = None) -> dict:
    """
    Populates SQLite database with the 3-route realistic development network.
    Clearly marked as synthetic DEVELOPMENT/SIMULATION reference data.
    """
    own_session = False
    if db is None:
        init_db()
        db = SessionLocal()
        own_session = True

    try:
        # Check if already seeded
        if db.query(Station).first():
            print("Database already contains data. Skipping seed.")
            return {"status": "already_seeded"}

        # 1. Seed Stations (20 stations)
        station_map = {}
        for s_data in STATIONS_DATA:
            station = Station(**s_data)
            db.add(station)
            db.flush()
            station_map[station.code] = station

        total_routes = 0
        total_trains = 0
        created_route_ids = []
        created_train_ids = []

        # 2. Seed Routes, RouteStations, and Trains
        for r_cfg in ROUTES_CONFIG:
            source_st = station_map[r_cfg["source"]]
            dest_st = station_map[r_cfg["destination"]]

            route = Route(
                source_station_id=source_st.id,
                destination_station_id=dest_st.id,
            )
            db.add(route)
            db.flush()
            total_routes += 1
            created_route_ids.append(route.id)

            # Route stations
            for stop in r_cfg["stations"]:
                rs = RouteStation(
                    route_id=route.id,
                    station_id=station_map[stop["code"]].id,
                    sequence=stop["seq"],
                    distance_from_source_km=stop["dist_km"],
                    scheduled_arrival=stop["arr"],
                    scheduled_departure=stop["dep"],
                    scheduled_stop_minutes=stop["stop_min"],
                )
                db.add(rs)

            # Trains for this route
            for t_data in r_cfg["trains"]:
                train = Train(
                    train_number=t_data["train_number"],
                    name=t_data["name"],
                    train_type=t_data["train_type"],
                    route_id=route.id,
                )
                db.add(train)
                db.flush()
                total_trains += 1
                created_train_ids.append(train.id)

        # 3. Seed sample active journeys for simulator baseline testing
        # Journey 1 on Route 1 (NDLS-HWH, Train 12302)
        utc_today = datetime.now(timezone.utc).date()
        j1 = Journey(
            train_id=created_train_ids[0],
            journey_date=utc_today,
            current_station_id=station_map["CNB"].id,
            current_sequence=2,
            current_delay_minutes=14.0,
            status="RUNNING",
        )
        db.add(j1)
        db.flush()

        # Disruption event on Journey 1
        e1 = Event(
            journey_id=j1.id,
            timestamp=datetime.now(timezone.utc),
            event_type="SIGNAL_HALT",
            delay_minutes=14.0,
            severity="MEDIUM",
            event_metadata={
                "data_source": "SIMULATION_DATA",
                "disclaimer": DATA_DISCLAIMER,
                "location": "CNB outer",
                "cause": "Synthetic signal halt test",
            },
        )
        db.add(e1)

        # Journey 2 on Route 2 (NDLS-MMCT, Train 12952)
        j2 = Journey(
            train_id=created_train_ids[3],
            journey_date=utc_today,
            current_station_id=station_map["KOTA"].id,
            current_sequence=3,
            current_delay_minutes=6.0,
            status="RUNNING",
        )
        db.add(j2)
        db.flush()

        # Disruption event on Journey 2
        e2 = Event(
            journey_id=j2.id,
            timestamp=datetime.now(timezone.utc),
            event_type="SPEED_RESTRICTION",
            delay_minutes=6.0,
            severity="LOW",
            event_metadata={
                "data_source": "SIMULATION_DATA",
                "disclaimer": DATA_DISCLAIMER,
                "location": "MTJ-KOTA section",
                "cause": "Synthetic track maintenance speed restriction",
            },
        )
        db.add(e2)

        db.commit()
        print("Development/Simulation railway network seeded successfully:")
        print(f"- Stations: {len(station_map)}")
        print(f"- Routes: {total_routes}")
        print(f"- Trains: {total_trains}")
        print(f"- Active Journeys: 2")
        print(f"Notice: {DATA_DISCLAIMER}")

        return {
            "status": "seeded",
            "stations_count": len(station_map),
            "routes_count": total_routes,
            "trains_count": total_trains,
            "route_ids": created_route_ids,
            "is_simulation_data": True,
            "disclaimer": DATA_DISCLAIMER,
        }

    except Exception:
        db.rollback()
        raise
    finally:
        if own_session:
            db.close()


if __name__ == "__main__":
    seed_data()
