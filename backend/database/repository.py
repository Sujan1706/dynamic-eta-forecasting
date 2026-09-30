from datetime import datetime, date, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.database.models import Station, Route, RouteStation, Train, Journey, Event


class StationRepository:
    @staticmethod
    def create(db: Session, code: str, name: str, latitude: float, longitude: float) -> Station:
        station = Station(
            code=code.upper().strip(),
            name=name.strip(),
            latitude=latitude,
            longitude=longitude,
        )
        db.add(station)
        db.commit()
        db.refresh(station)
        return station

    @staticmethod
    def get_by_id(db: Session, station_id: int) -> Optional[Station]:
        return db.query(Station).filter(Station.id == station_id).first()

    @staticmethod
    def get_by_code(db: Session, code: str) -> Optional[Station]:
        return db.query(Station).filter(Station.code == code.upper().strip()).first()

    @staticmethod
    def get_all(db: Session) -> List[Station]:
        return db.query(Station).order_by(Station.name.asc()).all()


class RouteRepository:
    @staticmethod
    def create(db: Session, source_station_id: int, destination_station_id: int) -> Route:
        route = Route(
            source_station_id=source_station_id,
            destination_station_id=destination_station_id,
        )
        db.add(route)
        db.commit()
        db.refresh(route)
        return route

    @staticmethod
    def get_by_id(db: Session, route_id: int) -> Optional[Route]:
        return db.query(Route).filter(Route.id == route_id).first()

    @staticmethod
    def add_station(
        db: Session,
        route_id: int,
        station_id: int,
        sequence: int,
        distance_from_source_km: float,
        scheduled_arrival: Optional[str] = None,
        scheduled_departure: Optional[str] = None,
        scheduled_stop_minutes: float = 0.0,
    ) -> RouteStation:
        rs = RouteStation(
            route_id=route_id,
            station_id=station_id,
            sequence=sequence,
            distance_from_source_km=distance_from_source_km,
            scheduled_arrival=scheduled_arrival,
            scheduled_departure=scheduled_departure,
            scheduled_stop_minutes=scheduled_stop_minutes,
        )
        db.add(rs)
        db.commit()
        db.refresh(rs)
        return rs

    @staticmethod
    def get_route_stations(db: Session, route_id: int) -> List[RouteStation]:
        return (
            db.query(RouteStation)
            .filter(RouteStation.route_id == route_id)
            .order_by(RouteStation.sequence.asc())
            .all()
        )


class TrainRepository:
    @staticmethod
    def create(db: Session, train_number: str, name: str, train_type: str, route_id: int) -> Train:
        train = Train(
            train_number=train_number.strip(),
            name=name.strip(),
            train_type=train_type.upper().strip(),
            route_id=route_id,
        )
        db.add(train)
        db.commit()
        db.refresh(train)
        return train

    @staticmethod
    def get_by_id(db: Session, train_id: int) -> Optional[Train]:
        return db.query(Train).filter(Train.id == train_id).first()

    @staticmethod
    def get_by_number(db: Session, train_number: str) -> Optional[Train]:
        return db.query(Train).filter(Train.train_number == train_number.strip()).first()

    @staticmethod
    def get_all(db: Session) -> List[Train]:
        return db.query(Train).order_by(Train.train_number.asc()).all()


class JourneyRepository:
    @staticmethod
    def create(
        db: Session,
        train_id: int,
        journey_date: date,
        current_station_id: Optional[int] = None,
        current_sequence: int = 1,
        current_delay_minutes: float = 0.0,
        status: str = "SCHEDULED",
    ) -> Journey:
        journey = Journey(
            train_id=train_id,
            journey_date=journey_date,
            current_station_id=current_station_id,
            current_sequence=current_sequence,
            current_delay_minutes=current_delay_minutes,
            status=status.upper(),
        )
        db.add(journey)
        db.commit()
        db.refresh(journey)
        return journey

    @staticmethod
    def get_by_id(db: Session, journey_id: int) -> Optional[Journey]:
        return db.query(Journey).filter(Journey.id == journey_id).first()

    @staticmethod
    def get_active(db: Session) -> List[Journey]:
        return db.query(Journey).filter(Journey.status == "RUNNING").all()

    @staticmethod
    def update_progress(
        db: Session,
        journey_id: int,
        current_station_id: Optional[int] = None,
        current_sequence: Optional[int] = None,
        current_delay_minutes: Optional[float] = None,
        status: Optional[str] = None,
    ) -> Optional[Journey]:
        journey = db.query(Journey).filter(Journey.id == journey_id).first()
        if not journey:
            return None
        if current_station_id is not None:
            journey.current_station_id = current_station_id
        if current_sequence is not None:
            journey.current_sequence = current_sequence
        if current_delay_minutes is not None:
            journey.current_delay_minutes = current_delay_minutes
        if status is not None:
            journey.status = status.upper()
        db.commit()
        db.refresh(journey)
        return journey


class EventRepository:
    @staticmethod
    def create(
        db: Session,
        journey_id: int,
        event_type: str,
        delay_minutes: float = 0.0,
        severity: Optional[str] = None,
        event_metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Event:
        event = Event(
            journey_id=journey_id,
            event_type=event_type.upper().strip(),
            delay_minutes=delay_minutes,
            severity=severity,
            event_metadata=event_metadata or {},
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    @staticmethod
    def get_by_journey(db: Session, journey_id: int) -> List[Event]:
        return (
            db.query(Event)
            .filter(Event.journey_id == journey_id)
            .order_by(Event.timestamp.desc())
            .all()
        )
