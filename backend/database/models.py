from datetime import datetime, date, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    JSON,
    Index,
)
from sqlalchemy.orm import relationship
from backend.database.connection import Base


class Station(Base):
    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    route_stops = relationship("RouteStation", back_populates="station")

    def __repr__(self):
        return f"<Station(id={self.id}, code='{self.code}', name='{self.name}')>"


class Route(Base):
    __tablename__ = "routes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_station_id = Column(Integer, ForeignKey("stations.id"), nullable=False, index=True)
    destination_station_id = Column(Integer, ForeignKey("stations.id"), nullable=False, index=True)

    source_station = relationship("Station", foreign_keys=[source_station_id])
    destination_station = relationship("Station", foreign_keys=[destination_station_id])
    route_stations = relationship(
        "RouteStation",
        back_populates="route",
        order_by="RouteStation.sequence",
        cascade="all, delete-orphan",
    )
    trains = relationship("Train", back_populates="route")

    def __repr__(self):
        return f"<Route(id={self.id}, source={self.source_station_id}, destination={self.destination_station_id})>"


class RouteStation(Base):
    __tablename__ = "route_stations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False, index=True)
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False, index=True)
    sequence = Column(Integer, nullable=False)
    distance_from_source_km = Column(Float, nullable=False)
    scheduled_arrival = Column(String(10), nullable=True)  # e.g., "16:55"
    scheduled_departure = Column(String(10), nullable=True)  # e.g., "17:00"
    scheduled_stop_minutes = Column(Float, default=0.0)

    __table_args__ = (
        UniqueConstraint("route_id", "sequence", name="uq_route_sequence"),
        UniqueConstraint("route_id", "station_id", name="uq_route_station"),
        Index("idx_route_station_seq", "route_id", "sequence"),
    )

    route = relationship("Route", back_populates="route_stations")
    station = relationship("Station", back_populates="route_stops")

    def __repr__(self):
        return (
            f"<RouteStation(route_id={self.route_id}, station_id={self.station_id}, "
            f"seq={self.sequence}, dist={self.distance_from_source_km}km)>"
        )


class Train(Base):
    __tablename__ = "trains"

    id = Column(Integer, primary_key=True, autoincrement=True)
    train_number = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    train_type = Column(String(50), nullable=False)  # "RAJDHANI", "SUPERFAST", "EXPRESS"
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False, index=True)

    route = relationship("Route", back_populates="trains")
    journeys = relationship("Journey", back_populates="train")

    def __repr__(self):
        return f"<Train(number='{self.train_number}', name='{self.name}', type='{self.train_type}')>"


class Journey(Base):
    __tablename__ = "journeys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    train_id = Column(Integer, ForeignKey("trains.id"), nullable=False, index=True)
    journey_date = Column(Date, nullable=False, default=date.today, index=True)
    current_station_id = Column(Integer, ForeignKey("stations.id"), nullable=True, index=True)
    current_sequence = Column(Integer, default=1, nullable=False)
    current_delay_minutes = Column(Float, default=0.0, nullable=False)
    status = Column(String(20), default="SCHEDULED", nullable=False, index=True)  # SCHEDULED, RUNNING, COMPLETED, CANCELLED

    train = relationship("Train", back_populates="journeys")
    current_station = relationship("Station", foreign_keys=[current_station_id])
    events = relationship("Event", back_populates="journey", cascade="all, delete-orphan")

    def __repr__(self):
        return (
            f"<Journey(id={self.id}, train_id={self.train_id}, "
            f"date={self.journey_date}, delay={self.current_delay_minutes}m, status='{self.status}')>"
        )


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    journey_id = Column(Integer, ForeignKey("journeys.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)  # "SIGNAL_HALT", "SPEED_RESTRICTION", "CONGESTION", "WEATHER"
    delay_minutes = Column(Float, default=0.0, nullable=False)
    severity = Column(String(50), nullable=True)  # e.g., "HIGH", "MEDIUM", "LOW" or numeric string
    event_metadata = Column("metadata", JSON, default=dict, nullable=False)

    journey = relationship("Journey", back_populates="events")

    def __repr__(self):
        return (
            f"<Event(id={self.id}, journey_id={self.journey_id}, "
            f"type='{self.event_type}', delay={self.delay_minutes}m)>"
        )
