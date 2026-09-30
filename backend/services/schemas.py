from datetime import date, datetime
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator


class TrainRunningState(BaseModel):
    """
    Normalized train-running state schema used by both the synthetic simulator
    and external live railway APIs (e.g. RailRadar, NTES).
    """
    train_number: str = Field(description="Unique train identifier/number")
    journey_date: date = Field(description="Date of the train journey")
    train_name: str = Field(description="Name of the train")
    status: str = Field(description="Operational status (e.g., RUNNING, NOT-STARTED, COMPLETED)")
    current_station_code: Optional[str] = Field(default=None, description="Current or last passed station code")
    current_station_sequence: Optional[int] = Field(default=None, description="Sequence number of current station")
    current_delay_minutes: float = Field(default=0.0, description="Delay in minutes at current position")
    previous_station_code: Optional[str] = Field(default=None, description="Previous station code along route")
    next_station_code: Optional[str] = Field(default=None, description="Immediate next scheduled station/halt code")
    next_station_distance_km: Optional[float] = Field(default=None, description="Distance remaining to next station in km")
    segment_progress: float = Field(default=0.0, description="Progress on current inter-station segment (0.0 to 1.0)")
    speed_kmh: float = Field(default=0.0, description="Current instantaneous speed in km/h")
    timestamp: datetime = Field(description="Timestamp of this telemetry state update")
    source: Literal["simulator", "external_api"] = Field(description="Data source origin ('simulator' or 'external_api')")

    @field_validator("journey_date", mode="before")
    @classmethod
    def coerce_journey_date(cls, v: Any) -> Any:
        if isinstance(v, str):
            clean_str = v.strip()
            if "T" in clean_str:
                clean_str = clean_str.split("T")[0]
            elif " " in clean_str:
                clean_str = clean_str.split(" ")[0]
            return date.fromisoformat(clean_str)
        if isinstance(v, datetime):
            return v.date()
        return v

    @field_validator("timestamp", mode="before")
    @classmethod
    def coerce_timestamp(cls, v: Any) -> Any:
        if isinstance(v, str):
            clean_str = v.strip().replace("Z", "+00:00")
            return datetime.fromisoformat(clean_str)
        return v

    def display(self) -> None:
        """Pretty-prints key normalized state fields to stdout."""
        print("-" * 55)
        print(f"Train Number      : {self.train_number}")
        print(f"Train Name        : {self.train_name}")
        print(f"Journey Date      : {self.journey_date}")
        print(f"Status            : {self.status}")
        delay_sign = "+" if self.current_delay_minutes > 0 else ""
        print(f"Current Delay     : {delay_sign}{self.current_delay_minutes:.1f} min")
        seq_str = f" (Seq: {self.current_station_sequence})" if self.current_station_sequence is not None else ""
        print(f"Current Station   : {self.current_station_code or 'N/A'}{seq_str}")
        print(f"Previous Station  : {self.previous_station_code or 'N/A'}")
        dist_str = f" ({self.next_station_distance_km:.1f} km)" if self.next_station_distance_km is not None else ""
        print(f"Next Station      : {self.next_station_code or 'N/A'}{dist_str}")
        print(f"Segment Progress  : {self.segment_progress:.1%}")
        print(f"Speed             : {self.speed_kmh:.1f} km/h")
        print(f"Timestamp         : {self.timestamp.isoformat()}")
        print(f"Source            : {self.source}")
        print("-" * 55)
