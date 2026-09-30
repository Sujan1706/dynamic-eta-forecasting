from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any


class EventType(str, Enum):
    SIGNAL_HALT = "SIGNAL_HALT"
    CONGESTION = "CONGESTION"
    SPEED_RESTRICTION = "SPEED_RESTRICTION"
    UNSCHEDULED_HALT = "UNSCHEDULED_HALT"
    WEATHER = "WEATHER"


@dataclass
class SimulationEvent:
    """
    Represents an operational disruption or weather event injected into a train journey.
    Modifies the train running state in a controlled, reproducible, deterministic manner.
    """
    event_type: EventType
    delay_minutes: float = 0.0
    severity: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH"
    metadata: Dict[str, Any] = field(default_factory=dict)
    remaining_duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if isinstance(self.event_type, str):
            clean_type = self.event_type.upper().strip()
            try:
                self.event_type = EventType(clean_type)
            except ValueError:
                valid_types = [e.value for e in EventType]
                raise ValueError(
                    f"Invalid event_type '{self.event_type}'. Must be one of: {valid_types}"
                )

        # Ensure delay_minutes is non-negative float
        self.delay_minutes = max(0.0, float(self.delay_minutes))

        # Default remaining duration if not explicitly provided
        if self.remaining_duration_seconds <= 0.0:
            if self.delay_minutes > 0.0:
                self.remaining_duration_seconds = self.delay_minutes * 60.0
            else:
                self.remaining_duration_seconds = 600.0  # 10 minutes default duration

    @property
    def is_halt(self) -> bool:
        """Returns True if this event causes the train to halt completely (speed = 0)."""
        return self.event_type in (EventType.SIGNAL_HALT, EventType.UNSCHEDULED_HALT)

    def calculate_effective_speed(self, nominal_speed_kmh: float) -> float:
        """
        Determines the modified speed (km/h) under this event's influence.
        Controlled and deterministic.
        """
        if self.is_halt:
            return 0.0

        if self.event_type == EventType.SPEED_RESTRICTION:
            # Respect explicit speed_limit_kmh in metadata or default to 30 km/h
            speed_limit = float(self.metadata.get("speed_limit_kmh", 30.0))
            return max(0.0, min(nominal_speed_kmh, speed_limit))

        elif self.event_type == EventType.CONGESTION:
            # Respect speed_factor in metadata or default to 50% cruise speed
            factor = float(self.metadata.get("speed_factor", 0.5))
            return max(0.0, nominal_speed_kmh * factor)

        elif self.event_type == EventType.WEATHER:
            # Adverse weather (fog, heavy rain) restricts to 60% cruise speed, capped at 50 km/h
            factor = float(self.metadata.get("speed_factor", 0.6))
            max_weather_speed = float(self.metadata.get("max_speed_kmh", 50.0))
            return max(0.0, min(nominal_speed_kmh * factor, max_weather_speed))

        return nominal_speed_kmh
