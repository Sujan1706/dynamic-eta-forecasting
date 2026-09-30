"""
Data Source Abstraction and Mode Management.

Coordinates data-source selection between:
- SIMULATOR (default / fallback)
- LIVE_API (RailRadar on-demand queries)

Enforces:
1. Secure key handling (RAILRADAR_API_KEY read only from environment, never logged or leaked).
2. Transparent normalization into standard TrainRunningState.
3. Controlled fallback to simulator mode on timeouts, 401, 404, 429, 503, or malformed responses.
4. Zero continuous polling.
"""

from __future__ import annotations

from enum import Enum
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any
from sqlalchemy.orm import Session

from backend.services.schemas import TrainRunningState
from backend.simulator.engine import SimulatorStoppedError, TrainSimulator
from backend.services.railway_api_client import (
    RailRadarClient,
    RailwayAPIError,
    MissingApiKeyError,
    AuthenticationError,
    TrainNotFoundError,
    RateLimitExceededError,
    APITimeoutError,
    MalformedResponseError,
    ServiceUnavailableError,
    sanitize_secret,
)

logger = logging.getLogger("dynamic_eta.data_source")


class DataSourceMode(str, Enum):
    """Supported data-source modes for train telemetry."""
    SIMULATOR = "SIMULATOR"
    LIVE_API = "LIVE_API"


def get_configured_mode(override: Optional[Union[DataSourceMode, str]] = None) -> DataSourceMode:
    """
    Resolves the operational data-source mode.
    Priority:
    1. Explicit call/query override (if provided and valid)
    2. Environment variable 'DATA_SOURCE_MODE'
    3. Default: SIMULATOR
    """
    if override is not None:
        raw_val = override.value if isinstance(override, DataSourceMode) else str(override)
        clean = raw_val.strip().upper()
        if clean in ("LIVE_API", "EXTERNAL_API", "LIVE"):
            return DataSourceMode.LIVE_API
        elif clean in ("SIMULATOR", "SIM"):
            return DataSourceMode.SIMULATOR

    env_val = os.getenv("DATA_SOURCE_MODE", "SIMULATOR").strip().upper()
    if env_val in ("LIVE_API", "EXTERNAL_API", "LIVE"):
        return DataSourceMode.LIVE_API
    return DataSourceMode.SIMULATOR


@dataclass
class StateProviderResult:
    """
    Standard envelope returned by TrainStateProvider.
    Maintains clean telemetry separation and transparent fallback provenance.
    """
    state: TrainRunningState
    effective_mode: DataSourceMode
    requested_mode: DataSourceMode
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    active_events: List[Any] = field(default_factory=list)
    delay_history: List[float] = field(default_factory=list)
    delay_trend: float = 0.0


class TrainStateProvider:
    """
    Provides unified train running state from either synthetic simulator or live RailRadar API.
    Guarantees that downstream ETA models receive 100% compliant TrainRunningState objects.
    """

    def __init__(
        self,
        client: Optional[RailRadarClient] = None,
        default_mode: Optional[DataSourceMode] = None,
    ):
        self._client = client
        self._default_mode = default_mode

    @property
    def client(self) -> RailRadarClient:
        if self._client is None:
            self._client = RailRadarClient()
        return self._client

    def get_train_state(
        self,
        train_number: str,
        db: Session,
        sim: Optional[TrainSimulator] = None,
        mode_override: Optional[Union[DataSourceMode, str]] = None,
        fallback_on_error: bool = True,
    ) -> StateProviderResult:
        """
        Retrieves the train running state using the requested mode.
        If LIVE_API fails (timeout, 401, 404, 429, 503, malformed payload),
        automatically logs a controlled error and falls back to SIMULATOR mode.
        """
        clean_number = str(train_number).strip()
        requested_mode = get_configured_mode(mode_override or self._default_mode)

        if requested_mode == DataSourceMode.LIVE_API:
            try:
                # Query live API (on-demand only, zero continuous polling)
                live_state = self.client.fetch_live_train(clean_number)
                delay = float(live_state.current_delay_minutes or 0.0)

                return StateProviderResult(
                    state=live_state,
                    effective_mode=DataSourceMode.LIVE_API,
                    requested_mode=DataSourceMode.LIVE_API,
                    is_fallback=False,
                    fallback_reason=None,
                    active_events=[],
                    delay_history=[round(delay, 1)],
                    delay_trend=0.0,
                )

            except (
                APITimeoutError,
                AuthenticationError,
                TrainNotFoundError,
                RateLimitExceededError,
                ServiceUnavailableError,
                MalformedResponseError,
                MissingApiKeyError,
                RailwayAPIError,
            ) as exc:
                if not fallback_on_error:
                    raise

                # Controlled error logging (Strictly sanitizing any secret key occurrence)
                clean_err_msg = sanitize_secret(str(exc))
                logger.warning(
                    "Live railway API failed for train '%s' (%s). Controlled fallback to SIMULATOR.",
                    clean_number,
                    clean_err_msg,
                )

                # Fallback to simulator
                return self._get_simulator_state(
                    train_number=clean_number,
                    db=db,
                    sim=sim,
                    requested_mode=DataSourceMode.LIVE_API,
                    is_fallback=True,
                    fallback_reason=clean_err_msg,
                )

        # Default / configured SIMULATOR mode
        return self._get_simulator_state(
            train_number=clean_number,
            db=db,
            sim=sim,
            requested_mode=DataSourceMode.SIMULATOR,
            is_fallback=False,
            fallback_reason=None,
        )

    def _get_simulator_state(
        self,
        train_number: str,
        db: Session,
        sim: Optional[TrainSimulator],
        requested_mode: DataSourceMode,
        is_fallback: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> StateProviderResult:
        """Retrieves and packages dynamic running state from active simulator."""
        if not sim:
            raise RuntimeError("Train simulator is not available.")

        if not getattr(sim, "is_running", True):
            raise SimulatorStoppedError("Train simulator is currently stopped.")

        # Ensure train journey is loaded in simulator
        if train_number not in sim.journeys:
            sim.load_from_db(db, train_numbers=[train_number])

        if train_number not in sim.journeys:
            raise KeyError(
                f"Train '{train_number}' is not managed by simulator or not found in database."
            )

        journey = sim.journeys[train_number]
        state = sim.get_state(train_number)
        active_events = getattr(journey, "active_events", [])

        raw_history = getattr(journey, "delay_history", [state.current_delay_minutes])
        delay_history = (
            [round(float(x), 1) for x in raw_history]
            if raw_history
            else [round(state.current_delay_minutes, 1)]
        )

        if len(delay_history) >= 3:
            delay_trend = round((delay_history[-1] - delay_history[-3]) / 2.0, 2)
        elif len(delay_history) == 2:
            delay_trend = round(delay_history[-1] - delay_history[-2], 2)
        else:
            delay_trend = 0.0

        return StateProviderResult(
            state=state,
            effective_mode=DataSourceMode.SIMULATOR,
            requested_mode=requested_mode,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
            active_events=active_events,
            delay_history=delay_history,
            delay_trend=delay_trend,
        )
