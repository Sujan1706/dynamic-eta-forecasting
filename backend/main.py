import os
import math
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, Depends, HTTPException, Query, Path, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError

from backend.database.connection import get_db, SessionLocal
from backend.database.init_db import init_db
from backend.database.seed import seed_data
from backend.database.models import Train, Route, RouteStation, Station, Journey
from backend.services.schemas import TrainRunningState
from backend.services.baseline_eta import BaselineETAService, BaselineETAPrediction
from backend.services.multi_station_eta import (
    MultiStationETAService,
    StationETAPrediction,
    SegmentPrediction,
)
from backend.features.feature_builder import FeatureBuilder, TrainFeatures
from backend.simulator.events import EventType, SimulationEvent
from backend.simulator.engine import TrainSimulator, SimulatorStoppedError
from pathlib import Path as FilePath
from backend.ml.evaluate_model import (
    load_latest_metrics,
    DEFAULT_METRICS_OUTPUT_PATH,
    DEFAULT_METADATA_PATH,
)
from backend.services.railway_api_client import (
    sanitize_secret,
    RailwayAPIError,
    MissingApiKeyError,
    AuthenticationError,
    TrainNotFoundError,
    RateLimitExceededError,
    APITimeoutError,
    MalformedResponseError,
    ServiceUnavailableError,
)
from backend.services.data_source import (
    DataSourceMode,
    TrainStateProvider,
    StateProviderResult,
    get_configured_mode,
)
from backend.services.live_train_service import (
    LiveTrainLookupService,
    LiveTrainResult,
    ConfidenceRangeDetails,
)
from backend.services.demo_scenario import (
    DemoScenarioManager,
    DemoScenarioResponse,
    DemoActionResult,
    DEMO_ACTIONS,
    DEMO_TRAIN_NUMBER,
)



# ---------------------------------------------------------------------------
# Pydantic Response Schemas for API Endpoints
# ---------------------------------------------------------------------------

class ConfidenceRange(BaseModel):
    """Heuristic confidence interval around the ML predicted arrival time."""
    lower_bound: datetime = Field(description="Earliest expected arrival datetime")
    upper_bound: datetime = Field(description="Latest expected arrival datetime")
    margin_minutes: float = Field(description="Heuristic uncertainty window in minutes (+/-)")
    lower_bound_minutes: Optional[float] = Field(default=None, description="Minutes to lower bound from observation time")
    upper_bound_minutes: Optional[float] = Field(default=None, description="Minutes to upper bound from observation time")


class UpcomingStationETA(BaseModel):
    """Forecast and schedule breakdown for an upcoming station stop."""
    station_code: str = Field(description="Station code (e.g. CNB, PRYJ)")
    station_name: Optional[str] = Field(default=None, description="Station name")
    station_sequence: int = Field(description="Sequence order of station along route")
    distance_to_go_km: float = Field(description="Remaining route distance to this station in km")
    segments_ahead: int = Field(description="Number of inter-station segments ahead (1=next stop, etc.)")
    intermediate_halts: int = Field(default=0, description="Number of intermediate scheduled halts")

    # Core ETAs (Requirements 6, 7, 8)
    scheduled_eta: datetime = Field(description="Timetable scheduled arrival datetime")
    baseline_eta: datetime = Field(description="Baseline heuristic predicted arrival datetime")
    ml_eta: datetime = Field(description="XGBoost chained predicted arrival datetime")
    predicted_eta: Optional[datetime] = Field(default=None, description="Alias for ml_eta")

    # Remaining minutes
    scheduled_remaining_minutes: Optional[float] = Field(default=None, description="Scheduled minutes to arrival from observation time")
    baseline_remaining_minutes: float = Field(description="Baseline predicted minutes to arrival")
    predicted_remaining_minutes: float = Field(description="ML predicted minutes to arrival")

    # Confidence Range (Requirement 9)
    confidence_lower_bound: datetime = Field(description="Confidence lower bound datetime")
    confidence_upper_bound: datetime = Field(description="Confidence upper bound datetime")
    confidence_range: ConfidenceRange = Field(description="Confidence interval details")

    # Segment predictions (Requirement 10)
    segment_predictions: List[SegmentPrediction] = Field(
        default_factory=list, description="Granular chained segment breakdown leading to this station"
    )


class RouteStationInfo(BaseModel):
    """Timetable stop details for a station along the train route."""
    sequence: int = Field(description="1-based sequence order along route")
    station_code: str = Field(description="Station code")
    station_name: str = Field(description="Station name")
    distance_from_source_km: float = Field(description="Cumulative track distance from route origin in km")
    scheduled_arrival: Optional[str] = Field(default=None, description="Timetable arrival time (HH:MM)")
    scheduled_departure: Optional[str] = Field(default=None, description="Timetable departure time (HH:MM)")
    scheduled_stop_minutes: Optional[float] = Field(default=0.0, description="Scheduled dwell buffer in minutes")


class TrainDetailResponse(BaseModel):
    """
    Comprehensive train response schema with live running state and multi-station ML ETAs.
    Maintains 100% backward compatibility with previous contract.
    """
    # Core identifiers & route stations (existing contract)
    id: int = Field(description="Database primary key")
    train_number: str = Field(description="Unique train identifier/number")
    name: str = Field(description="Train name")
    train_type: str = Field(description="Type of train (e.g. RAJDHANI, EXPRESS)")
    route_id: int = Field(description="Route primary key")
    route_stations: List[RouteStationInfo] = Field(default_factory=list, description="All stations on the route")

    # 1. Current train state
    current_state: Optional[TrainRunningState] = Field(default=None, description="Current normalized train running state")

    # 2. Current station
    current_station: Optional[str] = Field(default=None, description="Current or last-passed station code")

    # 3. Current delay
    current_delay: float = Field(default=0.0, description="Current delay in minutes")
    current_delay_minutes: float = Field(default=0.0, description="Alias for current_delay")

    # 4. Current timestamp
    current_timestamp: Optional[datetime] = Field(default=None, description="Current telemetry or observation timestamp")

    # 5. Upcoming stations
    upcoming_stations: List[UpcomingStationETA] = Field(default_factory=list, description="Forecasts for all upcoming stations")

    # Primary forecast for immediate next upcoming station (Requirements 6-9)
    # 6. Scheduled ETA
    scheduled_eta: Optional[datetime] = Field(default=None, description="Scheduled ETA to immediate next upcoming station")

    # 7. Baseline ETA
    baseline_eta: Optional[datetime] = Field(default=None, description="Baseline ETA to immediate next upcoming station")

    # 8. ML ETA
    ml_eta: Optional[datetime] = Field(default=None, description="ML ETA to immediate next upcoming station")

    # 9. Confidence range
    confidence_range: Optional[ConfidenceRange] = Field(default=None, description="Confidence range for immediate next upcoming station")

    # 10. Segment predictions
    segment_predictions: List[SegmentPrediction] = Field(
        default_factory=list, description="Chained segment transit predictions for the upcoming journey"
    )

    # 11. Operational events & delay metrics
    active_events: List[Dict[str, Any]] = Field(
        default_factory=list, description="Currently active operational disruption events"
    )
    delay_trend: float = Field(
        default=0.0, description="Delay slope over recent checkpoints (+ increasing, - recovering)"
    )
    delay_history: List[float] = Field(
        default_factory=list, description="Historical checkpoint delays in minutes"
    )
    data_source_mode: Optional[str] = Field(
        default="SIMULATOR", description="Effective telemetry data source mode ('SIMULATOR' or 'LIVE_API')"
    )
    is_fallback: bool = Field(
        default=False, description="True if system performed a controlled fallback to SIMULATOR"
    )
    fallback_reason: Optional[str] = Field(
        default=None, description="Sanitized rationale when fallback occurred"
    )

    # Freshness and cache metadata (Requirements 1, 3, 4)
    last_updated: Optional[datetime] = Field(
        default=None, description="Timestamp when telemetry was last observed/updated"
    )
    is_cached: bool = Field(
        default=False, description="True if telemetry was served from in-process cache"
    )
    cache_age_seconds: Optional[float] = Field(
        default=None, description="Age of cached telemetry in seconds"
    )


class SingleStationETAResponse(BaseModel):
    """
    Response schema for single upcoming station ETA forecast.
    Contains ML ETA, Baseline ETA, confidence bounds, operational delay, and fallback status.
    """
    train_id: int = Field(description="Database train primary key")
    train_number: str = Field(description="Unique train identifier/number")
    target_station: str = Field(description="Target upcoming station code")
    current_station: Optional[str] = Field(default=None, description="Current or last passed station code")
    scheduled_eta: Optional[datetime] = Field(default=None, description="Timetable scheduled arrival datetime")
    baseline_eta: Optional[datetime] = Field(default=None, description="Baseline heuristic arrival datetime")
    ml_eta: Optional[datetime] = Field(default=None, description="XGBoost predicted arrival datetime")
    confidence_lower: Optional[datetime] = Field(default=None, description="Confidence lower bound datetime")
    confidence_upper: Optional[datetime] = Field(default=None, description="Confidence upper bound datetime")
    current_delay_minutes: float = Field(default=0.0, description="Current train delay in minutes")
    prediction_timestamp: datetime = Field(description="Timestamp when prediction was calculated")
    model_version: str = Field(description="Version of the ML prediction model")
    ml_status: str = Field(default="AVAILABLE", description="Status of ML prediction: 'AVAILABLE' or 'UNAVAILABLE'")
    ml_error: Optional[str] = Field(default=None, description="Error details if ML prediction was unavailable")

    # Data source provenance
    data_source_mode: Optional[str] = Field(
        default="SIMULATOR", description="Effective telemetry data source mode ('SIMULATOR' or 'LIVE_API')"
    )
    is_fallback: bool = Field(
        default=False, description="True if system performed a controlled fallback to SIMULATOR"
    )
    fallback_reason: Optional[str] = Field(
        default=None, description="Sanitized rationale when fallback occurred"
    )

    # Optional fields for backward compatibility
    train_name: Optional[str] = Field(default=None, description="Name of the train")
    prediction: Optional[Dict[str, Any]] = Field(default=None, description="Baseline prediction details")
    features: Optional[Dict[str, Any]] = Field(default=None, description="Extracted 11 PRD features")
    feature_provenance: Optional[Dict[str, str]] = Field(
        default=None, description="Feature source categorization ('live data', 'historical data', 'simulator/default source')"
    )
    estimated_features: Optional[List[str]] = Field(
        default=None, description="List of features using default or estimated values"
    )
    segments_ahead: Optional[int] = Field(default=None, description="Number of segments ahead")
    distance_to_go_km: Optional[float] = Field(default=None, description="Distance remaining to target station in km")



class StationArrivalItem(BaseModel):
    """Details for a single train approaching a target station."""
    train_id: int = Field(description="Database primary key of the train")
    train_number: str = Field(description="Unique train identifier/number")
    train_name: str = Field(description="Name of the train")
    train_type: str = Field(description="Type of train (e.g. RAJDHANI, EXPRESS)")
    origin_station_code: Optional[str] = Field(default=None, description="Origin station code")
    destination_station_code: Optional[str] = Field(default=None, description="Terminus destination station code")
    current_station: Optional[str] = Field(default=None, description="Current or last passed station code")
    current_delay_minutes: float = Field(default=0.0, description="Current operational delay in minutes")
    distance_to_go_km: float = Field(description="Remaining route distance to target station in km")
    segments_ahead: int = Field(description="Number of inter-station segments remaining to target station")

    # ETAs (Requirements 2, 3, 4)
    scheduled_eta: Optional[datetime] = Field(default=None, description="Timetable scheduled arrival datetime")
    baseline_eta: Optional[datetime] = Field(default=None, description="Baseline heuristic arrival datetime")
    ml_eta: Optional[datetime] = Field(default=None, description="XGBoost predicted arrival datetime")

    # Confidence bounds (Requirement 5)
    confidence_lower: Optional[datetime] = Field(default=None, description="Confidence lower bound datetime")
    confidence_upper: Optional[datetime] = Field(default=None, description="Confidence upper bound datetime")
    confidence_range: Optional[ConfidenceRange] = Field(default=None, description="Confidence interval details")

    # Operational status
    ml_status: str = Field(default="AVAILABLE", description="Status of ML forecast ('AVAILABLE' or 'UNAVAILABLE')")
    minutes_to_arrival: Optional[float] = Field(default=None, description="Predicted remaining minutes until arrival")


class StationArrivalsResponse(BaseModel):
    """Live arrivals board response for a station."""
    station_code: str = Field(description="Target station code")
    station_name: str = Field(description="Full name of station")
    current_timestamp: datetime = Field(description="Current observation or simulation timestamp")
    window_hours: Optional[float] = Field(default=None, description="Time window in hours for upcoming arrivals")
    total_arrivals: int = Field(description="Number of approaching trains found in window")
    arrivals: List[StationArrivalItem] = Field(default_factory=list, description="Approaching trains sorted primarily by ML ETA")


class EventDetails(BaseModel):
    """Details of the injected operational event."""
    event_type: str = Field(description="Operational event type")
    delay_minutes: float = Field(description="Injected delay in minutes")
    severity: str = Field(description="Severity rating: LOW, MEDIUM, or HIGH")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operational event metadata")
    timestamp: datetime = Field(description="Timestamp of event creation/injection")


class EventInjectionRequest(BaseModel):
    """Payload for POST /simulate/event."""
    train_id: Union[int, str] = Field(description="Train identifier (e.g. '12302' or integer ID)")
    event_type: str = Field(
        description="Event type: SIGNAL_HALT, CONGESTION, SPEED_RESTRICTION, UNSCHEDULED_HALT, WEATHER"
    )
    delay_minutes: float = Field(ge=0.0, description="Delay duration in minutes caused by this event")
    severity: Optional[str] = Field(default="MEDIUM", description="Severity level: LOW, MEDIUM, or HIGH")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary event metadata dict")

    @field_validator("event_type", mode="before")
    @classmethod
    def validate_event_type(cls, v: Any) -> str:
        if isinstance(v, EventType):
            return v.value
        if isinstance(v, str):
            clean_type = v.upper().strip()
            valid_types = [e.value for e in EventType]
            if clean_type not in valid_types:
                raise ValueError(
                    f"Invalid event_type '{v}'. Supported event types: {', '.join(valid_types)}"
                )
            return clean_type
        raise ValueError(f"Invalid event_type '{v}'. Must be a string.")

    @field_validator("severity", mode="before")
    @classmethod
    def validate_severity(cls, v: Any) -> str:
        if isinstance(v, str):
            clean_sev = v.upper().strip()
            if clean_sev in ("LOW", "MEDIUM", "HIGH"):
                return clean_sev
        return "MEDIUM"


class EventInjectionResponse(BaseModel):
    """Response returned upon successful event injection."""
    event: EventDetails = Field(description="Details of the injected event")
    train_state: TrainRunningState = Field(description="Updated normalized train running state")
    target_station: Optional[str] = Field(default=None, description="Next target station code for ETAs")
    baseline_eta: Optional[datetime] = Field(default=None, description="Updated baseline ETA to next target station")
    updated_baseline_eta: Optional[datetime] = Field(default=None, description="Alias for baseline_eta")
    ml_eta: Optional[datetime] = Field(default=None, description="Updated ML ETA to next target station")
    updated_ml_eta: Optional[datetime] = Field(default=None, description="Alias for ml_eta")
    confidence_range: Optional[ConfidenceRange] = Field(default=None, description="Updated confidence interval")
    ml_status: str = Field(default="AVAILABLE", description="Status of ML forecast ('AVAILABLE' or 'UNAVAILABLE')")
    message: str = Field(default="Event injected successfully", description="Status message")


class ModelMetricsResponse(BaseModel):
    """Response schema for GET /model/metrics."""
    status: str = Field(default="AVAILABLE", description="Availability status: 'AVAILABLE' or 'UNAVAILABLE'")
    is_available: bool = Field(default=True, description="Whether evaluation metrics are loaded and available")
    model_name: Optional[str] = Field(default=None, description="Name of the model")
    model_version: Optional[str] = Field(default=None, description="Model release version")
    training_timestamp: Optional[Union[datetime, str]] = Field(default=None, description="Timestamp when model was trained")
    total_training_samples: Optional[int] = Field(default=None, description="Total number of training samples")
    test_samples: Optional[int] = Field(default=None, description="Total number of evaluated test samples")
    total_journeys: Optional[int] = Field(default=None, description="Total number of journeys in dataset")
    train_journeys: Optional[int] = Field(default=None, description="Number of training journeys")
    test_journeys: Optional[int] = Field(default=None, description="Number of held-out test journeys")
    dataset_info: Optional[Dict[str, Any]] = Field(default=None, description="Dataset metadata and split details")
    summary: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation summary statistics")
    baseline_mae: Optional[float] = Field(default=None, description="Overall baseline heuristic Mean Absolute Error (minutes)")
    ml_mae: Optional[float] = Field(default=None, description="Overall ML model Mean Absolute Error (minutes)")
    baseline_rmse: Optional[float] = Field(default=None, description="Overall baseline heuristic Root Mean Squared Error (minutes)")
    ml_rmse: Optional[float] = Field(default=None, description="Overall ML model Root Mean Squared Error (minutes)")
    metrics_by_horizon: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation metrics grouped by station horizon")
    by_horizon: Optional[Dict[str, Any]] = Field(default=None, description="Alias for metrics_by_horizon")
    metrics_by_disruption_status: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation metrics grouped by disruption status")
    metrics_by_disruption: Optional[Dict[str, Any]] = Field(default=None, description="Alias for metrics_by_disruption_status")
    by_disruption: Optional[Dict[str, Any]] = Field(default=None, description="Alias for metrics_by_disruption_status")
    cross_tabulation: Optional[Dict[str, Any]] = Field(default=None, description="Cross-tabulated metrics (horizon x disruption)")
    overall: Optional[Dict[str, Any]] = Field(default=None, description="Complete overall comparison stats")
    evaluation_timestamp: Optional[Union[datetime, str]] = Field(default=None, description="Timestamp when evaluation was executed")
    message: Optional[str] = Field(default=None, description="Status or explanatory message")
    disclaimer: Optional[str] = Field(default=None, description="Dataset synthetic disclaimer")


class LiveTrainResponse(LiveTrainResult):
    """Structured response for live train lookup."""
    pass


class LiveTrainErrorResponse(BaseModel):
    """Structured error schema when live train data is unavailable."""
    error: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    detail: Optional[str] = Field(default=None, description="Sanitized diagnostic failure details")
    train_number: str = Field(description="Requested train number")
    simulator_mode_available: bool = Field(default=True, description="Indicates simulator fallback is available")
    simulator_available: bool = Field(default=True, description="Alias indicating simulator availability")
    simulator_url: Optional[str] = Field(default=None, description="URL to access train in simulator mode")
    suggested_action: str = Field(
        default="Simulator mode is available. Access /train/{train_number} to view simulated journey.",
        description="Suggested action for user or client",
    )


# ---------------------------------------------------------------------------

# Global Service Singletons & Dependency Injection
# ---------------------------------------------------------------------------

simulator: Optional[TrainSimulator] = None
baseline_service = BaselineETAService()
feature_builder = FeatureBuilder()
multi_station_service: Optional[MultiStationETAService] = None


def get_model_version(eta_service: Optional[MultiStationETAService] = None) -> str:
    """Retrieves the trained ML model version, falling back to '1.0.0'."""
    try:
        if eta_service and eta_service.predictor and eta_service.predictor.metadata:
            return eta_service.predictor.metadata.model_version
    except Exception:
        pass
    return "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite database, seed network if empty, and loads the simulator."""
    global simulator
    init_db()
    db = SessionLocal()
    try:
        # Seed if empty
        if not db.query(Station).first():
            seed_data(db)

        # Initialize simulator
        simulator = TrainSimulator(simulation_speed=1.0, db=db)
        simulator.load_from_db(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title="Dynamic Train ETA Forecasting Engine",
    description="API for dynamic train running states, simulation control, and multi-station ML ETA predictions",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Gracefully handles database operational or connection errors with a structured 503 response."""
    clean_msg = sanitize_secret(str(exc))
    return JSONResponse(
        status_code=503,
        content={
            "error": "DATABASE_UNAVAILABLE",
            "message": "The database is temporarily unavailable or experiencing connection issues.",
            "detail": clean_msg,
            "status_code": 503,
            "suggested_action": "Verify database file accessibility and retry.",
        },
    )


@app.exception_handler(SimulatorStoppedError)
async def simulator_stopped_exception_handler(request: Request, exc: SimulatorStoppedError):
    """Gracefully handles simulator stopped or paused state with a structured 503 response."""
    clean_msg = sanitize_secret(str(exc))
    return JSONResponse(
        status_code=503,
        content={
            "error": "SIMULATOR_STOPPED",
            "message": "Train simulator is currently stopped or unavailable.",
            "detail": clean_msg,
            "status_code": 503,
            "suggested_action": "Start or resume the simulator engine.",
        },
    )


@app.exception_handler(RailwayAPIError)
async def railway_api_exception_handler(request: Request, exc: RailwayAPIError):
    """Maps Railway API client errors into structured responses indicating simulator fallback."""
    clean_msg = sanitize_secret(str(exc))
    status_code = 503
    err_code = "LIVE_DATA_UNAVAILABLE"

    if isinstance(exc, TrainNotFoundError):
        status_code = 404
        err_code = "TRAIN_NOT_FOUND"
    elif isinstance(exc, RateLimitExceededError):
        status_code = 429
        err_code = "RATE_LIMIT_EXCEEDED"
    elif isinstance(exc, (AuthenticationError, MissingApiKeyError)):
        status_code = 503
        err_code = "AUTHENTICATION_ERROR"
    elif isinstance(exc, APITimeoutError):
        status_code = 503
        err_code = "NETWORK_TIMEOUT"
    elif isinstance(exc, MalformedResponseError):
        status_code = 503
        err_code = "MALFORMED_EXTERNAL_DATA"
    elif isinstance(exc, ServiceUnavailableError):
        status_code = 503
        err_code = "SERVICE_UNAVAILABLE"

    return JSONResponse(
        status_code=status_code,
        content={
            "error": err_code,
            "message": clean_msg,
            "detail": clean_msg,
            "simulator_mode_available": True,
            "simulator_available": True,
            "suggested_action": "External railway telemetry is unavailable. Simulator mode is available as a fallback.",
            "status_code": status_code,
        },
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Enriches HTTPException with structured machine-readable error codes while preserving detail."""
    if isinstance(exc.detail, dict):
        content = {
            "error": exc.detail.get("error", "HTTP_ERROR"),
            "message": exc.detail.get("message", str(exc.detail.get("detail", "An error occurred"))),
            "detail": exc.detail.get("detail", str(exc.detail)),
            "status_code": exc.status_code,
            **exc.detail,
        }
    else:
        err_code = "HTTP_ERROR"
        if exc.status_code == 404:
            err_code = "NOT_FOUND"
            if "train" in str(exc.detail).lower():
                err_code = "TRAIN_NOT_FOUND"
            elif "station" in str(exc.detail).lower():
                err_code = "STATION_NOT_FOUND"
        elif exc.status_code == 400:
            err_code = "BAD_REQUEST"
            if "not on the route" in str(exc.detail).lower() or "route" in str(exc.detail).lower():
                err_code = "STATION_NOT_ON_ROUTE"
        elif exc.status_code == 503:
            err_code = "SERVICE_UNAVAILABLE"

        content = {
            "error": err_code,
            "message": str(exc.detail),
            "detail": str(exc.detail),
            "status_code": exc.status_code,
        }
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=exc.headers,
    )


def get_active_simulator() -> TrainSimulator:
    global simulator
    if simulator is None:
        db = SessionLocal()
        try:
            simulator = TrainSimulator(simulation_speed=1.0, db=db)
            simulator.load_from_db(db)
        finally:
            db.close()
    return simulator


def get_multi_station_service() -> MultiStationETAService:
    global multi_station_service
    if multi_station_service is None:
        multi_station_service = MultiStationETAService(
            baseline_service=baseline_service,
            feature_builder=feature_builder,
        )
    return multi_station_service


state_provider_instance: Optional[TrainStateProvider] = None


def get_state_provider() -> TrainStateProvider:
    """Dependency provider for TrainStateProvider."""
    global state_provider_instance
    if state_provider_instance is None:
        state_provider_instance = TrainStateProvider()
    return state_provider_instance


live_lookup_service_instance: Optional[LiveTrainLookupService] = None


def get_live_lookup_service(
    eta_service: MultiStationETAService = Depends(get_multi_station_service),
) -> LiveTrainLookupService:
    """Dependency provider for LiveTrainLookupService."""
    global live_lookup_service_instance
    if live_lookup_service_instance is None:
        live_lookup_service_instance = LiveTrainLookupService(
            feature_builder=feature_builder,
            baseline_service=baseline_service,
            multi_station_service=eta_service,
        )
    return live_lookup_service_instance


demo_manager_instance: Optional[DemoScenarioManager] = None


def get_demo_manager(
    eta_service: MultiStationETAService = Depends(get_multi_station_service),
) -> DemoScenarioManager:
    """Dependency provider for DemoScenarioManager."""
    global demo_manager_instance
    if demo_manager_instance is None:
        demo_manager_instance = DemoScenarioManager(
            baseline_service=baseline_service,
            multi_station_service=eta_service,
        )
    return demo_manager_instance



def get_metrics_filepath() -> FilePath:
    """Returns the default path to the evaluation metrics JSON file."""
    return DEFAULT_METRICS_OUTPUT_PATH



def get_metadata_filepath() -> FilePath:
    """Returns the default path to the model metadata JSON file."""
    return DEFAULT_METADATA_PATH


def find_train(db: Session, train_identifier: str) -> Optional[Train]:
    """Finds a train by its unique train_number or integer primary key."""
    if not train_identifier:
        return None
    clean_id = train_identifier.strip()
    if not clean_id:
        return None
    query = (
        db.query(Train)
        .options(
            joinedload(Train.route)
            .joinedload(Route.route_stations)
            .joinedload(RouteStation.station)
        )
    )
    if clean_id.isdigit():
        train = query.filter((Train.train_number == clean_id) | (Train.id == int(clean_id))).first()
    else:
        train = query.filter(Train.train_number == clean_id).first()
    return train


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint confirming API availability."""
    return {
        "status": "ok",
        "service": "dynamic-eta-forecasting",
        "version": "0.1.0",
    }


@app.get(
    "/system/data-source",
    summary="Get current data source telemetry configuration",
    description="Returns current configured telemetry data source mode and RailRadar key status (strictly masked).",
)
def get_data_source_config(
    provider: TrainStateProvider = Depends(get_state_provider),
):
    mode = get_configured_mode()
    has_key = bool(os.getenv("RAILRADAR_API_KEY"))
    effective_mode = mode.value if (mode != DataSourceMode.LIVE_API or has_key) else DataSourceMode.SIMULATOR.value
    cache_stats = provider.client.cache.stats() if provider.client and provider.client.cache else {}
    ttl = provider.client.cache.ttl_seconds if provider.client and provider.client.cache else None
    return {
        "configured_mode": mode.value,
        "effective_mode": effective_mode,
        "supported_modes": [m.value for m in DataSourceMode],
        "has_api_key": has_key,
        "fallback_to_simulator": True,
        "cache_active": bool(provider.client and provider.client.cache),
        "cache_ttl_seconds": ttl,
        "cache": cache_stats,
    }



@app.get("/trains")
def list_trains(
    data_source: Optional[str] = Query(None, description="Data source mode: SIMULATOR or LIVE_API"),
    db: Session = Depends(get_db),
    sim: TrainSimulator = Depends(get_active_simulator),
    provider: TrainStateProvider = Depends(get_state_provider),
    eta_service: MultiStationETAService = Depends(get_multi_station_service),
):
    """Lists all configured trains with their current dynamic running state and next-station ETAs."""
    trains = (
        db.query(Train)
        .options(
            joinedload(Train.route)
            .joinedload(Route.route_stations)
            .joinedload(RouteStation.station)
        )
        .order_by(Train.train_number.asc())
        .all()
    )

    result = []
    for train in trains:
        running_state = None
        ml_eta = None
        baseline_eta = None
        confidence_range = None
        delay_trend = 0.0
        delay_history = []
        current_station = None
        next_station = None
        current_delay = 0.0
        effective_mode = "SIMULATOR"
        is_fallback = False

        try:
            p_res = provider.get_train_state(
                train_number=train.train_number,
                db=db,
                sim=sim,
                mode_override=data_source,
                fallback_on_error=True,
            )
            state = p_res.state
            running_state = state.model_dump(mode="json")
            current_station = state.current_station_code
            next_station = state.next_station_code
            current_delay = state.current_delay_minutes
            delay_history = p_res.delay_history
            delay_trend = p_res.delay_trend
            effective_mode = p_res.effective_mode.value
            is_fallback = p_res.is_fallback

            # Predict ETA to immediate next station
            if next_station:
                try:
                    pred = eta_service.predict_station_eta(state, next_station, db=db)
                    ml_eta = pred.predicted_eta.isoformat() if pred.predicted_eta else None
                    baseline_eta = pred.baseline_eta.isoformat() if pred.baseline_eta else None
                    confidence_range = {
                        "lower_bound": pred.confidence_lower_bound.isoformat(),
                        "upper_bound": pred.confidence_upper_bound.isoformat(),
                        "margin_minutes": pred.uncertainty_margin_minutes,
                    }
                except Exception:
                    pass
        except Exception:
            pass

        result.append({
            "id": train.id,
            "train_number": train.train_number,
            "name": train.name,
            "train_type": train.train_type,
            "route_id": train.route_id,
            "current_state": running_state,
            "current_station": current_station,
            "next_station": next_station,
            "current_delay_minutes": current_delay,
            "ml_eta": ml_eta,
            "baseline_eta": baseline_eta,
            "confidence_range": confidence_range,
            "delay_trend": delay_trend,
            "delay_history": delay_history,
            "data_source_mode": effective_mode,
            "data_source": getattr(state, "source", None) or effective_mode,
            "is_fallback": is_fallback,
            "last_updated": state.timestamp.isoformat() if state and state.timestamp else None,
        })

    return {"total": len(result), "trains": result}


@app.get(
    "/live/train/{train_number}",
    response_model=LiveTrainResponse,
    summary="Live train lookup with dynamic Baseline and ML ETA",
    description=(
        "Executes live train lookup flow: RailRadar live API -> normalized TrainRunningState "
        "-> FeatureBuilder -> Baseline ETA + ML ETA for upcoming station. "
        "If live telemetry is unavailable, returns a structured error indicating simulator mode availability."
    ),
    responses={
        200: {"model": LiveTrainResponse, "description": "Live train state and dynamic predicted ETAs"},
        404: {"model": LiveTrainErrorResponse, "description": "Train not found in live telemetry"},
        503: {"model": LiveTrainErrorResponse, "description": "Live railway API unavailable (simulator mode available)"},
    },
)
def get_live_train_lookup(
    train_number: str = Path(..., description="Unique train identifier/number (e.g. 12302)"),
    db: Session = Depends(get_db),
    lookup_service: LiveTrainLookupService = Depends(get_live_lookup_service),
):
    """
    Live train lookup endpoint.
    Retrieves real-time telemetry from RailRadar, normalizes into TrainRunningState,
    builds the feature vector, and calculates Baseline ETA and ML ETA for the next upcoming station.
    If live data is unavailable, returns a structured error indicating simulator mode availability without crashing.
    """
    clean_number = str(train_number).strip()
    try:
        result = lookup_service.lookup_live_train(clean_number, db=db)
        return result
    except TrainNotFoundError as exc:
        clean_msg = sanitize_secret(str(exc))
        return JSONResponse(
            status_code=404,
            content={
                "error": "TRAIN_NOT_FOUND",
                "message": f"Train '{clean_number}' not found in live railway API. Simulator mode is available.",
                "detail": clean_msg,
                "train_number": clean_number,
                "simulator_mode_available": True,
                "simulator_available": True,
                "simulator_url": f"/train/{clean_number}",
                "suggested_action": f"Train '{clean_number}' is not tracked in live API. Switch to simulator mode at /train/{clean_number}.",
            },
        )
    except (
        APITimeoutError,
        AuthenticationError,
        RateLimitExceededError,
        ServiceUnavailableError,
        MalformedResponseError,
        MissingApiKeyError,
        RailwayAPIError,
    ) as exc:
        clean_msg = sanitize_secret(str(exc))
        return JSONResponse(
            status_code=503,
            content={
                "error": "LIVE_DATA_UNAVAILABLE",
                "message": f"Live railway data is currently unavailable for train '{clean_number}'. Simulator mode is available.",
                "detail": clean_msg,
                "train_number": clean_number,
                "simulator_mode_available": True,
                "simulator_available": True,
                "simulator_url": f"/train/{clean_number}",
                "suggested_action": f"Live railway telemetry is unavailable. Simulator mode is available as a fallback via GET /train/{clean_number} or GET /train/{clean_number}?data_source=SIMULATOR.",
            },
        )
    except Exception as exc:
        clean_msg = sanitize_secret(str(exc))
        return JSONResponse(
            status_code=503,
            content={
                "error": "LIVE_LOOKUP_ERROR",
                "message": f"An error occurred while fetching live telemetry for train '{clean_number}'. Simulator mode is available.",
                "detail": clean_msg,
                "train_number": clean_number,
                "simulator_mode_available": True,
                "simulator_available": True,
                "simulator_url": f"/train/{clean_number}",
                "suggested_action": f"Simulator mode is available at /train/{clean_number}.",
            },
        )


@app.get(
    "/train/{train_id}",
    response_model=TrainDetailResponse,
    summary="Get train details with multi-station ETA predictions",
    description=(
        "Retrieves full details for a train, including current running state, "
        "current station, current delay, upcoming stations, scheduled ETA, "
        "baseline ETA, ML ETA, confidence range, and segment predictions."
    ),
)
def get_train_details(
    train_id: str = Path(..., description="Train number (e.g. 12302) or integer ID"),
    data_source: Optional[str] = Query(None, description="Data source mode: SIMULATOR or LIVE_API"),
    db: Session = Depends(get_db),
    sim: TrainSimulator = Depends(get_active_simulator),
    provider: TrainStateProvider = Depends(get_state_provider),
    eta_service: MultiStationETAService = Depends(get_multi_station_service),
    lookup_service: LiveTrainLookupService = Depends(get_live_lookup_service),
):
    """
    Retrieves complete train information with live telemetry and multi-station ETA predictions.
    """
    train = find_train(db, train_id)
    if not train:
        clean_id = str(train_id).strip()
        try:
            live = lookup_service.lookup_live_train(clean_id, db=db)
            c_eta = live.ml_eta or live.baseline_eta or live.timestamp
            c_range = (
                ConfidenceRange(
                    lower_bound=live.confidence_range.lower_bound,
                    upper_bound=live.confidence_range.upper_bound,
                    margin_minutes=live.confidence_range.margin_minutes,
                )
                if live.confidence_range
                else ConfidenceRange(
                    lower_bound=c_eta,
                    upper_bound=c_eta,
                    margin_minutes=0.0,
                )
            )
            up_stations = []
            if live.next_station:
                up_stations.append(UpcomingStationETA(
                    station_code=live.next_station,
                    station_name=live.next_station,
                    station_sequence=1,
                    distance_to_go_km=0.0,
                    segments_ahead=1,
                    intermediate_halts=0,
                    scheduled_eta=c_eta,
                    baseline_eta=c_eta,
                    ml_eta=c_eta,
                    predicted_eta=c_eta,
                    scheduled_remaining_minutes=0.0,
                    baseline_remaining_minutes=0.0,
                    predicted_remaining_minutes=0.0,
                    confidence_lower_bound=c_range.lower_bound,
                    confidence_upper_bound=c_range.upper_bound,
                    confidence_range=c_range,
                    segment_predictions=[],
                ))
            c_state = TrainRunningState(
                train_number=live.train_number,
                journey_date=live.timestamp.date(),
                train_name=live.train_name,
                status=live.status,
                current_station_code=live.current_station,
                current_delay_minutes=live.current_delay,
                next_station_code=live.next_station,
                speed_kmh=live.speed or 0.0,
                segment_progress=live.segment_progress or 0.0,
                timestamp=live.timestamp,
                source="external_api",
            )
            return TrainDetailResponse(
                id=0,
                train_number=live.train_number,
                name=live.train_name,
                train_type="EXPRESS",
                route_id=0,
                route_stations=[],
                current_state=c_state,
                current_station=live.current_station,
                current_delay=live.current_delay,
                current_delay_minutes=live.current_delay,
                current_timestamp=live.timestamp,
                upcoming_stations=up_stations,
                scheduled_eta=c_eta,
                baseline_eta=c_eta,
                ml_eta=c_eta,
                confidence_range=c_range,
                segment_predictions=[],
                active_events=[],
                delay_history=[live.current_delay],
                delay_trend=0.0,
            )
        except (TrainNotFoundError, Exception):
            raise HTTPException(status_code=404, detail=f"Train '{train_id}' not found.")

    # 1. Running state via provider (SIMULATOR or LIVE_API with fallback)
    try:
        provider_result = provider.get_train_state(
            train_number=train.train_number,
            db=db,
            sim=sim,
            mode_override=data_source,
            fallback_on_error=True,
        )
    except SimulatorStoppedError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "SIMULATOR_STOPPED",
                "message": "Train simulator is currently stopped or unavailable.",
                "detail": str(exc),
            },
        )
    running_state = provider_result.state
    active_events: List[Dict[str, Any]] = []
    if provider_result.active_events:
        for evt in provider_result.active_events:
            evt_type = evt.event_type.value if hasattr(evt.event_type, "value") else str(evt.event_type)
            active_events.append({
                "event_type": evt_type,
                "delay_minutes": float(evt.delay_minutes),
                "severity": getattr(evt, "severity", "MEDIUM"),
                "metadata": getattr(evt, "metadata", {}) or {},
                "timestamp": evt.created_at.isoformat() if hasattr(evt, "created_at") and evt.created_at else None,
            })

    delay_history = provider_result.delay_history
    delay_trend = provider_result.delay_trend

    # 2. Ordered route stations
    route_stations: List[RouteStationInfo] = []
    sorted_stops = []
    if train.route and train.route.route_stations:
        sorted_stops = sorted(train.route.route_stations, key=lambda rs: rs.sequence)
        for rs in sorted_stops:
            stn_code = rs.station.code if hasattr(rs, "station") and rs.station else ""
            stn_name = rs.station.name if hasattr(rs, "station") and rs.station else ""
            setattr(rs, "station_code", stn_code)
            setattr(rs, "station_name", stn_name)
            route_stations.append(RouteStationInfo(
                sequence=rs.sequence,
                station_code=stn_code,
                station_name=stn_name,
                distance_from_source_km=rs.distance_from_source_km,
                scheduled_arrival=rs.scheduled_arrival,
                scheduled_departure=rs.scheduled_departure,
                scheduled_stop_minutes=rs.scheduled_stop_minutes or 0.0,
            ))

    # 3. Extract current operational telemetry
    current_station = (
        running_state.current_station_code
        if running_state and running_state.current_station_code
        else (sorted_stops[0].station_code if sorted_stops else None)
    )
    current_delay = float(running_state.current_delay_minutes) if running_state else 0.0
    current_timestamp = running_state.timestamp if running_state else datetime.now(timezone.utc)

    # 4. Compute multi-station ETA predictions
    upcoming_stations: List[UpcomingStationETA] = []
    if running_state and sorted_stops:
        try:
            raw_predictions = eta_service.predict_upcoming_stations(
                train_state=running_state,
                route_stations=sorted_stops,
                db=db,
            )
            for pred in raw_predictions:
                conf_range = ConfidenceRange(
                    lower_bound=pred.confidence_lower_bound,
                    upper_bound=pred.confidence_upper_bound,
                    margin_minutes=pred.uncertainty_margin_minutes,
                    lower_bound_minutes=pred.confidence_lower_bound_minutes,
                    upper_bound_minutes=pred.confidence_upper_bound_minutes,
                )
                sched_rem_min = (
                    max(0.0, (pred.scheduled_eta - running_state.timestamp).total_seconds() / 60.0)
                    if running_state.timestamp
                    else None
                )
                upcoming_stations.append(UpcomingStationETA(
                    station_code=pred.station_code,
                    station_name=pred.station_name,
                    station_sequence=pred.station_sequence,
                    distance_to_go_km=pred.distance_to_go_km,
                    segments_ahead=pred.segments_ahead,
                    intermediate_halts=pred.intermediate_halts,
                    scheduled_eta=pred.scheduled_eta,
                    baseline_eta=pred.baseline_eta,
                    ml_eta=pred.predicted_eta,
                    predicted_eta=pred.predicted_eta,
                    scheduled_remaining_minutes=sched_rem_min,
                    baseline_remaining_minutes=pred.baseline_remaining_minutes,
                    predicted_remaining_minutes=pred.predicted_remaining_minutes,
                    confidence_lower_bound=pred.confidence_lower_bound,
                    confidence_upper_bound=pred.confidence_upper_bound,
                    confidence_range=conf_range,
                    segment_predictions=pred.segment_predictions,
                ))
        except Exception:
            # Graceful fallback if ML prediction fails
            pass

    # Ensure baseline ETA fallback if ML predictions failed or were empty
    if running_state and sorted_stops and not upcoming_stations:
        curr_code = (running_state.current_station_code or "").upper().strip()
        curr_seq = running_state.current_station_sequence or 1
        curr_idx = 0
        for idx, s in enumerate(sorted_stops):
            code = getattr(s, "station_code", getattr(s, "code", "")).upper().strip()
            seq = getattr(s, "sequence", getattr(s, "seq", idx + 1))
            if code == curr_code or seq == curr_seq:
                curr_idx = idx
                break

        for j, target_stop in enumerate(sorted_stops[curr_idx + 1:]):
            t_code = getattr(target_stop, "station_code", getattr(target_stop, "code", "")).upper().strip()
            t_name = getattr(target_stop, "station_name", getattr(target_stop, "name", t_code))
            t_seq = int(getattr(target_stop, "sequence", getattr(target_stop, "seq", curr_idx + j + 2)))
            try:
                b_pred = baseline_service.predict_station(
                    train_state=running_state,
                    target_station_code=t_code,
                    route_stations=sorted_stops,
                    db=db,
                )
                b_eta = b_pred.baseline_eta
                s_eta = b_pred.scheduled_eta
                dist_km = b_pred.distance_to_go_km
                rem_min = (
                    max(0.0, (b_eta - running_state.timestamp).total_seconds() / 60.0)
                    if running_state.timestamp
                    else 0.0
                )
                uncertainty = round(5.0 * math.sqrt(j + 1), 1)
                c_range = ConfidenceRange(
                    lower_bound=b_eta - timedelta(minutes=uncertainty),
                    upper_bound=b_eta + timedelta(minutes=uncertainty),
                    margin_minutes=uncertainty,
                    lower_bound_minutes=max(0.0, rem_min - uncertainty),
                    upper_bound_minutes=rem_min + uncertainty,
                )
                sched_rem = (
                    max(0.0, (s_eta - running_state.timestamp).total_seconds() / 60.0)
                    if (running_state.timestamp and s_eta)
                    else None
                )
                upcoming_stations.append(UpcomingStationETA(
                    station_code=t_code,
                    station_name=t_name,
                    station_sequence=t_seq,
                    distance_to_go_km=dist_km,
                    segments_ahead=j + 1,
                    intermediate_halts=j,
                    scheduled_eta=s_eta or b_eta,
                    baseline_eta=b_eta,
                    ml_eta=b_eta,
                    predicted_eta=b_eta,
                    scheduled_remaining_minutes=sched_rem,
                    baseline_remaining_minutes=rem_min,
                    predicted_remaining_minutes=rem_min,
                    confidence_lower_bound=b_eta - timedelta(minutes=uncertainty),
                    confidence_upper_bound=b_eta + timedelta(minutes=uncertainty),
                    confidence_range=c_range,
                    segment_predictions=[],
                ))
            except Exception:
                pass

    # 5. Populate immediate next station ETAs and full upcoming segment breakdown
    next_stop_eta = upcoming_stations[0] if upcoming_stations else None
    scheduled_eta = next_stop_eta.scheduled_eta if next_stop_eta else None
    baseline_eta = next_stop_eta.baseline_eta if next_stop_eta else None
    ml_eta = next_stop_eta.ml_eta if next_stop_eta else None
    confidence_range = next_stop_eta.confidence_range if next_stop_eta else None
    segment_predictions = upcoming_stations[-1].segment_predictions if upcoming_stations else []

    is_cached_resp = False
    cache_age_resp = None
    if provider.client and getattr(provider.client, "cache", None):
        try:
            from backend.services.cache import LiveTrainCache, CacheEntry
            if isinstance(provider.client.cache, LiveTrainCache):
                c_entry = provider.client.cache.get_entry(train.train_number)
                if isinstance(c_entry, CacheEntry):
                    is_cached_resp = (c_entry.hit_count > 0)
                    cache_age_resp = c_entry.age_seconds
        except Exception:
            pass

    return TrainDetailResponse(
        id=train.id,
        train_number=train.train_number,
        name=train.name,
        train_type=train.train_type,
        route_id=train.route_id,
        route_stations=route_stations,
        current_state=running_state,
        current_station=current_station,
        current_delay=current_delay,
        current_delay_minutes=current_delay,
        current_timestamp=current_timestamp,
        upcoming_stations=upcoming_stations,
        scheduled_eta=scheduled_eta,
        baseline_eta=baseline_eta,
        ml_eta=ml_eta,
        confidence_range=confidence_range,
        segment_predictions=segment_predictions,
        active_events=active_events,
        delay_trend=delay_trend,
        delay_history=delay_history,
        data_source_mode=provider_result.effective_mode.value,
        is_fallback=provider_result.is_fallback,
        fallback_reason=provider_result.fallback_reason,
        last_updated=current_timestamp,
        is_cached=is_cached_resp,
        cache_age_seconds=cache_age_resp,
    )


@app.get(
    "/train/{train_id}/eta/{station_code}",
    response_model=SingleStationETAResponse,
    summary="Get single station ETA prediction",
    description=(
        "Calculates multi-station ML ETA, baseline ETA, and confidence bounds "
        "for a specific upcoming station on the train's route. "
        "If ML prediction fails, returns baseline ETA with ml_status='UNAVAILABLE'."
    ),
)
def get_train_station_eta(
    train_id: str = Path(..., description="Train number or ID"),
    station_code: str = Path(..., description="Target upcoming station code (e.g. CNB, PRYJ)"),
    data_source: Optional[str] = Query(None, description="Data source mode: SIMULATOR or LIVE_API"),
    db: Session = Depends(get_db),
    sim: TrainSimulator = Depends(get_active_simulator),
    provider: TrainStateProvider = Depends(get_state_provider),
    eta_service: MultiStationETAService = Depends(get_multi_station_service),
):
    """
    Calculates the dynamic ETA prediction (ML and Baseline) for an upcoming station.
    """
    train = find_train(db, train_id)
    if not train:
        raise HTTPException(status_code=404, detail=f"Train '{train_id}' not found.")

    target_code = station_code.upper().strip()

    # 1. Get train running state via provider (SIMULATOR or LIVE_API with fallback)
    try:
        provider_result = provider.get_train_state(
            train_number=train.train_number,
            db=db,
            sim=sim,
            mode_override=data_source,
            fallback_on_error=True,
        )
        train_state = provider_result.state
        active_events = provider_result.active_events
    except Exception as exc:
        clean_msg = sanitize_secret(str(exc))
        raise HTTPException(
            status_code=400,
            detail=f"Unable to resolve state for train '{train.train_number}': {clean_msg}",
        )

    # 2. Ordered route stops
    sorted_stops = []
    if train.route and train.route.route_stations:
        sorted_stops = sorted(train.route.route_stations, key=lambda rs: rs.sequence)
        for rs in sorted_stops:
            stn_code = rs.station.code if hasattr(rs, "station") and rs.station else ""
            stn_name = rs.station.name if hasattr(rs, "station") and rs.station else ""
            setattr(rs, "station_code", stn_code)
            setattr(rs, "station_name", stn_name)

    # 3. Calculate Baseline ETA (guaranteed baseline fallback)
    try:
        baseline_pred = baseline_service.predict_station(
            train_state=train_state,
            target_station_code=target_code,
            db=db,
        )
        scheduled_eta = baseline_pred.scheduled_eta
        baseline_eta = baseline_pred.baseline_eta
        distance_to_go_km = baseline_pred.distance_to_go_km
        baseline_dict = baseline_pred.model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 4. Extract 11 ML features (for backward compatibility / inspection)
    features_dict = None
    feature_prov_dict = None
    est_features_list = None
    try:
        features = feature_builder.build(
            train_state=train_state,
            target_station_code=target_code,
            db=db,
            active_events=active_events,
        )
        features_dict = features.to_model_input_dict()
        feature_prov_dict = features.feature_provenance
        est_features_list = features.estimated_features
    except Exception:
        pass

    # 5. Calculate ML ETA via MultiStationETAService
    ml_eta: Optional[datetime] = None
    confidence_lower: Optional[datetime] = None
    confidence_upper: Optional[datetime] = None
    ml_status = "AVAILABLE"
    ml_error: Optional[str] = None
    segments_ahead: Optional[int] = None

    try:
        station_pred = eta_service.predict_station_eta(
            train_state=train_state,
            target_station_code=target_code,
            route_stations=sorted_stops,
            db=db,
        )
        ml_eta = station_pred.predicted_eta
        confidence_lower = station_pred.confidence_lower_bound
        confidence_upper = station_pred.confidence_upper_bound
        segments_ahead = station_pred.segments_ahead
        distance_to_go_km = station_pred.distance_to_go_km
        if scheduled_eta is None:
            scheduled_eta = station_pred.scheduled_eta
        if baseline_eta is None:
            baseline_eta = station_pred.baseline_eta
    except ValueError as exc:
        # Off-route station or station behind current train position
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # ML prediction failed - fallback to baseline, clearly indicate ML status as unavailable, do not crash API
        ml_status = "UNAVAILABLE"
        ml_error = f"ML prediction failed: {str(exc)}"
        ml_eta = None
        confidence_lower = None
        confidence_upper = None

    model_ver = get_model_version(eta_service)
    prediction_ts = datetime.now(timezone.utc)

    return SingleStationETAResponse(
        train_id=train.id,
        train_number=train.train_number,
        train_name=train.name,
        target_station=target_code,
        current_station=train_state.current_station_code,
        scheduled_eta=scheduled_eta,
        baseline_eta=baseline_eta,
        ml_eta=ml_eta,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        current_delay_minutes=train_state.current_delay_minutes,
        prediction_timestamp=prediction_ts,
        model_version=model_ver,
        ml_status=ml_status,
        ml_error=ml_error,
        prediction=baseline_dict,
        features=features_dict,
        feature_provenance=feature_prov_dict,
        estimated_features=est_features_list,
        segments_ahead=segments_ahead,
        distance_to_go_km=distance_to_go_km,
        data_source_mode=provider_result.effective_mode.value,
        is_fallback=provider_result.is_fallback,
        fallback_reason=provider_result.fallback_reason,
    )


@app.get(
    "/station/{station_code}/arrivals",
    response_model=StationArrivalsResponse,
    summary="Get live station arrivals board",
    description=(
        "Retrieves currently running train journeys approaching the specified station. "
        "Calculates Scheduled ETA, Baseline ETA, and ML ETA with confidence ranges. "
        "Returns arrivals sorted primarily by predicted ML ETA within the requested time window."
    ),
)
def get_station_arrivals(
    station_code: str = Path(..., description="Station code (e.g. PRYJ, CNB, HWH)"),
    window_hours: Optional[float] = Query(
        default=24.0,
        ge=0.1,
        le=72.0,
        description="Upcoming arrival time window in hours (default: 24.0)",
    ),
    db: Session = Depends(get_db),
    sim: TrainSimulator = Depends(get_active_simulator),
    eta_service: MultiStationETAService = Depends(get_multi_station_service),
):
    """
    Station live arrivals board.
    Finds running journeys approaching the station, computes Scheduled, Baseline, and ML ETAs,
    and returns them sorted primarily by ML ETA within the requested time window.
    """
    clean_code = station_code.upper().strip()
    station = db.query(Station).filter(Station.code == clean_code).first()
    if not station:
        raise HTTPException(status_code=404, detail=f"Station '{station_code}' not found.")

    ref_time = datetime.now(timezone.utc)
    if sim and hasattr(sim, "sim_time") and sim.sim_time:
        ref_time = sim.sim_time
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

    arrivals: List[StationArrivalItem] = []
    if sim and not getattr(sim, "is_running", True):
        raise SimulatorStoppedError("Train simulator is currently stopped.")

    if not sim or not sim.journeys:
        return StationArrivalsResponse(
            station_code=station.code,
            station_name=station.name,
            current_timestamp=ref_time,
            window_hours=window_hours,
            total_arrivals=0,
            arrivals=[],
        )

    for train_number, journey in sim.journeys.items():
        if journey.status not in ("RUNNING", "HALTED"):
            continue

        stops = journey.stops
        target_idx = next(
            (i for i, s in enumerate(stops) if getattr(s, "station_code", getattr(s, "code", "")).upper() == clean_code),
            None,
        )
        if target_idx is None:
            continue

        # Check if train is approaching (station is ahead of train's current position)
        if target_idx <= journey.current_station_index:
            continue

        segments_ahead = target_idx - journey.current_station_index
        train_state = journey.to_train_running_state()

        origin_code = getattr(stops[0], "station_code", None) if stops else None
        dest_code = getattr(stops[-1], "station_code", None) if stops else None

        # 1. Baseline ETA (guaranteed fallback)
        scheduled_eta: Optional[datetime] = None
        baseline_eta: Optional[datetime] = None
        dist_km: float = 0.0
        try:
            base_pred = baseline_service.predict_station(
                train_state=train_state,
                target_station_code=clean_code,
                db=db,
            )
            scheduled_eta = base_pred.scheduled_eta
            baseline_eta = base_pred.baseline_eta
            dist_km = base_pred.distance_to_go_km
        except Exception:
            pass

        # 2. ML ETA & Confidence Range via MultiStationETAService
        ml_eta: Optional[datetime] = None
        confidence_lower: Optional[datetime] = None
        confidence_upper: Optional[datetime] = None
        conf_range: Optional[ConfidenceRange] = None
        ml_status = "AVAILABLE"

        try:
            stn_pred = eta_service.predict_station_eta(
                train_state=train_state,
                target_station_code=clean_code,
                db=db,
            )
            ml_eta = stn_pred.predicted_eta
            confidence_lower = stn_pred.confidence_lower_bound
            confidence_upper = stn_pred.confidence_upper_bound
            conf_range = ConfidenceRange(
                lower_bound=stn_pred.confidence_lower_bound,
                upper_bound=stn_pred.confidence_upper_bound,
                margin_minutes=stn_pred.uncertainty_margin_minutes,
                lower_bound_minutes=stn_pred.confidence_lower_bound_minutes,
                upper_bound_minutes=stn_pred.confidence_upper_bound_minutes,
            )
            if scheduled_eta is None:
                scheduled_eta = stn_pred.scheduled_eta
            if baseline_eta is None:
                baseline_eta = stn_pred.baseline_eta
            dist_km = stn_pred.distance_to_go_km
            segments_ahead = stn_pred.segments_ahead
        except Exception:
            ml_status = "UNAVAILABLE"
            ml_eta = None
            confidence_lower = None
            confidence_upper = None
            conf_range = None

        effective_eta = ml_eta or baseline_eta or scheduled_eta
        if effective_eta is None:
            continue

        # Window filtering (Requirement 8)
        if window_hours is not None:
            max_window_dt = ref_time + timedelta(hours=window_hours)
            if effective_eta < ref_time - timedelta(minutes=15) or effective_eta > max_window_dt:
                continue

        mins_to_arr = max(0.0, (effective_eta - ref_time).total_seconds() / 60.0)

        arrivals.append(
            StationArrivalItem(
                train_id=journey.train_id,
                train_number=journey.train_number,
                train_name=journey.train_name,
                train_type=journey.train_type,
                origin_station_code=origin_code,
                destination_station_code=dest_code,
                current_station=train_state.current_station_code,
                current_delay_minutes=train_state.current_delay_minutes,
                distance_to_go_km=dist_km,
                segments_ahead=segments_ahead,
                scheduled_eta=scheduled_eta,
                baseline_eta=baseline_eta,
                ml_eta=ml_eta,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                confidence_range=conf_range,
                ml_status=ml_status,
                minutes_to_arrival=round(mins_to_arr, 1),
            )
        )

    # Requirement 6: Sort primarily by predicted ML ETA (fallback to baseline/scheduled)
    arrivals.sort(key=lambda a: (a.ml_eta or a.baseline_eta or a.scheduled_eta))

    return StationArrivalsResponse(
        station_code=station.code,
        station_name=station.name,
        current_timestamp=ref_time,
        window_hours=window_hours,
        total_arrivals=len(arrivals),
        arrivals=arrivals,
    )


@app.post(
    "/simulate/event",
    response_model=EventInjectionResponse,
    summary="Inject operational disruption event",
    description=(
        "Injects an operational disruption event (SIGNAL_HALT, CONGESTION, "
        "SPEED_RESTRICTION, UNSCHEDULED_HALT, WEATHER) into a train's journey. "
        "Deterministically modifies delay and speed, persists to SQLite, and "
        "recalculates updated baseline and ML ETAs."
    ),
)
def inject_simulation_event(
    payload: EventInjectionRequest,
    db: Session = Depends(get_db),
    sim: TrainSimulator = Depends(get_active_simulator),
    eta_service: MultiStationETAService = Depends(get_multi_station_service),
):
    """
    Applies an operational disruption event via the TrainSimulator, updates journey state,
    persists the event, and returns updated Baseline and ML ETAs.
    """
    train = find_train(db, str(payload.train_id))
    if not train:
        raise HTTPException(
            status_code=404,
            detail=f"Train '{payload.train_id}' not found.",
        )

    # Ensure journey is loaded in simulator
    if train.train_number not in sim.journeys:
        sim.load_from_db(db, train_numbers=[train.train_number])
    if train.train_number not in sim.journeys:
        raise HTTPException(
            status_code=404,
            detail=f"Train '{train.train_number}' could not be loaded into the simulator.",
        )

    # 1. Apply the event through the existing simulator/event system
    # 2. Persist the event in SQLite (handled inside sim.inject_event when db is passed)
    sim_event = sim.inject_event(
        train_number=train.train_number,
        event_type=payload.event_type,
        delay_minutes=payload.delay_minutes,
        severity=payload.severity or "MEDIUM",
        metadata=payload.metadata,
        db=db,
    )

    # 3. Update journey state & persist to DB
    sim.sync_to_db(db)

    # 4. Get updated train running state
    updated_state = sim.get_state(train.train_number)

    # 5. Ensure the next ETA calculation uses the updated state
    target_station = updated_state.next_station_code
    baseline_eta: Optional[datetime] = None
    ml_eta: Optional[datetime] = None
    conf_range: Optional[ConfidenceRange] = None
    ml_status = "AVAILABLE"

    if target_station:
        # Calculate updated Baseline ETA
        try:
            base_pred = baseline_service.predict_station(
                train_state=updated_state,
                target_station_code=target_station,
                db=db,
            )
            baseline_eta = base_pred.baseline_eta
        except Exception:
            pass

        # Calculate updated ML ETA
        try:
            stn_pred = eta_service.predict_station_eta(
                train_state=updated_state,
                target_station_code=target_station,
                db=db,
            )
            ml_eta = stn_pred.predicted_eta
            conf_range = ConfidenceRange(
                lower_bound=stn_pred.confidence_lower_bound,
                upper_bound=stn_pred.confidence_upper_bound,
                margin_minutes=stn_pred.uncertainty_margin_minutes,
                lower_bound_minutes=stn_pred.confidence_lower_bound_minutes,
                upper_bound_minutes=stn_pred.confidence_upper_bound_minutes,
            )
            if baseline_eta is None:
                baseline_eta = stn_pred.baseline_eta
        except Exception:
            ml_status = "UNAVAILABLE"
            ml_eta = baseline_eta

    event_details = EventDetails(
        event_type=sim_event.event_type.value,
        delay_minutes=sim_event.delay_minutes,
        severity=sim_event.severity,
        metadata=sim_event.metadata,
        timestamp=sim_event.created_at,
    )

    return EventInjectionResponse(
        event=event_details,
        train_state=updated_state,
        target_station=target_station,
        baseline_eta=baseline_eta,
        updated_baseline_eta=baseline_eta,
        ml_eta=ml_eta,
        updated_ml_eta=ml_eta,
        confidence_range=conf_range,
        ml_status=ml_status,
        message=f"Event {sim_event.event_type.value} of {sim_event.delay_minutes:.1f}m injected successfully.",
    )


@app.get(
    "/model/metrics",
    response_model=ModelMetricsResponse,
    summary="Get latest model evaluation metrics",
    description=(
        "Loads and returns the latest evaluation metrics comparing the Baseline heuristic "
        "against the XGBoost ETA Regressor on held-out test journeys. If metrics are "
        "unavailable, returns a clear structured response without crashing."
    ),
)
def get_model_metrics(
    metrics_path: FilePath = Depends(get_metrics_filepath),
    metadata_path: FilePath = Depends(get_metadata_filepath),
):
    """
    Returns latest model performance metrics including overall and broken down by horizon and disruption.
    """
    metrics_data = load_latest_metrics(
        metrics_path=metrics_path,
        metadata_path=metadata_path,
    )
    return ModelMetricsResponse(**metrics_data)


# ---------------------------------------------------------------------------
# Deterministic Hackathon Demo Mode Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/demo/reset",
    response_model=DemoScenarioResponse,
    summary="Reset deterministic hackathon demo scenario (Seed 42)",
    description=(
        "Resets flagship train 12302 (Howrah Rajdhani Express) to the exact predefined baseline state "
        "(departed CNB bound for PRYJ, +2.0m initial headway delay, 110 km/h) with fixed random seed 42. "
        "Clears all active disruptions and recalculates Baseline and ML ETAs deterministically."
    ),
)
def reset_demo_scenario(
    db: Session = Depends(get_db),
    sim: TrainSimulator = Depends(get_active_simulator),
    demo_mgr: DemoScenarioManager = Depends(get_demo_manager),
):
    if sim and not getattr(sim, "is_running", True):
        raise SimulatorStoppedError("Train simulator is currently stopped.")
    try:
        return demo_mgr.reset_scenario(db, sim)
    except SimulatorStoppedError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to reset demo scenario: {str(exc)}")


@app.get(
    "/demo/scenario",
    response_model=DemoScenarioResponse,
    summary="Get deterministic hackathon demo scenario state",
    description="Returns current scenario status, train state, active events, and available demo actions.",
)
def get_demo_scenario(
    db: Session = Depends(get_db),
    sim: TrainSimulator = Depends(get_active_simulator),
    demo_mgr: DemoScenarioManager = Depends(get_demo_manager),
):
    if sim and not getattr(sim, "is_running", True):
        raise SimulatorStoppedError("Train simulator is currently stopped.")
    try:
        return demo_mgr.get_scenario(db, sim)
    except SimulatorStoppedError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve demo scenario: {str(exc)}")


@app.post(
    "/demo/action/{action_id}",
    response_model=DemoActionResult,
    summary="Execute prepared hackathon demo action",
    description=(
        "Executes one of the 3 prepared demo actions deterministically (seed 42): "
        "1. signal_halt (+15m, speed=0 km/h, status=HALTED) "
        "2. congestion (+10m, speed=49.5 km/h, status=RUNNING) "
        "3. speed_restriction (+8m, speed=30.0 km/h, status=RUNNING). "
        "State and ML ETAs recalculate and propagate automatically across all endpoints."
    ),
)
def execute_demo_action(
    action_id: str = Path(..., description="Action ID: signal_halt, congestion, or speed_restriction (or 1, 2, 3)"),
    db: Session = Depends(get_db),
    sim: TrainSimulator = Depends(get_active_simulator),
    demo_mgr: DemoScenarioManager = Depends(get_demo_manager),
):
    if sim and not getattr(sim, "is_running", True):
        raise SimulatorStoppedError("Train simulator is currently stopped.")
    try:
        return demo_mgr.execute_action(action_id, db, sim)
    except SimulatorStoppedError:
        raise
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to execute demo action: {str(exc)}")



