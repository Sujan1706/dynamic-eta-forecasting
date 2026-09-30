"""
RailRadar Live API Client & Adapter.
====================================
Retrieves real-time Indian Railways operational tracking data from RailRadar API
and normalizes external payloads into the existing TrainRunningState schema.

Ensures:
- Server-side API key handling via RAILRADAR_API_KEY environment variable.
- Zero client-side API key exposure.
- Safe secret scrubbing/masking in logs and error messages.
- Thread-safe caching with TTL and Stale-on-Error pattern for HTTP 429.
- Timeout and error handling for 401, 404, 429, 503, and network failures.
- Interoperability with existing simulator, feature engineering, and ETA prediction pipelines.
"""

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
    NormalizedLiveTrain,
)

__all__ = [
    "RailRadarClient",
    "RailwayAPIError",
    "MissingApiKeyError",
    "AuthenticationError",
    "TrainNotFoundError",
    "RateLimitExceededError",
    "APITimeoutError",
    "MalformedResponseError",
    "ServiceUnavailableError",
    "sanitize_secret",
    "NormalizedLiveTrain",
]
