import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, date, timezone
import httpx
from dotenv import load_dotenv

from backend.services.schemas import TrainRunningState

load_dotenv()


class RailwayAPIError(Exception):
    """Base exception for Railway API client errors."""
    pass


class MissingApiKeyError(RailwayAPIError):
    """Raised when RAILRADAR_API_KEY is not set."""
    pass


class AuthenticationError(RailwayAPIError):
    """Raised when API key is invalid or unauthorized (HTTP 401)."""
    pass


class TrainNotFoundError(RailwayAPIError):
    """Raised when requested train number does not exist (HTTP 404)."""
    pass


class RateLimitExceededError(RailwayAPIError):
    """Raised when request quota/rate limit is reached (HTTP 429)."""
    pass


class APITimeoutError(RailwayAPIError):
    """Raised when the API request times out."""
    pass


class MalformedResponseError(RailwayAPIError):
    """Raised when response is not valid JSON or missing expected structure."""
    pass


class ServiceUnavailableError(RailwayAPIError):
    """Raised when RailRadar server is temporarily unavailable (HTTP 503)."""
    pass


def sanitize_secret(text: str, secret: Optional[str] = None) -> str:
    """Scrubs any occurrence of API key or secret token from strings, logs, and error messages."""
    if not text:
        return ""
    key = secret or os.getenv("RAILRADAR_API_KEY")
    if key and len(key.strip()) > 2:
        clean_key = key.strip()
        text = text.replace(clean_key, "***MASKED_API_KEY***")
    return text


# Backward compatibility alias
NormalizedLiveTrain = TrainRunningState


class RailRadarClient:
    """
    Lightweight, standalone client for the RailRadar live train API endpoint.
    Follows zero-polling and strictly on-demand query patterns.
    """

    DEFAULT_BASE_URL = "https://api.railradar.in"
    cache: Optional[Any] = None
    enable_cache: bool = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 10.0,
        cache: Optional[Any] = None,
        cache_ttl_seconds: Optional[int] = None,
        enable_cache: bool = True,
    ):
        self.api_key = api_key or os.getenv("RAILRADAR_API_KEY")
        self.base_url = (base_url or os.getenv("RAILRADAR_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.enable_cache = enable_cache

        if not enable_cache:
            self.cache = None
        elif cache is not None:
            self.cache = cache
        else:
            from backend.services.cache import LiveTrainCache
            self.cache = LiveTrainCache(ttl_seconds=cache_ttl_seconds)

    def _ensure_api_key(self) -> str:
        if not self.api_key or not self.api_key.strip():
            raise MissingApiKeyError(
                "Missing API key: RAILRADAR_API_KEY is not set. "
                "Please configure RAILRADAR_API_KEY in your .env file or environment."
            )
        return self.api_key.strip()

    def fetch_live_train(
        self,
        train_number: str,
        debug_output_file: Optional[Path] = None,
        bypass_cache: bool = False,
    ) -> TrainRunningState:
        """
        Calls GET /v1/trains/{number}/live and normalizes the payload into TrainRunningState.
        Uses in-process cache with configurable TTL to prevent redundant requests.
        Gracefully returns stale cache on HTTP 429 rate limit.
        Optionally writes raw response JSON to debug_output_file if enabled.
        """
        cleaned_number = str(train_number).strip()

        # 1. In-process cache lookup (Requirements 1 & 2)
        if self.enable_cache and self.cache and not bypass_cache:
            cached_state, entry = self.cache.get(cleaned_number, allow_stale=False)
            if cached_state is not None:
                return cached_state

            # If currently in rate limit cooldown, serve stale data if available
            if self.cache.is_rate_limited():
                stale_state, entry = self.cache.get(cleaned_number, allow_stale=True)
                if stale_state is not None:
                    return stale_state
                raise RateLimitExceededError(
                    f"Rate limit exceeded (HTTP 429): In cooldown period for next {self.cache.rate_limit_remaining_seconds()}s."
                )

        key = self._ensure_api_key()
        url = f"{self.base_url}/v1/trains/{cleaned_number}/live"

        headers = {
            "Accept": "application/json",
            "X-API-Key": key,
            "Authorization": f"Bearer {key}",
            "User-Agent": "DynamicETAForecasting/1.0",
        }

        try:
            with httpx.Client(follow_redirects=True, timeout=self.timeout_seconds) as client:
                response = client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise APITimeoutError(
                f"Connection to RailRadar API timed out after {self.timeout_seconds}s for train {cleaned_number}."
            ) from exc
        except httpx.RequestError as exc:
            clean_err = sanitize_secret(str(exc), key)
            raise RailwayAPIError(f"Network error while calling RailRadar API: {clean_err}") from exc

        # Handle HTTP status codes
        if response.status_code == 401:
            raise AuthenticationError(
                "Authentication failed (HTTP 401): The provided RAILRADAR_API_KEY is invalid or unauthorized."
            )
        elif response.status_code == 404:
            raise TrainNotFoundError(
                f"Train not found (HTTP 404): No live tracking data found for train number '{cleaned_number}'."
            )
        elif response.status_code == 429:
            # Requirement 6: Handle API 429 gracefully via stale-on-error
            if self.enable_cache and self.cache:
                self.cache.mark_rate_limited()
                stale_state, entry = self.cache.get(cleaned_number, allow_stale=True)
                if stale_state is not None:
                    return stale_state
            raise RateLimitExceededError(
                "Rate limit exceeded (HTTP 429): Quota limit reached on RailRadar API. Please wait before retrying."
            )
        elif response.status_code == 503:
            raise ServiceUnavailableError(
                "RailRadar service temporarily unavailable (HTTP 503): The live railway server is under maintenance or overloaded."
            )
        elif response.status_code >= 400:
            clean_text = sanitize_secret(response.text[:200], key)
            raise RailwayAPIError(
                f"RailRadar API error (HTTP {response.status_code}): {clean_text}"
            )

        # Parse JSON
        try:
            raw_data = response.json()
        except Exception as exc:
            clean_text = sanitize_secret(response.text[:200], key)
            raise MalformedResponseError(
                f"Malformed JSON response received from {url}: {clean_text}"
            ) from exc

        # Write raw debug dump if requested
        if debug_output_file:
            try:
                debug_path = Path(debug_output_file)
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(raw_data, f, indent=2)
            except Exception as e:
                print(f"[RailRadarClient] Failed to save debug dump to {debug_output_file}: {e}")

        # Normalize into TrainRunningState
        try:
            normalized_state = self.normalize_payload(raw_data, fallback_train_number=cleaned_number)
        except MalformedResponseError:
            raise
        except Exception as exc:
            clean_err = sanitize_secret(str(exc), key)
            raise MalformedResponseError(f"Malformed external API data: {clean_err}") from exc

        # Store in cache (Requirement 1 & 2)
        if self.enable_cache and self.cache:
            self.cache.set(cleaned_number, normalized_state, raw_payload=raw_data)

        return normalized_state

    def normalize_payload(self, data: Dict[str, Any], fallback_train_number: str) -> TrainRunningState:
        """
        Normalizes varying payload structures (direct or nested under 'data')
        into a consistent TrainRunningState schema with source='external_api'.
        """
        if not isinstance(data, dict):
            raise MalformedResponseError(f"Expected JSON object, got {type(data).__name__}")

        # Unwrap common wrapper keys if present
        payload = data.get("data") if isinstance(data.get("data"), dict) else data
        if not isinstance(payload, dict):
            raise MalformedResponseError(f"Expected JSON dictionary payload, got {type(payload).__name__}")

        # 1. train_number
        number = str(
            payload.get("trainNumber")
            or payload.get("train_number")
            or payload.get("train_no")
            or payload.get("number")
            or fallback_train_number
        ).strip()

        # 2. journey_date
        raw_date = (
            payload.get("startDate")
            or payload.get("journey_date")
            or payload.get("journeyDate")
            or payload.get("date")
        )
        if raw_date:
            journey_date = raw_date
        else:
            journey_date = date.today()

        # 3. train_name
        name = str(
            payload.get("trainName")
            or payload.get("train_name")
            or payload.get("name")
            or (payload.get("train") or {}).get("name")
            or f"Train {number}"
        ).strip()

        # 4. status
        status = str(
            payload.get("status")
            or payload.get("running_status")
            or "UNKNOWN"
        ).upper().strip()

        # 5. Current station & sequence
        curr_loc = (
            payload.get("currentLocation")
            or payload.get("current_station")
            or payload.get("currentStation")
            or payload.get("last_station")
            or {}
        )
        if isinstance(curr_loc, dict):
            current_station_code = (
                curr_loc.get("stationCode")
                or curr_loc.get("code")
                or curr_loc.get("station_code")
                or payload.get("current_station_code")
            )
            current_station_sequence = (
                curr_loc.get("sequence")
                or payload.get("current_station_sequence")
            )
            raw_delay = curr_loc.get("delayMinutes")
            if raw_delay is None:
                raw_delay = payload.get("delayMinutes", payload.get("delay_minutes", payload.get("delay", 0.0)))
            try:
                segment_progress = float(curr_loc.get("segmentProgress", payload.get("segment_progress", 0.0)) or 0.0)
            except (TypeError, ValueError):
                segment_progress = 0.0
            try:
                speed_kmh = float(curr_loc.get("speedKmh", payload.get("speed_kmh", payload.get("speed", 0.0)) or 0.0))
            except (TypeError, ValueError):
                speed_kmh = 0.0
        else:
            current_station_code = str(curr_loc) if curr_loc else payload.get("current_station_code")
            current_station_sequence = payload.get("current_station_sequence")
            raw_delay = payload.get("delayMinutes", payload.get("delay_minutes", payload.get("delay", 0.0)))
            try:
                segment_progress = float(payload.get("segment_progress", 0.0) or 0.0)
            except (TypeError, ValueError):
                segment_progress = 0.0
            try:
                speed_kmh = float(payload.get("speed_kmh", payload.get("speed", 0.0)) or 0.0)
            except (TypeError, ValueError):
                speed_kmh = 0.0

        # Delay in minutes
        try:
            current_delay_minutes = float(raw_delay)
        except (TypeError, ValueError):
            current_delay_minutes = 0.0

        # 6. Previous station code
        prev_loc = (
            payload.get("previousStation")
            or payload.get("previous_station")
            or payload.get("lastStation")
            or payload.get("last_station")
            or payload.get("previousHalt")
            or payload.get("previous_halt")
        )
        if isinstance(prev_loc, dict):
            previous_station_code = (
                prev_loc.get("stationCode")
                or prev_loc.get("code")
                or prev_loc.get("station_code")
            )
        else:
            previous_station_code = str(prev_loc) if prev_loc else payload.get("previous_station_code")

        # 7. Next station code & distance
        next_loc = (
            payload.get("nextHalt")
            or payload.get("nextStation")
            or payload.get("next_station")
            or payload.get("next_halt")
            or payload.get("upcoming_station")
        )
        if isinstance(next_loc, dict):
            next_station_code = (
                next_loc.get("stationCode")
                or next_loc.get("code")
                or next_loc.get("station_code")
            )
            raw_nxt_dist = (
                next_loc.get("distance")
                or next_loc.get("distance_km")
                or next_loc.get("distanceFromCurrentKm")
                or payload.get("next_station_distance_km")
            )
        else:
            next_station_code = str(next_loc) if next_loc else payload.get("next_station_code")
            raw_nxt_dist = payload.get("next_station_distance_km")

        next_station_distance_km = None
        curr_dist_origin = (
            curr_loc.get("distanceFromOriginKm")
            if isinstance(curr_loc, dict)
            else payload.get("distance_from_source_km")
        )
        if raw_nxt_dist is not None:
            try:
                dist_val = float(raw_nxt_dist)
                if curr_dist_origin is not None and dist_val > float(curr_dist_origin):
                    next_station_distance_km = round(dist_val - float(curr_dist_origin), 2)
                else:
                    next_station_distance_km = round(dist_val, 2)
            except (TypeError, ValueError):
                next_station_distance_km = None

        # 8. Timestamp
        raw_ts = (
            payload.get("lastUpdatedAt")
            or payload.get("last_updated")
            or payload.get("updated_at")
            or payload.get("timestamp")
        )
        if raw_ts:
            timestamp = raw_ts
        else:
            timestamp = datetime.now(timezone.utc)

        return TrainRunningState(
            train_number=number,
            journey_date=journey_date,
            train_name=name,
            status=status,
            current_station_code=current_station_code,
            current_station_sequence=current_station_sequence,
            current_delay_minutes=current_delay_minutes,
            previous_station_code=previous_station_code,
            next_station_code=next_station_code,
            next_station_distance_km=next_station_distance_km,
            segment_progress=segment_progress,
            speed_kmh=speed_kmh,
            timestamp=timestamp,
            source="external_api",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Query RailRadar Live Train Running Status (GET /v1/trains/{number}/live)"
    )
    parser.add_argument("train_number", help="Train number (e.g. 12302, 12952)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable writing the raw API response to a separate debug file",
    )
    parser.add_argument(
        "--debug-file",
        default="data/raw/railradar_raw_response.json",
        help="Custom destination path for raw debug output (default: data/raw/railradar_raw_response.json)",
    )

    args = parser.parse_args()

    client = RailRadarClient()
    debug_path = Path(args.debug_file) if args.debug else None

    try:
        live_data = client.fetch_live_train(
            train_number=args.train_number,
            debug_output_file=debug_path,
        )
        live_data.display()
    except MissingApiKeyError as exc:
        print(f"\n[Configuration Error] {exc}", file=sys.stderr)
        print("Set your key in .env:\n  RAILRADAR_API_KEY=your_key_here\n", file=sys.stderr)
        sys.exit(1)
    except AuthenticationError as exc:
        print(f"\n[Unauthorized (401)] {exc}", file=sys.stderr)
        sys.exit(1)
    except TrainNotFoundError as exc:
        print(f"\n[Not Found (404)] {exc}", file=sys.stderr)
        sys.exit(1)
    except RateLimitExceededError as exc:
        print(f"\n[Rate Limit (429)] {exc}", file=sys.stderr)
        sys.exit(1)
    except APITimeoutError as exc:
        print(f"\n[Timeout Error] {exc}", file=sys.stderr)
        sys.exit(1)
    except MalformedResponseError as exc:
        print(f"\n[Data Error] {exc}", file=sys.stderr)
        sys.exit(1)
    except RailwayAPIError as exc:
        print(f"\n[API Error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
