"""
Unit & Integration Tests for Live Data Caching Layer.
=====================================================
Covers all requirements:
1. Cache live train responses for a configurable TTL.
2. Do not request the same train repeatedly within the TTL.
3. Make TTL configurable through environment/configuration (RAILRADAR_CACHE_TTL_SECONDS).
4. Clearly display when data was last updated (last_updated, cached_at, is_cached, cache_age_seconds).
5. Keep simulator mode independent (simulator does not touch or depend on cache).
6. Handle API HTTP 429 gracefully via stale-on-error and rate limit cooldown.
"""

import os
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
import pytest
from starlette.testclient import TestClient

from backend.main import app
from backend.services.schemas import TrainRunningState
from backend.services.cache import (
    LiveTrainCache,
    CacheEntry,
    get_configured_cache_ttl,
    DEFAULT_CACHE_TTL_SECONDS,
)
from backend.services.railway_api_client import (
    RailRadarClient,
    RateLimitExceededError,
)
from backend.services.live_train_service import LiveTrainLookupService
from backend.services.data_source import TrainStateProvider, DataSourceMode
from backend.simulator.engine import TrainSimulator


@pytest.fixture
def sample_payload():
    return {
        "success": True,
        "data": {
            "trainNumber": "12302",
            "trainName": "Howrah Rajdhani Express",
            "startDate": "2026-09-28",
            "status": "RUNNING",
            "currentLocation": {
                "stationCode": "CNB",
                "sequence": 2,
                "delayMinutes": 15.0,
                "segmentProgress": 0.4,
                "speedKmh": 115.0,
            },
            "previousStation": {
                "stationCode": "NDLS",
            },
            "nextHalt": {
                "stationCode": "PRYJ",
                "distance": 195.0,
            },
            "lastUpdatedAt": "2026-09-28T21:40:00Z",
        },
    }


# ---------------------------------------------------------------------------
# Requirement 3: Configurable TTL via Environment
# ---------------------------------------------------------------------------

def test_configurable_ttl_default():
    """Verify default TTL is 60s when no environment variables are set."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_configured_cache_ttl() == DEFAULT_CACHE_TTL_SECONDS
        assert get_configured_cache_ttl() == 60


def test_configurable_ttl_railradar_env():
    """Verify RAILRADAR_CACHE_TTL_SECONDS sets custom TTL."""
    with patch.dict(os.environ, {"RAILRADAR_CACHE_TTL_SECONDS": "180"}, clear=True):
        assert get_configured_cache_ttl() == 180


def test_configurable_ttl_live_data_fallback_env():
    """Verify LIVE_DATA_CACHE_TTL_SECONDS sets custom TTL when RAILRADAR_CACHE_TTL_SECONDS is absent."""
    with patch.dict(os.environ, {"LIVE_DATA_CACHE_TTL_SECONDS": "120"}, clear=True):
        assert get_configured_cache_ttl() == 120


def test_configurable_ttl_priority():
    """Verify RAILRADAR_CACHE_TTL_SECONDS takes precedence over LIVE_DATA_CACHE_TTL_SECONDS."""
    with patch.dict(
        os.environ,
        {
            "RAILRADAR_CACHE_TTL_SECONDS": "240",
            "LIVE_DATA_CACHE_TTL_SECONDS": "90",
        },
        clear=True,
    ):
        assert get_configured_cache_ttl() == 240


def test_configurable_ttl_invalid_values_fallback():
    """Verify invalid or non-positive TTL strings fall back gracefully to default 60s."""
    for invalid_val in ["invalid", "-10", "0", "   "]:
        with patch.dict(os.environ, {"RAILRADAR_CACHE_TTL_SECONDS": invalid_val}, clear=True):
            assert get_configured_cache_ttl() == 60


# ---------------------------------------------------------------------------
# Requirement 1 & 2: Cache train responses & Do not request repeatedly within TTL
# ---------------------------------------------------------------------------

def test_no_repeated_external_api_calls_within_ttl(sample_payload):
    """Verify that repeated queries for the same train within TTL do NOT call external API."""
    cache = LiveTrainCache(ttl_seconds=60)
    client = RailRadarClient(api_key="mock_key", cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sample_payload

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        # First call: cache miss -> network call
        state1 = client.fetch_live_train("12302")
        assert state1.train_number == "12302"
        assert mock_get.call_count == 1

        # Second call: within TTL -> served from cache, network NOT called
        state2 = client.fetch_live_train("12302")
        assert state2.train_number == "12302"
        assert mock_get.call_count == 1  # Still 1!

        # Third call: within TTL -> served from cache, network NOT called
        state3 = client.fetch_live_train("12302")
        assert state3.train_number == "12302"
        assert mock_get.call_count == 1  # Still 1!

    # Check cache telemetry
    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["cached_trains_count"] == 1
    assert stats["hit_rate"] == 0.667


def test_expired_ttl_triggers_fresh_network_request(sample_payload):
    """Verify that once TTL expires, the client makes a fresh network call."""
    cache = LiveTrainCache(ttl_seconds=10)
    client = RailRadarClient(api_key="mock_key", cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sample_payload

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        # First call: populates cache
        client.fetch_live_train("12302")
        assert mock_get.call_count == 1

        # Artificially expire the cache entry
        entry = cache.get_entry("12302")
        assert entry is not None
        entry.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert entry.is_expired is True

        # Second call: entry expired -> fresh network call
        client.fetch_live_train("12302")
        assert mock_get.call_count == 2


def test_bypass_cache_flag(sample_payload):
    """Verify bypass_cache=True forces network fetch even if cache entry is fresh."""
    cache = LiveTrainCache(ttl_seconds=60)
    client = RailRadarClient(api_key="mock_key", cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sample_payload

    with patch("httpx.Client.get", return_value=mock_resp) as mock_get:
        client.fetch_live_train("12302")
        assert mock_get.call_count == 1

        # Bypass cache
        client.fetch_live_train("12302", bypass_cache=True)
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Requirement 6: Handle API HTTP 429 Gracefully
# ---------------------------------------------------------------------------

def test_http_429_serves_stale_cached_data_gracefully(sample_payload):
    """
    Verify that when RailRadar returns HTTP 429, the client gracefully serves
    stale cached data rather than failing.
    """
    cache = LiveTrainCache(ttl_seconds=10, rate_limit_cooldown_seconds=30)
    client = RailRadarClient(api_key="mock_key", cache=cache)

    # 1. Populate cache with valid data
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = sample_payload

    with patch("httpx.Client.get", return_value=ok_resp):
        state = client.fetch_live_train("12302")
        assert state.train_number == "12302"

    # Expire entry
    entry = cache.get_entry("12302")
    entry.expires_at = datetime.now(timezone.utc) - timedelta(seconds=5)

    # 2. Next call hits 429 Rate Limit
    rate_limited_resp = MagicMock()
    rate_limited_resp.status_code = 429
    rate_limited_resp.text = '{"error": "Too Many Requests"}'

    with patch("httpx.Client.get", return_value=rate_limited_resp) as mock_get:
        # Should gracefully return stale state without raising RateLimitExceededError
        stale_state = client.fetch_live_train("12302")
        assert stale_state.train_number == "12302"
        assert cache.is_rate_limited() is True
        assert entry.is_stale is True

    # 3. Subsequent call during cooldown should immediately serve stale data without network call
    with patch("httpx.Client.get") as mock_get2:
        stale_state2 = client.fetch_live_train("12302")
        assert stale_state2.train_number == "12302"
        mock_get2.assert_not_called()  # No network call during cooldown!


def test_http_429_without_cache_raises_controlled_error():
    """Verify that when 429 occurs and NO cached data exists, controlled RateLimitExceededError is raised."""
    cache = LiveTrainCache(ttl_seconds=60, rate_limit_cooldown_seconds=30)
    client = RailRadarClient(api_key="mock_key", cache=cache)

    rate_limited_resp = MagicMock()
    rate_limited_resp.status_code = 429
    rate_limited_resp.text = '{"error": "Rate limit exceeded"}'

    with patch("httpx.Client.get", return_value=rate_limited_resp):
        with pytest.raises(RateLimitExceededError) as exc_info:
            client.fetch_live_train("12302")
        assert "Rate limit exceeded (HTTP 429)" in str(exc_info.value)
        assert cache.is_rate_limited() is True


# ---------------------------------------------------------------------------
# Requirement 4: Clearly Display When Data Was Last Updated
# ---------------------------------------------------------------------------

def test_provenance_and_last_updated_metadata(sample_payload):
    """Verify that LiveTrainResult and live lookup display freshness, cached_at, age, and ttl."""
    cache = LiveTrainCache(ttl_seconds=60)
    client = RailRadarClient(api_key="mock_key", cache=cache)
    service = LiveTrainLookupService(railway_client=client)

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = sample_payload

    with patch("httpx.Client.get", return_value=ok_resp):
        # 1. First lookup: fresh from external API
        res1 = service.lookup_live_train("12302")
        assert res1.train_number == "12302"
        assert res1.is_cached is False
        assert res1.cached_at is not None
        assert res1.last_updated is not None
        assert res1.cache_ttl_seconds == 60
        assert res1.is_stale is False

        # 2. Second lookup: served from cache
        res2 = service.lookup_live_train("12302")
        assert res2.train_number == "12302"
        assert res2.is_cached is True
        assert res2.cached_at == res1.cached_at
        assert res2.cache_age_seconds is not None
        assert res2.cache_age_seconds >= 0.0
        assert res2.is_stale is False


def test_api_endpoint_freshness_display(sample_payload):
    """Verify GET /live/train/{train_number} returns last_updated, is_cached, and cache_age_seconds."""
    from backend.main import get_live_lookup_service

    cache = LiveTrainCache(ttl_seconds=60)
    client = RailRadarClient(api_key="mock_key", cache=cache)
    lookup_svc = LiveTrainLookupService(railway_client=client)

    app.dependency_overrides[get_live_lookup_service] = lambda: lookup_svc
    try:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_payload

        with patch("backend.services.railway_api_client.httpx.Client") as mock_client_cls:
            mock_inst = MagicMock()
            mock_inst.__enter__.return_value = mock_inst
            mock_inst.get.return_value = mock_resp
            mock_client_cls.return_value = mock_inst

            with TestClient(app) as test_client:
                # First call -> is_cached=False
                r1 = test_client.get("/live/train/12302")
                assert r1.status_code == 200
                data1 = r1.json()
                assert data1["train_number"] == "12302"
                assert "last_updated" in data1
                assert data1["last_updated"] is not None
                assert data1["is_cached"] is False

                # Second call -> is_cached=True
                r2 = test_client.get("/live/train/12302")
                assert r2.status_code == 200
                data2 = r2.json()
                assert data2["train_number"] == "12302"
                assert data2["is_cached"] is True
                assert "cache_age_seconds" in data2
                assert data2["cache_age_seconds"] is not None
    finally:
        app.dependency_overrides.pop(get_live_lookup_service, None)


# ---------------------------------------------------------------------------
# Requirement 5: Keep Simulator Mode Independent
# ---------------------------------------------------------------------------

def test_simulator_mode_does_not_use_or_affect_live_cache():
    """Verify simulator queries do not touch, read, or populate LiveTrainCache."""
    from backend.database.connection import SessionLocal

    cache = LiveTrainCache(ttl_seconds=60)
    client = RailRadarClient(api_key="mock_key", cache=cache)
    provider = TrainStateProvider(client=client, default_mode=DataSourceMode.SIMULATOR)

    db = SessionLocal()
    try:
        sim = TrainSimulator(db=db)
        sim.load_from_db(db)

        # Query simulator state
        result = provider.get_train_state("12302", db=db, sim=sim, mode_override="SIMULATOR")

        assert result.effective_mode == DataSourceMode.SIMULATOR
        assert result.state.source == "simulator"

        # Cache must remain empty
        stats = cache.stats()
        assert stats["cached_trains_count"] == 0
        assert stats["total_lookups"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
    finally:
        db.close()


def test_system_data_source_reports_cache_stats():
    """Verify GET /system/data-source exposes cache telemetry stats."""
    with TestClient(app) as test_client:
        resp = test_client.get("/system/data-source")
        assert resp.status_code == 200
        data = resp.json()
        assert "cache" in data
        cache_data = data["cache"]
        assert "ttl_seconds" in cache_data
        assert "hits" in cache_data
        assert "misses" in cache_data
        assert "cached_trains_count" in cache_data
