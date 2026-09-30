"""
Live Railway Data In-Process Cache Layer.
=========================================
Provides thread-safe in-memory caching for live train telemetry with:
1. Configurable TTL via RAILRADAR_CACHE_TTL_SECONDS or LIVE_DATA_CACHE_TTL_SECONDS (default: 60s).
2. Elimination of repeated external API queries within the TTL window.
3. Graceful HTTP 429 handling via Stale-on-Error / Stale-While-Revalidate pattern.
4. Telemetry provenance tracking: cached_at, age_seconds, is_cached, is_stale.
5. Absolute independence from simulator mode (simulator does not read or touch this cache).
6. Lightweight pure-Python implementation (Zero Redis or external infrastructure).
"""

import os
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple

from backend.services.schemas import TrainRunningState

logger = logging.getLogger("dynamic_eta.cache")

DEFAULT_CACHE_TTL_SECONDS = 60  # Conservative hackathon default (60 seconds)
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 30  # Cooldown when HTTP 429 is encountered


def get_configured_cache_ttl() -> int:
    """
    Resolves cache TTL in seconds.
    Priority:
    1. RAILRADAR_CACHE_TTL_SECONDS env var
    2. LIVE_DATA_CACHE_TTL_SECONDS env var
    3. Default: 60 seconds
    """
    for var in ("RAILRADAR_CACHE_TTL_SECONDS", "LIVE_DATA_CACHE_TTL_SECONDS"):
        raw = os.getenv(var)
        if raw:
            try:
                val = int(raw.strip())
                if val > 0:
                    return val
            except ValueError:
                logger.warning("Invalid TTL '%s' in %s, falling back to default.", raw, var)
    return DEFAULT_CACHE_TTL_SECONDS


@dataclass
class CacheEntry:
    """Represents a cached train telemetry snapshot with freshness metadata."""
    state: TrainRunningState
    cached_at: datetime
    expires_at: datetime
    raw_payload: Optional[Dict[str, Any]] = None
    hit_count: int = 0
    is_stale: bool = False

    @property
    def is_expired(self) -> bool:
        """Returns True if the current UTC time exceeds the entry expiration."""
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def age_seconds(self) -> float:
        """Returns current age of this cache snapshot in seconds."""
        delta = (datetime.now(timezone.utc) - self.cached_at).total_seconds()
        return round(max(0.0, delta), 1)


class LiveTrainCache:
    """
    Thread-safe in-memory cache for live train states.
    Designed for conservative hackathon API usage.
    """

    def __init__(
        self,
        ttl_seconds: Optional[int] = None,
        rate_limit_cooldown_seconds: int = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    ):
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else get_configured_cache_ttl()
        self.rate_limit_cooldown_seconds = rate_limit_cooldown_seconds
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._rate_limit_until: Optional[datetime] = None

        # Telemetry metrics
        self._hits = 0
        self._misses = 0
        self._stale_hits = 0

    def get(
        self,
        train_number: str,
        allow_stale: bool = False,
    ) -> Tuple[Optional[TrainRunningState], Optional[CacheEntry]]:
        """
        Retrieves cached train running state.
        If valid and within TTL: returns (state, entry) with incremented hit count.
        If expired but allow_stale=True: returns (state, entry) marked as stale.
        If expired and allow_stale=False: returns (None, entry).
        If not in cache: returns (None, None).
        """
        clean_key = str(train_number).strip()
        with self._lock:
            entry = self._entries.get(clean_key)
            if not entry:
                self._misses += 1
                return None, None

            if not entry.is_expired:
                entry.hit_count += 1
                self._hits += 1
                return entry.state, entry

            # Entry is expired
            if allow_stale:
                entry.hit_count += 1
                entry.is_stale = True
                self._stale_hits += 1
                return entry.state, entry

            self._misses += 1
            return None, entry

    def get_entry(self, train_number: str) -> Optional[CacheEntry]:
        """Returns the raw CacheEntry for a train without mutating metrics."""
        clean_key = str(train_number).strip()
        with self._lock:
            return self._entries.get(clean_key)

    def set(
        self,
        train_number: str,
        state: TrainRunningState,
        raw_payload: Optional[Dict[str, Any]] = None,
        is_stale: bool = False,
    ) -> CacheEntry:
        """Stores a fresh or updated train state in cache."""
        clean_key = str(train_number).strip()
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=self.ttl_seconds)

        entry = CacheEntry(
            state=state,
            cached_at=now,
            expires_at=expires,
            raw_payload=raw_payload,
            hit_count=0,
            is_stale=is_stale,
        )

        with self._lock:
            self._entries[clean_key] = entry

        return entry

    def mark_rate_limited(self, cooldown_seconds: Optional[int] = None) -> None:
        """Registers an external API 429 event to initiate temporary cooldown."""
        cooldown = cooldown_seconds or self.rate_limit_cooldown_seconds
        now = datetime.now(timezone.utc)
        with self._lock:
            self._rate_limit_until = now + timedelta(seconds=cooldown)
        logger.warning(
            "API rate limited (HTTP 429). Live cache cooldown active for %ds.",
            cooldown,
        )

    def is_rate_limited(self) -> bool:
        """Checks if current time is within active 429 cooldown period."""
        with self._lock:
            if self._rate_limit_until is None:
                return False
            now = datetime.now(timezone.utc)
            if now < self._rate_limit_until:
                return True
            self._rate_limit_until = None
            return False

    def rate_limit_remaining_seconds(self) -> float:
        """Returns remaining seconds in rate limit cooldown (0.0 if not active)."""
        with self._lock:
            if not self._rate_limit_until:
                return 0.0
            delta = (self._rate_limit_until - datetime.now(timezone.utc)).total_seconds()
            return round(max(0.0, delta), 1)

    def clear(self) -> None:
        """Clears all cached telemetry entries."""
        with self._lock:
            self._entries.clear()
            self._rate_limit_until = None

    def stats(self) -> Dict[str, Any]:
        """Returns operational cache performance metrics."""
        with self._lock:
            total_lookups = self._hits + self._misses
            hit_rate = round(self._hits / total_lookups, 3) if total_lookups > 0 else 0.0
            return {
                "ttl_seconds": self.ttl_seconds,
                "cached_trains_count": len(self._entries),
                "cached_train_numbers": list(self._entries.keys()),
                "total_lookups": total_lookups,
                "hits": self._hits,
                "misses": self._misses,
                "stale_hits": self._stale_hits,
                "hit_rate": hit_rate,
                "is_rate_limited": self.is_rate_limited(),
                "rate_limit_remaining_seconds": self.rate_limit_remaining_seconds(),
            }


_shared_cache_instance: Optional[LiveTrainCache] = None


def get_shared_cache() -> LiveTrainCache:
    """Returns the shared process-wide LiveTrainCache singleton."""
    global _shared_cache_instance
    if _shared_cache_instance is None:
        _shared_cache_instance = LiveTrainCache()
    return _shared_cache_instance

