"""Caching interface and implementations for tool results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry[T]:
    """A cached entry with value and metadata."""

    value: T
    created_at: float
    expires_at: float | None = None
    access_count: int = 0
    last_accessed: float | None = None

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def mark_accessed(self) -> None:
        """Mark this entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()


class Cache[T](ABC):
    """Abstract base class for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """

    @abstractmethod
    async def set(self, key: str, value: T, ttl_seconds: float | None = None) -> None:
        """Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds, None for no expiration
        """

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            True if the key was deleted, False if not found
        """

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries from the cache."""

    @abstractmethod
    async def size(self) -> int:
        """Get the number of entries in the cache."""


class InMemoryCache[T](Cache[T]):
    """In-memory cache implementation with TTL support.

    Simple thread-safe cache that stores entries in memory with optional
    expiration times. Suitable for single-process applications.

    Example:
        >>> cache = InMemoryCache[str](max_size=100)
        >>> await cache.set("key", "value", ttl_seconds=300)
        >>> value = await cache.get("key")
        >>> print(value)  # "value"
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the in-memory cache.

        Args:
            max_size: Maximum number of entries to store
        """
        self._store: dict[str, CacheEntry[T]] = {}
        self._max_size = max_size
        logger.debug(f"InMemoryCache initialized with max_size={max_size}")

    async def get(self, key: str) -> T | None:
        """Get a value from the cache."""
        entry = self._store.get(key)
        if entry is None:
            logger.debug(f"Cache miss for key: {key}")
            return None

        if entry.is_expired():
            logger.debug(f"Cache entry expired for key: {key}")
            await self.delete(key)
            return None

        entry.mark_accessed()
        logger.debug(f"Cache hit for key: {key}")
        return entry.value

    async def set(self, key: str, value: T, ttl_seconds: float | None = None) -> None:
        """Set a value in the cache."""
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None

        # Evict oldest entries if cache is full
        if len(self._store) >= self._max_size and key not in self._store:
            await self._evict_oldest()

        entry = CacheEntry(
            value=value,
            created_at=now,
            expires_at=expires_at,
        )

        self._store[key] = entry
        logger.debug(
            f"Cached value for key: {key}, TTL: {ttl_seconds}s, "
            f"expires_at: {expires_at}"
        )

    async def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        if key in self._store:
            del self._store[key]
            logger.debug(f"Deleted cache entry for key: {key}")
            return True
        return False

    async def clear(self) -> None:
        """Clear all entries from the cache."""
        size_before = len(self._store)
        self._store.clear()
        logger.debug(f"Cleared cache, removed {size_before} entries")

    async def size(self) -> int:
        """Get the number of entries in the cache."""
        return len(self._store)

    async def _evict_oldest(self) -> None:
        """Evict the oldest entry from the cache."""
        if not self._store:
            return

        # Find the oldest entry by creation time
        oldest_key = min(self._store.keys(), key=lambda k: self._store[k].created_at)
        await self.delete(oldest_key)
        logger.debug(f"Evicted oldest cache entry: {oldest_key}")

    async def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of expired entries removed
        """
        expired_keys = [key for key, entry in self._store.items() if entry.is_expired()]

        for key in expired_keys:
            await self.delete(key)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self._store:
            return {
                "size": 0,
                "max_size": self._max_size,
                "hit_rate": 0.0,
                "expired_count": 0,
            }

        total_accesses = sum(entry.access_count for entry in self._store.values())
        expired_count = sum(1 for entry in self._store.values() if entry.is_expired())

        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "total_accesses": total_accesses,
            "expired_count": expired_count,
            "entries": [
                {
                    "created_at": entry.created_at,
                    "expires_at": entry.expires_at,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed,
                    "is_expired": entry.is_expired(),
                }
                for entry in self._store.values()
            ],
        }
