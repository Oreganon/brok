"""Tests for the tools caching system."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from brok.tools.cache import CacheEntry, InMemoryCache


class TestCacheEntry:
    """Test cases for CacheEntry class."""

    def test_cache_entry_creation(self) -> None:
        """Test basic cache entry creation."""
        now = time.time()
        entry = CacheEntry(value="test", created_at=now)

        assert entry.value == "test"
        assert entry.created_at == now
        assert entry.expires_at is None
        assert entry.access_count == 0
        assert entry.last_accessed is None

    def test_cache_entry_with_expiration(self) -> None:
        """Test cache entry with expiration time."""
        now = time.time()
        expires_at = now + 300  # 5 minutes
        entry = CacheEntry(value="test", created_at=now, expires_at=expires_at)

        assert not entry.is_expired()
        assert entry.expires_at == expires_at

    def test_cache_entry_expired(self) -> None:
        """Test cache entry expiration detection."""
        now = time.time()
        expires_at = now - 1  # 1 second ago
        entry = CacheEntry(value="test", created_at=now, expires_at=expires_at)

        assert entry.is_expired()

    def test_cache_entry_no_expiration(self) -> None:
        """Test cache entry without expiration never expires."""
        now = time.time()
        entry = CacheEntry(value="test", created_at=now)

        assert not entry.is_expired()

    def test_mark_accessed(self) -> None:
        """Test marking entry as accessed."""
        entry = CacheEntry(value="test", created_at=time.time())

        # Initially not accessed
        assert entry.access_count == 0
        assert entry.last_accessed is None

        # Mark as accessed
        before_access = time.time()
        entry.mark_accessed()
        after_access = time.time()

        assert entry.access_count == 1
        assert entry.last_accessed is not None
        assert before_access <= entry.last_accessed <= after_access

        # Mark as accessed again
        entry.mark_accessed()
        assert entry.access_count == 2


class TestInMemoryCache:
    """Test cases for InMemoryCache implementation."""

    @pytest.fixture
    def cache(self) -> InMemoryCache[str]:
        """Create a fresh cache for each test."""
        return InMemoryCache[str](max_size=10)

    @pytest.mark.asyncio
    async def test_cache_basic_operations(self, cache: InMemoryCache[str]) -> None:
        """Test basic cache set/get operations."""
        # Cache miss
        result = await cache.get("nonexistent")
        assert result is None

        # Cache set
        await cache.set("key1", "value1")
        assert await cache.size() == 1

        # Cache hit
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_cache_with_ttl(self, cache: InMemoryCache[str]) -> None:
        """Test cache operations with TTL."""
        # Set with TTL
        await cache.set("key1", "value1", ttl_seconds=0.1)  # 100ms

        # Should be available immediately
        result = await cache.get("key1")
        assert result == "value1"

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Should be expired and removed
        result = await cache.get("key1")
        assert result is None
        assert await cache.size() == 0

    @pytest.mark.asyncio
    async def test_cache_delete(self, cache: InMemoryCache[str]) -> None:
        """Test cache deletion."""
        await cache.set("key1", "value1")
        assert await cache.size() == 1

        # Delete existing key
        deleted = await cache.delete("key1")
        assert deleted is True
        assert await cache.size() == 0

        # Delete non-existing key
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache: InMemoryCache[str]) -> None:
        """Test cache clearing."""
        # Add multiple entries
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        assert await cache.size() == 3

        # Clear cache
        await cache.clear()
        assert await cache.size() == 0

        # Verify all entries are gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_cache_max_size_eviction(self, cache: InMemoryCache[str]) -> None:
        """Test cache eviction when max size is reached."""
        # Fill cache to max size
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}")

        assert await cache.size() == 10

        # Add one more - should evict oldest
        await cache.set("key10", "value10")
        assert await cache.size() == 10

        # First entry should be evicted
        assert await cache.get("key0") is None
        assert await cache.get("key10") == "value10"

    @pytest.mark.asyncio
    async def test_cache_update_existing_key(self, cache: InMemoryCache[str]) -> None:
        """Test updating an existing cache key."""
        await cache.set("key1", "value1")
        assert await cache.size() == 1

        # Update same key
        await cache.set("key1", "value2")
        assert await cache.size() == 1
        assert await cache.get("key1") == "value2"

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache: InMemoryCache[str]) -> None:
        """Test cleaning up expired entries."""
        # Add entries with different TTLs
        await cache.set("key1", "value1", ttl_seconds=0.1)  # Will expire
        await cache.set("key2", "value2")  # No expiration
        await cache.set("key3", "value3", ttl_seconds=10)  # Long TTL

        assert await cache.size() == 3

        # Wait for first entry to expire
        await asyncio.sleep(0.15)

        # Cleanup expired entries
        removed_count = await cache.cleanup_expired()
        assert removed_count == 1
        assert await cache.size() == 2

        # Verify correct entries remain
        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache: InMemoryCache[str]) -> None:
        """Test cache statistics."""
        # Empty cache stats
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 10
        assert stats["expired_count"] == 0

        # Add entries and access them
        await cache.set("key1", "value1")
        await cache.set("key2", "value2", ttl_seconds=0.1)

        # Access first entry multiple times
        await cache.get("key1")
        await cache.get("key1")

        # Wait for second entry to expire
        await asyncio.sleep(0.15)

        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["total_accesses"] >= 2
        assert stats["expired_count"] == 1

        # Check entry details
        entries = stats["entries"]
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_cache_access_tracking(self, cache: InMemoryCache[str]) -> None:
        """Test that cache tracks access counts correctly."""
        await cache.set("key1", "value1")

        # Access the entry multiple times
        for _ in range(5):
            await cache.get("key1")

        stats = cache.get_stats()
        # Find the entry for key1
        key1_entry = None
        for entry in stats["entries"]:
            if entry["access_count"] == 5:
                key1_entry = entry
                break

        assert key1_entry is not None
        assert key1_entry["access_count"] == 5
        assert key1_entry["last_accessed"] is not None

    @pytest.mark.asyncio
    async def test_cache_generic_types(self) -> None:
        """Test cache with different generic types."""
        # String cache
        str_cache = InMemoryCache[str](max_size=5)
        await str_cache.set("key", "value")
        assert await str_cache.get("key") == "value"

        # Integer cache
        int_cache = InMemoryCache[int](max_size=5)
        await int_cache.set("key", 42)
        assert await int_cache.get("key") == 42

        # Dictionary cache
        dict_cache = InMemoryCache[dict[str, Any]](max_size=5)
        test_dict = {"nested": {"value": 123}}
        await dict_cache.set("key", test_dict)
        result = await dict_cache.get("key")
        assert result == test_dict

    @pytest.mark.asyncio
    async def test_cache_concurrent_access(self, cache: InMemoryCache[str]) -> None:
        """Test cache behavior under concurrent access."""
        await cache.set("key1", "value1")

        # Simulate concurrent access
        async def access_cache() -> str | None:
            return await cache.get("key1")

        tasks = [access_cache() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should return the same value
        assert all(result == "value1" for result in results)

        # Access count should reflect all accesses
        stats = cache.get_stats()
        assert stats["total_accesses"] == 10
