"""Unit tests for language detection caching layer."""

import hashlib
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from threading import Thread

from src.project_watch_mcp.language_detection import (
    LanguageDetectionCache,
    CacheEntry,
    CacheStatistics,
    LanguageDetectionResult,
    DetectionMethod
)


class TestCacheEntry:
    """Test the CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        result = LanguageDetectionResult(
            language="python",
            confidence=0.95,
            method=DetectionMethod.TREE_SITTER
        )
        entry = CacheEntry(
            result=result,
            content_hash="abc123",
            timestamp=time.time()
        )
        assert entry.result == result
        assert entry.content_hash == "abc123"
        assert entry.timestamp > 0
    
    def test_cache_entry_is_valid(self):
        """Test cache entry validity check."""
        result = LanguageDetectionResult("python", 0.95, DetectionMethod.TREE_SITTER)
        entry = CacheEntry(
            result=result,
            content_hash="abc123",
            timestamp=time.time()
        )
        # Fresh entry should be valid
        assert entry.is_valid(max_age_seconds=3600)
        
        # Old entry should be invalid
        old_entry = CacheEntry(
            result=result,
            content_hash="abc123",
            timestamp=time.time() - 7200  # 2 hours ago
        )
        assert not old_entry.is_valid(max_age_seconds=3600)


class TestCacheStatistics:
    """Test the CacheStatistics dataclass."""
    
    def test_statistics_initialization(self):
        """Test statistics initialization."""
        stats = CacheStatistics()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.total_requests == 0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStatistics(hits=90, misses=10)
        assert stats.hit_rate == 0.9
        
        # Test with no requests
        empty_stats = CacheStatistics()
        assert empty_stats.hit_rate == 0.0
    
    def test_statistics_increment(self):
        """Test incrementing statistics."""
        stats = CacheStatistics()
        stats.record_hit()
        assert stats.hits == 1
        assert stats.total_requests == 1
        
        stats.record_miss()
        assert stats.misses == 1
        assert stats.total_requests == 2
        
        stats.record_eviction()
        assert stats.evictions == 1


class TestLanguageDetectionCache:
    """Test the LanguageDetectionCache class."""
    
    @pytest.fixture
    def cache(self):
        """Create a cache instance."""
        return LanguageDetectionCache(max_size=100, max_age_seconds=3600)
    
    def test_cache_initialization(self):
        """Test cache initialization with custom parameters."""
        cache = LanguageDetectionCache(max_size=50, max_age_seconds=1800)
        assert cache.max_size == 50
        assert cache.max_age_seconds == 1800
        assert len(cache._cache) == 0
        assert cache.statistics.total_requests == 0
    
    def test_content_hash_generation(self):
        """Test content hash generation."""
        cache = LanguageDetectionCache()
        
        # Same content should produce same hash
        content1 = "def hello(): pass"
        hash1 = cache._generate_hash(content1)
        hash2 = cache._generate_hash(content1)
        assert hash1 == hash2
        
        # Different content should produce different hash
        content2 = "function hello() {}"
        hash3 = cache._generate_hash(content2)
        assert hash1 != hash3
        
        # Hash should be a hex string
        assert all(c in '0123456789abcdef' for c in hash1)
    
    def test_cache_key_generation(self):
        """Test cache key generation with file path."""
        cache = LanguageDetectionCache()
        
        # Key with content only
        key1 = cache._get_cache_key("def hello(): pass")
        assert key1 is not None
        
        # Key with content and file path
        key2 = cache._get_cache_key("def hello(): pass", "test.py")
        assert key2 is not None
        assert key1 != key2  # Different keys with/without file path
        
        # Same content and path produce same key
        key3 = cache._get_cache_key("def hello(): pass", "test.py")
        assert key2 == key3
    
    def test_cache_get_and_put(self):
        """Test basic cache get and put operations."""
        cache = LanguageDetectionCache()
        
        content = "def hello(): pass"
        result = LanguageDetectionResult("python", 0.95, DetectionMethod.TREE_SITTER)
        
        # Initially cache should miss
        cached = cache.get(content)
        assert cached is None
        assert cache.statistics.misses == 1
        assert cache.statistics.hits == 0
        
        # Put result in cache
        cache.put(content, result)
        
        # Now cache should hit
        cached = cache.get(content)
        assert cached == result
        assert cache.statistics.hits == 1
        assert cache.statistics.misses == 1
    
    def test_cache_with_file_path(self):
        """Test caching with file path consideration."""
        cache = LanguageDetectionCache()
        
        content = "print('hello')"
        result_py = LanguageDetectionResult("python", 0.95, DetectionMethod.EXTENSION)
        result_rb = LanguageDetectionResult("ruby", 0.85, DetectionMethod.EXTENSION)
        
        # Cache with different file paths
        cache.put(content, result_py, "test.py")
        cache.put(content, result_rb, "test.rb")
        
        # Retrieve with correct file path
        assert cache.get(content, "test.py") == result_py
        assert cache.get(content, "test.rb") == result_rb
        
        # Without file path should miss (different key)
        assert cache.get(content) is None
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = LanguageDetectionCache(max_age_seconds=1)  # 1 second expiry
        
        content = "def hello(): pass"
        result = LanguageDetectionResult("python", 0.95, DetectionMethod.TREE_SITTER)
        
        cache.put(content, result)
        assert cache.get(content) == result  # Should hit
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should miss due to expiration
        assert cache.get(content) is None
        assert cache.statistics.misses == 1
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LanguageDetectionCache(max_size=3)
        
        # Fill cache
        results = []
        for i in range(3):
            content = f"content_{i}"
            result = LanguageDetectionResult(f"lang_{i}", 0.9, DetectionMethod.TREE_SITTER)
            results.append((content, result))
            cache.put(content, result)
        
        # All should be in cache
        for content, result in results:
            assert cache.get(content) == result
        
        # Add one more - should evict least recently used (content_0)
        cache.put("content_3", LanguageDetectionResult("lang_3", 0.9, DetectionMethod.TREE_SITTER))
        
        # First item should be evicted
        assert cache.get("content_0") is None
        assert cache.statistics.evictions == 1
        
        # Others should still be there
        assert cache.get("content_1") is not None
        assert cache.get("content_2") is not None
        assert cache.get("content_3") is not None
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = LanguageDetectionCache()
        
        # Add some entries
        for i in range(5):
            cache.put(f"content_{i}", LanguageDetectionResult(f"lang_{i}", 0.9, DetectionMethod.TREE_SITTER))
        
        assert len(cache._cache) == 5
        
        # Generate some statistics first
        cache.get("content_0")  # This will be a hit
        cache.get("content_missing")  # This will be a miss
        
        old_hits = cache.statistics.hits
        old_misses = cache.statistics.misses
        
        # Clear cache
        cache.clear()
        assert len(cache._cache) == 0
        
        # Statistics should be preserved after clear
        assert cache.statistics.hits == old_hits
        assert cache.statistics.misses == old_misses
    
    def test_cache_statistics_reset(self):
        """Test resetting cache statistics."""
        cache = LanguageDetectionCache()
        
        # Generate some statistics
        cache.get("content1")  # miss
        cache.put("content1", LanguageDetectionResult("python", 0.9, DetectionMethod.TREE_SITTER))
        cache.get("content1")  # hit
        
        assert cache.statistics.hits == 1
        assert cache.statistics.misses == 1
        
        # Reset statistics
        cache.reset_statistics()
        assert cache.statistics.hits == 0
        assert cache.statistics.misses == 0
        assert cache.statistics.total_requests == 0
    
    def test_cache_thread_safety(self):
        """Test thread safety of cache operations."""
        cache = LanguageDetectionCache()
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(100):
                    content = f"thread_{thread_id}_content_{i}"
                    result = LanguageDetectionResult(f"lang_{thread_id}", 0.9, DetectionMethod.TREE_SITTER)
                    
                    # Interleave put and get operations
                    cache.put(content, result)
                    retrieved = cache.get(content)
                    
                    # Verify we get back what we put
                    if retrieved != result:
                        errors.append(f"Thread {thread_id}: Mismatch at iteration {i}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Run multiple threads
        threads = [Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have no errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    def test_cache_size_limit(self):
        """Test that cache respects size limit."""
        cache = LanguageDetectionCache(max_size=10)
        
        # Add more entries than max_size
        for i in range(20):
            cache.put(f"content_{i}", LanguageDetectionResult(f"lang_{i}", 0.9, DetectionMethod.TREE_SITTER))
        
        # Cache size should not exceed max_size
        assert len(cache._cache) <= cache.max_size
        assert cache.statistics.evictions == 10  # Should have evicted 10 entries
    
    def test_cache_get_info(self):
        """Test getting cache information."""
        cache = LanguageDetectionCache(max_size=100)
        
        # Add some entries
        for i in range(5):
            cache.put(f"content_{i}", LanguageDetectionResult(f"lang_{i}", 0.9, DetectionMethod.TREE_SITTER))
        
        info = cache.get_info()
        assert info["size"] == 5
        assert info["max_size"] == 100
        assert info["max_age_seconds"] == cache.max_age_seconds
        assert info["hit_rate"] == 0.0  # No hits yet
        assert "statistics" in info
    
    @pytest.mark.parametrize("content,expected_hash_prefix", [
        ("def hello(): pass", None),  # Hash will be calculated
        ("function hello() {}", None),
        ("public class Test {}", None),
    ])
    def test_cache_deterministic_hashing(self, content, expected_hash_prefix):
        """Test that hashing is deterministic."""
        cache = LanguageDetectionCache()
        
        # Generate hash multiple times
        hashes = [cache._generate_hash(content) for _ in range(5)]
        
        # All hashes should be identical
        assert all(h == hashes[0] for h in hashes)
        
        # Hash should be 64 characters (SHA256 hex)
        assert len(hashes[0]) == 64