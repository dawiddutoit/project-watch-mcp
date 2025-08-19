"""Comprehensive tests for optimization modules to improve coverage."""

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.optimization.batch_processor import BatchProcessor
from project_watch_mcp.optimization.cache_manager import CacheManager
from project_watch_mcp.optimization.connection_pool import ConnectionPoolManager
from project_watch_mcp.optimization.optimizer import PerformanceOptimizer, OptimizationConfig


class TestBatchProcessor:
    """Tests for BatchProcessor class."""
    
    def test_initialization(self):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(batch_size=32, max_concurrent=5)
        assert processor.batch_size == 32
        assert processor.max_concurrent == 5
        assert processor.semaphore._value == 5
    
    @pytest.mark.asyncio
    async def test_process_batch_sync_function(self):
        """Test batch processing with synchronous function."""
        processor = BatchProcessor(batch_size=2)
        items = [1, 2, 3, 4, 5]
        
        def double(x):
            return x * 2
        
        results = await processor.process_batch(items, double)
        assert results == [2, 4, 6, 8, 10]
    
    @pytest.mark.asyncio
    async def test_process_batch_async_function(self):
        """Test batch processing with async function."""
        processor = BatchProcessor(batch_size=2)
        items = ["a", "b", "c"]
        
        async def async_upper(x):
            await asyncio.sleep(0.001)
            return x.upper()
        
        results = await processor.process_batch(items, async_upper)
        assert results == ["A", "B", "C"]
    
    @pytest.mark.asyncio
    async def test_process_batch_with_custom_size(self):
        """Test batch processing with custom batch size."""
        processor = BatchProcessor(batch_size=10)
        items = list(range(5))
        
        def square(x):
            return x ** 2
        
        results = await processor.process_batch(items, square, batch_size=2)
        assert results == [0, 1, 4, 9, 16]
    
    @pytest.mark.asyncio
    async def test_process_batch_with_exceptions(self):
        """Test batch processing handles exceptions gracefully."""
        processor = BatchProcessor()
        items = [1, 2, 0, 4]
        
        def divide_by_item(x):
            return 10 / x  # Will raise ZeroDivisionError for x=0
        
        results = await processor.process_batch(items, divide_by_item)
        assert len(results) == 3  # One item caused exception
        assert 10.0 in results  # 10/1
        assert 5.0 in results   # 10/2
        assert 2.5 in results   # 10/4
    
    @pytest.mark.asyncio
    async def test_concurrent_limiting(self):
        """Test that concurrent operations are limited."""
        processor = BatchProcessor(batch_size=10, max_concurrent=2)
        items = list(range(5))
        
        concurrent_count = 0
        max_concurrent_seen = 0
        
        async def track_concurrent(x):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return x
        
        await processor.process_batch(items, track_concurrent)
        assert max_concurrent_seen <= 2
    
    @pytest.mark.asyncio
    async def test_aggregate_batches(self):
        """Test batch aggregation method."""
        processor = BatchProcessor()
        items = list(range(10))
        
        async def sum_batch(batch):
            return sum(batch)
        
        results = await processor.aggregate_batches(items, sum_batch, batch_size=3)
        # Batches: [0,1,2]=3, [3,4,5]=12, [6,7,8]=21, [9]=9
        assert results == [3, 12, 21, 9]
    
    @pytest.mark.asyncio
    async def test_map_reduce(self):
        """Test map-reduce functionality."""
        processor = BatchProcessor()
        items = ["hello", "world", "test"]
        
        def mapper(item):
            return len(item)
        
        def reducer(lengths):
            return sum(lengths)
        
        result = await processor.map_reduce(items, mapper, reducer)
        assert result == 14  # 5 + 5 + 4


class TestCacheManager:
    """Tests for CacheManager class."""
    
    def test_initialization(self):
        """Test CacheManager initialization."""
        cache = CacheManager(max_size=100, ttl=300)
        assert cache.max_size == 100
        assert cache.ttl == 300
        assert len(cache._cache) == 0
        assert cache._hits == 0
        assert cache._misses == 0
    
    def test_get_set(self):
        """Test getting and setting cache values."""
        cache = CacheManager()
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Non-existent key
        assert cache.get("nonexistent") is None
        assert cache.get("nonexistent", default="default") == "default"
    
    def test_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = CacheManager()
        
        # Set with very short TTL
        cache.set("key1", "value1", ttl=0.001)
        
        # Wait for expiration
        import time
        time.sleep(0.002)
        
        assert cache.get("key1") is None
    
    def test_max_size_eviction(self):
        """Test cache eviction when max size reached."""
        cache = CacheManager(max_size=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_clear(self):
        """Test clearing cache."""
        cache = CacheManager()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.get("key1") is None
    
    def test_delete(self):
        """Test deleting specific cache entry."""
        cache = CacheManager()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.delete("key1")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
    
    def test_exists(self):
        """Test checking cache entry existence."""
        cache = CacheManager()
        cache.set("key1", "value1")
        
        assert cache.exists("key1")
        assert not cache.exists("key2")
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = CacheManager(max_size=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_lru_behavior(self):
        """Test LRU eviction behavior."""
        cache = CacheManager(max_size=3)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new item, should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Still exists
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"  # Still exists
        assert cache.get("key4") == "value4"  # New item


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""
    
    def test_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor.metrics == {}
        assert monitor.is_enabled
    
    def test_record_metric(self):
        """Test recording performance metrics."""
        monitor = PerformanceMonitor()
        
        monitor.record("operation_time", 0.5)
        monitor.record("operation_time", 0.3)
        monitor.record("memory_usage", 1024)
        
        assert "operation_time" in monitor.metrics
        assert len(monitor.metrics["operation_time"]) == 2
        assert len(monitor.metrics["memory_usage"]) == 1
    
    def test_get_average(self):
        """Test getting average metric value."""
        monitor = PerformanceMonitor()
        
        monitor.record("latency", 10)
        monitor.record("latency", 20)
        monitor.record("latency", 30)
        
        assert monitor.get_average("latency") == 20
        assert monitor.get_average("nonexistent") == 0
    
    def test_get_percentile(self):
        """Test getting percentile values."""
        monitor = PerformanceMonitor()
        
        for i in range(100):
            monitor.record("response_time", i)
        
        assert monitor.get_percentile("response_time", 50) == 49.5  # Median
        assert monitor.get_percentile("response_time", 95) == 94.5  # 95th percentile
        assert monitor.get_percentile("nonexistent", 50) == 0
    
    def test_get_summary(self):
        """Test getting performance summary."""
        monitor = PerformanceMonitor()
        
        monitor.record("api_call", 100)
        monitor.record("api_call", 200)
        monitor.record("api_call", 300)
        
        summary = monitor.get_summary("api_call")
        assert summary["count"] == 3
        assert summary["mean"] == 200
        assert summary["min"] == 100
        assert summary["max"] == 300
        assert "std" in summary
    
    def test_clear_metrics(self):
        """Test clearing metrics."""
        monitor = PerformanceMonitor()
        monitor.record("metric1", 10)
        monitor.record("metric2", 20)
        
        monitor.clear()
        assert len(monitor.metrics) == 0
    
    def test_context_manager(self):
        """Test using monitor as context manager."""
        monitor = PerformanceMonitor()
        
        with monitor.measure("operation"):
            import time
            time.sleep(0.01)
        
        assert "operation" in monitor.metrics
        assert monitor.metrics["operation"][0] >= 0.01
    
    def test_decorator(self):
        """Test using monitor as decorator."""
        monitor = PerformanceMonitor()
        
        @monitor.measure_function("test_func")
        def slow_function():
            import time
            time.sleep(0.01)
            return "result"
        
        result = slow_function()
        
        assert result == "result"
        assert "test_func" in monitor.metrics
        assert monitor.metrics["test_func"][0] >= 0.01
    
    def test_disable_enable(self):
        """Test disabling and enabling monitor."""
        monitor = PerformanceMonitor()
        
        monitor.disable()
        assert not monitor.is_enabled
        
        monitor.record("metric", 10)
        assert "metric" not in monitor.metrics
        
        monitor.enable()
        assert monitor.is_enabled
        
        monitor.record("metric", 20)
        assert "metric" in monitor.metrics
    
    def test_export_metrics(self):
        """Test exporting metrics."""
        monitor = PerformanceMonitor()
        
        monitor.record("metric1", 10)
        monitor.record("metric1", 20)
        monitor.record("metric2", 30)
        
        exported = monitor.export()
        assert "metric1" in exported
        assert "metric2" in exported
        assert exported["metric1"]["count"] == 2
        assert exported["metric2"]["count"] == 1
    
    def test_threshold_alerting(self):
        """Test threshold-based alerting."""
        monitor = PerformanceMonitor()
        
        # Set threshold
        monitor.set_threshold("response_time", max_value=100)
        
        # Record values
        monitor.record("response_time", 50)  # OK
        violations = monitor.check_thresholds()
        assert len(violations) == 0
        
        monitor.record("response_time", 150)  # Violation
        violations = monitor.check_thresholds()
        assert len(violations) == 1
        assert violations[0]["metric"] == "response_time"
        assert violations[0]["value"] == 150
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        monitor = PerformanceMonitor()
        
        with monitor.track_memory("operation"):
            # Allocate some memory
            data = [i for i in range(10000)]
        
        assert "operation_memory" in monitor.metrics
        assert monitor.metrics["operation_memory"][0] > 0