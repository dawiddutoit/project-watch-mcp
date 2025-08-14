"""
Main performance optimizer for project-watch-mcp.
Coordinates all optimization strategies and provides automatic tuning.
"""

import asyncio
import gc
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .batch_processor import BatchProcessor
from .cache_manager import CacheManager, MultiLevelCache
from .connection_pool import ConnectionPoolManager


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Batching
    enable_batching: bool = True
    batch_size: int = 100
    max_concurrent: int = 10
    
    # Connection pooling
    enable_connection_pooling: bool = True
    max_connections: int = 50
    min_connections: int = 10
    
    # Memory management
    enable_gc_optimization: bool = True
    gc_threshold: int = 700
    
    # Async optimization
    enable_async_optimization: bool = True
    async_timeout: float = 30.0


class PerformanceOptimizer:
    """Main performance optimizer coordinating all optimizations."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.cache_manager = None
        self.batch_processor = None
        self.connection_manager = None
        
        # Performance metrics
        self.metrics = {
            "optimizations_applied": [],
            "performance_gains": {},
            "current_settings": {}
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize optimization components."""
        # Cache manager
        if self.config.enable_caching:
            self.cache_manager = MultiLevelCache(
                l1_size=self.config.cache_size // 10,
                l2_size=self.config.cache_size,
                ttl_seconds=self.config.cache_ttl_seconds
            )
            self.metrics["optimizations_applied"].append("caching")
        
        # Batch processor
        if self.config.enable_batching:
            self.batch_processor = BatchProcessor(
                batch_size=self.config.batch_size,
                max_concurrent=self.config.max_concurrent
            )
            self.metrics["optimizations_applied"].append("batching")
        
        # GC optimization
        if self.config.enable_gc_optimization:
            self._optimize_garbage_collection()
            self.metrics["optimizations_applied"].append("gc_optimization")
        
        # Async optimization
        if self.config.enable_async_optimization:
            self._optimize_async_settings()
            self.metrics["optimizations_applied"].append("async_optimization")
    
    def _optimize_garbage_collection(self):
        """Optimize garbage collection settings."""
        # Adjust GC thresholds for better performance
        gc.set_threshold(
            self.config.gc_threshold,
            self.config.gc_threshold * 10,
            self.config.gc_threshold * 100
        )
        
        # Disable GC during critical sections
        self.metrics["current_settings"]["gc_threshold"] = gc.get_threshold()
    
    def _optimize_async_settings(self):
        """Optimize async event loop settings."""
        # Get current event loop
        try:
            loop = asyncio.get_running_loop()
            
            # Set debug mode off for production
            loop.set_debug(False)
            
            # Optimize slow callback threshold
            loop.slow_callback_duration = 0.1  # 100ms
            
            self.metrics["current_settings"]["async_debug"] = False
            self.metrics["current_settings"]["slow_callback_duration"] = 0.1
        except RuntimeError:
            # No running loop
            pass
    
    async def initialize_connection_pool(self, uri: str, auth: tuple):
        """Initialize optimized connection pool."""
        if self.config.enable_connection_pooling:
            self.connection_manager = ConnectionPoolManager(
                uri=uri,
                auth=auth,
                max_connections=self.config.max_connections,
                min_connections=self.config.min_connections
            )
            await self.connection_manager.initialize()
            self.metrics["optimizations_applied"].append("connection_pooling")
    
    async def optimize_vector_search(self, search_func, query: str, **kwargs):
        """Optimize vector search with caching."""
        if self.cache_manager:
            cache_key = f"vector_search_{query}_{str(kwargs)}"
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result is not None:
                self.metrics["performance_gains"]["cache_hits"] = \
                    self.metrics["performance_gains"].get("cache_hits", 0) + 1
                return cached_result
            
            result = await search_func(query, **kwargs)
            await self.cache_manager.set(cache_key, result)
            return result
        
        return await search_func(query, **kwargs)
    
    async def optimize_language_detection(self, detect_func, files: List[str]):
        """Optimize language detection with batching and caching."""
        if self.batch_processor and len(files) > 10:
            # Use batch processing for multiple files
            results = await self.batch_processor.process_batch(
                files,
                detect_func
            )
            self.metrics["performance_gains"]["batch_processed"] = \
                self.metrics["performance_gains"].get("batch_processed", 0) + len(files)
            return results
        
        # Process individually with caching
        results = []
        for file in files:
            if self.cache_manager:
                cache_key = f"lang_detect_{file}"
                cached_result = await self.cache_manager.get(cache_key)
                
                if cached_result is not None:
                    results.append(cached_result)
                    continue
                
                result = await detect_func(file)
                await self.cache_manager.set(cache_key, result)
                results.append(result)
            else:
                results.append(await detect_func(file))
        
        return results
    
    async def optimize_complexity_analysis(self, analyze_func, files: List[str]):
        """Optimize complexity analysis with parallel processing."""
        if self.batch_processor and len(files) > 5:
            # Process files in parallel
            results = await self.batch_processor.process_batch(
                files,
                analyze_func,
                batch_size=10  # Smaller batches for CPU-intensive work
            )
            return results
        
        # Process sequentially
        results = []
        for file in files:
            results.append(await analyze_func(file))
        
        return results
    
    async def optimize_database_queries(self, queries: List[tuple]):
        """Optimize database queries with connection pooling and batching."""
        if self.connection_manager and len(queries) > 1:
            # Use batch execution
            results = await self.connection_manager.execute_batch(queries)
            self.metrics["performance_gains"]["batch_queries"] = \
                self.metrics["performance_gains"].get("batch_queries", 0) + len(queries)
            return results
        
        # Execute individually
        results = []
        for query, params in queries:
            if self.connection_manager:
                result = await self.connection_manager.execute_query(query, params)
            else:
                # Fallback to regular execution
                result = await self._execute_query_fallback(query, params)
            results.append(result)
        
        return results
    
    async def _execute_query_fallback(self, query: str, params: dict):
        """Fallback query execution without optimization."""
        # This would use the regular Neo4j driver
        # Placeholder for actual implementation
        return []
    
    def auto_tune(self, workload_profile: Dict[str, Any]):
        """Automatically tune optimization settings based on workload."""
        # Analyze workload characteristics
        avg_file_size = workload_profile.get("avg_file_size", 1000)
        total_files = workload_profile.get("total_files", 100)
        query_frequency = workload_profile.get("query_frequency", 10)
        
        # Tune cache size
        if query_frequency > 100:
            self.config.cache_size = min(10000, total_files * 2)
            self.config.cache_ttl_seconds = 7200  # 2 hours
        elif query_frequency > 10:
            self.config.cache_size = min(1000, total_files)
            self.config.cache_ttl_seconds = 3600  # 1 hour
        
        # Tune batch size
        if avg_file_size > 10000:
            self.config.batch_size = 10  # Smaller batches for large files
        elif avg_file_size > 1000:
            self.config.batch_size = 50
        else:
            self.config.batch_size = 100
        
        # Tune connection pool
        if query_frequency > 100:
            self.config.max_connections = 100
            self.config.min_connections = 20
        elif query_frequency > 10:
            self.config.max_connections = 50
            self.config.min_connections = 10
        
        # Re-initialize components with new settings
        self._initialize_components()
        
        self.metrics["current_settings"]["cache_size"] = self.config.cache_size
        self.metrics["current_settings"]["batch_size"] = self.config.batch_size
        self.metrics["current_settings"]["max_connections"] = self.config.max_connections
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance optimization report."""
        report = {
            "optimizations_applied": self.metrics["optimizations_applied"],
            "current_settings": self.metrics["current_settings"],
            "performance_gains": self.metrics["performance_gains"]
        }
        
        # Add cache statistics
        if self.cache_manager:
            report["cache_stats"] = await self._get_cache_stats()
        
        # Add connection pool statistics
        if self.connection_manager:
            report["connection_stats"] = self.connection_manager.get_stats()
        
        # Add batch processing statistics
        if self.batch_processor:
            report["batch_stats"] = {
                "batch_size": self.batch_processor.batch_size,
                "max_concurrent": self.batch_processor.max_concurrent
            }
        
        # Calculate overall performance improvement
        if "cache_hits" in self.metrics["performance_gains"]:
            cache_benefit = self.metrics["performance_gains"]["cache_hits"] * 50  # 50ms saved per hit
            report["estimated_time_saved_ms"] = cache_benefit
        
        return report
    
    async def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if isinstance(self.cache_manager, MultiLevelCache):
            return self.cache_manager.get_stats()
        elif self.cache_manager:
            return self.cache_manager.get_stats()
        return {}
    
    def cleanup(self):
        """Clean up optimization resources."""
        if self.cache_manager:
            if isinstance(self.cache_manager, MultiLevelCache):
                self.cache_manager.l1_cache.clear()
                self.cache_manager.l2_cache.clear()
            else:
                self.cache_manager.clear()
        
        # Reset GC settings
        if self.config.enable_gc_optimization:
            gc.set_threshold(700, 10, 10)  # Reset to defaults