"""
Advanced caching system with TTL, LRU, and size management.
"""

import asyncio
import hashlib
import pickle
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple


class CacheManager:
    """Advanced cache manager with multiple eviction strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        max_memory_mb: float = 100
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_mb = max_memory_mb
        
        self.cache: OrderedDict[str, Tuple[Any, float, int]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.total_memory_bytes = 0
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1000  # Default estimate
    
    def _make_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            value, timestamp, size = self.cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                self.evict(key)
                self.miss_count += 1
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.hit_count += 1
            return value
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        size = self._estimate_size(value)
        
        # Check memory limit
        while self.total_memory_bytes + size > self.max_memory_mb * 1024 * 1024:
            if not self.cache:
                break
            oldest_key = next(iter(self.cache))
            self.evict(oldest_key)
        
        # Check size limit
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self.evict(oldest_key)
        
        # Store value
        self.cache[key] = (value, time.time(), size)
        self.total_memory_bytes += size
    
    def evict(self, key: str):
        """Evict item from cache."""
        if key in self.cache:
            _, _, size = self.cache[key]
            del self.cache[key]
            self.total_memory_bytes -= size
            self.eviction_count += 1
    
    def clear(self):
        """Clear entire cache."""
        self.cache.clear()
        self.total_memory_bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "memory_mb": self.total_memory_bytes / 1024 / 1024,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "eviction_count": self.eviction_count,
            "hit_rate": hit_rate,
            "avg_item_size_kb": self.total_memory_bytes / len(self.cache) / 1024 if self.cache else 0
        }
    
    def cache_decorator(self, func: Callable) -> Callable:
        """Decorator to cache function results."""
        if asyncio.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                key = self._make_key(func.__name__, *args, **kwargs)
                result = self.get(key)
                
                if result is None:
                    result = await func(*args, **kwargs)
                    self.set(key, result)
                
                return result
        else:
            def wrapper(*args, **kwargs):
                key = self._make_key(func.__name__, *args, **kwargs)
                result = self.get(key)
                
                if result is None:
                    result = func(*args, **kwargs)
                    self.set(key, result)
                
                return result
        
        return wrapper


class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (disk) layers."""
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        ttl_seconds: int = 3600
    ):
        self.l1_cache = CacheManager(max_size=l1_size, ttl_seconds=ttl_seconds)
        self.l2_cache = CacheManager(max_size=l2_size, ttl_seconds=ttl_seconds)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        # Check L1
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Check L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.set(key, value)
            return value
        
        return None
    
    async def set(self, key: str, value: Any):
        """Set value in multi-level cache."""
        # Always set in L1
        self.l1_cache.set(key, value)
        
        # Set in L2 for larger persistence
        self.l2_cache.set(key, value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get multi-level cache statistics."""
        return {
            "l1": self.l1_cache.get_stats(),
            "l2": self.l2_cache.get_stats(),
            "combined_hit_rate": self._calculate_combined_hit_rate()
        }
    
    def _calculate_combined_hit_rate(self) -> float:
        """Calculate combined hit rate across levels."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        total_hits = l1_stats["hit_count"] + l2_stats["hit_count"]
        total_requests = (l1_stats["hit_count"] + l1_stats["miss_count"])
        
        return total_hits / total_requests if total_requests > 0 else 0