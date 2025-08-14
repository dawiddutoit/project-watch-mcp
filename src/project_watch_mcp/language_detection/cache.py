"""Caching layer for language detection results."""

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Optional, Any

from .models import LanguageDetectionResult


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    
    result: LanguageDetectionResult
    content_hash: str
    timestamp: float
    
    def is_valid(self, max_age_seconds: int) -> bool:
        """Check if cache entry is still valid based on age."""
        age = time.time() - self.timestamp
        return age < max_age_seconds


@dataclass
class CacheStatistics:
    """Statistics for cache performance monitoring."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    puts: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        # Calculate based on hits + misses if total_requests not manually set
        total = self.total_requests if self.total_requests > 0 else (self.hits + self.misses)
        if total == 0:
            return 0.0
        return self.hits / total
    
    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1
        self.total_requests += 1
    
    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1
        self.total_requests += 1
    
    def record_eviction(self) -> None:
        """Record a cache eviction."""
        self.evictions += 1
    
    def record_put(self) -> None:
        """Record a cache put operation."""
        self.puts += 1


class LanguageDetectionCache:
    """
    Thread-safe LRU cache for language detection results.
    
    Uses content hash as the primary key with optional file path
    for disambiguation.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_age_seconds: int = 3600
    ):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to cache
            max_age_seconds: Maximum age of cache entries in seconds
        """
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self.statistics = CacheStatistics()
        
        logger.info(
            f"Initialized LanguageDetectionCache with max_size={max_size}, "
            f"max_age_seconds={max_age_seconds}"
        )
    
    def _generate_hash(self, content: str) -> str:
        """
        Generate SHA256 hash of content.
        
        Args:
            content: The content to hash
        
        Returns:
            Hex string representation of the hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_cache_key(
        self,
        content: str,
        file_path: Optional[str] = None
    ) -> str:
        """
        Generate cache key from content and optional file path.
        
        Args:
            content: The content to detect language for
            file_path: Optional file path for disambiguation
        
        Returns:
            Cache key string
        """
        content_hash = self._generate_hash(content)
        
        if file_path:
            # Include file path in key for disambiguation
            combined = f"{content_hash}:{file_path}"
            return hashlib.sha256(combined.encode('utf-8')).hexdigest()
        
        return content_hash
    
    def get(
        self,
        content: str,
        file_path: Optional[str] = None
    ) -> Optional[LanguageDetectionResult]:
        """
        Retrieve a cached detection result.
        
        Args:
            content: The content to look up
            file_path: Optional file path for disambiguation
        
        Returns:
            Cached LanguageDetectionResult or None if not found/expired
        """
        key = self._get_cache_key(content, file_path)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if entry is still valid
                if entry.is_valid(self.max_age_seconds):
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self.statistics.record_hit()
                    
                    logger.debug(
                        f"Cache hit for key={key[:8]}..., "
                        f"language={entry.result.language}"
                    )
                    return entry.result
                else:
                    # Remove expired entry
                    del self._cache[key]
                    logger.debug(f"Removed expired entry for key={key[:8]}...")
            
            self.statistics.record_miss()
            logger.debug(f"Cache miss for key={key[:8]}...")
            return None
    
    def put(
        self,
        content: str,
        result: LanguageDetectionResult,
        file_path: Optional[str] = None
    ) -> None:
        """
        Store a detection result in the cache.
        
        Args:
            content: The content that was analyzed
            result: The detection result to cache
            file_path: Optional file path for disambiguation
        """
        key = self._get_cache_key(content, file_path)
        content_hash = self._generate_hash(content)
        
        with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Evict least recently used
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self.statistics.record_eviction()
                logger.debug(f"Evicted oldest entry: key={oldest_key[:8]}...")
            
            # Add or update entry
            entry = CacheEntry(
                result=result,
                content_hash=content_hash,
                timestamp=time.time()
            )
            
            # If key exists, remove it first to maintain LRU order
            if key in self._cache:
                del self._cache[key]
            
            # Add to end (most recently used)
            self._cache[key] = entry
            
            # Record the put operation
            self.statistics.record_put()
            
            logger.debug(
                f"Cached result for key={key[:8]}..., "
                f"language={result.language}, confidence={result.confidence:.2f}"
            )
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def reset_statistics(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self.statistics = CacheStatistics()
            logger.info("Cache statistics reset")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get cache information and statistics.
        
        Returns:
            Dictionary with cache info and statistics
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "max_age_seconds": self.max_age_seconds,
                "hit_rate": self.statistics.hit_rate,
                "statistics": {
                    "hits": self.statistics.hits,
                    "misses": self.statistics.misses,
                    "evictions": self.statistics.evictions,
                    "puts": self.statistics.puts,
                    "total_requests": self.statistics.total_requests
                }
            }
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self._cache.items():
                if not entry.is_valid(self.max_age_seconds):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)