"""Language detection module for project-watch-mcp."""

from .models import (
    LanguageDetectionResult,
    DetectionMethod,
)
from .hybrid_detector import HybridLanguageDetector
from .cache import LanguageDetectionCache, CacheEntry, CacheStatistics

__all__ = [
    "HybridLanguageDetector",
    "LanguageDetectionResult",
    "DetectionMethod",
    "LanguageDetectionCache",
    "CacheEntry",
    "CacheStatistics",
]