"""Unified complexity analysis system for multiple programming languages.

This module provides a plugin-based architecture for analyzing code complexity
across different programming languages while maintaining consistent interfaces
and metrics.
"""

from .base_analyzer import BaseComplexityAnalyzer, AnalyzerRegistry
from .models import (
    ComplexityResult,
    ComplexitySummary,
    FunctionComplexity,
    ClassComplexity,
    ComplexityGrade,
)

__all__ = [
    "BaseComplexityAnalyzer",
    "AnalyzerRegistry",
    "ComplexityResult",
    "ComplexitySummary",
    "FunctionComplexity",
    "ClassComplexity",
    "ComplexityGrade",
]

# Package metadata
__version__ = "0.1.0"