"""Abstract base class and registry for language-specific complexity analyzers.

This module provides the foundation for implementing complexity analysis
across multiple programming languages with a consistent interface.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Tuple

from .models import ComplexityResult, ComplexityValidation

logger = logging.getLogger(__name__)


class BaseComplexityAnalyzer(ABC):
    """Abstract base class for language-specific complexity analyzers.
    
    All language analyzers must inherit from this class and implement
    the required methods for analyzing code complexity.
    """
    
    def __init__(self, language: str):
        """Initialize the analyzer with language information.
        
        Args:
            language: Programming language this analyzer supports
        """
        self.language = language
        self._cache: Dict[str, Any] = {}
    
    @abstractmethod
    async def analyze_file(self, file_path: Path) -> ComplexityResult:
        """Analyze complexity of a file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            ComplexityResult containing all metrics
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        pass
    
    @abstractmethod
    async def analyze_code(self, code: str) -> ComplexityResult:
        """Analyze complexity of code string.
        
        Args:
            code: Source code to analyze
            
        Returns:
            ComplexityResult containing all metrics
        """
        pass
    
    @abstractmethod
    def calculate_cyclomatic_complexity(self, ast_node: Any) -> int:
        """Calculate cyclomatic complexity for an AST node.
        
        Cyclomatic complexity measures the number of linearly independent
        paths through a program's source code.
        
        Args:
            ast_node: Abstract syntax tree node
            
        Returns:
            Cyclomatic complexity score
        """
        pass
    
    @abstractmethod
    def calculate_cognitive_complexity(self, ast_node: Any) -> int:
        """Calculate cognitive complexity for an AST node.
        
        Cognitive complexity measures how difficult code is to understand,
        taking into account nesting, control flow, and other factors.
        
        Args:
            ast_node: Abstract syntax tree node
            
        Returns:
            Cognitive complexity score
        """
        pass
    
    async def validate_analysis(self, result: ComplexityResult) -> ComplexityValidation:
        """Validate the accuracy of analysis results.
        
        Checks for common issues and validates metric calculations.
        
        Args:
            result: ComplexityResult to validate
            
        Returns:
            ComplexityValidation with status and any issues found
        """
        issues = []
        warnings = []
        
        # Check for negative values
        if result.summary.total_complexity < 0:
            issues.append("Total complexity cannot be negative")
        
        if result.summary.cognitive_complexity < 0:
            issues.append("Cognitive complexity cannot be negative")
            
        # Check for impossible values
        if result.summary.maintainability_index > 100:
            issues.append("Maintainability index cannot exceed 100")
        
        if result.summary.comment_ratio > 1.0:
            issues.append("Comment ratio cannot exceed 1.0")
            
        # Check function metrics
        for func in result.functions:
            if func.complexity < 1:
                issues.append(f"Function {func.name} has complexity < 1")
            if func.cognitive_complexity < 0:
                issues.append(f"Function {func.name} has negative cognitive complexity")
                
        # Warnings for suspicious values
        if result.summary.average_complexity > 50:
            warnings.append("Extremely high average complexity detected")
            
        if result.summary.function_count == 0 and result.summary.lines_of_code > 10:
            warnings.append("No functions found in non-trivial code")
        
        return ComplexityValidation(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )
    
    def get_language_specific_metrics(self) -> Dict[str, Any]:
        """Return language-specific metric definitions.
        
        Returns:
            Dictionary of language-specific metrics and thresholds
        """
        return {
            "language": self.language,
            "cyclomatic_thresholds": {
                "simple": 10,
                "moderate": 20,
                "complex": 50
            },
            "cognitive_thresholds": {
                "simple": 15,
                "moderate": 30,
                "complex": 60
            },
            "maintainability_thresholds": {
                "excellent": 80,
                "good": 60,
                "fair": 40,
                "poor": 20
            }
        }
    
    async def analyze_with_context(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ComplexityResult:
        """Analyze code with additional context information.
        
        Context can include information about:
        - File path for better recommendations
        - Project type (library, application, test)
        - Team preferences for complexity thresholds
        
        Args:
            code: Source code to analyze
            context: Optional context information
            
        Returns:
            ComplexityResult with context-aware recommendations
        """
        # Default implementation delegates to analyze_code
        result = await self.analyze_code(code)
        
        # Apply context-aware adjustments if provided
        if context:
            result.metadata["context"] = context
            
            # Adjust recommendations based on context
            if context.get("is_test_file"):
                # Test files can have higher complexity
                result.metadata["adjusted_thresholds"] = True
                
            if context.get("project_type") == "library":
                # Libraries should have lower complexity
                if result.summary.average_complexity > 5:
                    result.recommendations.append(
                        "Library code should maintain low complexity for ease of use"
                    )
        
        return result
    
    def calculate_maintainability_index(
        self,
        cyclomatic_complexity: int,
        lines_of_code: int,
        comment_ratio: float = 0.0
    ) -> float:
        """Calculate maintainability index.
        
        The Maintainability Index is calculated using the formula:
        MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC) + 50 * sin(sqrt(2.4 * CM))
        
        Where:
        - V = Halstead Volume (simplified here)
        - CC = Cyclomatic Complexity
        - LOC = Lines of Code
        - CM = Comment ratio
        
        This is a simplified version that doesn't require Halstead metrics.
        
        Args:
            cyclomatic_complexity: Cyclomatic complexity score
            lines_of_code: Number of lines of code
            comment_ratio: Ratio of comment lines to code lines
            
        Returns:
            Maintainability index (0-100, higher is better)
        """
        import math
        
        # Simplified calculation without Halstead volume
        # Using approximation based on cyclomatic complexity and LOC
        if lines_of_code == 0:
            return 100.0
        
        # Base calculation
        mi = 171.0
        
        # Cyclomatic complexity factor
        mi -= 0.23 * cyclomatic_complexity
        
        # Lines of code factor
        if lines_of_code > 0:
            mi -= 16.2 * math.log(lines_of_code)
        
        # Comment factor (bonus for documentation)
        if comment_ratio > 0:
            mi += 50 * math.sin(math.sqrt(2.4 * comment_ratio))
        
        # Normalize to 0-100 range
        mi = max(0, min(100, mi))
        
        return mi
    
    def clear_cache(self) -> None:
        """Clear any cached analysis data."""
        self._cache.clear()


class AnalyzerRegistry:
    """Registry for managing language-specific analyzers.
    
    Provides a factory pattern for creating appropriate analyzers
    based on file extension or explicit language specification.
    """
    
    _analyzers: Dict[str, Type[BaseComplexityAnalyzer]] = {}
    _instances: Dict[str, BaseComplexityAnalyzer] = {}
    
    @classmethod
    def register(cls, language: str, analyzer_class: Type[BaseComplexityAnalyzer]) -> None:
        """Register a new analyzer for a language.
        
        Args:
            language: Language identifier (e.g., 'python', 'java', 'kotlin')
            analyzer_class: Class implementing BaseComplexityAnalyzer
        """
        cls._analyzers[language.lower()] = analyzer_class
        logger.info(f"Registered {analyzer_class.__name__} for {language}")
    
    @classmethod
    def get_analyzer(cls, language: str) -> Optional[BaseComplexityAnalyzer]:
        """Get an analyzer instance for the specified language.
        
        Uses singleton pattern to reuse analyzer instances.
        
        Args:
            language: Language identifier
            
        Returns:
            Analyzer instance or None if not supported
        """
        language = language.lower()
        
        # Check if we have an instance
        if language in cls._instances:
            return cls._instances[language]
        
        # Check if we have a registered class
        if language not in cls._analyzers:
            return None
        
        # Create and cache instance
        analyzer_class = cls._analyzers[language]
        instance = analyzer_class()  # Language is set internally
        cls._instances[language] = instance
        
        return instance
    
    @classmethod
    def get_analyzer_for_file(cls, file_path: Path) -> Optional[BaseComplexityAnalyzer]:
        """Get an analyzer based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Analyzer instance or None if not supported
        """
        extension_map = {
            '.py': 'python',
            '.java': 'java',
            '.kt': 'kotlin',
            '.kts': 'kotlin',
            # Add more mappings as needed
        }
        
        suffix = file_path.suffix.lower()
        language = extension_map.get(suffix)
        
        if not language:
            return None
        
        return cls.get_analyzer(language)
    
    @classmethod
    def supported_languages(cls) -> list[str]:
        """Get list of supported languages.
        
        Returns:
            List of language identifiers
        """
        return list(cls._analyzers.keys())
    
    @classmethod
    def is_supported(cls, language: str) -> bool:
        """Check if a language is supported.
        
        Args:
            language: Language identifier
            
        Returns:
            True if language has a registered analyzer
        """
        return language.lower() in cls._analyzers