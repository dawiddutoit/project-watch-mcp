"""Data models for complexity analysis results.

Provides consistent data structures for representing complexity metrics
across different programming languages.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ComplexityGrade(Enum):
    """Letter grades for code complexity based on maintainability index."""
    
    A = "A"  # Excellent (MI >= 80)
    B = "B"  # Good (MI >= 60)
    C = "C"  # Fair (MI >= 40)
    D = "D"  # Poor (MI >= 20)
    F = "F"  # Very Poor (MI < 20)
    
    @classmethod
    def from_maintainability_index(cls, mi: float) -> "ComplexityGrade":
        """Calculate grade from maintainability index score."""
        if mi >= 80:
            return cls.A
        elif mi >= 60:
            return cls.B
        elif mi >= 40:
            return cls.C
        elif mi >= 20:
            return cls.D
        else:
            return cls.F
    
    @classmethod
    def from_complexity(cls, complexity: int) -> "ComplexityGrade":
        """Calculate grade from cyclomatic complexity."""
        if complexity <= 5:
            return cls.A
        elif complexity <= 10:
            return cls.B
        elif complexity <= 20:
            return cls.C
        elif complexity <= 30:
            return cls.D
        else:
            return cls.F


class ComplexityClassification(Enum):
    """Classification of code complexity levels."""
    
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very-complex"
    
    @classmethod
    def from_complexity(cls, complexity: int) -> "ComplexityClassification":
        """Classify based on cyclomatic complexity score."""
        if complexity <= 5:
            return cls.SIMPLE
        elif complexity <= 10:
            return cls.MODERATE
        elif complexity <= 20:
            return cls.COMPLEX
        else:
            return cls.VERY_COMPLEX


@dataclass
class FunctionComplexity:
    """Complexity metrics for a single function/method."""
    
    name: str
    complexity: int  # Cyclomatic complexity
    cognitive_complexity: int
    rank: str  # Letter grade A-F
    line_start: int
    line_end: int
    classification: str  # simple/moderate/complex/very-complex
    parameters: int
    depth: int  # Maximum nesting depth
    type: str = "function"  # function/method/constructor/lambda/async_function/async_method
    
    # Python-specific attributes (optional, with defaults)
    has_decorators: bool = False
    decorator_count: int = 0
    uses_context_managers: bool = False
    exception_handlers_count: int = 0
    lambda_count: int = 0
    is_generator: bool = False
    uses_walrus_operator: bool = False
    uses_pattern_matching: bool = False
    has_type_annotations: bool = False
    is_recursive: bool = False
    
    @property
    def lines_of_code(self) -> int:
        """Calculate lines of code for this function."""
        return self.line_end - self.line_start + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "complexity": self.complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "rank": self.rank,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "classification": self.classification,
            "parameters": self.parameters,
            "depth": self.depth,
            "type": self.type,
            "lines_of_code": self.lines_of_code,
        }
        
        # Add Python-specific attributes if they have non-default values
        if self.has_decorators:
            result["has_decorators"] = self.has_decorators
            result["decorator_count"] = self.decorator_count
        if self.uses_context_managers:
            result["uses_context_managers"] = self.uses_context_managers
        if self.exception_handlers_count > 0:
            result["exception_handlers_count"] = self.exception_handlers_count
        if self.lambda_count > 0:
            result["lambda_count"] = self.lambda_count
        if self.is_generator:
            result["is_generator"] = self.is_generator
        if self.uses_walrus_operator:
            result["uses_walrus_operator"] = self.uses_walrus_operator
        if self.uses_pattern_matching:
            result["uses_pattern_matching"] = self.uses_pattern_matching
        if self.has_type_annotations:
            result["has_type_annotations"] = self.has_type_annotations
        if self.is_recursive:
            result["is_recursive"] = self.is_recursive
        
        return result


@dataclass
class ClassComplexity:
    """Complexity metrics for a class."""
    
    name: str
    total_complexity: int
    average_method_complexity: float
    method_count: int
    line_start: int
    line_end: int
    methods: List[FunctionComplexity] = field(default_factory=list)
    nested_classes: List["ClassComplexity"] = field(default_factory=list)
    
    @property
    def lines_of_code(self) -> int:
        """Calculate lines of code for this class."""
        return self.line_end - self.line_start + 1
    
    @property
    def max_method_complexity(self) -> int:
        """Get the highest complexity among all methods."""
        if not self.methods:
            return 0
        return max(m.complexity for m in self.methods)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "total_complexity": self.total_complexity,
            "average_method_complexity": self.average_method_complexity,
            "method_count": self.method_count,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "lines_of_code": self.lines_of_code,
            "max_method_complexity": self.max_method_complexity,
            "methods": [m.to_dict() for m in self.methods],
            "nested_classes": [c.to_dict() for c in self.nested_classes],
        }


@dataclass
class ComplexitySummary:
    """Overall complexity summary for a file or module."""
    
    total_complexity: int
    average_complexity: float
    cognitive_complexity: int
    maintainability_index: float
    complexity_grade: str  # Letter grade A-F
    function_count: int
    class_count: int
    lines_of_code: int
    comment_ratio: float = 0.0  # Ratio of comment lines to code lines
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_complexity": self.total_complexity,
            "average_complexity": round(self.average_complexity, 2),
            "cognitive_complexity": self.cognitive_complexity,
            "maintainability_index": round(self.maintainability_index, 2),
            "complexity_grade": self.complexity_grade,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "lines_of_code": self.lines_of_code,
            "comment_ratio": round(self.comment_ratio, 3),
        }


@dataclass
class ComplexityResult:
    """Complete complexity analysis result for a file."""
    
    file_path: Optional[str]
    language: str
    summary: ComplexitySummary
    functions: List[FunctionComplexity] = field(default_factory=list)
    classes: List[ClassComplexity] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if analysis was successful."""
        return self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "file_path": self.file_path,
            "language": self.language,
            "summary": self.summary.to_dict(),
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }
        if self.error:
            result["error"] = self.error
        return result
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on complexity metrics including Python-specific advice."""
        recommendations = []
        
        # Check for very complex functions
        very_complex = [f for f in self.functions if f.complexity > 20]
        if very_complex:
            for func in very_complex[:3]:  # Top 3
                recommendations.append(
                    f"Urgent: Refactor '{func.name}' (complexity: {func.complexity})"
                )
        
        # Check for complex functions
        complex_funcs = [f for f in self.functions if 10 < f.complexity <= 20]
        if complex_funcs and not very_complex:
            for func in complex_funcs[:3]:  # Top 3
                recommendations.append(
                    f"Consider refactoring '{func.name}' (complexity: {func.complexity})"
                )
        
        # Check average complexity
        if self.summary.average_complexity > 10:
            recommendations.append(
                f"High average complexity ({self.summary.average_complexity:.1f}). "
                "Consider breaking down functions"
            )
        
        # Check maintainability
        if self.summary.maintainability_index < 20:
            recommendations.append(
                "Low maintainability index. Code needs significant refactoring"
            )
        elif self.summary.maintainability_index < 40:
            recommendations.append(
                "Moderate maintainability index. Consider improving code structure"
            )
        
        # Check for deep nesting
        deep_nested = [f for f in self.functions if f.depth > 4]
        if deep_nested:
            recommendations.append(
                f"{len(deep_nested)} function(s) have deep nesting (>4 levels). "
                "Consider extracting nested logic"
            )
        
        # Python-specific recommendations
        if self.language == "python":
            # Check for excessive decorators
            heavily_decorated = [f for f in self.functions if hasattr(f, 'decorator_count') and f.decorator_count > 3]
            if heavily_decorated:
                recommendations.append(
                    f"{len(heavily_decorated)} function(s) have excessive decorators (>3). "
                    "Consider simplifying decorator usage"
                )
            
            # Check for complex async functions
            complex_async = [f for f in self.functions 
                           if hasattr(f, 'type') and 'async' in f.type and f.complexity > 10]
            if complex_async:
                recommendations.append(
                    f"{len(complex_async)} async function(s) have high complexity. "
                    "Consider breaking down async operations"
                )
            
            # Check for excessive exception handling
            many_exceptions = [f for f in self.functions 
                             if hasattr(f, 'exception_handlers_count') and f.exception_handlers_count > 3]
            if many_exceptions:
                recommendations.append(
                    "Multiple exception handlers detected. Consider consolidating error handling"
                )
            
            # Check for complex comprehensions (high lambda count)
            complex_comprehensions = [f for f in self.functions 
                                    if hasattr(f, 'lambda_count') and f.lambda_count > 2]
            if complex_comprehensions:
                recommendations.append(
                    "Complex comprehensions or multiple lambdas detected. "
                    "Consider extracting to named functions for clarity"
                )
            
            # Check for recursive functions with high complexity
            complex_recursive = [f for f in self.functions 
                               if hasattr(f, 'is_recursive') and f.is_recursive and f.complexity > 10]
            if complex_recursive:
                recommendations.append(
                    "Complex recursive function(s) detected. Consider iterative approach or memoization"
                )
            
            # Check cognitive vs cyclomatic complexity disparity
            high_cognitive_disparity = [f for f in self.functions 
                                       if f.cognitive_complexity > f.complexity * 1.5 and f.complexity > 5]
            if high_cognitive_disparity:
                recommendations.append(
                    f"{len(high_cognitive_disparity)} function(s) have high cognitive complexity relative to cyclomatic. "
                    "Consider reducing nested logic"
                )
        
        # Positive feedback if code is good
        if not recommendations:
            if self.summary.complexity_grade in ["A", "B"]:
                recommendations.append("Code complexity is within acceptable limits")
        
        return recommendations