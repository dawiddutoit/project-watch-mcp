"""Tests for the unified complexity analysis system base architecture."""

import pytest
from pathlib import Path
import tempfile

from project_watch_mcp.complexity_analysis import (
    BaseComplexityAnalyzer,
    AnalyzerRegistry,
    ComplexityResult,
    ComplexitySummary,
    FunctionComplexity,
    ComplexityGrade,
)
from project_watch_mcp.complexity_analysis.languages.python_analyzer import PythonComplexityAnalyzer


@pytest.mark.asyncio
async def test_python_analyzer_registration():
    """Test that Python analyzer is properly registered."""
    analyzer = AnalyzerRegistry.get_analyzer("python")
    assert analyzer is not None
    assert isinstance(analyzer, PythonComplexityAnalyzer)
    assert analyzer.language == "python"


@pytest.mark.asyncio
async def test_analyzer_singleton():
    """Test that analyzer registry uses singleton pattern."""
    analyzer1 = AnalyzerRegistry.get_analyzer("python")
    analyzer2 = AnalyzerRegistry.get_analyzer("python")
    assert analyzer1 is analyzer2


@pytest.mark.asyncio
async def test_supported_languages():
    """Test listing of supported languages."""
    languages = AnalyzerRegistry.supported_languages()
    assert "python" in languages


@pytest.mark.asyncio
async def test_analyze_simple_python_code():
    """Test analyzing simple Python code."""
    code = """
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
"""
    
    analyzer = AnalyzerRegistry.get_analyzer("python")
    result = await analyzer.analyze_code(code)
    
    assert result.success
    assert result.language == "python"
    assert result.summary.function_count == 2
    assert result.summary.total_complexity == 2  # Each simple function has complexity 1
    assert len(result.functions) == 2


@pytest.mark.asyncio
async def test_analyze_complex_python_code():
    """Test analyzing complex Python code."""
    code = """
def complex_function(data, threshold=10):
    result = []
    for item in data:
        if item > threshold:
            if item % 2 == 0:
                result.append(item * 2)
            else:
                result.append(item * 3)
        elif item < 0:
            result.append(abs(item))
        else:
            result.append(item)
    return result
"""
    
    analyzer = AnalyzerRegistry.get_analyzer("python")
    result = await analyzer.analyze_code(code)
    
    assert result.success
    assert result.summary.function_count == 1
    assert result.summary.total_complexity > 1  # Should have higher complexity
    
    # Check function details
    func = result.functions[0]
    assert func.name == "complex_function"
    assert func.complexity > 3  # Multiple decision points
    assert func.classification in ["moderate", "complex"]


@pytest.mark.asyncio
async def test_analyze_file():
    """Test analyzing a Python file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("""
def test_function():
    x = 1
    if x > 0:
        return x
    return 0
""")
        temp_path = Path(f.name)
    
    try:
        analyzer = AnalyzerRegistry.get_analyzer("python")
        result = await analyzer.analyze_file(temp_path)
        
        assert result.success
        assert result.file_path == str(temp_path)
        assert result.summary.function_count == 1
    finally:
        temp_path.unlink()


@pytest.mark.asyncio
async def test_complexity_grades():
    """Test complexity grade calculations."""
    # Test grade from maintainability index
    assert ComplexityGrade.from_maintainability_index(85) == ComplexityGrade.A
    assert ComplexityGrade.from_maintainability_index(65) == ComplexityGrade.B
    assert ComplexityGrade.from_maintainability_index(45) == ComplexityGrade.C
    assert ComplexityGrade.from_maintainability_index(25) == ComplexityGrade.D
    assert ComplexityGrade.from_maintainability_index(15) == ComplexityGrade.F
    
    # Test grade from complexity
    assert ComplexityGrade.from_complexity(3) == ComplexityGrade.A
    assert ComplexityGrade.from_complexity(8) == ComplexityGrade.B
    assert ComplexityGrade.from_complexity(15) == ComplexityGrade.C
    assert ComplexityGrade.from_complexity(25) == ComplexityGrade.D
    assert ComplexityGrade.from_complexity(35) == ComplexityGrade.F


@pytest.mark.asyncio
async def test_recommendations_generation():
    """Test that recommendations are properly generated."""
    code = """
def very_complex_function(a, b, c, d, e):
    # Intentionally complex function for testing
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return a + b + c + d + e
                    else:
                        return a + b + c + d
                else:
                    if e > 0:
                        return a + b + c + e
                    else:
                        return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        if b > 0:
            if c > 0:
                return b + c
            else:
                return b
        else:
            return 0
"""
    
    analyzer = AnalyzerRegistry.get_analyzer("python")
    result = await analyzer.analyze_code(code)
    
    assert result.success
    assert len(result.recommendations) > 0
    # Should recommend refactoring for high complexity
    assert any("refactor" in rec.lower() for rec in result.recommendations)


@pytest.mark.asyncio 
async def test_cognitive_complexity():
    """Test cognitive complexity calculation."""
    code = """
def nested_function(x):
    if x > 0:  # +1 complexity, +1 cognitive
        for i in range(x):  # +1 complexity, +2 cognitive (nested)
            if i % 2 == 0:  # +1 complexity, +3 cognitive (double nested)
                print(i)
"""
    
    analyzer = AnalyzerRegistry.get_analyzer("python")
    result = await analyzer.analyze_code(code)
    
    assert result.success
    func = result.functions[0]
    assert func.cognitive_complexity > func.complexity  # Cognitive should be higher due to nesting


@pytest.mark.asyncio
async def test_class_analysis():
    """Test analyzing classes and methods."""
    code = """
class TestClass:
    def __init__(self):
        self.value = 0
    
    def simple_method(self):
        return self.value
    
    def complex_method(self, x):
        if x > 0:
            if x % 2 == 0:
                return x * 2
            else:
                return x * 3
        return 0
"""
    
    analyzer = AnalyzerRegistry.get_analyzer("python")
    result = await analyzer.analyze_code(code)
    
    assert result.success
    assert result.summary.class_count == 1
    assert result.summary.function_count == 3  # __init__ + 2 methods
    
    # Check that we have class information
    if result.classes:  # Only if radon is available
        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "TestClass"
        assert cls.method_count == 3


@pytest.mark.asyncio
async def test_error_handling():
    """Test handling of syntax errors."""
    code = """
def broken_function(:
    this is not valid python
"""
    
    analyzer = AnalyzerRegistry.get_analyzer("python")
    result = await analyzer.analyze_code(code)
    
    # Should return a result with error
    assert not result.success
    assert result.error is not None
    assert "syntax" in result.error.lower()


@pytest.mark.asyncio
async def test_file_not_found():
    """Test handling of non-existent files."""
    analyzer = AnalyzerRegistry.get_analyzer("python")
    
    with pytest.raises(FileNotFoundError):
        await analyzer.analyze_file(Path("/nonexistent/file.py"))


@pytest.mark.asyncio
async def test_non_python_file():
    """Test handling of non-Python files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is not a Python file")
        temp_path = Path(f.name)
    
    try:
        analyzer = AnalyzerRegistry.get_analyzer("python")
        with pytest.raises(ValueError, match="Not a Python file"):
            await analyzer.analyze_file(temp_path)
    finally:
        temp_path.unlink()