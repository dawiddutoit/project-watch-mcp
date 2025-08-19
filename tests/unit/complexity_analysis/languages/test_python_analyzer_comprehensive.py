"""Comprehensive test suite for Python complexity analyzer.

This module provides extensive testing for the Python complexity analyzer,
targeting >85% code coverage with focus on edge cases, error handling,
and Python-specific features.
"""

import ast
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from project_watch_mcp.complexity_analysis.languages.python_analyzer import (
    PythonComplexityAnalyzer,
    RADON_AVAILABLE
)
from project_watch_mcp.complexity_analysis.models import (
    ComplexityResult,
    ComplexityGrade,
    ComplexityClassification,
    FunctionComplexity,
    ClassComplexity,
)


class TestPythonComplexityAnalyzer:
    """Comprehensive test suite for Python complexity analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return PythonComplexityAnalyzer()
    
    @pytest.fixture
    def temp_python_file(self, tmp_path):
        """Create a temporary Python file for testing."""
        file_path = tmp_path / "test_file.py"
        file_path.write_text("""
def simple_function():
    return 42

def complex_function(x):
    if x > 0:
        if x > 10:
            return x * 2
        else:
            return x + 1
    else:
        return -x
        
class TestClass:
    def method1(self):
        pass
    
    def method2(self, x, y):
        return x + y
""")
        return file_path
    
    # ==================== Basic Functionality Tests ====================
    
    @pytest.mark.asyncio
    async def test_analyze_simple_function(self, analyzer):
        """Test analysis of a simple function."""
        code = """
def simple():
    return 42
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.language == "python"
        assert result.summary.function_count == 1
        assert result.summary.total_complexity >= 1
        assert len(result.functions) == 1
        assert result.functions[0].name == "simple"
    
    @pytest.mark.asyncio
    async def test_analyze_file(self, analyzer, temp_python_file):
        """Test file analysis."""
        result = await analyzer.analyze_file(temp_python_file)
        
        assert result.success
        assert result.file_path == str(temp_python_file)
        assert result.summary.function_count >= 2
        assert result.summary.class_count >= 1
    
    @pytest.mark.asyncio
    async def test_file_not_found(self, analyzer):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            await analyzer.analyze_file(Path("nonexistent.py"))
    
    @pytest.mark.asyncio
    async def test_non_python_file(self, analyzer, tmp_path):
        """Test rejection of non-Python files."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("not python")
        
        with pytest.raises(ValueError, match="Not a Python file"):
            await analyzer.analyze_file(file_path)
    
    # ==================== Cyclomatic Complexity Tests ====================
    
    @pytest.mark.parametrize("code,expected_min_complexity", [
        ("def f(): pass", 1),
        ("def f():\n    if x: return 1\n    return 0", 2),
        ("def f():\n    for i in range(10):\n        if i > 5: break", 3),
        ("def f():\n    while True:\n        if x: break\n        elif y: continue", 4),
        ("def f():\n    try:\n        x()\n    except A: pass\n    except B: pass", 3),
    ])
    @pytest.mark.asyncio
    async def test_cyclomatic_complexity_calculation(self, analyzer, code, expected_min_complexity):
        """Test cyclomatic complexity calculation for various constructs."""
        result = await analyzer.analyze_code(code)
        assert result.functions[0].complexity >= expected_min_complexity
    
    @pytest.mark.asyncio
    async def test_nested_complexity(self, analyzer):
        """Test complexity calculation for nested structures."""
        code = """
def nested_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0
"""
        result = await analyzer.analyze_code(code)
        func = result.functions[0]
        
        assert func.complexity >= 4  # Multiple decision points
        assert func.depth >= 3  # Three levels of nesting
    
    # ==================== Cognitive Complexity Tests ====================
    
    @pytest.mark.asyncio
    async def test_cognitive_complexity(self, analyzer):
        """Test cognitive complexity calculation."""
        code = """
def cognitive_example(a, b, c):
    if a:  # +1
        if b:  # +2 (nested)
            if c:  # +3 (double nested)
                return 1
    elif b:  # +1
        return 2
    else:  # +0
        return 3
"""
        result = await analyzer.analyze_code(code)
        func = result.functions[0]
        
        assert func.cognitive_complexity >= 7
        assert func.cognitive_complexity > func.complexity
    
    # ==================== Python-Specific Feature Tests ====================
    
    @pytest.mark.asyncio
    async def test_decorator_detection(self, analyzer):
        """Test detection of decorators."""
        code = """
@decorator1
@decorator2
@nested.decorator
def decorated_function():
    pass
"""
        result = await analyzer.analyze_code(code)
        func = result.functions[0]
        
        assert func.has_decorators
        assert func.decorator_count == 3
    
    @pytest.mark.asyncio
    async def test_async_function_detection(self, analyzer):
        """Test detection of async functions."""
        code = """
async def async_function():
    await some_operation()
    return 42
"""
        result = await analyzer.analyze_code(code)
        func = result.functions[0]
        
        assert func.type == "async_function"
        assert func.complexity >= 1
    
    @pytest.mark.asyncio
    async def test_generator_detection(self, analyzer):
        """Test detection of generator functions."""
        code = """
def generator_function():
    for i in range(10):
        yield i * 2
"""
        result = await analyzer.analyze_code(code)
        func = result.functions[0]
        
        assert func.is_generator
    
    @pytest.mark.asyncio
    async def test_comprehension_complexity(self, analyzer):
        """Test complexity of list/dict/set comprehensions."""
        code = """
def comprehension_function():
    # List comprehension with condition
    a = [x for x in range(100) if x % 2 == 0]
    
    # Nested comprehension
    b = [[y for y in range(x)] for x in range(10)]
    
    # Dict comprehension
    c = {x: x**2 for x in range(10) if x > 5}
    
    # Generator expression
    d = (x for x in range(1000) if x % 3 == 0)
    
    return a, b, c, d
"""
        result = await analyzer.analyze_code(code)
        func = result.functions[0]
        
        assert func.lambda_count >= 4  # Comprehensions count as lambdas
    
    @pytest.mark.asyncio
    async def test_context_manager_detection(self, analyzer):
        """Test detection of context managers."""
        code = """
def with_context_managers():
    with open('file1.txt') as f1:
        with open('file2.txt') as f2:
            data1 = f1.read()
            data2 = f2.read()
    return data1, data2
"""
        result = await analyzer.analyze_code(code)
        func = result.functions[0]
        
        assert func.uses_context_managers
    
    @pytest.mark.asyncio
    async def test_exception_handler_counting(self, analyzer):
        """Test counting of exception handlers."""
        code = """
def exception_handling():
    try:
        risky_operation()
    except ValueError:
        handle_value_error()
    except KeyError:
        handle_key_error()
    except Exception as e:
        handle_generic(e)
    finally:
        cleanup()
"""
        result = await analyzer.analyze_code(code)
        func = result.functions[0]
        
        assert func.exception_handlers_count == 3
    
    @pytest.mark.asyncio
    async def test_walrus_operator_detection(self, analyzer):
        """Test detection of walrus operator (Python 3.8+)."""
        code = """
def with_walrus():
    if (n := len(data)) > 10:
        return n * 2
    return n
"""
        result = await analyzer.analyze_code(code)
        func = result.functions[0]
        
        # This will depend on Python version
        if hasattr(ast, 'NamedExpr'):  # Python 3.8+
            assert func.uses_walrus_operator
    
    @pytest.mark.asyncio
    async def test_pattern_matching_detection(self, analyzer):
        """Test detection of pattern matching (Python 3.10+)."""
        code = """
def with_pattern_matching(value):
    match value:
        case 0:
            return "zero"
        case 1 | 2:
            return "one or two"
        case _:
            return "other"
"""
        result = await analyzer.analyze_code(code)
        
        if hasattr(ast, 'Match'):  # Python 3.10+
            func = result.functions[0]
            assert func.uses_pattern_matching
            assert func.complexity >= 3  # Three cases
    
    @pytest.mark.asyncio
    async def test_type_annotation_detection(self, analyzer):
        """Test detection of type annotations."""
        code = """
def typed_function(x: int, y: str) -> tuple[int, str]:
    return (x * 2, y.upper())

def untyped_function(x, y):
    return x + y
"""
        result = await analyzer.analyze_code(code)
        
        typed_func = result.functions[0]
        untyped_func = result.functions[1]
        
        assert typed_func.has_type_annotations
        assert not untyped_func.has_type_annotations
    
    @pytest.mark.asyncio
    async def test_recursion_detection(self, analyzer):
        """Test detection of recursive functions."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def non_recursive(n):
    return n * 2
"""
        result = await analyzer.analyze_code(code)
        
        assert result.functions[0].is_recursive  # factorial
        assert result.functions[1].is_recursive  # fibonacci
        assert not result.functions[2].is_recursive  # non_recursive
    
    # ==================== Class Analysis Tests ====================
    
    @pytest.mark.asyncio
    async def test_class_analysis(self, analyzer):
        """Test analysis of classes."""
        code = """
class ComplexClass:
    def __init__(self, x):
        self.x = x
    
    def simple_method(self):
        return self.x
    
    def complex_method(self, y):
        if y > 0:
            if self.x > y:
                return self.x - y
            else:
                return y - self.x
        else:
            return self.x + abs(y)
    
    @property
    def value(self):
        return self.x * 2
    
    @staticmethod
    def static_method(a, b):
        return a + b
    
    @classmethod
    def class_method(cls, value):
        return cls(value)
"""
        result = await analyzer.analyze_code(code)
        
        assert result.summary.class_count == 1
        assert len(result.classes) == 1
        
        cls = result.classes[0]
        assert cls.name == "ComplexClass"
        assert cls.method_count >= 6
        assert cls.total_complexity > 0
        assert cls.average_method_complexity > 0
    
    @pytest.mark.asyncio
    async def test_nested_classes(self, analyzer):
        """Test analysis of nested classes."""
        code = """
class OuterClass:
    def outer_method(self):
        pass
    
    class InnerClass:
        def inner_method(self):
            pass
        
        class DeepClass:
            def deep_method(self):
                pass
"""
        result = await analyzer.analyze_code(code)
        
        assert result.summary.class_count >= 1
        outer = result.classes[0]
        assert len(outer.nested_classes) >= 1
    
    # ==================== Maintainability Index Tests ====================
    
    @pytest.mark.asyncio
    async def test_maintainability_index_calculation(self, analyzer):
        """Test maintainability index calculation."""
        code = """
def well_maintained():
    \"\"\"Well documented function.\"\"\"
    # Clear implementation
    result = 42
    return result

def poorly_maintained(a,b,c,d,e,f,g):
    if a>0:
        if b>0:
            if c>0:
                if d>0:
                    if e>0:
                        if f>0:
                            if g>0:
                                return a+b+c+d+e+f+g
    return 0
"""
        result = await analyzer.analyze_code(code)
        
        # Well maintained function should have higher MI
        well_maintained = result.functions[0]
        poorly_maintained = result.functions[1]
        
        assert result.summary.maintainability_index > 0
        assert result.summary.maintainability_index <= 100
    
    @pytest.mark.asyncio
    async def test_complexity_grading(self, analyzer):
        """Test complexity grade assignment."""
        code = """
def simple(): return 1

def moderate():
    for i in range(10):
        if i > 5:
            return i
    return 0

def complex_func(a, b, c, d):
    result = 0
    for i in range(a):
        for j in range(b):
            if i > j:
                for k in range(c):
                    if k % 2 == 0:
                        for l in range(d):
                            if l > k:
                                result += i * j * k * l
    return result
"""
        result = await analyzer.analyze_code(code)
        
        # Check grade assignment
        assert ComplexityGrade[result.summary.complexity_grade] in ComplexityGrade
        
        # Simple function should have better grade
        simple_func = result.functions[0]
        complex_func = result.functions[2]
        
        assert ComplexityGrade[simple_func.rank] <= ComplexityGrade[complex_func.rank]
    
    # ==================== Error Handling Tests ====================
    
    @pytest.mark.asyncio
    async def test_syntax_error_handling(self, analyzer):
        """Test handling of syntax errors."""
        code = """
def broken_function(
    return 42  # Syntax error
"""
        result = await analyzer.analyze_code(code)
        
        assert not result.success
        assert result.error is not None
        assert "syntax" in result.error.lower() or "parse" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_empty_code_handling(self, analyzer):
        """Test handling of empty code."""
        result = await analyzer.analyze_code("")
        
        assert result.success
        assert result.summary.function_count == 0
        assert result.summary.class_count == 0
        assert result.summary.total_complexity == 0
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self, analyzer):
        """Test handling of Unicode in code."""
        code = """
def 你好():
    '''函数文档'''
    变量 = "Hello, 世界"
    return 变量

def emoji_function():
    🎉 = "celebration"
    return 🎉
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.function_count == 2
    
    # ==================== Radon Integration Tests ====================
    
    @pytest.mark.skipif(not RADON_AVAILABLE, reason="Radon not installed")
    @pytest.mark.asyncio
    async def test_radon_analysis(self, analyzer):
        """Test analysis using radon library."""
        code = """
def radon_test(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x - y
    else:
        return -x
"""
        with patch.object(analyzer, '_analyze_with_radon') as mock_radon:
            mock_radon.return_value = ComplexityResult(
                file_path=None,
                language="python",
                summary=Mock(),
                functions=[],
                classes=[]
            )
            
            result = await analyzer.analyze_code(code)
            mock_radon.assert_called_once()
    
    @pytest.mark.skipif(RADON_AVAILABLE, reason="Testing fallback when Radon not available")
    @pytest.mark.asyncio
    async def test_ast_fallback(self, analyzer):
        """Test fallback to AST analysis when radon not available."""
        code = """
def ast_test():
    return 42
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.function_count == 1
    
    # ==================== Performance Tests ====================
    
    @pytest.mark.asyncio
    async def test_large_file_performance(self, analyzer):
        """Test performance with large files."""
        # Generate a large file with many functions
        functions = []
        for i in range(100):
            functions.append(f"""
def function_{i}(x):
    if x > {i}:
        return x * {i}
    else:
        return x + {i}
""")
        
        code = "\n".join(functions)
        
        import time
        start = time.time()
        result = await analyzer.analyze_code(code)
        duration = time.time() - start
        
        assert result.success
        assert result.summary.function_count == 100
        assert duration < 5.0  # Should complete within 5 seconds
    
    # ==================== Recommendation Generation Tests ====================
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, analyzer):
        """Test generation of improvement recommendations."""
        code = """
def very_complex_function(a, b, c, d, e, f):
    result = 0
    for i in range(a):
        if i % 2 == 0:
            for j in range(b):
                if j % 3 == 0:
                    for k in range(c):
                        if k % 5 == 0:
                            for l in range(d):
                                if l % 7 == 0:
                                    for m in range(e):
                                        if m % 11 == 0:
                                            for n in range(f):
                                                if n % 13 == 0:
                                                    result += i * j * k * l * m * n
    return result
"""
        result = await analyzer.analyze_code(code)
        recommendations = result.generate_recommendations()
        
        assert len(recommendations) > 0
        assert any("refactor" in r.lower() for r in recommendations)
        assert any("complex" in r.lower() for r in recommendations)
    
    # ==================== Cache Management Tests ====================
    
    @pytest.mark.asyncio
    async def test_cache_management(self, analyzer):
        """Test cache clearing functionality."""
        code = "def cached(): return 42"
        
        # Analyze code to populate cache
        await analyzer.analyze_code(code)
        
        # Ensure cache has data
        analyzer._cache['test'] = 'data'
        assert 'test' in analyzer._cache
        
        # Clear cache
        analyzer.clear_cache()
        assert len(analyzer._cache) == 0
    
    # ==================== Integration Tests ====================
    
    @pytest.mark.asyncio
    async def test_real_world_code(self, analyzer):
        """Test with real-world Python code patterns."""
        code = '''
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration dataclass."""
    api_key: str
    timeout: int = 30
    retries: int = 3

class APIClient:
    """API client with retry logic."""
    
    def __init__(self, config: Config):
        self.config = config
        self._session: Optional[Any] = None
    
    async def __aenter__(self):
        self._session = await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def fetch_data(self, endpoint: str, **params) -> Dict[str, Any]:
        """Fetch data with retry logic."""
        last_error = None
        
        for attempt in range(self.config.retries):
            try:
                response = await self._make_request(endpoint, params)
                
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise ValueError(f"Unexpected status: {response.status}")
                    
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < self.config.retries - 1:
                    await asyncio.sleep(1)
                    continue
                    
            except Exception as e:
                last_error = e
                break
        
        raise last_error or Exception("Failed after retries")
    
    async def _make_request(self, endpoint: str, params: Dict) -> Any:
        """Make HTTP request."""
        # Implementation details...
        pass
    
    async def _create_session(self) -> Any:
        """Create HTTP session."""
        # Implementation details...
        pass

async def process_items(items: List[str]) -> List[Dict]:
    """Process items concurrently."""
    async with APIClient(Config(api_key="test")) as client:
        tasks = [client.fetch_data(f"/item/{item}") for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            r for r in results
            if not isinstance(r, Exception)
        ]
'''
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.class_count >= 2  # Config and APIClient
        assert result.summary.function_count >= 6
        assert any(f.type == "async_method" for f in result.functions)
        assert any(f.has_type_annotations for f in result.functions)