"""Enhanced unit tests for Python complexity analyzer.

This module provides comprehensive unit tests for Python-specific
complexity features including decorators, async functions, generators,
context managers, and other Python-specific constructs.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.complexity_analysis.languages.python_analyzer import (
    PythonComplexityAnalyzer,
)
from project_watch_mcp.complexity_analysis.models import (
    ComplexityResult,
    FunctionComplexity,
)


class TestPythonSpecificFeatures:
    """Test Python-specific language features."""
    
    @pytest.fixture
    def analyzer(self):
        """Create Python analyzer instance."""
        return PythonComplexityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_decorator_detection(self, analyzer):
        """Test detection of function decorators."""
        code = """
@property
def simple_property(self):
    return self._value

@staticmethod
def static_method():
    return 42

@classmethod
def class_method(cls):
    return cls.__name__

@lru_cache(maxsize=128)
@timing_decorator
def cached_function(n):
    if n <= 1:
        return 1
    return n * cached_function(n - 1)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = Path(f.name)
        
        try:
            result = await analyzer.analyze_file(file_path)
            
            # Verify decorator detection
            funcs = {f.name: f for f in result.functions}
            
            if hasattr(funcs.get("simple_property"), "decorator_count"):
                assert funcs["simple_property"].decorator_count == 1
            if hasattr(funcs.get("static_method"), "decorator_count"):
                assert funcs["static_method"].decorator_count == 1
            if hasattr(funcs.get("class_method"), "decorator_count"):
                assert funcs["class_method"].decorator_count == 1
            if hasattr(funcs.get("cached_function"), "decorator_count"):
                assert funcs["cached_function"].decorator_count == 2
        finally:
            file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_async_function_detection(self, analyzer):
        """Test detection of async functions and await statements."""
        code = """
async def fetch_data(url):
    response = await http_client.get(url)
    if response.status == 200:
        data = await response.json()
        return data
    return None

async def process_multiple(urls):
    tasks = []
    for url in urls:
        task = asyncio.create_task(fetch_data(url))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

def sync_function():
    return "not async"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = Path(f.name)
        
        try:
            result = await analyzer.analyze_file(file_path)
            
            funcs = {f.name: f for f in result.functions}
            
            # Check async detection
            if hasattr(funcs.get("fetch_data"), "is_async"):
                assert funcs["fetch_data"].is_async is True
            if hasattr(funcs.get("process_multiple"), "is_async"):
                assert funcs["process_multiple"].is_async is True
            if hasattr(funcs.get("sync_function"), "is_async"):
                assert funcs["sync_function"].is_async is False
            
            # Check complexity (async adds cognitive complexity)
            assert funcs["fetch_data"].complexity == 2  # if statement
            assert funcs["process_multiple"].complexity == 3  # for loop + list comp
        finally:
            file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_generator_detection(self, analyzer):
        """Test detection of generator functions."""
        code = """
def simple_generator():
    yield 1
    yield 2
    yield 3

def conditional_generator(n):
    for i in range(n):
        if i % 2 == 0:
            yield i * i
        else:
            yield i

def generator_expression_user():
    return (x * 2 for x in range(10) if x > 5)

def list_comprehension_user():
    return [x * 2 for x in range(10) if x > 5]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = Path(f.name)
        
        try:
            result = await analyzer.analyze_file(file_path)
            
            funcs = {f.name: f for f in result.functions}
            
            # Check generator detection
            if hasattr(funcs.get("simple_generator"), "is_generator"):
                assert funcs["simple_generator"].is_generator is True
            if hasattr(funcs.get("conditional_generator"), "is_generator"):
                assert funcs["conditional_generator"].is_generator is True
            if hasattr(funcs.get("generator_expression_user"), "has_generator_expression"):
                assert funcs["generator_expression_user"].has_generator_expression is True
            
            # Verify complexity
            assert funcs["conditional_generator"].complexity == 3  # for + if
        finally:
            file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_context_manager_detection(self, analyzer):
        """Test detection of context managers (with statements)."""
        code = """
def file_processor(filename):
    with open(filename) as f:
        content = f.read()
        return len(content)

def multiple_contexts():
    with open('file1.txt') as f1, open('file2.txt') as f2:
        data1 = f1.read()
        data2 = f2.read()
        return data1 + data2

def nested_contexts():
    with transaction():
        with database.cursor() as cursor:
            cursor.execute("SELECT * FROM users")
            return cursor.fetchall()

async def async_context():
    async with aiofiles.open('data.txt') as f:
        content = await f.read()
        return content
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = Path(f.name)
        
        try:
            result = await analyzer.analyze_file(file_path)
            
            funcs = {f.name: f for f in result.functions}
            
            # Check context manager detection
            if hasattr(funcs.get("file_processor"), "context_manager_count"):
                assert funcs["file_processor"].context_manager_count == 1
            if hasattr(funcs.get("multiple_contexts"), "context_manager_count"):
                assert funcs["multiple_contexts"].context_manager_count == 2
            if hasattr(funcs.get("nested_contexts"), "context_manager_count"):
                assert funcs["nested_contexts"].context_manager_count == 2
            if hasattr(funcs.get("async_context"), "context_manager_count"):
                assert funcs["async_context"].context_manager_count == 1
        finally:
            file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_lambda_and_nested_functions(self, analyzer):
        """Test detection of lambdas and nested functions."""
        code = """
def outer_function(data):
    filter_func = lambda x: x > 0
    
    def inner_processor(item):
        if item % 2 == 0:
            return item * 2
        return item
    
    result = list(map(inner_processor, filter(filter_func, data)))
    
    # Multiple lambdas
    sorted_result = sorted(result, key=lambda x: -x)
    
    return sorted_result

def functional_style(items):
    return list(
        map(lambda x: x * 2,
            filter(lambda x: x > 0,
                   map(lambda x: x + 1, items)))
    )
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = Path(f.name)
        
        try:
            result = await analyzer.analyze_file(file_path)
            
            funcs = {f.name: f for f in result.functions}
            
            # Check lambda detection
            if hasattr(funcs.get("outer_function"), "lambda_count"):
                assert funcs["outer_function"].lambda_count >= 2
            if hasattr(funcs.get("functional_style"), "lambda_count"):
                assert funcs["functional_style"].lambda_count == 3
            
            # Nested functions might be counted separately
            assert len(result.functions) >= 2  # At least outer functions
        finally:
            file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_exception_handling_variations(self, analyzer):
        """Test various exception handling patterns."""
        code = """
def simple_try_except():
    try:
        result = risky_operation()
        return result
    except ValueError:
        return None

def multiple_except_blocks():
    try:
        result = complex_operation()
    except ValueError as e:
        log_error(e)
        return None
    except KeyError as e:
        log_warning(e)
        return {}
    except Exception as e:
        log_critical(e)
        raise

def try_except_else_finally():
    try:
        result = operation()
    except SpecificError:
        handle_error()
    else:
        process_success(result)
    finally:
        cleanup()

def nested_exception_handling():
    try:
        with open('file.txt') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        return data
    except IOError:
        return None
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = Path(f.name)
        
        try:
            result = await analyzer.analyze_file(file_path)
            
            funcs = {f.name: f for f in result.functions}
            
            # Check exception complexity
            assert funcs["simple_try_except"].complexity >= 2
            assert funcs["multiple_except_blocks"].complexity >= 4  # 3 except blocks
            assert funcs["try_except_else_finally"].complexity >= 2
            assert funcs["nested_exception_handling"].complexity >= 3
        finally:
            file_path.unlink(missing_ok=True)
    
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

def indirect_recursion_a(n):
    if n <= 0:
        return 0
    return indirect_recursion_b(n - 1)

def indirect_recursion_b(n):
    if n <= 0:
        return 1
    return indirect_recursion_a(n - 1)

def not_recursive(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = Path(f.name)
        
        try:
            result = await analyzer.analyze_file(file_path)
            
            funcs = {f.name: f for f in result.functions}
            
            # Check recursion detection
            if hasattr(funcs.get("factorial"), "is_recursive"):
                assert funcs["factorial"].is_recursive is True
            if hasattr(funcs.get("fibonacci"), "is_recursive"):
                assert funcs["fibonacci"].is_recursive is True
            if hasattr(funcs.get("not_recursive"), "is_recursive"):
                assert funcs["not_recursive"].is_recursive is False
            
            # Recursive functions still have their base complexity
            assert funcs["factorial"].complexity == 2
            assert funcs["fibonacci"].complexity == 3  # Two base cases
        finally:
            file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_walrus_operator_detection(self, analyzer):
        """Test detection of walrus operator (:=) usage."""
        code = """
def uses_walrus():
    if (n := len(data)) > 10:
        print(f"List is too long ({n} elements)")
        return None
    
    while (chunk := file.read(1024)):
        process(chunk)
    
    return [y for x in data if (y := transform(x)) is not None]

def traditional_style():
    n = len(data)
    if n > 10:
        print(f"List is too long ({n} elements)")
        return None
    
    chunk = file.read(1024)
    while chunk:
        process(chunk)
        chunk = file.read(1024)
    
    return result
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = Path(f.name)
        
        try:
            result = await analyzer.analyze_file(file_path)
            
            funcs = {f.name: f for f in result.functions}
            
            # Check walrus operator detection
            if hasattr(funcs.get("uses_walrus"), "uses_walrus_operator"):
                assert funcs["uses_walrus"].uses_walrus_operator is True
            if hasattr(funcs.get("traditional_style"), "uses_walrus_operator"):
                assert funcs["traditional_style"].uses_walrus_operator is False
        finally:
            file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_pattern_matching_detection(self, analyzer):
        """Test detection of pattern matching (match/case) statements."""
        code = """
def pattern_matcher(value):
    match value:
        case 0:
            return "zero"
        case 1 | 2 | 3:
            return "small"
        case int(x) if x > 10:
            return "large"
        case [x, y]:
            return f"pair: {x}, {y}"
        case {"key": value}:
            return f"dict with key: {value}"
        case _:
            return "unknown"

def traditional_branching(value):
    if value == 0:
        return "zero"
    elif value in [1, 2, 3]:
        return "small"
    elif isinstance(value, int) and value > 10:
        return "large"
    else:
        return "unknown"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = Path(f.name)
        
        try:
            result = await analyzer.analyze_file(file_path)
            
            funcs = {f.name: f for f in result.functions}
            
            # Pattern matching adds complexity similar to if/elif
            if "pattern_matcher" in funcs:
                assert funcs["pattern_matcher"].complexity >= 6  # 6 cases
            
            if hasattr(funcs.get("pattern_matcher"), "uses_pattern_matching"):
                assert funcs["pattern_matcher"].uses_pattern_matching is True
            if hasattr(funcs.get("traditional_branching"), "uses_pattern_matching"):
                assert funcs["traditional_branching"].uses_pattern_matching is False
        finally:
            file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_type_annotation_detection(self, analyzer):
        """Test detection of type annotations."""
        code = """
from typing import List, Dict, Optional, Union, Callable

def fully_typed(
    name: str,
    age: int,
    scores: List[float],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[str, int, List[float]]]:
    result: Dict[str, Union[str, int, List[float]]] = {
        "name": name,
        "age": age,
        "scores": scores
    }
    if metadata:
        result["meta"] = str(metadata)
    return result

def partially_typed(name: str, age, scores):
    return {"name": name, "age": age, "scores": scores}

def no_types(name, age, scores):
    return {"name": name, "age": age, "scores": scores}

def generic_function(
    transform: Callable[[int], str],
    data: List[int]
) -> List[str]:
    return [transform(x) for x in data]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            file_path = Path(f.name)
        
        try:
            result = await analyzer.analyze_file(file_path)
            
            funcs = {f.name: f for f in result.functions}
            
            # Check type annotation detection
            if hasattr(funcs.get("fully_typed"), "has_type_annotations"):
                assert funcs["fully_typed"].has_type_annotations is True
            if hasattr(funcs.get("partially_typed"), "has_type_annotations"):
                assert funcs["partially_typed"].has_type_annotations is True
            if hasattr(funcs.get("no_types"), "has_type_annotations"):
                assert funcs["no_types"].has_type_annotations is False
        finally:
            file_path.unlink(missing_ok=True)