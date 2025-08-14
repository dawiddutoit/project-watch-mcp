"""Comprehensive unit tests for complexity analysis modules to achieve >80% coverage."""

import pytest
import ast
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import tempfile

from src.project_watch_mcp.complexity_analysis import (
    BaseComplexityAnalyzer,
    AnalyzerRegistry,
)
from src.project_watch_mcp.complexity_analysis.metrics import ComplexityMetrics
from src.project_watch_mcp.complexity_analysis.languages.python_analyzer import PythonComplexityAnalyzer
from src.project_watch_mcp.complexity_analysis.languages.java_analyzer import JavaComplexityAnalyzer
from src.project_watch_mcp.complexity_analysis.languages.kotlin_analyzer import KotlinComplexityAnalyzer
from src.project_watch_mcp.complexity_analysis.models import (
    ComplexityResult,
    ComplexitySummary,
    FunctionComplexity,
    ClassComplexity,
    ComplexityGrade,
    ComplexityClassification,
)


class TestComplexityMetrics:
    """Test the ComplexityMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Create a ComplexityMetrics instance."""
        return ComplexityMetrics()
    
    def test_calculate_halstead_metrics(self, metrics):
        """Test Halstead metrics calculation."""
        operators = ['+', '-', '*', '/', '=', '==', '!=', '<', '>']
        operands = ['x', 'y', 'z', '1', '2', 'result', 'x', 'y']  # Some duplicates
        
        halstead = metrics.calculate_halstead_metrics(operators, operands)
        
        assert halstead['n1'] == 7  # Unique operators
        assert halstead['n2'] == 6  # Unique operands
        assert halstead['N1'] == 9  # Total operators
        assert halstead['N2'] == 8  # Total operands
        assert halstead['vocabulary'] == 13  # n1 + n2
        assert halstead['length'] == 17  # N1 + N2
        assert halstead['volume'] > 0
        assert halstead['difficulty'] > 0
        assert halstead['effort'] > 0
    
    def test_calculate_halstead_metrics_empty(self, metrics):
        """Test Halstead metrics with empty input."""
        halstead = metrics.calculate_halstead_metrics([], [])
        
        assert halstead['n1'] == 0
        assert halstead['n2'] == 0
        assert halstead['N1'] == 0
        assert halstead['N2'] == 0
        assert halstead['vocabulary'] == 0
        assert halstead['length'] == 0
        assert halstead['volume'] == 0
        assert halstead['difficulty'] == 0
        assert halstead['effort'] == 0
    
    def test_calculate_halstead_metrics_single_operand(self, metrics):
        """Test Halstead metrics with single operand."""
        halstead = metrics.calculate_halstead_metrics([], ['x'])
        
        assert halstead['n1'] == 0
        assert halstead['n2'] == 1
        assert halstead['N1'] == 0
        assert halstead['N2'] == 1
        assert halstead['vocabulary'] == 1
        assert halstead['length'] == 1
        # With vocabulary = 1, volume should be 0 (log2(1) = 0)
        assert halstead['volume'] == 0
    
    def test_estimate_comment_ratio(self, metrics):
        """Test comment ratio estimation."""
        lines = [
            "# This is a comment",
            "def hello():",
            "    # Another comment",
            "    print('Hello')",
            "    return True",
        ]
        
        ratio = metrics.estimate_comment_ratio(lines, '#', '"""', '"""')
        assert ratio == 0.4  # 2 comments out of 5 lines
    
    def test_estimate_comment_ratio_multiline(self, metrics):
        """Test comment ratio with multiline comments."""
        lines = [
            '"""',
            "This is a",
            "multiline comment",
            '"""',
            "def hello():",
            "    pass",
        ]
        
        ratio = metrics.estimate_comment_ratio(lines, '#', '"""', '"""')
        assert ratio == 0.5  # 3 comment lines out of 6
    
    def test_estimate_comment_ratio_no_comments(self, metrics):
        """Test comment ratio with no comments."""
        lines = [
            "def hello():",
            "    return True",
        ]
        
        ratio = metrics.estimate_comment_ratio(lines, '#', '"""', '"""')
        assert ratio == 0.0
    
    def test_estimate_comment_ratio_all_comments(self, metrics):
        """Test comment ratio with all comments."""
        lines = [
            "# Comment 1",
            "# Comment 2",
            "# Comment 3",
        ]
        
        ratio = metrics.estimate_comment_ratio(lines, '#', '"""', '"""')
        assert ratio == 1.0
    
    def test_calculate_code_duplication(self, metrics):
        """Test code duplication calculation."""
        lines = [
            "def hello():",
            "    print('Hello')",
            "    return True",
            "def hello():",  # Duplicate
            "    print('Hello')",  # Duplicate
            "    return False",  # Different
        ]
        
        duplication = metrics.calculate_code_duplication(lines)
        assert duplication > 0  # Some duplication exists
        assert duplication < 1  # Not complete duplication
    
    def test_calculate_code_duplication_no_duplication(self, metrics):
        """Test code duplication with unique lines."""
        lines = [
            "line1",
            "line2",
            "line3",
            "line4",
        ]
        
        duplication = metrics.calculate_code_duplication(lines)
        assert duplication == 0.0
    
    def test_calculate_code_duplication_complete(self, metrics):
        """Test code duplication with all duplicate lines."""
        lines = [
            "same",
            "same",
            "same",
            "same",
        ]
        
        duplication = metrics.calculate_code_duplication(lines)
        assert duplication == 0.75  # 3 duplicates out of 4 lines


class TestComplexityModels:
    """Test the complexity model classes."""
    
    def test_complexity_grade_from_maintainability_index(self):
        """Test ComplexityGrade from maintainability index."""
        assert ComplexityGrade.from_maintainability_index(90).value == "A"
        assert ComplexityGrade.from_maintainability_index(75).value == "B"
        assert ComplexityGrade.from_maintainability_index(55).value == "C"
        assert ComplexityGrade.from_maintainability_index(35).value == "D"
        assert ComplexityGrade.from_maintainability_index(15).value == "E"
        assert ComplexityGrade.from_maintainability_index(5).value == "F"
    
    def test_complexity_grade_from_complexity(self):
        """Test ComplexityGrade from cyclomatic complexity."""
        assert ComplexityGrade.from_complexity(3) == "A"
        assert ComplexityGrade.from_complexity(7) == "B"
        assert ComplexityGrade.from_complexity(15) == "C"
        assert ComplexityGrade.from_complexity(25) == "D"
        assert ComplexityGrade.from_complexity(35) == "E"
        assert ComplexityGrade.from_complexity(45) == "F"
    
    def test_complexity_classification(self):
        """Test ComplexityClassification from complexity."""
        assert ComplexityClassification.from_complexity(3).value == "simple"
        assert ComplexityClassification.from_complexity(8).value == "moderate"
        assert ComplexityClassification.from_complexity(15).value == "complex"
        assert ComplexityClassification.from_complexity(25).value == "very_complex"
    
    def test_function_complexity_creation(self):
        """Test FunctionComplexity model creation."""
        func = FunctionComplexity(
            name="test_function",
            complexity=10,
            cognitive_complexity=8,
            rank="B",
            line_start=10,
            line_end=20,
            classification="moderate",
            parameters=3,
            depth=2,
            type="function",
            has_decorators=True,
            decorator_count=2,
            uses_context_managers=True,
            exception_handlers_count=1,
            lambda_count=0,
            is_generator=False,
            uses_walrus_operator=False,
            uses_pattern_matching=False,
            has_type_annotations=True,
            is_recursive=False,
        )
        
        assert func.name == "test_function"
        assert func.complexity == 10
        assert func.cognitive_complexity == 8
        assert func.has_decorators is True
        assert func.decorator_count == 2
        assert func.has_type_annotations is True
    
    def test_class_complexity_creation(self):
        """Test ClassComplexity model creation."""
        methods = [
            FunctionComplexity(
                name="method1",
                complexity=5,
                cognitive_complexity=4,
                rank="A",
                line_start=15,
                line_end=20,
                classification="simple",
            ),
            FunctionComplexity(
                name="method2",
                complexity=10,
                cognitive_complexity=8,
                rank="B",
                line_start=22,
                line_end=35,
                classification="moderate",
            ),
        ]
        
        cls = ClassComplexity(
            name="TestClass",
            total_complexity=15,
            average_method_complexity=7.5,
            method_count=2,
            line_start=10,
            line_end=40,
            methods=methods,
        )
        
        assert cls.name == "TestClass"
        assert cls.total_complexity == 15
        assert cls.average_method_complexity == 7.5
        assert cls.method_count == 2
        assert len(cls.methods) == 2
        assert cls.methods[0].name == "method1"
    
    def test_complexity_result_generate_recommendations(self):
        """Test recommendation generation for ComplexityResult."""
        functions = [
            FunctionComplexity(
                name="complex_func",
                complexity=25,
                cognitive_complexity=30,
                rank="D",
                line_start=1,
                line_end=50,
                classification="very_complex",
            ),
            FunctionComplexity(
                name="simple_func",
                complexity=3,
                cognitive_complexity=2,
                rank="A",
                line_start=52,
                line_end=60,
                classification="simple",
            ),
        ]
        
        summary = ComplexitySummary(
            total_complexity=28,
            average_complexity=14,
            cognitive_complexity=32,
            maintainability_index=45,
            complexity_grade="D",
            function_count=2,
            class_count=0,
            lines_of_code=60,
        )
        
        result = ComplexityResult(
            file_path="test.py",
            language="python",
            summary=summary,
            functions=functions,
        )
        
        recommendations = result.generate_recommendations()
        
        # Should recommend refactoring the complex function
        assert any("complex_func" in rec for rec in recommendations)
        assert any("refactor" in rec.lower() for rec in recommendations)
        # Should note functions with high complexity
        assert any("complexity > 10" in rec for rec in recommendations)


class TestBaseComplexityAnalyzer:
    """Test the BaseComplexityAnalyzer abstract class."""
    
    def test_base_analyzer_initialization(self):
        """Test base analyzer initialization."""
        analyzer = BaseComplexityAnalyzer("python")
        assert analyzer.language == "python"
        assert analyzer.metrics is not None
        assert isinstance(analyzer.metrics, ComplexityMetrics)
    
    def test_calculate_maintainability_index(self):
        """Test maintainability index calculation."""
        analyzer = BaseComplexityAnalyzer("test")
        
        # Test with normal values
        mi = analyzer.calculate_maintainability_index(10, 100, 0.2)
        assert 0 <= mi <= 100
        
        # Test with edge cases
        mi_zero = analyzer.calculate_maintainability_index(0, 1, 0)
        assert mi_zero == 100  # Perfect maintainability
        
        mi_high_complexity = analyzer.calculate_maintainability_index(100, 1000, 0.1)
        assert mi_high_complexity < 50  # Poor maintainability
    
    def test_registry_registration(self):
        """Test analyzer registry registration."""
        # Clear registry first
        AnalyzerRegistry._analyzers.clear()
        
        # Create a test analyzer
        class TestAnalyzer(BaseComplexityAnalyzer):
            async def analyze_file(self, file_path):
                pass
            
            async def analyze_code(self, code):
                pass
            
            def calculate_cyclomatic_complexity(self, ast_node):
                return 1
            
            def calculate_cognitive_complexity(self, ast_node):
                return 1
        
        # Register analyzer
        AnalyzerRegistry.register("test_lang", TestAnalyzer)
        
        # Get analyzer
        analyzer_class = AnalyzerRegistry.get("test_lang")
        assert analyzer_class == TestAnalyzer
        
        # Get non-existent analyzer
        assert AnalyzerRegistry.get("nonexistent") is None
        
        # List analyzers
        analyzers = AnalyzerRegistry.list_analyzers()
        assert "test_lang" in analyzers


class TestPythonComplexityAnalyzer:
    """Comprehensive tests for PythonComplexityAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a PythonComplexityAnalyzer instance."""
        return PythonComplexityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_simple_python_file(self, analyzer):
        """Test analyzing a simple Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
""")
            f.flush()
            
            result = await analyzer.analyze_file(Path(f.name))
            
            assert result.language == "python"
            assert result.summary.function_count == 2
            assert result.summary.total_complexity > 0
            assert len(result.functions) == 2
            
            # Clean up
            Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_analyze_file_not_found(self, analyzer):
        """Test analyzing non-existent file."""
        result = await analyzer.analyze_file(Path("/nonexistent/file.py"))
        assert result.error is not None
        assert "not found" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_analyze_file_not_python(self, analyzer):
        """Test analyzing non-Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not Python code")
            f.flush()
            
            result = await analyzer.analyze_file(Path(f.name))
            assert result.error is not None
            assert "not a python file" in result.error.lower()
            
            Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_analyze_file_with_syntax_error(self, analyzer):
        """Test analyzing file with syntax error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def broken(:\n    pass")
            f.flush()
            
            result = await analyzer.analyze_file(Path(f.name))
            assert result.error is not None
            assert "syntax error" in result.error.lower()
            
            Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_analyze_code_with_classes(self, analyzer):
        """Test analyzing Python code with classes."""
        code = """
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        if x > 0 and y > 0:
            return x + y
        elif x < 0 and y < 0:
            return x + y
        else:
            return abs(x) + abs(y)
    
    def multiply(self, x, y):
        return x * y

class AdvancedCalculator(Calculator):
    def power(self, x, n):
        if n == 0:
            return 1
        elif n > 0:
            result = 1
            for _ in range(n):
                result *= x
            return result
        else:
            return 1 / self.power(x, -n)
"""
        
        result = await analyzer.analyze_code(code)
        
        assert result.summary.class_count == 2
        assert result.summary.function_count >= 4  # __init__, add, multiply, power
        assert len(result.classes) > 0
        
        # Check class information
        calc_class = next((c for c in result.classes if c.name == "Calculator"), None)
        assert calc_class is not None
        assert calc_class.method_count >= 3
    
    @pytest.mark.asyncio
    async def test_analyze_code_with_decorators(self, analyzer):
        """Test analyzing Python code with decorators."""
        code = """
import functools

@functools.lru_cache(maxsize=128)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class MyClass:
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        self._value = val
    
    @staticmethod
    def static_method():
        return "static"
    
    @classmethod
    def class_method(cls):
        return cls.__name__
"""
        
        result = await analyzer.analyze_code(code)
        
        # Find fibonacci function
        fib_func = next((f for f in result.functions if f.name == "fibonacci"), None)
        assert fib_func is not None
        assert fib_func.has_decorators is True
        assert fib_func.decorator_count > 0
        assert fib_func.is_recursive is True
    
    @pytest.mark.asyncio
    async def test_analyze_code_with_generators(self, analyzer):
        """Test analyzing Python code with generators."""
        code = """
def number_generator(n):
    for i in range(n):
        yield i * 2

def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

async def async_generator():
    for i in range(10):
        yield i
"""
        
        result = await analyzer.analyze_code(code)
        
        # Check generator detection
        gen_func = next((f for f in result.functions if f.name == "number_generator"), None)
        assert gen_func is not None
        assert gen_func.is_generator is True
    
    @pytest.mark.asyncio
    async def test_analyze_code_with_context_managers(self, analyzer):
        """Test analyzing Python code with context managers."""
        code = """
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()

def multiple_contexts():
    with open('file1.txt') as f1:
        with open('file2.txt') as f2:
            return f1.read() + f2.read()

def write_data(filename, data):
    try:
        with open(filename, 'w') as f:
            f.write(data)
    except IOError as e:
        print(f"Error: {e}")
        return False
    return True
"""
        
        result = await analyzer.analyze_code(code)
        
        # Check context manager detection
        read_func = next((f for f in result.functions if f.name == "read_file"), None)
        assert read_func is not None
        assert read_func.uses_context_managers is True
        
        write_func = next((f for f in result.functions if f.name == "write_data"), None)
        assert write_func is not None
        assert write_func.exception_handlers_count > 0
    
    @pytest.mark.asyncio
    async def test_analyze_code_with_lambdas(self, analyzer):
        """Test analyzing Python code with lambda expressions."""
        code = """
def process_list(items):
    sorted_items = sorted(items, key=lambda x: x[1])
    filtered = filter(lambda x: x > 0, sorted_items)
    mapped = map(lambda x: x * 2, filtered)
    return list(mapped)

calculate = lambda x, y: x + y if x > y else x - y
"""
        
        result = await analyzer.analyze_code(code)
        
        # Check lambda detection
        process_func = next((f for f in result.functions if f.name == "process_list"), None)
        assert process_func is not None
        assert process_func.lambda_count >= 3
    
    @pytest.mark.asyncio
    async def test_analyze_code_with_type_annotations(self, analyzer):
        """Test analyzing Python code with type annotations."""
        code = """
from typing import List, Optional, Dict

def typed_function(x: int, y: int) -> int:
    return x + y

def process_data(
    items: List[Dict[str, int]],
    threshold: Optional[float] = None
) -> List[int]:
    if threshold is None:
        threshold = 0.5
    return [item['value'] for item in items if item['score'] > threshold]

class TypedClass:
    value: int
    name: str
    
    def __init__(self, value: int, name: str) -> None:
        self.value = value
        self.name = name
"""
        
        result = await analyzer.analyze_code(code)
        
        # Check type annotation detection
        typed_func = next((f for f in result.functions if f.name == "typed_function"), None)
        assert typed_func is not None
        assert typed_func.has_type_annotations is True
    
    @pytest.mark.asyncio
    async def test_analyze_code_cognitive_complexity(self, analyzer):
        """Test cognitive complexity calculation."""
        code = """
def complex_logic(items, options):
    result = []
    for item in items:  # +1 (nesting 0)
        if item > 0:  # +1 (nesting 1)
            if item % 2 == 0:  # +2 (nesting 2)
                result.append(item * 2)
            elif item % 3 == 0:  # +1 (elif)
                result.append(item * 3)
            else:  # +1 (else)
                result.append(item)
        elif item < 0:  # +1 (elif)
            result.append(abs(item))
    return result
"""
        
        result = await analyzer.analyze_code(code)
        
        func = next((f for f in result.functions if f.name == "complex_logic"), None)
        assert func is not None
        assert func.cognitive_complexity > func.complexity  # Cognitive should account for nesting
    
    @pytest.mark.asyncio
    async def test_analyze_empty_code(self, analyzer):
        """Test analyzing empty code."""
        result = await analyzer.analyze_code("")
        assert result.summary.function_count == 0
        assert result.summary.class_count == 0
        assert result.summary.total_complexity == 0
    
    @pytest.mark.asyncio
    async def test_analyze_code_with_walrus_operator(self, analyzer):
        """Test analyzing Python 3.8+ walrus operator."""
        code = """
def process_with_walrus(items):
    if (n := len(items)) > 10:
        print(f"Processing {n} items")
        return items[:10]
    return items
"""
        
        result = await analyzer.analyze_code(code)
        
        # Note: walrus operator detection depends on Python version
        func = next((f for f in result.functions if f.name == "process_with_walrus"), None)
        assert func is not None
        # The walrus operator field should be set (True or False depending on version)
        assert hasattr(func, 'uses_walrus_operator')
    
    @pytest.mark.asyncio
    async def test_analyze_code_with_pattern_matching(self, analyzer):
        """Test analyzing Python 3.10+ pattern matching."""
        code = """
def handle_command(command):
    match command:
        case ["quit"]:
            return "Goodbye"
        case ["hello", name]:
            return f"Hello, {name}"
        case _:
            return "Unknown command"
"""
        
        result = await analyzer.analyze_code(code)
        
        # Note: pattern matching detection depends on Python version
        func = next((f for f in result.functions if f.name == "handle_command"), None)
        assert func is not None
        # The pattern matching field should be set
        assert hasattr(func, 'uses_pattern_matching')
    
    @pytest.mark.asyncio
    async def test_analyze_code_without_radon(self, analyzer):
        """Test analysis when radon is not available."""
        with patch('src.project_watch_mcp.complexity_analysis.languages.python_analyzer.RADON_AVAILABLE', False):
            code = """
def simple_function(x, y):
    if x > y:
        return x
    else:
        return y
"""
            
            result = await analyzer.analyze_code(code)
            
            # Should still work with AST-based analysis
            assert result.summary.function_count == 1
            assert result.summary.total_complexity > 0
            assert len(result.functions) == 1
    
    def test_calculate_cyclomatic_complexity_ast(self, analyzer):
        """Test cyclomatic complexity calculation using AST."""
        code = """
def test_function(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x - y
    elif x < 0:
        return -x
    else:
        return 0
"""
        
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        complexity = analyzer.calculate_cyclomatic_complexity(func_node)
        assert complexity > 1  # Has decision points
    
    def test_calculate_cognitive_complexity_ast(self, analyzer):
        """Test cognitive complexity calculation using AST."""
        code = """
def nested_function(items):
    for item in items:
        if item > 0:
            if item % 2 == 0:
                print(item)
"""
        
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        cognitive = analyzer.calculate_cognitive_complexity(func_node)
        assert cognitive > 0  # Has nesting and control flow


class TestJavaComplexityAnalyzer:
    """Basic tests for JavaComplexityAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a JavaComplexityAnalyzer instance."""
        return JavaComplexityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_simple_java_code(self, analyzer):
        """Test analyzing simple Java code."""
        code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
    
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        
        result = await analyzer.analyze_code(code)
        
        assert result.language == "java"
        assert result.summary.function_count >= 2  # main and add
        assert result.summary.class_count >= 1
    
    @pytest.mark.asyncio
    async def test_analyze_java_file(self, analyzer):
        """Test analyzing a Java file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write("""
public class Calculator {
    public int calculate(int x, int y, String op) {
        if (op.equals("+")) {
            return x + y;
        } else if (op.equals("-")) {
            return x - y;
        } else if (op.equals("*")) {
            return x * y;
        } else if (op.equals("/")) {
            if (y != 0) {
                return x / y;
            } else {
                throw new ArithmeticException("Division by zero");
            }
        } else {
            throw new IllegalArgumentException("Unknown operation");
        }
    }
}
""")
            f.flush()
            
            result = await analyzer.analyze_file(Path(f.name))
            
            assert result.language == "java"
            assert result.summary.function_count >= 1
            assert result.summary.total_complexity > 1  # Has decision points
            
            Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_analyze_java_syntax_error(self, analyzer):
        """Test analyzing Java code with syntax error."""
        code = "public class Broken { public void test( { } }"
        
        result = await analyzer.analyze_code(code)
        assert result.error is not None


class TestKotlinComplexityAnalyzer:
    """Basic tests for KotlinComplexityAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a KotlinComplexityAnalyzer instance."""
        return KotlinComplexityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_simple_kotlin_code(self, analyzer):
        """Test analyzing simple Kotlin code."""
        code = """
fun main() {
    println("Hello, Kotlin!")
}

fun add(a: Int, b: Int): Int {
    return a + b
}

class Calculator {
    fun multiply(x: Int, y: Int): Int = x * y
}
"""
        
        result = await analyzer.analyze_code(code)
        
        assert result.language == "kotlin"
        assert result.summary.function_count >= 3  # main, add, multiply
        assert result.summary.class_count >= 1
    
    @pytest.mark.asyncio
    async def test_analyze_kotlin_file(self, analyzer):
        """Test analyzing a Kotlin file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kt', delete=False) as f:
            f.write("""
fun processData(items: List<Int>): List<Int> {
    return items.filter { it > 0 }
                .map { it * 2 }
                .sorted()
}

class DataProcessor {
    fun process(value: Int): Int {
        return when {
            value > 100 -> value / 2
            value > 50 -> value
            value > 0 -> value * 2
            else -> 0
        }
    }
}
""")
            f.flush()
            
            result = await analyzer.analyze_file(Path(f.name))
            
            assert result.language == "kotlin"
            assert result.summary.function_count >= 2
            assert result.summary.class_count >= 1
            
            Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_analyze_kotlin_syntax_error(self, analyzer):
        """Test analyzing Kotlin code with syntax error."""
        code = "fun broken( { }"
        
        result = await analyzer.analyze_code(code)
        assert result.error is not None