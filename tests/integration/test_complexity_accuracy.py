"""Integration tests for complexity analyzer accuracy validation.

This module validates the accuracy of complexity calculations against
manually verified expected values and established complexity tools.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from project_watch_mcp.complexity_analysis import AnalyzerRegistry
from project_watch_mcp.complexity_analysis.models import (
    ComplexityResult,
    FunctionComplexity,
)

# Import analyzers to ensure they register themselves
from project_watch_mcp.complexity_analysis.languages import (
    PythonComplexityAnalyzer,
    JavaComplexityAnalyzer,
    KotlinComplexityAnalyzer,
)


class TestComplexityAccuracy:
    """Test suite for validating complexity calculation accuracy."""
    
    @pytest.fixture
    def registry(self):
        """Get the analyzer registry."""
        return AnalyzerRegistry()
    
    def assert_complexity_match(
        self,
        actual: FunctionComplexity,
        expected_cyclomatic: int,
        expected_cognitive: int,
        tolerance: float = 0.05
    ):
        """Assert that complexity matches expected values within tolerance."""
        cyclo_diff = abs(actual.complexity - expected_cyclomatic)
        cog_diff = abs(actual.cognitive_complexity - expected_cognitive)
        
        # Allow 5% deviation by default
        cyclo_tolerance = max(1, expected_cyclomatic * tolerance)
        cog_tolerance = max(1, expected_cognitive * tolerance)
        
        assert cyclo_diff <= cyclo_tolerance, (
            f"Cyclomatic complexity {actual.complexity} deviates from "
            f"expected {expected_cyclomatic} by more than {tolerance*100}%"
        )
        assert cog_diff <= cog_tolerance, (
            f"Cognitive complexity {actual.cognitive_complexity} deviates from "
            f"expected {expected_cognitive} by more than {tolerance*100}%"
        )
    
    @pytest.mark.asyncio
    async def test_python_accuracy_simple(self, registry):
        """Test Python analyzer accuracy for simple functions."""
        test_cases = [
            # (code, function_name, expected_cyclomatic, expected_cognitive)
            (
                """
def simple_function():
    return 42
""",
                "simple_function", 1, 0
            ),
            (
                """
def single_if(x):
    if x > 0:
        return "positive"
    return "non-positive"
""",
                "single_if", 2, 1
            ),
            (
                """
def if_elif_else(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
""",
                "if_elif_else", 3, 2
            ),
            (
                """
def nested_if(x, y):
    if x > 0:
        if y > 0:
            return "both positive"
        else:
            return "x positive, y not"
    return "x not positive"
""",
                "nested_if", 3, 3
            ),
            (
                """
def simple_loop(items):
    count = 0
    for item in items:
        count += 1
    return count
""",
                "simple_loop", 2, 1
            ),
            (
                """
def loop_with_condition(items):
    count = 0
    for item in items:
        if item > 0:
            count += 1
    return count
""",
                "loop_with_condition", 3, 3
            ),
        ]
        
        analyzer = registry.get_analyzer("python")
        
        for code, func_name, expected_cyclo, expected_cog in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                file_path = Path(f.name)
            
            try:
                result = await analyzer.analyze_file(file_path)
                func = next((f for f in result.functions if f.name == func_name), None)
                assert func is not None, f"Function {func_name} not found"
                self.assert_complexity_match(func, expected_cyclo, expected_cog)
            finally:
                file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_java_accuracy_simple(self, registry):
        """Test Java analyzer accuracy for simple methods."""
        test_cases = [
            # (code, method_name, expected_cyclomatic, expected_cognitive)
            (
                """
public class Test {
    public int simpleMethod() {
        return 42;
    }
}
""",
                "simpleMethod", 1, 0
            ),
            (
                """
public class Test {
    public String singleIf(int x) {
        if (x > 0) {
            return "positive";
        }
        return "non-positive";
    }
}
""",
                "singleIf", 2, 1
            ),
            (
                """
public class Test {
    public String ifElseIf(int x) {
        if (x > 0) {
            return "positive";
        } else if (x < 0) {
            return "negative";
        } else {
            return "zero";
        }
    }
}
""",
                "ifElseIf", 3, 2
            ),
            (
                """
public class Test {
    public String nestedIf(int x, int y) {
        if (x > 0) {
            if (y > 0) {
                return "both positive";
            } else {
                return "x positive, y not";
            }
        }
        return "x not positive";
    }
}
""",
                "nestedIf", 3, 3
            ),
            (
                """
public class Test {
    public int simpleLoop(int[] items) {
        int count = 0;
        for (int item : items) {
            count++;
        }
        return count;
    }
}
""",
                "simpleLoop", 2, 1
            ),
            (
                """
public class Test {
    public int loopWithCondition(int[] items) {
        int count = 0;
        for (int item : items) {
            if (item > 0) {
                count++;
            }
        }
        return count;
    }
}
""",
                "loopWithCondition", 3, 3
            ),
        ]
        
        analyzer = registry.get_analyzer("java")
        
        for code, method_name, expected_cyclo, expected_cog in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                f.write(code)
                file_path = Path(f.name)
            
            try:
                result = await analyzer.analyze_file(file_path)
                func = next((f for f in result.functions if f.name == method_name), None)
                assert func is not None, f"Method {method_name} not found"
                self.assert_complexity_match(func, expected_cyclo, expected_cog)
            finally:
                file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_kotlin_accuracy_simple(self, registry):
        """Test Kotlin analyzer accuracy for simple functions."""
        test_cases = [
            # (code, function_name, expected_cyclomatic, expected_cognitive)
            (
                """
class Test {
    fun simpleFunction(): Int {
        return 42
    }
}
""",
                "simpleFunction", 1, 0
            ),
            (
                """
class Test {
    fun singleIf(x: Int): String {
        if (x > 0) {
            return "positive"
        }
        return "non-positive"
    }
}
""",
                "singleIf", 2, 1
            ),
            (
                """
class Test {
    fun ifElseIf(x: Int): String {
        return if (x > 0) {
            "positive"
        } else if (x < 0) {
            "negative"
        } else {
            "zero"
        }
    }
}
""",
                "ifElseIf", 3, 2
            ),
            (
                """
class Test {
    fun nestedIf(x: Int, y: Int): String {
        if (x > 0) {
            if (y > 0) {
                return "both positive"
            } else {
                return "x positive, y not"
            }
        }
        return "x not positive"
    }
}
""",
                "nestedIf", 3, 3
            ),
            (
                """
class Test {
    fun simpleLoop(items: IntArray): Int {
        var count = 0
        for (item in items) {
            count++
        }
        return count
    }
}
""",
                "simpleLoop", 2, 1
            ),
            (
                """
class Test {
    fun loopWithCondition(items: IntArray): Int {
        var count = 0
        for (item in items) {
            if (item > 0) {
                count++
            }
        }
        return count
    }
}
""",
                "loopWithCondition", 3, 3
            ),
        ]
        
        analyzer = registry.get_analyzer("kotlin")
        
        for code, func_name, expected_cyclo, expected_cog in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.kt', delete=False) as f:
                f.write(code)
                file_path = Path(f.name)
            
            try:
                result = await analyzer.analyze_file(file_path)
                func = next((f for f in result.functions if f.name == func_name), None)
                assert func is not None, f"Function {func_name} not found"
                self.assert_complexity_match(func, expected_cyclo, expected_cog)
            finally:
                file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_complex_real_world_patterns(self, registry):
        """Test accuracy for real-world complex code patterns."""
        # Python complex pattern
        python_code = """
def process_order(order, inventory, customer, config):
    '''Real-world order processing with multiple decision points.'''
    
    # Validate order
    if not order or not order.items:
        return {"status": "error", "message": "Invalid order"}
    
    # Check customer status
    if customer.is_blocked:
        return {"status": "error", "message": "Customer blocked"}
    
    discount = 0
    if customer.is_premium:
        if order.total > 1000:
            discount = 0.15
        elif order.total > 500:
            discount = 0.10
        else:
            discount = 0.05
    elif customer.is_member:
        if order.total > 1000:
            discount = 0.10
        else:
            discount = 0.03
    
    # Process each item
    processed_items = []
    for item in order.items:
        if item.quantity <= 0:
            continue
        
        # Check inventory
        stock = inventory.get(item.product_id)
        if not stock:
            if config.allow_backorder:
                item.status = "backorder"
            else:
                item.status = "unavailable"
                continue
        elif stock.quantity < item.quantity:
            if config.allow_partial:
                item.quantity = stock.quantity
                item.status = "partial"
            else:
                item.status = "insufficient"
                continue
        
        # Apply discount
        item.price = item.price * (1 - discount)
        
        # Check special conditions
        if item.is_fragile and not customer.accepts_fragile_shipping:
            item.requires_insurance = True
            item.shipping_cost *= 1.5
        
        processed_items.append(item)
    
    if not processed_items:
        return {"status": "error", "message": "No items available"}
    
    # Calculate totals
    subtotal = sum(item.price * item.quantity for item in processed_items)
    tax = subtotal * config.tax_rate if not customer.tax_exempt else 0
    shipping = calculate_shipping(processed_items, customer.location)
    total = subtotal + tax + shipping
    
    return {
        "status": "success",
        "items": processed_items,
        "subtotal": subtotal,
        "tax": tax,
        "shipping": shipping,
        "total": total,
        "discount_applied": discount
    }
"""
        
        # Expected complexity for the complex function
        # Cyclomatic: ~22 (many decision points)
        # Cognitive: ~35+ (nested conditions)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            file_path = Path(f.name)
        
        try:
            analyzer = registry.get_analyzer("python")
            result = await analyzer.analyze_file(str(file_path))
            func = result.functions[0]
            
            # Allow 10% deviation for complex real-world code
            assert 20 <= func.complexity <= 25, f"Expected cyclomatic ~22, got {func.complexity}"
            assert func.cognitive_complexity >= 30, f"Expected cognitive >= 30, got {func.cognitive_complexity}"
            
            # Check maintainability
            assert result.summary.maintainability_index < 50, "Complex code should have low maintainability"
        finally:
            file_path.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_edge_cases_accuracy(self, registry):
        """Test accuracy for edge cases."""
        edge_cases = {
            "python": [
                # Empty file
                ("", 0, 0, 0),
                # Only comments
                ("# This is a comment\n# Another comment", 0, 0, 0),
                # Single expression
                ("result = 2 + 2", 0, 0, 0),
                # Lambda function
                ("square = lambda x: x * x", 0, 0, 0),
                # Class with no methods
                ("class Empty:\n    pass", 0, 0, 0),
            ],
            "java": [
                # Empty class
                ("public class Empty {}", 0, 0, 0),
                # Only fields
                ("public class Data { private int x; private String y; }", 0, 0, 0),
                # Interface
                ("public interface Service { void process(); }", 0, 0, 0),
            ],
            "kotlin": [
                # Empty class
                ("class Empty", 0, 0, 0),
                # Data class
                ("data class Point(val x: Int, val y: Int)", 0, 0, 0),
                # Object declaration
                ("object Singleton { val instance = \"single\" }", 0, 0, 0),
            ]
        }
        
        for lang, cases in edge_cases.items():
            analyzer = registry.get_analyzer(lang)
            ext = ".py" if lang == "python" else ".java" if lang == "java" else ".kt"
            
            for code, expected_funcs, expected_cyclo, expected_cog in cases:
                with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                    f.write(code)
                    file_path = Path(f.name)
                
                try:
                    result = await analyzer.analyze_file(file_path)
                    assert len(result.functions) == expected_funcs, (
                        f"{lang}: Expected {expected_funcs} functions, got {len(result.functions)}"
                    )
                    if expected_funcs > 0:
                        total_cyclo = sum(f.cyclomatic for f in result.functions)
                        total_cog = sum(f.cognitive for f in result.functions)
                        assert total_cyclo == expected_cyclo
                        assert total_cog == expected_cog
                finally:
                    file_path.unlink(missing_ok=True)


class TestPerformanceBenchmarks:
    """Performance benchmarks for complexity analyzers."""
    
    @pytest.fixture
    def registry(self):
        """Get the analyzer registry."""
        return AnalyzerRegistry()
    
    def generate_large_file(self, language: str, num_functions: int = 100) -> str:
        """Generate a large file with many functions for benchmarking."""
        if language == "python":
            code = []
            for i in range(num_functions):
                code.append(f"""
def function_{i}(x, y):
    if x > {i}:
        for j in range(y):
            if j % 2 == 0:
                result = x * j
            else:
                result = x + j
        return result
    else:
        return x + y
""")
            return "\n".join(code)
        
        elif language == "java":
            methods = []
            for i in range(num_functions):
                methods.append(f"""
    public int function_{i}(int x, int y) {{
        if (x > {i}) {{
            int result = 0;
            for (int j = 0; j < y; j++) {{
                if (j % 2 == 0) {{
                    result = x * j;
                }} else {{
                    result = x + j;
                }}
            }}
            return result;
        }} else {{
            return x + y;
        }}
    }}
""")
            return f"public class Benchmark {{\n{''.join(methods)}\n}}"
        
        else:  # kotlin
            methods = []
            for i in range(num_functions):
                methods.append(f"""
    fun function_{i}(x: Int, y: Int): Int {{
        return if (x > {i}) {{
            var result = 0
            for (j in 0 until y) {{
                result = if (j % 2 == 0) {{
                    x * j
                }} else {{
                    x + j
                }}
            }}
            result
        }} else {{
            x + y
        }}
    }}
""")
            return f"class Benchmark {{\n{''.join(methods)}\n}}"
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_python_performance(self, registry):
        """Benchmark Python analyzer performance."""
        analyzer = registry.get_analyzer("python")
        
        # Test different file sizes
        sizes = [10, 50, 100, 200]
        times = []
        
        for size in sizes:
            code = self.generate_large_file("python", size)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                file_path = Path(f.name)
            
            try:
                start = time.time()
                result = await analyzer.analyze_file(file_path)
                elapsed = time.time() - start
                times.append((size, elapsed))
                
                # Performance assertions
                functions_per_second = size / elapsed
                assert functions_per_second > 50, f"Python analyzer too slow: {functions_per_second:.1f} funcs/sec"
                
                # Verify correctness
                assert len(result.functions) == size, f"Expected {size} functions, got {len(result.functions)}"
            finally:
                file_path.unlink(missing_ok=True)
        
        # Log performance results
        for size, elapsed in times:
            print(f"Python: {size} functions in {elapsed:.3f}s ({size/elapsed:.1f} funcs/sec)")
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_java_performance(self, registry):
        """Benchmark Java analyzer performance."""
        analyzer = registry.get_analyzer("java")
        
        sizes = [10, 50, 100, 200]
        times = []
        
        for size in sizes:
            code = self.generate_large_file("java", size)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                f.write(code)
                file_path = Path(f.name)
            
            try:
                start = time.time()
                result = await analyzer.analyze_file(file_path)
                elapsed = time.time() - start
                times.append((size, elapsed))
                
                # Performance assertions
                functions_per_second = size / elapsed
                assert functions_per_second > 30, f"Java analyzer too slow: {functions_per_second:.1f} funcs/sec"
                
                # Verify correctness
                assert len(result.functions) == size, f"Expected {size} methods, got {len(result.functions)}"
            finally:
                file_path.unlink(missing_ok=True)
        
        # Log performance results
        for size, elapsed in times:
            print(f"Java: {size} methods in {elapsed:.3f}s ({size/elapsed:.1f} methods/sec)")
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_kotlin_performance(self, registry):
        """Benchmark Kotlin analyzer performance."""
        analyzer = registry.get_analyzer("kotlin")
        
        sizes = [10, 50, 100, 200]
        times = []
        
        for size in sizes:
            code = self.generate_large_file("kotlin", size)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.kt', delete=False) as f:
                f.write(code)
                file_path = Path(f.name)
            
            try:
                start = time.time()
                result = await analyzer.analyze_file(file_path)
                elapsed = time.time() - start
                times.append((size, elapsed))
                
                # Performance assertions
                functions_per_second = size / elapsed
                assert functions_per_second > 30, f"Kotlin analyzer too slow: {functions_per_second:.1f} funcs/sec"
                
                # Verify correctness
                assert len(result.functions) == size, f"Expected {size} functions, got {len(result.functions)}"
            finally:
                file_path.unlink(missing_ok=True)
        
        # Log performance results
        for size, elapsed in times:
            print(f"Kotlin: {size} functions in {elapsed:.3f}s ({size/elapsed:.1f} funcs/sec)")
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_cross_language_performance_comparison(self, registry):
        """Compare performance across all language analyzers."""
        languages = ["python", "java", "kotlin"]
        file_size = 100
        results = {}
        
        for lang in languages:
            analyzer = registry.get_analyzer(lang)
            code = self.generate_large_file(lang, file_size)
            ext = ".py" if lang == "python" else ".java" if lang == "java" else ".kt"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                f.write(code)
                file_path = Path(f.name)
            
            try:
                # Warm up
                await analyzer.analyze_file(str(file_path))
                
                # Actual benchmark
                start = time.time()
                for _ in range(3):  # Average of 3 runs
                    await analyzer.analyze_file(str(file_path))
                elapsed = (time.time() - start) / 3
                
                results[lang] = {
                    "time": elapsed,
                    "functions_per_second": file_size / elapsed
                }
            finally:
                file_path.unlink(missing_ok=True)
        
        # Log comparison
        print("\nPerformance Comparison (100 functions):")
        for lang, metrics in results.items():
            print(f"{lang:10} {metrics['time']:.3f}s ({metrics['functions_per_second']:.1f} funcs/sec)")
        
        # Verify all analyzers meet minimum performance
        for lang, metrics in results.items():
            assert metrics['functions_per_second'] > 25, (
                f"{lang} analyzer below minimum performance threshold"
            )