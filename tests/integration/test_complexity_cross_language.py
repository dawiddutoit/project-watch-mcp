"""Integration tests for cross-language complexity consistency.

This module validates that complexity metrics are calculated consistently
across all supported languages (Python, Java, Kotlin) for equivalent code patterns.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

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


class TestCrossLanguageConsistency:
    """Test suite for cross-language complexity consistency validation."""
    
    @pytest.fixture
    def registry(self):
        """Get the analyzer registry with all language analyzers."""
        return AnalyzerRegistry()
    
    def create_temp_file(self, content: str, suffix: str) -> Path:
        """Create a temporary file with given content and suffix."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            return Path(f.name)
    
    async def analyze_code_samples(
        self, 
        registry: AnalyzerRegistry,
        samples: Dict[str, Tuple[str, str]]
    ) -> Dict[str, ComplexityResult]:
        """Analyze code samples across languages."""
        results = {}
        for lang, (code, ext) in samples.items():
            file_path = self.create_temp_file(code, ext)
            try:
                analyzer = registry.get_analyzer(lang)
                result = await analyzer.analyze_file(file_path)
                results[lang] = result
            finally:
                file_path.unlink(missing_ok=True)
        return results
    
    @pytest.mark.asyncio
    async def test_simple_function_consistency(self, registry):
        """Test that simple functions have consistent complexity across languages."""
        samples = {
            "python": (
                """
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
""",
                ".py"
            ),
            "java": (
                """
public class Math {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int multiply(int x, int y) {
        return x * y;
    }
}
""",
                ".java"
            ),
            "kotlin": (
                """
class Math {
    fun add(a: Int, b: Int): Int {
        return a + b
    }
    
    fun multiply(x: Int, y: Int): Int {
        return x * y
    }
}
""",
                ".kt"
            )
        }
        
        results = await self.analyze_code_samples(registry, samples)
        
        # All languages should report 2 functions with complexity 1
        for lang, result in results.items():
            assert len(result.functions) == 2, f"{lang} should have 2 functions"
            for func in result.functions:
                assert func.complexity == 1, f"{lang} function {func.name} should have complexity 1"
                # Cognitive complexity may have a base value of 0 or 1 depending on implementation
                assert func.cognitive_complexity <= 1, f"{lang} function {func.name} should have cognitive complexity <= 1"
    
    @pytest.mark.asyncio
    async def test_if_statement_consistency(self, registry):
        """Test that if statements are counted consistently."""
        samples = {
            "python": (
                """
def check_value(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
""",
                ".py"
            ),
            "java": (
                """
public class Checker {
    public String checkValue(int x) {
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
                ".java"
            ),
            "kotlin": (
                """
class Checker {
    fun checkValue(x: Int): String {
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
                ".kt"
            )
        }
        
        results = await self.analyze_code_samples(registry, samples)
        
        # All should have cyclomatic complexity of 3 (1 base + 2 branches)
        for lang, result in results.items():
            assert len(result.functions) == 1, f"{lang} should have 1 function"
            func = result.functions[0]
            assert func.complexity == 3, f"{lang} should have cyclomatic complexity 3, got {func.complexity}"
            # Cognitive complexity may vary slightly due to language idioms
            assert 1 <= func.cognitive_complexity <= 3, f"{lang} cognitive complexity out of range: {func.cognitive_complexity}"
    
    @pytest.mark.asyncio
    async def test_loop_consistency(self, registry):
        """Test that loops are counted consistently."""
        samples = {
            "python": (
                """
def process_list(items):
    result = 0
    for item in items:
        if item > 0:
            result += item
    return result
""",
                ".py"
            ),
            "java": (
                """
public class Processor {
    public int processList(int[] items) {
        int result = 0;
        for (int item : items) {
            if (item > 0) {
                result += item;
            }
        }
        return result;
    }
}
""",
                ".java"
            ),
            "kotlin": (
                """
class Processor {
    fun processList(items: IntArray): Int {
        var result = 0
        for (item in items) {
            if (item > 0) {
                result += item
            }
        }
        return result
    }
}
""",
                ".kt"
            )
        }
        
        results = await self.analyze_code_samples(registry, samples)
        
        # All should have cyclomatic complexity of 3 (1 base + 1 for loop + 1 for if)
        for lang, result in results.items():
            assert len(result.functions) == 1, f"{lang} should have 1 function"
            func = result.functions[0]
            assert func.complexity == 3, f"{lang} should have cyclomatic complexity 3, got {func.complexity}"
            # Cognitive complexity includes nesting
            assert 2 <= func.cognitive_complexity <= 4, f"{lang} cognitive complexity out of range: {func.cognitive_complexity}"
    
    @pytest.mark.asyncio
    async def test_nested_conditions_consistency(self, registry):
        """Test that nested conditions are handled consistently."""
        samples = {
            "python": (
                """
def validate_input(value, config):
    if value is not None:
        if isinstance(value, int):
            if value > 0:
                if value <= config.get('max', 100):
                    return True
    return False
""",
                ".py"
            ),
            "java": (
                """
public class Validator {
    public boolean validateInput(Integer value, Map<String, Integer> config) {
        if (value != null) {
            if (value instanceof Integer) {
                if (value > 0) {
                    if (value <= config.getOrDefault("max", 100)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
}
""",
                ".java"
            ),
            "kotlin": (
                """
class Validator {
    fun validateInput(value: Int?, config: Map<String, Int>): Boolean {
        if (value != null) {
            if (value is Int) {
                if (value > 0) {
                    if (value <= (config["max"] ?: 100)) {
                        return true
                    }
                }
            }
        }
        return false
    }
}
""",
                ".kt"
            )
        }
        
        results = await self.analyze_code_samples(registry, samples)
        
        # All should have cyclomatic complexity of 5 (1 base + 4 if statements)
        for lang, result in results.items():
            assert len(result.functions) == 1, f"{lang} should have 1 function"
            func = result.functions[0]
            assert func.complexity == 5, f"{lang} should have cyclomatic complexity 5, got {func.complexity}"
            # Cognitive complexity should be higher due to nesting
            assert func.cognitive_complexity >= 6, f"{lang} cognitive complexity too low: {func.cognitive_complexity}"
    
    @pytest.mark.asyncio
    async def test_switch_case_consistency(self, registry):
        """Test that switch/when statements are handled consistently."""
        samples = {
            "python": (
                """
def get_day_name(day_num):
    if day_num == 1:
        return "Monday"
    elif day_num == 2:
        return "Tuesday"
    elif day_num == 3:
        return "Wednesday"
    elif day_num == 4:
        return "Thursday"
    elif day_num == 5:
        return "Friday"
    elif day_num == 6:
        return "Saturday"
    elif day_num == 7:
        return "Sunday"
    else:
        return "Invalid"
""",
                ".py"
            ),
            "java": (
                """
public class DayHelper {
    public String getDayName(int dayNum) {
        switch (dayNum) {
            case 1: return "Monday";
            case 2: return "Tuesday";
            case 3: return "Wednesday";
            case 4: return "Thursday";
            case 5: return "Friday";
            case 6: return "Saturday";
            case 7: return "Sunday";
            default: return "Invalid";
        }
    }
}
""",
                ".java"
            ),
            "kotlin": (
                """
class DayHelper {
    fun getDayName(dayNum: Int): String {
        return when (dayNum) {
            1 -> "Monday"
            2 -> "Tuesday"
            3 -> "Wednesday"
            4 -> "Thursday"
            5 -> "Friday"
            6 -> "Saturday"
            7 -> "Sunday"
            else -> "Invalid"
        }
    }
}
""",
                ".kt"
            )
        }
        
        results = await self.analyze_code_samples(registry, samples)
        
        # All should have similar complexity for equivalent switch logic
        complexities = {lang: result.functions[0].cyclomatic for lang, result in results.items()}
        
        # Python will have higher complexity due to elif chain
        assert complexities["python"] == 8, f"Python should have complexity 8, got {complexities['python']}"
        
        # Java and Kotlin switch/when should be similar
        assert complexities["java"] == 8, f"Java should have complexity 8, got {complexities['java']}"
        assert complexities["kotlin"] == 8, f"Kotlin should have complexity 8, got {complexities['kotlin']}"
    
    @pytest.mark.asyncio
    async def test_exception_handling_consistency(self, registry):
        """Test that exception handling is counted consistently."""
        samples = {
            "python": (
                """
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return 0
    except TypeError:
        return None
    finally:
        print("Division attempted")
""",
                ".py"
            ),
            "java": (
                """
public class Calculator {
    public Double safeDivide(double a, double b) {
        try {
            double result = a / b;
            return result;
        } catch (ArithmeticException e) {
            return 0.0;
        } catch (Exception e) {
            return null;
        } finally {
            System.out.println("Division attempted");
        }
    }
}
""",
                ".java"
            ),
            "kotlin": (
                """
class Calculator {
    fun safeDivide(a: Double, b: Double): Double? {
        return try {
            val result = a / b
            result
        } catch (e: ArithmeticException) {
            0.0
        } catch (e: Exception) {
            null
        } finally {
            println("Division attempted")
        }
    }
}
""",
                ".kt"
            )
        }
        
        results = await self.analyze_code_samples(registry, samples)
        
        # All should count exception handlers
        for lang, result in results.items():
            assert len(result.functions) == 1, f"{lang} should have 1 function"
            func = result.functions[0]
            # Base 1 + 2 catch blocks
            assert func.complexity >= 3, f"{lang} should have cyclomatic complexity >= 3, got {func.complexity}"
    
    @pytest.mark.asyncio
    async def test_logical_operators_consistency(self, registry):
        """Test that logical operators are counted consistently."""
        samples = {
            "python": (
                """
def is_valid(x, y, z):
    if x > 0 and y > 0 and z > 0:
        return True
    if x < 0 or y < 0 or z < 0:
        return False
    return x == 0 and y == 0 and z == 0
""",
                ".py"
            ),
            "java": (
                """
public class Validator {
    public boolean isValid(int x, int y, int z) {
        if (x > 0 && y > 0 && z > 0) {
            return true;
        }
        if (x < 0 || y < 0 || z < 0) {
            return false;
        }
        return x == 0 && y == 0 && z == 0;
    }
}
""",
                ".java"
            ),
            "kotlin": (
                """
class Validator {
    fun isValid(x: Int, y: Int, z: Int): Boolean {
        if (x > 0 && y > 0 && z > 0) {
            return true
        }
        if (x < 0 || y < 0 || z < 0) {
            return false
        }
        return x == 0 && y == 0 && z == 0
    }
}
""",
                ".kt"
            )
        }
        
        results = await self.analyze_code_samples(registry, samples)
        
        # All should count logical operators
        for lang, result in results.items():
            assert len(result.functions) == 1, f"{lang} should have 1 function"
            func = result.functions[0]
            # Base 1 + 2 if statements + logical operators
            assert func.complexity >= 9, f"{lang} should have cyclomatic complexity >= 9, got {func.complexity}"
    
    @pytest.mark.asyncio
    async def test_recursion_consistency(self, registry):
        """Test that recursive functions are handled consistently."""
        samples = {
            "python": (
                """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
                ".py"
            ),
            "java": (
                """
public class Math {
    public int factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }
}
""",
                ".java"
            ),
            "kotlin": (
                """
class Math {
    fun factorial(n: Int): Int {
        if (n <= 1) {
            return 1
        }
        return n * factorial(n - 1)
    }
}
""",
                ".kt"
            )
        }
        
        results = await self.analyze_code_samples(registry, samples)
        
        # All should have similar complexity for recursive functions
        for lang, result in results.items():
            assert len(result.functions) == 1, f"{lang} should have 1 function"
            func = result.functions[0]
            assert func.complexity == 2, f"{lang} should have cyclomatic complexity 2, got {func.complexity}"
            # Check if recursion is detected (if supported)
            if hasattr(func, 'is_recursive'):
                assert func.is_recursive, f"{lang} should detect recursion"
    
    @pytest.mark.asyncio
    async def test_lambda_consistency(self, registry):
        """Test that lambda/anonymous functions are handled consistently."""
        samples = {
            "python": (
                """
def process_data(items):
    filtered = list(filter(lambda x: x > 0, items))
    mapped = list(map(lambda x: x * 2, filtered))
    result = sum(mapped)
    return result
""",
                ".py"
            ),
            "java": (
                """
import java.util.Arrays;
import java.util.stream.Collectors;

public class Processor {
    public int processData(int[] items) {
        int result = Arrays.stream(items)
            .filter(x -> x > 0)
            .map(x -> x * 2)
            .sum();
        return result;
    }
}
""",
                ".java"
            ),
            "kotlin": (
                """
class Processor {
    fun processData(items: IntArray): Int {
        val result = items
            .filter { it > 0 }
            .map { it * 2 }
            .sum()
        return result
    }
}
""",
                ".kt"
            )
        }
        
        results = await self.analyze_code_samples(registry, samples)
        
        # All should detect lambda usage
        for lang, result in results.items():
            assert len(result.functions) >= 1, f"{lang} should have at least 1 function"
            func = result.functions[0]
            # Complexity should account for lambdas
            assert func.complexity >= 1, f"{lang} should have cyclomatic complexity >= 1"
            # Check lambda count if available
            if hasattr(func, 'lambda_count'):
                assert func.lambda_count >= 2, f"{lang} should detect at least 2 lambdas"


class TestComplexityGradeConsistency:
    """Test that complexity grades are assigned consistently across languages."""
    
    @pytest.fixture
    def registry(self):
        """Get the analyzer registry."""
        return AnalyzerRegistry()
    
    def create_complex_code(self, language: str) -> Tuple[str, str]:
        """Create complex code samples for each language."""
        if language == "python":
            return (
                """
def complex_function(data, config, mode):
    result = []
    
    for item in data:
        if item is None:
            continue
        
        if mode == "strict":
            if item.get('type') == 'A':
                if item.get('value') > config.get('threshold', 10):
                    if item.get('status') == 'active':
                        result.append(item)
                    elif item.get('status') == 'pending':
                        if config.get('allow_pending'):
                            result.append(item)
            elif item.get('type') == 'B':
                if item.get('priority') == 'high':
                    result.append(item)
                elif item.get('priority') == 'medium' and config.get('include_medium'):
                    result.append(item)
        elif mode == "lenient":
            if item.get('type') in ['A', 'B', 'C']:
                result.append(item)
        else:
            try:
                processed = process_item(item)
                if processed:
                    result.append(processed)
            except ValueError:
                pass
            except Exception as e:
                log_error(e)
    
    return result
""",
                ".py"
            )
        elif language == "java":
            return (
                """
public class ComplexProcessor {
    public List<Item> complexFunction(List<Item> data, Config config, String mode) {
        List<Item> result = new ArrayList<>();
        
        for (Item item : data) {
            if (item == null) {
                continue;
            }
            
            if ("strict".equals(mode)) {
                if ("A".equals(item.getType())) {
                    if (item.getValue() > config.getThreshold(10)) {
                        if ("active".equals(item.getStatus())) {
                            result.add(item);
                        } else if ("pending".equals(item.getStatus())) {
                            if (config.isAllowPending()) {
                                result.add(item);
                            }
                        }
                    }
                } else if ("B".equals(item.getType())) {
                    if ("high".equals(item.getPriority())) {
                        result.add(item);
                    } else if ("medium".equals(item.getPriority()) && config.isIncludeMedium()) {
                        result.add(item);
                    }
                }
            } else if ("lenient".equals(mode)) {
                if (Arrays.asList("A", "B", "C").contains(item.getType())) {
                    result.add(item);
                }
            } else {
                try {
                    Item processed = processItem(item);
                    if (processed != null) {
                        result.add(processed);
                    }
                } catch (IllegalArgumentException e) {
                    // ignore
                } catch (Exception e) {
                    logError(e);
                }
            }
        }
        
        return result;
    }
}
""",
                ".java"
            )
        else:  # kotlin
            return (
                """
class ComplexProcessor {
    fun complexFunction(data: List<Item>, config: Config, mode: String): List<Item> {
        val result = mutableListOf<Item>()
        
        for (item in data) {
            if (item == null) {
                continue
            }
            
            when (mode) {
                "strict" -> {
                    if (item.type == "A") {
                        if (item.value > (config.threshold ?: 10)) {
                            if (item.status == "active") {
                                result.add(item)
                            } else if (item.status == "pending") {
                                if (config.allowPending) {
                                    result.add(item)
                                }
                            }
                        }
                    } else if (item.type == "B") {
                        if (item.priority == "high") {
                            result.add(item)
                        } else if (item.priority == "medium" && config.includeMedium) {
                            result.add(item)
                        }
                    }
                }
                "lenient" -> {
                    if (item.type in listOf("A", "B", "C")) {
                        result.add(item)
                    }
                }
                else -> {
                    try {
                        val processed = processItem(item)
                        if (processed != null) {
                            result.add(processed)
                        }
                    } catch (e: IllegalArgumentException) {
                        // ignore
                    } catch (e: Exception) {
                        logError(e)
                    }
                }
            }
        }
        
        return result
    }
}
""",
                ".kt"
            )
    
    @pytest.mark.asyncio
    async def test_grade_assignment_consistency(self, registry):
        """Test that complexity grades are assigned consistently."""
        languages = ["python", "java", "kotlin"]
        results = {}
        
        for lang in languages:
            code, ext = self.create_complex_code(lang)
            with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                f.write(code)
                file_path = Path(f.name)
            
            try:
                analyzer = registry.get_analyzer(lang)
                result = await analyzer.analyze_file(file_path)
                results[lang] = result
            finally:
                file_path.unlink(missing_ok=True)
        
        # All should assign similar grades for complex code
        grades = {lang: result.summary.complexity_grade for lang, result in results.items()}
        
        # Complex function should get poor grade (D, E, or F)
        for lang, grade in grades.items():
            assert grade.value in ['D', 'E', 'F'], f"{lang} should assign poor grade for complex code, got {grade.value}"
        
        # Maintainability index should be low
        for lang, result in results.items():
            assert result.summary.maintainability_index < 50, f"{lang} MI should be < 50, got {result.summary.maintainability_index}"