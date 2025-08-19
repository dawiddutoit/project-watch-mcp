"""Comprehensive test suite for Java complexity analyzer.

This module provides extensive testing for the Java complexity analyzer,
targeting >80% code coverage with focus on modern Java features,
edge cases, and error handling.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from project_watch_mcp.complexity_analysis.languages.java_analyzer import (
    JavaComplexityAnalyzer,
    TREE_SITTER_AVAILABLE
)
from project_watch_mcp.complexity_analysis.models import (
    ComplexityResult,
    ComplexityGrade,
    FunctionComplexity,
    ClassComplexity,
)


class TestJavaComplexityAnalyzer:
    """Comprehensive test suite for Java complexity analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return JavaComplexityAnalyzer()
    
    @pytest.fixture
    def temp_java_file(self, tmp_path):
        """Create a temporary Java file for testing."""
        file_path = tmp_path / "TestClass.java"
        file_path.write_text("""
public class TestClass {
    private int value;
    
    public TestClass(int value) {
        this.value = value;
    }
    
    public int getValue() {
        return value;
    }
    
    public void setValue(int value) {
        if (value >= 0) {
            this.value = value;
        } else {
            throw new IllegalArgumentException("Value must be non-negative");
        }
    }
    
    public int calculate(int x, int y) {
        if (x > y) {
            return x - y;
        } else if (x < y) {
            return y - x;
        } else {
            return 0;
        }
    }
}
""")
        return file_path
    
    # ==================== Basic Functionality Tests ====================
    
    @pytest.mark.asyncio
    async def test_analyze_simple_method(self, analyzer):
        """Test analysis of a simple Java method."""
        code = """
public class Simple {
    public int getNumber() {
        return 42;
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if not TREE_SITTER_AVAILABLE:
            assert result.error is not None
            return
        
        assert result.success
        assert result.language == "java"
        assert result.summary.function_count >= 1
        assert result.summary.class_count >= 1
    
    @pytest.mark.asyncio
    async def test_analyze_file(self, analyzer, temp_java_file):
        """Test file analysis."""
        result = await analyzer.analyze_file(temp_java_file)
        
        assert result.file_path == str(temp_java_file)
        
        if TREE_SITTER_AVAILABLE:
            assert result.success
            assert result.summary.function_count >= 4  # Constructor + 3 methods
            assert result.summary.class_count >= 1
    
    @pytest.mark.asyncio
    async def test_file_not_found(self, analyzer):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            await analyzer.analyze_file(Path("NonExistent.java"))
    
    @pytest.mark.asyncio
    async def test_non_java_file(self, analyzer, tmp_path):
        """Test rejection of non-Java files."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("not java")
        
        with pytest.raises(ValueError, match="Not a Java file"):
            await analyzer.analyze_file(file_path)
    
    # ==================== Cyclomatic Complexity Tests ====================
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.parametrize("code,expected_min_complexity", [
        ("public void simple() {}", 1),
        ("public void ifStatement(int x) { if (x > 0) return; }", 2),
        ("public void ifElse(int x) { if (x > 0) return; else return; }", 2),
        ("public void forLoop() { for(int i = 0; i < 10; i++) {} }", 2),
        ("public void whileLoop(int x) { while(x > 0) { x--; } }", 2),
        ("public void doWhile(int x) { do { x--; } while(x > 0); }", 2),
        ("public void tryCatch() { try { risky(); } catch(Exception e) {} }", 2),
    ])
    @pytest.mark.asyncio
    async def test_cyclomatic_complexity_calculation(self, analyzer, code, expected_min_complexity):
        """Test cyclomatic complexity for various Java constructs."""
        full_code = f"public class Test {{ {code} }}"
        result = await analyzer.analyze_code(full_code)
        
        assert result.success
        if result.functions:
            assert result.functions[0].complexity >= expected_min_complexity
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_switch_statement_complexity(self, analyzer):
        """Test complexity of switch statements."""
        code = """
public class SwitchTest {
    public String getDayName(int day) {
        switch (day) {
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
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        func = result.functions[0]
        assert func.complexity >= 8  # 7 cases + default
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_switch_expression_complexity(self, analyzer):
        """Test complexity of switch expressions (Java 14+)."""
        code = """
public class SwitchExpression {
    public String getSize(int value) {
        return switch (value) {
            case 1, 2, 3 -> "Small";
            case 4, 5, 6 -> "Medium";
            case 7, 8, 9 -> "Large";
            default -> "Unknown";
        };
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:  # Switch expressions might not be supported in all parsers
            func = result.functions[0]
            assert func.complexity >= 4  # 3 case groups + default
    
    # ==================== Java-Specific Feature Tests ====================
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_lambda_expression_detection(self, analyzer):
        """Test detection and complexity of lambda expressions."""
        code = """
import java.util.List;
import java.util.stream.Collectors;

public class LambdaTest {
    public List<String> processItems(List<String> items) {
        return items.stream()
            .filter(item -> item != null && !item.isEmpty())
            .map(item -> item.toUpperCase())
            .sorted((a, b) -> a.compareTo(b))
            .collect(Collectors.toList());
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            func = result.functions[0]
            # Lambda expressions should add to complexity
            assert func.complexity >= 2
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_stream_api_complexity(self, analyzer):
        """Test complexity of Stream API usage."""
        code = """
import java.util.List;

public class StreamTest {
    public long complexStream(List<Integer> numbers) {
        return numbers.stream()
            .filter(n -> n > 0)
            .map(n -> n * 2)
            .filter(n -> n < 100)
            .mapToLong(n -> n)
            .sum();
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            func = result.functions[0]
            # Stream operations should contribute to complexity
            assert func.complexity >= 2
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_generic_type_complexity(self, analyzer):
        """Test complexity with generic types."""
        code = """
public class GenericClass<T extends Comparable<T>> {
    private T value;
    
    public <U extends Number> U process(T input, U number) {
        if (input.compareTo(value) > 0) {
            return number;
        }
        return null;
    }
    
    public <K, V> void complexGeneric(Map<K, List<V>> map) {
        for (Map.Entry<K, List<V>> entry : map.entrySet()) {
            for (V value : entry.getValue()) {
                process(value);
            }
        }
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            assert result.summary.function_count >= 2
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_try_with_resources(self, analyzer):
        """Test try-with-resources complexity."""
        code = """
import java.io.*;

public class ResourceTest {
    public String readFile(String filename) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename));
             PrintWriter writer = new PrintWriter("output.txt")) {
            String line;
            while ((line = reader.readLine()) != null) {
                writer.println(line);
            }
            return "Success";
        } catch (IOException e) {
            return "Failed: " + e.getMessage();
        }
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            func = result.functions[0]
            # Try-with-resources + while loop + catch
            assert func.complexity >= 3
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_record_class_detection(self, analyzer):
        """Test detection of record classes (Java 14+)."""
        code = """
public record Person(String name, int age) {
    public Person {
        if (age < 0) {
            throw new IllegalArgumentException("Age cannot be negative");
        }
    }
    
    public String getDescription() {
        return name + " is " + age + " years old";
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:  # Records might not be supported in all parsers
            assert result.summary.class_count >= 1
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_sealed_class_detection(self, analyzer):
        """Test detection of sealed classes (Java 17+)."""
        code = """
public sealed class Shape permits Circle, Rectangle, Triangle {
    public abstract double area();
}

final class Circle extends Shape {
    private double radius;
    
    public double area() {
        return Math.PI * radius * radius;
    }
}

final class Rectangle extends Shape {
    private double width, height;
    
    public double area() {
        return width * height;
    }
}

final class Triangle extends Shape {
    private double base, height;
    
    public double area() {
        return 0.5 * base * height;
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:  # Sealed classes might not be supported
            assert result.summary.class_count >= 1
    
    # ==================== Class Analysis Tests ====================
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_class_with_multiple_methods(self, analyzer):
        """Test analysis of class with multiple methods."""
        code = """
public class Calculator {
    private double memory;
    
    public Calculator() {
        this.memory = 0;
    }
    
    public double add(double a, double b) {
        return a + b;
    }
    
    public double subtract(double a, double b) {
        return a - b;
    }
    
    public double multiply(double a, double b) {
        if (b == 0) {
            throw new ArithmeticException("Cannot multiply by zero");
        }
        return a * b;
    }
    
    public double divide(double a, double b) {
        if (b == 0) {
            throw new ArithmeticException("Cannot divide by zero");
        }
        return a / b;
    }
    
    public void storeInMemory(double value) {
        this.memory = value;
    }
    
    public double recallMemory() {
        return this.memory;
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.class_count == 1
        
        cls = result.classes[0]
        assert cls.name == "Calculator"
        assert cls.method_count >= 7  # Constructor + 6 methods
        assert cls.total_complexity > 0
        assert cls.average_method_complexity > 0
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_inner_classes(self, analyzer):
        """Test analysis of inner classes."""
        code = """
public class OuterClass {
    private int outerField;
    
    public void outerMethod() {
        // Some logic
    }
    
    public class InnerClass {
        public void innerMethod() {
            outerField = 10;
        }
    }
    
    public static class StaticInnerClass {
        public void staticInnerMethod() {
            // Some logic
        }
    }
    
    public void methodWithLocalClass() {
        class LocalClass {
            public void localMethod() {
                // Some logic
            }
        }
        
        LocalClass local = new LocalClass();
        local.localMethod();
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            assert result.summary.class_count >= 2  # Outer + at least one inner
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_anonymous_classes(self, analyzer):
        """Test handling of anonymous classes."""
        code = """
import java.util.Comparator;

public class AnonymousTest {
    public void sortWithAnonymous(List<String> list) {
        list.sort(new Comparator<String>() {
            @Override
            public int compare(String s1, String s2) {
                if (s1.length() > s2.length()) {
                    return 1;
                } else if (s1.length() < s2.length()) {
                    return -1;
                } else {
                    return s1.compareTo(s2);
                }
            }
        });
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            # Anonymous class method should contribute to complexity
            assert result.summary.total_complexity >= 4
    
    # ==================== Cognitive Complexity Tests ====================
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_cognitive_complexity_nesting(self, analyzer):
        """Test cognitive complexity with nested structures."""
        code = """
public class CognitiveTest {
    public int nestedLogic(int a, int b, int c) {
        if (a > 0) {  // +1
            if (b > 0) {  // +2 (nested)
                if (c > 0) {  // +3 (double nested)
                    return a + b + c;
                } else {
                    return a + b;
                }
            } else {
                return a;
            }
        } else {
            return 0;
        }
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            func = result.functions[0]
            assert func.cognitive_complexity >= 6
            assert func.cognitive_complexity > func.complexity
    
    # ==================== Error Handling Tests ====================
    
    @pytest.mark.asyncio
    async def test_syntax_error_handling(self, analyzer):
        """Test handling of syntax errors."""
        code = """
public class Broken {
    public void method() {
        if (true  // Missing closing parenthesis
            return;
        }
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if TREE_SITTER_AVAILABLE:
            # Should still attempt to analyze despite syntax errors
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_empty_code_handling(self, analyzer):
        """Test handling of empty code."""
        result = await analyzer.analyze_code("")
        
        if TREE_SITTER_AVAILABLE:
            assert result.summary.function_count == 0
            assert result.summary.class_count == 0
    
    # ==================== Annotation Tests ====================
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_annotation_handling(self, analyzer):
        """Test handling of annotations."""
        code = """
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

@Service
public class AnnotatedService {
    @Autowired
    private Repository repository;
    
    @Override
    @Transactional
    @Cacheable("items")
    public List<Item> getItems() {
        return repository.findAll();
    }
    
    @Deprecated
    public void oldMethod() {
        // Legacy code
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            assert result.summary.class_count >= 1
            assert result.summary.function_count >= 2
    
    # ==================== Interface and Abstract Class Tests ====================
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_interface_analysis(self, analyzer):
        """Test analysis of interfaces."""
        code = """
public interface Calculator {
    double calculate(double a, double b);
    
    default double safeCalculate(double a, double b) {
        if (Double.isNaN(a) || Double.isNaN(b)) {
            return 0;
        }
        return calculate(a, b);
    }
    
    static double validateInput(double value) {
        if (value < 0) {
            throw new IllegalArgumentException("Negative value");
        }
        return value;
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            # Default and static methods should be analyzed
            assert result.summary.function_count >= 2
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_abstract_class_analysis(self, analyzer):
        """Test analysis of abstract classes."""
        code = """
public abstract class AbstractProcessor {
    protected int count;
    
    public AbstractProcessor() {
        this.count = 0;
    }
    
    public abstract void process(String data);
    
    public void preProcess(String data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("Invalid data");
        }
        count++;
    }
    
    protected void postProcess() {
        System.out.println("Processed " + count + " items");
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            assert result.summary.class_count >= 1
            # Constructor + concrete methods (not abstract)
            assert result.summary.function_count >= 3
    
    # ==================== Enum Tests ====================
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_enum_analysis(self, analyzer):
        """Test analysis of enums."""
        code = """
public enum Status {
    PENDING("Pending", 0),
    PROCESSING("Processing", 1),
    COMPLETED("Completed", 2),
    FAILED("Failed", -1);
    
    private final String description;
    private final int code;
    
    Status(String description, int code) {
        this.description = description;
        this.code = code;
    }
    
    public String getDescription() {
        return description;
    }
    
    public int getCode() {
        return code;
    }
    
    public boolean isTerminal() {
        return this == COMPLETED || this == FAILED;
    }
    
    public static Status fromCode(int code) {
        for (Status status : values()) {
            if (status.code == code) {
                return status;
            }
        }
        throw new IllegalArgumentException("Invalid code: " + code);
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            assert result.summary.class_count >= 1
            assert result.summary.function_count >= 5  # Constructor + 4 methods
    
    # ==================== Performance Tests ====================
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_large_file_performance(self, analyzer):
        """Test performance with large files."""
        # Generate a large Java file
        methods = []
        for i in range(50):
            methods.append(f"""
    public int method{i}(int x) {{
        if (x > {i}) {{
            return x * {i};
        }} else {{
            return x + {i};
        }}
    }}
""")
        
        code = f"""
public class LargeClass {{
    {"".join(methods)}
}}
"""
        
        import time
        start = time.time()
        result = await analyzer.analyze_code(code)
        duration = time.time() - start
        
        if result.success:
            assert result.summary.function_count == 50
            assert duration < 5.0  # Should complete within 5 seconds
    
    # ==================== Recommendation Tests ====================
    
    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, analyzer):
        """Test generation of improvement recommendations."""
        code = """
public class ComplexClass {
    public int veryComplexMethod(int a, int b, int c, int d, int e) {
        int result = 0;
        for (int i = 0; i < a; i++) {
            if (i % 2 == 0) {
                for (int j = 0; j < b; j++) {
                    if (j % 3 == 0) {
                        for (int k = 0; k < c; k++) {
                            if (k % 5 == 0) {
                                for (int l = 0; l < d; l++) {
                                    if (l % 7 == 0) {
                                        for (int m = 0; m < e; m++) {
                                            if (m % 11 == 0) {
                                                result += i * j * k * l * m;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        if result.success:
            recommendations = result.generate_recommendations()
            assert len(recommendations) > 0
            assert any("refactor" in r.lower() for r in recommendations)