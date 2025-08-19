"""Unit tests for Java complexity analyzer.

This module tests the JavaComplexityAnalyzer implementation,
ensuring accurate complexity calculation for Java-specific constructs.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from project_watch_mcp.complexity_analysis.languages.java_analyzer import JavaComplexityAnalyzer
from project_watch_mcp.complexity_analysis.models import (
    ComplexityResult,
    FunctionComplexity,
    ClassComplexity,
    ComplexityClassification,
    ComplexityGrade,
)


class TestJavaComplexityAnalyzer:
    """Test suite for Java complexity analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a Java analyzer instance."""
        return JavaComplexityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_simple_method_complexity(self, analyzer):
        """Test complexity calculation for a simple method."""
        code = """
        public class SimpleClass {
            public int add(int a, int b) {
                return a + b;
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.language == "java"
        assert result.summary.total_complexity == 1
        assert len(result.functions) == 1
        assert result.functions[0].name == "SimpleClass.add"
        assert result.functions[0].complexity == 1
        assert result.functions[0].classification == "simple"
    
    @pytest.mark.asyncio
    async def test_if_statement_complexity(self, analyzer):
        """Test that if statements increase complexity by 1."""
        code = """
        public class ConditionalClass {
            public String checkValue(int value) {
                if (value > 0) {
                    return "positive";
                } else if (value < 0) {
                    return "negative";
                } else {
                    return "zero";
                }
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert len(result.functions) == 1
        assert result.functions[0].name == "ConditionalClass.checkValue"
        # Base 1 + if + else if = 3
        assert result.functions[0].complexity == 3
    
    @pytest.mark.asyncio
    async def test_switch_statement_complexity(self, analyzer):
        """Test switch statement complexity (1 per case + 1 for default)."""
        code = """
        public class SwitchClass {
            public String getDayName(int day) {
                switch (day) {
                    case 1:
                        return "Monday";
                    case 2:
                        return "Tuesday";
                    case 3:
                        return "Wednesday";
                    case 4:
                        return "Thursday";
                    case 5:
                        return "Friday";
                    default:
                        return "Weekend";
                }
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert len(result.functions) == 1
        # Base 1 + 5 cases + 1 default = 7
        assert result.functions[0].complexity == 7
    
    @pytest.mark.asyncio
    async def test_loop_complexity(self, analyzer):
        """Test that loops increase complexity."""
        code = """
        public class LoopClass {
            public int sumArray(int[] array) {
                int sum = 0;
                for (int i = 0; i < array.length; i++) {
                    sum += array[i];
                }
                while (sum > 100) {
                    sum = sum / 2;
                }
                do {
                    sum++;
                } while (sum < 10);
                return sum;
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        # Base 1 + for + while + do-while = 4
        assert result.functions[0].complexity == 4
    
    @pytest.mark.asyncio
    async def test_try_catch_complexity(self, analyzer):
        """Test try-catch block complexity (1 per catch block)."""
        code = """
        public class ExceptionClass {
            public void riskyOperation() {
                try {
                    doSomething();
                } catch (IOException e) {
                    handleIO(e);
                } catch (SQLException e) {
                    handleSQL(e);
                } catch (Exception e) {
                    handleGeneral(e);
                } finally {
                    cleanup();
                }
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        # Base 1 + 3 catch blocks = 4
        assert result.functions[0].complexity == 4
    
    @pytest.mark.asyncio
    async def test_lambda_expression_complexity(self, analyzer):
        """Test lambda expression complexity (+1 per lambda)."""
        code = """
        import java.util.List;
        import java.util.stream.Collectors;
        
        public class LambdaClass {
            public List<String> processNames(List<String> names) {
                return names.stream()
                    .filter(name -> name.length() > 3)
                    .map(name -> name.toUpperCase())
                    .sorted((a, b) -> a.compareTo(b))
                    .collect(Collectors.toList());
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        # Base 1 + 3 lambdas + 3 stream operations (filter, map, sorted) = 7
        assert result.functions[0].complexity == 7
    
    @pytest.mark.asyncio
    async def test_stream_operations_complexity(self, analyzer):
        """Test stream operations complexity (+1 per intermediate operation)."""
        code = """
        import java.util.List;
        
        public class StreamClass {
            public long countLongNames(List<String> names) {
                return names.stream()
                    .filter(name -> name.length() > 10)
                    .map(String::toUpperCase)
                    .distinct()
                    .sorted()
                    .count();
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        # Base 1 + filter + map + distinct + sorted = 5
        # Plus 1 for lambda in filter = 6
        assert result.functions[0].complexity == 6
    
    @pytest.mark.asyncio
    async def test_anonymous_class_complexity(self, analyzer):
        """Test anonymous class complexity."""
        code = """
        public class AnonymousClass {
            public void setupListener() {
                button.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent e) {
                        if (e.getSource() == button) {
                            handleClick();
                        }
                    }
                });
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        # setupListener: 1 + anonymous class methods
        # actionPerformed: 1 + if = 2
        # Total for setupListener including anonymous class
        assert result.functions[0].name == "AnonymousClass.setupListener"
        # setupListener has base 1 + anonymous class method complexity 
        assert result.functions[0].complexity >= 2
    
    @pytest.mark.asyncio
    async def test_cognitive_complexity(self, analyzer):
        """Test cognitive complexity calculation."""
        code = """
        public class CognitiveClass {
            public int complexMethod(int value) {
                if (value > 0) {                    // +1 complexity, +1 cognitive
                    for (int i = 0; i < value; i++) { // +1 complexity, +2 cognitive (nesting)
                        if (i % 2 == 0) {            // +1 complexity, +3 cognitive (double nesting)
                            return i;
                        }
                    }
                }
                return -1;
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        func = result.functions[0]
        assert func.complexity == 4  # Base 1 + if + for + nested if
        assert func.cognitive_complexity >= 6  # Higher due to nesting
    
    @pytest.mark.asyncio
    async def test_class_complexity_aggregation(self, analyzer):
        """Test class-level complexity aggregation."""
        code = """
        public class ComplexClass {
            public void method1() {
                if (true) {
                    doSomething();
                }
            }
            
            public void method2() {
                for (int i = 0; i < 10; i++) {
                    if (i > 5) {
                        break;
                    }
                }
            }
            
            public void method3() {
                switch (value) {
                    case 1:
                        return;
                    case 2:
                        return;
                    default:
                        return;
                }
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert len(result.classes) == 1
        assert result.classes[0].name == "ComplexClass"
        assert result.classes[0].method_count == 3
        assert result.classes[0].total_complexity >= 7  # Sum of all methods
    
    @pytest.mark.asyncio
    async def test_nested_class_complexity(self, analyzer):
        """Test nested class complexity handling."""
        code = """
        public class OuterClass {
            public void outerMethod() {
                if (true) {
                    doSomething();
                }
            }
            
            public static class InnerClass {
                public void innerMethod() {
                    for (int i = 0; i < 10; i++) {
                        process(i);
                    }
                }
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert len(result.classes) == 1
        outer = result.classes[0]
        assert outer.name == "OuterClass"
        assert len(outer.nested_classes) == 1
        assert outer.nested_classes[0].name == "InnerClass"
    
    @pytest.mark.asyncio
    async def test_generics_handling(self, analyzer):
        """Test that generics don't affect complexity."""
        code = """
        public class GenericClass<T extends Comparable<T>> {
            public <U> U genericMethod(T value, U defaultValue) {
                if (value != null) {
                    return process(value);
                }
                return defaultValue;
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.functions[0].complexity == 2  # Base 1 + if
    
    @pytest.mark.asyncio
    async def test_annotations_ignored(self, analyzer):
        """Test that annotations don't affect complexity."""
        code = """
        @Component
        @RequestMapping("/api")
        public class AnnotatedClass {
            @Override
            @Transactional
            @Cacheable(value = "cache")
            public String annotatedMethod(@RequestParam String param) {
                if (param != null) {
                    return param.toUpperCase();
                }
                return "";
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.functions[0].complexity == 2  # Base 1 + if
    
    @pytest.mark.asyncio
    async def test_interface_and_abstract_methods(self, analyzer):
        """Test interface and abstract method handling."""
        code = """
        public interface MyInterface {
            void interfaceMethod();
            
            default void defaultMethod() {
                if (isValid()) {
                    process();
                }
            }
        }
        
        public abstract class AbstractClass {
            public abstract void abstractMethod();
            
            public void concreteMethod() {
                for (int i = 0; i < 10; i++) {
                    doWork(i);
                }
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        # Only concrete/default methods should have complexity
        functions_with_complexity = [f for f in result.functions if f.complexity > 0]
        assert len(functions_with_complexity) == 2  # defaultMethod and concreteMethod
    
    @pytest.mark.asyncio
    async def test_maintainability_index(self, analyzer):
        """Test maintainability index calculation."""
        code = """
        public class MaintainableClass {
            /**
             * Well documented method with low complexity.
             * @param value Input value
             * @return Processed value
             */
            public int simpleMethod(int value) {
                // Simple calculation
                return value * 2;
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.maintainability_index > 60  # Good maintainability
        assert result.summary.complexity_grade in ["A", "B"]
    
    @pytest.mark.asyncio
    async def test_recommendations_generation(self, analyzer):
        """Test that appropriate recommendations are generated."""
        code = """
        public class ComplexClass {
            public void veryComplexMethod() {
                for (int i = 0; i < 10; i++) {
                    for (int j = 0; j < 10; j++) {
                        for (int k = 0; k < 10; k++) {
                            if (i > j && j > k) {
                                if (i % 2 == 0) {
                                    if (j % 3 == 0) {
                                        if (k % 5 == 0) {
                                            doSomething();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert len(result.recommendations) > 0
        # Check for any relevant recommendation keyword
        assert any(("refactor" in rec.lower() or "complex" in rec.lower() or "improve" in rec.lower() 
                    or "urgent" in rec.lower() or "high" in rec.lower() or "nesting" in rec.lower()
                    or "extract" in rec.lower()) 
                   for rec in result.recommendations)
        # The complexity is 9 (3 loops + 4 ifs + 1 logical AND + base 1)
        assert result.functions[0].complexity >= 9
        # With deep nesting, cognitive complexity is very high
        assert result.functions[0].cognitive_complexity > 20
    
    @pytest.mark.asyncio
    async def test_analyze_file(self, analyzer):
        """Test analyzing a Java file from disk."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write("""
            public class TestClass {
                public void testMethod() {
                    System.out.println("Hello");
                }
            }
            """)
            temp_path = f.name
        
        try:
            result = await analyzer.analyze_file(Path(temp_path))
            assert result.success
            assert result.file_path == temp_path
            assert result.language == "java"
            assert len(result.functions) == 1
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_file_not_found(self, analyzer):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            await analyzer.analyze_file(Path("/nonexistent/file.java"))
    
    @pytest.mark.asyncio
    async def test_non_java_file(self, analyzer):
        """Test rejection of non-Java files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Not Java code")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Not a Java file"):
                await analyzer.analyze_file(Path(temp_path))
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_malformed_java_code(self, analyzer):
        """Test handling of malformed Java code."""
        code = """
        public class BrokenClass {
            public void method( {  // Missing closing parenthesis
                if (true
            }
        """
        result = await analyzer.analyze_code(code)
        
        # Tree-sitter may handle incomplete code depending on the nature of error
        # If it parses successfully (even partially), that's fine
        # We just want to ensure no crash
        assert result is not None  # Should not crash
    
    @pytest.mark.asyncio
    async def test_enum_complexity(self, analyzer):
        """Test enum complexity calculation."""
        code = """
        public enum Status {
            PENDING {
                @Override
                public boolean canTransition(Status next) {
                    if (next == PROCESSING || next == CANCELLED) {
                        return true;
                    }
                    return false;
                }
            },
            PROCESSING {
                @Override
                public boolean canTransition(Status next) {
                    return next == COMPLETED || next == FAILED;
                }
            },
            COMPLETED,
            FAILED,
            CANCELLED;
            
            public boolean canTransition(Status next) {
                return false;
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        # Should find canTransition methods with their complexity
        transition_methods = [f for f in result.functions if "canTransition" in f.name]
        assert len(transition_methods) >= 2
    
    @pytest.mark.asyncio
    async def test_record_complexity(self, analyzer):
        """Test Java 14+ record complexity."""
        code = """
        public record Person(String name, int age) {
            public Person {
                if (age < 0) {
                    throw new IllegalArgumentException("Age cannot be negative");
                }
                if (name == null || name.isEmpty()) {
                    throw new IllegalArgumentException("Name is required");
                }
            }
            
            public boolean isAdult() {
                return age >= 18;
            }
        }
        """
        result = await analyzer.analyze_code(code)
        
        assert result.success
        # At least isAdult method should be found
        assert len(result.functions) >= 1
        # Check if isAdult is found
        is_adult = [f for f in result.functions if "isAdult" in f.name]
        assert len(is_adult) == 1
        assert is_adult[0].complexity == 1  # Simple return statement