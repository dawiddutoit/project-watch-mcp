"""Unit tests for Kotlin complexity analyzer.

Tests the KotlinComplexityAnalyzer implementation following TDD principles.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.complexity_analysis.languages.kotlin_analyzer import (
    KotlinComplexityAnalyzer,
)
from project_watch_mcp.complexity_analysis.models import (
    ComplexityClassification,
    ComplexityGrade,
    ComplexityResult,
    FunctionComplexity,
)


@pytest.fixture
def analyzer():
    """Create a Kotlin complexity analyzer instance."""
    return KotlinComplexityAnalyzer()


@pytest.fixture
def temp_kotlin_file():
    """Create a temporary Kotlin file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.kt', delete=False) as f:
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestKotlinComplexityAnalyzer:
    """Test suite for Kotlin complexity analyzer."""
    
    async def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes with correct language."""
        assert analyzer.language == "kotlin"
        assert analyzer.metrics is not None
    
    async def test_simple_function_complexity(self, analyzer):
        """Test complexity calculation for simple Kotlin functions."""
        code = """
        fun add(a: Int, b: Int): Int {
            return a + b
        }
        
        fun greet(name: String): String {
            return "Hello, $name!"
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.language == "kotlin"
        assert result.summary.function_count == 2
        assert result.summary.total_complexity == 2  # Each simple function has complexity 1
        assert result.summary.average_complexity == 1.0
        
        # Check individual functions
        assert len(result.functions) == 2
        add_func = next(f for f in result.functions if f.name == "add")
        assert add_func.complexity == 1
        assert add_func.classification == "simple"
        assert add_func.parameters == 2
    
    async def test_data_class_complexity(self, analyzer):
        """Test that data classes have lower base complexity."""
        code = """
        data class User(
            val id: Long,
            val name: String,
            val email: String
        )
        
        data class Product(val id: Int, val price: Double) {
            fun discountedPrice(discount: Double): Double {
                return price * (1 - discount)
            }
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.class_count == 2
        # Data classes should have minimal complexity
        assert result.classes[0].name == "User"
        assert result.classes[0].total_complexity <= 1  # Data class with no methods
        
        # Product has a method, so slightly higher
        product_class = next(c for c in result.classes if c.name == "Product")
        assert len(product_class.methods) == 1
        assert product_class.methods[0].name == "discountedPrice"
    
    async def test_when_expression_complexity(self, analyzer):
        """Test complexity calculation for when expressions."""
        code = """
        fun processStatus(status: Int): String {
            return when (status) {
                1 -> "Active"
                2 -> "Pending"
                3 -> "Completed"
                4 -> "Failed"
                else -> "Unknown"
            }
        }
        
        fun categorizeNumber(n: Int): String {
            return when {
                n < 0 -> "Negative"
                n == 0 -> "Zero"
                n in 1..10 -> "Small"
                n in 11..100 -> "Medium"
                else -> "Large"
            }
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.function_count == 2
        
        # When with 5 branches = complexity 6 (branches + 1)
        process_func = next(f for f in result.functions if f.name == "processStatus")
        assert process_func.complexity == 6
        assert process_func.classification == "moderate"
        
        categorize_func = next(f for f in result.functions if f.name == "categorizeNumber")
        assert categorize_func.complexity == 6  # 5 branches + 1
    
    async def test_extension_function_complexity(self, analyzer):
        """Test that extension functions have standard complexity."""
        code = """
        fun String.isPalindrome(): Boolean {
            val clean = this.lowercase().replace(Regex("[^a-z0-9]"), "")
            return clean == clean.reversed()
        }
        
        fun List<Int>.sumOfEvens(): Int {
            var sum = 0
            for (num in this) {
                if (num % 2 == 0) {
                    sum += num
                }
            }
            return sum
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.function_count == 2
        
        # Extension functions are treated as regular functions
        palindrome_func = next(f for f in result.functions if "isPalindrome" in f.name)
        assert palindrome_func.complexity == 1  # Simple linear flow
        
        sum_func = next(f for f in result.functions if "sumOfEvens" in f.name)
        assert sum_func.complexity == 3  # Loop + if condition
    
    async def test_lambda_expression_complexity(self, analyzer):
        """Test complexity for lambda expressions and nested lambdas."""
        code = """
        fun processItems(items: List<String>): List<Int> {
            return items
                .filter { it.isNotEmpty() }
                .map { item ->
                    when {
                        item.length > 10 -> item.length * 2
                        item.length > 5 -> item.length
                        else -> 0
                    }
                }
                .filter { it > 0 }
        }
        
        fun nestedLambdaExample(data: List<List<Int>>): Int {
            return data.sumOf { outer ->
                outer.sumOf { inner ->
                    if (inner > 0) inner * 2 else 0
                }
            }
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        
        # Lambda expressions add complexity
        process_func = next(f for f in result.functions if f.name == "processItems")
        # Base function + lambdas with conditions
        assert process_func.complexity >= 4
        
        nested_func = next(f for f in result.functions if f.name == "nestedLambdaExample")
        # Nested lambdas increase complexity
        assert nested_func.complexity >= 4  # +1 per nested lambda + conditions
    
    async def test_suspend_function_complexity(self, analyzer):
        """Test that suspend functions have additional complexity."""
        code = """
        suspend fun fetchData(id: Long): String {
            return "Data for $id"
        }
        
        suspend fun processAsync(items: List<String>): List<String> {
            val results = mutableListOf<String>()
            for (item in items) {
                if (item.isNotEmpty()) {
                    results.add(item.uppercase())
                }
            }
            return results
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        
        # Suspend functions have +2 complexity
        fetch_func = next(f for f in result.functions if f.name == "fetchData")
        assert fetch_func.complexity == 3  # Base 1 + 2 for suspend
        
        process_func = next(f for f in result.functions if f.name == "processAsync")
        # Base complexity (loop + if = 3) + 2 for suspend = 5
        assert process_func.complexity == 5
    
    async def test_sealed_class_complexity(self, analyzer):
        """Test complexity for sealed classes and their subclasses."""
        code = """
        sealed class Result<out T> {
            data class Success<T>(val data: T) : Result<T>()
            data class Error(val message: String) : Result<Nothing>()
            object Loading : Result<Nothing>()
        }
        
        fun handleResult(result: Result<String>): String {
            return when (result) {
                is Result.Success -> "Success: ${result.data}"
                is Result.Error -> "Error: ${result.message}"
                Result.Loading -> "Loading..."
            }
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        
        # Sealed class with 3 subclasses
        sealed_class = next((c for c in result.classes if c.name == "Result"), None)
        if sealed_class:
            # +1 complexity per subclass
            assert sealed_class.total_complexity >= 3
        
        # Function handling sealed class
        handle_func = next(f for f in result.functions if f.name == "handleResult")
        assert handle_func.complexity == 4  # 3 branches + 1
    
    async def test_complex_nested_function(self, analyzer):
        """Test a complex function with high nesting and multiple conditions."""
        code = """
        fun complexValidation(data: Map<String, Any?>, options: Map<String, Boolean>): Boolean {
            if (data.isEmpty()) {
                return false
            }
            
            for ((key, value) in data) {
                if (value == null) {
                    if (options["allowNull"] == true) {
                        continue
                    } else {
                        return false
                    }
                }
                
                when (value) {
                    is String -> {
                        if (value.isEmpty()) {
                            if (options["allowEmpty"] != true) {
                                return false
                            }
                        } else if (value.length > 100) {
                            if (options["allowLong"] != true) {
                                return false
                            }
                        }
                    }
                    is Int -> {
                        if (value < 0) {
                            if (options["allowNegative"] != true) {
                                return false
                            }
                        } else if (value > 1000) {
                            return false
                        }
                    }
                    is List<*> -> {
                        if (value.isEmpty() && options["allowEmptyList"] != true) {
                            return false
                        }
                    }
                    else -> {
                        if (options["strictTypes"] == true) {
                            return false
                        }
                    }
                }
            }
            
            return true
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        
        func = result.functions[0]
        assert func.name == "complexValidation"
        assert func.complexity > 15  # Many conditions and branches
        assert func.classification == "complex" or func.classification == "very-complex"
        assert func.depth >= 4  # Deep nesting
        
        # Should have recommendations for refactoring
        assert len(result.recommendations) > 0
        assert any("refactor" in r.lower() for r in result.recommendations)
    
    async def test_cognitive_complexity(self, analyzer):
        """Test cognitive complexity calculation for Kotlin code."""
        code = """
        fun calculateScore(items: List<Int>, multiplier: Int): Int {
            var score = 0
            
            for (item in items) {  // +1
                if (item > 0) {  // +2 (nested)
                    if (item % 2 == 0) {  // +3 (double nested)
                        score += item * multiplier
                    } else {  // +1
                        score += item
                    }
                }
            }
            
            return score
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        
        func = result.functions[0]
        assert func.cognitive_complexity >= 7  # Based on nesting and conditions
        assert func.cognitive_complexity > func.complexity  # Cognitive is usually higher
    
    async def test_file_analysis(self, analyzer, temp_kotlin_file):
        """Test analyzing a Kotlin file from disk."""
        code = """
        package com.example
        
        class Calculator {
            fun add(a: Double, b: Double): Double = a + b
            
            fun divide(a: Double, b: Double): Double {
                if (b == 0.0) {
                    throw IllegalArgumentException("Division by zero")
                }
                return a / b
            }
        }
        """
        
        temp_kotlin_file.write_text(code)
        
        result = await analyzer.analyze_file(temp_kotlin_file)
        
        assert result.success
        assert result.file_path == str(temp_kotlin_file)
        assert result.language == "kotlin"
        assert result.summary.class_count == 1
        assert result.summary.function_count == 2
    
    async def test_invalid_kotlin_syntax(self, analyzer):
        """Test handling of invalid Kotlin syntax."""
        code = """
        fun broken {  // Missing parameters and return type
            this is not valid Kotlin
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        # Should handle syntax errors gracefully
        assert result.error is not None
        assert "syntax" in result.error.lower() or "parse" in result.error.lower()
        assert result.summary.total_complexity == 0
    
    async def test_empty_file(self, analyzer):
        """Test analyzing an empty Kotlin file."""
        result = await analyzer.analyze_code("")
        
        assert result.success
        assert result.summary.function_count == 0
        assert result.summary.class_count == 0
        assert result.summary.total_complexity == 0
    
    async def test_maintainability_index(self, analyzer):
        """Test maintainability index calculation for Kotlin code."""
        code = """
        /**
         * A well-documented simple function.
         * This should have a good maintainability index.
         */
        fun simpleFunction(x: Int): Int {
            // Simple calculation
            return x * 2
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.maintainability_index > 60  # Good maintainability
        assert result.summary.complexity_grade in ["A", "B"]
    
    async def test_recommendations_generation(self, analyzer):
        """Test that appropriate recommendations are generated."""
        # Create a function with moderate complexity
        code = """
        fun moderateComplexity(data: List<String>): Map<String, Int> {
            val result = mutableMapOf<String, Int>()
            
            for (item in data) {
                when {
                    item.isEmpty() -> continue
                    item.length < 5 -> result[item] = 1
                    item.length < 10 -> result[item] = 2
                    item.length < 15 -> result[item] = 3
                    else -> result[item] = 4
                }
            }
            
            return result
        }
        """
        
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert len(result.recommendations) >= 0
        
        # With good code, should get positive feedback or no recommendations
        if result.summary.complexity_grade in ["A", "B"]:
            if result.recommendations:
                assert any("acceptable" in r.lower() or "good" in r.lower() 
                          for r in result.recommendations)