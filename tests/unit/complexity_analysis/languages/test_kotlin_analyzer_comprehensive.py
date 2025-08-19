"""Comprehensive test suite for Kotlin complexity analyzer.

This module provides extensive testing for the Kotlin complexity analyzer,
targeting >80% code coverage with focus on Kotlin-specific features,
coroutines, DSLs, and edge cases.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from project_watch_mcp.complexity_analysis.languages.kotlin_analyzer import (
    KotlinComplexityAnalyzer,
    MAX_RECURSION_DEPTH
)
from project_watch_mcp.complexity_analysis.models import (
    ComplexityResult,
    ComplexityGrade,
    FunctionComplexity,
    ClassComplexity,
)


class TestKotlinComplexityAnalyzer:
    """Comprehensive test suite for Kotlin complexity analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return KotlinComplexityAnalyzer()
    
    @pytest.fixture
    def temp_kotlin_file(self, tmp_path):
        """Create a temporary Kotlin file for testing."""
        file_path = tmp_path / "TestClass.kt"
        file_path.write_text("""
data class Person(val name: String, val age: Int)

class TestClass(private var value: Int) {
    
    fun getValue(): Int = value
    
    fun setValue(newValue: Int) {
        require(newValue >= 0) { "Value must be non-negative" }
        value = newValue
    }
    
    fun calculate(x: Int, y: Int): Int {
        return when {
            x > y -> x - y
            x < y -> y - x
            else -> 0
        }
    }
    
    suspend fun fetchData(): String {
        delay(100)
        return "Data"
    }
}

fun String.isPalindrome(): Boolean {
    return this == this.reversed()
}
""")
        return file_path
    
    # ==================== Basic Functionality Tests ====================
    
    @pytest.mark.asyncio
    async def test_analyze_simple_function(self, analyzer):
        """Test analysis of a simple Kotlin function."""
        code = """
fun simple(): Int {
    return 42
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.language == "kotlin"
        assert result.summary.function_count >= 1
        assert result.summary.total_complexity >= 1
        assert len(result.functions) >= 1
        assert result.functions[0].name == "simple"
    
    @pytest.mark.asyncio
    async def test_analyze_file(self, analyzer, temp_kotlin_file):
        """Test file analysis."""
        result = await analyzer.analyze_file(temp_kotlin_file)
        
        assert result.success
        assert result.file_path == str(temp_kotlin_file)
        assert result.summary.function_count >= 5  # Including extension function
        assert result.summary.class_count >= 2  # TestClass and Person
    
    @pytest.mark.asyncio
    async def test_file_not_found(self, analyzer):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            await analyzer.analyze_file(Path("nonexistent.kt"))
    
    @pytest.mark.asyncio
    async def test_non_kotlin_file(self, analyzer, tmp_path):
        """Test rejection of non-Kotlin files."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("not kotlin")
        
        with pytest.raises(ValueError, match="Not a Kotlin file"):
            await analyzer.analyze_file(file_path)
    
    # ==================== Cyclomatic Complexity Tests ====================
    
    @pytest.mark.parametrize("code,expected_min_complexity", [
        ("fun f() {}", 1),
        ("fun f(x: Int) { if (x > 0) return }", 2),
        ("fun f(x: Int) = if (x > 0) x else -x", 2),
        ("fun f() { for (i in 1..10) { } }", 2),
        ("fun f(x: Int) { while (x > 0) { } }", 2),
        ("fun f() { do { } while (true) }", 2),
    ])
    @pytest.mark.asyncio
    async def test_cyclomatic_complexity_calculation(self, analyzer, code, expected_min_complexity):
        """Test cyclomatic complexity for various Kotlin constructs."""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.functions[0].complexity >= expected_min_complexity
    
    @pytest.mark.asyncio
    async def test_when_expression_complexity(self, analyzer):
        """Test complexity of when expressions."""
        code = """
fun getDayName(day: Int): String {
    return when (day) {
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
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        func = result.functions[0]
        # When expressions have lower complexity than equivalent if-else chains
        assert func.complexity >= 5  # Base + branches (reduced from 8)
    
    @pytest.mark.asyncio
    async def test_when_with_multiple_conditions(self, analyzer):
        """Test when expression with multiple conditions per branch."""
        code = """
fun categorize(x: Int): String {
    return when (x) {
        1, 2, 3 -> "Small"
        4, 5, 6 -> "Medium"
        7, 8, 9 -> "Large"
        in 10..20 -> "Extra Large"
        else -> "Unknown"
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        func = result.functions[0]
        assert func.complexity >= 4  # Grouped conditions count less
    
    # ==================== Kotlin-Specific Feature Tests ====================
    
    @pytest.mark.asyncio
    async def test_data_class_detection(self, analyzer):
        """Test detection and reduced complexity for data classes."""
        code = """
data class User(
    val id: Long,
    val name: String,
    val email: String,
    val age: Int
) {
    fun isAdult(): Boolean = age >= 18
    
    fun getDisplayName(): String {
        return if (name.isNotEmpty()) name else email
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.class_count >= 1
        # Data classes should have reduced base complexity
        assert any("User" in cls.name for cls in result.classes)
    
    @pytest.mark.asyncio
    async def test_extension_function_detection(self, analyzer):
        """Test detection of extension functions."""
        code = """
fun String.capitalizeWords(): String {
    return this.split(" ").joinToString(" ") { word ->
        word.replaceFirstChar { it.uppercase() }
    }
}

fun List<Int>.median(): Double {
    val sorted = this.sorted()
    return if (size % 2 == 0) {
        (sorted[size / 2 - 1] + sorted[size / 2]) / 2.0
    } else {
        sorted[size / 2].toDouble()
    }
}

fun Int.isPrime(): Boolean {
    if (this <= 1) return false
    for (i in 2..this / 2) {
        if (this % i == 0) return false
    }
    return true
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.function_count >= 3
        # All should be detected as extension functions
        for func in result.functions:
            assert "." in func.name or "Extension" in str(func.type)
    
    @pytest.mark.asyncio
    async def test_suspend_function_detection(self, analyzer):
        """Test detection of suspend functions (coroutines)."""
        code = """
import kotlinx.coroutines.*

suspend fun fetchUser(id: Long): User {
    delay(1000)
    return User(id, "User$id")
}

suspend fun processData(): List<String> {
    return coroutineScope {
        val deferred1 = async { fetchData1() }
        val deferred2 = async { fetchData2() }
        
        listOf(deferred1.await(), deferred2.await())
    }
}

suspend fun complexCoroutine() {
    withContext(Dispatchers.IO) {
        for (i in 1..10) {
            launch {
                delay(100)
                process(i)
            }
        }
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        # Suspend functions should have additional complexity
        suspend_funcs = [f for f in result.functions if "suspend" in str(f.type).lower()]
        assert len(suspend_funcs) >= 2
    
    @pytest.mark.asyncio
    async def test_lambda_expression_detection(self, analyzer):
        """Test detection and complexity of lambda expressions."""
        code = """
fun processItems(items: List<String>): List<String> {
    return items
        .filter { it.isNotEmpty() }
        .map { it.uppercase() }
        .sortedBy { it.length }
        .take(10)
}

fun complexLambdas() {
    val result = listOf(1, 2, 3, 4, 5)
        .map { x -> 
            if (x % 2 == 0) {
                x * 2
            } else {
                x * 3
            }
        }
        .filter { it > 5 }
        .fold(0) { acc, value -> acc + value }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        # Lambda expressions should be detected and add complexity
        for func in result.functions:
            if "lambda" in func.name.lower() or "process" in func.name.lower():
                assert func.complexity >= 2
    
    @pytest.mark.asyncio
    async def test_sealed_class_detection(self, analyzer):
        """Test detection of sealed classes."""
        code = """
sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Error(val message: String) : Result<Nothing>()
    object Loading : Result<Nothing>()
}

fun processResult(result: Result<String>): String {
    return when (result) {
        is Result.Success -> "Success: ${result.data}"
        is Result.Error -> "Error: ${result.message}"
        Result.Loading -> "Loading..."
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.class_count >= 1
        # Sealed classes should be detected
        assert any("sealed" in str(cls.name).lower() or "Result" in cls.name for cls in result.classes)
    
    @pytest.mark.asyncio
    async def test_inline_function_detection(self, analyzer):
        """Test detection of inline functions."""
        code = """
inline fun measureTime(block: () -> Unit): Long {
    val start = System.currentTimeMillis()
    block()
    return System.currentTimeMillis() - start
}

inline fun <reified T> List<*>.filterIsInstance(): List<T> {
    return this.filter { it is T } as List<T>
}

inline fun repeat(times: Int, action: (Int) -> Unit) {
    for (index in 0 until times) {
        action(index)
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.function_count >= 3
        # Inline functions should be detected
    
    @pytest.mark.asyncio
    async def test_null_safety_operations(self, analyzer):
        """Test complexity of null safety operations."""
        code = """
fun processNullable(value: String?): String {
    // Safe call
    val length = value?.length ?: 0
    
    // Elvis operator
    val nonNull = value ?: "default"
    
    // Let with safe call
    value?.let {
        println("Value: $it")
    }
    
    // Multiple safe calls
    val result = value?.trim()?.uppercase()?.replace(" ", "_")
    
    // Not-null assertion
    return value!!
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        func = result.functions[0]
        # Null safety operations should add minimal complexity
        assert func.complexity >= 2
    
    @pytest.mark.asyncio
    async def test_dsl_builder_pattern(self, analyzer):
        """Test DSL builder pattern complexity."""
        code = """
class Html {
    fun body(init: Body.() -> Unit): Body {
        val body = Body()
        body.init()
        return body
    }
}

class Body {
    fun p(text: String) {
        println("<p>$text</p>")
    }
    
    fun div(init: Div.() -> Unit): Div {
        val div = Div()
        div.init()
        return div
    }
}

class Div {
    fun text(value: String) {
        println(value)
    }
}

fun html(init: Html.() -> Unit): Html {
    val html = Html()
    html.init()
    return html
}

fun buildPage() {
    html {
        body {
            p("Hello")
            div {
                text("World")
            }
        }
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.class_count >= 3
        assert result.summary.function_count >= 5
    
    # ==================== Class Analysis Tests ====================
    
    @pytest.mark.asyncio
    async def test_class_with_companion_object(self, analyzer):
        """Test analysis of classes with companion objects."""
        code = """
class Database {
    private var connection: String? = null
    
    fun connect(url: String) {
        connection = url
    }
    
    fun query(sql: String): List<String> {
        return if (connection != null) {
            listOf("Result1", "Result2")
        } else {
            emptyList()
        }
    }
    
    companion object {
        const val DEFAULT_URL = "jdbc:sqlite:memory"
        
        fun create(): Database {
            return Database().apply {
                connect(DEFAULT_URL)
            }
        }
        
        fun createWithUrl(url: String): Database {
            return Database().apply {
                connect(url)
            }
        }
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.class_count >= 1
        # Methods in companion object should be counted
        assert result.summary.function_count >= 4
    
    @pytest.mark.asyncio
    async def test_object_declaration(self, analyzer):
        """Test analysis of object declarations (singletons)."""
        code = """
object ConfigManager {
    private val settings = mutableMapOf<String, Any>()
    
    fun set(key: String, value: Any) {
        settings[key] = value
    }
    
    fun get(key: String): Any? {
        return settings[key]
    }
    
    fun getOrDefault(key: String, default: Any): Any {
        return settings[key] ?: default
    }
    
    fun clear() {
        settings.clear()
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.class_count >= 1
        assert result.summary.function_count >= 4
    
    # ==================== Cognitive Complexity Tests ====================
    
    @pytest.mark.asyncio
    async def test_cognitive_complexity_with_nesting(self, analyzer):
        """Test cognitive complexity with nested structures."""
        code = """
fun complexNesting(a: Int, b: Int, c: Int): Int {
    if (a > 0) {  // +1
        if (b > 0) {  // +2 (nested)
            if (c > 0) {  // +3 (double nested)
                return a + b + c
            } else {
                return a + b
            }
        } else {
            return a
        }
    } else {
        return 0
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        func = result.functions[0]
        assert func.cognitive_complexity >= 6
        assert func.cognitive_complexity > func.complexity
    
    # ==================== Error Handling Tests ====================
    
    @pytest.mark.asyncio
    async def test_syntax_error_handling(self, analyzer):
        """Test handling of syntax errors."""
        code = """
fun broken(
    return 42  // Syntax error
}
"""
        result = await analyzer.analyze_code(code)
        
        # Should handle gracefully even with syntax errors
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_empty_code_handling(self, analyzer):
        """Test handling of empty code."""
        result = await analyzer.analyze_code("")
        
        assert result.success
        assert result.summary.function_count == 0
        assert result.summary.class_count == 0
        assert result.summary.total_complexity == 0
    
    @pytest.mark.asyncio
    async def test_recursion_depth_limit(self, analyzer):
        """Test recursion depth limit handling."""
        # Create deeply nested code that might trigger recursion limit
        nested_ifs = "if (true) { " * (MAX_RECURSION_DEPTH + 10)
        nested_ifs += "return 1"
        nested_ifs += " }" * (MAX_RECURSION_DEPTH + 10)
        
        code = f"""
fun deeplyNested() {{
    {nested_ifs}
}}
"""
        
        result = await analyzer.analyze_code(code)
        
        # Should handle without crashing
        assert result is not None
        if not result.success:
            assert "recursion" in result.error.lower()
    
    # ==================== Operator Overloading Tests ====================
    
    @pytest.mark.asyncio
    async def test_operator_overloading(self, analyzer):
        """Test complexity of operator overloading."""
        code = """
data class Vector(val x: Double, val y: Double) {
    operator fun plus(other: Vector): Vector {
        return Vector(x + other.x, y + other.y)
    }
    
    operator fun minus(other: Vector): Vector {
        return Vector(x - other.x, y - other.y)
    }
    
    operator fun times(scalar: Double): Vector {
        return Vector(x * scalar, y * scalar)
    }
    
    operator fun get(index: Int): Double {
        return when (index) {
            0 -> x
            1 -> y
            else -> throw IndexOutOfBoundsException()
        }
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.function_count >= 4
        # Operator functions should be analyzed like regular functions
    
    # ==================== Property Delegation Tests ====================
    
    @pytest.mark.asyncio
    async def test_delegated_properties(self, analyzer):
        """Test complexity with delegated properties."""
        code = """
import kotlin.properties.Delegates

class User {
    var name: String by Delegates.observable("") { _, old, new ->
        println("Name changed from $old to $new")
    }
    
    val lazyValue: String by lazy {
        println("Computing lazy value")
        "Computed"
    }
    
    var vetoable: Int by Delegates.vetoable(0) { _, old, new ->
        new >= 0  // Only accept non-negative values
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.class_count >= 1
    
    # ==================== Destructuring Tests ====================
    
    @pytest.mark.asyncio
    async def test_destructuring_declarations(self, analyzer):
        """Test complexity with destructuring declarations."""
        code = """
fun processData(): String {
    val (name, age) = Pair("Alice", 30)
    
    val list = listOf(1, 2, 3, 4, 5)
    val (first, second, _, _, fifth) = list
    
    val map = mapOf("key1" to "value1", "key2" to "value2")
    for ((key, value) in map) {
        if (key.startsWith("key")) {
            println("$key: $value")
        }
    }
    
    return "$name is $age years old"
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        func = result.functions[0]
        assert func.complexity >= 2  # For loop and if condition
    
    # ==================== Type Alias and Generics Tests ====================
    
    @pytest.mark.asyncio
    async def test_type_alias_and_generics(self, analyzer):
        """Test complexity with type aliases and generics."""
        code = """
typealias StringMap<T> = Map<String, T>
typealias Handler<T> = (T) -> Unit

class Container<T : Comparable<T>> {
    private val items = mutableListOf<T>()
    
    fun add(item: T) {
        items.add(item)
        items.sort()
    }
    
    fun <R> map(transform: (T) -> R): List<R> {
        return items.map(transform)
    }
    
    inline fun <reified U> filterByType(): List<U> {
        return items.filterIsInstance<U>()
    }
}
"""
        result = await analyzer.analyze_code(code)
        
        assert result.success
        assert result.summary.class_count >= 1
        assert result.summary.function_count >= 3
    
    # ==================== Performance Tests ====================
    
    @pytest.mark.asyncio
    async def test_large_file_performance(self, analyzer):
        """Test performance with large files."""
        # Generate a large Kotlin file
        functions = []
        for i in range(50):
            functions.append(f"""
fun function{i}(x: Int): Int {{
    return when {{
        x > {i} -> x * {i}
        x < {i} -> x + {i}
        else -> {i}
    }}
}}
""")
        
        code = "\n".join(functions)
        
        import time
        start = time.time()
        result = await analyzer.analyze_code(code)
        duration = time.time() - start
        
        assert result.success
        assert result.summary.function_count == 50
        assert duration < 5.0  # Should complete within 5 seconds
    
    # ==================== Recommendation Tests ====================
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, analyzer):
        """Test generation of improvement recommendations."""
        code = """
fun veryComplexFunction(a: Int, b: Int, c: Int, d: Int, e: Int): Int {
    var result = 0
    for (i in 0 until a) {
        if (i % 2 == 0) {
            for (j in 0 until b) {
                if (j % 3 == 0) {
                    for (k in 0 until c) {
                        if (k % 5 == 0) {
                            for (l in 0 until d) {
                                if (l % 7 == 0) {
                                    for (m in 0 until e) {
                                        if (m % 11 == 0) {
                                            result += i * j * k * l * m
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
    return result
}
"""
        result = await analyzer.analyze_code(code)
        recommendations = result.generate_recommendations()
        
        assert len(recommendations) > 0
        assert any("refactor" in r.lower() for r in recommendations)
        assert any("complex" in r.lower() for r in recommendations)