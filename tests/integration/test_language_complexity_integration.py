"""Integration tests for language detection and complexity analysis modules."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from src.project_watch_mcp.language_detection import HybridLanguageDetector
from src.project_watch_mcp.complexity_analysis import AnalyzerRegistry
from src.project_watch_mcp.complexity_analysis.languages import (
    PythonComplexityAnalyzer,
    JavaComplexityAnalyzer,
    KotlinComplexityAnalyzer,
)


class TestLanguageDetectionAndComplexityIntegration:
    """Test integration between language detection and complexity analysis."""
    
    @pytest.fixture
    def detector(self):
        """Create a language detector with cache enabled."""
        return HybridLanguageDetector(enable_cache=True)
    
    @pytest.fixture
    def python_analyzer(self):
        """Create a Python complexity analyzer."""
        return PythonComplexityAnalyzer()
    
    @pytest.fixture
    def java_analyzer(self):
        """Create a Java complexity analyzer."""
        return JavaComplexityAnalyzer()
    
    @pytest.fixture
    def kotlin_analyzer(self):
        """Create a Kotlin complexity analyzer."""
        return KotlinComplexityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_detect_and_analyze_python(self, detector, python_analyzer):
        """Test detecting Python code and analyzing its complexity."""
        code = """
def calculate_fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

class MathOperations:
    def factorial(self, n):
        if n <= 1:
            return 1
        return n * self.factorial(n - 1)
    
    def is_prime(self, n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
"""
        
        # Detect language
        detection_result = detector.detect(code, file_path="math.py")
        assert detection_result.language == "python"
        assert detection_result.confidence > 0.8
        
        # Analyze complexity
        complexity_result = await python_analyzer.analyze_code(code)
        assert complexity_result.language == "python"
        assert complexity_result.summary.function_count >= 3
        assert complexity_result.summary.class_count == 1
        
        # Check for recursive function detection
        factorial_func = next(
            (f for f in complexity_result.functions if f.name == "factorial"),
            None
        )
        assert factorial_func is not None
        assert factorial_func.is_recursive is True
    
    @pytest.mark.asyncio
    async def test_detect_and_analyze_java(self, detector, java_analyzer):
        """Test detecting Java code and analyzing its complexity."""
        code = """
public class StringProcessor {
    public String processString(String input, boolean uppercase) {
        if (input == null || input.isEmpty()) {
            return "";
        }
        
        String result = input.trim();
        
        if (uppercase) {
            result = result.toUpperCase();
        } else {
            result = result.toLowerCase();
        }
        
        if (result.length() > 10) {
            return result.substring(0, 10);
        }
        
        return result;
    }
    
    public int countWords(String text) {
        if (text == null || text.trim().isEmpty()) {
            return 0;
        }
        
        String[] words = text.trim().split("\\s+");
        return words.length;
    }
}
"""
        
        # Detect language
        detection_result = detector.detect(code, file_path="StringProcessor.java")
        assert detection_result.language == "java"
        assert detection_result.confidence > 0.8
        
        # Analyze complexity
        complexity_result = await java_analyzer.analyze_code(code)
        assert complexity_result.language == "java"
        assert complexity_result.summary.function_count >= 2
        assert complexity_result.summary.class_count >= 1
        
        # Check complexity scores
        assert complexity_result.summary.total_complexity > 0
        assert complexity_result.summary.average_complexity > 0
    
    @pytest.mark.asyncio
    async def test_detect_and_analyze_kotlin(self, detector, kotlin_analyzer):
        """Test detecting Kotlin code and analyzing its complexity."""
        code = """
class DataValidator {
    fun validateEmail(email: String): Boolean {
        if (email.isEmpty()) return false
        
        val parts = email.split("@")
        if (parts.size != 2) return false
        
        val localPart = parts[0]
        val domainPart = parts[1]
        
        if (localPart.isEmpty() || domainPart.isEmpty()) return false
        
        if (!domainPart.contains(".")) return false
        
        return true
    }
    
    fun validateAge(age: Int?): Boolean {
        return when {
            age == null -> false
            age < 0 -> false
            age > 150 -> false
            else -> true
        }
    }
}

fun processNumbers(numbers: List<Int>): List<Int> {
    return numbers
        .filter { it > 0 }
        .map { it * 2 }
        .sorted()
}
"""
        
        # Detect language
        detection_result = detector.detect(code, file_path="DataValidator.kt")
        assert detection_result.language == "kotlin"
        assert detection_result.confidence > 0.6  # Kotlin detection might be lower
        
        # Analyze complexity
        complexity_result = await kotlin_analyzer.analyze_code(code)
        assert complexity_result.language == "kotlin"
        assert complexity_result.summary.function_count >= 3
        assert complexity_result.summary.class_count >= 1
    
    @pytest.mark.asyncio
    async def test_batch_detect_and_analyze(self, detector):
        """Test batch processing of multiple files with different languages."""
        # Create temporary files
        files = []
        
        # Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def greet(name):
    if name:
        return f"Hello, {name}!"
    return "Hello, World!"
""")
            files.append(('python', Path(f.name)))
        
        # Java file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write("""
public class Greeter {
    public String greet(String name) {
        if (name != null && !name.isEmpty()) {
            return "Hello, " + name + "!";
        }
        return "Hello, World!";
    }
}
""")
            files.append(('java', Path(f.name)))
        
        # Kotlin file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kt', delete=False) as f:
            f.write("""
fun greet(name: String?): String {
    return if (!name.isNullOrEmpty()) {
        "Hello, $name!"
    } else {
        "Hello, World!"
    }
}
""")
            files.append(('kotlin', Path(f.name)))
        
        try:
            # Process all files
            for expected_lang, file_path in files:
                content = file_path.read_text()
                
                # Detect language
                detection_result = detector.detect(content, file_path=str(file_path))
                assert detection_result.language == expected_lang
                
                # Get appropriate analyzer
                if expected_lang == 'python':
                    analyzer = PythonComplexityAnalyzer()
                elif expected_lang == 'java':
                    analyzer = JavaComplexityAnalyzer()
                else:
                    analyzer = KotlinComplexityAnalyzer()
                
                # Analyze complexity
                complexity_result = await analyzer.analyze_file(file_path)
                assert complexity_result.language == expected_lang
                assert complexity_result.summary.function_count >= 1
        
        finally:
            # Clean up
            for _, file_path in files:
                file_path.unlink()
    
    @pytest.mark.asyncio
    async def test_concurrent_detection_and_analysis(self, detector):
        """Test concurrent language detection and complexity analysis."""
        codes = [
            ("python", """
def process(x):
    if x > 0:
        return x * 2
    return -x
"""),
            ("java", """
public int process(int x) {
    if (x > 0) {
        return x * 2;
    }
    return -x;
}
"""),
            ("javascript", """
function process(x) {
    if (x > 0) {
        return x * 2;
    }
    return -x;
}
"""),
        ]
        
        results = []
        
        async def detect_and_store(lang, code):
            detection = detector.detect(code, file_path=f"test.{lang[:2]}")
            results.append((lang, detection))
        
        # Run detections concurrently
        tasks = [detect_and_store(lang, code) for lang, code in codes]
        await asyncio.gather(*tasks)
        
        # Verify results
        assert len(results) == 3
        for expected_lang, detection in results:
            if expected_lang in ['python', 'java', 'javascript']:
                assert detection.confidence > 0
    
    def test_cache_performance_improvement(self, detector):
        """Test that caching improves detection performance."""
        code = """
def complex_function(data):
    result = []
    for item in data:
        if isinstance(item, dict):
            for key, value in item.items():
                if value > 0:
                    result.append(value * 2)
                else:
                    result.append(abs(value))
        elif isinstance(item, list):
            result.extend(complex_function(item))
        else:
            result.append(item)
    return result
"""
        
        # First detection (no cache)
        start_time = time.time()
        for _ in range(10):
            detector.detect(code, file_path="test.py")
        first_run_time = time.time() - start_time
        
        # Clear cache statistics
        detector.reset_cache_statistics()
        
        # Second run (with cache)
        start_time = time.time()
        for _ in range(10):
            detector.detect(code, file_path="test.py")
        second_run_time = time.time() - start_time
        
        # Cache should make it faster
        assert second_run_time < first_run_time
        
        # Verify cache was used
        cache_info = detector.get_cache_info()
        assert cache_info['stats']['hits'] >= 9  # First call misses, rest hit
    
    @pytest.mark.asyncio
    async def test_analyzer_registry_integration(self):
        """Test the analyzer registry for dynamic analyzer selection."""
        # Ensure analyzers are registered
        assert AnalyzerRegistry.get("python") is not None
        assert AnalyzerRegistry.get("java") is not None
        assert AnalyzerRegistry.get("kotlin") is not None
        
        # Test with different code samples
        samples = [
            ("python", """
def test():
    return "Python"
"""),
            ("java", """
public String test() {
    return "Java";
}
"""),
            ("kotlin", """
fun test(): String {
    return "Kotlin"
}
"""),
        ]
        
        for language, code in samples:
            analyzer_class = AnalyzerRegistry.get(language)
            assert analyzer_class is not None
            
            analyzer = analyzer_class()
            result = await analyzer.analyze_code(code)
            assert result.language == language
    
    @pytest.mark.asyncio
    async def test_mixed_language_project_simulation(self, detector):
        """Simulate analyzing a mixed-language project."""
        project_files = [
            ("backend/server.py", """
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/data')
def get_data():
    data = process_request()
    if data:
        return jsonify(data)
    return jsonify({'error': 'No data'}), 404

def process_request():
    # Complex processing logic
    return {'result': 'success'}
"""),
            ("backend/Calculator.java", """
public class Calculator {
    public double calculate(double a, double b, String op) {
        switch (op) {
            case "+": return a + b;
            case "-": return a - b;
            case "*": return a * b;
            case "/": return b != 0 ? a / b : 0;
            default: return 0;
        }
    }
}
"""),
            ("backend/DataProcessor.kt", """
class DataProcessor {
    fun processData(items: List<Any>): List<String> {
        return items.mapNotNull { item ->
            when (item) {
                is String -> item.uppercase()
                is Int -> item.toString()
                else -> null
            }
        }
    }
}
"""),
            ("frontend/app.js", """
const fetchData = async () => {
    try {
        const response = await fetch('/api/data');
        if (response.ok) {
            return await response.json();
        }
        throw new Error('Failed to fetch');
    } catch (error) {
        console.error(error);
        return null;
    }
};
"""),
        ]
        
        results = []
        
        for file_path, code in project_files:
            # Detect language
            detection = detector.detect(code, file_path=file_path)
            
            # Determine expected language from extension
            ext = Path(file_path).suffix
            expected_langs = {
                '.py': 'python',
                '.java': 'java',
                '.kt': 'kotlin',
                '.js': 'javascript'
            }
            
            expected = expected_langs.get(ext, 'unknown')
            
            # Store result
            results.append({
                'file': file_path,
                'detected': detection.language,
                'expected': expected,
                'confidence': detection.confidence,
                'method': detection.method
            })
        
        # Verify all detections
        for result in results:
            assert result['detected'] == result['expected'], \
                f"Failed to detect {result['expected']} for {result['file']}"
            assert result['confidence'] > 0.5, \
                f"Low confidence for {result['file']}: {result['confidence']}"
    
    def test_cache_thread_safety_with_detection(self, detector):
        """Test thread-safe caching during concurrent detection."""
        code_samples = [
            ("def hello(): pass", "test1.py"),
            ("function hello() {}", "test2.js"),
            ("public void hello() {}", "test3.java"),
            ("fun hello() {}", "test4.kt"),
            ("SELECT * FROM users", "test5.sql"),
        ]
        
        errors = []
        results = []
        
        def detect_concurrently(code, path):
            try:
                for _ in range(10):  # Multiple detections per thread
                    result = detector.detect(code, file_path=path)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent detections
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(detect_concurrently, code, path)
                for code, path in code_samples
            ]
            for future in futures:
                future.result()
        
        # No errors should occur
        assert len(errors) == 0
        # All detections should complete
        assert len(results) == 50  # 5 samples * 10 iterations
        
        # Cache should have entries
        cache_info = detector.get_cache_info()
        assert cache_info['size'] > 0
        assert cache_info['stats']['hits'] > 0  # Should have cache hits


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in integration."""
    
    @pytest.mark.asyncio
    async def test_unsupported_language_handling(self):
        """Test handling of unsupported languages."""
        detector = HybridLanguageDetector()
        
        # COBOL code (not supported)
        cobol_code = """
       IDENTIFICATION DIVISION.
       PROGRAM-ID. HELLO-WORLD.
       PROCEDURE DIVISION.
           DISPLAY 'Hello, World!'.
           STOP RUN.
"""
        
        result = detector.detect(cobol_code, file_path="hello.cob")
        # Should still return a result, even if unknown
        assert result is not None
        # Confidence might be low or language might be unknown/text
        assert result.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_malformed_code_handling(self):
        """Test handling of malformed code."""
        detector = HybridLanguageDetector()
        analyzers = [
            PythonComplexityAnalyzer(),
            JavaComplexityAnalyzer(),
            KotlinComplexityAnalyzer(),
        ]
        
        malformed_codes = [
            "def broken(: pass",  # Python syntax error
            "public class { }",  # Java syntax error
            "fun broken( {",  # Kotlin syntax error
        ]
        
        for code, analyzer in zip(malformed_codes, analyzers):
            # Detection should still work
            detection = detector.detect(code)
            assert detection is not None
            
            # Complexity analysis should handle errors gracefully
            result = await analyzer.analyze_code(code)
            assert result is not None
            assert result.error is not None or result.summary.total_complexity == 0
    
    @pytest.mark.asyncio
    async def test_empty_file_handling(self):
        """Test handling of empty files."""
        detector = HybridLanguageDetector()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")  # Empty file
            f.flush()
            
            # Detection should use extension
            content = f.read()
            detection = detector.detect(content, file_path=f.name)
            assert detection.language == "python"
            assert detection.method.value == "extension"
            
            # Complexity analysis should handle empty file
            analyzer = PythonComplexityAnalyzer()
            result = await analyzer.analyze_file(Path(f.name))
            assert result.summary.function_count == 0
            assert result.summary.total_complexity == 0
            
            Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_very_large_file_handling(self):
        """Test handling of very large files."""
        detector = HybridLanguageDetector()
        
        # Generate a large Python file
        large_code = "def func_%d():\n    return %d\n\n" * 1000
        large_code = "".join(large_code % (i, i) for i in range(1000))
        
        # Detection should still work
        detection = detector.detect(large_code, file_path="large.py")
        assert detection.language == "python"
        
        # Complexity analysis should handle large file
        analyzer = PythonComplexityAnalyzer()
        result = await analyzer.analyze_code(large_code)
        assert result.summary.function_count == 1000
        assert result.summary.total_complexity > 0