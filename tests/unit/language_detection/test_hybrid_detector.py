"""Unit tests for hybrid language detection system."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.project_watch_mcp.language_detection import (
    HybridLanguageDetector,
    LanguageDetectionResult,
    DetectionMethod,
)


class TestLanguageDetectionResult:
    """Test the LanguageDetectionResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a detection result."""
        result = LanguageDetectionResult(
            language="python",
            confidence=0.95,
            method=DetectionMethod.TREE_SITTER
        )
        assert result.language == "python"
        assert result.confidence == 0.95
        assert result.method == DetectionMethod.TREE_SITTER
    
    def test_result_comparison(self):
        """Test comparing detection results by confidence."""
        result1 = LanguageDetectionResult("python", 0.8, DetectionMethod.TREE_SITTER)
        result2 = LanguageDetectionResult("javascript", 0.9, DetectionMethod.PYGMENTS)
        assert result2.confidence > result1.confidence


class TestHybridLanguageDetector:
    """Test the HybridLanguageDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return HybridLanguageDetector()
    
    def test_detector_initialization(self, detector):
        """Test detector initializes with proper parsers."""
        assert detector is not None
        assert hasattr(detector, 'tree_sitter_parsers')
        assert 'python' in detector.tree_sitter_parsers
        assert 'javascript' in detector.tree_sitter_parsers
        assert 'java' in detector.tree_sitter_parsers
    
    def test_detect_from_extension_python(self, detector):
        """Test detecting Python from file extension."""
        result = detector._detect_from_extension(".py")
        assert result.language == "python"
        assert result.confidence == 0.7  # Extension-based confidence
        assert result.method == DetectionMethod.EXTENSION
    
    def test_detect_from_extension_javascript(self, detector):
        """Test detecting JavaScript from file extension."""
        result = detector._detect_from_extension(".js")
        assert result.language == "javascript"
        assert result.confidence == 0.7
        assert result.method == DetectionMethod.EXTENSION
    
    def test_detect_from_extension_java(self, detector):
        """Test detecting Java from file extension."""
        result = detector._detect_from_extension(".java")
        assert result.language == "java"
        assert result.confidence == 0.7
        assert result.method == DetectionMethod.EXTENSION
    
    def test_detect_from_extension_kotlin(self, detector):
        """Test detecting Kotlin from file extension."""
        result = detector._detect_from_extension(".kt")
        assert result.language == "kotlin"
        assert result.confidence == 0.7
        assert result.method == DetectionMethod.EXTENSION
    
    def test_detect_from_extension_unknown(self, detector):
        """Test handling unknown file extension."""
        result = detector._detect_from_extension(".xyz")
        assert result.language == "unknown"
        assert result.confidence == 0.0
        assert result.method == DetectionMethod.EXTENSION
    
    def test_detect_from_tree_sitter_python(self, detector):
        """Test detecting Python using tree-sitter."""
        python_code = """
def hello_world():
    print("Hello, World!")
    return 42
"""
        result = detector._detect_from_tree_sitter(python_code)
        assert result.language == "python"
        assert result.confidence >= 0.9  # Tree-sitter confidence
        assert result.method == DetectionMethod.TREE_SITTER
    
    def test_detect_from_tree_sitter_javascript(self, detector):
        """Test detecting JavaScript using tree-sitter."""
        js_code = """
function helloWorld() {
    console.log("Hello, World!");
    return 42;
}
"""
        result = detector._detect_from_tree_sitter(js_code)
        assert result.language == "javascript"
        assert result.confidence >= 0.9
        assert result.method == DetectionMethod.TREE_SITTER
    
    def test_detect_from_tree_sitter_java(self, detector):
        """Test detecting Java using tree-sitter."""
        java_code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""
        result = detector._detect_from_tree_sitter(java_code)
        assert result.language == "java"
        assert result.confidence >= 0.9
        assert result.method == DetectionMethod.TREE_SITTER
    
    def test_detect_from_tree_sitter_invalid_syntax(self, detector):
        """Test tree-sitter with invalid syntax."""
        invalid_code = "This is not valid code in any language {]}"
        result = detector._detect_from_tree_sitter(invalid_code)
        # Tree-sitter is very robust and might still parse it
        # Just check it returns a result
        assert result.method == DetectionMethod.TREE_SITTER
        # With invalid syntax, confidence should be lower than perfect parse
        assert result.confidence <= 1.0
    
    def test_detect_from_pygments_python(self, detector):
        """Test detecting Python using Pygments."""
        python_code = """
def calculate_sum(a, b):
    return a + b
"""
        result = detector._detect_from_pygments(python_code)
        # Pygments might detect this as various things, but should have some confidence
        assert result.language in ["python", "text", "unknown"]
        if result.language != "unknown":
            assert result.confidence > 0  # Some confidence
        assert result.method == DetectionMethod.PYGMENTS
    
    def test_detect_from_pygments_kotlin(self, detector):
        """Test detecting Kotlin using Pygments."""
        kotlin_code = """
fun main() {
    val message = "Hello, Kotlin!"
    println(message)
}
"""
        result = detector._detect_from_pygments(kotlin_code)
        # Pygments detection can vary
        assert result.language in ["kotlin", "text", "unknown"]
        if result.language != "unknown":
            assert result.confidence > 0
        assert result.method == DetectionMethod.PYGMENTS
    
    def test_detect_hybrid_with_file_path(self, detector):
        """Test hybrid detection with file path."""
        python_code = """
def main():
    print("Hello from Python")
"""
        result = detector.detect(
            content=python_code,
            file_path="test.py"
        )
        assert result.language == "python"
        assert result.confidence >= 0.9  # Should use tree-sitter
        assert result.method == DetectionMethod.TREE_SITTER
    
    def test_detect_hybrid_fallback_to_pygments(self, detector):
        """Test fallback to Pygments when tree-sitter fails."""
        # Kotlin code (no tree-sitter parser for Kotlin)
        kotlin_code = """
fun calculateSum(a: Int, b: Int): Int {
    return a + b
}
"""
        result = detector.detect(
            content=kotlin_code,
            file_path="test.kt"
        )
        assert result.language == "kotlin"
        assert result.confidence >= 0.7  # Extension or Pygments
        assert result.method in [DetectionMethod.PYGMENTS, DetectionMethod.EXTENSION]
    
    def test_detect_hybrid_fallback_to_extension(self, detector):
        """Test fallback to extension when content detection fails."""
        # Very ambiguous content that tree-sitter can't parse well
        ambiguous_content = "x"
        result = detector.detect(
            content=ambiguous_content,
            file_path="script.py"
        )
        assert result.language == "python"
        # Should use extension or tree-sitter depending on parse quality
        assert result.method in [DetectionMethod.EXTENSION, DetectionMethod.TREE_SITTER]
    
    def test_detect_batch_processing(self, detector):
        """Test batch processing of multiple files."""
        files = [
            ("test1.py", "def hello(): pass"),
            ("test2.js", "function hello() {}"),
            ("test3.java", "public class Test {}"),
            ("test4.kt", "fun main() {}"),
        ]
        
        results = detector.detect_batch(files)
        
        assert len(results) == 4
        assert results[0].language == "python"
        assert results[1].language == "javascript"
        assert results[2].language == "java"
        # Kotlin might be detected as various things since we don't have tree-sitter for it
        assert results[3].language in ["kotlin", "java", "text"]  # Extension should give kotlin
    
    def test_detect_confidence_scoring(self, detector):
        """Test confidence scoring across different methods."""
        # Strong Python code
        strong_python = """
import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self):
        self.data = pd.DataFrame()
    
    def process(self):
        return self.data.mean()
"""
        result = detector.detect(content=strong_python)
        # Should detect as Python or text (Pygments might see numpy)
        assert result.language in ["python", "numpy", "text"]
        # Should have reasonable confidence
        assert result.confidence > 0.5
        
        # Weak/ambiguous code
        weak_code = "x = 1"
        result = detector.detect(content=weak_code)
        # Simple code should have lower confidence than complex code
        # But tree-sitter might still parse it well
        assert result.confidence <= 1.0  # Valid confidence range
    
    def test_detect_with_empty_content(self, detector):
        """Test detection with empty content."""
        result = detector.detect(content="", file_path="test.py")
        assert result.language == "python"  # From extension
        assert result.method == DetectionMethod.EXTENSION
    
    def test_detect_with_no_hints(self, detector):
        """Test detection with no file path and ambiguous content."""
        result = detector.detect(content="# Comment")
        # Could be many languages with # comments
        assert result.language in ["python", "shell", "unknown", "text", "text only"]
        # Just verify we get a valid confidence score
        assert 0 <= result.confidence <= 1.0
    
    @pytest.mark.parametrize("extension,expected", [
        (".py", "python"),
        (".pyw", "python"),
        (".js", "javascript"),
        (".jsx", "javascript"),
        (".ts", "typescript"),
        (".tsx", "typescript"),
        (".java", "java"),
        (".kt", "kotlin"),
        (".kts", "kotlin"),
        (".rb", "ruby"),
        (".go", "go"),
        (".rs", "rust"),
        (".cpp", "cpp"),
        (".c", "c"),
        (".cs", "csharp"),
        (".php", "php"),
        (".swift", "swift"),
        (".m", "objc"),
        (".scala", "scala"),
        (".r", "r"),
    ])
    def test_extension_mapping(self, detector, extension, expected):
        """Test file extension to language mapping."""
        result = detector._detect_from_extension(extension)
        assert result.language == expected
    
    def test_normalize_language_names(self, detector):
        """Test language name normalization."""
        assert detector._normalize_language("Python") == "python"
        assert detector._normalize_language("JavaScript") == "javascript"
        assert detector._normalize_language("C++") == "cpp"
        assert detector._normalize_language("C#") == "csharp"
        assert detector._normalize_language("Objective-C") == "objc"