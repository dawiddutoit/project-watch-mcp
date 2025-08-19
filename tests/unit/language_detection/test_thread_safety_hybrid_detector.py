"""Thread safety tests for HybridLanguageDetector."""

import concurrent.futures
import threading
import time
from typing import List
from unittest.mock import Mock, patch

import pytest

from project_watch_mcp.language_detection.hybrid_detector import HybridLanguageDetector
from project_watch_mcp.language_detection.models import DetectionMethod, LanguageDetectionResult


class TestHybridDetectorThreadSafety:
    """Test thread safety of HybridLanguageDetector."""
    
    def test_concurrent_parser_initialization(self):
        """Test that parser initialization is thread-safe."""
        # Create multiple detector instances concurrently
        detectors = []
        exceptions = []
        
        def create_detector():
            try:
                detector = HybridLanguageDetector(enable_cache=False)
                detectors.append(detector)
            except Exception as e:
                exceptions.append(e)
        
        # Create threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_detector)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no exceptions occurred
        assert len(exceptions) == 0, f"Exceptions during initialization: {exceptions}"
        assert len(detectors) == 10, "Not all detectors were created"
        
        # Verify all detectors have parsers initialized
        for detector in detectors:
            assert len(detector.tree_sitter_parsers) > 0, "Parsers not initialized"
    
    def test_concurrent_detection_no_corruption(self):
        """Test that concurrent detection doesn't corrupt parser state."""
        detector = HybridLanguageDetector(enable_cache=False)
        
        # Different code samples
        python_code = """
def hello_world():
    print("Hello, World!")
"""
        javascript_code = """
function helloWorld() {
    console.log("Hello, World!");
}
"""
        java_code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""
        
        samples = [
            (python_code, "python"),
            (javascript_code, "javascript"),
            (java_code, "java")
        ]
        
        results = []
        errors = []
        
        def detect_language(code, expected_lang):
            try:
                for _ in range(100):  # Multiple detections per thread
                    result = detector.detect(code)
                    if result.language != expected_lang and result.confidence > 0.5:
                        errors.append(f"Expected {expected_lang}, got {result.language}")
                    results.append(result)
            except Exception as e:
                errors.append(f"Exception during detection: {e}")
        
        # Run concurrent detections
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for _ in range(5):  # 5 rounds
                for code, expected in samples:
                    future = executor.submit(detect_language, code, expected)
                    futures.append(future)
            
            # Wait for all to complete
            concurrent.futures.wait(futures)
        
        # Verify no errors
        assert len(errors) == 0, f"Errors during detection: {errors[:5]}"  # Show first 5 errors
        assert len(results) > 0, "No results collected"
    
    def test_concurrent_parser_access(self):
        """Test that concurrent access to parsers is thread-safe."""
        detector = HybridLanguageDetector(enable_cache=False)
        
        # Track any race conditions
        race_conditions = []
        parser_states = {}
        lock = threading.Lock()
        
        def access_parser(lang_name, thread_id):
            try:
                for i in range(50):
                    # Try to access and use parser
                    if lang_name in detector.tree_sitter_parsers:
                        parser = detector.tree_sitter_parsers[lang_name]
                        
                        # Store parser id to check for corruption
                        with lock:
                            parser_id = id(parser)
                            if lang_name not in parser_states:
                                parser_states[lang_name] = parser_id
                            elif parser_states[lang_name] != parser_id:
                                race_conditions.append(
                                    f"Parser for {lang_name} changed from "
                                    f"{parser_states[lang_name]} to {parser_id}"
                                )
                        
                        # Use the parser
                        test_code = "test code"
                        tree = parser.parse(test_code.encode('utf-8'))
                        assert tree is not None
                        
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
            except Exception as e:
                race_conditions.append(f"Thread {thread_id} error: {e}")
        
        # Create threads that access parsers concurrently
        threads = []
        for i in range(10):
            for lang in ["python", "javascript", "java"]:
                thread = threading.Thread(target=access_parser, args=(lang, i))
                threads.append(thread)
                thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check for race conditions
        assert len(race_conditions) == 0, f"Race conditions detected: {race_conditions}"
    
    def test_concurrent_cache_operations(self):
        """Test that cache operations are thread-safe with detector."""
        detector = HybridLanguageDetector(enable_cache=True, cache_max_size=100)
        
        code_samples = [
            ("print('hello')", "python"),
            ("console.log('hello')", "javascript"),
            ("System.out.println('hello')", "java"),
        ]
        
        errors = []
        
        def detect_with_cache(code, expected, thread_id):
            try:
                for _ in range(50):
                    result = detector.detect(code)
                    # Cache operations happen internally
                    cache_info = detector.get_cache_info()
                    assert cache_info is not None
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Run concurrent cache operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(20):
                code, expected = code_samples[i % len(code_samples)]
                future = executor.submit(detect_with_cache, code, expected, i)
                futures.append(future)
            
            concurrent.futures.wait(futures)
        
        assert len(errors) == 0, f"Errors during cache operations: {errors}"
        
        # Verify cache is still functional
        cache_info = detector.get_cache_info()
        assert cache_info["size"] > 0, "Cache should have entries"
    
    def test_lazy_parser_initialization_thread_safe(self):
        """Test that lazy initialization of parsers is thread-safe."""
        # Mock the tree-sitter imports to simulate delayed initialization
        with patch('project_watch_mcp.language_detection.hybrid_detector.tree_sitter_python') as mock_py, \
             patch('project_watch_mcp.language_detection.hybrid_detector.tree_sitter_javascript') as mock_js, \
             patch('project_watch_mcp.language_detection.hybrid_detector.tree_sitter_java') as mock_java:
            
            # Setup mocks
            mock_py.language.return_value = Mock()
            mock_js.language.return_value = Mock()
            mock_java.language.return_value = Mock()
            
            # Track initialization calls
            init_calls = []
            init_lock = threading.Lock()
            
            original_init = HybridLanguageDetector._initialize_tree_sitter
            
            def tracked_init(self):
                with init_lock:
                    init_calls.append(threading.current_thread().name)
                return original_init(self)
            
            # Patch initialization
            HybridLanguageDetector._initialize_tree_sitter = tracked_init
            
            try:
                # Create multiple detectors concurrently
                detectors = []
                
                def create_and_use_detector():
                    detector = HybridLanguageDetector(enable_cache=False)
                    # Trigger parser usage
                    detector.detect("print('test')")
                    detectors.append(detector)
                
                threads = []
                for i in range(5):
                    thread = threading.Thread(target=create_and_use_detector, name=f"Thread-{i}")
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
                
                # Each detector should have initialized exactly once
                assert len(init_calls) == 5, f"Expected 5 init calls, got {len(init_calls)}"
                assert len(detectors) == 5, "All detectors should be created"
                
            finally:
                # Restore original method
                HybridLanguageDetector._initialize_tree_sitter = original_init
    
    def test_concurrent_batch_detection(self):
        """Test that batch detection is thread-safe."""
        detector = HybridLanguageDetector(enable_cache=True)
        
        # Create batches of files
        batch1 = [
            ("file1.py", "def test(): pass"),
            ("file2.js", "function test() {}"),
            ("file3.java", "public class Test {}")
        ]
        
        batch2 = [
            ("file4.py", "import os"),
            ("file5.js", "const x = 1;"),
            ("file6.java", "import java.util.*;")
        ]
        
        errors = []
        results = []
        
        def process_batch(batch, batch_id):
            try:
                batch_results = detector.detect_batch(batch)
                results.append((batch_id, batch_results))
            except Exception as e:
                errors.append(f"Batch {batch_id} error: {e}")
        
        # Process batches concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                batch = batch1 if i % 2 == 0 else batch2
                future = executor.submit(process_batch, batch, i)
                futures.append(future)
            
            concurrent.futures.wait(futures)
        
        assert len(errors) == 0, f"Errors during batch processing: {errors}"
        assert len(results) == 10, "All batches should be processed"
        
        # Verify results consistency
        for batch_id, batch_results in results:
            assert len(batch_results) == 3, f"Batch {batch_id} should have 3 results"
            for result in batch_results:
                assert result.language != "unknown" or result.confidence == 0.0