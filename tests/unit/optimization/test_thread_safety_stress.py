"""Stress tests for thread safety in HybridLanguageDetector."""

import concurrent.futures
import random
import threading
import time
from typing import Dict, List, Set

import pytest

from project_watch_mcp.language_detection.hybrid_detector import HybridLanguageDetector
from project_watch_mcp.language_detection.models import LanguageDetectionResult


class TestThreadSafetyStress:
    """Stress test thread safety under high concurrency."""
    
    def test_high_concurrency_detection(self):
        """Test detection under high concurrent load."""
        detector = HybridLanguageDetector(enable_cache=True, cache_max_size=500)
        
        # Various code samples for different languages
        code_samples = {
            "python": [
                "def hello(): print('Hello')",
                "import os\nprint(os.path.exists('.'))",
                "class MyClass:\n    def __init__(self):\n        pass",
                "for i in range(10):\n    print(i)",
                "[x**2 for x in range(5)]",
            ],
            "javascript": [
                "function hello() { console.log('Hello'); }",
                "const arr = [1, 2, 3].map(x => x * 2);",
                "async function fetchData() { return await fetch('/api'); }",
                "module.exports = { hello: 'world' };",
                "let obj = { name: 'test', value: 123 };",
            ],
            "java": [
                "public class Main { public static void main(String[] args) {} }",
                "import java.util.*;\nclass Test {}",
                "@Override\npublic String toString() { return 'test'; }",
                "private final String name = 'value';",
                "System.out.println('Hello World');",
            ]
        }
        
        results = []
        errors = []
        detection_counts: Dict[str, int] = {}
        lock = threading.Lock()
        
        def worker(thread_id: int, iterations: int):
            """Worker thread that performs multiple detections."""
            local_counts = {"python": 0, "javascript": 0, "java": 0, "unknown": 0}
            
            try:
                for _ in range(iterations):
                    # Randomly select a language and sample
                    lang = random.choice(list(code_samples.keys()))
                    code = random.choice(code_samples[lang])
                    
                    # Add some randomization to test different paths
                    use_cache = random.choice([True, False, None])
                    add_path = random.choice([True, False])
                    file_path = f"test_{thread_id}.{lang[:2]}" if add_path else None
                    
                    # Perform detection
                    result = detector.detect(code, file_path, use_cache)
                    
                    # Track results
                    detected_lang = result.language
                    if detected_lang in local_counts:
                        local_counts[detected_lang] += 1
                    else:
                        local_counts["unknown"] += 1
                    
                    # Occasionally clear cache to test cache operations
                    if random.random() < 0.01:  # 1% chance
                        detector.clear_cache()
                    
                    # Occasionally get cache info
                    if random.random() < 0.05:  # 5% chance
                        info = detector.get_cache_info()
                        assert info is not None or not detector.cache_enabled
                    
                    results.append(result)
                    
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
            finally:
                # Update global counts
                with lock:
                    for lang, count in local_counts.items():
                        detection_counts[lang] = detection_counts.get(lang, 0) + count
        
        # Run stress test with many threads
        num_threads = 50
        iterations_per_thread = 100
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker, i, iterations_per_thread)
                for i in range(num_threads)
            ]
            concurrent.futures.wait(futures)
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors[:10]}"  # Show first 10
        assert len(results) > 0, "No results collected"
        
        # Check that detections happened for all languages
        assert detection_counts.get("python", 0) > 0, "No Python detected"
        assert detection_counts.get("javascript", 0) > 0, "No JavaScript detected"
        assert detection_counts.get("java", 0) > 0, "No Java detected"
        
        # Verify cache is still functional
        if detector.cache_enabled:
            cache_info = detector.get_cache_info()
            assert cache_info is not None
            # Cache should have some entries unless it was just cleared
            # Due to random clearing, we can't assert a specific size
    
    def test_concurrent_parser_modifications(self):
        """Test concurrent parser additions and removals."""
        detector = HybridLanguageDetector(enable_cache=False)
        
        errors = []
        operations_completed = {"add": 0, "remove": 0, "detect": 0}
        lock = threading.Lock()
        
        def modifier_thread(thread_id: int):
            """Thread that adds/removes parsers."""
            try:
                for i in range(20):
                    if i % 3 == 0:
                        # Add a dummy parser (not functional, just for testing)
                        from tree_sitter import Parser, Language
                        # Use an existing language parser as dummy
                        import tree_sitter_python
                        dummy_parser = Parser(Language(tree_sitter_python.language()))
                        detector.add_parser(f"test_lang_{thread_id}_{i}", dummy_parser)
                        with lock:
                            operations_completed["add"] += 1
                    elif i % 3 == 1:
                        # Remove a parser
                        detector.remove_parser(f"test_lang_{thread_id}_{i-1}")
                        with lock:
                            operations_completed["remove"] += 1
                    else:
                        # Get supported languages
                        langs = detector.get_supported_languages()
                        assert isinstance(langs, list)
                    
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(f"Modifier {thread_id}: {e}")
        
        def detector_thread(thread_id: int):
            """Thread that performs detections."""
            try:
                for _ in range(50):
                    code = "def test(): pass"
                    result = detector.detect(code)
                    assert result is not None
                    with lock:
                        operations_completed["detect"] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Detector {thread_id}: {e}")
        
        # Mix modifier and detector threads
        threads = []
        for i in range(10):
            t1 = threading.Thread(target=modifier_thread, args=(i,))
            t2 = threading.Thread(target=detector_thread, args=(i,))
            threads.extend([t1, t2])
            t1.start()
            t2.start()
        
        for t in threads:
            t.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify operations completed
        assert operations_completed["add"] > 0, "No parsers added"
        assert operations_completed["remove"] > 0, "No parsers removed"
        assert operations_completed["detect"] > 0, "No detections performed"
        
        # Clean up - ensure base parsers still exist
        langs = detector.get_supported_languages()
        assert "python" in langs or len(langs) > 0  # At least some parsers remain
    
    def test_memory_consistency_under_load(self):
        """Test that detector maintains consistent state under load."""
        detector = HybridLanguageDetector(enable_cache=True)
        
        # Track unique parser IDs to detect any parser object changes
        parser_ids: Set[int] = set()
        id_changes = []
        lock = threading.Lock()
        
        def track_parser_ids(thread_id: int):
            """Track parser object IDs to detect unwanted changes."""
            try:
                for _ in range(100):
                    langs = detector.get_supported_languages()
                    for lang in langs:
                        # Access parser through internal dict with lock
                        with detector._parser_lock:
                            if lang in detector.tree_sitter_parsers:
                                parser = detector.tree_sitter_parsers[lang]
                                parser_id = id(parser)
                                
                                with lock:
                                    if lang == "python":  # Track a specific parser
                                        if parser_id not in parser_ids:
                                            parser_ids.add(parser_id)
                                            if len(parser_ids) > 1:
                                                id_changes.append(
                                                    f"Parser ID changed for {lang} in thread {thread_id}"
                                                )
                    
                    # Perform a detection to use the parsers
                    detector.detect("test code")
                    time.sleep(0.001)
            except Exception as e:
                with lock:
                    id_changes.append(f"Error in thread {thread_id}: {e}")
        
        # Run multiple threads
        threads = []
        for i in range(20):
            t = threading.Thread(target=track_parser_ids, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify parser objects remained consistent
        assert len(id_changes) == 0, f"Parser consistency issues: {id_changes}"
        assert len(parser_ids) == 1, f"Parser object changed during execution: {parser_ids}"
    
    def test_cache_consistency_with_concurrent_operations(self):
        """Test cache remains consistent with concurrent read/write/clear operations."""
        detector = HybridLanguageDetector(enable_cache=True, cache_max_size=100)
        
        test_codes = [
            ("print('test1')", "test1.py"),
            ("console.log('test2')", "test2.js"),
            ("public class Test3 {}", "test3.java"),
            ("def test4(): pass", "test4.py"),
            ("function test5() {}", "test5.js"),
        ]
        
        errors = []
        cache_operations = {"hits": 0, "misses": 0, "clears": 0}
        lock = threading.Lock()
        
        def cache_worker(thread_id: int):
            """Worker that performs various cache operations."""
            try:
                for _ in range(100):
                    operation = random.choice(["detect", "clear", "info", "reset_stats"])
                    
                    if operation == "detect":
                        code, path = random.choice(test_codes)
                        result = detector.detect(code, path)
                        assert result is not None
                        
                        # Check if it was a cache hit by detecting again
                        result2 = detector.detect(code, path)
                        if result.language == result2.language:
                            with lock:
                                cache_operations["hits"] += 1
                        else:
                            with lock:
                                cache_operations["misses"] += 1
                    
                    elif operation == "clear":
                        detector.clear_cache()
                        with lock:
                            cache_operations["clears"] += 1
                    
                    elif operation == "info":
                        info = detector.get_cache_info()
                        assert info is not None
                        assert "size" in info
                        assert "hit_rate" in info
                    
                    elif operation == "reset_stats":
                        detector.reset_cache_statistics()
                    
                    time.sleep(0.001)
                    
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
        
        # Run concurrent cache operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = [
                executor.submit(cache_worker, i)
                for i in range(30)
            ]
            concurrent.futures.wait(futures)
        
        # Verify no errors
        assert len(errors) == 0, f"Cache errors: {errors[:5]}"
        
        # Verify cache operations occurred
        assert cache_operations["hits"] > 0, "No cache hits recorded"
        assert cache_operations["clears"] > 0, "No cache clears performed"
        
        # Final cache should still be functional
        final_info = detector.get_cache_info()
        assert final_info is not None
        assert isinstance(final_info["size"], int)
        assert final_info["size"] >= 0
        assert final_info["size"] <= 100  # Should respect max_size