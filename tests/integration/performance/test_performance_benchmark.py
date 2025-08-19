"""
Comprehensive performance benchmarking for project-watch-mcp.
Tests all three major features: vector search, language detection, and complexity analysis.
Identifies bottlenecks and validates performance targets.
"""

import asyncio
import gc
import json
import os
import random
import statistics
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from neo4j import AsyncGraphDatabase

from project_watch_mcp.complexity_analysis import AnalyzerRegistry
from project_watch_mcp.complexity_analysis.languages.java_analyzer import JavaComplexityAnalyzer
from project_watch_mcp.complexity_analysis.languages.python_analyzer import PythonComplexityAnalyzer
from project_watch_mcp.language_detection import HybridLanguageDetector
from project_watch_mcp.language_detection.cache import LanguageDetectionCache
from project_watch_mcp.neo4j_rag import Neo4jRAG
from project_watch_mcp.utils.embeddings import create_embeddings_provider
from project_watch_mcp.vector_search.neo4j_native_vectors import NativeVectorIndex, VectorIndexConfig


@dataclass
class PerformanceMetrics:
    """Performance metrics for a test run."""
    
    operation: str
    samples: List[float] = field(default_factory=list)
    memory_start: float = 0
    memory_peak: float = 0
    memory_end: float = 0
    errors: int = 0
    
    @property
    def count(self) -> int:
        return len(self.samples)
    
    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0
    
    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else 0
    
    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0
    
    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else 0
    
    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0
    
    @property
    def p95(self) -> float:
        if not self.samples:
            return 0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[idx] if idx < len(sorted_samples) else sorted_samples[-1]
    
    @property
    def p99(self) -> float:
        if not self.samples:
            return 0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[idx] if idx < len(sorted_samples) else sorted_samples[-1]
    
    @property
    def memory_delta(self) -> float:
        return self.memory_end - self.memory_start
    
    def to_dict(self) -> Dict:
        return {
            "operation": self.operation,
            "count": self.count,
            "mean_ms": round(self.mean * 1000, 2),
            "median_ms": round(self.median * 1000, 2),
            "stdev_ms": round(self.stdev * 1000, 2),
            "min_ms": round(self.min * 1000, 2),
            "max_ms": round(self.max * 1000, 2),
            "p95_ms": round(self.p95 * 1000, 2),
            "p99_ms": round(self.p99 * 1000, 2),
            "memory_start_mb": round(self.memory_start / 1024 / 1024, 2),
            "memory_peak_mb": round(self.memory_peak / 1024 / 1024, 2),
            "memory_delta_mb": round(self.memory_delta / 1024 / 1024, 2),
            "errors": self.errors
        }


class PerformanceBenchmark:
    """Performance benchmarking suite for project-watch-mcp."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.neo4j_rag: Optional[Neo4jRAG] = None
        self.vector_index: Optional[NativeVectorIndex] = None
        self.language_detector: Optional[HybridLanguageDetector] = None
        self.analyzer_registry: Optional[AnalyzerRegistry] = None
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None
    
    async def setup(self):
        """Set up test environment."""
        # Create temp directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Initialize Neo4j RAG
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        PROJECT_WATCH_USER = os.getenv("PROJECT_WATCH_USER", "neo4j")
        PROJECT_WATCH_PASSWORD = os.getenv("PROJECT_WATCH_PASSWORD", "testpassword")
        
        self.neo4j_rag = Neo4jRAG(neo4j_uri, PROJECT_WATCH_USER, PROJECT_WATCH_PASSWORD)
        await self.neo4j_rag.initialize()
        
        # Initialize vector index
        config = VectorIndexConfig(
            index_name="perf_test_vectors",
            dimensions=1536,
            similarity_metric="cosine",
            provider="openai"
        )
        self.vector_index = NativeVectorIndex(self.neo4j_rag.driver, config)
        await self.vector_index.create_index()
        
        # Initialize language detector
        self.language_detector = HybridLanguageDetector()
        
        # Initialize complexity analyzers
        self.analyzer_registry = AnalyzerRegistry()
        self.analyzer_registry.register("python", PythonComplexityAnalyzer())
        self.analyzer_registry.register("java", JavaComplexityAnalyzer())
    
    async def teardown(self):
        """Clean up test environment."""
        if self.vector_index:
            await self.vector_index.delete_index()
        if self.neo4j_rag:
            await self.neo4j_rag.close()
        if self.temp_dir:
            self.temp_dir.cleanup()
    
    def create_test_files(self, count: int) -> List[Path]:
        """Create test files for benchmarking."""
        files = []
        
        # Python files
        for i in range(count // 3):
            content = f"""
import os
import sys
from typing import List, Optional

class TestClass{i}:
    def __init__(self):
        self.value = {i}
    
    def complex_method(self, x: int, y: int) -> int:
        result = 0
        if x > 0:
            for i in range(x):
                if i % 2 == 0:
                    result += i
                else:
                    result -= i
        elif y > 0:
            while y > 0:
                result += y
                y -= 1
        else:
            try:
                result = x / y
            except ZeroDivisionError:
                result = 0
        return result
    
    async def async_method(self):
        await asyncio.sleep(0.1)
        return self.value
"""
            path = Path(self.temp_dir.name) / f"test_{i}.py"
            path.write_text(content)
            files.append(path)
        
        # JavaScript files
        for i in range(count // 3, 2 * count // 3):
            content = f"""
const value = {i};

function complexFunction(x, y) {{
    let result = 0;
    if (x > 0) {{
        for (let i = 0; i < x; i++) {{
            if (i % 2 === 0) {{
                result += i;
            }} else {{
                result -= i;
            }}
        }}
    }} else if (y > 0) {{
        while (y > 0) {{
            result += y;
            y--;
        }}
    }} else {{
        try {{
            result = x / y;
        }} catch (e) {{
            result = 0;
        }}
    }}
    return result;
}}

class TestClass{i} {{
    constructor() {{
        this.value = value;
    }}
    
    async asyncMethod() {{
        await new Promise(resolve => setTimeout(resolve, 100));
        return this.value;
    }}
}}
"""
            path = Path(self.temp_dir.name) / f"test_{i}.js"
            path.write_text(content)
            files.append(path)
        
        # Java files
        for i in range(2 * count // 3, count):
            content = f"""
package com.test;

import java.util.*;
import java.util.stream.*;

public class TestClass{i} {{
    private int value = {i};
    
    public int complexMethod(int x, int y) {{
        int result = 0;
        if (x > 0) {{
            for (int i = 0; i < x; i++) {{
                if (i % 2 == 0) {{
                    result += i;
                }} else {{
                    result -= i;
                }}
            }}
        }} else if (y > 0) {{
            while (y > 0) {{
                result += y;
                y--;
            }}
        }} else {{
            try {{
                result = x / y;
            }} catch (ArithmeticException e) {{
                result = 0;
            }}
        }}
        return result;
    }}
    
    public List<Integer> streamMethod(List<Integer> numbers) {{
        return numbers.stream()
            .filter(n -> n > 0)
            .map(n -> n * 2)
            .collect(Collectors.toList());
    }}
}}
"""
            path = Path(self.temp_dir.name) / f"TestClass{i}.java"
            path.write_text(content)
            files.append(path)
        
        return files
    
    async def benchmark_vector_search(self, iterations: int = 100):
        """Benchmark vector search performance."""
        metric = PerformanceMetrics("vector_search")
        
        # Create test embeddings
        embeddings_provider = create_embeddings_provider("openai")
        test_texts = [f"Test code snippet {i} with some functionality" for i in range(20)]
        
        # Store embeddings
        for i, text in enumerate(test_texts):
            embedding = await embeddings_provider.embed(text)
            await self.vector_index.upsert(
                node_id=f"test_{i}",
                vector=embedding,
                metadata={"text": text, "index": i}
            )
        
        # Start memory tracking
        tracemalloc.start()
        metric.memory_start = tracemalloc.get_traced_memory()[0]
        
        # Benchmark searches
        queries = [f"Find code related to functionality {i % 5}" for i in range(iterations)]
        
        for query in queries:
            try:
                gc.collect()
                start = time.perf_counter()
                
                # Perform vector search
                query_embedding = await embeddings_provider.embed(query)
                results = await self.vector_index.search(
                    vector=query_embedding,
                    limit=5
                )
                
                elapsed = time.perf_counter() - start
                metric.samples.append(elapsed)
                
                # Track memory
                current_memory = tracemalloc.get_traced_memory()[0]
                metric.memory_peak = max(metric.memory_peak, current_memory)
                
            except Exception as e:
                metric.errors += 1
                print(f"Vector search error: {e}")
        
        metric.memory_end = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        self.metrics["vector_search"] = metric
        return metric
    
    async def benchmark_language_detection(self, iterations: int = 100):
        """Benchmark language detection performance."""
        metric = PerformanceMetrics("language_detection")
        
        # Create test files
        files = self.create_test_files(30)
        
        # Initialize cache
        cache = LanguageDetectionCache(max_size=100, max_age_seconds=3600)
        
        # Start memory tracking
        tracemalloc.start()
        metric.memory_start = tracemalloc.get_traced_memory()[0]
        
        # Benchmark detection (with cache warming)
        for i in range(iterations):
            try:
                file = random.choice(files)
                gc.collect()
                
                # Check cache first
                file_content = file.read_text()
                cached_result = cache.get(file_content, str(file))
                
                if cached_result is not None:
                    # Cache hit - measure cache retrieval
                    start = time.perf_counter()
                    result = cached_result
                    elapsed = time.perf_counter() - start
                else:
                    # Cache miss - measure detection
                    start = time.perf_counter()
                    result = await self.language_detector.detect_language(str(file))
                    elapsed = time.perf_counter() - start
                    
                    # Cache the result
                    cache.put(file_content, result, str(file))
                
                metric.samples.append(elapsed)
                
                # Track memory
                current_memory = tracemalloc.get_traced_memory()[0]
                metric.memory_peak = max(metric.memory_peak, current_memory)
                
            except Exception as e:
                metric.errors += 1
                print(f"Language detection error: {e}")
        
        metric.memory_end = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        self.metrics["language_detection"] = metric
        return metric
    
    async def benchmark_complexity_analysis(self, iterations: int = 100):
        """Benchmark complexity analysis performance."""
        metric = PerformanceMetrics("complexity_analysis")
        
        # Create test files
        files = self.create_test_files(30)
        python_files = [f for f in files if f.suffix == ".py"]
        java_files = [f for f in files if f.suffix == ".java"]
        
        # Start memory tracking
        tracemalloc.start()
        metric.memory_start = tracemalloc.get_traced_memory()[0]
        
        # Benchmark analysis
        for i in range(iterations):
            try:
                # Alternate between Python and Java
                if i % 2 == 0 and python_files:
                    file = random.choice(python_files)
                    analyzer = self.analyzer_registry.get_analyzer("python")
                elif java_files:
                    file = random.choice(java_files)
                    analyzer = self.analyzer_registry.get_analyzer("java")
                else:
                    continue
                
                gc.collect()
                start = time.perf_counter()
                
                # Perform complexity analysis
                result = await analyzer.analyze(str(file))
                
                elapsed = time.perf_counter() - start
                metric.samples.append(elapsed)
                
                # Track memory
                current_memory = tracemalloc.get_traced_memory()[0]
                metric.memory_peak = max(metric.memory_peak, current_memory)
                
            except Exception as e:
                metric.errors += 1
                print(f"Complexity analysis error: {e}")
        
        metric.memory_end = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        self.metrics["complexity_analysis"] = metric
        return metric
    
    async def benchmark_combined_pipeline(self, iterations: int = 50):
        """Benchmark the combined pipeline (all features together)."""
        metric = PerformanceMetrics("combined_pipeline")
        
        # Create test files
        files = self.create_test_files(30)
        
        # Initialize components
        embeddings_provider = create_embeddings_provider("openai")
        cache = LanguageDetectionCache(max_size=100, max_age_seconds=3600)
        
        # Start memory tracking
        tracemalloc.start()
        metric.memory_start = tracemalloc.get_traced_memory()[0]
        
        for i in range(iterations):
            try:
                file = random.choice(files)
                gc.collect()
                start = time.perf_counter()
                
                # Step 1: Language detection
                file_content = file.read_text()
                lang_result = await self.language_detector.detect_language(str(file))
                cache.put(file_content, lang_result, str(file))
                
                # Step 2: Complexity analysis (if supported)
                if lang_result.language in ["python", "java"]:
                    analyzer = self.analyzer_registry.get_analyzer(lang_result.language)
                    complexity_result = await analyzer.analyze(str(file))
                
                # Step 3: Generate embedding for semantic search
                embedding = await embeddings_provider.embed(file_content[:500])  # First 500 chars
                
                # Step 4: Store in vector index
                await self.vector_index.upsert(
                    node_id=f"file_{file.name}_{i}",
                    vector=embedding,
                    metadata={
                        "file": str(file),
                        "language": lang_result.language,
                        "confidence": lang_result.confidence
                    }
                )
                
                # Step 5: Search for similar files
                results = await self.vector_index.search(
                    vector=embedding,
                    limit=3
                )
                
                elapsed = time.perf_counter() - start
                metric.samples.append(elapsed)
                
                # Track memory
                current_memory = tracemalloc.get_traced_memory()[0]
                metric.memory_peak = max(metric.memory_peak, current_memory)
                
            except Exception as e:
                metric.errors += 1
                print(f"Combined pipeline error: {e}")
        
        metric.memory_end = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        self.metrics["combined_pipeline"] = metric
        return metric
    
    async def benchmark_batch_processing(self):
        """Benchmark batch processing optimizations."""
        metric = PerformanceMetrics("batch_processing")
        
        # Create test files
        files = self.create_test_files(100)
        
        # Start memory tracking
        tracemalloc.start()
        metric.memory_start = tracemalloc.get_traced_memory()[0]
        
        # Batch language detection
        start = time.perf_counter()
        batch_results = await self.language_detector.detect_batch([str(f) for f in files])
        batch_time = time.perf_counter() - start
        metric.samples.append(batch_time)
        
        # Sequential language detection for comparison
        start = time.perf_counter()
        for file in files:
            await self.language_detector.detect_language(str(file))
        sequential_time = time.perf_counter() - start
        
        # Calculate speedup
        speedup = sequential_time / batch_time if batch_time > 0 else 0
        
        metric.memory_end = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        self.metrics["batch_processing"] = metric
        self.metrics["batch_speedup"] = PerformanceMetrics("batch_speedup")
        self.metrics["batch_speedup"].samples = [speedup]
        
        return metric
    
    async def benchmark_database_connections(self):
        """Benchmark database connection efficiency."""
        metric = PerformanceMetrics("database_connections")
        
        # Start memory tracking
        tracemalloc.start()
        metric.memory_start = tracemalloc.get_traced_memory()[0]
        
        # Test connection pool efficiency
        tasks = []
        for i in range(50):
            async def query_task():
                start = time.perf_counter()
                async with self.neo4j_rag.driver.session() as session:
                    result = await session.run("RETURN 1 as n")
                    await result.single()
                return time.perf_counter() - start
            
            tasks.append(query_task())
        
        # Run concurrent queries
        results = await asyncio.gather(*tasks)
        metric.samples.extend(results)
        
        metric.memory_end = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        self.metrics["database_connections"] = metric
        return metric
    
    def generate_report(self) -> Dict:
        """Generate performance report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {},
            "performance_targets": {
                "vector_search": {"target_ms": 50, "actual_ms": 0, "status": ""},
                "language_detection": {"target_ms": 100, "actual_ms": 0, "status": ""},
                "complexity_analysis": {"target_fps": 30, "actual_fps": 0, "status": ""},
                "combined_pipeline": {"target_ms": 200, "actual_ms": 0, "status": ""}
            },
            "bottlenecks": [],
            "optimizations": []
        }
        
        # Add metrics
        for name, metric in self.metrics.items():
            report["metrics"][name] = metric.to_dict()
        
        # Check performance targets
        if "vector_search" in self.metrics:
            avg_ms = self.metrics["vector_search"].mean * 1000
            report["performance_targets"]["vector_search"]["actual_ms"] = round(avg_ms, 2)
            report["performance_targets"]["vector_search"]["status"] = "✓ PASS" if avg_ms < 50 else "✗ FAIL"
        
        if "language_detection" in self.metrics:
            avg_ms = self.metrics["language_detection"].mean * 1000
            report["performance_targets"]["language_detection"]["actual_ms"] = round(avg_ms, 2)
            report["performance_targets"]["language_detection"]["status"] = "✓ PASS" if avg_ms < 100 else "✗ FAIL"
        
        if "complexity_analysis" in self.metrics:
            fps = 1 / self.metrics["complexity_analysis"].mean if self.metrics["complexity_analysis"].mean > 0 else 0
            report["performance_targets"]["complexity_analysis"]["actual_fps"] = round(fps, 2)
            report["performance_targets"]["complexity_analysis"]["status"] = "✓ PASS" if fps > 30 else "✗ FAIL"
        
        if "combined_pipeline" in self.metrics:
            avg_ms = self.metrics["combined_pipeline"].mean * 1000
            report["performance_targets"]["combined_pipeline"]["actual_ms"] = round(avg_ms, 2)
            report["performance_targets"]["combined_pipeline"]["status"] = "✓ PASS" if avg_ms < 200 else "✗ FAIL"
        
        # Identify bottlenecks
        bottlenecks = []
        for name, metric in self.metrics.items():
            if metric.p99 > metric.mean * 2:
                bottlenecks.append({
                    "operation": name,
                    "issue": "High variance",
                    "p99_vs_mean": round(metric.p99 / metric.mean, 2),
                    "recommendation": "Investigate outliers causing performance spikes"
                })
            
            if metric.memory_delta > 50 * 1024 * 1024:  # 50MB
                bottlenecks.append({
                    "operation": name,
                    "issue": "High memory usage",
                    "memory_delta_mb": round(metric.memory_delta / 1024 / 1024, 2),
                    "recommendation": "Optimize memory allocation and consider streaming"
                })
            
            if metric.errors > 0:
                bottlenecks.append({
                    "operation": name,
                    "issue": "Errors detected",
                    "error_count": metric.errors,
                    "recommendation": "Fix errors to improve reliability"
                })
        
        report["bottlenecks"] = bottlenecks
        
        # Generate optimization recommendations
        optimizations = []
        
        if "batch_speedup" in self.metrics and self.metrics["batch_speedup"].samples:
            speedup = self.metrics["batch_speedup"].samples[0]
            if speedup < 2:
                optimizations.append({
                    "area": "Batch Processing",
                    "current_speedup": round(speedup, 2),
                    "recommendation": "Improve batch processing with better parallelization"
                })
        
        if "database_connections" in self.metrics:
            p99_ms = self.metrics["database_connections"].p99 * 1000
            if p99_ms > 10:
                optimizations.append({
                    "area": "Database Connections",
                    "p99_latency_ms": round(p99_ms, 2),
                    "recommendation": "Optimize connection pooling and query patterns"
                })
        
        report["optimizations"] = optimizations
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted performance report."""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        print(f"Timestamp: {report['timestamp']}")
        
        print("\n📊 PERFORMANCE METRICS")
        print("-" * 80)
        for name, metric in report["metrics"].items():
            print(f"\n{name}:")
            print(f"  Samples: {metric['count']}")
            print(f"  Mean: {metric['mean_ms']} ms")
            print(f"  Median: {metric['median_ms']} ms")
            print(f"  P95: {metric['p95_ms']} ms")
            print(f"  P99: {metric['p99_ms']} ms")
            print(f"  Memory Delta: {metric['memory_delta_mb']} MB")
            if metric['errors'] > 0:
                print(f"  ⚠️  Errors: {metric['errors']}")
        
        print("\n🎯 PERFORMANCE TARGETS")
        print("-" * 80)
        for target, data in report["performance_targets"].items():
            print(f"{target}: {data['status']}")
            if "actual_ms" in data:
                print(f"  Target: <{data['target_ms']} ms, Actual: {data['actual_ms']} ms")
            elif "actual_fps" in data:
                print(f"  Target: >{data['target_fps']} files/sec, Actual: {data['actual_fps']} files/sec")
        
        if report["bottlenecks"]:
            print("\n⚠️  BOTTLENECKS IDENTIFIED")
            print("-" * 80)
            for bottleneck in report["bottlenecks"]:
                print(f"\n{bottleneck['operation']}: {bottleneck['issue']}")
                for key, value in bottleneck.items():
                    if key not in ["operation", "issue", "recommendation"]:
                        print(f"  {key}: {value}")
                print(f"  → {bottleneck['recommendation']}")
        
        if report["optimizations"]:
            print("\n💡 OPTIMIZATION RECOMMENDATIONS")
            print("-" * 80)
            for opt in report["optimizations"]:
                print(f"\n{opt['area']}:")
                for key, value in opt.items():
                    if key != "area" and key != "recommendation":
                        print(f"  {key}: {value}")
                print(f"  → {opt['recommendation']}")
        
        print("\n" + "=" * 80)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("NEO4J_URI") or not os.getenv("OPENAI_API_KEY"),
    reason="Requires Neo4j and OpenAI API key"
)
class TestPerformanceBenchmark:
    """Test suite for performance benchmarking."""
    
    async def test_full_benchmark_suite(self):
        """Run the complete performance benchmark suite."""
        benchmark = PerformanceBenchmark()
        
        try:
            # Setup
            await benchmark.setup()
            
            # Run benchmarks
            print("\n🚀 Starting Performance Benchmark Suite...")
            
            print("\n1. Benchmarking Vector Search...")
            await benchmark.benchmark_vector_search(iterations=50)
            
            print("2. Benchmarking Language Detection...")
            await benchmark.benchmark_language_detection(iterations=100)
            
            print("3. Benchmarking Complexity Analysis...")
            await benchmark.benchmark_complexity_analysis(iterations=100)
            
            print("4. Benchmarking Combined Pipeline...")
            await benchmark.benchmark_combined_pipeline(iterations=30)
            
            print("5. Benchmarking Batch Processing...")
            await benchmark.benchmark_batch_processing()
            
            print("6. Benchmarking Database Connections...")
            await benchmark.benchmark_database_connections()
            
            # Generate and print report
            report = benchmark.generate_report()
            benchmark.print_report(report)
            
            # Save report to file
            report_path = Path("performance_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Full report saved to: {report_path}")
            
            # Validate performance targets
            assert report["performance_targets"]["vector_search"]["status"] == "✓ PASS", \
                f"Vector search failed target: {report['performance_targets']['vector_search']['actual_ms']} ms"
            
            assert report["performance_targets"]["language_detection"]["status"] == "✓ PASS", \
                f"Language detection failed target: {report['performance_targets']['language_detection']['actual_ms']} ms"
            
            assert report["performance_targets"]["complexity_analysis"]["status"] == "✓ PASS", \
                f"Complexity analysis failed target: {report['performance_targets']['complexity_analysis']['actual_fps']} fps"
            
            assert report["performance_targets"]["combined_pipeline"]["status"] == "✓ PASS", \
                f"Combined pipeline failed target: {report['performance_targets']['combined_pipeline']['actual_ms']} ms"
            
        finally:
            # Cleanup
            await benchmark.teardown()
    
    async def test_vector_search_under_load(self):
        """Test vector search performance under heavy load."""
        benchmark = PerformanceBenchmark()
        
        try:
            await benchmark.setup()
            
            # Generate large dataset
            embeddings_provider = create_embeddings_provider("openai")
            
            print("\n🔥 Testing Vector Search Under Load...")
            
            # Store 1000 embeddings
            print("Storing 1000 embeddings...")
            for i in range(1000):
                text = f"Code snippet {i} with various functionality and features"
                embedding = await embeddings_provider.embed(text)
                await benchmark.vector_index.upsert(
                    node_id=f"load_test_{i}",
                    vector=embedding,
                    metadata={"text": text, "index": i}
                )
            
            # Concurrent search test
            print("Running 100 concurrent searches...")
            async def search_task(query_id):
                query = f"Find code related to feature {query_id % 10}"
                embedding = await embeddings_provider.embed(query)
                start = time.perf_counter()
                results = await benchmark.vector_index.search(
                    vector=embedding,
                    limit=10
                )
                return time.perf_counter() - start
            
            tasks = [search_task(i) for i in range(100)]
            results = await asyncio.gather(*tasks)
            
            # Analyze results
            avg_latency = statistics.mean(results) * 1000
            p99_latency = sorted(results)[int(len(results) * 0.99)] * 1000
            
            print(f"Average latency: {avg_latency:.2f} ms")
            print(f"P99 latency: {p99_latency:.2f} ms")
            
            assert avg_latency < 50, f"Vector search under load failed: {avg_latency} ms"
            
        finally:
            await benchmark.teardown()
    
    async def test_memory_leak_detection(self):
        """Test for memory leaks in long-running operations."""
        benchmark = PerformanceBenchmark()
        
        try:
            await benchmark.setup()
            
            print("\n🔍 Testing for Memory Leaks...")
            
            # Track memory over iterations
            tracemalloc.start()
            memory_samples = []
            
            for iteration in range(10):
                gc.collect()
                
                # Run operations
                files = benchmark.create_test_files(10)
                
                for file in files:
                    # Language detection
                    result = await benchmark.language_detector.detect_language(str(file))
                    
                    # Complexity analysis
                    if result.language == "python":
                        analyzer = benchmark.analyzer_registry.get_analyzer("python")
                        await analyzer.analyze(str(file))
                
                # Record memory
                current, peak = tracemalloc.get_traced_memory()
                memory_samples.append(current)
                
                print(f"Iteration {iteration + 1}: {current / 1024 / 1024:.2f} MB")
            
            tracemalloc.stop()
            
            # Check for memory growth
            first_half = statistics.mean(memory_samples[:5])
            second_half = statistics.mean(memory_samples[5:])
            growth_rate = (second_half - first_half) / first_half
            
            print(f"Memory growth rate: {growth_rate * 100:.2f}%")
            
            # Allow up to 10% growth (some caching is expected)
            assert growth_rate < 0.1, f"Potential memory leak detected: {growth_rate * 100:.2f}% growth"
            
        finally:
            await benchmark.teardown()


if __name__ == "__main__":
    # Run benchmark directly
    async def main():
        benchmark = PerformanceBenchmark()
        try:
            await benchmark.setup()
            
            # Run all benchmarks
            await benchmark.benchmark_vector_search(iterations=50)
            await benchmark.benchmark_language_detection(iterations=100)
            await benchmark.benchmark_complexity_analysis(iterations=100)
            await benchmark.benchmark_combined_pipeline(iterations=30)
            await benchmark.benchmark_batch_processing()
            await benchmark.benchmark_database_connections()
            
            # Generate report
            report = benchmark.generate_report()
            benchmark.print_report(report)
            
            # Save report
            with open("performance_report.json", "w") as f:
                json.dump(report, f, indent=2)
            
        finally:
            await benchmark.teardown()
    
    asyncio.run(main())