"""
Performance profiling utilities for project-watch-mcp.
Provides detailed profiling, bottleneck identification, and optimization recommendations.
"""

import asyncio
import cProfile
import functools
import io
import pstats
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil


@dataclass
class ProfileResult:
    """Result of a profiling session."""
    
    name: str
    duration: float
    cpu_percent: float
    memory_mb: float
    memory_peak_mb: float
    call_count: int = 0
    stats: Optional[pstats.Stats] = None
    
    def get_top_functions(self, limit: int = 10) -> List[Tuple[str, float, int]]:
        """Get top time-consuming functions."""
        if not self.stats:
            return []
        
        result = []
        stats_io = io.StringIO()
        self.stats.stream = stats_io
        self.stats.sort_stats('cumulative')
        self.stats.print_stats(limit)
        
        # Parse the output
        lines = stats_io.getvalue().split('\n')
        for line in lines:
            if '{' in line and '}' in line:
                parts = line.split()
                if len(parts) >= 6:
                    func_name = parts[-1]
                    cumtime = float(parts[3])
                    ncalls = int(parts[0].split('/')[0])
                    result.append((func_name, cumtime, ncalls))
        
        return result


class PerformanceProfiler:
    """Performance profiling utility."""
    
    def __init__(self, enable_cpu_profiling: bool = True):
        self.enable_cpu_profiling = enable_cpu_profiling
        self.results: Dict[str, ProfileResult] = {}
        self.process = psutil.Process()
    
    @contextmanager
    def profile(self, name: str):
        """Profile a synchronous code block."""
        # Start CPU profiling
        profiler = cProfile.Profile() if self.enable_cpu_profiling else None
        if profiler:
            profiler.enable()
        
        # Record start state
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        self.process.cpu_percent()  # First call to initialize
        
        try:
            yield
        finally:
            # Record end state
            duration = time.perf_counter() - start_time
            cpu_percent = self.process.cpu_percent()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            peak_memory = end_memory  # Approximation
            
            # Stop CPU profiling
            stats = None
            if profiler:
                profiler.disable()
                stats = pstats.Stats(profiler)
            
            # Store result
            result = ProfileResult(
                name=name,
                duration=duration,
                cpu_percent=cpu_percent,
                memory_mb=end_memory - start_memory,
                memory_peak_mb=peak_memory,
                stats=stats
            )
            self.results[name] = result
    
    @asynccontextmanager
    async def profile_async(self, name: str):
        """Profile an asynchronous code block."""
        # Record start state
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        self.process.cpu_percent()  # First call to initialize
        
        try:
            yield
        finally:
            # Record end state
            duration = time.perf_counter() - start_time
            cpu_percent = self.process.cpu_percent()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            peak_memory = end_memory  # Approximation
            
            # Store result
            result = ProfileResult(
                name=name,
                duration=duration,
                cpu_percent=cpu_percent,
                memory_mb=end_memory - start_memory,
                memory_peak_mb=peak_memory
            )
            self.results[name] = result
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile(func.__name__):
                return func(*args, **kwargs)
        return wrapper
    
    def profile_async_function(self, func: Callable) -> Callable:
        """Decorator to profile an async function."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.profile_async(func.__name__):
                return await func(*args, **kwargs)
        return wrapper
    
    def get_bottlenecks(self, threshold_ms: float = 100) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for name, result in self.results.items():
            issues = []
            
            # Check duration
            if result.duration * 1000 > threshold_ms:
                issues.append({
                    "type": "slow_execution",
                    "duration_ms": result.duration * 1000,
                    "threshold_ms": threshold_ms
                })
            
            # Check CPU usage
            if result.cpu_percent > 80:
                issues.append({
                    "type": "high_cpu",
                    "cpu_percent": result.cpu_percent,
                    "threshold_percent": 80
                })
            
            # Check memory usage
            if result.memory_mb > 100:
                issues.append({
                    "type": "high_memory",
                    "memory_mb": result.memory_mb,
                    "threshold_mb": 100
                })
            
            if issues:
                bottlenecks.append({
                    "operation": name,
                    "issues": issues,
                    "top_functions": result.get_top_functions(5) if result.stats else []
                })
        
        return bottlenecks
    
    def generate_report(self) -> str:
        """Generate a performance report."""
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE PROFILING REPORT")
        lines.append("=" * 80)
        
        # Summary
        lines.append("\n📊 SUMMARY")
        lines.append("-" * 40)
        total_duration = sum(r.duration for r in self.results.values())
        total_memory = sum(r.memory_mb for r in self.results.values())
        lines.append(f"Total operations: {len(self.results)}")
        lines.append(f"Total duration: {total_duration * 1000:.2f} ms")
        lines.append(f"Total memory delta: {total_memory:.2f} MB")
        
        # Individual operations
        lines.append("\n📈 OPERATION DETAILS")
        lines.append("-" * 40)
        for name, result in sorted(self.results.items(), key=lambda x: x[1].duration, reverse=True):
            lines.append(f"\n{name}:")
            lines.append(f"  Duration: {result.duration * 1000:.2f} ms")
            lines.append(f"  CPU: {result.cpu_percent:.1f}%")
            lines.append(f"  Memory: {result.memory_mb:.2f} MB")
            
            if result.stats:
                lines.append("  Top functions:")
                for func_name, cumtime, ncalls in result.get_top_functions(3):
                    lines.append(f"    - {func_name}: {cumtime:.3f}s ({ncalls} calls)")
        
        # Bottlenecks
        bottlenecks = self.get_bottlenecks()
        if bottlenecks:
            lines.append("\n⚠️  BOTTLENECKS")
            lines.append("-" * 40)
            for bottleneck in bottlenecks:
                lines.append(f"\n{bottleneck['operation']}:")
                for issue in bottleneck['issues']:
                    if issue['type'] == 'slow_execution':
                        lines.append(f"  ⏱️  Slow: {issue['duration_ms']:.2f} ms (>{issue['threshold_ms']} ms)")
                    elif issue['type'] == 'high_cpu':
                        lines.append(f"  🔥 High CPU: {issue['cpu_percent']:.1f}% (>{issue['threshold_percent']}%)")
                    elif issue['type'] == 'high_memory':
                        lines.append(f"  💾 High Memory: {issue['memory_mb']:.2f} MB (>{issue['threshold_mb']} MB)")
        
        # Optimization recommendations
        lines.append("\n💡 OPTIMIZATION RECOMMENDATIONS")
        lines.append("-" * 40)
        
        # Find slowest operation
        if self.results:
            slowest = max(self.results.items(), key=lambda x: x[1].duration)
            lines.append(f"1. Optimize '{slowest[0]}' - taking {slowest[1].duration * 1000:.2f} ms")
        
        # Check for memory issues
        high_memory = [name for name, r in self.results.items() if r.memory_mb > 50]
        if high_memory:
            lines.append(f"2. Review memory usage in: {', '.join(high_memory)}")
        
        # Check for CPU-intensive operations
        high_cpu = [name for name, r in self.results.items() if r.cpu_percent > 70]
        if high_cpu:
            lines.append(f"3. Consider async/parallel processing for: {', '.join(high_cpu)}")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


class ConcurrencyAnalyzer:
    """Analyze concurrency and parallelization opportunities."""
    
    def __init__(self):
        self.task_timings: Dict[str, List[float]] = {}
        self.concurrent_tasks: List[Tuple[str, float, float]] = []
    
    async def measure_task(self, name: str, coro):
        """Measure the execution time of an async task."""
        start = time.perf_counter()
        try:
            result = await coro
            duration = time.perf_counter() - start
            
            if name not in self.task_timings:
                self.task_timings[name] = []
            self.task_timings[name].append(duration)
            
            self.concurrent_tasks.append((name, start, start + duration))
            return result
        except Exception as e:
            duration = time.perf_counter() - start
            self.task_timings[name].append(duration)
            raise
    
    def analyze_concurrency(self) -> Dict[str, Any]:
        """Analyze concurrency patterns and opportunities."""
        if not self.concurrent_tasks:
            return {}
        
        # Sort tasks by start time
        sorted_tasks = sorted(self.concurrent_tasks, key=lambda x: x[1])
        
        # Find max concurrent tasks
        max_concurrent = 0
        current_concurrent = []
        
        events = []
        for name, start, end in sorted_tasks:
            events.append((start, 'start', name))
            events.append((end, 'end', name))
        
        events.sort()
        
        for time, event_type, name in events:
            if event_type == 'start':
                current_concurrent.append(name)
                max_concurrent = max(max_concurrent, len(current_concurrent))
            else:
                current_concurrent.remove(name)
        
        # Calculate parallelization efficiency
        total_time = max(t[2] for t in sorted_tasks) - min(t[1] for t in sorted_tasks)
        sum_individual_times = sum(t[2] - t[1] for t in sorted_tasks)
        parallelization_efficiency = sum_individual_times / total_time if total_time > 0 else 1
        
        return {
            "max_concurrent_tasks": max_concurrent,
            "total_tasks": len(sorted_tasks),
            "parallelization_efficiency": parallelization_efficiency,
            "total_time": total_time,
            "sum_individual_times": sum_individual_times,
            "time_saved": sum_individual_times - total_time if sum_individual_times > total_time else 0
        }


class MemoryProfiler:
    """Detailed memory profiling utility."""
    
    def __init__(self):
        self.snapshots: List[Tuple[str, float, Dict]] = []
        self.process = psutil.Process()
    
    def take_snapshot(self, label: str):
        """Take a memory snapshot."""
        mem_info = self.process.memory_info()
        snapshot = {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
        self.snapshots.append((label, time.time(), snapshot))
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if len(self.snapshots) < 2:
            return {}
        
        # Calculate memory growth
        first_snapshot = self.snapshots[0][2]
        last_snapshot = self.snapshots[-1][2]
        
        memory_growth = last_snapshot["rss_mb"] - first_snapshot["rss_mb"]
        growth_rate = memory_growth / first_snapshot["rss_mb"] if first_snapshot["rss_mb"] > 0 else 0
        
        # Find peak memory usage
        peak_memory = max(s[2]["rss_mb"] for s in self.snapshots)
        peak_label = next(s[0] for s in self.snapshots if s[2]["rss_mb"] == peak_memory)
        
        # Detect potential leaks
        consecutive_increases = 0
        max_consecutive_increases = 0
        
        for i in range(1, len(self.snapshots)):
            if self.snapshots[i][2]["rss_mb"] > self.snapshots[i-1][2]["rss_mb"]:
                consecutive_increases += 1
                max_consecutive_increases = max(max_consecutive_increases, consecutive_increases)
            else:
                consecutive_increases = 0
        
        potential_leak = max_consecutive_increases > len(self.snapshots) * 0.7
        
        return {
            "memory_growth_mb": memory_growth,
            "growth_rate_percent": growth_rate * 100,
            "peak_memory_mb": peak_memory,
            "peak_operation": peak_label,
            "potential_memory_leak": potential_leak,
            "snapshots": len(self.snapshots)
        }


def profile_operation(name: str = None):
    """Decorator for profiling individual operations."""
    def decorator(func):
        operation_name = name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                profiler = PerformanceProfiler()
                async with profiler.profile_async(operation_name):
                    result = await func(*args, **kwargs)
                
                # Log if bottleneck detected
                bottlenecks = profiler.get_bottlenecks(threshold_ms=50)
                if bottlenecks:
                    print(f"⚠️  Performance issue in {operation_name}: {bottlenecks[0]['issues']}")
                
                return result
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                profiler = PerformanceProfiler()
                with profiler.profile(operation_name):
                    result = func(*args, **kwargs)
                
                # Log if bottleneck detected
                bottlenecks = profiler.get_bottlenecks(threshold_ms=50)
                if bottlenecks:
                    print(f"⚠️  Performance issue in {operation_name}: {bottlenecks[0]['issues']}")
                
                return result
        
        return wrapper
    return decorator