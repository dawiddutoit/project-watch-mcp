"""Comprehensive accuracy tests for language detection across multiple languages."""

import pytest
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

from src.project_watch_mcp.language_detection import (
    HybridLanguageDetector,
    LanguageDetectionResult,
    DetectionMethod,
)


@dataclass
class AccuracyMetrics:
    """Metrics for language detection accuracy."""
    total_tests: int = 0
    correct_detections: int = 0
    high_confidence: int = 0  # > 0.9
    medium_confidence: int = 0  # 0.7-0.9
    low_confidence: int = 0  # < 0.7
    method_breakdown: Dict[str, int] = None
    average_detection_time: float = 0.0
    cache_hit_rate: float = 0.0
    
    def __post_init__(self):
        if self.method_breakdown is None:
            self.method_breakdown = {}
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.correct_detections / self.total_tests) * 100
    
    @property
    def high_confidence_rate(self) -> float:
        """Calculate high confidence rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.high_confidence / self.total_tests) * 100


class TestLanguageDetectionAccuracy:
    """Test language detection accuracy across multiple languages."""
    
    # Test code samples for each language
    LANGUAGE_SAMPLES = {
        "python": [
            # Sample 1: Basic Python with type hints
            '''
import asyncio
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class User:
    """User model with type hints."""
    id: int
    name: str
    email: Optional[str] = None
    metadata: Dict[str, Union[str, int]] = None

async def fetch_users(limit: int = 10) -> List[User]:
    """Fetch users asynchronously."""
    await asyncio.sleep(0.1)
    return [User(id=i, name=f"User{i}") for i in range(limit)]

if __name__ == "__main__":
    users = asyncio.run(fetch_users())
    print(f"Fetched {len(users)} users")
''',
            # Sample 2: Python with decorators and context managers
            '''
import functools
import time
from contextlib import contextmanager

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@contextmanager
def database_connection(db_url: str):
    """Context manager for database connections."""
    conn = None
    try:
        conn = connect_to_db(db_url)
        yield conn
    finally:
        if conn:
            conn.close()

@timing_decorator
def process_data(data: list) -> dict:
    """Process data with timing."""
    return {item: len(item) for item in data}
''',
            # Sample 3: Python with advanced features
            '''
from __future__ import annotations
import sys
from pathlib import Path

class FileProcessor:
    """Process files with pattern matching."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
    
    def process_file(self, file_path: Path) -> None:
        match file_path.suffix:
            case '.py':
                self._process_python(file_path)
            case '.json':
                self._process_json(file_path)
            case _:
                self._process_generic(file_path)
    
    def _process_python(self, path: Path) -> None:
        with open(path) as f:
            lines = sum(1 for _ in f)
        print(f"Python file has {lines} lines")
    
    @classmethod
    def from_string(cls, path_str: str) -> FileProcessor:
        return cls(Path(path_str))

# Use walrus operator
if (processor := FileProcessor.from_string(".")):
    processor.process_file(Path(__file__))
'''
        ],
        
        "javascript": [
            # Sample 1: Modern JavaScript with ES6+ features
            '''
// Modern JavaScript with ES6+ features
import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3000';

class DataService {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.cache = new Map();
    }
    
    async fetchData(endpoint, options = {}) {
        const cacheKey = `${endpoint}-${JSON.stringify(options)}`;
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }
        
        try {
            const response = await axios.get(`${API_BASE_URL}${endpoint}`, {
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    ...options.headers
                },
                ...options
            });
            
            this.cache.set(cacheKey, response.data);
            return response.data;
        } catch (error) {
            console.error(`Failed to fetch ${endpoint}:`, error);
            throw error;
        }
    }
}

export default DataService;
''',
            # Sample 2: JavaScript with async/await and destructuring
            '''
const fs = require('fs').promises;
const path = require('path');

async function* walkDirectory(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory()) {
            yield* walkDirectory(fullPath);
        } else {
            yield fullPath;
        }
    }
}

const processFiles = async (rootDir, { extensions = ['.js', '.json'], limit = 100 } = {}) => {
    const results = [];
    let count = 0;
    
    for await (const filePath of walkDirectory(rootDir)) {
        if (extensions.some(ext => filePath.endsWith(ext))) {
            const stats = await fs.stat(filePath);
            results.push({
                path: filePath,
                size: stats.size,
                modified: stats.mtime
            });
            
            if (++count >= limit) break;
        }
    }
    
    return results;
};

module.exports = { walkDirectory, processFiles };
''',
            # Sample 3: React component with hooks
            '''
import React, { useState, useCallback, useMemo, useEffect } from 'react';
import PropTypes from 'prop-types';

const DataTable = ({ data, columns, onRowClick, sortable = true }) => {
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
    const [filter, setFilter] = useState('');
    
    const handleSort = useCallback((key) => {
        if (!sortable) return;
        
        setSortConfig(prev => ({
            key,
            direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
        }));
    }, [sortable]);
    
    const filteredData = useMemo(() => {
        if (!filter) return data;
        
        return data.filter(row =>
            Object.values(row).some(value =>
                String(value).toLowerCase().includes(filter.toLowerCase())
            )
        );
    }, [data, filter]);
    
    const sortedData = useMemo(() => {
        if (!sortConfig.key) return filteredData;
        
        return [...filteredData].sort((a, b) => {
            const aVal = a[sortConfig.key];
            const bVal = b[sortConfig.key];
            
            if (sortConfig.direction === 'asc') {
                return aVal > bVal ? 1 : -1;
            }
            return aVal < bVal ? 1 : -1;
        });
    }, [filteredData, sortConfig]);
    
    return (
        <div className="data-table">
            <input
                type="text"
                placeholder="Filter..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
            />
            <table>
                <thead>
                    <tr>
                        {columns.map(col => (
                            <th key={col.key} onClick={() => handleSort(col.key)}>
                                {col.label}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {sortedData.map((row, idx) => (
                        <tr key={idx} onClick={() => onRowClick?.(row)}>
                            {columns.map(col => (
                                <td key={col.key}>{row[col.key]}</td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

DataTable.propTypes = {
    data: PropTypes.arrayOf(PropTypes.object).isRequired,
    columns: PropTypes.arrayOf(PropTypes.shape({
        key: PropTypes.string.isRequired,
        label: PropTypes.string.isRequired
    })).isRequired,
    onRowClick: PropTypes.func,
    sortable: PropTypes.bool
};

export default DataTable;
'''
        ],
        
        "java": [
            # Sample 1: Java with generics and streams
            '''
package com.example.service;

import java.util.*;
import java.util.stream.Collectors;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DataProcessor<T extends Comparable<T>> {
    private final ExecutorService executor;
    private final List<T> data;
    
    public DataProcessor() {
        this.executor = Executors.newFixedThreadPool(4);
        this.data = new ArrayList<>();
    }
    
    public CompletableFuture<List<T>> processAsync(List<T> input) {
        return CompletableFuture.supplyAsync(() -> {
            return input.stream()
                .filter(Objects::nonNull)
                .sorted()
                .distinct()
                .collect(Collectors.toList());
        }, executor);
    }
    
    public Optional<T> findMax() {
        return data.stream()
            .max(Comparator.naturalOrder());
    }
    
    public Map<Boolean, List<T>> partitionByThreshold(T threshold) {
        return data.stream()
            .collect(Collectors.partitioningBy(
                item -> item.compareTo(threshold) > 0
            ));
    }
    
    @Override
    protected void finalize() throws Throwable {
        executor.shutdown();
        super.finalize();
    }
}
''',
            # Sample 2: Java with annotations and reflection
            '''
package com.example.framework;

import java.lang.annotation.*;
import java.lang.reflect.Method;
import java.util.Arrays;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
@interface Cacheable {
    int ttl() default 60;
    String key() default "";
}

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
@interface Service {
    String name() default "";
}

@Service(name = "userService")
public class UserService {
    private Map<String, Object> cache = new HashMap<>();
    
    @Cacheable(ttl = 300, key = "user")
    public User getUser(Long id) {
        String cacheKey = "user_" + id;
        
        if (cache.containsKey(cacheKey)) {
            return (User) cache.get(cacheKey);
        }
        
        User user = fetchUserFromDatabase(id);
        cache.put(cacheKey, user);
        return user;
    }
    
    public void processAnnotations() {
        Method[] methods = this.getClass().getDeclaredMethods();
        
        Arrays.stream(methods)
            .filter(m -> m.isAnnotationPresent(Cacheable.class))
            .forEach(method -> {
                Cacheable cacheable = method.getAnnotation(Cacheable.class);
                System.out.println("Method " + method.getName() + 
                    " is cacheable with TTL: " + cacheable.ttl());
            });
    }
    
    private User fetchUserFromDatabase(Long id) {
        // Simulate database fetch
        return new User(id, "User" + id);
    }
}
''',
            # Sample 3: Java with modern features (records, switch expressions)
            '''
package com.example.modern;

import java.util.List;
import java.time.LocalDateTime;

public sealed interface Event permits LoginEvent, LogoutEvent, ErrorEvent {
    LocalDateTime timestamp();
    String userId();
}

record LoginEvent(LocalDateTime timestamp, String userId, String ipAddress) implements Event {}
record LogoutEvent(LocalDateTime timestamp, String userId) implements Event {}
record ErrorEvent(LocalDateTime timestamp, String userId, String message) implements Event {}

public class EventProcessor {
    
    public String processEvent(Event event) {
        return switch (event) {
            case LoginEvent(var time, var user, var ip) -> 
                String.format("User %s logged in from %s at %s", user, ip, time);
            case LogoutEvent(var time, var user) -> 
                String.format("User %s logged out at %s", user, time);
            case ErrorEvent(var time, var user, var msg) -> 
                String.format("Error for user %s at %s: %s", user, time, msg);
        };
    }
    
    public List<Event> filterRecentEvents(List<Event> events, int hours) {
        LocalDateTime cutoff = LocalDateTime.now().minusHours(hours);
        
        return events.stream()
            .filter(e -> e.timestamp().isAfter(cutoff))
            .toList();
    }
}
'''
        ],
        
        "kotlin": [
            # Sample 1: Kotlin with coroutines and data classes
            '''
package com.example.app

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.time.Instant

data class Message(
    val id: String,
    val content: String,
    val timestamp: Instant = Instant.now(),
    val metadata: Map<String, Any> = emptyMap()
)

class MessageService(private val scope: CoroutineScope) {
    private val _messages = MutableSharedFlow<Message>()
    val messages: SharedFlow<Message> = _messages.asSharedFlow()
    
    suspend fun sendMessage(content: String): Message {
        val message = Message(
            id = generateId(),
            content = content
        )
        
        _messages.emit(message)
        return message
    }
    
    fun processMessages() = scope.launch {
        messages
            .filter { it.content.isNotBlank() }
            .map { it.copy(content = it.content.trim()) }
            .collect { message ->
                println("Processing: ${message.content}")
                delay(100)
            }
    }
    
    private fun generateId(): String = 
        java.util.UUID.randomUUID().toString()
}

fun main() = runBlocking {
    val service = MessageService(this)
    
    service.processMessages()
    
    repeat(5) { i ->
        service.sendMessage("Message $i")
        delay(50)
    }
    
    delay(1000)
}
''',
            # Sample 2: Kotlin with extension functions and DSL
            '''
package com.example.dsl

class HtmlBuilder {
    private val elements = mutableListOf<String>()
    
    fun tag(name: String, block: HtmlBuilder.() -> Unit = {}) {
        elements.add("<$name>")
        block()
        elements.add("</$name>")
    }
    
    fun text(content: String) {
        elements.add(content)
    }
    
    fun attribute(name: String, value: String) {
        elements[elements.lastIndex] = 
            elements.last().replace(">", " $name=\"$value\">")
    }
    
    override fun toString() = elements.joinToString("")
}

fun html(block: HtmlBuilder.() -> Unit): String {
    return HtmlBuilder().apply(block).toString()
}

// Extension functions
fun String.toSnakeCase(): String {
    return this.replace(Regex("([A-Z])"), "_$1")
        .lowercase()
        .removePrefix("_")
}

fun <T> List<T>.secondOrNull(): T? = 
    if (size >= 2) this[1] else null

// Usage
fun main() {
    val htmlContent = html {
        tag("div") {
            attribute("class", "container")
            tag("h1") {
                text("Welcome")
            }
            tag("p") {
                text("This is Kotlin DSL")
            }
        }
    }
    
    println(htmlContent)
    println("CamelCase".toSnakeCase())
    
    val items = listOf(1, 2, 3)
    println(items.secondOrNull() ?: "No second element")
}
''',
            # Sample 3: Kotlin with sealed classes and when expressions
            '''
package com.example.state

import kotlin.reflect.KProperty

sealed class State<out T> {
    data class Success<T>(val data: T) : State<T>()
    data class Error(val exception: Throwable) : State<Nothing>()
    object Loading : State<Nothing>()
    
    inline fun <R> map(transform: (T) -> R): State<R> = when (this) {
        is Success -> Success(transform(data))
        is Error -> this
        is Loading -> this
    }
    
    inline fun onSuccess(action: (T) -> Unit): State<T> {
        if (this is Success) action(data)
        return this
    }
    
    inline fun onError(action: (Throwable) -> Unit): State<T> {
        if (this is Error) action(exception)
        return this
    }
}

class Repository<T> {
    private var cache: T? = null
    
    suspend fun fetchData(loader: suspend () -> T): State<T> {
        return try {
            State.Loading
            val data = cache ?: loader().also { cache = it }
            State.Success(data)
        } catch (e: Exception) {
            State.Error(e)
        }
    }
}

// Delegated properties
class Config {
    var apiUrl: String by ConfigDelegate("http://localhost")
    var timeout: Int by ConfigDelegate(5000)
}

class ConfigDelegate<T>(private var value: T) {
    operator fun getValue(thisRef: Any?, property: KProperty<*>): T {
        println("Getting ${property.name} = $value")
        return value
    }
    
    operator fun setValue(thisRef: Any?, property: KProperty<*>, newValue: T) {
        println("Setting ${property.name} = $newValue")
        value = newValue
    }
}
'''
        ]
    }
    
    @pytest.fixture
    def detector(self):
        """Create a detector with caching enabled."""
        return HybridLanguageDetector(enable_cache=True)
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_python_detection_accuracy(self, detector: HybridLanguageDetector):
        """Test Python detection accuracy across multiple samples."""
        metrics = AccuracyMetrics()
        
        for i, sample in enumerate(self.LANGUAGE_SAMPLES["python"]):
            start_time = time.perf_counter()
            result = detector.detect(sample, f"test{i}.py")
            detection_time = time.perf_counter() - start_time
            
            metrics.total_tests += 1
            metrics.average_detection_time += detection_time
            
            # Check if correctly detected as Python
            if result.language == "python":
                metrics.correct_detections += 1
            
            # Track confidence levels
            if result.confidence > 0.9:
                metrics.high_confidence += 1
            elif result.confidence >= 0.7:
                metrics.medium_confidence += 1
            else:
                metrics.low_confidence += 1
            
            # Track detection method
            method_name = result.method.value
            metrics.method_breakdown[method_name] = metrics.method_breakdown.get(method_name, 0) + 1
            
            # Assertions for individual tests
            assert result.language == "python", f"Sample {i} not detected as Python: {result.language}"
            assert result.confidence >= 0.9, f"Sample {i} confidence too low: {result.confidence}"
            assert result.method == DetectionMethod.TREE_SITTER, f"Sample {i} not using tree-sitter: {result.method}"
        
        # Calculate averages
        metrics.average_detection_time /= metrics.total_tests
        
        # Overall assertions
        assert metrics.accuracy >= 90, f"Python accuracy below 90%: {metrics.accuracy:.1f}%"
        assert metrics.high_confidence_rate >= 90, f"High confidence rate below 90%: {metrics.high_confidence_rate:.1f}%"
        assert metrics.average_detection_time < 0.1, f"Detection too slow: {metrics.average_detection_time:.3f}s"
        
        print(f"\nPython Detection Metrics:")
        print(f"  Accuracy: {metrics.accuracy:.1f}%")
        print(f"  High Confidence Rate: {metrics.high_confidence_rate:.1f}%")
        print(f"  Avg Detection Time: {metrics.average_detection_time*1000:.1f}ms")
        print(f"  Method Breakdown: {metrics.method_breakdown}")
    
    def test_javascript_detection_accuracy(self, detector: HybridLanguageDetector):
        """Test JavaScript detection accuracy across multiple samples."""
        metrics = AccuracyMetrics()
        
        for i, sample in enumerate(self.LANGUAGE_SAMPLES["javascript"]):
            start_time = time.perf_counter()
            result = detector.detect(sample, f"test{i}.js")
            detection_time = time.perf_counter() - start_time
            
            metrics.total_tests += 1
            metrics.average_detection_time += detection_time
            
            # Check if correctly detected as JavaScript
            if result.language == "javascript":
                metrics.correct_detections += 1
            
            # Track confidence levels
            if result.confidence > 0.9:
                metrics.high_confidence += 1
            elif result.confidence >= 0.7:
                metrics.medium_confidence += 1
            else:
                metrics.low_confidence += 1
            
            # Track detection method
            method_name = result.method.value
            metrics.method_breakdown[method_name] = metrics.method_breakdown.get(method_name, 0) + 1
            
            # Assertions for individual tests
            assert result.language == "javascript", f"Sample {i} not detected as JavaScript: {result.language}"
            assert result.confidence >= 0.9, f"Sample {i} confidence too low: {result.confidence}"
            assert result.method == DetectionMethod.TREE_SITTER, f"Sample {i} not using tree-sitter: {result.method}"
        
        # Calculate averages
        metrics.average_detection_time /= metrics.total_tests
        
        # Overall assertions
        assert metrics.accuracy >= 90, f"JavaScript accuracy below 90%: {metrics.accuracy:.1f}%"
        assert metrics.high_confidence_rate >= 90, f"High confidence rate below 90%: {metrics.high_confidence_rate:.1f}%"
        assert metrics.average_detection_time < 0.1, f"Detection too slow: {metrics.average_detection_time:.3f}s"
        
        print(f"\nJavaScript Detection Metrics:")
        print(f"  Accuracy: {metrics.accuracy:.1f}%")
        print(f"  High Confidence Rate: {metrics.high_confidence_rate:.1f}%")
        print(f"  Avg Detection Time: {metrics.average_detection_time*1000:.1f}ms")
        print(f"  Method Breakdown: {metrics.method_breakdown}")
    
    def test_java_detection_accuracy(self, detector: HybridLanguageDetector):
        """Test Java detection accuracy across multiple samples."""
        metrics = AccuracyMetrics()
        
        for i, sample in enumerate(self.LANGUAGE_SAMPLES["java"]):
            start_time = time.perf_counter()
            result = detector.detect(sample, f"Test{i}.java")
            detection_time = time.perf_counter() - start_time
            
            metrics.total_tests += 1
            metrics.average_detection_time += detection_time
            
            # Check if correctly detected as Java
            if result.language == "java":
                metrics.correct_detections += 1
            
            # Track confidence levels
            if result.confidence > 0.9:
                metrics.high_confidence += 1
            elif result.confidence >= 0.7:
                metrics.medium_confidence += 1
            else:
                metrics.low_confidence += 1
            
            # Track detection method
            method_name = result.method.value
            metrics.method_breakdown[method_name] = metrics.method_breakdown.get(method_name, 0) + 1
            
            # Assertions for individual tests
            assert result.language == "java", f"Sample {i} not detected as Java: {result.language}"
            assert result.confidence >= 0.9, f"Sample {i} confidence too low: {result.confidence}"
            assert result.method == DetectionMethod.TREE_SITTER, f"Sample {i} not using tree-sitter: {result.method}"
        
        # Calculate averages
        metrics.average_detection_time /= metrics.total_tests
        
        # Overall assertions
        assert metrics.accuracy >= 90, f"Java accuracy below 90%: {metrics.accuracy:.1f}%"
        assert metrics.high_confidence_rate >= 90, f"High confidence rate below 90%: {metrics.high_confidence_rate:.1f}%"
        assert metrics.average_detection_time < 0.1, f"Detection too slow: {metrics.average_detection_time:.3f}s"
        
        print(f"\nJava Detection Metrics:")
        print(f"  Accuracy: {metrics.accuracy:.1f}%")
        print(f"  High Confidence Rate: {metrics.high_confidence_rate:.1f}%")
        print(f"  Avg Detection Time: {metrics.average_detection_time*1000:.1f}ms")
        print(f"  Method Breakdown: {metrics.method_breakdown}")
    
    def test_kotlin_detection_accuracy(self, detector: HybridLanguageDetector):
        """Test Kotlin detection accuracy across multiple samples."""
        metrics = AccuracyMetrics()
        
        for i, sample in enumerate(self.LANGUAGE_SAMPLES["kotlin"]):
            start_time = time.perf_counter()
            result = detector.detect(sample, f"Test{i}.kt")
            detection_time = time.perf_counter() - start_time
            
            metrics.total_tests += 1
            metrics.average_detection_time += detection_time
            
            # Kotlin might be detected as kotlin, java, or python (no tree-sitter parser)
            # Accept kotlin detection via any method
            if result.language == "kotlin":
                metrics.correct_detections += 1
            
            # Track confidence levels
            if result.confidence > 0.9:
                metrics.high_confidence += 1
            elif result.confidence >= 0.7:
                metrics.medium_confidence += 1
            else:
                metrics.low_confidence += 1
            
            # Track detection method
            method_name = result.method.value
            metrics.method_breakdown[method_name] = metrics.method_breakdown.get(method_name, 0) + 1
            
            # Kotlin-specific assertions (more lenient due to lack of tree-sitter parser)
            assert result.language in ["kotlin", "java"], f"Sample {i} not detected as Kotlin/Java: {result.language}"
            assert result.confidence >= 0.7, f"Sample {i} confidence too low: {result.confidence}"
        
        # Calculate averages
        metrics.average_detection_time /= metrics.total_tests
        
        # Kotlin has lower requirements due to lack of tree-sitter parser
        # But file extension should help achieve good accuracy
        assert metrics.accuracy >= 70, f"Kotlin accuracy below 70%: {metrics.accuracy:.1f}%"
        assert metrics.average_detection_time < 0.1, f"Detection too slow: {metrics.average_detection_time:.3f}s"
        
        print(f"\nKotlin Detection Metrics:")
        print(f"  Accuracy: {metrics.accuracy:.1f}%")
        print(f"  High Confidence Rate: {metrics.high_confidence_rate:.1f}%")
        print(f"  Avg Detection Time: {metrics.average_detection_time*1000:.1f}ms")
        print(f"  Method Breakdown: {metrics.method_breakdown}")
    
    def test_cache_effectiveness(self, detector: HybridLanguageDetector):
        """Test cache effectiveness with repeated detections."""
        sample = self.LANGUAGE_SAMPLES["python"][0]
        
        # Clear cache statistics
        detector.cache._cache.clear()
        detector.cache.statistics = detector.cache.statistics.__class__()
        
        # First round - all cache misses
        detection_times = []
        for i in range(10):
            start_time = time.perf_counter()
            result = detector.detect(sample, f"test{i}.py")
            detection_times.append(time.perf_counter() - start_time)
            assert result.language == "python"
        
        cache_info = detector.get_cache_info()
        assert cache_info["statistics"]["misses"] == 10
        assert cache_info["statistics"]["hits"] == 0
        
        # Second round - all cache hits
        cached_times = []
        for i in range(10):
            start_time = time.perf_counter()
            result = detector.detect(sample, f"test{i}.py")
            cached_times.append(time.perf_counter() - start_time)
            assert result.language == "python"
        
        cache_info = detector.get_cache_info()
        assert cache_info["statistics"]["hits"] == 10
        assert cache_info["statistics"]["misses"] == 10
        assert cache_info["hit_rate"] == 0.5
        
        # Cached detections should be significantly faster
        avg_detection_time = sum(detection_times) / len(detection_times)
        avg_cached_time = sum(cached_times) / len(cached_times)
        
        # Cache should be at least 10x faster
        speedup = avg_detection_time / avg_cached_time
        assert speedup > 10, f"Cache speedup only {speedup:.1f}x"
        
        print(f"\nCache Effectiveness:")
        print(f"  Avg Detection Time: {avg_detection_time*1000:.2f}ms")
        print(f"  Avg Cached Time: {avg_cached_time*1000:.4f}ms")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Hit Rate: {cache_info['hit_rate']*100:.1f}%")
    
    def test_batch_processing_performance(self, detector: HybridLanguageDetector, temp_dir: Path):
        """Test performance of batch processing for multiple files."""
        # Create test files
        files_data = []
        for lang, samples in self.LANGUAGE_SAMPLES.items():
            ext = {"python": ".py", "javascript": ".js", "java": ".java", "kotlin": ".kt"}[lang]
            for i, sample in enumerate(samples):
                file_path = temp_dir / f"{lang}_sample_{i}{ext}"
                file_path.write_text(sample)
                files_data.append((str(file_path), sample))
        
        # Batch detection
        start_time = time.perf_counter()
        results = detector.detect_batch(files_data)
        batch_time = time.perf_counter() - start_time
        
        # Verify results
        assert len(results) == len(files_data)
        
        # Count correct detections
        correct = 0
        for (file_path, _), result in zip(files_data, results):
            expected_lang = Path(file_path).stem.split('_')[0]
            if result.language == expected_lang:
                correct += 1
            # Kotlin might be detected as Java
            elif expected_lang == "kotlin" and result.language == "java":
                correct += 1
        
        accuracy = (correct / len(results)) * 100
        avg_time_per_file = batch_time / len(files_data)
        
        print(f"\nBatch Processing Performance:")
        print(f"  Total Files: {len(files_data)}")
        print(f"  Total Time: {batch_time:.2f}s")
        print(f"  Avg Time per File: {avg_time_per_file*1000:.1f}ms")
        print(f"  Overall Accuracy: {accuracy:.1f}%")
        
        # Performance requirements
        assert avg_time_per_file < 0.1, f"Batch processing too slow: {avg_time_per_file:.3f}s per file"
        assert accuracy >= 85, f"Batch accuracy below 85%: {accuracy:.1f}%"
    
    def test_cross_language_benchmark(self, detector: HybridLanguageDetector):
        """Benchmark detection across all supported languages."""
        all_metrics = {}
        
        for language, samples in self.LANGUAGE_SAMPLES.items():
            metrics = AccuracyMetrics()
            
            for i, sample in enumerate(samples):
                ext = {"python": ".py", "javascript": ".js", "java": ".java", "kotlin": ".kt"}[language]
                
                start_time = time.perf_counter()
                result = detector.detect(sample, f"test{i}{ext}")
                detection_time = time.perf_counter() - start_time
                
                metrics.total_tests += 1
                metrics.average_detection_time += detection_time
                
                # Check if correctly detected
                if result.language == language:
                    metrics.correct_detections += 1
                # Special case for Kotlin
                elif language == "kotlin" and result.language in ["java", "kotlin"]:
                    metrics.correct_detections += 1
                
                # Track confidence levels
                if result.confidence > 0.9:
                    metrics.high_confidence += 1
                elif result.confidence >= 0.7:
                    metrics.medium_confidence += 1
                else:
                    metrics.low_confidence += 1
            
            metrics.average_detection_time /= metrics.total_tests
            all_metrics[language] = metrics
        
        # Generate benchmark report
        print("\n" + "="*60)
        print("LANGUAGE DETECTION ACCURACY BENCHMARK")
        print("="*60)
        
        for language, metrics in all_metrics.items():
            print(f"\n{language.upper()}:")
            print(f"  Accuracy: {metrics.accuracy:.1f}%")
            print(f"  High Confidence: {metrics.high_confidence_rate:.1f}%")
            print(f"  Avg Time: {metrics.average_detection_time*1000:.1f}ms")
        
        # Overall requirements
        total_accuracy = sum(m.accuracy for m in all_metrics.values()) / len(all_metrics)
        avg_detection_time = sum(m.average_detection_time for m in all_metrics.values()) / len(all_metrics)
        
        print(f"\nOVERALL:")
        print(f"  Average Accuracy: {total_accuracy:.1f}%")
        print(f"  Average Detection Time: {avg_detection_time*1000:.1f}ms")
        
        # Cache statistics
        cache_info = detector.get_cache_info()
        if cache_info:
            print(f"\nCACHE STATISTICS:")
            print(f"  Hit Rate: {cache_info.get('hit_rate', 0)*100:.1f}%")
            print(f"  Cache Size: {cache_info['size']}/{cache_info['max_size']}")
        
        print("="*60)
        
        # Overall assertions
        assert total_accuracy >= 85, f"Overall accuracy below 85%: {total_accuracy:.1f}%"
        assert avg_detection_time < 0.1, f"Average detection time above 100ms: {avg_detection_time*1000:.1f}ms"
    
    def test_detection_with_minimal_context(self, detector: HybridLanguageDetector):
        """Test detection accuracy with minimal code context."""
        minimal_samples = [
            ("print('hello')", "python", ["python", "ruby"]),
            ("console.log('hello')", "javascript", ["javascript"]),
            ("System.out.println('hello');", "java", ["java"]),
            ("println('hello')", "kotlin", ["kotlin", "groovy", "scala"]),
            ("def func():", "python", ["python"]),
            ("function test() {}", "javascript", ["javascript"]),
            ("public class Test {}", "java", ["java"]),
            ("fun main() {}", "kotlin", ["kotlin"]),
        ]
        
        for code, expected, acceptable in minimal_samples:
            result = detector.detect(code)
            assert result.language in acceptable, \
                f"'{code}' detected as {result.language}, expected one of {acceptable}"
            
            # Minimal context should have lower confidence
            assert result.confidence <= 0.95, \
                f"Confidence too high for minimal context: {result.confidence}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])