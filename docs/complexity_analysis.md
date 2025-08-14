# Language-Specific Complexity Analysis

## Overview

The Complexity Analysis System provides comprehensive code complexity metrics with language-specific understanding. It supports multiple programming languages and offers both cyclomatic and cognitive complexity measurements, along with maintainability indices and actionable recommendations.

## Architecture

### Analysis Pipeline

```
┌─────────────────────────────────────────┐
│            Source File                  │
│         (.py, .java, .kt)               │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│       Language Detection                │
│    (Determines analyzer)                │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│       Analyzer Registry                 │
│    (Selects language analyzer)          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│    Language-Specific Analyzer           │
│   (Python/Java/Kotlin)                  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│        AST Parsing                      │
│   (Tree-sitter or native)               │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│     Complexity Calculation              │
│  (Cyclomatic & Cognitive)               │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│    Metrics & Recommendations            │
│  (MI, grades, suggestions)              │
└─────────────────────────────────────────┘
```

### Key Components

- **`BaseComplexityAnalyzer`**: Abstract base class for all analyzers
- **`AnalyzerRegistry`**: Plugin system for language analyzers
- **`ComplexityMetrics`**: Language-agnostic metric calculations
- **`PythonComplexityAnalyzer`**: Python-specific analyzer
- **`JavaComplexityAnalyzer`**: Java-specific analyzer
- **`KotlinComplexityAnalyzer`**: Kotlin-specific analyzer

## Configuration

### Basic Setup

```python
from project_watch_mcp.complexity_analysis import AnalyzerRegistry

# Get analyzer for a language
registry = AnalyzerRegistry()
analyzer = registry.get_analyzer("python")

# Analyze a file
result = await analyzer.analyze_file("src/main.py")

print(f"Total Complexity: {result.summary.total_complexity}")
print(f"Average Complexity: {result.summary.average_complexity:.2f}")
print(f"Maintainability Index: {result.summary.maintainability_index:.2f}")
print(f"Grade: {result.summary.complexity_grade}")
```

### Supported Languages

| Language | Cyclomatic | Cognitive | Special Features |
|----------|------------|-----------|------------------|
| Python | ✅ | ✅ | Decorators, async, generators, comprehensions |
| Java | ✅ | ✅ | Lambdas, streams, try-with-resources |
| Kotlin | ✅ | ✅ | Data classes, coroutines, when expressions |

### Environment Variables

```bash
# Enable complexity analysis
export COMPLEXITY_ANALYSIS_ENABLED=true

# Complexity thresholds
export COMPLEXITY_THRESHOLD_LOW=5
export COMPLEXITY_THRESHOLD_MEDIUM=10
export COMPLEXITY_THRESHOLD_HIGH=20
export COMPLEXITY_THRESHOLD_VERY_HIGH=30

# Maintainability index thresholds
export MI_THRESHOLD_GOOD=20
export MI_THRESHOLD_MODERATE=10
export MI_THRESHOLD_POOR=5

# Analysis options
export COMPLEXITY_INCLUDE_METRICS=true
export COMPLEXITY_INCLUDE_RECOMMENDATIONS=true
```

## Complexity Metrics

### Cyclomatic Complexity

Measures the number of linearly independent paths through code:

```python
def calculate_complexity(value):
    complexity = 1  # Base complexity
    
    if value > 10:      # +1 for if
        return "high"
    elif value > 5:     # +1 for elif
        return "medium"
    else:               # +0 for else
        return "low"
    # Total complexity: 3
```

#### Complexity Rules by Language

**Python:**
- +1 for each: `if`, `elif`, `for`, `while`, `except`, `with`
- +1 for each boolean operator: `and`, `or`
- +1 for comprehensions and generator expressions
- +1 for lambda expressions

**Java:**
- +1 for each: `if`, `for`, `while`, `do-while`, `case`, `catch`
- +1 for each: `&&`, `||`, `?:`
- +1 for lambda expressions
- +1 for stream operations: `filter`, `map`, `flatMap`

**Kotlin:**
- +1 for each: `if`, `when` branch, `for`, `while`, `catch`
- +1 for each: `&&`, `||`, `?:`
- +2 for suspend functions (coroutines)
- +1 per nested lambda

### Cognitive Complexity

Measures how difficult code is for humans to understand:

```python
def complex_function(data):
    result = []
    
    for item in data:           # +1 (depth 0)
        if item.is_valid():     # +2 (depth 1, +1 for nesting)
            for sub in item:    # +3 (depth 2, +2 for nesting)
                if sub > 0:     # +4 (depth 3, +3 for nesting)
                    result.append(sub)
    
    return result
    # Cognitive complexity: 10
```

#### Cognitive Complexity Rules

1. **Increments** for flow-breaking structures
2. **Nesting penalty** increases with depth
3. **No increment** for simple structures like `else`

### Maintainability Index

Composite metric combining complexity, lines of code, and comments:

```
MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * Cyclomatic Complexity - 16.2 * ln(Lines of Code)
```

Normalized to 0-100 scale:
- **>20**: Highly maintainable ✅
- **10-20**: Moderate maintainability ⚠️
- **<10**: Low maintainability ❌

## Usage Examples

### Basic Analysis

```python
# Analyze a single file
from project_watch_mcp.complexity_analysis import analyze_complexity

result = await analyze_complexity(
    file_path="src/auth/authentication.py",
    include_metrics=True
)

# Check overall health
if result.summary.maintainability_index < 10:
    print("⚠️ Low maintainability - consider refactoring")

# Find complex functions
for func in result.functions:
    if func.complexity > 10:
        print(f"Complex function: {func.name} (line {func.line})")
        print(f"  Complexity: {func.complexity}")
        print(f"  Classification: {func.classification}")
```

### Batch Analysis

```python
# Analyze entire directory
async def analyze_directory(path: str):
    results = {}
    
    for file in Path(path).rglob("*.py"):
        try:
            result = await analyze_complexity(str(file))
            results[str(file)] = result
        except Exception as e:
            print(f"Failed to analyze {file}: {e}")
    
    # Summary statistics
    total_complexity = sum(
        r.summary.total_complexity for r in results.values()
    )
    avg_maintainability = np.mean([
        r.summary.maintainability_index for r in results.values()
    ])
    
    print(f"Total files: {len(results)}")
    print(f"Total complexity: {total_complexity}")
    print(f"Average maintainability: {avg_maintainability:.2f}")
    
    return results
```

### Language-Specific Analysis

```python
# Python-specific features
python_result = await analyze_complexity("app.py")
for func in python_result.functions:
    if func.has_decorators:
        print(f"{func.name} uses decorators")
    if func.is_async:
        print(f"{func.name} is async")
    if func.has_generators:
        print(f"{func.name} uses generators")

# Java-specific features
java_result = await analyze_complexity("Main.java")
for func in java_result.functions:
    if func.has_lambdas:
        print(f"{func.name} uses lambdas")
    if func.has_streams:
        print(f"{func.name} uses streams")

# Kotlin-specific features
kotlin_result = await analyze_complexity("App.kt")
for cls in kotlin_result.classes:
    if cls.is_data_class:
        print(f"{cls.name} is a data class")
    if cls.is_sealed:
        print(f"{cls.name} is sealed")
```

## Language-Specific Features

### Python Analyzer

```python
class PythonComplexityAnalyzer(BaseComplexityAnalyzer):
    """
    Handles Python-specific constructs:
    - Decorators: Tracks but doesn't add complexity
    - Async functions: +1 complexity
    - Generators: +1 per yield
    - Comprehensions: +1 per comprehension
    - Context managers: +1 per with statement
    - Lambda functions: +1 each
    - Walrus operator: +1 per usage
    - Pattern matching: +1 per case
    """
    
    def analyze_function(self, node):
        complexity = 1  # Base
        
        # Check for decorators
        decorators = self.get_decorators(node)
        
        # Check if async
        is_async = node.type == "async_function_definition"
        if is_async:
            complexity += 1
        
        # Analyze body
        complexity += self.analyze_block(node.body)
        
        return FunctionComplexity(
            name=self.get_function_name(node),
            complexity=complexity,
            cognitive_complexity=self.calculate_cognitive(node),
            line=node.start_point[0],
            has_decorators=bool(decorators),
            is_async=is_async,
            # ... other Python-specific attributes
        )
```

### Java Analyzer

```python
class JavaComplexityAnalyzer(BaseComplexityAnalyzer):
    """
    Handles Java-specific constructs:
    - Switch statements: +1 per case
    - Try-catch blocks: +1 per catch
    - Lambda expressions: +1 each
    - Stream operations: +1 per intermediate operation
    - Enhanced for loops: +1
    - Ternary operators: +1
    - Anonymous classes: +2
    """
    
    def count_stream_operations(self, node):
        operations = ["filter", "map", "flatMap", "distinct", 
                     "sorted", "peek", "limit", "skip"]
        count = 0
        
        for child in node.children:
            if child.type == "method_invocation":
                method_name = self.get_method_name(child)
                if method_name in operations:
                    count += 1
        
        return count
```

### Kotlin Analyzer

```python
class KotlinComplexityAnalyzer(BaseComplexityAnalyzer):
    """
    Handles Kotlin-specific constructs:
    - When expressions: +1 per branch
    - Data classes: Lower base complexity (0 instead of 1)
    - Extension functions: Normal complexity
    - Coroutines: +2 for suspend functions
    - Sealed classes: +1 per subclass
    - Elvis operator: +1
    - Safe calls: No additional complexity
    """
    
    def analyze_when_expression(self, node):
        complexity = 1  # Base for when
        
        # Count branches
        branches = self.get_when_branches(node)
        complexity += len(branches) - 1  # -1 because base already counts 1
        
        return complexity
```

## Recommendations System

### Automatic Recommendations

```python
def generate_recommendations(result: ComplexityResult) -> List[str]:
    recommendations = []
    
    # File-level recommendations
    if result.summary.average_complexity > 10:
        recommendations.append(
            "High average complexity. Consider breaking down complex functions."
        )
    
    if result.summary.maintainability_index < 20:
        recommendations.append(
            "Low maintainability index. Refactoring recommended."
        )
    
    # Function-level recommendations
    complex_functions = [
        f for f in result.functions if f.complexity > 10
    ]
    
    if complex_functions:
        recommendations.append(
            f"Consider refactoring {len(complex_functions)} complex functions:"
        )
        for func in complex_functions[:3]:  # Top 3
            recommendations.append(
                f"  - {func.name} (complexity: {func.complexity})"
            )
    
    # Language-specific recommendations
    if result.language == "python":
        async_functions = [f for f in result.functions if f.is_async]
        if len(async_functions) > 10:
            recommendations.append(
                "Many async functions. Ensure proper error handling."
            )
    
    return recommendations
```

### Complexity Classifications

| Complexity | Classification | Recommendation |
|------------|---------------|----------------|
| 1-5 | Simple | Good - no action needed |
| 6-10 | Moderate | Acceptable - monitor |
| 11-20 | Complex | Consider refactoring |
| 21-30 | Very Complex | Refactoring recommended |
| 31+ | Extremely Complex | Urgent refactoring needed |

## Integration with Other Features

### With Language Detection

```python
# Automatic analyzer selection
async def analyze_with_detection(file_path: str):
    # Detect language
    content = read_file(file_path)
    lang_result = await detector.detect(content, file_path)
    
    # Get appropriate analyzer
    analyzer = registry.get_analyzer(lang_result.language)
    
    if analyzer:
        return await analyzer.analyze_file(file_path)
    else:
        print(f"No analyzer for {lang_result.language}")
        return None
```

### With Vector Search

```python
# Find and analyze complex code
async def find_complex_code(query: str, complexity_threshold: int = 15):
    # Search for relevant code
    search_results = await search_code(
        query=query,
        search_type="semantic",
        limit=20
    )
    
    complex_files = []
    
    for result in search_results:
        # Analyze complexity
        analysis = await analyze_complexity(result["file"])
        
        # Check if complex
        if analysis.summary.total_complexity > complexity_threshold:
            complex_files.append({
                "file": result["file"],
                "complexity": analysis.summary.total_complexity,
                "functions": [
                    f for f in analysis.functions 
                    if f.complexity > 10
                ]
            })
    
    return complex_files
```

## Performance Optimization

### Caching Analysis Results

```python
from functools import lru_cache
import hashlib

class CachedAnalyzer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.cache = {}
    
    async def analyze_with_cache(self, file_path: str):
        # Generate cache key from file content
        content = read_file(file_path)
        cache_key = hashlib.md5(content.encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = await self.analyzer.analyze_file(file_path)
        self.cache[cache_key] = result
        
        return result
```

### Parallel Analysis

```python
import asyncio

async def analyze_repository(repo_path: str):
    files = list(Path(repo_path).rglob("*.py"))
    
    # Analyze in parallel
    tasks = [
        analyze_complexity(str(file))
        for file in files
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [(f, r) for f, r in zip(files, results) 
              if isinstance(r, Exception)]
    
    print(f"Analyzed {len(successful)} files successfully")
    print(f"Failed to analyze {len(failed)} files")
    
    return successful, failed
```

## Visualization

### Complexity Heatmap

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_complexity_heatmap(results: Dict[str, ComplexityResult]):
    # Prepare data
    data = []
    for file, result in results.items():
        for func in result.functions:
            data.append({
                "file": Path(file).name,
                "function": func.name,
                "complexity": func.complexity
            })
    
    # Create pivot table
    df = pd.DataFrame(data)
    pivot = df.pivot_table(
        index="file",
        columns="function",
        values="complexity",
        fill_value=0
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "Complexity"}
    )
    plt.title("Code Complexity Heatmap")
    plt.tight_layout()
    plt.show()
```

### Trend Analysis

```python
def track_complexity_over_time(repo_path: str, num_days: int = 30):
    results = []
    
    for day in range(num_days):
        # Checkout revision from 'day' days ago
        checkout_revision(repo_path, days_ago=day)
        
        # Analyze complexity
        analysis = analyze_repository(repo_path)
        
        total_complexity = sum(
            r.summary.total_complexity for r in analysis
        )
        
        results.append({
            "date": datetime.now() - timedelta(days=day),
            "total_complexity": total_complexity,
            "num_files": len(analysis)
        })
    
    # Plot trend
    df = pd.DataFrame(results)
    df.plot(x="date", y="total_complexity", kind="line")
    plt.title("Complexity Trend Over Time")
    plt.show()
    
    return results
```

## Best Practices

1. **Set team-specific thresholds** based on your coding standards
2. **Run analysis in CI/CD** to catch complexity increases early
3. **Focus on trends** rather than absolute numbers
4. **Prioritize refactoring** of very complex functions (>20)
5. **Use cognitive complexity** for code review decisions
6. **Track maintainability index** over time
7. **Combine with test coverage** for refactoring safety
8. **Document complex algorithms** that legitimately need high complexity

## Troubleshooting

### Common Issues

1. **Parser Errors**
   ```python
   # Solution: Ensure code is syntactically valid
   try:
       result = await analyze_complexity(file_path)
   except SyntaxError as e:
       print(f"Syntax error in {file_path}: {e}")
   ```

2. **Missing Language Support**
   ```python
   # Solution: Check supported languages
   supported = registry.get_supported_languages()
   if language not in supported:
       print(f"Language {language} not supported")
   ```

3. **Performance Issues**
   ```python
   # Solution: Use caching or parallel processing
   cached_analyzer = CachedAnalyzer(analyzer)
   result = await cached_analyzer.analyze_with_cache(file_path)
   ```

## API Reference

See the [API Documentation](./api/complexity_analysis.md) for detailed class and method documentation.