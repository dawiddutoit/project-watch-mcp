# Advanced Features

Project Watch MCP includes three powerful advanced features that enhance code analysis and search capabilities beyond traditional text-based approaches. These features work together to provide a comprehensive understanding of your codebase.

## Overview

### 1. Neo4j Native Vector Search
Eliminates traditional Lucene-based search limitations by using Neo4j's native vector indexes. This provides:
- **Better Performance**: Direct vector similarity searches without text escaping overhead
- **Higher Accuracy**: True semantic similarity using cosine or Euclidean distance metrics
- **No Escaping Issues**: Completely eliminates Lucene special character problems
- **Flexible Dimensions**: Support for various embedding models (OpenAI, Voyage, etc.)

### 2. Multi-Language Detection System
A hybrid approach combining multiple detection methods for accurate language identification:
- **Tree-sitter Parsing**: AST-based detection for syntactic accuracy
- **Pygments Analysis**: Lexical analysis for broader language support
- **Extension Fallback**: File extension-based detection as final fallback
- **Confidence Scoring**: Each detection method provides confidence scores
- **Caching Layer**: Efficient caching for repeated file analysis

### 3. Language-Specific Complexity Analysis
Comprehensive code complexity analysis with language-specific understanding:
- **Multi-Language Support**: Python, Java, Kotlin analyzers
- **Cyclomatic Complexity**: Measure independent paths through code
- **Cognitive Complexity**: Human-oriented complexity measurement
- **Maintainability Index**: Overall code maintainability scoring
- **Language-Specific Features**: Handles language idioms correctly

## Feature Integration

These features work synergistically:

1. **Language Detection** identifies the programming language
2. **Complexity Analysis** uses language-specific rules for accurate metrics
3. **Vector Search** can be enriched with language-specific embeddings

This integration provides:
- More accurate semantic search results
- Language-aware code analysis
- Better code quality insights
- Improved repository understanding

## Configuration

### Environment Variables

```bash
# Vector Search Configuration
export NEO4J_VECTOR_INDEX_ENABLED=true
export VECTOR_SIMILARITY_METRIC=cosine  # or euclidean
export EMBEDDING_DIMENSION=1536  # Based on your embedding model

# Language Detection Configuration
export LANGUAGE_DETECTION_CACHE_SIZE=1000
export LANGUAGE_DETECTION_CACHE_TTL=3600  # seconds
export LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD=0.85

# Complexity Analysis Configuration
export COMPLEXITY_ANALYSIS_ENABLED=true
export COMPLEXITY_THRESHOLD_HIGH=10
export COMPLEXITY_THRESHOLD_VERY_HIGH=20
```

### Programmatic Configuration

```python
from project_watch_mcp.config import Config

config = Config(
    # Vector search settings
    vector_index_enabled=True,
    vector_similarity_metric="cosine",
    
    # Language detection settings
    language_cache_enabled=True,
    language_confidence_threshold=0.85,
    
    # Complexity analysis settings
    complexity_enabled=True,
    complexity_include_metrics=True
)
```

## Usage Examples

### Combined Feature Usage

```python
# Initialize repository with all features
await initialize_repository()

# Search for complex Python authentication code
results = await search_code(
    query="authentication and password hashing",
    search_type="semantic",
    language="python"  # Language-filtered search
)

# Analyze complexity of found files
for result in results:
    complexity = await analyze_complexity(
        file_path=result["file"],
        include_metrics=True
    )
    
    if complexity["summary"]["average_complexity"] > 10:
        print(f"High complexity file: {result['file']}")
        print(f"  Maintainability: {complexity['summary']['maintainability_index']}")
        print(f"  Grade: {complexity['summary']['complexity_grade']}")
```

### Language-Aware Repository Analysis

```python
# Get repository statistics with language breakdown
stats = await get_repository_stats()

# Analyze complexity by language
for language, info in stats["languages"].items():
    print(f"\n{language.upper()} Analysis:")
    print(f"  Files: {info['files']}")
    print(f"  Size: {info['size']} bytes")
    print(f"  Percentage: {info['percentage']}%")
    
    # Get average complexity for this language
    # (Feature automatically detects analyzer based on language)
```

## Performance Characteristics

### Vector Search Performance
- **Index Creation**: ~100ms per 1000 vectors
- **Search Latency**: <50ms for top-k similarity search
- **Memory Usage**: Approximately 4KB per vector (1536 dimensions)
- **Scalability**: Tested up to 1M vectors with linear performance

### Language Detection Performance
- **Single File**: <10ms with cache hit, <100ms without
- **Batch Processing**: ~1000 files/second with caching
- **Cache Hit Rate**: Typically >90% in normal usage
- **Memory Overhead**: ~1KB per cached file

### Complexity Analysis Performance
- **Small Files (<500 lines)**: <50ms
- **Medium Files (500-2000 lines)**: <200ms
- **Large Files (>2000 lines)**: <500ms
- **Memory Usage**: Minimal, analysis is streaming

## Extension Points

### Adding New Embedding Providers

```python
from project_watch_mcp.utils.embeddings.base import EmbeddingsProvider

class CustomEmbeddingsProvider(EmbeddingsProvider):
    def embed(self, text: str) -> List[float]:
        # Your embedding logic
        pass
    
    def to_neo4j_vector(self, embedding: List[float]) -> Any:
        # Convert to Neo4j native format
        return np.array(embedding, dtype=np.float32)
```

### Adding New Language Analyzers

```python
from project_watch_mcp.complexity_analysis.base_analyzer import BaseComplexityAnalyzer

class RustComplexityAnalyzer(BaseComplexityAnalyzer):
    def __init__(self):
        super().__init__("rust")
    
    def calculate_cyclomatic_complexity(self, source_code: str) -> int:
        # Rust-specific complexity calculation
        pass
```

### Custom Language Detection Methods

```python
from project_watch_mcp.language_detection.models import LanguageDetectionResult

class CustomDetector:
    def detect(self, content: str, filepath: str) -> LanguageDetectionResult:
        # Your detection logic
        return LanguageDetectionResult(
            language="rust",
            confidence=0.95,
            method="custom"
        )
```

## Best Practices

### 1. Vector Search Optimization
- Use appropriate embedding dimensions for your use case
- Consider cosine similarity for normalized embeddings
- Batch vector operations when possible
- Monitor index size and performance

### 2. Language Detection Tuning
- Adjust confidence thresholds based on your codebase
- Pre-warm cache for frequently accessed files
- Use tree-sitter parsers when available for accuracy
- Fall back gracefully for unknown languages

### 3. Complexity Analysis Guidelines
- Set appropriate complexity thresholds for your team
- Use cognitive complexity for code review decisions
- Monitor maintainability index trends over time
- Focus refactoring on high-complexity hotspots

## Troubleshooting

### Vector Search Issues
- **Problem**: Slow similarity searches
  - **Solution**: Ensure Neo4j has sufficient memory allocated
  - **Solution**: Consider reducing embedding dimensions

- **Problem**: Poor search relevance
  - **Solution**: Verify embedding model quality
  - **Solution**: Check vector normalization

### Language Detection Issues
- **Problem**: Incorrect language detection
  - **Solution**: Install language-specific tree-sitter parsers
  - **Solution**: Adjust confidence thresholds

- **Problem**: Slow detection performance
  - **Solution**: Enable caching
  - **Solution**: Increase cache size

### Complexity Analysis Issues
- **Problem**: Inaccurate complexity scores
  - **Solution**: Ensure correct language analyzer is used
  - **Solution**: Update to latest analyzer version

- **Problem**: Missing language support
  - **Solution**: Implement custom analyzer
  - **Solution**: Use base analyzer as fallback

## Migration Guide

### From Lucene to Native Vectors

1. **Update Configuration**:
   ```python
   config.vector_index_enabled = True
   config.search_backend = "neo4j_vector"
   ```

2. **Re-index Repository**:
   ```bash
   project-watch-mcp --initialize --force-reindex
   ```

3. **Update Search Queries**:
   ```python
   # Old Lucene search
   results = search_code(query="user AND authentication")
   
   # New vector search
   results = search_code(
       query="user authentication logic",
       search_type="semantic"
   )
   ```

## Feature Roadmap

### Planned Enhancements
- Additional language analyzers (Go, Rust, Swift)
- Multi-modal embeddings (code + comments)
- Incremental complexity tracking
- Cross-language dependency analysis
- Real-time complexity monitoring

### Community Contributions
We welcome contributions for:
- New language analyzers
- Embedding provider integrations
- Performance optimizations
- Documentation improvements

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.