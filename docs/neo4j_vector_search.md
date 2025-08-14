# Neo4j Native Vector Search

## Overview

Neo4j Native Vector Search replaces traditional Lucene-based full-text search with direct vector similarity operations. This eliminates common issues with special character escaping while providing superior semantic search capabilities.

## Architecture

### Components

```
┌─────────────────────────────────────────┐
│          Embedding Provider             │
│  (OpenAI, Voyage, Custom)               │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│       Native Vector Converter           │
│  (Float32 arrays, normalization)        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│        Neo4j Vector Index                │
│  (Cosine/Euclidean similarity)          │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│       Similarity Search Engine          │
│  (k-NN, metadata filtering)             │
└─────────────────────────────────────────┘
```

### Key Classes

- **`NativeVectorIndex`**: Manages Neo4j vector index operations
- **`VectorIndexConfig`**: Configuration for vector indexes
- **`VectorSearchResult`**: Encapsulates search results with scores
- **`EmbeddingsProvider`**: Base class with native vector support

## Configuration

### Basic Setup

```python
from project_watch_mcp.vector_search import NativeVectorIndex, VectorIndexConfig

# Configure vector index
config = VectorIndexConfig(
    index_name="code-embeddings",
    node_label="CodeChunk",
    embedding_property="embedding",
    dimensions=1536,  # OpenAI text-embedding-3-small
    similarity_metric="cosine",
    provider="openai"
)

# Initialize index
vector_index = NativeVectorIndex(driver, config)
await vector_index.create_index()
```

### Supported Configurations

| Provider | Model | Dimensions | Recommended Metric |
|----------|-------|------------|-------------------|
| OpenAI | text-embedding-3-small | 1536 | cosine |
| OpenAI | text-embedding-3-large | 3072 | cosine |
| Voyage | voyage-code-2 | 1024 | cosine |
| Voyage | voyage-code-2-lite | 2048 | cosine |
| Custom | Any | Variable | cosine/euclidean |

### Environment Variables

```bash
# Enable native vector search
export NEO4J_VECTOR_INDEX_ENABLED=true

# Configure similarity metric
export VECTOR_SIMILARITY_METRIC=cosine  # or euclidean

# Set embedding dimensions
export EMBEDDING_DIMENSION=1536

# Configure provider
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

## Usage

### Creating and Managing Indexes

```python
# Create a new vector index
await vector_index.create_index()

# Check if index exists
exists = await vector_index.index_exists()

# Drop existing index
await vector_index.drop_index()

# Get index statistics
stats = await vector_index.get_index_stats()
print(f"Total vectors: {stats['vector_count']}")
print(f"Index size: {stats['size_mb']} MB")
```

### Inserting Vectors

```python
# Single vector insertion
embedding = await embedding_provider.embed("def authenticate_user():")
await vector_index.upsert_vector(
    node_id="chunk_123",
    vector=embedding,
    metadata={
        "file_path": "auth.py",
        "language": "python",
        "line_number": 42
    }
)

# Batch insertion for efficiency
vectors = [
    {
        "node_id": "chunk_1",
        "vector": embedding1,
        "metadata": {"file": "main.py"}
    },
    {
        "node_id": "chunk_2",
        "vector": embedding2,
        "metadata": {"file": "utils.py"}
    }
]
await vector_index.batch_upsert(vectors)
```

### Searching

```python
# Basic similarity search
query_embedding = await embedding_provider.embed(
    "user authentication and JWT validation"
)
results = await vector_index.search(
    vector=query_embedding,
    limit=10
)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"File: {result.metadata['file_path']}")
    print(f"Content: {result.content[:100]}...")

# Search with metadata filtering
results = await vector_index.search(
    vector=query_embedding,
    limit=5,
    filters={
        "language": "python",
        "file_path": {"$contains": "auth"}
    }
)

# Hybrid search (vector + metadata)
results = await vector_index.hybrid_search(
    vector=query_embedding,
    metadata_query={"complexity": {"$gt": 10}},
    vector_weight=0.7,  # 70% vector, 30% metadata
    limit=10
)
```

## Performance Optimization

### Index Optimization

```python
# Optimize index for better query performance
await vector_index.optimize()

# Rebuild index if fragmented
await vector_index.rebuild()

# Configure index parameters
config = VectorIndexConfig(
    # ... other settings ...
    index_params={
        "m": 16,  # Number of bi-directional links
        "ef_construction": 200,  # Size of dynamic candidate list
        "ef_search": 50  # Size of dynamic candidate list for search
    }
)
```

### Batch Operations

```python
# Efficient batch processing
async def index_repository(repo_path: str):
    chunks = []
    
    # Collect chunks
    for file in get_code_files(repo_path):
        content = read_file(file)
        file_chunks = split_into_chunks(content)
        
        for chunk in file_chunks:
            embedding = await embedding_provider.embed(chunk.text)
            chunks.append({
                "node_id": chunk.id,
                "vector": embedding,
                "metadata": {
                    "file": file,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line
                }
            })
            
            # Batch insert every 100 chunks
            if len(chunks) >= 100:
                await vector_index.batch_upsert(chunks)
                chunks = []
    
    # Insert remaining chunks
    if chunks:
        await vector_index.batch_upsert(chunks)
```

### Caching Strategies

```python
from functools import lru_cache
import hashlib

class CachedVectorSearch:
    def __init__(self, vector_index):
        self.vector_index = vector_index
        self.cache = {}
    
    async def search_with_cache(self, query: str, limit: int = 10):
        # Generate cache key
        cache_key = hashlib.md5(
            f"{query}:{limit}".encode()
        ).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Perform search
        embedding = await embedding_provider.embed(query)
        results = await self.vector_index.search(embedding, limit)
        
        # Cache results
        self.cache[cache_key] = results
        return results
```

## Advanced Features

### Custom Similarity Metrics

```python
# Implement custom similarity calculation
class CustomVectorIndex(NativeVectorIndex):
    async def custom_similarity(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray
    ) -> float:
        # Example: Weighted cosine similarity
        weights = self.get_feature_weights()
        weighted_v1 = vector1 * weights
        weighted_v2 = vector2 * weights
        
        dot_product = np.dot(weighted_v1, weighted_v2)
        norm1 = np.linalg.norm(weighted_v1)
        norm2 = np.linalg.norm(weighted_v2)
        
        return dot_product / (norm1 * norm2)
```

### Multi-Modal Embeddings

```python
# Combine code and documentation embeddings
class MultiModalEmbedding:
    async def create_combined_embedding(
        self,
        code: str,
        documentation: str
    ) -> np.ndarray:
        # Get individual embeddings
        code_embedding = await code_provider.embed(code)
        doc_embedding = await doc_provider.embed(documentation)
        
        # Concatenate with weighting
        combined = np.concatenate([
            code_embedding * 0.7,  # 70% weight to code
            doc_embedding * 0.3    # 30% weight to docs
        ])
        
        # Normalize to unit length
        return combined / np.linalg.norm(combined)
```

### Incremental Updates

```python
# Efficiently update existing vectors
async def update_file_vectors(file_path: str):
    # Get existing chunks for file
    existing_chunks = await vector_index.get_file_chunks(file_path)
    
    # Parse updated file
    new_chunks = parse_file(file_path)
    
    # Determine changes
    to_delete = set(existing_chunks.keys()) - set(new_chunks.keys())
    to_update = set(existing_chunks.keys()) & set(new_chunks.keys())
    to_insert = set(new_chunks.keys()) - set(existing_chunks.keys())
    
    # Apply changes
    if to_delete:
        await vector_index.delete_vectors(list(to_delete))
    
    updates = []
    for chunk_id in to_update:
        if new_chunks[chunk_id].content != existing_chunks[chunk_id].content:
            embedding = await embedding_provider.embed(
                new_chunks[chunk_id].content
            )
            updates.append({
                "node_id": chunk_id,
                "vector": embedding,
                "metadata": new_chunks[chunk_id].metadata
            })
    
    if updates:
        await vector_index.batch_upsert(updates)
    
    inserts = []
    for chunk_id in to_insert:
        embedding = await embedding_provider.embed(
            new_chunks[chunk_id].content
        )
        inserts.append({
            "node_id": chunk_id,
            "vector": embedding,
            "metadata": new_chunks[chunk_id].metadata
        })
    
    if inserts:
        await vector_index.batch_upsert(inserts)
```

## Comparison with Lucene

### Advantages of Native Vectors

| Feature | Lucene Full-Text | Neo4j Native Vectors |
|---------|-----------------|---------------------|
| Semantic Search | Limited | Excellent |
| Special Characters | Requires escaping | No escaping needed |
| Query Complexity | Complex syntax | Simple similarity |
| Performance | Text parsing overhead | Direct vector ops |
| Scalability | Good | Excellent |
| Maintenance | Index rebuilding | Incremental updates |

### Migration from Lucene

```python
# Before: Lucene search with escaping
def search_lucene(query: str):
    # Escape special characters
    escaped = escape_lucene_chars(query)
    return neo4j.run(
        "CALL db.index.fulltext.queryNodes($query)",
        query=escaped
    )

# After: Native vector search
async def search_vectors(query: str):
    embedding = await embedding_provider.embed(query)
    return await vector_index.search(embedding)
```

## Monitoring and Metrics

### Performance Metrics

```python
# Track search performance
class VectorSearchMetrics:
    def __init__(self):
        self.search_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def monitored_search(self, query: str):
        start_time = time.time()
        
        results = await vector_index.search(query)
        
        elapsed = time.time() - start_time
        self.search_times.append(elapsed)
        
        # Log if slow
        if elapsed > 1.0:
            logger.warning(f"Slow search: {elapsed:.2f}s for '{query}'")
        
        return results
    
    def get_stats(self):
        return {
            "avg_search_time": np.mean(self.search_times),
            "p95_search_time": np.percentile(self.search_times, 95),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
        }
```

### Index Health Monitoring

```python
# Monitor index health
async def check_index_health():
    stats = await vector_index.get_index_stats()
    
    health_checks = {
        "index_exists": await vector_index.index_exists(),
        "vector_count": stats["vector_count"],
        "index_size_mb": stats["size_mb"],
        "avg_vector_dimension": stats["avg_dimension"],
        "null_vectors": stats["null_count"],
        "last_updated": stats["last_modified"]
    }
    
    # Alert on issues
    if health_checks["null_vectors"] > 0:
        logger.warning(f"Found {health_checks['null_vectors']} null vectors")
    
    if health_checks["index_size_mb"] > 1000:  # 1GB
        logger.warning(f"Large index size: {health_checks['index_size_mb']}MB")
    
    return health_checks
```

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**
   ```python
   # Error: Vector dimension 1536 doesn't match index dimension 3072
   # Solution: Ensure embedding model matches index configuration
   config = VectorIndexConfig(dimensions=1536)  # Match your model
   ```

2. **Memory Issues**
   ```python
   # Error: Out of memory during batch insert
   # Solution: Reduce batch size
   BATCH_SIZE = 50  # Instead of 1000
   ```

3. **Slow Searches**
   ```python
   # Solution: Optimize index parameters
   config.index_params = {
       "ef_search": 100  # Increase for better accuracy
   }
   ```

4. **Poor Search Quality**
   ```python
   # Solution: Normalize vectors
   vector = embedding / np.linalg.norm(embedding)
   ```

## Best Practices

1. **Always normalize vectors** for cosine similarity
2. **Use batch operations** for bulk inserts
3. **Monitor index size** and optimize regularly
4. **Cache frequently searched queries**
5. **Use appropriate embedding models** for your domain
6. **Implement incremental updates** instead of full reindexing
7. **Set up monitoring** for search performance
8. **Test different similarity metrics** for your use case

## API Reference

See the [API Documentation](./api/vector_search.md) for detailed class and method documentation.