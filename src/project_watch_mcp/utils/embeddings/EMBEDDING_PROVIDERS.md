# Embedding Providers for Project Watch MCP

## Overview

Project Watch MCP now supports multiple embedding providers for semantic code search, allowing you to choose between different providers based on your needs and benchmarking results.

## Supported Providers

### 1. OpenAI Embeddings
- **Model**: text-embedding-3-small (default)
- **Dimensions**: 1536
- **Best for**: General purpose text and code embeddings
- **API Key**: Set `OPENAI_API_KEY` environment variable

### 2. Voyage AI Embeddings
- **Model**: voyage-code-3 (default, optimized for code)
- **Dimensions**: 1024
- **Best for**: Code-specific embeddings with better performance on code search
- **API Key**: Set `VOYAGE_API_KEY` environment variable
- **Alternative models**:
  - `voyage-3`: General purpose (1024 dimensions)
  - `voyage-3-lite`: Lightweight model (512 dimensions)

### 3. Mock Embeddings
- **Dimensions**: Configurable (default 1536)
- **Best for**: Testing and development without API costs
- **No API key required**

## Configuration

### Environment Variables

```bash
# Choose provider (openai, voyage, mock)
export EMBEDDING_PROVIDER=voyage

# For OpenAI
export OPENAI_API_KEY=your-openai-key
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# For Voyage AI
export VOYAGE_API_KEY=your-voyage-key
export VOYAGE_EMBEDDING_MODEL=voyage-code-3

# Override dimensions if needed
export EMBEDDING_DIMENSION=1024
```

### Programmatic Configuration

```python
from project_watch_mcp.config import EmbeddingConfig
from project_watch_mcp.neo4j_rag import Neo4jRAG

# Using configuration
config = EmbeddingConfig(
    provider="voyage",
    model="voyage-code-3",
    api_key="your-api-key"
)

rag = Neo4jRAG(
    neo4j_driver=driver,
    project_name="my-project",
    embedding_config=config
)

# Or direct provider instantiation
from project_watch_mcp.utils.embedding import VoyageEmbeddingsProvider

provider = VoyageEmbeddingsProvider(
    api_key="your-api-key",
    model="voyage-code-3"
)

rag = Neo4jRAG(
    neo4j_driver=driver,
    project_name="my-project",
    embeddings=provider
)
```

## Switching Between Providers

### Method 1: Environment Variable (Easiest)

```bash
# Switch to Voyage
export EMBEDDING_PROVIDER=voyage
export VOYAGE_API_KEY=your-key

# Switch to OpenAI
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=your-key

# Run your application
project-watch-mcp
```

### Method 2: Factory Function

```python
from project_watch_mcp.utils.embedding import create_embeddings_provider

# Create providers dynamically
openai_provider = create_embeddings_provider("openai")
voyage_provider = create_embeddings_provider("voyage")
mock_provider = create_embeddings_provider("mock")
```

## Benchmarking Providers

### Simple Benchmark Script

```python
import asyncio
import time
from project_watch_mcp.utils.embedding import (
    OpenAIEmbeddingsProvider,
    VoyageEmbeddingsProvider,
    MockEmbeddingsProvider
)

async def benchmark_provider(provider, name, test_texts):
    """Benchmark a single provider."""
    start_time = time.time()
    embeddings = []
    
    for text in test_texts:
        embedding = await provider.embed_text(text)
        embeddings.append(embedding)
    
    elapsed = time.time() - start_time
    
    print(f"{name} Provider:")
    print(f"  - Time: {elapsed:.2f}s")
    print(f"  - Dimension: {provider.dimension}")
    print(f"  - Avg time per text: {elapsed/len(test_texts):.3f}s")
    
    return embeddings

async def main():
    # Sample code snippets for testing
    test_texts = [
        "def hello_world():\n    return 'Hello, World!'",
        "class UserAuthentication:\n    def validate_token(self, token):\n        pass",
        "async function fetchData(url) { return await fetch(url); }",
        # Add more test cases
    ]
    
    providers = [
        (OpenAIEmbeddingsProvider(), "OpenAI"),
        (VoyageEmbeddingsProvider(), "Voyage"),
        (MockEmbeddingsProvider(), "Mock (baseline)")
    ]
    
    for provider, name in providers:
        try:
            await benchmark_provider(provider, name, test_texts)
        except Exception as e:
            print(f"{name} failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Benchmarking Semantic Search Quality

```python
async def benchmark_search_quality(rag_openai, rag_voyage, query, expected_file):
    """Compare search quality between providers."""
    
    # Search with OpenAI
    openai_results = await rag_openai.search_semantic(query, limit=5)
    
    # Search with Voyage
    voyage_results = await rag_voyage.search_semantic(query, limit=5)
    
    # Check if expected file is in top results
    openai_found = any(r.file_path.name == expected_file for r in openai_results[:3])
    voyage_found = any(r.file_path.name == expected_file for r in voyage_results[:3])
    
    print(f"Query: {query}")
    print(f"OpenAI found in top 3: {openai_found}")
    print(f"Voyage found in top 3: {voyage_found}")
```

## Important Considerations

### Dimension Differences

When switching providers, be aware that they use different embedding dimensions:
- **OpenAI**: 1536 dimensions
- **Voyage Code**: 1024 dimensions
- **Voyage Lite**: 512 dimensions

This means you'll need to **re-index your entire repository** when switching providers, as the vector dimensions in Neo4j must match.

### Re-indexing After Provider Switch

```bash
# Clear existing index and re-index with new provider
project-watch-mcp initialize --force
```

### Cost Considerations

- **OpenAI**: $0.00002 per 1K tokens
- **Voyage AI**: Check current pricing at https://www.voyageai.com/pricing
- **Mock**: Free (for testing only)

### Performance Tips

1. **Batch Processing**: Use `embed_batch()` for multiple texts
2. **Caching**: Consider caching embeddings for frequently accessed code
3. **Model Selection**: 
   - Use `voyage-code-3` for code-specific tasks
   - Use `voyage-3-lite` for faster, less accurate embeddings
   - Use OpenAI for general-purpose text alongside code

## Troubleshooting

### API Key Issues
```bash
# Check if API keys are set
echo $OPENAI_API_KEY
echo $VOYAGE_API_KEY

# Test provider initialization
python -c "from project_watch_mcp.utils.embedding import VoyageEmbeddingsProvider; VoyageEmbeddingsProvider()"
```

### Fallback to Mock
If API keys are missing, the system automatically falls back to mock embeddings. Check logs for warnings:
```
WARNING: Voyage API key not found. Using mock embeddings.
```

### Vector Index Issues
If you see dimension mismatch errors:
1. Drop the existing vector index in Neo4j
2. Re-initialize the repository with the new provider
3. Ensure all files are re-indexed with consistent dimensions