# Integration Tests Organization

This directory contains integration tests for project-watch-mcp, organized by functional boundaries to ensure comprehensive testing of the system's major components and their interactions.

## Directory Structure

### `/server/`
Tests for server initialization and startup procedures.
- `test_mcp_server_startup.py` - MCP server initialization
- `test_repository_initialization.py` - Repository setup and initial indexing
- `test_file_indexing.py` - File indexing processes

### `/cli/`
Command-line interface tests (currently empty - CLI tests to be added).

### `/mcp/`
Model Context Protocol integration tests.
- `test_mcp_integration.py` - MCP server integration with project context
- `test_e2e_tool_execution.py` - End-to-end MCP tool execution

### `/database/`
Neo4j database connectivity and operations.
- `test_neo4j_connection.py` - Connection establishment, pooling, retry logic
- `test_neo4j_search_solution_validation.py` - Complete search solution validation

### `/embeddings/`
Embedding system integration tests.
- `test_embeddings_integration.py` - OpenAI embeddings API integration
- `test_embeddings_real.py` - Real-world embedding scenarios
- `test_voyage_embeddings.py` - Voyage AI embedding provider
- `test_embedding_provider_switching.py` - Provider switching functionality
- `test_embedding_enrichment.py` - Embedding enrichment features
- `test_native_vector_integration.py` - Neo4j native vector index

### `/search/`
Search functionality tests including Lucene and vector search.
- `test_lucene_pattern_search.py` - Lucene pattern matching
- `test_lucene_escaping_demo.py` - Lucene query escaping
- `test_vector_search_integration.py` - Neo4j vector search
- `test_lucene_vs_vector_proof.py` - Comparative testing
- `test_vector_vs_lucene_performance.py` - Performance benchmarks
- `test_fulltext_analyzer_comparison.py` - Full-text analysis methods

### `/complexity/`
Code complexity analysis across multiple languages.
- `test_complexity_cross_language.py` - Cross-language consistency
- `test_complexity_accuracy.py` - Complexity metrics accuracy
- `test_language_complexity_integration.py` - Language-specific integration
- `test_class_linkage_demo.py` - Class relationship analysis

### `/language_detection/`
Programming language detection tests.
- `test_language_detection.py` - Basic language detection
- `test_language_detection_accuracy.py` - Detection accuracy metrics
- `test_language_detection_caching.py` - Caching mechanisms

### `/performance/`
Performance benchmarks and resilience tests.
- `test_performance_benchmark.py` - System performance benchmarks
- `test_cache_performance_benchmark.py` - Cache performance
- `test_corruption_prevention.py` - Data corruption prevention
- `test_recursion_memory_fixes.py` - Memory leak fixes
- `test_resource_management_simple.py` - Resource management

### `/e2e/`
End-to-end and system-wide integration tests.
- `test_end_to_end.py` - Complete workflow testing
- `test_full_system_integration.py` - Full system scenarios
- `test_no_mocks.py` - Real integration without mocking
- `test_integration_monitoring.py` - Monitoring system
- `test_multi_project_isolation.py` - Multi-project support
- `test_project_isolation.py` - Project-level isolation
- `test_project_context.py` - Project context handling
- `test_monitoring_persistence.py` - Monitoring state persistence

## Testing Philosophy

These integration tests focus on the **boundaries** of the system:
1. **External APIs** - Testing real interactions with Neo4j, OpenAI, etc.
2. **System Interfaces** - MCP protocol, CLI commands, server endpoints
3. **Data Flow** - End-to-end data processing from files to embeddings to search
4. **Cross-component** - Testing interactions between major subsystems

## Running Tests

```bash
# Run all integration tests
pytest tests/integration/

# Run specific category
pytest tests/integration/server/
pytest tests/integration/embeddings/
pytest tests/integration/search/

# Run with coverage
pytest tests/integration/ --cov=src/project_watch_mcp

# Skip tests requiring external services
pytest tests/integration/ -m "not requires_neo4j"
pytest tests/integration/ -m "not requires_openai"
```

## Environment Variables

Many integration tests require external services. Set these environment variables:
- `NEO4J_URI` - Neo4j database URI
- `NEO4J_USERNAME` - Neo4j username
- `NEO4J_PASSWORD` - Neo4j password
- `OPENAI_API_KEY` - OpenAI API key for embeddings
- `VOYAGE_API_KEY` - Voyage AI API key (optional)

## Test Markers

Tests use pytest markers to indicate requirements:
- `@pytest.mark.requires_neo4j` - Requires Neo4j database
- `@pytest.mark.requires_openai` - Requires OpenAI API
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.benchmark` - Performance benchmarks