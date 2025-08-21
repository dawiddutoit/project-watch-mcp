# Real Database Integration Tests

This directory contains **REAL** integration tests that use actual Neo4j database instances instead of mocks.

## Overview

The original tests in this directory were using mocks extensively, which doesn't actually test database behavior. The new tests use Docker containers via `testcontainers-python` to spin up real Neo4j instances for testing.

## Test Files

### New Real Integration Tests
- `test_real_neo4j_integration.py` - Comprehensive real database tests
- `test_neo4j_connection_real.py` - Connection management with real database
- `test_neo4j_search_validation_real.py` - Search functionality with real database
- `test_docker_neo4j.py` - Simple test to verify Docker setup

### Original Mock-Based Tests (Should be refactored or removed)
- `test_neo4j_connection.py` - Uses mocks instead of real database
- `test_neo4j_search_solution_validation.py` - Uses mocks for search testing

## Requirements

1. **Docker** must be installed and running
2. **testcontainers-neo4j** package (already added to dev dependencies)

## Running the Tests

### Quick Docker Test
First, verify Docker and Neo4j container setup works:
```bash
python -m pytest tests/integration/database/test_docker_neo4j.py -v -s
```

### Run All Real Integration Tests
```bash
python -m pytest tests/integration/database/test_real_*.py -v -m integration
```

### Run Specific Test Suite
```bash
# Connection tests
python -m pytest tests/integration/database/test_neo4j_connection_real.py -v

# Search tests  
python -m pytest tests/integration/database/test_neo4j_search_validation_real.py -v

# Core integration tests
python -m pytest tests/integration/database/test_real_neo4j_integration.py -v
```

## Test Fixtures

The real database fixtures are defined in `/tests/conftest.py`:

- `neo4j_container` - Session-scoped fixture that starts a Neo4j Docker container
- `real_neo4j_driver` - Function-scoped fixture providing a clean Neo4j driver
- `real_neo4j_rag` - Function-scoped fixture providing a Neo4jRAG instance
- `real_embeddings_provider` - Provides real or mock embeddings based on API key availability

## Key Differences from Mock Tests

### Mock-Based Tests (OLD)
```python
# Uses mocks - doesn't test real database behavior
mock_driver = AsyncMock()
mock_session = AsyncMock()
mock_session.run = AsyncMock(return_value={"data": "fake"})
```

### Real Database Tests (NEW)
```python
# Uses actual Neo4j database in Docker
async with real_neo4j_driver.session() as session:
    result = await session.run("MATCH (n) RETURN n")
    # This actually queries a real database!
```

## Benefits of Real Integration Tests

1. **Actual Database Behavior**: Tests real Neo4j query execution, not mocked responses
2. **Transaction Testing**: Can test real ACID properties and rollbacks
3. **Index Performance**: Can measure actual index creation and query performance
4. **Constraint Validation**: Tests real database constraints and error handling
5. **Concurrency Testing**: Can test real concurrent query execution
6. **Connection Pooling**: Tests actual driver connection pool behavior

## Performance Considerations

- First test run will be slower as Docker pulls the Neo4j image
- Subsequent runs use cached images and are much faster
- Each test function gets a clean database (data is cleared between tests)
- The container is reused across the test session for efficiency

## Troubleshooting

### Docker Not Running
```
Error: Cannot connect to Docker daemon
Solution: Start Docker Desktop or Docker daemon
```

### Image Pull Timeout
```
Error: Timeout pulling neo4j image
Solution: Pull the image manually first:
docker pull neo4j:5
```

### Port Conflicts
```
Error: Port already in use
Solution: The container uses random ports, but check for conflicts
```

## Next Steps

1. **Remove Mock Tests**: The original mock-based tests should be removed or clearly marked as unit tests
2. **Add More Scenarios**: Add tests for specific Neo4j features like:
   - Graph algorithms
   - Spatial indexes
   - Full-text search with different analyzers
   - Vector similarity search
3. **Performance Benchmarks**: Add benchmark tests to track query performance over time
4. **CI/CD Integration**: Ensure CI pipeline has Docker available for these tests