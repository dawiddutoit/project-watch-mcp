# Test Structure Documentation

## Overview
The test suite for Project Watch MCP follows the test pyramid principle with a clear separation between unit and integration tests.

## Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Fast, isolated unit tests
│   ├── core/                   # Core module tests
│   │   ├── test_initializer.py
│   │   └── test_monitoring_manager.py
│   ├── utils/                  # Utility module tests
│   │   └── embeddings/         # Embedding provider tests
│   │       ├── test_base.py
│   │       ├── test_embeddings_unit.py
│   │       ├── test_embeddings_utils.py
│   │       ├── test_openai.py
│   │       └── test_voyage.py
│   ├── test_cli.py             # CLI module tests
│   ├── test_cli_initialize.py
│   ├── test_cli_monitoring.py
│   ├── test_config.py          # Configuration tests
│   ├── test_server.py          # Server module tests
│   ├── test_neo4j_rag.py       # Neo4j RAG tests
│   └── test_repository_monitor.py
└── integration/                # External dependency tests
    ├── test_e2e_tool_execution.py
    ├── test_mcp_integration.py
    ├── test_embeddings_integration.py
    ├── test_embeddings_real.py
    ├── test_voyage_embeddings.py
    └── test_multi_project_isolation.py
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Characteristics**:
  - Fast execution (< 100ms per test)
  - Heavy use of mocking for external dependencies
  - Test single units of functionality
  - Mirror production code structure

### Integration Tests (`tests/integration/`)
- **Purpose**: Test interaction between components and external systems
- **Characteristics**:
  - May use real Neo4j, embeddings APIs, or file systems
  - Longer execution time acceptable
  - Minimal mocking - test real integrations
  - Focus on end-to-end workflows

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

### Run tests for a specific module
```bash
pytest tests/unit/test_config.py
pytest tests/unit/core/test_initializer.py
```

### Run with coverage
```bash
pytest tests/ --cov=src/project_watch_mcp --cov-report=term-missing
```

## Test Utilities

### Shared Test Utilities
- `tests/unit/utils/embeddings/test_embeddings_utils.py` - Contains `TestEmbeddingsProvider` for mocking embeddings in tests

### Common Fixtures (in `conftest.py`)
- `mock_neo4j_driver` - Mocked Neo4j driver
- `temp_dir` - Temporary directory for file operations
- `mock_embeddings` - Mocked embeddings provider

## Guidelines

1. **Unit Test Coverage**: Each production module should have corresponding unit tests
2. **Test Naming**: Use descriptive names that explain what is being tested
3. **Test Organization**: Group related tests in classes
4. **Mocking**: Unit tests should mock all external dependencies
5. **Integration Scope**: Integration tests should focus on critical paths and external integrations
6. **Performance**: Unit tests should complete in < 100ms, integration tests in < 5s

## Adding New Tests

When adding new functionality:
1. Create unit tests in the appropriate `tests/unit/` subdirectory
2. Mirror the production code structure
3. Add integration tests only for external system interactions
4. Update this documentation if adding new test categories