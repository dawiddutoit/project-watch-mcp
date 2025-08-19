# Project Watch MCP - Test Suite

## Overview

This directory contains the comprehensive test suite for the Project Watch MCP server, organized following strict structural guidelines to ensure maintainability and clarity.

## Test Organization

### Structure

```
tests/
├── unit/           # Unit tests - MUST mirror src/project_watch_mcp/ structure
├── integration/    # Integration tests - flexible structure
└── conftest.py     # Shared pytest fixtures
```

### Critical Rule

⚠️ **IMPORTANT**: The `tests/unit/` directory structure MUST exactly mirror the `src/project_watch_mcp/` structure. See [TESTING_GUIDELINES.md](unit/TESTING_GUIDELINES.md) for detailed rules.

## Running Tests

### All Tests
```bash
# Using pytest directly
pytest

# Using make (recommended)
make test

# With coverage
pytest --cov=src/project_watch_mcp --cov-report=html
```

### Unit Tests Only
```bash
pytest tests/unit/
```

### Integration Tests Only
```bash
pytest tests/integration/
```

### Specific Test Categories
```bash
# Complexity analysis tests
pytest tests/unit/complexity_analysis/

# Language detection tests
pytest tests/unit/language_detection/

# Neo4j tests
pytest tests/unit/test_neo4j_rag*.py
```

## Test Coverage

Current coverage target: **80%+**

View coverage report:
```bash
# Generate HTML report
pytest --cov=src/project_watch_mcp --cov-report=html

# Open report
open htmlcov/index.html
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Structure**: MUST mirror source code structure exactly
- **Guidelines**: See [unit/TESTING_GUIDELINES.md](unit/TESTING_GUIDELINES.md)

### Integration Tests (`tests/integration/`)
- **Purpose**: Test interactions between components
- **Structure**: Organized by feature/functionality
- **Subdirectories**:
  - `complexity/` - Complexity analysis integration
  - `database/` - Neo4j database operations
  - `e2e/` - End-to-end MCP server tests
  - `embeddings/` - Embedding provider integrations
  - `language_detection/` - Language detection accuracy
  - `mcp/` - MCP protocol compliance
  - `performance/` - Performance benchmarks
  - `search/` - Search functionality tests
  - `server/` - Server integration tests

## Writing New Tests

### Before Writing Tests

1. **Check structure**: Ensure test location mirrors source location
2. **Review guidelines**: Read [unit/TESTING_GUIDELINES.md](unit/TESTING_GUIDELINES.md)
3. **Use fixtures**: Leverage existing fixtures from `conftest.py`

### Test File Template

```python
"""Test suite for [module_name]."""

import pytest
from unittest.mock import Mock, patch

from project_watch_mcp.[module_path] import [ClassToTest]


class Test[ClassName]:
    """Test suite for [ClassName]."""
    
    @pytest.fixture
    def setup(self):
        """Set up test fixtures."""
        # Setup code here
        pass
    
    def test_functionality(self, setup):
        """Test specific functionality."""
        # Arrange
        
        # Act
        
        # Assert
        assert result == expected
```

## Common Test Patterns

### Async Tests
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

### Mocking Neo4j
```python
@patch('project_watch_mcp.neo4j_rag.AsyncSession')
def test_with_mock_session(mock_session):
    mock_session.run.return_value = Mock(data=lambda: [{"result": "value"}])
    # Test code
```

### Testing with Temporary Files
```python
def test_with_temp_file(tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    # Test code
```

## Continuous Integration

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Daily scheduled runs

See `.github/workflows/` for CI configuration.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure PYTHONPATH includes project root
2. **Neo4j connection**: Check test fixtures use mock connections
3. **Async warnings**: Use `pytest.mark.asyncio` decorator

### Debug Mode

Run tests with verbose output:
```bash
pytest -vv --tb=short
```

## Maintenance

### Regular Tasks

1. **Weekly**: Review and update test coverage
2. **Monthly**: Performance benchmark review
3. **Quarterly**: Test structure audit (verify mirrors source)

### Adding New Source Modules

When adding new source files:
1. Create corresponding test file in mirrored location
2. Update this README if adding new test categories
3. Ensure new tests follow established patterns

## Recent Changes

- **2025-08-18**: Major reorganization to enforce structure mirroring
  - Moved 16 test files from unit root to proper subdirectories
  - Created comprehensive testing guidelines
  - Established enforcement rules

## Contact

For test-related questions or issues:
- Check [TESTING_GUIDELINES.md](unit/TESTING_GUIDELINES.md)
- Review existing test patterns
- Create an issue if unclear