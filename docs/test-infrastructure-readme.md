# Test Infrastructure Quick Start Guide

## Overview
Enhanced test infrastructure for Project Watch MCP that addresses timeout issues and improves coverage from 60% to >80%.

## Quick Start

### 1. Run Tests with Enhanced Runner
```bash
# Quick smoke tests (30s)
./enhanced_test_runner.sh smoke

# Fast unit tests (2min)
./enhanced_test_runner.sh fast

# Standard tests with coverage (5min)
./enhanced_test_runner.sh standard 70

# Full comprehensive suite
./enhanced_test_runner.sh full 80

# Generate detailed coverage report
./enhanced_test_runner.sh coverage
```

### 2. Use Test Segmenter for Optimal Parallelization
```bash
# Analyze and segment tests
python test_segmenter.py analyze

# Execute specific segment
python test_segmenter.py execute --segment=fast

# Execute all segments
python test_segmenter.py execute --segment=all

# Optimize worker allocation
python test_segmenter.py optimize
```

### 3. Verify Coverage with Coverage Guard
```bash
# Console report
python coverage_guard.py

# JSON report for CI/CD
python coverage_guard.py --format=json

# Markdown report for PRs
python coverage_guard.py --format=markdown

# Update baseline after successful run
python coverage_guard.py --update-baseline
```

## Test Execution Profiles

### Smoke Tests (30 seconds)
- Critical path validation
- 4 parallel workers
- 5-second timeout per test
- No coverage measurement

### Fast Tests (2 minutes)
- Unit tests only
- 4 parallel workers
- 10-second timeout
- Basic coverage tracking

### Standard Tests (5 minutes)
- All non-slow tests
- 2 parallel workers
- 30-second timeout
- 60% coverage requirement

### Comprehensive Tests (10 minutes)
- Full test suite
- Sequential execution
- 120-second timeout
- 80% coverage requirement

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_fast_unit_test():
    """Fast isolated unit test"""
    pass

@pytest.mark.integration
def test_database_integration():
    """Test requiring database"""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Test taking >1 second"""
    pass

@pytest.mark.smoke
def test_critical_path():
    """Critical functionality test"""
    pass
```

## Coverage Targets

| Component | Target | Critical |
|-----------|--------|----------|
| server.py | 85% | ✅ |
| repository_monitor.py | 85% | ✅ |
| neo4j_rag.py | 85% | ✅ |
| complexity_analyzer.py | 90% | ✅ |
| utils/ | 75% | |
| embeddings/ | 70% | |

## CI/CD Integration

The infrastructure includes GitHub Actions workflow that:
1. Runs smoke tests first for quick feedback
2. Executes tests in parallel segments
3. Combines coverage from all segments
4. Verifies coverage requirements
5. Generates reports for PRs

## Troubleshooting

### Timeout Issues
```bash
# Use segmented execution
./enhanced_test_runner.sh segment

# Or reduce parallelization
uv run pytest -n=1 --timeout=60
```

### Coverage Issues
```bash
# Check which files need coverage
python coverage_guard.py --format=console

# Run specific test segments
python test_segmenter.py execute --segment=medium
```

### Test Discovery Issues
```bash
# Analyze test distribution
python test_segmenter.py analyze

# Check test markers
uv run pytest --markers
```

## Best Practices

1. **Always use markers** - Mark tests appropriately for better segmentation
2. **Keep tests fast** - Unit tests should complete in <100ms
3. **Isolate tests** - Use mocks/fixtures to avoid external dependencies
4. **Monitor coverage** - Run coverage guard before committing
5. **Update baselines** - After major changes, update coverage baseline

## Files Reference

- `enhanced_test_runner.sh` - Main test execution script
- `test_segmenter.py` - Intelligent test segmentation
- `coverage_guard.py` - Coverage verification system
- `coverage_config.json` - Coverage thresholds configuration
- `.github/workflows/test-coverage.yml` - CI/CD pipeline
- `pyproject.toml` - Updated pytest configuration