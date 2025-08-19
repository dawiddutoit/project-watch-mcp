# Unit Testing Guidelines

## CRITICAL RULE: Mirror Source Structure

**The `tests/unit/` directory MUST exactly mirror the structure of `src/project_watch_mcp/`**

This is non-negotiable and prevents test organization issues.

## Directory Structure Mapping

```
src/project_watch_mcp/                    tests/unit/
├── __init__.py                     →     ├── __init__.py
├── __main__.py                     →     ├── test_main_module.py
├── cli.py                          →     ├── test_cli.py
├── config.py                       →     ├── test_config.py
├── neo4j_rag.py                    →     ├── test_neo4j_rag.py
├── repository_monitor.py           →     ├── test_repository_monitor.py
├── server.py                       →     ├── test_server.py
│                                          │
├── complexity_analysis/             →     ├── complexity_analysis/
│   ├── base_analyzer.py           →     │   ├── test_base_analyzer.py
│   ├── metrics.py                  →     │   ├── test_metrics.py
│   ├── models.py                   →     │   ├── test_models.py
│   └── languages/                  →     │   └── languages/
│       ├── java_analyzer.py       →     │       ├── test_java_analyzer.py
│       ├── kotlin_analyzer.py     →     │       ├── test_kotlin_analyzer.py
│       └── python_analyzer.py     →     │       └── test_python_analyzer.py
│                                          │
├── core/                            →     ├── core/
│   ├── initializer.py             →     │   ├── test_initializer.py
│   └── monitoring_manager.py      →     │   └── test_monitoring_manager.py
│                                          │
├── language_detection/              →     ├── language_detection/
│   ├── cache.py                   →     │   ├── test_cache.py
│   ├── hybrid_detector.py         →     │   ├── test_hybrid_detector.py
│   └── models.py                   →     │   └── test_models.py
│                                          │
├── optimization/                    →     ├── optimization/
│   ├── batch_processor.py         →     │   ├── test_batch_processor.py
│   ├── cache_manager.py           →     │   ├── test_cache_manager.py
│   ├── connection_pool.py         →     │   ├── test_connection_pool.py
│   └── optimizer.py                →     │   └── test_optimizer.py
│                                          │
├── utils/                           →     ├── utils/
│   ├── profiler.py                →     │   ├── test_profiler.py
│   └── embeddings/                →     │   └── embeddings/
│       ├── base.py                →     │       ├── test_base.py
│       ├── openai.py              →     │       ├── test_openai.py
│       ├── vector_support.py      →     │       ├── test_vector_support.py
│       └── voyage.py               →     │       └── test_voyage.py
│                                          │
└── vector_search/                   →     └── vector_search/
    └── neo4j_native_vectors.py    →         └── test_neo4j_native_vectors.py
```

## File Naming Conventions

### Standard Test Files
- **Pattern**: `test_{module_name}.py`
- **Example**: `src/project_watch_mcp/cli.py` → `tests/unit/test_cli.py`

### Multiple Test Files for One Module
When a module needs multiple test files (e.g., for organization):
- **Pattern**: `test_{module_name}_{suffix}.py`
- **Examples**:
  - `test_neo4j_rag.py` - Main tests
  - `test_neo4j_rag_comprehensive.py` - Comprehensive test suite
  - `test_neo4j_rag_extended.py` - Extended/edge case tests

### Test Utilities
- **Pattern**: `{area}_test_utils.py` or `test_helpers.py`
- **Location**: In the same directory as the tests that use them
- **Example**: `tests/unit/utils/embeddings/embeddings_test_utils.py`

## Rules for Creating New Tests

### 1. Check Source Structure First
Before creating a test file:
```bash
# Find the source file location
find src/project_watch_mcp -name "module_name.py"
```

### 2. Create Matching Test Structure
```bash
# If source is at: src/project_watch_mcp/new_feature/module.py
# Create test at:  tests/unit/new_feature/test_module.py

# Create directory if needed
mkdir -p tests/unit/new_feature

# Create test file
touch tests/unit/new_feature/test_module.py
touch tests/unit/new_feature/__init__.py
```

### 3. Never Place Tests in Root
**NEVER** create test files directly in `tests/unit/` unless the source module is directly in `src/project_watch_mcp/`

### 4. Import Path Consistency
Test imports should mirror source imports:
```python
# If source import is:
from project_watch_mcp.complexity_analysis.base_analyzer import BaseAnalyzer

# Test file should be in:
# tests/unit/complexity_analysis/test_base_analyzer.py
```

## Common Mistakes to Avoid

### ❌ DON'T: Create tests in root for submodule files
```bash
# WRONG
tests/unit/test_java_analyzer.py  # for src/.../languages/java_analyzer.py
```

### ✅ DO: Mirror the exact structure
```bash
# CORRECT
tests/unit/complexity_analysis/languages/test_java_analyzer.py
```

### ❌ DON'T: Use different directory names
```bash
# WRONG
tests/unit/complexity/  # when source has complexity_analysis/
```

### ✅ DO: Use exact directory names
```bash
# CORRECT
tests/unit/complexity_analysis/  # matches source exactly
```

## Verification Script

Run this to verify test structure matches source:
```bash
# List source modules
find src/project_watch_mcp -name "*.py" -not -path "*/__pycache__/*" | \
  sed 's|src/project_watch_mcp/||' | sort

# List test modules (should mirror above)
find tests/unit -name "test_*.py" -not -path "*/__pycache__/*" | \
  sed 's|tests/unit/||' | sed 's|test_||' | sort
```

## Integration Tests

Integration tests (`tests/integration/`) have more flexibility in structure as they test interactions between modules rather than individual modules.

## Enforcement

1. **Pre-commit Hook**: Consider adding a pre-commit hook to verify structure
2. **CI Check**: Add a CI job to verify test structure matches source
3. **Code Review**: Reviewers should verify new tests follow structure

## Quick Reference Checklist

When adding a new test file:
- [ ] Is the source file directly in `src/project_watch_mcp/`? → Test goes in `tests/unit/`
- [ ] Is the source file in a subdirectory? → Test goes in matching subdirectory
- [ ] Does the test directory exist? → Create it with `__init__.py`
- [ ] Does the test file name follow `test_{module_name}.py` pattern?
- [ ] Are imports using the correct paths?

## Maintenance

This structure should be reviewed and updated whenever:
- New source directories are added
- Source structure is reorganized
- Testing patterns change

Last Updated: 2025-08-18
Last Reorganization: 2025-08-18 (moved 16 files to proper subdirectories)