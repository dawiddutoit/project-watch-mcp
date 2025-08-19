# Test Consolidation Action Plan

## Priority 1: Neo4j RAG Test Consolidation

### Current State
- 3 test files with 40+ overlapping test methods
- ~2,500 lines of test code with ~60% duplication
- Inconsistent mocking approaches

### Target State
- 1 comprehensive test file: `test_neo4j_rag.py`
- ~1,200 lines of well-organized test code
- Consistent mocking strategy

### Detailed Consolidation Steps

#### Step 1: Analyze Unique Tests
```bash
# Run this to identify truly unique tests
grep -h "def test_" tests/unit/test_neo4j_rag*.py | sort | uniq -c | sort -rn
```

#### Step 2: Create New Structure
```python
# tests/unit/test_neo4j_rag.py

"""
Comprehensive test suite for Neo4j RAG functionality.

Test Classes:
- TestNeo4jRAGInitialization: Configuration and setup
- TestNeo4jRAGIndexing: File indexing and chunking
- TestNeo4jRAGSearch: Semantic and pattern search
- TestNeo4jRAGFileOperations: CRUD operations
- TestNeo4jRAGErrorHandling: Error scenarios and edge cases
- TestNeo4jRAGPerformance: Performance-related tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
# ... imports

class TestNeo4jRAGInitialization:
    """Test Neo4j RAG initialization and configuration."""
    
    async def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        
    async def test_initialization_custom_config(self):
        """Test initialization with custom chunk size and overlap."""
        
    async def test_create_constraints(self):
        """Test database constraint creation."""
        
    async def test_create_indexes(self):
        """Test database index creation."""
        
    async def test_connection_verification(self):
        """Test Neo4j connection verification."""
        
    async def test_close_connection(self):
        """Test proper connection cleanup."""

class TestNeo4jRAGIndexing:
    """Test file indexing and content chunking."""
    
    async def test_index_single_file(self):
        """Test indexing a single code file."""
        
    async def test_index_file_with_embeddings(self):
        """Test file indexing with embedding generation."""
        
    async def test_chunk_small_file(self):
        """Test chunking behavior for small files."""
        
    async def test_chunk_large_file(self):
        """Test chunking behavior for large files."""
        
    async def test_chunk_with_overlap(self):
        """Test chunk overlap functionality."""
        
    async def test_update_existing_file(self):
        """Test updating an already indexed file."""
        
    async def test_index_file_read_error(self):
        """Test handling of file read errors during indexing."""

class TestNeo4jRAGSearch:
    """Test search functionality."""
    
    async def test_semantic_search_basic(self):
        """Test basic semantic search."""
        
    async def test_semantic_search_with_language_filter(self):
        """Test semantic search filtered by language."""
        
    async def test_semantic_search_empty_query(self):
        """Test handling of empty search queries."""
        
    async def test_pattern_search_literal(self):
        """Test literal pattern search."""
        
    async def test_pattern_search_regex(self):
        """Test regex pattern search."""
        
    async def test_pattern_search_case_insensitive(self):
        """Test case-insensitive pattern search."""
        
    async def test_search_limit_parameter(self):
        """Test search result limiting."""

class TestNeo4jRAGFileOperations:
    """Test file CRUD operations."""
    
    async def test_get_file_metadata_existing(self):
        """Test retrieving metadata for existing file."""
        
    async def test_get_file_metadata_nonexistent(self):
        """Test metadata request for nonexistent file."""
        
    async def test_update_file_content(self):
        """Test updating file content and embeddings."""
        
    async def test_delete_file(self):
        """Test file deletion from index."""
        
    async def test_delete_nonexistent_file(self):
        """Test deletion of file not in index."""
        
    async def test_get_repository_stats(self):
        """Test repository statistics retrieval."""

class TestNeo4jRAGErrorHandling:
    """Test error handling and edge cases."""
    
    async def test_embedding_generation_failure(self):
        """Test handling of embedding API failures."""
        
    async def test_neo4j_connection_lost(self):
        """Test handling of lost database connection."""
        
    async def test_concurrent_file_updates(self):
        """Test handling of concurrent file modifications."""
        
    async def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        
    async def test_transaction_rollback(self):
        """Test transaction rollback on error."""

class TestNeo4jRAGPerformance:
    """Test performance-related functionality."""
    
    @pytest.mark.slow
    async def test_bulk_indexing(self):
        """Test indexing multiple files in bulk."""
        
    @pytest.mark.slow
    async def test_large_repository_scan(self):
        """Test performance with large repositories."""
        
    async def test_embedding_caching(self):
        """Test embedding result caching."""
```

#### Step 3: Migration Process

1. **Extract Unique Tests**
```python
# Create a script to identify and extract unique test implementations
import ast
import os

def extract_test_methods(filepath):
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    
    tests = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            tests[node.name] = {
                'file': filepath,
                'line': node.lineno,
                'body_length': len(node.body)
            }
    return tests

# Compare test methods across files
all_tests = {}
for file in ['test_neo4j_rag.py', 'test_neo4j_rag_comprehensive.py', 'test_neo4j_rag_extended.py']:
    all_tests[file] = extract_test_methods(f'tests/unit/{file}')

# Identify unique tests per file
# ... analysis code
```

2. **Merge Fixtures**
```python
# tests/conftest.py
# Consolidate all Neo4j RAG fixtures in one place

@pytest.fixture
def mock_neo4j_driver():
    """Unified mock Neo4j driver for all tests."""
    driver = AsyncMock()
    session = AsyncMock()
    # ... complete mock setup
    return driver

@pytest.fixture
def mock_embeddings_provider():
    """Unified mock embeddings provider."""
    # ... consolidated embedding mock
    return mock
```

3. **Update Imports**
```bash
# Update all files that import from the deleted test files
grep -r "from tests.unit.test_neo4j_rag_comprehensive" . --include="*.py"
grep -r "from tests.unit.test_neo4j_rag_extended" . --include="*.py"
```

#### Step 4: Validation

1. **Coverage Check**
```bash
# Before consolidation
pytest tests/unit/test_neo4j_rag*.py --cov=src/project_watch_mcp/neo4j_rag --cov-report=term-missing

# After consolidation
pytest tests/unit/test_neo4j_rag.py --cov=src/project_watch_mcp/neo4j_rag --cov-report=term-missing

# Ensure coverage hasn't decreased
```

2. **Test Count Verification**
```bash
# Count unique test methods before
grep "def test_" tests/unit/test_neo4j_rag*.py | sort | uniq | wc -l

# Count test methods after
grep "def test_" tests/unit/test_neo4j_rag.py | wc -l

# Numbers should be close (some genuine duplicates removed)
```

3. **Performance Comparison**
```bash
# Time before consolidation
time pytest tests/unit/test_neo4j_rag*.py

# Time after consolidation
time pytest tests/unit/test_neo4j_rag.py

# Should see improvement due to less duplicate setup/teardown
```

#### Step 5: Clean Up

1. **Delete Old Files**
```bash
git rm tests/unit/test_neo4j_rag_comprehensive.py
git rm tests/unit/test_neo4j_rag_extended.py
```

2. **Update Documentation**
```markdown
# tests/README.md
## Test Organization
- Each module has one comprehensive test file
- Test classes are organized by functionality
- No more "_comprehensive" or "_extended" variants
```

3. **Update CI Configuration**
```yaml
# .github/workflows/test.yml
# Remove references to deleted test files
# Update coverage thresholds if needed
```

### Success Criteria

✅ All unique test cases preserved
✅ No decrease in code coverage
✅ Test execution time reduced by at least 30%
✅ Clear test organization by functionality
✅ No broken imports or references
✅ CI/CD pipeline passes

### Rollback Plan

If issues arise:
1. Tests are in git history - can be restored
2. Keep backup branch: `git checkout -b backup/pre-consolidation`
3. Document any test-specific quirks discovered

## Priority 2: Repository Monitor Test Consolidation

[Similar detailed plan for repository_monitor tests...]

## Priority 3: Complexity Analysis Test Consolidation

[Similar detailed plan for complexity tests...]

## Automation Tools

### Test Deduplication Script
```python
#!/usr/bin/env python3
"""
Script to identify and report duplicate test methods.
Usage: python find_duplicate_tests.py tests/unit
"""

import ast
import hashlib
from pathlib import Path
from collections import defaultdict

def get_test_signature(node):
    """Generate a signature for a test method."""
    # Create signature from test name and key assertions
    signature_parts = [node.name]
    
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            signature_parts.append(ast.unparse(child.test))
        elif isinstance(child, ast.Call) and hasattr(child.func, 'attr'):
            if child.func.attr in ['assertEqual', 'assertTrue', 'assertFalse']:
                signature_parts.append(child.func.attr)
    
    return hashlib.md5(''.join(signature_parts).encode()).hexdigest()

def find_duplicate_tests(directory):
    """Find duplicate test methods across files."""
    test_signatures = defaultdict(list)
    
    for py_file in Path(directory).glob('test_*.py'):
        with open(py_file) as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                signature = get_test_signature(node)
                test_signatures[signature].append({
                    'file': py_file.name,
                    'method': node.name,
                    'line': node.lineno
                })
    
    # Report duplicates
    duplicates = {k: v for k, v in test_signatures.items() if len(v) > 1}
    return duplicates

if __name__ == '__main__':
    import sys
    duplicates = find_duplicate_tests(sys.argv[1])
    
    for signature, tests in duplicates.items():
        print(f"\nDuplicate test methods found:")
        for test in tests:
            print(f"  - {test['file']}:{test['line']} - {test['method']}")
```

### Test Migration Helper
```python
#!/usr/bin/env python3
"""
Helper script to migrate tests from multiple files to a single consolidated file.
"""

import ast
import astor
from pathlib import Path

class TestMigrator(ast.NodeTransformer):
    """AST transformer to help migrate and consolidate tests."""
    
    def __init__(self, target_class_mapping):
        self.target_class_mapping = target_class_mapping
    
    def visit_FunctionDef(self, node):
        # Categorize test methods into appropriate classes
        if node.name.startswith('test_'):
            for keyword, target_class in self.target_class_mapping.items():
                if keyword in node.name:
                    node.decorator_list.append(
                        ast.Name(id=f'# TARGET_CLASS: {target_class}', ctx=ast.Load())
                    )
                    break
        return node

def consolidate_tests(source_files, output_file, class_structure):
    """Consolidate multiple test files into one organized file."""
    all_tests = defaultdict(list)
    
    for source_file in source_files:
        with open(source_file) as f:
            tree = ast.parse(f.read())
        
        migrator = TestMigrator(class_structure)
        migrated_tree = migrator.visit(tree)
        
        # Extract and categorize test methods
        # ... implementation
    
    # Generate consolidated file
    # ... implementation

# Usage
consolidate_tests(
    source_files=[
        'test_neo4j_rag.py',
        'test_neo4j_rag_comprehensive.py',
        'test_neo4j_rag_extended.py'
    ],
    output_file='test_neo4j_rag_consolidated.py',
    class_structure={
        'init': 'TestNeo4jRAGInitialization',
        'index': 'TestNeo4jRAGIndexing',
        'search': 'TestNeo4jRAGSearch',
        'file': 'TestNeo4jRAGFileOperations',
        'error': 'TestNeo4jRAGErrorHandling'
    }
)
```

## Timeline

### Week 1 (Days 1-7)
- Day 1-2: Neo4j RAG consolidation
- Day 3-4: Repository Monitor consolidation
- Day 5: Testing and validation
- Day 6-7: Documentation and CI updates

### Week 2 (Days 8-14)
- Day 8-9: Complexity Analysis consolidation
- Day 10-11: CLI tests consolidation
- Day 12-13: E2E tests consolidation
- Day 14: Integration testing

### Week 3 (Days 15-21)
- Day 15-16: Embeddings tests consolidation
- Day 17-18: Language Detection consolidation
- Day 19-20: Final cleanup and optimization
- Day 21: Performance benchmarking

## Risk Mitigation

1. **Test Loss Risk**: Create comprehensive backup before starting
2. **Coverage Drop Risk**: Run coverage after each consolidation
3. **Breaking Changes Risk**: Run full test suite after each change
4. **Time Overrun Risk**: Prioritize high-value consolidations first

## Success Metrics

- [ ] Test file count reduced by 60%
- [ ] Test execution time reduced by 30%
- [ ] Code coverage maintained or improved
- [ ] Zero test failures after consolidation
- [ ] Clear documentation for test organization