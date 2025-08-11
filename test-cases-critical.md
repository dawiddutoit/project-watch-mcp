# Critical Test Case Specifications for Project Watch MCP

## Overview
This document provides detailed test case specifications for the critical features currently experiencing failures. These test cases are designed to ensure comprehensive coverage and identify edge cases.

## 1. MCP Integration Test Cases

### TC-MCP-001: Initialize Repository with Project Context
**Priority**: P0 (Critical)  
**Category**: Functional, Integration  
**Current Status**: FAILING

#### Prerequisites
- Neo4j database running and accessible
- Clean test environment
- Valid repository path available

#### Test Data
```python
test_data = {
    "repository_path": "/tmp/test_repo",
    "project_name": "test_project_alpha",
    "file_count": 10,
    "file_types": [".py", ".js", ".md", ".txt"],
    "gitignore_patterns": ["*.pyc", "__pycache__/", ".venv/"]
}
```

#### Test Steps
1. Create test repository with specified structure
2. Initialize MCP server with repository path and project context
3. Call `initialize_repository` tool with project name
4. Verify initialization response
5. Query Neo4j for project-specific nodes
6. Verify file count matches expected
7. Check gitignore patterns are respected

#### Expected Results
- Initialization completes without errors
- Response includes correct project name
- All eligible files are indexed
- Gitignore patterns exclude specified files
- Neo4j contains nodes with correct project context
- File monitoring is active

#### Validation Queries
```cypher
// Verify project context in Neo4j
MATCH (c:CodeChunk {project: 'test_project_alpha'})
RETURN count(c) as chunk_count

// Verify no cross-contamination
MATCH (c:CodeChunk)
WHERE c.project IS NULL OR c.project = ''
RETURN count(c) as orphan_chunks
```

#### Error Scenarios
- Invalid repository path
- Neo4j connection failure
- Permission denied on files
- Corrupted gitignore file

---

### TC-MCP-002: Concurrent MCP Operations
**Priority**: P0 (Critical)  
**Category**: Performance, Concurrency  
**Current Status**: FAILING

#### Prerequisites
- Initialized repository with 100+ files
- MCP server running with connection pool

#### Test Data
```python
concurrent_operations = [
    {"type": "search", "query": "function definition", "count": 10},
    {"type": "stats", "count": 5},
    {"type": "refresh", "files": ["main.py", "utils.py"], "count": 8},
    {"type": "file_info", "files": ["test.py"], "count": 7}
]
```

#### Test Steps
1. Initialize repository with test data
2. Create 30 concurrent MCP tool calls
3. Mix different operation types
4. Monitor response times
5. Verify all operations complete
6. Check for data consistency
7. Validate no race conditions

#### Expected Results
- All operations complete within 10 seconds
- No operations fail due to concurrency
- Response times remain consistent (±20%)
- Data remains consistent across operations
- No deadlocks or timeouts
- Memory usage stays below 500MB

#### Performance Metrics
```python
performance_thresholds = {
    "search_p95": 2000,  # ms
    "stats_p95": 500,     # ms
    "refresh_p95": 1000,  # ms
    "file_info_p95": 200, # ms
    "error_rate": 0.01,   # 1%
    "timeout_rate": 0.001 # 0.1%
}
```

---

## 2. Project Isolation Test Cases

### TC-ISO-001: Data Isolation Between Projects
**Priority**: P0 (Critical)  
**Category**: Data Integrity, Security  
**Current Status**: FAILING

#### Prerequisites
- Clean Neo4j database
- Two distinct test repositories

#### Test Data
```python
project_a = {
    "name": "project_alpha",
    "path": "/tmp/project_a",
    "unique_content": "ALPHA_UNIQUE_MARKER_12345",
    "files": ["alpha_main.py", "alpha_utils.py"]
}

project_b = {
    "name": "project_beta", 
    "path": "/tmp/project_b",
    "unique_content": "BETA_UNIQUE_MARKER_67890",
    "files": ["beta_main.py", "beta_utils.py"]
}
```

#### Test Steps
1. Initialize Project A with unique content
2. Initialize Project B with different unique content
3. Search in Project A for Project B's unique marker
4. Search in Project B for Project A's unique marker
5. Get stats for Project A
6. Get stats for Project B
7. Modify file in Project A
8. Verify Project B stats unchanged
9. Delete Project A data
10. Verify Project B data intact

#### Expected Results
- Cross-project searches return empty results
- Stats are accurate per project
- File modifications don't affect other projects
- Deletion is project-specific
- No shared state between projects

#### Validation Queries
```cypher
// Check for data leakage
MATCH (c:CodeChunk)
WHERE c.project = 'project_alpha' 
  AND c.content CONTAINS 'BETA_UNIQUE_MARKER'
RETURN c

// Verify project separation
MATCH (c:CodeChunk)
RETURN c.project, count(c) as count
ORDER BY c.project
```

---

### TC-ISO-002: Cross-Project Contamination Prevention
**Priority**: P0 (Critical)  
**Category**: Data Integrity  
**Current Status**: FAILING

#### Prerequisites
- Multiple projects initialized
- Active file monitoring

#### Test Scenarios
```python
contamination_tests = [
    {
        "name": "concurrent_search",
        "description": "Simultaneous searches shouldn't mix results",
        "operations": ["search_a", "search_b", "search_c"]
    },
    {
        "name": "rapid_context_switch",
        "description": "Fast project switching maintains isolation",
        "switch_count": 100,
        "switch_delay_ms": 10
    },
    {
        "name": "shared_filenames",
        "description": "Same filenames in different projects stay separate",
        "shared_files": ["main.py", "config.json", "README.md"]
    }
]
```

#### Test Steps
1. Initialize 5 projects with overlapping filenames
2. Perform rapid context switches
3. Execute concurrent operations
4. Monitor for data leakage
5. Verify result accuracy
6. Check performance degradation
7. Validate state consistency

#### Expected Results
- Zero contamination events
- Consistent query results
- Performance within thresholds
- Accurate project attribution
- No memory leaks

---

## 3. Search Functionality Test Cases

### TC-SEARCH-001: Semantic Search with Language Filter
**Priority**: P1 (High)  
**Category**: Functional  
**Current Status**: FAILING

#### Prerequisites
- Multi-language repository indexed
- Embeddings generated for all files

#### Test Data
```python
search_scenarios = [
    {
        "query": "database connection",
        "language": "python",
        "expected_files": ["db_utils.py", "models.py"],
        "excluded_files": ["db.js", "database.go"]
    },
    {
        "query": "error handling",
        "language": "javascript",
        "expected_files": ["error.js", "utils.js"],
        "excluded_files": ["errors.py", "error.rs"]
    }
]
```

#### Test Steps
1. Index multi-language repository
2. Perform semantic search with language filter
3. Verify results match language
4. Check relevance scores
5. Test with non-existent language
6. Test with empty query
7. Test with special characters

#### Expected Results
- Only specified language files returned
- Relevance scores > 0.6 for matches
- Graceful handling of edge cases
- Response time < 2 seconds
- Accurate language detection

---

### TC-SEARCH-002: Pattern Search with Regex
**Priority**: P1 (High)  
**Category**: Functional  
**Current Status**: FAILING

#### Prerequisites
- Repository with diverse code patterns
- Pattern search enabled

#### Test Data
```python
regex_patterns = [
    {
        "pattern": r"def\s+test_\w+\(",
        "description": "Python test functions",
        "min_matches": 10
    },
    {
        "pattern": r"TODO:\s*.+",
        "description": "TODO comments",
        "min_matches": 5
    },
    {
        "pattern": r"import\s+\{[^}]+\}",
        "description": "JavaScript named imports",
        "min_matches": 3
    }
]
```

#### Test Steps
1. Execute regex pattern searches
2. Verify match accuracy
3. Test invalid regex handling
4. Check performance with complex patterns
5. Test with large result sets
6. Verify line number accuracy

#### Expected Results
- All valid patterns match correctly
- Invalid regex returns error message
- Performance < 500ms for simple patterns
- Line numbers accurate
- No false positives

---

## 4. File Monitoring Test Cases

### TC-MON-001: Real-time File Change Detection
**Priority**: P1 (High)  
**Category**: Functional, Performance  
**Current Status**: PASSING (Included for completeness)

#### Prerequisites
- Active repository monitoring
- Write permissions on test files

#### Test Scenarios
```python
file_operations = [
    {"op": "modify", "file": "main.py", "detection_time_ms": 100},
    {"op": "create", "file": "new_file.py", "detection_time_ms": 100},
    {"op": "delete", "file": "temp.py", "detection_time_ms": 100},
    {"op": "rename", "from": "old.py", "to": "new.py", "detection_time_ms": 200},
    {"op": "batch", "count": 50, "detection_time_ms": 500}
]
```

#### Test Steps
1. Start file monitoring
2. Perform file operations
3. Measure detection latency
4. Verify index updates
5. Check for missed events
6. Test under load

#### Expected Results
- Detection within specified time
- 100% event capture rate
- Correct index updates
- No duplicate processing
- Stable under load

---

## 5. Error Recovery Test Cases

### TC-ERR-001: Neo4j Connection Recovery
**Priority**: P1 (High)  
**Category**: Reliability  
**Current Status**: NOT TESTED

#### Prerequisites
- Ability to stop/start Neo4j
- Connection retry logic implemented

#### Test Scenarios
```python
failure_scenarios = [
    {"type": "connection_lost", "duration_s": 10},
    {"type": "timeout", "duration_s": 5},
    {"type": "auth_failure", "retry_count": 3},
    {"type": "database_restart", "duration_s": 30}
]
```

#### Test Steps
1. Initialize system normally
2. Simulate Neo4j failure
3. Attempt operations during failure
4. Restore Neo4j connection
5. Verify automatic recovery
6. Check data consistency
7. Validate queued operations

#### Expected Results
- Graceful degradation during failure
- Automatic reconnection
- No data loss
- Clear error messages
- Recovery within 30 seconds

---

## 6. Performance Test Cases

### TC-PERF-001: Large Repository Indexing
**Priority**: P2 (Medium)  
**Category**: Performance, Scalability  
**Current Status**: NOT TESTED

#### Prerequisites
- Test repository with 10,000+ files
- Performance monitoring tools

#### Test Metrics
```python
performance_targets = {
    "files_per_second": 50,
    "memory_growth_mb_per_1000_files": 50,
    "cpu_usage_percent": 50,
    "io_operations_per_second": 1000,
    "total_time_10k_files_seconds": 200
}
```

#### Test Steps
1. Prepare large repository
2. Start resource monitoring
3. Initialize indexing
4. Track progress metrics
5. Monitor resource usage
6. Verify completion
7. Test search performance post-index

#### Expected Results
- Meets performance targets
- Linear scaling with file count
- Stable memory usage
- No system degradation
- Responsive during indexing

---

## Test Execution Matrix

| Test Case | Priority | Frequency | Automation | Environment |
|-----------|----------|-----------|------------|-------------|
| TC-MCP-001 | P0 | Every commit | Yes | CI/CD |
| TC-MCP-002 | P0 | Every PR | Yes | CI/CD |
| TC-ISO-001 | P0 | Every commit | Yes | CI/CD |
| TC-ISO-002 | P0 | Every PR | Yes | CI/CD |
| TC-SEARCH-001 | P1 | Daily | Yes | Integration |
| TC-SEARCH-002 | P1 | Daily | Yes | Integration |
| TC-MON-001 | P1 | Every PR | Yes | CI/CD |
| TC-ERR-001 | P1 | Weekly | Semi | Staging |
| TC-PERF-001 | P2 | Release | Yes | Performance |

## Test Data Requirements

### Standard Test Repositories
1. **Minimal**: 5 files, single language
2. **Small**: 50 files, 2 languages
3. **Medium**: 500 files, 5 languages
4. **Large**: 5000 files, 10 languages
5. **Edge Cases**: Binary files, symlinks, special chars

### Test Database States
1. **Empty**: Fresh Neo4j instance
2. **Populated**: Pre-indexed repository
3. **Corrupted**: Simulated corruption
4. **Large**: 1M+ nodes

## Automation Strategy

### Phase 1: Critical Path (Week 1)
- Automate P0 test cases
- Fix failing tests
- Add to CI/CD pipeline

### Phase 2: Extended Coverage (Week 2)
- Automate P1 test cases
- Add performance tests
- Implement test reporting

### Phase 3: Complete Automation (Week 3-4)
- Automate remaining tests
- Add stress testing
- Implement continuous monitoring

## Success Criteria

### Immediate (Week 1)
- All P0 tests passing
- Automated execution in CI
- Clear failure reporting

### Short-term (Month 1)
- 95% test pass rate
- All critical paths covered
- Performance benchmarks met

### Long-term (Quarter 1)
- 100% automation
- < 1% test flakiness
- < 5 minute test execution
- Zero production escapes