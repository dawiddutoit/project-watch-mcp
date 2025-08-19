# Project Watch MCP - Task Tracker

## =� Epic: Implement Incremental Indexing on Server Restart

**Goal**: Optimize server startup by only indexing new/changed files instead of re-indexing everything.

**Context**: Currently, the MCP server re-indexes all files on every startup (see cli.py lines 255-282). This is inefficient for large repositories and causes unnecessary delays.

### Current Behavior Issues
1. **Full Re-indexing**: Every server restart triggers complete repository re-indexing
2. **Performance Impact**: Large repositories take minutes to start
3. **Resource Waste**: Unchanged files are unnecessarily re-processed
4. **No Change Detection**: System doesn't track file modification times

### Proposed Solution
Implement smart incremental indexing that:
- Checks if repository is already indexed in Neo4j
- Compares file timestamps to detect changes
- Only indexes new/modified files
- Removes deleted files from index
- Falls back to full indexing if needed

---

## =� Task Breakdown

### =, Research & Design
- [x] **task-001**: Analyze current initialization flow and identify where incremental logic should be inserted
- [ ] **task-002**: Create incremental indexing strategy document

### =� Implementation Tasks

#### Core Functionality
- [x] **task-003**: Add method to Neo4jRAG to check if repository is already indexed ✅
  - Method: `async def is_repository_indexed(project_name: str) -> bool`
  - Query Neo4j for existing File nodes with project_name
  
- [x] **task-004**: Add method to Neo4jRAG to get list of indexed files with timestamps ✅
  - Method: `async def get_indexed_files(project_name: str) -> Dict[Path, datetime]`
  - Return map of file paths to last_modified timestamps
  
- [x] **task-005**: Add method to compare file timestamps and detect changes ✅
  - Method: `async def detect_changed_files(current_files: List[FileInfo], indexed_files: Dict[Path, datetime]) -> Tuple[List[FileInfo], List[FileInfo], List[Path]]`
  - Return (new_files, modified_files, deleted_paths)
  
- [x] **task-006**: Modify cli.py main() to use incremental indexing ✅
  - Check if repository is indexed before full scan
  - Use incremental logic if index exists
  - Log statistics (new/modified/deleted/unchanged)
  
- [x] **task-007**: Add method to remove deleted files from index ✅
  - Method: `async def remove_files(project_name: str, file_paths: List[Path])`
  - Delete File nodes and relationships for removed files

---

## >� Test Requirements

### Unit Tests

#### Neo4jRAG Tests (`tests/unit/test_neo4j_rag.py`)
- [x] **test-unit-001**: `test_check_repository_indexed()` ✅
  - Test with indexed repository (returns True)
  - Test with empty repository (returns False)
  - Test with different project names
  
- [x] **test-unit-002**: `test_get_indexed_files_with_timestamps()` ✅
  - Test retrieval of indexed files with timestamps
  - Test empty repository returns empty dict
  - Test handles missing timestamp gracefully
  
- [x] **test-unit-003**: `test_detect_changed_files()` ✅
  - Test detection of new files
  - Test detection of modified files (newer timestamp)
  - Test detection of deleted files
  - Test unchanged files are ignored
  
- [x] **test-unit-004**: `test_remove_deleted_files_from_index()` ✅
  - Test removal of single file
  - Test removal of multiple files
  - Test handles non-existent files gracefully

#### Initializer Tests (`tests/unit/core/test_initializer.py`)
- [ ] **test-unit-005**: Update `test_initialize_with_valid_repo()`
  - Add parameter for incremental mode
  - Verify correct behavior with existing index
  
- [ ] **test-unit-006**: `test_initialize_incremental_with_existing_index()`
  - Mock existing index in Neo4j
  - Verify only changed files are indexed
  - Assert performance improvement
  
- [ ] **test-unit-007**: `test_initialize_incremental_detect_modified_files()`
  - Create files with specific timestamps
  - Modify subset of files
  - Verify only modified files re-indexed

#### CLI Tests (`tests/unit/test_cli.py`)
- [ ] **test-unit-008**: Update `test_main_successful_connection()`
  - Mock Neo4jRAG.is_repository_indexed()
  - Verify incremental path is taken when index exists
  
- [ ] **test-unit-009**: `test_main_with_incremental_indexing()`
  - Test main() with existing index
  - Verify incremental indexing is used
  - Check logging of statistics
  
- [ ] **test-unit-010**: `test_main_skip_unchanged_files()`
  - Create scenario with unchanged files
  - Verify they are not re-indexed
  - Check performance metrics

#### Repository Monitor Tests (`tests/unit/test_repository_monitor.py`)
- [ ] **test-unit-011**: `test_detect_file_changes_by_timestamp()`
  - Test file modification detection
  - Test timestamp comparison logic
  - Test edge cases (same timestamp, missing timestamp)

---

### Integration Tests

#### Repository Initialization (`tests/integration/server/test_repository_initialization.py`)
- [ ] **test-integration-001**: `test_incremental_initialization()`
  - Full initialization on first run
  - Incremental on second run
  - Verify correctness and performance
  
- [ ] **test-integration-002**: `test_re_initialzation_only_indexes_changed()`
  - Initialize repository
  - Modify subset of files
  - Re-initialize and verify only changed files processed

#### File Indexing (`tests/integration/server/test_file_indexing.py`)
- [ ] **test-integration-003**: Update existing `test_incremental_indexing()`
  - Align with new incremental behavior
  - Test timestamp-based change detection
  
- [ ] **test-integration-004**: `test_skip_unchanged_files_on_restart()`
  - Index repository
  - Restart without changes
  - Verify no re-indexing occurs
  
- [ ] **test-integration-005**: `test_index_new_files_on_restart()`
  - Index repository
  - Add new files
  - Restart and verify only new files indexed
  
- [ ] **test-integration-006**: `test_remove_deleted_files_on_restart()`
  - Index repository
  - Delete files
  - Restart and verify files removed from index

#### Server Startup (`tests/integration/server/test_mcp_server_startup.py`)
- [ ] **test-integration-007**: `test_server_startup_with_existing_index()`
  - Pre-populate Neo4j with indexed files
  - Start server
  - Verify incremental indexing used
  
- [ ] **test-integration-008**: `test_server_startup_incremental_performance()`
  - Measure startup time with full indexing
  - Measure startup time with incremental indexing
  - Assert significant performance improvement

---

### End-to-End Tests

- [ ] **test-e2e-001**: Create `test_incremental_indexing_workflow.py`
  - Complete workflow: init � modify � restart � verify
  - Test with real Neo4j instance
  - Verify MCP tools work correctly after incremental indexing
  
- [ ] **test-e2e-002**: Multi-project incremental indexing isolation
  - Test multiple projects with separate indexes
  - Verify incremental indexing maintains isolation
  - Test concurrent project indexing

---

### Performance & Edge Cases

- [ ] **test-performance-001**: Benchmark incremental vs full indexing
  - Create repository with 1000+ files
  - Measure full indexing time
  - Measure incremental indexing time (10% changed)
  - Assert >50% performance improvement
  
- [ ] **test-edge-001**: Handle corrupted Neo4j index
  - Simulate corrupted index state
  - Verify graceful fallback to full indexing
  - Test recovery mechanisms
  
- [ ] **test-edge-002**: Handle file timestamp edge cases
  - Test timestamp rollback scenarios
  - Test missing timestamps
  - Test future timestamps
  
- [ ] **test-edge-003**: Handle concurrent modifications
  - Test file changes during indexing
  - Test race conditions
  - Verify data consistency

---

## =� Test Coverage Requirements

### Critical Paths (100% coverage required)
- Incremental indexing decision logic
- Change detection algorithm
- File timestamp comparison
- Index update operations

### Standard Paths (90% coverage target)
- Error handling and recovery
- Logging and statistics
- Performance optimizations
- Edge case handling

### Test Execution Strategy
1. **Unit Tests First**: Implement and pass all unit tests before integration
2. **TDD Approach**: Write failing tests, then implement features
3. **Incremental Testing**: Test each component in isolation first
4. **Performance Baseline**: Establish metrics before optimization

---

## <� Success Criteria

1. **Functional Requirements**
   -  Server correctly detects existing index
   -  Only new/modified files are indexed on restart
   -  Deleted files are removed from index
   -  Full indexing works as fallback

2. **Performance Requirements**
   -  50%+ faster startup for unchanged repositories
   -  80%+ faster for <10% changed files
   -  Memory usage remains constant

3. **Quality Requirements**
   -  All tests passing
   -  90%+ code coverage
   -  No performance regressions
   -  Backward compatibility maintained

---

## =� Notes

- The `--initialize` flag behavior remains unchanged (one-time full indexing)
- Incremental indexing only applies to normal server startup
- Consider adding a `--force-reindex` flag for manual full re-indexing
- Future enhancement: Use file hashes instead of timestamps for change detection