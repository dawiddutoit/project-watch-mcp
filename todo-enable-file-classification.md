# Todo: Enable File Classification Feature

Date: 2025-08-20
Project: project-watch-mcp
Primary Language: Python

## Objective
Enable the existing file classification feature by persisting classification fields to Neo4j, adding performance safeguards with feature flags, and updating search capabilities to leverage file type filtering.

## Context
- Current state: File classification fields exist in CodeFile dataclass but are not persisted to Neo4j database
- Desired outcome: Full file classification functionality with performance-safe implementation and enhanced search capabilities  
- Constraints: Must maintain current indexing performance, require feature flags for selective enabling

## Tasks

### Task 1: Enable Neo4j Persistence of Classification Fields
**Status:** ✅ Completed
**Priority:** High
**Assigned:** @python-developer

**Description:**
Update the Neo4j indexing logic to persist all file classification fields from the CodeFile dataclass to the database.

**Implementation Checklist:**
- [ ] Update `index_file()` method in `src/project_watch_mcp/neo4j_rag.py`
  - Status: Not Started
  - Notes: Add is_test, is_config, is_resource, is_documentation, namespace, filename fields to the MERGE query
- [ ] Update `batch_index_files()` method for batch operations
  - Status: Not Started
  - Notes: Ensure batch operations include all classification fields
- [ ] Create database migration script
  - Status: Not Started
  - Notes: `scripts/migrate_file_classification.py` to update existing nodes

**Dependencies:**
- None

**Acceptance Criteria:**
- [ ] All CodeFile classification fields are stored in Neo4j
- [ ] Existing batch indexing performance is maintained
- [ ] Migration script successfully updates existing file nodes
- [ ] Unit tests validate field persistence

**Notes:**
- Fields to persist: is_test, is_config, is_resource, is_documentation, namespace, filename
- Must preserve existing functionality while adding new fields

---

### Task 2: Implement Performance Safeguards and Feature Flags
**Status:** ✅ Completed
**Priority:** High
**Assigned:** @python-developer

**Description:**
Add configurable feature flags and performance safeguards to allow selective enabling of file classification features.

**Implementation Checklist:**
- [ ] Add feature flags to configuration in `src/project_watch_mcp/config.py`
  - Status: Not Started
  - Notes: Add ENABLE_FILE_CLASSIFICATION, ENABLE_NAMESPACE_EXTRACTION flags
- [ ] Update CodeFile.__post_init__() to respect feature flags
  - Status: Not Started
  - Notes: Skip classification logic when flags are disabled
- [ ] Add environment variable support
  - Status: Not Started
  - Coverage target: PROJECT_WATCH_ENABLE_FILE_CLASSIFICATION, PROJECT_WATCH_ENABLE_NAMESPACE_EXTRACTION
- [ ] Performance monitoring in indexing methods
  - Status: Not Started
  - Notes: Add timing logs for classification overhead

**Dependencies:**
- None

**Acceptance Criteria:**
- [ ] Feature flags can disable classification without breaking functionality
- [ ] Environment variables control feature availability  
- [ ] Performance monitoring shows classification overhead
- [ ] Default configuration maintains backward compatibility

**Notes:**
- Feature flags should be opt-in initially to ensure stability
- Monitor performance impact during rollout

---

### Task 3: Enhance Search Capabilities with File Type Filtering
**Status:** ✅ Completed
**Priority:** Medium
**Assigned:** @python-developer

**Description:**
Update search methods and MCP tools to leverage file classification for enhanced filtering and discovery.

**Implementation Checklist:**
- [ ] Add file type filtering to search methods in `src/project_watch_mcp/neo4j_rag.py`
  - Status: Not Started
  - Notes: Update search_files(), search_similar_files() with classification filters
- [ ] Create new MCP tools in `src/project_watch_mcp/server.py`
  - Status: Not Started
  - Notes: Add search_by_file_type, find_test_files, find_config_files tools
- [ ] Update existing search tools to support classification parameters
  - Status: Not Started
  - Notes: Add optional file_type parameters to existing search tools
- [ ] Add Cypher queries for classification-based searches
  - Status: Not Started
  - Notes: Efficient queries for finding files by type, namespace, etc.

**Dependencies:**
- Task 1 (Neo4j persistence must be complete)

**Acceptance Criteria:**
- [ ] Can search files by classification type (test, config, resource, documentation)
- [ ] Can filter search results by namespace
- [ ] New MCP tools are available via Claude CLI
- [ ] Search performance remains under 500ms for typical queries

**Notes:**
- Leverage Neo4j indexes on classification fields for performance
- Provide clear examples in tool descriptions

---

### Task 4: Comprehensive Testing Suite
**Status:** ✅ Completed
**Priority:** Medium
**Assigned:** @qa-testing-expert

**Description:**
Create comprehensive tests covering file classification persistence, feature flags, and search capabilities.

**Implementation Checklist:**
- [ ] Unit tests for classification field persistence
  - Status: Not Started
  - Coverage target: 95%
  - Notes: `tests/unit/test_file_classification_persistence.py`
- [ ] Unit tests for feature flag functionality
  - Status: Not Started
  - Coverage target: 90%
  - Notes: `tests/unit/test_classification_feature_flags.py`
- [ ] Integration tests for search with classification
  - Status: Not Started
  - Coverage target: 85%
  - Notes: `tests/integration/test_classification_search.py`
- [ ] Performance tests for classification overhead
  - Status: Not Started
  - Notes: `tests/performance/test_classification_performance.py`

**Dependencies:**
- Task 1, 2, 3 (implementation tasks must be complete)

**Acceptance Criteria:**
- [ ] All new functionality has comprehensive test coverage
- [ ] Tests validate feature flag behavior
- [ ] Performance tests confirm acceptable overhead
- [ ] Integration tests verify end-to-end functionality

**Notes:**
- Use existing test patterns and fixtures where possible
- Include both positive and negative test cases

---

### Task 5: Update Documentation and Usage Examples
**Status:** ✅ Completed
**Priority:** Low
**Assigned:** @documentation-architect

**Description:**
Document the new file classification features and provide clear usage examples.

**Implementation Checklist:**
- [ ] Update project README with classification features
  - Status: Not Started
  - Notes: Add section on file type detection and search capabilities
- [ ] Update MCP tool documentation
  - Status: Not Started
  - Notes: Document new search tools and filtering options
- [ ] Create configuration guide for feature flags
  - Status: Not Started
  - Notes: `docs/file-classification-config.md`
- [ ] Add usage examples
  - Status: Not Started
  - Notes: Examples of searching by file type, namespace filtering

**Dependencies:**
- Task 1, 2, 3 (implementation complete)

**Acceptance Criteria:**
- [ ] README clearly explains file classification capabilities
- [ ] Feature flag configuration is well documented
- [ ] Usage examples demonstrate key functionality
- [ ] MCP tools have clear descriptions and examples

**Notes:**
- Focus on practical usage patterns
- Include performance considerations in documentation

---

## Progress Summary
- Total Tasks: 5
- Completed: 5 (100%)
- In Progress: 0
- Blocked: 0

## Technical Implementation Details

### Database Schema Changes
```cypher
// Updated CodeFile node properties
MERGE (f:CodeFile {project_name: $project_name, path: $path})
SET f.language = $language,
    f.size = $size, 
    f.lines = $lines,
    f.last_modified = $last_modified,
    f.hash = $hash,
    f.filename = $filename,
    f.namespace = $namespace,
    f.is_test = $is_test,
    f.is_config = $is_config,
    f.is_resource = $is_resource,
    f.is_documentation = $is_documentation
```

### Feature Flag Configuration
```python
# config.py additions
@dataclass
class ClassificationConfig:
    enable_file_classification: bool = False
    enable_namespace_extraction: bool = False
    classification_performance_logging: bool = False
```

### New MCP Tools
- `search_by_file_type` - Search files by classification type
- `find_test_files` - Find all test files in project
- `find_config_files` - Find configuration files
- `search_by_namespace` - Search files within specific namespace

### Performance Targets
- Classification overhead: < 10ms per file
- Search with classification: < 500ms
- Batch indexing performance: maintain current 50x+ improvement

## Session Notes
- File classification fields already exist in CodeFile dataclass
- Current indexing logic needs updating to persist classification data
- Feature flags essential for safe rollout
- Search enhancements will provide significant value for code navigation