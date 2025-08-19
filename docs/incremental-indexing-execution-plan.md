# Incremental Indexing Implementation - Execution Plan

## Project: project-watch-mcp
**Created**: 2025-01-18
**Objective**: Implement incremental indexing to optimize server startup by only indexing new/changed files

## Overview

The project-watch-mcp server currently re-indexes all files on every startup, causing significant delays for large repositories. This implementation will add smart incremental indexing that detects and processes only changed files, improving startup performance by 50%+ for unchanged repositories.

## Task Breakdown & Agent Assignments

### Phase 1: Analysis & Design (Day 1)
#### Parallel Group A
- [x] **TASK-001**: Analyze current initialization flow (Completed)
  - **Outcome**: Identified cli.py lines 255-282 as insertion point
  - **Status**: Completed
  
- [ ] **TASK-002**: Create incremental indexing strategy document
  - **Assigned to**: strategic-research-analyst
  - **Outcome**: Detailed technical design document
  - **Dependencies**: None
  - **Priority**: High
  - **Status**: Not Started

### Phase 2: Core Implementation (Days 2-3)
#### Sequential Tasks - Neo4jRAG Methods
- [ ] **TASK-003**: Add is_repository_indexed() method to Neo4jRAG
  - **Assigned to**: python-developer
  - **Outcome**: Method to check if repository already indexed
  - **Dependencies**: TASK-002
  - **Priority**: High
  - **Approach**: TDD - Write tests first
  
- [ ] **TASK-004**: Add get_indexed_files() method to Neo4jRAG
  - **Assigned to**: python-developer
  - **Outcome**: Method returning indexed files with timestamps
  - **Dependencies**: TASK-003
  - **Priority**: High
  - **Approach**: TDD - Write tests first
  
- [ ] **TASK-005**: Add detect_changed_files() method
  - **Assigned to**: python-developer
  - **Outcome**: Method to identify new/modified/deleted files
  - **Dependencies**: TASK-004
  - **Priority**: High
  - **Approach**: TDD - Write tests first

- [ ] **TASK-007**: Add remove_files() method to Neo4jRAG
  - **Assigned to**: python-developer
  - **Outcome**: Method to remove deleted files from index
  - **Dependencies**: TASK-005
  - **Priority**: High
  - **Approach**: TDD - Write tests first

#### CLI Integration
- [ ] **TASK-006**: Modify cli.py to use incremental indexing
  - **Assigned to**: python-developer
  - **Outcome**: CLI using new incremental methods
  - **Dependencies**: TASK-003, TASK-004, TASK-005, TASK-007
  - **Priority**: High
  - **Approach**: TDD - Write tests first

### Phase 3: Testing Implementation (Days 4-5)
#### Parallel Group B - Unit Tests
- [ ] **TASK-008**: Neo4jRAG unit tests (test-unit-001 to test-unit-004)
  - **Assigned to**: qa-testing-expert
  - **Outcome**: Complete unit test coverage for Neo4jRAG methods
  - **Dependencies**: TASK-003, TASK-004, TASK-005, TASK-007
  - **Priority**: High
  
- [ ] **TASK-009**: Initializer unit tests (test-unit-005 to test-unit-007)
  - **Assigned to**: qa-testing-expert
  - **Outcome**: Unit tests for initialization logic
  - **Dependencies**: TASK-006
  - **Priority**: High
  
- [ ] **TASK-010**: CLI unit tests (test-unit-008 to test-unit-010)
  - **Assigned to**: qa-testing-expert
  - **Outcome**: Unit tests for CLI incremental logic
  - **Dependencies**: TASK-006
  - **Priority**: High

### Phase 4: Integration & E2E Testing (Days 6-7)
#### Sequential Tasks
- [ ] **TASK-011**: Repository initialization integration tests
  - **Assigned to**: test-automation-architect
  - **Outcome**: Integration tests for init scenarios
  - **Dependencies**: TASK-008, TASK-009, TASK-010
  - **Priority**: High
  
- [ ] **TASK-012**: File indexing integration tests
  - **Assigned to**: test-automation-architect
  - **Outcome**: Integration tests for file operations
  - **Dependencies**: TASK-011
  - **Priority**: High
  
- [ ] **TASK-013**: End-to-end workflow tests
  - **Assigned to**: test-automation-architect
  - **Outcome**: Complete workflow validation
  - **Dependencies**: TASK-012
  - **Priority**: High

### Phase 5: Performance Validation (Day 8)
- [ ] **TASK-014**: Performance and edge case tests
  - **Assigned to**: qa-testing-expert
  - **Outcome**: Performance benchmarks and edge case handling
  - **Dependencies**: TASK-013
  - **Priority**: Medium

## Execution Timeline
```
Week 1:
  Mon: Phase 1 - Analysis & Design (TASK-002)
  Tue-Wed: Phase 2 - Core Implementation (TASK-003 to TASK-007)
  Thu-Fri: Phase 3 - Unit Testing (TASK-008 to TASK-010)
  
Week 2:
  Mon-Tue: Phase 4 - Integration Testing (TASK-011 to TASK-013)
  Wed: Phase 5 - Performance Validation (TASK-014)
```

## Critical Path
1. TASK-002 → TASK-003 → TASK-004 → TASK-005 → TASK-006
2. Key bottleneck: Neo4jRAG method implementations must be complete before CLI integration

## Risk Register
| Risk | Impact | Mitigation |
|------|--------|------------|
| Neo4j version compatibility | High | Test with multiple Neo4j versions |
| Timestamp precision issues | Medium | Use file hashes as fallback |
| Large repository performance | High | Implement batching for large file sets |
| Concurrent file modifications | Medium | Add file locking mechanism |

## Success Criteria
1. **Functional Requirements**
   - ✓ Server correctly detects existing index
   - ✓ Only new/modified files are indexed on restart
   - ✓ Deleted files are removed from index
   - ✓ Full indexing works as fallback

2. **Performance Requirements**
   - ✓ 50%+ faster startup for unchanged repositories
   - ✓ 80%+ faster for <10% changed files
   - ✓ Memory usage remains constant

3. **Quality Requirements**
   - ✓ All tests passing
   - ✓ 90%+ code coverage
   - ✓ No performance regressions
   - ✓ Backward compatibility maintained

## Sub-Agent Instructions

### For python-developer:
1. Use TDD approach - write tests first
2. Follow existing code patterns in Neo4jRAG
3. Ensure proper async/await usage
4. Add comprehensive logging
5. Handle edge cases gracefully

### For qa-testing-expert:
1. Achieve 90%+ coverage for new code
2. Test both success and failure paths
3. Include edge cases and error scenarios
4. Use mocking appropriately for unit tests

### For test-automation-architect:
1. Use real Neo4j instance for integration tests
2. Test with various repository sizes
3. Validate performance improvements
4. Ensure test isolation

## Communication Protocol
1. Each agent should update task status upon completion
2. Report blockers immediately
3. Share test results and coverage metrics
4. Document any deviations from plan