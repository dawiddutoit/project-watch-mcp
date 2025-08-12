# Feature Todo: CLI Initialization Refactoring

**This document contains the complete implementation plan, test requirements, and quality gates for the CLI initialization refactoring.**

## Table of Contents
1. [Overview](#overview)
2. [End Goal](#end-goal-session-hook-behavior)
3. [Architecture Changes](#architecture-changes)
4. [Implementation Tasks](#implementation-tasks)
   - Phase 1: Core Module Extraction (with embedded tests)
   - Phase 2: Comprehensive Testing
   - Phase 3: Documentation
5. [Testing Requirements](#testing-requirements)
6. [Debugging Considerations](#debugging-considerations)
7. [Agent Assignments](#agent-assignments)
8. [Success Criteria](#success-criteria)
9. [Rollback Plan](#rollback-plan)

## Overview
Refactor the Project Watch MCP to support `--initialize` CLI flag by extracting initialization logic into a shared core module, eliminating the 200+ lines of duplicated code in the session-start hook.

**Target Command**: `uv run project-watch-mcp --initialize`  
**Estimated Effort**: 4-6 hours  
**Priority**: Critical (eliminates maintenance nightmare)

## End Goal: Session Hook Behavior

### Current Problem (Before Refactoring)
The session-start hook at `.claude/hooks/session-start/session-start.py` contains 200+ lines that duplicate the server's initialization logic. This creates maintenance issues where bugs must be fixed in two places.

### Desired Solution (After Refactoring)
The session-start hook should be a simple ~50 line script that:

```python
#!/usr/bin/env python3
# Simplified session-start.py after refactoring
import subprocess
import json
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent.parent
    
    # Check if auto-init should be skipped
    if (project_root / ".claude" / ".skip_auto_init").exists():
        print(json.dumps({"status": "skipped"}))
        return 0
    
    # Run the CLI initialization command
    result = subprocess.run(
        ['uv', 'run', 'project-watch-mcp', '--initialize'],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Format output for Claude
    if result.returncode == 0:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "✅ Repository auto-initialized successfully! " + result.stdout,
                "initializationStatus": "success"
            }
        }
    else:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "⚠️ Auto-initialization failed. Please run: mcp__project-watch-local__initialize_repository",
                "initializationStatus": "failed"
            }
        }
    
    print(json.dumps(output))
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
```

### Key Benefits of This Approach
1. **No Code Duplication**: Hook delegates to CLI, which uses the same core module as the server
2. **Single Source of Truth**: All initialization logic lives in `core/initializer.py`
3. **Easy Maintenance**: Changes to initialization logic only need to be made in one place
4. **Clear Separation**: Hook handles output formatting, CLI handles execution, core handles logic
5. **Testable**: Each layer can be tested independently

## Architecture Changes

### New Module Structure
```
src/project_watch_mcp/
├── core/                           # NEW directory
│   ├── __init__.py                # NEW
│   └── initializer.py             # NEW - Shared initialization logic
├── cli.py                         # MODIFY - Add --initialize flag
├── server.py                      # MODIFY - Use core.initializer
└── (other existing files)

.claude/hooks/session-start/
└── session-start.py               # MODIFY - Simplify to use CLI
```

## Implementation Tasks

### Phase 1: Core Module Extraction (2-3 hours)

#### Task 1.1: Create Core Initializer Module
**Files**: `src/project_watch_mcp/core/__init__.py`, `src/project_watch_mcp/core/initializer.py`
**Agent**: backend-system-architect
**Description**:
- Create `RepositoryInitializer` class with async `initialize()` method
- Extract initialization logic from `server.py`'s `initialize_repository` tool
- Add progress callback support for CLI reporting
- Implement `InitializationResult` dataclass for structured responses
- Ensure proper error handling and logging

**Test Requirements**:
- **Coverage Target**: 90% (100% for critical paths)
- **Unit Tests** (`tests/test_initializer.py`):
  - [ ] RepositoryInitializer class instantiation
  - [ ] Async initialize method with valid repo (happy path)
  - [ ] Progress callback invocation and accuracy
  - [ ] File filtering with gitignore patterns
  - [ ] Error handling for invalid paths
  - [ ] Error handling for Neo4j failures
  - [ ] Resource cleanup on exceptions
  - [ ] Concurrent initialization prevention
  - [ ] Memory leak detection with tracemalloc
- **Property-Based Tests**:
  - [ ] Path handling with Hypothesis
  - [ ] File encoding edge cases (UTF-8, UTF-16, ASCII)
  - [ ] Boundary testing for file size limits
- **Performance Tests**:
  - [ ] Benchmark initialization for 100, 500, 1000 files
  - [ ] Memory usage profiling
  - [ ] Async operation overhead measurement

#### Task 1.2: Refactor Server to Use Core Module
**File**: `src/project_watch_mcp/server.py`
**Agent**: python-developer
**Description**:
- Import `RepositoryInitializer` from core module
- Modify `initialize_repository` tool to delegate to initializer
- Maintain backward compatibility with existing MCP interface
- Preserve all existing tool annotations and error handling

**Test Requirements**:
- **Coverage Target**: 85% (100% for compatibility layer)
- **Integration Tests**:
  - [ ] MCP tool continues to work with existing clients
  - [ ] Tool annotations preserved (idempotent, non-destructive)
  - [ ] Structured output format unchanged
  - [ ] Error responses maintain same format
- **Regression Tests**:
  - [ ] All existing server tests still pass
  - [ ] No breaking changes to public API
  - [ ] Performance not degraded (±10% tolerance)

#### Task 1.3: Add CLI Initialization Support
**File**: `src/project_watch_mcp/cli.py`
**Agent**: python-developer
**Description**:
- Add `--initialize` flag (mutually exclusive with `--transport`)
- Implement `initialize_only()` async function
- Add progress reporting for verbose mode
- Ensure proper Neo4j connection handling and cleanup
- Exit with appropriate status codes (0 success, 1 failure)

**Test Requirements**:
- **Coverage Target**: 85% (100% for flag parsing)
- **Unit Tests** (`tests/test_cli_initialize.py`):
  - [ ] --initialize flag parsing
  - [ ] Mutual exclusivity with --transport
  - [ ] Repository path resolution (relative, absolute, ~)
  - [ ] Environment variable handling
  - [ ] Verbose mode output formatting
  - [ ] Exit codes: 0 (success), 1 (failure)
- **Integration Tests**:
  - [ ] End-to-end initialization with real Neo4j
  - [ ] Progress reporting in verbose mode
  - [ ] Signal handling (SIGINT, SIGTERM)
  - [ ] Timeout handling (60s default)
- **Security Tests**:
  - [ ] Path traversal prevention
  - [ ] Command injection prevention
  - [ ] Resource exhaustion protection

#### Task 1.4: Simplify Session Hook
**File**: `.claude/hooks/session-start/session-start.py`
**Agent**: python-developer
**Description**:
- Replace 200+ lines of duplicated logic with subprocess call
- Use `subprocess.run(['uv', 'run', 'project-watch-mcp', '--initialize'])`
- Maintain proper error handling and output formatting
- Keep initialization status markers (`.last_auto_init`)
- Reduce from ~400 lines to ~50 lines
- Hook should ONLY:
  - Check skip markers
  - Call CLI command
  - Format output for Claude
  - Return appropriate exit code

**Test Requirements**:
- **Coverage Target**: 80% (100% for error paths)
- **Unit Tests** (`tests/test_session_hook.py`):
  - [ ] Skip marker detection (.skip_auto_init)
  - [ ] Subprocess invocation with correct args
  - [ ] Timeout handling (60s)
  - [ ] Output formatting for success
  - [ ] Output formatting for failure
  - [ ] Exit code propagation
- **Integration Tests**:
  - [ ] Full session start workflow
  - [ ] Marker file creation (.last_auto_init)
  - [ ] JSON output validity
  - [ ] Claude integration compatibility

### Phase 2: Comprehensive Testing (2 hours)

#### Task 2.1: Test Infrastructure Setup
**Agent**: test-automation-architect
**Description**: Set up testing infrastructure and verify all test dependencies

**Pre-Test Checklist**:
- [ ] Neo4j test instance available (version 5.11+)
- [ ] Test database cleanup scripts ready
- [ ] Sample repositories created (minimal, small, medium, large)
- [ ] Edge case files prepared (Unicode, special chars, large files)
- [ ] Testing tools installed (pytest, pytest-asyncio, pytest-cov, hypothesis, pytest-benchmark)

#### Task 2.2: Create Comprehensive Test Suite
**Agent**: test-automation-architect
**Description**: Implement full test coverage across all components

**Test Matrix Coverage Requirements**:
| Component | Unit | Integration | E2E | Security | Performance | Target |
|-----------|------|-------------|-----|----------|-------------|--------|
| `core/initializer.py` | ✅ | ✅ | ✅ | ✅ | ✅ | 90% |
| `cli.py --initialize` | ✅ | ✅ | ✅ | ✅ | ✅ | 85% |
| `server.py` refactored | ✅ | ✅ | ✅ | ❌ | ✅ | 85% |
| Session Hook | ✅ | ✅ | ✅ | ✅ | ❌ | 80% |

**Test Scenarios by Priority**:
- **P0 (Release Blocking)**:
  - [ ] Basic initialization succeeds
  - [ ] CLI flag works correctly
  - [ ] MCP tool maintains compatibility
  - [ ] Session hook delegates to CLI
  - [ ] No code duplication remains
  
- **P1 (High Priority)**:
  - [ ] Error handling for Neo4j failures
  - [ ] Gitignore patterns respected
  - [ ] Progress reporting accurate
  - [ ] Resource cleanup on failure
  - [ ] Timeout handling (60s)
  
- **P2 (Medium Priority)**:
  - [ ] Unicode file handling
  - [ ] Large repository performance
  - [ ] Concurrent initialization prevention
  - [ ] Signal handling (SIGINT/SIGTERM)
  - [ ] Memory leak detection
  
- **P3 (Nice to Have)**:
  - [ ] Cross-platform validation
  - [ ] Accessibility features
  - [ ] Alternative Neo4j versions
  - [ ] Network failure recovery

#### Task 2.3: Security & Performance Validation
**Agent**: code-review-expert
**Description**: Validate security boundaries and performance benchmarks

**Security Tests**:
- [ ] Path traversal prevention (../../../etc/passwd)
- [ ] Command injection prevention in subprocess
- [ ] Resource exhaustion protection (file bombs)
- [ ] Neo4j injection prevention
- [ ] Environment variable sanitization

**Performance Benchmarks**:
- [ ] < 30s for 500 files
- [ ] < 500MB memory usage
- [ ] < 100ms per file average
- [ ] No performance regression vs current (±10%)

### Phase 3: Documentation (1 hour)

#### Task 3.1: Update README
**File**: `README.md`
**Agent**: documentation-architect
**Description**:
- Document new `--initialize` flag usage
- Add examples for different initialization scenarios
- Update architecture section if present

**Documentation Requirements**:
- [ ] CLI usage examples with --initialize
- [ ] Architecture diagram showing core module
- [ ] Migration notes for existing users
- [ ] Troubleshooting section

#### Task 3.2: Update CLAUDE.md
**File**: `CLAUDE.md`
**Agent**: documentation-architect
**Description**:
- Document new module structure
- Update development guidelines
- Add notes about initialization patterns

**Documentation Requirements**:
- [ ] Module structure explanation
- [ ] Hook behavior documentation
- [ ] Development workflow updates
- [ ] Testing instructions

#### Task 3.3: Create Migration Guide
**File**: `docs/MIGRATION.md` (NEW)
**Agent**: documentation-architect
**Description**:
- Document changes for existing users
- Explain hook simplification
- Provide troubleshooting steps

**Documentation Requirements**:
- [ ] Breaking changes (if any)
- [ ] Hook migration instructions
- [ ] Rollback procedures
- [ ] FAQ section

## Testing Requirements

### Test Coverage Requirements

#### Module-Specific Coverage Targets
1. **Core Modules** (Must achieve 100% coverage for critical paths):
   - `core/initializer.py`: ≥90% overall, 100% for error handling paths
   - `cli.py` (initialize function): ≥85% overall, 100% for flag validation
   - `server.py` (refactored tool): ≥85% overall, 100% for backward compatibility

2. **Integration Points** (High coverage required):
   - Neo4j connection handling: ≥95% coverage
   - File system operations: ≥90% coverage
   - Progress reporting callbacks: ≥85% coverage
   - Error propagation paths: 100% coverage

3. **Session Hook** (Critical compatibility):
   - Subprocess execution: ≥80% coverage
   - Error handling and retry logic: 100% coverage
   - Status marker management: ≥90% coverage

#### Critical Paths Requiring 100% Coverage
- **Error Handling**: All exception paths must be tested
- **Resource Cleanup**: Neo4j connection closure, file handles
- **State Consistency**: Initialization markers, index state
- **Data Validation**: Input sanitization, path validation
- **Security Boundaries**: Path traversal prevention, injection protection

### Test Scenarios

#### Unit Test Requirements (Task 2.1)

**Happy Path Scenarios**:
1. Initialize repository with typical Python project (50-100 files)
2. Initialize repository with mixed languages (Python, JS, TS, MD)
3. Initialize with custom gitignore patterns
4. Progressive callback reporting (0%, 25%, 50%, 75%, 100%)
5. Successful Neo4j vector index creation
6. Proper embedding generation for all file types

**Edge Cases**:
1. Empty repository (0 files matching patterns)
2. Single file repository
3. Repository with 10,000+ files (performance boundary)
4. Files with Unicode/special characters in names
5. Symbolic links and circular references
6. Repository at filesystem root
7. Hidden files and directories (.git, .env)
8. Binary files mixed with source code
9. Very large individual files (>10MB)
10. Deeply nested directory structures (>20 levels)

**Error Conditions**:
1. Invalid repository path (non-existent, not a directory)
2. No read permissions on repository
3. Neo4j connection timeout (network issues)
4. Neo4j authentication failure
5. Neo4j version incompatibility (< 5.11)
6. Corrupted/malformed source files
7. Disk space exhaustion during indexing
8. Memory exhaustion with large files
9. Concurrent initialization attempts
10. Interrupted initialization (SIGINT, SIGTERM)

#### Integration Test Requirements (Task 2.2)

**CLI Integration Scenarios**:
1. **Flag Validation**:
   - `--initialize` without repository path (should use current dir)
   - `--initialize` with explicit repository path
   - `--initialize --transport stdio` (mutual exclusivity error)
   - `--initialize --verbose` (progress output validation)
   - `--initialize --help` (help text includes new flag)

2. **Environment Variable Handling**:
   - Missing Neo4j credentials (should fail gracefully)
   - Partial credentials provided
   - Override via CLI args vs environment
   - Invalid URI formats

3. **Exit Code Validation**:
   - Success: exit(0) with indexed file count
   - Connection failure: exit(1) with error message
   - Invalid arguments: exit(2) with usage
   - Initialization failure: exit(3) with details
   - Keyboard interrupt: exit(130)

4. **Output Format Testing**:
   - JSON output mode for automation
   - Human-readable output for terminal
   - Progress bar in TTY environment
   - No progress in non-TTY (CI/CD)
   - Proper stderr vs stdout usage

#### Regression Testing Requirements (Task 2.3)

**Backward Compatibility Tests**:
1. **MCP Tool Interface**:
   - Existing tool signature unchanged
   - Response format compatibility
   - Error response structure maintained
   - Tool metadata/description preserved

2. **Performance Regression Tests**:
   - Indexing speed within 10% of baseline
   - Memory usage within 20% of baseline
   - Neo4j query performance unchanged
   - File monitoring latency < 100ms

3. **Session Hook Compatibility**:
   - Hook discovers and executes CLI correctly
   - Environment variable propagation
   - Working directory preservation
   - Signal handling compatibility

### Test Data Requirements

#### Sample Repository Structures
1. **Minimal Test Repo** (5-10 files):
   - Basic Python package with __init__.py
   - Simple README.md
   - Basic .gitignore
   - One test file

2. **Typical Project** (50-100 files):
   - Multi-module Python project
   - Tests directory with fixtures
   - Documentation in markdown
   - Configuration files (YAML, TOML, JSON)
   - Mixed file types and sizes

3. **Large Repository** (1000+ files):
   - Generated code files for scale testing
   - Nested package structures
   - Multiple programming languages
   - Large documentation sets

4. **Edge Case Repository**:
   - Files with special characters: `test-file (copy).py`, `código.js`
   - Very long filenames (255 chars)
   - Deeply nested paths
   - Mixed line endings (CRLF, LF)
   - Various encodings (UTF-8, UTF-16, ASCII)

#### Neo4j Test Database Setup
1. **Test Instance Requirements**:
   - Neo4j 5.11+ with vector index support
   - Clean database per test run
   - Known seed data for deterministic tests
   - Performance baseline measurements

2. **Mock Strategies**:
   - Full mock for unit tests (no real Neo4j)
   - Embedded Neo4j for integration tests
   - Docker container for E2E tests
   - Connection failure simulation

#### Mock Data Requirements
1. **File Content Mocks**:
   - Syntactically valid code samples
   - Edge cases (empty files, single line)
   - Large content (performance testing)
   - Malformed content (error handling)

2. **Embedding Mocks**:
   - Deterministic vectors for testing
   - Dimension validation (384D)
   - Similarity threshold testing
   - Batch processing validation

### Acceptance Criteria

#### Release Blocking Criteria (Must Pass)
1. **Functional Requirements**:
   - ✅ CLI --initialize successfully indexes ≥95% of test repositories
   - ✅ MCP tool maintains 100% backward compatibility
   - ✅ Session hook executes without manual intervention
   - ✅ No memory leaks during 1-hour continuous operation
   - ✅ Handles repositories up to 10,000 files

2. **Performance Requirements**:
   - ✅ Initialization < 30s for 500-file repository
   - ✅ Memory usage < 500MB for typical repository
   - ✅ Neo4j query response < 100ms for searches
   - ✅ File monitoring latency < 50ms
   - ✅ Progress callback frequency ≥ 10Hz

3. **Quality Requirements**:
   - ✅ Zero critical security vulnerabilities
   - ✅ Test coverage ≥ 85% overall
   - ✅ All linting checks pass (ruff, black)
   - ✅ Type checking passes (pyright)
   - ✅ Documentation complete and accurate

#### Non-Blocking but Tracked
1. **Performance Optimizations**:
   - Initialization < 10s for 500 files (target)
   - Memory usage < 200MB (target)
   - Parallel file processing

2. **Enhanced Features**:
   - Incremental re-initialization
   - Initialization resume capability
   - Detailed progress breakdown

### Test Automation Strategy

#### CI/CD Pipeline Tests

**On Every Commit** (~ 5 minutes):
1. Unit tests with coverage report
2. Linting (ruff, black --check)
3. Type checking (pyright)
4. Security scanning (bandit)
5. License compliance check

**On Pull Request** (~ 15 minutes):
1. All commit tests
2. Integration tests with mock Neo4j
3. Regression test suite
4. Performance baseline comparison
5. Documentation build verification

**Pre-Release** (~ 30 minutes):
1. Full E2E tests with real Neo4j
2. Multiple Python version testing (3.11, 3.12)
3. Cross-platform testing (Linux, macOS, Windows)
4. Large repository stress tests
5. Memory leak detection

**Manual Testing Requirements**:
1. User acceptance testing with real repositories
2. Session hook integration in Claude Desktop
3. Performance profiling on large codebases
4. Security penetration testing
5. Accessibility testing (screen readers)

#### Test Automation Tools
1. **Test Execution**: pytest with pytest-asyncio
2. **Coverage**: pytest-cov with codecov.io
3. **Mocking**: unittest.mock, pytest-mock
4. **Performance**: pytest-benchmark
5. **Property Testing**: hypothesis
6. **Mutation Testing**: mutmut
7. **Load Testing**: locust for Neo4j operations

### Additional Test Scenarios

#### Security Testing
1. **Path Traversal Prevention**:
   - Test with paths like `../../etc/passwd`
   - Symbolic link exploitation attempts
   - Unicode normalization attacks

2. **Injection Prevention**:
   - Neo4j Cypher injection in file paths
   - Command injection in subprocess calls
   - Environment variable injection

3. **Resource Exhaustion**:
   - Memory bombs (files generating huge embeddings)
   - Fork bombs (recursive initialization)
   - Disk exhaustion handling

#### Chaos Engineering
1. **Network Chaos**:
   - Intermittent Neo4j disconnections
   - High latency conditions
   - Packet loss simulation

2. **System Chaos**:
   - Process kills during initialization
   - Disk full during indexing
   - CPU throttling scenarios

3. **Data Chaos**:
   - Corrupted file system entries
   - Race conditions in concurrent access
   - Clock skew effects

#### Accessibility Testing
1. **CLI Accessibility**:
   - Screen reader compatibility
   - Color-blind friendly output
   - Keyboard-only navigation
   - Clear error messages for all users

2. **Progress Reporting**:
   - Non-visual progress indicators
   - Structured logging for automation
   - Machine-readable output formats

## Debugging Considerations

### 1. Potential Debugging Challenges

#### Async Execution Context Issues
- **Challenge**: Debugging async code paths across three different execution contexts (CLI, MCP server, session hook subprocess)
- **Symptoms**: Hanging processes, unclosed connections, event loop errors like "RuntimeError: There is no current event loop"
- **Debug Approach**: 
  - Add `asyncio.get_running_loop()` checks at critical points
  - Use `PYTHONASYNCIODEBUG=1` environment variable during development
  - Implement proper async context managers with explicit cleanup
  - Add timeout wrappers around all async operations

#### Module Import and Dependency Hell
- **Challenge**: Complex import chain from hook → CLI → core → Neo4j/embeddings with potential circular dependencies
- **Symptoms**: ImportError, AttributeError on module access, inconsistent behavior between direct execution and imports
- **Debug Approach**:
  - Use `python -X importtime` to profile import performance
  - Add `__all__` exports to control public API surface
  - Implement lazy imports for heavy dependencies
  - Create import dependency graph with `pydeps` tool

#### State Synchronization Across Processes
- **Challenge**: Three separate processes (hook, CLI, server) trying to coordinate initialization state
- **Symptoms**: Race conditions, duplicate initializations, inconsistent index state
- **Debug Approach**:
  - Implement distributed locking using Neo4j or file system locks
  - Add UUID-based initialization tracking
  - Use atomic file operations for state markers
  - Implement state machine with clear transitions

#### Error Propagation Through Layers
- **Challenge**: Errors must bubble up through subprocess → hook → Claude output → user understanding
- **Symptoms**: Silent failures, cryptic error messages, lost stack traces
- **Debug Approach**:
  - Implement structured error types with error codes
  - Add correlation IDs for tracing errors across processes
  - Use exception chaining to preserve original context
  - Log full stack traces while returning user-friendly messages

### 2. Maintainability Analysis

#### Architecture Assessment
**Strengths of Proposed Architecture:**
- **Clear Separation of Concerns**: Core logic isolated from transport layers (CLI/MCP)
- **DRY Principle**: Eliminates 200+ lines of duplicate code
- **Testability**: Core module can be unit tested independently
- **Extensibility**: Easy to add new initialization modes (API, scheduled, etc.)

**Potential Maintainability Issues:**
- **Hidden Complexity**: Three-layer architecture may obscure simple bugs
- **Async Abstraction Leak**: Async requirements of core module force all consumers to be async
- **Configuration Proliferation**: Each layer may need its own config handling
- **Version Coupling**: All three components must be version-synchronized

#### Code Smell Prevention
**Anti-patterns to Avoid:**
1. **God Object**: Don't let `RepositoryInitializer` become a dumping ground for all initialization logic
2. **Leaky Abstractions**: Don't expose Neo4j-specific details in the core API
3. **Shotgun Surgery**: Changes shouldn't require modifications across all three layers
4. **Feature Envy**: Each module should work with its own data, not reach into others

**Best Practices to Enforce:**
- Use dependency injection for all external services
- Implement interfaces/protocols for pluggable components
- Keep methods under 50 lines, classes under 200 lines
- Maintain cyclomatic complexity below 10

#### Long-term Sustainability
**Will this solve the duplicate code problem?** 
- **Yes**, if properly implemented with clear boundaries
- **Risk**: Without discipline, logic may leak back into consuming layers
- **Mitigation**: Code review checklist, architectural decision records (ADRs)

### 3. Error Handling Strategy

#### Error Hierarchy and Propagation
```python
# Proposed error hierarchy
class InitializationError(Exception):
    """Base class for all initialization errors"""
    error_code: str
    user_message: str
    technical_details: dict
    
class ConnectionError(InitializationError):
    """Neo4j connection issues"""
    
class FileAccessError(InitializationError):
    """Repository file access issues"""
    
class IndexingError(InitializationError):
    """Neo4j indexing failures"""
```

#### Error Propagation Rules
1. **Core → Server**: Exceptions wrapped in ToolError with structured content
2. **Core → CLI**: Exceptions converted to exit codes with stderr output
3. **CLI → Hook**: Exit codes and stderr captured, parsed for user display
4. **Hook → Claude**: JSON structured output with fallback instructions

#### Logging Architecture
```python
# Structured logging configuration
{
    "timestamp": "ISO-8601",
    "level": "INFO|WARN|ERROR|DEBUG",
    "component": "core|cli|server|hook",
    "correlation_id": "uuid",
    "operation": "initialize|index|scan",
    "duration_ms": 1234,
    "error_code": "NEO4J_CONN_001",
    "details": {}
}
```

### 4. Observability Requirements

#### Metrics and Telemetry
**Key Metrics to Track:**
- `initialization.duration` - Total time from start to completion
- `initialization.files_scanned` - Number of files discovered
- `initialization.files_indexed` - Successfully indexed count
- `initialization.failures` - Failed file count by error type
- `neo4j.connection.latency` - Database connection time
- `neo4j.query.duration` - Individual query performance
- `memory.heap_used` - Memory consumption during indexing
- `process.cpu_percent` - CPU usage during operations

**Implementation Approach:**
- Use `prometheus_client` for metrics collection
- Implement custom context managers for timing operations
- Add memory profiling with `tracemalloc` in debug mode
- Export metrics via `/metrics` endpoint in server mode

#### Distributed Tracing
**Trace Points:**
1. Hook entry → CLI subprocess spawn
2. CLI start → Core initialization
3. Core → Neo4j operations
4. Core → File system operations
5. Core → Embedding generation

**Implementation:**
- Use OpenTelemetry for standardized tracing
- Correlation IDs passed via environment variables
- Span attributes include file counts, error codes, durations

#### Debug Modes and Flags
```bash
# Proposed debug flags
--debug              # Verbose logging, no timeouts
--trace              # OpenTelemetry tracing enabled
--profile            # Performance profiling output
--dry-run            # Scan but don't index
--explain            # Show what would be done
--validate-only      # Check configuration without running
--benchmark          # Run performance benchmarks
```

### 5. Common Failure Points and Prevention

#### Likely Failure Scenarios

**1. Neo4j Connection Pool Exhaustion**
- **Cause**: Unclosed connections in error paths
- **Prevention**: Use connection pool with max size, implement circuit breaker
- **Detection**: Monitor active connection count
- **Recovery**: Automatic pool recycling after threshold

**2. Memory Exhaustion on Large Files**
- **Cause**: Loading entire file contents into memory
- **Prevention**: Stream processing for files >10MB
- **Detection**: Monitor process RSS, implement memory guards
- **Recovery**: Graceful degradation, skip large files with warning

**3. Deadlock in Async Operations**
- **Cause**: Improper await chains, blocking I/O in async context
- **Prevention**: Use `asyncio.create_task()` for concurrent operations
- **Detection**: Implement operation timeouts, deadlock detection
- **Recovery**: Timeout and retry with exponential backoff

**4. Race Condition in Concurrent Initialization**
- **Cause**: Multiple Claude sessions starting simultaneously
- **Prevention**: Distributed lock using Neo4j MERGE operation
- **Detection**: Check for lock before proceeding
- **Recovery**: Wait and retry or fail fast with clear message

**5. Corrupt Index State**
- **Cause**: Partial initialization failure leaving inconsistent state
- **Prevention**: Transactional operations, two-phase commit
- **Detection**: Index integrity checks on startup
- **Recovery**: Automatic cleanup and re-indexing

#### Debugging Tools and Techniques

**Development Phase:**
- `pytest-asyncio` for async test debugging
- `pytest-timeout` to catch hanging tests
- `hypothesis` for property-based testing of edge cases
- `memory_profiler` for memory leak detection
- `py-spy` for performance profiling without instrumentation

**Production Debugging:**
- Structured logging with correlation IDs
- Neo4j query logging for slow query analysis
- Process dumps for post-mortem debugging
- Distributed tracing for cross-process debugging
- Health check endpoints with detailed status

### 6. Code Review Checklist

#### Architecture Review
- [ ] No circular dependencies between modules
- [ ] Core module has no knowledge of CLI or server specifics
- [ ] All async functions have proper timeout handling
- [ ] Connection lifecycle is explicitly managed
- [ ] Error types are properly structured and documented

#### Code Quality
- [ ] All public methods have comprehensive docstrings
- [ ] Type hints are complete and accurate
- [ ] No synchronous I/O in async contexts
- [ ] Resource cleanup in finally blocks or context managers
- [ ] Logging at appropriate levels (no INFO in loops)

#### Error Handling
- [ ] All exceptions are caught and wrapped appropriately
- [ ] Error messages are actionable for users
- [ ] Stack traces are preserved in logs
- [ ] Fallback behavior is clearly defined
- [ ] No bare except clauses

#### Testing
- [ ] Unit tests cover happy path and error cases
- [ ] Integration tests verify cross-module interactions
- [ ] Performance tests establish baselines
- [ ] Chaos tests verify failure recovery
- [ ] Mock objects don't hide real issues

#### Security
- [ ] No path traversal vulnerabilities
- [ ] Neo4j queries are parameterized
- [ ] Sensitive data not logged
- [ ] Resource limits enforced
- [ ] Input validation on all boundaries

#### Common Implementation Mistakes
- [ ] Forgetting to close Neo4j driver in error paths
- [ ] Using `asyncio.run()` inside async context
- [ ] Not handling `KeyboardInterrupt` gracefully
- [ ] Hardcoding timeouts instead of making configurable
- [ ] Assuming file system operations are atomic
- [ ] Not validating Neo4j version compatibility
- [ ] Missing memory limits for large repositories
- [ ] Forgetting to handle Windows path separators
- [ ] Not escaping special characters in regex patterns
- [ ] Assuming UTF-8 encoding for all files

### Final Assessment: Architecture Viability

**Will this refactoring improve maintainability?**

**YES, with caveats.** The refactoring will significantly improve maintainability by:
1. **Eliminating code duplication** - Single source of truth for initialization logic
2. **Improving testability** - Core logic can be tested in isolation
3. **Enabling reusability** - Same logic works for CLI, server, and future interfaces
4. **Clarifying responsibilities** - Each layer has a clear, single purpose

**However, it could introduce new problems if:**
1. **Over-abstraction** - Making simple operations complex
2. **Tight coupling** - If layers aren't properly isolated
3. **Hidden complexity** - Debugging becomes harder across layers
4. **Performance overhead** - Additional abstraction layers add latency

**Recommendations for Success:**
1. **Start simple** - Extract only the essential shared logic first
2. **Maintain boundaries** - Use protocols/interfaces to prevent coupling
3. **Instrument thoroughly** - Add comprehensive logging and metrics
4. **Test exhaustively** - Especially integration and error paths
5. **Document decisions** - Maintain ADRs for architectural choices
6. **Monitor continuously** - Track performance and error rates post-deployment

**Risk Assessment:** 
- **Low risk** if implemented incrementally with proper testing
- **Medium risk** if rushed or if boundaries aren't well-defined
- **High risk** if async complexity isn't properly managed

The refactoring is worth pursuing but requires disciplined implementation and comprehensive testing to avoid introducing new maintenance burdens.

## Agent Assignments

### Phase 1: Core Module Extraction

**Task 1.1: Create Core Initializer Module**
- **Agent**: `backend-system-architect`
- **Rationale**: This task involves designing a new module structure and extracting core initialization logic. Requires architectural thinking about separation of concerns, proper abstraction patterns, and designing clean interfaces between components.

**Task 1.2: Refactor Server to Use Core Module**
- **Agent**: `python-developer`
- **Rationale**: Primarily a code refactoring task that requires deep understanding of Python imports, dependency injection patterns, and maintaining backward compatibility. Perfect for a TDD-focused Python developer.

**Task 1.3: Add CLI Initialization Support**
- **Agent**: `python-developer`
- **Rationale**: CLI implementation with argument parsing, async function development, and proper exit code handling. This is straightforward Python development work that benefits from TDD practices.

**Task 1.4: Simplify Session Hook**
- **Agent**: `python-developer`
- **Rationale**: Converting complex logic to subprocess calls and maintaining error handling. This is a code simplification task that requires Python expertise but is relatively straightforward.

### Phase 2: Testing

**Task 2.1: Unit Tests for Core Module**
- **Agent**: `test-automation-architect`
- **Rationale**: Requires expertise in test strategy, mocking patterns, and comprehensive test coverage design. The test architect can design proper test suites with appropriate mocking of Neo4j and file system operations.

**Task 2.2: Integration Tests for CLI**
- **Agent**: `test-automation-architect`
- **Rationale**: Integration testing requires understanding of CLI testing patterns, subprocess testing, and end-to-end scenario design. Best handled by testing specialist.

**Task 2.3: Update Existing Tests**
- **Agent**: `code-review-expert`
- **Rationale**: This involves reviewing existing tests for regression issues and ensuring compatibility with refactored code. Code review expertise is needed to identify potential breaking changes and maintain test quality.

### Phase 3: Documentation

**Task 3.1: Update README**
- **Agent**: `documentation-architect`
- **Rationale**: User-facing documentation requiring clear technical writing, usage examples, and proper documentation structure. Documentation specialist ensures consistency and clarity.

**Task 3.2: Update CLAUDE.md**
- **Agent**: `documentation-architect`
- **Rationale**: Developer-focused documentation about architecture and development patterns. Requires understanding of both the technical changes and how to communicate them to future developers.

**Task 3.3: Create Migration Guide**
- **Agent**: `documentation-architect`
- **Rationale**: Migration documentation requires understanding user impact and clear communication of breaking changes. Documentation architect can create comprehensive troubleshooting guides.

## Agent Assignment Rationale

### Why These Specific Assignments?

1. **Architecture Tasks → `backend-system-architect`**: Task 1.1 involves fundamental design decisions about module structure and abstraction patterns. This requires architectural thinking rather than just coding skills.

2. **Development Tasks → `python-developer`**: Tasks 1.2, 1.3, 1.4 are primarily coding tasks that benefit from TDD practices and Python expertise. These involve refactoring, CLI implementation, and subprocess management.

3. **Testing Tasks → `test-automation-architect`**: Tasks 2.1, 2.2 require specialized knowledge of testing patterns, mocking strategies, and comprehensive test design. The architect can design test suites that provide good coverage and maintainability.

4. **Code Quality → `code-review-expert`**: Task 2.3 involves reviewing existing code for regression issues, which is exactly what a code review expert specializes in.

5. **Documentation → `documentation-architect`**: All documentation tasks (3.1, 3.2, 3.3) require specialized technical writing skills and understanding of different audiences (users vs developers).

### Cross-Agent Collaboration

- **`backend-system-architect`** should review designs from tasks 1.2-1.4 to ensure architectural consistency
- **`code-review-expert`** should review all code changes before final implementation
- **`test-automation-architect`** should collaborate with developers on testable design patterns
- **`documentation-architect`** should coordinate with all other agents to ensure accurate documentation

## Success Criteria

1. **Functionality**:
   - `uv run project-watch-mcp --initialize` successfully indexes repository
   - MCP tool `initialize_repository` continues to work
   - Session hook triggers initialization automatically via CLI subprocess
   - Hook reduced from ~400 lines to ~50 lines

2. **Code Quality**:
   - Zero code duplication between CLI, server, and hook
   - All tests pass with >80% coverage
   - Clean separation of concerns
   - Session hook only handles: skip checking, CLI invocation, output formatting

3. **User Experience**:
   - Clear progress reporting in verbose mode
   - Helpful error messages on failure
   - Fast initialization (<30s for typical repo)
   - Session start shows: "✅ Repository auto-initialized successfully!" on success

4. **Hook Simplification Metrics**:
   - **Before**: ~400 lines with full initialization logic
   - **After**: ~50 lines delegating to CLI
   - **Code Reduction**: 87.5%
   - **Maintenance Points**: 1 (core/initializer.py) instead of 2 (server + hook)

## Rollback Plan

If issues arise:
1. Revert to previous commit
2. Session hook can temporarily use MCP tool call as fallback
3. Document known issues for users

## Notes

- The 200+ line duplication in session-start.py is critical to eliminate
- FastMCP's server-centric design prevents direct tool extraction
- This pattern (shared core module) is industry standard for similar projects
- Consider future migration to subcommands if more CLI operations needed