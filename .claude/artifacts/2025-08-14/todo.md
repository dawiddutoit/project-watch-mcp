# Todo: Project Watch MCP - Project Completion - 2025-08-14

## Overview
Complete the Project Watch MCP by fixing the remaining test failures and achieving 80% test coverage. The complexity analysis system is implemented but several critical tests are failing, preventing project completion.

## 🚀 Ready to Start

### TASK-001: Fix Core Server Test Failures
- **Agent**: @agent-debugging-expert
- **Priority**: CRITICAL
- **Actions**:
  - [ ] Fix `test_refresh_file_success` - assertion error (expects 'success' but gets 'error')
  - [ ] Fix `test_refresh_file_not_found` - should raise ToolError but doesn't
  - [ ] Fix `test_analyze_complexity_python` - path not in subpath error with temp files
  - [ ] Fix `test_analyze_complexity_non_python` - wrong error message returned
  - [ ] Fix `test_monitoring_status` - AttributeError with 'dict' object has no 'change_type' attribute
- **Files**: `/Users/dawiddutoit/projects/play/project-watch-mcp/src/project_watch_mcp/server.py`, `/Users/dawiddutoit/projects/play/project-watch-mcp/tests/unit/test_server.py`
- **Deliverables**: All 5 server tests passing
- **Testing**: `uv run pytest tests/unit/test_server.py -v`

### TASK-002: Fix Import and Collection Errors  
- **Agent**: @agent-python-developer
- **Priority**: HIGH
- **Actions**:
  - [ ] Fix missing `CacheStats` import in test_optimization_comprehensive.py
  - [ ] Add missing 'performance' marker to pytest configuration
  - [ ] Fix TestEmbeddingsProvider collection warnings (remove __init__ constructor)
- **Files**: 
  - `/Users/dawiddutoit/projects/play/project-watch-mcp/tests/unit/test_optimization_comprehensive.py`
  - `/Users/dawiddutoit/projects/play/project-watch-mcp/pyproject.toml`
  - `/Users/dawiddutoit/projects/play/project-watch-mcp/tests/unit/utils/embeddings/test_embeddings_utils.py`
- **Deliverables**: Clean pytest collection without import errors or warnings
- **Testing**: `uv run pytest --collect-only`

### TASK-003: Fix Complexity Analyzer Test Failures
- **Agent**: @agent-debugging-expert  
- **Priority**: HIGH
- **Actions**:
  - [ ] Fix failing complexity analyzer tests in test_python_analyzer_comprehensive.py
  - [ ] Fix failing complexity analyzer tests in test_java_analyzer_comprehensive.py
  - [ ] Fix failing complexity analyzer tests in test_kotlin_analyzer_comprehensive.py
  - [ ] Fix test_analyze_complexity.py failures
- **Files**: All files in `/Users/dawiddutoit/projects/play/project-watch-mcp/tests/unit/complexity/`
- **Deliverables**: All complexity analyzer tests passing
- **Testing**: `uv run pytest tests/unit/complexity/ -v`

### TASK-004: Achieve 80% Test Coverage
- **Agent**: @agent-qa-testing-expert
- **Priority**: HIGH  
- **Actions**:
  - [ ] Run coverage analysis on fixed tests
  - [ ] Identify uncovered code paths
  - [ ] Add targeted tests for uncovered areas
  - [ ] Validate 80% coverage achieved
- **Files**: Add missing tests as identified by coverage analysis
- **Deliverables**: 80% test coverage achieved
- **Testing**: `uv run pytest --cov=src/project_watch_mcp --cov-report=term-missing --cov-fail-under=80`

## ⏸️ Blocked (Dependencies)

### TASK-005: Final Project Validation
- **Agent**: @agent-critical-auditor
- **Blocked by**: TASK-001, TASK-002, TASK-003, TASK-004
- **Priority**: CRITICAL
- **Actions**:
  - [ ] Verify all tests pass
  - [ ] Verify 80% coverage achieved  
  - [ ] Validate all complexity analysis features work
  - [ ] Run full test suite
  - [ ] Validate MCP server functionality
- **Deliverables**: Complete project validation report
- **Testing**: Full test suite with coverage

## 🔧 Technical Details

### Specific Test Failure Analysis

**Server Test Failures (5 tests):**
1. `test_refresh_file_success` - Returns 'error' instead of expected 'success'
2. `test_refresh_file_not_found` - Should raise ToolError but doesn't
3. `test_analyze_complexity_python` - Path resolution issue with temp files vs repo path
4. `test_analyze_complexity_non_python` - Wrong error message for non-Python files
5. `test_monitoring_status` - AttributeError accessing 'change_type' on dict object

**Import/Collection Issues:**
- Missing `CacheStats` import causing ImportError
- Missing 'performance' marker in pytest config
- TestEmbeddingsProvider class has __init__ constructor causing collection warnings

### Coverage Target
- **Current**: ~60% (estimated based on test failures)
- **Target**: 80%
- **Strategy**: Fix existing tests first, then add targeted tests for uncovered code

## Completed
[Tasks will be moved here after validation by @agent-critical-auditor]