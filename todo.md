# Project Watch MCP - Fixes and Improvements Todo List

## Agent Assignment Guide

### Best Suited Agents by Task Category

| Priority | Task Category | Primary Agent | Supporting Agents | Reason |
|----------|--------------|---------------|-------------------|---------|
| **1** | Critical Fixes | `python-developer` | `debugging-expert` | Return type mismatches, MCP protocol violations |
| **2** | Critical Enhancements | `python-developer` | `backend-system-architect` | New tool creation, file pattern expansion |
| **3** | Testing Improvements | `qa-testing-expert` | `test-automation-architect` | Test coverage, integration tests, E2E scenarios |
| **4** | Usability Improvements | `python-developer` | `ux-design-expert` | New tools for better developer experience |
| **5** | Error Handling | `python-developer` | `debugging-expert` | User-friendly error messages, validation |
| **6** | Documentation | `documentation-architect` | `api-documentation-specialist` | README updates, docstrings, examples |
| **7** | Performance | `backend-system-architect` | `python-developer` | Caching, optimization, scalability |
| **8** | Advanced Features | `backend-system-architect` | `python-developer` | Code relationships, advanced search |

## Critical Fixes (Priority 1 - Emergency) ✅ COMPLETED
**Timeline: 2-4 hours** ✅ Completed in < 1 hour  
**Impact: Core functionality restoration** ✅ Achieved  
**🤖 Best Agent: `python-developer` with `debugging-expert` for troubleshooting**

### ✅ Fix Return Type Mismatches - COMPLETED 2025-08-11
- [x] **Fix `initialize_repository` return type** ✅
  - Changed function signature from `-> str` to `-> ToolResult`
  - Location: `src/project_watch_mcp/server.py` line 47
  - ~~Error: "Output validation error: 'result' is a required property"~~ FIXED
  - **Agent**: `python-developer` (MCP protocol expertise)
  
- [x] **Fix `refresh_file` return type** ✅
  - Changed function signature from `-> str` to `-> ToolResult`
  - Location: `src/project_watch_mcp/server.py` line 254
  - ~~Error: "Output validation error: 'result' is a required property"~~ FIXED
  - **Agent**: `python-developer` (MCP protocol expertise)

### ✅ Fix Search Output Structure - COMPLETED 2025-08-11
- [x] **Fix `search_code` output format** ✅
  - ~~Wrap results list in dictionary: `{"results": search_results}`~~ Already wrapped correctly
  - Changed return type annotation from `-> list[dict]` to `-> ToolResult`
  - Location: `src/project_watch_mcp/server.py` line 113
  - ~~Current error: "structured_content must be a dict or None. Got list"~~ FIXED
  - Note: Search IS working internally with good similarity scores ✅ NOW ACCESSIBLE
  - **Agent**: `python-developer` (MCP protocol compliance)

## Critical Enhancements (Priority 2 - Essential)
**Timeline: 1 day**  
**Impact: Basic usability**
**🤖 Best Agent: `python-developer` with `backend-system-architect` for design**

### 📄 File Content Access
- [ ] **Create `get_file_content` tool**
  - Parameters: `file_path`, `start_line` (optional), `end_line` (optional)
  - Return: Actual file content, not just metadata
  - Essential for code exploration
  - **Agent**: `python-developer` (Tool implementation)

### 📁 Expand File Type Support
- [ ] **Add configuration file patterns**
  - Add to supported patterns: `*.toml`, `*.ini`, `*.env`, `*.cfg`, `*.conf`
  - Update `file_patterns` in repository monitor
  - Currently missing critical project config files
  - **Agent**: `python-developer` (Pattern configuration)

## Testing Improvements (Priority 3 - Quality) 🔥 URGENT
**Timeline: 2 weeks (accelerated from 1-2 days)**  
**Impact: Critical - 20 test failures blocking production readiness**
**🤖 Lead Agents: `qa-testing-expert`, `test-automation-architect`, `context-manager`**
**Current Status: 87.8% pass rate (144/164 tests) → Target: 95%+**

### 🚨 IMMEDIATE FIXES NEEDED (Days 1-3)
**20 Failing Tests - Root Causes Identified:**

#### Async/Await Issues (8 failures - 40%)
- [ ] **Fix async handling in test_mcp_integration.py**
  - Add `await` to all `server.get_tools()` calls
  - Convert `AsyncMock` to proper async context managers
  - Fix mock_record async implementations
  - **Agent**: `python-developer` (Day 1)
  - **Files**: test_mcp_integration.py lines with "coroutine not iterable"

#### Data Structure Mismatches (7 failures - 35%)
- [ ] **Standardize Neo4j response keys**
  - Fix `chunk_content` vs expected key names
  - Add missing `line_number` fields
  - Ensure `total_files` in stats response
  - **Agent**: `python-developer` (Day 2)
  - **Files**: test_neo4j_rag_extended.py, test_corruption_prevention.py

#### Missing Attributes (5 failures - 25%)
- [ ] **Add missing class attributes**
  - Add `close()` method to Neo4jRAG class
  - Remove FileInfo.content references or add attribute
  - **Agent**: `python-developer` (Day 2)
  - **Files**: src/project_watch_mcp/neo4j_rag.py

### 📊 Test Coverage Expansion (Days 4-8)
**Current: ~60% → Week 1 Target: 70% → Week 2 Target: 85%**

#### Integration Tests
- [ ] **MCP Protocol Compliance Tests**
  - Tool registration validation
  - Return type compliance (ToolResult)
  - Parameter validation
  - **Agent**: `qa-testing-expert` (Days 5-6)

- [ ] **Project Isolation Tests**
  - Multi-project concurrent operations
  - Cross-contamination prevention
  - Namespace isolation verification
  - **Agent**: `backend-system-architect` (Day 4)

- [ ] **End-to-End Scenarios**
  - Complete workflow: init → index → search → update
  - Error recovery scenarios
  - Performance under load
  - **Agent**: `qa-testing-expert` (Days 7-8)

### 🏗️ Test Infrastructure Improvements (Days 3-6)

#### Mock Infrastructure
- [ ] **Create MockNeo4jDriver class**
  ```python
  class MockNeo4jDriver:
      def __init__(self):
          self._projects = {}
      async def execute_query(self, query, params=None, **kwargs):
          # Project-isolated mock responses
  ```
  - **Agent**: `python-developer` (Day 3)

- [ ] **Implement TestDataFactory**
  - Consistent test data generation
  - Project-scoped fixtures
  - Realistic mock responses
  - **Agent**: `test-automation-architect` (Day 3)

#### Test Fixtures
- [ ] **Async-safe fixtures with cleanup**
  ```python
  @pytest_asyncio.fixture
  async def async_server():
      server = create_mcp_server(...)
      tools = await server.get_tools()
      yield {"server": server, "tools": tools}
      # Cleanup
  ```
  - **Agent**: `test-automation-architect` (Day 3)

### 🎯 Quality Gates & Metrics (Days 9-10)

#### Coverage Requirements
- [ ] **Core functionality**: 100% coverage required
  - All MCP tool methods
  - Neo4j operations
  - File monitoring
- [ ] **Overall target**: 85% by end of Week 2
- [ ] **Integration tests**: Minimum 20 scenarios

#### Performance Benchmarks
- [ ] **Operation targets**:
  - Repository initialization: < 5 seconds for 1000 files
  - Semantic search: < 500ms for 10k chunks
  - Pattern search: < 200ms
  - File refresh: < 100ms
  - **Agent**: `backend-system-architect` (Day 9)

### 📈 Advanced Testing (Days 11-14)

#### Property-Based Testing
- [ ] **Implement hypothesis tests**
  - Fuzz input parameters
  - Test invariants
  - Edge case discovery
  - **Agent**: `test-automation-architect` (Days 11-12)

#### Performance Testing
- [ ] **Load and stress tests**
  - 10,000+ file repositories
  - Concurrent operations
  - Memory profiling
  - **Agent**: `backend-system-architect` (Day 13)

#### Security Testing
- [ ] **Vulnerability assessment**
  - Input validation
  - Path traversal prevention
  - Neo4j injection prevention
  - **Agent**: `qa-testing-expert` (Day 14)

### 🔄 CI/CD Integration (Day 10)

- [ ] **GitHub Actions workflow**
  ```yaml
  - Unit tests (no Neo4j)
  - Integration tests (Neo4j container)
  - Coverage reporting
  - Performance benchmarks
  ```
  - **Agent**: `test-automation-architect`

- [ ] **Test containers setup**
  - Neo4j 5.11+ container
  - Automated test data loading
  - Parallel test execution

### 📝 Documentation Updates (Throughout)

- [ ] **Test documentation**
  - Test plan document
  - Test case specifications
  - Coverage reports
  - **Agent**: `documentation-architect`

### 🚦 Success Criteria

#### Week 1 Milestones
- ✅ All 20 test failures fixed (Day 3)
- ✅ 70% code coverage achieved (Day 5)
- ✅ Integration test suite complete (Day 7)
- ✅ CI/CD pipeline operational (Day 5)

#### Week 2 Milestones
- ✅ 85% code coverage achieved (Day 10)
- ✅ Performance benchmarks met (Day 12)
- ✅ Security assessment complete (Day 14)
- ✅ Production ready (Day 14)

### 🔥 Critical Dependencies
- Neo4j 5.11+ (vector index support)
- Python 3.11+ (async features)
- FastMCP framework compatibility
- Proper async/await patterns

### 📊 Current Test Status Dashboard
```
Total Tests: 164
Passing: 144 (87.8%)
Failing: 20 (12.2%)
Skipped: 8

Failure Categories:
- Async/Await: 8 tests (40%)
- Data Structure: 7 tests (35%)
- Missing Attributes: 5 tests (25%)

Priority Fixes:
P0 (Critical): 20 tests
P1 (High): Coverage expansion
P2 (Medium): Performance tests
```

### 🎬 Next Actions
1. **IMMEDIATE**: Fix async/await in test_mcp_integration.py
2. **Day 1**: Add missing Neo4jRAG.close() method
3. **Day 2**: Standardize mock response structures
4. **Day 3**: Implement MockNeo4jDriver
5. **Day 5**: Begin integration test suite

## Usability Improvements (Priority 4 - Enhancement)
**Timeline: 2-3 days**  
**Impact: Developer experience**
**🤖 Best Agent: `python-developer` with `ux-design-expert` for API design**

### 🗂️ Directory Navigation
- [ ] **Create `list_directory` tool**
  - Parameters: `path`, `include_hidden` (optional)
  - Return: List of files and subdirectories
  - Essential for exploring repository structure
  - **Agent**: `python-developer` (Tool implementation)

### 🔄 Batch Operations
- [ ] **Create `get_multiple_files_info` tool**
  - Parameters: `file_paths` (list)
  - Return: Batch file metadata
  - Optimize for performance with single Neo4j query
  - **Agent**: `python-developer` with `backend-system-architect` (Performance optimization)

### 📊 Enhanced Statistics
- [ ] **Improve `get_repository_stats` output**
  - Add: Lines of code per language
  - Add: Number of functions/classes (for Python files)
  - Add: Test file count vs source file count
  - **Agent**: `python-developer` (Statistics calculation)

## Error Handling (Priority 5 - Polish)
**Timeline: 1 day**  
**Impact: User experience**
**🤖 Best Agent: `python-developer` with `debugging-expert` for error scenarios**

### 🚨 User-Friendly Error Messages
- [ ] **Replace internal errors with clear messages**
  - File not found → "File 'X' is not indexed. Supported types: ..."
  - Neo4j connection error → "Database connection failed. Please check Neo4j is running."
  - Invalid search type → "Search type must be 'semantic' or 'pattern'"
  - **Agent**: `python-developer` (Error message implementation)

- [ ] **Add input validation**
  - Validate file paths exist before processing
  - Check search query is not empty
  - Validate limit parameters are positive integers
  - **Agent**: `python-developer` (Validation logic)

### 🚨 Graceful Degradation
- [ ] **Handle Neo4j connection failures**
  - Provide fallback behavior when database is unavailable
  - Cache recent results for offline access
  - Clear error messages about what functionality is limited
  - **Agent**: `debugging-expert` (Failure handling strategies)

## Documentation Updates (Priority 6 - Documentation)
**Timeline: 0.5 days**  
**Impact: Adoption and maintenance**
**🤖 Best Agent: `documentation-architect` with `api-documentation-specialist`**

### 📚 Update README
- [ ] **Document fixed tool signatures**
  - Update all tool examples with correct return types
  - Add troubleshooting section for common errors
  - **Agent**: `documentation-architect` (Technical documentation)

- [ ] **Add usage examples**
  - Complete workflow examples
  - Common search patterns
  - File exploration scenarios
  - **Agent**: `api-documentation-specialist` (API examples)

### 📚 Code Documentation
- [ ] **Add docstrings for all tools**
  - Include parameter types and descriptions
  - Document return value structure
  - Add usage examples in docstrings
  - **Agent**: `documentation-architect` (Docstring standards)

## Performance Optimization (Priority 7 - Optimization)
**Timeline: 2-3 days**  
**Impact: Scalability**
**🤖 Best Agent: `backend-system-architect` with `python-developer`**

### ⚡ Search Performance
- [ ] **Optimize semantic search**
  - Add caching layer for embeddings
  - Batch embedding generation
  - Consider using approximate nearest neighbor algorithms
  - **Agent**: `backend-system-architect` (Caching architecture)

### ⚡ Indexing Performance
- [ ] **Optimize file monitoring**
  - Debounce rapid file changes
  - Batch index updates
  - Add progress indicators for large repositories
  - **Agent**: `backend-system-architect` (Performance optimization)

## Advanced Features (Priority 8 - Future)
**Timeline: 1 week+**  
**Impact: Advanced capabilities**
**🤖 Best Agent: `backend-system-architect` for design, `python-developer` for implementation**

### 🔗 Code Relationships
- [ ] **Add import/dependency tracking**
  - Create relationships between files based on imports
  - Track function/class definitions and usages
  - Enable "find all references" functionality
  - **Agent**: `backend-system-architect` (Graph relationship design)

### 🔍 Advanced Search
- [ ] **Add regex search support**
  - Implement proper regex pattern matching
  - Add search history and saved searches
  - Support complex queries (AND/OR/NOT)
  - **Agent**: `python-developer` (Search implementation)

### 📈 Analytics
- [ ] **Add code quality metrics**
  - Complexity analysis
  - Code duplication detection
  - Technical debt indicators
  - **Agent**: `backend-system-architect` (Metrics design)

## Bug Fixes Summary

### ✅ Previously Broken Tools (NOW FIXED)
1. ✅ `initialize_repository` - Return type mismatch FIXED
2. ✅ `search_code` - Output structure violation FIXED
3. ✅ `refresh_file` - Return type mismatch FIXED

### Partially Working (Should Fix)
4. ⚠️ `get_file_info` - Limited file type support

### Working (No Action Needed)
5. ✅ `get_repository_stats` - Functioning correctly
6. ✅ `monitoring_status` - Functioning correctly

## Success Metrics

### After Priority 1-2 Fixes
- [x] ~~All 6 tools return valid responses~~ 5 of 6 tools working (83% success rate)
- [x] Search functionality works for both semantic and pattern modes ✅
- [ ] File content can be retrieved (Priority 2 - pending)
- [ ] Configuration files are indexed (Priority 2 - pending)

### After Priority 3-5 Fixes
- [ ] 80%+ test coverage
- [ ] All error messages are user-friendly
- [ ] Directory navigation is possible
- [ ] Batch operations improve performance

### After Priority 6-8 Fixes
- [ ] Complete documentation with examples
- [ ] Performance suitable for 10,000+ file repositories
- [ ] Advanced search capabilities available
- [ ] Code relationship mapping functional

## Notes

- **Current tool success rate**: ✅ 83% (5 of 6 tools working) - IMPROVED FROM 33%
- **Estimated time to basic functionality**: ✅ ACHIEVED (< 1 hour for critical fixes)
- **Estimated time to production ready**: 3-5 days remaining (Priority 2-5 items)
- **Key insight**: Core logic confirmed working - annotation fixes resolved main issues
- **Neo4j requirement**: Must have Neo4j 5.11+ running locally
- **Fixes Applied**: 2025-08-11 - All Priority 1 critical fixes completed