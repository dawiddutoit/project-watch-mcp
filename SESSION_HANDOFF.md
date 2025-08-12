# Session Handoff: CLI Initialization Refactoring Implementation

## 🎯 Mission Critical Context

**Project**: Project Watch MCP  
**Location**: `/Users/dawiddutoit/projects/play/project-watch-mcp`  
**Task**: Implement CLI `--initialize` flag by extracting 200+ lines of duplicated initialization logic  
**Priority**: CRITICAL - Eliminates maintenance nightmare of duplicate code  
**Time Estimate**: 4-6 hours total implementation

## 🚨 Critical Warnings

### Environment Requirements
- **Neo4j**: Uses LOCAL Neo4j Desktop (NOT Docker!) at `neo4j://127.0.0.1:7687`
- **Python**: 3.11+ required (uses modern async features)
- **Package Manager**: Use `uv` commands, NOT pip
- **Virtual Environment**: Already exists in `.venv/`, managed by uv

### Do NOT Make These Mistakes
1. ❌ Don't assume Docker for Neo4j - it's Neo4j Desktop on macOS
2. ❌ Don't use `NEO4J_DATABASE` env var - use `NEO4J_DB` 
3. ❌ Don't create new files unless absolutely necessary
4. ❌ Don't use pip - always use `uv run` or `uv add`
5. ❌ Don't forget to handle async context properly (avoid `asyncio.run()` inside async)
6. ❌ Don't hardcode timeouts - make them configurable

## 📋 Implementation Plan Overview

### Current Problem
The session-start hook (`/.claude/hooks/session-start/session-start.py`) contains 200+ lines (lines 39-232) that duplicate the initialization logic from `server.py`. This creates a maintenance nightmare where changes must be made in multiple places.

### Solution Architecture
```
src/project_watch_mcp/
├── core/                        # NEW directory
│   ├── __init__.py             # NEW - Package init
│   └── initializer.py          # NEW - Shared initialization logic
├── cli.py                      # MODIFY - Add --initialize flag
├── server.py                   # MODIFY - Use core.initializer
└── [existing files]

.claude/hooks/session-start/
└── session-start.py            # SIMPLIFY - Just call CLI
```

## 🎬 Quick Start for Next Session

```python
# Start with this prompt to the next Claude:
"""
I need to implement the CLI initialization refactoring for Project Watch MCP.
The complete plan is in /Users/dawiddutoit/projects/play/project-watch-mcp/feature-todo.md

Please:
1. Review the feature-todo.md file 
2. Start with Phase 1.1 using the backend-system-architect agent
3. Follow the agent assignments specified in the todo
4. Remember: Neo4j runs locally via Neo4j Desktop, NOT Docker
5. Use `uv` for all Python operations

The goal is to extract 200+ lines of duplicate initialization code from the session-start hook
into a shared core module and add a --initialize CLI flag.
"""
```

## 🤖 Agent Implementation Sequence

### Phase 1: Core Module Extraction (2-3 hours)

#### Task 1.1: Create Core Initializer Module
**Agent Command**: Use `backend-system-architect` agent
```
Create the core initializer module at src/project_watch_mcp/core/initializer.py
Extract initialization logic from server.py's initialize_repository tool.
Design RepositoryInitializer class with async initialize() method.
Include progress callback support and InitializationResult dataclass.
```

#### Task 1.2: Refactor Server
**Agent Command**: Use `python-developer` agent
```
Refactor src/project_watch_mcp/server.py to use the new core module.
The initialize_repository tool should delegate to RepositoryInitializer.
Maintain full backward compatibility with existing MCP interface.
```

#### Task 1.3: Add CLI Support  
**Agent Command**: Use `python-developer` agent
```
Modify src/project_watch_mcp/cli.py to add --initialize flag.
Make it mutually exclusive with --transport.
Implement initialize_only() async function with proper Neo4j cleanup.
Exit codes: 0 for success, 1 for failure.
```

#### Task 1.4: Simplify Hook
**Agent Command**: Use `python-developer` agent
```
Simplify .claude/hooks/session-start/session-start.py to use subprocess.
Replace lines 39-232 with: subprocess.run(['uv', 'run', 'project-watch-mcp', '--initialize'])
Keep error handling and .last_auto_init marker.
```

### Phase 2: Testing (1-2 hours)

#### Task 2.1-2.2: Create Tests
**Agent Command**: Use `test-automation-architect` agent
```
Create comprehensive tests in tests/test_initializer.py and tests/test_cli_initialize.py
Include unit tests with mocked Neo4j, integration tests, and edge cases.
Test coverage target: 85% minimum for new code.
```

#### Task 2.3: Update Existing Tests
**Agent Command**: Use `code-review-expert` agent
```
Review and update existing tests for compatibility.
Ensure no regression in server.py tests.
Verify MCP tool backward compatibility.
```

### Phase 3: Documentation (1 hour)

**Agent Command**: Use `documentation-architect` agent
```
Update README.md with --initialize flag documentation.
Update CLAUDE.md with new module structure.
Create docs/MIGRATION.md for users.
```

## 📁 Key Files Reference

### Files to Create
- `/Users/dawiddutoit/projects/play/project-watch-mcp/src/project_watch_mcp/core/__init__.py`
- `/Users/dawiddutoit/projects/play/project-watch-mcp/src/project_watch_mcp/core/initializer.py`
- `/Users/dawiddutoit/projects/play/project-watch-mcp/tests/test_initializer.py`
- `/Users/dawiddutoit/projects/play/project-watch-mcp/tests/test_cli_initialize.py`

### Files to Modify
- `/Users/dawiddutoit/projects/play/project-watch-mcp/src/project_watch_mcp/cli.py` - Add --initialize flag
- `/Users/dawiddutoit/projects/play/project-watch-mcp/src/project_watch_mcp/server.py` - Use core module
- `/Users/dawiddutoit/projects/play/project-watch-mcp/.claude/hooks/session-start/session-start.py` - Simplify to subprocess

### Reference Documentation
- `/Users/dawiddutoit/projects/play/project-watch-mcp/feature-todo.md` - Complete implementation plan
- `/Users/dawiddutoit/projects/play/project-watch-mcp/CLAUDE.md` - Project conventions
- `/Users/dawiddutoit/projects/play/project-watch-mcp/README.md` - User documentation

## 🔍 Code Patterns to Extract

### From session-start.py (lines 39-232)
```python
# Key components to extract:
1. Neo4j connection setup (lines 58-103)
2. Project configuration (lines 105-108)  
3. Repository monitoring setup (lines 110-126)
4. Embeddings configuration (lines 128-137)
5. Neo4j RAG initialization (lines 139-150)
6. File scanning logic (lines 152-160)
7. File indexing loop (lines 162-190)
8. Cleanup and results (lines 192-210)
```

### From server.py's initialize_repository
```python
# Similar logic that needs to be unified:
- Neo4j connection handling
- Repository scanning with gitignore support
- File indexing with progress reporting
- Error handling and recovery
```

## ✅ Success Criteria

1. **Functionality**
   - `uv run project-watch-mcp --initialize` successfully indexes repository
   - Session hook auto-initializes without manual intervention
   - MCP tool maintains 100% backward compatibility

2. **Code Quality**
   - Zero code duplication between CLI, server, and hook
   - Test coverage >85% for new code
   - All linting passes (ruff, black, pyright)

3. **Performance**
   - Initialization <30s for 500-file repository
   - Memory usage <500MB during indexing
   - Progress reporting at least every 10 files

## 🐛 Known Issues & Gotchas

### Async Complexity
- The hook, CLI, and server all have different async contexts
- Use `asyncio.create_task()` for concurrent operations
- Ensure proper cleanup with try/finally blocks

### Neo4j Connection
- Must verify connectivity before operations
- Connection pool exhaustion is possible - implement limits
- Always close driver in finally blocks

### File System
- Repository monitor must respect .gitignore patterns
- Handle symbolic links and circular references
- Support Unicode filenames and special characters

### Testing Challenges
- Mock Neo4j properly for unit tests
- Integration tests need real Neo4j instance
- Session hook testing requires subprocess mocking

## 🚀 First Actions for Next Session

1. **Read and understand the plan**:
   ```bash
   cat /Users/dawiddutoit/projects/play/project-watch-mcp/feature-todo.md
   ```

2. **Check current implementation**:
   ```bash
   # See the duplicate code
   head -n 250 /Users/dawiddutoit/projects/play/project-watch-mcp/.claude/hooks/session-start/session-start.py
   
   # Check server implementation
   grep -A 50 "initialize_repository" /Users/dawiddutoit/projects/play/project-watch-mcp/src/project_watch_mcp/server.py
   ```

3. **Start Phase 1.1 with backend-system-architect**:
   ```
   Use the backend-system-architect agent to create the core initializer module
   ```

## 📊 Progress Tracking

Use the todo.md to track implementation progress:
```markdown
- [ ] Phase 1.1: Core initializer module created
- [ ] Phase 1.2: Server refactored to use core
- [ ] Phase 1.3: CLI --initialize flag added
- [ ] Phase 1.4: Session hook simplified
- [ ] Phase 2.1: Unit tests created
- [ ] Phase 2.2: Integration tests created
- [ ] Phase 2.3: Existing tests updated
- [ ] Phase 3.1: README updated
- [ ] Phase 3.2: CLAUDE.md updated
- [ ] Phase 3.3: Migration guide created
```

## 💡 Architecture Decisions

### Why Shared Core Module?
- Single source of truth for initialization logic
- Testable in isolation from transport layers
- Enables future initialization modes (API, scheduled, etc.)

### Why CLI Flag Instead of Subcommand?
- Simpler implementation for single operation
- Maintains backward compatibility
- Can migrate to subcommands later if needed

### Why Keep MCP Tool?
- Backward compatibility for existing users
- Allows initialization from within Claude sessions
- Provides programmatic access to initialization

## 📝 Final Notes

This refactoring is critical for maintainability. The current duplicate code is a significant technical debt that will only get worse as the project evolves. By extracting the shared logic into a core module, we create a clean, testable, and maintainable architecture.

Remember:
- Start with the backend-system-architect for architectural design
- Use python-developer for implementation with TDD
- Leverage test-automation-architect for comprehensive testing
- Let documentation-architect handle all docs

The complete, detailed plan with all test scenarios and edge cases is in feature-todo.md. Follow it closely for best results.

---
*Session handoff prepared by Context Architect on 2025-08-12*
*Next session should begin with Phase 1.1 implementation*