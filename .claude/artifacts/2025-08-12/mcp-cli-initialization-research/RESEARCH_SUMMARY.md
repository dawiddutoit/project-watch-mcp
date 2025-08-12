# Research Summary: MCP CLI Initialization Architecture

## Research Topics Investigated

### 1. FastMCP Architecture & Limitations
**Finding:** FastMCP is inherently server-centric. Tools are tightly coupled to server context through decorators and expect active transport layers.

**Evidence:**
- No examples found of FastMCP tools executing outside server context
- FastMCP 2.0 documentation confirms tools require server runtime
- Client-based invocation requires running server

**Implication:** Direct tool extraction is not feasible; refactoring required.

### 2. Industry CLI Patterns
**Finding:** Production projects with similar requirements use clear separation between indexing and serving.

**Key Examples:**
- **RepoGraph**: Separate CLI module (`python3 -m repograph.cli`)
- **Semantic Code Search**: Subcommands (`sem embed` vs `sem search`)
- **Code Indexer Loop**: Library-first with optional CLI wrapper

**Pattern:** Successful projects extract core logic to shared modules.

### 3. Python CLI Best Practices
**Finding:** Subparsers preferred over mutually exclusive groups for complex CLIs.

**Comparison:**
- **Mutually Exclusive Groups**: Simple but limited scalability
- **Subparsers**: Industry standard, allows future expansion
- **Hybrid Approach**: Start with flag, migrate to subcommands later

**Recommendation:** Use `--initialize` flag initially, plan for subcommands.

### 4. Dependency Injection Patterns
**Finding:** Async Neo4j operations require careful dependency management.

**Best Practices:**
- Single driver instance (Singleton pattern)
- Short-lived sessions per operation
- Dependency injection for testability
- Proper async context management

### 5. Real-World Implementation Analysis

#### Projects Analyzed:
1. **RepoGraph** (WilliamsCJ/repograph)
   - Three-component architecture (DB, backend, CLI)
   - Clean separation of concerns
   - Module-based CLI invocation

2. **Semantic Code Search** (sturdy-dev/semantic-code-search)
   - Explicit index management (`sem embed`)
   - No auto-update on file changes
   - Clear command separation

3. **Code Indexer Loop** (definitive-io/code-indexer-loop)
   - Core library pattern
   - Watchdog for file monitoring
   - CLI as thin wrapper

## Critical Insights

### What Works:
1. **Separation of Concerns**: Core logic independent of transport
2. **Explicit Initialization**: Clear, user-triggered indexing
3. **Progress Reporting**: Essential for large repositories
4. **Dependency Injection**: Enables testing and flexibility

### What Doesn't Work:
1. **Code Duplication**: Current hook approach is unmaintainable
2. **Tool Extraction**: FastMCP tools can't run standalone
3. **Client-Based Init**: Starting server just for init is wasteful
4. **Implicit Updates**: Auto-indexing on file changes problematic

## Risk Analysis

### Technical Risks:
- **High**: Code duplication causing maintenance issues
- **Medium**: Performance impact of refactoring
- **Low**: Breaking API changes (internal refactoring only)

### Implementation Risks:
- **High**: Continuing without refactoring
- **Medium**: Over-engineering the solution
- **Low**: Migration complexity (clear path forward)

## Recommended Architecture

### Immediate (Phase 1):
```
src/project_watch_mcp/
├── core/
│   └── initializer.py      # Shared initialization logic
├── cli.py                  # CLI with --initialize flag
├── server.py               # MCP server using core.initializer
└── hooks/
    └── session-start.py    # Calls CLI --initialize
```

### Future (Phase 2):
```
Commands:
- project-watch-mcp init    # Initialize repository
- project-watch-mcp serve   # Start MCP server
- project-watch-mcp status  # Check index status
- project-watch-mcp reindex # Force reindexing
```

## Implementation Priority

### Must Have (Critical):
1. Extract initialization to `core.initializer`
2. Add `--initialize` flag to CLI
3. Update server to use shared logic
4. Simplify hook to call CLI

### Should Have (Important):
1. Progress reporting for large repos
2. Comprehensive error handling
3. Unit and integration tests
4. Performance optimization

### Nice to Have (Future):
1. Subcommand structure
2. Parallel indexing
3. Incremental updates
4. Web UI for initialization

## Time Estimates

- **Core Refactoring**: 2-3 hours
- **Testing**: 1-2 hours
- **Documentation**: 1 hour
- **Total Phase 1**: 4-6 hours

## Success Criteria

1. **Zero code duplication** between components
2. **Single source of truth** for initialization logic
3. **Clean separation** of CLI and server modes
4. **Maintainable** hook implementation
5. **Testable** components with >80% coverage

## Final Verdict

The current architecture with duplicated initialization logic is **fundamentally flawed** and will become increasingly problematic. The proposed refactoring to extract core logic is not premature optimization but **essential technical debt resolution**.

The research strongly supports the "Extract Initialization Logic" approach as the most sustainable solution, aligning with industry best practices and successful similar projects.

## Next Steps

1. **Immediate**: Implement Phase 1 refactoring
2. **Short-term**: Add comprehensive testing
3. **Medium-term**: Consider subcommand migration
4. **Long-term**: Evaluate need for web UI or API

## References

### Primary Sources:
- FastMCP Documentation (gofastmcp.com)
- Neo4j Python Driver Async API (neo4j.com/docs)
- Python argparse documentation

### Code Examples Analyzed:
- github.com/WilliamsCJ/repograph
- github.com/sturdy-dev/semantic-code-search
- github.com/definitive-io/code-indexer-loop

### Patterns Referenced:
- Dependency Injection (python-dependency-injector)
- Command Pattern (Click, Typer frameworks)
- Repository Pattern (Domain-Driven Design)