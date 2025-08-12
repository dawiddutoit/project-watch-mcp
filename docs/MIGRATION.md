# Project Watch MCP - Migration Guide

## Overview

Project Watch MCP has undergone a significant internal refactoring to improve maintainability, reduce code duplication, and enhance developer experience. **For most users, these changes are completely transparent and require no action.**

The refactoring introduces:
- A streamlined CLI with a new `--initialize` flag for manual repository indexing
- A simplified session hook that delegates to the CLI (reduced from 406 lines to 85 lines)
- A modular core architecture for better code organization
- Improved error handling and progress reporting

## What Changed for Existing Users

### No Breaking Changes ✅

**The good news:** All MCP tools maintain backward compatibility. Your existing integrations will continue to work without any modifications.

### Transparent Improvements

1. **Session Hook Simplification**
   - The session-start hook now delegates to the CLI instead of duplicating logic
   - Initialization is more reliable and consistent
   - Error messages are clearer and more actionable

2. **CLI Enhancement**
   - New `--initialize` flag for manual repository indexing
   - Better progress reporting with `--verbose` flag
   - More consistent error handling

3. **Core Module Structure**
   - Business logic centralized in `src/project_watch_mcp/core/`
   - Improved testability and maintainability
   - No impact on external interfaces

## New Features Available

### 1. Manual Initialization via CLI

You can now initialize repositories directly from the command line without starting the MCP server:

```bash
# Basic initialization
uv run project-watch-mcp --initialize

# Initialize with verbose output
uv run project-watch-mcp --initialize --verbose

# Initialize a specific repository
uv run project-watch-mcp --initialize --repository /path/to/repo --verbose
```

This is useful for:
- Pre-indexing repositories before using MCP clients
- Batch processing multiple repositories
- CI/CD pipeline integration
- Debugging indexing issues

### 2. Improved Progress Reporting

The `--verbose` flag now provides detailed progress during initialization:

```bash
uv run project-watch-mcp --initialize --verbose
# Output:
# [  0%] Starting repository initialization...
# [ 10%] Loading .gitignore patterns...
# [ 20%] Scanning repository files...
# [ 50%] Processing file: src/main.py
# [100%] Initialization complete!
```

### 3. Better Error Messages

Error messages now provide more context and actionable suggestions:

```bash
# Before: "Failed to connect"
# After: "Failed to connect to Neo4j at bolt://localhost:7687. 
#         Make sure Neo4j is running and accessible.
#         You can start Neo4j with Neo4j Desktop or Docker:
#         docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j"
```

## Hook Migration Instructions

### For Users with Custom Hooks

If you've created custom hooks or modified the session-start hook, here's how to migrate:

#### Before (Old Hook Pattern)
```python
# Old hook with duplicated logic (406 lines)
def initialize_repository():
    # Direct Neo4j connection
    driver = AsyncGraphDatabase.driver(...)
    
    # Manual file scanning
    for file in repo.glob("**/*"):
        # Process file...
    
    # Manual embedding generation
    embeddings = generate_embeddings(...)
    
    # Manual index creation
    create_index(...)
```

#### After (New Hook Pattern)
```python
# New hook delegating to CLI (85 lines)
def main():
    # Simply run the CLI command
    result = subprocess.run(
        ['uv', 'run', 'project-watch-mcp', '--initialize'],
        capture_output=True,
        text=True
    )
    
    # Format output for Claude
    if result.returncode == 0:
        return {"status": "success", "output": result.stdout}
    else:
        return {"status": "failed", "error": result.stderr}
```

### Benefits of the New Approach

1. **Single Source of Truth**: All initialization logic lives in one place
2. **Consistent Behavior**: CLI and hook use identical code paths
3. **Easier Testing**: Can test initialization independently
4. **Better Maintenance**: Updates to initialization logic automatically apply everywhere

## Troubleshooting

### Issue: Session Hook Not Working After Update

**Symptom:** Auto-initialization fails at session start

**Solution:**
1. Ensure the CLI is properly installed:
   ```bash
   uv sync
   uv pip install -e .
   ```

2. Test the CLI directly:
   ```bash
   uv run project-watch-mcp --initialize --verbose
   ```

3. Check hook permissions:
   ```bash
   chmod +x .claude/hooks/session-start/session-start.py
   ```

### Issue: Different Behavior Between CLI and MCP Tool

**Symptom:** `--initialize` flag behaves differently than `initialize_repository` MCP tool

**Solution:** This should not happen as both use the same core module. If you encounter this:
1. Update to the latest version
2. Clear Neo4j database and re-initialize
3. Report the issue with specific examples

### Issue: Performance Degradation

**Symptom:** Initialization seems slower after update

**Solution:** The new architecture should be equally fast or faster. If you notice degradation:
1. Use `--verbose` to identify bottlenecks
2. Check Neo4j server performance
3. Verify embedding provider connectivity

## Rollback Procedures

If you need to rollback to the previous version:

### Quick Rollback
```bash
# Revert to previous commit
git checkout <previous-commit-hash>

# Reinstall dependencies
uv sync
```

### Manual Rollback
1. Keep a backup of the old session-start hook
2. Replace the new hook with your backup
3. No database changes are required

**Note:** Rollback is rarely necessary as the changes maintain full backward compatibility.

## FAQ

### Q: Do I need to re-index my repositories?

**A:** No, existing indexes remain valid and will continue to work. Re-indexing is only needed if you want to take advantage of improved indexing algorithms (optional).

### Q: Will my Claude Desktop configuration still work?

**A:** Yes, no changes to Claude Desktop configuration are required. The MCP server interface remains unchanged.

### Q: Can I still use the old initialization method?

**A:** The MCP tool `initialize_repository` continues to work exactly as before. The new CLI flag is an additional option, not a replacement.

### Q: What about custom file patterns and ignore rules?

**A:** All existing patterns and rules continue to work. The refactoring only affects internal code organization.

### Q: Is the Neo4j schema different?

**A:** No, the database schema remains unchanged. Your existing Neo4j data is fully compatible.

### Q: Do I need to update my environment variables?

**A:** No, all environment variables work exactly as before.

### Q: What if I have scripts that depend on the old hook structure?

**A:** The hook's external behavior (output format, exit codes) remains the same. Scripts that call the hook don't need changes. Only the internal implementation has changed.

### Q: Can I use both the CLI flag and MCP tool?

**A:** Yes, both methods are available and can be used interchangeably. They share the same underlying code.

## Summary

This refactoring is primarily an internal improvement that:
- **Maintains 100% backward compatibility** for MCP tools
- **Simplifies the codebase** from 406 to 85 lines for the session hook
- **Adds new convenience features** like the `--initialize` CLI flag
- **Improves error handling and progress reporting**

**For most users, no action is required.** Your existing setup will continue to work exactly as before, with the added benefit of improved reliability and new features available when you need them.

## Getting Help

If you encounter any issues during migration:

1. Run diagnostics:
   ```bash
   uv run project-watch-mcp --initialize --verbose
   ```

2. Check the logs in verbose mode for detailed error information

3. Verify your Neo4j connection:
   ```bash
   # Test Neo4j connectivity
   cypher-shell -u neo4j -p password "RETURN 1"
   ```

4. Report issues with:
   - The exact error message
   - Your configuration (Neo4j version, Python version)
   - Steps to reproduce the issue

Remember: **For 99% of users, this update requires no action and provides transparent improvements.**