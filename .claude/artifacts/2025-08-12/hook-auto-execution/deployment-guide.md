# Deployment Guide: Enhanced SessionStart Hook with Auto-Initialization

## Overview

This guide provides step-by-step instructions for deploying the enhanced SessionStart hook that automatically initializes the Project Watch MCP repository monitoring without requiring manual Claude commands.

## Pre-Deployment Checklist

### Environment Requirements
- [ ] Python 3.11+ installed
- [ ] Neo4j Desktop running locally (or accessible Neo4j instance)
- [ ] Project dependencies installed (`uv sync` in project root)
- [ ] Environment variables configured:
  - `NEO4J_URI` (default: neo4j://127.0.0.1:7687)
  - `NEO4J_USER` (default: neo4j)
  - `NEO4J_PASSWORD` (required, no default)
  - `NEO4J_DB` or `NEO4J_DATABASE` (default: memory)

### Backup Current Hook
```bash
# Create backup of existing hook
cp .claude/hooks/session_start.py .claude/hooks/session_start.py.backup
```

## Deployment Steps

### Step 1: Deploy Enhanced Hook

```bash
# Copy the enhanced hook to the hooks directory
cp .claude/artifacts/2025-08-12/hook-auto-execution/enhanced_session_start.py \
   .claude/hooks/session_start.py
```

### Step 2: Verify Hook Permissions

```bash
# Ensure the hook is executable
chmod +x .claude/hooks/session_start.py
```

### Step 3: Test Hook Manually

```bash
# Test the hook directly
python .claude/hooks/session_start.py
```

Expected output should show JSON with initialization status:
```json
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "✅ Repository auto-initialized successfully! ...",
    "initializationStatus": "success",
    "statistics": {
      "indexed": 60,
      "total": 60,
      "failed": 0
    }
  }
}
```

### Step 4: Check Logs

```bash
# View the latest log file
ls -la .claude/hooks/logs/
cat .claude/hooks/logs/session_start_*.log
```

## Configuration Options

### Skip Auto-Initialization

If you want to temporarily disable auto-initialization:

```bash
# Create skip marker file
touch .claude/.skip_auto_init

# Remove to re-enable
rm .claude/.skip_auto_init
```

### View Last Initialization Status

```bash
# Check last auto-initialization result
cat .claude/.last_auto_init
```

## Troubleshooting

### Issue: Import Errors

**Symptom:** Hook fails with `ImportError`

**Solution:**
1. Ensure project dependencies are installed:
   ```bash
   uv sync
   ```
2. Verify Python path in hook is correct
3. Check that src/project_watch_mcp directory exists

### Issue: Neo4j Connection Failed

**Symptom:** Hook reports Neo4j connection error

**Solution:**
1. Verify Neo4j is running:
   ```bash
   # Check if Neo4j is accessible
   nc -zv 127.0.0.1 7687
   ```
2. Confirm environment variables:
   ```bash
   echo $NEO4J_PASSWORD
   echo $NEO4J_URI
   ```
3. Test Neo4j connection manually:
   ```bash
   python .claude/artifacts/2025-08-12/hook-auto-execution/test_direct_init.py
   ```

### Issue: Timeout During Initialization

**Symptom:** Hook times out after 30 seconds

**Solution:**
1. Check repository size - large repos may need longer timeout
2. Modify timeout in hook (line ~95):
   ```python
   files = await asyncio.wait_for(
       repository_monitor.scan_repository(),
       timeout=60.0  # Increase from 30.0
   )
   ```

### Issue: No Files Found

**Symptom:** Hook reports "0 files indexed"

**Solution:**
1. Verify .gitignore patterns aren't excluding all files
2. Check file patterns in hook match your file types
3. Ensure hook is running from correct directory

## Rollback Procedure

If issues occur, rollback to the original hook:

```bash
# Restore backup
cp .claude/hooks/session_start.py.backup .claude/hooks/session_start.py

# Or use the original simple hook
cat > .claude/hooks/session_start.py << 'EOF'
#!/usr/bin/env python3
import json
import sys

output = {
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": "Please initialize: mcp__project-watch-local__initialize_repository"
    }
}
print(json.dumps(output))
sys.exit(0)
EOF

chmod +x .claude/hooks/session_start.py
```

## Monitoring and Validation

### Check Hook Execution in Claude

1. Start a new Claude Code session
2. Look for initialization message in Claude's first response
3. Verify you can immediately use tools like:
   - `mcp__project-watch-local__search_code`
   - `mcp__project-watch-local__get_file_info`

### Monitor Logs

```bash
# Watch logs in real-time during session start
tail -f .claude/hooks/logs/session_start_*.log
```

### Verify Neo4j Data

```bash
# Check if data was indexed in Neo4j
echo "MATCH (c:CodeChunk) WHERE c.project_name = 'project-watch-mcp' RETURN COUNT(c);" | \
  cypher-shell -u neo4j -p $NEO4J_PASSWORD
```

## Performance Expectations

- **Initial initialization:** 5-30 seconds depending on repository size
- **Subsequent runs:** Near instant (idempotent check)
- **Memory usage:** Minimal (< 100MB)
- **Neo4j storage:** ~10MB per 1000 files indexed

## Security Considerations

1. **Credentials:** Neo4j password is read from environment, never hardcoded
2. **File Access:** Hook only accesses files within repository boundaries
3. **Logging:** Sensitive information is not logged
4. **Timeouts:** Prevent resource exhaustion with 30-second timeout
5. **Error Handling:** Failures don't expose system information

## Support and Maintenance

### Log Rotation

Logs accumulate in `.claude/hooks/logs/`. Implement rotation:

```bash
# Clean logs older than 7 days
find .claude/hooks/logs -name "session_start_*.log" -mtime +7 -delete
```

### Update Hook

To update the hook with improvements:

1. Always backup current version
2. Test new version with test script first
3. Deploy during non-critical time
4. Monitor first few runs closely

## Conclusion

The enhanced SessionStart hook provides automatic repository initialization, significantly improving the developer experience by eliminating manual initialization steps. With proper deployment and monitoring, it provides a seamless, reliable auto-initialization experience while maintaining robust fallback mechanisms.