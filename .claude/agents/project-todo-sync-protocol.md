---
name: project-todo-sync-protocol
description: Mandatory protocol for keeping TodoWrite tool and actual todo.md file synchronized throughout project-watch-mcp implementation work.
model: sonnet
color: red
---

# Todo.md Synchronization Protocol

## Problem Statement
The TodoWrite tool and actual todo.md files operate independently, causing:
- TodoWrite shows "completed" tasks that aren't validated
- Actual todo.md file never gets updated during implementation
- Disconnect between session progress tracking and deliverable documentation

## Mandatory Protocol for All Agents

### 1. After Completing ANY Task
Every agent must:
```markdown
1. Update TodoWrite tool with status
2. IMMEDIATELY call @agent-project-todo-orchestrator to sync the actual todo.md file
3. Provide validation evidence to todo-orchestrator
4. Wait for todo.md file update confirmation before proceeding
```

### 2. Before Marking Task "Completed"
```markdown
1. Agent must provide specific validation evidence:
   - List of files created/modified
   - Test results and coverage metrics
   - Functional verification proof
   
2. @agent-project-todo-orchestrator must:
   - Update the actual todo.md file with evidence
   - Move task to appropriate section (Awaiting Validation)
   - Call @agent-critical-auditor for validation
   
3. Only after @agent-critical-auditor approval:
   - Task moves to "Completed Tasks" section in todo.md
   - TodoWrite tool updated to "completed"
```

### 3. Real-Time Todo.md Maintenance
```markdown
Every implementation agent must:
- Update todo.md after each major milestone
- Include specific progress metrics in todo.md
- Document any blockers or issues discovered
- Maintain evidence trail for validation
```

## Implementation for Project-Watch-MCP Agents

### Modified Agent Workflow
All specialist agents (python-developer, qa-testing-expert, etc.) must:

1. **Start Task**: 
   - Check todo.md current status
   - Update TodoWrite to "in_progress" 
   - Update todo.md with "In Progress" status

2. **Complete Task**:
   - Update TodoWrite to "completed"  
   - Call @agent-project-todo-orchestrator with evidence
   - Wait for todo.md file update
   - Request validation from @agent-critical-auditor

3. **Validation Complete**:
   - @agent-critical-auditor updates todo.md with validation results
   - Task officially marked complete in both TodoWrite and todo.md

### Required Evidence Format
When calling @agent-project-todo-orchestrator, agents must provide:

```markdown
Task: [Task Name]
Status: Completed
Evidence:
- Files Modified: [list of specific files with line counts]
- Tests: [test results, coverage percentages]
- Functionality: [specific features working]
- Integration: [how it connects to existing system]
- Performance: [benchmarks if applicable]

Validation Requested: @agent-critical-auditor
```

## Implementation Steps

### 1. Update project-todo-orchestrator.md
Add mandatory sync requirements:
```markdown
## Mandatory Sync Protocol
- NEVER mark tasks complete in TodoWrite without updating actual todo.md
- ALWAYS require validation evidence before moving tasks
- MUST coordinate with @agent-critical-auditor before marking complete
- Update todo.md file in real-time, not at project end
```

### 2. Update project-implementer.md  
Add sync checkpoints:
```markdown
## Phase 2 Implementation Coordination (MODIFIED)
1. Coordinate parallel work streams with specialist agents
2. **MANDATORY**: Ensure each agent syncs todo.md after task completion
3. Monitor todo.md file for real-time progress tracking
4. Validate todo.md reflects actual implementation status
5. Address gaps between TodoWrite and todo.md immediately
```

### 3. Create project-todo-validation-agent.md
New agent specifically for validation:
```markdown
## Todo Validation Agent
Responsibilities:
- Verify TodoWrite tool matches actual todo.md file
- Cross-check claimed completions with actual implementations  
- Identify discrepancies between tools
- Enforce synchronization protocol
```

## Quality Gates

### Before Any Task Marked "Complete":
1. ✅ TodoWrite updated
2. ✅ Actual todo.md file updated  
3. ✅ Validation evidence provided
4. ✅ @agent-critical-auditor approval received
5. ✅ Both tools show consistent status

### Project-Level Quality Gate:
1. ✅ TodoWrite completion percentage matches todo.md
2. ✅ All "completed" tasks have validation evidence  
3. ✅ No discrepancies between session tools and deliverable files
4. ✅ Todo.md file updated as final deliverable

This protocol ensures todo.md becomes a living, accurate document throughout implementation, not just an afterthought.