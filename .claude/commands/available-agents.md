# Available Agents Reference

This file provides a comprehensive list of all available agents for use in the project. Agents should reference this file when coordinating work or delegating tasks.

## Agent Orchestration Workflow

### 1. Initialize Context
- `@agent-project-context-expert` - Get project info, conventions, patterns
- `@agent-context-manager` - Ensure proper context management across agents

### 2. Plan & Organize
- `@agent-task-orchestration-manager` - Coordinate agent work, break down complex tasks
- `@agent-project-todo-orchestrator` - Create/manage todos if working on features

### 3. Discover & Research
- `@agent-project-memory-navigator` - Find files, locate code elements
- Architecture agents for domain expertise (see below)

### 4. Implement (if requested)
- Developer agents for implementation (see below)

### 5. Quality Assurance
- `@agent-qa-testing-expert` - Ensure tests are in place
- `@agent-test-automation-architect` - Design test strategies

### 6. Finalize
- `@agent-project-todo-orchestrator` - Update task statuses
- `@agent-code-review-expert` - Review work for mistakes or missed tasks

## Project-Specific Agents

| Agent | Purpose | When to Use |
|-------|---------|-------------|
| `@agent-project-context-expert` | Project questions, configuration, patterns | Understanding project setup, conventions, tooling |
| `@agent-project-memory-navigator` | File discovery, code search, navigation | Finding files, classes, functions, or code patterns |
| `@agent-project-todo-orchestrator` | Task management, todo creation/updates | Managing work items, tracking progress |

## Development Agents

| Agent | Specialization | Use Cases |
|-------|---------------|-----------|
| `@agent-python-developer` | Python with TDD | Python implementation, refactoring, test-first development |
| `@agent-typescript-pro-engineer` | TypeScript development | Type-safe code, complex type systems, Node.js/browser apps |
| `@agent-react-pro-engineer` | React development | React components, hooks, performance optimization |
| `@agent-kotlin-backend-tdd` | Kotlin backend with TDD | Kotlin Spring Boot services, test-driven development |

## Architecture & Design Agents

| Agent | Domain | Responsibilities |
|-------|--------|-----------------|
| `@agent-backend-system-architect` | Backend architecture | Microservices, API design, system patterns |
| `@agent-react-frontend-architect` | Frontend architecture | React architecture, component design, state management |
| `@agent-postgresql-pglite-architect` | Database design | PostgreSQL schemas, PgLite browser databases, query optimization |
| `@agent-electron-desktop-architect` | Desktop applications | Electron apps, IPC communication, native integrations |

## Quality & Testing Agents

| Agent | Focus Area | Key Tasks |
|-------|------------|-----------|
| `@agent-qa-testing-expert` | Quality assurance | Test planning, test cases, defect analysis |
| `@agent-test-automation-architect` | Test automation | Test strategies, CI/CD integration, coverage analysis |
| `@agent-debugging-expert` | Problem solving | Debug errors, analyze failures, root cause analysis |
| `@agent-code-review-expert` | Code quality | Review changes, identify issues, suggest improvements |
| `@agent-critical-auditor` | Verification | Verify truthfulness, check accuracy, validate claims |

## Documentation & Design Agents

| Agent | Specialty | Deliverables |
|-------|-----------|--------------|
| `@agent-documentation-architect` | Technical documentation | User guides, API docs, README files |
| `@agent-api-documentation-specialist` | API documentation | OpenAPI specs, code examples, authentication guides |
| `@agent-ux-design-expert` | User experience | UX research, usability testing, accessibility |
| `@agent-ui-design-expert` | User interface | UI mockups, design systems, visual components |

## Specialized Utility Agents

| Agent | Function | When Needed |
|-------|----------|-------------|
| `@agent-context-manager` | Multi-agent coordination | Complex projects requiring multiple agents |
| `@agent-task-orchestration-manager` | Task breakdown | Large features needing decomposition |
| `@agent-strategic-research-analyst` | Critical analysis | Evaluating plans, researching solutions |
| `@agent-visual-report-generator` | Report generation | Creating dashboards, visual reports |
| `@agent-hooks-creator` | Claude hooks | Creating automation hooks for Claude Code |
| `@agent-mcp-server-manager` | MCP configuration | Managing MCP server setup and testing |

## Usage Guidelines

### For Coordinating Agents
When you need to delegate work to another agent:
1. Check this file for the appropriate specialist
2. Use the exact `@agent-` prefix format
3. Provide clear context about what you need
4. Specify expected deliverables

### For Project-Specific Agents
- **project-context-expert**: Consult for any project-specific questions before making assumptions
- **project-memory-navigator**: Always use for file discovery instead of searching manually
- **project-todo-orchestrator**: Coordinate with for task status updates

### Agent Selection Examples

```markdown
# Need to find a specific class
Use: @agent-project-memory-navigator

# Need to understand testing conventions
Use: @agent-project-context-expert

# Need to implement a Python feature
Use: @agent-python-developer

# Need to review code quality
Use: @agent-code-review-expert

# Need to create API documentation
Use: @agent-api-documentation-specialist
```

## Important Notes

- Always use the full agent name with `@agent-` prefix
- Agents can call other agents when needed
- Prefer specialized agents over general-purpose ones
- Update this file when new agents are added to the system