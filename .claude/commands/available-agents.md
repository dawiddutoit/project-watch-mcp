# Available Agents Reference

This file provides a comprehensive list of all available agents for use in the project. Agents should reference this file when coordinating work or delegating tasks.

## Agent Orchestration Workflow

### 1. Initialize Context
- `project-context-expert` - Get project info, conventions, patterns
- `context-manager` - Ensure proper context management across agents

### 2. Plan & Organize
- `task-orchestration-manager` - Coordinate agent work, break down complex tasks
- `project-todo-orchestrator` - Create/manage todos if working on features

### 3. Discover & Research
- `project-file-navigator` - Find files, locate code elements
- `researcher` - Conduct research, analyze solutions
- Architecture agents for domain expertise (see below)

### 4. Implement (if requested)
- Developer agents for implementation (see below)

### 5. Quality Assurance
- `qa-testing-expert` - Ensure tests are in place
- `test-automation-architect` - Design test strategies

### 6. Finalize
- `project-todo-orchestrator` - Update task statuses
- `code-review-expert` - Review work for mistakes or missed tasks

## Project-Specific Agents

| Agent | Purpose | When to Use |
|-------|---------|-------------|
| `project-context-expert` | Project questions, configuration, patterns | Understanding project setup, conventions, tooling |
| `project-file-navigator` | File discovery, code search, navigation | Finding files, classes, functions, or code patterns |
| `project-todo-orchestrator` | Task management, todo creation/updates | Managing work items, tracking progress |

## Development Agents

| Agent | Specialization | Use Cases |
|-------|---------------|-----------|
| `python-developer` | Python with TDD | Python implementation, refactoring, test-first development |
| `typescript-pro-engineer` | TypeScript development | Type-safe code, complex type systems, Node.js/browser apps |
| `react-pro-engineer` | React development | React components, hooks, performance optimization |
| `kotlin-developer` | Kotlin backend with TDD | Kotlin Spring Boot services, test-driven development |

## Architecture & Design Agents

| Agent | Domain | Responsibilities |
|-------|--------|-----------------|
| `architect` | Backend architecture | Microservices, API design, system patterns |
| `react-frontend-architect` | Frontend architecture | React architecture, component design, state management |
| `postgresql-pglite-architect` | Database design | PostgreSQL schemas, PgLite browser databases, query optimization |
| `electron-desktop-architect` | Desktop applications | Electron apps, IPC communication, native integrations |

## Quality & Testing Agents

| Agent | Focus Area | Key Tasks |
|-------|------------|-----------|
| `qa-testing-expert` | Quality assurance | Test planning, test cases, defect analysis |
| `test-automation-architect` | Test automation | Test strategies, CI/CD integration, coverage analysis |
| `debugging-expert` | Problem solving | Debug errors, analyze failures, root cause analysis |
| `code-review-expert` | Code quality | Review changes, identify issues, suggest improvements |
| `critical-auditor` | Verification | Verify truthfulness, check accuracy, validate claims |

## Documentation & Design Agents

| Agent | Specialty | Deliverables |
|-------|-----------|--------------|
| `documentation-architect` | Technical documentation | User guides, API docs, README files |
| `api-documentation-specialist` | API documentation | OpenAPI specs, code examples, authentication guides |
| `ux-design-expert` | User experience | UX research, usability testing, accessibility |
| `ui-design-expert` | User interface | UI mockups, design systems, visual components |

## Specialized Utility Agents

| Agent | Function | When Needed |
|-------|----------|-------------|
| `context-manager` | Multi-agent coordination | Complex projects requiring multiple agents |
| `task-orchestration-manager` | Task breakdown | Large features needing decomposition |
| `researcher` | Critical analysis | Evaluating plans, researching solutions |
| `visual-report-generator` | Report generation | Creating dashboards, visual reports |
| `hooks-creator` | Claude hooks | Creating automation hooks for Claude Code |
| `mcp-server-manager` | MCP configuration | Managing MCP server setup and testing |

## Usage Guidelines

### For Coordinating Agents
When you need to delegate work to another agent:
1. Check this file for the appropriate specialist
2. Use the exact agent name without `@agent-` prefix
3. Provide clear context about what you need
4. Specify expected deliverables

### For Project-Specific Agents
- **project-context-expert**: Consult for any project-specific questions before making assumptions
- **project-file-navigator**: Always use for file discovery instead of searching manually
- **project-todo-orchestrator**: Coordinate with for task status updates

### Agent Selection Examples

```markdown
# Need to find a specific class
Use: project-file-navigator

# Need to understand testing conventions
Use: project-context-expert

# Need to implement a Python feature
Use: python-developer

# Need to review code quality
Use: code-review-expert

# Need to create API documentation
Use: api-documentation-specialist
```

## Important Notes

- Always use the exact agent name as specified in this file
- Agents can call other agents when needed
- Prefer specialized agents over general-purpose ones
- Update this file when new agents are added to the system