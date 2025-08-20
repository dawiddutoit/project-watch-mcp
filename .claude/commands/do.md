# Command: do

## Purpose
Execute exactly what the user asks - nothing more, nothing less.

## GOLDEN RULE
**If the user asks to fix a bug, fix ONLY that bug.**  
**If they ask for a feature, implement ONLY that feature.**  
**Never add improvements they didn't request.**

## Execution Flow

### 1. First Actions (Always)
```
1. Check CLAUDE.md for project rules
2. Understand EXACTLY what was requested
3. If unclear → ASK before doing anything
4. If user provides a todo.md path → That's your implementation plan (see step 2)
```

### 2. Decision Tree
```
Did user provide a todo.md file path?
  YES → This is your implementation plan:
        - Read the todo.md file
        - Use it as source of truth
        - UPDATE it with [x] marks as you complete tasks
        - Continue to step 3
  NO  → Continue below

Is it a simple change (1 file, <50 lines)?
  YES → Just do it directly
  NO  → Continue

Do you need to find files or understand code structure?
  YES → Use @agent-project-memory-navigator
  NO  → Continue

Is it complex with multiple interconnected changes?
  YES → Use @agent-project-todo-orchestrator to create todo.md
  NO  → Just implement it directly
```

### 3. Implementation Rules
- **Extend existing files** - Don't create temporary files unless absolutely necessary
- **Follow existing patterns** - Match the project's current style
- **Test as you go** - Verify each change works before moving on
- **Update todo.md if provided** - Mark [x] in the actual file as you complete tasks
- **Stop when done** - Don't add "bonus" features

## Common Scenarios

### Bug Fix
```bash
/do Fix the null reference error in DataService
```
Action: Find the file, fix the specific bug, test it works. Done.

### Simple Feature
```bash
/do Add a health check endpoint
```
Action: Add the endpoint to existing code, implement logic, test. Done.

### Complex Feature
```bash
/do Implement OAuth2 authentication with Google
```
Action: This touches multiple files → Create todo.md with clear tasks.

### Complex Feature with Existing TODO
```bash
/do implement /path/to/project/todo.md
```
Action: 
1. Read the todo.md file
2. Start implementing tasks in order
3. After EACH task completion:
   - Edit todo.md: change `[ ]` to `[x]`
   - Update status: `[ ] Not Started` → `[x] Completed`
4. User can check todo.md anytime to see progress

### Unclear Request
```bash
/do Make the API better
```
Action: ASK "What specific improvements do you want?"

## Anti-Patterns to Avoid

❌ **DON'T DO THIS**:
```
# User: "Fix the sorting bug"
# You: "I'll fix the bug AND refactor the entire class AND create an example class because it was too hard to edit the file AND add caching AND..."
```

❌ **DON'T DO THIS**:
```
# User: "implement todo.md"
# You: *tracks progress internally without updating the file*
```

✅ **DO THIS INSTEAD**:
```
# User: "Fix the sorting bug"
# You: "I'll fix the sorting bug in the comparator method. I will ensure that I do it in the comparator method and not a new file."
```

✅ **DO THIS INSTEAD**:
```
# User: "implement todo.md"
# You: *reads todo.md, implements task 1, edits todo.md to mark [x], continues*
```

## Agent Usage Guide

**@agent-project-memory-navigator**
- When: Can't find a file or need to understand code structure
- Example: "Where is the authentication logic?"

**@agent-project-todo-orchestrator**
- When: Multiple files need coordinated changes AND no todo.md provided
- Example: "Add a new module with controllers, services, and tests"

**Domain Agents** (@python-developer, @qa-testing-expert, etc.)
- When: You need specific expertise
- Example: "What's the best Python pattern for this?"

**NO AGENT NEEDED**
- When: You know what to do and where
- Example: "Add a missing import statement"

## Quick Reference

| Situation                  | Action                                    |
|----------------------------|-------------------------------------------|
| User provides todo.md      | Read it, implement it, UPDATE it         |
| Simple bug                 | Fix it directly                           |
| Can't find file            | Use memory navigator                      |
| Complex feature (no todo)  | Create todo.md                            |
| Complex feature (has todo) | Follow & update the provided todo.md     |
| Unclear request            | Ask for clarification                     |

## Remember

1. **Read the request carefully**
2. **If todo.md provided, it's your contract - UPDATE IT**
3. **Do exactly what's asked**
4. **Nothing more**
5. **When done, you're done**

The best code change is the smallest change that solves the exact problem.