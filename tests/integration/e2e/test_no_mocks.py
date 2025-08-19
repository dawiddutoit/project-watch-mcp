"""Test to ensure no mock implementations exist in production code."""

import ast
import re
from pathlib import Path


def get_context_for_line(tree: ast.AST, line_num: int) -> str:
    """Get the class/method context for a given line number."""
    context_parts = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                if node.lineno <= line_num <= (node.end_lineno or node.lineno):
                    context_parts.append(f"class {node.name}")

                    # Check if line is within a method of this class
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if hasattr(item, "lineno") and hasattr(item, "end_lineno"):
                                if item.lineno <= line_num <= (item.end_lineno or item.lineno):
                                    context_parts.append(f"method {item.name}")
                                    return " -> ".join(context_parts)
                    return " -> ".join(context_parts)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                if node.lineno <= line_num <= (node.end_lineno or node.lineno):
                    # Check if this function is not inside a class (already handled above)
                    parent_is_class = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef):
                            if node in ast.walk(parent):
                                parent_is_class = True
                                break
                    if not parent_is_class:
                        return f"function {node.name}"

    return "module level"


def test_no_mocks_in_production():
    """Ensure no mock or stub implementations exist in production code.
    
    Note: MockEmbeddingsProvider is an intentional development/testing tool,
    not a production mock. It's designed to allow development without requiring
    external API keys.
    """
    src_dir = Path(__file__).parent.parent / "src" / "project_watch_mcp"

    # Pattern to match mock/stub (case-insensitive)
    mock_pattern = re.compile(r"\b(mock|stub)\b", re.IGNORECASE)
    
    # Files and patterns that are allowed to have "mock" in them
    allowed_exceptions = {
        "utils/embedding.py": ["MockEmbeddingsProvider", "mock embeddings", '"mock"', "== 'mock'", '== "mock"'],  # Development embedding provider
        "config.py": ["mock"],  # Configuration for selecting the mock provider
        "cli.py": ["mock", "Mock"],  # CLI help text mentioning mock option
        "neo4j_rag.py": ["defaults to mock"],  # Documentation mentioning mock default
    }

    violations = []

    # Check all Python files in the src directory
    for py_file in src_dir.rglob("*.py"):
        # Skip __pycache__ directories
        if "__pycache__" in str(py_file):
            continue

        with open(py_file, encoding="utf-8") as f:
            content = f.read()

        # Parse the AST for context information
        try:
            tree = ast.parse(content, filename=str(py_file))
        except SyntaxError:
            tree = None

        lines = content.splitlines()
        
        # Get relative path for checking exceptions
        rel_path = py_file.relative_to(src_dir)
        rel_path_str = str(rel_path).replace("\\", "/")

        # Find all matches with line numbers
        for line_num, line in enumerate(lines, 1):
            if mock_pattern.search(line):
                # Skip if it's in a comment or docstring
                stripped = line.strip()
                if not (
                    stripped.startswith("#")
                    or stripped.startswith('"""')
                    or stripped.startswith("'''")
                ):
                    # Check if this is an allowed exception
                    is_allowed = False
                    for allowed_file, allowed_patterns in allowed_exceptions.items():
                        if allowed_file in rel_path_str:
                            # Check if any allowed pattern is in the line
                            for pattern in allowed_patterns:
                                if pattern.lower() in line.lower():
                                    is_allowed = True
                                    break
                            if is_allowed:
                                break
                    
                    if not is_allowed:
                        # Get context (class/method name)
                        context = get_context_for_line(tree, line_num) if tree else "unknown context"

                        # Format the violation
                        file_path = py_file.relative_to(src_dir.parent.parent)
                        violation = f"{file_path}:{line_num}"
                        violation += f"\n    Context: {context}"
                        violation += f"\n    Code: {line.strip()}"
                        violations.append(violation)

    if violations:
        violation_output = "\n\n  ".join(violations)
        assert (
            False
        ), f"No mock implementations allowed. Found {len(violations)} violation(s):\n\n  {violation_output}"
