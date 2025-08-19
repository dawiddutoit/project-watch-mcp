"""Tests for the analyze_complexity MCP tool."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.exceptions import ToolError

from project_watch_mcp.server import create_mcp_server


@pytest.fixture
async def temp_repository():
    """Create a temporary repository with Python files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        
        # Create a simple Python file
        simple_file = repo_path / "simple.py"
        simple_file.write_text("""
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

def get_greeting(name):
    return f"Hello, {name}!"
""")
        
        # Create a moderately complex file
        moderate_file = repo_path / "moderate.py"
        moderate_file.write_text("""
def process_data(items, threshold=10):
    '''Process items with moderate complexity.'''
    result = []
    for item in items:
        if item > threshold:
            if item % 2 == 0:
                result.append(item * 2)
            else:
                result.append(item * 3)
        elif item < 0:
            result.append(abs(item))
        else:
            result.append(item)
    return result

def validate_input(value, min_val=0, max_val=100):
    '''Validate input with multiple conditions.'''
    if value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    if value < min_val or value > max_val:
        return False
    if value == 50:
        return True
    if value > 75:
        return value % 5 == 0
    return True
""")
        
        # Create a complex file
        complex_file = repo_path / "complex.py"
        complex_file.write_text("""
def complex_function(data, options=None):
    '''A function with high cyclomatic complexity.'''
    if not data:
        return None
    
    result = []
    options = options or {}
    
    for item in data:
        if isinstance(item, dict):
            if 'value' in item:
                val = item['value']
                if val > 100:
                    if options.get('double_high'):
                        result.append(val * 2)
                    elif options.get('triple_high'):
                        result.append(val * 3)
                    else:
                        result.append(val)
                elif val < 0:
                    if options.get('abs_negative'):
                        result.append(abs(val))
                    else:
                        result.append(val * -1)
                elif val == 0:
                    result.append(1)
                else:
                    if val % 2 == 0:
                        if val % 3 == 0:
                            result.append(val // 6)
                        else:
                            result.append(val // 2)
                    elif val % 3 == 0:
                        result.append(val // 3)
                    else:
                        result.append(val)
            elif 'text' in item:
                if len(item['text']) > 10:
                    result.append(item['text'][:10])
                else:
                    result.append(item['text'])
        elif isinstance(item, (int, float)):
            if item > 50:
                result.append(item // 2)
            else:
                result.append(item * 2)
        else:
            result.append(str(item))
    
    return result

def another_complex(x, y, z, mode='default'):
    '''Another complex function for testing.'''
    if mode == 'add':
        if x > 0 and y > 0:
            return x + y + z
        elif x > 0:
            return x + z
        else:
            return y + z
    elif mode == 'multiply':
        if x != 0 and y != 0:
            if z != 0:
                return x * y * z
            else:
                return x * y
        elif x != 0:
            return x * z
        else:
            return y * z
    elif mode == 'complex':
        result = 0
        if x > y:
            if x > z:
                result = x
            else:
                result = z
        elif y > z:
            result = y
        else:
            result = z
        
        if result > 100:
            if mode == 'complex':
                return result ** 2
            else:
                return result * 2
        return result
    else:
        return 0
""")
        
        # Create a non-Python file
        text_file = repo_path / "readme.txt"
        text_file.write_text("This is not a Python file.")
        
        # Create a Python file with syntax error
        error_file = repo_path / "syntax_error.py"
        error_file.write_text("""
def broken_function(
    print("This has a syntax error"
""")
        
        yield repo_path


@pytest.fixture
async def mcp_server(temp_repository):
    """Create an MCP server instance for testing."""
    monitor = MagicMock()
    monitor.repo_path = temp_repository
    monitor.is_running = False
    monitor.file_patterns = ["*.py"]
    monitor.scan_repository = AsyncMock(return_value=[])
    monitor.start = AsyncMock()
    monitor.process_all_changes = AsyncMock(return_value=[])
    
    neo4j = MagicMock()
    neo4j.search_semantic = AsyncMock(return_value=[])
    neo4j.search_by_pattern = AsyncMock(return_value=[])
    neo4j.get_repository_stats = AsyncMock(return_value={
        "total_files": 0,
        "total_chunks": 0,
        "total_size": 0,
        "languages": []
    })
    neo4j.get_file_metadata = AsyncMock(return_value=None)
    neo4j.index_file = AsyncMock()
    neo4j.update_file = AsyncMock()
    neo4j.delete_file = AsyncMock()
    
    server = create_mcp_server(monitor, neo4j, "test_project")
    return server, monitor, neo4j


class TestAnalyzeComplexity:
    """Test the analyze_complexity tool."""
    
    async def test_analyze_simple_file(self, mcp_server, temp_repository):
        """Test analyzing a simple Python file."""
        server, monitor, _ = mcp_server
        
        # Get the tool
        tool = await server.get_tool("analyze_complexity")
        assert tool is not None, "analyze_complexity tool not found"
        
        # Analyze simple.py
        result = await tool.run({"file_path": "simple.py", "include_metrics": True})
        
        assert result is not None
        assert result.structured_content is not None
        
        data = result.structured_content
        assert data["file"] == "simple.py"
        assert data["summary"]["total_complexity"] == 3  # 3 simple functions
        assert data["summary"]["average_complexity"] == 1.0
        assert data["summary"]["function_count"] == 3
        assert "maintainability_index" in data["summary"]
        assert data["summary"]["complexity_grade"] in ["A", "B"]  # Should be high grade
        
        # Check functions
        assert len(data["functions"]) == 3
        for func in data["functions"]:
            assert func["complexity"] == 1
            assert func["rank"] == "A"
            assert func["classification"] == "simple"
        
        # Should have positive recommendation
        assert "within acceptable limits" in data["recommendations"][0]
    
    async def test_analyze_moderate_complexity(self, mcp_server, temp_repository):
        """Test analyzing a file with moderate complexity."""
        server, monitor, _ = mcp_server
        
        tool = await server.get_tool("analyze_complexity")
        assert tool is not None
        
        result = await tool.run({"file_path": "moderate.py", "include_metrics": True})
        
        assert result is not None
        data = result.structured_content
        
        assert data["file"] == "moderate.py"
        assert data["summary"]["total_complexity"] > 3
        assert data["summary"]["average_complexity"] > 1.0
        assert len(data["functions"]) == 2
        
        # Check that functions have moderate complexity
        for func in data["functions"]:
            assert func["complexity"] > 1
            assert func["classification"] in ["simple", "moderate"]
    
    async def test_analyze_complex_file(self, mcp_server, temp_repository):
        """Test analyzing a file with high complexity."""
        server, monitor, _ = mcp_server
        
        tool = await server.get_tool("analyze_complexity")
        assert tool is not None
        
        result = await tool.run({"file_path": "complex.py", "include_metrics": True})
        
        assert result is not None
        data = result.structured_content
        
        assert data["file"] == "complex.py"
        
        # Should have complex functions
        complex_funcs = [f for f in data["functions"] if f["complexity"] > 10]
        assert len(complex_funcs) > 0
        
        # Should have refactoring recommendations
        has_refactor_rec = any("refactor" in rec.lower() for rec in data["recommendations"])
        assert has_refactor_rec
        
        # Check classifications
        has_complex = any(f["classification"] in ["complex", "very-complex"] 
                         for f in data["functions"])
        assert has_complex
    
    async def test_analyze_without_metrics(self, mcp_server, temp_repository):
        """Test analyzing without additional metrics."""
        server, monitor, _ = mcp_server
        
        tool = await server.get_tool("analyze_complexity")
        assert tool is not None
        
        result = await tool.run({"file_path": "simple.py", "include_metrics": False})
        
        assert result is not None
        data = result.structured_content
        
        # Should not have maintainability index
        assert "maintainability_index" not in data["summary"]
        assert "complexity_grade" not in data["summary"]
        
        # Should still have basic metrics
        assert "total_complexity" in data["summary"]
        assert "average_complexity" in data["summary"]
        assert "function_count" in data["summary"]
    
    async def test_analyze_non_python_file(self, mcp_server, temp_repository):
        """Test that non-Python files are rejected."""
        server, monitor, _ = mcp_server
        
        tool = await server.get_tool("analyze_complexity")
        assert tool is not None
        
        with pytest.raises(ToolError) as exc_info:
            await tool.run({"file_path": "readme.txt"})
        
        assert "not a Python file" in str(exc_info.value)
    
    async def test_analyze_nonexistent_file(self, mcp_server, temp_repository):
        """Test analyzing a file that doesn't exist."""
        server, monitor, _ = mcp_server
        
        tool = await server.get_tool("analyze_complexity")
        assert tool is not None
        
        with pytest.raises(ToolError) as exc_info:
            await tool.run({"file_path": "nonexistent.py"})
        
        assert "does not exist" in str(exc_info.value)
    
    async def test_analyze_syntax_error_file(self, mcp_server, temp_repository):
        """Test analyzing a file with syntax errors."""
        server, monitor, _ = mcp_server
        
        tool = await server.get_tool("analyze_complexity")
        assert tool is not None
        
        with pytest.raises(ToolError) as exc_info:
            await tool.run({"file_path": "syntax_error.py"})
        
        assert "Failed to parse Python file" in str(exc_info.value)
    
    async def test_absolute_path(self, mcp_server, temp_repository):
        """Test analyzing with absolute path."""
        server, monitor, _ = mcp_server
        
        tool = await server.get_tool("analyze_complexity")
        assert tool is not None
        
        absolute_path = str(temp_repository / "simple.py")
        result = await tool.run({"file_path": absolute_path})
        
        assert result is not None
        data = result.structured_content
        assert data["file"] == "simple.py"
    
    async def test_complexity_rankings(self, mcp_server, temp_repository):
        """Test that complexity rankings are correctly assigned."""
        server, monitor, _ = mcp_server
        
        # Create a file with various complexity levels
        test_file = temp_repository / "rankings.py"
        test_file.write_text("""
def rank_a():  # Complexity 1
    return 1

def rank_b():  # Complexity 6-10
    x = 1
    if x > 0:
        if x > 1:
            if x > 2:
                if x > 3:
                    if x > 4:
                        if x > 5:
                            return x
    return 0

def rank_c():  # Complexity 11-20
    x = 1
    for i in range(5):
        if i == 0:
            x += 1
        elif i == 1:
            x += 2
        elif i == 2:
            x += 3
        elif i == 3:
            x += 4
        elif i == 4:
            x += 5
        elif i == 5:
            x += 6
        elif i == 6:
            x += 7
        elif i == 7:
            x += 8
        elif i == 8:
            x += 9
        elif i == 9:
            x += 10
        elif i == 10:
            x += 11
    return x
""")
        
        tool = await server.get_tool("analyze_complexity")
        assert tool is not None
        
        result = await tool.run({"file_path": "rankings.py"})
        
        assert result is not None
        data = result.structured_content
        
        # Functions should be sorted by complexity
        functions = data["functions"]
        assert functions[0]["complexity"] >= functions[-1]["complexity"]
        
        # Check various ranks exist
        ranks = {f["rank"] for f in functions}
        assert "A" in ranks  # Should have at least one simple function


@pytest.mark.parametrize("radon_available", [False])
async def test_radon_not_available(temp_repository, radon_available):
    """Test error when radon is not available."""
    with patch("project_watch_mcp.server.RADON_AVAILABLE", radon_available):
        monitor = MagicMock()
        monitor.repo_path = temp_repository
        neo4j = MagicMock()
        
        server = create_mcp_server(monitor, neo4j, "test_project")
        
        tool = await server.get_tool("analyze_complexity")
        assert tool is not None
        
        with pytest.raises(ToolError) as exc_info:
            await tool.run({"file_path": "simple.py"})
        
        assert "Radon library not available" in str(exc_info.value)