"""
Integration tests for multi-language complexity analysis in MCP server.

These tests verify the actual behavior of the analyze_complexity tool
with different language files.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import textwrap

from fastmcp.exceptions import ToolError
from src.project_watch_mcp.server import create_mcp_server
from src.project_watch_mcp.repository_monitor import RepositoryMonitor
from src.project_watch_mcp.neo4j_rag import Neo4jRAG
from src.project_watch_mcp.complexity_analysis import AnalyzerRegistry
from src.project_watch_mcp.complexity_analysis.languages.python_analyzer import PythonComplexityAnalyzer
from src.project_watch_mcp.complexity_analysis.languages.java_analyzer import JavaComplexityAnalyzer
from src.project_watch_mcp.complexity_analysis.languages.kotlin_analyzer import KotlinComplexityAnalyzer


class TestAnalyzeComplexityMultiLanguage:
    """Test the analyze_complexity tool with multiple languages."""
    
    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with test files."""
        # Python file
        python_file = tmp_path / "example.py"
        python_file.write_text(textwrap.dedent("""
            def simple_function():
                return 42
            
            def moderate_function(x):
                if x > 0:
                    return x * 2
                else:
                    return x / 2
            
            def complex_function(data):
                result = []
                for item in data:
                    if item > 10:
                        if item % 2 == 0:
                            result.append(item * 2)
                        else:
                            result.append(item + 1)
                    elif item < 0:
                        result.append(abs(item))
                    else:
                        result.append(item)
                return result
        """))
        
        # Java file
        java_file = tmp_path / "Example.java"
        java_file.write_text(textwrap.dedent("""
            public class Example {
                public int simpleMethod() {
                    return 42;
                }
                
                public int moderateMethod(int x) {
                    if (x > 0) {
                        return x * 2;
                    } else {
                        return x / 2;
                    }
                }
                
                public void complexMethod(int[] data) {
                    for (int item : data) {
                        if (item > 10) {
                            if (item % 2 == 0) {
                                System.out.println(item * 2);
                            } else {
                                System.out.println(item + 1);
                            }
                        } else if (item < 0) {
                            System.out.println(Math.abs(item));
                        } else {
                            System.out.println(item);
                        }
                    }
                }
            }
        """))
        
        # Kotlin file
        kotlin_file = tmp_path / "Example.kt"
        kotlin_file.write_text(textwrap.dedent("""
            class Example {
                fun simpleMethod(): Int = 42
                
                fun moderateMethod(x: Int): Int {
                    return if (x > 0) {
                        x * 2
                    } else {
                        x / 2
                    }
                }
                
                fun complexMethod(data: IntArray) {
                    for (item in data) {
                        when {
                            item > 10 -> {
                                if (item % 2 == 0) {
                                    println(item * 2)
                                } else {
                                    println(item + 1)
                                }
                            }
                            item < 0 -> println(kotlin.math.abs(item))
                            else -> println(item)
                        }
                    }
                }
            }
        """))
        
        return {
            "path": tmp_path,
            "python": python_file,
            "java": java_file,
            "kotlin": kotlin_file,
        }
    
    @pytest.fixture
    def mock_monitor(self, temp_repo):
        """Create a mock repository monitor."""
        monitor = Mock(spec=RepositoryMonitor)
        monitor.repo_path = temp_repo["path"]
        monitor.is_running = True
        return monitor
    
    @pytest.fixture
    def mock_neo4j(self):
        """Create a mock Neo4j RAG."""
        neo4j = Mock(spec=Neo4jRAG)
        return neo4j
    
    @pytest.fixture
    def mcp_server(self, mock_monitor, mock_neo4j):
        """Create MCP server with mocked dependencies."""
        # Ensure analyzers are registered
        if not AnalyzerRegistry.get_analyzer("python"):
            AnalyzerRegistry.register("python", PythonComplexityAnalyzer)
        if not AnalyzerRegistry.get_analyzer("java"):
            AnalyzerRegistry.register("java", JavaComplexityAnalyzer)
        if not AnalyzerRegistry.get_analyzer("kotlin"):
            AnalyzerRegistry.register("kotlin", KotlinComplexityAnalyzer)
        
        return create_mcp_server(
            repository_monitor=mock_monitor,
            neo4j_rag=mock_neo4j,
            project_name="test_project"
        )
    
    @pytest.mark.asyncio
    async def test_analyze_python_file_with_new_implementation(self, mcp_server, temp_repo):
        """Test analyzing a Python file with the new multi-language implementation."""
        # Verify tool is registered
        tools = await mcp_server.get_tools()
        assert "analyze_complexity" in tools
        
        # Get the actual tool function
        tool_func = mcp_server._tool_manager._tools["analyze_complexity"].fn
        
        # Call the tool with Python file
        result = await tool_func(file_path=str(temp_repo["python"]))
        
        assert result is not None
        assert result.structured_content is not None
        
        # Check that it detected Python
        content = result.structured_content
        assert "language" in content or "file" in content  # Should have language info
        
        # Check complexity results
        assert "summary" in content
        assert "functions" in content
        assert content["summary"]["function_count"] == 3  # 3 functions in our test file
    
    @pytest.mark.asyncio
    async def test_analyze_java_file(self, mcp_server, temp_repo):
        """Test analyzing a Java file."""
        tool_func = mcp_server._tool_manager._tools["analyze_complexity"].fn
        
        # This test will fail initially - Java support needs to be implemented
        result = await tool_func(file_path=str(temp_repo["java"]))
        
        assert result is not None
        assert result.structured_content is not None
        
        content = result.structured_content
        assert "language" in content
        assert content["language"] == "java"
        assert "summary" in content
        assert "functions" in content
    
    @pytest.mark.asyncio
    async def test_analyze_kotlin_file(self, mcp_server, temp_repo):
        """Test analyzing a Kotlin file."""
        tool_func = mcp_server._tool_manager._tools["analyze_complexity"].fn
        
        # This test will fail initially - Kotlin support needs to be implemented
        result = await tool_func(file_path=str(temp_repo["kotlin"]))
        
        assert result is not None
        assert result.structured_content is not None
        
        content = result.structured_content
        assert "language" in content
        assert content["language"] == "kotlin"
        assert "summary" in content
        assert "functions" in content
    
    @pytest.mark.asyncio
    async def test_unsupported_file_type_error(self, mcp_server, tmp_path):
        """Test that unsupported file types raise appropriate errors."""
        # Create a Rust file (unsupported)
        rust_file = tmp_path / "test.rs"
        rust_file.write_text("fn main() { println!(\"Hello\"); }")
        
        tool_func = mcp_server._tool_manager._tools["analyze_complexity"].fn
        
        # Should raise ToolError for unsupported file type
        with pytest.raises(ToolError) as exc_info:
            await tool_func(file_path=str(rust_file))
        
        error_msg = str(exc_info.value).lower()
        assert "not supported" in error_msg or "unsupported" in error_msg or "not a python file" in error_msg
    
    @pytest.mark.asyncio
    async def test_explicit_language_parameter(self, mcp_server, tmp_path):
        """Test using explicit language parameter to override extension detection."""
        # Create a file with wrong extension
        weird_file = tmp_path / "code.txt"
        weird_file.write_text(textwrap.dedent("""
            def test_function():
                return 42
        """))
        
        tool_func = mcp_server._tool_manager._tools["analyze_complexity"].fn
        
        # This test defines expected behavior - should support language parameter
        # Will fail initially as this needs to be implemented
        result = await tool_func(
            file_path=str(weird_file),
            language="python"  # Force Python analysis
        )
        
        assert result is not None
        assert result.structured_content["language"] == "python"