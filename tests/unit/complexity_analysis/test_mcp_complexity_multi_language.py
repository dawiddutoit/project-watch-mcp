"""
Test suite for multi-language complexity analysis MCP tool integration.

This test suite follows TDD principles to define the expected behavior
of the analyze_complexity MCP tool supporting multiple languages.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock, create_autospec
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult, TextContent
import tempfile
import textwrap
import asyncio

from src.project_watch_mcp.complexity_analysis import (
    AnalyzerRegistry,
    ComplexityResult,
    ComplexitySummary,
    FunctionComplexity,
    ComplexityGrade,
)
from src.project_watch_mcp.complexity_analysis.languages.python_analyzer import PythonComplexityAnalyzer
from src.project_watch_mcp.complexity_analysis.languages.java_analyzer import JavaComplexityAnalyzer
from src.project_watch_mcp.complexity_analysis.languages.kotlin_analyzer import KotlinComplexityAnalyzer


class TestMultiLanguageComplexityMCPTool:
    """Test suite for multi-language complexity analysis via MCP tool."""
    
    @pytest.fixture
    def mock_repo_monitor(self):
        """Create a mock repository monitor."""
        monitor = Mock()
        monitor.repo_path = Path("/test/repo")
        return monitor
    
    @pytest.fixture
    def temp_test_files(self, tmp_path):
        """Create temporary test files for different languages."""
        # Python test file
        python_file = tmp_path / "test.py"
        python_file.write_text(textwrap.dedent("""
            def simple_function():
                return 42
            
            def complex_function(x):
                if x > 0:
                    for i in range(x):
                        if i % 2 == 0:
                            print(i)
                        elif i % 3 == 0:
                            print(i * 2)
                        else:
                            print(i * 3)
                else:
                    return -1
        """))
        
        # Java test file
        java_file = tmp_path / "Test.java"
        java_file.write_text(textwrap.dedent("""
            public class Test {
                public int simpleMethod() {
                    return 42;
                }
                
                public void complexMethod(int x) {
                    if (x > 0) {
                        for (int i = 0; i < x; i++) {
                            if (i % 2 == 0) {
                                System.out.println(i);
                            } else if (i % 3 == 0) {
                                System.out.println(i * 2);
                            } else {
                                System.out.println(i * 3);
                            }
                        }
                    }
                }
            }
        """))
        
        # Kotlin test file
        kotlin_file = tmp_path / "Test.kt"
        kotlin_file.write_text(textwrap.dedent("""
            class Test {
                fun simpleMethod(): Int {
                    return 42
                }
                
                fun complexMethod(x: Int) {
                    if (x > 0) {
                        for (i in 0 until x) {
                            when {
                                i % 2 == 0 -> println(i)
                                i % 3 == 0 -> println(i * 2)
                                else -> println(i * 3)
                            }
                        }
                    }
                }
            }
        """))
        
        return {
            "python": python_file,
            "java": java_file,
            "kotlin": kotlin_file,
        }
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_with_explicit_language_parameter(self, mock_repo_monitor, temp_test_files):
        """Test that analyze_complexity accepts an explicit language parameter."""
        from src.project_watch_mcp.server import create_mcp_server
        from src.project_watch_mcp.neo4j_rag import Neo4jRAG
        
        # This test defines the expected behavior - the tool should accept a language parameter
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = temp_test_files["python"].parent
            
            # Create mock Neo4jRAG
            mock_neo4j = Mock(spec=Neo4jRAG)
            
            # The analyze_complexity tool should accept a language parameter
            # This will fail initially - we need to implement this
            server = create_mcp_server(
                repository_monitor=mock_repo_monitor,
                neo4j_rag=mock_neo4j,
                project_name="test_project"
            )
            
            # Get the analyze_complexity tool
            analyze_tool = None
            for name, tool in server._tool_manager._tools.items():
                if name == "analyze_complexity":
                    analyze_tool = tool
                    break
            
            assert analyze_tool is not None, "analyze_complexity tool not found"
            
            # Check if the tool signature includes language parameter
            import inspect
            sig = inspect.signature(analyze_tool)
            params = list(sig.parameters.keys())
            
            # This assertion will fail initially - we need to add language parameter
            assert 'language' in params, "analyze_complexity should accept a language parameter"
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_auto_detects_language(self, mock_repo_monitor, temp_test_files):
        """Test that analyze_complexity automatically detects language from file extension."""
        from src.project_watch_mcp.server import create_server
        
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = temp_test_files["python"].parent
            
            # Mock the AnalyzerRegistry to track calls
            with patch.object(AnalyzerRegistry, 'get_analyzer_for_file') as mock_get_analyzer:
                mock_analyzer = Mock(spec=PythonComplexityAnalyzer)
                mock_analyzer.analyze.return_value = ComplexityResult(
                    file_path=temp_test_files["python"],
                    language="python",
                    summary=ComplexitySummary(
                        total_complexity=5,
                        average_complexity=2.5,
                        function_count=2,
                    ),
                    functions=[],
                    classes=[],
                )
                mock_get_analyzer.return_value = mock_analyzer
                
                server = await create_server()
                
                # Should auto-detect Python from .py extension
                with patch.object(server, 'analyze_complexity', wraps=server.analyze_complexity):
                    # This will fail initially - need to implement auto-detection
                    result = await server.analyze_complexity(
                        file_path=str(temp_test_files["python"])
                    )
                    
                    mock_get_analyzer.assert_called_once()
                    assert mock_analyzer.analyze.called
    
    @pytest.mark.asyncio 
    async def test_analyze_complexity_supports_python(self, mock_repo_monitor, temp_test_files):
        """Test that Python files are properly analyzed."""
        from src.project_watch_mcp.server import create_server
        
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = temp_test_files["python"].parent
            
            # Ensure Python analyzer is registered
            if not AnalyzerRegistry.get_analyzer("python"):
                AnalyzerRegistry.register("python", PythonComplexityAnalyzer)
            
            server = await create_server()
            
            # This should work after implementation
            result = await server.analyze_complexity(
                file_path=str(temp_test_files["python"])
            )
            
            assert result is not None
            assert isinstance(result, ToolResult)
            assert result.structured_content is not None
            assert result.structured_content["language"] == "python"
            assert "summary" in result.structured_content
            assert "functions" in result.structured_content
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_supports_java(self, mock_repo_monitor, temp_test_files):
        """Test that Java files are properly analyzed."""
        from src.project_watch_mcp.server import create_server
        
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = temp_test_files["java"].parent
            
            # Ensure Java analyzer is registered
            if not AnalyzerRegistry.get_analyzer("java"):
                AnalyzerRegistry.register("java", JavaComplexityAnalyzer)
            
            server = await create_server()
            
            # This should work after implementation
            result = await server.analyze_complexity(
                file_path=str(temp_test_files["java"])
            )
            
            assert result is not None
            assert isinstance(result, ToolResult)
            assert result.structured_content is not None
            assert result.structured_content["language"] == "java"
            assert "summary" in result.structured_content
            assert "functions" in result.structured_content
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_supports_kotlin(self, mock_repo_monitor, temp_test_files):
        """Test that Kotlin files are properly analyzed."""
        from src.project_watch_mcp.server import create_server
        
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = temp_test_files["kotlin"].parent
            
            # Ensure Kotlin analyzer is registered
            if not AnalyzerRegistry.get_analyzer("kotlin"):
                AnalyzerRegistry.register("kotlin", KotlinComplexityAnalyzer)
            
            server = await create_server()
            
            # This should work after implementation
            result = await server.analyze_complexity(
                file_path=str(temp_test_files["kotlin"])
            )
            
            assert result is not None
            assert isinstance(result, ToolResult)
            assert result.structured_content is not None
            assert result.structured_content["language"] == "kotlin"
            assert "summary" in result.structured_content
            assert "functions" in result.structured_content
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_error_on_unsupported_language(self, mock_repo_monitor, tmp_path):
        """Test that unsupported file types raise appropriate errors."""
        from src.project_watch_mcp.server import create_server
        
        # Create an unsupported file type
        rust_file = tmp_path / "test.rs"
        rust_file.write_text("fn main() { println!(\"Hello\"); }")
        
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = tmp_path
            
            server = await create_server()
            
            # Should raise ToolError for unsupported language
            with pytest.raises(ToolError) as exc_info:
                await server.analyze_complexity(file_path=str(rust_file))
            
            assert "not supported" in str(exc_info.value).lower() or "unsupported" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_preserves_backward_compatibility(self, mock_repo_monitor, temp_test_files):
        """Test that existing Python-only functionality remains unchanged."""
        from src.project_watch_mcp.server import create_server
        
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = temp_test_files["python"].parent
            
            server = await create_server()
            
            # The original Python-only interface should still work
            result = await server.analyze_complexity(
                file_path=str(temp_test_files["python"]),
                include_metrics=True
            )
            
            assert result is not None
            assert isinstance(result, ToolResult)
            # Should have all the original fields
            assert "summary" in result.structured_content
            assert "functions" in result.structured_content
            assert "recommendations" in result.structured_content
            
            summary = result.structured_content["summary"]
            assert "total_complexity" in summary
            assert "average_complexity" in summary
            
            # When include_metrics=True, should have MI
            if "maintainability_index" in summary:
                assert "complexity_grade" in summary
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_tool_description_mentions_multi_language(self):
        """Test that the tool description reflects multi-language support."""
        from src.project_watch_mcp.server import create_server
        
        server = await create_server()
        
        # Get the tool definition
        tool_def = None
        for tool in server._tool_manager.list_tools():
            if tool.name == "analyze_complexity":
                tool_def = tool
                break
        
        assert tool_def is not None
        
        # The description should mention multi-language support
        description = tool_def.description.lower()
        assert "python" in description or "java" in description or "kotlin" in description or "multi" in description or "language" in description
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_with_language_override(self, mock_repo_monitor, tmp_path):
        """Test that explicit language parameter overrides file extension detection."""
        # Create a file with misleading extension
        weird_file = tmp_path / "test.txt"
        weird_file.write_text(textwrap.dedent("""
            def python_function():
                return 42
        """))
        
        from src.project_watch_mcp.server import create_server
        
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = tmp_path
            
            server = await create_server()
            
            # Should be able to force Python analysis on .txt file
            result = await server.analyze_complexity(
                file_path=str(weird_file),
                language="python"  # Override extension detection
            )
            
            assert result is not None
            assert result.structured_content["language"] == "python"
    
    @pytest.mark.asyncio
    async def test_registry_initialization_on_server_start(self):
        """Test that all language analyzers are registered when server starts."""
        from src.project_watch_mcp.server import create_server
        
        # Clear registry to test initialization
        AnalyzerRegistry._analyzers.clear()
        AnalyzerRegistry._instances.clear()
        
        server = await create_server()
        
        # After server creation, all analyzers should be registered
        assert AnalyzerRegistry.get_analyzer("python") is not None
        assert AnalyzerRegistry.get_analyzer("java") is not None
        assert AnalyzerRegistry.get_analyzer("kotlin") is not None
        
        # Check supported languages
        supported = AnalyzerRegistry.supported_languages()
        assert "python" in supported
        assert "java" in supported
        assert "kotlin" in supported


class TestAnalyzerRegistryIntegration:
    """Test the AnalyzerRegistry integration with MCP server."""
    
    def test_registry_singleton_pattern(self):
        """Test that AnalyzerRegistry maintains singleton instances."""
        # Get analyzer twice
        analyzer1 = AnalyzerRegistry.get_analyzer("python")
        analyzer2 = AnalyzerRegistry.get_analyzer("python")
        
        # Should be the same instance
        assert analyzer1 is analyzer2
    
    def test_registry_file_extension_mapping(self):
        """Test that file extensions correctly map to analyzers."""
        test_cases = [
            (Path("test.py"), "python"),
            (Path("Test.java"), "java"),
            (Path("Test.kt"), "kotlin"),
            (Path("script.kts"), "kotlin"),
            (Path("test.rs"), None),  # Unsupported
            (Path("test.cpp"), None),  # Unsupported
        ]
        
        for file_path, expected_language in test_cases:
            analyzer = AnalyzerRegistry.get_analyzer_for_file(file_path)
            if expected_language:
                assert analyzer is not None
                assert analyzer.language == expected_language
            else:
                assert analyzer is None
    
    def test_registry_case_insensitive(self):
        """Test that language names are case-insensitive."""
        # These should all return the same analyzer
        assert AnalyzerRegistry.get_analyzer("python") is not None
        assert AnalyzerRegistry.get_analyzer("Python") is not None
        assert AnalyzerRegistry.get_analyzer("PYTHON") is not None
        
        # And they should be the same instance
        assert AnalyzerRegistry.get_analyzer("python") is AnalyzerRegistry.get_analyzer("PYTHON")


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for multi-language support."""
    
    @pytest.mark.asyncio
    async def test_malformed_code_handling(self, mock_repo_monitor, tmp_path):
        """Test that malformed code is handled gracefully."""
        from src.project_watch_mcp.server import create_server
        
        # Create malformed Java file
        bad_java = tmp_path / "Bad.java"
        bad_java.write_text("public class Bad { unclosed method")
        
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = tmp_path
            
            server = await create_server()
            
            # Should handle parse errors gracefully
            with pytest.raises(ToolError) as exc_info:
                await server.analyze_complexity(file_path=str(bad_java))
            
            assert "parse" in str(exc_info.value).lower() or "syntax" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_empty_file_handling(self, mock_repo_monitor, tmp_path):
        """Test that empty files are handled correctly."""
        from src.project_watch_mcp.server import create_server
        
        empty_python = tmp_path / "empty.py"
        empty_python.write_text("")
        
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = tmp_path
            
            server = await create_server()
            
            result = await server.analyze_complexity(file_path=str(empty_python))
            
            assert result is not None
            assert result.structured_content["summary"]["total_complexity"] == 0
            assert result.structured_content["summary"]["function_count"] == 0
    
    @pytest.mark.asyncio
    async def test_large_file_performance(self, mock_repo_monitor, tmp_path):
        """Test that large files are analyzed within reasonable time."""
        from src.project_watch_mcp.server import create_server
        import time
        
        # Create a large Python file
        large_python = tmp_path / "large.py"
        code_lines = []
        for i in range(100):
            code_lines.append(f"""
def function_{i}(x):
    if x > 0:
        return x * 2
    else:
        return x / 2
""")
        large_python.write_text("\n".join(code_lines))
        
        with patch('src.project_watch_mcp.server.repository_monitor', mock_repo_monitor):
            mock_repo_monitor.repo_path = tmp_path
            
            server = await create_server()
            
            start_time = time.time()
            result = await server.analyze_complexity(file_path=str(large_python))
            elapsed_time = time.time() - start_time
            
            assert result is not None
            assert elapsed_time < 5.0  # Should complete within 5 seconds
            assert result.structured_content["summary"]["function_count"] == 100