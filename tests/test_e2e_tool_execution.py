"""End-to-end tests for MCP tool execution.

These tests execute tools through the complete MCP framework stack
without heavy mocking, ensuring that the entire pipeline works correctly.
This would have caught the 67% tool failure rate before production.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.tools.tool import ToolResult

from src.project_watch_mcp.neo4j_rag import Neo4jRAG, SearchResult
from src.project_watch_mcp.repository_monitor import FileInfo, RepositoryMonitor
from src.project_watch_mcp.server import create_mcp_server


class TestE2EToolExecution:
    """End-to-end tests that execute tools through the complete MCP stack."""

    @pytest.fixture
    async def test_repository(self):
        """Create a real test repository with actual files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create test files
            (repo_path / "main.py").write_text("""
def main():
    '''Main entry point for the application.'''
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
""")
            
            (repo_path / "utils.py").write_text("""
def calculate_sum(a: int, b: int) -> int:
    '''Calculate the sum of two numbers.'''
    return a + b

def format_output(value: str) -> str:
    '''Format output string.'''
    return f"Output: {value}"
""")
            
            src_dir = repo_path / "src"
            src_dir.mkdir()
            (src_dir / "module.py").write_text("""
class DataProcessor:
    '''Process data for the application.'''
    
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        '''Add an item to the processor.'''
        self.data.append(item)
""")
            
            # Create .gitignore
            (repo_path / ".gitignore").write_text("*.pyc\n__pycache__/\n.venv/\n")
            
            yield repo_path

    @pytest.fixture
    async def mcp_server_with_minimal_mocks(self, test_repository):
        """Create MCP server with minimal mocking - only external dependencies."""
        # Mock only the Neo4j driver, not our own code
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_driver.close = AsyncMock()
        
        # Storage for indexed data
        indexed_files = {}
        indexed_chunks = []
        
        async def mock_execute_query(query, params=None, *args, **kwargs):
            """Minimal mock that simulates Neo4j behavior."""
            project_name = params.get("project_name", "test") if params else "test"
            
            if "MERGE (f:CodeFile" in query:
                # Store file indexing
                file_path = params.get("path", "")
                indexed_files[file_path] = params
                return MagicMock(records=[])
            
            elif "MERGE (c:CodeChunk" in query:
                # Store chunk indexing
                indexed_chunks.append(params)
                return MagicMock(records=[])
            
            elif "MATCH (c:CodeChunk" in query and "similarity" in query.lower():
                # Return search results
                return MagicMock(records=[
                    {
                        "file_path": str(test_repository / "main.py"),
                        "chunk_content": "def main():",
                        "line_number": 2,
                        "similarity": 0.95,
                    },
                    {
                        "file_path": str(test_repository / "utils.py"),
                        "chunk_content": "def calculate_sum(a: int, b: int)",
                        "line_number": 2,
                        "similarity": 0.89,
                    }
                ])
            
            elif "MATCH (c:CodeChunk" in query and "~" in query:
                # Pattern search
                return MagicMock(records=[
                    {
                        "file_path": str(test_repository / "main.py"),
                        "chunk_content": "def main():",
                        "line_number": 2,
                    }
                ])
            
            elif "MATCH (f:CodeFile" in query and "file_path" in str(params):
                # Get file metadata
                file_path = params.get("file_path", "")
                if file_path in indexed_files:
                    return MagicMock(records=[indexed_files[file_path]])
                return MagicMock(records=[])
            
            elif "count(DISTINCT f)" in query:
                # Get repository stats
                return MagicMock(records=[{
                    "total_files": len(indexed_files),
                    "total_chunks": len(indexed_chunks),
                    "total_size": sum(f.get("size", 0) for f in indexed_files.values()),
                    "languages": ["python"],
                }])
            
            return MagicMock(records=[])
        
        mock_driver.execute_query = mock_execute_query
        
        # Create real instances with mocked driver
        monitor = RepositoryMonitor(
            repo_path=test_repository,
            project_name="test_project",
            neo4j_driver=mock_driver,
            file_patterns=["*.py"],
            ignore_patterns=["*.pyc", "__pycache__", ".venv"],
        )
        
        # Use mock embeddings provider
        from src.project_watch_mcp.utils.embedding import MockEmbeddingsProvider
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=MockEmbeddingsProvider(),
        )
        await rag.initialize()
        
        # Create server
        server = create_mcp_server(monitor, rag, "test_project")
        
        return server, mock_driver, indexed_files, indexed_chunks

    async def test_initialize_repository_e2e(self, mcp_server_with_minimal_mocks):
        """Test initialize_repository through complete MCP stack."""
        server, mock_driver, indexed_files, indexed_chunks = mcp_server_with_minimal_mocks
        
        # Find the tool
        init_tool = await server.get_tool("initialize_repository")
        assert init_tool is not None, "initialize_repository tool not found"
        
        # Execute through MCP framework
        result = await init_tool.run({})
        
        # CRITICAL VALIDATIONS THAT WOULD HAVE CAUGHT THE BUG
        # 1. Must return ToolResult
        assert isinstance(result, ToolResult), \
            f"Expected ToolResult, got {type(result).__name__}"
        
        # 2. Must have valid structure
        assert result.content is not None
        assert isinstance(result.content, list)
        assert len(result.content) > 0
        
        # 3. Must have proper structured_content
        assert result.structured_content is not None
        assert isinstance(result.structured_content, dict)
        assert "indexed" in result.structured_content
        assert "total" in result.structured_content
        
        # 4. Verify files were actually indexed
        assert len(indexed_files) > 0, "No files were indexed"
        assert any("main.py" in path for path in indexed_files.keys()), \
            "main.py should have been indexed"

    async def test_search_code_semantic_e2e(self, mcp_server_with_minimal_mocks):
        """Test semantic search through complete MCP stack."""
        server, mock_driver, indexed_files, indexed_chunks = mcp_server_with_minimal_mocks
        
        # Initialize repository first
        init_tool = await server.get_tool("initialize_repository")
        await init_tool.run({})
        
        # Find search tool
        search_tool = await server.get_tool("search_code")
        assert search_tool is not None, "search_code tool not found"
        
        # Execute semantic search
        result = await search_tool.run({
            "query": "calculate sum of numbers",
            "search_type": "semantic",
            "limit": 5
        })
        
        # CRITICAL VALIDATIONS THAT WOULD HAVE CAUGHT THE BUG
        # 1. Must return ToolResult, not list
        assert isinstance(result, ToolResult), \
            f"Expected ToolResult, got {type(result).__name__}"
        
        # 2. structured_content must NOT be a list (MCP protocol)
        assert not isinstance(result.structured_content, list), \
            "structured_content cannot be a list - MCP protocol violation!"
        
        # 3. If returning results, must be wrapped in dict
        assert isinstance(result.structured_content, dict) or result.structured_content is None, \
            "structured_content must be dict or None"
        
        # 4. Results should be properly structured
        if result.structured_content:
            # THIS IS THE EXACT CHECK THAT WOULD HAVE CAUGHT THE BUG
            # The original code returned a list directly, this ensures it's wrapped
            assert not isinstance(result.structured_content, list), \
                "Results must be wrapped in a dict, not returned as bare list"
            
            # If there are results, they should be under a key
            if "results" in result.structured_content:
                assert isinstance(result.structured_content["results"], list)

    async def test_search_code_pattern_e2e(self, mcp_server_with_minimal_mocks):
        """Test pattern search through complete MCP stack."""
        server, mock_driver, indexed_files, indexed_chunks = mcp_server_with_minimal_mocks
        
        # Initialize repository first
        init_tool = await server.get_tool("initialize_repository")
        await init_tool.run({})
        
        # Find search tool
        search_tool = await server.get_tool("search_code")
        
        # Execute pattern search
        result = await search_tool.run({
            "query": "def.*main",
            "search_type": "pattern",
            "is_regex": True,
            "limit": 3
        })
        
        # Validate MCP compliance
        assert isinstance(result, ToolResult)
        assert not isinstance(result.structured_content, list), \
            "Pattern search also must not return bare list"

    async def test_refresh_file_e2e(self, mcp_server_with_minimal_mocks, test_repository):
        """Test refresh_file through complete MCP stack."""
        server, mock_driver, indexed_files, indexed_chunks = mcp_server_with_minimal_mocks
        
        # Initialize repository first
        init_tool = await server.get_tool("initialize_repository")
        await init_tool.run({})
        
        # Find refresh tool
        refresh_tool = await server.get_tool("refresh_file")
        assert refresh_tool is not None, "refresh_file tool not found"
        
        # Execute refresh
        result = await refresh_tool.run({"file_path": "main.py"})
        
        # CRITICAL VALIDATION THAT WOULD HAVE CAUGHT THE BUG
        assert isinstance(result, ToolResult), \
            f"refresh_file must return ToolResult, got {type(result).__name__}"
        
        # Validate structure
        assert result.structured_content is not None
        assert isinstance(result.structured_content, dict)
        assert "status" in result.structured_content

    async def test_all_tools_e2e_execution(self, mcp_server_with_minimal_mocks, test_repository):
        """Execute ALL tools through MCP stack to ensure they work."""
        server, mock_driver, indexed_files, indexed_chunks = mcp_server_with_minimal_mocks
        
        # Initialize first
        init_tool = await server.get_tool("initialize_repository")
        init_result = await init_tool.run({})
        assert isinstance(init_result, ToolResult), "initialize_repository failed"
        
        # Test each tool
        tool_tests = [
            ("search_code", {"query": "test", "search_type": "semantic"}),
            ("get_repository_stats", {}),
            ("get_file_info", {"file_path": "main.py"}),
            ("refresh_file", {"file_path": "main.py"}),
            ("monitoring_status", {}),
        ]
        
        for tool_name, params in tool_tests:
            tool = await server.get_tool(tool_name)
            assert tool is not None, f"{tool_name} not found"
            
            # Execute tool
            result = await tool.run(params)
            
            # EVERY TOOL MUST RETURN ToolResult
            assert isinstance(result, ToolResult), \
                f"{tool_name} must return ToolResult, got {type(result).__name__}"
            
            # EVERY structured_content must be dict or None
            assert result.structured_content is None or isinstance(result.structured_content, dict), \
                f"{tool_name}: structured_content must be dict or None, got {type(result.structured_content).__name__}"
            
            # NEVER return bare list
            assert not isinstance(result.structured_content, list), \
                f"{tool_name}: structured_content cannot be bare list (MCP violation)"


class TestMCPFrameworkIntegration:
    """Test that tools work correctly when called through MCP framework methods."""
    
    @pytest.fixture
    async def test_repository(self):
        """Create a real test repository with actual files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create test files
            (repo_path / "main.py").write_text("""
def main():
    '''Main entry point for the application.'''
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
""")
            
            (repo_path / "utils.py").write_text("""
def calculate_sum(a: int, b: int) -> int:
    '''Calculate the sum of two numbers.'''
    return a + b

def format_output(value: str) -> str:
    '''Format output string.'''
    return f"Output: {value}"
""")
            
            src_dir = repo_path / "src"
            src_dir.mkdir()
            (src_dir / "module.py").write_text("""
class DataProcessor:
    '''Process data for the application.'''
    
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        '''Add an item to the processor.'''
        self.data.append(item)
""")
            
            # Create .gitignore
            (repo_path / ".gitignore").write_text("*.pyc\n__pycache__/\n.venv/\n")
            
            yield repo_path

    @pytest.fixture
    async def mcp_server_with_minimal_mocks(self, test_repository):
        """Create MCP server with minimal mocking - only external dependencies."""
        # Mock only the Neo4j driver, not our own code
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_driver.close = AsyncMock()
        
        # Storage for indexed data
        indexed_files = {}
        indexed_chunks = []
        
        async def mock_execute_query(query, params=None, *args, **kwargs):
            """Minimal mock that simulates Neo4j behavior."""
            project_name = params.get("project_name", "test") if params else "test"
            
            if "MERGE (f:CodeFile" in query:
                # Store file indexing
                file_path = params.get("path", "")
                indexed_files[file_path] = params
                return MagicMock(records=[])
            
            elif "MERGE (c:CodeChunk" in query:
                # Store chunk indexing
                indexed_chunks.append(params)
                return MagicMock(records=[])
            
            elif "MATCH (c:CodeChunk" in query and "similarity" in query.lower():
                # Return search results
                return MagicMock(records=[
                    {
                        "file_path": str(test_repository / "main.py"),
                        "chunk_content": "def main():",
                        "line_number": 2,
                        "similarity": 0.95,
                    },
                    {
                        "file_path": str(test_repository / "utils.py"),
                        "chunk_content": "def calculate_sum(a: int, b: int)",
                        "line_number": 2,
                        "similarity": 0.89,
                    }
                ])
            
            elif "MATCH (c:CodeChunk" in query and "~" in query:
                # Pattern search
                return MagicMock(records=[
                    {
                        "file_path": str(test_repository / "main.py"),
                        "chunk_content": "def main():",
                        "line_number": 2,
                    }
                ])
            
            elif "MATCH (f:CodeFile" in query and "file_path" in str(params):
                # Get file metadata
                file_path = params.get("file_path", "")
                if file_path in indexed_files:
                    return MagicMock(records=[indexed_files[file_path]])
                return MagicMock(records=[])
            
            elif "count(DISTINCT f)" in query:
                # Get repository stats
                return MagicMock(records=[{
                    "total_files": len(indexed_files),
                    "total_chunks": len(indexed_chunks),
                    "total_size": sum(f.get("size", 0) for f in indexed_files.values()),
                    "languages": ["python"],
                }])
            
            return MagicMock(records=[])
        
        mock_driver.execute_query = mock_execute_query
        
        # Create real instances with mocked driver
        monitor = RepositoryMonitor(
            repo_path=test_repository,
            project_name="test_project",
            neo4j_driver=mock_driver,
            file_patterns=["*.py"],
            ignore_patterns=["*.pyc", "__pycache__", ".venv"],
        )
        
        # Use mock embeddings provider
        from src.project_watch_mcp.utils.embedding import MockEmbeddingsProvider
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=MockEmbeddingsProvider(),
        )
        await rag.initialize()
        
        # Create server
        server = create_mcp_server(monitor, rag, "test_project")
        
        return server, mock_driver, indexed_files, indexed_chunks
    
    async def test_tool_execution_through_framework(self, mcp_server_with_minimal_mocks):
        """Test executing tools through MCP framework's execute method."""
        server, mock_driver, indexed_files, indexed_chunks = mcp_server_with_minimal_mocks
        
        # This simulates how the MCP client would actually call tools
        # If we had access to server.execute_tool or similar method
        
        # Get tool and validate it has proper metadata
        init_tool = await server.get_tool("initialize_repository")
        
        # Validate tool registration
        assert init_tool is not None
        # FunctionTool has a run method, not directly callable
        assert hasattr(init_tool, 'run') and callable(init_tool.run)
        
        # Execute and validate
        result = await init_tool.run({})
        
        # This is what MCP framework expects
        assert isinstance(result, ToolResult), "MCP framework expects ToolResult"
        
        # Simulate MCP framework serialization
        # This would fail if structured_content is a bare list
        if result.structured_content is not None:
            try:
                # MCP tries to serialize structured_content
                import json
                serialized = json.dumps(result.structured_content)
                assert serialized is not None
            except TypeError as e:
                pytest.fail(f"MCP framework cannot serialize result: {e}")


class TestRealWorldScenarios:
    """Test real-world usage scenarios that would expose the bugs."""
    
    @pytest.fixture
    async def test_repository(self):
        """Create a real test repository with actual files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create test files
            (repo_path / "main.py").write_text("""
def main():
    '''Main entry point for the application.'''
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
""")
            
            (repo_path / "utils.py").write_text("""
def calculate_sum(a: int, b: int) -> int:
    '''Calculate the sum of two numbers.'''
    return a + b

def format_output(value: str) -> str:
    '''Format output string.'''
    return f"Output: {value}"
""")
            
            src_dir = repo_path / "src"
            src_dir.mkdir()
            (src_dir / "module.py").write_text("""
class DataProcessor:
    '''Process data for the application.'''
    
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        '''Add an item to the processor.'''
        self.data.append(item)
""")
            
            # Create .gitignore
            (repo_path / ".gitignore").write_text("*.pyc\n__pycache__/\n.venv/\n")
            
            yield repo_path

    @pytest.fixture
    async def mcp_server_with_minimal_mocks(self, test_repository):
        """Create MCP server with minimal mocking - only external dependencies."""
        # Mock only the Neo4j driver, not our own code
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()
        mock_driver.close = AsyncMock()
        
        # Storage for indexed data
        indexed_files = {}
        indexed_chunks = []
        
        async def mock_execute_query(query, params=None, *args, **kwargs):
            """Minimal mock that simulates Neo4j behavior."""
            project_name = params.get("project_name", "test") if params else "test"
            
            if "MERGE (f:CodeFile" in query:
                # Store file indexing
                file_path = params.get("path", "")
                indexed_files[file_path] = params
                return MagicMock(records=[])
            
            elif "MERGE (c:CodeChunk" in query:
                # Store chunk indexing
                indexed_chunks.append(params)
                return MagicMock(records=[])
            
            elif "MATCH (c:CodeChunk" in query and "similarity" in query.lower():
                # Return search results
                return MagicMock(records=[
                    {
                        "file_path": str(test_repository / "main.py"),
                        "chunk_content": "def main():",
                        "line_number": 2,
                        "similarity": 0.95,
                    },
                    {
                        "file_path": str(test_repository / "utils.py"),
                        "chunk_content": "def calculate_sum(a: int, b: int)",
                        "line_number": 2,
                        "similarity": 0.89,
                    }
                ])
            
            elif "MATCH (c:CodeChunk" in query and "~" in query:
                # Pattern search
                return MagicMock(records=[
                    {
                        "file_path": str(test_repository / "main.py"),
                        "chunk_content": "def main():",
                        "line_number": 2,
                    }
                ])
            
            elif "MATCH (f:CodeFile" in query and "file_path" in str(params):
                # Get file metadata
                file_path = params.get("file_path", "")
                if file_path in indexed_files:
                    return MagicMock(records=[indexed_files[file_path]])
                return MagicMock(records=[])
            
            elif "count(DISTINCT f)" in query:
                # Get repository stats
                return MagicMock(records=[{
                    "total_files": len(indexed_files),
                    "total_chunks": len(indexed_chunks),
                    "total_size": sum(f.get("size", 0) for f in indexed_files.values()),
                    "languages": ["python"],
                }])
            
            return MagicMock(records=[])
        
        mock_driver.execute_query = mock_execute_query
        
        # Create real instances with mocked driver
        monitor = RepositoryMonitor(
            repo_path=test_repository,
            project_name="test_project",
            neo4j_driver=mock_driver,
            file_patterns=["*.py"],
            ignore_patterns=["*.pyc", "__pycache__", ".venv"],
        )
        
        # Use mock embeddings provider
        from src.project_watch_mcp.utils.embedding import MockEmbeddingsProvider
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            embeddings=MockEmbeddingsProvider(),
        )
        await rag.initialize()
        
        # Create server
        server = create_mcp_server(monitor, rag, "test_project")
        
        return server, mock_driver, indexed_files, indexed_chunks
    
    async def test_search_and_retrieve_workflow(self, mcp_server_with_minimal_mocks, test_repository):
        """Test a complete search and retrieve workflow."""
        server, mock_driver, indexed_files, indexed_chunks = mcp_server_with_minimal_mocks
        
        # 1. Initialize repository
        init_tool = await server.get_tool("initialize_repository")
        init_result = await init_tool.run({})
        assert isinstance(init_result, ToolResult)
        
        # 2. Search for code
        search_tool = await server.get_tool("search_code")
        search_result = await search_tool.run({"query": "calculate", "search_type": "semantic"})
        
        # THIS WOULD HAVE CAUGHT THE BUG
        assert isinstance(search_result, ToolResult)
        assert not isinstance(search_result.structured_content, list), \
            "Search results must be wrapped, not bare list"
        
        # 3. Get file info for first result
        if search_result.structured_content and isinstance(search_result.structured_content, dict):
            # Extract file from results (however they're structured)
            results_key = "results" if "results" in search_result.structured_content else None
            if results_key and search_result.structured_content[results_key]:
                first_result = search_result.structured_content[results_key][0]
                file_path = first_result.get("file", "").replace(str(test_repository) + "/", "")
                
                if file_path:
                    info_tool = await server.get_tool("get_file_info")
                    info_result = await info_tool.run({"file_path": file_path})
                    assert isinstance(info_result, ToolResult)
        
        # 4. Check monitoring status
        status_tool = await server.get_tool("monitoring_status")
        status_result = await status_tool.run({})
        assert isinstance(status_result, ToolResult)
        
        # All tools in workflow must return ToolResult
        assert all(isinstance(r, ToolResult) for r in [init_result, search_result, status_result])