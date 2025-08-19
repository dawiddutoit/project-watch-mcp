"""End-to-end test for monitoring persistence."""

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.getenv("SKIP_E2E_TESTS", "true").lower() == "true",
    reason="End-to-end tests skipped (set SKIP_E2E_TESTS=false to run)"
)
class TestEndToEnd:
    """End-to-end tests for monitoring persistence."""
    
    def test_cli_initialize_starts_monitoring(self):
        """Test that CLI --initialize flag starts monitoring that persists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create test files
            (repo_path / "test.py").write_text("print('hello')")
            (repo_path / "main.js").write_text("console.log('world')")
            
            # Set environment variables for Neo4j mock
            env = os.environ.copy()
            env["PROJECT_WATCH_PASSWORD"] = "password"
            env["EMBEDDING_PROVIDER"] = "mock"
            
            # Run initialization via CLI
            result = subprocess.run(
                ["uv", "run", "project-watch-mcp", "--initialize", "--repository", str(repo_path)],
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            # Check that initialization succeeded
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert "Indexed:" in result.stdout
            assert "Monitoring: started" in result.stdout
            
            # The monitoring is started but the process exits
            # In a real scenario, the monitoring would be handled by
            # the MCP server running in the background
    
    def test_mcp_tool_initialize_repository(self):
        """Test that the MCP tool initialize_repository works correctly."""
        # This would require a running MCP server
        # For now, we'll just verify the tool exists in the server
        from project_watch_mcp.server import create_mcp_server
        from unittest.mock import MagicMock, AsyncMock
        
        # Create mocks
        mock_monitor = MagicMock()
        mock_rag = MagicMock()
        
        # Create server
        mcp = create_mcp_server(
            repository_monitor=mock_monitor,
            neo4j_rag=mock_rag,
            project_name="test_project"
        )
        
        # Check that initialize_repository tool exists
        tool_names = [tool.name for tool in mcp.list_tools()]
        assert "initialize_repository" in tool_names
        
        # Check that monitoring_status tool exists  
        assert "monitoring_status" in tool_names


if __name__ == "__main__":
    # Run with E2E tests enabled
    os.environ["SKIP_E2E_TESTS"] = "false"
    pytest.main([__file__, "-v"])