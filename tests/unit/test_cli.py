"""Comprehensive tests for CLI module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from project_watch_mcp.cli import cli, main


class TestMainFunction:
    """Test main function."""

    @pytest.mark.asyncio
    async def test_main_successful_connection(self):
        """Test main with successful Neo4j connection."""
        with (
            patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class,
            patch("project_watch_mcp.cli.RepositoryMonitor") as mock_monitor_class,
            patch("project_watch_mcp.cli.create_embeddings_provider") as mock_embeddings,
            patch("project_watch_mcp.cli.Neo4jRAG") as mock_rag_class,
            patch("project_watch_mcp.cli.create_mcp_server") as mock_server,
        ):

            # Mock driver
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity.return_value = None
            mock_driver_class.return_value = mock_driver

            # Mock other components
            mock_monitor = AsyncMock()
            mock_monitor_class.return_value = mock_monitor

            mock_rag = AsyncMock()
            mock_rag_class.return_value = mock_rag

            mock_mcp = AsyncMock()
            mock_server.return_value = mock_mcp

            # Run main
            await main(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
                neo4j_database="neo4j",
                repository_path="/test/repo",
                transport="stdio",
            )

            # Verify calls
            mock_driver.verify_connectivity.assert_called_once()
            mock_rag.initialize.assert_called_once()
            mock_mcp.run_stdio_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_connection_failure(self):
        """Test main with Neo4j connection failure."""
        with patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class:
            mock_driver = AsyncMock()
            mock_driver.verify_connectivity.side_effect = Exception("Connection failed")
            mock_driver_class.return_value = mock_driver

            with pytest.raises(SystemExit) as exc_info:
                await main(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password",
                    neo4j_database="neo4j",
                    repository_path="/test/repo",
                )
            assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_main_http_transport(self):
        """Test main with HTTP transport."""
        with (
            patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class,
            patch("project_watch_mcp.cli.RepositoryMonitor") as mock_monitor_class,
            patch("project_watch_mcp.cli.create_embeddings_provider"),
            patch("project_watch_mcp.cli.Neo4jRAG") as mock_rag_class,
            patch("project_watch_mcp.cli.create_mcp_server") as mock_server,
        ):

            mock_driver = AsyncMock()
            mock_driver_class.return_value = mock_driver

            mock_monitor = AsyncMock()
            mock_monitor_class.return_value = mock_monitor

            mock_rag = AsyncMock()
            mock_rag_class.return_value = mock_rag

            mock_mcp = AsyncMock()
            mock_server.return_value = mock_mcp

            await main(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
                neo4j_database="neo4j",
                repository_path="/test/repo",
                transport="http",
                host="0.0.0.0",
                port=8080,
            )

            mock_mcp.run_http_async.assert_called_once_with(
                host="0.0.0.0", port=8080, path="/mcp/", stateless_http=True
            )

    @pytest.mark.asyncio
    async def test_main_sse_transport(self):
        """Test main with SSE transport."""
        with (
            patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class,
            patch("project_watch_mcp.cli.RepositoryMonitor") as mock_monitor_class,
            patch("project_watch_mcp.cli.create_embeddings_provider"),
            patch("project_watch_mcp.cli.Neo4jRAG") as mock_rag_class,
            patch("project_watch_mcp.cli.create_mcp_server") as mock_server,
        ):

            mock_driver = AsyncMock()
            mock_driver_class.return_value = mock_driver

            mock_monitor = AsyncMock()
            mock_monitor_class.return_value = mock_monitor

            mock_rag = AsyncMock()
            mock_rag_class.return_value = mock_rag

            mock_mcp = AsyncMock()
            mock_server.return_value = mock_mcp

            await main(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
                neo4j_database="neo4j",
                repository_path="/test/repo",
                transport="sse",
                host="localhost",
                port=9000,
                path="/sse/",
            )

            mock_mcp.run_sse_async.assert_called_once_with(
                host="localhost", port=9000, path="/sse/"
            )

    @pytest.mark.asyncio
    async def test_main_invalid_transport(self):
        """Test main with invalid transport."""
        with (
            patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class,
            patch("project_watch_mcp.cli.RepositoryMonitor") as mock_monitor_class,
            patch("project_watch_mcp.cli.create_embeddings_provider"),
            patch("project_watch_mcp.cli.Neo4jRAG") as mock_rag_class,
            patch("project_watch_mcp.cli.create_mcp_server") as mock_server,
        ):

            mock_driver = AsyncMock()
            mock_driver_class.return_value = mock_driver

            mock_monitor = AsyncMock()
            mock_monitor_class.return_value = mock_monitor

            mock_rag = AsyncMock()
            mock_rag_class.return_value = mock_rag

            mock_mcp = AsyncMock()
            mock_server.return_value = mock_mcp

            with pytest.raises(ValueError, match="Unsupported transport"):
                await main(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password",
                    neo4j_database="neo4j",
                    repository_path="/test/repo",
                    transport="invalid",  # type: ignore
                )

    @pytest.mark.asyncio
    async def test_main_cleanup_on_error(self):
        """Test cleanup happens on error."""
        with (
            patch("project_watch_mcp.cli.AsyncGraphDatabase.driver") as mock_driver_class,
            patch("project_watch_mcp.cli.RepositoryMonitor") as mock_monitor_class,
            patch("project_watch_mcp.cli.create_embeddings_provider"),
            patch("project_watch_mcp.cli.Neo4jRAG") as mock_rag_class,
            patch("project_watch_mcp.cli.create_mcp_server") as mock_server,
        ):

            mock_driver = AsyncMock()
            mock_driver_class.return_value = mock_driver

            mock_monitor = AsyncMock()
            mock_monitor_class.return_value = mock_monitor

            mock_rag = AsyncMock()
            mock_rag_class.return_value = mock_rag

            mock_mcp = AsyncMock()
            # Make server raise an error
            mock_mcp.run_stdio_async.side_effect = Exception("Server error")
            mock_server.return_value = mock_mcp

            with pytest.raises(Exception, match="Server error"):
                await main(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password",
                    neo4j_database="neo4j",
                    repository_path="/test/repo",
                )

            # Verify cleanup
            mock_monitor.stop.assert_called_once()
            mock_driver.close.assert_called_once()


class TestCLI:
    """Test command-line interface."""

    def test_cli_help(self):
        """Test CLI help output."""
        with patch("sys.argv", ["project-watch-mcp", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cli()
            assert exc_info.value.code == 0

    def test_cli_with_all_args(self):
        """Test CLI with all arguments."""
        with patch(
            "sys.argv",
            [
                "project-watch-mcp",
                "--repository",
                "/test/repo",
                "--neo4j-uri",
                "bolt://custom:7687",
                "--neo4j-user",
                "custom_user",
                "--neo4j-password",
                "custom_pass",
                "--neo4j-database",
                "custom_db",
                "--transport",
                "http",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--path",
                "/custom/",
                "--file-patterns",
                "*.py,*.md",
            ],
        ):
            # Mock path validation and asyncio.run
            with patch("project_watch_mcp.cli.Path") as mock_path:
                # Create a mock path instance
                mock_path_instance = mock_path.return_value
                mock_path_instance.exists.return_value = True
                mock_path_instance.is_dir.return_value = True
                mock_path_instance.absolute.return_value = Path("/test/repo")
                
                # Also handle Path() constructor being called with the result
                mock_path.side_effect = lambda x: mock_path_instance if x == "/test/repo" else Path(x)
                
                with patch("project_watch_mcp.cli.asyncio.run") as mock_run:
                    cli()
                    mock_run.assert_called_once()
                    args = mock_run.call_args[0][0]
                    # Verify the coroutine was called with correct args
                    assert args.cr_code.co_name == "main"

    def test_cli_with_env_vars(self):
        """Test CLI with environment variables."""
        with patch.dict(
            "os.environ",
            {
                "NEO4J_URI": "bolt://env:7687",
                "NEO4J_USER": "env_user",
                "NEO4J_PASSWORD": "env_pass",
                "NEO4J_DATABASE": "env_db",
            },
        ):
            with patch("sys.argv", ["project-watch-mcp", "--repository", "/test/repo"]):
                with patch("project_watch_mcp.cli.Path") as mock_path:
                    mock_repo_path = MagicMock()
                    mock_repo_path.exists.return_value = True
                    mock_repo_path.is_dir.return_value = True
                    mock_path.return_value = mock_repo_path
                    
                    with patch("project_watch_mcp.cli.asyncio.run") as mock_run:
                        cli()
                        mock_run.assert_called_once()

    def test_cli_missing_repository(self):
        """Test CLI without required repository argument."""
        with patch("sys.argv", ["project-watch-mcp"]):
            with pytest.raises(SystemExit):
                cli()

    def test_cli_invalid_transport(self):
        """Test CLI with invalid transport."""
        with patch(
            "sys.argv",
            ["project-watch-mcp", "--repository", "/test/repo", "--transport", "invalid"],
        ):
            with pytest.raises(SystemExit):
                cli()
