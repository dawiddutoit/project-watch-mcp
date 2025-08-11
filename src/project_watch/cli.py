"""Command-line interface for Project Watch."""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Literal

from neo4j import AsyncGraphDatabase

from .neo4j_rag import Neo4jRAG
from .repository_monitor import RepositoryMonitor
from .server import create_mcp_server

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    repository_path: str,
    transport: Literal["stdio", "sse", "http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
    file_patterns: str = "*.py,*.js,*.ts,*.java,*.go,*.rs,*.md,*.json,*.yaml,*.yml",
) -> None:
    """
    Main entry point for the Project Watch MCP server.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        repository_path: Path to repository to monitor
        transport: Transport type (stdio, sse, http)
        host: HTTP/SSE server host
        port: HTTP/SSE server port
        path: HTTP/SSE server path
        file_patterns: Comma-separated file patterns to monitor
    """
    logger.info("Starting Project Watch MCP Server")
    logger.info(f"Repository: {repository_path}")
    logger.info(f"Neo4j URI: {neo4j_uri}")

    # Parse file patterns
    patterns = [p.strip() for p in file_patterns.split(",")]

    # Connect to Neo4j
    neo4j_driver = AsyncGraphDatabase.driver(
        neo4j_uri, auth=(neo4j_user, neo4j_password), database=neo4j_database
    )

    # Verify connection
    try:
        await neo4j_driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        logger.error("Make sure Neo4j is running and accessible")
        logger.error("You can start Neo4j with Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j")
        exit(1)

    # Create repository monitor
    repository_monitor = RepositoryMonitor(
        repo_path=Path(repository_path),
        neo4j_driver=neo4j_driver,
        file_patterns=patterns,
    )
    logger.info("Repository monitor created")

    # Create Neo4j RAG system
    neo4j_rag = Neo4jRAG(
        neo4j_driver=neo4j_driver,
        chunk_size=100,  # Lines per chunk
        chunk_overlap=20,  # Overlapping lines
    )
    await neo4j_rag.initialize()
    logger.info("Neo4j RAG system initialized")

    # Create MCP server
    mcp = create_mcp_server(
        repository_monitor=repository_monitor,
        neo4j_rag=neo4j_rag,
    )
    logger.info("MCP server created")

    # Run the server
    logger.info(f"Starting server with transport: {transport}")

    try:
        match transport:
            case "http":
                logger.info(f"HTTP server starting on {host}:{port}{path}")
                await mcp.run_http_async(host=host, port=port, path=path, stateless_http=True)
            case "stdio":
                logger.info("STDIO server starting")
                await mcp.run_stdio_async()
            case "sse":
                logger.info(f"SSE server starting on {host}:{port}{path}")
                await mcp.run_sse_async(host=host, port=port, path=path)
            case _:
                raise ValueError(f"Unsupported transport: {transport}")
    finally:
        # Cleanup
        await repository_monitor.stop()
        await neo4j_driver.close()


def cli():
    """Command-line interface for Project Watch."""
    parser = argparse.ArgumentParser(
        prog="project-watch",
        description="Project Watch - Repository Monitoring MCP Server with Neo4j RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (requires Neo4j running locally)
  project-watch --repository /path/to/repo
  
  # Run with custom Neo4j connection
  project-watch --repository /path/to/repo --neo4j-uri bolt://myserver:7687 --neo4j-password mypassword
  
  # Run as HTTP server
  project-watch --repository /path/to/repo --transport http --port 8080
  
  # Monitor specific file types only
  project-watch --repository /path/to/repo --file-patterns "*.py,*.js,*.ts"
  
  # Using environment variables
  export NEO4J_URI=bolt://localhost:7687
  export NEO4J_PASSWORD=mypassword
  project-watch --repository /path/to/repo

Docker Neo4j Quick Start:
  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
        """
    )

    # Neo4j connection arguments
    parser.add_argument(
        "--neo4j-uri",
        default=None,
        help="Neo4j connection URI (default: bolt://localhost:7687)",
    )
    parser.add_argument(
        "--neo4j-user",
        default=None,
        help="Neo4j username (default: neo4j)",
    )
    parser.add_argument(
        "--neo4j-password",
        default=None,
        help="Neo4j password (default: password)",
    )
    parser.add_argument(
        "--neo4j-database",
        default=None,
        help="Neo4j database name (default: neo4j)",
    )

    # Repository arguments
    parser.add_argument(
        "--repository",
        "-r",
        default=None,
        help="Path to repository to monitor (required)",
    )
    parser.add_argument(
        "--file-patterns",
        "-p",
        default=None,
        help="Comma-separated file patterns to monitor (default: common code files)",
    )

    # Server arguments
    parser.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "http", "sse"],
        default=None,
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="HTTP/SSE server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP/SSE server port (default: 8000)",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="HTTP/SSE server path (default: /mcp/)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get configuration from environment variables or defaults
    neo4j_uri = args.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = args.neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database = args.neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

    repository_path = args.repository or os.getenv("REPOSITORY_PATH")
    if not repository_path:
        parser.error("--repository is required (or set REPOSITORY_PATH environment variable)")

    # Validate repository path
    repo_path = Path(repository_path)
    if not repo_path.exists():
        parser.error(f"Repository path does not exist: {repository_path}")
    if not repo_path.is_dir():
        parser.error(f"Repository path is not a directory: {repository_path}")

    file_patterns = args.file_patterns or os.getenv(
        "FILE_PATTERNS", "*.py,*.js,*.ts,*.java,*.go,*.rs,*.md,*.json,*.yaml,*.yml"
    )

    transport = args.transport or os.getenv("MCP_TRANSPORT", "stdio")
    host = args.host or os.getenv("MCP_HOST", "127.0.0.1")
    port = args.port or int(os.getenv("MCP_PORT", "8000"))
    path = args.path or os.getenv("MCP_PATH", "/mcp/")

    # Run the server
    try:
        asyncio.run(
            main(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                neo4j_database=neo4j_database,
                repository_path=str(repo_path.absolute()),
                transport=transport,
                host=host,
                port=port,
                path=path,
                file_patterns=file_patterns,
            )
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        exit(1)


if __name__ == "__main__":
    cli()