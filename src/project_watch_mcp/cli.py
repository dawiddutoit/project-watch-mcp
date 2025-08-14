"""Command-line interface for Project Watch."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Literal

from neo4j import AsyncGraphDatabase

from .config import EmbeddingConfig, ProjectConfig
from .core import RepositoryInitializer
from .neo4j_rag import Neo4jRAG
from .repository_monitor import RepositoryMonitor
from .server import create_mcp_server
from .utils.embeddings import create_embeddings_provider

# Configure logging - default to WARNING to reduce verbosity
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def initialize_only(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    repository_path: str,
    project_name: str | None = None,
    verbose: bool = False,
) -> int:
    """
    Initialize repository without starting the server.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        repository_path: Path to repository to monitor
        project_name: Optional project name for context isolation
        verbose: Enable verbose progress reporting

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    try:
        repo_path = Path(repository_path)
        if not project_name:
            project_name = repo_path.name

        # Progress callback for verbose mode
        def progress_callback(percentage: float, message: str):
            if verbose:
                print(f"[{percentage:3.0f}%] {message}", file=sys.stderr)

        # Create initializer
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            repository_path=repo_path,
            project_name=project_name,
            progress_callback=progress_callback if verbose else None,
        )

        # Run initialization with persistent monitoring
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=True)

        # Print clean, simple result
        print(f"Project: {project_name}")
        print(f"Indexed: {result.indexed}/{result.total} files")
        if result.skipped:
            print(f"Skipped: {len(result.skipped)} files")
        print(f"Monitoring: {'started' if result.monitoring else 'not started'}")

        return 0

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


async def main(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    repository_path: str,
    project_name: str | None = None,
    transport: Literal["stdio", "sse", "http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
    file_patterns: str = "*.py,*.js,*.ts,*.java,*.go,*.rs,*.md,*.json,*.yaml,*.yml,*.toml",
) -> None:
    """
    Main entry point for the Project Watch MCP server.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        repository_path: Path to repository to monitor
        project_name: Optional project name for context isolation
        transport: Transport type (stdio, sse, http)
        host: HTTP/SSE server host
        port: HTTP/SSE server port
        path: HTTP/SSE server path
        file_patterns: Comma-separated file patterns to monitor
    """
    logger.debug("Starting Project Watch MCP Server")
    logger.debug(f"Repository: {repository_path}")
    logger.debug(f"Neo4j URI: {neo4j_uri}")

    # Parse file patterns
    patterns = [p.strip() for p in file_patterns.split(",")]

    # Connect to Neo4j
    neo4j_driver = AsyncGraphDatabase.driver(
        neo4j_uri, auth=(neo4j_user, neo4j_password), database=neo4j_database
    )

    # Verify connection
    try:
        await neo4j_driver.verify_connectivity()
        logger.debug(f"Connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        logger.error("Make sure Neo4j is running and accessible")
        logger.error(
            "You can start Neo4j with Neo4j Desktop (recommended) or Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j"
        )
        exit(1)

    # Create project configuration
    if project_name:
        project_config = ProjectConfig(name=project_name, repository_path=Path(repository_path))
    else:
        project_config = ProjectConfig.from_repository_path(Path(repository_path))
    logger.debug(f"Project name: {project_config.name}")

    # Create repository monitor with project context
    repository_monitor = RepositoryMonitor(
        repo_path=Path(repository_path),
        project_name=project_config.name,
        neo4j_driver=neo4j_driver,
        file_patterns=patterns,
    )
    logger.debug("Repository monitor created")

    # Create embeddings configuration from environment
    embedding_config = EmbeddingConfig.from_env()

    # Create embeddings provider
    if embedding_config.provider == "disabled":
        embeddings = None
        logger.warning("Embeddings disabled - no API key configured")
        logger.warning("Semantic search features will not be available")
    else:
        embeddings = create_embeddings_provider(
            provider_type=embedding_config.provider,
            api_key=embedding_config.api_key,
            model=embedding_config.model,
            dimension=embedding_config.dimension,
        )
        
        if embeddings is None:
            logger.warning("Failed to create embeddings provider - semantic search disabled")
        else:
            logger.debug(f"Using {embedding_config.provider} embeddings provider")

    # Create Neo4j RAG system with project context
    neo4j_rag = Neo4jRAG(
        neo4j_driver=neo4j_driver,
        project_name=project_config.name,
        embeddings=embeddings,
        chunk_size=100,  # Lines per chunk
        chunk_overlap=20,  # Overlapping lines
    )
    await neo4j_rag.initialize()
    logger.debug("Neo4j RAG system initialized")

    # Start the repository monitor to enable file watching
    await repository_monitor.start(daemon=True)
    logger.debug("Repository monitor started")
    
    # Create MCP server with project context
    mcp = create_mcp_server(
        repository_monitor=repository_monitor,
        neo4j_rag=neo4j_rag,
        project_name=project_config.name,
    )
    logger.debug("MCP server created")

    # Run the server
    logger.debug(f"Starting server with transport: {transport}")

    try:
        match transport:
            case "http":
                logger.debug(f"HTTP server starting on {host}:{port}{path}")
                await mcp.run_http_async(host=host, port=port, path=path, stateless_http=True)
            case "stdio":
                logger.debug("STDIO server starting")
                await mcp.run_stdio_async()
            case "sse":
                logger.debug(f"SSE server starting on {host}:{port}{path}")
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
        prog="project-watch-mcp",
        description="Project Watch - Repository Monitoring MCP Server with Neo4j RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (requires Neo4j running locally)
  project-watch-mcp --repository /path/to/repo

  # Run with custom Neo4j connection
  project-watch-mcp --repository /path/to/repo --neo4j-uri bolt://myserver:7687 --neo4j-password mypassword

  # Run as HTTP server
  project-watch-mcp --repository /path/to/repo --transport http --port 8080

  # Monitor specific file types only
  project-watch-mcp --repository /path/to/repo --file-patterns "*.py,*.js,*.ts"

  # Using environment variables
  export NEO4J_URI=bolt://localhost:7687
  export NEO4J_PASSWORD=mypassword
  project-watch-mcp --repository /path/to/repo

Neo4j Quick Start:
  Option 1 - Neo4j Desktop (recommended): Download from https://neo4j.com/download/
  Option 2 - Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

Embedding Provider Configuration:
  Set EMBEDDING_PROVIDER environment variable to one of: openai, local, mock

  For OpenAI:
    export EMBEDDING_PROVIDER=openai
    export OPENAI_API_KEY=your-api-key
    export OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # optional

  For Local API:
    export EMBEDDING_PROVIDER=local
    export LOCAL_EMBEDDING_API_URL=http://localhost:8080/embeddings

  For Mock (default):
    export EMBEDDING_PROVIDER=mock
        """,
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
        "--project-name",
        default=None,
        help="Project name for context isolation (default: generated from repository path)",
    )
    parser.add_argument(
        "--file-patterns",
        "-p",
        default=None,
        help="Comma-separated file patterns to monitor (default: common code files)",
    )

    # Initialization mode
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Initialize repository and exit (mutually exclusive with --transport)",
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

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("project_watch_mcp").setLevel(logging.DEBUG)
    else:
        # Keep default WARNING level for non-verbose mode
        logging.getLogger().setLevel(logging.WARNING)

    # Check for mutually exclusive options
    if args.initialize and args.transport:
        parser.error("--initialize and --transport are mutually exclusive")

    # Get configuration from environment variables or defaults
    neo4j_uri = args.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = args.neo4j_user or os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = args.neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database = args.neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

    # For initialization mode, use current directory if no repository specified
    if args.initialize:
        repository_path = args.repository or os.getenv("REPOSITORY_PATH", ".")
    else:
        repository_path = args.repository or os.getenv("REPOSITORY_PATH")
        if not repository_path:
            parser.error("--repository is required (or set REPOSITORY_PATH environment variable)")

    project_name = args.project_name or os.getenv("PROJECT_NAME")

    # Validate repository path
    repo_path = Path(repository_path)
    if not repo_path.exists():
        parser.error(f"Repository path does not exist: {repository_path}")
    if not repo_path.is_dir():
        parser.error(f"Repository path is not a directory: {repository_path}")

    file_patterns = args.file_patterns or os.getenv(
        "FILE_PATTERNS", "*.py,*.js,*.ts,*.java,*.go,*.rs,*.md,*.json,*.yaml,*.yml,*.toml"
    )

    transport = args.transport or os.getenv("MCP_TRANSPORT", "stdio")
    host = args.host or os.getenv("MCP_HOST", "127.0.0.1")
    port = args.port or int(os.getenv("MCP_PORT", "8000"))
    path = args.path or os.getenv("MCP_PATH", "/mcp/")

    # Run in the appropriate mode
    try:
        if args.initialize:
            # Run initialization only
            exit_code = asyncio.run(
                initialize_only(
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    neo4j_database=neo4j_database,
                    repository_path=str(repo_path.absolute()),
                    project_name=project_name,
                    verbose=args.verbose,
                )
            )
            sys.exit(exit_code)
        else:
            # Run the server
            asyncio.run(
                main(
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    neo4j_database=neo4j_database,
                    repository_path=str(repo_path.absolute()),
                    project_name=project_name,
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
