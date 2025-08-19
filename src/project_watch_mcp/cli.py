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
from .neo4j_rag import CodeFile, Neo4jRAG
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
    PROJECT_WATCH_USER: str,
    PROJECT_WATCH_PASSWORD: str,
    PROJECT_WATCH_DATABASE: str,
    repository_path: str,
    project_name: str | None = None,
    verbose: bool = False,
) -> int:
    """
    Initialize repository without starting the server.

    Args:
        neo4j_uri: Neo4j connection URI
        PROJECT_WATCH_USER: Neo4j username
        PROJECT_WATCH_PASSWORD: Neo4j password
        PROJECT_WATCH_DATABASE: Neo4j database name
        repository_path: Path to repository to monitor
        project_name: Optional project name for context isolation
        verbose: Enable verbose progress reporting

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    initializer = None
    try:
        repo_path = Path(repository_path)
        if not project_name:
            project_name = repo_path.name

        # Log initialization start with masked credentials
        logger.info(f"Starting repository initialization for project: {project_name}")
        logger.info(f"Repository path: {repo_path}")
        logger.info(f"Neo4j URI: {neo4j_uri}")
        logger.info(f"Neo4j user: {PROJECT_WATCH_USER}")
        logger.info(f"Neo4j database: {PROJECT_WATCH_DATABASE}")
        # Mask password in logs
        if PROJECT_WATCH_PASSWORD:
            masked_pwd = f"{PROJECT_WATCH_PASSWORD[:2]}{'*' * (len(PROJECT_WATCH_PASSWORD) - 4)}{PROJECT_WATCH_PASSWORD[-2:]}" if len(PROJECT_WATCH_PASSWORD) > 4 else "*" * len(PROJECT_WATCH_PASSWORD)
            logger.debug(f"Neo4j password: {masked_pwd}")

        # Progress callback for verbose mode
        def progress_callback(percentage: float, message: str):
            if verbose:
                print(f"[{percentage:3.0f}%] {message}", file=sys.stderr)

        # Create initializer
        initializer = RepositoryInitializer(
            neo4j_uri=neo4j_uri,
            PROJECT_WATCH_USER=PROJECT_WATCH_USER,
            PROJECT_WATCH_PASSWORD=PROJECT_WATCH_PASSWORD,
            PROJECT_WATCH_DATABASE=PROJECT_WATCH_DATABASE,
            repository_path=repo_path,
            project_name=project_name,
            progress_callback=progress_callback if verbose else None,
        )

        # Run initialization without monitoring for --initialize mode
        # This mode is just for one-time indexing, not for continuous monitoring
        # Monitoring will be handled by the MCP server when it runs normally
        async with initializer:
            result = await initializer.initialize(persistent_monitoring=False)

        # Print clean, simple result
        print(f"Project: {project_name}")
        print(f"Indexed: {result.indexed}/{result.total} files")
        if result.skipped:
            print(f"Skipped: {len(result.skipped)} files")
        print(f"Status: Initialization complete")

        return 0

    except KeyboardInterrupt:
        logger.info("Initialization interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        # Ensure proper cleanup even if something goes wrong
        if initializer:
            try:
                await initializer._cleanup_connections()
            except Exception:
                pass


async def main(
    neo4j_uri: str,
    PROJECT_WATCH_USER: str,
    PROJECT_WATCH_PASSWORD: str,
    PROJECT_WATCH_DATABASE: str,
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
        PROJECT_WATCH_USER: Neo4j username
        PROJECT_WATCH_PASSWORD: Neo4j password
        PROJECT_WATCH_DATABASE: Neo4j database name
        repository_path: Path to repository to monitor
        project_name: Optional project name for context isolation
        transport: Transport type (stdio, sse, http)
        host: HTTP/SSE server host
        port: HTTP/SSE server port
        path: HTTP/SSE server path
        file_patterns: Comma-separated file patterns to monitor
    """
    # Log server startup with configuration
    logger.info("=" * 60)
    logger.info("Starting Project Watch MCP Server")
    logger.info("=" * 60)
    logger.info(f"Repository: {repository_path}")
    logger.info(f"Project name: {project_name or 'auto-detect'}")
    logger.info(f"Transport: {transport}")
    if transport in ["http", "sse"]:
        logger.info(f"Server endpoint: {transport}://{host}:{port}{path}")

    # Log Neo4j configuration with masked password
    logger.info("Neo4j Configuration:")
    logger.info(f"  URI: {neo4j_uri}")
    logger.info(f"  User: {PROJECT_WATCH_USER}")
    logger.info(f"  Database: {PROJECT_WATCH_DATABASE}")
    if PROJECT_WATCH_PASSWORD:
        masked_pwd = f"{PROJECT_WATCH_PASSWORD[:2]}{'*' * (len(PROJECT_WATCH_PASSWORD) - 4)}{PROJECT_WATCH_PASSWORD[-2:]}" if len(PROJECT_WATCH_PASSWORD) > 4 else "*" * len(PROJECT_WATCH_PASSWORD)
        logger.debug(f"  Password: {masked_pwd}")

    logger.info(f"File patterns: {file_patterns}")

    # Parse file patterns
    patterns = [p.strip() for p in file_patterns.split(",")]

    # Connect to Neo4j
    logger.info("Initializing Neo4j connection...")
    neo4j_driver = AsyncGraphDatabase.driver(
        neo4j_uri, auth=(PROJECT_WATCH_USER, PROJECT_WATCH_PASSWORD), database=PROJECT_WATCH_DATABASE
    )

    # Verify connection
    try:
        await neo4j_driver.verify_connectivity()
        logger.info("✓ Successfully connected to Neo4j")
        logger.debug(f"  Neo4j endpoint: {neo4j_uri}")
        logger.debug(f"  Database: {PROJECT_WATCH_DATABASE}")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Neo4j: {e}")
        logger.error("Make sure Neo4j is running and accessible")
        logger.error("You can start Neo4j with Neo4j Desktop")
        exit(1)

    # Create project configuration
    logger.info("Configuring project...")
    if project_name:
        project_config = ProjectConfig(name=project_name, repository_path=Path(repository_path))
    else:
        project_config = ProjectConfig.from_repository_path(Path(repository_path))
    logger.info(f"✓ Project configured: {project_config.name}")
    logger.debug(f"  Repository path: {repository_path}")

    # Create repository monitor with project context
    logger.info("Initializing repository monitor...")
    repository_monitor = RepositoryMonitor(
        repo_path=Path(repository_path),
        project_name=project_config.name,
        neo4j_driver=neo4j_driver,
        file_patterns=patterns,
    )
    logger.info("✓ Repository monitor created")
    logger.debug(f"  Monitoring patterns: {patterns}")

    # Create embeddings configuration from environment
    logger.info("Configuring embeddings...")
    embedding_config = EmbeddingConfig.from_env()

    # Create embeddings provider
    if embedding_config.provider == "disabled":
        embeddings = None
        logger.warning("⚠ Embeddings disabled - no API key configured")
        logger.warning("  Semantic search features will not be available")
    else:
        logger.info(f"  Provider: {embedding_config.provider}")
        logger.info(f"  Model: {embedding_config.model}")
        logger.debug(f"  Dimension: {embedding_config.dimension}")

        # Mask API key in logs
        if embedding_config.api_key:
            masked_key = f"{embedding_config.api_key[:7]}...{embedding_config.api_key[-4:]}" if len(embedding_config.api_key) > 11 else "*" * len(embedding_config.api_key)
            logger.debug(f"  API Key: {masked_key}")

        embeddings = create_embeddings_provider(
            provider_type=embedding_config.provider,
            api_key=embedding_config.api_key,
            model=embedding_config.model,
            dimension=embedding_config.dimension,
        )

        if embeddings is None:
            logger.warning("⚠ Failed to create embeddings provider - semantic search disabled")
        else:
            logger.info(f"✓ Embeddings provider initialized: {embedding_config.provider}")

    # Create Neo4j RAG system with project context
    logger.info("Initializing Neo4j RAG system...")
    neo4j_rag = Neo4jRAG(
        neo4j_driver=neo4j_driver,
        project_name=project_config.name,
        embeddings=embeddings,
        chunk_size=100,  # Lines per chunk
        chunk_overlap=20,  # Overlapping lines
    )
    await neo4j_rag.initialize()
    logger.info("✓ Neo4j RAG system initialized")
    logger.debug("  Chunk size: 100 lines")
    logger.debug("  Chunk overlap: 20 lines")

    # Initialize repository (index files) before starting monitor
    # TODO: Implement incremental indexing - only index new/changed files
    # For now, this re-indexes everything on server start which is inefficient
    # Future: Check Neo4j for existing index and only update changed files
    logger.info("Initializing repository index...")
    try:
        files = await repository_monitor.scan_repository()
        indexed_count = 0
        
        for file_info in files:
            try:
                # Check if file needs indexing (new or modified)
                content = file_info.path.read_text(encoding="utf-8")
                code_file = CodeFile(
                    project_name=project_config.name,
                    path=file_info.path,
                    content=content,
                    language=file_info.language,
                    size=file_info.size,
                    last_modified=file_info.last_modified,
                )
                await neo4j_rag.index_file(code_file)
                indexed_count += 1
            except UnicodeDecodeError:
                logger.debug(f"Skipping binary file: {file_info.path}")
            except Exception as e:
                logger.warning(f"Failed to index {file_info.path}: {e}")
        
        logger.info(f"✓ Indexed {indexed_count} files")
    except Exception as e:
        logger.warning(f"Failed to initialize repository index: {e}")
        logger.warning("Continuing without initial index - use initialize_repository tool to index manually")
    
    # Start the repository monitor to enable file watching
    logger.info("Starting repository monitor...")
    await repository_monitor.start(daemon=True)
    logger.info("✓ Repository monitor started in background")

    # Create MCP server with project context
    logger.info("Creating MCP server...")
    mcp = create_mcp_server(
        repository_monitor=repository_monitor,
        neo4j_rag=neo4j_rag,
        project_name=project_config.name,
    )
    logger.info(f"✓ MCP server created for project: {project_config.name}")

    # Run the server
    logger.info("=" * 60)
    logger.info(f"Starting MCP server with {transport} transport...")
    logger.info("=" * 60)

    try:
        match transport:
            case "http":
                logger.info(f"🚀 HTTP server starting on http://{host}:{port}{path}")
                await mcp.run_http_async(host=host, port=port, path=path, stateless_http=True)
            case "stdio":
                logger.info("🚀 STDIO server starting - ready for connections")
                await mcp.run_stdio_async()
            case "sse":
                logger.info(f"🚀 SSE server starting on http://{host}:{port}{path}")
                await mcp.run_sse_async(host=host, port=port, path=path)
            case _:
                raise ValueError(f"Unsupported transport: {transport}")
    finally:
        # Cleanup
        logger.info("Shutting down server...")
        await repository_monitor.stop()
        logger.info("✓ Repository monitor stopped")
        await neo4j_driver.close()
        logger.info("✓ Neo4j connection closed")
        logger.info("Server shutdown complete")


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
  export PROJECT_WATCH_PASSWORD=mypassword
  project-watch-mcp --repository /path/to/repo

Neo4j Quick Start:
  Option 1 - Neo4j Desktop (recommended): Download from https://neo4j.com/download/
  Option 2 - Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

Embedding Provider Configuration:
  Set EMBEDDING_PROVIDER environment variable to one of: voyage, openai

  For OpenAI:
    export EMBEDDING_PROVIDER=openai
    export OPENAI_API_KEY=your-api-key
    export OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # optional
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
    PROJECT_WATCH_USER = args.neo4j_user or os.getenv("PROJECT_WATCH_USER", "neo4j")
    PROJECT_WATCH_PASSWORD = args.neo4j_password or os.getenv("PROJECT_WATCH_PASSWORD", "password")
    PROJECT_WATCH_DATABASE = args.neo4j_database or os.getenv("PROJECT_WATCH_DATABASE", "neo4j")

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
                    PROJECT_WATCH_USER=PROJECT_WATCH_USER,
                    PROJECT_WATCH_PASSWORD=PROJECT_WATCH_PASSWORD,
                    PROJECT_WATCH_DATABASE=PROJECT_WATCH_DATABASE,
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
                    PROJECT_WATCH_USER=PROJECT_WATCH_USER,
                    PROJECT_WATCH_PASSWORD=PROJECT_WATCH_PASSWORD,
                    PROJECT_WATCH_DATABASE=PROJECT_WATCH_DATABASE,
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
