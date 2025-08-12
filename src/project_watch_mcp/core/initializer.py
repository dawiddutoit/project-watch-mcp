"""Core repository initialization logic shared between CLI, server, and hooks."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from ..config import EmbeddingConfig, ProjectConfig
from ..neo4j_rag import CodeFile, Neo4jRAG
from ..repository_monitor import RepositoryMonitor
from ..utils.embedding import create_embeddings_provider
from .monitoring_manager import MonitoringManager

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


class InitializationError(Exception):
    """Base class for initialization errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "INIT_ERROR",
        technical_details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.user_message = message
        self.technical_details = technical_details or {}


class ConnectionError(InitializationError):
    """Neo4j connection issues."""

    def __init__(self, message: str, technical_details: dict[str, Any] | None = None):
        super().__init__(message, "NEO4J_CONN_ERROR", technical_details)


class FileAccessError(InitializationError):
    """Repository file access issues."""

    def __init__(self, message: str, technical_details: dict[str, Any] | None = None):
        super().__init__(message, "FILE_ACCESS_ERROR", technical_details)


class IndexingError(InitializationError):
    """Neo4j indexing failures."""

    def __init__(self, message: str, technical_details: dict[str, Any] | None = None):
        super().__init__(message, "INDEXING_ERROR", technical_details)


@dataclass
class InitializationResult:
    """Result of repository initialization."""

    indexed: int
    total: int
    skipped: list[str] = field(default_factory=list)
    monitoring: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "indexed": self.indexed,
            "total": self.total,
            "skipped": self.skipped,
            "monitoring": self.monitoring,
            "message": self.message,
        }


class RepositoryInitializer:
    """Handles repository initialization logic for all interfaces."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = "memory",
        repository_path: Path | None = None,
        project_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        """
        Initialize the repository initializer.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            repository_path: Path to repository (defaults to current directory)
            project_name: Project name (defaults to repository directory name)
            progress_callback: Optional callback for progress reporting
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.repository_path = repository_path or Path.cwd()
        self.project_name = project_name or self.repository_path.name
        self.progress_callback = progress_callback

        self._driver: AsyncDriver | None = None
        self._neo4j_rag: Neo4jRAG | None = None
        self._repository_monitor: RepositoryMonitor | None = None
        self._initialization_lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry."""
        await self._setup_connections()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_connections()

    async def _setup_connections(self):
        """Set up Neo4j driver and related components."""
        try:
            # Create Neo4j driver
            self._driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                database=self.neo4j_database,
            )

            # Verify connectivity
            await asyncio.wait_for(self._driver.verify_connectivity(), timeout=10.0)

            # Create project configuration (not used directly but may be needed later)
            _ = ProjectConfig(
                name=self.project_name,
                repository_path=self.repository_path,
            )

            # Create embeddings provider
            embedding_config = EmbeddingConfig()
            embeddings_provider = create_embeddings_provider(
                provider_type=embedding_config.provider,
                dimension=embedding_config.dimension,
            )

            # Initialize Neo4j RAG
            self._neo4j_rag = Neo4jRAG(
                neo4j_driver=self._driver,
                project_name=self.project_name,
                embeddings=embeddings_provider,
            )

            # Initialize repository monitor
            self._repository_monitor = RepositoryMonitor(
                repo_path=self.repository_path,
                project_name=self.project_name,
                neo4j_driver=self._driver,
            )

        except TimeoutError as e:
            raise ConnectionError(
                "Neo4j connection timeout", {"uri": self.neo4j_uri, "timeout": 10}
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Neo4j: {e}", {"uri": self.neo4j_uri, "error": str(e)}
            ) from e

    async def _cleanup_connections(self):
        """Clean up connections and resources."""
        if self._driver:
            try:
                await self._driver.close()
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {e}")

        self._driver = None
        self._neo4j_rag = None
        self._repository_monitor = None

    def _report_progress(self, percentage: float, message: str):
        """Report progress if callback is provided."""
        if self.progress_callback:
            try:
                self.progress_callback(percentage, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    async def initialize(self, persistent_monitoring: bool = False) -> InitializationResult:
        """
        Initialize the repository with monitoring and indexing.

        Args:
            persistent_monitoring: If True, keep monitoring running after initialization

        Returns:
            InitializationResult with details about the initialization

        Raises:
            InitializationError: When initialization fails
        """
        async with self._initialization_lock:
            if not self._neo4j_rag or not self._repository_monitor:
                await self._setup_connections()

            try:
                self._report_progress(0.0, "Scanning repository...")

                # Scan repository for files
                files = await self._repository_monitor.scan_repository()
                total_files = len(files)

                if total_files == 0:
                    self._report_progress(100.0, "No files to index")
                    return InitializationResult(
                        indexed=0,
                        total=0,
                        message="No files found matching patterns",
                    )

                self._report_progress(10.0, f"Found {total_files} files to index")

                # Index each file
                indexed_count = 0
                skipped_files = []

                for i, file_info in enumerate(files):
                    progress = 10.0 + (80.0 * i / total_files)
                    # Handle potential symlinks in temp directories
                    try:
                        file_path_str = str(file_info.path.relative_to(self.repository_path))
                    except ValueError:
                        # Try with resolved paths for symlinked temp directories
                        file_path_str = str(file_info.path.resolve().relative_to(self.repository_path.resolve()))

                    self._report_progress(
                        progress, f"Indexing {file_path_str} ({i+1}/{total_files})"
                    )

                    try:
                        # Read file content
                        content = file_info.path.read_text(encoding="utf-8")

                        # Create CodeFile object
                        code_file = CodeFile(
                            project_name=self.project_name,
                            path=file_info.path,
                            content=content,
                            language=file_info.language,
                            size=file_info.size,
                            last_modified=file_info.last_modified,
                        )

                        # Index in Neo4j
                        await self._neo4j_rag.index_file(code_file)
                        indexed_count += 1

                    except UnicodeDecodeError:
                        logger.debug(f"Skipping binary file: {file_path_str}")
                        skipped_files.append(file_path_str)
                    except Exception as e:
                        logger.error(f"Failed to index {file_path_str}: {e}")
                        skipped_files.append(file_path_str)

                self._report_progress(90.0, "Starting file monitoring...")

                # Start monitoring
                monitoring_started = False
                try:
                    if persistent_monitoring:
                        # Use monitoring manager for persistent monitoring
                        manager = MonitoringManager(
                            repo_path=self.repository_path,
                            project_name=self.project_name,
                            neo4j_driver=self._driver,
                        )
                        monitoring_started = await manager.start_persistent_monitoring()
                        if monitoring_started:
                            self._report_progress(95.0, "Persistent monitoring started")
                    else:
                        # Normal monitoring (stops when process exits)
                        await self._repository_monitor.start()
                        monitoring_started = True
                        self._report_progress(95.0, "Monitoring started")
                except Exception as e:
                    logger.warning(f"Failed to start monitoring: {e}")

                self._report_progress(100.0, "Initialization complete")

                # Prepare result message
                if indexed_count == total_files:
                    message = (
                        f"Repository initialized. Indexed {indexed_count}/{total_files} files."
                    )
                else:
                    message = f"Repository initialized. Indexed {indexed_count}/{total_files} files ({len(skipped_files)} skipped)."

                return InitializationResult(
                    indexed=indexed_count,
                    total=total_files,
                    skipped=skipped_files,
                    monitoring=monitoring_started,
                    message=message,
                )

            except FileAccessError:
                raise
            except IndexingError:
                raise
            except Exception as e:
                raise InitializationError(
                    f"Initialization failed: {e}", technical_details={"error": str(e)}
                ) from e
