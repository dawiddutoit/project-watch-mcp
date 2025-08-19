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
from ..utils.embeddings import create_embeddings_provider
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
        PROJECT_WATCH_USER: str,
        PROJECT_WATCH_PASSWORD: str,
        PROJECT_WATCH_DATABASE: str = "memory",
        repository_path: Path | None = None,
        project_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        """
        Initialize the repository initializer.

        Args:
            neo4j_uri: Neo4j connection URI
            PROJECT_WATCH_USER: Neo4j username
            PROJECT_WATCH_PASSWORD: Neo4j password
            PROJECT_WATCH_DATABASE: Neo4j database name
            repository_path: Path to repository (defaults to current directory)
            project_name: Project name (defaults to repository directory name)
            progress_callback: Optional callback for progress reporting
        """
        self.neo4j_uri = neo4j_uri
        self.PROJECT_WATCH_USER = PROJECT_WATCH_USER
        self.PROJECT_WATCH_PASSWORD = PROJECT_WATCH_PASSWORD
        self.PROJECT_WATCH_DATABASE = PROJECT_WATCH_DATABASE
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
                auth=(self.PROJECT_WATCH_USER, self.PROJECT_WATCH_PASSWORD),
                database=self.PROJECT_WATCH_DATABASE,
            )

            # Verify connectivity
            await asyncio.wait_for(self._driver.verify_connectivity(), timeout=10.0)

            # Create project configuration (not used directly but may be needed later)
            _ = ProjectConfig(
                name=self.project_name,
                repository_path=self.repository_path,
            )

            # Create embeddings provider
            embedding_config = EmbeddingConfig.from_env()
            embeddings_provider = create_embeddings_provider(
                provider_type=embedding_config.provider,
                api_key=embedding_config.api_key,
                model=embedding_config.model,
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

                self._report_progress(5.0, f"Found {total_files} files in repository")

                # Check if repository is already indexed (incremental indexing)
                is_indexed = await self._neo4j_rag.is_repository_indexed(self.project_name)
                
                if is_indexed:
                    self._report_progress(10.0, "Repository already indexed, checking for changes...")
                    
                    # Get indexed files with timestamps
                    indexed_files = await self._neo4j_rag.get_indexed_files(self.project_name)
                    
                    # Detect changes
                    new_files, modified_files, deleted_paths = await self._neo4j_rag.detect_changed_files(
                        files, indexed_files
                    )
                    
                    # Calculate statistics
                    unchanged_count = total_files - len(new_files) - len(modified_files)
                    
                    logger.info(
                        f"Incremental indexing: {len(new_files)} new, "
                        f"{len(modified_files)} modified, {len(deleted_paths)} deleted, "
                        f"{unchanged_count} unchanged"
                    )
                    
                    # Remove deleted files
                    if deleted_paths:
                        self._report_progress(15.0, f"Removing {len(deleted_paths)} deleted files...")
                        await self._neo4j_rag.remove_files(self.project_name, deleted_paths)
                    
                    # Combine files to index (new + modified)
                    files_to_index = new_files + modified_files
                    
                    if not files_to_index:
                        self._report_progress(100.0, "No changes detected")
                        return InitializationResult(
                            indexed=0,
                            total=total_files,
                            message=f"No changes detected. {unchanged_count} files unchanged.",
                        )
                    
                    self._report_progress(20.0, f"Indexing {len(files_to_index)} changed files...")
                    
                else:
                    # Full indexing for new repository
                    self._report_progress(10.0, f"New repository, indexing all {total_files} files...")
                    files_to_index = files
                    unchanged_count = 0

                # Index files that need updating
                indexed_count = 0
                skipped_files = []

                for i, file_info in enumerate(files_to_index):
                    progress = 20.0 + (70.0 * i / len(files_to_index))
                    # Handle potential symlinks in temp directories
                    try:
                        file_path_str = str(file_info.path.relative_to(self.repository_path))
                    except ValueError:
                        # Try with resolved paths for symlinked temp directories
                        file_path_str = str(file_info.path.resolve().relative_to(self.repository_path.resolve()))

                    self._report_progress(
                        progress, f"Indexing {file_path_str} ({i+1}/{len(files_to_index)})"
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

                # Start monitoring only if requested
                monitoring_started = False
                if persistent_monitoring:
                    try:
                        # Use monitoring manager for persistent monitoring
                        manager = MonitoringManager(
                            repo_path=self.repository_path,
                            project_name=self.project_name,
                            neo4j_driver=self._driver,
                        )
                        monitoring_started = await manager.start_persistent_monitoring()
                        if monitoring_started:
                            self._report_progress(95.0, "Persistent monitoring started")
                    except Exception as e:
                        logger.warning(f"Failed to start monitoring: {e}")
                    
                self._report_progress(95.0, "Indexing complete")

                self._report_progress(100.0, "Initialization complete")

                # Prepare result message
                if is_indexed:
                    # Incremental indexing message
                    if indexed_count == 0:
                        message = f"Repository up to date. {unchanged_count} files unchanged."
                    else:
                        new_count = len(new_files) if 'new_files' in locals() else 0
                        modified_count = len(modified_files) if 'modified_files' in locals() else 0
                        deleted_count = len(deleted_paths) if 'deleted_paths' in locals() else 0
                        
                        message = (
                            f"Incremental update complete. "
                            f"Indexed {indexed_count} files "
                            f"({new_count} new, {modified_count} modified), "
                            f"{unchanged_count} unchanged."
                        )
                        if deleted_count > 0:
                            message += f" Removed {deleted_count} deleted files."
                else:
                    # Full indexing message
                    if indexed_count == total_files:
                        message = f"Repository initialized. Indexed {indexed_count}/{total_files} files."
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
