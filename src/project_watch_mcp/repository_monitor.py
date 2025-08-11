"""Repository monitoring module using watchfiles."""

import asyncio
import fnmatch
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from neo4j import AsyncDriver
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from watchfiles import Change, awatch

logger = logging.getLogger(__name__)


class FileChangeType(Enum):
    """Types of file changes."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class FileInfo:
    """Information about a file in the repository."""

    path: Path
    size: int
    last_modified: datetime
    language: str | None = None

    def __post_init__(self):
        """Detect language from file extension."""
        if self.language is None:
            self.language = self._detect_language()

    def _detect_language(self) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "css",
            ".sql": "sql",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "bash",
            ".dockerfile": "dockerfile",
            ".dockerignore": "dockerfile",
        }
        return extension_map.get(self.path.suffix.lower(), "text")


@dataclass
class FileChangeEvent:
    """Represents a file change event."""

    path: Path
    change_type: FileChangeType
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RepositoryMonitor:
    """Monitors a repository for file changes using watchfiles."""

    def __init__(
        self,
        repo_path: Path,
        project_name: str,
        neo4j_driver: AsyncDriver,
        file_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        use_gitignore: bool = True,
    ):
        """
        Initialize the repository monitor.

        Args:
            repo_path: Path to the repository to monitor
            project_name: Name of the project for context isolation
            neo4j_driver: Neo4j driver for database connection
            file_patterns: List of file patterns to include (e.g., ["*.py", "*.js"])
            ignore_patterns: List of patterns to ignore (in addition to .gitignore)
            use_gitignore: Whether to use the project's .gitignore file
        """
        self.repo_path = Path(repo_path).resolve()
        self.project_name = project_name
        self.neo4j_driver = neo4j_driver
        self.file_patterns = file_patterns or ["*"]
        self.use_gitignore = use_gitignore

        # Default ignore patterns (used if no .gitignore or as fallback)
        default_ignore_patterns = [
            ".*",  # Hidden files
            "__pycache__",
            "*.pyc",
            "node_modules",
            ".git",
            ".venv",
            "venv",
            "env",
            "build",
            "dist",
            "*.egg-info",
        ]

        # Load gitignore patterns if available
        self.gitignore_spec = self._load_gitignore() if use_gitignore else None

        # Combine user-provided patterns with defaults if no gitignore
        if not self.gitignore_spec:
            self.ignore_patterns = ignore_patterns or default_ignore_patterns
        else:
            # If gitignore is loaded, only use additional user patterns
            self.ignore_patterns = ignore_patterns or []

        self.is_running = False
        self._watch_task: asyncio.Task | None = None
        self._change_queue: asyncio.Queue[FileChangeEvent] = asyncio.Queue()

    def _load_gitignore(self) -> PathSpec | None:
        """Load patterns from .gitignore file if it exists."""
        gitignore_path = self.repo_path / ".gitignore"

        if not gitignore_path.exists():
            logger.debug(f"No .gitignore file found at {gitignore_path}")
            return None

        try:
            with gitignore_path.open("r", encoding="utf-8") as f:
                patterns = f.read().splitlines()

            # Filter out empty lines and comments
            patterns = [
                line.strip()
                for line in patterns
                if line.strip() and not line.strip().startswith("#")
            ]

            if not patterns:
                logger.debug(".gitignore file is empty or contains only comments")
                return None

            # Create PathSpec with GitWildMatchPattern for proper gitignore behavior
            spec = PathSpec.from_lines(GitWildMatchPattern, patterns)
            logger.info(f"Loaded {len(patterns)} patterns from .gitignore")
            return spec

        except Exception as e:
            logger.warning(f"Failed to load .gitignore: {e}")
            return None

    def reload_gitignore(self) -> None:
        """Reload .gitignore patterns (useful if .gitignore has changed)."""
        if self.use_gitignore:
            self.gitignore_spec = self._load_gitignore()
            logger.info("Reloaded .gitignore patterns")

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if a file should be included based on patterns."""
        # Get relative path from repo root for gitignore matching
        try:
            relative_path = file_path.relative_to(self.repo_path)
        except ValueError:
            # File is outside repo path
            return False

        # Check gitignore patterns first if available
        if self.gitignore_spec:
            # PathSpec.match_file expects a string path relative to the gitignore location
            if self.gitignore_spec.match_file(str(relative_path)):
                return False

        # Check additional ignore patterns
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return False
            if any(fnmatch.fnmatch(part, pattern) for part in file_path.parts):
                return False

        # Check include patterns
        for pattern in self.file_patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return True

        return False

    async def scan_repository(self) -> list[FileInfo]:
        """
        Scan the repository and return information about all matching files.

        Returns:
            List of FileInfo objects for matching files
        """
        files = []

        for file_path in self.repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            if not self._should_include_file(file_path):
                continue

            try:
                stat = file_path.stat()
                files.append(
                    FileInfo(
                        path=file_path,
                        size=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime),
                    )
                )
            except OSError as e:
                logger.warning(f"Could not stat file {file_path}: {e}")

        logger.info(f"Scanned repository: found {len(files)} matching files")
        return files

    async def start(self):
        """Start monitoring the repository for changes."""
        if self.is_running:
            logger.warning("Monitor is already running")
            return

        self.is_running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info(f"Started monitoring repository: {self.repo_path}")

    async def stop(self):
        """Stop monitoring the repository."""
        if not self.is_running:
            return

        self.is_running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None

        logger.info("Stopped monitoring repository")

    async def _watch_loop(self):
        """Main watch loop using watchfiles."""
        try:
            async for changes in awatch(
                self.repo_path,
                stop_event=asyncio.Event() if not self.is_running else None,
            ):
                for change_type, path_str in changes:
                    path = Path(path_str)

                    if not self._should_include_file(path):
                        continue

                    # Map watchfiles change types to our types
                    if change_type == Change.added:
                        event_type = FileChangeType.CREATED
                    elif change_type == Change.modified:
                        event_type = FileChangeType.MODIFIED
                    elif change_type == Change.deleted:
                        event_type = FileChangeType.DELETED
                    else:
                        continue

                    event = FileChangeEvent(
                        path=path,
                        change_type=event_type,
                    )

                    await self._change_queue.put(event)
                    logger.debug(f"Detected {event_type.value}: {path}")

                if not self.is_running:
                    break

        except asyncio.CancelledError:
            logger.debug("Watch loop cancelled")
        except Exception as e:
            logger.error(f"Error in watch loop: {e}")
            self.is_running = False

    async def get_next_change(self, timeout: float | None = None) -> FileChangeEvent | None:
        """
        Get the next file change event.

        Args:
            timeout: Maximum time to wait for a change (in seconds)

        Returns:
            FileChangeEvent or None if timeout
        """
        try:
            if timeout:
                return await asyncio.wait_for(self._change_queue.get(), timeout=timeout)
            else:
                return await self._change_queue.get()
        except TimeoutError:
            return None

    def has_pending_changes(self) -> bool:
        """Check if there are pending changes in the queue."""
        return not self._change_queue.empty()

    async def process_all_changes(self) -> list[FileChangeEvent]:
        """Process all pending changes and return them."""
        changes = []
        while not self._change_queue.empty():
            try:
                change = self._change_queue.get_nowait()
                changes.append(change)
            except asyncio.QueueEmpty:
                break
        return changes
