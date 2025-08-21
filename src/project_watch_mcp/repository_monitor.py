"""Repository monitoring module using watchfiles."""

import asyncio
import fnmatch
import logging
import time
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
            ".toml": "toml",
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


class DebouncedChangeProcessor:
    """Processes file changes with debouncing to reduce redundant operations."""

    def __init__(self, debounce_delay: float = 0.5):
        """
        Initialize the debounced change processor.
        
        Args:
            debounce_delay: Time in seconds to wait before processing a change (default: 0.5)
        """
        self.debounce_delay = debounce_delay
        self._pending_changes: dict[Path, FileChangeEvent] = {}
        self._last_process_time: dict[Path, float] = {}
        self._lock = asyncio.Lock()

    async def add_change(self, event: FileChangeEvent) -> bool:
        """
        Add a change event, debouncing if necessary.
        
        Args:
            event: The file change event to process
            
        Returns:
            True if the change should be processed immediately, False if debounced
        """
        async with self._lock:
            path = event.path
            current_time = time.time()

            # Check if we've processed this file recently
            last_time = self._last_process_time.get(path, 0)
            time_since_last = current_time - last_time

            if time_since_last < self.debounce_delay:
                # Too soon since last process, debounce this change
                self._pending_changes[path] = event
                logger.debug(f"Debouncing change for {path} (last processed {time_since_last:.2f}s ago)")
                return False

            # Enough time has passed, process this change
            self._last_process_time[path] = current_time
            # Remove from pending if it was there
            self._pending_changes.pop(path, None)
            return True

    async def get_pending_changes(self) -> list[FileChangeEvent]:
        """
        Get all pending changes that have been debounced long enough.
        
        Returns:
            List of file change events ready to be processed
        """
        async with self._lock:
            current_time = time.time()
            ready_changes = []

            for path, event in list(self._pending_changes.items()):
                last_time = self._last_process_time.get(path, 0)
                time_since_last = current_time - last_time

                if time_since_last >= self.debounce_delay:
                    # This change is ready to be processed
                    ready_changes.append(event)
                    self._last_process_time[path] = current_time
                    del self._pending_changes[path]

            return ready_changes

    def clear_history(self, older_than: float = 60.0):
        """
        Clear old processing history to prevent memory growth.
        
        Args:
            older_than: Clear entries older than this many seconds (default: 60)
        """
        current_time = time.time()
        cutoff_time = current_time - older_than

        # Remove old entries from last_process_time
        self._last_process_time = {
            path: timestamp
            for path, timestamp in self._last_process_time.items()
            if timestamp > cutoff_time
        }

    @property
    def pending_count(self) -> int:
        """Get the number of pending changes."""
        return len(self._pending_changes)


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
        debounce_delay: float = 0.5,
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
            debounce_delay: Time in seconds to debounce file changes (default: 0.5)
        """
        self.repo_path = Path(repo_path).resolve()
        self.project_name = project_name
        self.neo4j_driver = neo4j_driver
        self.file_patterns = file_patterns or ["*"]
        self.use_gitignore = use_gitignore

        # Initialize debouncing processor
        self.debounce_processor = DebouncedChangeProcessor(debounce_delay)

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
            # Lock files and large generated files
            "*-lock.lock",
            "package-lock.json",
            "*.lock",
            "yarn.lock",
            "poetry.lock",
            "Pipfile.lock",
            "uv.lock",
        ]

        # Load gitignore patterns if available
        self.gitignore_spec = self._load_gitignore() if use_gitignore else None

        # Always include default ignore patterns for lock files and other generated files
        # These are critical patterns that should always be excluded
        # Combine with user-provided patterns if any, ensuring uniqueness
        combined_patterns = set(default_ignore_patterns)
        if ignore_patterns:
            combined_patterns.update(ignore_patterns)
        self.ignore_patterns = list(combined_patterns)

        self.is_running = False
        self._watch_task: asyncio.Task | None = None
        self._change_queue: asyncio.Queue[FileChangeEvent] = asyncio.Queue()
        self.monitoring_since: datetime | None = None

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
                if file_path.name == "uv.lock":
                    logger.debug(f"Excluding {file_path.name} due to pattern: {pattern}")
                return False
            if any(fnmatch.fnmatch(part, pattern) for part in file_path.parts):
                if file_path.name == "uv.lock":
                    logger.debug(f"Excluding {file_path.name} due to path part pattern: {pattern}")
                return False

        # Check include patterns
        for pattern in self.file_patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return True

        return False

    def _should_skip_directory(self, dir_path: Path) -> bool:
        """Check if a directory should be skipped entirely."""
        # Always skip these directories
        skip_dirs = {'.git', '.idea', '__pycache__', 'node_modules', '.venv', 'venv', 'env', '.ruff_cache', '.mypy_cache', '.pytest_cache'}

        # Check if any parent or the directory itself is in skip list
        for part in dir_path.parts:
            if part in skip_dirs:
                return True

        # Check gitignore patterns for directories
        if self.gitignore_spec:
            try:
                relative_path = dir_path.relative_to(self.repo_path)
                # Add trailing slash for directory matching in gitignore
                if self.gitignore_spec.match_file(str(relative_path) + '/'):
                    return True
            except ValueError:
                pass

        return False

    async def scan_repository(self) -> list[FileInfo]:
        """
        Scan the repository and return information about all matching files.

        Returns:
            List of FileInfo objects for matching files
        """
        files = []

        # Use a stack for iterative directory traversal to avoid recursing into ignored dirs
        dirs_to_scan = [self.repo_path]

        while dirs_to_scan:
            current_dir = dirs_to_scan.pop()

            try:
                for item in current_dir.iterdir():
                    if item.is_dir():
                        # Skip directories that should be ignored
                        if not self._should_skip_directory(item):
                            dirs_to_scan.append(item)
                    elif item.is_file():
                        # Check if file should be included
                        if self._should_include_file(item):
                            try:
                                stat = item.stat()
                                files.append(
                                    FileInfo(
                                        path=item,
                                        size=stat.st_size,
                                        last_modified=datetime.fromtimestamp(stat.st_mtime),
                                    )
                                )
                            except OSError as e:
                                logger.warning(f"Could not stat file {item}: {e}")
            except PermissionError as e:
                logger.debug(f"Permission denied accessing {current_dir}: {e}")
                continue

        logger.info(f"Scanned repository: found {len(files)} matching files")
        return files

    async def start(self, daemon: bool = False):
        """
        Start monitoring the repository for changes.
        
        Args:
            daemon: If True, the monitoring task will continue running 
                   even after the main process exits (not fully daemonized,
                   but allows the task to persist in the event loop)
        """
        if self.is_running:
            logger.warning("Monitor is already running")
            return

        self.is_running = True
        self.monitoring_since = datetime.now()
        self._watch_task = asyncio.create_task(self._watch_loop())

        # Set the task name for easier debugging
        self._watch_task.set_name(f"monitor_{self.project_name}")

        logger.info(f"Started monitoring repository: {self.repo_path} (daemon={daemon})")

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

    async def get_file_info(self, file_path: Path) -> FileInfo | None:
        """
        Get information about a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileInfo object or None if file doesn't exist or shouldn't be included
        """
        if not file_path.exists():
            return None

        if not self._should_include_file(file_path):
            return None

        try:
            stat = file_path.stat()
            return FileInfo(
                path=file_path,
                size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
            )
        except OSError as e:
            logger.warning(f"Could not stat file {file_path}: {e}")
            return None
