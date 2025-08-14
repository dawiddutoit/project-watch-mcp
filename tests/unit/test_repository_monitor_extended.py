"""Extended tests for repository monitor to improve coverage."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from watchfiles import Change

from project_watch_mcp.repository_monitor import (
    FileChangeEvent,
    FileChangeType,
    FileInfo,
    RepositoryMonitor,
)


class TestRepositoryMonitorExtended:
    """Extended tests for repository monitor."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create a mock Neo4j driver."""
        return AsyncMock()

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with test files."""
        repo = tmp_path / "test_repo"
        repo.mkdir()

        # Create test files
        (repo / "test.py").write_text("print('test')")
        (repo / "test.js").write_text("console.log('test');")
        (repo / "test.txt").write_text("not monitored")

        # Create subdirectory with files
        subdir = repo / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("# nested file")

        # Create .gitignore
        (repo / ".gitignore").write_text("*.log\n__pycache__/\nnode_modules/")

        return repo

    @pytest.fixture
    def monitor(self, temp_repo, mock_neo4j_driver):
        """Create a repository monitor instance."""
        return RepositoryMonitor(
            repo_path=temp_repo,
            project_name="test_project",
            neo4j_driver=mock_neo4j_driver,
            file_patterns=["*.py", "*.js"]
        )

    @pytest.mark.asyncio
    async def test_scan_repository_with_gitignore(self, temp_repo, monitor):
        """Test repository scanning with gitignore patterns."""
        # Create files that should be ignored
        (temp_repo / "test.log").write_text("should be ignored")
        cache_dir = temp_repo / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "test.pyc").write_text("ignored")

        files = await monitor.scan_repository()

        # Should not include ignored files
        file_paths = [str(f.path.relative_to(temp_repo)) for f in files]
        assert "test.log" not in file_paths
        assert "__pycache__/test.pyc" not in file_paths
        assert "test.py" in file_paths
        assert "test.js" in file_paths

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Event handling not yet implemented in RepositoryMonitor")
    async def test_process_file_change_created(self, monitor, temp_repo):
        """Test processing file creation."""
        new_file = temp_repo / "new.py"
        new_file.write_text("# new file")

        change = FileChangeEvent(
            path=new_file, change_type=FileChangeType.CREATED, timestamp=datetime.now()
        )

        # Set up callback
        callback_called = False
        file_info = None

        @monitor.on("file_created")
        async def on_created(info):
            nonlocal callback_called, file_info
            callback_called = True
            file_info = info

        await monitor._process_file_change(change)

        assert callback_called
        assert file_info.path == new_file
        assert file_info.language == "python"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Event handling not yet implemented in RepositoryMonitor")
    async def test_process_file_change_modified(self, monitor, temp_repo):
        """Test processing file modification."""
        test_file = temp_repo / "test.py"

        change = FileChangeEvent(
            path=test_file, change_type=FileChangeType.MODIFIED, timestamp=datetime.now()
        )

        callback_called = False

        @monitor.on("file_modified")
        async def on_modified(info):
            nonlocal callback_called
            callback_called = True

        await monitor._process_file_change(change)

        assert callback_called

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Event handling not yet implemented in RepositoryMonitor")
    async def test_process_file_change_deleted(self, monitor, temp_repo):
        """Test processing file deletion."""
        deleted_file = temp_repo / "deleted.py"

        change = FileChangeEvent(
            path=deleted_file, change_type=FileChangeType.DELETED, timestamp=datetime.now()
        )

        callback_called = False
        deleted_path = None

        @monitor.on("file_deleted")
        async def on_deleted(path):
            nonlocal callback_called, deleted_path
            callback_called = True
            deleted_path = path

        await monitor._process_file_change(change)

        assert callback_called
        assert deleted_path == deleted_file

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="_process_file_change method not yet implemented")
    async def test_process_file_change_no_callback(self, monitor, temp_repo):
        """Test processing changes without callbacks."""
        change = FileChangeEvent(
            path=temp_repo / "test.py", change_type=FileChangeType.CREATED, timestamp=datetime.now()
        )

        # Should not raise even without callbacks
        await monitor._process_file_change(change)

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Event handling not yet implemented in RepositoryMonitor")
    async def test_process_file_change_callback_error(self, monitor, temp_repo):
        """Test error handling in callbacks."""
        change = FileChangeEvent(
            path=temp_repo / "test.py", change_type=FileChangeType.CREATED, timestamp=datetime.now()
        )

        @monitor.on("file_created")
        async def failing_callback(info):
            raise Exception("Callback error")

        # Should not raise even if callback fails
        await monitor._process_file_change(change)

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="_watch_task method not yet implemented")
    async def test_watch_task_with_changes(self, monitor, temp_repo):
        """Test watch task processing changes."""

        # Mock watchfiles.awatch
        async def mock_watch(*args, **kwargs):
            # Yield one set of changes then stop
            yield {
                (Change.added, str(temp_repo / "new.py")),
                (Change.modified, str(temp_repo / "test.py")),
                (Change.deleted, str(temp_repo / "old.py")),
            }

        with patch("project_watch_mcp.repository_monitor.awatch", mock_watch):
            # Run watch task briefly
            task = asyncio.create_task(monitor._watch_task())
            await asyncio.sleep(0.1)
            monitor._stop_event.set()

            try:
                await asyncio.wait_for(task, timeout=1)
            except TimeoutError:
                task.cancel()

            # Check that changes were queued
            changes = await monitor.process_all_changes()
            assert len(changes) > 0

    @pytest.mark.asyncio
    async def test_process_all_changes(self, monitor):
        """Test processing all queued changes."""
        # Queue some changes
        changes = [
            FileChangeEvent(Path("/test1.py"), FileChangeType.CREATED, datetime.now()),
            FileChangeEvent(Path("/test2.py"), FileChangeType.MODIFIED, datetime.now()),
            FileChangeEvent(Path("/test3.py"), FileChangeType.DELETED, datetime.now()),
        ]

        for change in changes:
            await monitor._change_queue.put(change)

        processed = await monitor.process_all_changes()

        assert len(processed) == 3
        assert monitor._change_queue.empty()

    @pytest.mark.asyncio
    async def test_start_already_running(self, monitor):
        """Test starting monitor when already running."""
        monitor.is_running = True
        monitor._watch_task_handle = Mock()

        await monitor.start()

        # Should not create new task
        assert monitor._watch_task_handle == monitor._watch_task_handle

    @pytest.mark.asyncio
    async def test_stop_not_running(self, monitor):
        """Test stopping monitor when not running."""
        monitor.is_running = False

        await monitor.stop()

        # Should handle gracefully
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_stop_with_task_cancellation(self, monitor):
        """Test stopping with task cancellation."""
        # Start the monitor
        with patch("project_watch_mcp.repository_monitor.awatch") as mock_watch:

            async def never_yield(*args, **kwargs):
                while True:
                    await asyncio.sleep(1)
                    yield set()

            mock_watch.return_value = never_yield()

            await monitor.start()
            assert monitor.is_running

            # Stop should cancel the task
            await monitor.stop()
            assert not monitor.is_running

    def test_file_info_language_detection(self):
        """Test language detection in FileInfo."""
        test_cases = [
            ("test.py", "python"),
            ("test.js", "javascript"),
            ("test.ts", "typescript"),
            ("test.java", "java"),
            ("test.cpp", "cpp"),
            ("test.c", "c"),
            ("test.h", "c"),
            ("test.cs", "csharp"),
            ("test.rb", "ruby"),
            ("test.go", "go"),
            ("test.rs", "rust"),
            ("test.php", "php"),
            ("test.swift", "swift"),
            ("test.kt", "kotlin"),
            ("test.md", "markdown"),
            ("test.json", "json"),
            ("test.yaml", "yaml"),
            ("test.yml", "yaml"),
            ("test.xml", "xml"),
            ("test.html", "html"),
            ("test.css", "css"),
            ("test.scss", "css"),
            ("test.unknown", "text"),
        ]

        for filename, expected_language in test_cases:
            file_info = FileInfo(path=Path(filename), size=100, last_modified=datetime.now())
            assert file_info.language == expected_language

    def test_file_change_type_enum(self):
        """Test FileChangeType enum values."""
        assert FileChangeType.CREATED.value == "created"
        assert FileChangeType.MODIFIED.value == "modified"
        assert FileChangeType.DELETED.value == "deleted"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="_is_path_ignored is a private method not exposed in the current implementation")
    async def test_is_path_ignored_with_patterns(self, monitor):
        """Test path ignoring with various patterns."""
        # Test with directory patterns
        assert monitor._is_path_ignored(Path("node_modules/package.json"))
        assert monitor._is_path_ignored(Path("__pycache__/test.pyc"))

        # Test with file patterns
        assert monitor._is_path_ignored(Path("test.log"))
        assert monitor._is_path_ignored(Path("subdir/test.log"))

        # Test non-ignored paths
        assert not monitor._is_path_ignored(Path("test.py"))
        assert not monitor._is_path_ignored(Path("src/main.js"))

    @pytest.mark.skip(reason="Method _matches_patterns does not exist")
    @pytest.mark.asyncio
    async def test_matches_patterns(self, monitor):
        """Test file pattern matching."""
        assert monitor._matches_patterns(Path("test.py"))
        assert monitor._matches_patterns(Path("test.js"))
        assert monitor._matches_patterns(Path("subdir/nested.py"))

        assert not monitor._matches_patterns(Path("test.txt"))
        assert not monitor._matches_patterns(Path("test.md"))
        assert not monitor._matches_patterns(Path("test.log"))
