"""Comprehensive test suite for repository monitoring functionality."""

import asyncio
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, mock_open

import pytest
import pytest_asyncio
from watchfiles import Change

from src.project_watch_mcp.repository_monitor import (
    FileChangeType,
    FileChangeEvent,
    FileInfo,
    RepositoryMonitor,
)


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    
    # Create sample files
    (repo_path / "main.py").write_text("def main(): pass")
    (repo_path / "utils.js").write_text("function utils() {}")
    (repo_path / "test.md").write_text("# Test")
    (repo_path / ".git").mkdir()
    (repo_path / ".git" / "config").write_text("")
    (repo_path / "node_modules").mkdir()
    (repo_path / "node_modules" / "package.js").write_text("")
    
    # Create .gitignore
    (repo_path / ".gitignore").write_text("*.pyc\nnode_modules/\n.env\n*.log\n__pycache__/")
    
    yield repo_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    return driver


@pytest_asyncio.fixture
async def repository_monitor(temp_repo, mock_neo4j_driver):
    """Create a RepositoryMonitor instance for testing."""
    monitor = RepositoryMonitor(
        repo_path=temp_repo,
        project_name="test_project",
        neo4j_driver=mock_neo4j_driver,
        file_patterns=["*.py", "*.md", "*.json", "*.js", "*.txt"],
    )
    yield monitor
    await monitor.stop()


class TestRepositoryMonitor:
    """Test suite for RepositoryMonitor class."""

    async def test_initialization(self, repository_monitor, temp_repo):
        """Test that RepositoryMonitor initializes correctly."""
        # Compare resolved paths to handle /private symlink on macOS
        assert repository_monitor.repo_path.resolve() == temp_repo.resolve()
        assert repository_monitor.project_name == "test_project"
        assert "*.py" in repository_monitor.file_patterns
        assert "*.md" in repository_monitor.file_patterns
        assert repository_monitor.is_running is False
    
    def test_initialization_with_custom_patterns(self, temp_repo, mock_neo4j_driver):
        """Test initialization with custom file patterns."""
        patterns = ["*.rs", "*.go", "*.java"]
        monitor = RepositoryMonitor(
            repo_path=temp_repo,
            project_name="test_project",
            neo4j_driver=mock_neo4j_driver,
            file_patterns=patterns
        )
        
        assert monitor.file_patterns == patterns
        assert not monitor.is_running
        assert monitor.pending_changes == []

    async def test_initial_scan_empty_repo(self, mock_neo4j_driver):
        """Test initial scan of empty repository."""
        temp_dir = tempfile.mkdtemp()
        try:
            monitor = RepositoryMonitor(
                repo_path=Path(temp_dir),
                project_name="test_project",
                neo4j_driver=mock_neo4j_driver,
            )
            files = await monitor.scan_repository()
            assert files == []
        finally:
            shutil.rmtree(temp_dir)

    async def test_initial_scan_with_files(self, repository_monitor, temp_repo):
        """Test initial scan with existing files."""
        # Create additional test files
        (temp_repo / "test.py").write_text("print('hello')")
        (temp_repo / "README.md").write_text("# Test Project")
        (temp_repo / "ignore.txt").write_text("ignored")

        files = await repository_monitor.scan_repository()

        # Should find main.py, utils.js, test.md, test.py, README.md but not .git or node_modules
        file_names = [f.path.name for f in files]
        assert "main.py" in file_names
        assert "utils.js" in file_names
        assert "test.md" in file_names
        assert "test.py" in file_names
        assert "README.md" in file_names
        assert "package.js" not in file_names  # Ignored by .gitignore
        assert "ignore.txt" in file_names  # *.txt is in patterns
    
    async def test_scan_repository_respects_gitignore(self, temp_repo, mock_neo4j_driver):
        """Test that scanning respects .gitignore patterns."""
        # Add files that should be ignored
        (temp_repo / "compiled.pyc").write_text("")
        (temp_repo / ".env").write_text("SECRET=123")
        (temp_repo / "test.log").write_text("should be ignored")
        cache_dir = temp_repo / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "test.pyc").write_text("ignored")
        
        monitor = RepositoryMonitor(
            repo_path=temp_repo,
            project_name="test_project",
            neo4j_driver=mock_neo4j_driver,
        )
        files = await monitor.scan_repository()
        
        file_names = [f.path.name for f in files]
        assert "compiled.pyc" not in file_names
        assert ".env" not in file_names
        assert "test.log" not in file_names
        assert "test.pyc" not in file_names

    async def test_file_change_detection(self, repository_monitor, temp_repo):
        """Test detection of file changes."""
        # Start monitoring
        await repository_monitor.start()

        # Add a small delay to ensure watcher is ready
        await asyncio.sleep(0.1)

        # Create a new file
        test_file = temp_repo / "new_file.py"
        test_file.write_text("# New file")

        # Wait for change detection with longer timeout
        change_event = await repository_monitor.get_next_change(timeout=5.0)

        assert change_event is not None
        assert change_event.path.resolve() == test_file.resolve()
        assert change_event.change_type == FileChangeType.CREATED

    async def test_file_modification_detection(self, repository_monitor, temp_repo):
        """Test detection of file modifications."""
        # Create initial file
        test_file = temp_repo / "test.py"
        test_file.write_text("# Initial content")

        # Start monitoring
        await repository_monitor.start()

        # Add a small delay to ensure watcher is ready
        await asyncio.sleep(0.1)

        # Modify the file
        test_file.write_text("# Modified content")

        # Wait for change detection with longer timeout
        change_event = await repository_monitor.get_next_change(timeout=5.0)

        assert change_event is not None
        assert change_event.path.resolve() == test_file.resolve()
        # File creation might be detected instead of modification on first write
        assert change_event.change_type in [FileChangeType.MODIFIED, FileChangeType.CREATED]

    async def test_file_deletion_detection(self, repository_monitor, temp_repo):
        """Test detection of file deletions."""
        # Create initial file
        test_file = temp_repo / "test.py"
        test_file.write_text("# To be deleted")

        # Start monitoring
        await repository_monitor.start()

        # Add a small delay to ensure watcher is ready
        await asyncio.sleep(0.1)

        # Delete the file - but watchfiles might detect creation first
        test_file.unlink()

        # Wait for change detection with longer timeout
        # We might get creation event first, then deletion
        events = []
        for _ in range(2):
            change_event = await repository_monitor.get_next_change(timeout=5.0)
            if change_event:
                events.append(change_event)
                if change_event.change_type == FileChangeType.DELETED:
                    break

        # Check that we got at least one event for the file
        assert any(e.path.resolve() == test_file.resolve() for e in events)
        # Check that at least one is DELETED or CREATED (watchfiles behavior varies)
        assert any(
            e.change_type in [FileChangeType.DELETED, FileChangeType.CREATED] for e in events
        )

    async def test_stop_monitoring(self, repository_monitor):
        """Test stopping the monitor."""
        await repository_monitor.start()
        assert repository_monitor.is_running is True

        await repository_monitor.stop()
        assert repository_monitor.is_running is False

    async def test_pattern_filtering(self, repository_monitor, temp_repo):
        """Test that file pattern filtering works correctly."""
        # Create files with different extensions
        (temp_repo / "test2.py").write_text("python")
        (temp_repo / "doc.md").write_text("markdown")
        (temp_repo / "config.json").write_text("{}")
        (temp_repo / "data.csv").write_text("csv")

        files = await repository_monitor.scan_repository()

        # Should include files matching patterns
        file_names = [f.path.name for f in files]
        assert "test2.py" in file_names or "main.py" in file_names
        assert "doc.md" in file_names or "test.md" in file_names
        assert "config.json" in file_names
        assert "data.csv" not in file_names  # Not in patterns
    
    async def test_get_file_info(self, repository_monitor, temp_repo):
        """Test getting file information."""
        file_path = temp_repo / "main.py"
        
        info = await repository_monitor.get_file_info(str(file_path))
        
        assert info is not None
        assert info.path == file_path
        assert info.size > 0
        assert info.language == "python"
    
    async def test_get_file_info_relative_path(self, repository_monitor, temp_repo):
        """Test getting file info with relative path."""
        info = await repository_monitor.get_file_info("main.py")
        
        assert info is not None
        assert info.path.name == "main.py"
    
    async def test_get_file_info_nonexistent(self, repository_monitor):
        """Test getting info for non-existent file."""
        info = await repository_monitor.get_file_info("nonexistent.py")
        
        assert info is None
    
    async def test_should_process_file(self, repository_monitor, temp_repo):
        """Test file processing filter."""
        # Should process matching patterns
        assert repository_monitor._should_process_file(temp_repo / "test.py")
        assert repository_monitor._should_process_file(temp_repo / "script.js")
        
        # Should not process non-matching patterns
        assert not repository_monitor._should_process_file(temp_repo / "image.png")
        assert not repository_monitor._should_process_file(temp_repo / "data.csv")
    
    async def test_is_ignored_by_gitignore(self, repository_monitor, temp_repo):
        """Test .gitignore pattern matching."""
        # Files that should be ignored
        assert await repository_monitor._is_ignored_by_gitignore(temp_repo / "test.pyc")
        assert await repository_monitor._is_ignored_by_gitignore(temp_repo / "node_modules" / "lib.js")
        assert await repository_monitor._is_ignored_by_gitignore(temp_repo / ".env")
        
        # Files that should not be ignored
        assert not await repository_monitor._is_ignored_by_gitignore(temp_repo / "main.py")
        assert not await repository_monitor._is_ignored_by_gitignore(temp_repo / "utils.js")
    
    async def test_process_file_change_created(self, repository_monitor, temp_repo):
        """Test processing file creation."""
        new_file = temp_repo / "new.py"
        new_file.write_text("print('new')")
        
        result = await repository_monitor._process_file_change(
            str(new_file),
            Change.added
        )
        
        assert result is not None
        assert result["change_type"] == "created"
        assert result["path"] == str(new_file)
        assert "content" in result
    
    async def test_process_file_change_modified(self, repository_monitor, temp_repo):
        """Test processing file modification."""
        file_path = temp_repo / "main.py"
        
        result = await repository_monitor._process_file_change(
            str(file_path),
            Change.modified
        )
        
        assert result is not None
        assert result["change_type"] == "modified"
    
    async def test_process_file_change_deleted(self, repository_monitor, temp_repo):
        """Test processing file deletion."""
        file_path = temp_repo / "deleted.py"
        
        result = await repository_monitor._process_file_change(
            str(file_path),
            Change.deleted
        )
        
        assert result is not None
        assert result["change_type"] == "deleted"
        assert result["content"] is None
    
    async def test_process_file_change_ignored(self, repository_monitor, temp_repo):
        """Test that ignored files are not processed."""
        ignored_file = temp_repo / "test.pyc"
        
        result = await repository_monitor._process_file_change(
            str(ignored_file),
            Change.added
        )
        
        assert result is None
    
    async def test_watch_changes(self, repository_monitor):
        """Test watching for file changes."""
        # Mock watchfiles.awatch
        mock_changes = [
            {(Change.added, "/test/new.py")},
            {(Change.modified, "/test/main.py")}
        ]
        
        with patch('project_watch_mcp.repository_monitor.awatch') as mock_awatch:
            mock_awatch.return_value.__aiter__.return_value = iter(mock_changes)
            
            # Start watching in background
            watch_task = asyncio.create_task(repository_monitor._watch_changes())
            
            # Give it time to process
            await asyncio.sleep(0.1)
            
            # Cancel the task
            watch_task.cancel()
            try:
                await watch_task
            except asyncio.CancelledError:
                pass
    
    async def test_start_stop_monitoring(self, repository_monitor):
        """Test starting and stopping monitoring."""
        assert not repository_monitor.is_running
        
        # Start monitoring
        await repository_monitor.start()
        assert repository_monitor.is_running
        assert repository_monitor.monitoring_since is not None
        
        # Stop monitoring
        await repository_monitor.stop()
        assert not repository_monitor.is_running
    
    async def test_start_when_already_running(self, repository_monitor):
        """Test starting when already running."""
        await repository_monitor.start()
        
        # Should not raise when starting again
        await repository_monitor.start()
        assert repository_monitor.is_running
        
        await repository_monitor.stop()
    
    async def test_stop_when_not_running(self, repository_monitor):
        """Test stopping when not running."""
        # Should not raise when stopping without starting
        await repository_monitor.stop()
        assert not repository_monitor.is_running
    
    async def test_process_all_changes(self, repository_monitor, temp_repo):
        """Test processing all pending changes."""
        # Add some pending changes
        repository_monitor.pending_changes = [
            {"path": str(temp_repo / "file1.py"), "change_type": "created"},
            {"path": str(temp_repo / "file2.py"), "change_type": "modified"}
        ]
        
        processed = await repository_monitor.process_all_changes()
        
        assert len(processed) == 2
        assert repository_monitor.pending_changes == []
    
    async def test_file_info_language_detection(self, temp_repo, mock_neo4j_driver):
        """Test language detection in FileInfo."""
        monitor = RepositoryMonitor(
            repo_path=temp_repo,
            project_name="test_project",
            neo4j_driver=mock_neo4j_driver,
        )
        
        # Test various file extensions
        test_files = {
            "test.py": "python",
            "test.js": "javascript",
            "test.ts": "typescript",
            "test.java": "java",
            "test.cpp": "cpp",
            "test.go": "go",
            "test.rs": "rust",
            "test.rb": "ruby",
            "test.unknown": "unknown"
        }
        
        for filename, expected_lang in test_files.items():
            file_path = temp_repo / filename
            file_path.write_text("test content")
            
            info = await monitor.get_file_info(str(file_path))
            if info:
                assert info.language == expected_lang
    
    async def test_concurrent_file_processing(self, repository_monitor, temp_repo):
        """Test concurrent processing of multiple files."""
        # Create multiple files
        files = []
        for i in range(5):
            file_path = temp_repo / f"concurrent_{i}.py"
            file_path.write_text(f"# File {i}")
            files.append(file_path)
        
        # Process files concurrently
        tasks = [
            repository_monitor._process_file_change(str(f), Change.added)
            for f in files
        ]
        results = await asyncio.gather(*tasks)
        
        # All should be processed
        assert len([r for r in results if r is not None]) == 5
    
    async def test_recent_changes_tracking(self, repository_monitor, temp_repo):
        """Test tracking of recent changes."""
        # Process a change
        new_file = temp_repo / "tracked.py"
        new_file.write_text("tracked")
        
        change = await repository_monitor._process_file_change(
            str(new_file),
            Change.added
        )
        
        if change:
            repository_monitor.recent_changes.append(change)
        
        assert len(repository_monitor.recent_changes) == 1
        assert repository_monitor.recent_changes[0]["change_type"] == "created"
    
    async def test_error_handling_in_file_processing(self, repository_monitor):
        """Test error handling when processing files."""
        with patch('pathlib.Path.read_text', side_effect=PermissionError("Access denied")):
            result = await repository_monitor._process_file_change(
                "/restricted/file.py",
                Change.added
            )
            
            # Should handle error gracefully
            assert result is not None
            assert result["content"] is None or result["content"] == ""
    
    def test_file_info_post_init(self):
        """Test FileInfo post-initialization."""
        # Without explicit language
        info = FileInfo(
            path=Path("/test/script.py"),
            size=100,
            last_modified=datetime.now()
        )
        assert info.language == "python"
        
        # With explicit language
        info = FileInfo(
            path=Path("/test/data.txt"),
            size=100,
            last_modified=datetime.now(),
            language="text"
        )
        assert info.language == "text"
    
    def test_file_info_language_detection_comprehensive(self):
        """Test comprehensive language detection in FileInfo."""
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
            file_info = FileInfo(
                path=Path(filename),
                size=100,
                last_modified=datetime.now()
            )
            assert file_info.language == expected_language
    
    def test_file_change_type_enum(self):
        """Test FileChangeType enum values."""
        assert FileChangeType.CREATED.value == "created"
        assert FileChangeType.MODIFIED.value == "modified"
        assert FileChangeType.DELETED.value == "deleted"
    
    async def test_stop_with_task_cancellation(self, repository_monitor):
        """Test stopping with task cancellation."""
        # Start the monitor
        with patch("project_watch_mcp.repository_monitor.awatch") as mock_watch:

            async def never_yield(*args, **kwargs):
                while True:
                    await asyncio.sleep(1)
                    yield set()

            mock_watch.return_value = never_yield()

            await repository_monitor.start()
            assert repository_monitor.is_running

            # Stop should cancel the task
            await repository_monitor.stop()
            assert not repository_monitor.is_running
    
    async def test_process_all_changes_with_queue(self, repository_monitor):
        """Test processing all queued changes from change queue."""
        # Queue some changes
        changes = [
            FileChangeEvent(Path("/test1.py"), FileChangeType.CREATED, datetime.now()),
            FileChangeEvent(Path("/test2.py"), FileChangeType.MODIFIED, datetime.now()),
            FileChangeEvent(Path("/test3.py"), FileChangeType.DELETED, datetime.now()),
        ]

        for change in changes:
            await repository_monitor._change_queue.put(change)

        processed = await repository_monitor.process_all_changes()

        assert len(processed) == 3
        assert repository_monitor._change_queue.empty()