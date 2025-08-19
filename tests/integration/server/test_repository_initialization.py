"""
Integration tests for repository initialization with real file system operations.

These tests verify:
1. Repository scanning and file discovery
2. File pattern matching and filtering
3. .gitignore respect
4. Initialization state persistence
5. Re-initialization handling
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Set
import pytest

from project_watch_mcp.neo4j_rag import CodeFile
from project_watch_mcp.repository_monitor import FileInfo, RepositoryMonitor
from project_watch_mcp.neo4j_rag import Neo4jRAG
from project_watch_mcp.core.initializer import RepositoryInitializer
from project_watch_mcp.config import ProjectWatchConfig


class TestRepositoryInitialization:
    """Test suite for repository initialization process."""

    @pytest.fixture
    async def complex_repo(self):
        """Create a complex repository structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create directory structure
            (repo_path / ".git").mkdir()
            (repo_path / "src" / "core").mkdir(parents=True)
            (repo_path / "src" / "utils").mkdir(parents=True)
            (repo_path / "tests" / "unit").mkdir(parents=True)
            (repo_path / "tests" / "integration").mkdir(parents=True)
            (repo_path / "docs").mkdir()
            (repo_path / "scripts").mkdir()
            (repo_path / "node_modules" / "package").mkdir(parents=True)
            (repo_path / "__pycache__").mkdir()
            (repo_path / ".venv" / "lib").mkdir(parents=True)
            
            # Create .gitignore
            (repo_path / ".gitignore").write_text("""
# Python
__pycache__/
*.pyc
.venv/
venv/
env/

# Node
node_modules/
dist/
build/

# IDE
.idea/
.vscode/
*.swp

# Temporary
*.tmp
*.log
""")
            
            # Create various file types
            # Python files
            (repo_path / "src" / "core" / "main.py").write_text("""
import asyncio
from typing import Optional

class Application:
    def __init__(self, config: dict):
        self.config = config
    
    async def run(self) -> int:
        print("Starting application...")
        await asyncio.sleep(1)
        return 0
""")
            
            (repo_path / "src" / "core" / "__init__.py").write_text("")
            
            (repo_path / "src" / "utils" / "helpers.py").write_text("""
def format_timestamp(ts: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(ts).isoformat()

def calculate_hash(content: str) -> str:
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()
""")
            
            # JavaScript/TypeScript files
            (repo_path / "src" / "index.js").write_text("""
const express = require('express');
const app = express();

app.get('/', (req, res) => {
    res.send('Hello World!');
});

module.exports = app;
""")
            
            (repo_path / "src" / "types.ts").write_text("""
export interface User {
    id: string;
    name: string;
    email: string;
}

export type Role = 'admin' | 'user' | 'guest';
""")
            
            # Test files
            (repo_path / "tests" / "unit" / "test_main.py").write_text("""
import pytest
from src.core.main import Application

@pytest.mark.asyncio
async def test_application_run():
    app = Application({})
    result = await app.run()
    assert result == 0
""")
            
            # Configuration files
            (repo_path / "pyproject.toml").write_text("""
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["pytest", "asyncio"]
""")
            
            (repo_path / "package.json").write_text("""
{
    "name": "test-project",
    "version": "1.0.0",
    "dependencies": {
        "express": "^4.18.0"
    }
}
""")
            
            # Documentation
            (repo_path / "README.md").write_text("""
# Test Project

A comprehensive test project for initialization testing.

## Features
- Python backend
- JavaScript frontend
- Comprehensive test suite
""")
            
            (repo_path / "docs" / "API.md").write_text("""
# API Documentation

## Endpoints
- GET / - Returns hello world
""")
            
            # Files that should be ignored
            (repo_path / "__pycache__" / "cache.pyc").write_text("compiled bytecode")
            (repo_path / "node_modules" / "package" / "index.js").write_text("external dependency")
            (repo_path / ".venv" / "lib" / "module.py").write_text("virtual env file")
            (repo_path / "temp.tmp").write_text("temporary file")
            (repo_path / "debug.log").write_text("log file content")
            
            # Shell scripts
            (repo_path / "scripts" / "deploy.sh").write_text("""
#!/bin/bash
echo "Deploying application..."
docker build -t app .
docker push app:latest
""")
            
            yield repo_path

    @pytest.mark.asyncio
    async def test_repository_scanner_discovers_all_files(self, complex_repo):
        """Test that repository scanner discovers all non-ignored files."""
        monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.md", "*.json", "*.toml", "*.sh"]
        )
        
        # Scan repository
        files = await monitor.scan_repository()
        
        # Verify file discovery
        file_paths = [f.relative_path for f in files]
        
        # Should include these files
        expected_files = [
            "src/core/main.py",
            "src/core/__init__.py",
            "src/utils/helpers.py",
            "src/index.js",
            "src/types.ts",
            "tests/unit/test_main.py",
            "pyproject.toml",
            "package.json",
            "README.md",
            "docs/API.md",
            "scripts/deploy.sh"
        ]
        
        for expected in expected_files:
            assert expected in file_paths, f"Expected file {expected} not found"
        
        # Should NOT include these files
        ignored_files = [
            "__pycache__/cache.pyc",
            "node_modules/package/index.js",
            ".venv/lib/module.py",
            "temp.tmp",
            "debug.log"
        ]
        
        for ignored in ignored_files:
            assert ignored not in file_paths, f"Ignored file {ignored} was included"

    @pytest.mark.asyncio
    async def test_repository_initialization_with_monitoring(self, complex_repo):
        """Test complete repository initialization with monitoring."""
        monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*.py", "*.js", "*.ts", "*.md", "*.json", "*.toml"]
        )
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Verify monitoring is active
        assert monitor.is_monitoring
        assert monitor.monitoring_since is not None
        assert isinstance(monitor.monitoring_since, datetime)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        assert not monitor.is_monitoring

    @pytest.mark.asyncio
    async def test_file_pattern_matching(self, complex_repo):
        """Test that file pattern matching works correctly."""
        # Test with Python files only
        python_monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*.py"]
        )
        
        python_files = await python_monitor.scan_repository()
        python_paths = [f.relative_path for f in python_files]
        
        # Should only include Python files
        for path in python_paths:
            assert path.endswith(".py"), f"Non-Python file included: {path}"
        
        # Test with JavaScript/TypeScript files
        js_monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*.js", "*.ts", "*.jsx", "*.tsx"]
        )
        
        js_files = await js_monitor.scan_repository()
        js_paths = [f.relative_path for f in js_files]
        
        # Should include JS/TS files
        assert "src/index.js" in js_paths
        assert "src/types.ts" in js_paths
        
        # Should not include Python files
        assert "src/core/main.py" not in js_paths

    @pytest.mark.asyncio
    async def test_gitignore_patterns_respected(self, complex_repo):
        """Test that .gitignore patterns are properly respected."""
        monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*"]  # Match everything
        )
        
        files = await monitor.scan_repository()
        file_paths = [f.relative_path for f in files]
        
        # Verify ignored directories are excluded
        for path in file_paths:
            assert not path.startswith("__pycache__/"), f"__pycache__ file included: {path}"
            assert not path.startswith("node_modules/"), f"node_modules file included: {path}"
            assert not path.startswith(".venv/"), f".venv file included: {path}"
            assert not path.endswith(".pyc"), f".pyc file included: {path}"
            assert not path.endswith(".tmp"), f".tmp file included: {path}"
            assert not path.endswith(".log"), f".log file included: {path}"

    @pytest.mark.asyncio
    async def test_file_metadata_extraction(self, complex_repo):
        """Test that file metadata is correctly extracted."""
        monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*.py"]
        )
        
        files = await monitor.scan_repository()
        
        for file_info in files:
            # Check required metadata
            assert file_info.path is not None
            assert file_info.relative_path is not None
            assert file_info.size >= 0
            assert file_info.modified_time > 0
            assert file_info.language == "python"
            
            # Verify file actually exists
            full_path = Path(file_info.path)
            assert full_path.exists()
            assert full_path.is_file()
            
            # Verify size matches
            actual_size = full_path.stat().st_size
            assert file_info.size == actual_size

    @pytest.mark.asyncio
    async def test_language_detection(self, complex_repo):
        """Test that programming languages are correctly detected."""
        monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*.py", "*.js", "*.ts", "*.md", "*.json", "*.toml", "*.sh"]
        )
        
        files = await monitor.scan_repository()
        
        # Group files by language
        languages = {}
        for file_info in files:
            if file_info.language not in languages:
                languages[file_info.language] = []
            languages[file_info.language].append(file_info.relative_path)
        
        # Verify language detection
        assert "python" in languages
        assert "javascript" in languages
        assert "typescript" in languages
        assert "markdown" in languages
        assert "json" in languages
        assert "toml" in languages
        assert "shell" in languages
        
        # Check specific files
        assert "src/core/main.py" in languages["python"]
        assert "src/index.js" in languages["javascript"]
        assert "src/types.ts" in languages["typescript"]
        assert "README.md" in languages["markdown"]
        assert "package.json" in languages["json"]
        assert "pyproject.toml" in languages["toml"]
        assert "scripts/deploy.sh" in languages["shell"]

    @pytest.mark.asyncio
    async def test_incremental_repository_scan(self, complex_repo):
        """Test that incremental scans detect new and modified files."""
        monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*.py"]
        )
        
        # Initial scan
        initial_files = await monitor.scan_repository()
        initial_count = len(initial_files)
        
        # Add a new file
        new_file = complex_repo / "src" / "new_module.py"
        new_file.write_text("""
def new_function():
    return "New functionality"
""")
        
        # Rescan
        updated_files = await monitor.scan_repository()
        updated_count = len(updated_files)
        
        # Verify new file is detected
        assert updated_count == initial_count + 1
        
        file_paths = [f.relative_path for f in updated_files]
        assert "src/new_module.py" in file_paths
        
        # Modify existing file
        existing_file = complex_repo / "src" / "core" / "main.py"
        original_content = existing_file.read_text()
        existing_file.write_text(original_content + "\n# Modified")
        
        # Get file info for modified file
        modified_files = await monitor.scan_repository()
        
        # Find the modified file
        for file_info in modified_files:
            if file_info.relative_path == "src/core/main.py":
                # Modification time should be recent
                assert file_info.modified_time > 0
                break

    @pytest.mark.asyncio
    async def test_repository_initialization_idempotency(self, complex_repo):
        """Test that repository initialization is idempotent."""
        monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*.py", "*.js", "*.ts"]
        )
        
        # First initialization
        files1 = await monitor.scan_repository()
        paths1 = set(f.relative_path for f in files1)
        
        # Second initialization (should return same results)
        files2 = await monitor.scan_repository()
        paths2 = set(f.relative_path for f in files2)
        
        # Results should be identical
        assert paths1 == paths2
        assert len(files1) == len(files2)

    @pytest.mark.asyncio
    async def test_empty_repository_initialization(self):
        """Test initialization of an empty repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            (repo_path / ".git").mkdir()
            
            monitor = RepositoryMonitor(
                repo_path=str(repo_path),
                file_patterns=["*.py"]
            )
            
            # Scan empty repository
            files = await monitor.scan_repository()
            
            # Should return empty list, not error
            assert files == []

    @pytest.mark.asyncio
    async def test_large_repository_initialization(self, complex_repo):
        """Test initialization of a repository with many files."""
        # Create many files
        for i in range(100):
            file_path = complex_repo / f"file_{i}.py"
            file_path.write_text(f"# File {i}\nprint('File {i}')")
        
        monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*.py"]
        )
        
        # Scan large repository
        files = await monitor.scan_repository()
        
        # Should handle many files
        assert len(files) > 100
        
        # Verify all generated files are found
        file_paths = [f.relative_path for f in files]
        for i in range(100):
            assert f"file_{i}.py" in file_paths

    @pytest.mark.asyncio
    async def test_concurrent_initialization_requests(self, complex_repo):
        """Test handling of concurrent initialization requests."""
        monitor = RepositoryMonitor(
            repo_path=str(complex_repo),
            file_patterns=["*.py"]
        )
        
        # Start multiple concurrent scans
        tasks = [
            monitor.scan_repository(),
            monitor.scan_repository(),
            monitor.scan_repository()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed and return same results
        assert len(results) == 3
        
        # Compare results
        paths0 = set(f.relative_path for f in results[0])
        paths1 = set(f.relative_path for f in results[1])
        paths2 = set(f.relative_path for f in results[2])
        
        assert paths0 == paths1 == paths2

    @pytest.mark.asyncio
    async def test_initialization_with_symbolic_links(self, complex_repo):
        """Test that symbolic links are handled correctly."""
        # Create a symbolic link
        target = complex_repo / "src" / "core" / "main.py"
        link = complex_repo / "main_link.py"
        
        if os.name != 'nt':  # Skip on Windows
            os.symlink(target, link)
            
            monitor = RepositoryMonitor(
                repo_path=str(complex_repo),
                file_patterns=["*.py"]
            )
            
            files = await monitor.scan_repository()
            file_paths = [f.relative_path for f in files]
            
            # Symbolic link handling depends on gitignore configuration
            # Just ensure it doesn't crash
            assert len(files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])