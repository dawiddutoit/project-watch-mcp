"""
Performance tests for incremental indexing feature.

This module contains performance benchmarks to verify that incremental indexing
provides at least 50% speed improvement over full re-indexing.
"""

import asyncio
import os
import random
import string
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase

from src.project_watch_mcp.neo4j_rag import Neo4jRAG
from src.project_watch_mcp.repository_monitor import FileInfo, RepositoryMonitor
from src.project_watch_mcp.core.initializer import RepositoryInitializer
from src.project_watch_mcp.config import ProjectWatchConfig


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_python_file(lines: int = 50) -> str:
    """Generate a realistic Python file with specified number of lines."""
    content = []
    content.append('"""Generated test file."""')
    content.append("")
    content.append("import random")
    content.append("import string")
    content.append("from typing import List, Dict")
    content.append("")
    
    # Generate some functions
    for i in range(lines // 10):
        content.append(f"def function_{i}(param_{i}: str) -> str:")
        content.append(f'    """Function {i} docstring."""')
        content.append(f'    result = param_{i} + "_{i}"')
        content.append(f'    return result')
        content.append("")
    
    # Generate a class
    content.append("class TestClass:")
    content.append('    """Test class with methods."""')
    content.append("    ")
    content.append("    def __init__(self):")
    content.append("        self.data = {}")
    content.append("    ")
    
    for i in range(max(1, (lines - len(content)) // 5)):
        content.append(f"    def method_{i}(self, value: int) -> int:")
        content.append(f'        """Method {i}."""')
        content.append(f"        return value * {i + 1}")
        content.append("    ")
    
    return "\n".join(content)


def generate_javascript_file(lines: int = 50) -> str:
    """Generate a realistic JavaScript file."""
    content = []
    content.append("// Generated test file")
    content.append("")
    content.append("const utils = require('./utils');")
    content.append("")
    
    for i in range(lines // 8):
        content.append(f"function function{i}(param{i}) {{")
        content.append(f"    // Function {i} implementation")
        content.append(f"    const result = param{i} + '_{i}';")
        content.append(f"    return result;")
        content.append("}")
        content.append("")
    
    content.append("class TestClass {")
    content.append("    constructor() {")
    content.append("        this.data = {};")
    content.append("    }")
    
    for i in range(max(1, (lines - len(content)) // 6)):
        content.append(f"    method{i}(value) {{")
        content.append(f"        return value * {i + 1};")
        content.append("    }")
    
    content.append("}")
    content.append("")
    content.append("module.exports = { TestClass };")
    
    return "\n".join(content)


async def create_large_repository(
    path: Path, 
    num_files: int = 100,
    avg_lines: int = 100
) -> List[Path]:
    """Create a repository with many files for performance testing."""
    created_files = []
    
    # Create directory structure
    dirs = [
        path / "src",
        path / "src" / "core",
        path / "src" / "utils",
        path / "src" / "models",
        path / "tests",
        path / "tests" / "unit",
        path / "tests" / "integration",
        path / "docs",
        path / "scripts",
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Distribute files across directories
    for i in range(num_files):
        # Choose directory
        if i < num_files * 0.6:  # 60% in src
            parent = random.choice([
                path / "src",
                path / "src" / "core",
                path / "src" / "utils",
                path / "src" / "models"
            ])
        elif i < num_files * 0.8:  # 20% in tests
            parent = random.choice([
                path / "tests",
                path / "tests" / "unit",
                path / "tests" / "integration"
            ])
        else:  # 20% in other directories
            parent = random.choice([path, path / "docs", path / "scripts"])
        
        # Choose file type
        if i % 3 == 0:
            filename = f"file_{i}.js"
            content = generate_javascript_file(avg_lines + random.randint(-20, 20))
        else:
            filename = f"module_{i}.py"
            content = generate_python_file(avg_lines + random.randint(-20, 20))
        
        file_path = parent / filename
        file_path.write_text(content)
        created_files.append(file_path)
    
    # Create a .gitignore
    (path / ".gitignore").write_text("""
__pycache__/
*.pyc
.venv/
node_modules/
*.log
*.tmp
""")
    
    return created_files


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
async def neo4j_config():
    """Provide Neo4j configuration for tests."""
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j")
    }


@pytest.fixture
async def clean_neo4j(neo4j_config):
    """Ensure Neo4j is clean before and after tests."""
    driver = AsyncGraphDatabase.driver(
        neo4j_config["uri"],
        auth=(neo4j_config["user"], neo4j_config["password"])
    )
    
    # Clean before test
    async with driver.session(database=neo4j_config["database"]) as session:
        await session.run("MATCH (n) DETACH DELETE n")
    
    yield driver
    
    # Clean after test
    async with driver.session(database=neo4j_config["database"]) as session:
        await session.run("MATCH (n) DETACH DELETE n")
    
    await driver.close()


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

@pytest.mark.performance
@pytest.mark.integration
class TestIncrementalIndexingPerformance:
    """Performance benchmarks for incremental indexing."""
    
    @pytest.mark.asyncio
    async def test_50_percent_improvement_small_repo(
        self, neo4j_config, clean_neo4j
    ):
        """Verify 50%+ improvement on small repository (50 files)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create repository with 50 files
            files = await create_large_repository(repo_path, num_files=50, avg_lines=50)
            
            initializer = RepositoryInitializer(
                neo4j_uri=neo4j_config["uri"],
                PROJECT_WATCH_USER=neo4j_config["user"],
                PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                PROJECT_WATCH_DATABASE=neo4j_config["database"],
                repository_path=repo_path,
                project_name="perf_test_small"
            )
            
            # Measure full indexing time
            start_time = time.perf_counter()
            async with initializer:
                first_result = await initializer.initialize(persistent_monitoring=False)
            full_index_time = time.perf_counter() - start_time
            
            assert first_result.indexed == len(files)
            
            # Measure incremental indexing time (no changes)
            start_time = time.perf_counter()
            async with initializer:
                second_result = await initializer.initialize(persistent_monitoring=False)
            incremental_time = time.perf_counter() - start_time
            
            assert second_result.indexed == 0
            
            # Calculate improvement
            improvement = (full_index_time - incremental_time) / full_index_time * 100
            
            print(f"\nSmall Repository Performance:")
            print(f"  Files: {len(files)}")
            print(f"  Full indexing: {full_index_time:.3f}s")
            print(f"  Incremental (no changes): {incremental_time:.3f}s")
            print(f"  Improvement: {improvement:.1f}%")
            
            assert improvement > 50, f"Expected >50% improvement, got {improvement:.1f}%"
    
    @pytest.mark.asyncio
    async def test_80_percent_improvement_large_repo(
        self, neo4j_config, clean_neo4j
    ):
        """Verify 80%+ improvement on large repository (200 files) with 10% changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create repository with 200 files
            files = await create_large_repository(repo_path, num_files=200, avg_lines=100)
            
            initializer = RepositoryInitializer(
                neo4j_uri=neo4j_config["uri"],
                PROJECT_WATCH_USER=neo4j_config["user"],
                PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                PROJECT_WATCH_DATABASE=neo4j_config["database"],
                repository_path=repo_path,
                project_name="perf_test_large"
            )
            
            # Measure full indexing time
            start_time = time.perf_counter()
            async with initializer:
                first_result = await initializer.initialize(persistent_monitoring=False)
            full_index_time = time.perf_counter() - start_time
            
            assert first_result.indexed == len(files)
            
            # Modify 10% of files
            num_to_modify = len(files) // 10
            files_to_modify = random.sample(files, num_to_modify)
            
            await asyncio.sleep(0.1)  # Ensure timestamp difference
            
            for file_path in files_to_modify:
                content = file_path.read_text()
                file_path.write_text(content + "\n# Modified")
                os.utime(file_path, None)  # Update timestamp
            
            # Measure incremental indexing time (10% changes)
            start_time = time.perf_counter()
            async with initializer:
                second_result = await initializer.initialize(persistent_monitoring=False)
            incremental_time = time.perf_counter() - start_time
            
            assert second_result.indexed == num_to_modify
            
            # Calculate improvement
            improvement = (full_index_time - incremental_time) / full_index_time * 100
            
            print(f"\nLarge Repository Performance (10% changes):")
            print(f"  Files: {len(files)}")
            print(f"  Modified: {num_to_modify}")
            print(f"  Full indexing: {full_index_time:.3f}s")
            print(f"  Incremental (10% changed): {incremental_time:.3f}s")
            print(f"  Improvement: {improvement:.1f}%")
            
            assert improvement > 80, f"Expected >80% improvement, got {improvement:.1f}%"
    
    @pytest.mark.asyncio
    async def test_performance_scaling(
        self, neo4j_config, clean_neo4j
    ):
        """Test performance scaling with different repository sizes."""
        results = []
        
        for num_files in [10, 50, 100, 200]:
            with tempfile.TemporaryDirectory() as tmpdir:
                repo_path = Path(tmpdir)
                
                # Create repository
                files = await create_large_repository(
                    repo_path, 
                    num_files=num_files, 
                    avg_lines=50
                )
                
                initializer = RepositoryInitializer(
                    neo4j_uri=neo4j_config["uri"],
                    PROJECT_WATCH_USER=neo4j_config["user"],
                    PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                    PROJECT_WATCH_DATABASE=neo4j_config["database"],
                    repository_path=repo_path,
                    project_name=f"perf_scaling_{num_files}"
                )
                
                # Measure full indexing
                start_time = time.perf_counter()
                async with initializer:
                    await initializer.initialize(persistent_monitoring=False)
                full_time = time.perf_counter() - start_time
                
                # Measure incremental (no changes)
                start_time = time.perf_counter()
                async with initializer:
                    await initializer.initialize(persistent_monitoring=False)
                incremental_time = time.perf_counter() - start_time
                
                improvement = (full_time - incremental_time) / full_time * 100
                
                results.append({
                    "files": num_files,
                    "full_time": full_time,
                    "incremental_time": incremental_time,
                    "improvement": improvement
                })
                
                # Clean up for next iteration
                async with clean_neo4j.session(database=neo4j_config["database"]) as session:
                    await session.run("MATCH (n) DETACH DELETE n")
        
        print("\nPerformance Scaling Results:")
        print("Files | Full Index | Incremental | Improvement")
        print("------|------------|-------------|------------")
        for r in results:
            print(f"{r['files']:5d} | {r['full_time']:10.3f}s | {r['incremental_time']:11.3f}s | {r['improvement']:10.1f}%")
        
        # All sizes should show significant improvement
        for r in results:
            assert r["improvement"] > 50, f"Poor improvement for {r['files']} files: {r['improvement']:.1f}%"
    
    @pytest.mark.asyncio
    async def test_memory_usage_constant(
        self, neo4j_config, clean_neo4j
    ):
        """Verify memory usage remains constant with incremental indexing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create large repository
            files = await create_large_repository(repo_path, num_files=100, avg_lines=100)
            
            initializer = RepositoryInitializer(
                neo4j_uri=neo4j_config["uri"],
                PROJECT_WATCH_USER=neo4j_config["user"],
                PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                PROJECT_WATCH_DATABASE=neo4j_config["database"],
                repository_path=repo_path,
                project_name="perf_memory_test"
            )
            
            # Measure memory for full indexing
            memory_before_full = process.memory_info().rss / 1024 / 1024  # MB
            async with initializer:
                await initializer.initialize(persistent_monitoring=False)
            memory_after_full = process.memory_info().rss / 1024 / 1024  # MB
            memory_used_full = memory_after_full - memory_before_full
            
            # Force garbage collection
            import gc
            gc.collect()
            await asyncio.sleep(0.1)
            
            # Measure memory for incremental indexing
            memory_before_inc = process.memory_info().rss / 1024 / 1024  # MB
            async with initializer:
                await initializer.initialize(persistent_monitoring=False)
            memory_after_inc = process.memory_info().rss / 1024 / 1024  # MB
            memory_used_inc = memory_after_inc - memory_before_inc
            
            print(f"\nMemory Usage:")
            print(f"  Full indexing: {memory_used_full:.2f} MB")
            print(f"  Incremental: {memory_used_inc:.2f} MB")
            
            # Incremental should use significantly less memory
            assert memory_used_inc < memory_used_full * 0.5, \
                f"Incremental used too much memory: {memory_used_inc:.2f} MB vs {memory_used_full:.2f} MB"


@pytest.mark.performance
@pytest.mark.integration
class TestRealWorldScenarios:
    """Test performance in real-world usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_typical_development_workflow(
        self, neo4j_config, clean_neo4j
    ):
        """Simulate typical development workflow with frequent small changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create a moderate-sized repository
            files = await create_large_repository(repo_path, num_files=75, avg_lines=75)
            
            initializer = RepositoryInitializer(
                neo4j_uri=neo4j_config["uri"],
                PROJECT_WATCH_USER=neo4j_config["user"],
                PROJECT_WATCH_PASSWORD=neo4j_config["password"],
                PROJECT_WATCH_DATABASE=neo4j_config["database"],
                repository_path=repo_path,
                project_name="dev_workflow_test"
            )
            
            # Initial indexing
            async with initializer:
                await initializer.initialize(persistent_monitoring=False)
            
            workflow_times = []
            
            # Simulate 10 development iterations
            for iteration in range(10):
                # Modify 1-3 files (typical small change)
                num_changes = random.randint(1, 3)
                changed_files = random.sample(files, num_changes)
                
                await asyncio.sleep(0.05)  # Small delay
                
                for file_path in changed_files:
                    content = file_path.read_text()
                    file_path.write_text(content + f"\n# Iteration {iteration}")
                    os.utime(file_path, None)
                
                # Measure re-indexing time
                start_time = time.perf_counter()
                async with initializer:
                    result = await initializer.initialize(persistent_monitoring=False)
                iteration_time = time.perf_counter() - start_time
                
                workflow_times.append(iteration_time)
                assert result.indexed == num_changes
            
            avg_time = statistics.mean(workflow_times)
            median_time = statistics.median(workflow_times)
            
            print(f"\nDevelopment Workflow Performance:")
            print(f"  Repository size: {len(files)} files")
            print(f"  Iterations: 10")
            print(f"  Average re-index time: {avg_time:.3f}s")
            print(f"  Median re-index time: {median_time:.3f}s")
            print(f"  Min time: {min(workflow_times):.3f}s")
            print(f"  Max time: {max(workflow_times):.3f}s")
            
            # Should be fast for small changes
            assert avg_time < 2.0, f"Re-indexing too slow: {avg_time:.3f}s average"
            assert median_time < 1.5, f"Re-indexing too slow: {median_time:.3f}s median"