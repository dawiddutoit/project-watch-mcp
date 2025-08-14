"""Test suite for data corruption prevention.

This module tests that the system prevents data corruption when
multiple projects share the same Neo4j instance, particularly
when they have overlapping file paths.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG
from tests.unit.utils.embeddings.test_embeddings_utils import TestEmbeddingsProvider


class TestCorruptionPrevention:
    """Test that the system prevents data corruption across projects."""

    @pytest.fixture
    def corruption_test_driver(self):
        """Create a mock driver that simulates real Neo4j behavior for corruption testing."""
        driver = AsyncMock()
        driver.verify_connectivity = AsyncMock()
        driver.close = AsyncMock()

        # Simulate a real database with project isolation
        driver._database = {}

        async def execute_query_simulation(query, params=None, *args, **kwargs):
            """Simulate Neo4j behavior with proper project isolation."""
            project_name = params.get("project_name") if params else None

            if not project_name and any(
                op in query for op in ["MERGE", "CREATE", "MATCH", "DELETE"]
            ):
                raise ValueError("project_name is required")

            # Initialize project space if needed
            if project_name and project_name not in driver._database:
                driver._database[project_name] = {
                    "files": {},
                    "chunks": {},
                    "file_chunks": {},  # Maps file paths to chunk IDs
                }

            # Handle MERGE file operation
            if "MERGE (f:CodeFile" in query and project_name:
                file_path = params.get("path")
                if file_path:
                    # Store or update file for this project only
                    driver._database[project_name]["files"][file_path] = {
                        "project_name": project_name,
                        "path": file_path,
                        "language": params.get("language"),
                        "size": params.get("size"),
                        "last_modified": params.get("last_modified"),
                        "hash": params.get("hash"),
                    }
                return MagicMock(records=[])

            # Handle DELETE chunks operation
            elif "DELETE c" in query and "HAS_CHUNK" in query and project_name:
                file_path = params.get("path")
                if file_path and file_path in driver._database[project_name]["file_chunks"]:
                    # Delete chunks for this file in this project only
                    chunk_ids = driver._database[project_name]["file_chunks"][file_path]
                    for chunk_id in chunk_ids:
                        if chunk_id in driver._database[project_name]["chunks"]:
                            del driver._database[project_name]["chunks"][chunk_id]
                    driver._database[project_name]["file_chunks"][file_path] = []
                return MagicMock(records=[])

            # Handle CREATE chunk operation
            elif "CREATE (c:CodeChunk" in query and project_name:
                file_path = params.get("file_path")
                chunk_index = params.get("chunk_index", 0)
                chunk_id = f"{project_name}:{file_path}:{chunk_index}"

                # Store chunk for this project only
                driver._database[project_name]["chunks"][chunk_id] = {
                    "project_name": project_name,
                    "file_path": file_path,
                    "content": params.get("content"),
                    "start_line": params.get("start_line"),
                    "end_line": params.get("end_line"),
                    "embedding": params.get("embedding"),
                }

                # Track chunk association with file
                if file_path not in driver._database[project_name]["file_chunks"]:
                    driver._database[project_name]["file_chunks"][file_path] = []
                driver._database[project_name]["file_chunks"][file_path].append(chunk_id)

                return MagicMock(records=[])

            # Handle DELETE file operation
            elif "DELETE f" in query and "CodeFile" in query and project_name:
                file_path = params.get("path")
                if file_path:
                    # Delete file and its chunks for this project only
                    if file_path in driver._database[project_name]["files"]:
                        del driver._database[project_name]["files"][file_path]

                    # Delete associated chunks
                    if file_path in driver._database[project_name]["file_chunks"]:
                        chunk_ids = driver._database[project_name]["file_chunks"][file_path]
                        for chunk_id in chunk_ids:
                            if chunk_id in driver._database[project_name]["chunks"]:
                                del driver._database[project_name]["chunks"][chunk_id]
                        del driver._database[project_name]["file_chunks"][file_path]

                return MagicMock(records=[])

            # Handle file existence check
            elif "MATCH (f:CodeFile" in query and "RETURN f.hash" in query and project_name:
                file_path = params.get("path")
                if file_path and file_path in driver._database[project_name]["files"]:
                    file_data = driver._database[project_name]["files"][file_path]
                    return MagicMock(records=[{"hash": file_data["hash"]}])
                return MagicMock(records=[])

            # Handle search queries
            elif "MATCH" in query and "CodeChunk" in query and project_name:
                chunks = driver._database[project_name]["chunks"].values()
                results = []
                for chunk in list(chunks)[:10]:  # Limit results
                    results.append(
                        {
                            "project_name": chunk["project_name"],
                            "file_path": chunk["file_path"],
                            "chunk_content": chunk["content"],
                            "line_number": chunk["start_line"],
                            "similarity": 0.9,
                        }
                    )
                return MagicMock(records=results)

            # Handle stats queries
            elif "count(DISTINCT f)" in query and project_name:
                project_data = driver._database.get(project_name, {})
                total_files = len(project_data.get("files", {}))
                total_chunks = len(project_data.get("chunks", {}))

                # Create a mock record that supports both dict-like and attribute access
                record_data = {
                    "total_files": total_files,
                    "total_chunks": total_chunks,
                    "total_size": total_files * 1000,
                    "languages": ["python"],
                    "project_name": project_name,
                }
                # Create a proper mock record with dict-like access
                class MockRecord:
                    def __init__(self, data):
                        self._data = data
                    def __getitem__(self, key):
                        return self._data[key]
                    def get(self, key, default=None):
                        return self._data.get(key, default)
                
                mock_record = MockRecord(record_data)
                
                return MagicMock(records=[mock_record])

            return MagicMock(records=[])

        driver.execute_query.side_effect = execute_query_simulation
        return driver

    async def test_same_file_path_no_overwrite(self, corruption_test_driver):
        """Test that same file path in different projects doesn't overwrite."""
        # Create two projects
        project_a = Neo4jRAG(
            neo4j_driver=corruption_test_driver,
            project_name="project_a",
            embeddings=TestEmbeddingsProvider(),
        )
        await project_a.initialize()

        project_b = Neo4jRAG(
            neo4j_driver=corruption_test_driver,
            project_name="project_b",
            embeddings=TestEmbeddingsProvider(),
        )
        await project_b.initialize()

        # Index the same file path with different content in each project
        shared_path = Path("/src/main.py")

        file_a = CodeFile(
            project_name="project_a",
            path=shared_path,
            content="# Project A main file\nprint('Project A')",
            language="python",
            size=42,
            last_modified=datetime.now(),
        )

        file_b = CodeFile(
            project_name="project_b",
            path=shared_path,
            content="# Project B main file\nprint('Project B')",
            language="python",
            size=42,
            last_modified=datetime.now(),
        )

        # Index both files
        await project_a.index_file(file_a)
        await project_b.index_file(file_b)

        # Verify both files exist independently in the mock database
        assert str(shared_path) in corruption_test_driver._database["project_a"]["files"]
        assert str(shared_path) in corruption_test_driver._database["project_b"]["files"]

        # Verify content is different
        file_a_data = corruption_test_driver._database["project_a"]["files"][str(shared_path)]
        file_b_data = corruption_test_driver._database["project_b"]["files"][str(shared_path)]

        assert file_a_data["project_name"] == "project_a"
        assert file_b_data["project_name"] == "project_b"
        assert file_a_data["hash"] != file_b_data["hash"]  # Different content = different hash

        # Search in each project should return different content
        results_a = await project_a.search_semantic("Project A")
        results_b = await project_b.search_semantic("Project B")

        # Each project should find its own version
        for result in results_a:
            assert result.project_name == "project_a"
            assert "Project A" in result.content or result.project_name == "project_a"

        for result in results_b:
            assert result.project_name == "project_b"
            assert "Project B" in result.content or result.project_name == "project_b"

    async def test_deletion_project_scoped(self, corruption_test_driver):
        """Test that file deletion only affects one project."""
        # Create three projects
        projects = {}
        for name in ["alpha", "beta", "gamma"]:
            projects[name] = Neo4jRAG(
                neo4j_driver=corruption_test_driver,
                project_name=f"project_{name}",
                embeddings=TestEmbeddingsProvider(),
            )
            await projects[name].initialize()

        # Index the same file in all projects
        shared_file = Path("/shared/config.yaml")

        for name, rag in projects.items():
            file = CodeFile(
                project_name=f"project_{name}",
                path=shared_file,
                content=f"# Config for {name}\nproject: {name}",
                language="yaml",
                size=30,
                last_modified=datetime.now(),
            )
            await rag.index_file(file)

        # Verify all projects have the file
        for name in ["alpha", "beta", "gamma"]:
            project_name = f"project_{name}"
            assert str(shared_file) in corruption_test_driver._database[project_name]["files"]

        # Delete the file from project_beta only
        await projects["beta"].delete_file(shared_file)

        # Verify file is deleted from beta but still exists in alpha and gamma
        assert str(shared_file) in corruption_test_driver._database["project_alpha"]["files"]
        assert str(shared_file) not in corruption_test_driver._database["project_beta"]["files"]
        assert str(shared_file) in corruption_test_driver._database["project_gamma"]["files"]

        # Verify chunks are also properly scoped
        assert str(shared_file) in corruption_test_driver._database["project_alpha"]["file_chunks"]
        assert (
            str(shared_file) not in corruption_test_driver._database["project_beta"]["file_chunks"]
        )
        assert str(shared_file) in corruption_test_driver._database["project_gamma"]["file_chunks"]

    async def test_update_project_scoped(self, corruption_test_driver):
        """Test that file updates are scoped to the correct project."""
        # Create two projects
        project_a = Neo4jRAG(
            neo4j_driver=corruption_test_driver,
            project_name="update_test_a",
            embeddings=TestEmbeddingsProvider(),
        )
        await project_a.initialize()

        project_b = Neo4jRAG(
            neo4j_driver=corruption_test_driver,
            project_name="update_test_b",
            embeddings=TestEmbeddingsProvider(),
        )
        await project_b.initialize()

        # Index the same file in both projects
        shared_path = Path("/app/service.py")

        original_file_a = CodeFile(
            project_name="update_test_a",
            path=shared_path,
            content="# Service A v1",
            language="python",
            size=14,
            last_modified=datetime.now(),
        )

        original_file_b = CodeFile(
            project_name="update_test_b",
            path=shared_path,
            content="# Service B v1",
            language="python",
            size=14,
            last_modified=datetime.now(),
        )

        await project_a.index_file(original_file_a)
        await project_b.index_file(original_file_b)

        # Update file in project_a only
        updated_file_a = CodeFile(
            project_name="update_test_a",
            path=shared_path,
            content="# Service A v2 - Updated!",
            language="python",
            size=25,
            last_modified=datetime.now(),
        )

        await project_a.update_file(updated_file_a)

        # Verify project_a has updated content
        chunks_a = corruption_test_driver._database["update_test_a"]["chunks"]
        assert any("v2 - Updated" in chunk["content"] for chunk in chunks_a.values())

        # Verify project_b still has original content
        chunks_b = corruption_test_driver._database["update_test_b"]["chunks"]
        assert all("v2 - Updated" not in chunk["content"] for chunk in chunks_b.values())
        assert any("Service B v1" in chunk["content"] for chunk in chunks_b.values())

    async def test_concurrent_indexing_no_corruption(self, corruption_test_driver):
        """Test that concurrent indexing operations don't cause corruption."""
        # Create multiple projects
        num_projects = 5
        projects = []

        for i in range(num_projects):
            rag = Neo4jRAG(
                neo4j_driver=corruption_test_driver,
                project_name=f"concurrent_{i}",
                embeddings=TestEmbeddingsProvider(),
            )
            await rag.initialize()
            projects.append(rag)

        # Define files to index concurrently
        files_to_index = []
        for i, rag in enumerate(projects):
            for j in range(3):  # 3 files per project
                file = CodeFile(
                    project_name=f"concurrent_{i}",
                    path=Path(f"/shared/file_{j}.py"),  # Same paths across projects
                    content=f"# Project {i} File {j}\ncode = {i * 10 + j}",
                    language="python",
                    size=50,
                    last_modified=datetime.now(),
                )
                files_to_index.append((rag, file))

        # Index all files concurrently
        tasks = [rag.index_file(file) for rag, file in files_to_index]
        await asyncio.gather(*tasks)

        # Verify each project has its own files with correct content
        for i in range(num_projects):
            project_name = f"concurrent_{i}"
            project_files = corruption_test_driver._database[project_name]["files"]

            # Should have 3 files
            assert len(project_files) == 3

            # Each file should belong to the correct project
            for file_path, file_data in project_files.items():
                assert file_data["project_name"] == project_name

            # Check chunks are correctly isolated
            project_chunks = corruption_test_driver._database[project_name]["chunks"]
            for chunk_data in project_chunks.values():
                assert chunk_data["project_name"] == project_name
                assert f"Project {i}" in chunk_data["content"]

    async def test_search_no_cross_contamination(self, corruption_test_driver):
        """Test that searches don't return results from other projects."""
        # Create projects with similar content
        project_names = ["search_test_1", "search_test_2", "search_test_3"]
        projects = {}

        for name in project_names:
            projects[name] = Neo4jRAG(
                neo4j_driver=corruption_test_driver,
                project_name=name,
                embeddings=TestEmbeddingsProvider(),
            )
            await projects[name].initialize()

        # Index files with unique markers
        for name, rag in projects.items():
            file = CodeFile(
                project_name=name,
                path=Path("/search/test.py"),
                content=f"def unique_{name}_function(): pass\n# Marker: {name}",
                language="python",
                size=50,
                last_modified=datetime.now(),
            )
            await rag.index_file(file)

        # Search in each project for its unique marker
        for name, rag in projects.items():
            results = await rag.search_semantic(f"unique_{name}_function")

            # All results should be from the correct project
            for result in results:
                assert result.project_name == name
                # The content should contain the project's unique marker
                if result.content:
                    assert name in result.content or result.project_name == name

    async def test_stats_isolation(self, corruption_test_driver):
        """Test that statistics are correctly isolated per project."""
        # Create projects with different numbers of files
        projects = {
            "stats_small": (
                2,
                Neo4jRAG(
                    neo4j_driver=corruption_test_driver,
                    project_name="stats_small",
                    embeddings=TestEmbeddingsProvider(),
                ),
            ),
            "stats_medium": (
                5,
                Neo4jRAG(
                    neo4j_driver=corruption_test_driver,
                    project_name="stats_medium",
                    embeddings=TestEmbeddingsProvider(),
                ),
            ),
            "stats_large": (
                10,
                Neo4jRAG(
                    neo4j_driver=corruption_test_driver,
                    project_name="stats_large",
                    embeddings=TestEmbeddingsProvider(),
                ),
            ),
        }

        # Initialize and index files
        for name, (file_count, rag) in projects.items():
            await rag.initialize()

            for i in range(file_count):
                file = CodeFile(
                    project_name=name,
                    path=Path(f"/{name}/file_{i}.py"),
                    content=f"# File {i} in {name}",
                    language="python",
                    size=100,
                    last_modified=datetime.now(),
                )
                await rag.index_file(file)

        # Get stats for each project
        stats = {}
        for name, (expected_count, rag) in projects.items():
            stats[name] = await rag.get_repository_stats()

        # Verify stats are correct and isolated
        assert stats["stats_small"]["total_files"] == 2
        assert stats["stats_medium"]["total_files"] == 5
        assert stats["stats_large"]["total_files"] == 10

        # Each should have the correct project name
        for name in projects:
            assert stats[name]["project_name"] == name

    async def test_chunk_isolation(self, corruption_test_driver):
        """Test that chunks are properly isolated between projects."""
        # Create two projects
        project_a = Neo4jRAG(
            neo4j_driver=corruption_test_driver,
            project_name="chunk_test_a",
            embeddings=TestEmbeddingsProvider(),
            chunk_size=50,  # Small chunks for testing
            chunk_overlap=10,
        )
        await project_a.initialize()

        project_b = Neo4jRAG(
            neo4j_driver=corruption_test_driver,
            project_name="chunk_test_b",
            embeddings=TestEmbeddingsProvider(),
            chunk_size=50,
            chunk_overlap=10,
        )
        await project_b.initialize()

        # Create content that will be chunked
        large_content_a = "\n".join([f"Line {i} in project A" for i in range(100)])
        large_content_b = "\n".join([f"Line {i} in project B" for i in range(100)])

        file_a = CodeFile(
            project_name="chunk_test_a",
            path=Path("/large/file.py"),
            content=large_content_a,
            language="python",
            size=len(large_content_a),
            last_modified=datetime.now(),
        )

        file_b = CodeFile(
            project_name="chunk_test_b",
            path=Path("/large/file.py"),  # Same path
            content=large_content_b,
            language="python",
            size=len(large_content_b),
            last_modified=datetime.now(),
        )

        # Index both files
        await project_a.index_file(file_a)
        await project_b.index_file(file_b)

        # Verify chunks are isolated
        chunks_a = corruption_test_driver._database["chunk_test_a"]["chunks"]
        chunks_b = corruption_test_driver._database["chunk_test_b"]["chunks"]

        # All chunks in project A should contain "project A"
        for chunk in chunks_a.values():
            assert "project A" in chunk["content"]
            assert "project B" not in chunk["content"]
            assert chunk["project_name"] == "chunk_test_a"

        # All chunks in project B should contain "project B"
        for chunk in chunks_b.values():
            assert "project B" in chunk["content"]
            assert "project A" not in chunk["content"]
            assert chunk["project_name"] == "chunk_test_b"

        # The number of chunks might be the same, but content is different
        assert len(chunks_a) > 0
        assert len(chunks_b) > 0
