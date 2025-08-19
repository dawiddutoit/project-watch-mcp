"""Test suite for multi-project isolation and data integrity.

This module tests that multiple projects can share the same Neo4j instance
without data corruption or cross-contamination.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG
from tests.unit.utils.embeddings.embeddings_test_utils import MockEmbeddingsProvider


class TestMultiProjectIsolation:
    """Test that multiple projects remain isolated in shared Neo4j instance."""

    @pytest.fixture
    def shared_neo4j_driver(self):
        """Create a shared mock Neo4j driver for testing multiple projects."""
        driver = AsyncMock()
        driver.verify_connectivity = AsyncMock()
        driver.execute_query = AsyncMock()
        driver.close = AsyncMock()

        # Track all indexed data by project
        driver._data = {}

        async def execute_query_side_effect(query, params=None, *args, **kwargs):
            """Simulate Neo4j query execution with project isolation."""
            project_name = params.get("project_name") if params else None

            if not project_name:
                # Queries without project_name should fail in a real scenario
                raise ValueError("project_name is required for all queries")

            # Initialize project data if needed
            if project_name not in driver._data:
                driver._data[project_name] = {
                    "files": {},
                    "chunks": [],
                    "stats": {"total_files": 0, "total_chunks": 0},
                }

            # Handle different query patterns
            if "MERGE (f:CodeFile" in query:
                # Store file data
                file_path = str(params.get("path", ""))
                driver._data[project_name]["files"][file_path] = {
                    "language": params.get("language"),
                    "size": params.get("size"),
                    "hash": params.get("hash"),
                    "project_name": project_name,
                }
                driver._data[project_name]["stats"]["total_files"] = len(
                    driver._data[project_name]["files"]
                )
                return MagicMock(records=[])

            elif "CREATE (c:CodeChunk" in query:
                # Store chunk data
                driver._data[project_name]["chunks"].append(
                    {
                        "file_path": params.get("file_path"),
                        "content": params.get("content"),
                        "project_name": project_name,
                        "embedding": params.get("embedding"),
                    }
                )
                driver._data[project_name]["stats"]["total_chunks"] = len(
                    driver._data[project_name]["chunks"]
                )
                return MagicMock(records=[])

            elif "MATCH (f:CodeFile)" in query and "count(DISTINCT f)" in query.lower():
                # Return stats for project
                stats = driver._data[project_name]["stats"]
                return MagicMock(
                    records=[
                        {
                            "total_files": stats["total_files"],
                            "total_chunks": stats["total_chunks"],
                            "total_size": 1000 * stats["total_files"],
                            "languages": ["python"],
                            "project_name": project_name,
                        }
                    ]
                )

            elif "MATCH (c:CodeChunk)" in query and "similarity" in query.lower():
                # Search chunks only from specified project
                chunks = driver._data[project_name]["chunks"]
                results = []
                for chunk in chunks[:10]:  # Limit results
                    results.append(
                        {
                            "file_path": chunk["file_path"],
                            "chunk_content": chunk["content"],
                            "line_number": 1,
                            "similarity": 0.9,
                            "project_name": project_name,
                        }
                    )
                return MagicMock(records=results)

            elif "DELETE" in query:
                # Delete file from project
                if params and "path" in params:
                    file_path = params["path"]
                    if file_path in driver._data[project_name]["files"]:
                        del driver._data[project_name]["files"][file_path]
                        # Remove associated chunks
                        driver._data[project_name]["chunks"] = [
                            c
                            for c in driver._data[project_name]["chunks"]
                            if c["file_path"] != file_path
                        ]
                        # Update stats
                        driver._data[project_name]["stats"]["total_files"] = len(
                            driver._data[project_name]["files"]
                        )
                        driver._data[project_name]["stats"]["total_chunks"] = len(
                            driver._data[project_name]["chunks"]
                        )
                return MagicMock(records=[])

            return MagicMock(records=[])

        driver.execute_query.side_effect = execute_query_side_effect
        return driver

    @pytest_asyncio.fixture
    async def project_alpha(self, shared_neo4j_driver):
        """Create RAG instance for project Alpha."""
        rag = Neo4jRAG(
            neo4j_driver=shared_neo4j_driver,
            project_name="project_alpha",
            embeddings=MockEmbeddingsProvider(),
            chunk_size=100,
            chunk_overlap=20,
        )
        await rag.initialize()
        return rag

    @pytest_asyncio.fixture
    async def project_beta(self, shared_neo4j_driver):
        """Create RAG instance for project Beta."""
        rag = Neo4jRAG(
            neo4j_driver=shared_neo4j_driver,
            project_name="project_beta",
            embeddings=MockEmbeddingsProvider(),
            chunk_size=100,
            chunk_overlap=20,
        )
        await rag.initialize()
        return rag

    @pytest_asyncio.fixture
    async def project_gamma(self, shared_neo4j_driver):
        """Create RAG instance for project Gamma."""
        rag = Neo4jRAG(
            neo4j_driver=shared_neo4j_driver,
            project_name="project_gamma",
            embeddings=MockEmbeddingsProvider(),
            chunk_size=100,
            chunk_overlap=20,
        )
        await rag.initialize()
        return rag

    async def test_data_isolation_between_projects(
        self, project_alpha, project_beta, shared_neo4j_driver
    ):
        """Test that data from different projects remains completely isolated."""
        # Index different files in each project
        file_alpha = CodeFile(
            project_name="project_alpha",
            path=Path("/src/main.py"),
            content="# Alpha project code\ndef alpha_function(): pass",
            language="python",
            size=42,
            last_modified=datetime.now(),
        )

        file_beta = CodeFile(
            project_name="project_beta",
            path=Path("/src/utils.py"),
            content="# Beta project code\ndef beta_function(): pass",
            language="python",
            size=40,
            last_modified=datetime.now(),
        )

        await project_alpha.index_file(file_alpha)
        await project_beta.index_file(file_beta)

        # Verify data is isolated in the mock driver
        assert "/src/main.py" in shared_neo4j_driver._data["project_alpha"]["files"]
        assert "/src/utils.py" not in shared_neo4j_driver._data["project_alpha"]["files"]

        assert "/src/utils.py" in shared_neo4j_driver._data["project_beta"]["files"]
        assert "/src/main.py" not in shared_neo4j_driver._data["project_beta"]["files"]

    async def test_search_respects_project_boundaries(
        self, project_alpha, project_beta, shared_neo4j_driver
    ):
        """Test that searches only return results from the specified project."""
        # Index similar code in both projects
        common_code = "def process_data(data): return data.transform()"

        file_alpha = CodeFile(
            project_name="project_alpha",
            path=Path("/src/processor.py"),
            content=common_code,
            language="python",
            size=len(common_code),
            last_modified=datetime.now(),
        )

        file_beta = CodeFile(
            project_name="project_beta",
            path=Path("/lib/transformer.py"),
            content=common_code,
            language="python",
            size=len(common_code),
            last_modified=datetime.now(),
        )

        await project_alpha.index_file(file_alpha)
        await project_beta.index_file(file_beta)

        # Search in project_alpha
        results_alpha = await project_alpha.search_semantic("process data")

        # All results should be from project_alpha only
        for result in results_alpha:
            assert result.project_name == "project_alpha"
            assert "processor.py" in str(result.file_path)
            assert "transformer.py" not in str(result.file_path)

        # Search in project_beta
        results_beta = await project_beta.search_semantic("process data")

        # All results should be from project_beta only
        for result in results_beta:
            assert result.project_name == "project_beta"
            assert "transformer.py" in str(result.file_path)
            assert "processor.py" not in str(result.file_path)

    async def test_stats_calculated_per_project(
        self, project_alpha, project_beta, project_gamma, shared_neo4j_driver
    ):
        """Test that statistics are calculated separately for each project."""
        # Index different numbers of files in each project
        files_alpha = [
            CodeFile(
                project_name="project_alpha",
                path=Path(f"/alpha/file{i}.py"),
                content=f"# Alpha file {i}",
                language="python",
                size=100 * i,
                last_modified=datetime.now(),
            )
            for i in range(3)
        ]

        files_beta = [
            CodeFile(
                project_name="project_beta",
                path=Path(f"/beta/file{i}.py"),
                content=f"# Beta file {i}",
                language="python",
                size=200 * i,
                last_modified=datetime.now(),
            )
            for i in range(5)
        ]

        files_gamma = [
            CodeFile(
                project_name="project_gamma",
                path=Path(f"/gamma/file{i}.py"),
                content=f"# Gamma file {i}",
                language="python",
                size=300 * i,
                last_modified=datetime.now(),
            )
            for i in range(2)
        ]

        # Index all files
        for file in files_alpha:
            await project_alpha.index_file(file)
        for file in files_beta:
            await project_beta.index_file(file)
        for file in files_gamma:
            await project_gamma.index_file(file)

        # Get stats for each project
        stats_alpha = await project_alpha.get_repository_stats()
        stats_beta = await project_beta.get_repository_stats()
        stats_gamma = await project_gamma.get_repository_stats()

        # Verify stats are independent
        assert stats_alpha["project_name"] == "project_alpha"
        assert stats_alpha["total_files"] == 3

        assert stats_beta["project_name"] == "project_beta"
        assert stats_beta["total_files"] == 5

        assert stats_gamma["project_name"] == "project_gamma"
        assert stats_gamma["total_files"] == 2

        # Each project should have different totals
        assert stats_alpha["total_files"] != stats_beta["total_files"]
        assert stats_beta["total_files"] != stats_gamma["total_files"]
        assert stats_alpha["total_files"] != stats_gamma["total_files"]

    async def test_concurrent_operations_different_projects(
        self, project_alpha, project_beta, project_gamma
    ):
        """Test that concurrent operations on different projects don't interfere."""
        # Define files for each project
        files = {
            "alpha": CodeFile(
                project_name="project_alpha",
                path=Path("/concurrent/alpha.py"),
                content="# Alpha concurrent test",
                language="python",
                size=100,
                last_modified=datetime.now(),
            ),
            "beta": CodeFile(
                project_name="project_beta",
                path=Path("/concurrent/beta.py"),
                content="# Beta concurrent test",
                language="python",
                size=200,
                last_modified=datetime.now(),
            ),
            "gamma": CodeFile(
                project_name="project_gamma",
                path=Path("/concurrent/gamma.py"),
                content="# Gamma concurrent test",
                language="python",
                size=300,
                last_modified=datetime.now(),
            ),
        }

        # Run concurrent operations
        results = await asyncio.gather(
            project_alpha.index_file(files["alpha"]),
            project_beta.index_file(files["beta"]),
            project_gamma.index_file(files["gamma"]),
            project_alpha.search_semantic("concurrent"),
            project_beta.search_semantic("concurrent"),
            project_gamma.search_semantic("concurrent"),
            project_alpha.get_repository_stats(),
            project_beta.get_repository_stats(),
            project_gamma.get_repository_stats(),
            return_exceptions=True,
        )

        # Check that no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception), f"Concurrent operation failed: {result}"

    async def test_project_name_validation(self, shared_neo4j_driver):
        """Test that project names are validated and sanitized."""
        # Test valid project names
        valid_names = [
            "my-project",
            "my_project",
            "MyProject123",
            "project-123-test",
            "UPPERCASE",
            "lowercase",
            "MixedCase",
        ]

        for name in valid_names:
            rag = Neo4jRAG(
                neo4j_driver=shared_neo4j_driver,
                project_name=name,
                embeddings=MockEmbeddingsProvider(),
            )
            assert rag.project_name == name

        # Test that project names are used consistently
        special_name = "test_project-123"
        rag = Neo4jRAG(
            neo4j_driver=shared_neo4j_driver,
            project_name=special_name,
            embeddings=MockEmbeddingsProvider(),
        )

        file = CodeFile(
            project_name="different_name",  # This should be corrected
            path=Path("/test.py"),
            content="test",
            language="python",
            size=4,
            last_modified=datetime.now(),
        )

        await rag.index_file(file)

        # The RAG should enforce its own project name
        calls = shared_neo4j_driver.execute_query.call_args_list
        for call_args in calls:
            if call_args[0] and len(call_args[0]) > 1:
                params = call_args[0][1]
                if "project_name" in params:
                    assert params["project_name"] == special_name

    async def test_cross_project_contamination_prevention(
        self, project_alpha, project_beta, shared_neo4j_driver
    ):
        """Test that operations in one project don't affect another."""
        # Index a file in project_alpha
        file = CodeFile(
            project_name="project_alpha",
            path=Path("/shared/config.json"),
            content='{"setting": "alpha"}',
            language="json",
            size=20,
            last_modified=datetime.now(),
        )
        await project_alpha.index_file(file)

        # Try to search for it from project_beta
        results = await project_beta.search_semantic("setting alpha")

        # Should not find anything from project_alpha
        for result in results:
            assert result.project_name != "project_alpha"

        # Delete the file from project_alpha
        await project_alpha.delete_file(Path("/shared/config.json"))

        # Index the same path in project_beta with different content
        file_beta = CodeFile(
            project_name="project_beta",
            path=Path("/shared/config.json"),
            content='{"setting": "beta"}',
            language="json",
            size=19,
            last_modified=datetime.now(),
        )
        await project_beta.index_file(file_beta)

        # Verify project_beta has its own version
        results = await project_beta.search_semantic("setting beta")
        assert len(results) > 0
        for result in results:
            assert result.project_name == "project_beta"
            assert "beta" in result.content

    async def test_large_scale_multi_project_operations(
        self, project_alpha, project_beta, project_gamma
    ):
        """Test system behavior with many files across multiple projects."""
        # Create many files for each project
        num_files_per_project = 50

        async def index_many_files(rag, project_name, file_count):
            """Index many files for a project."""
            for i in range(file_count):
                file = CodeFile(
                    project_name=project_name,
                    path=Path(f"/{project_name}/file_{i}.py"),
                    content=f"# {project_name} file {i}\n" * 10,  # Larger content
                    language="python",
                    size=1000 + i,
                    last_modified=datetime.now(),
                )
                await rag.index_file(file)

        # Index files concurrently
        await asyncio.gather(
            index_many_files(project_alpha, "project_alpha", num_files_per_project),
            index_many_files(project_beta, "project_beta", num_files_per_project),
            index_many_files(project_gamma, "project_gamma", num_files_per_project // 2),
        )

        # Verify each project has correct file count
        stats_alpha = await project_alpha.get_repository_stats()
        stats_beta = await project_beta.get_repository_stats()
        stats_gamma = await project_gamma.get_repository_stats()

        assert stats_alpha["total_files"] == num_files_per_project
        assert stats_beta["total_files"] == num_files_per_project
        assert stats_gamma["total_files"] == num_files_per_project // 2

        # Perform searches and verify isolation
        results_alpha = await project_alpha.search_semantic("project_alpha")
        results_beta = await project_beta.search_semantic("project_beta")

        # Each search should only return results from its project
        for result in results_alpha:
            assert result.project_name == "project_alpha"
            assert "project_alpha" in str(result.file_path)

        for result in results_beta:
            assert result.project_name == "project_beta"
            assert "project_beta" in str(result.file_path)
