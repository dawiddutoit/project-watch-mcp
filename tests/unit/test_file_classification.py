"""Tests for file classification feature in Neo4j RAG."""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from neo4j import AsyncDriver, RoutingControl

from project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG


class TestFileClassification:
    """Test file classification features."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock(spec=AsyncDriver)
        driver.execute_query = AsyncMock()
        return driver

    @pytest.fixture
    async def neo4j_rag(self, mock_neo4j_driver):
        """Create a Neo4jRAG instance with classification enabled."""
        rag = Neo4jRAG(
            neo4j_driver=mock_neo4j_driver,
            project_name="test_project",
            enable_file_classification=True,
        )
        await rag.initialize()
        return rag

    @pytest.fixture
    async def neo4j_rag_disabled(self, mock_neo4j_driver):
        """Create a Neo4jRAG instance with classification disabled."""
        rag = Neo4jRAG(
            neo4j_driver=mock_neo4j_driver,
            project_name="test_project",
            enable_file_classification=False,
        )
        await rag.initialize()
        return rag

    def test_code_file_classification(self):
        """Test that CodeFile correctly classifies different file types."""
        # Test file
        test_file = CodeFile(
            project_name="test",
            path=Path("tests/test_example.py"),
            content="def test_something(): pass",
            language="python",
            size=100,
            last_modified=datetime.now(),
        )
        assert test_file.is_test is True
        assert test_file.file_category == "test"

        # Config file
        config_file = CodeFile(
            project_name="test",
            path=Path("config.yaml"),
            content="key: value",
            language="yaml",
            size=50,
            last_modified=datetime.now(),
        )
        assert config_file.is_config is True
        assert config_file.file_category == "config"

        # Documentation file
        doc_file = CodeFile(
            project_name="test",
            path=Path("README.md"),
            content="# Documentation",
            language="markdown",
            size=200,
            last_modified=datetime.now(),
        )
        assert doc_file.is_documentation is True
        assert doc_file.file_category == "documentation"

        # Resource file
        resource_file = CodeFile(
            project_name="test",
            path=Path("data.csv"),
            content="col1,col2",
            language="csv",
            size=30,
            last_modified=datetime.now(),
        )
        assert resource_file.is_resource is True
        assert resource_file.file_category == "resource"

        # Source file
        source_file = CodeFile(
            project_name="test",
            path=Path("src/main.py"),
            content="def main(): pass",
            language="python",
            size=150,
            last_modified=datetime.now(),
        )
        assert source_file.is_test is False
        assert source_file.is_config is False
        assert source_file.is_resource is False
        assert source_file.is_documentation is False
        assert source_file.file_category == "source"

    @pytest.mark.asyncio
    async def test_index_file_with_classification(self, neo4j_rag, mock_neo4j_driver):
        """Test that index_file includes classification fields when enabled."""
        test_file = CodeFile(
            project_name="test_project",
            path=Path("tests/test_example.py"),
            content="def test_something(): pass",
            language="python",
            size=100,
            last_modified=datetime.now(),
        )

        await neo4j_rag.index_file(test_file)

        # Check that execute_query was called with classification fields
        calls = mock_neo4j_driver.execute_query.call_args_list
        
        # Find the file creation query
        file_query_call = None
        for call in calls:
            if "MERGE (f:CodeFile" in str(call[0][0]):
                file_query_call = call
                break
        
        assert file_query_call is not None
        params = file_query_call[0][1]
        
        assert params["is_test"] is True
        assert params["is_config"] is False
        assert params["is_resource"] is False
        assert params["is_documentation"] is False
        assert params["file_category"] == "test"

    @pytest.mark.asyncio
    async def test_index_file_without_classification(self, neo4j_rag_disabled, mock_neo4j_driver):
        """Test that index_file excludes classification fields when disabled."""
        test_file = CodeFile(
            project_name="test_project",
            path=Path("tests/test_example.py"),
            content="def test_something(): pass",
            language="python",
            size=100,
            last_modified=datetime.now(),
        )

        await neo4j_rag_disabled.index_file(test_file)

        # Check that execute_query was called without classification fields
        calls = mock_neo4j_driver.execute_query.call_args_list
        
        # Find the file creation query
        file_query_call = None
        for call in calls:
            if "MERGE (f:CodeFile" in str(call[0][0]):
                file_query_call = call
                break
        
        assert file_query_call is not None
        params = file_query_call[0][1]
        
        # Classification fields should not be in params
        assert "is_test" not in params
        assert "is_config" not in params
        assert "is_resource" not in params
        assert "is_documentation" not in params
        assert "file_category" not in params

    @pytest.mark.asyncio
    async def test_create_indexes_with_classification(self, neo4j_rag, mock_neo4j_driver):
        """Test that classification indexes are created when enabled."""
        # Reset mock to clear initialization calls
        mock_neo4j_driver.execute_query.reset_mock()
        
        await neo4j_rag.create_indexes()
        
        # Check that classification indexes were created
        calls = mock_neo4j_driver.execute_query.call_args_list
        index_queries = [str(call[0][0]) for call in calls]
        
        # Check for classification-specific indexes
        assert any("project_file_category_index" in q for q in index_queries)
        assert any("project_is_test_index" in q for q in index_queries)
        assert any("project_is_config_index" in q for q in index_queries)
        assert any("project_is_documentation_index" in q for q in index_queries)
        assert any("project_is_resource_index" in q for q in index_queries)
        assert any("project_namespace_index" in q for q in index_queries)

    @pytest.mark.asyncio
    async def test_create_indexes_without_classification(self, neo4j_rag_disabled, mock_neo4j_driver):
        """Test that classification indexes are not created when disabled."""
        # Reset mock to clear initialization calls
        mock_neo4j_driver.execute_query.reset_mock()
        
        await neo4j_rag_disabled.create_indexes()
        
        # Check that classification indexes were NOT created
        calls = mock_neo4j_driver.execute_query.call_args_list
        index_queries = [str(call[0][0]) for call in calls]
        
        # Check that classification-specific indexes are absent
        assert not any("project_file_category_index" in q for q in index_queries)
        assert not any("project_is_test_index" in q for q in index_queries)
        assert not any("project_is_config_index" in q for q in index_queries)

    @pytest.mark.asyncio
    async def test_search_with_file_category_filter(self, neo4j_rag, mock_neo4j_driver):
        """Test search methods with file category filtering."""
        # Mock search results
        mock_results = MagicMock()
        mock_results.records = [
            {
                "file_path": "tests/test_example.py",
                "content": "test content",
                "line_number": 1,
                "similarity": 0.9,
                "project_name": "test_project",
            }
        ]
        mock_neo4j_driver.execute_query.return_value = mock_results

        # Test pattern search with file category filter
        results = await neo4j_rag.search_by_pattern(
            pattern="test",
            file_category="test",
            is_test=True,
        )

        # Check that the query included category filters
        calls = mock_neo4j_driver.execute_query.call_args_list
        last_call = calls[-1]
        params = last_call[0][1]
        
        assert params.get("file_category") == "test"
        assert params.get("is_test") is True

    @pytest.mark.asyncio
    async def test_list_files_by_category(self, neo4j_rag, mock_neo4j_driver):
        """Test listing files by category."""
        # Mock query results
        mock_results = MagicMock()
        mock_results.records = [
            {
                "path": "tests/test_example.py",
                "language": "python",
                "size": 100,
                "lines": 10,
                "category": "test",
                "is_test": True,
                "is_config": False,
                "is_documentation": False,
                "is_resource": False,
                "namespace": None,
            },
            {
                "path": "tests/test_another.py",
                "language": "python",
                "size": 200,
                "lines": 20,
                "category": "test",
                "is_test": True,
                "is_config": False,
                "is_documentation": False,
                "is_resource": False,
                "namespace": None,
            },
        ]
        mock_neo4j_driver.execute_query.return_value = mock_results

        # List test files
        files = await neo4j_rag.list_files_by_category(category="test")

        assert len(files) == 2
        assert all(f["category"] == "test" for f in files)
        assert all(f["is_test"] is True for f in files)

        # Verify query parameters
        last_call = mock_neo4j_driver.execute_query.call_args_list[-1]
        params = last_call[0][1]
        assert params["category"] == "test"

    @pytest.mark.asyncio
    async def test_list_files_by_category_disabled(self, neo4j_rag_disabled):
        """Test that list_files_by_category returns empty list when disabled."""
        files = await neo4j_rag_disabled.list_files_by_category(category="test")
        assert files == []

    @pytest.mark.asyncio
    async def test_batch_index_with_classification(self, neo4j_rag, mock_neo4j_driver):
        """Test batch indexing with classification fields."""
        test_files = [
            CodeFile(
                project_name="test_project",
                path=Path("tests/test_1.py"),
                content="def test_one(): pass",
                language="python",
                size=50,
                last_modified=datetime.now(),
            ),
            CodeFile(
                project_name="test_project",
                path=Path("config.yaml"),
                content="key: value",
                language="yaml",
                size=30,
                last_modified=datetime.now(),
            ),
            CodeFile(
                project_name="test_project",
                path=Path("src/main.py"),
                content="def main(): pass",
                language="python",
                size=100,
                last_modified=datetime.now(),
            ),
        ]

        await neo4j_rag.batch_index_files(test_files)

        # Find the batch upsert query
        calls = mock_neo4j_driver.execute_query.call_args_list
        batch_query_call = None
        for call in calls:
            if "UNWIND $files as file" in str(call[0][0]):
                batch_query_call = call
                break

        assert batch_query_call is not None
        params = batch_query_call[0][1]
        files_data = params["files"]

        # Check first file (test)
        assert files_data[0]["is_test"] is True
        assert files_data[0]["file_category"] == "test"

        # Check second file (config)
        assert files_data[1]["is_config"] is True
        assert files_data[1]["file_category"] == "config"

        # Check third file (source)
        assert files_data[2]["is_test"] is False
        assert files_data[2]["is_config"] is False
        assert files_data[2]["file_category"] == "source"

    def test_namespace_extraction(self):
        """Test namespace extraction for different languages."""
        # Python namespace
        python_file = CodeFile(
            project_name="test",
            path=Path("src/module/submodule/file.py"),
            content="class MyClass: pass",
            language="python",
            size=100,
            last_modified=datetime.now(),
        )
        assert python_file.namespace == "module.submodule"

        # Java namespace
        java_file = CodeFile(
            project_name="test",
            path=Path("src/main/java/com/example/MyClass.java"),
            content="package com.example;\nclass MyClass {}",
            language="java",
            size=100,
            last_modified=datetime.now(),
        )
        assert java_file.namespace == "com.example"

        # TypeScript namespace
        ts_file = CodeFile(
            project_name="test",
            path=Path("src/components/Button.tsx"),
            content="export const Button = () => {}",
            language="typescript",
            size=100,
            last_modified=datetime.now(),
        )
        assert ts_file.namespace == "components"