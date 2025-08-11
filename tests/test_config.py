"""Comprehensive tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from project_watch_mcp.config import (
    EmbeddingConfig,
    Neo4jConfig,
    ProjectConfig,
    ProjectWatchConfig,
)


class TestProjectConfig:
    """Test project configuration."""

    def test_initialization(self):
        """Test project config initialization."""
        config = ProjectConfig(name="test_project")
        assert config.name == "test_project"
        assert config.repository_path is None

    def test_validation(self):
        """Test project name validation."""
        # Valid names
        ProjectConfig(name="valid_project")
        ProjectConfig(name="project-123")
        ProjectConfig(name="project_456")

        # Invalid names
        with pytest.raises(ValueError, match="cannot be empty"):
            ProjectConfig(name="")

        with pytest.raises(ValueError, match="100 characters or less"):
            ProjectConfig(name="a" * 101)

        with pytest.raises(ValueError, match="alphanumeric"):
            ProjectConfig(name="project@123")

    def test_from_repository_path(self):
        """Test generating project name from repository path."""
        repo_path = Path("/tmp/test-repo")
        config = ProjectConfig.from_repository_path(repo_path)

        # Project name should include repo name and hash
        assert "test-repo" in config.name
        assert config.repository_path == repo_path

    def test_from_env_with_name(self):
        """Test loading from environment with explicit name."""
        with patch.dict(os.environ, {"PROJECT_NAME": "env_project"}):
            config = ProjectConfig.from_env()
            assert config.name == "env_project"

    def test_from_env_with_path(self):
        """Test loading from environment with repository path."""
        repo_path = Path("/tmp/env-repo")
        with patch.dict(os.environ, {}, clear=True):
            config = ProjectConfig.from_env(repo_path)
            assert "env-repo" in config.name
            assert config.repository_path == repo_path

    def test_from_env_default(self):
        """Test loading from environment with no data."""
        with patch.dict(os.environ, {}, clear=True):
            config = ProjectConfig.from_env()
            assert config.name == "default_project"
            assert config.repository_path is None


class TestNeo4jConfig:
    """Test Neo4j configuration."""

    def test_default_values(self):
        """Test default Neo4j configuration values."""
        config = Neo4jConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.username == "neo4j"
        assert config.password == "password"
        assert config.database == "neo4j"

    def test_custom_values(self):
        """Test custom Neo4j configuration values."""
        config = Neo4jConfig(
            uri="bolt://custom:7688",
            username="custom_user",
            password="custom_pass",
            database="custom_db",
        )
        assert config.uri == "bolt://custom:7688"
        assert config.username == "custom_user"
        assert config.password == "custom_pass"
        assert config.database == "custom_db"

    def test_from_env(self):
        """Test loading from environment variables."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://env:7687",
                "NEO4J_USERNAME": "env_user",
                "NEO4J_PASSWORD": "env_pass",
                "NEO4J_DATABASE": "env_db",
            },
        ):
            config = Neo4jConfig.from_env()
            assert config.uri == "bolt://env:7687"
            assert config.username == "env_user"
            assert config.password == "env_pass"
            assert config.database == "env_db"

    def test_from_env_with_defaults(self):
        """Test loading from environment with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = Neo4jConfig.from_env()
            assert config.uri == "bolt://localhost:7687"
            assert config.username == "neo4j"
            assert config.password == "password"
            assert config.database == "neo4j"


class TestEmbeddingConfig:
    """Test embeddings configuration."""

    def test_default_values(self):
        """Test default embeddings configuration values."""
        config = EmbeddingConfig()
        assert config.provider == "mock"
        assert config.openai_api_key is None
        assert config.openai_model == "text-embedding-3-small"
        assert config.local_api_url == "http://localhost:8080/embeddings"
        assert config.dimension == 384

    def test_openai_provider(self):
        """Test OpenAI embeddings provider configuration."""
        config = EmbeddingConfig(
            provider="openai", openai_api_key="test-key", openai_model="text-embedding-ada-002"
        )
        assert config.provider == "openai"
        assert config.openai_api_key == "test-key"
        assert config.openai_model == "text-embedding-ada-002"

    def test_local_provider(self):
        """Test local embeddings provider configuration."""
        config = EmbeddingConfig(provider="local", local_api_url="http://custom:9000")
        assert config.provider == "local"
        assert config.local_api_url == "http://custom:9000"

    def test_from_env_openai(self):
        """Test loading OpenAI config from environment."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-api-key",
                "OPENAI_EMBEDDING_MODEL": "text-embedding-ada-002",
                "EMBEDDING_DIMENSION": "1536",
            },
        ):
            config = EmbeddingConfig.from_env()
            assert config.provider == "openai"
            assert config.openai_api_key == "test-api-key"
            assert config.openai_model == "text-embedding-ada-002"
            assert config.dimension == 1536

    def test_from_env_local(self):
        """Test loading local embeddings config from environment."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_PROVIDER": "local",
                "LOCAL_EMBEDDING_API_URL": "http://custom:8080/embeddings",
            },
        ):
            config = EmbeddingConfig.from_env()
            assert config.provider == "local"
            assert config.local_api_url == "http://custom:8080/embeddings"

    def test_from_env_invalid_provider(self):
        """Test loading with invalid provider falls back to mock."""
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "invalid"}):
            config = EmbeddingConfig.from_env()
            assert config.provider == "mock"

    def test_from_env_with_defaults(self):
        """Test loading from environment with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = EmbeddingConfig.from_env()
            assert config.provider == "mock"
            assert config.openai_api_key is None
            assert config.openai_model == "text-embedding-3-small"
            assert config.local_api_url == "http://localhost:8080/embeddings"
            assert config.dimension == 384


class TestProjectWatchConfig:
    """Test main configuration class."""

    def test_initialization(self):
        """Test configuration initialization."""
        project_config = ProjectConfig(name="test_project")
        neo4j_config = Neo4jConfig(uri="bolt://test:7687")
        embedding_config = EmbeddingConfig(provider="openai")

        config = ProjectWatchConfig(
            project=project_config,
            neo4j=neo4j_config,
            embedding=embedding_config,
            chunk_size=1000,
            chunk_overlap=100,
        )

        assert config.project.name == "test_project"
        assert config.neo4j.uri == "bolt://test:7687"
        assert config.embedding.provider == "openai"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 100

    def test_from_env(self):
        """Test loading complete config from environment."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://env:7687",
                "NEO4J_USERNAME": "env_user",
                "NEO4J_PASSWORD": "env_pass",
                "NEO4J_DATABASE": "env_db",
                "EMBEDDING_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "CHUNK_SIZE": "1000",
                "CHUNK_OVERLAP": "200",
            },
        ):
            config = ProjectWatchConfig.from_env()

            # Check project config
            assert config.project.name == "default_project"

            # Check Neo4j config
            assert config.neo4j.uri == "bolt://env:7687"
            assert config.neo4j.username == "env_user"
            assert config.neo4j.password == "env_pass"
            assert config.neo4j.database == "env_db"

            # Check embedding config
            assert config.embedding.provider == "openai"
            assert config.embedding.openai_api_key == "test-key"

            # Check chunk config
            assert config.chunk_size == 1000
            assert config.chunk_overlap == 200

    def test_from_env_with_defaults(self):
        """Test loading config from environment with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = ProjectWatchConfig.from_env()

            # Check project defaults
            assert config.project.name == "default_project"

            # Check Neo4j defaults
            assert config.neo4j.uri == "bolt://localhost:7687"
            assert config.neo4j.username == "neo4j"
            assert config.neo4j.password == "password"

            # Check embedding defaults
            assert config.embedding.provider == "mock"

            # Check chunk defaults
            assert config.chunk_size == 500
            assert config.chunk_overlap == 50

    def test_partial_env_override(self):
        """Test partial environment override."""
        with patch.dict(os.environ, {"NEO4J_URI": "bolt://partial:7687", "CHUNK_SIZE": "750"}):
            config = ProjectWatchConfig.from_env()

            # Check overridden values
            assert config.neo4j.uri == "bolt://partial:7687"
            assert config.chunk_size == 750

            # Check defaults
            assert config.neo4j.username == "neo4j"
            assert config.embedding.provider == "mock"
            assert config.chunk_overlap == 50
