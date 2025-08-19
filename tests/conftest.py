"""Shared pytest fixtures for all tests."""

import os
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock configuration for testing."""
    from project_watch_mcp.config import (
        ProjectWatchConfig,
        ProjectConfig, 
        Neo4jConfig,
        EmbeddingConfig
    )
    
    # Create sub-configs
    project_config = ProjectConfig(
        name="test_project",
        repository_path=temp_dir
    )
    
    neo4j_config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="testpassword",
        database="neo4j"
    )
    
    embedding_config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        api_key="test-api-key",
        dimension=1536
    )
    
    # Create main config
    config = ProjectWatchConfig(
        project=project_config,
        neo4j=neo4j_config,
        embedding=embedding_config,
        chunk_size=500,
        chunk_overlap=50
    )
    
    return config


@pytest.fixture
def neo4j_available():
    """Check if Neo4j is available for testing."""
    import socket
    try:
        # Try to connect to default Neo4j port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 7687))
        sock.close()
        return result == 0
    except:
        return False


@pytest.fixture
def openai_available():
    """Check if OpenAI API key is available for testing."""
    return os.getenv("OPENAI_API_KEY") is not None


@pytest.fixture
def voyage_available():
    """Check if Voyage API key is available for testing."""
    return os.getenv("VOYAGE_API_KEY") is not None


# Pytest markers for conditional test execution
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_neo4j: mark test as requiring Neo4j database"
    )
    config.addinivalue_line(
        "markers", "requires_openai: mark test as requiring OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "requires_voyage: mark test as requiring Voyage API key"
    )


# Skip conditions
requires_neo4j = pytest.mark.skipif(
    not os.getenv("TEST_NEO4J", "").lower() in ["true", "1", "yes"],
    reason="Neo4j tests disabled (set TEST_NEO4J=true to enable)"
)

requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not found"
)

requires_voyage = pytest.mark.skipif(
    not os.getenv("VOYAGE_API_KEY"),
    reason="Voyage API key not found"
)