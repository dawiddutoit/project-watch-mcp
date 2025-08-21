"""Shared pytest fixtures for all tests."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
import pytest
from neo4j import AsyncGraphDatabase, AsyncDriver
from testcontainers.neo4j import Neo4jContainer


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


@pytest.fixture(scope="session")
def neo4j_container() -> Generator[Neo4jContainer, None, None]:
    """Create a Neo4j test container for integration tests.
    
    This fixture provides a real Neo4j database instance in a Docker container
    for integration testing. The container is created once per test session
    and shared across all tests.
    """
    container = Neo4jContainer(image="neo4j:5-enterprise")
    container.with_env("NEO4J_ACCEPT_LICENSE_AGREEMENT", "yes")
    container.with_env("NEO4J_AUTH", "neo4j/testpassword")
    container.with_env("NEO4J_dbms_security_auth__minimum__password__length", "1")
    
    # Start the container
    container.start()
    
    # Wait for Neo4j to be ready
    import time
    max_retries = 30
    for i in range(max_retries):
        try:
            # Get connection URL and parse auth from container
            connection_url = container.get_connection_url()
            driver = AsyncGraphDatabase.driver(
                connection_url,
                auth=("neo4j", "testpassword")
            )
            # Try to connect
            asyncio.run(driver.verify_connectivity())
            asyncio.run(driver.close())
            break
        except Exception:
            if i == max_retries - 1:
                raise
            time.sleep(1)
    
    yield container
    
    # Cleanup
    container.stop()


@pytest.fixture(scope="function")
async def real_neo4j_driver(neo4j_container) -> AsyncGenerator[AsyncDriver, None]:
    """Create a real Neo4j driver connected to the test container.
    
    This fixture provides a fresh driver for each test function,
    ensuring test isolation.
    """
    # Get connection URL from container
    connection_url = neo4j_container.get_connection_url()
    driver = AsyncGraphDatabase.driver(
        connection_url,
        auth=("neo4j", "testpassword")
    )
    
    # Clear the database before each test
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
        # Drop all indexes
        result = await session.run("SHOW INDEXES")
        indexes = await result.data()
        for index in indexes:
            if index.get('name'):
                try:
                    await session.run(f"DROP INDEX {index['name']}")
                except:
                    pass  # Ignore errors for system indexes
    
    yield driver
    
    await driver.close()


@pytest.fixture
async def real_neo4j_rag(real_neo4j_driver, real_embeddings_provider):
    """Create a Neo4jRAG instance with real database and embeddings.
    
    This fixture provides a fully functional Neo4jRAG instance for
    integration testing.
    """
    from project_watch_mcp.neo4j_rag import Neo4jRAG
    
    rag = Neo4jRAG(
        neo4j_driver=real_neo4j_driver,
        project_name="test_project",
        embeddings=real_embeddings_provider
    )
    
    await rag.initialize()
    
    return rag


@pytest.fixture
def real_embeddings_provider():
    """Create a real embeddings provider for integration tests.
    
    This can be configured to use either OpenAI or a mock provider
    depending on whether API keys are available.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        # Use real OpenAI embeddings if API key is available
        from project_watch_mcp.embeddings.openai_embeddings import OpenAIEmbeddingsProvider
        return OpenAIEmbeddingsProvider(
            api_key=api_key,
            model="text-embedding-3-small"
        )
    else:
        # Use a deterministic mock for testing without API key
        from unittest.mock import AsyncMock
        mock = AsyncMock()
        
        # Generate deterministic embeddings based on input text
        async def mock_embed_documents(texts):
            embeddings = []
            for text in texts:
                # Create a deterministic embedding based on text hash
                import hashlib
                hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
                embedding = [(hash_val >> (i * 8) & 0xFF) / 255.0 for i in range(1536)]
                embeddings.append(embedding)
            return embeddings
        
        async def mock_embed_query(text):
            embeddings = await mock_embed_documents([text])
            return embeddings[0]
        
        mock.embed_documents = mock_embed_documents
        mock.embed_query = mock_embed_query
        mock.dimension = 1536
        
        return mock


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