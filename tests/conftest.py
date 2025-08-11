"""Shared test fixtures and configuration."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.execute_query = AsyncMock()
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def mock_embeddings():
    """Mock embeddings provider."""
    embeddings = MagicMock()
    embeddings.embed_text = AsyncMock(return_value=[0.1] * 384)
    embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 384, [0.2] * 384])
    return embeddings
