"""Test suite for embedding provider switching functionality."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import AsyncDriver

from project_watch_mcp.config import EmbeddingConfig
from project_watch_mcp.neo4j_rag import Neo4jRAG
from project_watch_mcp.utils.embeddings import (
    OpenAIEmbeddingsProvider,
    VoyageEmbeddingsProvider,
    create_embeddings_provider,
)
from tests.unit.utils.embeddings.embeddings_test_utils import MockEmbeddingsProvider


class TestEmbeddingProviderFactory:
    """Test the create_embeddings_provider factory function."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider via factory."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = create_embeddings_provider("openai")
            assert isinstance(provider, OpenAIEmbeddingsProvider)
            assert provider.dimension == 1536

    def test_create_voyage_provider(self):
        """Test creating Voyage provider via factory."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            provider = create_embeddings_provider("voyage")
            assert isinstance(provider, VoyageEmbeddingsProvider)
            assert provider.dimension == 1024  # voyage-code-3 default

    def test_create_voyage_provider_with_model(self):
        """Test creating Voyage provider with specific model."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            provider = create_embeddings_provider("voyage", model="voyage-3-lite")
            assert isinstance(provider, VoyageEmbeddingsProvider)
            assert provider.dimension == 512  # voyage-3-lite dimension

    def test_create_mock_provider(self):
        """Test creating Mock provider via factory."""
        provider = create_embeddings_provider("mock")
        assert isinstance(provider, MockEmbeddingsProvider)
        assert provider.dimension == 1536

    def test_create_mock_provider_with_custom_dimension(self):
        """Test creating Mock provider with custom dimension."""
        provider = create_embeddings_provider("mock", dimension=768)
        assert isinstance(provider, MockEmbeddingsProvider)
        assert provider.dimension == 768

    def test_fallback_to_mock_on_missing_api_key(self):
        """Test fallback to mock provider when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            # OpenAI should fall back to mock
            provider = create_embeddings_provider("openai")
            assert isinstance(provider, MockEmbeddingsProvider)

            # Voyage should fall back to mock
            provider = create_embeddings_provider("voyage")
            assert isinstance(provider, MockEmbeddingsProvider)

    def test_invalid_provider_type(self):
        """Test error on invalid provider type."""
        with pytest.raises(ValueError, match="Unknown provider type: invalid"):
            create_embeddings_provider("invalid")


class TestNeo4jRAGProviderIntegration:
    """Test Neo4jRAG integration with different embedding providers."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Mock Neo4j driver for testing."""
        driver = MagicMock(spec=AsyncDriver)
        driver.execute_query = AsyncMock()
        driver.close = AsyncMock()
        return driver

    def test_neo4j_rag_with_openai_provider(self, mock_neo4j_driver):
        """Test Neo4jRAG initialization with OpenAI provider."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            openai_provider = OpenAIEmbeddingsProvider()
            rag = Neo4jRAG(
                neo4j_driver=mock_neo4j_driver,
                project_name="test-project",
                embeddings=openai_provider,
            )
            assert rag.embeddings == openai_provider
            assert rag.embeddings.dimension == 1536

    def test_neo4j_rag_with_voyage_provider(self, mock_neo4j_driver):
        """Test Neo4jRAG initialization with Voyage provider."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            voyage_provider = VoyageEmbeddingsProvider()
            rag = Neo4jRAG(
                neo4j_driver=mock_neo4j_driver,
                project_name="test-project",
                embeddings=voyage_provider,
            )
            assert rag.embeddings == voyage_provider
            assert rag.embeddings.dimension == 1024

    def test_neo4j_rag_with_mock_provider(self, mock_neo4j_driver):
        """Test Neo4jRAG initialization with mock provider."""
        mock_provider = MockEmbeddingsProvider(dimension=768)
        rag = Neo4jRAG(
            neo4j_driver=mock_neo4j_driver, project_name="test-project", embeddings=mock_provider
        )
        assert rag.embeddings == mock_provider
        assert rag.embeddings.dimension == 768

    @pytest.mark.asyncio
    async def test_neo4j_rag_creates_correct_vector_index_dimension(self, mock_neo4j_driver):
        """Test that Neo4jRAG creates vector index with correct dimension for provider."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            voyage_provider = VoyageEmbeddingsProvider()
            rag = Neo4jRAG(
                neo4j_driver=mock_neo4j_driver,
                project_name="test-project",
                embeddings=voyage_provider,
            )

            await rag.create_indexes()

            # Check that vector index was created with correct dimension
            calls = mock_neo4j_driver.execute_query.call_args_list
            vector_index_call = None
            for call in calls:
                if "CREATE VECTOR INDEX" in str(call):
                    vector_index_call = call
                    break

            assert vector_index_call is not None
            query = vector_index_call[0][0]
            assert "vector.dimensions`: 1024" in query  # Voyage dimension


class TestEmbeddingConfigIntegration:
    """Test EmbeddingConfig integration with providers."""

    def test_embedding_config_openai(self):
        """Test EmbeddingConfig for OpenAI provider."""
        config = EmbeddingConfig(provider="openai", model="text-embedding-3-small", dimension=1536)
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.dimension == 1536

    def test_embedding_config_voyage(self):
        """Test EmbeddingConfig for Voyage provider."""
        config = EmbeddingConfig(provider="voyage", model="voyage-code-3", dimension=1024)
        assert config.provider == "voyage"
        assert config.model == "voyage-code-3"
        assert config.dimension == 1024

    def test_embedding_config_to_provider(self):
        """Test converting EmbeddingConfig to provider instance."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            config = EmbeddingConfig(provider="voyage", model="voyage-code-3")

            provider = create_embeddings_provider(config.provider, model=config.model)

            assert isinstance(provider, VoyageEmbeddingsProvider)
            assert provider.model == "voyage-code-3"
            assert provider.dimension == 1024


class TestProviderSwitchingScenarios:
    """Test real-world provider switching scenarios."""

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Mock Neo4j driver for testing."""
        driver = MagicMock(spec=AsyncDriver)
        driver.execute_query = AsyncMock()
        driver.close = AsyncMock()
        return driver

    @pytest.mark.asyncio
    async def test_switch_from_openai_to_voyage(self, mock_neo4j_driver):
        """Test switching from OpenAI to Voyage provider."""
        # Start with OpenAI
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            openai_rag = Neo4jRAG(
                neo4j_driver=mock_neo4j_driver,
                project_name="test-project",
                embeddings=OpenAIEmbeddingsProvider(),
            )
            assert openai_rag.embeddings.dimension == 1536

        # Switch to Voyage
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            voyage_rag = Neo4jRAG(
                neo4j_driver=mock_neo4j_driver,
                project_name="test-project",
                embeddings=VoyageEmbeddingsProvider(),
            )
            assert voyage_rag.embeddings.dimension == 1024

    @pytest.mark.asyncio
    async def test_benchmark_different_providers(self, mock_neo4j_driver):
        """Test benchmarking setup with different providers."""
        providers = []

        # Setup Mock provider for baseline
        providers.append(("mock", MockEmbeddingsProvider()))

        # Verify each provider can be used
        for name, provider in providers:
            rag = Neo4jRAG(
                neo4j_driver=mock_neo4j_driver,
                project_name=f"benchmark-{name}",
                embeddings=provider,
            )
            assert rag.embeddings == provider

            # Test embedding generation
            test_text = "def hello_world(): return 'Hello, World!'"
            embedding = await provider.embed_text(test_text)
            assert isinstance(embedding, list)
            assert len(embedding) == provider.dimension

        # Test that OpenAI and Voyage providers can be instantiated
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            openai_provider = OpenAIEmbeddingsProvider()
            assert openai_provider.dimension == 1536

        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            voyage_provider = VoyageEmbeddingsProvider()
            assert voyage_provider.dimension == 1024
