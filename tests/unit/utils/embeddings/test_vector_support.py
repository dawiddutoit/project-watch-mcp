"""Unit tests for vector search support utilities."""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

from src.project_watch_mcp.utils.embeddings.vector_support import (
    create_langchain_embeddings,
    create_neo4j_vector_store,
    convert_to_numpy_array,
    cosine_similarity,
    VectorIndexManager,
)


class TestLangChainEmbeddings:
    """Test LangChain embeddings creation."""
    
    @patch("src.project_watch_mcp.utils.embeddings.vector_support.OpenAIEmbeddings")
    def test_create_openai_embeddings(self, mock_openai):
        """Test creating OpenAI embeddings with LangChain."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance
        
        embeddings = create_langchain_embeddings(
            provider_type="openai",
            api_key="test-key",
            model="text-embedding-3-large",
            dimensions=3072,
        )
        
        assert embeddings == mock_instance
        mock_openai.assert_called_once_with(
            api_key="test-key",
            model="text-embedding-3-large",
            dimensions=3072,
            max_retries=3,
        )
    
    @patch("src.project_watch_mcp.utils.embeddings.vector_support.VoyageEmbeddings")
    def test_create_voyage_embeddings(self, mock_voyage):
        """Test creating Voyage embeddings with LangChain."""
        mock_instance = MagicMock()
        mock_voyage.return_value = mock_instance
        
        embeddings = create_langchain_embeddings(
            provider_type="voyage",
            api_key="test-voyage-key",
            model="voyage-code-2",
            batch_size=16,
        )
        
        assert embeddings == mock_instance
        mock_voyage.assert_called_once_with(
            voyage_api_key="test-voyage-key",
            model="voyage-code-2",
            batch_size=16,
        )
    
    def test_create_embeddings_invalid_provider(self):
        """Test error handling for invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_langchain_embeddings(provider_type="invalid")


class TestNeo4jVectorStore:
    """Test Neo4j vector store creation."""
    
    @patch("src.project_watch_mcp.utils.embeddings.vector_support.Neo4jVector")
    def test_create_neo4j_vector_store(self, mock_neo4j_vector):
        """Test creating Neo4j vector store."""
        mock_store = MagicMock()
        mock_neo4j_vector.return_value = mock_store
        mock_embeddings = MagicMock()
        
        store = create_neo4j_vector_store(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password",
            embeddings=mock_embeddings,
            index_name="test-index",
            node_label="TestNode",
            text_property="text",
            embedding_property="vector",
            search_type="hybrid",
        )
        
        assert store == mock_store
        mock_neo4j_vector.assert_called_once_with(
            embedding=mock_embeddings,
            url="bolt://localhost:7687",
            username="neo4j",
            password="password",
            index_name="test-index",
            node_label="TestNode",
            text_node_property="text",
            embedding_node_property="vector",
            search_type="hybrid",
            keyword_index_name="test-index-keywords",
            retrieval_query=None,
        )


class TestVectorOperations:
    """Test vector operation utilities."""
    
    def test_convert_to_numpy_array(self):
        """Test converting list to numpy array."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        arr = convert_to_numpy_array(embedding)
        
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (4,)
        assert np.allclose(arr, np.array([0.1, 0.2, 0.3, 0.4]))
    
    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity for identical vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        
        similarity = cosine_similarity(vec1, vec2)
        assert pytest.approx(similarity, 0.0001) == 1.0
    
    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity for orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        similarity = cosine_similarity(vec1, vec2)
        assert pytest.approx(similarity, 0.0001) == 0.0
    
    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity for opposite vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        
        similarity = cosine_similarity(vec1, vec2)
        assert pytest.approx(similarity, 0.0001) == -1.0
    
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0


class TestVectorIndexManager:
    """Test vector index management."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver."""
        driver = MagicMock()
        session = AsyncMock()
        
        # Create a mock context manager for session
        session_cm = AsyncMock()
        session_cm.__aenter__.return_value = session
        session_cm.__aexit__.return_value = None
        
        driver.session.return_value = session_cm
        return driver, session
    
    @pytest.mark.asyncio
    async def test_create_vector_index(self, mock_driver):
        """Test creating a vector index."""
        driver, session = mock_driver
        manager = VectorIndexManager(driver, dimensions=1536)
        
        session.run = AsyncMock()
        
        result = await manager.create_vector_index(
            index_name="test-index",
            node_label="TestNode",
            property_name="embedding",
            similarity_function="cosine",
        )
        
        assert result == {"status": "created", "index_name": "test-index"}
        
        # Verify the query was executed
        session.run.assert_called_once()
        query = session.run.call_args[0][0]
        assert "CREATE VECTOR INDEX `test-index`" in query
        assert "FOR (n:TestNode)" in query
        assert "ON (n.embedding)" in query
        assert "`vector.dimensions`: 1536" in query
        assert "`vector.similarity_function`: 'cosine'" in query
    
    @pytest.mark.asyncio
    async def test_drop_vector_index(self, mock_driver):
        """Test dropping a vector index."""
        driver, session = mock_driver
        manager = VectorIndexManager(driver)
        
        session.run = AsyncMock()
        
        result = await manager.drop_vector_index("test-index")
        
        assert result == {"status": "dropped", "index_name": "test-index"}
        session.run.assert_called_once_with("DROP INDEX `test-index` IF EXISTS")
    
    @pytest.mark.asyncio
    async def test_list_vector_indexes(self, mock_driver):
        """Test listing vector indexes."""
        driver, session = mock_driver
        manager = VectorIndexManager(driver)
        
        # Mock index data
        mock_record = {
            "name": "test-index",
            "state": "ONLINE",
            "labelsOrTypes": ["TestNode"],
            "properties": ["embedding"],
            "options": {"vector.dimensions": 1536},
        }
        
        mock_result = AsyncMock()
        mock_result.__aiter__.return_value = [mock_record]
        session.run = AsyncMock(return_value=mock_result)
        
        indexes = await manager.list_vector_indexes()
        
        assert len(indexes) == 1
        assert indexes[0]["name"] == "test-index"
        assert indexes[0]["state"] == "ONLINE"
        assert indexes[0]["labels"] == ["TestNode"]