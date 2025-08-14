"""Integration tests for embeddings with real APIs.

These tests require actual API keys to be set in environment variables:
- OPENAI_API_KEY for OpenAI embeddings
- VOYAGE_API_KEY for Voyage AI embeddings

Run these tests with:
    OPENAI_API_KEY=your_key VOYAGE_API_KEY=your_key pytest tests/test_embeddings_integration.py

Note: These tests will make real API calls and may incur costs.
"""

import asyncio
import math
import os

import pytest

from project_watch_mcp.utils.embeddings import (
    OpenAIEmbeddingsProvider,
    VoyageEmbeddingsProvider,
    create_embeddings_provider,
)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping OpenAI integration tests"
)
class TestOpenAIIntegration:
    """Integration tests for OpenAI embeddings using real API."""

    @pytest.mark.asyncio
    async def test_openai_real_embeddings(self):
        """Test that OpenAI returns real embeddings."""
        provider = OpenAIEmbeddingsProvider()
        
        # Embed a simple text
        embedding = await provider.embed_text("Hello, world!")
        
        # Verify embedding properties
        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # Default dimension for text-embedding-3-small
        assert all(isinstance(x, float) for x in embedding)
        
        # Check that values are normalized (typical for embeddings)
        magnitude = math.sqrt(sum(x * x for x in embedding))
        assert 0.9 < magnitude < 1.1  # Should be close to unit vector

    @pytest.mark.asyncio
    async def test_openai_semantic_similarity(self):
        """Test that OpenAI embeddings capture semantic similarity."""
        provider = OpenAIEmbeddingsProvider()
        
        # Embed similar texts
        emb1 = await provider.embed_text("The cat sits on the mat")
        emb2 = await provider.embed_text("A feline rests on the rug")
        emb3 = await provider.embed_text("import numpy as np")  # Very different
        
        # Calculate similarities
        sim_similar = cosine_similarity(emb1, emb2)
        sim_different = cosine_similarity(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim_similar > sim_different
        assert sim_similar > 0.7  # Similar sentences typically > 0.7
        assert sim_different < 0.5  # Different topics typically < 0.5

    @pytest.mark.asyncio
    async def test_openai_code_embeddings(self):
        """Test OpenAI embeddings for code understanding."""
        provider = OpenAIEmbeddingsProvider()
        
        # Python function examples
        code1 = """
        def calculate_sum(numbers):
            total = 0
            for num in numbers:
                total += num
            return total
        """
        
        code2 = """
        def add_numbers(list_of_nums):
            result = sum(list_of_nums)
            return result
        """
        
        code3 = """
        class DatabaseConnection:
            def __init__(self, host, port):
                self.host = host
                self.port = port
        """
        
        # Get embeddings
        emb1 = await provider.embed_text(code1)
        emb2 = await provider.embed_text(code2)
        emb3 = await provider.embed_text(code3)
        
        # Similar functions should be more similar than different code
        sim_functions = cosine_similarity(emb1, emb2)
        sim_different = cosine_similarity(emb1, emb3)
        
        assert sim_functions > sim_different
        assert sim_functions > 0.6  # Similar functions

    @pytest.mark.asyncio
    async def test_openai_consistency(self):
        """Test that same text produces same embedding."""
        provider = OpenAIEmbeddingsProvider()
        
        text = "Consistency test for embeddings"
        
        # Get embeddings multiple times
        emb1 = await provider.embed_text(text)
        emb2 = await provider.embed_text(text)
        
        # Should be identical
        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_openai_different_models(self):
        """Test different OpenAI embedding models."""
        # Test with ada-002 if needed (older model)
        provider_small = OpenAIEmbeddingsProvider(model="text-embedding-3-small")
        
        embedding = await provider_small.embed_text("Test text")
        assert len(embedding) == 1536  # Default dimension
        
        # You can test text-embedding-3-large if needed
        # provider_large = OpenAIEmbeddingsProvider(model="text-embedding-3-large", dimension=3072)
        # embedding_large = await provider_large.embed_text("Test text")
        # assert len(embedding_large) == 3072


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY not set - skipping Voyage integration tests"
)
class TestVoyageIntegration:
    """Integration tests for Voyage AI embeddings using real API."""

    @pytest.mark.asyncio
    async def test_voyage_real_embeddings(self):
        """Test that Voyage returns real embeddings."""
        provider = VoyageEmbeddingsProvider()
        
        # Embed a code snippet
        embedding = await provider.embed_text("def hello(): return 'world'")
        
        # Verify embedding properties
        assert isinstance(embedding, list)
        assert len(embedding) == 1024  # voyage-code-3 dimension
        assert all(isinstance(x, float) for x in embedding)
        
        # Check normalization
        magnitude = math.sqrt(sum(x * x for x in embedding))
        assert 0.9 < magnitude < 1.1  # Should be close to unit vector

    @pytest.mark.asyncio
    async def test_voyage_code_optimized_embeddings(self):
        """Test that Voyage is optimized for code understanding."""
        provider = VoyageEmbeddingsProvider(model="voyage-code-3")
        
        # Similar Python functions
        func1 = "def add(a, b): return a + b"
        func2 = "def sum(x, y): return x + y"
        
        # Different language but similar concept
        func3 = "function add(a, b) { return a + b; }"  # JavaScript
        
        # Completely different
        text = "The weather is nice today"
        
        # Get embeddings
        emb1 = await provider.embed_text(func1)
        emb2 = await provider.embed_text(func2)
        emb3 = await provider.embed_text(func3)
        emb4 = await provider.embed_text(text)
        
        # Calculate similarities
        sim_python = cosine_similarity(emb1, emb2)
        sim_cross_lang = cosine_similarity(emb1, emb3)
        sim_different = cosine_similarity(emb1, emb4)
        
        # Code should be more similar to code than to text
        assert sim_python > sim_different
        assert sim_cross_lang > sim_different
        assert sim_python > 0.7  # Very similar Python functions

    @pytest.mark.asyncio
    async def test_voyage_document_vs_query(self):
        """Test Voyage's document vs query embedding types."""
        provider = VoyageEmbeddingsProvider()
        
        # Document text
        document = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        """
        
        # Embed as document
        doc_embedding = await provider.embed_text(document, input_type="document")
        
        # Search queries
        query1 = "recursive fibonacci implementation"
        query2 = "weather forecast"
        
        # Embed as queries
        query1_embedding = await provider.embed_text(query1, input_type="query")
        query2_embedding = await provider.embed_text(query2, input_type="query")
        
        # Calculate similarities
        sim_relevant = cosine_similarity(doc_embedding, query1_embedding)
        sim_irrelevant = cosine_similarity(doc_embedding, query2_embedding)
        
        # Relevant query should have higher similarity
        assert sim_relevant > sim_irrelevant
        assert sim_relevant > 0.5  # Should be reasonably similar

    @pytest.mark.asyncio
    async def test_voyage_batch_embeddings(self):
        """Test Voyage batch embedding functionality."""
        provider = VoyageEmbeddingsProvider()
        
        texts = [
            "def func1(): pass",
            "def func2(): pass",
            "def func3(): pass",
            "class MyClass: pass",
            "import numpy as np",
        ]
        
        # Batch embed
        embeddings = await provider.embed_batch(texts)
        
        # Verify results
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert len(embedding) == 1024
            assert all(isinstance(x, float) for x in embedding)
        
        # Each should be different
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert embeddings[i] != embeddings[j]

    @pytest.mark.asyncio
    async def test_voyage_consistency(self):
        """Test that Voyage returns consistent embeddings."""
        provider = VoyageEmbeddingsProvider()
        
        code = "class Example: def __init__(self): pass"
        
        # Get embeddings multiple times
        emb1 = await provider.embed_text(code)
        emb2 = await provider.embed_text(code)
        
        # Should be identical
        assert emb1 == emb2


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("VOYAGE_API_KEY")),
    reason="Both API keys needed for comparison tests"
)
class TestProviderComparison:
    """Compare different embedding providers."""

    @pytest.mark.asyncio
    async def test_provider_code_understanding_comparison(self):
        """Compare how different providers understand code."""
        openai_provider = OpenAIEmbeddingsProvider()
        voyage_provider = VoyageEmbeddingsProvider()
        
        # Same code snippet
        code = """
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        """
        
        # Get embeddings from both
        openai_emb = await openai_provider.embed_text(code)
        voyage_emb = await voyage_provider.embed_text(code)
        
        # Different dimensions but both should be valid
        assert len(openai_emb) == 1536  # OpenAI dimension
        assert len(voyage_emb) == 1024  # Voyage dimension
        
        # Both should be normalized
        openai_mag = math.sqrt(sum(x * x for x in openai_emb))
        voyage_mag = math.sqrt(sum(x * x for x in voyage_emb))
        assert 0.9 < openai_mag < 1.1
        assert 0.9 < voyage_mag < 1.1

    @pytest.mark.asyncio
    async def test_search_accuracy_comparison(self):
        """Compare search accuracy between providers."""
        openai_provider = OpenAIEmbeddingsProvider()
        voyage_provider = VoyageEmbeddingsProvider()
        
        # Documents (code snippets)
        documents = [
            "def sort_array(arr): return sorted(arr)",  # Sorting
            "def search_item(lst, item): return item in lst",  # Searching
            "def connect_database(host, port): pass",  # Database
        ]
        
        # Search query
        query = "array sorting algorithm"
        
        # Get OpenAI embeddings
        openai_docs = [await openai_provider.embed_text(doc) for doc in documents]
        openai_query = await openai_provider.embed_text(query)
        
        # Get Voyage embeddings
        voyage_docs = await voyage_provider.embed_batch(documents, input_type="document")
        voyage_query = await voyage_provider.embed_text(query, input_type="query")
        
        # Calculate similarities for OpenAI
        openai_sims = [cosine_similarity(openai_query, doc) for doc in openai_docs]
        openai_best = openai_sims.index(max(openai_sims))
        
        # Calculate similarities for Voyage
        voyage_sims = [cosine_similarity(voyage_query, doc) for doc in voyage_docs]
        voyage_best = voyage_sims.index(max(voyage_sims))
        
        # Both should identify the sorting function as most relevant
        assert openai_best == 0  # First document is about sorting
        assert voyage_best == 0  # Both should agree


class TestEmbeddingPerformance:
    """Test embedding performance characteristics."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_openai_latency(self):
        """Measure OpenAI embedding latency."""
        import time
        
        provider = OpenAIEmbeddingsProvider()
        
        text = "Performance test text"
        
        # Measure time for single embedding
        start = time.time()
        await provider.embed_text(text)
        single_time = time.time() - start
        
        # Should complete in reasonable time (< 2 seconds typically)
        assert single_time < 2.0
        
        print(f"\nOpenAI single embedding latency: {single_time:.3f}s")

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("VOYAGE_API_KEY"),
        reason="VOYAGE_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_voyage_batch_performance(self):
        """Test Voyage batch embedding performance."""
        import time
        
        provider = VoyageEmbeddingsProvider()
        
        # Multiple texts
        texts = [f"Test text {i}" for i in range(10)]
        
        # Measure batch time
        start = time.time()
        await provider.embed_batch(texts)
        batch_time = time.time() - start
        
        # Batch should be efficient (< 3 seconds for 10 texts typically)
        assert batch_time < 3.0
        
        print(f"\nVoyage batch embedding (10 texts) latency: {batch_time:.3f}s")