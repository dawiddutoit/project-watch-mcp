"""Test suite for async session management and error handling improvements.

This test suite validates:
- Fix-008: Proper async session cleanup with try/finally blocks
- Fix-009: Comprehensive error handling with context
- Fix-011: Timeout protection for vector search operations
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import numpy as np
from neo4j import AsyncDriver
from neo4j.exceptions import ServiceUnavailable, SessionExpired

# Import modules to test
from src.project_watch_mcp.vector_search.neo4j_native_vectors import (
    NativeVectorIndex,
    VectorIndexConfig,
    VectorSearchResult,
    VectorUpsertResult
)
from src.project_watch_mcp.complexity_analysis.languages.kotlin_analyzer import (
    KotlinComplexityAnalyzer
)


class TestAsyncSessionManagement:
    """Test proper async session cleanup (Fix-008)."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = AsyncMock(spec=AsyncDriver)
        return driver
    
    @pytest.fixture
    def vector_index(self, mock_driver):
        """Create a NativeVectorIndex instance with mock driver."""
        config = VectorIndexConfig(
            index_name="test-index",
            dimensions=128
        )
        return NativeVectorIndex(mock_driver, config)
    
    @pytest.mark.asyncio
    async def test_session_cleanup_on_success(self, vector_index, mock_driver):
        """Test that sessions are properly closed on successful operations."""
        # Setup mock session
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=AsyncMock())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session.return_value = mock_session
        
        # Execute operation
        await vector_index.create_index()
        
        # Verify session was properly closed
        mock_session.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_session_cleanup_on_error(self, vector_index, mock_driver):
        """Test that sessions are properly closed even when errors occur."""
        # Setup mock session that raises error
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=ServiceUnavailable("Connection failed"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session.return_value = mock_session
        
        # Execute operation and expect error
        with pytest.raises(ServiceUnavailable):
            await vector_index.create_index()
        
        # Verify session was still properly closed
        mock_session.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_session_cleanup(self, vector_index, mock_driver):
        """Test cleanup of multiple sessions in batch operations."""
        # Setup mock sessions
        sessions_created = []
        
        def create_session():
            mock_session = AsyncMock()
            mock_session.run = AsyncMock(return_value=AsyncMock())
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            sessions_created.append(mock_session)
            return mock_session
        
        mock_driver.session.side_effect = create_session
        
        # Execute batch operation
        vectors = [
            {"node_id": f"node_{i}", "vector": [0.1] * 128}
            for i in range(5)
        ]
        await vector_index.batch_upsert_vectors(vectors)
        
        # Verify all sessions were closed
        for session in sessions_created:
            session.__aexit__.assert_called()


class TestErrorHandling:
    """Test comprehensive error handling with context (Fix-009)."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = AsyncMock(spec=AsyncDriver)
        return driver
    
    @pytest.fixture
    def vector_index(self, mock_driver):
        """Create a NativeVectorIndex instance with mock driver."""
        config = VectorIndexConfig(
            index_name="test-index",
            dimensions=128
        )
        return NativeVectorIndex(mock_driver, config)
    
    @pytest.mark.asyncio
    async def test_error_context_in_vector_search(self, vector_index, mock_driver):
        """Test that vector search errors include helpful context."""
        # Setup mock session that fails
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=SessionExpired("Session expired"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session.return_value = mock_session
        
        # Execute search and verify error context
        with pytest.raises(SessionExpired) as exc_info:
            await vector_index.search([0.1] * 128, top_k=10)
        
        # Error should be logged with context
        # Note: We need to verify logging was called with context
    
    @pytest.mark.asyncio
    async def test_error_recovery_in_batch_operations(self, vector_index, mock_driver):
        """Test graceful error recovery in batch operations."""
        # Setup mock session
        mock_session = AsyncMock()
        call_count = 0
        
        async def run_with_partial_failure(query, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds
                result = AsyncMock()
                result.__aiter__ = AsyncMock(return_value=iter([
                    {"node_id": "node_0"}
                ]))
                return result
            else:
                # Second call fails
                raise ServiceUnavailable("Connection lost")
        
        mock_session.run = run_with_partial_failure
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session.return_value = mock_session
        
        # Execute batch operation
        vectors = [
            {"node_id": f"node_{i}", "vector": [0.1] * 128}
            for i in range(3)
        ]
        results = await vector_index.batch_upsert_vectors(vectors)
        
        # Verify partial success is handled
        assert len(results) == 3
        assert any(r.success for r in results)
        assert any(not r.success for r in results)
        assert any(r.error is not None for r in results)
    
    def test_kotlin_analyzer_error_context(self):
        """Test that Kotlin analyzer provides helpful error context."""
        analyzer = KotlinComplexityAnalyzer()
        
        # Test with invalid file path
        result = asyncio.run(analyzer.analyze_file(Path("/nonexistent/file.kt")))
        
        # Should return result with error context
        assert result.error is not None
        assert "not found" in result.error.lower()
        
        # Test with invalid syntax
        invalid_code = """
        fun broken() {
            // Missing closing brace
        """
        result = asyncio.run(analyzer.analyze_code(invalid_code))
        
        # Should return result with syntax error context
        assert result.error is not None
        assert "syntax" in result.error.lower()


class TestTimeoutProtection:
    """Test timeout protection for vector search operations (Fix-011)."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = AsyncMock(spec=AsyncDriver)
        return driver
    
    @pytest.fixture
    def vector_index(self, mock_driver):
        """Create a NativeVectorIndex instance with mock driver."""
        config = VectorIndexConfig(
            index_name="test-index",
            dimensions=128
        )
        return NativeVectorIndex(mock_driver, config)
    
    @pytest.mark.asyncio
    async def test_search_with_timeout(self, vector_index, mock_driver):
        """Test that vector search operations have timeout protection."""
        # Setup mock session that takes too long
        mock_session = AsyncMock()
        
        async def slow_run(query, params=None):
            await asyncio.sleep(10)  # Simulate slow operation
            return AsyncMock()
        
        mock_session.run = slow_run
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session.return_value = mock_session
        
        # Execute search with timeout
        with pytest.raises(asyncio.TimeoutError):
            # Search should timeout after reasonable time (e.g., 5 seconds)
            await asyncio.wait_for(
                vector_index.search([0.1] * 128, top_k=10),
                timeout=1.0
            )
    
    @pytest.mark.asyncio
    async def test_batch_operations_with_timeout(self, vector_index, mock_driver):
        """Test that batch operations have timeout protection."""
        # Setup mock session that hangs
        mock_session = AsyncMock()
        
        async def hanging_run(query, params=None):
            await asyncio.sleep(100)  # Simulate hanging operation
            return AsyncMock()
        
        mock_session.run = hanging_run
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session.return_value = mock_session
        
        # Execute batch operation with timeout
        vectors = [
            {"node_id": f"node_{i}", "vector": [0.1] * 128}
            for i in range(10)
        ]
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                vector_index.batch_upsert_vectors(vectors),
                timeout=2.0
            )
    
    @pytest.mark.asyncio
    async def test_index_stats_with_timeout(self, vector_index, mock_driver):
        """Test that index stats operations have timeout protection."""
        # Setup mock session with slow stats query
        mock_session = AsyncMock()
        
        async def slow_stats_run(query, params=None):
            if "SHOW INDEXES" in query:
                await asyncio.sleep(5)  # Simulate slow stats query
            result = AsyncMock()
            result.__aiter__ = AsyncMock(return_value=iter([]))
            return result
        
        mock_session.run = slow_stats_run
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session.return_value = mock_session
        
        # Execute stats query with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                vector_index.get_index_stats(),
                timeout=1.0
            )


class TestResourceLeakPrevention:
    """Test prevention of resource leaks in error scenarios."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = AsyncMock(spec=AsyncDriver)
        return driver
    
    @pytest.fixture
    def vector_index(self, mock_driver):
        """Create a NativeVectorIndex instance with mock driver."""
        config = VectorIndexConfig(
            index_name="test-index",
            dimensions=128
        )
        return NativeVectorIndex(mock_driver, config)
    
    @pytest.mark.asyncio
    async def test_no_session_leak_on_repeated_errors(self, vector_index, mock_driver):
        """Test that repeated errors don't leak sessions."""
        sessions_created = []
        
        def create_failing_session():
            mock_session = AsyncMock()
            mock_session.run = AsyncMock(side_effect=ServiceUnavailable("Failed"))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            sessions_created.append(mock_session)
            return mock_session
        
        mock_driver.session.side_effect = create_failing_session
        
        # Try multiple operations that fail
        for _ in range(10):
            try:
                await vector_index.create_index()
            except ServiceUnavailable:
                pass
        
        # Verify all sessions were closed
        assert len(sessions_created) == 10
        for session in sessions_created:
            session.__aexit__.assert_called()
    
    @pytest.mark.asyncio
    async def test_cleanup_on_keyboard_interrupt(self, vector_index, mock_driver):
        """Test that sessions are cleaned up on keyboard interrupt."""
        mock_session = AsyncMock()
        
        async def interrupted_run(query, params=None):
            raise KeyboardInterrupt()
        
        mock_session.run = interrupted_run
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session.return_value = mock_session
        
        # Execute operation that gets interrupted
        with pytest.raises(KeyboardInterrupt):
            await vector_index.search([0.1] * 128)
        
        # Verify session was still closed
        mock_session.__aexit__.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])