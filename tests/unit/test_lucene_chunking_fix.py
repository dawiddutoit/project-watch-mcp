"""Test that chunking respects Lucene's 32KB byte limit."""

import pytest
from pathlib import Path
from datetime import datetime

from project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile


class TestLuceneChunkingFix:
    """Test that chunking implementation prevents Lucene index failures."""
    
    def test_chunk_content_respects_byte_limit(self):
        """Test that no chunk exceeds the 32KB Lucene limit."""
        # Create a RAG instance (no driver needed for chunking tests)
        rag = Neo4jRAG(
            neo4j_driver=None,
            project_name="test",
            embeddings=None,
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Test 1: Small content should remain as single chunk
        small_content = "Hello World\n" * 100
        chunks = rag.chunk_content(small_content)
        assert len(chunks) == 1
        assert len(chunks[0].encode('utf-8')) <= 30000
        
        # Test 2: Large content should be split
        large_content = "x" * 40000  # 40KB of ASCII characters
        chunks = rag.chunk_content(large_content)
        assert len(chunks) > 1
        for chunk in chunks:
            chunk_bytes = len(chunk.encode('utf-8'))
            assert chunk_bytes <= 30000, f"Chunk exceeds limit: {chunk_bytes} bytes"
        
        # Test 3: Content with multi-byte UTF-8 characters
        # Each emoji is 4 bytes in UTF-8
        emoji_content = "🎉" * 10000  # 40KB of emojis
        chunks = rag.chunk_content(emoji_content)
        assert len(chunks) > 1
        for chunk in chunks:
            chunk_bytes = len(chunk.encode('utf-8'))
            assert chunk_bytes <= 30000, f"Emoji chunk exceeds limit: {chunk_bytes} bytes"
        
        # Test 4: Very long single line
        long_line = "a" * 35000  # Single line exceeding limit
        chunks = rag.chunk_content(long_line)
        assert len(chunks) > 1
        for chunk in chunks:
            chunk_bytes = len(chunk.encode('utf-8'))
            assert chunk_bytes <= 30000, f"Long line chunk exceeds limit: {chunk_bytes} bytes"
        
        # Test 5: Mixed content with long and short lines
        mixed_content = "\n".join([
            "short line",
            "x" * 35000,  # Very long line
            "another short line",
            "y" * 20000,  # Medium long line
            "final line"
        ])
        chunks = rag.chunk_content(mixed_content)
        for i, chunk in enumerate(chunks):
            chunk_bytes = len(chunk.encode('utf-8'))
            assert chunk_bytes <= 30000, f"Mixed content chunk {i} exceeds limit: {chunk_bytes} bytes"
    
    def test_split_large_line(self):
        """Test the _split_large_line helper method."""
        rag = Neo4jRAG(
            neo4j_driver=None,
            project_name="test",
            embeddings=None
        )
        
        # Test splitting a large line
        large_line = "a" * 40000
        chunks = rag._split_large_line(large_line, 30000)
        
        # Should produce at least 2 chunks
        assert len(chunks) >= 2
        
        # Each chunk should be within limits
        for chunk in chunks:
            assert len(chunk.encode('utf-8')) <= 30000
        
        # Chunks should reconstruct the original
        reconstructed = "".join(chunks)
        assert reconstructed == large_line
        
        # Test with multi-byte characters
        emoji_line = "🎉" * 10000  # 40KB
        chunks = rag._split_large_line(emoji_line, 30000)
        
        for chunk in chunks:
            assert len(chunk.encode('utf-8')) <= 30000
        
        reconstructed = "".join(chunks)
        assert reconstructed == emoji_line
    
    def test_sanitize_for_lucene(self):
        """Test that _sanitize_for_lucene handles oversized terms correctly."""
        rag = Neo4jRAG(
            neo4j_driver=None,
            project_name="test",
            embeddings=None
        )
        
        # Test 1: Normal text should pass through unchanged
        normal_text = "This is a normal sentence with regular words"
        sanitized = rag._sanitize_for_lucene(normal_text)
        assert sanitized == normal_text
        
        # Test 2: Text with an oversized term (like base64 data)
        # Create a single "word" that exceeds 32KB
        huge_word = "x" * 40000  # 40KB single word
        text_with_huge = f"normal text {huge_word} more text"
        sanitized = rag._sanitize_for_lucene(text_with_huge)
        
        # The huge word should be truncated
        assert huge_word not in sanitized
        assert "normal text" in sanitized
        assert "more text" in sanitized
        
        # Each term in the result should be under the limit
        for term in sanitized.split():
            assert len(term.encode('utf-8')) <= 32000
        
        # Test 3: Multiple oversized terms
        huge1 = "a" * 35000
        huge2 = "b" * 35000
        text = f"start {huge1} middle {huge2} end"
        sanitized = rag._sanitize_for_lucene(text)
        
        assert "start" in sanitized
        assert "middle" in sanitized
        assert "end" in sanitized
        
        # Original huge terms should be truncated
        for term in sanitized.split():
            assert len(term.encode('utf-8')) <= 32000
    
    def test_chunk_by_tokens_respects_byte_limit(self):
        """Test that token-based chunking also respects byte limits."""
        rag = Neo4jRAG(
            neo4j_driver=None,
            project_name="test",
            embeddings=None
        )
        
        # Create content that would trigger token-based chunking
        # This simulates a very large file
        large_content = "x" * 200000  # 200KB
        
        chunks = rag._chunk_by_tokens(large_content, max_tokens=2000, overlap_tokens=200)
        
        # Verify all chunks are within byte limits
        for i, chunk in enumerate(chunks):
            chunk_bytes = len(chunk.encode('utf-8'))
            assert chunk_bytes <= 30000, f"Token chunk {i} exceeds limit: {chunk_bytes} bytes"
    
    def test_realistic_code_file(self):
        """Test with a realistic code file that might have caused the original issue."""
        rag = Neo4jRAG(
            neo4j_driver=None,
            project_name="test",
            embeddings=None,
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Simulate a lock file or config file with very long lines
        # These are common culprits for the Lucene error
        lockfile_content = """version = 1
revision = 2
requires_python = ">=3.11"

[[packages]]
name = "example-package"
version = "1.0.0"
source = "pypi"
dependencies = [
    """ + ", ".join([f'"dependency-{i}"' for i in range(5000)]) + """
]

[metadata]
content_hash = '""" + "a" * 100000 + """'
"""
        
        chunks = rag.chunk_content(lockfile_content)
        
        # Verify all chunks are within limits
        for i, chunk in enumerate(chunks):
            chunk_bytes = len(chunk.encode('utf-8'))
            assert chunk_bytes <= 30000, f"Lockfile chunk {i} exceeds limit: {chunk_bytes} bytes"
            
        # Verify we didn't lose too much content
        total_chunk_size = sum(len(chunk) for chunk in chunks)
        # Allow for some overlap
        assert total_chunk_size >= len(lockfile_content) * 0.9


if __name__ == "__main__":
    # Run the tests
    test = TestLuceneChunkingFix()
    test.test_chunk_content_respects_byte_limit()
    print("✓ Chunk content respects byte limit")
    
    test.test_split_large_line()
    print("✓ Split large line works correctly")
    
    test.test_chunk_by_tokens_respects_byte_limit()
    print("✓ Token-based chunking respects byte limit")
    
    test.test_realistic_code_file()
    print("✓ Realistic code file chunks correctly")
    
    print("\n✅ All tests passed! Lucene chunking fix is working correctly.")