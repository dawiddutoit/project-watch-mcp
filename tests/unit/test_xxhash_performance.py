"""Test xxHash performance improvements."""

import hashlib
import time
import xxhash
import pytest
from pathlib import Path
from datetime import datetime

from project_watch_mcp.neo4j_rag import CodeFile


class TestXXHashPerformance:
    """Test that xxHash provides significant performance improvements."""
    
    def test_xxhash_is_faster_than_sha256(self):
        """Test that xxHash is at least 10x faster than SHA-256."""
        # Create a large content string (1MB)
        content = "x" * (1024 * 1024)
        
        # Time SHA-256
        start = time.perf_counter()
        for _ in range(100):
            hashlib.sha256(content.encode()).hexdigest()
        sha256_time = time.perf_counter() - start
        
        # Time xxHash
        start = time.perf_counter()
        for _ in range(100):
            xxhash.xxh64(content.encode()).hexdigest()
        xxhash_time = time.perf_counter() - start
        
        # xxHash should be at least 10x faster
        speedup = sha256_time / xxhash_time
        print(f"SHA-256 time: {sha256_time:.4f}s")
        print(f"xxHash time: {xxhash_time:.4f}s")
        print(f"Speedup: {speedup:.1f}x")
        
        assert speedup > 10, f"xxHash only {speedup:.1f}x faster, expected >10x"
    
    def test_code_file_uses_xxhash(self):
        """Test that CodeFile now uses xxHash for hashing."""
        content = "print('hello world')"
        code_file = CodeFile(
            path=Path("test.py"),
            content=content,
            language="python",
            project_name="test",
            size=len(content),
            last_modified=datetime.now()
        )
        
        # The hash should be an xxHash
        file_hash = code_file.file_hash
        
        # Verify it matches what xxHash would produce
        expected = xxhash.xxh64("print('hello world')".encode()).hexdigest()
        assert file_hash == expected
        
        # Verify it doesn't match SHA-256 (different algorithm)
        sha256_hash = hashlib.sha256("print('hello world')".encode()).hexdigest()
        assert file_hash != sha256_hash
    
    def test_hash_consistency(self):
        """Test that xxHash produces consistent results."""
        content = "test content for hashing"
        now = datetime.now()
        
        # Create multiple CodeFile instances with same content
        file1 = CodeFile(
            path=Path("file1.py"),
            content=content,
            language="python",
            project_name="test",
            size=len(content),
            last_modified=now
        )
        
        file2 = CodeFile(
            path=Path("file2.py"),
            content=content,
            language="python", 
            project_name="test",
            size=len(content),
            last_modified=now
        )
        
        # Hashes should be identical for same content
        assert file1.file_hash == file2.file_hash
        
        # Different content should produce different hash
        modified_content = content + " modified"
        file3 = CodeFile(
            path=Path("file3.py"),
            content=modified_content,
            language="python",
            project_name="test",
            size=len(modified_content),
            last_modified=now
        )
        
        assert file3.file_hash != file1.file_hash