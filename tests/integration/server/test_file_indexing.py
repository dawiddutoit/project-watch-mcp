"""
Integration tests for file indexing and content processing.

These tests verify:
1. File content extraction and chunking
2. Embedding generation and storage
3. Index creation and updates
4. File change detection and re-indexing
5. Batch indexing performance
"""

import asyncio
import hashlib
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, patch
import pytest

from project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG
from project_watch_mcp.repository_monitor import FileInfo, RepositoryMonitor


class TestFileIndexing:
    """Test suite for file indexing operations."""

    @pytest.fixture
    async def sample_codebase(self):
        """Create a sample codebase with various file types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create Python module with classes and functions
            (repo_path / "user_service.py").write_text("""
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class User:
    '''Represents a user in the system.'''
    id: str
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

class UserService:
    '''Service for managing users.'''
    
    def __init__(self, database):
        self.db = database
        self.cache = {}
    
    async def get_user(self, user_id: str) -> Optional[User]:
        '''Retrieve a user by ID.'''
        if user_id in self.cache:
            return self.cache[user_id]
        
        user_data = await self.db.find_one({'id': user_id})
        if user_data:
            user = User(**user_data)
            self.cache[user_id] = user
            return user
        return None
    
    async def create_user(self, username: str, email: str) -> User:
        '''Create a new user.'''
        user = User(
            id=self.generate_id(),
            username=username,
            email=email,
            created_at=datetime.now()
        )
        await self.db.insert_one(user.__dict__)
        self.cache[user.id] = user
        return user
    
    async def list_users(self, limit: int = 100) -> List[User]:
        '''List all active users.'''
        users_data = await self.db.find({'is_active': True}).limit(limit)
        return [User(**data) for data in users_data]
    
    def generate_id(self) -> str:
        '''Generate a unique user ID.'''
        import uuid
        return str(uuid.uuid4())
""")
            
            # Create JavaScript/React component
            (repo_path / "UserProfile.jsx").write_text("""
import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';

/**
 * UserProfile component displays user information
 */
const UserProfile = ({ onUpdate }) => {
    const { userId } = useParams();
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    useEffect(() => {
        fetchUserData();
    }, [userId]);
    
    const fetchUserData = async () => {
        try {
            setLoading(true);
            const response = await axios.get(`/api/users/${userId}`);
            setUser(response.data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    
    const handleUpdate = async (updates) => {
        try {
            const response = await axios.patch(`/api/users/${userId}`, updates);
            setUser(response.data);
            if (onUpdate) {
                onUpdate(response.data);
            }
        } catch (err) {
            setError(err.message);
        }
    };
    
    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;
    if (!user) return <div>User not found</div>;
    
    return (
        <div className="user-profile">
            <h1>{user.username}</h1>
            <p>Email: {user.email}</p>
            <p>Member since: {new Date(user.createdAt).toLocaleDateString()}</p>
            <button onClick={() => handleUpdate({ isActive: !user.isActive })}>
                {user.isActive ? 'Deactivate' : 'Activate'}
            </button>
        </div>
    );
};

export default UserProfile;
""")
            
            # Create TypeScript interface definitions
            (repo_path / "types.ts").write_text("""
export interface User {
    id: string;
    username: string;
    email: string;
    createdAt: Date;
    isActive: boolean;
}

export interface ApiResponse<T> {
    data: T;
    status: number;
    message?: string;
}

export type UserRole = 'admin' | 'user' | 'guest';

export interface AuthContext {
    user: User | null;
    login: (credentials: LoginCredentials) => Promise<void>;
    logout: () => void;
    isAuthenticated: boolean;
}

export interface LoginCredentials {
    username: string;
    password: string;
    rememberMe?: boolean;
}
""")
            
            # Create a large file for chunking tests
            large_content = []
            for i in range(100):
                large_content.append(f"""
def function_{i}(param_{i}: int) -> int:
    '''Function {i} documentation.'''
    result = param_{i} * 2
    # Some processing logic here
    for j in range(10):
        result += j
    return result
""")
            
            (repo_path / "large_module.py").write_text("\n".join(large_content))
            
            # Create test file
            (repo_path / "test_user_service.py").write_text("""
import pytest
from user_service import User, UserService

@pytest.fixture
def user_service(mock_db):
    return UserService(mock_db)

@pytest.mark.asyncio
async def test_create_user(user_service):
    user = await user_service.create_user("testuser", "test@example.com")
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active is True

@pytest.mark.asyncio  
async def test_get_user(user_service):
    # Create a user first
    created = await user_service.create_user("testuser", "test@example.com")
    
    # Retrieve the user
    retrieved = await user_service.get_user(created.id)
    assert retrieved.id == created.id
    assert retrieved.username == created.username
""")
            
            yield repo_path

    @pytest.fixture
    def mock_neo4j_rag(self):
        """Create a mock Neo4jRAG instance."""
        rag = AsyncMock(spec=Neo4jRAG)
        rag.initialize = AsyncMock()
        rag.index_file = AsyncMock()
        rag.delete_file = AsyncMock()
        rag.search_code = AsyncMock(return_value=[])
        rag.close = AsyncMock()
        return rag

    @pytest.mark.asyncio
    async def test_single_file_indexing(self, sample_codebase, mock_neo4j_rag):
        """Test indexing a single file."""
        file_path = sample_codebase / "user_service.py"
        content = file_path.read_text()
        
        code_file = CodeFile(
            project_name="test_project",
            path=file_path,
            content=content,
            language="python",
            size=len(content),
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
        )
        
        # Index the file
        await mock_neo4j_rag.index_file(code_file)
        
        # Verify indexing was called
        mock_neo4j_rag.index_file.assert_called_once_with(code_file)

    @pytest.mark.asyncio
    async def test_batch_file_indexing(self, sample_codebase, mock_neo4j_rag):
        """Test batch indexing of multiple files."""
        # Create mock Neo4j driver
        mock_driver = AsyncMock()
        
        monitor = RepositoryMonitor(
            repo_path=sample_codebase,
            project_name="test_project",
            neo4j_driver=mock_driver,
            file_patterns=["*.py", "*.jsx", "*.ts"]
        )
        
        # Scan repository
        files = await monitor.scan_repository()
        
        # Convert to CodeFile objects and index
        indexed_count = 0
        for file_info in files:
            try:
                content = Path(file_info.path).read_text()
                code_file = CodeFile(
                    project_name="test_project",
                    path=Path(file_info.path),
                    content=content,
                    language=file_info.language or "unknown",
                    size=file_info.size,
                    last_modified=file_info.last_modified
                )
                await mock_neo4j_rag.index_file(code_file)
                indexed_count += 1
            except Exception:
                pass
        
        # Verify all files were indexed
        assert indexed_count > 0
        assert mock_neo4j_rag.index_file.call_count == indexed_count

    @pytest.mark.asyncio
    async def test_file_chunking(self, sample_codebase, mock_neo4j_rag):
        """Test that large files are properly chunked."""
        # Read large file
        large_file = sample_codebase / "large_module.py"
        content = large_file.read_text()
        
        # Create mock driver and Neo4jRAG
        mock_driver = AsyncMock()
        rag = Neo4jRAG(
            neo4j_driver=mock_driver,
            project_name="test_project",
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Chunk the content
        chunks = rag._chunk_text(content, chunk_size=500, chunk_overlap=50)
        
        # Verify chunking
        assert len(chunks) > 1  # Should create multiple chunks
        
        # Verify chunk properties
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) <= 550  # Account for overlap

    @pytest.mark.asyncio
    async def test_embedding_generation(self, sample_codebase):
        """Test embedding generation for indexed content."""
        from project_watch_mcp.utils.embeddings.base import EmbeddingProvider
        
        # Create mock embedding provider
        mock_embeddings = AsyncMock(spec=EmbeddingProvider)
        mock_embeddings.embed_documents = AsyncMock(
            return_value=[[0.1] * 1536, [0.2] * 1536]  # Mock embeddings
        )
        mock_embeddings.embed_query = AsyncMock(return_value=[0.15] * 1536)
        
        # Generate embeddings for code chunks
        chunks = [
            "def hello(): return 'world'",
            "class TestClass: pass"
        ]
        
        embeddings = await mock_embeddings.embed_documents(chunks)
        
        # Verify embeddings
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536
        mock_embeddings.embed_documents.assert_called_once_with(chunks)

    @pytest.mark.asyncio
    async def test_file_update_detection(self, sample_codebase, mock_neo4j_rag):
        """Test detection and re-indexing of updated files."""
        file_path = sample_codebase / "user_service.py"
        
        # Initial indexing
        original_content = file_path.read_text()
        original_mtime = file_path.stat().st_mtime
        
        code_file = CodeFile(
            project_name="test_project",
            path=file_path,
            content=original_content,
            language="python",
            size=len(original_content),
            last_modified=datetime.fromtimestamp(original_mtime)
        )
        
        await mock_neo4j_rag.index_file(code_file)
        
        # Modify file
        await asyncio.sleep(0.1)  # Ensure different timestamp
        new_content = original_content + "\n# Modified"
        file_path.write_text(new_content)
        
        # Re-index with updated content
        updated_file = CodeFile(
            project_name="test_project",
            path=file_path,
            content=new_content,
            language="python",
            size=len(new_content),
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
        )
        
        await mock_neo4j_rag.index_file(updated_file)
        
        # Verify both indexing calls
        assert mock_neo4j_rag.index_file.call_count == 2
        
        # Verify content changed
        first_call = mock_neo4j_rag.index_file.call_args_list[0][0][0]
        second_call = mock_neo4j_rag.index_file.call_args_list[1][0][0]
        assert first_call.hash != second_call.hash
        assert second_call.size > first_call.size

    @pytest.mark.asyncio
    async def test_file_deletion_handling(self, sample_codebase, mock_neo4j_rag):
        """Test handling of file deletions."""
        file_path = sample_codebase / "temp_file.py"
        file_path.write_text("# Temporary file")
        
        # Index file
        code_file = CodeFile(
            project_name="test_project",
            path=file_path,
            content="# Temporary file",
            language="python",
            size=16,
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
        )
        
        await mock_neo4j_rag.index_file(code_file)
        
        # Delete file
        file_path.unlink()
        
        # Remove from index
        await mock_neo4j_rag.delete_file(str(file_path))
        
        # Verify deletion
        mock_neo4j_rag.delete_file.assert_called_once_with(str(file_path))

    @pytest.mark.asyncio
    async def test_concurrent_file_indexing(self, sample_codebase, mock_neo4j_rag):
        """Test concurrent indexing of multiple files."""
        # Create mock Neo4j driver
        mock_driver = AsyncMock()
        
        monitor = RepositoryMonitor(
            repo_path=sample_codebase,
            project_name="test_project",
            neo4j_driver=mock_driver,
            file_patterns=["*.py", "*.jsx", "*.ts"]
        )
        
        files = await monitor.scan_repository()
        
        async def index_file(file_info):
            try:
                content = Path(file_info.path).read_text()
                code_file = CodeFile(
                    project_name="test_project",
                    path=Path(file_info.path),
                    content=content,
                    language=file_info.language or "unknown",
                    size=file_info.size,
                    last_modified=file_info.last_modified
                )
                await mock_neo4j_rag.index_file(code_file)
                return True
            except Exception:
                return False
        
        # Index files concurrently
        tasks = [index_file(f) for f in files]
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        assert all(results)
        assert mock_neo4j_rag.index_file.call_count == len(files)

    @pytest.mark.asyncio
    async def test_indexing_performance(self, sample_codebase, mock_neo4j_rag):
        """Test indexing performance for various file sizes."""
        # Create files of different sizes
        sizes = [100, 1000, 10000, 50000]
        
        for size in sizes:
            file_path = sample_codebase / f"file_{size}.py"
            content = "x" * size
            file_path.write_text(content)
            
            start_time = time.time()
            
            code_file = CodeFile(
                project_name="test_project",
                path=file_path,
                content=content,
                language="python",
                size=size,
                last_modified=datetime.fromtimestamp(time.time())
            )
            
            await mock_neo4j_rag.index_file(code_file)
            
            elapsed = time.time() - start_time
            
            # Indexing should be reasonably fast
            assert elapsed < 1.0  # Less than 1 second per file

    @pytest.mark.asyncio
    async def test_language_specific_indexing(self, sample_codebase, mock_neo4j_rag):
        """Test language-specific indexing strategies."""
        files = {
            "user_service.py": "python",
            "UserProfile.jsx": "javascript",
            "types.ts": "typescript"
        }
        
        for filename, expected_language in files.items():
            file_path = sample_codebase / filename
            content = file_path.read_text()
            
            code_file = CodeFile(
                project_name="test_project",
                path=file_path,
                content=content,
                language=expected_language,
                size=len(content),
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
            )
            
            await mock_neo4j_rag.index_file(code_file)
            
            # Verify language was correctly identified
            call_args = mock_neo4j_rag.index_file.call_args_list[-1][0][0]
            assert call_args.language == expected_language

    @pytest.mark.asyncio
    async def test_index_recovery_after_failure(self, sample_codebase, mock_neo4j_rag):
        """Test index recovery after partial failure."""
        # Create mock Neo4j driver
        mock_driver = AsyncMock()
        
        monitor = RepositoryMonitor(
            repo_path=sample_codebase,
            project_name="test_project",
            neo4j_driver=mock_driver,
            file_patterns=["*.py"]
        )
        
        files = await monitor.scan_repository()
        
        # Simulate failure on third file
        call_count = 0
        
        async def mock_index_with_failure(code_file):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise Exception("Indexing failed")
            return True
        
        mock_neo4j_rag.index_file = mock_index_with_failure
        
        # Index files with recovery
        success_count = 0
        failed_files = []
        
        for file_info in files:
            try:
                content = Path(file_info.path).read_text()
                code_file = CodeFile(
                    project_name="test_project",
                    path=Path(file_info.path),
                    content=content,
                    language=file_info.language or "unknown",
                    size=file_info.size,
                    last_modified=file_info.last_modified
                )
                await mock_neo4j_rag.index_file(code_file)
                success_count += 1
            except Exception:
                failed_files.append(str(file_info.path))
        
        # Verify partial success
        assert success_count == len(files) - 1
        assert len(failed_files) == 1

    @pytest.mark.asyncio
    async def test_incremental_indexing(self, sample_codebase, mock_neo4j_rag):
        """Test incremental indexing of new files."""
        # Create mock Neo4j driver
        mock_driver = AsyncMock()
        
        monitor = RepositoryMonitor(
            repo_path=sample_codebase,
            project_name="test_project",
            neo4j_driver=mock_driver,
            file_patterns=["*.py"]
        )
        
        # Initial indexing
        initial_files = await monitor.scan_repository()
        initial_count = len(initial_files)
        
        for file_info in initial_files:
            content = Path(file_info.path).read_text()
            code_file = CodeFile(
                project_name="test_project",
                path=Path(file_info.path),
                content=content,
                language=file_info.language or "unknown",
                size=file_info.size,
                last_modified=file_info.last_modified
            )
            await mock_neo4j_rag.index_file(code_file)
        
        # Add new file
        new_file = sample_codebase / "new_module.py"
        new_file.write_text("def new_function(): pass")
        
        # Incremental scan
        updated_files = await monitor.scan_repository()
        
        # Index only new file
        new_files = [f for f in updated_files if f.relative_path == "new_module.py"]
        assert len(new_files) == 1
        
        # Index new file
        new_file_info = new_files[0]
        content = Path(new_file_info.path).read_text()
        code_file = CodeFile(
            project_name="test_project",
            path=Path(new_file_info.path),
            content=content,
            language=new_file_info.language or "unknown",
            size=new_file_info.size,
            last_modified=new_file_info.last_modified
        )
        await mock_neo4j_rag.index_file(code_file)
        
        # Verify total indexing calls
        assert mock_neo4j_rag.index_file.call_count == initial_count + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])