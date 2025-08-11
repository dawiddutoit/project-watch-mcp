"""Neo4j-based RAG system for code retrieval.

Requires Neo4j 5.11+ with vector index support.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from neo4j import AsyncDriver, RoutingControl

logger = logging.getLogger(__name__)


@dataclass
class CodeFile:
    """Represents a code file to be indexed."""

    path: Path
    content: str
    language: str
    size: int
    last_modified: datetime

    @property
    def file_hash(self) -> str:
        """Generate a hash of the file content."""
        return hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class CodeChunk:
    """Represents a chunk of code."""

    file_path: Path
    content: str
    start_line: int
    end_line: int
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """Represents a search result."""

    file_path: Path
    content: str
    line_number: int
    similarity: float
    context: str | None = None


class EmbeddingsProvider:
    """Mock embeddings provider interface."""

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text."""
        # Mock implementation - in production, use real embeddings
        # (e.g., OpenAI, Sentence Transformers, etc.)
        return [0.1] * 384  # Mock 384-dimensional embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.embed_text(text) for text in texts]


class Neo4jRAG:
    """Neo4j-based RAG system for code retrieval."""

    def __init__(
        self,
        neo4j_driver: AsyncDriver,
        embeddings: EmbeddingsProvider | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the Neo4j RAG system.

        Args:
            neo4j_driver: Neo4j driver for database connection
            embeddings: Embeddings provider (defaults to mock)
            chunk_size: Size of code chunks in lines
            chunk_overlap: Overlap between chunks in lines
        """
        self.neo4j_driver = neo4j_driver
        self.embeddings = embeddings or EmbeddingsProvider()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def initialize(self):
        """Initialize the RAG system and create necessary indexes."""
        await self.create_indexes()

    async def create_indexes(self):
        """Create Neo4j indexes for efficient querying."""
        queries = [
            # Create index for file paths
            """
            CREATE INDEX file_path_index IF NOT EXISTS
            FOR (f:CodeFile) ON (f.path)
            """,
            # Create index for languages
            """
            CREATE INDEX file_language_index IF NOT EXISTS
            FOR (f:CodeFile) ON (f.language)
            """,
            # Create fulltext index for code search
            """
            CREATE FULLTEXT INDEX code_search IF NOT EXISTS
            FOR (c:CodeChunk) ON EACH [c.content]
            """,
        ]

        for query in queries:
            try:
                await self.neo4j_driver.execute_query(query, routing_control=RoutingControl.WRITE)
                logger.debug(f"Created index: {query.split()[2]}")
            except Exception as e:
                # Index might already exist or syntax not supported
                logger.debug(f"Index creation: {e}")
        
        # Try to create vector index if Neo4j version supports it
        try:
            vector_query = """
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:CodeChunk) ON (c.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """
            await self.neo4j_driver.execute_query(vector_query, routing_control=RoutingControl.WRITE)
            logger.info("Created vector index for semantic search")
        except Exception as e:
            logger.warning(f"Vector index not supported in this Neo4j version: {e}")
            logger.info("Semantic search will use fallback cosine similarity calculation")

    def chunk_content(self, content: str, chunk_size: int, overlap: int) -> list[str]:
        """
        Split content into overlapping chunks.

        Args:
            content: Content to chunk
            chunk_size: Size of each chunk in lines
            overlap: Number of overlapping lines

        Returns:
            List of content chunks
        """
        lines = content.split("\n")
        chunks = []

        if len(lines) <= chunk_size:
            return [content]

        i = 0
        while i < len(lines):
            chunk_lines = lines[i : i + chunk_size]
            chunks.append("\n".join(chunk_lines))
            i += chunk_size - overlap

        return chunks

    async def index_file(self, code_file: CodeFile):
        """
        Index a code file in Neo4j.

        Args:
            code_file: CodeFile object to index
        """
        # Create or update file node
        file_query = """
        MERGE (f:CodeFile {path: $path})
        SET f.language = $language,
            f.size = $size,
            f.last_modified = $last_modified,
            f.hash = $hash
        RETURN f
        """

        file_result = await self.neo4j_driver.execute_query(
            file_query,
            {
                "path": str(code_file.path),
                "language": code_file.language,
                "size": code_file.size,
                "last_modified": code_file.last_modified.isoformat(),
                "hash": code_file.file_hash,
            },
            routing_control=RoutingControl.WRITE,
        )

        # Delete existing chunks for this file
        delete_chunks_query = """
        MATCH (f:CodeFile {path: $path})-[:HAS_CHUNK]->(c:CodeChunk)
        DETACH DELETE c
        """

        await self.neo4j_driver.execute_query(
            delete_chunks_query, {"path": str(code_file.path)}, routing_control=RoutingControl.WRITE
        )

        # Create chunks and index them
        chunks = self.chunk_content(code_file.content, self.chunk_size, self.chunk_overlap)

        lines = code_file.content.split("\n")
        current_line = 0

        for i, chunk in enumerate(chunks):
            chunk_lines = chunk.split("\n")
            start_line = current_line + 1
            end_line = current_line + len(chunk_lines)

            # Generate embedding for chunk
            embedding = await self.embeddings.embed_text(chunk)

            # Create chunk node
            chunk_query = """
            MATCH (f:CodeFile {path: $file_path})
            CREATE (c:CodeChunk {
                content: $content,
                start_line: $start_line,
                end_line: $end_line,
                embedding: $embedding,
                chunk_index: $chunk_index
            })
            CREATE (f)-[:HAS_CHUNK]->(c)
            """

            await self.neo4j_driver.execute_query(
                chunk_query,
                {
                    "file_path": str(code_file.path),
                    "content": chunk,
                    "start_line": start_line,
                    "end_line": end_line,
                    "embedding": embedding,
                    "chunk_index": i,
                },
                routing_control=RoutingControl.WRITE,
            )

            current_line = start_line + (self.chunk_size - self.chunk_overlap) - 1

        logger.info(f"Indexed file {code_file.path} with {len(chunks)} chunks")

    async def update_file(self, code_file: CodeFile):
        """Update an existing file in the graph."""
        # Check if file hash has changed
        check_query = """
        MATCH (f:CodeFile {path: $path})
        RETURN f.hash as hash
        """

        result = await self.neo4j_driver.execute_query(
            check_query, {"path": str(code_file.path)}, routing_control=RoutingControl.READ
        )

        if result.records:
            existing_hash = result.records[0].get("hash")
            if existing_hash == code_file.file_hash:
                logger.debug(f"File {code_file.path} unchanged, skipping update")
                return

        # Re-index the file
        await self.index_file(code_file)

    async def delete_file(self, file_path: Path):
        """
        Delete a file and its chunks from the graph.

        Args:
            file_path: Path of the file to delete
        """
        query = """
        MATCH (f:CodeFile {path: $path})
        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CodeChunk)
        DETACH DELETE f, c
        """

        await self.neo4j_driver.execute_query(
            query, {"path": str(file_path)}, routing_control=RoutingControl.WRITE
        )

        logger.info(f"Deleted file {file_path} from graph")

    async def search_semantic(
        self,
        query: str,
        limit: int = 10,
        language: str | None = None,
    ) -> list[SearchResult]:
        """
        Perform semantic search using embeddings.

        Args:
            query: Search query
            limit: Maximum number of results
            language: Filter by programming language

        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = await self.embeddings.embed_text(query)

        # Try vector index first (Neo4j 5.11+)
        try:
            cypher = """
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
            YIELD node as c, score
            MATCH (f:CodeFile)-[:HAS_CHUNK]->(c)
            """
            if language:
                cypher += " WHERE f.language = $language"
            
            cypher += """
            RETURN f.path as file_path,
                   c.content as chunk_content,
                   c.start_line as line_number,
                   score as similarity
            ORDER BY score DESC
            """
            
            params = {
                "query_embedding": query_embedding,
                "k": limit,
            }
            if language:
                params["language"] = language
            
            result = await self.neo4j_driver.execute_query(
                cypher, params, routing_control=RoutingControl.READ
            )
            
        except Exception as e:
            logger.debug(f"Vector index search failed, using fallback: {e}")
            
            # Fallback: Calculate cosine similarity manually
            cypher = """
            MATCH (f:CodeFile)-[:HAS_CHUNK]->(c:CodeChunk)
            """
            
            if language:
                cypher += " WHERE f.language = $language"
            
            # Manual cosine similarity calculation
            cypher += """
            WITH f, c,
                 reduce(dot = 0.0, i IN range(0, size($query_embedding)-1) | 
                        dot + c.embedding[i] * $query_embedding[i]) AS dotProduct,
                 reduce(qNorm = 0.0, i IN range(0, size($query_embedding)-1) | 
                        qNorm + $query_embedding[i] * $query_embedding[i]) AS queryNorm,
                 reduce(cNorm = 0.0, i IN range(0, size(c.embedding)-1) | 
                        cNorm + c.embedding[i] * c.embedding[i]) AS chunkNorm
            WITH f, c, 
                 CASE WHEN queryNorm > 0 AND chunkNorm > 0 
                      THEN dotProduct / (sqrt(queryNorm) * sqrt(chunkNorm))
                      ELSE 0.0 END AS similarity
            WHERE similarity > 0.3
            RETURN f.path as file_path,
                   c.content as chunk_content,
                   c.start_line as line_number,
                   similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            
            params = {
                "query_embedding": query_embedding,
                "limit": limit,
            }
            if language:
                params["language"] = language
            
            result = await self.neo4j_driver.execute_query(
                cypher, params, routing_control=RoutingControl.READ
            )

        search_results = []
        for record in result.records:
            search_results.append(
                SearchResult(
                    file_path=Path(record["file_path"]),
                    content=record["chunk_content"],
                    line_number=record["line_number"],
                    similarity=record["similarity"],
                )
            )

        return search_results

    async def search_by_pattern(
        self,
        pattern: str,
        is_regex: bool = False,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search for code by pattern or regex.

        Args:
            pattern: Search pattern
            is_regex: Whether pattern is a regex
            limit: Maximum number of results

        Returns:
            List of search results
        """
        if is_regex:
            # Use regex matching
            query = """
            MATCH (f:CodeFile)-[:HAS_CHUNK]->(c:CodeChunk)
            WHERE c.content =~ $pattern
            RETURN f.path as file_path,
                   c.content as content,
                   c.start_line as line_number
            LIMIT $limit
            """
        else:
            # Use fulltext search
            query = """
            CALL db.index.fulltext.queryNodes('code_search', $pattern)
            YIELD node as c, score
            MATCH (f:CodeFile)-[:HAS_CHUNK]->(c)
            RETURN f.path as file_path,
                   c.content as content,
                   c.start_line as line_number,
                   score as similarity
            ORDER BY score DESC
            LIMIT $limit
            """

        result = await self.neo4j_driver.execute_query(
            query, {"pattern": pattern, "limit": limit}, routing_control=RoutingControl.READ
        )

        search_results = []
        for record in result.records:
            search_results.append(
                SearchResult(
                    file_path=Path(record["file_path"]),
                    content=record["content"],
                    line_number=record["line_number"],
                    similarity=record.get("similarity", 1.0),
                )
            )

        return search_results

    async def get_file_metadata(self, file_path: Path) -> dict[str, Any] | None:
        """
        Get metadata for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            File metadata or None if not found
        """
        query = """
        MATCH (f:CodeFile {path: $path})
        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CodeChunk)
        RETURN f.path as path,
               f.language as language,
               f.size as size,
               f.last_modified as last_modified,
               f.hash as hash,
               count(c) as chunk_count
        """

        result = await self.neo4j_driver.execute_query(
            query, {"path": str(file_path)}, routing_control=RoutingControl.READ
        )

        if result.records:
            record = result.records[0]
            return {
                "path": record["path"],
                "language": record["language"],
                "size": record["size"],
                "last_modified": record["last_modified"],
                "hash": record["hash"],
                "chunk_count": record["chunk_count"],
            }

        return None

    async def get_repository_stats(self) -> dict[str, Any]:
        """
        Get statistics about the indexed repository.

        Returns:
            Dictionary with repository statistics
        """
        query = """
        MATCH (f:CodeFile)
        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CodeChunk)
        RETURN count(DISTINCT f) as total_files,
               count(c) as total_chunks,
               sum(f.size) as total_size,
               collect(DISTINCT f.language) as languages
        """

        result = await self.neo4j_driver.execute_query(query, routing_control=RoutingControl.READ)

        if result.records:
            record = result.records[0]
            return {
                "total_files": record["total_files"],
                "total_chunks": record["total_chunks"],
                "total_size": record["total_size"] or 0,
                "languages": record["languages"],
            }

        return {
            "total_files": 0,
            "total_chunks": 0,
            "total_size": 0,
            "languages": [],
        }
