"""Neo4j-based RAG system for code retrieval.

Requires Neo4j 5.11+ with vector index support.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from neo4j import AsyncDriver, RoutingControl

from .config import EmbeddingConfig
from .utils.embeddings import (
    EmbeddingsProvider,
    OpenAIEmbeddingsProvider,
    create_embeddings_provider,
)

logger = logging.getLogger(__name__)


def lucene_phrase(query: str) -> str:
    """Wrap user text as a Lucene phrase for literal matching.

    We only escape backslashes and quotes inside the phrase.
    Example: search [item] -> "search [item]"
             path("a\b")  -> "path(\"a\\b\")"
    """
    if not query:
        return query
    
    # Import here to avoid circular imports
    from . import __lucene_fix_version__, __build_timestamp__
    logger.info(f"[LUCENE-PHRASE {__lucene_fix_version__}] Converting to phrase: {query!r}")
    logger.debug(f"Build timestamp: {__build_timestamp__}")
    
    q = query.replace("\\", "\\\\").replace('"', '\\"')
    result = f'"{q}"'
    
    logger.info(f"[LUCENE-PHRASE {__lucene_fix_version__}] Result: {result!r}")
    return result


def escape_lucene_query(query: str) -> str:
    """Escape special characters in Lucene query syntax.
    
    Escapes special characters that have meaning in Lucene query syntax.
    Note: In Neo4j's Lucene implementation, backslashes need to be double-escaped.
    
    Args:
        query: The raw query string to escape
        
    Returns:
        Query string with special characters escaped for Lucene
    """
    if not query:
        return query
    
    # Special characters that need escaping in Lucene
    # Note: && and || are operators, single & and | are not
    special_chars = [
        '+', '-', '!', '(', ')', '{', '}', '[', ']', '^',
        '"', '~', '*', '?', ':', '\\', '/'
    ]
    
    result = query
    
    # Escape && and || operators (but not single & or |)
    result = result.replace('&&', '\\&&')
    result = result.replace('||', '\\||')
    
    # Escape other special characters with double backslash for Neo4j
    for char in special_chars:
        result = result.replace(char, f'\\\\{char}')
    
    # Handle arrow operator specially
    result = result.replace('->', '\\\\->')
    
    return result


@dataclass
class CodeFile:
    """Represents a code file to be indexed."""

    project_name: str
    path: Path
    content: str
    language: str
    size: int
    last_modified: datetime

    # Optional enhanced metadata fields
    filename: str | None = None
    namespace: str | None = None
    is_test: bool = False
    is_config: bool = False
    is_resource: bool = False
    is_documentation: bool = False
    package_path: str | None = None
    module_imports: list[str] | None = None
    exported_symbols: list[str] | None = None
    dependencies: list[str] | None = None
    complexity_score: int | None = None
    line_count: int | None = None

    def __post_init__(self):
        """Initialize derived fields after dataclass initialization."""
        # Auto-populate filename if not provided
        if self.filename is None:
            self.filename = self.path.name

        # Auto-detect file type if not explicitly set
        if not any([self.is_test, self.is_config, self.is_resource, self.is_documentation]):
            self._detect_file_type()

        # Calculate line count if not provided
        if self.line_count is None:
            # Empty content should still count as 1 line (like most editors show)
            self.line_count = max(1, len(self.content.splitlines()))

        # Extract namespace/package for supported languages
        if self.namespace is None and self.language in [
            "python",
            "java",
            "csharp",
            "typescript",
            "javascript",
        ]:
            self._extract_namespace()

    def _detect_file_type(self):
        """Auto-detect the type of file based on naming patterns and content."""
        filename_lower = self.filename.lower()
        path_str = str(self.path).lower()

        # Test file detection
        test_patterns = [
            "test_",
            "_test.",
            "spec.",
            ".test.",
            ".spec.",
            "tests/",
            "test/",
            "__tests__/",
        ]
        self.is_test = any(
            pattern in filename_lower or pattern in path_str for pattern in test_patterns
        )

        # Config file detection
        config_patterns = ["config", "settings", ".yaml", ".yml", ".toml", ".ini", ".env", ".json"]
        config_names = [
            "package.json",
            "pyproject.toml",
            "tsconfig.json",
            "webpack.config",
            "jest.config",
        ]
        # Check for rc files (e.g., .bashrc, .vimrc)
        is_rc_file = filename_lower.startswith(".") and filename_lower.endswith("rc")
        self.is_config = (
            any(pattern in filename_lower for pattern in config_patterns)
            or any(name in filename_lower for name in config_names)
            or is_rc_file
        )

        # Resource file detection
        resource_extensions = [
            ".sql",
            ".xml",
            ".csv",
            ".txt",
            ".dat",
            ".png",
            ".jpg",
            ".svg",
            ".css",
            ".scss",
        ]
        self.is_resource = any(filename_lower.endswith(ext) for ext in resource_extensions)

        # Documentation detection
        doc_patterns = [".md", ".rst", ".adoc", "readme", "changelog", "license", "contributing"]
        self.is_documentation = any(pattern in filename_lower for pattern in doc_patterns)

    def _extract_namespace(self):
        """Extract namespace or package information based on language."""
        if self.language == "python":
            # Extract from package structure
            parts = self.path.parts
            if "src" in parts:
                idx = parts.index("src")
                package_parts = [p for p in parts[idx + 1 : -1] if not p.startswith("__")]
                if package_parts:
                    self.namespace = ".".join(package_parts)
            elif "site-packages" in str(self.path):
                # Handle installed packages
                path_str = str(self.path)
                idx = path_str.find("site-packages/") + len("site-packages/")
                remaining = path_str[idx:]
                package_name = remaining.split("/")[0]
                self.namespace = package_name
        elif self.language == "java":
            # Extract package declaration
            package_match = re.search(
                r"^\s*package\s+([a-zA-Z0-9_.]+)\s*;", self.content, re.MULTILINE
            )
            if package_match:
                self.namespace = package_match.group(1)
        elif self.language == "csharp":
            # Extract namespace declaration
            namespace_match = re.search(
                r"^\s*namespace\s+([a-zA-Z0-9_.]+)", self.content, re.MULTILINE
            )
            if namespace_match:
                self.namespace = namespace_match.group(1)
        elif self.language in ["typescript", "javascript"]:
            # Use module path relative to src or project root
            parts = self.path.parts
            if "src" in parts:
                idx = parts.index("src")
                module_parts = parts[idx + 1 : -1]
                if module_parts:
                    self.namespace = "/".join(module_parts)

    @property
    def file_hash(self) -> str:
        """Generate a hash of the file content."""
        return hashlib.sha256(self.content.encode()).hexdigest()

    @property
    def file_category(self) -> str:
        """Get the primary category of the file."""
        if self.is_test:
            return "test"
        elif self.is_config:
            return "config"
        elif self.is_resource:
            return "resource"
        elif self.is_documentation:
            return "documentation"
        else:
            return "source"


@dataclass
class CodeChunk:
    """Represents a chunk of code."""

    project_name: str
    file_path: Path
    content: str
    start_line: int
    end_line: int
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """Represents a search result."""

    project_name: str
    file_path: Path
    content: str
    line_number: int
    similarity: float
    context: str | None = None


class Neo4jRAG:
    """Neo4j-based RAG system for code retrieval."""

    def __init__(
        self,
        neo4j_driver: AsyncDriver,
        project_name: str,
        embeddings: EmbeddingsProvider | None = None,
        embedding_config: EmbeddingConfig | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the Neo4j RAG system.

        Args:
            neo4j_driver: Neo4j driver for database connection
            project_name: Name of the project for context isolation
            embeddings: Embeddings provider (takes precedence over config)
            embedding_config: Configuration for creating embeddings provider
            chunk_size: Size of code chunks in lines
            chunk_overlap: Overlap between chunks in lines
        """
        self.neo4j_driver = neo4j_driver
        self.project_name = project_name

        # Initialize embeddings provider
        if embeddings is not None:
            # Use provided embeddings provider
            self.embeddings = embeddings
        elif embedding_config is not None:
            # Create provider from config
            self.embeddings = self._create_provider_from_config(embedding_config)
        else:
            # Default behavior: try OpenAI, disable if no API key
            try:
                self.embeddings = OpenAIEmbeddingsProvider()
            except ValueError:
                logger.warning("OpenAI API key not found. Embeddings disabled.")
                logger.warning("Semantic search features will not be available.")
                self.embeddings = None

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _create_provider_from_config(self, config: EmbeddingConfig) -> EmbeddingsProvider:
        """
        Create embeddings provider from configuration.

        Args:
            config: Embedding configuration

        Returns:
            Configured embeddings provider
        """
        kwargs = {}

        if config.model:
            kwargs["model"] = config.model
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.dimension:
            kwargs["dimension"] = config.dimension

        return create_embeddings_provider(config.provider, **kwargs)

    async def initialize(self):
        """Initialize the RAG system and create necessary indexes."""
        await self.create_indexes()

    async def create_indexes(self):
        """Create Neo4j indexes for efficient querying with project isolation."""
        queries = [
            # Composite indexes for project isolation
            """
            CREATE INDEX project_file_path_index IF NOT EXISTS
            FOR (f:CodeFile) ON (f.project_name, f.path)
            """,
            """
            CREATE INDEX project_name_index IF NOT EXISTS
            FOR (f:CodeFile) ON (f.project_name)
            """,
            # Create index for languages with project
            """
            CREATE INDEX project_language_index IF NOT EXISTS
            FOR (f:CodeFile) ON (f.project_name, f.language)
            """,
            # Create composite index for chunks
            """
            CREATE INDEX chunk_project_path_index IF NOT EXISTS
            FOR (c:CodeChunk) ON (c.project_name, c.file_path)
            """,
            """
            CREATE INDEX chunk_project_index IF NOT EXISTS
            FOR (c:CodeChunk) ON (c.project_name)
            """,
            # Create fulltext index for code search with code-friendly analyzer
            """
            CREATE FULLTEXT INDEX code_search IF NOT EXISTS
            FOR (c:CodeChunk) ON EACH [c.content]
            OPTIONS {
                indexConfig: {
                    `fulltext.analyzer`: 'keyword',
                    `fulltext.eventually_consistent`: false
                }
            }
            """,
        ]

        for query in queries:
            try:
                await self.neo4j_driver.execute_query(query, routing_control=RoutingControl.WRITE)
                logger.debug(f"Created index: {query.split()[2]}")
            except Exception as e:
                # Index might already exist or syntax not supported
                logger.debug(f"Index creation: {e}")

        # Try to create vector index if Neo4j version supports it and embeddings are enabled
        if self.embeddings:
            try:
                # Get embedding dimension from provider
                embedding_dim = self.embeddings.dimension

                vector_query = f"""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:CodeChunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {embedding_dim},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
                await self.neo4j_driver.execute_query(
                    vector_query, routing_control=RoutingControl.WRITE
                )
                logger.info(f"Created vector index for semantic search with {embedding_dim} dimensions")
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
        Index a code file in Neo4j with project context.

        Args:
            code_file: CodeFile object to index
        """
        # Ensure the code_file has the correct project name
        if code_file.project_name != self.project_name:
            code_file = CodeFile(
                project_name=self.project_name,
                path=code_file.path,
                content=code_file.content,
                language=code_file.language,
                size=code_file.size,
                last_modified=code_file.last_modified,
            )

        # Create or update file node with project context
        file_query = """
        MERGE (f:CodeFile {project_name: $project_name, path: $path})
        SET f.language = $language,
            f.size = $size,
            f.last_modified = $last_modified,
            f.hash = $hash
        RETURN f
        """

        await self.neo4j_driver.execute_query(
            file_query,
            {
                "project_name": self.project_name,
                "path": str(code_file.path),
                "language": code_file.language,
                "size": code_file.size,
                "last_modified": code_file.last_modified.isoformat(),
                "hash": code_file.file_hash,
            },
            routing_control=RoutingControl.WRITE,
        )

        # Delete existing chunks for this file IN THIS PROJECT ONLY
        delete_chunks_query = """
        MATCH (f:CodeFile {project_name: $project_name, path: $path})-[:HAS_CHUNK]->(c:CodeChunk)
        DETACH DELETE c
        """

        await self.neo4j_driver.execute_query(
            delete_chunks_query,
            {"project_name": self.project_name, "path": str(code_file.path)},
            routing_control=RoutingControl.WRITE,
        )

        # Create chunks and index them
        chunks = self.chunk_content(code_file.content, self.chunk_size, self.chunk_overlap)

        current_line = 0

        for i, chunk in enumerate(chunks):
            chunk_lines = chunk.split("\n")
            start_line = current_line + 1
            end_line = current_line + len(chunk_lines)

            # Generate embedding for chunk if embeddings are enabled
            if self.embeddings:
                embedding = await self.embeddings.embed_text(chunk)
            else:
                embedding = None

            # Create chunk node with project context
            chunk_query = """
            MATCH (f:CodeFile {project_name: $project_name, path: $file_path})
            CREATE (c:CodeChunk {
                project_name: $project_name,
                file_path: $file_path,
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
                    "project_name": self.project_name,
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
        """Update an existing file in the graph with project context."""
        # Check if file hash has changed for this project
        check_query = """
        MATCH (f:CodeFile {project_name: $project_name, path: $path})
        RETURN f.hash as hash
        """

        result = await self.neo4j_driver.execute_query(
            check_query,
            {"project_name": self.project_name, "path": str(code_file.path)},
            routing_control=RoutingControl.READ,
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
        Delete a file and its chunks from the graph for this project only.

        Args:
            file_path: Path of the file to delete
        """
        query = """
        MATCH (f:CodeFile {project_name: $project_name, path: $path})
        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CodeChunk)
        DETACH DELETE f, c
        """

        await self.neo4j_driver.execute_query(
            query,
            {"project_name": self.project_name, "path": str(file_path)},
            routing_control=RoutingControl.WRITE,
        )

        logger.info(f"Deleted file {file_path} from project {self.project_name}")

    async def search_semantic(
        self,
        query: str,
        limit: int = 10,
        language: str | None = None,
    ) -> list[SearchResult]:
        """
        Perform semantic search using embeddings with project context.

        Args:
            query: Search query
            limit: Maximum number of results
            language: Filter by programming language

        Returns:
            List of search results
        """
        # Check if embeddings are available
        if not self.embeddings:
            logger.warning("Semantic search unavailable - embeddings disabled")
            return []

        # Generate query embedding
        query_embedding = await self.embeddings.embed_text(query)

        # Try vector index first (Neo4j 5.11+)
        try:
            cypher = """
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
            YIELD node as c, score
            MATCH (f:CodeFile)-[:HAS_CHUNK]->(c)
            WHERE c.project_name = $project_name
            """
            if language:
                cypher += " AND f.language = $language"

            cypher += """
            RETURN f.path as file_path,
                   c.content as chunk_content,
                   c.start_line as line_number,
                   score as similarity,
                   c.project_name as project_name
            ORDER BY score DESC
            """

            params = {
                "project_name": self.project_name,
                "query_embedding": query_embedding,
                "k": limit * 3,  # Get more results initially to filter by project
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
            WHERE c.project_name = $project_name
            """

            if language:
                cypher += " AND f.language = $language"

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
                   similarity,
                   c.project_name as project_name
            ORDER BY similarity DESC
            LIMIT $limit
            """

            params = {
                "project_name": self.project_name,
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
                    project_name=record.get("project_name", self.project_name),
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
        Search for code by pattern (literal phrase via Lucene) or by regex (Cypher =~).

        Args:
            pattern: Search pattern
            is_regex: Whether pattern is a regex
            limit: Maximum number of results

        Returns:
            List of search results
        """
        if is_regex:
            # Regex path (no Lucene or fulltext index used)
            query = """
            MATCH (f:CodeFile)-[:HAS_CHUNK]->(c:CodeChunk)
            WHERE c.project_name = $project_name AND c.content =~ $pattern
            RETURN f.path as file_path,
                   c.content as content,
                   c.start_line as line_number,
                   c.project_name as project_name
            LIMIT $limit
            """
            query_pattern = pattern

        else:
            # Treat the user input as a literal phrase, not as Lucene syntax.
            # This makes characters like [ ] ( ) : etc. safe without fiddly escapes.
            phrase = lucene_phrase(pattern)

            logger.info(f"Pattern search (phrase): raw={pattern!r} phrase={phrase!r}")

            query = """
            CALL db.index.fulltext.queryNodes('code_search', $pattern)
            YIELD node as c, score
            MATCH (f:CodeFile)-[:HAS_CHUNK]->(c)
            WHERE c.project_name = $project_name
            RETURN f.path as file_path,
                   c.content as content,
                   c.start_line as line_number,
                   score as similarity,
                   c.project_name as project_name
            ORDER BY score DESC
            LIMIT $limit
            """
            query_pattern = phrase

        result = await self.neo4j_driver.execute_query(
            query,
            {"project_name": self.project_name, "pattern": query_pattern, "limit": limit},
            routing_control=RoutingControl.READ,
        )

        out: list[SearchResult] = []
        for record in result.records:
            out.append(
                SearchResult(
                    project_name=record.get("project_name", self.project_name),
                    file_path=Path(record["file_path"]),
                    content=record["content"],
                    line_number=record["line_number"],
                    similarity=record.get("similarity", 1.0),
                )
            )
        return out

    async def get_file_metadata(self, file_path: Path) -> dict[str, Any] | None:
        """
        Get metadata for a specific file in this project.

        Args:
            file_path: Path to the file

        Returns:
            File metadata or None if not found
        """
        query = """
        MATCH (f:CodeFile {project_name: $project_name, path: $path})
        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CodeChunk)
        RETURN f.path as path,
               f.language as language,
               f.size as size,
               f.last_modified as last_modified,
               f.hash as hash,
               f.project_name as project_name,
               count(c) as chunk_count
        """

        result = await self.neo4j_driver.execute_query(
            query,
            {"project_name": self.project_name, "path": str(file_path)},
            routing_control=RoutingControl.READ,
        )

        if result.records:
            record = result.records[0]
            return {
                "path": record["path"],
                "language": record["language"],
                "size": record["size"],
                "last_modified": record["last_modified"],
                "hash": record["hash"],
                "project_name": record.get("project_name", self.project_name),
                "chunk_count": record["chunk_count"],
            }

        return None

    async def get_repository_stats(self) -> dict[str, Any]:
        """
        Get statistics about the indexed repository for this project.

        Returns:
            Dictionary with repository statistics
        """
        query = """
        MATCH (f:CodeFile)
        WHERE f.project_name = $project_name
        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CodeChunk)
        RETURN count(DISTINCT f) as total_files,
               count(c) as total_chunks,
               sum(f.size) as total_size,
               collect(DISTINCT f.language) as languages,
               $project_name as project_name
        """

        result = await self.neo4j_driver.execute_query(
            query, {"project_name": self.project_name}, routing_control=RoutingControl.READ
        )

        if result.records:
            record = result.records[0]
            return {
                "project_name": record.get("project_name", self.project_name),
                "total_files": record["total_files"],
                "total_chunks": record["total_chunks"],
                "total_size": record["total_size"] or 0,
                "languages": record["languages"],
            }

        return {
            "project_name": self.project_name,
            "total_files": 0,
            "total_chunks": 0,
            "total_size": 0,
            "languages": [],
        }

    async def close(self):
        """
        Close the Neo4j RAG system and clean up resources.

        Properly closes the Neo4j driver connection to prevent resource leaks.
        This method should be called when the RAG system is no longer needed.
        """
        if self.neo4j_driver:
            await self.neo4j_driver.close()
            logger.info(f"Closed Neo4j RAG system for project {self.project_name}")
