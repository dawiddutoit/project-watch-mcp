"""Neo4j-based RAG system for code retrieval.

Requires Neo4j 5.11+ with vector index support.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import xxhash
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
    from . import __build_timestamp__, __lucene_fix_version__
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

    def __post_init__(self):
        """Initialize derived fields after dataclass initialization."""
        # Auto-populate filename if not provided
        if self.filename is None:
            self.filename = self.path.name

        # Auto-detect file type if not explicitly set
        if not any([self.is_test, self.is_config, self.is_resource, self.is_documentation]):
            self._detect_file_type()

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
        """Generate a hash of the file content using xxHash for speed."""
        return xxhash.xxh64(self.content.encode()).hexdigest()

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
        enable_file_classification: bool = True,
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
        self.enable_file_classification = enable_file_classification

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

        # Add classification indexes if enabled
        if self.enable_file_classification:
            queries.extend([
                # Index for file categories
                """
                CREATE INDEX project_file_category_index IF NOT EXISTS
                FOR (f:CodeFile) ON (f.project_name, f.file_category)
                """,
                # Index for test files
                """
                CREATE INDEX project_is_test_index IF NOT EXISTS
                FOR (f:CodeFile) ON (f.project_name, f.is_test)
                """,
                # Index for config files
                """
                CREATE INDEX project_is_config_index IF NOT EXISTS
                FOR (f:CodeFile) ON (f.project_name, f.is_config)
                """,
                # Index for documentation files
                """
                CREATE INDEX project_is_documentation_index IF NOT EXISTS
                FOR (f:CodeFile) ON (f.project_name, f.is_documentation)
                """,
                # Index for resource files
                """
                CREATE INDEX project_is_resource_index IF NOT EXISTS
                FOR (f:CodeFile) ON (f.project_name, f.is_resource)
                """,
                # Index for namespace
                """
                CREATE INDEX project_namespace_index IF NOT EXISTS
                FOR (f:CodeFile) ON (f.project_name, f.namespace)
                """,
            ])

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

    def _sanitize_for_lucene(self, text: str) -> str:
        """
        Sanitize text for Lucene indexing by ensuring no single term exceeds byte limit.
        
        Lucene has a hard limit of 32,766 bytes per TERM (not per field).
        A term is typically a word separated by whitespace or punctuation.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Text with oversized terms truncated or split
        """
        LUCENE_TERM_BYTE_LIMIT = 32000  # Slightly less than 32,766 for safety

        # Split by whitespace to get individual terms
        terms = text.split()
        sanitized_terms = []

        for term in terms:
            term_bytes = len(term.encode('utf-8'))

            if term_bytes <= LUCENE_TERM_BYTE_LIMIT:
                sanitized_terms.append(term)
            else:
                # This is an extremely long "word" (likely base64 data or similar)
                # We need to break it up or truncate it
                logger.warning(f"Found oversized term with {term_bytes} bytes, truncating for Lucene safety")

                # Truncate the term to fit within the byte limit
                # Use binary search to find the right character count
                left, right = 0, len(term)
                best_fit = 0

                while left <= right:
                    mid = (left + right) // 2
                    test_term = term[:mid]
                    test_bytes = len(test_term.encode('utf-8'))

                    if test_bytes <= LUCENE_TERM_BYTE_LIMIT:
                        best_fit = mid
                        left = mid + 1
                    else:
                        right = mid - 1

                truncated = term[:best_fit]
                sanitized_terms.append(truncated)

                # Log that we truncated
                logger.info(f"Truncated term from {len(term)} chars to {len(truncated)} chars")

        return " ".join(sanitized_terms)

    def chunk_content(self, content: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """
        Split content into overlapping chunks with Lucene byte limit awareness.
        
        Ensures chunks never exceed Neo4j Lucene's 32,766 byte limit while 
        maintaining semantic coherence and proper overlap for context.

        Args:
            content: Content to chunk
            chunk_size: Target size of each chunk in lines (for normal files)
            overlap: Number of overlapping lines (for normal files)

        Returns:
            List of content chunks that are safe for Lucene indexing
        """
        # CRITICAL: Neo4j Lucene has a hard limit of 32,766 bytes per term
        # We use 30,000 bytes as safe limit to account for any encoding overhead
        LUCENE_SAFE_BYTE_LIMIT = 30000

        # Check if content is small enough to be a single chunk
        content_bytes = len(content.encode('utf-8'))
        if content_bytes <= LUCENE_SAFE_BYTE_LIMIT:
            return [content]

        # For large content, we need to chunk intelligently
        # First try line-based chunking for better semantic boundaries
        lines = content.split("\n")
        chunks = []

        # If content is a single line that's too large, split it directly
        if len(lines) == 1 and content_bytes > LUCENE_SAFE_BYTE_LIMIT:
            return self._split_large_line(content, LUCENE_SAFE_BYTE_LIMIT)

        if len(lines) <= chunk_size and content_bytes <= LUCENE_SAFE_BYTE_LIMIT:
            return [content]

        # Line-based chunking with byte size validation
        i = 0
        while i < len(lines):
            # First check if the current line alone exceeds the limit
            current_line = lines[i]
            current_line_bytes = len(current_line.encode('utf-8'))

            if current_line_bytes > LUCENE_SAFE_BYTE_LIMIT:
                # This single line is too large, split it
                line_chunks = self._split_large_line(current_line, LUCENE_SAFE_BYTE_LIMIT)
                chunks.extend(line_chunks)
                i += 1
                continue

            # Try to build a chunk with multiple lines
            chunk_lines = []
            chunk_bytes = 0
            j = i

            while j < len(lines) and j < i + chunk_size:
                line = lines[j]
                line_bytes = len(line.encode('utf-8'))

                # Check if this individual line is too large
                if line_bytes > LUCENE_SAFE_BYTE_LIMIT:
                    # Stop here and process what we have so far
                    break

                # Check if adding this line would exceed the limit
                potential_chunk = "\n".join(chunk_lines + [line])
                potential_bytes = len(potential_chunk.encode('utf-8'))

                if potential_bytes > LUCENE_SAFE_BYTE_LIMIT:
                    # Stop here, don't add this line
                    break

                chunk_lines.append(line)
                chunk_bytes = potential_bytes
                j += 1

            # Add the chunk if we collected any lines
            if chunk_lines:
                chunk = "\n".join(chunk_lines)
                chunks.append(chunk)
                # Move forward with overlap
                lines_consumed = len(chunk_lines)
                i += max(1, lines_consumed - overlap)
            else:
                # No lines collected (shouldn't happen due to earlier checks)
                i += 1

        return chunks

    def _split_large_line(self, line: str, max_bytes: int) -> list[str]:
        """
        Split a single large line that exceeds the byte limit.
        
        Args:
            line: The line to split
            max_bytes: Maximum bytes per chunk
            
        Returns:
            List of line chunks
        """
        chunks = []
        line_bytes = line.encode('utf-8')

        if len(line_bytes) <= max_bytes:
            return [line]

        # Binary search to find safe split points
        start = 0
        while start < len(line):
            # Find the maximum substring that fits
            left, right = 1, min(len(line) - start, max_bytes)
            best_fit = 1

            while left <= right:
                mid = (left + right) // 2
                test_chunk = line[start:start + mid]
                test_bytes = len(test_chunk.encode('utf-8'))

                if test_bytes <= max_bytes:
                    best_fit = mid
                    left = mid + 1
                else:
                    right = mid - 1

            # Try to find a natural break point (space, punctuation)
            chunk_end = start + best_fit
            if chunk_end < len(line):
                # Look for a natural break point in the last 10% of the chunk
                search_start = start + int(best_fit * 0.9)
                for break_char in [' ', ',', ';', ')', '}', ']', '>', '\t']:
                    break_pos = line.rfind(break_char, search_start, chunk_end)
                    if break_pos > start:
                        chunk_end = break_pos + 1
                        break

            chunks.append(line[start:chunk_end])
            start = chunk_end

        return chunks

    def _chunk_by_tokens(self, content: str, max_tokens: int = 2000, overlap_tokens: int = 200) -> list[str]:
        """
        Chunk content by estimated token count with Lucene byte limit awareness.
        
        Args:
            content: Content to chunk
            max_tokens: Maximum tokens per chunk (default 2000, safe for embeddings)
            overlap_tokens: Overlap between chunks in tokens
        
        Returns:
            List of content chunks that are safe for Lucene indexing
        """
        LUCENE_SAFE_BYTE_LIMIT = 30000
        chunks = []

        # Convert to approximate character counts
        max_chars = max_tokens * 5
        overlap_chars = overlap_tokens * 5

        # Ensure we don't exceed Lucene byte limit even with token-based chunking
        max_chars = min(max_chars, LUCENE_SAFE_BYTE_LIMIT // 3)  # Conservative estimate for UTF-8

        # Split by natural boundaries if possible (paragraphs, then sentences)
        lines = content.split("\n")
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line)
            line_bytes = len(line.encode('utf-8'))

            # Check if single line exceeds byte limit
            if line_bytes > LUCENE_SAFE_BYTE_LIMIT:
                # This single line is too long, need to split it
                line_chunks = self._split_large_line(line, LUCENE_SAFE_BYTE_LIMIT)
                chunks.extend(line_chunks)
                continue

            # Check both character and byte limits
            potential_chunk = "\n".join(current_chunk + [line])
            potential_bytes = len(potential_chunk.encode('utf-8'))

            # If adding this line would exceed either limit, save current chunk
            if (current_size + line_size > max_chars or potential_bytes > LUCENE_SAFE_BYTE_LIMIT) and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)

                # Calculate overlap: keep last few lines for context
                overlap_size = 0
                overlap_lines = []
                for prev_line in reversed(current_chunk):
                    overlap_size += len(prev_line)
                    overlap_lines.insert(0, prev_line)
                    if overlap_size >= overlap_chars:
                        break

                current_chunk = overlap_lines
                current_size = sum(len(line) for line in current_chunk)

            current_chunk.append(line)
            current_size += line_size

        # Add remaining content
        if current_chunk:
            final_text = "\n".join(current_chunk)
            final_bytes = len(final_text.encode('utf-8'))

            # Check if the final chunk exceeds byte limit
            if final_bytes > LUCENE_SAFE_BYTE_LIMIT:
                # Split using our byte-aware splitter
                remaining_chunks = self._split_large_line(final_text, LUCENE_SAFE_BYTE_LIMIT)
                chunks.extend(remaining_chunks)
            else:
                chunks.append(final_text)

        # If no lines were processed, fall back to byte-aware splitting
        if not chunks and content:
            chunks = self._split_large_line(content, LUCENE_SAFE_BYTE_LIMIT)

        logger.info(f"Token-based chunking: {len(content)} chars -> {len(chunks)} chunks (byte-safe)")
        return chunks

    async def batch_index_files(self, code_files: list[CodeFile], batch_size: int = 100):
        """
        Index multiple code files using batch operations for improved performance.
        
        This method processes files in batches, using UNWIND operations to significantly
        reduce database round-trips and improve indexing speed by 50-100x.
        
        Args:
            code_files: List of CodeFile objects to index
            batch_size: Number of files to process in each batch (default: 100)
        """
        if not code_files:
            return

        logger.info(f"Starting batch indexing of {len(code_files)} files with batch size {batch_size}")

        # Process files in batches
        for batch_start in range(0, len(code_files), batch_size):
            batch_end = min(batch_start + batch_size, len(code_files))
            batch = code_files[batch_start:batch_end]

            # Prepare batch data for files
            file_data = []
            chunks_data = []

            for code_file in batch:
                # Ensure correct project name
                if code_file.project_name != self.project_name:
                    code_file = CodeFile(
                        project_name=self.project_name,
                        path=code_file.path,
                        content=code_file.content,
                        language=code_file.language,
                        size=code_file.size,
                        last_modified=code_file.last_modified,
                    )

                line_count = len(code_file.content.splitlines()) if code_file.content else 0

                file_item = {
                    "project_name": self.project_name,
                    "path": str(code_file.path),
                    "language": code_file.language,
                    "size": code_file.size,
                    "lines": line_count,
                    "last_modified": code_file.last_modified.isoformat(),
                    "hash": code_file.file_hash,
                }

                # Add classification fields if enabled
                if self.enable_file_classification:
                    file_item.update({
                        "is_test": code_file.is_test,
                        "is_config": code_file.is_config,
                        "is_resource": code_file.is_resource,
                        "is_documentation": code_file.is_documentation,
                        "file_category": code_file.file_category,
                        "namespace": code_file.namespace,
                    })

                file_data.append(file_item)

                # Create chunks for this file
                chunks = self.chunk_content(code_file.content, self.chunk_size, self.chunk_overlap)
                current_line = 0

                for i, chunk in enumerate(chunks):
                    # Sanitize the chunk
                    sanitized_chunk = self._sanitize_for_lucene(chunk)

                    # Validate chunk size
                    chunk_bytes = len(sanitized_chunk.encode('utf-8'))
                    if chunk_bytes > 32000:
                        logger.error(f"Skipping oversized chunk {i} from {code_file.path}: {chunk_bytes} bytes")
                        continue

                    chunk_lines = sanitized_chunk.split("\n")
                    start_line = current_line + 1
                    end_line = current_line + len(chunk_lines)

                    # Generate embedding if enabled
                    embedding = None
                    if self.embeddings:
                        embedding = await self.embeddings.embed_text(sanitized_chunk)

                    chunks_data.append({
                        "project_name": self.project_name,
                        "file_path": str(code_file.path),
                        "content": sanitized_chunk,
                        "start_line": start_line,
                        "end_line": end_line,
                        "embedding": embedding,
                        "chunk_index": i,
                    })

                    current_line = start_line + (self.chunk_size - self.chunk_overlap) - 1

            # Batch upsert files
            if self.enable_file_classification:
                file_upsert_query = """
                UNWIND $files as file
                MERGE (f:CodeFile {project_name: file.project_name, path: file.path})
                SET f.language = file.language,
                    f.size = file.size,
                    f.lines = file.lines,
                    f.last_modified = file.last_modified,
                    f.hash = file.hash,
                    f.is_test = file.is_test,
                    f.is_config = file.is_config,
                    f.is_resource = file.is_resource,
                    f.is_documentation = file.is_documentation,
                    f.file_category = file.file_category,
                    f.namespace = file.namespace
                """
            else:
                file_upsert_query = """
                UNWIND $files as file
                MERGE (f:CodeFile {project_name: file.project_name, path: file.path})
                SET f.language = file.language,
                    f.size = file.size,
                    f.lines = file.lines,
                    f.last_modified = file.last_modified,
                    f.hash = file.hash
                """

            await self.neo4j_driver.execute_query(
                file_upsert_query,
                {"files": file_data},
                routing_control=RoutingControl.WRITE,
            )

            # Delete existing chunks for these files
            paths = [f["path"] for f in file_data]
            delete_chunks_query = """
            UNWIND $paths as path
            MATCH (f:CodeFile {project_name: $project_name, path: path})-[:HAS_CHUNK]->(c:CodeChunk)
            DETACH DELETE c
            """

            await self.neo4j_driver.execute_query(
                delete_chunks_query,
                {"project_name": self.project_name, "paths": paths},
                routing_control=RoutingControl.WRITE,
            )

            # Batch create chunks (process in smaller sub-batches to avoid memory issues)
            chunk_batch_size = 1000
            for chunk_start in range(0, len(chunks_data), chunk_batch_size):
                chunk_end = min(chunk_start + chunk_batch_size, len(chunks_data))
                chunk_batch = chunks_data[chunk_start:chunk_end]

                chunk_create_query = """
                UNWIND $chunks as chunk
                MATCH (f:CodeFile {project_name: chunk.project_name, path: chunk.file_path})
                CREATE (c:CodeChunk {
                    project_name: chunk.project_name,
                    file_path: chunk.file_path,
                    content: chunk.content,
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    embedding: chunk.embedding,
                    chunk_index: chunk.chunk_index
                })
                CREATE (f)-[:HAS_CHUNK]->(c)
                """

                await self.neo4j_driver.execute_query(
                    chunk_create_query,
                    {"chunks": chunk_batch},
                    routing_control=RoutingControl.WRITE,
                )

            logger.info(f"Batch indexed files {batch_start+1}-{batch_end}/{len(code_files)}")

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
        line_count = len(code_file.content.splitlines()) if code_file.content else 0

        if self.enable_file_classification:
            file_query = """
            MERGE (f:CodeFile {project_name: $project_name, path: $path})
            SET f.language = $language,
                f.size = $size,
                f.lines = $lines,
                f.last_modified = $last_modified,
                f.hash = $hash,
                f.is_test = $is_test,
                f.is_config = $is_config,
                f.is_resource = $is_resource,
                f.is_documentation = $is_documentation,
                f.file_category = $file_category,
                f.namespace = $namespace
            RETURN f
            """
        else:
            file_query = """
            MERGE (f:CodeFile {project_name: $project_name, path: $path})
            SET f.language = $language,
                f.size = $size,
                f.lines = $lines,
                f.last_modified = $last_modified,
                f.hash = $hash
            RETURN f
            """

        query_params = {
            "project_name": self.project_name,
            "path": str(code_file.path),
            "language": code_file.language,
            "size": code_file.size,
            "lines": line_count,
            "last_modified": code_file.last_modified.isoformat(),
            "hash": code_file.file_hash,
        }

        # Add classification fields if enabled
        if self.enable_file_classification:
            query_params.update({
                "is_test": code_file.is_test,
                "is_config": code_file.is_config,
                "is_resource": code_file.is_resource,
                "is_documentation": code_file.is_documentation,
                "file_category": code_file.file_category,
                "namespace": code_file.namespace,
            })

        await self.neo4j_driver.execute_query(
            file_query,
            query_params,
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
        indexed_chunks = 0
        skipped_chunks = 0

        for i, chunk in enumerate(chunks):
            # Sanitize the chunk to ensure no single term exceeds Lucene's limit
            sanitized_chunk = self._sanitize_for_lucene(chunk)

            # Validate chunk size (should never exceed limit with new implementation)
            chunk_bytes = len(sanitized_chunk.encode('utf-8'))
            if chunk_bytes > 32000:
                # This should not happen with the new implementation
                logger.error(f"CRITICAL: Chunk {i} from {code_file.path} exceeds 32KB limit ({chunk_bytes} bytes) - This indicates a bug in chunking!")
                logger.error(f"Chunk preview: {sanitized_chunk[:100]}...")
                skipped_chunks += 1
                continue

            # Log large chunks for monitoring
            if chunk_bytes > 25000:
                logger.warning(f"Large chunk {i} from {code_file.path}: {chunk_bytes} bytes (approaching limit)")

            chunk_lines = sanitized_chunk.split("\n")
            start_line = current_line + 1
            end_line = current_line + len(chunk_lines)

            # Generate embedding for chunk if embeddings are enabled
            if self.embeddings:
                embedding = await self.embeddings.embed_text(sanitized_chunk)
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
                    "content": sanitized_chunk,
                    "start_line": start_line,
                    "end_line": end_line,
                    "embedding": embedding,
                    "chunk_index": i,
                },
                routing_control=RoutingControl.WRITE,
            )

            indexed_chunks += 1
            current_line = start_line + (self.chunk_size - self.chunk_overlap) - 1

        if skipped_chunks > 0:
            logger.error(f"Indexed file {code_file.path}: {indexed_chunks} chunks indexed, {skipped_chunks} chunks SKIPPED due to size!")
        else:
            logger.info(f"Indexed file {code_file.path} with {indexed_chunks} chunks successfully")

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
        file_category: str | None = None,
        is_test: bool | None = None,
        is_config: bool | None = None,
        is_documentation: bool | None = None,
        is_resource: bool | None = None,
    ) -> list[SearchResult]:
        """
        Perform semantic search using embeddings with project context.

        Args:
            query: Search query
            limit: Maximum number of results
            language: Filter by programming language
            file_category: Filter by file category (test/config/resource/documentation/source)
            is_test: Filter to show only test files (or non-test if False)
            is_config: Filter to show only config files (or non-config if False)
            is_documentation: Filter to show only documentation files (or non-documentation if False)
            is_resource: Filter to show only resource files (or non-resource if False)

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

            # Add file classification filters if enabled
            if self.enable_file_classification:
                if file_category:
                    cypher += " AND f.file_category = $file_category"
                if is_test is not None:
                    cypher += " AND f.is_test = $is_test"
                if is_config is not None:
                    cypher += " AND f.is_config = $is_config"
                if is_documentation is not None:
                    cypher += " AND f.is_documentation = $is_documentation"
                if is_resource is not None:
                    cypher += " AND f.is_resource = $is_resource"

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

            # Add classification parameters if enabled
            if self.enable_file_classification:
                if file_category:
                    params["file_category"] = file_category
                if is_test is not None:
                    params["is_test"] = is_test
                if is_config is not None:
                    params["is_config"] = is_config
                if is_documentation is not None:
                    params["is_documentation"] = is_documentation
                if is_resource is not None:
                    params["is_resource"] = is_resource

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

            # Add file classification filters if enabled
            if self.enable_file_classification:
                if file_category:
                    cypher += " AND f.file_category = $file_category"
                if is_test is not None:
                    cypher += " AND f.is_test = $is_test"
                if is_config is not None:
                    cypher += " AND f.is_config = $is_config"
                if is_documentation is not None:
                    cypher += " AND f.is_documentation = $is_documentation"
                if is_resource is not None:
                    cypher += " AND f.is_resource = $is_resource"

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

            # Add classification parameters if enabled
            if self.enable_file_classification:
                if file_category:
                    params["file_category"] = file_category
                if is_test is not None:
                    params["is_test"] = is_test
                if is_config is not None:
                    params["is_config"] = is_config
                if is_documentation is not None:
                    params["is_documentation"] = is_documentation
                if is_resource is not None:
                    params["is_resource"] = is_resource

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
        language: str | None = None,
        file_category: str | None = None,
        is_test: bool | None = None,
        is_config: bool | None = None,
        is_documentation: bool | None = None,
        is_resource: bool | None = None,
    ) -> list[SearchResult]:
        """
        Search for code by pattern (literal phrase via Lucene) or by regex (Cypher =~).

        Args:
            pattern: Search pattern
            is_regex: Whether pattern is a regex
            limit: Maximum number of results
            language: Filter by programming language
            file_category: Filter by file category (test/config/resource/documentation/source)
            is_test: Filter to show only test files (or non-test if False)
            is_config: Filter to show only config files (or non-config if False)
            is_documentation: Filter to show only documentation files (or non-documentation if False)
            is_resource: Filter to show only resource files (or non-resource if False)

        Returns:
            List of search results
        """
        if is_regex:
            # Regex path (no Lucene or fulltext index used)
            query = """
            MATCH (f:CodeFile)-[:HAS_CHUNK]->(c:CodeChunk)
            WHERE c.project_name = $project_name AND c.content =~ $pattern
            """
            if language:
                query += " AND f.language = $language"

            # Add file classification filters if enabled
            if self.enable_file_classification:
                if file_category:
                    query += " AND f.file_category = $file_category"
                if is_test is not None:
                    query += " AND f.is_test = $is_test"
                if is_config is not None:
                    query += " AND f.is_config = $is_config"
                if is_documentation is not None:
                    query += " AND f.is_documentation = $is_documentation"
                if is_resource is not None:
                    query += " AND f.is_resource = $is_resource"

            query += """
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
            """
            if language:
                query += " AND f.language = $language"

            # Add file classification filters if enabled
            if self.enable_file_classification:
                if file_category:
                    query += " AND f.file_category = $file_category"
                if is_test is not None:
                    query += " AND f.is_test = $is_test"
                if is_config is not None:
                    query += " AND f.is_config = $is_config"
                if is_documentation is not None:
                    query += " AND f.is_documentation = $is_documentation"
                if is_resource is not None:
                    query += " AND f.is_resource = $is_resource"

            query += """
            RETURN f.path as file_path,
                   c.content as content,
                   c.start_line as line_number,
                   score as similarity,
                   c.project_name as project_name
            ORDER BY score DESC
            LIMIT $limit
            """
            query_pattern = phrase

        params = {
            "project_name": self.project_name,
            "pattern": query_pattern,
            "limit": limit,
        }
        if language:
            params["language"] = language

        # Add classification parameters if enabled
        if self.enable_file_classification:
            if file_category:
                params["file_category"] = file_category
            if is_test is not None:
                params["is_test"] = is_test
            if is_config is not None:
                params["is_config"] = is_config
            if is_documentation is not None:
                params["is_documentation"] = is_documentation
            if is_resource is not None:
                params["is_resource"] = is_resource

        result = await self.neo4j_driver.execute_query(
            query,
            params,
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

    async def list_files_by_category(
        self,
        category: str | None = None,
        is_test: bool | None = None,
        is_config: bool | None = None,
        is_documentation: bool | None = None,
        is_resource: bool | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List files in the project filtered by category or classification flags.
        
        Args:
            category: Filter by file category (test/config/resource/documentation/source)
            is_test: Filter to show only test files (or non-test if False)
            is_config: Filter to show only config files (or non-config if False)
            is_documentation: Filter to show only documentation files (or non-documentation if False)
            is_resource: Filter to show only resource files (or non-resource if False)
            limit: Maximum number of files to return
            
        Returns:
            List of file metadata dictionaries
        """
        if not self.enable_file_classification:
            logger.warning("File classification is disabled. Enable it to use category filters.")
            return []

        query = """
        MATCH (f:CodeFile)
        WHERE f.project_name = $project_name
        """

        if category:
            query += " AND f.file_category = $category"
        if is_test is not None:
            query += " AND f.is_test = $is_test"
        if is_config is not None:
            query += " AND f.is_config = $is_config"
        if is_documentation is not None:
            query += " AND f.is_documentation = $is_documentation"
        if is_resource is not None:
            query += " AND f.is_resource = $is_resource"

        query += """
        RETURN f.path as path,
               f.language as language,
               f.size as size,
               f.lines as lines,
               f.file_category as category,
               f.is_test as is_test,
               f.is_config as is_config,
               f.is_documentation as is_documentation,
               f.is_resource as is_resource,
               f.namespace as namespace
        LIMIT $limit
        """

        params = {
            "project_name": self.project_name,
            "limit": limit,
        }

        if category:
            params["category"] = category
        if is_test is not None:
            params["is_test"] = is_test
        if is_config is not None:
            params["is_config"] = is_config
        if is_documentation is not None:
            params["is_documentation"] = is_documentation
        if is_resource is not None:
            params["is_resource"] = is_resource

        result = await self.neo4j_driver.execute_query(
            query,
            params,
            routing_control=RoutingControl.READ,
        )

        files = []
        for record in result.records:
            files.append({
                "path": record["path"],
                "language": record["language"],
                "size": record["size"],
                "lines": record["lines"],
                "category": record["category"],
                "is_test": record["is_test"],
                "is_config": record["is_config"],
                "is_documentation": record["is_documentation"],
                "is_resource": record["is_resource"],
                "namespace": record["namespace"],
            })

        return files

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
               f.lines as lines,
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
                "lines": record.get("lines", 0),
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

    async def is_repository_indexed(self, project_name: str) -> bool:
        """
        Check if a repository is already indexed in Neo4j.
        
        Args:
            project_name: Name of the project to check
            
        Returns:
            True if the repository has indexed files, False otherwise
        """
        query = """
        MATCH (f:CodeFile {project_name: $project_name})
        RETURN count(f) as file_count
        """

        result = await self.neo4j_driver.execute_query(
            query,
            {"project_name": project_name},
            routing_control=RoutingControl.READ
        )

        if result.records and len(result.records) > 0:
            file_count = result.records[0].get("file_count", 0)
            return file_count > 0
        return False

    async def get_indexed_files(self, project_name: str) -> dict[Path, datetime]:
        """
        Get a list of indexed files with their timestamps.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary mapping file paths to their last modified timestamps
        """
        query = """
        MATCH (f:CodeFile {project_name: $project_name})
        RETURN f.path as path, f.last_modified as last_modified
        """

        result = await self.neo4j_driver.execute_query(
            query,
            {"project_name": project_name},
            routing_control=RoutingControl.READ
        )

        file_map = {}
        for record in result.records:
            path = Path(record["path"])
            last_modified_str = record.get("last_modified")

            # Handle missing or invalid timestamps
            if last_modified_str:
                try:
                    # Parse ISO format timestamp
                    last_modified = datetime.fromisoformat(last_modified_str)
                except (ValueError, TypeError):
                    # Use a very old timestamp for files with invalid timestamps
                    last_modified = datetime.min
            else:
                # Use a very old timestamp for files without timestamps
                last_modified = datetime.min

            file_map[path] = last_modified

        return file_map

    async def detect_changed_files(
        self,
        current_files: list[Any],  # List[FileInfo]
        indexed_files: dict[Path, datetime]
    ) -> tuple[list[Any], list[Any], list[Path]]:
        """
        Compare current files with indexed files to detect changes.
        
        Args:
            current_files: List of FileInfo objects representing current repository state
            indexed_files: Dictionary of indexed file paths to timestamps
            
        Returns:
            Tuple of (new_files, modified_files, deleted_paths)
        """
        new_files = []
        modified_files = []
        deleted_paths = []

        # Track which indexed files we've seen
        seen_indexed_files = set()

        # Check current files against indexed files
        for file_info in current_files:
            file_path = file_info.path

            if file_path not in indexed_files:
                # This is a new file
                new_files.append(file_info)
            else:
                # File exists in index, check if modified
                seen_indexed_files.add(file_path)
                indexed_timestamp = indexed_files[file_path]

                # Compare timestamps (files are modified if current timestamp is newer)
                if file_info.last_modified > indexed_timestamp:
                    modified_files.append(file_info)
                # If timestamps are equal or older, file is unchanged (ignore it)

        # Find deleted files (in index but not in current files)
        for indexed_path in indexed_files:
            if indexed_path not in seen_indexed_files:
                deleted_paths.append(indexed_path)

        return new_files, modified_files, deleted_paths

    async def remove_files(self, project_name: str, file_paths: list[Path]):
        """
        Remove files and their chunks from the Neo4j index.
        
        Args:
            project_name: Name of the project
            file_paths: List of file paths to remove
        """
        if not file_paths:
            # Nothing to remove
            return

        # Convert paths to strings for the query
        path_strings = [str(path) for path in file_paths]

        # Delete files and their chunks in a single query
        query = """
        UNWIND $paths as path
        MATCH (f:CodeFile {project_name: $project_name, path: path})
        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CodeChunk)
        DETACH DELETE f, c
        """

        await self.neo4j_driver.execute_query(
            query,
            {
                "project_name": project_name,
                "paths": path_strings
            },
            routing_control=RoutingControl.WRITE
        )

        logger.info(f"Removed {len(file_paths)} files from project {project_name}")

    async def close(self):
        """
        Close the Neo4j RAG system and clean up resources.

        Properly closes the Neo4j driver connection to prevent resource leaks.
        This method should be called when the RAG system is no longer needed.
        """
        if self.neo4j_driver:
            await self.neo4j_driver.close()
            logger.info(f"Closed Neo4j RAG system for project {self.project_name}")
