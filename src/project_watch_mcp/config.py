"""Configuration for project-watch-mcp."""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ProjectConfig:
    """Configuration for project identification and isolation."""

    name: str
    repository_path: Path | None = None

    def __post_init__(self):
        """Validate and normalize project name."""
        self.name = self.validate_project_name(self.name)

    @staticmethod
    def validate_project_name(name: str) -> str:
        """
        Validate and sanitize project name.

        Args:
            name: Project name to validate

        Returns:
            Sanitized project name

        Raises:
            ValueError: If project name is invalid
        """
        if not name:
            raise ValueError("Project name cannot be empty")

        # Remove any leading/trailing whitespace
        name = name.strip()

        # Check length
        if len(name) > 100:
            raise ValueError("Project name must be 100 characters or less")

        # Only allow alphanumeric, dash, and underscore
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(
                "Project name can only contain alphanumeric characters, " "dashes, and underscores"
            )

        return name

    @classmethod
    def from_repository_path(cls, repo_path: Path) -> "ProjectConfig":
        """
        Create project config from repository path.
        Uses path hash as project name if not specified.

        Args:
            repo_path: Path to repository

        Returns:
            ProjectConfig instance
        """
        # Generate a deterministic project name from path
        path_str = str(repo_path.resolve())
        path_hash = hashlib.sha256(path_str.encode()).hexdigest()[:8]
        project_name = repo_path.name

        # Sanitize the generated name
        project_name = re.sub(r"[^a-zA-Z0-9_-]", "_", project_name)

        return cls(name=project_name, repository_path=repo_path)

    @classmethod
    def from_env(cls, repo_path: Path | None = None) -> "ProjectConfig":
        """
        Create project config from environment variable or repository path.

        Args:
            repo_path: Optional repository path

        Returns:
            ProjectConfig instance
        """
        project_name = os.getenv("PROJECT_NAME")

        if project_name:
            return cls(name=project_name, repository_path=repo_path)
        elif repo_path:
            return cls.from_repository_path(repo_path)
        else:
            # Default project name if nothing else is available
            return cls(name="default_project", repository_path=repo_path)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider."""

    provider: Literal["openai", "local", "mock"] = "mock"
    # OpenAI configuration
    openai_api_key: str | None = None
    openai_model: str = "text-embedding-3-small"
    # Local configuration
    local_api_url: str = "http://localhost:8080/embeddings"
    # Embedding dimensions
    dimension: int = 384

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create configuration from environment variables."""
        provider = os.getenv("EMBEDDING_PROVIDER", "mock")

        # Validate provider
        if provider not in ["openai", "local", "mock"]:
            provider = "mock"

        return cls(
            provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            local_api_url=os.getenv("LOCAL_EMBEDDING_API_URL", "http://localhost:8080/embeddings"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
        )


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Create configuration from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


@dataclass
class ProjectWatchConfig:
    """Main configuration for project-watch-mcp."""

    project: ProjectConfig
    neo4j: Neo4jConfig
    embedding: EmbeddingConfig
    chunk_size: int = 500
    chunk_overlap: int = 50

    @classmethod
    def from_env(cls, repo_path: Path | None = None) -> "ProjectWatchConfig":
        """Create configuration from environment variables."""
        return cls(
            project=ProjectConfig.from_env(repo_path),
            neo4j=Neo4jConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        )
