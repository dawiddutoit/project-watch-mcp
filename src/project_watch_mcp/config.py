"""Configuration for project-watch-mcp."""

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

    provider: Literal["openai", "voyage", "disabled"] = "disabled"
    # Model name (used by both OpenAI and Voyage)
    model: str | None = None
    # API keys
    api_key: str | None = None
    # Embedding dimensions (auto-detected based on model if not specified)
    dimension: int | None = None

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create configuration from environment variables."""
        provider = os.getenv("EMBEDDING_PROVIDER", "voyage").lower()

        # Validate provider
        if provider not in ["openai", "voyage"]:
            provider = "disabled"

        # Determine model based on provider
        model = None
        api_key = None
        dimension = None

        if provider == "openai":
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            api_key = os.getenv("OPENAI_API_KEY")
            dimension = int(os.getenv("OPENAI_EMBEDDING_DIMENSION", "1536"))
        elif provider == "voyage":
            model = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-code-3")
            api_key = os.getenv("VOYAGE_API_KEY")
            dimension = int(os.getenv("VOYAGE_EMBEDDING_DIMENSION", "1024"))
        else:
            # Disabled provider
            dimension = None

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            dimension=dimension,
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
