"""Project Watch - Repository monitoring MCP server with Neo4j RAG."""

import datetime

# Version and build tracking
__version__ = "0.1.0"
__build_timestamp__ = datetime.datetime.now().isoformat()
__lucene_fix_version__ = "v3.0-phrase-search"

from .cli import cli, main
from .neo4j_rag import Neo4jRAG
from .repository_monitor import RepositoryMonitor
from .server import create_mcp_server

# Expose main components
__all__ = [
    "main", "cli", "create_mcp_server", "RepositoryMonitor", "Neo4jRAG",
    "__version__", "__build_timestamp__", "__lucene_fix_version__"
]
