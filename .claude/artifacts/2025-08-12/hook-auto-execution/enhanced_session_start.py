#!/usr/bin/env python3
"""
Enhanced SessionStart hook for Project Watch MCP with direct initialization.

This hook directly initializes the repository monitoring without requiring
Claude to execute any commands. It imports the necessary modules and runs
the initialization directly, with proper error handling and fallback.
"""

import asyncio
import json
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Setup project paths
hook_path = Path(__file__).resolve()
project_root = hook_path.parent.parent.parent
log_dir = project_root / ".claude" / "hooks" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
log_file = log_dir / f"session_start_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to stderr for debugging
    ]
)
logger = logging.getLogger(__name__)


async def direct_initialization() -> Dict[str, Any]:
    """
    Directly initialize the repository using imported server components.
    
    Returns:
        Dict with initialization status and details
    """
    try:
        # Add project source to Python path
        sys.path.insert(0, str(project_root / "src"))
        
        # Import required components
        logger.info("Importing server components...")
        from neo4j import AsyncGraphDatabase
        from project_watch_mcp.repository_monitor import RepositoryMonitor
        from project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile
        from project_watch_mcp.config import ProjectConfig, EmbeddingConfig
        from project_watch_mcp.utils.embedding import create_embeddings_provider
        
        # Get Neo4j configuration from environment
        neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DB", os.getenv("NEO4J_DATABASE", "memory"))
        
        if not neo4j_password:
            logger.warning("NEO4J_PASSWORD not set in environment")
            return {
                "status": "skipped",
                "reason": "Neo4j password not configured in environment",
                "message": "Please set NEO4J_PASSWORD environment variable"
            }
        
        logger.info(f"Connecting to Neo4j at {neo4j_uri}")
        
        # Create Neo4j driver
        neo4j_driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
            database=neo4j_database
        )
        
        # Verify Neo4j connectivity
        try:
            await asyncio.wait_for(
                neo4j_driver.verify_connectivity(),
                timeout=5.0
            )
            logger.info("Neo4j connection verified")
        except asyncio.TimeoutError:
            logger.error("Neo4j connection timeout")
            await neo4j_driver.close()
            return {
                "status": "error",
                "reason": "Neo4j connection timeout",
                "message": "Could not connect to Neo4j within 5 seconds"
            }
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            await neo4j_driver.close()
            return {
                "status": "error",
                "reason": "Neo4j connection failed",
                "message": str(e)
            }
        
        # Create project configuration
        project_config = ProjectConfig.from_repository_path(project_root)
        project_name = project_config.name
        logger.info(f"Project name: {project_name}")
        
        # Create repository monitor
        file_patterns = [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", 
            "*.cpp", "*.c", "*.h", "*.hpp", "*.cs", "*.go", 
            "*.rs", "*.rb", "*.php", "*.swift", "*.kt", "*.scala",
            "*.r", "*.m", "*.sql", "*.sh", "*.yaml", "*.yml",
            "*.toml", "*.json", "*.xml", "*.html", "*.css", 
            "*.scss", "*.md", "*.txt"
        ]
        
        repository_monitor = RepositoryMonitor(
            repo_path=project_root,
            project_name=project_name,
            neo4j_driver=neo4j_driver,
            file_patterns=file_patterns
        )
        logger.info("Repository monitor created")
        
        # Create embeddings configuration
        embedding_config = EmbeddingConfig.from_env()
        embeddings = create_embeddings_provider(
            provider_type=embedding_config.provider,
            api_key=embedding_config.openai_api_key,
            model=embedding_config.openai_model,
            api_url=embedding_config.local_api_url,
            dimension=embedding_config.dimension,
        )
        logger.info(f"Using {embedding_config.provider} embeddings provider")
        
        # Create Neo4j RAG system
        neo4j_rag = Neo4jRAG(
            neo4j_driver=neo4j_driver,
            project_name=project_name,
            embeddings=embeddings,
            chunk_size=100,
            chunk_overlap=20
        )
        
        # Initialize RAG system (create indexes)
        await neo4j_rag.initialize()
        logger.info("Neo4j RAG system initialized")
        
        # Scan repository for files
        logger.info("Scanning repository for files...")
        files = await asyncio.wait_for(
            repository_monitor.scan_repository(),
            timeout=30.0
        )
        total_files = len(files)
        logger.info(f"Found {total_files} files to index")
        
        # Index files
        indexed_count = 0
        failed_files = []
        
        for file_info in files:
            try:
                # Read file content
                content = file_info.path.read_text(encoding="utf-8", errors="ignore")
                
                # Create CodeFile object
                code_file = CodeFile(
                    project_name=project_name,
                    path=file_info.path,
                    content=content,
                    language=file_info.language,
                    size=file_info.size,
                    last_modified=file_info.last_modified,
                )
                
                # Index the file
                await neo4j_rag.index_file(code_file)
                indexed_count += 1
                
                # Log progress every 10 files
                if indexed_count % 10 == 0:
                    logger.info(f"Indexed {indexed_count}/{total_files} files...")
                    
            except Exception as e:
                logger.warning(f"Failed to index {file_info.path}: {e}")
                failed_files.append(str(file_info.path.relative_to(project_root)))
        
        # Clean up
        await neo4j_driver.close()
        logger.info("Neo4j connection closed")
        
        # Prepare result
        result = {
            "status": "success",
            "indexed": indexed_count,
            "total": total_files,
            "method": "direct",
            "project_name": project_name
        }
        
        if failed_files:
            result["failed_files"] = failed_files[:10]  # Limit to first 10 for output
            result["failed_count"] = len(failed_files)
        
        logger.info(f"Initialization complete: {indexed_count}/{total_files} files indexed")
        return result
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return {
            "status": "import_error",
            "reason": str(e),
            "message": "Could not import required modules. Please ensure project dependencies are installed."
        }
    except asyncio.TimeoutError:
        logger.error("Repository scan timeout")
        return {
            "status": "timeout",
            "reason": "Repository scan took too long",
            "message": "Repository scanning timed out after 30 seconds"
        }
    except Exception as e:
        logger.exception("Unexpected error during initialization")
        return {
            "status": "error",
            "reason": str(e),
            "message": f"Unexpected error: {e}"
        }


def create_fallback_output() -> Dict[str, Any]:
    """
    Create fallback output instructing Claude to initialize manually.
    
    Returns:
        Dict with instruction for manual initialization
    """
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": (
                "Please initialize the Project Watch MCP repository monitoring "
                "by calling: mcp__project-watch-local__initialize_repository"
            ),
            "initializationStatus": "manual_required"
        }
    }


def create_success_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create success output for Claude with initialization results.
    
    Args:
        result: Initialization result dictionary
        
    Returns:
        Dict with formatted output for Claude
    """
    indexed = result.get("indexed", 0)
    total = result.get("total", 0)
    project_name = result.get("project_name", "project-watch-mcp")
    
    context_parts = [
        f"✅ Repository auto-initialized successfully!",
        f"Project: {project_name}",
        f"Indexed: {indexed}/{total} files",
    ]
    
    if result.get("failed_count"):
        context_parts.append(f"Failed: {result['failed_count']} files (check logs for details)")
    
    context_parts.append(
        "The repository is now indexed and ready for semantic search. "
        "You can use search_code, get_file_info, and other tools immediately."
    )
    
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": " ".join(context_parts),
            "initializationStatus": "success",
            "statistics": {
                "indexed": indexed,
                "total": total,
                "failed": result.get("failed_count", 0)
            }
        }
    }


def create_error_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create error output for Claude with fallback instructions.
    
    Args:
        result: Error result dictionary
        
    Returns:
        Dict with error information and fallback instructions
    """
    status = result.get("status", "unknown")
    reason = result.get("reason", "Unknown error")
    message = result.get("message", "")
    
    context_parts = [
        f"⚠️ Auto-initialization {status}: {reason}",
    ]
    
    if message:
        context_parts.append(f"Details: {message}")
    
    context_parts.append(
        "Please manually initialize by calling: mcp__project-watch-local__initialize_repository"
    )
    
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": " ".join(context_parts),
            "initializationStatus": status,
            "error": reason
        }
    }


def main() -> int:
    """
    Main entry point for the SessionStart hook.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        logger.info("=" * 60)
        logger.info("SessionStart hook triggered")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Log file: {log_file}")
        
        # Check if we should skip initialization
        skip_marker = project_root / ".claude" / ".skip_auto_init"
        if skip_marker.exists():
            logger.info("Skip marker found, not auto-initializing")
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": "Auto-initialization skipped (marker file present)",
                    "initializationStatus": "skipped"
                }
            }
            print(json.dumps(output))
            return 0
        
        # Run the initialization
        logger.info("Starting direct initialization...")
        result = asyncio.run(direct_initialization())
        logger.info(f"Initialization result: {result}")
        
        # Create appropriate output based on result
        if result["status"] == "success":
            output = create_success_output(result)
        elif result["status"] in ["skipped", "import_error"]:
            # For these statuses, fall back to manual instruction
            output = create_fallback_output()
        else:
            # For errors, provide error details and fallback
            output = create_error_output(result)
        
        # Output the result for Claude
        print(json.dumps(output, indent=2))
        
        # Create initialization marker with timestamp
        init_marker = project_root / ".claude" / ".last_auto_init"
        init_marker.parent.mkdir(parents=True, exist_ok=True)
        init_marker.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "result": result
        }, indent=2))
        
        logger.info("SessionStart hook completed successfully")
        return 0
        
    except Exception as e:
        logger.exception("Fatal error in SessionStart hook")
        
        # Provide minimal fallback output on complete failure
        emergency_output = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": (
                    "SessionStart hook encountered an error. "
                    "Please manually initialize: mcp__project-watch-local__initialize_repository"
                ),
                "initializationStatus": "hook_error",
                "error": str(e)
            }
        }
        print(json.dumps(emergency_output))
        return 1


if __name__ == "__main__":
    sys.exit(main())