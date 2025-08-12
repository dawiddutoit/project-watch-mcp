#!/usr/bin/env python3
"""
Test script to validate direct initialization approach.

This script tests whether we can successfully import and run
the initialization code directly without going through the MCP server.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Setup paths
script_path = Path(__file__).resolve()
# Navigate up from .claude/artifacts/2025-08-12/hook-auto-execution/test_direct_init.py
# to the actual project root
project_root = script_path.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

print(f"Project root: {project_root}")
print(f"Python path includes: {project_root / 'src'}")
print("-" * 60)


async def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        from neo4j import AsyncGraphDatabase
        print("✅ neo4j import successful")
    except ImportError as e:
        print(f"❌ neo4j import failed: {e}")
        return False
    
    try:
        from project_watch_mcp.repository_monitor import RepositoryMonitor
        print("✅ RepositoryMonitor import successful")
    except ImportError as e:
        print(f"❌ RepositoryMonitor import failed: {e}")
        return False
    
    try:
        from project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile
        print("✅ Neo4jRAG import successful")
    except ImportError as e:
        print(f"❌ Neo4jRAG import failed: {e}")
        return False
    
    try:
        from project_watch_mcp.config import ProjectConfig, EmbeddingConfig
        print("✅ Config imports successful")
    except ImportError as e:
        print(f"❌ Config imports failed: {e}")
        return False
    
    try:
        from project_watch_mcp.utils.embedding import create_embeddings_provider
        print("✅ Embedding utils import successful")
    except ImportError as e:
        print(f"❌ Embedding utils import failed: {e}")
        return False
    
    print("All imports successful!")
    return True


async def test_neo4j_connection():
    """Test Neo4j connection."""
    print("\nTesting Neo4j connection...")
    
    from neo4j import AsyncGraphDatabase
    
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DB", os.getenv("NEO4J_DATABASE", "memory"))
    
    if not neo4j_password:
        print("❌ NEO4J_PASSWORD not set in environment")
        return False
    
    print(f"URI: {neo4j_uri}")
    print(f"User: {neo4j_user}")
    print(f"Database: {neo4j_database}")
    
    try:
        driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
            database=neo4j_database
        )
        
        await driver.verify_connectivity()
        print("✅ Neo4j connection successful")
        
        await driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False


async def test_repository_scan():
    """Test repository scanning."""
    print("\nTesting repository scan...")
    
    from neo4j import AsyncGraphDatabase
    from project_watch_mcp.repository_monitor import RepositoryMonitor
    from project_watch_mcp.config import ProjectConfig
    
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DB", os.getenv("NEO4J_DATABASE", "memory"))
    
    if not neo4j_password:
        print("❌ NEO4J_PASSWORD not set - skipping")
        return False
    
    try:
        # Create Neo4j driver
        driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
            database=neo4j_database
        )
        
        # Create project config
        project_config = ProjectConfig.from_repository_path(project_root)
        print(f"Project name: {project_config.name}")
        
        # Create repository monitor
        monitor = RepositoryMonitor(
            repo_path=project_root,
            project_name=project_config.name,
            neo4j_driver=driver,
            file_patterns=["*.py", "*.md", "*.json", "*.yaml", "*.yml"]
        )
        
        # Scan repository
        files = await monitor.scan_repository()
        print(f"✅ Found {len(files)} files")
        
        # Show first 5 files
        for i, file_info in enumerate(files[:5]):
            rel_path = file_info.path.relative_to(project_root)
            print(f"  {i+1}. {rel_path} ({file_info.language})")
        
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")
        
        await driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Repository scan failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mini_initialization():
    """Test a minimal initialization flow."""
    print("\nTesting mini initialization...")
    
    from neo4j import AsyncGraphDatabase
    from project_watch_mcp.repository_monitor import RepositoryMonitor
    from project_watch_mcp.neo4j_rag import Neo4jRAG, CodeFile
    from project_watch_mcp.config import ProjectConfig, EmbeddingConfig
    from project_watch_mcp.utils.embedding import create_embeddings_provider
    
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not neo4j_password:
        print("❌ NEO4J_PASSWORD not set - skipping")
        return False
    
    try:
        # Setup
        neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_database = os.getenv("NEO4J_DB", os.getenv("NEO4J_DATABASE", "memory"))
        
        driver = AsyncGraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
            database=neo4j_database
        )
        
        project_config = ProjectConfig.from_repository_path(project_root)
        
        # Create monitor
        monitor = RepositoryMonitor(
            repo_path=project_root,
            project_name=project_config.name,
            neo4j_driver=driver,
            file_patterns=["*.md"]  # Just markdown files for quick test
        )
        
        # Create RAG
        embedding_config = EmbeddingConfig.from_env()
        embeddings = create_embeddings_provider(
            provider_type=embedding_config.provider,
            api_key=embedding_config.openai_api_key,
            model=embedding_config.openai_model,
            api_url=embedding_config.local_api_url,
            dimension=embedding_config.dimension,
        )
        
        rag = Neo4jRAG(
            neo4j_driver=driver,
            project_name=project_config.name,
            embeddings=embeddings,
            chunk_size=100,
            chunk_overlap=20
        )
        
        # Initialize RAG
        await rag.initialize()
        print("✅ RAG initialized")
        
        # Scan files
        files = await monitor.scan_repository()
        print(f"✅ Found {len(files)} markdown files")
        
        # Index first file only for test
        if files:
            file_info = files[0]
            content = file_info.path.read_text(encoding="utf-8")
            
            code_file = CodeFile(
                project_name=project_config.name,
                path=file_info.path,
                content=content,
                language=file_info.language,
                size=file_info.size,
                last_modified=file_info.last_modified,
            )
            
            await rag.index_file(code_file)
            rel_path = file_info.path.relative_to(project_root)
            print(f"✅ Indexed: {rel_path}")
        
        await driver.close()
        print("✅ Mini initialization successful!")
        return True
        
    except Exception as e:
        print(f"❌ Mini initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Direct Initialization Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    result = await test_imports()
    results.append(("Imports", result))
    
    # Test 2: Neo4j Connection
    result = await test_neo4j_connection()
    results.append(("Neo4j Connection", result))
    
    # Test 3: Repository Scan
    result = await test_repository_scan()
    results.append(("Repository Scan", result))
    
    # Test 4: Mini Initialization
    result = await test_mini_initialization()
    results.append(("Mini Initialization", result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Direct initialization is viable.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)