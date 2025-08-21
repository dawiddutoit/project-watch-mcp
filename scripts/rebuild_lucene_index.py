#!/usr/bin/env python
"""
Script to drop and rebuild the Lucene fulltext index with proper chunking.

This script:
1. Drops the existing (potentially corrupted) fulltext index
2. Recreates it with proper configuration
3. Re-indexes all existing chunks with sanitization
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neo4j import AsyncGraphDatabase, RoutingControl
from project_watch_mcp.config import Neo4jConfig

# Force logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def drop_and_rebuild_index(driver):
    """Drop the existing fulltext index and recreate it."""
    
    logger.info("Step 1: Dropping existing fulltext index if it exists...")
    
    try:
        # Drop the existing index
        drop_query = "DROP INDEX code_search IF EXISTS"
        await driver.execute_query(drop_query, routing_control=RoutingControl.WRITE)
        logger.info("Dropped existing code_search index")
    except Exception as e:
        logger.warning(f"Could not drop index (might not exist): {e}")
    
    logger.info("Step 2: Creating new fulltext index...")
    
    try:
        # Create new fulltext index with proper configuration
        create_query = """
        CREATE FULLTEXT INDEX code_search IF NOT EXISTS
        FOR (c:CodeChunk) ON EACH [c.content]
        OPTIONS {
            indexConfig: {
                `fulltext.analyzer`: 'standard-no-stop-words',
                `fulltext.eventually_consistent`: true
            }
        }
        """
        await driver.execute_query(create_query, routing_control=RoutingControl.WRITE)
        logger.info("Created new code_search fulltext index")
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        raise
    
    logger.info("Step 3: Verifying index creation...")
    
    # Verify the index was created
    verify_query = "SHOW INDEXES YIELD name, state WHERE name = 'code_search'"
    result = await driver.execute_query(verify_query, routing_control=RoutingControl.READ)
    
    if result.records:
        state = result.records[0].get("state")
        logger.info(f"Index 'code_search' is in state: {state}")
        if state == "ONLINE":
            logger.info("✅ Index successfully created and online!")
        else:
            logger.warning(f"⚠️ Index created but in state: {state}")
    else:
        logger.error("❌ Index not found after creation!")


async def sanitize_existing_chunks(driver):
    """Update existing chunks to ensure no terms exceed Lucene limits."""
    
    logger.info("Step 4: Checking for oversized terms in existing chunks...")
    
    # Get all chunks
    count_query = "MATCH (c:CodeChunk) RETURN count(c) as total"
    result = await driver.execute_query(count_query, routing_control=RoutingControl.READ)
    total = result.records[0].get("total") if result.records else 0
    
    logger.info(f"Found {total} chunks to check")
    
    if total == 0:
        logger.info("No chunks found, skipping sanitization")
        return
    
    # Process chunks in batches
    batch_size = 100
    processed = 0
    updated = 0
    
    for offset in range(0, total, batch_size):
        # Get batch of chunks
        batch_query = """
        MATCH (c:CodeChunk)
        RETURN id(c) as id, c.content as content
        ORDER BY id(c)
        SKIP $offset LIMIT $limit
        """
        
        result = await driver.execute_query(
            batch_query,
            {"offset": offset, "limit": batch_size},
            routing_control=RoutingControl.READ
        )
        
        for record in result.records:
            chunk_id = record.get("id")
            content = record.get("content")
            
            if content:
                # Check for oversized terms
                terms = content.split()
                has_oversized = False
                
                for term in terms:
                    if len(term.encode('utf-8')) > 32000:
                        has_oversized = True
                        logger.warning(f"Found oversized term in chunk {chunk_id}: {len(term)} chars")
                        break
                
                if has_oversized:
                    # This chunk needs sanitization
                    from project_watch_mcp.neo4j_rag import Neo4jRAG
                    
                    # Create temporary RAG instance for sanitization
                    rag = Neo4jRAG(neo4j_driver=None, project_name="temp", embeddings=None)
                    sanitized = rag._sanitize_for_lucene(content)
                    
                    # Update the chunk
                    update_query = """
                    MATCH (c:CodeChunk)
                    WHERE id(c) = $id
                    SET c.content = $content
                    """
                    
                    await driver.execute_query(
                        update_query,
                        {"id": chunk_id, "content": sanitized},
                        routing_control=RoutingControl.WRITE
                    )
                    
                    updated += 1
                    logger.info(f"Updated chunk {chunk_id} with sanitized content")
            
            processed += 1
        
        logger.info(f"Processed {processed}/{total} chunks, updated {updated} chunks")
    
    logger.info(f"✅ Sanitization complete: {updated} chunks were updated")


async def main():
    """Main function to rebuild the Lucene index."""
    
    print("Starting Lucene index rebuild process...", flush=True)
    logger.info("Starting Lucene index rebuild process...")
    
    try:
        # Load configuration - override with bolt protocol
        os.environ["NEO4J_URI"] = "bolt://localhost:7687"
        config = Neo4jConfig.from_env()
        print(f"Config loaded: URI={config.uri}", flush=True)
        
        # Create Neo4j driver
        driver = AsyncGraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password)
        )
        print("Driver created", flush=True)
        
        try:
            # Test connection
            print("Testing connectivity...", flush=True)
            await driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {config.uri}")
            print(f"Connected to Neo4j at {config.uri}", flush=True)
            
            # Drop and rebuild index
            await drop_and_rebuild_index(driver)
            
            # Sanitize existing chunks
            await sanitize_existing_chunks(driver)
            
            logger.info("✅ Index rebuild complete!")
            print("✅ Index rebuild complete!", flush=True)
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            print(f"Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        finally:
            await driver.close()
    except Exception as e:
        print(f"Main error: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())