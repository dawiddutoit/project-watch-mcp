#!/usr/bin/env python3
"""
Script to fix the corrupted Lucene fulltext index in Neo4j.

This script will:
1. Drop the corrupted 'code_search' fulltext index
2. Recreate it with proper configuration
3. Re-index all existing CodeChunk nodes
"""

import asyncio
import os
import sys
from neo4j import AsyncGraphDatabase
from pathlib import Path


async def fix_lucene_index():
    """Fix the corrupted Lucene index by dropping and recreating it."""
    
    # Get Neo4j connection details from environment variables
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")
    
    # Create Neo4j driver
    driver = AsyncGraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password)
    )
    
    try:
        print("🔧 Starting Lucene index fix...")
        print(f"📍 Connected to Neo4j at: {neo4j_uri}")
        
        # Step 1: Drop the corrupted index
        print("\n1️⃣ Dropping corrupted 'code_search' index...")
        try:
            await driver.execute_query(
                "DROP INDEX code_search IF EXISTS"
            )
            print("   ✅ Dropped corrupted index")
        except Exception as e:
            print(f"   ⚠️  Could not drop index (may not exist): {e}")
        
        # Step 2: Wait a moment for Neo4j to clean up
        print("\n2️⃣ Waiting for Neo4j to clean up...")
        await asyncio.sleep(2)
        
        # Step 3: Recreate the fulltext index with proper configuration
        print("\n3️⃣ Creating new 'code_search' fulltext index...")
        create_index_query = """
        CREATE FULLTEXT INDEX code_search IF NOT EXISTS
        FOR (c:CodeChunk) ON EACH [c.content]
        OPTIONS {
            indexConfig: {
                `fulltext.analyzer`: 'standard-no-stop-words',
                `fulltext.eventually_consistent`: true
            }
        }
        """
        
        try:
            await driver.execute_query(create_index_query)
            print("   ✅ Created new fulltext index")
        except Exception as e:
            print(f"   ❌ Failed to create index: {e}")
            return False
        
        # Step 4: Check how many chunks need re-indexing
        print("\n4️⃣ Checking existing CodeChunk nodes...")
        count_result = await driver.execute_query(
            "MATCH (c:CodeChunk) RETURN count(c) as chunk_count"
        )
        
        chunk_count = 0
        if count_result.records:
            chunk_count = count_result.records[0].get("chunk_count", 0)
        
        print(f"   📊 Found {chunk_count} chunks to re-index")
        
        # Step 5: Fix oversized chunks that will break the index
        print("\n5️⃣ Checking and fixing oversized chunks...")
        oversized_query = """
        MATCH (c:CodeChunk)
        WHERE size(c.content) > 30000
        RETURN c.file_path as file_path, 
               c.chunk_index as chunk_index,
               size(c.content) as content_size,
               id(c) as node_id
        ORDER BY content_size DESC
        """
        
        oversized_result = await driver.execute_query(oversized_query)
        
        if oversized_result.records:
            print(f"   ⚠️  Found {len(oversized_result.records)} oversized chunks")
            
            # Show the worst offenders
            for record in oversized_result.records[:5]:
                print(f"      - {record['file_path']} (chunk {record['chunk_index']}): {record['content_size']} chars")
            
            # Ask user what to do
            print("\n   🔴 CRITICAL: These chunks will break the Lucene index!")
            print("   Options:")
            print("   1. Delete oversized chunks (recommended)")
            print("   2. Truncate to 30KB limit")
            print("   3. Skip and try anyway (will likely fail)")
            
            choice = input("\n   Enter choice (1/2/3) [1]: ").strip() or "1"
            
            if choice == "1":
                print("   🗑️  Deleting oversized chunks...")
                delete_result = await driver.execute_query("""
                    MATCH (c:CodeChunk)
                    WHERE size(c.content) > 30000
                    WITH c, c.file_path as file_path
                    DETACH DELETE c
                    RETURN count(c) as deleted_count, collect(DISTINCT file_path) as affected_files
                """)
                if delete_result.records:
                    record = delete_result.records[0]
                    print(f"   ✅ Deleted {record['deleted_count']} oversized chunks")
                    print(f"   📝 Affected files: {', '.join(record['affected_files'][:5])}...")
                    print("   💡 Run 'mcp__project-watch-mcp__initialize_repository' to re-index these files")
            
            elif choice == "2":
                print("   ✂️  Truncating oversized chunks to 30KB...")
                truncate_result = await driver.execute_query("""
                    MATCH (c:CodeChunk)
                    WHERE size(c.content) > 30000
                    SET c.content = substring(c.content, 0, 30000) + '\n... [TRUNCATED - content too large for index]'
                    RETURN count(c) as truncated_count
                """)
                if truncate_result.records:
                    print(f"   ✅ Truncated {truncate_result.records[0]['truncated_count']} chunks")
            
            else:
                print("   ⚠️  Skipping fix - index creation may fail!")
        else:
            print("   ✅ No oversized chunks detected")
        
        # Step 6: Force Neo4j to rebuild the index
        print("\n6️⃣ Triggering index population...")
        print("   ℹ️  Neo4j will automatically populate the index with existing data")
        print("   ⏳ This may take a few moments for large repositories")
        
        # Wait for index to be populated
        await asyncio.sleep(3)
        
        # Step 7: Check index status
        print("\n7️⃣ Checking index status...")
        status_query = """
        SHOW INDEXES
        WHERE name = 'code_search'
        """
        
        try:
            status_result = await driver.execute_query(status_query)
            if status_result.records:
                record = status_result.records[0]
                state = record.get("state", "UNKNOWN")
                print(f"   📊 Index state: {state}")
                
                if state == "ONLINE":
                    print("   ✅ Index is ONLINE and ready to use!")
                elif state == "POPULATING":
                    print("   ⏳ Index is still POPULATING - wait a moment and try searching")
                elif state == "FAILED":
                    print("   ❌ Index is in FAILED state - may need to remove oversized chunks")
                else:
                    print(f"   ⚠️  Index is in {state} state")
        except Exception as e:
            print(f"   ⚠️  Could not check index status: {e}")
        
        print("\n✨ Index recreation complete!")
        print("\n📝 Next steps:")
        print("1. If you have oversized chunks, run: uv run project-watch-mcp init")
        print("2. This will re-chunk files with proper size limits")
        print("3. Then test pattern search again")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error fixing index: {e}")
        return False
        
    finally:
        await driver.close()
        print("\n🔒 Closed Neo4j connection")


async def main():
    """Main entry point."""
    success = await fix_lucene_index()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())