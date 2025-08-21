import asyncio
from neo4j import AsyncGraphDatabase
import os

async def check_indexes():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
    
    try:
        async with driver.session() as session:
            # Get all indexes with details
            result = await session.run("SHOW INDEXES")
            indexes = await result.data()
            
            print("Mystery Indexes Details:")
            print("-" * 80)
            
            for idx in indexes:
                if idx['name'] in ['index_343aff4e', 'index_f7700477']:
                    print(f"\nIndex: {idx['name']}")
                    print(f"  Type: {idx.get('type', 'N/A')}")
                    print(f"  Entity Type: {idx.get('entityType', 'N/A')}")
                    print(f"  Labels/Types: {idx.get('labelsOrTypes', 'N/A')}")
                    print(f"  Properties: {idx.get('properties', 'N/A')}")
                    print(f"  State: {idx.get('state', 'N/A')}")
                    print(f"  Index Provider: {idx.get('indexProvider', 'N/A')}")
                    
    finally:
        await driver.close()

asyncio.run(check_indexes())
