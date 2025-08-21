"""
Simple test to verify Docker Neo4j setup works.
Run this to test if Neo4j container can be started properly.
"""

import asyncio
import pytest
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer


@pytest.mark.asyncio
@pytest.mark.integration
async def test_neo4j_docker_container():
    """Test that we can start a Neo4j container and connect to it."""
    
    # Use community edition which is smaller and faster to download
    with Neo4jContainer(image="neo4j:5") as container:
        # Get connection details
        connection_url = container.get_connection_url()
        print(f"Neo4j container started at: {connection_url}")
        
        # Create async driver
        driver = AsyncGraphDatabase.driver(
            connection_url,
            auth=("neo4j", "test")  # Default password for test container
        )
        
        try:
            # Verify connectivity
            await driver.verify_connectivity()
            print("✓ Connected to Neo4j")
            
            # Run a simple query
            async with driver.session() as session:
                result = await session.run("RETURN 'Hello from Neo4j!' as message")
                record = await result.single()
                message = record["message"]
                print(f"✓ Query result: {message}")
                
                assert message == "Hello from Neo4j!"
                
            print("✓ Neo4j Docker container test passed!")
            
        finally:
            await driver.close()


if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_neo4j_docker_container())