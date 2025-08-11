#!/usr/bin/env python3
"""
PROOF OF CONCEPT: Data Corruption in Neo4j RAG System
Demonstrates how the lack of project context causes data mixing and corruption.

WARNING: This script WILL corrupt data if run against a production Neo4j instance.
Use only in a test environment.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from neo4j import AsyncGraphDatabase

# Simulating the current implementation's behavior
async def demonstrate_corruption():
    """Demonstrate how data corruption occurs in the current implementation."""
    
    # Connect to Neo4j (adjust credentials as needed)
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687", 
        auth=("neo4j", "password")
    )
    
    try:
        # Verify connection
        await driver.verify_connectivity()
        print("✓ Connected to Neo4j")
        
        # Clear any existing test data
        await driver.execute_query("MATCH (n:TestNode) DETACH DELETE n")
        
        print("\n=== DEMONSTRATION: Project Context Data Corruption ===\n")
        
        # Scenario 1: Path Collision
        print("1. PATH COLLISION SCENARIO")
        print("-" * 40)
        
        # Simulate indexing from Project A
        result_a = await driver.execute_query("""
            MERGE (f:TestNode {path: '/src/main.py', type: 'CodeFile'})
            SET f.content = 'Project A: Authentication Module',
                f.project_context = 'MISSING',
                f.indexed_at = $timestamp,
                f.actual_project = 'Project_A'
            RETURN f.content as content
        """, timestamp=datetime.now().isoformat())
        
        print(f"Indexed from Project A: {result_a.records[0]['content']}")
        
        # Simulate indexing same path from Project B
        result_b = await driver.execute_query("""
            MERGE (f:TestNode {path: '/src/main.py', type: 'CodeFile'})
            SET f.content = 'Project B: Payment Processing',
                f.project_context = 'MISSING',
                f.indexed_at = $timestamp,
                f.actual_project = 'Project_B'
            RETURN f.content as content, f.actual_project as project
        """, timestamp=datetime.now().isoformat())
        
        print(f"Indexed from Project B: {result_b.records[0]['content']}")
        
        # Check what's actually in the database
        check_result = await driver.execute_query("""
            MATCH (f:TestNode {path: '/src/main.py'})
            RETURN f.content as content, f.actual_project as project
        """)
        
        print(f"\n⚠️  DATABASE NOW CONTAINS: {check_result.records[0]['content']}")
        print(f"⚠️  Project A's data is GONE! Overwritten by: {check_result.records[0]['project']}")
        
        # Scenario 2: Search Contamination
        print("\n2. SEARCH CONTAMINATION SCENARIO")
        print("-" * 40)
        
        # Create multiple files from different projects
        projects_data = [
            ("Project_Banking", "/api/auth.py", "banking authentication: OAuth2"),
            ("Project_Healthcare", "/api/auth.py", "healthcare auth: HIPAA compliant"),
            ("Project_Gaming", "/api/auth.py", "gaming auth: Steam integration"),
        ]
        
        for project, path, content in projects_data:
            await driver.execute_query("""
                CREATE (f:TestNode {
                    path: $path,
                    type: 'SearchTest',
                    content: $content,
                    actual_project: $project,
                    project_context: 'MISSING'
                })
            """, path=path, content=content, project=project)
            print(f"Indexed: {project} - {path}")
        
        # Now search for "auth" - should only get one project's results
        print("\nSearching for 'auth' (expecting only current project results)...")
        search_result = await driver.execute_query("""
            MATCH (f:TestNode)
            WHERE f.type = 'SearchTest' AND f.content CONTAINS 'auth'
            RETURN f.actual_project as project, f.content as content
        """)
        
        print("\n⚠️  CONTAMINATED SEARCH RESULTS:")
        for record in search_result.records:
            print(f"  - {record['project']}: {record['content']}")
        
        print("\n❌ ALL PROJECTS MIXED IN RESULTS! No way to filter by project!")
        
        # Scenario 3: Statistics Confusion
        print("\n3. STATISTICS CONFUSION SCENARIO")
        print("-" * 40)
        
        stats_result = await driver.execute_query("""
            MATCH (f:TestNode)
            RETURN 
                COUNT(f) as total_files,
                COUNT(DISTINCT f.actual_project) as projects_mixed,
                COLLECT(DISTINCT f.actual_project) as all_projects
        """)
        
        stats = stats_result.records[0]
        print(f"Total files in database: {stats['total_files']}")
        print(f"Projects mixed together: {stats['projects_mixed']}")
        print(f"Projects: {', '.join(stats['all_projects'])}")
        print("\n❌ Statistics show combined data from ALL projects!")
        
        # Demonstrate the fix
        print("\n=== PROPOSED FIX DEMONSTRATION ===\n")
        
        # Clear test data
        await driver.execute_query("MATCH (n:TestNode) DETACH DELETE n")
        
        # Correct implementation with project context
        print("Indexing with proper project context...")
        
        for project in ["Project_A", "Project_B"]:
            await driver.execute_query("""
                MERGE (p:TestProject {name: $project})
                MERGE (f:TestNode {
                    path: '/src/main.py',
                    project_name: $project,
                    type: 'FixedImplementation'
                })
                SET f.content = $content
                MERGE (p)-[:HAS_FILE]->(f)
            """, project=project, content=f"{project}: main.py content")
            print(f"✓ Indexed /src/main.py for {project}")
        
        # Verify isolation
        fixed_result = await driver.execute_query("""
            MATCH (f:TestNode {type: 'FixedImplementation'})
            RETURN f.path as path, f.project_name as project, f.content as content
            ORDER BY f.project_name
        """)
        
        print("\n✅ WITH PROJECT CONTEXT:")
        for record in fixed_result.records:
            print(f"  - {record['project']}: {record['path']} -> {record['content']}")
        
        print("\n✅ Same path, different projects = Different nodes!")
        
        # Project-scoped search
        scoped_search = await driver.execute_query("""
            MATCH (f:TestNode {project_name: $project, type: 'FixedImplementation'})
            RETURN f.content as content
        """, project="Project_A")
        
        print(f"\n✅ Project-scoped search for Project_A:")
        print(f"  Returns: {scoped_search.records[0]['content']}")
        print("  (Project_B data is correctly excluded)")
        
        # Cleanup
        await driver.execute_query("MATCH (n:TestNode) DETACH DELETE n")
        await driver.execute_query("MATCH (n:TestProject) DETACH DELETE n")
        
        print("\n" + "=" * 50)
        print("DEMONSTRATION COMPLETE")
        print("=" * 50)
        print("\nKEY FINDINGS:")
        print("1. ❌ Current implementation has NO project isolation")
        print("2. ❌ Files with same paths overwrite each other")
        print("3. ❌ Searches return mixed results from all projects")
        print("4. ❌ Statistics combine all projects together")
        print("5. ✅ Adding project_name to nodes enables proper isolation")
        print("6. ✅ Project-scoped queries prevent data mixing")
        
    finally:
        await driver.close()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          NEO4J RAG SYSTEM - DATA CORRUPTION PROOF            ║
║                                                              ║
║  This script demonstrates critical flaws in the current      ║
║  implementation that cause data corruption when multiple     ║
║  projects use the same Neo4j instance.                       ║
║                                                              ║
║  WARNING: Run only in test environment!                      ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(demonstrate_corruption())