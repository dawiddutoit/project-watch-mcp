"""Integration tests for Lucene pattern search with special characters.

This test directly interacts with Neo4j to verify that our phrase-based search
approach correctly handles special characters that would cause Lucene parsing errors.
"""

import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from neo4j import AsyncDriver, AsyncGraphDatabase

from project_watch_mcp.neo4j_rag import CodeFile, Neo4jRAG


@pytest.fixture
async def neo4j_driver():
    """Create a Neo4j driver for testing."""
    uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        pytest.skip("NEO4J_PASSWORD not set - skipping integration test")
    
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    # Test connection
    try:
        await driver.verify_connectivity()
    except Exception as e:
        pytest.skip(f"Cannot connect to Neo4j: {e}")
    
    yield driver
    
    await driver.close()


@pytest.fixture
async def neo4j_rag(neo4j_driver):
    """Create a Neo4j RAG instance for testing."""
    test_project = f"test_lucene_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    rag = Neo4jRAG(
        neo4j_driver=neo4j_driver,
        project_name=test_project,
        embeddings=None,  # Disable embeddings for this test
    )
    
    await rag.initialize()
    
    yield rag
    
    # Clean up test data
    await neo4j_driver.execute_query(
        "MATCH (n) WHERE n.project_name = $project_name DETACH DELETE n",
        {"project_name": test_project}
    )


@pytest.fixture
def sample_code_files(tmp_path):
    """Create sample code files with various special characters."""
    files = []
    
    # Python file with class definition (colon)
    python_file = tmp_path / "models.py"
    python_content = '''class Neo4jRAG:
    """Neo4j-based RAG system."""
    
    def __init__(self, driver):
        self.driver = driver
    
    def search_by_pattern(self, pattern: str):
        """Search using pattern."""
        return []

class UserManager:
    """Manages user operations."""
    pass
'''
    python_file.write_text(python_content)
    files.append(CodeFile(
        project_name="test_project",
        path=python_file,
        content=python_content,
        language="python",
        size=len(python_content),
        last_modified=datetime.now()
    ))
    
    # JavaScript file with array access (brackets)
    js_file = tmp_path / "utils.js"
    js_content = '''function processArray(data) {
    const result = data[0];
    if (result && result.items) {
        return result.items[index];
    }
    return null;
}

const config = {
    "database": "neo4j",
    "settings": {
        "timeout": 5000
    }
};

// Test with parentheses and quotes
console.log("Processing data[item]");
'''
    js_file.write_text(js_content)
    files.append(CodeFile(
        project_name="test_project",
        path=js_file,
        content=js_content,
        language="javascript",
        size=len(js_content),
        last_modified=datetime.now()
    ))
    
    # Config file with various special characters
    config_file = tmp_path / "config.yml"
    config_content = '''database:
  uri: "neo4j://localhost:7687"
  auth:
    user: "neo4j"
    password: "secret!"

search:
  patterns: ["*.py", "*.js"]
  exclude: [".git/*", "*/node_modules/*"]
  
# Special characters test
regex_patterns:
  - "\\d{3}-\\d{3}-\\d{4}"  # Phone number
  - "[A-Za-z]+@[A-Za-z]+\\.[A-Za-z]{2,}"  # Email
  - "function\\(.*\\)"  # Function calls
'''
    config_file.write_text(config_content)
    files.append(CodeFile(
        project_name="test_project",
        path=config_file,
        content=config_content,
        language="yaml",
        size=len(config_content),
        last_modified=datetime.now()
    ))
    
    return files


@pytest.mark.asyncio
async def test_lucene_phrase_search_special_characters(neo4j_rag, sample_code_files):
    """Test that pattern search correctly handles special characters using phrase search."""
    
    # Index the sample files
    for code_file in sample_code_files:
        await neo4j_rag.index_file(code_file)
    
    # Test cases that would have failed with the old escaping approach
    test_cases = [
        # Colon in class definition - the original failing case
        {
            "pattern": "class Neo4jRAG:",
            "expected_matches": 1,
            "description": "Class definition with colon"
        },
        
        # Array access with brackets
        {
            "pattern": "result.items[index]",
            "expected_matches": 1,
            "description": "Array access with brackets"
        },
        
        # Function call with parentheses
        {
            "pattern": "processArray(data)",
            "expected_matches": 1,
            "description": "Function call with parentheses"
        },
        
        # String with quotes and brackets
        {
            "pattern": '"Processing data[item]"',
            "expected_matches": 1,
            "description": "String literal with quotes and brackets"
        },
        
        # YAML key-value with colon
        {
            "pattern": 'uri: "neo4j://localhost:7687"',
            "expected_matches": 1,
            "description": "YAML with colon and quotes"
        },
        
        # Regex pattern with backslashes
        {
            "pattern": '"\\\\d{3}-\\\\d{3}-\\\\d{4}"',
            "expected_matches": 1,
            "description": "Regex pattern with escaped backslashes"
        },
        
        # Multiple special characters
        {
            "pattern": "function\\(.*\\)",
            "expected_matches": 1,
            "description": "Complex pattern with multiple special chars"
        },
        
        # Simple word should still work
        {
            "pattern": "database",
            "expected_matches": 2,  # Should match in both JS and YAML files
            "description": "Simple word search"
        }
    ]
    
    # Run each test case
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['description']}")
        print(f"Pattern: {test_case['pattern']!r}")
        
        try:
            # This should NOT raise a Lucene ParseException
            results = await neo4j_rag.search_by_pattern(
                pattern=test_case['pattern'],
                is_regex=False,
                limit=10
            )
            
            print(f"Found {len(results)} results")
            for result in results:
                print(f"  - {result.file_path.name}:{result.line_number}")
                print(f"    Content: {result.content[:100]}...")
            
            # Verify we found the expected number of matches
            assert len(results) >= test_case['expected_matches'], \
                f"Expected at least {test_case['expected_matches']} matches for pattern {test_case['pattern']!r}, got {len(results)}"
            
        except Exception as e:
            pytest.fail(f"Pattern search failed for {test_case['pattern']!r}: {e}")


@pytest.mark.asyncio
async def test_regex_search_still_works(neo4j_rag, sample_code_files):
    """Test that regex search functionality remains intact and doesn't error."""
    
    # Index the sample files
    for code_file in sample_code_files:
        await neo4j_rag.index_file(code_file)
    
    # Test that regex search executes without errors (even if no matches)
    regex_test_cases = [
        r"class.*:",
        r"function.*\(",
        r"\w+\[.*\]",
        r"Neo4j",
        r".*password.*",  # This should match something
    ]
    
    total_results = 0
    for i, pattern in enumerate(regex_test_cases):
        print(f"\nRegex test {i+1}: {pattern!r}")
        
        # The key test: this should NOT raise an exception
        results = await neo4j_rag.search_by_pattern(
            pattern=pattern,
            is_regex=True,  # Use regex mode
            limit=10
        )
        
        print(f"Found {len(results)} results")
        total_results += len(results)
        
        # Just verify it didn't error - we don't care about exact matches
        # because chunking can affect what matches
    
    # At least one of our patterns should have matched something
    assert total_results > 0, "Expected at least some regex matches across all patterns"
    print(f"✅ Regex search works - total matches across all patterns: {total_results}")


@pytest.mark.asyncio
async def test_phrase_search_logging(neo4j_rag, sample_code_files, caplog):
    """Test that phrase search logging works correctly."""
    import logging
    
    # Set logging level to capture our debug messages
    caplog.set_level(logging.INFO, logger="project_watch_mcp.neo4j_rag")
    
    # Index a sample file
    await neo4j_rag.index_file(sample_code_files[0])
    
    # Perform a search with special characters
    pattern = "class Neo4jRAG:"
    await neo4j_rag.search_by_pattern(pattern, is_regex=False, limit=5)
    
    # Check that logging occurred
    log_messages = [record.message for record in caplog.records]
    
    # Should have logs for phrase conversion
    phrase_logs = [msg for msg in log_messages if "LUCENE-PHRASE" in msg]
    assert len(phrase_logs) >= 2, f"Expected phrase conversion logs, got: {log_messages}"
    
    # Should show the pattern being converted to a phrase
    conversion_log = [msg for msg in phrase_logs if "Converting to phrase" in msg]
    assert len(conversion_log) >= 1, "Expected phrase conversion log"
    
    # Should show the result
    result_log = [msg for msg in phrase_logs if "Result:" in msg]
    assert len(result_log) >= 1, "Expected phrase result log"
    
    print("✅ Logging verification passed")
    for msg in phrase_logs:
        print(f"  Log: {msg}")


if __name__ == "__main__":
    # Allow running this test directly for development
    pytest.main([__file__, "-v", "-s"])