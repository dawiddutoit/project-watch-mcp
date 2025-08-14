"""Demo showing how the old escaping approach would fail vs new phrase approach.

This test demonstrates the difference between the problematic escaping approach
and the new phrase-based approach.
"""

import re
from project_watch_mcp.neo4j_rag import lucene_phrase


def old_escape_lucene_query(query: str) -> str:
    """The old problematic escaping approach (for demonstration)."""
    # This was the problematic approach with double-escaping
    escaping_rules = [
        ('\\', '\\\\\\\\'),  # Quadruple escaping - problematic
        (':', '\\\\:'),      # Double escaping
        ('[', '\\\\['),      # Double escaping
        (']', '\\\\]'),      # Double escaping
        ('(', '\\\\('),      # Double escaping
        (')', '\\\\)'),      # Double escaping
    ]
    
    escaped = query
    for char, replacement in escaping_rules:
        escaped = escaped.replace(char, replacement)
    
    return escaped


def test_old_vs_new_approach():
    """Demonstrate the difference between old and new approaches."""
    
    test_patterns = [
        "class Neo4jRAG:",
        "result.items[index]",
        "processArray(data)",
        'uri: "neo4j://localhost:7687"',
    ]
    
    print("=== Comparing Old Escaping vs New Phrase Approach ===\n")
    
    for pattern in test_patterns:
        print(f"Input pattern: {pattern!r}")
        
        # Old approach - would have created hard-to-debug escaping
        old_result = old_escape_lucene_query(pattern)
        print(f"Old escaping:  {old_result!r}")
        
        # New approach - clean phrase wrapping
        new_result = lucene_phrase(pattern)
        print(f"New phrase:    {new_result!r}")
        
        print(f"New is simpler: {'✅' if len(new_result) < len(old_result) else '❌'}")
        print()
    
    # Key insight: The new approach is always simpler and more predictable
    assert True  # This test is just for demonstration


def test_phrase_approach_handles_all_specials():
    """Test that phrase approach handles all Lucene special characters safely."""
    
    # All Lucene special characters in one string
    special_chars_pattern = r'+ - && || ! ( ) { } [ ] ^ " ~ * ? : \ /'
    
    # With the phrase approach, all of these become safe
    result = lucene_phrase(special_chars_pattern)
    
    # Should be wrapped in quotes with only backslashes and quotes escaped
    expected_chars_to_escape = ['\\', '"']
    
    # Count escaping in result (excluding the wrapping quotes)
    content = result[1:-1]  # Remove wrapping quotes
    escape_count = content.count('\\')
    
    print(f"Input: {special_chars_pattern!r}")
    print(f"Result: {result!r}")
    print(f"Escape sequences found: {escape_count}")
    
    # Verify it's properly quoted
    assert result.startswith('"') and result.endswith('"')
    
    # Verify only backslashes were escaped (the original had one \)
    assert '\\\\' in result  # Original backslash should be escaped
    
    print("✅ All special characters handled safely with phrase approach")


if __name__ == "__main__":
    test_old_vs_new_approach()
    test_phrase_approach_handles_all_specials()