"""Test suite for Lucene query escaping functionality."""

import pytest

from src.project_watch_mcp.neo4j_rag import escape_lucene_query


class TestLuceneEscaping:
    """Test Lucene special character escaping."""

    def test_escape_colon(self):
        """Test escaping colon character."""
        assert escape_lucene_query("class Neo4jRAG:") == "class Neo4jRAG\\\\:"

    def test_escape_parentheses(self):
        """Test escaping parentheses."""
        assert escape_lucene_query("function test()") == "function test\\\\(\\\\)"

    def test_escape_brackets(self):
        """Test escaping square brackets."""
        assert escape_lucene_query("search [item]") == "search \\\\[item\\\\]"

    def test_escape_ampersand(self):
        """Test escaping ampersand - single & and | are not escaped, only && and ||."""
        assert escape_lucene_query("query & filter") == "query & filter"
        assert escape_lucene_query("query && filter") == "query \\\\&& filter"
        assert escape_lucene_query("query | filter") == "query | filter"
        assert escape_lucene_query("query || filter") == "query \\\\|| filter"

    def test_escape_forward_slash(self):
        """Test escaping forward slash."""
        assert escape_lucene_query("test/path") == "test\\\\/path"

    def test_escape_multiple_special_chars(self):
        """Test escaping multiple special characters."""
        input_query = "class Test(): def method(self, param: str) -> bool"
        expected = "class Test\\\\(\\\\)\\\\: def method\\\\(self, param\\\\: str\\\\) \\\\-> bool"
        assert escape_lucene_query(input_query) == expected

    def test_escape_all_special_chars(self):
        """Test escaping all supported special characters."""
        special_chars = "+-&|!(){}[]^\"~*?:\\/"
        escaped = escape_lucene_query(special_chars)

        # Single & and | are not escaped, backslash is quadruple-escaped
        expected = "\\\\+\\\\-&|\\\\!\\\\(\\\\)\\\\{\\\\}\\\\[\\\\]\\\\^\\\\\"\\\\~\\\\*\\\\?\\\\:\\\\\\\\\\\\/"
        assert escaped == expected

    def test_no_escaping_normal_text(self):
        """Test that normal text without special characters is unchanged."""
        normal_text = "def function_name with spaces"
        assert escape_lucene_query(normal_text) == normal_text

    def test_empty_string(self):
        """Test escaping empty string."""
        assert escape_lucene_query("") == ""

    def test_whitespace_only(self):
        """Test escaping whitespace-only string."""
        assert escape_lucene_query("   ") == "   "


class TestLuceneEscapingEdgeCases:
    """Edge case tests for Lucene escaping."""

    def test_escape_backslash(self):
        """Test escaping backslash character gets quadruple-escaped."""
        assert escape_lucene_query("path\\\\to\\\\file") == "path\\\\\\\\\\\\\\\\to\\\\\\\\\\\\\\\\file"
        assert escape_lucene_query("\\\\") == "\\\\\\\\\\\\\\\\"
    
    def test_escape_quotes(self):
        """Test escaping quote characters."""
        assert escape_lucene_query('"quoted text"') == '\\\\"quoted text\\\\"'
        assert escape_lucene_query("it's") == "it's"  # Single quotes not escaped
    
    def test_escape_operators(self):
        """Test escaping boolean operators."""
        assert escape_lucene_query("AND") == "AND"  # Words not escaped
        assert escape_lucene_query("&&") == "\\\\&&"
        assert escape_lucene_query("||") == "\\\\||"
        assert escape_lucene_query("NOT") == "NOT"  # Words not escaped
        assert escape_lucene_query("!important") == "\\\\!important"
    
    def test_escape_wildcards(self):
        """Test escaping wildcard characters."""
        assert escape_lucene_query("test*") == "test\\\\*"
        assert escape_lucene_query("te?t") == "te\\\\?t"
        assert escape_lucene_query("~fuzzy") == "\\\\~fuzzy"
    
    def test_escape_special_sequences(self):
        """Test escaping special character sequences."""
        assert escape_lucene_query("->") == "\\\\->"
        assert escape_lucene_query("a+b") == "a\\\\+b"
        assert escape_lucene_query("field^2") == "field\\\\^2"
    
    def test_real_world_patterns(self):
        """Test real-world code patterns."""
        # Python type hints
        assert escape_lucene_query("def func(param: str) -> bool:") == "def func\\\\(param\\\\: str\\\\) \\\\-> bool\\\\:"
        
        # JavaScript arrow function
        assert escape_lucene_query("const fn = () => {}") == "const fn = \\\\(\\\\) => \\\\{\\\\}"
        
        # Regex pattern
        assert escape_lucene_query("[a-zA-Z0-9]+") == "\\\\[a\\\\-zA\\\\-Z0\\\\-9\\\\]\\\\+"
        
        # File path with extension
        assert escape_lucene_query("src/main.py:42") == "src\\\\/main.py\\\\:42"


class TestLuceneEscapingIntegration:
    """Integration tests for Lucene escaping in pattern search."""

    @pytest.mark.parametrize("pattern,expected", [
        ("class Neo4jRAG:", "class Neo4jRAG\\\\:"),
        ("function test()", "function test\\\\(\\\\)"),
        ("key:value", "key\\\\:value"),
        ("search [item]", "search \\\\[item\\\\]"),
        ("test/path", "test\\\\/path"),  # Forward slash gets double-escaped
    ])
    def test_various_patterns_escape_correctly(self, pattern, expected):
        """Test that various problematic patterns are escaped correctly."""
        assert escape_lucene_query(pattern) == expected

