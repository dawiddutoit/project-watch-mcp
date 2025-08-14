"""Unit tests for ComplexityMetrics class."""

import pytest
from src.project_watch_mcp.complexity_analysis.metrics import ComplexityMetrics


class TestComplexityMetrics:
    """Test the ComplexityMetrics class methods."""
    
    @pytest.fixture
    def metrics(self):
        """Create a ComplexityMetrics instance."""
        return ComplexityMetrics()
    
    def test_calculate_nesting_depth(self, metrics):
        """Test calculating nesting depth."""
        # Mock AST node with children
        class MockNode:
            def __init__(self, children=None):
                self.children = children or []
        
        # Simple node with no children
        node = MockNode()
        assert metrics.calculate_nesting_depth(node) == 0
        
        # Nested structure
        child2 = MockNode()
        child1 = MockNode([child2])
        root = MockNode([child1])
        assert metrics.calculate_nesting_depth(root) == 2
    
    def test_count_decision_points(self, metrics):
        """Test counting decision points."""
        # No decision points
        assert metrics.count_decision_points(0, 0, 0) == 0
        
        # Various decision points
        assert metrics.count_decision_points(3, 2, 4) == 9
        assert metrics.count_decision_points(1, 1, 1) == 3
    
    def test_calculate_halstead_volume(self, metrics):
        """Test Halstead volume calculation."""
        # Empty code
        assert metrics.calculate_halstead_volume(0, 0, 0, 0) == 0.0
        
        # Simple calculation
        volume = metrics.calculate_halstead_volume(10, 8, 5, 4)
        assert volume > 0
        # N=18, n=9, volume = 18 * log2(9) ≈ 18 * 3.17 ≈ 57
        assert 55 < volume < 60
        
        # Single unique item
        volume = metrics.calculate_halstead_volume(1, 0, 1, 0)
        assert volume == 0.0  # log2(1) = 0
    
    def test_calculate_halstead_difficulty(self, metrics):
        """Test Halstead difficulty calculation."""
        # No operands
        assert metrics.calculate_halstead_difficulty(5, 0, 10) == 0.0
        
        # Normal calculation
        difficulty = metrics.calculate_halstead_difficulty(6, 4, 12)
        # (6/2) * (12/4) = 3 * 3 = 9
        assert difficulty == 9.0
        
        # Edge case with 1 unique operand
        difficulty = metrics.calculate_halstead_difficulty(2, 1, 5)
        # (2/2) * (5/1) = 1 * 5 = 5
        assert difficulty == 5.0
    
    def test_calculate_cognitive_increment(self, metrics):
        """Test cognitive complexity increment calculation."""
        # No nesting
        assert metrics.calculate_cognitive_increment(0, False) == 1
        
        # With nesting
        assert metrics.calculate_cognitive_increment(1, False) == 2
        assert metrics.calculate_cognitive_increment(2, False) == 3
        assert metrics.calculate_cognitive_increment(3, False) == 4
        
        # Nested flow break
        assert metrics.calculate_cognitive_increment(0, True) == 1  # No extra at level 0
        assert metrics.calculate_cognitive_increment(1, True) == 3  # Base 1 + nesting 1 + flow 1
        assert metrics.calculate_cognitive_increment(2, True) == 4  # Base 1 + nesting 2 + flow 1
    
    def test_count_logical_operators(self, metrics):
        """Test counting logical operators."""
        # No operators
        assert metrics.count_logical_operators("x = 1") == 0
        
        # Various operators
        assert metrics.count_logical_operators("if a && b || c") == 2
        assert metrics.count_logical_operators("if a and b or not c") == 3
        assert metrics.count_logical_operators("x = a && (b || c) && d") == 3
        
        # Mixed operators
        assert metrics.count_logical_operators("if (a and b) || (c && d)") == 3
    
    def test_estimate_comment_ratio(self, metrics):
        """Test comment ratio estimation."""
        # No lines
        assert metrics.estimate_comment_ratio([]) == 0.0
        
        # No comments
        lines = ["def hello():", "    return 'hi'"]
        assert metrics.estimate_comment_ratio(lines) == 0.0
        
        # Single-line comments
        lines = [
            "# This is a comment",
            "def hello():",
            "    # Another comment",
            "    return 'hi'"
        ]
        ratio = metrics.estimate_comment_ratio(lines, '#')
        assert ratio == 0.5  # 2 out of 4 lines
        
        # Multi-line comments
        lines = [
            "/*",
            "Multi-line",
            "comment",
            "*/",
            "code line"
        ]
        ratio = metrics.estimate_comment_ratio(lines, '#', '/*', '*/')
        assert ratio == 0.8  # 4 out of 5 lines
        
        # Python docstrings
        lines = [
            '"""',
            "This is a",
            "docstring",
            '"""',
            "def hello():",
            "    pass"
        ]
        ratio = metrics.estimate_comment_ratio(lines, '#', '"""', '"""')
        assert ratio == 4/6  # 4 out of 6 lines
    
    def test_classify_complexity_level(self, metrics):
        """Test complexity level classification."""
        # Simple complexity
        level, desc = metrics.classify_complexity_level(3)
        assert level == "simple"
        assert "simple" in desc.lower()
        
        # Moderate complexity
        level, desc = metrics.classify_complexity_level(8)
        assert level == "moderate"
        assert "moderate" in desc.lower()
        
        # Complex
        level, desc = metrics.classify_complexity_level(15)
        assert level == "complex"
        assert "attention" in desc.lower()
        
        # Very complex
        level, desc = metrics.classify_complexity_level(25)
        assert level == "very_complex"
        assert "refactor" in desc.lower()
        
        # Boundary values
        level, _ = metrics.classify_complexity_level(5)
        assert level == "simple"
        level, _ = metrics.classify_complexity_level(10)
        assert level == "moderate"
        level, _ = metrics.classify_complexity_level(20)
        assert level == "complex"
    
    def test_calculate_risk_score(self, metrics):
        """Test risk score calculation."""
        # No risk
        score = metrics.calculate_risk_score(0, 0, 0, 0)
        assert score == 0.0
        
        # Low risk
        score = metrics.calculate_risk_score(5, 5, 2, 50)
        assert 0 < score < 30
        
        # Medium risk
        score = metrics.calculate_risk_score(15, 15, 4, 100)
        assert 30 < score < 70
        
        # High risk
        score = metrics.calculate_risk_score(30, 25, 6, 200)
        assert score >= 70
        
        # Maximum risk (capped at 100)
        score = metrics.calculate_risk_score(100, 100, 10, 500)
        assert score == 100.0
        
        # Check weighting
        # Only cyclomatic complexity
        score = metrics.calculate_risk_score(30, 0, 0, 0)
        assert score == pytest.approx(35.0, rel=0.1)  # 100 * 0.35
        
        # Only cognitive complexity
        score = metrics.calculate_risk_score(0, 25, 0, 0)
        assert score == pytest.approx(35.0, rel=0.1)  # 100 * 0.35
        
        # Only nesting
        score = metrics.calculate_risk_score(0, 0, 6, 0)
        assert score == pytest.approx(20.0, rel=0.1)  # 100 * 0.20
        
        # Only size
        score = metrics.calculate_risk_score(0, 0, 0, 200)
        assert score == pytest.approx(10.0, rel=0.1)  # 100 * 0.10