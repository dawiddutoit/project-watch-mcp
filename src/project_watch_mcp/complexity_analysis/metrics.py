"""Common metrics and calculation utilities for complexity analysis.

This module provides language-agnostic metric calculations and utilities
that can be used across different language analyzers.
"""

import math
from typing import Any, List, Tuple


class ComplexityMetrics:
    """Utility class for calculating various complexity metrics."""
    
    @staticmethod
    def calculate_nesting_depth(ast_node: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth in an AST node.
        
        Args:
            ast_node: Abstract syntax tree node
            current_depth: Current nesting level
            
        Returns:
            Maximum nesting depth found
        """
        max_depth = current_depth
        
        # This is a generic implementation
        # Language-specific analyzers should override with proper AST traversal
        if hasattr(ast_node, 'children'):
            for child in ast_node.children:
                child_depth = ComplexityMetrics.calculate_nesting_depth(
                    child, current_depth + 1
                )
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    @staticmethod
    def count_decision_points(conditions: int, loops: int, cases: int) -> int:
        """Count decision points for cyclomatic complexity.
        
        Args:
            conditions: Number of if/elif conditions
            loops: Number of loops (for/while)
            cases: Number of switch/match cases
            
        Returns:
            Total decision points
        """
        return conditions + loops + cases
    
    @staticmethod
    def calculate_halstead_volume(
        operators: int,
        operands: int,
        unique_operators: int,
        unique_operands: int
    ) -> float:
        """Calculate Halstead volume metric.
        
        Halstead Volume = N * log2(n)
        Where N = total operators + operands
        And n = unique operators + operands
        
        Args:
            operators: Total number of operators
            operands: Total number of operands
            unique_operators: Number of unique operators
            unique_operands: Number of unique operands
            
        Returns:
            Halstead volume
        """
        total_items = operators + operands
        unique_items = unique_operators + unique_operands
        
        if unique_items == 0:
            return 0.0
        
        return total_items * math.log2(unique_items)
    
    @staticmethod
    def calculate_halstead_difficulty(
        unique_operators: int,
        unique_operands: int,
        total_operands: int
    ) -> float:
        """Calculate Halstead difficulty metric.
        
        Difficulty = (n1 / 2) * (N2 / n2)
        Where n1 = unique operators, N2 = total operands, n2 = unique operands
        
        Args:
            unique_operators: Number of unique operators
            unique_operands: Number of unique operands
            total_operands: Total number of operands
            
        Returns:
            Halstead difficulty
        """
        if unique_operands == 0:
            return 0.0
        
        return (unique_operators / 2) * (total_operands / unique_operands)
    
    @staticmethod
    def calculate_cognitive_increment(
        nesting_level: int,
        is_nested_flow: bool = False
    ) -> int:
        """Calculate cognitive complexity increment.
        
        Cognitive complexity increments based on:
        - Nesting level (exponential increase)
        - Whether it's a nested flow break
        
        Args:
            nesting_level: Current nesting level
            is_nested_flow: Whether this is a nested flow break
            
        Returns:
            Cognitive complexity increment
        """
        base_increment = 1
        
        # Add nesting penalty
        if nesting_level > 0:
            base_increment += nesting_level
        
        # Additional penalty for nested flow breaks
        if is_nested_flow and nesting_level > 0:
            base_increment += 1
        
        return base_increment
    
    @staticmethod
    def count_logical_operators(expression: str) -> int:
        """Count logical operators in an expression.
        
        Counts &&, ||, and, or, not operators.
        
        Args:
            expression: Code expression as string
            
        Returns:
            Number of logical operators
        """
        operators = ['&&', '||', ' and ', ' or ', ' not ']
        count = 0
        
        for op in operators:
            count += expression.count(op)
        
        return count
    
    @staticmethod
    def estimate_comment_ratio(
        lines: List[str],
        single_comment: str = '#',
        multi_start: str = '/*',
        multi_end: str = '*/'
    ) -> float:
        """Estimate the ratio of comment lines to total lines.
        
        Args:
            lines: List of code lines
            single_comment: Single-line comment marker
            multi_start: Multi-line comment start marker
            multi_end: Multi-line comment end marker
            
        Returns:
            Ratio of comment lines to total lines (0.0 to 1.0)
        """
        if not lines:
            return 0.0
        
        comment_lines = 0
        in_multiline = False
        
        for line in lines:
            line = line.strip()
            
            # Check for multi-line comment markers
            if multi_start in line:
                in_multiline = True
            
            # Count as comment if it's a comment line
            if in_multiline or line.startswith(single_comment):
                comment_lines += 1
            
            # Check for multi-line comment end
            if multi_end in line:
                in_multiline = False
        
        return comment_lines / len(lines)
    
    @staticmethod
    def classify_complexity_level(score: int) -> Tuple[str, str]:
        """Classify complexity level and provide description.
        
        Args:
            score: Complexity score
            
        Returns:
            Tuple of (classification, description)
        """
        if score <= 5:
            return ("simple", "Easy to understand and maintain")
        elif score <= 10:
            return ("moderate", "Reasonably clear, some refactoring may help")
        elif score <= 20:
            return ("complex", "Difficult to understand, should be refactored")
        else:
            return ("very-complex", "Very difficult to understand, needs refactoring")
    
    @staticmethod
    def calculate_risk_score(
        cyclomatic: int,
        cognitive: int,
        nesting: int,
        lines: int
    ) -> float:
        """Calculate an overall risk score for the code.
        
        Combines multiple metrics into a single risk score.
        
        Args:
            cyclomatic: Cyclomatic complexity
            cognitive: Cognitive complexity
            nesting: Maximum nesting depth
            lines: Lines of code
            
        Returns:
            Risk score (0-100, higher is riskier)
        """
        # Normalize each metric
        cyc_risk = min(cyclomatic / 30 * 100, 100)  # 30+ is max risk
        cog_risk = min(cognitive / 25 * 100, 100)    # 25+ is max risk
        nest_risk = min(nesting / 6 * 100, 100)      # 6+ levels is max risk
        size_risk = min(lines / 200 * 100, 100)      # 200+ lines is max risk
        
        # Weighted average
        weights = {
            'cyclomatic': 0.35,
            'cognitive': 0.35,
            'nesting': 0.20,
            'size': 0.10
        }
        
        risk = (
            cyc_risk * weights['cyclomatic'] +
            cog_risk * weights['cognitive'] +
            nest_risk * weights['nesting'] +
            size_risk * weights['size']
        )
        
        return min(risk, 100.0)