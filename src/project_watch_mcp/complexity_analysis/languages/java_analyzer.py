"""Java-specific complexity analyzer implementation.

This module provides complexity analysis for Java code using tree-sitter parser
while conforming to the unified analyzer interface.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

from ..base_analyzer import BaseComplexityAnalyzer
from ..metrics import ComplexityMetrics
from ..models import (
    ClassComplexity,
    ComplexityClassification,
    ComplexityGrade,
    ComplexityResult,
    ComplexitySummary,
    FunctionComplexity,
)

logger = logging.getLogger(__name__)

# Import tree-sitter for Java parsing
try:
    import tree_sitter
    from tree_sitter import Language, Parser
    import tree_sitter_java as tsjava
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter-java not available. Java analysis will be limited.")


class JavaComplexityAnalyzer(BaseComplexityAnalyzer):
    """Complexity analyzer for Java code.
    
    Uses tree-sitter parser for accurate AST-based analysis,
    supporting modern Java features including lambdas, streams, and records.
    """
    
    def __init__(self):
        """Initialize Java analyzer."""
        super().__init__("java")
        self.metrics = ComplexityMetrics()
        self._parser = None
        self._init_parser()
    
    def _init_parser(self):
        """Initialize tree-sitter parser for Java."""
        if not TREE_SITTER_AVAILABLE:
            return
        
        try:
            # Build Java language - newer API
            JAVA_LANGUAGE = Language(tsjava.language())
            self._parser = Parser(JAVA_LANGUAGE)
        except Exception as e:
            logger.error(f"Failed to initialize Java parser: {e}")
            self._parser = None
    
    async def analyze_file(self, file_path: Path) -> ComplexityResult:
        """Analyze complexity of a Java file.
        
        Args:
            file_path: Path to the Java file
            
        Returns:
            ComplexityResult with all metrics
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not str(file_path).endswith('.java'):
            raise ValueError(f"Not a Java file: {file_path}")
        
        try:
            code = file_path.read_text(encoding='utf-8')
            result = await self.analyze_code(code)
            result.file_path = str(file_path)
            return result
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return self._create_error_result(str(file_path), str(e))
    
    async def analyze_code(self, code: str) -> ComplexityResult:
        """Analyze complexity of Java code string.
        
        Args:
            code: Java source code
            
        Returns:
            ComplexityResult with all metrics
        """
        if not TREE_SITTER_AVAILABLE or not self._parser:
            return self._create_error_result(None, "Tree-sitter Java parser not available")
        
        try:
            # Parse the code
            tree = self._parser.parse(bytes(code, 'utf-8'))
            root_node = tree.root_node
            
            # Extract functions and classes
            functions = self._extract_functions(root_node, code)
            classes = self._extract_classes(root_node, code)
            
            # Calculate summary metrics
            summary = self._calculate_summary(functions, classes, code)
            
            # Generate recommendations
            result = ComplexityResult(
                file_path=None,
                language=self.language,
                summary=summary,
                functions=functions,
                classes=classes,
            )
            result.recommendations = result.generate_recommendations()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze Java code: {e}")
            return self._create_error_result(None, str(e))
    
    def _extract_functions(self, node, code: str) -> List[FunctionComplexity]:
        """Extract all functions/methods from the AST."""
        functions = []
        self._walk_tree_for_functions(node, code, functions)
        return functions
    
    def _walk_tree_for_functions(self, node, code: str, functions: List[FunctionComplexity], 
                                  parent_class: Optional[str] = None, depth: int = 0):
        """Recursively walk the tree to find functions with depth protection."""
        # Maximum recursion depth to prevent stack overflow
        MAX_DEPTH = 100
        
        if depth > MAX_DEPTH:
            logger.warning(f"Maximum recursion depth {MAX_DEPTH} reached in function extraction")
            return
        
        # Method declarations
        if node.type == 'method_declaration':
            func = self._analyze_method(node, code, parent_class)
            if func:
                functions.append(func)
        
        # Constructor declarations
        elif node.type == 'constructor_declaration':
            func = self._analyze_constructor(node, code, parent_class)
            if func:
                functions.append(func)
        
        # Lambda expressions
        elif node.type == 'lambda_expression':
            func = self._analyze_lambda(node, code, parent_class)
            if func:
                functions.append(func)
        
        # Track class context for nested methods
        class_name = parent_class
        if node.type in ['class_declaration', 'interface_declaration', 'enum_declaration', 'record_declaration']:
            name_node = self._find_child_by_type(node, 'identifier')
            if name_node:
                class_name = name_node.text.decode('utf-8')
        
        # Recurse through children with incremented depth
        for child in node.children:
            self._walk_tree_for_functions(child, code, functions, class_name, depth + 1)
    
    def _analyze_method(self, node, code: str, parent_class: Optional[str]) -> Optional[FunctionComplexity]:
        """Analyze a method declaration."""
        # Get method name
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return None
        
        method_name = name_node.text.decode('utf-8')
        if parent_class:
            method_name = f"{parent_class}.{method_name}"
        
        # Check if it's abstract or interface method (no body)
        body_node = self._find_child_by_type(node, 'block')
        if not body_node:
            return None  # Abstract method, no complexity
        
        # Calculate complexity
        cyclomatic = self.calculate_cyclomatic_complexity(body_node)
        cognitive = self.calculate_cognitive_complexity(body_node)
        
        # Get parameters
        params_node = self._find_child_by_type(node, 'formal_parameters')
        param_count = self._count_parameters(params_node) if params_node else 0
        
        # Get line numbers
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        # Calculate nesting depth
        depth = self._calculate_max_depth(body_node, 0, 0)
        
        return FunctionComplexity(
            name=method_name,
            complexity=cyclomatic,
            cognitive_complexity=cognitive,
            rank=ComplexityGrade.from_complexity(cyclomatic).value,
            line_start=start_line,
            line_end=end_line,
            classification=ComplexityClassification.from_complexity(cyclomatic).value,
            parameters=param_count,
            depth=depth,
            type="method"
        )
    
    def _analyze_constructor(self, node, code: str, parent_class: Optional[str]) -> Optional[FunctionComplexity]:
        """Analyze a constructor declaration."""
        # Constructor name is the class name
        constructor_name = parent_class or "Constructor"
        
        # Get constructor body
        body_node = self._find_child_by_type(node, 'constructor_body')
        if not body_node:
            return None
        
        # Calculate complexity
        cyclomatic = self.calculate_cyclomatic_complexity(body_node)
        cognitive = self.calculate_cognitive_complexity(body_node)
        
        # Get parameters
        params_node = self._find_child_by_type(node, 'formal_parameters')
        param_count = self._count_parameters(params_node) if params_node else 0
        
        # Get line numbers
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        # Calculate nesting depth
        depth = self._calculate_max_depth(body_node, 0, 0)
        
        return FunctionComplexity(
            name=constructor_name,
            complexity=cyclomatic,
            cognitive_complexity=cognitive,
            rank=ComplexityGrade.from_complexity(cyclomatic).value,
            line_start=start_line,
            line_end=end_line,
            classification=ComplexityClassification.from_complexity(cyclomatic).value,
            parameters=param_count,
            depth=depth,
            type="constructor"
        )
    
    def _analyze_lambda(self, node, code: str, parent_class: Optional[str]) -> Optional[FunctionComplexity]:
        """Analyze a lambda expression."""
        # Generate a name for the lambda
        lambda_name = f"lambda@{node.start_point[0]+1}:{node.start_point[1]}"
        if parent_class:
            lambda_name = f"{parent_class}.{lambda_name}"
        
        # Get lambda body
        body_node = self._find_child_by_type(node, 'block')
        if not body_node:
            # Single expression lambda
            body_node = node
        
        # Lambda adds 1 to complexity
        cyclomatic = 1
        cognitive = 1
        
        # Add complexity from body
        if body_node and body_node != node:
            cyclomatic = self.calculate_cyclomatic_complexity(body_node)
            cognitive = self.calculate_cognitive_complexity(body_node)
        
        # Get line numbers
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        return FunctionComplexity(
            name=lambda_name,
            complexity=cyclomatic,
            cognitive_complexity=cognitive,
            rank=ComplexityGrade.from_complexity(cyclomatic).value,
            line_start=start_line,
            line_end=end_line,
            classification=ComplexityClassification.from_complexity(cyclomatic).value,
            parameters=0,  # Simplified for lambdas
            depth=0,
            type="lambda"
        )
    
    def _extract_classes(self, node, code: str) -> List[ClassComplexity]:
        """Extract all classes from the AST."""
        classes = []
        self._walk_tree_for_classes(node, code, classes)
        return classes
    
    def _walk_tree_for_classes(self, node, code: str, classes: List[ClassComplexity], 
                                parent_class: Optional[ClassComplexity] = None, depth: int = 0):
        """Recursively walk the tree to find classes with depth protection."""
        # Maximum recursion depth to prevent stack overflow
        MAX_DEPTH = 100
        
        if depth > MAX_DEPTH:
            logger.warning(f"Maximum recursion depth {MAX_DEPTH} reached in class extraction")
            return
        
        if node.type in ['class_declaration', 'interface_declaration', 'enum_declaration', 'record_declaration']:
            class_obj = self._analyze_class(node, code)
            if class_obj:
                if parent_class:
                    parent_class.nested_classes.append(class_obj)
                else:
                    classes.append(class_obj)
                
                # Look for nested classes within this class with incremented depth
                for child in node.children:
                    self._walk_tree_for_classes(child, code, classes, class_obj, depth + 1)
                return
        
        # Continue searching with incremented depth
        for child in node.children:
            self._walk_tree_for_classes(child, code, classes, parent_class, depth + 1)
    
    def _analyze_class(self, node, code: str) -> Optional[ClassComplexity]:
        """Analyze a class declaration."""
        # Get class name
        name_node = self._find_child_by_type(node, 'identifier')
        if not name_node:
            return None
        
        class_name = name_node.text.decode('utf-8')
        
        # Find all methods in this class
        methods = []
        for child in node.children:
            if child.type == 'class_body':
                for body_item in child.children:
                    if body_item.type == 'method_declaration':
                        method = self._analyze_method(body_item, code, None)
                        if method:
                            methods.append(method)
                    elif body_item.type == 'constructor_declaration':
                        constructor = self._analyze_constructor(body_item, code, class_name)
                        if constructor:
                            methods.append(constructor)
        
        # Calculate class metrics
        total_complexity = sum(m.complexity for m in methods)
        avg_complexity = total_complexity / len(methods) if methods else 0
        
        # Get line numbers
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        return ClassComplexity(
            name=class_name,
            total_complexity=total_complexity,
            average_method_complexity=avg_complexity,
            method_count=len(methods),
            line_start=start_line,
            line_end=end_line,
            methods=methods,
            nested_classes=[]
        )
    
    def calculate_cyclomatic_complexity(self, ast_node: Any) -> int:
        """Calculate cyclomatic complexity for an AST node.
        
        Java-specific rules:
        - Base complexity: 1
        - +1 for: if, else if, for, while, do-while, case, catch, &&, ||, ?:
        - +1 for each lambda expression
        - +1 for each stream intermediate operation
        """
        if not ast_node:
            return 1
        
        complexity = 1
        
        def walk_node(node, parent_type=None):
            nonlocal complexity
            
            # Control flow statements
            if node.type in ['if_statement', 'for_statement', 'while_statement', 
                              'do_statement', 'enhanced_for_statement']:
                complexity += 1
            
            # Switch statement itself doesn't add complexity, only the cases do
            # Skip switch_statement node
            
            # Switch cases (each case label adds complexity)
            elif node.type == 'switch_label':
                # Check if it's a case or default
                has_case = any(child.type == 'case' for child in node.children)
                has_default = any(child.type == 'default' for child in node.children)
                if has_case or has_default:
                    complexity += 1
            
            # Catch clauses
            elif node.type == 'catch_clause':
                complexity += 1
            
            # Ternary operator
            elif node.type == 'ternary_expression':
                complexity += 1
            
            # Logical operators (short-circuit evaluation)
            elif node.type == 'binary_expression':
                operator = self._get_binary_operator(node)
                if operator in ['&&', '||']:
                    complexity += 1
            
            # Lambda expressions
            elif node.type == 'lambda_expression':
                complexity += 1
            
            # Stream operations
            elif node.type == 'method_invocation':
                method_name = self._get_method_name(node)
                if method_name in ['filter', 'map', 'flatMap', 'distinct', 'sorted', 
                                    'peek', 'limit', 'skip']:
                    complexity += 1
            
            # Else if detection (else followed by if)
            elif node.type == 'else' and parent_type == 'if_statement':
                # Check if next sibling is another if
                for child in node.children:
                    if child.type == 'if_statement':
                        complexity += 1
                        break
            
            # Recurse through children
            for child in node.children:
                walk_node(child, node.type)
        
        walk_node(ast_node)
        return complexity
    
    def calculate_cognitive_complexity(self, ast_node: Any) -> int:
        """Calculate cognitive complexity for an AST node.
        
        Cognitive complexity adds extra weight for:
        - Nesting (multiplies increment by nesting level)
        - Recursion
        - Break in loops
        - Continue statements
        """
        if not ast_node:
            return 0
        
        cognitive = 0
        
        def walk_node(node, nesting_level=0):
            nonlocal cognitive
            
            increment = 0
            increases_nesting = False
            
            # Control flow that increases nesting
            if node.type in ['if_statement', 'for_statement', 'while_statement', 
                              'do_statement', 'enhanced_for_statement']:
                increment = 1 + nesting_level
                increases_nesting = True
            
            # Switch statement
            elif node.type == 'switch_statement':
                increment = 1 + nesting_level
                increases_nesting = True
            
            # Catch clause
            elif node.type == 'catch_clause':
                increment = 1 + nesting_level
                increases_nesting = True
            
            # Ternary operator
            elif node.type == 'ternary_expression':
                increment = 1 + nesting_level
            
            # Logical operators
            elif node.type == 'binary_expression':
                operator = self._get_binary_operator(node)
                if operator in ['&&', '||']:
                    increment = 1
            
            # Break and continue add cognitive load
            elif node.type in ['break_statement', 'continue_statement']:
                increment = 1
            
            # Lambda expressions
            elif node.type == 'lambda_expression':
                increment = 1 + nesting_level
                increases_nesting = True
            
            cognitive += increment
            
            # Recurse through children
            new_nesting = nesting_level + 1 if increases_nesting else nesting_level
            for child in node.children:
                walk_node(child, new_nesting)
        
        walk_node(ast_node)
        return cognitive
    
    def _calculate_summary(self, functions: List[FunctionComplexity], 
                           classes: List[ClassComplexity], code: str) -> ComplexitySummary:
        """Calculate overall summary metrics."""
        # Count lines
        lines = code.split('\n')
        total_lines = len(lines)
        
        # Count comments
        comment_lines = self._count_comment_lines(code)
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        
        # Calculate complexity metrics
        total_complexity = sum(f.complexity for f in functions)
        total_cognitive = sum(f.cognitive_complexity for f in functions)
        avg_complexity = total_complexity / len(functions) if functions else 0
        
        # Calculate maintainability index
        mi = self.calculate_maintainability_index(
            total_complexity,
            total_lines,
            comment_ratio
        )
        
        grade = ComplexityGrade.from_maintainability_index(mi)
        
        # Count all classes including nested ones
        def count_all_classes(class_list):
            count = len(class_list)
            for cls in class_list:
                if cls.nested_classes:
                    count += count_all_classes(cls.nested_classes)
            return count
        
        total_class_count = count_all_classes(classes)
        
        return ComplexitySummary(
            total_complexity=total_complexity,
            average_complexity=avg_complexity,
            cognitive_complexity=total_cognitive,
            maintainability_index=mi,
            complexity_grade=grade.value,
            function_count=len(functions),
            class_count=total_class_count,
            lines_of_code=total_lines,
            comment_ratio=comment_ratio
        )
    
    def _count_comment_lines(self, code: str) -> int:
        """Count the number of comment lines in Java code."""
        lines = code.split('\n')
        comment_count = 0
        in_block_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            # Block comment handling
            if '/*' in stripped:
                in_block_comment = True
            if in_block_comment:
                comment_count += 1
            if '*/' in stripped:
                in_block_comment = False
                continue
            
            # Single line comment
            if stripped.startswith('//'):
                comment_count += 1
        
        return comment_count
    
    def _calculate_max_depth(self, node, current_depth=0, recursion_count=0) -> int:
        """Calculate maximum nesting depth in the AST with recursion protection."""
        # Maximum recursion depth to prevent stack overflow
        MAX_RECURSION = 100
        
        if recursion_count > MAX_RECURSION:
            logger.warning(f"Maximum recursion depth {MAX_RECURSION} reached in depth calculation")
            return current_depth
        
        if not node:
            return current_depth
        
        increases_depth = node.type in [
            'if_statement', 'for_statement', 'while_statement',
            'do_statement', 'switch_statement', 'try_statement',
            'lambda_expression', 'block'
        ]
        
        new_depth = current_depth + 1 if increases_depth else current_depth
        max_child_depth = new_depth
        
        for child in node.children:
            child_depth = self._calculate_max_depth(child, new_depth, recursion_count + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _count_parameters(self, params_node) -> int:
        """Count the number of parameters in a parameter list."""
        if not params_node:
            return 0
        
        count = 0
        for child in params_node.children:
            if child.type == 'formal_parameter':
                count += 1
        return count
    
    def _find_child_by_type(self, node, child_type: str):
        """Find the first child node of a specific type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None
    
    def _get_binary_operator(self, node) -> Optional[str]:
        """Extract the operator from a binary expression."""
        for child in node.children:
            if child.type in ['&&', '||', '==', '!=', '<', '>', '<=', '>=', '+', '-', '*', '/', '%']:
                return child.type
        return None
    
    def _get_method_name(self, node) -> Optional[str]:
        """Extract method name from a method invocation."""
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8')
        return None
    
    def _create_error_result(self, file_path: Optional[str], error: str) -> ComplexityResult:
        """Create an error result."""
        return ComplexityResult(
            file_path=file_path,
            language=self.language,
            summary=ComplexitySummary(
                total_complexity=0,
                average_complexity=0.0,
                cognitive_complexity=0,
                maintainability_index=0.0,
                complexity_grade="F",
                function_count=0,
                class_count=0,
                lines_of_code=0,
            ),
            error=error
        )


# Register the analyzer
from ..base_analyzer import AnalyzerRegistry
AnalyzerRegistry.register("java", JavaComplexityAnalyzer)