"""Python-specific complexity analyzer implementation.

This module provides enhanced complexity analysis for Python code using the radon library
while conforming to the unified analyzer interface. Includes cognitive complexity,
Python-specific rules, and comprehensive metrics.
"""

import ast
import logging
from pathlib import Path
from typing import Any

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

# Optional radon import for enhanced metrics
try:
    from radon.complexity import cc_rank, cc_visit
    from radon.metrics import mi_visit
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False
    logger.warning("Radon library not available. Using basic complexity analysis.")


class PythonComplexityAnalyzer(BaseComplexityAnalyzer):
    """Complexity analyzer for Python code.
    
    Uses radon library when available for accurate metrics,
    falls back to AST-based analysis if not.
    """

    def __init__(self):
        """Initialize Python analyzer."""
        super().__init__("python")
        self.metrics = ComplexityMetrics()

    async def analyze_file(self, file_path: Path) -> ComplexityResult:
        """Analyze complexity of a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            ComplexityResult with all metrics
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not str(file_path).endswith('.py'):
            raise ValueError(f"Not a Python file: {file_path}")

        try:
            code = file_path.read_text(encoding='utf-8')
            result = await self.analyze_code(code)
            result.file_path = str(file_path)
            return result
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return ComplexityResult(
                file_path=str(file_path),
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
                error=str(e)
            )

    async def analyze_code(self, code: str) -> ComplexityResult:
        """Analyze complexity of Python code string.
        
        Args:
            code: Python source code
            
        Returns:
            ComplexityResult with all metrics
        """
        if RADON_AVAILABLE:
            return self._analyze_with_radon(code)
        else:
            return self._analyze_with_ast(code)

    def _analyze_with_radon(self, code: str) -> ComplexityResult:
        """Analyze using radon library for accurate metrics."""
        try:
            # Get cyclomatic complexity
            cc_results = cc_visit(code)

            # Get maintainability index
            mi_score = mi_visit(code, multi=False)

            # Also parse with AST to get Python-specific features
            tree = ast.parse(code)
            feature_extractor = FeatureExtractor()
            feature_extractor.visit(tree)

            # Process results
            functions = []
            total_complexity = 0
            total_cognitive = 0

            for item in cc_results:
                complexity = item.complexity
                total_complexity += complexity

                # Estimate cognitive complexity (radon doesn't provide this)
                cognitive = self._estimate_cognitive_complexity(code, item)
                total_cognitive += cognitive

                # Get Python-specific features for this function
                features = feature_extractor.function_features.get(item.name, {})

                func = FunctionComplexity(
                    name=item.name,
                    complexity=complexity,
                    cognitive_complexity=cognitive,
                    rank=cc_rank(complexity),
                    line_start=item.lineno,
                    line_end=item.endline,
                    classification=ComplexityClassification.from_complexity(complexity).value,
                    parameters=len(getattr(item, 'args', [])) if hasattr(item, 'args') else 0,
                    depth=self._calculate_nesting_depth(code, item),
                    type=features.get('type', item.__class__.__name__.lower()),
                    # Python-specific features
                    has_decorators=features.get('has_decorators', False),
                    decorator_count=features.get('decorator_count', 0),
                    uses_context_managers=features.get('uses_context_managers', False),
                    exception_handlers_count=features.get('exception_handlers_count', 0),
                    lambda_count=features.get('lambda_count', 0),
                    is_generator=features.get('is_generator', False),
                    uses_walrus_operator=features.get('uses_walrus_operator', False),
                    uses_pattern_matching=features.get('uses_pattern_matching', False),
                    has_type_annotations=features.get('has_type_annotations', False),
                    is_recursive=features.get('is_recursive', False),
                )
                functions.append(func)

            # Sort by line number to maintain source order
            functions.sort(key=lambda x: x.line_start)

            # Calculate summary
            lines = code.split('\n')
            comment_ratio = self.metrics.estimate_comment_ratio(lines, '#', '"""', '"""')

            summary = ComplexitySummary(
                total_complexity=total_complexity,
                average_complexity=total_complexity / len(functions) if functions else 0,
                cognitive_complexity=total_cognitive,
                maintainability_index=mi_score,
                complexity_grade=ComplexityGrade.from_maintainability_index(mi_score).value,
                function_count=len(functions),
                class_count=self._count_classes(code),
                lines_of_code=len(lines),
                comment_ratio=comment_ratio,
            )

            # Create result
            result = ComplexityResult(
                file_path=None,
                language=self.language,
                summary=summary,
                functions=functions,
                classes=self._extract_classes(code, functions),
            )

            # Generate recommendations
            result.recommendations = result.generate_recommendations()

            return result

        except SyntaxError as e:
            return ComplexityResult(
                file_path=None,
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
                error=f"Syntax error: {e}"
            )

    def _analyze_with_ast(self, code: str) -> ComplexityResult:
        """Fallback analysis using Python AST when radon is not available."""
        try:
            tree = ast.parse(code)

            # Visit all nodes to calculate complexity
            visitor = ComplexityVisitor()
            visitor.visit(tree)

            # Build function complexity list
            functions = []
            for func_name, func_data in visitor.functions.items():
                func = FunctionComplexity(
                    name=func_name,
                    complexity=func_data['complexity'],
                    cognitive_complexity=func_data['cognitive'],
                    rank=ComplexityGrade.from_complexity(func_data['complexity']).value,
                    line_start=func_data['line_start'],
                    line_end=func_data['line_end'],
                    classification=ComplexityClassification.from_complexity(
                        func_data['complexity']
                    ).value,
                    parameters=func_data['parameters'],
                    depth=func_data['depth'],
                    type=func_data['type'],
                    # Python-specific features
                    has_decorators=func_data.get('has_decorators', False),
                    decorator_count=func_data.get('decorator_count', 0),
                    uses_context_managers=func_data.get('uses_context_managers', False),
                    exception_handlers_count=func_data.get('exception_handlers_count', 0),
                    lambda_count=func_data.get('lambda_count', 0),
                    is_generator=func_data.get('is_generator', False),
                    uses_walrus_operator=func_data.get('uses_walrus_operator', False),
                    uses_pattern_matching=func_data.get('uses_pattern_matching', False),
                    has_type_annotations=func_data.get('has_type_annotations', False),
                    is_recursive=func_data.get('is_recursive', False),
                )
                functions.append(func)

            # Calculate summary
            lines = code.split('\n')
            total_complexity = sum(f.complexity for f in functions)
            total_cognitive = sum(f.cognitive_complexity for f in functions)

            mi = self.calculate_maintainability_index(
                total_complexity,
                len(lines),
                self.metrics.estimate_comment_ratio(lines)
            )

            summary = ComplexitySummary(
                total_complexity=total_complexity,
                average_complexity=total_complexity / len(functions) if functions else 0,
                cognitive_complexity=total_cognitive,
                maintainability_index=mi,
                complexity_grade=ComplexityGrade.from_maintainability_index(mi).value,
                function_count=len(functions),
                class_count=len(visitor.classes),
                lines_of_code=len(lines),
            )

            result = ComplexityResult(
                file_path=None,
                language=self.language,
                summary=summary,
                functions=functions,
            )

            result.recommendations = result.generate_recommendations()
            return result

        except SyntaxError as e:
            return ComplexityResult(
                file_path=None,
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
                error=f"Syntax error: {e}"
            )

    def calculate_cyclomatic_complexity(self, ast_node: Any) -> int:
        """Calculate cyclomatic complexity for a Python AST node."""
        if isinstance(ast_node, ast.FunctionDef):
            complexity = 1  # Base complexity

            for node in ast.walk(ast_node):
                # Decision points
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(node, ast.Assert):
                    complexity += 1
                elif isinstance(node, ast.With):
                    complexity += 1
                # Boolean operators
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1

            return complexity
        return 1

    def calculate_cognitive_complexity(self, ast_node: Any) -> int:
        """Calculate cognitive complexity for a Python AST node."""
        return self._calculate_cognitive_recursive(ast_node, 0)

    def _calculate_cognitive_recursive(self, node: Any, nesting: int) -> int:
        """Recursively calculate cognitive complexity with proper nesting."""
        complexity = 0
        local_nesting = nesting

        # Control flow structures that increase nesting
        if isinstance(node, (ast.If, ast.While, ast.For)):
            # Base increment + nesting penalty
            complexity += 1 + nesting
            local_nesting = nesting + 1

            # Process the body with increased nesting
            for child in node.body:
                complexity += self._calculate_cognitive_recursive(child, local_nesting)

            # Process else/elif branches
            if hasattr(node, 'orelse') and node.orelse:
                # Check if it's an elif (single If node) or else
                if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                    # elif - continue at same nesting level, no extra +1
                    complexity += self._calculate_cognitive_recursive(node.orelse[0], nesting)
                else:
                    # else clause - only add +1 if it contains complex logic
                    has_complex_logic = any(
                        isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With))
                        for child in node.orelse
                    )
                    if has_complex_logic:
                        complexity += 1

                    for child in node.orelse:
                        complexity += self._calculate_cognitive_recursive(child, local_nesting)

            # Don't process children again
            return complexity

        elif isinstance(node, ast.ExceptHandler):
            complexity += 1 + nesting
            local_nesting = nesting + 1

        elif isinstance(node, ast.BoolOp):
            # Each additional logical operator adds complexity
            complexity += len(node.values) - 1

        elif isinstance(node, ast.With):
            # Context managers add complexity
            complexity += 1
            local_nesting = nesting + 1

        # Process child nodes with potentially updated nesting
        for child in ast.iter_child_nodes(node):
            complexity += self._calculate_cognitive_recursive(child, local_nesting)

        return complexity

    def _estimate_cognitive_complexity(self, code: str, radon_item: Any) -> int:
        """Estimate cognitive complexity from radon results."""
        # Since radon doesn't provide cognitive complexity, we need to parse the function
        # and calculate it properly using AST
        try:
            tree = ast.parse(code)
            # Find the function node
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == radon_item.name and node.lineno == radon_item.lineno:
                        return self.calculate_cognitive_complexity(node)

            # Fallback to estimation based on cyclomatic complexity
            base = radon_item.complexity
            nesting_depth = self._calculate_nesting_depth(code, radon_item)

            # Apply nesting penalty (rough approximation)
            if nesting_depth > 1:
                return base + (nesting_depth - 1) * 2

            return base
        except:
            # Final fallback
            return radon_item.complexity

    def _calculate_nesting_depth(self, code: str, item: Any) -> int:
        """Calculate maximum nesting depth for a function."""
        try:
            # Extract function code
            lines = code.split('\n')
            func_lines = lines[item.lineno - 1:item.endline]

            # Simple indentation-based nesting calculation
            max_indent = 0
            base_indent = len(func_lines[0]) - len(func_lines[0].lstrip()) if func_lines else 0

            for line in func_lines[1:]:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    relative_indent = (indent - base_indent) // 4  # Assume 4-space indents
                    max_indent = max(max_indent, relative_indent)

            return max_indent
        except:
            return 0

    def _count_classes(self, code: str) -> int:
        """Count number of classes in the code."""
        try:
            tree = ast.parse(code)
            return len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        except:
            return 0

    def _extract_classes(self, code: str, functions: list[FunctionComplexity]) -> list[ClassComplexity]:
        """Extract class information from the code."""

        try:
            tree = ast.parse(code)

            # Build hierarchy of classes
            def extract_class_hierarchy(node, depth=0):
                """Recursively extract classes and their nested classes."""
                result = []

                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.ClassDef):
                        # Find methods belonging to this class (direct children only)
                        class_methods = []
                        for item in child.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                # Find matching function in our functions list
                                for f in functions:
                                    if f.name == item.name and f.line_start == item.lineno:
                                        class_methods.append(f)
                                        break

                        # Recursively find nested classes
                        nested_classes = extract_class_hierarchy(child, depth + 1)

                        total_complexity = sum(m.complexity for m in class_methods)

                        class_comp = ClassComplexity(
                            name=child.name,
                            total_complexity=total_complexity,
                            average_method_complexity=(
                                total_complexity / len(class_methods) if class_methods else 0
                            ),
                            method_count=len(class_methods),
                            line_start=child.lineno,
                            line_end=child.end_lineno if hasattr(child, 'end_lineno') else child.lineno,
                            methods=class_methods,
                            nested_classes=nested_classes,
                        )
                        result.append(class_comp)
                    else:
                        # Continue looking for classes in nested nodes (but not the whole tree)
                        if depth == 0:  # Only at top level
                            result.extend(extract_class_hierarchy(child, depth))

                return result

            return extract_class_hierarchy(tree)
        except Exception as e:
            logger.warning(f"Failed to extract classes: {e}")
            return []


class FeatureExtractor(ast.NodeVisitor):
    """Lightweight AST visitor just for extracting Python-specific features."""

    def __init__(self):
        self.function_features = {}
        self.current_function = None
        self.current_class = None

    def visit_FunctionDef(self, node):
        """Extract features from regular functions."""
        self._extract_function_features(node, "function")

    def visit_AsyncFunctionDef(self, node):
        """Extract features from async functions."""
        func_type = "async_method" if self.current_class else "async_function"
        self._extract_function_features(node, func_type)

    def visit_ClassDef(self, node):
        """Track class context."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def _extract_function_features(self, node, base_type):
        """Extract Python-specific features from a function node."""
        old_function = self.current_function
        self.current_function = node.name

        # Determine actual type
        if self.current_class and base_type in ["function"]:
            func_type = "method"
        else:
            func_type = base_type

        features = {
            'type': func_type,
            'has_decorators': bool(node.decorator_list),
            'decorator_count': len(node.decorator_list),
            'uses_context_managers': False,
            'exception_handlers_count': 0,
            'lambda_count': 0,
            'is_generator': False,
            'uses_walrus_operator': False,
            'uses_pattern_matching': False,
            'has_type_annotations': False,
            'is_recursive': False,
        }

        # Check type annotations
        if node.args.args:
            for arg in node.args.args:
                if arg.annotation is not None:
                    features['has_type_annotations'] = True
                    break
        if not features['has_type_annotations'] and node.returns:
            features['has_type_annotations'] = True

        # Walk function body for other features
        recursion_calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.With):
                features['uses_context_managers'] = True
            elif isinstance(child, ast.ExceptHandler):
                features['exception_handlers_count'] += 1
            elif isinstance(child, ast.Lambda):
                features['lambda_count'] += 1
            # Comprehensions also count as implicit lambdas
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                features['lambda_count'] += 1
            elif isinstance(child, (ast.Yield, ast.YieldFrom)):
                features['is_generator'] = True
            elif hasattr(ast, 'NamedExpr') and isinstance(child, ast.NamedExpr):
                features['uses_walrus_operator'] = True
            elif hasattr(ast, 'Match') and isinstance(child, ast.Match):
                features['uses_pattern_matching'] = True
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id == node.name:
                    features['is_recursive'] = True

        self.function_features[node.name] = features
        self.generic_visit(node)
        self.current_function = old_function


class ComplexityVisitor(ast.NodeVisitor):
    """Enhanced AST visitor for calculating complexity metrics with Python-specific features."""

    def __init__(self):
        self.functions = {}
        self.classes = []
        self.current_class = None
        self.current_function = None
        self.nesting_level = 0
        self.recursion_candidates = set()  # Track function calls for recursion detection

    def visit_FunctionDef(self, node):
        """Visit function definition nodes with enhanced feature detection."""
        self.nesting_level += 1
        old_function = self.current_function
        self.current_function = node.name

        # Calculate complexity
        complexity = self._calculate_complexity(node)
        cognitive = self._calculate_cognitive(node, 0)

        func_type = "method" if self.current_class else "function"

        # Detect Python-specific features
        features = self._detect_python_features(node)

        self.functions[node.name] = {
            'complexity': complexity,
            'cognitive': cognitive,
            'line_start': node.lineno,
            'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            'parameters': len(node.args.args),
            'depth': self.nesting_level - 1,
            'type': func_type,
            **features  # Add all detected features
        }

        # Check for recursion
        self.recursion_candidates = set()
        self.generic_visit(node)
        if node.name in self.recursion_candidates:
            self.functions[node.name]['is_recursive'] = True

        self.nesting_level -= 1
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition nodes."""
        self.nesting_level += 1
        old_function = self.current_function
        self.current_function = node.name

        # Calculate complexity (+1 for async)
        complexity = self._calculate_complexity(node) + 1
        cognitive = self._calculate_cognitive(node, 0) + 1

        func_type = "async_method" if self.current_class else "async_function"

        # Detect Python-specific features
        features = self._detect_python_features(node)

        self.functions[node.name] = {
            'complexity': complexity,
            'cognitive': cognitive,
            'line_start': node.lineno,
            'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            'parameters': len(node.args.args),
            'depth': self.nesting_level - 1,
            'type': func_type,
            **features
        }

        # Check for recursion
        self.recursion_candidates = set()
        self.generic_visit(node)
        if node.name in self.recursion_candidates:
            self.functions[node.name]['is_recursive'] = True

        self.nesting_level -= 1
        self.current_function = old_function

    def visit_ClassDef(self, node):
        """Visit class definition nodes."""
        self.classes.append(node.name)
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def _calculate_complexity(self, node):
        """Calculate cyclomatic complexity for a function node with Python-specific rules."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.With):
                # Context managers add complexity
                complexity += 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # Comprehensions add complexity
                complexity += 1
                # Add extra for conditions in comprehensions
                for generator in child.generators:
                    complexity += len(generator.ifs)
            elif isinstance(child, ast.Lambda):
                # Lambda expressions add complexity
                complexity += 1

        return complexity

    def _calculate_cognitive(self, node, nesting):
        """Calculate cognitive complexity for a function node."""
        complexity = 0

        for child in node.body if hasattr(node, 'body') else ast.iter_child_nodes(node):
            complexity += self._calculate_cognitive_node(child, nesting)

        return complexity

    def _calculate_cognitive_node(self, node, nesting, is_elif=False):
        """Calculate cognitive complexity for a single node."""
        complexity = 0

        if isinstance(node, ast.If):
            # If statement: +1 + nesting (but elif is just +1 without nesting increment)
            if is_elif:
                complexity += 1  # elif adds +1 but stays at same nesting level
            else:
                complexity += 1 + nesting
            # Process if body
            for child in node.body:
                complexity += self._calculate_cognitive_node(child, nesting + 1)
            # Process else/elif
            if node.orelse:
                if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                    # elif case - process as elif at same nesting level
                    # Pass is_elif=True to avoid nesting penalty
                    complexity += self._calculate_cognitive_node(node.orelse[0], nesting, is_elif=True)
                else:
                    # else case - only add complexity if there's actual logic in the else block
                    # Simple returns or single statements don't add cognitive complexity
                    has_complex_logic = False
                    for child in node.orelse:
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                            has_complex_logic = True
                            break

                    if has_complex_logic:
                        complexity += 1

                    for child in node.orelse:
                        complexity += self._calculate_cognitive_node(child, nesting + 1)

        elif isinstance(node, (ast.While, ast.For)):
            complexity += 1 + nesting
            for child in node.body:
                complexity += self._calculate_cognitive_node(child, nesting + 1)
            if hasattr(node, 'orelse') and node.orelse:
                for child in node.orelse:
                    complexity += self._calculate_cognitive_node(child, nesting + 1)

        elif isinstance(node, ast.ExceptHandler):
            complexity += 1 + nesting
            for child in node.body:
                complexity += self._calculate_cognitive_node(child, nesting + 1)

        elif isinstance(node, ast.With):
            complexity += 1
            for child in node.body:
                complexity += self._calculate_cognitive_node(child, nesting + 1)

        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1

        else:
            # Process children without increasing nesting
            for child in ast.iter_child_nodes(node):
                complexity += self._calculate_cognitive_node(child, nesting)

        return complexity

    def visit_Call(self, node):
        """Track function calls for recursion detection."""
        if self.current_function and isinstance(node.func, ast.Name):
            if node.func.id == self.current_function:
                self.recursion_candidates.add(self.current_function)
        self.generic_visit(node)

    def _detect_python_features(self, node):
        """Detect Python-specific features in a function/method."""
        features = {
            'has_decorators': False,
            'decorator_count': 0,
            'uses_context_managers': False,
            'exception_handlers_count': 0,
            'lambda_count': 0,
            'is_generator': False,
            'uses_walrus_operator': False,
            'uses_pattern_matching': False,
            'has_type_annotations': False,
            'is_recursive': False,
        }

        # Check decorators
        if hasattr(node, 'decorator_list') and node.decorator_list:
            features['has_decorators'] = True
            features['decorator_count'] = len(node.decorator_list)

        # Check type annotations
        if node.args.args:
            for arg in node.args.args:
                if arg.annotation is not None:
                    features['has_type_annotations'] = True
                    break
        if not features['has_type_annotations'] and hasattr(node, 'returns') and node.returns:
            features['has_type_annotations'] = True

        # Walk the function body to detect other features
        for child in ast.walk(node):
            # Context managers (with statements)
            if isinstance(child, ast.With):
                features['uses_context_managers'] = True

            # Exception handlers
            elif isinstance(child, ast.ExceptHandler):
                features['exception_handlers_count'] += 1

            # Lambda expressions
            elif isinstance(child, ast.Lambda):
                features['lambda_count'] += 1

            # Comprehensions also count as implicit lambdas
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                features['lambda_count'] += 1

            # Generators (yield statements)
            elif isinstance(child, (ast.Yield, ast.YieldFrom)):
                features['is_generator'] = True

            # Walrus operator (named expressions) - Python 3.8+
            elif hasattr(ast, 'NamedExpr') and isinstance(child, ast.NamedExpr):
                features['uses_walrus_operator'] = True

            # Pattern matching (match statements) - Python 3.10+
            elif hasattr(ast, 'Match') and isinstance(child, ast.Match):
                features['uses_pattern_matching'] = True

        return features


# Register the analyzer
from ..base_analyzer import AnalyzerRegistry

AnalyzerRegistry.register("python", PythonComplexityAnalyzer)
