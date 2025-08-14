"""Kotlin-specific complexity analyzer implementation.

This module provides complexity analysis for Kotlin code, supporting
Kotlin-specific constructs like data classes, when expressions, extension
functions, lambdas, coroutines, and sealed classes.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


class KotlinComplexityAnalyzer(BaseComplexityAnalyzer):
    """Complexity analyzer for Kotlin code.
    
    Implements Kotlin-specific complexity rules including:
    - Data classes: Lower base complexity
    - Extension functions: Standard function complexity
    - When expressions: Complexity = number of branches + 1
    - Lambda expressions: +1 complexity per nested lambda
    - Coroutines: +2 complexity for suspend functions
    - Sealed classes: +1 per subclass
    """
    
    def __init__(self):
        """Initialize Kotlin analyzer."""
        super().__init__("kotlin")
        self.metrics = ComplexityMetrics()
        
        # Kotlin-specific patterns
        self.patterns = {
            'function': re.compile(
                r'(?:(?:public|private|protected|internal|suspend|inline|infix|operator|override|open|final)\s+)*'
                r'fun\s+(?:<[^>]+>\s+)?(?:([^.(\s]+)\.)?([^(\s]+)\s*\([^)]*\)[^{]*\{',
                re.MULTILINE
            ),
            'class': re.compile(
                r'(?:(?:public|private|protected|internal|abstract|open|final|sealed|data|enum|annotation)\s+)*'
                r'class\s+([^(<\s]+)',
                re.MULTILINE
            ),
            'when': re.compile(r'\bwhen\s*(?:\([^)]*\))?\s*\{', re.MULTILINE),
            'when_branch': re.compile(r'->', re.MULTILINE),
            'if': re.compile(r'\bif\s*\([^)]+\)', re.MULTILINE),
            'else_if': re.compile(r'\belse\s+if\s*\([^)]+\)', re.MULTILINE),
            'for': re.compile(r'\bfor\s*\([^)]+\)', re.MULTILINE),
            'while': re.compile(r'\bwhile\s*\([^)]+\)', re.MULTILINE),
            'do_while': re.compile(r'\bdo\s*\{', re.MULTILINE),
            'lambda': re.compile(r'\.\w+\s*\{|\{\s*\w+\s*->|\{\s*it\b', re.MULTILINE),
            'suspend': re.compile(r'\bsuspend\s+fun\s+', re.MULTILINE),
            'data_class': re.compile(r'\bdata\s+class\s+([^(<\s]+)', re.MULTILINE),
            'sealed_class': re.compile(r'\bsealed\s+class\s+([^(<\s]+)', re.MULTILINE),
            'extension': re.compile(r'fun\s+([^.(\s]+)\.([^(\s]+)\s*\(', re.MULTILINE),
            'comment_single': re.compile(r'//.*$', re.MULTILINE),
            'comment_multi': re.compile(r'/\*.*?\*/', re.DOTALL),
            'comment_doc': re.compile(r'/\*\*.*?\*/', re.DOTALL),
        }
    
    async def analyze_file(self, file_path: Path) -> ComplexityResult:
        """Analyze complexity of a Kotlin file.
        
        Args:
            file_path: Path to the Kotlin file
            
        Returns:
            ComplexityResult with all metrics
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not str(file_path).endswith(('.kt', '.kts')):
            raise ValueError(f"Not a Kotlin file: {file_path}")
        
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
        """Analyze complexity of Kotlin code string.
        
        Args:
            code: Kotlin source code
            
        Returns:
            ComplexityResult with all metrics
        """
        try:
            # Basic syntax validation
            if not self._validate_kotlin_syntax(code):
                raise SyntaxError("Invalid Kotlin syntax detected")
            
            # Extract functions and calculate their complexity
            functions = self._extract_functions(code)
            
            # Extract classes and their complexity
            classes = self._extract_classes(code)
            
            # Calculate summary metrics
            lines = code.split('\n')
            total_complexity = sum(f.complexity for f in functions)
            total_cognitive = sum(f.cognitive_complexity for f in functions)
            
            # Add class method complexities
            for cls in classes:
                total_complexity += cls.total_complexity
                # Add methods from classes to function list for complete view
                functions.extend(cls.methods)
            
            # Calculate comment ratio
            comment_ratio = self._calculate_comment_ratio(code)
            
            # Calculate maintainability index
            mi = self.calculate_maintainability_index(
                total_complexity,
                len(lines),
                comment_ratio
            )
            
            # Sort functions by complexity
            functions.sort(key=lambda x: x.complexity, reverse=True)
            
            summary = ComplexitySummary(
                total_complexity=total_complexity,
                average_complexity=total_complexity / len(functions) if functions else 0,
                cognitive_complexity=total_cognitive,
                maintainability_index=mi,
                complexity_grade=ComplexityGrade.from_maintainability_index(mi).value,
                function_count=len(functions),
                class_count=len(classes),
                lines_of_code=len(lines),
                comment_ratio=comment_ratio,
            )
            
            result = ComplexityResult(
                file_path=None,
                language=self.language,
                summary=summary,
                functions=functions,
                classes=classes,
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
        except Exception as e:
            logger.error(f"Failed to analyze Kotlin code: {e}")
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
                error=str(e)
            )
    
    def calculate_cyclomatic_complexity(self, ast_node: Any) -> int:
        """Calculate cyclomatic complexity for a Kotlin code block.
        
        For Kotlin, we use the code string directly since we don't have
        a full AST parser. The ast_node parameter is the code string.
        
        Args:
            ast_node: Code string to analyze
            
        Returns:
            Cyclomatic complexity score
        """
        if not isinstance(ast_node, str):
            return 1
        
        code = ast_node
        complexity = 1  # Base complexity
        
        # Count decision points
        complexity += len(self.patterns['if'].findall(code))
        complexity += len(self.patterns['else_if'].findall(code))
        complexity += len(self.patterns['for'].findall(code))
        complexity += len(self.patterns['while'].findall(code))
        complexity += len(self.patterns['do_while'].findall(code))
        
        # Count when expressions - each branch adds to complexity
        when_matches = list(self.patterns['when'].finditer(code))
        for when_match in when_matches:
            # Extract the when block
            when_start = when_match.start()
            when_block = self._extract_block(code[when_start:])
            if when_block:
                # Count arrow operators (branches) in this specific when block
                branches = len(self.patterns['when_branch'].findall(when_block))
                # Each when adds branches to complexity
                complexity += branches
        
        # Count logical operators
        complexity += self.metrics.count_logical_operators(code)
        
        return complexity
    
    def calculate_cognitive_complexity(self, ast_node: Any) -> int:
        """Calculate cognitive complexity for Kotlin code.
        
        Cognitive complexity takes into account nesting and mental effort
        required to understand the code.
        
        Args:
            ast_node: Code string to analyze
            
        Returns:
            Cognitive complexity score
        """
        if not isinstance(ast_node, str):
            return 0
        
        code = ast_node
        cognitive = 0
        nesting_level = 0
        
        # Simplified cognitive complexity calculation
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Track nesting
            nesting_level += line.count('{') - line.count('}')
            nesting_level = max(0, nesting_level)
            
            # Add complexity for control structures with nesting penalty
            if any(pattern in line for pattern in ['if ', 'else if ', 'when ', 'for ', 'while ']):
                cognitive += self.metrics.calculate_cognitive_increment(nesting_level)
            
            # Lambda expressions add cognitive load
            if '->' in line and '{' in line:
                cognitive += self.metrics.calculate_cognitive_increment(nesting_level, is_nested_flow=True)
        
        return cognitive
    
    def _validate_kotlin_syntax(self, code: str) -> bool:
        """Basic validation of Kotlin syntax.
        
        This is a simplified check - a full parser would be better.
        
        Args:
            code: Kotlin source code
            
        Returns:
            True if syntax appears valid
        """
        # Check for basic syntax errors
        if code.strip() == "":
            return True  # Empty code is valid
        
        # Check for unmatched braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            return False
        
        # Check for unmatched parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            return False
        
        # Check for malformed function declarations
        func_pattern = re.compile(r'\bfun\s+\w+\s*(?:\([^)]*\))?\s*(?::\s*\w+)?\s*\{')
        malformed_func = re.compile(r'\bfun\s+\w+\s*\{')  # Function without parameters
        
        if malformed_func.search(code):
            # Found a function without proper parameter list
            return False
        
        for match in self.patterns['function'].finditer(code):
            # Basic check that function has a body or is abstract
            func_start = match.start()
            func_text = code[func_start:func_start + 200]  # Look ahead
            if '{' not in func_text and '=' not in func_text and 'abstract' not in func_text:
                return False
        
        return True
    
    def _extract_functions(self, code: str) -> List[FunctionComplexity]:
        """Extract functions and their complexity from Kotlin code.
        
        Args:
            code: Kotlin source code
            
        Returns:
            List of FunctionComplexity objects
        """
        functions = []
        
        # Remove comments for cleaner parsing
        clean_code = self._remove_comments(code)
        
        # Find all function declarations
        for match in self.patterns['function'].finditer(clean_code):
            receiver = match.group(1)  # Extension function receiver
            func_name = match.group(2)
            
            # Handle extension functions
            if receiver:
                func_name = f"{receiver}.{func_name}"
            
            # Extract function body
            func_start = match.start()
            func_body = self._extract_block(clean_code[func_start:])
            
            if not func_body:
                continue
            
            # Calculate line numbers
            line_start = clean_code[:func_start].count('\n') + 1
            line_end = line_start + func_body.count('\n')
            
            # Check if it's a suspend function
            # Look at the matched text itself since suspend is in the optional modifiers
            match_text = clean_code[func_start:func_start + 100]
            is_suspend = match_text.startswith('suspend ') or ' suspend ' in clean_code[max(0, func_start - 50):func_start]
            
            # Calculate complexity
            cyclomatic = self.calculate_cyclomatic_complexity(func_body)
            
            # Add complexity for suspend functions
            if is_suspend:
                cyclomatic += 2
            
            # Count lambdas and add complexity (exclude when expressions)
            # Remove when blocks from func_body before counting lambdas
            temp_body = func_body
            for when_match in self.patterns['when'].finditer(func_body):
                when_start = when_match.start()
                when_block = self._extract_block(func_body[when_start:])
                if when_block:
                    temp_body = temp_body.replace(when_block, '')
            
            lambda_count = len(self.patterns['lambda'].findall(temp_body))
            cyclomatic += lambda_count
            
            # Calculate cognitive complexity
            cognitive = self.calculate_cognitive_complexity(func_body)
            if is_suspend:
                cognitive += 2
            
            # Count parameters
            param_match = re.search(r'\([^)]*\)', clean_code[func_start:])
            params = 0
            if param_match:
                param_str = param_match.group(0)
                if param_str != "()":
                    # Count commas + 1 for parameters
                    params = param_str.count(',') + 1
            
            # Calculate nesting depth
            depth = self._calculate_max_nesting(func_body)
            
            func_complexity = FunctionComplexity(
                name=func_name,
                complexity=cyclomatic,
                cognitive_complexity=cognitive,
                rank=ComplexityGrade.from_complexity(cyclomatic).value,
                line_start=line_start,
                line_end=line_end,
                classification=ComplexityClassification.from_complexity(cyclomatic).value,
                parameters=params,
                depth=depth,
                type="suspend_function" if is_suspend else "function",
            )
            
            functions.append(func_complexity)
        
        return functions
    
    def _extract_classes(self, code: str) -> List[ClassComplexity]:
        """Extract classes and their complexity from Kotlin code.
        
        Args:
            code: Kotlin source code
            
        Returns:
            List of ClassComplexity objects
        """
        classes = []
        clean_code = self._remove_comments(code)
        
        # Find all class declarations
        for match in self.patterns['class'].finditer(clean_code):
            class_name = match.group(1)
            class_start = match.start()
            
            # Check if it's a data class
            is_data_class = 'data class' in clean_code[max(0, class_start - 20):class_start + 20]
            
            # Check if it's a sealed class
            is_sealed = 'sealed class' in clean_code[max(0, class_start - 20):class_start + 20]
            
            # Extract class body
            class_body = self._extract_block(clean_code[class_start:])
            if not class_body:
                continue
            
            # Calculate line numbers
            line_start = clean_code[:class_start].count('\n') + 1
            line_end = line_start + class_body.count('\n')
            
            # Extract methods within the class
            methods = []
            method_total_complexity = 0
            
            # Find functions within class body
            for func_match in self.patterns['function'].finditer(class_body):
                method_name = func_match.group(2)
                method_start = func_match.start()
                method_body = self._extract_block(class_body[method_start:])
                
                if method_body:
                    method_complexity = self.calculate_cyclomatic_complexity(method_body)
                    method_cognitive = self.calculate_cognitive_complexity(method_body)
                    
                    method = FunctionComplexity(
                        name=method_name,
                        complexity=method_complexity,
                        cognitive_complexity=method_cognitive,
                        rank=ComplexityGrade.from_complexity(method_complexity).value,
                        line_start=line_start + class_body[:method_start].count('\n'),
                        line_end=line_start + class_body[:method_start].count('\n') + method_body.count('\n'),
                        classification=ComplexityClassification.from_complexity(method_complexity).value,
                        parameters=0,  # Simplified
                        depth=self._calculate_max_nesting(method_body),
                        type="method",
                    )
                    methods.append(method)
                    method_total_complexity += method_complexity
            
            # Base class complexity
            class_complexity = method_total_complexity
            
            # Data classes have lower complexity
            if is_data_class and not methods:
                class_complexity = 1
            elif is_data_class:
                class_complexity = max(1, class_complexity)
            
            # Sealed classes add complexity per subclass
            if is_sealed:
                # Count subclasses within the sealed class body
                # Look for: data class, class, or object declarations
                subclass_patterns = [
                    r'\bdata\s+class\s+\w+',
                    r'\bclass\s+\w+.*:\s*' + class_name,
                    r'\bobject\s+\w+\s*:\s*' + class_name,
                ]
                subclass_count = 0
                for pattern in subclass_patterns:
                    subclass_count += len(re.findall(pattern, class_body))
                # Add 1 per subclass to complexity
                class_complexity += max(1, subclass_count)
            
            cls = ClassComplexity(
                name=class_name,
                total_complexity=class_complexity,
                average_method_complexity=method_total_complexity / len(methods) if methods else 0,
                method_count=len(methods),
                line_start=line_start,
                line_end=line_end,
                methods=methods,
            )
            
            classes.append(cls)
        
        return classes
    
    def _extract_block(self, code: str) -> str:
        """Extract a code block starting from the current position.
        
        Finds matching braces to extract a complete block.
        
        Args:
            code: Code starting at block beginning
            
        Returns:
            The complete block including braces
        """
        if not code:
            return ""
        
        # Find the opening brace
        brace_pos = code.find('{')
        if brace_pos == -1:
            # Might be an expression body (single line)
            equals_pos = code.find('=')
            if equals_pos != -1:
                # Find the end of the expression
                newline_pos = code.find('\n', equals_pos)
                if newline_pos != -1:
                    return code[:newline_pos]
            return ""
        
        # Count braces to find matching close brace
        brace_count = 0
        i = brace_pos
        
        while i < len(code):
            if code[i] == '{':
                brace_count += 1
            elif code[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return code[:i + 1]
            i += 1
        
        return ""
    
    def _calculate_max_nesting(self, code: str, max_allowed_depth: int = 100) -> int:
        """Calculate maximum nesting depth in code with depth protection.
        
        Args:
            code: Code to analyze
            max_allowed_depth: Maximum allowed nesting depth to prevent issues
            
        Returns:
            Maximum nesting depth (capped at max_allowed_depth)
        """
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char == '{':
                current_depth += 1
                if current_depth > max_allowed_depth:
                    logger.warning(f"Maximum nesting depth {max_allowed_depth} reached")
                    return max_allowed_depth
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _remove_comments(self, code: str) -> str:
        """Remove comments from Kotlin code.
        
        Args:
            code: Original code
            
        Returns:
            Code without comments
        """
        # Remove single-line comments
        code = self.patterns['comment_single'].sub('', code)
        
        # Remove multi-line comments
        code = self.patterns['comment_multi'].sub('', code)
        
        # Remove documentation comments
        code = self.patterns['comment_doc'].sub('', code)
        
        return code
    
    def _calculate_comment_ratio(self, code: str) -> float:
        """Calculate the ratio of comment lines to total lines.
        
        Args:
            code: Kotlin source code
            
        Returns:
            Comment ratio (0.0 to 1.0)
        """
        lines = code.split('\n')
        if not lines:
            return 0.0
        
        comment_lines = 0
        in_multiline = False
        
        for line in lines:
            line = line.strip()
            
            # Check for multi-line comment start
            if '/*' in line:
                in_multiline = True
            
            # Count as comment line
            if in_multiline or line.startswith('//'):
                comment_lines += 1
            
            # Check for multi-line comment end
            if '*/' in line:
                in_multiline = False
        
        return comment_lines / len(lines)


# Register the analyzer
from ..base_analyzer import AnalyzerRegistry
AnalyzerRegistry.register("kotlin", KotlinComplexityAnalyzer)