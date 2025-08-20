"""
Implementation Example: Enhanced Code Classification using Tree-sitter
This is a concrete implementation proposal for project-watch-mcp
"""

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Query, Node
from neo4j import AsyncDriver

# Initialize tree-sitter for Python
PY_LANGUAGE = Language(tspython.language())

@dataclass
class ClassInfo:
    """Information about a class definition"""
    name: str
    qualified_name: str
    base_classes: List[str]
    methods: List['MethodInfo']
    decorators: List[str]
    docstring: Optional[str]
    is_abstract: bool
    line_start: int
    line_end: int
    complexity: int = 0

@dataclass
class MethodInfo:
    """Information about a method/function"""
    name: str
    qualified_name: str
    parameters: List[str]
    return_type: Optional[str]
    decorators: List[str]
    docstring: Optional[str]
    is_async: bool
    is_generator: bool
    is_static: bool
    is_classmethod: bool
    line_start: int
    line_end: int
    complexity: int = 0
    calls: List[str] = field(default_factory=list)

@dataclass
class ImportInfo:
    """Information about imports"""
    module: str
    names: List[str]
    alias: Optional[str]
    is_from_import: bool
    line: int

@dataclass
class VariableInfo:
    """Information about module-level variables"""
    name: str
    type_hint: Optional[str]
    value: Optional[str]
    is_constant: bool  # ALL_CAPS naming
    line: int

@dataclass
class CodeStructure:
    """Complete structure of a Python file"""
    file_path: Path
    module_name: str
    classes: List[ClassInfo]
    functions: List[MethodInfo]
    imports: List[ImportInfo]
    variables: List[VariableInfo]
    dependencies: List[str]
    
    def to_neo4j_nodes(self) -> List[Dict[str, Any]]:
        """Convert to Neo4j node representations"""
        nodes = []
        
        # Module node
        nodes.append({
            'type': 'Module',
            'properties': {
                'path': str(self.file_path),
                'name': self.module_name,
                'language': 'python',
                'import_count': len(self.imports),
                'class_count': len(self.classes),
                'function_count': len(self.functions)
            }
        })
        
        # Class nodes
        for cls in self.classes:
            nodes.append({
                'type': 'Class',
                'properties': {
                    'name': cls.name,
                    'qualified_name': cls.qualified_name,
                    'is_abstract': cls.is_abstract,
                    'method_count': len(cls.methods),
                    'base_classes': cls.base_classes,
                    'decorators': cls.decorators,
                    'docstring': cls.docstring,
                    'line_start': cls.line_start,
                    'line_end': cls.line_end,
                    'complexity': cls.complexity
                }
            })
        
        # Function/Method nodes
        all_functions = self.functions.copy()
        for cls in self.classes:
            all_functions.extend(cls.methods)
        
        for func in all_functions:
            nodes.append({
                'type': 'Function',
                'properties': {
                    'name': func.name,
                    'qualified_name': func.qualified_name,
                    'parameters': func.parameters,
                    'return_type': func.return_type,
                    'is_async': func.is_async,
                    'is_generator': func.is_generator,
                    'decorators': func.decorators,
                    'docstring': func.docstring,
                    'line_start': func.line_start,
                    'line_end': func.line_end,
                    'complexity': func.complexity,
                    'calls': func.calls
                }
            })
        
        return nodes


class EnhancedPythonAnalyzer:
    """Enhanced Python code analyzer using tree-sitter"""
    
    def __init__(self):
        self.parser = Parser()
        self.parser.language = PY_LANGUAGE
        self._setup_queries()
    
    def _setup_queries(self):
        """Setup tree-sitter queries for various code elements"""
        
        # Query for classes with inheritance
        self.class_query = Query(PY_LANGUAGE, """
            (class_definition
              name: (identifier) @class.name
              superclasses: (argument_list)? @class.bases
              body: (block) @class.body) @class
        """)
        
        # Query for functions and methods
        self.function_query = Query(PY_LANGUAGE, """
            (function_definition
              name: (identifier) @func.name
              parameters: (parameters) @func.params
              return_type: (type)? @func.return
              body: (block) @func.body) @function
        """)
        
        # Query for imports
        self.import_query = Query(PY_LANGUAGE, """
            [
              (import_statement
                name: (dotted_name) @import.module)
              (import_from_statement
                module_name: (dotted_name)? @import.module
                name: (dotted_name)? @import.name)
            ] @import
        """)
        
        # Query for decorators
        self.decorator_query = Query(PY_LANGUAGE, """
            (decorator
              (identifier) @decorator.name)
        """)
    
    def analyze_file(self, file_path: Path) -> CodeStructure:
        """Analyze a Python file and extract its structure"""
        
        with open(file_path, 'rb') as f:
            source_code = f.read()
        
        tree = self.parser.parse(source_code)
        
        # Extract all components
        classes = self._extract_classes(tree.root_node, source_code)
        functions = self._extract_functions(tree.root_node, source_code, in_class=False)
        imports = self._extract_imports(tree.root_node, source_code)
        variables = self._extract_variables(tree.root_node, source_code)
        dependencies = self._extract_dependencies(imports)
        
        # Build module name from path
        module_name = file_path.stem
        if file_path.parent.name != '':
            module_name = f"{file_path.parent.name}.{module_name}"
        
        return CodeStructure(
            file_path=file_path,
            module_name=module_name,
            classes=classes,
            functions=functions,
            imports=imports,
            variables=variables,
            dependencies=dependencies
        )
    
    def _extract_classes(self, root_node: Node, source: bytes) -> List[ClassInfo]:
        """Extract class definitions from AST"""
        classes = []
        
        def traverse(node: Node):
            if node.type == 'class_definition':
                name_node = node.child_by_field_name('name')
                if not name_node:
                    return
                
                class_name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
                
                # Extract base classes
                base_classes = []
                superclasses_node = node.child_by_field_name('superclasses')
                if superclasses_node:
                    for child in superclasses_node.children:
                        if child.type == 'identifier' or child.type == 'attribute':
                            base_classes.append(
                                source[child.start_byte:child.end_byte].decode('utf-8')
                            )
                
                # Extract decorators
                decorators = self._extract_decorators(node, source)
                
                # Extract methods
                body_node = node.child_by_field_name('body')
                methods = []
                if body_node:
                    methods = self._extract_functions(body_node, source, in_class=True)
                
                # Extract docstring
                docstring = self._extract_docstring(body_node, source) if body_node else None
                
                # Check if abstract
                is_abstract = 'abstractmethod' in str(decorators) or 'ABC' in base_classes
                
                # Calculate complexity (simplified)
                complexity = self._calculate_complexity(node)
                
                classes.append(ClassInfo(
                    name=class_name,
                    qualified_name=class_name,  # Would need module path for full qualification
                    base_classes=base_classes,
                    methods=methods,
                    decorators=decorators,
                    docstring=docstring,
                    is_abstract=is_abstract,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    complexity=complexity
                ))
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return classes
    
    def _extract_functions(self, node: Node, source: bytes, in_class: bool = False) -> List[MethodInfo]:
        """Extract function/method definitions"""
        functions = []
        
        def traverse(node: Node, depth: int = 0):
            # Don't traverse into nested classes when extracting module-level functions
            if not in_class and node.type == 'class_definition':
                return
            
            if node.type == 'function_definition':
                name_node = node.child_by_field_name('name')
                if not name_node:
                    return
                
                func_name = source[name_node.start_byte:name_node.end_byte].decode('utf-8')
                
                # Extract parameters
                params = []
                params_node = node.child_by_field_name('parameters')
                if params_node:
                    for param in params_node.children:
                        if param.type in ['identifier', 'typed_parameter']:
                            param_text = source[param.start_byte:param.end_byte].decode('utf-8')
                            params.append(param_text)
                
                # Extract return type
                return_type = None
                return_node = node.child_by_field_name('return_type')
                if return_node:
                    return_type = source[return_node.start_byte:return_node.end_byte].decode('utf-8')
                
                # Extract decorators
                decorators = self._extract_decorators(node, source)
                
                # Check function properties
                is_async = any(child.type == 'async' for child in node.children)
                is_generator = self._is_generator(node)
                is_static = 'staticmethod' in str(decorators)
                is_classmethod = 'classmethod' in str(decorators)
                
                # Extract docstring
                body_node = node.child_by_field_name('body')
                docstring = self._extract_docstring(body_node, source) if body_node else None
                
                # Extract function calls
                calls = self._extract_function_calls(body_node, source) if body_node else []
                
                functions.append(MethodInfo(
                    name=func_name,
                    qualified_name=func_name,  # Would need full path for qualification
                    parameters=params,
                    return_type=return_type,
                    decorators=decorators,
                    docstring=docstring,
                    is_async=is_async,
                    is_generator=is_generator,
                    is_static=is_static,
                    is_classmethod=is_classmethod,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    complexity=self._calculate_complexity(node),
                    calls=calls
                ))
            
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(node)
        return functions
    
    def _extract_imports(self, root_node: Node, source: bytes) -> List[ImportInfo]:
        """Extract import statements"""
        imports = []
        
        def traverse(node: Node):
            if node.type == 'import_statement':
                # Handle: import module1, module2
                for child in node.children:
                    if child.type == 'dotted_name':
                        module = source[child.start_byte:child.end_byte].decode('utf-8')
                        imports.append(ImportInfo(
                            module=module,
                            names=[],
                            alias=None,
                            is_from_import=False,
                            line=node.start_point[0] + 1
                        ))
            
            elif node.type == 'import_from_statement':
                # Handle: from module import name1, name2
                module = None
                names = []
                
                for child in node.children:
                    if child.type == 'dotted_name' and not module:
                        module = source[child.start_byte:child.end_byte].decode('utf-8')
                    elif child.type == 'dotted_name':
                        names.append(source[child.start_byte:child.end_byte].decode('utf-8'))
                
                if module or names:
                    imports.append(ImportInfo(
                        module=module or '',
                        names=names,
                        alias=None,
                        is_from_import=True,
                        line=node.start_point[0] + 1
                    ))
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return imports
    
    def _extract_variables(self, root_node: Node, source: bytes) -> List[VariableInfo]:
        """Extract module-level variable definitions"""
        variables = []
        
        # Only look at top-level assignments
        for child in root_node.children:
            if child.type == 'expression_statement':
                for subchild in child.children:
                    if subchild.type == 'assignment':
                        left_node = subchild.child_by_field_name('left')
                        right_node = subchild.child_by_field_name('right')
                        
                        if left_node and left_node.type == 'identifier':
                            var_name = source[left_node.start_byte:left_node.end_byte].decode('utf-8')
                            
                            # Check if constant (ALL_CAPS)
                            is_constant = var_name.isupper() and '_' in var_name
                            
                            # Extract value (simplified)
                            value = None
                            if right_node:
                                value = source[right_node.start_byte:right_node.end_byte].decode('utf-8')
                            
                            variables.append(VariableInfo(
                                name=var_name,
                                type_hint=None,  # Would need more complex parsing
                                value=value,
                                is_constant=is_constant,
                                line=child.start_point[0] + 1
                            ))
        
        return variables
    
    def _extract_dependencies(self, imports: List[ImportInfo]) -> List[str]:
        """Extract unique dependencies from imports"""
        deps = set()
        for imp in imports:
            if imp.module:
                # Get top-level module name
                module_parts = imp.module.split('.')
                if module_parts[0]:
                    deps.add(module_parts[0])
        return list(deps)
    
    def _extract_decorators(self, node: Node, source: bytes) -> List[str]:
        """Extract decorators from a function or class"""
        decorators = []
        
        # Look for decorator nodes before the definition
        for i, child in enumerate(node.parent.children):
            if child == node:
                # Look backwards for decorators
                for j in range(i-1, -1, -1):
                    if node.parent.children[j].type == 'decorator':
                        dec_text = source[node.parent.children[j].start_byte:node.parent.children[j].end_byte].decode('utf-8')
                        decorators.append(dec_text)
                    else:
                        break
                break
        
        return decorators
    
    def _extract_docstring(self, body_node: Node, source: bytes) -> Optional[str]:
        """Extract docstring from function or class body"""
        if not body_node:
            return None
        
        for child in body_node.children:
            if child.type == 'expression_statement':
                for subchild in child.children:
                    if subchild.type == 'string':
                        docstring = source[subchild.start_byte:subchild.end_byte].decode('utf-8')
                        # Clean up quotes
                        return docstring.strip('"""').strip("'''").strip()
                break  # Docstring must be first statement
        
        return None
    
    def _extract_function_calls(self, body_node: Node, source: bytes) -> List[str]:
        """Extract function calls from a function body"""
        calls = []
        
        def traverse(node: Node):
            if node.type == 'call':
                func_node = node.child_by_field_name('function')
                if func_node:
                    call_text = source[func_node.start_byte:func_node.end_byte].decode('utf-8')
                    calls.append(call_text)
            
            for child in node.children:
                traverse(child)
        
        if body_node:
            traverse(body_node)
        
        return calls
    
    def _is_generator(self, func_node: Node) -> bool:
        """Check if function is a generator (contains yield)"""
        
        def has_yield(node: Node) -> bool:
            if node.type in ['yield_expression', 'yield']:
                return True
            
            for child in node.children:
                if has_yield(child):
                    return True
            
            return False
        
        body_node = func_node.child_by_field_name('body')
        return has_yield(body_node) if body_node else False
    
    def _calculate_complexity(self, node: Node) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1  # Base complexity
        
        def count_branches(node: Node):
            nonlocal complexity
            
            # Increment for control flow statements
            if node.type in ['if_statement', 'elif_clause', 'while_statement', 
                            'for_statement', 'except_clause', 'with_statement']:
                complexity += 1
            elif node.type == 'boolean_operator':  # and/or
                complexity += 1
            
            for child in node.children:
                count_branches(child)
        
        count_branches(node)
        return complexity


class Neo4jCodeGraphUpdater:
    """Updates Neo4j with enhanced code structure information"""
    
    def __init__(self, driver: AsyncDriver, project_name: str):
        self.driver = driver
        self.project_name = project_name
    
    async def update_code_structure(self, structure: CodeStructure):
        """Update Neo4j with detailed code structure"""
        
        # Create module node
        await self._create_module_node(structure)
        
        # Create class nodes and relationships
        for cls in structure.classes:
            await self._create_class_node(cls, structure.file_path)
            
            # Create INHERITS relationships
            for base in cls.base_classes:
                await self._create_inheritance_relationship(cls.name, base)
            
            # Create methods and CONTAINS relationships
            for method in cls.methods:
                await self._create_method_node(method, cls.name, structure.file_path)
        
        # Create module-level functions
        for func in structure.functions:
            await self._create_function_node(func, structure.file_path)
        
        # Create IMPORTS relationships
        for imp in structure.imports:
            await self._create_import_relationship(structure.module_name, imp.module)
        
        # Create CALLS relationships
        await self._create_call_relationships(structure)
    
    async def _create_module_node(self, structure: CodeStructure):
        """Create or update module node"""
        query = """
        MERGE (m:Module {project_name: $project_name, path: $path})
        SET m.name = $name,
            m.language = 'python',
            m.import_count = $import_count,
            m.class_count = $class_count,
            m.function_count = $function_count,
            m.last_analyzed = datetime()
        """
        
        await self.driver.execute_query(
            query,
            {
                'project_name': self.project_name,
                'path': str(structure.file_path),
                'name': structure.module_name,
                'import_count': len(structure.imports),
                'class_count': len(structure.classes),
                'function_count': len(structure.functions)
            }
        )
    
    async def _create_class_node(self, cls: ClassInfo, file_path: Path):
        """Create or update class node"""
        query = """
        MATCH (m:Module {project_name: $project_name, path: $file_path})
        MERGE (c:Class {project_name: $project_name, qualified_name: $qualified_name})
        SET c.name = $name,
            c.is_abstract = $is_abstract,
            c.base_classes = $base_classes,
            c.decorators = $decorators,
            c.docstring = $docstring,
            c.line_start = $line_start,
            c.line_end = $line_end,
            c.complexity = $complexity
        MERGE (m)-[:CONTAINS]->(c)
        """
        
        await self.driver.execute_query(
            query,
            {
                'project_name': self.project_name,
                'file_path': str(file_path),
                'qualified_name': cls.qualified_name,
                'name': cls.name,
                'is_abstract': cls.is_abstract,
                'base_classes': cls.base_classes,
                'decorators': cls.decorators,
                'docstring': cls.docstring,
                'line_start': cls.line_start,
                'line_end': cls.line_end,
                'complexity': cls.complexity
            }
        )
    
    async def _create_method_node(self, method: MethodInfo, class_name: str, file_path: Path):
        """Create or update method node"""
        query = """
        MATCH (c:Class {project_name: $project_name, name: $class_name})
        MERGE (m:Method {project_name: $project_name, qualified_name: $qualified_name})
        SET m.name = $name,
            m.parameters = $parameters,
            m.return_type = $return_type,
            m.is_async = $is_async,
            m.is_generator = $is_generator,
            m.is_static = $is_static,
            m.is_classmethod = $is_classmethod,
            m.decorators = $decorators,
            m.docstring = $docstring,
            m.line_start = $line_start,
            m.line_end = $line_end,
            m.complexity = $complexity
        MERGE (c)-[:CONTAINS]->(m)
        """
        
        qualified_name = f"{class_name}.{method.name}"
        
        await self.driver.execute_query(
            query,
            {
                'project_name': self.project_name,
                'class_name': class_name,
                'qualified_name': qualified_name,
                'name': method.name,
                'parameters': method.parameters,
                'return_type': method.return_type,
                'is_async': method.is_async,
                'is_generator': method.is_generator,
                'is_static': method.is_static,
                'is_classmethod': method.is_classmethod,
                'decorators': method.decorators,
                'docstring': method.docstring,
                'line_start': method.line_start,
                'line_end': method.line_end,
                'complexity': method.complexity
            }
        )
    
    async def _create_function_node(self, func: MethodInfo, file_path: Path):
        """Create or update function node"""
        query = """
        MATCH (m:Module {project_name: $project_name, path: $file_path})
        MERGE (f:Function {project_name: $project_name, qualified_name: $qualified_name})
        SET f.name = $name,
            f.parameters = $parameters,
            f.return_type = $return_type,
            f.is_async = $is_async,
            f.is_generator = $is_generator,
            f.decorators = $decorators,
            f.docstring = $docstring,
            f.line_start = $line_start,
            f.line_end = $line_end,
            f.complexity = $complexity
        MERGE (m)-[:CONTAINS]->(f)
        """
        
        await self.driver.execute_query(
            query,
            {
                'project_name': self.project_name,
                'file_path': str(file_path),
                'qualified_name': func.qualified_name,
                'name': func.name,
                'parameters': func.parameters,
                'return_type': func.return_type,
                'is_async': func.is_async,
                'is_generator': func.is_generator,
                'decorators': func.decorators,
                'docstring': func.docstring,
                'line_start': func.line_start,
                'line_end': func.line_end,
                'complexity': func.complexity
            }
        )
    
    async def _create_inheritance_relationship(self, child_class: str, parent_class: str):
        """Create INHERITS relationship between classes"""
        query = """
        MATCH (child:Class {project_name: $project_name, name: $child_name})
        MERGE (parent:Class {project_name: $project_name, name: $parent_name})
        MERGE (child)-[:INHERITS]->(parent)
        """
        
        await self.driver.execute_query(
            query,
            {
                'project_name': self.project_name,
                'child_name': child_class,
                'parent_name': parent_class
            }
        )
    
    async def _create_import_relationship(self, module_name: str, imported_module: str):
        """Create IMPORTS relationship between modules"""
        if not imported_module:
            return
        
        query = """
        MATCH (m:Module {project_name: $project_name, name: $module_name})
        MERGE (imported:Module {name: $imported_module})
        MERGE (m)-[:IMPORTS]->(imported)
        """
        
        await self.driver.execute_query(
            query,
            {
                'project_name': self.project_name,
                'module_name': module_name,
                'imported_module': imported_module
            }
        )
    
    async def _create_call_relationships(self, structure: CodeStructure):
        """Create CALLS relationships between functions"""
        # This would require more sophisticated analysis to resolve
        # function names to their definitions across files
        pass  # Simplified for this example


# Example usage
async def main():
    """Example of how to use the enhanced analyzer"""
    
    # Analyze a Python file
    analyzer = EnhancedPythonAnalyzer()
    structure = analyzer.analyze_file(Path("example.py"))
    
    # Print extracted information
    print(f"Module: {structure.module_name}")
    print(f"Classes: {[c.name for c in structure.classes]}")
    print(f"Functions: {[f.name for f in structure.functions]}")
    print(f"Imports: {[i.module for i in structure.imports]}")
    
    # Convert to Neo4j nodes
    nodes = structure.to_neo4j_nodes()
    print(f"Generated {len(nodes)} Neo4j nodes")
    
    # Would update Neo4j here
    # updater = Neo4jCodeGraphUpdater(driver, "my_project")
    # await updater.update_code_structure(structure)


if __name__ == "__main__":
    asyncio.run(main())