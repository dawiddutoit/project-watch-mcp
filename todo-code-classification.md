# Code Classification Implementation TODO

## 🎯 Goal
Enhance project-watch-mcp with Pygments-based code classification to identify classes, functions, variables, tests, and configurations.

## 📋 Implementation Tasks

### Phase 1: Core Classification Module
- [ ] **Create code_classifier.py module**
  - Location: `src/project_watch_mcp/code_classifier.py`
  - Implement `PygmentsCodeClassifier` class
  - Add `CodeElement` dataclass
  - Define token mappings for element types
  - Implement file type pattern matching
  - Tests: `tests/unit/test_code_classifier.py`

- [ ] **Enhance CodeFile dataclass**
  - Location: `src/project_watch_mcp/neo4j_rag.py`
  - Add `code_elements` field (dict)
  - Add `element_count` field (dict)
  - Integrate classification in `__post_init__`
  - Tests: Update existing tests in `test_neo4j_rag.py`

### Phase 2: Neo4j Integration
- [ ] **Update Neo4j schema**
  - Add `CodeElement` node type
  - Add `CONTAINS_ELEMENT` relationship
  - Add `HAS_METHOD` relationship for class methods
  - Add `TESTS` relationship for test coverage
  - Create migration script: `scripts/add_code_elements_schema.py`

- [ ] **Enhance indexing with classification**
  - Location: `neo4j_rag.py::index_file()`
  - Call classifier during indexing
  - Store elements in Neo4j
  - Batch create element nodes
  - Tests: `tests/integration/test_element_indexing.py`

- [ ] **Update batch_index_files**
  - Location: `neo4j_rag.py::batch_index_files()`
  - Add classification to batch processing
  - Maintain performance with UNWIND operations
  - Tests: Update batch operation tests

### Phase 3: Search Capabilities
- [ ] **Add element search methods**
  - Location: `neo4j_rag.py`
  - Implement `search_by_element_type()`
  - Implement `find_test_coverage()`
  - Implement `get_class_methods()`
  - Tests: `tests/unit/test_element_search.py`

- [ ] **Create MCP tools for element search**
  - Location: `server.py`
  - Add `search_code_elements` tool
  - Add `find_related_tests` tool
  - Add `get_code_structure` tool
  - Tests: `tests/unit/test_server_element_tools.py`

### Phase 4: Testing & Validation
- [ ] **Unit tests for classifier**
  - Test Python classification
  - Test JavaScript classification
  - Test file type detection
  - Test element counting
  - Location: `tests/unit/test_code_classifier.py`

- [ ] **Integration tests**
  - Test end-to-end classification
  - Test search with classified elements
  - Test performance impact
  - Location: `tests/integration/test_classification_integration.py`

- [ ] **Performance validation**
  - Measure classification overhead
  - Ensure <100ms per file classification
  - Validate batch performance maintained
  - Create benchmark: `tests/performance/test_classification_performance.py`

### Phase 5: Documentation
- [ ] **Update README**
  - Add classification features section
  - Document new search capabilities
  - Add example queries

- [ ] **Create usage guide**
  - Location: `docs/code-classification-guide.md`
  - Examples of element search
  - Test discovery examples
  - Performance considerations

## 🔧 Implementation Details

### File Structure
```
src/project_watch_mcp/
├── code_classifier.py      # NEW: Classification module
├── neo4j_rag.py            # UPDATE: Add classification
├── server.py               # UPDATE: Add new tools
└── utils/
    └── classification/     # FUTURE: Advanced classifiers
        ├── __init__.py
        └── tree_sitter.py  # FUTURE: AST parsing

tests/
├── unit/
│   ├── test_code_classifier.py     # NEW
│   ├── test_element_search.py      # NEW
│   └── test_server_element_tools.py # NEW
└── integration/
    ├── test_element_indexing.py     # NEW
    └── test_classification_integration.py # NEW
```

### Key Classes
```python
# Core classes to implement
class PygmentsCodeClassifier:
    def classify_file(file_path, content) -> dict
    def _extract_elements(content, lexer) -> List[CodeElement]
    def _classify_file_type(file_path) -> str

@dataclass
class CodeElement:
    name: str
    type: str  # 'class', 'function', 'test', etc.
    line_number: int
    context: str
    metadata: dict
```

### Neo4j Queries to Add
```cypher
// Find all classes in project
MATCH (e:CodeElement {project_name: $project, type: 'class'})
RETURN e.name, e.file_path, e.line

// Find tests for a class
MATCH (class:CodeElement {name: $className})
MATCH (test:CodeElement {type: 'test_function'})
WHERE test.name CONTAINS class.name
RETURN test.name, test.file_path

// Get code structure
MATCH (f:CodeFile {path: $path})
MATCH (f)-[:CONTAINS_ELEMENT]->(e:CodeElement)
RETURN e.type, collect(e.name) as elements
```

## 📊 Success Metrics

### Functionality
- [ ] Can identify classes in Python, JavaScript, TypeScript
- [ ] Can identify functions and methods
- [ ] Can distinguish test functions from regular functions
- [ ] Can identify configuration files
- [ ] Can find tests related to specific code

### Performance
- [ ] Classification adds <100ms per file
- [ ] Batch indexing maintains 50x+ improvement
- [ ] Search queries return in <500ms
- [ ] No memory increase >10%

### Quality
- [ ] 95% accuracy in element identification
- [ ] All existing tests still pass
- [ ] New features have 80%+ test coverage

## 🚀 Quick Start

1. **Start with Python-only support**
   ```python
   # Begin with Python classification
   classifier = PygmentsCodeClassifier()
   result = classifier.classify_file(Path("file.py"), content)
   ```

2. **Test with existing codebase**
   ```bash
   # Run on project-watch-mcp itself
   python -m project_watch_mcp.code_classifier src/
   ```

3. **Validate with simple examples**
   ```python
   # Test files should be detected
   assert classify_file("test_example.py")['file_type'] == 'test'
   
   # Classes should be found
   assert 'MyClass' in [e.name for e in elements if e.type == 'class']
   ```

## ⚠️ Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Performance impact | Use caching, batch operations |
| Pygments limitations | Plan migration path to tree-sitter |
| Schema complexity | Start simple, expand incrementally |
| Breaking changes | Feature flag for classification |

## 📅 Timeline Estimate

- **Phase 1**: Core module - 2 hours
- **Phase 2**: Neo4j integration - 3 hours  
- **Phase 3**: Search capabilities - 2 hours
- **Phase 4**: Testing - 2 hours
- **Phase 5**: Documentation - 1 hour

**Total**: ~10 hours of implementation

## 🎉 Done When

- [ ] Code classification works for Python files
- [ ] Elements are stored in Neo4j
- [ ] Can search by element type via MCP
- [ ] Can find related tests
- [ ] Performance impact <10%
- [ ] All tests pass
- [ ] Documentation complete