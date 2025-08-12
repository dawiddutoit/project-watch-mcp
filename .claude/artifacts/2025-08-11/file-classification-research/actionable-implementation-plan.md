# Actionable Implementation Plan for File Classification Enhancement
**Date**: 2025-08-11  
**Target Project**: project-watch-mcp  
**Estimated Effort**: 2-4 weeks for full implementation

## Quick Wins (Day 1-2)

### 1. Fix Mock Embeddings - CRITICAL

**Current Code** (neo4j_rag.py:199):
```python
self.embeddings = embeddings or MockEmbeddingsProvider()
```

**Immediate Fix**:
```python
# In neo4j_rag.py
from .utils.embedding import create_embeddings_provider
import os

class Neo4jRAG:
    def __init__(self, ...):
        # Check for API key and fall back gracefully
        if embeddings is None:
            if os.getenv("OPENAI_API_KEY"):
                self.embeddings = create_embeddings_provider("openai")
                logger.info("Using OpenAI embeddings for semantic search")
            else:
                logger.warning("No OPENAI_API_KEY found. Using local embeddings (degraded performance)")
                self.embeddings = create_embeddings_provider("local")
        else:
            self.embeddings = embeddings
```

**Config Update** (.env):
```bash
# Add embedding configuration
EMBEDDING_PROVIDER=openai  # or "local" or "mock" for testing
OPENAI_API_KEY=sk-...  # If using OpenAI
LOCAL_EMBEDDING_URL=http://localhost:8080/embeddings  # If using local
```

### 2. Add Content-Based Classification

**New Method** for CodeFile class:
```python
# In neo4j_rag.py, add to CodeFile class

def _analyze_content_indicators(self):
    """Enhance classification with content analysis."""
    content_lower = self.content.lower()
    
    # Test indicators (weighted)
    test_indicators = {
        'assert ': 3,
        'describe(': 5,  # Jest/Mocha
        'it(': 5,
        'test(': 4,
        '@test': 5,
        '@pytest': 5,
        'unittest.': 4,
        'expect(': 4,
        '.to.equal': 4,
        '.toBe(': 4,
    }
    
    # API indicators
    api_indicators = {
        '@app.route': 5,
        '@router.': 5,
        'fastapi': 4,
        'flask': 3,
        '@get(': 4,
        '@post(': 4,
        'express.': 3,
        'apicontroller': 4,
    }
    
    # Calculate scores
    test_score = sum(
        weight for pattern, weight in test_indicators.items() 
        if pattern in content_lower
    )
    
    api_score = sum(
        weight for pattern, weight in api_indicators.items()
        if pattern in content_lower
    )
    
    # Enhance classification with confidence
    if test_score > 10:
        self.is_test = True
        self.classification_confidence = min(test_score / 20, 1.0)
    
    # Add new category for API files
    if api_score > 8:
        self.is_api = True
        
    # Store scores for transparency
    self.content_scores = {
        'test': test_score,
        'api': api_score,
    }
```

### 3. Implement Smart Language Detection

**Add Tree-sitter Lite** (without full AST parsing):
```python
# New file: src/project_watch_mcp/utils/language_detector.py
import re
from pathlib import Path

class SmartLanguageDetector:
    """Enhanced language detection beyond file extensions."""
    
    SHEBANG_MAP = {
        'python': ['python', 'python3', 'python2'],
        'node': ['node'],
        'bash': ['bash', 'sh', 'zsh'],
        'ruby': ['ruby'],
        'perl': ['perl'],
    }
    
    CONTENT_PATTERNS = {
        'python': [
            r'^import\s+\w+',
            r'^from\s+\w+\s+import',
            r'^def\s+\w+\(',
            r'^class\s+\w+[\(:]',
        ],
        'javascript': [
            r'^const\s+\w+\s*=',
            r'^let\s+\w+\s*=',
            r'^function\s+\w+\(',
            r'^import\s+.+\s+from\s+[\'"]',
        ],
        'typescript': [
            r'^interface\s+\w+',
            r'^type\s+\w+\s*=',
            r':\s*(string|number|boolean|any)\s*[;,\)]',
        ],
    }
    
    @classmethod
    def detect_language(cls, file_path: Path, content: str) -> tuple[str, float]:
        """
        Detect language with confidence score.
        Returns: (language, confidence)
        """
        # Check shebang first (highest confidence)
        if content.startswith('#!'):
            first_line = content.split('\n')[0]
            for lang, interpreters in cls.SHEBANG_MAP.items():
                if any(interp in first_line for interp in interpreters):
                    return lang, 0.95
        
        # Check content patterns
        for lang, patterns in cls.CONTENT_PATTERNS.items():
            matches = sum(
                1 for pattern in patterns 
                if re.search(pattern, content, re.MULTILINE)
            )
            if matches >= 2:
                confidence = min(matches * 0.2 + 0.5, 0.9)
                return lang, confidence
        
        # Fall back to extension-based detection
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            # ... existing mappings
        }
        
        lang = extension_map.get(file_path.suffix.lower(), 'text')
        confidence = 0.7 if lang != 'text' else 0.3
        
        return lang, confidence
```

## Medium-Priority Enhancements (Day 3-5)

### 4. Replace Chunking with LangChain

**Install Dependencies**:
```bash
uv add langchain-text-splitters
```

**New Chunking Implementation**:
```python
# In neo4j_rag.py, replace chunk_content method

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

def chunk_content(self, content: str, language: str) -> list[dict]:
    """
    Smart chunking using LangChain splitters.
    Returns list of dicts with content and metadata.
    """
    
    # Map our language names to LangChain Language enum
    language_map = {
        'python': Language.PYTHON,
        'javascript': Language.JS,
        'typescript': Language.TS,
        'java': Language.JAVA,
        'cpp': Language.CPP,
        'go': Language.GO,
        'rust': Language.RUST,
        'ruby': Language.RUBY,
    }
    
    # Use language-specific splitter if available
    if language in language_map:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language_map[language],
            chunk_size=1000,  # Characters, not lines
            chunk_overlap=100,
        )
    else:
        # Generic splitter for unknown languages
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )
    
    # Split and create metadata
    chunks = splitter.split_text(content)
    
    # Enhanced chunk metadata
    chunk_data = []
    current_pos = 0
    
    for i, chunk_text in enumerate(chunks):
        # Find line numbers
        start_line = content[:current_pos].count('\n') + 1
        end_line = start_line + chunk_text.count('\n')
        
        # Detect what's in this chunk
        has_imports = 'import ' in chunk_text or 'from ' in chunk_text
        has_class = 'class ' in chunk_text
        has_function = 'def ' in chunk_text or 'function ' in chunk_text
        
        chunk_data.append({
            'content': chunk_text,
            'start_line': start_line,
            'end_line': end_line,
            'chunk_index': i,
            'has_imports': has_imports,
            'has_class': has_class,
            'has_function': has_function,
            'chunk_type': self._determine_chunk_type(chunk_text),
        })
        
        current_pos += len(chunk_text)
    
    return chunk_data

def _determine_chunk_type(self, content: str) -> str:
    """Determine the primary type of content in a chunk."""
    # Simplified heuristic
    if 'import ' in content[:200] or 'from ' in content[:200]:
        return 'imports'
    elif 'class ' in content:
        return 'class_definition'
    elif 'def ' in content or 'function ' in content:
        return 'function_definition'
    elif content.strip().startswith('#') or content.strip().startswith('//'):
        return 'comments'
    else:
        return 'code'
```

### 5. Add Classification Confidence and Multi-Label Support

**Enhanced CodeFile with Confidence**:
```python
@dataclass
class CodeFile:
    # ... existing fields ...
    
    # New fields for enhanced classification
    classification_confidence: dict[str, float] | None = None
    secondary_categories: list[str] | None = None
    content_features: dict[str, any] | None = None
    
    def __post_init__(self):
        # ... existing code ...
        self._calculate_classification_confidence()
    
    def _calculate_classification_confidence(self):
        """Calculate confidence scores for each classification."""
        self.classification_confidence = {}
        
        # Pattern-based confidence
        if self.is_test:
            # High confidence if multiple test patterns match
            test_patterns = ['test_', '_test', 'spec.', '.test.']
            matches = sum(1 for p in test_patterns if p in str(self.path).lower())
            self.classification_confidence['test'] = min(0.5 + matches * 0.2, 1.0)
        
        if self.is_config:
            # Config files are usually obvious
            self.classification_confidence['config'] = 0.9
        
        # Add secondary categories for multi-label support
        self.secondary_categories = []
        
        # A test file might also be documentation (doctest)
        if self.is_test and '.md' in str(self.path):
            self.secondary_categories.append('documentation')
        
        # A config file might also be code (e.g., webpack.config.js)
        if self.is_config and self.language in ['javascript', 'typescript']:
            self.secondary_categories.append('source')
    
    @property
    def all_categories(self) -> list[str]:
        """Return all applicable categories with primary first."""
        categories = [self.file_category]
        if self.secondary_categories:
            categories.extend(self.secondary_categories)
        return categories
```

## Advanced Features (Week 2)

### 6. Add Import/Dependency Extraction

```python
# In neo4j_rag.py, add to CodeFile class

def _extract_imports(self):
    """Extract import statements and dependencies."""
    self.imports = []
    self.external_dependencies = []
    self.internal_dependencies = []
    
    if self.language == 'python':
        # Python imports
        import_pattern = r'^(?:from\s+([\w.]+)|import\s+([\w.]+))'
        for match in re.finditer(import_pattern, self.content, re.MULTILINE):
            module = match.group(1) or match.group(2)
            if module:
                self.imports.append(module)
                # Classify as internal vs external
                if module.startswith('.') or module in self.namespace:
                    self.internal_dependencies.append(module)
                else:
                    self.external_dependencies.append(module.split('.')[0])
    
    elif self.language in ['javascript', 'typescript']:
        # JS/TS imports
        import_patterns = [
            r"import\s+.*?\s+from\s+['\"](.+?)['\"]",
            r"require\(['\"](.+?)['\"]\)",
        ]
        for pattern in import_patterns:
            for match in re.finditer(pattern, self.content):
                module = match.group(1)
                self.imports.append(module)
                if module.startswith('.'):
                    self.internal_dependencies.append(module)
                else:
                    self.external_dependencies.append(module)
    
    # Remove duplicates
    self.imports = list(set(self.imports))
    self.external_dependencies = list(set(self.external_dependencies))
    self.internal_dependencies = list(set(self.internal_dependencies))
```

### 7. Create Graph Relationships

```python
# In neo4j_rag.py, add after indexing files

async def create_file_relationships(self, code_file: CodeFile):
    """Create relationships between files based on imports."""
    
    if not code_file.internal_dependencies:
        return
    
    for dep in code_file.internal_dependencies:
        # Resolve import to file path
        dep_path = self._resolve_import_path(dep, code_file.path)
        
        if dep_path:
            query = """
            MATCH (f1:CodeFile {project_name: $project_name, path: $from_path})
            MATCH (f2:CodeFile {project_name: $project_name, path: $to_path})
            MERGE (f1)-[r:IMPORTS]->(f2)
            SET r.import_name = $import_name
            """
            
            await self.neo4j_driver.execute_query(
                query,
                {
                    "project_name": self.project_name,
                    "from_path": str(code_file.path),
                    "to_path": str(dep_path),
                    "import_name": dep,
                },
                routing_control=RoutingControl.WRITE,
            )

def _resolve_import_path(self, import_name: str, from_file: Path) -> Path | None:
    """Resolve an import name to an actual file path."""
    # Simplified implementation
    if import_name.startswith('.'):
        # Relative import
        base_dir = from_file.parent
        levels = import_name.count('.')
        for _ in range(levels - 1):
            base_dir = base_dir.parent
        
        module_path = import_name.lstrip('.').replace('.', '/')
        
        # Try common patterns
        candidates = [
            base_dir / f"{module_path}.py",
            base_dir / module_path / "__init__.py",
            base_dir / f"{module_path}.js",
            base_dir / f"{module_path}.ts",
            base_dir / module_path / "index.js",
            base_dir / module_path / "index.ts",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
    
    return None
```

## Testing and Validation (Ongoing)

### 8. Add Classification Metrics

```python
# New file: src/project_watch_mcp/utils/metrics.py

from dataclasses import dataclass
from typing import Dict, List
import json
from pathlib import Path

@dataclass
class ClassificationMetrics:
    """Track classification accuracy and performance."""
    
    total_files: int = 0
    correct_classifications: int = 0
    confidence_scores: List[float] = None
    classification_times: List[float] = None
    misclassified_files: List[Dict] = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = []
        if self.classification_times is None:
            self.classification_times = []
        if self.misclassified_files is None:
            self.misclassified_files = []
    
    def record_classification(self, file_path: str, predicted: str, 
                            actual: str, confidence: float, time_ms: float):
        """Record a classification result."""
        self.total_files += 1
        self.confidence_scores.append(confidence)
        self.classification_times.append(time_ms)
        
        if predicted == actual:
            self.correct_classifications += 1
        else:
            self.misclassified_files.append({
                'file': file_path,
                'predicted': predicted,
                'actual': actual,
                'confidence': confidence,
            })
    
    @property
    def accuracy(self) -> float:
        """Calculate classification accuracy."""
        if self.total_files == 0:
            return 0.0
        return self.correct_classifications / self.total_files
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    @property
    def avg_time_ms(self) -> float:
        """Average classification time in milliseconds."""
        if not self.classification_times:
            return 0.0
        return sum(self.classification_times) / len(self.classification_times)
    
    def save_report(self, path: Path):
        """Save metrics report to file."""
        report = {
            'total_files': self.total_files,
            'accuracy': self.accuracy,
            'avg_confidence': self.avg_confidence,
            'avg_time_ms': self.avg_time_ms,
            'misclassified_samples': self.misclassified_files[:10],  # Top 10
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
```

## Rollout Strategy

### Phase 1: Silent Testing (Day 1-3)
1. Implement new classification alongside old
2. Log both results but use old for production
3. Compare accuracy and performance

### Phase 2: A/B Testing (Day 4-7)
1. Route 10% of traffic to new classifier
2. Monitor error rates and user feedback
3. Gradually increase percentage if stable

### Phase 3: Full Rollout (Week 2)
1. Switch to new classifier as default
2. Keep old classifier as fallback
3. Monitor metrics dashboard

### Phase 4: Optimization (Week 3-4)
1. Analyze misclassification patterns
2. Fine-tune confidence thresholds
3. Add project-specific rules based on data

## Configuration Management

```yaml
# config/classification.yaml
classification:
  # Feature flags
  use_content_analysis: true
  use_langchain_splitter: true
  use_smart_language_detection: true
  
  # Confidence thresholds
  min_confidence_threshold: 0.6
  multi_label_threshold: 0.7
  
  # Performance settings
  max_file_size_mb: 10
  chunk_size_chars: 1000
  chunk_overlap_chars: 100
  
  # Embedding settings
  embedding_provider: ${EMBEDDING_PROVIDER:-openai}
  embedding_cache_ttl: 3600
  
  # Fallback behavior
  fallback_to_extension: true
  fallback_to_mock_embeddings: false
```

## Success Metrics

Target metrics after implementation:

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|--------------|
| Classification Accuracy | ~70% | 85% | 95% |
| Semantic Search Relevance | 40% | 70% | 85% |
| Avg Classification Time | 50ms | 30ms | 20ms |
| Chunk Coherence | 50% | 80% | 90% |
| User Satisfaction | Unknown | 80% | 90% |

## Risk Mitigation

1. **Performance Degradation**: Add caching layer for embeddings
2. **Memory Issues**: Implement streaming for large files
3. **API Rate Limits**: Add retry logic with exponential backoff
4. **Breaking Changes**: Version the classification API
5. **Data Loss**: Keep classification history for rollback

---

*This plan prioritizes pragmatic improvements over theoretical perfection. Start with quick wins, measure everything, and only add complexity where data shows it's needed.*