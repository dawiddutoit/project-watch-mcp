# Voyage AI Token Limit Research Report
## Executive Summary and Critical Findings

### 🚨 Key Discovery: 32,000 Token Limit is Per-Document, Not Total
The 32,000 token limit in Voyage AI is a **context length limit per document**, not a hard limit on what can be embedded. The solution is to implement intelligent chunking strategies that split large documents into smaller pieces while preserving semantic meaning.

### Critical Findings:
1. **Token Limit is Configurable**: The `truncation` parameter can be set to `False` to raise errors instead of silently truncating
2. **Multiple Models Available**: Different models have different token limits and capabilities
3. **Contextualized Embeddings Solution**: `voyage-context-3` model specifically designed for handling chunked documents
4. **Batch Processing Supported**: Can process up to 1,000 texts in a single API call with varying total token limits

## 1. Voyage AI Token Limits by Model

### Standard Embedding Models
| Model | Context Length | Total API Tokens | Dimensions |
|-------|---------------|------------------|------------|
| voyage-3-large | 32,000 | 120K | 1024 |
| voyage-3.5 | 32,000 | 320K | 1024 |
| voyage-3.5-lite | 32,000 | 1M | 1024 |
| voyage-code-3 | 32,000 | 120K | 1536 |
| voyage-finance-2 | 32,000 | 120K | 1024 |
| voyage-law-2 | 32,000 | 120K | 1024 |

### Contextualized Embedding Model
| Model | Context Length | Total API Tokens | Total Chunks |
|-------|---------------|------------------|--------------|
| voyage-context-3 | 32,000 | 120K | 16K |

**Note**: Older models (voyage-2, voyage-large-2) have 16,000 or 4,000 token limits.

## 2. Recommended Strategies for Handling Text Exceeding Limits

### Strategy 1: Smart Chunking (Recommended)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Estimate tokens: ~5 characters per token
def estimate_tokens(text):
    return len(text) / 5

# Create chunker with token-aware sizing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,  # ~1000 tokens per chunk
    chunk_overlap=500,  # ~100 tokens overlap (10%)
    length_function=len,
    is_separator_regex=False,
)

# Split document
chunks = text_splitter.split_text(long_document)
```

### Strategy 2: Use Contextualized Embeddings (Best for Retrieval)
```python
import voyageai

vo = voyageai.Client()

# Split document into chunks
chunks = split_document(document)  # Your chunking logic

# Embed with context
embeddings = vo.contextualized_embed(
    inputs=[chunks],  # List of lists - each inner list is a document's chunks
    model="voyage-context-3",
    input_type="document"
).embeddings
```

### Strategy 3: Batch Processing for Multiple Documents
```python
batch_size = 128
all_embeddings = []

for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    embeddings = vo.embed(
        batch,
        model="voyage-3.5",
        input_type="document",
        truncation=False  # Raise error instead of truncating
    ).embeddings
    all_embeddings.extend(embeddings)
```

## 3. Chunking Best Practices

### Optimal Chunk Sizes
- **Standard Embeddings**: 1000-2000 tokens per chunk
- **Contextualized Embeddings**: 500-1000 tokens per chunk (performs better with smaller chunks)
- **Code**: 500-1000 tokens (preserve function/class boundaries)

### Overlap Recommendations
- **General Text**: 10-15% overlap (100-150 tokens for 1000-token chunks)
- **Technical Documentation**: 15-20% overlap
- **No Overlap**: When using `voyage-context-3` (handles context automatically)

### Implementation Example
```python
def chunk_for_voyage(text, max_tokens=1000, overlap_tokens=100):
    """
    Chunk text for Voyage AI embeddings
    """
    # Estimate character count per token (avg 5 chars)
    chars_per_token = 5
    max_chars = max_tokens * chars_per_token
    overlap_chars = overlap_tokens * chars_per_token
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chars, len(text))
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind('.', start, end)
            if last_period > start + (max_chars * 0.8):
                end = last_period + 1
        
        chunks.append(text[start:end])
        start = end - overlap_chars if end < len(text) else end
    
    return chunks
```

## 4. Token Estimation and Counting

### Token-Character Relationship
- **Average**: 1 token ≈ 5 characters
- **Word-Token Ratio**: 1 word ≈ 1.2-1.5 tokens
- **Voyage vs OpenAI**: Voyage produces 1.1-1.2x more tokens than tiktoken

### Token Counting Tools
```python
import voyageai

vo = voyageai.Client()

# Count tokens
token_count = vo.count_tokens(["text1", "text2"])
print(f"Total tokens: {token_count}")

# Get detailed tokenization
tokenized = vo.tokenize(["text to analyze"])
for result in tokenized:
    print(f"Tokens: {result.tokens}")
    print(f"Token count: {len(result.tokens)}")
```

## 5. Recommended Solution for project-watch-mcp

### Implementation Strategy

1. **Update Voyage Embedding Class**:
```python
class VoyageEmbeddings:
    def __init__(self, model="voyage-code-3", chunk_size=1000, use_context=True):
        self.client = voyageai.Client()
        self.model = model
        self.chunk_size = chunk_size
        self.use_context = use_context
        self.max_tokens = 32000
        
    def embed_large_text(self, text, metadata=None):
        """Embed text that may exceed token limits"""
        
        # Estimate tokens
        estimated_tokens = len(text) / 5
        
        if estimated_tokens <= self.max_tokens * 0.9:  # Single embedding
            return self._embed_single(text)
        
        # Chunk the text
        chunks = self._smart_chunk(text)
        
        if self.use_context and self.model == "voyage-code-3":
            # Use contextualized embeddings
            return self._embed_contextualized(chunks, metadata)
        else:
            # Standard chunked embeddings
            return self._embed_chunks(chunks, metadata)
    
    def _smart_chunk(self, text):
        """Intelligently chunk text preserving code structure"""
        # Implementation here
        pass
    
    def _embed_contextualized(self, chunks, metadata):
        """Use voyage-context-3 for better retrieval"""
        embeddings = self.client.contextualized_embed(
            inputs=[chunks],
            model="voyage-context-3",
            input_type="document"
        ).embeddings
        return embeddings
```

2. **Database Schema Update**:
```sql
-- Store chunks with relationships
CREATE TABLE code_chunks (
    id TEXT PRIMARY KEY,
    file_id TEXT REFERENCES files(id),
    chunk_index INTEGER,
    content TEXT,
    embedding VECTOR(1024),
    start_line INTEGER,
    end_line INTEGER,
    tokens INTEGER
);
```

3. **Retrieval Strategy**:
- For search: Embed query, find similar chunks
- For context: Retrieve adjacent chunks
- For code: Preserve function/class boundaries

## 6. Specific Recommendations

### For Code Files (voyage-code-3)
1. **Chunk at natural boundaries**: Functions, classes, methods
2. **Preserve context**: Include imports and class definitions
3. **Size**: 500-1000 tokens per chunk
4. **Overlap**: Minimal or none (use AST parsing)

### For Documentation (voyage-3.5)
1. **Chunk by sections**: Headers, paragraphs
2. **Size**: 1000-1500 tokens
3. **Overlap**: 10-15%
4. **Metadata**: Include section headers

### For Mixed Content
1. **Use voyage-context-3**: Handles context automatically
2. **Smaller chunks**: 500-750 tokens
3. **No overlap needed**: Model handles context
4. **Batch process**: Up to 16K chunks total

## 7. Error Handling

```python
def safe_embed(text, vo_client, model="voyage-code-3"):
    """Safely embed text with error handling"""
    try:
        # Try without truncation first
        result = vo_client.embed(
            [text],
            model=model,
            truncation=False
        )
        return result.embeddings[0]
    except Exception as e:
        if "exceeds" in str(e).lower():
            # Text too long, chunk it
            chunks = chunk_for_voyage(text)
            embeddings = []
            for chunk in chunks:
                emb = vo_client.embed(
                    [chunk],
                    model=model,
                    truncation=True
                ).embeddings[0]
                embeddings.append(emb)
            return embeddings
        else:
            raise
```

## 8. Performance Considerations

### Latency vs Accuracy Trade-offs
- **voyage-3.5-lite**: Fastest, good for real-time
- **voyage-3.5**: Balanced performance
- **voyage-3-large**: Best accuracy, slower
- **voyage-context-3**: Best for chunked documents

### Batch Processing Limits
- Process up to 1000 texts per API call
- Stay within total token limits
- Use async/parallel processing for large datasets

## 9. Migration Path for project-watch-mcp

### Phase 1: Immediate Fix
1. Implement chunking for files > 25,000 tokens
2. Store chunks with file relationships
3. Update search to work across chunks

### Phase 2: Optimization
1. Migrate to voyage-context-3 for better retrieval
2. Implement AST-based chunking for code
3. Add chunk caching

### Phase 3: Enhancement
1. Dynamic model selection based on content
2. Implement hybrid search (keyword + semantic)
3. Add chunk relationship graph

## 10. Code Implementation Template

```python
# Complete implementation for project-watch-mcp

import voyageai
from typing import List, Dict, Any
import hashlib

class EnhancedVoyageEmbeddings:
    def __init__(self, api_key: str):
        self.client = voyageai.Client(api_key=api_key)
        self.models = {
            'code': 'voyage-code-3',
            'general': 'voyage-3.5',
            'context': 'voyage-context-3'
        }
        
    def process_file(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Process a file and return chunks with embeddings"""
        
        # Detect file type
        file_type = self._detect_file_type(file_path)
        
        # Choose strategy
        if file_type == 'code':
            chunks = self._chunk_code(content)
            model = self.models['code']
        else:
            chunks = self._chunk_text(content)
            model = self.models['general']
        
        # Generate embeddings
        results = []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{file_path}:{i}:{chunk[:100]}".encode()).hexdigest()
            
            embedding = self.client.embed(
                [chunk['content']],
                model=model,
                input_type='document'
            ).embeddings[0]
            
            results.append({
                'id': chunk_id,
                'file_path': file_path,
                'chunk_index': i,
                'content': chunk['content'],
                'embedding': embedding,
                'metadata': chunk.get('metadata', {}),
                'tokens': len(chunk['content']) / 5  # Estimate
            })
        
        return results
    
    def _chunk_code(self, content: str) -> List[Dict[str, Any]]:
        """Chunk code preserving structure"""
        # Implementation with AST parsing
        pass
    
    def _chunk_text(self, content: str) -> List[Dict[str, Any]]:
        """Chunk text with overlap"""
        # Implementation with smart chunking
        pass
```

## Risk Assessment and Mitigation

### Identified Risks
1. **Silent Truncation**: Currently truncating without warning
   - **Mitigation**: Set `truncation=False`, implement proper chunking
   
2. **Loss of Context**: Chunks may lose important context
   - **Mitigation**: Use voyage-context-3 or maintain overlap
   
3. **Search Quality Degradation**: Multiple chunks per file
   - **Mitigation**: Implement chunk aggregation in search

4. **Performance Impact**: More embeddings = higher cost
   - **Mitigation**: Cache embeddings, batch processing

### Monitoring Recommendations
- Track files exceeding token limits
- Monitor search quality metrics
- Log embedding generation times
- Alert on truncation events

## Next Steps

1. **Immediate** (Today):
   - Implement basic chunking to prevent truncation
   - Add logging for files > 25,000 tokens
   
2. **Short-term** (This Week):
   - Implement smart chunking strategies
   - Update database schema for chunks
   - Test voyage-context-3 model

3. **Long-term** (This Month):
   - Full migration to chunked architecture
   - Implement AST-based code chunking
   - Add performance monitoring

## Conclusion

The 32,000 token limit is not a blocker but requires architectural changes to handle properly. The recommended approach is:

1. Implement intelligent chunking that preserves semantic meaning
2. Use voyage-context-3 for better retrieval with chunked documents
3. Store chunks with proper relationships in the database
4. Update search to aggregate results across chunks

This will not only solve the current truncation issue but also improve search quality and scalability for larger codebases.