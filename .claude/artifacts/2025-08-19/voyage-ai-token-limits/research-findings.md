# Voyage AI Token Limits and Chunking Strategies Research

## Executive Summary

### Critical Findings
1. **Token Limit Reality**: The 32,000 token limit is NOT a per-document limit but a **context length** limit. Different models have different **total token processing limits** per API request.
2. **Solution Available**: Voyage AI provides `voyage-context-3` model specifically designed for handling large documents through contextualized chunk embeddings.
3. **Best Practice**: Use chunking with contextualized embeddings rather than truncation - this preserves all information while maintaining semantic relationships.

### Key Recommendations
1. **Immediate Action**: Switch from truncation to chunking strategy
2. **Model Selection**: Consider `voyage-context-3` for documents requiring context preservation
3. **Implementation**: Use batching for large document sets
4. **Architecture**: Implement proper chunk management with metadata tracking

## Detailed Research Findings

## 1. Voyage AI's Actual Token Limits

### Context Length vs Total Token Limits
- **Context Length**: 32,000 tokens (maximum tokens per individual input)
- **Total Token Limits** (per API request):
  - `voyage-3.5-lite`: 1,000,000 tokens
  - `voyage-3.5`: 320,000 tokens  
  - `voyage-3-large`: 120,000 tokens
  - `voyage-code-3`: 120,000 tokens
  - `voyage-context-3`: 120,000 tokens (with max 16,000 chunks)

### Additional Constraints
- Maximum batch size: 1,000 inputs per request
- Rate limits: 8M TPM (tokens per minute) for voyage-3.5

## 2. Recommended Strategies for Handling Large Texts

### Strategy 1: Chunking with Standard Embeddings
```python
def chunk_document(text, max_tokens=30000):
    """Split document into chunks under token limit"""
    # Use tokenizer to count tokens
    vo = voyageai.Client()
    total_tokens = vo.count_tokens([text])
    
    if total_tokens <= max_tokens:
        return [text]
    
    # Split into chunks (simplified)
    chunk_size = len(text) * max_tokens // total_tokens
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    
    return chunks

# Embed chunks separately
chunks = chunk_document(large_text)
embeddings = vo.embed(
    chunks,
    model="voyage-3.5",
    input_type="document"
).embeddings
```

### Strategy 2: Contextualized Chunk Embeddings (RECOMMENDED)
```python
def create_contextualized_embeddings(document_chunks):
    """Create embeddings that preserve document context"""
    vo = voyageai.Client()
    
    # Group chunks by document
    result = vo.contextualized_embed(
        inputs=[document_chunks],  # List of chunks from same document
        model="voyage-context-3",
        input_type="document",
        output_dimension=1024
    )
    
    return result.embeddings
```

**Advantages**:
- Preserves semantic relationships between chunks
- No manual context augmentation needed
- 20% better retrieval accuracy than standard chunking

## 3. Splitting Large Texts Into Parts

### Yes, You Can and Should Split!

**Best Practices**:
1. **Semantic Boundaries**: Split at natural boundaries (paragraphs, sections)
2. **Overlap Strategy**: Include 10-20% overlap between chunks
3. **Metadata Preservation**: Track chunk position and document origin

### Implementation Example:
```python
def smart_chunk_with_overlap(text, chunk_size=25000, overlap=0.1):
    """Create overlapping chunks for better context preservation"""
    chunks = []
    overlap_size = int(chunk_size * overlap)
    
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Find natural boundary (paragraph/sentence end)
        if end < len(text):
            boundary = text.rfind('\n\n', start, end)
            if boundary > start:
                end = boundary
        
        chunks.append({
            'text': text[start:end],
            'start': start,
            'end': end,
            'chunk_id': len(chunks)
        })
        
        start = end - overlap_size if end < len(text) else end
    
    return chunks
```

## 4. Best Practices for Chunking Strategies

### Chunking Strategy Selection
1. **Recursive Character Splitting**: Best for structured documents
2. **Semantic Chunking**: Best for narrative text
3. **Token-Based Chunking**: Best for precise token control

### Voyage AI Specific Best Practices

#### For Standard Models (voyage-3.5, voyage-3-large):
```python
class VoyageChunker:
    def __init__(self, model="voyage-3.5"):
        self.vo = voyageai.Client()
        self.model = model
        self.max_tokens = 30000  # Safe buffer under 32K limit
        
    def chunk_and_embed(self, documents):
        all_chunks = []
        all_embeddings = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            
            # Batch processing for efficiency
            batch_size = 128
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                embeddings = self.vo.embed(
                    batch,
                    model=self.model,
                    input_type="document"
                ).embeddings
                
                all_chunks.extend(batch)
                all_embeddings.extend(embeddings)
        
        return all_chunks, all_embeddings
```

#### For Contextualized Model (voyage-context-3):
```python
class ContextualVoyageChunker:
    def __init__(self):
        self.vo = voyageai.Client()
        self.max_chunks_per_doc = 100  # Well under 16K limit
        
    def process_documents(self, documents):
        """Process documents with contextual awareness"""
        results = []
        
        for doc in documents:
            # Split into semantic chunks
            chunks = self.semantic_chunk(doc, max_chunks=self.max_chunks_per_doc)
            
            # Create contextualized embeddings
            embeddings = self.vo.contextualized_embed(
                inputs=[chunks],
                model="voyage-context-3",
                input_type="document"
            ).embeddings
            
            results.append({
                'chunks': chunks,
                'embeddings': embeddings,
                'metadata': {'doc_id': doc['id'], 'total_chunks': len(chunks)}
            })
        
        return results
```

## 5. Different Models with Different Token Limits

### Model Comparison Table

| Model | Context Length | Total Tokens/Request | Best Use Case | Price/1M tokens |
|-------|---------------|---------------------|---------------|-----------------|
| voyage-3.5-lite | 32K | 1M | High-volume, cost-sensitive | $0.02 |
| voyage-3.5 | 32K | 320K | General purpose, high quality | $0.06 |
| voyage-3-large | 32K | 120K | Maximum quality | $0.12 |
| voyage-code-3 | 32K | 120K | Code search and retrieval | $0.06 |
| voyage-context-3 | 32K | 120K (16K chunks) | Document chunking with context | $0.12 |

### Model Selection Criteria
1. **For Code**: Use `voyage-code-3`
2. **For Documents with Context**: Use `voyage-context-3`
3. **For General Purpose**: Use `voyage-3.5`
4. **For Budget Constraints**: Use `voyage-3.5-lite`

## 6. Handling Semantic Search Across Multiple Chunks

### Architecture for Multi-Chunk Search

```python
class MultiChunkSearchSystem:
    def __init__(self):
        self.vo = voyageai.Client()
        self.chunk_index = {}  # chunk_id -> document_id mapping
        self.embeddings_store = {}  # chunk_id -> embedding
        
    def index_document(self, doc_id, text):
        """Index a document by chunking and embedding"""
        chunks = self.create_chunks_with_metadata(text, doc_id)
        
        # Store chunk-document relationships
        for chunk in chunks:
            chunk_id = f"{doc_id}_{chunk['chunk_num']}"
            self.chunk_index[chunk_id] = {
                'doc_id': doc_id,
                'chunk_num': chunk['chunk_num'],
                'total_chunks': chunk['total_chunks'],
                'text': chunk['text']
            }
        
        # Create embeddings
        texts = [c['text'] for c in chunks]
        embeddings = self.vo.embed(
            texts,
            model="voyage-3.5",
            input_type="document"
        ).embeddings
        
        # Store embeddings
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{chunk['chunk_num']}"
            self.embeddings_store[chunk_id] = embeddings[i]
    
    def search(self, query, top_k=5, aggregate_by_document=True):
        """Search across all chunks"""
        # Embed query
        query_embedding = self.vo.embed(
            [query],
            model="voyage-3.5",
            input_type="query"
        ).embeddings[0]
        
        # Calculate similarities
        similarities = []
        for chunk_id, embedding in self.embeddings_store.items():
            similarity = self.cosine_similarity(query_embedding, embedding)
            similarities.append({
                'chunk_id': chunk_id,
                'similarity': similarity,
                'metadata': self.chunk_index[chunk_id]
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        if aggregate_by_document:
            # Aggregate scores by document
            doc_scores = {}
            for result in similarities[:top_k*3]:  # Consider more chunks
                doc_id = result['metadata']['doc_id']
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'max_score': result['similarity'],
                        'avg_score': result['similarity'],
                        'chunks': [result]
                    }
                else:
                    doc_scores[doc_id]['chunks'].append(result)
                    scores = [c['similarity'] for c in doc_scores[doc_id]['chunks']]
                    doc_scores[doc_id]['max_score'] = max(scores)
                    doc_scores[doc_id]['avg_score'] = sum(scores) / len(scores)
            
            # Return top documents
            top_docs = sorted(
                doc_scores.items(),
                key=lambda x: x[1]['max_score'],
                reverse=True
            )[:top_k]
            
            return top_docs
        else:
            return similarities[:top_k]
```

### Best Practices for Multi-Chunk Search

1. **Chunk Overlap**: Maintain 10-20% overlap to avoid losing context at boundaries
2. **Metadata Tracking**: Always track chunk position, document ID, and total chunks
3. **Scoring Strategies**:
   - **Max Pooling**: Use highest chunk score as document score
   - **Average Pooling**: Average all chunk scores
   - **Weighted Average**: Weight by chunk position or relevance
4. **Result Aggregation**: Return adjacent chunks for better context

## Implementation Recommendations for project-watch-mcp

### Immediate Changes Needed

1. **Stop Truncation**: Replace the current truncation at 32,000 tokens with proper chunking
2. **Implement Chunking Strategy**:
```python
class VoyageEmbedder:
    def __init__(self, model="voyage-3.5"):
        self.client = voyageai.Client()
        self.model = model
        self.max_tokens_per_chunk = 30000  # Safe buffer
        
    def embed_large_file(self, file_content, file_path):
        """Embed a potentially large file"""
        # Check token count
        token_count = self.client.count_tokens([file_content])
        
        if token_count <= self.max_tokens_per_chunk:
            # Single chunk
            return self.client.embed(
                [file_content],
                model=self.model,
                input_type="document"
            ).embeddings[0]
        else:
            # Multiple chunks needed
            chunks = self.create_semantic_chunks(file_content)
            chunk_embeddings = []
            
            # Process in batches
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                embeddings = self.client.embed(
                    batch,
                    model=self.model,
                    input_type="document"
                ).embeddings
                chunk_embeddings.extend(embeddings)
            
            # Store with metadata
            return {
                'chunks': chunks,
                'embeddings': chunk_embeddings,
                'metadata': {
                    'file_path': file_path,
                    'total_chunks': len(chunks),
                    'total_tokens': token_count
                }
            }
```

3. **Consider voyage-context-3**: For better retrieval accuracy
4. **Update Database Schema**: Store chunk relationships and metadata
5. **Implement Chunk Aggregation**: For search results

### Risk Mitigation

1. **Token Counting**: Always pre-check token counts before embedding
2. **Error Handling**: Implement retry logic for rate limits
3. **Monitoring**: Track chunking effectiveness and retrieval accuracy
4. **Testing**: Validate retrieval quality with chunked vs truncated approach

## Conclusion

The current truncation approach loses information unnecessarily. Voyage AI provides robust solutions for handling large documents through:
1. Generous token limits (up to 1M for voyage-3.5-lite)
2. Contextualized embeddings (voyage-context-3) for maintaining semantic relationships
3. Flexible chunking strategies with proper API support

The recommended approach is to implement semantic chunking with overlap, use contextualized embeddings where possible, and maintain proper metadata for chunk-to-document relationships. This will ensure no information loss while maintaining high retrieval accuracy.

## Next Steps

1. Implement proof-of-concept chunking system
2. Test with current large files that are being truncated
3. Compare retrieval accuracy: truncated vs chunked
4. Deploy updated embedding strategy
5. Monitor performance and adjust chunk sizes as needed

## Confidence Levels

- Token limit understanding: **High** (verified from official docs)
- Chunking strategies: **High** (well-documented best practices)
- voyage-context-3 benefits: **High** (official benchmarks available)
- Implementation recommendations: **Medium-High** (based on patterns, needs testing)
- Multi-chunk search architecture: **Medium** (requires validation with actual use case)