# TODO: Fix Mock Embeddings Bug

## Critical Issue
The system currently uses **mock embeddings** (random vectors) instead of real semantic embeddings, making semantic search completely non-functional.

## Location
File: `src/project_watch_mcp/neo4j_rag.py`

## Current Problem
- The `_generate_embedding()` method returns random 1536-dimensional vectors
- This makes semantic search return random/meaningless results
- The system pretends to work but semantic similarity is completely broken

## Fix Required
Replace mock implementation with real embedding generation:
1. Integrate an actual embedding model (e.g., OpenAI, sentence-transformers, or similar)
2. Generate real semantic embeddings based on code content
3. Ensure embeddings capture semantic meaning of code snippets

## Impact
- **Current**: Semantic search returns random results
- **After fix**: Semantic search will find conceptually similar code

## Priority
**CRITICAL** - This is the most important bug to fix. Without real embeddings, the semantic search feature is completely broken despite appearing to work.

## Notes from Research
- The strategic-research-analyst identified this as the only genuinely critical bug
- All other "improvements" are nice-to-haves compared to this fundamental issue
- System appears to work but returns meaningless results for semantic queries