# Agentic RAG POC - Architecture Guide

## System Overview

This document details the architecture and data flow of the agentic RAG application.

## Components & Data Flow

### 1. Web UI → Flask Server

**Request**: User uploads file or submits query
**Response**: JSON acknowledgment or streaming tokens

```
Browser
  ├─ GET /              → Serve UI (index.html)
  ├─ POST /api/upload   → Upload document
  ├─ POST /api/query    → Submit question (returns Server-Sent Events stream)
  ├─ GET /api/status    → Get system status
  └─ GET /api/job-status/{id} → Check upload progress
```

### 2. Document Upload Pipeline

```
File Upload
    ↓
[app.py] /api/upload route
    ├─ Validate file type (.pdf, .txt, .md)
    ├─ Save to disk (uploads/)
    ├─ Assign job_id (UUID)
    └─ Start async processing
        ↓
    [document/processor.py] process_file_async()
        ├─ Extract text (_extract_text)
        │   ├─ For PDF: PyPDF2.PdfReader
        │   └─ For TXT/MD: read file
        ├─ Chunk text (_chunk_text)
        │   └─ Split by 500 words, 50-word overlap
        └─ Return chunks with metadata
        ↓
    [embedding/embedder.py] embed_batch()
        ├─ Load Sentence-Transformers model
        ├─ Encode chunks → 384-dim vectors
        └─ Return numpy array
        ↓
    [embedding/vector_store.py] add_documents()
        ├─ Add embeddings to FAISS index
        ├─ Store metadata (text, source, chunk_id)
        ├─ Save index to disk
        └─ Save metadata to JSON
```

**Example Flow**:
1. Upload `aws-guide.pdf`
2. Extract 2000 words → 5 chunks (500 words each)
3. Generate 5 embeddings (384-dim each)
4. Add to FAISS: index.ntotal = 5
5. Store metadata for each chunk

### 3. Query Processing Pipeline

```
User Query: "What is AWS Bedrock?"
    ↓
[app.py] /api/query route
    ├─ Receive JSON: {"query": "..."}
    └─ Initialize Server-Sent Events stream
        ↓
    [agent/orchestrator.py] reason()
        ├─ Parse query
        ├─ Decide: retrieve context or direct answer?
        │   └─ Use heuristic: keywords like "what", "how", "explain"
        │   └─ If documents indexed: retrieve context
        │
        ├─ [embedder.py] embed_text(query)
        │   └─ Generate 384-dim vector for query
        │
        ├─ [vector_store.py] search(query_embedding, k=5)
        │   ├─ FAISS finds top-5 nearest chunks
        │   ├─ Compute L2 distances
        │   └─ Return: [{text, distance, metadata}, ...]
        │
        ├─ Format prompt:
        │   ```
        │   Context from documents:
        │   - {chunk1}
        │   - {chunk2}
        │   - {chunk3}
        │   - {chunk4}
        │   - {chunk5}
        │   
        │   Question: What is AWS Bedrock?
        │   ```
        │
        ├─ [bedrock_client.py] invoke_stream(prompt)
        │   ├─ Call boto3.invoke_model_with_response_stream
        │   ├─ Send to Claude 3.5 Sonnet
        │   └─ Stream tokens back
        │
        └─ For each token:
            ├─ Yield to Server-Sent Events
            └─ Browser receives in real-time
```

**Example Tokens Streamed**:
```
data: AWS
data: Bedrock
data: is
data: a
data: fully
data: managed
...
data: [DONE]
```

## Core Modules

### `embedding/embedder.py`
- **Class**: `Embedder`
- **Model**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Methods**:
  - `embed_text(text: str) → ndarray` - Single text embedding
  - `embed_batch(texts: list) → ndarray` - Batch embeddings (32 at a time)

### `embedding/vector_store.py`
- **Class**: `VectorStore`
- **Backend**: FAISS (IndexFlatL2)
- **Methods**:
  - `add_documents(embeddings, metadatas)` - Add chunks to index
  - `search(query_embedding, k=5)` - Find top-k neighbors
  - `save()` / `_load_or_create()` - Persist to disk
  - `get_size()` - Number of indexed documents

### `document/processor.py`
- **Class**: `DocumentProcessor`
- **Features**: Async processing with ThreadPoolExecutor
- **Methods**:
  - `process_file_async(path, job_id)` - Main entry point
  - `_extract_text(path)` - Format-specific parsing
  - `_extract_pdf(path)` - PDF text extraction
  - `_chunk_text(text)` - Splitting with overlap
  - `get_job_status(job_id)` - Track progress

### `llm/bedrock_client.py`
- **Class**: `BedrockClient`
- **Service**: AWS Bedrock
- **Model**: Claude 3.5 Sonnet
- **Methods**:
  - `invoke(prompt, max_tokens, temperature)` - Non-streaming call
  - `invoke_stream(prompt, ...)` - Token streaming

### `agent/orchestrator.py`
- **Class**: `AgentOrchestrator`
- **Logic**: Simple decision tree (not LangChain framework)
- **Methods**:
  - `reason(query, max_iterations)` - Main reasoning loop
  - `_should_retrieve(query)` - Heuristic: keyword-based
  - `_retrieve_context(query, k=5)` - Vector search and format

### `app.py`
- **Framework**: Flask
- **Routes**:
  - GET `/` - Serve UI
  - POST `/api/upload` - Handle file upload
  - POST `/api/query` - Process question (streaming)
  - GET `/api/status` - System metrics
  - GET `/api/job-status/{id}` - Upload progress

## Data Structures

### FAISS Index State
```python
# In memory during app runtime
vector_store.index  # IndexFlatL2(384)
vector_store.metadata  # [
                       #   {"text": "chunk1", "source_file": "doc.pdf", ...},
                       #   {"text": "chunk2", "source_file": "doc.pdf", ...},
                       #   ...
                       # ]

# On disk
indexes/faiss.index        # Binary FAISS index file
indexes/metadata.json      # JSON list of chunk metadata
```

### Upload Job State
```python
document_processor.processing_jobs = {
    "job-uuid-1": {
        "status": "completed",
        "chunks": [...],
        "chunk_count": 5,
        "progress": 100
    },
    "job-uuid-2": {
        "status": "processing",
        "progress": 50
    }
}
```

### Vector Embedding (384-dim)
```python
# For text: "AWS Bedrock is a managed service"
embedding = ndarray(shape=(384,), dtype=float32)
# Example: [0.123, -0.456, 0.789, ..., -0.321]
```

## Performance Characteristics

### Latencies
| Operation | Time | Notes |
|-----------|------|-------|
| Embed 500-word chunk | ~50ms | CPU, Sentence-Transformers |
| FAISS search (k=5) | <10ms | In-memory L2 distance |
| Bedrock first token | 1-2s | API latency + model inference |
| Subsequent tokens | ~20-50ms | Streaming |
| Full response (500 tokens) | 5-10s | Total end-to-end |

### Memory Usage
| Component | Size |
|-----------|------|
| Sentence-Transformers model | ~150MB |
| FAISS index (10k docs) | ~50MB |
| App process baseline | ~200MB |
| **Total (10k docs)** | **~400MB** |

### Throughput
- Concurrent uploads: 3 (configurable)
- Queries: Sequential (agent processes one at a time)
- FAISS capacity: No hard limit (tested to 100M+ vectors)

## Configuration

**config.yaml** controls:
```yaml
embedding.model_name        # Which Sentence-Transformers model
embedding.dimension         # Vector dimensionality (384 for all-MiniLM)
vector_store.index_path     # Where to save FAISS index
document_processing.chunk_size    # Tokens per chunk
document_processing.max_workers   # Concurrent upload threads
bedrock.model_id            # Which Claude model
bedrock.max_tokens          # Output length limit
bedrock.temperature         # Randomness (0.0 = deterministic, 1.0 = creative)
```

## Failure Scenarios & Recovery

### Document Upload Fails
- Status: "failed" with error message
- Original file remains in `uploads/`
- No index corruption (transaction-like)
- Recovery: Re-upload file

### FAISS Index Corruption
- Next startup detects missing metadata.json
- Index automatically recreated (empty)
- Can reindex documents by re-uploading

### Bedrock Timeout
- Boto3 retries automatically (3x default)
- Stream may end early
- UI shows partial response

### Query with No Documents
- Agent detects empty index (get_size() == 0)
- Falls back to direct LLM query (no retrieval)
- Still provides response, just not grounded

## Trade-offs & Design Choices

### Why FAISS?
✅ Pros: No server, instant startup, CPU-friendly
❌ Cons: All vectors in memory, no filtering

### Why ThreadPoolExecutor?
✅ Pros: Simple, built-in, no external deps
❌ Cons: Not distributed, Python GIL

### Why Sentence-Transformers?
✅ Pros: Small (22MB), fast (CPU), good quality
❌ Cons: General-purpose (not domain-fine-tuned)

### Why simple decision heuristic?
✅ Pros: Transparent, debuggable, no ML overhead
❌ Cons: Not learned from data, brittle on edge cases

## Future Scaling

To handle 10M+ documents:
1. Replace FAISS with: Pinecone, Weaviate, or Qdrant (hosted)
2. Replace ThreadPoolExecutor with: Celery + Redis (distributed)
3. Replace embedder with: Fine-tuned model for domain
4. Replace decision heuristic with: Learned classifier
5. Add: Query rewriting, semantic routing, re-ranking

For now, this POC scales to ~100k documents locally.
