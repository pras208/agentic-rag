# Agentic RAG POC

A minimal, proof-of-concept implementation of an agentic Retrieval-Augmented Generation (RAG) system with a web UI.

## Architecture Overview

```
┌─────────────────────────────┐
│      Browser UI (HTML)      │
│  - Upload documents         │
│  - Query with streaming     │
└──────────────┬──────────────┘
               │ HTTP
               ▼
        ┌──────────────┐
        │ Flask Server │
        └──────┬───────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
┌────────────┐ ┌──────────┐ ┌──────────────┐
│  Agent     │ │Document  │ │Embedding &   │
│Orchestrator│ │Processor │ │Vector Store  │
└────┬───────┘ └──────────┘ └──────────────┘
     │
     └────────────────────────┬──────────────────┐
                              │                  │
                              ▼                  ▼
                         ┌─────────┐        ┌──────────┐
                         │ FAISS   │        │ AWS      │
                         │ Index   │        │ Bedrock  │
                         └─────────┘        └──────────┘
```

## Components

### 1. **Document Processor** (`document/`)
- Extracts text from PDF, TXT, MD files
- Chunks documents into overlapping segments
- Handles async processing with thread pool

### 2. **Embedding & Vector Store** (`embedding/`)
- Sentence-Transformers for generating embeddings
- FAISS for in-memory vector search
- JSON metadata storage for document references

### 3. **LLM Integration** (`llm/`)
- AWS Bedrock wrapper for Claude 3.5 Sonnet
- Supports streaming responses

### 4. **Agent Orchestrator** (`agent/`)
- Simple reasoning loop (not using LangChain framework)
- Decides when to retrieve vs. answer directly
- Returns streamed responses to UI

### 5. **Web UI** (`templates/index.html`)
- Drag-and-drop document upload
- Real-time streaming query responses
- Document indexing status

## Setup

### Prerequisites
- Python 3.9+
- AWS Bedrock access (Claude 3.5 Sonnet model)
- AWS credentials configured locally

### Installation

```bash
# Clone and navigate
cd agentic-rag-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Configure AWS credentials in .env or ~/.aws/credentials
```

### Running the Application

```bash
# Start Flask server
python app.py

# Open browser
open http://localhost:5000
```

Server runs on `http://0.0.0.0:5000` by default.

## Usage

### 1. Upload Documents
- Click the upload area or drag-and-drop PDF/TXT/MD files
- Maximum 50MB per file
- Documents are processed asynchronously and indexed

### 2. Query Documents
- Type a question in the query box
- Press "Send Query" or Ctrl+Enter
- Response streams token-by-token from Claude

### 3. Architecture Flow
- Query is embedded using Sentence-Transformers
- FAISS searches for top-5 relevant chunks
- Chunks are sent as context to Claude
- Claude generates answer and streams back

## Configuration

Edit `config.yaml` to customize:

```yaml
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"  # 22MB, 384-dim vectors
  
vector_store:
  index_path: "indexes/faiss.index"
  
document_processing:
  chunk_size: 500  # tokens per chunk
  chunk_overlap: 50
  max_workers: 3  # concurrent uploads
  
bedrock:
  model_id: "anthropic.claude-3-5-sonnet-20241022"
  max_tokens: 2048
  temperature: 0.7
```

## Project Structure

```
agentic-rag-poc/
├── app.py                      # Flask server
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
│
├── agent/                      # Agent reasoning
│   ├── orchestrator.py         # Main agentic loop
│   └── prompts.py              # LLM prompts
│
├── document/                   # Document handling
│   └── processor.py            # Parser & chunking
│
├── embedding/                  # Vector operations
│   ├── embedder.py             # Sentence-Transformers wrapper
│   └── vector_store.py         # FAISS index wrapper
│
├── llm/                        # LLM integration
│   └── bedrock_client.py       # AWS Bedrock wrapper
│
├── templates/                  # Web UI
│   └── index.html              # Single-page app
│
└── uploads/                    # (Generated) Uploaded files
```

## Design Decisions

### Why FAISS over hosted vector DB?
- No external service needed (runs locally)
- Instant startup (no docker)
- Perfect for POC with <100k documents

### Why not use LangChain agents?
- Simpler to understand the reasoning flow
- No framework overhead
- Full control over tool selection

### Why Sentence-Transformers?
- 22MB model, runs on CPU
- Good quality (384-dim vectors)
- No API calls needed

### Why ThreadPoolExecutor?
- Simple async document processing
- No Redis/Celery complexity
- Handles 3-5 concurrent uploads gracefully

## Limitations & TODOs

- ❌ No multi-turn conversation memory
- ❌ No vector metadata filtering
- ❌ No streaming document uploads
- ❌ No error recovery/retry logic
- ❌ FAISS index not optimized for large scale (>100k docs)
- ❌ No authentication/multi-user support

## Quick Demo

1. **Create sample document** (`sample.md`):
```markdown
# AWS Bedrock Architecture

AWS Bedrock provides managed API access to foundation models.
The service handles scaling, security, and model updates.

## Key Features
- Model versioning
- Token-based billing
- Streaming responses
```

2. **Upload** the file via UI

3. **Ask questions**:
   - "What is AWS Bedrock?"
   - "What are the key features of Bedrock?"
   - "How does Bedrock handle billing?"

4. **Watch Claude** retrieve context and answer based on your document

## Performance Notes

- Embedding 500-word chunk: ~50ms (CPU)
- FAISS search (k=5): <10ms
- Bedrock latency: 1-2s (first token), then streaming
- Full response (500 tokens): 5-10s

## Troubleshooting

### "No such module: embedding"
```bash
# Ensure you're in the project root
python app.py  # Run from here, not subdirectories
```

### "FAISS index not found"
- Expected on first run, will be created after first document upload

### "AWS Bedrock error"
- Check AWS credentials: `aws sts get-caller-identity`
- Verify Bedrock access in your region

### "Connection refused on localhost:5000"
- Check if port 5000 is in use: `lsof -i :5000`
- Change port in `config.yaml` if needed

## Future Enhancements

- [ ] Multi-turn conversation with memory
- [ ] Hybrid search (semantic + keyword)
- [ ] Document metadata filtering
- [ ] Streaming document processing
- [ ] Query rewriting / query expansion
- [ ] Response quality scoring
- [ ] Fine-tuned embeddings for domain
- [ ] REST API for programmatic access
