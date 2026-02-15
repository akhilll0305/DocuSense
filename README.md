# DocuSense

Intelligent Document Q&A System using Retrieval-Augmented Generation (RAG)

## 🆓 100% FREE & Open-Source

**No API keys or credits required!** DocuSense uses completely free models:
- **LLM**: Ollama (Llama 3.2, Mistral, Phi-3) - runs locally
- **Embeddings**: Sentence Transformers - local embeddings
- **Vector Store**: FAISS - in-memory search
- **Backend**: FastAPI
- **Frontend**: Gradio
- **Deployment**: Modal.com (free tier available)

## 🚧 Project Status: Phase 0 - Setup Complete ✅

### Current Phase
- ✅ Project structure created
- ✅ Configuration management setup
- ✅ Logging infrastructure configured
- 🔄 Virtual environment setup (next step)

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for complete roadmap.

## Quick Start

### Prerequisites

1. **Install Ollama** (FREE local LLM runtime)
   ```powershell
   # Download from: https://ollama.ai/
   # Or via winget:
   winget install Ollama.Ollama
   
   # Pull a model (one-time, ~2GB):
   ollama pull llama3.2:3b
   ```

2. **Python 3.10+** (you have 3.13.0 ✅)

### 1. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate
```
No API keys needed! Default config uses FREE models.
# Optionally edit .env if you want to customize models.
```

### 4. Verify Ollama is Running

```powershell
# Check Ollama is running
ollama list

# Test Ollama
ollama run llama3.2:3b "Hello!"ll Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment

```powershell
# Copy example env file
copy .env.example .env

# Edit .env and add your API keys
notepad .env✅ Using FREE models: {settings.is_using_free_models}')"
```

---

## 🎯 Technology Stack (100% FREE)

### Core Components
- **LLM**: Ollama (Llama 3.2, Mistral, Phi-3, etc.)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS (CPU version)
- **Keyword Search**: BM25
- **Re-ranker**: Cross-encoder (ms-marco-MiniLM-L-6-v2)

### Infrastructure
- **Backend**: FastAPI
- **Frontend**: Gradio
- **Deployment**: Modal.com
- **Storage**: SQLite / Local files

### Why These Choices?
✅ **No cost** - Everything runs locally or uses free tiers  
✅ **No API keys** - No external dependencies  
✅ **Privacy** - Your documents stay on your machine  
✅ **Production-ready** - Can deploy to Modal.com free tier  
✅ **Fast** - Optimized models for speed  

---

### 4. Verify Setup

```powershell
# Run tests
pytest tests/

# Check if package imports correctly
python -c "from docusense.config import settings; print(f'Project: {settings.project_name}')"
```

## Project Structure

```
LLM COURSE PROJECT/
├── docusense/              # Main package
│   ├── llms/              # LLM provider abstractions
│   ├── retrieval/         # Vector stores, chunking, search
│   ├── agents/            # Query planning (Phase 5 - Paused)
│   ├── evaluation/        # Metrics & benchmarks
│   ├── api/               # REST API
│   ├── ui/                # Gradio interface
│   ├── utils/             # Utilities
│   └── config/            # Configuration
├── data/
│   ├── raw/               # Original documents
│   ├── processed/         # Chunked documents
│   └── vector_stores/     # FAISS indexes
├── tests/                 # Tests
├── logs/                  # Application logs
└── PROJECT_PLAN.md        # Detailed roadmap
```

## Next Steps

Moving to **Phase 1**: Knowledge Ingestion & Chunking
- Document loaders
- Text preprocessing
- Chunking strategies
- Chunk storage

## Development

```powershell
# Format code
black docusense/

# Lint
ruff check docusense/

# Type check
mypy docusense/
```

## Resources

- [Project Plan](PROJECT_PLAN.md) - Complete phase-by-phase guide
- [Configuration](.env.example) - Environment variables

---

Built as a learning project for LLM Engineering
