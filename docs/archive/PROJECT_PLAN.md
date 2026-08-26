# DocuSense — Intelligent Document Q&A System

## 📋 Project Overview

**DocuSense** is a smart document-intelligence system that interprets user queries, retrieves relevant information from a document knowledge base, and synthesizes grounded answers with transparency and measurable quality.

### Core Objectives
- Build a production-quality RAG (Retrieval-Augmented Generation) system
- Learn by experimenting with different approaches
- Implement modern LLM engineering best practices
- Create a measurable, debuggable, and extensible system

### Key Technologies & Concepts
- Modern RAG architecture principles
- Hybrid retrieval strategies (vector + keyword)
- Structured decision pipelines
- Tool invocation and bounded agent behavior
- Evaluation and observability
- Multiple LLM provider integration

---

## 🎓 Course Learning Objectives Mapping

### A. LLM Fundamentals & APIs
- Calling LLM APIs
- Switching between providers
- Understanding latency, cost, limits

### B. Prompt Engineering
- System vs user prompts
- Structured outputs (JSON)
- Prompt iteration & failure handling

### C. Embeddings & Vector Databases
- Creating embeddings
- Chunking strategies
- Vector search basics

### D. Retrieval-Augmented Generation (RAG)
- Retrieval → context → generation
- Hybrid search
- Re-ranking

### E. Tool / Function Calling
- Defining tools
- LLM deciding which tool to use
- Controlled execution

### F. Agents & Reasoning Pipelines ⏸️ **(PAUSED)**
- Breaking tasks into steps
- Planning before answering
- Limiting autonomy

### G. Evaluation & Benchmarking
- Measuring answer quality
- Retrieval evaluation
- Comparing models & prompts

### H. UI & System Integration
- Simple UIs (Gradio)
- APIs
- End-to-end systems

---

## 📅 Phase-by-Phase Implementation Plan

---

## 🚀 Phase 0 — Architecture & Environment Setup

**Timeline:** 1-2 days  
**Goal:** Create a clean, maintainable foundation with no AI logic yet

### Why This Phase Matters
Setting up proper architecture prevents tangled code and enables easy debugging, model swapping, and feature additions later. This is the difference between a toy project and production-ready code.

### Tasks

#### 1. Project Structure
Create a modular directory structure:
```
docusense/
├── llms/              # LLM provider abstractions
├── retrieval/         # Vector stores, chunking, search
├── agents/            # Query planning & orchestration (Phase 5)
├── evaluation/        # Metrics, benchmarks, judges
├── api/               # REST API endpoints
├── ui/                # Gradio/Streamlit interface
├── utils/             # Shared utilities
└── config/            # Configuration management

data/
├── raw/               # Original documents
├── processed/         # Cleaned, chunked documents
└── vector_stores/     # Persisted embeddings

tests/                 # Unit and integration tests
logs/                  # Application logs
```

#### 2. Environment Setup
- Create Python virtual environment (Python 3.10+)
- Set up dependency management (requirements.txt or Poetry)
- Configure IDE (VS Code with Python, Pylance extensions)
- Set up Git repository with .gitignore

#### 3. Configuration Management
- Create `.env` file for API keys and secrets
- Build `config.py` for centralized configuration
- Support multiple environments (dev, test, prod)
- Example config items:
  ```python
  # LLM providers
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  
  # Embedding models
  EMBEDDING_MODEL
  EMBEDDING_DIMENSION
  
  # Retrieval settings
  CHUNK_SIZE
  CHUNK_OVERLAP
  TOP_K_RESULTS
  
  # System settings
  LOG_LEVEL
  CACHE_ENABLED
  ```

#### 4. Logging & Error Tracking
- Set up structured logging (Python logging or loguru)
- Create log rotation strategy
- Add error tracking hooks
- Log format should include:
  - Timestamp
  - Module name
  - Log level
  - Function name
  - Message
  - Context data

#### 5. Dependency Installation
Initial dependencies:
```
python-dotenv       # Environment management
pydantic           # Data validation
loguru             # Enhanced logging
pytest             # Testing framework
black              # Code formatting
ruff               # Linting
```

### Deliverables
- ✅ Clean project structure
- ✅ Working virtual environment
- ✅ Configuration system
- ✅ Logging infrastructure
- ✅ Git repository initialized

### Success Criteria
- Can import modules from any package
- Environment variables load correctly
- Logs write to files with proper formatting
- Code passes linting checks

---

## 📚 Phase 1 — Knowledge Ingestion & Chunking

**Timeline:** 2-3 days  
**Goal:** Build a reliable "knowledge base" from raw documents

### Why This Phase Matters
The quality of your retrieval depends entirely on how well you prepare your data. Good chunking preserves semantic meaning while staying within embedding model limits. This phase is the foundation of your RAG system's accuracy.

### Learning Focus
- Why chunking size and strategy matter for retrieval quality
- How text preprocessing affects embedding quality
- Metadata preservation for citation and filtering
- Trade-offs between chunk size and retrieval precision

### Tasks

#### 1. Document Loaders
Build loaders for different formats (start with text, expand later):
- `.txt` files
- `.pdf` files (using PyPDF2 or pdfplumber)
- `.docx` files (using python-docx)
- Markdown files

Each loader should:
- Extract text content
- Preserve document metadata (filename, page numbers, sections)
- Handle encoding issues gracefully
- Report parsing errors without crashing

#### 2. Text Cleaning & Normalization
Create preprocessing pipeline:
- Remove excessive whitespace
- Normalize unicode characters
- Handle special characters (bullets, tables)
- Optionally remove headers/footers
- Preserve paragraph structure

**Don't over-clean:** Keep enough structure for meaningful embeddings.

#### 3. Chunking Strategy Implementation
Implement multiple chunking strategies to experiment:

##### a) Fixed-Size Chunking
- Chunk size: 500-1000 tokens (experiment)
- Overlap: 50-200 tokens
- Simple but can break semantic units

##### b) Semantic Chunking (Recommended)
- Split on paragraph boundaries
- Keep related sentences together
- Use NLP sentence segmentation
- More expensive but better quality

##### c) Sliding Window
- Fixed window with configurable stride
- Good for dense technical documents

**Chunk Metadata:** Each chunk should store:
```python
{
    "chunk_id": "doc_001_chunk_005",
    "text": "chunk content...",
    "document_id": "doc_001",
    "document_name": "whitepaper.pdf",
    "page_number": 5,
    "chunk_index": 5,
    "char_count": 847,
    "token_count": 203
}
```

#### 4. Storage Layer
Create a simple database for chunks (before embeddings):
- Option A: SQLite database with schema:
  ```sql
  CREATE TABLE chunks (
      id TEXT PRIMARY KEY,
      document_id TEXT,
      text TEXT,
      metadata JSON,
      created_at TIMESTAMP
  );
  ```
- Option B: JSON file structure (simpler for experimentation)

#### 5. Quality Checks
- Count chunks per document
- Verify chunk size distribution
- Check for empty or malformed chunks
- Validate metadata completeness

### Dependencies
```
pypdf2              # PDF parsing
python-docx         # Word documents
tiktoken            # Token counting
spacy              # Sentence segmentation (optional)
```

### Deliverables
- ✅ Document loading functions
- ✅ Configurable chunking pipeline
- ✅ Chunk storage system
- ✅ Metadata preservation
- ✅ Processing scripts for sample documents

### Success Criteria
- Can process 10+ documents without errors
- Chunks maintain semantic coherence
- Metadata is complete and queryable
- Can reproduce chunking with same settings

### Experiments to Try
1. Compare chunk sizes: 200 vs 500 vs 1000 tokens
2. Test overlap impact: 0% vs 10% vs 20%
3. Semantic vs fixed-size chunking quality

---

## 🧠 Phase 2 — Embeddings & Vector Store

**Timeline:** 2-3 days  
**Goal:** Build the semantic retrieval layer

### Why This Phase Matters
Embeddings transform text into numerical representations that capture semantic meaning. This enables "fuzzy" search where queries don't need exact keyword matches. The vector store is the engine of your RAG system.

### Learning Focus
- How embeddings represent semantic meaning
- Vector database indexing strategies
- Similarity search mechanics (cosine, dot product)
- Performance vs accuracy trade-offs
- Embedding model selection criteria

### Tasks

#### 1. Embedding Model Selection
Research and choose embedding model(s):

##### Option A: OpenAI Embeddings
- Model: `text-embedding-3-small` or `text-embedding-3-large`
- Pros: High quality, easy to use
- Cons: API cost, latency
- Dimensions: 1536 (small) or 3072 (large)

##### Option B: Open-Source Models
- Models: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`
- Pros: Free, local execution, no API calls
- Cons: Slightly lower quality, requires GPU for speed
- Use: `sentence-transformers` library

##### Option C: Cohere / Anthropic
- Experiment with provider diversity

**Recommendation:** Start with OpenAI for simplicity, add local models later for comparison.

#### 2. Embedding Generation Pipeline
Build embedding creation workflow:

```python
def create_embeddings(chunks: List[dict], model: str) -> List[dict]:
    """
    Generate embeddings for chunks.
    
    Returns:
        List of chunks with 'embedding' field added
    """
    # Batch processing for efficiency
    # Handle rate limits
    # Cache embeddings to avoid re-computation
    # Log progress
```

**Key considerations:**
- Batch embed chunks (e.g., 100 at a time)
- Implement retry logic for API failures
- Cache embeddings to disk
- Track embedding costs
- Add progress bars for long operations

#### 3. Vector Store Implementation
Set up vector database for similarity search:

##### Option A: FAISS (Recommended for Learning)
```python
import faiss
import numpy as np

# Create index
dimension = 1536  # OpenAI embedding size
index = faiss.IndexFlatL2(dimension)  # Start simple

# Add vectors
vectors = np.array([chunk['embedding'] for chunk in chunks])
index.add(vectors)

# Search
query_vector = get_embedding(query)
distances, indices = index.search(query_vector, k=5)
```

**Pros:** Fast, no external dependencies, good for learning  
**Cons:** In-memory only (unless using disk persistence)

##### Option B: Chroma
- Easier metadata filtering
- Built-in persistence
- Good for smaller datasets

##### Option C: Milvus
- Production-ready
- Distributed capabilities
- Overkill for learning phase

#### 4. Metadata Storage
Keep chunk metadata separate from vectors:
- Store text and metadata in SQLite/JSON
- Store only embeddings in vector index
- Link by chunk ID

This enables:
- Efficient vector search
- Rich metadata filtering
- Easy text retrieval

#### 5. Search Implementation
Build search function:

```python
def semantic_search(
    query: str,
    top_k: int = 5,
    filters: dict = None
) -> List[dict]:
    """
    Perform semantic similarity search.
    
    Returns:
        List of {chunk, score, metadata}
    """
    # 1. Embed query
    # 2. Search vector index
    # 3. Retrieve chunk text/metadata
    # 4. Apply filters
    # 5. Return ranked results
```

#### 6. Persistence
Implement save/load for vector store:
- Save FAISS index to disk
- Save metadata database
- Load on application startup
- Version your indexes

### Dependencies
```
openai                    # OpenAI embeddings
sentence-transformers     # Local embeddings
faiss-cpu                 # Vector search
numpy                     # Array operations
chromadb                  # Alternative vector DB (optional)
```

### Deliverables
- ✅ Embedding generation pipeline
- ✅ Vector store with indexed embeddings
- ✅ Semantic search function
- ✅ Persistence layer
- ✅ Performance benchmarks

### Success Criteria
- Can embed 1000+ chunks efficiently
- Search returns relevant results in <100ms
- Can reload index from disk
- Metadata filtering works correctly

### Experiments to Try
1. **Embedding models:** Compare OpenAI vs local models
2. **Index types:** FAISS Flat vs IVF vs HNSW
3. **Distance metrics:** Cosine vs L2 vs dot product
4. **Top-k values:** How does k=3 vs k=10 affect results?

### Example Queries to Test
- "How do I install the software?"
- "What are the system requirements?"
- "Explain the authentication process"
- "troubleshooting connection errors"

---

## 🔍 Phase 3 — Hybrid Retrieval

**Timeline:** 3-4 days  
**Goal:** Improve search quality by combining semantic and keyword search

### Why This Phase Matters
Vector search alone misses exact keyword matches (names, codes, specific terms). Keyword search alone misses semantic similarity. Hybrid retrieval combines both, dramatically improving accuracy, especially for technical documents with specific terminology.

### Learning Focus
- Why vector search and keyword search complement each other
- Score normalization across different retrievers
- Re-ranking strategies and their impact
- Ensemble retrieval patterns

### Tasks

#### 1. BM25 Keyword Search Implementation
Implement traditional keyword-based retrieval:

```python
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, chunks: List[dict]):
        # Tokenize all chunks
        self.chunks = chunks
        tokenized_corpus = [chunk['text'].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        # Tokenize query
        # Get BM25 scores
        # Return top-k chunks with scores
```

**BM25 strengths:**
- Exact term matching
- Acronyms and codes (e.g., "API-2024-XYZ")
- Proper nouns
- Low-frequency important terms

#### 2. Score Normalization
Different retrievers produce incomparable scores:
- FAISS returns distances (lower = better)
- BM25 returns relevance scores (higher = better)

Implement normalization strategies:

##### Min-Max Normalization
```python
def normalize_scores(scores: List[float]) -> List[float]:
    min_score = min(scores)
    max_score = max(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]
```

##### Z-Score Normalization
```python
def z_normalize(scores: List[float]) -> List[float]:
    mean = np.mean(scores)
    std = np.std(scores)
    return [(s - mean) / std for s in scores]
```

#### 3. Hybrid Fusion Strategies
Combine results from multiple retrievers:

##### Reciprocal Rank Fusion (RRF) - Recommended
```python
def reciprocal_rank_fusion(
    results_list: List[List[dict]],
    k: int = 60
) -> List[dict]:
    """
    Combine ranked lists without needing score normalization.
    
    Score = sum(1 / (k + rank))
    """
    scores = defaultdict(float)
    for results in results_list:
        for rank, result in enumerate(results, 1):
            scores[result['chunk_id']] += 1 / (k + rank)
    
    # Sort by combined score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

##### Weighted Score Fusion
```python
def weighted_fusion(
    vector_results: List[dict],
    bm25_results: List[dict],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> List[dict]:
    """
    Combine with weighted scores (requires normalization).
    """
    # Normalize both sets of scores
    # Combine: final_score = w1*vec_score + w2*bm25_score
    # Deduplicate and sort
```

#### 4. Retrieval Pipeline
Build unified retrieval interface:

```python
class HybridRetriever:
    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        fusion_method: str = 'rrf'
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion_method = fusion_method
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7
    ) -> List[dict]:
        """
        Hybrid retrieval with configurable fusion.
        """
        # 1. Get vector results
        # 2. Get BM25 results
        # 3. Fuse results
        # 4. Return top-k
```

#### 5. Re-Ranking Layer (Optional but Recommended)
Improve result ordering with re-ranking:

##### Cross-Encoder Re-Ranking
```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        chunks: List[dict],
        top_k: int = 5
    ) -> List[dict]:
        """
        Re-score chunks based on query-chunk relevance.
        """
        # Create query-chunk pairs
        # Score with cross-encoder
        # Sort by new scores
        # Return top-k
```

**Why re-ranking works:**
- Cross-encoders see query + chunk together
- More accurate than bi-encoders (embeddings)
- Expensive, so only re-rank top candidates

##### LLM-Based Re-Ranking
```python
def llm_rerank(query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
    """
    Use LLM to score relevance (expensive but flexible).
    """
    prompt = f"""
    Query: {query}
    
    Rate relevance of each chunk (0-10):
    {format_chunks(chunks)}
    
    Return JSON: [{{"chunk_id": "...", "score": 8}}, ...]
    """
    # Get scores from LLM
    # Re-sort chunks
```

#### 6. Experiment Framework
Build system to compare retrieval strategies:

```python
class RetrievalExperiment:
    def run_experiment(
        self,
        queries: List[str],
        strategies: List[dict]
    ):
        """
        Compare: vector-only, bm25-only, hybrid, hybrid+rerank
        """
        results = {}
        for strategy in strategies:
            results[strategy['name']] = self.evaluate_strategy(
                queries, strategy
            )
        return results
```

### Dependencies
```
rank-bm25              # BM25 implementation
sentence-transformers  # Cross-encoder reranking
nltk                   # Tokenization
scikit-learn          # Score normalization utilities
```

### Deliverables
- ✅ BM25 keyword search implementation
- ✅ Score normalization functions
- ✅ Hybrid fusion (RRF + weighted)
- ✅ Optional re-ranking layer
- ✅ Unified retrieval API
- ✅ Comparison experiments

### Success Criteria
- Hybrid retrieval outperforms vector-only on test queries
- Can toggle between fusion strategies
- Re-ranking improves top-3 precision
- System handles edge cases (no results, tie scores)

### Experiments to Try
1. **Vector-only vs BM25-only vs Hybrid**
   - Measure precision@k for each
   
2. **Fusion methods**
   - RRF vs weighted fusion
   - Optimal weight combinations
   
3. **Re-ranking impact**
   - With/without re-ranking
   - Cross-encoder vs LLM re-ranking
   
4. **Retrieval depth**
   - Retrieve 20, re-rank to 5
   - Retrieve 50, re-rank to 10
   
5. **Query types**
   - Factual: "What is the price?"
   - Conceptual: "How does authentication work?"
   - Keyword-heavy: "API error code 404"

### Sample Test Queries
Create a test set covering different query types:
- Exact term queries (product names, codes)
- Semantic queries (conceptual questions)
- Multi-hop queries (requires multiple chunks)
- Ambiguous queries (multiple valid interpretations)

---

## 🧠 Phase 4 — Query Understanding & Planner

**Timeline:** 3-4 days  
**Goal:** Add intelligence layer to interpret and optimize queries before retrieval

### Why This Phase Matters
Users rarely ask perfect questions. Query understanding transforms messy natural language into structured, retrievable queries. The planner decides the best strategy for each query type, improving both accuracy and efficiency.

### Learning Focus
- Structured JSON outputs from LLMs
- Prompt design for classification tasks
- Query rewriting techniques
- Decision pipeline architecture
- Few-shot prompting for consistency

### Tasks

#### 1. Intent Classification System
Build LLM-based intent classifier:

```python
class IntentClassifier:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.intent_types = [
            "factual_question",      # Who, what, when, where
            "conceptual_question",   # How, why, explain
            "procedural_question",   # Steps, instructions
            "comparison_question",   # Difference between X and Y
            "troubleshooting",       # Error, problem, fix
            "definition_request",    # What is, define
            "example_request"        # Show me, give example
        ]
    
    def classify(self, query: str) -> dict:
        """
        Classify query intent using LLM.
        
        Returns:
            {
                "intent": "factual_question",
                "confidence": 0.95,
                "reasoning": "Query asks for specific fact"
            }
        """
```

**Prompt template for intent classification:**
```
You are a query classifier for a document Q&A system.

Classify the user query into one of these categories:
- factual_question: Asking for specific facts (who, what, when, where)
- conceptual_question: Seeking understanding (how, why, explain)
- procedural_question: Requesting steps or instructions
- comparison_question: Comparing two or more things
- troubleshooting: Reporting error or seeking solution
- definition_request: Asking for term definition
- example_request: Requesting examples or demonstrations

Query: "{query}"

Return JSON:
{{
    "intent": "<category>",
    "confidence": <0-1>,
    "reasoning": "<brief explanation>"
}}
```

#### 2. Query Rewriting Module
Transform user queries into better retrieval queries:

```python
class QueryRewriter:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def rewrite(self, query: str, intent: str) -> dict:
        """
        Rewrite query for better retrieval.
        
        Returns:
            {
                "original": "...",
                "rewritten": "...",
                "expansions": [...],
                "keywords": [...]
            }
        """
```

**Query rewriting strategies:**

##### a) Clarification & Expansion
```
Original: "How to install?"
Rewritten: "How to install the software? What are the installation steps?"
```

##### b) Keyword Extraction
```
Original: "Why isn't my login working?"
Keywords: ["login", "authentication", "not working", "troubleshooting"]
```

##### c) Decomposition (for complex queries)
```
Original: "Compare authentication methods and explain which is more secure"
Sub-queries:
  1. "What authentication methods are available?"
  2. "Security comparison of authentication methods"
```

##### d) Term Normalization
```
Original: "API auth isn't working"
Rewritten: "API authentication is not working"
```

**Rewriting prompt template:**
```
Rewrite this query to improve document retrieval:

Original query: "{query}"
Intent: {intent}

Rewriting goals:
1. Expand abbreviations and acronyms
2. Add context if ambiguous
3. Extract key terms
4. Rephrase for clarity

Return JSON:
{{
    "rewritten_query": "<improved version>",
    "key_terms": ["term1", "term2"],
    "expansions": ["alternative phrasing 1", "..."]
}}
```

#### 3. Retrieval Strategy Planner
Decide which retrieval approach to use:

```python
class RetrievalPlanner:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def plan(self, query: str, intent: str, metadata: dict) -> dict:
        """
        Decide retrieval strategy based on query characteristics.
        
        Returns:
            {
                "strategy": "hybrid",
                "parameters": {
                    "top_k": 10,
                    "use_reranking": true,
                    "vector_weight": 0.7
                },
                "reasoning": "Query contains specific terms..."
            }
        """
```

**Strategy selection logic:**

| Query Type | Recommended Strategy | Reasoning |
|------------|---------------------|-----------|
| Factual (names, codes) | BM25-heavy hybrid (0.3/0.7) | Exact terms matter |
| Conceptual | Vector-heavy hybrid (0.8/0.2) | Semantic understanding key |
| Procedural | Hybrid + re-ranking | Need sequential chunks |
| Troubleshooting | Vector + keyword | Mix of concepts and error codes |

**Planning prompt:**
```
You are a retrieval strategy planner.

Query: "{query}"
Intent: {intent}

Available strategies:
1. vector_only: Pure semantic search
2. keyword_only: BM25 keyword search
3. hybrid_balanced: 50/50 vector and keyword
4. hybrid_semantic: 70/30 vector and keyword
5. hybrid_keyword: 30/70 vector and keyword

Consider:
- Does query contain specific terms/codes? → prefer keyword
- Is query conceptual/semantic? → prefer vector
- Is high precision critical? → add re-ranking

Return JSON:
{{
    "strategy": "<strategy_name>",
    "top_k": <number>,
    "use_reranking": <true/false>,
    "vector_weight": <0-1>,
    "reasoning": "<explanation>"
}}
```

#### 4. Unified Query Processing Pipeline
Combine all components:

```python
class QueryProcessor:
    def __init__(self, llm_client):
        self.classifier = IntentClassifier(llm_client)
        self.rewriter = QueryRewriter(llm_client)
        self.planner = RetrievalPlanner(llm_client)
    
    def process(self, query: str) -> dict:
        """
        Full query understanding pipeline.
        
        Returns:
            {
                "original_query": "...",
                "intent": {...},
                "rewritten": {...},
                "plan": {...}
            }
        """
        # Step 1: Classify intent
        intent = self.classifier.classify(query)
        
        # Step 2: Rewrite query
        rewritten = self.rewriter.rewrite(query, intent['intent'])
        
        # Step 3: Plan retrieval strategy
        plan = self.planner.plan(
            query=rewritten['rewritten_query'],
            intent=intent['intent'],
            metadata={'original': query}
        )
        
        return {
            'original_query': query,
            'intent': intent,
            'rewritten': rewritten,
            'plan': plan,
            'timestamp': datetime.now().isoformat()
        }
```

#### 5. Structured Output Validation
Ensure LLM outputs are valid:

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class IntentClassification(BaseModel):
    intent: Literal[
        "factual_question",
        "conceptual_question",
        "procedural_question",
        "comparison_question",
        "troubleshooting",
        "definition_request",
        "example_request"
    ]
    confidence: float = Field(ge=0, le=1)
    reasoning: str

class QueryRewrite(BaseModel):
    rewritten_query: str
    key_terms: List[str]
    expansions: List[str]

class RetrievalPlan(BaseModel):
    strategy: str
    top_k: int = Field(ge=1, le=50)
    use_reranking: bool
    vector_weight: float = Field(ge=0, le=1)
    reasoning: str
```

Use Pydantic models with LLM JSON mode:
- Validates structure automatically
- Catches malformed responses
- Provides type safety

#### 6. Prompt Iteration & Testing
Build prompt evaluation system:

```python
class PromptTester:
    def test_classification(self, test_cases: List[dict]):
        """
        Test intent classification on known examples.
        
        test_cases = [
            {"query": "...", "expected_intent": "factual_question"},
            ...
        ]
        """
        results = []
        for case in test_cases:
            predicted = self.classifier.classify(case['query'])
            results.append({
                'query': case['query'],
                'expected': case['expected_intent'],
                'predicted': predicted['intent'],
                'correct': predicted['intent'] == case['expected_intent']
            })
        
        accuracy = sum(r['correct'] for r in results) / len(results)
        return accuracy, results
```

**Create test sets:**
- 20-30 queries per intent type
- Include edge cases
- Track prompt version and performance

### Dependencies
```
openai / anthropic     # LLM for query processing
pydantic              # Structured output validation
tenacity              # Retry logic
```

### Deliverables
- ✅ Intent classification system
- ✅ Query rewriting module
- ✅ Retrieval strategy planner
- ✅ Unified query processing pipeline
- ✅ Structured output validation
- ✅ Prompt testing framework

### Success Criteria
- Intent classification accuracy >85% on test set
- Query rewrites improve retrieval (measured in Phase 7)
- Strategy planner chooses appropriate approach
- All outputs conform to JSON schemas
- Pipeline handles malformed queries gracefully

### Experiments to Try
1. **Intent classification prompts**
   - Zero-shot vs few-shot
   - Different LLM models (GPT-3.5 vs GPT-4)
   
2. **Rewriting strategies**
   - Simple expansion vs decomposition
   - Impact on retrieval quality
   
3. **Planning accuracy**
   - Does LLM choose optimal strategy?
   - Manual vs automated planning comparison
   
4. **Prompt engineering**
   - System prompt variations
   - Few-shot example selection
   - Chain-of-thought prompting

### Example Test Cases
```python
test_queries = [
    {
        "query": "What is OAuth?",
        "expected_intent": "definition_request",
        "expected_strategy": "hybrid_semantic"
    },
    {
        "query": "Error code 500 when calling API",
        "expected_intent": "troubleshooting",
        "expected_strategy": "hybrid_keyword"
    },
    {
        "query": "How do I set up authentication?",
        "expected_intent": "procedural_question",
        "expected_strategy": "hybrid_balanced"
    }
]
```

---

## 🤖 Phase 5 — Agentic Decision Logic (Bounded Agent)

**Status:** ⏸️ **PAUSED** — Will be implemented after completing agent module in course

**Timeline:** 4-5 days (when resumed)  
**Goal:** Give LLM limited agency to orchestrate retrieval steps dynamically

### Why This Phase Matters
Static RAG pipelines can't adapt to complex queries requiring multi-step reasoning. Bounded agents can break down questions, decide which tools to use, and synthesize information from multiple sources — while staying controlled and transparent.

### Learning Focus
- Breaking tasks into executable steps
- Tool definition and invocation
- Planning before answering
- Limiting agent autonomy (safety)
- Agent observability and debugging

### Tasks (To Be Completed Later)

#### 1. Tool Definition System
Convert retrieval functions into LLM-callable tools:

```python
tools = [
    {
        "name": "semantic_search",
        "description": "Search documents using semantic similarity",
        "parameters": {
            "query": "search query string",
            "top_k": "number of results (default: 5)"
        }
    },
    {
        "name": "keyword_search",
        "description": "Search for exact keyword matches",
        "parameters": {
            "query": "search query",
            "top_k": "number of results"
        }
    },
    {
        "name": "hybrid_search",
        "description": "Combined semantic and keyword search",
        "parameters": {
            "query": "search query",
            "vector_weight": "0-1, weight for semantic search"
        }
    }
]
```

#### 2. Agent Planning Module
LLM generates execution plan:

```python
class BoundedAgent:
    def plan(self, query: str) -> List[dict]:
        """
        Generate step-by-step plan.
        
        Returns:
            [
                {"step": 1, "tool": "semantic_search", "args": {...}},
                {"step": 2, "tool": "keyword_search", "args": {...}},
                {"step": 3, "tool": "synthesize", "args": {...}}
            ]
        """
```

#### 3. Tool Execution Engine
Execute plan steps with safety controls:
- Max steps limit (e.g., 5 steps)
- Timeout per step
- Tool whitelist
- Output validation

#### 4. Multi-Step Reasoning
Enable agent to:
- Search multiple times with different strategies
- Combine results
- Ask clarifying questions (simulated)
- Decide when enough information is gathered

#### 5. Observability
Track agent behavior:
- Log each decision
- Record tool calls and results
- Measure plan quality
- Detect failure patterns

### Deliverables (When Resumed)
- ✅ Tool definition framework
- ✅ Agent planning system
- ✅ Controlled execution engine
- ✅ Multi-step reasoning capability
- ✅ Safety constraints
- ✅ Execution logging

---

## 🧠 Phase 6 — Answer Synthesis & Model Orchestration

**Timeline:** 3-4 days  
**Goal:** Generate accurate, grounded answers with citations

### Why This Phase Matters
Retrieval without good generation wastes all prior work. This phase ensures answers are accurate, grounded in retrieved context, properly cited, and don't hallucinate. Model orchestration optimizes cost and latency by using the right model for each task.

### Learning Focus
- Context injection into prompts
- Controlling hallucinations through grounding
- Citation and attribution strategies
- Structured answer formatting
- Multi-model orchestration patterns
- Fallback strategies

### Tasks

#### 1. Answer Synthesis Prompt Design
Create sophisticated answer generation prompt:

```python
class AnswerGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.synthesis_prompt_template = """
You are a helpful AI assistant answering questions based on provided documents.

**CRITICAL RULES:**
1. Only use information from the provided context
2. If context doesn't contain the answer, say so explicitly
3. Cite sources using [doc_id] format
4. Be concise but complete
5. If information conflicts, note the discrepancy

**User Query:**
{query}

**Retrieved Context:**
{context}

**Task:**
Provide a clear, accurate answer. Include citations for each claim.

**Answer Format:**
- Direct answer first
- Supporting details with citations
- Confidence level (high/medium/low)
"""
```

**Key prompt elements:**
- **Grounding instruction:** Forces model to use only provided context
- **Citation requirement:** Enables verification
- **Uncertainty handling:** Model should admit when it doesn't know
- **Format specification:** Ensures consistent structure

#### 2. Context Formatting
Prepare retrieved chunks for prompt injection:

```python
def format_context(chunks: List[dict]) -> str:
    """
    Format retrieved chunks into prompt context.
    
    Returns formatted string like:
    ---
    [Document 1] (score: 0.89)
    Source: technical_manual.pdf, page 5
    Content: "Authentication is handled via OAuth 2.0..."
    
    [Document 2] (score: 0.82)
    Source: api_guide.pdf, page 12
    Content: "To authenticate, send a POST request..."
    ---
    """
    formatted_chunks = []
    for i, chunk in enumerate(chunks, 1):
        formatted = f"""
[Document {i}] (relevance: {chunk['score']:.2f})
Source: {chunk['document_name']}, page {chunk.get('page', 'N/A')}
Content: "{chunk['text']}"
        """.strip()
        formatted_chunks.append(formatted)
    
    return "\n\n".join(formatted_chunks)
```

**Context optimization strategies:**
- **Truncation:** If context exceeds token limit, keep highest-scoring chunks
- **Deduplication:** Remove near-duplicate chunks
- **Ordering:** Place most relevant chunks first (or last, experiment)
- **Metadata inclusion:** Add document metadata for richer citations

#### 3. Structured Answer Schema
Define consistent answer format:

```python
from pydantic import BaseModel
from typing import List, Literal

class Citation(BaseModel):
    document_id: str
    document_name: str
    page_number: int = None
    chunk_text: str
    relevance_score: float

class Answer(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    confidence: Literal["high", "medium", "low"]
    reasoning: str
    has_answer: bool  # False if no relevant info found
    tokens_used: int
    latency_ms: float
    model_used: str
```

**Benefits:**
- Consistent output structure
- Easy to log and analyze
- Frontend-friendly format
- Type-safe

#### 4. Citation Extraction
Automatically extract and validate citations:

```python
def extract_citations(answer_text: str, chunks: List[dict]) -> List[dict]:
    """
    Find [Document X] references in answer and map to chunks.
    """
    import re
    
    citation_pattern = r'\[Document (\d+)\]'
    cited_indices = re.findall(citation_pattern, answer_text)
    
    citations = []
    for idx in cited_indices:
        idx = int(idx) - 1  # Convert to 0-indexed
        if idx < len(chunks):
            chunk = chunks[idx]
            citations.append({
                'document_id': chunk['document_id'],
                'document_name': chunk['document_name'],
                'page': chunk.get('page_number'),
                'text_snippet': chunk['text'][:200]
            })
    
    return citations
```

**Advanced citation strategies:**
- **Inline citations:** "OAuth 2.0 is used [1]"
- **Footnote style:** Numbered references at end
- **Direct quotes:** Highlight exact text from source
- **Multi-source synthesis:** "According to [1] and [3]..."

#### 5. Model Orchestration
Use different models for different tasks:

```python
class ModelOrchestrator:
    def __init__(self):
        self.models = {
            'planning': 'gpt-4',              # Best reasoning
            'classification': 'gpt-3.5-turbo', # Fast, cheap
            'synthesis': 'gpt-4',              # Quality answers
            'reranking': 'gpt-3.5-turbo'       # Good enough
        }
    
    def get_model_for_task(self, task: str) -> str:
        return self.models.get(task, 'gpt-3.5-turbo')
    
    def estimate_cost(self, task: str, tokens: int) -> float:
        """Calculate cost based on model pricing."""
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
            'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002}
        }
        # Calculate cost...
```

**Orchestration strategies:**

| Task | Recommended Model | Reasoning |
|------|------------------|-----------|
| Query classification | GPT-3.5 / Claude Haiku | Fast, cheap, simple task |
| Intent understanding | GPT-4 / Claude Sonnet | Better comprehension |
| Answer synthesis | GPT-4 / Claude Sonnet | Quality matters |
| Re-ranking | GPT-3.5 | Comparative task, good enough |
| Complex reasoning | GPT-4 / Claude Opus | Hard problems only |

#### 6. Hallucination Prevention
Techniques to keep answers grounded:

##### a) Explicit Grounding
```python
synthesis_prompt = """
ONLY use information from the context below.
If the answer is not in the context, respond:
"I don't have enough information to answer this question."

DO NOT use any external knowledge.
"""
```

##### b) Verification Step
```python
def verify_answer(answer: str, context: str, query: str) -> dict:
    """
    Use LLM to check if answer is grounded in context.
    """
    verification_prompt = f"""
    Query: {query}
    Context: {context}
    Answer: {answer}
    
    Is this answer fully supported by the context?
    Return JSON:
    {{
        "is_grounded": true/false,
        "unsupported_claims": [],
        "confidence": 0-1
    }}
    """
    # Get verification result
    # Flag potentially hallucinated answers
```

##### c) Attribution Enforcement
```python
# Require model to quote directly
prompt = """
For each claim:
1. Quote the exact text from context
2. Cite the document
3. Then explain in your own words
"""
```

#### 7. Answer Quality Metrics
Track answer generation performance:

```python
class AnswerMetrics:
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'answers_provided': 0,
            'no_answer_count': 0,
            'avg_latency_ms': 0,
            'avg_tokens': 0,
            'avg_cost': 0,
            'citation_rate': 0  # % of answers with citations
        }
    
    def log_answer(self, result: Answer):
        # Update metrics
        self.metrics['total_queries'] += 1
        if result.has_answer:
            self.metrics['answers_provided'] += 1
        # ... update other metrics
```

#### 8. Fallback Strategies
Handle failure cases gracefully:

```python
class AnswerPipeline:
    def generate_answer(self, query: str, chunks: List[dict]) -> Answer:
        try:
            # Primary: GPT-4 synthesis
            return self.synthesize_with_gpt4(query, chunks)
        except Exception as e:
            logger.error(f"GPT-4 synthesis failed: {e}")
            
            try:
                # Fallback 1: GPT-3.5
                return self.synthesize_with_gpt35(query, chunks)
            except Exception as e2:
                logger.error(f"GPT-3.5 synthesis failed: {e2}")
                
                # Fallback 2: Template-based response
                return self.template_answer(query, chunks)
```

**Fallback scenarios:**
- API failures
- Rate limiting
- Timeout errors
- Malformed outputs
- No relevant chunks retrieved

#### 9. Answer Types & Templates
Different formats for different query types:

##### Factual Answers
```
**Answer:** OAuth 2.0 [Document 1]

**Details:**
The system uses OAuth 2.0 for authentication [Document 1, page 5].
Supported grant types include authorization code and client credentials [Document 2].

**Sources:**
- technical_manual.pdf, page 5
- api_guide.pdf, page 12

**Confidence:** High
```

##### Procedural Answers
```
**Steps:**
1. Install dependencies [Document 1]
2. Configure environment variables [Document 2]
3. Run initialization script [Document 1]

**Prerequisites:**
- Python 3.8+ [Document 1]
- Database access [Document 3]

**Sources:** ...
```

##### Comparison Answers
```
**Comparison: OAuth vs API Keys**

| Aspect | OAuth 2.0 | API Keys |
|--------|-----------|----------|
| Security | High [Doc 1] | Medium [Doc 2] |
| Complexity | High [Doc 1] | Low [Doc 2] |
| Use Case | User auth [Doc 1] | Service auth [Doc 2] |

**Recommendation:** Use OAuth for user-facing apps [Document 1]
```

### Dependencies
```
openai / anthropic      # Answer generation
tiktoken               # Token counting
pydantic              # Answer schemas
```

### Deliverables
- ✅ Answer synthesis prompts
- ✅ Context formatting system
- ✅ Citation extraction
- ✅ Structured answer schema
- ✅ Model orchestration layer
- ✅ Hallucination prevention
- ✅ Fallback mechanisms
- ✅ Answer quality metrics

### Success Criteria
- Answers are grounded in retrieved context
- >90% of answers include proper citations
- No hallucination in spot checks
- Latency <3 seconds for typical queries
- Handles "no answer available" gracefully
- Different models used appropriately

### Experiments to Try
1. **Prompt variations**
   - Compare grounding instruction formats
   - Test citation format preferences
   
2. **Model comparison**
   - GPT-3.5 vs GPT-4 answer quality
   - Cost vs quality trade-offs
   
3. **Context optimization**
   - Top-3 vs top-5 vs top-10 chunks
   - Chunk ordering impact
   
4. **Citation styles**
   - Inline vs footnote
   - User preference testing

### Example Queries to Test
```python
test_cases = [
    {
        "query": "How do I authenticate API requests?",
        "expected": "Step-by-step answer with citations",
        "type": "procedural"
    },
    {
        "query": "What is the rate limit?",
        "expected": "Direct factual answer",
        "type": "factual"
    },
    {
        "query": "Why am I getting error 403?",
        "expected": "Troubleshooting with diagnostics",
        "type": "troubleshooting"
    }
]
```

---

## 📊 Phase 7 — Evaluation & Observability

**Timeline:** 4-5 days  
**Goal:** Measure system quality scientifically and make data-driven improvements

### Why This Phase Matters
Without evaluation, you're flying blind. This phase transforms your project from "seems to work" to "provably effective". You'll measure both retrieval quality (are you finding the right chunks?) and generation quality (are answers good?). This is what separates hobbyist projects from professional systems.

### Learning Focus
- Retrieval evaluation metrics (precision, recall, MRR, NDCG)
- Answer quality assessment (LLM-as-judge, human eval)
- A/B testing different system configurations
- Performance benchmarking
- Systematic failure analysis

### Tasks

#### 1. Create Golden Evaluation Dataset
Build high-quality test set:

```python
# data/evaluation/golden_queries.json
[
    {
        "query_id": "Q001",
        "query": "How do I reset my password?",
        "query_type": "procedural",
        "relevant_chunks": ["chunk_042", "chunk_043"],
        "expected_answer_key_points": [
            "Click forgot password link",
            "Enter email address",
            "Check email for reset link"
        ],
        "difficulty": "easy"
    },
    {
        "query_id": "Q002",
        "query": "What's the difference between OAuth and API keys?",
        "query_type": "comparison",
        "relevant_chunks": ["chunk_087", "chunk_091", "chunk_103"],
        "expected_answer_key_points": [
            "OAuth for user authentication",
            "API keys for service authentication",
            "Security differences"
        ],
        "difficulty": "medium"
    }
]
```

**Dataset requirements:**
- 30-50 diverse queries minimum
- Mix of difficulty levels (easy/medium/hard)
- All query types represented
- Ground truth relevance judgments
- Expected answer elements identified

**How to create it:**
1. **Manual curation:** Write queries you expect users to ask
2. **Document-based:** Extract questions from FAQs, support tickets
3. **Synthetic:** Use LLM to generate diverse questions from docs
4. **User logs:** Sample real queries (if available)

#### 2. Retrieval Evaluation Metrics
Measure how well you're finding relevant chunks:

```python
class RetrievalEvaluator:
    def __init__(self, golden_dataset):
        self.golden_dataset = golden_dataset
    
    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Precision@K: What fraction of top-k results are relevant?
        
        P@K = (# relevant in top-k) / k
        """
        top_k = retrieved[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant))
        return relevant_in_top_k / k
    
    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Recall@K: What fraction of relevant docs are in top-k?
        
        R@K = (# relevant in top-k) / (total # relevant)
        """
        top_k = retrieved[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant))
        return relevant_in_top_k / len(relevant) if relevant else 0
    
    def mean_reciprocal_rank(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        MRR: How quickly do we find the first relevant result?
        
        MRR = 1 / (rank of first relevant doc)
        """
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0
    
    def ndcg_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        Accounts for ranking quality (highly relevant docs should be ranked higher)
        """
        from sklearn.metrics import ndcg_score
        import numpy as np
        
        # Create relevance scores for retrieved docs
        relevance = [1 if doc in relevant else 0 for doc in retrieved[:k]]
        
        # Ideal ranking (all relevant docs first)
        ideal = sorted(relevance, reverse=True)
        
        if sum(ideal) == 0:
            return 0.0
        
        return ndcg_score([ideal], [relevance])
    
    def evaluate_retrieval(
        self,
        retrieval_function,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> dict:
        """
        Run retrieval on all golden queries and compute metrics.
        """
        results = {
            f'precision@{k}': [] for k in k_values
        }
        results.update({f'recall@{k}': [] for k in k_values})
        results['mrr'] = []
        results['ndcg@5'] = []
        
        for item in self.golden_dataset:
            query = item['query']
            relevant = item['relevant_chunks']
            
            # Run retrieval
            retrieved = retrieval_function(query, top_k=max(k_values))
            retrieved_ids = [chunk['chunk_id'] for chunk in retrieved]
            
            # Compute metrics
            for k in k_values:
                results[f'precision@{k}'].append(
                    self.precision_at_k(retrieved_ids, relevant, k)
                )
                results[f'recall@{k}'].append(
                    self.recall_at_k(retrieved_ids, relevant, k)
                )
            
            results['mrr'].append(
                self.mean_reciprocal_rank(retrieved_ids, relevant)
            )
            results['ndcg@5'].append(
                self.ndcg_at_k(retrieved_ids, relevant, 5)
            )
        
        # Average across all queries
        return {metric: np.mean(scores) for metric, scores in results.items()}
```

**Key retrieval metrics:**

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| Precision@5 | Accuracy of top 5 results | >0.6 |
| Recall@10 | Coverage in top 10 | >0.8 |
| MRR | Rank of first good result | >0.7 |
| NDCG@5 | Ranking quality | >0.75 |

#### 3. Answer Quality Evaluation
Assess generated answers:

##### A. LLM-as-Judge
```python
class AnswerEvaluator:
    def __init__(self, judge_llm):
        self.judge = judge_llm
    
    def evaluate_answer(
        self,
        query: str,
        answer: str,
        golden_answer: dict,
        context: List[dict]
    ) -> dict:
        """
        Use LLM to judge answer quality.
        """
        judge_prompt = f"""
You are evaluating the quality of an AI-generated answer.

**Query:** {query}

**Generated Answer:** {answer}

**Expected Key Points:**
{json.dumps(golden_answer['expected_answer_key_points'], indent=2)}

**Retrieved Context:**
{self.format_context(context)}

Rate the answer on these dimensions (0-5 scale):

1. **Relevance:** Does it answer the question?
2. **Completeness:** Are all key points covered?
3. **Accuracy:** Is information correct based on context?
4. **Grounding:** Is answer supported by provided context?
5. **Clarity:** Is it well-written and understandable?

Return JSON:
{{
    "relevance": <0-5>,
    "completeness": <0-5>,
    "accuracy": <0-5>,
    "grounding": <0-5>,
    "clarity": <0-5>,
    "overall_score": <0-5>,
    "reasoning": "Brief explanation",
    "missing_points": ["point 1", "..."],
    "hallucinations": ["claim 1", "..."]
}}
        """
        
        # Get LLM judgment
        judgment = self.judge.generate(judge_prompt, response_format='json')
        return judgment
```

##### B. Automated Metrics
```python
def compute_answer_metrics(generated: str, golden: dict) -> dict:
    """
    Automated metrics for answer quality.
    """
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    
    # ROUGE: N-gram overlap with reference answer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_scores = scorer.score(
        golden['reference_answer'],
        generated
    )
    
    # BERTScore: Semantic similarity
    P, R, F1 = bert_score(
        [generated],
        [golden['reference_answer']],
        lang='en'
    )
    
    # Key point coverage
    key_points = golden['expected_answer_key_points']
    covered_points = sum(
        1 for point in key_points
        if point.lower() in generated.lower()
    )
    coverage = covered_points / len(key_points)
    
    return {
        'rouge1_f1': rouge_scores['rouge1'].fmeasure,
        'rouge2_f1': rouge_scores['rouge2'].fmeasure,
        'rougeL_f1': rouge_scores['rougeL'].fmeasure,
        'bert_f1': F1.mean().item(),
        'key_point_coverage': coverage
    }
```

##### C. Human Evaluation (Gold Standard)
```python
# Create evaluation interface
evaluation_template = """
Query: {query}

Generated Answer:
{answer}

Rate (1-5):
□ Relevance: ___
□ Accuracy: ___
□ Completeness: ___
□ Clarity: ___

Issues:
□ Hallucination
□ Missing info
□ Incorrect
□ Unclear

Notes: ________________
"""

# Collect ratings from 2-3 evaluators
# Calculate inter-rater agreement (Cohen's kappa)
```

#### 4. End-to-End System Evaluation
Evaluate complete pipeline:

```python
class SystemEvaluator:
    def __init__(self, rag_pipeline, golden_dataset):
        self.pipeline = rag_pipeline
        self.golden_dataset = golden_dataset
        self.retrieval_eval = RetrievalEvaluator(golden_dataset)
        self.answer_eval = AnswerEvaluator(judge_llm)
    
    def run_full_evaluation(self) -> dict:
        """
        Evaluate entire system on golden dataset.
        """
        results = {
            'retrieval_metrics': {},
            'answer_metrics': {},
            'system_metrics': {},
            'per_query_results': []
        }
        
        for item in self.golden_dataset:
            query = item['query']
            
            # Run full pipeline
            start_time = time.time()
            response = self.pipeline.query(query)
            latency = (time.time() - start_time) * 1000  # ms
            
            # Evaluate retrieval
            retrieval_metrics = self.retrieval_eval.evaluate_single(
                retrieved=response['chunks'],
                relevant=item['relevant_chunks']
            )
            
            # Evaluate answer
            answer_metrics = self.answer_eval.evaluate_answer(
                query=query,
                answer=response['answer'],
                golden_answer=item,
                context=response['chunks']
            )
            
            # System metrics
            system_metrics = {
                'latency_ms': latency,
                'tokens_used': response['tokens_used'],
                'cost': self.calculate_cost(response),
                'chunks_retrieved': len(response['chunks'])
            }
            
            # Store per-query results
            results['per_query_results'].append({
                'query_id': item['query_id'],
                'query': query,
                'retrieval': retrieval_metrics,
                'answer': answer_metrics,
                'system': system_metrics
            })
        
        # Aggregate metrics
        results['retrieval_metrics'] = self.aggregate_metrics(
            [r['retrieval'] for r in results['per_query_results']]
        )
        results['answer_metrics'] = self.aggregate_metrics(
            [r['answer'] for r in results['per_query_results']]
        )
        results['system_metrics'] = self.aggregate_metrics(
            [r['system'] for r in results['per_query_results']]
        )
        
        return results
```

#### 5. A/B Testing Framework
Compare different system configurations:

```python
class ABTest:
    def compare_systems(
        self,
        system_a: dict,
        system_b: dict,
        golden_dataset: List[dict]
    ) -> dict:
        """
        Compare two system configurations.
        
        Example:
        system_a = {
            'name': 'Hybrid + Rerank',
            'retrieval': hybrid_retriever,
            'rerank': True
        }
        system_b = {
            'name': 'Vector Only',
            'retrieval': vector_retriever,
            'rerank': False
        }
        """
        results_a = self.evaluate_system(system_a, golden_dataset)
        results_b = self.evaluate_system(system_b, golden_dataset)
        
        comparison = {
            'system_a': results_a,
            'system_b': results_b,
            'winner': {},
            'statistical_significance': {}
        }
        
        # Compare metrics
        for metric in results_a['retrieval_metrics']:
            val_a = results_a['retrieval_metrics'][metric]
            val_b = results_b['retrieval_metrics'][metric]
            
            comparison['winner'][metric] = 'A' if val_a > val_b else 'B'
            
            # Statistical significance test (t-test)
            from scipy import stats
            _, p_value = stats.ttest_rel(
                [r[metric] for r in results_a['per_query']],
                [r[metric] for r in results_b['per_query']]
            )
            comparison['statistical_significance'][metric] = p_value
        
        return comparison
```

**Experiments to run:**
1. Vector-only vs Hybrid retrieval
2. With vs without re-ranking
3. GPT-3.5 vs GPT-4 for synthesis
4. Different chunk sizes (500 vs 1000 tokens)
5. Top-5 vs Top-10 retrieval
6. Different embedding models

#### 6. Performance Benchmarking
Track system performance:

```python
class PerformanceBenchmark:
    def benchmark_pipeline(self, num_queries: int = 100) -> dict:
        """
        Measure latency, throughput, cost.
        """
        latencies = []
        token_usage = []
        
        for i in range(num_queries):
            query = self.generate_test_query()
            
            start = time.time()
            result = self.pipeline.query(query)
            latency = time.time() - start
            
            latencies.append(latency)
            token_usage.append(result['tokens_used'])
        
        return {
            'avg_latency_ms': np.mean(latencies) * 1000,
            'p50_latency_ms': np.percentile(latencies, 50) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
            'avg_tokens': np.mean(token_usage),
            'queries_per_second': 1 / np.mean(latencies),
            'estimated_cost_per_1k_queries': self.estimate_cost(token_usage)
        }
```

**Performance targets:**

| Metric | Target |
|--------|--------|
| P50 latency | <2s |
| P95 latency | <5s |
| P99 latency | <10s |
| Cost per query | <$0.05 |

#### 7. Failure Analysis
Systematically analyze mistakes:

```python
class FailureAnalyzer:
    def analyze_failures(self, eval_results: dict) -> dict:
        """
        Find patterns in system failures.
        """
        failures = [
            r for r in eval_results['per_query_results']
            if r['answer']['overall_score'] < 3
        ]
        
        analysis = {
            'total_failures': len(failures),
            'failure_rate': len(failures) / len(eval_results['per_query_results']),
            'failure_by_type': defaultdict(int),
            'failure_by_difficulty': defaultdict(int),
            'common_issues': defaultdict(int),
            'examples': []
        }
        
        for failure in failures:
            # Categorize failure
            query_type = failure['query_type']
            difficulty = failure['difficulty']
            
            analysis['failure_by_type'][query_type] += 1
            analysis['failure_by_difficulty'][difficulty] += 1
            
            # Identify issue type
            if failure['answer']['grounding'] < 3:
                analysis['common_issues']['hallucination'] += 1
            if failure['retrieval']['precision@5'] < 0.4:
                analysis['common_issues']['poor_retrieval'] += 1
            if failure['answer']['completeness'] < 3:
                analysis['common_issues']['incomplete_answer'] += 1
            
            # Save example
            if len(analysis['examples']) < 10:
                analysis['examples'].append({
                    'query': failure['query'],
                    'issue': failure['answer']['reasoning'],
                    'score': failure['answer']['overall_score']
                })
        
        return analysis
```

**Use failure analysis to:**
- Identify weak query types
- Find retrieval gaps
- Improve prompts
- Add more documents
- Refine chunking strategy

#### 8. Logging & Observability
Track everything in production:

```python
class QueryLogger:
    def log_query(self, query_data: dict):
        """
        Log query execution for analysis.
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query_id': str(uuid.uuid4()),
            'query': query_data['query'],
            'intent': query_data['intent'],
            'retrieval_strategy': query_data['strategy'],
            'chunks_retrieved': len(query_data['chunks']),
            'chunks': [
                {
                    'chunk_id': c['chunk_id'],
                    'score': c['score'],
                    'document': c['document_name']
                }
                for c in query_data['chunks']
            ],
            'answer': query_data['answer'],
            'citations': query_data['citations'],
            'confidence': query_data['confidence'],
            'latency_ms': query_data['latency_ms'],
            'tokens_used': query_data['tokens_used'],
            'cost': query_data['cost'],
            'model': query_data['model']
        }
        
        # Write to log file
        with open(f'logs/queries_{date.today()}.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

**Dashboard metrics:**
- Queries per day
- Average latency
- Daily cost
- Answer quality distribution
- Popular query types
- Failure rate trend

### Dependencies
```
scikit-learn          # Metrics computation
scipy                 # Statistical tests
rouge-score           # ROUGE metrics
bert-score            # Semantic similarity
plotly / matplotlib   # Visualization
pandas               # Data analysis
```

### Deliverables
- ✅ Golden evaluation dataset (30-50 queries)
- ✅ Retrieval evaluation metrics
- ✅ Answer quality evaluation (LLM-as-judge)
- ✅ End-to-end system evaluation
- ✅ A/B testing framework
- ✅ Performance benchmarks
- ✅ Failure analysis tools
- ✅ Logging infrastructure

### Success Criteria
- Precision@5 > 0.6 on golden dataset
- MRR > 0.7
- Answer quality score > 4.0/5 (LLM judge)
- P95 latency < 5 seconds
- Can identify and explain failures
- Logging captures all necessary data

### Experiments to Run
1. **Retrieval comparison**
   - Vector vs BM25 vs Hybrid
   - Different fusion weights
   
2. **Generation comparison**
   - GPT-3.5 vs GPT-4
   - Different prompts
   
3. **Re-ranking impact**
   - With/without re-ranking
   - Cross-encoder vs LLM re-ranking
   
4. **Chunk size impact**
   - 500 vs 1000 vs 1500 tokens
   
5. **Top-k impact**
   - 3 vs 5 vs 10 chunks for generation

### Evaluation Report Template
```markdown
# DocuSense Evaluation Report

## Dataset
- 50 queries across 7 types
- 15 easy, 25 medium, 10 hard

## Retrieval Performance
- Precision@5: 0.72
- Recall@10: 0.85
- MRR: 0.78
- NDCG@5: 0.80

## Answer Quality
- Relevance: 4.3/5
- Accuracy: 4.5/5
- Completeness: 4.0/5
- Overall: 4.2/5

## System Performance
- P50 latency: 1.8s
- P95 latency: 4.2s
- Cost per query: $0.03

## Key Findings
1. Hybrid retrieval beats vector-only by 15%
2. Re-ranking improves precision@5 from 0.65 to 0.72
3. GPT-4 synthesis worth the extra cost (quality +0.6 points)
4. Struggles with multi-hop questions (35% failure rate)

## Recommendations
1. Add multi-hop reasoning capability
2. Expand dataset with more technical terms
3. Optimize chunk size for technical content
```

---

## 📦 Phase 8 — API / UI for Debugging

**Timeline:** 3-4 days  
**Goal:** Build transparent interfaces that expose system internals

### Why This Phase Matters
A good interface isn't just about aesthetics—it's about understanding what your system is doing. This phase focuses on observability: seeing query processing steps, retrieval results, and answer generation. This is critical for debugging, demos, and understanding system behavior.

### Learning Focus
- REST API design for RAG systems
- Building informative UIs
- Execution tracing and visibility
- Interactive debugging tools
- Gradio for rapid prototyping

### Tasks

#### 1. REST API Design
Build FastAPI endpoints:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="DocuSense API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranking: bool = True
    strategy: Optional[str] = None  # auto, vector, hybrid, etc.

class QueryResponse(BaseModel):
    query_id: str
    query: str
    intent: dict
    rewritten_query: str
    retrieval_plan: dict
    chunks: List[dict]
    answer: str
    citations: List[dict]
    confidence: str
    execution_trace: dict
    metrics: dict

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Main query endpoint - full RAG pipeline.
    """
    try:
        result = rag_pipeline.process_query(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify-intent")
async def classify_intent(query: str):
    """
    Just intent classification.
    """
    return intent_classifier.classify(query)

@app.post("/api/retrieve")
async def retrieve_chunks(query: str, strategy: str = "hybrid", top_k: int = 5):
    """
    Just retrieval, no answer generation.
    """
    if strategy == "vector":
        results = vector_retriever.search(query, top_k)
    elif strategy == "bm25":
        results = bm25_retriever.search(query, top_k)
    else:  # hybrid
        results = hybrid_retriever.retrieve(query, top_k)
    
    return {"chunks": results}

@app.post("/api/synthesize")
async def synthesize_answer(query: str, chunks: List[dict]):
    """
    Just answer generation from provided chunks.
    """
    answer = answer_generator.generate(query, chunks)
    return answer

@app.get("/api/documents")
async def list_documents():
    """
    List indexed documents.
    """
    return document_store.list_documents()

@app.get("/api/metrics")
async def get_metrics():
    """
    System metrics and stats.
    """
    return {
        "total_queries": metrics.total_queries,
        "avg_latency_ms": metrics.avg_latency,
        "avg_cost": metrics.avg_cost,
        "index_stats": {
            "total_chunks": vector_store.get_count(),
            "total_documents": document_store.count()
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "vector_store": "connected",
        "llm_api": "available"
    }
```

#### 2. Execution Tracing
Track every step for transparency:

```python
class ExecutionTracer:
    def __init__(self):
        self.steps = []
    
    def log_step(self, step_name: str, data: dict, duration_ms: float):
        """
        Record a pipeline step.
        """
        self.steps.append({
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'duration_ms': duration_ms,
            'data': data
        })
    
    def get_trace(self) -> dict:
        """
        Return execution trace.
        """
        return {
            'total_steps': len(self.steps),
            'total_duration_ms': sum(s['duration_ms'] for s in self.steps),
            'steps': self.steps
        }

# Use in pipeline
tracer = ExecutionTracer()

# Step 1: Intent classification
start = time.time()
intent = intent_classifier.classify(query)
tracer.log_step("intent_classification", intent, (time.time() - start) * 1000)

# Step 2: Query rewriting
start = time.time()
rewritten = query_rewriter.rewrite(query, intent['intent'])
tracer.log_step("query_rewriting", rewritten, (time.time() - start) * 1000)

# ... etc for each step
```

#### 3. Gradio Debug Dashboard
Build interactive UI for testing:

```python
import gradio as gr

def process_query_with_details(query, strategy, top_k, use_rerank):
    """
    Process query and return detailed results for UI.
    """
    result = rag_pipeline.process_query(
        query=query,
        strategy=strategy,
        top_k=top_k,
        use_reranking=use_rerank
    )
    
    # Format for display
    intent_display = f"""
**Intent:** {result['intent']['intent']}  
**Confidence:** {result['intent']['confidence']:.2f}  
**Reasoning:** {result['intent']['reasoning']}
    """
    
    rewritten_display = f"""
**Original:** {result['query']}  
**Rewritten:** {result['rewritten_query']}  
**Key Terms:** {', '.join(result['key_terms'])}
    """
    
    retrieval_display = "\\n\\n".join([
        f"""
**Chunk {i+1}** (score: {chunk['score']:.3f})  
*Source: {chunk['document_name']}, page {chunk.get('page', 'N/A')}*  
{chunk['text'][:300]}...
        """
        for i, chunk in enumerate(result['chunks'])
    ])
    
    answer_display = f"""
{result['answer']}

---
**Confidence:** {result['confidence']}  
**Tokens Used:** {result['tokens_used']}  
**Latency:** {result['latency_ms']:.0f}ms  
**Cost:** ${result['cost']:.4f}
    """
    
    # Execution trace as JSON
    trace_json = json.dumps(result['execution_trace'], indent=2)
    
    return intent_display, rewritten_display, retrieval_display, answer_display, trace_json

# Build Gradio interface
with gr.Blocks(title="DocuSense Debug Dashboard") as demo:
    gr.Markdown("# DocuSense - RAG System Debugger")
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Query",
                placeholder="Ask a question...",
                lines=2
            )
            
            with gr.Row():
                strategy_dropdown = gr.Dropdown(
                    choices=["auto", "vector", "bm25", "hybrid"],
                    value="hybrid",
                    label="Retrieval Strategy"
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Top K"
                )
                rerank_checkbox = gr.Checkbox(
                    label="Use Re-ranking",
                    value=True
                )
            
            submit_btn = gr.Button("Process Query", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### System Stats")
            stats_display = gr.JSON(label="Metrics")
    
    with gr.Tabs():
        with gr.Tab("Intent & Planning"):
            intent_output = gr.Markdown(label="Intent Classification")
            rewrite_output = gr.Markdown(label="Query Rewriting")
        
        with gr.Tab("Retrieval Results"):
            retrieval_output = gr.Markdown(label="Retrieved Chunks")
        
        with gr.Tab("Answer"):
            answer_output = gr.Markdown(label="Generated Answer")
        
        with gr.Tab("Execution Trace"):
            trace_output = gr.Code(label="Step-by-Step Trace", language="json")
    
    submit_btn.click(
        fn=process_query_with_details,
        inputs=[query_input, strategy_dropdown, top_k_slider, rerank_checkbox],
        outputs=[intent_output, rewrite_output, retrieval_output, answer_output, trace_output]
    )
    
    # Auto-refresh stats
    demo.load(fn=get_system_stats, outputs=stats_display)

demo.launch(share=True)
```

**Dashboard features:**
- Query input with strategy selection
- Step-by-step execution visibility
- Retrieved chunks with scores
- Final answer with metrics
- Execution trace (JSON)
- System statistics

#### 4. Comparison View
Compare different strategies side-by-side:

```python
def compare_strategies(query):
    """
    Run same query with different strategies.
    """
    strategies = ["vector", "bm25", "hybrid", "hybrid+rerank"]
    results = {}
    
    for strategy in strategies:
        result = rag_pipeline.process_query(
            query=query,
            strategy=strategy
        )
        results[strategy] = result
    
    # Format comparison
    comparison = {
        'query': query,
        'strategies': {}
    }
    
    for strat, res in results.items():
        comparison['strategies'][strat] = {
            'answer': res['answer'],
            'top_chunks': [c['chunk_id'] for c in res['chunks'][:3]],
            'confidence': res['confidence'],
            'latency_ms': res['latency_ms'],
            'cost': res['cost']
        }
    
    return comparison

# Gradio comparison interface
with gr.Blocks() as comparison_demo:
    query_input = gr.Textbox(label="Query")
    compare_btn = gr.Button("Compare Strategies")
    comparison_output = gr.JSON(label="Strategy Comparison")
    
    compare_btn.click(
        fn=compare_strategies,
        inputs=query_input,
        outputs=comparison_output
    )
```

#### 5. Evaluation Results Viewer
Visualize evaluation metrics:

```python
def create_evaluation_dashboard():
    """
    Dashboard for evaluation results.
    """
    import plotly.graph_objects as go
    import pandas as pd
    
    # Load evaluation results
    eval_results = load_evaluation_results()
    
    # Metrics over time
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Scatter(
        x=eval_results['date'],
        y=eval_results['precision@5'],
        name='Precision@5'
    ))
    fig_metrics.add_trace(go.Scatter(
        x=eval_results['date'],
        y=eval_results['mrr'],
        name='MRR'
    ))
    
    # Query type breakdown
    query_type_stats = eval_results.groupby('query_type')['score'].mean()
    fig_query_types = go.Figure(data=[
        go.Bar(x=query_type_stats.index, y=query_type_stats.values)
    ])
    
    # Failure analysis
    failures = eval_results[eval_results['score'] < 3]
    fig_failures = go.Figure(data=[
        go.Pie(labels=failures['failure_reason'].value_counts().index,
               values=failures['failure_reason'].value_counts().values)
    ])
    
    return fig_metrics, fig_query_types, fig_failures

with gr.Blocks() as eval_dashboard:
    gr.Markdown("# Evaluation Dashboard")
    
    with gr.Row():
        metrics_plot = gr.Plot(label="Metrics Over Time")
        query_type_plot = gr.Plot(label="Performance by Query Type")
    
    failures_plot = gr.Plot(label="Failure Breakdown")
    
    eval_dashboard.load(
        fn=create_evaluation_dashboard,
        outputs=[metrics_plot, query_type_plot, failures_plot]
    )
```

#### 6. Document Explorer
Browse indexed documents:

```python
@app.get("/api/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    """
    Get all chunks for a document.
    """
    chunks = chunk_store.get_by_document(doc_id)
    return {
        'document_id': doc_id,
        'total_chunks': len(chunks),
        'chunks': chunks
    }

# Gradio document browser
def browse_documents():
    """
    UI for browsing indexed documents.
    """
    docs = document_store.list_documents()
    
    with gr.Blocks() as doc_browser:
        doc_dropdown = gr.Dropdown(
            choices=[d['name'] for d in docs],
            label="Select Document"
        )
        
        doc_info = gr.JSON(label="Document Info")
        chunks_display = gr.Dataframe(label="Chunks")
        
        def show_document(doc_name):
            doc = document_store.get_by_name(doc_name)
            chunks = chunk_store.get_by_document(doc['id'])
            
            return doc, pd.DataFrame(chunks)
        
        doc_dropdown.change(
            fn=show_document,
            inputs=doc_dropdown,
            outputs=[doc_info, chunks_display]
        )
    
    return doc_browser
```

#### 7. Query Logs Viewer
Analyze historical queries:

```python
@app.get("/api/logs/queries")
async def get_query_logs(limit: int = 100, date: Optional[str] = None):
    """
    Retrieve query logs.
    """
    logs = query_logger.get_logs(limit=limit, date=date)
    return logs

# Gradio log viewer
def view_query_logs(date_filter, limit):
    """
    Interactive query log viewer.
    """
    logs = query_logger.get_logs(date=date_filter, limit=limit)
    
    df = pd.DataFrame(logs)
    
    # Summary stats
    stats = {
        'total_queries': len(df),
        'avg_latency_ms': df['latency_ms'].mean(),
        'avg_cost': df['cost'].mean(),
        'confidence_distribution': df['confidence'].value_counts().to_dict()
    }
    
    return df, stats
```

### Dependencies
```
fastapi               # REST API
uvicorn              # ASGI server
gradio               # UI framework
plotly               # Visualizations
pandas               # Data handling
```

### Deliverables
- ✅ REST API with comprehensive endpoints
- ✅ Execution tracing system
- ✅ Gradio debug dashboard
- ✅ Strategy comparison tool
- ✅ Evaluation results viewer
- ✅ Document explorer
- ✅ Query logs viewer
- ✅ API documentation (auto-generated by FastAPI)

### Success Criteria
- API responds to all queries
- Dashboard shows all pipeline steps
- Can compare strategies interactively
- Execution traces are complete and readable
- All UI components work without errors
- API documentation is clear

### Example API Usage
```python
import requests

# Query documents
response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "query": "How do I reset my password?",
        "top_k": 5,
        "use_reranking": True
    }
)
result = response.json()

# Just retrieve chunks
response = requests.post(
    "http://localhost:8000/api/retrieve",
    json={
        "query": "authentication methods",
        "strategy": "hybrid",
        "top_k": 10
    }
)
chunks = response.json()['chunks']
```

---

## 📚 Phase 9 — Polish & Documentation

**Timeline:** 3-4 days  
**Goal:** Transform engineering work into a portfolio-ready showcase

### Why This Phase Matters
This phase elevates your project from "working code" to "professional artifact". Good documentation demonstrates your understanding, makes the project maintainable, and showcases your work to potential employers or collaborators. This is what makes your project stand out.

### Learning Focus
- Technical writing and documentation
- Architectural communication
- Experiment reporting
- Code organization and best practices
- Portfolio presentation

### Tasks

#### 1. Project Documentation Structure
Create comprehensive documentation:

```
docs/
├── README.md                    # Project overview
├── ARCHITECTURE.md              # System design
├── SETUP.md                     # Installation & setup
├── API_REFERENCE.md             # API documentation
├── EVALUATION_REPORT.md         # Experiments & results
├── DESIGN_DECISIONS.md          # Key choices & rationale
├── FAILURE_MODES.md             # Known limitations
├── FUTURE_WORK.md               # Potential improvements
└── images/
    ├── architecture_diagram.png
    ├── pipeline_flow.png
    └── evaluation_charts/
```

#### 2. Main README.md
Create compelling project README:

```markdown
# DocuSense - Intelligent Document Q&A System

> A production-quality RAG (Retrieval-Augmented Generation) system with hybrid retrieval, query understanding, and comprehensive evaluation.

![Architecture Diagram](docs/images/architecture_diagram.png)

## 🎯 What is DocuSense?

DocuSense transforms document collections into an intelligent Q&A system. It understands user queries, retrieves relevant information using hybrid search, and generates accurate, grounded answers with citations.

### Key Features
- **Hybrid Retrieval:** Combines semantic (vector) and keyword (BM25) search
- **Query Understanding:** Intent classification and query rewriting
- **Smart Re-ranking:** Cross-encoder re-ranking for precision
- **Grounded Answers:** Citations and source attribution
- **Comprehensive Evaluation:** Measured retrieval and answer quality
- **Transparent Pipeline:** Full execution tracing and debugging tools

## 📊 Performance

| Metric | Value |
|--------|-------|
| Precision@5 | 0.72 |
| MRR | 0.78 |
| Answer Quality (LLM-Judge) | 4.2/5 |
| P95 Latency | 4.2s |
| Cost per Query | $0.03 |

*Evaluated on 50-query golden dataset*

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation
\`\`\`bash
# Clone repository
git clone <repo-url>
cd docusense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your API keys to .env
\`\`\`

### Usage
\`\`\`python
from docusense import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Index documents
pipeline.index_documents("data/raw/")

# Query
result = pipeline.query("How do I reset my password?")

print(result['answer'])
print(f"Sources: {result['citations']}")
\`\`\`

### Web Interface
\`\`\`bash
# Launch Gradio dashboard
python app.py

# Or use REST API
uvicorn api:app --reload
\`\`\`

## 📁 Project Structure
\`\`\`
docusense/
├── llms/              # LLM provider abstractions
├── retrieval/         # Vector stores, chunking, hybrid search
├── agents/            # Query planning & orchestration
├── evaluation/        # Metrics, benchmarks, evaluation
├── api/               # REST API endpoints
└── ui/                # Gradio dashboard

data/
├── raw/               # Original documents
├── processed/         # Chunked data
└── vector_stores/     # FAISS indexes

tests/                 # Unit and integration tests
docs/                  # Documentation
logs/                  # Application logs
\`\`\`

## 🏗️ Architecture

DocuSense implements a modern RAG architecture:

1. **Document Ingestion:** Parse, chunk, embed documents
2. **Query Processing:** Classify intent, rewrite query, plan strategy
3. **Hybrid Retrieval:** Combine vector + BM25 search
4. **Re-ranking:** Refine results with cross-encoder
5. **Answer Synthesis:** Generate grounded response with citations

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design.

## 🔬 Key Experiments

### Retrieval Strategy Comparison
| Strategy | Precision@5 | MRR | Latency |
|----------|-------------|-----|---------|
| Vector Only | 0.58 | 0.65 | 1.2s |
| BM25 Only | 0.52 | 0.61 | 0.8s |
| **Hybrid** | **0.72** | **0.78** | 1.5s |
| Hybrid + Rerank | **0.76** | **0.82** | 2.8s |

**Finding:** Hybrid retrieval + re-ranking improves precision by 31% vs vector-only.

See [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) for full results.

## 🛠️ Technologies
- **LLMs:** OpenAI GPT-4, GPT-3.5
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector Store:** FAISS
- **Keyword Search:** BM25 (rank-bm25)
- **Re-ranking:** Cross-encoder (sentence-transformers)
- **API:** FastAPI
- **UI:** Gradio
- **Evaluation:** Custom framework + LLM-as-judge

## 📚 Documentation
- [Setup Guide](docs/SETUP.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Evaluation Report](docs/EVALUATION_REPORT.md)
- [Design Decisions](docs/DESIGN_DECISIONS.md)

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Modern RAG architecture implementation
- ✅ Hybrid retrieval strategies (vector + keyword)
- ✅ Query understanding and planning
- ✅ Systematic evaluation methodology
- ✅ LLM prompt engineering
- ✅ Production-ready API design
- ✅ Observability and debugging tools

## 🔮 Future Work
- Multi-hop reasoning for complex queries
- Support for multi-modal documents (images, tables)
- Streaming responses
- User feedback loop
- Advanced agent capabilities
- Fine-tuned embeddings

## 📄 License
MIT

## 🤝 Contributing
Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)
\`\`\`

#### 3. Architecture Documentation (ARCHITECTURE.md)
Explain system design:

```markdown
# DocuSense Architecture

## System Overview

DocuSense is a Retrieval-Augmented Generation (RAG) system designed for document question-answering with emphasis on transparency, quality, and measurability.

## High-Level Architecture

\`\`\`
┌─────────────┐
│  User Query │
└──────┬──────┘
       │
       ▼
┌────────────────────┐
│ Query Processor    │
│ - Intent Classifier│
│ - Query Rewriter   │
│ - Strategy Planner │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Hybrid Retriever   │
│ ┌────────────────┐ │
│ │ Vector Search  │ │
│ └────────────────┘ │
│ ┌────────────────┐ │
│ │  BM25 Search   │ │
│ └────────────────┘ │
│ ┌────────────────┐ │
│ │     Fusion     │ │
│ └────────────────┘ │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│    Re-ranker       │
│ (Cross-Encoder)    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Answer Generator   │
│ - Context Format   │
│ - LLM Synthesis    │
│ - Citation Extract │
└─────────┬──────────┘
          │
          ▼
┌─────────────┐
│   Answer    │
│ + Citations │
└─────────────┘
\`\`\`

## Component Details

### 1. Document Ingestion Pipeline
**Purpose:** Convert raw documents into searchable knowledge base

**Components:**
- **Document Loaders:** Parse PDF, DOCX, TXT files
- **Text Cleaner:** Normalize and clean text
- **Chunker:** Split into semantic units (500-1000 tokens)
- **Embedder:** Generate vector representations
- **Vector Store:** Index in FAISS

**Key Design Decision:** Semantic chunking over fixed-size to preserve context.

### 2. Query Processing Module
**Purpose:** Understand and optimize user queries

**Components:**
- **Intent Classifier:** Categorize query type
- **Query Rewriter:** Improve retrieval effectiveness
- **Strategy Planner:** Select optimal retrieval approach

**Key Design Decision:** LLM-based classification for flexibility and accuracy.

### 3. Hybrid Retrieval System
**Purpose:** Find most relevant chunks

**Components:**
- **Vector Retriever:** Semantic similarity (FAISS)
- **BM25 Retriever:** Keyword matching
- **Fusion Module:** Reciprocal Rank Fusion
- **Re-ranker:** Cross-encoder scoring

**Key Design Decision:** Hybrid approach to balance semantic and lexical matching.

### 4. Answer Synthesis
**Purpose:** Generate grounded, cited answers

**Components:**
- **Context Formatter:** Prepare chunks for prompt
- **LLM Generator:** Synthesize answer
- **Citation Extractor:** Map claims to sources
- **Validator:** Check grounding

**Key Design Decision:** Explicit grounding instructions to prevent hallucination.

## Data Flow

1. **Indexing (Offline)**
   ```
   Documents → Parse → Chunk → Embed → Store in Vector DB
   ```

2. **Query Processing (Online)**
   ```
   Query → Classify Intent → Rewrite → Plan Strategy →
   Retrieve (Vector + BM25) → Fuse → Re-rank →
   Format Context → Generate Answer → Extract Citations
   ```

## Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Embeddings | OpenAI text-embedding-3-small | High quality, cost-effective |
| Vector Store | FAISS | Fast, local, no dependencies |
| Keyword Search | BM25 (rank-bm25) | Standard, effective |
| Re-ranking | Cross-encoder | Better than bi-encoder |
| LLM | GPT-4 / GPT-3.5 | Quality vs cost trade-off |
| API | FastAPI | Modern, fast, auto-docs |
| UI | Gradio | Rapid prototyping |

## Scalability Considerations

- **Current:** Single-machine, in-memory FAISS
- **Future:** Distributed vector DB (Milvus), caching, async processing

## Security & Privacy

- API keys stored in environment variables
- No user data logged by default
- Option to disable query logging
\`\`\`

#### 4. Evaluation Report (EVALUATION_REPORT.md)
Document experiments and results:

```markdown
# DocuSense Evaluation Report

## Executive Summary
DocuSense achieves 0.72 precision@5 and 4.2/5 answer quality on a 50-query golden dataset. Hybrid retrieval with re-ranking outperforms vector-only by 31%.

## Evaluation Methodology

### Golden Dataset
- **Size:** 50 queries
- **Query Types:** 7 categories (factual, conceptual, procedural, etc.)
- **Difficulty:** 15 easy, 25 medium, 10 hard
- **Ground Truth:** Manual relevance judgments

### Metrics

#### Retrieval Metrics
- **Precision@K:** Relevance of top-K results
- **Recall@K:** Coverage of relevant docs in top-K
- **MRR:** Mean Reciprocal Rank (rank of first relevant result)
- **NDCG@5:** Normalized Discounted Cumulative Gain

#### Answer Quality Metrics
- **LLM-as-Judge:** GPT-4 scoring (0-5 scale)
- **Human Evaluation:** Expert ratings on subset
- **Key Point Coverage:** % of expected points mentioned

## Results

### Retrieval Performance

| Strategy | Precision@5 | Recall@10 | MRR | NDCG@5 |
|----------|-------------|-----------|-----|--------|
| Vector Only | 0.58 | 0.73 | 0.65 | 0.68 |
| BM25 Only | 0.52 | 0.69 | 0.61 | 0.63 |
| Hybrid (RRF) | 0.72 | 0.85 | 0.78 | 0.80 |
| Hybrid + Rerank | **0.76** | **0.87** | **0.82** | **0.84** |

**Key Finding:** Hybrid retrieval improves precision by 24% vs vector-only. Re-ranking adds another 5%.

### Answer Quality

| Dimension | Score (0-5) |
|-----------|-------------|
| Relevance | 4.3 |
| Accuracy | 4.5 |
| Completeness | 4.0 |
| Grounding | 4.6 |
| Clarity | 4.2 |
| **Overall** | **4.2** |

**Key Finding:** High grounding score (4.6) indicates minimal hallucination.

### Performance by Query Type

| Query Type | Precision@5 | Answer Quality |
|------------|-------------|----------------|
| Factual | 0.82 | 4.5 |
| Conceptual | 0.68 | 4.1 |
| Procedural | 0.75 | 4.3 |
| Comparison | 0.64 | 3.8 |
| Troubleshooting | 0.71 | 4.0 |

**Key Finding:** Struggles with comparison questions (requires multiple sources).

## Ablation Studies

### Experiment 1: Chunk Size Impact
| Chunk Size | Precision@5 | Answer Completeness |
|------------|-------------|---------------------|
| 300 tokens | 0.68 | 3.7 |
| 500 tokens | 0.72 | 4.0 |
| 1000 tokens | 0.70 | 4.2 |

**Conclusion:** 500 tokens balances retrieval precision and context completeness.

### Experiment 2: Model Comparison
| Model | Answer Quality | Latency | Cost/Query |
|-------|----------------|---------|------------|
| GPT-3.5 | 3.8 | 1.5s | $0.015 |
| GPT-4 | 4.2 | 2.8s | $0.030 |

**Conclusion:** GPT-4 worth the extra cost for quality-critical applications.

### Experiment 3: Re-ranking Impact
| Configuration | Precision@5 | Latency |
|---------------|-------------|---------|
| No re-ranking | 0.72 | 1.5s |
| Cross-encoder | 0.76 | 2.8s |
| LLM re-rank | 0.78 | 4.2s |

**Conclusion:** Cross-encoder offers best precision/latency trade-off.

## Failure Analysis

### Common Failure Modes
1. **Multi-hop queries** (35% failure rate)
   - Example: "Compare X and Y and recommend the best"
   - Issue: Requires synthesizing multiple chunks
   
2. **Ambiguous queries** (25% failure rate)
   - Example: "How do I fix it?"
   - Issue: Missing context

3. **Out-of-domain queries** (20% failure rate)
   - Example: Questions not in documents
   - Issue: System tries to answer anyway

### Mitigation Strategies
- Add multi-hop reasoning (Phase 5)
- Improve query clarification
- Better "I don't know" detection

## Comparison to Baselines

| System | Precision@5 | Answer Quality |
|--------|-------------|----------------|
| Naive RAG (vector-only, no rewrite) | 0.48 | 3.5 |
| **DocuSense** | **0.76** | **4.2** |

**Improvement:** +58% precision, +20% answer quality

## Conclusion
DocuSense demonstrates that systematic engineering (hybrid retrieval, query understanding, re-ranking) significantly improves RAG quality over naive approaches.
\`\`\`

#### 5. Design Decisions Document (DESIGN_DECISIONS.md)
Explain key choices:

```markdown
# Design Decisions

## 1. Hybrid Retrieval Over Vector-Only

**Decision:** Implement both vector and BM25 search, combined with fusion.

**Alternatives Considered:**
- Vector-only (simpler)
- BM25-only (no embeddings needed)

**Rationale:**
- Vector search excels at semantic similarity
- BM25 catches exact keyword matches (names, codes, technical terms)
- Evaluation showed 24% precision improvement
- Minimal added complexity (both are fast)

**Trade-offs:**
- Added complexity (two retrievers to maintain)
- Slightly increased latency (+200ms)
- Worth it: precision gain justifies cost

## 2. Reciprocal Rank Fusion Over Score Weighting

**Decision:** Use RRF to combine vector and BM25 results.

**Alternatives:**
- Weighted score fusion
- Linear combination

**Rationale:**
- RRF doesn't require score normalization
- More robust to score scale differences
- Literature supports effectiveness
- Simpler to implement

## 3. Cross-Encoder Re-ranking

**Decision:** Add optional cross-encoder re-ranking step.

**Alternatives:**
- No re-ranking
- LLM-based re-ranking

**Rationale:**
- Improves precision@5 by 5%
- Faster than LLM re-ranking (500ms vs 2s)
- Good precision/latency trade-off
- Can be toggled per query

**Trade-offs:**
- Adds latency
- Optional feature for quality-critical queries

## 4. Semantic Chunking Over Fixed-Size

**Decision:** Split on paragraph/sentence boundaries, target 500-1000 tokens.

**Alternatives:**
- Fixed 512-token chunks
- Sliding window

**Rationale:**
- Preserves semantic coherence
- Better for long-form answers
- Evaluation showed improved answer completeness

## 5. OpenAI Embeddings Over Local Models

**Decision:** Use OpenAI text-embedding-3-small by default.

**Alternatives:**
- all-MiniLM-L6-v2 (local)
- all-mpnet-base-v2 (local)

**Rationale:**
- Higher quality (better retrieval precision)
- No GPU needed
- Cost is acceptable ($0.00002 per 1K tokens)
- Can swap easily due to abstraction

**Trade-offs:**
- API dependency
- Recurring cost
- Mitigated by: support for local models as alternative

## 6. FAISS Over Managed Vector DBs

**Decision:** Use FAISS for vector storage.

**Alternatives:**
- Pinecone (managed)
- Milvus (self-hosted)
- Chroma

**Rationale:**
- Learning focus: understand internals
- No external dependencies
- Fast for small-medium datasets (<1M vectors)
- Easy to swap later

**Trade-offs:**
- No built-in distributed scaling
- In-memory (but supports disk persistence)
- Fine for prototype/learning

## 7. LLM-Based Query Understanding

**Decision:** Use GPT for intent classification and query rewriting.

**Alternatives:**
- Rule-based classification
- Fine-tuned BERT classifier

**Rationale:**
- Flexibility: easy to add new intent types
- High accuracy with good prompts
- Handles edge cases better
- Fast enough (GPT-3.5 <1s)

**Trade-offs:**
- API cost
- Non-deterministic
- Mitigated by: structured outputs (JSON mode)

## 8. GPT-4 for Answer Synthesis

**Decision:** Use GPT-4 for answer generation (with GPT-3.5 fallback).

**Rationale:**
- Evaluation showed 0.4-point quality improvement
- Better instruction-following
- Less hallucination
- Worth $0.015 extra cost per query

**Trade-offs:**
- Higher cost
- Slightly higher latency
- Mitigated by: configurable, can use GPT-3.5

## 9. Explicit Citation Requirements

**Decision:** Force model to cite sources in prompts.

**Rationale:**
- Enables verification
- Builds user trust
- Reduces hallucination
- Industry best practice

**Implementation:**
- [Document N] citation format
- Extract and validate citations
- Return sources with answer

## 10. Comprehensive Evaluation First

**Decision:** Build evaluation framework early (Phase 7).

**Rationale:**
- Can't optimize what you can't measure
- Enables A/B testing
- Validates design decisions
- Professional approach

**Impact:**
- Guided retrieval strategy choice
- Justified re-ranking addition
- Informed model selection
\`\`\`

#### 6. Code Quality & Organization
Clean up and document code:

```python
# Good practices to apply:

# 1. Type hints everywhere
def semantic_search(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Perform semantic similarity search.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        filters: Optional metadata filters
    
    Returns:
        List of chunk dictionaries with scores
        
    Example:
        >>> results = semantic_search("authentication", top_k=3)
        >>> print(results[0]['text'])
    """
    pass

# 2. Configuration classes
from dataclasses import dataclass

@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    use_reranking: bool = True
    vector_weight: float = 0.7

# 3. Error handling
class DocuSenseError(Exception):
    """Base exception for DocuSense."""
    pass

class RetrievalError(DocuSenseError):
    """Raised when retrieval fails."""
    pass

# 4. Logging
import logging

logger = logging.getLogger(__name__)

def process_query(query: str) -> dict:
    logger.info(f"Processing query: {query}")
    try:
        result = ...
        logger.debug(f"Retrieved {len(result['chunks'])} chunks")
        return result
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise
```

#### 7. Create Architectural Diagrams
Visual documentation:

```python
# Use diagrams-as-code
from diagrams import Diagram, Cluster
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL
from diagrams.custom import Custom

with Diagram("DocuSense Architecture", show=False):
    user = Custom("User", "./icons/user.png")
    
    with Cluster("Query Processing"):
        intent = Python("Intent Classifier")
        rewrite = Python("Query Rewriter")
        planner = Python("Strategy Planner")
    
    with Cluster("Retrieval"):
        vector = Custom("Vector Search", "./icons/faiss.png")
        bm25 = Python("BM25 Search")
        fusion = Python("Fusion")
        rerank = Python("Re-ranker")
    
    with Cluster("Generation"):
        llm = Custom("LLM", "./icons/openai.png")
        formatter = Python("Context Formatter")
    
    db = PostgreSQL("Vector Store")
    
    user >> intent >> rewrite >> planner
    planner >> [vector, bm25]
    [vector, bm25] >> fusion >> rerank
    rerank >> formatter >> llm >> user
```

#### 8. Create Demo Video/GIF
Show the system in action:

- Record Gradio dashboard usage
- Show query → results → answer flow
- Demonstrate comparison feature
- Show execution trace
- Highlight key features

### Dependencies
```
mkdocs              # Documentation site (optional)
diagrams            # Architecture diagrams
```

### Deliverables
- ✅ Comprehensive README
- ✅ Architecture documentation
- ✅ Evaluation report
- ✅ Design decisions document
- ✅ API documentation
- ✅ Code comments and docstrings
- ✅ Architectural diagrams
- ✅ Demo video/screenshots

### Success Criteria
- README is clear and compelling
- Architecture is well-explained
- Experiments are documented with results
- Design decisions are justified
- Code is clean and documented
- Can hand project to someone new and they understand it

---

## 🎯 Overall Project Success Criteria

### Technical Criteria
✅ System retrieves relevant chunks (Precision@5 > 0.6)  
✅ Generates accurate answers (Quality score > 4.0/5)  
✅ Handles edge cases gracefully  
✅ Responds in reasonable time (P95 < 5s)  
✅ Cost-effective (<$0.05 per query)  
✅ Modular, maintainable codebase  
✅ Comprehensive test coverage  

### Learning Criteria
✅ Understand RAG architecture deeply  
✅ Master hybrid retrieval strategies  
✅ Know how to evaluate LLM systems  
✅ Comfortable with prompt engineering  
✅ Experience with multiple LLM providers  
✅ Understand embedding and vector search  
✅ Can explain design trade-offs  

### Portfolio Criteria
✅ Professional documentation  
✅ Working demo  
✅ Measurable results  
✅ Clear narrative  
✅ Reproducible experiments  
✅ Shows engineering rigor  

---

## 📈 Recommended Learning Path

### Phase Sequencing
1. **Foundation** (Phases 0-2): Get infrastructure right
2. **Core RAG** (Phases 3-4): Build retrieval quality
3. **Polish** (Phases 6-7): Make it production-ready
4. **Evaluation** (Phase 7): Validate everything
5. **Showcase** (Phases 8-9): Make it presentable
6. **Agents** (Phase 5): Add later when ready

### Time Investment
- **Minimum viable:** Phases 0-2, 6 (1-2 weeks)
- **Solid project:** Add Phases 3-4, 7 (3-4 weeks)
- **Portfolio piece:** All phases except 5 (5-6 weeks)
- **Complete system:** All phases (7-8 weeks)

### Experiment-Driven Approach
Don't just implement—experiment:
- Try different configurations
- Measure everything
- Document what works and what doesn't
- Learn from failures
- Iterate based on data

---

## 🛠️ Technology Stack Summary

| Category | Technologies |
|----------|-------------|
| **LLMs** | OpenAI (GPT-4, GPT-3.5), Anthropic Claude (optional) |
| **Embeddings** | OpenAI text-embedding-3-small, sentence-transformers |
| **Vector Store** | FAISS (Chroma, Milvus as alternatives) |
| **Search** | BM25 (rank-bm25) |
| **Re-ranking** | Cross-encoder (sentence-transformers) |
| **Backend** | Python 3.10+, FastAPI |
| **UI** | Gradio |
| **Evaluation** | scikit-learn, rouge-score, bert-score |
| **Data** | pandas, numpy |
| **Logging** | loguru |
| **Testing** | pytest |
| **Docs** | Markdown, diagrams (optional: mkdocs) |

---

## 📚 Learning Resources

### Recommended Reading
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al.)
- "Dense Passage Retrieval for Open-Domain QA" (Karpukhin et al.)
- "Measuring Massive Multitask Language Understanding" (Hendrycks et al.)

### Industry Best Practices
- LangChain RAG patterns
- LlamaIndex retrieval strategies
- OpenAI RAG cookbook
- Anthropic prompt engineering guide

---

## ⚠️ Important Notes

### What This Project Is
- A comprehensive learning vehicle for LLM engineering
- A production-quality RAG system foundation
- A portfolio-ready demonstration of skills
- An experiment-driven exploration

### What This Project Is Not
- A production deployment (no auth, scaling, etc.)
- A SaaS product (focused on learning)
- A one-size-fits-all solution (specific use case)
- A black box (transparency emphasized)

### Key Principles
1. **Experiment > Perfection:** Try things, measure, learn
2. **Measure Everything:** No guessing, use data
3. **Transparency:** Understand what the system does
4. **Quality Over Speed:** Do it right, not fast
5. **Document As You Go:** Don't leave it for later

---

## 🎬 Next Steps

After reviewing this plan:
1. Set up development environment (Phase 0)
2. Collect sample documents for testing
3. Create initial golden query set (20 queries)
4. Start with Phase 0, work sequentially
5. Document experiments and learnings
6. Iterate based on evaluation results

**Remember:** This is a learning journey. The goal isn't just a working system—it's deep understanding through experimentation.

Good luck building DocuSense! 🚀
