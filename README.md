# DocuSense

**A research-paper RAG system that understands academic structure.**

Most RAG systems treat a PDF as a flat wall of text. DocuSense parses the *structure* of a
research paper — title, authors, venue, year, sections, citations — and uses it at query time.
Ask "how did they train it?" and it searches the methodology section. Ask "what accuracy did
they get?" and it searches results. Ask for "papers from 2020–2023 by Bengio" and it filters
on metadata before it ever runs a vector search.

Answers come back with inline academic citations and a formatted reference list.

---

## Architecture

```
PDF / DOCX / TXT
      |
      v
[ Ingestion ]      converters -> preprocessor -> paper_metadata -> chunker
      |            Markitdown        cleaning     title/authors/    semantic,
      |                                           sections/cites    section-tagged
      v
[ Storage ]        SQLite (users, documents, chunks, conversations)
      |            every row scoped to its owning user
      |
      v
[ Embeddings ]     all-MiniLM-L6-v2 (384-dim, local)
      |
      v
[ Vector Store ]   Qdrant + user_id tenant index + 8 academic payload indexes
      |                    (paper_title, authors, year, section_type,
      |                     venue, paper_type, has_equations, has_citations)
      v
[ Retrieval ]      query_processor -> hybrid_search -> reranker
                   section routing    Vector + BM25    cross-encoder
                   + metadata filters  -> RRF fusion   ms-marco-MiniLM
      |
      v
[ Generation ]     answer_generator -> citation_formatter
                   Ollama llama3.2      APA / references / BibTeX
      |
      v
[ API + Web UI ]   FastAPI (14 endpoints, JWT-protected) + vanilla HTML/CSS/JS
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module-level detail.

---

## Status

The pipeline runs end to end: ingest a paper, ask a question, get a grounded answer with
inline citations and an APA reference list. Multi-turn chat resolves pronouns against
conversation history.

**Working:** email/password accounts with bcrypt + JWT · per-user document isolation ·
ingestion with paper metadata extraction · section-tagged chunking · hybrid vector + BM25
retrieval with RRF · section routing and academic filters · cross-encoder reranking ·
cited answer generation with fabricated-citation filtering · streamed responses (SSE) ·
multi-turn chat · REST API · web UI

**Not built yet:** token revocation, password reset, and published benchmark numbers. See [Known limitations](docs/ARCHITECTURE.md#known-limitations).

Tests: 165 passing (unit + integration), 74% coverage.
Run `python scripts/doctor.py` to check your environment before reporting a problem.

---

## What makes it different

| Capability | How it works |
|---|---|
| **Paper metadata extraction** | Pulls title, authors, year, venue, DOI/arXiv ID, abstract, 20+ section types, and both numbered `[1]` and author-year `(Smith, 2020)` citations — with a confidence score for "is this actually a paper?" |
| **Section-aware routing** | `detect_section_intent()` maps question phrasing to the right section, so "how did they train" doesn't retrieve from the related-work section |
| **Metadata filtering from natural language** | `extract_academic_filters()` parses "recent papers", "2020–2023", "by Yoshua Bengio", "NeurIPS papers" into structured Qdrant filters |
| **Hybrid retrieval** | Vector search for meaning + BM25 for exact terms (`BERT-base` shouldn't match `RoBERTa`), fused with Reciprocal Rank Fusion |
| **Cross-encoder reranking** | Retrieves a wide candidate set, then reranks for precision |
| **Grounded citations** | Every claim is traced to a source chunk; exports APA reference lists and BibTeX |
| **Fabricated citations removed** | Small models invent citations even when told not to, so every citation is checked against the retrieved sources by author and year, and unsupported ones are deleted rather than trusted |
| **Streamed answers** | Local generation takes tens of seconds, so answers arrive token by token over SSE, with progress before the first token |
| **Per-user isolation** | Documents, vectors, BM25 corpora, and conversations are all scoped to their owner; cross-tenant reads return 404 rather than revealing that an id exists |

---

## Setup

### Docker (everything included)

Brings up the API, Qdrant, and Ollama together — no local Python or Ollama install:

```bash
echo "JWT_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" > .env
docker compose up -d
docker compose exec ollama ollama pull llama3.2:3b   # one time, ~2GB
```

Then open http://localhost:8000.

Compose runs Qdrant as a **server** rather than the embedded on-disk store, which matters:
the embedded store allows only one process at a time, and it silently ignores payload
indexes, so every metadata filter degrades to a full scan.

### Local install

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.ai/), and a Qdrant instance
(local disk mode works out of the box — no server needed).

```bash
# 1. Virtual environment
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash)
# source venv/bin/activate        # macOS / Linux

# 2. Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configuration
cp .env.example .env
# Optional: add GEMINI_API_KEY for LLM-based query rewriting.
# Everything else has working local defaults.

# 4. Local LLM for answer generation
ollama pull llama3.2:3b
ollama serve
```

### Run

```bash
uvicorn docusense.api.app:app --reload
```

- Web UI — http://localhost:8000
- API docs — http://localhost:8000/docs

### Sign in

Open http://localhost:8000 and create an account. Documents you upload are visible
only to you.

For production, set a signing key — the app refuses to start without one when
`ENVIRONMENT=prod`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"   # -> JWT_SECRET_KEY
```

### Ingest documents

```bash
python scripts/ingest.py path/to/paper.pdf
python scripts/ingest.py data/papers/            # a whole directory
python scripts/ingest.py --reset data/papers/    # wipe the vector store first
```

### Test

```bash
pytest                          # everything (165 tests)
pytest -m integration           # real components, no mocks
pytest -m "not integration"     # unit tests only
python scripts/doctor.py        # check Qdrant / Ollama / Gemini / embeddings
```

---

## Configuration

All settings live in `docusense/config/settings.py` and are overridable via `.env`.
The ones that matter most:

| Variable | Default | Notes |
|---|---|---|
| `QDRANT_MODE` | `disk` | `memory` / `disk` / `server`. Auto-switches to `server` when `QDRANT_URL` + `QDRANT_API_KEY` are both set |
| `QDRANT_URL` | — | Set only for Qdrant Cloud or a self-hosted server |
| `OLLAMA_MODEL` | `llama3.2:3b` | Any model available to your Ollama install |
| `GEMINI_API_KEY` | — | Optional. Enables query rewriting/expansion; the system degrades gracefully without it |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 384-dim. Changing this requires re-ingesting |
| `TARGET_CHUNK_TOKENS` | `500` | Chunk sizing (range 200–800) |

---

## Project layout

```
docusense/
├── api/            FastAPI app, routes, dependencies, Pydantic schemas
├── auth/           Password hashing, JWT, user store
├── config/         Centralized pydantic-settings configuration
├── ingestion/      Converters, preprocessing, paper metadata, chunking
├── embeddings/     sentence-transformers wrapper
├── vectorstore/    Qdrant client + academic payload indexes
├── retrieval/      Query processing, hybrid search, reranking
├── generation/     Answer generation, citations, conversation memory
├── evaluation/     Retrieval + answer metrics, QASPER, benchmark runner
├── storage/        SQLite chunk and conversation stores
├── web/            Landing page, auth page, chat UI
└── rag_pipeline.py Top-level orchestrator (ingest / ask / chat)

tests/              Unit + integration tests (test_integration.py, test_auth.py)
docs/               Architecture notes; docs/archive/ holds the original course plan
data/               Local documents, SQLite DB, vector store (gitignored)
scripts/            doctor.py (environment diagnostics), ingest.py (bulk ingestion)

Dockerfile          Two-stage build; models baked in so the first query isn't a download
docker-compose.yml  API + Qdrant + Ollama
.github/workflows/  CI: lint, tests on Python 3.11 and 3.13, Docker build
```

---

## Tech stack

Python · FastAPI · Qdrant · sentence-transformers · Ollama · rank-bm25 ·
Markitdown · SQLite · Gemini API · pytest

Built to run entirely on free/local infrastructure — no per-query API cost.
