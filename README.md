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
multi-turn chat · REST API · editorial web UI in light and dark

**Not built yet:** token revocation and password reset. See [Known limitations](docs/ARCHITECTURE.md#known-limitations).

Tests: 210 passing (unit + integration).
Run `python scripts/doctor.py` to check your environment before reporting a problem.

---

## Measured performance

Retrieval is evaluated on [QASPER](https://allenai.org/data/qasper) — 80 papers, 1,048
chunks, **259 questions** with evidence grounded to real chunk ids. Each arm answers the
same questions, so the comparison is paired; intervals are from 10,000 bootstrap
resamples. Full methodology, including what was dropped and why:
**[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

| Retrieval | MRR | NDCG@10 | P@1 | Recall@10 | ms/query |
|---|---|---|---|---|---|
| Vector only | 0.1867 | 0.1949 | 0.1004 | 0.2938 | 17 |
| + BM25, fused with RRF | 0.2244 | 0.2348 | 0.1313 | 0.3438 | 19 |
| **+ cross-encoder rerank** | **0.2824** | **0.2771** | **0.2046** | **0.3816** | 1782 |
| + query processing *(current default)* | 0.2777 | 0.2691 | 0.2046 | 0.3683 | 1791 |

What holds up under a significance test, and what does not:

- **Hybrid search helps.** +20.2% MRR over vector-only — Δ +0.0377, 95% CI
  [+0.0189, +0.0584], p = 0.0003.
- **Reranking helps most.** +25.8% MRR over hybrid alone — Δ +0.0581, 95% CI
  [+0.0194, +0.0962], p = 0.0044. It doubles P@1 and costs ~1.8s per query on CPU.
- **Query processing shows no measurable benefit.** Δ −0.0048, 95% CI
  [−0.0152, +0.0066], p = 0.43. Not a demonstrated penalty either — simply not doing
  anything detectable. [Why, and the section-labelling problem behind it.](docs/BENCHMARKS.md#what-the-numbers-say)

Running this exposed that the shipped default had reranking **switched off** — the
`USE_RERANKING` setting was overridden by the pipeline's `mode="balanced"`. The system
scored 0.2041 MRR, which is not significantly better than plain vector search
(p = 0.16), while the same components configured as documented reach 0.2777. That is
fixed, and it is why the numbers above are worth having: the gap was invisible until
something measured it.

Answer generation is scored separately on 30 questions with `llama3.2:3b`
(ROUGE-1 0.0594, completeness 0.6998). Those ROUGE figures are a regression signal, not
an accuracy score, and citation accuracy is *not* reported because QASPER carries no
author metadata to score it against — [both explained here](docs/BENCHMARKS.md#answer-quality).

```bash
python scripts/benchmark.py --papers 80        # reproduce (~20 min, CPU)
```

---

## What makes it different

| Capability | How it works |
|---|---|
| **Paper metadata extraction** | Pulls title, authors, year, venue, DOI/arXiv ID, abstract, 20+ section types, and both numbered `[1]` and author-year `(Smith, 2020)` citations — with a confidence score for "is this actually a paper?" |
| **Section-aware routing** | `detect_section_intent()` maps question phrasing to the right section, so "how did they train" doesn't retrieve from the related-work section. **Measured: no detectable benefit** on QASPER, because section labels are missing on 63% of chunks. On by default, `USE_SECTION_ROUTING=false` to disable — [the numbers](docs/BENCHMARKS.md#what-the-numbers-say) |
| **Metadata filtering from natural language** | `extract_academic_filters()` parses "recent papers", "2020–2023", "by Yoshua Bengio", "NeurIPS papers" into structured Qdrant filters — ranges included, which is what the benchmark caught: they raised a validation error and returned nothing until `build_filter()` learned to emit `Range` |
| **Hybrid retrieval** | Vector search for meaning + BM25 for exact terms (`BERT-base` shouldn't match `RoBERTa`), fused with Reciprocal Rank Fusion. **Measured: +20.2% MRR** over vector-only (p = 0.0003) |
| **Cross-encoder reranking** | Retrieves a wide candidate set, then reranks for precision. **Measured: +25.8% MRR** over hybrid alone (p = 0.0044), for ~1.8s per query |
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
pytest                          # everything (210 tests)
pytest -m integration           # real components, no mocks
pytest -m "not integration"     # unit tests only
python scripts/doctor.py        # check Qdrant / Ollama / Gemini / embeddings
```

### Benchmark

```bash
# One-time: fetch QASPER (~18MB extracted)
mkdir -p data/benchmarks/qasper
curl -L -o /tmp/qasper.tgz   https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
tar xzf /tmp/qasper.tgz -C data/benchmarks/qasper

python scripts/benchmark.py --papers 80               # retrieval ablation (~20 min)
python scripts/benchmark.py --papers 80 --answers 30  # + answer quality (needs Ollama)
python scripts/benchmark.py --papers 5 --arms vector,hybrid   # quick smoke test
```

Papers are ingested under a dedicated benchmark user id, so a run cannot see or disturb
your own documents. Results and methodology: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

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
| `USE_RERANKING` | `true` | Cross-encoder reranking. Worth +25.8% MRR for ~1.8s per query; set `false` to trade accuracy for latency |
| `USE_SECTION_ROUTING` | `true` | Restrict a question to the section it seems to be about. No measurable benefit on QASPER — see [BENCHMARKS.md](docs/BENCHMARKS.md) |

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
├── evaluation/     Retrieval + answer metrics, QASPER harness, benchmark runner
├── storage/        SQLite chunk and conversation stores
├── web/            Landing page, auth page, chat UI
└── rag_pipeline.py Top-level orchestrator (ingest / ask / chat)

tests/              Unit + integration tests (test_integration.py, test_auth.py)
docs/               Architecture notes, benchmark results and methodology;
                    docs/archive/ holds the original course plan
data/               Local documents, SQLite DB, vector store (gitignored)
scripts/            doctor.py (environment diagnostics), ingest.py (bulk ingestion),
                    benchmark.py (QASPER retrieval ablation)

Dockerfile          Two-stage build; models baked in so the first query isn't a download
docker-compose.yml  API + Qdrant + Ollama
.github/workflows/  CI: lint, tests on Python 3.11 and 3.13, Docker build
```

---

## Tech stack

Python · FastAPI · Qdrant · sentence-transformers · Ollama · rank-bm25 ·
Markitdown · SQLite · Gemini API · pytest

Built to run entirely on free/local infrastructure — no per-query API cost.
