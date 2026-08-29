# DocuSense

**A research-paper RAG system that understands academic structure.**

Most RAG systems treat a PDF as a flat wall of text. DocuSense parses the *structure* of a
research paper — title, authors, venue, year, sections, citations — and uses it at query time.
Ask for "papers from 2020–2023 by Bengio" and it filters on metadata before it ever runs a
vector search. Every chunk is tagged with the section it came from, and every answer is
traced back to the chunks that support it.

Routing a question to a single section is also built in, and is switched **off** by
default: it was measured, it costs accuracy, and [the numbers are
published](docs/BENCHMARKS.md#what-the-numbers-say) rather than the feature quietly
removed.

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
                   metadata filters   Vector + BM25    cross-encoder
                   (section routing    -> RRF fusion   ms-marco-MiniLM
                    off by default)
      |
      v
[ Generation ]     answer_generator -> citation_formatter
                   Ollama llama3.2      APA / references / BibTeX
      |
      v
[ API + Web UI ]   FastAPI (18 endpoints, JWT-protected) + vanilla HTML/CSS/JS
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module-level detail.

---

## Status

The pipeline runs end to end: ingest a paper, ask a question, get a grounded answer with
inline citations and an APA reference list. Multi-turn chat resolves pronouns against
conversation history.

**Working:** email/password accounts with bcrypt + JWT · revocable sessions (sign out here
or everywhere) · password change · per-user document isolation ·
ingestion with paper metadata extraction · section-tagged chunking · hybrid vector + BM25
retrieval with RRF · academic metadata filters · cross-encoder reranking ·
cited answer generation with fabricated-citation filtering · streamed responses (SSE) ·
multi-turn chat · REST API · editorial web UI in light and dark

**Not built yet:** self-service password reset and email verification — both need an
email channel, and resetting on request alone would hand any account to anyone who knows
its address. Recovery is `python scripts/reset_password.py <email>`, run by whoever
operates the instance. See [Known limitations](docs/ARCHITECTURE.md#known-limitations).

Tests: 315 passing (unit + integration).
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
| Vector only | 0.1867 | 0.1949 | 0.1004 | 0.2938 | 31 |
| + BM25, fused with RRF | 0.2244 | 0.2348 | 0.1313 | 0.3438 | 37 |
| **+ cross-encoder rerank** *(current default)* | **0.2824** | **0.2771** | **0.2046** | **0.3816** | 1908 |
| + query processing | 0.2824 | 0.2771 | 0.2046 | 0.3816 | 1904 |

What holds up under a significance test, and what does not:

- **Hybrid search helps.** +20.2% MRR over vector-only — Δ +0.0377, 95% CI
  [+0.0189, +0.0584], p = 0.0003.
- **Reranking helps most.** +25.8% MRR over hybrid alone — Δ +0.0581, 95% CI
  [+0.0194, +0.0962], p = 0.0044. It doubles P@1 and costs ~1.8s per query on CPU.
- **Query processing does nothing at all here.** Not "not significant" — the difference
  is exactly 0.0000 on all 259 queries. All three of its parts are inert in the shipped
  configuration: LLM rewriting is off, academic filters fire on no QASPER question,
  and section routing is now off. [Details.](docs/BENCHMARKS.md#retrieval-ablation)
- **LLM query rewriting was never running, and when switched on it costs accuracy.**
  It was wired to Gemini alone, whose key on this project returns 403 — so three
  benchmark runs measured it as *absent* and reported an exact zero. Routed through the
  same seam generation uses, it rewrote 259 of 259 queries and dropped MRR from 0.2824
  to 0.2305 — Δ −0.0519, 95% CI [−0.0859, −0.0174], p = 0.0033 — losing the top-ranked
  evidence on 24 of the 53 questions that had it. The rewriter cannot see the corpus,
  so it fills gaps from its own priors: *"What are the five domains?"* became *"...of
  the CompTIA A+ certification exam"*. Off by default.
  [The whole investigation.](docs/BENCHMARKS.md#what-the-numbers-say)
- **Section routing costs accuracy, so it is off by default.** On the 60 questions it
  fires on it drops MRR from 0.2426 to 0.1903 and Recall@5 by 44% relative. The reason is
  not the section labels — those were repaired, and routing got *worse*. The evidence
  that answers a question is in the section it routes to only **13.3%** of the time.
  [The whole investigation.](docs/BENCHMARKS.md#what-the-numbers-say)

Running this exposed that the shipped default had reranking **switched off** — the
`USE_RERANKING` setting was overridden by the pipeline's `mode="balanced"`. The system
scored 0.2041 MRR, which is not significantly better than plain vector search
(p = 0.16), while the same components configured as documented reach 0.2824. That is
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
| **Section-tagged chunks** | Every chunk carries the section it came from, classified from its full chain of enclosing headers, so "Experiments > Baseline Models" is tagged `experiments` and "Model > Background" is tagged `methodology` rather than `introduction`. **Measured: chunks with no usable label fell from 62.6% to 27.3%** on the benchmark corpus |
| **Section-aware routing** *(off by default)* | `detect_section_intent()` maps question phrasing to a section and filters on it. **Measured: it costs accuracy** — the evidence answering a question is in the section it routes to only 13.3% of the time, so the filter hides the answer more often than it finds it. `USE_SECTION_ROUTING=true` to enable — [the numbers, and why better labels made it worse](docs/BENCHMARKS.md#what-the-numbers-say) |
| **Query rewriting** *(off by default)* | An LLM restates the question before it is searched, through the same seam that generates answers, so it runs wherever generation does. **Measured: it costs 18% MRR** (p = 0.0033) — the rewriter cannot see the corpus, so it invents the context an under-specified question is missing. Kept, off, with [the numbers published](docs/BENCHMARKS.md#what-the-numbers-say) |
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

Sessions can be ended server-side, which a signed JWT does not allow on its own:

```bash
POST /api/auth/logout        # this session only; other devices keep working
POST /api/auth/logout-all    # every session for the account, in one write
POST /api/auth/password      # change the password; ends every other session
```

Forgotten passwords are reset from the machine running DocuSense, because
nothing here can prove who owns an email address:

```bash
python scripts/reset_password.py user@example.com --generate
python scripts/reset_password.py --list
```

### Ingest documents

```bash
python scripts/ingest.py path/to/paper.pdf
python scripts/ingest.py data/papers/            # a whole directory
python scripts/ingest.py --reset data/papers/    # wipe the vector store first
```

### Test

```bash
pytest                          # everything (315 tests)
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

## Deploy

Locally, answers are generated by Ollama on your own machine: no per-query cost,
no rate limit, and documents never leave the laptop. That does not survive
contact with a free hosting tier — `llama3.2:3b` wants about 4GB of RAM and
answers in roughly 27 seconds on a shared CPU.

So generation is a seam. `LLM_PROVIDER=groq` points it at a hosted model on a
free tier and answers come back in one to two seconds; retrieval, reranking,
citation checking and the UI are unchanged. Everything else the deployment needs
is in the repo: the image reads `PORT`, seeds a demo account from `data/demo/` on
start (free tiers have ephemeral storage), and caps documents and upload size
because a public URL with open sign-up is otherwise a public disk.

What retrieval still needs locally is memory: torch, the embedding model and the
cross-encoder peak at **730 MB** answering one question, and 627 MB with
reranking switched off — so the 512 MB free tiers cannot run this at all, and
the target is Google Cloud Run at 1–2 GiB. Hugging Face Spaces was the previous
plan and is no longer available on a free account: both the Docker and Gradio
SDKs now answer `402 … requires a PRO subscription`.

Step by step, with the measurements behind those numbers:
**[deploy/DEPLOY.md](deploy/DEPLOY.md)**.

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
| `USE_SECTION_ROUTING` | `false` | Restrict a question to the section it seems to be about. Measured to cost accuracy on QASPER — see [BENCHMARKS.md](docs/BENCHMARKS.md) |
| `LLM_PROVIDER` | `ollama` | Which backend generates answers. `groq` points at a hosted model for deployments, where a 3B local model does not fit — see [deploy/DEPLOY.md](deploy/DEPLOY.md) |
| `QUERY_LLM_BACKEND` | `off` | Which model rewrites the query before searching: `gemini`, `provider` (whatever `LLM_PROVIDER` points at), or `off`. Measured on QASPER it costs 18% MRR — [why](docs/BENCHMARKS.md#what-the-numbers-say) |
| `GROQ_API_KEY` | — | Required only when `LLM_PROVIDER=groq` |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Which hosted model answers. Groq retires ids without notice, so `doctor.py` checks the id against the account's model list and a failed answer names the models the key can actually use |
| `MAX_DOCUMENTS_PER_USER` | `0` | Per-account document cap; `0` means no limit. Set a real number on a public instance |
| `SEED_DEMO` | `false` | Seed a shared demo account from `data/demo/` on start. For public instances with ephemeral storage |

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
                    benchmark.py (QASPER retrieval ablation),
                    reset_password.py (operator account recovery),
                    seed_demo.py (demo shelf for a public instance)
deploy/             DEPLOY.md, the Cloud Run deploy script, and a
                    Hugging Face Space card kept for a PRO account

Dockerfile          Two-stage build; models baked in so the first query isn't a download
docker-compose.yml  API + Qdrant + Ollama
.github/workflows/  CI: lint, tests on Python 3.11 and 3.13, Docker build
```

---

## Tech stack

Python · FastAPI · Qdrant · sentence-transformers · Ollama · rank-bm25 ·
Markitdown · SQLite · Gemini API · pytest

Built to run entirely on free/local infrastructure — no per-query API cost.
