# DocuSense

**A research-paper RAG system that understands academic structure.**

Most RAG systems treat a PDF as a flat wall of text. DocuSense parses the *structure* of a
research paper — title, authors, venue, year, sections, citations — and uses it at query
time. Every chunk is tagged with the section it came from, and every answer comes back
with citations checked against the sources they claim.

**[Try it live](https://akhilll0305--docusense-web.modal.run)** — sign in with
`demo@docusense.app` / `read-the-papers`. It sleeps when idle, so the first load takes a
few seconds.

---

## What it does

```
PDF / DOCX / TXT
      ↓
[ Ingestion ]   converters → preprocessing → paper metadata → section-tagged chunks
      ↓
[ Retrieval ]   vector + BM25, fused with RRF → cross-encoder rerank
      ↓
[ Generation ]  grounded answer → citations verified against the sources
      ↓
[ API + UI ]    FastAPI, JWT auth, per-user isolation, vanilla HTML/CSS/JS
```

Storage is Qdrant for vectors and SQLite for documents and conversations. Embeddings and
reranking run locally on ONNX Runtime; answers come from Ollama locally, or a hosted model
when deployed.

---

## Measured performance

Retrieval is evaluated on [QASPER](https://allenai.org/data/qasper) — 80 papers, 1,048
chunks, **259 questions** with evidence grounded to real chunk ids. Arms are paired;
intervals come from 10,000 bootstrap resamples.

| Retrieval | MRR | NDCG@10 | P@1 | Recall@10 |
|---|---|---|---|---|
| Vector only | 0.1867 | 0.1949 | 0.1004 | 0.2938 |
| + BM25, fused with RRF | 0.2244 | 0.2348 | 0.1313 | 0.3438 |
| **+ cross-encoder rerank** *(default)* | **0.2824** | **0.2771** | **0.2046** | **0.3816** |

- **Hybrid search helps** — +20.2% MRR, p = 0.0003.
- **Reranking helps most** — a further +25.8% MRR, p = 0.0044. It doubles P@1 and is where
  the query time goes.
- **Three features were measured and turned off.** Section routing costs accuracy because
  the evidence is in the section it routes to only 13.3% of the time. LLM query rewriting
  costs 18% MRR because the rewriter cannot see the corpus and invents context. Query
  expansion and intent classification reach nothing downstream at all.

The full methodology, the significance tests, and how each of those was found:
**[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

```bash
python scripts/benchmark.py --papers 80        # reproduce (~25 min, CPU)
```

---

## What makes it different

| | |
|---|---|
| **Paper metadata extraction** | Title, authors, year, venue, DOI/arXiv ID, abstract, 20+ section types, and both `[1]` and `(Smith, 2020)` citation styles |
| **Section-tagged chunks** | Each chunk is classified from its full chain of enclosing headers, so "Experiments > Baseline Models" is `experiments`. Chunks with no usable label fell from 62.6% to 27.3% |
| **Hybrid retrieval** | Vector search for meaning, BM25 for exact terms (`BERT-base` shouldn't match `RoBERTa`), fused with RRF |
| **Verified citations** | Every citation is checked against the retrieved sources by author and year; unsupported ones are deleted rather than shown. Small models invent references even when told not to |
| **Runs on ONNX, not torch** | Same models, same numbers to four decimal places, 326MB resident instead of 758MB — which is what makes free hosting possible |
| **Per-user isolation** | Documents, vectors, BM25 corpora and conversations are scoped to their owner; cross-tenant reads return 404 |

---

## Setup

**Docker** — API, Qdrant and Ollama together:

```bash
echo "JWT_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" > .env
docker compose up -d
docker compose exec ollama ollama pull llama3.2:3b   # one time, ~2GB
```

**Local** — needs Python 3.10+ and [Ollama](https://ollama.ai/):

```bash
python -m venv venv && source venv/Scripts/activate   # or venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2:3b && ollama serve

uvicorn docusense.api.app:app --reload                # → http://localhost:8000
```

Then open the app, create an account, and drop in a PDF. `python scripts/doctor.py` checks
your environment if anything misbehaves.

---

## Common tasks

```bash
python scripts/ingest.py path/to/paper.pdf     # or a whole directory
python scripts/doctor.py                       # check Qdrant / models / provider
python scripts/reset_password.py --list        # operator account recovery

pip install -r requirements-dev.txt && pytest  # 338 tests
python scripts/benchmark.py --papers 80        # QASPER ablation
```

Sessions are revocable server-side, which a signed JWT does not allow on its own:
`POST /api/auth/logout`, `/logout-all`, `/password`. There is no self-service password
reset by design — nothing here can prove who owns an email address, so recovery is
`scripts/reset_password.py`, run by whoever operates the instance.

---

## Configuration

Everything lives in `docusense/config/settings.py` and is overridable via `.env`. The ones
that matter:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `groq` for deployments, where a 3B local model does not fit |
| `MODEL_RUNTIME` | `onnx` | Or `torch`. Same weights, same numbers; ONNX costs 326MB against 758MB |
| `USE_RERANKING` | `true` | +25.8% MRR, and most of the query time. Needs a real vCPU |
| `USE_SECTION_ROUTING` | `false` | Measured to cost accuracy — see [BENCHMARKS.md](docs/BENCHMARKS.md) |
| `QUERY_LLM_BACKEND` | `off` | LLM query rewriting. Measured to cost 18% MRR |
| `QDRANT_MODE` | `disk` | Auto-switches to `server` when `QDRANT_URL` and `QDRANT_API_KEY` are set |
| `MAX_DOCUMENTS_PER_USER` | `0` | Per-account cap; `0` is unlimited. Set a real number on a public instance |
| `SEED_DEMO` | `false` | Seed a shared demo account on start, for instances with ephemeral storage |

---

## Deploy

Locally, Ollama generates answers on your own machine — no per-query cost, and documents
never leave the laptop. That does not survive a free hosting tier, so generation is a seam:
`LLM_PROVIDER=groq` points it at a hosted model and everything else is unchanged.

It is deployed on **Modal's free tier** — no credit card — and answers in about two seconds
once warm. What was measured to get there, and why not Hugging Face, Render or Cloud Run:
**[deploy/DEPLOY.md](deploy/DEPLOY.md)**.

---

## Layout

```
docusense/
├── api/            FastAPI app, routes, auth, schemas
├── auth/           Password hashing, JWT, user store
├── config/         Centralised pydantic-settings configuration
├── ingestion/      Converters, preprocessing, paper metadata, chunking
├── embeddings/     Embedding models, and the ONNX/torch runtime seam
├── vectorstore/    Qdrant client + academic payload indexes
├── retrieval/      Query processing, hybrid search, reranking
├── generation/     Answer generation, citations, conversation memory
├── llms/           Ollama and Groq behind one interface
├── evaluation/     Metrics, QASPER harness, benchmark runner
├── storage/        SQLite chunk and conversation stores
├── web/            Landing page, auth page, chat UI
└── rag_pipeline.py Top-level orchestrator

docs/               Architecture notes, benchmarks and methodology
deploy/             Modal app, Cloud Run script, deployment notes
scripts/            doctor, ingest, benchmark, seed_demo, reset_password
tests/              338 tests, unit and integration
```

Architecture and known limitations: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Stack

Python · FastAPI · Qdrant · ONNX Runtime · Ollama · Groq · rank-bm25 · Markitdown ·
SQLite · pytest
