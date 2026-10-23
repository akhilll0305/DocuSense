# DocuSense

**A research-paper RAG system where every component had to earn its place on a benchmark.**

DocuSense answers questions over research papers with hybrid retrieval, a cross-encoder
reranker, citations checked against their sources, and every sentence of the answer
checked against the retrieved text by an entailment model. Each of those was measured on
[QASPER](https://allenai.org/data/qasper) with paired significance tests before it was
kept. So was everything that *sounded* helpful and turned out not to be. Five features
were measured and switched off, including a reranker fine-tuned for this project.

**[Try it live](https://akhilll0305--docusense-web.modal.run)**. It sleeps when idle, so
the first load takes a few seconds.

---

## What it does

```
PDF / DOCX / TXT
      ↓
[ Ingestion ]   converters → preprocessing → paper metadata → section-tagged chunks
      ↓
[ Retrieval ]   vector + BM25, fused with RRF → cross-encoder rerank
      ↓
[ Generation ]  grounded answer → citations checked against the sources
      ↓
[ Claim check ] every sentence vs. the retrieved passages (NLI) → flagged in the UI
      ↓
[ API + UI ]    FastAPI, JWT auth, per-user isolation, vanilla HTML/CSS/JS
```

Storage is Qdrant for vectors and SQLite for documents and conversations. Embeddings,
reranking and claim checking run on ONNX Runtime with no torch in the deployed image.
Answers come from Ollama locally, or a hosted model when deployed.

---

## What was measured

Retrieval is evaluated on QASPER: 80 papers, 1,048 chunks, **259 questions** with
evidence grounded to real chunk ids. Arms answer the same questions, so they are compared
pairwise, with 95% intervals from 10,000 bootstrap resamples.

| Retrieval | MRR | NDCG@10 | P@1 | Recall@10 |
|---|---|---|---|---|
| Vector only | 0.1867 | 0.1949 | 0.1004 | 0.2938 |
| + BM25, fused with RRF | 0.2244 | 0.2348 | 0.1313 | 0.3438 |
| **+ cross-encoder rerank** *(default)* | **0.2824** | **0.2771** | **0.2046** | **0.3816** |

A clean re-run reproduces every figure to four decimal places.

### Kept

| Component | Evidence |
|---|---|
| Hybrid search | +20.2% MRR over vector only, p = 0.0003 |
| Cross-encoder rerank | a further +25.8% MRR, p = 0.0044; doubles P@1 |
| ONNX instead of torch | identical metrics to four decimals, 326MB resident instead of 758MB |
| Claim verification | NLI judge validated on QASPER dev before use: ROC-AUC 0.750 |
| Embedder fine-tuned on QASPER train | on test, **+32% MRR** for vector search and **+26%** for hybrid, both p = 0.0002. Fine-tuned hybrid alone matches stock hybrid + rerank (0.283 vs 0.282 MRR) at 26 ms a query instead of 2.2 s. Deployed |

### Measured and switched off

| Component | Why |
|---|---|
| Section routing | the evidence is in the routed section only 13.3% of the time |
| LLM query rewriting | **−18% MRR**, p = 0.003; the rewriter invents context it cannot see |
| Query expansion, intent classification | nothing downstream reads their output |
| Fine-tuned reranker | +8% on dev, then **+0.0062 MRR on test, p = 0.47**: not a demonstrated gain |
| xsmall NLI model | flagged 74% of *supported* claims; replaced by DeBERTa-v3-base |

### In progress

| Component | Status |
|---|---|
| QASPER Answer-F1 / Evidence-F1 end to end | the official metric is implemented and cross-checked against AllenAI's evaluator; the full run is pending |

Methodology, significance tests, every negative result, and the two ways the benchmark
was caught measuring the wrong corpus: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

```bash
python scripts/benchmark.py --papers 80        # retrieval ablation (~25 min, CPU)
python scripts/qasper_answers.py --papers 80   # Answer-F1 / Evidence-F1
```

---

## What makes it different

| | |
|---|---|
| **Claims are checked, not just citations** | Every sentence of an answer is scored against windows of the retrieved text by an entailment model; the UI shows which ones nothing supports, with the closest passage beside them. Citation checking alone cannot catch the right paper cited for the wrong number |
| **Citations are verified** | Each citation is matched against the retrieved sources by author and year; unsupported ones are deleted rather than shown |
| **Trained on the task** | `scripts/finetune_embedder.py` and `scripts/finetune_reranker.py` fine-tune on QASPER train (chosen on dev, never on test) and export ONNX that reproduces torch exactly |
| **Hybrid retrieval that honours filters** | Vector search for meaning, BM25 for exact terms (`BERT-base` shouldn't match `RoBERTa`). Document, year and section filters apply to *both* halves, which they did not until this was measured |
| **Paper metadata extraction** | Title, authors, year, venue, DOI/arXiv ID, abstract, 20+ section types, and both `[1]` and `(Smith, 2020)` citation styles |
| **Per-user isolation** | Documents, vectors, BM25 corpora and conversations are scoped to their owner; cross-tenant reads return 404 |
| **Benchmarks that refuse to lie** | The harness refuses to report on a partly ingested corpus, on stale duplicate vectors, or on two runs that asked different questions |

---

## Setup

**Docker** runs the API, Qdrant and Ollama together:

```bash
echo "JWT_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" > .env
docker compose up -d
docker compose exec ollama ollama pull llama3.2:3b   # one time, ~2GB
```

**Local** needs Python 3.10+ and [Ollama](https://ollama.ai/):

```bash
python -m venv venv && source venv/Scripts/activate   # or venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2:3b && ollama serve

uvicorn docusense.api.app:app --reload                # → http://localhost:8000
```

Then open the app, create an account, and drop in a PDF. `python scripts/doctor.py` checks
your environment if anything misbehaves. The claim checker downloads its model (~240MB)
on the first answer; `ENABLE_CLAIM_VERIFICATION=false` turns it off.

---

## Common tasks

```bash
python scripts/ingest.py path/to/paper.pdf     # or a whole directory
python scripts/doctor.py                       # check Qdrant / models / provider
python scripts/reset_password.py --list        # operator account recovery

pip install -r requirements-dev.txt && pytest  # 390 tests
python scripts/benchmark.py --papers 80        # QASPER retrieval ablation
python scripts/qasper_answers.py --papers 80   # QASPER Answer-F1 / Evidence-F1
python scripts/validate_claim_verifier.py      # how good the claim checker is
```

Fine-tuning needs the torch runtime and, realistically, a GPU:

```bash
pip install -r requirements-torch.txt accelerate datasets
python scripts/finetune_embedder.py            # ~10 min on an RTX 4060
python scripts/finetune_reranker.py --eval-steps 20
```

Sessions are revocable server-side, which a signed JWT does not allow on its own:
`POST /api/auth/logout`, `/logout-all`, `/password`. There is no self-service password
reset by design. Nothing here can prove who owns an email address, so recovery is
`scripts/reset_password.py`, run by whoever operates the instance.

---

## Configuration

Everything lives in `docusense/config/settings.py` and is overridable via `.env`. The ones
that matter:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `groq` for deployments, where a 3B local model does not fit |
| `MODEL_RUNTIME` | `onnx` | Or `torch`. Same weights, same numbers; ONNX costs 326MB against 758MB |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | `models/minilm-qasper` after `scripts/finetune_embedder.py` (+26% hybrid MRR on test; the deployment uses it). Re-ingest after switching |
| `USE_RERANKING` | `true` | +25.8% MRR, and most of the query time. Needs a real vCPU |
| `RERANK_CANDIDATES` | 4 × top_k | How many candidates the cross-encoder scores; the main latency lever |
| `ENABLE_CLAIM_VERIFICATION` | `true` | Check each answer sentence against the sources (NLI, ~500MB resident) |
| `USE_SECTION_ROUTING` | `false` | Measured to cost accuracy; see [BENCHMARKS.md](docs/BENCHMARKS.md) |
| `QUERY_LLM_BACKEND` | `off` | LLM query rewriting. Measured to cost 18% MRR |
| `QDRANT_MODE` | `disk` | Auto-switches to `server` when `QDRANT_URL` and `QDRANT_API_KEY` are set |
| `MAX_DOCUMENTS_PER_USER` | `0` | Per-account cap; `0` is unlimited. Set a real number on a public instance |
| `SEED_DEMO` | `false` | Seed a shared demo account on start (idempotent, so safe on persistent storage) |

---

## Deploy

Locally, Ollama generates answers on your own machine, with no per-query cost, and
documents never leave the laptop. That does not survive a free hosting tier, so
generation is a seam: `LLM_PROVIDER=groq` points it at a hosted model and everything
else is unchanged.

It is deployed on **Modal's free tier** with no credit card. Measured on the live
instance once warm: retrieval with reranking takes about 0.4s, the hosted model takes 1–19s
depending on how much it writes, and the claim check adds roughly 0.5s per claim (1–6s an
answer). The UI streams the answer first and fills in the check after, so the check does
not delay the text. What was measured to choose the host, and why not Hugging Face,
Render or Cloud Run: **[deploy/DEPLOY.md](deploy/DEPLOY.md)**.

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
├── generation/     Answer generation, citations, claim verification, conversation memory
├── llms/           Ollama and Groq behind one interface
├── training/       QASPER training pairs for the embedder and reranker
├── evaluation/     Metrics, QASPER harness, official QASPER scorer
├── storage/        SQLite chunk and conversation stores
├── web/            Landing page, auth page, chat UI
└── rag_pipeline.py Top-level orchestrator

docs/               Architecture notes, benchmarks and methodology
deploy/             Modal app, Cloud Run script, deployment notes
scripts/            benchmark, qasper_answers, finetune_*, validate_claim_verifier, …
tests/              390 tests, unit and integration
```

Architecture and known limitations: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Stack

Python · FastAPI · Qdrant · ONNX Runtime · sentence-transformers (training) · DeBERTa-v3
NLI · Ollama · Groq · rank-bm25 · Markitdown · SQLite · pytest
