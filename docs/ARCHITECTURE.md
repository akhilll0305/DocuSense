# DocuSense Architecture

Module-level reference for the system. For setup and overview see the [README](../README.md).

---

## Request flow

There are two entry paths, both landing on `DocuSenseRAG` (`docusense/rag_pipeline.py`),
which is the single orchestrator for the whole system.

### Ingestion: `POST /api/ingest`

```
UploadFile
  -> routes.ingest_document()        writes to a temp file
  -> DocuSenseRAG.ingest()
       -> DocumentPipeline.process_document()
            converters.py        Markitdown -> Markdown (PyPDF2/pdfplumber fallback)
            image_processor.py   Gemini Vision or OCR (optional, off by default)
            preprocessor.py      Unicode normalization, whitespace, artifact removal
            paper_metadata.py    title/authors/year/venue/sections/citations + confidence
            chunker.py           semantic chunking, header-aware, section-tagged
            chunk_store.py       persist to SQLite
       -> EmbeddingGenerator.embed_batch()     all-MiniLM-L6-v2, 384-dim
       -> QdrantVectorStore batch upsert       payload carries academic metadata
  -> IngestResult
```

### Query: `POST /api/ask`

```
AskRequest
  -> routes.ask_question()
  -> DocuSenseRAG.ask()
  -> GenerationPipeline.generate()
       -> RetrievalPipeline.retrieve()
            query_processor.py   rewrite/expand (Gemini), section intent,
                                 academic filter extraction
            hybrid_search.py     vector search + BM25 -> RRF fusion
            reranker.py          cross-encoder rerank (accurate mode)
       -> AnswerGenerator.generate()      Ollama, context-grounded prompt
       -> CitationFormatter               inline citations, reference list, BibTeX
  -> PipelineResponse (answer, sources, citations, timings, confidence)
```

---

## Modules

### `config/`
`settings.py` — one `pydantic-settings` class for the entire system, loaded from `.env`.
Creates required data directories on instantiation. Exposes `effective_qdrant_mode`, which
promotes the mode to `server` when both `QDRANT_URL` and `QDRANT_API_KEY` are set.

### `ingestion/`
| File | Responsibility |
|---|---|
| `converters.py` | Any supported format -> Markdown, via Markitdown with PDF-parser fallbacks |
| `image_processor.py` | Figure/chart description via Gemini Vision, OCR fallback. Off by default (slow, rate-limited) |
| `preprocessor.py` | Unicode normalization, whitespace collapsing, PDF artifact removal |
| `paper_metadata.py` | The academic parser: bibliographic fields, 20+ section types, numbered and author-year citations, equation/table/figure counts, and a confidence score gating whether paper features apply |
| `chunker.py` | Semantic chunking (200–800 tokens, target 500), splits on headers, keeps code blocks and tables intact, tracks `start_char`/`end_char`, and enriches each chunk with its section type |
| `pipeline.py` | Orchestrates the above and writes to SQLite |

### `embeddings/`
`embedding_generator.py` — `sentence-transformers` wrapper. Batched encoding, normalized
vectors for cosine similarity. Local and free; changing the model requires re-ingestion
because vector dimensionality is baked into the Qdrant collection.

### `vectorstore/`
`qdrant_store.py` — supports `memory` / `disk` / `server` modes behind one interface.
Beyond the standard payload fields it creates eight academic indexes so metadata filtering
happens inside Qdrant rather than in Python:

`paper_title`, `authors`, `year`, `section_type`, `venue`, `paper_type`,
`has_equations`, `has_citations`

### `retrieval/`
| File | Responsibility |
|---|---|
| `query_processor.py` | `detect_section_intent()` routes questions to sections; `extract_academic_filters()` turns natural language into structured filters; `expand_with_academic_terms()` adds domain synonyms. Gemini-backed rewriting degrades gracefully when the API is unavailable |
| `hybrid_search.py` | Vector + BM25, merged with Reciprocal Rank Fusion. BM25 needs an in-memory chunk corpus, indexed via `index_chunks()` |
| `reranker.py` | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranking of a wide candidate set |
| `retrieval_pipeline.py` | Orchestrator with three modes — `fast` (vector only), `balanced` (+ query processing + hybrid), `accurate` (+ reranking). Falls back to an unfiltered search when a section filter yields nothing |

### `generation/`
| File | Responsibility |
|---|---|
| `answer_generator.py` | Builds the grounded prompt, calls Ollama, and streams or buffers the answer. Modes: `answer`, `compare`, `conflicts`. Also validates citations after generation (below) |
| `citation_formatter.py` | Inline citations, APA reference lists, BibTeX export |
| `conversation_manager.py` | Multi-turn memory — history is folded into follow-up queries so "what about its accuracy?" resolves against the prior turn |
| `generation_pipeline.py` | Retrieval -> generation -> citation assembly, returning `PipelineResponse`; `generate_stream()` yields the same result incrementally |

### Citation validation

A fabricated citation is worse than none in a system whose purpose is grounded attribution,
and a 3B model invents plausible ones even when the prompt forbids it — observed in practice
citing "(Saadi et al., 2023)" for a document with no authors at all. Prompting cannot
guarantee this, so every citation is checked after generation and deleted if unsupported.

Matching is on **author surname plus year**, not on the exact string: the same source is
legitimately rendered "Saadi et al.", "Saadi and Abghour", or "A. Saadi", and an exact-string
rule deleted valid citations. Both renderings are handled — parenthetical
"(Saadi et al., 2025)" and narrative "According to Saadi et al. (2025)" — and removing a
narrative citation takes its lead-in phrase with it so no dangling "According to," is left.
Ordinary parentheses are untouched.

### Streaming

`POST /api/ask/stream` and `POST /api/chat/{id}/stream` return Server-Sent Events:
`status` while retrieval runs, `token` per answer fragment, then one `done` carrying
sources, citations, and metrics. Retrieval finishes before the first token because its
result is needed to build the prompt, so status events cover that gap.

Once the response has begun, an HTTP error status is no longer possible, so failures are
reported as an `error` event inside the stream. Citation validation runs on the complete
text, so the `done` payload can differ slightly from the concatenated tokens; the UI
re-renders from `done`.

A streamed chat turn persists the same messages as a buffered one — the user message before
generation, the assistant message after — so history is identical either way.

### `evaluation/`
| File | Responsibility |
|---|---|
| `retrieval_metrics.py` | MRR, NDCG, Precision@K, Recall@K, MAP |
| `answer_metrics.py` | ROUGE, citation accuracy, completeness |
| `qasper_loader.py` | Loads QASPER (QA over scientific papers) as evaluation samples |
| `evaluator.py` | End-to-end evaluation orchestration |
| `benchmark_runner.py` | Runs a benchmark config and writes a JSON report |

### `storage/`
SQLite via `chunk_store.py` (documents, chunks, images) and `conversation_store.py`
(conversations, messages, query history). Chosen for zero-setup local persistence.

### `auth/`
| File | Responsibility |
|---|---|
| `security.py` | bcrypt password hashing and JWT issue/verify. Rejects passwords over bcrypt's 72-byte limit rather than silently truncating them |
| `store.py` | SQLite user accounts, sharing the application database so document ownership is a plain foreign-key relationship |

### Multi-tenancy

Every document, chunk, and conversation belongs to a user, and the tenant key is
enforced at all three storage layers:

| Layer | Enforcement |
|---|---|
| SQLite documents | `user_id` column, indexed; `get_all_documents(user_id=...)` |
| SQLite conversations | `user_id` column, indexed; ownership checked before read or write |
| Qdrant | `user_id` in every point payload, indexed, and added to the filter of every search |
| BM25 | Per-user in-memory index built only from that user's chunks |

BM25 is the reason `RetrievalPipeline` and `GenerationPipeline` are cached per user rather
than shared: the index lives in memory and holds raw chunk text, so a single shared corpus
would let one tenant's keyword query score against another's documents. The vector store is
shared because Qdrant can filter server-side.

Cross-tenant access returns **404, not 403**, so the API cannot be used to probe for another
user's document or conversation ids.

### `api/` and `web/`
FastAPI with a lifespan hook that constructs one `DocuSenseRAG` and injects it into routes
via `Depends`. Components inside it are lazily initialized, so heavy models (embeddings,
reranker) load on first use rather than at boot. The web UI is dependency-free
HTML/CSS/JS served as static files from the same origin, so no CORS or build step.

---

## Design decisions

**Qdrant over FAISS.** FAISS is an index, not a database — no persistence and no metadata
filtering. Academic filtering ("NeurIPS papers from 2023") is the core feature, and it needs
filters applied *inside* the search, not as a post-filter that breaks top-k.

**Hybrid over pure vector.** Embeddings blur exact identifiers. `BERT-base` and `RoBERTa`
are semantically adjacent but factually distinct — BM25 preserves that distinction, and RRF
merges the two rankings without needing calibrated scores.

**RRF over weighted score fusion.** Vector similarity and BM25 scores live on incomparable
scales; tuning weights across corpora is fragile. RRF only uses rank position.

**Lazy component initialization.** The reranker and embedding models cost seconds and
hundreds of MB. Loading on first use keeps boot fast and lets the API serve metadata
endpoints without loading any ML model.

**Local models throughout.** sentence-transformers for embeddings and Ollama for generation
means no per-query cost, no rate limits, and documents never leave the machine.

---

## Known limitations

Tracked honestly rather than hidden — see the README for current status.

- **No token revocation.** JWTs are stateless, so a token stays valid until it expires;
  there is no logout-everywhere or blocklist. Logout clears the client's copy only.
- **No password reset or email verification.** Accounts are email plus password, and
  nothing confirms the address is real.
- **Health check is shallow.** `/api/health` reports whether components have been lazily
  constructed, not whether Qdrant and Ollama are actually reachable. `scripts/doctor.py`
  does the real check in the meantime.
- **Errors degrade silently.** Retrieval failures are logged as warnings and return an empty
  list, which is indistinguishable to the caller from a genuine no-match.
- **Streaming is answer-only.** Retrieval still completes before the first token, so there is
  a few-second wait before text begins; progress events cover it.
- **Payload indexes are server-only.** Qdrant's local disk mode ignores the eight academic
  indexes, so metadata filters fall back to full scans. Correct, but not fast at scale.
- **Benchmarks unpublished.** The evaluation framework exists but has not been run against
  QASPER to produce reportable numbers.
- **Metadata extraction is heuristic.** Title, author, and section detection are
  regex-based and tuned against a small sample; unusual layouts will still mis-parse.

## Fixed in the repair pass

Recorded because the failure modes are instructive:

- **Retrieval was never connected to the vector store.** `DocuSenseRAG` built
  `RetrievalPipeline()` with no arguments, so `vector_store` stayed `None`, `hybrid_search`
  was never constructed, and every query returned zero results regardless of backend health.
- **BM25 was never indexed.** The same call passed no chunk corpus, leaving the keyword half
  of hybrid search inert.
- **The similarity threshold silently emptied vector results.** A 0.7 cutoff sat above the
  range all-MiniLM-L6-v2 actually produces (~0.25–0.8), so the vector side returned nothing
  and hybrid search ran on BM25 alone.
- **BM25 hits lost their metadata.** Those results read `chunk['metadata']`, but the corpus
  dicts were flat, so citations degraded to "Unknown Document".
- **Sections were never detected in real papers.** Detection matched only Markdown `#`
  headers, which PDF conversion never produces.
- **Document deletion stranded vectors.** `delete_document()` removed SQLite rows only.

Every one of these is now covered by `tests/test_integration.py`.
