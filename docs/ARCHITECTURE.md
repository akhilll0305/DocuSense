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
| `answer_generator.py` | Builds the grounded prompt, calls Ollama, enforces citation formatting. Modes: `answer`, `compare`, `conflicts` |
| `citation_formatter.py` | Inline citations, APA reference lists, BibTeX export |
| `conversation_manager.py` | Multi-turn memory — history is folded into follow-up queries so "what about its accuracy?" resolves against the prior turn |
| `generation_pipeline.py` | Retrieval -> generation -> citation assembly, returning `PipelineResponse` |

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

- **Authentication is a UI mockup.** `web/auth.html` posts nothing; there is no user model,
  session handling, or password storage. All data is effectively single-tenant.
- **No multi-tenancy.** One SQLite database and one Qdrant collection are shared globally,
  so every user would see every document.
- **Document deletion leaks vectors.** `delete_document()` removes SQLite rows but leaves
  the corresponding Qdrant points orphaned.
- **Health check is shallow.** `/api/health` reports whether components have been lazily
  constructed, not whether Qdrant and Ollama are actually reachable.
- **Errors degrade silently.** Retrieval failures are logged as warnings and return an empty
  list, which is indistinguishable to the caller from a genuine no-match.
- **No streaming.** Responses are fully buffered before returning.
- **Benchmarks unpublished.** The evaluation framework exists but has not been run against
  QASPER to produce reportable numbers.
