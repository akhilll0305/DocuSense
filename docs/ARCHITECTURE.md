# DocuSense Architecture

Module-level reference for the system. For setup and overview see the [README](../README.md).

---

## Request flow

There are two entry paths, both landing on `DocuSenseRAG` (`docusense/rag_pipeline.py`),
which is the single orchestrator for the whole system.

### Ingestion: `POST /api/ingest`

```
UploadFile
  -> routes.ingest_document()        writes to a temp file; the uploaded name
                                     travels separately as original_filename
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
| `chunker.py` | Semantic chunking (200–800 tokens, target 500), splits on headers, keeps code blocks and tables intact, tracks `start_char`/`end_char` and the full chain of enclosing headers, and enriches each chunk with its section type |
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
| `retrieval_pipeline.py` | Orchestrator with three modes — `fast` (vector only), `balanced` (+ query processing + hybrid), `accurate` (+ reranking). Widens the search when a filter inferred from the query leaves too few candidates |

#### Inferred filters are advisory

Filters that come from the *query* (section routing, year/author/venue) are
tracked separately from filters the *caller* passed in. If an inferred filter
leaves fewer candidates than the search asked for, the pipeline re-runs without
it and appends the wider pool behind the filtered hits, so routing keeps its
precedence where it works without excluding the corpus where it does not.

Caller filters are never dropped: `user_id` is one of them, and widening past it
would search another tenant's documents. A caller who explicitly asks for
`year=2021` gets an empty result rather than a silently broadened one.

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
| `qasper_loader.py` | Parses QASPER and rebuilds each paper as ingestible Markdown |
| `qasper_harness.py` | Ingests QASPER, grounds evidence to chunk ids, runs the ablation arms, paired bootstrap |
| `evaluator.py` | End-to-end evaluation orchestration |
| `benchmark_runner.py` | Runs a benchmark config and writes a JSON report |

Results and full methodology: [BENCHMARKS.md](BENCHMARKS.md). Reproduce with
`python scripts/benchmark.py --papers 80`.

The division of labour matters here. `retrieval_metrics.py` scores a ranking
against a ground truth; it cannot produce a ground truth. QASPER marks evidence
as paragraph *text*, and relevance in this system is a set of chunk ids that
only exist after ingestion, so something has to ingest the papers and match
evidence paragraphs to the chunks they landed in. That is `qasper_harness.py`,
and its absence is why the framework produced empty reports for so long: the
runner handed the evaluator samples with no retrieved ids and no relevant ids,
and the evaluator's own filters discarded every one of them.

Benchmark papers are ingested under a dedicated `user_id`, so the existing
per-user isolation keeps a benchmark corpus out of any real user's documents,
BM25 index, and search results.

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

#### Visual design — "The Reading Room"

The interface is styled as an academic journal rather than a dark SaaS product: warm paper
stock, hairline rules, Crimson Pro for display and Atkinson Hyperlegible (designed for low
vision) for UI, scholarly navy for structure, and a single citation-gold accent used the way
a reader's pen would be. Tokens live in `web/css/design-system.css`; light is the primary
palette, with a warm dark mode for night reading and a per-browser toggle.

Motion is hand-written (IntersectionObserver plus CSS transitions, no animation library) and
is meant to carry meaning: the landing hero annotates a paper in reading order — highlight
the method sentence, note the matched section in the margin, rule under the citation, then
show the answer that falls out of it.

Two rules the implementation enforces:

- **Content never depends on JS to be visible.** Scroll-reveal is scoped to a `.js-reveal`
  class set by an inline head script, and a watchdog reveals everything after 4s. A
  throttled or backgrounded tab can delay IntersectionObserver indefinitely; without these
  guards the page renders blank, which was observed during development.
- **Contrast is measured, not eyeballed.** Text/background pairs are computed against
  WCAG AA. The first pass found `--ink-faint` failing at 2.99:1 on sunk paper and the auth
  panel's gold at 3.61:1 on navy, and both were darkened.

  That pass checked colours in the light theme and assumed they carried over. Re-measured
  at a 430px viewport in **both** themes, four pairs were still failing, all of them the
  same mistake: a foreground hardcoded as a literal against a background that inverts
  between themes.

  | | ratio | fixed |
  |---|---|---|
  | Auth panel: gold eyebrow and headline accent | **1.02:1** | Panel pinned to `#1e3a5f`; `var(--navy)` inverts to a light blue while the panel's cream and gold foregrounds are literals |
  | Auth panel: cream wordmark | 1.81:1 | Same fix |
  | `.btn-accent` label on gold, dark theme | 2.39:1 (1.91:1 hovered) | Label reads `var(--on-gold)`, which is near-white in light and near-black in dark |
  | `--gold` on `--paper-sunk`, light theme | 4.31:1 | `--gold` darkened `#b45309` → `#a94e08` |
  | `--ink-faint` on `--paper-raised`, dark theme | 4.45:1 | `--ink-faint` lightened `#8a7f70` → `#8f8475` |

  The 1.02:1 pair was not low-contrast text; it was invisible text, and it had shipped.
  Every text/background pair in both themes now clears 4.5:1, worst case 4.70:1.

  The lesson is narrower than "check contrast": a colour token that inverts between themes
  can only be paired with another token that inverts with it. `.btn-primary` already
  carried a comment saying exactly this; `.auth__aside` and `.btn-accent` did not.

- **The narrow layout is confirmed, not assumed.** All three pages were checked at a real
  430px viewport: no element overflows horizontally on any of them, the landing page's
  mobile nav opens with all five links, and the chat sidebar becomes a drawer that opens
  over an overlay and closes again. One layout bug was found and fixed — the composer's
  placeholder wraps to two lines at that width and a one-row textarea clipped the second
  line behind its own scrollbar, so the placeholder shortens below 560px and the textarea
  is sized to its contents at load rather than only on `input`.

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
- **Metadata extraction is heuristic.** Title, author, section and year detection are
  regex-based; unusual layouts will still mis-parse. It now fails toward "unknown"
  rather than toward a confident wrong answer: a document with no author line gets no
  authors, and an undetermined year is `None` and is left off the payload rather than
  written as `0`. Measured on the QASPER corpus, 27.3% of chunks end up tagged `other`,
  down from 62.6% with no usable label.
- **Section routing costs accuracy, and is off by default.** On the 60 of 259 benchmark
  questions it fires on, it lowers MRR from 0.2426 to 0.1903 and Recall@5 by 44%
  relative. Repairing the section labels made it *worse*, not better, because a filter
  that matches more chunks actually restricts the search instead of collapsing into the
  widening fallback. The measured cause is the question→section mapping: the evidence
  answering a question is in the section it routes to only 13.3% of the time. Enable
  with `USE_SECTION_ROUTING=true`. See [BENCHMARKS.md](BENCHMARKS.md).
- **Section labels come from headers.** A converted document with no headings at all
  falls back to matching chunk offsets against detected sections, and those offsets are
  measured on two different strings (sections on the raw markdown, chunks on the
  preprocessed text). They agree only while preprocessing removes little, which is the
  case today but is not guaranteed by anything.
- **Reranking costs about 1.8s per query.** It is the largest single accuracy gain in
  the ablation (+25.8% MRR over hybrid alone) but takes retrieval from ~20ms to ~1.9s
  on CPU. Set `USE_RERANKING=false` to trade the accuracy back for latency.
- **Answer quality is bounded by a 3B local model.** Retrieval is measured over 259
  questions; generation quality is measured over 30 and is limited by `llama3.2:3b`,
  which answers tersely.

## Fixed in the benchmark pass

The evaluation framework had never been run. Running it end to end surfaced six
defects, four of them in the product rather than in the harness:

- **The benchmark never ran the pipeline.** `BenchmarkRunner` loaded questions and
  passed them straight to `RAGEvaluator` with `retrieved_ids` and `generated_answer`
  empty. The evaluator skips such samples, so every report came back empty in 0.00s —
  indistinguishable from a genuine score of zero. The runner now records a warning
  explaining exactly which metrics could not be computed and why.
- **The QASPER loader could not read QASPER.** `full_text` is a list of
  `{section_name, paragraphs}`, not a dict, so zero sections parsed; and `evidence`
  lives inside the `answer` object, not beside it, so zero evidence parsed. The unit
  tests encoded both mistakes in their fixtures and passed throughout.
- **Ground truth was synthetic.** `to_evaluation_samples()` set
  `relevant_ids=["ev_0", "ev_1", ...]`, placeholders that can never equal a retrieved
  chunk id. Every retrieval metric would have scored exactly 0.0 even with a working
  pipeline.
- **Year filters crashed the search.** `extract_academic_filters()` emits Mongo-style
  ranges (`{"$gte": 2020}`), but `QdrantVectorStore.search` built only `MatchValue`
  conditions, which accept bool/int/str. Any query mentioning a year range raised a
  pydantic `ValidationError` that the pipeline caught and turned into an empty result,
  so the documented year filtering had never worked. `build_filter()` now translates
  ranges to `Range` and lists to `MatchAny`.
- **"new" meant "published in the last two years".** The recency heuristic matched
  `\b(recent|latest|new)\b` anywhere, so "what is the new metric?" — a paper describing
  its own contribution — was restricted to 2024 onwards and returned nothing. It hit 17
  of 1310 QASPER questions. The pattern now requires the word to modify the literature
  ("recent papers", "the latest work"), and an explicit year in the query wins.
- **`USE_RERANKING` did nothing.** `DocuSenseRAG` built its `RetrievalPipeline` with the
  default `mode="balanced"`, which forces reranking off regardless of the setting. The
  shipped system scored 0.2041 MRR while the same components configured as documented
  reached 0.2824 — a 38% gap that was purely configuration. The pipeline is now built in
  `mode="accurate"` with `enable_reranking=settings.use_reranking`.

One regression came out of that last fix and was caught by an existing test:
`RankedResult` did not carry `vector_score`/`bm25_score`, so turning reranking on
zeroed the per-stage scores in every result. Both fields are now carried through.

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

## Fixed in the metadata pass

Three defects in paper metadata extraction, all of the same shape: a field with no
evidence behind it was filled with something that looked like evidence.

- **The title was returned as the author list.** `_extract_authors` ran a bare
  Title-Case regex over the first 2000 characters, which matches a title as readily as a
  name, so a document with no author line came back with authors like
  `["New Multimodal Benchmark Dataset"]`. Those values then flowed into every citation
  the document produced, and are why citation accuracy could not be scored on QASPER.
  The search is now bounded below the title and above the first front-matter section,
  candidates must have the shape of a personal name, and the block must show positive
  evidence that it is an author list — two or more names, an affiliation marker, or an
  email. No author line now means no authors.
- **An undetermined year was reported as 0.** `pm.year or 0` wrote a value that reads as
  real and renders in a citation as "(Smith, 0)". The year is now omitted from both the
  Qdrant payload and the chunk metadata when it is unknown, so it renders as "n.d.". The
  extraction itself was also wrong: it took the maximum 20xx anywhere in the first 2000
  characters, which picks up a year discussed in the abstract. Signals are now ordered by
  what each is worth, and the frequency fallback is confined to the front matter.
- **Fixing the authors nearly disabled all metadata.** Author presence was worth 0.15 of
  the research-paper confidence score. Once extraction stopped inventing authors, 82 of
  83 reconstructed QASPER papers fell below the 0.5 threshold — and below that threshold
  no chunk is enriched, so every chunk silently loses its `section_type`. The scoring now
  weights the structural signals (abstract, sections, references) that actually separate
  a paper from a memo. This was caught by measuring the confidence distribution before
  and after, not by a test; there was no test that would have failed.

### Section tagging

`section_type` was unusable on 62.6% of the benchmark corpus, now 27.3%. Four causes,
all fixed:

- **The classification vocabulary was too narrow.** Nine patterns covering the canonical
  IMRaD labels. Real papers name sections "Data set", "Setup", "Ablation Study", "Error
  Analysis", "Baselines" — all of which fell through to `other`. The patterns are now
  ordered by specificity, because order is load-bearing: testing
  `abstract: (abstract|summary)` first classified "Summary and Future Work" as an
  abstract.
- **Chunks were classified by character offset.** Section offsets are recorded on the raw
  converted markdown; chunk offsets are recorded on the preprocessed text. Chunks now
  carry their full header path and are classified from it — the outermost heading that
  can be classified wins, so a subsection inherits its parent's role. Offsets remain only
  as a fallback for header-less documents.
- **The first section of every document was skipped.** `get_section_type` truth-tested
  `start_char`, and the first section starts at character 0.

- **The document title was classified as a section.** It heads every header path, so a
  paper titled "A Neural Model for Question Answering" had every chunk tagged
  `methodology`. Excluding it from the chain fixed most of that, but only where something
  followed it -- the front-matter chunk, whose path is the title alone, was still
  labelled from the title. Both the header path and the offset lookup now ignore the
  title outright, and that chunk is `other`.

Neither half of the last one showed up as a failing test. The first appeared as a 34%
`methodology` share while measuring the label distribution; the second appeared as a
`methodology` label on the abstract of a hand-written paper while checking a live upload.
Both are now covered by `tests/test_paper_metadata.py`.
