# Benchmarks

Measured numbers for DocuSense retrieval, and how to reproduce them.

Everything below comes from `python scripts/benchmark.py --papers 80`, which writes
[`data/benchmarks/qasper_ablation_report.json`](../data/benchmarks/qasper_ablation_report.json).
That file is committed as the evidence behind these tables.

---

## What is being measured

**Dataset.** [QASPER](https://allenai.org/data/qasper) v0.3 test split — 416 NLP papers
with 1,451 questions written by NLP practitioners who had read only the title and
abstract, and answered by others with the full text in front of them. Each answer cites
the paragraphs that support it.

**Corpus.** 80 papers, sampled deterministically (seed `20260827`) from the 408 of 416
that carry at least one answerable question with body-paragraph evidence. QASPER ships parsed
text rather than PDFs, so each paper is rebuilt as Markdown — title, abstract, section
headers, paragraphs — and ingested through the **real** pipeline: converters,
preprocessor, metadata extraction, chunker, embeddings, Qdrant. That produces **1,048
chunks**.

**Queries.** All 259 questions from those 80 papers that survive filtering (below).
Retrieval runs over the whole 1,048-chunk corpus, not one paper at a time: a question
about paper A competes against 79 other papers, which is what the product actually does.

**Ground truth.** QASPER marks evidence as paragraph *text*; relevance here is a set of
chunk ids that exist only after ingestion. Each evidence paragraph is matched to the
chunk(s) it landed in — verbatim containment after unicode and whitespace
normalization, falling back to longest-contiguous-token-run coverage of ≥50% of either
side when a paragraph straddles a chunk boundary. Chunks carry a one-sentence overlap,
so a paragraph legitimately maps to more than one chunk: **1.71 relevant chunks per
question** on average.

**Metrics.** Standard IR metrics at top-10: MRR, NDCG@k, Precision@k, Recall@k, MAP.
MAP is normalized by the total number of relevant chunks, so it is an AP@10.

### What was dropped, and why

| | Count |
|---|---|
| Questions in the 80 papers | 284 |
| — dropped, no annotator could answer them | 14 |
| — dropped, no evidence that could be grounded to a chunk | 11 |
| **Questions measured** | **259** |
| Evidence paragraphs | 745 |
| — grounded to at least one chunk | 738 (99.1%) |
| — ungrounded | 7 (0.9%) |

Evidence marked `FLOAT SELECTED: Table 1…` points at a figure or table, which the
reconstructed document does not contain; those entries are excluded before the counts
above. A small number of entries name a section rather than quote a paragraph and are
excluded the same way.

---

## Retrieval ablation

259 questions, 1,048 chunks, top-10, single run, CPU (no GPU). Each arm adds one
component to the one above it.

| Arm | MRR | NDCG@10 | P@1 | Recall@5 | Recall@10 | MAP | ms/query |
|---|---|---|---|---|---|---|---|
| Vector only | 0.1867 | 0.1949 | 0.1004 | 0.2268 | 0.2938 | 0.1476 | 17 |
| Previous default (balanced, no rerank) | 0.2041 | 0.2142 | 0.1081 | 0.2507 | 0.3219 | 0.1630 | 23 |
| Hybrid (vector + BM25, RRF) | 0.2244 | 0.2348 | 0.1313 | 0.2662 | 0.3438 | 0.1832 | 19 |
| **Hybrid + cross-encoder rerank** | **0.2824** | **0.2771** | **0.2046** | **0.3149** | **0.3816** | **0.2223** | 1782 |
| Hybrid + rerank + query processing (current default) | 0.2777 | 0.2691 | 0.2046 | 0.3022 | 0.3683 | 0.2157 | 1791 |

Retrieval metrics reproduced identically across four separate runs. Latencies vary by a
few ms between runs (and the reranking arms by a few hundred) because the machine was not
otherwise idle; treat the millisecond column as an order of magnitude, not a benchmark.

### Is the difference real?

Every arm answers the same 259 questions, so the arms can be compared pairwise.
Reported below: the mean per-query difference in reciprocal rank, a 95% confidence
interval from 10,000 paired bootstrap resamples over queries, and a two-sided sign-flip
permutation p-value.

| Comparison | Δ MRR | 95% CI | p | Verdict |
|---|---|---|---|---|
| vector → hybrid | +0.0377 | [+0.0189, +0.0584] | 0.0003 | **significant** |
| hybrid → hybrid + rerank | +0.0581 | [+0.0194, +0.0962] | 0.0044 | **significant** |
| hybrid + rerank → + query processing | −0.0048 | [−0.0152, +0.0066] | 0.4316 | not significant |
| vector → hybrid + rerank | +0.0958 | [+0.0554, +0.1374] | 0.0001 | **significant** |
| vector → previous default | +0.0175 | [−0.0067, +0.0412] | 0.1639 | not significant |

Two rows deserve attention.

**Query processing is not distinguishable from noise.** Its point estimate is negative,
but the interval straddles zero. This run cannot show that it hurts overall — only that
it does not help.

**The previous default was not significantly better than plain vector search.** Adding
BM25 and query processing while leaving reranking off produced a +9.3% point estimate
whose interval includes zero. The shipped system was, statistically, no better than the
simplest possible baseline. That is the finding that prompted the default change.

The committed report carries every comparison under `paired_comparisons`.

---

## What the numbers say

**Hybrid search helps, and now it is measured rather than asserted.** Adding BM25 and
fusing with RRF lifts MRR from 0.1867 to 0.2244 (+20.2%) and Recall@10 from 0.2938 to
0.3438 (+17.0%), for about 2ms per query. This is the claim the README used to make
without evidence.

*A caution about sample size, from this project.* An early pilot on 3 papers and 12
questions showed hybrid **14% worse** than vector-only. That result was noise. Twelve
questions cannot separate a 20% effect from chance, and had it been published it would
have been a confident, wrong claim in the opposite direction.

**Reranking is the largest single gain — and it was switched off.** The cross-encoder
takes MRR from 0.2244 to 0.2824 (+25.8% over hybrid, p=0.0044) and nearly doubles P@1,
from 0.1004 to 0.2046. It costs about 1.8s per query on CPU, which is small next to the
~27s the local model spends generating the answer.

It had never actually run. `DocuSenseRAG` built its pipeline in `mode="balanced"`, which
forces reranking off regardless of the `USE_RERANKING` setting, so the shipped system
scored **0.2041** while its own components, configured as documented, reach **0.2777** —
a 36% improvement that was purely configuration. That is what the "previous default" row
measures, and it was not significantly better than plain vector search. It is fixed: the
pipeline is now built in `mode="accurate"` and honours `USE_RERANKING`.

**Section routing does not earn its place.** This is the feature the README leads with —
"how did they train it?" searches methodology. It fires on 60 of the 259 questions.
Measured on just those 60:

| Arm (60 routed questions) | MRR | NDCG@10 | Recall@5 | MAP |
|---|---|---|---|---|
| Hybrid + rerank | 0.2426 | 0.2521 | 0.3000 | 0.2003 |
| + section routing | 0.2221 | 0.2178 | 0.2454 | 0.1718 |

Δ MRR = −0.0205, 95% CI [−0.0638, +0.0287], p = 0.42 over n = 60.

**That interval spans zero, so this run does not show that routing hurts.** Sixty
questions cannot resolve an effect this size. What it does bound is the upside: the
interval's upper end is +0.029 MRR, so any benefit is small at best, and the feature is
paying complexity and latency for it. The honest summary is "no measurable benefit", not
"measurably worse" — a distinction worth keeping, because the point estimate is tempting
to quote and would not survive a larger sample either way.

The *reason* it cannot help, however, is not a statistical question. Across the benchmark
corpus:

| section_type | share of chunks |
|---|---|
| `other` | 46.9% |
| `introduction` | 9.2% |
| *(missing)* | 8.7% |
| `unknown` | 7.0% |
| `experiments` | 6.9% |
| `methodology` | 5.1% |
| `conclusion` | 5.0% |
| `related_work` | 4.3% |
| `results` | 3.4% |
| `discussion` | 3.4% |
| `abstract` | 0.1% |

63% of chunks carry no usable section label (`other`, missing, or `unknown`). Routing a
question to `results` searches 3.4% of the corpus; routing to `abstract` searches a
single chunk. The passage that answers the question is usually sitting in an
`other`-tagged chunk, unreachable. That is a counted fact about the corpus, not an
inference from the metrics, and it is sufficient on its own to explain why routing
cannot be adding anything.

A partially populated filter is a particularly bad failure mode: because it returns
*some* results, a zero-results fallback never fires. The pipeline now widens the search
when an inferred filter leaves fewer candidates than requested — though on this corpus
that changed nothing measurable, because BM25 ignores filters entirely and had already
filled the candidate pool. Routing can also be disabled with `USE_SECTION_ROUTING=false`.

Routing is left **on** by default. It is existing behaviour, the measurement is on
reconstructed QASPER text rather than PDFs, and no significant harm was demonstrated.
The actionable finding is the labelling: fix section detection first, then re-measure.

---

## Answer quality

Generation is scored end to end through `rag.ask()` — the same path the UI uses — on a
30-question deterministic subsample, because each answer takes about 27s on CPU with
`llama3.2:3b`.

| Metric | Value |
|---|---|
| ROUGE-1 | 0.0594 |
| ROUGE-2 | 0.0181 |
| ROUGE-L | 0.0464 |
| Token overlap | 0.0644 |
| Completeness (query terms covered) | 0.6998 |
| Mean answer length | 1050 chars |
| BERTScore | not computed (`bert-score` not installed) |

**These ROUGE figures are close to meaningless as an accuracy score, and are published
only as a regression signal.** QASPER's reference answers are short extractive spans —
"3,044 sentences in 100 dialogs" — while DocuSense answers in prose. Unigram overlap
between a 12-word span and a 1050-character paragraph is low almost by construction, and
would stay low for a perfect answer. A rise or fall across future runs is informative;
the absolute value is not.

### What is *not* measured here

**Citation accuracy cannot be measured on QASPER, and is not reported.** The metric
matches cited surnames against the source paper's authors, and QASPER carries no author
or venue metadata. The reconstructed documents therefore have none either — the metadata
extractor recovers fragments of the title (`["New Multimodal Benchmark Dataset", "Fake
News Detection"]`) where authors should be. Scoring citations against that measures the
benchmark corpus, not the system, and the resulting `Citation-F1 = 0.0` would read as a
product failure it is not. The report keeps those figures under `not_applicable` with the
reason attached.

The same trap nearly caught ROUGE. `rouge-score` was listed in `requirements.txt` but was
not installed in the environment, and `to_dict()` published a flat `0.0` — a missing
library was indistinguishable from answers that overlapped the references not at all.
`bert-score` is not a declared dependency and is still absent. Both now serialize as
`null` with a `not_computed` note when their library is unavailable, so an uninstalled
package can no longer masquerade as a score of zero.

Citation behaviour *is* covered — by `tests/test_answer_generator.py` and the citation
validation described in [ARCHITECTURE.md](ARCHITECTURE.md#citation-validation), which
verifies every generated citation against the retrieved sources by author and year and
deletes the unsupported ones. That mechanism was visibly active during this run,
removing invented citations such as `(Devlin et al., 2018)` from answers grounded in
papers that have no such author.

---

## Reproducing

```bash
# 1. Get the dataset (~18MB extracted)
mkdir -p data/benchmarks/qasper
curl -L -o /tmp/qasper.tgz \
  https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
tar xzf /tmp/qasper.tgz -C data/benchmarks/qasper

# 2. Run the ablation (~20 min on CPU; reranking arms dominate)
python scripts/benchmark.py --papers 80

# Variations
python scripts/benchmark.py --papers 80 --answers 30      # + answer quality (needs Ollama)
python scripts/benchmark.py --papers 80 --reuse-corpus    # skip re-ingestion
python scripts/benchmark.py --papers 80 --only-routed \
    --arms hybrid_rerank,full                             # section routing, where it fires
python scripts/benchmark.py --papers 5 --arms vector,hybrid   # quick smoke test
```

Notes:

- Papers are ingested under the `qasper_benchmark` user id. Per-user isolation keeps them
  out of your own documents, BM25 index, and search results; `--reuse-corpus` reuses
  them, and the default re-ingests from scratch.
- Local Qdrant disk mode allows one process at a time. Stop any running `uvicorn` first.
- Query rewriting via Gemini was **unavailable** during these runs (the API key returns
  403/429). Section routing and academic filters are pattern-based and did run; LLM-based
  rewriting and expansion did not. The "query processing" arm therefore measures the
  pattern-based half only.
- The corpus is rebuilt from QASPER's parsed text, not from PDFs, so PDF extraction
  quality is not in scope here. Real uploads go through Markitdown and will chunk
  differently.
