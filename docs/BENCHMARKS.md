# Benchmarks

Measured numbers for DocuSense retrieval, and how to reproduce them.

The ablation below comes from `python scripts/benchmark.py --papers 80`, which writes
[`data/benchmarks/qasper_ablation_report.json`](../data/benchmarks/qasper_ablation_report.json).
The section-routing numbers come from

```bash
USE_SECTION_ROUTING=true python scripts/benchmark.py --papers 80 --reuse-corpus \
    --only-routed --arms hybrid_rerank,full \
    --out data/benchmarks/routed_subset_report.json
```

which writes
[`data/benchmarks/routed_subset_report.json`](../data/benchmarks/routed_subset_report.json).
The query-rewriting numbers come from

```bash
LLM_PROVIDER=groq QUERY_LLM_BACKEND=provider python scripts/benchmark.py \
    --papers 80 --reuse-corpus --arms hybrid_rerank,full \
    --out data/benchmarks/query_rewriting_report.json
```

which writes
[`data/benchmarks/query_rewriting_report.json`](../data/benchmarks/query_rewriting_report.json).
All three files are committed as the evidence behind these tables.

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
| Vector only | 0.1867 | 0.1949 | 0.1004 | 0.2268 | 0.2938 | 0.1476 | 31 |
| Previous default (balanced, no rerank)* | 0.2041 | 0.2142 | 0.1081 | 0.2507 | 0.3219 | 0.1630 | 23 |
| Hybrid (vector + BM25, RRF) | 0.2244 | 0.2348 | 0.1313 | 0.2662 | 0.3438 | 0.1832 | 37 |
| **Hybrid + cross-encoder rerank** | **0.2824** | **0.2771** | **0.2046** | **0.3149** | **0.3816** | **0.2223** | 1908 |
| Hybrid + rerank + query processing *(current default)* | 0.2824 | 0.2771 | 0.2046 | 0.3149 | 0.3816 | 0.2223 | 1904 |

\* The "previous default" row is not part of the four-arm ladder and is not in the
committed report. It comes from a separate run with `--arms legacy_default`, kept here
because it is what the system scored before reranking was switched on. Reproduce it with
`python scripts/benchmark.py --papers 80 --arms legacy_default --reuse-corpus`.

The last two rows are identical to four decimal places because, in the shipped
configuration, query processing does nothing at all. Its three components are LLM
rewriting, which is off by default (`QUERY_LLM_BACKEND=off`; switched on it is
measurably *harmful*, [below](#what-the-numbers-say)); academic metadata filters, which
fire on none of the 259 questions, since QASPER questions do not mention years, authors
or venues; and section routing, which is now off by default. Every query in both arms
produced the same ranking — the paired difference is exactly 0.0000 with p = 1.0, not a
small effect that failed to reach significance. The arm is kept in the ladder because it
is the shipped default, and stating a cost of zero is worth as much as stating a gain.
(The two latency figures differ by less than the run-to-run noise, which is why the
reranking arm appears marginally slower here.)

An earlier run measured this arm at 0.2777 with section routing on. Why routing was
turned off is [below](#what-the-numbers-say).

Retrieval metrics for the first three arms reproduced identically across seven separate
runs, including across two full re-ingestions of the corpus with different section
tagging. Latencies vary by a
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
| hybrid + rerank → + query processing | 0.0000 | [0.0000, 0.0000] | 1.0000 | no effect |
| vector → hybrid + rerank | +0.0958 | [+0.0554, +0.1374] | 0.0001 | **significant** |
| vector → previous default | +0.0175 | [−0.0067, +0.0412] | 0.1639 | not significant |

Two rows deserve attention.

**Query processing has no effect here — which is not the same as "not significant".**
The difference is exactly zero on every one of the 259 queries, because none of its
three components does anything on this corpus in this environment (see the note under
the table). This is a measurement of the benchmark's coverage as much as of the feature:
it says query processing is not being exercised, not that query rewriting is worthless.
An earlier run, with section routing still on, put this comparison at −0.0048, 95% CI
[−0.0152, +0.0066], p = 0.43 — genuinely "not significant", and worth distinguishing
from the exact zero above.

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
scored **0.2041** while its own components, configured as documented, reach **0.2824** —
a 38% improvement that was purely configuration. That is what the "previous default" row
measures, and it was not significantly better than plain vector search. It is fixed: the
pipeline is now built in `mode="accurate"` and honours `USE_RERANKING`.

**Section routing does not earn its place, and better labels made that clearer.** This
was the feature the README led with — "how did they train it?" searches methodology. It
is now **off by default**. The reasoning took two passes, and the first one was wrong.

*First pass: blame the labels.* Routing fires on 60 of the 259 questions. Measured on
just those 60, with the section labels the system produced at the time:

| Arm (60 routed questions) | MRR | NDCG@10 | Recall@5 | MAP |
|---|---|---|---|---|
| Hybrid + rerank | 0.2426 | 0.2521 | 0.3000 | 0.2003 |
| + section routing | 0.2221 | 0.2178 | 0.2454 | 0.1718 |

Δ MRR = −0.0205, 95% CI [−0.0638, +0.0287], p = 0.42 over n = 60 — an interval spanning
zero, so no harm was demonstrated. The apparent explanation was the labelling: 62.6% of
chunks carried no usable `section_type`, so the filter hid the corpus rather than
focusing it. The conclusion drawn was "fix section detection first, then re-measure".

*Section detection was then fixed.* Chunks now carry their full chain of enclosing
headers and are classified from it — the outermost heading that can be classified wins,
so "Experiments > Baseline Models" is `experiments` and "Model > Background" is
`methodology` rather than `introduction`. The classification vocabulary was widened to
cover how papers actually name sections, and the document title is excluded from the
chain (it heads every path, so a paper titled "A Neural Model for Question Answering"
would otherwise have every chunk tagged `methodology`).

On the same 1,048-chunk corpus:

| section_type | before | after |
|---|---|---|
| `other` | 46.9% | 27.3% |
| *(missing)* | 8.7% | 0% |
| `unknown` | 7.0% | 0% |
| `introduction` | 9.2% | 16.4% |
| `methodology` | 5.1% | 12.8% |
| `experiments` | 6.9% | 12.4% |
| `dataset` | — | 7.1% |
| `related_work` | 4.3% | 7.0% |
| `conclusion` | 5.0% | 7.0% |
| `results` | 3.4% | 4.3% |
| `discussion` | 3.4% | 3.4% |
| `acknowledgements` | — | 1.8% |
| `appendix` | — | 0.4% |
| `abstract` | 0.1% | 0.1% |
| `references` | 0.1% | 0.1% |

Chunks with no usable label fall from **62.6% to 27.3%**.

`other` is now a real answer rather than a default. It covers headings that name
something specific to one paper ("Latent Dirichlet Allocation", "Fakeddit"), and the
front-matter chunk of every document, which sits above the first section and is
deliberately left unlabelled — classifying it from the title would tag it with whatever
keyword the title happens to contain.

*Second pass: routing got worse.* Re-measured on the same 60 questions with the repaired
labels:

| Arm (60 routed questions) | MRR | NDCG@10 | Recall@5 | MAP |
|---|---|---|---|---|
| Hybrid + rerank | 0.2426 | 0.2521 | 0.3000 | 0.2003 |
| + section routing, old labels | 0.2221 | 0.2178 | 0.2454 | 0.1718 |
| + section routing, repaired labels | 0.1903 | 0.1518 | 0.1676 | 0.1208 |

Δ MRR against no routing = −0.0523, 95% CI [−0.1310, +0.0233], p = 0.21. That interval
still spans zero at n = 60, so this is again not a significant MRR result — but the
point estimate doubled in the wrong direction, and Recall@5 fell by 44% relative.

The mechanism is straightforward once seen. A filter that matches almost nothing does
not restrict the search: it returns fewer candidates than requested, the pipeline widens
back to the unfiltered pool, and the query is effectively unrouted. Repairing the labels
made the filter match enough chunks to clear that threshold, so for the first time it
actually restricted the search — and restricting the search is what costs accuracy. The
old labels were not the problem; they were what was hiding it.

Broadening the filter instead of narrowing it does not rescue this. Routing each
question to a *group* of related sections (`results` → results, experiments, discussion)
was measured on the same 60 questions and scored **0.1561** MRR — worse still, for the
same reason: a broader filter is a filter that bites harder. That experiment is not in
the shipped code.

*What is actually wrong.* This one is not a statistical question, and it is measured
directly rather than inferred from a metric. For each of the 60 routed questions, take
the chunks QASPER marks as the evidence and look at the `section_type` those chunks
actually carry:

| | share of routed questions |
|---|---|
| evidence is in the section routed to | **13.3%** |
| evidence is in the routed section or a related one | 58.3% |

Questions do not respect section boundaries. "What accuracy did they get?" routes to
`results`, but on this corpus the numbers usually sit under `experiments`. "What is X?"
routes to `abstract`, which is one chunk per paper. Across the routed questions the
evidence is spread over `experiments` (21), `introduction` (14), `other` (13), `dataset`
(10), `results` (9), `methodology` (8) and four more — no single section holds it.

A pre-filter that excludes the answer 86.7% of the time cannot help no matter how good
the labels behind it are. The fix would have to be in the question→section mapping, not
in the tagging, and this measurement bounds how much that could ever be worth.

Routing is therefore **off by default** (`USE_SECTION_ROUTING=true` re-enables it). The
label repair is kept: it is correct on its own terms, it makes `section_type` usable for
display and for caller-supplied filters, and it is what made the real problem visible.

The widening fallback stays as well. A partially populated filter is a bad failure mode
precisely because it returns *some* results, so a zero-results check never fires; the
pipeline widens when an inferred filter leaves fewer candidates than requested. It is
the reason routing looked harmless in the first pass, which is worth keeping in mind
when reading a benign measurement of a filter.

**LLM query rewriting costs accuracy, and it had never run.** This is the
feature the ablation table above reports as an exact zero. That zero was
honest about what it measured and misleading about why: rewriting was wired to
Gemini alone, the key on this project returns 403, and so every run measured
the feature as **absent** rather than as ineffective. "No effect" and "never
executed" are not the same claim.

Rewriting now goes through the same seam that generation does
(`QUERY_LLM_BACKEND=provider` uses whatever `LLM_PROVIDER` points at), so it
can be run wherever answers can. Measured on the same 259 questions and the
same 1,048 chunks, with `qwen/qwen3.8-27b` doing the rewriting:

| Arm | MRR | NDCG@10 | P@1 | Recall@5 | Recall@10 | MAP | ms/query |
|---|---|---|---|---|---|---|---|
| Hybrid + rerank *(current default)* | **0.2824** | **0.2771** | **0.2046** | **0.3149** | **0.3816** | **0.2223** | 1932 |
| + LLM query rewriting | 0.2305 | 0.2407 | 0.1467 | 0.2759 | 0.3536 | 0.1879 | 2600 |

Δ MRR = **−0.0519**, 95% CI [−0.0859, −0.0174], p = 0.0033, n = 259. The
interval excludes zero, so unlike section routing this is a demonstrated
*harm*, not an unproven one. P@1 falls by 28% relative and each query costs
about 670ms more.

The run rewrote **259 of 259** queries — none fell back — so this measures the
feature working rather than failing. The report carries that count under
`query_llm`, because a rewrite that errored and fell back to the original
query scores identically to one that simply did not help, and a number that
cannot tell those apart is the trap this whole exercise came out of.

Per query, against no rewriting:

| | count |
|---|---|
| worse | 57 |
| better | 27 |
| unchanged | 175 |
| had the evidence at rank 1, lost it | 24 of 53 |
| gained rank 1 | 9 |
| found evidence in the top 10, then found none | 21 |
| found none, then found some | 9 |

*Why.* The rewriter cannot see the corpus, so when a question is
under-specified it supplies the missing context from its own priors — and on a
question about one particular paper, those priors are usually wrong. Over a
30-question sample it rewrote every one, keeping 88 content terms, dropping
20, and **adding 146**: the rewritten query is mostly words the asker never
wrote.

```
What are the five domains?
  -> What are the five domains of the CompTIA A+ certification exam?

How long is the vocabulary of subwords?
  -> What is the typical size of the subword vocabulary used in modern
     natural language processing models?

Which of two design architectures have better performance?
  -> Which of the two proposed software design architectures demonstrates
     superior performance in terms of speed, efficiency, and resource
     utilization?
```

The first invents a certification exam that appears nowhere in the paper. The
second turns a question about *this* paper's vocabulary into a question about
the field. The third decides the architectures are software. Each added term
is something BM25 will now match against a corpus that does not contain it,
and something the embedding will pull toward a topic the paper is not about.

This is the same shape as the section-routing result: a component that sounds
obviously helpful, is measured, and turns out to cost accuracy — for a reason
that is specific and visible once looked at. `QUERY_LLM_BACKEND` therefore
defaults to `off`, and the ablation table's exact zero remains the honest
description of the shipped default.

*What this does not say.* QASPER questions are already well-formed: they were
written by NLP practitioners as questions about a paper. Rewriting is meant to
help the opposite case — a vague, conversational query — and this benchmark
contains none. The measurement bounds what rewriting is worth on well-posed
questions, which is what an academic search box mostly receives; it is not
evidence that rewriting is worthless everywhere.

*Two other LLM calls were removed from the query path entirely.* Query
expansion and intent classification each cost a round trip per query, and
neither could ever change a result: `retrieval_pipeline` searches on
`rewritten_query` alone, and reads `expanded_queries` and `intent` only to
count them in a metrics field. They are now off by default
(`ENABLE_QUERY_EXPANSION`, `ENABLE_INTENT_CLASSIFICATION`) and stay available
for a caller who wants the metadata.

---

## The model runtime does not change the numbers

The embedding model and the cross-encoder run on ONNX Runtime rather than
torch. That was a deployment decision — 326MB resident against 758MB, which is
the difference between needing a host that will rent a gigabyte and fitting in
a free tier — and it is only defensible if the numbers above survive it.

They do, exactly. The whole ablation re-run under `MODEL_RUNTIME=onnx`, after a
full re-ingestion of the same 80 papers:

| Arm | MRR | NDCG@10 | P@1 | Recall@5 | Recall@10 | MAP |
|---|---|---|---|---|---|---|
| Vector only | 0.1867 | 0.1949 | 0.1004 | 0.2268 | 0.2938 | 0.1476 |
| Hybrid | 0.2244 | 0.2348 | 0.1313 | 0.2662 | 0.3438 | 0.1832 |
| Hybrid + rerank | 0.2824 | 0.2771 | 0.2046 | 0.3149 | 0.3816 | 0.2223 |
| + query processing | 0.2824 | 0.2771 | 0.2046 | 0.3149 | 0.3816 | 0.2223 |

Every figure matches the torch run to four decimal places, and the corpus it
was measured on is identical down to the chunk count (80 papers, 1,048 chunks,
259 questions). The report is
[`data/benchmarks/onnx_ablation_report.json`](../data/benchmarks/onnx_ablation_report.json).

This is not a coincidence to be grateful for. They are the *same weights* —
`sentence-transformers/all-MiniLM-L6-v2` and the ONNX export of
`cross-encoder/ms-marco-MiniLM-L-6-v2` — so identical output is the correct
result, and anything else would have meant a bug. `tests/test_model_runtime.py`
asserts it directly: embeddings agree to a cosine above 0.9999 on short and
long text, and the cross-encoder agrees to 1e-3 and ranks identically.

**Latency is the one thing that moved**, in both directions:

| | torch | ONNX |
|---|---|---|
| Vector only | 31 ms | **29 ms** |
| Hybrid | 37 ms | **32 ms** |
| Hybrid + rerank | **1908 ms** | 3350 ms |

Retrieval without the cross-encoder is slightly faster; reranking is about 1.7×
slower on this corpus. sentence-transformers sorts a batch by length before
running it, so short candidates are not padded up to the longest one;
fastembed does not. Sorting the candidates before handing them over recovers
that where it applies — measured on 40 candidates of mixed length, 3162ms
became 1787ms, against 2220ms for torch — but QASPER chunks are near-uniform
in length, so on this corpus there is nothing for the sort to recover. It is
kept because real uploads are not uniform, and because the reordering provably
cannot change a score.

### Two ways this nearly went wrong quietly

Both are in the repository as tests now, because both produced no error.

**The tokenizers truncate at different lengths.** fastembed defaults to 128
tokens; `all-MiniLM-L6-v2` under sentence-transformers uses 256; this project
chunks to ~500. On the defaults the two runtimes agreed to a cosine of
1.000000 on a one-sentence test and diverged on every real chunk — 0.976 at
~180 tokens, **0.921 at ~600**. A 0.92 cosine is a different embedding, and
nothing anywhere would have said so; the retrieval numbers would simply have
drifted.

**Fixing that broke ingestion, and the benchmark reported anyway.** fastembed
pads to a fixed width equal to its truncation length, so raising truncation
alone made any batch holding both a long chunk and a short one ragged.
Ingestion failures are logged as warnings by design, so 78 of 80 papers failed
to ingest and the run went on to print a complete four-arm ablation over the 9
questions that survived — internally consistent, plausible, and about a
two-paper corpus. `scripts/benchmark.py` now refuses to report when fewer
papers ingested than were asked for.


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
or venue metadata, so the reconstructed documents have none either. Scoring citations
against an empty author list measures the benchmark corpus, not the system, and the
resulting `Citation-F1 = 0.0` would read as a product failure it is not. The report keeps
those figures under `not_applicable` with the reason attached.

This used to be worse than "no authors". The extractor scanned the first 2000 characters
with a bare Title-Case regex, which matches a title as readily as a name, so every
reconstructed paper came back with authors like `["New Multimodal Benchmark Dataset",
"Fake News Detection"]` — values that look real, are wrong, and flow into every citation
the document produces. Extraction now returns an empty list when there is no author line,
which is the truthful answer and makes the reason this metric is inapplicable explicit
rather than disguised. It does not make the metric measurable: an author-less corpus
cannot score author matching either way.

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
LLM_PROVIDER=groq QUERY_LLM_BACKEND=provider \
  python scripts/benchmark.py --papers 80 --reuse-corpus \
    --arms hybrid_rerank,full                             # LLM query rewriting
python scripts/benchmark.py --papers 5 --arms vector,hybrid   # quick smoke test
```

Notes:

- Papers are ingested under the `qasper_benchmark` user id. Per-user isolation keeps them
  out of your own documents, BM25 index, and search results; `--reuse-corpus` reuses
  them, and the default re-ingests from scratch.
- Local Qdrant disk mode allows one process at a time. Stop any running `uvicorn` first.
- In the four-arm ladder, LLM rewriting is off (`QUERY_LLM_BACKEND=off`), so the
  "query processing" arm measures the pattern-based half only — section routing and
  academic filters. Rewriting is measured separately, with `QUERY_LLM_BACKEND=provider`,
  [above](#what-the-numbers-say). Gemini was never reachable from this project at all:
  its key returns 403, which is what hid the feature behind an exact zero for three
  benchmark runs.
- The corpus is rebuilt from QASPER's parsed text, not from PDFs, so PDF extraction
  quality is not in scope here. Real uploads go through Markitdown and will chunk
  differently.
