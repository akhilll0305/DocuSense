"""
Tests for metadata filtering: how natural language becomes a Qdrant filter,
and what happens when an inferred filter matches nothing.

Every case here comes from a failure the QASPER benchmark surfaced. Filtering
is the feature the README leads with, and all three of these bugs made it
return nothing rather than fail loudly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docusense.retrieval.query_processor import QueryProcessor
from docusense.vectorstore.qdrant_store import QdrantVectorStore


# ==============================================================================
# Filter construction
# ==============================================================================

class TestBuildFilter:
    """QdrantVectorStore.build_filter turns plain dicts into Qdrant filters."""

    def test_none_and_empty(self):
        assert QdrantVectorStore.build_filter(None) is None
        assert QdrantVectorStore.build_filter({}) is None

    def test_scalar_becomes_match_value(self):
        f = QdrantVectorStore.build_filter({"user_id": "u1"})
        assert len(f.must) == 1
        assert f.must[0].key == "user_id"
        assert f.must[0].match.value == "u1"

    def test_range_dict_becomes_range(self):
        """
        A year range must become a Range condition.

        `extract_academic_filters` emits {"$gte": .., "$lte": ..} for "papers
        from 2020-2023". Feeding that to MatchValue, which accepts only
        bool/int/str, raised a pydantic ValidationError inside search; the
        pipeline swallowed it and returned no results, so year filtering could
        never have worked.
        """
        f = QdrantVectorStore.build_filter({"year": {"$gte": 2020, "$lte": 2023}})
        condition = f.must[0]
        assert condition.key == "year"
        assert condition.match is None
        assert condition.range.gte == 2020
        assert condition.range.lte == 2023

    def test_open_ended_range(self):
        f = QdrantVectorStore.build_filter({"year": {"$gte": 2024}})
        assert f.must[0].range.gte == 2024
        assert f.must[0].range.lte is None

    def test_strict_bounds(self):
        f = QdrantVectorStore.build_filter({"year": {"$gt": 2019, "$lt": 2024}})
        assert f.must[0].range.gt == 2019
        assert f.must[0].range.lt == 2024

    def test_list_becomes_match_any(self):
        f = QdrantVectorStore.build_filter({"authors": ["Bengio", "Hinton"]})
        assert f.must[0].match.any == ["Bengio", "Hinton"]

    def test_empty_list_is_dropped(self):
        assert QdrantVectorStore.build_filter({"authors": []}) is None

    def test_unsupported_operator_is_dropped_not_raised(self):
        """An unknown comparison is skipped with a warning, never a crash."""
        assert QdrantVectorStore.build_filter({"year": {"$regex": "20.."}}) is None

    def test_mixed_filters(self):
        f = QdrantVectorStore.build_filter({
            "user_id": "u1",
            "year": {"$gte": 2020},
            "authors": ["Bengio"],
        })
        assert len(f.must) == 3


# ==============================================================================
# Natural-language filter extraction
# ==============================================================================

class TestAcademicFilterExtraction:
    """The heuristics have to be precise, not just present."""

    @pytest.fixture(scope="class")
    def processor(self):
        return QueryProcessor()

    @pytest.mark.parametrize("query,expected_gte", [
        ("recent papers on transformers", True),
        ("the latest research in NLP", True),
        ("newest work on retrieval", True),
    ])
    def test_recency_still_detected(self, processor, query, expected_gte):
        """The documented behaviour keeps working."""
        filters = processor.extract_academic_filters(query)
        assert ("year" in filters) is expected_gte

    @pytest.mark.parametrize("query", [
        "What languages does the new dataset contain?",
        "What is the new metric?",
        "who annotated the new dataset?",
        "what are the recent models they compare with?",
        "How does the new architecture differ?",
    ])
    def test_recency_does_not_fire_on_paper_contributions(self, processor, query):
        """
        "new"/"recent" usually describe the paper's own contribution.

        Matching them anywhere restricted these questions to the last two
        years, which matches almost nothing and returns an empty result set.
        On QASPER this hit 17 of 1310 questions and zeroed every one.
        """
        assert "year" not in processor.extract_academic_filters(query)

    def test_explicit_year_wins_over_recency(self, processor):
        """"recent papers from 2015" means 2015."""
        filters = processor.extract_academic_filters("recent papers from 2015")
        assert filters["year"] == 2015

    def test_year_range(self, processor):
        filters = processor.extract_academic_filters("papers from 2020-2023")
        assert filters["year"] == {"$gte": 2020, "$lte": 2023}

    def test_author_and_venue(self, processor):
        filters = processor.extract_academic_filters("NeurIPS papers by Yoshua Bengio")
        assert filters["authors"] == "Yoshua Bengio"
        assert filters["venue"] == "NeurIPS"

    def test_plain_question_has_no_filters(self, processor):
        assert processor.extract_academic_filters("How did they train the model?") == {}

    def test_extracted_filters_are_buildable(self, processor):
        """
        Whatever the extractor emits, the vector store must accept.

        These two halves drifted apart once already: the extractor produced
        Mongo-style ranges the store could not express.
        """
        queries = [
            "recent papers on transformers",
            "papers from 2020-2023",
            "NeurIPS papers by Yoshua Bengio",
            "arxiv papers from 2021",
        ]
        for query in queries:
            filters = processor.extract_academic_filters(query)
            QdrantVectorStore.build_filter(filters)  # must not raise


# ==============================================================================
# Fallback when an inferred filter matches nothing
# ==============================================================================

class FakeVectorStore:
    """Records the filters it is searched with; returns hits only unfiltered."""

    def __init__(self, filtered_keys=("section_type", "year")):
        self.calls = []
        self.filtered_keys = filtered_keys

    def search(self, query, top_k=5, filters=None, **kwargs):
        self.calls.append(dict(filters or {}))

        # Any inferred filter present means "no match", so the fallback path
        # is what produces results.
        if filters and any(k in filters for k in self.filtered_keys):
            return []

        return [
            SimpleNamespace(
                chunk_id="c1",
                document_id="d1",
                text="a chunk",
                score=0.5,
                metadata={"user_id": filters.get("user_id") if filters else None},
            )
        ]


def pipeline_with(processor_metadata, store):
    """A vector-only pipeline whose query processor returns fixed metadata."""
    from docusense.retrieval.retrieval_pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline(
        vector_store=store,
        chunks=[],
        enable_query_processing=True,
        enable_hybrid_search=False,
        enable_reranking=False,
        mode="accurate",
    )
    pipeline.query_processor = SimpleNamespace(
        process=lambda query, context=None, num_expansions=2: SimpleNamespace(
            rewritten_query=query,
            metadata=processor_metadata,
            get_all_queries=lambda: [query],
        )
    )
    return pipeline


@pytest.fixture
def section_routing_on(monkeypatch):
    """
    Turn section routing on for one test.

    It ships off (it costs accuracy on QASPER), so tests of the routing filter
    have to ask for it rather than inherit whatever the default happens to be.
    Two of these silently stopped exercising the filter when the default
    changed, and still passed on the year filter alone.
    """
    from docusense.config.settings import settings

    monkeypatch.setattr(settings, "use_section_routing", True)
    return settings


class TestInferredFilterFallback:
    """An inferred filter that finds nothing must not end the search."""

    def test_section_filter_falls_back(self, section_routing_on):
        store = FakeVectorStore()
        pipeline = pipeline_with({"section_intent": "methodology"}, store)

        results, _ = pipeline.retrieve("how did they train it?", top_k=5)

        assert results, "expected the unfiltered retry to return results"
        assert "section_type" in store.calls[0]
        assert "section_type" not in store.calls[1]

    def test_academic_filter_falls_back(self):
        """
        A year filter that matches nothing is retried without it.

        Only section filters used to be retried, so an over-eager year filter
        ended the search with zero results and no explanation.
        """
        store = FakeVectorStore()
        pipeline = pipeline_with(
            {"academic_filters": {"year": {"$gte": 2024}}}, store
        )

        results, _ = pipeline.retrieve("what is the new metric?", top_k=5)

        assert results
        assert store.calls[0]["year"] == {"$gte": 2024}
        assert "year" not in store.calls[1]

    def test_fallback_keeps_the_tenant_scope(self, section_routing_on):
        """
        The caller's filters survive the retry.

        user_id is how one user's documents stay invisible to another. Dropping
        every filter on fallback would search the whole corpus.
        """
        store = FakeVectorStore()
        pipeline = pipeline_with(
            {"section_intent": "results", "academic_filters": {"year": {"$gte": 2024}}},
            store,
        )

        results, _ = pipeline.retrieve(
            "recent results", top_k=5, filters={"user_id": "user-a"}
        )

        assert results
        assert store.calls[1]["user_id"] == "user-a"
        assert "section_type" not in store.calls[1]
        assert "year" not in store.calls[1]

    def test_thin_filtered_result_is_widened_not_accepted(self, section_routing_on):
        """
        A filter that returns *some* results, but too few, is still widened.

        This is the bug section routing actually had. 63% of chunks carry no
        usable section_type, so routing to "results" searched 3.4% of the
        corpus and came back with a handful of hits. Because that is not zero,
        an empty-results check never fired, and the passage that answered the
        question — sitting in an untagged chunk — was unreachable.
        """
        class ThinStore:
            def __init__(self):
                self.calls = []

            def search(self, query, top_k=5, filters=None, **kwargs):
                self.calls.append(dict(filters or {}))
                if filters and "section_type" in filters:
                    return [SimpleNamespace(
                        chunk_id="section-hit",
                        document_id="d1",
                        text="a narrowly matched chunk",
                        score=0.4,
                        metadata={},
                    )]
                return [
                    SimpleNamespace(
                        chunk_id=f"c{i}",
                        document_id="d1",
                        text=f"chunk {i}",
                        score=0.9 - i / 100,
                        metadata={},
                    )
                    for i in range(20)
                ]

        store = ThinStore()
        pipeline = pipeline_with({"section_intent": "results"}, store)

        results, metrics = pipeline.retrieve("what accuracy did they get?", top_k=5)

        assert len(store.calls) == 2, "a thin filtered result should widen"
        assert "section_type" not in store.calls[1]
        assert "filter_widening" in metrics.stages_used

        ids = [r.chunk_id for r in results]
        # The routed hit keeps its precedence; the rest of the corpus fills in
        # behind it instead of being excluded.
        assert ids[0] == "section-hit"
        assert len(ids) > 1

    def test_caller_filter_is_never_treated_as_inferred(self):
        """
        A caller-supplied year is not dropped just because the query mentions one.

        If the caller asked for year=2021 explicitly, an empty result is the
        honest answer; silently widening it would answer a different question.
        """
        store = FakeVectorStore(filtered_keys=("year",))
        pipeline = pipeline_with({"academic_filters": {"year": 2021}}, store)

        results, _ = pipeline.retrieve("q", top_k=5, filters={"year": 2021})

        assert results == []
        assert len(store.calls) == 1, "no retry should have been attempted"


# ==============================================================================
# BM25 honours the same filters as the vector search
# ==============================================================================

# Chunks without the query terms. BM25Okapi's IDF is non-positive for a term
# present in most of a tiny corpus, so the fixtures need a background.
FILLER = [
    {"chunk_id": f"z{i}", "document_id": "Z", "user_id": "u",
     "text": f"unrelated background paragraph number {i} about gardening"}
    for i in range(8)
]


class TestBM25Filters:
    """
    A filter used to reach Qdrant only. With hybrid search on, a question
    scoped to one document got BM25 hits from every other document fused in.
    """

    @staticmethod
    def _search():
        from docusense.retrieval.hybrid_search import HybridSearch

        chunks = [
            {"chunk_id": "a1", "document_id": "A", "user_id": "u", "year": 2019,
             "authors": ["Smith"], "text": "transformer attention heads pruning"},
            {"chunk_id": "b1", "document_id": "B", "user_id": "u", "year": 2022,
             "authors": ["Jones"], "text": "transformer attention heads distillation"},
            {"chunk_id": "c1", "document_id": "C", "user_id": "u",
             "text": "transformer attention heads quantization"},
        ]
        return HybridSearch(vector_store=None, chunks=chunks + FILLER)

    def test_document_scope_is_respected(self):
        hits = self._search()._bm25_search("transformer attention", 10, {"document_id": "B"})
        assert [c["chunk_id"] for c, _ in hits] == ["b1"]

    def test_unfiltered_search_is_unchanged(self):
        hits = self._search()._bm25_search("transformer attention", 10)
        assert {c["chunk_id"] for c, _ in hits} == {"a1", "b1", "c1"}

    def test_year_range(self):
        hits = self._search()._bm25_search("attention", 10, {"year": {"$gte": 2020}})
        assert [c["chunk_id"] for c, _ in hits] == ["b1"]

    def test_missing_key_fails_like_qdrant(self):
        """c1 has no year; Qdrant drops a point without the key, so BM25 does too."""
        hits = self._search()._bm25_search("attention", 10, {"year": {"$lte": 2030}})
        assert "c1" not in {c["chunk_id"] for c, _ in hits}

    def test_list_payload_matches_membership(self):
        hits = self._search()._bm25_search("attention", 10, {"authors": "Smith"})
        assert [c["chunk_id"] for c, _ in hits] == ["a1"]

    def test_any_of(self):
        hits = self._search()._bm25_search("attention", 10, {"document_id": ["A", "C"]})
        assert {c["chunk_id"] for c, _ in hits} == {"a1", "c1"}

    def test_filter_does_not_shrink_below_top_k(self):
        """Filtering happens before the cut, not after it."""
        from docusense.retrieval.hybrid_search import HybridSearch

        chunks = [
            {"chunk_id": f"x{i}", "document_id": "X", "text": "attention " * 5}
            for i in range(5)
        ] + [{"chunk_id": "y", "document_id": "Y", "text": "attention"}]
        hits = HybridSearch(vector_store=None, chunks=chunks + FILLER)._bm25_search(
            "attention", 2, {"document_id": "Y"}
        )
        assert [c["chunk_id"] for c, _ in hits] == ["y"]


# ==============================================================================
# Deleting from a collection that does not exist
# ==============================================================================

def test_delete_from_missing_collection_is_a_no_op():
    """
    Raising here made delete_document() keep the document's SQLite rows, so a
    fresh QDRANT_COLLECTION_NAME left every old document in the BM25 corpus.
    """
    from unittest.mock import MagicMock

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.collection_name = "never_created"
    store.client = MagicMock()
    store.client.collection_exists.return_value = False

    assert store.delete_by_document("doc_1") == 0
    store.client.delete.assert_not_called()


def test_delete_by_filter_refuses_an_empty_filter():
    """An empty filter would match every point in the collection."""
    from unittest.mock import MagicMock

    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.collection_name = "c"
    store.client = MagicMock()

    with pytest.raises(ValueError):
        store.delete_by_filter({})
    store.client.delete.assert_not_called()


def test_rerank_candidates_default_and_override(monkeypatch):
    from docusense.config.settings import settings
    from docusense.retrieval.retrieval_pipeline import RetrievalPipeline

    monkeypatch.setattr(settings, "rerank_candidates", None)
    assert RetrievalPipeline._candidate_count(10) == 40

    monkeypatch.setattr(settings, "rerank_candidates", 20)
    assert RetrievalPipeline._candidate_count(10) == 20
    # Never fewer candidates than results asked for.
    assert RetrievalPipeline._candidate_count(30) == 30
