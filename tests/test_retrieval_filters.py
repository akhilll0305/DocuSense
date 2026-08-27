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


class TestInferredFilterFallback:
    """An inferred filter that finds nothing must not end the search."""

    def test_section_filter_falls_back(self):
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

    def test_fallback_keeps_the_tenant_scope(self):
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

    def test_thin_filtered_result_is_widened_not_accepted(self):
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
