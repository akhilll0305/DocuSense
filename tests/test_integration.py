"""
Integration tests: real components wired together, no mocks.

The rest of the suite mocks every boundary, which is why it stayed green while
the retrieval pipeline was constructed without a vector store and returned zero
results for every query. These tests exercise the actual wiring:

    ingest -> embed -> Qdrant -> retrieve -> metadata

They use an in-memory Qdrant collection and a temporary SQLite database, so
they need no running services. Answer generation is covered separately because
it requires Ollama.

Run with:   pytest -m integration
Skip with:  pytest -m "not integration"
"""

from __future__ import annotations

import pytest

from docusense.config.settings import settings


PAPER_TEXT = """\
# Attention Routing for Traffic Networks

## Abstract
We introduce ARTNet, a routing model for coordinating signalized intersections.

## Methodology
ARTNet trains a multi-agent deep Q-network with a shared replay buffer.
Each agent observes queue length and phase duration at its own intersection.
We optimize with Adam at a learning rate of 0.0003 for 500 episodes.

## Results
ARTNet reduced average vehicle delay by 23.4% against a fixed-time baseline.
Throughput improved by 11.8% on the Hangzhou benchmark.

## Conclusion
Attention-based routing coordinates intersections without central control.
"""


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    """A real DocuSenseRAG backed by temp SQLite and in-memory Qdrant."""
    tmp = tmp_path_factory.mktemp("docusense_integration")

    original = (
        settings.sqlite_db_path,
        settings.qdrant_mode,
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.qdrant_collection_name,
    )
    settings.sqlite_db_path = tmp / "test.db"
    settings.qdrant_mode = "memory"
    settings.qdrant_url = None          # keep effective_qdrant_mode off "server"
    settings.qdrant_api_key = None
    settings.qdrant_collection_name = "integration_chunks"

    from docusense.rag_pipeline import DocuSenseRAG

    instance = DocuSenseRAG(enable_images=False)
    yield instance

    instance.close()
    (
        settings.sqlite_db_path,
        settings.qdrant_mode,
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.qdrant_collection_name,
    ) = original


@pytest.fixture(scope="module")
def ingested(rag, tmp_path_factory):
    """Ingest one synthetic paper and return the result."""
    doc = tmp_path_factory.mktemp("docs") / "artnet.md"
    doc.write_text(PAPER_TEXT, encoding="utf-8")
    result = rag.ingest(doc)
    assert result.success, f"ingestion failed: {result.error}"
    return result


@pytest.mark.integration
def test_ingest_creates_chunks_and_vectors(rag, ingested):
    """Ingestion must persist chunks and push an equal number of vectors."""
    assert ingested.num_chunks > 0
    assert ingested.num_embeddings == ingested.num_chunks

    points = rag.qdrant_store.get_collection_info().get("points_count")
    assert points == ingested.num_chunks


@pytest.mark.integration
def test_retrieval_pipeline_is_wired_to_backends(rag, ingested):
    """
    Guards the bug this suite previously missed.

    RetrievalPipeline was built with no arguments, leaving vector_store None
    and hybrid_search None, so every query returned zero results.
    """
    pipeline = rag.retrieval_pipeline

    assert pipeline.vector_store is not None, "retrieval pipeline has no vector store"
    assert pipeline.hybrid_search is not None, "hybrid search was never constructed"
    assert pipeline.hybrid_search.bm25_index is not None, "BM25 corpus was never indexed"


@pytest.mark.integration
def test_retrieval_returns_results(rag, ingested):
    """A query matching ingested content must return non-empty results."""
    results, metrics = rag.retrieval_pipeline.retrieve(
        "How was the model trained?", top_k=3
    )

    assert results, "retrieval returned no results for ingested content"
    assert metrics.num_final_results == len(results)


@pytest.mark.integration
def test_both_retrieval_signals_contribute(rag, ingested):
    """
    Vector and BM25 must each produce hits.

    A similarity threshold above the model's typical score range silently
    zeroed the vector side, leaving hybrid search running on BM25 alone.
    """
    results, _ = rag.retrieval_pipeline.retrieve(
        "multi-agent deep Q-network replay buffer", top_k=5
    )
    assert results

    assert any(r.vector_score > 0 for r in results), "vector search contributed nothing"
    assert any(r.bm25_score > 0 for r in results), "BM25 contributed nothing"


@pytest.mark.integration
def test_results_carry_paper_metadata(rag, ingested):
    """
    Retrieved chunks must keep their paper metadata.

    BM25-sourced hits read chunk['metadata']; when the corpus was flat, that
    lookup returned {} and citations degraded to "Unknown Document".
    """
    results, _ = rag.retrieval_pipeline.retrieve("reduced average vehicle delay", top_k=3)
    assert results

    assert any(
        (r.metadata or {}).get("paper_title") for r in results
    ), "no retrieved chunk carried a paper title"


@pytest.mark.integration
def test_section_routing_targets_the_right_section(rag, ingested):
    """A methodology question should surface methodology chunks."""
    results, _ = rag.retrieval_pipeline.retrieve(
        "What optimizer and learning rate did they use?", top_k=5
    )
    assert results

    sections = {(r.metadata or {}).get("section_type") for r in results}
    assert "methodology" in sections, f"expected a methodology chunk, got {sections}"


@pytest.mark.integration
def test_delete_document_removes_vectors(rag, tmp_path):
    """Deleting a document must not strand its vectors in Qdrant."""
    doc = tmp_path / "throwaway.md"
    doc.write_text("## Notes\nA temporary document about widget calibration.\n", encoding="utf-8")

    result = rag.ingest(doc)
    assert result.success

    before = rag.qdrant_store.get_collection_info().get("points_count")
    assert rag.delete_document(result.document_id) is True
    after = rag.qdrant_store.get_collection_info().get("points_count")

    assert after == before - result.num_chunks
    assert result.document_id not in {d["document_id"] for d in rag.list_documents()}
