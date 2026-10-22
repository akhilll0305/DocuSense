"""
Query and passage instructions for asymmetric embedding models.

bge and e5 expect an instruction on the query side; leaving it off does not
raise anything, it just retrieves worse. These tests pin which side of the
encoder gets which prefix, using a fake backend that records its inputs.
"""

import numpy as np
import pytest

from docusense.embeddings import embedding_generator as eg
from docusense.embeddings.backends import embedding_prefixes


class RecordingBackend:
    dimension = 4

    def __init__(self):
        self.seen = []

    def encode(self, texts, batch_size=32, normalize=True, show_progress=False):
        self.seen.extend(texts)
        return np.ones((len(texts), self.dimension))


@pytest.fixture
def make_generator(monkeypatch):
    def build(model_name):
        backend = RecordingBackend()
        monkeypatch.setattr(eg, "load_embedding_backend", lambda *a, **k: backend)
        return eg.EmbeddingGenerator(model_name=model_name), backend
    return build


def test_symmetric_model_has_no_prefixes():
    assert embedding_prefixes("all-MiniLM-L6-v2") == ("", "")


def test_bge_prefixes_the_query_only(make_generator):
    generator, backend = make_generator("BAAI/bge-small-en-v1.5")

    generator.embed_query("what dataset is used?")
    generator.embed_batch(["a chunk of the paper"])

    assert backend.seen[0].startswith("Represent this sentence")
    assert backend.seen[0].endswith("what dataset is used?")
    assert backend.seen[1] == "a chunk of the paper"


def test_e5_prefixes_both_sides(make_generator):
    generator, backend = make_generator("intfloat/e5-small-v2")

    generator.embed_query("q")
    generator.embed_batch(["p"])

    assert backend.seen == ["query: q", "passage: p"]


def test_minilm_query_is_unchanged(make_generator):
    generator, backend = make_generator("all-MiniLM-L6-v2")

    generator.embed_query("what dataset is used?")

    assert backend.seen == ["what dataset is used?"]


def test_a_registry_name_is_not_a_local_model(tmp_path):
    from docusense.embeddings.backends import _local_onnx_model

    assert _local_onnx_model("all-MiniLM-L6-v2") is None
    # A directory without onnx/model.onnx is a torch checkpoint at most.
    assert _local_onnx_model(str(tmp_path)) is None
