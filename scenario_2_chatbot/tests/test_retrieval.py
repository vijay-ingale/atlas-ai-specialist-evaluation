from types import SimpleNamespace
import numpy as np
from src.core.config import Settings
from src.schemas.contracts import Chunk
from src.services.guardrails import Guardrails
from src.services.retrieval import RetrievalService

def service(scores, **overrides):
    chunks = [Chunk(id="one", source="a.md", section="A", text="First engineering passage."),
              Chunk(id="two", source="b.md", section="B", text="Second engineering passage.")]
    vectors = np.array([[1.0, 0.0], [0.8, 0.6]])
    store = SimpleNamespace(load=lambda _: (chunks, vectors))
    models = SimpleNamespace(encode=lambda _: np.array([[1.0, 0.0]]), score=lambda *_: np.array(scores))
    return RetrievalService(Settings(**overrides), models, store, Guardrails())

def test_reranker_can_reverse_cosine_order():
    passages = service([1.0, 3.0]).retrieve("engineering question")
    assert [passage.id for passage in passages] == ["two", "one"]

def test_weak_candidates_produce_no_context():
    assert service([-2.0, -1.0]).retrieve("unsupported question") == []

def test_context_top_k_is_respected():
    assert len(service([1.0, 3.0], context_top_k=1).retrieve("question")) == 1
