import numpy as np
import pytest
from src.core.errors import IndexNotReadyError, StorageError
from src.infrastructure.vector_store import VectorStore
from src.schemas.contracts import Chunk

def test_snapshot_roundtrip_and_model_mismatch(tmp_path):
    store = VectorStore(tmp_path)
    chunk = Chunk(id="one", source="guide.md", section="Guide", text="Example passage")
    version = store.save([chunk], np.array([[1.0, 0.0]]), {"embedding_model": "model-a"})
    chunks, vectors = store.load("model-a")
    assert chunks == [chunk]
    np.testing.assert_array_equal(vectors, [[1.0, 0.0]])
    assert (tmp_path / version / "vectors.npy").exists()
    with pytest.raises(IndexNotReadyError):
        store.load("model-b")

def test_missing_index_has_explicit_error(tmp_path):
    with pytest.raises(IndexNotReadyError):
        VectorStore(tmp_path).load("model-a")

def test_corrupt_metadata_becomes_storage_error(tmp_path):
    store = VectorStore(tmp_path)
    (tmp_path / "current.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(StorageError):
        store.load("model-a")

def test_failed_publish_preserves_previous_snapshot(tmp_path, monkeypatch):
    store = VectorStore(tmp_path)
    first = Chunk(id="one", source="guide.md", section="Guide", text="Original passage")
    second = Chunk(id="two", source="guide.md", section="Guide", text="Replacement passage")
    store.save([first], np.array([[1.0, 0.0]]), {"embedding_model": "model-a"})
    def fail_replace(*args):
        raise OSError("Simulated disk failure")
    monkeypatch.setattr("src.infrastructure.vector_store.os.replace", fail_replace)
    with pytest.raises(StorageError):
        store.save([second], np.array([[0.0, 1.0]]), {"embedding_model": "model-a"})
    chunks, _ = store.load("model-a")
    assert chunks == [first]
