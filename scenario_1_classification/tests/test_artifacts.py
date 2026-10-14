import pytest
from src.core.errors import ModelNotReadyError, StorageError
from src.infrastructure.artifacts import ArtifactStore

def test_missing_model_requires_training(tmp_path):
    with pytest.raises(ModelNotReadyError):
        ArtifactStore(tmp_path).load()

def test_republication_switches_complete_snapshot(tmp_path):
    store = ArtifactStore(tmp_path)
    first = store.save({"value": 1}, {"accuracy": 0.5})
    second = store.save({"value": 2}, {"accuracy": 0.6})
    assert first != second
    assert store.load() == {"value": 2}
    assert (tmp_path / first / "evaluation.json").exists()

def test_invalid_pointer_cannot_escape_artifact_directory(tmp_path):
    (tmp_path / "current.json").write_text('{"version":"../private"}', encoding="utf-8")
    with pytest.raises(StorageError):
        ArtifactStore(tmp_path).load()
