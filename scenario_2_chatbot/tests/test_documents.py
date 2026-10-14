import pytest
from src.core.errors import InvalidInputError, StorageError
from src.infrastructure.documents import DocumentLoader

def test_supported_documents_only(tmp_path):
    (tmp_path / "guide.md").write_text("# Guide\nApproved instructions.", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("  ", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("not a document", encoding="utf-8")
    documents = DocumentLoader(tmp_path, 1000).load()
    assert len(documents) == 1
    assert documents[0].source == "guide.md"

def test_oversized_document_is_rejected(tmp_path):
    (tmp_path / "large.txt").write_text("a" * 100, encoding="utf-8")
    with pytest.raises(InvalidInputError, match="size limit"):
        DocumentLoader(tmp_path, 10).load()

def test_invalid_encoding_becomes_storage_error(tmp_path):
    (tmp_path / "invalid.txt").write_bytes(b"\xff\xfe\xff")
    with pytest.raises(StorageError):
        DocumentLoader(tmp_path, 1000).load()
