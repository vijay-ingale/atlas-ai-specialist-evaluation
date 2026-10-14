from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.core.config import Settings
from src.core.errors import InvalidInputError
from src.infrastructure.documents import Document
from src.services.guardrails import Guardrails
from src.services.ingestion import IngestionService

def test_secret_document_rejected_before_embedding():
    models = Mock()
    loader = SimpleNamespace(load=lambda: [Document("private.md", "key: sk-abcdefghijklmnopqrstuv")])
    service = IngestionService(Settings(), loader, models, Mock(), Guardrails())
    with pytest.raises(InvalidInputError):
        service.ingest()
    models.encode.assert_not_called()
    models.tokenizer_and_limit.assert_not_called()
