from unittest.mock import Mock
import httpx
import pytest
from openai import APIConnectionError
from src.core.config import Settings
from src.core.errors import ConfigurationError, ProviderError
from src.infrastructure.generation import OpenAIGenerator

def test_placeholder_key_does_not_construct_provider_client(monkeypatch, passage):
    factory = Mock()
    monkeypatch.setattr("src.infrastructure.generation.OpenAI", factory)
    with pytest.raises(ConfigurationError):
        OpenAIGenerator(Settings(openai_api_key="replace_with_your_api_key")).generate("Question", [passage])
    factory.assert_not_called()

def test_provider_failure_is_translated(monkeypatch, passage):
    factory = Mock(side_effect=APIConnectionError(request=httpx.Request("POST", "https://example.invalid")))
    monkeypatch.setattr("src.infrastructure.generation.OpenAI", factory)
    with pytest.raises(ProviderError, match="Generation provider failed"):
        OpenAIGenerator(Settings(openai_api_key="dummy-unit-test-key")).generate("Question", [passage])
