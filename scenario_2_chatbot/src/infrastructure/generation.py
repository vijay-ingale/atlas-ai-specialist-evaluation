from openai import OpenAI, OpenAIError
from pydantic import ValidationError
from src.core.errors import ConfigurationError, ProviderError
from src.schemas.contracts import GroundedAnswer
from src.services.prompts import build_messages

class OpenAIGenerator:
    def __init__(self, settings):
        self.settings = settings

    def generate(self, question, passages):
        key = self.settings.openai_api_key
        if not key or key == "replace_with_your_api_key":
            raise ConfigurationError("Configure OPENAI_API_KEY before answer generation.")
        try:
            with OpenAI(api_key=key, timeout=self.settings.provider_timeout_seconds, max_retries=1) as client:
                response = client.responses.parse(
                    model=self.settings.openai_model,
                    input=build_messages(question, passages),
                    text_format=GroundedAnswer,
                    max_output_tokens=1600,
                    store=False,
                )
                return response.output_parsed
        except (OpenAIError, ValidationError) as exc:
            raise ProviderError("Generation provider failed or returned an unusable response.") from exc
