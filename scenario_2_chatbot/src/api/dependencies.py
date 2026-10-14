from dataclasses import dataclass
from fastapi import Request
from src.infrastructure.documents import DocumentLoader
from src.infrastructure.generation import OpenAIGenerator
from src.infrastructure.models import LocalModels
from src.infrastructure.vector_store import VectorStore
from src.services.chat import ChatService
from src.services.guardrails import Guardrails
from src.services.ingestion import IngestionService
from src.services.retrieval import RetrievalService

@dataclass
class Container:
    store: VectorStore
    ingestion: IngestionService
    retrieval: RetrievalService
    chat: ChatService
    guardrails: Guardrails

def build_container(settings):
    models = LocalModels(settings)
    store = VectorStore(settings.index_dir)
    guardrails = Guardrails(settings.openai_api_key)
    loader = DocumentLoader(settings.document_dir, settings.max_document_bytes)
    retrieval = RetrievalService(settings, models, store, guardrails)
    ingestion = IngestionService(settings, loader, models, store, guardrails)
    chat = ChatService(retrieval, OpenAIGenerator(settings), guardrails)
    return Container(store, ingestion, retrieval, chat, guardrails)

def get_container(request: Request):
    return request.app.state.container
