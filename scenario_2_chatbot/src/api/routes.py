from typing import Annotated
from fastapi import APIRouter, Depends
from src.api.dependencies import Container, get_container
from src.schemas.contracts import ChatResponse, ErrorResponse, HealthResponse, IngestionResponse, QuestionRequest, RetrievalResponse

router = APIRouter(responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse},
                            422: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
                            503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
Services = Annotated[Container, Depends(get_container)]

@router.get("/health", response_model=HealthResponse, tags=["Operations"])
def health(services: Services):
    return HealthResponse(status="ok", index_ready=services.store.ready())

@router.post("/ingest", response_model=IngestionResponse, tags=["Documents"])
def ingest(services: Services):
    return services.ingestion.ingest()

@router.post("/retrieve", response_model=RetrievalResponse, tags=["Retrieval"])
def retrieve(request: QuestionRequest, services: Services):
    question = services.guardrails.validate_question(request.question)
    return RetrievalResponse(sources=services.retrieval.retrieve(question))

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: QuestionRequest, services: Services):
    return services.chat.answer(request.question)
