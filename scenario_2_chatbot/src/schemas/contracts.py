from typing import Literal
from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

class Chunk(BaseModel):
    id: str
    source: str
    section: str
    text: str

class Passage(Chunk):
    cosine_score: float
    rerank_score: float

class Evidence(BaseModel):
    source_id: str
    quote: str

class Claim(BaseModel):
    text: str
    evidence: list[Evidence]

class GroundedAnswer(BaseModel):
    status: Literal["answered", "insufficient_evidence", "out_of_scope"]
    claims: list[Claim]

class ChatResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence", "out_of_scope"]
    message: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    sources: list[Passage] = Field(default_factory=list)

class RetrievalResponse(BaseModel):
    sources: list[Passage]

class IngestionResponse(BaseModel):
    index_version: str
    documents: int
    chunks: int
    embedding_model: str
    chunk_tokens: int
    overlap_tokens: int

class HealthResponse(BaseModel):
    status: str
    index_ready: bool

class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str
