from typing import Any, Literal
from pydantic import BaseModel, Field

class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)

class PredictionResponse(BaseModel):
    predicted_category: Literal["billing", "general_inquiry", "technical_issue"]
    model_probability: float
    probabilities: dict[str, float]
    top_two_margin: float
    needs_human_review: bool
    review_reasons: list[str]

class TrainingResponse(BaseModel):
    model_version: str
    evaluation: dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_ready: bool

class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str
