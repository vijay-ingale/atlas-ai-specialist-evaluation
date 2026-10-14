from typing import Annotated
from fastapi import APIRouter, Depends
from src.api.dependencies import Container, get_container
from src.schemas.contracts import ErrorResponse, HealthResponse, MessageRequest, PredictionResponse, TrainingResponse

router = APIRouter(responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse},
                            422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
Services = Annotated[Container, Depends(get_container)]

@router.get("/health", response_model=HealthResponse, tags=["Operations"])
def health(services: Services):
    return HealthResponse(status="ok", model_ready=services.artifacts.ready())

@router.post("/train", response_model=TrainingResponse, tags=["Training"])
def train(services: Services):
    return services.training.train()

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: MessageRequest, services: Services):
    return services.prediction.predict(request.message)
