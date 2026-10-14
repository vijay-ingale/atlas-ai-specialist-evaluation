from dataclasses import dataclass
from fastapi import Request
from src.infrastructure.artifacts import ArtifactStore
from src.infrastructure.data import CsvDataset
from src.services.prediction import PredictionService
from src.services.training import TrainingService

@dataclass
class Container:
    artifacts: ArtifactStore
    training: TrainingService
    prediction: PredictionService

def build_container(settings):
    artifacts = ArtifactStore(settings.artifact_dir)
    return Container(artifacts, TrainingService(settings, CsvDataset(settings.data_path), artifacts),
                     PredictionService(settings, artifacts))

def get_container(request: Request):
    return request.app.state.container
