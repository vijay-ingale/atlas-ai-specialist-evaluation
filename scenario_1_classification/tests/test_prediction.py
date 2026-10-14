from types import SimpleNamespace
import numpy as np
from src.core.config import Settings
from src.services.prediction import PredictionService

class FakePipeline:
    classes_ = ["billing", "general_inquiry", "technical_issue"]
    def __init__(self, probabilities, known_words):
        self.probabilities = probabilities
        self.named_steps = {"tfidf": SimpleNamespace(transform=lambda _: SimpleNamespace(nnz=known_words))}
    def predict_proba(self, _):
        return np.array([self.probabilities])

def predict(probabilities, known_words=1):
    model = FakePipeline(probabilities, known_words)
    return PredictionService(Settings(), SimpleNamespace(load=lambda: model)).predict("example message")

def test_high_probability_unknown_text_still_requires_review():
    result = predict([0.95, 0.03, 0.02], known_words=0)
    assert result.needs_human_review
    assert "No recognized vocabulary" in result.review_reasons

def test_small_margin_requires_review():
    result = predict([0.48, 0.45, 0.07])
    assert "Small gap between top categories" in result.review_reasons

def test_strong_prediction_can_be_accepted():
    result = predict([0.8, 0.1, 0.1])
    assert result.predicted_category == "billing"
    assert not result.needs_human_review
