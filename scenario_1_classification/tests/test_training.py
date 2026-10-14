from types import SimpleNamespace
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from src.core.config import Settings, LABELS
from src.core.errors import InvalidInputError
from src.services.training import TrainingService

class CapturingArtifacts:
    def save(self, pipeline, evaluation):
        self.pipeline, self.evaluation = pipeline, evaluation
        return "test-version"

def dataset():
    return pd.DataFrame([{"message": f"{label} request uniqueword{index}", "category": label}
                         for index, label in enumerate(LABELS * 8)])

def test_vectorizer_does_not_learn_test_only_vocabulary():
    frame = dataset()
    settings = Settings()
    artifacts = CapturingArtifacts()
    service = TrainingService(settings, SimpleNamespace(load=lambda: frame), artifacts)
    result = service.train()
    _, held_out, _, _ = train_test_split(frame.message, frame.category,
        test_size=settings.test_fraction, random_state=settings.random_seed, stratify=frame.category)
    vocabulary = artifacts.pipeline.named_steps["tfidf"].vocabulary_
    for message in held_out:
        assert message.split()[-1] not in vocabulary
    assert result.evaluation["train_size"] + result.evaluation["test_size"] == len(frame)
    assert 0 <= result.evaluation["coverage"] <= 1

def test_conflicting_duplicate_labels_are_rejected():
    frame = dataset()
    frame.loc[len(frame)] = {"message": frame.loc[0, "message"], "category": "technical_issue"}
    service = TrainingService(Settings(), SimpleNamespace(load=lambda: frame), CapturingArtifacts())
    with pytest.raises(InvalidInputError, match="conflicting"):
        service.train()

def test_missing_category_is_rejected():
    frame = dataset().query("category != 'billing'")
    service = TrainingService(Settings(), SimpleNamespace(load=lambda: frame), CapturingArtifacts())
    with pytest.raises(InvalidInputError, match="all categories"):
        service.train()
