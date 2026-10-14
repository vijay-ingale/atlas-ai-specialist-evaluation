from threading import Lock
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.core.config import LABELS
from src.core.errors import InvalidInputError
from src.schemas.contracts import TrainingResponse
from src.services.evaluation import evaluate
from src.services.preprocessing import normalize

class TrainingService:
    def __init__(self, settings, dataset, artifacts):
        self.settings, self.dataset, self.artifacts = settings, dataset, artifacts
        self.lock = Lock()

    def train(self):
        with self.lock:
            frame = self.dataset.load()
            if not {"message", "category"}.issubset(frame.columns):
                raise InvalidInputError("CSV requires message and category columns.")
            frame = frame.copy()
            frame["message"] = frame["message"].map(normalize)
            if not set(frame["category"]).issubset(LABELS):
                raise InvalidInputError("Dataset contains unsupported categories.")
            if (frame.groupby("message")["category"].nunique() > 1).any():
                raise InvalidInputError("Identical messages have conflicting labels.")
            frame = frame.drop_duplicates("message")
            if set(frame["category"]) != set(LABELS) or frame["category"].value_counts().min() < 5:
                raise InvalidInputError("Provide all categories with at least five unique examples each.")
            try:
                x_train, x_test, y_train, y_test = train_test_split(
                    frame["message"], frame["category"], test_size=self.settings.test_fraction,
                    random_state=self.settings.random_seed, stratify=frame["category"])
                pipeline = Pipeline([
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.settings.random_seed)),
                ])
                pipeline.fit(x_train, y_train)
            except ValueError as exc:
                raise InvalidInputError("Dataset cannot be split or fitted; review text and per-class sample counts.") from exc
            report = evaluate(pipeline, x_train, x_test, y_test, self.settings)
            return TrainingResponse(model_version=self.artifacts.save(pipeline, report), evaluation=report)
