from src.schemas.contracts import PredictionResponse
from src.services.preprocessing import normalize

class PredictionService:
    def __init__(self, settings, artifacts):
        self.settings, self.artifacts = settings, artifacts

    def predict(self, message):
        text = normalize(message)
        pipeline = self.artifacts.load()
        features = pipeline.named_steps["tfidf"].transform([text])
        probabilities = pipeline.predict_proba([text])[0]
        ranked = sorted(zip(pipeline.classes_, probabilities), key=lambda pair: pair[1], reverse=True)
        best, score = ranked[0]
        margin = float(score - ranked[1][1])
        reasons = []
        if features.nnz == 0:
            reasons.append("No recognized vocabulary")
        if score < self.settings.review_threshold:
            reasons.append("Low model probability")
        if margin < self.settings.margin_threshold:
            reasons.append("Small gap between top categories")
        return PredictionResponse(predicted_category=best, model_probability=float(score),
            probabilities={label: float(value) for label, value in ranked}, top_two_margin=margin,
            needs_human_review=bool(reasons), review_reasons=reasons)
