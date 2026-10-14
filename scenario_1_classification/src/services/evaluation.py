import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.core.config import LABELS

def evaluate(pipeline, x_train, x_test, y_test, settings):
    predictions = pipeline.predict(x_test)
    probabilities = np.sort(pipeline.predict_proba(x_test), axis=1)
    recognized = np.asarray(pipeline.named_steps["tfidf"].transform(x_test).getnnz(axis=1)).reshape(-1) > 0
    accepted = (probabilities[:, -1] >= settings.review_threshold) & (
        probabilities[:, -1] - probabilities[:, -2] >= settings.margin_threshold) & recognized
    return {
        "dataset": "Synthetic demonstration data; not production evidence",
        "train_size": len(x_train), "test_size": len(x_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(y_test, predictions, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABELS).tolist(),
        "review_rate": float(1 - accepted.mean()),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": float(accuracy_score(y_test[accepted], predictions[accepted])) if accepted.any() else None,
        "thresholds": {"probability": settings.review_threshold, "margin": settings.margin_threshold},
        "errors": [{"message": text, "actual": actual, "predicted": predicted}
                   for text, actual, predicted in zip(x_test, y_test, predictions) if actual != predicted],
    }
