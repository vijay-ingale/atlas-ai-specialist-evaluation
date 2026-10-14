"""Versioned artifacts and atomic publication. Never load untrusted joblib files."""
import json
import os
from pathlib import Path
from uuid import uuid4
import joblib
from src.core.errors import ModelNotReadyError, StorageError

class ArtifactStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def ready(self):
        return (self.directory / "current.json").exists()

    def save(self, pipeline, evaluation):
        version = uuid4().hex
        snapshot = self.directory / version
        pointer = self.directory / f"pointer-{version}.tmp"
        try:
            snapshot.mkdir(parents=True)
            joblib.dump(pipeline, snapshot / "model.joblib")
            (snapshot / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
            pointer.write_text(json.dumps({"version": version}), encoding="utf-8")
            os.replace(pointer, self.directory / "current.json")
        except (OSError, ValueError) as exc:
            raise StorageError("The trained model could not be published.") from exc
        return version

    def load(self):
        if not self.ready():
            raise ModelNotReadyError("Train the model through POST /train first.")
        try:
            version = json.loads((self.directory / "current.json").read_text(encoding="utf-8"))["version"]
            if len(version) != 32 or any(c not in "0123456789abcdef" for c in version):
                raise ValueError("Invalid model version")
            return joblib.load(self.directory / version / "model.joblib")
        except Exception as exc:
            # Deserialization is an infrastructure boundary; suppress raw internals.
            raise StorageError("The saved model could not be read; retrain it.") from exc
