"""Immutable snapshots; one atomic pointer publishes vectors and metadata together."""
import json
import os
from pathlib import Path
from uuid import uuid4
import numpy as np
from pydantic import ValidationError
from src.core.errors import IndexNotReadyError, StorageError
from src.schemas.contracts import Chunk

class VectorStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def ready(self):
        return (self.directory / "current.json").exists()

    def save(self, chunks, vectors, metadata):
        version = uuid4().hex
        snapshot = self.directory / version
        temporary_pointer = self.directory / f"pointer-{version}.tmp"
        try:
            snapshot.mkdir(parents=True)
            np.save(snapshot / "vectors.npy", vectors, allow_pickle=False)
            (snapshot / "metadata.json").write_text(json.dumps({**metadata,
                "chunks": [chunk.model_dump() for chunk in chunks]}), encoding="utf-8")
            temporary_pointer.write_text(json.dumps({"version": version}), encoding="utf-8")
            os.replace(temporary_pointer, self.directory / "current.json")
        except OSError as exc:
            raise StorageError("The document index could not be published.") from exc
        return version

    def load(self, expected_model):
        if not self.ready():
            raise IndexNotReadyError("Build the index through POST /ingest first.")
        try:
            pointer = json.loads((self.directory / "current.json").read_text(encoding="utf-8"))
            version = pointer["version"]
            if len(version) != 32 or any(c not in "0123456789abcdef" for c in version):
                raise ValueError("Invalid index version")
            snapshot = self.directory / version
            metadata = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
            if metadata["embedding_model"] != expected_model:
                raise IndexNotReadyError("Embedding model changed; rebuild the index.")
            chunks = [Chunk.model_validate(item) for item in metadata["chunks"]]
            vectors = np.load(snapshot / "vectors.npy", allow_pickle=False)
            if vectors.ndim != 2 or len(vectors) != len(chunks) or not chunks or not np.isfinite(vectors).all():
                raise ValueError("Invalid index dimensions or values")
            return chunks, vectors
        except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
            raise StorageError("Index is unreadable or inconsistent; rebuild it.") from exc
