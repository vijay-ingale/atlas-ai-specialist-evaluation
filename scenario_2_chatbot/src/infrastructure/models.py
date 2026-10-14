from functools import cached_property
from threading import Lock
import numpy as np
from src.core.errors import ProviderError

class LocalModels:
    """Lazy model loading: API startup does not download or initialize models."""
    def __init__(self, settings):
        self.settings = settings
        self.lock = Lock()

    @cached_property
    def encoder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.settings.embedding_model)

    @cached_property
    def cross_encoder(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(self.settings.reranker_model)

    def encode(self, texts):
        try:
            with self.lock:
                return np.asarray(self.encoder.encode(texts, normalize_embeddings=True), dtype=np.float32)
        except Exception as exc:
            # Adapter boundary: model/download failures must not leak library internals.
            raise ProviderError("Local embedding model could not process the input.") from exc

    def score(self, question, passages):
        try:
            with self.lock:
                scores = self.cross_encoder.predict([(question, passage.text) for passage in passages])
                return np.asarray(scores, dtype=float).reshape(-1)
        except Exception as exc:
            raise ProviderError("Local reranker could not process the input.") from exc

    def tokenizer_and_limit(self):
        try:
            with self.lock:
                return self.encoder.tokenizer, self.encoder.max_seq_length
        except Exception as exc:
            raise ProviderError("Embedding tokenizer could not be loaded.") from exc
