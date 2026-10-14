import hashlib
import numpy as np
from src.core.errors import StorageError, ProviderError
from src.schemas.contracts import Passage

class RetrievalService:
    def __init__(self, settings, models, store, guardrails):
        self.settings, self.models, self.store = settings, models, store
        self.guardrails = guardrails

    def retrieve(self, question):
        chunks, vectors = self.store.load(self.settings.embedding_model)
        query = self.models.encode([question])[0]
        if vectors.shape[1] != query.shape[0]:
            raise StorageError("Index dimensions differ from query embeddings; rebuild the index.")
        cosine = vectors @ query
        ids = np.argsort(cosine)[::-1][:self.settings.retrieval_top_k]
        candidates = [chunks[int(i)] for i in ids]
        self.guardrails.validate_passages(candidates)
        relevance = self.models.score(question, candidates)
        if len(relevance) != len(candidates) or not np.isfinite(relevance).all():
            raise ProviderError("Reranker returned invalid relevance scores.")
        ranked = sorted(zip(ids, candidates, relevance), key=lambda entry: entry[2], reverse=True)
        selected, seen, used_characters = [], set(), 0
        for position, chunk, score in ranked:
            digest = hashlib.sha256(chunk.text.encode()).hexdigest()
            if score < self.settings.min_rerank_score or digest in seen:
                continue
            if used_characters + len(chunk.text) > self.settings.max_context_characters:
                continue
            selected.append(Passage(**chunk.model_dump(), cosine_score=float(cosine[position]), rerank_score=float(score)))
            seen.add(digest)
            used_characters += len(chunk.text)
            if len(selected) == self.settings.context_top_k:
                break
        return selected
