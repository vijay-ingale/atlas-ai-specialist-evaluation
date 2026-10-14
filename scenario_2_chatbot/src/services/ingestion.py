from threading import Lock
from src.core.errors import InvalidInputError
from src.schemas.contracts import IngestionResponse
from src.services.chunking import SectionChunker

class IngestionService:
    def __init__(self, settings, loader, models, store, guardrails):
        self.settings, self.loader, self.models = settings, loader, models
        self.store, self.guardrails = store, guardrails
        self.lock = Lock()

    def ingest(self):
        with self.lock:
            documents = self.loader.load()
            self.guardrails.validate_documents(documents)
            tokenizer, limit = self.models.tokenizer_and_limit()
            chunker = SectionChunker(tokenizer, limit, self.settings.chunk_tokens, self.settings.overlap_tokens)
            chunks = [chunk for document in documents for chunk in chunker.split(document)]
            if not chunks:
                raise InvalidInputError("Documents contain no indexable content.")
            vectors = self.models.encode([chunk.text for chunk in chunks])
            metadata = {"embedding_model": self.settings.embedding_model,
                        "chunk_tokens": chunker.chunk_tokens, "overlap_tokens": chunker.overlap_tokens}
            version = self.store.save(chunks, vectors, metadata)
            return IngestionResponse(index_version=version, documents=len(documents), chunks=len(chunks), **metadata)
