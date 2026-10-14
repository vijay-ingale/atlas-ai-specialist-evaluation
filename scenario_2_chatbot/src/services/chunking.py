import hashlib
import re
from src.core.errors import InvalidInputError
from src.schemas.contracts import Chunk

class SectionChunker:
    def __init__(self, tokenizer, encoder_limit, chunk_tokens, overlap_tokens):
        self.tokenizer = tokenizer
        self.chunk_tokens = min(chunk_tokens, encoder_limit - 40)
        self.overlap_tokens = overlap_tokens
        if self.chunk_tokens <= overlap_tokens:
            raise InvalidInputError("Chunk overlap must be smaller than the encoder-safe chunk size.")

    def split(self, document):
        chunks = []
        section = "Overview"
        for part in re.split(r"(?m)^(#{1,6} .+)$", document.content):
            if re.match(r"^#{1,6} ", part):
                section = part.lstrip("# ").strip()
                continue
            tokens = self.tokenizer.encode(part.strip(), add_special_tokens=False)
            title = self.tokenizer.decode(self.tokenizer.encode(section, add_special_tokens=False)[:24])
            for start in range(0, len(tokens), self.chunk_tokens - self.overlap_tokens):
                body = self.tokenizer.decode(tokens[start:start + self.chunk_tokens], skip_special_tokens=True)
                if not body.strip():
                    continue
                identity = f"{document.source}|{section}|{start}|{body}"
                chunks.append(Chunk(id=hashlib.sha256(identity.encode()).hexdigest()[:16],
                    source=document.source, section=section, text=f"{title}\n{body}"))
                if start + self.chunk_tokens >= len(tokens):
                    break
        return chunks
