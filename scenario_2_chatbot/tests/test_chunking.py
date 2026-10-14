from src.infrastructure.documents import Document
from src.services.chunking import SectionChunker

class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()
    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)

def test_chunks_preserve_heading_and_overlap():
    chunker = SectionChunker(WordTokenizer(), 256, 4, 1)
    chunks = chunker.split(Document("guide.md", "# Deploy\na b c d e f\n## Rollback\nx y z"))
    assert len(chunks) == 3
    assert chunks[0].text == "Deploy\na b c d"
    assert chunks[1].text == "Deploy\nd e f"
    assert chunks[2].section == "Rollback"
    assert all(chunk.source == "guide.md" for chunk in chunks)

def test_content_change_changes_chunk_id():
    chunker = SectionChunker(WordTokenizer(), 256, 4, 1)
    first = chunker.split(Document("guide.md", "# Title\na b c"))
    second = chunker.split(Document("guide.md", "# Title\na b changed"))
    assert first[0].id != second[0].id
