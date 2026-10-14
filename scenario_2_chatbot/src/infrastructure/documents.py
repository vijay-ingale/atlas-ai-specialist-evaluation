from dataclasses import dataclass
from pathlib import Path
from src.core.errors import InvalidInputError, StorageError

@dataclass(frozen=True)
class Document:
    source: str
    content: str

class DocumentLoader:
    def __init__(self, directory: Path, max_bytes: int):
        self.directory = directory.resolve()
        self.max_bytes = max_bytes

    def load(self) -> list[Document]:
        documents = []
        try:
            if not self.directory.is_dir():
                raise InvalidInputError("The configured document directory does not exist.")
            for path in sorted(self.directory.rglob("*")):
                if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                    continue
                if not path.resolve().is_relative_to(self.directory):
                    raise InvalidInputError("Document path escapes the configured directory.")
                if path.stat().st_size > self.max_bytes:
                    raise InvalidInputError("A document exceeds the configured size limit.")
                content = path.read_text(encoding="utf-8")
                if content.strip():
                    documents.append(Document(path.relative_to(self.directory).as_posix(), content))
        except (OSError, UnicodeError) as exc:
            raise StorageError("Documents could not be read as UTF-8 text.") from exc
        if not documents:
            raise InvalidInputError("No non-empty Markdown or text documents found.")
        return documents
