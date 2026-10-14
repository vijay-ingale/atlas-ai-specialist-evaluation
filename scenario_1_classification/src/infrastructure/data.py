from pathlib import Path
import pandas as pd
from src.core.errors import StorageError

class CsvDataset:
    def __init__(self, path: Path):
        self.path = path

    def load(self):
        try:
            return pd.read_csv(self.path, keep_default_na=False)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise StorageError("The training CSV could not be read.") from exc
