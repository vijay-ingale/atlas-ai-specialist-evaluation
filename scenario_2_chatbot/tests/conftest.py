import pytest
from src.schemas.contracts import Passage

@pytest.fixture
def passage():
    return Passage(id="source-1", source="deployment.md", section="Rollback",
        text="Roll back when the error rate exceeds 2% for five consecutive minutes.",
        cosine_score=0.8, rerank_score=2.0)
