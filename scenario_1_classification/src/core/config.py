from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")
    data_path: Path = ROOT / "data/messages.csv"
    artifact_dir: Path = ROOT / "artifacts"
    review_threshold: float = Field(default=0.65, ge=0, le=1)
    margin_threshold: float = Field(default=0.15, ge=0, le=1)
    test_fraction: float = Field(default=0.25, gt=0, lt=1)
    random_seed: int = 42

LABELS = ["billing", "general_inquiry", "technical_issue"]
