"""Project-wide configuration loaded from environment variables.

Paths resolve relative to the project root so scripts behave the same whether
they're invoked from the repo root or from inside ``notebooks/``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration.

    Override any field via ``SARCASM_RADAR_<NAME>`` environment variables,
    e.g. ``SARCASM_RADAR_RANDOM_SEED=7``.
    """

    model_config = SettingsConfigDict(
        env_prefix="SARCASM_RADAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_curated: Path = PROJECT_ROOT / "data" / "curated"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    configs_dir: Path = PROJECT_ROOT / "configs"

    random_seed: int = 42
    test_size: float = Field(default=0.20, ge=0.05, le=0.40)
    val_size: float = Field(default=0.15, ge=0.05, le=0.40)

    text_column: str = "text"
    label_column: str = "label"


settings = Settings()
