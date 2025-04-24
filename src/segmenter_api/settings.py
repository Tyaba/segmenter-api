from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class CommonSettings(BaseSettings):
    google_cloud_storage_bucket: str = "segmenter-api"
    florence2_base_model_path: Path = Path("models/microsoft/Florence-2-base")
    florence2_large_model_path: Path = Path("models/microsoft/Florence-2-large")
    sam2_model_path: Path = Path("models/facebook/sam2.1-hiera-large")


@lru_cache
def get_settings():
    settings = CommonSettings()
    return settings
