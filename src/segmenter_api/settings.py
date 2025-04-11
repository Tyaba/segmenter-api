from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

class CommonSettings(BaseSettings):
    florence2_model_path: Path = Path("models/microsoft/Florence-2-large")
    sam2_model_path: Path = Path("models/facebook/sam2.1-hiera-large")


@lru_cache
def get_settings():
    settings = CommonSettings()
    return settings