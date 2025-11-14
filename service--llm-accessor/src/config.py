import os

from mmar_llm import EntrypointsConfig
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.legacy_load_pydantic import LoadPydantic


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    version: str = "dev"
    files_dir: str = "/mnt/data/maestro/files"

    entrypoints_path: str
    llm: LoadPydantic[EntrypointsConfig, "entrypoints_path"] = None

    warmup_entrypoints: bool = False


def load_config(env_file: str | None = None) -> Config:
    env_file = env_file or os.getenv("ENV_FILE")
    return Config(_env_file=env_file)
