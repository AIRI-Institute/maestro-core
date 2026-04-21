import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    version: str = "dev"


def load_config(env_file=None):
    env_file = env_file or os.getenv("ENV_FILE")
    return Config(_env_file=env_file)
