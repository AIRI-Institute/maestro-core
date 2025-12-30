import os

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = "ENV_FILE"


class SettingsModel(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    version: str = "dev"

    @classmethod
    def load(cls, env_file=None) -> "SettingsModel":
        env_file = env_file or os.getenv(ENV_FILE)
        return cls(_env_file=env_file)  # type: ignore[call-arg]
