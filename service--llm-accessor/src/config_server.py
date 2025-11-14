import os

from mmar_ptag import LogLevelEnum
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class LoggerConfig(BaseModel):
    level: LogLevelEnum = LogLevelEnum.DEBUG
    name: str = "llm-accessor"


class ConfigServer(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")

    max_workers: int = 10
    port: int = 40631
    logger: LoggerConfig = Field(default_factory=LoggerConfig)


def load_config_server(env_file=None):
    env_file = env_file or os.getenv("ENV_FILE")
    return ConfigServer(_env_file=env_file)
