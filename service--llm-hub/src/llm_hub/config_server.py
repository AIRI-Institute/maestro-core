import os

from mmar_mimpl import SettingsModel
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict


class LoggerConfig(BaseModel):
    name: str = "llm-hub-openai"
    level: str = "DEBUG"


class ConfigServer(SettingsModel):
    max_workers: int = 10
    host: str = "0.0.0.0"
    port: int = 40631
    logger: LoggerConfig
