import os

from mmar_mimpl import SettingsModel
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class LoggerConfig(BaseModel):
    level: str = "DEBUG"
    name: str = "question-detector"


class ConfigServer(SettingsModel):
    max_workers: int = 10
    port: int = 11611
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
