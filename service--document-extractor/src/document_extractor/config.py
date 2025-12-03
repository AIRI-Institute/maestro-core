import os
from typing import Literal

from mmar_ptag import LogLevelEnum
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["CPU", "CUDA"]


class LoggerConfig(BaseModel):
    name: str = "ocr"
    level: LogLevelEnum = LogLevelEnum.DEBUG


class ServerConfig(BaseModel):
    max_workers: int = 10
    port: int = 9671


class PdfConfig(BaseModel):
    num_threads: int = 8
    device: Device = "CPU"
    workers: int = 1
    empty_page_chars_threshold: int = 5


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    files_dir: str = Field(default="/mnt/data/maestro/files")
    cache_maxsize: int = Field(default=100, description="-1: disabled, 0: enabled infinity, 1+: etc")
    logger: LoggerConfig = LoggerConfig()
    server: ServerConfig = ServerConfig()
    pdf: PdfConfig = Field(default_factory=PdfConfig)

    debug: bool = False
    version: str = "dev"


def load_config(env_file=None):
    env_file = env_file or os.getenv("ENV_FILE")
    return Config(_env_file=env_file)
