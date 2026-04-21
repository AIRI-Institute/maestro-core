import os
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FilesConfig(BaseModel):
    allowed_content_types: set[str] = {
        "text/csv",
        "text/plain",
        "application/pdf",
        "application/zip",
        "image/jpeg",
        "image/png",
        "application/octet-stream",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    allowed_mime_types: set[str] = {
        "text/plain",
        "application/pdf",
        "application/zip",
        "image/jpeg",
        "image/png",
        "application/octet-stream",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }  # only if mime-type != content-type
    max_file_size: int = 200 * 1024**2  # 200 MB
    read_chunk_size: int = 10 * 1024  # 10 KB


class FastApiConfig(BaseModel):
    files: FilesConfig = FilesConfig()

    max_workers: int = 5
    hostname: str = "0.0.0.0"
    port: int = 7731
    log_level: str = "error"


class AddressesConfig(BaseModel):
    chat_manager: str = Field(default="chat-manager:17211")


class LoggingConfig(BaseModel):
    name: str = "gateway"
    level: str = "DEBUG"


class ChatStorageConfig(BaseModel):
    uri: str
    extra: dict[str, Any] | None = None


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    version: str = "dev"
    files_dir: str = "/mnt/data/maestro/files"

    logger: LoggingConfig = LoggingConfig()
    fastapi: FastApiConfig = FastApiConfig()
    addresses: AddressesConfig = AddressesConfig()
    chat_storage: ChatStorageConfig

    openai_api_base: str = ""
    openai_api_key: str = ""
    # hotfix
    hide_models: list[str] = []

    end_message: str = "Прочитайте список возможных диагнозов и выберите специалиста для записи на приём."

    default_model: str = "giga-max-2"


def load_config(env_file: str | None = None) -> Config:
    env_file = env_file or os.getenv("ENV_FILE")
    return Config(_env_file=env_file)
