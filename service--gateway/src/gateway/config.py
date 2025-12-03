import os

from mmar_ptag import LogLevelEnum
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
    port: int = 7732
    log_level: str = "error"


class AddressesConfig(BaseModel):
    chat_manager: str = Field(default="chat-manager:17211")


class LoggingConfig(BaseModel):
    name: str = "gateway"
    level: LogLevelEnum = LogLevelEnum.DEBUG


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")
    version: str = "dev"
    files_dir: str = "/mnt/data/maestro/files"

    logger: LoggingConfig = LoggingConfig()
    fastapi: FastApiConfig = FastApiConfig()
    addresses: AddressesConfig = AddressesConfig()

    end_message: str = "Прочитайте список возможных диагнозов и выберите специалиста для записи на приём."

    db_path: str = "/mnt/data/maestro/dev/logs"
    archive_path: str = "/mnt/data/maestro/dev/archived_logs"

    default_entrypoint: dict = {"entrypoint_key": "giga-max-2", "caption": "GigaChat MAX 2"}


def load_config(env_file: str | None = None) -> Config:
    env_file = env_file or os.getenv("ENV_FILE")
    return Config(_env_file=env_file)
