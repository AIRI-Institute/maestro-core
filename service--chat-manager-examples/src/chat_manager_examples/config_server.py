from mmar_mimpl import SettingsModel
from pydantic import BaseModel, Field


class LoggerConfig(BaseModel):
    level: str = "DEBUG"
    name: str = "chat-manager-examples"


class ConfigServer(SettingsModel):
    max_workers: int = 10
    port: int = 17231
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
