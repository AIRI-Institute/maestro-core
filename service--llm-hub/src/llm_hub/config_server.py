from mmar_mimpl import SettingsModel
from mmar_ptag import LogLevelEnum
from pydantic import BaseModel, Field


class LoggerConfig(BaseModel):
    level: LogLevelEnum = LogLevelEnum.DEBUG
    name: str = "llm-hub"


class ConfigServer(SettingsModel):
    max_workers: int = 10
    port: int = 40631
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
