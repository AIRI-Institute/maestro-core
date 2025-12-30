from pydantic import BaseModel

from mmar_mimpl import SettingsModel
from mmar_ptag import LogLevelEnum


class LoggerConfig(BaseModel):
    level: LogLevelEnum = LogLevelEnum.DEBUG
    name: str = "llm-hub-monitoring"


class ConfigServer(SettingsModel):
    max_workers: int = 10
    port: int = 40641
    logger: LoggerConfig = LoggerConfig()
