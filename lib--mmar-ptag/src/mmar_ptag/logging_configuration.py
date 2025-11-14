import sys
from enum import StrEnum, auto

from loguru import logger

TRACE_ID = "trace_id"
TRACE_ID_DEFAULT = "UNSET"


class LogLevelEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return name.upper()

    TRACE = auto()
    DEBUG = auto()
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


def init_logger(log_level: str | LogLevelEnum):
    if isinstance(log_level, str):
        log_level = LogLevelEnum.__members__.get(log_level.upper(), LogLevelEnum.DEBUG)
    logger.remove()
    extra = {TRACE_ID: TRACE_ID_DEFAULT}
    format_parts = [
        "{time:DD-MM-YYYY HH:mm:ss}",
        "<level>{level: <8}</level>",
        "{extra[%s]}" % TRACE_ID,
        "<level>{message}</level>",
    ]
    format_ = " | ".join(format_parts)

    logger.add(sys.stdout, colorize=True, format=format_, level=log_level)
    logger.configure(extra=extra)
