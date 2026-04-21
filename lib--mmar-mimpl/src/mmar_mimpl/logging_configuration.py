import sys
import warnings
from typing import Literal

from loguru import logger

from mmar_mimpl.trace_id import TRACE_ID, TRACE_ID_DEFAULT

LOG_LEVELS = {
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
}
LogLevel = Literal[*LOG_LEVELS]


def init_logger(log_level: LogLevel = "DEBUG"):
    if log_level not in LOG_LEVELS:
        warnings.warn(f"Bad log level: {log_level}, fallback to DEBUG")
        log_level = "DEBUG"

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
