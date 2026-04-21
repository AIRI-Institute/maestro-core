from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

from loguru import logger

T = TypeVar("T")
TRACE_ID = "trace_id"
TRACE_ID_DEFAULT = "UNSET"
TRACE_ID_VAR: ContextVar[str] = ContextVar(TRACE_ID, default=TRACE_ID_DEFAULT)


@contextmanager
def installed_trace_id(trace_id: str | None, contextualize_logger: bool = True):
    if trace_id is None:
        yield
        return

    token = TRACE_ID_VAR.set(trace_id)
    try:
        if contextualize_logger:
            with logger.contextualize(trace_id=trace_id):
                yield
        else:
            yield
    finally:
        TRACE_ID_VAR.reset(token)
