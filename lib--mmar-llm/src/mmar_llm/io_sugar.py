import warnings
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from typing import Any, Literal

OnError = Literal["ignore", "warn", "fail"]
ErrorHandler = Callable[[Exception], Any]


def _on_exception(error_handler: ErrorHandler):
    try:
        yield
    except Exception as e:
        error_handler(e)


def make_on_exception(on_error: OnError, logger, err_msg: str):
    error_handler = _make_error_handler(on_error, logger, err_msg)
    res = contextmanager(partial(_on_exception, error_handler=error_handler))
    return res


def _make_error_handler(on_error: OnError, logger, err_msg: str) -> ErrorHandler:
    if on_error == "ignore":
        return _error_ignorer
    elif on_error == "warn":
        return partial(_error_warner, logger=logger, err_msg=err_msg)
    elif on_error == "fail":
        return _error_raiser
    else:
        raise ValueError(f"Expected OnError value, found: {on_error}")


def _error_ignorer(ex):
    pass


def _error_warner(ex, logger, err_msg):
    if logger:
        logger.warning(err_msg)
    else:
        warnings.warn(err_msg)


def _error_raiser(ex):
    raise ex
