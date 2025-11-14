from collections.abc import Callable
from functools import lru_cache

from loguru import logger


def maybe_lru_cache(maxsize: int, func: Callable) -> tuple[str, Callable]:
    if maxsize >= 0:
        maxsize = maxsize or None
        logger.info(f"Caching for {func.__name__}: enabled: maxsize={maxsize}")
        func = lru_cache(maxsize=maxsize)(func)
    else:
        logger.info(f"Caching for {func.__name__}: disabled")
    return func
