from collections.abc import Callable
from functools import lru_cache

from loguru import logger


def maybe_lru_cache(maxsize: int, func: Callable) -> Callable:
    if maxsize >= 0:
        effective_maxsize: int | None = maxsize or None
        logger.info(f"Caching for {func.__name__}: enabled: maxsize={effective_maxsize}")
        func = lru_cache(maxsize=effective_maxsize)(func)
    else:
        logger.info(f"Caching for {func.__name__}: disabled")
    return func
