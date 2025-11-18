import asyncio
import inspect
import time
from functools import wraps
from typing import Any, Callable, Protocol, Tuple, Type, TypeVar, Union

from loguru import logger

Ex = Union[Type[BaseException], Tuple[Type[BaseException], ...]]
T = TypeVar("T")
Func = Callable[..., T]


class LoggerI(Protocol):
    def debug(self, msg): ...
    def warning(self, msg): ...


def retry_on_ex(
    attempts: int | None = None,
    wait_seconds: int | list = 5,
    catch: Ex = Exception,
    nocatch: Ex = (),
    logger: Callable | None = None,
) -> Callable[[Func], Func]:
    """
    Decorator that retries a function (sync or async) if it raises exceptions.
    """
    if isinstance(wait_seconds, list):
        attempts = len(wait_seconds)
    else:
        attempts = attempts or 5
        wait_seconds = [wait_seconds] * attempts

    def should_retry(ex: Exception) -> bool:
        return isinstance(ex, catch) and not isinstance(ex, nocatch)

    def log_fail(attempt: int, func: Func, ex: Exception) -> None:
        if logger:
            logger(f"Failed attempt={attempt} to call {func.__name__} ( {type(ex).__name__}: {ex} )")

    def decorator(func: Func) -> Func:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                for attempt, wait_s in zip(range(1, attempts + 1), wait_seconds):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as ex:
                        if not should_retry(ex) or attempt == attempts:
                            raise
                        log_fail(attempt, func, ex)
                        await asyncio.sleep(wait_s)

            return async_wrapper

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                for attempt, wait_s in zip(range(1, attempts + 1), wait_seconds):
                    try:
                        return func(*args, **kwargs)
                    except Exception as ex:
                        if not should_retry(ex) or attempt == attempts:
                            raise
                        log_fail(attempt, func, ex)
                        time.sleep(wait_s)

            return sync_wrapper  # type: ignore

    return decorator


def retry_on_cond(
    title: str | None = None,
    attempts: int = 3,
    condition: Callable[[T], bool] = bool,
) -> Callable[[Func], Func]:
    def decorator(func: Func) -> Func:
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            total_attempts = max(1, attempts)
            for attempt in range(total_attempts):
                if title is not None and attempt != 0:
                    logger.debug(f"{title}, attempt: {attempt + 1}")
                result = func(*args, **kwargs)
                if condition(result):
                    return result
            return None

        return wrapper
    return decorator
