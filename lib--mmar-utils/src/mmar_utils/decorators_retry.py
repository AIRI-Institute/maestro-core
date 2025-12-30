import asyncio
import inspect
import time
from functools import wraps
from typing import Any, Callable, Protocol, Tuple, Type, TypeVar, Union

Ex = Union[Type[BaseException], Tuple[Type[BaseException], ...]]
T = TypeVar("T")
Func = Callable[..., T]
NOTHING = object()


class LoggerI(Protocol):
    def debug(self, msg): ...
    def warning(self, msg): ...


def retry_on_ex(
    attempts: int | None = None,
    wait_seconds: int | list = 5,
    catch: Ex = Exception,
    nocatch: Ex = (),
    logger: LoggerI | None = None,
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
            logger.warning(f"Failed attempt={attempt} to call {func.__name__} ( {type(ex).__name__}: {ex} )")

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
    wait_seconds: int | float | None = 1,
    attempts: int = 3,
    condition: Callable[[T], bool] = bool,
    logger: LoggerI | None = None,
) -> Callable[[Func], Func]:
    def decorator(func: Func) -> Func:
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            total_attempts = max(1, attempts)
            for attempt in range(1, total_attempts + 1):
                if title is not None and attempt != 1 and logger:
                    logger.debug(f"{title}, attempt: {attempt}")
                result = func(*args, **kwargs)
                if condition(result):
                    return result
                if wait_seconds:
                    time.sleep(wait_seconds)
            return None

        return wrapper

    return decorator


def retry_on_cond_and_ex(
    title: str | None = None,
    wait_seconds: int | float | list[int | float] | None = 1,
    catch: Ex = Exception,
    nocatch: Ex = (),
    attempts: int = 3,
    condition: Callable[[T], bool] = bool,
    logger: LoggerI | None = None,
) -> Callable[[Func], Func]:
    if isinstance(wait_seconds, list):
        attempts = len(wait_seconds)
    else:
        if not isinstance(wait_seconds, (int, float)):
            wait_seconds = 1
        attempts = attempts or 5
        wait_seconds = [wait_seconds] * attempts

    def should_retry(ex: Exception) -> bool:
        return isinstance(ex, catch) and not isinstance(ex, nocatch)

    def log_fail(attempt: int, func: Func, ex: Exception) -> None:
        if logger:
            logger.warning(f"Failed attempt={attempt} to call {func.__name__} ( {type(ex).__name__}: {ex} )")

    def decorator(func: Func) -> Func:
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            total_attempts = max(1, attempts)
            for attempt, wait_s in zip(range(1, total_attempts + 1), wait_seconds):
                if title is not None and attempt != 1 and logger:
                    logger.debug(f"{title}, attempt: {attempt}")
                try:
                    result = func(*args, **kwargs)
                except Exception as ex:
                    if not should_retry(ex) or attempt == attempts:
                        if logger:
                            logger.warning(f"{title}, all {attempts} attempts failed")
                        raise
                    log_fail(attempt, func, ex)
                    result = NOTHING
                if result is not NOTHING and condition(result):
                    return result
                if logger:
                    logger.debug(f"{title}, failed attempt {attempt}, going to sleep {wait_s} seconds and retry")
                if wait_s:
                    time.sleep(wait_s)
            return None

        return wrapper

    return decorator
