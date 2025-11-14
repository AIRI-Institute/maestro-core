import inspect
from collections.abc import Callable
from functools import wraps


def on_error_log_and_none(log_exception: Callable, error_message: str | None = None):
    def decorator(func):
        err_msg = f"Error in {func.__name__}"
        if error_message:
            err_msg = f"{err_msg}: {error_message}"

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception:
                log_exception(err_msg)
                return None

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                log_exception(err_msg)
                return None

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator
