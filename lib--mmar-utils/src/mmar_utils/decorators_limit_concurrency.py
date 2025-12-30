from functools import wraps
from threading import Semaphore


def limit_concurrency(concurrency_limit: int):
    semaphore = Semaphore(concurrency_limit)

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with semaphore:
                return fn(*args, **kwargs)

        return wrapper

    return decorator
