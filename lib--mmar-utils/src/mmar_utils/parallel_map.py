from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, TypeVar

from tqdm import tqdm


X = TypeVar("X")
C = TypeVar("C")
ContextGetSet = tuple[Callable[[], C], Callable[[C], None]]

def contextualize_func(func: Callable[..., X], context_get_set: ContextGetSet | None):
    if context_get_set is None:
        return func

    context_get, context_set = context_get_set
    context = context_get()
    if context is None:
        return func

    def wrapper(*args, **kwargs):
        context_set(context)
        return func(*args, **kwargs)
    return wrapper
    

def parallel_map(
    func: Callable[..., X],
    items: Iterable[Any],
    *,
    process: bool = False,
    multiple_args: bool = False,
    kwargs_args: bool = False,
    max_workers: int = 2,
    show_tqdm: bool = False,
    desc: str = "",
    context_get_set: tuple[Callable[[], Any], Callable[[Any], None]] | None = None
) -> list[X]:
    func_fix = contextualize_func(func, context_get_set)

    pool = (ProcessPoolExecutor if process else ThreadPoolExecutor)(max_workers=max_workers)
    with pool as executor:
        futures = []
        for item in items:
            if kwargs_args:
                future = executor.submit(func_fix, **item)
            elif multiple_args:
                future = executor.submit(func_fix, *item)
            else:
                future = executor.submit(func_fix, item)
            futures.append(future)
        futures_w = tqdm(futures, desc=desc) if show_tqdm else futures
        results: list[X] = [future.result() for future in futures_w]
    return results
