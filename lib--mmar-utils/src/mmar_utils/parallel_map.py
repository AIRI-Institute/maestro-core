from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, TypeVar

from tqdm import tqdm


X = TypeVar("X")


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
) -> list[X]:
    pool = (ProcessPoolExecutor if process else ThreadPoolExecutor)(max_workers=max_workers)
    with pool as executor:
        futures = []
        for item in items:
            if kwargs_args:
                future = executor.submit(func, **item)
            elif multiple_args:
                future = executor.submit(func, *item)
            else:
                future = executor.submit(func, item)
            futures.append(future)
        futures_w = tqdm(futures, desc=desc) if show_tqdm else futures
        results: list[X] = [future.result() for future in futures_w]
    return results
