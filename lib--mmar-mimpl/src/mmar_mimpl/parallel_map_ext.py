from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, TypeVar

from mmar_ptag.ptag_framework import TRACE_ID_VAR
from tqdm import tqdm

X = TypeVar("X")


def parallel_map_ext(
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
    """
    extended version of `parallel_map` which respects value of mmar_ptag.ptag_framework.TRACE_ID_VAR
    the goal: simplify parallel calls in services which uses PTAG
    """
    # Capture current trace_id context to propagate to worker threads
    current_trace_id = TRACE_ID_VAR.get()

    def wrapper(*args, **kwargs):
        # Restore the trace_id context in the worker thread
        if current_trace_id is not None:
            TRACE_ID_VAR.set(current_trace_id)
        return func(*args, **kwargs)

    pool = (ProcessPoolExecutor if process else ThreadPoolExecutor)(max_workers=max_workers)
    with pool as executor:
        futures = []
        for item in items:
            if kwargs_args:
                future = executor.submit(wrapper, **item)
            elif multiple_args:
                future = executor.submit(wrapper, *item)
            else:
                future = executor.submit(wrapper, item)
            futures.append(future)
        futures_w = tqdm(futures, desc=desc) if show_tqdm else futures
        results: list[X] = [future.result() for future in futures_w]
    return results
