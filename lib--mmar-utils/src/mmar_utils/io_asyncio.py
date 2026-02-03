import asyncio
from typing import Any, List

async def gather_with_limit(
    *coros_or_futures,
    return_exceptions: bool = False,
    max_workers: int = 1
) -> List[Any]:
    """
    Like asyncio.gather but with concurrency limit.
    
    Args:
        *coros_or_futures: Coroutines or futures to execute
        return_exceptions: If True, exceptions are returned as results
        max_workers: Maximum number of concurrent tasks
    
    Returns:
        List of results in the same order as inputs
    """
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    
    semaphore = asyncio.Semaphore(max_workers)
    
    async def limited_task(coro_or_future):
        async with semaphore:
            return await coro_or_future
    
    tasks = map(limited_task, coros_or_futures)
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
