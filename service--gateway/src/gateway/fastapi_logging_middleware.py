from collections.abc import Callable
from typing import Awaitable

from fastapi import Request, Response
from loguru import logger


async def dummy_trace_id_getter(request: Request) -> str | None:
    return None


class LoggingMiddleware:
    def __init__(self, trace_id_getter: Callable[[Request], Awaitable[str | None]] | None = None):
        self.trace_id_getter = trace_id_getter or dummy_trace_id_getter

    async def __call__(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        trace_id = (await self.trace_id_getter(request)) or ""

        with logger.contextualize(trace_id=trace_id):
            route_info = f"{request.method.upper()} {request.url.path}"
            # also can add here some metadata
            logger.info(f"Request {route_info}")
            response: Response = await call_next(request)
            logger.info(f"Response {route_info}, status_code={response.status_code}")
            return response
