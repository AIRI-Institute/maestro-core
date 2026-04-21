import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI, Request
from loguru import logger

from gateway.config import Config, load_config
from gateway.fastapi_errors import loguru_exception_handler
from gateway.fastapi_logging_middleware import LoggingMiddleware
from gateway.ioc import IOC
from gateway.legacy_routes import router_legacy
from gateway.routes import router


async def trace_id_getter(request: Request) -> str | None:
    path_params = request.path_params
    headers = request.headers
    try:
        if chat_id := path_params.get("chat_id"):
            return str(chat_id)
        if headers.get("content-type") == "application/json":
            pass
        if client_id := headers.get("client-id"):
            return str(client_id)
    except Exception as ex:
        logger.exception(f"Failed to extract trace_id: {ex}, path_params: {path_params}, headers: {headers}")
    return None


async def create_app(container: Any, lifespan: Any = None) -> FastAPI:
    config = await container.get(Config)
    app = FastAPI(version=config.version, lifespan=lifespan)

    app.add_exception_handler(Exception, loguru_exception_handler)
    app.middleware("http")(LoggingMiddleware(trace_id_getter))

    app.include_router(router)
    app.include_router(router_legacy)

    setup_dishka(container, app)
    return app


async def main() -> None:
    config = load_config()
    container = make_async_container(IOC(), context={Config: config})

    logger.info(f"Config: {config!r}")

    @asynccontextmanager
    async def lifespan(*_: Any) -> AsyncGenerator[None, None]:
        yield
        await container.close()

    app: FastAPI = await create_app(container, lifespan=lifespan)

    server_config = uvicorn.Config(
        app,
        host=config.fastapi.hostname,
        port=config.fastapi.port,
        workers=config.fastapi.max_workers,
        log_level="critical",
    )
    server = uvicorn.Server(server_config)
    host_port = f"{server_config.host}:{server_config.port}"
    logger.info(f"Starting uvicorn server: {host_port}")
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
