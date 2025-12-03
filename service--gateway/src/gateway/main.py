import uvicorn
from fastapi import FastAPI, Request
from loguru import logger

from gateway.dependencies import Deps
from gateway.legacy_routes import router_legacy
from gateway.fastapi_errors import loguru_exception_handler
from gateway.fastapi_logging_middleware import LoggingMiddleware
from gateway.routes import router


async def trace_id_getter(request: Request) -> str | None:
    path_params = request.path_params
    headers = request.headers
    try:
        if chat_id := path_params.get("chat_id"):
            return str(chat_id)
        if headers.get("content-type") == "application/json":
            pass
            # body = json.loads((await request.body()).decode())
            # todo: try to find chat_id inside request, if ok, use it as trace_id
        if client_id := headers.get("client-id"):
            return str(client_id)
    except Exception as ex:
        logger.exception(f"Failed to extract trace_id: {ex}, path_params: {path_params}, headers: {headers}")
    return None


def create_app(deps: Deps) -> FastAPI:
    config = deps.config
    app = FastAPI(version=config.version)

    app.add_exception_handler(Exception, loguru_exception_handler)
    app.middleware("http")(LoggingMiddleware(trace_id_getter))

    app.dependency_overrides.update({Deps: lambda: deps})
    app.include_router(router)
    app.include_router(router_legacy)
    return app


def main() -> None:
    deps = Deps()
    config = deps.config
    app: FastAPI = create_app(deps)
    server_config = uvicorn.Config(
        app,
        host=config.fastapi.hostname,
        port=config.fastapi.port,
        workers=config.fastapi.max_workers,
        # critical to eliminate non-colored logs: there is already logging middleware for logging
        log_level="critical",
    )
    server = uvicorn.Server(server_config)
    host_port = f"{server_config.host}:{server_config.port}"
    logger.info(f"Starting uvicorn server: {host_port}")
    server.run()


if __name__ == "__main__":
    main()
