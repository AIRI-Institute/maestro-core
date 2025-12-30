from concurrent import futures
from types import SimpleNamespace
from typing import Any, Callable, Protocol, Type

import grpc
from loguru import logger

from mmar_ptag.logging_configuration import LogLevelEnum, init_logger
from mmar_ptag.ptag_framework import ptag_attach


class ConfigLogger(Protocol):
    level: LogLevelEnum


CONFIG_LOGGER_DEFAULT = SimpleNamespace(level=LogLevelEnum.DEBUG)


class ConfigServer(Protocol):
    port: int
    max_workers: int
    logger: ConfigLogger | None


def grpc_server(*, port: int, max_workers: int) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    server.add_insecure_port(f"[::]:{port}")
    return server


def deploy_server(
    config_server: ConfigServer | Callable[[], ConfigServer],
    service: Any | Callable[..., Any] | Type,
    config: Any | Callable[[], Any] | None = None,
    initialize_logger: bool = True
) -> None:
    # normalize config_server and config if they are callables
    if callable(config_server):
        config_server = config_server()
    if callable(config):
        config = config()

    # instantiate service if it's a class / factory
    if isinstance(service, type) or callable(service):
        try:
            service = service(config)
        except TypeError:
            service = service()

    # logging setup and server start
    level = getattr(config_server, "logger", CONFIG_LOGGER_DEFAULT).level
    if initialize_logger:
        init_logger(level)
    logger.debug(f"Config server: {repr(config_server)}")
    logger.debug(f"Config: {repr(config)}")

    server = grpc_server(port=config_server.port, max_workers=config_server.max_workers)
    ptag_attach(server, service)
    server.start()
    logger.info(f"Server started, listening on {config_server.port}")
    server.wait_for_termination()
