from concurrent import futures
from typing import Any, Callable, Protocol, Type

import grpc
from loguru import logger

from .logging_configuration import init_logger
from .ptag_framework import ptag_attach


class ConfigServer(Protocol):
    max_workers: int
    port: int
    logger: Any


def grpc_server(max_workers: int, port: int) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    server.add_insecure_port(f"[::]:{port}")
    return server


def deploy_server(
    config_server: ConfigServer | Callable[[], ConfigServer],
    service: Any | Callable[..., Any] | Type,
    config: Any | Callable[[], Any] | None = None,
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
    init_logger(config_server.logger.level)
    logger.debug(f"Config: {config}")

    server = grpc_server(max_workers=config_server.max_workers, port=config_server.port)
    ptag_attach(server, service)
    server.start()
    logger.info(f"Server started, listening on {config_server.port}")
    server.wait_for_termination()
