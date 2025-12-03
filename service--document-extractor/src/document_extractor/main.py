import os

from mmar_ptag import grpc_server, init_logger, ptag_attach
from loguru import logger

from document_extractor.config import Config, load_config
from document_extractor.document_extractor import DocumentExtractor


def main():
    config: Config = load_config()
    init_logger(config.logger.level)
    logger.debug(f"Config: {config}")

    # parallel_map inside is not need fork, so enabling it to eliminate warning
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "False"

    server = grpc_server(max_workers=config.server.max_workers, port=config.server.port)
    ptag_attach(server, DocumentExtractor(config))
    server.start()
    logger.info(f"Server started, listening on {config.server.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
