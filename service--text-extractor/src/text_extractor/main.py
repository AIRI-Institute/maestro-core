from dishka import make_container
from mmar_ptag import deploy_server

from text_extractor.config import Config
from text_extractor.config_server import load_config_server
from text_extractor.ioc import IOCS
from text_extractor.text_extractor import TextExtractor


def main():
    container = make_container(*(ioc() for ioc in IOCS))
    config = container.get(Config)
    service = container.get(TextExtractor)

    deploy_server(
        service=service,
        config=config,
        config_server=load_config_server,
    )


if __name__ == "__main__":
    main()
