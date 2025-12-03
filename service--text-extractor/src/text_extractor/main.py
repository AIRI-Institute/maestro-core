from mmar_ptag import deploy_server

from text_extractor.config import Config, load_config
from text_extractor.config_server import ConfigServer, load_config_server
from text_extractor.text_extractor import TextExtractor


def main():
    deploy_server(
        service=TextExtractor,
        config=load_config,
        config_server=load_config_server,
    )

if __name__ == "__main__":
    main()
