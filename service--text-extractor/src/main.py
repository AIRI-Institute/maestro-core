from mmar_ptag import deploy_server

from src.config import Config, load_config
from src.config_server import ConfigServer, load_config_server
from src.text_extractor import TextExtractor


def main():
    deploy_server(
        service=TextExtractor,
        config=load_config,
        config_server=load_config_server,
    )

if __name__ == "__main__":
    main()
