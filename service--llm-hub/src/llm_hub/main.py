from mmar_ptag import deploy_server

from llm_hub.config import Config
from llm_hub.config_server import ConfigServer
from mmar_llm import LLMHub


def main():
    deploy_server(
        service=LLMHub,
        config=Config.load,
        config_server=ConfigServer.load,
    )


if __name__ == "__main__":
    main()
